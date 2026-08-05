//! Read per-cell annotations straight out of an h5ad's `/obs`.
//!
//! The scanpy-resident tools write their results back into an h5ad in place —
//! `sc.pp.scrublet(adata)` sets `adata.obs["doublet_score"]` and
//! `adata.obs["predicted_doublet"]` and leaves the file otherwise untouched.
//! Round-tripping that through a CSV just to import it is a step with no
//! purpose, so this reader takes the h5ad directly.
//!
//! # Simpler than the table reader, in the two places that one had to work
//!
//! [`crate::annotation_table`] fights schema inference (a single `NA` turning a
//! score column into text) and an unnamed leading index column. Neither exists
//! here: an h5ad's columns carry their own types, and `read_dataframe_group`
//! already resolves the `_index` attribute and renames the sentinel to the
//! canonical `__index_level_0__` — which is in the ops crate's key-fallback
//! list, so auto key resolution works without any special casing.
//!
//! # What it does add
//!
//! A pandas Categorical arrives as `Dictionary(Int32, Utf8)`, and a class or
//! prediction column is exactly what pandas stores that way. The doublet
//! derivation decodes it (`crate::doublet::undictionary`); without that, values
//! that import fine from a CSV would fail from the h5ad the same tool wrote.
//!
//! # Gated
//!
//! Behind `hdf5` like every other h5ad path. The dispatch in
//! [`crate::annotation_table::read_obs_source`] keeps a clear "write it to a
//! CSV first" error in a build without it — the *source format* is unavailable
//! there, never the command.

use std::path::Path;

use serde_json::{Map, Value};

use scx_ops::{
    build_composite_key, obs_key_values, resolve_obs_key_column, ExternalObsData, OpsError, Result,
};

use crate::annotation_table::{
    project_annotations, AnnotationTableInfo, AnnotationTableOptions, ObsSourceFormat,
};
use crate::file_checksum::blake3_of_file;
use crate::h5ad::read::{read_dataframe_group, read_uns};
use crate::warnings::WarningSink;

/// Read `/obs` (and optionally selected `/uns` keys) from an h5ad.
///
/// Returns the same pair the delimited reader does, so both feed the doublet
/// wrapper and the generic importer unchanged.
pub fn read_h5ad_obs(
    path: &Path,
    opts: &AnnotationTableOptions,
    uns_keys: &[String],
) -> Result<(ExternalObsData, AnnotationTableInfo)> {
    let file = hdf5::File::open(path).map_err(|e| {
        OpsError::InvalidInput(format!("could not open '{}' as HDF5: {e}", path.display()))
    })?;

    let mut sink = WarningSink::log();
    let obs = read_dataframe_group(&file, "obs", &mut sink).map_err(|e| {
        OpsError::InvalidInput(format!(
            "could not read /obs from '{}': {e}",
            path.display()
        ))
    })?;

    if obs.num_rows() == 0 {
        return Err(OpsError::InvalidInput(format!(
            "'{}' has an empty /obs; there is nothing to import",
            path.display()
        )));
    }

    // Key resolution goes through the ops crate, exactly as the table reader
    // does, so the source and target sides of the join can never disagree about
    // what a key is. No all-Utf8 view is needed here: the table reader pins one
    // because a numeric-looking barcode *infers* as an integer, whereas an
    // h5ad column carries its own declared type.
    // Bind the schema: `RecordBatch::schema()` hands back a temporary Arc, and
    // borrowing field names straight out of it does not outlive the statement.
    let schema = obs.schema();
    let key_columns: Vec<String> = if opts.key_columns.is_empty() {
        vec![resolve_obs_key_column(&obs, None)?]
    } else {
        // See `annotation_table.rs`: the axis-index alias resolves on the
        // source side too, so `key="obs_names"` works against an h5ad's own
        // obs index without a `source_key=`.
        let mut resolved = Vec::with_capacity(opts.key_columns.len());
        for k in &opts.key_columns {
            let physical = scx_ops::resolve_key_alias("obs", &schema, k);
            if schema.field_with_name(&physical).is_err() {
                let present: Vec<String> = schema
                    .fields()
                    .iter()
                    .map(|f| scx_ops::display_key_name("obs", f.name()))
                    .collect();
                return Err(OpsError::KeyColumnUnresolved {
                    axis: "obs",
                    detail: format!(
                        "key column '{k}' not found in /obs of '{}'; columns present are \
                         {present:?}",
                        path.display()
                    ),
                });
            }
            resolved.push(physical);
        }
        resolved
    };

    let row_keys = if key_columns.len() == 1 {
        obs_key_values(&obs, &key_columns[0])?
    } else {
        build_composite_key(&obs, &key_columns)?
    };

    let row_annotations = project_annotations(&obs, &key_columns, opts)?;
    let columns_imported: Vec<String> = row_annotations
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let (uns, uns_keys_imported) = select_uns(&file, uns_keys, path, &mut sink)?;

    let info = AnnotationTableInfo {
        n_rows: obs.num_rows(),
        format: ObsSourceFormat::H5ad,
        // No such concept for HDF5, and a fabricated `,` would misreport how
        // the file was read.
        delimiter: None,
        key_columns: key_columns.clone(),
        columns_imported,
        // The index arrives properly named; there is nothing to rename.
        renamed_index_column: false,
        uns_keys_imported: uns_keys_imported.clone(),
    };

    Ok((
        ExternalObsData {
            row_keys,
            row_annotations,
            row_embeddings: Vec::new(),
            uns,
            source_checksum: blake3_of_file(path).ok(),
            source_name: path.file_name().map(|s| s.to_string_lossy().to_string()),
        },
        info,
    ))
}

/// Pull the requested `/uns` keys, and only those.
///
/// Opt-in rather than wholesale: `/uns` routinely holds large arrays and
/// compound types the reader cannot represent, so importing all of it would be
/// a surprise in both size and content. A requested key that is not there is an
/// error naming what is — the same rule the column selectors follow, and the
/// alternative is a silently empty record that looks like the tool wrote
/// nothing.
fn select_uns(
    file: &hdf5::File,
    uns_keys: &[String],
    path: &Path,
    sink: &mut WarningSink,
) -> Result<(Option<Value>, Vec<String>)> {
    if uns_keys.is_empty() {
        return Ok((None, Vec::new()));
    }

    let all = read_uns(file, false, sink).map_err(|e| {
        OpsError::InvalidInput(format!(
            "uns keys {uns_keys:?} were requested but /uns of '{}' could not be read: {e}",
            path.display()
        ))
    })?;
    let Value::Object(all) = all else {
        return Err(OpsError::InvalidInput(format!(
            "/uns of '{}' is not a group",
            path.display()
        )));
    };

    let mut picked = Map::new();
    let mut names = Vec::with_capacity(uns_keys.len());
    for k in uns_keys {
        match all.get(k) {
            Some(v) => {
                picked.insert(k.clone(), v.clone());
                names.push(k.clone());
            }
            None => {
                let present: Vec<&String> = all.keys().collect();
                return Err(OpsError::InvalidInput(format!(
                    "uns key '{k}' is not in '{}'; keys present are {present:?}. Note that \
                     entries the reader cannot represent (compound/structured arrays) are \
                     skipped and will not appear here.",
                    path.display()
                )));
            }
        }
    }

    Ok((Some(Value::Object(picked)), names))
}

#[cfg(test)]
#[path = "h5ad_obs_tests.rs"]
mod tests;
