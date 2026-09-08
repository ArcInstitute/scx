//! The external-var join family: `var_import` (delimited tables / an h5ad's
//! `/var`), `attach_var_columns` (in-memory DataFrames) and
//! `diagnose_var_key`.
//!
//! The var-axis mirror of [`super::obs_attach`], sharing its option parsing,
//! key pairing and `uns` payload normalisation — the rules there are about
//! policies and `uns`, not about an axis, and a second copy would be a second
//! place to drift.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use super::obs_attach::{
    key_diagnosis_dict, parse_obs_extra_rows, parse_obs_missing_rows, parse_uns_payload,
    resolve_join_key_for,
};
use super::*;

/// The summary fields both var entry points report.
fn var_summary_dict<'py>(
    py: Python<'py>,
    summary: &scx_ops::AttachVarSummary,
    dry_run: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("dry_run", dry_run)?;
    d.set_item("n_vars", summary.n_vars)?;
    d.set_item("n_matched", summary.n_matched)?;
    d.set_item("n_target_rows_absent", summary.n_target_rows_absent)?;
    d.set_item("n_source_rows_absent", summary.n_source_rows_absent)?;
    d.set_item(
        "var_key_column",
        scx_ops::display_key_name("var", &summary.var_key_column),
    )?;
    d.set_item("var_columns_added", summary.var_columns_added.clone())?;
    // Whether the var predicate index had to be rebuilt (an overwrite of a
    // column it covered) rather than carried verbatim (a pure add).
    d.set_item("var_index_rebuilt", summary.var_index_rebuilt)?;
    d.set_item(
        "var_columns_not_carried",
        summary.var_columns_not_carried.clone(),
    )?;
    // Whether var was rewritten shard by shard, which is decided by the layout
    // it already had. Not inferable from the output: a one-shard sharded var
    // and a single section both hold every gene.
    d.set_item("var_streamed", summary.var_streamed)?;
    Ok(d)
}

// ---------------------------------------------------------------------------
// Delimited / h5ad var import
// ---------------------------------------------------------------------------

/// Import a delimited annotation table (CSV / TSV) — or an h5ad's `/var` on an
/// `hdf5` build — as `var` columns, in place, joined to the target's own var
/// axis by key string.
///
/// Full user-facing documentation lives on the `pyscx.var_import` Python
/// wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (
    path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="",
    keep_key_columns=false, delimiter=None, status_column=None, uns_key=None,
    uns_keys=None, overwrite=false, on_missing_rows="null", on_extra_rows="warn",
    dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn var_import(
    py: Python<'_>,
    path: &str,
    table: &str,
    key: Option<Vec<String>>,
    source_key: Option<Vec<String>>,
    columns: Option<Vec<String>>,
    rename: Option<std::collections::HashMap<String, String>>,
    prefix: &str,
    keep_key_columns: bool,
    delimiter: Option<&str>,
    status_column: Option<&str>,
    uns_key: Option<&str>,
    uns_keys: Option<Vec<String>>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> PyResult<Py<PyDict>> {
    let uns_keys = uns_keys.unwrap_or_default();
    let delimiter_byte = match delimiter {
        None => None,
        Some(s) => {
            let bytes = s.as_bytes();
            if bytes.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "delimiter must be exactly one byte; got {s:?}"
                )));
            }
            Some(bytes[0])
        }
    };

    let (key_columns, join_key) = resolve_join_key_for("var", key, source_key)?;

    let read_opts = scx_convert::AnnotationTableOptions {
        key_columns,
        delimiter: delimiter_byte,
        columns,
        rename: rename.unwrap_or_default(),
        prefix: prefix.to_string(),
        keep_key_columns,
        infer_max_records: None,
    };
    let attach_opts = scx_ops::AttachVarOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        status_column: status_column.map(str::to_string),
        overwrite,
        provenance_action: "var_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let table_path = PathBuf::from(table);

    let (summary, info, diagnosis) = py.detach(|| -> PyResult<_> {
        let (mut data, info) = scx_convert::read_var_source(&table_path, &read_opts, &uns_keys)
            .map_err(ops_to_pyerr)?;
        // Without `uns_key` the selected source keys land at top level under
        // their own names; with it they nest under the one key.
        if let Some(k) = uns_key {
            data.nest_uns_under(k);
        }
        let summary =
            scx_ops::attach_external_var(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        // Only on a dry run: the diagnosis costs a pass per var column.
        let diagnosis = if dry_run {
            scx_ops::diagnose_var_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, info, diagnosis))
    })?;

    let d = var_summary_dict(py, &summary, dry_run)?;
    d.set_item("n_rows_in_source", info.n_rows)?;
    d.set_item("format", info.format.as_str())?;
    d.set_item("delimiter", info.delimiter.map(|b| (b as char).to_string()))?;
    d.set_item("uns_keys_imported", info.uns_keys_imported)?;
    d.set_item("renamed_index_column", info.renamed_index_column)?;
    d.set_item("source_key_columns", info.key_columns)?;
    d.set_item("columns_in_source", info.columns_imported)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// In-memory var-column attach (DataFrame)
// ---------------------------------------------------------------------------

/// Land an in-memory pandas `DataFrame` (or pyarrow `Table`) as `var` columns
/// on an existing file, **in place** — the DataFrame twin of [`var_import`].
///
/// Full user-facing documentation lives on the `pyscx.attach_var_columns`
/// Python wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (
    path, df, *, key=None, positional=false, status_column=None, uns=None,
    uns_key=None, overwrite=false, on_missing_rows="null", on_extra_rows="warn",
    dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn attach_var_columns(
    py: Python<'_>,
    path: &str,
    df: &Bound<'_, PyAny>,
    key: Option<Vec<String>>,
    positional: bool,
    status_column: Option<&str>,
    uns: Option<&Bound<'_, PyAny>>,
    uns_key: Option<&str>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> PyResult<Py<PyDict>> {
    if positional && key.is_some() {
        return Err(PyValueError::new_err(
            "key= and positional=True are mutually exclusive: positional means \
             row i of df annotates var row i, so there is no key to join on",
        ));
    }
    let uns_entries = parse_uns_payload(py, uns, uns_key)?;

    let batch = obs_var_to_record_batch(py, df, "attach_var_columns", "df")?;
    // Unlike obs there is no live/physical split on var, so a zero-row frame
    // can never be the right length for a file with genes — refuse it on both
    // modes rather than letting the ops layer report a shape mismatch.
    if batch.num_rows() == 0 {
        return Err(PyValueError::new_err(
            "df has no rows; there is nothing to attach",
        ));
    }

    // Both sides of a key join are built by the same ops-crate helpers the CSV
    // reader uses, so a DataFrame attach and an imported table cannot disagree
    // about what a key is. `drop` names the columns of `df` the key consumed —
    // they are already on the target's var axis, so re-importing them would
    // only duplicate them.
    let key = key.unwrap_or_default();
    let (row_keys, join_key, mut drop): (Vec<String>, scx_ops::AxisJoinKey, Vec<String>) =
        if positional {
            // Positional joins on nothing — but a frame carrying a labelled
            // pandas index (every `read_var()` frame does) hands its labels to
            // the ops layer as an ALIGNMENT CHECK, so a frame sorted or
            // reindexed after `read_var()` is refused by row instead of landing
            // every value on the wrong gene.
            let index_cols = scx_format_io::pandas_index_columns(&batch.schema());
            let index_col = index_cols.first().cloned().or_else(|| {
                batch
                    .schema()
                    .index_of("__index_level_0__")
                    .ok()
                    .map(|_| "__index_level_0__".to_string())
            });
            let keys = match index_col {
                Some(col) if index_cols.len() <= 1 => {
                    scx_ops::obs_key_values(&batch, &col).map_err(ops_to_pyerr)?
                }
                Some(_) => scx_ops::build_composite_key_for("var", &batch, &index_cols)
                    .map_err(ops_to_pyerr)?,
                None => Vec::new(),
            };
            (keys, scx_ops::AxisJoinKey::Positional, Vec::new())
        } else {
            match key.len() {
                0 => {
                    // Both sides resolve independently, exactly as `var_import`
                    // with no `key=`: the source uses its own index (named or
                    // not) / gene-id fallbacks, the target its own.
                    let col = scx_ops::resolve_axis_key_column("var", &batch, None)
                        .map_err(ops_to_pyerr)?;
                    let values = scx_ops::obs_key_values(&batch, &col).map_err(ops_to_pyerr)?;
                    (values, scx_ops::AxisJoinKey::Auto, vec![col])
                }
                1 => {
                    // Alias-resolved (`var_names` names the df index).
                    let col = scx_ops::resolve_axis_key_column("var", &batch, Some(&key[0]))
                        .map_err(ops_to_pyerr)?;
                    let values = scx_ops::obs_key_values(&batch, &col).map_err(ops_to_pyerr)?;
                    (
                        values,
                        scx_ops::AxisJoinKey::Column(key[0].clone()),
                        vec![col],
                    )
                }
                _ => {
                    let fused = scx_ops::build_composite_key_for("var", &batch, &key)
                        .map_err(ops_to_pyerr)?;
                    (
                        fused,
                        scx_ops::AxisJoinKey::Composite {
                            columns: key.clone(),
                        },
                        key.clone(),
                    )
                }
            }
        };
    // The pandas index is never data to attach: under a key join it is either
    // the key itself or a RangeIndex, and under positional the row order
    // already carries the identity.
    drop.extend(scx_format_io::pandas_index_columns(&batch.schema()));
    drop.push("__index_level_0__".to_string());
    let annotations = scx_ops::drop_batch_columns(&batch, &drop).map_err(ops_to_pyerr)?;

    let data = scx_ops::ExternalVarData {
        row_keys,
        row_annotations: annotations,
        uns: uns_entries,
        source_checksum: None,
        source_name: Some("<DataFrame>".to_string()),
    };
    let attach_opts = scx_ops::AttachVarOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        status_column: status_column.map(str::to_string),
        overwrite,
        provenance_action: "attach_var_columns".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let (summary, diagnosis) = py.detach(|| -> PyResult<_> {
        let summary =
            scx_ops::attach_external_var(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        let diagnosis = if dry_run && !positional {
            scx_ops::diagnose_var_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, diagnosis))
    })?;

    let d = var_summary_dict(py, &summary, dry_run)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Key reconnaissance
// ---------------------------------------------------------------------------

/// Report which var columns could serve as a join key for `var_import`.
/// Read-only.
///
/// Full user-facing documentation lives on the `pyscx.diagnose_var_key` Python
/// wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (path, key=None))]
pub fn diagnose_var_key(
    py: Python<'_>,
    path: &str,
    key: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    let (_, join_key) = resolve_join_key_for("var", key, None)?;
    let scx_path = PathBuf::from(path);
    let diag = py
        .detach(|| scx_ops::diagnose_var_key(&scx_path, Some(&join_key)))
        .map_err(ops_to_pyerr)?;
    Ok(key_diagnosis_dict(py, &diag)?.into())
}
