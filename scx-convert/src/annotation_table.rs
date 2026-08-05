//! Reader for delimited per-cell annotation tables (CSV / TSV).
//!
//! Produces a [`scx_ops::ExternalObsData`] ready for
//! `scx_ops::attach_external_obs`. The motivating case is doublet-caller
//! interop — every popular caller can be persuaded to emit
//! `barcode,score,call` in a line or two of user glue — but nothing here knows
//! about doublets.
//!
//! # Why a generic table reader rather than one reader per tool
//!
//! CellBender needed a native HDF5 reader because its file has four separately
//! nasty layout quirks. A doublet caller's output is a table. Five of the six
//! surveyed tools reach SCX through this one reader; only the scanpy-resident
//! ones justify a native path, and that is a later phase.
//!
//! # Ungated on purpose
//!
//! This module must build without the `hdf5` feature — a CSV reader that
//! requires libhdf5 would block `scx obs-import` in a no-hdf5 build. That is
//! also why it returns [`scx_ops::OpsError`]: `ConvertError` lives in the
//! `hdf5`-gated `pipeline` module and does not exist here. Returning the ops
//! error has a second benefit — the CLI and Python surfaces map **one** error
//! type across read-then-attach instead of two.
//!
//! # Two real-world shapes that need explicit handling
//!
//! * **`NA` poisons type inference.** arrow's default null rule treats only the
//!   empty string as null, and inference feeds every other token to the type
//!   guesser. R's `write.csv` writes `NA`, so a single missing value turns a
//!   score column into `Utf8` and the score silently lands as a string. Since
//!   the flagship path is *scDblFinder → `write.csv` → import*, and
//!   `scDblFinder.mostLikelyOrigin` is `NA` for random-origin doublets, this
//!   would fire on the very first real file. [`NULL_TOKENS`] fixes both
//!   inference and parsing.
//! * **`obs.to_csv()` writes an unnamed index column.** pandas emits
//!   `,doublet_score\nAAAC-1,0.03`; arrow names that first field `""`, which no
//!   key fallback matches, so auto-resolution fails on the most likely file a
//!   user produces. It is renamed to `_index` before resolution.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::csv::reader::Format;
use arrow::csv::ReaderBuilder;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

use scx_ops::{
    build_composite_key, obs_key_values, resolve_obs_key_column, ExternalObsData, OpsError, Result,
};

use crate::file_checksum::blake3_of_file;

/// Tokens treated as missing, for **both** schema inference and parsing.
///
/// Anchored so a barcode that merely contains `NA` (they routinely do — the
/// alphabet is ACGT but sample prefixes are arbitrary) is not silently nulled.
/// Covers R (`NA`), pandas (`NaN`, `nan`, `None`, empty), and SQL-ish exports
/// (`NULL`, `null`).
///
/// `Inf` / `-Inf` are deliberately **not** here. They are how R and pandas
/// write real IEEE infinities, which are values rather than absences — a
/// score column legitimately containing `Inf` should arrive as an infinity,
/// not be silently turned into a null the way a genuinely missing entry is.
/// pandas' own `read_csv` parses them as floats, and diverging from that here
/// would quietly drop outliers.
const NULL_TOKENS: &str = r"^(|NA|N/A|n/a|NaN|nan|None|NULL|null)$";

/// Name given to a leading unnamed column, chosen because it is already in the
/// ops crate's key-fallback list.
const PANDAS_INDEX_NAME: &str = "_index";

/// Rows sampled for type inference when the caller does not say.
///
/// Bounded rather than whole-file: inference is a second full pass, and a
/// per-cell table can be millions of rows. 8192 is far past the point where a
/// column's type stops being ambiguous.
const DEFAULT_INFER_RECORDS: usize = 8192;

#[derive(Debug, Clone, Default)]
pub struct AnnotationTableOptions {
    /// Key column(s). Empty resolves automatically with the same preference
    /// order the attach op uses. More than one is fused into a composite key.
    pub key_columns: Vec<String>,
    /// Field delimiter. `None` sniffs from the extension, then the header.
    pub delimiter: Option<u8>,
    /// Import only these columns. `None` imports every non-key column.
    pub columns: Option<Vec<String>>,
    /// Applied after selection, before `prefix`.
    pub rename: HashMap<String, String>,
    /// Prepended to every imported column name.
    pub prefix: String,
    /// Also keep the key column(s) as ordinary annotations.
    pub keep_key_columns: bool,
    /// Rows sampled for type inference. `None` uses [`DEFAULT_INFER_RECORDS`];
    /// `Some(0)` reads the whole file.
    pub infer_max_records: Option<usize>,
}

/// Which reader handled a source file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObsSourceFormat {
    /// Delimited text, read by [`read_annotation_table`].
    Table,
    /// An h5ad's `/obs`, read by `read_h5ad_obs` (requires the `hdf5` feature).
    H5ad,
}

impl ObsSourceFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ObsSourceFormat::Table => "table",
            ObsSourceFormat::H5ad => "h5ad",
        }
    }
}

#[derive(Debug, Clone)]
pub struct AnnotationTableInfo {
    pub n_rows: usize,
    pub format: ObsSourceFormat,
    /// Field delimiter, for a delimited source. `None` for h5ad, where the
    /// concept does not exist — reporting a fabricated `,` would be a small lie
    /// in the one line users read to confirm the file parsed as they meant.
    pub delimiter: Option<u8>,
    /// The resolved key column(s), in order.
    pub key_columns: Vec<String>,
    /// Final (post-rename, post-prefix) names of the appended columns.
    pub columns_imported: Vec<String>,
    /// Whether a leading unnamed column was renamed to `_index`. Always false
    /// for h5ad, where the index arrives properly named.
    pub renamed_index_column: bool,
    /// `/uns` keys pulled across, in request order. Always empty for a table.
    pub uns_keys_imported: Vec<String>,
}

// ---------------------------------------------------------------------------
// Delimiter
// ---------------------------------------------------------------------------

/// Pick a delimiter: explicit override, then extension, then the header line.
///
/// The header sniff exists for `.txt`, which says nothing about its own format
/// and which several tools use anyway. Counting fields rather than raw bytes so
/// a comma inside a quoted field does not outvote the real separator.
fn resolve_delimiter(path: &Path, explicit: Option<u8>) -> Result<u8> {
    if let Some(d) = explicit {
        return Ok(d);
    }
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("csv") => return Ok(b','),
        Some("tsv") | Some("tab") => return Ok(b'\t'),
        _ => {}
    }

    let mut header = String::new();
    BufReader::new(File::open(path)?).read_line(&mut header)?;
    let tabs = header.matches('\t').count();
    let commas = header.matches(',').count();
    Ok(if tabs > commas { b'\t' } else { b',' })
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

fn base_format(delimiter: u8) -> Result<Format> {
    let re = regex::Regex::new(NULL_TOKENS).map_err(|e| {
        OpsError::InvalidInput(format!("internal null-token regex is invalid: {e}"))
    })?;
    Ok(Format::default()
        .with_header(true)
        .with_delimiter(delimiter)
        .with_null_regex(re))
}

/// Rename a leading unnamed column and reject anything else unusable.
///
/// A blank name elsewhere, or a duplicate, makes column selection ambiguous —
/// `--columns score` cannot mean two different columns — so both are refused
/// rather than silently resolved to whichever came first.
fn normalize_field_names(schema: Schema) -> Result<(Schema, bool)> {
    let mut renamed = false;
    let mut fields: Vec<Field> = Vec::with_capacity(schema.fields().len());

    for (i, f) in schema.fields().iter().enumerate() {
        let name = f.name().as_str();
        if name.is_empty() {
            if i == 0 {
                renamed = true;
                fields.push(Field::new(
                    PANDAS_INDEX_NAME,
                    f.data_type().clone(),
                    f.is_nullable(),
                ));
                continue;
            }
            return Err(OpsError::InvalidInput(format!(
                "column {i} has a blank name; only a leading unnamed column (the \
                 pandas index) is understood, and it is renamed to '{PANDAS_INDEX_NAME}'"
            )));
        }
        fields.push(f.as_ref().clone());
    }

    let mut seen = HashSet::new();
    for f in &fields {
        if !seen.insert(f.name().clone()) {
            return Err(OpsError::InvalidInput(format!(
                "duplicate column name '{}' in the table; column selection would be \
                 ambiguous",
                f.name()
            )));
        }
    }
    Ok((Schema::new(fields), renamed))
}

/// Force key columns to `Utf8` and every field nullable.
///
/// The `Utf8` pin has to happen **before** the real read: inference would
/// narrow a barcode column of `0012`-style tokens to `Int64`, and casting that
/// back to text yields `12`, which matches nothing on the target side. Pinning
/// at parse time keeps the raw token.
fn pin_schema(schema: &Schema, key_columns: &[String]) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| {
            let dt = if key_columns.iter().any(|k| k == f.name()) {
                DataType::Utf8
            } else {
                f.data_type().clone()
            };
            Field::new(f.name(), dt, true)
        })
        .collect();
    Arc::new(Schema::new(fields))
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Read a delimited annotation table into an [`ExternalObsData`].
pub fn read_annotation_table(
    path: &Path,
    opts: &AnnotationTableOptions,
) -> Result<(ExternalObsData, AnnotationTableInfo)> {
    let delimiter = resolve_delimiter(path, opts.delimiter)?;
    let format = base_format(delimiter)?;

    // --- Infer, then fix up the header -------------------------------------
    let max_records = match opts.infer_max_records {
        Some(0) => None,
        Some(n) => Some(n),
        None => Some(DEFAULT_INFER_RECORDS),
    };
    let (inferred, n_sampled) = format
        .infer_schema(BufReader::new(File::open(path)?), max_records)
        .map_err(|e| {
            OpsError::InvalidInput(format!(
                "could not read '{}' as a delimited table: {e}",
                path.display()
            ))
        })?;
    if inferred.fields().is_empty() || n_sampled == 0 {
        return Err(OpsError::InvalidInput(format!(
            "'{}' has a header but no data rows",
            path.display()
        )));
    }
    let (schema, renamed_index_column) = normalize_field_names(inferred)?;

    // --- Resolve the key ----------------------------------------------------
    // Resolve against an all-`Utf8` view of the schema. `resolve_obs_key_column`
    // only considers string columns — correct for an obs table, where a numeric
    // column is never a barcode — but here the types are *inferred*, and a
    // barcode of `0012`/`0034` infers as `Int64`. Auto-resolution would then
    // refuse the one column that is obviously the key. Every column is read as
    // text for key purposes anyway, so the inferred type is not information we
    // want at this step.
    let as_text = Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| Field::new(f.name(), DataType::Utf8, true))
            .collect::<Vec<_>>(),
    );
    let header_batch = RecordBatch::new_empty(Arc::new(as_text));
    let key_columns: Vec<String> = if opts.key_columns.is_empty() {
        vec![resolve_obs_key_column(&header_batch, None)?]
    } else {
        // `obs_names` resolves here too, not just on the target side: an
        // unnamed CSV index column is renamed to `_index` above, and the
        // diagnosis tells users every name it reports is paste-able into
        // `key=`. Resolving per component keeps that promise for a source table
        // whose key IS its index. The error still quotes what the user typed.
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
                        "key column '{k}' not found in '{}'; columns present are {present:?}",
                        path.display()
                    ),
                });
            }
            resolved.push(physical);
        }
        resolved
    };

    // --- Real read ----------------------------------------------------------
    let pinned = pin_schema(&schema, &key_columns);
    let reader = ReaderBuilder::new(Arc::clone(&pinned))
        .with_format(base_format(delimiter)?)
        .build(BufReader::new(File::open(path)?))
        .map_err(|e| OpsError::InvalidInput(format!("could not open '{}': {e}", path.display())))?;

    let mut batches = Vec::new();
    for b in reader {
        batches.push(b.map_err(|e| {
            OpsError::InvalidInput(format!("failed parsing '{}': {e}", path.display()))
        })?);
    }
    let table = arrow::compute::concat_batches(&pinned, &batches)?;
    if table.num_rows() == 0 {
        return Err(OpsError::InvalidInput(format!(
            "'{}' has a header but no data rows",
            path.display()
        )));
    }

    // --- Build the join keys ------------------------------------------------
    // Single and composite both go through the ops crate, so the target side
    // of the join is constructed by exactly the same code.
    let row_keys = if key_columns.len() == 1 {
        obs_key_values(&table, &key_columns[0])?
    } else {
        build_composite_key(&table, &key_columns)?
    };

    // --- Project the annotations -------------------------------------------
    let row_annotations = project_annotations(&table, &key_columns, opts)?;
    let columns_imported: Vec<String> = row_annotations
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();

    let info = AnnotationTableInfo {
        n_rows: table.num_rows(),
        format: ObsSourceFormat::Table,
        delimiter: Some(delimiter),
        key_columns: key_columns.clone(),
        columns_imported,
        renamed_index_column,
        uns_keys_imported: Vec::new(),
    };

    Ok((
        ExternalObsData {
            row_keys,
            row_annotations,
            row_embeddings: Vec::new(),
            uns: None,
            source_checksum: blake3_of_file(path).ok(),
            source_name: path.file_name().map(|s| s.to_string_lossy().to_string()),
        },
        info,
    ))
}

/// Select, rename and prefix the columns that become obs annotations.
///
/// Shared with the h5ad reader on purpose: the doublet wrapper consumes this
/// batch whichever source produced it, so the two paths must agree exactly on
/// what "import these columns under this prefix" means.
pub(crate) fn project_annotations(
    table: &RecordBatch,
    key_columns: &[String],
    opts: &AnnotationTableOptions,
) -> Result<RecordBatch> {
    let schema = table.schema();

    // Which source columns to take, in table order so the output is stable.
    let selected: Vec<String> = match &opts.columns {
        Some(wanted) => {
            for c in wanted {
                if schema.field_with_name(c).is_err() {
                    let present: Vec<&str> =
                        schema.fields().iter().map(|f| f.name().as_str()).collect();
                    return Err(OpsError::InvalidInput(format!(
                        "requested column '{c}' is not in the table; columns present \
                         are {present:?}"
                    )));
                }
            }
            schema
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .filter(|n| wanted.contains(n))
                .collect()
        }
        None => schema
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .filter(|n| opts.keep_key_columns || !key_columns.contains(n))
            .collect(),
    };

    for src in opts.rename.keys() {
        if schema.field_with_name(src).is_err() {
            let present: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
            return Err(OpsError::InvalidInput(format!(
                "rename source column '{src}' is not in the table; columns present \
                 are {present:?}"
            )));
        }
    }

    let mut fields: Vec<Field> = Vec::with_capacity(selected.len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(selected.len());
    let mut out_names = HashSet::new();
    for name in &selected {
        let idx = schema.index_of(name)?;
        let renamed = opts
            .rename
            .get(name)
            .cloned()
            .unwrap_or_else(|| name.clone());
        let final_name = format!("{}{}", opts.prefix, renamed);
        if !out_names.insert(final_name.clone()) {
            return Err(OpsError::InvalidInput(format!(
                "column name '{final_name}' is produced twice after renaming/prefixing"
            )));
        }
        // Nullable regardless of what the source looked like: the attach op
        // scatters nulls into every target row the table does not cover.
        fields.push(Field::new(
            &final_name,
            table.column(idx).data_type().clone(),
            true,
        ));
        columns.push(Arc::clone(table.column(idx)));
    }

    if fields.is_empty() {
        return Err(OpsError::InvalidInput(
            "the table has no columns left to import after removing the key column(s); \
             pass keep_key_columns to import the key itself"
                .into(),
        ));
    }

    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| OpsError::InvalidInput(format!("failed to build annotation columns: {e}")))
}

// ---------------------------------------------------------------------------
// Source dispatch
// ---------------------------------------------------------------------------

/// Classify a source file by extension.
///
/// Extension rather than content sniffing: the caller named this file, and a
/// mis-typed extension should produce "that is not a table" rather than a
/// reader silently disagreeing with what the user thinks they passed.
pub fn sniff_obs_source(path: &Path) -> ObsSourceFormat {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("h5ad") | Some("h5") => ObsSourceFormat::H5ad,
        _ => ObsSourceFormat::Table,
    }
}

/// Read per-cell annotations from a delimited table **or** an h5ad's `/obs`.
///
/// The single entry point both `obs_import` and `doublet_import` go through, so
/// a new source format reaches every surface at once. `uns_keys` is honoured
/// only by the h5ad reader; a delimited table carries no `uns`.
pub fn read_obs_source(
    path: &Path,
    opts: &AnnotationTableOptions,
    uns_keys: &[String],
) -> Result<(ExternalObsData, AnnotationTableInfo)> {
    // An h5mu keeps obs per modality at `/mod/<name>/obs`, so there is no one
    // `/obs` to read. Rejected with the extraction route rather than supported
    // behind a `--modality` flag that would mean something for exactly one
    // source format — and a doublet caller is run per modality anyway.
    if path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("h5mu"))
    {
        return Err(OpsError::InvalidInput(format!(
            "'{}' is multimodal: its obs lives at /mod/<modality>/obs, so there is no \
             single obs table to import. Extract one modality first — \
             `scx subset --modality rna` — or read the modality in Python and write \
             its obs to a CSV.",
            path.display()
        )));
    }

    match sniff_obs_source(path) {
        ObsSourceFormat::Table => {
            if !uns_keys.is_empty() {
                return Err(OpsError::InvalidInput(format!(
                    "uns keys {uns_keys:?} were requested, but '{}' is a delimited table \
                     and carries no uns. Drop the request, or import from the h5ad the \
                     tool wrote.",
                    path.display()
                )));
            }
            read_annotation_table(path, opts)
        }
        ObsSourceFormat::H5ad => read_h5ad_obs_dispatch(path, opts, uns_keys),
    }
}

#[cfg(feature = "hdf5")]
fn read_h5ad_obs_dispatch(
    path: &Path,
    opts: &AnnotationTableOptions,
    uns_keys: &[String],
) -> Result<(ExternalObsData, AnnotationTableInfo)> {
    crate::h5ad_obs::read_h5ad_obs(path, opts, uns_keys)
}

/// Without `hdf5` there is no reader, but the failure must still tell the user
/// what to do instead — the CSV path works in this build and produces the same
/// result.
#[cfg(not(feature = "hdf5"))]
fn read_h5ad_obs_dispatch(
    path: &Path,
    _opts: &AnnotationTableOptions,
    _uns_keys: &[String],
) -> Result<(ExternalObsData, AnnotationTableInfo)> {
    Err(OpsError::InvalidInput(format!(
        "'{}' is an HDF5 file, but this build has no HDF5 support (the 'hdf5' feature \
         is off). Write the columns to a table first, e.g. \
         `adata.obs[[\"doublet_score\", \"predicted_doublet\"]].to_csv(\"calls.csv\")`, \
         and import that.",
        path.display()
    )))
}

#[cfg(test)]
#[path = "annotation_table_tests.rs"]
mod tests;
