//! The external-obs join family: `obs_import` (delimited tables /
//! h5ad obs), `attach_obs_columns` (in-memory DataFrames),
//! `diagnose_obs_key`, and the shared join-key / row-policy parsing.

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use super::*;
use crate::convert;

// ---------------------------------------------------------------------------
// Generic external obs import (CSV / TSV annotation tables)
// ---------------------------------------------------------------------------

/// Shared option parsing, so the CLI and Python surfaces cannot drift on the
/// enum spellings. Ungated, unlike the CellBender pair above — a delimited
/// table reader has no business requiring libhdf5.
pub fn parse_obs_missing_rows(s: &str) -> PyResult<scx_ops::MissingRowPolicy> {
    match s {
        // The obs paths scatter Arrow *nulls*, not zeros, and null-aware
        // consensus depends on that — so "null" is the accurate spelling.
        // "zero" stays accepted: it is the name the CellBender-era policy
        // enum carries and what earlier callers pass.
        "null" | "zero" => Ok(scx_ops::MissingRowPolicy::ZeroFill),
        "error" => Ok(scx_ops::MissingRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_missing_rows must be 'null' (leave uncovered rows NULL, the \
             default), 'zero' (a legacy alias for the same thing) or 'error'; \
             got '{other}'"
        ))),
    }
}

pub fn parse_obs_extra_rows(s: &str) -> PyResult<scx_ops::ExtraRowPolicy> {
    match s {
        "warn" => Ok(scx_ops::ExtraRowPolicy::WarnSkip),
        "error" => Ok(scx_ops::ExtraRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_extra_rows must be 'warn' or 'error'; got '{other}'"
        ))),
    }
}

/// Turn the caller's `key` / `source_key` into the reader's column list and the
/// op's join-key spec.
///
/// Without `source_key` both sides are built from the same names, which is the
/// common case and the only one expressible before: a target obs keyed on
/// (`sample_id`, obs index) could not be joined to a tool output keyed on
/// (`sample_id`, `barcode`) without renaming a column in pandas first. The two
/// lists pair up **positionally**, mirroring pandas `left_on` / `right_on`.
///
/// Nothing downstream needs to know the names differ: `build_composite_key`
/// fuses each side from its own columns in the given order and the fused key
/// never carries a column name.
pub(crate) fn resolve_join_key(
    key: Option<Vec<String>>,
    source_key: Option<Vec<String>>,
) -> PyResult<(Vec<String>, scx_ops::ObsJoinKey)> {
    let target = key.unwrap_or_default();
    let source = source_key.unwrap_or_default();
    if !source.is_empty() && target.is_empty() {
        return Err(PyValueError::new_err(
            "source_key= needs key=: it names the source-side column for each \
             target-side key component, positionally. To key on the target's obs \
             index, pass key=\"obs_names\".",
        ));
    }
    if !source.is_empty() && source.len() != target.len() {
        return Err(PyValueError::new_err(format!(
            "key= has {} component(s) but source_key= has {}; they pair up \
             positionally, so the counts must match",
            target.len(),
            source.len()
        )));
    }
    let join_key = match target.len() {
        0 => scx_ops::ObsJoinKey::Auto,
        1 => scx_ops::ObsJoinKey::Column(target[0].clone()),
        _ => scx_ops::ObsJoinKey::Composite {
            columns: target.clone(),
        },
    };
    // Absent `source_key`, the reader gets the target names — the historical
    // behaviour, and still what makes the two sides impossible to desync.
    let source_columns = if source.is_empty() { target } else { source };
    Ok((source_columns, join_key))
}

pub(crate) fn key_diagnosis_dict<'py>(
    py: Python<'py>,
    d: &scx_ops::KeyDiagnosis,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n_obs", d.n_obs)?;
    out.set_item("resolved_key", d.resolved_key.clone())?;
    out.set_item("resolved_cardinality", d.resolved_cardinality)?;
    out.set_item("unique_columns", d.unique_columns.clone())?;
    // Unique, but refused as a key — a float, or any other type that is not
    // guaranteed to render identically on two independently written sides. Kept
    // out of `unique_columns` so nothing in that list is a key the join rejects.
    out.set_item("unusable_unique_columns", d.unusable_unique_columns.clone())?;
    let pairs: Vec<(String, String)> = d.unique_pairs.clone();
    out.set_item("unique_pairs", pairs)?;
    out.set_item("pair_search_capped", d.pair_search_capped)?;
    out.set_item("suggestion", d.suggestion.clone())?;
    out.set_item("summary", d.describe())?;
    Ok(out)
}

/// Import a delimited annotation table (CSV / TSV) as `obs` columns, in
/// place, joined to the target's own obs axis by key string — never by row
/// position.
///
/// Full user-facing documentation lives on the `pyscx.obs_import` Python
/// wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (
    path, table, *, key=None, source_key=None, columns=None, rename=None, prefix="",
    keep_key_columns=false, delimiter=None, status_column=None, uns_key=None,
    uns_keys=None, overwrite=false, on_missing_rows="null", on_extra_rows="warn",
    dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn obs_import(
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

    let (key_columns, join_key) = resolve_join_key(key, source_key)?;

    let read_opts = scx_convert::AnnotationTableOptions {
        key_columns,
        delimiter: delimiter_byte,
        columns,
        rename: rename.unwrap_or_default(),
        prefix: prefix.to_string(),
        keep_key_columns,
        infer_max_records: None,
    };
    let attach_opts = scx_ops::AttachObsOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        status_column: status_column.map(str::to_string),
        uns_key: uns_key.map(str::to_string),
        overwrite,
        provenance_action: "obs_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let table_path = PathBuf::from(table);

    // Both halves return `OpsError`, so one mapping covers read-then-attach —
    // a malformed CSV surfaces as a clean `ValueError`, not a panic.
    let (summary, info, diagnosis) = py.detach(|| -> PyResult<_> {
        let (data, info) = scx_convert::read_obs_source(&table_path, &read_opts, &uns_keys)
            .map_err(ops_to_pyerr)?;
        let summary =
            scx_ops::attach_external_obs(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        // Only on a dry run: the diagnosis costs a pass per obs column, which
        // is not something a successful import should pay for.
        let diagnosis = if dry_run {
            scx_ops::diagnose_obs_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, info, diagnosis))
    })?;

    let d = PyDict::new(py);
    d.set_item("n_rows_in_source", info.n_rows)?;
    d.set_item("format", info.format.as_str())?;
    d.set_item("delimiter", info.delimiter.map(|b| (b as char).to_string()))?;
    d.set_item("uns_keys_imported", info.uns_keys_imported)?;
    d.set_item("renamed_index_column", info.renamed_index_column)?;
    d.set_item("source_key_columns", info.key_columns)?;
    d.set_item("columns_in_source", info.columns_imported)?;
    d.set_item("dry_run", dry_run)?;
    d.set_item("n_obs", summary.n_obs)?;
    d.set_item("n_matched", summary.n_matched)?;
    d.set_item("n_target_rows_absent", summary.n_target_rows_absent)?;
    d.set_item("n_source_rows_absent", summary.n_source_rows_absent)?;
    d.set_item(
        "obs_key_column",
        scx_ops::display_key_name("obs", &summary.obs_key_column),
    )?;
    d.set_item("obs_columns_added", summary.obs_columns_added)?;
    d.set_item("obsm_keys_added", summary.obsm_keys_added)?;
    d.set_item("obs_index_dropped", summary.obs_index_dropped)?;
    // Whether the obs rewrite streamed shard-by-shard or had to assemble the
    // whole table (which a legacy single-section obs forces). Not inferable
    // from the output file: both paths write a sharded obs.
    d.set_item("obs_streamed", summary.obs_streamed)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

// ---------------------------------------------------------------------------
// Generic in-memory obs-column attach (DataFrame)
// ---------------------------------------------------------------------------

/// Land an in-memory pandas `DataFrame` (or pyarrow `Table`) as `obs` columns
/// on an existing file, **in place** — the DataFrame twin of [`obs_import`],
/// driving the same `scx_ops::attach_external_obs` seam rscx's
/// `scx_attach_obs` uses. Ungated, like `obs_import`: no libhdf5 involved.
///
/// Full user-facing documentation lives on the `pyscx.attach_obs_columns`
/// Python wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (
    path, df, *, key=None, positional=false, status_column=None, uns=None,
    uns_key=None, overwrite=false, on_missing_rows="null", on_extra_rows="warn",
    dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn attach_obs_columns(
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
             row i of df annotates physical obs row i, so there is no key to \
             join on",
        ));
    }
    if uns.is_some() != uns_key.is_some() {
        return Err(PyValueError::new_err(
            "uns= and uns_key= go together: uns is the payload, uns_key names \
             the single uns key it is merged under",
        ));
    }

    let batch = obs_var_to_record_batch(py, df, "attach_obs_columns", "df")?;
    if batch.num_rows() == 0 {
        return Err(PyValueError::new_err(
            "df has no rows; there is nothing to attach",
        ));
    }

    // Both sides of a key join are built by the same ops-crate helpers the CSV
    // reader and rscx use, so a DataFrame attach and an imported table cannot
    // disagree about what a key is. `drop` names the columns of `df` the key
    // consumed — they are already on the target's obs axis, so re-importing
    // them would only duplicate them.
    let key = key.unwrap_or_default();
    let (row_keys, join_key, mut drop): (Vec<String>, scx_ops::ObsJoinKey, Vec<String>) =
        if positional {
            (Vec::new(), scx_ops::ObsJoinKey::Positional, Vec::new())
        } else {
            match key.len() {
                0 => {
                    // Both sides resolve independently, exactly as `obs_import`
                    // with no `key=`: the source uses its own index (named or
                    // not) / fallbacks, the target its own obs index /
                    // fallbacks. `Column(col)` here would send the SOURCE
                    // index's field name to the target — wrong the moment the
                    // df's index is named. (Round-1 finding: codex.)
                    let col =
                        scx_ops::resolve_obs_key_column(&batch, None).map_err(ops_to_pyerr)?;
                    let values = scx_ops::obs_key_values(&batch, &col).map_err(ops_to_pyerr)?;
                    (values, scx_ops::ObsJoinKey::Auto, vec![col])
                }
                1 => {
                    // Alias-resolved (`obs_names` names the df index), same as
                    // the delimited reader's source side.
                    let col = scx_ops::resolve_obs_key_column(&batch, Some(&key[0]))
                        .map_err(ops_to_pyerr)?;
                    let values = scx_ops::obs_key_values(&batch, &col).map_err(ops_to_pyerr)?;
                    (
                        values,
                        scx_ops::ObsJoinKey::Column(key[0].clone()),
                        vec![col],
                    )
                }
                _ => {
                    let fused = scx_ops::build_composite_key(&batch, &key).map_err(ops_to_pyerr)?;
                    (
                        fused,
                        scx_ops::ObsJoinKey::Composite {
                            columns: key.clone(),
                        },
                        key.clone(),
                    )
                }
            }
        };
    // The pandas index is never data to attach: under a key join it is either
    // the key itself or a RangeIndex, and under positional the row order
    // already carries the identity. `pyarrow.Table.from_pandas` materializes a
    // NAMED index under its own name — only an unnamed one becomes
    // `__index_level_0__` — so the drop list comes from the schema's pandas
    // `index_columns` metadata, with the literal kept as the fallback for an
    // envelope-less batch. (Round-1 finding: Cursor Agent / codex /
    // Antigravity, independently.)
    drop.extend(scx_format_io::pandas_index_columns(&batch.schema()));
    drop.push("__index_level_0__".to_string());
    let annotations = scx_ops::drop_batch_columns(&batch, &drop).map_err(ops_to_pyerr)?;

    let uns_json = match uns {
        Some(u) => Some(convert::uns_py_to_json(py, u, convert::UnsFormat::Tagged)?),
        None => None,
    };

    let data = scx_ops::ExternalObsData {
        row_keys,
        row_annotations: annotations,
        row_embeddings: Vec::new(),
        uns: uns_json,
        source_checksum: None,
        source_name: Some("<DataFrame>".to_string()),
    };
    let attach_opts = scx_ops::AttachObsOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        status_column: status_column.map(str::to_string),
        uns_key: uns_key.map(str::to_string),
        overwrite,
        provenance_action: "attach_obs_columns".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let (summary, diagnosis) = py.detach(|| -> PyResult<_> {
        let summary =
            scx_ops::attach_external_obs(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        // Only on a key-mode dry run: the diagnosis costs a pass per obs
        // column, and under positional there is no key to diagnose.
        let diagnosis = if dry_run && !positional {
            scx_ops::diagnose_obs_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, diagnosis))
    })?;

    let d = PyDict::new(py);
    d.set_item("dry_run", dry_run)?;
    d.set_item("n_obs", summary.n_obs)?;
    d.set_item("n_matched", summary.n_matched)?;
    d.set_item("n_target_rows_absent", summary.n_target_rows_absent)?;
    d.set_item("n_source_rows_absent", summary.n_source_rows_absent)?;
    d.set_item(
        "obs_key_column",
        scx_ops::display_key_name("obs", &summary.obs_key_column),
    )?;
    d.set_item("obs_columns_added", summary.obs_columns_added)?;
    d.set_item("obsm_keys_added", summary.obsm_keys_added)?;
    d.set_item("obs_index_dropped", summary.obs_index_dropped)?;
    d.set_item("obs_streamed", summary.obs_streamed)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

/// Report which obs columns could serve as a join key for `obs_import`.
/// Read-only.
///
/// Full user-facing documentation lives on the `pyscx.diagnose_obs_key`
/// Python wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (path, key=None))]
pub fn diagnose_obs_key(
    py: Python<'_>,
    path: &str,
    key: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    let (_, join_key) = resolve_join_key(key, None)?;
    let scx_path = PathBuf::from(path);
    let diag = py
        .detach(|| scx_ops::diagnose_obs_key(&scx_path, Some(&join_key)))
        .map_err(ops_to_pyerr)?;
    Ok(key_diagnosis_dict(py, &diag)?.into())
}
