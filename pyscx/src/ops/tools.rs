//! External-tool interop: CellBender layer import and the
//! doublet-caller wrapper (`doublet_import` / `doublet_tools` /
//! `doublet_profiles`).

use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;

use super::*;

// ---------------------------------------------------------------------------
// CellBender interop
// ---------------------------------------------------------------------------

/// Shared option parsing for `cellbender_import`, so the CLI and Python
/// surfaces cannot drift on the enum spellings.
#[cfg(feature = "hdf5")]
fn parse_missing_rows(s: &str) -> PyResult<scx_ops::MissingRowPolicy> {
    match s {
        "zero" => Ok(scx_ops::MissingRowPolicy::ZeroFill),
        "error" => Ok(scx_ops::MissingRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_missing_rows must be 'zero' or 'error'; got '{other}'"
        ))),
    }
}

#[cfg(feature = "hdf5")]
fn parse_extra_rows(s: &str) -> PyResult<scx_ops::ExtraRowPolicy> {
    match s {
        "warn" => Ok(scx_ops::ExtraRowPolicy::WarnSkip),
        "error" => Ok(scx_ops::ExtraRowPolicy::Error),
        other => Err(PyValueError::new_err(format!(
            "on_extra_rows must be 'warn' or 'error'; got '{other}'"
        ))),
    }
}

#[cfg(feature = "hdf5")]
fn parse_gene_axis(s: &str) -> PyResult<scx_ops::ColumnAxisPolicy> {
    match s {
        "identical" => Ok(scx_ops::ColumnAxisPolicy::RequireIdentical),
        "reorder" => Ok(scx_ops::ColumnAxisPolicy::AllowReorder),
        "subset" => Ok(scx_ops::ColumnAxisPolicy::AllowSubset),
        other => Err(PyValueError::new_err(format!(
            "gene_axis must be 'identical', 'reorder' or 'subset'; got '{other}'"
        ))),
    }
}

/// Import a CellBender `remove-background` output into an existing SCX file.
///
/// Reads the corrected count matrix and lands it as a new layer on `path`,
/// **in place**, joined to the target's own obs axis by barcode. X, the CSC
/// sidecar, `.raw` and deletion vectors are preserved; the whole import is
/// undoable with `scx rollback`.
///
/// Predicate indexes survive a pure column *add* on either axis. An
/// `overwrite=True` of an indexed column does not: the obs index is dropped,
/// and the **var** index is rebuilt over the new values — or retired, when none
/// of the columns it covered can still be indexed. The returned dict says which
/// happened, on a `dry_run` too: `var_index_rebuilt`, `var_index_dropped` and
/// `var_columns_not_carried`.
///
/// The join is always by barcode string, never by position: CellBender's
/// `_filtered.h5` is in descending-UMI order, so a positional import would
/// silently put every cell's corrected counts on the wrong barcode.
///
/// Returns a dict summarising the join — inspect `n_matched` before trusting
/// the result.
#[cfg(feature = "hdf5")]
#[pyfunction]
#[pyo3(signature = (
    path, cellbender_h5, *, layer="cellbender", obs_key=None, var_key=None,
    prefix="cellbender_", uns_key="cellbender", overwrite=false,
    on_missing_rows="zero", on_extra_rows="warn", gene_axis="identical",
    latent_embedding=false, dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn cellbender_import(
    py: Python<'_>,
    path: &str,
    cellbender_h5: &str,
    layer: &str,
    obs_key: Option<String>,
    var_key: Option<String>,
    prefix: &str,
    uns_key: Option<&str>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    gene_axis: &str,
    latent_embedding: bool,
    dry_run: bool,
) -> PyResult<Py<pyo3::types::PyDict>> {
    use pyo3::types::PyDict;

    let read_opts = scx_convert::CellBenderReadOptions {
        column_prefix: prefix.to_string(),
        latent_embedding,
        // `None` omits the diagnostics record; the reader keys it itself.
        uns_key: uns_key.map(str::to_string),
        ..Default::default()
    };
    let attach_opts = scx_ops::AttachLayerOptions {
        layer_name: layer.to_string(),
        obs_key_column: obs_key,
        var_key_column: var_key,
        missing_row_policy: parse_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_extra_rows(on_extra_rows)?,
        column_axis_policy: parse_gene_axis(gene_axis)?,
        status_column: Some(format!("{prefix}status")),
        row_sum_column: Some(format!("{prefix}total_counts")),
        overwrite,
        provenance_action: "cellbender_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let h5_path = PathBuf::from(cellbender_h5);

    // `dry_run` is honoured inside the op: it runs every validation and the
    // join, then returns without writing, so `n_matched` below is real.
    let (summary, info) = py.detach(|| -> PyResult<_> {
        let mut sink = scx_convert::WarningSink::log();
        let out = scx_convert::read_cellbender_h5(&h5_path, &read_opts, &mut sink)
            .map_err(crate::convert_to_pyerr)?;
        let summary = scx_ops::attach_external_layer(&scx_path, &out.data, &attach_opts)
            .map_err(ops_to_pyerr)?;
        Ok((summary, out.info))
    })?;

    let d = PyDict::new(py);
    d.set_item(
        "output_kind",
        format!("{:?}", info.output_kind).to_lowercase(),
    )?;
    d.set_item(
        "latent_alignment",
        format!("{:?}", info.latent_alignment).to_lowercase(),
    )?;
    d.set_item("n_rows_in_source", info.n_rows)?;
    d.set_item("n_features_in_source", info.n_features)?;
    d.set_item("estimator", info.estimator.clone())?;
    d.set_item("all_values_integer", info.all_values_integer)?;
    d.set_item("dry_run", dry_run)?;
    {
        let s = summary;
        d.set_item("layer", layer)?;
        d.set_item("n_obs", s.n_obs)?;
        d.set_item("n_matched", s.n_matched)?;
        d.set_item("n_target_rows_absent", s.n_target_rows_absent)?;
        d.set_item("n_source_rows_absent", s.n_source_rows_absent)?;
        d.set_item(
            "n_source_rows_absent_nonzero",
            s.n_source_rows_absent_nonzero,
        )?;
        d.set_item("obs_key_column", s.obs_key_column)?;
        d.set_item("var_key_column", s.var_key_column)?;
        d.set_item(
            "gene_axis_match",
            format!("{:?}", s.column_axis_match).to_lowercase(),
        )?;
        d.set_item("layer_nnz", s.layer_nnz)?;
        d.set_item("value_encoding", s.value_encoding.numpy_name())?;
        d.set_item("obs_columns_added", s.obs_columns_added)?;
        d.set_item("var_columns_added", s.var_columns_added)?;
        d.set_item("obsm_keys_added", s.obsm_keys_added)?;
        // Whether the obs rewrite streamed shard-by-shard or had to assemble
        // the whole table (which a legacy single-section obs forces). Not
        // inferable from the output file: both paths write a sharded obs.
        d.set_item("obs_streamed", s.obs_streamed)?;
        // The var predicate index outcome. An `overwrite` of an indexed var
        // column rebuilds the index from the new values; when none of the
        // covered columns can still be indexed the stale section is retired
        // instead, and `var_columns_not_carried` names what pushdown lost.
        d.set_item("var_index_rebuilt", s.var_index_rebuilt)?;
        d.set_item("var_index_dropped", s.var_index_dropped)?;
        d.set_item("var_columns_not_carried", s.var_columns_not_carried.clone())?;
    }
    Ok(d.into())
}

/// Probe whether a file looks like a CellBender `remove-background` output
/// (as opposed to a plain 10x CellRanger matrix).
#[cfg(feature = "hdf5")]
#[pyfunction]
pub fn is_cellbender_h5(path: &str) -> bool {
    scx_convert::is_cellbender_h5(std::path::Path::new(path))
}

// ---------------------------------------------------------------------------
// Doublet-caller wrapper
// ---------------------------------------------------------------------------

/// Import a doublet caller's output table, normalising it to canonical
/// columns (`<K>_score` / `<K>_predicted` / `<K>_status` + `uns["<K>"]`) —
/// the doublet-specific wrapper over [`obs_import`].
///
/// Full user-facing documentation lives on the `pyscx.doublet_import`
/// Python wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (
    path, table, *, tool, key=None, source_key=None, key_added=None, score_column=None,
    call_column=None, call_true=None, call_false=None, keep_native_columns=true,
    delimiter=None, uns_keys=None, overwrite=false, on_missing_rows="null",
    on_extra_rows="warn", dry_run=false
))]
#[allow(clippy::too_many_arguments)]
pub fn doublet_import(
    py: Python<'_>,
    path: &str,
    table: &str,
    tool: &str,
    key: Option<Vec<String>>,
    source_key: Option<Vec<String>>,
    key_added: Option<&str>,
    score_column: Option<&str>,
    call_column: Option<&str>,
    call_true: Option<&str>,
    call_false: Option<&str>,
    keep_native_columns: bool,
    delimiter: Option<&str>,
    uns_keys: Option<Vec<String>>,
    overwrite: bool,
    on_missing_rows: &str,
    on_extra_rows: &str,
    dry_run: bool,
) -> PyResult<Py<PyDict>> {
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

    // Resolve the profile up front so a typo'd tool name fails before any I/O,
    // and so `key_added` can default to the profile's own name.
    let profile = scx_convert::doublet_profile(tool).map_err(ops_to_pyerr)?;
    let resolved_key_added = key_added
        .filter(|s| !s.is_empty())
        .unwrap_or(profile.name)
        .to_string();

    let (key_columns, join_key) = resolve_join_key(key, source_key)?;

    let read_opts = scx_convert::DoubletImportOptions {
        tool: tool.to_string(),
        key_added: resolved_key_added.clone(),
        score_column: score_column.map(str::to_string),
        call_column: call_column.map(str::to_string),
        call_true: call_true.map(str::to_string),
        call_false: call_false.map(str::to_string),
        keep_native_columns,
        key_columns,
        delimiter: delimiter_byte,
        uns_keys: uns_keys.unwrap_or_default(),
    };
    let attach_opts = scx_ops::AttachObsOptions {
        join_key,
        missing_row_policy: parse_obs_missing_rows(on_missing_rows)?,
        extra_row_policy: parse_obs_extra_rows(on_extra_rows)?,
        // Part of the canonical contract rather than a knob: `_status` says
        // which cells the tool actually covered. (The `uns["<K>"]` record is
        // keyed by `read_doublet_table` itself.) `obs_import` remains the
        // surface where the status column is optional.
        status_column: Some(format!("{resolved_key_added}_status")),
        overwrite,
        provenance_action: "doublet_import".to_string(),
        dry_run,
        ..Default::default()
    };

    let scx_path = PathBuf::from(path);
    let table_path = PathBuf::from(table);

    let (summary, info, diagnosis) = py.detach(|| -> PyResult<_> {
        let (data, info) =
            scx_convert::read_doublet_table(&table_path, &read_opts).map_err(ops_to_pyerr)?;
        let summary =
            scx_ops::attach_external_obs(&scx_path, &data, &attach_opts).map_err(ops_to_pyerr)?;
        let diagnosis = if dry_run {
            scx_ops::diagnose_obs_key(&scx_path, Some(&attach_opts.join_key)).ok()
        } else {
            None
        };
        Ok((summary, info, diagnosis))
    })?;

    // The profile declares a call column and the table carried none of its
    // spellings, so `<K>_predicted` was NOT written even though this tool does
    // emit a call. Warn HERE (after `py.detach` closes — `warnings.warn` is
    // unreachable inside it) rather than leave the user to discover it at
    // `doublet_consensus`, which is where the dogfood run found it, under the
    // false claim that the tool emits no call column. In-module pattern: see
    // `process_index_summary`.
    if let Some(m) = &info.call_column_missing {
        // `m.expected` already renders aliases AND prefix in one phrase
        // (built by `resolve_column`), so do not re-append the prefix.
        let expected = &m.expected;
        // A near-miss is usually the user having named the wrong `--tool`, so
        // point at the actual column when exactly one unconsumed column is a
        // declared call spelling of some other profile.
        let candidates: Vec<String> = m
            .present_columns
            .iter()
            .filter(|c| {
                scx_convert::DOUBLET_PROFILE_NAMES.iter().any(|t| {
                    scx_convert::doublet_profile(t)
                        .map(|p| p.call_columns.contains(&c.as_str()))
                        .unwrap_or(false)
                })
            })
            .cloned()
            .collect();
        let suggestion = if candidates.len() == 1 {
            format!(
                " The table does carry {:?}, which is another tool's call column — \
                 pass call_column={:?} if that is your call.",
                candidates[0], candidates[0]
            )
        } else {
            String::new()
        };
        let key = &info.key_added;
        let msg = format!(
            "doublet_import: tool={:?} declares a call column ({expected}) but the table has \
             none of those names — columns present are {:?}. Imported SCORE ONLY: \
             obs[{:?}] was written, obs[{:?}] was NOT, so this tool cannot vote on a call in \
             pyscx.doublet_consensus. The unmatched column is preserved verbatim under the \
             {:?} prefix.{suggestion} Re-import with call_column=<your column>, or pass the \
             tool= whose profile matches this table.",
            info.tool,
            m.present_columns,
            format!("{key}_score"),
            format!("{key}_predicted"),
            key,
        );
        crate::pyimport::import_module(py, "warnings")?.call_method1("warn", (msg,))?;
    }

    let d = PyDict::new(py);
    d.set_item("tool", &info.tool)?;
    d.set_item("key_added", &info.key_added)?;
    d.set_item("score_source_column", info.score_source_column)?;
    d.set_item("call_source_column", &info.call_source_column)?;
    // Lets a script branch without parsing the warning string. Mirrors the
    // `call_column_status` recorded in `uns["<K>"]`.
    d.set_item(
        "call_column_status",
        match (&info.call_source_column, &info.call_column_missing) {
            (Some(_), _) => "resolved",
            (None, None) => "not_declared",
            (None, Some(_)) => "declared_but_absent",
        },
    )?;
    d.set_item(
        "expected_call_columns",
        info.call_column_missing
            .as_ref()
            .map(|m| m.expected_columns.clone())
            .unwrap_or_default(),
    )?;
    d.set_item("canonical_columns", info.canonical_columns)?;
    d.set_item("native_columns", info.native_columns)?;
    d.set_item("dropped_alias_columns", info.dropped_alias_columns)?;
    d.set_item("n_rows_in_source", info.table.n_rows)?;
    d.set_item("format", info.table.format.as_str())?;
    d.set_item(
        "delimiter",
        info.table.delimiter.map(|b| (b as char).to_string()),
    )?;
    d.set_item("uns_keys_imported", info.table.uns_keys_imported)?;
    d.set_item("renamed_index_column", info.table.renamed_index_column)?;
    d.set_item("source_key_columns", info.table.key_columns)?;
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
    d.set_item("obs_index_dropped", summary.obs_index_dropped)?;
    // See `obs_import`: which obs rewrite path ran is not inferable from the
    // output file, because both write a sharded obs.
    d.set_item("obs_streamed", summary.obs_streamed)?;
    if let Some(diag) = diagnosis {
        d.set_item("key_diagnosis", key_diagnosis_dict(py, &diag)?)?;
    }
    Ok(d.into())
}

/// The valid `tool=` values for [`doublet_import`], in table order.
#[pyfunction]
pub fn doublet_tools() -> Vec<String> {
    scx_convert::DOUBLET_PROFILE_NAMES
        .iter()
        .map(|s| s.to_string())
        .collect()
}

/// Every `tool=` profile's column vocabulary, straight from the definitions.
///
/// Exists so the per-tool table in `docs/scanpy.md` is machine-checkable rather
/// than hand-maintained, and so a user surprised by an import can look up what
/// their `--tool` actually expects from a REPL instead of reading Rust. That
/// lookup being unavailable is what made a call-column name mismatch a
/// silent score-only import.
#[pyfunction]
pub fn doublet_profiles(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let out = PyDict::new(py);
    for name in scx_convert::DOUBLET_PROFILE_NAMES {
        let p = scx_convert::doublet_profile(name).map_err(ops_to_pyerr)?;
        let d = PyDict::new(py);
        d.set_item("score_columns", p.score_columns.to_vec())?;
        d.set_item("score_prefix", p.score_prefix)?;
        d.set_item("call_columns", p.call_columns.to_vec())?;
        d.set_item("call_prefix", p.call_prefix)?;
        d.set_item("call_tokens", p.call_tokens.map(|t| (t.doublet, t.singlet)))?;
        // `!call_columns.is_empty() || call_prefix.is_some()` — the one fact
        // that decides whether `<K>_predicted` can exist at all.
        d.set_item("emits_call", scx_convert::profile_has_call_column(p))?;
        out.set_item(*name, d)?;
    }
    Ok(out.into())
}
