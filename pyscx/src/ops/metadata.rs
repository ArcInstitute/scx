//! In-place metadata replacement: `set_uns`, `modify_metadata`.

use std::path::PathBuf;

use scx_engine::ConversionPredicateIndexOptions;

use super::*;
use crate::convert;

// ---------------------------------------------------------------------------
// In-place metadata replacement (set_uns / modify_metadata)
// ---------------------------------------------------------------------------

/// Replace the whole `uns` block of an existing `.scx` file in place,
/// without re-encoding `X`. Replace semantics, not merge.
///
/// Full user-facing documentation lives on the `pyscx.set_uns` Python
/// wrapper, which is what `help()` shows.
#[pyfunction]
pub fn set_uns(py: Python<'_>, path: &str, uns: &Bound<'_, PyAny>) -> PyResult<()> {
    let json = convert::uns_py_to_json(py, uns, convert::UnsFormat::Tagged)?;
    let path_buf = PathBuf::from(path);
    py.detach(|| scx_ops::set_uns(&path_buf, &json))
        .map_err(ops_to_pyerr)?;
    Ok(())
}

/// Replace metadata sections (`uns` / `obs` / `var` / `obsm` / `varm`) of an
/// existing `.scx` file in place, without re-encoding `X`. Replace semantics,
/// not merge; a replaced axis keeps (rebuilds) the predicate index it had.
///
/// Full user-facing documentation lives on the `pyscx.modify_metadata`
/// Python wrapper, which is what `help()` shows.
#[pyfunction]
#[pyo3(signature = (
    path, *, uns=None, obs=None, var=None, obsm=None, varm=None,
    index_obs=None, index_var=None, index_preset=None, index_auto_threshold=None,
    modality=None,
))]
#[allow(clippy::too_many_arguments)]
pub fn modify_metadata(
    py: Python<'_>,
    path: &str,
    uns: Option<&Bound<'_, PyAny>>,
    obs: Option<&Bound<'_, PyAny>>,
    var: Option<&Bound<'_, PyAny>>,
    obsm: Option<&Bound<'_, PyAny>>,
    varm: Option<&Bound<'_, PyAny>>,
    index_obs: Option<Vec<String>>,
    index_var: Option<Vec<String>>,
    index_preset: Option<String>,
    index_auto_threshold: Option<usize>,
    modality: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let uns_json = match uns {
        Some(u) => Some(convert::uns_py_to_json(py, u, convert::UnsFormat::Tagged)?),
        None => None,
    };
    let obs_batch = match obs {
        Some(o) => Some(obs_var_to_record_batch(py, o, "modify_metadata", "obs")?),
        None => None,
    };
    let var_batch = match var {
        Some(v) => Some(obs_var_to_record_batch(py, v, "modify_metadata", "var")?),
        None => None,
    };
    let obsm_batches = match obsm {
        Some(d) => Some(dense_dict_to_batches(py, d, "obsm")?),
        None => None,
    };
    let varm_batches = match varm {
        Some(d) => Some(dense_dict_to_batches(py, d, "varm")?),
        None => None,
    };
    let modality_id = resolve_modality_id(modality)?;
    // Built here rather than via `build_index_options`, which defaults
    // `index_auto_threshold` to 1000 whenever *any* index kwarg is set. That
    // default is right for the conversion ops, and wrong here: this op reads
    // `index_auto_threshold > 0` as "the caller wants auto-detection on both
    // axes", so `modify_metadata(obs=…, index_var=[…])` would silently take the
    // obs axis off carry-forward and onto auto-detect. `0` means "no auto
    // unless you asked for it", which is what an omitted kwarg means.
    let index = ConversionPredicateIndexOptions {
        index_obs: index_obs.unwrap_or_default(),
        index_var: index_var.unwrap_or_default(),
        index_preset,
        index_auto_threshold: index_auto_threshold.unwrap_or(0),
    };

    let patch = scx_ops::MetadataPatch {
        uns: uns_json,
        obs: obs_batch,
        var: var_batch,
        obsm: obsm_batches,
        varm: varm_batches,
        index,
        modality_id,
    };
    let path_buf = PathBuf::from(path);
    let summary = py
        .detach(|| scx_ops::modify_metadata(&path_buf, &patch))
        .map_err(ops_to_pyerr)?;
    report_modify_metadata_index(py, summary)
}

/// Surface what the op did to the file's predicate indexes.
///
/// Two channels, one warning per column and no doubling between them:
///
/// * `process_index_summary` reports every per-column build outcome with its
///   precise reason — which on the carry-forward path is every column that
///   could not be carried.
/// * The remaining `*_columns_not_carried` entries are the explicit-request
///   path, where the builder knows nothing about the index the file had and so
///   emits no outcome for a column the caller simply did not name.
fn report_modify_metadata_index(
    py: Python<'_>,
    summary: scx_ops::ModifyMetadataSummary,
) -> PyResult<()> {
    // Computed before `process_index_summary` consumes the summary. The filter
    // lives in `scx-ops` so this and the CLI cannot drift on it — the CLI
    // rendering both channels unfiltered is exactly how one column came to be
    // warned about twice.
    let obs_unreported = summary.obs_not_carried_unreported();
    let var_unreported = summary.var_not_carried_unreported();

    process_index_summary(py, summary.index)?;

    for (axis, columns) in [("obs", obs_unreported), ("var", var_unreported)] {
        if columns.is_empty() {
            continue;
        }
        crate::pyimport::import_module(py, "warnings")?.call_method1(
            "warn",
            (format!(
                "the {axis} predicate index no longer covers {columns:?}: this file indexed \
                 {axis} on them, and the index_{axis} / index_preset passed to this call does \
                 not. Queries on those columns fall back to a full scan. Include them to keep \
                 the pushdown, or omit index_* entirely to carry the file's own index forward."
            ),),
        )?;
    }
    Ok(())
}
