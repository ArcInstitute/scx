//! Energy-distance kernels: backend/dtype parsing and the
//! `energy_distance` / `energy_distance_details` entry points.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::types::PyDict;

// ──────────────────────────────────────────────────────────────────────────────
// device scaffolding (Phase 0 of CELL-EVAL-SCX-GPU-ACC.md)
// ──────────────────────────────────────────────────────────────────────────────
use super::*;

/// Parse `backend` kwarg into `DistanceBackend`.
fn parse_backend(backend: Option<&str>) -> PyResult<scx_accel::DistanceBackend> {
    match backend.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("auto") => Ok(scx_accel::DistanceBackend::Auto),
        Some("gemm") | Some("blas") => Ok(scx_accel::DistanceBackend::Gemm),
        Some("scalar") => Ok(scx_accel::DistanceBackend::Scalar),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown backend '{other}'. Valid: auto, gemm, scalar"
        ))),
    }
}

/// Dtype dispatch tag, picked out of the `dtype` kwarg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    F32,
    F64,
}

/// Parse `dtype` kwarg. Default is `f32` to maximise throughput.
fn parse_dtype(dtype: Option<&str>) -> PyResult<Dtype> {
    match dtype.map(|s| s.to_ascii_lowercase()).as_deref() {
        None | Some("f32") | Some("float32") => Ok(Dtype::F32),
        Some("f64") | Some("float64") => Ok(Dtype::F64),
        Some(other) => Err(PyValueError::new_err(format!(
            "unknown dtype '{other}'. Valid: f32, f64"
        ))),
    }
}

/// Inner generic implementation of `run_energy_distance`. Materialises both
/// AnnData matrices in the chosen `F` precision, then dispatches to the
/// generic `compute_energy_distance::<F>` kernel.
#[allow(clippy::too_many_arguments)]
fn run_energy_distance_inner<'py, F>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    dist_metric: scx_accel::DistanceMetric,
    embed_key: Option<&str>,
    dist_backend: scx_accel::DistanceBackend,
    gpu_dev: &EvalGpuDev,
) -> PyResult<scx_accel::EDistanceResult>
where
    F: scx_accel::PairwiseFloat + NumpyDtype,
{
    #[cfg(not(feature = "gpu"))]
    let _ = gpu_dev;
    // ── Extract dense matrices from both AnnData objects ────────────
    let (real_flat, n_real, n_dims_real) = materialize_dense::<F>(py, np, adata_real, embed_key)?;
    let (pred_flat, n_pred, n_dims_pred) = materialize_dense::<F>(py, np, adata_pred, embed_key)?;

    if n_dims_real != n_dims_pred {
        return Err(PyValueError::new_err(format!(
            "dimension mismatch: real has {} features but pred has {}",
            n_dims_real, n_dims_pred
        )));
    }
    let n_dims = n_dims_real;

    // ── Extract perturbation labels ─────────────────────────────────
    let real_labels = extract_obs_column(py, adata_real, pert_col)?;
    let pred_labels = extract_obs_column(py, adata_pred, pert_col)?;

    if real_labels.len() != n_real {
        return Err(PyValueError::new_err(format!(
            "real adata has {} obs but X has {} rows",
            real_labels.len(),
            n_real
        )));
    }
    if pred_labels.len() != n_pred {
        return Err(PyValueError::new_err(format!(
            "pred adata has {} obs but X has {} rows",
            pred_labels.len(),
            n_pred
        )));
    }

    // ── Build group → index mapping ─────────────────────────────────
    let mut all_groups = std::collections::BTreeSet::new();
    for l in real_labels.iter().chain(pred_labels.iter()) {
        all_groups.insert(l.as_str());
    }
    let group_to_idx: std::collections::HashMap<&str, u32> = all_groups
        .iter()
        .enumerate()
        .map(|(i, &g)| (g, i as u32))
        .collect();

    let real_groups: Vec<u32> = real_labels
        .iter()
        .map(|l| group_to_idx[l.as_str()])
        .collect();
    let pred_groups: Vec<u32> = pred_labels
        .iter()
        .map(|l| group_to_idx[l.as_str()])
        .collect();

    let ctrl_group_idx = *group_to_idx.get(control).ok_or_else(|| {
        PyValueError::new_err(format!(
            "control '{}' not found in perturbation labels. Available: {:?}",
            control,
            all_groups.iter().collect::<Vec<_>>()
        ))
    })?;

    let mut pert_names = Vec::new();
    let mut pert_group_indices = Vec::new();
    for &g in &all_groups {
        if g != control {
            pert_names.push(g.to_string());
            pert_group_indices.push(group_to_idx[g]);
        }
    }

    if pert_names.is_empty() {
        return Err(PyValueError::new_err(
            "no non-control perturbation groups found",
        ));
    }

    // GPU path (f32 + euclidean/cosine only): F::edist_gpu returns None for f64
    // so the generic dispatch falls through to CPU. GpuDevice is !Send, so the
    // GIL is held around the device work (mirrors agg_backed / pdex_nb_glm).
    #[cfg(feature = "gpu")]
    if let Some(dev) = gpu_dev.as_ref() {
        if !matches!(dist_metric, scx_accel::DistanceMetric::L1) {
            if let Some(res) = F::edist_gpu(
                dev,
                &real_flat,
                &pred_flat,
                &real_groups,
                &pred_groups,
                ctrl_group_idx,
                &pert_names,
                &pert_group_indices,
                n_dims,
                dist_metric,
            ) {
                return res.map_err(|e| PyRuntimeError::new_err(e.to_string()));
            }
        }
    }

    // CPU path. Release the GIL for the O(N²) rayon-parallel kernel. All
    // arguments are owned Vecs or plain scalars.
    py.detach(|| {
        scx_accel::compute_energy_distance::<F>(
            &real_flat,
            &pred_flat,
            &real_groups,
            &pred_groups,
            ctrl_group_idx,
            &pert_names,
            &pert_group_indices,
            n_dims,
            dist_metric,
            dist_backend,
        )
    })
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// Shared prep + compute for energy_distance / energy_distance_details.
#[allow(clippy::too_many_arguments)]
fn run_energy_distance<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
    backend: Option<&str>,
    dtype: Option<&str>,
    gpu_dev: &EvalGpuDev,
) -> PyResult<scx_accel::EDistanceResult> {
    let np = crate::pyimport::import_module(py, "numpy")?;

    // Parse distance metric.
    let dist_metric = match metric.to_lowercase().as_str() {
        "euclidean" | "l2" => scx_accel::DistanceMetric::Euclidean,
        "l1" | "manhattan" | "cityblock" => scx_accel::DistanceMetric::L1,
        "cosine" => scx_accel::DistanceMetric::Cosine,
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown metric '{}'. Valid: euclidean, l1, cosine",
                metric
            )))
        }
    };

    let dist_backend = parse_backend(backend)?;
    let dtype = parse_dtype(dtype)?;

    match dtype {
        Dtype::F32 => run_energy_distance_inner::<f32>(
            py,
            &np,
            adata_real,
            adata_pred,
            pert_col,
            control,
            dist_metric,
            embed_key,
            dist_backend,
            gpu_dev,
        ),
        Dtype::F64 => run_energy_distance_inner::<f64>(
            py,
            &np,
            adata_real,
            adata_pred,
            pert_col,
            control,
            dist_metric,
            embed_key,
            dist_backend,
            gpu_dev,
        ),
    }
}

/// Score a perturbation-effect prediction by the cell-eval energy-distance metric.
///
/// For each perturbation, computes the energy distance (e-distance) between
/// perturbation cells and control cells — separately on the real and the
/// predicted side — then returns the **Pearson correlation** of the
/// per-perturbation real-vs-predicted e-distance vectors.
///
/// IMPORTANT — polarity: despite the name, the returned scalar is a
/// *correlation*, not a distance. It lies in ``[-1.0, 1.0]`` where
/// **1.0 = perfect prediction and higher is better** (the opposite of a raw
/// distance, where lower is better). Use it directly as a score; do not invert
/// it. The per-perturbation e-distances themselves (where smaller = more
/// similar to control) are exposed via :func:`energy_distance_details`.
///
/// Returns ``nan`` when the correlation is undefined — i.e. when either side's
/// e-distance vector has zero variance (e.g. a constant or control-broadcast
/// predictor that gives every perturbation the same e-distance), or when there
/// is only a single non-control perturbation.
///
/// This is the most expensive cell-eval metric — O(N²) pairwise distances
/// per perturbation. The Rust implementation avoids materializing N×N
/// distance matrices (streaming accumulation), precomputes control
/// self-distances once, and parallelizes across perturbations with rayon.
///
/// Args:
///     adata_real: AnnData with real (ground truth) data
///     adata_pred: AnnData with predicted data
///     pert_col: Column name in obs for perturbation labels (default: "perturbation")
///     control: Label for control perturbation (default: "control")
///     metric: Distance metric — "euclidean" (default), "l1", or "cosine"
///     embed_key: If set, use adata.obsm[embed_key] instead of X (default: None)
///     backend: Distance kernel backend — "auto" (default), "gemm", or "scalar".
///         "auto" uses faer-backed gemm for euclidean/cosine and the scalar
///         row-by-row path for L1. "gemm" forces the gemm path (errors on L1).
///         "scalar" forces the scalar path (matches the historical implementation).
///     dtype: Element precision for the dense kernel — "f32" (default) or "f64".
///         Reductions always accumulate in f64; the dtype only controls the
///         matmul / per-pair compute precision. f32 is ~1.5–2× faster on AVX2
///         and matches f64 within atol=1e-4 on log-normalised inputs.
///
/// Returns:
///     float — Pearson correlation of per-perturbation real-vs-predicted
///     e-distances, in [-1.0, 1.0] (1.0 = perfect, higher = better; `nan` if
///     either side's e-distance vector is constant). For the underlying
///     per-perturbation e-distance vectors, use `energy_distance_details`.
///
/// Example:
///     corr = pyscx.accel.energy_distance(adata_real, adata_pred)
///     # corr ≈ 0.85 means real and predicted perturbation effects
///     # have similar relative magnitudes (higher = better, 1.0 = perfect)
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn energy_distance<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
    backend: Option<&str>,
    dtype: Option<&str>,
    device: &str,
) -> PyResult<f64> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    crate::accel::prepare_target_no_var_guard(py, adata_pred, "energy_distance")?;
    let (gpu_dev, info) = edist_device_dispatch(py, "energy_distance", device, metric, dtype)?;
    let result = run_energy_distance(
        py, adata_real, adata_pred, pert_col, control, metric, embed_key, backend, dtype, &gpu_dev,
    )?;
    // Stamped after the compute, so there is nothing to roll back — a raise
    // above never reaches this line. If this ever moves to a pre-dispatch
    // stamp (as the thirteen progress-reporting ops did), it must switch to
    // `RouteStamp::write` or it will leave a stamp behind on failure.
    crate::accel::route::write_accel_route(py, adata_pred, "energy_distance", &info)?;
    Ok(result.correlation)
}

/// Compute energy distance with per-perturbation details.
///
/// Same inputs, distance kernel, and `nan` behavior as `energy_distance` (see
/// that function for the full argument reference), but returns the full
/// per-perturbation e-distance vectors alongside the summary Pearson
/// correlation.
///
/// The two scalar conventions differ — keep them straight:
///   - `"correlation"` is the score returned by `energy_distance`: a Pearson
///     correlation in [-1.0, 1.0] where **1.0 = perfect, higher = better**
///     (`nan` if either side's e-distance vector is constant / zero-variance).
///   - `"d_real"` / `"d_pred"` are raw **e-distances**: 0 = identical to
///     control, larger = more perturbed (lower is *not* better — they are the
///     per-perturbation effect magnitudes being compared, not a score).
///
/// Returns:
///     dict with keys:
///     - `"correlation"`: float — Pearson correlation of real vs pred e-distances
///       (1.0 = perfect, higher = better; `nan` on a constant predictor)
///     - `"d_real"`: dict[str, float] — per-perturbation e-distance on real side
///     - `"d_pred"`: dict[str, float] — per-perturbation e-distance on pred side
///     - `"pert_names"`: list[str] — perturbation names in order
///
/// Example:
///     out = pyscx.accel.energy_distance_details(adata_real, adata_pred)
///     # out["correlation"] ≈ 0.85   (higher = better, 1.0 = perfect)
///     # out["d_real"]["drug_A"] == 12.34
#[pyfunction]
#[pyo3(signature = (adata_real, adata_pred, pert_col="perturbation", control="control", metric="euclidean", embed_key=None, backend=None, dtype=None, device="auto"))]
#[allow(clippy::too_many_arguments)]
pub fn energy_distance_details<'py>(
    py: Python<'py>,
    adata_real: &Bound<'py, PyAny>,
    adata_pred: &Bound<'py, PyAny>,
    pert_col: &str,
    control: &str,
    metric: &str,
    embed_key: Option<&str>,
    backend: Option<&str>,
    dtype: Option<&str>,
    device: &str,
) -> PyResult<Py<PyAny>> {
    // Rebuild an AnnData view as actual before the route stamp / result
    // writes below, so a backed X is not gathered by anndata's
    // copy-on-write. Gene-order agnostic, so no var-order guard.
    crate::accel::prepare_target_no_var_guard(py, adata_pred, "energy_distance_details")?;
    let (gpu_dev, info) =
        edist_device_dispatch(py, "energy_distance_details", device, metric, dtype)?;
    let result = run_energy_distance(
        py, adata_real, adata_pred, pert_col, control, metric, embed_key, backend, dtype, &gpu_dev,
    )?;
    crate::accel::route::write_accel_route(py, adata_pred, "energy_distance_details", &info)?;

    let d_real = PyDict::new(py);
    let d_pred = PyDict::new(py);
    for (i, name) in result.pert_names.iter().enumerate() {
        d_real.set_item(name.as_str(), result.d_real[i])?;
        d_pred.set_item(name.as_str(), result.d_pred[i])?;
    }

    let out = PyDict::new(py);
    out.set_item("correlation", result.correlation)?;
    out.set_item("d_real", d_real)?;
    out.set_item("d_pred", d_pred)?;
    out.set_item(
        "pert_names",
        pyo3::types::PyList::new(py, &result.pert_names)?,
    )?;

    Ok(out.into_any().unbind())
}

// ──────────────────────────────────────────────────────────────────────────────
// discrimination_score
// ──────────────────────────────────────────────────────────────────────────────
