//! Perturbation evaluation metrics — pseudobulk means, perturbation_metrics,
//! energy_distance, discrimination_score, knockdown_efficiency, clustering_agreement.
//!
//! Split into result-family submodules (ORG-10.16-6): `pseudobulk` (grouped
//! mean aggregation), `energy` (e-distance kernels), `scores`
//! (perturbation/discrimination/knockdown effect scores), `clustering`
//! (clustering agreement + label metrics). This module keeps the shared
//! device scaffolding and array marshalling, and re-exports every
//! `#[pyfunction]` so `lib.rs`'s `accel::eval_metrics::<fn>` registration
//! paths are unchanged.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

// ──────────────────────────────────────────────────────────────────────────────
// device scaffolding (Phase 0 of CELL-EVAL-SCX-GPU-ACC.md)
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) mod clustering;
pub(crate) mod energy;
pub(crate) mod pseudobulk;
pub(crate) mod scores;

pub(crate) use clustering::*;
pub(crate) use energy::*;
pub(crate) use pseudobulk::*;
pub(crate) use scores::*;

/// Validate the `device=` selector and record the resolved route on
/// `adata.uns["scx_accel"][op]` for a CPU-only eval metric.
///
/// None of the perturbation-evaluation metrics has a GPU kernel yet (the GPU
/// kernels land in later phases of `CELL-EVAL-SCX-GPU-ACC.md`), so this is the
/// no-op API surface that lets callers pass `device=` today and observe the
/// route via `uns["scx_accel"]`:
///
/// * [`resolve_device`](super::gpu::resolve_device) validates the string, so an
///   explicit `device="gpu"`/`"gpu:N"` errors loudly when pyscx was built
///   without the `gpu` feature or no CUDA device is visible. `"auto"`/`"cpu"`
///   always resolve to CPU here.
/// * The route is planned via [`cpu_only_exec_info`](super::route::cpu_only_exec_info):
///   `UserForcedCpu` for `device="cpu"`, `NoCuda` when no GPU is present, and
///   `UnsupportedInputLayout` for an explicit GPU request on a GPU host (there
///   is no GPU eval-metric kernel yet). An explicit `device="gpu"` landing on
///   CPU emits the standard one-shot [`announce_route`](super::route::announce_route)
///   `UserWarning`; `"auto"` stays quiet.
///
/// Stamp on the AnnData the caller would inspect for the route — the prediction
/// (`adata_pred`) for the pair metrics, the sole input for single-input ops.
///
/// Returns the [`RouteStamp`](super::route::RouteStamp) guard: these ops stamp
/// before their compute, so the caller must `commit()` on success or the stamp
/// is rolled back — `uns["scx_accel"][op]` is present iff the op completed.
#[must_use = "commit the returned RouteStamp on success, or the route stamp is rolled back"]
pub(crate) fn scaffold_device_route<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    op: &'static str,
    device: &str,
) -> PyResult<super::route::RouteStamp<'py>> {
    super::gpu::resolve_device(device)?;
    let info = super::route::cpu_only_exec_info(device);
    super::route::announce_route(py, op, device, &info);
    super::route::RouteStamp::write(py, adata, op, &info)
}

// ──────────────────────────────────────────────────────────────────────────────
// GPU dispatch for the pseudobulk aggregation.
//
// `perturbation_metrics` / `pseudobulk_means` GPU-accelerate the pseudobulk
// aggregation (the cell-count-scaling step) via `scx_accel::pseudobulk_means_gpu_*`,
// which wrap the DE pseudobulk kernels. The resulting `[P×G]` f64 means then flow
// through the unchanged host alignment + `compute_bulk_metrics`. The GPU device
// handle is `!Send`, so the GPU folds hold the GIL (they run on-device); the CPU
// fallbacks release it via `py.detach`.
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
pub(crate) type EvalGpuDev = Option<scx_accel::GpuDevice>;
#[cfg(not(feature = "gpu"))]
pub(crate) type EvalGpuDev = Option<()>;

/// Build the reusable GPU device handle (`None` → CPU). Errors only on CUDA
/// context-creation failure.
pub(crate) fn eval_make_gpu_dev(gpu_device_id: Option<usize>) -> PyResult<EvalGpuDev> {
    #[cfg(feature = "gpu")]
    {
        match gpu_device_id {
            Some(id) => Ok(Some(scx_accel::GpuDevice::new(id).map_err(|e| {
                PyRuntimeError::new_err(format!("GPU eval-metrics device init failed: {e}"))
            })?)),
            None => Ok(None),
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = gpu_device_id;
        Ok(None)
    }
}

/// Resolve device + plan the route for `energy_distance` / `energy_distance_details`.
///
/// GPU `energy_distance` is gemm-based (f32) and covers **euclidean/cosine only**;
/// L1 (no gemm decomposition) and f64 stay on CPU. So the op is GPU-eligible only
/// when built with the gpu feature AND the metric isn't L1 AND the dtype is f32
/// (the default). Returns the (possibly `None`) device handle and the planned
/// `AccelExecutionInfo`; the caller stamps it via `write_accel_route` after
/// compute. Route pair: `gpu_dense` / `cpu_csr` (cells materialise dense).
pub(crate) fn edist_device_dispatch(
    py: Python<'_>,
    op: &'static str,
    device: &str,
    metric: &str,
    dtype: Option<&str>,
) -> PyResult<(EvalGpuDev, scx_accel::route::AccelExecutionInfo)> {
    let resolved = super::gpu::resolve_device(device)?;
    let metric_is_l1 = matches!(
        metric.to_ascii_lowercase().as_str(),
        "l1" | "manhattan" | "cityblock"
    );
    let dtype_is_f32 = matches!(
        dtype.map(|s| s.to_ascii_lowercase()).as_deref(),
        None | Some("f32") | Some("float32")
    );
    let gpu_eligible = cfg!(feature = "gpu") && !metric_is_l1 && dtype_is_f32;
    let info = super::route::simple_exec_info(
        device,
        gpu_eligible,
        scx_accel::route::AccelRoute::GpuDense,
        scx_accel::route::AccelRoute::CpuCsr,
    );
    super::route::announce_route(py, op, device, &info);
    let gpu_id = if info.route.is_gpu() {
        resolved.gpu_id()
    } else {
        None
    };
    let gpu_dev = eval_make_gpu_dev(gpu_id)?;
    Ok((gpu_dev, info))
}

/// Trait bridging Rust `f32`/`f64` to numpy dtype strings for
/// `materialize_dense`. Sealed in spirit (only impls in this module).
pub(crate) trait NumpyDtype: numpy::Element + Sized {
    /// numpy `dtype` string passed to `.astype(...)`.
    const NUMPY_NAME: &'static str;
    /// Bytes per element — used to gate the "large materialization" warning.
    const BYTES: usize;

    /// GPU energy distance (Phase 3), specialised by element type: `Some(..)`
    /// for f32 (the GPU path runs), `None` for f64 (no f64 cuBLAS gemm → caller
    /// falls back to the CPU path). Lets the generic `run_energy_distance_inner`
    /// dispatch to the GPU kernel only for f32 without runtime type juggling.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments)]
    fn edist_gpu(
        dev: &scx_accel::GpuDevice,
        real: &[Self],
        pred: &[Self],
        real_groups: &[u32],
        pred_groups: &[u32],
        ctrl_group_idx: u32,
        pert_names: &[String],
        pert_group_indices: &[u32],
        n_dims: usize,
        metric: scx_accel::DistanceMetric,
    ) -> Option<scx_accel::Result<scx_accel::EDistanceResult>>;
}

impl NumpyDtype for f32 {
    const NUMPY_NAME: &'static str = "float32";
    const BYTES: usize = 4;

    #[cfg(feature = "gpu")]
    fn edist_gpu(
        dev: &scx_accel::GpuDevice,
        real: &[f32],
        pred: &[f32],
        real_groups: &[u32],
        pred_groups: &[u32],
        ctrl_group_idx: u32,
        pert_names: &[String],
        pert_group_indices: &[u32],
        n_dims: usize,
        metric: scx_accel::DistanceMetric,
    ) -> Option<scx_accel::Result<scx_accel::EDistanceResult>> {
        Some(scx_accel::compute_energy_distance_gpu(
            dev,
            real,
            pred,
            real_groups,
            pred_groups,
            ctrl_group_idx,
            pert_names,
            pert_group_indices,
            n_dims,
            metric,
        ))
    }
}

impl NumpyDtype for f64 {
    const NUMPY_NAME: &'static str = "float64";
    const BYTES: usize = 8;

    #[cfg(feature = "gpu")]
    fn edist_gpu(
        _dev: &scx_accel::GpuDevice,
        _real: &[f64],
        _pred: &[f64],
        _real_groups: &[u32],
        _pred_groups: &[u32],
        _ctrl_group_idx: u32,
        _pert_names: &[String],
        _pert_group_indices: &[u32],
        _n_dims: usize,
        _metric: scx_accel::DistanceMetric,
    ) -> Option<scx_accel::Result<scx_accel::EDistanceResult>> {
        None // f64 has no cuBLAS gemm → CPU fallback
    }
}

/// Extract a dense `[N, D]` matrix from `adata.X` (or `adata.obsm[embed_key]`)
/// as a flat `Vec<F>`, narrowing to F via numpy's `.astype(...)`.
///
/// Handles scipy sparse (converts to dense), numpy arrays, and SCX backed types.
/// Emits a Python warning for large materializations (> 2 GB estimated).
///
/// Generic over `F: PairwiseFloat + NumpyDtype` — the dtype string passed to
/// `.astype(...)` is inferred from `F::NUMPY_NAME`. This stops the historical
/// unconditional upcast to f64 and lets f32-native callers stay in f32.
pub(crate) fn materialize_dense<'py, F>(
    py: Python<'py>,
    np: &Bound<'py, PyModule>,
    adata: &Bound<'py, PyAny>,
    embed_key: Option<&str>,
) -> PyResult<(Vec<F>, usize, usize)>
where
    F: scx_accel::PairwiseFloat + NumpyDtype,
{
    let matrix_obj = if let Some(key) = embed_key {
        let obsm = adata.getattr("obsm")?;
        obsm.get_item(key).map_err(|_| {
            PyValueError::new_err(format!("embed_key '{}' not found in adata.obsm", key))
        })?
    } else {
        adata.getattr("X")?
    };

    // Check if it's a SCX backed or lazy-transformed type — must materialize.
    let dense = if matrix_obj.is_instance_of::<ScxBackedSparseDataset>()
        || matrix_obj.is_instance_of::<ScxLazyTransformedDataset>()
    {
        let arr = matrix_obj.call_method0("toarray")?;
        arr.call_method1("astype", (F::NUMPY_NAME,))?
    } else {
        // Check for scipy sparse.
        let scipy_sparse = crate::pyimport::import_module(py, "scipy.sparse")?;
        let is_sparse: bool = scipy_sparse
            .call_method1("issparse", (&matrix_obj,))?
            .extract()?;

        if is_sparse {
            let arr = matrix_obj.call_method0("toarray")?;
            arr.call_method1("astype", (F::NUMPY_NAME,))?
        } else {
            np.call_method1("asarray", (&matrix_obj,))?
                .call_method1("astype", (F::NUMPY_NAME,))?
        }
    };

    let shape: (usize, usize) = dense.getattr("shape")?.extract()?;

    // Warn if the dense matrix is very large (> 2 GB).
    let estimated_bytes = shape.0 * shape.1 * F::BYTES;
    if estimated_bytes > 2_000_000_000 {
        let gb = estimated_bytes as f64 / 1e9;
        let warnings = crate::pyimport::import_module(py, "warnings")?;
        warnings.call_method1(
            "warn",
            (format!(
                "energy_distance: materializing a {:.1} GB dense matrix ({} × {} × {} bytes). \
                 Consider using embed_key='X_pca' or subsetting the data.",
                gb,
                shape.0,
                shape.1,
                F::BYTES,
            ),),
        )?;
    }

    // Use numpy's typed array protocol to extract a Vec<F> without going
    // through pyo3's `extract::<Vec<F>>` (which has no generic impl).
    let raveled = dense.call_method0("ravel")?;
    let arr: numpy::PyReadonlyArray1<F> = raveled.extract()?;
    let flat: Vec<F> = arr.as_slice()?.to_vec();

    Ok((flat, shape.0, shape.1))
}

/// Extract a column from adata.obs as Vec<String>.
///
/// Raises ValueError if the column contains NaN values (which would
/// silently become the string `"nan"` after `.astype(str)`).
pub(crate) fn extract_obs_column<'py>(
    _py: Python<'py>,
    adata: &Bound<'py, PyAny>,
    col: &str,
) -> PyResult<Vec<String>> {
    let obs = adata.getattr("obs")?;
    let series = obs
        .get_item(col)
        .map_err(|_| PyValueError::new_err(format!("column '{}' not found in adata.obs", col)))?;

    // Detect NaN values before string conversion (NaN → "nan" is silent and wrong).
    let pd = crate::pyimport::import_module(_py, "pandas")?;
    let isna = pd.call_method1("isna", (&series,))?;
    let any_na: bool = isna.call_method0("any")?.extract()?;
    if any_na {
        let n_na: usize = isna.call_method0("sum")?.extract()?;
        return Err(PyValueError::new_err(format!(
            "column '{}' has {} NaN values. Remove or fill NaN before calling.",
            col, n_na
        )));
    }

    let labels: Vec<String> = series
        .call_method1("astype", ("str",))?
        .call_method0("tolist")?
        .extract()?;
    Ok(labels)
}

// ──────────────────────────────────────────────────────────────────────────────
// perturbation_metrics
// ──────────────────────────────────────────────────────────────────────────────
