//! PCA bindings — randomized and covariance PCA, streaming + in-memory.

use scx_format::ShardSource;

use numpy::PyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

use super::gpu::resolve_device;

/// Single-shard `ShardSource` adapter wrapping a borrowed `ScxCsr`.
///
/// Used on the GPU path when the AnnData's `X` is a materialized
/// scipy sparse / dense matrix that has been extracted into an in-memory
/// `ScxCsr`. Parallels the CPU-path's `*_inmemory` entry points by presenting
/// the whole CSR as a single shard to the streaming GPU code.
#[cfg(feature = "gpu")]
struct ScxCsrSource<'a> {
    csr: &'a scx_sparse::ScxCsr,
}

#[cfg(feature = "gpu")]
impl ShardSource for ScxCsrSource<'_> {
    fn n_shards(&self) -> usize {
        1
    }
    fn n_obs(&self) -> usize {
        self.csr.n_rows()
    }
    fn n_vars(&self) -> usize {
        self.csr.n_cols()
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format::Result<scx_sparse::ScxCsr> {
        if shard_idx != 0 {
            return Err(scx_format::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: 1,
            });
        }
        Ok(self.csr.clone())
    }
    fn max_shard_rows(&self) -> scx_format::Result<usize> {
        Ok(self.csr.n_rows())
    }
}

/// Pick the GPU PCA method: explicit user override, or auto-route by `n_vars`.
///
/// Returns `"covariance"` or `"randomized"`. `method == "auto"` dispatches to
/// the covariance path when `n_vars <= GPU_COVARIANCE_PCA_THRESHOLD` and the
/// randomized path otherwise.
#[cfg(feature = "gpu")]
fn resolve_gpu_method(method: &str, n_vars: usize) -> PyResult<&'static str> {
    match method {
        "auto" => {
            if n_vars <= scx_accel::GPU_COVARIANCE_PCA_THRESHOLD {
                Ok("covariance")
            } else {
                Ok("randomized")
            }
        }
        "covariance" => Ok("covariance"),
        "randomized" => Ok("randomized"),
        other => Err(PyValueError::new_err(format!(
            "Invalid method={other:?}; expected 'auto', 'covariance', or 'randomized'"
        ))),
    }
}

/// Parse a `qr_method` string into a [`scx_gpu::QrMethod`]. Both values are
/// valid now (Phase 4 wired up CholeskyQR2 behind `"cholesky"`).
#[cfg(feature = "gpu")]
fn parse_qr_method(qr_method: &str) -> PyResult<scx_accel::QrMethod> {
    match qr_method {
        "householder" => Ok(scx_accel::QrMethod::Householder),
        "cholesky" => Ok(scx_accel::QrMethod::Cholesky),
        other => Err(PyValueError::new_err(format!(
            "Invalid qr_method={other:?}; expected 'householder' or 'cholesky'"
        ))),
    }
}

/// Dispatch to either `covariance_pca_gpu` or `randomized_pca_gpu` based on
/// the resolved method. All GPU-branch call-sites funnel through this helper.
///
/// `qr_method` is threaded to `randomized_pca_gpu` (Householder default,
/// Cholesky opt-in) and **silently ignored** on the covariance path — the
/// Python docstring already documents that covariance PCA has no QR step.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn gpu_pca_dispatch<S: ShardSource + Sync>(
    source: &S,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    method: &str,
    qr_method: scx_accel::QrMethod,
) -> Result<scx_accel::PcaResult, scx_accel::AccelError> {
    match method {
        "covariance" => scx_accel::covariance_pca_gpu(0, source, n_comps, zero_center),
        "randomized" => scx_accel::randomized_pca_gpu(
            0,
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            qr_method,
        ),
        // Unreachable after resolve_gpu_method() normalisation.
        _ => Err(scx_accel::AccelError::LinAlg(format!(
            "internal: unknown resolved method {method:?}"
        ))),
    }
}

/// Run randomized PCA on an AnnData whose X is backed by SCX.
///
/// Results are written to `adata.obsm["X_pca"]`, `adata.varm["PCs"]`,
/// and `adata.uns["pca"]`, matching scanpy's output format.
///
/// Args:
///     adata: AnnData object with X as ScxBackedSparseDataset (or materialized)
///     n_comps: Number of principal components (default: 50)
///     zero_center: Whether to mean-center data (default: True)
///     random_state: Random seed for reproducibility (default: 0)
///     n_oversamples: Extra dimensions for accuracy (default: 10)
///     n_power_iterations: Power iterations for spectral accuracy (default: 2)
///     device: Device selection — "auto" (default), "cpu", or "gpu"
///     method: PCA method — "auto" (default), "covariance", or "randomized".
///         "auto" chooses covariance for n_vars <= GPU_COVARIANCE_PCA_THRESHOLD
///         (8000) on GPU, and n_vars <= COVARIANCE_PCA_THRESHOLD (5000) on CPU.
///     qr_method: QR algorithm for randomized PCA — "householder" (default)
///         or "cholesky". `cholesky` selects CholeskyQR2 (Phase 4, not yet
///         implemented). Ignored for method="covariance".
///
/// Note: GPU mode uses f32 precision throughout (CPU uses f64 intermediates),
/// producing slightly different but equally valid results. See docs/scanpy.md.
#[pyfunction]
#[pyo3(signature = (adata, n_comps=50, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder"))]
#[allow(clippy::too_many_arguments)]
pub fn pca(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    n_comps: usize,
    zero_center: bool,
    random_state: u64,
    n_oversamples: usize,
    n_power_iterations: usize,
    device: &str,
    method: &str,
    qr_method: &str,
) -> PyResult<()> {
    let _use_gpu = resolve_device(device)?;
    // Validate user args even on CPU path — catches typos regardless of device.
    if !matches!(method, "auto" | "covariance" | "randomized") {
        return Err(PyValueError::new_err(format!(
            "Invalid method={method:?}; expected 'auto', 'covariance', or 'randomized'"
        )));
    }
    if !matches!(qr_method, "householder" | "cholesky") {
        return Err(PyValueError::new_err(format!(
            "Invalid qr_method={qr_method:?}; expected 'householder' or 'cholesky'"
        )));
    }
    let backend: &str;

    // Extract X from adata
    let x = adata.getattr("X")?;

    // ------- GPU path -------
    #[cfg(feature = "gpu")]
    if _use_gpu {
        let qr = parse_qr_method(qr_method)?;

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            let reader = &*backed.backed;
            let (_n_obs, n_vars) = reader.shape();
            let m = resolve_gpu_method(method, n_vars)?;
            let result = gpu_pca_dispatch(
                reader,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
                m,
                qr,
            )
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse")?;
            return Ok(());
        }

        if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
            let source = lazy.as_shard_source();
            let (_n_obs, n_vars) = source.shape();
            let m = resolve_gpu_method(method, n_vars)?;
            let result = gpu_pca_dispatch(
                &source,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
                m,
                qr,
            )
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse")?;
            return Ok(());
        }

        // Materialized scipy/dense → ScxCsr → ScxCsrSource (single-shard).
        let csr = extract_materialized_csr(py, &x)?;
        let source = ScxCsrSource { csr: &csr };
        let n_vars = source.n_vars();
        let m = resolve_gpu_method(method, n_vars)?;
        let result = gpu_pca_dispatch(
            &source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            m,
            qr,
        )
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
        write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse")?;
        return Ok(());
    }

    // ------- CPU path (fallback) -------
    // Auto-route: covariance when n_vars <= threshold (faster for HVG data).
    let cov_threshold = scx_accel::COVARIANCE_PCA_THRESHOLD;
    let pick_cpu_method = |n_vars: usize| -> &'static str {
        match method {
            "covariance" => "covariance",
            "randomized" => "randomized",
            _ => {
                if n_vars <= cov_threshold {
                    "covariance"
                } else {
                    "randomized"
                }
            }
        }
    };

    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        backend = "scx-accel-cpu";
        let reader = &*backed.backed;
        let (_n_obs, n_vars) = reader.shape();
        match pick_cpu_method(n_vars) {
            "covariance" => scx_accel::covariance_pca(reader, n_comps, zero_center),
            _ => scx_accel::randomized_pca(
                reader,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            ),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        backend = "scx-accel-cpu";
        let source = lazy.as_shard_source();
        let (_n_obs, n_vars) = source.shape();
        match pick_cpu_method(n_vars) {
            "covariance" => scx_accel::covariance_pca(&source, n_comps, zero_center),
            _ => scx_accel::randomized_pca(
                &source,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            ),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backend = "scx-accel-cpu";
        let csr = extract_materialized_csr(py, &x)?;
        let n_vars = csr.n_cols();
        match pick_cpu_method(n_vars) {
            "covariance" => scx_accel::covariance_pca_inmemory(&csr, n_comps, zero_center),
            _ => scx_accel::randomized_pca_inmemory(
                &csr,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            ),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    };

    write_pca_to_adata(py, adata, &result, backend)?;

    Ok(())
}

/// Extract a materialized scipy sparse or dense `X` into an in-memory
/// [`ScxCsr`]. Handles both `scipy.sparse.*` and dense numpy arrays.
fn extract_materialized_csr(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<scx_sparse::ScxCsr> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let is_sparse = scipy_sparse
        .call_method1("issparse", (x,))?
        .extract::<bool>()?;

    let csr_py = scipy_sparse.call_method1("csr_matrix", (x,))?;
    let shape: (usize, usize) = csr_py.getattr("shape")?.extract()?;

    let (indptr, indices, data): (Vec<i64>, Vec<i32>, Vec<f32>) = if is_sparse {
        let np = py.import("numpy")?;
        let indptr_np = csr_py.getattr("indptr")?;
        let indices_np = csr_py.getattr("indices")?;
        let data_np = csr_py.getattr("data")?;
        let indptr = np
            .call_method1("asarray", (&indptr_np,))?
            .call_method1("astype", ("int64",))?
            .extract::<Vec<i64>>()?;
        let indices = np
            .call_method1("asarray", (&indices_np,))?
            .call_method1("astype", ("int32",))?
            .extract::<Vec<i32>>()?;
        let data = np
            .call_method1("asarray", (&data_np,))?
            .call_method1("astype", ("float32",))?
            .extract::<Vec<f32>>()?;
        (indptr, indices, data)
    } else {
        let indptr = csr_py
            .getattr("indptr")?
            .call_method1("astype", ("int64",))?
            .extract::<Vec<i64>>()?;
        let indices = csr_py
            .getattr("indices")?
            .call_method1("astype", ("int32",))?
            .extract::<Vec<i32>>()?;
        let data = csr_py
            .getattr("data")?
            .call_method1("astype", ("float32",))?
            .extract::<Vec<f32>>()?;
        (indptr, indices, data)
    };

    Ok(scx_sparse::ScxCsr::new_unchecked(
        shape, indptr, indices, data,
    ))
}

/// Write PCA results to AnnData slots matching scanpy's format.
fn write_pca_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::PcaResult,
    backend: &str,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // adata.obsm["X_pca"] = embeddings (n_obs × n_components) as float32
    let embeddings_arr = PyArray2::<f32>::from_vec2(
        py,
        &(0..result.n_obs)
            .map(|i| {
                (0..result.n_components)
                    .map(|j| result.embeddings[i * result.n_components + j] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
    )?;
    let obsm = adata.getattr("obsm")?;
    obsm.set_item("X_pca", embeddings_arr)?;

    // adata.varm["PCs"] = components transposed to (n_vars × n_components) as float32
    let pcs_arr = PyArray2::<f32>::from_vec2(
        py,
        &(0..result.n_vars)
            .map(|v| {
                (0..result.n_components)
                    .map(|pc| result.components[pc * result.n_vars + v] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>(),
    )?;
    let varm = adata.getattr("varm")?;
    varm.set_item("PCs", pcs_arr)?;

    // adata.uns["pca"] = dict with variance info + backend
    let pca_dict = PyDict::new(py);

    let var_explained = numpy.call_method1("array", (result.variance_explained.clone(),))?;
    pca_dict.set_item("variance", var_explained)?;

    let var_ratio = numpy.call_method1("array", (result.variance_ratio.clone(),))?;
    pca_dict.set_item("variance_ratio", var_ratio)?;

    pca_dict.set_item("backend", backend)?;

    let uns = adata.getattr("uns")?;
    uns.set_item("pca", pca_dict)?;

    Ok(())
}
