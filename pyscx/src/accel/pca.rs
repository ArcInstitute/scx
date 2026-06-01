//! PCA bindings — randomized and covariance PCA, streaming + in-memory.

use scx_format::ShardSource;

use numpy::PyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

use super::gpu::resolve_device;
use super::util::extract_materialized_csr;
#[cfg(feature = "gpu")]
use super::util::{extract_csr_slices, CsrSlices};

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

/// Single-shard `ShardSource` adapter over **borrowed** numpy slices
/// (Phase 10).
///
/// `extract_materialized_csr` + [`ScxCsrSource`] performs two full host
/// copies of `indptr` / `indices` / `data`: one when `extract::<Vec<T>>`
/// converts numpy → Rust `Vec`, and a second when `read_shard` clones
/// the resulting `ScxCsr` for dispatch. For 1M × 2K HVGs that doubles
/// host RSS during upload.
///
/// `BorrowedCsrSource` skips the first copy by holding `&[T]` views into
/// numpy buffers (kept alive by the [`CsrSlices`] handle held by the
/// caller). `read_shard` is invoked exactly once by the GPU shard loader
/// for a single-shard source, so `slice.to_vec()` here replaces *both* the
/// original `extract::<Vec>`
/// step and the `ScxCsrSource::read_shard` clone — net one memcpy per
/// dispatch instead of two.
#[cfg(feature = "gpu")]
struct BorrowedCsrSource<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
    data: &'a [f32],
    shape: (usize, usize),
}

#[cfg(feature = "gpu")]
impl ShardSource for BorrowedCsrSource<'_> {
    fn n_shards(&self) -> usize {
        1
    }
    fn n_obs(&self) -> usize {
        self.shape.0
    }
    fn n_vars(&self) -> usize {
        self.shape.1
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format::Result<scx_sparse::ScxCsr> {
        if shard_idx != 0 {
            return Err(scx_format::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: 1,
            });
        }
        Ok(scx_sparse::ScxCsr::new_unchecked(
            self.shape,
            self.indptr.to_vec(),
            self.indices.to_vec(),
            self.data.to_vec(),
        ))
    }
    fn max_shard_rows(&self) -> scx_format::Result<usize> {
        Ok(self.shape.0)
    }
}

/// Try to obtain borrowed `&[T]` views over a materialised scipy / dense `X`
/// for the GPU PCA fast-lane.
///
/// Returns `Ok(Some(...))` on success — caller holds the [`CsrSlices`] for
/// the duration of dispatch so the underlying numpy buffers stay alive.
/// Returns `Ok(None)` if `X` cannot be presented as a scipy CSR (e.g. a type
/// scipy refuses to convert); callers fall through to the owned-`Vec` path.
#[cfg(feature = "gpu")]
fn try_extract_borrowed_csr<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
) -> PyResult<Option<(CsrSlices<'py>, (usize, usize))>> {
    let scipy_sparse = py.import("scipy.sparse")?;
    let csr_py = match scipy_sparse.call_method1("csr_matrix", (x,)) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let shape: (usize, usize) = csr_py.getattr("shape")?.extract()?;
    let np = py.import("numpy")?;
    let slices = extract_csr_slices(py, &np, &csr_py, "pca(device=\"gpu\")", 0)?;
    Ok(Some((slices, shape)))
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
    device_id: usize,
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
        "covariance" => scx_accel::covariance_pca_gpu(device_id, source, n_comps, zero_center),
        "randomized" => scx_accel::randomized_pca_gpu(
            device_id,
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

/// Catch any cudarc dlsym / FFI panic that escapes `gpu_pca_dispatch` and
/// translate it to a normal `AccelError::LinAlg`. The proactive
/// `cusparse_modern_abi_available()` probe at the GPU branch entry handles the
/// *known* `cusparseBsrSetStridedBatch`-missing failure; this wrapper is a
/// backstop for any other future cudarc symbol surprise so users never see a
/// raw `pyo3_runtime.PanicException`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
fn gpu_pca_dispatch_unwind_safe<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    method: &str,
    qr_method: scx_accel::QrMethod,
) -> Result<scx_accel::PcaResult, scx_accel::AccelError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        gpu_pca_dispatch(
            device_id,
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            method,
            qr_method,
        )
    }))
    .unwrap_or_else(|panic_payload| {
        let msg = panic_payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| {
                panic_payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
            })
            .unwrap_or_else(|| "unknown panic payload".to_string());
        Err(scx_accel::AccelError::LinAlg(format!(
            "GPU PCA panicked: {msg}. If this mentions libcusparse / cusparse* \
             undefined symbol, set \
             LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH so the \
             toolkit's cuSPARSE 12.5+ wins over the system 12.0 — see \
             docs/gpu-setup.md."
        )))
    })
}

/// Emit a one-shot `UserWarning` when the runtime libcusparse predates
/// cuSPARSE 12.5 and cudarc's `cusparseBsrSetStridedBatch` symbol probe fails.
/// The PCA dispatcher then falls through to the CPU path instead of letting
/// cudarc's lazy `dlsym` panic deep in an FFI call.
#[cfg(feature = "gpu")]
fn emit_cusparse_abi_warning(py: Python<'_>, device: &str) -> PyResult<()> {
    let warnings = py.import("warnings")?;
    warnings.call_method1(
        "warn",
        (
            format!(
                "pyscx.accel.pca(device={device:?}) falling back to CPU: the runtime \
                 libcusparse.so is older than cuSPARSE 12.5 and lacks the \
                 cusparseBsrSetStridedBatch symbol that cudarc 0.19+ requires. \
                 Ubuntu's libcusparse-dev is typically 12.0.1.140 (2023-01); the \
                 CUDA Toolkit at /usr/local/cuda*/lib64 ships 12.5+. Fix: export \
                 LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH (or the \
                 matching toolkit path) before invoking Python. See \
                 docs/gpu-setup.md for details."
            ),
            py.get_type::<pyo3::exceptions::PyUserWarning>(),
        ),
    )?;
    Ok(())
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
///     device: Device selection — "auto" (default), "cpu", "gpu", or
///         "gpu:N" to target CUDA device N on multi-GPU systems.
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
#[pyo3(signature = (adata, n_comps=50, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", prefer_format="csr"))]
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
    prefer_format: &str,
) -> PyResult<()> {
    let _device = resolve_device(device)?;
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
    // CSC dispatch is intentionally not implemented for PCA: the
    // covariance build (`X^T @ X`) and randomized SpMM both consume
    // shards in row-major order, where CSC offers no measurable
    // speedup over CSR. Reject `prefer_format="csc"` explicitly so
    // callers don't silently fall back and get confused.
    if prefer_format != "csr" {
        return Err(PyValueError::new_err(format!(
            "pyscx.accel.pca only supports prefer_format='csr' \
             (got {prefer_format:?}). The covariance build and \
             randomized SpMM paths are row-major; CSC offers no \
             measurable speed-up and is not implemented."
        )));
    }
    let backend: &str;

    // Extract X from adata
    let x = adata.getattr("X")?;

    // Record the planned route on adata.uns["scx_accel"]["pca"]. PCA has a
    // single GPU route (cuSPARSE + cuBLAS) gated on the modern cuSPARSE ABI;
    // when that probe fails the dispatch falls back to CPU, which the planner
    // records as UnsupportedInputLayout (vs NoCuda when CUDA is simply absent).
    // The covariance-vs-randomized choice is orthogonal math policy and stays in
    // adata.uns["pca"]["backend"], not the route string.
    #[cfg(feature = "gpu")]
    let pca_gpu_eligible = scx_accel::cusparse_modern_abi_available();
    #[cfg(not(feature = "gpu"))]
    let pca_gpu_eligible = false;
    super::route::write_accel_route(
        py,
        adata,
        "pca",
        &super::route::simple_exec_info(
            device,
            pca_gpu_eligible,
            scx_accel::AccelRoute::GpuCsrV1,
            scx_accel::AccelRoute::CpuCsr,
        ),
    )?;

    // ------- GPU path -------
    // Probe libcusparse for the cuSPARSE 12.5+ ABI before dispatching, so an
    // Ubuntu host running with the system libcusparse-dev (12.0.1.140) gets
    // a graceful CPU fallback + actionable UserWarning instead of a deep
    // cudarc dlsym panic. See `scx_gpu::cusparse_modern_abi_available`.
    #[cfg(feature = "gpu")]
    let gpu_device_id = match _device.gpu_id() {
        Some(id) if scx_accel::cusparse_modern_abi_available() => Some(id),
        Some(_) => {
            emit_cusparse_abi_warning(py, device)?;
            None
        }
        None => None,
    };
    #[cfg(feature = "gpu")]
    if let Some(device_id) = gpu_device_id {
        let qr = parse_qr_method(qr_method)?;

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            let reader = &*backed.backed;
            let (_n_obs, n_vars) = reader.shape();
            let m = resolve_gpu_method(method, n_vars)?;
            let result = gpu_pca_dispatch_unwind_safe(
                device_id,
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
            let result = gpu_pca_dispatch_unwind_safe(
                device_id,
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

        // Materialised scipy / dense X. Phase 10 fast-lane: borrow numpy
        // buffers via PyReadonlyArray1 instead of copying them into owned
        // `Vec`s (halves the host RSS spike during dispatch). Falls back to
        // the owned-Vec path if scipy cannot produce a CSR view.
        if let Some((slices, shape)) = try_extract_borrowed_csr(py, &x)? {
            let source = BorrowedCsrSource {
                indptr: slices.indptr(),
                indices: slices.indices(),
                data: slices.data(),
                shape,
            };
            let n_vars = source.n_vars();
            let m = resolve_gpu_method(method, n_vars)?;
            let result = gpu_pca_dispatch_unwind_safe(
                device_id,
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

        // Fallback: owned-Vec path (e.g. exotic X types scipy can't view).
        let csr = extract_materialized_csr(py, &x)?;
        let source = ScxCsrSource { csr: &csr };
        let n_vars = source.n_vars();
        let m = resolve_gpu_method(method, n_vars)?;
        let result = gpu_pca_dispatch_unwind_safe(
            device_id,
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
        let reader = std::sync::Arc::clone(&backed.backed);
        let (_n_obs, n_vars) = reader.shape();
        drop(backed);
        let m = pick_cpu_method(n_vars);
        py.detach(|| match m {
            "covariance" => scx_accel::covariance_pca(&*reader, n_comps, zero_center),
            _ => scx_accel::randomized_pca(
                &*reader,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            ),
        })
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        backend = "scx-accel-cpu";
        let source = lazy.as_shard_source();
        let (_n_obs, n_vars) = source.shape();
        drop(lazy);
        let m = pick_cpu_method(n_vars);
        py.detach(|| match m {
            "covariance" => scx_accel::covariance_pca(&source, n_comps, zero_center),
            _ => scx_accel::randomized_pca(
                &source,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            ),
        })
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backend = "scx-accel-cpu";
        let csr = extract_materialized_csr(py, &x)?;
        let n_vars = csr.n_cols();
        let m = pick_cpu_method(n_vars);
        py.detach(|| match m {
            "covariance" => scx_accel::covariance_pca_inmemory(&csr, n_comps, zero_center),
            _ => scx_accel::randomized_pca_inmemory(
                &csr,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
            ),
        })
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    };

    write_pca_to_adata(py, adata, &result, backend)?;

    Ok(())
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
