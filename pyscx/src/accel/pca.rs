//! PCA bindings — randomized and covariance PCA, streaming + in-memory.

use scx_format_io::ShardSource;

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::backed::ScxBackedSparseDataset;
use crate::lazy_transform::ScxLazyTransformedDataset;

use super::gpu::resolve_device;
use super::util::extract_materialized_csr;

/// Default RAM ceiling for the out-of-core backed-PCA shard cache when the
/// caller passes no `memory_budget`. Large enough to hold the working set for
/// typical multi-shard files while bounding growth on a count-only-opened
/// reader (whose byte budget is otherwise unbounded). Raise via `memory_budget`
/// for atlas-scale matrices that exceed this.
const DEFAULT_PCA_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
#[cfg(feature = "gpu")]
use super::util::{extract_csr_slices, CsrSlices};

/// Single-shard `ShardSource` adapter wrapping a borrowed `ScxCsr`.
///
/// Used on the GPU path when the AnnData's `X` is a materialized
/// scipy sparse / dense matrix that has been extracted into an in-memory
/// `ScxCsr`. Parallels the CPU-path's `*_inmemory` entry points by presenting
/// the whole CSR as a single shard to the streaming GPU code. This is the
/// shared `scx_format_io::shard_source::SingleShardSource` (the rscx binding uses
/// the same adapter); aliased here to keep the call sites'
/// `ScxCsrSource { csr }` spelling.
#[cfg(feature = "gpu")]
pub(crate) use scx_format_io::shard_source::SingleShardSource as ScxCsrSource;

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
pub(crate) struct BorrowedCsrSource<'a> {
    pub(crate) indptr: &'a [i64],
    pub(crate) indices: &'a [i32],
    pub(crate) data: &'a [f32],
    pub(crate) shape: (usize, usize),
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
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsr> {
        if shard_idx != 0 {
            return Err(scx_format_io::ScxError::ShardIndexOutOfBounds {
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
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
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
pub(super) fn try_extract_borrowed_csr<'py>(
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

/// Resolve the native GPU PCA method.
///
/// The in-VRAM covariance-PCA kernels were removed: in-VRAM `device="gpu"` PCA
/// now routes to rapids-singlecell, and the only surviving native GPU path is
/// randomized (the streaming / device-resident moat). All accepted `method`
/// values therefore resolve to `"randomized"` here; `method="covariance"` is
/// still honored on the **CPU** path (`covariance_pca`).
#[cfg(feature = "gpu")]
pub(crate) fn resolve_gpu_method(method: &str, _n_vars: usize) -> PyResult<&'static str> {
    match method {
        "auto" | "covariance" | "randomized" => Ok("randomized"),
        other => Err(PyValueError::new_err(format!(
            "Invalid method={other:?}; expected 'auto', 'covariance', or 'randomized'"
        ))),
    }
}

/// Parse a `qr_method` string into a [`scx_gpu::QrMethod`]. Both values are
/// valid now (Phase 4 wired up CholeskyQR2 behind `"cholesky"`).
#[cfg(feature = "gpu")]
pub(crate) fn parse_qr_method(qr_method: &str) -> PyResult<scx_accel::QrMethod> {
    match qr_method {
        "householder" => Ok(scx_accel::QrMethod::Householder),
        "cholesky" => Ok(scx_accel::QrMethod::Cholesky),
        other => Err(PyValueError::new_err(format!(
            "Invalid qr_method={other:?}; expected 'householder' or 'cholesky'"
        ))),
    }
}

/// Build the [`scx_accel::GpuPcaTuning`] for the GPU PCA path from the Python
/// kwargs (Task 2.5). `spmm_policy` is pre-validated by the caller.
#[cfg(feature = "gpu")]
fn build_pca_tuning(allow_tf32: bool, spmm_policy: &str) -> scx_accel::GpuPcaTuning {
    let policy = match spmm_policy {
        "deterministic" => scx_accel::SpmmAlgPolicy::Deterministic,
        "benchmark_once" => scx_accel::SpmmAlgPolicy::BenchmarkOnce,
        _ => scx_accel::SpmmAlgPolicy::Default,
    };
    scx_accel::GpuPcaTuning::new(scx_accel::GpuMathMode::from_allow_tf32(allow_tf32), policy)
}

/// Dispatch native GPU PCA. The in-VRAM covariance path was removed, so
/// `resolve_gpu_method` always yields `"randomized"` and this helper routes to
/// `randomized_pca_gpu`. All GPU-branch call-sites funnel through here.
/// `qr_method` selects the QR step (Householder default, Cholesky opt-in).
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
    tuning: scx_accel::GpuPcaTuning,
) -> Result<scx_accel::PcaResult, scx_accel::AccelError> {
    match method {
        "randomized" => scx_accel::randomized_pca_gpu(
            device_id,
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            qr_method,
            tuning,
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
    tuning: scx_accel::GpuPcaTuning,
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
            tuning,
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
pub(crate) fn emit_cusparse_abi_warning(py: Python<'_>, device: &str) -> PyResult<()> {
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
///         On CPU, "auto" chooses covariance for n_vars <=
///         COVARIANCE_PCA_THRESHOLD (5000), else randomized. On GPU the native
///         in-VRAM covariance core was removed, so every method resolves to
///         randomized (in-memory `device="gpu"` routes
///         to rapids-singlecell, which runs its own covariance/randomized PCA).
///     qr_method: QR algorithm for randomized PCA — "householder" (default)
///         or "cholesky". `cholesky` selects CholeskyQR2. Ignored for
///         method="covariance".
///
/// Note: GPU mode uses f32 precision throughout (CPU uses f64 intermediates),
/// producing slightly different but equally valid results. See docs/scanpy.md.
/// Stable `&'static str` label for a validated `spmm_policy` kwarg.
#[cfg(feature = "gpu")]
fn spmm_policy_label(spmm_policy: &str) -> &'static str {
    match spmm_policy {
        "deterministic" => "deterministic",
        "benchmark_once" => "benchmark_once",
        _ => "default",
    }
}

/// Stamp the PCA route + Task 2.5 tuning metadata on
/// `adata.uns["scx_accel"]["pca"]`. Called once before dispatch (everything
/// `None`) and re-stamped on the GPU branch after dispatch. `math_mode` /
/// `spmm_policy` are passed `Some` only by the route that actually consumes
/// them — the **randomized** GPU path (which runs the cuBLAS math mode and the
/// cuSPARSE SpMM). The **covariance** path passes `None` for both (it applies no
/// SpMM, and does not thread the math mode), so the metadata never claims a knob
/// that wasn't used. All fields stay `None` on a CPU route regardless.
#[allow(clippy::too_many_arguments)]
fn stamp_pca_route(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    device: &str,
    gpu_eligible: bool,
    math_mode: Option<&'static str>,
    spmm_policy: Option<&'static str>,
    graph_replay: Option<bool>,
) -> PyResult<()> {
    let mut info = super::route::simple_exec_info(
        device,
        gpu_eligible,
        scx_accel::AccelRoute::GpuCsr,
        scx_accel::AccelRoute::CpuCsr,
    );
    if info.route.is_gpu() {
        info.math_mode = math_mode;
        info.spmm_policy = spmm_policy;
        info.graph_replay = graph_replay;
    }
    super::route::announce_route(py, "pca", device, &info);
    super::route::write_accel_route(py, adata, "pca", &info)
}

/// Extract a boolean array (from a numpy bool array or a pandas Series) into a
/// `Vec<bool>`.
fn extract_bool_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
    // Coerce any sequence (numpy array, pandas Series, list, tuple, or a
    // non-contiguous view/slice) to a C-contiguous bool array so `as_slice()`
    // below cannot fail with AsSliceError on a non-contiguous input.
    let np = obj.py().import("numpy")?;
    let kwargs = PyDict::new(obj.py());
    kwargs.set_item("dtype", "bool")?;
    let arr = np
        .call_method("ascontiguousarray", (obj,), Some(&kwargs))
        .map_err(|_| {
            PyValueError::new_err(
                "mask_var must be a boolean array (or a var-column name resolving to one)",
            )
        })?;
    let ro: PyReadonlyArray1<bool> = arr.extract().map_err(|_| {
        PyValueError::new_err(
            "mask_var must be 1-D and boolean (or a var-column name resolving to one)",
        )
    })?;
    Ok(ro.as_slice()?.to_vec())
}

/// Resolve the PCA column mask (scanpy `mask_var` semantics).
///
/// Returns `Some((col_indices, mask_name))` when a mask applies, or `None` to
/// analyze all genes. `mask_var`: a `str` (var-column name), a boolean array of
/// length `n_vars`, or `None`. `None` auto-consumes `adata.var['highly_variable']`
/// when that column exists (matching scanpy's default), else `None` (all genes).
/// `col_indices` is ascending (sorted-unique), suitable for `project_csr` and a
/// stable projected→original mapping. Errors on wrong length or an all-false mask.
fn resolve_mask_var(
    adata: &Bound<'_, PyAny>,
    mask_var: Option<&Bound<'_, PyAny>>,
    n_vars: usize,
) -> PyResult<Option<(Vec<u32>, Option<String>)>> {
    let resolved: Option<(Vec<bool>, Option<String>)> = match mask_var {
        Some(obj) => {
            if let Ok(name) = obj.extract::<String>() {
                let col = adata.getattr("var")?.get_item(&name).map_err(|_| {
                    PyValueError::new_err(format!(
                        "mask_var column '{name}' not found in adata.var"
                    ))
                })?;
                Some((extract_bool_vec(&col)?, Some(name)))
            } else {
                Some((extract_bool_vec(obj)?, None))
            }
        }
        None => {
            // Auto-consume adata.var['highly_variable'] when present.
            let var = adata.getattr("var")?;
            let has_hvg = var
                .call_method1("__contains__", ("highly_variable",))?
                .extract::<bool>()?;
            if has_hvg {
                let col = var.get_item("highly_variable")?;
                Some((extract_bool_vec(&col)?, Some("highly_variable".to_string())))
            } else {
                None
            }
        }
    };

    match resolved {
        None => Ok(None),
        Some((mask, name)) => {
            if mask.len() != n_vars {
                return Err(PyValueError::new_err(format!(
                    "mask_var length {} != n_vars {n_vars}",
                    mask.len()
                )));
            }
            let cols: Vec<u32> = mask
                .iter()
                .enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| i as u32)
                .collect();
            if cols.is_empty() {
                return Err(PyValueError::new_err(
                    "mask_var selects zero genes (all-false mask)",
                ));
            }
            Ok(Some((cols, name)))
        }
    }
}

/// Extra write-back context for [`write_pca_to_adata`]: records
/// `uns['pca']['params']` and, when a column mask was applied, scatters the
/// masked components back onto the full var axis.
pub(crate) struct PcaWriteParams<'a> {
    pub zero_center: bool,
    pub n_comps: usize,
    pub use_highly_variable: bool,
    /// The `mask_var` value recorded in params (column name, or None).
    pub mask_var: Option<&'a str>,
    /// masked→original column indices (length = result.n_vars) when masked.
    pub mask_cols: Option<&'a [u32]>,
    /// Full var-axis length for the scattered `varm["PCs"]`.
    pub full_n_vars: usize,
}

/// Run streaming CPU PCA on a `ShardSource` (covariance or randomized per `m`).
fn cpu_pca_stream<S: ShardSource + Sync>(
    source: &S,
    m: &str,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
) -> std::result::Result<scx_accel::PcaResult, scx_accel::AccelError> {
    match m {
        "covariance" => scx_accel::covariance_pca(source, n_comps, zero_center),
        _ => scx_accel::randomized_pca(
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
        ),
    }
}

#[pyfunction]
#[pyo3(signature = (adata, n_comps=50, zero_center=true, random_state=0, n_oversamples=10, n_power_iterations=2, device="auto", method="auto", qr_method="householder", prefer_format="csr", allow_tf32=false, spmm_policy="default", memory_budget=None, mask_var=None))]
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
    // Task 2.5 GPU tuning knobs. `allow_tf32` enables cuBLAS/cuSPARSE TF32
    // tensor-op math (faster, reduced-precision — validate by subspace, not
    // bitwise). `spmm_policy` selects the cuSPARSE SpMM algorithm policy. Both
    // are no-ops on the CPU path (recorded on `uns["scx_accel"]["pca"]`).
    allow_tf32: bool,
    spmm_policy: &str,
    // RAM ceiling (bytes, or a string like "8GB") for the out-of-core
    // backed-PCA shard cache. PCA makes several passes over every shard, so a
    // budget large enough to hold all shards turns the default warn-and-rescan
    // into decode-once-per-pass. `None` uses a conservative default ceiling.
    memory_budget: Option<&Bound<'_, PyAny>>,
    // Column mask (scanpy `mask_var`): a var-column name, a boolean array of
    // length n_vars, or `None`. `None` auto-consumes `adata.var['highly_variable']`
    // when present (scanpy semantics), else uses all genes. PCA runs on the
    // selected columns only; `varm["PCs"]` stays aligned to the full var axis
    // (excluded rows filled with 0).
    mask_var: Option<&Bound<'_, PyAny>>,
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
    if !matches!(spmm_policy, "default" | "deterministic" | "benchmark_once") {
        return Err(PyValueError::new_err(format!(
            "Invalid spmm_policy={spmm_policy:?}; expected 'default', 'deterministic', \
             or 'benchmark_once'"
        )));
    }
    // `allow_tf32` is consumed only on the GPU PCA path; silence the unused-var
    // lint on CPU-only builds (validation above still runs for `spmm_policy`).
    #[cfg(not(feature = "gpu"))]
    let _ = allow_tf32;
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

    // In the in-VRAM regime (X in memory, not a backed/lazy streaming source)
    // hand `device="gpu"` PCA to rapids-singlecell. backed/lazy X stays on the
    // native streaming path (the >VRAM moat).
    #[cfg(feature = "gpu")]
    {
        let x_in_memory = x.cast::<ScxBackedSparseDataset>().is_err()
            && x.cast::<ScxLazyTransformedDataset>().is_err();
        if x_in_memory {
            match super::rapids::decide(py, _device, "pca") {
                super::rapids::RapidsDecision::Rapids(gid) => {
                    super::rapids::run(py, adata, "pca", gid, |py, adata| {
                        let kw = super::rapids::kwargs(py);
                        kw.set_item("n_comps", n_comps)?;
                        kw.set_item("zero_center", zero_center)?;
                        kw.set_item("random_state", random_state)?;
                        // Delegate column masking to rapids-singlecell (it
                        // auto-consumes highly_variable itself when mask_var is
                        // absent, matching our None default).
                        if let Some(mv) = mask_var {
                            kw.set_item("mask_var", mv)?;
                        }
                        // n_oversamples / n_power_iterations / method / qr_method
                        // are native-SVD-solver internals with no rapids analogue
                        // (rapids picks its own svd_solver) — intentionally not
                        // forwarded.
                        super::rapids::call_rsc_pca(py, adata, &kw)?;
                        Ok(())
                    })?;
                    return Ok(());
                }
                super::rapids::RapidsDecision::NoRapidsCpu => {
                    pca(
                        py,
                        adata,
                        n_comps,
                        zero_center,
                        random_state,
                        n_oversamples,
                        n_power_iterations,
                        "cpu",
                        method,
                        qr_method,
                        prefer_format,
                        allow_tf32,
                        spmm_policy,
                        memory_budget,
                        mask_var,
                    )?;
                    return super::rapids::stamp_no_rapids(
                        py,
                        adata,
                        "pca",
                        scx_accel::route::AccelRoute::CpuCsr,
                    );
                }
                super::rapids::RapidsDecision::Native => {}
            }
        }
    }

    // Resolve the column mask (scanpy `mask_var`) once. Used by the native GPU
    // and CPU dispatch below to project columns via a ProjectedShardSource
    // (no materialization) and to scatter components back onto the full var axis.
    let full_n_vars = adata.getattr("n_vars")?.extract::<usize>()?;
    let mask = resolve_mask_var(adata, mask_var, full_n_vars)?;
    let mask_cols: Option<&[u32]> = mask.as_ref().map(|(c, _)| c.as_slice());
    let mask_name: Option<&str> = mask.as_ref().and_then(|(_, n)| n.as_deref());
    // Build the params bundle recorded on uns['pca']['params'] + drives the
    // full-axis varm["PCs"] scatter when masked.
    let write_params = PcaWriteParams {
        zero_center,
        n_comps,
        use_highly_variable: mask_cols.is_some(),
        mask_var: mask_name,
        mask_cols,
        full_n_vars,
    };

    // Record the planned route on adata.uns["scx_accel"]["pca"]. PCA has a
    // single GPU route (cuSPARSE + cuBLAS) gated on the modern cuSPARSE ABI;
    // when that probe fails the dispatch falls back to CPU, which the planner
    // records as UnsupportedInputLayout (vs NoCuda when CUDA is simply absent).
    // The covariance-vs-randomized choice is orthogonal math policy and stays in
    // adata.uns["pca"]["backend"], not the route string.
    //
    // INVARIANT (pre-dispatch stamp): this is safe to stamp *before* dispatch
    // only because (a) `pca_gpu_eligible` is the *same* probe the GPU dispatch
    // re-checks below (`gpu_device_id` is `Some` iff
    // `cusparse_modern_abi_available()`), and (b) the GPU kernel propagates
    // errors via `.map_err(..)?` rather than silently falling back to CPU — so
    // the only way to reach the CPU path is one this probe already predicted.
    // If a silent GPU→CPU runtime fallback is ever added here, switch to
    // stamping *after* dispatch on the branch that actually ran (see umap.rs),
    // or this gate will false-pass.
    #[cfg(feature = "gpu")]
    let pca_gpu_eligible = scx_accel::cusparse_modern_abi_available();
    #[cfg(not(feature = "gpu"))]
    let pca_gpu_eligible = false;
    // Pre-dispatch stamp records the route only; the tuning knobs + graph_replay
    // are filled by the per-branch re-stamp after GPU dispatch (and only for the
    // randomized route that actually uses them).
    stamp_pca_route(py, adata, device, pca_gpu_eligible, None, None, None)?;

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
        let tuning = build_pca_tuning(allow_tf32, spmm_policy);
        // Task 2.5 metadata labels — recorded only for the randomized route that
        // actually consumes them (see the per-branch re-stamp below).
        let math_mode_label = if allow_tf32 {
            "allow_tf32"
        } else {
            "strict_fp32"
        };
        let spmm_policy_lbl = spmm_policy_label(spmm_policy);
        // Re-stamp the route after dispatch. `math_mode` / `spmm_policy` are
        // recorded only for the randomized route that actually consumes them
        // (covariance passes `None`); `stamp_pca_route` itself drops every knob
        // on a non-GPU route.
        let stamp = |graph_replayed: Option<bool>, m: &str| -> PyResult<()> {
            let is_rand = m == "randomized";
            stamp_pca_route(
                py,
                adata,
                device,
                pca_gpu_eligible,
                is_rand.then_some(math_mode_label),
                is_rand.then_some(spmm_policy_lbl),
                graph_replayed,
            )
        };

        if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
            let reader = &*backed.backed;
            let (_n_obs, n_vars) = reader.shape();
            let m = resolve_gpu_method(method, mask_cols.map(|c| c.len()).unwrap_or(n_vars))?;
            let result = match mask_cols {
                Some(cols) => {
                    let proj = scx_accel::ProjectedShardSource::new(reader, cols.to_vec());
                    gpu_pca_dispatch_unwind_safe(
                        device_id,
                        &proj,
                        n_comps,
                        n_oversamples,
                        n_power_iterations,
                        zero_center,
                        random_state,
                        m,
                        qr,
                        tuning,
                    )
                }
                None => gpu_pca_dispatch_unwind_safe(
                    device_id,
                    reader,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    m,
                    qr,
                    tuning,
                ),
            }
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            stamp(result.graph_replayed, m)?;
            write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse", Some(&write_params))?;
            return Ok(());
        }

        if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
            let source = lazy.as_shard_source();
            let (_n_obs, n_vars) = source.shape();
            let m = resolve_gpu_method(method, mask_cols.map(|c| c.len()).unwrap_or(n_vars))?;
            let result = match mask_cols {
                Some(cols) => {
                    let proj = scx_accel::ProjectedShardSource::new(&source, cols.to_vec());
                    gpu_pca_dispatch_unwind_safe(
                        device_id,
                        &proj,
                        n_comps,
                        n_oversamples,
                        n_power_iterations,
                        zero_center,
                        random_state,
                        m,
                        qr,
                        tuning,
                    )
                }
                None => gpu_pca_dispatch_unwind_safe(
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    m,
                    qr,
                    tuning,
                ),
            }
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            stamp(result.graph_replayed, m)?;
            write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse", Some(&write_params))?;
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
            let m = resolve_gpu_method(method, mask_cols.map(|c| c.len()).unwrap_or(n_vars))?;
            let result = match mask_cols {
                Some(cols) => {
                    let proj = scx_accel::ProjectedShardSource::new(&source, cols.to_vec());
                    gpu_pca_dispatch_unwind_safe(
                        device_id,
                        &proj,
                        n_comps,
                        n_oversamples,
                        n_power_iterations,
                        zero_center,
                        random_state,
                        m,
                        qr,
                        tuning,
                    )
                }
                None => gpu_pca_dispatch_unwind_safe(
                    device_id,
                    &source,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    m,
                    qr,
                    tuning,
                ),
            }
            .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
            stamp(result.graph_replayed, m)?;
            write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse", Some(&write_params))?;
            return Ok(());
        }

        // Fallback: owned-Vec path (e.g. exotic X types scipy can't view).
        let csr = extract_materialized_csr(py, &x)?;
        let source = ScxCsrSource { csr: &csr };
        let n_vars = source.n_vars();
        let m = resolve_gpu_method(method, mask_cols.map(|c| c.len()).unwrap_or(n_vars))?;
        let result = match mask_cols {
            Some(cols) => {
                let proj = scx_accel::ProjectedShardSource::new(&source, cols.to_vec());
                gpu_pca_dispatch_unwind_safe(
                    device_id,
                    &proj,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                    m,
                    qr,
                    tuning,
                )
            }
            None => gpu_pca_dispatch_unwind_safe(
                device_id,
                &source,
                n_comps,
                n_oversamples,
                n_power_iterations,
                zero_center,
                random_state,
                m,
                qr,
                tuning,
            ),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?;
        stamp(result.graph_replayed, m)?;
        write_pca_to_adata(py, adata, &result, "scx-gpu-cusparse", Some(&write_params))?;
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

    // RAM ceiling for the backed-PCA shard cache. `None` → a conservative
    // default so the common case gets the multi-pass speedup without
    // unbounded growth on a count-only-opened reader.
    let pca_cache_bytes = crate::convert::parse_memory_budget(memory_budget)?
        .unwrap_or(DEFAULT_PCA_CACHE_BYTES) as usize;

    // Effective var count after masking (drives covariance-vs-randomized route).
    let n_vars_eff = |full: usize| mask_cols.map(|c| c.len()).unwrap_or(full);

    let result = if let Ok(backed) = x.extract::<PyRef<ScxBackedSparseDataset>>() {
        backend = "scx-accel-cpu";
        let reader = std::sync::Arc::clone(&backed.backed);
        let full_vars = reader.shape().1;
        drop(backed);
        // Out-of-core PCA re-reads every shard once per pass; size the decoded
        // shard cache to hold the whole working set within the RAM ceiling so
        // each shard decodes once per pass instead of every pass.
        reader.ensure_cache_capacity(reader.n_shards(), pca_cache_bytes);
        let m = pick_cpu_method(n_vars_eff(full_vars));
        match mask_cols {
            Some(cols) => {
                let proj = scx_accel::ProjectedShardSource::new(&*reader, cols.to_vec());
                py.detach(|| {
                    cpu_pca_stream(
                        &proj,
                        m,
                        n_comps,
                        n_oversamples,
                        n_power_iterations,
                        zero_center,
                        random_state,
                    )
                })
            }
            None => py.detach(|| {
                cpu_pca_stream(
                    &*reader,
                    m,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                )
            }),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        backend = "scx-accel-cpu";
        let source = lazy.as_shard_source();
        let full_vars = source.shape().1;
        drop(lazy);
        let m = pick_cpu_method(n_vars_eff(full_vars));
        match mask_cols {
            Some(cols) => {
                let proj = scx_accel::ProjectedShardSource::new(&source, cols.to_vec());
                py.detach(|| {
                    cpu_pca_stream(
                        &proj,
                        m,
                        n_comps,
                        n_oversamples,
                        n_power_iterations,
                        zero_center,
                        random_state,
                    )
                })
            }
            None => py.detach(|| {
                cpu_pca_stream(
                    &source,
                    m,
                    n_comps,
                    n_oversamples,
                    n_power_iterations,
                    zero_center,
                    random_state,
                )
            }),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else {
        backend = "scx-accel-cpu";
        let csr0 = extract_materialized_csr(py, &x)?;
        // Materialized in-memory path: project columns directly (no streaming
        // source needed) so `mask_var` works identically here.
        let csr = match mask_cols {
            Some(cols) => scx_engine::project_csr(&csr0, cols),
            None => csr0,
        };
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

    write_pca_to_adata(py, adata, &result, backend, Some(&write_params))?;

    Ok(())
}

/// Write PCA results to AnnData slots matching scanpy's format.
///
/// `params`, when `Some`, drives two scanpy-parity behaviors: (a) if a column
/// mask was applied (`params.mask_cols`), `varm["PCs"]` is scattered back onto
/// the **full** var axis (`params.full_n_vars` rows, excluded vars = 0); and
/// (b) `uns['pca']['params']` is written (`zero_center`, `use_highly_variable`,
/// `mask_var`, `n_comps`). `None` preserves the legacy behavior (varm sized to
/// `result.n_vars`, no params dict) for callers that don't mask (e.g. fused).
pub(crate) fn write_pca_to_adata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    result: &scx_accel::PcaResult,
    backend: &str,
    params: Option<&PcaWriteParams>,
) -> PyResult<()> {
    let numpy = py.import("numpy")?;

    // adata.obsm["X_pca"] = embeddings (n_obs × n_components) as float32
    let _m = scx_accel::cpu_profile::start();
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
    scx_accel::cpu_profile::record_marshalling_since(
        _m,
        result.n_obs * result.n_components * std::mem::size_of::<f32>(),
    );
    let obsm = adata.getattr("obsm")?;
    obsm.set_item("X_pca", embeddings_arr)?;

    // adata.varm["PCs"] = components as (var × n_components) f32. When a column
    // mask was applied, scatter the masked components back onto the full var
    // axis (excluded vars → 0) so PCs stays aligned to `adata.var` (scanpy).
    let mask_cols = params.and_then(|p| p.mask_cols);
    let pcs_rows: Vec<Vec<f32>> = match mask_cols {
        Some(cols) => {
            let full_n_vars = params.map(|p| p.full_n_vars).unwrap_or(result.n_vars);
            let mut rows = vec![vec![0.0f32; result.n_components]; full_n_vars];
            // result is in masked space: local column j ↔ original var cols[j].
            for (j, &orig) in cols.iter().enumerate() {
                let row = &mut rows[orig as usize];
                for (pc, slot) in row.iter_mut().enumerate() {
                    *slot = result.components[pc * result.n_vars + j] as f32;
                }
            }
            rows
        }
        None => (0..result.n_vars)
            .map(|v| {
                (0..result.n_components)
                    .map(|pc| result.components[pc * result.n_vars + v] as f32)
                    .collect::<Vec<f32>>()
            })
            .collect(),
    };
    let pcs_arr = PyArray2::<f32>::from_vec2(py, &pcs_rows)?;
    let varm = adata.getattr("varm")?;
    varm.set_item("PCs", pcs_arr)?;

    // adata.uns["pca"] = dict with variance info + backend
    let pca_dict = PyDict::new(py);

    let var_explained = numpy.call_method1("array", (result.variance_explained.clone(),))?;
    pca_dict.set_item("variance", var_explained)?;

    let var_ratio = numpy.call_method1("array", (result.variance_ratio.clone(),))?;
    pca_dict.set_item("variance_ratio", var_ratio)?;

    pca_dict.set_item("backend", backend)?;

    // scanpy-style uns['pca']['params'] (only when the caller supplies context).
    if let Some(p) = params {
        let params_dict = PyDict::new(py);
        params_dict.set_item("zero_center", p.zero_center)?;
        params_dict.set_item("use_highly_variable", p.use_highly_variable)?;
        params_dict.set_item("mask_var", p.mask_var)?;
        params_dict.set_item("n_comps", p.n_comps)?;
        pca_dict.set_item("params", params_dict)?;
    }

    let uns = adata.getattr("uns")?;
    uns.set_item("pca", pca_dict)?;

    Ok(())
}
