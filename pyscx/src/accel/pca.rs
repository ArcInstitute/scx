//! PCA bindings — randomized and covariance PCA, streaming + in-memory.

use scx_format_io::ShardSource;

use numpy::PyArray1;
use numpy::PyReadonlyArray1;
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
pub(super) const DEFAULT_PCA_CACHE_BYTES: u64 = 8 * 1024 * 1024 * 1024;

#[cfg(feature = "gpu")]
use super::util::{extract_csr_slices, CsrSlices};

/// Floor on the share of `memory_budget` the decoded-shard LRU keeps.
///
/// A backstop only. [`resolve_pca_prefetch`] clamps the *depth* so the reserve
/// can never exceed half the budget, which makes this floor unreachable —
/// `prefetch_reserve_floor_is_unreachable` asserts exactly that. Kept because
/// the alternative is an invariant that lives only in a comment.
const MIN_LRU_SHARE_OF_BUDGET: u64 = 2;

/// How PCA splits `memory_budget` between the decoded-shard LRU and the
/// decode-prefetch pipeline: `(depth, lru_bytes)`.
///
/// `pca(memory_budget=…)` is documented as the RAM ceiling for out-of-core PCA,
/// and prefetch keeps up to `depth` decoded shards alive on top of whatever the
/// LRU holds — so the depth *is* a memory knob and has to be resolved against
/// that ceiling rather than taken from the process-wide default.
///
/// **The depth is clamped first, and that is what makes the ceiling hold.** With
/// the reserve bounded to half the budget, worst-case live bytes are
/// `(B − (depth−1)·s) + depth·s = B + s` — exactly what the pre-prefetch loop
/// peaked at, since it too held one decoded shard outside the cache. An earlier
/// version clamped the *reserve* against a floor instead, which broke that
/// identity whenever `(depth−1)·s > B/2`: the LRU stopped shrinking while the
/// pipeline kept holding `depth` shards, so peak could reach `B/2 + depth·s`.
/// Clamping the depth cannot do that, because both terms move together.
///
/// `depth == 1` reserves nothing, which keeps the capture's
/// `SCX_ACCEL_PREFETCH_DEPTH=1` arm byte-for-byte the pre-prefetch behaviour.
///
/// `per_shard_bytes = None` leaves both alone: with no catalog statistics there
/// is nothing to resolve *against*, and inventing an estimate would be worse
/// than the documented over-run.
///
/// `lru_bytes` is meaningless for a source that has no LRU — the lazy and pflog
/// paths decode fresh on every read — so those callers take `depth` and discard
/// it. They need the clamp more, not less: without an LRU nothing else bounds
/// what the pipeline holds.
pub(super) fn resolve_pca_prefetch(
    per_shard_bytes: Option<u64>,
    budget_bytes: u64,
) -> (usize, u64) {
    let base = scx_accel::pca_prefetch_depth();
    let Some(per_shard) = per_shard_bytes else {
        return (base, budget_bytes);
    };
    let depth = scx_format_io::clamp_prefetch_depth(
        base,
        per_shard,
        budget_bytes / MIN_LRU_SHARE_OF_BUDGET,
    );
    let reserve = per_shard.saturating_mul(depth.saturating_sub(1) as u64);
    (depth, budget_bytes.saturating_sub(reserve))
}

/// Per-shard decoded-byte estimate, or `None` when the catalog carries no
/// statistics to derive one from.
pub(super) fn per_shard_estimate<S: ShardSource + ?Sized>(source: &S) -> Option<u64> {
    ShardSource::shard_size_hint(source).map(|h| h.decoded_bytes())
}

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
#[allow(clippy::too_many_arguments)]
fn cpu_pca_stream<S: ShardSource + Sync>(
    source: &S,
    m: &str,
    n_comps: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    random_state: u64,
    depth: usize,
) -> std::result::Result<scx_accel::PcaResult, scx_accel::AccelError> {
    // The depth-explicit entries, never the convenience wrappers: every caller
    // has resolved the depth against `memory_budget` via `resolve_pca_prefetch`,
    // and falling back to the process-wide default here would put the ceiling
    // back outside the user's control.
    match m {
        "covariance" => scx_accel::covariance_pca_with_depth(source, n_comps, zero_center, depth),
        _ => scx_accel::randomized_pca_with_depth(
            source,
            n_comps,
            n_oversamples,
            n_power_iterations,
            zero_center,
            random_state,
            depth,
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
    // A presentation-ordered backed `X` (`preserve_var_order=True`) cannot be
    // expressed as a `ShardSource`: the source emits columns in sorted
    // on-disk order while `adata.var` is in request order, so `varm["PCs"]`
    // would silently misalign against the gene names. Same guard the other
    // streaming accel ops already apply.
    super::prepare_target(py, adata, "pca")?;

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
    //
    // The invariant is about which *branch* the recorded route names; it is
    // orthogonal to whether the op finished. `RouteStamp` covers the latter —
    // the stamp is rolled back if any branch below raises, so a present entry
    // means the route ran *and* completed.
    #[cfg(feature = "gpu")]
    let pca_gpu_eligible = scx_accel::cusparse_modern_abi_available();
    #[cfg(not(feature = "gpu"))]
    let pca_gpu_eligible = false;
    // Pre-dispatch stamp records the route only; the tuning knobs + graph_replay
    // are filled by the per-branch re-stamp after GPU dispatch (and only for the
    // randomized route that actually uses them).
    // One guard for the whole op, opened at the earliest stamp: the GPU
    // branches re-stamp after dispatch, and a failure after that must still
    // restore whatever a previous successful pca recorded.
    let route = super::route::RouteStamp::begin(adata, "pca")?;
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
            // The handle's *view*, not the raw reader: `kept_to_global` and
            // `col_projection` folded in, so `n_vars` is the visible width and
            // `mask_cols` (resolved against `adata.n_vars`) composes directly.
            // NOT `with_cached_reads()`: `RawGpuShardSource` stages via
            // `ShardSource::read_shard`, so a cached source would hand back the
            // shared LRU `Arc`, fail `try_unwrap`, and deep-clone every shard —
            // where the raw reader decoded fresh and `MADV_DONTNEED`d after. The
            // GPU arm also never calls `ensure_cache_capacity` (that is CPU-only,
            // below), so the LRU would stay at the open-time `cache_shards` and
            // thrash. Uncached matches the pre-existing behaviour exactly.
            let source = backed.as_shard_source();
            let n_vars = ShardSource::n_vars(&source);
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
            route.commit();
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
            route.commit();
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
            route.commit();
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
        route.commit();
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
        // The handle's *view* — `kept_to_global` + `col_projection` applied, so
        // the embedding rows line up with `adata.obs` and the components with
        // `adata.var`. `with_cached_reads` keeps the LRU that
        // `ensure_cache_capacity` below sizes.
        let source = backed.as_shard_source().with_cached_reads();
        let full_vars = ShardSource::n_vars(&source);
        drop(backed);
        // Out-of-core PCA re-reads every shard once per pass; size the decoded
        // shard cache to hold the whole working set within the RAM ceiling so
        // each shard decodes once per pass instead of every pass. The
        // decode-prefetch pipeline holds shards too, so its share comes out of
        // the same ceiling rather than sitting on top of it.
        let (depth, lru_bytes) =
            resolve_pca_prefetch(per_shard_estimate(&source), pca_cache_bytes as u64);
        reader.ensure_cache_capacity(reader.n_shards(), lru_bytes as usize);
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
                        depth,
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
                    depth,
                )
            }),
        }
        .map_err(|e: scx_accel::AccelError| PyRuntimeError::new_err(e.to_string()))?
    } else if let Ok(lazy) = x.extract::<PyRef<ScxLazyTransformedDataset>>() {
        backend = "scx-accel-cpu";
        let source = lazy.as_shard_source();
        let full_vars = source.shape().1;
        drop(lazy);
        // No LRU on this path — a lazy source decodes and re-transforms on every
        // read — so the whole budget is available to the pipeline, and the
        // clamp is the *only* thing bounding what it holds. Before this the lazy
        // and pflog paths took the process-wide depth and went from holding one
        // transformed shard to `depth` of them with nothing binding, while
        // `memory_budget` was documented as covering prefetch. It only did on
        // the backed branch.
        let (depth, _) = resolve_pca_prefetch(per_shard_estimate(&source), pca_cache_bytes as u64);
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
                        depth,
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
                    depth,
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

    route.commit();
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
    // adata.obsm["X_pca"] = embeddings (n_obs × n_components) as float32.
    // `result.embeddings` is already row-major flat (element (i,j) at
    // i*n_components+j), so the flat f32 buffer matches shape [n_obs, n_comp].
    let marshal_start = scx_accel::cpu_profile::start();
    let n_obs = result.n_obs;
    let n_components = result.n_components;
    let embeddings_flat: Vec<f32> = result.embeddings.iter().map(|&v| v as f32).collect();
    let embeddings_arr = super::util::flat_pyarray2(py, embeddings_flat, n_obs, n_components)?;
    scx_accel::cpu_profile::record_marshalling_since(
        marshal_start,
        n_obs * n_components * std::mem::size_of::<f32>(),
    );
    let obsm = adata.getattr("obsm")?;
    obsm.set_item("X_pca", embeddings_arr)?;

    // adata.varm["PCs"] = components as (var × n_components) f32, transposed
    // from the component-major `result.components` (indexed pc*n_vars+v). When
    // a column mask was applied, scatter the masked components back onto the
    // full var axis (excluded vars → 0) so PCs stays aligned to `adata.var`
    // (scanpy). Both branches fill one flat row-major buffer at
    // full_var_rows*n_components (element (var,pc) at var*n_components+pc).
    let mask_cols = params.and_then(|p| p.mask_cols);
    let full_var_rows = match mask_cols {
        Some(_) => params.map(|p| p.full_n_vars).unwrap_or(result.n_vars),
        None => result.n_vars,
    };
    let mut pcs_flat = vec![0.0f32; full_var_rows * result.n_components];
    match mask_cols {
        Some(cols) => {
            // result is in masked space: local column j ↔ original var cols[j].
            for (j, &orig) in cols.iter().enumerate() {
                let base = orig as usize * result.n_components;
                for pc in 0..result.n_components {
                    pcs_flat[base + pc] = result.components[pc * result.n_vars + j] as f32;
                }
            }
        }
        None => {
            for v in 0..result.n_vars {
                let base = v * result.n_components;
                for pc in 0..result.n_components {
                    pcs_flat[base + pc] = result.components[pc * result.n_vars + v] as f32;
                }
            }
        }
    }
    let pcs_arr = super::util::flat_pyarray2(py, pcs_flat, full_var_rows, result.n_components)?;
    let varm = adata.getattr("varm")?;
    varm.set_item("PCs", pcs_arr)?;

    // adata.uns["pca"] = dict with variance info + backend
    let pca_dict = PyDict::new(py);

    let var_explained = PyArray1::from_slice(py, &result.variance_explained);
    pca_dict.set_item("variance", var_explained)?;

    let var_ratio = PyArray1::from_slice(py, &result.variance_ratio);
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

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;
    const MIB: u64 = 1024 * 1024;

    #[test]
    fn prefetch_reserve_is_memory_neutral_against_the_sequential_loop() {
        let shard = 90 * 1024 * 1024; // ~census_1m
        let budget = 8 * GIB;
        let (depth, lru) = resolve_pca_prefetch(Some(shard), budget);
        // Peak live bytes = LRU + the shards the pipeline holds. The pre-prefetch
        // loop peaked at `budget + shard` (the cache, plus the one decoded shard
        // it held outside it), so matching that exactly is the whole claim.
        assert_eq!(lru + shard * depth as u64, budget + shard);
        assert_eq!(lru, budget - shard * (depth as u64 - 1));
    }

    #[test]
    fn prefetch_reserve_is_neutral_at_every_budget_and_shard_size() {
        // Including the regime the old floor-clamped version broke: a shard that
        // is a large fraction of the budget. The identity must hold there too,
        // which it now does because the *depth* absorbs the pressure.
        for budget in [1_000u64, 64 * MIB, GIB, 8 * GIB] {
            for shard in [
                1u64,
                1_000,
                401,
                MIB,
                90 * MIB,
                budget / 3,
                budget,
                budget * 2,
            ] {
                let (depth, lru) = resolve_pca_prefetch(Some(shard), budget);
                assert!(depth >= 1, "depth must never be zero");
                assert_eq!(
                    lru + shard * depth as u64,
                    budget + shard,
                    "neutrality broken at budget={budget} shard={shard} depth={depth} lru={lru}"
                );
            }
        }
    }

    #[test]
    fn prefetch_reserve_floor_is_unreachable() {
        // `MIN_LRU_SHARE_OF_BUDGET` is a backstop, not a policy: clamping the
        // depth first bounds the reserve to half the budget, so the LRU can never
        // be driven below that. If this ever fails, the floor has started binding
        // and the neutrality identity above is no longer guaranteed.
        for budget in [1_000u64, 64 * MIB, GIB, 8 * GIB] {
            for shard in [1u64, 401, MIB, 90 * MIB, budget / 3, budget, budget * 2] {
                let (_, lru) = resolve_pca_prefetch(Some(shard), budget);
                assert!(
                    lru >= budget / MIN_LRU_SHARE_OF_BUDGET,
                    "floor bound at budget={budget} shard={shard}: lru={lru}"
                );
            }
        }
    }

    #[test]
    fn prefetch_reserve_needs_a_per_shard_estimate() {
        // No catalog statistics → nothing to resolve *against*. Guessing would be
        // worse than the documented over-run, so the budget is left whole and the
        // depth stays the process default.
        let (depth, lru) = resolve_pca_prefetch(None, 8 * GIB);
        assert_eq!(lru, 8 * GIB);
        assert_eq!(depth, scx_accel::pca_prefetch_depth());
    }

    #[test]
    fn a_shard_larger_than_the_budget_falls_back_to_depth_one() {
        // Degenerate but reachable via a tight `memory_budget=`. Prefetching at
        // all would blow the ceiling, so it must not: depth 1 reserves nothing
        // and the run behaves exactly as it did before prefetch existed.
        let (depth, lru) = resolve_pca_prefetch(Some(4 * GIB), GIB);
        assert_eq!(depth, 1);
        assert_eq!(lru, GIB);
    }
}
