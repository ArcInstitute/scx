//! rapids-singlecell routing helpers (ACC-RUST-OPT-V4 Phase 1.3 / 1.4).
//!
//! In the **in-VRAM** regime, `device="gpu"` analysis ops hand off to
//! [rapids-singlecell](https://rapids-singlecell.readthedocs.io/) — the GPU
//! compute layer — rather than SCX's native CUDA kernels. rapids is a *detected
//! runtime dependency* (§4.3): present → route to it; absent → route CPU with a
//! one-shot diagnostic and `fallback_reason="no_rapids"`. The **surviving** native
//! GPU paths stay reachable via `SCX_FORCE_NATIVE_GPU=1` — after Phase 3 these are
//! the streaming preprocess kernels (also fed by the ML loader) and randomized
//! PCA; the in-VRAM native UMAP, covariance PCA, and CAGRA kNN paths the override
//! used to also pin were removed in Phase 3. The out-of-VRAM streaming moat
//! (backed / lazy `X`) never routes here regardless of the override.
//!
//! This module is GPU-gated; callers reference it from `#[cfg(feature = "gpu")]`
//! blocks only.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use scx_accel::route::{AccelExecutionInfo, AccelRoute, FallbackReason};

use super::gpu::{rapids_singlecell_info, ResolvedDevice};
use super::route::write_accel_route;

/// `SCX_FORCE_NATIVE_GPU` override — keep the native SCX GPU kernels instead of
/// routing in-VRAM GPU compute to rapids-singlecell. After Phase 3 it pins only
/// the surviving native paths (streaming preprocess kernels, randomized PCA);
/// the in-VRAM UMAP / covariance-PCA / CAGRA-kNN kernels it also used to pin were
/// removed, so for those ops it now falls through to the CPU path.
pub(crate) fn force_native_gpu() -> bool {
    matches!(std::env::var("SCX_FORCE_NATIVE_GPU"), Ok(v) if v != "0" && !v.is_empty())
}

/// `SCX_DISABLE_RAPIDS` override — treat rapids-singlecell as if it were not
/// importable, so a GPU op takes the `no_rapids` CPU-fallback path even on a host
/// where rapids *is* installed. Exists so the Phase 2.2 fallback gate can exercise
/// the rapids-absent contract without uninstalling rapids; honored ahead of the
/// `force_native` override so the fallback (not the native kernels) is what runs.
pub(crate) fn rapids_disabled() -> bool {
    matches!(std::env::var("SCX_DISABLE_RAPIDS"), Ok(v) if v != "0" && !v.is_empty())
}

/// How a GPU-eligible standalone op should dispatch in the in-VRAM regime.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RapidsDecision {
    /// Hand off to rapids-singlecell on this device.
    Rapids(usize),
    /// Proceed with the existing native / CPU dispatch unchanged — `device="cpu"`,
    /// or `SCX_FORCE_NATIVE_GPU=1` selecting the native GPU kernels.
    Native,
    /// GPU requested + rapids absent + not forcing native → route CPU and stamp
    /// `no_rapids` (a one-shot diagnostic has been emitted).
    NoRapidsCpu,
}

/// Decide how a GPU-eligible op dispatches. `op` names the op for the one-shot
/// rapids-absent diagnostic.
pub(crate) fn decide(py: Python<'_>, resolved: ResolvedDevice, op: &str) -> RapidsDecision {
    let gpu_id = match resolved.gpu_id() {
        Some(i) => i,
        None => return RapidsDecision::Native, // device="cpu" → existing CPU path
    };
    // SCX_DISABLE_RAPIDS forces the rapids-absent fallback (Phase 2.2 gate),
    // checked before force_native so it pins the `no_rapids` CPU path.
    if rapids_disabled() {
        warn_no_rapids_once(py, op);
        return RapidsDecision::NoRapidsCpu;
    }
    if force_native_gpu() {
        return RapidsDecision::Native;
    }
    if rapids_singlecell_info(py).available {
        RapidsDecision::Rapids(gpu_id)
    } else {
        warn_no_rapids_once(py, op);
        RapidsDecision::NoRapidsCpu
    }
}

static NO_RAPIDS_WARNED: AtomicBool = AtomicBool::new(false);

/// Emit a single `UserWarning` per process when a GPU op falls back to CPU
/// because rapids-singlecell is not importable, naming the published install
/// path and the native-GPU override.
fn warn_no_rapids_once(py: Python<'_>, op: &str) {
    if NO_RAPIDS_WARNED.swap(true, Ordering::Relaxed) {
        return;
    }
    let msg = format!(
        "pyscx.accel.{op}(device=\"gpu\"): rapids-singlecell is not importable — routing \
         GPU analysis to CPU (fallback_reason=\"no_rapids\"). Install the rapids analysis \
         backend (conda env create -f benchmarks/comprehensive/envs/scx-gpu-analysis.yml, \
         then on a GPU node: pip install --no-deps 'rapids-singlecell>=0.12'), or set \
         SCX_FORCE_NATIVE_GPU=1 to use the native SCX GPU kernels. See docs/gpu-setup.md."
    );
    if let Ok(warnings) = py.import("warnings") {
        let _ = warnings.call_method1(
            "warn",
            (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
        );
    }
}

/// True if `x` is a `cupyx.scipy.sparse` matrix already resident on device.
pub(crate) fn is_cupy_sparse(py: Python<'_>, x: &Bound<'_, PyAny>) -> bool {
    py.import("cupyx.scipy.sparse")
        .and_then(|m| m.getattr("issparse"))
        .and_then(|f| f.call1((x,)))
        .and_then(|r| r.extract::<bool>())
        .unwrap_or(false)
}

/// Run a closure pinned to `gpu_id` via the `cupy.cuda.Device` context manager
/// (the canonical rapids device-selection idiom; see `leiden.rs`). `__exit__` is
/// always called, even when the body errors.
pub(crate) fn with_device<R>(
    py: Python<'_>,
    gpu_id: usize,
    f: impl FnOnce() -> PyResult<R>,
) -> PyResult<R> {
    let device_ctx = py
        .import("cupy.cuda")?
        .getattr("Device")?
        .call1((gpu_id,))?;
    device_ctx.call_method0("__enter__")?;
    let r = f();
    let _ = device_ctx.call_method1("__exit__", (py.None(), py.None(), py.None()));
    r
}

/// Ensure `adata.X` is GPU-resident for a rapids op. If `X` is already a `cupyx`
/// sparse matrix (e.g. from `to_gpu_anndata`) this is a no-op and reports
/// `"scx_device_handoff"`; otherwise it uploads via `rsc.get.anndata_to_GPU`
/// under `gpu_id` and reports `"anndata_to_gpu"` (a host re-upload).
pub(crate) fn ensure_gpu_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    gpu_id: usize,
) -> PyResult<&'static str> {
    let x = adata.getattr("X")?;
    if is_cupy_sparse(py, &x) {
        return Ok("scx_device_handoff");
    }
    let rsc_get = py.import("rapids_singlecell")?.getattr("get")?;
    with_device(py, gpu_id, || {
        rsc_get.call_method1("anndata_to_GPU", (adata,))?;
        Ok(())
    })?;
    Ok("anndata_to_gpu")
}

/// Bring a GPU-uploaded AnnData back to host via `rsc.get.anndata_to_CPU`
/// (`convert_all=True` so result slots — `obsm`/`obsp` as well as `X` — return
/// to host and the device buffers are released). Called only when *we* uploaded
/// the host `X` (`transfer_mode == "anndata_to_gpu"`); a `to_gpu_anndata`-sourced
/// cupy `X` (`"scx_device_handoff"`) is deliberately left GPU-resident.
pub(crate) fn restore_host_anndata(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    gpu_id: usize,
) -> PyResult<()> {
    let rsc_get = py.import("rapids_singlecell")?.getattr("get")?;
    with_device(py, gpu_id, || {
        let kw = PyDict::new(py);
        kw.set_item("convert_all", true)?;
        rsc_get.call_method("anndata_to_CPU", (adata,), Some(&kw))?;
        Ok(())
    })
}

/// Stamp `uns["scx_accel"][op]` with the rapids-singlecell route + detected
/// rapids / cuML / cuPy versions + device id + transfer mode.
pub(crate) fn stamp(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &str,
    gpu_id: usize,
    transfer_mode: &'static str,
) -> PyResult<()> {
    let probe = rapids_singlecell_info(py);
    let mut info = AccelExecutionInfo::new(AccelRoute::RapidsSinglecell, FallbackReason::None);
    info.device_id = Some(gpu_id);
    info.transfer_mode = Some(transfer_mode);
    info.rapids_version = probe.rapids_version;
    info.cuml_version = probe.cuml_version;
    info.cupy_version = probe.cupy_version;
    write_accel_route(py, adata, op, &info)
}

/// Stamp `uns["scx_accel"][op]` with the CPU route + `no_rapids` fallback reason.
pub(crate) fn stamp_no_rapids(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &str,
    cpu_route: AccelRoute,
) -> PyResult<()> {
    let info = AccelExecutionInfo::new(cpu_route, FallbackReason::NoRapids);
    write_accel_route(py, adata, op, &info)
}

/// Hand an op off to rapids-singlecell: ensure `X` is on device, run `call`
/// (which invokes the `rsc.*` function) pinned to the device, then stamp the
/// route. `call` receives the GPU-resident `adata`.
///
/// Host-in → host-out contract: when `call`'s input arrived as a host AnnData
/// (we did the upload), the results and `X` are restored to host afterwards and
/// the device buffers freed — so `pyscx.accel.<op>(adata, device="gpu")` leaves
/// `adata.X` host-resident, matching the native path. A `to_gpu_anndata`-sourced
/// cupy `X` is left GPU-resident for zero-re-upload chaining.
pub(crate) fn run(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &str,
    gpu_id: usize,
    call: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    let transfer_mode = ensure_gpu_anndata(py, adata, gpu_id)?;
    let result = with_device(py, gpu_id, || call(py, adata));
    if transfer_mode == "anndata_to_gpu" {
        // Restore host on success *and* failure: a translated error (e.g. the
        // F11 zero-expression PCA hint) must leave `adata.X` host-resident so the
        // recovery path (`filter_genes(min_cells=1)` → re-run) works on a CPU
        // matrix instead of a stranded cupy one. Evaluate both eagerly so the
        // restore side effect always runs; the original op error takes precedence.
        let restored = restore_host_anndata(py, adata, gpu_id);
        if let Err(ref e) = restored {
            // `Result::and` drops this when the op itself errored; log it so a
            // restore failure isn't fully silent in the both-failed case.
            log::warn!(target: "pyscx.accel", "{op}: restore_host_anndata failed after dispatch: {e}");
        }
        result.and(restored)?;
    } else {
        result?;
    }
    stamp(py, adata, op, gpu_id, transfer_mode)
}

/// Run the fused PCA → neighbors [→ UMAP] rapids pipeline on the device AnnData
/// and stamp every stage (ACC-RUST-OPT-V4 Phase 1.4). `umap` carries the UMAP
/// stage params `(n_components, min_dist, spread, negative_sample_rate, n_epochs,
/// learning_rate)` when the UMAP stage should run; `None` runs PCA → neighbors
/// only. `n_epochs`/`learning_rate` map to rapids' `maxiter`/`alpha`.
///
/// Same host-in → host-out contract as [`run`]: an uploaded host AnnData is
/// restored to host (results + `X`) after all stages; a `to_gpu_anndata`-sourced
/// cupy `X` is left GPU-resident.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fused(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    gpu_id: usize,
    n_comps: usize,
    zero_center: bool,
    random_state: u64,
    n_neighbors: usize,
    use_rep: &str,
    umap: Option<(usize, f64, f64, usize, usize, f64)>,
) -> PyResult<()> {
    let transfer_mode = ensure_gpu_anndata(py, adata, gpu_id)?;
    let result = with_device(py, gpu_id, || {
        let kw = PyDict::new(py);
        kw.set_item("n_comps", n_comps)?;
        kw.set_item("zero_center", zero_center)?;
        kw.set_item("random_state", random_state)?;
        call_rsc_pca(py, adata, &kw)?;

        let kw = PyDict::new(py);
        kw.set_item("n_neighbors", n_neighbors)?;
        kw.set_item("use_rep", use_rep)?;
        kw.set_item("random_state", random_state)?;
        rsc_fn(py, "pp", "neighbors")?.call((adata,), Some(&kw))?;

        if let Some((
            n_components,
            min_dist,
            spread,
            negative_sample_rate,
            n_epochs,
            learning_rate,
        )) = umap
        {
            let kw = PyDict::new(py);
            kw.set_item("n_components", n_components)?;
            kw.set_item("min_dist", min_dist)?;
            kw.set_item("spread", spread)?;
            kw.set_item("negative_sample_rate", negative_sample_rate)?;
            // rapids `rsc.tl.umap` names: n_epochs → maxiter, learning_rate → alpha.
            kw.set_item("maxiter", n_epochs)?;
            kw.set_item("alpha", learning_rate)?;
            kw.set_item("random_state", random_state)?;
            rsc_fn(py, "tl", "umap")?.call((adata,), Some(&kw))?;
        }
        Ok(())
    });

    if transfer_mode == "anndata_to_gpu" {
        // See `run`: restore host on success *and* failure so a translated PCA
        // error (F11) leaves `adata.X` host-resident for the recovery path.
        let restored = restore_host_anndata(py, adata, gpu_id);
        if let Err(ref e) = restored {
            log::warn!(target: "pyscx.accel", "fused pipeline: restore_host_anndata failed after dispatch: {e}");
        }
        result.and(restored)?;
    } else {
        result?;
    }

    stamp(py, adata, "pca", gpu_id, transfer_mode)?;
    stamp(py, adata, "neighbors", gpu_id, transfer_mode)?;
    if umap.is_some() {
        stamp(py, adata, "umap", gpu_id, transfer_mode)?;
        stamp(py, adata, "pca_neighbors_umap", gpu_id, transfer_mode)?;
    } else {
        stamp(py, adata, "pca_neighbors", gpu_id, transfer_mode)?;
    }
    Ok(())
}

/// Build a `rapids_singlecell` submodule function handle, e.g.
/// `rsc_fn(py, "pp", "pca")`.
pub(crate) fn rsc_fn<'py>(
    py: Python<'py>,
    submodule: &str,
    func: &str,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("rapids_singlecell")?
        .getattr(submodule)?
        .getattr(func)
}

/// Call `rsc.pp.pca` on the (GPU-resident) `adata`, translating rapids'
/// all-zero-gene rejection into an actionable scx-level message (report F11).
///
/// rapids-singlecell raises a bare `ValueError: There are genes with zero
/// expression. Please remove them before running PCA.` when any gene is zero
/// across all cells. SCX's CPU PCA tolerates such genes, so a user moving a raw
/// (unfiltered) matrix to `device="gpu"` hits an opaque rapids error with no
/// scx-side hint. We catch that specific case and re-raise pointing at
/// `filter_genes(min_cells=1)`; any other error propagates unchanged. Shared by
/// the standalone `pca` op and the fused `pca_neighbors[_umap]` pipeline so the
/// translation lives in exactly one place.
pub(crate) fn call_rsc_pca(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    kw: &Bound<'_, PyDict>,
) -> PyResult<()> {
    rsc_fn(py, "pp", "pca")?
        .call((adata,), Some(kw))
        .map(|_| ())
        .map_err(|err| {
            if err.is_instance_of::<PyValueError>(py) && err.to_string().contains("zero expression")
            {
                PyValueError::new_err(format!(
                    "GPU PCA (rapids-singlecell) rejects genes with zero expression across all \
                     cells, unlike the CPU PCA path. Drop all-zero genes first — e.g. \
                     `pyscx.accel.filter_genes(adata, min_cells=1)`, or select highly-variable \
                     genes with `pyscx.accel.highly_variable_genes(...)` — then retry the GPU \
                     pipeline. (underlying rapids error: {err})"
                ))
            } else {
                err
            }
        })
}

/// Convenience: a fresh kwargs dict.
pub(crate) fn kwargs(py: Python<'_>) -> Bound<'_, PyDict> {
    PyDict::new(py)
}
