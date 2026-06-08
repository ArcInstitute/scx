//! rapids-singlecell routing helpers (ACC-RUST-OPT-V4 Phase 1.3 / 1.4).
//!
//! In the **in-VRAM** regime, `device="gpu"` analysis ops hand off to
//! [rapids-singlecell](https://rapids-singlecell.readthedocs.io/) — the GPU
//! compute layer — rather than SCX's native CUDA kernels. rapids is a *detected
//! runtime dependency* (§4.3): present → route to it; absent → route CPU with a
//! one-shot diagnostic and `fallback_reason="no_rapids"`. The native GPU kernels
//! stay reachable for A/B + rollback via `SCX_FORCE_NATIVE_GPU=1` (a
//! transition-only override removed at the end of Phase 3) and for the
//! out-of-VRAM streaming moat (backed / lazy `X`, which never routes here).
//!
//! This module is GPU-gated; callers reference it from `#[cfg(feature = "gpu")]`
//! blocks only.

use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use scx_accel::route::{AccelExecutionInfo, AccelRoute, FallbackReason};

use super::gpu::{rapids_singlecell_info, ResolvedDevice};
use super::route::write_accel_route;

/// `SCX_FORCE_NATIVE_GPU` override — keep the native SCX GPU kernels instead of
/// routing in-VRAM GPU compute to rapids-singlecell. Transition-only A/B +
/// rollback switch (removed at the end of Phase 3).
pub(crate) fn force_native_gpu() -> bool {
    matches!(std::env::var("SCX_FORCE_NATIVE_GPU"), Ok(v) if v != "0" && !v.is_empty())
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
pub(crate) fn run(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &str,
    gpu_id: usize,
    call: impl FnOnce(Python<'_>, &Bound<'_, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    let transfer_mode = ensure_gpu_anndata(py, adata, gpu_id)?;
    with_device(py, gpu_id, || call(py, adata))?;
    stamp(py, adata, op, gpu_id, transfer_mode)
}

/// Run the fused PCA → neighbors [→ UMAP] rapids pipeline on the device AnnData
/// and stamp every stage (ACC-RUST-OPT-V4 Phase 1.4). `umap` carries the UMAP
/// stage params `(n_components, min_dist, spread, negative_sample_rate)` when the
/// UMAP stage should run; `None` runs PCA → neighbors only.
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
    umap: Option<(usize, f64, f64, usize)>,
) -> PyResult<()> {
    let transfer_mode = ensure_gpu_anndata(py, adata, gpu_id)?;
    with_device(py, gpu_id, || {
        let kw = PyDict::new(py);
        kw.set_item("n_comps", n_comps)?;
        kw.set_item("zero_center", zero_center)?;
        kw.set_item("random_state", random_state)?;
        rsc_fn(py, "pp", "pca")?.call((adata,), Some(&kw))?;

        let kw = PyDict::new(py);
        kw.set_item("n_neighbors", n_neighbors)?;
        kw.set_item("use_rep", use_rep)?;
        kw.set_item("random_state", random_state)?;
        rsc_fn(py, "pp", "neighbors")?.call((adata,), Some(&kw))?;

        if let Some((n_components, min_dist, spread, negative_sample_rate)) = umap {
            let kw = PyDict::new(py);
            kw.set_item("n_components", n_components)?;
            kw.set_item("min_dist", min_dist)?;
            kw.set_item("spread", spread)?;
            kw.set_item("negative_sample_rate", negative_sample_rate)?;
            kw.set_item("random_state", random_state)?;
            rsc_fn(py, "tl", "umap")?.call((adata,), Some(&kw))?;
        }
        Ok(())
    })?;

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

/// Convenience: a fresh kwargs dict.
pub(crate) fn kwargs(py: Python<'_>) -> Bound<'_, PyDict> {
    PyDict::new(py)
}
