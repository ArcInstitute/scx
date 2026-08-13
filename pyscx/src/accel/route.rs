//! Surfacing the accelerator execution route to Python.
//!
//! Every DE call stamps a [`scx_accel::route::AccelExecutionInfo`] onto its
//! result. GPU routes are stamped authoritatively inside `scx-accel` (the
//! dispatch `match`es on the planned route); CPU routes call
//! [`scx_accel::route::plan_de_route`] at the pyscx dispatch point where the
//! input layout is known. Either way the route + fallback reason come from the
//! single planner, so there is no post-hoc reason inference here. These helpers
//! serialise that info into `adata.uns["scx_accel"][op]` so users and
//! benchmarks can see exactly which route ran and why any fallback happened.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_accel::route::{
    plan_de_route, plan_hvg_route, plan_nb_glm_route, plan_simple_gpu_route, AccelExecutionInfo,
    AccelRoute, DeviceRequest, FallbackReason, InputLayout,
};

/// Serialise an [`AccelExecutionInfo`] into a Python dict. `Option` fields map
/// to `None`/value.
pub(crate) fn exec_info_to_pydict<'py>(
    py: Python<'py>,
    info: &AccelExecutionInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("route", info.route.as_str())?;
    d.set_item("fallback_reason", info.fallback_reason.as_str())?;
    d.set_item("chunk_size", info.chunk_size)?;
    d.set_item("graph_replay", info.graph_replay)?;
    d.set_item("csc_available", info.csc_available)?;
    d.set_item("shards_decoded", info.shards_decoded)?;
    d.set_item("shards_uploaded", info.shards_uploaded)?;
    d.set_item("math_mode", info.math_mode)?;
    d.set_item("spmm_policy", info.spmm_policy)?;
    // rapids-route / device-handoff metadata.
    d.set_item("rapids_version", info.rapids_version.as_deref())?;
    d.set_item("cuml_version", info.cuml_version.as_deref())?;
    d.set_item("cupy_version", info.cupy_version.as_deref())?;
    d.set_item("transfer_mode", info.transfer_mode)?;
    d.set_item("device_id", info.device_id)?;
    d.set_item("bytes_uploaded", info.bytes_uploaded)?;
    d.set_item("n_shards_shufdelta_gpu", info.n_shards_shufdelta_gpu)?;
    d.set_item("resident_csr", info.resident_csr)?;
    Ok(d)
}

/// Map the pyscx `device` string to a [`DeviceRequest`] intent. `resolve_device`
/// collapses `"auto"` to CPU when no GPU is present (losing the user's intent),
/// so the planner — which needs to distinguish `NoCuda` from `UserForcedCpu` —
/// takes the raw intent from here. Validation/errors stay in `resolve_device`.
pub(crate) fn device_request(device: &str) -> DeviceRequest {
    if device == "cpu" {
        DeviceRequest::Cpu
    } else if device == "auto" {
        DeviceRequest::Auto
    } else {
        // "gpu" / "gpu:N"
        DeviceRequest::Gpu
    }
}

/// Build the execution info for a CPU DE dispatch via the single planner.
///
/// `gpu_eligible` is `false` for layouts the GPU has no kernel for (e.g.
/// `prefer_format="csc"` → no GPU CSC kernel), so a `device="auto"`/`"gpu"`
/// request on a GPU host records `FallbackReason::UnsupportedInputLayout`
/// rather than implying CUDA was absent.
pub(crate) fn cpu_exec_info(
    device: &str,
    layout: InputLayout,
    gpu_eligible: bool,
    csc_available: bool,
    chunk_size: Option<usize>,
) -> AccelExecutionInfo {
    let mut info = plan_de_route(
        device_request(device),
        layout,
        gpu_available(),
        gpu_eligible,
        csc_available,
    );
    info.chunk_size = chunk_size;
    info
}

/// Build the execution info for an HVG dispatch via the single planner.
///
/// `gpu_eligible` is `true` only for the `seurat_v3` flavor family (the one
/// flavor with a GPU kernel). `csc_available` is `true` when a CSC sidecar is
/// reachable for a single-batch run, routing GPU to the column-major reduce
/// (`gpu_csc_v3`) and CPU to `cpu_csc`. The caller passes the resolved `device`
/// string and the actual flavor/layout so the recorded route matches the code
/// that ran.
pub(crate) fn hvg_exec_info(
    device: &str,
    gpu_eligible: bool,
    csc_available: bool,
) -> AccelExecutionInfo {
    plan_hvg_route(
        device_request(device),
        gpu_available(),
        gpu_eligible,
        csc_available,
    )
}

/// Build the execution info for a single-route op (PCA / kNN / UMAP / Leiden /
/// preprocessing) via the generic planner. `gpu_eligible` reflects whether the
/// op's GPU library was actually usable at dispatch (cuVS / cuML / cuGraph /
/// cuSPARSE present); `gpu_route` / `cpu_route` are the op's CSR- or
/// dense-shaped route pair.
pub(crate) fn simple_exec_info(
    device: &str,
    gpu_eligible: bool,
    gpu_route: AccelRoute,
    cpu_route: AccelRoute,
) -> AccelExecutionInfo {
    plan_simple_gpu_route(
        device_request(device),
        gpu_available(),
        gpu_eligible,
        gpu_route,
        cpu_route,
    )
}

/// Build the execution info for a pseudobulk NB-GLM dispatch via
/// [`plan_nb_glm_route`]. `gpu_eligible` is `false` when the design width or
/// sample count exceeds the kernel's register bounds (`scx_gpu::GPU_NB_GLM_PMAX`
/// / `GPU_NB_GLM_NSUB_MAX`) — recorded as `UnsupportedDimensions`. Stage A is
/// always CSR (host-fed dense); never a rapids fallback.
pub(crate) fn nb_glm_exec_info(device: &str, gpu_eligible: bool) -> AccelExecutionInfo {
    plan_nb_glm_route(device_request(device), gpu_available(), gpu_eligible)
}

/// Build the execution info for a CPU-only op that has no GPU kernel at all
/// (gene-set scoring). `gpu_eligible=false` records `UserForcedCpu` for
/// `device="cpu"`, `NoCuda` when no GPU is present, and `UnsupportedInputLayout`
/// for an explicit GPU request on a GPU host — there is no GPU score_genes path.
pub(crate) fn cpu_only_exec_info(device: &str) -> AccelExecutionInfo {
    simple_exec_info(device, false, AccelRoute::CpuCsr, AccelRoute::CpuCsr)
}

/// Build the execution info for an op that can hand in-VRAM GPU compute to
/// rapids-singlecell via [`plan_rapids_route`].
/// `rapids_available` comes from the import probe, `fits_vram` from the VRAM
/// pre-flight; `gpu_route`/`cpu_route` are the op's native fallback routes.
#[allow(dead_code)] // first consumers land in Phase 1.3/1.4
pub(crate) fn rapids_exec_info(
    device: &str,
    rapids_available: bool,
    fits_vram: bool,
    gpu_route: AccelRoute,
    cpu_route: AccelRoute,
) -> AccelExecutionInfo {
    scx_accel::route::plan_rapids_route(
        device_request(device),
        gpu_available(),
        rapids_available,
        fits_vram,
        gpu_route,
        cpu_route,
    )
}

/// Whether a CUDA GPU is available (always `false` without the `gpu` feature).
pub(crate) fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        scx_accel::gpu_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Read `adata.uns["scx_accel"]` into a **fresh** dict, or an empty one.
///
/// Always a copy, never the live container — see [`store_accel_container`] for
/// why the pair must be read-modify-write rather than an in-place mutation.
fn load_accel_container<'py>(
    py: Python<'py>,
    adata: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let uns = adata.getattr("uns")?;
    if let Some(existing) = uns
        .call_method1("get", ("scx_accel",))
        .ok()
        .filter(|o| !o.is_none())
    {
        if let Ok(existing) = existing.cast::<PyDict>() {
            out.update(existing.as_mapping())?;
        }
    }
    Ok(out)
}

/// Write the container back with a **top-level** `uns["scx_accel"] = …`.
///
/// The top level is load-bearing. On an anndata *view*, `uns` is a `DictView`
/// that only overrides `__setitem__`, and it is a *shallow* copy — so mutating
/// the nested `scx_accel` dict in place writes straight into the **parent's**
/// `uns` and the view never sees the stamp. Setting the top-level key instead
/// goes through `DictView.__setitem__`, which is the documented copy-on-write
/// trigger. (Every accel op that writes back also runs
/// [`crate::accel::prepare_target`] first, so a backed view is already an
/// actual `AnnData` by the time we get here; this keeps the plain-scipy view
/// case — deliberately left to anndata — correct too.)
fn store_accel_container(adata: &Bound<'_, PyAny>, container: &Bound<'_, PyDict>) -> PyResult<()> {
    adata.getattr("uns")?.set_item("scx_accel", container)
}

/// Merge `info` into `adata.uns["scx_accel"][op]`, creating the `scx_accel`
/// dict if absent. Non-destructive across ops run on the same AnnData.
pub(crate) fn write_accel_route(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &str,
    info: &AccelExecutionInfo,
) -> PyResult<()> {
    let container = load_accel_container(py, adata)?;
    container.set_item(op, exec_info_to_pydict(py, info)?)?;
    store_accel_container(adata, &container)
}

/// Restores `uns["scx_accel"][op]` to its pre-op state unless committed.
///
/// # The contract it enforces
///
/// **`adata.uns["scx_accel"][op]` is present if and only if the op completed.**
///
/// Thirteen ops stamp their route *before* dispatch, deliberately: the route is
/// planned up front and a long backed/atlas-scale run otherwise gives no
/// in-flight signal (see [`announce_route`]). The cost was that a raise left
/// the stamp behind, so the metadata claimed an op ran when it didn't — and a
/// caller reading `uns["scx_accel"]` to confirm what happened got a false
/// positive on a result that was never produced.
///
/// # Restore, don't delete
///
/// Rollback puts back whatever was there *before* this attempt. A failing
/// re-run of an op that previously succeeded must not erase the good stamp, and
/// a failing *first* accel op must leave `uns` byte-identical to before —
/// including not creating the `scx_accel` container at all.
///
/// # Why `Drop` and not an explicit combinator
///
/// Thirteen sites with three to five exits apiece: a missed path is inevitable.
/// Rolling back by default makes the failure mode *a missing stamp after a
/// successful op*, which the ~20 existing `uns["scx_accel"]` assertions across
/// the test suite catch loudly. The alternative fails silently.
#[must_use = "an uncommitted RouteStamp rolls the route stamp back when dropped"]
pub(crate) struct RouteStamp<'py> {
    adata: Bound<'py, PyAny>,
    op: &'static str,
    /// `uns["scx_accel"][op]` as it was before this op stamped.
    prior: Option<Bound<'py, PyAny>>,
    /// `uns` carried no `"scx_accel"` key at all when we started.
    container_was_absent: bool,
    committed: std::cell::Cell<bool>,
}

impl<'py> RouteStamp<'py> {
    /// Snapshot the current stamp without writing one.
    ///
    /// For ops whose first stamp goes through a helper that is called more than
    /// once (`stamp_pca_route` re-stamps after GPU dispatch): open exactly one
    /// guard, at the earliest stamp, and leave the re-stamps as plain
    /// [`write_accel_route`] calls.
    pub(crate) fn begin(adata: &Bound<'py, PyAny>, op: &'static str) -> PyResult<Self> {
        let uns = adata.getattr("uns")?;
        let container = uns
            .call_method1("get", ("scx_accel",))
            .ok()
            .filter(|o| !o.is_none());
        let container_was_absent = container.is_none();
        let prior = container
            .and_then(|c| c.cast_into::<PyDict>().ok())
            .and_then(|c| c.get_item(op).ok().flatten());
        Ok(Self {
            adata: adata.clone(),
            op,
            prior,
            container_was_absent,
            committed: std::cell::Cell::new(false),
        })
    }

    /// Snapshot, then stamp `info`.
    pub(crate) fn write(
        py: Python<'py>,
        adata: &Bound<'py, PyAny>,
        op: &'static str,
        info: &AccelExecutionInfo,
    ) -> PyResult<Self> {
        let guard = Self::begin(adata, op)?;
        write_accel_route(py, adata, op, info)?;
        Ok(guard)
    }

    /// Keep the stamp. Takes `&self` so it can be called from any of several
    /// early-return branches without moving the guard out.
    pub(crate) fn commit(&self) {
        self.committed.set(true);
    }

    /// [`commit`](Self::commit) iff `res` is `Ok`; returns `res` unchanged.
    pub(crate) fn settle<T>(self, res: PyResult<T>) -> PyResult<T> {
        if res.is_ok() {
            self.commit();
        }
        res
    }
}

impl Drop for RouteStamp<'_> {
    fn drop(&mut self) {
        if self.committed.get() {
            return;
        }
        // `Bound<'py>` carries the GIL token, so `Drop` can call Python
        // directly — it can only run inside the `'py` scope. Never panics:
        // every step is best-effort with a warning, because unwinding out of
        // `Drop` while a `PyErr` is already propagating would lose the real
        // error.
        let py = self.adata.py();
        if let Err(e) = self.rollback(py) {
            log::warn!(
                target: "pyscx.accel",
                "{}: could not roll back the uns[\"scx_accel\"] route stamp after a failure \
                 ({e}); the recorded route may describe an op that did not complete",
                self.op,
            );
        }
    }
}

impl RouteStamp<'_> {
    fn rollback(&self, py: Python<'_>) -> PyResult<()> {
        if self.container_was_absent {
            // A failing first-ever accel op leaves `uns` exactly as it was.
            self.adata
                .getattr("uns")?
                .call_method1("pop", ("scx_accel", py.None()))?;
            return Ok(());
        }
        let container = load_accel_container(py, &self.adata)?;
        match &self.prior {
            Some(prior) => container.set_item(self.op, prior)?,
            // Absent before, absent after. `del_item` raises `KeyError` when
            // the op never got as far as stamping; that is a no-op, not a
            // failure.
            None => {
                let _ = container.del_item(self.op);
            }
        }
        store_accel_container(&self.adata, &container)
    }
}

/// Copy an already-stamped op's route dict from `uns["scx_accel"][from_op]` to
/// `uns["scx_accel"][to_op]`. Used by fused entries (e.g. `pca_neighbors`) whose
/// summary route should mirror what an underlying sequential op actually
/// recorded, rather than re-synthesizing one. No-op if `from_op` is absent.
pub(crate) fn copy_accel_route(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    from_op: &str,
    to_op: &str,
) -> PyResult<()> {
    let container = load_accel_container(py, adata)?;
    let Some(route) = container.get_item(from_op)? else {
        return Ok(());
    };
    container.set_item(to_op, route)?;
    store_accel_container(adata, &container)
}

/// Whether a planned route warrants a default-visible GPU→CPU fallback
/// `UserWarning`. Pure (no Python) so it can be unit-tested.
///
/// True only when the user **explicitly** asked for a GPU (`"gpu"` / `"gpu:N"`,
/// not `"auto"` — `auto`→CPU on a CPU host is expected), the planned route is a
/// CPU route, and the reason is a GPU-was-unusable reason. `NoRapids` is excluded
/// (the rapids path already emits its own richer one-shot warning via
/// [`super::rapids`]); `UserForcedCpu` / `None` / `NoCscSidecar` (still a GPU
/// route) are not fallbacks worth warning about. Note `resolve_device` already
/// hard-errors an explicit `device="gpu"` when no CUDA GPU is present, so in
/// practice this fires for GPU-present-but-unsupported-layout/-dimensions cases.
fn should_warn_gpu_fallback(device: &str, info: &AccelExecutionInfo) -> bool {
    // Negative match (forward-compatible): a *new* `FallbackReason` variant
    // added to `scx-accel` warns by default rather than silently passing
    // through. The excluded reasons are the non-fallbacks: `None` (took the
    // GPU), `UserForcedCpu` (the user asked for CPU), `NoRapids` (the rapids
    // path emits its own richer one-shot warning), and `NoCscSidecar` (still a
    // GPU route — CSR-direct instead of CSC-direct).
    device.starts_with("gpu")
        && !info.route.is_gpu()
        && !matches!(
            info.fallback_reason,
            FallbackReason::None
                | FallbackReason::UserForcedCpu
                | FallbackReason::NoRapids
                | FallbackReason::NoCscSidecar
        )
}

/// One-shot-per-`(op, reason)` registry for the fallback warning, so a loop of
/// per-gene/per-batch calls doesn't flood the user with duplicates.
static FALLBACK_WARNED: OnceLock<Mutex<HashSet<(&'static str, &'static str)>>> = OnceLock::new();

/// Announce the resolved accelerator route at the dispatch point.
///
/// - Emits an INFO log (`target: "pyscx.accel"`) naming the route, requested
///   device, and fallback reason — visible when the user raises the `pyscx`
///   logger to INFO (`logging.basicConfig(level=logging.INFO)`); silent
///   otherwise.
/// - Emits a one-shot `UserWarning` (always visible) when an explicit
///   `device="gpu"` request silently lands on a CPU route — so a misconfigured
///   GPU environment is surfaced rather than only stamped into
///   `uns["scx_accel"]`.
///
/// Call this *before* the heavy dispatch so the route is known at op start (the
/// long-running backed/atlas-scale path otherwise gives no in-flight signal —
/// see report P1). The route-at-start guarantee therefore holds for the ops
/// that plan their route from `(device, gpu_eligible)` up front — HVG, PCA,
/// UMAP, Leiden, kNN. **DE is the exception**: its route is decided inside
/// `scx-accel` during compute (CSC-sidecar detection, etc.), so DE announces on
/// completion — the warning still fires, just not at start.
pub(crate) fn announce_route(
    py: Python<'_>,
    op: &'static str,
    device: &str,
    info: &AccelExecutionInfo,
) {
    log::info!(
        target: "pyscx.accel",
        "{op}: route={} device={} fallback={}",
        info.route.as_str(),
        device,
        info.fallback_reason.as_str(),
    );
    if !should_warn_gpu_fallback(device, info) {
        return;
    }
    let reason = info.fallback_reason.as_str();
    {
        let set = FALLBACK_WARNED.get_or_init(|| Mutex::new(HashSet::new()));
        let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
        // `op` is always a static literal at every dispatch site, so the
        // registry key avoids a per-call `String` allocation.
        if !guard.insert((op, reason)) {
            return; // already warned for this (op, reason) this process
        }
    }
    let hint = match info.fallback_reason {
        FallbackReason::NoCuda => {
            "no CUDA GPU was detected — build with `--features hdf5,gpu` and ensure a GPU is \
             visible (CUDA_VISIBLE_DEVICES)"
        }
        FallbackReason::UnsupportedInputLayout => "this op has no GPU kernel for the input layout",
        FallbackReason::UnsupportedDimensions => {
            "the input dimensions exceed the GPU path's supported limit"
        }
        FallbackReason::PerfPolicy => "a performance policy selected CPU",
        FallbackReason::GpuRuntimeError => {
            "the GPU path failed at run time and a slower path produced the result — the \
             original device error was reported at the point of failure"
        }
        _ => "see adata.uns[\"scx_accel\"] for details",
    };
    let msg = format!(
        "pyscx.accel.{op}(device=\"{device}\"): GPU was requested but the op ran on CPU \
         (route={}, fallback_reason=\"{reason}\") — {hint}. The final route is recorded in \
         adata.uns[\"scx_accel\"][\"{op}\"].",
        info.route.as_str(),
    );
    if let Ok(warnings) = py.import("warnings") {
        let _ = warnings.call_method1(
            "warn",
            (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
        );
    }
}

/// One-shot-per-op registry for the materialized-CSC-sidecar warning (F10).
static MATERIALIZED_CSC_WARNED: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();

/// Warn once when a GPU-eligible DE request lands on the CSR-direct route
/// *because the input was materialized*, even though the source file had an
/// on-disk CSC sidecar (report F10).
///
/// `NoCscSidecar` on a GPU route is normally not worth warning about (most files
/// have no sidecar, and CSR-direct is still a valid GPU route — see
/// [`should_warn_gpu_fallback`]). The narrow trap this catches is: the user built
/// a CSC sidecar *specifically* for GPU-fast DE, then reached for the obvious
/// `exp.to_anndata()` (in-memory CSR) instead of `to_anndata(backed=True)`, and
/// silently lost the `gpu_csc_v3` route. The non-backed `Experiment.to_anndata`
/// path stamps `adata.uns["scx_source_has_csc_sidecar"] = True` when the source
/// file has a sidecar; we key off that hint so the warning fires only for the
/// materialized-from-a-CSC-file case (a backed input on a sidecar-less file takes
/// the same route but carries no hint, so it stays silent).
///
/// Both `device="gpu"`/`"gpu:N"` and the default `device="auto"` qualify — `auto`
/// is the more common path and equally loses the fast route here. The
/// `info.route.is_gpu()` guard means an `auto` request that resolved to a CPU
/// route (no GPU host) never warns, so widening to `auto` only adds the
/// genuinely-on-GPU-but-slow case.
pub(crate) fn warn_materialized_csc_sidecar(
    py: Python<'_>,
    op: &'static str,
    device: &str,
    adata: &Bound<'_, PyAny>,
    info: &AccelExecutionInfo,
) {
    // Any non-CPU request that actually ran on a GPU route qualifies (explicit
    // gpu / gpu:N / auto). `device="cpu"` never reaches a GPU route anyway.
    if device == "cpu" {
        return;
    }
    if !info.route.is_gpu() || !matches!(info.fallback_reason, FallbackReason::NoCscSidecar) {
        return;
    }
    // Only when the *source file* actually had a CSC sidecar (stamped at
    // materialization by `Experiment.to_anndata`). Absent hint → stay silent.
    let had_sidecar = adata
        .getattr("uns")
        .and_then(|uns| uns.call_method1("get", ("scx_source_has_csc_sidecar",)))
        .map(|v| v.is_truthy().unwrap_or(false))
        .unwrap_or(false);
    if !had_sidecar {
        return;
    }
    {
        let set = MATERIALIZED_CSC_WARNED.get_or_init(|| Mutex::new(HashSet::new()));
        let mut guard = set.lock().unwrap_or_else(|e| e.into_inner());
        if !guard.insert(op) {
            return; // already warned for this op this process
        }
    }
    let msg = format!(
        "pyscx.accel.{op}(device=\"gpu\"): the source file has a CSC sidecar, but X was \
         materialized to an in-memory CSR via to_anndata(), so GPU DE took the slower \
         gpu_csr_v3 route (fallback_reason=\"no_csc_sidecar\"). Re-open with \
         to_anndata(backed=True) to engage the gpu_csc_v3 fast route. The final route is \
         recorded in adata.uns[\"scx_accel\"][\"{op}\"]."
    );
    if let Ok(warnings) = py.import("warnings") {
        let _ = warnings.call_method1(
            "warn",
            (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::should_warn_gpu_fallback;
    use scx_accel::route::{AccelExecutionInfo, AccelRoute, FallbackReason};

    fn info(route: AccelRoute, reason: FallbackReason) -> AccelExecutionInfo {
        AccelExecutionInfo::new(route, reason)
    }

    #[test]
    fn warns_explicit_gpu_request_landing_on_cpu() {
        assert!(should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::CpuCsr, FallbackReason::NoCuda)
        ));
        assert!(should_warn_gpu_fallback(
            "gpu:1",
            &info(AccelRoute::CpuCsr, FallbackReason::UnsupportedInputLayout)
        ));
        assert!(should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::CpuCsc, FallbackReason::UnsupportedDimensions)
        ));
        assert!(should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::CpuCsr, FallbackReason::PerfPolicy)
        ));
    }

    #[test]
    fn no_warn_for_auto_or_cpu_requests() {
        // `auto`→CPU on a CPU host is expected, not a misconfiguration.
        assert!(!should_warn_gpu_fallback(
            "auto",
            &info(AccelRoute::CpuCsr, FallbackReason::NoCuda)
        ));
        assert!(!should_warn_gpu_fallback(
            "cpu",
            &info(AccelRoute::CpuCsr, FallbackReason::UserForcedCpu)
        ));
    }

    #[test]
    fn no_warn_for_gpu_route_or_benign_reasons() {
        // Took the GPU as asked.
        assert!(!should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::GpuCsr, FallbackReason::None)
        ));
        // CPU route but not a GPU-unusable reason.
        assert!(!should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::CpuCsr, FallbackReason::UserForcedCpu)
        ));
        // NoRapids has its own dedicated one-shot warning (super::rapids).
        assert!(!should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::CpuCsr, FallbackReason::NoRapids)
        ));
        // NoCscSidecar still runs on the GPU (CSR-direct), so no warning.
        assert!(!should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::GpuCsrV3, FallbackReason::NoCscSidecar)
        ));
    }

    /// The negative match in `should_warn_gpu_fallback` is what makes a new
    /// `FallbackReason` warn by default rather than pass silently. Pin it on
    /// the variant that exercised the property, so a future edit that turns
    /// the guard into a positive allow-list fails here.
    #[test]
    fn a_runtime_failure_landing_on_cpu_warns() {
        assert!(should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::CpuCsr, FallbackReason::GpuRuntimeError)
        ));
        assert!(should_warn_gpu_fallback(
            "gpu:1",
            &info(AccelRoute::CpuDense, FallbackReason::GpuRuntimeError)
        ));
        // Still on a GPU route (to_gpu_anndata's host-assemble fallback ends up
        // on the device either way), so this warning — which is specifically
        // "you asked for GPU and got CPU" — must stay silent. That path emits
        // its own warning naming the device failure.
        assert!(!should_warn_gpu_fallback(
            "gpu",
            &info(AccelRoute::GpuCsr, FallbackReason::GpuRuntimeError)
        ));
    }
}
