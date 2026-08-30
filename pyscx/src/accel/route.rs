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
use scx_accel::route::{should_warn_gpu_fallback, AccelExecutionInfo, FallbackReason, RouteValue};

// The binding-agnostic routing layer lives in `scx_accel::route` (ORG-10.16-3):
// the exec-info builders (which take a validated [`DeviceRequest`], so the
// shared public API cannot be fed an unparsed string) and the runtime GPU
// availability flag. The `&str` shims below keep the ~30 pyscx dispatch sites
// reading unchanged; their precondition is that `resolve_device` already
// validated the string at the op entry — every pyscx accel op does, so the
// grammar-blind `DeviceRequest::from_device_str` intent parse is safe here.
pub(crate) use scx_accel::route::gpu_runtime_available as gpu_available;
use scx_accel::route::{DeviceRequest, InputLayout};

pub(crate) fn cpu_exec_info(
    device: &str,
    layout: InputLayout,
    gpu_eligible: bool,
    csc_available: bool,
    chunk_size: Option<usize>,
) -> AccelExecutionInfo {
    scx_accel::route::cpu_exec_info(
        DeviceRequest::from_device_str(device),
        layout,
        gpu_eligible,
        csc_available,
        chunk_size,
    )
}

pub(crate) fn hvg_exec_info(
    device: &str,
    gpu_eligible: bool,
    csc_available: bool,
) -> AccelExecutionInfo {
    scx_accel::route::hvg_exec_info(
        DeviceRequest::from_device_str(device),
        gpu_eligible,
        csc_available,
    )
}

pub(crate) fn simple_exec_info(
    device: &str,
    gpu_eligible: bool,
    gpu_route: scx_accel::AccelRoute,
    cpu_route: scx_accel::AccelRoute,
) -> AccelExecutionInfo {
    scx_accel::route::simple_exec_info(
        DeviceRequest::from_device_str(device),
        gpu_eligible,
        gpu_route,
        cpu_route,
    )
}

pub(crate) fn nb_glm_exec_info(device: &str, gpu_eligible: bool) -> AccelExecutionInfo {
    scx_accel::route::nb_glm_exec_info(DeviceRequest::from_device_str(device), gpu_eligible)
}

pub(crate) fn cpu_only_exec_info(device: &str) -> AccelExecutionInfo {
    scx_accel::route::cpu_only_exec_info(DeviceRequest::from_device_str(device))
}

/// Serialise an [`AccelExecutionInfo`] into a Python dict — a loop over
/// [`AccelExecutionInfo::fields`], the shared wire serialization, so the key
/// set and order cannot drift from what rscx emits. `Option` fields map to
/// `None`/value.
pub(crate) fn exec_info_to_pydict<'py>(
    py: Python<'py>,
    info: &AccelExecutionInfo,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (key, value) in info.fields() {
        match value {
            RouteValue::Str(v) => d.set_item(key, v)?,
            RouteValue::OptStr(v) => d.set_item(key, v)?,
            RouteValue::OptBool(v) => d.set_item(key, v)?,
            RouteValue::OptUsize(v) => d.set_item(key, v)?,
            RouteValue::OptU32(v) => d.set_item(key, v)?,
            RouteValue::OptU64(v) => d.set_item(key, v)?,
        }
    }
    Ok(d)
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
    if let Ok(warnings) = crate::pyimport::import_module(py, "warnings") {
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
    if let Ok(warnings) = crate::pyimport::import_module(py, "warnings") {
        let _ = warnings.call_method1(
            "warn",
            (msg, py.get_type::<pyo3::exceptions::PyUserWarning>()),
        );
    }
}
