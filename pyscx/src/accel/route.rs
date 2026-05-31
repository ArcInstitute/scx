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

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_accel::route::{plan_de_route, AccelExecutionInfo, DeviceRequest, InputLayout};

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
/// rather than implying CUDA was absent. `v2`/`v3` are irrelevant on CPU and
/// passed `false` (the planner's CPU branch ignores them).
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
        false,
        false,
        csc_available,
    );
    info.chunk_size = chunk_size;
    info
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

/// Merge `info` into `adata.uns["scx_accel"][op]`, creating the `scx_accel`
/// dict if absent. Non-destructive across ops run on the same AnnData.
pub(crate) fn write_accel_route(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    op: &str,
    info: &AccelExecutionInfo,
) -> PyResult<()> {
    let uns = adata.getattr("uns")?;
    let existing = uns.call_method1("get", ("scx_accel",)).ok();
    let scx_accel: Bound<'_, PyDict> = match existing {
        Some(obj) if !obj.is_none() => match obj.cast_into::<PyDict>() {
            Ok(d) => d,
            Err(_) => {
                let d = PyDict::new(py);
                uns.set_item("scx_accel", &d)?;
                d
            }
        },
        _ => {
            let d = PyDict::new(py);
            uns.set_item("scx_accel", &d)?;
            d
        }
    };
    scx_accel.set_item(op, exec_info_to_pydict(py, info)?)?;
    Ok(())
}
