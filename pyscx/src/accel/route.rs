//! Surfacing the accelerator execution route to Python.
//!
//! Every DE call stamps a [`scx_accel::route::AccelExecutionInfo`] onto its
//! result (GPU routes are stamped authoritatively inside `scx-accel`; CPU
//! routes are stamped at the pyscx dispatch point where the input layout is
//! known). These helpers serialise that info into `adata.uns["scx_accel"][op]`
//! so users and benchmarks can see exactly which route ran and why any
//! fallback happened.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scx_accel::route::{AccelExecutionInfo, FallbackReason};

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

/// Fill in the device-level fallback reason for a CPU route from the user's
/// `device` request and GPU availability. GPU routes keep the reason
/// `scx-accel` already stamped (`None` / `NoCscSidecar`).
pub(crate) fn finalize_exec_info(
    mut info: AccelExecutionInfo,
    device: &str,
    gpu_available: bool,
) -> AccelExecutionInfo {
    if info.route.is_gpu() {
        return info;
    }
    info.fallback_reason = if device == "cpu" {
        FallbackReason::UserForcedCpu
    } else if !gpu_available {
        FallbackReason::NoCuda
    } else {
        // Auto/GPU requested with a GPU present, yet a CPU route ran — e.g.
        // prefer_format="csc" (no GPU CSC kernel) or a layout with no GPU
        // path. Records the layout as the reason rather than implying CUDA
        // was missing.
        FallbackReason::UnsupportedInputLayout
    };
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
