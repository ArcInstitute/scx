//! CPU per-stage timing profiler surface (io / decode / reduction /
//! marshalling), the CPU-path twin of the GPU `gpu_profile_*` functions.
//!
//! Enabled by setting `SCX_CPU_PROFILE=1` before the process starts. The
//! counters are process-global; call [`cpu_profile_reset`] before a timed op
//! and [`cpu_profile_snapshot`] after so each snapshot reflects only that op.
//! This is the ranking oracle for the Phase-2 performance work — it tells you
//! whether a streaming accelerator op is decode/I-O-bound or
//! reduction/marshalling-bound at a given scale.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use scx_accel::cpu_profile;

fn stage_dict<'py>(py: Python<'py>, s: cpu_profile::StageStat) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("ms", s.ns as f64 / 1.0e6)?;
    d.set_item("count", s.count)?;
    d.set_item("bytes", s.bytes)?;
    Ok(d)
}

/// Snapshot the CPU per-stage timing profiler.
///
/// Returns a dict with `enabled` and one sub-dict per bucket (`io`,
/// `decode_scx1`, `decode_generic`, `reduction`, `marshalling`), each holding
/// `ms` (float milliseconds), `count`, and `bytes`. All buckets are zero unless
/// `SCX_CPU_PROFILE=1` was set at process start.
///
/// Example:
///     import os; os.environ["SCX_CPU_PROFILE"] = "1"   # before importing pyscx
///     pyscx.accel.cpu_profile_reset()
///     pyscx.accel.pca(adata, device="cpu")
///     prof = pyscx.accel.cpu_profile_snapshot()
#[pyfunction]
pub fn cpu_profile_snapshot(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let snap = cpu_profile::snapshot();
    let dict = PyDict::new(py);
    dict.set_item("enabled", snap.enabled)?;
    dict.set_item("io", stage_dict(py, snap.io)?)?;
    dict.set_item("decode_scx1", stage_dict(py, snap.decode_scx1)?)?;
    dict.set_item("decode_generic", stage_dict(py, snap.decode_generic)?)?;
    dict.set_item("reduction", stage_dict(py, snap.reduction)?)?;
    dict.set_item("marshalling", stage_dict(py, snap.marshalling)?)?;
    Ok(dict.into_any().unbind())
}

/// Reset the CPU per-stage timing profiler counters to zero.
///
/// Call between benchmark runs so each snapshot reflects only the most recent
/// operation.
#[pyfunction]
pub fn cpu_profile_reset() -> PyResult<()> {
    cpu_profile::reset();
    Ok(())
}
