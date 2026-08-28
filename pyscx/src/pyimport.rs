//! Frame-insensitive Python module imports.
//!
//! The helper lives in `scx_loader::pyimport` (the lowest pyo3 crate in the
//! workspace, so scx-loader's own `python`-feature code can use it too); see
//! its module docs for the full why. Short version: pyo3's `Python::import`
//! resolves `__import__` from the innermost Python frame's builtins, so a
//! native entry point called directly from restricted-exec globals (a
//! pipeline runner's sandbox with no `__import__`) dies with
//! `KeyError: '__import__'` at its first call-time import.
//!
//! Every call-time import in this crate MUST go through [`import_module`];
//! the workspace `clippy.toml` `disallowed-methods` entry rejects bare
//! `Python::import` / `PyModule::import`. Covered end to end by
//! `pyscx/tests/test_sandbox_exec.py`.

use pyo3::prelude::*;
use pyo3::types::PyModule;

pub(crate) use scx_loader::pyimport::import_module;

/// `obj.dtype.name` without tripping numpy's frame-sensitive lazy import.
///
/// numpy's C code imports `numpy._core._dtype` on **every** `dtype.name` /
/// `str(dtype)` / `repr(dtype)` via `PyImport_Import`, which needs
/// `__import__` in the innermost Python frame's builtins — absent under
/// restricted-exec globals. Calling through the pure-Python
/// `pyscx._frame_safe.dtype_name` pushes a frame with real builtins, so
/// numpy's import resolves no matter who called us.
pub(crate) fn dtype_name_of(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = obj.py();
    import_module(py, "pyscx._frame_safe")?
        .getattr("dtype_name")?
        .call1((obj,))?
        .extract()
}

/// Test-only hook so `test_sandbox_exec.py` can exercise both branches of
/// [`import_module`] (a `sys.modules` hit and the core-import fallback) from
/// inside restricted-exec globals. Not public API.
#[pyfunction]
pub(crate) fn _sandbox_import_probe<'py>(
    py: Python<'py>,
    name: &str,
) -> PyResult<Bound<'py, PyModule>> {
    import_module(py, name)
}
