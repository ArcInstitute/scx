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

// NOTE: no `_sandbox_import_probe` pyfunction here, deliberately. An earlier
// revision registered one on the production module so tests could call the
// helper directly — and a probe that imports an arbitrary module name IS an
// `__import__` replacement, handed to exactly the sandboxed code this crate
// promises to contain (`pyscx.pyscx._sandbox_import_probe("os")` worked from
// inside the restricted frame; found in review, PR #471). The helper's two
// branches are instead covered through real entry points in
// `test_sandbox_exec.py`, by evicting a lazily-imported module from
// `sys.modules` before the sandboxed call.
