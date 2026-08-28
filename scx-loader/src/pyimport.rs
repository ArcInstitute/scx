//! Frame-insensitive Python module imports (shared by every pyo3 crate in
//! the workspace — scx-loader's `python` feature and pyscx).
//!
//! pyo3's `Python::import` lowers to CPython's `PyImport_Import`, which
//! resolves `__import__` from the **innermost Python frame's builtins**. When
//! a native entry point is called directly from code `exec`'d under
//! restricted globals — a pipeline runner's sandbox whose `__builtins__`
//! deliberately omits `__import__` — every call-time `py.import(...)`
//! inherits that restricted dict and dies with `KeyError: '__import__'`.
//!
//! [`import_module`] never consults the calling frame: it resolves from
//! `sys.modules` first (`PyImport_GetModule`), then falls back to the core
//! import machinery (`PyImport_ImportModuleLevelObject` with NULL globals) —
//! the C entry the `__import__` builtin itself wraps. Dotted names return the
//! leaf module, matching both `importlib.import_module` and the
//! `Python::import` behaviour the call sites were written against.
//!
//! Every call-time import in the workspace's pyo3 code MUST go through this
//! helper; the workspace `clippy.toml` `disallowed-methods` entry rejects
//! bare `Python::import` / `PyModule::import` so the class of breakage
//! cannot return. Covered end to end by `pyscx/tests/test_sandbox_exec.py`,
//! which runs the runner-facing entry points inside
//! `exec(code, {"__builtins__": <dict without __import__>})`.

use pyo3::exceptions::PyModuleNotFoundError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyModule, PyString};

/// Import `name`, immune to the calling frame's (possibly restricted)
/// builtins. Drop-in replacement for `py.import(name)`.
pub fn import_module<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyModule>> {
    let name_obj = PyString::new(py, name);

    // Fast path: an already-imported module straight out of `sys.modules`.
    // In practice this hits for everything the hot paths need (numpy, scipy,
    // pandas, pyarrow, anndata, warnings, builtins) — they are imported long
    // before an AnnData exists.
    if let Some(module) = sys_modules_get(py, &name_obj)? {
        return Ok(module);
    }

    // Core import machinery. NULL globals/locals never consult frame
    // builtins; level 0 means absolute import. For a dotted name this
    // returns the TOP-LEVEL package (the `__import__` contract with an
    // empty fromlist); the leaf lands in `sys.modules` as a side effect.
    let top = unsafe {
        ffi::PyImport_ImportModuleLevelObject(
            name_obj.as_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    };
    if top.is_null() {
        return Err(PyErr::fetch(py));
    }
    let top = unsafe { Bound::from_owned_ptr(py, top) };
    if !name.contains('.') {
        return Ok(top.cast_into::<PyModule>()?);
    }
    match sys_modules_get(py, &name_obj)? {
        Some(module) => Ok(module),
        // The import "succeeded" but the leaf is absent from sys.modules —
        // e.g. `pkg` imported fine but never registers `pkg.sub`. The real
        // `__import__` raises ModuleNotFoundError here too — WITH its `.name`
        // attribute set, which `pyscx::optional_deps::is_missing_module` keys
        // on to rewrite "not installed" errors. `new_err(msg)` alone would
        // leave `.name` as None and silently fork that behaviour (found in
        // review, PR #471), so construct the exception with the kwarg.
        None => {
            let exc = py
                .get_type::<PyModuleNotFoundError>()
                .call(
                    (format!("No module named '{name}'"),),
                    Some(&[("name", name)].into_py_dict(py)?),
                )?
                .unbind();
            Err(PyErr::from_value(exc.into_bound(py)))
        }
    }
}

/// `sys.modules[name]` without the import system: `PyImport_GetModule`.
/// Returns `Ok(None)` on a clean miss.
fn sys_modules_get<'py>(
    py: Python<'py>,
    name: &Bound<'py, PyString>,
) -> PyResult<Option<Bound<'py, PyModule>>> {
    let ptr = unsafe { ffi::PyImport_GetModule(name.as_ptr()) };
    if ptr.is_null() {
        if unsafe { !ffi::PyErr_Occurred().is_null() } {
            return Err(PyErr::fetch(py));
        }
        return Ok(None);
    }
    let module = unsafe { Bound::from_owned_ptr(py, ptr) };
    if module.is_none() {
        // `sys.modules[name] = None` (the stdlib convention for blocking a
        // module, used by pyscx's own optional-dep tests): fall through to
        // the core import, which raises the canonical
        // `ImportError: import of X halted; None in sys.modules`.
        return Ok(None);
    }
    // A module can sit in sys.modules while another thread is still executing
    // its body (`__spec__._initializing`). Real import semantics block on the
    // per-module lock until initialization finishes; handing the entry out
    // here would let a concurrent first import of a big optional stack
    // (scanpy, CuPy, rapids) observe a half-initialized module and die with a
    // misleading AttributeError (found in review, PR #471). Fall through to
    // the core import, which honors the lock.
    if module_is_initializing(&module) {
        return Ok(None);
    }
    Ok(Some(module.cast_into::<PyModule>()?))
}

/// `module.__spec__._initializing`, defaulting to false when either
/// attribute is missing or unreadable.
fn module_is_initializing(module: &Bound<'_, PyAny>) -> bool {
    module
        .getattr("__spec__")
        .ok()
        .filter(|spec| !spec.is_none())
        .and_then(|spec| spec.getattr("_initializing").ok())
        .and_then(|v| v.extract::<bool>().ok())
        .unwrap_or(false)
}
