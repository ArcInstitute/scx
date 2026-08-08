//! Importing an optional Python dependency, and failing usefully when it is absent.
//!
//! `pyscx`'s hard dependencies are numpy / scipy / pyarrow / anndata (which
//! itself pulls in pandas and h5py). Everything else — `scanpy`, `scikit-misc`,
//! `pydeseq2`, `mudata`, `polars`, `formulaic` — is optional, imported at the
//! point of use rather than at module load.
//!
//! A bare `py.import("skmisc.loess")?` on such a path is technically correct and
//! practically useless: the user sees `ModuleNotFoundError: No module named
//! 'skmisc'`, which names neither the operation that failed nor anything they
//! can pip-install under that spelling. [`import_optional`] rewrites that into a
//! sentence with the op, the distribution name and the exact install command.
//!
//! # The distinction that matters
//!
//! It rewrites **only** when the module that was not found is the one asked for.
//! Anything else propagates untouched, and that is not a nicety:
//!
//! - `scikit-misc` ships a compiled extension. Against a mismatched numpy it is
//!   `skmisc.loess` that fails, not `skmisc` — reporting that as "scikit-misc is
//!   not installed, run `pip install 'pyscx[hvg]'`" sends the user to reinstall a
//!   package they already have and buries the one line that says what is wrong.
//! - `scanpy` imports numba, llvmlite, matplotlib and networkx. Any of those
//!   failing is a `ModuleNotFoundError` raised *through* `import scanpy`, and
//!   the user needs to see which one.
//!
//! `mudata.rs`'s `import_mudata` and `pseudobulk.rs`'s pydeseq2 import both used
//! `map_err(|_| …)`, which swallows exactly these cases; the pattern to copy is
//! the one in `from_10x` (`lib.rs`), generalised here.
//!
//! Tested from `pyscx/tests/test_optional_deps.py`, not from a `#[cfg(test)]`
//! module here: `pyscx` links with `pyo3/extension-module`, so its lib-test
//! target has no libpython and anything calling `Python::attach` fails to link
//! (`undefined symbol: PyGILState_Ensure`) — which breaks `cargo test
//! --workspace` for the whole repo, not just this crate.

use pyo3::exceptions::PyModuleNotFoundError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// The extra that installs `mudata`, for `pyscx.from_mudata` / `to_mudata`.
pub(crate) const EXTRA_MUDATA: &str = "mudata";
/// The extra that installs `scanpy`, for `pyscx.from_10x`.
pub(crate) const EXTRA_10X: &str = "10x";
/// The extra that installs `scanpy` for the accel ops that delegate to it on an
/// in-memory scipy / dense `X`.
pub(crate) const EXTRA_SCANPY: &str = "scanpy";
/// The extra that installs `scikit-misc`, whose loess drives `seurat_v3` HVG.
pub(crate) const EXTRA_HVG: &str = "hvg";
/// The extra that installs `pydeseq2`, for `pseudobulk_dex(backend="pydeseq2")`.
pub(crate) const EXTRA_PYDESEQ2: &str = "pydeseq2";
/// The extra that installs `polars`, for `output="polars"` on the DE frames.
pub(crate) const EXTRA_EVAL: &str = "eval";
/// The extra that installs `formulaic`, for a custom NB-GLM `design=`.
pub(crate) const EXTRA_NBGLM: &str = "nbglm";

/// Trailer appended to the message of every op that only needs scanpy because
/// `adata.X` is a materialised scipy / dense matrix. Naming the escape hatch
/// matters more than naming the extra: the backed path is both faster and the
/// project's actual recommendation, and it needs no optional package at all.
pub(crate) const BACKED_ESCAPE_HATCH: &str =
    "This op only needs scanpy because `adata.X` is an in-memory scipy/dense \
     matrix; on a backed or lazy X (`pyscx.open(path).to_anndata(backed=True)`) \
     it runs the scx-native streaming kernel and needs no scanpy.";

/// Import an optional dependency, or raise an error that says what to install.
///
/// * `module` — the import path, e.g. `"scanpy"` or `"skmisc.loess"`.
/// * `dist` — the **distribution** name as it appears on PyPI, which is not
///   always the import name (`skmisc` ships as `scikit-misc`).
/// * `extra` — the `pyscx` extra that installs it. Must exist in
///   `pyproject.toml`'s `[project.optional-dependencies]`;
///   `test_optional_deps.py::test_every_advertised_extra_is_declared` enforces
///   that by parsing both sides.
/// * `op` — what the user called, e.g. `pyscx.accel.log1p()`. Written into the
///   message because the traceback frame is Rust and says nothing.
/// * `hint` — an optional extra sentence (see [`BACKED_ESCAPE_HATCH`]).
pub(crate) fn import_optional_with_hint<'py>(
    py: Python<'py>,
    module: &str,
    extra: &str,
    op: &str,
    dist: &str,
    hint: Option<&str>,
) -> PyResult<Bound<'py, PyModule>> {
    py.import(module).map_err(|e| {
        if !is_missing_module(py, &e, module) {
            // A transitive dependency, or the submodule itself, failed to
            // import. That is a real diagnosis; do not relabel it.
            return e;
        }
        let imported_as = if dist == module {
            String::new()
        } else {
            format!(" (imported as `{module}`)")
        };
        let tail = match hint {
            Some(h) => format!("\n{h}"),
            None => String::new(),
        };
        PyModuleNotFoundError::new_err(format!(
            "{op} requires the `{dist}` package{imported_as}, which is not installed.\n\
             Install it with: pip install 'pyscx[{extra}]'{tail}"
        ))
    })
}

/// [`import_optional_with_hint`] with no trailing hint. The common case.
pub(crate) fn import_optional<'py>(
    py: Python<'py>,
    module: &str,
    extra: &str,
    op: &str,
    dist: &str,
) -> PyResult<Bound<'py, PyModule>> {
    import_optional_with_hint(py, module, extra, op, dist, None)
}

/// Whether `err` says the **distribution** is absent, as opposed to something
/// inside it having failed to load.
///
/// `ModuleNotFoundError` carries the missing module in `.name`. Importing
/// `skmisc.loess`:
///
/// | `.name` | means | verdict |
/// |---|---|---|
/// | `skmisc` | scikit-misc is not installed | rewrite — install the extra |
/// | `skmisc.loess` | installed, but its extension will not load | propagate |
/// | `numpy.<x>` | a dependency of it is broken | propagate |
///
/// So the test is equality against the **top-level** package only. Matching the
/// dotted name too would fold the second row into the first and send a user
/// with a working install off to reinstall it, hiding the ABI error that is the
/// only line saying what actually went wrong.
fn is_missing_module(py: Python<'_>, err: &PyErr, module: &str) -> bool {
    if !err.is_instance_of::<PyModuleNotFoundError>(py) {
        return false;
    }
    let Some(missing) = err
        .value(py)
        .getattr("name")
        .ok()
        .and_then(|n| n.extract::<String>().ok())
    else {
        // No `.name` to go on (a hand-raised ModuleNotFoundError). Propagating
        // is the safe default: a wrong "not installed" is worse than a raw one.
        return false;
    };
    missing == module.split('.').next().unwrap_or(module)
}
