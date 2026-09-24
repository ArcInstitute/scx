# Python API (`pyscx`)

> Part of the [SCX API reference](README.md).

> [!NOTE]
> For the most up-to-date function signatures, see the
> [auto-generated Python API reference](../python_api.rst). For wrapped functions
> (`open`, `from_h5ad`, `obs_import`, …) the canonical docstring is the one on
> the **Python wrapper** — it is what `help()` and the rendered site show, and
> the wrapper's input coercions (`Experiment` paths, pandas Series masks) are
> part of the contract; the Rust `///` docs on those natives are pointers back
> at it. Unwrapped functions (`from_anndata`, `merge`, `compact`, …) render
> from the Rust docstrings via autodoc. A pytest guard
> (`pyscx/tests/test_docstring_coverage.py`) pins every native kwarg to an
> `Args:` entry on the wrapper docstring.

## Pages

- [Module-level functions](python-functions.md)
- [`Experiment`](python-experiment.md)
- [Queries and cloud handles](python-query.md)
- [`pyscx.accel`](python-accel.md)
- [Backed and lazy datasets](python-datasets.md)
- [ML training and tokenisation](python-training.md)

## Restricted-exec (sandbox) safety

pyscx entry points work when called from code `exec`'d under **restricted
globals whose `__builtins__` lack `__import__`** — the sandbox shape pipeline
runners use for escape-hatch `python` steps (imports forbidden, modules
pre-injected). This is a maintained guarantee, not an accident:

- CPython resolves `__import__` for C-level imports (`PyImport_Import`) from
  the **innermost Python frame's builtins**, so a native function called
  directly from such a frame would otherwise die with
  `KeyError: '__import__'` at its first call-time import.
- All call-time imports in pyscx and scx-loader go through the
  frame-insensitive `scx_loader::pyimport::import_module` (`sys.modules`
  lookup, then the core import machinery — neither consults frame builtins).
  A workspace `clippy.toml` `disallowed-methods` entry rejects bare
  `Python::import` / `PyModule::import`.
- Third-party lazy imports are handled too: rust-numpy's C-API and
  borrow-capsule inits are primed at `import pyscx` time. numpy re-imports
  `numpy._core._dtype` on **every** `dtype.name` / `str(dtype)` /
  `repr(dtype)`, so pyscx never stringifies a dtype on a native path: the
  write-path dtype-name reads route through the pure-Python trampoline
  `pyscx._frame_safe.dtype_name` (whose frame carries real builtins), and
  the backed/lazy `__getitem__` selector paths read `dtype.kind`, a plain C
  descriptor that triggers no import.

Covered surfaces (regression-tested in `pyscx/tests/test_sandbox_exec.py`,
which runs each one inside `exec(code, {"__builtins__": <no __import__>})`):
`pyscx.write`, the native `pyscx.pyscx.from_anndata`, `pyscx.obs_import`,
`pyscx.attach_obs_columns` (a positional DataFrame attach),
`pyscx.modify_metadata` (with an obs replacement) / `pyscx.set_uns`,
`open(...).to_anndata()` eager and `backed=True`, boolean-mask and
integer-array indexing on backed and lazy `X`, and the import helper's
core-import fallback (via `sys.modules` eviction — there is deliberately no
importable probe hook, since an arbitrary-name importer on the module would
re-create the `__import__` the sandbox removed).
Note that numpy itself is *not* sandbox-safe (`str(x.dtype)` in step code
will still raise); the guarantee covers pyscx's own entry points.
