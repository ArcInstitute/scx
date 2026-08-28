"""pyscx entry points must work under restricted-exec globals (no ``__import__``).

Pipeline runners (arc-reactor's ``run_scrna_pipeline`` ``python`` steps) execute
user code via ``exec(code, {"__builtins__": <allowlist dict>, ...})`` where the
builtins dict deliberately omits ``__import__`` — imports are forbidden and
modules are pre-injected. pyo3's ``Python::import`` lowers to CPython's
``PyImport_Import``, which resolves ``__import__`` from the **innermost Python
frame's builtins**, so a native pyscx function called directly from such a frame
inherits the restricted dict and dies with ``KeyError: '__import__'`` at its
first call-time import (observed: Chimera SLURM job 2853320, 2026-08-27,
``pyscx.from_anndata`` from an arc-reactor ``python`` step).

These tests pin the fix (``pyscx/src/pyimport.rs``: every call-time import goes
through a frame-insensitive helper). The builtins allowlist below mirrors
arc-reactor's ``build_restricted_builtins`` (arc-orchestrate-core
``analysis_pipeline/capabilities.py``): a plain dict of safe builtins, no
``__import__``, no ``getattr``, no ``open``.

Empirical note (Phase 0 of the fix): before the fix, the *native* entry points
(``pyscx.pyscx.from_anndata``, and native methods like ``exp.to_anndata()``)
failed here, while the pure-Python wrappers (``pyscx.write``, ``obs_import``,
``set_uns``) already passed — a wrapper pushes a Python frame whose builtins
come from pyscx's module globals, shielding the native imports underneath it.
That protection is an accident of call depth, not a contract; both spellings
are pinned.

The fix has three layers (all regression-covered here): the
``scx_loader::pyimport::import_module`` sweep for pyscx/scx-loader's own
imports, priming rust-numpy's lazy C-API + borrow-capsule inits at
``import pyscx`` time, and the ``pyscx._frame_safe`` trampoline for numpy's
own per-call ``numpy._core._dtype`` import on every ``dtype.name`` read.
numpy itself is NOT sandbox-safe (``str(x.dtype)`` in step code still
raises); the guarantee covers pyscx entry points only.
"""

import builtins as _builtins

import numpy as np
import pytest

import pyscx

# Mirrors arc-reactor's SAFE_BUILTINS allowlist (the subset that matters for
# these tests). Deliberately NO __import__, NO getattr, NO open.
_SAFE_BUILTIN_NAMES = [
    "bool", "bytes", "dict", "float", "int", "list", "object", "set", "str",
    "tuple", "type", "abs", "all", "any", "enumerate", "isinstance", "len",
    "map", "max", "min", "print", "range", "repr", "round", "sorted", "sum",
    "zip", "True", "False", "None", "Exception", "ValueError", "TypeError",
    "KeyError",
]


def _restricted_builtins():
    return {
        name: getattr(_builtins, name)
        for name in _SAFE_BUILTIN_NAMES
        if hasattr(_builtins, name)
    }


def run_sandboxed(code, **names):
    """Exec ``code`` the way a pipeline runner's ``python`` step does.

    ``exec`` with an explicit globals dict whose ``__builtins__`` lacks
    ``__import__``; everything the code needs is pre-injected via ``names``.
    Returns the globals dict so tests can pull results back out.
    """
    glb = {"__builtins__": _restricted_builtins(), **names}
    exec(compile(code, "<sandbox-step>", "exec"), glb)
    return glb


@pytest.fixture
def sandbox_env(synthetic_adata, tmp_path):
    """The pre-injected symbols an escape-hatch step would get."""
    return {
        "pyscx": pyscx,
        "adata": synthetic_adata,
        "outputs": {"result": str(tmp_path / "out.scx")},
    }


def test_native_from_anndata_under_restricted_exec(sandbox_env):
    """The exact failing spelling from SLURM job 2853320."""
    run_sandboxed(
        "pyscx.pyscx.from_anndata(adata, outputs['result'])",
        # Native module attribute access: inject the already-bound native
        # function too, in case a runner injects natives directly.
        **sandbox_env,
    )
    assert pyscx.validate(sandbox_env["outputs"]["result"])


def test_native_from_anndata_bound_directly(sandbox_env):
    """A runner may inject the bare native callable, not the pyscx module."""
    run_sandboxed(
        "from_anndata(adata, outputs['result'])",
        from_anndata=pyscx.pyscx.from_anndata,
        adata=sandbox_env["adata"],
        outputs=sandbox_env["outputs"],
    )
    assert pyscx.validate(sandbox_env["outputs"]["result"])


def test_write_wrapper_under_restricted_exec(sandbox_env):
    run_sandboxed("pyscx.write(adata, outputs['result'])", **sandbox_env)
    assert pyscx.validate(sandbox_env["outputs"]["result"])


def test_read_under_restricted_exec(sandbox_env, synthetic_adata):
    """Read symmetry: open + to_anndata from inside the sandbox."""
    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    glb = run_sandboxed(
        "exp = pyscx.open(path)\n"
        "result = exp.to_anndata()\n"
        "n = result.n_obs",
        pyscx=pyscx,
        path=path,
    )
    assert glb["n"] == synthetic_adata.n_obs


def test_obs_import_under_restricted_exec(sandbox_env, synthetic_adata, tmp_path):
    """R3: the in-place obs patch path must be sandbox-safe too."""
    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    table = tmp_path / "anno.csv"
    lines = ["cell_id,my_score"]
    lines += [f"cell_{i},{i * 0.5}" for i in range(synthetic_adata.n_obs)]
    table.write_text("\n".join(lines) + "\n")
    run_sandboxed(
        "pyscx.obs_import(path, table, key='cell_id')",
        pyscx=pyscx,
        path=path,
        table=str(table),
    )
    obs = pyscx.open(path).read_obs()
    assert "my_score" in obs.columns


def test_set_uns_and_modify_metadata_under_restricted_exec(
    sandbox_env, synthetic_adata
):
    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    run_sandboxed(
        "pyscx.set_uns(path, {'species': 'human', 'sandboxed': True})",
        pyscx=pyscx,
        path=path,
    )
    exp = pyscx.open(path)
    assert exp.read_uns()["sandboxed"] is True


def test_sandbox_import_probe_both_branches(tmp_path):
    """Unit-test the Rust helper's two branches through the probe hook.

    Branch 1 (sys.modules hit): numpy is already imported.
    Branch 2 (core-import fallback): a stdlib module we evict from
    sys.modules first — the fallback must import it without consulting the
    restricted frame's builtins.
    """
    import sys

    probe = pyscx.pyscx._sandbox_import_probe

    # Branch 1: already-imported module resolves from sys.modules.
    glb = run_sandboxed("m = probe('numpy')", probe=probe)
    assert glb["m"] is np or glb["m"].__name__ == "numpy"

    # Branch 2: evict a small, safely re-importable stdlib module. `colorsys`
    # is dependency-free and nothing in pyscx or the test stack holds it.
    sys.modules.pop("colorsys", None)
    glb = run_sandboxed("m = probe('colorsys')", probe=probe)
    assert glb["m"].__name__ == "colorsys"
    assert "colorsys" in sys.modules

    # Dotted names return the leaf module (importlib.import_module
    # semantics), matching what py.import gave callers before.
    glb = run_sandboxed("m = probe('scipy.sparse')", probe=probe)
    assert glb["m"].__name__ == "scipy.sparse"

    # A missing module raises ModuleNotFoundError, not KeyError.
    with pytest.raises(ModuleNotFoundError):
        run_sandboxed("probe('pyscx_no_such_module')", probe=probe)
