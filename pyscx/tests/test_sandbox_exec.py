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
    import pandas as pd

    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    run_sandboxed(
        "pyscx.set_uns(path, {'species': 'human', 'sandboxed': True})",
        pyscx=pyscx,
        path=path,
    )
    exp = pyscx.open(path)
    assert exp.read_uns()["sandboxed"] is True

    # A DataFrame value too: writing one calls `.dtype` / `.to_numpy()` on each
    # column and reading one constructs `pd.DataFrame`, both of which can drive
    # C-level lazy imports that resolve from the innermost Python frame's
    # builtins — the failure mode `_frame_safe` exists for.
    frame = pd.DataFrame(
        {"v": [1.0, 2.0], "c": pd.Categorical(["a", "b"], ordered=True)},
        index=pd.Index(["r0", "r1"], name="row"),
    )
    run_sandboxed(
        "pyscx.set_uns(path, {'table': frame})",
        pyscx=pyscx,
        path=path,
        frame=frame,
    )
    pd.testing.assert_frame_equal(
        pyscx.open(path).read_uns()["table"], frame, check_dtype=True
    )

    # modify_metadata too — with an actual obs replacement, so the
    # pandas/pyarrow interop path runs inside the sandbox (an earlier
    # revision claimed this entry point was covered while only set_uns was
    # called; found in review, PR #471).
    obs = pyscx.open(path).read_obs()
    obs["sandbox_flag"] = np.arange(len(obs), dtype=np.int32)
    run_sandboxed(
        "pyscx.modify_metadata(path, obs=obs)",
        pyscx=pyscx,
        path=path,
        obs=obs,
    )
    assert "sandbox_flag" in pyscx.open(path).read_obs().columns


def test_attach_obs_columns_under_restricted_exec(sandbox_env, synthetic_adata):
    """The DataFrame attach seam — same pandas/pyarrow interop path as
    modify_metadata's obs, through the positional attach instead."""
    import pandas as pd

    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    df = pd.DataFrame(
        {"attached_flag": np.arange(synthetic_adata.n_obs, dtype=np.int32)}
    )
    run_sandboxed(
        "pyscx.attach_obs_columns(path, df, positional=True)",
        pyscx=pyscx,
        path=path,
        df=df,
    )
    assert "attached_flag" in pyscx.open(path).read_obs().columns


def test_attach_var_columns_under_restricted_exec(sandbox_env, synthetic_adata):
    """The var-axis attach seam. Same pure-Python wrapper shape as its obs
    twin, so it must be sandbox-safe for the same reason: the pipeline steps
    that land per-gene annotations run under restricted `exec`."""
    import pandas as pd

    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    df = pd.DataFrame(
        {"attached_gene_flag": np.arange(synthetic_adata.n_vars, dtype=np.int32)}
    )
    run_sandboxed(
        "pyscx.attach_var_columns(path, df, positional=True)",
        pyscx=pyscx,
        path=path,
        df=df,
    )
    assert "attached_gene_flag" in pyscx.open(path).read_var().columns


def test_var_import_under_restricted_exec(sandbox_env, synthetic_adata, tmp_path):
    """The delimited var import, from inside the sandbox."""
    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    var_names = list(pyscx.open(path).read_var().index)
    table = tmp_path / "genes.csv"
    lines = ["var_names,gene_flag"]
    lines += [f"{g},{i}" for i, g in enumerate(var_names)]
    table.write_text("\n".join(lines) + "\n")
    run_sandboxed(
        "pyscx.var_import(path, table, key='var_names')",
        pyscx=pyscx,
        path=path,
        table=str(table),
    )
    assert "gene_flag" in pyscx.open(path).read_var().columns


def test_backed_and_lazy_fancy_indexing_under_restricted_exec(
    sandbox_env, synthetic_adata
):
    """Boolean-mask and integer-array __getitem__ on backed / lazy X.

    These selector paths stringified the numpy dtype (`str(dtype)`), which
    makes numpy's C code import `numpy._core._dtype` through the
    frame-sensitive path on every call — reproduced as KeyError:
    '__import__' by review on PR #471. They now read `dtype.kind` instead.
    """
    path = sandbox_env["outputs"]["result"]
    pyscx.write(synthetic_adata, path)
    mask = np.zeros(synthetic_adata.n_obs, dtype=bool)
    mask[:7] = True
    glb = run_sandboxed(
        # normalize_total on a backed AnnData swaps X for the lazy dataset
        # in place, so backed.X exercises ScxBackedSparseDataset first and
        # ScxLazyTransformedDataset after.
        "backed = pyscx.open(path).to_anndata(backed=True)\n"
        "sub_mask = backed.X[mask]\n"
        "sub_list = backed.X[[0, 3, 5]]\n"
        "pyscx.accel.normalize_total(backed, target_sum=10000.0)\n"
        "lazy_mask = backed.X[mask]\n"
        "lazy_list = backed.X[[1, 2]]\n"
        "shapes = (sub_mask.shape, sub_list.shape, lazy_mask.shape, lazy_list.shape)",
        pyscx=pyscx,
        path=path,
        mask=mask,
    )
    n_vars = synthetic_adata.n_vars
    assert glb["shapes"] == ((7, n_vars), (3, n_vars), (7, n_vars), (2, n_vars))


def test_core_import_fallback_under_restricted_exec(sandbox_env, synthetic_adata):
    """The helper's core-import fallback branch, driven through real entry
    points (there is deliberately no importable probe hook — an
    arbitrary-name importer on the production module would BE the
    `__import__` the sandbox removed; found in review, PR #471).

    Evicting a lazily-imported module from sys.modules forces the next
    sandboxed call through `PyImport_ImportModuleLevelObject`:
    `pyscx._frame_safe` (dotted, package child — also proves leaf-module
    semantics: the write only succeeds if the leaf came back) and
    `scipy.sparse` (dotted, third-party) both re-import from inside the
    restricted frame.
    """
    import sys

    path = sandbox_env["outputs"]["result"]

    sys.modules.pop("pyscx._frame_safe", None)
    run_sandboxed(
        "pyscx.pyscx.from_anndata(adata, outputs['result'])", **sandbox_env
    )
    assert pyscx.validate(path)
    assert "pyscx._frame_safe" in sys.modules

    # The pop is process-global: other test modules imported at collection
    # hold the original module object as a type-identity oracle
    # (`isinstance(x, sp.csr_matrix)`), and a raise before the re-import
    # would leave scipy.sparse missing for the rest of the session. Restore
    # the original object no matter what happens inside the sandbox.
    orig_scipy_sparse = sys.modules.pop("scipy.sparse", None)
    scipy_pkg = sys.modules.get("scipy")
    try:
        glb = run_sandboxed(
            "result = pyscx.open(path).to_anndata()\nn = result.n_obs",
            pyscx=pyscx,
            path=path,
        )
        assert glb["n"] == synthetic_adata.n_obs
        assert "scipy.sparse" in sys.modules
    finally:
        if orig_scipy_sparse is not None:
            sys.modules["scipy.sparse"] = orig_scipy_sparse
            # The re-import also rebound the parent package's `sparse`
            # attribute to the NEW module object; `import scipy.sparse as sp`
            # follows that attribute, so restore it too or later tests see
            # two different module objects.
            if scipy_pkg is not None:
                scipy_pkg.sparse = orig_scipy_sparse
