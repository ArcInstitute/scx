"""pyscx.to_h5ad / from_h5ad / from_h5mu / to_h5mu accept str,
os.PathLike (pathlib.Path), and a pyscx Experiment handle for the
source argument.

The bug was that the Rust bindings declared `path: &str`, so passing a
`pathlib.Path` raised `TypeError: argument 'path': '...' object cannot
be converted to 'PyString'`, and passing an open Experiment was also
rejected (the SKILL example was the canonical foot-gun).
"""
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _make_src_adata(n_obs=8, n_vars=4):
    x = sp.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    obs = pd.DataFrame(index=pd.Index([f"CELL_{i}" for i in range(n_obs)]))
    var = pd.DataFrame(
        {"gene_ids": [f"ENSG{i:07d}" for i in range(n_vars)]},
        index=pd.Index([f"GENE_{i}" for i in range(n_vars)]),
    )
    return ad.AnnData(X=x, obs=obs, var=var)


@pytest.fixture
def src_h5ad(tmp_path):
    p = tmp_path / "src.h5ad"
    _make_src_adata().write_h5ad(p)
    return p


def test_from_h5ad_accepts_str(src_h5ad, tmp_path):
    import pyscx
    pyscx.from_h5ad(str(src_h5ad), str(tmp_path / "out_str.scx"))
    assert (tmp_path / "out_str.scx").exists()


def test_from_h5ad_accepts_pathlib(src_h5ad, tmp_path):
    import pyscx
    pyscx.from_h5ad(src_h5ad, tmp_path / "out_path.scx")  # both Path objects
    assert (tmp_path / "out_path.scx").exists()


def test_to_h5ad_accepts_pathlib(src_h5ad, tmp_path):
    import pyscx
    scx_path = tmp_path / "src.scx"
    rt_path = tmp_path / "rt.h5ad"
    pyscx.from_h5ad(src_h5ad, scx_path)
    pyscx.to_h5ad(scx_path, rt_path)
    assert rt_path.exists()
    rt = ad.read_h5ad(rt_path)
    assert rt.n_obs == 8 and rt.n_vars == 4


def test_to_h5ad_accepts_experiment_handle(src_h5ad, tmp_path):
    """Was the canonical SKILL example before the F1 fix: passing the
    open Experiment as the first arg used to raise TypeError. With the
    wrapper, the Experiment's `.path` is auto-unwrapped."""
    import pyscx
    scx_path = tmp_path / "src.scx"
    rt_path = tmp_path / "rt.h5ad"
    pyscx.from_h5ad(src_h5ad, scx_path)
    exp = pyscx.open(scx_path)
    # This used to TypeError; now works.
    pyscx.to_h5ad(exp, rt_path)
    assert rt_path.exists()
    rt = ad.read_h5ad(rt_path)
    assert rt.n_obs == 8 and rt.n_vars == 4


def test_experiment_exposes_path_getter(src_h5ad, tmp_path):
    """The `.path` getter on PyExperiment is the load-bearing piece of
    the F1 wrapper. Make sure it returns a non-empty str matching the
    file we opened."""
    import pyscx
    scx_path = tmp_path / "src.scx"
    pyscx.from_h5ad(src_h5ad, scx_path)
    exp = pyscx.open(scx_path)
    assert isinstance(exp.path, str)
    assert Path(exp.path) == scx_path


def test_coerce_path_uses_fspath_not_str(src_h5ad, tmp_path):
    """`_coerce_path` must invoke `__fspath__` (via `os.fspath`), not
    `str()`. A PathLike whose `__str__` returns the default object
    repr would silently corrupt the path otherwise."""
    import os

    import pyscx

    class FsOnlyPath(os.PathLike):
        def __init__(self, p):
            self._p = str(p)

        def __fspath__(self):
            return self._p
        # Deliberately NO __str__ override — str(self) returns the
        # default "<...FsOnlyPath object at 0x...>" repr.

    out = tmp_path / "out_fspath.scx"
    pyscx.from_h5ad(FsOnlyPath(src_h5ad), FsOnlyPath(out))
    assert out.exists()


def test_from_h5ad_rejects_experiment_handle(src_h5ad, tmp_path):
    """`from_h5ad` converts an h5ad file to SCX, so passing an open
    SCX Experiment as the source is semantically wrong. The wrapper
    must refuse it with a clear TypeError rather than silently
    extracting `.path` and feeding an SCX file to the h5ad reader."""
    import pyscx

    scx_path = tmp_path / "src.scx"
    pyscx.from_h5ad(src_h5ad, scx_path)
    exp = pyscx.open(scx_path)
    with pytest.raises(TypeError, match="SCX Experiment"):
        pyscx.from_h5ad(exp, tmp_path / "should_not_exist.scx")


def test_from_h5mu_rejects_experiment_handle(src_h5ad, tmp_path):
    """Same contract as `from_h5ad`: source must be a file path."""
    import pyscx

    scx_path = tmp_path / "src.scx"
    pyscx.from_h5ad(src_h5ad, scx_path)
    exp = pyscx.open(scx_path)
    with pytest.raises(TypeError, match="SCX Experiment"):
        pyscx.from_h5mu(exp, tmp_path / "should_not_exist.scx")


def test_to_h5ad_bad_type_raises_scx_specific(tmp_path):
    """E2 regression: when callers pass something that isn't a
    str / os.PathLike / pyscx.Experiment, the wrapper must raise a
    scx-named TypeError naming the expected types and the actual one,
    instead of the bare `os.fspath` message."""
    import pyscx
    with pytest.raises(TypeError, match=r"pyscx expects .* got list"):
        pyscx.to_h5ad([], tmp_path / "x.h5ad")
    with pytest.raises(TypeError, match=r"pyscx expects .* got dict"):
        pyscx.from_h5ad({}, tmp_path / "x.scx")
