"""F1 regression: pyscx.to_h5ad / from_h5ad / from_h5mu / to_h5mu accept
str, os.PathLike (pathlib.Path), and a pyscx Experiment handle for the
source argument. Closes F1 from SCX-USER-REPORT-2026-05-19.md.

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
