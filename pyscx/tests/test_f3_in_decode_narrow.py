"""In-decode narrow wiring for the eager `to_anndata` path.

The existing `test_f3_dtype_materialization.py` covers the public dtype/container
API surface (default zero-copy, dtype matrix, dense, layers, lossy, query). This
file adds coverage specific to the Phase-3 rewiring, where X and raw now narrow
**in-decode** (via the typed reader) instead of being built as f32 then retyped:

- raw (`adata.raw.X`) now respects `data_dtype` (previously left f32);
- the integer-narrow path works end-to-end for values above the uint16 range;
- the query (`obs_filter`) branch narrows correctly. It used to do so by casting
  the assembled f32 matrix; it now decodes at the requested dtype, because that
  route knows the plan before it collects. See `test_query_typed_collect.py` for
  the `>2**24` half, which only the typed decode can serve.
"""

import anndata
import numpy as np
import pandas as pd
import pyscx
import scipy.sparse as sp
import pytest


def _counts_adata(n_obs=40, n_vars=12, seed=0, max_count=200):
    rng = np.random.default_rng(seed)
    dense = rng.integers(0, max_count, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.4] = 0
    obs = pd.DataFrame(
        {"batch": pd.Categorical(rng.choice(["a", "b"], size=n_obs))},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)


def _write(tmp_dir, adata, name="f3nd.scx"):
    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)
    return path


# --------------------------------------------------------------------------
# raw now narrows in-decode (Phase-3 new behavior)
# --------------------------------------------------------------------------
#
# Both write paths persist `adata.raw`. These fixtures go through the
# file-ingest path (`from_h5ad`) — mirror `test_raw_round_trip.py`'s fixture.


def _write_with_raw(tmp_dir, name="raw_narrow"):
    rng = np.random.default_rng(1)
    dense = rng.integers(0, 50, size=(20, 12)).astype(np.float32)
    dense[rng.random((20, 12)) > 0.3] = 0.0
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(20)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(12)])
    adata = anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)
    raw_dense = rng.integers(0, 40, size=(20, 30)).astype(np.float32)
    raw_dense[rng.random((20, 30)) > 0.4] = 0.0
    raw_var = pd.DataFrame(index=[f"raw_gene_{i}" for i in range(30)])
    adata.raw = anndata.AnnData(X=sp.csr_matrix(raw_dense), var=raw_var)

    h5ad_in = str(tmp_dir / f"{name}.h5ad")
    adata.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / f"{name}.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)
    return scx_path, dense, raw_dense


def test_raw_narrowed_to_uint8(tmp_dir):
    # Previously raw was left f32 under a non-default plan (the post-assembly
    # retype only touched X/layers); the in-decode path now narrows raw too.
    path, dense, raw_dense = _write_with_raw(tmp_dir)
    rt = pyscx.open(path).to_anndata(data_dtype="uint8")
    assert rt.X.data.dtype == np.uint8
    assert rt.raw is not None
    assert rt.raw.X.data.dtype == np.uint8
    np.testing.assert_array_equal(
        rt.raw.X.toarray().astype(np.float64), raw_dense.astype(np.float64)
    )
    np.testing.assert_array_equal(
        rt.X.toarray().astype(np.float64), dense.astype(np.float64)
    )


# --------------------------------------------------------------------------
# integer-narrow end-to-end above the uint16 range
# --------------------------------------------------------------------------


def test_integer_narrow_above_u16(tmp_dir):
    # A value > 65535 (but <= 2**24 so it survives the f32 write path exactly).
    adata = _counts_adata(max_count=250)
    big = 100_000
    dense = adata.X.toarray()
    dense[0, 0] = big
    adata2 = anndata.AnnData(X=sp.csr_matrix(dense), obs=adata.obs, var=adata.var)
    path = _write(tmp_dir, adata2, "big_u16.scx")

    # uint32: exact, lossless in-decode.
    rt = pyscx.open(path).to_anndata(data_dtype="uint32")
    assert rt.X.data.dtype == np.uint32
    assert rt.X.toarray()[0, 0] == big
    np.testing.assert_array_equal(
        rt.X.toarray().astype(np.float64), dense.astype(np.float64)
    )

    # uint16 cannot represent 100_000 -> fail loud without allow_lossy.
    with pytest.raises(ValueError):
        pyscx.open(path).to_anndata(data_dtype="uint16")

    # allow_lossy narrows (wraps) without error.
    rt_lossy = pyscx.open(path).to_anndata(data_dtype="uint16", allow_lossy=True)
    assert rt_lossy.X.data.dtype == np.uint16


# --------------------------------------------------------------------------
# query (obs_filter) branch still narrows — now by decoding at the dtype
# --------------------------------------------------------------------------


def test_obs_filter_with_dtype(tmp_dir):
    adata = _counts_adata(max_count=250)
    path = _write(tmp_dir, adata, "obs_filter.scx")
    rt = pyscx.open(path).to_anndata(obs_filter="batch == 'a'", data_dtype="uint8")
    assert sp.issparse(rt.X) and rt.X.format == "csr"
    assert rt.X.data.dtype == np.uint8
    # Rows match the same filter applied in-memory.
    want = adata[adata.obs["batch"] == "a"].X.toarray().astype(np.float64)
    np.testing.assert_array_equal(rt.X.toarray().astype(np.float64), want)
