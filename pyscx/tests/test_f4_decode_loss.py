"""F4 — fail-loud guard for the silent u32→f32 decode loss above 2**24.

scx decodes on-disk integer counts to an f32 CSR before any dtype
materialization, so counts above 2**24 (16,777,216) — routine in pseudobulk
aggregation — are silently rounded. The read paths now consult the catalog's
per-shard ``value_max`` and fail loud (unless ``allow_lossy=True``) instead of
returning a corrupted matrix.

The complementary f32-stream narrowing gate (e.g. f16/int8 requests on values
that fit in f32) is covered in test_f3_dtype_materialization.py.
"""

import anndata
import numpy as np
import pandas as pd
import pyscx
import pytest
import scipy.sparse as sp

# An integer above 2**24 that IS exact in f32 (even numbers up to 2**25 are), so
# it survives the float32 AnnData X unrounded and lands on disk as a uint32 count
# with value_max > 2**24. The guard is conservative: it trips on value_max > 2**24
# whether or not each individual value happens to be f32-exact.
BIG = 20_000_000


def _big_count_adata(n_obs=6, n_vars=4, big=BIG):
    """Integer-count AnnData with one > 2**24 entry in a batch=='a' cell."""
    dense = np.zeros((n_obs, n_vars), dtype=np.float32)
    dense[0, 0] = float(big)  # cell 0 (batch 'a') carries the big count
    dense[1, 1] = 7.0
    dense[3, 2] = 12.0
    batch = ["a" if i % 2 == 0 else "b" for i in range(n_obs)]
    obs = pd.DataFrame(
        {"batch": pd.Categorical(batch)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)


def _float_adata(n_obs=6, n_vars=4):
    """Non-integer float AnnData → Float32 encoding (catalog value_max == 0)."""
    rng = np.random.default_rng(0)
    dense = (rng.random((n_obs, n_vars)).astype(np.float32) * 3.0) + 0.5
    dense[rng.random((n_obs, n_vars)) > 0.5] = 0.0
    obs = pd.DataFrame(
        {"batch": pd.Categorical(["a", "b"] * (n_obs // 2))},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)


def _write(tmp_dir, adata, name="f4.scx"):
    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)
    return path


# --------------------------------------------------------------------------
# Eager read: default and widening requests now fail loud
# --------------------------------------------------------------------------


def test_default_read_fails_loud_above_2pow24(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata()


def test_allow_lossy_escapes_default(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    rt = pyscx.open(path).to_anndata(allow_lossy=True)
    assert rt.X.data.dtype == np.float32
    # The value is rounded to the nearest f32 (2**24), documenting the loss.
    assert rt.X.max() == pytest.approx(float(np.float32(BIG)))


def test_widening_dtype_also_fails_loud(tmp_dir):
    # float64 *could* hold the value, but Phase-1 decode intermediates through
    # f32, so the read is still lossy — it must fail loud rather than silently
    # widen a rounded f32. (Lossless typed decode is the deferred Phase 2.)
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(data_dtype="float64")
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(data_dtype="int64")
    # allow_lossy accepts the rounding.
    rt = pyscx.open(path).to_anndata(data_dtype="float64", allow_lossy=True)
    assert rt.X.data.dtype == np.float64


def test_dense_container_fails_loud(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(container="dense")


# --------------------------------------------------------------------------
# Query paths (catalog is gone by PyQueryResult; guard uses QueryResult.max_value)
# --------------------------------------------------------------------------


def test_query_result_to_anndata_fails_loud(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    exp = pyscx.open(path)
    result = exp.query().filter_obs("batch == 'a'").collect()
    with pytest.raises(ValueError, match="allow_lossy"):
        result.to_anndata()


def test_query_result_to_csr_fails_loud(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    exp = pyscx.open(path)
    result = exp.query().filter_obs("batch == 'a'").collect()
    with pytest.raises(ValueError, match="allow_lossy"):
        result.to_csr()


def test_query_result_allow_lossy_escapes(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    exp = pyscx.open(path)
    result = exp.query().filter_obs("batch == 'a'").collect()
    x = result.to_csr(allow_lossy=True)
    assert sp.issparse(x)
    assert x.data.dtype == np.float32


def test_experiment_obs_filter_fails_loud(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(obs_filter="batch == 'a'")


# --------------------------------------------------------------------------
# No spurious failures
# --------------------------------------------------------------------------


def test_float_encoded_unaffected(tmp_dir):
    # Float encodings record value_max == 0 in the catalog, so the integer-only
    # guard never trips on log-normalized / continuous data.
    path = _write(tmp_dir, _float_adata(), name="f4_float.scx")
    rt = pyscx.open(path).to_anndata()
    assert rt.X.data.dtype == np.float32


def test_small_counts_still_zero_copy(tmp_dir):
    # value_max <= 2**24 → default read unchanged (f32 / i32).
    rng = np.random.default_rng(1)
    dense = rng.integers(0, 500, size=(8, 5)).astype(np.float32)
    dense[rng.random((8, 5)) > 0.5] = 0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    path = _write(tmp_dir, adata, name="f4_small.scx")
    rt = pyscx.open(path).to_anndata()
    assert rt.X.data.dtype == np.float32
    assert rt.X.indices.dtype == np.int32
    np.testing.assert_array_equal(rt.X.toarray(), adata.X.toarray())
