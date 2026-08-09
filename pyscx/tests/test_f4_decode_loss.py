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


def test_widening_to_exact_dtype_now_lossless(tmp_dir):
    # The in-decode narrow assembles X directly from the
    # native u32 stream, so a > 2**24 integer count read into an exactly
    # representable dtype (float64 / int64 / uint32) now SUCCEEDS losslessly —
    # where the old f32-intermediate path failed loud. This is the documented
    # behavior change: a prior hard error becomes an exact read.
    path = _write(tmp_dir, _big_count_adata())
    for dt, np_dt in [("float64", np.float64), ("int64", np.int64), ("uint32", np.uint32)]:
        rt = pyscx.open(path).to_anndata(data_dtype=dt)
        assert rt.X.data.dtype == np_dt
        # The > 2**24 value round-trips exactly (no f32 rounding).
        assert int(rt.X.max()) == BIG


def test_narrow_f32_target_still_fails_loud(tmp_dir):
    # float32 (and the default) genuinely cannot hold a > 2**24 integer exactly,
    # so the conservative guard still fails loud unless allow_lossy.
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(data_dtype="float32")
    rt = pyscx.open(path).to_anndata(data_dtype="float32", allow_lossy=True)
    assert rt.X.data.dtype == np.float32


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


def test_query_result_tripped_guard_is_retryable(tmp_dir):
    """A tripped guard must leave the QueryResult intact.

    The guard's own message tells the caller to pass `allow_lossy=True`; that
    retry has to work on the *same* object. Previously both methods consumed
    the result before the guard ran, so the recommended retry raised
    "QueryResult already consumed by to_anndata() or to_csr()" instead.
    """
    path = _write(tmp_dir, _big_count_adata())
    exp = pyscx.open(path)

    result = exp.query().filter_obs("batch == 'a'").collect()
    with pytest.raises(ValueError, match="allow_lossy"):
        result.to_anndata()
    rt = result.to_anndata(allow_lossy=True)  # same object
    assert rt.X.max() == pytest.approx(float(np.float32(BIG)))

    result2 = exp.query().filter_obs("batch == 'a'").collect()
    with pytest.raises(ValueError, match="allow_lossy"):
        result2.to_csr()
    x = result2.to_csr(allow_lossy=True)  # same object
    assert sp.issparse(x)
    assert x.max() == pytest.approx(float(np.float32(BIG)))


def test_experiment_obs_filter_fails_loud(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(obs_filter="batch == 'a'")


# --------------------------------------------------------------------------
# Other eager X paths: var_names projection
# --------------------------------------------------------------------------


def test_var_names_only_fails_loud(tmp_dir):
    # A var_names-only read (no obs_filter) goes through the eager assembler,
    # which reads every shard — the whole-catalog max is exact and trips even
    # when the selected gene isn't the big-count one (conservative).
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(var_names=["gene_1"])
    rt = pyscx.open(path).to_anndata(var_names=["gene_0"], allow_lossy=True)
    assert rt.X.data.dtype == np.float32


# --------------------------------------------------------------------------
# Eager layers (lazy layers stay ungated, like backed)
# --------------------------------------------------------------------------


def _small_x_big_layer_adata(n_obs=4, n_vars=3):
    small = np.array([[3, 0, 5], [0, 7, 0], [1, 0, 0], [0, 0, 2]], dtype=np.float32)
    big = np.zeros((n_obs, n_vars), dtype=np.float32)
    big[0, 0] = float(BIG)
    ad = anndata.AnnData(
        X=sp.csr_matrix(small),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    ad.layers["counts"] = sp.csr_matrix(big)
    return ad


def test_eager_layers_fail_loud(tmp_dir):
    path = _write(tmp_dir, _small_x_big_layer_adata(), name="f4_layer.scx")
    # Eager materialization decodes the big-count layer -> fail loud.
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(eager=True)
    # allow_lossy escapes.
    rt = pyscx.open(path).to_anndata(eager=True, allow_lossy=True)
    assert rt.layers["counts"].data.dtype == np.float32


def test_default_lazy_layers_not_gated(tmp_dir):
    # Default (non-eager) read wraps layers lazily and never decodes them here,
    # so the read itself does not raise (X is small). Lazy layer decode is
    # ungated, consistent with backed reads. The corruption surfaces only on
    # later access — documented, not silently claimed as guarded.
    path = _write(tmp_dir, _small_x_big_layer_adata(), name="f4_layer2.scx")
    rt = pyscx.open(path).to_anndata()  # must not raise
    assert rt.X.data.dtype == np.float32


# --------------------------------------------------------------------------
# Eager to_mudata (per-modality X)
# --------------------------------------------------------------------------


def test_to_mudata_fails_loud(tmp_dir):
    mudata = pytest.importorskip("mudata")
    rna = anndata.AnnData(
        X=sp.csr_matrix(np.array([[3, 0], [0, 5]], dtype=np.float32)),
        var=pd.DataFrame(index=["rna0", "rna1"]),
    )
    big = np.zeros((2, 2), dtype=np.float32)
    big[0, 0] = float(BIG)
    adt = anndata.AnnData(X=sp.csr_matrix(big), var=pd.DataFrame(index=["adt0", "adt1"]))
    mu = mudata.MuData({"rna": rna, "adt": adt})
    path = str(tmp_dir / "f4_mm.scx")
    pyscx.from_mudata(mu, path)
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_mudata()
    mu_back = pyscx.open(path).to_mudata(allow_lossy=True)
    assert mu_back is not None


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
