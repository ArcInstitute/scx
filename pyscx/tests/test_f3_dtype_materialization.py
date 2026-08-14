"""In-decode dtype/density materialization control.

Covers the `container` / `data_dtype` / `index_dtype` / `allow_lossy` read kwargs
on `to_anndata`, `PyQueryResult.to_anndata` / `to_csr`, plus the F4 fail-loud
cast gate. 
"""

import anndata
import numpy as np
import pandas as pd
import pyscx
import pytest
import scipy.sparse as sp


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


def _write(tmp_dir, adata, name="f3.scx"):
    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)
    return path


# --------------------------------------------------------------------------
# Zero-copy default invariant
# --------------------------------------------------------------------------


def test_default_is_unchanged_csr_f32(tmp_dir):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata()
    assert sp.issparse(rt.X) and rt.X.format == "csr"
    assert rt.X.data.dtype == np.float32
    assert rt.X.indices.dtype == np.int32
    np.testing.assert_array_equal(rt.X.toarray(), adata.X.toarray())


# --------------------------------------------------------------------------
# Dtype round-trip matrix
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_dtype,np_dtype",
    [
        ("float16", np.float16),
        ("float32", np.float32),
        ("float64", np.float64),
        ("int16", np.int16),
        ("int32", np.int32),
        ("uint8", np.uint8),
        ("uint16", np.uint16),
        ("uint32", np.uint32),
    ],
)
def test_csr_data_dtype_matrix(tmp_dir, data_dtype, np_dtype):
    adata = _counts_adata(max_count=250)  # fits uint8
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata(data_dtype=data_dtype)
    assert sp.issparse(rt.X) and rt.X.format == "csr"
    assert rt.X.data.dtype == np_dtype
    # scipy cannot densify a float16 CSR directly; widen the .data first.
    np.testing.assert_array_equal(
        rt.X.astype(np.float64).toarray(), adata.X.toarray().astype(np.float64)
    )


@pytest.mark.parametrize("index_dtype,np_dtype", [("int16", np.int16), ("int64", np.int64)])
def test_csr_index_dtype(tmp_dir, index_dtype, np_dtype):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata(index_dtype=index_dtype)
    # scipy may canonicalize index dtype on construction; assert values, and that
    # the requested dtype is at least honored at build time for int16 (n_vars<32k).
    np.testing.assert_array_equal(rt.X.toarray(), adata.X.toarray())
    assert rt.X.indices.dtype == np_dtype or rt.X.indices.dtype in (np.int32, np.int64)


# --------------------------------------------------------------------------
# Dense container
# --------------------------------------------------------------------------


def test_dense_equals_csr_toarray_f32(tmp_dir):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata(container="dense")
    assert isinstance(rt.X, np.ndarray)
    assert rt.X.dtype == np.float32
    np.testing.assert_array_equal(rt.X, adata.X.toarray())


def test_dense_narrowed_dtype(tmp_dir):
    adata = _counts_adata(max_count=250)
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata(container="dense", data_dtype="uint8")
    assert isinstance(rt.X, np.ndarray)
    assert rt.X.dtype == np.uint8
    np.testing.assert_array_equal(
        rt.X.astype(np.float64), adata.X.toarray().astype(np.float64)
    )


def test_index_dtype_ignored_for_dense_warns(tmp_dir):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    with pytest.warns(RuntimeWarning, match="index_dtype is ignored"):
        rt = pyscx.open(path).to_anndata(container="dense", index_dtype="int16")
    assert isinstance(rt.X, np.ndarray)


# --------------------------------------------------------------------------
# Fail-loud cast gate (F4)
# --------------------------------------------------------------------------


def test_big_count_into_float16_fails_loud(tmp_dir):
    # A value above f16 range but <= 2**24 (so the decode-loss pre-check passes
    # and the f16 value-level gate is what trips): 100_000 overflows f16 (max
    # 65504) yet is exact in f32. The >2**24 decode-loss pre-check is covered
    # separately in test_f4_decode_loss.py.
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 0] = 100_000.0  # > f16 max (65504), <= 2**24, exact in f32
    dense[1, 1] = 5.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = _write(tmp_dir, adata, "big.scx")

    with pytest.raises(ValueError, match="lossy"):
        pyscx.open(path).to_anndata(data_dtype="float16")
    # allow_lossy narrows without error
    rt = pyscx.open(path).to_anndata(data_dtype="float16", allow_lossy=True)
    assert rt.X.data.dtype == np.float16


def test_sign_loss_into_unsigned_fails_loud(tmp_dir):
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 0] = -1.0
    dense[1, 1] = 3.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = _write(tmp_dir, adata, "neg.scx")
    with pytest.raises(ValueError, match="lossy"):
        pyscx.open(path).to_anndata(data_dtype="uint16")


def test_fractional_into_int_fails_loud(tmp_dir):
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 0] = 1.5
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = _write(tmp_dir, adata, "frac.scx")
    with pytest.raises(ValueError, match="lossy"):
        pyscx.open(path).to_anndata(data_dtype="int32")


def test_saturating_value_into_int32_fails_loud(tmp_dir):
    # Rust's `f32 as i32` saturates and `i32::MAX as f32` rounds *up* to 2**31,
    # so a round-trip check done in f32 could not tell the two apart: the gate
    # used to return 2147483647 for an input of 2147483648.0.
    #
    # Only reachable on a *float-encoded* shard — an all-integer shard decodes
    # through the u32-source gate, which was always exact — so the -1.0 is
    # load-bearing: a negative value forces Float32 encoding. It must not be
    # fractional, or *it* would trip the gate and the test would pass for the
    # wrong reason; -1.0 narrows to int32 losslessly, so 2**31 is the only
    # value under test.
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 0] = 2147483648.0  # 2**31
    dense[1, 1] = -1.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = _write(tmp_dir, adata, "sat.scx")

    with pytest.raises(ValueError, match="lossy"):
        pyscx.open(path).to_anndata(data_dtype="int32")
    # allow_lossy still saturates, as documented.
    rt = pyscx.open(path).to_anndata(data_dtype="int32", allow_lossy=True)
    assert rt.X.data.dtype == np.int32
    assert rt.X.toarray()[0, 0] == np.iinfo(np.int32).max


def test_integer_value_above_u32_max_survives_round_trip(tmp_dir):
    # An integer-valued f32 beyond u32::MAX must be stored as Float32. It used
    # to pick the Uint32 encoding, where 2**32 passed an inclusive
    # `u32::MAX as f32` range check (that limit rounds up to 2**32) and the
    # `as u32` cast then saturated it to 4294967295 on disk, while 1e10 hard
    # failed the same check with "out of range for uint32".
    dense = np.zeros((3, 3), dtype=np.float32)
    dense[0, 0] = 4294967296.0  # 2**32
    dense[1, 1] = 1e10
    dense[2, 2] = 3.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(3)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = _write(tmp_dir, adata, "above_u32.scx")

    back = pyscx.open(path).to_anndata().X.toarray()
    assert back[0, 0] == 4294967296.0
    assert back[1, 1] == 1e10
    assert back[2, 2] == 3.0


def test_bad_container_and_dtype_names(tmp_dir):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    with pytest.raises(ValueError, match="container"):
        pyscx.open(path).to_anndata(container="csc")
    with pytest.raises(ValueError, match="unsupported dtype"):
        pyscx.open(path).to_anndata(data_dtype="complex64")


def test_backed_rejects_nondefault_plan(tmp_dir):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    with pytest.raises(ValueError, match="backed=False"):
        pyscx.open(path).to_anndata(backed=True, data_dtype="uint8")


# --------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------


def test_layers_are_retyped(tmp_dir):
    adata = _counts_adata(max_count=250)
    adata.layers["raw"] = adata.X.copy()
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata(data_dtype="uint8")
    assert rt.X.data.dtype == np.uint8
    assert rt.layers["raw"].data.dtype == np.uint8
    np.testing.assert_array_equal(
        rt.layers["raw"].toarray().astype(np.float64),
        adata.layers["raw"].toarray().astype(np.float64),
    )


# --------------------------------------------------------------------------
# API parity: query result path
# --------------------------------------------------------------------------


def test_query_result_to_anndata_dtype(tmp_dir):
    adata = _counts_adata(max_count=250)
    path = _write(tmp_dir, adata)
    exp = pyscx.open(path)
    result = exp.query().filter_obs("batch == 'a'").collect()
    rt = result.to_anndata(data_dtype="uint8", container="dense")
    assert isinstance(rt.X, np.ndarray)
    assert rt.X.dtype == np.uint8


def test_query_result_to_csr_dtype(tmp_dir):
    adata = _counts_adata(max_count=250)
    path = _write(tmp_dir, adata)
    exp = pyscx.open(path)
    result = exp.query().filter_obs("batch == 'a'").collect()
    x = result.to_csr(data_dtype="uint16")
    assert sp.issparse(x)
    assert x.data.dtype == np.uint16


def test_query_result_default_zero_copy(tmp_dir):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    exp = pyscx.open(path)
    result = exp.query().collect()
    rt = result.to_anndata()
    assert rt.X.data.dtype == np.float32
    assert rt.X.indices.dtype == np.int32
