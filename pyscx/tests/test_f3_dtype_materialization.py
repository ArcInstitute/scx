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


@pytest.mark.parametrize("index_dtype", ["int16", "int64"])
def test_csr_index_dtype(tmp_dir, index_dtype):
    adata = _counts_adata()
    path = _write(tmp_dir, adata)
    rt = pyscx.open(path).to_anndata(index_dtype=index_dtype)
    np.testing.assert_array_equal(rt.X.toarray(), adata.X.toarray())
    # Neither width survives on a small matrix, and for the same reason:
    # `csr_matrix` resolves the index dtype from `max(nnz, n_rows)` and ignores
    # what it was handed, so int16 is upcast and int64 is *downcast* — both to
    # int32. The narrow int16 buffer is still built and range-gated on the way
    # through. See `test_scipy_resolves_the_index_dtype_from_the_contents`.
    assert rt.X.indices.dtype == np.int32


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


# --------------------------------------------------------------------------
# Automatic int64 indices above 2**31 nonzeros
# --------------------------------------------------------------------------


def _run_with_threshold(script, threshold):
    """Run `script` in a child with `SCX_EAGER_INT64_NNZ_THRESHOLD` set.

    A child rather than `monkeypatch`, because the threshold is resolved once
    per process (`OnceLock`) — a same-process flip would be read after the
    first eager read had already fixed it.
    """
    import os
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "SCX_EAGER_INT64_NNZ_THRESHOLD": str(threshold)},
    )


def test_scipy_resolves_the_index_dtype_from_the_contents():
    """The premise the auto-widen exists for, pinned against scipy itself.

    `csr_matrix` resolves **one** index dtype for `indices` and `indptr` from
    `max(nnz, n_rows)`, not from the arrays it was handed. Two consequences,
    and the widen depends on both:

    * above `i32::MAX` it picks int64, so the int32 `indices` array pyscx hands
      it is **copied** — while that array is still alive. That is the
      16.3 B/nnz assembly transient measured on a 2.65e9-nonzero file, and
      what decoding int64 directly removes.
    * at or below it, int64 inputs are **downcast** — so widening a matrix that
      does not need it would *add* a copy. That is why the widen is gated on
      the same threshold rather than applied whenever it might help.

    Three elements are enough to state the rule; building a real 2**31-nonzero
    matrix is not an option in a test. If this ever changes, the widen in
    `to_anndata` stops buying anything and should go.
    """
    try:
        from scipy.sparse._sputils import get_index_dtype
    except ImportError:  # pragma: no cover - scipy < 1.8 spelling
        from scipy.sparse.sputils import get_index_dtype

    i32 = (np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int32))
    i64 = (np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64))

    assert get_index_dtype(i32, maxval=2**31 - 1, check_contents=True) == np.int32
    assert get_index_dtype(i32, maxval=2**31, check_contents=True) == np.int64
    # The input width does not enter into it, in either direction.
    assert get_index_dtype(i64, maxval=1, check_contents=True) == np.int32
    assert get_index_dtype(i64, maxval=2**31, check_contents=True) == np.int64


def test_eager_read_widens_indices_above_the_threshold(tmp_dir):
    """The widened decode produces the same matrix, value for value.

    No fixture can carry 2**31 nonzeros, so the threshold is lowered instead.
    That exercises the real typed decode end to end — the reader assembles an
    int64 index buffer and hands it to scipy — rather than the plan selection
    alone, which `widen_indices_only_touches_a_default_plan_over_the_line`
    covers on the Rust side.

    What it cannot assert is `indices.dtype == int64` on the way out: on a
    fixture this small scipy downcasts it straight back to int32 (see
    `test_scipy_resolves_the_index_dtype_from_the_contents`). The dtype is only
    observable above the real threshold, which is what the cluster arm on the
    2.65e9-nonzero object measures. What *is* observable here, and is the risk
    a cast introduces, is whether the values and column indices survive it.
    """
    adata = _counts_adata(n_obs=60, n_vars=20, seed=3)
    path = _write(tmp_dir, adata, name="widen.scx")

    control = pyscx.open(path).to_anndata()
    assert control.X.indices.dtype == np.int32

    out = str(tmp_dir / "widened.npz")
    _run_with_threshold(
        f"""
import numpy as np, pyscx, scipy.sparse as sp
a = pyscx.open({path!r}).to_anndata()
assert sp.issparse(a.X) and a.X.format == "csr", a.X
assert a.X.data.dtype == np.float32, a.X.data.dtype
sp.save_npz({out!r}, a.X)
""",
        0,
    )

    widened = sp.load_npz(out)
    assert widened.shape == control.X.shape
    np.testing.assert_array_equal(widened.indptr, control.X.indptr)
    np.testing.assert_array_equal(widened.indices, control.X.indices)
    np.testing.assert_array_equal(widened.data, control.X.data)
    np.testing.assert_array_equal(widened.toarray(), adata.X.toarray())
def test_widen_leaves_an_explicit_index_dtype_alone(tmp_dir):
    """An explicit `index_dtype=` is the caller's decision, threshold or not."""
    adata = _counts_adata(n_obs=30, n_vars=10, seed=4)
    path = _write(tmp_dir, adata, name="explicit.scx")
    _run_with_threshold(
        f"""
import numpy as np, pyscx
a = pyscx.open({path!r}).to_anndata(index_dtype="int32")
assert a.X.indices.dtype == np.int32, a.X.indices.dtype
""",
        0,
    )
