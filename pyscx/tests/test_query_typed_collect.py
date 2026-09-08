"""`collect(data_dtype=...)` — decoding a query's X at the caller's dtype.

The query path used to assemble X as f32 and cast afterwards, so a file whose
surviving shards hold counts above 2**24 could only be refused or served rounded
values: `data_dtype="float64"` was accepted by the signature and rejected by the
guard. The dtype now has to be declared at `collect()`, because that call *is*
the decode — a dtype named later can only cast values that already rounded.

**What these tests can and cannot prove.** `pyscx.from_anndata` casts X through
float32, and every other Python write door does too, so the largest count these
fixtures can carry is an f32-*exact* one (`20_000_000` is, being 4 x 5_000_000).
An f32-exact value comes back correct whether or not the decode was typed, so no
test here can tell a typed decode from an f32 detour. Two things close that gap:
`test_wide_dtype_after_default_collect_raises` below, which fails if the guard
is merely made dtype-aware without changing the assembly, and the Rust tests in
`scx-engine/src/collect/native_tests.rs`, which write an **odd** value above
2**24 through the raw-bytes shard writer and are mutation-checked against an f32
detour.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import anndata as ad
import pyscx

BIG = 20_000_000


def _big_count_adata():
    """4 cells x 2 genes; row 0 carries a count above 2**24."""
    x = np.array([[BIG, 0], [1, 2], [3, 0], [4, 5]], dtype=np.int32)
    return ad.AnnData(
        X=sp.csr_matrix(x),
        obs={"batch": ["a", "b", "a", "b"]},
        var={"gene": ["g0", "g1"]},
    )


def _small_count_adata():
    x = np.array([[7, 0], [1, 2], [3, 0], [4, 5]], dtype=np.int32)
    return ad.AnnData(
        X=sp.csr_matrix(x),
        obs={"batch": ["a", "b", "a", "b"]},
        var={"gene": ["g0", "g1"]},
    )


def _write(tmp_dir, adata, name="typed.scx"):
    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)
    return path


# ---------------------------------------------------------------------------
# The assertion that separates the fix from a guard-only change
# ---------------------------------------------------------------------------


def test_wide_dtype_after_default_collect_raises(tmp_dir):
    """A wide dtype named *after* a default collect must be refused.

    This is the one Python assertion that fails against an implementation which
    only makes the guard dtype-aware: `float64` can hold the value, so such an
    implementation returns f32-rounded values labelled float64. The values are
    already f32 by then — the decode has happened — so the honest answer is a
    refusal that says where the dtype belongs.
    """
    path = _write(tmp_dir, _big_count_adata())
    result = pyscx.open(path).query().filter_obs("batch == 'a'").collect()
    with pytest.raises(ValueError, match=r"collect\(data_dtype="):
        result.to_csr(data_dtype="float64")


def test_wide_dtype_after_default_collect_raises_on_to_anndata(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    result = pyscx.open(path).query().filter_obs("batch == 'a'").collect()
    with pytest.raises(ValueError, match=r"collect\(data_dtype="):
        result.to_anndata(data_dtype="float64")
    # Non-destructive, like every other tripped guard on this object.
    assert result.to_csr(allow_lossy=True).data.dtype == np.float32


# ---------------------------------------------------------------------------
# The fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,np_dtype",
    [("float64", np.float64), ("uint32", np.uint32), ("int64", np.int64)],
)
def test_collect_data_dtype_reads_wide_exactly(tmp_dir, dtype, np_dtype):
    path = _write(tmp_dir, _big_count_adata())
    x = (
        pyscx.open(path)
        .query()
        .filter_obs("batch == 'a'")
        .collect(data_dtype=dtype)
        .to_csr()
    )
    assert x.data.dtype == np_dtype
    assert int(x.max()) == BIG


def test_collect_dtype_is_carried_not_recast(tmp_dir):
    """`data_dtype=None` at materialize means "what this was decoded at".

    Resolving it to float32 instead would narrow the buffer straight back down
    and refuse the read the caller just asked for.
    """
    path = _write(tmp_dir, _big_count_adata())
    result = pyscx.open(path).query().collect(data_dtype="uint32")
    x = result.to_csr()
    assert x.data.dtype == np.uint32
    assert int(x.max()) == BIG


def test_obs_filter_route_reads_wide_exactly(tmp_dir):
    """`to_anndata(obs_filter=, data_dtype=)` needs no new kwarg.

    That route builds the pipeline and collects in one call, so it already knows
    the dtype before the decode.
    """
    path = _write(tmp_dir, _big_count_adata())
    rt = pyscx.open(path).to_anndata(obs_filter="batch == 'a'", data_dtype="float64")
    assert rt.X.data.dtype == np.float64
    assert int(rt.X.max()) == BIG
    assert rt.n_obs == 2

    # Narrower than the value: still refused on that route.
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).to_anndata(obs_filter="batch == 'a'", data_dtype="uint16")


def test_obs_filter_with_var_names_reads_wide_exactly(tmp_dir):
    """The projection sub-case: the typed decode projects columns natively."""
    path = _write(tmp_dir, _big_count_adata())
    rt = pyscx.open(path).to_anndata(
        obs_filter="batch == 'a'", var_names=["g0"], data_dtype="float64"
    )
    assert rt.shape == (2, 1)
    assert rt.X.data.dtype == np.float64
    assert int(rt.X.max()) == BIG


# ---------------------------------------------------------------------------
# What still refuses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["uint16", "float16", "int8"])
def test_collect_narrow_still_fails_loud(tmp_dir, dtype):
    path = _write(tmp_dir, _big_count_adata())
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(path).query().collect(data_dtype=dtype)


def test_collect_allow_lossy_escapes(tmp_dir):
    path = _write(tmp_dir, _big_count_adata())
    x = (
        pyscx.open(path)
        .query()
        .collect(data_dtype="uint16", allow_lossy=True)
        .to_csr()
    )
    assert x.data.dtype == np.uint16


def test_collect_dtype_failure_does_not_consume_the_pipeline(tmp_dir):
    """A refused cast is a new way for `collect` itself to fail.

    The documented contract is that a failed collect leaves the pipeline usable,
    so the caller can correct the offending step and re-collect.
    """
    path = _write(tmp_dir, _big_count_adata())
    pipeline = pyscx.open(path).query().filter_obs("batch == 'a'")
    with pytest.raises(ValueError, match="allow_lossy"):
        pipeline.collect(data_dtype="uint16")
    assert pipeline.count() == 2
    assert pipeline.collect(data_dtype="float64").n_obs == 2


def test_materialize_dtype_mismatch_raises(tmp_dir):
    """A second, different dtype at materialize time is a wrong-knob error.

    Not a loss question, so `allow_lossy` does not unlock it: the caller either
    wants a different decode (re-collect) or a numpy cast of what they hold.
    """
    path = _write(tmp_dir, _small_count_adata())
    result = pyscx.open(path).query().collect(data_dtype="float64")
    with pytest.raises(ValueError, match="astype"):
        result.to_csr(data_dtype="float32")

    result = pyscx.open(path).query().collect(data_dtype="float64")
    with pytest.raises(ValueError, match="collected as float64"):
        result.to_csr(data_dtype="uint16", allow_lossy=True)

    # Naming the same dtype again is a no-op, not an error.
    result = pyscx.open(path).query().collect(data_dtype="float64")
    assert result.to_csr(data_dtype="float64").data.dtype == np.float64


def test_fused_transform_with_a_dtype_is_refused(tmp_dir):
    """normalize / log1p replace the counts with floats.

    No requested dtype can reproduce the stored data exactly then, and serving it
    from the f32 route would change which guard ran under the caller's feet.
    """
    path = _write(tmp_dir, _small_count_adata())
    for build in (
        lambda p: p.query().with_log1p(),
        lambda p: p.query().with_normalize(target_sum=1e4),
    ):
        with pytest.raises(ValueError, match="with_normalize"):
            build(pyscx.open(path)).collect(data_dtype="float64")

    # Unchanged without a dtype, and float32 keeps the f32 route.
    a = pyscx.open(path).query().with_log1p().collect().to_anndata()
    assert a.X.data.dtype == np.float32
    b = pyscx.open(path).query().with_log1p().collect(data_dtype="float32").to_csr()
    assert b.data.dtype == np.float32


# ---------------------------------------------------------------------------
# The default path is untouched
# ---------------------------------------------------------------------------


def test_default_collect_is_unchanged(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    x = pyscx.open(path).query().filter_obs("batch == 'b'").collect().to_csr()
    assert x.data.dtype == np.float32
    assert x.indices.dtype == np.int32
    np.testing.assert_array_equal(x.toarray(), np.array([[1, 2], [4, 5]], dtype=np.float32))

    # And a default collect on a >2**24 file still refuses at float32, which is
    # the honest answer for a decode that already happened.
    big = _write(tmp_dir, _big_count_adata(), "big.scx")
    with pytest.raises(ValueError, match="allow_lossy"):
        pyscx.open(big).query().collect().to_csr()


def test_typed_collect_matches_the_default_where_f32_is_exact(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    for expr in ["batch == 'a'", "batch == 'b'"]:
        default = pyscx.open(path).query().filter_obs(expr).collect()
        typed = pyscx.open(path).query().filter_obs(expr).collect(data_dtype="float64")
        assert (typed.n_obs, typed.n_vars, typed.nnz) == (
            default.n_obs,
            default.n_vars,
            default.nnz,
        )
        a, b = default.to_anndata(), typed.to_anndata()
        np.testing.assert_array_equal(a.X.toarray().astype(np.float64), b.X.toarray())
        assert list(a.obs.index) == list(b.obs.index)
        assert list(a.var.index) == list(b.var.index)


def test_limit_truncates_a_typed_result_like_the_default(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    for limit in [1, 2, 3, 99]:
        default = pyscx.open(path).query().limit(limit).collect().to_csr()
        typed = (
            pyscx.open(path).query().limit(limit).collect(data_dtype="float64").to_csr()
        )
        assert typed.shape == default.shape, limit
        np.testing.assert_array_equal(
            typed.toarray(), default.toarray().astype(np.float64)
        )


# ---------------------------------------------------------------------------
# Surface details
# ---------------------------------------------------------------------------


def test_dense_container_on_a_typed_result(tmp_dir):
    """`container` is presentation, applied after the decode — so it stays on
    the materialize call rather than joining `collect`."""
    path = _write(tmp_dir, _big_count_adata())
    a = (
        pyscx.open(path)
        .query()
        .collect(data_dtype="float64")
        .to_anndata(container="dense")
    )
    assert isinstance(a.X, np.ndarray)
    assert a.X.dtype == np.float64
    assert int(a.X.max()) == BIG


def test_repr_names_the_decoded_dtype(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    assert "dtype=float32" in repr(pyscx.open(path).query().collect())
    assert "dtype=float64" in repr(
        pyscx.open(path).query().collect(data_dtype="float64")
    )


def test_a_consumed_typed_result_says_so(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    result = pyscx.open(path).query().collect(data_dtype="float64")
    result.to_csr()
    with pytest.raises(RuntimeError, match="already consumed"):
        result.to_csr()
    # Getters still work after consumption.
    assert result.n_obs == 4
    assert "consumed" in repr(result)


def test_collect_kwargs_are_keyword_only(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    with pytest.raises(TypeError):
        pyscx.open(path).query().collect("float64")


def test_an_empty_typed_result_keeps_its_shape(tmp_dir):
    path = _write(tmp_dir, _small_count_adata())
    result = (
        pyscx.open(path)
        .query()
        .filter_obs("batch == 'nobody'")
        .collect(data_dtype="float64")
    )
    x = result.to_csr()
    assert x.shape == (0, 2)
    assert x.data.dtype == np.float64


def test_index_dtype_after_a_typed_collect_is_refused_not_ignored(tmp_dir):
    """A typed result's indices are already narrowed, so an `index_dtype=` at
    materialize time would be accepted and silently ignored."""
    path = _write(tmp_dir, _small_count_adata())
    result = pyscx.open(path).query().collect(data_dtype="float64")
    with pytest.raises(ValueError, match=r"index_dtype"):
        result.to_csr(index_dtype="int64")

    # Declared at collect, it is honoured; repeating the same value is a no-op.
    x = (
        pyscx.open(path)
        .query()
        .collect(data_dtype="float64", index_dtype="int64")
        .to_csr(index_dtype="int64")
    )
    assert x.data.dtype == np.float64

    # The f32 arm still applies it post-assembly, as it always has.
    assert (
        pyscx.open(path).query().collect().to_csr(index_dtype="int64").data.dtype
        == np.float32
    )
