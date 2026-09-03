"""Tests for Experiment.gather_rows_sparse (SCX-DATA-LOADER Phase 0.2; PR C).

The synchronous sparse gather over BackedCsrReader::read_row_indices must match
the backed `adata.X[rows]` analysis path element-for-element, in request order,
for scattered / duplicated / unsorted row sets — and, since PR C, for a layer
(`layer=`) and for the logical row space of a deletion-vector file
(`logical=True`, the default).
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def scx_path(synthetic_adata, tmp_dir):
    import pyscx

    path = str(tmp_dir / "gather_test.scx")
    # Ten shards, so every scattered request spans several of them.
    pyscx.from_anndata(synthetic_adata, path, shard_size=10)
    return path


@pytest.fixture
def adata_backed(scx_path):
    import pyscx

    return pyscx.open(scx_path).to_anndata(backed=True)


def _gather(scx_path, rows, **kw):
    import pyscx

    exp = pyscx.open(scx_path)
    return exp.gather_rows_sparse(np.asarray(rows, dtype=np.uint64), **kw)


def test_gather_matches_backed_contiguous(scx_path, adata_backed):
    rows = list(range(10, 20))
    got = _gather(scx_path, rows)
    assert isinstance(got, sp.csr_matrix)
    expected = adata_backed.X[rows]
    assert got.shape == expected.shape
    np.testing.assert_array_equal(got.toarray(), expected.toarray())


def test_gather_scattered_unsorted_preserves_request_order(scx_path, adata_backed):
    # Unsorted, spanning the whole file so multiple shards are touched.
    rows = [99, 0, 57, 3, 80, 12, 41]
    got = _gather(scx_path, rows)
    expected = adata_backed.X[rows]
    np.testing.assert_array_equal(got.toarray(), expected.toarray())


def test_gather_with_duplicates(scx_path, adata_backed):
    rows = [5, 5, 5, 7, 5, 7]
    got = _gather(scx_path, rows)
    expected = adata_backed.X[rows]
    assert got.shape[0] == len(rows)
    np.testing.assert_array_equal(got.toarray(), expected.toarray())


def test_gather_empty(scx_path):
    got = _gather(scx_path, [])
    assert isinstance(got, sp.csr_matrix)
    assert got.shape[0] == 0


@pytest.mark.parametrize("cache_shards", [4, 16])
def test_gather_cache_shards_invariant(scx_path, adata_backed, cache_shards):
    rows = [99, 0, 57, 3, 80, 12, 41]
    got = _gather(scx_path, rows, cache_shards=cache_shards)
    expected = adata_backed.X[rows]
    np.testing.assert_array_equal(got.toarray(), expected.toarray())


def test_gather_out_of_range_raises_index_error(scx_path):
    with pytest.raises(IndexError, match="10000"):
        _gather(scx_path, [0, 1, 10_000])


# ---------------------------------------------------------------------------
# PR C: selector forms
# ---------------------------------------------------------------------------


def test_gather_accepts_a_plain_list_and_any_integer_dtype(scx_path, adata_backed):
    import pyscx

    exp = pyscx.open(scx_path)
    rows = [99, 0, 57]
    expected = adata_backed.X[rows].toarray()
    np.testing.assert_array_equal(exp.gather_rows_sparse(rows).toarray(), expected)
    for dtype in (np.int32, np.int64, np.uint32, np.int8):
        got = exp.gather_rows_sparse(np.asarray(rows, dtype=dtype))
        np.testing.assert_array_equal(got.toarray(), expected)
    np.testing.assert_array_equal(
        exp.gather_rows_sparse(range(10, 13)).toarray(), adata_backed.X[10:13].toarray()
    )


def test_gather_negative_indices_wrap_once(scx_path, adata_backed):
    import pyscx

    exp = pyscx.open(scx_path)
    got = exp.gather_rows_sparse(np.asarray([-1, 0, -100], dtype=np.int64))
    np.testing.assert_array_equal(got.toarray(), adata_backed.X[[99, 0, 0]].toarray())
    with pytest.raises(IndexError, match="-101"):
        exp.gather_rows_sparse(np.asarray([-101], dtype=np.int64))


def test_gather_accepts_a_boolean_mask_and_checks_its_length(scx_path, adata_backed):
    import pyscx

    exp = pyscx.open(scx_path)
    mask = np.zeros(exp.n_obs, dtype=bool)
    mask[::7] = True
    got = exp.gather_rows_sparse(mask)
    np.testing.assert_array_equal(got.toarray(), adata_backed.X[mask].toarray())
    with pytest.raises(IndexError, match="boolean row mask"):
        exp.gather_rows_sparse(mask[:-1])
    with pytest.raises(IndexError, match="boolean row mask"):
        exp.gather_rows_sparse(np.ones(exp.n_obs + 1, dtype=bool))


def test_gather_rejects_a_float_selector_and_a_2d_selector(scx_path):
    import pyscx

    exp = pyscx.open(scx_path)
    with pytest.raises(IndexError, match="integer array or a boolean mask"):
        exp.gather_rows_sparse(np.asarray([0.0, 1.0]))
    with pytest.raises(IndexError, match="one-dimensional"):
        exp.gather_rows_sparse(np.asarray([[0, 1], [2, 3]]))


# ---------------------------------------------------------------------------
# PR C: layer=
# ---------------------------------------------------------------------------


def test_gather_from_a_layer_matches_the_layer_slice(scx_path, synthetic_adata):
    import pyscx

    exp = pyscx.open(scx_path)
    rows = [42, 3, 3, 99, 0]
    got = exp.gather_rows_sparse(rows, layer="raw")
    expected = sp.csr_matrix(synthetic_adata.layers["raw"])[rows]
    np.testing.assert_array_equal(got.toarray(), expected.toarray())
    # ... and differs from X where the layer does.
    x_rows = exp.gather_rows_sparse(rows)
    assert x_rows.shape == got.shape
    backed = exp.to_anndata(backed=True)
    np.testing.assert_array_equal(
        got.toarray(), backed.layers["raw"][rows].toarray()
    )


def test_gather_unknown_layer_is_a_value_error(scx_path):
    import pyscx

    exp = pyscx.open(scx_path)
    with pytest.raises(ValueError, match="layer 'nope' not found"):
        exp.gather_rows_sparse([0, 1], layer="nope")


def test_gather_layer_on_a_multimodal_file_is_refused(tmp_dir):
    import anndata
    import pyscx

    mudata = pytest.importorskip("mudata")
    rna = anndata.AnnData(X=sp.random(30, 8, density=0.4, format="csr", dtype=np.float32))
    adt = anndata.AnnData(X=sp.random(30, 5, density=0.4, format="csr", dtype=np.float32))
    rna.obs_names = adt.obs_names = [f"c{i}" for i in range(30)]
    rna.var_names = [f"g{i}" for i in range(8)]
    adt.var_names = [f"p{i}" for i in range(5)]
    mdata = mudata.MuData({"rna": rna, "adt": adt})
    path = str(tmp_dir / "mm.scx")
    pyscx.from_mudata(mdata, path)
    exp = pyscx.open(path)
    # Modality-scoped X gather still works ...
    got = exp.gather_rows_sparse([3, 1], modality="rna")
    np.testing.assert_array_equal(got.toarray(), rna.X[[3, 1]].toarray())
    # ... but a layer on a multimodal file is refused, not guessed.
    with pytest.raises(ValueError, match="multimodal"):
        exp.gather_rows_sparse([0], modality="rna", layer="raw")


# ---------------------------------------------------------------------------
# PR C: logical row space
# ---------------------------------------------------------------------------


@pytest.fixture
def deleted_path(synthetic_adata, tmp_dir):
    """Ten shards with deletions straddling shard boundaries (rows 9, 10, 11
    span the first boundary) — a single-shard file would pass this by accident."""
    import pyscx

    path = str(tmp_dir / "gather_deleted.scx")
    pyscx.from_anndata(synthetic_adata, path, shard_size=10)
    pyscx.mark_deleted(path, [0, 9, 10, 11, 50, 99])
    return path


def test_logical_default_matches_the_backed_handle_on_a_deleted_file(deleted_path):
    import pyscx

    exp = pyscx.open(deleted_path)
    backed = exp.to_anndata(backed=True)
    assert exp.n_obs == backed.n_obs == 94
    rows = [0, 93, 8, 9, 10, 47, 47]
    got = exp.gather_rows_sparse(rows)
    np.testing.assert_array_equal(got.toarray(), backed.X[rows].toarray())
    # Bounds are the logical count, not the physical one.
    with pytest.raises(IndexError, match="94 rows"):
        exp.gather_rows_sparse([94])
    # A full logical mask works and equals the whole backed matrix.
    full = exp.gather_rows_sparse(np.ones(exp.n_obs, dtype=bool))
    np.testing.assert_array_equal(full.toarray(), backed.X[:].toarray())


def test_logical_false_addresses_physical_rows(deleted_path, synthetic_adata):
    import pyscx

    exp = pyscx.open(deleted_path)
    assert exp.n_obs_physical == 100
    physical = sp.csr_matrix(synthetic_adata.X)
    rows = [0, 9, 10, 11, 99, 5]  # includes deleted cells
    got = exp.gather_rows_sparse(rows, logical=False)
    np.testing.assert_array_equal(got.toarray(), physical[rows].toarray())
    with pytest.raises(IndexError, match="100 rows"):
        exp.gather_rows_sparse([100], logical=False)


def test_logical_and_physical_agree_without_deletions(scx_path):
    import pyscx

    exp = pyscx.open(scx_path)
    rows = [7, 0, 99]
    np.testing.assert_array_equal(
        exp.gather_rows_sparse(rows, logical=True).toarray(),
        exp.gather_rows_sparse(rows, logical=False).toarray(),
    )
