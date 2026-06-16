"""Tests for Experiment.gather_rows_sparse (SCX-DATA-LOADER Phase 0.2).

The synchronous sparse gather over BackedCsrReader::read_rows_with must match
the backed `adata.X[rows]` analysis path element-for-element, in request order,
for scattered / duplicated / unsorted row sets.
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def scx_path(synthetic_adata, tmp_dir):
    import pyscx

    path = str(tmp_dir / "gather_test.scx")
    pyscx.from_anndata(synthetic_adata, path)
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
    with pytest.raises(IndexError):
        _gather(scx_path, [0, 1, 10_000])
