"""Regression tests for C1: global-vs-kept row indexing bug.

These tests create datasets where some cells are filtered (activating the
deletion vector), then verify that normalize_total, __mul__, and __truediv__
don't panic from index-out-of-bounds.

The bug was that transform parameter vectors (row_sums, factors) were
stored in kept-row space (length n_kept) but indexed by global row
offset in apply_transforms_to_csr.

To work around anndata 0.12's strict obs/shape validation, we construct
the backed dataset with a deletion vector and build an anndata with
matching obs dimensions.
"""

import numpy as np
import scipy.sparse as sp
import pytest
import tempfile
import os

import anndata
import pandas as pd
import pyscx


def _make_scx_with_sparse_cells(n_obs=20, n_vars=10, n_sparse_cells=3, seed=42):
    """Create SCX file where some cells have very few nonzero genes.

    First `n_sparse_cells` cells get exactly 1 nonzero gene each.
    Remaining cells get 5+ nonzero genes each.

    Returns (path, X_full, keep_mask) where keep_mask[i] is True if
    cell i has >= 3 nonzero genes.
    """
    rng = np.random.RandomState(seed)

    rows, cols, vals = [], [], []

    # Sparse cells: 1 nonzero gene each
    for i in range(n_sparse_cells):
        col = rng.randint(0, n_vars)
        rows.append(i)
        cols.append(col)
        vals.append(float(rng.randint(1, 10)))

    # Dense cells: 5-8 nonzero genes each
    for i in range(n_sparse_cells, n_obs):
        n_genes = rng.randint(5, min(9, n_vars + 1))
        gene_idx = rng.choice(n_vars, size=n_genes, replace=False)
        for col in gene_idx:
            rows.append(i)
            cols.append(col)
            vals.append(float(rng.randint(1, 100)))

    X = sp.csr_matrix(
        (np.array(vals, dtype=np.float32), (rows, cols)),
        shape=(n_obs, n_vars),
    )

    # Compute per-row nnz to determine keep mask
    row_nnz = np.diff(X.indptr)
    keep_mask = row_nnz >= 3

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_sparse.scx")
    pyscx.from_anndata(adata, path)
    return path, X, keep_mask


def _open_backed_with_deletions(path, keep_mask):
    """Open an SCX file in backed mode with a deletion vector applied.

    Returns an anndata where X has the deletion vector set and obs is
    sliced to match, bypassing anndata's strict shape validation.
    """
    # Open full dataset
    adata = pyscx.open(path).to_anndata(backed=True)
    backed_x = adata.X

    # Build kept_to_global mapping
    kept_indices = np.where(keep_mask)[0].astype(np.uint64)

    # Apply deletion vector via the Rust object's internal method.
    # We construct a new anndata with matching dimensions.
    # First, change X shape via set_kept_to_global (called by filter_cells internally).
    # Then build anndata with correct obs.

    # Use pyscx.accel.filter_cells but set obs BEFORE it via _obs to bypass validation.
    # Actually, let's directly use normalize_total which is the bug site.
    # We need a way to set the deletion vector...

    # The simplest approach: build new anndata with the X object already filtered.
    # We can do this by:
    # 1. Get the backed X object
    # 2. Trick it into having a deletion vector by calling filter_cells
    #    with the right parameters on a raw adata

    # Actually, the cleanest approach is to directly test the Rust objects
    # without going through anndata's setter validation.
    # We can call normalize_total by passing a mock adata.

    # Simplest: create adata with kept-only obs from the start, then swap in X
    n_kept = int(keep_mask.sum())
    kept_obs = pd.DataFrame(index=[f"cell_{i}" for i in np.where(keep_mask)[0]])
    var = adata.var.copy()

    # Create adata with a dummy X of the right shape, then replace
    dummy_X = sp.csr_matrix((n_kept, var.shape[0]), dtype=np.float32)
    new_adata = anndata.AnnData(X=dummy_X, obs=kept_obs, var=var)

    # Now replace X with the backed object that has a deletion vector
    # We need to set kept_to_global on the backed_x object.
    # Since set_kept_to_global is pub(crate), we can't call it from Python.
    # Instead, let's use a workaround: use filter_cells on the original adata
    # and catch the error, knowing the X object was already mutated.

    original_adata = pyscx.open(path).to_anndata(backed=True)
    try:
        # min_genes=3 will filter out cells with < 3 genes
        pyscx.accel.filter_cells(original_adata, min_genes=3)
    except ValueError:
        # anndata obs/shape mismatch — but X already has deletion vector set
        pass

    # The backed X now has kept_to_global set
    filtered_x = original_adata._X
    # Verify shape was updated
    assert filtered_x.shape[0] == n_kept, (
        f"Expected {n_kept} kept cells, got {filtered_x.shape[0]}"
    )

    # Replace the new adata's X with the filtered backed X
    new_adata._X = filtered_x
    return new_adata


@pytest.fixture
def sparse_scx():
    """SCX file with 3 sparse cells (1 gene each) and 17 dense cells."""
    path, X, keep_mask = _make_scx_with_sparse_cells()
    yield path, X, keep_mask


class TestNormalizeTotalAfterFilterCells:
    """C1b: normalize_total on lazy dataset after filter_cells with active deletions."""

    def test_backed_filter_then_normalize(self, sparse_scx):
        """filter_cells → normalize_total should not panic."""
        path, X_ref, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)

        n_kept = adata.n_obs
        assert n_kept < X_ref.shape[0], "Should have fewer cells after filtering"

        # This was the crash site: normalize_total stored filtered_sums
        # (length n_kept) but apply_transforms indexes by global row
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # Read some data — exercises apply_transforms_to_csr
        result = adata.X[0:5]
        assert result.shape == (min(5, n_kept), adata.n_vars)

        # Verify row sums equal target_sum
        row_sums = np.array(result.sum(axis=1)).flatten()
        nonzero_mask = row_sums > 0
        np.testing.assert_allclose(
            row_sums[nonzero_mask], 1e4, rtol=1e-5,
            err_msg="Normalized row sums should equal target_sum",
        )

    def test_lazy_filter_then_normalize_then_normalize(self, sparse_scx):
        """Double normalize_total on lazy dataset (Case 2 path)."""
        path, _, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)

        # First normalize creates lazy wrapper (Case 1)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        # Second normalize appends to lazy (Case 2 — the buggy path)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        result = adata.X[0:5]
        assert result.shape[0] == min(5, adata.n_obs)

    def test_filter_normalize_log1p_getitem(self, sparse_scx):
        """Full standard pipeline: filter → normalize → log1p → read."""
        path, _, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)

        # Read all data
        result = adata.X[:]
        assert result.shape == (adata.n_obs, adata.n_vars)
        # log1p values should be non-negative
        assert result.min() >= 0.0

    def test_normalize_read_all_rows(self, sparse_scx):
        """Read every row after normalize — ensures no OOB on any shard."""
        path, X_ref, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # Read all rows
        full = adata.X[:]
        assert full.shape[0] == adata.n_obs


class TestMulDivAfterFilterCells:
    """C1c/C1d: __mul__/__truediv__ with active deletion vectors."""

    def test_backed_mul_with_deletions(self, sparse_scx):
        """__mul__ on ScxBackedSparseDataset after filter_cells."""
        path, _, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)

        n_kept = adata.n_obs
        factors = np.full(n_kept, 2.0)

        # This was a crash site: factors had length n_kept but
        # RowScale indexes by global row
        result = adata.X * factors.reshape(-1, 1)
        mat = result[:]

        # Compare with direct materialization * 2
        original = adata.X[:]
        expected = original.toarray() * 2.0
        np.testing.assert_allclose(
            mat.toarray(), expected, rtol=1e-5,
            err_msg="__mul__ with deletions should scale correctly",
        )

    def test_backed_truediv_with_deletions(self, sparse_scx):
        """__truediv__ on ScxBackedSparseDataset after filter_cells."""
        path, _, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)

        n_kept = adata.n_obs
        divisors = np.full(n_kept, 2.0)

        result = adata.X / divisors.reshape(-1, 1)
        mat = result[:]

        original = adata.X[:]
        expected = original.toarray() / 2.0
        np.testing.assert_allclose(
            mat.toarray(), expected, rtol=1e-5,
            err_msg="__truediv__ with deletions should scale correctly",
        )

    def test_lazy_mul_with_deletions(self, sparse_scx):
        """__mul__ on ScxLazyTransformedDataset after filter + normalize."""
        path, _, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        n_kept = adata.n_obs
        factors = np.full(n_kept, 3.0)
        result = adata.X * factors.reshape(-1, 1)
        mat = result[:]

        # Verify it's 3x the normalized values
        normalized = adata.X[:]
        expected = normalized.toarray() * 3.0
        np.testing.assert_allclose(
            mat.toarray(), expected, rtol=1e-5,
        )

    def test_lazy_truediv_with_deletions(self, sparse_scx):
        """__truediv__ on ScxLazyTransformedDataset after filter + normalize."""
        path, _, keep_mask = sparse_scx
        adata = _open_backed_with_deletions(path, keep_mask)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        n_kept = adata.n_obs
        divisors = np.ones(n_kept)

        # Identity division — result should equal normalized values
        result = adata.X / divisors.reshape(-1, 1)
        mat = result[:]

        normalized = adata.X[:]
        np.testing.assert_allclose(
            mat.toarray(), normalized.toarray(), rtol=1e-5,
        )
