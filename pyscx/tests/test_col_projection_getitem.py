"""Tests for non-materializing column __getitem__ on backed/lazy datasets.

When X[:, col_array] is called with all-rows selection and an array/mask
column selector, the result should be a new backed/lazy dataset with
col_projection set — NOT a materialized scipy matrix.  This keeps
subsequent aggregation on the f64 streaming path.
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def scx_path(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic data."""
    import pyscx

    path = str(tmp_dir / "col_proj_getitem.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def reference_dense(synthetic_adata):
    """Dense array of the original data for comparison."""
    return synthetic_adata.X.toarray()


# ---------------------------------------------------------------------------
# ScxBackedSparseDataset: X[:, array] returns backed dataset
# ---------------------------------------------------------------------------


class TestBackedColProjectionGetitem:
    """X[:, col] on ScxBackedSparseDataset returns a new backed dataset."""

    def test_bool_mask_returns_backed(self, scx_path):
        """X[:, bool_mask] returns ScxBackedSparseDataset, not scipy."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        mask = np.zeros(adata.shape[1], dtype=bool)
        mask[[0, 5, 10, 20]] = True

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (adata.shape[0], int(mask.sum()))

    def test_int_array_returns_backed(self, scx_path):
        """X[:, int_array] returns ScxBackedSparseDataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([2, 7, 15, 30, 42], dtype=np.int64)

        result = adata.X[:, cols]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (adata.shape[0], len(cols))

    def test_sum_axis0_uses_streaming(self, scx_path, reference_dense):
        """X[:, mask].sum(axis=0) uses f64 streaming, matches reference."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([1, 5, 10, 25, 40])
        projected = adata.X[:, cols]

        result = np.asarray(projected.sum(axis=0)).ravel()
        expected = reference_dense[:, cols].sum(axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_var_axis0_uses_streaming(self, scx_path, reference_dense):
        """X[:, mask].var(axis=0) uses f64 streaming, matches reference."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([3, 8, 15, 30])
        projected = adata.X[:, cols]

        result = np.asarray(projected.var(axis=0)).ravel()
        expected = reference_dense[:, cols].var(axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_getnnz_axis0_uses_streaming(self, scx_path, reference_dense):
        """X[:, mask].getnnz(axis=0) uses streaming, matches reference."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([0, 10, 20, 30, 40, 49])
        projected = adata.X[:, cols]

        result = np.asarray(projected.getnnz(axis=0))
        expected = (reference_dense[:, cols] != 0).sum(axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_composition(self, scx_path, reference_dense):
        """X[:, mask1][:, mask2] composes projections correctly."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        # First projection: select 10 columns
        cols1 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        proj1 = adata.X[:, cols1]
        assert proj1.shape == (100, 10)

        # Second projection: select 3 of those 10
        cols2 = np.array([1, 4, 8])  # indices into the projected space
        proj2 = proj1[:, cols2]
        assert proj2.shape == (100, 3)

        # Should be equivalent to selecting [5, 20, 40] from original
        result = np.asarray(proj2.sum(axis=0)).ravel()
        expected = reference_dense[:, [5, 20, 40]].sum(axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_to_memory_respects_projection(self, scx_path, reference_dense):
        """X[:, mask].to_memory() materializes only projected columns."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([2, 7, 15])
        projected = adata.X[:, cols]

        mat = projected.to_memory()
        assert sp.issparse(mat)
        assert mat.shape == (100, 3)
        np.testing.assert_allclose(
            mat.toarray(), reference_dense[:, cols], rtol=1e-6
        )


# ---------------------------------------------------------------------------
# ScxLazyTransformedDataset: X[:, array] stays lazy
# ---------------------------------------------------------------------------


class TestLazyColProjectionGetitem:
    """After normalize_total+log1p, X[:, col] stays lazy-transformed."""

    def test_lazy_mask_returns_lazy(self, scx_path):
        """X[:, bool_mask] on lazy dataset returns ScxLazyTransformedDataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        pyscx.accel.log1p(adata)

        mask = np.zeros(adata.shape[1], dtype=bool)
        mask[[0, 5, 10, 20]] = True

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)
        assert result.shape == (adata.shape[0], int(mask.sum()))

    def test_lazy_int_array_returns_lazy(self, scx_path):
        """X[:, int_array] on lazy dataset returns ScxLazyTransformedDataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        cols = np.array([2, 7, 15, 30, 42], dtype=np.int64)

        result = adata.X[:, cols]
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)
        assert result.shape == (adata.shape[0], len(cols))

    def test_lazy_sum_axis0_correctness(self, scx_path):
        """X[:, cols].sum(axis=0) on lazy data matches full materialization."""
        import pyscx

        # Lazy path: normalize+log1p then project then sum
        adata_b = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_b)
        pyscx.accel.log1p(adata_b)
        cols = np.array([1, 5, 10, 25, 40])
        projected = adata_b.X[:, cols]
        result = np.asarray(projected.sum(axis=0)).ravel()

        # Reference: materialize full lazy data, then subset columns
        adata_b2 = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_b2)
        pyscx.accel.log1p(adata_b2)
        full_mat = adata_b2.X[:].toarray()
        expected = full_mat[:, cols].sum(axis=0)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_lazy_composition(self, scx_path):
        """X[:, mask1][:, mask2] composes on lazy dataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)

        cols1 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        proj1 = adata.X[:, cols1]
        assert isinstance(proj1, pyscx.ScxLazyTransformedDataset)
        assert proj1.shape == (100, 10)

        cols2 = np.array([2, 5, 9])
        proj2 = proj1[:, cols2]
        assert isinstance(proj2, pyscx.ScxLazyTransformedDataset)
        assert proj2.shape == (100, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for column projection __getitem__."""

    def test_row_subset_still_materializes(self, scx_path):
        """X[0:10, cols] with row subset still returns scipy (not backed)."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([1, 5, 10])

        result = adata.X[0:10, cols]
        assert sp.issparse(result)
        assert result.shape == (10, 3)

    def test_scalar_col_still_works(self, scx_path):
        """X[0, 5] still returns a scalar float."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        result = adata.X[0, 5]
        assert isinstance(result, (float, np.floating, int, np.integer))

    def test_empty_mask(self, scx_path):
        """X[:, empty_mask] returns a 0-column dataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        mask = np.zeros(adata.shape[1], dtype=bool)

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (100, 0)

    def test_full_mask(self, scx_path, reference_dense):
        """X[:, all_true_mask] returns all columns."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        mask = np.ones(adata.shape[1], dtype=bool)

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (100, 50)

        sums = np.asarray(result.sum(axis=0)).ravel()
        expected = reference_dense.sum(axis=0)
        np.testing.assert_allclose(sums, expected, rtol=1e-6)
