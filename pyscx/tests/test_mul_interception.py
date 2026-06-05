"""Tests for __mul__ interception on ``ScxLazyTransformedDataset``.

Verifies that `ScxLazyTransformedDataset.__mul__()` detects per-row scaling
vectors and returns a lazy `ScxLazyTransformedDataset` with a `RowScale`
transform instead of materializing the full matrix.

Test coverage:
1. Basic __mul__ with row factors returns ScxLazyTransformedDataset (not scipy CSR)
2. Values match materialized reference (X * row_factors)
3. Handles 1D arrays, (n,1) column vectors, (1,n) row vectors
4. Non-row-factor shapes (scalar, wrong-length) fall back to materialization
5. Chaining: lazy.__mul__() appends RowScale transform
6. Streaming aggregation works after __mul__
"""

import numpy as np
import scipy.sparse as sp
import pytest
import tempfile
import os

import pyscx


def _make_test_scx(n_obs=100, n_vars=50, density=0.3, seed=42):
    """Create a temp SCX file with random integer sparse data."""
    X = sp.random(n_obs, n_vars, density=density, random_state=seed,
                  format='csr', dtype=np.float32)
    X.data = np.ceil(X.data * 100).astype(np.float32)
    X.eliminate_zeros()

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.scx")

    import anndata
    import pandas as pd

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    pyscx.from_anndata(adata, path)
    return path, X


@pytest.fixture
def scx_file():
    """Create a test SCX file and return (path, reference_X)."""
    path, X = _make_test_scx()
    yield path, X


class TestLazyMulInterception:
    """Test __mul__ on ScxLazyTransformedDataset appends RowScale lazily."""

    def test_1d_array_returns_lazy(self, scx_file):
        """Multiplying lazy dataset by 1D row-factor array should stay lazy."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        # Create a lazy dataset first (normalize_total)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)

        factors = np.random.default_rng(0).uniform(0.5, 2.0, size=n_obs)
        result = adata.X * factors

        assert isinstance(result, pyscx.ScxLazyTransformedDataset), \
            f"Expected ScxLazyTransformedDataset, got {type(result)}"

    def test_1d_values_match(self, scx_file):
        """Lazy mul values should match materialized X * factors."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        # Path 1: lazy chain (normalize → mul)
        adata_lazy = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_lazy, target_sum=1e4)
        factors = np.random.default_rng(0).uniform(0.5, 2.0, size=n_obs)
        result = adata_lazy.X * factors
        mat_lazy = result.to_memory().toarray()

        # Path 2: materialized reference
        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref = adata_mat.X.toarray() * factors[:, np.newaxis]

        np.testing.assert_allclose(
            mat_lazy, ref, rtol=1e-4,
            err_msg="Lazy mul values don't match materialized reference"
        )

    def test_column_vector_returns_lazy(self, scx_file):
        """Multiplying lazy dataset by (n_obs, 1) column vector should stay lazy."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.random.default_rng(1).uniform(0.5, 2.0, size=(n_obs, 1))
        result = adata.X * factors

        assert isinstance(result, pyscx.ScxLazyTransformedDataset)

    def test_column_vector_values_match(self, scx_file):
        """(n_obs, 1) column vector mul matches reference."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata_lazy = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_lazy, target_sum=1e4)
        factors = np.random.default_rng(1).uniform(0.5, 2.0, size=(n_obs, 1))
        result = adata_lazy.X * factors
        mat_lazy = result.to_memory().toarray()

        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref = adata_mat.X.toarray() * factors

        np.testing.assert_allclose(mat_lazy, ref, rtol=1e-4)

    def test_row_vector_not_intercepted(self, scx_file):
        """A (1, n_obs) row vector is NOT a per-row scale (B3).

        Only `(n_obs,)` and `(n_obs, 1)` are unambiguously row-oriented. A
        `(1, n_obs)` operand is no longer intercepted as a transposed RowScale;
        it falls through to scipy, where `*` is matrix multiplication and the
        non-conformable shape raises rather than being silently mis-applied.
        """
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.random.default_rng(2).uniform(0.5, 2.0, size=(1, n_obs))
        with pytest.raises(ValueError, match="dimension mismatch"):
            adata.X * factors

    def test_scalar_falls_back(self, scx_file):
        """Multiplying lazy dataset by scalar should materialize."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        result = adata.X * 2.0

        assert not isinstance(result, pyscx.ScxLazyTransformedDataset), \
            "Scalar mul should fall back to materialization"
        assert sp.issparse(result)

    def test_wrong_length_falls_back(self, scx_file):
        """Multiplying lazy dataset by array with wrong length should fall back."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.ones(17)  # wrong length
        # scipy will raise ValueError on shape mismatch after materialization
        with pytest.raises((ValueError, Exception)):
            adata.X * factors


class TestMulChaining:
    """Test __mul__ chaining with other transforms."""

    def test_mul_then_truediv(self, scx_file):
        """mul then truediv should produce two RowScale transforms."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        factors_mul = np.random.default_rng(10).uniform(0.5, 2.0, size=n_obs)
        factors_div = np.random.default_rng(11).uniform(0.5, 2.0, size=n_obs)

        result = adata.X * factors_mul
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)

        result2 = result / factors_div
        assert isinstance(result2, pyscx.ScxLazyTransformedDataset)

        # Verify values
        mat = result2.to_memory().toarray()

        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref = adata_mat.X.toarray() * factors_mul[:, np.newaxis] / factors_div[:, np.newaxis]

        np.testing.assert_allclose(mat, ref, rtol=1e-4)

    def test_normalize_log1p_mul(self, scx_file):
        """normalize_total → log1p → mul should chain all lazily."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata_lazy = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_lazy, target_sum=1e4)
        pyscx.accel.log1p(adata_lazy)
        factors = np.random.default_rng(12).uniform(0.5, 2.0, size=n_obs)
        result = adata_lazy.X * factors
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)

        mat_lazy = result.to_memory().toarray()

        # Reference
        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        sc.pp.log1p(adata_mat)
        ref = adata_mat.X.toarray() * factors[:, np.newaxis]

        np.testing.assert_allclose(mat_lazy, ref, rtol=1e-4)


class TestAggregationAfterMul:
    """Test that streaming aggregation works after __mul__."""

    def test_sum_axis1_after_mul(self, scx_file):
        """sum(axis=1) after mul should match reference."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.random.default_rng(13).uniform(0.5, 2.0, size=n_obs)
        result = adata.X * factors

        lazy_sum = np.array(result.sum(axis=1)).flatten()

        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref_sum = (adata_mat.X.toarray() * factors[:, np.newaxis]).sum(axis=1)

        np.testing.assert_allclose(
            lazy_sum, ref_sum, rtol=1e-4,
            err_msg="sum(axis=1) after mul mismatch"
        )

    def test_sum_axis0_after_mul(self, scx_file):
        """sum(axis=0) after mul should match reference."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.random.default_rng(14).uniform(0.5, 2.0, size=n_obs)
        result = adata.X * factors

        lazy_sum = np.array(result.sum(axis=0)).flatten()

        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref_sum = (adata_mat.X.toarray() * factors[:, np.newaxis]).sum(axis=0)

        np.testing.assert_allclose(
            lazy_sum, ref_sum, rtol=1e-4,
            err_msg="sum(axis=0) after mul mismatch"
        )

    def test_getnnz_preserved_after_mul(self, scx_file):
        """NNZ should be preserved after mul (non-zeros stay non-zero)."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.random.default_rng(15).uniform(0.5, 2.0, size=n_obs)
        result = adata.X * factors

        # NNZ should be identical before and after mul
        nnz_before = adata.X.getnnz()
        nnz_after = result.getnnz()
        assert nnz_before == nnz_after, \
            f"NNZ changed: {nnz_before} → {nnz_after}"

    def test_getitem_after_mul(self, scx_file):
        """Slicing after __mul__ should apply the RowScale transform."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        factors = np.random.default_rng(16).uniform(0.5, 2.0, size=n_obs)
        result = adata.X * factors

        chunk = result[10:20]

        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref_chunk = adata_mat.X[10:20].toarray() * factors[10:20, np.newaxis]

        np.testing.assert_allclose(
            chunk.toarray(), ref_chunk, rtol=1e-4,
            err_msg="__getitem__ after mul should apply RowScale transform"
        )
