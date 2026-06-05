"""Tests for __truediv__ interception — Step 5 of Phase 4d.

Verifies that `ScxBackedSparseDataset.__truediv__()` and
`ScxLazyTransformedDataset.__truediv__()` detect per-row scaling vectors
and return a lazy `ScxLazyTransformedDataset` instead of materializing
the full matrix.

Test coverage:
1. Basic __truediv__ returns ScxLazyTransformedDataset (not scipy CSR)
2. Values match materialized reference (X / row_factors)
3. Handles 1D arrays, (n,1) column vectors, (1,n) row vectors
4. Non-row-factor shapes (e.g., scalar, 2D matrix) fall back to materialization
5. Zero divisor leaves the row unscaled — matches scanpy normalize_total
   (empty rows stay empty), NOT raw scipy float division (which would yield
   inf/nan)
6. Chaining: lazy.__truediv__() appends RowScale transform
7. Chaining with normalize_total → truediv preserves lazy chain
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


class TestBackedTruedivInterception:
    """Test __truediv__ on ScxBackedSparseDataset."""

    def test_1d_array_returns_lazy(self, scx_file):
        """Dividing by a 1D array with length n_obs should return a lazy wrapper."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.random.default_rng(0).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        assert isinstance(result, pyscx.ScxLazyTransformedDataset), \
            f"Expected ScxLazyTransformedDataset, got {type(result)}"

    def test_1d_values_match(self, scx_file):
        """Lazy truediv values should match materialized X / factors."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.random.default_rng(0).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        # Materialize and compare
        mat = result.to_memory().toarray()

        # Reference: X / factors (broadcast per row)
        ref = X_ref.toarray() / factors[:, np.newaxis]

        np.testing.assert_allclose(
            mat, ref, rtol=1e-5,
            err_msg="Lazy truediv values don't match materialized reference"
        )

    def test_column_vector_returns_lazy(self, scx_file):
        """Dividing by (n_obs, 1) column vector should return a lazy wrapper."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.random.default_rng(1).uniform(0.5, 2.0, size=(n_obs, 1))
        result = adata.X / factors

        assert isinstance(result, pyscx.ScxLazyTransformedDataset)

    def test_column_vector_values_match(self, scx_file):
        """(n_obs, 1) column vector division matches reference."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.random.default_rng(1).uniform(0.5, 2.0, size=(n_obs, 1))
        result = adata.X / factors

        mat = result.to_memory().toarray()
        ref = X_ref.toarray() / factors

        np.testing.assert_allclose(mat, ref, rtol=1e-5)

    def test_row_vector_not_intercepted(self, scx_file):
        """A (1, n_obs) row vector is NOT a per-row scale (B3).

        Only `(n_obs,)` and `(n_obs, 1)` are unambiguously row-oriented. A
        `(1, n_obs)` operand is a per-column broadcast in numpy/scipy
        semantics; with a non-square matrix (n_obs != n_vars) it is not
        broadcastable, so it falls through to scipy and raises — rather than
        being silently mis-applied as a transposed RowScale.
        """
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.random.default_rng(2).uniform(0.5, 2.0, size=(1, n_obs))
        with pytest.raises(ValueError, match="inconsistent shapes"):
            adata.X / factors

    def test_scalar_falls_back(self, scx_file):
        """Dividing by a scalar should fall back to materialization (not lazy)."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        result = adata.X / 2.0

        # Scalar doesn't match n_obs, so should materialize (scipy CSR)
        assert not isinstance(result, pyscx.ScxLazyTransformedDataset), \
            "Scalar division should fall back to materialization"
        assert sp.issparse(result)

    def test_wrong_length_falls_back(self, scx_file):
        """Dividing by array with wrong length should fall back and raise."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        factors = np.ones(17)  # wrong length — not n_obs or n_vars
        # scipy will raise ValueError on shape mismatch after materialization
        with pytest.raises(ValueError, match="inconsistent shapes"):
            adata.X / factors

    def test_division_by_zero(self, scx_file):
        """Division by zero should produce 0, not inf or nan."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.ones(n_obs)
        factors[0] = 0.0  # zero divisor for row 0
        result = adata.X / factors

        assert isinstance(result, pyscx.ScxLazyTransformedDataset)

        mat = result.to_memory().toarray()
        # Row 0 should be all zeros (0 * anything = 0 in CSR)
        np.testing.assert_array_equal(
            mat[0], np.zeros(X_ref.shape[1]),
            err_msg="Division by zero should produce zeros"
        )
        # Other rows should be unchanged (divided by 1.0)
        np.testing.assert_allclose(
            mat[1:], X_ref.toarray()[1:], rtol=1e-5,
            err_msg="Non-zero divisor rows should match original"
        )

    def test_getitem_after_truediv(self, scx_file):
        """Slicing after __truediv__ should apply the transform."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        factors = np.random.default_rng(3).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        # Slice a subset and compare
        chunk = result[10:20]
        ref_chunk = X_ref[10:20].toarray() / factors[10:20, np.newaxis]

        np.testing.assert_allclose(
            chunk.toarray(), ref_chunk, rtol=1e-5,
            err_msg="__getitem__ after truediv should apply transform"
        )


class TestLazyTruedivChaining:
    """Test __truediv__ on ScxLazyTransformedDataset (chaining)."""

    def test_lazy_truediv_returns_lazy(self, scx_file):
        """Dividing a lazy dataset by row factors should return another lazy dataset."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        # First: create lazy via pyscx.accel.normalize_total
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)

        # Then: truediv on the lazy dataset
        factors = np.random.default_rng(4).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        assert isinstance(result, pyscx.ScxLazyTransformedDataset), \
            f"Expected ScxLazyTransformedDataset, got {type(result)}"

    def test_lazy_truediv_values_match(self, scx_file):
        """Chained normalize_total → truediv should match sequential application."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        # Path 1: lazy chain
        adata_lazy = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_lazy, target_sum=1e4)
        factors = np.random.default_rng(5).uniform(0.5, 2.0, size=n_obs)
        result_lazy = adata_lazy.X / factors
        mat_lazy = result_lazy.to_memory().toarray()

        # Path 2: materialized reference
        import scanpy as sc
        adata_mat = pyscx.open(path).to_anndata(backed=False)
        sc.pp.normalize_total(adata_mat, target_sum=1e4)
        ref = adata_mat.X.toarray() / factors[:, np.newaxis]

        np.testing.assert_allclose(
            mat_lazy, ref, rtol=1e-4,
            err_msg="Chained lazy normalize + truediv doesn't match materialized"
        )

    def test_backed_truediv_then_normalize(self, scx_file):
        """truediv on backed → lazy, then normalize_total appends to chain."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        n_obs = X_ref.shape[0]

        # First: truediv creates lazy
        factors1 = np.random.default_rng(6).uniform(0.5, 2.0, size=n_obs)
        adata.X = adata.X / factors1

        assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)

        # Then: normalize_total appends to the lazy chain
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)

    def test_scalar_truediv_on_lazy_falls_back(self, scx_file):
        """Dividing a lazy dataset by a scalar should materialize."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.normalize_total(adata, target_sum=1e4)

        result = adata.X / 2.0
        # Scalar doesn't match n_obs → materialization
        assert not isinstance(result, pyscx.ScxLazyTransformedDataset)


class TestAggregationAfterTruediv:
    """Test that streaming aggregation works on truediv results."""

    def test_sum_axis1_after_truediv(self, scx_file):
        """sum(axis=1) after truediv should match reference."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        factors = np.random.default_rng(7).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        lazy_sum = np.array(result.sum(axis=1)).flatten()
        ref_sum = (X_ref.toarray() / factors[:, np.newaxis]).sum(axis=1)

        np.testing.assert_allclose(
            lazy_sum, ref_sum, rtol=1e-4,
            err_msg="sum(axis=1) after truediv mismatch"
        )

    def test_sum_axis0_after_truediv(self, scx_file):
        """sum(axis=0) after truediv should match reference."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        factors = np.random.default_rng(8).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        lazy_sum = np.array(result.sum(axis=0)).flatten()
        ref_sum = (X_ref.toarray() / factors[:, np.newaxis]).sum(axis=0)

        np.testing.assert_allclose(
            lazy_sum, ref_sum, rtol=1e-4,
            err_msg="sum(axis=0) after truediv mismatch"
        )

    def test_getnnz_preserved_after_truediv(self, scx_file):
        """NNZ should be preserved after truediv (non-zeros stay non-zero)."""
        path, X_ref = scx_file
        n_obs = X_ref.shape[0]

        adata = pyscx.open(path).to_anndata(backed=True)
        factors = np.random.default_rng(9).uniform(0.5, 2.0, size=n_obs)
        result = adata.X / factors

        lazy_nnz = np.array(result.getnnz(axis=1))
        ref_nnz = np.diff(X_ref.indptr)

        np.testing.assert_array_equal(
            lazy_nnz, ref_nnz,
            err_msg="NNZ should be preserved after truediv"
        )


class TestSquareMatrixAmbiguity:
    """B3: on a square matrix (n_obs == n_vars) a bare length-n operand is
    orientation-ambiguous and must NOT be silently folded into a per-row
    RowScale."""

    def test_square_matrix_per_gene_not_row_intercepted(self):
        path, X = _make_test_scx(n_obs=40, n_vars=40, seed=11)
        adata = pyscx.open(path).to_anndata(backed=True)
        n = X.shape[0]
        v = np.random.default_rng(3).uniform(0.5, 2.0, size=n)

        result = adata.X / v
        # Ambiguous square case is not intercepted as a lazy row scale.
        assert not isinstance(result, pyscx.ScxLazyTransformedDataset)
        # Falls through to scipy per-gene (per-column) semantics.
        got = result.toarray() if sp.issparse(result) else np.asarray(result)
        expected = X.toarray() / v
        np.testing.assert_allclose(got, expected, rtol=1e-5)

    def test_square_matrix_column_vector_still_row_scale(self):
        """An explicit (n, 1) column vector is unambiguously per-row even when
        the matrix is square, so it is still intercepted lazily."""
        path, X = _make_test_scx(n_obs=40, n_vars=40, seed=12)
        adata = pyscx.open(path).to_anndata(backed=True)
        n = X.shape[0]
        v = np.random.default_rng(4).uniform(0.5, 2.0, size=(n, 1))

        result = adata.X / v
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)
