"""Tests for pyscx.accel.normalize_total() — Step 3 of Phase 4d.

Tests verify:
1. normalize_total on backed ScxBackedSparseDataset creates ScxLazyTransformedDataset
2. normalize_total on existing ScxLazyTransformedDataset appends transform
3. normalize_total on scipy CSR delegates to scanpy
4. Output matches scanpy's normalize_total within tolerance
5. Type checks and shape preservation
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
    # Make integer-valued (like UMI counts)
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


class TestNormalizeTotalBacked:
    """Test normalize_total on ScxBackedSparseDataset."""

    def test_creates_lazy_wrapper(self, scx_file):
        """normalize_total should replace X with ScxLazyTransformedDataset."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        assert type(adata.X).__name__ == 'ScxBackedSparseDataset'

        pyscx.accel.normalize_total(adata, target_sum=1e4)

        assert type(adata.X).__name__ == 'ScxLazyTransformedDataset'

    def test_shape_preserved(self, scx_file):
        """Shape should be unchanged after normalize_total."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        orig_shape = adata.X.shape

        pyscx.accel.normalize_total(adata, target_sum=1e4)

        assert adata.X.shape == orig_shape

    def test_repr_shows_transform(self, scx_file):
        """Repr should include NormalizeTotal."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.normalize_total(adata, target_sum=1e4)

        r = repr(adata.X)
        assert 'NormalizeTotal' in r

    def test_matches_scanpy(self, scx_file):
        """Lazy normalize_total should match scanpy's normalize_total."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference: scanpy normalize_total on materialized data
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        ref_X = adata_ref.X

        # Lazy: pyscx.accel.normalize_total on backed data
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # Materialize the lazy result
        lazy_X = adata.X.to_memory()

        np.testing.assert_allclose(
            lazy_X.toarray(),
            ref_X.toarray(),
            atol=1e-3,
            err_msg="normalize_total: lazy vs scanpy mismatch"
        )

    def test_matches_scanpy_default_target(self, scx_file):
        """Default target_sum=None uses scanpy's median-of-positive-totals.
        """
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref)  # scanpy default: None → median

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)  # new default: None → median

        lazy_X = adata.X.to_memory()

        np.testing.assert_allclose(
            lazy_X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-3,
            err_msg="default (None → median) mismatch vs scanpy default",
        )

    def test_default_differs_from_1e4(self, scx_file):
        """Sanity: the new None-median default is NOT the old 1e4 behavior."""
        path, X_ref = scx_file

        orig_sums = np.asarray(X_ref.sum(axis=1)).ravel()
        expected_median = float(np.median(orig_sums[orig_sums > 0]))
        # Fixture median must be distinct from 1e4 for this test to be meaningful.
        assert abs(expected_median - 1e4) > 1.0

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)  # None → median
        row_sums = np.asarray(adata.X.to_memory().sum(axis=1)).ravel()

        nonzero = orig_sums > 0
        # Non-zero rows now sum to the median, not 1e4.
        np.testing.assert_allclose(
            row_sums[nonzero], expected_median, rtol=1e-5,
            err_msg="None default should scale nonzero rows to the median total",
        )

    def test_custom_target_sum(self, scx_file):
        """Custom target_sum should be applied correctly."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        for target in [1.0, 1e6, 100.0]:
            adata_ref = anndata.AnnData(X=X_ref.copy())
            sc.pp.normalize_total(adata_ref, target_sum=target)

            adata = pyscx.open(path).to_anndata(backed=True)
            pyscx.accel.normalize_total(adata, target_sum=target)

            lazy_X = adata.X.to_memory()
            np.testing.assert_allclose(
                lazy_X.toarray(),
                adata_ref.X.toarray(),
                atol=5e-2,
                err_msg=f"target_sum={target} mismatch"
            )

    def test_row_sums_match_target(self, scx_file):
        """After normalize_total, row sums should equal target_sum for non-zero rows."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        target = 1e4
        pyscx.accel.normalize_total(adata, target_sum=target)

        # Materialize and check row sums
        lazy_X = adata.X.to_memory()
        row_sums = np.array(lazy_X.sum(axis=1)).flatten()

        # For rows with non-zero original sum, result should be ~target_sum
        orig_sums = np.array(X_ref.sum(axis=1)).flatten()
        nonzero_mask = orig_sums > 0
        np.testing.assert_allclose(
            row_sums[nonzero_mask],
            target,
            rtol=1e-5,
            err_msg="row sums should equal target_sum"
        )

    def test_getitem_sliced(self, scx_file):
        """X[100:200] should work on normalized data."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        ref_slice = adata_ref.X[5:15]

        # Lazy
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        lazy_slice = adata.X[5:15]

        np.testing.assert_allclose(
            lazy_slice.toarray(),
            ref_slice.toarray(),
            atol=1e-3,
            err_msg="sliced access after normalize_total mismatch"
        )


class TestNormalizeTotalChaining:
    """Test chaining normalize_total with another lazy wrapper."""

    def test_append_to_existing_lazy(self, scx_file):
        """normalize_total on ScxLazyTransformedDataset should append transform."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        # First normalize
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert type(adata.X).__name__ == 'ScxLazyTransformedDataset'

        # Second normalize (unusual but valid — tests chaining)
        pyscx.accel.normalize_total(adata, target_sum=1.0)
        r = repr(adata.X)
        # Should have two NormalizeTotal transforms
        assert r.count('NormalizeTotal') == 2


class TestNormalizeTotalScipy:
    """Test normalize_total fallback on regular scipy CSR."""

    def test_scipy_fallback(self, scx_file):
        """Should delegate to scanpy for scipy CSR data."""
        import scanpy as sc
        import anndata

        _, X_ref = scx_file

        # Direct scipy CSR in AnnData
        adata = anndata.AnnData(X=X_ref.copy())
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)

        np.testing.assert_allclose(
            adata.X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-5,
            err_msg="scipy CSR fallback mismatch"
        )

    def test_scipy_fallback_none_median(self, scx_file):
        """scipy path with default None forwards to scanpy's median default."""
        import scanpy as sc
        import anndata

        _, X_ref = scx_file

        adata = anndata.AnnData(X=X_ref.copy())
        pyscx.accel.normalize_total(adata)  # None → scanpy median

        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref)  # scanpy default

        np.testing.assert_allclose(
            adata.X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-5,
            err_msg="scipy CSR fallback (None median) mismatch",
        )


class TestNormalizeTotalIssparse:
    """Test that issparse works on lazy-transformed data."""

    def test_issparse_after_normalize(self, scx_file):
        """scipy.sparse.issparse should return True after normalize_total."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # The lazy wrapper should be registered with anndata ABC
        # and have format='csr', ndim=2
        assert adata.X.format == 'csr'
        assert adata.X.ndim == 2

    def test_dtype_after_normalize(self, scx_file):
        """dtype should be float32 after normalize."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        assert str(adata.X.dtype) == 'float32'


class TestStreamingAggAfterNormalize:
    """Test streaming aggregation on normalized data."""

    def test_sum_axis1_after_normalize(self, scx_file):
        """sum(axis=1) on normalized data should return target_sum for nonzero rows."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        target = 1e4
        pyscx.accel.normalize_total(adata, target_sum=target)

        # Streaming sum(axis=1) through transforms
        sums = np.array(adata.X.sum(axis=1)).flatten()

        # Reference: materialize and compute
        lazy_X = adata.X.to_memory()
        ref_sums = np.array(lazy_X.sum(axis=1)).flatten()

        np.testing.assert_allclose(
            sums, ref_sums, rtol=1e-5,
            err_msg="streaming sum(axis=1) after normalize mismatch"
        )

    def test_sum_axis0_after_normalize(self, scx_file):
        """sum(axis=0) on normalized data should match materialized."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        ref_sums = np.array(adata_ref.X.sum(axis=0)).flatten()

        # Lazy streaming
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        lazy_sums = np.array(adata.X.sum(axis=0)).flatten()

        np.testing.assert_allclose(
            lazy_sums, ref_sums, rtol=1e-5,
            err_msg="streaming sum(axis=0) after normalize mismatch"
        )

    def test_getnnz_preserved(self, scx_file):
        """NNZ should be unchanged by normalize_total (nonzeros stay nonzero)."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        nnz_before = adata.X.getnnz(axis=0)

        pyscx.accel.normalize_total(adata, target_sum=1e4)

        nnz_after = adata.X.getnnz(axis=0)

        np.testing.assert_array_equal(
            np.array(nnz_before), np.array(nnz_after),
            err_msg="NNZ should be unchanged by normalize_total"
        )
