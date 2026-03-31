"""Tests for B4: normalize_total respects col_projection (scanpy compat).

After filter_genes(), normalize_total() should compute row sums only over
the visible (projected) gene set, matching scanpy's behavior where adata.X
is already sliced to kept genes.

Tests both code paths:
  1. Backed (ScxBackedSparseDataset) → new lazy wrapper
  2. Lazy (ScxLazyTransformedDataset) → appended transform
"""

import numpy as np
import scipy.sparse as sp
import pytest
import tempfile
import shutil
import os

import pyscx


def _make_test_scx(tmpdir, n_obs=100, n_vars=50, density=0.3, seed=42):
    """Create a temp SCX file with random integer sparse data."""
    X = sp.random(n_obs, n_vars, density=density, random_state=seed,
                  format='csr', dtype=np.float32)
    X.data = np.ceil(X.data * 100).astype(np.float32)
    X.eliminate_zeros()

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
    tmpdir = tempfile.mkdtemp()
    path, X = _make_test_scx(tmpdir)
    yield path, X
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestNormalizeTotalColProjectionBacked:
    """B4: normalize_total on backed data after filter_genes() should match scanpy."""

    def test_filter_genes_then_normalize_matches_scanpy(self, scx_file):
        """After filter_genes(), normalize_total should sum only visible genes."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file
        n_obs, n_vars = X_ref.shape

        # --- Reference: scanpy on materialized data ---
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.filter_genes(adata_ref, min_cells=3)
        kept_var_count = adata_ref.n_vars
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        ref_X = adata_ref.X

        # --- SCX backed path: filter_genes → normalize_total ---
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.filter_genes(adata, min_cells=3)
        assert adata.X.shape[1] == kept_var_count, "filter_genes should reduce n_vars"

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert adata.X.shape == (n_obs, kept_var_count), \
            "normalize_total should preserve shape after filter_genes"

        # Materialize and compare
        lazy_X = adata.X.to_memory()
        np.testing.assert_allclose(
            lazy_X.toarray(),
            ref_X.toarray(),
            atol=1e-3,
            err_msg="B4: backed: filter_genes → normalize_total mismatch vs scanpy"
        )

    def test_row_sums_after_filter_and_normalize(self, scx_file):
        """Row sums should equal target_sum (using only projected genes)."""
        path, X_ref = scx_file

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.filter_genes(adata, min_cells=3)

        target = 1e4
        pyscx.accel.normalize_total(adata, target_sum=target)

        lazy_X = adata.X.to_memory()
        row_sums = np.array(lazy_X.sum(axis=1)).flatten()

        # All rows should have sum == target (assuming no zero-sum rows
        # after projection; our density=0.3 * 50 genes guarantees this)
        np.testing.assert_allclose(
            row_sums,
            target,
            rtol=1e-5,
            err_msg="Row sums should equal target_sum after filter_genes+normalize"
        )

    def test_no_col_projection_unchanged(self, scx_file):
        """Without filter_genes, normalize_total still sums all columns."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)

        # SCX backed (no filter_genes)
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        lazy_X = adata.X.to_memory()
        np.testing.assert_allclose(
            lazy_X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-3,
            err_msg="Without filter_genes, normalize_total should match scanpy"
        )


class TestNormalizeTotalColProjectionLazy:
    """B4: normalize_total on lazy data (chained) after filter_genes()."""

    def test_log1p_then_filter_then_normalize_matches(self, scx_file):
        """Chained: log1p → filter_genes → normalize_total on lazy data."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # --- Reference: scanpy pipeline ---
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.log1p(adata_ref)
        sc.pp.filter_genes(adata_ref, min_cells=3)
        sc.pp.normalize_total(adata_ref, target_sum=1e4)

        # --- SCX lazy pipeline ---
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)
        assert type(adata.X).__name__ == 'ScxLazyTransformedDataset'

        pyscx.accel.filter_genes(adata, min_cells=3)
        assert adata.X.shape[1] == adata_ref.n_vars

        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # Materialize and compare
        lazy_X = adata.X.to_memory()
        np.testing.assert_allclose(
            lazy_X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-3,
            err_msg="B4: lazy: log1p → filter_genes → normalize_total mismatch"
        )

    def test_normalize_then_filter_then_normalize(self, scx_file):
        """Double normalize with filter in between — unusual but valid."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.filter_genes(adata_ref, min_cells=3)
        sc.pp.normalize_total(adata_ref, target_sum=1e4)

        # SCX path
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.filter_genes(adata, min_cells=3)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        lazy_X = adata.X.to_memory()
        np.testing.assert_allclose(
            lazy_X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-3,
            err_msg="normalize → filter → normalize mismatch"
        )

    def test_streaming_sum_axis1_after_filter_normalize(self, scx_file):
        """Streaming sum(axis=1) should match materialized after filter+normalize."""
        path, _ = scx_file

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.filter_genes(adata, min_cells=3)
        pyscx.accel.normalize_total(adata, target_sum=1e4)

        # Streaming
        sums = np.array(adata.X.sum(axis=1)).flatten()

        # Materialized
        ref_sums = np.array(adata.X.to_memory().sum(axis=1)).flatten()

        np.testing.assert_allclose(
            sums, ref_sums, rtol=1e-5,
            err_msg="Streaming sum(axis=1) after filter+normalize mismatch"
        )
