"""Integration tests for SCX backed mode (Phase 5).

Tests the backed (lazy-loading) AnnData integration, validating:
- Round-trip correctness (backed X[:] matches non-backed)
- Row slicing, fancy indexing, boolean masks, column filtering
- to_memory() materializationfrom backed
- Layer backed access
- Deletion vector support
- anndata.abc.CSRDataset isinstance check
- Full scanpy pipeline on backed data
- Memory efficiency vs full materialization
- Cache configuration options
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_scx(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic_adata for backed tests."""
    import pyscx

    path = str(tmp_dir / "backed_test.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def adata_non_backed(backed_scx):
    """Load the full (non-backed) AnnData for comparison."""
    import pyscx

    return pyscx.open(backed_scx).to_anndata()


@pytest.fixture
def adata_backed(backed_scx):
    """Load the backed AnnData."""
    import pyscx

    return pyscx.open(backed_scx).to_anndata(backed=True)


def test_backed_roundtrip(adata_backed, adata_non_backed):
    """Backed X[:] matches non-backed X element-wise."""
    backed_full = adata_backed.X[:]
    assert isinstance(backed_full, sp.csr_matrix)
    assert backed_full.shape == adata_non_backed.X.shape

    # Element-wise comparison
    np.testing.assert_array_equal(
        backed_full.toarray(), adata_non_backed.X.toarray()
    )


def test_backed_row_slice(adata_backed, adata_non_backed):
    """X[10:20] matches non-backed equivalent."""
    backed_slice = adata_backed.X[10:20]
    expected = adata_non_backed.X[10:20]

    assert backed_slice.shape == expected.shape
    np.testing.assert_array_equal(backed_slice.toarray(), expected.toarray())


def test_backed_fancy_index(adata_backed, adata_non_backed):
    """X[[0, 5, 10]] returns correct rows."""
    indices = [0, 5, 10]
    backed_fancy = adata_backed.X[indices]
    expected = adata_non_backed.X[indices]

    assert backed_fancy.shape == expected.shape
    np.testing.assert_array_equal(backed_fancy.toarray(), expected.toarray())


def test_backed_boolean_mask(adata_backed, adata_non_backed):
    """X[mask] returns correct subset."""
    n_obs = adata_non_backed.X.shape[0]
    np.random.seed(42)
    mask = np.random.choice([True, False], size=n_obs)

    backed_masked = adata_backed.X[mask]
    expected = adata_non_backed.X[mask]

    assert backed_masked.shape == expected.shape
    np.testing.assert_array_equal(backed_masked.toarray(), expected.toarray())


def test_backed_column_slice(adata_backed, adata_non_backed):
    """X[10:20, :500] filters columns correctly."""
    # Note: adata_non_backed has 50 vars, so use :25
    backed_2d = adata_backed.X[10:20, :25]
    expected = adata_non_backed.X[10:20, :25]

    assert backed_2d.shape == expected.shape
    np.testing.assert_array_equal(backed_2d.toarray(), expected.toarray())


def test_backed_to_memory(adata_backed, adata_non_backed):
    """to_memory() matches full to_anndata()."""
    materialized = adata_backed.X.to_memory()
    assert isinstance(materialized, sp.csr_matrix)
    np.testing.assert_array_equal(
        materialized.toarray(), adata_non_backed.X.toarray()
    )


def test_backed_layers(backed_scx, adata_non_backed):
    """Layer backed access works and matches non-backed."""
    import pyscx

    adata = pyscx.open(backed_scx).to_anndata(backed=True)

    # Check that the layer exists and is backed
    assert "raw" in adata.layers
    layer_type = type(adata.layers["raw"]).__name__
    assert "Backed" in layer_type or "Dataset" in layer_type

    # Full slice should match non-backed
    backed_layer = adata.layers["raw"][:]
    expected_layer = adata_non_backed.layers["raw"]

    np.testing.assert_array_equal(
        backed_layer.toarray(), expected_layer.toarray()
    )

    # Row slice should match
    backed_layer_slice = adata.layers["raw"][5:15]
    expected_layer_slice = adata_non_backed.layers["raw"][5:15]
    np.testing.assert_array_equal(
        backed_layer_slice.toarray(), expected_layer_slice.toarray()
    )


def test_backed_with_deletions(tmp_dir):
    """Backed mode on file with mark_deleted() excludes deleted rows."""
    import anndata
    import pyscx

    np.random.seed(99)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "backed_del.scx")
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True
    n_deleted = int(delete_mask.sum())

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)

    # Non-backed should exclude deleted rows
    adata_full = pyscx.open(path).to_anndata()
    assert adata_full.n_obs == n_obs - n_deleted

    # Backed should also exclude deleted rows
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.X.shape[0] == n_obs - n_deleted

    # Backed full read should match non-backed
    backed_full = adata_backed.X[:]
    np.testing.assert_array_equal(
        backed_full.toarray(), adata_full.X.toarray()
    )


def test_backed_obsm_with_deletions(tmp_dir):
    """obsm arrays are filtered by deletion vectors (shape matches obs)."""
    import anndata
    import pyscx

    np.random.seed(42)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    obsm = {"X_pca": np.random.randn(n_obs, 10).astype(np.float32)}
    adata = anndata.AnnData(X=x, obsm=obsm)

    path = str(tmp_dir / "backed_obsm_del.scx")
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True
    n_deleted = int(delete_mask.sum())
    n_kept = n_obs - n_deleted

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)

    # Non-backed: obs and obsm should have matching shapes
    adata_full = pyscx.open(path).to_anndata()
    assert adata_full.n_obs == n_kept
    assert adata_full.obsm["X_pca"].shape[0] == n_kept

    # Backed: obs and obsm should also have matching shapes
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.n_obs == n_kept
    assert adata_backed.obsm["X_pca"].shape[0] == n_kept

    # obsm values should match between backed and non-backed
    np.testing.assert_array_equal(
        adata_backed.obsm["X_pca"], adata_full.obsm["X_pca"]
    )


def test_backed_isinstance(adata_backed):
    """isinstance(adata.X, anndata.abc.CSRDataset) is True."""
    import anndata.abc

    assert isinstance(adata_backed.X, anndata.abc.CSRDataset)


def test_backed_scanpy_pipeline(backed_scx):
    """Full scanpy pipeline works with backed mode.

    Backed mode is read-only, so we subset → copy → preprocess.
    """
    import pyscx
    import scanpy as sc

    adata = pyscx.open(backed_scx).to_anndata(backed=True)

    # Subset (first 50 cells) and materialize
    adata_sub = adata[:50].copy()

    # Standard scanpy pipeline on materialized data
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.pca(adata_sub)

    # Verify PCA ran
    assert "X_pca" in adata_sub.obsm
    assert adata_sub.obsm["X_pca"].shape[0] == 50


def test_backed_memory(synthetic_adata, tmp_dir):
    """Backed mode peak memory << full materialization.

    We verify that the backed object itself is lightweight by checking
    that the X attribute does not hold a materialized array.
    """
    import sys

    import pyscx

    path = str(tmp_dir / "backed_mem.scx")
    pyscx.from_anndata(synthetic_adata, path)

    # Backed: X is a lightweight proxy object
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    x_backed_size = sys.getsizeof(adata_backed.X)

    # Non-backed: X is a full scipy sparse matrix
    adata_full = pyscx.open(path).to_anndata()
    # The full X stores actual data arrays
    full_data_size = (
        adata_full.X.data.nbytes
        + adata_full.X.indices.nbytes
        + adata_full.X.indptr.nbytes
    )

    # Backed proxy should be much smaller than the actual data
    assert x_backed_size < full_data_size


def test_backed_cache_config(synthetic_adata, tmp_dir):
    """cache_shards=0 and cache_shards=16 both work."""
    import pyscx

    path = str(tmp_dir / "backed_cache.scx")
    pyscx.from_anndata(synthetic_adata, path)

    # No cache
    adata0 = pyscx.open(path).to_anndata(backed=True, cache_shards=0)
    slice0 = adata0.X[0:10]
    assert slice0.shape[0] == 10

    # Large cache
    adata16 = pyscx.open(path).to_anndata(backed=True, cache_shards=16)
    slice16 = adata16.X[0:10]
    assert slice16.shape[0] == 10

    # Both should return the same data
    np.testing.assert_array_equal(slice0.toarray(), slice16.toarray())
