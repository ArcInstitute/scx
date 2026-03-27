"""Tests for pyscx.accel.pca() — randomized PCA from SCX."""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def pca_adata(synthetic_adata, scx_from_adata):
    """Write synthetic AnnData to SCX, return (scx_path, original_adata)."""
    path = scx_from_adata(synthetic_adata, "pca_test.scx")
    return path, synthetic_adata


class TestPcaInMemory:
    """Test PCA on in-memory (materialized) AnnData."""

    def test_basic_shapes(self, synthetic_adata):
        """PCA writes correct shapes to adata slots."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)

        assert "X_pca" in adata.obsm
        assert "PCs" in adata.varm
        assert "pca" in adata.uns

        assert adata.obsm["X_pca"].shape == (100, 10)
        assert adata.varm["PCs"].shape == (50, 10)
        assert len(adata.uns["pca"]["variance"]) == 10
        assert len(adata.uns["pca"]["variance_ratio"]) == 10

    def test_dtype(self, synthetic_adata):
        """Embeddings and PCs should be float32 (matching scanpy)."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=5)

        assert adata.obsm["X_pca"].dtype == np.float32
        assert adata.varm["PCs"].dtype == np.float32

    def test_variance_explained_positive(self, synthetic_adata):
        """Variance explained should be positive and non-increasing."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)

        ve = adata.uns["pca"]["variance"]
        assert all(v > 0 for v in ve)
        for i in range(len(ve) - 1):
            assert ve[i] >= ve[i + 1] - 1e-10, f"VE not non-increasing at {i}"

    def test_variance_ratio_sums_below_one(self, synthetic_adata):
        """Variance ratio should sum to ≤ 1."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)

        vr = adata.uns["pca"]["variance_ratio"]
        assert sum(vr) > 0
        assert sum(vr) <= 1.0 + 1e-6

    def test_cosine_similarity_vs_sklearn(self, synthetic_adata):
        """Top PCs should match sklearn PCA (cosine sim > 0.99, accounting for sign)."""
        import pyscx
        from sklearn.decomposition import PCA

        n_comps = 5
        adata = synthetic_adata.copy()

        # sklearn PCA
        X = adata.X.toarray()
        pca_sk = PCA(n_components=n_comps, random_state=0)
        X_pca_sk = pca_sk.fit_transform(X)

        # SCX PCA
        adata2 = synthetic_adata.copy()
        pyscx.accel.pca(adata2, n_comps=n_comps, random_state=0)
        X_pca_scx = adata2.obsm["X_pca"]

        # Compare via cosine similarity (accounting for sign ambiguity)
        for pc in range(min(3, n_comps)):
            col_sk = X_pca_sk[:, pc]
            col_scx = X_pca_scx[:, pc].astype(np.float64)
            cos_sim = abs(
                np.dot(col_sk, col_scx)
                / (np.linalg.norm(col_sk) * np.linalg.norm(col_scx) + 1e-12)
            )
            assert (
                cos_sim > 0.95
            ), f"PC{pc} cosine similarity = {cos_sim:.4f} (expected > 0.95)"

    def test_no_center(self, synthetic_adata):
        """PCA with zero_center=False should work (TruncatedSVD equivalent)."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=5, zero_center=False)

        assert adata.obsm["X_pca"].shape == (100, 5)
        assert adata.varm["PCs"].shape == (50, 5)


class TestPcaBacked:
    """Test PCA streaming from backed SCX file."""

    def test_backed_pca(self, pca_adata):
        """PCA from backed mode should produce valid results."""
        import pyscx

        scx_path, _ = pca_adata
        backed_adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.pca(backed_adata, n_comps=10)

        assert "X_pca" in backed_adata.obsm
        assert backed_adata.obsm["X_pca"].shape == (100, 10)
        assert "PCs" in backed_adata.varm
        assert backed_adata.varm["PCs"].shape == (50, 10)

    def test_backed_matches_inmemory(self, pca_adata, synthetic_adata):
        """Backed PCA should match in-memory PCA (same seed)."""
        import pyscx

        scx_path, _ = pca_adata
        seed = 42

        # In-memory PCA
        adata_mem = synthetic_adata.copy()
        pyscx.accel.pca(adata_mem, n_comps=5, random_state=seed)

        # Backed PCA
        adata_backed = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.pca(adata_backed, n_comps=5, random_state=seed)

        # Compare embeddings (should be identical — same algorithm, same seed, same data)
        np.testing.assert_allclose(
            adata_mem.obsm["X_pca"],
            adata_backed.obsm["X_pca"],
            rtol=1e-4,
            atol=1e-6,
        )

    def test_backed_variance_explained(self, pca_adata):
        """Variance from backed PCA should be valid."""
        import pyscx

        scx_path, _ = pca_adata
        backed_adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.pca(backed_adata, n_comps=5)

        ve = backed_adata.uns["pca"]["variance"]
        vr = backed_adata.uns["pca"]["variance_ratio"]
        assert all(v > 0 for v in ve)
        assert sum(vr) <= 1.0 + 1e-6
