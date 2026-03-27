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


class TestNeighbors:
    """Test pyscx.accel.neighbors() — kNN graph construction."""

    def test_basic_shapes(self, synthetic_adata):
        """Neighbors writes correct shapes and keys to adata slots."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=5)

        assert "distances" in adata.obsp
        assert "connectivities" in adata.obsp
        assert "neighbors" in adata.uns

        # Distance matrix: n_obs × n_obs sparse
        assert adata.obsp["distances"].shape == (100, 100)
        assert adata.obsp["connectivities"].shape == (100, 100)

    def test_distance_matrix_properties(self, synthetic_adata):
        """Distance CSR should have non-negative values and k entries per row."""
        import pyscx

        k = 10
        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=k)

        dist = adata.obsp["distances"]

        # All distances non-negative
        assert (dist.data >= 0).all(), "distances should be non-negative"

        # Each row has exactly k entries
        for i in range(dist.shape[0]):
            row_nnz = dist.indptr[i + 1] - dist.indptr[i]
            assert row_nnz == k, f"row {i} has {row_nnz} entries, expected {k}"

    def test_connectivities_properties(self, synthetic_adata):
        """Connectivities should be in [0, 1] and have >= k entries per row (symmetrized)."""
        import pyscx

        k = 5
        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=k)

        conn = adata.obsp["connectivities"]

        # All values in [0, 1]
        assert (conn.data >= 0).all(), "connectivities should be >= 0"
        assert (conn.data <= 1.0 + 1e-10).all(), "connectivities should be <= 1"

        # Symmetrized: at least k entries per row
        for i in range(conn.shape[0]):
            row_nnz = conn.indptr[i + 1] - conn.indptr[i]
            assert row_nnz >= k, f"row {i} has {row_nnz} entries, expected >= {k}"

    def test_recall_vs_brute_force(self, synthetic_adata):
        """HNSW recall@k should be >= 0.90 compared to exact brute-force kNN."""
        import pyscx
        from sklearn.neighbors import NearestNeighbors

        k = 10
        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)

        X_pca = adata.obsm["X_pca"]

        # Brute-force exact kNN
        nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
        nn.fit(X_pca)
        _, bf_indices = nn.kneighbors(X_pca)

        # SCX HNSW kNN
        pyscx.accel.neighbors(adata, n_neighbors=k)

        # Extract SCX indices from the distance CSR matrix
        dist = adata.obsp["distances"]
        recalls = []
        for i in range(adata.n_obs):
            start, end = dist.indptr[i], dist.indptr[i + 1]
            scx_neighbors = set(dist.indices[start:end])
            bf_neighbors = set(bf_indices[i])
            recall = len(scx_neighbors & bf_neighbors) / k
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        assert (
            mean_recall >= 0.90
        ), f"mean recall@{k} = {mean_recall:.3f} (expected >= 0.90)"

    def test_uns_metadata(self, synthetic_adata):
        """adata.uns['neighbors'] should have expected structure."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=15)

        nb = adata.uns["neighbors"]
        assert nb["connectivities_key"] == "connectivities"
        assert nb["distances_key"] == "distances"
        assert nb["params"]["n_neighbors"] == 15
        assert nb["params"]["method"] == "hnsw"
        assert nb["params"]["use_rep"] == "X_pca"

    def test_leiden_on_scx_graph(self, synthetic_adata):
        """Leiden clustering should work on SCX-built kNN graph."""
        import pyscx

        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)

        # Leiden should work on our graph
        sc.tl.leiden(adata, key_added="scx_leiden")
        assert "scx_leiden" in adata.obs.columns
        n_clusters = adata.obs["scx_leiden"].nunique()
        assert n_clusters >= 2, f"expected >= 2 clusters, got {n_clusters}"

    def test_backed_neighbors(self, pca_adata):
        """kNN from backed mode should work (PCA → neighbors pipeline)."""
        import pyscx

        scx_path, _ = pca_adata
        backed_adata = pyscx.open(scx_path).to_anndata(backed=True)

        # PCA from backed mode
        pyscx.accel.pca(backed_adata, n_comps=10)
        # Neighbors from the PCA result
        pyscx.accel.neighbors(backed_adata, n_neighbors=5)

        assert "distances" in backed_adata.obsp
        assert backed_adata.obsp["distances"].shape == (100, 100)
        assert "connectivities" in backed_adata.obsp
        assert backed_adata.obsp["connectivities"].shape == (100, 100)


class TestUmap:
    """Test pyscx.accel.umap() — UMAP embedding."""

    def test_basic_shapes(self, synthetic_adata):
        """UMAP writes correct shapes to adata.obsm['X_umap']."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.umap(adata)

        assert "X_umap" in adata.obsm
        assert adata.obsm["X_umap"].shape == (100, 2)
        assert adata.obsm["X_umap"].dtype == np.float32

    def test_embeddings_finite(self, synthetic_adata):
        """All embedding values should be finite and not all identical."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.umap(adata)

        emb = adata.obsm["X_umap"]
        assert np.all(np.isfinite(emb)), "all embedding values should be finite"
        assert emb.std() > 0, "embeddings should not all be identical"

    def test_trustworthiness(self, synthetic_adata):
        """Trustworthiness metric should be > 0.75 (local neighborhoods preserved)."""
        import pyscx
        from sklearn.manifold import trustworthiness

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.umap(adata, n_epochs=500, random_state=42)

        X_pca = adata.obsm["X_pca"]
        X_umap = adata.obsm["X_umap"]

        tw = trustworthiness(X_pca, X_umap, n_neighbors=10)
        assert tw > 0.75, f"trustworthiness = {tw:.4f} (expected > 0.75)"

    def test_backed_umap(self, pca_adata):
        """Full pipeline from backed SCX: PCA → neighbors → UMAP."""
        import pyscx

        scx_path, _ = pca_adata
        backed_adata = pyscx.open(scx_path).to_anndata(backed=True)

        pyscx.accel.pca(backed_adata, n_comps=10)
        pyscx.accel.neighbors(backed_adata, n_neighbors=5)
        pyscx.accel.umap(backed_adata, n_epochs=100)

        assert "X_umap" in backed_adata.obsm
        assert backed_adata.obsm["X_umap"].shape == (100, 2)
        assert np.all(np.isfinite(backed_adata.obsm["X_umap"]))
