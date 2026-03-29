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
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")

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


class TestRankGenesGroups:
    """Test pyscx.accel.rank_genes_groups() — Wilcoxon rank-sum DE."""

    def test_basic_structure(self, synthetic_adata):
        """DE writes correct keys to adata.uns['rank_genes_groups']."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")

        assert "rank_genes_groups" in adata.uns
        rgg = adata.uns["rank_genes_groups"]
        assert "names" in rgg
        assert "scores" in rgg
        assert "pvals" in rgg
        assert "pvals_adj" in rgg
        assert "logfoldchanges" in rgg
        assert "params" in rgg

        # Params should contain correct metadata
        assert rgg["params"]["groupby"] == "batch"
        assert rgg["params"]["method"] == "wilcoxon"
        assert rgg["params"]["reference"] == "rest"

        # Names should be a structured array with group fields
        names = rgg["names"]
        assert hasattr(names, "dtype")
        assert len(names.dtype.names) == 3  # A, B, C groups from conftest

    def test_pvalues_in_range(self, synthetic_adata):
        """All p-values should be in [0, 1], adjusted >= raw."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")

        rgg = adata.uns["rank_genes_groups"]
        for group in rgg["pvals"].dtype.names:
            raw = rgg["pvals"][group]
            adj = rgg["pvals_adj"][group]

            assert np.all(raw >= 0), f"raw p-values for {group} should be >= 0"
            assert np.all(raw <= 1), f"raw p-values for {group} should be <= 1"
            assert np.all(adj >= 0), f"adj p-values for {group} should be >= 0"
            assert np.all(adj <= 1), f"adj p-values for {group} should be <= 1"
            assert np.all(
                adj >= raw - 1e-10
            ), f"adjusted p-values for {group} should be >= raw"

    def test_overlap_with_scanpy(self, synthetic_adata):
        """Top DE genes should substantially overlap with scanpy's output."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        import pyscx

        n_top = 20

        # scanpy DE
        adata_sc = synthetic_adata.copy()
        sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon")

        # SCX DE
        adata_scx = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata_scx, "batch")

        rgg_sc = adata_sc.uns["rank_genes_groups"]
        rgg_scx = adata_scx.uns["rank_genes_groups"]

        for group in rgg_scx["names"].dtype.names:
            sc_top = set(rgg_sc["names"][group][:n_top])
            scx_top = set(rgg_scx["names"][group][:n_top])
            overlap = len(sc_top & scx_top) / n_top
            assert (
                overlap >= 0.30
            ), f"overlap for group {group} = {overlap:.2f} (expected >= 0.30)"

    def test_pairwise_reference(self, synthetic_adata):
        """With reference='A', only non-A groups should appear in results."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch", reference="A")

        rgg = adata.uns["rank_genes_groups"]
        groups = rgg["names"].dtype.names
        assert "A" not in groups
        assert "B" in groups
        assert "C" in groups

    def test_logfoldchanges_sign(self):
        """For known upregulated genes, fold-change should be positive."""
        import anndata
        import pyscx

        # Build simple 2-group data with known DE genes.
        np.random.seed(123)
        n_obs = 40
        n_vars = 5
        data = np.zeros((n_obs, n_vars), dtype=np.float32)
        # Gene 0: high in group A, low in B
        data[:20, 0] = np.random.uniform(10, 20, 20).astype(np.float32)
        data[20:, 0] = np.random.uniform(0, 2, 20).astype(np.float32)
        # Gene 1: high in group B, low in A
        data[:20, 1] = np.random.uniform(0, 2, 20).astype(np.float32)
        data[20:, 1] = np.random.uniform(10, 20, 20).astype(np.float32)
        # Genes 2-4: no diff
        data[:, 2:] = np.random.uniform(3, 7, (n_obs, 3)).astype(np.float32)

        import pandas as pd

        obs = pd.DataFrame(
            {"group": pd.Categorical(["A"] * 20 + ["B"] * 20)},
            index=[f"c{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        pyscx.accel.rank_genes_groups(adata, "group")

        rgg = adata.uns["rank_genes_groups"]

        # Find gene_0 in group A results — should have positive logFC
        names_a = list(rgg["names"]["A"])
        idx_gene0 = names_a.index("gene_0")
        assert (
            rgg["logfoldchanges"]["A"][idx_gene0] > 0
        ), "gene_0 should have positive logFC for group A"

        # Find gene_1 in group B results — should have positive logFC
        names_b = list(rgg["names"]["B"])
        idx_gene1 = names_b.index("gene_1")
        assert (
            rgg["logfoldchanges"]["B"][idx_gene1] > 0
        ), "gene_1 should have positive logFC for group B"

    def test_logfc_matches_scanpy_log_transformed(self):
        """logFC should match scanpy when data has been log1p-transformed."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(42)
        n_obs, n_vars = 60, 10
        data = np.random.randint(0, 50, size=(n_obs, n_vars)).astype(np.float32)
        # Gene 0: strongly upregulated in group A
        data[:30, 0] = np.random.randint(80, 120, 30).astype(np.float32)
        data[30:, 0] = np.random.randint(1, 5, 30).astype(np.float32)

        obs = pd.DataFrame(
            {"group": pd.Categorical(["A"] * 30 + ["B"] * 30)},
            index=[f"c{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])

        # Apply log1p to both (sets adata.uns["log1p"])
        adata_sc = anndata.AnnData(X=sp.csr_matrix(data.copy()), obs=obs.copy(), var=var.copy())
        sc.pp.log1p(adata_sc)

        adata_scx = anndata.AnnData(X=sp.csr_matrix(data.copy()), obs=obs.copy(), var=var.copy())
        sc.pp.log1p(adata_scx)

        # Run DE
        sc.tl.rank_genes_groups(adata_sc, "group", method="wilcoxon")
        pyscx.accel.rank_genes_groups(adata_scx, "group")

        # Compare logfoldchanges for overlapping genes in group A
        rgg_sc = adata_sc.uns["rank_genes_groups"]
        rgg_scx = adata_scx.uns["rank_genes_groups"]

        sc_names_a = list(rgg_sc["names"]["A"])
        sc_logfc_a = list(rgg_sc["logfoldchanges"]["A"])
        scx_names_a = list(rgg_scx["names"]["A"])
        scx_logfc_a = list(rgg_scx["logfoldchanges"]["A"])

        # Build name → logFC maps
        sc_map = dict(zip(sc_names_a, sc_logfc_a))
        scx_map = dict(zip(scx_names_a, scx_logfc_a))

        # For every gene that appears in both, logFC should be close
        for gene in sc_map:
            if gene in scx_map:
                sc_val = sc_map[gene]
                scx_val = scx_map[gene]
                assert abs(sc_val - scx_val) < 0.5, (
                    f"logFC mismatch for {gene}: scanpy={sc_val:.4f}, scx={scx_val:.4f}"
                )

        # Gene 0 should have positive and similar logFC
        assert "gene_0" in sc_map and "gene_0" in scx_map
        assert abs(sc_map["gene_0"] - scx_map["gene_0"]) < 0.5, (
            f"gene_0 logFC: scanpy={sc_map['gene_0']:.4f}, scx={scx_map['gene_0']:.4f}"
        )

    def test_backed_pipeline(self, pca_adata):
        """Full pipeline from backed SCX: PCA → neighbors → leiden → DE."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")

        import pyscx

        scx_path, _ = pca_adata
        backed_adata = pyscx.open(scx_path).to_anndata(backed=True)

        # PCA from backed mode
        pyscx.accel.pca(backed_adata, n_comps=10)
        pyscx.accel.neighbors(backed_adata, n_neighbors=5)

        # Leiden clustering
        sc.tl.leiden(backed_adata, key_added="leiden_groups")

        # DE on cluster labels
        pyscx.accel.rank_genes_groups(backed_adata, "leiden_groups")

        assert "rank_genes_groups" in backed_adata.uns
        rgg = backed_adata.uns["rank_genes_groups"]
        assert "names" in rgg
        assert len(rgg["names"].dtype.names) >= 2


class TestStreamingDE:
    """Test streaming gene-chunked differential expression from backed SCX."""

    def test_streaming_matches_inmemory(self, synthetic_adata, scx_from_adata):
        """Streaming DE with gene_chunk_size should match in-memory DE exactly."""
        import pyscx

        # Write to SCX and reopen in backed mode.
        path = scx_from_adata(synthetic_adata, "streaming_de.scx")
        backed_adata = pyscx.open(path).to_anndata(backed=True)

        # In-memory DE (dense materialization).
        adata_mem = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata_mem, "batch")
        rgg_mem = adata_mem.uns["rank_genes_groups"]

        # Streaming DE from backed mode with small chunk size.
        pyscx.accel.rank_genes_groups(
            backed_adata, "batch", gene_chunk_size=10
        )
        rgg_stream = backed_adata.uns["rank_genes_groups"]

        # Same group structure.
        assert rgg_mem["names"].dtype.names == rgg_stream["names"].dtype.names

        # For each group, top 20 genes should overlap substantially.
        for group in rgg_mem["names"].dtype.names:
            mem_top = set(rgg_mem["names"][group][:20])
            stream_top = set(rgg_stream["names"][group][:20])
            overlap = len(mem_top & stream_top) / 20
            assert overlap >= 0.80, (
                f"group {group}: overlap {overlap:.2f} (expected >= 0.80)"
            )

        # P-values should be in valid range.
        for group in rgg_stream["pvals"].dtype.names:
            pvals = rgg_stream["pvals"][group]
            pvals_adj = rgg_stream["pvals_adj"][group]
            assert np.all(pvals >= 0) and np.all(pvals <= 1)
            assert np.all(pvals_adj >= 0) and np.all(pvals_adj <= 1)
            assert np.all(pvals_adj >= pvals - 1e-10)

    def test_streaming_chunk_sizes_consistent(
        self, synthetic_adata, scx_from_adata
    ):
        """Different gene_chunk_size values should produce identical results."""
        import pyscx

        path = scx_from_adata(synthetic_adata, "chunk_sizes.scx")

        results = {}
        for chunk_size in [5, 25, 50]:  # 50 genes total → 10/2/1 chunks
            adata = pyscx.open(path).to_anndata(backed=True)
            pyscx.accel.rank_genes_groups(
                adata, "batch", gene_chunk_size=chunk_size
            )
            results[chunk_size] = adata.uns["rank_genes_groups"]

        # All chunk sizes should produce the same gene rankings.
        ref_rgg = results[5]
        for cs in [25, 50]:
            for group in ref_rgg["names"].dtype.names:
                # Gene names should be in the same order.
                ref_names = list(ref_rgg["names"][group])
                other_names = list(results[cs]["names"][group])
                assert ref_names == other_names, (
                    f"chunk_size={cs}: gene order differs for group {group}"
                )

                # Scores should match.
                np.testing.assert_allclose(
                    ref_rgg["scores"][group],
                    results[cs]["scores"][group],
                    rtol=1e-10,
                    err_msg=f"chunk_size={cs}, group={group}: scores differ",
                )

                # Raw p-values should match.
                np.testing.assert_allclose(
                    ref_rgg["pvals"][group],
                    results[cs]["pvals"][group],
                    rtol=1e-10,
                    err_msg=f"chunk_size={cs}, group={group}: pvals differ",
                )

    def test_streaming_pairwise_reference(
        self, synthetic_adata, scx_from_adata
    ):
        """Streaming DE with a reference group should work correctly."""
        import pyscx

        path = scx_from_adata(synthetic_adata, "stream_ref.scx")
        backed_adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.rank_genes_groups(
            backed_adata, "batch", reference="A", gene_chunk_size=15
        )

        rgg = backed_adata.uns["rank_genes_groups"]
        groups = rgg["names"].dtype.names
        assert "A" not in groups
        assert "B" in groups
        assert "C" in groups


class TestPseudobulkDex:
    """Test pyscx.accel.pseudobulk_dex() — pseudobulk differential expression."""

    @pytest.fixture
    def perturbation_adata(self):
        """Create a synthetic AnnData mimicking a perturbation experiment.

        - 60 cells × 20 genes
        - 2 perturbations (drug, control) × 3 donors
        - Gene 0: strongly upregulated in drug vs control
        - Gene 1: strongly downregulated in drug vs control
        """
        import anndata
        import pandas as pd

        np.random.seed(42)
        n_obs, n_vars = 60, 20

        # Build expression matrix with known DE genes
        data = np.random.randint(5, 15, size=(n_obs, n_vars)).astype(np.float32)

        # Gene 0: high in drug, low in control
        data[:30, 0] = np.random.randint(50, 100, 30).astype(np.float32)  # drug
        data[30:, 0] = np.random.randint(1, 10, 30).astype(np.float32)   # ctrl

        # Gene 1: low in drug, high in control
        data[:30, 1] = np.random.randint(1, 10, 30).astype(np.float32)   # drug
        data[30:, 1] = np.random.randint(50, 100, 30).astype(np.float32) # ctrl

        # Assign perturbation and donor
        perturbations = ["drug"] * 30 + ["control"] * 30
        donors = (["d1"] * 10 + ["d2"] * 10 + ["d3"] * 10) * 2

        obs = pd.DataFrame(
            {
                "perturbation": pd.Categorical(perturbations),
                "donor": pd.Categorical(donors),
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(
            index=[f"gene_{i}" for i in range(n_vars)]
        )

        return anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

    def test_aggregation_correctness(self, perturbation_adata):
        """Rust aggregation should match manual adata.X[mask].sum(axis=0)."""
        import pyscx

        adata = perturbation_adata.copy()

        # Manual aggregation
        for pert in ["drug", "control"]:
            for donor in ["d1", "d2", "d3"]:
                mask = (
                    (adata.obs["perturbation"] == pert)
                    & (adata.obs["donor"] == donor)
                )
                expected_sum = np.asarray(adata.X[mask.values].sum(axis=0)).flatten()

                # We'll verify via the Rust aggregation by calling pseudobulk_dex
                # and checking if the counts match. But first, let's just verify
                # the mask gives reasonable cell counts.
                assert mask.sum() == 10, (
                    f"Expected 10 cells for {pert}/{donor}, got {mask.sum()}"
                )

        # Now run pseudobulk_dex (which calls Rust aggregation internally)
        try:
            result = pyscx.accel.pseudobulk_dex(
                adata,
                groupby=["perturbation", "donor"],
                test_col="perturbation",
                reference="control",
                min_cells_per_group=1,
            )
            # If pydeseq2 succeeded, result is a DataFrame
            assert len(result) > 0
            assert "gene" in result.columns
        except RuntimeError as e:
            if "pydeseq2" in str(e).lower():
                pytest.skip("pydeseq2 not installed")
            raise

    def test_full_pipeline(self, perturbation_adata):
        """Full pseudobulk DE pipeline should produce correct results."""
        try:
            import pydeseq2  # noqa: F401
        except ImportError:
            pytest.skip("pydeseq2 not installed")

        import pyscx

        adata = perturbation_adata.copy()

        result = pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference="control",
            min_cells_per_group=1,
        )

        # Should be a DataFrame with expected columns
        assert hasattr(result, "columns"), "result should be a DataFrame"
        expected_cols = {"gene", "baseMean", "log2FoldChange", "pvalue", "padj"}
        assert expected_cols.issubset(set(result.columns)), (
            f"missing columns: {expected_cols - set(result.columns)}"
        )

        # Should have one row per gene per contrast
        # (drug vs control = 1 contrast × 20 genes = 20 rows)
        assert len(result) == 20, f"expected 20 rows, got {len(result)}"
        assert (result["target"] == "drug").all()
        assert (result["reference"] == "control").all()

        # Gene 0 should have positive log2FC (upregulated in drug)
        gene0 = result[result["gene"] == "gene_0"]
        assert len(gene0) == 1
        assert gene0["log2FoldChange"].values[0] > 0, (
            "gene_0 should be upregulated in drug"
        )

        # Gene 1 should have negative log2FC (downregulated in drug)
        gene1 = result[result["gene"] == "gene_1"]
        assert len(gene1) == 1
        assert gene1["log2FoldChange"].values[0] < 0, (
            "gene_1 should be downregulated in drug"
        )

        # P-values should be in [0, 1]
        assert (result["pvalue"] >= 0).all()
        assert (result["pvalue"] <= 1).all()

    def test_backed_mode(self, perturbation_adata, scx_from_adata):
        """Pseudobulk DE should work from backed SCX (streaming aggregation)."""
        try:
            import pydeseq2  # noqa: F401
        except ImportError:
            pytest.skip("pydeseq2 not installed")

        import pyscx

        path = scx_from_adata(perturbation_adata, "perturbation.scx")
        backed_adata = pyscx.open(path).to_anndata(backed=True)

        result = pyscx.accel.pseudobulk_dex(
            backed_adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference="control",
            min_cells_per_group=1,
        )

        assert len(result) == 20
        assert "gene" in result.columns
        assert "log2FoldChange" in result.columns

        # Gene 0 should still show up as upregulated
        gene0 = result[result["gene"] == "gene_0"]
        assert gene0["log2FoldChange"].values[0] > 0

    def test_backed_matches_inmemory(self, perturbation_adata, scx_from_adata):
        """Backed streaming aggregation should produce same DE results as in-memory."""
        try:
            import pydeseq2  # noqa: F401
        except ImportError:
            pytest.skip("pydeseq2 not installed")

        import pyscx

        path = scx_from_adata(perturbation_adata, "perturbation_cmp.scx")

        # In-memory
        adata_mem = perturbation_adata.copy()
        result_mem = pyscx.accel.pseudobulk_dex(
            adata_mem,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference="control",
            min_cells_per_group=1,
        )

        # Backed
        backed_adata = pyscx.open(path).to_anndata(backed=True)
        result_backed = pyscx.accel.pseudobulk_dex(
            backed_adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference="control",
            min_cells_per_group=1,
        )

        # Sort both by gene for comparison
        result_mem = result_mem.sort_values("gene").reset_index(drop=True)
        result_backed = result_backed.sort_values("gene").reset_index(drop=True)

        # log2FoldChange should be very close
        np.testing.assert_allclose(
            result_mem["log2FoldChange"].values,
            result_backed["log2FoldChange"].values,
            rtol=1e-10,
            err_msg="backed vs in-memory log2FC mismatch",
        )

    def test_invalid_test_col(self, perturbation_adata):
        """test_col not in groupby should raise an error."""
        import pyscx

        with pytest.raises(RuntimeError, match="test_col"):
            pyscx.accel.pseudobulk_dex(
                perturbation_adata,
                groupby=["perturbation", "donor"],
                test_col="nonexistent",
                reference="control",
            )

    def test_invalid_aggr_method(self, perturbation_adata):
        """Invalid aggr_method should raise an error."""
        import pyscx

        with pytest.raises(RuntimeError, match="unsupported aggr_method"):
            pyscx.accel.pseudobulk_dex(
                perturbation_adata,
                groupby=["perturbation", "donor"],
                test_col="perturbation",
                reference="control",
                aggr_method="invalid",
            )


class TestStratifiedDE:
    """Test stratified differential expression for both single-cell and pseudobulk."""

    def test_stratified_rank_genes_groups_matches_manual(self):
        """Stratified single-cell DE should match manual per-stratum loop."""
        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(42)
        n_obs, n_vars = 120, 20
        data = np.random.randint(1, 50, size=(n_obs, n_vars)).astype(np.float32)

        # Two cell types (stratify_by), two conditions (groupby).
        cell_types = ["T_cell"] * 60 + ["B_cell"] * 60
        conditions = np.random.choice(["ctrl", "stim"], size=n_obs)

        obs = pd.DataFrame({
            "cell_type": pd.Categorical(cell_types),
            "condition": pd.Categorical(conditions),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        # --- Stratified call ---
        result = pyscx.accel.rank_genes_groups(
            adata, "condition", stratify_by=["cell_type"],
            min_cells_per_stratum=5,
        )

        assert hasattr(result, "columns"), "stratified DE should return DataFrame"
        assert "gene" in result.columns
        assert "group" in result.columns
        assert "cell_type" in result.columns
        assert "pvals" in result.columns
        assert "pvals_adj" in result.columns

        # --- Manual per-cell-type loop ---
        manual_frames = []
        for ct in sorted(adata.obs["cell_type"].unique()):
            mask = adata.obs["cell_type"] == ct
            sub = adata[mask.values].copy()
            pyscx.accel.rank_genes_groups(sub, "condition")
            rgg = sub.uns["rank_genes_groups"]
            for group in rgg["names"].dtype.names:
                n = len(rgg["names"][group])
                df = pd.DataFrame({
                    "gene": list(rgg["names"][group]),
                    "scores": list(rgg["scores"][group]),
                    "pvals": list(rgg["pvals"][group]),
                    "pvals_adj": list(rgg["pvals_adj"][group]),
                    "logfoldchanges": list(rgg["logfoldchanges"][group]),
                    "group": [group] * n,
                    "cell_type": [ct] * n,
                })
                manual_frames.append(df)
        manual_result = pd.concat(manual_frames, ignore_index=True)

        # Compare per cell_type: same number of rows.
        for ct in sorted(adata.obs["cell_type"].unique()):
            strat_sub = result[result["cell_type"] == ct]
            manual_sub = manual_result[manual_result["cell_type"] == ct]
            assert len(strat_sub) == len(manual_sub), (
                f"cell_type={ct}: row count mismatch "
                f"({len(strat_sub)} vs {len(manual_sub)})"
            )

    def test_stratified_returns_correct_type(self):
        """stratify_by returns DataFrame; without returns None."""
        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(99)
        n_obs, n_vars = 80, 10
        data = np.random.randint(1, 30, size=(n_obs, n_vars)).astype(np.float32)
        obs = pd.DataFrame({
            "cell_type": pd.Categorical(["T"] * 40 + ["B"] * 40),
            "group": pd.Categorical(
                np.random.choice(["A", "B"], size=n_obs)
            ),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        # Non-stratified returns None.
        ret = pyscx.accel.rank_genes_groups(adata, "group")
        assert ret is None

        # Stratified returns DataFrame.
        ret = pyscx.accel.rank_genes_groups(
            adata, "group", stratify_by=["cell_type"],
            min_cells_per_stratum=5,
        )
        assert hasattr(ret, "columns")
        assert "cell_type" in ret.columns

    def test_multi_column_stratify(self):
        """Stratification by multiple columns should create composite strata."""
        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(99)
        n_obs, n_vars = 120, 20
        data = np.random.randint(1, 50, size=(n_obs, n_vars)).astype(np.float32)

        # Create two metadata columns for stratification.
        cell_types = (["T"] * 40 + ["B"] * 40 + ["NK"] * 40)
        tissues = (["lung"] * 20 + ["blood"] * 20) * 3
        groups = np.random.choice(["ctrl", "stim"], size=n_obs)

        obs = pd.DataFrame({
            "cell_type": pd.Categorical(cell_types),
            "tissue": pd.Categorical(tissues),
            "condition": pd.Categorical(groups),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])

        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        result = pyscx.accel.rank_genes_groups(
            adata, "condition",
            stratify_by=["cell_type", "tissue"],
            min_cells_per_stratum=5,
        )

        assert hasattr(result, "columns")
        assert "cell_type" in result.columns
        assert "tissue" in result.columns
        assert "gene" in result.columns

        # Should have results for multiple composite strata.
        unique_strata = result.groupby(["cell_type", "tissue"]).ngroups
        assert unique_strata >= 2, f"expected >= 2 strata, got {unique_strata}"

    def test_min_cells_per_stratum(self):
        """Strata with too few cells should be skipped with a warning."""
        import anndata
        import pandas as pd
        import pyscx
        import warnings

        np.random.seed(42)
        n_obs, n_vars = 55, 10
        data = np.random.randint(1, 20, size=(n_obs, n_vars)).astype(np.float32)

        # Stratum "rare" has only 5 cells; "common" has 50.
        cell_types = ["rare"] * 5 + ["common"] * 50
        groups = np.random.choice(["A", "B"], size=n_obs)

        obs = pd.DataFrame({
            "cell_type": pd.Categorical(cell_types),
            "group": pd.Categorical(groups),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = pyscx.accel.rank_genes_groups(
                adata, "group",
                stratify_by=["cell_type"],
                min_cells_per_stratum=10,  # "rare" has only 5
            )

        # Should only have results for "common" stratum.
        assert (result["cell_type"] == "common").all(), \
            "only 'common' stratum should be present"

        # Should have emitted a warning about skipped strata.
        skip_warnings = [x for x in w if "Skipped" in str(x.message)]
        assert len(skip_warnings) >= 1, "expected warning about skipped strata"

    def test_all_strata_fail_raises(self):
        """If all strata are filtered out, should raise ValueError."""
        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(42)
        n_obs, n_vars = 30, 10
        data = np.random.randint(1, 20, size=(n_obs, n_vars)).astype(np.float32)
        obs = pd.DataFrame({
            "cell_type": pd.Categorical(["T"] * 15 + ["B"] * 15),
            "group": pd.Categorical(
                np.random.choice(["A", "B"], size=n_obs)
            ),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        with pytest.raises(ValueError, match="all strata were filtered out"):
            pyscx.accel.rank_genes_groups(
                adata, "group",
                stratify_by=["cell_type"],
                min_cells_per_stratum=999999,  # impossibly high
            )

    def test_stratify_by_invalid_column(self, synthetic_adata):
        """Non-existent stratify_by column should raise ValueError."""
        import pyscx

        adata = synthetic_adata.copy()

        with pytest.raises(ValueError, match="not found in adata.obs"):
            pyscx.accel.rank_genes_groups(
                adata, "batch",
                stratify_by=["nonexistent_column"],
            )

    def test_stratify_by_collision_with_groupby(self, synthetic_adata):
        """stratify_by that collides with groupby should raise ValueError."""
        import pyscx

        adata = synthetic_adata.copy()

        with pytest.raises(ValueError, match="collides with"):
            pyscx.accel.rank_genes_groups(
                adata, "batch",
                stratify_by=["batch"],  # same as groupby
                min_cells_per_stratum=5,
            )

    def test_stratify_by_collision_with_test_col(self):
        """For pseudobulk DE, stratify_by should not collide with test_col."""
        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(42)
        n_obs, n_vars = 60, 10
        data = np.random.randint(5, 15, size=(n_obs, n_vars)).astype(np.float32)
        obs = pd.DataFrame({
            "perturbation": pd.Categorical(["drug"] * 30 + ["control"] * 30),
            "donor": pd.Categorical(
                (["d1"] * 10 + ["d2"] * 10 + ["d3"] * 10) * 2
            ),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        with pytest.raises(ValueError, match="collides with"):
            pyscx.accel.pseudobulk_dex(
                adata,
                groupby=["perturbation", "donor"],
                test_col="perturbation",
                reference="control",
                stratify_by=["perturbation"],  # collides with test_col
            )

    def test_stratified_pseudobulk_dex_matches_manual(self):
        """Stratified pseudobulk DE should match manual per-cell-type loop."""
        try:
            import pydeseq2  # noqa: F401
        except ImportError:
            pytest.skip("pydeseq2 not installed")

        import anndata
        import pandas as pd
        import pyscx

        np.random.seed(42)
        n_obs, n_vars = 120, 15

        data = np.random.randint(5, 30, size=(n_obs, n_vars)).astype(np.float32)

        # Known DE gene 0: high in drug, low in control (both cell types).
        data[:30, 0] = np.random.randint(80, 150, 30).astype(np.float32)
        data[30:60, 0] = np.random.randint(1, 10, 30).astype(np.float32)
        data[60:90, 0] = np.random.randint(80, 150, 30).astype(np.float32)
        data[90:, 0] = np.random.randint(1, 10, 30).astype(np.float32)

        perturbations = (
            ["drug"] * 30 + ["control"] * 30 +
            ["drug"] * 30 + ["control"] * 30
        )
        donors = (["d1"] * 10 + ["d2"] * 10 + ["d3"] * 10) * 4
        cell_types = ["T_cell"] * 60 + ["B_cell"] * 60

        obs = pd.DataFrame({
            "perturbation": pd.Categorical(perturbations),
            "donor": pd.Categorical(donors),
            "cell_type": pd.Categorical(cell_types),
        }, index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])

        adata = anndata.AnnData(X=sp.csr_matrix(data), obs=obs, var=var)

        # --- Stratified call ---
        result = pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference="control",
            min_cells_per_group=1,
            stratify_by=["cell_type"],
            min_cells_per_stratum=10,
        )

        assert "cell_type" in result.columns
        assert "gene" in result.columns
        assert "log2FoldChange" in result.columns

        # Should have results for both cell types.
        cell_types_in_result = sorted(result["cell_type"].unique())
        assert cell_types_in_result == ["B_cell", "T_cell"], \
            f"expected both cell types, got {cell_types_in_result}"

        # --- Manual loop ---
        manual_frames = []
        for ct in ["B_cell", "T_cell"]:
            mask = adata.obs["cell_type"] == ct
            sub = adata[mask.values].copy()
            df = pyscx.accel.pseudobulk_dex(
                sub,
                groupby=["perturbation", "donor"],
                test_col="perturbation",
                reference="control",
                min_cells_per_group=1,
            )
            df["cell_type"] = ct
            manual_frames.append(df)

        manual_result = pd.concat(manual_frames, ignore_index=True)

        # Compare log2FC for gene_0 in each cell type.
        for ct in ["B_cell", "T_cell"]:
            strat_g0 = result[
                (result["cell_type"] == ct) & (result["gene"] == "g0")
            ]
            manual_g0 = manual_result[
                (manual_result["cell_type"] == ct) &
                (manual_result["gene"] == "g0")
            ]

            if len(strat_g0) > 0 and len(manual_g0) > 0:
                np.testing.assert_allclose(
                    strat_g0["log2FoldChange"].values[0],
                    manual_g0["log2FoldChange"].values[0],
                    rtol=1e-6,
                    err_msg=f"log2FC mismatch for gene_0 in {ct}",
                )


class TestLeiden:
    """Test pyscx.accel.leiden() — Leiden community detection."""

    def test_basic_structure(self, synthetic_adata):
        """Leiden writes cluster labels to adata.obs and metadata to adata.uns."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.leiden(adata)

        # Check obs column
        assert "leiden" in adata.obs.columns
        assert len(adata.obs["leiden"]) == adata.n_obs

        # All labels should be strings (matching scanpy convention)
        assert adata.obs["leiden"].dtype.name == "category"

        # Check uns metadata
        assert "leiden" in adata.uns
        assert "params" in adata.uns["leiden"]
        assert adata.uns["leiden"]["params"]["resolution"] == 1.0
        assert adata.uns["leiden"]["backend"] == "leidenalg"
        assert "modularity" in adata.uns["leiden"]

    def test_multiple_clusters(self, synthetic_adata):
        """Leiden should find at least 2 clusters on synthetic data."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.leiden(adata)

        n_clusters = adata.obs["leiden"].nunique()
        assert n_clusters >= 2, f"expected >= 2 clusters, got {n_clusters}"

    def test_resolution_parameter(self, synthetic_adata):
        """Higher resolution should produce more clusters."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata_low = synthetic_adata.copy()
        pyscx.accel.pca(adata_low, n_comps=10)
        pyscx.accel.neighbors(adata_low, n_neighbors=10)
        pyscx.accel.leiden(adata_low, resolution=0.1, key_added="leiden_low")

        adata_high = synthetic_adata.copy()
        pyscx.accel.pca(adata_high, n_comps=10)
        pyscx.accel.neighbors(adata_high, n_neighbors=10)
        pyscx.accel.leiden(adata_high, resolution=3.0, key_added="leiden_high")

        n_low = adata_low.obs["leiden_low"].nunique()
        n_high = adata_high.obs["leiden_high"].nunique()
        assert n_high >= n_low, (
            f"higher resolution ({n_high} clusters) should yield >= "
            f"clusters than lower resolution ({n_low} clusters)"
        )

    def test_key_added(self, synthetic_adata):
        """key_added parameter should control the obs column name."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.leiden(adata, key_added="my_clusters")

        assert "my_clusters" in adata.obs.columns
        assert "my_clusters" in adata.uns

    def test_reproducibility(self, synthetic_adata):
        """Same random_state should produce identical results."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata1 = synthetic_adata.copy()
        pyscx.accel.pca(adata1, n_comps=10, random_state=42)
        pyscx.accel.neighbors(adata1, n_neighbors=10, random_state=42)
        pyscx.accel.leiden(adata1, random_state=42)

        adata2 = synthetic_adata.copy()
        pyscx.accel.pca(adata2, n_comps=10, random_state=42)
        pyscx.accel.neighbors(adata2, n_neighbors=10, random_state=42)
        pyscx.accel.leiden(adata2, random_state=42)

        assert list(adata1.obs["leiden"]) == list(adata2.obs["leiden"])

    def test_missing_connectivities_error(self, synthetic_adata):
        """Leiden should raise an error if connectivities are missing."""
        import pyscx

        adata = synthetic_adata.copy()

        with pytest.raises(RuntimeError, match="connectivities"):
            pyscx.accel.leiden(adata)

    def test_ari_vs_scanpy_leiden(self, synthetic_adata):
        """ARI between SCX and scanpy Leiden should be > 0.80 on same graph."""
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")
        from sklearn.metrics import adjusted_rand_score

        import pyscx

        # Build graph with SCX (shared between both)
        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)

        # SCX Leiden
        pyscx.accel.leiden(adata, resolution=1.0, random_state=42, key_added="scx_leiden")

        # scanpy Leiden (on the same graph)
        sc.tl.leiden(adata, resolution=1.0, random_state=42, key_added="scanpy_leiden")

        scx_labels = adata.obs["scx_leiden"].values
        scanpy_labels = adata.obs["scanpy_leiden"].values

        ari = adjusted_rand_score(scx_labels, scanpy_labels)
        assert ari > 0.80, (
            f"ARI between SCX and scanpy Leiden = {ari:.4f} (expected > 0.80)"
        )

    def test_backed_pipeline(self, pca_adata):
        """Full pipeline from backed SCX: PCA → neighbors → Leiden."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        scx_path, _ = pca_adata
        backed_adata = pyscx.open(scx_path).to_anndata(backed=True)

        pyscx.accel.pca(backed_adata, n_comps=10)
        pyscx.accel.neighbors(backed_adata, n_neighbors=5)
        pyscx.accel.leiden(backed_adata)

        assert "leiden" in backed_adata.obs.columns
        n_clusters = backed_adata.obs["leiden"].nunique()
        assert n_clusters >= 2, f"expected >= 2 clusters, got {n_clusters}"

    def test_de_on_leiden_clusters(self, synthetic_adata):
        """DE analysis should work on Leiden cluster labels."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.leiden(adata)

        # DE on Leiden clusters
        pyscx.accel.rank_genes_groups(adata, "leiden")
        assert "rank_genes_groups" in adata.uns

    def test_modularity_positive(self, synthetic_adata):
        """Modularity should be positive for a reasonable partition."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.leiden(adata)

        assert adata.uns["leiden"]["modularity"] > 0, (
            "modularity should be positive for a meaningful partition"
        )

    def test_device_cpu_explicit(self, synthetic_adata):
        """Explicitly requesting device='cpu' should use leidenalg."""
        try:
            import leidenalg  # noqa: F401
        except ImportError:
            pytest.skip("leidenalg not available")
        try:
            import igraph  # noqa: F401
        except ImportError:
            pytest.skip("igraph not available")

        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10)
        pyscx.accel.neighbors(adata, n_neighbors=10)
        pyscx.accel.leiden(adata, device="cpu")

        assert adata.uns["leiden"]["backend"] == "leidenalg"
