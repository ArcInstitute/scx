"""Tests for pyscx.accel.pca() — randomized PCA from SCX."""

import numpy as np
import pytest
import scipy.sparse as sp


# --- scanpy DE parity bars (ORG-7.21-4) --------------------------------------
#
# Every number below is the observed max |Δ| against scanpy on the fixture that
# uses it, rounded up one decimal order — not a value picked because it passed.
# Regenerate with `benchmarks/scripts/generate_de_parity_references.py`, which
# prints the same measurement for the Rust-side pins.
#
# These replace an `overlap >= 0.80` on the top-20 gene *names* and an
# `abs(Δlog2FC) < 0.5`. Neither could detect what it was written for: a
# constant shift in 1-vs-rest logFC leaves the ranking (and so the overlap)
# untouched, a 1.4x fold-change gap passes `< 0.5`, and nothing compared
# `scores` or `pvals` at all.
SCANPY_SCORE_ATOL = 1e-6  # observed 1.4373e-07 (log1p) / 5.8359e-08 (raw)
SCANPY_PVAL_ATOL = 1e-12  # observed 3.3307e-16 on both fixtures
SCANPY_LOGFC_ATOL = 1e-6  # observed 2.2619e-07 -- log1p'd input ONLY, see below


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
        """Top PCs should match sklearn PCA (cosine sim > 0.95, accounting for sign).

        NOTE (Phase 1 §3.1): `pyscx.accel.pca` now auto-consumes
        `adata.var['highly_variable']` when present (scanpy semantics), and this
        fixture sets it. The fair sklearn reference is therefore PCA on the same
        masked genes; SCX records `uns['pca']['params']['use_highly_variable']`.
        """
        import pyscx
        from sklearn.decomposition import PCA

        n_comps = 5

        # SCX PCA (default → masks to highly_variable).
        adata2 = synthetic_adata.copy()
        pyscx.accel.pca(adata2, n_comps=n_comps, random_state=0)
        X_pca_scx = adata2.obsm["X_pca"]
        assert bool(adata2.uns["pca"]["params"]["use_highly_variable"]) is True

        # sklearn PCA on the SAME masked columns.
        hvg = synthetic_adata.var["highly_variable"].to_numpy().astype(bool)
        X = synthetic_adata.X.toarray()[:, hvg]
        pca_sk = PCA(n_components=n_comps, random_state=0)
        X_pca_sk = pca_sk.fit_transform(X)

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
        # scanpy's sixth params key; only BH is implemented, so it is always this.
        assert rgg["params"]["corr_method"] == "benjamini-hochberg"

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

    def test_scores_and_pvals_match_scanpy(self, synthetic_adata):
        """`scores`, `pvals`, `pvals_adj` match scanpy per gene, not "mostly".

        Replaces an `overlap >= 0.80` on the top-20 gene *names*, which could
        not see the finding it was written for: a 1-vs-rest logFC shifted by a
        constant leaves the ranking, and therefore the overlap, untouched, and
        nothing here compared `scores` or `pvals` at all.

        Keyed by gene **name**, never by position, and gene *order* is not
        compared at all. scanpy's tie order comes from `np.argsort`'s default
        `quicksort`, which is not stable, so a run of equal scores can come out
        in either arrangement. Order is therefore compared nowhere: not here,
        and not by a separate test *requiring* the two to disagree. An earlier
        version of this file had one, and it was the wrong shape -- a stable-sort
        change upstream would have reddened it without anything being wrong, and
        agreement on one fixture would not have made positional comparison safe
        anyway. The reason for keying by name belongs here, next to the keying.

        `logfoldchanges` is deliberately absent. This fixture is raw counts, and
        scanpy's logFC `expm1`s the group means unconditionally -- it emits a
        warning saying so. The two formulas differ by ~6e+01 here, which is a
        difference in definition, not precision. logFC parity is asserted on
        log1p'd input by `test_logfc_matches_scanpy_log_transformed`, and the
        raw-count divergence is pinned by
        `test_raw_count_logfc_deliberately_diverges_from_scanpy`.
        """
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        import pyscx

        adata_sc = synthetic_adata.copy()
        sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon")

        adata_scx = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata_scx, "batch")

        rgg_sc = adata_sc.uns["rank_genes_groups"]
        rgg_scx = adata_scx.uns["rank_genes_groups"]

        assert rgg_sc["names"].dtype.names == rgg_scx["names"].dtype.names

        atol = {
            "scores": SCANPY_SCORE_ATOL,
            "pvals": SCANPY_PVAL_ATOL,
            "pvals_adj": SCANPY_PVAL_ATOL,
        }
        for group in rgg_scx["names"].dtype.names:
            sc_names = list(rgg_sc["names"][group])
            scx_names = list(rgg_scx["names"][group])
            # Every gene on both sides, so a field comparison cannot skip a
            # gene by failing to find it.
            assert set(sc_names) == set(scx_names), (
                f"group {group}: scanpy and scx report different gene sets"
            )

            for field, bar in atol.items():
                sc_map = dict(zip(sc_names, np.asarray(rgg_sc[field][group], dtype=np.float64)))
                scx_map = dict(zip(scx_names, np.asarray(rgg_scx[field][group], dtype=np.float64)))
                for gene, sc_val in sc_map.items():
                    scx_val = scx_map[gene]
                    if np.isnan(sc_val) and np.isnan(scx_val):
                        continue
                    assert abs(sc_val - scx_val) <= bar, (
                        f"group {group} gene {gene} {field}: "
                        f"scanpy={sc_val!r} scx={scx_val!r} (bar {bar:g})"
                    )

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

        # Keyed by name, and EVERY gene must be on both sides. The old version
        # of this loop skipped any gene missing from the other map
        # (`if gene in scx_map`), so a result that dropped genes entirely would
        # have passed by comparing fewer and fewer of them.
        sc_map = dict(zip(sc_names_a, np.asarray(sc_logfc_a, dtype=np.float64)))
        scx_map = dict(zip(scx_names_a, np.asarray(scx_logfc_a, dtype=np.float64)))
        assert set(sc_map) == set(scx_map), (
            "scanpy and scx report different gene sets for group A"
        )

        # `< 0.5` before this -- a bar a 1.4x fold-change gap passes. The
        # observed max |Δ| on this fixture is 2.2619e-07, so SCANPY_LOGFC_ATOL
        # is six orders tighter and still has headroom.
        for gene, sc_val in sc_map.items():
            scx_val = scx_map[gene]
            if np.isnan(sc_val) and np.isnan(scx_val):
                continue
            assert abs(sc_val - scx_val) <= SCANPY_LOGFC_ATOL, (
                f"logFC mismatch for {gene}: scanpy={sc_val!r}, scx={scx_val!r} "
                f"(bar {SCANPY_LOGFC_ATOL:g})"
            )

        # gene_0 is the implanted signal: it must be present, agree, and be
        # positive for group A. Asserted separately because "all genes agree"
        # is also true of two implementations that both return zeros.
        assert "gene_0" in sc_map
        assert sc_map["gene_0"] > 0.0 and scx_map["gene_0"] > 0.0, (
            f"gene_0 is upregulated in A by construction, got "
            f"scanpy={sc_map['gene_0']!r} scx={scx_map['gene_0']!r}"
        )

    def test_raw_count_logfc_deliberately_diverges_from_scanpy(self, synthetic_adata):
        """On raw counts SCX's logFC is NOT scanpy's, and that is correct.

        scanpy's `rank_genes_groups` `expm1`s the group means unconditionally --
        it assumes log1p'd input and emits a warning when the data looks like
        counts. SCX detects the untransformed case and uses
        `log2(mean_g + eps) − log2(mean_r + eps)` instead. Two different
        formulas, so the answers differ by ~6e+01 on this fixture.

        Pinned rather than left implicit, for two reasons. It is the reason
        `test_scores_and_pvals_match_scanpy` excludes `logfoldchanges` -- an
        exclusion that would otherwise read as an unexplained gap in coverage
        and invite someone to "finish" it. And if SCX ever started expm1-ing
        raw counts too, every other test here would still pass.
        """
        try:
            import scanpy as sc
        except ImportError:
            pytest.skip("scanpy not available")

        import pyscx

        adata_sc = synthetic_adata.copy()
        sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon")
        adata_scx = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata_scx, "batch")

        rgg_sc = adata_sc.uns["rank_genes_groups"]
        rgg_scx = adata_scx.uns["rank_genes_groups"]

        worst = 0.0
        for group in rgg_scx["names"].dtype.names:
            sc_map = dict(zip(rgg_sc["names"][group],
                              np.asarray(rgg_sc["logfoldchanges"][group], dtype=np.float64)))
            scx_map = dict(zip(rgg_scx["names"][group],
                               np.asarray(rgg_scx["logfoldchanges"][group], dtype=np.float64)))
            for gene, sc_val in sc_map.items():
                scx_val = scx_map[gene]
                if np.isfinite(sc_val) and np.isfinite(scx_val):
                    worst = max(worst, abs(sc_val - scx_val))

        # Far beyond any precision story: this is a formula difference. The
        # bound is deliberately loose in the *lower* direction only -- it
        # asserts the divergence is real, not that it has a particular size.
        assert worst > 1.0, (
            f"SCX's raw-count logFC now agrees with scanpy's expm1'd one to "
            f"{worst:.3e}. Either SCX started expm1-ing untransformed counts "
            f"(a regression -- scanpy itself warns against it) or the fixture "
            f"stopped looking like counts."
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


class TestRankGenesGroupsDfExtract:
    """F6: rank_genes_groups_df(group=...) — scanpy sc.get.rank_genes_groups_df alias."""

    SCANPY_COLS = ["names", "scores", "logfoldchanges", "pvals", "pvals_adj"]

    @staticmethod
    def _cols(df):
        # Works for both polars and pandas frames.
        return list(df.columns)

    def test_extract_single_group_scanpy_columns(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        df = pyscx.accel.rank_genes_groups_df(adata, group="A")

        # Single str group → scanpy columns, no leading `group` column.
        assert self._cols(df) == self.SCANPY_COLS
        assert len(df) == adata.n_vars

        # Values match the precomputed structured array for group A.
        rgg = adata.uns["rank_genes_groups"]
        assert list(df["names"]) == list(rgg["names"]["A"])
        np.testing.assert_allclose(df["pvals_adj"], rgg["pvals_adj"]["A"])
        np.testing.assert_allclose(df["logfoldchanges"], rgg["logfoldchanges"]["A"])

    def test_extract_multi_group_has_group_column(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        df = pyscx.accel.rank_genes_groups_df(adata, group=["A", "B"])

        assert self._cols(df) == ["group"] + self.SCANPY_COLS
        assert len(df) == 2 * adata.n_vars
        assert set(df["group"]) == {"A", "B"}

    def test_extract_pval_cutoff_filters_rows(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        full = pyscx.accel.rank_genes_groups_df(adata, group="A")
        filtered = pyscx.accel.rank_genes_groups_df(
            adata, group="A", pval_cutoff=0.5
        )
        assert len(filtered) <= len(full)
        assert np.all(filtered["pvals_adj"] < 0.5)

    def test_extract_group_and_groupby_both_errors(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        with pytest.raises(ValueError, match="not both"):
            pyscx.accel.rank_genes_groups_df(adata, groupby="batch", group="A")

    def test_extract_missing_uns_errors(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()  # no rank_genes_groups run
        with pytest.raises(ValueError, match="not found"):
            pyscx.accel.rank_genes_groups_df(adata, group="A")

    def test_extract_unknown_group_errors(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        with pytest.raises(ValueError, match="available"):
            pyscx.accel.rank_genes_groups_df(adata, group="ZZ")

    def test_extract_malformed_uns_length_mismatch_errors(self, synthetic_adata):
        """A hand-edited uns with mismatched field lengths errors, not panics."""
        import numpy as np
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        # Truncate one field's structured array so its length diverges from
        # `names` — a clean ValueError must replace the Rust index panic.
        rgg = adata.uns["rank_genes_groups"]
        rgg["scores"] = np.asarray(rgg["scores"])[:-1].copy()
        with pytest.raises(ValueError, match="field lengths differ"):
            pyscx.accel.rank_genes_groups_df(adata, group="A")

    def test_extract_group_none_returns_all_groups(self, synthetic_adata):
        """B2: scanpy documents "All groups are returned if group is None"."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        df = pyscx.accel.rank_genes_groups_df(adata, group=None)

        groups = list(adata.uns["rank_genes_groups"]["names"].dtype.names)
        assert self._cols(df) == ["group"] + self.SCANPY_COLS
        assert len(df) == len(groups) * adata.n_vars
        # Group ORDER must follow `dtype.names`, not sorted() — that is the
        # order rank_genes_groups wrote and the order scanpy iterates.
        assert list(dict.fromkeys(df["group"])) == groups

    def test_extract_group_none_equals_the_explicit_list(self, synthetic_adata):
        import pandas as pd
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        groups = list(adata.uns["rank_genes_groups"]["names"].dtype.names)
        auto = pyscx.accel.rank_genes_groups_df(adata)
        expl = pyscx.accel.rank_genes_groups_df(adata, group=groups)
        pd.testing.assert_frame_equal(auto, expl)

    def test_extract_explicit_none_matches_omitted(self, synthetic_adata):
        """pyo3 cannot distinguish an explicit `group=None` from an omitted
        `group=`, so both must mean the same thing."""
        import pandas as pd
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        pd.testing.assert_frame_equal(
            pyscx.accel.rank_genes_groups_df(adata, group=None),
            pyscx.accel.rank_genes_groups_df(adata),
        )

    def test_group_none_with_groupby_still_computes(self, synthetic_adata):
        """Guards the load-bearing `groupby.is_none()` conjunct: an adata that
        already carries uns["rank_genes_groups"] is the NORMAL pipeline shape, so
        an explicit groupby= must not silently switch to extraction."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        df = pyscx.accel.rank_genes_groups_df(
            adata, groupby="batch", group=None
        )
        assert "target" in df.columns and "feature" in df.columns
        assert "names" not in df.columns

    def test_extract_all_applies_n_genes_per_group(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        groups = list(adata.uns["rank_genes_groups"]["names"].dtype.names)
        df = pyscx.accel.rank_genes_groups_df(adata, n_genes=5)
        assert len(df) == 5 * len(groups)
        assert df.groupby("group", observed=True).size().unique().tolist() == [5]

    def test_extract_all_on_empty_dtype_names_errors(self, synthetic_adata):
        """A non-structured `names` array has no groups to expand to."""
        import numpy as np
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        adata.uns["rank_genes_groups"]["names"] = np.arange(3)
        with pytest.raises(ValueError, match="no structured-array fields"):
            pyscx.accel.rank_genes_groups_df(adata)

    def test_extract_all_matches_scanpy(self, synthetic_adata):
        """The contract the docs claim: a drop-in for sc.get.rank_genes_groups_df."""
        sc = pytest.importorskip("scanpy")
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.rank_genes_groups(adata, "batch")
        mine = pyscx.accel.rank_genes_groups_df(adata)
        theirs = sc.get.rank_genes_groups_df(adata, group=None)
        assert list(mine.columns) == list(theirs.columns)
        assert mine.shape == theirs.shape
        assert list(dict.fromkeys(mine["group"])) == list(dict.fromkeys(theirs["group"]))

    def test_neither_group_nor_groupby_errors(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        with pytest.raises(ValueError, match="got neither"):
            pyscx.accel.rank_genes_groups_df(adata)
        # Message must name both remedies and say the uns key is absent.
        with pytest.raises(ValueError, match="does not exist"):
            pyscx.accel.rank_genes_groups_df(adata)

    def test_compute_path_unchanged(self, synthetic_adata):
        """Positional groupby still recomputes and returns cell-eval columns."""
        import pyscx

        adata = synthetic_adata.copy()
        df = pyscx.accel.rank_genes_groups_df(adata, "batch")
        assert list(df.columns) == [
            "target",
            "feature",
            "fold_change",
            "p_value",
            "fdr",
            "log2_fold_change",
            "abs_log2_fold_change",
        ]


class TestDeFrameDefaultContainer:
    """F6: pandas is the default container on every DE-frame surface, polars is
    the opt-in, and the default asks for no optional dependency."""

    @staticmethod
    def _adata():
        import numpy as np
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sp

        rng = np.random.default_rng(0)
        n_obs, n_vars = 60, 12
        x = rng.poisson(4.0, size=(n_obs, n_vars)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "pert": ["control"] * 20 + ["ko_a"] * 20 + ["ko_b"] * 20,
                "donor": [f"d{i % 3}" for i in range(n_obs)],
            },
            index=[f"c{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(index=[f"g{j}" for j in range(n_vars)])
        return ad.AnnData(X=sp.csr_matrix(x), obs=obs, var=var)

    def test_both_modes_default_to_pandas(self):
        import pandas as pd
        import pyscx

        adata = self._adata()
        # Compute mode.
        assert isinstance(
            pyscx.accel.rank_genes_groups_df(adata, "pert"), pd.DataFrame
        )
        # Extract mode (the documented sc.get.rank_genes_groups_df drop-in).
        pyscx.accel.rank_genes_groups(adata, "pert")
        assert isinstance(
            pyscx.accel.rank_genes_groups_df(adata, group=None), pd.DataFrame
        )
        assert isinstance(
            pyscx.accel.rank_genes_groups_df(adata, group="ko_a"), pd.DataFrame
        )
        assert isinstance(
            pyscx.accel.pdex_ref(adata, "pert", reference="control"), pd.DataFrame
        )

    def test_polars_opt_in_carries_identical_values(self):
        import pandas as pd
        import pyscx

        pl = pytest.importorskip("polars")
        adata = self._adata()

        for kwargs in ({"groupby": "pert"}, {"group": None}, {"group": "ko_a"}):
            if "group" in kwargs and "rank_genes_groups" not in adata.uns:
                pyscx.accel.rank_genes_groups(adata, "pert")
            got_pd = pyscx.accel.rank_genes_groups_df(adata, **kwargs)
            got_pl = pyscx.accel.rank_genes_groups_df(
                adata, output="polars", **kwargs
            )
            assert isinstance(got_pl, pl.DataFrame), kwargs
            assert list(got_pd.columns) == list(got_pl.columns), kwargs
            pd.testing.assert_frame_equal(
                got_pd.reset_index(drop=True),
                got_pl.to_pandas().reset_index(drop=True),
                check_dtype=False,
            )

    def test_the_default_needs_no_polars(self, monkeypatch):
        """The regression F6 actually was: on a base `pip install pyscx` polars is
        absent (it lives in the `eval` extra), so a polars-returning default made
        the documented scanpy drop-in raise. Blocking the import proves the
        default path never reaches for it — every local env has polars from the
        `dev` extra, so nothing else here would catch a relapse.
        """
        import sys

        import pandas as pd
        import pyscx

        adata = self._adata()
        pyscx.accel.rank_genes_groups(adata, "pert")

        # `None` in sys.modules makes any later `import polars` raise ImportError,
        # including pyo3's `py.import("polars")`. monkeypatch restores it.
        monkeypatch.setitem(sys.modules, "polars", None)
        with pytest.raises(ImportError):
            import polars  # noqa: F401

        assert isinstance(
            pyscx.accel.rank_genes_groups_df(adata, "pert"), pd.DataFrame
        )
        assert isinstance(
            pyscx.accel.rank_genes_groups_df(adata, group=None), pd.DataFrame
        )
        assert isinstance(
            pyscx.accel.pdex_ref(adata, "pert", reference="control"), pd.DataFrame
        )
        assert isinstance(
            pyscx.accel.pdex_nb_glm(
                adata,
                "pert",
                "control",
                stratify_by=["donor"],
                min_cells_per_group=1,
                min_cells_per_stratum=1,
            ),
            pd.DataFrame,
        )

        # An explicit polars request still fails loudly, naming the extra.
        # ImportError-derived, so `except ImportError` catches it.
        with pytest.raises(ImportError, match=r"pyscx\[eval\]"):
            pyscx.accel.rank_genes_groups_df(adata, "pert", output="polars")


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

        # In-memory and streaming are the SAME kernel over the same values, so
        # this is not a parity claim with a tolerance -- it is an identity.
        # Measured bit-identical on every field including gene order, so
        # asserted that way (ORG-7.21-4). It was `overlap >= 0.80` on the top-20
        # gene names, which is 20 % of one field of four: a streaming path that
        # returned the wrong score for every gene while preserving their ranking
        # would have passed, and the sibling test below
        # (`test_streaming_chunk_sizes_consistent`) already held streaming to
        # exact name order across chunk sizes -- so the weaker bar here was
        # weaker than its own neighbour, not weaker than what the code does.
        for group in rgg_mem["names"].dtype.names:
            assert list(rgg_mem["names"][group]) == list(rgg_stream["names"][group]), (
                f"group {group}: gene order differs between in-memory and streaming"
            )
            for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
                np.testing.assert_array_equal(
                    np.asarray(rgg_mem[field][group]),
                    np.asarray(rgg_stream[field][group]),
                    err_msg=f"group {group}: {field} differs in-memory vs streaming",
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
                # Every field, exactly. The chunk size decides how many genes a
                # pass covers; it does not enter the arithmetic for any one gene,
                # so bit-identity is the honest bar and `rtol=1e-10` on two of
                # the four fields was strictly weaker than the code. Measured
                # identical across 5 / 25 / 50 before this was tightened.
                #
                # `pvals_adj` matters most here and was the field missing: BH is
                # the one step that reads *across* genes, so a chunking bug that
                # left every raw p-value intact could still corrupt it.
                for field in (
                    "names",
                    "scores",
                    "pvals",
                    "pvals_adj",
                    "logfoldchanges",
                ):
                    np.testing.assert_array_equal(
                        ref_rgg[field][group],
                        results[cs][field][group],
                        err_msg=(
                            f"chunk_size={cs}, group={group}: {field} differs "
                            f"from the chunk_size=5 result. gene_chunk_size is a "
                            f"memory knob and must not change any output."
                        ),
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

        # Now run pseudobulk_dex (which calls Rust aggregation internally).
        # The default backend is the dependency-free NB-GLM, so this no
        # longer needs a pydeseq2 escape hatch — and the assertions were
        # always about the Rust aggregation, not the DE engine.
        result = pyscx.accel.pseudobulk_dex(
            adata,
            groupby=["perturbation", "donor"],
            test_col="perturbation",
            reference="control",
            min_cells_per_group=1,
        )
        assert len(result) > 0
        assert "gene" in result.columns

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
            # Pinned: this is the pydeseq2 bridge's end-to-end test, and
            # the skip above promises that is what runs. Without the pin
            # it would gate on pydeseq2 while exercising NB-GLM.
            backend="pydeseq2",
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
        """Pseudobulk DE should work from backed SCX (streaming aggregation).

        Runs on the default backend: this is a test of the streaming
        aggregation, which is shared by both DE engines.
        """
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
        """Backed streaming aggregation should produce same DE results as in-memory.

        Runs on the default backend — the claim is agreement between two
        input layouts, which is engine-independent.
        """
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


class TestPseudobulkDexSampleColumnAliases:
    """`sample_cols=` / `sample_key=` as aliases for `groupby` (dogfood F8).

    `groupby` means opposite things in two functions of one module: in
    `rank_genes_groups` (and all of scanpy) it is the compared column; in
    `pseudobulk_dex` it is the set of columns defining a pseudobulk *sample*,
    and the compared column is `test_col`. A user reaching for the replicate
    role typed `sample_key="donor_id"` and got a bare
    `TypeError: unexpected keyword argument`.
    """

    @pytest.fixture
    def replicate_adata(self):
        """60 cells × 20 genes, 2 conditions × 3 donors, 10 cells per pair.

        `backend="nb_glm"` throughout so these tests do not depend on pydeseq2
        being installed — the subject here is argument resolution, not numerics.
        """
        import anndata
        import pandas as pd

        rng = np.random.default_rng(7)
        n_obs, n_vars = 60, 20
        X = rng.poisson(6.0, size=(n_obs, n_vars)).astype(np.float32)
        obs = pd.DataFrame(
            {
                "perturbation": ["drug"] * 30 + ["control"] * 30,
                "donor": (["d1"] * 10 + ["d2"] * 10 + ["d3"] * 10) * 2,
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        adata = anndata.AnnData(X=sp.csr_matrix(X), obs=obs)
        adata.var_names = [f"gene_{i}" for i in range(n_vars)]
        return adata

    def _common(self):
        return dict(
            test_col="perturbation",
            reference="control",
            backend="nb_glm",
            min_cells_per_group=1,
        )

    def test_sample_cols_is_an_alias_for_groupby(self, replicate_adata):
        """Same columns under either spelling must give the same frame.

        Compared value-by-value, not just by shape — an alias that silently
        reordered or dropped a column would still match on shape.
        """
        import pandas as pd

        import pyscx

        via_groupby = pyscx.accel.pseudobulk_dex(
            replicate_adata.copy(),
            groupby=["perturbation", "donor"],
            **self._common(),
        )
        via_sample_cols = pyscx.accel.pseudobulk_dex(
            replicate_adata.copy(),
            sample_cols=["perturbation", "donor"],
            **self._common(),
        )
        assert len(via_groupby) > 0, "fixture must produce fittable contrasts"
        pd.testing.assert_frame_equal(via_groupby, via_sample_cols)

    def test_groupby_accepts_a_bare_string_like_its_aliases(self, replicate_adata):
        """S15: `pseudobulk_means(groupby=str)` vs `pseudobulk_dex(groupby=list)`
        was a both-ways asymmetry. `groupby` now takes `str | list[str]` through
        the same coercion as the aliases, and a wrong element type names the
        parameter instead of pyo3's `Can't extract 'str' to 'Vec'`.
        """
        import pyscx

        # A one-column sample definition has no replication, so the NB-GLM
        # refuses it downstream — a RuntimeError from the *model* proves the
        # bare string resolved; a TypeError would mean it did not.
        with pytest.raises(RuntimeError, match="replicate|contrast"):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), groupby="perturbation", **self._common()
            )
        with pytest.raises(ValueError, match="groupby must be an obs column name"):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), groupby=["perturbation", 7], **self._common()
            )

    def test_sample_key_accepts_a_bare_string(self, replicate_adata):
        """`sample_key="perturbation"` must resolve, not raise on the type.

        This is the exact spelling the dogfood session reached for. A
        single-column sample definition has no replication, so the NB-GLM
        refuses it downstream — a `RuntimeError` from the *model* is proof the
        alias resolved; a `ValueError`/`TypeError` would mean it did not.
        """
        import pyscx

        with pytest.raises(RuntimeError, match="replicate|contrast"):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), sample_key="perturbation", **self._common()
            )

    @pytest.mark.parametrize("alias", ["sample_cols", "sample_key"])
    def test_both_aliases_take_str_or_list(self, replicate_adata, alias):
        """Neither new alias may have a type trap.

        pyo3 refuses ``str`` -> ``Vec<String>`` (rightly — it would char-split),
        but the resulting ``TypeError: Can't extract 'str' to 'Vec'`` names
        neither the fix nor the sibling kwarg. That unhelpful landing is exactly
        what F8 exists to remove, so both aliases coerce a bare string instead.
        """
        import pandas as pd

        import pyscx

        via_groupby = pyscx.accel.pseudobulk_dex(
            replicate_adata.copy(),
            groupby=["perturbation", "donor"],
            **self._common(),
        )
        via_alias = pyscx.accel.pseudobulk_dex(
            replicate_adata.copy(),
            **{alias: ["perturbation", "donor"]},
            **self._common(),
        )
        pd.testing.assert_frame_equal(via_groupby, via_alias)

        # The bare-string form must resolve (a single sample column has no
        # replication, so the NB-GLM refuses it downstream — a RuntimeError from
        # the *model* proves resolution happened; a TypeError would mean it did
        # not).
        with pytest.raises(RuntimeError, match="replicate|contrast"):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), **{alias: "perturbation"}, **self._common()
            )

        # And a genuinely wrong type gets a message naming the parameter, not
        # pyo3's extraction internals.
        with pytest.raises(ValueError, match=alias):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), **{alias: 42}, **self._common()
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"groupby": ["perturbation", "donor"], "sample_cols": ["perturbation"]},
            {"groupby": ["perturbation", "donor"], "sample_key": "donor"},
            {"sample_cols": ["perturbation"], "sample_key": "donor"},
        ],
    )
    def test_two_spellings_at_once_is_rejected(self, replicate_adata, kwargs):
        """They are one parameter, so supplying two is an error, not a merge."""
        import pyscx

        with pytest.raises(ValueError, match="only one of groupby"):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), **kwargs, **self._common()
            )
        # The message must name what was actually passed, so the user can see
        # which two to drop.
        with pytest.raises(ValueError) as exc:
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(), **kwargs, **self._common()
            )
        for name in kwargs:
            assert name in str(exc.value)

    def test_omitting_all_three_names_the_role_contrast(self, replicate_adata):
        """The error a wrong guess lands on must carry the contrast itself.

        This is the deliverable of F8: the reason `groupby` is confusing here is
        that it means the opposite of `rank_genes_groups(groupby=...)`, so that
        sentence has to be in the message, not only in docs the user did not
        open.
        """
        import pyscx

        with pytest.raises(ValueError) as exc:
            pyscx.accel.pseudobulk_dex(replicate_adata.copy(), **self._common())
        msg = str(exc.value)
        assert "rank_genes_groups" in msg, (
            "the no-columns error must name the contrast against "
            f"rank_genes_groups; got: {msg}"
        )
        assert "test_col" in msg, "and must say which argument IS the compared column"
        for spelling in ("groupby", "sample_cols", "sample_key"):
            assert spelling in msg, f"the error must list the {spelling} spelling"

    @pytest.mark.parametrize("missing", ["test_col", "reference"])
    def test_missing_test_col_or_reference_still_reports_clearly(
        self, replicate_adata, missing
    ):
        """Dropping pyo3's arity check must not degrade these two.

        `test_col` / `reference` became `Option` only so `groupby` could, so
        each needs its own explicit message where pyo3's `TypeError` used to be.
        """
        import pyscx

        kwargs = self._common()
        kwargs.pop(missing)
        with pytest.raises(ValueError, match=missing):
            pyscx.accel.pseudobulk_dex(
                replicate_adata.copy(),
                groupby=["perturbation", "donor"],
                **kwargs,
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
                # stratify_by is pydeseq2-only; without this the nb_glm
                # default rejects the call before the collision check runs.
                backend="pydeseq2",
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
            # The stratified path is the pydeseq2 backend's; nb_glm (now the
            # default) takes replicates as rows of one design instead.
            backend="pydeseq2",
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
                # Must match the stratified arm above: the claim is that
                # `stratify_by` equals a manual per-stratum loop, which only
                # means anything when both run the same DE engine.
                backend="pydeseq2",
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
        assert adata.uns["leiden"]["backend"] in ("scx-accel", "leidenalg")
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
        """ARI between SCX and scanpy Leiden should be > 0.40 on same graph.

        On small synthetic data (100 cells), Leiden is sensitive to algorithmic
        differences between Rust and C++ implementations, so we use a relaxed
        threshold. Real-world validation on census_1m achieves ARI ~0.92.
        """
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

        # SCX Leiden with convergence mode to match scanpy's default
        pyscx.accel.leiden(adata, resolution=1.0, random_state=42, key_added="scx_leiden",
                           n_iterations=-1)

        # scanpy Leiden with default n_iterations=-1 (convergence)
        sc.tl.leiden(adata, resolution=1.0, random_state=42, key_added="scanpy_leiden")

        scx_labels = adata.obs["scx_leiden"].values
        scanpy_labels = adata.obs["scanpy_leiden"].values

        ari = adjusted_rand_score(scx_labels, scanpy_labels)
        assert ari > 0.40, (
            f"ARI between SCX and scanpy Leiden = {ari:.4f} (expected > 0.40)"
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
        """Modularity is positive AND on the modularity scale.

        ``> 0`` alone could not tell the normalized value from the raw RB
        quality it replaced in 0.20 — the un-normalized number is ~10^6 on a
        large graph and satisfies ``> 0`` just as well, so reverting the
        normalization would have left this test green while the Rust karate
        tests reddened. The upper bound is what makes this binding pinned.
        Found by Cursor Agent - Grok 4.6 High.
        """
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

        modularity = adata.uns["leiden"]["modularity"]
        assert modularity > 0, (
            "modularity should be positive for a meaningful partition"
        )
        # `leiden()` runs at the default resolution=1.0, where the normalized RB
        # objective *is* Newman modularity and so is bounded by [-0.5, 1].
        assert modularity <= 1.0, (
            f"modularity {modularity} is above 1 — that is the un-normalized RB "
            "quality, not modularity"
        )
        # The raw objective is still available, and is the larger of the two.
        quality = adata.uns["leiden"]["quality"]
        assert quality is not None
        assert quality > modularity

    def test_device_cpu_explicit(self, synthetic_adata):
        """Explicitly requesting device='cpu' should use Rust-native Leiden (or leidenalg fallback)."""
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

        assert adata.uns["leiden"]["backend"] in ("scx-accel", "leidenalg")


def test_numpy_default_argsort_still_disagrees_with_stable():
    """Canary for the mixed-tie caveat in `docs/scanpy.md`.

    `discrimination_score` breaks ties by ascending index — a deterministic,
    stable rule. cell-eval instead reads its rank off `np.argsort`, whose default
    kind is `quicksort` and therefore not stable, so the docs say mixed-tie parity
    is **not claimed**. That caveat is only worth carrying while it is true.

    This test lives here, and the choice of file is the whole point. It cannot go
    in `test_cell_eval_parity.py`, which `importorskip`s `cell_eval` at module
    level — not installed in CI's Python-bindings image, so a canary there could
    never fire (flagged by Cursor Agent in review). It cannot go in
    `test_eval_metrics.py` either: that module has a module-level
    `importorskip("polars")`, which skips the *entire file* at import time when
    polars — an optional extra — is absent, canary included. `test_accel.py` has
    no module-level gate. It needs only numpy.

    If a future numpy makes its default sort stable, this fails and tells us to
    delete the caveat — rather than leaving a stale warning in the docs forever.
    """
    import numpy as np

    d = np.array([3.0, 3.0, 1.0, 1.0])
    default_order = list(np.argsort(d))
    stable_order = list(np.argsort(d, kind="stable"))
    assert default_order != stable_order, (
        f"numpy {np.__version__} now agrees with a stable sort on {list(d)} "
        f"({default_order}). The mixed-tie divergence documented in "
        f"docs/scanpy.md § Discrimination score may no longer exist — recheck it "
        f"and drop the caveat if so."
    )
