"""Integration tests: SCX ↔ h5ad equivalence with scanpy.

These tests verify that data processed through the SCX format produces
identical results to the same operations on the original AnnData/h5ad data.
Each test follows the pattern:
  1. Create reference AnnData
  2. Run a scanpy operation on the original (the "h5ad path")
  3. Round-trip through SCX, then run the same operation (the "SCX path")
  4. Assert results are equivalent within tolerance
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def equivalence_adata():
    """Create synthetic AnnData suitable for full scanpy pipeline equivalence.

    - 500 cells × 200 genes (enough for meaningful HVG / PCA / clustering)
    - Sparse integer UMI counts, geometric-like distribution
    - obs: batch (categorical), cell_type (categorical), donor (str)
    - var: gene_id (str), mt (bool — mitochondrial gene flag)
    """
    import anndata

    rng = np.random.RandomState(2024)
    n_obs, n_vars = 500, 200

    # Geometric-like UMI counts (realistic scRNA-seq distribution)
    dense = rng.geometric(p=0.3, size=(n_obs, n_vars)).astype(np.float32)
    # Sparsify ~60% zeros
    mask = rng.random((n_obs, n_vars)) > 0.4
    dense[mask] = 0
    x = sp.csr_matrix(dense)

    # obs metadata
    batches = rng.choice(["batch_A", "batch_B", "batch_C"], size=n_obs)
    cell_types = rng.choice(["T cell", "B cell", "NK cell", "Monocyte"], size=n_obs)
    donors = [f"donor_{i % 10}" for i in range(n_obs)]
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(batches),
            "cell_type": pd.Categorical(cell_types),
            "donor": donors,
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    # var metadata — mark first 10 genes as "mitochondrial"
    gene_names = [f"MT-gene_{i}" if i < 10 else f"gene_{i}" for i in range(n_vars)]
    var = pd.DataFrame(
        {
            "gene_id": gene_names,
            "mt": [i < 10 for i in range(n_vars)],
        },
        index=gene_names,
    )

    return anndata.AnnData(X=x, obs=obs, var=var)


@pytest.fixture
def scx_roundtrip(tmp_path):
    """Helper: AnnData → SCX → AnnData round-trip."""
    import pyscx

    def _roundtrip(adata, name="equiv.scx", **kwargs):
        path = str(tmp_path / name)
        pyscx.from_anndata(adata, path, **kwargs)
        return pyscx.open(path).to_anndata()

    return _roundtrip


# ---------------------------------------------------------------------------
# Scanpy Equivalence Tests
# ---------------------------------------------------------------------------


class TestQCMetricEquivalence:
    """Gap 2: QC metrics must match between h5ad and SCX paths."""

    def test_qc_metrics_equivalence(self, equivalence_adata, scx_roundtrip):
        """sc.pp.calculate_qc_metrics() produces identical results."""
        import scanpy as sc

        # h5ad path
        adata_ref = equivalence_adata.copy()
        sc.pp.calculate_qc_metrics(
            adata_ref, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        # SCX path
        adata_scx = scx_roundtrip(equivalence_adata)
        # Ensure the 'mt' column survives round-trip (may be stored as string)
        if "mt" in adata_scx.var.columns:
            adata_scx.var["mt"] = adata_scx.var["mt"].astype(bool)
        sc.pp.calculate_qc_metrics(
            adata_scx, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
        )

        # Compare obs QC columns
        for col in ["n_genes_by_counts", "total_counts"]:
            np.testing.assert_allclose(
                adata_ref.obs[col].values,
                adata_scx.obs[col].values,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"obs QC column '{col}' mismatch",
            )

        if "pct_counts_mt" in adata_ref.obs.columns:
            np.testing.assert_allclose(
                adata_ref.obs["pct_counts_mt"].values,
                adata_scx.obs["pct_counts_mt"].values,
                rtol=1e-5,
                atol=1e-6,
                err_msg="pct_counts_mt mismatch",
            )

        # Compare var QC columns
        for col in ["n_cells_by_counts", "mean_counts", "total_counts"]:
            np.testing.assert_allclose(
                adata_ref.var[col].values,
                adata_scx.var[col].values,
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"var QC column '{col}' mismatch",
            )


class TestNormalizationEquivalence:
    """Gap 1 (partial): Normalization + log1p equivalence."""

    def test_normalize_log1p_equivalence(self, equivalence_adata, scx_roundtrip):
        """sc.pp.normalize_total() + sc.pp.log1p() produce identical X."""
        import scanpy as sc

        # h5ad path
        adata_ref = equivalence_adata.copy()
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)

        # SCX path
        adata_scx = scx_roundtrip(equivalence_adata)
        sc.pp.normalize_total(adata_scx, target_sum=1e4)
        sc.pp.log1p(adata_scx)

        # Compare X
        X_ref = adata_ref.X.toarray() if sp.issparse(adata_ref.X) else adata_ref.X
        X_scx = adata_scx.X.toarray() if sp.issparse(adata_scx.X) else adata_scx.X
        np.testing.assert_allclose(
            X_ref, X_scx, rtol=1e-5, atol=1e-6, err_msg="normalize+log1p X mismatch"
        )


class TestHVGEquivalence:
    """Gap 3: HVG selection must identify the same genes."""

    def test_hvg_selection_equivalence(self, equivalence_adata, scx_roundtrip):
        """sc.pp.highly_variable_genes() selects the same genes."""
        import scanpy as sc

        # Preprocess both identically
        adata_ref = equivalence_adata.copy()
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        sc.pp.highly_variable_genes(adata_ref, min_mean=0.0125, max_mean=3, min_disp=0.5)

        adata_scx = scx_roundtrip(equivalence_adata)
        sc.pp.normalize_total(adata_scx, target_sum=1e4)
        sc.pp.log1p(adata_scx)
        sc.pp.highly_variable_genes(adata_scx, min_mean=0.0125, max_mean=3, min_disp=0.5)

        # Same set of HVGs
        hvg_ref = set(adata_ref.var_names[adata_ref.var["highly_variable"]])
        hvg_scx = set(adata_scx.var_names[adata_scx.var["highly_variable"]])
        assert hvg_ref == hvg_scx, (
            f"HVG sets differ: {len(hvg_ref)} ref vs {len(hvg_scx)} scx, "
            f"diff={hvg_ref.symmetric_difference(hvg_scx)}"
        )

        # Same dispersion values
        np.testing.assert_allclose(
            adata_ref.var["dispersions"].values,
            adata_scx.var["dispersions"].values,
            rtol=1e-5,
            atol=1e-6,
            err_msg="dispersions mismatch",
        )


class TestScaleEquivalence:
    """Gap 4: sc.pp.scale() equivalence."""

    def test_scale_equivalence(self, equivalence_adata, scx_roundtrip):
        """sc.pp.scale() produces identical dense X arrays."""
        import scanpy as sc

        # Full preprocess up to scale
        adata_ref = equivalence_adata.copy()
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        sc.pp.scale(adata_ref, max_value=10)

        adata_scx = scx_roundtrip(equivalence_adata)
        sc.pp.normalize_total(adata_scx, target_sum=1e4)
        sc.pp.log1p(adata_scx)
        sc.pp.scale(adata_scx, max_value=10)

        # After scale, X is dense
        X_ref = adata_ref.X if isinstance(adata_ref.X, np.ndarray) else adata_ref.X.toarray()
        X_scx = adata_scx.X if isinstance(adata_scx.X, np.ndarray) else adata_scx.X.toarray()
        np.testing.assert_allclose(
            X_ref, X_scx, rtol=1e-5, atol=1e-6, err_msg="scale X mismatch"
        )


class TestPCAEquivalence:
    """Gap 1: PCA embeddings equivalence."""

    def test_pca_equivalence(self, equivalence_adata, scx_roundtrip):
        """PCA embeddings match between h5ad and SCX paths."""
        import scanpy as sc

        n_comps = 20

        adata_ref = equivalence_adata.copy()
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        sc.pp.scale(adata_ref, max_value=10)
        sc.tl.pca(adata_ref, n_comps=n_comps, svd_solver="arpack", random_state=42)

        adata_scx = scx_roundtrip(equivalence_adata)
        sc.pp.normalize_total(adata_scx, target_sum=1e4)
        sc.pp.log1p(adata_scx)
        sc.pp.scale(adata_scx, max_value=10)
        sc.tl.pca(adata_scx, n_comps=n_comps, svd_solver="arpack", random_state=42)

        # PCA is deterministic with fixed solver+seed; compare absolute values
        # because eigenvector sign can flip
        np.testing.assert_allclose(
            np.abs(adata_ref.obsm["X_pca"]),
            np.abs(adata_scx.obsm["X_pca"]),
            rtol=1e-4,
            atol=1e-5,
            err_msg="PCA embeddings mismatch",
        )

        # Variance ratios should match
        np.testing.assert_allclose(
            adata_ref.uns["pca"]["variance_ratio"],
            adata_scx.uns["pca"]["variance_ratio"],
            rtol=1e-5,
            atol=1e-6,
            err_msg="PCA variance ratios mismatch",
        )


class TestClusteringEquivalence:
    """Gap 1: Neighbors + Leiden clustering equivalence."""

    def test_neighbors_leiden_equivalence(self, equivalence_adata, scx_roundtrip):
        """Neighbor graph + Leiden clustering produce same clusters."""
        import scanpy as sc

        def _preprocess_and_cluster(adata):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, n_comps=20, svd_solver="arpack", random_state=42)
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=42)
            sc.tl.leiden(
                adata,
                flavor="igraph",
                n_iterations=2,
                directed=False,
                random_state=42,
            )
            return adata

        adata_ref = _preprocess_and_cluster(equivalence_adata.copy())
        adata_scx = _preprocess_and_cluster(scx_roundtrip(equivalence_adata))

        # Leiden assignments should be identical
        assert list(adata_ref.obs["leiden"]) == list(adata_scx.obs["leiden"]), (
            "Leiden cluster assignments differ"
        )


class TestDEEquivalence:
    """Gap 1: Differential expression equivalence."""

    def test_de_equivalence(self, equivalence_adata, scx_roundtrip):
        """sc.tl.rank_genes_groups() produces equivalent results."""
        import scanpy as sc

        def _preprocess_and_de(adata):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, n_comps=20, svd_solver="arpack", random_state=42)
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=42)
            sc.tl.leiden(
                adata, flavor="igraph", n_iterations=2, directed=False, random_state=42
            )
            sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
            return adata

        adata_ref = _preprocess_and_de(equivalence_adata.copy())
        adata_scx = _preprocess_and_de(scx_roundtrip(equivalence_adata))

        ref_rgg = adata_ref.uns["rank_genes_groups"]
        scx_rgg = adata_scx.uns["rank_genes_groups"]

        # Same group names
        assert list(ref_rgg["names"].dtype.names) == list(scx_rgg["names"].dtype.names)

        # Same gene rankings per group
        for group in ref_rgg["names"].dtype.names:
            assert list(ref_rgg["names"][group]) == list(scx_rgg["names"][group]), (
                f"DE gene names differ for group '{group}'"
            )

            # p-values should be close
            np.testing.assert_allclose(
                ref_rgg["pvals"][group].astype(np.float64),
                scx_rgg["pvals"][group].astype(np.float64),
                rtol=1e-5,
                atol=1e-10,
                err_msg=f"DE p-values differ for group '{group}'",
            )


class TestUMAPEquivalence:
    """Gap 11: UMAP embedding equivalence."""

    def test_umap_equivalence(self, equivalence_adata, scx_roundtrip):
        """UMAP produces equivalent embeddings with fixed random_state."""
        import scanpy as sc

        def _preprocess_and_umap(adata):
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            sc.pp.scale(adata, max_value=10)
            sc.tl.pca(adata, n_comps=20, svd_solver="arpack", random_state=42)
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=42)
            sc.tl.umap(adata, random_state=42)
            return adata

        adata_ref = _preprocess_and_umap(equivalence_adata.copy())
        adata_scx = _preprocess_and_umap(scx_roundtrip(equivalence_adata))

        np.testing.assert_allclose(
            adata_ref.obsm["X_umap"],
            adata_scx.obsm["X_umap"],
            rtol=1e-4,
            atol=1e-4,
            err_msg="UMAP embeddings mismatch",
        )


class TestFullPipelineEquivalence:
    """Gap 1 (comprehensive): Full scanpy tutorial pipeline equivalence."""

    def test_full_scanpy_pipeline_equivalence(self, equivalence_adata, scx_roundtrip):
        """Complete scanpy workflow: filter → norm → HVG → scale → PCA → neighbors → leiden → UMAP → DE.

        Compares every intermediate and final result between h5ad and SCX paths.
        """
        import scanpy as sc

        def _full_pipeline(adata):
            """Run standard scanpy pipeline, returning intermediate results."""
            results = {}

            sc.pp.filter_cells(adata, min_genes=5)
            sc.pp.filter_genes(adata, min_cells=3)
            results["post_filter_shape"] = adata.shape

            # Store raw counts before normalization for seurat_v3 HVG
            adata.raw = adata.copy()

            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

            # Use seurat_v3 flavor with n_top_genes — robust on synthetic data
            sc.pp.highly_variable_genes(
                adata, n_top_genes=50, flavor="seurat_v3", layer=None,
            )
            results["n_hvg"] = int(adata.var["highly_variable"].sum())
            results["hvg_set"] = set(adata.var_names[adata.var["highly_variable"]])

            adata = adata[:, adata.var.highly_variable].copy()
            results["post_hvg_shape"] = adata.shape

            sc.pp.scale(adata, max_value=10)
            n_comps = min(20, adata.shape[1] - 1)
            sc.tl.pca(adata, n_comps=n_comps, svd_solver="arpack", random_state=42)
            results["pca"] = adata.obsm["X_pca"].copy()
            results["variance_ratio"] = adata.uns["pca"]["variance_ratio"].copy()

            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=20, random_state=42)
            sc.tl.leiden(
                adata, flavor="igraph", n_iterations=2, directed=False, random_state=42
            )
            results["leiden"] = list(adata.obs["leiden"])

            sc.tl.umap(adata, random_state=42)
            results["umap"] = adata.obsm["X_umap"].copy()

            sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
            results["de_names"] = {
                g: list(adata.uns["rank_genes_groups"]["names"][g])
                for g in adata.uns["rank_genes_groups"]["names"].dtype.names
            }

            return results

        ref = _full_pipeline(equivalence_adata.copy())
        scx = _full_pipeline(scx_roundtrip(equivalence_adata))

        # Shape checks
        assert ref["post_filter_shape"] == scx["post_filter_shape"]
        assert ref["post_hvg_shape"] == scx["post_hvg_shape"]

        # HVG equivalence
        assert ref["n_hvg"] == scx["n_hvg"]
        assert ref["hvg_set"] == scx["hvg_set"]

        # PCA equivalence (abs for sign flip tolerance)
        np.testing.assert_allclose(
            np.abs(ref["pca"]), np.abs(scx["pca"]), rtol=1e-4, atol=1e-5
        )
        np.testing.assert_allclose(
            ref["variance_ratio"], scx["variance_ratio"], rtol=1e-5, atol=1e-6
        )

        # Leiden equivalence
        assert ref["leiden"] == scx["leiden"]

        # UMAP equivalence
        np.testing.assert_allclose(
            ref["umap"], scx["umap"], rtol=1e-4, atol=1e-4
        )

        # DE equivalence
        assert ref["de_names"] == scx["de_names"]


# ---------------------------------------------------------------------------
# Edge Case Round-Trip Tests
# ---------------------------------------------------------------------------


class TestEdgeCaseRoundTrips:
    """Gaps 5–9: Edge cases for data format handling."""

    def test_float_value_roundtrip(self, tmp_path):
        """Gap 5: True float32 values (log-normalized) survive round-trip."""
        import anndata
        import pyscx

        rng = np.random.RandomState(99)
        n_obs, n_vars = 100, 50

        # Create pre-normalized float data (not integer-valued)
        dense = rng.exponential(scale=2.0, size=(n_obs, n_vars)).astype(np.float32)
        mask = rng.random((n_obs, n_vars)) > 0.4
        dense[mask] = 0.0
        # Apply log1p to create realistic float values
        dense = np.log1p(dense)
        x = sp.csr_matrix(dense)

        adata = anndata.AnnData(X=x)
        path = str(tmp_path / "float.scx")
        pyscx.from_anndata(adata, path)
        adata_rt = pyscx.open(path).to_anndata()

        np.testing.assert_allclose(
            adata.X.toarray(),
            adata_rt.X.toarray(),
            rtol=1e-5,
            atol=1e-6,
            err_msg="Float values changed after round-trip",
        )

    def test_csc_input_roundtrip(self, tmp_path):
        """Gap 6: CSC matrix X is correctly converted to CSR during write."""
        import anndata
        import pyscx

        rng = np.random.RandomState(77)
        n_obs, n_vars = 80, 40

        dense = rng.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
        mask = rng.random((n_obs, n_vars)) > 0.3
        dense[mask] = 0
        x_csc = sp.csc_matrix(dense)

        adata = anndata.AnnData(X=x_csc)
        assert adata.X.format == "csc", "fixture should be CSC"

        path = str(tmp_path / "csc.scx")
        pyscx.from_anndata(adata, path)
        adata_rt = pyscx.open(path).to_anndata()

        np.testing.assert_array_equal(
            dense, adata_rt.X.toarray(), err_msg="CSC input data corrupted"
        )

    def test_dense_input_roundtrip(self, tmp_path):
        """Gap 7: Dense numpy array X is correctly handled."""
        import anndata
        import pyscx

        rng = np.random.RandomState(55)
        n_obs, n_vars = 60, 30

        dense = rng.randint(0, 50, size=(n_obs, n_vars)).astype(np.float32)
        mask = rng.random((n_obs, n_vars)) > 0.3
        dense[mask] = 0

        adata = anndata.AnnData(X=dense)  # Dense!
        assert not sp.issparse(adata.X), "fixture should be dense"

        path = str(tmp_path / "dense.scx")
        pyscx.from_anndata(adata, path)
        adata_rt = pyscx.open(path).to_anndata()

        np.testing.assert_array_equal(
            dense, adata_rt.X.toarray(), err_msg="Dense input data corrupted"
        )

    def test_multi_shard_roundtrip(self, tmp_path):
        """Gap 8: Data split across multiple shards reassembles correctly."""
        import anndata
        import pyscx

        rng = np.random.RandomState(33)
        n_obs, n_vars = 300, 50

        dense = rng.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
        mask = rng.random((n_obs, n_vars)) > 0.3
        dense[mask] = 0
        x = sp.csr_matrix(dense)

        obs = pd.DataFrame(
            {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        adata = anndata.AnnData(X=x, obs=obs)

        path = str(tmp_path / "multishard.scx")
        # Force small shards: 50 rows each → 6 shards for 300 cells
        pyscx.from_anndata(adata, path, shard_size=50)
        exp = pyscx.open(path)
        assert exp.shard_count >= 6, f"expected >=6 shards, got {exp.shard_count}"

        adata_rt = exp.to_anndata()

        # Per-cell value comparison
        np.testing.assert_array_equal(
            adata.X.toarray(),
            adata_rt.X.toarray(),
            err_msg="Multi-shard data corrupted during reassembly",
        )

        # obs must be complete
        assert list(adata_rt.obs["cell_id"]) == list(adata.obs["cell_id"])

    def test_all_zero_row_roundtrip(self, tmp_path):
        """Gap 9: Cells with zero total counts survive round-trip."""
        import anndata
        import pyscx

        rng = np.random.RandomState(22)
        n_obs, n_vars = 50, 20

        dense = rng.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
        mask = rng.random((n_obs, n_vars)) > 0.3
        dense[mask] = 0
        # Force some all-zero rows
        dense[0, :] = 0
        dense[10, :] = 0
        dense[49, :] = 0
        x = sp.csr_matrix(dense)

        adata = anndata.AnnData(X=x)
        path = str(tmp_path / "zeros.scx")
        pyscx.from_anndata(adata, path)
        adata_rt = pyscx.open(path).to_anndata()

        np.testing.assert_array_equal(
            adata.X.toarray(),
            adata_rt.X.toarray(),
            err_msg="All-zero row data corrupted",
        )
        # Verify those specific rows are all zeros
        rt_dense = adata_rt.X.toarray()
        assert np.all(rt_dense[0] == 0)
        assert np.all(rt_dense[10] == 0)
        assert np.all(rt_dense[49] == 0)


# ---------------------------------------------------------------------------
# Post-Ops Data Integrity
# ---------------------------------------------------------------------------


class TestOpsDataIntegrity:
    """Gap 10: Verify actual matrix values (not just counts) after ops."""

    def test_append_compact_data_integrity(self, tmp_path):
        """After append + compact, surviving cell values are bit-exact."""
        import anndata
        import pyscx

        rng = np.random.RandomState(11)
        n_obs, n_vars = 60, 30

        # Base dataset
        dense1 = rng.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
        mask1 = rng.random((n_obs, n_vars)) > 0.3
        dense1[mask1] = 0
        x1 = sp.csr_matrix(dense1)
        obs1 = pd.DataFrame(
            {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(
            {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
            index=[f"gene_{i}" for i in range(n_vars)],
        )
        adata1 = anndata.AnnData(X=x1, obs=obs1, var=var)

        # Extra dataset to append
        dense2 = rng.randint(0, 100, size=(30, n_vars)).astype(np.float32)
        mask2 = rng.random((30, n_vars)) > 0.3
        dense2[mask2] = 0
        x2 = sp.csr_matrix(dense2)
        obs2 = pd.DataFrame(
            {"cell_id": [f"extra_{i}" for i in range(30)]},
            index=[f"extra_{i}" for i in range(30)],
        )
        adata2 = anndata.AnnData(X=x2, obs=obs2, var=var)

        # Write base + append
        base_path = str(tmp_path / "base.scx")
        extra_path = str(tmp_path / "extra.scx")
        pyscx.from_anndata(adata1, base_path)
        pyscx.from_anndata(adata2, extra_path)
        pyscx.append(base_path, extra_path)

        # Delete some cells (first 5)
        pyscx.mark_deleted(base_path, [0, 1, 2, 3, 4])

        # Read pre-compact data
        pre_compact_adata = pyscx.open(base_path).to_anndata()
        pre_compact_X = pre_compact_adata.X.toarray()

        # Compact
        compacted_path = str(tmp_path / "compacted.scx")
        pyscx.compact(base_path, compacted_path)
        post_compact_adata = pyscx.open(compacted_path).to_anndata()
        post_compact_X = post_compact_adata.X.toarray()

        # Values must be identical
        np.testing.assert_array_equal(
            pre_compact_X,
            post_compact_X,
            err_msg="Matrix values changed after compact",
        )

        # Verify deleted cells are excluded: should have 60 + 30 - 5 = 85 cells
        assert post_compact_adata.n_obs == 85

        # Verify surviving cell values from base match originals
        # Cells 5-59 from base should match dense1[5:60]
        base_surviving = dense1[5:]  # rows 5-59
        for i in range(55):
            np.testing.assert_array_equal(
                post_compact_X[i],
                base_surviving[i],
                err_msg=f"Base cell {i + 5} values corrupted after compact",
            )

        # Cells from extra dataset should match dense2
        for i in range(30):
            np.testing.assert_array_equal(
                post_compact_X[55 + i],
                dense2[i],
                err_msg=f"Appended cell {i} values corrupted after compact",
            )
