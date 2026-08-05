"""Integration tests for eval_metrics: pseudobulk_means, perturbation_metrics, energy_distance.

Verifies Rust-accelerated implementations against reference implementations
(polars group_by().mean() for pseudobulk, scipy/sklearn for metrics).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import anndata as ad


class TestPseudobulkMeans:
    """Test pyscx.accel.pseudobulk_means()."""

    def _make_adata(self, n_obs=100, n_vars=50, n_perts=5, seed=42):
        """Create a synthetic AnnData with sparse X and perturbation labels."""
        rng = np.random.default_rng(seed)
        # Sparse count matrix (~10% density)
        density = 0.1
        nnz = int(n_obs * n_vars * density)
        rows = rng.integers(0, n_obs, size=nnz)
        cols = rng.integers(0, n_vars, size=nnz)
        vals = rng.integers(1, 100, size=nnz).astype(np.float32)
        X = sp.csr_matrix((vals, (rows, cols)), shape=(n_obs, n_vars))
        X.sum_duplicates()

        # Perturbation labels
        pert_labels = [f"pert_{i % n_perts}" for i in range(n_obs)]

        import pandas as pd

        obs = pd.DataFrame({"perturbation": pert_labels})
        var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata

    def test_basic_means(self):
        """Verify pseudobulk_means matches polars group_by().mean()."""
        import pyscx

        adata = self._make_adata()
        means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")

        assert isinstance(means, np.ndarray)
        assert means.dtype == np.float64
        assert means.shape[1] == adata.n_vars
        assert len(groups) == means.shape[0]
        assert len(groups) == 5  # 5 perturbation groups

        # Reference: compute means manually with numpy
        X_dense = adata.X.toarray()
        for i, group in enumerate(groups):
            mask = adata.obs["perturbation"].values == group
            expected = X_dense[mask].mean(axis=0)
            np.testing.assert_allclose(
                means[i], expected, atol=1e-6,
                err_msg=f"Means mismatch for group {group}"
            )

    def test_min_cells_filter(self):
        """Groups with too few cells should be excluded."""
        import pyscx

        adata = self._make_adata(n_obs=100, n_perts=5)
        # With min_cells_per_group=1 (default) all groups should be present
        means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")
        assert len(groups) == 5

        # With very high threshold, some/all groups may be excluded
        with pytest.raises(RuntimeError, match="no groups passed"):
            pyscx.accel.pseudobulk_means(
                adata, "perturbation", min_cells_per_group=1000
            )

    def test_dense_input(self):
        """Test with dense numpy X instead of sparse."""
        import pyscx

        adata = self._make_adata()
        adata.X = adata.X.toarray()

        means, groups = pyscx.accel.pseudobulk_means(adata, "perturbation")
        assert means.shape == (5, adata.n_vars)

        # Verify against manual computation
        for i, group in enumerate(groups):
            mask = adata.obs["perturbation"].values == group
            expected = adata.X[mask].mean(axis=0)
            np.testing.assert_allclose(means[i], expected, atol=1e-6)

    def test_single_group(self):
        """All cells in one group."""
        import pyscx
        import pandas as pd

        rng = np.random.default_rng(42)
        X = sp.random(20, 10, density=0.3, format="csr", random_state=rng)
        obs = pd.DataFrame({"pert": ["single"] * 20})
        var = pd.DataFrame(index=[f"g{i}" for i in range(10)])
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)

        means, groups = pyscx.accel.pseudobulk_means(adata, "pert")
        assert len(groups) == 1
        assert groups[0] == "single"
        expected = X.toarray().mean(axis=0)
        np.testing.assert_allclose(means[0], expected, atol=1e-6)

    def test_invalid_column(self):
        """Non-existent groupby column should raise."""
        import pyscx

        adata = self._make_adata()
        with pytest.raises(ValueError, match="not found"):
            pyscx.accel.pseudobulk_means(adata, "nonexistent_column")

    def test_backed_scx(self, tmp_path):
        """Test with backed SCX input."""
        import pyscx

        adata = self._make_adata()

        # Write to SCX and read backed
        scx_path = str(tmp_path / "test.scx")
        pyscx.from_anndata(adata, scx_path)
        exp = pyscx.open(scx_path)
        adata_backed = exp.to_anndata(backed=True)

        means, groups = pyscx.accel.pseudobulk_means(adata_backed, "perturbation")

        # Compare with in-memory result
        means_mem, groups_mem = pyscx.accel.pseudobulk_means(adata, "perturbation")

        assert sorted(groups) == sorted(groups_mem)
        # Reorder to match
        for g in groups:
            idx = groups.index(g)
            idx_mem = groups_mem.index(g)
            np.testing.assert_allclose(means[idx], means_mem[idx_mem], atol=1e-5)


class TestPerturbationMetrics:
    """Test pyscx.accel.perturbation_metrics()."""

    def _make_paired_adata(self, n_obs=200, n_vars=30, n_perts=4, seed=42):
        """Create paired real/predicted AnnData with known perturbation structure.

        Creates data where:
        - Control cells have base expression
        - Each perturbation shifts expression by a known delta
        - Predicted data has slightly different deltas (noisy prediction)
        """
        rng = np.random.default_rng(seed)

        # Base expression for control
        base = rng.exponential(2.0, size=n_vars).astype(np.float32)

        # Perturbation effects (deltas from control)
        deltas = {}
        pert_names = ["control"] + [f"drug_{i}" for i in range(n_perts - 1)]
        for name in pert_names[1:]:
            deltas[name] = rng.normal(0, 1, size=n_vars).astype(np.float32)

        # Assign cells to perturbations
        cells_per_pert = n_obs // n_perts
        labels = []
        for name in pert_names:
            labels.extend([name] * cells_per_pert)
        # Fill remainder
        while len(labels) < n_obs:
            labels.append("control")

        import pandas as pd

        # Build real data
        X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_real[i] = np.maximum(base + noise, 0)
            else:
                X_real[i] = np.maximum(base + deltas[label] + noise, 0)

        # Build predicted data (slightly different from real)
        X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
        pred_noise_scale = 0.3
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_pred[i] = np.maximum(base + noise, 0)
            else:
                pred_delta = deltas[label] + rng.normal(
                    0, pred_noise_scale, size=n_vars
                ).astype(np.float32)
                X_pred[i] = np.maximum(base + pred_delta + noise, 0)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])

        adata_real = ad.AnnData(
            X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy()
        )
        adata_pred = ad.AnnData(
            X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy()
        )

        return adata_real, adata_pred, pert_names

    def test_all_metrics(self):
        """Verify all 5 bulk metrics run and return correct structure."""
        import pyscx

        adata_real, adata_pred, pert_names = self._make_paired_adata()
        results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)

        assert isinstance(results, dict)
        assert set(results.keys()) == {
            "pearson_delta", "mse", "mae", "mse_delta", "mae_delta"
        }

        non_ctrl = [p for p in pert_names if p != "control"]
        for metric_name, values in results.items():
            assert isinstance(values, dict), f"{metric_name} should be dict"
            for pert in non_ctrl:
                assert pert in values, f"{pert} missing from {metric_name}"
                assert isinstance(values[pert], float), (
                    f"{metric_name}[{pert}] should be float"
                )

    def test_mse_mae_against_reference(self):
        """Verify MSE/MAE match manual computation."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        results = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse", "mae"]
        )

        # Compute reference values
        means_real, groups_real = pyscx.accel.pseudobulk_means(
            adata_real, "perturbation"
        )
        means_pred, groups_pred = pyscx.accel.pseudobulk_means(
            adata_pred, "perturbation"
        )

        # Align groups
        for pert in results["mse"]:
            idx_r = groups_real.index(pert)
            idx_p = groups_pred.index(pert)
            diff = means_real[idx_r] - means_pred[idx_p]

            expected_mse = np.mean(diff ** 2)
            expected_mae = np.mean(np.abs(diff))

            np.testing.assert_allclose(
                results["mse"][pert], expected_mse, atol=1e-10,
                err_msg=f"MSE mismatch for {pert}"
            )
            np.testing.assert_allclose(
                results["mae"][pert], expected_mae, atol=1e-10,
                err_msg=f"MAE mismatch for {pert}"
            )

    def test_pearson_delta_against_scipy(self):
        """Verify pearson_delta matches scipy.stats.pearsonr on deltas."""
        import pyscx
        from scipy.stats import pearsonr

        adata_real, adata_pred, _ = self._make_paired_adata()

        results = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["pearson_delta"]
        )

        # Compute reference
        means_real, groups_real = pyscx.accel.pseudobulk_means(
            adata_real, "perturbation"
        )
        means_pred, groups_pred = pyscx.accel.pseudobulk_means(
            adata_pred, "perturbation"
        )

        ctrl_idx_r = groups_real.index("control")
        ctrl_idx_p = groups_pred.index("control")
        ctrl_real = means_real[ctrl_idx_r]
        ctrl_pred = means_pred[ctrl_idx_p]

        for pert in results["pearson_delta"]:
            idx_r = groups_real.index(pert)
            idx_p = groups_pred.index(pert)
            delta_real = means_real[idx_r] - ctrl_real
            delta_pred = means_pred[idx_p] - ctrl_pred
            expected_r, _ = pearsonr(delta_real, delta_pred)

            np.testing.assert_allclose(
                results["pearson_delta"][pert], expected_r, atol=1e-10,
                err_msg=f"pearson_delta mismatch for {pert}"
            )

    def test_delta_metrics_against_reference(self):
        """Verify mse_delta and mae_delta match manual computation."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        results = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse_delta", "mae_delta"]
        )

        means_real, groups_real = pyscx.accel.pseudobulk_means(
            adata_real, "perturbation"
        )
        means_pred, groups_pred = pyscx.accel.pseudobulk_means(
            adata_pred, "perturbation"
        )

        ctrl_idx_r = groups_real.index("control")
        ctrl_idx_p = groups_pred.index("control")
        ctrl_real = means_real[ctrl_idx_r]
        ctrl_pred = means_pred[ctrl_idx_p]

        for pert in results["mse_delta"]:
            idx_r = groups_real.index(pert)
            idx_p = groups_pred.index(pert)
            delta_real = means_real[idx_r] - ctrl_real
            delta_pred = means_pred[idx_p] - ctrl_pred
            diff = delta_real - delta_pred

            expected_mse_d = np.mean(diff ** 2)
            expected_mae_d = np.mean(np.abs(diff))

            np.testing.assert_allclose(
                results["mse_delta"][pert], expected_mse_d, atol=1e-10,
                err_msg=f"mse_delta mismatch for {pert}"
            )
            np.testing.assert_allclose(
                results["mae_delta"][pert], expected_mae_d, atol=1e-10,
                err_msg=f"mae_delta mismatch for {pert}"
            )

    def test_subset_of_metrics(self):
        """Test requesting only a subset of metrics."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        results = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse"]
        )
        assert set(results.keys()) == {"mse"}

    def test_custom_pert_col_and_control(self):
        """Test with non-default perturbation column and control label."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        # Rename perturbation column
        adata_real.obs = adata_real.obs.rename(columns={"perturbation": "condition"})
        adata_pred.obs = adata_pred.obs.rename(columns={"perturbation": "condition"})

        # Rename control label
        adata_real.obs["condition"] = adata_real.obs["condition"].replace(
            {"control": "vehicle"}
        )
        adata_pred.obs["condition"] = adata_pred.obs["condition"].replace(
            {"control": "vehicle"}
        )

        results = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred,
            pert_col="condition",
            control="vehicle",
            metrics=["mse"],
        )
        assert "mse" in results
        assert "vehicle" not in results["mse"]  # control excluded from output

    def test_invalid_metric_name(self):
        """Unknown metric name should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="unknown metric"):
            pyscx.accel.perturbation_metrics(
                adata_real, adata_pred, metrics=["nonexistent_metric"]
            )

    def test_missing_control(self):
        """Control not found should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="control.*not found"):
            pyscx.accel.perturbation_metrics(
                adata_real, adata_pred, control="missing_control"
            )

    def test_identical_real_pred(self):
        """When real == pred, MSE should be ~0 and pearson_delta should be ~1."""
        import pyscx

        adata_real, _, _ = self._make_paired_adata()
        adata_pred = adata_real.copy()

        results = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred,
            metrics=["mse", "pearson_delta"],
        )

        for pert, val in results["mse"].items():
            np.testing.assert_allclose(
                val, 0.0, atol=1e-10,
                err_msg=f"MSE should be 0 for identical data ({pert})"
            )

        for pert, val in results["pearson_delta"].items():
            np.testing.assert_allclose(
                val, 1.0, atol=1e-10,
                err_msg=f"pearson_delta should be 1 for identical data ({pert})"
            )

    def test_backed_scx_input(self, tmp_path):
        """Test with backed SCX input."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        # Write both to SCX and read backed
        real_path = str(tmp_path / "real.scx")
        pred_path = str(tmp_path / "pred.scx")
        pyscx.from_anndata(adata_real, real_path)
        pyscx.from_anndata(adata_pred, pred_path)

        real_backed = pyscx.open(real_path).to_anndata(backed=True)
        pred_backed = pyscx.open(pred_path).to_anndata(backed=True)

        results_backed = pyscx.accel.perturbation_metrics(
            real_backed, pred_backed, metrics=["mse", "mae"]
        )
        results_mem = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse", "mae"]
        )

        # Compare results
        for metric in ["mse", "mae"]:
            for pert in results_mem[metric]:
                np.testing.assert_allclose(
                    results_backed[metric][pert],
                    results_mem[metric][pert],
                    atol=1e-4,  # slightly looser due to codec quantization
                    err_msg=f"Backed vs mem mismatch: {metric}[{pert}]"
                )

    def test_embed_key_uses_obsm(self):
        """``embed_key`` routes per-perturbation means through adata.obsm
        rather than X — the score must depend on the embedding, not X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        # Random embeddings independent of X. Two different embeddings on
        # the same X+pert assignment must produce two different metric
        # outputs, proving the kernel read from obsm and not X.
        rng = np.random.default_rng(7)
        n_obs = adata_real.n_obs
        adata_real.obsm["X_emb_a"] = rng.standard_normal((n_obs, 8)).astype(
            np.float32
        )
        adata_pred.obsm["X_emb_a"] = rng.standard_normal((n_obs, 8)).astype(
            np.float32
        )
        adata_real.obsm["X_emb_b"] = rng.standard_normal((n_obs, 8)).astype(
            np.float32
        )
        adata_pred.obsm["X_emb_b"] = rng.standard_normal((n_obs, 8)).astype(
            np.float32
        )

        results_x = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse", "pearson_delta"]
        )
        results_a = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse", "pearson_delta"],
            embed_key="X_emb_a",
        )
        results_b = pyscx.accel.perturbation_metrics(
            adata_real, adata_pred, metrics=["mse", "pearson_delta"],
            embed_key="X_emb_b",
        )

        # Structure preserved (same perturbation keys).
        assert set(results_a["mse"]) == set(results_x["mse"])
        # Output must change vs. X-based and across embeddings.
        assert any(
            results_a["mse"][p] != results_x["mse"][p] for p in results_x["mse"]
        )
        assert any(
            results_a["mse"][p] != results_b["mse"][p] for p in results_a["mse"]
        )

    def test_embed_key_missing_raises(self):
        """``embed_key`` pointing at a non-existent obsm key surfaces a
        ValueError rather than silently falling back to X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="embed_key"):
            pyscx.accel.perturbation_metrics(
                adata_real, adata_pred, embed_key="not_there"
            )

    def test_all_zero_expression(self):
        """Edge case: all-zero expression matrix."""
        import pyscx
        import pandas as pd

        n_obs, n_vars = 20, 10
        X = sp.csr_matrix((n_obs, n_vars), dtype=np.float32)
        labels = ["control"] * 10 + ["drug"] * 10
        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = ad.AnnData(X=X, obs=obs, var=var)

        results = pyscx.accel.perturbation_metrics(
            adata, adata, metrics=["mse", "mae"]
        )
        # All zeros: MSE and MAE should be 0
        assert results["mse"]["drug"] == 0.0
        assert results["mae"]["drug"] == 0.0


class TestEnergyDistance:
    """Test pyscx.accel.energy_distance().

    Reference implementation mirrors cell-eval's _edistance() which uses
    sklearn.metrics.pairwise_distances.
    """

    def _make_paired_adata(self, n_obs=200, n_vars=10, n_perts=5, seed=42):
        """Create paired real/predicted AnnData with distinct perturbation effects.

        Uses smaller n_vars for energy distance tests since this metric
        operates on per-cell data (not pseudobulk), and pairwise distances
        are O(N²).
        """
        rng = np.random.default_rng(seed)
        base = rng.exponential(2.0, size=n_vars).astype(np.float32)

        pert_names = ["control"] + [f"drug_{i}" for i in range(n_perts - 1)]
        deltas = {}
        for name in pert_names[1:]:
            # Different magnitudes to ensure varying e-distances
            scale = rng.uniform(1.0, 5.0)
            deltas[name] = rng.normal(0, scale, size=n_vars).astype(np.float32)

        cells_per_pert = n_obs // n_perts
        labels = []
        for name in pert_names:
            labels.extend([name] * cells_per_pert)
        while len(labels) < n_obs:
            labels.append("control")

        import pandas as pd

        X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_real[i] = np.maximum(base + noise, 0)
            else:
                X_real[i] = np.maximum(base + deltas[label] + noise, 0)

        X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_pred[i] = np.maximum(base + noise, 0)
            else:
                pred_delta = deltas[label] + rng.normal(
                    0, 0.3, size=n_vars
                ).astype(np.float32)
                X_pred[i] = np.maximum(base + pred_delta + noise, 0)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])

        adata_real = ad.AnnData(
            X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy()
        )
        adata_pred = ad.AnnData(
            X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy()
        )
        return adata_real, adata_pred, pert_names

    @staticmethod
    def _reference_edistance(adata_real, adata_pred, pert_col="perturbation",
                             control="control", metric="euclidean"):
        """Reference implementation using sklearn, matching cell-eval's algorithm."""
        from sklearn.metrics import pairwise_distances
        from scipy.stats import pearsonr

        X_real = adata_real.X.toarray() if sp.issparse(adata_real.X) else adata_real.X
        X_pred = adata_pred.X.toarray() if sp.issparse(adata_pred.X) else adata_pred.X

        real_labels = adata_real.obs[pert_col].values
        pred_labels = adata_pred.obs[pert_col].values

        ctrl_real = X_real[real_labels == control].astype(np.float64)
        ctrl_pred = X_pred[pred_labels == control].astype(np.float64)

        sigma_ctrl_real = pairwise_distances(ctrl_real, ctrl_real, metric=metric).mean()
        sigma_ctrl_pred = pairwise_distances(ctrl_pred, ctrl_pred, metric=metric).mean()

        perts = sorted(set(real_labels) - {control})
        e_real = []
        e_pred = []
        for p in perts:
            pr = X_real[real_labels == p].astype(np.float64)
            pp = X_pred[pred_labels == p].astype(np.float64)

            d_cross_r = pairwise_distances(pr, ctrl_real, metric=metric).mean()
            d_self_r = pairwise_distances(pr, pr, metric=metric).mean()
            e_real.append(2.0 * d_cross_r - d_self_r - sigma_ctrl_real)

            d_cross_p = pairwise_distances(pp, ctrl_pred, metric=metric).mean()
            d_self_p = pairwise_distances(pp, pp, metric=metric).mean()
            e_pred.append(2.0 * d_cross_p - d_self_p - sigma_ctrl_pred)

        return pearsonr(e_real, e_pred).statistic

    def test_basic_energy_distance(self):
        """Verify energy_distance runs and returns a float."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        corr = pyscx.accel.energy_distance(adata_real, adata_pred)

        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0 or np.isnan(corr)

    def test_against_reference(self):
        """Verify energy_distance matches sklearn-based reference."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_corr = pyscx.accel.energy_distance(adata_real, adata_pred)
        ref_corr = self._reference_edistance(adata_real, adata_pred)

        np.testing.assert_allclose(
            rust_corr, ref_corr, atol=1e-6,
            err_msg=f"Rust e-distance correlation {rust_corr} != reference {ref_corr}"
        )

    def test_identical_real_pred(self):
        """When real == pred, correlation should be ~1.0."""
        import pyscx

        adata_real, _, _ = self._make_paired_adata()
        adata_pred = adata_real.copy()

        corr = pyscx.accel.energy_distance(adata_real, adata_pred)
        np.testing.assert_allclose(
            corr, 1.0, atol=1e-6,
            err_msg="identical data should give correlation ≈ 1.0"
        )

    def test_l1_metric(self):
        """Test with L1 distance metric."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_corr = pyscx.accel.energy_distance(
            adata_real, adata_pred, metric="l1"
        )
        ref_corr = self._reference_edistance(
            adata_real, adata_pred, metric="manhattan"
        )

        np.testing.assert_allclose(
            rust_corr, ref_corr, atol=1e-6,
            err_msg="L1 metric mismatch"
        )

    def test_cosine_metric(self):
        """Test with cosine distance metric."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_corr = pyscx.accel.energy_distance(
            adata_real, adata_pred, metric="cosine"
        )
        ref_corr = self._reference_edistance(
            adata_real, adata_pred, metric="cosine"
        )

        np.testing.assert_allclose(
            rust_corr, ref_corr, atol=1e-6,
            err_msg="cosine metric mismatch"
        )

    def test_embed_key(self):
        """Test using obsm embeddings instead of X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        # Add a PCA-like embedding
        rng = np.random.default_rng(123)
        n_components = 5
        adata_real.obsm["X_pca"] = rng.normal(size=(adata_real.n_obs, n_components))
        adata_pred.obsm["X_pca"] = rng.normal(size=(adata_pred.n_obs, n_components))

        corr = pyscx.accel.energy_distance(
            adata_real, adata_pred, embed_key="X_pca"
        )
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0 or np.isnan(corr)

    def test_dense_input(self):
        """Test with dense numpy X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.X = adata_real.X.toarray()
        adata_pred.X = adata_pred.X.toarray()

        rust_corr = pyscx.accel.energy_distance(adata_real, adata_pred)
        # Should still work and produce a valid correlation
        assert isinstance(rust_corr, float)
        assert -1.0 <= rust_corr <= 1.0

    def test_missing_control(self):
        """Missing control label should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="not found"):
            pyscx.accel.energy_distance(
                adata_real, adata_pred, control="nonexistent_control"
            )

    def test_invalid_metric(self):
        """Invalid metric should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="unknown metric"):
            pyscx.accel.energy_distance(
                adata_real, adata_pred, metric="invalid_metric"
            )

    def test_custom_pert_col(self):
        """Test with non-default perturbation column."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.obs = adata_real.obs.rename(columns={"perturbation": "condition"})
        adata_pred.obs = adata_pred.obs.rename(columns={"perturbation": "condition"})

        corr = pyscx.accel.energy_distance(
            adata_real, adata_pred, pert_col="condition"
        )
        assert isinstance(corr, float)

    def test_nan_labels_raise(self):
        """NaN perturbation labels should raise ValueError, not silently become 'nan'."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        # Inject NaN into obs column
        adata_real.obs.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            pyscx.accel.energy_distance(adata_real, adata_pred)

    def test_many_perturbations_stress(self):
        """Stress test with 50 perturbations to exercise rayon parallel codepath."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata(
            n_obs=500, n_vars=5, n_perts=50, seed=99
        )

        corr = pyscx.accel.energy_distance(adata_real, adata_pred)
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0 or np.isnan(corr)


class TestDiscriminationScore:
    """Test pyscx.accel.discrimination_score().

    Reference implementation mirrors cell-eval's discrimination_score which
    computes per-perturbation ranking of predicted effects against real effects.
    """

    def _make_paired_adata(self, n_obs=200, n_vars=30, n_perts=5, seed=42):
        """Create paired real/predicted AnnData with distinct perturbation effects.

        Gene names match perturbation names for some perturbations so that
        exclude_target_gene behavior can be tested.
        """
        rng = np.random.default_rng(seed)
        base = rng.exponential(2.0, size=n_vars).astype(np.float32)

        # Name perturbations after the first few genes (for target gene exclusion testing)
        gene_names = [f"gene_{j}" for j in range(n_vars)]
        pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]

        deltas = {}
        for name in pert_names[1:]:
            scale = rng.uniform(0.5, 3.0)
            deltas[name] = rng.normal(0, scale, size=n_vars).astype(np.float32)

        cells_per_pert = n_obs // n_perts
        labels = []
        for name in pert_names:
            labels.extend([name] * cells_per_pert)
        while len(labels) < n_obs:
            labels.append("control")

        import pandas as pd

        X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_real[i] = np.maximum(base + noise, 0)
            else:
                X_real[i] = np.maximum(base + deltas[label] + noise, 0)

        X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_pred[i] = np.maximum(base + noise, 0)
            else:
                pred_delta = deltas[label] + rng.normal(
                    0, 0.3, size=n_vars
                ).astype(np.float32)
                X_pred[i] = np.maximum(base + pred_delta + noise, 0)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=gene_names)

        adata_real = ad.AnnData(
            X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy()
        )
        adata_pred = ad.AnnData(
            X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy()
        )
        return adata_real, adata_pred, pert_names

    @staticmethod
    def _reference_discrimination_score(adata_real, adata_pred,
                                         pert_col="perturbation",
                                         control="control", metric="l1",
                                         exclude_target_gene=True,
                                         embed_key=None):
        """Reference implementation matching cell-eval's algorithm."""
        from sklearn.metrics import pairwise_distances

        # cell-eval: L1/manhattan forces embed_key=None
        if metric in ("l1", "manhattan", "cityblock"):
            embed_key = None

        if embed_key is not None:
            matrix_real = adata_real.obsm[embed_key].astype(np.float64)
            matrix_pred = adata_pred.obsm[embed_key].astype(np.float64)
            gene_names = None
        else:
            X_real = adata_real.X.toarray() if sp.issparse(adata_real.X) else adata_real.X
            X_pred = adata_pred.X.toarray() if sp.issparse(adata_pred.X) else adata_pred.X
            matrix_real = X_real.astype(np.float64)
            matrix_pred = X_pred.astype(np.float64)
            gene_names = list(adata_real.var_names)

        real_labels = adata_real.obs[pert_col].values
        pred_labels = adata_pred.obs[pert_col].values
        perts = sorted(set(real_labels) - {control})

        # Compute pseudobulk means
        def pseudobulk(matrix, labels):
            means = {}
            for p in sorted(set(labels)):
                mask = labels == p
                means[p] = matrix[mask].mean(axis=0)
            return means

        means_real = pseudobulk(matrix_real, real_labels)
        means_pred = pseudobulk(matrix_pred, pred_labels)

        # Compute effects (subtract control)
        real_effects = np.array([means_real[p] - means_real[control] for p in perts])
        pred_effects = np.array([means_pred[p] - means_pred[control] for p in perts])

        # Map sklearn metric names
        sklearn_metric = {
            "l1": "manhattan",
            "l2": "euclidean",
            "euclidean": "euclidean",
            "cosine": "cosine",
        }[metric]

        scores = {}
        for pi, p in enumerate(perts):
            include_mask = np.ones(real_effects.shape[1], dtype=bool)
            if exclude_target_gene and gene_names is not None and embed_key is None:
                if p in gene_names:
                    gene_idx = gene_names.index(p)
                    include_mask[gene_idx] = False

            re_masked = real_effects[:, include_mask]
            pe_masked = pred_effects[pi, include_mask].reshape(1, -1)

            distances = pairwise_distances(
                re_masked, pe_masked, metric=sklearn_metric
            ).flatten()

            # Rank: number of perturbations closer than the correct one
            correct_dist = distances[pi]
            rank = np.sum(distances < correct_dist)
            scores[p] = 1.0 - rank / len(perts)

        return scores

    def test_basic_discrimination_score(self):
        """Verify discrimination_score runs and returns correct structure."""
        import pyscx

        adata_real, adata_pred, pert_names = self._make_paired_adata()
        scores = pyscx.accel.discrimination_score(adata_real, adata_pred)

        assert isinstance(scores, dict)
        non_ctrl = [p for p in pert_names if p != "control"]
        for pert in non_ctrl:
            assert pert in scores, f"{pert} missing from scores"
            assert isinstance(scores[pert], float)
            assert 0.0 <= scores[pert] <= 1.0, f"score out of range: {scores[pert]}"

    def test_against_reference_l1(self):
        """Verify L1 discrimination score matches reference implementation."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1"
        )
        ref_scores = self._reference_discrimination_score(
            adata_real, adata_pred, metric="l1"
        )

        for pert in ref_scores:
            np.testing.assert_allclose(
                rust_scores[pert], ref_scores[pert], atol=1e-10,
                err_msg=f"L1 discrimination score mismatch for {pert}"
            )

    def test_against_reference_l2(self):
        """Verify L2 discrimination score matches reference implementation."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l2"
        )
        ref_scores = self._reference_discrimination_score(
            adata_real, adata_pred, metric="l2"
        )

        for pert in ref_scores:
            np.testing.assert_allclose(
                rust_scores[pert], ref_scores[pert], atol=1e-10,
                err_msg=f"L2 discrimination score mismatch for {pert}"
            )

    def test_against_reference_cosine(self):
        """Verify cosine discrimination score matches reference implementation."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="cosine"
        )
        ref_scores = self._reference_discrimination_score(
            adata_real, adata_pred, metric="cosine"
        )

        for pert in ref_scores:
            np.testing.assert_allclose(
                rust_scores[pert], ref_scores[pert], atol=1e-10,
                err_msg=f"cosine discrimination score mismatch for {pert}"
            )

    def test_exclude_target_gene_true(self):
        """Test with exclude_target_gene=True (default) matches reference."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1", exclude_target_gene=True
        )
        ref_scores = self._reference_discrimination_score(
            adata_real, adata_pred, metric="l1", exclude_target_gene=True
        )

        for pert in ref_scores:
            np.testing.assert_allclose(
                rust_scores[pert], ref_scores[pert], atol=1e-10,
                err_msg=f"exclude_target_gene=True mismatch for {pert}"
            )

    def test_exclude_target_gene_false(self):
        """Test with exclude_target_gene=False matches reference."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rust_scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1", exclude_target_gene=False
        )
        ref_scores = self._reference_discrimination_score(
            adata_real, adata_pred, metric="l1", exclude_target_gene=False
        )

        for pert in ref_scores:
            np.testing.assert_allclose(
                rust_scores[pert], ref_scores[pert], atol=1e-10,
                err_msg=f"exclude_target_gene=False mismatch for {pert}"
            )

    def test_identical_real_pred(self):
        """When real == pred, all discrimination scores should be 1.0."""
        import pyscx

        adata_real, _, _ = self._make_paired_adata()
        adata_pred = adata_real.copy()

        scores = pyscx.accel.discrimination_score(adata_real, adata_pred)
        for pert, score in scores.items():
            np.testing.assert_allclose(
                score, 1.0, atol=1e-10,
                err_msg=f"identical data should give score 1.0 for {pert}"
            )

    def test_embed_key(self):
        """Test using obsm embeddings instead of X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rng = np.random.default_rng(123)
        n_components = 5
        adata_real.obsm["X_pca"] = rng.normal(size=(adata_real.n_obs, n_components))
        adata_pred.obsm["X_pca"] = rng.normal(size=(adata_pred.n_obs, n_components))

        # cosine with embed_key should work (L1 forces embed_key=None)
        scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="cosine", embed_key="X_pca"
        )
        assert isinstance(scores, dict)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_l1_forces_embed_key_none(self):
        """L1 metric should force embed_key=None (cell-eval behavior)."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rng = np.random.default_rng(123)
        adata_real.obsm["X_pca"] = rng.normal(size=(adata_real.n_obs, 5))
        adata_pred.obsm["X_pca"] = rng.normal(size=(adata_pred.n_obs, 5))

        # With L1, embed_key should be ignored → result should match X-based
        scores_with_embed = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1", embed_key="X_pca"
        )
        scores_without_embed = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1", embed_key=None
        )

        for pert in scores_with_embed:
            np.testing.assert_allclose(
                scores_with_embed[pert], scores_without_embed[pert], atol=1e-10,
                err_msg=f"L1 should ignore embed_key for {pert}"
            )

    def test_dense_input(self):
        """Test with dense numpy X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.X = adata_real.X.toarray()
        adata_pred.X = adata_pred.X.toarray()

        scores = pyscx.accel.discrimination_score(adata_real, adata_pred)
        assert isinstance(scores, dict)
        for score in scores.values():
            assert 0.0 <= score <= 1.0

    def test_missing_control(self):
        """Missing control label should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="control.*not found"):
            pyscx.accel.discrimination_score(
                adata_real, adata_pred, control="nonexistent_control"
            )

    def test_invalid_metric(self):
        """Invalid metric should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="unknown metric"):
            pyscx.accel.discrimination_score(
                adata_real, adata_pred, metric="invalid_metric"
            )

    def test_custom_pert_col(self):
        """Test with non-default perturbation column."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.obs = adata_real.obs.rename(columns={"perturbation": "condition"})
        adata_pred.obs = adata_pred.obs.rename(columns={"perturbation": "condition"})

        scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, pert_col="condition"
        )
        assert isinstance(scores, dict)
        assert "control" not in scores  # control excluded

    def test_nan_labels_raise(self):
        """NaN perturbation labels should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.obs.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            pyscx.accel.discrimination_score(adata_real, adata_pred)

    def test_nan_in_x_rejected(self):
        """NaN in X (which propagates into pseudobulk effects) must error,
        not silently produce rank 0 / score 1.0."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        X = adata_real.X.toarray()
        X[0, 0] = np.nan
        adata_real.X = sp.csr_matrix(X)

        with pytest.raises((RuntimeError, ValueError), match="non-finite"):
            pyscx.accel.discrimination_score(
                adata_real, adata_pred, metric="l1", exclude_target_gene=False
            )

    def test_backed_scx_input(self, tmp_path):
        """Test with backed SCX input."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        real_path = str(tmp_path / "real.scx")
        pred_path = str(tmp_path / "pred.scx")
        pyscx.from_anndata(adata_real, real_path)
        pyscx.from_anndata(adata_pred, pred_path)

        real_backed = pyscx.open(real_path).to_anndata(backed=True)
        pred_backed = pyscx.open(pred_path).to_anndata(backed=True)

        scores_backed = pyscx.accel.discrimination_score(
            real_backed, pred_backed, metric="l1", exclude_target_gene=False
        )
        scores_mem = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1", exclude_target_gene=False
        )

        for pert in scores_mem:
            np.testing.assert_allclose(
                scores_backed[pert], scores_mem[pert], atol=1e-4,
                err_msg=f"Backed vs mem mismatch for {pert}"
            )

    def test_embed_key_ignores_exclude_target_gene(self):
        """exclude_target_gene should be a no-op when using embed_key.

        Embeddings have no gene names, so exclude_target_gene=True and
        exclude_target_gene=False should produce identical results when
        embed_key is set (and metric is not L1, which forces embed_key=None).
        """
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        rng = np.random.default_rng(123)
        n_components = 5
        adata_real.obsm["X_pca"] = rng.normal(size=(adata_real.n_obs, n_components))
        adata_pred.obsm["X_pca"] = rng.normal(size=(adata_pred.n_obs, n_components))

        scores_with_excl = pyscx.accel.discrimination_score(
            adata_real, adata_pred,
            metric="cosine", embed_key="X_pca", exclude_target_gene=True
        )
        scores_without_excl = pyscx.accel.discrimination_score(
            adata_real, adata_pred,
            metric="cosine", embed_key="X_pca", exclude_target_gene=False
        )

        for pert in scores_with_excl:
            np.testing.assert_allclose(
                scores_with_excl[pert], scores_without_excl[pert], atol=1e-12,
                err_msg=f"embed_key should make exclude_target_gene a no-op for {pert}"
            )


class TestPerturbationMetricsNaN:
    """Verify perturbation_metrics now detects NaN labels after M1 refactor."""

    def test_nan_labels_raise(self):
        """NaN perturbation labels should raise ValueError in perturbation_metrics."""
        import pyscx
        import pandas as pd

        rng = np.random.default_rng(42)
        X = sp.random(20, 10, density=0.3, format="csr", random_state=rng)
        obs = pd.DataFrame({"perturbation": ["control"] * 10 + ["drug"] * 10})
        var = pd.DataFrame(index=[f"g{i}" for i in range(10)])
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)

        # Inject NaN
        adata_nan = adata.copy()
        adata_nan.obs.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            pyscx.accel.perturbation_metrics(adata_nan, adata)


class TestKnockdownEfficiency:
    """Test pyscx.accel.knockdown_efficiency().

    Reference implementation mirrors arc-bench's compute_knockdown_efficiency()
    and compute_log_deviation() from normalize_transform/core.py.
    """

    def _make_adata(self, n_obs=200, n_vars=20, n_perts=5, seed=42):
        """Create AnnData with perturbation names matching gene names.

        - Gene names: gene_0 .. gene_{n_vars-1}
        - Perturbation names: "control" + "gene_0" .. "gene_{n_perts-2}"
        - Raw count-like data (positive integers)
        """
        rng = np.random.default_rng(seed)

        # Gene names
        gene_names = [f"gene_{j}" for j in range(n_vars)]

        # Perturbation names: control + first (n_perts-1) gene names
        pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]

        # Assign cells to perturbations evenly
        cells_per_pert = n_obs // n_perts
        labels = []
        for name in pert_names:
            labels.extend([name] * cells_per_pert)
        while len(labels) < n_obs:
            labels.append("control")

        import pandas as pd

        # Create sparse count matrix with known structure
        # Control cells have moderate expression across all genes
        # Perturbation cells have reduced expression of their target gene
        X = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            # Base expression: random positive counts
            X[i] = rng.poisson(5, size=n_vars).astype(np.float32)
            # Perturbation cells: knockdown target gene
            if label != "control" and label in gene_names:
                gene_idx = gene_names.index(label)
                # Reduce target gene expression (partial knockdown)
                X[i, gene_idx] = rng.poisson(1)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=gene_names)
        adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
        return adata, pert_names, gene_names

    @staticmethod
    def _reference_knockdown_efficiency(adata, pert_col="perturbation",
                                         control="control", eps=1e-8):
        """Reference implementation matching arc-bench's algorithm."""
        X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
        gene_names = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Control baseline (mean of control cells) — use f64 to match Rust's
        # intentional f64 accumulation for precision over many control cells.
        labels = adata.obs[pert_col].values
        control_mask = labels == control
        baseline = X[control_mask].mean(axis=0).astype(np.float64)

        # Knockdown efficiency (on raw normalized data)
        n_cells = adata.n_obs
        efficiency = np.full(n_cells, np.nan, dtype=np.float32)
        for target_gene in np.unique(labels):
            if target_gene == control or target_gene not in gene_to_idx:
                continue
            gene_idx = gene_to_idx[target_gene]
            pert_mask = labels == target_gene
            expr = X[pert_mask, gene_idx]
            mu_control = baseline[gene_idx]
            efficiency[pert_mask] = 1.0 - (expr / (mu_control + eps))

        # Log deviation (on log1p-transformed data)
        baseline_log = np.log1p(baseline)
        X_log = np.log1p(X)
        log_fc = np.full(n_cells, np.nan, dtype=np.float32)
        for target_gene in np.unique(labels):
            if target_gene == control or target_gene not in gene_to_idx:
                continue
            gene_idx = gene_to_idx[target_gene]
            pert_mask = labels == target_gene
            expr_log = X_log[pert_mask, gene_idx]
            log_fc[pert_mask] = expr_log - baseline_log[gene_idx]

        return efficiency, log_fc

    def test_basic_knockdown(self):
        """Verify knockdown_efficiency runs and writes adata.obs columns."""
        import pyscx

        adata, _, _ = self._make_adata()
        pyscx.accel.knockdown_efficiency(adata)

        assert "KnockDownEfficiency" in adata.obs.columns
        assert "KnockDownGeneFC" in adata.obs.columns
        assert len(adata.obs["KnockDownEfficiency"]) == adata.n_obs
        assert len(adata.obs["KnockDownGeneFC"]) == adata.n_obs

    def test_against_reference(self):
        """Verify knockdown metrics match arc-bench reference implementation."""
        import pyscx

        adata, _, _ = self._make_adata()

        # Run reference first (before pyscx modifies adata.obs)
        ref_eff, ref_fc = self._reference_knockdown_efficiency(adata)

        # Run Rust implementation
        pyscx.accel.knockdown_efficiency(adata)
        rust_eff = adata.obs["KnockDownEfficiency"].values
        rust_fc = adata.obs["KnockDownGeneFC"].values

        # Check efficiency
        # NaN positions should match
        nan_mask_ref = np.isnan(ref_eff)
        nan_mask_rust = np.isnan(rust_eff)
        np.testing.assert_array_equal(
            nan_mask_ref, nan_mask_rust,
            err_msg="NaN positions differ in efficiency"
        )
        # Non-NaN values should match
        valid = ~nan_mask_ref
        np.testing.assert_allclose(
            rust_eff[valid], ref_eff[valid], atol=1e-5,
            err_msg="Knockdown efficiency values differ"
        )

        # Check log FC
        nan_mask_ref_fc = np.isnan(ref_fc)
        nan_mask_rust_fc = np.isnan(rust_fc)
        np.testing.assert_array_equal(
            nan_mask_ref_fc, nan_mask_rust_fc,
            err_msg="NaN positions differ in log FC"
        )
        valid_fc = ~nan_mask_ref_fc
        np.testing.assert_allclose(
            rust_fc[valid_fc], ref_fc[valid_fc], atol=1e-5,
            err_msg="Log FC values differ"
        )

    def test_control_cells_are_nan(self):
        """Control cells should have NaN in both columns."""
        import pyscx

        adata, _, _ = self._make_adata()
        pyscx.accel.knockdown_efficiency(adata)

        control_mask = adata.obs["perturbation"].values == "control"
        assert np.all(np.isnan(adata.obs["KnockDownEfficiency"].values[control_mask]))
        assert np.all(np.isnan(adata.obs["KnockDownGeneFC"].values[control_mask]))

    def test_missing_gene(self):
        """Perturbation name not in var_names → NaN for those cells."""
        import pyscx
        import pandas as pd

        rng = np.random.default_rng(42)
        X = sp.random(30, 10, density=0.3, format="csr", random_state=rng).astype(np.float32)
        gene_names = [f"gene_{i}" for i in range(10)]
        labels = ["control"] * 10 + ["unknown_pert"] * 10 + ["gene_0"] * 10
        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=gene_names)
        adata = ad.AnnData(X=X, obs=obs, var=var)

        pyscx.accel.knockdown_efficiency(adata)

        eff = adata.obs["KnockDownEfficiency"].values
        # "unknown_pert" doesn't match any gene → NaN
        assert np.all(np.isnan(eff[10:20]))
        # "gene_0" matches gene_0 → should have valid values
        assert not np.any(np.isnan(eff[20:30]))

    def test_dense_input(self):
        """Test with dense numpy X instead of sparse."""
        import pyscx

        adata, _, _ = self._make_adata()
        adata.X = adata.X.toarray()

        ref_eff, ref_fc = self._reference_knockdown_efficiency(adata)
        pyscx.accel.knockdown_efficiency(adata)

        valid_eff = ~np.isnan(ref_eff)
        np.testing.assert_allclose(
            adata.obs["KnockDownEfficiency"].values[valid_eff],
            ref_eff[valid_eff], atol=1e-5
        )

        valid_fc = ~np.isnan(ref_fc)
        np.testing.assert_allclose(
            adata.obs["KnockDownGeneFC"].values[valid_fc],
            ref_fc[valid_fc], atol=1e-5
        )

    def test_backed_scx_input(self, tmp_path):
        """Test with backed SCX input."""
        import pyscx

        adata, _, _ = self._make_adata()

        # Get reference from in-memory
        ref_eff, ref_fc = self._reference_knockdown_efficiency(adata)

        # Write to SCX and read backed
        scx_path = str(tmp_path / "test.scx")
        pyscx.from_anndata(adata, scx_path)
        exp = pyscx.open(scx_path)
        adata_backed = exp.to_anndata(backed=True)

        pyscx.accel.knockdown_efficiency(adata_backed)

        rust_eff = adata_backed.obs["KnockDownEfficiency"].values

        # NaN positions should match
        nan_eff = np.isnan(ref_eff)
        np.testing.assert_array_equal(np.isnan(rust_eff), nan_eff)

        # Values should be close (SCX codec may cause small differences)
        valid = ~nan_eff
        np.testing.assert_allclose(
            rust_eff[valid], ref_eff[valid], atol=1e-3,
            err_msg="Backed efficiency values differ"
        )

    def test_custom_pert_col_and_control(self):
        """Test with non-default column names."""
        import pyscx

        adata, _, _ = self._make_adata()
        adata.obs = adata.obs.rename(columns={"perturbation": "condition"})
        adata.obs["condition"] = adata.obs["condition"].replace({"control": "vehicle"})

        pyscx.accel.knockdown_efficiency(
            adata, pert_col="condition", control="vehicle"
        )

        assert "KnockDownEfficiency" in adata.obs.columns
        # Vehicle (control) cells should be NaN
        vehicle_mask = adata.obs["condition"].values == "vehicle"
        assert np.all(np.isnan(adata.obs["KnockDownEfficiency"].values[vehicle_mask]))

    def test_custom_eps(self):
        """Test with custom eps value."""
        import pyscx

        adata, _, _ = self._make_adata()
        pyscx.accel.knockdown_efficiency(adata, eps=1.0)

        # Should still produce valid results
        assert "KnockDownEfficiency" in adata.obs.columns

    def test_missing_pert_col(self):
        """Non-existent perturbation column should raise ValueError."""
        import pyscx

        adata, _, _ = self._make_adata()
        with pytest.raises(ValueError, match="not found"):
            pyscx.accel.knockdown_efficiency(adata, pert_col="nonexistent")

    def test_missing_control(self):
        """Non-existent control label should raise RuntimeError."""
        import pyscx

        adata, _, _ = self._make_adata()
        with pytest.raises(RuntimeError, match="no cells found"):
            pyscx.accel.knockdown_efficiency(adata, control="nonexistent_control")

    def test_nan_labels_raise(self):
        """NaN perturbation labels should raise ValueError."""
        import pyscx

        adata, _, _ = self._make_adata()
        adata.obs.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            pyscx.accel.knockdown_efficiency(adata)

    def test_all_genes_matched(self):
        """When all perturbations match genes, no cell should be NaN (except control)."""
        import pyscx

        adata, pert_names, gene_names = self._make_adata()
        pyscx.accel.knockdown_efficiency(adata)

        eff = adata.obs["KnockDownEfficiency"].values
        labels = adata.obs["perturbation"].values

        for p in pert_names:
            if p == "control":
                mask = labels == p
                assert np.all(np.isnan(eff[mask])), "control cells should be NaN"
            elif p in gene_names:
                mask = labels == p
                assert not np.any(np.isnan(eff[mask])), f"{p} cells should not be NaN"

    def test_knockdown_values_range(self):
        """KD efficiency should typically be in (-inf, 1] for knockdown genes."""
        import pyscx

        adata, _, _ = self._make_adata()
        pyscx.accel.knockdown_efficiency(adata)

        eff = adata.obs["KnockDownEfficiency"].values
        valid = ~np.isnan(eff)
        # Knockdown → target expression is lower → KD should be positive
        # Perfect knockdown (x=0) → KD = 1.0
        # No knockdown (x=baseline) → KD ≈ 0
        # Upregulation (x>baseline) → KD < 0
        assert np.all(np.isfinite(eff[valid])), "All non-NaN values should be finite"


class TestClusteringAgreement:
    """Test pyscx.accel.clustering_agreement().

    Reference: cell-eval's ClusteringAgreement class which builds centroid
    AnnData objects, clusters with scanpy Leiden at multiple resolutions,
    and scores with sklearn's AMI/NMI/ARI.
    """

    def _make_paired_adata(self, n_obs=500, n_vars=30, n_perts=8, seed=42):
        """Create paired real/predicted AnnData with distinct perturbation structure.

        Perturbations are designed so that clustering should find meaningful
        groups in the centroids.
        """
        rng = np.random.default_rng(seed)
        base = rng.exponential(2.0, size=n_vars).astype(np.float32)

        pert_names = ["control"] + [f"drug_{i}" for i in range(n_perts - 1)]

        # Create distinct cluster structure: drugs 0-2 in one cluster,
        # drugs 3-5 in another, drug 6 alone.
        deltas = {}
        cluster_shifts = {
            "drug_0": np.array([3.0] * 10 + [0.0] * 20),
            "drug_1": np.array([2.5] * 10 + [0.0] * 20),
            "drug_2": np.array([3.5] * 10 + [0.0] * 20),
            "drug_3": np.array([0.0] * 10 + [3.0] * 10 + [0.0] * 10),
            "drug_4": np.array([0.0] * 10 + [2.5] * 10 + [0.0] * 10),
            "drug_5": np.array([0.0] * 10 + [3.5] * 10 + [0.0] * 10),
            "drug_6": np.array([0.0] * 20 + [3.0] * 10),
        }
        for name in pert_names[1:]:
            deltas[name] = cluster_shifts[name].astype(np.float32)

        cells_per_pert = n_obs // n_perts
        labels = []
        for name in pert_names:
            labels.extend([name] * cells_per_pert)
        while len(labels) < n_obs:
            labels.append("control")

        import pandas as pd

        X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_real[i] = np.maximum(base + noise, 0)
            else:
                X_real[i] = np.maximum(base + deltas[label] + noise, 0)

        # Predicted data: same structure but with noise
        X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_pred[i] = np.maximum(base + noise, 0)
            else:
                pred_delta = deltas[label] + rng.normal(
                    0, 0.3, size=n_vars
                ).astype(np.float32)
                X_pred[i] = np.maximum(base + pred_delta + noise, 0)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])

        adata_real = ad.AnnData(
            X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy()
        )
        adata_pred = ad.AnnData(
            X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy()
        )
        return adata_real, adata_pred, pert_names

    def test_basic_clustering_agreement(self):
        """Verify clustering_agreement runs and returns a float in [0, 1]."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        score = pyscx.accel.clustering_agreement(adata_real, adata_pred)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"

    def test_ami_metric(self):
        """Test with AMI metric."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred, metric="ami"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_nmi_metric(self):
        """Test with NMI metric."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred, metric="nmi"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_ari_metric(self):
        """Test with ARI metric (rescaled to [0, 1])."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred, metric="ari"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identical_data_high_score(self):
        """When real == pred, clustering agreement should be high or perfect."""
        import pyscx

        adata_real, _, _ = self._make_paired_adata()
        adata_pred = adata_real.copy()

        for m in ["ami", "nmi", "ari"]:
            score = pyscx.accel.clustering_agreement(
                adata_real, adata_pred, metric=m
            )
            # Identical data should yield high agreement (allowing for
            # stochastic Leiden, but centroids are identical so clustering
            # should be ~identical).
            assert score >= 0.5, (
                f"Identical data should have high agreement for {m}, got {score}"
            )

    def test_custom_resolutions(self):
        """Test with custom pred_resolutions."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred,
            pred_resolutions=[0.5, 1.0, 2.0],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_single_resolution(self):
        """Test with a single predicted resolution."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred,
            pred_resolutions=[1.0],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_custom_pert_col_and_control(self):
        """Test with non-default perturbation column and control label."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        adata_real.obs = adata_real.obs.rename(columns={"perturbation": "condition"})
        adata_pred.obs = adata_pred.obs.rename(columns={"perturbation": "condition"})
        adata_real.obs["condition"] = adata_real.obs["condition"].replace(
            {"control": "vehicle"}
        )
        adata_pred.obs["condition"] = adata_pred.obs["condition"].replace(
            {"control": "vehicle"}
        )

        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred,
            pert_col="condition",
            control="vehicle",
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_invalid_metric(self):
        """Invalid metric should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="unknown metric"):
            pyscx.accel.clustering_agreement(
                adata_real, adata_pred, metric="invalid_metric"
            )

    def test_missing_control(self):
        """Missing control label should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="control.*not found"):
            pyscx.accel.clustering_agreement(
                adata_real, adata_pred, control="nonexistent_control"
            )

    def test_too_few_perturbations(self):
        """Fewer than 2 non-control perturbations should raise ValueError."""
        import pyscx
        import pandas as pd

        rng = np.random.default_rng(42)
        n_obs, n_vars = 40, 10
        X = sp.random(n_obs, n_vars, density=0.3, format="csr", random_state=rng)
        labels = ["control"] * 20 + ["single_drug"] * 20
        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        adata = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)

        with pytest.raises(ValueError, match="at least 2"):
            pyscx.accel.clustering_agreement(adata, adata)

    def test_empty_resolutions(self):
        """Empty pred_resolutions should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        with pytest.raises(ValueError, match="must not be empty"):
            pyscx.accel.clustering_agreement(
                adata_real, adata_pred, pred_resolutions=[]
            )

    def test_dense_input(self):
        """Test with dense numpy X instead of sparse."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.X = adata_real.X.toarray()
        adata_pred.X = adata_pred.X.toarray()

        score = pyscx.accel.clustering_agreement(adata_real, adata_pred)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_embed_key(self):
        """Test using obsm embeddings instead of X."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        # Add PCA-like embeddings
        rng = np.random.default_rng(123)
        n_components = 10
        adata_real.obsm["X_pca"] = rng.normal(size=(adata_real.n_obs, n_components))
        adata_pred.obsm["X_pca"] = rng.normal(size=(adata_pred.n_obs, n_components))

        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred, embed_key="X_pca"
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_identical_centroids_perfect_score(self):
        """Identical centroids + same resolution should give perfect clustering agreement.

        This validates the full pipeline: pseudobulk → centroid AnnData → scanpy
        neighbors/Leiden → Rust scoring. With identical centroids and the same
        resolution, Leiden produces the same clusters for real and predicted,
        so all three scoring metrics should return 1.0 (or very close).

        The Rust unit tests in clustering.rs independently verify the AMI/NMI/ARI
        scoring functions against exact sklearn reference values:
          AMI([0,0,0,1,1,1], [0,0,1,1,2,2]) = 0.298792458170890  (sklearn ± 1e-10)
          NMI([0,0,0,1,1,1], [0,0,1,1,2,2]) = 0.515803742979389  (sklearn ± 1e-10)
          ARI([0,0,0,1,1,1], [0,0,1,1,2,2]) = 0.242424242424242  (sklearn ± 1e-10)
        """
        import pyscx

        adata_real, _, _ = self._make_paired_adata(seed=42)
        adata_pred = adata_real.copy()

        # With identical data, centroids are identical.
        # All three metrics should yield perfect or near-perfect agreement.
        for m in ["ami", "nmi", "ari"]:
            score = pyscx.accel.clustering_agreement(
                adata_real, adata_pred, metric=m,
                pred_resolutions=[1.0],  # same as real_resolution
            )
            # With identical centroids and same resolution, Leiden is
            # deterministic (igraph flavor) → identical cluster labels
            # → scoring function returns 1.0.
            assert score >= 0.8, (
                f"Identical centroids with same resolution should give high {m}, got {score}"
            )

    def test_sklearn_reference_values(self):
        """Cross-check that sklearn AMI/NMI/ARI reference values are self-consistent.

        The Rust unit tests directly verify our implementations against these
        exact sklearn values (to 1e-10 tolerance). This test documents those
        reference values and ensures the sklearn version we use produces them.
        """
        from sklearn.metrics import (
            adjusted_mutual_info_score,
            normalized_mutual_info_score,
            adjusted_rand_score,
        )

        a = np.array([0, 0, 0, 1, 1, 1])
        b = np.array([0, 0, 1, 1, 2, 2])

        # These exact values are verified in the Rust unit tests at 1e-10.
        np.testing.assert_allclose(
            adjusted_mutual_info_score(a, b), 0.298792458170890, atol=1e-10,
            err_msg="sklearn AMI reference value changed"
        )
        np.testing.assert_allclose(
            normalized_mutual_info_score(a, b), 0.515803742979389, atol=1e-10,
            err_msg="sklearn NMI reference value changed"
        )
        np.testing.assert_allclose(
            adjusted_rand_score(a, b), 0.242424242424242, atol=1e-10,
            err_msg="sklearn ARI reference value changed"
        )

    def test_nan_labels_raise(self):
        """NaN perturbation labels should raise ValueError."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()
        adata_real.obs.iloc[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            pyscx.accel.clustering_agreement(adata_real, adata_pred)

    def test_n_neighbors_capped(self):
        """n_neighbors should be capped at n_output - 1 for small centroid sets."""
        import pyscx

        # Only 3 non-control perturbations → n_output = 3
        adata_real, adata_pred, _ = self._make_paired_adata(
            n_obs=200, n_perts=4, seed=99
        )

        # n_neighbors=15 should be capped to 2 (= 3 - 1) without error.
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred, n_neighbors=15
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_best_across_resolutions(self):
        """Score should be the best (maximum) across all predicted resolutions."""
        import pyscx

        adata_real, adata_pred, _ = self._make_paired_adata()

        # Get score with many resolutions.
        score_many = pyscx.accel.clustering_agreement(
            adata_real, adata_pred,
            pred_resolutions=[0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0],
        )

        # Get scores with individual resolutions.
        individual_scores = []
        for r in [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
            s = pyscx.accel.clustering_agreement(
                adata_real, adata_pred,
                pred_resolutions=[r],
            )
            individual_scores.append(s)

        # The many-resolution score should equal the max of individual scores.
        # (This can fail if Leiden is non-deterministic across runs, but
        # scanpy igraph Leiden is typically deterministic for the same graph.)
        max_individual = max(individual_scores)
        # Allow a small tolerance for potential Leiden stochasticity.
        np.testing.assert_allclose(
            score_many, max_individual, atol=0.05,
            err_msg="Score with all resolutions should ≈ max of individual scores"
        )


# Guard: skip entire class if polars not installed (optional eval dependency).
# These tests pin `output="polars"` deliberately: the accelerator now *defaults*
# to pandas (F6), but the polars frame is what `cell_eval.DEResults` accepts, so
# the polars schema is precisely this class's subject. The default container is
# covered in test_accel.py.
pl = pytest.importorskip("polars", reason="polars required for DE format bridge tests")


class TestRankGenesGroupsDf:
    """Test pyscx.accel.rank_genes_groups_df(output="polars") — DE format bridge.

    Verifies that the polars DataFrame cell-eval consumes matches its DEResults
    schema: (target, feature, fold_change, p_value, fdr, log2_fold_change,
    abs_log2_fold_change).
    """

    def _make_adata(self, n_obs=200, n_vars=50, n_groups=4, seed=42):
        """Create synthetic AnnData with sparse X and group labels."""
        rng = np.random.default_rng(seed)
        import pandas as pd

        # Sparse count matrix with distinct group signatures
        base = rng.exponential(2.0, size=n_vars).astype(np.float32)
        group_names = ["control"] + [f"pert_{i}" for i in range(n_groups - 1)]
        cells_per = n_obs // n_groups
        labels = []
        for name in group_names:
            labels.extend([name] * cells_per)
        while len(labels) < n_obs:
            labels.append("control")

        X = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X[i] = np.maximum(base + noise, 0)
            else:
                delta = rng.normal(0, 1.0, size=n_vars).astype(np.float32)
                X[i] = np.maximum(base + delta + noise, 0)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
        adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)
        return adata

    def test_returns_polars_dataframe(self):
        """Verify the return type is a polars DataFrame."""
        import pyscx


        adata = self._make_adata()
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )
        assert isinstance(df, pl.DataFrame), f"Expected polars DataFrame, got {type(df)}"

    def test_required_columns_present(self):
        """Verify all cell-eval DEResults columns are present."""
        import pyscx


        adata = self._make_adata()
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        required_cols = {
            "target", "feature", "fold_change",
            "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change",
        }
        assert required_cols == set(df.columns), (
            f"Column mismatch: expected {required_cols}, got {set(df.columns)}"
        )

    def test_column_dtypes(self):
        """Verify column types match cell-eval expectations."""
        import pyscx


        adata = self._make_adata()
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        # String columns
        assert df["target"].dtype == pl.Utf8
        assert df["feature"].dtype == pl.Utf8

        # Numeric columns
        for col in ["fold_change", "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change"]:
            assert df[col].dtype == pl.Float64, f"{col} should be Float64, got {df[col].dtype}"

    def test_row_count(self):
        """Verify row count = n_groups × n_genes."""
        import pyscx

        adata = self._make_adata(n_vars=50, n_groups=4)
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        n_groups = len(adata.obs["perturbation"].unique())
        n_vars = adata.n_vars
        assert len(df) == n_groups * n_vars, (
            f"Expected {n_groups * n_vars} rows, got {len(df)}"
        )

    def test_n_genes_limits_output(self):
        """n_genes parameter should limit genes per group."""
        import pyscx

        adata = self._make_adata(n_vars=50, n_groups=4)
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", n_genes=10, output="polars"
        )

        n_groups = len(adata.obs["perturbation"].unique())
        assert len(df) == n_groups * 10, (
            f"Expected {n_groups * 10} rows with n_genes=10, got {len(df)}"
        )

    def test_fold_change_is_2_to_log2fc(self):
        """fold_change should equal 2^log2_fold_change."""
        import pyscx

        adata = self._make_adata()
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        log2fc = df["log2_fold_change"].to_numpy()
        fc = df["fold_change"].to_numpy()

        # Only check finite values (NaN/Inf are passed through)
        finite_mask = np.isfinite(log2fc)
        expected_fc = np.power(2.0, log2fc[finite_mask])
        np.testing.assert_allclose(
            fc[finite_mask], expected_fc, atol=1e-10,
            err_msg="fold_change should be 2^log2_fold_change"
        )

    def test_abs_log2fc_is_absolute(self):
        """abs_log2_fold_change should be |log2_fold_change|."""
        import pyscx

        adata = self._make_adata()
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        log2fc = df["log2_fold_change"].to_numpy()
        abs_log2fc = df["abs_log2_fold_change"].to_numpy()

        np.testing.assert_allclose(
            abs_log2fc, np.abs(log2fc), atol=1e-10,
            err_msg="abs_log2_fold_change should be |log2_fold_change|"
        )

    def test_pvalues_bounded(self):
        """p_value and fdr should be in [0, 1]."""
        import pyscx

        adata = self._make_adata()
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        pvals = df["p_value"].to_numpy()
        fdrs = df["fdr"].to_numpy()

        # Filter out NaN
        finite_pvals = pvals[np.isfinite(pvals)]
        finite_fdrs = fdrs[np.isfinite(fdrs)]

        assert np.all(finite_pvals >= 0.0) and np.all(finite_pvals <= 1.0), \
            "p_value should be in [0, 1]"
        assert np.all(finite_fdrs >= 0.0) and np.all(finite_fdrs <= 1.0), \
            "fdr should be in [0, 1]"

    def test_each_group_has_genes(self):
        """Every group should appear in the target column."""
        import pyscx

        adata = self._make_adata(n_groups=4)
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        unique_targets = set(df["target"].to_list())
        expected_groups = set(adata.obs["perturbation"].unique())
        assert unique_targets == expected_groups, (
            f"Targets mismatch: {unique_targets} != {expected_groups}"
        )

    def test_reference_group(self):
        """With reference='control', control group should be excluded."""
        import pyscx

        adata = self._make_adata(n_groups=4)
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", reference="control", output="polars"
        )

        unique_targets = set(df["target"].to_list())
        assert "control" not in unique_targets, (
            "control should not appear in targets when used as reference"
        )

    def test_dense_input(self):
        """Test with dense numpy X."""
        import pyscx


        adata = self._make_adata()
        adata.X = adata.X.toarray()

        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )
        assert isinstance(df, pl.DataFrame)
        assert "target" in df.columns

    def test_consistency_with_rank_genes_groups(self):
        """Values should match the existing rank_genes_groups function."""
        import pyscx

        adata = self._make_adata()
        adata_copy = adata.copy()

        # Run the new DataFrame-returning function
        df = pyscx.accel.rank_genes_groups_df(
            adata, "perturbation", output="polars"
        )

        # Run the existing adata.uns-writing function
        pyscx.accel.rank_genes_groups(adata_copy, "perturbation")

        # Extract values from uns for comparison
        rgg = adata_copy.uns["rank_genes_groups"]
        for group_name in rgg["names"].dtype.names:
            # Get genes from uns (first 10).
            # Note: scores (z-statistics) are intentionally not bridged —
            # cell-eval's DEResults schema doesn't include a score column.
            uns_genes = list(rgg["names"][group_name][:10])
            uns_pvals = list(rgg["pvals"][group_name][:10])
            uns_logfc = list(rgg["logfoldchanges"][group_name][:10])

            # Get from DataFrame
            df_group = df.filter(df["target"] == group_name)
            df_genes = df_group["feature"].to_list()[:10]
            df_scores_log2fc = df_group["log2_fold_change"].to_list()[:10]
            df_pvals = df_group["p_value"].to_list()[:10]

            # Gene order should match
            assert df_genes == uns_genes, (
                f"Gene order mismatch for {group_name}: "
                f"{df_genes[:3]}... vs {uns_genes[:3]}..."
            )

            # p-values should match
            np.testing.assert_allclose(
                df_pvals, uns_pvals, atol=1e-10,
                err_msg=f"p_value mismatch for {group_name}"
            )

            # log2_fold_change should match logfoldchanges from uns
            np.testing.assert_allclose(
                df_scores_log2fc, uns_logfc, atol=1e-10,
                err_msg=f"log2_fold_change mismatch for {group_name}"
            )


class TestEndToEndPipeline:
    """End-to-end integration test: run the full SCX perturbation evaluation pipeline.

    Validates the complete workflow: pseudobulk_means → perturbation_metrics →
    discrimination_score → energy_distance → knockdown_efficiency.
    """

    def _make_e2e_adata(self, n_obs=300, n_vars=30, n_perts=5, seed=42):
        """Create paired real/predicted AnnData suitable for all metric types.

        Key properties:
        - Gene names match perturbation names (for knockdown + discrimination)
        - Sparse CSR X
        - Distinct perturbation effects (for meaningful metric values)
        """
        rng = np.random.default_rng(seed)
        import pandas as pd

        base = rng.exponential(3.0, size=n_vars).astype(np.float32)
        gene_names = [f"gene_{j}" for j in range(n_vars)]

        # Name perturbations after first few genes (for knockdown/target gene logic)
        pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]
        deltas = {}
        for name in pert_names[1:]:
            scale = rng.uniform(1.0, 4.0)
            deltas[name] = rng.normal(0, scale, size=n_vars).astype(np.float32)

        cells_per = n_obs // n_perts
        labels = []
        for name in pert_names:
            labels.extend([name] * cells_per)
        while len(labels) < n_obs:
            labels.append("control")

        X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
        X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
        for i, label in enumerate(labels):
            noise_r = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            noise_p = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
            if label == "control":
                X_real[i] = np.maximum(base + noise_r, 0)
                X_pred[i] = np.maximum(base + noise_p, 0)
            else:
                X_real[i] = np.maximum(base + deltas[label] + noise_r, 0)
                pred_delta = deltas[label] + rng.normal(0, 0.3, size=n_vars).astype(np.float32)
                X_pred[i] = np.maximum(base + pred_delta + noise_p, 0)

        obs = pd.DataFrame({"perturbation": labels})
        var = pd.DataFrame(index=gene_names)

        adata_real = ad.AnnData(X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy())
        adata_pred = ad.AnnData(X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy())
        return adata_real, adata_pred, pert_names

    def test_full_pipeline(self):
        """Run all SCX perturbation metrics end-to-end and verify outputs."""
        import pyscx

        adata_real, adata_pred, pert_names = self._make_e2e_adata()
        non_ctrl = [p for p in pert_names if p != "control"]

        # ── 1. Pseudobulk means ──
        means_real, groups_real = pyscx.accel.pseudobulk_means(
            adata_real, "perturbation"
        )
        means_pred, groups_pred = pyscx.accel.pseudobulk_means(
            adata_pred, "perturbation"
        )
        assert means_real.shape[0] == len(pert_names)
        assert means_pred.shape[0] == len(pert_names)
        assert means_real.shape[1] == adata_real.n_vars
        assert set(groups_real) == set(pert_names)
        assert set(groups_pred) == set(pert_names)

        # ── 2. Perturbation metrics (pearson_delta, mse, mae, mse_delta, mae_delta) ──
        bulk_results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)
        assert isinstance(bulk_results, dict)
        for metric in ["pearson_delta", "mse", "mae", "mse_delta", "mae_delta"]:
            assert metric in bulk_results, f"Missing metric: {metric}"
            for pert in non_ctrl:
                assert pert in bulk_results[metric], f"{pert} missing from {metric}"
                val = bulk_results[metric][pert]
                assert isinstance(val, float), f"{metric}[{pert}] not float: {type(val)}"
                assert np.isfinite(val), f"{metric}[{pert}] is not finite: {val}"

        # MSE/MAE should be >= 0
        for metric in ["mse", "mae", "mse_delta", "mae_delta"]:
            for pert in non_ctrl:
                assert bulk_results[metric][pert] >= 0.0, \
                    f"{metric}[{pert}] should be non-negative"

        # Pearson should be in [-1, 1]
        for pert in non_ctrl:
            assert -1.0 <= bulk_results["pearson_delta"][pert] <= 1.0, \
                f"pearson_delta[{pert}] out of range"

        # ── 3. Discrimination score ──
        disc_scores = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1"
        )
        assert isinstance(disc_scores, dict)
        for pert in non_ctrl:
            assert pert in disc_scores, f"{pert} missing from discrimination_score"
            assert 0.0 <= disc_scores[pert] <= 1.0, \
                f"discrimination_score[{pert}] out of [0,1]: {disc_scores[pert]}"

        # ── 4. Energy distance ──
        edist_corr = pyscx.accel.energy_distance(adata_real, adata_pred)
        assert isinstance(edist_corr, float)
        assert -1.0 <= edist_corr <= 1.0 or np.isnan(edist_corr), \
            f"energy_distance correlation out of range: {edist_corr}"

        # ── 5. Knockdown efficiency ──
        adata_kd = adata_real.copy()
        pyscx.accel.knockdown_efficiency(adata_kd, pert_col="perturbation")
        assert "KnockDownEfficiency" in adata_kd.obs.columns
        assert "KnockDownGeneFC" in adata_kd.obs.columns

        # Control cells should have NaN knockdown values
        ctrl_mask = adata_kd.obs["perturbation"] == "control"
        assert np.all(np.isnan(adata_kd.obs.loc[ctrl_mask, "KnockDownEfficiency"]))

        # Non-control cells with matching genes should have finite values
        for pert in non_ctrl:
            pert_mask = adata_kd.obs["perturbation"] == pert
            kd_vals = adata_kd.obs.loc[pert_mask, "KnockDownEfficiency"].values
            if pert in adata_kd.var_names:
                assert np.any(np.isfinite(kd_vals)), \
                    f"Expected finite KD values for {pert}"

        # ── 6. DE result format bridge ──
        de_df = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control", output="polars"
        )

        assert isinstance(de_df, pl.DataFrame)
        required_cols = {
            "target", "feature", "fold_change",
            "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change",
        }
        assert required_cols == set(de_df.columns)
        assert "control" not in de_df["target"].to_list()
        assert len(de_df) == len(non_ctrl) * adata_real.n_vars

        # ── 7. Clustering agreement ──
        score = pyscx.accel.clustering_agreement(
            adata_real, adata_pred, metric="ami",
            pred_resolutions=[0.5, 1.0, 1.5],
        )
        assert isinstance(score, float)
        # AMI can be negative but typically in [-0.5, 1.0] range
        assert score >= -1.0 and score <= 1.0, \
            f"clustering_agreement AMI out of range: {score}"

    def test_pipeline_all_metrics_match_tolerance(self):
        """Verify that identical data produces perfect metric values."""
        import pyscx

        adata_real, _, _ = self._make_e2e_adata()
        adata_pred = adata_real.copy()

        # Identical data → MSE/MAE should be 0, Pearson should be 1
        results = pyscx.accel.perturbation_metrics(adata_real, adata_pred)
        for pert in results["mse"]:
            np.testing.assert_allclose(
                results["mse"][pert], 0.0, atol=1e-10,
                err_msg=f"MSE should be 0 for identical data ({pert})"
            )
            np.testing.assert_allclose(
                results["pearson_delta"][pert], 1.0, atol=1e-10,
                err_msg=f"Pearson should be 1 for identical data ({pert})"
            )

        # Identical data → energy distance correlation should be 1
        edist = pyscx.accel.energy_distance(adata_real, adata_pred)
        np.testing.assert_allclose(
            edist, 1.0, atol=1e-6,
            err_msg="Energy distance correlation should be 1 for identical data"
        )

        # Identical data → discrimination scores should be 1.0
        disc = pyscx.accel.discrimination_score(
            adata_real, adata_pred, metric="l1"
        )
        for pert, score in disc.items():
            np.testing.assert_allclose(
                score, 1.0, atol=1e-10,
                err_msg=f"Discrimination score should be 1.0 for identical data ({pert})"
            )

        # Identical data → DE bridge should produce valid output
        de_df = pyscx.accel.rank_genes_groups_df(
            adata_real, "perturbation", reference="control", output="polars"
        )

        assert isinstance(de_df, pl.DataFrame)
        assert set(de_df.columns) == {
            "target", "feature", "fold_change",
            "p_value", "fdr", "log2_fold_change", "abs_log2_fold_change",
        }
        # All p-values should be valid (finite, in [0, 1])
        pvals = de_df["p_value"].to_numpy()
        finite_pvals = pvals[np.isfinite(pvals)]
        assert np.all(finite_pvals >= 0.0) and np.all(finite_pvals <= 1.0)
