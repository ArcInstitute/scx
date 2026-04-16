"""Integration tests for eval_metrics: pseudobulk_means and perturbation_metrics.

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
