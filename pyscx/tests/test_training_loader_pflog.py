"""PFlog (v4) training-loader mode.

Smoke + correctness tests for the `pflog` / `pflog_alpha` kwargs on
`TrainingDataset` / `IndexPlanDataset`. v4 shifts **raw counts** by the
matrix-wide Anscombe pseudocount `1/(4α)` (no per-cell depth); the critical
property is that the centering denominator D is over the FULL transcriptome,
NOT the HVG-projected panel. `pflog_alpha=None` estimates α once at loader
construction; a float pins it.
"""

import numpy as np
import pyscx
import pytest

# Pinned α for deterministic reference comparison (α = 0.5 → pseudocount 0.5).
ALPHA = 0.5


@pytest.fixture
def scx_path(tmp_path, synthetic_adata):
    path = str(tmp_path / "pflog_loader.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


def _dense_X(adata):
    x = adata.X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def reference_dense(X, alpha):
    """v4 exact reference over the full transcriptome:
    z_ij = log1p(4α·x_ij) − mean_j log1p(4α·x_ik)."""
    X = np.asarray(X, dtype=np.float64)
    L = np.log1p(4.0 * alpha * X)
    return L - L.mean(axis=1, keepdims=True)


class TestLoaderPFlog:
    def test_matches_full_reference_no_projection(self, scx_path, synthetic_adata):
        full = reference_dense(_dense_X(synthetic_adata), ALPHA)
        ds = pyscx.TrainingDataset(scx_path, batch_size=100, pflog=True, pflog_alpha=ALPHA)
        seen = 0
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            idx = batch["cell_indices"]
            for row_pos, cell in enumerate(idx):
                np.testing.assert_allclose(X[row_pos], full[cell], rtol=1e-4, atol=1e-4)
                # Centering property: each row sums to ~0.
                assert abs(X[row_pos].sum()) < 1e-4
                seen += 1
        assert seen == synthetic_adata.n_obs

    def test_various_alpha(self, scx_path, synthetic_adata):
        alpha = 1.0
        full = reference_dense(_dense_X(synthetic_adata), alpha)
        ds = pyscx.TrainingDataset(scx_path, batch_size=100, pflog=True, pflog_alpha=alpha)
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            for row_pos, cell in enumerate(batch["cell_indices"]):
                np.testing.assert_allclose(X[row_pos], full[cell], rtol=1e-4, atol=1e-4)

    def test_alpha_none_auto_estimates(self, scx_path, synthetic_adata):
        # `pflog_alpha=None` estimates α once at construction over the file's raw
        # counts. We don't know the exact α here, but the centering property holds
        # for any α, and the output must be finite with the right shape.
        ds = pyscx.TrainingDataset(scx_path, batch_size=100, pflog=True)
        seen = 0
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            assert X.shape[1] == synthetic_adata.n_vars
            assert np.all(np.isfinite(X))
            for row_pos in range(X.shape[0]):
                assert abs(X[row_pos].sum()) < 1e-4  # centered regardless of α
                seen += 1
        assert seen == synthetic_adata.n_obs

    def test_projection_uses_full_transcriptome(self, scx_path, synthetic_adata):
        # 🔴 The load-bearing property: projected PFlog == the corresponding
        # columns of FULL-transcriptome PFlog (centering D over all 50 genes),
        # NOT panel-local PFlog over the 5-gene subset.
        hvg = [0, 5, 10, 20, 40]
        full = reference_dense(_dense_X(synthetic_adata), ALPHA)
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=100, hvg_indices=hvg, pflog=True, pflog_alpha=ALPHA
        )
        seen = 0
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            assert X.shape[1] == len(hvg)
            for row_pos, cell in enumerate(batch["cell_indices"]):
                np.testing.assert_allclose(
                    X[row_pos], full[cell, hvg], rtol=1e-4, atol=1e-4
                )
                seen += 1
        assert seen == synthetic_adata.n_obs

    def test_projection_differs_from_panel_local(self, scx_path, synthetic_adata):
        # Guard that the property above is non-trivial: panel-local PFlog
        # (centering D over the 5 panel genes only) genuinely differs — the
        # per-element log1p(4α·x) matches but the baseline does not.
        hvg = [0, 5, 10, 20, 40]
        Xfull = _dense_X(synthetic_adata)
        Xpanel = Xfull[:, hvg]
        panel_local = reference_dense(Xpanel, ALPHA)
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=100, hvg_indices=hvg, pflog=True, pflog_alpha=ALPHA
        )
        max_diff = 0.0
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            for row_pos, cell in enumerate(batch["cell_indices"]):
                max_diff = max(
                    max_diff, float(np.abs(X[row_pos] - panel_local[cell]).max())
                )
        assert max_diff > 1e-3, "full-transcriptome and panel-local must differ"

    def test_invalid_alpha_raises(self, scx_path):
        with pytest.raises(RuntimeError, match="pflog_alpha"):
            pyscx.TrainingDataset(scx_path, pflog=True, pflog_alpha=0.0)

    def test_index_plan_dataset_pflog(self, scx_path, synthetic_adata):
        # Plan-driven (paired) loader honors pflog=True: both the perturbed and
        # control rows of each pair match the full-transcriptome reference.
        full = reference_dense(_dense_X(synthetic_adata), ALPHA)
        ds = pyscx.IndexPlanDataset(scx_path, pflog=True, pflog_alpha=ALPHA)
        plans = [[(0, 1), (2, 3)], [(4, 5), (6, 7)]]
        seen = 0
        for batch in ds.iter_with_plans(iter(plans)):
            Xp = np.asarray(batch["X"], dtype=np.float64)
            Xc = np.asarray(batch["X_paired"], dtype=np.float64)
            for row, (p, c) in enumerate((int(a), int(b)) for a, b in batch["pairs"]):
                np.testing.assert_allclose(Xp[row], full[p], rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(Xc[row], full[c], rtol=1e-4, atol=1e-4)
                seen += 1
        assert seen == 4
