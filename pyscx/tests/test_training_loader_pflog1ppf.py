"""PFlog1pPF training-loader mode (scx-loader Phase 4b).

Smoke + correctness tests for the `pflog1ppf` / `pflog1ppf_c` kwargs on
`TrainingDataset`. The exact transform is computed per dense batch row; the
critical property is that depth and the centering denominator D are over the
FULL transcriptome, NOT the HVG-projected panel.
"""

import numpy as np
import pyscx
import pytest


@pytest.fixture
def scx_path(tmp_path, synthetic_adata):
    path = str(tmp_path / "pflog1ppf_loader.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


def _dense_X(adata):
    x = adata.X
    return np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)


def reference_dense(X, c=1.0):
    """Spec §11.1 exact reference over the full transcriptome."""
    depth = X.sum(axis=1, keepdims=True)
    L = np.log(X / depth + c)
    return L - L.mean(axis=1, keepdims=True)


class TestLoaderPFlog1pPF:
    def test_matches_full_reference_no_projection(self, scx_path, synthetic_adata):
        full = reference_dense(_dense_X(synthetic_adata), 1.0)
        ds = pyscx.TrainingDataset(scx_path, batch_size=100, pflog1ppf=True)
        seen = 0
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            idx = batch["cell_indices"]
            for row_pos, cell in enumerate(idx):
                np.testing.assert_allclose(
                    X[row_pos], full[cell], rtol=1e-4, atol=1e-4
                )
                # Centering property: each row sums to ~0.
                assert abs(X[row_pos].sum()) < 1e-4
                seen += 1
        assert seen == synthetic_adata.n_obs

    def test_custom_c(self, scx_path, synthetic_adata):
        c = 0.5
        full = reference_dense(_dense_X(synthetic_adata), c)
        ds = pyscx.TrainingDataset(scx_path, batch_size=100, pflog1ppf=True, pflog1ppf_c=c)
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            for row_pos, cell in enumerate(batch["cell_indices"]):
                np.testing.assert_allclose(X[row_pos], full[cell], rtol=1e-4, atol=1e-4)

    def test_projection_uses_full_transcriptome(self, scx_path, synthetic_adata):
        # 🔴 The load-bearing property: projected PFlog1pPF == the corresponding
        # columns of FULL-transcriptome PFlog1pPF (depth & D over all 50 genes),
        # NOT panel-local PFlog1pPF over the 5-gene subset.
        hvg = [0, 5, 10, 20, 40]
        full = reference_dense(_dense_X(synthetic_adata), 1.0)
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=100, hvg_indices=hvg, pflog1ppf=True
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
        # Guard that the property above is non-trivial: panel-local PFlog1pPF
        # (depth & D over the 5 panel genes only) genuinely differs.
        hvg = [0, 5, 10, 20, 40]
        Xfull = _dense_X(synthetic_adata)
        Xpanel = Xfull[:, hvg]
        panel_depth = Xpanel.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            # NaN for empty-panel cells (depth 0); those are filtered below.
            panel_local = reference_dense(Xpanel, 1.0)
        ds = pyscx.TrainingDataset(
            scx_path, batch_size=100, hvg_indices=hvg, pflog1ppf=True
        )
        max_diff = 0.0
        for batch in ds:
            X = np.asarray(batch["X"], dtype=np.float64)
            for row_pos, cell in enumerate(batch["cell_indices"]):
                if panel_depth[cell] > 0:  # panel-local undefined for empty-panel cells
                    max_diff = max(
                        max_diff, float(np.abs(X[row_pos] - panel_local[cell]).max())
                    )
        assert max_diff > 1e-3, "full-transcriptome and panel-local must differ"

    def test_invalid_c_raises(self, scx_path):
        with pytest.raises(RuntimeError, match="pflog1ppf_c"):
            pyscx.TrainingDataset(scx_path, pflog1ppf=True, pflog1ppf_c=0.0)

    def test_index_plan_dataset_pflog1ppf(self, scx_path, synthetic_adata):
        # Plan-driven (paired) loader honors pflog1ppf=True: both the perturbed
        # and control rows of each pair match the full-transcriptome reference.
        full = reference_dense(_dense_X(synthetic_adata), 1.0)
        ds = pyscx.IndexPlanDataset(scx_path, pflog1ppf=True)
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
