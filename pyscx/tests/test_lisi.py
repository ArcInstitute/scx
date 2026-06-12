"""Tests for pyscx.accel.compute_lisi()."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def lisi_adata(synthetic_adata):
    """Provide an AnnData with obs['batch'] and a PCA embedding."""
    import pyscx

    adata = synthetic_adata.copy()
    pyscx.accel.pca(adata, n_comps=10, random_state=0)
    return adata


class TestLisiBasic:
    def test_returns_array_of_length_n(self, lisi_adata):
        import pyscx

        adata = lisi_adata
        lisi = pyscx.accel.compute_lisi(
            adata, "batch", perplexity=10.0
        )
        assert isinstance(lisi, np.ndarray)
        assert lisi.shape == (adata.n_obs,)
        assert np.all(np.isfinite(lisi))

    def test_values_in_expected_range(self, lisi_adata):
        import pyscx

        adata = lisi_adata
        n_batches = adata.obs["batch"].nunique()
        lisi = pyscx.accel.compute_lisi(
            adata, "batch", perplexity=10.0
        )
        # LISI should live in [1, n_batches] (approximately, with small
        # slack for kernel-weight numerics).
        assert lisi.min() >= 1.0 - 1e-6
        assert lisi.max() <= float(n_batches) + 1e-3

    def test_writes_obs_column(self, lisi_adata):
        import pyscx

        adata = lisi_adata
        pyscx.accel.compute_lisi(adata, "batch", perplexity=10.0)
        assert "lisi_batch" in adata.obs.columns
        # Rayon reduction order causes ~1e-15 drift between runs; the obs
        # column must match the returned array up to that precision.
        np.testing.assert_allclose(
            adata.obs["lisi_batch"].to_numpy(),
            pyscx.accel.compute_lisi(adata, "batch", perplexity=10.0),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_single_batch_gives_lisi_one(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca(adata, n_comps=10, random_state=0)
        adata.obs["only"] = pd.Categorical(["A"] * adata.n_obs)
        lisi = pyscx.accel.compute_lisi(adata, "only", perplexity=10.0)
        np.testing.assert_allclose(lisi, 1.0, atol=1e-6)


class TestLisiApproximateKnn:
    def test_approximate_knn_kwarg_accepted(self, lisi_adata):
        """F14: the `approximate_knn` lever the perf warning recommends is
        exposed on the Python binding (not just the Rust core)."""
        import inspect

        import pyscx

        sig = inspect.signature(pyscx.accel.compute_lisi)
        assert "approximate_knn" in sig.parameters

        adata = lisi_adata
        lisi = pyscx.accel.compute_lisi(
            adata, "batch", perplexity=10.0, approximate_knn=True
        )
        assert isinstance(lisi, np.ndarray)
        assert lisi.shape == (adata.n_obs,)
        assert np.all(np.isfinite(lisi))

    def test_approximate_matches_exact_within_drift(self, lisi_adata):
        """HNSW approximate kNN should track the exact sweep closely on a
        small fixture (documented ~0.01–0.05 mean-LISI drift)."""
        import pyscx

        adata = lisi_adata
        exact = pyscx.accel.compute_lisi(
            adata, "batch", perplexity=10.0, approximate_knn=False
        )
        approx = pyscx.accel.compute_lisi(
            adata, "batch", perplexity=10.0, approximate_knn=True
        )
        assert abs(float(exact.mean()) - float(approx.mean())) < 0.25


class TestLisiErrors:
    def test_invalid_key(self, lisi_adata):
        import pyscx

        with pytest.raises(Exception) as ei:
            pyscx.accel.compute_lisi(lisi_adata, "does_not_exist")
        assert "does_not_exist" in str(ei.value)

    def test_bad_perplexity(self, lisi_adata):
        import pyscx

        with pytest.raises(Exception):
            pyscx.accel.compute_lisi(lisi_adata, "batch", perplexity=-1.0)
