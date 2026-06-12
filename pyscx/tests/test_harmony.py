"""Tests for pyscx.accel.harmony_integrate() — batch integration via Harmony2."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def harmony_adata(synthetic_adata):
    """Provide an AnnData with a real PCA embedding + batch column.

    The synthetic fixture's obsm["X_pca"] is random Gaussian (good enough
    for a smoke test). obs["batch"] is categorical with 3 levels (A, B, C).
    """
    import pyscx

    adata = synthetic_adata.copy()
    # Run PCA to get a representative embedding; overwrite the random one.
    pyscx.accel.pca(adata, n_comps=10, random_state=0)
    return adata


class TestHarmonyBasic:
    """Shape / in-place / metadata contracts."""

    def test_runs_and_shape_matches(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        input_shape = adata.obsm["X_pca"].shape
        pyscx.accel.harmony_integrate(
            adata, "batch", max_iter=2, random_state=0
        )
        assert adata.obsm["X_pca"].shape == input_shape

    def test_overwrites_basis_in_place(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        original = adata.obsm["X_pca"].copy()
        pyscx.accel.harmony_integrate(
            adata, "batch", max_iter=2, random_state=0
        )
        # Default behaviour: write back to "X_pca" (scanpy-compatible).
        assert "X_pca" in adata.obsm
        # Values should differ after correction.
        assert not np.allclose(adata.obsm["X_pca"], original)

    def test_adjusted_basis_preserves_original(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        original = adata.obsm["X_pca"].copy()
        pyscx.accel.harmony_integrate(
            adata,
            "batch",
            adjusted_basis="X_pca_harmony",
            max_iter=2,
            random_state=0,
        )
        assert "X_pca_harmony" in adata.obsm
        assert adata.obsm["X_pca_harmony"].shape == original.shape
        # Original preserved.
        np.testing.assert_array_equal(adata.obsm["X_pca"], original)

    def test_output_dtype_is_float32(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        pyscx.accel.harmony_integrate(
            adata,
            "batch",
            adjusted_basis="X_pca_harmony",
            max_iter=2,
            random_state=0,
        )
        assert adata.obsm["X_pca_harmony"].dtype == np.float32

    def test_uns_metadata(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        pyscx.accel.harmony_integrate(
            adata, "batch", max_iter=2, random_state=0
        )
        assert "harmony" in adata.uns
        info = adata.uns["harmony"]
        for required in (
            "params",
            "converged",
            "n_iterations",
            "objective_harmony",
            "backend",
        ):
            assert required in info, f"missing key: {required}"
        assert info["params"]["key"] == "batch"
        assert isinstance(info["n_iterations"], int)
        assert info["n_iterations"] >= 1
        assert info["backend"] == "scx-accel-cpu"


class TestHarmonyMultiKey:
    """Multi-covariate path (list[str] key)."""

    def test_multi_key_runs(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        # Add a second categorical covariate.
        rng = np.random.default_rng(7)
        adata.obs["technology"] = pd.Categorical(
            rng.choice(["10x", "smartseq"], size=adata.n_obs)
        )
        pyscx.accel.harmony_integrate(
            adata, ["batch", "technology"], max_iter=2, random_state=0
        )
        assert adata.obsm["X_pca"].shape == (adata.n_obs, 10)
        assert adata.uns["harmony"]["params"]["key"] == ["batch", "technology"]


class TestHarmonyDevice:
    """Device routing."""

    def test_device_cpu_explicit(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        pyscx.accel.harmony_integrate(
            adata, "batch", device="cpu", max_iter=2, random_state=0
        )
        assert adata.uns["harmony"]["backend"] == "scx-accel-cpu"

    def test_stamps_scx_accel_route(self, harmony_adata):
        """F15: harmony_integrate must record the same scx_accel route envelope
        as PCA / kNN / UMAP so GPU-vs-CPU dispatch is verifiable, not None."""
        import pyscx

        adata = harmony_adata
        pyscx.accel.harmony_integrate(
            adata, "batch", device="cpu", max_iter=2, random_state=0
        )
        assert "scx_accel" in adata.uns
        assert "harmony_integrate" in adata.uns["scx_accel"]
        route = adata.uns["scx_accel"]["harmony_integrate"]
        # device="cpu" → CpuDense path, recorded as a user-forced CPU fallback.
        assert route["route"] == "cpu_dense"
        assert route["fallback_reason"] == "user_forced_cpu"


class TestHarmonyErrors:
    """Informative error surface for common misuse."""

    def test_invalid_key_raises(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        with pytest.raises(Exception) as ei:
            pyscx.accel.harmony_integrate(adata, "does_not_exist", max_iter=1)
        msg = str(ei.value)
        assert "does_not_exist" in msg

    def test_nan_embeddings_raise(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        adata.obsm["X_pca"][0, 0] = np.nan
        with pytest.raises(Exception) as ei:
            pyscx.accel.harmony_integrate(adata, "batch", max_iter=1)
        msg = str(ei.value)
        assert "NaN" in msg or "Inf" in msg

    def test_missing_basis_raises(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        with pytest.raises(Exception) as ei:
            pyscx.accel.harmony_integrate(
                adata, "batch", basis="X_not_here", max_iter=1
            )
        assert "X_not_here" in str(ei.value)
