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
        # Default writes a new key matching the input shape.
        assert "X_pca_harmony" in adata.obsm
        assert adata.obsm["X_pca_harmony"].shape == input_shape

    def test_default_writes_x_pca_harmony_preserves_input(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        original = adata.obsm["X_pca"].copy()
        pyscx.accel.harmony_integrate(
            adata, "batch", max_iter=2, random_state=0
        )
        # Default behaviour (scanpy-compatible): write the corrected embedding
        # to a new "X_pca_harmony" key and leave "X_pca" untouched.
        assert "X_pca_harmony" in adata.obsm
        np.testing.assert_array_equal(adata.obsm["X_pca"], original)
        # Corrected embedding differs from the input.
        assert not np.allclose(adata.obsm["X_pca_harmony"], original)

    def test_adjusted_basis_in_place_overwrites(self, harmony_adata):
        import pyscx

        adata = harmony_adata
        original = adata.obsm["X_pca"].copy()
        # Explicit in-place: pass adjusted_basis=basis.
        pyscx.accel.harmony_integrate(
            adata, "batch", adjusted_basis="X_pca", max_iter=2, random_state=0
        )
        assert "X_pca" in adata.obsm
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

    def test_graph_replay_is_none_on_cpu(self, harmony_adata):
        """§8.12: the CPU route must report no CUDA-graph decision.

        Scope warning, because it is easy to over-read: this assertion
        **passes without the §8.12 fix** — the pre-fix stamp left every
        optional field None, so `graph_replay` was already None here. It is
        checked and stated rather than implied. What it guards is the Python
        surface: that the key still reaches `uns`, and that CPU never reports
        a `False` (which would claim a capture was attempted and did not
        replay — a different and untrue statement).

        The assertions that actually distinguish fixed from unfixed are in
        Rust: `test_cpu_harmony_reports_no_graph_decision` (the field exists at
        all) and `test_gpu_harmony_reports_graph_replay` (`Some(True)` under
        capture, `Some(False)` under the kill switch), the latter running only
        on the Chimera GPU suite.
        """
        import pyscx

        adata = harmony_adata
        pyscx.accel.harmony_integrate(
            adata, "batch", device="cpu", max_iter=2, random_state=0
        )
        route = adata.uns["scx_accel"]["harmony_integrate"]
        assert "graph_replay" in route
        assert route["graph_replay"] is None


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
