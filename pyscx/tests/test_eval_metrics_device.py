"""Phase-0 device-scaffolding tests for the perturbation-evaluation metrics.

Covers the ``device=`` API surface added in Phase 0 of
``CELL-EVAL-SCX-GPU-ACC.md``: the eval metrics accept ``device=`` and stamp a
route on ``uns["scx_accel"]``, but have no GPU kernel yet, so every device
resolves to the CPU path.

Contract under test:
- ``device="cpu"`` and ``device="auto"`` (default) are byte-identical to the
  pre-Phase-0 behavior on a CPU host (no GPU kernel exists, so both run CPU).
- Each call stamps ``uns["scx_accel"][<op>]`` with a ``cpu_*`` route.
- ``device="gpu"`` errors loudly when no usable CUDA GPU is present (no-gpu
  build or no CUDA device); on a GPU host it runs CPU + warns (no kernel yet).
- A malformed device string is a clear error.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import pyscx


def _make_paired_adata(n_obs=200, n_vars=30, n_perts=4, seed=42):
    """Small paired real/pred AnnData with a known perturbation structure."""
    rng = np.random.default_rng(seed)
    base = rng.exponential(2.0, size=n_vars).astype(np.float32)

    pert_names = ["control"] + [f"drug_{i}" for i in range(n_perts - 1)]
    deltas = {
        name: rng.normal(0, 1, size=n_vars).astype(np.float32)
        for name in pert_names[1:]
    }

    cells_per_pert = n_obs // n_perts
    labels = []
    for name in pert_names:
        labels.extend([name] * cells_per_pert)
    while len(labels) < n_obs:
        labels.append("control")

    import pandas as pd

    X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
    X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
    for i, label in enumerate(labels):
        noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
        if label == "control":
            X_real[i] = np.maximum(base + noise, 0)
            X_pred[i] = np.maximum(base + noise, 0)
        else:
            X_real[i] = np.maximum(base + deltas[label] + noise, 0)
            pred_delta = deltas[label] + rng.normal(0, 0.3, size=n_vars).astype(
                np.float32
            )
            X_pred[i] = np.maximum(base + pred_delta + noise, 0)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
    adata_real = ad.AnnData(X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy())
    adata_pred = ad.AnnData(X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy())
    return adata_real, adata_pred


def _route(adata, op):
    return adata.uns["scx_accel"][op]["route"]


class TestEvalMetricsDeviceScaffolding:
    def test_perturbation_metrics_cpu_matches_default(self):
        """device="cpu" matches the default (auto). Bit-identical on a CPU host;
        on a GPU host the default runs GPU (f64 pseudobulk), so allow a tiny atol."""
        real, pred = _make_paired_adata()
        default = pyscx.accel.perturbation_metrics(real, pred)
        cpu = pyscx.accel.perturbation_metrics(real, pred, device="cpu")
        atol = 1e-6 if pyscx.accel.gpu_available() else 0.0
        assert default.keys() == cpu.keys()
        for metric, per_pert in default.items():
            for name, value in per_pert.items():
                assert abs(cpu[metric][name] - value) <= atol

    def test_discrimination_score_cpu_matches_default(self):
        # discrimination_score has no GPU kernel yet (deferred), so the default
        # (auto) runs CPU on every host → bit-identical to device="cpu".
        real, pred = _make_paired_adata()
        default = pyscx.accel.discrimination_score(real, pred)
        cpu = pyscx.accel.discrimination_score(real, pred, device="cpu")
        assert default == cpu

    def test_energy_distance_cpu_matches_default(self):
        """device="cpu" matches the default (auto). Bit-identical on a CPU host;
        on a GPU host the default runs the f32-gemm GPU kernel, so parity is at
        the 1e-4 bar (energy_distance is a correlation)."""
        real, pred = _make_paired_adata()
        default = pyscx.accel.energy_distance(real, pred)
        cpu = pyscx.accel.energy_distance(real, pred, device="cpu")
        atol = 1e-4 if pyscx.accel.gpu_available() else 0.0
        assert (abs(default - cpu) <= atol) or (np.isnan(default) and np.isnan(cpu))

    def test_route_stamped(self):
        """Each metric stamps a route in uns["scx_accel"] on the pred under the
        default (auto) device. perturbation_metrics (Phase 2) and energy_distance
        (Phase 3, default euclidean → gemm) have GPU kernels, so on a GPU host
        auto→gpu; discrimination_score is still CPU-only (deferred), so cpu_*."""
        real, pred = _make_paired_adata()
        pyscx.accel.perturbation_metrics(real, pred)
        pyscx.accel.energy_distance(real, pred)
        pyscx.accel.discrimination_score(real, pred)
        gpu_expected = "gpu" if pyscx.accel.gpu_available() else "cpu"
        for op in ("perturbation_metrics", "energy_distance"):
            assert _route(pred, op).startswith(gpu_expected), op
        assert _route(pred, "discrimination_score").startswith("cpu")

    def test_pseudobulk_means_route_stamped_single_input(self):
        real, _ = _make_paired_adata()
        pyscx.accel.pseudobulk_means(real, "perturbation")
        # Pseudobulk_means has a GPU kernel, so auto→gpu on a GPU host.
        expected = "gpu" if pyscx.accel.gpu_available() else "cpu"
        assert _route(real, "pseudobulk_means").startswith(expected)

    def test_inner_pseudobulk_does_not_shadow_outer_route(self):
        """perturbation_metrics stamps its own op, not the inner pseudobulk_means.

        The internal means computation goes through the non-scaffolded impl, so
        the only route stamped by perturbation_metrics is "perturbation_metrics".
        """
        real, pred = _make_paired_adata()
        pyscx.accel.perturbation_metrics(real, pred)
        assert "perturbation_metrics" in pred.uns["scx_accel"]
        assert "pseudobulk_means" not in pred.uns["scx_accel"]

    @pytest.mark.skipif(
        pyscx.accel.gpu_available(),
        reason="requires a build/host without a usable CUDA GPU",
    )
    def test_gpu_raises_without_gpu(self):
        """device="gpu" errors loudly when no usable CUDA GPU is present."""
        real, pred = _make_paired_adata()
        with pytest.raises((RuntimeError, ValueError)):
            pyscx.accel.perturbation_metrics(real, pred, device="gpu")

    @pytest.mark.skipif(
        not pyscx.accel.gpu_available(),
        reason="requires a GPU host to exercise the no-kernel-yet CPU fallback",
    )
    def test_gpu_on_host_effect_metric_runs_cpu_and_warns(self):
        """On a GPU host, an eval metric without a GPU kernel yet
        (discrimination_score — deferred) runs CPU and emits the one-shot
        fallback warning; route cpu_*."""
        real, pred = _make_paired_adata()
        with pytest.warns(UserWarning):
            pyscx.accel.discrimination_score(real, pred, device="gpu")
        assert _route(pred, "discrimination_score").startswith("cpu")

    @pytest.mark.skipif(
        not pyscx.accel.gpu_available(),
        reason="requires a GPU host to run the perturbation_metrics GPU kernel",
    )
    def test_gpu_on_host_perturbation_metrics_runs_gpu(self):
        """Phase 2: perturbation_metrics runs on the GPU on a GPU host (route gpu_*).
        (GPU-vs-CPU numeric parity is covered in test_eval_metrics_gpu_parity.py.)"""
        real, pred = _make_paired_adata()
        pyscx.accel.perturbation_metrics(real, pred, device="gpu")
        assert _route(pred, "perturbation_metrics").startswith("gpu")

    def test_invalid_device_string_raises(self):
        real, pred = _make_paired_adata()
        with pytest.raises((ValueError, RuntimeError)):
            pyscx.accel.perturbation_metrics(real, pred, device="tpu")


class TestEvalMetricsRouteStampOnFailure:
    """`scaffold_device_route` stamps before the compute — so it must roll back.

    These three ops share one scaffold, so they share one failure mode: a raise
    after the stamp used to leave `uns["scx_accel"][op]` behind, asserting a
    route for a result that was never produced. Both triggers below reject an
    unknown `metric`, which happens after the scaffold has already stamped.
    """

    def test_failing_discrimination_score_leaves_no_stamp(self):
        real, pred = _make_paired_adata()
        with pytest.raises(ValueError):
            pyscx.accel.discrimination_score(real, pred, metric="bogus")
        assert "discrimination_score" not in pred.uns.get("scx_accel", {})

    def test_failing_clustering_agreement_leaves_no_stamp(self):
        real, pred = _make_paired_adata()
        with pytest.raises(ValueError):
            pyscx.accel.clustering_agreement(real, pred, metric="bogus")
        assert "clustering_agreement" not in pred.uns.get("scx_accel", {})

    def test_a_failure_does_not_disturb_an_earlier_ops_stamp(self):
        real, pred = _make_paired_adata()
        pyscx.accel.perturbation_metrics(real, pred)
        with pytest.raises(ValueError):
            pyscx.accel.discrimination_score(real, pred, metric="bogus")
        assert "perturbation_metrics" in pred.uns["scx_accel"]
