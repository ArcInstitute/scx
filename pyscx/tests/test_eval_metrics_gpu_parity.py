"""GPU-vs-CPU parity for perturbation_metrics.

GPU-accelerates the pseudobulk aggregation behind
`pyscx.accel.perturbation_metrics` (and `pseudobulk_means`) by wrapping the DE
pseudobulk kernels; the five bulk metrics run on the host. GPU must match CPU:

- Sparse gene-space X (backed .scx + in-memory CSR): the GPU f64 sum matches the
  CPU f32→f64 accumulate up to atomic-add ordering → tight tolerance.
- Dense `embed_key`/obsm: the GPU dense kernel is f32, so parity on f64
  embeddings is at the looser f32 bar.

The route is stamped `gpu_*` when the aggregation ran on the GPU.

Skips cleanly when pyscx was built without the `gpu` feature or no CUDA device
is present. Run isolated with `SCX_DISABLE_CUDA_GRAPHS=1` on a GPU node.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with the gpu feature, or no CUDA device)",
)


def _make_paired_adata(n_obs=400, n_vars=60, n_perts=5, seed=0):
    """Paired real/pred AnnData (sparse CSR) with a known perturbation structure."""
    rng = np.random.default_rng(seed)
    base = rng.exponential(2.0, size=n_vars).astype(np.float32)
    pert_names = ["control"] + [f"drug_{i}" for i in range(n_perts - 1)]
    deltas = {n: rng.normal(0, 1, size=n_vars).astype(np.float32) for n in pert_names[1:]}

    labels = [pert_names[i % n_perts] for i in range(n_obs)]
    import pandas as pd

    X_real = np.zeros((n_obs, n_vars), dtype=np.float32)
    X_pred = np.zeros((n_obs, n_vars), dtype=np.float32)
    for i, lab in enumerate(labels):
        noise = rng.normal(0, 0.1, size=n_vars).astype(np.float32)
        if lab == "control":
            X_real[i] = np.maximum(base + noise, 0)
            X_pred[i] = np.maximum(base + noise, 0)
        else:
            X_real[i] = np.maximum(base + deltas[lab] + noise, 0)
            pdelta = deltas[lab] + rng.normal(0, 0.3, size=n_vars).astype(np.float32)
            X_pred[i] = np.maximum(base + pdelta + noise, 0)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)])
    real = ad.AnnData(X=sp.csr_matrix(X_real), obs=obs.copy(), var=var.copy())
    pred = ad.AnnData(X=sp.csr_matrix(X_pred), obs=obs.copy(), var=var.copy())
    return real, pred


def _assert_metrics_close(cpu, gpu, atol):
    assert cpu.keys() == gpu.keys()
    for metric, per_pert in cpu.items():
        assert per_pert.keys() == gpu[metric].keys(), metric
        for pert, val in per_pert.items():
            g = gpu[metric][pert]
            if np.isnan(val):
                assert np.isnan(g), f"{metric}/{pert}: cpu nan, gpu {g}"
            else:
                assert abs(val - g) <= atol, f"{metric}/{pert}: cpu={val} gpu={g}"


def _route(adata):
    return adata.uns["scx_accel"]["perturbation_metrics"]


def test_perturbation_metrics_gpu_matches_cpu_inmemory():
    real, pred = _make_paired_adata()
    cpu = pyscx.accel.perturbation_metrics(real, pred, device="cpu")
    gpu = pyscx.accel.perturbation_metrics(real, pred, device="gpu")
    assert _route(pred)["route"].startswith("gpu"), _route(pred)
    _assert_metrics_close(cpu, gpu, atol=1e-5)


def test_perturbation_metrics_gpu_matches_cpu_backed(tmp_path):
    real, pred = _make_paired_adata(seed=1)
    rp = str(tmp_path / "real.scx")
    pp = str(tmp_path / "pred.scx")
    pyscx.from_anndata(real, rp, csc="always")
    pyscx.from_anndata(pred, pp, csc="always")
    real_b = pyscx.open(rp).to_anndata(backed=True)
    pred_b = pyscx.open(pp).to_anndata(backed=True)

    cpu = pyscx.accel.perturbation_metrics(real_b, pred_b, device="cpu")
    gpu = pyscx.accel.perturbation_metrics(real_b, pred_b, device="gpu")
    assert _route(pred_b)["route"].startswith("gpu"), _route(pred_b)
    _assert_metrics_close(cpu, gpu, atol=1e-5)


def test_perturbation_metrics_gpu_matches_cpu_embed_key():
    real, pred = _make_paired_adata(seed=2)
    # Attach a shared dense embedding to both halves.
    rng = np.random.default_rng(7)
    for adata in (real, pred):
        adata.obsm["X_emb"] = rng.normal(0, 1, size=(adata.n_obs, 12)).astype(np.float64)

    cpu = pyscx.accel.perturbation_metrics(real, pred, embed_key="X_emb", device="cpu")
    gpu = pyscx.accel.perturbation_metrics(real, pred, embed_key="X_emb", device="gpu")
    assert _route(pred)["route"].startswith("gpu"), _route(pred)
    # Dense embed GPU kernel is f32 → looser bar than the sparse gene-space paths.
    _assert_metrics_close(cpu, gpu, atol=1e-3)


def test_pseudobulk_means_gpu_matches_cpu():
    real, _ = _make_paired_adata(seed=3)
    means_cpu, groups_cpu = pyscx.accel.pseudobulk_means(real, "perturbation", device="cpu")
    means_gpu, groups_gpu = pyscx.accel.pseudobulk_means(real, "perturbation", device="gpu")
    assert list(groups_cpu) == list(groups_gpu)
    np.testing.assert_allclose(means_cpu, means_gpu, atol=1e-5, rtol=0.0)
    assert real.uns["scx_accel"]["pseudobulk_means"]["route"].startswith("gpu")
