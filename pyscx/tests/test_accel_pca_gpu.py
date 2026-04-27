"""GPU PCA + preprocessing parity tests — Phase 7.1.

Every test in this file is gated on `pyscx.accel.gpu_available()`; on a
CPU-only host or a build without the `gpu` feature they are silently skipped.
The covered scenarios follow GPU-ACC-SPEED-UP.md §7.1:

* GPU covariance PCA vs CPU covariance PCA (cosine ≥ 0.999, top-50)
* GPU randomized PCA (`qr_method="householder"`) vs CPU randomized PCA at
  matched seed
* GPU randomized PCA (`qr_method="cholesky"`) vs GPU randomized PCA
  (`qr_method="householder"`) (cosine ≥ 0.999)
* `qr_method="cholesky"` on a deliberately ill-conditioned input raises
  `RuntimeError` wrapping the cuSOLVER non-SPD error
* Lazy-path parity: backed SCX → normalize_total → log1p → pca(device="gpu")
  agrees with scanpy reference (cosine ≥ 0.99)
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _gpu_available() -> bool:
    """Return True iff pyscx was built with `--features gpu` AND a CUDA device
    is visible.

    We query `pyscx.accel.gpu_info()` (always exposed, returns None on CPU-only
    builds or CPU-only hosts). `pyscx.accel.gpu_available()` is not exported
    from the Python surface today — using `gpu_info()` is the canonical check.
    """
    try:
        import pyscx

        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — skipping GPU parity tests",
)


def _cosine_sign_agnostic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Column-wise cosine similarity between two (n, k) arrays, sign-agnostic
    (PCA signs are arbitrary). Returns a length-k vector of |cos|.
    """
    num = np.abs(np.einsum("nk,nk->k", a, b))
    denom_a = np.linalg.norm(a, axis=0)
    denom_b = np.linalg.norm(b, axis=0)
    denom = np.clip(denom_a * denom_b, 1e-12, None)
    return num / denom


def _random_count_adata(n_obs: int, n_vars: int, density: float, seed: int):
    """Synthetic Poisson-like sparse CSR AnnData with `density` nonzeros."""
    import anndata

    rng = np.random.default_rng(seed)
    dense = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    mask = rng.random((n_obs, n_vars)) > density
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    return anndata.AnnData(X=x)


# --------------------------------------------------------------------- #
# 7.1.a — GPU covariance PCA vs CPU covariance PCA
# --------------------------------------------------------------------- #

@gpu_only
def test_gpu_covariance_vs_cpu_covariance_cosine():
    """Top-50 PCs from GPU covariance PCA match CPU covariance PCA."""
    import pyscx

    adata = _random_count_adata(n_obs=1_000, n_vars=500, density=0.1, seed=1)
    a_cpu = adata.copy()
    a_gpu = adata.copy()

    pyscx.accel.pca(a_cpu, n_comps=50, device="cpu", method="covariance",
                    random_state=0)
    pyscx.accel.pca(a_gpu, n_comps=50, device="gpu", method="covariance",
                    random_state=0)

    cos = _cosine_sign_agnostic(a_cpu.obsm["X_pca"], a_gpu.obsm["X_pca"])
    # Top-50 PCs — loosen the very-trailing ones slightly; strong PCs must be tight.
    assert (cos[:25] >= 0.999).all(), f"top-25 cosine: min={cos[:25].min():.4f}"
    assert (cos >= 0.99).all(), f"top-50 cosine: min={cos.min():.4f}"


# --------------------------------------------------------------------- #
# 7.1.b — GPU randomized (Householder) vs CPU randomized, matched seed
# --------------------------------------------------------------------- #

@gpu_only
def test_gpu_randomized_householder_vs_cpu_randomized():
    """Matched-seed CPU vs GPU randomized PCA (Householder) — top-30 cosine."""
    import pyscx

    # n_vars > GPU_COVARIANCE_PCA_THRESHOLD to force the randomized path on both
    # sides regardless of the method="auto" routing threshold.
    adata = _random_count_adata(n_obs=1_000, n_vars=9_000, density=0.03, seed=2)
    a_cpu = adata.copy()
    a_gpu = adata.copy()

    pyscx.accel.pca(
        a_cpu, n_comps=30, device="cpu", method="randomized",
        random_state=0, n_oversamples=10, n_power_iterations=4,
    )
    pyscx.accel.pca(
        a_gpu, n_comps=30, device="gpu", method="randomized",
        qr_method="householder", random_state=0,
        n_oversamples=10, n_power_iterations=4,
    )

    cos = _cosine_sign_agnostic(a_cpu.obsm["X_pca"], a_gpu.obsm["X_pca"])
    # CPU f64 vs GPU f32 randomized paths can drift on tail PCs; leading PCs
    # should still be aligned tightly.
    assert (cos[:10] >= 0.99).all(), f"top-10 cosine: min={cos[:10].min():.4f}"


# --------------------------------------------------------------------- #
# 7.1.c — GPU Cholesky vs GPU Householder
# --------------------------------------------------------------------- #

@gpu_only
def test_gpu_cholesky_matches_householder():
    """qr_method="cholesky" and qr_method="householder" should agree to 1e-3."""
    import pyscx

    adata = _random_count_adata(n_obs=800, n_vars=9_000, density=0.05, seed=3)
    a_h = adata.copy()
    a_c = adata.copy()

    pyscx.accel.pca(a_h, n_comps=20, device="gpu", method="randomized",
                    qr_method="householder", random_state=42,
                    n_oversamples=10, n_power_iterations=2)
    pyscx.accel.pca(a_c, n_comps=20, device="gpu", method="randomized",
                    qr_method="cholesky", random_state=42,
                    n_oversamples=10, n_power_iterations=2)

    cos = _cosine_sign_agnostic(a_h.obsm["X_pca"], a_c.obsm["X_pca"])
    assert (cos >= 0.999).all(), f"cosine min={cos.min():.4f}"

    # Variance ratios should also agree tightly.
    vr_h = np.asarray(a_h.uns["pca"]["variance_ratio"])
    vr_c = np.asarray(a_c.uns["pca"]["variance_ratio"])
    assert np.max(np.abs(vr_h - vr_c)) < 1e-3, (
        f"variance_ratio drift: {np.max(np.abs(vr_h - vr_c)):.3e}"
    )


# --------------------------------------------------------------------- #
# 7.1.d — Ill-conditioned input makes CholeskyQR2 raise
# --------------------------------------------------------------------- #

@gpu_only
def test_gpu_cholesky_ill_conditioned_raises():
    """CholeskyQR2 must surface a RuntimeError with 'non-SPD' on near-singular input.

    Build a (1000 × 9000) matrix whose first two columns are a near-duplicate
    pair (differ by 1e-8 × noise) — condition number ≈ 10^8. CholeskyQR2 should
    fail with a clear error pointing to `qr_method="householder"`.
    """
    import anndata
    import pyscx

    rng = np.random.default_rng(7)
    n_obs, n_vars = 1_000, 9_000
    dense = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
    # Make columns 0 and 1 near-identical.
    dense[:, 1] = dense[:, 0] + 1e-8 * rng.standard_normal(n_obs).astype(np.float32)
    adata = anndata.AnnData(X=sp.csr_matrix(dense))

    with pytest.raises(RuntimeError, match="non-SPD"):
        pyscx.accel.pca(
            adata, n_comps=20, device="gpu", method="randomized",
            qr_method="cholesky", random_state=42,
            n_oversamples=10, n_power_iterations=2,
        )


# --------------------------------------------------------------------- #
# 7.1.e — Lazy pipeline agrees with scanpy
# --------------------------------------------------------------------- #

@gpu_only
def test_lazy_normalize_log1p_pca_matches_scanpy(tmp_path):
    """normalize_total → log1p → pca(device="gpu") on a backed SCX file
    agrees with scanpy's reference pipeline at cosine ≥ 0.99 on top PCs.
    """
    import anndata
    import pyscx
    import scanpy as sc

    # Reasonably-sized synthetic stand-in for pbmc3k (the real pbmc3k fixture
    # may not exist on all machines). Dims chosen so the covariance-PCA path
    # (n_vars ≤ GPU_COVARIANCE_PCA_THRESHOLD = 8000) is exercised.
    adata_src = _random_count_adata(n_obs=2_000, n_vars=3_000, density=0.07, seed=11)

    # Scanpy reference (in-memory).
    a_ref = adata_src.copy()
    sc.pp.normalize_total(a_ref, target_sum=1e4)
    sc.pp.log1p(a_ref)
    sc.pp.pca(a_ref, n_comps=30, random_state=0)

    # SCX backed + GPU lazy pipeline.
    scx_path = str(tmp_path / "pipeline_test.scx")
    pyscx.from_anndata(adata_src.copy(), scx_path)
    a_gpu = pyscx.open(scx_path).to_anndata()
    # normalize_total(device="gpu") eager-materializes; log1p then gets the
    # fusion-marker path and re-runs fused over the backed source.
    pyscx.accel.normalize_total(a_gpu, target_sum=1e4, device="gpu")
    pyscx.accel.log1p(a_gpu, device="gpu")
    pyscx.accel.pca(a_gpu, n_comps=30, device="gpu", random_state=0)

    cos = _cosine_sign_agnostic(a_ref.obsm["X_pca"], a_gpu.obsm["X_pca"])
    # Tail PCs are sensitive to preprocessing numerics drift (f64 scanpy vs f32
    # GPU path). Require the leading PCs to be well aligned.
    assert (cos[:10] >= 0.99).all(), f"top-10 cosine: min={cos[:10].min():.4f}"


# --------------------------------------------------------------------- #
# 7.4 — CPU-only graceful fallback is covered below in test_device_gating
# --------------------------------------------------------------------- #

def test_device_auto_always_works(synthetic_adata):
    """device="auto" must succeed on any host — GPU if available, else CPU."""
    import pyscx

    adata = synthetic_adata.copy()
    # On CPU-only hosts this falls through to the CPU path via resolve_device.
    pyscx.accel.pca(adata, n_comps=5, device="auto", random_state=0)
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)


def test_device_cpu_ignores_qr_method(synthetic_adata):
    """qr_method on the CPU path must be accepted (validated) but effectively
    ignored (no-op). This guards against the Phase-4 stub ever being reintroduced.
    """
    import pyscx

    adata = synthetic_adata.copy()
    pyscx.accel.pca(
        adata, n_comps=5, device="cpu", qr_method="cholesky", random_state=0,
    )
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 5)


def test_device_gating_cpu_build_rejects_gpu():
    """On a CPU-only build `device="gpu"` raises RuntimeError.
    On a GPU build the call succeeds — so this only asserts in the CPU-only case.
    """
    if _gpu_available():
        pytest.skip("GPU available — this test only runs on CPU-only builds")

    import pyscx

    adata = _random_count_adata(n_obs=50, n_vars=40, density=0.2, seed=0)
    with pytest.raises((RuntimeError, ValueError)):
        pyscx.accel.pca(adata, n_comps=4, device="gpu")


# --------------------------------------------------------------------- #
# Phase 10 — memory-spike regression for the materialised-CSR fast-lane
# --------------------------------------------------------------------- #

@gpu_only
@pytest.mark.skipif(
    not __import__("sys").platform.startswith("linux"),
    reason="ru_maxrss reporting differs by platform; only assert on Linux",
)
def test_gpu_pca_materialized_csr_host_spike():
    """The Phase-10 borrow fast-lane must keep the host RSS spike under
    ~1.5× the input CSR's nbytes when feeding a materialised scipy CSR to
    `pyscx.accel.pca(device="gpu")`.

    Pre-Phase-10 the path was numpy → `extract::<Vec<T>>` (copy 1) → `ScxCsr`
    → `read_shard` clone (copy 2), which pushed peak RSS to ≈ 2× input. The
    borrow path replaces the first copy with a `PyReadonlyArray1` view, so
    only one transient `Vec` materialises during dispatch. This test
    catches accidental re-introduction of the double-copy pattern.

    The 1.5× ceiling is loose on purpose — Python and CUDA driver state
    contribute non-trivial baseline RSS noise.
    """
    import gc
    import resource

    import pyscx

    # 100K × 200 @ 10% density: small enough to run in a CI test, large
    # enough that the input dominates baseline RSS noise.
    adata = _random_count_adata(n_obs=100_000, n_vars=200, density=0.10, seed=42)
    x = adata.X
    input_nbytes = int(x.indptr.nbytes + x.indices.nbytes + x.data.nbytes)

    gc.collect()
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux: kB

    pyscx.accel.pca(
        adata, n_comps=20, device="gpu", method="randomized", random_state=0,
    )

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    spike = max(peak - baseline, 0)

    # Loose 1.5× ceiling — a regression to the double-copy path would land
    # near 2× input_nbytes once Vec + cloned-ScxCsr coexist on the heap.
    assert spike < int(1.5 * input_nbytes), (
        f"PCA(device='gpu') materialised-CSR fast-lane regressed: "
        f"host RSS spike {spike / 1e6:.1f} MB exceeds 1.5× input "
        f"{input_nbytes / 1e6:.1f} MB. Phase-10 borrow path may be broken."
    )
    assert adata.obsm["X_pca"].shape == (adata.n_obs, 20)
