"""GPU PCA + preprocessing parity tests.

Every test in this file is gated on `pyscx.accel.gpu_available()`; on a
CPU-only host or a build without the `gpu` feature they are silently skipped.
Covered scenarios:

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


def _clustered_count_adata(
    n_obs: int,
    n_vars: int,
    n_clusters: int,
    density: float,
    seed: int,
):
    """Synthetic count AnnData with planted low-rank cluster structure.

    Produces a CSR matrix with a flat sparse Poisson(0.5) background plus
    cluster-specific marker-gene signal (50 markers per cluster, Poisson(15)
    expression boost). The leading `n_clusters - 1` PCs separate the
    clusters and have eigenvalues well above the noise floor — the
    eigenvalue gap between PC[n_clusters-1] and PC[n_clusters] is typically
    ~100×, so randomized PCA at 4 power iterations converges to the same
    cluster-discrimination subspace regardless of the random Ω seed/RNG.

    Use this fixture for tests that need a stable top-`k` subspace to
    compare against a different PCA implementation. `_random_count_adata`'s
    flat-spectrum output is fine for testing PCA correctness in isolation
    but breaks cross-implementation per-PC comparisons because all leading
    eigenvalues are near-degenerate.
    """
    import anndata

    rng = np.random.default_rng(seed)
    dense = rng.poisson(0.5, size=(n_obs, n_vars)).astype(np.float32)
    mask = rng.random((n_obs, n_vars)) > density
    dense[mask] = 0

    # Per-cell cluster assignment + planted marker-gene signal.
    cluster = rng.integers(n_clusters, size=n_obs)
    markers = rng.choice(n_vars, size=(n_clusters, 50), replace=True)
    for i in range(n_obs):
        for g in markers[cluster[i]]:
            dense[i, g] += rng.poisson(15.0)

    return anndata.AnnData(X=sp.csr_matrix(dense))


# --------------------------------------------------------------------- #
# 7.1.a — GPU covariance PCA was removed.
#   The native in-VRAM covariance core no longer exists (in-memory
#   `device="gpu"` PCA routes to rapids-singlecell, and `resolve_gpu_method`
#   always yields "randomized"), so there is no GPU covariance path to compare
#   against CPU covariance — the former test_gpu_covariance_vs_cpu_covariance
#   test was dropped. GPU randomized parity is covered below.
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# 7.1.b — GPU randomized (Householder) vs CPU randomized, matched seed
# --------------------------------------------------------------------- #

@gpu_only
def test_gpu_randomized_householder_vs_cpu_randomized(monkeypatch):
    """CPU vs GPU randomized PCA (Householder) on a fixture with a planted
    well-separated top-`(n_clusters-1)` subspace.

    Pins `SCX_FORCE_NATIVE_GPU=1` so the GPU side exercises the **native**
    randomized PCA (cuSPARSE+cuSOLVER) rather than rapids-singlecell — in-memory
    `device="gpu"` PCA routes to rapids by default after Phase 1.3, and this
    test is specifically about native CPU-vs-GPU randomized parity.

    The CPU path seeds a Rust `ChaCha8` RNG; the GPU path seeds cuRAND's
    XORWOW generator. Even at matched `random_state`, the two streams
    produce *different* Ω matrices. Randomized PCA at 4 power iterations
    therefore only converges to the same per-PC basis where the data has a
    spectral gap — within a degenerate subspace the two paths can pick
    different orthonormal bases that nonetheless span the same subspace.

    Pre-fix this test used `_random_count_adata`'s flat Poisson(2) ×
    density=0.03 spectrum (all top-10 eigenvalues within ~5 %, no spectral
    gap) and asserted top-10 per-PC cosine ≥ 0.99 — impossible to satisfy
    across two valid PCA decompositions of a near-degenerate signal.
    `_clustered_count_adata` plants 5 clusters with marker-gene signal,
    producing 4 well-separated top PCs (eigenvalue gap ≈ 250× between
    PC 4 and PC 5).
    """
    import pyscx

    monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
    n_clusters = 5
    well_separated = n_clusters - 1
    # method="randomized" on both sides. (On GPU every method resolves to
    # randomized since Phase 3.2 removed the in-VRAM covariance core; the CPU
    # side is pinned to randomized explicitly.)
    adata = _clustered_count_adata(
        n_obs=1_000, n_vars=9_000, n_clusters=n_clusters, density=0.03, seed=2,
    )
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

    # 1. Top-`(n_clusters - 1)` per-PC cosine: the cluster-discrimination axes
    #    are well separated, so both algorithms recover the same basis. The
    #    f64 (CPU) vs f32 (GPU) gap allows for 1e-4 numerical drift here.
    cos = _cosine_sign_agnostic(a_cpu.obsm["X_pca"], a_gpu.obsm["X_pca"])
    assert (cos[:well_separated] >= 0.99).all(), (
        f"top-{well_separated} cosine: {cos[:well_separated]}"
    )

    # 2. variance_ratio across all 30 PCs must agree to f32 numerical noise.
    #    Unlike per-PC subspace rotation in near-degenerate regimes, the
    #    eigenvalues themselves are a deterministic property of the data —
    #    both algorithms must recover them within float precision.
    vr_cpu = np.asarray(a_cpu.uns["pca"]["variance_ratio"])
    vr_gpu = np.asarray(a_gpu.uns["pca"]["variance_ratio"])
    drift = np.max(np.abs(vr_cpu - vr_gpu))
    assert drift < 1e-3, (
        f"variance_ratio drift {drift:.3e} exceeds 1e-3 tolerance"
    )

    # 3. Spectral-gap sanity: the fixture must satisfy our pre-condition
    #    that the top-`(n_clusters-1)` PCs are well separated from the bulk.
    #    Without this, assertion (1) is meaningless. ~100× is typical here.
    gap = vr_cpu[well_separated - 1] / vr_cpu[well_separated]
    assert gap > 50.0, (
        f"fixture broke: spectral gap vr[{well_separated - 1}]/vr[{well_separated}]"
        f" = {gap:.2f} (expected > 50); regenerate _clustered_count_adata"
    )


# --------------------------------------------------------------------- #
# 7.1.c — GPU Cholesky vs GPU Householder
# --------------------------------------------------------------------- #

@gpu_only
def test_gpu_cholesky_matches_householder(monkeypatch):
    """qr_method="cholesky" and qr_method="householder" should agree to 1e-3.

    Pins `SCX_FORCE_NATIVE_GPU=1` so both runs exercise the native cuSOLVER QR
    step (rapids ignores `qr_method`, so without this both would route to rapids
    and the comparison would be vacuous).
    """
    import pyscx

    monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
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
def test_gpu_cholesky_ill_conditioned_raises(monkeypatch):
    """CholeskyQR2 must surface a RuntimeError with 'non-SPD' on a true
    rank-deficient input — pointing the caller at `qr_method="householder"`.

    Pins `SCX_FORCE_NATIVE_GPU=1`: in-memory `device="gpu"` PCA routes to
    rapids-singlecell by default (Phase 1.3), which ignores `qr_method` and
    never hits the native CholeskyQR2 non-SPD detection — so the native path
    must be forced for this test to exercise it.

    Build a `(n_obs × n_vars)` matrix as `X = U @ V^T` with `rank(X) = 5`,
    well below `k = n_comps + n_oversamples = 30`. Then `Y = (X - μ) @ Ω` has
    rank ≤ 6 < k, so `Y^T @ Y` is singular and `cusolverDnSpotrf` reports
    `devInfo > 0`.

    The Rust-level analogue `cusolver::tests::test_gpu_cholesky_qr2_ill_conditioned`
    tests the primitive directly on a hand-crafted (m × 10) input with two
    near-duplicate columns — that fixture is enough at the Rust layer
    because cholesky operates directly on the input. At the Python layer
    the randomized-PCA pipeline applies a `(n_vars × k)` random projection
    `Ω` before cholesky sees the data, and that projection hides
    column-level rank deficiencies in `X` whenever `rank(X) >= k`. We have
    to make `X` itself rank-deficient at the `k` scale to push singularity
    through to `Y`.
    """
    import anndata
    import pyscx

    monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
    rng = np.random.default_rng(7)
    n_obs, n_vars, rank = 1_000, 9_000, 5
    u = rng.standard_normal((n_obs, rank)).astype(np.float32)
    v = rng.standard_normal((n_vars, rank)).astype(np.float32)
    dense = (u @ v.T).astype(np.float32)
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
    # may not exist on all machines). In-memory `device="gpu"` PCA routes to
    # rapids-singlecell (or the native randomized core under
    # SCX_FORCE_NATIVE_GPU=1); the leading PCs must still align with scanpy.
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
def test_gpu_pca_materialized_csr_host_spike(monkeypatch):
    """The Phase-10 borrow fast-lane must keep the host RSS spike under
    ~1.5× the input CSR's nbytes when feeding a materialised scipy CSR to
    `pyscx.accel.pca(device="gpu")`.

    Pins `SCX_FORCE_NATIVE_GPU=1` — the borrow fast-lane is on the native GPU
    PCA dispatch; in-memory `device="gpu"` would otherwise route to rapids
    (which has its own, unrelated host-memory profile).

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

    monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
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
