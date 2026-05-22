"""Symmetric `UserWarning` on `normalize_total(device="gpu")` scipy/dense fallback.

Closes B4 from `SCX-USER-REPORT-2026-05-21-Tier3.md`: pre-fix,
`accel.normalize_total(device="gpu")` on a materialized scipy/dense X
silently fell back to `scanpy.pp.normalize_total` with NO `UserWarning` —
asymmetric with `log1p`'s noisy fallback. Post-fix the call emits a
warning that names the device string, the gating condition (X must be
backed / lazy SCX), and the actionable fix (`to_anndata(backed=True)`).

The numerical result is unchanged: scanpy CPU runs in both cases.
"""

import numpy as np
import pyscx
import pytest
import scipy.sparse as sp


def _gpu_available() -> bool:
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — B4 fallback warning is GPU-only",
)


_MARKER_KEY = "__scx_gpu_pending_normalize__"


def _make_scipy_adata():
    import anndata

    rng = np.random.default_rng(0)
    dense = rng.poisson(2.0, size=(200, 100)).astype(np.float32)
    dense[rng.random(dense.shape) > 0.3] = 0
    return anndata.AnnData(X=sp.csr_matrix(dense))


def test_normalize_total_gpu_on_scipy_emits_fallback_warning():
    """B4: a `UserWarning` mentioning 'falls back to CPU' must fire."""
    a = _make_scipy_adata()
    with pytest.warns(UserWarning, match="falls back to CPU"):
        pyscx.accel.normalize_total(a, target_sum=1e4, device="gpu")


def test_normalize_total_gpu_on_scipy_does_not_stash_marker():
    """The CPU-fallback path must not write the GPU normalize fusion marker
    (otherwise a subsequent log1p would silently replay a stale fused pass).
    """
    a = _make_scipy_adata()
    with pytest.warns(UserWarning, match="falls back to CPU"):
        pyscx.accel.normalize_total(a, target_sum=1e4, device="gpu")
    assert _MARKER_KEY not in a.uns


def test_normalize_total_gpu_on_scipy_matches_scanpy():
    """Numerical parity: the CPU-fallback path produces the same result as
    `scanpy.pp.normalize_total` (which it actually calls under the hood).
    """
    import scanpy as sc

    a_gpu = _make_scipy_adata()
    a_ref = _make_scipy_adata()

    with pytest.warns(UserWarning, match="falls back to CPU"):
        pyscx.accel.normalize_total(a_gpu, target_sum=1e4, device="gpu")
    sc.pp.normalize_total(a_ref, target_sum=1e4)

    np.testing.assert_allclose(
        a_gpu.X.toarray(), a_ref.X.toarray(), rtol=1e-6, atol=1e-7
    )


def test_normalize_total_fallback_warning_names_backed_workaround():
    """The warning text must point the user at the actionable workaround
    (`to_anndata(backed=True)` / `ScxLazyTransformedDataset`), not at the
    misleading "call normalize_total first" recommendation B3 fixed for log1p.
    """
    a = _make_scipy_adata()
    with pytest.warns(UserWarning) as recorded:
        pyscx.accel.normalize_total(a, target_sum=1e4, device="gpu")
    msgs = [str(w.message) for w in recorded]
    relevant = [m for m in msgs if "falls back to CPU" in m]
    assert relevant, f"Expected fallback warning; got: {msgs}"
    assert "backed=True" in relevant[0] or "ScxLazyTransformedDataset" in relevant[0]
