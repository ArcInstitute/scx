"""Dispatch and kwarg-handling tests for `pyscx.accel.leiden`.

Two backends as peers (selected by `device`):
* `device="cpu"` → Rust-native (`scx_accel::leiden`), backend tag `scx-accel`.
* `device="gpu"` / `"gpu:N"` → cuGraph, backend tag `cugraph`. `gpu:N` pins
  the call to CUDA device N via `cupy.cuda.Device(N)`.
* `device="auto"` (default) → cuGraph if a CUDA device is visible and
  `cugraph` imports cleanly, else Rust-native.

There is no silent fallback. The Python `leidenalg` shim has been deleted —
callers who want it run `scanpy.tl.leiden(flavor="leidenalg")` themselves.

GPU-requiring tests are gated on `pyscx.accel.gpu_info() is not None`. The
`gpu:N` index-honoring test additionally requires ≥ 2 visible CUDA devices.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest


def _gpu_available() -> bool:
    try:
        import pyscx

        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


def _visible_cuda_device_count() -> int:
    try:
        import cupy.cuda.runtime as rt

        return int(rt.getDeviceCount())
    except Exception:
        return 0


gpu_only = pytest.mark.skipif(
    not _gpu_available(), reason="CUDA GPU not available"
)


@pytest.fixture
def adata_with_neighbors(synthetic_adata):
    """`synthetic_adata` + PCA + kNN graph — ready for Leiden."""
    import pyscx

    adata = synthetic_adata.copy()
    pyscx.accel.pca(adata, n_comps=10)
    pyscx.accel.neighbors(adata, n_neighbors=10)
    return adata


# ---------------------------------------------------------------------------
# Dispatch resolution
# ---------------------------------------------------------------------------


def test_device_cpu_uses_rust_native(adata_with_neighbors):
    """`device="cpu"` always resolves to Rust-native, regardless of host."""
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="cpu")
    assert adata.uns["leiden"]["backend"] == "scx-accel"
    assert adata.uns["leiden"]["params"]["device"] == "cpu"


def test_device_auto_resolution_cpu_host(adata_with_neighbors):
    """On a CPU-only host, `device="auto"` falls back to Rust-native."""
    if _gpu_available():
        pytest.skip("GPU available — auto resolution is exercised in gpu_host test")
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="auto")
    assert adata.uns["leiden"]["backend"] == "scx-accel"
    assert adata.uns["leiden"]["params"]["device"] == "auto"


@gpu_only
def test_device_auto_resolution_gpu_host(adata_with_neighbors):
    """On a GPU host with cuGraph, `device="auto"` picks cuGraph."""
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="auto")
    assert adata.uns["leiden"]["backend"] == "cugraph"
    assert adata.uns["leiden"]["params"]["device"] == "auto"


def test_device_gpu_on_cpu_only_host_raises(adata_with_neighbors):
    """`device="gpu"` must raise RuntimeError on a CPU-only host. No fallback."""
    if _gpu_available():
        pytest.skip("GPU available — hard-error path only reachable on CPU-only host")
    import pyscx

    adata = adata_with_neighbors
    with pytest.raises(RuntimeError):
        pyscx.accel.leiden(adata, device="gpu")


@gpu_only
def test_device_gpu_uses_cugraph(adata_with_neighbors):
    """`device="gpu"` runs cuGraph and records `gpu_id == 0`."""
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="gpu")
    assert adata.uns["leiden"]["backend"] == "cugraph"
    assert adata.uns["leiden"]["params"]["device"] == "gpu"
    assert adata.uns["leiden"]["params"]["gpu_id"] == 0


@gpu_only
def test_device_gpu_zero_explicit(adata_with_neighbors):
    """`device="gpu:0"` is identical to `device="gpu"` and records `gpu_id == 0`."""
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="gpu:0")
    assert adata.uns["leiden"]["backend"] == "cugraph"
    assert adata.uns["leiden"]["params"]["device"] == "gpu:0"
    assert adata.uns["leiden"]["params"]["gpu_id"] == 0


@gpu_only
def test_device_gpu_index_honored(adata_with_neighbors):
    """`device="gpu:N"` pins cuGraph allocation to CUDA device N."""
    if _visible_cuda_device_count() < 2:
        pytest.skip("requires ≥ 2 visible CUDA devices")
    import pyscx

    for device_id in (0, 1):
        adata = adata_with_neighbors.copy()
        pyscx.accel.leiden(adata, device=f"gpu:{device_id}")
        assert adata.uns["leiden"]["backend"] == "cugraph"
        assert adata.uns["leiden"]["params"]["gpu_id"] == device_id


@gpu_only
def test_device_gpu_index_out_of_range_raises(adata_with_neighbors):
    """`device="gpu:99"` is rejected by `resolve_device`'s index validation."""
    import pyscx

    adata = adata_with_neighbors
    with pytest.raises(RuntimeError, match="CUDA device"):
        pyscx.accel.leiden(adata, device="gpu:99")


# ---------------------------------------------------------------------------
# Ignored-kwarg warnings
# ---------------------------------------------------------------------------


def test_theta_warns_on_cpu(adata_with_neighbors):
    """`theta != 1.0` with `device="cpu"` emits UserWarning."""
    import pyscx

    adata = adata_with_neighbors
    with pytest.warns(UserWarning, match="theta"):
        pyscx.accel.leiden(adata, device="cpu", theta=2.0)


def test_theta_default_no_warning(adata_with_neighbors):
    """Default `theta=1.0` with `device="cpu"` must NOT warn."""
    import pyscx

    adata = adata_with_neighbors
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pyscx.accel.leiden(adata, device="cpu")  # default theta


@gpu_only
def test_parallel_warns_on_gpu(adata_with_neighbors):
    """`parallel=True` with `device="gpu"` emits UserWarning."""
    import pyscx

    adata = adata_with_neighbors
    with pytest.warns(UserWarning, match="parallel"):
        pyscx.accel.leiden(adata, device="gpu", parallel=True)


def test_parallel_on_cpu_no_warning(adata_with_neighbors):
    """`parallel=True` with `device="cpu"` is honored (no warning)."""
    import pyscx

    adata = adata_with_neighbors
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pyscx.accel.leiden(adata, device="cpu", parallel=True)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_no_silent_fallback_to_leidenalg(adata_with_neighbors):
    """`adata.uns["leiden"]["backend"]` is never "leidenalg" — that path was
    deleted. Even when leidenalg is installed and importable, we must not
    reach it.
    """
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="cpu")
    assert adata.uns["leiden"]["backend"] != "leidenalg"
    assert adata.uns["leiden"]["backend"] in {"scx-accel", "cugraph"}


def test_ignored_field_populated_for_theta(adata_with_neighbors):
    """`params["ignored"]` lists `theta` when it was set on the CPU path."""
    import pyscx

    adata = adata_with_neighbors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pyscx.accel.leiden(adata, device="cpu", theta=2.0)
    assert list(adata.uns["leiden"]["params"]["ignored"]) == ["theta"]


def test_ignored_field_empty_when_kwargs_default(adata_with_neighbors):
    """`params["ignored"]` is empty when no ignored kwarg was set."""
    import pyscx

    adata = adata_with_neighbors
    pyscx.accel.leiden(adata, device="cpu")
    assert list(adata.uns["leiden"]["params"]["ignored"]) == []


# ---------------------------------------------------------------------------
# Sanity (ARI floor)
# ---------------------------------------------------------------------------


def test_ari_vs_scanpy_leiden_cpu(adata_with_neighbors):
    """Rust-native Leiden ARI vs scanpy's leidenalg ≥ 0.40 (loose floor —
    matches the existing `TestLeiden::test_ari_vs_scanpy_leiden` style).
    """
    try:
        import leidenalg  # noqa: F401
        import igraph  # noqa: F401
    except ImportError:
        pytest.skip("leidenalg/igraph unavailable — ARI baseline cannot run")
    import pyscx
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score

    adata_pyscx = adata_with_neighbors.copy()
    pyscx.accel.leiden(adata_pyscx, device="cpu", random_state=0)

    adata_sc = adata_with_neighbors.copy()
    sc.tl.leiden(adata_sc, flavor="leidenalg", random_state=0, directed=False)

    ari = adjusted_rand_score(
        np.asarray(adata_pyscx.obs["leiden"]),
        np.asarray(adata_sc.obs["leiden"]),
    )
    assert ari >= 0.40, f"ARI {ari:.3f} below 0.40 floor"
