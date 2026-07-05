"""End-to-end test for ``pyscx.open(...).to_gpu_anndata()`` transfer metadata.

A framed Scx1 file decodes its shards *fully in VRAM* (group-by-group via
``decode_framed_scx1_gpu``) and the handoff stamps an honest ``transfer_mode`` /
``bytes_uploaded`` on ``uns["scx_accel"]["to_gpu_anndata"]``. The Rust GPU
unit tests cover byte-level decode parity; this asserts the Python-visible
metadata path and GPU↔CPU value parity end to end.

GPU-gated: skipped on a CPU-only host or a pyscx not built with the gpu feature.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx


def _gpu_available() -> bool:
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)


@pytest.fixture
def scx1_scx(tmp_path):
    """A small framed Scx1 SCX file that decodes fully in VRAM.

    Small integer UMI-like counts (``poisson(2.0)``, median ~2) auto-route to the
    Scx1 codec, the only codec with GPU decode kernels. The default framed (v4)
    layout decodes group-by-group in VRAM (``decode_framed_scx1_gpu``).

    NB: do *not* reuse ``conftest.py::synthetic_adata`` here — its
    ``randint(0, 200)`` values (median ~100) route to Zstd, which host-bounces.
    """
    import anndata

    rng = np.random.default_rng(0)
    n_obs, n_vars = 200, 60
    dense = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random(dense.shape) > 0.3] = 0.0
    adata = anndata.AnnData(X=sp.csr_matrix(dense))

    path = str(tmp_path / "scx1.scx")
    pyscx.from_anndata(adata, path)  # codec="auto" → framed Scx1
    return path


def test_to_gpu_anndata_decode_gpu_transfer_mode(scx1_scx):
    """Framed Scx1 file → fully-on-device decode + small ``bytes_uploaded``."""
    gpu_adata = pyscx.open(scx1_scx).to_gpu_anndata()

    meta = gpu_adata.uns["scx_accel"]["to_gpu_anndata"]
    nnz = int(gpu_adata.X.nnz)

    # The strongest single guard: this mode is reachable *only* when every Scx1
    # shard decoded in VRAM (framed group-by-group device decode).
    assert meta["transfer_mode"] == "scx_device_decode_gpu", (
        f"expected fully-on-device decode, got transfer_mode="
        f"{meta['transfer_mode']!r} (codec not Scx1?)"
    )

    # Only the tiny indptr is uploaded — far below a full-matrix HtoD (nnz * 8B
    # for the i64 indices + f32 data the host path would otherwise upload).
    bytes_uploaded = meta["bytes_uploaded"]
    assert bytes_uploaded is not None
    assert 0 < bytes_uploaded < nnz * 8, (
        f"bytes_uploaded={bytes_uploaded} should be ~indptr-only, well under "
        f"nnz*8={nnz * 8}"
    )


def test_to_gpu_anndata_values_match_cpu(scx1_scx):
    """GPU-decoded ``X`` is identical to the CPU decode path."""
    cpu_x = pyscx.open(scx1_scx).to_anndata().X
    cpu_x = cpu_x.tocsr() if not sp.isspmatrix_csr(cpu_x) else cpu_x
    cpu_x.sort_indices()

    gpu_x = pyscx.open(scx1_scx).to_gpu_anndata().X.get()  # cupyx CSR → host
    gpu_x.sort_indices()

    assert gpu_x.shape == cpu_x.shape
    assert gpu_x.nnz == cpu_x.nnz
    np.testing.assert_array_equal(gpu_x.indptr, cpu_x.indptr)
    np.testing.assert_array_equal(gpu_x.indices, cpu_x.indices)
    np.testing.assert_allclose(gpu_x.data, cpu_x.data, rtol=1e-5, atol=1e-5)
