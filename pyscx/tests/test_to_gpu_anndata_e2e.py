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

    Small integer UMI-like counts (``poisson(2.0)``, median ~2). The framed (v4)
    layout decodes group-by-group in VRAM (``decode_framed_scx1_gpu``); Scx1 is
    the only codec with GPU decode kernels.

    ``codec="scx1"`` is **explicit, and must stay explicit.** This fixture used
    to rely on ``codec="auto"`` picking Scx1 from the median-≤-8 heuristic, but
    ``auto`` is cost-aware now and adopts ``ShufDeltaZstd`` whenever it comes out
    ≥5% smaller — which it does here (``scx info`` reports ``Codec: shufdelta``).
    A ShufDeltaZstd shard host-bounces, so every assertion below about in-VRAM
    decode silently became untestable. The `accel_to_gpu_anndata` benchmark
    already forces ``--codec scx1`` for exactly this reason; this fixture had not
    caught up. A test whose premise is "these shards are Scx1" must not leave the
    codec to a heuristic that is free to change.

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
    pyscx.from_anndata(adata, path, codec="scx1")
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


def test_device_decode_failure_falls_back_to_host_assemble(scx1_scx, monkeypatch):
    """A failed in-VRAM decode still produces X, via host-assemble (§8.10).

    ``SCX_FORCE_DEVICE_DECODE_FAILURE=1`` makes the device decode report a
    module-load failure — the realistic trigger, since a build whose PTX did not
    compile bakes empty stubs that fail exactly there. Before the fallback this
    raised ``RuntimeError: GPU shard assembly failed``, even though the
    host-assemble path beside it reaches the same cupy ``X``.

    The fallback must be *visible*: a silently-degraded handoff is a broken
    build nobody notices. Both the warning and the recorded reason are asserted,
    since ``transfer_mode`` alone cannot distinguish this from host-assemble
    chosen up front for a filtered request.
    """
    monkeypatch.setenv("SCX_FORCE_DEVICE_DECODE_FAILURE", "1")

    with pytest.warns(UserWarning, match="in-VRAM shard decode failed"):
        gpu_adata = pyscx.open(scx1_scx).to_gpu_anndata()

    meta = gpu_adata.uns["scx_accel"]["to_gpu_anndata"]
    assert meta["transfer_mode"] == "scx_device_handoff"
    assert meta["fallback_reason"] == "gpu_runtime_error"
    # Still a GPU route: the result is on the device either way, just uploaded
    # from the host rather than decoded in VRAM.
    assert meta["route"] == "gpu_csr"

    # And the answer is right, not merely present.
    cpu_x = pyscx.open(scx1_scx).to_anndata().X
    cpu_x = cpu_x.tocsr() if not sp.isspmatrix_csr(cpu_x) else cpu_x
    cpu_x.sort_indices()
    fallback_x = gpu_adata.X.get()
    fallback_x.sort_indices()
    np.testing.assert_array_equal(fallback_x.indptr, cpu_x.indptr)
    np.testing.assert_array_equal(fallback_x.indices, cpu_x.indices)
    np.testing.assert_allclose(fallback_x.data, cpu_x.data, rtol=1e-5, atol=1e-5)


def test_device_decode_fallback_is_off_by_default(scx1_scx):
    """The knob is opt-in: an unset env leaves the fast path alone.

    Guards the fault-injection hook itself — a hook that fired unconditionally
    would make the test above pass while silently disabling the in-VRAM decode
    for every user.
    """
    import os

    assert "SCX_FORCE_DEVICE_DECODE_FAILURE" not in os.environ
    meta = pyscx.open(scx1_scx).to_gpu_anndata().uns["scx_accel"]["to_gpu_anndata"]
    # The fast path ran. Asserted as "not host-assemble" rather than as a
    # specific device-decode mode: which of the two fast-path modes you get
    # depends on the shard codec, and this test is about the knob, not the
    # codec. `test_to_gpu_anndata_decode_gpu_transfer_mode` is what pins the
    # fully-in-VRAM mode.
    assert meta["transfer_mode"] != "scx_device_handoff"
    assert meta["fallback_reason"] == "none"


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


@pytest.fixture
def scx1_with_slots(tmp_path):
    """The same framed Scx1 X, plus two keys in each aligned slot and a raw.

    Separate from ``scx1_scx`` so the transfer-mode assertions above keep
    running against the minimal file they were written for.
    """
    import anndata
    import pandas as pd

    rng = np.random.default_rng(1)
    n_obs, n_vars, raw_n_vars = 120, 40, 60
    dense = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random(dense.shape) > 0.3] = 0.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)]),
        varm={
            "PCs": rng.random((n_vars, 4)).astype(np.float32),
            "loadings": rng.random((n_vars, 3)).astype(np.float32),
        },
        obsp={
            "connectivities": sp.csr_matrix(
                rng.random((n_obs, n_obs)).astype(np.float32)
            ),
            "distances": sp.csr_matrix(rng.random((n_obs, n_obs)).astype(np.float32)),
        },
    )
    raw_dense = rng.poisson(2.0, size=(n_obs, raw_n_vars)).astype(np.float32)
    adata.raw = anndata.AnnData(
        X=sp.csr_matrix(raw_dense),
        var=pd.DataFrame(index=[f"raw_gene_{j}" for j in range(raw_n_vars)]),
    )

    path = str(tmp_path / "slots.scx")
    pyscx.from_anndata(adata, path, codec="scx1")
    return path


def test_to_gpu_anndata_honours_slot_filters(scx1_with_slots):
    """`to_gpu_anndata` shares the eager host assembler, so the slot filters
    reach it — but it takes its own branch into that assembler, so nothing on
    the CPU side proves it."""
    gpu_adata = pyscx.open(scx1_with_slots).to_gpu_anndata(varm=["PCs"], obsp=[])

    assert set(gpu_adata.varm) == {"PCs"}
    assert len(gpu_adata.obsp) == 0


def test_to_gpu_anndata_raw_false_skips_the_rebuild(scx1_with_slots):
    import warnings

    # Control: the default still reconstructs raw on this path, so the
    # assertion below is about the kwarg and not about raw being broken here.
    assert pyscx.open(scx1_with_slots).to_gpu_anndata().raw is not None

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        gpu_adata = pyscx.open(scx1_with_slots).to_gpu_anndata(raw=False)

    assert gpu_adata.raw is None
    assert [w for w in rec if str(w.message).startswith("dropped_raw:")] == []
