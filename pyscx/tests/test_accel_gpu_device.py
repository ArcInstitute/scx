"""Device-string parsing tests for `pyscx.accel.*` (multi-GPU dispatch).

`device="gpu:N"` is documented in README + docs/scanpy.md but historically
only the bool aspect of the string was honoured: the suffix was parsed and
discarded, so every accelerator op ran on GPU 0. This file exercises the
post-fix parser through the public surface.

* Pure parsing tests run on any host (CPU-only or GPU) — they reach the
  parser through `pyscx.accel.normalize_total(adata, device=...)` with a
  scipy CSR adata and assert on the raised exception type/message.
* End-to-end dispatch tests are gated on `pyscx.accel.gpu_info() is not None`.
"""

from __future__ import annotations

import anndata
import numpy as np
import pytest
import scipy.sparse as sp

import pyscx


def _gpu_available() -> bool:
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


def _small_adata() -> anndata.AnnData:
    rng = np.random.default_rng(0)
    dense = rng.poisson(2.0, size=(50, 30)).astype(np.float32)
    dense[rng.random(dense.shape) > 0.3] = 0
    return anndata.AnnData(X=sp.csr_matrix(dense))


# ---------------------------------------------------------------------------
# Pure parsing — runs on any host. The parser is reached via the public API.
# ---------------------------------------------------------------------------


def test_unknown_device_raises_value_error():
    a = _small_adata()
    with pytest.raises(ValueError, match="unknown device"):
        pyscx.accel.normalize_total(a, device="not-a-device")


# ---------------------------------------------------------------------------
# `gpu_available()` boolean probe (always runs, no skipif).
# Pre-fix, the same `gpu_info() is not None` test had to be repeated in 8
# pytest files. Post-fix it's a single named API call.
# ---------------------------------------------------------------------------


def test_gpu_available_returns_bool():
    """`pyscx.accel.gpu_available()` must return a real `bool`, not None or
    a dict, so `if pyscx.accel.gpu_available():` works cleanly.
    """
    result = pyscx.accel.gpu_available()
    assert isinstance(result, bool), f"expected bool, got {type(result).__name__}"


def test_gpu_available_agrees_with_gpu_info():
    """`gpu_available()` must agree with the `gpu_info() is not None` idiom
    that 8 existing test files use as their `_gpu_available` helper.
    """
    via_new = pyscx.accel.gpu_available()
    via_old = pyscx.accel.gpu_info() is not None
    assert via_new == via_old


def test_gpu_with_non_integer_suffix_raises_value_error():
    a = _small_adata()
    with pytest.raises(ValueError, match="non-integer suffix"):
        pyscx.accel.normalize_total(a, device="gpu:abc")


def test_gpu_with_garbage_suffix_raises_value_error():
    """`gpu` with a non-`:N` suffix is treated as unknown (e.g. `gpux`)."""
    a = _small_adata()
    with pytest.raises(ValueError, match="unknown device"):
        pyscx.accel.normalize_total(a, device="gpux")


# ---------------------------------------------------------------------------
# CPU-only build — `device="gpu*"` should raise RuntimeError.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(_gpu_available(), reason="CPU-only build check")
def test_gpu_on_cpu_only_build_raises_runtime_error():
    a = _small_adata()
    with pytest.raises(RuntimeError):
        pyscx.accel.normalize_total(a, device="gpu")
    with pytest.raises(RuntimeError):
        pyscx.accel.normalize_total(a, device="gpu:0")


# ---------------------------------------------------------------------------
# GPU host — exercise the index-validation path.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")
def test_gpu_zero_runs():
    """`device="gpu:0"` is accepted and behaves like `device="gpu"`."""
    a = _small_adata()
    pyscx.accel.normalize_total(a, device="gpu:0")
    # Running again with bare `gpu` on a fresh fixture must not error.
    b = _small_adata()
    pyscx.accel.normalize_total(b, device="gpu")


@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")
def test_gpu_out_of_range_raises_runtime_error():
    """An index past the visible CUDA-device count raises `RuntimeError`."""
    info = pyscx.accel.gpu_info()
    assert info is not None  # gated above
    # `gpu_info()` defaults to device 0; `gpu_info(device=...)` reports on the
    # card named. An out-of-range ordinal has no info to report, so this is
    # also the cheap probe for "that device does not exist".
    assert pyscx.accel.gpu_info(device="gpu:0") is not None
    a = _small_adata()
    with pytest.raises(RuntimeError, match="CUDA device"):
        pyscx.accel.normalize_total(a, device="gpu:99")


@pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")
def test_gpu_info_reports_the_device_it_was_given():
    """`gpu_info()` used to hardcode device 0 — on a multi-GPU host it reported
    the wrong card's free VRAM and created a CUDA context on device 0 as a side
    effect of being asked (review 8.13).

    This node has whatever cards it has, so what is checked here is that the
    default and the explicit ordinal agree and that a bad selector is refused
    rather than silently answered from device 0. Distinguishing two *different*
    cards needs a multi-GPU host and is not asserted.
    """
    default = pyscx.accel.gpu_info()
    explicit = pyscx.accel.gpu_info(device="gpu:0")
    assert explicit is not None
    assert explicit["device"] == default["device"]

    # An ordinal the host does not have resolves to a RuntimeError from the
    # shared device grammar, not to a silent report on device 0.
    with pytest.raises(RuntimeError, match="CUDA device"):
        pyscx.accel.gpu_info(device="gpu:99")

    # `cpu` is a category error: there is no CPU VRAM to report.
    with pytest.raises(ValueError, match="selects the CPU"):
        pyscx.accel.gpu_info(device="cpu")

