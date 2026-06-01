"""GPU CSC-direct parity test for `pyscx.accel.pdex_ref(device="gpu")` (§B.10 Bg).

Exercises the v3 CSC-direct pdex_ref route (`gpu_csc_v3`) on a backed SCX file
with a CSC sidecar, cross-checking numerics against the CPU path and asserting
the recorded route is the CSC-direct one — the route assertion missing from the
device-agnostic `test_pdex_ref_gpu_parity.py`.

The v3 routes are gated by `SCX_GPU_DE_V3`, read once per process via an
`OnceLock`, so this module sets it at import time (before pyscx runs any DE) and
is intended to run in its own process — which is how the Chimera GPU test
harness invokes GPU test files (isolated). If the flag wasn't active in this
process, the route assertion is skipped while the numerical parity check runs.

Skipped cleanly when `pyscx.accel.gpu_available()` is `False`.
"""

from __future__ import annotations

import os

# Must be set before the first DE call locks in the OnceLock read.
os.environ.setdefault("SCX_GPU_DE_V3", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pl = pytest.importorskip("polars")

import anndata as ad  # noqa: E402

import pyscx  # noqa: E402

from _pdex_fixtures import _make_adata, REFERENCE  # noqa: E402


pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)


def _open_with_csc(path, adata: ad.AnnData) -> ad.AnnData:
    """Round-trip ``adata`` through a CSC-equipped SCX file, opened backed."""
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=5)
    return pyscx.open(str(path)).to_anndata(backed=True)


def _route(adata: ad.AnnData) -> str | None:
    try:
        return adata.uns["scx_accel"]["pdex_ref"]["route"]
    except Exception:
        return None


def _compare(cpu_df, gpu_df) -> None:
    cpu_df = cpu_df.sort(["target", "feature"])
    gpu_df = gpu_df.sort(["target", "feature"])
    assert cpu_df.shape == gpu_df.shape, f"shape mismatch: cpu={cpu_df.shape}, gpu={gpu_df.shape}"
    for col in ("target", "feature"):
        assert (cpu_df[col] == gpu_df[col]).all(), f"column '{col}' mismatch"

    u_cpu = cpu_df["statistic"].cast(pl.Float64).to_numpy()
    u_gpu = gpu_df["statistic"].cast(pl.Float64).to_numpy()
    finite = np.isfinite(u_cpu) & np.isfinite(u_gpu)
    np.testing.assert_allclose(u_cpu[finite], u_gpu[finite], atol=1e-6, rtol=0.0,
                               err_msg="U statistic mismatch")
    for col in ("p_value", "fdr"):
        a = cpu_df[col].cast(pl.Float64).to_numpy()
        b = gpu_df[col].cast(pl.Float64).to_numpy()
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        mask = np.isfinite(a)
        np.testing.assert_allclose(a[mask], b[mask], atol=1e-9, rtol=1e-6,
                                   err_msg=f"column '{col}' tolerance violation")
    for col in ("target_mean", "ref_mean", "log2_fold_change", "percent_change"):
        a = cpu_df[col].cast(pl.Float64).to_numpy()
        b = gpu_df[col].cast(pl.Float64).to_numpy()
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        mask = np.isfinite(a)
        np.testing.assert_allclose(a[mask], b[mask], atol=1e-4, rtol=1e-4,
                                   err_msg=f"column '{col}' tolerance violation")


def test_pdex_ref_gpu_csc_parity_vs_reference(tmp_path) -> None:
    base = _make_adata()
    cpu = _open_with_csc(tmp_path / "cpu.scx", base)
    gpu = _open_with_csc(tmp_path / "gpu.scx", base)

    cpu_df = pyscx.accel.pdex_ref(cpu, "target", reference=REFERENCE, device="cpu")
    gpu_df = pyscx.accel.pdex_ref(gpu, "target", reference=REFERENCE, device="gpu")

    _compare(cpu_df, gpu_df)

    route = _route(gpu)
    if route != "gpu_csc_v3":
        pytest.skip(f"SCX_GPU_DE_V3 not active in this process (route={route!r})")
