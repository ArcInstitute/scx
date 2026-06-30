"""GPU CSC-direct parity test for `pyscx.accel.pdex_ref(device="gpu")` (§B.10 Bg).

Exercises the v3 CSC-direct pdex_ref route (`gpu_csc_v3`) on a backed SCX file
with a CSC sidecar, cross-checking numerics against the CPU path and asserting
the recorded route is the CSC-direct one — the route assertion missing from the
device-agnostic `test_pdex_ref_gpu_parity.py`.

GPU DE v3 is the unconditional default route (the former `SCX_GPU_DE_V3` gate
was removed), so a backed SCX file with a CSC sidecar always takes the
`gpu_csc_v3` route — the assertion below proves it.

Skipped cleanly when `pyscx.accel.gpu_available()` is `False`.
"""

from __future__ import annotations

import numpy as np
import pytest

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
    # Assert the finite masks agree first, so a CPU-finite / GPU-NaN value (or
    # vice versa) fails loudly instead of being silently dropped by the &-mask.
    np.testing.assert_array_equal(np.isfinite(u_cpu), np.isfinite(u_gpu),
                                  err_msg="U statistic finite-mask mismatch")
    finite = np.isfinite(u_cpu)
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

    # GPU is available (module skipif) and v3 CSC-direct is the unconditional
    # default route, so a CSC-sidecar fixture must have taken it — fail hard,
    # never skip, otherwise this test would silently stop proving the route
    # (§B.10 Bg).
    route = _route(gpu)
    assert route == "gpu_csc_v3", f"expected CSC-direct route, got {route!r}"


@pytest.mark.parametrize("cpm_filter", [5.0, 50.0])
def test_pdex_ref_gpu_csc_parity_cpm_filter(cpm_filter: float, tmp_path) -> None:
    """CSC-direct GPU `cpm_filter` matches the CPU path (drops the same genes,
    recomputes the same FDR). Default geometric_mean exercises the second
    arithmetic CSC pseudobulk pass on the device."""
    base = _make_adata()
    cpu = _open_with_csc(tmp_path / "cpu.scx", base)
    gpu = _open_with_csc(tmp_path / "gpu.scx", base)

    cpu_df = pyscx.accel.pdex_ref(
        cpu, "target", reference=REFERENCE, cpm_filter=cpm_filter, device="cpu"
    )
    gpu_df = pyscx.accel.pdex_ref(
        gpu, "target", reference=REFERENCE, cpm_filter=cpm_filter, device="gpu"
    )

    _compare(cpu_df, gpu_df)
    assert _route(gpu) == "gpu_csc_v3", f"expected CSC-direct route, got {_route(gpu)!r}"
