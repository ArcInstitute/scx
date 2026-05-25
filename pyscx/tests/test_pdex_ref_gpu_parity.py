"""GPU parity tests for `pyscx.accel.pdex_ref(device="gpu")`.

Cross-checks the GPU device dispatch against the CPU `pdex_ref` path on the
same fixtures the CPU-vs-upstream-pdex test (`test_pdex_ref_parity.py`) uses.
The CPU path is already pinned to upstream `pdex.pdex(mode="ref")` to fp32
tolerance there, so CPU↔GPU parity here transitively pins GPU to upstream.

Skipped cleanly when:
  * `pyscx` was built without `--features gpu`, OR
  * no CUDA device is visible at runtime.

The tiled merge-sort upgrade removed the prior 8192-cell sort cap;
fixtures of any size now dispatch correctly via the multi-tile path.

Tolerances:
  * U statistic: exact (integer-valued for integer counts).
  * p-value / fdr: atol=1e-9 rtol=1e-6 (sort-order + erfc numerics).
  * means / log2_fold_change / percent_change: atol=1e-4 rtol=1e-4
    (matches the upstream CPU oracle).
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")

import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import pyscx  # noqa: E402

# Reuse the synthetic-fixture builder from the shared fixtures module so
# this test doesn't transitively pull in the pdex importorskip from
# `test_pdex_ref_parity.py` (which is unrelated to GPU dispatch).
from _pdex_fixtures import (  # noqa: E402
    REFERENCE,
    _make_adata,
)


pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)


def _compare_gpu_to_cpu(adata: ad.AnnData, *, geometric_mean: bool, is_log1p: bool, epsilon: float) -> None:
    cpu_df = pyscx.accel.pdex_ref(
        adata,
        "target",
        reference=REFERENCE,
        is_log1p=is_log1p,
        geometric_mean=geometric_mean,
        epsilon=epsilon,
        device="cpu",
    )
    gpu_df = pyscx.accel.pdex_ref(
        adata,
        "target",
        reference=REFERENCE,
        is_log1p=is_log1p,
        geometric_mean=geometric_mean,
        epsilon=epsilon,
        device="gpu",
    )

    cpu_df = cpu_df.sort(["target", "feature"])
    gpu_df = gpu_df.sort(["target", "feature"])
    assert cpu_df.shape == gpu_df.shape, f"shape mismatch: cpu={cpu_df.shape}, gpu={gpu_df.shape}"

    # String + integer columns must match exactly.
    for col in ("target", "feature"):
        assert (cpu_df[col] == gpu_df[col]).all(), f"column '{col}' mismatch"
    for col in ("target_membership", "ref_membership"):
        if col in cpu_df.columns:
            ai = cpu_df[col].cast(pl.Int64).to_numpy()
            bi = gpu_df[col].cast(pl.Int64).to_numpy()
            np.testing.assert_array_equal(ai, bi, err_msg=f"column '{col}' int mismatch")

    # U statistic must match exactly (integer-valued).
    u_cpu = cpu_df["statistic"].cast(pl.Float64).to_numpy()
    u_gpu = gpu_df["statistic"].cast(pl.Float64).to_numpy()
    finite = np.isfinite(u_cpu) & np.isfinite(u_gpu)
    np.testing.assert_allclose(
        u_cpu[finite], u_gpu[finite], atol=1e-6, rtol=0.0,
        err_msg="U statistic must agree exactly within 1e-6",
    )

    # p-value and fdr: loose tolerance (erfc + sort numerics).
    for col in ("p_value", "fdr"):
        a = cpu_df[col].cast(pl.Float64).to_numpy()
        b = gpu_df[col].cast(pl.Float64).to_numpy()
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        mask = np.isfinite(a)
        np.testing.assert_allclose(a[mask], b[mask], atol=1e-9, rtol=1e-6,
                                   err_msg=f"column '{col}' tolerance violation")

    # Means / log2fc / percent: match upstream-CPU tolerance.
    for col in ("target_mean", "ref_mean", "log2_fold_change", "percent_change"):
        a = cpu_df[col].cast(pl.Float64).to_numpy()
        b = gpu_df[col].cast(pl.Float64).to_numpy()
        np.testing.assert_array_equal(np.isfinite(a), np.isfinite(b))
        mask = np.isfinite(a)
        np.testing.assert_allclose(a[mask], b[mask], atol=1e-4, rtol=1e-4,
                                   err_msg=f"column '{col}' tolerance violation")


@pytest.mark.parametrize(
    "geometric_mean,is_log1p,epsilon",
    [
        (True, False, 0.0),
        (False, False, 0.0),
        (True, True, 0.0),
        (False, True, 0.0),
        (True, False, 0.5),
    ],
)
def test_pdex_ref_gpu_parity_dense(geometric_mean: bool, is_log1p: bool, epsilon: float) -> None:
    adata = _make_adata()
    if is_log1p:
        adata.X = np.log1p(adata.X)
    _compare_gpu_to_cpu(adata, geometric_mean=geometric_mean, is_log1p=is_log1p, epsilon=epsilon)


@pytest.mark.parametrize(
    "geometric_mean,is_log1p",
    [
        (True, False),
        (False, True),
    ],
)
def test_pdex_ref_gpu_parity_sparse(geometric_mean: bool, is_log1p: bool) -> None:
    adata = _make_adata()
    adata.X = sp.csr_matrix(np.log1p(adata.X) if is_log1p else adata.X)
    _compare_gpu_to_cpu(adata, geometric_mean=geometric_mean, is_log1p=is_log1p, epsilon=0.0)


def test_pdex_ref_gpu_device_string_variants() -> None:
    """`device="auto"` resolves to GPU when one is visible; `gpu:0` is accepted."""
    adata = _make_adata()
    for dev_str in ("auto", "gpu", "gpu:0"):
        df = pyscx.accel.pdex_ref(adata, "target", reference=REFERENCE, device=dev_str)
        assert df.height > 0, f"device={dev_str!r} produced empty result"
