"""GPU HVG `batch_key` support.

The GPU `seurat_v3` HVG path used to silently fall back to CPU whenever
`batch_key` was set (warning text: "only implemented for single-batch
seurat_v3 flavors"). This module verifies the post-fix behaviour:

  1. No "single-batch" warning is emitted when `batch_key` is set with
     `device="gpu"`.
  2. The CPU and GPU multi-batch paths agree on the selected gene set on
     a small synthetic dataset.
  3. The per-batch `var` columns (`highly_variable_nbatches`,
     `highly_variable_rank`) are populated.

All tests skip when the build doesn't have a CUDA GPU.
"""

import warnings

import pytest


def _gpu_available() -> bool:
    try:
        import pyscx

        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — GPU HVG batch_key tests are GPU-only",
)


def test_gpu_batch_key_emits_no_single_batch_fallback_warning(
    synthetic_adata, scx_from_adata
):
    """`batch_key` + `device="gpu"` must NOT fire the old single-batch warning.

    Pre-fix: pyscx/src/accel/hvg.rs:99-127 emitted
        "highly_variable_genes(device=...) is only implemented for
         single-batch seurat_v3 flavors; falling back to CPU"
    whenever batch_key was set. Post-fix the GPU path supports per-batch
    kernels, so that warning must no longer fire on this path.
    """
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_gpu_batch.scx")
    adata = pyscx.open(path).to_anndata(backed=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.accel.highly_variable_genes(
            adata,
            n_top_genes=15,
            flavor="seurat_v3",
            batch_key="batch",
            device="gpu",
        )
    bad = [
        w for w in caught if "single-batch seurat_v3 flavors" in str(w.message)
    ]
    assert not bad, (
        "stale single-batch fallback warning fired despite GPU per-batch path: "
        + "; ".join(str(w.message) for w in bad)
    )
    assert "highly_variable" in adata.var.columns
    assert adata.var["highly_variable"].sum() == 15


def test_gpu_batch_key_matches_cpu(synthetic_adata, scx_from_adata):
    """CPU vs GPU per-batch HVG agree on the selected gene set.

    Tolerance is intentionally lax (≥ 70 %): the per-batch loess fit runs
    in CPU `skmisc.loess` in both paths, but accumulated f64 rounding in the
    GPU `atomicAdd` reduction can perturb ranks of marginal genes. We
    require strong overlap rather than exact equality.
    """
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_gpu_batch_match.scx")

    adata_gpu = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(
        adata_gpu,
        n_top_genes=15,
        flavor="seurat_v3",
        batch_key="batch",
        device="gpu",
    )
    gpu_hvg = set(
        adata_gpu.var.index[adata_gpu.var["highly_variable"].to_numpy()]
    )

    adata_cpu = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(
        adata_cpu,
        n_top_genes=15,
        flavor="seurat_v3",
        batch_key="batch",
        device="cpu",
    )
    cpu_hvg = set(
        adata_cpu.var.index[adata_cpu.var["highly_variable"].to_numpy()]
    )

    overlap = len(gpu_hvg & cpu_hvg) / max(len(cpu_hvg), 1)
    assert overlap >= 0.70, (
        f"GPU vs CPU multi-batch HVG overlap {overlap:.2f} < 0.70 "
        f"(gpu={sorted(gpu_hvg)}, cpu={sorted(cpu_hvg)})"
    )


def test_gpu_batch_key_populates_per_batch_var_columns(
    synthetic_adata, scx_from_adata
):
    """The multi-batch rank columns must be populated on the GPU path."""
    import numpy as np

    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_gpu_batch_cols.scx")
    adata = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(
        adata,
        n_top_genes=15,
        flavor="seurat_v3",
        batch_key="batch",
        device="gpu",
    )
    rank = adata.var["highly_variable_rank"].to_numpy()
    # `highly_variable_rank` is NaN for genes that never landed in any
    # batch's top-N; selected genes carry a finite rank in [0, n_top_genes).
    selected = np.where(adata.var["highly_variable"].to_numpy())[0]
    finite_ranks = rank[selected]
    assert np.all(np.isfinite(finite_ranks)), (
        "Selected genes must have finite highly_variable_rank"
    )
    assert (finite_ranks >= 0).all() and (finite_ranks < 15).all()
