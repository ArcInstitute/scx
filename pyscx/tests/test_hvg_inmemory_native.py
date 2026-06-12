"""In-memory (scipy/dense X) native HVG path.

`pyscx.accel.highly_variable_genes` runs the scx-native streaming kernel on a
materialized scipy/dense `adata.X` (wrapped in a single-shard `ShardSource`) for
flavor seurat_v3 / seurat_v3_paper / seurat. This file covers:

- B1 regression: a multi-batch in-memory seurat_v3 run with a singular per-batch
  LOESS fit does NOT raise (the in-memory path inherits the backed path's
  per-batch tolerance — the original Tier-2 crash was the scanpy-delegation path
  having no such tolerance).
- Parity: in-memory native HVG matches the backed native HVG bit-for-bit
  (seurat_v3 with/without batch_key, and seurat on log-normalized data).
- GPU (device="gpu") on a materialized X runs the native kernel (no scanpy
  delegation) and its gene set overlaps the in-memory CPU path.
- subset=True on an in-memory X slices the AnnData to n_top_genes columns.
"""
import warnings

import numpy as np
import pytest


def _gpu_available() -> bool:
    try:
        import pyscx

        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — GPU in-memory HVG test is GPU-only",
)


def _patch_loess_to_raise_on_first_batch(monkeypatch):
    """Force `skmisc.loess.loess(...).fit()` to raise ValueError on the first
    call, then behave normally — mirrors a singular per-batch Census fit."""
    import skmisc.loess as loess_mod

    real_loess = loess_mod.loess
    state = {"n_calls": 0}

    class FailingLoess:
        def __init__(self, *args, **kwargs):
            self._inner = real_loess(*args, **kwargs)

        def __getattr__(self, name):
            if name == "fit":
                state["n_calls"] += 1
                if state["n_calls"] == 1:
                    def _raise():
                        raise ValueError(
                            "b'There are other near singularities as well. "
                            "0.22764'"
                        )

                    return _raise
            return getattr(self._inner, name)

    monkeypatch.setattr(loess_mod, "loess", FailingLoess)


def test_inmemory_seurat_v3_batch_key_does_not_raise(
    synthetic_adata, scx_from_adata, monkeypatch
):
    """B1 regression: in-memory (eager) scipy X + batch_key with a singular
    per-batch LOESS fit must NOT raise; HVG completes and populates var."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_inmem_batch.scx")
    adata = pyscx.open(path).to_anndata()  # eager → scipy CSR
    import scipy.sparse as sp

    assert sp.issparse(adata.X), "expected an in-memory scipy CSR X"
    _patch_loess_to_raise_on_first_batch(monkeypatch)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.accel.highly_variable_genes(
            adata,
            n_top_genes=10,
            flavor="seurat_v3",
            batch_key="batch",
            device="cpu",
        )

    # No scanpy delegation on the native path.
    assert not [
        w for w in caught if "scanpy.pp.highly_variable_genes" in str(w.message)
    ]
    # The singular batch was caught and surfaced in the coalesced summary
    # warning (per-batch tolerance; F10 dedup).
    assert [
        w for w in caught if "skmisc.loess fit failed on" in str(w.message)
    ], "expected the per-batch singularity to be caught and warned, not raised"
    assert "highly_variable" in adata.var.columns
    assert int(adata.var["highly_variable"].sum()) == 10


def test_inmemory_native_matches_backed(synthetic_adata, scx_from_adata):
    """Parity: in-memory native HVG == backed native HVG (same single-shard
    data → identical accumulation order → identical mask & rank)."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_parity_inmem.scx")

    backed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(
        backed, n_top_genes=15, flavor="seurat_v3", batch_key="batch", device="cpu"
    )

    eager = pyscx.open(path).to_anndata()
    pyscx.accel.highly_variable_genes(
        eager, n_top_genes=15, flavor="seurat_v3", batch_key="batch", device="cpu"
    )

    np.testing.assert_array_equal(
        np.asarray(backed.var["highly_variable"]),
        np.asarray(eager.var["highly_variable"]),
        err_msg="in-memory native HVG mask differs from the backed path",
    )
    rb = np.asarray(backed.var["highly_variable_rank"], dtype=float)
    re = np.asarray(eager.var["highly_variable_rank"], dtype=float)
    np.testing.assert_array_equal(np.isnan(rb), np.isnan(re))
    np.testing.assert_array_equal(rb[~np.isnan(rb)], re[~np.isnan(re)])


def test_inmemory_native_single_batch_matches_backed(synthetic_adata, scx_from_adata):
    """Parity without batch_key (single global fit)."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_parity_inmem_nobatch.scx")

    backed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(backed, n_top_genes=15, flavor="seurat_v3", device="cpu")

    eager = pyscx.open(path).to_anndata()
    pyscx.accel.highly_variable_genes(eager, n_top_genes=15, flavor="seurat_v3", device="cpu")

    np.testing.assert_array_equal(
        np.asarray(backed.var["highly_variable"]),
        np.asarray(eager.var["highly_variable"]),
    )


def test_inmemory_native_seurat_matches_backed(synthetic_adata, scx_from_adata):
    """flavor="seurat" parity: in-memory native == backed native.

    seurat expects log-normalized data, so normalize+log1p once in memory and
    persist *that* to SCX. Opening the same file backed and eager guarantees
    byte-identical kernel inputs (both decode the same on-disk CSR), so the
    native seurat path must agree exactly — this covers the dispersion-binning
    path that test_inmemory_native_matches_backed (seurat_v3) does not.
    """
    import pyscx
    import scanpy as sc

    raw_path = scx_from_adata(synthetic_adata, "hvg_seurat_raw.scx")
    ref = pyscx.open(raw_path).to_anndata()
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)

    # Persist the log-normalized matrix so backed and eager read identical data.
    norm_path = scx_from_adata(ref, "hvg_seurat_norm.scx")

    backed = pyscx.open(norm_path).to_anndata(backed=True)
    pyscx.accel.highly_variable_genes(
        backed, n_top_genes=15, flavor="seurat", n_bins=5, device="cpu"
    )

    eager = pyscx.open(norm_path).to_anndata()
    import scipy.sparse as sp

    assert sp.issparse(eager.X), "expected an in-memory scipy CSR X"
    pyscx.accel.highly_variable_genes(
        eager, n_top_genes=15, flavor="seurat", n_bins=5, device="cpu"
    )

    np.testing.assert_array_equal(
        np.asarray(backed.var["highly_variable"]),
        np.asarray(eager.var["highly_variable"]),
        err_msg="in-memory native seurat HVG mask differs from the backed path",
    )
    np.testing.assert_allclose(
        np.asarray(backed.var["dispersions_norm"], dtype=float),
        np.asarray(eager.var["dispersions_norm"], dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


@gpu_only
def test_inmemory_gpu_seurat_v3_runs_native_and_matches_cpu(
    synthetic_adata, scx_from_adata
):
    """device="gpu" on a materialized scipy X runs the GPU native kernel.

    `InMemoryCsrSource` is `Sync`, so the GPU batched kernels read the single
    in-memory shard directly — previously this combination delegated to scanpy.
    No scanpy delegation must fire, and the selected gene set strongly overlaps
    the in-memory CPU path (lax overlap, not exact: GPU atomicAdd f64 reductions
    perturb marginal-gene ranks — cf. test_hvg_gpu_batch_key.py).
    """
    import pyscx
    import scipy.sparse as sp

    path = scx_from_adata(synthetic_adata, "hvg_inmem_gpu.scx")

    eager_gpu = pyscx.open(path).to_anndata()
    assert sp.issparse(eager_gpu.X), "expected an in-memory scipy CSR X"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.accel.highly_variable_genes(
            eager_gpu,
            n_top_genes=15,
            flavor="seurat_v3",
            batch_key="batch",
            device="gpu",
        )
    assert not [
        w for w in caught if "scanpy.pp.highly_variable_genes" in str(w.message)
    ], "GPU in-memory path must run native, not delegate to scanpy"
    assert "highly_variable" in eager_gpu.var.columns
    assert int(eager_gpu.var["highly_variable"].sum()) == 15
    gpu_hvg = set(eager_gpu.var.index[eager_gpu.var["highly_variable"].to_numpy()])

    eager_cpu = pyscx.open(path).to_anndata()
    pyscx.accel.highly_variable_genes(
        eager_cpu,
        n_top_genes=15,
        flavor="seurat_v3",
        batch_key="batch",
        device="cpu",
    )
    cpu_hvg = set(eager_cpu.var.index[eager_cpu.var["highly_variable"].to_numpy()])

    overlap = len(gpu_hvg & cpu_hvg) / max(len(cpu_hvg), 1)
    assert overlap >= 0.70, (
        f"GPU vs CPU in-memory HVG overlap {overlap:.2f} < 0.70 "
        f"(gpu={sorted(gpu_hvg)}, cpu={sorted(cpu_hvg)})"
    )


def test_inmemory_subset_true_slices_adata(synthetic_adata, scx_from_adata):
    """subset=True on an in-memory X slices the AnnData to n_top_genes columns
    while preserving the written var result columns."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "hvg_inmem_subset.scx")
    eager = pyscx.open(path).to_anndata()
    n_top = 12
    pyscx.accel.highly_variable_genes(
        eager, n_top_genes=n_top, flavor="seurat_v3", subset=True, device="cpu"
    )
    assert eager.n_vars == n_top
    assert eager.X.shape[1] == n_top
    # The result columns survive the in-place subset.
    assert "highly_variable" in eager.var.columns
    assert bool(eager.var["highly_variable"].all())
