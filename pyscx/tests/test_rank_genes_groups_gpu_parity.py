"""GPU parity tests for `pyscx.accel.rank_genes_groups(device="gpu")`.

Cross-checks the GPU Wilcoxon path (`scx_accel::wilcoxon_rank_sum_gpu_*`)
against the CPU path on a small fixture. Both ref-mode and 1-vs-rest are
exercised; tie correction is on (matches the harder code path).

Skipped cleanly when `pyscx.accel.gpu_available()` is `False`.

Tolerances:
  * scores (z): atol=1e-6 (same formula, host-computed from device U + tie).
  * pvals: atol=1e-9 rtol=1e-6.
  * logfoldchanges: atol=1e-4 rtol=1e-4 (counts are integer; fp32→f64 conv).

The per-group gene order can differ between CPU and GPU when scores tie —
we compare on a per-gene-name dict, not on a positional index.
"""

from __future__ import annotations

import numpy as np
import pytest

import anndata as ad  # noqa: E402

import pyscx  # noqa: E402

# Reuse the synthetic-fixture builder from the shared fixtures module
# (90 cells × 15 genes, 3 groups). Avoids the polars / pdex importorskips
# in `test_pdex_ref_parity.py` — neither is needed for the GPU Wilcoxon path.
from _pdex_fixtures import _make_adata, REFERENCE  # noqa: E402


pytestmark = pytest.mark.skipif(
    not pyscx.accel.gpu_available(),
    reason="GPU not available (pyscx not built with gpu feature, or no CUDA device)",
)


def _result_to_gene_dict(adata: ad.AnnData):
    """Pull the rank_genes_groups result out of adata.uns and key by gene name."""
    rgg = adata.uns["rank_genes_groups"]
    groups = list(rgg["names"].dtype.names)
    out = {}
    for g in groups:
        names = rgg["names"][g]
        scores = rgg["scores"][g]
        pvals = rgg["pvals"][g]
        logfc = rgg["logfoldchanges"][g]
        out[g] = {
            str(name): (float(scores[i]), float(pvals[i]), float(logfc[i]))
            for i, name in enumerate(names)
        }
    return out


def _compare_results(cpu, gpu) -> None:
    assert set(cpu.keys()) == set(gpu.keys()), f"group set mismatch: {set(cpu)} vs {set(gpu)}"
    for g in cpu:
        cpu_g = cpu[g]
        gpu_g = gpu[g]
        assert set(cpu_g.keys()) == set(gpu_g.keys()), f"gene set mismatch in group {g}"
        for gene, (s_c, p_c, l_c) in cpu_g.items():
            s_g, p_g, l_g = gpu_g[gene]
            if np.isfinite(s_c) and np.isfinite(s_g):
                assert abs(s_c - s_g) < 1e-6 or abs(s_c - s_g) / max(abs(s_c), 1e-12) < 1e-6, (
                    f"score mismatch group={g} gene={gene}: cpu={s_c}, gpu={s_g}"
                )
            assert abs(p_c - p_g) < 1e-9 or abs(p_c - p_g) / max(p_c, 1e-12) < 1e-6, (
                f"pval mismatch group={g} gene={gene}: cpu={p_c}, gpu={p_g}"
            )
            if np.isfinite(l_c) and np.isfinite(l_g):
                assert abs(l_c - l_g) < 1e-4 or abs(l_c - l_g) / max(abs(l_c), 1e-9) < 1e-4, (
                    f"logfc mismatch group={g} gene={gene}: cpu={l_c}, gpu={l_g}"
                )


def _partially_labelled_adata():
    """The shared fixture with a tenth of its cells carrying a NaN label.

    Also plants a gene that is nonzero *only* in those cells: to any route that
    leaves them out of the pool it is identical to an all-zero gene, so the
    fixture separates the two pools rather than merely containing both.
    """
    import pandas as pd
    import scipy.sparse as sp

    adata = _make_adata()
    cats = list(pd.unique(adata.obs["target"]))
    labels = adata.obs["target"].astype(object).to_numpy()
    unlabelled = list(range(0, adata.n_obs, 10))
    for i in unlabelled:
        labels[i] = None
    adata.obs["target"] = pd.Categorical(labels, categories=cats)

    X = (adata.X.tolil() if sp.issparse(adata.X) else sp.lil_matrix(adata.X))
    X[:, 0] = 0
    for i in unlabelled:
        X[i, 0] = 5.0
    adata.X = X.tocsr()
    return adata


@pytest.mark.parametrize("tie_correct", [True, False])
def test_rank_genes_groups_gpu_parity_one_vs_rest(tie_correct: bool) -> None:
    adata_cpu = _make_adata()
    adata_gpu = adata_cpu.copy()

    pyscx.accel.rank_genes_groups(
        adata_cpu, "target", reference="rest", tie_correct=tie_correct, device="cpu"
    )
    pyscx.accel.rank_genes_groups(
        adata_gpu, "target", reference="rest", tie_correct=tie_correct, device="gpu"
    )
    _compare_results(_result_to_gene_dict(adata_cpu), _result_to_gene_dict(adata_gpu))


@pytest.mark.parametrize("tie_correct", [True, False])
def test_rank_genes_groups_gpu_parity_vs_reference(tie_correct: bool) -> None:
    adata_cpu = _make_adata()
    adata_gpu = adata_cpu.copy()

    pyscx.accel.rank_genes_groups(
        adata_cpu, "target", reference=REFERENCE, tie_correct=tie_correct, device="cpu"
    )
    pyscx.accel.rank_genes_groups(
        adata_gpu, "target", reference=REFERENCE, tie_correct=tie_correct, device="gpu"
    )
    _compare_results(_result_to_gene_dict(adata_cpu), _result_to_gene_dict(adata_gpu))


def test_rank_genes_groups_gpu_rejects_csc() -> None:
    """`device='gpu'` with `prefer_format='csc'` must error explicitly (v1)."""
    adata = _make_adata()
    with pytest.raises(RuntimeError, match="csc"):
        pyscx.accel.rank_genes_groups(
            adata, "target", reference=REFERENCE, prefer_format="csc", device="gpu"
        )


@pytest.mark.parametrize("tie_correct", [True, False])
def test_rank_genes_groups_gpu_parity_one_vs_rest_partially_labelled(
    tie_correct: bool,
) -> None:
    """X9 on the GPU: unlabelled cells are in the pool and in every "rest".

    The GPU reaches that differently from the CPU — slot 0 of the v3 main
    table, empty in 1-vs-rest until X9, carries the unlabelled cells purely so
    the per-slot pseudobulk sums cover every cell. Get that wrong and the ranks
    are right while the logFC numerator is short, which the fully-labelled arms
    above cannot see.
    """
    import warnings

    adata_cpu = _partially_labelled_adata()
    adata_gpu = adata_cpu.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pyscx.accel.rank_genes_groups(
            adata_cpu, "target", reference="rest", tie_correct=tie_correct, device="cpu"
        )
        pyscx.accel.rank_genes_groups(
            adata_gpu, "target", reference="rest", tie_correct=tie_correct, device="gpu"
        )
    cpu, gpu = _result_to_gene_dict(adata_cpu), _result_to_gene_dict(adata_gpu)
    _compare_results(cpu, gpu)

    # The fixture has to be able to tell the two pools apart, or the parity
    # above would hold with both sides dropping the cells.
    planted = str(adata_cpu.var_names[0])
    for g, per_gene in cpu.items():
        assert np.isfinite(per_gene[planted][0]) and per_gene[planted][0] != 0.0, (
            f"group {g}: the gene nonzero only in unlabelled cells scored 0 — "
            f"those cells never reached the pool"
        )
