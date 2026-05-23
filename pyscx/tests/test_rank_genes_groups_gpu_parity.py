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

# Reuse the fixture from the pdex parity test (90 cells × 15 genes, 3 groups).
from test_pdex_ref_parity import _make_adata, REFERENCE  # noqa: E402


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
