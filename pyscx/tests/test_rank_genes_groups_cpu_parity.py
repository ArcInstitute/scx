"""CPU regression tests for `pyscx.accel.rank_genes_groups(device="cpu")`.

The Wilcoxon CPU sparse path shares the same `scx_engine::project_csr_row`
precondition as `pdex_ref`: column indices in each CSR row must be sorted
ascending. Inputs with `has_sorted_indices == False` (e.g.
`pbmc10k.h5ad`'s `adata.X`) silently produced wrong results before the
`ensure_csr` fix at the pyscx CPU dispatch boundary.

This test mirrors the pdex regression test in
`test_pdex_ref_parity.py::test_pdex_ref_cpu_unsorted_scipy_csr_matches_sorted`,
swapping `pdex_ref` for `rank_genes_groups`.
"""

from __future__ import annotations

import numpy as np
import pytest

import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import pyscx  # noqa: E402

# Reuse the fixture + helper from the pdex parity test (90 cells × 15 genes,
# 3 groups with deliberate per-group DE signal).
from test_pdex_ref_parity import (  # noqa: E402
    _make_adata,
    _csr_with_descending_indices,
    REFERENCE,
)


def _scores_and_pvals_by_gene(adata: ad.AnnData) -> dict:
    """Pull rank_genes_groups results out of adata.uns and key by
    (group, gene) so we can compare two runs row-by-row.
    """
    rgg = adata.uns["rank_genes_groups"]
    groups = list(rgg["names"].dtype.names)
    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    for g in groups:
        names = rgg["names"][g]
        scores = rgg["scores"][g]
        pvals = rgg["pvals"][g]
        logfc = rgg["logfoldchanges"][g]
        for i, name in enumerate(names):
            out[(str(g), str(name))] = (
                float(scores[i]),
                float(pvals[i]),
                float(logfc[i]),
            )
    return out


def test_rank_genes_groups_cpu_unsorted_scipy_csr_matches_sorted():
    """Wilcoxon CPU sparse path must produce identical results on an
    unsorted scipy CSR (`has_sorted_indices == False`) as on the
    canonical sorted equivalent.

    Before the `ensure_csr` boundary fix, the unsorted-CSR run collapsed
    every gene's U-statistic to the trivial `n_g·n_ref/2` value and
    every p-value to 1.0 because `scx_engine::project_csr_row`'s
    monotonic merge-scan dropped most column entries.
    """
    adata_sorted = _make_adata()
    adata_sorted.X = sp.csr_matrix(adata_sorted.X)
    adata_sorted.X.sort_indices()
    assert adata_sorted.X.has_sorted_indices

    adata_unsorted = adata_sorted.copy()
    adata_unsorted.X = _csr_with_descending_indices(adata_sorted)
    assert not adata_unsorted.X.has_sorted_indices

    # rank_genes_groups writes results into adata.uns["rank_genes_groups"];
    # operate on copies so the two runs don't clobber each other.
    a_sorted = adata_sorted.copy()
    a_unsorted = adata_unsorted.copy()

    pyscx.accel.rank_genes_groups(a_sorted, "target", reference=REFERENCE, device="cpu")
    pyscx.accel.rank_genes_groups(
        a_unsorted, "target", reference=REFERENCE, device="cpu"
    )

    sorted_results = _scores_and_pvals_by_gene(a_sorted)
    unsorted_results = _scores_and_pvals_by_gene(a_unsorted)

    assert set(sorted_results.keys()) == set(
        unsorted_results.keys()
    ), "sorted vs unsorted result keysets diverge"

    for key in sorted_results:
        s_score, s_p, s_lfc = sorted_results[key]
        u_score, u_p, u_lfc = unsorted_results[key]
        np.testing.assert_allclose(
            u_score, s_score, atol=0.0, rtol=0.0, err_msg=f"score mismatch at {key}"
        )
        np.testing.assert_allclose(
            u_p, s_p, atol=0.0, rtol=0.0, err_msg=f"pval mismatch at {key}"
        )
        # logfoldchange can be ±inf for zero-mean genes — compare on
        # finite mask first, then assert exact equality on the rest.
        if np.isfinite(s_lfc) or np.isfinite(u_lfc):
            assert np.isfinite(s_lfc) == np.isfinite(u_lfc), (
                f"finite-mask mismatch in logfoldchange at {key}: "
                f"sorted={s_lfc}, unsorted={u_lfc}"
            )
            if np.isfinite(s_lfc):
                np.testing.assert_allclose(
                    u_lfc, s_lfc, atol=0.0, rtol=0.0, err_msg=f"logfc at {key}"
                )

    # Sanity: the fixture has real signal — at least one gene should
    # have p < 0.99. Catches the "trivial U on every gene" failure mode
    # in case both runs regress together.
    n_nontrivial = sum(1 for _, p, _ in sorted_results.values() if p < 0.99)
    assert n_nontrivial > 0, (
        f"fixture lost DE signal: all {len(sorted_results)} p-values ≥ 0.99 — "
        f"either _make_adata changed or rank_genes_groups CPU is broken in a "
        f"different way"
    )

    # Caller's AnnData must not be mutated (ensure_csr called with in_place=False).
    assert not adata_unsorted.X.has_sorted_indices, (
        "ensure_csr(in_place=False) mutated caller's CSR — "
        "has_sorted_indices flipped to True after rank_genes_groups"
    )
