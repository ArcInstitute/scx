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

import anndata as ad
import pandas as pd
import scipy.sparse as sp

import pyscx

# Reuse the fixture + helper from the shared fixtures module
# (90 cells × 15 genes, 3 groups with deliberate per-group DE signal).
# Importing from `_pdex_fixtures` instead of `test_pdex_ref_parity` keeps
# this CPU regression test out of the polars / pdex importorskip cascade.
from _pdex_fixtures import (
    REFERENCE,
    _csr_with_descending_indices,
    _make_adata,
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
    # `a_unsorted` is the deep-copy that was actually handed to
    # `rank_genes_groups` — that's the object the function could have
    # mutated. Checking `adata_unsorted` (the pre-copy original) would
    # always pass regardless of what the function did.
    assert not a_unsorted.X.has_sorted_indices, (
        "ensure_csr(in_place=False) mutated caller's CSR — "
        "has_sorted_indices flipped to True after rank_genes_groups"
    )


def _tie_adata() -> ad.AnnData:
    """40 cells × 8 genes, two groups. Genes 1/3/5/6 are constant (so their
    tie-corrected Wilcoxon score collapses to an exact 0.0 tie); genes
    0/2/4/7 carry real per-group signal. Used to pin tie-break ordering.
    """
    n_obs, n_vars = 40, 8
    rng = np.random.default_rng(0)
    X = np.zeros((n_obs, n_vars), dtype=np.float32)
    in_g0 = np.arange(n_obs) < 20
    # Signal genes: separable between the two groups.
    X[in_g0, 0] = 10.0 + rng.random(20)
    X[~in_g0, 0] = 1.0
    X[in_g0, 2] = 1.0
    X[~in_g0, 2] = 9.0 + rng.random(20)
    X[in_g0, 4] = 8.0 + rng.random(20)
    X[~in_g0, 4] = 2.0
    X[in_g0, 7] = 2.0
    X[~in_g0, 7] = 7.0 + rng.random(20)
    # Constant genes → exact score ties.
    X[:, 1] = 5.0
    X[:, 3] = 0.0
    X[:, 5] = 3.0
    X[:, 6] = 7.0
    obs = {"target": ["g0" if i < 20 else "g1" for i in range(n_obs)]}
    var_names = [f"gene_{i}" for i in range(n_vars)]
    a = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(obs, index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=var_names),
    )
    return a


def test_rank_genes_groups_cpu_tie_order_is_ascending_var_index():
    """On equal scores, scx breaks ties by ascending var (gene) index —
    matching scanpy's stable argsort. With `tie_correct=True` the constant
    genes share an exact 0.0 score, so their relative order must follow
    `adata.var_names` position. Also pins chunk-size determinism: the
    single-chunk and gene-chunked CPU paths must produce identical orders.
    """
    adata = _tie_adata()
    var_index = {name: i for i, name in enumerate(adata.var_names)}

    a_full = adata.copy()
    a_chunked = adata.copy()
    pyscx.accel.rank_genes_groups(
        a_full, "target", reference="rest", tie_correct=True, device="cpu"
    )
    pyscx.accel.rank_genes_groups(
        a_chunked,
        "target",
        reference="rest",
        tie_correct=True,
        gene_chunk_size=2,
        device="cpu",
    )

    rgg_full = a_full.uns["rank_genes_groups"]
    rgg_chunked = a_chunked.uns["rank_genes_groups"]
    groups = list(rgg_full["names"].dtype.names)

    for g in groups:
        names_full = [str(n) for n in rgg_full["names"][g]]
        names_chunked = [str(n) for n in rgg_chunked["names"][g]]
        # Chunk-size invariance: identical ranking regardless of chunking.
        assert names_full == names_chunked, (
            f"group {g}: name order differs between single-chunk and "
            f"gene_chunk_size=2:\n  full={names_full}\n  chunked={names_chunked}"
        )

        # Among tied genes (score ≈ 0.0), var index must be strictly ascending.
        scores = rgg_full["scores"][g]
        tied = [
            var_index[str(n)]
            for n, s in zip(rgg_full["names"][g], scores)
            if abs(float(s)) < 1e-9
        ]
        assert tied == sorted(tied), (
            f"group {g}: tied genes not in ascending var-index order: {tied}"
        )
        assert len(tied) >= 2, f"group {g}: fixture lost its score ties"


def test_rank_genes_groups_cpu_drops_unknown_group_labels():
    """Cells whose groupby label is missing (NaN → "nan" after astype(str))
    or otherwise outside the category set must be DROPPED from DE, not folded
    into group 0.

    Regression for the `.unwrap_or(&0)` contamination bug at
    pyscx/src/accel/de.rs:185 (2026-06-21 review): unknown labels now map to
    the out-of-range sentinel `unique_groups.len()` so the Wilcoxon kernels
    skip them, matching `pdex_ref`/`resolve_groups_and_reference`.

    Construction: take the clean 3-group fixture, then append phantom cells
    with a NaN group label and extreme expression. Under the old code those
    cells landed in group index 0 and shifted that group's scores/pvals/logFC;
    with the fix they are dropped, so DE is byte-for-byte identical to the
    clean run for every group.
    """
    base = _make_adata()
    # Explicit categorical so the injected NaN label is NOT a category — keeps
    # unique_groups at the 3 real groups (no spurious "nan" group).
    cats = ["non-targeting", "ko_a", "ko_b"]
    base.obs["target"] = pd.Categorical(base.obs["target"], categories=cats)
    base.X = sp.csr_matrix(base.X)

    a_clean = base.copy()
    pyscx.accel.rank_genes_groups(
        a_clean, "target", reference=REFERENCE, device="cpu"
    )
    clean = _scores_and_pvals_by_gene(a_clean)

    # Phantom cells: missing group label + extreme counts. Under the bug these
    # contaminate the first group's DE; with the fix they are dropped.
    n_extra = 30
    n_vars = base.n_vars
    extra_counts = np.full((n_extra, n_vars), 1000.0, dtype=np.float32)
    extra_obs = pd.DataFrame(
        {"target": pd.Categorical([np.nan] * n_extra, categories=cats)},
        index=[f"phantom_{i}" for i in range(n_extra)],
    )
    extra = ad.AnnData(
        X=sp.csr_matrix(extra_counts), obs=extra_obs, var=base.var.copy()
    )
    contaminated = ad.concat([base.copy(), extra], join="outer")
    # ad.concat may relax the categorical / densify; pin both back.
    contaminated.obs["target"] = pd.Categorical(
        contaminated.obs["target"], categories=cats
    )
    contaminated.X = sp.csr_matrix(contaminated.X)

    pyscx.accel.rank_genes_groups(
        contaminated, "target", reference=REFERENCE, device="cpu"
    )
    got = _scores_and_pvals_by_gene(contaminated)

    assert got.keys() == clean.keys(), (
        "group/gene set changed — phantom NaN-label cells leaked a group or "
        "altered the gene ranking"
    )
    for key in clean:
        np.testing.assert_allclose(
            got[key],
            clean[key],
            rtol=1e-6,
            atol=1e-6,
            err_msg=(
                f"DE for {key} differs after appending NaN-label cells — they "
                f"were folded into a real group instead of being dropped"
            ),
        )
