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

import warnings

import numpy as np
import pytest

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


def _make_unlabelled_fixture(n_unlabelled: int = 30):
    """The 3-group fixture with `n_unlabelled` extra cells carrying a NaN label.

    Returns `(with_nan, filtered)` — the same expression matrix twice, once with
    the unlabelled rows present and once with them physically removed. Any
    correct definition of "rest" makes those two runs identical.
    """
    cats = ["non-targeting", "ko_a", "ko_b"]
    base = _make_adata()
    base.obs["target"] = pd.Categorical(base.obs["target"], categories=cats)
    base.X = sp.csr_matrix(base.X)

    rng = np.random.default_rng(7)
    extra_counts = rng.poisson(4.0, size=(n_unlabelled, base.n_vars)).astype(
        np.float32
    )
    extra = ad.AnnData(
        X=sp.csr_matrix(extra_counts),
        obs=pd.DataFrame(
            {"target": pd.Categorical([np.nan] * n_unlabelled, categories=cats)},
            index=[f"unlabelled_{i}" for i in range(n_unlabelled)],
        ),
        var=base.var.copy(),
    )
    with_nan = ad.concat([base.copy(), extra], join="outer")
    with_nan.obs["target"] = pd.Categorical(with_nan.obs["target"], categories=cats)
    with_nan.X = sp.csr_matrix(with_nan.X)

    filtered = base.copy()
    return with_nan, filtered


def test_rank_genes_groups_rest_excludes_unlabelled_cells():
    """1-vs-rest DE must not count unlabelled cells in "rest".

    `reference="rest"` is the arm the existing unknown-label test above does not
    reach — it runs pairwise against a named reference, which gathers only
    `group ∪ ref` and was always correct.

    In 1-vs-rest the rest *numerator* summed only labelled cells while the rest
    *denominator* was `n_obs`, so every logFC in every group came out inflated
    by `log2(n_obs − n1) − log2(n_labelled − n1)`; the rank pool kept the
    unlabelled cells as competitors, so the z-scores were off too. The oracle is
    the same matrix with those rows physically deleted.
    """
    with_nan, filtered = _make_unlabelled_fixture()
    assert with_nan.n_obs > filtered.n_obs

    pyscx.accel.rank_genes_groups(with_nan, "target", reference="rest", device="cpu")
    pyscx.accel.rank_genes_groups(filtered, "target", reference="rest", device="cpu")

    got = _scores_and_pvals_by_gene(with_nan)
    want = _scores_and_pvals_by_gene(filtered)
    assert got.keys() == want.keys()
    for key in want:
        np.testing.assert_allclose(
            got[key],
            want[key],
            rtol=1e-9,
            atol=1e-12,
            err_msg=(
                f"1-vs-rest DE for {key} differs from the physically-filtered "
                f"run — unlabelled cells are leaking into 'rest' or the rank pool"
            ),
        )


def test_rank_genes_groups_rest_with_unlabelled_matches_scanpy_on_log1p():
    """End-to-end oracle: scanpy on the filtered matrix.

    Runs on log1p data deliberately. scanpy's `rank_genes_groups` applies
    `expm1` to the group means unconditionally (the `uns["log1p"]["base"]` entry
    only rescales it), so its logFC is only comparable to ours on log-space
    input — on raw counts scanpy warns and the two formulas legitimately differ.
    scanpy is run on the *filtered* matrix on purpose: pyscx's rule is that
    unlabelled cells take no part, and this pins that rule against scanpy's
    numbers for the labelled cells alone. scanpy 1.12 itself keeps NaN-labelled
    cells in its 1-vs-rest pool, so scanpy-on-NaN is a different computation —
    a pre-existing divergence, tracked separately.
    """
    sc = pytest.importorskip("scanpy")

    with_nan, filtered = _make_unlabelled_fixture()
    sc.pp.log1p(with_nan)
    sc.pp.log1p(filtered)

    pyscx.accel.rank_genes_groups(with_nan, "target", reference="rest", device="cpu")
    sc.tl.rank_genes_groups(filtered, "target", method="wilcoxon")

    got = _scores_and_pvals_by_gene(with_nan)
    want = _scores_and_pvals_by_gene(filtered)
    assert got.keys() == want.keys()
    # Looser than the self-parity test above (1e-9/1e-12), which compares two
    # runs of the same kernel. Here the two sides are different implementations:
    # scanpy accumulates its means and its tie correction in a different order
    # and partly in float32, so agreement is to float tolerance, not to the bit.
    for key in want:
        np.testing.assert_allclose(
            got[key], want[key], rtol=1e-5, atol=1e-6,
            err_msg=f"scanpy parity broken for {key}",
        )


def test_rank_genes_groups_warns_about_unlabelled_cells():
    """Dropping rows silently is the part that made this invisible."""
    with_nan, _ = _make_unlabelled_fixture(n_unlabelled=12)
    with pytest.warns(UserWarning, match=r"12 of \d+ cells have no group label"):
        pyscx.accel.rank_genes_groups(
            with_nan, "target", reference="rest", device="cpu"
        )


def test_rank_genes_groups_fully_labelled_emits_no_unlabelled_warning():
    """The warning must not fire on ordinary, fully-annotated input."""
    _, filtered = _make_unlabelled_fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        pyscx.accel.rank_genes_groups(
            filtered, "target", reference="rest", device="cpu"
        )
