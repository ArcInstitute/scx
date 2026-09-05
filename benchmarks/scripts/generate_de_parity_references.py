#!/usr/bin/env python
"""Generate the pinned DE parity reference values for `scx-accel` (ORG-7.21-4).

Prints paste-ready Rust literals for
`scx-accel/src/diffexp/wilcoxon_reference_tests.rs`, plus the observed
max |Δ| per field for the scanpy comparison in `pyscx/tests/test_accel.py`.

Run it from the repo root, in the uv venv — never system Python:

    .venv/bin/python benchmarks/scripts/generate_de_parity_references.py

Why this script exists
----------------------
Every documented DE parity claim used to be backed by an overlap or a
tolerance that could not detect the finding it was written for: an 80 %
top-20 *name* overlap, and `|Δlog2FC| < 0.5` (a 1.4x fold-change gap passes).
`scx-accel/src/eval_metrics/clustering.rs` is the model this replaces them
with — it pins sklearn's exact `0.298792458170890`, so the Rust suite alone
is a real gate with no Python installed.

The fixture is the single source of truth: this script emits both the input
matrix and the expected values, so the two cannot drift.

The one convention that matters
-------------------------------
`scipy.stats.mannwhitneyu(..., method="asymptotic")` ALWAYS applies the
Σ(t³−t) tie correction. SCX's `tie_correct` **defaults to false**, matching
scanpy's default — `scanpy.tl.rank_genes_groups` takes a `tie_correct`
parameter whose default is `False`, so scanpy *can* produce the corrected
convention and simply does not by default. Measured on the fixture below the
two conventions differ by 7.6e-02 in z, so they are different answers rather
than different precisions, and each arm needs its own reference.

Every reference here is a value some other implementation PRODUCED, never one
this script derives:

  * corrected p   ← `mannwhitneyu(...).pvalue`         (scipy's own p-value)
  * corrected z   ← `scanpy(tie_correct=True).scores`  (f32 in the recarray)
  * uncorrected z ← `scanpy(tie_correct=False).scores` (f32)
  * uncorrected p ← `scanpy(tie_correct=False).pvals`  (f64)

An earlier version of this script took only `U` from scipy and rebuilt z and p
with `sigma_sq = (n1·n2/12)·((n+1) − tc/(n(n−1)))` — the same transform as
`wilcoxon_stats_from_rank_sum`. The numbers were identical (verified: max
|Δp| = 0.0), but the *provenance* was not: a bug in the tie term copied into
this script would have matched SCX exactly, and the uncorrected arm could not
have caught it because it runs with `tc = 0`. That is precisely the "sibling
implementations agreeing" failure ORG-7.21-3 exists to retire, relocated into
the oracle generator. Do not reintroduce a computed reference here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Shared with the other reference generators in this directory (Phase 7d
# extracted them so a second generator would not copy them).
from _rust_literals import f64_literal, rust_matrix

# --- the fixture -------------------------------------------------------------
#
# 12 cells x 6 genes, 2 groups, 2 unlabelled cells. Every value is exactly
# representable in f32, so the f32 -> f64 promotion inside the kernels is
# lossless and a reference computed here in f64 is comparable at abs=0.
#
# Rows 10 and 11 carry the unlabelled sentinel. Their values are deliberately
# extreme: an unlabelled cell is in no group but IS in the rank pool and in
# every group's "rest" (scanpy's rule, pyscx's since 0.17 / X9), so a kernel
# that drops them from either could not hide behind a tolerance.
#
# The six genes, each chosen for a path some arm treats differently:
#   g0  distinct separating values      — the clean signal
#   g1  constant everywhere             — a total tie; sigma^2 collapses to 0
#   g2  all zeros                       — degenerate, and no stored nonzeros
#   g3  negatives + explicit zeros      — the nnz path's `neg` block, and
#                                         stored zeros that must join the
#                                         *implicit*-zero block, not `pos`
#   g4  small counts                    — partial ties, the realistic shape
#   g5  nonzero ONLY in unlabelled rows — the inclusion canary: it is all-zero
#                                         over the labelled cells, so it is
#                                         distinguishable from the all-zero g2
#                                         only if unlabelled cells are in the
#                                         pool. Before X9 the two were identical.
GROUPS = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 2, 2]  # 2 == unlabelled sentinel
N_GROUPS = 2

X = np.array(
    [
        # g0    g1    g2    g3    g4    g5
        [1.0,  3.0,  0.0, -2.0,  0.0,  0.0],  # 0  grp0
        [2.0,  3.0,  0.0, -1.0,  1.0,  0.0],  # 1  grp0
        [3.0,  3.0,  0.0,  0.0,  1.0,  0.0],  # 2  grp0
        [4.0,  3.0,  0.0,  0.0,  2.0,  0.0],  # 3  grp0
        [11.0, 3.0,  0.0, -1.0,  1.0,  0.0],  # 4  grp1
        [12.0, 3.0,  0.0,  0.0,  2.0,  0.0],  # 5  grp1
        [13.0, 3.0,  0.0,  2.0,  2.0,  0.0],  # 6  grp1
        [14.0, 3.0,  0.0,  3.0,  3.0,  0.0],  # 7  grp1
        [5.0,  3.0,  0.0,  1.0,  2.0,  0.0],  # 8  grp0
        [15.0, 3.0,  0.0,  4.0,  3.0,  0.0],  # 9  grp1
        [100.0, 3.0, 0.0, -50.0, 9.0,  7.0],  # 10 unlabelled
        [200.0, 3.0, 0.0,  50.0, 9.0,  8.0],  # 11 unlabelled
    ],
    dtype=np.float32,
)

N_OBS, N_VARS = X.shape


def scipy_reference() -> np.ndarray:
    """Per-(gene, group) 1-vs-rest two-sided p from **scipy's own p-value**.

    Over EVERY cell: "rest" is every row outside the group, unlabelled rows
    included, so the reference is `mannwhitneyu(group, everything-else)` on the
    unfiltered matrix. scipy needs no notion of an unlabelled cell for that —
    such a cell is simply on the `g != grp` side of every split, which is
    exactly what scanpy's `X[~mask_g]` means.

    `res.pvalue` is taken verbatim. scipy applies its own tie correction
    internally, which is what makes this an independent check on SCX's
    `Σ(t³−t)/(n(n−1))` term rather than a restatement of it.
    """
    from scipy.stats import mannwhitneyu

    p = np.zeros((N_VARS, N_GROUPS))
    d = X.astype(np.float64)
    g = np.asarray(GROUPS)
    for j in range(N_VARS):
        col = d[:, j]
        for grp in range(N_GROUPS):
            if col.min() == col.max():
                # Fully tied column: sigma^2 collapses to exactly 0. Every SCX
                # kernel returns the documented (0.0, 1.0); scipy warns and
                # yields nan, so it is no oracle at all for this cell. The pin
                # is SCX's documented contract, and this branch says so.
                p[j, grp] = 1.0
                continue
            res = mannwhitneyu(
                col[g == grp], col[g != grp],
                use_continuity=False, alternative="two-sided", method="asymptotic",
            )
            p[j, grp] = res.pvalue
    return p


def scanpy_reference(tie_correct: bool) -> tuple[np.ndarray, np.ndarray]:
    """scanpy `rank_genes_groups(method="wilcoxon")`, either convention.

    Run on the FULL matrix with the two sentinel rows carrying a NaN label —
    the input the kernels are handed, not a pre-filtered stand-in. scanpy ranks
    the whole matrix and keeps a NaN-labelled cell in every group's "rest", and
    since 0.17 (X9) so does SCX; running scanpy on a filtered matrix would pin
    the old, divergent rule instead of the one being claimed.

    The categories are declared explicitly so a NaN row stays NaN rather than
    becoming a third level.
    """
    import anndata
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    cats = [f"grp{g}" for g in range(N_GROUPS)]
    labels = [f"grp{g}" if g < N_GROUPS else None for g in GROUPS]
    ad = anndata.AnnData(
        X=sp.csr_matrix(X.copy()),
        obs=pd.DataFrame({"g": pd.Categorical(labels, categories=cats)},
                         index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(N_VARS)]),
    )
    sc.tl.rank_genes_groups(ad, "g", method="wilcoxon", tie_correct=tie_correct)
    rgg = ad.uns["rank_genes_groups"]
    z = np.zeros((N_VARS, N_GROUPS))
    p = np.zeros((N_VARS, N_GROUPS))
    for grp in range(N_GROUPS):
        key = f"grp{grp}"
        names = list(rgg["names"][key])
        zs = dict(zip(names, np.asarray(rgg["scores"][key], dtype=np.float64)))
        ps = dict(zip(names, np.asarray(rgg["pvals"][key], dtype=np.float64)))
        for j in range(N_VARS):
            name = f"g{j}"
            zj, pj = zs[name], ps[name]
            # scanpy emits an exact 0.0 / 1.0 on a fully-tied column, the same
            # contract SCX has. Asserted rather than substituted: if a future
            # scanpy starts emitting nan there, this must be a loud failure,
            # not a silently invented reference value.
            assert np.isfinite(zj) and np.isfinite(pj), (
                f"scanpy returned non-finite ({zj}, {pj}) for {name}/{key}; "
                "the fully-tied contract changed -- do not paste over it"
            )
            z[j, grp] = zj
            p[j, grp] = pj
    return z, p






# --- NB-GLM vs pydeseq2 ------------------------------------------------------
#
# `docs/pseudobulk_nb_glm.md` is explicit about the bar: "ranking / effect-sign
# / significance parity, not numerical equality". So this section pins exactly
# that, and nothing tighter -- a numerical-equality assertion here would be a
# claim the docs deliberately do not make, and would go red on the apeglm
# shrinkage and dispersion-outlier handling SCX omits on purpose.
#
# The counts are emitted as literals rather than as a seed + distribution, so
# pydeseq2 and the Rust fitter see byte-identical input. Gene-major, matching
# `pseudobulk_nb_glm`'s `counts_gene_major` parameter directly.

NB_N_GENES = 24
NB_N_SAMPLES = 8
NB_ALPHA = 0.1  # true NB dispersion used to simulate; 9 genes carry a real LFC


def nb_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(counts[sample][gene], condition[sample], true_lfc[gene])`, deterministic."""
    rng = np.random.default_rng(20260823)
    mu = np.round(rng.uniform(30, 400, NB_N_GENES))
    lfc = np.zeros(NB_N_GENES)
    lfc[:5] = np.round(rng.uniform(1.2, 2.5, 5), 2)
    lfc[5:9] = -np.round(rng.uniform(1.2, 2.5, 4), 2)
    cond = np.array([0] * 4 + [1] * 4)
    counts = np.empty((NB_N_SAMPLES, NB_N_GENES), dtype=np.int64)
    for s in range(NB_N_SAMPLES):
        m = mu * (2.0 ** (lfc * cond[s]))
        counts[s] = rng.negative_binomial(1.0 / NB_ALPHA, 1.0 / (1.0 + NB_ALPHA * m))
    return counts, cond, lfc


def pydeseq2_reference() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(counts_gene_major, cond, log2FoldChange, padj)` from a real pydeseq2 run."""
    import pandas as pd
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts, cond, _ = nb_fixture()
    genes = [f"g{j}" for j in range(NB_N_GENES)]
    samples = [f"s{i}" for i in range(NB_N_SAMPLES)]
    cts = pd.DataFrame(counts, index=samples, columns=genes)
    meta = pd.DataFrame({"condition": ["A"] * 4 + ["B"] * 4}, index=samples)

    dds = DeseqDataSet(counts=cts, metadata=meta, design="~condition", quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", "B", "A"], quiet=True)
    st.summary()
    ref = st.results_df.reindex(genes)
    return (
        counts.T.astype(np.float64),  # gene-major, as pseudobulk_nb_glm wants
        cond.astype(np.float64),
        ref["log2FoldChange"].to_numpy(dtype=np.float64),
        ref["padj"].to_numpy(dtype=np.float64),
    )


def measure_python_side_bars() -> None:
    """Print the Python-side max |Δ| figures `docs/scanpy.md` quotes.

    The Rust tables above are the reference *values*; these are the *bars* the
    pytest suite compares at. They lived only in a session transcript until
    review pointed out that the page says "regenerate with this script" beside
    numbers the script did not produce — so a reader following the instruction
    got the tables and silently not the bars.

    Emitted as a comment block rather than as Rust, because these bars live in
    `pyscx/tests/test_accel.py` as module-level constants.
    """
    import warnings

    warnings.filterwarnings("ignore")
    try:
        import anndata
        import pandas as pd
        import scanpy as sc
        import scipy.sparse as sp

        import pyscx  # noqa: F401
    except ImportError as e:
        # LOUD, and non-zero. `docs/scanpy.md` says this script "fails if any
        # field there exceeds the same bar" -- a silent `return` made that true
        # only when the imports happened to succeed, so running it in a bare env
        # printed the reference tables, skipped the check, and exited 0. Found in
        # round-2 review. `SCX_SKIP_PYTHON_BARS=1` is the deliberate opt-out for
        # regenerating the Rust tables alone.
        if os.environ.get("SCX_SKIP_PYTHON_BARS") == "1":
            print(f"// python-side bars SKIPPED on request (SCX_SKIP_PYTHON_BARS=1): {e}")
            return
        raise SystemExit(
            f"// cannot measure the Python-side bars: {e}\n"
            f"// This script owns those bars as well as the Rust tables, so a run\n"
            f"// that cannot measure them has regenerated only half of what the\n"
            f"// docs say it does. Use the .venv (scanpy + pyscx installed), or\n"
            f"// set SCX_SKIP_PYTHON_BARS=1 to regenerate the Rust tables alone."
        )

    rng = np.random.default_rng(0)
    counts = rng.poisson(2.0, size=(120, 50)).astype(np.float32)
    counts[:40, :10] += rng.poisson(6.0, size=(40, 10))
    obs = pd.DataFrame(
        {"batch": pd.Categorical(["A"] * 40 + ["B"] * 40 + ["C"] * 40)},
        index=[f"c{i}" for i in range(120)],
    )
    var = pd.DataFrame(index=[f"g{j}" for j in range(50)])

    def measure(log1p: bool) -> dict[str, float]:
        ad = anndata.AnnData(X=sp.csr_matrix(counts.copy()), obs=obs.copy(), var=var.copy())
        if log1p:
            sc.pp.normalize_total(ad, target_sum=1e4)
            sc.pp.log1p(ad)
        a, b = ad.copy(), ad.copy()
        sc.tl.rank_genes_groups(a, "batch", method="wilcoxon")
        pyscx.accel.rank_genes_groups(b, "batch", device="cpu")
        ra, rb = a.uns["rank_genes_groups"], b.uns["rank_genes_groups"]
        # Both sides must report the SAME gene set before any field is compared.
        # Without this the loop below iterates SCX's genes and looks each up in
        # scanpy's map, so a result that dropped genes would be compared over
        # fewer of them -- the identical hole this PR removed from the pytest
        # logFC loop, reintroduced in the script that measures its bars. Found
        # in round-2 review.
        for grp in rb["names"].dtype.names:
            if set(ra["names"][grp]) != set(rb["names"][grp]):
                raise SystemExit(
                    f"// group {grp}: scanpy and scx report different gene sets, so "
                    f"the measured bars below would cover only the intersection"
                )
        worst: dict[str, float] = {}
        for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
            w = 0.0
            for grp in rb["names"].dtype.names:
                ma = dict(zip(ra["names"][grp], np.asarray(ra[field][grp], np.float64)))
                mb = dict(zip(rb["names"][grp], np.asarray(rb[field][grp], np.float64)))
                for gene in mb:
                    x, y = ma[gene], mb[gene]
                    if np.isfinite(x) and np.isfinite(y):
                        w = max(w, abs(x - y))
            worst[field] = w
        return worst

    raw, logged = measure(log1p=False), measure(log1p=True)
    print("// Python-side bars (pyscx/tests/test_accel.py), max |delta| vs scanpy:")
    for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        note = ""
        if field == "logfoldchanges":
            note = "  <- raw-count value is a DEFINITION difference, not a bar"
        print(f"//   {field:16s} raw={raw[field]:.4e}  log1p={logged[field]:.4e}{note}")
    print("// scanpy expm1s the group means unconditionally, so the raw-count")
    print("// logfoldchanges figure is pinned as a divergence, not as parity.")

    # And CHECK them, rather than only printing. Emitting a number a reader is
    # expected to eyeball against a doc table is how the bars drifted out of the
    # script in the first place. These are the constants in
    # pyscx/tests/test_accel.py; the figures above are this script's own fixture,
    # which is NOT the pytest fixture, so they differ -- what has to hold on both
    # is that every measurement stays under the shared bar.
    bars = {"scores": 1e-6, "pvals": 1e-12, "pvals_adj": 1e-12}
    over = [
        f"{field} {which}={vals[field]:.4e} exceeds its bar {bar:.0e}"
        for field, bar in bars.items()
        for which, vals in (("raw", raw), ("log1p", logged))
        if vals[field] > bar
    ]
    if logged["logfoldchanges"] > 1e-6:
        over.append(
            f"logfoldchanges log1p={logged['logfoldchanges']:.4e} exceeds its bar 1e-06"
        )
    if over:
        raise SystemExit(
            "// the measured divergence has outgrown the pinned bar(s):\n  "
            + "\n  ".join(over)
            + "\n// re-measure and widen deliberately, or find what regressed."
        )
    print("// every measurement above is inside the bar it justifies.")


def emit_nb_glm() -> None:
    import importlib.metadata as md

    counts_gm, cond, lfc, padj = pydeseq2_reference()
    _, _, true_lfc = nb_fixture()

    print()
    print(f"// --- NB-GLM vs pydeseq2 {md.version('pydeseq2')} ---")
    print(f"// bar: ranking / sign / significance parity (docs/pseudobulk_nb_glm.md),")
    print(f"// NOT numerical equality. {int((true_lfc != 0).sum())} of {NB_N_GENES} genes")
    print(f"// carry a real log2 fold change; NB dispersion alpha = {NB_ALPHA}.")
    print()
    print(f"const NB_N_GENES: usize = {NB_N_GENES};")
    print(f"const NB_N_SAMPLES: usize = {NB_N_SAMPLES};")
    print()
    print(rust_matrix("NB_COUNTS_GENE_MAJOR", counts_gm, ty="f64"))
    print()
    cond_lits = ", ".join(f64_literal(v) for v in cond)
    print(f"const NB_CONDITION: [f64; NB_N_SAMPLES] = [{cond_lits}];")
    print()
    for name, arr in (("NB_PYDESEQ2_LOG2FC", lfc), ("NB_PYDESEQ2_PADJ", padj)):
        lits = ",\n    ".join(
            ", ".join(f64_literal(v) for v in arr[i : i + 4]) for i in range(0, len(arr), 4)
        )
        print(f"const {name}: [f64; NB_N_GENES] = [\n    {lits},\n];")
        print()





def main() -> int:
    import importlib.metadata as md
    import warnings

    # scanpy's log2 of an all-zero gene's fold change is a legitimate nan here
    # (two of the six genes are deliberately degenerate) and is not what the
    # reference reads.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    p_scipy_tc = scipy_reference()
    z_scanpy_tc, p_scanpy_tc = scanpy_reference(tie_correct=True)
    z_scanpy_un, p_scanpy_un = scanpy_reference(tie_correct=False)

    versions = ", ".join(
        f"{d} {md.version(d)}" for d in ("scipy", "scanpy", "numpy")
    )
    print(f"// {versions}")
    print(f"// generated by benchmarks/scripts/{__file__.split('/')[-1]}")
    print()
    print(rust_matrix("FIXTURE_X", X.astype(np.float64), ty="f32"))
    print()
    print(f"const FIXTURE_GROUPS: [usize; {N_OBS}] = {GROUPS};")
    print()
    print(rust_matrix("SCIPY_P_TIE_CORRECTED", p_scipy_tc))
    print()
    print(rust_matrix("SCANPY_Z_TIE_CORRECTED", z_scanpy_tc))
    print()
    print(rust_matrix("SCANPY_P_TIE_CORRECTED", p_scanpy_tc))
    print()
    print(rust_matrix("SCANPY_Z_UNCORRECTED", z_scanpy_un))
    print()
    print(rust_matrix("SCANPY_P_UNCORRECTED", p_scanpy_un))
    print()
    gap = float(np.max(np.abs(z_scanpy_tc - z_scanpy_un)))
    print(f"// max |z_corrected - z_uncorrected| on this fixture = {gap:.3e},")
    print("// both from scanpy, so this is a convention difference and not a")
    print("// cross-library one. Pinning either arm against the other's oracle")
    print("// would be wrong rather than merely loose.")
    print()
    cross = float(np.max(np.abs(p_scipy_tc - p_scanpy_tc)))
    print(f"// scipy's corrected p vs scanpy's corrected p = {cross:.3e} --")
    print("// two independent implementations of the tie term, agreeing. That")
    print("// agreement is asserted in wilcoxon_reference_tests.rs, because it")
    print("// is what makes either of them an oracle for SCX rather than a")
    print("// second opinion of the same arithmetic.")
    print()
    print("// scanpy stores `scores` as float32 and `pvals` as float64 (verified")
    print("// against the recarray dtypes, not assumed), so every z bar is f32-wide")
    print("// and every p bar is tight. scipy is f64 throughout.")
    measure_python_side_bars()
    emit_nb_glm()
    return 0


if __name__ == "__main__":
    sys.exit(main())
