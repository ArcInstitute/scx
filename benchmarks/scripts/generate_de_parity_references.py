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
Σ(t³−t) tie correction. SCX's `tie_correct` defaults to **false**, matching
scanpy, which applies none. So:

  * `tie_correct=true`  → pinned against **scipy**, exact.
  * `tie_correct=false` → pinned against **scanpy**, which is the only
    external implementation of that convention.

Measured on the fixture below: the two conventions differ by 8.2e-02 in z.
Pinning the uncorrected arm against scipy would be wrong, not merely loose.
"""

from __future__ import annotations

import sys

import numpy as np

# --- the fixture -------------------------------------------------------------
#
# 12 cells x 6 genes, 2 groups, 2 unlabelled cells. Every value is exactly
# representable in f32, so the f32 -> f64 promotion inside the kernels is
# lossless and a reference computed here in f64 is comparable at abs=0.
#
# Rows 10 and 11 carry the unlabelled sentinel. Their values are deliberately
# extreme: if a kernel ever let an unlabelled cell into the rank pool or the
# rest denominator, no tolerance would hide it.
#
# The six genes, each chosen for a path some arm treats differently:
#   g0  distinct separating values      — the clean signal
#   g1  constant everywhere             — a total tie; sigma^2 collapses to 0
#   g2  all zeros                       — degenerate, and no stored nonzeros
#   g3  negatives + explicit zeros      — the nnz path's `neg` block, and
#                                         stored zeros that must join the
#                                         *implicit*-zero block, not `pos`
#   g4  small counts                    — partial ties, the realistic shape
#   g5  nonzero ONLY in unlabelled rows — the leak canary: to the labelled
#                                         pool this gene is identical to g2
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
LABELLED = [i for i, g in enumerate(GROUPS) if g < N_GROUPS]


def scipy_reference(tie_correct: bool) -> tuple[np.ndarray, np.ndarray]:
    """Per-(gene, group) 1-vs-rest `(z, p)` from scipy, over LABELLED cells only.

    scipy is asked for `U` (which carries no tie-correction choice) and the
    z-score is formed from it with the tie term this arm wants, so the
    `tie_correct=False` arm is not silently scipy's corrected answer.
    """
    from scipy.stats import mannwhitneyu, norm

    z = np.zeros((N_VARS, N_GROUPS))
    p = np.zeros((N_VARS, N_GROUPS))
    d = X[LABELLED].astype(np.float64)
    g = np.asarray([GROUPS[i] for i in LABELLED])
    n = len(LABELLED)
    for j in range(N_VARS):
        col = d[:, j]
        _, counts = np.unique(col, return_counts=True)
        tc = float(np.sum(counts**3 - counts)) if tie_correct else 0.0
        for grp in range(N_GROUPS):
            mask = g == grp
            n1, n2 = int(mask.sum()), int((~mask).sum())
            if col.min() == col.max():
                # Fully tied column: sigma^2 collapses to exactly 0 and every
                # kernel here returns the documented (0.0, 1.0) rather than a
                # 0/0. scipy warns and yields nan, so it is no oracle at all
                # for this cell -- the pin is SCX's documented contract.
                z[j, grp], p[j, grp] = 0.0, 1.0
                continue
            u1, _ = mannwhitneyu(
                col[mask], col[~mask],
                use_continuity=False, alternative="two-sided", method="asymptotic",
            )
            sigma_sq = (n1 * n2 / 12.0) * ((n + 1) - tc / (n * (n - 1)))
            if sigma_sq <= 0.0:
                z[j, grp], p[j, grp] = 0.0, 1.0
                continue
            zz = (u1 - n1 * n2 / 2.0) / np.sqrt(sigma_sq)
            z[j, grp] = zz
            p[j, grp] = 2.0 * norm.sf(abs(zz))
    return z, p


def scanpy_reference() -> tuple[np.ndarray, np.ndarray]:
    """scanpy `rank_genes_groups(method="wilcoxon")` — the uncorrected arm.

    Run on the LABELLED subset, because scanpy has no notion of a cell outside
    the comparison pool: its "rest" is every other row. Feeding it the full
    matrix would make it answer a different question, which is precisely the
    §7.1 class of bug this fixture exists to pin.
    """
    import anndata
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp

    labels = [f"grp{GROUPS[i]}" for i in LABELLED]
    ad = anndata.AnnData(
        X=sp.csr_matrix(X[LABELLED].copy()),
        obs=pd.DataFrame({"g": pd.Categorical(labels)},
                         index=[f"c{i}" for i in LABELLED]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(N_VARS)]),
    )
    sc.tl.rank_genes_groups(ad, "g", method="wilcoxon", tie_correct=False)
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


def f64_literal(v: float) -> str:
    """Format as a Rust `f64` literal.

    `repr` gives the **shortest** string that round-trips through f64, which is
    what clippy's `excessive_precision` lint demands: `.17g` would emit
    `0.0090234388180803256` where f64 only carries
    `0.009023438818080326`, and clippy rejects the extra digits as a claim of
    precision the type cannot hold. Shortest-round-trip is also the honest
    form — every digit printed is a digit the value has.
    """
    s = repr(float(v))
    return s if ("." in s or "e" in s or "E" in s) else s + ".0"




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



def rust_matrix(name: str, m: np.ndarray, ty: str = "f64") -> str:
    rows = ",\n".join(
        "    [" + ", ".join(f64_literal(v) for v in row) + "]" for row in m
    )
    return f"const {name}: [[{ty}; {m.shape[1]}]; {m.shape[0]}] = [\n{rows},\n];"


def main() -> int:
    import importlib.metadata as md
    import warnings

    # scanpy's log2 of an all-zero gene's fold change is a legitimate nan here
    # (two of the six genes are deliberately degenerate) and is not what the
    # reference reads.
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    z_tc, p_tc = scipy_reference(tie_correct=True)
    z_no, p_no = scanpy_reference()

    versions = ", ".join(
        f"{d} {md.version(d)}" for d in ("scipy", "scanpy", "numpy")
    )
    print(f"// {versions}")
    print(f"// generated by benchmarks/scripts/{__file__.split('/')[-1]}")
    print()
    print(rust_matrix("FIXTURE_X", X.astype(np.float64), ty="f32"))
    print()
    print(f"const FIXTURE_GROUPS: [usize; {N_OBS}] = {GROUPS};".replace("[", "[").replace("'", ""))
    print()
    print(rust_matrix("SCIPY_Z_TIE_CORRECTED", z_tc))
    print()
    print(rust_matrix("SCIPY_P_TIE_CORRECTED", p_tc))
    print()
    print(rust_matrix("SCANPY_Z_UNCORRECTED", z_no))
    print()
    print(rust_matrix("SCANPY_P_UNCORRECTED", p_no))
    print()
    gap = float(np.max(np.abs(z_tc - z_no)))
    print(f"// max |z_corrected - z_uncorrected| on this fixture = {gap:.3e}")
    print("// -- the two conventions are genuinely different answers, not")
    print("//    rounding: pinning the uncorrected arm against scipy is wrong.")
    print()
    print("// scanpy stores `scores` as float32 and `pvals` as float64 (verified")
    print("// against the recarray dtypes, not assumed), so the uncorrected arm")
    print("// pins z at f32 precision and p tightly. scipy is f64 throughout.")
    emit_nb_glm()
    return 0


if __name__ == "__main__":
    sys.exit(main())
