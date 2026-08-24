"""Generate the pinned reference fixtures for PFlog's NB overdispersion `α`.

Run it from the repo root, in the uv venv — never system Python:

    .venv/bin/python benchmarks/scripts/generate_pflog_alpha_references.py

Paste the output into `scx-accel/src/pflog_reference_values.rs`.

## Why this script exists

`estimate_alpha` had three implementations of one definition and no test against
a generating process. `scx-accel/src/pflog_tests.rs` carried a `reference_alpha`
that re-implemented the estimator *including its `var > mean` filter*, and
`pyscx/tests/test_accel_pflog.py` carried a third copy in numpy. All three
agreed, and all three were wrong in the same way: the filter drops every gene
whose method-of-moments estimate is ≤ 0 **before** the median, so on a
low-dispersion matrix the survivors are the upper tail of sampling noise and the
pooled `α` is biased high (review §7.14).

## The oracle here is the GENERATING PARAMETER, not another implementation

This is the one place in the reference-generator family where the reference is
not something a second library produced — it is `α_true`, the dispersion the
counts were simulated from. Ground truth beats any sibling implementation, and
it is the reference the review asked for ("add a test simulating NB(μ, α) with
known α").

Two consequences worth stating, because both are ways this script could quietly
become self-referential:

  * **The bars are not calibrated against SCX's answer.** A bar set from "what
    SCX returns, plus headroom" would move together with any bug in SCX and the
    test would keep passing — the exact defect the first version of
    `wilcoxon_reference_values.rs` had. Each `REL_BAR_*` here is a quantile of
    the *estimator's sampling spread under the simulation design*, computed from
    independent draws of the same `(n_cells, μ, α_true)` design. The **reference**
    stays `α_true`; the bar only says how wide one 50-cell draw can be.
  * **`TRUNCATED_ALPHA_*` is a repulsion target, not an agreement target.** It is
    what the pre-7d truncating estimator returns on this exact matrix. Nothing
    asserts SCX equals it; the tests assert SCX is *closer to `α_true`* than it
    is, and that the bar rejects it. Its exact value therefore does not need to
    be authoritative, which is why computing it here is not a self-reference.

## The three fixtures

  * **A** — a realistic overdispersed matrix (μ spread over two orders of
    magnitude, `α_true = 0.02`, 50 cells). The truncation bites: it discards
    roughly a quarter of the genes and biases `α` high. Statistical arm.
  * **B** — strongly overdispersed (`α_true = 0.5`), where **no** gene is
    discarded and the two estimators must coincide exactly. The accept side: the
    fix must not be "always return something smaller".
  * **C** — an under-dispersed majority (binomial counts, `Var < μ` by
    construction) plus four genuinely over-dispersed genes. Deterministic, and
    the sharpest statement of the mechanism: the truncating estimator keeps only
    the four and reports *their* dispersion as the matrix's, with `fell_back =
    false`. The pooled median is negative, so the honest answer is the
    documented fallback. No sampling luck is involved in that verdict, which is
    why this fixture exists alongside A.

Counts are emitted as integer literals rather than as a seed plus a
distribution, so the Rust estimator sees byte-identical input to the matrix
these numbers were measured on. A seed would make the reference depend on
numpy's generator staying put across versions.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from _rust_literals import f64_literal, rust_int_matrix

# `AlphaOptions::default().mu_min` in scx-accel/src/pflog.rs. Emitted below so
# the two cannot drift silently: the Rust side asserts the value it compiled
# against equals this.
MU_MIN = 1e-3

# Independent draws used to characterise the estimator's sampling spread. Seeded,
# so a regeneration reproduces the same bars.
CALIBRATION_DRAWS = 400
# Quantile of |α̂ − α_true| / α_true taken as the bar, plus a small headroom
# factor. High enough that the pinned fixture is not a lucky draw; the premise
# check below still requires the bar to reject the truncated answer.
BAR_QUANTILE = 0.99
BAR_HEADROOM = 1.25
# Fixture A's real separator: how much closer to α_true the pooled estimate must
# be than the truncated one. Deterministic for the pinned matrix.
A_SEPARATION = 2.0


# --- the simulation design ---------------------------------------------------


def nb_counts(n_cells: int, mus: np.ndarray, alpha: float, rng) -> np.ndarray:
    """`counts[cell][gene]` from NB(μ_g, α) with `Var = μ + α·μ²`.

    `negative_binomial(n=1/α, p=1/(1+α·μ))` is that parameterisation — the same
    one `generate_de_parity_references.py::nb_fixture` uses, and the same one
    `estimate_alpha`'s doc comment states.
    """
    out = np.empty((n_cells, len(mus)), dtype=np.int64)
    for g, mu in enumerate(mus):
        if alpha == 0.0:
            out[:, g] = rng.poisson(mu, size=n_cells)
        else:
            out[:, g] = rng.negative_binomial(
                1.0 / alpha, 1.0 / (1.0 + alpha * mu), size=n_cells
            )
    return out


def log_uniform_mus(n_genes: int, lo: float, hi: float, rng) -> np.ndarray:
    """Gene means spread log-uniformly over `[lo, hi]`.

    A *flat* μ makes both estimators look bad and represents no real matrix: at
    μ = 3 and 50 cells the per-gene MoM noise is seven times α itself, so the
    pooled median is a coin flip about zero. Real counts span orders of
    magnitude, and the high-μ genes are what anchor the median.
    """
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n_genes))


# --- the two estimators ------------------------------------------------------


def _gene_moments(counts: np.ndarray, mu_min: float = MU_MIN):
    mean = counts.mean(axis=0)
    var = counts.var(axis=0, ddof=1)  # ddof=1: streaming_mean_var is Bessel-corrected
    above = mean > mu_min
    alpha_g = (var[above] - mean[above]) / (mean[above] ** 2)
    return mean, var, above, alpha_g


def pooled_median_alpha(counts: np.ndarray, mu_min: float = MU_MIN) -> float:
    """The fixed estimator: median over every gene with a mean, negatives kept."""
    _, _, _, alpha_g = _gene_moments(counts, mu_min)
    return float(np.median(alpha_g))


def truncated_alpha(counts: np.ndarray, mu_min: float = MU_MIN):
    """The pre-7d estimator: `var > mean` applied *before* the median.

    Returns `(alpha, n_survivors)`, or `(None, 0)` when nothing survives.
    """
    mean, var, above, alpha_g = _gene_moments(counts, mu_min)
    keep = (var[above] > mean[above]) & (alpha_g > 0.0)
    if not keep.any():
        return None, 0
    return float(np.median(alpha_g[keep])), int(keep.sum())


def calibrate_bar(n_cells: int, mus: np.ndarray, alpha_true: float, seed: int) -> float:
    """Bar for `|α̂ − α_true| / α_true`, from the design's own sampling spread.

    Deliberately does **not** look at the pinned fixture: it draws
    `CALIBRATION_DRAWS` fresh matrices from the same design and takes a high
    quantile. That keeps the bar a statement about how wide one draw of this size
    can be, rather than a restatement of what SCX happened to return.
    """
    rng = np.random.default_rng(seed)
    rel = np.empty(CALIBRATION_DRAWS)
    for i in range(CALIBRATION_DRAWS):
        c = nb_counts(n_cells, mus, alpha_true, rng)
        rel[i] = abs(pooled_median_alpha(c) - alpha_true) / alpha_true
    return float(np.quantile(rel, BAR_QUANTILE) * BAR_HEADROOM)


# --- fixture A: realistic overdispersion, the truncation bites ---------------
#
# What the bar on this fixture is NOT, because two premise checks in this script
# rejected the fixtures that tried to make it so:
#
#   * The truncation's bias and the estimator's own sampling spread are both
#     proportional to the per-gene MoM noise `√(2/(n−1))·(1/μ + α)`, so their
#     ratio is set by the gene count, not by the cell count. At `α = 0.02` the
#     bias (17%) is smaller than the 99th-percentile spread (27%): no honest
#     single-draw tolerance can separate the two estimators here.
#   * Pushing `α` down to 0.002 does invert that (bias ~3.5x vs spread ~1.4x) —
#     and puts the pooled median at −3e−6, i.e. the fixed estimator *falls back*
#     and the arm stops testing an estimate at all. Every regime where the bias
#     is large is a regime where a matrix-wide α is not determined by 50 cells.
#
# So the separator on this fixture is not the tolerance. It is two facts about
# this exact matrix, both exact: the pool must contain **every** gene with a mean
# (120, where the truncation kept 91), and the pooled estimate must be strictly
# closer to `α_true` than the pinned truncated answer is. `A_REL_BAR` remains as
# a gross-breakage bound, and the tests say so rather than presenting it as the
# thing that catches the bug. The tight accuracy claim lives on fixture B.
A_N_CELLS, A_N_GENES, A_ALPHA_TRUE = 50, 120, 0.02
A_MU_LO, A_MU_HI = 0.5, 200.0
A_SEED_MUS, A_SEED_COUNTS, A_SEED_CAL = 70001, 70002, 70003

# --- fixture B: strong overdispersion, the truncation is inert ---------------

B_N_CELLS, B_N_GENES, B_ALPHA_TRUE = 40, 40, 0.5
B_MU_LO, B_MU_HI = 5.0, 200.0
B_SEED_MUS, B_SEED_COUNTS, B_SEED_CAL = 70011, 70012, 70013

# --- fixture C: under-dispersed majority, four over-dispersed genes ----------

C_N_CELLS, C_N_UNDER, C_N_OVER, C_ALPHA_OVER = 30, 20, 4, 0.5
C_SEED = 70021


def fixture_a():
    mus = log_uniform_mus(A_N_GENES, A_MU_LO, A_MU_HI, np.random.default_rng(A_SEED_MUS))
    counts = nb_counts(A_N_CELLS, mus, A_ALPHA_TRUE, np.random.default_rng(A_SEED_COUNTS))
    return counts, mus


def fixture_b():
    mus = log_uniform_mus(B_N_GENES, B_MU_LO, B_MU_HI, np.random.default_rng(B_SEED_MUS))
    counts = nb_counts(B_N_CELLS, mus, B_ALPHA_TRUE, np.random.default_rng(B_SEED_COUNTS))
    return counts, mus


def fixture_c():
    """Binomial (under-dispersed) majority plus `C_N_OVER` NB genes.

    Binomial counts have `Var = N·p·(1−p) < mean = N·p` for every `p`, so the
    under-dispersed block is under-dispersed by construction rather than by luck.
    """
    rng = np.random.default_rng(C_SEED)
    cols = []
    for _ in range(C_N_UNDER):
        n_trials = int(rng.integers(6, 30))
        p = float(rng.uniform(0.3, 0.8))
        cols.append(rng.binomial(n_trials, p, size=C_N_CELLS))
    for _ in range(C_N_OVER):
        mu = float(rng.uniform(20.0, 120.0))
        cols.append(
            rng.negative_binomial(
                1.0 / C_ALPHA_OVER, 1.0 / (1.0 + C_ALPHA_OVER * mu), size=C_N_CELLS
            )
        )
    return np.stack(cols, axis=1).astype(np.int64)


def check_u16(name: str, counts: np.ndarray) -> None:
    if counts.min() < 0 or counts.max() > 65535:
        raise SystemExit(
            f"// {name}: counts must fit u16, saw [{counts.min()}, {counts.max()}]"
        )


def main() -> int:
    versions = []
    for dist in ("numpy",):
        try:
            from importlib.metadata import version

            versions.append(f"{dist} {version(dist)}")
        except Exception:  # pragma: no cover - provenance only
            pass
    print(f"// {', '.join(versions)}")
    print(f"// generated by benchmarks/scripts/{os.path.basename(__file__)}")
    print()
    print(f"pub const MU_MIN: f64 = {f64_literal(MU_MIN)};")
    print()

    failures: list[str] = []

    # ---- A -----------------------------------------------------------------
    a_counts, a_mus = fixture_a()
    check_u16("fixture A", a_counts)
    a_new = pooled_median_alpha(a_counts)
    a_old, a_survivors = truncated_alpha(a_counts)
    a_pooled_n = int((a_counts.mean(axis=0) > MU_MIN).sum())
    a_bar = calibrate_bar(A_N_CELLS, a_mus, A_ALPHA_TRUE, A_SEED_CAL)
    a_rel_new = abs(a_new - A_ALPHA_TRUE) / A_ALPHA_TRUE
    a_rel_old = abs(a_old - A_ALPHA_TRUE) / A_ALPHA_TRUE

    print("// --- fixture A: realistic overdispersion, the truncation bites ---")
    print(f"pub const A_N_CELLS: usize = {A_N_CELLS};")
    print(f"pub const A_N_GENES: usize = {A_N_GENES};")
    print(f"pub const A_ALPHA_TRUE: f64 = {f64_literal(A_ALPHA_TRUE)};")
    print(f"pub const A_REL_BAR: f64 = {f64_literal(round(a_bar, 4))};")
    print(f"pub const A_SEPARATION: f64 = {f64_literal(A_SEPARATION)};")
    print(f"pub const A_TRUNCATED_ALPHA: f64 = {f64_literal(a_old)};")
    print(f"pub const A_TRUNCATED_SURVIVORS: usize = {a_survivors};")
    print(f"pub const A_N_GENES_POOLED: usize = {a_pooled_n};")
    print(rust_int_matrix("A_COUNTS", a_counts))
    print()

    if a_rel_new > a_bar:
        failures.append(
            f"fixture A: the pooled estimate is {a_rel_new:.4f} off α_true, outside "
            f"its own sampling bar {a_bar:.4f} — this draw is an outlier, reseed it"
        )
    if a_new <= 0.0:
        failures.append(
            f"fixture A: the pooled median is {a_new:+.6f}, so the fixed estimator "
            f"would FALL BACK on this fixture and the arm would stop testing an "
            f"estimate at all — pick a dispersion this many cells can resolve"
        )
    if not abs(a_new - A_ALPHA_TRUE) * A_SEPARATION < abs(a_old - A_ALPHA_TRUE):
        failures.append(
            f"fixture A: the pooled estimate is {abs(a_old - A_ALPHA_TRUE) / abs(a_new - A_ALPHA_TRUE):.2f}x "
            f"closer to α_true than the truncated one, under the {A_SEPARATION}x this "
            f"fixture is pinned for — it has stopped demonstrating the bias, and the "
            f"tolerance cannot demonstrate it either (see the note above)"
        )
    if a_survivors >= a_pooled_n:
        failures.append(
            f"fixture A: the truncation discarded nothing ({a_survivors} of "
            f"{a_pooled_n}) — it cannot be the fixture for a truncation bug"
        )

    # ---- B -----------------------------------------------------------------
    b_counts, b_mus = fixture_b()
    check_u16("fixture B", b_counts)
    b_new = pooled_median_alpha(b_counts)
    b_old, b_survivors = truncated_alpha(b_counts)
    b_pooled_n = int((b_counts.mean(axis=0) > MU_MIN).sum())
    b_bar = calibrate_bar(B_N_CELLS, b_mus, B_ALPHA_TRUE, B_SEED_CAL)
    b_rel_new = abs(b_new - B_ALPHA_TRUE) / B_ALPHA_TRUE

    print("// --- fixture B: strong overdispersion, the truncation is inert ---")
    print(f"pub const B_N_CELLS: usize = {B_N_CELLS};")
    print(f"pub const B_N_GENES: usize = {B_N_GENES};")
    print(f"pub const B_ALPHA_TRUE: f64 = {f64_literal(B_ALPHA_TRUE)};")
    print(f"pub const B_REL_BAR: f64 = {f64_literal(round(b_bar, 4))};")
    print(f"pub const B_N_GENES_POOLED: usize = {b_pooled_n};")
    print(rust_int_matrix("B_COUNTS", b_counts))
    print()

    if b_rel_new > b_bar:
        failures.append(
            f"fixture B: the pooled estimate is {b_rel_new:.4f} off α_true, outside "
            f"its sampling bar {b_bar:.4f}"
        )
    if b_survivors != b_pooled_n:
        failures.append(
            f"fixture B: {b_pooled_n - b_survivors} gene(s) were discarded by the "
            f"truncation, so this is not the inert case the accept-side arm needs"
        )
    if b_old is None or b_old != b_new:
        failures.append(
            f"fixture B: the two estimators disagree ({b_old!r} vs {b_new!r}) on a "
            f"fixture where no gene is discarded — they must coincide exactly here"
        )

    # ---- C -----------------------------------------------------------------
    c_counts = fixture_c()
    check_u16("fixture C", c_counts)
    c_new = pooled_median_alpha(c_counts)
    c_old, c_survivors = truncated_alpha(c_counts)
    c_pooled_n = int((c_counts.mean(axis=0) > MU_MIN).sum())
    c_mean, c_var = c_counts.mean(axis=0), c_counts.var(axis=0, ddof=1)
    c_under = int((c_var <= c_mean).sum())

    print("// --- fixture C: under-dispersed majority, four over-dispersed genes ---")
    print(f"pub const C_N_CELLS: usize = {C_N_CELLS};")
    print(f"pub const C_N_GENES: usize = {C_N_UNDER + C_N_OVER};")
    print(f"pub const C_ALPHA_OVER: f64 = {f64_literal(C_ALPHA_OVER)};")
    print(f"pub const C_N_UNDERDISPERSED: usize = {c_under};")
    print(f"pub const C_TRUNCATED_ALPHA: f64 = {f64_literal(c_old)};")
    print(f"pub const C_TRUNCATED_SURVIVORS: usize = {c_survivors};")
    print(f"pub const C_N_GENES_POOLED: usize = {c_pooled_n};")
    print(rust_int_matrix("C_COUNTS", c_counts))
    print()

    if c_new >= 0.0:
        failures.append(
            f"fixture C: the pooled median is {c_new:+.5f}, not negative — the "
            f"fallback arm needs a matrix whose honest pooled answer is 'not "
            f"overdispersed'"
        )
    if c_old is None or c_old <= 0.0:
        failures.append(
            f"fixture C: the truncated estimator returned {c_old!r} rather than a "
            f"confident positive α — the point of this fixture is that it reports "
            f"four genes' dispersion as the matrix's"
        )
    if c_under != C_N_UNDER:
        failures.append(
            f"fixture C: {c_under} genes came out under-dispersed, not the "
            f"{C_N_UNDER} the binomial block should guarantee"
        )

    # ---- measured commentary ----------------------------------------------
    print("// Measured, at generation time:")
    print(
        f"//   A: pooled median {a_new:+.6f} (rel {a_rel_new:.4f}) vs truncated "
        f"{a_old:+.6f} (rel {a_rel_old:.4f}); α_true {A_ALPHA_TRUE}"
    )
    print(
        f"//      the truncation kept {a_survivors} of {a_pooled_n} genes; bar "
        f"{a_bar:.4f} from the {BAR_QUANTILE:.0%} quantile of "
        f"{CALIBRATION_DRAWS} independent draws x {BAR_HEADROOM}"
    )
    print(
        f"//      pseudocount 1/(4α): truth {1 / (4 * A_ALPHA_TRUE):.3f}, pooled "
        f"{1 / (4 * a_new):.3f}, truncated {1 / (4 * a_old):.3f}"
    )
    print(
        f"//   B: both estimators {b_new:+.6f} (rel {b_rel_new:.4f}); no gene "
        f"discarded ({b_survivors} of {b_pooled_n})"
    )
    print(
        f"//   C: pooled median {c_new:+.6f} -> fallback; truncated {c_old:+.6f} "
        f"from {c_survivors} of {c_pooled_n} genes ({c_under} under-dispersed)"
    )

    if failures:
        raise SystemExit(
            "// the fixtures no longer support the claims they are pinned for:\n  "
            + "\n  ".join(failures)
        )
    print("// every fixture supports the claim it is pinned for.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
