#!/usr/bin/env python3
"""Generate SCX's harmonypy-pinned reference modules.

Two targets, because harmonypy owns two of SCX's documented parity claims:

    .venv/bin/python benchmarks/scripts/generate_harmony_references.py harmony \
        > scx-accel/src/harmony/harmony_reference_values.rs
    .venv/bin/python benchmarks/scripts/generate_harmony_references.py lisi \
        > scx-accel/src/lisi_reference_values.rs


Phase 7e / ORG-7.21-4. Harmony was the last documented parity claim in
`scx-accel` with no checked-in reference number: `harmony/tests.rs` is entirely
self-consistency, and the one R-reference test skips whenever its gitignored
50 MB `.npz` is absent — which is for every contributor and for CI.

Four rules this generator inherits from `generate_de_parity_references.py` and
`generate_pflog_alpha_references.py`, because breaking any one of them turns the
"reference" back into SCX agreeing with itself:

1. **Every expected value is one harmonypy PRODUCED**, never one this script
   derives. The formulas under test — the M-step, the cosine distance, the ridge
   solve, the diversity penalty, the cross-entropy — are never written here.
   `Harmony.harmonize` is monkeypatched to a no-op so `__init__` stops after
   `init_cluster`, and then harmonypy's own `cluster`, `moe_correct_ridge`,
   `update_R` and `compute_objective` are called directly. Nothing else is
   patched; no arithmetic is replaced.
2. **The fixture is emitted as literals beside the expected values**, so the two
   cannot drift apart. The fixture is itself harmonypy's `init_cluster` output.
3. **The bars are CHECKED here, not merely printed.** A bar that only appears in
   a comment is a bar nobody ran.
4. **A missing dependency exits non-zero and names the opt-out.** A generator
   that quietly emits a shorter file is how a table silently stops being pinned.

harmonypy 0.2.0 is torch **float32** throughout (`allocate_buffers`), so every
expected value here carries f32 precision and the bars say so. SCX accumulates
in f64 and stores `r`/`dist_mat` as f32, so agreement at ~1e-6 relative is the
most either side can offer.
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _rust_literals import f64_literal, rust_f64_array, rust_matrix  # noqa: E402

try:
    import torch
    from harmonypy.harmony import Harmony
except ImportError as e:  # pragma: no cover - environment guard
    sys.exit(
        f"FATAL: {e}\n"
        "This generator pins Harmony against harmonypy; it cannot run without it.\n"
        "Use the repo venv, which has harmonypy 0.2.0 and torch:\n"
        "    .venv/bin/python benchmarks/scripts/generate_harmony_references.py\n"
    )

# ─── Fixture design ──────────────────────────────────────────────────────────
# Small enough to read as Rust literals, large enough that the ridge solve has
# a well-conditioned per-cluster covariance and every batch survives SCX's
# `batch_prop_cutoff` (so the pruning branch is not silently under test here).
N_CELLS = 36
N_PCS = 5
N_CLUSTERS = 3
N_BATCHES = 3
SIGMA = 0.1
THETA = 2.0
LAMB = 1.0
ALPHA = 0.2
SEED = 20260824

# `update_R` shuffles cells and updates O/E block by block, and SCX draws that
# permutation from ChaCha8 while harmonypy draws it from torch — so the general
# case is not comparable at all. One block (`block_size = 1.0`) makes both sides
# permutation-invariant, but it is NOT enough on its own: removing every cell at
# once drives O and E to exactly zero, and harmonypy's `clamp(E/(O+E), min=1e-8)`
# then returns 1e-8 for every entry, so `R` collapses to its `R_block_sum` clamp
# and no longer sums to one. (SCX's `+1` smoothing happens to survive that —
# `(2*0+1)/(0+0+1) = 1` — which is worth knowing but is not a parity arm.)
# Setting theta to zero as well makes the penalty exactly 1 on both sides for
# any O/E, so the arm pins `update_R`'s softmax half and nothing else.
BLOCK_SIZE = 1.0
SOFTMAX_ARM_THETA = 0.0

# ─── Bars ────────────────────────────────────────────────────────────────────
# These are the ONLY numbers in this file that harmonypy did not produce: each
# is the max |SCX - harmonypy| observed by `cargo test -p scx-accel --lib
# harmony_reference`, rounded UP one decimal order. They are measured on the
# Rust side because they describe the gap between two implementations, not a
# property of either — and they are recorded here so a regeneration cannot
# silently drop them. Nothing below is calibrated from what SCX returns in the
# sense that matters: widening a bar cannot make a wrong SCX pass an arm whose
# expected value is harmonypy's.
BARS = {
    "HP_MSTEP_Y_ATOL": (1e-7, "observed 6.233e-08 — f32 reference vs f64 accumulation"),
    "HP_MSTEP_DIST_ATOL": (1e-6, "observed 4.768e-07 — dist is f32 on both sides"),
    "HP_RIDGE_ATOL": (1e-5, "observed 2.258e-06 — an f32 matrix inverse, the loosest arm"),
    "HP_SOFTMAX_R_ATOL": (1e-7, "observed 5.961e-08 — R is f32 on both sides"),
    "HP_OBJ_ATOL": (1e-4, "observed 3.523e-05 on a term of magnitude 407 (8.7e-08 relative)"),
    "HP_OBJ_CROSS_GAP": (277.5595425793, "SCX 3.7439426553 vs harmonypy 281.3034852346"),
    "HP_OBJ_CROSS_GAP_ATOL": (1e-3, "the gap itself is a difference of two f32-derived sums"),
}


def build_fixture():
    rng = np.random.default_rng(SEED)
    # Three batch-shifted Gaussian blobs, so the diversity penalty has something
    # to push against and O and E genuinely differ.
    labels = np.repeat(np.arange(N_BATCHES), N_CELLS // N_BATCHES)
    centers = rng.normal(0.0, 1.0, size=(N_CLUSTERS, N_PCS))
    assign = rng.integers(0, N_CLUSTERS, size=N_CELLS)
    batch_shift = rng.normal(0.0, 0.4, size=(N_BATCHES, N_PCS))
    z = (
        centers[assign]
        + batch_shift[labels]
        + rng.normal(0.0, 0.25, size=(N_CELLS, N_PCS))
    )
    # harmonypy takes Z as (d x N).
    z_dn = np.ascontiguousarray(z.T, dtype=np.float32)
    phi = np.zeros((N_BATCHES, N_CELLS), dtype=np.float32)
    phi[labels, np.arange(N_CELLS)] = 1.0
    pr_b = phi.sum(axis=1) / N_CELLS
    return z_dn, phi, pr_b, labels


def new_state(z_dn, phi, pr_b, theta=THETA, block_size=BLOCK_SIZE):
    """A Harmony whose `__init__` stops after `init_cluster`.

    Only the driver loop is disabled. Every formula stays harmonypy's.
    """
    real_harmonize = Harmony.harmonize
    Harmony.harmonize = lambda self, *a, **kw: None
    try:
        return Harmony(
            Z=z_dn,
            Phi=phi,
            Pr_b=pr_b,
            sigma=np.full(N_CLUSTERS, SIGMA, dtype=np.float32),
            theta=np.full(N_BATCHES, theta, dtype=np.float32),
            lamb=np.concatenate([[0.0], np.full(N_BATCHES, LAMB)]).astype(np.float32),
            alpha=ALPHA,
            lambda_estimation=False,
            max_iter_harmony=1,
            max_iter_kmeans=1,
            epsilon_kmeans=1e-5,
            epsilon_harmony=1e-4,
            K=N_CLUSTERS,
            block_size=block_size,
            verbose=False,
            random_state=0,
            device="cpu",
        )
    finally:
        Harmony.harmonize = real_harmonize


def np_(t):
    return t.detach().cpu().numpy().astype(np.float64)


def main() -> None:
    torch.manual_seed(SEED)
    z_dn, phi, pr_b, labels = build_fixture()

    # ── The fixture is harmonypy's own `init_cluster` output ─────────────────
    ho = new_state(z_dn, phi, pr_b)
    fix_y = np_(ho._Y)  # (d x K)
    fix_r = np_(ho._R)  # (K x N)
    fix_o = np_(ho._O)  # (K x B)
    fix_e = np_(ho._E)  # (K x B)
    fix_dist = np_(ho._dist_mat)  # (K x N)
    fix_z_cos = np_(ho._Z_cos)  # (d x N)

    # ── Arm 1: the M-step and the distances it invalidates ───────────────────
    # `cluster()` at max_iter_kmeans=1 does exactly: Y <- normalize(Z_cos R'),
    # dist <- 2(1 - Y' Z_cos), then update_R and compute_objective. Y and
    # dist_mat are written before those two, so reading them after the call
    # gives the M-step and the distance kernel applied to the fixture's R.
    a1 = new_state(z_dn, phi, pr_b)
    a1.cluster()
    mstep_y = np_(a1._Y)
    mstep_dist = np_(a1._dist_mat)

    # ── Arm 2: the ridge correction ──────────────────────────────────────────
    a2 = new_state(z_dn, phi, pr_b)
    a2.moe_correct_ridge()
    ridge_z_corr = np_(a2._Z_corr)  # (d x N)

    # ── Arm 3: update_R's softmax half, with the penalty switched off ────────
    a3 = new_state(z_dn, phi, pr_b, theta=SOFTMAX_ARM_THETA)
    a3.update_R()
    softmax_r = np_(a3._R)

    # ── Arm 4: the objective, decomposed ─────────────────────────────────────
    # Two of the three components are parity arms (SCX and harmonypy compute the
    # same formula); the cross-entropy is the divergence arm.
    a4 = new_state(z_dn, phi, pr_b)
    a4.compute_objective()
    obj_dist = a4.objective_kmeans_dist[-1]
    obj_entropy = a4.objective_kmeans_entropy[-1]
    obj_cross = a4.objective_kmeans_cross[-1]

    # ── Checked premises. A generator that only prints its bars has no bars. ──
    assert fix_r.shape == (N_CLUSTERS, N_CELLS)
    assert np.allclose(fix_r.sum(axis=0), 1.0, atol=1e-5), "fixture R is not a distribution"

    # The M-step must MOVE the centroids, or the red-first test is vacuous:
    # a fixture whose init centroids already equal the R-weighted means would
    # pass with the M-step deleted.
    y_shift = float(np.abs(mstep_y - fix_y).max())
    assert y_shift > 1e-2, (
        f"M-step barely moved the centroids (max |dY| = {y_shift:.3e}); this "
        "fixture cannot distinguish an implementation with no M-step"
    )
    dist_shift = float(np.abs(mstep_dist - fix_dist).max())
    assert dist_shift > 1e-3, (
        f"distances barely moved (max |dD| = {dist_shift:.3e}); same problem"
    )

    # The ridge correction must actually correct something.
    ridge_shift = float(np.abs(ridge_z_corr - np_(a2._Z_orig)).max())
    assert ridge_shift > 1e-2, f"ridge moved nothing (max = {ridge_shift:.3e})"

    # The softmax arm must still be a distribution — this is the check that
    # would have caught the `block_size = 1.0` degeneracy described above, where
    # harmonypy's clamp left every column summing to ~3e-8.
    col = softmax_r.sum(axis=0)
    assert np.allclose(col, 1.0, atol=1e-5), (
        f"update_R's output is not a distribution (col sums in "
        f"[{col.min():.3e}, {col.max():.3e}]) - the penalty clamp degenerated"
    )

    # The objective's cross-entropy must be large enough that the divergence
    # from SCX's form is visible above f32 noise on this fixture.
    assert abs(obj_cross) > 1.0, (
        f"cross-entropy is {obj_cross:.3e}; too small to carry a divergence claim"
    )

    # Every batch must survive SCX's default `batch_prop_cutoff` of 1e-5, so the
    # ridge arm compares full solves rather than SCX's pruned one.
    n_b = phi.sum(axis=1).astype(np.float64)
    avg_r = fix_o / n_b[None, :]
    assert avg_r.min() > 1e-5, (
        f"a batch would be pruned by SCX (min avg R = {avg_r.min():.3e})"
    )

    v = {p: md.version(p) for p in ("harmonypy", "torch", "numpy", "scikit-learn")}

    # ── Emit ─────────────────────────────────────────────────────────────────
    print("//! Pinned harmonypy reference values for Harmony (§7.4, ORG-7.21-4).")
    print("//!")
    print("//! GENERATED FILE — do not edit by hand. See `# Regenerating` below.")
    print("//!")
    print("//! # Why an external reference at all")
    print("//!")
    print("//! `harmony/tests.rs` is entirely self-consistency: soft assignments sum")
    print("//! to one, O and E agree, the arrowhead inverse matches an LU of the same")
    print("//! matrix, the same seed gives the same bytes. None of it could see that")
    print("//! the soft k-means sub-loop had **no M-step** — `y` and `dist_mat` were")
    print("//! frozen for the whole loop, so the E-step could only trade `Σ R·dist`")
    print("//! against the entropy penalty at fixed centroids. The one test that")
    print("//! compared against a third party (`test_harmony_validation.py`) skips")
    print("//! whenever its gitignored 50 MB `.npz` is absent, which is always in CI.")
    print("//!")
    print("//! # Provenance")
    print("//!")
    print("//! | constant | produced by |")
    print("//! |---|---|")
    print("//! | `HP_Z`, `HP_LABELS` | this generator's RNG — the fixture INPUT |")
    print("//! | `HP_FIX_*` | harmonypy `Harmony.init_cluster` |")
    print("//! | `HP_MSTEP_Y`, `HP_MSTEP_DIST` | harmonypy `Harmony.cluster` |")
    print("//! | `HP_RIDGE_Z_CORR` | harmonypy `Harmony.moe_correct_ridge` |")
    print("//! | `HP_SOFTMAX_R` | harmonypy `Harmony.update_R` at `theta = 0` |")
    print("//! | `HP_OBJ_*` | harmonypy `Harmony.compute_objective` |")
    print("//!")
    for p, ver in v.items():
        print(f"//! {p} {ver}")
    print("//!")
    print("//! # Conventions, and how the two layouts line up")
    print("//!")
    print("//! harmonypy holds `Z` and `Y` as `(d × N)` / `(d × K)` torch tensors and")
    print("//! `R` / `dist_mat` as `(K × N)`. SCX holds `z_orig` column-major `d × N`")
    print("//! (`z[i*d + t]`), `y` column-major `d × K` (`y[ku*d + t]`), and `r` /")
    print("//! `dist_mat` row-major `K × N`. So a harmonypy `(d × ·)` matrix is")
    print("//! emitted **transposed** — one source row per column — and a `(K × N)`")
    print("//! one is emitted as-is. The tests flatten in that order and nowhere else.")
    print("//!")
    print("//! # The bars, and why they are not tighter")
    print("//!")
    print("//! harmonypy 0.2.0 is torch **float32** end to end (`allocate_buffers`),")
    print("//! while SCX accumulates in f64 and stores `r` / `dist_mat` as f32. The")
    print("//! parity arms are therefore f32-reference comparisons and cannot be")
    print("//! pinned at `abs=0`; each `*_ATOL` below carries the value observed when")
    print("//! this file was generated, and is that value rounded up one decimal")
    print("//! order. Nothing here is calibrated from what SCX returns.")
    print("//!")
    print("//! # Two divergences are pinned AS divergences")
    print("//!")
    print("//! SCX's diversity penalty is `((2E+1)/(O+E+1))^θ` where harmonypy 0.2.0")
    print("//! uses `(E/(O+E))^θ`, and SCX's objective cross-entropy is")
    print("//! `log((O+E+1)/(2E+1))` where harmonypy uses `log((O+E)/E)`. The factor")
    print("//! of 2 cancels in `update_R` (it is constant across `k` for a fixed cell,")
    print("//! so the per-cell L1 normalization removes it); the `+1` smoothing does")
    print("//! not, and in the objective the factor survives as a `−log(2)·Σ σ O θ`")
    print("//! term. Both forms are self-consistent and GPU-matched, so they are")
    print("//! recorded with their measured magnitude rather than changed.")
    print("//!")
    print("//! The **penalty** divergence cannot be pinned against harmonypy directly:")
    print("//! `update_R` shuffles cells and updates O/E block by block, and the two")
    print("//! implementations draw that permutation from different RNGs. Forcing one")
    print("//! block removes every cell at once, which sends O and E to zero and drops")
    print("//! harmonypy into its `clamp(E/(O+E), min=1e-8)` floor — its R stops")
    print("//! summing to one entirely. So [`HP_OBJ_CROSS`] carries the divergence for")
    print("//! the whole penalty family (same ratio, inverted, and deterministic), and")
    print("//! [`HP_SOFTMAX_R`] pins the half of `update_R` that has no penalty in it.")
    print("//!")
    print("//! # Regenerating")
    print("//!")
    print("//! ```text")
    print("//! .venv/bin/python benchmarks/scripts/generate_harmony_references.py \\")
    print("//!     > scx-accel/src/harmony/harmony_reference_values.rs")
    print("//! ```")
    print()
    print("#![allow(clippy::unreadable_literal)]")
    print()
    print(f"pub const HP_N_CELLS: usize = {N_CELLS};")
    print(f"pub const HP_N_PCS: usize = {N_PCS};")
    print(f"pub const HP_N_CLUSTERS: usize = {N_CLUSTERS};")
    print(f"pub const HP_N_BATCHES: usize = {N_BATCHES};")
    print(f"pub const HP_SIGMA: f64 = {f64_literal(SIGMA)};")
    print(f"pub const HP_THETA: f64 = {f64_literal(THETA)};")
    print(f"pub const HP_LAMB: f64 = {f64_literal(LAMB)};")
    print(f"pub const HP_BLOCK_SIZE: f64 = {f64_literal(BLOCK_SIZE)};")
    print()
    print("// --- Fixture input: the embedding, row-major (N x d) as pyscx hands it in.")
    print("pub " + rust_matrix("HP_Z", z_dn.T.astype(np.float64), ty="f64"))
    print()
    lab = ", ".join(str(int(x)) for x in labels)
    print(f"pub const HP_LABELS: [u32; HP_N_CELLS] = [{lab}];")
    print()
    print("// --- Fixture state, produced by harmonypy `init_cluster` ---------------")
    print("// `HP_FIX_Y` is transposed to SCX's (K rows of d) centroid layout.")
    print("pub " + rust_matrix("HP_FIX_Y", fix_y.T, ty="f64"))
    print()
    print("pub " + rust_matrix("HP_FIX_R", fix_r, ty="f64"))
    print()
    print("pub " + rust_matrix("HP_FIX_DIST", fix_dist, ty="f64"))
    print()
    print("pub " + rust_matrix("HP_FIX_O", fix_o, ty="f64"))
    print()
    print("pub " + rust_matrix("HP_FIX_E", fix_e, ty="f64"))
    print()
    print("// --- Arm 1: the M-step, and the distances it invalidates ---------------")
    print(f"// `HP_MSTEP_Y` moves the fixture centroids by max |dY| = {y_shift:.3e},")
    print(f"// and `HP_MSTEP_DIST` moves the distances by {dist_shift:.3e}. Both are")
    print("// checked by the generator: a fixture whose init centroids already are the")
    print("// R-weighted means would pass with the M-step deleted.")
    print("pub " + rust_matrix("HP_MSTEP_Y", mstep_y.T, ty="f64"))
    print()
    print("pub " + rust_matrix("HP_MSTEP_DIST", mstep_dist, ty="f64"))
    print()
    print("// --- Arm 2: the ridge correction ---------------------------------------")
    print(f"// Moves the embedding by max |dZ| = {ridge_shift:.3e}.")
    print("// Transposed to SCX's row-major (N x d) `z_corr` read order.")
    print("pub " + rust_matrix("HP_RIDGE_Z_CORR", ridge_z_corr.T, ty="f64"))
    print()
    print("// --- Arm 3: update_R's softmax half, penalty switched off --------------")
    print(f"// `theta = {SOFTMAX_ARM_THETA}` makes the diversity penalty exactly 1 on both")
    print("// sides for any O/E, and `block_size = 1.0` makes both permutation-")
    print("// invariant, so this arm is deterministic across the two RNGs. It does")
    print("// NOT pin the penalty - see the module doc for why that cannot be pinned")
    print("// against harmonypy at all, and what carries the divergence instead.")
    print(f"pub const HP_SOFTMAX_ARM_THETA: f64 = {f64_literal(SOFTMAX_ARM_THETA)};")
    print("pub " + rust_matrix("HP_SOFTMAX_R", softmax_r, ty="f64"))
    print()
    print("// --- Arm 4: the objective, decomposed ----------------------------------")
    print("// `DIST` and `ENTROPY` are parity arms — SCX computes the same formula.")
    print("// `CROSS` is the divergence arm. All three carry the 2000/N scaling both")
    print("// implementations apply.")
    print(f"pub const HP_OBJ_DIST: f64 = {f64_literal(obj_dist)};")
    print(f"pub const HP_OBJ_ENTROPY: f64 = {f64_literal(obj_entropy)};")
    print(f"pub const HP_OBJ_CROSS: f64 = {f64_literal(obj_cross)};")
    print()
    print("// --- Bars ---------------------------------------------------------------")
    print("// The only constants here harmonypy did not produce. Each is the max")
    print("// |SCX - harmonypy| measured Rust-side, rounded up one decimal order.")
    for name, (val, why) in BARS.items():
        print(f"/// {why}")
        print(f"pub const {name}: f64 = {f64_literal(val)};")


# ─── LISI (§7.19) ────────────────────────────────────────────────────────────

LISI_N_CELLS = 60
LISI_N_DIMS = 4
LISI_N_LABELS = 3
LISI_PERPLEXITY = 4  # int: harmonypy passes `perplexity * 3` straight to sklearn
LISI_SEED = 20260825

# Measured Rust-side, as in BARS above.
LISI_ATOL = (1e-6, "observed 2.220e-16 on f64 both sides; the sweep is exact")


def emit_lisi() -> None:
    """Pin `compute_lisi` against harmonypy's own `compute_lisi`.

    The neighbourhood is the whole point of this arm. harmonypy retrieves
    `3 * perplexity` neighbours and drops column 0 — the self-match — so its
    effective neighbourhood is `3 * perplexity - 1`. SCX skips `j == i` while
    collecting, so it must ask for one fewer to see the same cells. That
    off-by-one was in three places (§7.19) and no test could see it, because
    the only harmonypy comparison in the tree ran at `atol = 1e-2`, which is
    wider than the error one extra neighbour causes.

    A small `perplexity` is deliberate: at the default 30 the neighbourhood is
    89 of any tractable fixture's cells, so dropping one changes almost nothing
    and the arm would pass either way. The generator checks that below.
    """
    import pandas as pd
    from harmonypy.lisi import compute_lisi as hpy_lisi

    rng = np.random.default_rng(LISI_SEED)
    labels = rng.integers(0, LISI_N_LABELS, size=LISI_N_CELLS)
    # Label-correlated blobs that OVERLAP. Well-separated blobs pin LISI at
    # 1.0 for every cell, where the metric is flat and one neighbour more or
    # less changes nothing — the fixture would pass against either convention.
    # At this separation LISI spans roughly [1.0, 2.6] on 3 labels.
    centers = rng.normal(0.0, 0.8, size=(LISI_N_LABELS, LISI_N_DIMS))
    x = centers[labels] + rng.normal(0.0, 1.4, size=(LISI_N_CELLS, LISI_N_DIMS))
    x = x.astype(np.float32).astype(np.float64)

    meta = pd.DataFrame({"batch": labels})
    want = hpy_lisi(x, meta, ["batch"], perplexity=LISI_PERPLEXITY)[:, 0]

    k_hpy = int(np.ceil(LISI_PERPLEXITY * 3))

    # The wrong-convention answer, as an explicit repulsion target. Only the
    # *retrieval width* is patched — harmonypy's kernel, its self-match drop and
    # its Simpson index all still run — so this is what harmonypy would return
    # if SCX's neighbourhood were handed to it. Nothing asserts SCX equals it;
    # the arm asserts SCX is far from it, the shape `pflog_reference_values`
    # uses for its pre-fix estimator.
    import harmonypy.lisi as _hl

    real_nn = _hl.NearestNeighbors
    _hl.NearestNeighbors = lambda n_neighbors, **kw: real_nn(
        n_neighbors=n_neighbors + 1, **kw
    )
    try:
        off_by_one = hpy_lisi(x, meta, ["batch"], perplexity=LISI_PERPLEXITY)[:, 0]
    finally:
        _hl.NearestNeighbors = real_nn

    sep = float(np.abs(want - off_by_one).max())
    assert sep > 1e-2, (
        f"one extra neighbour moves LISI by only {sep:.3e} on this fixture; "
        "the arm cannot detect the off-by-one it was written for"
    )
    assert LISI_ATOL[0] * 100 < sep, (
        f"the bar {LISI_ATOL[0]:.1e} is not comfortably below the one-neighbour "
        f"shift {sep:.3e}; the arm would not separate the two conventions"
    )
    assert 1.2 < want.mean() < LISI_N_LABELS, (
        f"mean LISI {want.mean():.3f} is too close to the unmixed bound of 1.0; "
        "the metric is flat there and the fixture cannot separate the two "
        "neighbourhood conventions"
    )

    v = {p: md.version(p) for p in ("harmonypy", "numpy", "scikit-learn", "pandas")}

    print("//! Pinned harmonypy reference values for LISI (§7.19, ORG-7.21-4).")
    print("//!")
    print("//! GENERATED FILE — do not edit by hand. See `# Regenerating` below.")
    print("//!")
    print("//! # Why an external reference at all")
    print("//!")
    print("//! SCX's LISI asked for `3 * perplexity` neighbours where harmonypy's")
    print("//! effective neighbourhood is `3 * perplexity - 1`: it retrieves")
    print("//! `3 * perplexity` from `NearestNeighbors` and drops column 0, the")
    print("//! self-match, while SCX's sweep skips `j == i` as it collects. The")
    print("//! derivation was written out three times — `LisiConfig::default`,")
    print("//! `pyscx/src/accel/lisi.rs` and `rscx/src/lisi.rs` — and all three said")
    print("//! `3 * perplexity`. The one harmonypy comparison in the tree")
    print("//! (`test_harmony_validation.py::test_lisi_matches_harmonypy_per_cell`)")
    print("//! ran at `atol = 1e-2` and passed with the bug.")
    print("//!")
    print("//! # Provenance")
    print("//!")
    print("//! | constant | produced by |")
    print("//! |---|---|")
    print("//! | `LISI_X`, `LISI_LABELS` | this generator's RNG — the fixture INPUT |")
    print("//! | [`LISI_EXPECTED`] | harmonypy `harmonypy.lisi.compute_lisi` |")
    print("//! | [`LISI_OFF_BY_ONE`] | the same, with only the retrieval width patched |")
    print("//!")
    for p_, ver in v.items():
        print(f"//! {p_} {ver}")
    print("//!")
    print("//! # Why the perplexity is 4 and not the default 30")
    print("//!")
    print("//! At perplexity 30 the neighbourhood is 89 cells, so on any fixture")
    print("//! small enough to read as literals one neighbour more or less changes")
    print("//! almost nothing and the arm would pass against either convention. At")
    print(f"//! perplexity {LISI_PERPLEXITY} the neighbourhood is {k_hpy - 1} of {LISI_N_CELLS} cells and one extra")
    print(f"//! neighbour moves LISI by up to {sep:.3e} — checked by the generator, which")
    print("//! refuses a fixture that cannot separate the two conventions.")
    print("//!")
    print("//! # Regenerating")
    print("//!")
    print("//! ```text")
    print("//! .venv/bin/python benchmarks/scripts/generate_harmony_references.py lisi \\")
    print("//!     > scx-accel/src/lisi_reference_values.rs")
    print("//! ```")
    print()
    print("#![allow(clippy::unreadable_literal)]")
    print()
    print(f"pub const LISI_N_CELLS: usize = {LISI_N_CELLS};")
    print(f"pub const LISI_N_DIMS: usize = {LISI_N_DIMS};")
    print(f"pub const LISI_N_LABELS: usize = {LISI_N_LABELS};")
    print(f"pub const LISI_PERPLEXITY: f64 = {f64_literal(float(LISI_PERPLEXITY))};")
    print("/// harmonypy retrieves this many, then drops the self-match.")
    print(f"pub const LISI_HARMONYPY_RETRIEVED: usize = {k_hpy};")
    print(f"/// {LISI_ATOL[1]}")
    print(f"pub const LISI_ATOL: f64 = {f64_literal(LISI_ATOL[0])};")
    print("/// Max |LISI| shift from one extra neighbour, measured by the generator.")
    print("/// The off-by-one arm asserts the bar is far below this.")
    print(f"pub const LISI_ONE_NEIGHBOUR_SHIFT: f64 = {f64_literal(sep)};")
    print()
    print("pub " + rust_matrix("LISI_X", x, ty="f64"))
    print()
    lab = ", ".join(str(int(t)) for t in labels)
    print(f"pub const LISI_LABELS: [u32; LISI_N_CELLS] = [{lab}];")
    print()
    print("pub " + rust_f64_array("LISI_EXPECTED", want))
    print()
    print("/// harmonypy's answer if it were handed SCX's pre-fix neighbourhood —")
    print("/// one cell wider. A REPULSION target: nothing asserts SCX equals it.")
    print("pub " + rust_f64_array("LISI_OFF_BY_ONE", off_by_one))


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "harmony"
    if target == "harmony":
        main()
    elif target == "lisi":
        emit_lisi()
    else:
        sys.exit(f"unknown target {target!r}; expected 'harmony' or 'lisi'")
