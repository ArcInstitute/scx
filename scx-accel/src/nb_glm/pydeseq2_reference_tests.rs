//! NB-GLM against a real pydeseq2 run, at the bar the docs actually claim
//! (ORG-7.21-4).
//!
//! # The bar, and why it is not numerical equality
//!
//! `docs/pseudobulk_nb_glm.md` states it outright: *"The bar is ranking /
//! effect-sign / significance parity, not numerical equality. If you need exact
//! DESeq2 behaviour, use PyDESeq2."* SCX omits apeglm LFC shrinkage and handles
//! dispersion outliers differently, both deliberately. So this file asserts
//! ranking, sign and significance — and a deliberately loose gross-drift canary
//! that is labelled as *not* a parity claim.
//!
//! Asserting numerical equality here would pin behaviour the documentation
//! promises not to have, and would go red on the next legitimate divergence
//! rather than on a bug.
//!
//! # What this replaces
//!
//! `scx-accel/tests/nb_glm_reference.rs` is named "reference" and contains only
//! never-panic property tests — the review's point was that no pinned
//! pydeseq2/DESeq2 comparison existed anywhere in the Rust suite. It now says
//! what it is and points here.
//!
//! # Regenerating
//!
//! ```text
//! .venv/bin/python benchmarks/scripts/generate_de_parity_references.py
//! ```
//!
//! The counts are literals rather than a seed plus a distribution, so pydeseq2
//! and this fitter see byte-identical input. A seed would make the reference
//! depend on numpy's generator staying put across versions.

use crate::diffexp::cpu::for_each_tie_run;
use crate::nb_glm::{pseudobulk_nb_glm, NbGlmContrast, NbGlmOptions};
const NB_N_GENES: usize = 24;

const NB_N_SAMPLES: usize = 8;

const NB_COUNTS_GENE_MAJOR: [[f64; 8]; 24] = [
    [34.0, 61.0, 65.0, 53.0, 193.0, 196.0, 142.0, 239.0],
    [67.0, 107.0, 86.0, 100.0, 179.0, 196.0, 202.0, 263.0],
    [94.0, 94.0, 58.0, 73.0, 883.0, 759.0, 498.0, 983.0],
    [128.0, 278.0, 156.0, 197.0, 396.0, 374.0, 285.0, 573.0],
    [483.0, 526.0, 495.0, 243.0, 1848.0, 1450.0, 1012.0, 971.0],
    [146.0, 72.0, 70.0, 119.0, 25.0, 32.0, 22.0, 41.0],
    [294.0, 296.0, 353.0, 280.0, 62.0, 106.0, 35.0, 58.0],
    [133.0, 124.0, 68.0, 87.0, 25.0, 24.0, 40.0, 23.0],
    [278.0, 212.0, 316.0, 341.0, 161.0, 135.0, 157.0, 141.0],
    [428.0, 330.0, 372.0, 323.0, 554.0, 435.0, 456.0, 487.0],
    [141.0, 191.0, 254.0, 415.0, 132.0, 249.0, 424.0, 227.0],
    [501.0, 601.0, 380.0, 463.0, 273.0, 476.0, 500.0, 286.0],
    [183.0, 122.0, 129.0, 198.0, 132.0, 163.0, 231.0, 205.0],
    [114.0, 109.0, 77.0, 142.0, 193.0, 241.0, 230.0, 131.0],
    [451.0, 433.0, 392.0, 235.0, 440.0, 303.0, 494.0, 341.0],
    [137.0, 178.0, 222.0, 223.0, 177.0, 228.0, 230.0, 140.0],
    [354.0, 398.0, 195.0, 215.0, 300.0, 218.0, 838.0, 279.0],
    [100.0, 85.0, 97.0, 97.0, 84.0, 50.0, 51.0, 48.0],
    [119.0, 98.0, 127.0, 76.0, 133.0, 141.0, 234.0, 152.0],
    [244.0, 196.0, 129.0, 131.0, 189.0, 113.0, 139.0, 108.0],
    [392.0, 227.0, 357.0, 297.0, 218.0, 355.0, 330.0, 149.0],
    [404.0, 311.0, 288.0, 316.0, 230.0, 169.0, 309.0, 287.0],
    [23.0, 22.0, 34.0, 46.0, 39.0, 45.0, 49.0, 45.0],
    [140.0, 84.0, 172.0, 157.0, 131.0, 120.0, 108.0, 133.0],
];

const NB_CONDITION: [f64; NB_N_SAMPLES] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

const NB_PYDESEQ2_LOG2FC: [f64; NB_N_GENES] = [
    1.786553474739395,
    1.15024419287027,
    3.2477336378120767,
    1.0314179897922129,
    1.5058333908404595,
    -1.7838632683845612,
    -2.2922354739899307,
    -1.972602031992389,
    -1.0110410715093314,
    0.34958224878262345,
    -0.05065701467463062,
    -0.44897274759469324,
    0.15333571546068206,
    0.7670296425502828,
    -0.037014081248639086,
    -0.06702976615424872,
    0.34350612888257487,
    -0.7671106627673717,
    0.5555392083273769,
    -0.4254887790275968,
    -0.3608848358843099,
    -0.4732338665328264,
    0.44653043827646993,
    -0.21437737670883789,
];

const NB_PYDESEQ2_PADJ: [f64; NB_N_GENES] = [
    6.914957231132138e-08,
    0.00013192524714671409,
    1.7689395169443884e-24,
    0.005900763641760173,
    1.0360059018767435e-05,
    9.573843777391615e-08,
    6.883463040968317e-12,
    1.4560069704723782e-09,
    4.632999937263793e-05,
    0.2043981383363031,
    0.8942148122069095,
    0.16514062838300073,
    0.6249774957343404,
    0.007808171932202532,
    0.8942148122069095,
    0.8693444433104985,
    0.4406096601779062,
    0.010885016957806696,
    0.0776681086850401,
    0.2043981383363031,
    0.25305903641461597,
    0.11457012289397461,
    0.2043981383363031,
    0.5102956614685423,
];

/// Significance threshold. DESeq2's own default, and pydeseq2's.
const NB_ALPHA_LEVEL: f64 = 0.05;

/// 1-based mid-ranks, tie-aware, via the crate's **one** tie-run walk.
///
/// Deliberately not a local ranking loop: ORG-7.21-3 collapsed the two copies
/// of this walk that existed in the DE kernels, and re-adding a third here — in
/// a test whose whole subject is single-sourcing — would be the same mistake in
/// miniature. Spearman needs the mid-ranks and not the `Σ(t³−t)` correction, so
/// the returned tie term is dropped.
fn mid_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0f64; values.len()];
    let _tie_correction = for_each_tie_run(
        values.len(),
        0,
        |i, j| values[order[j]] == values[order[i]],
        |mid_rank, run| {
            for &idx in &order[run] {
                ranks[idx] = mid_rank;
            }
        },
    );
    ranks
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += (x - ma) * (y - mb);
        da += (x - ma) * (x - ma);
        db += (y - mb) * (y - mb);
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

/// Spearman rank correlation, tie-aware through [`mid_ranks`].
fn spearman(a: &[f64], b: &[f64]) -> f64 {
    pearson(&mid_ranks(a), &mid_ranks(b))
}

/// A filtered-out gene reads as "not significant", never as rank-first.
///
/// DESeq2 semantics put `NaN` in `padj` for genes dropped by Cook's-distance or
/// base-mean independent filtering. Mapping those to 1.0 is what pydeseq2's own
/// consumers do; leaving them as `NaN` would sort them arbitrarily and make the
/// rank correlation meaningless rather than merely lower.
fn padj_or_one(p: &[f64]) -> Vec<f64> {
    p.iter()
        .map(|&v| if v.is_nan() { 1.0 } else { v })
        .collect()
}

fn fit() -> crate::nb_glm::NbGlmResult {
    let counts: Vec<f64> = NB_COUNTS_GENE_MAJOR
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    // Row-major `[n_samples × n_features]`: intercept + condition, the design
    // `~condition` compiles to.
    let design: Vec<f64> = NB_CONDITION.iter().flat_map(|&c| [1.0, c]).collect();
    pseudobulk_nb_glm(
        &counts,
        NB_N_GENES,
        NB_N_SAMPLES,
        &design,
        2,
        None, // median-ratio size factors, as DESeq2 computes them
        NbGlmContrast::Coefficient { index: 1 },
        NbGlmOptions::default(),
    )
    .expect("nb_glm on the pydeseq2 reference fixture")
}

fn significant(padj: &[f64]) -> Vec<usize> {
    padj_or_one(padj)
        .iter()
        .enumerate()
        .filter(|(_, &p)| p < NB_ALPHA_LEVEL)
        .map(|(g, _)| g)
        .collect()
}

/// The fixture separates significant from non-significant, in both directions.
///
/// The premise every assertion below rests on. Without it, "the significant
/// sets match" is satisfied by both sides calling everything significant, or
/// nothing — and a set-equality assertion over two empty sets is the quietest
/// possible pass.
#[test]
fn the_fixture_has_genes_on_both_sides_of_significance_and_of_zero() {
    let sig = significant(&NB_PYDESEQ2_PADJ);
    assert!(
        !sig.is_empty() && sig.len() < NB_N_GENES,
        "pydeseq2 calls {} of {} genes significant — a set-equality assertion \
         over all or nothing proves nothing",
        sig.len(),
        NB_N_GENES
    );
    let up = sig.iter().filter(|&&g| NB_PYDESEQ2_LOG2FC[g] > 0.0).count();
    let down = sig.len() - up;
    assert!(
        up > 0 && down > 0,
        "the significant set is {up} up / {down} down; a sign-parity assertion \
         needs both directions or it cannot see a sign flip"
    );
}

/// **Significance parity** — the significant sets match exactly.
///
/// Exact set equality, not a Jaccard overlap: this is the decision an analyst
/// actually takes off the output, and a gene appearing on one side only is a
/// different answer, not a rounding difference.
#[test]
fn nb_glm_calls_the_same_genes_significant_as_pydeseq2() {
    let got = fit();
    let ours = significant(&got.p_adj);
    let theirs = significant(&NB_PYDESEQ2_PADJ);
    assert_eq!(
        ours, theirs,
        "significant sets differ at padj < {}: scx {:?} vs pydeseq2 {:?}",
        NB_ALPHA_LEVEL, ours, theirs
    );
}

/// **Effect-sign parity** — every gene, not just the significant ones.
///
/// All 24, because a sign flip on a gene nobody calls significant is still a
/// sign flip, and restricting to the significant set would let one hide behind
/// its own p-value.
#[test]
fn nb_glm_agrees_with_pydeseq2_on_every_effect_sign() {
    let got = fit();
    for (gene, (&ours, &theirs)) in got
        .log2_fold_change
        .iter()
        .zip(NB_PYDESEQ2_LOG2FC.iter())
        .enumerate()
    {
        assert_eq!(
            ours.signum(),
            theirs.signum(),
            "gene g{gene}: scx log2FC {ours} vs pydeseq2 {theirs} — opposite signs"
        );
    }
}

/// **Ranking parity** — Spearman on `log2FoldChange` and on `padj`.
///
/// Observed on this fixture: `log2FC` ρ = 1.00000 (the two orderings are
/// identical), `padj` ρ = 0.99151. The `padj` bar is the looser of the two
/// because the two implementations *do* swap a few adjacent genes inside the
/// significant block — so **ordered-list equality is explicitly not the claim**,
/// and asserting it would fail today for a reason the docs already allow.
#[test]
fn nb_glm_ranks_genes_the_way_pydeseq2_does() {
    let got = fit();

    let rho_lfc = spearman(&got.log2_fold_change, &NB_PYDESEQ2_LOG2FC);
    assert!(
        rho_lfc >= 0.999,
        "log2FC rank correlation with pydeseq2 is {rho_lfc:.6}, below 0.999"
    );

    let rho_padj = spearman(&padj_or_one(&got.p_adj), &padj_or_one(&NB_PYDESEQ2_PADJ));
    assert!(
        rho_padj >= 0.99,
        "padj rank correlation with pydeseq2 is {rho_padj:.6}, below 0.99"
    );
}

/// A gross-drift canary. **Not** a parity claim.
///
/// The docs promise ranking / sign / significance and explicitly not numerical
/// equality, so this bound is set two decimal orders above the observed
/// max |Δlog2FC| of 0.0024 — loose enough that the divergences SCX documents
/// (no apeglm shrinkage; different dispersion-outlier handling) stay inside it,
/// tight enough that a fitter that had stopped fitting would not.
///
/// It exists because the three tests above are all rank- and sign-based: a
/// systematic scaling of every effect by, say, 1.5 preserves order, sign and
/// significance, and would pass all of them.
#[test]
fn nb_glm_effects_have_not_drifted_grossly_from_pydeseq2() {
    let got = fit();
    let worst = got
        .log2_fold_change
        .iter()
        .zip(NB_PYDESEQ2_LOG2FC.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst <= 0.1,
        "max |Δlog2FC| vs pydeseq2 is {worst:.4}, above the 0.1 gross-drift \
         bound (observed 0.0024 when pinned). This is not a parity failure by \
         itself — check whether the fitter changed or the reference did."
    );
}
