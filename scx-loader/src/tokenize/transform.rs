//! Per-row value transforms — the normalise / log / pflog family, as kernels.
//!
//! These are the numerics `collate_cell` used to inline under
//! [`PreprocessMode`](crate::sparse_cellset_collate::PreprocessMode). They are
//! lifted here unchanged so a consumer can apply one to a gathered CSR without
//! going through the collator, and so the collator's mode dispatch is a table
//! over named kernels rather than a `match` with four bodies.
//!
//! Every kernel writes `dst` from `src` element-wise and **clips negatives to
//! zero first**, matching the gather stage's `clip_negatives`. NaN clips to zero
//! too, because `f32::max(0.0)` returns the non-NaN operand — that differs from
//! `numpy.maximum`, which propagates NaN, and it is the behaviour the gather
//! stage has always had.
//!
//! # These are the sparse-row twins of `crate::normalize`
//!
//! `normalize.rs` holds the **dense**-row versions used by `TrainingDataset`
//! (R1): `normalize_dense_row`, `log1p_dense_row`, `pflog_baseline_row`. The two
//! families stay separate. They take different inputs (a dense row over the full
//! vocabulary vs. a CSR row's values), and `pflog` in particular means different
//! things in the two places — the dense path decomposes into a per-cell baseline
//! plus a delta, while the collate path centres by a caller-declared measured
//! panel size. Unifying them is a behaviour change, not a cleanup.
//!
//! # Library size is two different quantities; name which
//!
//! [`library_size`] sums the row as given. If the row has already been projected
//! onto a gene panel, that is the library size **after** feature filtering, and
//! it is not the cell's sequencing depth. The collator makes the distinction
//! explicit with its `lib_size_redef` flag; a standalone caller has to make it
//! themselves, and a model whose normalisation statistic was computed against
//! whole-cell depth must not be fed a panel-filtered sum.

/// Sum of the row's values, negatives clipped to zero.
///
/// Accumulated in `f64`. Raw counts below 2^24 are exact in `f32`, so the sum is
/// order-independent and a parallel reduction would give the same answer; the
/// `f64` accumulator keeps that true for transformed values too.
#[inline]
pub fn library_size(values: &[f32]) -> f64 {
    values.iter().map(|&v| v.max(0.0) as f64).sum()
}

/// `dst[i] = max(src[i], 0)`.
pub fn pass_through(src: &[f32], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.max(0.0);
    }
}

/// `dst[i] = ln(1 + max(src[i], 0))` — Geneformer/scGPT-style shifted log on raw
/// counts, with no depth normalisation.
pub fn log1p_raw(src: &[f32], dst: &mut [f32]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = s.max(0.0).ln_1p();
    }
}

/// `dst[i] = ln(1 + max(src[i], 0) * target_sum / lib)` — scanpy's
/// `normalize_total` + `log1p`, fused.
///
/// `lib` is passed in rather than recomputed because the caller usually already
/// has it and because a panel-filtered row's depth is not its own sum (see the
/// module docs). `lib <= 0` leaves the values unscaled rather than dividing by
/// zero, which is what an empty cell needs.
pub fn normalize_log1p(src: &[f32], dst: &mut [f32], target_sum: f64, lib: f64) {
    let positive = lib > 0.0;
    let factor = if positive {
        (target_sum / lib) as f32
    } else {
        1.0
    };
    for (d, &s) in dst.iter_mut().zip(src) {
        let rc = s.max(0.0);
        let scaled = if positive { rc * factor } else { rc };
        *d = scaled.ln_1p();
    }
}

/// PFlog **v4**: `dst[i] = ln(1 + 4α·max(src[i], 0)) − centre`, where
/// `centre = Σ ln(1 + 4α·rc) / n_measured`.
///
/// The matrix-wide Anscombe pseudocount is `1/(4α)`; `α` is the NB overdispersion
/// and is an **input**, never estimated here. There is no per-cell depth term, so
/// an all-zero row gives all-zero encoder values (every term 0 ⇒ centre 0).
///
/// ⚠️ This is not state3 `main`'s `pflog1ppf_raw`, which is
/// `ln(1 + c / library_size)` centred by the measured-gene **count**. The two are
/// distinct named transforms and must not be substituted for one another; that
/// substitution once shipped under an unchanged contract version, which is why
/// the version's scope is now enumerated.
///
/// `n_measured` is the measured panel size — the denominator `D`. Passing `0`
/// yields a non-finite centre; the collator validates it upstream.
pub fn pflog_raw(src: &[f32], dst: &mut [f32], alpha: f64, n_measured: usize) {
    let four_alpha = 4.0 * alpha;
    for (d, &s) in dst.iter_mut().zip(src) {
        let rc = s.max(0.0) as f64;
        *d = (four_alpha * rc).ln_1p() as f32;
    }
    let centre = (dst.iter().map(|&v| v as f64).sum::<f64>() / n_measured as f64) as f32;
    for d in dst.iter_mut() {
        *d -= centre;
    }
}

/// `out[q] = 1` where `panel[q]` is a gene this row carries, else `0`.
///
/// "Measured", not "non-zero": a gene the row does not carry is absent from the
/// CSR, and on a heterogeneous panel that is not the same claim as a zero count
/// (§8's *missing vs zero* boundary). A caller that needs "non-zero" as well can
/// read the gathered values at the same positions.
///
/// `gene_ids` must be sorted ascending and unique, which is what every gathered
/// row is; the lookup is an exact-match binary search. `panel` carries no
/// ordering contract and may repeat ids — each position is answered
/// independently, so a repeated id sets every one of its positions.
pub fn measured_mask(gene_ids: &[i32], panel: &[i32], out: &mut [u8]) {
    for (o, &g) in out.iter_mut().zip(panel) {
        *o = u8::from(gene_ids.binary_search(&g).is_ok());
    }
}

#[cfg(test)]
#[path = "transform_tests.rs"]
mod tests;
