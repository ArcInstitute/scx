//! LISI pinned against harmonypy 0.2.0 (§7.19, ORG-7.21-4).
//!
//! Values and provenance live in [`super::lisi_reference_values`]. The claim
//! under test is the **neighbourhood**, not the kernel: harmonypy retrieves
//! `3 * perplexity` neighbours and drops column 0 — every point's own
//! self-match — while SCX's sweep skips `j == i` as it collects, so SCX must
//! ask for one fewer to see the same cells.
//!
//! That derivation was spelled out in three places and all three said
//! `3 * perplexity`. The only harmonypy comparison in the tree ran at
//! `atol = 1e-2`, which is wider than the shift one neighbour causes.

use super::lisi::{compute_lisi, default_n_neighbors, LisiConfig};
use super::lisi_reference_values as r;

fn fixture() -> (Vec<f32>, Vec<u32>) {
    let emb: Vec<f32> = r::LISI_X
        .iter()
        .flat_map(|row| row.iter().map(|&v| v as f32))
        .collect();
    (emb, r::LISI_LABELS.to_vec())
}

fn run(k: usize) -> Vec<f64> {
    let (emb, labels) = fixture();
    let cfg = LisiConfig {
        perplexity: r::LISI_PERPLEXITY,
        n_neighbors: k,
        ..Default::default()
    };
    compute_lisi(&emb, r::LISI_N_CELLS, r::LISI_N_DIMS, &labels, &cfg)
        .expect("lisi")
        .lisi
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// The shared derivation must produce harmonypy's effective neighbourhood.
///
/// This is the constant the bug was in, and it is asserted against the width
/// harmonypy actually retrieves rather than against a literal 89 — a literal
/// would still be right for the wrong reason if the retrieval convention moved.
#[test]
fn the_shared_neighbour_count_drops_the_self_match() {
    assert_eq!(
        default_n_neighbors(r::LISI_PERPLEXITY),
        r::LISI_HARMONYPY_RETRIEVED - 1,
        "default_n_neighbors must be harmonypy's retrieval width minus its \
         self-match; SCX skips j == i while collecting, harmonypy drops it after"
    );
    // The default must AGREE with the shared derivation. Note this asserts the
    // value, not the call: hardcoding 89 here passes, and does the same thing.
    // What it cannot see is a *second* copy of `3 * perplexity` reappearing in
    // a binding — pyscx and rscx both had one, and neither is visible from this
    // crate's tests. The ORG-7.21-4 CI guard covers that half.
    assert_eq!(
        LisiConfig::default().n_neighbors,
        default_n_neighbors(30.0),
        "the default config disagrees with the shared derivation"
    );
    assert_eq!(LisiConfig::default().n_neighbors, 89);
}

/// The fixture must be able to see one neighbour. The generator refuses one
/// that cannot; assert it on the emitted literals too, since the generator's
/// check and this file can drift if someone edits either by hand.
#[test]
fn the_fixture_can_separate_the_two_neighbourhood_conventions() {
    let shift = max_abs(&r::LISI_EXPECTED, &r::LISI_OFF_BY_ONE);
    assert!(
        shift > r::LISI_ATOL * 100.0,
        "one extra neighbour moves harmonypy's own LISI by only {shift:.3e}, \
         which is not comfortably above the bar {:.3e} — this fixture would \
         pass against either convention",
        r::LISI_ATOL
    );
    assert!(
        (shift - r::LISI_ONE_NEIGHBOUR_SHIFT).abs() < 1e-9,
        "the pinned shift {} no longer matches the tables ({shift})",
        r::LISI_ONE_NEIGHBOUR_SHIFT
    );
}

/// The parity arm.
#[test]
fn lisi_matches_harmonypy_per_cell() {
    let got = run(default_n_neighbors(r::LISI_PERPLEXITY));
    let d = max_abs(&got, &r::LISI_EXPECTED);
    assert!(
        d <= r::LISI_ATOL,
        "per-cell LISI: max |delta| {d:.3e} > {:.3e}",
        r::LISI_ATOL
    );
}

/// **This is §7.19.** The repulsion arm: the pre-fix neighbourhood — one cell
/// wider — must be measurably *wrong*, not merely different. Without it, a
/// bar wide enough to swallow the shift would let both conventions pass and
/// the parity arm above would stop meaning anything.
#[test]
fn the_pre_fix_neighbourhood_reproduces_the_off_by_one_answer() {
    let got = run(r::LISI_HARMONYPY_RETRIEVED);
    let to_wrong = max_abs(&got, &r::LISI_OFF_BY_ONE);
    assert!(
        to_wrong <= r::LISI_ATOL,
        "asking for {} neighbours should reproduce harmonypy's wider-retrieval \
         answer, but differs by {to_wrong:.3e} — the two implementations are \
         not disagreeing about the neighbourhood alone",
        r::LISI_HARMONYPY_RETRIEVED
    );
    let to_right = max_abs(&got, &r::LISI_EXPECTED);
    assert!(
        to_right > r::LISI_ATOL * 100.0,
        "the pre-fix neighbourhood lands {to_right:.3e} from harmonypy, inside \
         100x the bar; this test could not have failed"
    );
}
