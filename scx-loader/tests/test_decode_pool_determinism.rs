//! ORG-9.10-6: same seed ⇒ same batch **contents**, at any decode-pool size.
//!
//! `TrainingPipeline` documents seeded reproducibility, and the shuffle is
//! seeded per epoch, but the only test of the claim
//! (`pipeline.rs::tests::test_same_seed_reproducible`) compares `cell_indices`
//! and nothing else, and runs both arms at the same pool size. Two things it
//! therefore cannot see:
//!
//! * a divergence in `x` or `obs` while `cell_indices` still matches — the
//!   decode stage scatters rows in parallel (`fill_batch_parallel`'s
//!   `par_chunks_mut().zip(par_iter())`), so a row/index pairing bug lands in
//!   `x`, which was never compared;
//! * any dependence on the *number of decode threads*, which is exactly what
//!   changes between a developer's laptop, a 128-core node and a
//!   `SCX_LOADER_CPU_THREADS`-tuned run.
//!
//! This test has its own binary so it can set the process-global
//! `SCX_LOADER_CPU_THREADS` without racing a sibling test for it.
//!
//! ## Two traps, both hit while falsifying this test
//!
//! **The mutant must depend on the pool.** The obvious falsification — perturb
//! the value written at `projection.rs`'s `output_row[idx] = value` — changes
//! *both* arms by the same amount, so the equality assertion stays green and
//! the falsification reads as passed. A mutant here has to vary with the pool,
//! e.g. `+ rayon::current_thread_index().unwrap_or(0) as f32`. Recorded because
//! the first attempt at this fell into it.
//!
//! **The fixture has to distinguish rows.** `write_multi_shard_fixture` writes
//! one nonzero per row at column `row % n_vars`. With `n_vars < n_obs` the
//! columns repeat, and because this test normalizes, a single-nonzero row
//! always scales to exactly `target_sum` whatever its stored count was — so
//! rows `r` and `r + n_vars` decode to **byte-identical dense rows** and a
//! pairing error that swapped them would not show. Measured, not assumed:
//! against that fixture the `every decoded row in a batch must be distinct`
//! premise below fails outright.
//!
//! It is only the *pairing* half that the old fixture hid. A value corruption
//! was always visible, because the normalization denominator comes from
//! `csr_data` before the scatter (`decode_stage.rs`'s `transform_depth` call),
//! not from the output row — so a perturbed written value does not normalize
//! back. Two different blind spots; `write_row_distinguishable_fixture` closes
//! the one that was open, and `the_fixture_can_tell_two_rows_apart` plus the
//! in-test distinctness premise keep it closed.

mod common;

use common::{row_values, write_row_distinguishable_fixture};
use scx_loader::{Batch, LoaderConfig, ObsColumn, TrainingPipeline};

const N_OBS: usize = 240;
/// `> N_OBS`: row `r` occupies columns `r` and `r + 1`, so no two rows share a
/// dense footprint. See `write_row_distinguishable_fixture`.
const N_VARS: usize = 256;
const N_SHARDS: usize = 6;
const SEED: u64 = 0xC0FFEE;
/// One decode thread, two, and more than two. The 1-vs-many split is the one
/// that matters, but a bug that only appears once work is stolen across more
/// than two workers would hide in a two-point sweep.
const POOL_WIDTHS: [usize; 3] = [1, 2, 4];

/// One batch reduced to a comparable form: cell indices, the dense matrix as
/// raw bit patterns (so `-0.0` and any NaN payload count as a difference), and
/// the obs columns in key order.
type BatchRepr = (Vec<u64>, (usize, usize), Vec<u32>, Vec<(String, String)>);

fn repr(b: &Batch) -> BatchRepr {
    let mut obs: Vec<(String, String)> = b
        .obs
        .iter()
        .map(|(k, v): (&String, &ObsColumn)| (k.clone(), format!("{v:?}")))
        .collect();
    obs.sort();
    (
        b.cell_indices.clone(),
        b.x_shape,
        b.x.iter().map(|v| v.to_bits()).collect(),
        obs,
    )
}

fn config() -> LoaderConfig {
    LoaderConfig {
        batch_size: 16,
        seed: SEED,
        normalize: true,
        log1p: true,
        target_sum: 1e4,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    }
}

fn run_epoch(path: &std::path::Path, decode_threads: usize) -> Vec<BatchRepr> {
    // Read at pool-construction time by `TrainingPipeline::ensure_decode_pool`,
    // which builds a pool private to this pipeline — so setting it here does
    // not leak into any other pipeline built later in this binary.
    std::env::set_var(
        scx_loader::pool::CPU_THREADS_ENV,
        decode_threads.to_string(),
    );
    assert_eq!(
        scx_loader::pool::resolve_pool_threads(Some(&decode_threads.to_string())),
        decode_threads,
        "premise: the decode-pool override must be honoured, or every arm runs \
         at the same width and the test is vacuous"
    );

    let mut pipeline = TrainingPipeline::new(path, config()).expect("TrainingPipeline::new");
    pipeline.start_epoch().expect("start_epoch");

    let mut out = Vec::new();
    while let Some(batch) = pipeline.next_batch().expect("next_batch") {
        out.push(repr(&batch));
    }
    pipeline.shutdown();
    out
}

/// The premise the contents comparison rests on: no two rows of this fixture
/// decode to the same dense row, so a row/value pairing error has somewhere to
/// show.
///
/// Asserted on the **normalized** values the pipeline actually compares, not on
/// the stored counts — normalization is what flattens the previous fixture, and
/// a premise checked before it would pass while the comparison stayed blind.
#[test]
fn the_fixture_can_tell_two_rows_apart() {
    // Columns first: `r` and `r + 1`, distinct for every `r` because
    // `N_VARS > N_OBS`. A `const` block so a future edit to either constant
    // fails to compile rather than failing at run time.
    const { assert!(N_VARS > N_OBS) };

    // Then values. `normalize_total` scales each row to `target_sum`, so what
    // survives is the ratio between the row's two counts.
    let mut seen = std::collections::HashSet::new();
    let mut distinct_ratios = 0usize;
    for row in 0..N_OBS {
        let (v0, v1) = row_values(row);
        assert!(v0 > 0 && v1 > 0, "row {row} must have two stored nonzeros");
        // Compare the reduced fraction, which is exactly what a scale by
        // `target_sum / depth` preserves.
        let g = gcd(v0 as u32, v1 as u32);
        if seen.insert((v0 as u32 / g, v1 as u32 / g)) {
            distinct_ratios += 1;
        }
    }
    assert!(
        distinct_ratios > 1,
        "premise: the fixture's values must survive normalization as more than \
         one ratio, or the dense values carry no signal and only the column \
         positions do"
    );
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// The claim: one seed, one epoch ordering, one set of batch contents —
/// independent of how many threads scatter the rows.
#[test]
fn same_seed_yields_identical_batch_contents_at_any_decode_pool_size() {
    let dir = tempfile::tempdir().unwrap();
    let path =
        write_row_distinguishable_fixture(&dir.path().join("fixture.scx"), N_OBS, N_VARS, N_SHARDS);

    let runs: Vec<(usize, Vec<BatchRepr>)> = POOL_WIDTHS
        .iter()
        .map(|&w| (w, run_epoch(&path, w)))
        .collect();

    let (base_width, base) = &runs[0];

    assert!(
        base.len() > 1,
        "premise: the fixture must produce more than one batch, got {}",
        base.len()
    );

    // Premise: the epoch must actually be shuffled, otherwise `cell_indices`
    // is the identity permutation and agreement proves nothing.
    let identity: Vec<u64> = (0..N_OBS as u64).collect();
    let observed: Vec<u64> = base.iter().flat_map(|b| b.0.clone()).collect();
    assert_eq!(
        observed.len(),
        N_OBS,
        "every cell must appear exactly once in an epoch"
    );
    assert_ne!(
        observed, identity,
        "premise: the epoch must be shuffled for this comparison to be meaningful"
    );

    // Premise: two rows in the same batch really do decode to different dense
    // rows. `the_fixture_can_tell_two_rows_apart` proves the fixture *can*
    // distinguish them; this proves the pipeline's output actually does, after
    // normalization, projection and the log.
    let (n_rows, n_genes) = base[0].1;
    assert!(n_rows > 1);
    let rows: std::collections::HashSet<&[u32]> = (0..n_rows)
        .map(|r| &base[0].2[r * n_genes..(r + 1) * n_genes])
        .collect();
    assert_eq!(
        rows.len(),
        n_rows,
        "premise: every decoded row in a batch must be distinct, or a pairing \
         error between two identical rows would not show in `x`"
    );

    for (width, run) in &runs[1..] {
        assert_eq!(
            base.len(),
            run.len(),
            "batch count must not depend on the decode-pool size \
             ({base_width} vs {width} threads)"
        );
        for (i, (a, b)) in base.iter().zip(run.iter()).enumerate() {
            assert_eq!(
                a.0, b.0,
                "batch {i}: cell_indices diverged at {width} threads"
            );
            assert_eq!(a.1, b.1, "batch {i}: x_shape diverged at {width} threads");
            assert_eq!(
                a.2, b.2,
                "batch {i}: dense X diverged between a {base_width}-thread and a \
                 {width}-thread decode pool (cell_indices agreed, so this is a \
                 row/value pairing difference, not an ordering one)"
            );
            assert_eq!(
                a.3, b.3,
                "batch {i}: obs columns diverged at {width} threads"
            );
        }
    }
}
