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

mod common;

use std::collections::HashMap;

use common::write_multi_shard_fixture;
use scx_loader::{Batch, LoaderConfig, ObsColumn, TrainingPipeline};

const N_OBS: usize = 240;
const N_VARS: usize = 32;
const N_SHARDS: usize = 6;
const SEED: u64 = 0xC0FFEE;

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
        "premise: the decode-pool override must be honoured, or both arms run \
         at the same width and the test is vacuous"
    );

    let config = LoaderConfig {
        batch_size: 16,
        seed: SEED,
        normalize: true,
        log1p: true,
        target_sum: 1e4,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    let mut pipeline = TrainingPipeline::new(path, config).expect("TrainingPipeline::new");
    pipeline.start_epoch().expect("start_epoch");

    let mut out = Vec::new();
    while let Some(batch) = pipeline.next_batch().expect("next_batch") {
        out.push(repr(&batch));
    }
    pipeline.shutdown();
    out
}

/// The claim: one seed, one epoch ordering, one set of batch contents —
/// independent of how many threads scatter the rows.
#[test]
fn same_seed_yields_identical_batch_contents_at_any_decode_pool_size() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_multi_shard_fixture(&dir.path().join("fixture.scx"), N_OBS, N_VARS, N_SHARDS);

    let single = run_epoch(&path, 1);
    let many = run_epoch(&path, 4);

    assert!(
        single.len() > 1,
        "premise: the fixture must produce more than one batch, got {}",
        single.len()
    );
    assert_eq!(
        single.len(),
        many.len(),
        "batch count must not depend on the decode-pool size"
    );

    // Premise: the epoch must actually be shuffled, otherwise `cell_indices`
    // is the identity permutation and agreement proves nothing.
    let identity: Vec<u64> = (0..N_OBS as u64).collect();
    let observed: Vec<u64> = single.iter().flat_map(|b| b.0.clone()).collect();
    assert_eq!(
        observed.len(),
        N_OBS,
        "every cell must appear exactly once in an epoch"
    );
    assert_ne!(
        observed, identity,
        "premise: the epoch must be shuffled for this comparison to be meaningful"
    );

    for (i, (a, b)) in single.iter().zip(many.iter()).enumerate() {
        assert_eq!(a.0, b.0, "batch {i}: cell_indices diverged");
        assert_eq!(a.1, b.1, "batch {i}: x_shape diverged");
        assert_eq!(
            a.2, b.2,
            "batch {i}: dense X diverged between a 1-thread and a 4-thread \
             decode pool (cell_indices agreed, so this is a row/value pairing \
             difference, not an ordering one)"
        );
        assert_eq!(a.3, b.3, "batch {i}: obs columns diverged");
    }

    // And the whole epoch, not just batch-by-batch: a scatter that swapped two
    // rows *across* a batch boundary would pass every per-batch check above.
    let flat = |e: &[BatchRepr]| -> HashMap<u64, Vec<u32>> {
        let mut m = HashMap::new();
        for (cells, (_, n_genes), x, _) in e {
            for (r, &cell) in cells.iter().enumerate() {
                m.insert(cell, x[r * n_genes..(r + 1) * n_genes].to_vec());
            }
        }
        m
    };
    assert_eq!(
        flat(&single),
        flat(&many),
        "per-cell rows must be identical across the whole epoch"
    );
}
