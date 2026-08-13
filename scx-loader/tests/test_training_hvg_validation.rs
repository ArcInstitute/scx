//! `TrainingPipeline` rejects HVG indices outside the file's gene axis.
//!
//! The sibling assertion for `IndexPlanLoader` lives in
//! `test_index_plan.rs::ctor_rejects_hvg_out_of_range`. For a long time only
//! that one existed: `IndexPlanLoader` range-checked its panel and
//! `TrainingPipeline` did not, so the same `hvg_indices` argument raised on one
//! loader and silently produced an always-zero output column on the other — a
//! dead input feature that trains to a zero weight and is never diagnosed.
//!
//! Both checks now come from the single validating `HvgProjection::new`, so
//! these tests and `ctor_rejects_hvg_out_of_range` are two witnesses of one
//! constructor rather than two independent implementations.

mod common;

use common::write_multi_shard_fixture;
use scx_loader::{LoaderConfig, LoaderError, TrainingPipeline};

const N_OBS: usize = 100;
const N_VARS: usize = 50;
const N_SHARDS: usize = 5;

fn fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    write_multi_shard_fixture(&dir.path().join("fixture.scx"), N_OBS, N_VARS, N_SHARDS)
}

fn config_with_hvg(hvg: Vec<u32>) -> LoaderConfig {
    LoaderConfig {
        batch_size: 32,
        hvg_indices: Some(hvg),
        max_memory_mb: 1024,
        ..LoaderConfig::default()
    }
}

/// The headline case: an index far past `n_vars`. Before the fix this
/// constructed happily and every batch carried a third column of zeros.
#[test]
fn ctor_rejects_hvg_out_of_range() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    match TrainingPipeline::new(&path, config_with_hvg(vec![0, 1, 99_999])) {
        Err(LoaderError::ConfigError { reason }) => {
            assert!(reason.contains("HVG index"), "unexpected reason: {reason}");
            assert!(
                reason.contains("out of range"),
                "unexpected reason: {reason}"
            );
        }
        Err(other) => panic!("expected ConfigError, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

/// The boundary, which is where this actually bites in practice: a panel built
/// against a file with one more gene than this one. `n_vars` itself is already
/// out of range.
#[test]
fn ctor_rejects_an_index_equal_to_n_vars() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    match TrainingPipeline::new(&path, config_with_hvg(vec![N_VARS as u32])) {
        Err(LoaderError::ConfigError { reason }) => {
            assert!(
                reason.contains("out of range"),
                "unexpected reason: {reason}"
            );
        }
        Err(other) => panic!("expected ConfigError, got {other}"),
        Ok(_) => panic!("expected error, got Ok"),
    }
}

/// The other half of the boundary: rejecting one gene too many would be just as
/// wrong, and would not be caught by either test above.
#[test]
fn ctor_accepts_the_last_valid_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture(&dir);

    let pipeline =
        TrainingPipeline::new(&path, config_with_hvg(vec![0, N_VARS as u32 - 1])).unwrap();
    assert_eq!(pipeline.n_output_genes(), 2);
}
