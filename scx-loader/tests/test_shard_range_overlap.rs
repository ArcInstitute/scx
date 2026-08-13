//! `TrainingPipeline` must refuse a file whose CSR shard row ranges overlap.
//!
//! The loader's per-group row index is keyed on the global obs row, so two
//! shards claiming the same row cannot both be represented: one silently
//! displaces the other and its cells never reach a batch. Worse, the loss is
//! non-deterministic — it only materialises when the two collide inside the
//! same shard *group*, and group membership is re-drawn from the shard shuffle
//! every epoch, so the epoch is short by a number of cells that varies run to
//! run with no warning.
//!
//! Two ways to get there, both covered here: a malformed file (a merge /
//! append / compact defect), and the reachable one — a multimodal file opened
//! with no `modality_id`, where each modality legitimately tiles `[0, n_obs)`
//! and the flattened catalog therefore overlaps by construction.

mod common;

use scx_loader::{LoaderConfig, TrainingPipeline};

fn config() -> LoaderConfig {
    LoaderConfig {
        batch_size: 4,
        shard_group_size: 2,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    }
}

#[test]
fn rejects_a_file_whose_shard_row_ranges_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let path = common::write_overlapping_shards_fixture(&dir.path().join("bad.scx"), 8, 16, 3);

    let err = TrainingPipeline::new(&path, config())
        .err()
        .expect("expected construction to fail on overlapping shard ranges");
    let msg = err.to_string();
    assert!(
        msg.contains("overlap") && msg.contains("malformed"),
        "a single-modality overlap is corruption, and the message should say so: {msg}"
    );
}

#[test]
fn rejects_a_multimodal_file_opened_without_a_modality() {
    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multimodal_fixture(&dir.path().join("mm.scx"), 8, 16, 4);

    let err = TrainingPipeline::new(&path, config())
        .err()
        .expect("expected construction to fail without a modality_id");
    let msg = err.to_string();
    assert!(
        msg.contains("modality_id") && msg.contains("MultimodalTrainingDataset"),
        "message should name both ways out: {msg}"
    );
}

/// The same multimodal file opens cleanly once a modality is selected — the
/// half that keeps the guard from being "reject everything multimodal".
#[test]
fn accepts_the_same_multimodal_file_with_a_modality_selected() {
    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multimodal_fixture(&dir.path().join("mm.scx"), 8, 16, 4);

    for (mid, expected_vars) in [(1u8, 4u64), (2, 16)] {
        let pipeline = TrainingPipeline::new(
            &path,
            LoaderConfig {
                modality_id: Some(mid),
                ..config()
            },
        )
        .unwrap_or_else(|e| panic!("modality {mid} should open: {e}"));
        assert_eq!(
            pipeline.n_output_genes(),
            expected_vars as usize,
            "modality {mid} should report its own width"
        );
    }
}

/// A well-formed single-modality file is unaffected.
#[test]
fn accepts_a_clean_single_modality_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multi_shard_fixture(&dir.path().join("ok.scx"), 16, 16, 4);
    TrainingPipeline::new(&path, config()).expect("a clean tiling must still open");
}
