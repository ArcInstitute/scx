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

/// Selecting a modality is not a free pass: that modality's **own** shards
/// still have to tile the obs axis exactly once.
///
/// `ShardGroupIndex::build` cannot stand in for this. It only sees one shard
/// group, so two overlapping shards have to be drawn into the same group before
/// it notices — which the shuffle re-decides every epoch, and never at
/// `shard_group_size == 1`. Found by codex - gpt-5.6-sol.
#[test]
fn rejects_a_scoped_modality_whose_own_shards_overlap() {
    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multimodal_overlapping_fixture(&dir.path().join("mm_bad.scx"), 8, 6);

    // Modality 1 ("adt") is written as one clean shard; modality 2 ("rna") as
    // two that overlap. Scoping to the clean one still works, so this is not
    // "reject anything multimodal".
    TrainingPipeline::new(
        &path,
        LoaderConfig {
            modality_id: Some(1),
            shard_group_size: 1,
            ..config()
        },
    )
    .expect("the well-formed modality must still open");

    let err = TrainingPipeline::new(
        &path,
        LoaderConfig {
            modality_id: Some(2),
            // The group size at which the mid-epoch check provably cannot fire.
            shard_group_size: 1,
            ..config()
        },
    )
    .err()
    .expect("expected the malformed modality to be rejected at construction");
    let msg = err.to_string();
    assert!(
        msg.contains("modality 2") && msg.contains("tile the obs axis"),
        "message should name the modality and the invariant: {msg}"
    );
}

/// `IndexPlanLoader` and `SparseCellSetLoader` have no modality surface, so on
/// a multimodal file they read the flattened shard list and `BackedCsrReader`'s
/// row index keeps an arbitrary modality's shard per row — answering from a
/// modality the caller never chose, silently. Verified before the fix: an
/// equal-width two-modality file constructed fine and `process_plan` returned
/// rows. Found by codex - gpt-5.6-sol.
#[test]
fn the_plan_driven_loaders_refuse_a_multimodal_file() {
    use scx_loader::IndexPlanLoader;

    let dir = tempfile::tempdir().unwrap();
    // Equal widths, so nothing would trip on a shape mismatch either.
    let path = common::write_multimodal_fixture(&dir.path().join("mm.scx"), 8, 6, 6);

    let mut cfg = LoaderConfig {
        normalize: false,
        log1p: false,
        max_memory_mb: 512,
        ..Default::default()
    };
    cfg.obs_columns.clear();
    let err = IndexPlanLoader::new(&path, cfg, 4, true, 4, 16)
        .err()
        .expect("IndexPlanLoader must refuse a multimodal file");
    let msg = err.to_string();
    assert!(
        msg.contains("IndexPlanLoader") && msg.contains("overlap"),
        "message should name the loader and the cause: {msg}"
    );
}

/// `SparseCellSetLoader` gets the same guard, and it needs its own assertion:
/// the test above constructs only `IndexPlanLoader`, so disabling this guard
/// would have left an "all three loaders" claim green. Found by
/// codex - gpt-5.6-sol.
#[test]
fn the_cell_set_loader_refuses_a_multimodal_file() {
    use scx_format_io::ScxReader;
    use scx_loader::SparseCellSetLoader;

    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multimodal_fixture(&dir.path().join("mm.scx"), 8, 6, 6);

    let readers = vec![ScxReader::open(&path).unwrap()];
    let err = SparseCellSetLoader::new(
        readers, 4, None, 4, None, None, /*normalize=*/ false, /*log1p=*/ false, 1e4,
        None, /*scatter_block_index*/ false,
    )
    .err()
    .expect("SparseCellSetLoader must refuse a multimodal file");
    let msg = err.to_string();
    assert!(
        msg.contains("SparseCellSetLoader (file 0)") && msg.contains("overlap"),
        "message should name the loader, the file index, and the cause: {msg}"
    );
}

/// The control: a single-modality file still opens through the cell-set loader.
#[test]
fn the_cell_set_loader_still_opens_a_single_modality_file() {
    use scx_format_io::ScxReader;
    use scx_loader::SparseCellSetLoader;

    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multi_shard_fixture(&dir.path().join("ok.scx"), 16, 16, 4);
    let readers = vec![ScxReader::open(&path).unwrap()];
    SparseCellSetLoader::new(
        readers, 4, None, 4, None, None, false, false, 1e4, None,
        /*scatter_block_index*/ false,
    )
    .expect("a clean file must still open");
}

/// Overlap is only half of "exactly once". A cover with a **gap** passes
/// `has_overlapping_csr_ranges` and then loses the uncovered cells — a short
/// epoch on `TrainingPipeline`, and an in-bounds row reported as "out of range"
/// mid-iteration on the plan-driven loaders. Found by codex - gpt-5.6-sol.
#[test]
fn rejects_a_file_whose_shards_leave_a_gap() {
    let dir = tempfile::tempdir().unwrap();
    let path = common::write_gapped_shards_fixture(&dir.path().join("gap.scx"), 16, 8, 6, 3);

    let err = TrainingPipeline::new(&path, config())
        .err()
        .expect("expected construction to fail on a gapped cover");
    let msg = err.to_string();
    assert!(
        msg.contains("exactly once") && msg.contains("belong to no shard"),
        "message should name the invariant and the consequence: {msg}"
    );
    // The rows that actually went missing, not just "something is wrong".
    assert!(
        msg.contains("rows [6, 9)"),
        "message should name the uncovered rows: {msg}"
    );
}

/// A file whose **only** modality is registered in the modality table — so its
/// X is stamped `modality_id = 1` and modality 0 owns nothing — is valid and
/// unambiguous, and must open on every unscoped entry point.
///
/// This is what `from_mudata(MuData({"rna": adata}))` and a single-modality
/// h5mu ingest write. An earlier version of the cover check asked "does
/// modality 0 tile the obs axis?" and rejected all of them with a false "leave
/// a gap" error — valid input broken by a guard meant to catch corruption.
/// Every other single-modality fixture here writes id-0 shards, which is why
/// nothing caught it. Found by codex - gpt-5.6-sol.
#[test]
fn accepts_a_file_whose_only_modality_is_registered_as_id_1() {
    use scx_format_io::ScxReader;
    use scx_loader::{IndexPlanLoader, SparseCellSetLoader};

    let dir = tempfile::tempdir().unwrap();
    let path =
        common::write_single_registered_modality_fixture(&dir.path().join("mono.scx"), 16, 8, 4);

    TrainingPipeline::new(&path, config())
        .expect("unscoped TrainingPipeline must accept a sole registered modality");

    let mut cfg = LoaderConfig {
        normalize: false,
        log1p: false,
        max_memory_mb: 512,
        ..Default::default()
    };
    cfg.obs_columns.clear();
    IndexPlanLoader::new(&path, cfg, 4, true, 4, 16)
        .expect("IndexPlanLoader must accept a sole registered modality");

    let readers = vec![ScxReader::open(&path).unwrap()];
    SparseCellSetLoader::new(
        readers, 4, None, 4, None, None, false, false, 1e4, None,
        /*scatter_block_index*/ false,
    )
    .expect("SparseCellSetLoader must accept a sole registered modality");
}

/// The control for the test above: a single-modality file still opens.
#[test]
fn the_plan_driven_loader_still_opens_a_single_modality_file() {
    use scx_loader::IndexPlanLoader;

    let dir = tempfile::tempdir().unwrap();
    let path = common::write_multi_shard_fixture(&dir.path().join("ok.scx"), 16, 16, 4);
    let mut cfg = LoaderConfig {
        normalize: false,
        log1p: false,
        max_memory_mb: 512,
        ..Default::default()
    };
    cfg.obs_columns.clear();
    IndexPlanLoader::new(&path, cfg, 4, true, 4, 16).expect("a clean file must still open");
}
