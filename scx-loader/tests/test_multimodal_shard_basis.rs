//! The loader's per-modality **shard-index basis**, end to end.
//!
//! `shard_groups` carries positions in the *selected modality's* CSR shard
//! list. Three places have to agree on that list:
//! `TrainingPipeline::start_epoch` (which builds the shuffle keys),
//! `io_stage`'s deletion bucketing, and `io_stage`'s per-group shard reads.
//! Getting it wrong reads the **wrong shard** — silently, with a plausible row
//! count and plausible values.
//!
//! Nothing pinned that before PR-06. Every other `io_stage` test passes
//! `modality_id = None`; the one content-correctness test uses a single-shard
//! file, where *any* index basis resolves to the same entry; and no test
//! anywhere ran the training pipeline on a multimodal file with more than one
//! CSR shard per modality. The shared fixture
//! (`common::write_multimodal_layout_fixture`) is built so each wrong basis is
//! observable — see its doc comment.

mod common;

use std::sync::Arc;

use scx_format_io::reader::ScxReader;
use scx_loader::{io_stage, LoaderConfig, TrainingPipeline};

const N_OBS: usize = 10;
const RNA_VARS: usize = 7;
const RNA_BASE: u8 = 200;

/// Premises the three tests below rest on. Asserted once, on their own, so a
/// fixture change that quietly makes them vacuous fails *here* rather than
/// leaving the real assertions passing for the wrong reason.
#[test]
fn the_fixture_can_tell_the_index_bases_apart() {
    let dir = tempfile::tempdir().unwrap();
    let (path, adt_id, rna_id) =
        common::write_multimodal_layout_fixture(&dir.path().join("mm.scx"), N_OBS);
    let reader = ScxReader::open(&path).unwrap();
    let catalog = reader.catalog();

    let rna = catalog.csr_shard_indices(Some(rna_id));
    assert_eq!(rna.len(), 2, "rna must be two shards");
    assert!(
        rna[0] > rna[1],
        "rna's catalog order must be the reverse of its row order, else dropping \
         the sort is unobservable: {rna:?}"
    );
    assert_eq!(
        catalog.csr_shard_indices(Some(adt_id)).len(),
        1,
        "adt must be a single shard, so the two modalities' lists differ in length"
    );
    assert_eq!(
        catalog.csr_shard_indices(None).len(),
        3,
        "the unfiltered list must differ in length from either modality's"
    );

    let adt_first = catalog.csr_shards_for_modality(adt_id)[0];
    let rna_first = catalog.csr_shards_for_modality(rna_id)[0];
    assert_ne!(
        reader.read_shard_from_entry(adt_first).unwrap().2,
        reader.read_shard_from_entry(rna_first).unwrap().2,
        "the two modalities' first shards must be distinguishable by payload"
    );
}

/// `io_stage` resolves `shard_groups` positions against the selected
/// modality's shard list, in row order.
#[test]
fn io_stage_reads_the_selected_modalitys_shards() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let (path, _adt_id, rna_id) =
            common::write_multimodal_layout_fixture(&dir.path().join("mm.scx"), N_OBS);
        let reader = Arc::new(ScxReader::open(&path).unwrap());

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let r2 = Arc::clone(&reader);
        let handle = tokio::spawn(io_stage(r2, vec![vec![0, 1]], None, Some(rna_id), tx));
        let group = rx.recv().await.unwrap();
        assert!(rx.recv().await.is_none());
        handle.await.unwrap().unwrap();

        assert_eq!(group.shards.len(), 2);
        let entries = reader.catalog().csr_shards_for_modality(rna_id);
        for (pos, shard) in group.shards.iter().enumerate() {
            let (indptr, indices, data) = reader.read_shard_from_entry(entries[pos]).unwrap();
            assert_eq!(shard.indptr, indptr, "position {pos} indptr");
            assert_eq!(shard.indices, indices, "position {pos} indices");
            assert_eq!(shard.data, data, "position {pos} data");
            assert_eq!(shard.n_rows, (N_OBS / 2) as u32, "position {pos} n_rows");
        }
        // Spelled out, so a wrong basis cannot pass by agreeing with itself.
        assert_eq!(
            group
                .shards
                .iter()
                .map(|s| s.global_row_offset)
                .collect::<Vec<_>>(),
            vec![0, 5],
            "positions are rna's shards in ROW order, not catalog order"
        );
        assert_eq!(
            group.shards[0].data,
            vec![200.0, 201.0, 202.0, 203.0, 204.0]
        );
        assert_eq!(
            group.shards[1].data,
            vec![205.0, 206.0, 207.0, 208.0, 209.0]
        );
    });
}

/// Deletions on the per-modality path. `reconstruct_deletion_map`'s
/// cross-modality behaviour was only ever asserted against synthetic range
/// arrays; this drives it through a real file, where the ranges that bucket the
/// deleted cells come from the *selected modality's* shard list.
#[test]
fn io_stage_applies_deletions_on_the_per_modality_path() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let dir = tempfile::tempdir().unwrap();
        let (path, _adt_id, rna_id) =
            common::write_multimodal_layout_fixture(&dir.path().join("mm_del.scx"), N_OBS);
        let reader = Arc::new(ScxReader::open(&path).unwrap());

        // One deleted cell in each of rna's two shards. Global row 6 is local 1
        // of rna's second shard — but local 6 of adt's single `[0, 10)` shard,
        // so a run that bucketed against the unfiltered list would attach
        // `{1, 6}` to position 0.
        let mut dv = scx_format_io::deletion_vectors::DeletionVectors::new();
        dv.insert_global([1u32, 6]);

        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let handle = tokio::spawn(io_stage(
            reader,
            vec![vec![0, 1]],
            Some(dv),
            Some(rna_id),
            tx,
        ));
        let group = rx.recv().await.unwrap();
        handle.await.unwrap().unwrap();

        assert_eq!(group.shards.len(), 2, "neither shard is fully deleted");
        let bitmaps: Vec<Vec<u32>> = group
            .shards
            .iter()
            .map(|s| {
                s.deleted_rows
                    .as_ref()
                    .map(|bm| bm.iter().collect())
                    .unwrap_or_default()
            })
            .collect();
        assert_eq!(
            bitmaps,
            vec![vec![1u32], vec![1u32]],
            "global rows 1 and 6 are local row 1 of rna's two 5-row shards"
        );
    });
}

/// The whole pipeline, on a multimodal file with two CSR shards in the selected
/// modality — the case nothing covered. `start_epoch` derives the shuffle keys
/// from the same `csr_shard_indices` call `io_stage` resolves positions with,
/// so a basis mismatch here surfaces as either wrong rows or an out-of-bounds
/// shard index, depending on which side drifts.
#[test]
fn training_pipeline_delivers_the_selected_modalitys_rows() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _adt_id, rna_id) =
        common::write_multimodal_layout_fixture(&dir.path().join("mm_pipe.scx"), N_OBS);

    let mut pipeline = TrainingPipeline::new(
        &path,
        LoaderConfig {
            batch_size: 4,
            shard_group_size: 2,
            obs_columns: vec!["cell_id".to_string()],
            max_memory_mb: 1024,
            // Raw counts: the point is *which* rows arrive, and
            // `LoaderConfig::default()` normalizes + log1p's them.
            normalize: false,
            log1p: false,
            modality_id: Some(rna_id),
            ..Default::default()
        },
    )
    .expect("a modality-scoped multimodal file must open");
    assert_eq!(pipeline.n_vars(), RNA_VARS as u64, "rna's own width");

    pipeline.start_epoch().unwrap();
    let mut seen: Vec<(u64, Vec<f32>)> = Vec::new();
    while let Some(batch) = pipeline.next_batch().unwrap() {
        let (n_rows, n_genes) = batch.x_shape;
        assert_eq!(n_genes, RNA_VARS, "rna's width, not adt's");
        for r in 0..n_rows {
            seen.push((
                batch.cell_indices[r],
                batch.x[r * n_genes..(r + 1) * n_genes].to_vec(),
            ));
        }
    }
    seen.sort_by_key(|(cell, _)| *cell);

    assert_eq!(
        seen.iter().map(|(c, _)| *c).collect::<Vec<_>>(),
        (0..N_OBS as u64).collect::<Vec<_>>(),
        "every cell exactly once, across both of rna's shards"
    );
    for (cell, row) in &seen {
        assert_eq!(
            *row,
            common::expected_modality_row(RNA_BASE, RNA_VARS, *cell as usize),
            "cell {cell} must carry rna's values, not adt's"
        );
    }
}
