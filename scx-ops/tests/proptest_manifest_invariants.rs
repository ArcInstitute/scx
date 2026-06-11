//! Property-based test for manifest append+delete+rollback invariants
//!
//! Strategy: start from a small seed SCX file (one CSR shard, K rows,
//! manifest_sequence = 1). Run a random sequence of `append` and
//! `mark_deleted` operations, capturing a snapshot of the readable
//! state after each step. Then `rollback_to(target_seq)` to a random
//! intermediate sequence and assert the file reads back exactly the
//! snapshot recorded for that sequence.
//!
//! Invariants:
//! - After every op (in-bounds append or in-bounds delete), the file is
//!   readable and `manifest_sequence` strictly advances. `mark_deleted`
//!   always writes a new manifest entry, even when re-deleting an
//!   already-deleted row.
//! - After every op, the `prev_catalog_offset` chain is well-formed:
//!   walking it from the header back must reach `manifest_sequence = 1`.
//! - `rollback_to(seq)` returns the file to exactly the state recorded
//!   when that sequence was current — same `n_obs`, same CSR contents.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use proptest::prelude::*;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION, MAGIC};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_ops::AppendOptions;

// =========================================================================
// Fixture construction
// =========================================================================

fn sample_obs(start: usize, n: usize) -> RecordBatch {
    let ids: Vec<String> = (start..start + n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_shard(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        let c0 = (row * 2) % n_vars;
        let c1 = (row * 2 + 1) % n_vars;
        indices.push(c0 as u32);
        indices.push(c1 as u32);
        values.push((row % 250 + 1) as u8);
        values.push(((row + 1) % 250 + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn make_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: 16384,
        codec_id: 0,
        index_dtype: 0,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    }
}

fn write_seed(path: &std::path::Path, n_obs: usize, n_vars: usize) {
    let header = make_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&sample_obs(0, n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    let (indptr, indices, values) = sample_shard(n_obs, n_vars);
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "proptest".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

// =========================================================================
// State capture
// =========================================================================

#[derive(Debug, Clone)]
struct StateSnapshot {
    sequence: u64,
    n_obs: u64,
    csr_indptr: Vec<i64>,
    csr_indices: Vec<i32>,
    csr_data: Vec<f32>,
}

fn snapshot(path: &PathBuf) -> StateSnapshot {
    let reader = ScxReader::open(path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    StateSnapshot {
        sequence: reader.header().manifest_sequence,
        n_obs: reader.n_obs(),
        csr_indptr: csr.indptr.clone(),
        csr_indices: csr.indices.clone(),
        csr_data: csr.data.clone(),
    }
}

// =========================================================================
// Operation enum
// =========================================================================

#[derive(Debug, Clone)]
enum Op {
    Append { n_rows: usize },
    Delete { row_idx: u64 },
}

// Note: proptest strategies are evaluated once per case, before the op
// sequence runs, so we can't bound `row_idx` by the post-op `n_obs`
// (which changes between ops). The runtime modulo at the execution
// site (`row_idx % current_n_obs`) handles bounding instead.
fn arb_op() -> impl Strategy<Value = Op> {
    prop_oneof![
        2 => (1usize..=4).prop_map(|n_rows| Op::Append { n_rows }),
        1 => (0u64..1024).prop_map(|row_idx| Op::Delete { row_idx }),
    ]
}

fn arb_op_sequence() -> impl Strategy<Value = Vec<Op>> {
    prop::collection::vec(arb_op(), 1..=6)
}

// =========================================================================
// Property tests
// =========================================================================

proptest! {
    // Each iteration creates a tempfile and runs append/delete on disk —
    // keep cases low to bound test runtime.
    #![proptest_config(ProptestConfig::with_cases(15))]

    /// Manifest chain stays well-formed after random append+delete
    /// sequences, and `rollback_to` restores prior states.
    #[test]
    fn random_append_delete_rollback(ops in arb_op_sequence()) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seed.scx");
        let seed_n_obs = 8usize;
        let n_vars = 10usize;
        write_seed(&path, seed_n_obs, n_vars);

        let mut snapshots: Vec<StateSnapshot> = vec![snapshot(&path.to_path_buf())];
        prop_assert_eq!(snapshots[0].sequence, 1);
        prop_assert_eq!(snapshots[0].n_obs, seed_n_obs as u64);

        let mut current_n_obs = seed_n_obs;
        for op in &ops {
            let before = snapshot(&path.to_path_buf());
            match op {
                Op::Append { n_rows } => {
                    let new_obs = sample_obs(current_n_obs, *n_rows);
                    let (indptr, indices, values) = sample_shard(*n_rows, n_vars);
                    scx_ops::append(
                        &path,
                        &new_obs,
                        &indptr,
                        &indices,
                        &values,
                        ValueEncoding::Uint8,
                        &AppendOptions::default(),
                    ).unwrap();
                    current_n_obs += n_rows;
                }
                Op::Delete { row_idx } => {
                    // Modulo keeps idx in bounds, so mark_deleted must
                    // succeed. Re-deleting an already-deleted row still
                    // writes a new manifest entry (no no-op path).
                    let idx = row_idx % (current_n_obs as u64);
                    scx_ops::mark_deleted(&path, &[idx])
                        .expect("in-bounds mark_deleted must not fail");
                }
            }

            let after = snapshot(&path.to_path_buf());
            // Every op must produce a strictly increasing sequence.
            prop_assert!(
                after.sequence > before.sequence,
                "manifest_sequence did not advance: before={} after={}",
                before.sequence, after.sequence,
            );
            snapshots.push(after);
        }

        // For each historical snapshot, rollback_to and assert.
        // Walk backwards so each rollback target is reachable via
        // prev_catalog_offset chain.
        for target in snapshots.iter().rev() {
            scx_ops::rollback_to(&path, target.sequence).unwrap();
            let now = snapshot(&path.to_path_buf());
            prop_assert_eq!(now.sequence, target.sequence,
                "rollback_to did not reach target sequence");
            prop_assert_eq!(now.n_obs, target.n_obs,
                "rollback_to gave wrong n_obs");
            prop_assert_eq!(&now.csr_indptr, &target.csr_indptr,
                "rollback_to gave wrong indptr");
            prop_assert_eq!(&now.csr_indices, &target.csr_indices,
                "rollback_to gave wrong indices");
            prop_assert_eq!(&now.csr_data, &target.csr_data,
                "rollback_to gave wrong data");
        }
    }
}
