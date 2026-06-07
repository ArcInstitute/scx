//! Regression tests for the atlas-scale `filter_obs` / `count` OOM fix.
//!
//! On row-sharded files the query engine must NEVER materialise the full obs
//! table (`read_obs()`), which at atlas scale concatenated thousands of obs
//! shards into hundreds of GB. Instead it streams one obs shard at a time
//! (`read_obs_shard`) into a bounded mask, and — when per-shard catalog stats
//! are present — only decodes the obs shards that overlap surviving CSR
//! shards.
//!
//! These tests build a file whose obs-metadata shards have DIFFERENT
//! boundaries than its CSR shards (4 obs shards of 2 rows vs 2 CSR shards of 4
//! rows) to exercise the cross-sharding row mapping, and assert via the reader
//! debug counters that:
//!   * `read_obs` is never called on the sharded path, and
//!   * only the obs shards overlapping surviving CSR shards are decoded.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use arrow::array::{Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{
    build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions, QueryPipeline,
};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::reader::ScxReader;
use scx_format::writer::ScxWriter;
use tempfile::TempDir;

const N_VARS: usize = 4;
const N_OBS: u64 = 8;

fn header(n_obs: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars: N_VARS as u64,
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

/// obs for 8 rows. `cell_type` is "A" in the first CSR shard (rows 0..4) and
/// "B" in the second (rows 4..8) — so each value is absent from exactly one
/// CSR shard. `donor` spans both shards. `n_counts` is small in shard 0 and
/// large in shard 1.
fn full_obs() -> RecordBatch {
    let cell_id: Vec<String> = (0..N_OBS).map(|i| format!("cell_{i}")).collect();
    let cell_type: Vec<&str> = vec!["A", "A", "A", "A", "B", "B", "B", "B"];
    let donor: Vec<&str> = vec!["d1", "d1", "d2", "d2", "d1", "d1", "d2", "d2"];
    let n_counts: Vec<i64> = vec![10, 11, 12, 13, 1000, 1001, 1002, 1003];
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                cell_id.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_type)),
            Arc::new(StringArray::from(donor)),
            Arc::new(Int64Array::from(n_counts)),
        ],
    )
    .unwrap()
}

fn sample_var() -> RecordBatch {
    let ids: Vec<String> = (0..N_VARS).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// A trivial CSR shard of `n_rows` rows × N_VARS cols (1 nnz/row).
fn write_csr_shard(writer: &mut ScxWriter, n_rows: usize, row_start: u64) {
    let indptr: Vec<u64> = (0..=n_rows as u64).collect();
    let indices: Vec<u32> = (0..n_rows as u32).map(|i| i % N_VARS as u32).collect();
    let values: Vec<u8> = vec![1u8; n_rows];
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            row_start,
        )
        .unwrap();
}

/// Build a file with SHARDED obs (4 shards of 2 rows) and CSR shards of 4 rows
/// each — deliberately mismatched boundaries — plus a production obs predicate
/// index over `cell_type` / `donor` / `n_counts`.
fn build_sharded_indexed_file(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("sharded_indexed.scx");
    let obs = full_obs();
    let var = sample_var();
    let mut writer = ScxWriter::new(&path, header(N_OBS)).unwrap();

    // 4 obs metadata shards of 2 rows each: [0,2) [2,4) [4,6) [6,8).
    let obs_splits = [(0u64, 2u64), (2, 2), (4, 2), (6, 2)];
    for (i, (row_start, n)) in obs_splits.iter().enumerate() {
        let slice = obs.slice(*row_start as usize, *n as usize);
        writer
            .write_obs_shard(i as u32, *row_start, *n, N_OBS, &slice)
            .unwrap();
    }
    writer.write_var(&var).unwrap();

    // 2 CSR shards of 4 rows each: [0,4) [4,8).
    write_csr_shard(&mut writer, 4, 0);
    write_csr_shard(&mut writer, 4, 4);

    // The predicate index / per-shard column stats are keyed to the CSR shard
    // ranges, mirroring `compact`.
    let opts = ConversionPredicateIndexOptions {
        index_obs: vec![
            "cell_type".to_string(),
            "donor".to_string(),
            "n_counts".to_string(),
        ],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 1000,
    };
    let csr_row_ranges = [(0u64, 4u64), (4u64, 8u64)];
    build_and_write_conversion_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        N_VARS,
        &opts,
    )
    .unwrap();
    writer.finish().unwrap();
    path
}

fn debug_reader(pipeline: &QueryPipeline) -> &ScxReader {
    pipeline
        .reader()
        .as_any()
        .downcast_ref::<ScxReader>()
        .expect("local query reader should be a ScxReader")
}

/// Decode a (possibly dictionary-encoded) Utf8 column to owned strings for
/// value comparison regardless of the result's categorical dtype.
fn string_values(batch: &RecordBatch, col: &str) -> Vec<Option<String>> {
    let idx = batch.schema().index_of(col).unwrap();
    let arr = arrow::compute::cast(batch.column(idx), &DataType::Utf8).unwrap();
    let s = arr.as_any().downcast_ref::<StringArray>().unwrap();
    (0..s.len())
        .map(|i| {
            if s.is_valid(i) {
                Some(s.value(i).to_string())
            } else {
                None
            }
        })
        .collect()
}

#[test]
fn count_streams_obs_without_full_read() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    // "A" lives only in CSR shard 0 → shard 1 is pruned, so only the obs
    // shards overlapping [0,4) — obs shards 0 and 1 — should be decoded.
    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'A'")
        .unwrap();
    let c = pipeline.count().unwrap();
    assert_eq!(c.total_shards, 2);
    assert_eq!(c.skipped_shards, 1, "CSR shard 1 (no 'A') must be skipped");
    assert_eq!(c.matched_rows, 4);

    let reader = debug_reader(&pipeline);
    assert_eq!(
        reader.debug_counts().read_obs.load(Ordering::Relaxed),
        0,
        "the sharded path must never materialise the full obs table"
    );
    assert_eq!(
        reader.debug_counts().read_obs_shard.load(Ordering::Relaxed),
        2,
        "only the 2 obs shards overlapping surviving CSR shard 0 may be read"
    );
}

#[test]
fn collect_filter_parity_with_reference() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'A'")
        .unwrap()
        .collect()
        .unwrap();

    // Rows 0..4 carry "A".
    assert_eq!(r.x.n_rows(), 4);
    assert_eq!(r.obs.num_rows(), 4);
    assert_eq!(
        string_values(&r.obs, "cell_id"),
        vec![
            Some("cell_0".into()),
            Some("cell_1".into()),
            Some("cell_2".into()),
            Some("cell_3".into()),
        ]
    );
    assert!(string_values(&r.obs, "cell_type")
        .iter()
        .all(|v| v.as_deref() == Some("A")));
}

#[test]
fn collect_no_skip_reads_all_overlapping_shards() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    // "d1" is present in both CSR shards → nothing skipped; rows 0,1,4,5.
    // count() runs only the planning half, so we can assert obs streaming
    // (no full read) on the borrowed pipeline before consuming it.
    let pipeline = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("donor == 'd1'")
        .unwrap();
    let c = pipeline.count().unwrap();
    assert_eq!(c.skipped_shards, 0);
    assert_eq!(c.matched_rows, 4);
    let reader = debug_reader(&pipeline);
    assert_eq!(reader.debug_counts().read_obs.load(Ordering::Relaxed), 0);
    assert_eq!(
        reader.debug_counts().read_obs_shard.load(Ordering::Relaxed),
        4,
        "no-skip predicate must read all 4 obs shards (full cover)"
    );

    let r = pipeline.collect().unwrap();
    assert_eq!(r.skipped_shards, 0);
    assert_eq!(r.matched_rows, 4);
    assert_eq!(
        string_values(&r.obs, "cell_id"),
        vec![
            Some("cell_0".into()),
            Some("cell_1".into()),
            Some("cell_4".into()),
            Some("cell_5".into()),
        ]
    );
}

#[test]
fn limit_reads_minimal_obs_shards() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    // limit(1) over a no-skip predicate: the matching row (cell_0) is in obs
    // shard 0, so materialize should touch only that shard.
    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("donor == 'd1'")
        .unwrap()
        .limit(1)
        .collect()
        .unwrap();
    assert_eq!(r.x.n_rows(), 1);
    assert_eq!(r.obs.num_rows(), 1);
    assert_eq!(
        string_values(&r.obs, "cell_id"),
        vec![Some("cell_0".into())]
    );
    // matched_rows is the pre-limit Level-2 count (4), not the returned 1.
    assert_eq!(r.matched_rows, 4);
}

#[test]
fn empty_result_has_correct_schema() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_id == 'nonexistent'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.x.n_rows(), 0);
    assert_eq!(r.obs.num_rows(), 0);
    // Schema is intact even with zero rows.
    let schema = r.obs.schema();
    let cols: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(cols.contains(&"cell_id"));
    assert!(cols.contains(&"cell_type"));
}
