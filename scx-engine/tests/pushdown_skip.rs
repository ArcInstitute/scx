//! End-to-end test that the production index-build path
//! (`build_and_write_conversion_predicate_indexes`) populates per-shard
//! catalog column stats so query-time `filter_obs` actually skips shards.
//!
//! Before this wiring, `set_shard_column_stats` had no production caller and
//! every `filter_obs` scanned all shards (`skipped_shards == 0`). These tests
//! drive the real builder (not the hand-rolled `bench_query.rs` fixture) and
//! assert shards are skipped for both categorical and numeric predicates.

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{
    build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions, QueryPipeline,
};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::writer::ScxWriter;
use tempfile::TempDir;

const N_VARS: usize = 4;

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

/// obs for 8 rows in two shards of 4. `cell_type` is "A" in shard 0 and "B" in
/// shard 1 (so each value is absent from exactly one shard). `n_counts` is
/// small (10..13) in shard 0 and large (1000..1003) in shard 1.
fn two_shard_obs() -> RecordBatch {
    let cell_id: Vec<String> = (0..8).map(|i| format!("cell_{i}")).collect();
    let cell_type: Vec<&str> = vec!["A", "A", "A", "A", "B", "B", "B", "B"];
    let n_counts: Vec<i64> = vec![10, 11, 12, 13, 1000, 1001, 1002, 1003];
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                cell_id.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_type)),
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
fn write_shard(writer: &mut ScxWriter, n_rows: usize, row_start: u64) {
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

/// Build a two-shard SCX file with an obs predicate index over `cell_type`
/// (categorical) and `n_counts` (numeric), through the production path.
fn build_indexed_file(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("indexed.scx");
    let obs = two_shard_obs();
    let var = sample_var();
    let mut writer = ScxWriter::new(&path, header(8)).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var).unwrap();
    write_shard(&mut writer, 4, 0);
    write_shard(&mut writer, 4, 4);

    let opts = ConversionPredicateIndexOptions {
        index_obs: vec!["cell_type".to_string(), "n_counts".to_string()],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 1000,
    };
    let obs_row_ranges = [(0u64, 4u64), (4u64, 8u64)];
    build_and_write_conversion_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &obs_row_ranges,
        N_VARS,
        &opts,
    )
    .unwrap();
    writer.finish().unwrap();
    path
}

#[test]
fn categorical_filter_skips_shard_without_value() {
    let dir = TempDir::new().unwrap();
    let path = build_indexed_file(&dir);

    // "A" lives only in shard 0 → shard 1 must be skipped.
    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'A'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.total_shards, 2);
    assert_eq!(r.skipped_shards, 1, "shard 1 (no 'A') must be skipped");
    assert_eq!(r.matched_rows, 4, "only shard 0's 4 rows match");

    // "B" lives only in shard 1 → shard 0 must be skipped.
    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'B'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.skipped_shards, 1, "shard 0 (no 'B') must be skipped");
    assert_eq!(r.matched_rows, 4);
}

#[test]
fn numeric_filter_skips_shard_out_of_range() {
    let dir = TempDir::new().unwrap();
    let path = build_indexed_file(&dir);

    // n_counts > 500 → shard 0 (max 13) can't match and must be skipped.
    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("n_counts > 500")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.total_shards, 2);
    assert_eq!(
        r.skipped_shards, 1,
        "shard 0 (max 13 <= 500) must be skipped"
    );
    assert_eq!(r.matched_rows, 4, "shard 1's 4 rows match");
}
