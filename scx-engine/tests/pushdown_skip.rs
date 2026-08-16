//! End-to-end test that the production index-build path
//! (`build_and_write_conversion_predicate_indexes`) populates per-shard
//! catalog column stats so query-time `filter_obs` actually skips shards.
//!
//! Before this wiring, `set_shard_column_stats` had no production caller and
//! every `filter_obs` scanned all shards (`skipped_shards == 0`). These tests
//! drive the real builder (not the hand-rolled `bench_query.rs` fixture) and
//! assert shards are skipped for both categorical and numeric predicates.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{
    build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions, QueryPipeline,
};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use tempfile::TempDir;

const N_VARS: usize = 4;

/// How the fixture's categorical obs columns are encoded on disk.
///
/// Every test below runs both. `Dictionary(Int32, Utf8)` is the shape a pandas
/// categorical reaches SCX as (`scx-convert/src/h5ad_obs.rs`), and until this
/// parameterisation no end-to-end engine test wrote one — so the Level-1
/// `CategoryBitset` pruning asserted here had only ever been exercised against
/// `Utf8`. The `skipped_shards` assertions double as the canary: an encoding
/// the builder declines to classify as categorical produces no bitset, no
/// pruning, and `skipped_shards == 0`.
///
/// (`tests/rowset_differential.rs` carries the same enum. Integration test
/// binaries do not share code, and a `tests/common/` module for twenty lines
/// would be a heavier pattern than the duplication.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ObsEncoding {
    Utf8,
    Dictionary,
}

impl ObsEncoding {
    fn data_type(self) -> DataType {
        match self {
            Self::Utf8 => DataType::Utf8,
            Self::Dictionary => {
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
            }
        }
    }

    fn categorical(self, values: Vec<&str>) -> ArrayRef {
        let utf8: ArrayRef = Arc::new(StringArray::from(values));
        match self {
            Self::Utf8 => utf8,
            Self::Dictionary => arrow::compute::cast(&utf8, &self.data_type()).unwrap(),
        }
    }
}

fn header(n_obs: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, N_VARS as u64, 0, 16384, 0, 0)
}

/// obs for 8 rows in two shards of 4. `cell_type` is "A" in shard 0 and "B" in
/// shard 1 (so each value is absent from exactly one shard). `n_counts` is
/// small (10..13) in shard 0 and large (1000..1003) in shard 1. `donor` spans
/// both shards ("d1"/"d2" appear in each) to exercise the no-skip direction.
fn two_shard_obs(enc: ObsEncoding) -> RecordBatch {
    let cell_id: Vec<String> = (0..8).map(|i| format!("cell_{i}")).collect();
    let cell_type: Vec<&str> = vec!["A", "A", "A", "A", "B", "B", "B", "B"];
    let donor: Vec<&str> = vec!["d1", "d1", "d2", "d2", "d1", "d1", "d2", "d2"];
    let n_counts: Vec<i64> = vec![10, 11, 12, 13, 1000, 1001, 1002, 1003];
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", enc.data_type(), false),
        Field::new("donor", enc.data_type(), false),
        Field::new("n_counts", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                cell_id.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            enc.categorical(cell_type),
            enc.categorical(donor),
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

/// Build a two-shard SCX file with an obs predicate index over `cell_type` +
/// `donor` (categorical) and `n_counts` (numeric), through the production path.
fn build_indexed_file(dir: &TempDir, enc: ObsEncoding) -> std::path::PathBuf {
    let path = dir.path().join("indexed.scx");
    let obs = two_shard_obs(enc);
    let var = sample_var();
    let mut writer = ScxWriter::new(&path, header(8)).unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var).unwrap();
    write_shard(&mut writer, 4, 0);
    write_shard(&mut writer, 4, 4);

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
    let obs_row_ranges = [(0u64, 4u64), (4u64, 8u64)];
    let result = build_and_write_conversion_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &obs_row_ranges,
        N_VARS,
        &opts,
    )
    .unwrap();

    // Canary: an encoding whose categoricals the builder declines to index
    // would leave every `skipped_shards` assertion below reading 0 for a reason
    // that is not a bug — a residual query is slow, not wrong. Fail here, where
    // the diagnosis is legible, rather than three asserts later.
    let mut indexed = result.obs_indexed_columns.clone();
    indexed.sort();
    assert_eq!(
        indexed,
        vec![
            "cell_type".to_string(),
            "donor".to_string(),
            "n_counts".to_string()
        ],
        "{enc:?}: obs index does not cover the queried columns; outcomes: {:?}",
        result.obs_outcomes
    );

    writer.finish().unwrap();
    path
}

fn categorical_filter_skips_shard_without_value(enc: ObsEncoding) {
    let dir = TempDir::new().unwrap();
    let path = build_indexed_file(&dir, enc);

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

fn numeric_filter_skips_shard_out_of_range(enc: ObsEncoding) {
    let dir = TempDir::new().unwrap();
    let path = build_indexed_file(&dir, enc);

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

/// Negative direction: a categorical value present in *every* shard must NOT be
/// skipped anywhere. Guards against a shard-space misalignment that unsets a
/// "present" bit and silently drops matching rows (incorrect over-skip).
fn categorical_filter_present_everywhere_skips_nothing(enc: ObsEncoding) {
    let dir = TempDir::new().unwrap();
    let path = build_indexed_file(&dir, enc);

    // "d1" lives in both shards → neither shard may be skipped.
    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("donor == 'd1'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.total_shards, 2);
    assert_eq!(
        r.skipped_shards, 0,
        "'d1' is present in both shards — nothing may be skipped"
    );
    assert_eq!(r.matched_rows, 4, "rows 0,1,4,5 carry donor 'd1'");
}

/// Negative direction (numeric): a range that overlaps every shard's [min,max]
/// must NOT be skipped anywhere.
fn numeric_filter_overlapping_all_shards_skips_nothing(enc: ObsEncoding) {
    let dir = TempDir::new().unwrap();
    let path = build_indexed_file(&dir, enc);

    // n_counts > 5 → shard 0 (max 13) and shard 1 (max 1003) both qualify.
    let r = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("n_counts > 5")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.total_shards, 2);
    assert_eq!(
        r.skipped_shards, 0,
        "both shards have values > 5 — nothing may be skipped"
    );
    assert_eq!(r.matched_rows, 8, "all 8 rows match");
}

// Each fixture above runs in both obs encodings, as separate `#[test]`s so a
// failure names the encoding rather than making the reader infer it.

#[test]
fn categorical_filter_skips_shard_without_value_utf8() {
    categorical_filter_skips_shard_without_value(ObsEncoding::Utf8);
}

#[test]
fn categorical_filter_skips_shard_without_value_dictionary() {
    categorical_filter_skips_shard_without_value(ObsEncoding::Dictionary);
}

#[test]
fn numeric_filter_skips_shard_out_of_range_utf8() {
    numeric_filter_skips_shard_out_of_range(ObsEncoding::Utf8);
}

#[test]
fn numeric_filter_skips_shard_out_of_range_dictionary() {
    numeric_filter_skips_shard_out_of_range(ObsEncoding::Dictionary);
}

#[test]
fn categorical_filter_present_everywhere_skips_nothing_utf8() {
    categorical_filter_present_everywhere_skips_nothing(ObsEncoding::Utf8);
}

#[test]
fn categorical_filter_present_everywhere_skips_nothing_dictionary() {
    categorical_filter_present_everywhere_skips_nothing(ObsEncoding::Dictionary);
}

#[test]
fn numeric_filter_overlapping_all_shards_skips_nothing_utf8() {
    numeric_filter_overlapping_all_shards_skips_nothing(ObsEncoding::Utf8);
}

#[test]
fn numeric_filter_overlapping_all_shards_skips_nothing_dictionary() {
    numeric_filter_overlapping_all_shards_skips_nothing(ObsEncoding::Dictionary);
}
