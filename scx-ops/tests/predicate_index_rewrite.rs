//! Integration tests for the merge / append / compact rewrites that
//! accept `ConversionPredicateIndexOptions`. Verifies that the
//! resulting output files actually contain `ObsPredicateIndex` /
//! `VarPredicateIndex` sections — the silent-data-loss bug being
//! fixed here.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;
use scx_ops::{AppendOptions, PredicateIndexBuildSummary};
use tempfile::TempDir;

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
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

/// obs with two indexable low-cardinality categorical columns. The
/// `perturbation` column varies per file so we can later assert that
/// merged-file shard skipping looks plausible.
fn obs_with_categories(n: usize, perturbation: &str) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let pert: Vec<String> = (0..n).map(|_| perturbation.to_string()).collect();
    let cell_types: Vec<String> = (0..n)
        .map(|i| if i % 2 == 0 { "A" } else { "B" }.to_string())
        .collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("perturbation", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                pert.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                cell_types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
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
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

/// Write a single-modality test file with a categorical obs schema
/// suitable for predicate-index forced columns.
fn write_test_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    perturbation: &str,
) -> PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs(&obs_with_categories(n_obs, perturbation))
        .unwrap();
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
            tool: "test".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
    path
}

fn count_section(path: &std::path::Path, section_type: SectionType) -> usize {
    let reader = ScxReader::open(path).unwrap();
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == section_type)
        .count()
}

fn forced_obs_pert_options() -> ConversionPredicateIndexOptions {
    ConversionPredicateIndexOptions {
        index_obs: vec!["perturbation".to_string(), "cell_type".to_string()],
        index_var: vec![],
        index_preset: None,
        index_auto_threshold: 1000,
    }
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

#[test]
fn merge_with_index_options_writes_predicate_index() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 64, 16, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 64, 16, "DRUG_B");
    let out = dir.path().join("merged.scx");

    let summary = scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &out,
        &forced_obs_pert_options(),
    )
    .unwrap();

    assert!(summary.result.is_some(), "expected index result");
    assert!(summary.multimodal_skip.is_none());
    assert_eq!(
        count_section(&out, SectionType::ObsPredicateIndex),
        1,
        "merged file should have one obs_predicate_index section"
    );
}

#[test]
fn merge_without_index_options_writes_no_predicate_index() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 32, 8, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 32, 8, "DRUG_B");
    let out = dir.path().join("merged.scx");

    let summary = scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &out,
        &ConversionPredicateIndexOptions::default(),
    )
    .unwrap();

    assert!(summary.result.is_none(), "no index requested → no result");
    assert_eq!(
        count_section(&out, SectionType::ObsPredicateIndex),
        0,
        "merged file should NOT have an obs_predicate_index section when \
         called with empty index options"
    );
}

#[test]
fn merge_wrapper_preserves_legacy_no_index_behaviour() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 16, 4, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 16, 4, "DRUG_B");
    let out = dir.path().join("merged.scx");

    // Legacy entry point — must compile and produce a file with NO
    // predicate-index sections (matches pre-fix behaviour).
    scx_ops::merge(&[a.as_path(), b.as_path()], &out).unwrap();
    assert_eq!(count_section(&out, SectionType::ObsPredicateIndex), 0);
}

// ---------------------------------------------------------------------------
// compact
// ---------------------------------------------------------------------------

#[test]
fn compact_with_index_options_writes_predicate_index() {
    let dir = TempDir::new().unwrap();
    let input = write_test_file(&dir, "src.scx", 64, 16, "DRUG_A");
    let out = dir.path().join("compacted.scx");

    let summary =
        scx_ops::compact_with_index_options(&input, &out, &forced_obs_pert_options()).unwrap();

    assert!(summary.result.is_some());
    assert!(summary.multimodal_skip.is_none());
    assert_eq!(count_section(&out, SectionType::ObsPredicateIndex), 1);
}

#[test]
fn compact_without_index_options_writes_no_predicate_index() {
    let dir = TempDir::new().unwrap();
    let input = write_test_file(&dir, "src.scx", 32, 8, "DRUG_A");
    let out = dir.path().join("compacted.scx");

    scx_ops::compact(&input, &out).unwrap();
    assert_eq!(count_section(&out, SectionType::ObsPredicateIndex), 0);
}

// ---------------------------------------------------------------------------
// append
// ---------------------------------------------------------------------------

#[test]
fn append_from_reader_with_index_options_writes_predicate_index() {
    let dir = TempDir::new().unwrap();
    let target = write_test_file(&dir, "target.scx", 32, 16, "DRUG_A");
    let source = write_test_file(&dir, "source.scx", 32, 16, "DRUG_B");

    // Verify there's no predicate index pre-append.
    assert_eq!(
        count_section(&target, SectionType::ObsPredicateIndex),
        0,
        "test setup: target must not have a pre-existing predicate index"
    );

    let source_reader = ScxReader::open(&source).unwrap();
    let summary = scx_ops::append_from_reader_with_index_options(
        &target,
        &source_reader,
        &AppendOptions {
            codec: CodecSelection::Auto,
            shard_target_rows: NonZeroU32::new(64).unwrap(),
            modality_id: 0,
        },
        0,
        &forced_obs_pert_options(),
    )
    .unwrap();
    assert!(summary.result.is_some());
    assert!(summary.multimodal_skip.is_none());

    // After append + rebuild, the predicate-index section exists.
    assert_eq!(count_section(&target, SectionType::ObsPredicateIndex), 1);

    // Sanity: the file still opens cleanly with the new catalog.
    let reader = ScxReader::open(&target).unwrap();
    assert_eq!(reader.n_obs(), 64, "obs grew by source row count");
}

#[test]
fn append_without_index_options_preserves_legacy_behaviour() {
    let dir = TempDir::new().unwrap();
    let target = write_test_file(&dir, "target.scx", 16, 8, "DRUG_A");
    let source = write_test_file(&dir, "source.scx", 16, 8, "DRUG_B");
    let source_reader = ScxReader::open(&source).unwrap();

    scx_ops::append_from_reader(
        &target,
        &source_reader,
        &AppendOptions {
            codec: CodecSelection::Auto,
            shard_target_rows: NonZeroU32::new(64).unwrap(),
            modality_id: 0,
        },
        0,
    )
    .unwrap();
    assert_eq!(count_section(&target, SectionType::ObsPredicateIndex), 0);
}

// ---------------------------------------------------------------------------
// type-level smoke test for PredicateIndexBuildSummary
// ---------------------------------------------------------------------------

#[test]
fn predicate_index_build_summary_defaults_to_skipped() {
    let s = PredicateIndexBuildSummary::skipped();
    assert!(s.result.is_none());
    assert!(s.multimodal_skip.is_none());
    assert!(!s.was_multimodal_skip());
}
