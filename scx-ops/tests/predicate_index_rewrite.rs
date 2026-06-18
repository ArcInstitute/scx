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
use scx_engine::{ConversionPredicateIndexOptions, QueryPipeline};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_ops::{AppendOptions, PredicateIndexBuildSummary};
use tempfile::TempDir;

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
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

/// Like `write_test_file` but stamps a custom `shard_target_rows` in the
/// header so downstream `compact` re-sharding produces multiple output shards.
fn write_test_file_with_target(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    perturbation: &str,
    shard_target_rows: u32,
) -> PathBuf {
    let path = dir.path().join(filename);
    let mut header = sample_header(n_obs as u64, n_vars as u64);
    header.shard_target_rows = shard_target_rows;
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
fn merge_with_zero_threshold_sentinel_writes_no_predicate_index() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 32, 8, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 32, 8, "DRUG_B");
    let out = dir.path().join("merged.scx");

    // Empty obs/var + no preset + `index_auto_threshold = 0` is the
    // sentinel that the legacy `merge` wrapper passes — disables both
    // auto-detect and forced-column rebuilds. Verifies the wrapper's
    // contract is plumbed through the engine.
    let summary = scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &out,
        &ConversionPredicateIndexOptions {
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
    )
    .unwrap();

    assert!(summary.result.is_none(), "no index requested → no result");
    assert_eq!(
        count_section(&out, SectionType::ObsPredicateIndex),
        0,
        "merged file should NOT have an obs_predicate_index section when \
         called with the zero-threshold sentinel"
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
        scx_ops::compact_with_index_options(&input, &out, &forced_obs_pert_options(), false)
            .unwrap();

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

// ---------------------------------------------------------------------------
// Issue 5 — multimodal-skip, index_var, stale-index-survives coverage
// ---------------------------------------------------------------------------

/// Build a minimal multimodal SCX file with `rna` and `adt` modalities,
/// each carrying a single CSR shard. Modelled on
/// `scx-ops/tests/integration.rs::write_multimodal_with_csc` but trimmed
/// to the bits the predicate-index multimodal-skip test needs.
fn write_multimodal_test_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    rna_n_vars: usize,
    adt_n_vars: usize,
    perturbation: &str,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, rna_n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer
        .write_obs(&obs_with_categories(n_obs, perturbation))
        .unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    let adt_id = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer
        .write_var_for(rna_id, &sample_var(rna_n_vars))
        .unwrap();
    writer
        .write_var_for(adt_id, &sample_var(adt_n_vars))
        .unwrap();
    writer
        .set_modality_n_vars(rna_id, rna_n_vars as u64)
        .unwrap();
    writer
        .set_modality_n_vars(adt_id, adt_n_vars as u64)
        .unwrap();

    let (rna_indptr, rna_indices, rna_values) = sample_shard(n_obs, rna_n_vars);
    writer
        .write_csr_shard_for(
            rna_id,
            &rna_indptr,
            &rna_indices,
            &rna_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    let (adt_indptr, adt_indices, adt_values) = sample_shard(n_obs, adt_n_vars);
    writer
        .write_csr_shard_for(
            adt_id,
            &adt_indptr,
            &adt_indices,
            &adt_values,
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

/// Multimodal merge requests an index → engine read-side is unimodal-only
/// today, so the merged output must contain NO `ObsPredicateIndex` /
/// `VarPredicateIndex` sections and `summary.multimodal_skip` must list
/// the requested columns.
#[test]
fn merge_multimodal_skip_records_summary() {
    let dir = TempDir::new().unwrap();
    let a = write_multimodal_test_file(&dir, "mm_a.scx", 32, 8, 4, "DRUG_A");
    let b = write_multimodal_test_file(&dir, "mm_b.scx", 32, 8, 4, "DRUG_B");
    let out = dir.path().join("mm_merged.scx");

    let summary = scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &out,
        &forced_obs_pert_options(),
    )
    .unwrap();

    assert!(
        summary.result.is_none(),
        "no engine result on multimodal skip"
    );
    let columns = summary
        .multimodal_skip
        .expect("multimodal merge with index opts must record skip");
    assert!(columns.contains(&"perturbation".to_string()));
    assert!(columns.contains(&"cell_type".to_string()));

    assert_eq!(
        count_section(&out, SectionType::ObsPredicateIndex),
        0,
        "multimodal merge must not write obs_predicate_index"
    );
    assert_eq!(
        count_section(&out, SectionType::VarPredicateIndex),
        0,
        "multimodal merge must not write var_predicate_index"
    );
}

/// Asking for `index_var` columns must emit a `VarPredicateIndex` section.
#[test]
fn merge_with_index_var_writes_var_predicate_index() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 32, 8, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 32, 8, "DRUG_B");
    let out = dir.path().join("merged.scx");

    let options = ConversionPredicateIndexOptions {
        index_obs: vec![],
        index_var: vec!["gene_id".to_string()],
        index_preset: None,
        index_auto_threshold: 1000,
    };
    let summary =
        scx_ops::merge_with_index_options(&[a.as_path(), b.as_path()], &out, &options).unwrap();

    assert!(summary.result.is_some());
    assert!(summary.multimodal_skip.is_none());
    assert_eq!(
        count_section(&out, SectionType::VarPredicateIndex),
        1,
        "merged file should have one var_predicate_index section"
    );
}

/// `append_from_reader` (legacy entry point, no index opts) must leave a
/// pre-existing `ObsPredicateIndex` section in place — the documented
/// "stale entries preserved unless --index-*" contract from
/// docs/operations.md.
#[test]
fn append_stale_predicate_index_survives_without_index_options() {
    let dir = TempDir::new().unwrap();
    // Step 1: build a target with an `ObsPredicateIndex` via
    // `compact_with_index_options` (the only existing way to bake one
    // into a test fixture without poking at the engine directly).
    let src = write_test_file(&dir, "src.scx", 32, 8, "DRUG_A");
    let target = dir.path().join("target.scx");
    scx_ops::compact_with_index_options(&src, &target, &forced_obs_pert_options(), false).unwrap();
    assert_eq!(
        count_section(&target, SectionType::ObsPredicateIndex),
        1,
        "test setup: target must have a predicate index pre-append"
    );

    // Step 2: legacy append (no index options) into the same target.
    // The stale obs_predicate_index entry should remain in the catalog
    // (it covers only the pre-append rows — semantically stale, but
    // not dropped).
    let source = write_test_file(&dir, "source.scx", 32, 8, "DRUG_B");
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
    assert_eq!(
        count_section(&target, SectionType::ObsPredicateIndex),
        1,
        "legacy append must preserve stale predicate-index entries"
    );
}

/// Issue 1 (P1) — Forced index column that doesn't exist must fail BEFORE
/// any output file is created, so retrying is safe. Validates the
/// `merge_with_index_options` upfront validation in
/// `validate_forced_columns`.
#[test]
fn merge_with_missing_forced_column_does_not_create_output() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 16, 4, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 16, 4, "DRUG_B");
    let out = dir.path().join("merged.scx");

    let bogus = ConversionPredicateIndexOptions {
        index_obs: vec!["nonexistent_column".to_string()],
        index_var: vec![],
        index_preset: None,
        index_auto_threshold: 1000,
    };
    let err = scx_ops::merge_with_index_options(&[a.as_path(), b.as_path()], &out, &bogus)
        .expect_err("must fail upfront");
    let msg = format!("{err}");
    assert!(
        msg.contains("nonexistent_column"),
        "error message should name the missing column: {msg}"
    );
    assert!(
        !out.exists(),
        "merged output must NOT be on disk after upfront validation failure"
    );
}

// ---------------------------------------------------------------------------
// MERGE-INDEX-OBS-DROPPED.md — Bug 3 (compact preserves sharded obs layout)
// + provenance audit trail for the rebuilt predicate indexes (Bugs 1 & 2).
// ---------------------------------------------------------------------------

/// Write a single-modality test file whose obs metadata is row-sharded
/// (`ObsMetadataShard` sections) across `n_obs_shards` shards — the v0.6.x
/// default layout. Mirrors `write_test_file` but uses `write_obs_shard`
/// instead of `write_obs`.
fn write_sharded_test_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    perturbation: &str,
    n_obs_shards: usize,
) -> PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = obs_with_categories(n_obs, perturbation);
    let total = n_obs as u64;
    let chunk = (n_obs + n_obs_shards - 1) / n_obs_shards.max(1);
    let chunk = chunk.max(1);
    let (mut shard_idx, mut row_start) = (0u32, 0usize);
    while row_start < n_obs {
        let take = chunk.min(n_obs - row_start);
        let slice = obs.slice(row_start, take); // zero-copy
        writer
            .write_obs_shard(shard_idx, row_start as u64, take as u64, total, &slice)
            .unwrap();
        shard_idx += 1;
        row_start += take;
    }

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

/// Bug 3: compacting a sharded-obs input with the default kwargs
/// (`reshape_obs=false`, no `index_obs`) must PRESERVE the row-sharded layout —
/// it must not collapse N `ObsMetadataShard` sections into one legacy
/// `ObsMetadata` section (the atlas-scale OOM path).
#[test]
fn compact_preserves_sharded_obs_layout() {
    let dir = TempDir::new().unwrap();
    let input = write_sharded_test_file(&dir, "src.scx", 64, 16, "DRUG_A", 2);

    let in_reader = ScxReader::open(&input).unwrap();
    assert_eq!(
        in_reader.obs_metadata_shard_count(),
        2,
        "fixture precondition: input obs must be row-sharded"
    );
    drop(in_reader);

    let out = dir.path().join("compacted.scx");
    // Default compact: reshape_obs=false, no predicate index requested.
    scx_ops::compact(&input, &out).unwrap();

    let out_reader = ScxReader::open(&out).unwrap();
    assert!(
        out_reader.obs_metadata_shard_count() > 0,
        "compact must preserve the sharded obs layout (Bug 3), got {} shards",
        out_reader.obs_metadata_shard_count()
    );
}

/// Inverse direction (documented `--reshape-obs` behaviour): a legacy
/// single-section obs input migrates to a sharded layout when
/// `reshape_obs=true`.
#[test]
fn compact_reshape_obs_migrates_legacy_to_sharded() {
    let dir = TempDir::new().unwrap();
    let input = write_test_file(&dir, "legacy.scx", 64, 16, "DRUG_A");

    let in_reader = ScxReader::open(&input).unwrap();
    assert_eq!(
        in_reader.obs_metadata_shard_count(),
        0,
        "fixture precondition: input obs must be legacy single-section"
    );
    drop(in_reader);

    let out = dir.path().join("resharded.scx");
    let sentinel = ConversionPredicateIndexOptions {
        index_obs: Vec::new(),
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 0,
    };
    scx_ops::compact_with_index_options(&input, &out, &sentinel, true).unwrap();

    let out_reader = ScxReader::open(&out).unwrap();
    assert!(
        out_reader.obs_metadata_shard_count() > 0,
        "reshape_obs=true must migrate legacy obs → sharded layout"
    );
}

/// Helper: assert the last provenance record on `path` has `action == action`
/// and that its `params_json.predicate_index.obs_columns` contains `column`.
fn assert_provenance_indexes_obs(path: &std::path::Path, action: &str, column: &str) {
    let reader = ScxReader::open(path).unwrap();
    let prov = reader.read_provenance().unwrap();
    let last = prov
        .operations
        .last()
        .unwrap_or_else(|| panic!("expected at least one provenance record"));
    assert_eq!(last.action, action, "last provenance record action");
    let v: serde_json::Value = serde_json::from_str(&last.params_json)
        .unwrap_or_else(|e| panic!("params_json must be valid JSON: {e} ({})", last.params_json));
    let cols = v["predicate_index"]["obs_columns"]
        .as_array()
        .unwrap_or_else(|| {
            panic!(
                "{action} provenance must carry predicate_index.obs_columns; got {}",
                last.params_json
            )
        });
    let names: Vec<&str> = cols.iter().filter_map(|c| c.as_str()).collect();
    assert!(
        names.contains(&column),
        "{action} provenance must record indexed obs column '{column}'; got {names:?}"
    );
}

/// Bug 1: the merge provenance record must record the rebuilt predicate-index
/// columns in `params_json` (the "NO index field in params_json" report).
#[test]
fn merge_provenance_records_predicate_index() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 64, 16, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 64, 16, "DRUG_B");
    let out = dir.path().join("merged.scx");

    scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &out,
        &forced_obs_pert_options(),
    )
    .unwrap();

    assert_provenance_indexes_obs(&out, "merge", "perturbation");
}

/// Bug 2: the compact provenance record must record the rebuilt
/// predicate-index columns in `params_json`.
#[test]
fn compact_provenance_records_predicate_index() {
    let dir = TempDir::new().unwrap();
    let input = write_test_file(&dir, "src.scx", 64, 16, "DRUG_A");
    let out = dir.path().join("compacted.scx");

    scx_ops::compact_with_index_options(&input, &out, &forced_obs_pert_options(), false).unwrap();

    assert_provenance_indexes_obs(&out, "compact", "perturbation");
}

// ---------------------------------------------------------------------------
// Predicate pushdown: per-shard column stats let `filter_obs` skip shards.
// Each input contributes one CSR shard whose `perturbation` value is unique
// to that file, so a merged/compacted file has per-shard stats that exclude
// the shards of the other file.
// ---------------------------------------------------------------------------

/// `merge_with_index_options` output supports query-time shard skipping.
#[test]
fn merge_output_enables_shard_skipping() {
    let dir = TempDir::new().unwrap();
    let a = write_test_file(&dir, "a.scx", 64, 16, "DRUG_A");
    let b = write_test_file(&dir, "b.scx", 64, 16, "DRUG_B");
    let out = dir.path().join("merged.scx");
    scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &out,
        &forced_obs_pert_options(),
    )
    .unwrap();

    let r = QueryPipeline::open(&out)
        .unwrap()
        .filter_obs("perturbation == 'DRUG_A'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.total_shards, 2, "two inputs → two CSR shards");
    assert_eq!(
        r.skipped_shards, 1,
        "DRUG_B's shard must be skipped when filtering DRUG_A"
    );
    assert_eq!(r.matched_rows, 64, "only DRUG_A's 64 rows match");
}

/// `compact_with_index_options` output supports query-time shard skipping.
#[test]
fn compact_output_enables_shard_skipping() {
    let dir = TempDir::new().unwrap();
    // Small shard target so compact re-sharding keeps ≥2 shards (each still
    // single-perturbation) rather than collapsing both inputs into one shard.
    let a = write_test_file_with_target(&dir, "a.scx", 64, 16, "DRUG_A", 32);
    let b = write_test_file_with_target(&dir, "b.scx", 64, 16, "DRUG_B", 32);
    let merged = dir.path().join("merged.scx");
    scx_ops::merge_with_index_options(
        &[a.as_path(), b.as_path()],
        &merged,
        &forced_obs_pert_options(),
    )
    .unwrap();

    let compacted = dir.path().join("compacted.scx");
    scx_ops::compact_with_index_options(&merged, &compacted, &forced_obs_pert_options(), false)
        .unwrap();

    let r = QueryPipeline::open(&compacted)
        .unwrap()
        .filter_obs("perturbation == 'DRUG_B'")
        .unwrap()
        .collect()
        .unwrap();
    assert!(r.total_shards >= 1);
    assert!(
        r.skipped_shards >= 1,
        "DRUG_A's shard must be skipped when filtering DRUG_B (got {}/{})",
        r.skipped_shards,
        r.total_shards
    );
    assert_eq!(r.matched_rows, 64);
}

/// `append_from_reader_with_index_options` rebuilds per-shard stats across the
/// pre-existing + newly-appended CSR shards, so the appended file supports
/// shard skipping (Phase 2 — append's bespoke catalog assembly).
#[test]
fn append_output_enables_shard_skipping() {
    let dir = TempDir::new().unwrap();
    let target = write_test_file(&dir, "target.scx", 32, 16, "DRUG_A");
    let source = write_test_file(&dir, "source.scx", 32, 16, "DRUG_B");
    let source_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader_with_index_options(
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

    // After append: shard 0 == old DRUG_A rows, shard 1 == appended DRUG_B
    // rows. Filtering DRUG_B must skip the old (DRUG_A) shard.
    let r = QueryPipeline::open(&target)
        .unwrap()
        .filter_obs("perturbation == 'DRUG_B'")
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(r.total_shards, 2, "old shard + one appended shard");
    assert_eq!(
        r.skipped_shards, 1,
        "the pre-existing DRUG_A shard is skipped"
    );
    assert_eq!(r.matched_rows, 32, "only the appended DRUG_B rows match");
}

/// Build a single-modality test file whose var metadata is row-sharded
/// (`VarMetadataShard` sections) across two shards — the layout
/// `pyscx.from_anndata` emits for full-transcriptome files (`n_vars >
/// 16384`). Mirrors `write_test_file` but splits var across two
/// `write_var_shard` calls instead of one `write_var`. Obs stays legacy
/// single-section so the indexed-append exercises the var-sharded read
/// path specifically.
fn write_var_sharded_test_file(
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

    // Split the var RecordBatch rows across exactly two shards.
    let var = sample_var(n_vars);
    let total = n_vars as u64;
    let first = n_vars / 2;
    let second = n_vars - first;
    writer
        .write_var_shard(0, 0, first as u64, total, &var.slice(0, first))
        .unwrap();
    writer
        .write_var_shard(
            1,
            first as u64,
            second as u64,
            total,
            &var.slice(first, second),
        )
        .unwrap();

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

/// Regression: indexed append (`index_obs` non-empty → index-rebuild
/// branch) onto a base whose var is stored as `VarMetadataShard` shards
/// must NOT fail with `SectionNotFound("var")`. The index-rebuild branch
/// reads the pre-existing var to validate forced columns / build the var
/// predicate index; the sharded-var layout has no single `"var"` section,
/// so a `get("var")`-only reader corrupts every full-transcriptome file.
#[test]
fn append_with_index_obs_on_sharded_var_base_preserves_var_section() {
    let dir = TempDir::new().unwrap();
    let n_vars = 64;
    let target = write_var_sharded_test_file(&dir, "target.scx", 32, n_vars, "DRUG_A");
    let source = write_test_file(&dir, "source.scx", 32, n_vars, "DRUG_B");

    // Precondition: the base really exercises the sharded-var path.
    {
        let reader = ScxReader::open(&target).unwrap();
        assert!(
            reader.var_metadata_shard_count() >= 2,
            "fixture precondition: base var must be row-sharded (got {} shards)",
            reader.var_metadata_shard_count()
        );
    }

    let source_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader_with_index_options(
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
    .expect("indexed append onto a var-sharded base must succeed");

    // Re-open and verify the var section survived, obs is readable, and
    // exactly one obs predicate index was built.
    let reader = ScxReader::open(&target).unwrap();
    let var = reader
        .read_var()
        .expect("read_var must succeed after append");
    assert_eq!(
        var.num_rows(),
        n_vars,
        "var row count must equal the original n_vars"
    );
    assert!(
        reader.read_obs().is_ok(),
        "read_obs must succeed after append"
    );
    assert_eq!(
        count_section(&target, SectionType::ObsPredicateIndex),
        1,
        "appended file should have exactly one obs_predicate_index section"
    );
}
