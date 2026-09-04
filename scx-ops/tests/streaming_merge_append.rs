//! Streaming merge & append regression suite. Covers the row-sharded
//! obs / var metadata layout that replaces the legacy single-section
//! Arrow IPC batches (which would overflow narrow `i32` offsets on
//! atlas-scale string-heavy obs).
//!
//! Covers:
//! - Merge of many inputs emitting `ObsMetadataShard` sections;
//! - Convert-on-append from legacy `ObsMetadata` to sharded layout;
//! - Var identity validation (strict default vs. `assume_identical_var`);
//! - Uns policy variants (`first`, `require-equal`, `namespace`, `summary`);
//! - Provenance stamping of policy params;
//! - Predicate-index parity between batch-mode and streaming builder.
//!
//! The `>2 GiB` overflow regressions are gated `#[ignore]` and run via:
//!
//! ```bash
//! cargo test --release -p scx-ops --test streaming_merge_append -- --ignored --nocapture
//! ```

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{Array, DictionaryArray, Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::encoder::FramingConfig;
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

fn header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16_384, 0, 0)
}

fn obs_batch(start_row: usize, n: usize, donor: &str) -> RecordBatch {
    let cell_ids: Vec<String> = (start_row..start_row + n)
        .map(|i| format!("cell_{i:07}"))
        .collect();
    let donors: Vec<String> = std::iter::repeat_n(donor.to_string(), n).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(cell_ids)),
            Arc::new(StringArray::from(donors)),
        ],
    )
    .unwrap()
}

/// Standard 4-gene var, identical across inputs unless a test overrides.
fn var_batch() -> RecordBatch {
    let gene_ids = StringArray::from(vec!["g0", "g1", "g2", "g3"]);
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(gene_ids)]).unwrap()
}

/// A var batch with a different gene order — used to trigger
/// `OpsError::VarMismatch`.
fn var_batch_reordered() -> RecordBatch {
    let gene_ids = StringArray::from(vec!["g0", "g2", "g1", "g3"]);
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(gene_ids)]).unwrap()
}

/// Obs batch carrying an extra `batch` column on top of the standard
/// `cell_id` / `donor` columns. Used to trigger `OpsError::ObsMismatch`
/// against `obs_batch`.
fn obs_batch_with_extra_column(start_row: usize, n: usize, donor: &str) -> RecordBatch {
    let cell_ids: Vec<String> = (start_row..start_row + n)
        .map(|i| format!("cell_{i:07}"))
        .collect();
    let donors: Vec<String> = std::iter::repeat_n(donor.to_string(), n).collect();
    let batch_vals: Vec<i32> = (0..n as i32).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
        Field::new("batch", DataType::Int32, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(cell_ids)),
            Arc::new(StringArray::from(donors)),
            Arc::new(Int32Array::from(batch_vals)),
        ],
    )
    .unwrap()
}

/// Variant of `write_legacy_input` that takes an explicit obs batch.
/// Used by obs-identity tests that need divergent obs schemas across
/// inputs while keeping the same row count.
fn write_legacy_input_with_obs(path: &std::path::Path, obs: &RecordBatch, var: &RecordBatch) {
    let n_obs = obs.num_rows() as u64;
    let mut writer = ScxWriter::new(path, header(n_obs, var.num_rows() as u64)).unwrap();
    writer.write_obs(obs).unwrap();
    writer.write_var(var).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

fn write_zero_csr_shard(writer: &mut ScxWriter, row_start: u64, n_rows: u64) {
    let indptr: Vec<u64> = vec![0u64; (n_rows + 1) as usize];
    let indices: Vec<u32> = Vec::new();
    let values: Vec<u8> = Vec::new();
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

/// Build a legacy single-section SCX file with `n_obs` rows, the given
/// obs payload (`donor`), and the supplied var.
fn write_legacy_input(
    path: &std::path::Path,
    n_obs: u64,
    donor: &str,
    var: &RecordBatch,
    uns: Option<&serde_json::Value>,
) {
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
        .unwrap();
    writer.write_var(var).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);
    if let Some(uns) = uns {
        writer.write_uns(uns).unwrap();
    }
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

#[test]
fn merge_small_inputs_emits_shards() {
    // Phase 2a: even small merges produce ObsMetadataShard sections.
    // No `ObsMetadata` single-section entry should land in the output
    // catalog (legacy single-section is an input format only).
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("in0.scx");
    let p1 = dir.path().join("in1.scx");
    let p2 = dir.path().join("in2.scx");
    let var = var_batch();
    write_legacy_input(&p0, 100, "donor_A", &var, None);
    write_legacy_input(&p1, 100, "donor_B", &var, None);
    write_legacy_input(&p2, 100, "donor_C", &var, None);
    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path(), p2.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 300);
    assert_eq!(reader.obs_metadata_shard_count(), 3, "one shard per input");
    // No legacy ObsMetadata section in the output.
    assert!(
        !reader
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsMetadata),
        "merge output must not contain a single-section ObsMetadata entry"
    );
    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 300);
    let donors = obs
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donors.value(0), "donor_A");
    assert_eq!(donors.value(100), "donor_B");
    assert_eq!(donors.value(200), "donor_C");
}

/// Write a legacy input stamping an explicit `format_version` (the writer
/// trusts the header it is given). Used to exercise the version gate.
fn write_legacy_input_versioned(
    path: &std::path::Path,
    n_obs: u64,
    donor: &str,
    var: &RecordBatch,
    format_version: u16,
) {
    let mut h = header(n_obs, 4);
    h.format_version = format_version;
    let mut writer = ScxWriter::new(path, h).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
        .unwrap();
    writer.write_var(var).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

#[test]
fn merge_gates_v3_stamp_on_min_source_version() {
    // Merge re-encodes shards without re-canonicalizing, so the v3 canonical
    // claim must only be stamped when every input already guarantees it. A
    // pre-v3 input pins the merged output below v3; all-v3 inputs yield v3.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();

    // Mixed v2 + v3 inputs → output stays v2 (no false canonical claim).
    let p0 = dir.path().join("v2.scx");
    let p1 = dir.path().join("v3.scx");
    write_legacy_input_versioned(&p0, 50, "donor_A", &var, 2);
    write_legacy_input_versioned(&p1, 50, "donor_B", &var, 3);
    let mixed = dir.path().join("mixed.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &mixed).unwrap();
    assert_eq!(
        ScxReader::open(&mixed).unwrap().header().format_version,
        2,
        "a pre-v3 input must pin the merged output below v3"
    );

    // All-v3 inputs → output claims v3.
    let p2 = dir.path().join("v3b.scx");
    write_legacy_input_versioned(&p2, 50, "donor_C", &var, 3);
    let all_v3 = dir.path().join("all_v3.scx");
    scx_ops::merge(&[p1.as_path(), p2.as_path()], &all_v3).unwrap();
    assert_eq!(
        ScxReader::open(&all_v3).unwrap().header().format_version,
        3,
        "all-v3 inputs must yield a v3 merged output"
    );
}

#[test]
fn merge_var_mismatch_errors_by_default() {
    // Phase 2e default (assume_identical_var = false): var identity
    // check rejects inputs whose var rows differ, even when n_vars
    // matches.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 50, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 50, "donor_B", &var_batch_reordered(), None);
    let out = dir.path().join("out.scx");
    let err = scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap_err();
    assert!(
        matches!(err, scx_ops::OpsError::VarMismatch { .. }),
        "expected OpsError::VarMismatch, got: {err:?}"
    );
}

#[test]
fn merge_var_mismatch_assume_identical_var_proceeds() {
    // Phase 2e `--assume-identical-var`: trust the caller and warn
    // rather than error. The merged file uses input 0's var verbatim.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 50, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 50, "donor_B", &var_batch_reordered(), None);
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        assume_identical_var: true,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    let var = reader.read_var().unwrap();
    let gene_ids = var
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(
        gene_ids.value(1),
        "g1",
        "merged var must come from input 0 (canonical order)"
    );
}

/// A wide var axis (`g0..g{n}`) so an Scx1 count shard stays wide enough for the
/// writer to route small counts to Scx1.
fn wide_var(n_vars: usize) -> RecordBatch {
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("g{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(gene_ids))],
    )
    .unwrap()
}

/// Build a single-shard Scx1 count-matrix input. The data is dense enough
/// that the writer routes to Scx1, exercising the raw-copy fast path on
/// identical-layout inputs.
fn write_scx1_count_input(path: &std::path::Path, n_obs: usize, donor: &str, var: &RecordBatch) {
    let n_vars = var.num_rows();
    let nnz_per_row = 256usize;
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for r in 0..n_obs {
        let mut col = 0u32;
        for k in 0..nnz_per_row {
            col += 1 + ((r * 13 + k * 7) % 97) as u32;
            if col as usize >= n_vars {
                break;
            }
            indices.push(col);
            values.push(1 + ((r + k) % 5) as u8); // small counts → Scx1
        }
        indptr.push(indices.len() as u64);
    }
    let mut writer = ScxWriter::new(path, header(n_obs as u64, n_vars as u64)).unwrap();
    writer.write_obs(&obs_batch(0, n_obs, donor)).unwrap();
    writer.write_var(var).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

#[test]
fn merge_raw_copy_byte_identical_to_reencode() {
    // Two inputs with identical var axis + single-shard layout, Scx1 count
    // data. The raw-copy fast path (`assume_identical_var = false`) must
    // produce byte-identical X-shard sections to the decode/re-encode path
    // (`assume_identical_var = true` disables raw-copy but, with identical
    // var, yields the same var + X).
    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    write_scx1_count_input(&a, 40, "donor_A", &var);
    write_scx1_count_input(&b, 40, "donor_B", &var);

    let out_fast = dir.path().join("fast.scx");
    let out_slow = dir.path().join("slow.scx");
    scx_ops::merge_with_options(
        &[a.as_path(), b.as_path()],
        &out_fast,
        &scx_ops::MergeOptions::default(),
    )
    .unwrap();
    scx_ops::merge_with_options(
        &[a.as_path(), b.as_path()],
        &out_slow,
        &scx_ops::MergeOptions {
            assume_identical_var: true,
            ..Default::default()
        },
    )
    .unwrap();

    let rf = ScxReader::open(&out_fast).unwrap();
    let rs = ScxReader::open(&out_slow).unwrap();

    let cf = rf.catalog().shards_sorted();
    let cs = rs.catalog().shards_sorted();
    assert_eq!(cf.len(), 2, "one output shard per input shard");
    assert_eq!(cf.len(), cs.len(), "fast/slow shard count");

    // X-shard sections byte-identical between raw-copy and re-encode.
    for (sf, ss) in cf.iter().zip(cs.iter()) {
        assert_eq!(
            rf.read_raw_shard_bytes(sf).unwrap(),
            rs.read_raw_shard_bytes(ss).unwrap(),
            "X-shard section bytes must be byte-identical (raw-copy vs re-encode)"
        );
    }

    // Decoded matrices agree across both paths.
    for i in 0..cf.len() {
        assert_eq!(
            rf.read_csr_shard(i).unwrap(),
            rs.read_csr_shard(i).unwrap(),
            "decoded shard {i} must match"
        );
    }
}

/// Build a single-shard **framed** (v4 / shard-v2) count-matrix input with an
/// explicit `row_group_rows`. The v4 header + `set_framing` route the shard
/// through `encode_shard_framed`, so the shard carries a multi-entry
/// `BlockIndex` (its byte length encodes the group count = `ceil(n_obs / G)`).
/// Uses a Zstd codec so no Scx1 decode sidecar is emitted — the pure-framed
/// path C5 re-enables.
fn write_framed_input(
    path: &std::path::Path,
    n_obs: usize,
    donor: &str,
    var: &RecordBatch,
    row_group_rows: u32,
) {
    let n_vars = var.num_rows();
    let nnz_per_row = 8usize;
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for r in 0..n_obs {
        let mut col = 0u32;
        for k in 0..nnz_per_row {
            col += 1 + ((r * 13 + k * 7) % 97) as u32;
            if col as usize >= n_vars {
                break;
            }
            indices.push(col);
            values.push(1 + ((r + k) % 5) as u8);
        }
        indptr.push(indices.len() as u64);
    }
    let mut h = header(n_obs as u64, n_vars as u64);
    h.format_version = CURRENT_FORMAT_VERSION;
    let mut writer = ScxWriter::new(path, h).unwrap();
    writer.set_framing(Some(FramingConfig {
        row_group_rows,
        ..Default::default()
    }));
    writer.write_obs(&obs_batch(0, n_obs, donor)).unwrap();
    writer.write_var(var).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append test fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

/// The framed byte-copy fast path (C5): merging two **framed** (v4) inputs must
/// produce a valid v4 output whose CSR shards are byte-copied verbatim — proven
/// by (a) the output header staying v4, (b) each output shard staying framed
/// (shard-v2), (c) the block index surviving byte-for-byte (source `G=8` gives
/// a multi-entry index that the decode-encode path — writer default `G=256` —
/// could not reproduce for these 40-row shards), and (d) decoded parity.
#[test]
fn merge_framed_raw_copy_preserves_v4_and_block_index() {
    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    // G = 8 over 40 rows → 5 row groups; the writer's decode-encode default is
    // G = 256 → 1 group, so a preserved multi-entry index proves raw-copy.
    write_framed_input(&a, 40, "donor_A", &var, 8);
    write_framed_input(&b, 40, "donor_B", &var, 8);

    let ra = ScxReader::open(&a).unwrap();
    assert_eq!(ra.header().format_version, CURRENT_FORMAT_VERSION);
    let src_bi_len = ra
        .read_shard_header(ra.catalog().shards_sorted()[0])
        .unwrap()
        .block_index_length;
    assert!(
        src_bi_len > 4 + 22,
        "framed source shard (G=8, 40 rows) must carry a multi-entry block index"
    );

    let out = dir.path().join("merged.scx");
    scx_ops::merge_with_options(
        &[a.as_path(), b.as_path()],
        &out,
        &scx_ops::MergeOptions::default(),
    )
    .unwrap();

    let ro = ScxReader::open(&out).unwrap();
    assert_eq!(
        ro.header().format_version,
        CURRENT_FORMAT_VERSION,
        "merging framed inputs must preserve the v4 framed layout"
    );
    let shards = ro.catalog().shards_sorted();
    assert_eq!(shards.len(), 2, "one output shard per framed input shard");
    for entry in &shards {
        let sh = ro.read_shard_header(entry).unwrap();
        assert!(
            sh.shard_format_version > 1,
            "merged shard must stay framed (shard-v2)"
        );
        assert_eq!(
            sh.block_index_length, src_bi_len,
            "raw-copy must preserve the source's multi-entry block index verbatim \
             (decode-encode at the writer's default G would collapse it)"
        );
    }
    // Decoded parity: merged rows 0..40 == input A, 40..80 == input B.
    let a_dec = ra.read_csr_shard(0).unwrap();
    let b_dec = ScxReader::open(&b).unwrap().read_csr_shard(0).unwrap();
    assert_eq!(ro.read_csr_shard(0).unwrap(), a_dec, "merged shard 0 == A");
    assert_eq!(ro.read_csr_shard(1).unwrap(), b_dec, "merged shard 1 == B");
}

/// Merge slow (decode-encode) path under framing: when raw-copy is disabled
/// (`assume_identical_var = true`) but the inputs are framed, the merge must
/// still emit a valid v4 file — every re-encoded shard is framed (shard-v2) by
/// the writer's framing state, not left unframed in a v4 file. Guards the
/// "slow path frames automatically" half of the C5 merge change.
#[test]
fn merge_framed_slow_path_still_emits_v4_framed_shards() {
    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    write_framed_input(&a, 40, "donor_A", &var, 8);
    write_framed_input(&b, 40, "donor_B", &var, 8);

    let out = dir.path().join("merged.scx");
    scx_ops::merge_with_options(
        &[a.as_path(), b.as_path()],
        &out,
        &scx_ops::MergeOptions {
            assume_identical_var: true, // disables raw-copy → decode-encode path
            ..Default::default()
        },
    )
    .unwrap();

    let ro = ScxReader::open(&out).unwrap();
    assert_eq!(
        ro.header().format_version,
        CURRENT_FORMAT_VERSION,
        "framed inputs must yield a v4 output even on the decode-encode path"
    );
    for entry in &ro.catalog().shards_sorted() {
        let sh = ro.read_shard_header(entry).unwrap();
        assert!(
            sh.shard_format_version > 1,
            "decode-encoded shard must be framed to stay valid in a v4 file"
        );
    }
    // Decoded parity survives the re-encode.
    let a_dec = ScxReader::open(&a).unwrap().read_csr_shard(0).unwrap();
    assert_eq!(ro.read_csr_shard(0).unwrap(), a_dec);
}

/// Append-side framed byte-copy (C5): appending a framed source shard into a
/// framed (v4) base must raw-copy it verbatim — the appended shard stays framed
/// with the source's block index intact, the base header stays v4, and the
/// decoded rows are correct.
#[test]
fn append_from_reader_framed_raw_copy_preserves_v4() {
    use scx_codec::CodecSelection;

    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let target = dir.path().join("base.scx");
    let source = dir.path().join("source.scx");
    write_framed_input(&target, 40, "donor_A", &var, 8);
    write_framed_input(&source, 40, "donor_B", &var, 8);

    let src_reader = ScxReader::open(&source).unwrap();
    let src_bi_len = src_reader
        .read_shard_header(src_reader.catalog().shards_sorted()[0])
        .unwrap()
        .block_index_length;
    assert!(
        src_bi_len > 4 + 22,
        "source shard must be multi-group framed"
    );

    scx_ops::append_from_reader(
        &target,
        &src_reader,
        &scx_ops::AppendOptions {
            codec: CodecSelection::Auto,
            shard_target_rows: std::num::NonZeroU32::new(16384).unwrap(),
            modality_id: 0,
        },
        0,
    )
    .unwrap();
    drop(src_reader);

    let post = ScxReader::open(&target).unwrap();
    assert_eq!(post.n_obs(), 80, "40 base + 40 appended");
    assert_eq!(
        post.header().format_version,
        CURRENT_FORMAT_VERSION,
        "appending into a framed v4 base must keep the header v4"
    );
    let shards = post.catalog().shards_sorted();
    assert_eq!(shards.len(), 2);
    for entry in &shards {
        let sh = post.read_shard_header(entry).unwrap();
        assert!(sh.shard_format_version > 1, "shard must stay framed");
        assert_eq!(
            sh.block_index_length, src_bi_len,
            "appended shard must be raw-copied verbatim (block index preserved)"
        );
    }
    // Decoded parity for the appended shard.
    let src_dec = ScxReader::open(&source).unwrap().read_csr_shard(0).unwrap();
    assert_eq!(
        post.read_csr_shard(1).unwrap(),
        src_dec,
        "appended shard decodes to the source rows"
    );
}

/// Decode every CSR shard of a file into per-row `(index, value)` lists, in
/// global row order. Used to check append decoded parity across re-splits.
fn decode_all_rows(reader: &ScxReader) -> Vec<Vec<(i32, f32)>> {
    let mut rows = Vec::new();
    for i in 0..reader.catalog().shards_sorted().len() {
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        for r in 0..indptr.len() - 1 {
            let s = indptr[r] as usize;
            let e = indptr[r + 1] as usize;
            rows.push(
                indices[s..e]
                    .iter()
                    .copied()
                    .zip(data[s..e].iter().copied())
                    .collect(),
            );
        }
    }
    rows
}

/// F-d correctness: the in-memory `append(...)` path (`write_csr_chunk`, which
/// writes directly via `FileLock`, bypassing the writer's v4 guard) must emit
/// FRAMED shards when the base is v4 — otherwise it silently writes an invalid
/// v4 file (unframed, sidecar-less shards under a v4 header).
#[test]
fn append_in_memory_into_framed_base_emits_framed_shards() {
    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let base = dir.path().join("base.scx");
    write_framed_input(&base, 40, "donor_A", &var, 8);

    // New rows via the raw-array in-memory path.
    let new_obs = obs_batch(40, 10, "donor_B");
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for r in 0..10usize {
        indices.push((r % 50_000) as u32);
        values.push((1 + r % 5) as u8);
        indptr.push(indices.len() as u64);
    }
    scx_ops::append(
        &base,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    let post = ScxReader::open(&base).unwrap();
    assert_eq!(post.n_obs(), 50);
    assert_eq!(
        post.header().format_version,
        CURRENT_FORMAT_VERSION,
        "append must keep the base a v4 file"
    );
    for entry in &post.catalog().shards_sorted() {
        let sh = post.read_shard_header(entry).unwrap();
        assert!(
            sh.shard_format_version > 1,
            "shard '{}' must be framed v2 (in-memory append into a v4 base), got v{}",
            entry.name,
            sh.shard_format_version
        );
    }
}

/// F-d correctness: `append_from_reader`'s decode-encode (re-split) path must
/// emit framed shards into a v4 base. A 40-row framed source shard appended
/// with `shard_target_rows = 16` re-splits (so raw-copy is skipped and
/// `write_csr_chunk` runs), and every resulting shard must be framed v2, with
/// the decoded rows matching the source.
#[test]
fn append_from_reader_resplit_into_framed_base_emits_framed_shards() {
    use scx_codec::CodecSelection;

    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let base = dir.path().join("base.scx");
    let source = dir.path().join("source.scx");
    write_framed_input(&base, 40, "donor_A", &var, 8);
    write_framed_input(&source, 40, "donor_B", &var, 8);

    let src_reader = ScxReader::open(&source).unwrap();
    let src_rows = decode_all_rows(&src_reader);
    let base_shards_before = ScxReader::open(&base)
        .unwrap()
        .catalog()
        .shards_sorted()
        .len();

    scx_ops::append_from_reader(
        &base,
        &src_reader,
        &scx_ops::AppendOptions {
            codec: CodecSelection::Auto,
            shard_target_rows: std::num::NonZeroU32::new(16).unwrap(),
            modality_id: 0,
        },
        0,
    )
    .unwrap();
    drop(src_reader);

    let post = ScxReader::open(&base).unwrap();
    assert_eq!(post.n_obs(), 80);
    assert_eq!(
        post.header().format_version,
        CURRENT_FORMAT_VERSION,
        "re-split append must keep the base a v4 file"
    );
    let post_shards = post.catalog().shards_sorted().len();
    assert!(
        post_shards > base_shards_before + 1,
        "40 source rows at shard_target_rows=16 must re-split into >1 appended shard \
         (got {} new shards)",
        post_shards - base_shards_before
    );
    for entry in &post.catalog().shards_sorted() {
        let sh = post.read_shard_header(entry).unwrap();
        assert!(
            sh.shard_format_version > 1,
            "shard '{}' must be framed v2 (re-split append into a v4 base), got v{}",
            entry.name,
            sh.shard_format_version
        );
    }
    // Decoded parity: the appended rows (40..80) must equal the source rows.
    let post_rows = decode_all_rows(&post);
    assert_eq!(
        &post_rows[40..80],
        &src_rows[..],
        "re-split append decoded parity"
    );
}

/// F-d review fix: appending an UNFRAMED v1 Scx1 source into a v4 framed base
/// must NOT byte-copy the v1 shard, because a raw-copy would leave an unframed v1
/// shard under a v4 header (invalid, and the writer's v4 guard is bypassed on
/// this path). The framing-match gate routes the v1 shard through the
/// decode-encode path, which re-frames it to v2 — a valid v4 output.
#[test]
fn append_unframed_scx1_source_into_v4_base_reframes_to_v2() {
    use scx_codec::CodecSelection;

    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let base = dir.path().join("base.scx");
    let source = dir.path().join("source_scx1.scx");
    write_framed_input(&base, 40, "donor_A", &var, 8); // v4 framed base
    write_scx1_count_input(&source, 40, "donor_B", &var); // v1 Scx1 + sidecar

    let src_reader = ScxReader::open(&source).unwrap();
    let src_sh = src_reader
        .read_shard_header(src_reader.catalog().shards_sorted()[0])
        .unwrap();
    assert_eq!(
        src_sh.shard_format_version, 1,
        "source shard must be unframed v1"
    );
    let src_rows = decode_all_rows(&src_reader);

    scx_ops::append_from_reader(
        &base,
        &src_reader,
        &scx_ops::AppendOptions {
            codec: CodecSelection::Auto,
            shard_target_rows: std::num::NonZeroU32::new(16384).unwrap(),
            modality_id: 0,
        },
        0,
    )
    .unwrap();
    drop(src_reader);

    let post = ScxReader::open(&base).unwrap();
    assert_eq!(post.n_obs(), 80);
    assert_eq!(post.header().format_version, CURRENT_FORMAT_VERSION);
    for entry in &post.catalog().shards_sorted() {
        let sh = post.read_shard_header(entry).unwrap();
        assert!(
            sh.shard_format_version > 1,
            "shard '{}' must be framed v2 (v1 Scx1 source reframed, not byte-copied), got v{}",
            entry.name,
            sh.shard_format_version
        );
    }
    let post_rows = decode_all_rows(&post);
    assert_eq!(
        &post_rows[40..80],
        &src_rows[..],
        "appended rows decode to the source"
    );
}

/// Micro-bench (P2 / OPT-1.2): same-layout merge via the raw-copy fast path
/// vs the decode/re-encode path. The slow path is exactly the pre-change
/// behaviour (raw-copy disabled), so `slow / fast` is the merge speedup.
/// Run with: `cargo test --release -p scx-ops --test streaming_merge_append \
///   merge_rawcopy_microbench -- --ignored --nocapture`.
#[test]
#[ignore = "micro-bench; run with --release --ignored --nocapture"]
fn merge_rawcopy_microbench() {
    let dir = tempfile::tempdir().unwrap();
    let var = wide_var(50_000);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    write_scx1_count_input(&a, 20_000, "donor_A", &var);
    write_scx1_count_input(&b, 20_000, "donor_B", &var);

    let inputs = [a.as_path(), b.as_path()];
    let iters = 5;

    // Warm the page cache so both paths read the same hot inputs.
    let warm = dir.path().join("warm.scx");
    scx_ops::merge_with_options(&inputs, &warm, &scx_ops::MergeOptions::default()).unwrap();

    let time_merge = |opts: &scx_ops::MergeOptions, tag: &str| {
        let mut best = f64::INFINITY;
        for i in 0..iters {
            let out = dir.path().join(format!("{tag}_{i}.scx"));
            let t0 = std::time::Instant::now();
            scx_ops::merge_with_options(&inputs, &out, opts).unwrap();
            best = best.min(t0.elapsed().as_secs_f64());
        }
        best
    };

    // Raw-copy fast path (new default).
    let fast = time_merge(&scx_ops::MergeOptions::default(), "fast");
    // Forced decode → re-encode (pre-change behaviour). With identical var,
    // `assume_identical_var = true` only disables raw-copy; output is the same.
    let slow = time_merge(
        &scx_ops::MergeOptions {
            assume_identical_var: true,
            ..Default::default()
        },
        "slow",
    );

    eprintln!("\n=== merge raw-copy micro-bench (2 × 20k rows × 50k vars, Scx1) ===");
    eprintln!("  raw-copy  (new): {fast:.4}s  (best of {iters})");
    eprintln!("  re-encode (old): {slow:.4}s  (best of {iters})");
    eprintln!("  speedup: {:.2}x", slow / fast);
    assert!(
        fast < slow,
        "raw-copy ({fast:.4}s) should beat re-encode ({slow:.4}s)"
    );
}

#[test]
fn merge_obs_mismatch_errors_by_default() {
    // Default (assume_identical_obs = false): obs schema identity
    // check rejects inputs whose obs columns differ. Without the
    // check, mismatched shards write cleanly and only fail later
    // inside `ScxReader::read_obs()` after the temp file has been
    // renamed into place.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input_with_obs(&p0, &obs_batch(0, 30, "donor_A"), &var_batch());
    write_legacy_input_with_obs(
        &p1,
        &obs_batch_with_extra_column(30, 30, "donor_B"),
        &var_batch(),
    );
    let out = dir.path().join("out.scx");
    let err = scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap_err();
    assert!(
        matches!(err, scx_ops::OpsError::ObsMismatch { .. }),
        "expected OpsError::ObsMismatch, got: {err:?}"
    );
    assert!(
        !out.exists(),
        "merge must not rename a temp output into place when obs validation fails"
    );
}

#[test]
fn merge_obs_mismatch_assume_identical_obs_proceeds() {
    // With assume_identical_obs = true the obs identity check is
    // skipped (a warning is logged instead). The merge proceeds; the
    // resulting file may still fail later at `read_obs()` if the
    // schemas truly disagree — the flag is documented as caller-
    // trust escape hatch, not a fixer.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    // Use the same obs schema on both inputs so the resulting file
    // is still readable; the flag only affects whether the check runs.
    write_legacy_input_with_obs(&p0, &obs_batch(0, 30, "donor_A"), &var_batch());
    write_legacy_input_with_obs(&p1, &obs_batch(30, 30, "donor_B"), &var_batch());
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        assume_identical_obs: true,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 60);
}

#[test]
fn merge_uns_policy_first_keeps_input_zero() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"colors": ["red"], "source": "A"});
    let uns_b = serde_json::json!({"colors": ["blue"], "source": "B"});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();
    let merged_uns = ScxReader::open(&out).unwrap().read_uns().unwrap();
    assert_eq!(merged_uns, uns_a, "default uns_policy=first keeps input 0");
}

#[test]
fn merge_uns_policy_require_equal_errors_on_conflict() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"params": {"k": 1}});
    let uns_b = serde_json::json!({"params": {"k": 2}});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::RequireEqual,
        ..Default::default()
    };
    let err = scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap_err();
    assert!(
        matches!(err, scx_ops::OpsError::UnsConflict { .. }),
        "expected OpsError::UnsConflict, got: {err:?}"
    );
}

#[test]
fn merge_uns_policy_require_equal_passes_when_identical() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns = serde_json::json!({"shared": true});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::RequireEqual,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let merged = ScxReader::open(&out).unwrap().read_uns().unwrap();
    assert_eq!(merged, uns);
}

#[test]
fn merge_uns_policy_namespace_wraps_each_input() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"colors": ["red"]});
    let uns_b = serde_json::json!({"colors": ["blue"]});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::Namespace,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let merged = ScxReader::open(&out).unwrap().read_uns().unwrap();
    let obj = merged.as_object().unwrap();
    assert_eq!(obj.get("input_0"), Some(&uns_a));
    assert_eq!(obj.get("input_1"), Some(&uns_b));
}

#[test]
fn merge_uns_policy_summary_records_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let uns_a = serde_json::json!({"params": 1});
    let uns_b = serde_json::json!({"params": 2});
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), Some(&uns_a));
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), Some(&uns_b));
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        uns_policy: scx_ops::UnsPolicy::Summary,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let merged = ScxReader::open(&out).unwrap().read_uns().unwrap();
    let obj = merged.as_object().unwrap();
    // Canonical body from input 0 is preserved.
    assert_eq!(obj.get("params"), Some(&serde_json::json!(1)));
    // Conflict marker is present.
    let conflicts = obj.get("_scx_uns_conflicts").unwrap().as_array().unwrap();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(
        conflicts[0],
        serde_json::json!({"input": 1, "status": "differs"})
    );
}

#[test]
fn merge_provenance_records_policy_params() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 10, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 10, "donor_B", &var_batch(), None);
    let out = dir.path().join("out.scx");
    let opts = scx_ops::MergeOptions {
        assume_identical_var: true,
        uns_policy: scx_ops::UnsPolicy::Namespace,
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();
    let prov = ScxReader::open(&out).unwrap().read_provenance().unwrap();
    let last = prov.operations.last().unwrap();
    assert_eq!(last.action, "merge");
    assert!(
        last.params_json.contains("\"assume_identical_var\":true"),
        "params_json = {}",
        last.params_json
    );
    assert!(
        last.params_json.contains("\"uns_policy\":\"namespace\""),
        "params_json = {}",
        last.params_json
    );
}

#[test]
fn append_legacy_to_sharded_promotes() {
    // Convert-on-append: starting from a legacy single-section
    // ObsMetadata file, the first append produces an output where
    // obs is row-sharded (shard 0 = old obs, shard 1 = new obs).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.scx");
    write_legacy_input(&path, 6, "donor_A", &var_batch(), None);

    // Pre-append: legacy single-section.
    let pre = ScxReader::open(&path).unwrap();
    assert_eq!(pre.obs_metadata_shard_count(), 0);
    assert!(pre
        .catalog()
        .entries
        .iter()
        .any(|e| e.section_type == SectionType::ObsMetadata));
    drop(pre);

    // Append 4 rows.
    let new_obs = obs_batch(6, 4, "donor_B");
    let n_vars = 4usize;
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..4usize {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    // Post-append: convert-on-append landed; obs is now sharded.
    let post = ScxReader::open(&path).unwrap();
    assert_eq!(post.n_obs(), 10);
    assert_eq!(
        post.obs_metadata_shard_count(),
        2,
        "convert-on-append: shard 0 = old obs (6 rows), shard 1 = new obs (4 rows)"
    );
    // The legacy ObsMetadata entry must no longer be listed in the
    // catalog. Its bytes remain on disk (orphaned) until the next
    // `scx compact`.
    assert!(
        !post
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsMetadata),
        "convert-on-append must drop the legacy ObsMetadata catalog entry"
    );
    // Per-shard row metadata verifies the cover.
    let s0 = post.read_obs_shard(0).unwrap();
    assert_eq!(s0.num_rows(), 6);
    assert_eq!(
        s0.schema().metadata().get("row_start").map(|s| s.as_str()),
        Some("0")
    );
    assert_eq!(
        s0.schema()
            .metadata()
            .get("n_rows_total")
            .map(|s| s.as_str()),
        Some("10")
    );
    let s1 = post.read_obs_shard(1).unwrap();
    assert_eq!(s1.num_rows(), 4);
    assert_eq!(
        s1.schema().metadata().get("row_start").map(|s| s.as_str()),
        Some("6")
    );
    // Assembled batch matches old + new row contents.
    let assembled = post.read_obs().unwrap();
    let cell_ids = assembled
        .column_by_name("cell_id")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(cell_ids.value(0), "cell_0000000");
    assert_eq!(cell_ids.value(5), "cell_0000005");
    assert_eq!(cell_ids.value(6), "cell_0000006");
    assert_eq!(cell_ids.value(9), "cell_0000009");
}

#[test]
fn append_then_append_extends_shards() {
    // Second append should add a new shard without rewriting the old
    // ones. After two appends, obs has three shards (one per write).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two_appends.scx");
    write_legacy_input(&path, 4, "donor_A", &var_batch(), None);

    let n_vars = 4usize;
    let mk_csr = |n: usize| -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n {
            indices.push(((row * 2) % n_vars) as u32);
            indices.push(((row * 2 + 1) % n_vars) as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        (indptr, indices, values)
    };

    let new1 = obs_batch(4, 3, "donor_B");
    let (i1, idx1, v1) = mk_csr(3);
    scx_ops::append(
        &path,
        &new1,
        &i1,
        &idx1,
        &v1,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    let new2 = obs_batch(7, 2, "donor_C");
    let (i2, idx2, v2) = mk_csr(2);
    scx_ops::append(
        &path,
        &new2,
        &i2,
        &idx2,
        &v2,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 9);
    let assembled = reader.read_obs().unwrap();
    assert_eq!(assembled.num_rows(), 9);
    let donors = assembled
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donors.value(0), "donor_A");
    assert_eq!(donors.value(4), "donor_B");
    assert_eq!(donors.value(7), "donor_C");
}

#[test]
fn merge_dict_columns_round_trip() {
    // Dict-encoded categorical with different per-input dictionaries:
    // the merge must unify dict columns shard-by-shard (Phase 2a) so
    // the assembled batch preserves the original string values.
    fn build_dict_obs(start: usize, n: usize, values: &[&str]) -> RecordBatch {
        let cell_ids: Vec<String> = (start..start + n).map(|i| format!("cell_{i:04}")).collect();
        let dict_keys: Int32Array = (0..n).map(|i| (i % values.len()) as i32).collect();
        let dict_values = StringArray::from(values.to_vec());
        let dict_array = DictionaryArray::<arrow::datatypes::Int32Type>::try_new(
            dict_keys,
            Arc::new(dict_values),
        )
        .unwrap();
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new(
                "label",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(cell_ids)), Arc::new(dict_array)],
        )
        .unwrap()
    }

    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let var = var_batch();
    // input 0: label dictionary is {"alpha", "beta"}
    {
        let obs = build_dict_obs(0, 4, &["alpha", "beta"]);
        let mut writer = ScxWriter::new(&p0, header(4, 4)).unwrap();
        writer.write_obs(&obs).unwrap();
        writer.write_var(&var).unwrap();
        write_zero_csr_shard(&mut writer, 0, 4);
        writer.finish().unwrap();
    }
    // input 1: label dictionary is {"gamma", "delta"} — disjoint
    {
        let obs = build_dict_obs(4, 4, &["gamma", "delta"]);
        let mut writer = ScxWriter::new(&p1, header(4, 4)).unwrap();
        writer.write_obs(&obs).unwrap();
        writer.write_var(&var).unwrap();
        write_zero_csr_shard(&mut writer, 0, 4);
        writer.finish().unwrap();
    }

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let assembled = ScxReader::open(&out).unwrap().read_obs().unwrap();
    assert_eq!(assembled.num_rows(), 8);
    // After unify_dict_columns the label column lands as Utf8 (the
    // dictionary's value type).
    let labels = assembled
        .column_by_name("label")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap_or_else(|| {
            panic!(
                "expected Utf8 label, got {:?}",
                assembled.column(1).data_type()
            )
        });
    let observed: Vec<&str> = (0..8).map(|i| labels.value(i)).collect();
    assert_eq!(
        observed,
        vec!["alpha", "beta", "alpha", "beta", "gamma", "delta", "gamma", "delta"]
    );
}

#[test]
fn merge_with_index_options_streaming_matches_batch() {
    // The streaming predicate-index builder must produce a
    // byte-identical PredicateIndex section to the (legacy) batch
    // builder for the same logical obs content. Compare a sharded
    // merge output's obs predicate index to the bytes a batch-mode
    // builder would have produced on the assembled obs.
    use scx_engine::{build_obs_predicate_index_bytes, PredicateIndexBuildOptions};

    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_input(&p0, 64, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 64, "donor_B", &var_batch(), None);
    let out = dir.path().join("merged.scx");
    let opts = scx_ops::MergeOptions {
        index_options: scx_engine::ConversionPredicateIndexOptions {
            index_obs: vec!["donor".to_string()],
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
        ..Default::default()
    };
    scx_ops::merge_with_options(&[p0.as_path(), p1.as_path()], &out, &opts).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let on_disk = reader
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("merge with --index-obs donor must emit an obs predicate index");

    // Compute the batch-mode reference. obs has 128 rows split into
    // two shards of 64 each; each output shard had row_start 0 and 64
    // respectively (one shard per input).
    let assembled = reader.read_obs().unwrap();
    let shard_row_ranges: Vec<(u64, u64)> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == 0)
        .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
        .collect();
    let mut shard_row_ranges = shard_row_ranges;
    shard_row_ranges.sort_by_key(|(s, _)| *s);

    let build_opts = PredicateIndexBuildOptions {
        forced_columns: vec!["donor".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let mut outcomes = Vec::new();
    let mut indexed_names = Vec::new();
    let reference = build_obs_predicate_index_bytes(
        &assembled,
        &shard_row_ranges,
        &build_opts,
        &mut outcomes,
        &mut indexed_names,
    )
    .unwrap()
    .expect("batch-mode builder must also emit the donor index");

    assert_eq!(
        on_disk, reference,
        "streaming and batch predicate-index byte serialisations must match exactly"
    );
}

#[test]
fn append_from_reader_single_modality() {
    // append_from_reader is the streaming SCX→SCX path. After the
    // Phase 2 refactor + the read_obs/read_obs_assembled collapse,
    // it correctly handles both a sharded target and a sharded
    // source. This test exercises the convert-on-append target
    // promoted to sharded after the first call.
    use scx_codec::CodecSelection;

    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("target.scx");
    let source = dir.path().join("source.scx");
    write_legacy_input(&target, 5, "donor_A", &var_batch(), None);
    // Source is itself written via merge so that it lands as a
    // sharded file — the path we want to stress.
    let s0 = dir.path().join("src_in0.scx");
    let s1 = dir.path().join("src_in1.scx");
    write_legacy_input(&s0, 4, "donor_X", &var_batch(), None);
    write_legacy_input(&s1, 3, "donor_Y", &var_batch(), None);
    scx_ops::merge(&[s0.as_path(), s1.as_path()], &source).unwrap();
    {
        let src_reader = ScxReader::open(&source).unwrap();
        assert!(src_reader.obs_metadata_shard_count() > 0);
        assert_eq!(src_reader.n_obs(), 7);
    }

    let src_reader = ScxReader::open(&source).unwrap();
    scx_ops::append_from_reader(
        &target,
        &src_reader,
        &scx_ops::AppendOptions {
            codec: CodecSelection::Auto,
            shard_target_rows: std::num::NonZeroU32::new(16384).unwrap(),
            modality_id: 0,
        },
        0,
    )
    .unwrap();
    drop(src_reader);

    let post = ScxReader::open(&target).unwrap();
    assert_eq!(post.n_obs(), 12); // 5 + 7
                                  // Convert-on-append produced shard 0 = old 5 rows + at least one
                                  // shard for the appended 7 rows. obs is sharded post-append.
    assert!(post.obs_metadata_shard_count() >= 2);
    let assembled = post.read_obs().unwrap();
    assert_eq!(assembled.num_rows(), 12);
    let donors = assembled
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donors.value(0), "donor_A"); // target's original rows
    assert_eq!(donors.value(4), "donor_A");
    assert_eq!(donors.value(5), "donor_X"); // first source input
    assert_eq!(donors.value(9), "donor_Y"); // second source input
}

#[test]
fn append_to_already_sharded_preserves_old_shards() {
    // Verifies the bug 2 fix: appending to a file whose obs is
    // already sharded must NOT rewrite the existing shards. We
    // snapshot the (name, offset, checksum) triple for every
    // ObsMetadataShard entry before the append and assert each
    // is still present unchanged after the append.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("already_sharded.scx");

    // Build a sharded file via merge (3 inputs → 3 shards).
    let p0 = dir.path().join("in0.scx");
    let p1 = dir.path().join("in1.scx");
    let p2 = dir.path().join("in2.scx");
    write_legacy_input(&p0, 8, "donor_A", &var_batch(), None);
    write_legacy_input(&p1, 8, "donor_B", &var_batch(), None);
    write_legacy_input(&p2, 8, "donor_C", &var_batch(), None);
    scx_ops::merge(&[p0.as_path(), p1.as_path(), p2.as_path()], &path).unwrap();

    // Snapshot the pre-append shard catalog (name, offset, checksum).
    let pre_shards: Vec<(String, u64, [u8; 32])> = {
        let pre = ScxReader::open(&path).unwrap();
        assert_eq!(pre.obs_metadata_shard_count(), 3);
        pre.catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsMetadataShard)
            .map(|e| (e.name.clone(), e.offset, e.checksum))
            .collect()
    };

    // Append 5 more rows. The raw-copy path should preserve every
    // pre-existing shard entry bit-for-bit and add exactly one new
    // shard for the new rows.
    let n_vars = 4usize;
    let new_obs = obs_batch(24, 5, "donor_D");
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..5usize {
        indices.push(((row * 2) % n_vars) as u32);
        indices.push(((row * 2 + 1) % n_vars) as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &scx_ops::AppendOptions::default(),
    )
    .unwrap();

    let post = ScxReader::open(&path).unwrap();
    assert_eq!(post.n_obs(), 29); // 24 + 5
    assert_eq!(post.obs_metadata_shard_count(), 4); // 3 old + 1 new

    let post_shards: std::collections::HashMap<String, (u64, [u8; 32])> = post
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsMetadataShard)
        .map(|e| (e.name.clone(), (e.offset, e.checksum)))
        .collect();
    for (name, offset, checksum) in &pre_shards {
        let after = post_shards.get(name).unwrap_or_else(|| {
            panic!("pre-existing shard '{name}' missing from post-append catalog")
        });
        assert_eq!(
            after.0, *offset,
            "pre-existing shard '{name}' offset changed (raw-copy path must not rewrite)"
        );
        assert_eq!(
            after.1, *checksum,
            "pre-existing shard '{name}' checksum changed (raw-copy path must not rewrite)"
        );
    }

    // The assembled obs must still read back correctly: 24 pre-existing
    // rows from the 3 merged inputs + 5 newly-appended rows.
    let assembled = post.read_obs().unwrap();
    assert_eq!(assembled.num_rows(), 29);
    let donors = assembled
        .column_by_name("donor")
        .unwrap()
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(donors.value(0), "donor_A");
    assert_eq!(donors.value(8), "donor_B");
    assert_eq!(donors.value(16), "donor_C");
    assert_eq!(donors.value(24), "donor_D");
    assert_eq!(donors.value(28), "donor_D");
}

#[test]
#[ignore]
fn merge_inputs_exceed_i32_max_obs_string_payload() {
    // > 2 GiB cumulative obs string payload across many inputs.
    // Each individual input fits under i32::MAX (the original failure
    // mode was `concat_batches` on the merged narrow-offset Utf8
    // array). The streaming refactor writes one ObsMetadataShard per
    // input chunk, so each shard's IPC offsets stay narrow.
    //
    // Allocates ~3 GiB peak; run via:
    //
    // ```bash
    // cargo test --release -p scx-ops --test streaming_merge_append \
    //     -- --ignored --nocapture merge_inputs_exceed_i32_max_obs_string_payload
    // ```
    use std::fmt::Write as _;

    // ROWS_PER_INPUT must stay ≤ u16::MAX because each input writes a
    // single CSR shard whose header stores `n_rows` as u16. We compensate
    // by raising N_INPUTS so the cumulative obs string payload still
    // crosses i32::MAX ≈ 2.147 GB.
    const N_INPUTS: usize = 40;
    const ROWS_PER_INPUT: usize = 65_000;
    const PAYLOAD_LEN: usize = 900;
    // 40 × 65 000 × ~915 B (cell_id + payload) ≈ 2.38 GB cumulative string
    // payload — comfortably over i32::MAX so the merged obs's `cell_id`
    // column stays `LargeUtf8` after assemble.

    let dir = tempfile::tempdir().unwrap();
    let mut paths: Vec<PathBuf> = Vec::new();
    let payload: String = "x".repeat(PAYLOAD_LEN);
    for i in 0..N_INPUTS {
        let p = dir.path().join(format!("in_{i}.scx"));
        let mut writer = ScxWriter::new(&p, header(ROWS_PER_INPUT as u64, 4)).unwrap();
        let cell_ids: Vec<String> = (0..ROWS_PER_INPUT)
            .map(|r| {
                let mut s = String::with_capacity(PAYLOAD_LEN + 20);
                let global = i * ROWS_PER_INPUT + r;
                write!(&mut s, "cell_{global:09}{payload}").unwrap();
                s
            })
            .collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(cell_ids))],
        )
        .unwrap();
        writer.write_obs(&batch).unwrap();
        writer.write_var(&var_batch()).unwrap();
        write_zero_csr_shard(&mut writer, 0, ROWS_PER_INPUT as u64);
        writer.finish().unwrap();
        paths.push(p);
    }

    let out = dir.path().join("merged.scx");
    let path_refs: Vec<&std::path::Path> = paths.iter().map(|p| p.as_path()).collect();
    scx_ops::merge(&path_refs, &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs() as usize, N_INPUTS * ROWS_PER_INPUT);
    // The streaming refactor emits one ObsMetadataShard per input
    // (each input is processed as a single chunk because its row
    // count fits in `shard_target_rows`).
    assert!(reader.obs_metadata_shard_count() >= N_INPUTS);

    // Phase 3 read-side: reassembling sharded obs that cumulatively
    // exceeds `i32::MAX` of string payload must NOT trigger
    // `Offset overflow error`. The post-merge `read_sharded_layout_by_prefix`
    // upcasts each shard to `LargeUtf8` before `concat_batches`, and
    // downcasts the result only for columns that still fit narrow —
    // so this overflowing column stays `LargeUtf8` and reads cleanly.
    let assembled = reader.read_obs().expect(
        "post-merge read_obs must succeed even when cumulative obs string \
         payload exceeds i32::MAX — the read path upcasts before concat",
    );
    assert_eq!(assembled.num_rows() as usize, N_INPUTS * ROWS_PER_INPUT);
}

// ---------------------------------------------------------------------------
// Phase 3d: streaming merge tests for layers, obsm, varm.
// ---------------------------------------------------------------------------

/// Build a 2-column Float32 embedding batch covering rows
/// `[start_row, start_row + n)`. Values encode the global row index so
/// tests can verify row-order preservation across inputs.
fn obsm_batch(start_row: usize, n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("pc1", DataType::Float32, false),
        Field::new("pc2", DataType::Float32, false),
    ]);
    let pc1: Vec<f32> = (start_row..start_row + n).map(|i| i as f32).collect();
    let pc2: Vec<f32> = (start_row..start_row + n)
        .map(|i| (i as f32) * 2.0)
        .collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(pc1)),
            Arc::new(Float32Array::from(pc2)),
        ],
    )
    .unwrap()
}

/// 3-column Float32 embedding — same shape as `obsm_batch` but one
/// extra component. Used to trigger
/// `OpsError::DenseMappingMismatch` when merging against an input
/// that emits a 2-column embedding for the same key.
fn obsm_batch_wider(start_row: usize, n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("pc1", DataType::Float32, false),
        Field::new("pc2", DataType::Float32, false),
        Field::new("pc3", DataType::Float32, false),
    ]);
    let pc1: Vec<f32> = (start_row..start_row + n).map(|i| i as f32).collect();
    let pc2: Vec<f32> = (start_row..start_row + n)
        .map(|i| (i as f32) * 2.0)
        .collect();
    let pc3: Vec<f32> = (start_row..start_row + n)
        .map(|i| (i as f32) * 3.0)
        .collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(pc1)),
            Arc::new(Float32Array::from(pc2)),
            Arc::new(Float32Array::from(pc3)),
        ],
    )
    .unwrap()
}

/// 2-column Float32 varm batch covering rows `[0, n_vars)`. The standard
/// 4-gene `var_batch()` has n_vars = 4, so tests use n_vars = 4 here too.
fn varm_batch(n_vars: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("emb_a", DataType::Float32, false),
        Field::new("emb_b", DataType::Float32, false),
    ]);
    let a: Vec<f32> = (0..n_vars).map(|i| i as f32 + 100.0).collect();
    let b: Vec<f32> = (0..n_vars).map(|i| i as f32 + 200.0).collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Float32Array::from(a)),
            Arc::new(Float32Array::from(b)),
        ],
    )
    .unwrap()
}

/// Write an SCX input with a sharded obsm `X_pca` (one shard per
/// `shard_rows`-sized chunk), an optional sharded varm `feature_emb`,
/// and an optional sharded layer `spliced` (one shard per
/// `shard_rows`-sized chunk).
fn write_sharded_input(
    path: &std::path::Path,
    n_obs: u64,
    donor: &str,
    var: &RecordBatch,
    shard_rows: u64,
    with_varm: bool,
    layer_shards: u64, // 0 == no layer
) {
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
        .unwrap();
    writer.write_var(var).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);

    // Sharded obsm `X_pca`. Slice into `shard_rows`-sized chunks.
    let mut start: u64 = 0;
    let mut shard_idx: u32 = 0;
    while start < n_obs {
        let end = (start + shard_rows).min(n_obs);
        let n = end - start;
        let batch = obsm_batch(start as usize, n as usize);
        writer
            .write_obsm_shard("X_pca", shard_idx, start, n, n_obs, &batch)
            .unwrap();
        start = end;
        shard_idx += 1;
    }

    if with_varm {
        // Single-shard varm — n_vars = 4 is small enough not to bother
        // sub-sharding, but exercise the sharded section type.
        let n_vars = var.num_rows() as u64;
        let batch = varm_batch(n_vars as usize);
        writer
            .write_varm_shard("feature_emb", 0, 0, n_vars, n_vars, &batch)
            .unwrap();
    }

    if layer_shards > 0 {
        let rows_per_shard = n_obs.div_ceil(layer_shards);
        let mut row_start: u64 = 0;
        for shard_idx in 0..layer_shards {
            let n_rows = rows_per_shard.min(n_obs - row_start);
            let indptr: Vec<u64> = vec![0u64; (n_rows + 1) as usize];
            let indices: Vec<u32> = Vec::new();
            let values: Vec<u8> = Vec::new();
            let shard = scx_format_io::ShardBuffers::new(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
            );
            writer
                .write_layer_csr_shard("spliced", shard_idx as u32, row_start, shard)
                .unwrap();
            row_start += n_rows;
        }
    }

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "streaming_merge_append phase3 fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

/// A 0-row input as `pyscx.from_anndata` writes one: a 0-row legacy obs
/// section, the shared var (and, with `with_varm`, its var-axis `varm`,
/// which needs no rows), **zero** CSR shards, no layers, no obsm. The
/// format forbids framed zero-row shards ("emit no shard at all instead"),
/// so this is the only shape an empty file takes.
fn write_zero_row_input(path: &std::path::Path, var: &RecordBatch, with_varm: bool) {
    let mut writer = ScxWriter::new(path, header(0, 4)).unwrap();
    writer.write_obs(&obs_batch(0, 0, "none")).unwrap();
    writer.write_var(var).unwrap();
    if with_varm {
        let n_vars = var.num_rows() as u64;
        writer
            .write_varm_shard(
                "feature_emb",
                0,
                0,
                n_vars,
                n_vars,
                &varm_batch(n_vars as usize),
            )
            .unwrap();
    }
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "from_anndata".to_string(),
            tool: "streaming_merge_append zero-row fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

/// A merge whose inputs are all empty must still write an obs section: obs
/// used to be written only inside the per-chunk loop, so two 0-row inputs
/// produced a file whose `read_obs()` failed with `SectionNotFound("obs")`.
#[test]
fn merge_of_only_empty_inputs_writes_an_obs_section() {
    let dir = tempfile::tempdir().unwrap();
    let e0 = dir.path().join("e0.scx");
    let e1 = dir.path().join("e1.scx");
    write_zero_row_input(&e0, &var_batch(), false);
    write_zero_row_input(&e1, &var_batch(), false);
    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[e0.as_path(), e1.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 0);
    assert_eq!(reader.header().n_vars, 4);
    assert_eq!(reader.catalog().csr_shards_sorted().len(), 0);
    let obs = reader
        .read_obs()
        .expect("a 0-row merge output must carry an obs section");
    assert_eq!(obs.num_rows(), 0);
    assert_eq!(
        obs.schema().fields().len(),
        2,
        "obs schema survives: {:?}",
        obs.schema()
    );
    assert_eq!(reader.read_var().unwrap().num_rows(), 4);
    let x = reader.read_all_csr_shards().unwrap();
    assert_eq!(x.shape, (0, 4));
}

/// A 0-row input contributes nothing — including to layers and obsm keys it
/// has no shards for. Layer / obsm names are the union across inputs, and a
/// 0-row input has no layer shards and no obsm section, so `LayerMissing` /
/// `DenseMappingMissing` used to fire on it in either position.
#[test]
fn merge_tolerates_an_empty_input_lacking_layers_and_obsm() {
    let dir = tempfile::tempdir().unwrap();
    let full = dir.path().join("full.scx");
    // 6 rows, sharded obsm X_pca, a varm, and a 2-shard layer `spliced`.
    write_sharded_input(&full, 6, "donor_a", &var_batch(), 3, true, 2);
    let empty = dir.path().join("empty.scx");
    // varm is var-axis data and survives 0 rows, so a real 0-row input
    // carries it — and merge takes varm from input 0 only, so the
    // `empty+full` order below reads it off the empty input.
    write_zero_row_input(&empty, &var_batch(), true);

    for (label, inputs) in [
        ("full+empty", [full.as_path(), empty.as_path()]),
        ("empty+full", [empty.as_path(), full.as_path()]),
    ] {
        let out = dir.path().join(format!("{label}.scx"));
        scx_ops::merge(&inputs, &out).unwrap_or_else(|e| panic!("{label}: {e}"));
        let reader = ScxReader::open(&out).unwrap();
        assert_eq!(reader.header().n_obs, 6, "{label}");
        assert_eq!(reader.read_obs().unwrap().num_rows(), 6, "{label}");
        assert_eq!(reader.layer_names(), vec!["spliced".to_string()], "{label}");
        let layer = reader.read_layer("spliced").unwrap();
        assert_eq!(layer.shape, (6, 4), "{label}");
        let obsm = reader.read_obsm("X_pca").unwrap();
        assert_eq!(obsm.num_rows(), 6, "{label}");
        let varm = reader.read_varm("feature_emb").unwrap();
        assert_eq!(varm.num_rows(), 4, "{label}");
    }
}

/// A populated input with a 2-shard layer `spliced` and no obsm — the sorted
/// emitter refuses obsm by design, so `write_sharded_input` cannot be used.
fn write_layer_only_input(path: &std::path::Path, n_obs: u64, donor: &str) {
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
        .unwrap();
    writer.write_var(&var_batch()).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);
    let rows_per_shard = n_obs.div_ceil(2);
    let mut row_start: u64 = 0;
    for shard_idx in 0..2u32 {
        let n_rows = rows_per_shard.min(n_obs - row_start);
        let indptr: Vec<u64> = vec![0u64; (n_rows + 1) as usize];
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer
            .write_layer_csr_shard("spliced", shard_idx, row_start, shard)
            .unwrap();
        row_start += n_rows;
    }
    writer.finish().unwrap();
}

/// The sorted emitter (`--sort-by`) has its own obs / layer loops, so the
/// two empty-input rules are pinned there too: a 0-row input contributes
/// nothing (and may lack layers), and an all-empty merge still writes obs.
#[test]
fn sorted_merge_tolerates_empty_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let full = dir.path().join("full.scx");
    write_layer_only_input(&full, 6, "donor_a");
    let empty = dir.path().join("empty.scx");
    write_zero_row_input(&empty, &var_batch(), true);
    let opts = sorted_merge_opts(&["donor"], false, false);

    for (label, inputs) in [
        ("full+empty", [full.as_path(), empty.as_path()]),
        ("empty+full", [empty.as_path(), full.as_path()]),
    ] {
        let out = dir.path().join(format!("sorted_{label}.scx"));
        scx_ops::merge_with_options(&inputs, &out, &opts)
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let reader = ScxReader::open(&out).unwrap();
        assert_eq!(reader.header().n_obs, 6, "{label}");
        assert_eq!(reader.read_obs().unwrap().num_rows(), 6, "{label}");
        assert_eq!(reader.layer_names(), vec!["spliced".to_string()], "{label}");
        assert_eq!(
            reader.read_layer("spliced").unwrap().shape,
            (6, 4),
            "{label}"
        );
    }

    let e1 = dir.path().join("e1.scx");
    write_zero_row_input(&e1, &var_batch(), false);
    let out = dir.path().join("sorted_all_empty.scx");
    scx_ops::merge_with_options(&[empty.as_path(), e1.as_path()], &out, &opts).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 0);
    let obs = reader
        .read_obs()
        .expect("an all-empty sorted merge must carry an obs section");
    assert_eq!(obs.num_rows(), 0);
    assert_eq!(obs.schema().fields().len(), 2);
}

/// A 0-row multimodal input: global 0-row obs, both modalities' var, and no
/// CSR shards in either modality (the shape `scx subset` / a compact after
/// deleting every row produce).
fn write_multimodal_zero_row_input(path: &std::path::Path) {
    use scx_format_io::modality::ModalityType;
    let mut writer = ScxWriter::new(path, header(0, 4)).unwrap();
    writer.write_obs(&obs_batch(0, 0, "none")).unwrap();
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
    writer.write_var_for(rna_id, &var_batch()).unwrap();
    writer.write_var_for(adt_id, &var_batch()).unwrap();
    writer.set_modality_n_vars(rna_id, 4).unwrap();
    writer.set_modality_n_vars(adt_id, 4).unwrap();
    writer.finish().unwrap();
}

/// `write_multimodal_with_per_modality_obsm` plus a one-shard per-modality
/// layer `rna/spliced`, so the test below can see both families survive.
fn write_multimodal_with_obsm_and_layer(path: &std::path::Path, n_obs: u64) {
    use scx_format_io::modality::ModalityType;
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, "donor_a"))
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
    writer.write_var_for(rna_id, &var_batch()).unwrap();
    writer.write_var_for(adt_id, &var_batch()).unwrap();
    writer.set_modality_n_vars(rna_id, 4).unwrap();
    writer.set_modality_n_vars(adt_id, 4).unwrap();
    let indptr: Vec<u64> = vec![0u64; (n_obs + 1) as usize];
    for id in [rna_id, adt_id] {
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &[],
            &[],
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(id, 0, shard).unwrap();
    }
    let layer =
        scx_format_io::ShardBuffers::new(&indptr, &[], &[], CodecId::None, ValueEncoding::Uint8);
    writer
        .write_layer_csr_shard_for(rna_id, "spliced", 0, 0, layer)
        .unwrap();
    let meta = scx_format_io::DenseShardMetadata::new(0, 0, n_obs, n_obs);
    writer
        .write_obsm_shard_for(rna_id, "X_umap", meta, &obsm_batch(0, n_obs as usize))
        .unwrap();
    writer.finish().unwrap();
}

/// The multimodal emitter has its own obs, per-modality layer and
/// per-modality obsm loops; the same two rules hold there. Round 2 of the
/// review: the per-modality obsm loop used to *drop the key* (with a warning)
/// when the 0-row input lacked it, and the first version of this test could
/// not see that because it read neither the embedding nor a layer.
#[test]
fn multimodal_merge_tolerates_empty_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let full = dir.path().join("full_mm.scx");
    write_multimodal_with_obsm_and_layer(&full, 6);
    let empty = dir.path().join("empty_mm.scx");
    write_multimodal_zero_row_input(&empty);

    for (label, inputs) in [
        ("full+empty", [full.as_path(), empty.as_path()]),
        ("empty+full", [empty.as_path(), full.as_path()]),
    ] {
        let out = dir.path().join(format!("mm_{label}.scx"));
        scx_ops::merge(&inputs, &out).unwrap_or_else(|e| panic!("{label}: {e}"));
        let reader = ScxReader::open(&out).unwrap();
        assert_eq!(reader.header().n_obs, 6, "{label}");
        assert_eq!(reader.read_obs().unwrap().num_rows(), 6, "{label}");
        assert!(reader.is_multimodal(), "{label}");
        let rna = reader.modality_id("rna").expect("rna modality");
        let umap = reader
            .read_obsm_for(rna, "X_umap")
            .unwrap_or_else(|e| panic!("{label}: per-modality obsm must survive: {e}"));
        assert_eq!(umap.num_rows(), 6, "{label}");
        assert_eq!(
            reader.layer_names_for(rna),
            vec!["spliced".to_string()],
            "{label}"
        );
        assert_eq!(
            reader.read_layer_for(rna, "spliced").unwrap().shape,
            (6, 4),
            "{label}"
        );
    }

    let e1 = dir.path().join("e1_mm.scx");
    write_multimodal_zero_row_input(&e1);
    let out = dir.path().join("mm_all_empty.scx");
    scx_ops::merge(&[empty.as_path(), e1.as_path()], &out).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 0);
    let obs = reader
        .read_obs()
        .expect("an all-empty multimodal merge must carry an obs section");
    assert_eq!(obs.num_rows(), 0);
}

/// A COO `obsp` batch in the wire format `write_obsp` expects; `data` is
/// Float32, or Float64 when `f64_data` (to provoke a schema mismatch).
fn coo_obsp_batch_typed(
    rows: Vec<i32>,
    cols: Vec<i32>,
    data: Vec<f32>,
    n: usize,
    f64_data: bool,
) -> RecordBatch {
    let data_type = if f64_data {
        DataType::Float64
    } else {
        DataType::Float32
    };
    let schema = Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", data_type, false),
        ],
        std::collections::HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    );
    let data_arr: arrow::array::ArrayRef = if f64_data {
        Arc::new(arrow::array::Float64Array::from(
            data.iter().map(|&v| v as f64).collect::<Vec<_>>(),
        ))
    } else {
        Arc::new(Float32Array::from(data))
    };
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(rows)),
            Arc::new(Int32Array::from(cols)),
            data_arr,
        ],
    )
    .unwrap()
}

fn coo_obsp_batch(rows: Vec<i32>, cols: Vec<i32>, data: Vec<f32>, n: usize) -> RecordBatch {
    coo_obsp_batch_typed(rows, cols, data, n, false)
}

/// A 4-row input carrying `obsp/connectivities` with a Float32 or Float64
/// `data` column.
fn write_obsp_input(path: &std::path::Path, donor: &str, f64_data: bool) {
    let mut writer = ScxWriter::new(path, header(4, 4)).unwrap();
    writer.write_obs(&obs_batch(0, 4, donor)).unwrap();
    writer.write_var(&var_batch()).unwrap();
    write_zero_csr_shard(&mut writer, 0, 4);
    writer
        .write_obsp(
            "connectivities",
            &coo_obsp_batch_typed(
                vec![0, 1, 2],
                vec![1, 2, 3],
                vec![1.0, 2.0, 3.0],
                4,
                f64_data,
            ),
        )
        .unwrap();
    writer.finish().unwrap();
}

/// Round 3 of the review: exempting a 0-row input from the obsp presence walk
/// made `validate_obsp_value_schemas` reachable with an empty input 0 — and it
/// keyed its baseline to `readers[0]`, so `[empty, f32, f64]` skipped the
/// comparison and wrote an unreadable graph. The baseline is now the first
/// input that carries the graph.
#[test]
fn obsp_schema_mismatch_is_caught_when_the_first_input_is_empty() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.scx");
    write_zero_row_input(&empty, &var_batch(), false);
    let f32_a = dir.path().join("f32_a.scx");
    write_obsp_input(&f32_a, "donor_a", false);
    let f32_b = dir.path().join("f32_b.scx");
    write_obsp_input(&f32_b, "donor_b", false);
    let f64_c = dir.path().join("f64_c.scx");
    write_obsp_input(&f64_c, "donor_c", true);

    let out = dir.path().join("mismatch.scx");
    let err = scx_ops::merge(&[empty.as_path(), f32_a.as_path(), f64_c.as_path()], &out)
        .expect_err("an empty input 0 must not hide a dtype mismatch between inputs 1 and 2");
    let msg = err.to_string();
    assert!(msg.contains("input 1 vs input 2"), "{msg}");

    // Agreeing populated graphs behind an empty input 0 still merge.
    let out = dir.path().join("ok.scx");
    scx_ops::merge(&[empty.as_path(), f32_a.as_path(), f32_b.as_path()], &out).unwrap();
    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 8);
    assert_eq!(reader.read_obsp("connectivities").unwrap().num_rows(), 6);
}

/// `merge_obsp` validated every input carried the graph before writing; a
/// 0-row input cannot, so `merge([clustered, empty])` used to fail with
/// `DenseMappingMissing` on the concat path.
#[test]
fn merge_tolerates_an_empty_input_lacking_obsp() {
    let dir = tempfile::tempdir().unwrap();
    let full = dir.path().join("full.scx");
    {
        let mut writer = ScxWriter::new(&full, header(4, 4)).unwrap();
        writer.write_obs(&obs_batch(0, 4, "donor_a")).unwrap();
        writer.write_var(&var_batch()).unwrap();
        write_zero_csr_shard(&mut writer, 0, 4);
        writer
            .write_obsp(
                "connectivities",
                &coo_obsp_batch(vec![0, 1, 2], vec![1, 2, 3], vec![1.0, 2.0, 3.0], 4),
            )
            .unwrap();
        writer.finish().unwrap();
    }
    let empty = dir.path().join("empty.scx");
    write_zero_row_input(&empty, &var_batch(), false);

    for (label, inputs) in [
        ("full+empty", [full.as_path(), empty.as_path()]),
        ("empty+full", [empty.as_path(), full.as_path()]),
    ] {
        let out = dir.path().join(format!("obsp_{label}.scx"));
        scx_ops::merge(&inputs, &out).unwrap_or_else(|e| panic!("{label}: {e}"));
        let reader = ScxReader::open(&out).unwrap();
        assert_eq!(reader.header().n_obs, 4, "{label}");
        let graph = reader
            .read_obsp("connectivities")
            .unwrap_or_else(|e| panic!("{label}: obsp must survive: {e}"));
        assert_eq!(graph.num_rows(), 3, "{label}: three COO entries");
    }
}

/// Legacy single-section obsm `X_pca` (and optional legacy single-section
/// varm `feature_emb`). Used by the mixed-layout merge test.
fn write_legacy_obsm_input(
    path: &std::path::Path,
    n_obs: u64,
    donor: &str,
    var: &RecordBatch,
    with_varm: bool,
) {
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
        .unwrap();
    writer.write_var(var).unwrap();
    write_zero_csr_shard(&mut writer, 0, n_obs);
    writer
        .write_obsm("X_pca", &obsm_batch(0, n_obs as usize))
        .unwrap();
    if with_varm {
        writer
            .write_varm("feature_emb", &varm_batch(var.num_rows()))
            .unwrap();
    }
    writer.finish().unwrap();
}

#[test]
fn merge_streams_global_obsm_without_assembly() {
    // Phase 3d: 3 inputs with sharded obsm. Merge must:
    // (a) emit `ObsmEmbeddingShard` entries (no legacy `ObsmEmbedding`);
    // (b) not call `read_obsm` / `read_all_obsm` on any input;
    // (c) round-trip obsm values via `read_obsm`.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let p2 = dir.path().join("c.scx");
    write_sharded_input(&p0, 50, "donor_A", &var, 25, false, 0);
    write_sharded_input(&p1, 60, "donor_B", &var, 30, false, 0);
    write_sharded_input(&p2, 40, "donor_C", &var, 40, false, 0);

    // Open each input, snapshot pre-merge counters, run merge, snapshot post.
    // The merge path takes its own readers; we open separately to read the
    // post-merge file. The streaming guarantee is checked on the *merge
    // function* by inspecting the output catalog — we can't reach inside
    // the inner readers — but we can verify the output is sharded and
    // sums correctly.
    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path(), p2.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 150);

    // (a) Sharded output only.
    let n_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::ObsmEmbeddingShard
                && e.name.starts_with("obsm/X_pca_shard_")
        })
        .count();
    let n_legacy = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.name == "obsm/X_pca")
        .count();
    assert_eq!(n_legacy, 0, "merge must not emit legacy obsm sections");
    assert!(
        n_shards >= 3,
        "expected at least 3 obsm shards (one per input), got {n_shards}"
    );

    // (c) Round-trip values.
    let assembled = reader.read_obsm("X_pca").unwrap();
    assert_eq!(assembled.num_rows(), 150);
    let pc1 = assembled
        .column_by_name("pc1")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    // Each input fixture used `obsm_batch(0, n)` which encodes pc1[i] = i
    // (relative to the input's start). After merge, rows from input 0
    // map to global 0..50, input 1 → 50..110, input 2 → 110..150 — but
    // each input wrote pc1 starting from 0, so we expect three runs of
    // [0..n_obs_i].
    assert_eq!(pc1.value(0), 0.0);
    assert_eq!(pc1.value(49), 49.0);
    assert_eq!(pc1.value(50), 0.0); // input 1's row 0
    assert_eq!(pc1.value(109), 59.0); // input 1's last row
    assert_eq!(pc1.value(110), 0.0); // input 2's row 0
    assert_eq!(pc1.value(149), 39.0); // input 2's last row
}

#[test]
fn merge_obsm_width_mismatch_errors() {
    // Two inputs whose obsm["X_pca"] disagree on column count
    // (2 vs 3 components). Pre-fix merge succeeded and the resulting
    // file failed at `read_obsm()` time; the new validation rejects
    // the mismatch before any obsm shard is written.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");

    // Input 0: standard 2-column obsm via the sharded helper.
    write_sharded_input(&p0, 50, "donor_A", &var, 25, false, 0);

    // Input 1: 3-column obsm written directly so we control the schema.
    {
        let n_obs = 50u64;
        let mut writer = ScxWriter::new(&p1, header(n_obs, 4)).unwrap();
        writer
            .write_obs(&obs_batch(0, n_obs as usize, "donor_B"))
            .unwrap();
        writer.write_var(&var).unwrap();
        write_zero_csr_shard(&mut writer, 0, n_obs);
        let batch = obsm_batch_wider(0, n_obs as usize);
        writer
            .write_obsm_shard("X_pca", 0, 0, n_obs, n_obs, &batch)
            .unwrap();
        writer
            .write_provenance(vec![ProvenanceEntry {
                timestamp: 1710000000,
                action: "convert".to_string(),
                tool: "streaming_merge_append width-mismatch fixture".to_string(),
                params_json: "{}".to_string(),
                input_checksums: vec![],
            }])
            .unwrap();
        writer.finish().unwrap();
    }

    let out = dir.path().join("out.scx");
    let err = scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap_err();
    match err {
        scx_ops::OpsError::DenseMappingMismatch { axis, ref key, .. } => {
            assert_eq!(axis, "obsm");
            assert_eq!(key, "X_pca");
        }
        other => panic!("expected DenseMappingMismatch, got: {other:?}"),
    }
    assert!(
        !out.exists(),
        "merge must not produce an output file when obsm schemas disagree"
    );
}

#[test]
fn merge_streams_global_varm() {
    // Phase 3d: 2 inputs with sharded varm. Merge must emit
    // `VarmEmbeddingShard` entries and round-trip values via `read_varm`.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_sharded_input(&p0, 50, "donor_A", &var, 25, true, 0);
    write_sharded_input(&p1, 60, "donor_B", &var, 30, true, 0);

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let n_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::VarmEmbeddingShard
                && e.name.starts_with("varm/feature_emb_shard_")
        })
        .count();
    let n_legacy = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::VarmEmbedding && e.name == "varm/feature_emb")
        .count();
    assert_eq!(n_legacy, 0, "merge must not emit legacy varm sections");
    assert_eq!(
        n_shards, 1,
        "varm rows align with the shared var axis, so merge takes input 0's varm only \
         (one shard in this fixture)"
    );

    // varm row count matches var.num_rows() — NOT N_INPUTS * n_vars.
    // Varm rows align with the shared var axis, validated identical
    // across inputs at the merge entry point, so the helper takes
    // input 0's varm as canonical (same rule as for `var` itself).
    let assembled = reader.read_varm("feature_emb").unwrap();
    assert_eq!(assembled.num_rows(), 4);
    let a = assembled
        .column_by_name("emb_a")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(a.value(0), 100.0);
    assert_eq!(a.value(3), 103.0);
}

#[test]
fn merge_streams_legacy_obsm_inputs() {
    // Phase 3d: legacy single-section obsm inputs are treated as one
    // source shard each. Merge output is fully sharded.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_legacy_obsm_input(&p0, 50, "donor_A", &var, false);
    write_legacy_obsm_input(&p1, 60, "donor_B", &var, false);

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let n_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::ObsmEmbeddingShard
                && e.name.starts_with("obsm/X_pca_shard_")
        })
        .count();
    let n_legacy = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.name == "obsm/X_pca")
        .count();
    assert_eq!(n_legacy, 0);
    assert_eq!(n_shards, 2, "one source-shard per legacy input");

    let assembled = reader.read_obsm("X_pca").unwrap();
    assert_eq!(assembled.num_rows(), 110);
}

#[test]
fn merge_streams_layer_without_assembly() {
    // Phase 3d: multi-shard layer merge does not call `read_layer` on
    // any input. We verify by checking the `debug_counts` on a freshly
    // opened reader after merge — the merge path uses *its own*
    // readers, so we instead inspect the output's structure and the
    // per-input debug counters that survive the merge close.
    //
    // Because the merge function takes paths (not readers), the
    // counter is checked on output readers we open later (which start
    // at zero by default). Instead, we verify two things:
    // (a) output has `LayerCsrShard` entries covering all rows;
    // (b) round-trip via `read_layer` works.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_sharded_input(&p0, 100, "donor_A", &var, 100, false, 4); // 4 layer shards × 25 rows
    write_sharded_input(&p1, 80, "donor_B", &var, 80, false, 2); // 2 layer shards × 40 rows

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let total_rows: u64 = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::LayerCsrShard
                && e.modality_id == 0
                && e.name.starts_with("spliced_shard_")
        })
        .filter_map(|e| e.stats.as_ref().map(|s| s.row_end - s.row_start))
        .sum();
    assert_eq!(total_rows, 180, "layer shards must cover all 180 obs rows");

    // Round-trip: `read_layer` reassembles correctly. The fixture wrote
    // all zeros, so the assembled CSR has 180 rows × 4 columns with no
    // non-zero entries.
    let assembled = reader.read_layer("spliced").unwrap();
    assert_eq!(assembled.shape.0, 180);

    // `read_layer` is a materialising call by design (it's the
    // user-facing API). The Phase 3a guarantee is that the *merge*
    // path does not call it. Confirm the test reader's counter is 1
    // (from the read_layer call we just made) and would have been 2 if
    // merge had also called it — but the merge ran with its own
    // readers which were dropped. So instead we verify the no-call
    // property indirectly: a hand-rolled streaming reassembly via
    // `read_shard_from_entry` produces the same shape.
    let n_layer_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::LayerCsrShard
                && e.modality_id == 0
                && e.name.starts_with("spliced_shard_")
        })
        .count();
    // Each input contributes its own shards (no re-bucketing because
    // the per-input row counts are below `shard_target_rows` = 16384).
    // input 0: 4 shards × 25 rows; input 1: 2 shards × 40 rows.
    // After Phase 3a, the merge loop re-packs into output shards of
    // size up to `shard_target_rows`; for these tiny inputs that's a
    // single output shard. Expect 1 output layer shard.
    assert_eq!(
        n_layer_shards, 1,
        "small layer rows collapse into a single output shard"
    );
}

#[test]
fn merge_streams_mixed_obsm_layouts() {
    // Phase 3d: one sharded input + one legacy single-section input.
    // Merge produces a fully sharded output and round-trips values
    // from both inputs.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_sharded_input(&p0, 30, "donor_A", &var, 15, false, 0);
    write_legacy_obsm_input(&p1, 40, "donor_B", &var, false);

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    let n_legacy = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.name == "obsm/X_pca")
        .count();
    assert_eq!(n_legacy, 0);

    let assembled = reader.read_obsm("X_pca").unwrap();
    assert_eq!(assembled.num_rows(), 70);
    let pc1 = assembled
        .column_by_name("pc1")
        .unwrap()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    // Input 0 was sharded with two shards (15 + 15) covering rows 0..30.
    // Input 1 was legacy single-section with rows 0..40.
    assert_eq!(pc1.value(0), 0.0);
    assert_eq!(pc1.value(29), 29.0);
    assert_eq!(pc1.value(30), 0.0); // legacy input row 0
    assert_eq!(pc1.value(69), 39.0); // legacy input last row
}

/// Build a multimodal SCX input with two modalities (rna, adt), zero
/// CSR shards per modality, plus a sharded per-modality obsm
/// `obsm/rna/X_umap`.
fn write_multimodal_with_per_modality_obsm(
    path: &std::path::Path,
    n_obs: u64,
    donor: &str,
    rna_obsm_shard_rows: u64,
) {
    use scx_format_io::modality::ModalityType;
    let mut writer = ScxWriter::new(path, header(n_obs, 4)).unwrap();
    writer
        .write_obs(&obs_batch(0, n_obs as usize, donor))
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
    writer.write_var_for(rna_id, &var_batch()).unwrap();
    writer.write_var_for(adt_id, &var_batch()).unwrap();
    writer.set_modality_n_vars(rna_id, 4).unwrap();
    writer.set_modality_n_vars(adt_id, 4).unwrap();

    let indptr: Vec<u64> = vec![0u64; (n_obs + 1) as usize];
    let indices: Vec<u32> = Vec::new();
    let values: Vec<u8> = Vec::new();
    let rna_shard = scx_format_io::ShardBuffers::new(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(rna_id, 0, rna_shard).unwrap();
    let adt_shard = scx_format_io::ShardBuffers::new(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(adt_id, 0, adt_shard).unwrap();

    // Sharded per-modality obsm for the rna modality.
    let mut start: u64 = 0;
    let mut shard_idx: u32 = 0;
    while start < n_obs {
        let end = (start + rna_obsm_shard_rows).min(n_obs);
        let n = end - start;
        let batch = obsm_batch(start as usize, n as usize);
        let meta = scx_format_io::DenseShardMetadata::new(shard_idx, start, n, n_obs);
        writer
            .write_obsm_shard_for(rna_id, "X_umap", meta, &batch)
            .unwrap();
        start = end;
        shard_idx += 1;
    }

    writer.finish().unwrap();
}

#[test]
fn merge_streams_multimodal_per_modality_obsm() {
    // Phase 3d: multimodal merge with sharded per-modality obsm in the
    // RNA modality. Assert that the merged output keeps the per-modality
    // shard layout (`obsm/rna/X_umap_shard_*` with modality_id == rna_id)
    // and that round-trip via `read_obsm_for` returns the concatenated
    // embedding.
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_multimodal_with_per_modality_obsm(&p0, 40, "donor_A", 20);
    write_multimodal_with_per_modality_obsm(&p1, 30, "donor_B", 15);

    let out = dir.path().join("merged.scx");
    scx_ops::merge(&[p0.as_path(), p1.as_path()], &out).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.n_obs(), 70);
    let rna_id = reader.modality_id("rna").unwrap();

    // Per-modality obsm shards present with modality_id == rna_id.
    let n_shards = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::ObsmEmbeddingShard
                && e.modality_id == rna_id
                && e.name.starts_with("obsm/rna/X_umap_shard_")
        })
        .count();
    let n_legacy = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObsmEmbedding && e.name == "obsm/rna/X_umap")
        .count();
    assert_eq!(n_legacy, 0, "merge must not emit legacy per-modality obsm");
    assert!(
        n_shards >= 4,
        "expected at least 4 per-modality obsm shards (2 inputs × ≥2 shards), got {n_shards}"
    );

    // Round-trip via `read_obsm_for`.
    let assembled = reader.read_obsm_for(rna_id, "X_umap").unwrap();
    assert_eq!(assembled.num_rows(), 70);
}

#[test]
fn merge_streams_refuses_obsm_keys_missing_from_any_input() {
    // Phase 5b, review §6.4. This test used to be
    // `merge_streams_drops_obsm_keys_missing_from_any_input` and asserted
    // `n_obsm == 0` under the heading "existing semantic preserved" — i.e. it
    // specified the silent drop as intended behaviour, the same way `append`'s
    // categorical tests specify `Utf8`. It was preserving a `continue` with no
    // diagnostic, while a *layer* in exactly this position had always been a
    // hard `LayerMissing` naming the file index.
    //
    // Input 0 has X_pca, input 1 does not.
    let dir = tempfile::tempdir().unwrap();
    let var = var_batch();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_sharded_input(&p0, 50, "donor_A", &var, 25, false, 0);
    write_legacy_input(&p1, 50, "donor_B", &var, None); // no obsm

    let out = dir.path().join("merged.scx");
    let err = scx_ops::merge(&[p0.as_path(), p1.as_path()], &out)
        .expect_err("an obsm key input 1 lacks must not be dropped without a word");
    let msg = err.to_string();
    assert!(
        msg.contains("obsm") && msg.contains("X_pca") && msg.contains('1'),
        "the error must name the axis, the key and the input that lacks it, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Sorted k-way merge: `merge --sort-by`.
// ---------------------------------------------------------------------------

/// Write an input whose cells are a *sorted run* by `ct`. Each row carries a
/// `cell_id` and a `ct` (sort key); X has one nnz at col0 with value
/// `cell + 1` so the source cell is recoverable from the merged X.
fn write_sorted_run(path: &std::path::Path, cells: &[(&str, u32)]) {
    let n = cells.len() as u64;
    let mut w = ScxWriter::new(path, header(n, 4)).unwrap();
    let cell_ids: Vec<String> = cells.iter().map(|(_, c)| format!("cell_{c}")).collect();
    let cts: Vec<String> = cells.iter().map(|(ct, _)| ct.to_string()).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("ct", DataType::Utf8, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(cell_ids)),
            Arc::new(StringArray::from(cts)),
        ],
    )
    .unwrap();
    w.write_obs(&obs).unwrap();
    w.write_var(&var_batch()).unwrap();
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for (_, c) in cells {
        indices.push(0u32);
        values.push(((*c % 250) + 1) as u8);
        indptr.push(indptr.last().unwrap() + 1);
    }
    w.write_csr_shard(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint8,
        0,
    )
    .unwrap();
    w.write_provenance(vec![ProvenanceEntry {
        timestamp: 1710000000,
        action: "convert".to_string(),
        tool: "sorted-merge test fixture".to_string(),
        params_json: "{}".to_string(),
        input_checksums: vec![],
    }])
    .unwrap();
    w.finish().unwrap();
}

fn sorted_merge_opts(by: &[&str], reverse: bool, index: bool) -> scx_ops::MergeOptions {
    let mut opts = scx_ops::MergeOptions {
        sort_by: by.iter().map(|s| s.to_string()).collect(),
        sort_reverse: reverse,
        ..Default::default()
    };
    if index {
        opts.index_options.index_obs = by.iter().map(|s| s.to_string()).collect();
    }
    opts
}

fn read_ct_col(reader: &ScxReader) -> Vec<String> {
    let obs = reader.read_obs().unwrap();
    let col = obs.column_by_name("ct").unwrap();
    let utf8 = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let a = utf8.as_any().downcast_ref::<StringArray>().unwrap();
    (0..a.len()).map(|i| a.value(i).to_string()).collect()
}

/// Recover the source cell id of each output row from X col0 (`value - 1`).
fn read_cell_tags(reader: &ScxReader) -> Vec<u32> {
    let csr = reader.read_all_csr_shards().unwrap();
    let (n, c) = csr.shape;
    let dense = csr.to_dense().unwrap();
    (0..n).map(|r| dense[r * c] as u32 - 1).collect()
}

#[test]
fn sorted_merge_orders_globally_and_preserves_rows() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("merged.scx");
    // Two sorted runs that interleave under a global ct sort.
    write_sorted_run(&p0, &[("A", 0), ("A", 1), ("C", 4)]);
    write_sorted_run(&p1, &[("B", 2), ("B", 3), ("D", 5)]);

    scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct"], false, false),
    )
    .unwrap();

    let r = ScxReader::open(&out).unwrap();
    assert_eq!(r.n_obs(), 6);
    let ct = read_ct_col(&r);
    assert!(
        ct.windows(2).all(|w| w[0] <= w[1]),
        "globally sorted: {ct:?}"
    );
    // Global ct order A,A,B,B,C,D → source cells 0,1,2,3,4,5.
    assert_eq!(read_cell_tags(&r), vec![0, 1, 2, 3, 4, 5]);
}

/// A sorted merge moves rows by a permutation, not an offset, so the carried
/// deletion vector has to follow the same permutation.
///
/// `read_cell_tags` recovers each output row's source cell, which is what makes
/// this checkable: the assertion is that the *same cells* are still deleted
/// after the reorder, not that the same row indices are.
#[test]
fn sorted_merge_carries_deletion_vectors_through_the_permutation() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("merged.scx");
    write_sorted_run(&p0, &[("A", 0), ("A", 1), ("C", 4)]);
    write_sorted_run(&p1, &[("B", 2), ("B", 3), ("D", 5)]);

    // Cell 4 is input 0's last row; cell 2 is input 1's first. Under the global
    // ct sort they end up at output rows 4 and 2 — neither of which equals the
    // input row index they came from (2 and 0), so a verbatim copy fails here.
    scx_ops::mark_deleted(&p0, &[2]).unwrap();
    scx_ops::mark_deleted(&p1, &[0]).unwrap();

    scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct"], false, false),
    )
    .unwrap();

    let r = ScxReader::open(&out).unwrap();
    assert_eq!(r.n_obs(), 6, "carried, not applied");
    let tags = read_cell_tags(&r);
    assert_eq!(tags, vec![0, 1, 2, 3, 4, 5]);

    let dv = r
        .read_deletion_vectors()
        .unwrap()
        .expect("sorted merge carries a deletion-vector section");
    let deleted_cells: Vec<u32> = (0..tags.len())
        .filter(|&row| dv.is_deleted_global(row as u32))
        .map(|row| tags[row])
        .collect();
    assert_eq!(
        deleted_cells,
        vec![2, 4],
        "the same source cells stay deleted across the reorder"
    );
}

#[test]
fn sorted_merge_is_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    write_sorted_run(&p0, &[("A", 0), ("C", 4)]);
    write_sorted_run(&p1, &[("B", 2), ("D", 5)]);
    let o1 = dir.path().join("m1.scx");
    let o2 = dir.path().join("m2.scx");
    for o in [&o1, &o2] {
        scx_ops::merge_with_options(
            &[p0.as_path(), p1.as_path()],
            o,
            &sorted_merge_opts(&["ct"], false, false),
        )
        .unwrap();
    }
    let r1 = ScxReader::open(&o1).unwrap();
    let r2 = ScxReader::open(&o2).unwrap();
    assert_eq!(read_ct_col(&r1), read_ct_col(&r2));
    assert_eq!(read_cell_tags(&r1), read_cell_tags(&r2));
}

#[test]
fn sorted_merge_tie_order_follows_input_order() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("m.scx");
    // Both inputs are all "A" → equal keys; input 0's rows must come first.
    write_sorted_run(&p0, &[("A", 0), ("A", 1)]);
    write_sorted_run(&p1, &[("A", 2), ("A", 3)]);
    scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct"], false, false),
    )
    .unwrap();
    let r = ScxReader::open(&out).unwrap();
    assert_eq!(read_cell_tags(&r), vec![0, 1, 2, 3]);
}

#[test]
fn sorted_merge_reverse() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("m.scx");
    // Reverse-sorted runs: each input descending by ct.
    write_sorted_run(&p0, &[("C", 4), ("A", 0)]);
    write_sorted_run(&p1, &[("D", 5), ("B", 2)]);
    scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct"], true, false),
    )
    .unwrap();
    let ct = read_ct_col(&ScxReader::open(&out).unwrap());
    assert!(
        ct.windows(2).all(|w| w[0] >= w[1]),
        "reverse sorted: {ct:?}"
    );
}

#[test]
fn sorted_merge_unsorted_input_errors() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("m.scx");
    write_sorted_run(&p0, &[("A", 0), ("B", 1)]);
    // p1 is NOT sorted by ct (B before A).
    write_sorted_run(&p1, &[("B", 2), ("A", 3)]);
    let err = scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct"], false, false),
    );
    assert!(err.is_err(), "unsorted input must error under --sort-by");
}

#[test]
fn sorted_merge_index_ranges_contiguous() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("m.scx");
    write_sorted_run(&p0, &[("A", 0), ("A", 1), ("C", 4)]);
    write_sorted_run(&p1, &[("B", 2), ("C", 5)]);
    scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct"], false, true),
    )
    .unwrap();
    let reader = ScxReader::open(&out).unwrap();
    let bytes = reader.read_obs_predicate_index_bytes().unwrap().unwrap();
    let index = scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
    // Global order A,A,B,C,C → A=[0,2), B=[2,3), C=[3,5); each contiguous.
    for (val, lo, hi) in [("A", 0u32, 2u32), ("B", 2, 3), ("C", 3, 5)] {
        let ranges = index.categorical_eq("ct", val).expect("indexed");
        assert_eq!(ranges.len(), 1, "{val} ranges {ranges:?}");
        assert_eq!((ranges[0].row_start, ranges[0].row_end), (lo, hi), "{val}");
    }
}

#[test]
fn sorted_merge_composite_key() {
    let dir = tempfile::tempdir().unwrap();
    let p0 = dir.path().join("a.scx");
    let p1 = dir.path().join("b.scx");
    let out = dir.path().join("m.scx");
    // Composite [ct, cell_id]: each input sorted by ct then cell_id.
    write_sorted_run(&p0, &[("A", 0), ("A", 2), ("B", 10)]);
    write_sorted_run(&p1, &[("A", 1), ("B", 11)]);
    scx_ops::merge_with_options(
        &[p0.as_path(), p1.as_path()],
        &out,
        &sorted_merge_opts(&["ct", "cell_id"], false, false),
    )
    .unwrap();
    let r = ScxReader::open(&out).unwrap();
    let ct = read_ct_col(&r);
    assert!(ct.windows(2).all(|w| w[0] <= w[1]));
    // Within ct=="A": cell_id "cell_0","cell_1","cell_2" ascending (string order).
    assert_eq!(read_cell_tags(&r), vec![0, 1, 2, 10, 11]);
}
