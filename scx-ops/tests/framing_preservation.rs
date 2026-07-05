//! F-d regression suite: mutating ops must preserve row-group framing.
//!
//! A framed (v4 / shard-format-v2) input rewritten by `compact` / `sort` must
//! stay a valid v4 file with framed shards — not silently downgrade to unframed
//! v3. An unframed (v3) input must stay v3 (no over-framing). append's framing
//! preservation is covered in `streaming_merge_append.rs`.

use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::encoder::FramingConfig;
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

const N_OBS: usize = 40;
const N_VARS: usize = 50_000;

fn obs_batch() -> RecordBatch {
    // cell_ids are zero-padded and already ascending, so an ascending sort by
    // `cell_id` preserves row order (lets the sort test assert exact parity).
    let cell_ids: Vec<String> = (0..N_OBS).map(|i| format!("cell_{i:07}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(cell_ids))],
    )
    .unwrap()
}

fn var_batch() -> RecordBatch {
    let gene_ids: Vec<String> = (0..N_VARS).map(|i| format!("g{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(gene_ids))],
    )
    .unwrap()
}

/// Deterministic sparse CSR (u8 counts) for the fixture.
fn build_csr() -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let nnz_per_row = 8usize;
    let mut indptr = vec![0u64];
    let mut indices: Vec<u32> = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for r in 0..N_OBS {
        let mut col = 0u32;
        for k in 0..nnz_per_row {
            col += 1 + ((r * 13 + k * 7) % 97) as u32;
            if col as usize >= N_VARS {
                break;
            }
            indices.push(col);
            values.push(1 + ((r + k) % 5) as u8);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Write a single-shard input. `framing = Some(G)` produces a v4/shard-v2 file;
/// `None` the legacy unframed v3.
fn write_input(path: &std::path::Path, framing: Option<u32>) {
    let (indptr, indices, values) = build_csr();
    let mut h = FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, 16_384, 0, 0);
    if framing.is_some() {
        h.format_version = CURRENT_FORMAT_VERSION;
    }
    let mut writer = ScxWriter::new(path, h).unwrap();
    if let Some(g) = framing {
        writer.set_framing(Some(FramingConfig {
            row_group_rows: g,
            ..Default::default()
        }));
    }
    writer.write_obs(&obs_batch()).unwrap();
    writer.write_var(&var_batch()).unwrap();
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
            timestamp: 1_710_000_000,
            action: "convert".to_string(),
            tool: "framing_preservation fixture".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();
    writer.finish().unwrap();
}

/// Concatenate all CSR shards of a file into one dense-ish (indptr, indices,
/// data) triple for order-independent decoded comparison.
fn read_all_rows(reader: &ScxReader) -> Vec<Vec<(i32, f32)>> {
    let mut rows: Vec<Vec<(i32, f32)>> = Vec::new();
    let n_shards = reader.catalog().shards_sorted().len();
    for i in 0..n_shards {
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        for r in 0..indptr.len() - 1 {
            let s = indptr[r] as usize;
            let e = indptr[r + 1] as usize;
            rows.push(indices[s..e].iter().copied().zip(data[s..e].iter().copied()).collect());
        }
    }
    rows
}

fn assert_all_shards_framed(reader: &ScxReader) {
    for entry in &reader.catalog().shards_sorted() {
        let sh = reader.read_shard_header(entry).unwrap();
        assert!(
            sh.shard_format_version > 1,
            "shard '{}' must be framed (shard-v2) in a v4 output, got v{}",
            entry.name,
            sh.shard_format_version
        );
    }
}

#[test]
fn compact_preserves_framing_v4() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.scx");
    let output = dir.path().join("out.scx");
    write_input(&input, Some(8));

    let src = ScxReader::open(&input).unwrap();
    assert_eq!(src.header().format_version, CURRENT_FORMAT_VERSION);
    let expected = read_all_rows(&src);

    scx_ops::compact(&input, &output).unwrap();

    let out = ScxReader::open(&output).unwrap();
    assert_eq!(
        out.header().format_version,
        CURRENT_FORMAT_VERSION,
        "compacting a framed file must keep it v4"
    );
    assert_all_shards_framed(&out);
    assert_eq!(read_all_rows(&out), expected, "compact decoded parity");
}

#[test]
fn compact_unframed_stays_v3() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in_v3.scx");
    let output = dir.path().join("out_v3.scx");
    write_input(&input, None);

    scx_ops::compact(&input, &output).unwrap();

    let out = ScxReader::open(&output).unwrap();
    assert!(
        out.header().format_version < CURRENT_FORMAT_VERSION,
        "compacting an unframed file must not over-frame to v4 (got v{})",
        out.header().format_version
    );
}

#[test]
fn sort_preserves_framing_v4() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in.scx");
    let output = dir.path().join("out.scx");
    write_input(&input, Some(8));

    let src = ScxReader::open(&input).unwrap();
    // Ascending sort by the already-ascending cell_id preserves row order.
    let expected = read_all_rows(&src);

    let opts = scx_ops::SortOptions {
        by: vec!["cell_id".to_string()],
        ..Default::default()
    };
    scx_ops::sort(&input, &output, &opts).unwrap();

    let out = ScxReader::open(&output).unwrap();
    assert_eq!(
        out.header().format_version,
        CURRENT_FORMAT_VERSION,
        "sorting a framed file must keep it v4"
    );
    assert_all_shards_framed(&out);
    assert_eq!(read_all_rows(&out), expected, "sort decoded parity (order preserved)");
}

#[test]
fn sort_unframed_stays_v3() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("in_v3.scx");
    let output = dir.path().join("out_v3.scx");
    write_input(&input, None);

    let opts = scx_ops::SortOptions {
        by: vec!["cell_id".to_string()],
        ..Default::default()
    };
    scx_ops::sort(&input, &output, &opts).unwrap();

    let out = ScxReader::open(&output).unwrap();
    assert!(
        out.header().format_version < CURRENT_FORMAT_VERSION,
        "sorting an unframed file must not over-frame to v4 (got v{})",
        out.header().format_version
    );
}
