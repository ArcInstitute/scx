//! Guards against silent `adata.raw` loss in raw-unaware ops.
//!
//! `compact` and `append` do not yet preserve / extend the raw section
//! family. Until they do, they must not silently drop raw (compact) or
//! leave it misaligned (append) — `compact` warns and produces a
//! consistent raw-free file; `append` hard-errors. These tests lock that
//! behavior in.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_ops::AppendOptions;
use tempfile::TempDir;

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
}

fn str_batch(field: &str, prefix: &str, n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("{prefix}{i}")).collect();
    let schema = Schema::new(vec![Field::new(field, DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Two nonzeros per row, sorted columns, u8 values.
fn shard(n_rows: usize, n_cols: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        let a = (row % n_cols) as u32;
        let b = (n_cols - 1 - (row % n_cols)) as u32;
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        if lo == hi {
            indices.push(lo);
            values.push((row % 250 + 1) as u8);
        } else {
            indices.push(lo);
            values.push((row % 250 + 1) as u8);
            indices.push(hi);
            values.push((row % 250 + 2) as u8);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Write a single-modality SCX file with X, obs, var, and (when
/// `raw_n_vars` is Some) a raw section family.
fn write_file(
    dir: &TempDir,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    raw_n_vars: Option<usize>,
) -> PathBuf {
    let path = dir.path().join(name);
    let mut writer = ScxWriter::new(&path, sample_header(n_obs as u64, n_vars as u64)).unwrap();
    writer
        .write_obs(&str_batch("cell_id", "cell_", n_obs))
        .unwrap();
    writer
        .write_var(&str_batch("gene_id", "gene_", n_vars))
        .unwrap();

    let (ip, ix, val) = shard(n_obs, n_vars);
    writer
        .write_csr_shard(&ip, &ix, &val, CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();

    if let Some(rn) = raw_n_vars {
        let (rip, rix, rval) = shard(n_obs, rn);
        writer.set_raw_n_vars(rn as u64);
        writer
            .write_raw_csr_shard(&rip, &rix, &rval, CodecId::None, ValueEncoding::Uint8, 0)
            .unwrap();
        writer
            .write_raw_var(&str_batch("gene_id", "raw_gene_", rn))
            .unwrap();
    }

    writer.finish().unwrap();
    path
}

#[test]
fn append_to_raw_file_hard_errors() {
    let dir = TempDir::new().unwrap();
    let target = write_file(&dir, "target.scx", 8, 10, Some(15));

    // Sanity: the target really carries raw.
    assert!(ScxReader::open(&target).unwrap().has_raw());

    let new_obs = str_batch("cell_id", "newcell_", 4);
    let (ip, ix, val) = shard(4, 10);
    let err = scx_ops::append(
        &target,
        &new_obs,
        &ip,
        &ix,
        &val,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("does not yet support adata.raw"),
        "expected RawUnsupported, got: {msg}"
    );

    // The target is untouched (the guard fires before any mutation).
    let reader = ScxReader::open(&target).unwrap();
    assert!(reader.has_raw());
    assert_eq!(reader.n_obs(), 8);
}

#[test]
fn compact_drops_raw_but_keeps_x() {
    let dir = TempDir::new().unwrap();
    let input = write_file(&dir, "in.scx", 8, 10, Some(15));
    let output = dir.path().join("out.scx");

    // X data for the post-compact comparison.
    let x_before = ScxReader::open(&input)
        .unwrap()
        .read_all_csr_shards()
        .unwrap();

    scx_ops::compact(&input, &output).unwrap();

    let reader = ScxReader::open(&output).unwrap();
    // Raw is dropped (warned, not silent) → consistent raw-free output.
    assert!(!reader.has_raw(), "compact output must not claim has_raw");
    assert!(reader.raw_n_vars().is_none());
    // The main matrix survives intact.
    let x_after = reader.read_all_csr_shards().unwrap();
    assert_eq!(x_before.shape, x_after.shape);
    assert_eq!(x_before.indptr, x_after.indptr);
    assert_eq!(x_before.indices, x_after.indices);
    assert_eq!(x_before.data, x_after.data);
}
