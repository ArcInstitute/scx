//! Shared test scaffolding for CSC kernel tests.
//!
//! Builds tiny SCX files with both CSR and CSC shards from a dense
//! reference matrix so kernel parity tests can compare CSR-path vs
//! CSC-path numerics on identical data.

#![cfg(test)]

use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::ScxWriter;
use std::path::PathBuf;

use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

pub(crate) fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: 1,
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
        reserved: [0u8; 132],
    }
}

pub(crate) fn sample_obs(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

pub(crate) fn sample_var(n: usize) -> RecordBatch {
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

/// Encode a u8 dense matrix as CSR arrays (indptr, indices, values).
fn dense_to_csr(dense: &[u8], n_obs: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr: Vec<u64> = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..n_obs {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v != 0 {
                indices.push(c as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Encode a column range of a u8 dense matrix as CSC arrays.
fn dense_to_csc_range(
    dense: &[u8],
    n_obs: usize,
    n_vars: usize,
    col_start: usize,
    col_end: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr: Vec<u64> = vec![0];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for c in col_start..col_end {
        for r in 0..n_obs {
            let v = dense[r * n_vars + c];
            if v != 0 {
                indices.push(r as u32);
                values.push(v);
            }
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Write a small SCX file with one CSR shard and `cols_per_csc_shard`-
/// per-shard CSC sidecar from a u8 dense matrix. Returns the file path.
pub(crate) fn write_csr_csc_test_file(
    dir: &std::path::Path,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    dense: &[u8],
    cols_per_csc_shard: usize,
) -> PathBuf {
    let path = dir.join(format!("{name}.scx"));
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let (csr_indptr, csr_indices, csr_values) = dense_to_csr(dense, n_obs, n_vars);
    writer
        .write_csr_shard(
            &csr_indptr,
            &csr_indices,
            &csr_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    let mut col_start = 0usize;
    while col_start < n_vars {
        let col_end = (col_start + cols_per_csc_shard).min(n_vars);
        let (ip, ix, vb) = dense_to_csc_range(dense, n_obs, n_vars, col_start, col_end);
        writer
            .write_csc_shard(
                &ip,
                &ix,
                &vb,
                CodecId::None,
                ValueEncoding::Uint8,
                col_start as u64,
            )
            .unwrap();
        col_start = col_end;
    }

    writer.finish().unwrap();
    path
}

/// Build a deterministic dense reference matrix. Pattern: `((r*7 +
/// c*11) % 200 + 1)` for cells where `(r + c) % 3 == 0`, zero
/// elsewhere.
pub(crate) fn deterministic_dense(n_obs: usize, n_vars: usize) -> Vec<u8> {
    let mut dense = vec![0u8; n_obs * n_vars];
    for r in 0..n_obs {
        for c in 0..n_vars {
            if (r + c) % 3 == 0 {
                dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
            }
        }
    }
    dense
}
