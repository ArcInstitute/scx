// Shared test helpers for scx-cli tests.
//
// Provides sample_header, sample_obs, sample_var, and write_test_file
// used across build_csc, upgrade, and subset test modules.

#![cfg(test)]

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION, MAGIC};
use scx_format_io::writer::ScxWriter;
use std::sync::Arc;

/// Create a sample FileHeader for testing.
pub fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
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
        shard_target_rows: 10000,
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

/// Create a sample obs RecordBatch with cell_id and cell_type columns.
pub fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

/// Create a sample var RecordBatch with a gene_id column.
pub fn sample_var(n: usize) -> arrow::array::RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Write a test SCX file with CSR shards and obs metadata for filtering.
pub fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
    let path = dir.path().join("test.scx");
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Build CSR data: each row has 2 nnz at deterministic column positions.
    // SCX expects ascending column indices per row (see
    // `scx_engine::project_csr_row`'s precondition); the modulo arithmetic
    // can produce `col0 > col1` when `row * 2 + 1` wraps past `n_vars`, so
    // sort the (col, val) pair before writing.
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        let val0 = ((row + 1) % 256) as u8;
        let val1 = ((row + 2) % 256) as u8;
        let ((c0, v0), (c1, v1)) = if col0 <= col1 {
            ((col0, val0), (col1, val1))
        } else {
            ((col1, val1), (col0, val0))
        };
        indices.push(c0 as u32);
        indices.push(c1 as u32);
        values.push(v0);
        values.push(v1);
        indptr.push(indptr.last().unwrap() + 2);
    }

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
