// Shared test helpers for scx-cli tests.
//
// Provides sample_header, sample_obs, sample_var, and write_test_file
// used across build_csc, upgrade, and subset test modules.

#![cfg(test)]

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use std::sync::Arc;

/// Create a sample FileHeader for testing.
pub fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 10000, 0, 0)
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

/// Write a 2-modality test SCX file (rna: `rna_vars`, adt: `adt_vars`) over a
/// shared `n_obs` obs axis with a global `cell_type` column. Each modality has
/// one CSR shard covering `[0, n_obs)`; row `r` in modality m expresses gene
/// `r % m_vars`. Returns the file path. Used by the `--modality` query tests.
pub fn write_multimodal_test_file(
    dir: &tempfile::TempDir,
    n_obs: usize,
    rna_vars: usize,
    adt_vars: usize,
) -> std::path::PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join("mm.scx");
    let header = sample_header(n_obs as u64, rna_vars.max(adt_vars) as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();

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
    writer.write_var_for(rna_id, &sample_var(rna_vars)).unwrap();
    writer.write_var_for(adt_id, &sample_var(adt_vars)).unwrap();
    writer.set_modality_n_vars(rna_id, rna_vars as u64).unwrap();
    writer.set_modality_n_vars(adt_id, adt_vars as u64).unwrap();

    for (id, m_vars) in [(rna_id, rna_vars), (adt_id, adt_vars)] {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..n_obs {
            indices.push((r % m_vars) as u32);
            values.push(((r + 1) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(id, 0, shard).unwrap();
    }

    writer.finish().unwrap();
    path
}
