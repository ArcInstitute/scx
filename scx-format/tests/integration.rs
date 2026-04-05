//! Integration tests for the SCX format: ScxWriter → ScxReader round-trips.
//!
//! These tests exercise the full write-read pipeline as external consumers would,
//! covering tasks 18.1–18.8 from Phase1.md.

use arrow::array::{Float32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::dispatch::{CodecId, ValueEncoding};
use scx_format::header::{HEADER_SIZE, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::shard::SHARD_HEADER_SIZE;
use scx_format::{FileHeader, ScxError, ScxReader, ScxWriter};
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: HEADER_SIZE as u16,
        flags: 0,
        n_obs,
        n_vars,
        nnz,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: 16384,
        codec_id: 0,
        index_dtype: if n_vars <= 65535 { 0 } else { 1 },
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

fn sample_obs(n: usize) -> arrow::record_batch::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    arrow::record_batch::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> arrow::record_batch::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    arrow::record_batch::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Build a deterministic shard: n_rows rows, each with 2 nonzeros, u8 values.
fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices.push(col0 as u32);
        indices.push(col1 as u32);
        values.push(((row + 1) % 255 + 1) as u8); // 1..=255, never zero
        values.push(((row + 2) % 255 + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

/// Write a complete test file and return its path.
fn write_test_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    codec: CodecId,
    include_extras: bool,
) -> PathBuf {
    let path = dir.path().join(filename);
    let total_nnz = n_obs * 2;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = n_obs / n_shards;
    for s in 0..n_shards {
        let shard_rows = if s == n_shards - 1 {
            n_obs - rows_per_shard * s
        } else {
            rows_per_shard
        };
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                codec,
                ValueEncoding::Uint8,
                (s * rows_per_shard) as u64,
            )
            .unwrap();
    }

    if include_extras {
        // obsm
        let obsm_schema = Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]);
        let obsm_batch = arrow::record_batch::RecordBatch::try_new(
            Arc::new(obsm_schema),
            vec![
                Arc::new(Float32Array::from(
                    (0..n_obs).map(|i| i as f32).collect::<Vec<_>>(),
                )),
                Arc::new(Float32Array::from(
                    (0..n_obs).map(|i| (i as f32) * 2.0).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();
        writer.write_obsm("X_pca", &obsm_batch).unwrap();

        // uns
        writer
            .write_uns(&serde_json::json!({
                "species": "human",
                "version": 2,
                "nested": {"a": [1, 2, 3]}
            }))
            .unwrap();

        // provenance
        writer
            .write_provenance(vec![ProvenanceEntry {
                timestamp: 1710000000,
                action: "convert".to_string(),
                tool: "scx-cli 0.1.0".to_string(),
                params_json: r#"{"input":"test.h5ad"}"#.to_string(),
                input_checksums: vec![[0xAA; 32]],
            }])
            .unwrap();
    }

    writer.finish().unwrap();
    path
}

// ===========================================================================
// 18.1: ScxWriter → ScxReader → verify equality
// ===========================================================================

#[test]
fn test_18_1_round_trip_codec_none() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "rt_none.scx", 10, 20, 1, CodecId::None, false);

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    assert_eq!(reader.n_vars(), 20);

    let obs = reader.read_obs().unwrap();
    assert_eq!(obs.num_rows(), 10);
    let var = reader.read_var().unwrap();
    assert_eq!(var.num_rows(), 20);

    // Verify CSR round-trip
    let (indptr, indices, data) = reader.read_csr_shard(0).unwrap();
    assert_eq!(indptr.len(), 11); // 10 rows + 1
    assert_eq!(indices.len(), 20); // 10 rows * 2 nnz
    assert_eq!(data.len(), 20);

    // Verify values match what we wrote
    let (expected_indptr, expected_indices, expected_values) = sample_shard_data(10, 20);
    let expected_indptr_i64: Vec<i64> = expected_indptr.iter().map(|&v| v as i64).collect();
    let expected_indices_i32: Vec<i32> = expected_indices.iter().map(|&v| v as i32).collect();
    let expected_data_f32: Vec<f32> = expected_values.iter().map(|&v| v as f32).collect();

    assert_eq!(indptr, expected_indptr_i64);
    assert_eq!(indices, expected_indices_i32);
    assert_eq!(data, expected_data_f32);
}

#[test]
fn test_18_1_round_trip_codec_scx1() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "rt_scx1.scx", 10, 20, 1, CodecId::Scx1, false);

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), 10);
    assert_eq!(reader.n_vars(), 20);

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (10, 20));
    assert_eq!(csr.nnz(), 20);

    // Verify values match
    let (expected_indptr, expected_indices, expected_values) = sample_shard_data(10, 20);
    let expected_indptr_i64: Vec<i64> = expected_indptr.iter().map(|&v| v as i64).collect();
    let expected_indices_i32: Vec<i32> = expected_indices.iter().map(|&v| v as i32).collect();
    let expected_data_f32: Vec<f32> = expected_values.iter().map(|&v| v as f32).collect();

    assert_eq!(csr.indptr, expected_indptr_i64);
    assert_eq!(csr.indices, expected_indices_i32);
    assert_eq!(csr.data, expected_data_f32);
}

// ===========================================================================
// 18.2: Multi-shard file (6 shards) → read_all_csr_shards
// ===========================================================================

#[test]
fn test_18_2_multi_shard_assembly() {
    let dir = tempfile::tempdir().unwrap();
    let n_obs = 30;
    let n_vars = 50;
    let n_shards = 6;
    let path = write_test_file(
        &dir,
        "multi.scx",
        n_obs,
        n_vars,
        n_shards,
        CodecId::None,
        false,
    );

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();

    // Shape
    assert_eq!(csr.shape, (30, 50));
    // Total nnz = 30 rows * 2
    assert_eq!(csr.nnz(), 60);
    // indptr length = 30 + 1
    assert_eq!(csr.indptr.len(), 31);
    // indptr is monotonic
    for w in csr.indptr.windows(2) {
        assert!(w[1] >= w[0], "indptr not monotonic");
    }

    // Individual shard reads combine to same result
    let mut individual_indptr: Vec<i64> = Vec::new();
    let mut individual_indices: Vec<i32> = Vec::new();
    let mut individual_data: Vec<f32> = Vec::new();
    let mut cumulative_nnz: i64 = 0;

    for i in 0..n_shards {
        let (indptr, indices, data) = reader.read_csr_shard(i).unwrap();
        if i == 0 {
            individual_indptr.extend_from_slice(&indptr);
        } else {
            for &v in &indptr[1..] {
                individual_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *indptr.last().unwrap_or(&0);
        individual_indices.extend_from_slice(&indices);
        individual_data.extend_from_slice(&data);
    }

    assert_eq!(csr.indptr, individual_indptr);
    assert_eq!(csr.indices, individual_indices);
    assert_eq!(csr.data, individual_data);
}

#[test]
fn test_18_2_multi_shard_scx1() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "multi_scx1.scx", 30, 50, 6, CodecId::Scx1, false);

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (30, 50));
    assert_eq!(csr.nnz(), 60);
    assert_eq!(csr.indptr.len(), 31);
}

// ===========================================================================
// 18.3: File with layers + obsm + uns → full round-trip
// ===========================================================================

#[test]
fn test_18_3_full_sections_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("full_sections.scx");
    let n_obs = 10;
    let n_vars = 20;
    let total_nnz = n_obs * 2;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // X shard
    let (indptr, indices, values) = sample_shard_data(n_obs, n_vars);
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

    // Layer "raw"
    let (l_indptr, l_indices, l_values) = sample_shard_data(n_obs, n_vars);
    writer
        .write_layer_csr_shard(
            &l_indptr,
            &l_indices,
            &l_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
            "raw",
            0,
        )
        .unwrap();

    // obsm "X_pca"
    let obsm_schema = Schema::new(vec![
        Field::new("pc1", DataType::Float32, false),
        Field::new("pc2", DataType::Float32, false),
        Field::new("pc3", DataType::Float32, false),
    ]);
    let obsm_batch = arrow::record_batch::RecordBatch::try_new(
        Arc::new(obsm_schema),
        vec![
            Arc::new(Float32Array::from(
                (0..n_obs).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n_obs).map(|i| i as f32 * 0.2).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n_obs).map(|i| i as f32 * 0.3).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca", &obsm_batch).unwrap();

    // uns
    let uns_json = serde_json::json!({
        "species": "human",
        "version": 2,
        "nested": {"a": [1, 2, 3]}
    });
    writer.write_uns(&uns_json).unwrap();

    // provenance
    let prov_entries = vec![ProvenanceEntry {
        timestamp: 1710000000,
        action: "convert".to_string(),
        tool: "scx-cli 0.1.0".to_string(),
        params_json: r#"{"input":"test.h5ad"}"#.to_string(),
        input_checksums: vec![[0xAA; 32]],
    }];
    writer.write_provenance(prov_entries).unwrap();

    writer.finish().unwrap();

    // Read back
    let reader = ScxReader::open(&path).unwrap();

    // Layer names
    let layer_names = reader.layer_names();
    assert_eq!(layer_names, vec!["raw"]);

    // Layer CSR matches
    let layer_csr = reader.read_layer("raw").unwrap();
    assert_eq!(layer_csr.shape, (n_obs, n_vars));
    assert_eq!(layer_csr.nnz(), total_nnz);
    let expected_indptr_i64: Vec<i64> = l_indptr.iter().map(|&v| v as i64).collect();
    let expected_indices_i32: Vec<i32> = l_indices.iter().map(|&v| v as i32).collect();
    let expected_data_f32: Vec<f32> = l_values.iter().map(|&v| v as f32).collect();
    assert_eq!(layer_csr.indptr, expected_indptr_i64);
    assert_eq!(layer_csr.indices, expected_indices_i32);
    assert_eq!(layer_csr.data, expected_data_f32);

    // obsm shape and values
    let obsm = reader.read_obsm("X_pca").unwrap();
    assert_eq!(obsm.num_rows(), n_obs);
    assert_eq!(obsm.num_columns(), 3);

    // uns round-trip
    let uns = reader.read_uns().unwrap();
    assert_eq!(uns, uns_json);

    // provenance
    let prov = reader.read_provenance().unwrap();
    assert_eq!(prov.operations.len(), 1);
    assert_eq!(prov.operations[0].action, "convert");
    assert_eq!(prov.operations[0].tool, "scx-cli 0.1.0");
    assert_eq!(prov.operations[0].input_checksums.len(), 1);
    assert_eq!(prov.operations[0].input_checksums[0], [0xAA; 32]);
}

// ===========================================================================
// 18.4: Corrupt file bytes → appropriate error messages
// ===========================================================================

#[test]
fn test_18_4_corrupt_magic() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "bad_magic.scx", 6, 10, 1, CodecId::None, false);

    let mut data = std::fs::read(&path).unwrap();
    data[0] ^= 0xFF; // corrupt first byte of magic
    std::fs::write(&path, &data).unwrap();

    let result = ScxReader::open(&path);
    assert!(
        matches!(&result, Err(ScxError::InvalidMagic)),
        "expected InvalidMagic error"
    );
}

#[test]
fn test_18_4_corrupt_shard_payload() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "bad_shard.scx", 6, 10, 1, CodecId::None, false);

    // Find shard offset from catalog
    let reader = ScxReader::open(&path).unwrap();
    let shards = reader.catalog().shards_sorted();
    assert!(!shards.is_empty());
    let shard_offset = shards[0].offset as usize;
    drop(reader);

    let mut data = std::fs::read(&path).unwrap();
    // Corrupt a byte in the shard payload (after 76-byte header)
    let corrupt_pos = shard_offset + SHARD_HEADER_SIZE + 1;
    data[corrupt_pos] ^= 0xFF;
    std::fs::write(&path, &data).unwrap();

    // Open still succeeds (catalog checksum covers catalog, not shard payloads)
    let reader = ScxReader::open(&path).unwrap();

    // validate() should detect the corruption
    let result = reader.validate();
    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), ScxError::ChecksumMismatch),
        "expected ChecksumMismatch error"
    );
}

#[test]
fn test_18_4_truncated_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "truncated.scx", 6, 10, 1, CodecId::None, false);

    // Truncate to 100 bytes (not even a full header)
    let data = std::fs::read(&path).unwrap();
    std::fs::write(&path, &data[..100]).unwrap();

    let result = ScxReader::open(&path);
    assert!(result.is_err(), "expected error on truncated file");
}

#[test]
fn test_18_4_bad_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_test_file(&dir, "bad_ver.scx", 6, 10, 1, CodecId::None, false);

    let mut data = std::fs::read(&path).unwrap();
    // format_version is at offset 4 (after 4-byte magic), u16 LE
    data[4] = 99;
    data[5] = 0;
    std::fs::write(&path, &data).unwrap();

    let result = ScxReader::open(&path);
    assert!(
        matches!(&result, Err(ScxError::UnsupportedVersion)),
        "expected UnsupportedVersion error"
    );
}

// ===========================================================================
// 18.5: Very sparse matrix (99% zeros) → round-trip
// ===========================================================================

#[test]
fn test_18_5_very_sparse_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sparse.scx");

    let n_obs: usize = 1000;
    let n_vars: usize = 500;

    // ~0.5% density: deterministic pseudo-random with simple LCG
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values_u8 = Vec::new();
    let mut seed: u64 = 42;

    for _row in 0..n_obs {
        // Each row gets 0-5 nonzeros (average ~2.5 → ~0.5% density)
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let nnz_this_row = (seed >> 61) % 6; // 0..5
        let mut cols: Vec<u32> = Vec::new();
        for _ in 0..nnz_this_row {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let col = (seed >> 48) as u32 % n_vars as u32;
            if !cols.contains(&col) {
                cols.push(col);
            }
        }
        cols.sort();
        for &c in &cols {
            indices.push(c);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            values_u8.push(((seed >> 56) as u8).max(1)); // never zero
        }
        indptr.push(indptr.last().unwrap() + cols.len() as u64);
    }

    let total_nnz = *indptr.last().unwrap();
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values_u8,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();

    assert_eq!(csr.shape, (n_obs, n_vars));
    assert_eq!(csr.nnz(), total_nnz as usize);

    let expected_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
    let expected_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
    let expected_data: Vec<f32> = values_u8.iter().map(|&v| v as f32).collect();

    assert_eq!(csr.indptr, expected_indptr);
    assert_eq!(csr.indices, expected_indices);
    assert_eq!(csr.data, expected_data);
}

// ===========================================================================
// 18.6: Single-row, single-column, empty rows edge cases
// ===========================================================================

#[test]
fn test_18_6_single_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single_row.scx");
    let n_obs = 1;
    let n_vars = 1000;
    let nnz = 10;

    let indptr = vec![0u64, nnz as u64];
    let indices: Vec<u32> = (0..nnz).map(|i| (i * 100) as u32).collect();
    let values: Vec<u8> = (1..=nnz as u8).collect();

    let header = sample_header(n_obs as u64, n_vars as u64, nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (1, 1000));
    assert_eq!(csr.nnz(), 10);
    assert_eq!(csr.indptr, vec![0i64, 10]);
}

#[test]
fn test_18_6_single_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("single_col.scx");
    let n_obs = 1000;
    let n_vars = 1;
    let nnz = 500;

    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        if row < nnz {
            indices.push(0u32);
            values.push(((row % 255) + 1) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        } else {
            indptr.push(*indptr.last().unwrap());
        }
    }

    let header = sample_header(n_obs as u64, n_vars as u64, nnz as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (1000, 1));
    assert_eq!(csr.nnz(), 500);
}

#[test]
fn test_18_6_empty_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_rows.scx");
    let n_obs = 10;
    let n_vars = 100;

    // Rows 0,2,4,6,8 have 0 nnz; rows 1,3,5,7,9 have 3 nnz each
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        if row % 2 == 0 {
            // empty row
            indptr.push(*indptr.last().unwrap());
        } else {
            indices.push(0u32);
            indices.push(50u32);
            indices.push(99u32);
            values.push(1u8);
            values.push(2u8);
            values.push(3u8);
            indptr.push(indptr.last().unwrap() + 3);
        }
    }
    let total_nnz = *indptr.last().unwrap();

    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (10, 100));
    assert_eq!(csr.nnz(), total_nnz as usize);

    // Verify empty rows are preserved
    for row in [0, 2, 4, 6, 8] {
        assert_eq!(
            csr.indptr[row],
            csr.indptr[row + 1],
            "row {row} should be empty"
        );
    }
    // Verify non-empty rows have 3 nnz
    for row in [1, 3, 5, 7, 9] {
        assert_eq!(
            csr.indptr[row + 1] - csr.indptr[row],
            3,
            "row {row} should have 3 nnz"
        );
    }
}

// ===========================================================================
// 18.7: Maximum value sizes (uint8, uint16, uint32)
// ===========================================================================

#[test]
fn test_18_7_uint8_max_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u8_max.scx");

    let indptr = vec![0u64, 3];
    let indices = vec![0u32, 1, 2];
    let values: Vec<u8> = vec![1, 128, 255];

    let header = sample_header(1, 10, 3);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(1)).unwrap();
    writer.write_var(&sample_var(10)).unwrap();
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
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let (_, _, data) = reader.read_csr_shard(0).unwrap();
    assert_eq!(data, vec![1.0f32, 128.0, 255.0]);
}

#[test]
fn test_18_7_uint16_max_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u16_max.scx");

    let indptr = vec![0u64, 3];
    let indices = vec![0u32, 1, 2];
    let raw_values: Vec<u8> = [1u16, 32768, 65535]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let header = sample_header(1, 10, 3);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(1)).unwrap();
    writer.write_var(&sample_var(10)).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &raw_values,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let (_, _, data) = reader.read_csr_shard(0).unwrap();
    assert_eq!(data, vec![1.0f32, 32768.0, 65535.0]);
}

#[test]
fn test_18_7_uint32_max_values() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u32_max.scx");

    let indptr = vec![0u64, 3];
    let indices = vec![0u32, 1, 2];
    let raw_values: Vec<u8> = [1u32, 100_000, 4_294_967_295u32]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    let header = sample_header(1, 10, 3);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(1)).unwrap();
    writer.write_var(&sample_var(10)).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &raw_values,
            CodecId::None, // u32 max doesn't work with Rice (needs shift-1, overflow)
            ValueEncoding::Uint32,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let (_, _, data) = reader.read_csr_shard(0).unwrap();
    assert_eq!(data, vec![1.0f32, 100_000.0, 4_294_967_295.0f32]);
}

// ===========================================================================
// 18.8: u16 vs u32 index dtype based on n_vars
// ===========================================================================

#[test]
fn test_18_8_u16_index_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let n_vars = 1000;
    let path = write_test_file(&dir, "u16_idx.scx", 10, n_vars, 1, CodecId::None, false);

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().index_dtype, 0, "expected u16 index_dtype");

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.1, n_vars);
    // All indices fit in u16 range
    for &idx in &csr.indices {
        assert!(idx >= 0 && idx < n_vars as i32);
    }
}

#[test]
fn test_18_8_u32_index_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("u32_idx.scx");
    let n_obs = 4;
    let n_vars: usize = 70_000;

    // Create indices that exceed u16 range
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_obs {
        // Two columns per row, one below u16 max, one above
        let col_low = (row * 10) as u32;
        let col_high = 66_000 + row as u32; // > 65535
        indices.push(col_low);
        indices.push(col_high);
        values.push(((row + 1) as u8).max(1));
        values.push(((row + 2) as u8).max(1));
        indptr.push(indptr.last().unwrap() + 2);
    }
    let total_nnz = *indptr.last().unwrap();

    let mut header = sample_header(n_obs as u64, n_vars as u64, total_nnz);
    header.index_dtype = 1; // u32

    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
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

    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.header().index_dtype, 1, "expected u32 index_dtype");

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (n_obs, n_vars));

    // Verify indices > 65535 survive round-trip
    let has_large_index = csr.indices.iter().any(|&idx| idx > 65535);
    assert!(has_large_index, "expected indices > 65535");

    // Verify exact values
    let expected_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
    assert_eq!(csr.indices, expected_indices);
}

// ---------------------------------------------------------------------------
// Phase 1B: Mixed per-shard value encoding round-trip
// ---------------------------------------------------------------------------

/// Write a file with two shards using different value encodings (Uint8 and
/// Uint16) and verify the reader correctly decodes both shards.
#[test]
fn test_mixed_value_encoding_shards() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("mixed_encoding.scx");

    let n_rows_per_shard = 100;
    let n_obs = n_rows_per_shard * 2;
    let n_vars = 500;

    // Shard 1: values in [1, 200] → Uint8
    let mut indptr1 = vec![0u64];
    let mut indices1 = Vec::new();
    let mut values1_u8 = Vec::new();
    for row in 0..n_rows_per_shard {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices1.push(col0 as u32);
        indices1.push(col1 as u32);
        values1_u8.push(((row % 200) + 1) as u8);
        values1_u8.push(((row % 200) + 2) as u8);
        indptr1.push(indptr1.last().unwrap() + 2);
    }

    // Shard 2: values in [256, 60000] → Uint16
    let mut indptr2 = vec![0u64];
    let mut indices2 = Vec::new();
    let mut values2_u16_bytes = Vec::new();
    let mut expected_values2_f32 = Vec::new();
    for row in 0..n_rows_per_shard {
        let col0 = (row * 2) % n_vars;
        let col1 = (row * 2 + 1) % n_vars;
        indices2.push(col0 as u32);
        indices2.push(col1 as u32);
        let v0 = (row * 100 + 256) as u16;
        let v1 = (row * 100 + 356) as u16;
        values2_u16_bytes.extend_from_slice(&v0.to_le_bytes());
        values2_u16_bytes.extend_from_slice(&v1.to_le_bytes());
        expected_values2_f32.push(v0 as f32);
        expected_values2_f32.push(v1 as f32);
        indptr2.push(indptr2.last().unwrap() + 2);
    }

    let total_nnz = (n_obs * 2) as u64;
    let header = sample_header(n_obs as u64, n_vars as u64, total_nnz);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Write shard 1 as Uint8
    writer
        .write_csr_shard(
            &indptr1,
            &indices1,
            &values1_u8,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();

    // Write shard 2 as Uint16
    writer
        .write_csr_shard(
            &indptr2,
            &indices2,
            &values2_u16_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            n_rows_per_shard as u64,
        )
        .unwrap();

    writer.finish().unwrap();

    // Read back and verify
    let reader = ScxReader::open(&path).unwrap();
    assert!(
        reader.header().file_checksum != 0,
        "file checksum should be non-zero"
    );

    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (n_obs, n_vars));
    assert_eq!(csr.data.len(), n_obs * 2);

    // Verify shard 1 values (uint8 range)
    let expected1: Vec<f32> = values1_u8.iter().map(|&v| v as f32).collect();
    assert_eq!(&csr.data[..n_rows_per_shard * 2], &expected1[..]);

    // Verify shard 2 values (uint16 range)
    assert_eq!(&csr.data[n_rows_per_shard * 2..], &expected_values2_f32[..]);
}
