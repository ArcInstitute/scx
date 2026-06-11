//! Property-based test for explode → pack section-payload byte identity
//!
//! Strategy: generate small SCX files with random shape (n_obs, n_vars,
//! shard_target_rows) and content; `explode` them into an `.scxd/`
//! directory; `pack` the directory back into a packed `.scx`. Compare
//! every section's payload bytes between the original and packed file
//! using their respective full catalogs.
//!
//! Invariant: each catalog entry's section bytes (offset..offset+length)
//! must be bit-exact equal between the original packed input and the
//! re-packed output. The catalog itself and front-catalog metadata may
//! differ — the pack reorders sections — but section payloads must not.

use std::sync::Arc;

use arrow::array::{Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use proptest::prelude::*;
use scx_cloud::explode::explode;
use scx_cloud::pack::pack;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::FullCatalog;
use scx_format_io::header::{FileHeader, HEADER_SIZE};
use scx_format_io::writer::ScxWriter;

// =========================================================================
// SCX fixture construction
// =========================================================================

fn sample_obs(n: usize) -> RecordBatch {
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

fn sample_shard(n_rows: usize, n_vars: usize, seed: u64) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let mut state = seed.wrapping_add(1);
    for _ in 0..n_rows {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let row_nnz = ((state >> 33) % 4) as usize + 1;
        let mut row_indices: Vec<u32> = (0..row_nnz)
            .map(|k| ((state >> (10 + k)) as u32) % (n_vars as u32))
            .collect();
        row_indices.sort_unstable();
        row_indices.dedup();
        for &idx in &row_indices {
            indices.push(idx);
            values.push((idx % 250 + 1) as u8);
        }
        indptr.push(indptr.last().unwrap() + row_indices.len() as u64);
    }
    (indptr, indices, values)
}

fn make_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
}

fn write_test_file(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    seed: u64,
    with_extras: bool,
) {
    let header = make_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 40usize;
    let mut row_offset = 0usize;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) =
            sample_shard(shard_rows, n_vars, seed.wrapping_add(row_offset as u64));
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }

    if with_extras {
        // obsm
        let pc1: Vec<f32> = (0..n_obs).map(|i| i as f32 * 0.1).collect();
        let pc2: Vec<f32> = (0..n_obs).map(|i| i as f32 * 0.2).collect();
        let obsm_schema = Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]);
        let obsm = RecordBatch::try_new(
            Arc::new(obsm_schema),
            vec![
                Arc::new(Float32Array::from(pc1)),
                Arc::new(Float32Array::from(pc2)),
            ],
        )
        .unwrap();
        writer.write_obsm("X_pca", &obsm).unwrap();

        // uns
        let uns = serde_json::json!({"description": "proptest fixture", "seed": seed});
        writer.write_uns(&uns).unwrap();
    }

    writer.finish().unwrap();
}

// =========================================================================
// Section-payload comparison
// =========================================================================

fn read_full_catalog(bytes: &[u8]) -> FullCatalog {
    let header = FileHeader::read_from(&mut std::io::Cursor::new(&bytes[..HEADER_SIZE])).unwrap();
    let fc_off = header.full_catalog_offset as usize;
    let fc_len = header.full_catalog_length as usize;
    FullCatalog::read_from(
        &mut std::io::Cursor::new(&bytes[fc_off..fc_off + fc_len]),
        fc_len,
        true,
    )
    .unwrap()
}

/// Compare every original catalog entry's payload bytes against the
/// matching payload in the packed file. Matches by `(section_type,
/// name, modality_id)`.
fn assert_section_payloads_byte_identical(orig_bytes: &[u8], packed_bytes: &[u8]) {
    let orig_cat = read_full_catalog(orig_bytes);
    let packed_cat = read_full_catalog(packed_bytes);

    assert_eq!(
        orig_cat.entries.len(),
        packed_cat.entries.len(),
        "entry count differs"
    );

    for orig_entry in &orig_cat.entries {
        let matched = packed_cat
            .entries
            .iter()
            .find(|e| {
                e.section_type as u8 == orig_entry.section_type as u8
                    && e.name == orig_entry.name
                    && e.modality_id == orig_entry.modality_id
            })
            .unwrap_or_else(|| {
                panic!(
                    "no matching packed entry for {:?} '{}' modality {}",
                    orig_entry.section_type, orig_entry.name, orig_entry.modality_id
                )
            });

        assert_eq!(
            orig_entry.length, matched.length,
            "length mismatch for {:?} '{}'",
            orig_entry.section_type, orig_entry.name
        );

        let orig_payload = &orig_bytes
            [orig_entry.offset as usize..(orig_entry.offset + orig_entry.length) as usize];
        let packed_payload =
            &packed_bytes[matched.offset as usize..(matched.offset + matched.length) as usize];

        assert_eq!(
            orig_payload, packed_payload,
            "payload mismatch for {:?} '{}'",
            orig_entry.section_type, orig_entry.name
        );
    }
}

// =========================================================================
// Property tests
// =========================================================================

proptest! {
    // Lower case count: each iteration writes/explodes/packs a real
    // SCX file on disk. 20 is enough to exercise variability without
    // making the test slow.
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Section payloads survive explode → pack byte-for-byte for
    /// minimal SCX files (obs + var + CSR shards only).
    #[test]
    fn explode_pack_preserves_section_payloads_minimal(
        n_obs in 10usize..=120,
        n_vars in 10usize..=80,
        seed in any::<u64>(),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let orig = dir.path().join("orig.scx");
        let exploded = dir.path().join("orig.scxd");
        let packed = dir.path().join("packed.scx");

        write_test_file(&orig, n_obs, n_vars, seed, false);
        explode(&orig, &exploded).unwrap();
        pack(&exploded, &packed).unwrap();

        let orig_bytes = std::fs::read(&orig).unwrap();
        let packed_bytes = std::fs::read(&packed).unwrap();
        assert_section_payloads_byte_identical(&orig_bytes, &packed_bytes);
    }

    /// Same invariant with obsm + uns sections added.
    #[test]
    fn explode_pack_preserves_section_payloads_with_extras(
        n_obs in 30usize..=100,
        n_vars in 20usize..=60,
        seed in any::<u64>(),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let orig = dir.path().join("orig.scx");
        let exploded = dir.path().join("orig.scxd");
        let packed = dir.path().join("packed.scx");

        write_test_file(&orig, n_obs, n_vars, seed, true);
        explode(&orig, &exploded).unwrap();
        pack(&exploded, &packed).unwrap();

        let orig_bytes = std::fs::read(&orig).unwrap();
        let packed_bytes = std::fs::read(&packed).unwrap();
        assert_section_payloads_byte_identical(&orig_bytes, &packed_bytes);
    }
}
