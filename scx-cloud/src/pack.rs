//! Pack: convert an exploded `.scxd` directory back into a packed `.scx` file.
//!
//! Implements docs/cloud.md (Exploded layout).  Reads `_catalog.bin` as the authoritative section
//! index, reads each section file, and writes a new packed `.scx` file.
//! The output includes a front-of-file catalog (cloud-ready by default).

use std::io::{BufWriter, Cursor, Seek, SeekFrom, Write};
use std::path::Path;

use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
use scx_format_io::header::{FileHeader, HEADER_SIZE};
use scx_format_io::section::{align_to_8, SectionType};

use crate::error::Result;
use crate::explode::section_name_to_path;
use crate::layout::{
    build_root_catalog, compute_file_checksum, estimate_catalog_size, order_entries_for_layout,
    SECTIONS_START_OFFSET,
};

/// Pack an exploded `.scxd` directory back into a single packed `.scx` file.
///
/// Reads `_catalog.bin` as the authoritative section index, reads each section
/// file, and writes a packed file via atomic rename.
///
/// The output is cloud-ready by default (front catalog included).
pub fn pack(input_dir: &Path, output: &Path) -> Result<()> {
    // 1. Read _catalog.bin (authoritative index)
    let catalog_bytes = std::fs::read(input_dir.join("_catalog.bin"))?;
    let original_catalog =
        FullCatalog::read_from(&mut Cursor::new(&catalog_bytes), catalog_bytes.len(), true)?;

    // 2. Read _header.bin
    let header_bytes = std::fs::read(input_dir.join("_header.bin"))?;
    let header = FileHeader::read_from(&mut Cursor::new(&header_bytes))?;

    // 3. Order entries into the cloud-optimized layout.
    let ordered_entries = order_entries_for_layout(&original_catalog.entries);

    // 4. Create output via collision-safe sibling tempfile.
    let (raw_file, tmp_path) = scx_format_io::make_sibling_tempfile(output)?;
    let mut writer = BufWriter::new(raw_file);

    // Write placeholder for header + root catalog
    writer.write_all(&vec![0u8; SECTIONS_START_OFFSET as usize])?;
    let mut write_offset = SECTIONS_START_OFFSET;

    // 5. Estimate front catalog size and write placeholder
    //    We'll come back and write the real front catalog after we know all offsets.
    let estimated_front_catalog_size = estimate_catalog_size(original_catalog.entries.len());
    let front_catalog_offset = write_offset;
    writer.write_all(&vec![0u8; estimated_front_catalog_size])?;
    write_offset += estimated_front_catalog_size as u64;

    // Ensure 8-byte alignment after front catalog placeholder
    let aligned = align_to_8(write_offset);
    let pad = (aligned - write_offset) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
        write_offset = aligned;
    }

    // 6. Read and write each section, recording new offsets
    let mut new_entries: Vec<FullCatalogEntry> = Vec::with_capacity(ordered_entries.len());
    // Phase G.2: track the modality table's new offset/length so the
    // header can be updated to point at the post-pack location.
    let mut modality_table_offset_new: u64 = 0;
    let mut modality_table_length_new: u64 = 0;

    for &entry in &ordered_entries {
        // Pad to 8-byte alignment
        let aligned = align_to_8(write_offset);
        let pad = (aligned - write_offset) as usize;
        if pad > 0 {
            writer.write_all(&vec![0u8; pad])?;
            write_offset = aligned;
        }

        let new_offset = write_offset;

        // Read section file from exploded directory
        let rel_path = section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
            crate::error::CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        let file_path = input_dir.join(&rel_path);
        let section_data = std::fs::read(&file_path)?;
        writer.write_all(&section_data)?;
        write_offset += section_data.len() as u64;

        // Recompute checksum from actual section data (may differ from
        // original if section files were modified on disk)
        let checksum = scx_format_io::blake3_hash(&section_data);

        if entry.section_type == SectionType::ModalityTable {
            modality_table_offset_new = new_offset;
            modality_table_length_new = section_data.len() as u64;
        }

        new_entries.push(FullCatalogEntry {
            name: entry.name.clone(),
            offset: new_offset,
            length: section_data.len() as u64,
            section_type: entry.section_type,
            checksum,
            modality_id: entry.modality_id,
            stats: entry.stats.clone(),
        });
    }

    // 7. Write the full catalog at EOF
    let catalog_aligned = align_to_8(write_offset);
    let pad = catalog_aligned.checked_sub(write_offset).ok_or_else(|| {
        crate::error::CloudError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "catalog alignment underflow (inconsistent layout)",
        ))
    })? as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
    }
    let full_catalog_offset_new = catalog_aligned;

    let new_full_catalog = FullCatalog {
        catalog_version: original_catalog.catalog_version,
        manifest_sequence: original_catalog.manifest_sequence,
        prev_catalog_offset: original_catalog.prev_catalog_offset,
        n_obs: original_catalog.n_obs,
        entries: new_entries,
        // Repack does not change CSR or CSC content — preserve both
        // generations so the freshness invariant survives the round-trip.
        data_generation: original_catalog.data_generation,
        csc_build_generation: original_catalog.csc_build_generation,
    };
    let mut new_catalog_bytes = Vec::new();
    new_full_catalog.write_to(&mut new_catalog_bytes)?;
    let new_full_catalog_length = new_catalog_bytes.len() as u64;
    writer.write_all(&new_catalog_bytes)?;

    // 8. Write front catalog (copy of full catalog) at the reserved position
    let front_catalog_length = new_catalog_bytes.len();
    if front_catalog_length <= estimated_front_catalog_size {
        // Fits in reservation — write and zero-pad the gap
        writer.seek(SeekFrom::Start(front_catalog_offset))?;
        writer.write_all(&new_catalog_bytes)?;
        let remaining = estimated_front_catalog_size - front_catalog_length;
        if remaining > 0 {
            writer.write_all(&vec![0u8; remaining])?;
        }
    } else {
        // Exceeded reservation — should be extremely rare. The front catalog
        // still fits because we sized generously. But if it doesn't, fall
        // back: just zero out the front catalog region (no front catalog).
        // This is safe because we always write the full catalog at EOF.
        writer.seek(SeekFrom::Start(front_catalog_offset))?;
        writer.write_all(&vec![0u8; estimated_front_catalog_size])?;
    }

    // 9. Build and write root catalog at offset 256
    let root_catalog = build_root_catalog(&new_full_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    writer.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    writer.write_all(&root_buf)?;

    // 10. Write header
    let mut new_header = header;
    new_header.root_catalog_offset = HEADER_SIZE as u64;
    new_header.root_catalog_length = root_catalog_length;
    new_header.full_catalog_offset = full_catalog_offset_new;
    new_header.full_catalog_length = new_full_catalog_length;
    // Phase G.2: pack re-lays out sections, so the source's
    // modality_table_offset is stale. Update the header to point at
    // the post-pack location (or zero if the source had no
    // modality table). The struct-copy at line 201 preserves
    // `n_modalities` and the has_modalities flag.
    new_header.modality_table_offset = modality_table_offset_new;
    new_header.modality_table_length = modality_table_length_new;

    if front_catalog_length <= estimated_front_catalog_size {
        new_header.front_catalog_offset = front_catalog_offset;
        new_header.front_catalog_length = front_catalog_length as u64;
        new_header.set_front_catalog();
    } else {
        new_header.front_catalog_offset = 0;
        new_header.front_catalog_length = 0;
        new_header.clear_front_catalog();
    }

    new_header.file_checksum = 0;
    writer.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut writer)?;

    // 11. Compute file checksum
    writer.flush()?;
    let mut file = writer.into_inner().map_err(std::io::Error::from)?;
    let file_checksum = compute_file_checksum(&mut file)?;
    new_header.file_checksum = file_checksum;
    file.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut file)?;

    // 12. fsync + atomic rename
    file.sync_all()?;
    drop(file);
    tmp_path
        .persist(output)
        .map_err(|e| crate::error::CloudError::Io(e.error))?;
    scx_format_io::chmod_to_umask(output)?;
    scx_format_io::fsync_parent_dir(output)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format_io::reader::ScxReader;
    use scx_format_io::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
    }

    fn sample_obs(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
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

    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
        let path = dir.path().join("input.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per_shard = 50;
        let mut row_offset = 0;
        while row_offset < n_obs {
            let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
            let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
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
        writer.finish().unwrap();
        path
    }

    fn write_test_file_with_extras(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
        use arrow::array::Float32Array;
        let path = dir.path().join("input_extras.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

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

        // Layer shard
        let (indptr2, indices2, values2) = sample_shard_data(n_obs, n_vars);
        let layer_shard = scx_format_io::ShardBuffers::new(
            &indptr2,
            &indices2,
            &values2,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer
            .write_layer_csr_shard("raw_counts", 0, 0, layer_shard)
            .unwrap();

        // obsm
        let obsm_schema = Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]);
        let pc1: Vec<f32> = (0..n_obs).map(|i| i as f32 * 0.1).collect();
        let pc2: Vec<f32> = (0..n_obs).map(|i| i as f32 * 0.2).collect();
        let obsm_batch = arrow::array::RecordBatch::try_new(
            Arc::new(obsm_schema),
            vec![
                Arc::new(Float32Array::from(pc1)),
                Arc::new(Float32Array::from(pc2)),
            ],
        )
        .unwrap();
        writer.write_obsm("X_pca", &obsm_batch).unwrap();

        // uns
        let uns = serde_json::json!({"description": "test dataset"});
        writer.write_uns(&uns).unwrap();

        writer.finish().unwrap();
        path
    }

    #[test]
    fn test_explode_pack_roundtrip_checksums() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded_dir = dir.path().join("exploded.scxd");
        let packed = dir.path().join("packed.scx");

        // Explode
        crate::explode::explode(&input, &exploded_dir).unwrap();

        // Pack
        pack(&exploded_dir, &packed).unwrap();

        // Read both files and compare data
        let reader_orig = ScxReader::open(&input).unwrap();
        let reader_packed = ScxReader::open(&packed).unwrap();

        assert_eq!(reader_orig.n_obs(), reader_packed.n_obs());
        assert_eq!(reader_orig.n_vars(), reader_packed.n_vars());
        assert_eq!(reader_orig.nnz(), reader_packed.nnz());

        // Read CSR data from both and compare
        let csr_orig = reader_orig.read_all_csr_shards().unwrap();
        let csr_packed = reader_packed.read_all_csr_shards().unwrap();
        assert_eq!(csr_orig.indptr, csr_packed.indptr);
        assert_eq!(csr_orig.indices, csr_packed.indices);
        assert_eq!(csr_orig.data, csr_packed.data);

        // Validate packed file checksums
        let results = reader_packed.validate().unwrap();
        for (name, passed) in &results {
            assert!(passed, "checksum failed for section: {name}");
        }
    }

    #[test]
    fn test_explode_shard_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded_dir = dir.path().join("exploded.scxd");

        crate::explode::explode(&input, &exploded_dir).unwrap();

        // Read original file and compare shard bytes
        let input_data = std::fs::read(&input).unwrap();
        let header = FileHeader::read_from(&mut Cursor::new(&input_data[..HEADER_SIZE])).unwrap();
        let fc_offset = header.full_catalog_offset as usize;
        let fc_length = header.full_catalog_length as usize;
        let catalog = FullCatalog::read_from(
            &mut Cursor::new(&input_data[fc_offset..fc_offset + fc_length]),
            fc_length,
            true,
        )
        .unwrap();

        for entry in &catalog.entries {
            if entry.section_type == SectionType::CsrShard {
                let rel_path =
                    crate::explode::section_name_to_path(&entry.name, entry.section_type).unwrap();
                let shard_file = exploded_dir.join(&rel_path);
                let shard_bytes = std::fs::read(&shard_file).unwrap();
                let original_bytes =
                    &input_data[entry.offset as usize..(entry.offset + entry.length) as usize];
                assert_eq!(
                    shard_bytes, original_bytes,
                    "shard {} bytes differ",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn test_exploded_directory_structure() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded_dir = dir.path().join("exploded.scxd");

        crate::explode::explode(&input, &exploded_dir).unwrap();

        // Check required files exist
        assert!(exploded_dir.join("_header.bin").exists());
        assert!(exploded_dir.join("_catalog.bin").exists());
        assert!(exploded_dir.join("obs.arrow").exists());
        assert!(exploded_dir.join("var.arrow").exists());
        assert!(exploded_dir.join("X").is_dir());
        assert!(exploded_dir.join("X/000000.shard").exists());
        assert!(exploded_dir.join("X/000001.shard").exists());
    }

    #[test]
    fn test_pack_reads_catalog_as_authoritative() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let exploded_dir = dir.path().join("exploded.scxd");
        let packed = dir.path().join("packed.scx");

        crate::explode::explode(&input, &exploded_dir).unwrap();

        // Pack reads _catalog.bin, not directory listing
        pack(&exploded_dir, &packed).unwrap();

        let reader = ScxReader::open(&packed).unwrap();
        assert_eq!(reader.n_obs(), 100);
        assert_eq!(reader.n_vars(), 50);

        // Packed file should be cloud-ready (has front catalog)
        assert!(reader.header().has_front_catalog());
    }

    #[test]
    fn test_explode_with_extras() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file_with_extras(&dir, 100, 50);
        let exploded_dir = dir.path().join("exploded.scxd");

        crate::explode::explode(&input, &exploded_dir).unwrap();

        // Check all section files exist
        assert!(exploded_dir.join("obs.arrow").exists());
        assert!(exploded_dir.join("var.arrow").exists());
        assert!(exploded_dir.join("X/000000.shard").exists());
        assert!(exploded_dir.join("layers/raw_counts/000000.shard").exists());
        assert!(exploded_dir.join("obsm/X_pca.arrow").exists());
        assert!(exploded_dir.join("uns.json").exists());
    }

    #[test]
    fn test_explode_pack_with_extras_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file_with_extras(&dir, 100, 50);
        let exploded_dir = dir.path().join("exploded.scxd");
        let packed = dir.path().join("packed.scx");

        crate::explode::explode(&input, &exploded_dir).unwrap();
        pack(&exploded_dir, &packed).unwrap();

        let reader = ScxReader::open(&packed).unwrap();
        assert_eq!(reader.n_obs(), 100);
        assert_eq!(reader.n_vars(), 50);

        // Verify obs/var data
        let obs = reader.read_obs().unwrap();
        assert_eq!(obs.num_rows(), 100);
        let var = reader.read_var().unwrap();
        assert_eq!(var.num_rows(), 50);

        // Verify CSR
        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.shape.0, 100);

        // Verify layer
        let layer = reader.read_layer("raw_counts").unwrap();
        assert_eq!(layer.shape.0, 100);

        // Verify obsm
        let obsm = reader.read_obsm("X_pca").unwrap();
        assert_eq!(obsm.num_rows(), 100);

        // Verify uns
        let uns = reader.read_uns().unwrap();
        assert_eq!(uns["description"], "test dataset");

        // Validate all checksums
        let results = reader.validate().unwrap();
        for (name, passed) in &results {
            assert!(passed, "checksum failed for section: {name}");
        }
    }

    #[test]
    fn test_pack_with_deletion_vectors() {
        use scx_format_io::deletion_vectors::DeletionVectors;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("with_dv.scx");
        let header = sample_header(100, 50);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(100)).unwrap();
        writer.write_var(&sample_var(50)).unwrap();

        let (indptr, indices, values) = sample_shard_data(100, 50);
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

        // Write deletion vectors (global obs rows 5 and 10).
        let mut dv = DeletionVectors::new();
        dv.insert_global([5u32, 10]);
        writer.write_deletion_vectors(&dv).unwrap();
        writer.finish().unwrap();

        // Explode → Pack roundtrip
        let exploded_dir = dir.path().join("exploded.scxd");
        let packed = dir.path().join("packed.scx");

        crate::explode::explode(&path, &exploded_dir).unwrap();
        assert!(exploded_dir.join("_deletion_vectors.bin").exists());

        pack(&exploded_dir, &packed).unwrap();

        let reader = ScxReader::open(&packed).unwrap();
        assert!(reader.header().has_deletion_vectors());

        let dv_read = reader.read_deletion_vectors().unwrap().unwrap();
        assert_eq!(dv_read.deletions.len(), 1);
        assert_eq!(dv_read.total_deleted(), 2);
    }

    /// explode + pack roundtrip on a CSC-equipped file.
    /// Verifies that the new `Xc/NNNNNN.shard` paths flow through the
    /// exploded directory and that the writer's `write_raw_shard`
    /// (after the I.1 counter fix) repopulates `n_csc_shards` and
    /// `has_csc` correctly on the packed output.
    #[test]
    fn test_explode_pack_roundtrip_preserves_csc() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("with_csc.scx");

        // Build CSR + 2-shard CSC test file directly.
        let n_obs = 12usize;
        let n_vars = 8usize;
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&input, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let mut dense = vec![0u8; n_obs * n_vars];
        for r in 0..n_obs {
            for c in 0..n_vars {
                if (r + c) % 3 == 0 {
                    dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
                }
            }
        }

        // CSR
        let mut indptr = vec![0u64];
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

        // CSC (2 shards × 4 cols)
        for chunk_start in (0..n_vars).step_by(4) {
            let chunk_end = (chunk_start + 4).min(n_vars);
            let mut ip = vec![0u64];
            let mut ix = Vec::new();
            let mut vb = Vec::new();
            for c in chunk_start..chunk_end {
                for r in 0..n_obs {
                    let v = dense[r * n_vars + c];
                    if v != 0 {
                        ix.push(r as u32);
                        vb.push(v);
                    }
                }
                ip.push(ix.len() as u64);
            }
            writer
                .write_csc_shard(
                    &ip,
                    &ix,
                    &vb,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    chunk_start as u64,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        // Read pre-roundtrip CSC state.
        let r = ScxReader::open(&input).unwrap();
        let pre_csc = r.read_all_csc_shards().unwrap();
        drop(r);

        // Explode → directory should contain Xc/000000.shard and Xc/000001.shard.
        let exploded_dir = dir.path().join("exploded.scxd");
        crate::explode::explode(&input, &exploded_dir).unwrap();
        assert!(exploded_dir.join("Xc/000000.shard").exists());
        assert!(exploded_dir.join("Xc/000001.shard").exists());

        // Pack → reassembled file preserves CSC.
        let packed = dir.path().join("packed.scx");
        pack(&exploded_dir, &packed).unwrap();

        let r = ScxReader::open(&packed).unwrap();
        assert!(r.header().has_csc(), "packed output should advertise CSC");
        assert_eq!(r.header().n_csc_shards, 2);

        // CSC contents round-trip equal.
        let post_csc = r.read_all_csc_shards().unwrap();
        assert_eq!(pre_csc.shape, post_csc.shape);
        assert_eq!(pre_csc.indptr, post_csc.indptr);
        assert_eq!(pre_csc.indices, post_csc.indices);
        assert_eq!(pre_csc.data, post_csc.data);

        // Validate per-section checksums on the packed file.
        for (name, passed) in r.validate().unwrap() {
            assert!(passed, "checksum failed for section: {name}");
        }
    }

    /// Phase G.2: write a synthetic CITE-seq-style multimodal SCX
    /// file for use by the modality-table round-trip tests below.
    /// Two modalities: rna (n_vars = 12) + adt (n_vars = 4). Each
    /// gets one CSR shard.
    fn write_multimodal_test_file(dir: &tempfile::TempDir, n_obs: usize) -> std::path::PathBuf {
        use scx_format_io::ModalityType;
        let path = dir.path().join("multi.scx");
        let mut header = sample_header(n_obs as u64, 12);
        header.n_vars = 12; // global = max across modalities
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
        writer.set_modality_n_vars(rna_id, 12).unwrap();
        writer.write_var_for(rna_id, &sample_var(12)).unwrap();
        let (rna_indptr, rna_indices, rna_values) = sample_shard_data(n_obs, 12);
        let rna_shard = scx_format_io::ShardBuffers::new(
            &rna_indptr,
            &rna_indices,
            &rna_values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(rna_id, 0, rna_shard).unwrap();

        let adt_id = writer
            .add_modality(
                "adt",
                ModalityType::Protein,
                CodecId::Zstd,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        writer.set_modality_n_vars(adt_id, 4).unwrap();
        writer.write_var_for(adt_id, &sample_var(4)).unwrap();
        let (adt_indptr, adt_indices, adt_values) = sample_shard_data(n_obs, 4);
        let adt_shard = scx_format_io::ShardBuffers::new(
            &adt_indptr,
            &adt_indices,
            &adt_values,
            CodecId::Zstd,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(adt_id, 0, adt_shard).unwrap();

        writer.finish().unwrap();
        path
    }

    /// Phase G.2: explode + pack on a CITE-seq SCX file preserves the
    /// modality table and updates header offsets to the post-pack
    /// location.
    #[test]
    fn test_explode_pack_multimodal_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_test_file(&dir, 16);
        let exploded = dir.path().join("multi.scxd");
        let repacked = dir.path().join("multi_repacked.scx");

        crate::explode::explode(&input, &exploded).unwrap();
        // The exploded directory should contain _modality_table.bin
        // (Phase G.2 layout).
        assert!(
            exploded.join("_modality_table.bin").exists(),
            "exploded layout should contain _modality_table.bin"
        );
        // Per-modality CSR shards live under X/{modality}/.
        assert!(
            exploded.join("X/rna").is_dir(),
            "exploded layout should have X/rna/ subdirectory"
        );
        assert!(
            exploded.join("X/adt").is_dir(),
            "exploded layout should have X/adt/ subdirectory"
        );

        pack(&exploded, &repacked).unwrap();
        let r = ScxReader::open(&repacked).unwrap();
        assert!(r.is_multimodal(), "round-tripped file should be multimodal");
        assert_eq!(r.n_modalities(), 2);
        let mut names: Vec<String> = r.modality_names().iter().map(|s| s.to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["adt".to_string(), "rna".to_string()]);
        // Modality table offset/length point at the post-pack
        // location, not the source.
        assert!(r.header().modality_table_offset > 0);
        assert!(r.header().modality_table_length > 0);
        // Section bytes round-trip; per-modality CSR reads still work.
        let rna_id = r.modality_id("rna").unwrap();
        let adt_id = r.modality_id("adt").unwrap();
        let rna_csr = r.read_all_csr_shards_for(rna_id).unwrap();
        assert_eq!(rna_csr.shape, (16, 12));
        let adt_csr = r.read_all_csr_shards_for(adt_id).unwrap();
        assert_eq!(adt_csr.shape, (16, 4));
        for (name, passed) in r.validate().unwrap() {
            assert!(passed, "checksum failed for section: {name}");
        }
    }

    /// Phase G.1a: cloud_optimize on a multimodal SCX file preserves
    /// the modality table and updates the header's
    /// modality_table_offset to the post-optimize location.
    #[test]
    fn test_cloud_optimize_multimodal_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_test_file(&dir, 12);
        let output = dir.path().join("multi_opt.scx");

        crate::cloud_optimize::cloud_optimize(&input, &output).unwrap();
        let r = ScxReader::open(&output).unwrap();
        assert!(r.is_multimodal());
        assert_eq!(r.n_modalities(), 2);
        let mut names: Vec<String> = r.modality_names().iter().map(|s| s.to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["adt".to_string(), "rna".to_string()]);
        assert!(r.header().modality_table_offset > 0);
        assert!(r.header().modality_table_length > 0);
        // Modality table content must round-trip exactly.
        let table = r.modality_table().unwrap();
        let rna = table.entries.iter().find(|e| e.name == "rna").unwrap();
        let adt = table.entries.iter().find(|e| e.name == "adt").unwrap();
        assert_eq!(rna.n_vars, 12);
        assert_eq!(adt.n_vars, 4);
        for (name, passed) in r.validate().unwrap() {
            assert!(passed, "checksum failed for section: {name}");
        }
    }
}
