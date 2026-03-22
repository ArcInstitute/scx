//! Pack: convert an exploded `.scxd` directory back into a packed `.scx` file.
//!
//! Implements SPEC §12.5.  Reads `_catalog.bin` as the authoritative section
//! index, reads each section file, and writes a new packed `.scx` file.
//! The output includes a front-of-file catalog (cloud-ready by default).

use std::collections::BTreeMap;
use std::io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use scx_format::catalog::{FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry};
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::{align_to_8, SectionType};

use crate::error::Result;
use crate::explode::section_name_to_path;

/// Offset where sections begin: 256 (header) + 4096 (root catalog placeholder).
const SECTIONS_START_OFFSET: u64 = 4352;

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
        FullCatalog::read_from(&mut Cursor::new(&catalog_bytes), catalog_bytes.len())?;

    // 2. Read _header.bin
    let header_bytes = std::fs::read(input_dir.join("_header.bin"))?;
    let header = FileHeader::read_from(&mut Cursor::new(&header_bytes))?;

    // 3. Define section ordering for cloud-optimized layout
    let section_order: &[SectionType] = &[
        SectionType::ObsMetadata,
        SectionType::ObsIndex,
        SectionType::VarMetadata,
        SectionType::VarIndex,
        SectionType::CsrShard,
        SectionType::LayerCsrShard,
        SectionType::ObsmEmbedding,
        SectionType::ObspCsrShard,
        SectionType::UnsBlob,
        SectionType::ObsPredicateIndex,
        SectionType::VarPredicateIndex,
        SectionType::Provenance,
        SectionType::DeletionVectors,
    ];

    // Group entries by section type, preserving order within each group
    let mut grouped: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
    for entry in &original_catalog.entries {
        grouped
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let mut ordered_entries: Vec<&FullCatalogEntry> =
        Vec::with_capacity(original_catalog.entries.len());
    for &st in section_order {
        if let Some(entries) = grouped.get(&(st as u8)) {
            ordered_entries.extend(entries);
        }
    }

    // 4. Create output via atomic temp file
    let tmp_path = std::path::PathBuf::from(format!(
        "{}.tmp.{}",
        output.display(),
        std::process::id()
    ));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(&tmp_path)?;
    let mut writer = BufWriter::new(file);

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
        let rel_path = section_name_to_path(&entry.name, entry.section_type);
        let file_path = input_dir.join(&rel_path);
        let section_data = std::fs::read(&file_path)?;
        writer.write_all(&section_data)?;
        write_offset += section_data.len() as u64;

        new_entries.push(FullCatalogEntry {
            name: entry.name.clone(),
            offset: new_offset,
            length: section_data.len() as u64,
            section_type: entry.section_type,
            checksum: entry.checksum,
            stats: entry.stats.clone(),
        });
    }

    // 7. Write the full catalog at EOF
    let catalog_aligned = align_to_8(write_offset);
    let pad = (catalog_aligned - write_offset) as usize;
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
    std::fs::rename(&tmp_path, output)?;

    Ok(())
}

/// Estimate catalog size: 30 bytes header + n_entries × 100 bytes.
/// Generous upper bound to avoid two-pass writes.
fn estimate_catalog_size(n_entries: usize) -> usize {
    let size = 30 + n_entries * 100 + 32; // +32 for trailing checksum
    // Round up to 8-byte alignment
    (size + 7) & !7
}

/// Build a root catalog from a full catalog.
fn build_root_catalog(catalog: &FullCatalog) -> RootCatalog {
    let mut groups: BTreeMap<u8, Vec<&FullCatalogEntry>> = BTreeMap::new();
    for entry in &catalog.entries {
        groups
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    let mut root_entries = Vec::new();
    for (&group_type, entries) in &groups {
        let first_offset = entries.iter().map(|e| e.offset).min().unwrap_or(0);
        let total_length: u64 = entries.iter().map(|e| e.length).sum();
        let n_sections = entries.len() as u32;
        root_entries.push(RootCatalogEntry {
            group_type,
            first_section_offset: first_offset,
            total_group_length: total_length,
            n_sections,
            summary: [0u8; 32],
        });
    }

    RootCatalog {
        n_section_groups: root_entries.len() as u16,
        entries: root_entries,
    }
}

/// Compute file checksum: BLAKE3 of entire file, truncated to u64.
fn compute_file_checksum(file: &mut (impl Read + Seek)) -> Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = blake3::Hasher::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = file.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        hasher.update(&chunk[..n]);
    }
    let hash = hasher.finalize();
    Ok(u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_format::header::MAGIC;
    use scx_format::reader::ScxReader;
    use scx_format::writer::ScxWriter;

    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
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
        writer
            .write_layer_csr_shard(
                &indptr2,
                &indices2,
                &values2,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
                "raw_counts",
                0,
            )
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
        )
        .unwrap();

        for entry in &catalog.entries {
            if entry.section_type == SectionType::CsrShard {
                let rel_path = crate::explode::section_name_to_path(&entry.name, entry.section_type);
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
        use scx_format::deletion_vectors::{DeletionVectors, ShardDeletion};

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

        // Write deletion vectors
        let mut bitmap = roaring::RoaringBitmap::new();
        bitmap.insert(5);
        bitmap.insert(10);
        let dv = DeletionVectors {
            dv_version: 1,
            shards: vec![ShardDeletion {
                shard_id: 0,
                bitmap,
            }],
        };
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
        assert_eq!(dv_read.shards.len(), 1);
        assert_eq!(dv_read.total_deleted(), 2);
    }
}
