//! Cloud-optimize: rewrite an SCX file with a front-of-file catalog.
//!
//! Implements SPEC §12.2. Copies all sections into a new file with the
//! full catalog duplicated near the file header, enabling single-read
//! file opening from cloud object stores.
//!
//! Layout of output:
//!   [Header 256B] [RootCatalog ≤4096B] [FrontCatalog] [Obs] [Var]
//!   [Shards...] [Layers...] [ObsmEmbeddings...] [Obsp...] [Uns]
//!   [PredicateIndexes] [Provenance] [DeletionVectors] [FullCatalog]

use std::collections::BTreeMap;
use std::io::{BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::Path;

use scx_format::catalog::{FullCatalog, FullCatalogEntry, RootCatalog, RootCatalogEntry};

use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::{align_to_8, SectionType};

use crate::error::Result;

/// Offset where sections begin: 256 (header) + 4096 (root catalog placeholder).
const SECTIONS_START_OFFSET: u64 = 4352;

/// Rewrite an SCX file with a front-of-file catalog for cloud-optimized access.
///
/// The front catalog is a byte-identical copy of the full catalog placed right
/// after the root catalog region, followed by obs/var metadata for colocation.
///
/// If the input already has a valid front catalog, this is a no-op.
///
/// Uses atomic rename: writes to `output.tmp.PID`, then renames to `output`.
pub fn cloud_optimize(input: &Path, output: &Path) -> Result<()> {
    // 1. Open and read the input file
    let input_data = std::fs::read(input)?;
    let header = FileHeader::read_from(&mut Cursor::new(&input_data[..HEADER_SIZE]))?;

    // Idempotency: if already cloud-optimized, no-op
    if header.has_front_catalog() && header.front_catalog_offset != 0 {
        // Copy if output differs from input
        if input != output {
            std::fs::copy(input, output)?;
        }
        return Ok(());
    }

    // Read full catalog
    let fc_offset = header.full_catalog_offset as usize;
    let fc_length = header.full_catalog_length as usize;
    let fc_end = fc_offset.checked_add(fc_length).ok_or_else(|| {
        crate::error::CloudError::SliceBoundsExceeded {
            offset: fc_offset,
            length: fc_length,
            data_len: input_data.len(),
        }
    })?;
    if fc_end > input_data.len() {
        return Err(crate::error::CloudError::SliceBoundsExceeded {
            offset: fc_offset,
            length: fc_length,
            data_len: input_data.len(),
        });
    }
    let full_catalog = FullCatalog::read_from(
        &mut Cursor::new(&input_data[fc_offset..fc_end]),
        fc_length,
    )?;

    // We'll write the front catalog after we know the new section offsets.
    // First, estimate its size from the original catalog so we can reserve space.
    let mut orig_catalog_bytes = Vec::new();
    full_catalog.write_to(&mut orig_catalog_bytes)?;
    let front_catalog_reserved_size = orig_catalog_bytes.len() as u64;

    // 2. Determine section ordering for the output file
    //    Order: obs → var → CsrShard → LayerCsrShard → ObsmEmbedding → ObspCsrShard
    //         → UnsBlob → ObsPredicateIndex → VarPredicateIndex → Provenance → DeletionVectors
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
    for entry in &full_catalog.entries {
        grouped
            .entry(entry.section_type as u8)
            .or_default()
            .push(entry);
    }

    // Build ordered list of entries following section_order
    let mut ordered_entries: Vec<&FullCatalogEntry> = Vec::with_capacity(full_catalog.entries.len());
    for &st in section_order {
        if let Some(entries) = grouped.get(&(st as u8)) {
            ordered_entries.extend(entries);
        }
    }

    // 3. Create output file (atomic temp path)
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

    // 4. Reserve space for front catalog (placeholder; written later with correct offsets)
    let front_catalog_offset = write_offset;
    writer.write_all(&vec![0u8; front_catalog_reserved_size as usize])?;
    write_offset += front_catalog_reserved_size;

    // Pad to 8-byte alignment after front catalog
    let aligned = align_to_8(write_offset);
    let pad = (aligned - write_offset) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
        write_offset = aligned;
    }

    // 5. Copy sections in the specified order, recording new offsets/checksums
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

        // Copy section bytes verbatim from input
        let src_start = entry.offset as usize;
        let src_len = entry.length as usize;
        let src_end = src_start.checked_add(src_len).ok_or_else(|| {
            crate::error::CloudError::SliceBoundsExceeded {
                offset: src_start,
                length: src_len,
                data_len: input_data.len(),
            }
        })?;
        if src_end > input_data.len() {
            return Err(crate::error::CloudError::SliceBoundsExceeded {
                offset: src_start,
                length: src_len,
                data_len: input_data.len(),
            });
        }
        let section_data = &input_data[src_start..src_end];
        writer.write_all(section_data)?;
        write_offset += entry.length;

        // Recompute checksum (section bytes unchanged, so checksum matches)
        new_entries.push(FullCatalogEntry {
            name: entry.name.clone(),
            offset: new_offset,
            length: entry.length,
            section_type: entry.section_type,
            checksum: entry.checksum,
            stats: entry.stats.clone(),
        });
    }

    // 6. Write the full catalog at EOF
    let catalog_aligned = align_to_8(write_offset);
    let pad = (catalog_aligned - write_offset) as usize;
    if pad > 0 {
        writer.write_all(&vec![0u8; pad])?;
    }
    let full_catalog_offset_new = catalog_aligned;

    let new_full_catalog = FullCatalog {
        catalog_version: full_catalog.catalog_version,
        manifest_sequence: full_catalog.manifest_sequence,
        prev_catalog_offset: full_catalog.prev_catalog_offset,
        n_obs: full_catalog.n_obs,
        entries: new_entries,
    };
    let mut new_catalog_bytes = Vec::new();
    new_full_catalog.write_to(&mut new_catalog_bytes)?;
    writer.write_all(&new_catalog_bytes)?;
    let new_full_catalog_length = new_catalog_bytes.len() as u64;

    // 6b. Write front catalog with correct (new) offsets by seeking back
    //     to the reserved space. The front catalog has the same entries as
    //     the full catalog (with new offsets), so it is byte-identical.
    let front_catalog_size = new_catalog_bytes.len() as u64;
    assert!(
        front_catalog_size <= front_catalog_reserved_size,
        "front catalog size {front_catalog_size} exceeds reserved {front_catalog_reserved_size}"
    );
    writer.seek(SeekFrom::Start(front_catalog_offset))?;
    writer.write_all(&new_catalog_bytes)?;
    // Zero-fill any remaining reserved bytes (entries are the same so sizes
    // should match, but be safe)
    let remaining = (front_catalog_reserved_size - front_catalog_size) as usize;
    if remaining > 0 {
        writer.write_all(&vec![0u8; remaining])?;
    }
    // Seek back to where the full catalog ended
    writer.seek(SeekFrom::Start(full_catalog_offset_new + new_full_catalog_length))?;

    // 7. Build and write root catalog at offset 256
    let root_catalog = build_root_catalog(&new_full_catalog);
    let mut root_buf = Vec::new();
    root_catalog.write_to(&mut root_buf)?;
    let root_catalog_length = root_buf.len() as u64;
    root_buf.resize(4096, 0);

    writer.seek(SeekFrom::Start(HEADER_SIZE as u64))?;
    writer.write_all(&root_buf)?;

    // 8. Write header
    let mut new_header = header;
    new_header.root_catalog_offset = HEADER_SIZE as u64;
    new_header.root_catalog_length = root_catalog_length;
    new_header.full_catalog_offset = full_catalog_offset_new;
    new_header.full_catalog_length = new_full_catalog_length;
    new_header.front_catalog_offset = front_catalog_offset;
    new_header.front_catalog_length = front_catalog_size;
    new_header.set_front_catalog();
    new_header.file_checksum = 0;

    writer.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut writer)?;

    // 9. Compute file checksum
    writer.flush()?;
    let mut file = writer.into_inner().map_err(std::io::Error::from)?;
    let file_checksum = compute_file_checksum(&mut file)?;

    new_header.file_checksum = file_checksum;
    file.seek(SeekFrom::Start(0))?;
    new_header.write_to(&mut file)?;

    // 10. fsync + atomic rename
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp_path, output)?;

    Ok(())
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

    #[test]
    fn test_cloud_optimize_sets_flag_and_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let output = dir.path().join("output.scx");

        cloud_optimize(&input, &output).unwrap();

        let data = std::fs::read(&output).unwrap();
        let hdr = FileHeader::read_from(&mut Cursor::new(&data[..HEADER_SIZE])).unwrap();

        assert!(hdr.has_front_catalog(), "front_catalog flag should be set");
        assert!(hdr.front_catalog_offset > 0, "front_catalog_offset should be non-zero");
        assert!(hdr.front_catalog_length > 0, "front_catalog_length should be non-zero");
        assert_eq!(
            hdr.front_catalog_offset, SECTIONS_START_OFFSET,
            "front catalog should start right after root catalog"
        );
    }

    #[test]
    fn test_front_catalog_identical_to_full_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let output = dir.path().join("output.scx");

        cloud_optimize(&input, &output).unwrap();

        let data = std::fs::read(&output).unwrap();
        let hdr = FileHeader::read_from(&mut Cursor::new(&data[..HEADER_SIZE])).unwrap();

        let front_start = hdr.front_catalog_offset as usize;
        let front_end = front_start + hdr.front_catalog_length as usize;
        let front_bytes = &data[front_start..front_end];

        let full_start = hdr.full_catalog_offset as usize;
        let full_end = full_start + hdr.full_catalog_length as usize;
        let full_bytes = &data[full_start..full_end];

        // They won't be identical byte-for-byte because the offsets changed.
        // But both should parse to equivalent catalogs.
        let front_catalog =
            FullCatalog::read_from(&mut Cursor::new(front_bytes), front_bytes.len()).unwrap();
        let full_catalog =
            FullCatalog::read_from(&mut Cursor::new(full_bytes), full_bytes.len()).unwrap();

        assert_eq!(front_catalog.entries.len(), full_catalog.entries.len());
        assert_eq!(front_catalog.n_obs, full_catalog.n_obs);
        assert_eq!(front_catalog.manifest_sequence, full_catalog.manifest_sequence);

        // Front catalog should now have identical offsets to the full catalog
        for (fc_entry, full_entry) in front_catalog.entries.iter().zip(full_catalog.entries.iter()) {
            assert_eq!(fc_entry.name, full_entry.name);
            assert_eq!(
                fc_entry.offset, full_entry.offset,
                "front catalog offset for '{}' should match full catalog",
                fc_entry.name
            );
            assert_eq!(fc_entry.length, full_entry.length);
        }
    }

    #[test]
    fn test_cloud_optimized_readable_by_scx_reader() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let output = dir.path().join("output.scx");

        cloud_optimize(&input, &output).unwrap();

        // ScxReader uses the full catalog at EOF, not the front catalog
        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(reader.n_obs(), 100);
        assert_eq!(reader.n_vars(), 50);

        // Read obs and var
        let obs = reader.read_obs().unwrap();
        assert_eq!(obs.num_rows(), 100);
        let var = reader.read_var().unwrap();
        assert_eq!(var.num_rows(), 50);

        // Read all shards
        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.shape.0, 100);
        assert_eq!(csr.shape.1, 50);

        // Validate checksums
        let results = reader.validate().unwrap();
        for (name, passed) in &results {
            assert!(passed, "checksum failed for section: {name}");
        }
    }

    #[test]
    fn test_cloud_optimize_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let first_output = dir.path().join("first.scx");
        let second_output = dir.path().join("second.scx");

        cloud_optimize(&input, &first_output).unwrap();
        let first_data = std::fs::read(&first_output).unwrap();

        // Running again on an already-optimized file should be a no-op (copy)
        cloud_optimize(&first_output, &second_output).unwrap();
        let second_data = std::fs::read(&second_output).unwrap();

        assert_eq!(first_data, second_data, "idempotent cloud-optimize should produce identical output");
    }

    #[test]
    fn test_cloud_optimize_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);

        // In-place optimization
        cloud_optimize(&input, &input).unwrap();

        let hdr = {
            let data = std::fs::read(&input).unwrap();
            FileHeader::read_from(&mut Cursor::new(&data[..HEADER_SIZE])).unwrap()
        };
        assert!(hdr.has_front_catalog());

        // Verify still readable
        let reader = ScxReader::open(&input).unwrap();
        assert_eq!(reader.n_obs(), 100);
    }

    #[test]
    fn test_obs_var_placed_after_front_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 100, 50);
        let output = dir.path().join("output.scx");

        cloud_optimize(&input, &output).unwrap();

        let data = std::fs::read(&output).unwrap();
        let hdr = FileHeader::read_from(&mut Cursor::new(&data[..HEADER_SIZE])).unwrap();

        // Read the full catalog to get section offsets
        let fc_start = hdr.full_catalog_offset as usize;
        let fc_len = hdr.full_catalog_length as usize;
        let catalog =
            FullCatalog::read_from(&mut Cursor::new(&data[fc_start..fc_start + fc_len]), fc_len)
                .unwrap();

        let obs_entry = catalog.get("obs").unwrap();
        let var_entry = catalog.get("var").unwrap();
        let front_end = hdr.front_catalog_offset + hdr.front_catalog_length;

        // Obs should be placed right after the front catalog (aligned)
        assert!(
            obs_entry.offset >= front_end,
            "obs should come after front catalog: obs={} front_end={}",
            obs_entry.offset,
            front_end
        );

        // Var should come right after obs
        assert!(
            var_entry.offset > obs_entry.offset,
            "var should come after obs"
        );

        // First shard should come after var
        let first_shard = catalog
            .entries
            .iter()
            .find(|e| e.section_type == SectionType::CsrShard)
            .unwrap();
        assert!(
            first_shard.offset > var_entry.offset,
            "first shard should come after var"
        );
    }

    #[test]
    fn test_front_catalog_flag_methods() {
        let mut hdr = sample_header(100, 50);
        assert!(!hdr.has_front_catalog());

        hdr.set_front_catalog();
        assert!(hdr.has_front_catalog());

        hdr.clear_front_catalog();
        assert!(!hdr.has_front_catalog());

        // Multiple flags should coexist
        hdr.set_front_catalog();
        hdr.set_deletion_vectors();
        assert!(hdr.has_front_catalog());
        assert!(hdr.has_deletion_vectors());

        hdr.clear_front_catalog();
        assert!(!hdr.has_front_catalog());
        assert!(hdr.has_deletion_vectors());
    }
}
