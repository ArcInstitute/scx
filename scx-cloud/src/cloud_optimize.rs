//! Cloud-optimize: rewrite an SCX file with a front-of-file catalog.
//!
//! Implements docs/cloud.md (Cloud-optimized layout). Copies all sections into a new file with the
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
/// Uses atomic rename: writes to a randomized temp file, then renames to `output`.
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
        true,
    )?;

    // We'll write the front catalog after we know the new section offsets.
    // First, estimate its size from the original catalog so we can reserve space.
    let mut orig_catalog_bytes = Vec::new();
    full_catalog.write_to(&mut orig_catalog_bytes)?;
    let front_catalog_reserved_size = orig_catalog_bytes.len() as u64;

    // 2. Determine section ordering for the output file
    //    Order: obs → var → CsrShard → CscShard → LayerCsrShard
    //         → ObsmEmbedding → ObspCsrShard → UnsBlob
    //         → ObsPredicateIndex → VarPredicateIndex → Provenance
    //         → DeletionVectors
    //
    // CscShard placed adjacent to CsrShard so column-major reads stay
    // in a contiguous prefix region of the cloud-optimized file.
    let section_order: &[SectionType] = &[
        SectionType::ObsMetadata,
        SectionType::ObsIndex,
        SectionType::VarMetadata,
        SectionType::VarIndex,
        SectionType::CsrShard,
        SectionType::CscShard,
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
    let known_types: std::collections::HashSet<u8> =
        section_order.iter().map(|&st| st as u8).collect();
    let mut ordered_entries: Vec<&FullCatalogEntry> =
        Vec::with_capacity(full_catalog.entries.len());
    for &st in section_order {
        if let Some(entries) = grouped.get(&(st as u8)) {
            ordered_entries.extend(entries);
        }
    }
    // Include any section types not in section_order (unknown/future types)
    for (&group_type, entries) in &grouped {
        if !known_types.contains(&group_type) {
            ordered_entries.extend(entries);
        }
    }

    // 3. Create output file via collision-safe sibling tempfile.
    let (raw_file, tmp_path) = scx_format::make_sibling_tempfile(output)?;
    let mut writer = BufWriter::new(raw_file);

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
    // Phase G.1a: track the modality table's new offset/length so the
    // header can be updated to point at the post-cloud_optimize
    // location. The struct-copy below preserves `n_modalities` and
    // the has_modalities flag.
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

        if entry.section_type == SectionType::ModalityTable {
            modality_table_offset_new = new_offset;
            modality_table_length_new = entry.length;
        }

        // Recompute checksum (section bytes unchanged, so checksum matches)
        new_entries.push(FullCatalogEntry {
            name: entry.name.clone(),
            offset: new_offset,
            length: entry.length,
            section_type: entry.section_type,
            checksum: entry.checksum,
            modality_id: entry.modality_id,
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
    writer.seek(SeekFrom::Start(
        full_catalog_offset_new + new_full_catalog_length,
    ))?;

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
    // Phase G.1a: cloud_optimize re-lays out sections, so the
    // source's modality_table_offset is stale. Point at the new
    // location (or 0 if the source had none).
    new_header.modality_table_offset = modality_table_offset_new;
    new_header.modality_table_length = modality_table_length_new;
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
    tmp_path
        .persist(output)
        .map_err(|e| crate::error::CloudError::Io(e.error))?;
    scx_format::chmod_to_umask(output)?;
    scx_format::fsync_parent_dir(output)?;

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
    Ok(scx_format::checksum::truncate_hash_to_u64(&hash))
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
            format_version: scx_format::CURRENT_FORMAT_VERSION,
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
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
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
        assert!(
            hdr.front_catalog_offset > 0,
            "front_catalog_offset should be non-zero"
        );
        assert!(
            hdr.front_catalog_length > 0,
            "front_catalog_length should be non-zero"
        );
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
            FullCatalog::read_from(&mut Cursor::new(front_bytes), front_bytes.len(), true).unwrap();
        let full_catalog =
            FullCatalog::read_from(&mut Cursor::new(full_bytes), full_bytes.len(), true).unwrap();

        assert_eq!(front_catalog.entries.len(), full_catalog.entries.len());
        assert_eq!(front_catalog.n_obs, full_catalog.n_obs);
        assert_eq!(
            front_catalog.manifest_sequence,
            full_catalog.manifest_sequence
        );

        // Front catalog should now have identical offsets to the full catalog
        for (fc_entry, full_entry) in front_catalog
            .entries
            .iter()
            .zip(full_catalog.entries.iter())
        {
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

        assert_eq!(
            first_data, second_data,
            "idempotent cloud-optimize should produce identical output"
        );
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
        let catalog = FullCatalog::read_from(
            &mut Cursor::new(&data[fc_start..fc_start + fc_len]),
            fc_len,
            true,
        )
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

    /// cloud_optimize on a CSC-equipped file preserves the
    /// `n_csc_shards` count and `has_csc` flag bit, and the CscShard
    /// catalog entries survive the section-copy reorder.
    #[test]
    fn test_cloud_optimize_preserves_csc() {
        let dir = tempfile::tempdir().unwrap();
        let n_obs = 12usize;
        let n_vars = 8usize;
        let input = dir.path().join("with_csc.scx");

        // Build a CSR + CSC test file directly.
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&input, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // Build the same dense matrix that the read assertions expect.
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

        // CSC: two shards of 4 columns each
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

        // Sanity: pre-cloud-optimize file has CSC.
        let r = ScxReader::open(&input).unwrap();
        assert!(r.header().has_csc());
        assert_eq!(r.header().n_csc_shards, 2);
        let pre_csc = r.read_all_csc_shards().unwrap();
        drop(r);

        // Run cloud_optimize.
        let output = dir.path().join("optimized.scx");
        cloud_optimize(&input, &output).unwrap();

        // Post-cloud-optimize: CSC count + flag preserved.
        let r = ScxReader::open(&output).unwrap();
        assert!(r.header().has_front_catalog());
        assert!(
            r.header().has_csc(),
            "has_csc flag should be preserved through cloud_optimize"
        );
        assert_eq!(r.header().n_csc_shards, 2);

        // CSC catalog entries survive the reorder.
        let csc_entries: Vec<_> = r
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .collect();
        assert_eq!(csc_entries.len(), 2);

        // CSC contents round-trip equal: sorted (col, row, value)
        // triples match the pre-optimize file.
        let post_csc = r.read_all_csc_shards().unwrap();
        assert_eq!(pre_csc.shape, post_csc.shape);
        assert_eq!(pre_csc.indptr, post_csc.indptr);
        assert_eq!(pre_csc.indices, post_csc.indices);
        assert_eq!(pre_csc.data, post_csc.data);
    }
}
