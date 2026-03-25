// scx upgrade — Upgrade an SCX file to the latest format version.
//
// Rewrites the file fully through the current ScxWriter, which produces
// the latest format version. If old and current versions match, no-op.

use std::path::Path;

use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;

use crate::rewrite_helpers;

pub fn run_upgrade(
    input: &Path,
    output: Option<&Path>,
    in_place: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Validate: either output or --in-place required
    if output.is_none() && !in_place {
        return Err("Specify an output path or use --in-place".into());
    }

    // 2. Open and read existing file
    let reader = ScxReader::open(input)?;
    let old_version = reader.header().format_version;

    if old_version == CURRENT_FORMAT_VERSION {
        println!(
            "File is already at format version {} (current). Nothing to do.",
            CURRENT_FORMAT_VERSION
        );
        return Ok(());
    }

    // 3. Determine output path
    let output_path = if in_place {
        // Write to temp file, then atomic rename
        let mut tmp = input.to_path_buf();
        tmp.set_extension("scx.upgrading");
        tmp
    } else {
        output.unwrap().to_path_buf()
    };

    // 4. Re-read and re-write using current writer
    let rewrite_result = rewrite_with_current_version(&reader, &output_path);
    if rewrite_result.is_err() && in_place {
        // Clean up orphaned temp file on failure
        let _ = std::fs::remove_file(&output_path);
    }
    rewrite_result?;

    let new_reader = ScxReader::open(&output_path)?;
    let new_version = new_reader.header().format_version;
    drop(new_reader);

    // 5. If in-place, atomic rename
    if in_place {
        std::fs::rename(&output_path, input)?;
        println!(
            "Upgraded {} from v{} \u{2192} v{} (in-place)",
            input.display(),
            old_version,
            new_version
        );
    } else {
        println!(
            "Upgraded {} \u{2192} {} (v{} \u{2192} v{})",
            input.display(),
            output_path.display(),
            old_version,
            new_version
        );
    }

    Ok(())
}

/// Re-read all sections from the reader and re-write them using the current
/// ScxWriter, which produces the latest format_version.
fn rewrite_with_current_version(
    reader: &ScxReader,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let in_header = reader.header();

    let csr_entries = reader.catalog().shards_sorted();

    // Set up output header
    let out_header = FileHeader {
        magic: scx_format::MAGIC,
        format_version: CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: in_header.flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: in_header.codec_id,
        index_dtype: in_header.index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: in_header.manifest_sequence + 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        reserved: [0u8; 132],
    };

    // Read metadata
    let obs = reader.read_obs()?;
    let var = reader.read_var()?;

    // Create writer
    let mut writer = ScxWriter::new(output, out_header)?;
    writer.write_obs(&obs)?;
    writer.write_var(&var)?;

    // Re-write CSR shards (per-shard codec)
    for shard_entry in &csr_entries {
        let sh = reader.read_shard_header(shard_entry)?;
        let ve = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
        let ci = CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;

        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
        let raw_values = ve.encode_f32_batch(&data)?;

        writer.write_csr_shard(
            &indptr_u64,
            &indices_u32,
            &raw_values,
            ci,
            ve,
            shard_row_start,
        )?;
    }

    // Re-write CSC shards (if present, per-shard codec)
    let csc_entries: Vec<&scx_format::FullCatalogEntry> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .collect();

    for csc_entry in &csc_entries {
        let sh = reader.read_shard_header(csc_entry)?;
        let ve = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
        let ci = CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;

        let (indptr, indices, data) = reader.read_shard_from_entry(csc_entry)?;
        let col_start = csc_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
        let raw_values = ve.encode_f32_batch(&data)?;

        writer.write_csc_shard(&indptr_u64, &indices_u32, &raw_values, ci, ve, col_start)?;
    }

    // Copy auxiliary sections (layers, obsm, uns, predicate indices, provenance)
    rewrite_helpers::copy_auxiliary_sections(reader, &mut writer, "upgrade", "{}")?;

    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_header, write_test_file};
    use scx_codec::{CodecId, ValueEncoding};

    #[test]
    fn test_upgrade_preserves_data() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 8, 5);
        let output = dir.path().join("upgraded.scx");

        // Current version is 1 and file is at version 1, so this should no-op.
        // To test data preservation, we call rewrite_with_current_version directly.
        let reader = ScxReader::open(&input).unwrap();
        rewrite_with_current_version(&reader, &output).unwrap();

        // Verify output data matches input
        let orig_reader = ScxReader::open(&input).unwrap();
        let new_reader = ScxReader::open(&output).unwrap();

        let orig_hdr = orig_reader.header();
        let new_hdr = new_reader.header();
        assert_eq!(new_hdr.n_obs, orig_hdr.n_obs);
        assert_eq!(new_hdr.n_vars, orig_hdr.n_vars);
        assert_eq!(new_hdr.format_version, 1);

        // Verify CSR data matches
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = new_reader.read_all_csr_shards().unwrap();
        assert_eq!(new_csr.shape, orig_csr.shape);
        assert_eq!(new_csr.indptr, orig_csr.indptr);
        assert_eq!(new_csr.indices, orig_csr.indices);
        assert_eq!(new_csr.data, orig_csr.data);

        // Verify obs metadata
        let orig_obs = orig_reader.read_obs().unwrap();
        let new_obs = new_reader.read_obs().unwrap();
        assert_eq!(new_obs.num_rows(), orig_obs.num_rows());
        assert_eq!(new_obs.num_columns(), orig_obs.num_columns());

        // Verify var metadata
        let orig_var = orig_reader.read_var().unwrap();
        let new_var = new_reader.read_var().unwrap();
        assert_eq!(new_var.num_rows(), orig_var.num_rows());
    }

    #[test]
    fn test_upgrade_noop_current() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 4, 3);
        let output = dir.path().join("upgraded.scx");

        // File is at version 1 (current), should no-op
        let result = run_upgrade(&input, Some(output.as_path()), false);
        assert!(result.is_ok());
        // Output should NOT have been created (no-op)
        assert!(!output.exists(), "no-op upgrade should not create output");
    }

    #[test]
    fn test_upgrade_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 4);

        // Read original data for comparison
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let orig_n_obs = orig_reader.header().n_obs;
        drop(orig_reader);

        // rewrite_with_current_version directly, then atomic rename to simulate
        // an in-place upgrade (since version is already 1, run_upgrade would no-op)
        let tmp_path = dir.path().join("test.scx.upgrading");
        let reader = ScxReader::open(&input).unwrap();
        rewrite_with_current_version(&reader, &tmp_path).unwrap();
        drop(reader);
        std::fs::rename(&tmp_path, &input).unwrap();

        // Verify data preserved after in-place rewrite
        let reader = ScxReader::open(&input).unwrap();
        assert_eq!(reader.header().n_obs, orig_n_obs);
        assert_eq!(reader.header().format_version, 1);

        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.shape, orig_csr.shape);
        assert_eq!(csr.indptr, orig_csr.indptr);
        assert_eq!(csr.indices, orig_csr.indices);
        assert_eq!(csr.data, orig_csr.data);
    }

    #[test]
    fn test_upgrade_no_output_no_inplace_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 4, 3);

        let err = run_upgrade(&input, None, false);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("--in-place"));
    }
}
