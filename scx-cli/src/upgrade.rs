// scx upgrade — Upgrade an SCX file to the latest format version.
//
// Rewrites the file fully through the current ScxWriter, which produces
// the latest format version. If old and current versions match, no-op.

use std::path::Path;

use scx_format_io::header::{FileHeader, DEFAULT_WRITE_FORMAT_VERSION};
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;

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

    // `scx upgrade` re-writes to the newest **unframed** version (v4 requires
    // row-group framing, which upgrade does not add).
    if old_version == DEFAULT_WRITE_FORMAT_VERSION {
        println!(
            "File is already at format version {} (current). Nothing to do.",
            DEFAULT_WRITE_FORMAT_VERSION
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

    let csr_entries = reader.catalog().csr_shards_sorted();

    // Set up output header (canonical v3 upgrade: bump manifest, preserve
    // flags/codec/index dtype from the source; writer fills nnz + shard counts).
    let out_header = FileHeader {
        flags: in_header.flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: in_header.codec_id,
        index_dtype: in_header.index_dtype,
        manifest_sequence: in_header.manifest_sequence + 1,
        ..Default::default()
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
        let ve = crate::shard_utils::decode_value_encoding(sh.value_encoding)?;
        let ci = crate::shard_utils::decode_codec_id(sh.codec_id)?;

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

    // Re-write CSC shards (if present, per-shard codec). Sorted by
    // major_start() (= col_start for CSC entries via the axis-overload
    // in ShardStats; on-disk fields are unchanged).
    let csc_entries = reader.catalog().csc_shards_sorted();

    for csc_entry in &csc_entries {
        let sh = reader.read_shard_header(csc_entry)?;
        let ve = crate::shard_utils::decode_value_encoding(sh.value_encoding)?;
        let ci = crate::shard_utils::decode_codec_id(sh.codec_id)?;

        let (indptr, indices, data) = reader.read_shard_from_entry(csc_entry)?;
        let col_start = csc_entry
            .stats
            .as_ref()
            .map(|s| s.major_start(csc_entry.section_type))
            .unwrap_or(0);

        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
        let raw_values = ve.encode_f32_batch(&data)?;

        writer.write_csc_shard(&indptr_u64, &indices_u32, &raw_values, ci, ve, col_start)?;
    }

    // Copy auxiliary sections (layers, obsm, uns, predicate indices, provenance)
    scx_ops::copy_auxiliary_sections(reader, &mut writer, "upgrade", "{}")?;

    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::write_test_file;
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
        assert_eq!(
            new_hdr.format_version,
            scx_format_io::DEFAULT_WRITE_FORMAT_VERSION
        );

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
        assert_eq!(
            reader.header().format_version,
            scx_format_io::DEFAULT_WRITE_FORMAT_VERSION
        );

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

    /// write a file with CSR + multi-shard CSC, run
    /// rewrite_with_current_version, and confirm the output preserves
    /// has_csc, the CSC shard count, per-shard column ranges, and
    /// densified contents.
    #[test]
    fn test_upgrade_preserves_csc_multi_shard() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use scx_format_io::section::SectionType;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("with_csc.scx");

        // 4 rows × 6 cols dense reference (column-by-column nnz).
        let n_rows = 4usize;
        let n_cols = 6usize;
        #[rustfmt::skip]
        let dense: Vec<f32> = vec![
            // col: 0    1    2    3    4    5
                   1.0, 0.0, 0.0, 4.0, 0.0, 7.0,
                   0.0, 2.0, 5.0, 0.0, 0.0, 8.0,
                   0.0, 0.0, 0.0, 0.0, 6.0, 0.0,
                   3.0, 0.0, 0.0, 0.0, 0.0, 9.0,
        ];

        // Build CSC arrays for a column range from the dense reference.
        let csc_arrays = |col_start: usize, col_end: usize| -> (Vec<u64>, Vec<u32>, Vec<u8>) {
            let mut indptr: Vec<u64> = vec![0];
            let mut indices: Vec<u32> = Vec::new();
            let mut values: Vec<u8> = Vec::new();
            for col in col_start..col_end {
                for row in 0..n_rows {
                    let v = dense[row * n_cols + col];
                    if v != 0.0 {
                        indices.push(row as u32);
                        values.push(v as u8);
                    }
                }
                indptr.push(indices.len() as u64);
            }
            (indptr, indices, values)
        };

        let header = sample_header(n_rows as u64, n_cols as u64);
        let mut writer = ScxWriter::new(&input, header).unwrap();
        writer.write_obs(&sample_obs(n_rows)).unwrap();
        writer.write_var(&sample_var(n_cols)).unwrap();

        // Empty CSR shard for file invariants.
        let csr_indptr = vec![0u64; n_rows + 1];
        writer
            .write_csr_shard(
                &csr_indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // 3 CSC shards: cols [0..2), [2..4), [4..6) — non-uniform on
        // purpose so the col_range preservation is meaningful.
        for (cs, ce) in [(0usize, 2usize), (2, 4), (4, 6)] {
            let (ip, ix, vb) = csc_arrays(cs, ce);
            writer
                .write_csc_shard(
                    &ip,
                    &ix,
                    &vb,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    cs as u64,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        // Sanity: input has CSC.
        let in_reader = ScxReader::open(&input).unwrap();
        assert!(in_reader.header().has_csc());
        assert_eq!(in_reader.header().n_csc_shards, 3);
        let in_csc = in_reader.read_all_csc_shards().unwrap();
        assert_eq!(in_csc.to_dense().unwrap(), dense);

        // Rewrite through current writer.
        let output = dir.path().join("upgraded.scx");
        rewrite_with_current_version(&in_reader, &output).unwrap();
        drop(in_reader);

        // Output preserves CSC: flag, count, per-shard col ranges, contents.
        let out_reader = ScxReader::open(&output).unwrap();
        assert!(
            out_reader.header().has_csc(),
            "has_csc must survive upgrade"
        );
        assert_eq!(out_reader.header().n_csc_shards, 3);

        let out_csc_entries = out_reader.catalog().csc_shards_sorted();
        assert_eq!(out_csc_entries.len(), 3);
        let ranges: Vec<std::ops::Range<u64>> = out_csc_entries
            .iter()
            .map(|e| e.stats.as_ref().unwrap().col_range())
            .collect();
        assert_eq!(ranges, vec![0..2, 2..4, 4..6]);

        // On-disk shard_type byte must remain `1` for re-emitted CSC
        // shards (invariant survives rewrite).
        let out_data = std::fs::read(&output).unwrap();
        for entry in &out_csc_entries {
            let section = &out_data[entry.offset as usize..][..entry.length as usize];
            let sh = scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format_io::shard::SHARD_HEADER_SIZE],
            ))
            .unwrap();
            assert_eq!(sh.shard_type, 1, "CSC shard byte must be 1 after upgrade");
            assert!(sh.is_csc(SectionType::CscShard));
        }

        // Densify and compare.
        let out_csc = out_reader.read_all_csc_shards().unwrap();
        assert_eq!(out_csc.shape, (n_rows, n_cols));
        assert_eq!(out_csc.to_dense().unwrap(), dense);
    }
}
