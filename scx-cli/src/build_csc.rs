// scx build-csc — Build CSC (column-major) shards from existing CSR data.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, CURRENT_FORMAT_VERSION, MAGIC};
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

use crate::rewrite_helpers;

pub fn run_build_csc(
    input: &Path,
    output: &Path,
    memory_limit: &str,
    force: bool,
    csc_cols_per_shard: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Parse memory limit string ("4G" → 4 * 1024^3 bytes)
    let max_bytes = parse_memory_limit(memory_limit)?;

    // 2. Validate input exists
    if !input.exists() {
        return Err(format!("input file does not exist: {}", input.display()).into());
    }

    // 3. Check output doesn't exist (unless --force)
    if output.exists() && !force {
        return Err(format!(
            "{} already exists (use --force to overwrite)",
            output.display()
        )
        .into());
    }
    if output.exists() && force {
        std::fs::remove_file(output)?;
    }

    // 4. Open input file
    let reader = ScxReader::open(input)?;
    let in_header = reader.header();

    // 5. Validate: input has CSR shards
    if in_header.n_csr_shards == 0 {
        return Err("Input file has no CSR shards".into());
    }

    let n_rows = in_header.n_obs as usize;
    let n_cols = in_header.n_vars as usize;

    // Show progress
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .expect("valid template"),
    );
    pb.set_message(format!(
        "Building CSC shards ({} rows × {} cols)...",
        n_rows, n_cols
    ));
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // 6. Read CSR shard entries and determine default codec from first shard
    let csr_entries = reader.catalog().csr_shards_sorted();
    let first_sh = reader.read_shard_header(csr_entries[0])?;
    let csc_value_encoding = ValueEncoding::from_u8(first_sh.value_encoding).ok_or(format!(
        "unknown value encoding: {}",
        first_sh.value_encoding
    ))?;
    let csc_codec = CodecId::from_u8(first_sh.codec_id)
        .ok_or(format!("unknown codec: {}", first_sh.codec_id))?;

    // 7. Read CSR shards individually (preserves shard boundaries for streaming transpose)
    let csr_shards: Vec<scx_sparse::ScxCsr> = csr_entries
        .iter()
        .map(|entry| {
            let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;
            let n_shard_rows = indptr.len() - 1;
            Ok(scx_sparse::ScxCsr::new(
                (n_shard_rows, n_cols),
                indptr,
                indices,
                data,
            )?)
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;

    // 9. Set up output header
    let out_header = FileHeader {
        magic: MAGIC,
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

    // 10. Read all metadata from input
    let obs = reader.read_obs()?;
    let var = reader.read_var()?;

    // 11. Create writer and write metadata
    pb.set_message("Writing output file...");
    let mut writer = ScxWriter::new(output, out_header)?;
    writer.write_obs(&obs)?;
    writer.write_var(&var)?;

    // 12. Re-write CSR shards from input (decode + re-encode, per-shard codec)
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

    // 13. Write CSC shards via the streaming transpose iterator. Each
    //     iterator chunk becomes one CSC shard; col_start is read from
    //     the iterator *before* advancing to the next chunk.
    pb.set_message("Transposing CSR → CSC (streaming)...");
    let mut iter = scx_sparse::streaming_csr_to_csc_iter_with_cap(
        &csr_shards,
        n_rows,
        n_cols,
        max_bytes,
        csc_cols_per_shard,
    )?;

    let mut total_csc_nnz: usize = 0;
    let mut n_csc_shards_written: u32 = 0;
    loop {
        // current_col_start() returns the start of the NEXT chunk
        // (== end of the previous chunk, == 0 on first iteration).
        let col_start = iter.current_col_start() as u64;
        let chunk = match iter.next() {
            Some(c) => c?,
            None => break,
        };

        let csc_indptr_u64: Vec<u64> = chunk.indptr.iter().map(|&v| v as u64).collect();
        let csc_indices_u32: Vec<u32> = chunk.indices.iter().map(|&i| i as u32).collect();
        let csc_raw_values = csc_value_encoding.encode_f32_batch(&chunk.data)?;

        writer.write_csc_shard(
            &csc_indptr_u64,
            &csc_indices_u32,
            &csc_raw_values,
            csc_codec,
            csc_value_encoding,
            col_start,
        )?;

        total_csc_nnz += chunk.data.len();
        n_csc_shards_written += 1;
    }

    // 14. Copy auxiliary sections (layers, obsm, uns, predicate indices, provenance)
    let params_json = format!(
        "{{\"memory_limit\":\"{memory_limit}\",\"csc_cols_per_shard\":{csc_cols_per_shard}}}"
    );
    rewrite_helpers::copy_auxiliary_sections(&reader, &mut writer, "build-csc", &params_json)?;

    // 15. Finalize
    writer.finish()?;
    pb.finish_and_clear();

    println!(
        "Built CSC: {} → {} ({} rows × {} cols, {} nnz, {} CSC shard{})",
        input.display(),
        output.display(),
        n_rows,
        n_cols,
        total_csc_nnz,
        n_csc_shards_written,
        if n_csc_shards_written == 1 { "" } else { "s" },
    );
    Ok(())
}

/// Parse human-readable memory limit ("100M", "4G", "1024K") to bytes.
fn parse_memory_limit(s: &str) -> Result<usize, Box<dyn std::error::Error>> {
    let s = s.trim();
    let (num_str, multiplier) = match s.as_bytes().last() {
        Some(b'K' | b'k') => (&s[..s.len() - 1], 1024usize),
        Some(b'M' | b'm') => (&s[..s.len() - 1], 1024 * 1024),
        Some(b'G' | b'g') => (&s[..s.len() - 1], 1024 * 1024 * 1024),
        _ => (s, 1),
    };
    let n: usize = num_str.parse()?;
    Ok(n * multiplier)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_header, sample_obs, sample_var};

    /// Write a test SCX file with CSR shards.
    fn write_test_input(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join("input.scx");
        let header = sample_header(n_obs as u64, n_vars as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // Build CSR data: each row has 2 nnz
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
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

    #[test]
    fn test_build_csc_creates_csc_shards() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 6, 4);
        let output = dir.path().join("output.scx");

        run_build_csc(&input, &output, "4G", false, 5000).unwrap();

        // Verify output file
        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();

        // Should have CSR + CSC shards
        assert!(hdr.n_csr_shards > 0, "output must have CSR shards");
        assert!(hdr.n_csc_shards > 0, "output must have CSC shards");
        assert!(hdr.has_csc(), "has_csc flag must be set");

        // CSR data should match original
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(new_csr.shape, orig_csr.shape);
        assert_eq!(new_csr.indptr, orig_csr.indptr);
        assert_eq!(new_csr.indices, orig_csr.indices);
        assert_eq!(new_csr.data, orig_csr.data);

        // Verify CSC data is a valid transpose
        // Convert both CSR and CSC to dense and compare
        let dense_csr = orig_csr.to_dense().unwrap();

        // Read CSC shard from catalog
        let csc_entries: Vec<_> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == scx_format::section::SectionType::CscShard)
            .collect();
        assert_eq!(csc_entries.len(), 1, "should have exactly 1 CSC shard");

        // Decode the CSC shard
        let (csc_indptr, csc_indices, csc_data) =
            reader.read_shard_from_entry(csc_entries[0]).unwrap();

        // Reconstruct dense from CSC
        let (n_rows, n_cols) = orig_csr.shape;
        let mut dense_csc = vec![0.0f32; n_rows * n_cols];
        for col in 0..n_cols {
            let start = csc_indptr[col] as usize;
            let end = csc_indptr[col + 1] as usize;
            for j in start..end {
                let row = csc_indices[j] as usize;
                dense_csc[row * n_cols + col] = csc_data[j];
            }
        }
        assert_eq!(dense_csc, dense_csr, "CSC transpose must match CSR data");
    }

    #[test]
    fn test_build_csc_memory_limit() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 10, 6);
        let output = dir.path().join("output_mem.scx");

        // Use a small memory limit that still allows at least 1 col per pass
        // 10 rows × 12 bytes = 120 bytes/col, so 150 → 1 col per pass → 6 passes
        run_build_csc(&input, &output, "150", false, 5000).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert!(reader.header().has_csc());
        assert!(reader.header().n_csc_shards > 0);

        // Verify data integrity
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(new_csr.data, orig_csr.data);
    }

    #[test]
    fn test_build_csc_force_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 4, 3);
        let output = dir.path().join("output_force.scx");

        // Create the output file first
        std::fs::write(&output, b"placeholder").unwrap();

        // Without --force should fail
        let err = run_build_csc(&input, &output, "4G", false, 5000);
        assert!(err.is_err());

        // With --force should succeed
        run_build_csc(&input, &output, "4G", true, 5000).unwrap();
        let reader = ScxReader::open(&output).unwrap();
        assert!(reader.header().has_csc());
    }

    #[test]
    fn test_build_csc_no_csr_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.scx");

        // Write a file with obs/var but no CSR shards
        let header = sample_header(3, 2);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(3)).unwrap();
        writer.write_var(&sample_var(2)).unwrap();

        // Need at least one shard for finish to write a catalog with data
        // Actually, let's just test with a proper file that has no shards
        writer.finish().unwrap();

        let output = dir.path().join("output.scx");
        let err = run_build_csc(&path, &output, "4G", false, 5000);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("no CSR shards"));
    }

    /// write a small CSR-only file, run build-csc with
    /// `--csc-cols-per-shard 3` over n_vars=10, and verify that the
    /// output has exactly ceil(10/3) = 4 CSC shards with correct,
    /// non-overlapping `[col_start, col_end)` ranges.
    #[test]
    fn test_build_csc_multi_shard_layout() {
        let dir = tempfile::tempdir().unwrap();
        // 4 rows × 10 cols. write_test_input gives each row 2 nnz.
        let input = write_test_input(&dir, 4, 10);
        let output = dir.path().join("multi_csc.scx");

        run_build_csc(&input, &output, "4G", false, 3).unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();
        assert!(hdr.has_csc());
        assert_eq!(hdr.n_csc_shards, 4, "ceil(10/3) = 4 CSC shards");

        let csc_entries = reader.catalog().csc_shards_sorted();
        assert_eq!(csc_entries.len(), 4);
        let ranges: Vec<std::ops::Range<u64>> = csc_entries
            .iter()
            .map(|e| e.stats.as_ref().unwrap().col_range())
            .collect();
        // Chunks: [0..3), [3..6), [6..9), [9..10).
        assert_eq!(ranges, vec![0..3, 3..6, 6..9, 9..10]);

        // CSC ranges are contiguous and cover [0, n_vars).
        for w in ranges.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert_eq!(ranges.first().unwrap().start, 0);
        assert_eq!(ranges.last().unwrap().end, hdr.n_vars);

        // Every CSC shard's on-disk shard_type byte is 1.
        let data = std::fs::read(&output).unwrap();
        for entry in &csc_entries {
            let section = &data[entry.offset as usize..][..entry.length as usize];
            let sh = scx_format::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::shard::SHARD_HEADER_SIZE],
            ))
            .unwrap();
            assert_eq!(sh.shard_type, 1);
        }

        // High-level reads round-trip: CSC == densify(CSR).
        let orig_reader = ScxReader::open(&input).unwrap();
        let dense_csr = orig_reader
            .read_all_csr_shards()
            .unwrap()
            .to_dense()
            .unwrap();
        let csc_concat = reader.read_all_csc_shards().unwrap();
        assert_eq!(csc_concat.shape, (4, 10));
        assert_eq!(csc_concat.to_dense().unwrap(), dense_csr);
    }

    /// `read_csc_columns(2..7)` over the multi-shard layout
    /// returns the same densified slice as densifying the full matrix
    /// then column-slicing. Validates that partial-overlap shards are
    /// `col_slice`d post-decode and that fully-skipped shards do not
    /// affect the result.
    #[test]
    fn test_build_csc_multi_shard_read_columns_range() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_input(&dir, 4, 10);
        let output = dir.path().join("multi_csc_range.scx");

        run_build_csc(&input, &output, "4G", false, 3).unwrap();

        let orig_reader = ScxReader::open(&input).unwrap();
        let dense = orig_reader
            .read_all_csr_shards()
            .unwrap()
            .to_dense()
            .unwrap();
        let n_cols = orig_reader.header().n_vars as usize;

        let reader = ScxReader::open(&output).unwrap();

        // Reference dense slice for [c_lo, c_hi).
        let dense_slice = |c_lo: usize, c_hi: usize| -> Vec<f32> {
            let cols = c_hi - c_lo;
            let mut out = vec![0.0f32; 4 * cols];
            for r in 0..4 {
                for (oc, sc) in (c_lo..c_hi).enumerate() {
                    out[r * cols + oc] = dense[r * n_cols + sc];
                }
            }
            out
        };

        let cases = [
            (2u32, 7u32), // partial overlap on shards 0 and 2; full shard 1
            (0, 10),      // entire range
            (3, 6),       // exact-boundary single shard
            (4, 5),       // single column
            (8, 10),      // crosses the trailing 1-col shard
            (0, 0),       // empty
        ];
        for (c_lo, c_hi) in cases {
            let csc = reader.read_csc_columns(c_lo..c_hi).unwrap();
            let got = csc.to_dense().unwrap();
            let want = dense_slice(c_lo as usize, c_hi as usize);
            assert_eq!(got, want, "mismatch on cols [{c_lo}..{c_hi})");
        }
    }

    ///  codec parity: rerun the A.4 codec sweep idea
    /// (None / Scx1 / Zstd / Lz4Shuffle / Pcodec × Uint8) on a
    /// multi-shard CSC layout. Inputs are integer Uint8 throughout; the
    /// codec from the input shards drives the output codec.
    #[test]
    fn test_build_csc_multi_shard_codec_parity() {
        let codecs = [
            CodecId::None,
            CodecId::Scx1,
            CodecId::Zstd,
            CodecId::Lz4Shuffle,
            CodecId::Pcodec,
        ];

        for codec in codecs {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("input.scx");
            // Build a 4×10 file using the requested codec.
            let n_obs = 4usize;
            let n_vars = 10usize;
            let header = sample_header(n_obs as u64, n_vars as u64);
            let mut writer = ScxWriter::new(&path, header).unwrap();
            writer.write_obs(&sample_obs(n_obs)).unwrap();
            writer.write_var(&sample_var(n_vars)).unwrap();

            let mut indptr = vec![0u64];
            let mut indices = Vec::new();
            let mut values = Vec::new();
            for row in 0..n_obs {
                let c0 = (row * 2) % n_vars;
                let c1 = (row * 2 + 1) % n_vars;
                indices.push(c0 as u32);
                indices.push(c1 as u32);
                values.push(((row + 1) % 256) as u8);
                values.push(((row + 2) % 256) as u8);
                indptr.push(indptr.last().unwrap() + 2);
            }
            writer
                .write_csr_shard(&indptr, &indices, &values, codec, ValueEncoding::Uint8, 0)
                .unwrap();
            writer.finish().unwrap();

            let output = dir.path().join("multi_csc.scx");
            run_build_csc(&path, &output, "4G", false, 3).unwrap();

            let reader = ScxReader::open(&output).unwrap();
            assert_eq!(reader.header().n_csc_shards, 4, "codec={codec:?}");

            // CSC density equals CSR density.
            let orig = ScxReader::open(&path).unwrap();
            let dense_csr = orig.read_all_csr_shards().unwrap().to_dense().unwrap();
            let csc = reader.read_all_csc_shards().unwrap();
            assert_eq!(
                csc.to_dense().unwrap(),
                dense_csr,
                "round-trip mismatch for codec={codec:?}"
            );
        }
    }

    #[test]
    fn test_parse_memory_limit() {
        assert_eq!(parse_memory_limit("100").unwrap(), 100);
        assert_eq!(parse_memory_limit("1K").unwrap(), 1024);
        assert_eq!(parse_memory_limit("1k").unwrap(), 1024);
        assert_eq!(parse_memory_limit("100M").unwrap(), 100 * 1024 * 1024);
        assert_eq!(parse_memory_limit("4G").unwrap(), 4 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_limit("2g").unwrap(), 2 * 1024 * 1024 * 1024);
    }
}
