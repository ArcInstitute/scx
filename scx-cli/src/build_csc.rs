// scx build-csc — Build CSC (column-major) shards from existing CSR data.

use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

pub fn run_build_csc(
    input: &Path,
    output: &Path,
    memory_limit: &str,
    force: bool,
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

    // 6. Determine value encoding and codec from first CSR shard header
    let csr_entries = reader.catalog().shards_sorted();
    let (value_encoding, shard_codec) = {
        let section = reader.section_bytes(csr_entries[0])?;
        let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
            &section[..scx_format::SHARD_HEADER_SIZE],
        ))?;
        let ve = ValueEncoding::from_u8(sh.value_encoding)
            .ok_or(format!("unknown value encoding: {}", sh.value_encoding))?;
        let ci = CodecId::from_u8(sh.codec_id).ok_or(format!("unknown codec: {}", sh.codec_id))?;
        (ve, ci)
    };

    // 7. Read all CSR shards and assemble into ScxCsr for transpose
    let csr = reader.read_all_csr_shards()?;

    // 8. Perform streaming transpose
    pb.set_message("Transposing CSR → CSC...");
    let csc = scx_sparse::transpose::streaming_csr_to_csc(&[csr], n_rows, n_cols, max_bytes)?;

    // 9. Set up output header
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: 1,
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

    // 12. Re-write CSR shards from input (decode + re-encode, preserving data)
    for shard_entry in &csr_entries {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

        // Convert indices (i32) to u32 for writer
        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();

        // Convert indptr (i64) to u64 for writer
        let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();

        // Encode values back to raw bytes
        let mut raw_values = Vec::new();
        for &v in &data {
            encode_value(&mut raw_values, v, value_encoding)?;
        }

        writer.write_csr_shard(
            &indptr_u64,
            &indices_u32,
            &raw_values,
            shard_codec,
            value_encoding,
            shard_row_start,
        )?;
    }

    // 13. Write CSC shard(s) from transpose result
    //     CSC indptr (i64) → u64, CSC indices (i32) → u32
    let csc_indptr_u64: Vec<u64> = csc.indptr.iter().map(|&v| v as u64).collect();
    let csc_indices_u32: Vec<u32> = csc.indices.iter().map(|&i| i as u32).collect();

    let mut csc_raw_values = Vec::new();
    for &v in &csc.data {
        encode_value(&mut csc_raw_values, v, value_encoding)?;
    }

    writer.write_csc_shard(
        &csc_indptr_u64,
        &csc_indices_u32,
        &csc_raw_values,
        shard_codec,
        value_encoding,
        0, // col_start = 0 (single CSC shard covering all columns)
    )?;

    // 14. Copy layers
    let layer_names = reader.layer_names();
    for layer_name in &layer_names {
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();

        let layer_ve = if let Some(first) = layer_shard_entries.first() {
            let section = reader.section_bytes(first)?;
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            ValueEncoding::from_u8(sh.value_encoding).unwrap_or(value_encoding)
        } else {
            value_encoding
        };
        let layer_codec = if let Some(first) = layer_shard_entries.first() {
            let section = reader.section_bytes(first)?;
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            CodecId::from_u8(sh.codec_id).unwrap_or(shard_codec)
        } else {
            shard_codec
        };

        let mut sorted_entries = layer_shard_entries;
        sorted_entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

        for (shard_idx, entry) in sorted_entries.iter().enumerate() {
            let (indptr, indices, data) = reader.read_shard_from_entry(entry)?;
            let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
            let mut raw_values = Vec::new();
            for &v in &data {
                encode_value(&mut raw_values, v, layer_ve)?;
            }
            writer.write_layer_csr_shard(
                &indptr_u64,
                &indices_u32,
                &raw_values,
                layer_codec,
                layer_ve,
                row_start,
                layer_name,
                shard_idx as u32,
            )?;
        }
    }

    // 15. Copy obsm
    if in_header.has_obsm() {
        let all_obsm = reader.read_all_obsm()?;
        for (name, batch) in &all_obsm {
            writer.write_obsm(name, batch)?;
        }
    }

    // 16. Copy uns
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // 17. Copy predicate indices
    if let Ok(Some(data)) = reader.read_obs_predicate_index_bytes() {
        writer.write_obs_predicate_index(data)?;
    }
    if let Ok(Some(data)) = reader.read_var_predicate_index_bytes() {
        writer.write_var_predicate_index(data)?;
    }

    // 18. Add provenance
    let mut prov_entries = if let Ok(prov) = reader.read_provenance() {
        prov.operations
    } else {
        Vec::new()
    };
    prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "build-csc".to_string(),
        tool: "scx-cli 0.1.0".to_string(),
        params_json: format!("{{\"memory_limit\":\"{memory_limit}\"}}"),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;

    // 19. Finalize
    writer.finish()?;
    pb.finish_and_clear();

    println!(
        "Built CSC: {} → {} ({} rows × {} cols, {} nnz)",
        input.display(),
        output.display(),
        n_rows,
        n_cols,
        csc.data.len(),
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

/// Encode a single f32 value back to raw bytes according to the value encoding.
fn encode_value(
    buf: &mut Vec<u8>,
    value: f32,
    encoding: ValueEncoding,
) -> Result<(), Box<dyn std::error::Error>> {
    match encoding {
        ValueEncoding::Uint8 => buf.push(value as u8),
        ValueEncoding::Uint16 => buf.extend_from_slice(&(value as u16).to_le_bytes()),
        ValueEncoding::Uint32 => buf.extend_from_slice(&(value as u32).to_le_bytes()),
        ValueEncoding::Float32 => buf.extend_from_slice(&value.to_le_bytes()),
        ValueEncoding::Float16 => {
            let f16_val = half::f16::from_f32(value);
            buf.extend_from_slice(&f16_val.to_le_bytes());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_format::header::MAGIC;
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

        run_build_csc(&input, &output, "4G", false).unwrap();

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
        run_build_csc(&input, &output, "150", false).unwrap();

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
        let err = run_build_csc(&input, &output, "4G", false);
        assert!(err.is_err());

        // With --force should succeed
        run_build_csc(&input, &output, "4G", true).unwrap();
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
        let err = run_build_csc(&path, &output, "4G", false);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("no CSR shards"));
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
