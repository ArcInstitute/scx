// scx subset — Extract a subset of cells and/or genes into a new SCX file.

use std::io::{BufRead, BufReader};
use std::path::Path;

use scx_codec::ValueEncoding;
use scx_engine::QueryPipeline;
use scx_format::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};
use scx_format::writer::ScxWriter;

#[allow(clippy::too_many_arguments)]
pub fn run_subset(
    input: &Path,
    output: Option<&Path>,
    filter: Option<&str>,
    gene_file: Option<&Path>,
    dry_run: bool,
    shard_size: u32,
    codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Validate: at least one of --filter or --genes must be specified
    if filter.is_none() && gene_file.is_none() {
        return Err("At least one of --filter or --genes is required".into());
    }

    // 2. If not dry-run, --output is required
    if !dry_run && output.is_none() {
        return Err("--output is required (or use --dry-run)".into());
    }

    // 3. Build query pipeline
    let mut pipeline = QueryPipeline::open(input)?;

    if let Some(expr) = filter {
        pipeline = pipeline.filter_obs(expr)?;
    }

    // 4. Parse gene index file if provided
    let gene_indices = if let Some(gene_path) = gene_file {
        let indices = parse_gene_indices(gene_path)?;
        pipeline = pipeline.select_genes(indices.clone());
        Some(indices)
    } else {
        None
    };

    // 5. Execute query
    let result = pipeline.collect()?;

    // 6. Report stats
    let reader = ScxReader::open(input)?;
    let header = reader.header();
    let n_output_cells = result.x.n_rows();
    let n_output_genes = result.x.n_cols();
    let output_nnz = result.x.indptr.last().copied().unwrap_or(0);

    println!(
        "Subset: {}/{} cells, {}/{} genes, {} nnz",
        n_output_cells, header.n_obs, n_output_genes, header.n_vars, output_nnz,
    );

    if dry_run {
        println!("(dry run — no output written)");
        return Ok(());
    }

    // 7. Determine value encoding from input file
    let value_encoding = detect_value_encoding(input)?;

    // 8. Parse codec
    let explicit_codec = match codec {
        "auto" => None,
        "none" => Some(scx_codec::CodecId::None),
        "scx1" => Some(scx_codec::CodecId::Scx1),
        "zstd" => Some(scx_codec::CodecId::Zstd),
        other => {
            return Err(
                format!("Unknown codec: '{}'. Use auto, none, scx1, or zstd.", other).into(),
            )
        }
    };

    // 9. Write output SCX file
    let output = output.unwrap();
    write_subset_scx(
        output,
        &result,
        shard_size,
        value_encoding,
        explicit_codec,
        filter,
        gene_indices.as_deref(),
    )?;

    println!("Wrote {}", output.display());
    Ok(())
}

/// Parse a gene index file: one u32 index per line, skip comments (#) and blank lines.
fn parse_gene_indices(path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut indices = Vec::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();

        // Skip comments and blank lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let idx: u32 = trimmed.parse().map_err(|_| {
            format!(
                "{}:{}: non-numeric content: '{}'",
                path.display(),
                line_num + 1,
                trimmed
            )
        })?;
        indices.push(idx);
    }

    if indices.is_empty() {
        return Err(format!("gene index file is empty: {}", path.display()).into());
    }

    // Deduplicate and sort
    indices.sort_unstable();
    indices.dedup();

    Ok(indices)
}

/// Detect the ValueEncoding from the first CSR shard of an SCX file.
fn detect_value_encoding(path: &Path) -> Result<ValueEncoding, Box<dyn std::error::Error>> {
    let reader = ScxReader::open(path)?;
    let csr_entries = reader.catalog().shards(SectionType::CsrShard);

    if let Some(first) = csr_entries.first() {
        let bytes = reader.section_bytes(first)?;
        if bytes.len() >= SHARD_HEADER_SIZE {
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&bytes[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding)
                .ok_or_else(|| format!("unknown value encoding: {}", sh.value_encoding).into())
        } else {
            Err("CSR shard too small to read header".into())
        }
    } else {
        // Default for files with no shards
        Ok(ValueEncoding::Uint16)
    }
}

/// Write a subset QueryResult to a new SCX file.
fn write_subset_scx(
    output: &Path,
    result: &scx_engine::QueryResult,
    shard_size: u32,
    value_encoding: ValueEncoding,
    explicit_codec: Option<scx_codec::CodecId>,
    filter_expr: Option<&str>,
    gene_indices: Option<&[u32]>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_obs = result.x.n_rows() as u64;
    let n_vars = result.x.n_cols() as u64;

    let index_dtype = if n_vars <= 65535 { 0u8 } else { 1u8 };

    let header = FileHeader {
        magic: scx_format::MAGIC,
        format_version: CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0, // filled in by finish()
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: shard_size,
        codec_id: 0,
        index_dtype,
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
    };

    let mut writer = ScxWriter::new(output, header)?;

    // Write obs and var metadata
    writer.write_obs(&result.obs)?;
    writer.write_var(&result.var)?;

    // Convert from in-memory types (i64/i32/f32) to on-disk types (u64/u32/u8-raw)
    let indptr: Vec<u64> = result.x.indptr.iter().map(|&v| v as u64).collect();
    let indices: Vec<u32> = result.x.indices.iter().map(|&v| v as u32).collect();
    let raw_values = f32_to_raw_values(&result.x.data, value_encoding)?;

    // Shard the data
    let shard_target = shard_size as usize;
    let total_rows = indptr.len() - 1;
    let mut row_offset = 0usize;

    while row_offset < total_rows {
        let shard_rows = std::cmp::min(shard_target, total_rows - row_offset);
        let shard_indptr_start = indptr[row_offset];

        // Extract shard-local indptr (rebased to 0)
        let shard_indptr: Vec<u64> = indptr[row_offset..=row_offset + shard_rows]
            .iter()
            .map(|&v| v - shard_indptr_start)
            .collect();

        let shard_nnz = *shard_indptr.last().unwrap();

        // Extract shard-local indices
        let idx_start = shard_indptr_start as usize;
        let idx_end = (shard_indptr_start + shard_nnz) as usize;
        let shard_indices = &indices[idx_start..idx_end];

        // Extract shard-local values
        let value_byte_size = match value_encoding {
            ValueEncoding::Uint8 => 1,
            ValueEncoding::Uint16 | ValueEncoding::Float16 => 2,
            ValueEncoding::Uint32 | ValueEncoding::Float32 => 4,
        };
        let val_start = idx_start * value_byte_size;
        let val_end = idx_end * value_byte_size;
        let shard_values = &raw_values[val_start..val_end];

        // Codec selection: explicit or auto
        let codec_id = if let Some(c) = explicit_codec {
            c
        } else {
            scx_format::select_codec(shard_values, value_encoding)
        };

        writer.write_csr_shard(
            &shard_indptr,
            shard_indices,
            shard_values,
            codec_id,
            value_encoding,
            row_offset as u64,
        )?;

        row_offset += shard_rows;
    }

    // Build provenance params
    let mut params = String::from("{");
    if let Some(f) = filter_expr {
        params.push_str(&format!("\"filter\":\"{}\"", f.replace('\"', "\\\"")));
    }
    if let Some(genes) = gene_indices {
        if filter_expr.is_some() {
            params.push(',');
        }
        params.push_str(&format!("\"n_genes\":{}", genes.len()));
    }
    params.push('}');

    // Write provenance
    writer.write_provenance(vec![scx_format::ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "subset".to_string(),
        tool: "scx-cli 0.1.0".to_string(),
        params_json: params,
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Convert f32 data to raw LE bytes matching the given ValueEncoding.
fn f32_to_raw_values(
    data: &[f32],
    encoding: ValueEncoding,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::with_capacity(data.len() * encoding.byte_width());
    match encoding {
        ValueEncoding::Uint8 => {
            for &v in data {
                if !(0.0..=255.0).contains(&v) {
                    return Err(format!("value {v} out of range for uint8 (0..255)").into());
                }
                bytes.push(v as u8);
            }
        }
        ValueEncoding::Uint16 => {
            for &v in data {
                if !(0.0..=65535.0).contains(&v) {
                    return Err(format!("value {v} out of range for uint16 (0..65535)").into());
                }
                bytes.extend_from_slice(&(v as u16).to_le_bytes());
            }
        }
        ValueEncoding::Uint32 => {
            for &v in data {
                if !(0.0..=u32::MAX as f32).contains(&v) {
                    return Err(format!("value {v} out of range for uint32").into());
                }
                bytes.extend_from_slice(&(v as u32).to_le_bytes());
            }
        }
        ValueEncoding::Float32 => {
            for &v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        ValueEncoding::Float16 => {
            for &v in data {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
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
            shard_target_rows: 10000,
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
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
        ]);
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let types: Vec<&str> = (0..n)
            .map(|i| match i % 3 {
                0 => "T cell",
                1 => "B cell",
                _ => "NK cell",
            })
            .collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(types)),
            ],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> arrow::array::RecordBatch {
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    /// Write a test SCX file with CSR shards and obs metadata for filtering.
    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
        let path = dir.path().join("test.scx");
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

    /// Write a gene index file for testing.
    fn write_gene_file(dir: &tempfile::TempDir, indices: &[u32]) -> std::path::PathBuf {
        let path = dir.path().join("genes.txt");
        let content = indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_subset_filter_only() {
        let dir = tempfile::tempdir().unwrap();
        // 9 cells: indices 0,3,6 are "T cell" (3 cells match)
        let input = write_test_file(&dir, 9, 5);
        let output = dir.path().join("subset.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            Some("cell_type == 'T cell'"),
            None,
            false,
            10000,
            "auto",
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();
        // 9 cells: 0,1,2..8, every 3rd is "T cell" → 3 cells
        assert_eq!(hdr.n_obs, 3, "should have 3 T cells");
        assert_eq!(hdr.n_vars, 5, "genes should be preserved");
    }

    #[test]
    fn test_subset_genes_only() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 5);
        let gene_file = write_gene_file(&dir, &[0, 2]);
        let output = dir.path().join("subset_genes.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            None,
            Some(gene_file.as_path()),
            false,
            10000,
            "auto",
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();
        assert_eq!(hdr.n_obs, 6, "all cells should be preserved");
        assert_eq!(hdr.n_vars, 2, "should have 2 genes");
    }

    #[test]
    fn test_subset_combined() {
        let dir = tempfile::tempdir().unwrap();
        // 9 cells, 5 genes
        let input = write_test_file(&dir, 9, 5);
        let gene_file = write_gene_file(&dir, &[0, 1]);
        let output = dir.path().join("subset_combined.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            Some("cell_type == 'T cell'"),
            Some(gene_file.as_path()),
            false,
            10000,
            "auto",
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let hdr = reader.header();
        assert_eq!(hdr.n_obs, 3, "should have 3 T cells");
        assert_eq!(hdr.n_vars, 2, "should have 2 genes");
    }

    #[test]
    fn test_subset_dry_run() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 9, 5);

        // dry run: no output path needed
        run_subset(
            &input,
            None,
            Some("cell_type == 'T cell'"),
            None,
            true,
            10000,
            "auto",
        )
        .unwrap();
        // Should succeed without writing any file
    }

    #[test]
    fn test_subset_no_filter_no_genes_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 5);
        let output = dir.path().join("subset.scx");

        let err = run_subset(
            &input,
            Some(output.as_path()),
            None,
            None,
            false,
            10000,
            "auto",
        );
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("--filter"));
        assert!(msg.contains("--genes"));
    }

    #[test]
    fn test_subset_no_output_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 5);

        let err = run_subset(
            &input,
            None,
            Some("cell_type == 'T cell'"),
            None,
            false,
            10000,
            "auto",
        );
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("--output"));
    }
}
