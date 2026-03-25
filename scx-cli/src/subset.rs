// scx subset — Extract a subset of cells and/or genes into a new SCX file.

use std::io::{BufRead, BufReader};
use std::path::Path;

use scx_codec::ValueEncoding;
use scx_engine::QueryPipeline;
use scx_format::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format::reader::ScxReader;
use scx_format::section::SectionType;
use scx_format::shard::SHARD_HEADER_SIZE;
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

    // 3. Build query pipeline (single file open for all operations)
    let mut pipeline = QueryPipeline::open(input)?;

    if let Some(expr) = filter {
        pipeline = pipeline.filter_obs(expr)?;
    }

    // 4. Parse gene file if provided — supports both names and numeric indices
    let gene_indices = if let Some(gene_path) = gene_file {
        let indices = parse_gene_list(gene_path, pipeline.reader())?;
        pipeline = pipeline.select_genes(indices.clone());
        Some(indices)
    } else {
        None
    };

    // 5. Extract info from reader BEFORE collect() consumes the pipeline
    let in_header = pipeline.reader().header().clone();
    let value_encoding = detect_value_encoding_from_reader(pipeline.reader())?;

    // Check for sections that will be dropped and warn
    let dropped_layers = pipeline.reader().layer_names();
    let has_obsm = in_header.has_obsm();
    let has_obs_pred_idx = pipeline
        .reader()
        .read_obs_predicate_index_bytes()
        .ok()
        .flatten()
        .is_some();
    let has_var_pred_idx = pipeline
        .reader()
        .read_var_predicate_index_bytes()
        .ok()
        .flatten()
        .is_some();
    // Read uns before collect consumes the reader
    let uns = pipeline.reader().read_uns().ok();

    // 6. Execute query (consumes pipeline)
    let result = pipeline.collect()?;

    // 7. Report stats
    let n_output_cells = result.x.n_rows();
    let n_output_genes = result.x.n_cols();
    let output_nnz = result.x.indptr.last().copied().unwrap_or(0);

    println!(
        "Subset: {}/{} cells, {}/{} genes, {} nnz",
        n_output_cells, in_header.n_obs, n_output_genes, in_header.n_vars, output_nnz,
    );

    // Warn about dropped sections
    if !dropped_layers.is_empty() {
        eprintln!(
            "Warning: {} layer(s) dropped (layer subsetting not yet supported): {}",
            dropped_layers.len(),
            dropped_layers.join(", ")
        );
    }
    if has_obsm {
        eprintln!("Warning: obsm embeddings dropped (per-cell data cannot be subset)");
    }
    if has_obs_pred_idx || has_var_pred_idx {
        eprintln!("Warning: predicate indices dropped (invalid after subsetting)");
    }

    if dry_run {
        println!("(dry run — no output written)");
        return Ok(());
    }

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
        uns.as_ref(),
    )?;

    println!("Wrote {}", output.display());
    Ok(())
}

/// Parse a gene list file: supports both gene names and numeric indices.
///
/// If all non-comment, non-blank lines parse as u32, they are treated as
/// numeric column indices (backwards compatible). Otherwise, each line is
/// treated as a gene name and resolved against the var metadata's first
/// string column (typically `gene_id`).
fn parse_gene_list(
    path: &Path,
    reader: &ScxReader,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let buf = BufReader::new(file);
    let mut entries = Vec::new();

    for line in buf.lines() {
        let line = line?;
        let trimmed = line.trim();

        // Skip comments and blank lines
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        entries.push(trimmed.to_string());
    }

    if entries.is_empty() {
        return Err(format!("gene file is empty: {}", path.display()).into());
    }

    // Try parsing all entries as u32 indices first
    let all_numeric: Option<Vec<u32>> = entries.iter().map(|e| e.parse::<u32>().ok()).collect();

    let mut indices = if let Some(numeric_indices) = all_numeric {
        numeric_indices
    } else {
        // Resolve gene names against var metadata
        resolve_gene_names(&entries, reader)?
    };

    // Deduplicate and sort
    indices.sort_unstable();
    indices.dedup();

    Ok(indices)
}

/// Resolve gene names to column indices using the var metadata.
///
/// Looks up each name in the first string column of the var RecordBatch
/// (typically `gene_id`). Returns an error if any name is not found.
fn resolve_gene_names(
    names: &[String],
    reader: &ScxReader,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let var = reader.read_var()?;

    // Find the first string column to use as the gene name column
    let (col_name, col_idx) = var
        .schema()
        .fields()
        .iter()
        .enumerate()
        .find(|(_, f)| {
            matches!(
                f.data_type(),
                arrow::datatypes::DataType::Utf8 | arrow::datatypes::DataType::LargeUtf8
            )
        })
        .map(|(i, f)| (f.name().clone(), i))
        .ok_or("var metadata has no string column for gene name resolution")?;

    let col = var.column(col_idx);
    let string_array = col
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or_else(|| {
            format!(
                "var column '{}' is not a StringArray (type: {:?})",
                col_name,
                col.data_type()
            )
        })?;

    // Build name → index map
    let name_to_idx: std::collections::HashMap<&str, u32> = string_array
        .iter()
        .enumerate()
        .filter_map(|(i, val)| val.map(|v| (v, i as u32)))
        .collect();

    let mut indices = Vec::with_capacity(names.len());
    let mut missing = Vec::new();

    for name in names {
        match name_to_idx.get(name.as_str()) {
            Some(&idx) => indices.push(idx),
            None => missing.push(name.as_str()),
        }
    }

    if !missing.is_empty() {
        let shown: Vec<&str> = missing.iter().take(5).copied().collect();
        let suffix = if missing.len() > 5 {
            format!(" ... and {} more", missing.len() - 5)
        } else {
            String::new()
        };
        return Err(format!(
            "{} gene name(s) not found in var '{}': {}{}",
            missing.len(),
            col_name,
            shown.join(", "),
            suffix,
        )
        .into());
    }

    Ok(indices)
}

/// Detect the ValueEncoding from the first CSR shard using an existing reader.
fn detect_value_encoding_from_reader(
    reader: &ScxReader,
) -> Result<ValueEncoding, Box<dyn std::error::Error>> {
    let csr_entries = reader.catalog().shards(SectionType::CsrShard);

    if let Some(first) = csr_entries.first() {
        let bytes = reader.section_bytes(first)?;
        if bytes.len() >= SHARD_HEADER_SIZE {
            let sh = scx_format::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
                &bytes[..SHARD_HEADER_SIZE],
            ))?;
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
#[allow(clippy::too_many_arguments)]
fn write_subset_scx(
    output: &Path,
    result: &scx_engine::QueryResult,
    shard_size: u32,
    value_encoding: ValueEncoding,
    explicit_codec: Option<scx_codec::CodecId>,
    filter_expr: Option<&str>,
    gene_indices: Option<&[u32]>,
    uns: Option<&serde_json::Value>,
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
    let raw_values = value_encoding.encode_f32_batch(&result.x.data)?;

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
        let value_byte_size = value_encoding.byte_width();
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

    // Write uns if present in the input file
    if let Some(uns_data) = uns {
        writer.write_uns(uns_data)?;
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
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: params,
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::write_test_file;
    use scx_codec::CodecId;

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

    /// Write a gene name file for testing.
    fn write_gene_name_file(dir: &tempfile::TempDir, names: &[&str]) -> std::path::PathBuf {
        let path = dir.path().join("gene_names.txt");
        std::fs::write(&path, names.join("\n")).unwrap();
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

    #[test]
    fn test_subset_gene_names() {
        let dir = tempfile::tempdir().unwrap();
        // test_utils writes var with gene_id: gene_0, gene_1, gene_2, gene_3, gene_4
        let input = write_test_file(&dir, 6, 5);
        let gene_file = write_gene_name_file(&dir, &["gene_0", "gene_2"]);
        let output = dir.path().join("subset_names.scx");

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
        assert_eq!(hdr.n_vars, 2, "should have 2 genes from name resolution");
    }

    #[test]
    fn test_subset_gene_names_missing() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 5);
        let gene_file = write_gene_name_file(&dir, &["gene_0", "NONEXISTENT"]);

        let err = run_subset(
            &input,
            Some(dir.path().join("out.scx").as_path()),
            None,
            Some(gene_file.as_path()),
            false,
            10000,
            "auto",
        );
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("NONEXISTENT"),
            "error should name the missing gene"
        );
        assert!(msg.contains("not found"), "error should say 'not found'");
    }

    #[test]
    fn test_subset_preserves_uns() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input_path = dir.path().join("with_uns.scx");

        // Write a file with uns data
        let header = sample_header(6, 5);
        let mut writer = ScxWriter::new(&input_path, header).unwrap();
        writer.write_obs(&sample_obs(6)).unwrap();
        writer.write_var(&sample_var(5)).unwrap();

        // Build CSR data
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..6usize {
            let col0 = (row * 2) % 5;
            let col1 = (row * 2 + 1) % 5;
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
                scx_codec::ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // Write uns
        let uns_data = serde_json::json!({"experiment": "test_subset", "version": 42});
        writer.write_uns(&uns_data).unwrap();
        writer.finish().unwrap();

        // Run subset
        let output = dir.path().join("subset_uns.scx");
        run_subset(
            &input_path,
            Some(output.as_path()),
            Some("cell_type == 'T cell'"),
            None,
            false,
            10000,
            "auto",
        )
        .unwrap();

        // Verify uns is preserved
        let reader = ScxReader::open(&output).unwrap();
        let roundtrip_uns = reader.read_uns().unwrap();
        assert_eq!(
            roundtrip_uns["experiment"], "test_subset",
            "uns should be preserved"
        );
        assert_eq!(roundtrip_uns["version"], 42, "uns values should round-trip");
    }

    #[test]
    fn test_subset_gene_names_with_comments() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 5);

        // Gene file with comments and blank lines
        let gene_path = dir.path().join("genes_commented.txt");
        std::fs::write(
            &gene_path,
            "# Highly variable genes\ngene_1\n\n# Another comment\ngene_3\n",
        )
        .unwrap();

        let output = dir.path().join("subset_comments.scx");
        run_subset(
            &input,
            Some(output.as_path()),
            None,
            Some(gene_path.as_path()),
            false,
            10000,
            "auto",
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(
            reader.header().n_vars,
            2,
            "should have 2 genes after filtering comments"
        );
    }
}
