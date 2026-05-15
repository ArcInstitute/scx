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
    modality: Option<&str>,
    dry_run: bool,
    shard_size: u32,
    codec: &str,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    // Phase F.4: pure modality extraction (no filter / no genes).
    // The output is a single-modality v2 file containing just the
    // chosen modality's CSR + var, with the file's global obs.
    if let Some(name) = modality {
        if filter.is_none() && gene_file.is_none() {
            if dry_run {
                return Err("--dry-run is not supported for `--modality NAME` extraction".into());
            }
            let out_path = output.ok_or("--output is required for `--modality NAME` extraction")?;
            return extract_modality(input, out_path, name, shard_size, codec);
        }
        // Filter/genes scoping for a specific modality is a Phase F+
        // follow-on: the QueryPipeline below is single-modality and
        // would need per-modality plumbing to honour `--modality`.
        return Err(
            "`--modality NAME` combined with `--filter` / `--genes` is not yet supported; \
             extract the modality first via `scx subset --modality NAME --output …`, then \
             rerun the filter on the extracted single-modality file"
                .into(),
        );
    }

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
    // CSC sidecars are dropped on subset: row / column projection
    // changes the global index space, so input CSC `indices` arrays
    // would silently reference rows / columns that no longer exist.
    // Caller can opt back in via `--rebuild-csc` to re-run `build-csc`
    // against the projected output.
    if in_header.has_csc() {
        eprintln!(
            "Warning: subset dropped CSC shards from {input}: rerun \
             `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar",
            input = input.display()
        );
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
        "lz4" => Some(scx_codec::CodecId::Lz4Shuffle),
        "pcodec" => Some(scx_codec::CodecId::Pcodec),
        other => {
            return Err(format!(
                "Unknown codec: '{}'. Use auto, none, scx1, zstd, lz4, or pcodec.",
                other
            )
            .into())
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

    // Re-emit the CSC sidecar against the projected output.
    if rebuild_csc {
        scx_ops::rebuild_csc_inplace(output, csc_cols_per_shard, "4G")?;
        println!("Rebuilt CSC sidecar on {}", output.display());
    }

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

/// Phase F.4: extract a single modality from a multimodal SCX file
/// to a new single-modality v2 file. Cells (obs) are global across
/// modalities, so the output's obs matches the input's obs.
fn extract_modality(
    input: &Path,
    output: &Path,
    modality_name: &str,
    shard_size: u32,
    codec: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = ScxReader::open(input)?;
    if !reader.is_multimodal() {
        return Err(format!(
            "input file is single-modality; `--modality {modality_name}` is not applicable"
        )
        .into());
    }
    let modality_id = reader.modality_id(modality_name).ok_or_else(|| {
        format!(
            "input file does not have a modality named '{modality_name}'; \
             run `scx info {}` to list modalities",
            input.display()
        )
    })?;
    let info = reader
        .modality_info(modality_id)
        .expect("modality_id resolved above");

    let n_obs = reader.header().n_obs;
    let n_vars = info.n_vars;

    // Detect value encoding from the chosen modality's first CSR shard.
    let value_encoding = {
        let csr_entries: Vec<&scx_format::FullCatalogEntry> = reader
            .catalog()
            .shards(SectionType::CsrShard)
            .into_iter()
            .filter(|e| e.modality_id == modality_id)
            .collect();
        if let Some(first_shard) = csr_entries.first() {
            let bytes = reader.section_bytes(first_shard)?;
            if bytes.len() >= SHARD_HEADER_SIZE {
                let sh = scx_format::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
                    &bytes[..SHARD_HEADER_SIZE],
                ))?;
                ValueEncoding::from_u8(sh.value_encoding).ok_or_else(|| {
                    format!(
                        "unknown value encoding {} on modality '{modality_name}'",
                        sh.value_encoding
                    )
                })?
            } else {
                return Err("input CSR shard too small to read header".into());
            }
        } else {
            return Err(format!(
                "modality '{modality_name}' has no CSR shards in {}",
                input.display()
            )
            .into());
        }
    };

    let explicit_codec = match codec {
        "auto" => None,
        "none" => Some(scx_codec::CodecId::None),
        "scx1" => Some(scx_codec::CodecId::Scx1),
        "zstd" => Some(scx_codec::CodecId::Zstd),
        "lz4" => Some(scx_codec::CodecId::Lz4Shuffle),
        "pcodec" => Some(scx_codec::CodecId::Pcodec),
        other => {
            return Err(format!(
                "unknown codec: '{other}'. Use auto, none, scx1, zstd, lz4, or pcodec."
            )
            .into());
        }
    };
    let modality_type = info.modality_type;

    let csr = reader.read_all_csr_shards_for(modality_id)?;
    let var = reader.read_var_for(modality_id)?;
    let obs = reader.read_obs()?;
    // PR #68: preserve metadata when extracting a single modality.
    // Per-modality `uns/{name}` wins; fall back to the file-wide
    // `uns` so a multimodal file with only one of the two still
    // round-trips its annotations through `extract`.
    let uns = reader
        .read_uns_for(modality_id)
        .ok()
        .or_else(|| reader.read_uns().ok());

    let index_dtype = if n_vars <= 65535 { 0u8 } else { 1u8 };
    let header = FileHeader {
        magic: scx_format::MAGIC,
        format_version: CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
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
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output, header)?;
    writer.write_obs(&obs)?;
    writer.write_var(&var)?;

    let indptr: Vec<u64> = csr.indptr.iter().map(|&v| v as u64).collect();
    let indices: Vec<u32> = csr.indices.iter().map(|&v| v as u32).collect();
    let raw_values = value_encoding.encode_f32_batch(&csr.data)?;

    let shard_target = shard_size as usize;
    // `indptr.len() - 1` would underflow on an empty indptr;
    // `saturating_sub` matches the row-sharding template in
    // pyscx::from_mudata and is also rust-1.95 clippy-clean
    // (`unnecessary_min_or_max` flags the prior `.max(0)`).
    let total_rows = indptr.len().saturating_sub(1);
    let mut row_offset = 0usize;
    while row_offset < total_rows {
        let shard_rows = std::cmp::min(shard_target, total_rows - row_offset);
        let shard_indptr_start = indptr[row_offset];
        let shard_indptr: Vec<u64> = indptr[row_offset..=row_offset + shard_rows]
            .iter()
            .map(|&v| v - shard_indptr_start)
            .collect();
        let shard_nnz = *shard_indptr.last().unwrap();
        let idx_start = shard_indptr_start as usize;
        let idx_end = (shard_indptr_start + shard_nnz) as usize;
        let shard_indices = &indices[idx_start..idx_end];
        let value_byte_size = value_encoding.byte_width();
        let val_start = idx_start * value_byte_size;
        let val_end = idx_end * value_byte_size;
        let shard_values = &raw_values[val_start..val_end];
        let codec_id = match explicit_codec {
            Some(c) => c,
            None => {
                scx_format::select_codec_for_modality(shard_values, value_encoding, modality_type)
            }
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

    // PR #68: preserve uns + provenance — `write_subset_scx` does this
    // for filter/gene-index subsets; the modality-extraction path was
    // dropping them silently.
    if let Some(uns_data) = &uns {
        writer.write_uns(uns_data)?;
    }
    writer.write_provenance(vec![scx_format::ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "modality_extract".to_string(),
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: serde_json::json!({ "modality": modality_name }).to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    println!(
        "Extracted modality '{modality_name}' ({n_vars} vars) from {} to {}",
        input.display(),
        output.display()
    );
    Ok(())
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
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
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
            scx_format::select_codec_for_modality(
                shard_values,
                value_encoding,
                scx_format::ModalityType::Rna,
            )
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            true,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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

    /// Helper: build a 2-modality SCX file for the extract_modality
    /// tests. `with_global_uns` / `with_rna_uns` control which uns
    /// sections are written. Returns the input path and a clone of
    /// whichever uns blob the caller asked to be written, for asserts.
    #[allow(clippy::type_complexity)]
    fn write_multimodal_test_file(
        dir: &tempfile::TempDir,
        global_uns: Option<serde_json::Value>,
        rna_uns: Option<serde_json::Value>,
    ) -> std::path::PathBuf {
        use crate::test_utils::{sample_obs, sample_var};
        use scx_format::modality::ModalityType;

        let path = dir.path().join("multimodal_uns.scx");
        // header.n_vars = max across modalities = 5.
        let mut header = crate::test_utils::sample_header(3, 5);
        header.n_vars = 5;

        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(3)).unwrap();

        let rna_id = writer
            .add_modality(
                "rna",
                ModalityType::Rna,
                CodecId::None,
                scx_codec::ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        let adt_id = writer
            .add_modality(
                "adt",
                ModalityType::Protein,
                CodecId::None,
                scx_codec::ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        writer.write_var_for(rna_id, &sample_var(5)).unwrap();
        writer.write_var_for(adt_id, &sample_var(3)).unwrap();
        writer.set_modality_n_vars(rna_id, 5).unwrap();
        writer.set_modality_n_vars(adt_id, 3).unwrap();

        // One small CSR shard per modality so the file is well-formed.
        let indptr = vec![0u64, 1, 2, 3];
        let indices = vec![0u32, 1, 2];
        let values = vec![1u8, 2, 3];
        writer
            .write_csr_shard_for(
                rna_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                scx_codec::ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer
            .write_csr_shard_for(
                adt_id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                scx_codec::ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        if let Some(u) = &global_uns {
            writer.write_uns(u).unwrap();
        }
        if let Some(u) = &rna_uns {
            writer.write_uns_for(rna_id, u).unwrap();
        }
        writer.finish().unwrap();
        path
    }

    /// PR #68: `scx subset --modality NAME` (the `extract_modality`
    /// path) must preserve `uns`, preferring the per-modality
    /// `uns/{name}` over the file-wide `uns`, and must record a
    /// `modality_extract` provenance entry. The pre-fix path silently
    /// dropped both.
    #[test]
    fn test_extract_modality_prefers_modality_uns() {
        let dir = tempfile::tempdir().unwrap();
        let global = serde_json::json!({"source": "global", "version": 1});
        let rna = serde_json::json!({"source": "rna", "version": 42});
        let input = write_multimodal_test_file(&dir, Some(global), Some(rna));

        let output = dir.path().join("extracted_rna.scx");
        extract_modality(&input, &output, "rna", 10000, "none").unwrap();

        let reader = ScxReader::open(&output).unwrap();

        // Per-modality uns wins.
        let roundtrip = reader.read_uns().unwrap();
        assert_eq!(
            roundtrip["source"], "rna",
            "per-modality uns must win over global"
        );
        assert_eq!(roundtrip["version"], 42);

        // Provenance entry recorded.
        let prov = reader
            .read_provenance()
            .expect("extract_modality must write a provenance entry");
        let extract = prov
            .operations
            .iter()
            .find(|e| e.action == "modality_extract")
            .expect("provenance must include a modality_extract action");
        assert!(extract.params_json.contains("\"modality\""));
        assert!(extract.params_json.contains("\"rna\""));
    }

    /// Fallback: only the global `uns` is set on the input → output
    /// inherits that global uns (per-modality is absent for rna).
    #[test]
    fn test_extract_modality_falls_back_to_global_uns() {
        let dir = tempfile::tempdir().unwrap();
        let global = serde_json::json!({"source": "global", "version": 7});
        let input = write_multimodal_test_file(&dir, Some(global), None);

        let output = dir.path().join("extracted_rna_global.scx");
        extract_modality(&input, &output, "rna", 10000, "none").unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let roundtrip = reader.read_uns().unwrap();
        assert_eq!(roundtrip["source"], "global");
        assert_eq!(roundtrip["version"], 7);
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
            None,
            false,
            10000,
            "auto",
            false,
            5000,
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
