// scx query — Query cells by predicate with options for count, output, projection.

use std::io::{BufRead, BufReader};
use std::path::Path;

use scx_codec::ValueEncoding;
use scx_engine::{QueryPipeline, QueryResult};
use scx_format::header::FileHeader;
use scx_format::reader::ScxReader;
use scx_format::writer::ScxWriter;

#[allow(clippy::too_many_arguments)]
pub fn run_query(
    file: &Path,
    filter: &str,
    count: bool,
    output: Option<&Path>,
    select_genes: Option<&Path>,
    normalize: Option<f64>,
    log1p: bool,
    limit: Option<usize>,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Build pipeline
    let mut pipeline = QueryPipeline::open(file)?.filter_obs(filter)?;

    if let Some(gene_file) = select_genes {
        let indices = parse_gene_indices(gene_file)?;
        pipeline = pipeline.select_genes(indices);
    }

    if let Some(target_sum) = normalize {
        pipeline = pipeline.with_normalize(target_sum);
    }

    if log1p {
        pipeline = pipeline.with_log1p();
    }

    if let Some(n) = limit {
        pipeline = pipeline.limit(n);
    }

    // Execute
    let result = pipeline.collect()?;

    // Report pushdown stats
    eprintln!(
        "Skipped {}/{} shards via pushdown",
        result.skipped_shards, result.total_shards
    );

    let n_cells = result.x.n_rows();
    let n_genes = result.x.n_cols();

    if count {
        if json {
            let json_out = serde_json::json!({
                "count": n_cells,
                "skipped_shards": result.skipped_shards,
                "total_shards": result.total_shards,
            });
            println!("{}", serde_json::to_string_pretty(&json_out)?);
        } else {
            println!("{}", n_cells);
        }
        return Ok(());
    }

    if let Some(out_path) = output {
        // Determine value encoding: if normalize/log1p was applied, use Float32;
        // otherwise detect from original file.
        let value_encoding = if normalize.is_some() || log1p {
            ValueEncoding::Float32
        } else {
            detect_value_encoding(file)?
        };

        write_query_result(&result, out_path, value_encoding)?;
        println!("Wrote {} cells to {}", n_cells, out_path.display());
        return Ok(());
    }

    // Default: print summary
    let nnz = result.x.indptr.last().copied().unwrap_or(0);
    println!(
        "Query result: {} cells x {} genes, {} nnz",
        n_cells, n_genes, nnz
    );
    println!(
        "Pushdown: skipped {} of {} shards",
        result.skipped_shards, result.total_shards
    );

    Ok(())
}

/// Parse a gene index file: one u32 index per line, skip comments (#) and blank lines.
pub fn parse_gene_indices(path: &Path) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
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
    use scx_format::section::SectionType;
    use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};

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

/// Write a QueryResult to a new SCX file.
fn write_query_result(
    result: &QueryResult,
    output: &Path,
    value_encoding: ValueEncoding,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_obs = result.x.n_rows() as u64;
    let n_vars = result.x.n_cols() as u64;

    let index_dtype = if n_vars <= 65535 { 0u8 } else { 1u8 };

    // `flags: 0` and `n_csc_shards: 0` are intentional: query-result
    // materialization writes a freshly-projected CSR matrix; the
    // input file's CSC sidecar (if any) does not match the projected
    // row/column space, so we don't carry it forward. Re-run
    // `scx build-csc` against the output if a CSC sidecar is needed.
    // CSC-SUPPORT.md Phase I.3.
    let header = FileHeader {
        magic: scx_format::MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0, // filled in by finish()
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: 10000,
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
    let raw_values = f32_to_raw_values(&result.x.data, value_encoding);

    // Shard the data
    let shard_target = 10000usize;
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

        // Auto-codec selection
        let codec_id = scx_format::select_codec(shard_values, value_encoding);

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

    // Write provenance
    writer.write_provenance(vec![scx_format::ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "query".to_string(),
        tool: "scx-cli 0.1.0".to_string(),
        params_json: "{}".to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Convert f32 data to raw LE bytes matching the given ValueEncoding.
fn f32_to_raw_values(data: &[f32], encoding: ValueEncoding) -> Vec<u8> {
    match encoding {
        ValueEncoding::Uint8 => data.iter().map(|&v| v as u8).collect(),
        ValueEncoding::Uint16 => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&(v as u16).to_le_bytes());
            }
            bytes
        }
        ValueEncoding::Uint32 => {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &v in data {
                bytes.extend_from_slice(&(v as u32).to_le_bytes());
            }
            bytes
        }
        ValueEncoding::Float32 => {
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for &v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes
        }
        ValueEncoding::Float16 => {
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            bytes
        }
    }
}
