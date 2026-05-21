// scx query — Query cells by predicate with options for count, output, projection.

use std::io::{BufRead, BufReader};
use std::path::Path;

use scx_codec::ValueEncoding;
use scx_engine::{QueryPipeline, QueryResult};
use scx_format::header::FileHeader;
use scx_format::reader::ScxReader;
use scx_format::writer::ScxWriter;

/// Return true if `source` should be opened via `scx_cloud::open_cloud`.
///
/// Routes to the cloud path when:
///   - `source` carries a recognised URL scheme (`s3`, `gs`, `az`,
///     `azure`, `http`, `https`, `file`), or
///   - `source` is an existing local **directory** (typically an
///     exploded `.scxd/`), which `scx_cloud::open_cloud` handles via
///     the `LocalFileSystem` backend.
///
/// Plain regular files fall through to the local mmap path.
fn is_cloud_url(source: &str) -> bool {
    has_cloud_scheme(source) || Path::new(source).is_dir()
}

/// Return true if `source` carries a recognised remote URL scheme.
///
/// Narrower than [`is_cloud_url`]: this returns `false` for local
/// directories, which are routed through `LocalFileSystem` but are not
/// actually remote. Callers that need to gate behaviour on "round-trip
/// cost" (e.g. whether peeking a shard header is cheap) want this
/// helper, not `is_cloud_url`.
fn has_cloud_scheme(source: &str) -> bool {
    source.split_once("://").is_some_and(|(scheme, rest)| {
        !rest.is_empty()
            && matches!(
                scheme.to_ascii_lowercase().as_str(),
                "s3" | "gs" | "az" | "azure" | "http" | "https" | "file"
            )
    })
}

/// Format the `--explain` "Level 2 row eliminations" line, or `None`
/// when the inputs aren't interpretable as an elimination count.
///
/// Returns `None` when:
///   * `candidate_shard_rows == 0` — Level 1 took every shard, so the
///     line would divide by zero and carry no information.
///   * `matched_rows > candidate_shard_rows` — reachable when some
///     candidate shards lack catalog `stats`. `candidate_shard_rows`
///     drops those via `filter_map`
///     (`scx-engine::collect::candidate_shard_rows`) but `matched_rows`
///     still counts the rows they contributed, so the difference would
///     underflow `usize` (panic in debug, >100% in release).
fn format_level2_eliminations(candidate_shard_rows: usize, matched_rows: usize) -> Option<String> {
    if candidate_shard_rows == 0 {
        return None;
    }
    let eliminated = candidate_shard_rows.checked_sub(matched_rows)?;
    let pct = 100.0 * eliminated as f64 / candidate_shard_rows as f64;
    Some(format!(
        "  Level 2 row eliminations: {eliminated} ({pct:.1}% of candidate rows)"
    ))
}

#[allow(clippy::too_many_arguments)]
pub fn run_query(
    source: &str,
    filter: &str,
    count: bool,
    output: Option<&Path>,
    select_genes: Option<&Path>,
    normalize: Option<f64>,
    log1p: bool,
    limit: Option<usize>,
    json: bool,
    explain: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // Cloud URLs require the tokio runtime to outlive the pipeline
    // (the `CloudSectionReader` stores a `Handle` into it). Hold the
    // runtime in a guard binding tied to the function's lifetime.
    let opened = open_pipeline_for(source)?;
    let mut pipeline = opened.pipeline;
    #[cfg(feature = "cloud")]
    let _rt_guard = opened.runtime; // kept alive for the pipeline's lifetime
    pipeline = pipeline.filter_obs(filter)?;

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

    let n_cells = result.x.n_rows();
    let n_genes = result.x.n_cols();

    // Report pushdown stats. F4-2026-05-20-Tier2: two-line output that
    // separates Level 1 (catalog-stats) shard elimination from Level 2
    // (PredicateIndex / row-evaluator) row-mask narrowing. The previous
    // single line `Skipped 0/31 shards via pushdown` confused users when
    // every shard had at least one matching row (Level 1 skipped zero)
    // even though Level 2 was doing all the work.
    let candidate_shards = result.total_shards - result.skipped_shards;
    // matched_rows is the pre-limit
    // Level-2 match count; n_cells (= result.x.n_rows()) reflects post-
    // limit truncation, so reporting that as "matched" would underreport
    // whenever --limit truncates.
    eprintln!(
        "Pushdown: {}/{} shards eliminated by catalog stats (Level 1); \
         {} of {} candidate-shard rows matched (Level 2 index/row-eval)",
        result.skipped_shards,
        result.total_shards,
        result.matched_rows,
        result.candidate_shard_rows,
    );

    // F3-2026-05-20-Tier2: minimal --explain block. Prints the parsed
    // filter expression plus the Level 1 / Level 2 breakdown so power-
    // users can verify the predicate index is being engaged. No per-
    // predicate path tracing yet (deferred — multi-crate plumbing).
    if explain {
        eprintln!("Query plan (explain):");
        eprintln!("  source: {source}");
        eprintln!("  obs filter: {filter}");
        eprintln!("  total shards: {}", result.total_shards);
        eprintln!(
            "  Level 1 (catalog-stats) eliminated: {}/{}",
            result.skipped_shards, result.total_shards
        );
        eprintln!(
            "  Level 2 candidate shards: {}, candidate rows: {}",
            candidate_shards, result.candidate_shard_rows
        );
        eprintln!("  matched rows (pre-limit): {}", result.matched_rows);
        // F1-2026-05-21-Tier2: surface the Level-2 row-mask
        // elimination count directly rather than making the user
        // compute `candidate_rows - matched_rows` mentally. Skipped
        // when the numbers can't be interpreted as elimination — see
        // `format_level2_eliminations` for the guards.
        if let Some(line) =
            format_level2_eliminations(result.candidate_shard_rows, result.matched_rows)
        {
            eprintln!("{line}");
        }
        if result.matched_rows != n_cells {
            eprintln!("  returned rows (post-limit): {n_cells}");
        }
    }

    if count {
        if json {
            let json_out = serde_json::json!({
                "count": n_cells,
                "skipped_shards": result.skipped_shards,
                "total_shards": result.total_shards,
                "candidate_shard_rows": result.candidate_shard_rows,
                "matched_rows": result.matched_rows,
            });
            println!("{}", serde_json::to_string_pretty(&json_out)?);
        } else {
            println!("{}", n_cells);
        }
        return Ok(());
    }

    if let Some(out_path) = output {
        // Default to Float32 when normalize/log1p was applied (the
        // result is no longer integer-valued) or when the source is a
        // true remote URL (no easy way to peek a shard header without
        // a second round-trip). Local files AND local exploded
        // directories peek the first shard header.
        let value_encoding = if normalize.is_some() || log1p || has_cloud_scheme(source) {
            ValueEncoding::Float32
        } else {
            let path = Path::new(source);
            if path.is_dir() {
                detect_value_encoding_from_dir(path)?
            } else {
                detect_value_encoding(path)?
            }
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

    Ok(())
}

/// Pipeline plus the tokio runtime that owns its I/O thread pool.
///
/// The `runtime` slot is present only under `--features cloud`: it's
/// `Some` for cloud-backed pipelines (the runtime owns the threads
/// the pipeline's `block_on` calls land on) and `None` for local
/// pipelines. Outside the cloud feature, every pipeline is local, so
/// the field is omitted entirely.
struct OpenedPipeline {
    pipeline: QueryPipeline,
    #[cfg(feature = "cloud")]
    runtime: Option<std::sync::Arc<tokio::runtime::Runtime>>,
}

#[cfg(feature = "cloud")]
fn open_pipeline_for(source: &str) -> Result<OpenedPipeline, Box<dyn std::error::Error>> {
    if is_cloud_url(source) {
        // Multi-threaded runtime sized to match rayon's default pool
        // so that rayon-parallel shard decodes (each calling
        // `block_on`) can't starve the runtime they're waiting on.
        let worker_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(4);
        let rt = std::sync::Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(worker_threads)
                .enable_all()
                .thread_name("scx-cloud-io")
                .build()?,
        );
        let reader = rt.block_on(scx_cloud::open_cloud(source))?;
        let adapter = scx_cloud::CloudSectionReader::new(
            std::sync::Arc::new(reader),
            std::sync::Arc::clone(&rt),
        );
        let pipeline = QueryPipeline::from_reader(Box::new(adapter))?;
        Ok(OpenedPipeline {
            pipeline,
            runtime: Some(rt),
        })
    } else {
        Ok(OpenedPipeline {
            pipeline: QueryPipeline::open(Path::new(source))?,
            runtime: None,
        })
    }
}

#[cfg(not(feature = "cloud"))]
fn open_pipeline_for(source: &str) -> Result<OpenedPipeline, Box<dyn std::error::Error>> {
    if is_cloud_url(source) {
        return Err("cloud URLs require `scx-cli` to be built with `--features cloud`".into());
    }
    Ok(OpenedPipeline {
        pipeline: QueryPipeline::open(Path::new(source))?,
    })
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

/// Detect the ValueEncoding from the first CSR shard of a local
/// exploded `.scxd/` directory. Reads only the 76-byte shard header.
///
/// The exploded layout (see `scx_cloud::explode::section_name_to_path`)
/// places single-modality shards at `X/{idx:06}.shard` and
/// per-modality shards at `X/{modality}/{idx:06}.shard`. We probe the
/// canonical first shard `X/000000.shard`, and if absent, walk one
/// level deep to find any `*.shard` file under `X/`.
fn detect_value_encoding_from_dir(dir: &Path) -> Result<ValueEncoding, Box<dyn std::error::Error>> {
    use scx_format::shard::{ShardHeader, SHARD_HEADER_SIZE};
    use std::fs::File;
    use std::io::Read;

    let canonical = dir.join("X").join("000000.shard");
    let shard_path: std::path::PathBuf = if canonical.is_file() {
        canonical
    } else {
        // Walk one level deep under X/ looking for any *.shard file.
        let x_dir = dir.join("X");
        let mut found: Option<std::path::PathBuf> = None;
        if x_dir.is_dir() {
            'outer: for entry in std::fs::read_dir(&x_dir)? {
                let entry = entry?;
                let p = entry.path();
                if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("shard") {
                    found = Some(p);
                    break;
                }
                if p.is_dir() {
                    for sub in std::fs::read_dir(&p)? {
                        let sub = sub?;
                        let sp = sub.path();
                        if sp.is_file() && sp.extension().and_then(|e| e.to_str()) == Some("shard")
                        {
                            found = Some(sp);
                            break 'outer;
                        }
                    }
                }
            }
        }
        match found {
            Some(p) => p,
            None => return Ok(ValueEncoding::Uint16), // empty / no shards
        }
    };

    let mut f = File::open(&shard_path)?;
    let mut buf = [0u8; SHARD_HEADER_SIZE];
    f.read_exact(&mut buf)?;
    let sh = ShardHeader::read_from(&mut std::io::Cursor::new(&buf[..]))?;
    ValueEncoding::from_u8(sh.value_encoding)
        .ok_or_else(|| format!("unknown value encoding: {}", sh.value_encoding).into())
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
    let header = FileHeader {
        magic: scx_format::MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
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

        // Auto-codec selection (single-modality query → RNA default)
        let codec_id = scx_format::select_codec_for_modality(
            shard_values,
            value_encoding,
            scx_format::ModalityType::Rna,
        );

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
        tool: format!("scx {}", env!("CARGO_PKG_VERSION")),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_cloud_scheme_matches_known_schemes() {
        assert!(has_cloud_scheme("gs://bucket/path"));
        assert!(has_cloud_scheme("s3://bucket/path"));
        assert!(has_cloud_scheme("az://acct/container"));
        assert!(has_cloud_scheme("azure://acct/container"));
        assert!(has_cloud_scheme("http://example.com/data.scx"));
        assert!(has_cloud_scheme("https://example.com/data.scx"));
        assert!(has_cloud_scheme("file:///tmp/data.scx"));
        assert!(has_cloud_scheme("GS://Bucket/Path")); // case-insensitive
    }

    #[test]
    fn has_cloud_scheme_rejects_local_paths_and_dirs() {
        assert!(!has_cloud_scheme("/tmp/data.scx"));
        assert!(!has_cloud_scheme("./local.scxd"));
        assert!(!has_cloud_scheme("relative/path"));
        // Local directory paths must NOT trigger the cloud-scheme branch
        // (this is the Fix 1 regression: previously is_cloud_url returned
        // true for any local directory, forcing Float32 output even when
        // the source shards were integer-encoded).
        assert!(!has_cloud_scheme("/var/data/atlas.scxd"));
        // Empty / scheme-only inputs.
        assert!(!has_cloud_scheme("gs://"));
        assert!(!has_cloud_scheme("://"));
    }

    #[test]
    fn detect_value_encoding_from_dir_reads_first_shard_header() {
        use scx_codec::CodecId;
        use scx_format::shard::SHARD_HEADER_SIZE;

        let dir = tempfile::tempdir().unwrap();
        let x_dir = dir.path().join("X");
        std::fs::create_dir_all(&x_dir).unwrap();
        let shard_path = x_dir.join("000000.shard");

        // Synthesize a minimal valid ShardHeader with Uint8 encoding.
        let sh = scx_format::shard::ShardHeader {
            magic: scx_format::shard::SHARD_MAGIC,
            shard_format_version: 1,
            shard_type: 0,
            codec_id: CodecId::None as u8,
            value_encoding: ValueEncoding::Uint8 as u8,
            index_dtype: 1,
            reserved_flags: [0u8; 3],
            n_major: 1,
            n_minor: 1,
            nnz: 0,
            global_offset: 0,
            indptr_rel_offset: SHARD_HEADER_SIZE as u32,
            indptr_length: 0,
            indices_rel_offset: SHARD_HEADER_SIZE as u32,
            indices_length: 0,
            values_rel_offset: SHARD_HEADER_SIZE as u32,
            values_length: 0,
            block_index_rel_offset: SHARD_HEADER_SIZE as u32,
            block_index_length: 0,
            checksum: [0u8; 8],
        };
        let mut buf = Vec::with_capacity(SHARD_HEADER_SIZE);
        sh.write_to(&mut buf).unwrap();
        std::fs::write(&shard_path, &buf).unwrap();

        let encoding = detect_value_encoding_from_dir(dir.path()).unwrap();
        assert_eq!(encoding, ValueEncoding::Uint8);
    }

    #[test]
    fn detect_value_encoding_from_dir_handles_empty_dir() {
        // No shards present → conservative Uint16 default (matches the
        // local detect_value_encoding behaviour for shard-less files).
        let dir = tempfile::tempdir().unwrap();
        let encoding = detect_value_encoding_from_dir(dir.path()).unwrap();
        assert_eq!(encoding, ValueEncoding::Uint16);
    }

    #[test]
    fn level2_eliminations_formats_when_candidate_exceeds_matched() {
        let line = format_level2_eliminations(100, 30).expect("should format");
        assert!(line.contains("70"), "{line}");
        assert!(line.contains("70.0%"), "{line}");
    }

    #[test]
    fn level2_eliminations_skips_when_no_candidates() {
        assert!(format_level2_eliminations(0, 0).is_none());
    }

    #[test]
    fn level2_eliminations_skips_when_matched_exceeds_candidates() {
        // Mixed-catalog case: candidate_shard_rows drops stats-less
        // shards via filter_map, but matched_rows still counts their
        // rows. The naive subtraction would underflow usize.
        assert!(format_level2_eliminations(30, 100).is_none());
    }

    #[test]
    fn level2_eliminations_formats_zero_when_matched_equals_candidates() {
        let line = format_level2_eliminations(50, 50).expect("should format");
        assert!(line.contains(" 0 "), "{line}");
        assert!(line.contains("0.0%"), "{line}");
    }
}
