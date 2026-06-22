// scx query — Query cells by predicate with options for count, output, projection.

use std::io::{BufRead, BufReader};
use std::path::Path;

use scx_engine::{QueryPipeline, QueryResult};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;

use crate::cloud_url::is_cloud_url;

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
    filter: Option<&str>,
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

    // No predicate => query over all cells (mirrors pyscx `query()` with no
    // `filter_obs`). The engine already supports an empty predicate set.
    if let Some(f) = filter {
        pipeline = pipeline.filter_obs(f)?;
    }

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

    // Count-only fast path (CLI2): plan + mask without decoding X. `count()`
    // ignores `limit` entirely, so `--count --limit` reports the true matched
    // count rather than `min(matched, limit)` (CLI6).
    if count {
        let c = pipeline.count()?;
        let candidate_shards = c.total_shards - c.skipped_shards;
        eprintln!(
            "Pushdown: {}/{} shards eliminated by catalog stats (Level 1); \
             {} of {} candidate-shard rows matched (Level 2 index/row-eval)",
            c.skipped_shards, c.total_shards, c.matched_rows, c.candidate_shard_rows,
        );
        if explain {
            eprintln!("Query plan (explain):");
            eprintln!("  source: {source}");
            eprintln!("  obs filter: {}", filter.unwrap_or("(none — all cells)"));
            eprintln!("  total shards: {}", c.total_shards);
            eprintln!(
                "  Level 1 (catalog-stats) eliminated: {}/{}",
                c.skipped_shards, c.total_shards
            );
            eprintln!(
                "  Level 2 candidate shards: {}, candidate rows: {}",
                candidate_shards, c.candidate_shard_rows
            );
            eprintln!("  matched rows: {}", c.matched_rows);
            if let Some(line) = format_level2_eliminations(c.candidate_shard_rows, c.matched_rows) {
                eprintln!("{line}");
            }
        }
        if json {
            let json_out = serde_json::json!({
                "count": c.matched_rows,
                "skipped_shards": c.skipped_shards,
                "total_shards": c.total_shards,
                "candidate_shard_rows": c.candidate_shard_rows,
                "matched_rows": c.matched_rows,
            });
            println!("{}", serde_json::to_string_pretty(&json_out)?);
        } else {
            println!("{}", c.matched_rows);
        }
        return Ok(());
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
        eprintln!("  obs filter: {}", filter.unwrap_or("(none — all cells)"));
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

    if let Some(out_path) = output {
        // The value encoding is auto-detected per output shard from the
        // actual `f32` values (see `write_query_result` →
        // `write_csr_shards_auto`), so a result whose values are wider
        // than the source's first-shard encoding — or non-integer after
        // normalize/log1p — is written losslessly without first-shard
        // guesswork.
        write_query_result(&result, out_path)?;
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

/// Write a QueryResult to a new SCX file.
fn write_query_result(
    result: &QueryResult,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_obs = result.x.n_rows() as u64;
    let n_vars = result.x.n_cols() as u64;

    let index_dtype = if n_vars <= 65535 { 0u8 } else { 1u8 };

    // `flags: 0` and `n_csc_shards: 0` are intentional: query-result
    // materialization writes a freshly-projected CSR matrix; the
    // input file's CSC sidecar (if any) does not match the projected
    // row/column space, so we don't carry it forward. Re-run
    // `scx build-csc` against the output if a CSC sidecar is needed.
    // nnz filled in by finish().
    let header = FileHeader::new_single_modality(n_obs, n_vars, 0, 10000, 0, index_dtype);

    let mut writer = ScxWriter::new(output, header)?;

    // Write obs and var metadata
    writer.write_obs(&result.obs)?;
    writer.write_var(&result.var)?;

    // Write X as row-major shards through the shared per-shard auto-encode
    // path (same as `scx subset` / `scx convert`). The value encoding and
    // codec are detected per shard from the actual `f32` values, so a result
    // whose values are wider than any single source shard's encoding is
    // written losslessly, and normalize/log1p float results auto-route to
    // Float32.
    crate::subset::write_csr_shards_auto(
        &mut writer,
        &result.x.indptr,
        &result.x.indices,
        &result.x.data,
        n_vars as u32,
        10_000,
        index_dtype,
        None,
        scx_format_io::ModalityType::Rna,
    )?;

    // Write provenance
    writer.write_provenance(vec![scx_format_io::ProvenanceEntry {
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

#[cfg(test)]
mod tests {
    use super::*;

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

    // Regression: `scx query --output` must not truncate values wider than the
    // source's first-shard encoding. `write_query_result` now auto-detects the
    // encoding per output shard from the actual `f32` values, so a value beyond
    // Uint16's 65_535 ceiling is written as Uint32 and round-trips losslessly
    // (the old first-shard-peek + `v as u16` path wrapped it).
    #[test]
    fn write_query_result_preserves_values_wider_than_uint16() {
        use crate::test_utils::{sample_obs, sample_var};
        use scx_sparse::ScxCsr;

        // 2 cells × 3 genes; nonzeros span Uint8 / Uint16 / Uint32 ranges.
        let x = ScxCsr::new(
            (2, 3),
            vec![0, 2, 3],
            vec![0, 2, 1],
            vec![5.0, 70_000.0, 130_000.0],
        )
        .unwrap();
        let result = QueryResult {
            x,
            obs: sample_obs(2),
            var: sample_var(3),
            skipped_shards: 0,
            total_shards: 1,
            candidate_shard_rows: 2,
            matched_rows: 2,
        };

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("query_out.scx");
        write_query_result(&result, &out).unwrap();

        // Reopen via the engine (unfiltered collect = all rows) and confirm the
        // full value set survived without truncation/wrapping.
        let roundtrip = QueryPipeline::open(&out).unwrap().collect().unwrap();
        let mut got: Vec<f32> = roundtrip.x.data.clone();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            got,
            vec![5.0, 70_000.0, 130_000.0],
            "wide values must round-trip losslessly"
        );
    }
}
