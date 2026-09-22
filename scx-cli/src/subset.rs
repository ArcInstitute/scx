// scx subset — Extract a subset of cells and/or genes into a new SCX file.

use std::io::{BufRead, BufReader};
use std::path::Path;

use scx_engine::{ConversionPredicateIndexOptions, ConversionPredicateIndexResult, QueryPipeline};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::reader::ScxReader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::FramingConfig;

/// Render a completed predicate-index build: the per-column outcomes through
/// the shared `merge` / `append` / `compact` renderer (so a missing preset
/// column reads identically across all four ops), then a summary of what was
/// actually indexed — mirroring `scx sort`'s line.
// Takes the result by value because `ConversionPredicateIndexResult` is not
// `Clone` and `PredicateIndexBuildSummary` owns its copy. Both callers are done
// with it by this point (provenance is written before `finish()`).
fn report_index_outcome(index_result: Option<ConversionPredicateIndexResult>) {
    if index_result.is_none() {
        return;
    }
    // `multimodal_skip` is None on both subset paths: the predicate-index
    // sections are unimodal-only and both outputs are single-modality files.
    let summary = scx_ops::PredicateIndexBuildSummary {
        result: index_result,
        multimodal_skip: None,
    };
    crate::index_warnings::emit_index_summary("subset", &summary);
    let result = summary
        .result
        .as_ref()
        .expect("populated from the early-return guard above");
    if !result.obs_indexed_columns.is_empty() {
        println!(
            "  indexed obs columns: {}",
            result.obs_indexed_columns.join(", ")
        );
    }
    if !result.var_indexed_columns.is_empty() {
        println!(
            "  indexed var columns: {}",
            result.var_indexed_columns.join(", ")
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_subset(
    input: &Path,
    output: Option<&Path>,
    force: bool,
    filter: Option<&str>,
    gene_file: Option<&Path>,
    modality: Option<&str>,
    dry_run: bool,
    shard_size: u32,
    codec: &str,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    // Refuse to clobber before any work, on every arm below. `--dry-run`
    // writes nothing, so an existing output is not a collision there.
    match (dry_run, output) {
        (false, Some(output)) => crate::cli_utils::guard_destination(
            &[input],
            crate::cli_utils::Destination::File(output),
            crate::cli_utils::SamePath::Reject,
            force,
        )?,
        (true, _) => crate::cli_utils::reject_inert_force(force, "--dry-run writes nothing")?,
        // The missing-output case is reported below, with the guidance the
        // caller actually needs.
        (false, None) => {}
    }

    // Pure modality extraction (no filter / no genes). The output is a
    // single-modality v2 file containing just the chosen modality's
    // CSR + var, with the file's global obs.
    if let Some(name) = modality {
        // Pure modality extraction keeps its stricter guards: dry-run is
        // unsupported and the output path is mandatory. Combined with
        // --filter / --genes the predicate path allows --dry-run and only
        // requires the output path when actually writing.
        if filter.is_none() && gene_file.is_none() {
            if dry_run {
                return Err("--dry-run is not supported for `--modality NAME` extraction".into());
            }
            if output.is_none() {
                return Err("an output path is required for `--modality NAME` extraction".into());
            }
        } else if !dry_run && output.is_none() {
            return Err("an output path is required (or use --dry-run)".into());
        }
        return extract_modality(
            input,
            output,
            name,
            filter,
            gene_file,
            dry_run,
            shard_size,
            codec,
            rebuild_csc,
            csc_cols_per_shard,
            csc_memory_limit,
            index_options,
        );
    }

    // 1. Validate: at least one of --filter or --genes must be specified
    if filter.is_none() && gene_file.is_none() {
        return Err("At least one of --filter or --genes is required".into());
    }

    // 2. If not dry-run, the output path is required
    if !dry_run && output.is_none() {
        return Err("an output path is required (or use --dry-run)".into());
    }

    // 3. Build query pipeline (single file open for all operations)
    let mut pipeline = QueryPipeline::open(input)?;

    if let Some(expr) = filter {
        pipeline = pipeline.filter_obs(expr)?;
    }

    // `scx subset` operates on local files only — it pulls metadata
    // that lives outside `SectionReader` (uns, layer names, on-disk
    // value encoding) from the underlying `ScxReader`. Collect
    // everything in one borrow scope; `pipeline.select_genes()` below
    // consumes `pipeline`, so the borrow must end before that point.
    let (in_header, dropped_layers, has_obs_pred_idx, has_var_pred_idx, uns, gene_indices_opt) = {
        let local_reader = pipeline
            .local_reader()
            .ok_or("scx subset requires a local SCX file")?;
        let gi = gene_file
            .map(|p| parse_gene_list(p, local_reader))
            .transpose()?;
        (
            local_reader.header().clone(),
            local_reader.layer_names(),
            local_reader
                .read_obs_predicate_index_bytes()
                .ok()
                .flatten()
                .is_some(),
            local_reader
                .read_var_predicate_index_bytes()
                .ok()
                .flatten()
                .is_some(),
            local_reader.read_uns().ok(),
            gi,
        )
    };
    let has_obsm = in_header.has_obsm();

    // 4. Apply gene projection (consumes & rebuilds pipeline)
    let gene_indices = if let Some(indices) = gene_indices_opt {
        pipeline = pipeline.select_genes(indices.clone());
        Some(indices)
    } else {
        None
    };

    // 5. Execute query (consumes pipeline)
    let result = pipeline.collect()?;

    // 7. Report stats
    let n_output_cells = result.x.n_rows();
    let n_output_genes = result.x.n_cols();
    let output_nnz = result.x.indptr.last().copied().unwrap_or(0);

    println!(
        "Subset: {}/{} cells, {}/{} genes, {} nnz",
        n_output_cells, in_header.n_obs, n_output_genes, in_header.n_vars, output_nnz,
    );

    // Validate any forced `--index-obs` / `--index-var` column against the
    // schemas the output will actually carry, BEFORE writing — so a typo costs
    // an error, not a half-written file. Runs on the dry-run path too: the
    // whole point of `--dry-run` is to learn what a real run would do.
    scx_ops::predicate_index::validate_forced_columns(
        index_options,
        result.obs.schema().as_ref(),
        result.var.schema().as_ref(),
    )?;

    if dry_run {
        println!("(dry run — no output written)");
        return Ok(());
    }

    // Warn about dropped sections. These describe what happens when the
    // output is *written*, so they only fire on a real run — a dry run
    // mutates nothing and must not claim otherwise.
    if !dropped_layers.is_empty() {
        eprintln!(
            "Warning: {} layer(s) dropped (layer subsetting not yet supported): {}",
            dropped_layers.len(),
            dropped_layers.join(", ")
        );
    }
    if has_obsm {
        eprintln!(
            "Warning: obsm/obsp embeddings dropped (not yet re-indexed on subset — \
             recompute downstream, or keep the full dataset and subset in Python)"
        );
    }
    // An input index cannot be carried over verbatim — row / column projection
    // invalidates every `ShardRange` in it. It CAN be rebuilt against the
    // output, which is what the `--index-*` flags do; say so, since without a
    // remedy this warning left "index once, hand out subsets" with no one-step
    // form (F7). Suppressed when a rebuild was requested — the outcome is
    // reported instead.
    if (has_obs_pred_idx || has_var_pred_idx)
        && !scx_ops::predicate_index::user_wants_index(index_options)
    {
        eprintln!(
            "Warning: predicate indices dropped (their shard ranges are invalid \
             after subsetting) — query-time `filter_obs` pushdown on {output} \
             falls back to a full obs scan. Pass --index-obs / --index-var / \
             --index-preset to rebuild them against the subset.",
            output = output.map(|p| p.display().to_string()).unwrap_or_default()
        );
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

    // 8. Resolve the codec intent. The full axis (`auto`/`fast`/`compact`/...),
    // not just `CodecId::parse_cli`'s explicit set: `auto` previously collapsed
    // to `None` here and the framing config below carried no `decode_target`,
    // so a subset silently ran `fast` and could nearly double bytes/nnz.
    let resolved_codec = scx_format_io::resolve_codec(Some(codec))?;
    let explicit_codec = resolved_codec.explicit_codec;

    // 9. Write output SCX file. Preserve framing from the source: a v4 (framed)
    // input yields a v4 framed output (default G) instead of a v3 downgrade.
    let framing = scx_ops::framing_for_rewrite(
        resolved_codec,
        in_header.format_version >= CURRENT_FORMAT_VERSION,
        "the input",
    )?;
    let output = output.unwrap();
    let index_result = write_subset_scx(
        output,
        &result,
        shard_size,
        explicit_codec,
        filter,
        gene_indices.as_deref(),
        uns.as_ref(),
        framing,
        index_options,
    )?;

    println!("Wrote {}", output.display());
    report_index_outcome(index_result);

    // Re-emit the CSC sidecar against the projected output. NOT the rewrite
    // `framing` above: under `--codec auto` that carries `decode_target:
    // Some(_)`, which authorises the writer to re-select each shard's codec and
    // so defeats the preservation a CSC rebuild depends on. `framing_for_file`
    // is the one correct source here — see its contract.
    if rebuild_csc {
        let csc_framing = crate::cli_utils::framing_for_file(output);
        // No `--temp-dir` on this op: the CSC builder's spill root defaults to the
        // output's own directory, which is already where the rewrite staged a whole
        // copy of it, and the spill only happens at all on an undersized
        // `--csc-memory-limit`.
        scx_ops::rebuild_csc_inplace(
            output,
            csc_cols_per_shard,
            csc_memory_limit,
            csc_framing,
            None,
        )?;
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

/// Resolve gene names to column indices against a pre-fetched var batch.
///
/// Looks up each name in the first string column of `var` (typically
/// `gene_id`). `context` is interpolated into error messages (e.g. `"var"`
/// or `"modality 'rna' var"`).
fn resolve_gene_names_from_var(
    names: &[String],
    var: &arrow::array::RecordBatch,
    context: &str,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
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
        .ok_or_else(|| format!("{context} has no string column for gene name resolution"))?;

    let col = var.column(col_idx);
    let string_array = col
        .as_any()
        .downcast_ref::<arrow::array::StringArray>()
        .ok_or_else(|| {
            format!(
                "{context} column '{}' is not a StringArray (type: {:?})",
                col_name,
                col.data_type()
            )
        })?;

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
            "{} gene name(s) not found in {context} column '{}': {}{}",
            missing.len(),
            col_name,
            shown.join(", "),
            suffix,
        )
        .into());
    }

    Ok(indices)
}

/// Resolve gene names against the global var (`modality_id == 0` /
/// single-modality). Thin wrapper around [`resolve_gene_names_from_var`].
fn resolve_gene_names(
    names: &[String],
    reader: &ScxReader,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let var = reader.read_var()?;
    resolve_gene_names_from_var(names, &var, "var")
}

/// Write an in-memory CSR matrix to `writer` as row-major shards, routing
/// each shard through the shared [`scx_format_io::encode_one_shard`] path that
/// `scx convert` and pyscx use.
///
/// Value encoding (uint8/uint16/uint32/float32) is auto-detected **per shard**
/// from the actual `f32` values, so a subset/projection that retains values
/// wider than the input file's first-shard encoding still writes a valid file
/// (B3). `index_dtype` (the file-wide gene-index width) is threaded through so
/// every shard header agrees with the file header.
///
/// Returns the emitted shards' `(row_start, row_end_exclusive)` ranges, in
/// ascending order — the convention `scx-ops`'s rewrite ops use for
/// `obs_row_ranges`. Predicate-index builders key `ShardRange.shard_id` to
/// these **output** CSR shards, so they must come from the write loop rather
/// than be re-derived from the input's layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_csr_shards_auto(
    writer: &mut ScxWriter,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_vars: u32,
    shard_size: u32,
    index_dtype: u8,
    explicit_codec: Option<scx_codec::CodecId>,
    modality_type: scx_format_io::ModalityType,
    framing: Option<scx_format_io::FramingConfig>,
) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
    let shard_target = shard_size as usize;
    debug_assert!(shard_target > 0, "shard_size must be greater than 0");
    // `indptr.len() - 1` would underflow on an empty indptr; `saturating_sub`
    // yields 0 rows (no shards written), matching the convert/from_mudata
    // row-sharding template.
    let total_rows = indptr.len().saturating_sub(1);
    let mut row_offset = 0usize;
    let mut shard_idx = 0usize;
    let mut row_ranges: Vec<(u64, u64)> = Vec::new();
    while row_offset < total_rows {
        let shard_rows = std::cmp::min(shard_target, total_rows - row_offset);
        let shard_indptr_start = indptr[row_offset];

        // Shard-local indptr, rebased to 0 (on-disk u64).
        let shard_indptr: Vec<u64> = indptr[row_offset..=row_offset + shard_rows]
            .iter()
            .map(|&v| (v - shard_indptr_start) as u64)
            .collect();

        let idx_start = shard_indptr_start as usize;
        let idx_end = indptr[row_offset + shard_rows] as usize;

        // Shard-local indices (on-disk u32) and raw f32 values — the encoder
        // detects the value encoding and selects the codec per shard.
        let shard_indices: Vec<u32> = indices[idx_start..idx_end]
            .iter()
            .map(|&v| v as u32)
            .collect();
        let shard_values = &data[idx_start..idx_end];

        let mut enc_opts = scx_format_io::EncodeShardOptions::new(
            format!("X_shard_{shard_idx}"),
            SectionType::CsrShard,
            n_vars as u64,
            row_offset as u64,
            index_dtype,
        );
        enc_opts.explicit_codec = explicit_codec;
        enc_opts.modality_type = modality_type;
        enc_opts.framing = framing;
        let pre = scx_format_io::encode_one_shard(
            &shard_indptr,
            &shard_indices,
            shard_values,
            &enc_opts,
        )?;
        writer.write_preencoded_shard(pre)?;

        row_ranges.push((row_offset as u64, (row_offset + shard_rows) as u64));
        row_offset += shard_rows;
        shard_idx += 1;
    }
    Ok(row_ranges)
}

/// `scx subset --modality NAME [--filter ... --genes ...]`.
///
/// Extract a single modality from a multimodal SCX file into a new
/// single-modality v2 file, optionally applying a row predicate and/or
/// gene projection in the same pass. Cells (obs) are global across
/// modalities, so on an input with no deletions and no filter the output's
/// obs matches the input's.
///
/// **Deletions are applied**, not carried: a subset materialises a new row
/// space, so logically deleted cells are dropped, intersected with `--filter`
/// rather than substituted for it. A deletion-bearing input therefore is *not*
/// a pure extraction even without `--filter` — it records a `subset`
/// provenance action, because rows are in fact being removed. With neither
/// `--filter`, `--genes`, nor deletions this is a pure modality extraction and
/// records `modality_extract`.
#[allow(clippy::too_many_arguments)]
fn extract_modality(
    input: &Path,
    output: Option<&Path>,
    modality_name: &str,
    filter: Option<&str>,
    gene_file: Option<&Path>,
    dry_run: bool,
    shard_size: u32,
    codec: &str,
    rebuild_csc: bool,
    csc_cols_per_shard: usize,
    csc_memory_limit: &str,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    use arrow::array::Array;

    // No filter and no gene list → pure modality extraction. Drives the
    // provenance action and the stdout summary below.
    let is_pure_extract = filter.is_none() && gene_file.is_none();

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
    let modality_type = info.modality_type;
    let n_vars = info.n_vars;

    // Read the global obs and apply the optional filter.
    let obs = reader.read_obs()?;
    let n_obs_global = reader.header().n_obs as usize;
    let mut row_mask: Option<Vec<bool>> = if let Some(expr) = filter {
        let schema = obs.schema();
        let pred = scx_engine::parse_predicate(expr, &schema, "obs")?;
        let bool_arr = scx_engine::evaluate(&pred, &obs)?;
        let mut mask = vec![false; n_obs_global];
        for (i, slot) in mask.iter_mut().enumerate() {
            if bool_arr.is_valid(i) && bool_arr.value(i) {
                *slot = true;
            }
        }
        Some(mask)
    } else {
        None
    };

    // Intersect the deletion keep mask, never substitute for it: a subset is
    // building a new row space, so deletions are applied here rather than
    // carried, and a logically deleted cell must not come back just because it
    // matched `--filter`. This path assembles from the physical readers instead
    // of going through the query engine, so — unlike the single-modality subset
    // — nothing else applies the mask for it.
    // `_for(modality_id)`, not the global mask: the v2 format defines a
    // modality's live rows as `global(0) || scoped(modality_id)`, and merge
    // already remaps every bitmap rather than just the global one. No shipped
    // writer emits scoped deletions yet, so today the two are equal — which is
    // precisely why the call should name the modality now, while the surface is
    // being written, rather than becoming a resurrection bug the day scoped
    // deletion is exposed.
    if let Some(keep) = reader.deletion_keep_mask_for(modality_id)? {
        // Both masks are sized from the same `n_obs`, so a mismatch is
        // impossible on a well-formed file — which is exactly why it must not
        // be a silent `zip`. A short `keep` would leave the tail of the filter
        // mask un-ANDed and resurrect the deleted rows it covers: the failure
        // this whole change exists to stop, reintroduced by the fix for it.
        // `filter_batch_by_keep_mask` refuses the same class for the same
        // reason.
        if keep.len() != n_obs_global {
            return Err(format!(
                "deletion keep mask has {} entries but the file declares n_obs={}; \
                 refusing to subset '{modality_name}' against a mask that does not \
                 cover every row",
                keep.len(),
                n_obs_global
            )
            .into());
        }
        match &mut row_mask {
            Some(mask) => {
                for (slot, &live) in mask.iter_mut().zip(keep.iter()) {
                    *slot &= live;
                }
            }
            None => row_mask = Some(keep),
        }
    }

    // A deletion-bearing input is not a pure extract even without `--filter`:
    // rows are being dropped, so the run should say so and record itself as a
    // `subset` rather than a `modality_extract`.
    let is_pure_extract = is_pure_extract && row_mask.is_none();

    // Parse the optional gene list (resolved against the modality var).
    let var = reader.read_var_for(modality_id)?;
    let gene_indices: Option<Vec<u32>> = if let Some(gene_path) = gene_file {
        let mut entries = Vec::new();
        for line in
            std::io::BufRead::lines(std::io::BufReader::new(std::fs::File::open(gene_path)?))
        {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            entries.push(trimmed.to_string());
        }
        let all_numeric: Option<Vec<u32>> = entries.iter().map(|e| e.parse::<u32>().ok()).collect();
        let mut indices = if let Some(num) = all_numeric {
            num
        } else {
            resolve_gene_names_from_var(&entries, &var, &format!("modality '{modality_name}' var"))?
        };
        indices.sort_unstable();
        indices.dedup();
        Some(indices)
    } else {
        None
    };

    // Read the modality's CSR (full assembly — simple and correct).
    // For very large modalities we could stream shards instead, but
    // subset compositions are typically targeted enough that this is
    // fine.
    let csr = reader.read_all_csr_shards_for(modality_id)?;
    let n_modality_rows = csr.shape.0;
    if n_modality_rows != n_obs_global {
        return Err(format!(
            "modality '{modality_name}' has {n_modality_rows} rows but global n_obs={n_obs_global}; \
             cannot subset"
        )
        .into());
    }

    // Apply row mask and gene projection.
    let filtered_csr = if let Some(ref mask) = row_mask {
        let kept: Vec<usize> = (0..n_obs_global).filter(|&i| mask[i]).collect();
        let mut new_indptr: Vec<i64> = Vec::with_capacity(kept.len() + 1);
        new_indptr.push(0);
        let mut new_indices: Vec<i32> = Vec::new();
        let mut new_data: Vec<f32> = Vec::new();
        for &row in &kept {
            let s = csr.indptr[row] as usize;
            let e = csr.indptr[row + 1] as usize;
            new_indices.extend_from_slice(&csr.indices[s..e]);
            new_data.extend_from_slice(&csr.data[s..e]);
            new_indptr.push(*new_indptr.last().unwrap() + (e - s) as i64);
        }
        scx_sparse::ScxCsr::new_unchecked(
            (kept.len(), csr.shape.1),
            new_indptr,
            new_indices,
            new_data,
        )
    } else {
        csr
    };

    let projected_csr = if let Some(ref idx) = gene_indices {
        scx_engine::project_csr(&filtered_csr, idx)
    } else {
        filtered_csr
    };

    // Build the filtered obs and projected var record batches.
    let filtered_obs = if let Some(ref mask) = row_mask {
        let bool_arr = arrow::array::BooleanArray::from(mask.clone());
        arrow::compute::filter_record_batch(&obs, &bool_arr)?
    } else {
        obs
    };
    let projected_var = if let Some(ref idx) = gene_indices {
        scx_engine::project_var(&var, idx)?
    } else {
        var
    };

    if !is_pure_extract {
        println!(
            "Subset (modality '{}'): {}/{} cells, {}/{} genes, {} nnz",
            modality_name,
            projected_csr.n_rows(),
            n_obs_global,
            projected_csr.n_cols(),
            n_vars,
            projected_csr.indptr.last().copied().unwrap_or(0),
        );
    }

    // Validate any forced `--index-obs` / `--index-var` column against the
    // schemas the output will actually carry, BEFORE writing — so a typo costs
    // an error, not a half-written file. Runs on the dry-run path too: the
    // whole point of `--dry-run` is to learn what a real run would do.
    scx_ops::predicate_index::validate_forced_columns(
        index_options,
        filtered_obs.schema().as_ref(),
        projected_var.schema().as_ref(),
    )?;

    if dry_run {
        println!("(dry run — no output written)");
        return Ok(());
    }

    let resolved_codec = scx_format_io::resolve_codec(Some(codec))?;
    let explicit_codec = resolved_codec.explicit_codec;

    let output = output.unwrap();
    let n_obs_out = projected_csr.n_rows() as u64;
    let n_vars_out = projected_csr.n_cols() as u64;
    let index_dtype = if n_vars_out <= 65535 { 0u8 } else { 1u8 };
    // Preserve framing from the source (a v4 input yields a framed v4 output).
    let framing = scx_ops::framing_for_rewrite(
        resolved_codec,
        reader.header().format_version >= CURRENT_FORMAT_VERSION,
        "the input",
    )?;
    let mut header =
        FileHeader::new_single_modality(n_obs_out, n_vars_out, 0, shard_size, 0, index_dtype);
    if framing.is_some() {
        header.format_version = CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    writer.write_obs(&filtered_obs)?;
    writer.write_var(&projected_var)?;

    let output_shard_row_ranges = write_csr_shards_auto(
        &mut writer,
        &projected_csr.indptr,
        &projected_csr.indices,
        &projected_csr.data,
        n_vars_out as u32,
        shard_size,
        index_dtype,
        explicit_codec,
        modality_type,
        framing,
    )?;

    // Preserve uns: prefer per-modality, fall back to global.
    let uns = reader
        .read_uns_for(modality_id)
        .ok()
        .or_else(|| reader.read_uns().ok());
    if let Some(u) = &uns {
        writer.write_uns(u)?;
    }

    // Predicate indexes (F7). Safe to write here even though the *input* is
    // multimodal: the output is a single-modality v2 file, and it is the output
    // being unimodal that the predicate-index sections require (see
    // `PredicateIndexBuildSummary::multimodal_skip`).
    let index_result = if scx_ops::predicate_index::user_wants_index(index_options) {
        Some(scx_engine::build_and_write_conversion_predicate_indexes(
            &mut writer,
            &filtered_obs,
            &projected_var,
            &output_shard_row_ranges,
            n_vars_out as usize,
            index_options,
        )?)
    } else {
        None
    };

    let (action, mut params) = if is_pure_extract {
        (
            "modality_extract",
            serde_json::json!({ "modality": modality_name }),
        )
    } else {
        (
            "subset",
            serde_json::json!({
                "modality": modality_name,
                "filter": filter,
                "n_genes": gene_indices.as_ref().map(|g| g.len()),
            }),
        )
    };
    if let Some(ref result) = index_result {
        params["predicate_index"] = serde_json::json!({
            "obs_columns": result.obs_indexed_columns,
            "var_columns": result.var_indexed_columns,
            "preset": index_options.index_preset,
        });
    }
    writer.write_provenance(vec![scx_format_io::ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: action.to_string(),
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: params.to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    if is_pure_extract {
        println!(
            "Extracted modality '{modality_name}' ({n_vars} vars) from {} to {}",
            input.display(),
            output.display()
        );
    } else {
        println!("Wrote {}", output.display());
    }
    report_index_outcome(index_result);

    if rebuild_csc {
        // See the note on the sibling site above: the rewrite framing would
        // re-authorise codec selection on the sidecar.
        let csc_framing = crate::cli_utils::framing_for_file(output);
        // No `--temp-dir` on this op: the CSC builder's spill root defaults to the
        // output's own directory, which is already where the rewrite staged a whole
        // copy of it, and the spill only happens at all on an undersized
        // `--csc-memory-limit`.
        scx_ops::rebuild_csc_inplace(
            output,
            csc_cols_per_shard,
            csc_memory_limit,
            csc_framing,
            None,
        )?;
        println!("Rebuilt CSC sidecar on {}", output.display());
    }
    Ok(())
}

/// Write a subset QueryResult to a new SCX file.
#[allow(clippy::too_many_arguments)]
fn write_subset_scx(
    output: &Path,
    result: &scx_engine::QueryResult,
    shard_size: u32,
    explicit_codec: Option<scx_codec::CodecId>,
    filter_expr: Option<&str>,
    gene_indices: Option<&[u32]>,
    uns: Option<&serde_json::Value>,
    framing: Option<FramingConfig>,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<Option<ConversionPredicateIndexResult>, Box<dyn std::error::Error>> {
    let n_obs = result.x.n_rows() as u64;
    let n_vars = result.x.n_cols() as u64;

    let index_dtype = if n_vars <= 65535 { 0u8 } else { 1u8 };

    // nnz filled in by finish(). Preserve row-group framing: a framed (v4)
    // source yields a v4 output whose shards are framed (via `framing` below),
    // so subsetting a default file no longer silently downgrades it to v3.
    let mut header = FileHeader::new_single_modality(n_obs, n_vars, 0, shard_size, 0, index_dtype);
    if framing.is_some() {
        header.format_version = CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;

    // Write obs and var metadata
    writer.write_obs(&result.obs)?;
    writer.write_var(&result.var)?;

    // Write X as row-major shards. Value encoding is auto-detected per shard
    // from the projected f32 values (so retained values wider than the input
    // file's first-shard encoding are handled — B3).
    let output_shard_row_ranges = write_csr_shards_auto(
        &mut writer,
        &result.x.indptr,
        &result.x.indices,
        &result.x.data,
        n_vars as u32,
        shard_size,
        index_dtype,
        explicit_codec,
        scx_format_io::ModalityType::Rna,
        framing,
    )?;

    // Write uns if present in the input file
    if let Some(uns_data) = uns {
        writer.write_uns(uns_data)?;
    }

    // Predicate indexes (F7): built against the *subset* obs/var and the shard
    // ranges just emitted, so `ShardRange.shard_id` addresses output shards.
    // Both batches are already in memory here (the query engine assembled
    // them), so the eager builder is the right entry point — no re-read.
    // Requested only via an explicit `--index-*` flag, matching `merge` /
    // `compact` / `sort`; `user_wants_index` also treats a non-zero
    // `--index-auto-threshold` as a request.
    let index_result = if scx_ops::predicate_index::user_wants_index(index_options) {
        Some(scx_engine::build_and_write_conversion_predicate_indexes(
            &mut writer,
            &result.obs,
            &result.var,
            &output_shard_row_ranges,
            n_vars as usize,
            index_options,
        )?)
    } else {
        None
    };

    // Build provenance params. `serde_json` rather than hand-rolled string
    // concatenation so the added `predicate_index` key cannot desync the
    // comma placement (and so a filter expression containing a quote is
    // escaped by the serializer rather than by hand).
    let mut params = serde_json::Map::new();
    if let Some(f) = filter_expr {
        params.insert("filter".to_string(), serde_json::json!(f));
    }
    if let Some(genes) = gene_indices {
        params.insert("n_genes".to_string(), serde_json::json!(genes.len()));
    }
    // Record which columns were actually indexed, mirroring the convert /
    // compact / sort entries, so a query-ready subset is self-describing.
    if let Some(ref result) = index_result {
        params.insert(
            "predicate_index".to_string(),
            serde_json::json!({
                "obs_columns": result.obs_indexed_columns,
                "var_columns": result.var_indexed_columns,
                "preset": index_options.index_preset,
            }),
        );
    }

    // Write provenance
    writer.write_provenance(vec![scx_format_io::ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "subset".to_string(),
        tool: format!("scx-cli {}", env!("CARGO_PKG_VERSION")),
        params_json: serde_json::Value::Object(params).to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(index_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::write_test_file;
    use scx_codec::{CodecId, ValueEncoding};

    /// No `--index-*` flag was passed: `index_auto_threshold: 0` disables
    /// auto-detection too, so `user_wants_index` is false and the output carries
    /// no predicate-index sections. This is the CLI default.
    fn no_index() -> ConversionPredicateIndexOptions {
        ConversionPredicateIndexOptions {
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        }
    }

    /// `--index-obs a,b` (and optionally `--index-var`).
    fn forced_index(obs: &[&str], var: &[&str]) -> ConversionPredicateIndexOptions {
        ConversionPredicateIndexOptions {
            index_obs: obs.iter().map(|s| s.to_string()).collect(),
            index_var: var.iter().map(|s| s.to_string()).collect(),
            index_preset: None,
            index_auto_threshold: 0,
        }
    }

    /// Names of the columns an on-disk obs predicate index actually covers.
    /// Reads the section back rather than trusting the build summary, so the
    /// assertions below are about the file a user would query.
    fn indexed_obs_columns(path: &std::path::Path) -> Vec<String> {
        let reader = ScxReader::open(path).unwrap();
        let Some(bytes) = reader.read_obs_predicate_index_bytes().unwrap() else {
            return Vec::new();
        };
        predicate_index_columns(bytes)
    }

    fn indexed_var_columns(path: &std::path::Path) -> Vec<String> {
        let reader = ScxReader::open(path).unwrap();
        let Some(bytes) = reader.read_var_predicate_index_bytes().unwrap() else {
            return Vec::new();
        };
        predicate_index_columns(bytes)
    }

    fn predicate_index_columns(bytes: &[u8]) -> Vec<String> {
        let index =
            scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
        index
            .columns
            .iter()
            .map(|c| match c {
                scx_engine::index::IndexedColumn::Categorical(cat) => cat.column_name.clone(),
                scx_engine::index::IndexedColumn::Numeric(num) => num.column_name.clone(),
            })
            .collect()
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

    /// Write a gene name file for testing.
    fn write_gene_name_file(dir: &tempfile::TempDir, names: &[&str]) -> std::path::PathBuf {
        let path = dir.path().join("gene_names.txt");
        std::fs::write(&path, names.join("\n")).unwrap();
        path
    }

    /// Write a **framed** (v4 / shard-v2) test input, mirroring
    /// `test_utils::write_test_file` but with a v4 header + row-group framing so
    /// the F-d framing-preservation path is exercised.
    fn write_framed_test_file(
        dir: &tempfile::TempDir,
        n_obs: usize,
        n_vars: usize,
    ) -> std::path::PathBuf {
        let path = dir.path().join("test_framed.scx");
        let mut header = crate::test_utils::sample_header(n_obs as u64, n_vars as u64);
        header.format_version = CURRENT_FORMAT_VERSION;
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.set_framing(Some(FramingConfig {
            row_group_rows: 4,
            ..Default::default()
        }));
        writer
            .write_obs(&crate::test_utils::sample_obs(n_obs))
            .unwrap();
        writer
            .write_var(&crate::test_utils::sample_var(n_vars))
            .unwrap();
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            let val0 = ((row + 1) % 256) as u8;
            let val1 = ((row + 2) % 256) as u8;
            let ((c0, v0), (c1, v1)) = if col0 <= col1 {
                ((col0, val0), (col1, val1))
            } else {
                ((col1, val1), (col0, val0))
            };
            indices.push(c0 as u32);
            indices.push(c1 as u32);
            values.push(v0);
            values.push(v1);
            indptr.push(indptr.last().unwrap() + 2);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::Zstd,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();
        path
    }

    /// F-d: subsetting a framed (v4) file must produce a framed v4 output, not
    /// silently downgrade it to unframed v3.
    #[test]
    fn test_subset_preserves_framing_v4() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_framed_test_file(&dir, 12, 5);
        let output = dir.path().join("subset_framed.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(
            reader.header().format_version,
            CURRENT_FORMAT_VERSION,
            "subsetting a framed file must keep the output v4"
        );
        for entry in &reader.catalog().shards_sorted() {
            let sh = reader.read_shard_header(entry).unwrap();
            assert!(
                sh.shard_format_version > 1,
                "subset output shard '{}' must be framed v2, got v{}",
                entry.name,
                sh.shard_format_version
            );
        }
    }

    /// Control: subsetting an unframed (v3) file must not over-frame to v4.
    #[test]
    fn test_subset_unframed_stays_v3() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 12, 5);
        let output = dir.path().join("subset_v3.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert!(
            reader.header().format_version < CURRENT_FORMAT_VERSION,
            "subsetting an unframed file must not over-frame to v4 (got v{})",
            reader.header().format_version
        );
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
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            None,
            Some(gene_file.as_path()),
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            Some("cell_type == 'T cell'"),
            Some(gene_file.as_path()),
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            true,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            None,
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        );
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        // Output is now a positional arg — the message must not tell users to
        // pass the removed `--output` flag (PR #260 review).
        assert!(
            msg.contains("an output path is required"),
            "expected positional-output wording, got: {msg}"
        );
        assert!(!msg.contains("--output"));
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
            false,
            None,
            Some(gene_file.as_path()),
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            None,
            Some(gene_file.as_path()),
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
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
        use scx_format_io::modality::ModalityType;

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
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            scx_codec::ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(rna_id, 0, shard).unwrap();
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            scx_codec::ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(adt_id, 0, shard).unwrap();

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
        extract_modality(
            &input,
            Some(output.as_path()),
            "rna",
            None,
            None,
            false,
            10000,
            "none",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

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
        extract_modality(
            &input,
            Some(output.as_path()),
            "rna",
            None,
            None,
            false,
            10000,
            "none",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let roundtrip = reader.read_uns().unwrap();
        assert_eq!(roundtrip["source"], "global");
        assert_eq!(roundtrip["version"], 7);
    }

    /// Phase 6: `scx subset --modality NAME --filter ...` extracts the
    /// chosen modality and applies the row predicate against the
    /// global obs. The output is a single-modality v2 SCX whose
    /// n_obs matches the number of cells the filter matched, and
    /// whose var matches the modality's var.
    #[test]
    fn test_subset_modality_with_filter() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_test_file(&dir, None, None);

        // sample_obs assigns cell_type cyclically T/B/NK over 3 cells:
        // index 0 → T cell, 1 → B cell, 2 → NK cell. Filter matches
        // exactly one row.
        let output = dir.path().join("subset_modality_filtered.scx");
        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            Some("rna"),
            false,
            10000,
            "none",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let header = reader.header();
        assert_eq!(header.n_obs, 1, "filter should keep exactly one cell");
        assert_eq!(header.n_vars, 5, "rna has 5 vars in the fixture");
        assert!(!reader.is_multimodal(), "output is single-modality v2");

        // The filter path (is_pure_extract == false) must record a
        // `subset` provenance action, mirroring the pure path's
        // `modality_extract` (see test_extract_modality_prefers_modality_uns).
        let prov = reader
            .read_provenance()
            .expect("subset must write a provenance entry");
        assert!(
            prov.operations.iter().any(|e| e.action == "subset"),
            "filter path must record a `subset` provenance action"
        );
    }

    /// `subset --modality` builds a new row space out of the *physical*
    /// readers, so nothing else applies the deletion mask for it — unlike the
    /// single-modality subset, which goes through the query engine.
    ///
    /// Both arms matter: without `--filter` the deletion mask is the only row
    /// mask there is, and with `--filter` it has to be intersected rather than
    /// replaced, or a deleted cell comes back the moment it matches a predicate.
    #[test]
    fn subset_modality_applies_deletions() {
        for filter in [None, Some("cell_type != 'NK cell'")] {
            let dir = tempfile::tempdir().unwrap();
            let input = write_multimodal_test_file(&dir, None, None);
            // sample_obs cycles T/B/NK over the 3 cells; delete the T cell,
            // which is one of the two the filter arm matches.
            scx_ops::mark_deleted(&input, &[0]).unwrap();

            let output = dir.path().join("subset_dv.scx");
            run_subset(
                &input,
                Some(output.as_path()),
                false,
                filter,
                None,
                Some("rna"),
                false,
                10000,
                "none",
                false,
                5000,
                "4G",
                &no_index(),
            )
            .unwrap();

            let reader = ScxReader::open(&output).unwrap();
            let expected = if filter.is_some() { 1 } else { 2 };
            assert_eq!(
                reader.header().n_obs,
                expected,
                "filter={filter:?}: the deleted cell must not be in the subset"
            );
            assert_eq!(reader.read_obs().unwrap().num_rows(), expected as usize);
            assert_eq!(
                reader.read_all_csr_shards().unwrap().shape.0,
                expected as usize,
                "filter={filter:?}: X and obs must drop the same rows"
            );
            // Deletions were applied, so the output carries none of its own.
            assert!(!reader.header().has_deletion_vectors());
        }
    }

    /// Phase 6: `scx subset --modality NAME --genes ...` extracts the
    /// chosen modality and projects to the named gene indices.
    #[test]
    fn test_subset_modality_with_genes() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_test_file(&dir, None, None);
        let gene_file = dir.path().join("genes.txt");
        std::fs::write(&gene_file, "0\n2\n").unwrap();

        let output = dir.path().join("subset_modality_genes.scx");
        run_subset(
            &input,
            Some(output.as_path()),
            false,
            None,
            Some(gene_file.as_path()),
            Some("rna"),
            false,
            10000,
            "none",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let header = reader.header();
        assert_eq!(header.n_obs, 3, "no filter → all cells preserved");
        assert_eq!(header.n_vars, 2, "two genes selected");
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
            false,
            None,
            Some(gene_path.as_path()),
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(
            reader.header().n_vars,
            2,
            "should have 2 genes after filtering comments"
        );
    }

    /// Build a 2-shard SCX file where shard 0 holds uint16-range values and
    /// shard 1 holds a value (66279) that overflows uint16. The first shard's
    /// header therefore reports a *narrower* encoding than some later shard —
    /// the exact precondition that made `scx subset` crash (B3) when it
    /// inherited the first shard's encoding for the whole output.
    ///
    /// All 4 rows get `cell_type == "KEEP"` so a `--filter` retains them.
    fn write_mixed_encoding_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        use arrow::array::{RecordBatch, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use scx_codec::ValueEncoding;
        use std::sync::Arc;

        let path = dir.path().join("mixed_enc.scx");
        let mut header = crate::test_utils::sample_header(4, 5);
        header.shard_target_rows = 2;
        let mut writer = ScxWriter::new(&path, header).unwrap();

        // obs: 4 cells, all cell_type "KEEP".
        let obs = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cell_id", DataType::Utf8, false),
                Field::new("cell_type", DataType::Utf8, true),
            ])),
            vec![
                Arc::new(StringArray::from(vec![
                    "cell_0", "cell_1", "cell_2", "cell_3",
                ])),
                Arc::new(StringArray::from(vec!["KEEP", "KEEP", "KEEP", "KEEP"])),
            ],
        )
        .unwrap();
        writer.write_obs(&obs).unwrap();
        writer.write_var(&crate::test_utils::sample_var(5)).unwrap();

        // Shard 0: rows 0,1 — values in (255, 65535] → Uint16.
        let s0_vals = ValueEncoding::Uint16
            .encode_f32_batch(&[300.0, 400.0, 500.0, 600.0])
            .unwrap();
        writer
            .write_csr_shard(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &s0_vals,
                CodecId::None,
                ValueEncoding::Uint16,
                0,
            )
            .unwrap();

        // Shard 1: rows 2,3 — contains 66279 (> u16 max) → Uint32.
        let s1_vals = ValueEncoding::Uint32
            .encode_f32_batch(&[66279.0, 5.0, 7.0, 8.0])
            .unwrap();
        writer
            .write_csr_shard(
                &[0, 2, 4],
                &[0, 1, 0, 1],
                &s1_vals,
                CodecId::None,
                ValueEncoding::Uint32,
                2,
            )
            .unwrap();

        writer.finish().unwrap();
        path
    }

    /// Collect the per-shard value encodings of all CSR shards in a file.
    fn output_value_encodings(path: &std::path::Path) -> Vec<scx_codec::ValueEncoding> {
        let reader = ScxReader::open(path).unwrap();
        reader
            .catalog()
            .shards(SectionType::CsrShard)
            .iter()
            .map(|e| {
                let sh = reader.read_shard_header(e).unwrap();
                scx_codec::ValueEncoding::from_u8(sh.value_encoding).unwrap()
            })
            .collect()
    }

    /// B3 regression: subsetting a file whose first shard is narrower than a
    /// later shard must auto-widen the output value encoding per shard rather
    /// than inheriting the first shard's width (which used to crash with
    /// `value 66279 out of range for uint16`).
    #[test]
    fn test_subset_auto_widens_value_encoding() {
        use scx_codec::ValueEncoding;
        let dir = tempfile::tempdir().unwrap();
        let input = write_mixed_encoding_file(&dir);

        // Anti-vacuous: confirm the input actually has mixed per-shard
        // encodings (shard 0 Uint16, shard 1 Uint32).
        assert_eq!(
            output_value_encodings(&input),
            vec![ValueEncoding::Uint16, ValueEncoding::Uint32],
            "fixture precondition: input must have mixed per-shard encodings"
        );

        let output = dir.path().join("subset_widened.scx");
        // shard_size = 2 keeps the small- and large-value rows in separate
        // output shards so we can assert per-shard differentiation.
        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'KEEP'"),
            None,
            None,
            false,
            2,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .expect("subset must succeed (pre-fix this returned `out of range for uint16`)");

        // (i) all rows retained, (ii) the >65535 value round-trips exactly.
        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(reader.header().n_obs, 4);
        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.indptr.last().copied().unwrap_or(0), 8, "nnz preserved");
        assert!(
            csr.data.contains(&66279.0),
            "the >u16 value must survive the subset round-trip"
        );

        // (iii) per-shard auto-detection: the small-value output shard stays
        // Uint16 while the shard holding 66279 widens to Uint32.
        let encs = output_value_encodings(&output);
        assert!(
            encs.contains(&ValueEncoding::Uint32),
            "output shard with 66279 must be Uint32, got {encs:?}"
        );
        assert!(
            encs.contains(&ValueEncoding::Uint16),
            "small-value output shard should stay Uint16, got {encs:?}"
        );
    }

    /// A subset whose predicate matches no rows must write a valid 0-row file
    /// (guards the `saturating_sub` empty-indptr path in `write_csr_shards_auto`).
    #[test]
    fn test_subset_zero_rows_ok() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_mixed_encoding_file(&dir);
        let output = dir.path().join("subset_empty.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'NONE'"),
            None,
            None,
            false,
            2,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .expect("0-row subset must not panic or error");

        let reader = ScxReader::open(&output).unwrap();
        assert_eq!(reader.header().n_obs, 0, "no rows should match");
        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.indptr.last().copied().unwrap_or(0), 0, "empty matrix");
    }

    // -----------------------------------------------------------------------
    // F7 — `--index-obs` / `--index-var` / `--index-preset` on the subset output
    // -----------------------------------------------------------------------

    /// The pair that matters: the same fixture and the same predicate, once
    /// without any `--index-*` flag and once with one. Only the contrast is
    /// evidence — asserting presence alone would also pass on a build that
    /// wrote an index unconditionally, and asserting absence alone would pass
    /// on a build that never writes one.
    #[test]
    fn test_subset_index_obs_is_written_only_when_requested() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 9, 5);

        let without = dir.path().join("no_index.scx");
        run_subset(
            &input,
            Some(without.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &no_index(),
        )
        .unwrap();
        assert!(
            indexed_obs_columns(&without).is_empty(),
            "no --index-* flag must leave the subset without predicate indexes \
             (parity with merge / compact / sort)"
        );

        let with = dir.path().join("with_index.scx");
        run_subset(
            &input,
            Some(with.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&["cell_type"], &[]),
        )
        .unwrap();
        assert_eq!(
            indexed_obs_columns(&with),
            vec!["cell_type".to_string()],
            "--index-obs cell_type must write an obs predicate index covering it"
        );
    }

    /// A `--genes` projection re-keys the gene axis, so a var index must be
    /// built against the *projected* var — not carried over from the input.
    #[test]
    fn test_subset_index_var_keyed_to_projected_genes() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 5);
        let gene_file = write_gene_file(&dir, &[0, 2]);
        let output = dir.path().join("var_indexed.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            None,
            Some(gene_file.as_path()),
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&[], &["gene_id"]),
        )
        .unwrap();

        assert_eq!(
            indexed_var_columns(&output),
            vec!["gene_id".to_string()],
            "--index-var must write a var predicate index"
        );

        // The index must describe the 2 retained genes, not the input's 5.
        let reader = ScxReader::open(&output).unwrap();
        let bytes = reader.read_var_predicate_index_bytes().unwrap().unwrap();
        let index =
            scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
        let scx_engine::index::IndexedColumn::Categorical(cat) = &index.columns[0] else {
            panic!("gene_id is a string column, so it must index as categorical");
        };
        let values: Vec<&str> = cat.entries.iter().map(|e| e.value.as_str()).collect();
        assert_eq!(
            values,
            vec!["gene_0", "gene_2"],
            "the var index must cover the projected genes only"
        );
    }

    /// `ShardRange.shard_id` addresses **output** CSR shards, so the ranges fed
    /// to the builder have to come from the write loop. A single-shard output
    /// cannot distinguish a correct id from a hardcoded 0, so force two.
    #[test]
    fn test_subset_index_shard_ranges_follow_output_shards() {
        let dir = tempfile::tempdir().unwrap();
        // 9 rows, every 3rd is "T cell" -> 3 retained rows; shard_size 2 splits
        // them across 2 output shards.
        let input = write_test_file(&dir, 9, 5);
        let output = dir.path().join("multishard_index.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            2,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&["cell_id"], &[]),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let n_csr_shards = reader.catalog().shards(SectionType::CsrShard).len();
        assert!(
            n_csr_shards >= 2,
            "fixture precondition: shard_size=2 must produce >=2 output shards, got {n_csr_shards}"
        );

        let bytes = reader.read_obs_predicate_index_bytes().unwrap().unwrap();
        let index =
            scx_engine::PredicateIndex::read_from(&mut std::io::Cursor::new(bytes)).unwrap();
        let scx_engine::index::IndexedColumn::Categorical(cat) = &index.columns[0] else {
            panic!("cell_id is a string column");
        };
        // Every referenced shard id must be a real output shard, and the set
        // must span more than one — proving the ranges track the write loop
        // rather than collapsing onto shard 0.
        let mut seen: Vec<u32> = cat
            .entries
            .iter()
            .flat_map(|e| e.shard_ranges.iter().map(|r| r.shard_id))
            .collect();
        seen.sort_unstable();
        seen.dedup();
        assert!(
            seen.iter().all(|&id| (id as usize) < n_csr_shards),
            "shard ids {seen:?} must all address output CSR shards (n={n_csr_shards})"
        );
        assert!(
            seen.len() >= 2,
            "the 3 retained cells span 2 output shards, so the index must \
             reference both; got {seen:?}"
        );
    }

    /// A typo'd forced column must fail before anything is written — that is
    /// what the upfront `validate_forced_columns` call buys over the engine's
    /// post-write `ForcedColumnError` outcome.
    #[test]
    fn test_subset_missing_forced_index_column_errors_without_writing() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 9, 5);
        let output = dir.path().join("never_written.scx");

        let err = run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&["nonexistent_column"], &[]),
        );
        assert!(err.is_err(), "a missing --index-obs column must error");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("nonexistent_column"),
            "the error must name the missing column, got: {msg}"
        );
        assert!(
            !output.exists(),
            "the output must not exist after a rejected --index-obs column"
        );
    }

    /// Same check on `--dry-run`: learning what a real run would do is the
    /// point of the flag, so the column typo has to surface there too.
    #[test]
    fn test_subset_dry_run_validates_forced_index_columns() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 9, 5);

        let err = run_subset(
            &input,
            None,
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            true,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&["nonexistent_column"], &[]),
        );
        assert!(
            err.is_err(),
            "--dry-run must report a bad --index-obs column rather than \
             reporting success and failing on the real run"
        );
        assert!(format!("{}", err.unwrap_err()).contains("nonexistent_column"));

        // Control: the same dry run with a valid column still succeeds.
        run_subset(
            &input,
            None,
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            true,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&["cell_type"], &[]),
        )
        .unwrap();
    }

    /// `--modality NAME` writes a single-modality output, which is exactly the
    /// condition the predicate-index sections require — so the extraction path
    /// can index too, even though its *input* is multimodal.
    #[test]
    fn test_subset_modality_extract_writes_index() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_test_file(&dir, None, None);
        let output = dir.path().join("rna_indexed.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            None,
            None,
            Some("rna"),
            false,
            10000,
            "none",
            false,
            5000,
            "4G",
            &forced_index(&["cell_type"], &[]),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        assert!(!reader.is_multimodal(), "output is single-modality v2");
        drop(reader);
        assert_eq!(
            indexed_obs_columns(&output),
            vec!["cell_type".to_string()],
            "a modality extraction must be able to carry a predicate index"
        );
    }

    /// `--index-auto-threshold N` **alone**, with no forced column and no
    /// preset, must still build an index.
    ///
    /// Worth its own test because this exact combination was historically a
    /// silent no-op on `compact` (its `user_wants_index` call site carries a
    /// comment recording the bug). `subset` routes through the same helper, so
    /// pin the behaviour here rather than inherit it on trust.
    #[test]
    fn test_subset_index_auto_threshold_alone_builds_an_index() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 9, 5);
        let output = dir.path().join("auto_threshold.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &ConversionPredicateIndexOptions {
                index_obs: Vec::new(),
                index_var: Vec::new(),
                index_preset: None,
                index_auto_threshold: 100,
            },
        )
        .unwrap();

        let columns = indexed_obs_columns(&output);
        assert!(
            !columns.is_empty(),
            "--index-auto-threshold alone must auto-detect low-cardinality obs \
             columns, not silently no-op"
        );
        assert!(
            columns.iter().any(|c| c == "cell_type"),
            "cell_type is low-cardinality and should be auto-detected; got {columns:?}"
        );
    }

    /// The indexed columns are recorded in provenance, mirroring convert /
    /// compact / sort, so a query-ready subset is self-describing.
    #[test]
    fn test_subset_records_indexed_columns_in_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 9, 5);
        let output = dir.path().join("prov_index.scx");

        run_subset(
            &input,
            Some(output.as_path()),
            false,
            Some("cell_type == 'T cell'"),
            None,
            None,
            false,
            10000,
            "auto",
            false,
            5000,
            "4G",
            &forced_index(&["cell_type"], &[]),
        )
        .unwrap();

        let reader = ScxReader::open(&output).unwrap();
        let prov = reader.read_provenance().unwrap();
        let entry = prov
            .operations
            .iter()
            .find(|e| e.action == "subset")
            .expect("subset must record a provenance entry");
        let params: serde_json::Value = serde_json::from_str(&entry.params_json).unwrap();
        assert_eq!(
            params["predicate_index"]["obs_columns"],
            serde_json::json!(["cell_type"]),
            "provenance must record which obs columns were indexed; got {}",
            entry.params_json
        );
        // The pre-existing keys must survive the switch to serde_json.
        assert_eq!(params["filter"], "cell_type == 'T cell'");
    }
}
