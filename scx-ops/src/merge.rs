// Merge operation: combine multiple SCX files into one.

use std::path::Path;

use arrow::array::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use crate::append::unify_dict_columns;
use crate::codec_intent::{framing_for_rewrite, seed_codec};
use crate::error::{OpsError, Result};
use crate::flock::SharedFileLock;
use crate::helpers::{encode_value, widest_value_encoding};
use crate::merge_options::MergeOptions;
use crate::merge_pairwise;
use crate::merge_sorted;
use crate::predicate_index::{
    requested_columns, user_wants_index, validate_forced_columns, PredicateIndexBuildSummary,
};
use crate::rewrite_helpers::{build_raw_copied_csr_section, raw_copy_csr_eligible};

/// Merge multiple SCX files into a single output file.
/// All inputs must have the same n_vars.
///
/// Drops `ObsPredicateIndex` / `VarPredicateIndex` sections from the
/// output (the merged row layout invalidates per-shard row ranges).
/// Use [`merge_with_index_options`] to rebuild predicate indexes on the
/// merged file in the same pass.
pub fn merge(input_paths: &[&Path], output_path: &Path) -> Result<()> {
    // `index_auto_threshold = 0` is the sentinel that disables all
    // index work (no forced columns, no preset, no auto-detect). The
    // bare `merge(...)` entry now also enforces the default var- and
    // obs-identity checks via `MergeOptions::default()` — pre-strict
    // pipelines opt back in via the richer entry point.
    merge_with_index_options(
        input_paths,
        output_path,
        &ConversionPredicateIndexOptions {
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
    )
    .map(|_| ())
}

/// Merge multiple SCX files and optionally rebuild predicate indexes
/// on the output.
///
/// When `index_options` requests one or more obs / var columns (or
/// names a preset), the helper writes the matching `ObsPredicateIndex`
/// / `VarPredicateIndex` sections after the obs / var sections and
/// before `provenance`. The returned [`PredicateIndexBuildSummary`]
/// carries per-axis outcomes so the caller can emit user-facing warnings
/// on its own channel (see `scx-convert::pipeline::process_predicate_index_outcomes`
/// for the reference outcome → `ConvertWarning` mapping).
///
/// Multimodal merge currently cannot persist predicate-index sections
/// per modality — the helper sets `summary.multimodal_skip` and leaves it
/// to the caller to surface (e.g. as `scx-convert`'s
/// `ConvertWarning::PredicateIndexSkippedMultimodal`).
pub fn merge_with_index_options(
    input_paths: &[&Path],
    output_path: &Path,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<PredicateIndexBuildSummary> {
    merge_with_options(
        input_paths,
        output_path,
        &MergeOptions::legacy_with_index_options(index_options.clone()),
    )
}

/// Richest merge entry point. Accepts the full [`MergeOptions`]
/// surface: predicate-index configuration, var-identity strictness,
/// obs-schema strictness, uns-conflict policy, and shard-target
/// overrides. The flat [`merge_with_index_options`] / [`merge`]
/// entry points delegate to this function with the legacy defaults
/// baked in.
///
/// Peak RSS: legacy single-section obs inputs go through
/// [`input_obs_chunks`], which calls `read_obs()` once per input to
/// load the full obs batch before slicing it into shard-sized
/// chunks. Peak memory is therefore bounded by the largest
/// individual input's obs payload — not by the merged total — and
/// the output is still emitted shard-by-shard. Already-sharded
/// inputs stream without that one-shot read. Mixed inputs hit the
/// worst case only for their legacy members.
pub fn merge_with_options(
    input_paths: &[&Path],
    output_path: &Path,
    options: &MergeOptions,
) -> Result<PredicateIndexBuildSummary> {
    // Sorted merge auto-indexes the sort key so its
    // now-contiguous `shard_ranges` are emitted — but only when the caller
    // already asked for a predicate index. Non-sorted (or no-index) merges
    // keep the original options unchanged (byte-identical concat path).
    let augmented_index_options;
    let index_options: &ConversionPredicateIndexOptions =
        if !options.sort_by.is_empty() && user_wants_index(&options.index_options) {
            let mut io = options.index_options.clone();
            for key in &options.sort_by {
                if !io.index_obs.iter().any(|c| c == key) {
                    io.index_obs.push(key.clone());
                }
            }
            augmented_index_options = io;
            &augmented_index_options
        } else {
            &options.index_options
        };
    if input_paths.is_empty() {
        return Err(OpsError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no input files provided",
        )));
    }

    // Acquire shared locks on all inputs to prevent concurrent writers
    // from modifying files while we read them. Locks held until `_locks` dropped.
    let _locks: Vec<SharedFileLock> = input_paths
        .iter()
        .map(|p| SharedFileLock::acquire(p))
        .collect::<Result<Vec<_>>>()?;

    // Open all inputs
    let readers: Vec<ScxReader> = input_paths
        .iter()
        .map(|p| ScxReader::open(p).map_err(OpsError::Format))
        .collect::<Result<Vec<_>>>()?;

    // Validate n_vars consistency
    let n_vars = readers[0].n_vars();
    for (_i, reader) in readers.iter().enumerate().skip(1) {
        if reader.n_vars() != n_vars {
            return Err(OpsError::IncompatibleVars {
                expected: n_vars,
                found: reader.n_vars(),
            });
        }
    }

    // Sorted k-way merge: `sort_by` reorders the
    // merged obs axis globally instead of concatenating.
    let sorted = !options.sort_by.is_empty();

    // Phase 6: validate modality structure consistency. Multimodal
    // merge requires every input to expose the same set of
    // modalities (name + type + n_vars). On mismatch, raise with a
    // clear error directing to extract-then-merge.
    let any_multimodal = readers.iter().any(|r| r.is_multimodal());
    if any_multimodal {
        if sorted {
            return Err(OpsError::InvalidInput(
                "merge --sort-by does not yet support multimodal inputs (Phase 5); \
                 merge without --sort-by, or sort a single-modality extract"
                    .into(),
            ));
        }
        let first = readers[0].modality_table();
        for (i, reader) in readers.iter().enumerate().skip(1) {
            let here = reader.modality_table();
            if !modality_tables_match(first, here) {
                return Err(OpsError::ModalityMismatch {
                    detail: format!(
                        "input 0 and input {i} have different modality structures \
                         (name / modality_type / n_vars must match across all inputs); \
                         use `scx subset --modality NAME` on each input to extract a \
                         single modality first, then `scx merge` the single-modality files"
                    ),
                });
            }
        }
        // Phase 6: dispatch to multimodal merge — concatenate the
        // global obs row axis and per-modality CSR shards in input
        // order, preserving each modality's var. Predicate indexes
        // are unimodal-only today; record the skip so the caller can
        // emit a single `PredicateIndexSkippedMultimodal` warning.
        let multimodal_skip =
            user_wants_index(index_options).then(|| requested_columns(index_options));
        merge_multimodal(&readers, input_paths, output_path, options)?;
        return Ok(PredicateIndexBuildSummary {
            result: None,
            multimodal_skip,
        });
    }

    let total_n_obs: u64 = readers.iter().map(|r| r.n_obs()).sum();

    let first_header = readers[0].header();

    // Merge concatenates every input's rows, so a deleted cell must stay
    // deleted in the merged row space — see `remap_deletion_vectors`, called
    // before `finish()` on both the sorted and the concatenating path. Carried
    // rather than applied: merge is not a compaction, and the reasoning
    // `optimize` documents applies here too — the caller runs `scx compact` when
    // they want the rows physically gone.

    // CSC sidecars are dropped on merge: row layout is
    // re-concatenated across inputs, so any per-input CSC `indices`
    // arrays would reference stale row indices in the merged output.
    // Caller can opt back in via `--rebuild-csc` on the CLI.
    // Phase H.3.
    let any_input_had_csc = readers.iter().any(|r| r.header().has_csc());
    if any_input_had_csc {
        let inputs_str = input_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        log::warn!(
            "merge dropped CSC shards from at least one input ({inputs_str}): \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar on the merged output"
        );
    }

    // Raw (`adata.raw`) is not carried through merge: the raw obs axis would
    // need to be concatenated in lockstep with X across inputs, which is not
    // yet implemented. Warn loudly rather than drop it silently (SCX-002),
    // matching compact/sort behavior.
    if readers.iter().any(|r| r.header().has_raw()) {
        log::warn!(
            "merge: at least one input carries an adata.raw matrix, which is not \
             yet preserved through merge — raw will be dropped from the merged output"
        );
    }

    // Build output header. codec_id is 0 (None) because actual codec is
    // selected per-shard via select_codec(). `flags` stays 0 — merge
    // intentionally produces a clean output header rather than carrying
    // input flag state forward.
    // Merge re-encodes shards via `encode_one_shard` (no re-canonicalization),
    // so the v3 canonical-CSR invariant only holds if every input already
    // guarantees it. Gate the stamp on the minimum input version (floor 1 —
    // this is the single-modality merge).
    let input_format_versions: Vec<u16> =
        readers.iter().map(|r| r.header().format_version).collect();
    // Preserve row-group framing when every input is already framed (v4): stamp
    // v4 and enable framing on the writer. Framed source shards then byte-copy
    // verbatim (block index in-body), and any decode-encoded shard (var/dtype
    // mismatch) is re-emitted framed by `write_csr_shard` — so the output stays
    // a valid v4 file (every CSR shard framed). A mixed v3/v4 input set stays
    // unframed v3 (a v3 input holds no framed shards; a v4 input's framed shards
    // then decode-encode to unframed via the eligibility gate below).
    let output_framed = !input_format_versions.is_empty()
        && input_format_versions
            .iter()
            .all(|&v| v >= CURRENT_FORMAT_VERSION);
    let out_format_version = if output_framed {
        CURRENT_FORMAT_VERSION
    } else {
        scx_format_io::rewrite_output_format_version(&input_format_versions, 1)
    };
    let out_header = FileHeader {
        format_version: out_format_version,
        n_obs: total_n_obs,
        n_vars,
        shard_target_rows: first_header.shard_target_rows,
        index_dtype: first_header.index_dtype,
        ..Default::default()
    };

    // Merge produces new CSR shards from multiple inputs and drops every
    // CSC sidecar. Bump past the max input generation so the merged file's
    // generation strictly exceeds any source; `csc_build_generation`
    // defaults to 0 (no CSC emitted).
    let merged_data_generation = readers
        .iter()
        .map(|r| r.catalog().data_generation)
        .max()
        .unwrap_or(0)
        + 1;
    let mut writer =
        ScxWriter::new(output_path, out_header)?.with_data_generation(merged_data_generation);
    writer.set_framing(framing_for_rewrite(
        options.codec,
        output_framed,
        "at least one input",
    )?);

    // ---------------------------------------------------------------
    // Phase 2: streaming obs across all inputs.
    //
    // Replace the legacy `read_obs → unify_dict_columns → concat_batches`
    // chain (which materialised the entire merged obs table and hit
    // Arrow IPC's 2 GB narrow-offset ceiling for atlas-scale workloads)
    // with a shard-by-shard pipeline writing `ObsMetadataShard`
    // sections. Each input is read either via `obs_shards()` (if it is
    // already sharded) or by re-chunking its single-section
    // `read_obs()` into shards bounded by `shard_target`. The
    // per-shard `write_obs_shard` path upcasts Utf8 → LargeUtf8 inside
    // `write_arrow_ipc`, so individual shards stay below the offset
    // ceiling regardless of the merged total.
    // ---------------------------------------------------------------

    let unified_obs_schema = first_input_obs_schema(&readers)?;
    let var = readers[0].read_var().map_err(OpsError::Format)?;
    validate_var_identity(&readers, &var, options.assume_identical_var)?;

    // Obs schema identity is checked through the logical-lossy schema
    // so legacy single-section inputs (typically narrow `Utf8`) and
    // already-sharded inputs (typically `LargeUtf8` post-upcast) line
    // up. Schema-only — row contents intentionally differ.
    let first_obs_schema_lossy = readers[0]
        .read_obs_schema_logical_lossy()
        .map_err(OpsError::Format)?;
    validate_obs_identity(
        &readers,
        &first_obs_schema_lossy,
        options.assume_identical_obs,
    )?;

    // Fail fast if forced index columns are missing from the unified
    // schemas — before any output bytes are written.
    validate_forced_columns(index_options, &unified_obs_schema, &var.schema())?;

    let shard_target_rows: u64 = options
        .shard_target_rows
        .unwrap_or(first_header.shard_target_rows) as u64;

    let want_index = user_wants_index(index_options);
    let mut obs_index_builder = if want_index {
        Some(obs_predicate_index_builder(
            unified_obs_schema.clone(),
            index_options,
        )?)
    } else {
        None
    };

    // ---------------------------------------------------------------
    // Sorted k-way merge. Self-contained path so
    // the legacy concatenation below stays byte-identical. Reorders obs /
    // X / layers globally by the key; var-axis sections preserved (including
    // varm and varp, which a row permutation does not touch); obsm and COO obsp
    // both rejected up front — a sorted merge interleaves rows, so there is no
    // per-input output row range to stamp on an obsp shard. The CSR-backed obsp
    // encoding is dropped with a warning here as on every merge path.
    // ---------------------------------------------------------------
    if sorted {
        if merge_sorted::has_obsm(&readers) {
            return Err(OpsError::InvalidInput(
                "merge --sort-by does not yet support obsm; merge without --sort-by then \
                 `scx sort`, or drop obsm before a sorted merge"
                    .into(),
            ));
        }
        // Same rejection, same reason. `write_obsp_shard_coo` stamps the output
        // row range a shard covers and the reader verifies a contiguous cover,
        // and a sorted merge interleaves every input's rows — so there is no
        // per-input range to stamp. Refusing is the honest answer; carrying the
        // graph as one un-sharded section would work but would reintroduce the
        // whole-graph materialisation the sharded path exists to avoid.
        if merge_pairwise::has_global_obsp(&readers) {
            return Err(OpsError::InvalidInput(
                "merge --sort-by does not yet support obsp; merge without --sort-by then \
                 `scx sort`, or drop obsp before a sorted merge"
                    .into(),
            ));
        }
        let order = merge_sorted::compute_merge_order(
            &readers,
            &unified_obs_schema,
            &options.sort_by,
            options.sort_reverse,
            shard_target_rows,
            total_n_obs,
        )?;
        let output_shard_row_ranges = merge_sorted::emit_sorted(
            &readers,
            &mut writer,
            &order,
            shard_target_rows,
            total_n_obs,
            &mut obs_index_builder,
        )?;
        writer.write_var(&var)?;
        // varm and varp are var-axis (shared, unchanged, taken from input 0);
        // obsm and obsp both errored above.
        merge_global_dense_mapping_sharded(&readers, &mut writer, DenseMappingAxis::Varm, n_vars)?;
        merge_pairwise::carry_varp_only(&readers, &mut writer)?;
        let mut uns_conflicts_warned: usize = 0;
        if let Some(combined_uns) =
            combine_uns_for_merge(&readers, options.uns_policy, &mut uns_conflicts_warned)?
        {
            writer.write_uns(&combined_uns)?;
        }

        // Predicate index (same shape as the concat path).
        let index_result = if let Some(builder) = obs_index_builder.take() {
            let mut result = scx_engine::ConversionPredicateIndexResult::default();
            let obs_bytes = builder.finish(
                &output_shard_row_ranges,
                &mut result.obs_outcomes,
                &mut result.obs_indexed_columns,
            )?;
            if let Some(bytes) = obs_bytes {
                writer.write_obs_predicate_index(&bytes)?;
                scx_engine::apply_obs_shard_column_stats(
                    &mut writer,
                    &bytes,
                    output_shard_row_ranges.len(),
                )?;
            }
            let preset_var = match index_options.index_preset.as_deref() {
                Some(name) => scx_engine::index_preset_columns(name)
                    .map(|p| p.var_columns.iter().map(|s| (*s).to_string()).collect())
                    .unwrap_or_default(),
                None => Vec::new(),
            };
            let var_row_ranges: [(u64, u64); 1] = [(0, n_vars)];
            let var_build_opts = scx_engine::PredicateIndexBuildOptions {
                forced_columns: index_options.index_var.clone(),
                preset_columns: preset_var,
                auto_threshold: index_options.index_auto_threshold,
                high_cardinality_threshold: 100_000,
            };
            let var_bytes = scx_engine::build_var_predicate_index_bytes(
                &var,
                &var_row_ranges,
                &var_build_opts,
                &mut result.var_outcomes,
                &mut result.var_indexed_columns,
            )?;
            if let Some(bytes) = var_bytes {
                writer.write_var_predicate_index(&bytes)?;
            }
            Some(result)
        } else {
            None
        };

        // Provenance (records the sort key alongside the merge audit trail).
        let mut all_prov_entries = Vec::new();
        for reader in &readers {
            if let Ok(prov) = reader.read_provenance() {
                all_prov_entries.extend(prov.operations);
            }
        }
        let mut params = serde_json::json!({
            "n_inputs": input_paths.len(),
            "assume_identical_var": options.assume_identical_var,
            "assume_identical_obs": options.assume_identical_obs,
            "uns_policy": options.uns_policy.as_str(),
            "uns_conflicts_warned": uns_conflicts_warned,
            "sort_by": options.sort_by,
            "sort_reverse": options.sort_reverse,
        });
        if let Some(ref result) = index_result {
            params["predicate_index"] = serde_json::json!({
                "obs_columns": result.obs_indexed_columns,
                "var_columns": result.var_indexed_columns,
                "preset": options.index_options.index_preset,
            });
        }
        all_prov_entries.push(ProvenanceEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            action: "merge".to_string(),
            tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
            params_json: params.to_string(),
            input_checksums: readers
                .iter()
                .map(|r| {
                    let mut cs = [0u8; 32];
                    cs[..8].copy_from_slice(&r.header().file_checksum.to_le_bytes());
                    cs
                })
                .collect(),
        });
        writer.write_provenance(all_prov_entries)?;

        if let Some(dv) = remap_deletion_vectors(&readers, Some(&order))? {
            writer.write_deletion_vectors(&dv)?;
        }

        crate::carry::audit_staged(
            crate::carry::RewriteOp::Merge,
            &readers.iter().map(|r| r.catalog()).collect::<Vec<_>>(),
            &writer,
        )?;
        writer.finish()?;
        return Ok(PredicateIndexBuildSummary {
            result: index_result,
            multimodal_skip: None,
        });
    }

    // The output CSR shard ranges the index will be finished over, predicted
    // from the inputs' own CSR shards re-based onto the merged obs axis — the
    // write loop below concatenates them in exactly this order. obs, by
    // contrast, is chunked by `shard_target_rows`, so the two partitions
    // diverge whenever an input was written with a different shard size, and a
    // push spanning a CSR boundary would hand both shards the same widened
    // numeric `[min, max]`. This is a precision hint only: `finish` still uses
    // the ranges the write loop actually produced, so a mispredicted entry
    // costs a little pruning and nothing else.
    let predicted_csr_ranges: Vec<(u64, u64)> = if obs_index_builder.is_some() {
        let mut ranges = Vec::new();
        let mut offset = 0u64;
        for reader in &readers {
            for entry in reader.catalog().shards_sorted() {
                if let Some(stats) = entry.stats.as_ref() {
                    let n = stats.row_end.saturating_sub(stats.row_start);
                    ranges.push((offset, offset + n));
                    offset += n;
                }
            }
        }
        ranges
    } else {
        Vec::new()
    };

    let mut out_shard_idx: u32 = 0;
    let mut cumulative_obs_rows: u64 = 0;
    for reader in &readers {
        for chunk in input_obs_chunks(reader, shard_target_rows)? {
            let chunk = chunk?;
            let unified = unify_dict_columns(&chunk)?;
            if let Some(builder) = obs_index_builder.as_mut() {
                builder.push_shard_split(&unified, cumulative_obs_rows, &predicted_csr_ranges)?;
            }
            let n_shard_rows = unified.num_rows() as u64;
            writer.write_obs_shard(
                out_shard_idx,
                cumulative_obs_rows,
                n_shard_rows,
                total_n_obs,
                &unified,
            )?;
            out_shard_idx += 1;
            cumulative_obs_rows += n_shard_rows;
        }
    }

    writer.write_var(&var)?;

    // For each input: decode CSR shards and write to output with adjusted row_start.
    // Per-shard codec is auto-selected via select_codec() on the re-encoded values,
    // and value_encoding is read from each shard's header for correctness.
    let mut cumulative_rows = 0u64;
    // Per-output-shard `(row_start, row_end)` ranges, captured during the
    // write loop so the predicate-index builder can map global row ids to
    // output-shard local rows.
    let mut output_shard_row_ranges: Vec<(u64, u64)> = Vec::new();
    // The merged output stamps each CSR shard's index dtype from the merged
    // column count (mirrors `write_shard_inner`): u16 indices when `n_vars - 1`
    // fits, else u32. Raw-copy is only valid when the source shard already uses
    // this width.
    let target_index_dtype: u8 = if n_vars.saturating_sub(1) <= u16::MAX as u64 {
        0
    } else {
        1
    };
    for reader in &readers {
        let shards = reader.catalog().shards_sorted();
        for shard_entry in &shards {
            // Read this shard's header (value_encoding + raw-copy eligibility).
            let sh = reader.read_shard_header(shard_entry)?;
            let shard_value_encoding = ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?;
            let row_start = cumulative_rows;

            // Raw-copy fast path: when the source shard's on-disk layout already
            // matches the merged output (index dtype + column extent) and the
            // var axis is verified-identical, byte-copy the section instead of
            // decode → re-encode → codec-search. Merge always auto-selects the
            // codec, so the codec precondition is `Auto`.
            //
            // Disabled under `assume_identical_var`: that flag tells merge to
            // trust the caller's claim that the var axes match without
            // verifying, so column indices are not guaranteed identical. Note
            // merge never remaps indices (pure i32→u32 cast) on *either* path,
            // so for valid inputs the decode/re-encode and raw-copy outputs are
            // byte-identical — this gate is a conservative guard, not a
            // behavioural fork. (See `merge_var_mismatch_assume_identical_var_proceeds`.)
            let raw_copy_ok = raw_copy_csr_eligible(
                &sh,
                target_index_dtype,
                n_vars,
                options.codec,
                output_framed,
            ) && !options.assume_identical_var;
            if raw_copy_ok {
                // `raw_copy_csr_eligible` already guarantees the source shard's
                // framing matches the output's (framed→v4, unframed→≤v3), so the
                // byte-copy is self-contained via its in-body block index and
                // passes the writer's v4 framing guard.
                let (section_bytes, stats) = build_raw_copied_csr_section(
                    reader,
                    shard_entry,
                    &sh,
                    n_vars,
                    row_start,
                    shard_value_encoding,
                )?;
                writer.write_csr_shard_raw_copy(&section_bytes, stats, sh.nnz)?;
                cumulative_rows += sh.n_major as u64;
                output_shard_row_ranges.push((row_start, cumulative_rows));
                continue;
            }

            // Slow path: decode → re-encode with the shard's own value encoding.
            let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
            let n_rows = indptr.len() - 1;
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
            let mut values_bytes = Vec::new();
            for &v in &data {
                encode_value(&mut values_bytes, v, shard_value_encoding)?;
            }

            // Auto-select optimal codec for this shard's data
            let shard_codec = seed_codec(options.codec, &values_bytes, shard_value_encoding);

            writer.write_csr_shard(
                &indptr_u64,
                &indices_u32,
                &values_bytes,
                shard_codec,
                shard_value_encoding,
                row_start,
            )?;
            cumulative_rows += n_rows as u64;
            output_shard_row_ranges.push((row_start, cumulative_rows));
        }
    }

    // Phase 3b: stream global obsm and varm shard-by-shard. Each input's
    // shards are re-stamped with cumulative `row_start` and emitted as
    // the next output shard via `write_obsm_shard` / `write_varm_shard`.
    // Legacy single-section inputs are treated as one source shard. A key any
    // input lacks is a hard `DenseMappingMissing` (§6.4); this said "dropped
    // (existing semantic)" until Phase 5b, which is the semantic that lost a
    // 100-file atlas its `X_umap` without a word.
    merge_global_dense_mapping_sharded(&readers, &mut writer, DenseMappingAxis::Obsm, total_n_obs)?;
    merge_global_dense_mapping_sharded(&readers, &mut writer, DenseMappingAxis::Varm, n_vars)?;

    // §6.4: the pairwise graphs. obsp is obs×obs, so every endpoint is rebased
    // by its input's offset in the concatenated row space; varp is var×var on a
    // shared axis, so input 0's is the canonical one.
    merge_pairwise::merge_pairwise_sections(
        &readers,
        &mut writer,
        &input_row_offsets(&readers),
        total_n_obs,
    )?;

    // Merge uns according to the policy. Default `UnsPolicy::First`
    // matches today's pre-refactor behaviour (read first input's
    // `uns`, drop the rest); the other policies surface conflicts
    // explicitly. See [`crate::merge_options::UnsPolicy`] for the
    // semantics of each policy.
    let mut uns_conflicts_warned: usize = 0;
    if let Some(combined_uns) =
        combine_uns_for_merge(&readers, options.uns_policy, &mut uns_conflicts_warned)?
    {
        writer.write_uns(&combined_uns)?;
    }

    // Merge layers — collect layer names from ALL inputs, not just the first,
    // so layers present only in subsequent inputs are not silently dropped.
    let shard_target = first_header.shard_target_rows;
    let layer_names = {
        let mut all_names: Vec<String> = readers.iter().flat_map(|r| r.layer_names()).collect();
        all_names.sort();
        all_names.dedup();
        all_names
    };
    for layer_name in &layer_names {
        // Determine this layer's value encoding as the widest across *every*
        // input's layer shards. Sampling only `readers[0]`'s first shard would
        // abort with `ValueOutOfRange` when a later input stored the layer with
        // a wider encoding (rows from all inputs are re-packed into shared
        // output shards, so per-shard encoding does not apply — see the sorted
        // path in `merge_sorted.rs`).
        let layer_prefix = format!("{layer_name}_shard_");
        let mut layer_encs: Vec<ValueEncoding> = Vec::new();
        for reader in &readers {
            for entry in reader.catalog().entries.iter().filter(|e| {
                e.section_type == SectionType::LayerCsrShard
                    && e.modality_id == 0
                    && e.name.starts_with(&layer_prefix)
            }) {
                let sh = reader.read_shard_header(entry)?;
                layer_encs.push(
                    ValueEncoding::from_u8(sh.value_encoding)
                        .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?,
                );
            }
        }
        let layer_value_encoding = if layer_encs.is_empty() {
            ValueEncoding::Uint8
        } else {
            widest_value_encoding(&layer_encs)
        };

        // Accumulate and flush at shard_target_rows, matching compact.rs pattern.
        //
        // TODO(#221): the X loop raw-copies identical-layout shards verbatim,
        // but this layer loop re-packs across `shard_target` boundaries, so a
        // source layer shard does not map 1:1 to an output shard — the
        // byte-copy fast path does not apply here as-is. A future change could
        // raw-copy a layer shard only when it already aligns to an output
        // boundary; deferred for now (see PR #221 / OPT-1.2).
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;
        let mut emitted_layer_rows = 0u64;

        // Phase 3a: stream each input's `LayerCsrShard` entries one at a
        // time and re-pack into output shards bounded by `shard_target`.
        // Mirror of the multimodal per-modality layer streaming pattern
        // (see `merge_multimodal` below). No `read_layer` call — peak
        // memory is one input shard plus the in-flight output shard.
        for (file_idx, reader) in readers.iter().enumerate() {
            let mut input_shards: Vec<&scx_format_io::catalog::FullCatalogEntry> = reader
                .catalog()
                .entries
                .iter()
                .filter(|e| {
                    e.section_type == SectionType::LayerCsrShard
                        && e.modality_id == 0
                        && e.name.starts_with(&layer_prefix)
                })
                .collect();
            if input_shards.is_empty() {
                return Err(OpsError::LayerMissing {
                    name: layer_name.clone(),
                    file_index: file_idx,
                });
            }
            input_shards.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

            for shard_entry in input_shards {
                let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
                let n_rows = indptr.len() - 1;
                for local_row in 0..n_rows {
                    let s = indptr[local_row] as usize;
                    let e = indptr[local_row + 1] as usize;
                    for j in s..e {
                        layer_indices.push(indices[j] as u32);
                        encode_value(&mut layer_values, data[j], layer_value_encoding)?;
                    }
                    let prev = *layer_indptr.last().unwrap();
                    layer_indptr.push(prev + (e - s) as u64);
                    layer_row_count += 1;

                    if layer_row_count >= shard_target as u64 {
                        let layer_shard_codec =
                            seed_codec(options.codec, &layer_values, layer_value_encoding);
                        writer.write_layer_csr_shard(
                            &layer_indptr,
                            &layer_indices,
                            &layer_values,
                            layer_shard_codec,
                            layer_value_encoding,
                            emitted_layer_rows,
                            layer_name,
                            layer_shard_idx,
                        )?;
                        emitted_layer_rows += layer_row_count;
                        layer_indptr = vec![0];
                        layer_indices.clear();
                        layer_values.clear();
                        layer_row_count = 0;
                        layer_shard_idx += 1;
                    }
                }
            }
        }

        // Flush remaining layer rows
        if layer_row_count > 0 {
            // Auto-select codec for remaining layer shard
            let layer_shard_codec = seed_codec(options.codec, &layer_values, layer_value_encoding);
            writer.write_layer_csr_shard(
                &layer_indptr,
                &layer_indices,
                &layer_values,
                layer_shard_codec,
                layer_value_encoding,
                emitted_layer_rows,
                layer_name,
                layer_shard_idx,
            )?;
        }
    }

    // Build + write predicate indexes (obs + var) using the streaming
    // builder we accumulated during the obs-shard write loop. The
    // builder consumed each obs shard inline so peak memory stayed
    // bounded to one shard at a time. Forced-column-missing was
    // already caught by `validate_forced_columns` above; engine
    // outcomes here are limited to preset skips and supported-type
    // checks.
    let index_result = if let Some(builder) = obs_index_builder {
        let mut result = scx_engine::ConversionPredicateIndexResult::default();
        let obs_bytes = builder.finish(
            &output_shard_row_ranges,
            &mut result.obs_outcomes,
            &mut result.obs_indexed_columns,
        )?;
        if let Some(bytes) = obs_bytes {
            writer.write_obs_predicate_index(&bytes)?;
            // Populate per-shard catalog column stats so query-time shard
            // skipping works on the merged output. `output_shard_row_ranges`
            // is the CSR-shard space the index was built against.
            scx_engine::apply_obs_shard_column_stats(
                &mut writer,
                &bytes,
                output_shard_row_ranges.len(),
            )?;
        }
        // var stays on the batch-mode builder — var rarely overflows
        // and the streaming path doesn't help small-axis predicate
        // indexes. Reuse `build_var_predicate_index_bytes` with the
        // same column-resolution policy used by
        // `build_and_write_conversion_predicate_indexes` so behaviour
        // is byte-identical to the non-streaming entry point.
        let preset_var = match index_options.index_preset.as_deref() {
            Some(name) => scx_engine::index_preset_columns(name)
                .map(|p| p.var_columns.iter().map(|s| (*s).to_string()).collect())
                .unwrap_or_default(),
            None => Vec::new(),
        };
        let var_row_ranges: [(u64, u64); 1] = [(0, n_vars)];
        let var_build_opts = scx_engine::PredicateIndexBuildOptions {
            forced_columns: index_options.index_var.clone(),
            preset_columns: preset_var,
            auto_threshold: index_options.index_auto_threshold,
            high_cardinality_threshold: 100_000,
        };
        let var_bytes = scx_engine::build_var_predicate_index_bytes(
            &var,
            &var_row_ranges,
            &var_build_opts,
            &mut result.var_outcomes,
            &mut result.var_indexed_columns,
        )?;
        if let Some(bytes) = var_bytes {
            writer.write_var_predicate_index(&bytes)?;
        }
        Some(result)
    } else {
        None
    };

    // Merge provenance
    let mut all_prov_entries = Vec::new();
    for reader in &readers {
        if let Ok(prov) = reader.read_provenance() {
            all_prov_entries.extend(prov.operations);
        }
    }
    // Record the actually-indexed columns in provenance (mirrors the convert
    // path at `scx-convert/src/pipeline.rs`) so the merge record carries an
    // audit trail of which predicate indexes were rebuilt — closes the
    // "NO index field in params_json" gap reported against atlas merges.
    let mut params = serde_json::json!({
        "n_inputs": input_paths.len(),
        "assume_identical_var": options.assume_identical_var,
        "assume_identical_obs": options.assume_identical_obs,
        "uns_policy": options.uns_policy.as_str(),
        "uns_conflicts_warned": uns_conflicts_warned,
    });
    if let Some(ref result) = index_result {
        params["predicate_index"] = serde_json::json!({
            "obs_columns": result.obs_indexed_columns,
            "var_columns": result.var_indexed_columns,
            "preset": options.index_options.index_preset,
        });
    }
    all_prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "merge".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: params.to_string(),
        input_checksums: readers
            .iter()
            .map(|r| {
                let mut cs = [0u8; 32];
                cs[..8].copy_from_slice(&r.header().file_checksum.to_le_bytes());
                cs
            })
            .collect(),
    });
    writer.write_provenance(all_prov_entries)?;

    if let Some(dv) = remap_deletion_vectors(&readers, None)? {
        writer.write_deletion_vectors(&dv)?;
    }

    crate::carry::audit_staged(
        crate::carry::RewriteOp::Merge,
        &readers.iter().map(|r| r.catalog()).collect::<Vec<_>>(),
        &writer,
    )?;
    writer.finish()?;
    Ok(PredicateIndexBuildSummary {
        result: index_result,
        multimodal_skip: None,
    })
}

/// Each input's first row in the concatenated obs space.
///
/// A concatenating merge drops no rows, so this is just a running sum of
/// `n_obs`. Shared by the deletion-vector remap and the obsp remap because they
/// are the same rebase of the same axis, and having them derive it separately is
/// how the two would drift.
fn input_row_offsets(readers: &[ScxReader]) -> Vec<u64> {
    readers
        .iter()
        .scan(0u64, |acc, r| {
            let start = *acc;
            *acc += r.n_obs();
            Some(start)
        })
        .collect()
}

/// Union the inputs' deletion vectors into the merged output's row space, or
/// `Ok(None)` when no input has any.
///
/// Deletion vectors are obs-row-indexed, so merging them is the same
/// index-remapping problem the obs rows themselves pose, and the answer has to
/// match whichever way the rows were laid out:
///
/// * `order = None` — plain concatenation. Input `i` occupies the output rows
///   `[Σ_{j<i} n_obs_j, …)`, so a deleted row moves by that prefix sum.
/// * `order = Some(order)` — sorted k-way merge. `order[out_row]` names the
///   contributing input, and each input is consumed strictly in its own row
///   order, so replaying it with per-input forward cursors reconstructs the
///   (input, input row) → output row map. This is the same replay
///   `merge_sorted::emit_sorted` performs over the same slice.
///
/// Every bitmap is remapped, not just the global one: scoped (`modality_id >=
/// 1`) bitmaps are reserved and unpopulated by shipped writers, but they are
/// obs-row-indexed too, so treating them uniformly is both correct and one less
/// special case to get wrong later.
fn remap_deletion_vectors(
    readers: &[ScxReader],
    order: Option<&[u32]>,
) -> Result<Option<scx_format_io::deletion_vectors::DeletionVectors>> {
    use scx_format_io::deletion_vectors::DeletionVectors;

    let input_dvs: Vec<Option<DeletionVectors>> = readers
        .iter()
        .map(|r| r.read_deletion_vectors())
        .collect::<std::result::Result<_, _>>()?;
    if input_dvs.iter().all(|dv| dv.is_none()) {
        return Ok(None);
    }

    // out_rows[input][input_row] — built once, then reused for every bitmap of
    // that input. For the concatenation case this is just an offset, so only the
    // sorted case materializes a map.
    let offsets = input_row_offsets(readers);
    let sorted_map: Option<Vec<Vec<u64>>> = match order {
        None => None,
        Some(order) => {
            // The per-input cursor is implicit: each input is consumed in
            // strict row order, so pushing onto `map[input]` puts output row
            // `out_row` at exactly that input's next row index. `map[i][row]`
            // is therefore the output row of input `i`'s row `row`.
            let mut map: Vec<Vec<u64>> = readers
                .iter()
                .map(|r| Vec::with_capacity(r.n_obs() as usize))
                .collect();
            for (out_row, &input) in order.iter().enumerate() {
                let input = input as usize;
                let slot = map.get_mut(input).ok_or_else(|| {
                    OpsError::InvalidInput(format!(
                        "merge order names input {input} but only {} inputs were opened",
                        readers.len()
                    ))
                })?;
                slot.push(out_row as u64);
            }
            Some(map)
        }
    };

    let mut out = DeletionVectors::new();
    let mut any = false;
    for (i, dv) in input_dvs.iter().enumerate() {
        let Some(dv) = dv else { continue };
        for (&modality_id, bitmap) in &dv.deletions {
            let mut rows: Vec<u32> = Vec::with_capacity(bitmap.len() as usize);
            for row in bitmap.iter() {
                let out_row = match &sorted_map {
                    Some(map) => *map[i].get(row as usize).ok_or_else(|| {
                        OpsError::InvalidInput(format!(
                            "input {i} marks row {row} deleted but the merge order only \
                             placed {} of its rows",
                            map[i].len()
                        ))
                    })?,
                    None => offsets[i] + row as u64,
                };
                // Roaring bitmaps are u32-keyed while `n_obs` is u64. Refuse
                // rather than wrap: a wrapped index would mark an unrelated cell
                // deleted, which is the failure this whole change exists to stop.
                rows.push(u32::try_from(out_row).map_err(|_| {
                    OpsError::InvalidInput(format!(
                        "merged obs row {out_row} exceeds the u32 range a deletion vector \
                         can address; compact the inputs to materialize their deletions \
                         before merging"
                    ))
                })?);
            }
            if rows.is_empty() {
                continue;
            }
            any = true;
            out.deletions
                .entry(modality_id)
                .or_default()
                .extend(rows.iter().copied());
        }
    }

    Ok(any.then_some(out))
}

/// Phase 6: merge multimodal SCX files with matching modality
/// structure. Per-modality CSR shards and per-modality layers are
/// concatenated in input order with `row_start` adjusted for the
/// cumulative global obs offset; global `obsm` and per-modality
/// `obsm` are concatenated row-wise (any key missing in any input is
/// dropped, matching single-modality semantics). Per-modality CSC
/// sidecars are dropped (caller can `--rebuild-csc`).
///
/// Honours [`MergeOptions`] in full:
/// * `assume_identical_var` gates the per-modality var-identity
///   check (each modality's var is validated independently against
///   input 0).
/// * `assume_identical_obs` gates the global obs schema check.
/// * `shard_target_rows` overrides the obs-shard target on the
///   merged output.
/// * `uns_policy` applies independently to global uns and to each
///   modality's uns; the conflict counter aggregates across both
///   levels and lands in the merge provenance entry.
///
/// Predicate-index construction is intentionally skipped — the
/// dispatch in `merge_with_options` records the skip so the caller
/// emits a single `PredicateIndexSkippedMultimodal` warning.
///
/// Peak RSS for all-legacy obs inputs is bounded by one input's
/// `read_obs()` plus the in-flight output shard (sharded inputs
/// stream directly without that materialisation).
fn merge_multimodal(
    readers: &[ScxReader],
    input_paths: &[&Path],
    output_path: &Path,
    options: &MergeOptions,
) -> Result<()> {
    let table = readers[0]
        .modality_table()
        .ok_or_else(|| {
            scx_format_io::ScxError::InvalidCatalog(
                "merge_multimodal: input has no modality table".to_string(),
            )
        })?
        .clone();

    let total_n_obs: u64 = readers.iter().map(|r| r.n_obs()).sum();
    let first_header = readers[0].header();

    let any_input_had_csc = readers.iter().any(|r| {
        r.modality_table()
            .map(|t| t.entries.iter().any(|info| info.flags.has_csc()))
            .unwrap_or(false)
    });
    if any_input_had_csc {
        let inputs_str = input_paths
            .iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        log::warn!(
            "merge dropped per-modality CSC shards from at least one input ({inputs_str}): \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar on the merged output"
        );
    }

    let max_n_vars = table.entries.iter().map(|i| i.n_vars).max().unwrap_or(0);

    // Gate the v3 canonical claim on the minimum input version (floor 2 —
    // multimodal output requires v2+). See the single-modality merge above.
    let input_format_versions: Vec<u16> =
        readers.iter().map(|r| r.header().format_version).collect();
    // Preserve framing when every input is framed (v4). See the single-modality
    // merge above for the rationale (v4 stamp + writer framing → framed shards
    // byte-copy verbatim, decode-encoded shards re-emit framed).
    let output_framed = !input_format_versions.is_empty()
        && input_format_versions
            .iter()
            .all(|&v| v >= CURRENT_FORMAT_VERSION);
    let out_format_version = if output_framed {
        CURRENT_FORMAT_VERSION
    } else {
        scx_format_io::rewrite_output_format_version(&input_format_versions, 2)
    };
    let out_header = FileHeader {
        format_version: out_format_version,
        n_obs: total_n_obs,
        n_vars: max_n_vars,
        shard_target_rows: first_header.shard_target_rows,
        index_dtype: first_header.index_dtype,
        ..Default::default()
    };

    // As with single-modality merge: new CSR shards, CSC dropped — bump
    // past the max input generation.
    let merged_data_generation = readers
        .iter()
        .map(|r| r.catalog().data_generation)
        .max()
        .unwrap_or(0)
        + 1;
    let mut writer =
        ScxWriter::new(output_path, out_header)?.with_data_generation(merged_data_generation);
    writer.set_framing(framing_for_rewrite(
        options.codec,
        output_framed,
        "at least one input",
    )?);

    // Validate global obs schema across all inputs before any output
    // bytes are written. Mirrors the single-modality call site.
    let first_obs_schema_lossy = readers[0]
        .read_obs_schema_logical_lossy()
        .map_err(OpsError::Format)?;
    validate_obs_identity(
        readers,
        &first_obs_schema_lossy,
        options.assume_identical_obs,
    )?;

    // Phase 2b: Global obs streams shard-by-shard exactly as the
    // single-modality path does (see `merge_with_options`).
    let shard_target_rows: u64 = options
        .shard_target_rows
        .unwrap_or(first_header.shard_target_rows) as u64;
    let mut out_shard_idx: u32 = 0;
    let mut cumulative_obs_rows: u64 = 0;
    for reader in readers {
        for chunk in input_obs_chunks(reader, shard_target_rows)? {
            let chunk = chunk?;
            let unified = unify_dict_columns(&chunk)?;
            let n_shard_rows = unified.num_rows() as u64;
            writer.write_obs_shard(
                out_shard_idx,
                cumulative_obs_rows,
                n_shard_rows,
                total_n_obs,
                &unified,
            )?;
            out_shard_idx += 1;
            cumulative_obs_rows += n_shard_rows;
        }
    }

    // Phase 3b: stream global obsm shard-by-shard (multimodal). Same
    // pattern as single-modality: every input's shards (or legacy
    // single-section as one source shard) are re-stamped and emitted
    // as the next output shard. A key any input lacks is a hard
    // `DenseMappingMissing` (§6.4), not the silent drop this said until 5b.
    // Multimodal global varm is omitted by design — `n_vars` differs
    // per modality, so there is no canonical `n_rows_total` for a
    // global varm shard. Per-modality varm is handled below in Step 5.
    merge_global_dense_mapping_sharded(readers, &mut writer, DenseMappingAxis::Obsm, total_n_obs)?;

    // §6.4, multimodal: a file-scope obs×obs graph is carried on the same
    // offsets as everything else on the shared obs axis. A file-scope `varp` is
    // omitted for the same reason global `varm` is, just above.
    // Modality-scoped pairwise graphs are dropped with a warning — there is no
    // per-modality pairwise reader to round-trip them through.
    merge_pairwise::merge_obsp_only(
        readers,
        &mut writer,
        &input_row_offsets(readers),
        total_n_obs,
    )?;

    // Register modalities in input order.
    for info in &table.entries {
        let codec = CodecId::from_u8(info.default_codec_id)
            .ok_or(OpsError::UnknownCodec(info.default_codec_id))?;
        let value_encoding = ValueEncoding::from_u8(info.default_value_encoding)
            .ok_or(OpsError::UnknownValueEncoding(info.default_value_encoding))?;
        writer.add_modality(&info.name, info.modality_type, codec, value_encoding, false)?;
        writer.set_modality_n_vars(writer.n_modalities() as u8, info.n_vars)?;
    }

    // Per-modality var: validate every input's var matches input 0
    // (column-by-column, not just `n_vars`), then write input 0's
    // copy. The modality-table check only confirms shape + name —
    // it can't catch reordered genes or differing feature IDs.
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let var = readers[0]
            .read_var_for(modality_id)
            .map_err(OpsError::Format)?;
        validate_var_identity_for_modality(
            readers,
            modality_id,
            &var,
            options.assume_identical_var,
        )?;
        writer.write_var_for(modality_id, &var)?;
        let _ = info; // suppress unused if no other field needed
    }

    // Per-modality CSR shards: concatenate across inputs with row_start
    // adjusted for the cumulative global obs offset. Within an input,
    // the modality's shards collectively cover the input's n_obs rows;
    // after one input we advance the offset by that input's n_obs so
    // the next input's shards line up against the merged obs.
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let modality_n_vars = info.n_vars;
        // Per-modality CSR shards stamp their index dtype from the modality's
        // own column count (see `write_shard_inner`'s `row_major_n_minor`).
        let target_index_dtype: u8 = if modality_n_vars.saturating_sub(1) <= u16::MAX as u64 {
            0
        } else {
            1
        };
        let mut input_offset: u64 = 0;
        for reader in readers {
            let entries = reader.catalog().csr_shards_for_modality(modality_id);
            for shard_entry in entries {
                let sh = reader.read_shard_header(shard_entry)?;
                let shard_value_encoding = ValueEncoding::from_u8(sh.value_encoding)
                    .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?;
                let shard_local_row_start =
                    shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
                let row_start = input_offset + shard_local_row_start;

                // Raw-copy fast path (see the single-modality X loop above for
                // the full rationale). Eligibility uses the modality's own
                // column extent / index dtype.
                let raw_copy_ok = raw_copy_csr_eligible(
                    &sh,
                    target_index_dtype,
                    modality_n_vars,
                    options.codec,
                    output_framed,
                ) && !options.assume_identical_var;
                if raw_copy_ok {
                    let (section_bytes, stats) = build_raw_copied_csr_section(
                        reader,
                        shard_entry,
                        &sh,
                        modality_n_vars,
                        row_start,
                        shard_value_encoding,
                    )?;
                    writer.write_csr_shard_raw_copy_for(
                        modality_id,
                        &section_bytes,
                        stats,
                        sh.nnz,
                    )?;
                    continue;
                }

                let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
                let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
                let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
                let mut values_bytes = Vec::new();
                for &v in &data {
                    encode_value(&mut values_bytes, v, shard_value_encoding)?;
                }
                let shard_codec = seed_codec(options.codec, &values_bytes, shard_value_encoding);
                writer.write_csr_shard_for(
                    modality_id,
                    &indptr_u64,
                    &indices_u32,
                    &values_bytes,
                    shard_codec,
                    shard_value_encoding,
                    row_start,
                )?;
            }
            input_offset += reader.n_obs();
        }
    }

    // Phase 3b: stream per-modality obsm and varm shard-by-shard via
    // the new `write_obsm_shard_for` / `write_varm_shard_for` writer
    // APIs. Each input's shards (or legacy single-section as one source
    // shard) are re-stamped and emitted as the next output shard. A key any
    // input lacks is dropped **with a warning** here — unlike the global helper,
    // which hard-errors: an input may legitimately carry nothing at all for a
    // given modality+axis, and §6.4 measured the global path. Per-modality varm
    // support is newly added in Phase 3b — it was silently dropped pre-Phase-3.
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        merge_per_modality_dense_mapping_sharded(
            readers,
            &mut writer,
            DenseMappingAxis::Obsm,
            modality_id,
            &info.name,
            total_n_obs,
        )?;
        merge_per_modality_dense_mapping_sharded(
            readers,
            &mut writer,
            DenseMappingAxis::Varm,
            modality_id,
            &info.name,
            info.n_vars,
        )?;
    }

    // Per-modality layers: concatenate per-modality `layer/{mod}/{layer}/...`
    // shards across inputs with cumulative emitted-rows as row_start.
    // Layer names are unioned across all readers (matching single-modality
    // merge); any reader missing a layer that another input has fails fast
    // with `LayerMissing`.
    let shard_target = first_header.shard_target_rows;
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let layer_prefix = format!("layer/{}/", info.name);
        let mut layer_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for reader in readers {
            for entry in &reader.catalog().entries {
                if entry.section_type == SectionType::LayerCsrShard
                    && entry.modality_id == modality_id
                    && entry.name.starts_with(&layer_prefix)
                {
                    if let Some(remainder) = entry.name.strip_prefix(&layer_prefix) {
                        if let Some(pos) = remainder.find("/shard_") {
                            layer_names.insert(remainder[..pos].to_string());
                        }
                    }
                }
            }
        }

        for layer_name in &layer_names {
            // Value encoding = widest across *every* input's shards for this
            // (modality, layer), falling back to the modality default when no
            // input has a shard. Sampling only the first input's first shard
            // would abort with `ValueOutOfRange` when a later input stored the
            // layer with a wider encoding (mirror of the single-modality path
            // above and the sorted merge).
            let layer_value_encoding = {
                let mut encs: Vec<ValueEncoding> = Vec::new();
                for reader in readers {
                    for shard in reader
                        .catalog()
                        .layer_csr_shards_for_modality(modality_id, layer_name)
                    {
                        let sh = reader.read_shard_header(shard)?;
                        encs.push(
                            ValueEncoding::from_u8(sh.value_encoding)
                                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?,
                        );
                    }
                }
                if encs.is_empty() {
                    ValueEncoding::from_u8(info.default_value_encoding)
                        .unwrap_or(ValueEncoding::Uint8)
                } else {
                    widest_value_encoding(&encs)
                }
            };

            let mut l_indptr: Vec<u64> = vec![0];
            let mut l_indices: Vec<u32> = Vec::new();
            let mut l_values: Vec<u8> = Vec::new();
            let mut l_rows = 0u64;
            let mut l_shard_idx = 0u32;
            let mut l_emitted = 0u64;

            for (file_idx, reader) in readers.iter().enumerate() {
                let shards = reader
                    .catalog()
                    .layer_csr_shards_for_modality(modality_id, layer_name);
                if shards.is_empty() {
                    return Err(OpsError::LayerMissing {
                        name: format!("{}/{}", info.name, layer_name),
                        file_index: file_idx,
                    });
                }
                for shard_entry in shards {
                    let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
                    let n_rows = indptr.len() - 1;
                    for local_row in 0..n_rows {
                        let s = indptr[local_row] as usize;
                        let e = indptr[local_row + 1] as usize;
                        for j in s..e {
                            l_indices.push(indices[j] as u32);
                            encode_value(&mut l_values, data[j], layer_value_encoding)?;
                        }
                        let prev = *l_indptr.last().unwrap();
                        l_indptr.push(prev + (e - s) as u64);
                        l_rows += 1;

                        if l_rows >= shard_target as u64 {
                            let codec = seed_codec(options.codec, &l_values, layer_value_encoding);
                            writer.write_layer_csr_shard_for(
                                modality_id,
                                layer_name,
                                l_shard_idx,
                                &l_indptr,
                                &l_indices,
                                &l_values,
                                codec,
                                layer_value_encoding,
                                l_emitted,
                            )?;
                            l_emitted += l_rows;
                            l_indptr = vec![0];
                            l_indices.clear();
                            l_values.clear();
                            l_rows = 0;
                            l_shard_idx += 1;
                        }
                    }
                }
            }

            if l_rows > 0 {
                let codec = seed_codec(options.codec, &l_values, layer_value_encoding);
                writer.write_layer_csr_shard_for(
                    modality_id,
                    layer_name,
                    l_shard_idx,
                    &l_indptr,
                    &l_indices,
                    &l_values,
                    codec,
                    layer_value_encoding,
                    l_emitted,
                )?;
            }
        }
    }

    // Per-modality uns: apply `options.uns_policy` independently at
    // each modality level. A modality whose readers all lack a uns
    // section emits no section (matches today's behaviour). Conflicts
    // aggregate into the same counter as the global level so the
    // single provenance entry reflects total drift across the file.
    let mut uns_conflicts_warned: usize = 0;
    for (idx, _info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let per_input: Vec<Option<serde_json::Value>> = readers
            .iter()
            .map(|r| r.read_uns_for(modality_id).ok())
            .collect();
        if let Some(combined) =
            combine_uns_per_input(per_input, options.uns_policy, &mut uns_conflicts_warned)?
        {
            writer.write_uns_for(modality_id, &combined)?;
        }
    }

    // Global uns: same policy applied at the global level.
    if let Some(combined) =
        combine_uns_for_merge(readers, options.uns_policy, &mut uns_conflicts_warned)?
    {
        writer.write_uns(&combined)?;
    }

    // Provenance: concatenate input chains, then stamp merge op.
    let mut all_prov = Vec::new();
    for reader in readers {
        if let Ok(prov) = reader.read_provenance() {
            all_prov.extend(prov.operations);
        }
    }
    all_prov.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "merge".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: format!(
            "{{\"n_inputs\":{},\"assume_identical_var\":{},\"assume_identical_obs\":{},\
             \"uns_policy\":\"{}\",\"uns_conflicts_warned\":{}}}",
            input_paths.len(),
            options.assume_identical_var,
            options.assume_identical_obs,
            options.uns_policy.as_str(),
            uns_conflicts_warned,
        ),
        input_checksums: readers
            .iter()
            .map(|r| {
                let mut cs = [0u8; 32];
                cs[..8].copy_from_slice(&r.header().file_checksum.to_le_bytes());
                cs
            })
            .collect(),
    });
    writer.write_provenance(all_prov)?;
    // Deletion is whole-cell and the obs axis is shared across modalities, so
    // the multimodal concatenation moves rows by exactly the same per-input
    // offsets the single-modality path uses.
    if let Some(dv) = remap_deletion_vectors(readers, None)? {
        writer.write_deletion_vectors(&dv)?;
    }
    crate::carry::audit_staged(
        crate::carry::RewriteOp::Merge,
        &readers.iter().map(|r| r.catalog()).collect::<Vec<_>>(),
        &writer,
    )?;
    writer.finish()?;
    Ok(())
}

/// Phase F.4 helper: two modality tables match iff they have the same
/// length and every entry agrees on (name, modality_type, n_vars).
/// Two `None`s also match (both inputs single-modality). Mixed
/// `Some` / `None` does not match.
fn modality_tables_match(
    a: Option<&scx_format_io::ModalityTable>,
    b: Option<&scx_format_io::ModalityTable>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(_), None) | (None, Some(_)) => false,
        (Some(x), Some(y)) => {
            if x.len() != y.len() {
                return false;
            }
            x.entries.iter().zip(y.entries.iter()).all(|(p, q)| {
                p.name == q.name && p.modality_type == q.modality_type && p.n_vars == q.n_vars
            })
        }
    }
}

// ============================================================================
// Phase 2a helpers — unified schema, var validation, obs streaming, uns merge
// ============================================================================

/// Resolve the first input's unified obs schema. Falls back to the
/// sharded layout's schema (via `read_obs_schema_*`) when the input is
/// row-sharded. The schema is used to drive predicate-index column
/// selection and `validate_forced_columns` — exact per-column widths
/// (`Utf8` vs `LargeUtf8`) are not load-bearing because the streaming
/// builder's per-shard check normalises across width via
/// [`scx_engine::index::column_class_compatible`].
fn first_input_obs_schema(readers: &[ScxReader]) -> Result<arrow::datatypes::SchemaRef> {
    let schema = readers[0]
        .read_obs_schema_physical()
        .map_err(OpsError::Format)?;
    Ok(std::sync::Arc::new(schema))
}

/// Validate that every input agrees with the first input on var
/// identity (column names, types, row count, row content). Each
/// pairwise mismatch produces a `OpsError::VarMismatch` unless
/// `assume_identical` is true — in which case the function logs a
/// warning per mismatch and proceeds with the first input's var.
///
/// `n_vars` was already validated upstream (count-only check), so this
/// is the second, stronger check that prevents silent column-axis
/// corruption when inputs disagree on gene order.
fn validate_var_identity(
    readers: &[ScxReader],
    first_var: &RecordBatch,
    assume_identical: bool,
) -> Result<()> {
    for (i, reader) in readers.iter().enumerate().skip(1) {
        let other = reader.read_var().map_err(OpsError::Format)?;
        if let Some(detail) = var_diff(first_var, &other) {
            if assume_identical {
                log::warn!(
                    "merge: input {i}'s var differs from input 0 (--assume-identical-var \
                     in effect; using input 0's var verbatim): {detail}"
                );
            } else {
                return Err(OpsError::VarMismatch {
                    detail: format!("input 0 vs input {i}: {detail}"),
                });
            }
        }
    }
    Ok(())
}

/// Compare two var `RecordBatch`es for identity. Returns `None` when
/// they are identical for the purposes of merge (column names, dtypes,
/// row count, and per-column content all match), or a human-readable
/// description of the first observed difference. Because per-column
/// content is compared element-wise (positionally), this also enforces
/// that the two batches share the same row/gene order.
fn var_diff(a: &RecordBatch, b: &RecordBatch) -> Option<String> {
    if a.num_rows() != b.num_rows() {
        return Some(format!(
            "n_vars differ ({} vs {})",
            a.num_rows(),
            b.num_rows()
        ));
    }
    let a_schema = a.schema();
    let b_schema = b.schema();
    let a_names: Vec<&str> = a_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let b_names: Vec<&str> = b_schema
        .fields()
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    if a_names != b_names {
        return Some(format!(
            "column names differ ({:?} vs {:?})",
            a_names, b_names,
        ));
    }
    for (i, (af, bf)) in a_schema
        .fields()
        .iter()
        .zip(b_schema.fields().iter())
        .enumerate()
    {
        if af.data_type() != bf.data_type() {
            return Some(format!(
                "column '{}' dtype differs ({:?} vs {:?})",
                af.name(),
                af.data_type(),
                bf.data_type(),
            ));
        }
        let a_col = a.column(i);
        let b_col = b.column(i);
        if a_col.as_ref() != b_col.as_ref() {
            return Some(format!(
                "column '{}' content differs (use --assume-identical-var to override)",
                af.name(),
            ));
        }
    }
    None
}

/// Per-modality counterpart to [`validate_var_identity`]. Reads
/// each input's `var_for(modality_id)` and compares it to the first
/// input's. `assume_identical` downgrades errors to log warnings —
/// same semantics and CLI flag wiring as the global path.
fn validate_var_identity_for_modality(
    readers: &[ScxReader],
    modality_id: u8,
    first_var: &RecordBatch,
    assume_identical: bool,
) -> Result<()> {
    for (i, reader) in readers.iter().enumerate().skip(1) {
        let other = reader.read_var_for(modality_id).map_err(OpsError::Format)?;
        if let Some(detail) = var_diff(first_var, &other) {
            if assume_identical {
                log::warn!(
                    "merge: input {i}'s var (modality_id={modality_id}) differs from input 0 \
                     (--assume-identical-var in effect; using input 0's var verbatim): {detail}"
                );
            } else {
                return Err(OpsError::VarMismatch {
                    detail: format!("modality_id={modality_id}: input 0 vs input {i}: {detail}"),
                });
            }
        }
    }
    Ok(())
}

/// Validate that every input agrees with the first input on obs
/// schema (column names + dtypes, normalised through the
/// logical-lossy schema so `Utf8` and `LargeUtf8` count as the same
/// column). Each pairwise mismatch produces a
/// [`OpsError::ObsMismatch`] unless `assume_identical` is true — in
/// which case the function logs a warning per mismatch and trusts
/// the caller's assertion that columns line up.
///
/// Schema-only (no content comparison): obs row content
/// intentionally differs across merge inputs — that's the whole
/// point. The check that matters is that the per-shard schemas line
/// up well enough for [`ScxReader::read_obs`]'s `concat_batches` to
/// succeed after the merge completes.
fn validate_obs_identity(
    readers: &[ScxReader],
    first_schema: &arrow::datatypes::Schema,
    assume_identical: bool,
) -> Result<()> {
    for (i, reader) in readers.iter().enumerate().skip(1) {
        let other = reader
            .read_obs_schema_logical_lossy()
            .map_err(OpsError::Format)?;
        if let Some(detail) = schema_field_diff(first_schema, &other) {
            if assume_identical {
                log::warn!(
                    "merge: input {i}'s obs schema differs from input 0 \
                     (--assume-identical-obs in effect; trusting columns line up): {detail}"
                );
            } else {
                return Err(OpsError::ObsMismatch {
                    detail: format!("input 0 vs input {i}: {detail}"),
                });
            }
        }
    }
    Ok(())
}

/// Compare two schemas field-by-field (names + dtypes). Returns
/// `None` when they line up, or a human-readable description of the
/// first observed difference. Used by both obs identity validation
/// (where widths are normalised through `read_obs_schema_logical_lossy`)
/// and dense-mapping schema validation.
fn schema_field_diff(a: &arrow::datatypes::Schema, b: &arrow::datatypes::Schema) -> Option<String> {
    let a_names: Vec<&str> = a.fields().iter().map(|f| f.name().as_str()).collect();
    let b_names: Vec<&str> = b.fields().iter().map(|f| f.name().as_str()).collect();
    if a_names.len() != b_names.len() {
        return Some(format!(
            "column count differs ({} vs {})",
            a_names.len(),
            b_names.len(),
        ));
    }
    if a_names != b_names {
        return Some(format!(
            "column names differ ({:?} vs {:?})",
            a_names, b_names,
        ));
    }
    for (af, bf) in a.fields().iter().zip(b.fields().iter()) {
        if af.data_type() != bf.data_type() {
            return Some(format!(
                "column '{}' dtype differs ({:?} vs {:?})",
                af.name(),
                af.data_type(),
                bf.data_type(),
            ));
        }
    }
    None
}

/// Resolve which catalog entry to read first for a given input's
/// dense-mapping presence — used by [`validate_dense_mapping_schemas`]
/// to compare across inputs without re-implementing the legacy-vs-
/// sharded fallback at every call site.
fn dense_mapping_first_entry<'a>(
    per_input_entry: &'a (
        usize,
        Vec<&scx_format_io::catalog::FullCatalogEntry>,
        Option<&scx_format_io::catalog::FullCatalogEntry>,
    ),
) -> &'a scx_format_io::catalog::FullCatalogEntry {
    if let Some(legacy) = per_input_entry.2 {
        return legacy;
    }
    per_input_entry
        .1
        .iter()
        .min_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start))
        .copied()
        .expect("per_input is only populated when the input has at least one shard or legacy entry")
}

/// Compare each input's first dense-mapping batch schema against
/// input 0's for a shared key. Catches mismatched embedding widths
/// (e.g. PCA 50 vs 100) or dtype drift before any shard is written
/// — without that, the merge succeeds and only fails later when
/// `ScxReader::read_obsm()` tries to concatenate heterogeneous
/// shards.
///
/// Reads one batch per input per shared key (small bounded cost: a
/// shard's worth of float / int data, typically tens of KB). The
/// subsequent write loop re-reads the same entry; we accept the
/// duplicate read to keep the validation surface readable.
fn validate_dense_mapping_schemas(
    axis: &'static str,
    key: &str,
    source_readers: &[ScxReader],
    per_input: &[(
        usize,
        Vec<&scx_format_io::catalog::FullCatalogEntry>,
        Option<&scx_format_io::catalog::FullCatalogEntry>,
    )],
) -> Result<()> {
    if per_input.len() < 2 {
        return Ok(());
    }
    let first_entry = dense_mapping_first_entry(&per_input[0]);
    let first_reader = &source_readers[per_input[0].0];
    let first_batch = first_reader
        .read_dense_mapping_entry(first_entry)
        .map_err(OpsError::Format)?;
    let first_schema = first_batch.schema();
    for per in &per_input[1..] {
        let other_entry = dense_mapping_first_entry(per);
        let other_reader = &source_readers[per.0];
        let other_batch = other_reader
            .read_dense_mapping_entry(other_entry)
            .map_err(OpsError::Format)?;
        if let Some(detail) = schema_field_diff(&first_schema, &other_batch.schema()) {
            return Err(OpsError::DenseMappingMismatch {
                axis,
                key: key.to_string(),
                detail: format!("input 0 vs input {}: {}", per.0, detail),
            });
        }
    }
    Ok(())
}

/// Construct the streaming obs predicate-index builder using the same
/// preset resolution and option defaults that
/// [`scx_engine::build_and_write_conversion_predicate_indexes`] applies.
fn obs_predicate_index_builder(
    schema: arrow::datatypes::SchemaRef,
    index_options: &ConversionPredicateIndexOptions,
) -> Result<scx_engine::ObsPredicateIndexBuilder> {
    let preset_obs = match index_options.index_preset.as_deref() {
        Some(name) => scx_engine::index_preset_columns(name)
            .map(|p| p.obs_columns.iter().map(|s| (*s).to_string()).collect())
            .unwrap_or_default(),
        None => Vec::new(),
    };
    let build_opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: index_options.index_obs.clone(),
        preset_columns: preset_obs,
        auto_threshold: index_options.index_auto_threshold,
        high_cardinality_threshold: 100_000,
    };
    scx_engine::ObsPredicateIndexBuilder::new(schema, &build_opts).map_err(OpsError::Engine)
}

/// Phase 3b: which dense-mapping axis to stream during merge.
#[derive(Clone, Copy)]
enum DenseMappingAxis {
    Obsm,
    Varm,
}

impl DenseMappingAxis {
    fn prefix(self) -> &'static str {
        match self {
            DenseMappingAxis::Obsm => "obsm",
            DenseMappingAxis::Varm => "varm",
        }
    }

    fn shard_type(self) -> SectionType {
        match self {
            DenseMappingAxis::Obsm => SectionType::ObsmEmbeddingShard,
            DenseMappingAxis::Varm => SectionType::VarmEmbeddingShard,
        }
    }

    fn legacy_type(self) -> SectionType {
        match self {
            DenseMappingAxis::Obsm => SectionType::ObsmEmbedding,
            DenseMappingAxis::Varm => SectionType::VarmEmbedding,
        }
    }
}

/// Phase 3b: stream merge of a global (modality_id == 0) dense mapping
/// axis (obsm or varm) shard-by-shard.
///
/// Row semantics differ by axis:
/// - **obsm**: rows align with `obs`, which is concatenated across
///   inputs, so this helper walks every input's shards in order and
///   re-stamps each as the next output shard. Keys missing from any
///   input are dropped (existing semantic).
/// - **varm**: rows align with `var`, which is **shared** across
///   inputs (validated by var-identity check at merge entry).
///   Concatenating varm would duplicate rows, so this helper takes
///   only input 0's varm shards as the canonical output — matching
///   the way `var` itself is taken from input 0.
///
/// Legacy single-section inputs are treated as a one-source-shard
/// input per the Phase 3b spec. Peak memory is one input shard at a time.
fn merge_global_dense_mapping_sharded(
    readers: &[ScxReader],
    writer: &mut ScxWriter,
    axis: DenseMappingAxis,
    n_rows_total: u64,
) -> Result<()> {
    use std::collections::BTreeSet;
    let prefix = axis.prefix();
    let shard_type = axis.shard_type();
    let legacy_type = axis.legacy_type();
    let prefix_slash = format!("{prefix}/");

    // varm rows align with the shared var axis; take input 0 only.
    let source_readers: &[ScxReader] = match axis {
        DenseMappingAxis::Obsm => readers,
        DenseMappingAxis::Varm => &readers[..1],
    };

    // The **union** across every source input, not input 0's set. Taking it
    // from `readers[0]` alone (§6.4) meant a key only a later input carried was
    // never considered — so merging files where the first happened to lack
    // `X_umap` produced an atlas with no UMAP and nothing to notice it by. For
    // `varm`, `source_readers` is `&readers[..1]`, so the union *is* input 0's
    // set and this changes nothing on that axis.
    let keys: BTreeSet<String> = source_readers
        .iter()
        .flat_map(|reader| reader.catalog().entries.iter())
        .filter(|e| e.modality_id == 0)
        .filter_map(|e| {
            if e.section_type == shard_type {
                let stem = e.name.strip_prefix(&prefix_slash)?;
                // Global obsm/varm names: `{prefix}/{key}_shard_{idx}`. We
                // need the key, not the multimodal subpath, so reject
                // anything with a `/` (those belong to a per-modality
                // namespace).
                if stem.contains('/') {
                    return None;
                }
                let pos = stem.rfind("_shard_")?;
                Some(stem[..pos].to_string())
            } else if e.section_type == legacy_type {
                let stem = e.name.strip_prefix(&prefix_slash)?;
                if stem.contains('/') || stem.contains("_shard_") {
                    return None;
                }
                Some(stem.to_string())
            } else {
                None
            }
        })
        .collect();

    for key in &keys {
        // Validate presence in every *source* input before writing
        // anything. For obsm, source_readers == readers (concatenate).
        // For varm, source_readers == &readers[..1] (input 0 only).
        let mut per_input: Vec<(
            usize,
            Vec<&scx_format_io::catalog::FullCatalogEntry>,
            Option<&scx_format_io::catalog::FullCatalogEntry>,
        )> = Vec::with_capacity(source_readers.len());
        let shard_name_prefix = format!("{prefix_slash}{key}_shard_");
        let legacy_name = format!("{prefix_slash}{key}");
        for (idx, reader) in source_readers.iter().enumerate() {
            // Already sorted by row_start by the catalog helper.
            let shards =
                reader
                    .catalog()
                    .dense_mapping_shards_sorted(shard_type, 0, &shard_name_prefix);
            let legacy = if shards.is_empty() {
                reader.catalog().entries.iter().find(|e| {
                    e.section_type == legacy_type && e.modality_id == 0 && e.name == legacy_name
                })
            } else {
                None
            };
            if shards.is_empty() && legacy.is_none() {
                // §6.4: this was a bare `continue 'next_key` — the key was
                // dropped from the output with no diagnostic, while a *layer*
                // in this exact position was already a hard `LayerMissing`
                // naming the file index. Same loss, same unrecoverability, so
                // now the same answer.
                return Err(OpsError::DenseMappingMissing {
                    axis: prefix,
                    key: key.clone(),
                    file_index: idx,
                    total: source_readers.len(),
                });
            }
            per_input.push((idx, shards, legacy));
        }

        // Validate every input's first batch schema against input 0's
        // before writing any shard for this key. Catches embedding-
        // width / dtype drift (e.g. PCA 50 vs 100) at merge time
        // rather than at `read_obsm()` time.
        validate_dense_mapping_schemas(prefix, key, source_readers, &per_input)?;

        // All inputs present — stream-write the merged shard chain.
        let mut out_shard_idx: u32 = 0;
        let mut cumulative_rows: u64 = 0;
        for (idx, shards, legacy) in &per_input {
            let reader = &source_readers[*idx];
            if let Some(entry) = legacy {
                let batch = reader
                    .read_dense_mapping_entry(entry)
                    .map_err(OpsError::Format)?;
                let n = batch.num_rows() as u64;
                match axis {
                    DenseMappingAxis::Obsm => writer.write_obsm_shard(
                        key,
                        out_shard_idx,
                        cumulative_rows,
                        n,
                        n_rows_total,
                        &batch,
                    )?,
                    DenseMappingAxis::Varm => writer.write_varm_shard(
                        key,
                        out_shard_idx,
                        cumulative_rows,
                        n,
                        n_rows_total,
                        &batch,
                    )?,
                }
                cumulative_rows += n;
                out_shard_idx += 1;
            } else {
                // `shards` is already sorted by row_start (catalog helper).
                for &shard_entry in shards {
                    let batch = reader
                        .read_dense_mapping_entry(shard_entry)
                        .map_err(OpsError::Format)?;
                    let n = batch.num_rows() as u64;
                    match axis {
                        DenseMappingAxis::Obsm => writer.write_obsm_shard(
                            key,
                            out_shard_idx,
                            cumulative_rows,
                            n,
                            n_rows_total,
                            &batch,
                        )?,
                        DenseMappingAxis::Varm => writer.write_varm_shard(
                            key,
                            out_shard_idx,
                            cumulative_rows,
                            n,
                            n_rows_total,
                            &batch,
                        )?,
                    }
                    cumulative_rows += n;
                    out_shard_idx += 1;
                }
            }
        }
    }
    Ok(())
}

/// Phase 3b: per-modality counterpart to
/// [`merge_global_dense_mapping_sharded`]. Walks each input's catalog
/// for `obsm/{mname}/{key}_shard_*` (sharded) or `obsm/{mname}/{key}`
/// (legacy), filtered by `modality_id`, and re-stamps via the new
/// `write_obsm_shard_for` / `write_varm_shard_for` writer APIs. Keys
/// missing from any input are dropped **with a warning** (the global
/// helper hard-errors instead — see §6.4 and the note at the `continue`
/// below). Inputs that have no entries
/// for this modality+axis combination are also tolerated — the helper
/// is a no-op when there's nothing to merge (consistent with the
/// existing varm-not-present behaviour in multimodal files).
fn merge_per_modality_dense_mapping_sharded(
    readers: &[ScxReader],
    writer: &mut ScxWriter,
    axis: DenseMappingAxis,
    modality_id: u8,
    modality_name: &str,
    n_rows_total: u64,
) -> Result<()> {
    use std::collections::BTreeSet;
    let prefix = axis.prefix();
    let shard_type = axis.shard_type();
    let legacy_type = axis.legacy_type();
    let key_prefix = format!("{prefix}/{modality_name}/");

    // Per-modality varm rows align with the shared per-modality `var`
    // axis (validated identical across inputs), so take input 0 only —
    // same rule as the global helper.
    let source_readers: &[ScxReader] = match axis {
        DenseMappingAxis::Obsm => readers,
        DenseMappingAxis::Varm => &readers[..1],
    };

    // The union across source inputs, for the same reason as the global helper:
    // input 0's set alone made a key only a later input carried invisible. What
    // happens to a key some input lacks differs, though — see the `continue`
    // below.
    let keys: BTreeSet<String> = source_readers
        .iter()
        .flat_map(|reader| reader.catalog().entries.iter())
        .filter(|e| e.modality_id == modality_id && e.name.starts_with(&key_prefix))
        .filter_map(|e| {
            let stem = e.name.strip_prefix(&key_prefix)?;
            if e.section_type == shard_type {
                let pos = stem.rfind("_shard_")?;
                Some(stem[..pos].to_string())
            } else if e.section_type == legacy_type {
                if stem.contains("_shard_") {
                    None
                } else {
                    Some(stem.to_string())
                }
            } else {
                None
            }
        })
        .collect();

    'next_key: for key in &keys {
        let shard_name_prefix = format!("{key_prefix}{key}_shard_");
        let legacy_name = format!("{key_prefix}{key}");
        let mut per_input: Vec<(
            usize,
            Vec<&scx_format_io::catalog::FullCatalogEntry>,
            Option<&scx_format_io::catalog::FullCatalogEntry>,
        )> = Vec::with_capacity(source_readers.len());
        for (idx, reader) in source_readers.iter().enumerate() {
            // Already sorted by row_start by the catalog helper.
            let shards = reader.catalog().dense_mapping_shards_sorted(
                shard_type,
                modality_id,
                &shard_name_prefix,
            );
            let legacy = if shards.is_empty() {
                reader.catalog().entries.iter().find(|e| {
                    e.section_type == legacy_type
                        && e.modality_id == modality_id
                        && e.name == legacy_name
                })
            } else {
                None
            };
            if shards.is_empty() && legacy.is_none() {
                // **Not** the global helper's hard error, deliberately. That
                // helper's tolerance was an accident of a bare `continue`; this
                // one is documented above as intentional, because an input may
                // legitimately carry nothing at all for a given modality+axis
                // and refusing the whole merge over that would be refusing the
                // ordinary multimodal case. What was wrong here was only the
                // silence — a dropped key is now named. Review §6.4 measured
                // the global path; narrowing the hard error to it is a
                // deliberate scope choice, not an oversight.
                log::warn!(
                    "scx merge: dropping {prefix}['{modality_name}/{key}'] — input file \
                     {idx} of {} does not carry it. Every input must have a key for it \
                     to survive the merge.",
                    source_readers.len()
                );
                continue 'next_key;
            }
            per_input.push((idx, shards, legacy));
        }

        // Mirror the global helper's pre-write schema validation.
        validate_dense_mapping_schemas(prefix, key, source_readers, &per_input)?;

        let mut out_shard_idx: u32 = 0;
        let mut cumulative_rows: u64 = 0;
        for (idx, shards, legacy) in &per_input {
            let reader = &source_readers[*idx];
            if let Some(entry) = legacy {
                let batch = reader
                    .read_dense_mapping_entry(entry)
                    .map_err(OpsError::Format)?;
                let n = batch.num_rows() as u64;
                match axis {
                    DenseMappingAxis::Obsm => writer.write_obsm_shard_for(
                        modality_id,
                        key,
                        out_shard_idx,
                        cumulative_rows,
                        n,
                        n_rows_total,
                        &batch,
                    )?,
                    DenseMappingAxis::Varm => writer.write_varm_shard_for(
                        modality_id,
                        key,
                        out_shard_idx,
                        cumulative_rows,
                        n,
                        n_rows_total,
                        &batch,
                    )?,
                }
                cumulative_rows += n;
                out_shard_idx += 1;
            } else {
                // `shards` is already sorted by row_start (catalog helper).
                for &shard_entry in shards {
                    let batch = reader
                        .read_dense_mapping_entry(shard_entry)
                        .map_err(OpsError::Format)?;
                    let n = batch.num_rows() as u64;
                    match axis {
                        DenseMappingAxis::Obsm => writer.write_obsm_shard_for(
                            modality_id,
                            key,
                            out_shard_idx,
                            cumulative_rows,
                            n,
                            n_rows_total,
                            &batch,
                        )?,
                        DenseMappingAxis::Varm => writer.write_varm_shard_for(
                            modality_id,
                            key,
                            out_shard_idx,
                            cumulative_rows,
                            n,
                            n_rows_total,
                            &batch,
                        )?,
                    }
                    cumulative_rows += n;
                    out_shard_idx += 1;
                }
            }
        }
    }
    Ok(())
}

/// Yield this input's obs as a sequence of shard batches, each at most
/// `shard_target_rows` rows. Sharded inputs pass through via
/// [`ScxReader::obs_shards`]; legacy single-section inputs are sliced
/// into chunks of `shard_target_rows` so the output keeps a uniform
/// shard size regardless of the input layout.
pub(crate) fn input_obs_chunks<'a>(
    reader: &'a ScxReader,
    shard_target_rows: u64,
) -> Result<Box<dyn Iterator<Item = Result<RecordBatch>> + 'a>> {
    if reader.obs_metadata_shard_count() > 0 {
        Ok(Box::new(
            reader.obs_shards().map(|r| r.map_err(OpsError::Format)),
        ))
    } else {
        let single = reader.read_obs().map_err(OpsError::Format)?;
        Ok(Box::new(SingleObsChunker {
            batch: single,
            shard_target_rows: shard_target_rows.max(1) as usize,
            cursor: 0,
        }))
    }
}

/// Slices a single legacy obs `RecordBatch` into shard-sized chunks
/// without allocating a new buffer per chunk — `RecordBatch::slice`
/// shares the underlying Arrow array.
struct SingleObsChunker {
    batch: RecordBatch,
    shard_target_rows: usize,
    cursor: usize,
}

impl Iterator for SingleObsChunker {
    type Item = Result<RecordBatch>;
    fn next(&mut self) -> Option<Self::Item> {
        let n = self.batch.num_rows();
        if self.cursor >= n {
            return None;
        }
        let take = std::cmp::min(self.shard_target_rows, n - self.cursor);
        let chunk = self.batch.slice(self.cursor, take);
        self.cursor += take;
        Some(Ok(chunk))
    }
}

/// Apply [`crate::merge_options::UnsPolicy`] across input `uns`
/// sections. Returns the JSON payload to write (or `None` when nothing
/// should be written — e.g. no input had uns, or `Namespace` ended up
/// with an empty wrapper). Conflicts are counted into
/// `conflicts_warned` for the provenance stamp.
fn combine_uns_for_merge(
    readers: &[ScxReader],
    policy: crate::merge_options::UnsPolicy,
    conflicts_warned: &mut usize,
) -> Result<Option<serde_json::Value>> {
    // Collect each input's global uns (or `None` when absent) and
    // hand off to the per-input combiner. Splitting the read step out
    // lets the per-modality multimodal path reuse the same policy
    // logic with `read_uns_for(modality_id)`.
    // OE2: distinguish genuine absence from corruption. `.ok()` would
    // collapse SectionNotFound, byte/checksum failure, and JSON-parse error
    // all to `None`, silently dropping corrupt uns (or, under RequireEqual,
    // misreporting it as "input has no uns section"). Only a missing section
    // is `None`; any other read error propagates.
    let mut per_input: Vec<Option<serde_json::Value>> = Vec::with_capacity(readers.len());
    for r in readers {
        match r.read_uns() {
            Ok(v) => per_input.push(Some(v)),
            Err(scx_format_io::ScxError::SectionNotFound(_)) => per_input.push(None),
            Err(e) => return Err(e.into()),
        }
    }
    combine_uns_per_input(per_input, policy, conflicts_warned)
}

/// Lower-level uns merge: combine an already-collected
/// `per_input` vec under the given [`UnsPolicy`]. Used by
/// [`combine_uns_for_merge`] for global uns and by the multimodal
/// merge path for each modality's uns. `None` entries represent
/// inputs that had no uns section.
fn combine_uns_per_input(
    per_input: Vec<Option<serde_json::Value>>,
    policy: crate::merge_options::UnsPolicy,
    conflicts_warned: &mut usize,
) -> Result<Option<serde_json::Value>> {
    use crate::merge_options::UnsPolicy;

    // Pull out the first non-None as the canonical body for `First` /
    // `Summary`; bail early when no input has uns. OE10: `per_input` is
    // consumed by value so the chosen canonical payload is moved out rather
    // than deep-cloned (these uns blobs can be large).
    let first_present = per_input.iter().position(|v| v.is_some());
    let Some(first_idx) = first_present else {
        return Ok(None);
    };

    /// Move the `Some(value)` at `idx` out of an owned vec.
    fn take_canonical(per_input: Vec<Option<serde_json::Value>>, idx: usize) -> serde_json::Value {
        per_input
            .into_iter()
            .nth(idx)
            .expect("first_present index in bounds")
            .expect("first_present index points at Some")
    }

    match policy {
        UnsPolicy::First => {
            // Warn on any input whose uns differs from the chosen one
            // so silent data loss is at least visible in logs.
            {
                let canonical = per_input[first_idx].as_ref().unwrap();
                for (i, other) in per_input.iter().enumerate() {
                    if i == first_idx {
                        continue;
                    }
                    if let Some(o) = other {
                        if o != canonical {
                            *conflicts_warned += 1;
                            log::warn!(
                                "merge: input {i}'s uns differs from input {first_idx}; \
                                 dropping under uns_policy=first"
                            );
                        }
                    }
                }
            }
            Ok(Some(take_canonical(per_input, first_idx)))
        }
        UnsPolicy::RequireEqual => {
            {
                let canonical = per_input[first_idx].as_ref().unwrap();
                for (i, other) in per_input.iter().enumerate() {
                    if i == first_idx {
                        continue;
                    }
                    if let Some(o) = other {
                        if o != canonical {
                            return Err(OpsError::UnsConflict {
                                policy: "require-equal",
                                detail: format!(
                                    "input {first_idx} vs input {i}: uns differs (see input \
                                     files for full payload)"
                                ),
                            });
                        }
                    } else {
                        return Err(OpsError::UnsConflict {
                            policy: "require-equal",
                            detail: format!(
                                "input {i} has no uns section but input {first_idx} does"
                            ),
                        });
                    }
                }
            }
            Ok(Some(take_canonical(per_input, first_idx)))
        }
        UnsPolicy::Namespace => {
            let mut obj = serde_json::Map::new();
            for (i, v) in per_input.into_iter().enumerate() {
                if let Some(payload) = v {
                    obj.insert(format!("input_{i}"), payload);
                }
            }
            if obj.is_empty() {
                Ok(None)
            } else {
                Ok(Some(serde_json::Value::Object(obj)))
            }
        }
        UnsPolicy::Summary => {
            let mut conflicts: Vec<serde_json::Value> = Vec::new();
            {
                let canonical = per_input[first_idx].as_ref().unwrap();
                for (i, other) in per_input.iter().enumerate() {
                    if i == first_idx {
                        continue;
                    }
                    match other {
                        Some(o) if o != canonical => {
                            *conflicts_warned += 1;
                            conflicts.push(serde_json::json!({
                                "input": i,
                                "status": "differs",
                            }));
                        }
                        None => {
                            conflicts.push(serde_json::json!({
                                "input": i,
                                "status": "missing",
                            }));
                        }
                        _ => {}
                    }
                }
            }
            let canonical = take_canonical(per_input, first_idx);
            if conflicts.is_empty() {
                Ok(Some(canonical))
            } else {
                let mut payload = match canonical {
                    serde_json::Value::Object(m) => m,
                    other => {
                        let mut wrapper = serde_json::Map::new();
                        wrapper.insert("_scx_canonical".to_string(), other);
                        wrapper
                    }
                };
                payload.insert(
                    "_scx_uns_conflicts".to_string(),
                    serde_json::Value::Array(conflicts),
                );
                Ok(Some(serde_json::Value::Object(payload)))
            }
        }
    }
}
