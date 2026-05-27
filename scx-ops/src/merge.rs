// Merge operation: combine multiple SCX files into one.

use std::path::Path;

use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format::codec_select::select_codec;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::{ScxReader, ShardHeader, SHARD_HEADER_SIZE};

use crate::append::unify_dict_columns;
use crate::error::{OpsError, Result};
use crate::flock::SharedFileLock;
use crate::helpers::encode_value;
use crate::merge_options::MergeOptions;
use crate::predicate_index::{
    requested_columns, user_wants_index, validate_forced_columns, PredicateIndexBuildSummary,
};

/// Merge multiple SCX files into a single output file.
/// All inputs must have the same n_vars.
///
/// Drops `ObsPredicateIndex` / `VarPredicateIndex` sections from the
/// output (the merged row layout invalidates per-shard row ranges).
/// Use [`merge_with_index_options`] to rebuild predicate indexes on the
/// merged file in the same pass.
pub fn merge(input_paths: &[&Path], output_path: &Path) -> Result<()> {
    // `index_auto_threshold = 0` is the sentinel that disables all
    // index work (no forced columns, no preset, no auto-detect). Keeps
    // legacy `merge(...)` byte-identical to its pre-fix behaviour.
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
/// before `provenance`. The returned summary carries per-axis outcomes
/// so the caller can emit user-facing warnings (see
/// `scx-convert::pipeline::process_predicate_index_outcomes` for the
/// reference outcome → `ConvertWarning` mapping).
///
/// Multimodal merge currently cannot persist predicate-index sections
/// per modality — the helper sets `summary.multimodal_skip` and the
/// caller surfaces `ConvertWarning::PredicateIndexSkippedMultimodal`.
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
/// uns-conflict policy, and shard-target overrides. The flat
/// [`merge_with_index_options`] / [`merge`] entry points delegate to
/// this function with the legacy defaults baked in.
pub fn merge_with_options(
    input_paths: &[&Path],
    output_path: &Path,
    options: &MergeOptions,
) -> Result<PredicateIndexBuildSummary> {
    let index_options = &options.index_options;
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

    // Phase 6: validate modality structure consistency. Multimodal
    // merge requires every input to expose the same set of
    // modalities (name + type + n_vars). On mismatch, raise with a
    // clear error directing to extract-then-merge.
    let any_multimodal = readers.iter().any(|r| r.is_multimodal());
    if any_multimodal {
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
        merge_multimodal(&readers, input_paths, output_path)?;
        return Ok(PredicateIndexBuildSummary {
            result: None,
            multimodal_skip,
        });
    }

    let total_n_obs: u64 = readers.iter().map(|r| r.n_obs()).sum();

    let first_header = readers[0].header();

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

    // Build output header. codec_id is 0 (None) because actual codec is
    // selected per-shard via select_codec(). `flags` stays 0 — merge
    // intentionally produces a clean output header rather than carrying
    // input flag state forward.
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: total_n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: first_header.shard_target_rows,
        codec_id: 0,
        index_dtype: first_header.index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output_path, out_header)?;

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

    let mut out_shard_idx: u32 = 0;
    let mut cumulative_obs_rows: u64 = 0;
    for reader in &readers {
        for chunk in input_obs_chunks(reader, shard_target_rows)? {
            let chunk = chunk?;
            let unified = unify_dict_columns(&chunk)?;
            if let Some(builder) = obs_index_builder.as_mut() {
                builder.push_shard(&unified, cumulative_obs_rows)?;
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
    for reader in &readers {
        let shards = reader.catalog().shards_sorted();
        for shard_entry in &shards {
            // Read this shard's value_encoding from its header
            let shard_section = reader.section_bytes(shard_entry)?;
            let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
                &shard_section[..SHARD_HEADER_SIZE],
            ))?;
            let shard_value_encoding = ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?;

            let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
            let n_rows = indptr.len() - 1;

            // Convert back to on-disk format using the shard's own value encoding
            let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
            let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
            let mut values_bytes = Vec::new();
            for &v in &data {
                encode_value(&mut values_bytes, v, shard_value_encoding)?;
            }

            // Auto-select optimal codec for this shard's data
            let shard_codec = select_codec(&values_bytes, shard_value_encoding);

            let row_start = cumulative_rows;
            writer.write_csr_shard(
                &indptr_u64,
                &indices_u32,
                &values_bytes,
                shard_codec,
                shard_value_encoding,
                cumulative_rows,
            )?;
            cumulative_rows += n_rows as u64;
            output_shard_row_ranges.push((row_start, cumulative_rows));
        }
    }

    // Merge obsm
    let first_obsm = readers[0].read_all_obsm()?;
    for name in first_obsm.keys() {
        let mut batches = Vec::new();
        for reader in &readers {
            if let Ok(batch) = reader.read_obsm(name) {
                batches.push(batch);
            }
        }
        if batches.len() == readers.len() {
            let merged = concat_batches(&batches[0].schema(), &batches)?;
            writer.write_obsm(name, &merged)?;
        }
    }

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
        // Determine this layer's value encoding from its first shard header
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::catalog::FullCatalogEntry> = readers[0]
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();
        let layer_value_encoding = if let Some(first_entry) = layer_shard_entries.first() {
            let section = readers[0].section_bytes(first_entry)?;
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?
        } else {
            ValueEncoding::Uint8
        };

        // Accumulate and flush at shard_target_rows, matching compact.rs pattern
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;
        let mut emitted_layer_rows = 0u64;

        for (file_idx, reader) in readers.iter().enumerate() {
            let layer = reader
                .read_layer(layer_name)
                .map_err(|_| OpsError::LayerMissing {
                    name: layer_name.clone(),
                    file_index: file_idx,
                })?;
            for row_idx in 0..layer.shape.0 {
                let row_start = layer.indptr[row_idx] as usize;
                let row_end = layer.indptr[row_idx + 1] as usize;
                for j in row_start..row_end {
                    layer_indices.push(layer.indices[j] as u32);
                    encode_value(&mut layer_values, layer.data[j], layer_value_encoding)?;
                }
                let prev = *layer_indptr.last().unwrap();
                layer_indptr.push(prev + (row_end - row_start) as u64);
                layer_row_count += 1;

                if layer_row_count >= shard_target as u64 {
                    // Auto-select codec per layer shard
                    let layer_shard_codec = select_codec(&layer_values, layer_value_encoding);
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

        // Flush remaining layer rows
        if layer_row_count > 0 {
            // Auto-select codec for remaining layer shard
            let layer_shard_codec = select_codec(&layer_values, layer_value_encoding);
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
    all_prov_entries.push(ProvenanceEntry {
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64,
        action: "merge".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: format!(
            "{{\"n_inputs\":{},\"assume_identical_var\":{},\"uns_policy\":\"{}\",\
             \"uns_conflicts_warned\":{}}}",
            input_paths.len(),
            options.assume_identical_var,
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
    writer.write_provenance(all_prov_entries)?;

    writer.finish()?;
    Ok(PredicateIndexBuildSummary {
        result: index_result,
        multimodal_skip: None,
    })
}

/// Phase 6: merge multimodal SCX files with matching modality
/// structure. Per-modality CSR shards and per-modality layers are
/// concatenated in input order with `row_start` adjusted for the
/// cumulative global obs offset; global `obsm` and per-modality
/// `obsm` are concatenated row-wise (any key missing in any input is
/// dropped, matching single-modality semantics); per-modality var
/// and per-modality / global `uns` are copied from the first input
/// (var is already validated identical by the modality-table check;
/// uns is treated as a single source of truth). Per-modality CSC
/// sidecars are dropped (caller can `--rebuild-csc`).
fn merge_multimodal(
    readers: &[ScxReader],
    input_paths: &[&Path],
    output_path: &Path,
) -> Result<()> {
    let table = readers[0]
        .modality_table()
        .ok_or_else(|| {
            scx_format::ScxError::InvalidCatalog(
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

    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: 0,
        n_obs: total_n_obs,
        n_vars: max_n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: first_header.shard_target_rows,
        codec_id: 0,
        index_dtype: first_header.index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output_path, out_header)?;

    // Phase 2b: Global obs streams shard-by-shard exactly as the
    // single-modality path does (see `merge_with_options`). The
    // multimodal merge_multimodal entry currently doesn't surface
    // `MergeOptions` to its callers — the orchestrator
    // (`merge_with_options`) doesn't pass policy fields here because
    // multimodal merge has a separate uns / var ownership model
    // (per-modality var, per-modality uns). Stream obs with the
    // default shard target from the first input and skip var-identity
    // / uns-policy enforcement (handled per-modality below).
    let shard_target_rows: u64 = first_header.shard_target_rows as u64;
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

    // Global obsm: concatenate across inputs row-wise. Drop a key if
    // any input is missing it (consistent with single-modality merge).
    let first_obsm = readers[0].read_all_obsm()?;
    for name in first_obsm.keys() {
        let mut batches = Vec::new();
        for reader in readers {
            if let Ok(batch) = reader.read_obsm(name) {
                batches.push(batch);
            }
        }
        if batches.len() == readers.len() {
            let merged = concat_batches(&batches[0].schema(), &batches)?;
            writer.write_obsm(name, &merged)?;
        }
    }

    // Register modalities in input order.
    for info in &table.entries {
        let codec = CodecId::from_u8(info.default_codec_id)
            .ok_or(OpsError::UnknownCodec(info.default_codec_id))?;
        let value_encoding = ValueEncoding::from_u8(info.default_value_encoding)
            .ok_or(OpsError::UnknownValueEncoding(info.default_value_encoding))?;
        writer.add_modality(&info.name, info.modality_type, codec, value_encoding, false)?;
        writer.set_modality_n_vars(writer.n_modalities() as u8, info.n_vars)?;
    }

    // Per-modality var (from first input — already validated identical).
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let var = readers[0].read_var_for(modality_id)?;
        writer.write_var_for(modality_id, &var)?;
        let _ = info; // suppress unused if no other field needed
    }

    // Per-modality CSR shards: concatenate across inputs with row_start
    // adjusted for the cumulative global obs offset. Within an input,
    // the modality's shards collectively cover the input's n_obs rows;
    // after one input we advance the offset by that input's n_obs so
    // the next input's shards line up against the merged obs.
    for (idx, _info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let mut input_offset: u64 = 0;
        for reader in readers {
            let entries = reader.catalog().csr_shards_for_modality(modality_id);
            for shard_entry in entries {
                let section = reader.section_bytes(shard_entry)?;
                let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
                    &section[..SHARD_HEADER_SIZE],
                ))?;
                let shard_value_encoding = ValueEncoding::from_u8(sh.value_encoding)
                    .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?;
                let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
                let shard_local_row_start =
                    shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

                let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
                let indices_u32: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
                let mut values_bytes = Vec::new();
                for &v in &data {
                    encode_value(&mut values_bytes, v, shard_value_encoding)?;
                }
                let shard_codec = select_codec(&values_bytes, shard_value_encoding);
                writer.write_csr_shard_for(
                    modality_id,
                    &indptr_u64,
                    &indices_u32,
                    &values_bytes,
                    shard_codec,
                    shard_value_encoding,
                    input_offset + shard_local_row_start,
                )?;
            }
            input_offset += reader.n_obs();
        }
    }

    // Per-modality obsm: concatenate across inputs row-wise. Drop if
    // any input is missing it (consistent with single-modality merge).
    for (idx, info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        let prefix = format!("obsm/{}/", info.name);
        let mut obsm_keys = std::collections::BTreeSet::new();
        for entry in &readers[0].catalog().entries {
            if entry.section_type == SectionType::ObsmEmbedding
                && entry.modality_id == modality_id
                && entry.name.starts_with(&prefix)
            {
                if let Some(k) = entry.name.strip_prefix(&prefix) {
                    obsm_keys.insert(k.to_string());
                }
            }
        }
        for key in obsm_keys {
            let mut batches = Vec::new();
            let mut all_present = true;
            for reader in readers {
                match reader.read_obsm_for(modality_id, &key) {
                    Ok(b) => batches.push(b),
                    Err(_) => {
                        all_present = false;
                        break;
                    }
                }
            }
            if all_present && !batches.is_empty() {
                let merged = concat_batches(&batches[0].schema(), &batches)?;
                writer.write_obsm_for(modality_id, &key, &merged)?;
            }
        }
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
            // Probe value_encoding from the first input that has a shard for
            // this (modality, layer); fall back to the modality default.
            let layer_value_encoding = {
                let mut enc: Option<ValueEncoding> = None;
                for reader in readers {
                    let shards = reader
                        .catalog()
                        .layer_csr_shards_for_modality(modality_id, layer_name);
                    if let Some(first) = shards.first() {
                        let section = reader.section_bytes(first)?;
                        let sh = ShardHeader::read_from(&mut std::io::Cursor::new(
                            &section[..SHARD_HEADER_SIZE],
                        ))?;
                        enc = Some(
                            ValueEncoding::from_u8(sh.value_encoding)
                                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?,
                        );
                        break;
                    }
                }
                enc.unwrap_or_else(|| {
                    ValueEncoding::from_u8(info.default_value_encoding)
                        .unwrap_or(ValueEncoding::Uint8)
                })
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
                            let codec = select_codec(&l_values, layer_value_encoding);
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
                let codec = select_codec(&l_values, layer_value_encoding);
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

    // Per-modality uns: copy from first input (single source of truth).
    for (idx, _info) in table.entries.iter().enumerate() {
        let modality_id = (idx + 1) as u8;
        if let Ok(uns) = readers[0].read_uns_for(modality_id) {
            writer.write_uns_for(modality_id, &uns)?;
        }
    }

    // Global uns from first input.
    if let Ok(uns) = readers[0].read_uns() {
        writer.write_uns(&uns)?;
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
        params_json: format!("{{\"n_inputs\":{}}}", input_paths.len()),
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
    writer.finish()?;
    Ok(())
}

/// Phase F.4 helper: two modality tables match iff they have the same
/// length and every entry agrees on (name, modality_type, n_vars).
/// Two `None`s also match (both inputs single-modality). Mixed
/// `Some` / `None` does not match.
fn modality_tables_match(
    a: Option<&scx_format::ModalityTable>,
    b: Option<&scx_format::ModalityTable>,
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
/// description of the first observed difference.
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

/// Yield this input's obs as a sequence of shard batches, each at most
/// `shard_target_rows` rows. Sharded inputs pass through via
/// [`ScxReader::obs_shards`]; legacy single-section inputs are sliced
/// into chunks of `shard_target_rows` so the output keeps a uniform
/// shard size regardless of the input layout.
fn input_obs_chunks<'a>(
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
    use crate::merge_options::UnsPolicy;

    // Collect each input's uns (or `None` when absent). We need the
    // full set for any policy that compares across inputs.
    let mut per_input: Vec<Option<serde_json::Value>> = Vec::with_capacity(readers.len());
    for reader in readers {
        match reader.read_uns() {
            Ok(v) => per_input.push(Some(v)),
            Err(_) => per_input.push(None),
        }
    }

    // Pull out the first non-None as the canonical body for `First` /
    // `Summary`; bail early when no input has uns.
    let first_present = per_input.iter().position(|v| v.is_some());
    let Some(first_idx) = first_present else {
        return Ok(None);
    };

    match policy {
        UnsPolicy::First => {
            // Warn on any input whose uns differs from the chosen one
            // so silent data loss is at least visible in logs.
            let canonical = per_input[first_idx].as_ref().unwrap().clone();
            for (i, other) in per_input.iter().enumerate() {
                if i == first_idx {
                    continue;
                }
                if let Some(o) = other {
                    if o != &canonical {
                        *conflicts_warned += 1;
                        log::warn!(
                            "merge: input {i}'s uns differs from input {first_idx}; \
                             dropping under uns_policy=first"
                        );
                    }
                }
            }
            Ok(Some(canonical))
        }
        UnsPolicy::RequireEqual => {
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
                        detail: format!("input {i} has no uns section but input {first_idx} does"),
                    });
                }
            }
            Ok(Some(canonical.clone()))
        }
        UnsPolicy::Namespace => {
            let mut obj = serde_json::Map::new();
            for (i, v) in per_input.iter().enumerate() {
                if let Some(payload) = v {
                    obj.insert(format!("input_{i}"), payload.clone());
                }
            }
            if obj.is_empty() {
                Ok(None)
            } else {
                Ok(Some(serde_json::Value::Object(obj)))
            }
        }
        UnsPolicy::Summary => {
            let canonical = per_input[first_idx].as_ref().unwrap();
            let mut conflicts: Vec<serde_json::Value> = Vec::new();
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
            if conflicts.is_empty() {
                Ok(Some(canonical.clone()))
            } else {
                let mut payload = match canonical.clone() {
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
