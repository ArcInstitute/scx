// Compact operation: rewrite file without deleted rows or stale catalogs.

use std::path::Path;

use arrow::compute;
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions};
use scx_format::codec_select::select_codec;
use scx_format::header::{FileHeader, MAGIC};
use scx_format::provenance::ProvenanceEntry;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;

use crate::error::Result;
use crate::flock::SharedFileLock;
use crate::helpers::encode_value;
use crate::predicate_index::{
    requested_columns, user_wants_index, validate_forced_columns, PredicateIndexBuildSummary,
};

/// Compact an SCX file: removes deleted rows, stale catalogs, and produces
/// a clean single-catalog file.
///
/// Drops any input predicate indexes — the row layout is re-sharded
/// against the post-deletion row count, so per-shard row ranges in the
/// input index are stale. Use [`compact_with_index_options`] to rebuild
/// predicate indexes against the compacted output in the same pass.
pub fn compact(input_path: &Path, output_path: &Path) -> Result<()> {
    // See `merge` for the `index_auto_threshold = 0` sentinel rationale.
    compact_with_index_options(
        input_path,
        output_path,
        &ConversionPredicateIndexOptions {
            index_obs: Vec::new(),
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 0,
        },
        false,
    )
    .map(|_| ())
}

/// Compact an SCX file and optionally rebuild predicate indexes on the
/// output. See [`merge_with_index_options`](crate::merge_with_index_options)
/// for the shape of `index_options` and the multimodal-skip semantics.
///
/// When `reshape_obs` is true the output's obs metadata is written as
/// row-sharded [`SectionType::ObsMetadataShard`] sections (bounded by the
/// file's `shard_target_rows`) instead of a single legacy `ObsMetadata`
/// section. This is the in-place migration path for files written before
/// sharded obs metadata existed; it is idempotent on already-sharded inputs.
pub fn compact_with_index_options(
    input_path: &Path,
    output_path: &Path,
    index_options: &ConversionPredicateIndexOptions,
    reshape_obs: bool,
) -> Result<PredicateIndexBuildSummary> {
    // Acquire shared lock to prevent concurrent writers from modifying the
    // file while we read it. The lock is held until `_lock` is dropped.
    let _lock = SharedFileLock::acquire(input_path)?;
    let reader = ScxReader::open(input_path)?;
    let in_header = reader.header().clone();

    // Phase 6: dispatch to the multimodal compact path. The single-modality
    // path below assumes one modality and would silently flatten the
    // ModalityTable.
    if reader.is_multimodal() {
        // Predicate indexes are unimodal-only today; capture the skip so
        // the caller can emit a single `PredicateIndexSkippedMultimodal`
        // warning.
        let multimodal_skip =
            user_wants_index(index_options).then(|| requested_columns(index_options));
        compact_multimodal(reader, in_header, input_path, output_path, reshape_obs)?;
        return Ok(PredicateIndexBuildSummary {
            result: None,
            multimodal_skip,
        });
    }
    let in_header = &in_header;

    // Load deletion vectors
    let dv = reader.read_deletion_vectors()?;

    // Read obs and build deletion mask. `read_obs` / `read_var`
    // transparently handle both legacy single-section and row-sharded
    // Phase 2 layouts; the assembled batch is what the deletion-mask
    // + filter logic expects.
    let obs = reader.read_obs()?;
    let var = reader.read_var()?;

    let n_obs = in_header.n_obs as usize;
    let n_vars = in_header.n_vars;

    // Build a boolean array: true = keep, false = deleted
    let keep_mask = build_keep_mask(n_obs, &dv, reader.catalog());

    // Filter obs
    let filtered_obs = if let Some(ref mask) = keep_mask {
        let bool_array = arrow::array::BooleanArray::from(mask.clone());
        compute::filter_record_batch(&obs, &bool_array)?
    } else {
        obs
    };

    // Fail fast if forced index columns are missing from the compacted
    // schemas. Must run before `ScxWriter::new` materialises a temp
    // output file so an invalid request leaves no on-disk artefact.
    validate_forced_columns(index_options, &filtered_obs.schema(), &var.schema())?;

    let new_n_obs = filtered_obs.num_rows();

    // CSC sidecars (column-major shards) are dropped by `compact`: the
    // operation re-shards CSR rows on a different row layout, so any
    // input CSC shards would silently reference stale row indices.
    // Caller can opt back in via `--rebuild-csc` on the CLI to re-run
    // `build-csc` against the compacted output. Phase H.2.
    let had_csc = in_header.has_csc();
    if had_csc {
        log::warn!(
            "compact dropped CSC shards from {input}: rerun \
             `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar",
            input = input_path.display()
        );
    }
    // Carry all input flags except `has_deletion_vectors` (bit 5) — the
    // compacted output applies the deletion vector and drops it — and
    // `has_csc` (bit 0) — the CSC sidecar is dropped explicitly above.
    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);

    // Set up output header
    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: out_flags,
        n_obs: new_n_obs as u64,
        n_vars,
        nnz: 0, // will be set by finish
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: 0,
        index_dtype: in_header.index_dtype,
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

    // Copy obsm flag if present
    let has_obsm = in_header.has_obsm();

    let mut writer = ScxWriter::new(output_path, out_header)?;
    write_obs_section(
        &mut writer,
        &filtered_obs,
        reshape_obs,
        in_header.shard_target_rows,
    )?;
    writer.write_var(&var)?;

    // Process CSR shards: decode, filter deleted rows, re-shard.
    // Read value_encoding from first shard header for the main X matrix.
    let shards = reader.catalog().shards_sorted();
    let value_encoding = {
        if !shards.is_empty() {
            let section = reader.section_bytes(shards[0])?;
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            ValueEncoding::from_u8(sh.value_encoding).ok_or(
                crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
            )?
        } else {
            ValueEncoding::Uint8
        }
    };
    let shard_target = in_header.shard_target_rows;

    // Accumulate filtered CSR data
    let mut acc_indptr: Vec<u64> = vec![0];
    let mut acc_indices: Vec<u32> = Vec::new();
    let mut acc_values: Vec<u8> = Vec::new();
    let mut acc_row_count = 0u64;
    let mut emitted_rows = 0u64; // tracks output row numbering
                                 // Per-output-shard `(row_start, row_end)` ranges captured during the
                                 // re-shard loop so predicate-index builders see the actual post-
                                 // deletion shard boundaries.
    let mut output_shard_row_ranges: Vec<(u64, u64)> = Vec::new();

    for shard_entry in &shards {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        let shard_n_rows = indptr.len() - 1;

        for local_row in 0..shard_n_rows {
            let global_idx = shard_row_start + local_row as u64;

            // Check if deleted
            let deleted = keep_mask
                .as_ref()
                .is_some_and(|mask| !mask[global_idx as usize]);

            if deleted {
                continue;
            }

            let row_start_nnz = indptr[local_row] as usize;
            let row_end_nnz = indptr[local_row + 1] as usize;
            let row_nnz = row_end_nnz - row_start_nnz;

            for j in row_start_nnz..row_end_nnz {
                acc_indices.push(indices[j] as u32);
                // Convert f32 back to raw bytes per value_encoding
                encode_value(&mut acc_values, data[j], value_encoding)?;
            }

            let prev = *acc_indptr.last().unwrap();
            acc_indptr.push(prev + row_nnz as u64);
            acc_row_count += 1;

            // Flush accumulated rows as a shard when hitting target
            if acc_row_count >= shard_target as u64 {
                let shard_row_start = emitted_rows;
                // Auto-select optimal codec for this shard's data
                let shard_codec = select_codec(&acc_values, value_encoding);
                writer.write_csr_shard(
                    &acc_indptr,
                    &acc_indices,
                    &acc_values,
                    shard_codec,
                    value_encoding,
                    shard_row_start,
                )?;
                emitted_rows += acc_row_count;
                output_shard_row_ranges.push((shard_row_start, emitted_rows));
                acc_indptr = vec![0];
                acc_indices.clear();
                acc_values.clear();
                acc_row_count = 0;
            }
        }
    }

    // Flush remaining accumulated rows
    if acc_row_count > 0 {
        let shard_row_start = emitted_rows;
        // Auto-select optimal codec for remaining shard
        let shard_codec = select_codec(&acc_values, value_encoding);
        writer.write_csr_shard(
            &acc_indptr,
            &acc_indices,
            &acc_values,
            shard_codec,
            value_encoding,
            shard_row_start,
        )?;
        emitted_rows += acc_row_count;
        output_shard_row_ranges.push((shard_row_start, emitted_rows));
    }

    // Copy obsm (row-filtered)
    if has_obsm {
        let all_obsm = reader.read_all_obsm()?;
        for (name, batch) in &all_obsm {
            let filtered_batch = if let Some(ref mask) = keep_mask {
                let bool_array = arrow::array::BooleanArray::from(mask.clone());
                compute::filter_record_batch(batch, &bool_array)?
            } else {
                batch.clone()
            };
            writer.write_obsm(name, &filtered_batch)?;
        }
    }

    // Copy uns
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Copy layers (row-filtered)
    let layer_names = reader.layer_names();
    for layer_name in &layer_names {
        // Determine this layer's value encoding from its first shard header
        let layer_prefix = format!("{layer_name}_shard_");
        let layer_shard_entries: Vec<&scx_format::FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&layer_prefix)
            })
            .collect();
        let layer_value_encoding = if let Some(first_entry) = layer_shard_entries.first() {
            let section = reader.section_bytes(first_entry)?;
            let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format::SHARD_HEADER_SIZE],
            ))?;
            ValueEncoding::from_u8(sh.value_encoding).ok_or(
                crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
            )?
        } else {
            ValueEncoding::Uint8
        };

        let layer = reader.read_layer(layer_name)?;
        // Filter and write layer shards
        let mut layer_indptr: Vec<u64> = vec![0];
        let mut layer_indices: Vec<u32> = Vec::new();
        let mut layer_values: Vec<u8> = Vec::new();
        let mut layer_row_count = 0u64;
        let mut layer_shard_idx = 0u32;
        let mut emitted_layer_rows = 0u64;

        for row_idx in 0..layer.shape.0 {
            let deleted = keep_mask.as_ref().is_some_and(|mask| !mask[row_idx]);
            if deleted {
                continue;
            }

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

        if layer_row_count > 0 {
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

    // Copy varm (var-axis; vars are not deleted by compact, so unfiltered).
    // `read_all_varm` transparently assembles legacy + sharded layouts and
    // returns an empty map when absent, so no header-flag guard is needed.
    let mut varm: Vec<_> = reader.read_all_varm()?.into_iter().collect();
    varm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varm {
        writer.write_varm(name, batch)?;
    }

    // Copy varp (var×var; var-axis on both dimensions → unfiltered).
    let mut varp: Vec<_> = reader.read_all_varp()?.into_iter().collect();
    varp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varp {
        writer.write_varp(name, batch)?;
    }

    // Copy obsp (obs×obs COO). Under deletions both COO axes are remapped
    // through the obs keep-mask; entries touching a deleted obs are dropped.
    let mut obsp: Vec<_> = reader.read_all_obsp()?.into_iter().collect();
    obsp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsp {
        let out = match keep_mask {
            Some(ref mask) => filter_obsp_coo(batch, mask)?,
            None => batch.clone(),
        };
        writer.write_obsp(name, &out)?;
    }

    // Predicate indexes: rebuild against the post-deletion obs + the
    // freshly emitted shard row ranges. `user_wants_index` treats a
    // non-zero `index_auto_threshold` as an explicit request — fixes
    // the pre-fix bug where `--index-auto-threshold N` alone was a
    // no-op. Forced-column-missing was caught upfront by
    // `validate_forced_columns`.
    let index_result = if user_wants_index(index_options) {
        Some(build_and_write_conversion_predicate_indexes(
            &mut writer,
            &filtered_obs,
            &var,
            &output_shard_row_ranges,
            n_vars as usize,
            index_options,
        )?)
    } else {
        None
    };

    // Add provenance
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
        action: "compact".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: format!("{{\"reshape_obs\":{reshape_obs}}}"),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;

    writer.finish()?;
    Ok(PredicateIndexBuildSummary {
        result: index_result,
        multimodal_skip: None,
    })
}

/// Read a COO coordinate column (`row` / `col`) as `i64`, accepting either
/// the v1 `Int32` or the v2 `Int64` coordinate width. The Arrow schema
/// self-describes the width (see `scx-format/tests/pairwise_v2_int64.rs`), so
/// compact must round-trip both.
fn obsp_coords_as_i64(batch: &arrow::array::RecordBatch, name: &str) -> Result<Vec<i64>> {
    use arrow::array::{Int32Array, Int64Array};
    let col = batch.column_by_name(name).ok_or_else(|| {
        crate::error::OpsError::InvalidInput(format!("obsp COO column `{name}` is missing"))
    })?;
    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Ok((0..a.len()).map(|i| a.value(i)).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        Ok((0..a.len()).map(|i| a.value(i) as i64).collect())
    } else {
        Err(crate::error::OpsError::InvalidInput(format!(
            "obsp COO column `{name}` is neither Int32 nor Int64"
        )))
    }
}

/// Remap an obsp COO `RecordBatch` (obs×obs) through the obs keep-mask.
///
/// Drops entries whose `row` OR `col` references a deleted obs and renumbers
/// surviving endpoints into the compacted index space. The `n_rows` /
/// `n_cols` schema metadata is rewritten to the kept count; all other
/// metadata keys are preserved. `keep_mask[i] == true` means obs `i` is kept.
///
/// Both the v1 `Int32` and v2 `Int64` coordinate widths are accepted on input;
/// the output width is chosen from the compacted dimension so a >`i32::MAX`
/// axis that survives compaction stays `Int64` (no silent wrap).
fn filter_obsp_coo(
    batch: &arrow::array::RecordBatch,
    keep_mask: &[bool],
) -> Result<arrow::array::RecordBatch> {
    use arrow::array::{Float32Array, Int32Array, Int64Array};
    use arrow::datatypes::{DataType, Field};

    // old obs index -> new obs index (or -1 if the obs was deleted).
    let mut old_to_new = vec![-1i64; keep_mask.len()];
    let mut next = 0i64;
    for (i, &keep) in keep_mask.iter().enumerate() {
        if keep {
            old_to_new[i] = next;
            next += 1;
        }
    }
    let new_dim = next;

    let rows = obsp_coords_as_i64(batch, "row")?;
    let cols = obsp_coords_as_i64(batch, "col")?;
    let data = batch
        .column_by_name("data")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| {
            crate::error::OpsError::InvalidInput(
                "obsp COO column `data` is missing or not Float32".to_string(),
            )
        })?;

    let mut out_rows: Vec<i64> = Vec::new();
    let mut out_cols: Vec<i64> = Vec::new();
    let mut out_data: Vec<f32> = Vec::new();
    for i in 0..batch.num_rows() {
        let r = usize::try_from(rows[i]).ok();
        let c = usize::try_from(cols[i]).ok();
        let (nr, nc) = match (
            r.and_then(|r| old_to_new.get(r)),
            c.and_then(|c| old_to_new.get(c)),
        ) {
            (Some(&nr), Some(&nc)) if nr >= 0 && nc >= 0 => (nr, nc),
            _ => continue,
        };
        out_rows.push(nr);
        out_cols.push(nc);
        out_data.push(data.value(i));
    }

    // Choose the output coordinate width from the compacted dimension: keep
    // Int64 when the surviving axis still exceeds i32::MAX, otherwise narrow
    // to Int32 (the common case).
    let use_i64 = new_dim > i64::from(i32::MAX);
    let coord_type = if use_i64 {
        DataType::Int64
    } else {
        DataType::Int32
    };
    let (row_arr, col_arr): (
        std::sync::Arc<dyn arrow::array::Array>,
        std::sync::Arc<dyn arrow::array::Array>,
    ) = if use_i64 {
        (
            std::sync::Arc::new(Int64Array::from(out_rows)),
            std::sync::Arc::new(Int64Array::from(out_cols)),
        )
    } else {
        (
            std::sync::Arc::new(Int32Array::from(
                out_rows.into_iter().map(|v| v as i32).collect::<Vec<_>>(),
            )),
            std::sync::Arc::new(Int32Array::from(
                out_cols.into_iter().map(|v| v as i32).collect::<Vec<_>>(),
            )),
        )
    };

    // Rebuild the schema with the chosen coordinate width; rewrite
    // n_rows / n_cols metadata and preserve all other metadata keys.
    let mut metadata = batch.schema().metadata().clone();
    metadata.insert("n_rows".to_string(), new_dim.to_string());
    metadata.insert("n_cols".to_string(), new_dim.to_string());
    let schema = std::sync::Arc::new(arrow::datatypes::Schema::new_with_metadata(
        vec![
            Field::new("row", coord_type.clone(), false),
            Field::new("col", coord_type, false),
            Field::new("data", DataType::Float32, false),
        ],
        metadata,
    ));
    Ok(arrow::array::RecordBatch::try_new(
        schema,
        vec![
            row_arr,
            col_arr,
            std::sync::Arc::new(Float32Array::from(out_data)),
        ],
    )?)
}

/// Discover the deduplicated logical keys for a per-modality dense-mapping
/// family (obsm/varm), spanning both the legacy single-section type and the
/// sharded type. Shard entry names (`{key}_shard_{idx}`) are reduced to their
/// logical `{key}`. Keys are returned sorted for deterministic output order.
fn discover_modality_keys(
    reader: &ScxReader,
    modality_id: u8,
    prefix: &str,
    single: SectionType,
    shard: SectionType,
) -> Vec<String> {
    let mut keys = std::collections::BTreeSet::new();
    for entry in &reader.catalog().entries {
        if entry.modality_id != modality_id || !entry.name.starts_with(prefix) {
            continue;
        }
        let Some(rest) = entry.name.strip_prefix(prefix) else {
            continue;
        };
        if entry.section_type == single {
            keys.insert(rest.to_string());
        } else if entry.section_type == shard {
            let key = match rest.rfind("_shard_") {
                Some(pos) => &rest[..pos],
                None => rest,
            };
            keys.insert(key.to_string());
        }
    }
    keys.into_iter().collect()
}

/// Write the (filtered) obs metadata batch to `writer`.
///
/// When `reshape` is false this writes a single legacy `ObsMetadata`
/// section (the historical compact behaviour). When true the batch is
/// sliced into `shard_target_rows`-sized chunks and emitted as
/// `ObsMetadataShard` sections — the in-place migration path for files
/// written before sharded obs metadata existed. `write_obs_shard`
/// upcasts `Utf8 → LargeUtf8` per shard internally, so individual shards
/// never hit the Arrow IPC 2 GB narrow-offset ceiling.
fn write_obs_section(
    writer: &mut ScxWriter,
    obs: &arrow::array::RecordBatch,
    reshape: bool,
    shard_target_rows: u32,
) -> Result<()> {
    let n = obs.num_rows();
    if !reshape || n == 0 {
        // Nothing to shard (or reshape not requested); a single section
        // keeps an empty file well-formed.
        writer.write_obs(obs)?;
        return Ok(());
    }
    let chunk = (shard_target_rows.max(1)) as usize;
    let total = n as u64;
    let (mut shard_idx, mut row_start, mut cursor) = (0u32, 0u64, 0usize);
    while cursor < n {
        let take = chunk.min(n - cursor);
        let slice = obs.slice(cursor, take); // zero-copy
        writer.write_obs_shard(shard_idx, row_start, take as u64, total, &slice)?;
        shard_idx += 1;
        row_start += take as u64;
        cursor += take;
    }
    Ok(())
}

/// Phase 6: compact a multimodal SCX file. Applies the global keep
/// mask to every modality's CSR shards (and per-modality layers) while
/// preserving the ModalityTable and per-modality var / obsm / uns.
/// Per-modality CSC sidecars are dropped (rebuild via
/// `--rebuild-csc`).
fn compact_multimodal(
    reader: ScxReader,
    in_header: FileHeader,
    input_path: &Path,
    output_path: &Path,
    reshape_obs: bool,
) -> Result<()> {
    let dv = reader.read_deletion_vectors()?;

    let n_obs = in_header.n_obs as usize;
    let keep_mask = build_keep_mask(n_obs, &dv, reader.catalog());
    let obs = reader.read_obs()?;
    let filtered_obs = if let Some(ref mask) = keep_mask {
        let bool_array = arrow::array::BooleanArray::from(mask.clone());
        compute::filter_record_batch(&obs, &bool_array)?
    } else {
        obs
    };
    let new_n_obs = filtered_obs.num_rows();

    // Carry input flags except has_deletion_vectors (applied) and
    // has_csc (per-modality CSC sidecars are dropped on compact;
    // caller rebuilds via --rebuild-csc).
    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);
    let table = reader
        .modality_table()
        .ok_or_else(|| {
            scx_format::ScxError::InvalidCatalog(
                "compact_multimodal: file has no modality table".to_string(),
            )
        })?
        .clone();

    let any_input_had_csc = table.entries.iter().any(|info| info.flags.has_csc());
    if any_input_had_csc {
        log::warn!(
            "compact dropped per-modality CSC shards from {input}: \
             rerun `scx build-csc` (or pass --rebuild-csc) to restore the \
             column-major sidecar",
            input = input_path.display()
        );
    }

    let max_n_vars = table
        .entries
        .iter()
        .map(|info| info.n_vars)
        .max()
        .unwrap_or(0);

    let out_header = FileHeader {
        magic: MAGIC,
        format_version: scx_format::CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: out_flags,
        n_obs: new_n_obs as u64,
        n_vars: max_n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: 0,
        index_dtype: in_header.index_dtype,
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
    write_obs_section(
        &mut writer,
        &filtered_obs,
        reshape_obs,
        in_header.shard_target_rows,
    )?;

    let shard_target = in_header.shard_target_rows;

    // Iterate modalities in registration order so the output's
    // modality_id assignment matches the input.
    for (idx, info) in table.entries.iter().enumerate() {
        let in_modality_id = (idx + 1) as u8;
        let default_codec = CodecId::from_u8(info.default_codec_id)
            .ok_or(crate::error::OpsError::UnknownCodec(info.default_codec_id))?;
        let default_value_encoding = ValueEncoding::from_u8(info.default_value_encoding).ok_or(
            crate::error::OpsError::UnknownValueEncoding(info.default_value_encoding),
        )?;
        let out_modality_id = writer.add_modality(
            &info.name,
            info.modality_type,
            default_codec,
            default_value_encoding,
            false,
        )?;
        debug_assert_eq!(in_modality_id, out_modality_id);
        writer.set_modality_n_vars(out_modality_id, info.n_vars)?;

        // Per-modality var.
        let var = reader.read_var_for(in_modality_id)?;
        writer.write_var_for(out_modality_id, &var)?;

        // Per-modality CSR shards. Filter by global keep_mask and
        // re-shard to shard_target_rows.
        let value_encoding = {
            let entries = reader.catalog().csr_shards_for_modality(in_modality_id);
            if let Some(first) = entries.first() {
                let section = reader.section_bytes(first)?;
                let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                    &section[..scx_format::SHARD_HEADER_SIZE],
                ))?;
                ValueEncoding::from_u8(sh.value_encoding).ok_or(
                    crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
                )?
            } else {
                default_value_encoding
            }
        };

        let mut acc_indptr: Vec<u64> = vec![0];
        let mut acc_indices: Vec<u32> = Vec::new();
        let mut acc_values: Vec<u8> = Vec::new();
        let mut acc_rows = 0u64;
        let mut emitted_rows = 0u64;

        for shard_entry in reader.catalog().csr_shards_for_modality(in_modality_id) {
            let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
            let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            let shard_n_rows = indptr.len() - 1;

            for local_row in 0..shard_n_rows {
                let global_idx = shard_row_start + local_row as u64;
                let deleted = keep_mask
                    .as_ref()
                    .is_some_and(|mask| !mask[global_idx as usize]);
                if deleted {
                    continue;
                }

                let row_start_nnz = indptr[local_row] as usize;
                let row_end_nnz = indptr[local_row + 1] as usize;
                let row_nnz = row_end_nnz - row_start_nnz;
                for j in row_start_nnz..row_end_nnz {
                    acc_indices.push(indices[j] as u32);
                    encode_value(&mut acc_values, data[j], value_encoding)?;
                }
                let prev = *acc_indptr.last().unwrap();
                acc_indptr.push(prev + row_nnz as u64);
                acc_rows += 1;

                if acc_rows >= shard_target as u64 {
                    let codec = select_codec(&acc_values, value_encoding);
                    writer.write_csr_shard_for(
                        out_modality_id,
                        &acc_indptr,
                        &acc_indices,
                        &acc_values,
                        codec,
                        value_encoding,
                        emitted_rows,
                    )?;
                    emitted_rows += acc_rows;
                    acc_indptr = vec![0];
                    acc_indices.clear();
                    acc_values.clear();
                    acc_rows = 0;
                }
            }
        }

        if acc_rows > 0 {
            let codec = select_codec(&acc_values, value_encoding);
            writer.write_csr_shard_for(
                out_modality_id,
                &acc_indptr,
                &acc_indices,
                &acc_values,
                codec,
                value_encoding,
                emitted_rows,
            )?;
        }

        // Per-modality obsm — row-filter by keep_mask. Section names are
        // `obsm/{modality_name}/{key}`. Discovery spans both the legacy
        // single-section (`ObsmEmbedding`) and the sharded
        // (`ObsmEmbeddingShard`) layouts; `read_obsm_for` reassembles a
        // sharded input transparently.
        let obsm_prefix = format!("obsm/{}/", info.name);
        for key in discover_modality_keys(
            &reader,
            in_modality_id,
            &obsm_prefix,
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        ) {
            let batch = reader.read_obsm_for(in_modality_id, &key)?;
            let filtered = if let Some(ref mask) = keep_mask {
                let bool_array = arrow::array::BooleanArray::from(mask.clone());
                compute::filter_record_batch(&batch, &bool_array)?
            } else {
                batch
            };
            writer.write_obsm_for(out_modality_id, &key, &filtered)?;
        }

        // Per-modality varm — var-axis (not row-filtered). Only a sharded
        // per-modality writer exists, so emit each as one full-coverage
        // shard; `read_varm_for` reads it back transparently.
        let varm_prefix = format!("varm/{}/", info.name);
        for key in discover_modality_keys(
            &reader,
            in_modality_id,
            &varm_prefix,
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        ) {
            let batch = reader.read_varm_for(in_modality_id, &key)?;
            let n = batch.num_rows() as u64;
            writer.write_varm_shard_for(out_modality_id, &key, 0, 0, n, n, &batch)?;
        }

        // Per-modality obsp/varp are not preserved: the format has no
        // per-modality pairwise reader, so they cannot be round-tripped.
        // Warn rather than drop them silently.
        let has_pairwise = reader.catalog().entries.iter().any(|e| {
            e.modality_id == in_modality_id
                && matches!(
                    e.section_type,
                    SectionType::ObspEmbedding
                        | SectionType::ObspEmbeddingShard
                        | SectionType::ObspCsrShard
                        | SectionType::VarpEmbedding
                        | SectionType::VarpEmbeddingShard
                )
        });
        if has_pairwise {
            log::warn!(
                "compact does not preserve per-modality obsp/varp for modality \
                 `{name}` (no per-modality pairwise reader); these sections are \
                 dropped from {input}",
                name = info.name,
                input = input_path.display()
            );
        }

        // Per-modality uns.
        if let Ok(uns) = reader.read_uns_for(in_modality_id) {
            writer.write_uns_for(out_modality_id, &uns)?;
        }

        // Per-modality layers. Catalog entries follow
        // `layer/{modality_name}/{layer_name}/shard_{idx}`. Re-shard
        // with the same keep_mask using `write_layer_csr_shard_for`.
        let layer_prefix = format!("layer/{}/", info.name);
        let mut layer_names = std::collections::BTreeSet::new();
        for entry in &reader.catalog().entries {
            if entry.section_type == SectionType::LayerCsrShard
                && entry.modality_id == in_modality_id
                && entry.name.starts_with(&layer_prefix)
            {
                if let Some(remainder) = entry.name.strip_prefix(&layer_prefix) {
                    if let Some(pos) = remainder.find("/shard_") {
                        layer_names.insert(remainder[..pos].to_string());
                    }
                }
            }
        }

        for layer_name in layer_names {
            let layer_value_encoding = {
                let shards = reader
                    .catalog()
                    .layer_csr_shards_for_modality(in_modality_id, &layer_name);
                if let Some(first) = shards.first() {
                    let section = reader.section_bytes(first)?;
                    let sh = scx_format::ShardHeader::read_from(&mut std::io::Cursor::new(
                        &section[..scx_format::SHARD_HEADER_SIZE],
                    ))?;
                    ValueEncoding::from_u8(sh.value_encoding).ok_or(
                        crate::error::OpsError::UnknownValueEncoding(sh.value_encoding),
                    )?
                } else {
                    value_encoding
                }
            };

            let mut l_indptr: Vec<u64> = vec![0];
            let mut l_indices: Vec<u32> = Vec::new();
            let mut l_values: Vec<u8> = Vec::new();
            let mut l_rows = 0u64;
            let mut l_emitted = 0u64;
            let mut l_shard_idx = 0u32;

            // Stream layer shards one at a time (peak memory: one shard,
            // not the whole layer) — mirrors the X-stream pattern above.
            for shard_entry in reader
                .catalog()
                .layer_csr_shards_for_modality(in_modality_id, &layer_name)
            {
                let (indptr, indices, data) = reader.read_shard_from_entry(shard_entry)?;
                let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
                let shard_n_rows = indptr.len() - 1;
                for local_row in 0..shard_n_rows {
                    let global_idx = shard_row_start + local_row as u64;
                    let deleted = keep_mask
                        .as_ref()
                        .is_some_and(|mask| !mask[global_idx as usize]);
                    if deleted {
                        continue;
                    }
                    let row_start = indptr[local_row] as usize;
                    let row_end = indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        l_indices.push(indices[j] as u32);
                        encode_value(&mut l_values, data[j], layer_value_encoding)?;
                    }
                    let prev = *l_indptr.last().unwrap();
                    l_indptr.push(prev + (row_end - row_start) as u64);
                    l_rows += 1;

                    if l_rows >= shard_target as u64 {
                        let codec = select_codec(&l_values, layer_value_encoding);
                        writer.write_layer_csr_shard_for(
                            out_modality_id,
                            &layer_name,
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
            if l_rows > 0 {
                let codec = select_codec(&l_values, layer_value_encoding);
                writer.write_layer_csr_shard_for(
                    out_modality_id,
                    &layer_name,
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

    // Global mappings (modality_id == 0), mirroring the single-modality
    // path. `read_all_*` scans by name prefix and does NOT filter by
    // modality, so it would also return per-modality entries (keyed
    // `{modality}/{key}`) and re-emit them as spurious global sections.
    // `discover_modality_keys(&reader, 0, ...)` filters on `modality_id == 0`,
    // isolating the true globals; the bare-key readers
    // (`read_obsm`/`read_varm`/`read_varp`/`read_obsp`) use exact-name lookup
    // so they never pick up a per-modality `{prefix}/{modality}/...` section.

    // Global obsm (obs-axis → row-filter under deletions).
    for key in discover_modality_keys(
        &reader,
        0,
        "obsm/",
        SectionType::ObsmEmbedding,
        SectionType::ObsmEmbeddingShard,
    ) {
        let batch = reader.read_obsm(&key)?;
        let filtered = if let Some(ref mask) = keep_mask {
            let bool_array = arrow::array::BooleanArray::from(mask.clone());
            compute::filter_record_batch(&batch, &bool_array)?
        } else {
            batch
        };
        writer.write_obsm(&key, &filtered)?;
    }

    // Global varm (var-axis → unfiltered).
    for key in discover_modality_keys(
        &reader,
        0,
        "varm/",
        SectionType::VarmEmbedding,
        SectionType::VarmEmbeddingShard,
    ) {
        let batch = reader.read_varm(&key)?;
        writer.write_varm(&key, &batch)?;
    }

    // Global varp (var×var → unfiltered).
    for key in discover_modality_keys(
        &reader,
        0,
        "varp/",
        SectionType::VarpEmbedding,
        SectionType::VarpEmbeddingShard,
    ) {
        let batch = reader.read_varp(&key)?;
        writer.write_varp(&key, &batch)?;
    }

    // Global obsp (obs×obs COO → remap both axes under deletions).
    for key in discover_modality_keys(
        &reader,
        0,
        "obsp/",
        SectionType::ObspEmbedding,
        SectionType::ObspEmbeddingShard,
    ) {
        let batch = reader.read_obsp(&key)?;
        let out = match keep_mask {
            Some(ref mask) => filter_obsp_coo(&batch, mask)?,
            None => batch,
        };
        writer.write_obsp(&key, &out)?;
    }

    // Global uns.
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

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
        action: "compact".to_string(),
        tool: concat!("scx-ops ", env!("CARGO_PKG_VERSION")).to_string(),
        params_json: format!("{{\"reshape_obs\":{reshape_obs}}}"),
        input_checksums: vec![],
    });
    writer.write_provenance(prov_entries)?;
    writer.finish()?;
    Ok(())
}

/// Build a keep mask based on deletion vectors.
fn build_keep_mask(
    n_obs: usize,
    dv: &Option<scx_format::DeletionVectors>,
    catalog: &scx_format::FullCatalog,
) -> Option<Vec<bool>> {
    let dv = dv.as_ref()?;
    if dv.total_deleted() == 0 {
        return None;
    }

    let mut mask = vec![true; n_obs];

    // Map shard_id (sort-order index from shards_sorted()) to deletion bitmaps.
    // This is consistent with delete.rs which uses the same sort-order index.
    let shards = catalog.shards_sorted();
    for (shard_idx, shard_entry) in shards.iter().enumerate() {
        if let Some(ref stats) = shard_entry.stats {
            let shard_idx = shard_idx as u32;

            if let Some(bitmap) = dv.shards.get(&shard_idx) {
                for local_row in bitmap.iter() {
                    let global_row = stats.row_start + local_row as u64;
                    if (global_row as usize) < n_obs {
                        mask[global_row as usize] = false;
                    }
                }
            }
        }
    }

    Some(mask)
}
