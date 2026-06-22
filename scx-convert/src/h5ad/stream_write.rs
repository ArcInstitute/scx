// Streaming SCX → h5ad writer.
//
// Symmetrical to `stream.rs` on the read side: walks SCX CSR
// shards in row order and writes hyperslab slices into pre-allocated
// `/X/{indptr,indices,data}` (and `/layers/{name}/…`) HDF5 datasets.
// Peak RSS is bounded by one shard's worth of CSR plus encode buffers
// regardless of total file size.
//
// Layout decisions
// ---------------
// * Pre-allocate the `indptr` / `indices` / `data` triplet by walking
//   the catalog stats once (`stats.nnz` over CSR shards). This avoids
//   HDF5 extendable datasets and keeps the on-disk layout
//   deterministic.
// * Deletion vectors: when present, the pre-scan decodes each shard
//   once to count kept nnz, then the write loop decodes again. The
//   second decode is acceptable for export — deterministic on-disk
//   layout is worth one extra pass over the CSR shards.
// * Metadata writes (`obs`, `var`, `obsm`, `varm`, `obsp`, `varp`,
//   `uns`) reuse the non-streaming helpers in `write.rs`. Only
//   `/X` and `/layers/{name}` change.

use std::collections::BTreeMap;
use std::path::Path;

use hdf5::types::VarLenUnicode;
use ndarray::ArrayView1;
use scx_format_io::catalog::{FullCatalogEntry, ShardStats};
use scx_format_io::error::ScxError;
use scx_format_io::reader::ScxReader;
use scx_format_io::section::SectionType;

use super::write::{
    build_unified_export_schema, scan_column_export_layout, write_dataframe_group_at,
    write_dataframe_group_streaming, write_obsm_entry_at, write_uns_entries_at,
};
use crate::pipeline::{ConvertError, ConvertOptions};
use crate::warnings::WarningSink;

/// Write obs into `parent` under `name="obs"`. Routes to the streaming
/// path when the source has `ObsMetadataShard` sections, else falls back
/// to the eager `read_obs() + write_dataframe_group_at` path (a single
/// `ObsMetadata` section has nothing to stream).
///
/// `keep_mask_opt` is the global deletion-vector keep mask. The
/// streaming path filters per shard; the eager fallback filters the
/// assembled batch before writing so both branches honour the mask
/// symmetrically (and match the `/X` streaming path's filtered row
/// count). Pre-task-6a, this fallback ignored the mask — a latent
/// row-count mismatch with `/X` when DVs were active on a legacy file.
pub(crate) fn write_obs_streaming_or_eager(
    parent: &hdf5::Group,
    reader: &ScxReader,
    keep_mask_opt: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if reader.obs_metadata_shard_count() == 0 {
        // Legacy single-section path: nothing to stream — produce the
        // same output as before by going through the eager helper.
        match reader.read_obs() {
            Ok(obs) => {
                let filtered = match keep_mask_opt {
                    Some(mask) => filter_record_batch_by_mask(&obs, mask)?,
                    None => obs,
                };
                write_dataframe_group_at(parent, "obs", &filtered, sink)?;
            }
            // A genuinely absent obs section is the only tolerable
            // miss (some legacy files carry none). Any other failure —
            // decode error, schema mismatch, checksum, IO — must
            // surface: AnnData requires an `/obs` group, so silently
            // skipping turns a corrupt section into a malformed h5ad
            // instead of a clear conversion error.
            Err(ScxError::SectionNotFound(_)) => {}
            Err(e) => {
                return Err(ConvertError::Other(format!(
                    "failed to read obs section for h5ad export: {e}"
                )));
            }
        }
        return Ok(());
    }

    let schema = reader.read_obs_schema_logical_lossy()?;
    let n_rows_total = reader.n_obs() as usize;
    let n_rows_kept = match keep_mask_opt {
        Some(mask) => mask.iter().take(n_rows_total).filter(|&&b| b).count(),
        None => n_rows_total,
    };
    // Pre-scan (one decode pass over the metadata shards) to decide
    // plain-dataset vs nullable-group layout per int/string column AND to
    // detect columns that are a `Dictionary` in any shard. Both signals are
    // needed before any HDF5 dataset is allocated. The unified schema then
    // declares a dict-anywhere column categorical even when shard 0 was plain.
    let layout = scan_column_export_layout(reader.obs_shards(), &schema)?;
    let unified_schema = build_unified_export_schema(&schema, &layout);
    write_dataframe_group_streaming(
        parent,
        "obs",
        &unified_schema,
        reader.obs_shards(),
        n_rows_kept,
        keep_mask_opt,
        &layout.needs_nullable,
        sink,
    )
}

/// Var counterpart of `write_obs_streaming_or_eager`. Deletion vectors
/// never apply on the var axis.
pub(crate) fn write_var_streaming_or_eager(
    parent: &hdf5::Group,
    reader: &ScxReader,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if reader.var_metadata_shard_count() == 0 {
        match reader.read_var() {
            Ok(var) => {
                write_dataframe_group_at(parent, "var", &var, sink)?;
            }
            // Mirror the obs path: tolerate only an absent section;
            // propagate decode / schema / IO failures so a corrupt var
            // section fails the conversion instead of producing a
            // malformed h5ad.
            Err(ScxError::SectionNotFound(_)) => {}
            Err(e) => {
                return Err(ConvertError::Other(format!(
                    "failed to read var section for h5ad export: {e}"
                )));
            }
        }
        return Ok(());
    }

    let schema = reader.read_var_schema_logical_lossy()?;
    let n_rows_total = reader.n_vars() as usize;
    let layout = scan_column_export_layout(reader.var_shards(), &schema)?;
    let unified_schema = build_unified_export_schema(&schema, &layout);
    write_dataframe_group_streaming(
        parent,
        "var",
        &unified_schema,
        reader.var_shards(),
        n_rows_total,
        None,
        &layout.needs_nullable,
        sink,
    )
}

/// Filter a `RecordBatch` by a global keep mask. Used by the legacy
/// obs path and by obsm writers so eager / streaming branches honour
/// the deletion-vector filter symmetrically.
pub(crate) fn filter_record_batch_by_mask(
    batch: &arrow::array::RecordBatch,
    mask: &[bool],
) -> Result<arrow::array::RecordBatch, ConvertError> {
    use arrow::array::BooleanArray;
    let n = batch.num_rows();
    if mask.len() < n {
        return Err(ConvertError::Other(format!(
            "keep_mask length {} < obs batch rows {n} (catalog/header drift)",
            mask.len()
        )));
    }
    let bool_arr = BooleanArray::from(mask[..n].to_vec());
    arrow::compute::filter_record_batch(batch, &bool_arr).map_err(ConvertError::Arrow)
}

fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}

/// Streaming SCX → h5ad entry point. Mirrors `write_scx_to_h5ad`
/// (`write.rs`) but writes `/X` and `/layers/{name}` shard-
/// by-shard via pre-allocated hyperslab datasets. Single-modality
/// SCX files only; multimodal files must use `scx_to_h5mu_streaming`
/// or `scx_modality_to_h5ad_streaming`.
pub fn write_scx_to_h5ad_streaming(
    scx_path: &Path,
    h5ad_path: &Path,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    if reader.is_multimodal() {
        return Err(ConvertError::Other(format!(
            "SCX file '{}' is multimodal ({} modalities); use scx_to_h5mu_streaming \
             or scx_modality_to_h5ad_streaming",
            scx_path.display(),
            reader.n_modalities()
        )));
    }

    let file = hdf5::File::create(h5ad_path)?;
    let root = file.as_group()?;

    let keep_mask = build_keep_mask(&reader)?;

    // /X.
    let n_vars = reader.n_vars() as usize;
    stream_csr_to_group_at(
        &root,
        "X",
        &reader,
        0,
        SectionType::CsrShard,
        None,
        n_vars,
        keep_mask.as_deref(),
        opts,
        sink,
    )?;

    // obs. Stream over `ObsMetadataShard` sections when present; fall
    // back to the eager path for legacy single-section `ObsMetadata`.
    write_obs_streaming_or_eager(&root, &reader, keep_mask.as_deref(), sink)?;

    // var. Mirror of obs. The keep mask is obs-only (deletion vectors
    // do not filter var), so var streaming never carries a mask.
    write_var_streaming_or_eager(&root, &reader, sink)?;

    // obsm. Obs-axis embeddings must be filtered by the same keep
    // mask as /X and obs so anndata sees consistent row counts.
    if let Ok(obsm_map) = reader.read_all_obsm() {
        if !obsm_map.is_empty() {
            let obsm_group = root.create_group("obsm")?;
            for (name, batch) in &obsm_map {
                let filtered = match keep_mask.as_deref() {
                    Some(mask) => filter_record_batch_by_mask(batch, mask)?,
                    None => batch.clone(),
                };
                write_obsm_entry_at(&obsm_group, name, &filtered)?;
            }
        }
    }

    // varm. Var-axis features are not affected by deletion vectors
    // (DVs are obs-only), so no filtering here.
    if let Ok(varm_map) = reader.read_all_varm() {
        if !varm_map.is_empty() {
            let varm_group = root.create_group("varm")?;
            for (name, batch) in &varm_map {
                write_obsm_entry_at(&varm_group, name, batch)?;
            }
        }
    }

    // uns.
    if let Ok(uns) = reader.read_uns() {
        let uns_group = root.create_group("uns")?;
        write_uns_entries_at(&uns_group, &uns)?;
    }

    // /layers/{name}.
    stream_layers_at(&root, &reader, 0, keep_mask.as_deref(), opts, sink)?;

    // /raw (DV-filtered on the obs axis like /X). Read eagerly; shared
    // with the eager exporter.
    super::write::write_raw_to_h5ad(&root, &reader, keep_mask.as_deref(), sink)?;

    // obsp / varp pairwise matrices (COO → csr_matrix groups). Read eagerly,
    // mirroring obsm/varm above. obsp filters both axes by the obs keep mask;
    // varp (var axis) is never obs-deleted. Shared with the eager exporter.
    if let Ok(obsp) = reader.read_all_obsp() {
        super::write::write_pairwise_group(&root, "obsp", &obsp, keep_mask.as_deref())?;
    }
    if let Ok(varp) = reader.read_all_varp() {
        super::write::write_pairwise_group(&root, "varp", &varp, None)?;
    }

    Ok(())
}

/// Drain one named CSR matrix (`/X` or `/layers/{layer}`) from `reader`
/// into the HDF5 group `parent` under `name`. Pre-allocates the CSR
/// triplet using catalog stats (or a per-shard decode if deletion
/// vectors are active), then walks shards in row order writing
/// hyperslab slices.
///
/// Phase 8d dispatcher: when `opts.reader_threads` resolves to > 1
/// (and the memory-budget derate allows), routes to
/// [`stream_csr_into_prealloc_parallel`] which decodes shards in a
/// rayon worker pool and drains them in order on the calling
/// thread. The HDF5 writer side stays single-threaded; output is
/// byte-identical to the sequential path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_csr_to_group_at(
    parent: &hdf5::Group,
    name: &str,
    reader: &ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    n_vars: usize,
    keep_mask_opt: Option<&[bool]>,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let shards = collect_shards(reader, modality_id, section_type, layer_name);

    let n_obs_total = total_rows_in_shards(&shards);
    let n_obs_kept = match keep_mask_opt {
        Some(mask) => {
            // Mask length must cover the union of shard row ranges. We
            // index by global row, so any short mask is a catalog /
            // header drift — surface as a real error rather than a
            // panic (callers may be in Python).
            if mask.len() < n_obs_total as usize {
                return Err(ConvertError::Other(format!(
                    "keep_mask length {} < n_obs_total {} (catalog/header drift)",
                    mask.len(),
                    n_obs_total
                )));
            }
            count_kept_rows_in_shards(&shards, mask) as u64
        }
        None => n_obs_total,
    };
    let total_nnz = precompute_total_nnz(reader, &shards, keep_mask_opt, layer_name, modality_id)?;

    let group = parent.create_group(name)?;
    create_csr_triplet(&group, n_obs_kept as usize, n_vars, total_nnz as usize)?;

    // Re-open the three datasets just created so we can hyperslab into them.
    let indptr_ds = group.dataset("indptr")?;
    let indices_ds = group.dataset("indices")?;
    let data_ds = group.dataset("data")?;

    // indptr[0] = 0; written once before the loop.
    indptr_ds.write_slice(ArrayView1::from(&[0i64][..]), ndarray::s![0..1])?;

    let datasets = TripletDatasets {
        indptr: &indptr_ds,
        indices: &indices_ds,
        data: &data_ds,
    };

    let requested_threads = crate::pipeline::resolve_reader_threads(opts);
    let queue_depth = opts.writer_queue_depth.max(1);

    let (granted_threads, granted_depth) = if requested_threads <= 1 {
        (1, queue_depth)
    } else {
        // Memory-budget derate. Per-shard working set comes from
        // exact catalog stats — no density heuristic needed because
        // every CSR shard's `nnz` and row range are recorded at
        // write time.
        let max_shard_bytes = shards
            .iter()
            .filter_map(|e| e.stats.as_ref())
            .map(per_shard_export_bytes)
            .max()
            .unwrap_or(0);
        crate::pipeline::derate_threads_and_depth(
            opts.memory_budget,
            max_shard_bytes,
            requested_threads,
            queue_depth,
            "export shard",
            "raise --memory-budget or pass --reader-threads 1",
            sink,
        )?
    };

    if granted_threads <= 1 {
        stream_csr_into_prealloc_sequential(
            datasets,
            reader,
            &shards,
            modality_id,
            section_type,
            layer_name,
            keep_mask_opt,
        )?;
    } else {
        stream_csr_into_prealloc_parallel(
            datasets,
            reader,
            &shards,
            modality_id,
            section_type,
            layer_name,
            keep_mask_opt,
            granted_threads,
            granted_depth,
        )?;
    }

    let _ = (n_obs_kept, total_nnz); // referenced inside debug_assert paths
    Ok(())
}

/// Borrowed handles to the three CSR triplet datasets pre-allocated
/// by `create_csr_triplet`. Passed to the sequential / parallel
/// drain functions so they don't each re-open the datasets.
struct TripletDatasets<'a> {
    indptr: &'a hdf5::Dataset,
    indices: &'a hdf5::Dataset,
    data: &'a hdf5::Dataset,
}

/// Sequential drain — the original loop body, factored out so the
/// dispatcher can fall back to it when threads <= 1 or the
/// memory-budget derate forces it.
fn stream_csr_into_prealloc_sequential(
    datasets: TripletDatasets<'_>,
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    keep_mask_opt: Option<&[bool]>,
) -> Result<(), ConvertError> {
    let mut nnz_offset: u64 = 0;
    let mut row_offset_kept: u64 = 0;

    for (shard_idx, entry) in shards.iter().enumerate() {
        let stats = require_stats(entry)?;
        let shard_row_start = stats.row_start as usize;

        let (indptr_local_i64, indices_local_i32, data_local_f32) =
            read_shard_payload(reader, modality_id, section_type, layer_name, shard_idx)?;

        write_one_shard_to_prealloc(
            &datasets,
            indptr_local_i64,
            indices_local_i32,
            data_local_f32,
            keep_mask_opt,
            shard_row_start,
            &mut nnz_offset,
            &mut row_offset_kept,
        )?;
    }
    Ok(())
}

/// Phase 8d — parallel drain. Workers decode shards in a rayon pool;
/// the calling thread drains the channel in shard-index order and
/// performs the HDF5 hyperslab writes.
///
/// Bounded outstanding shards = `reader_threads + writer_queue_depth`
/// via a rolling-window spawn (mirrors the ingest coordinator in
/// `pipeline.rs::streaming_writer_coordinator_parallel`). This caps
/// peak RSS at roughly that many decoded shards in flight, regardless
/// of how slow shard 0 is relative to shard N.
#[allow(clippy::too_many_arguments)]
fn stream_csr_into_prealloc_parallel(
    datasets: TripletDatasets<'_>,
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    keep_mask_opt: Option<&[bool]>,
    reader_threads: usize,
    writer_queue_depth: usize,
) -> Result<(), ConvertError> {
    use crossbeam_channel::bounded;
    use rayon::ThreadPoolBuilder;

    if shards.is_empty() {
        return Ok(());
    }

    let pool = ThreadPoolBuilder::new()
        .num_threads(reader_threads)
        .thread_name(|i| format!("scx-export-{i}"))
        .build()
        .map_err(|e| {
            ConvertError::Other(format!(
                "failed to build rayon pool with {reader_threads} threads: {e}"
            ))
        })?;

    let (tx, rx) = bounded::<(u32, Result<DecodedShard, ConvertError>)>(writer_queue_depth.max(1));

    // Build the worker-task closure factory once; each spawned
    // closure captures the next shard's index and row_start.
    let n_shards = shards.len();
    let shard_row_starts: Vec<u64> = shards
        .iter()
        .map(|e| e.stats.as_ref().map(|s| s.row_start).unwrap_or(0))
        .collect();
    let source_label = source_label_for(modality_id, section_type, layer_name, reader);

    // Macro-style local spawn (mirrors `pipeline.rs:1325` in the
    // ingest coordinator): extracting this as a closure runs afoul
    // of rayon's invariant `'scope` lifetime — the nested closure
    // would re-borrow `reader` / `shards` / `layer_name` shorter
    // than `'scope`. Inlining the spawn body via a macro keeps the
    // borrows on the scope's lifetime directly.
    //
    // `move` is load-bearing: it moves `rx` into the closure so that
    // an early `return Err(...)` from the drain loop drops `rx` on
    // unwind, unblocking workers parked in `tx.send(...)` on the
    // bounded channel. Without `move` `rx` lives in the parent frame
    // and the scope can never join those workers.
    pool.in_place_scope(move |s| -> Result<(), ConvertError> {
        macro_rules! spawn_shard {
            ($scope:expr, $idx:expr) => {{
                let idx_ = $idx;
                let row_start = shard_row_starts[idx_];
                let source = source_label.clone();
                let entry = shards[idx_];
                let tx = tx.clone();
                $scope.spawn(move |_| {
                    let res = match read_shard_payload(
                        reader,
                        modality_id,
                        section_type,
                        layer_name,
                        idx_,
                    ) {
                        Ok((indptr, indices, data)) => {
                            let n_rows = indptr.len().saturating_sub(1) as u32;
                            Ok(DecodedShard {
                                shard_idx: idx_ as u32,
                                shard_row_start: row_start as usize,
                                n_rows,
                                indptr,
                                indices,
                                data,
                            })
                        }
                        Err(inner) => Err(wrap_shard_read_error(inner, row_start, entry, &source)),
                    };
                    let _ = tx.send((idx_ as u32, res));
                });
            }};
        }

        // Rolling-window spawn: prime the pool with at most
        // `reader_threads + writer_queue_depth` outstanding tasks,
        // then spawn one per drained shard. Without this cap a slow
        // shard 0 would let the BTreeMap accumulate all later
        // shards (matches the ingest coordinator's invariant).
        let outstanding_cap = reader_threads.saturating_add(writer_queue_depth);
        let prime = outstanding_cap.min(n_shards);
        let mut next_to_spawn: usize = 0;
        for _ in 0..prime {
            spawn_shard!(s, next_to_spawn);
            next_to_spawn += 1;
        }

        // Drain in shard-index order. The main thread's `tx`
        // keepalive stays alive across the loop — worker spawns
        // continue to clone it. The `received < n_shards` counter
        // terminates regardless of channel closure.
        let mut buffer: BTreeMap<u32, DecodedShard> = BTreeMap::new();
        let mut next_idx: u32 = 0;
        let mut nnz_offset: u64 = 0;
        let mut row_offset_kept: u64 = 0;
        let mut received: usize = 0;

        while received < n_shards {
            let (idx, res) = rx.recv().map_err(|_| {
                ConvertError::Other(
                    "parallel export worker channel closed before all shards arrived".into(),
                )
            })?;
            received += 1;
            match res {
                Err(e) => return Err(e),
                Ok(out) => {
                    buffer.insert(idx, out);
                }
            }
            while let Some(shard) = buffer.remove(&next_idx) {
                let DecodedShard {
                    shard_row_start,
                    indptr,
                    indices,
                    data,
                    ..
                } = shard;
                write_one_shard_to_prealloc(
                    &datasets,
                    indptr,
                    indices,
                    data,
                    keep_mask_opt,
                    shard_row_start,
                    &mut nnz_offset,
                    &mut row_offset_kept,
                )?;
                next_idx += 1;
                if next_to_spawn < n_shards {
                    spawn_shard!(s, next_to_spawn);
                    next_to_spawn += 1;
                }
            }
        }

        // Drop our keepalive sender; the scope joins the spawned
        // tasks before returning. Workers have already finished by
        // construction (we counted `received == n_shards`).
        drop(tx);
        Ok(())
    })?;

    Ok(())
}

/// Build a stable label used in `ConvertError::ShardRead { source }`.
/// Mirrors the naming the sequential coordinator uses for diagnostics.
fn source_label_for(
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    reader: &ScxReader,
) -> String {
    let mod_part: String = if modality_id == 0 {
        String::new()
    } else {
        reader
            .modality_info(modality_id)
            .map(|info| format!("mod/{}/", info.name))
            .unwrap_or_default()
    };
    match (section_type, layer_name) {
        (SectionType::CsrShard, _) => format!("{mod_part}X"),
        (SectionType::LayerCsrShard, Some(name)) => format!("{mod_part}layers/{name}"),
        _ => format!("{mod_part}?"),
    }
}

fn wrap_shard_read_error(
    inner: ConvertError,
    row_start: u64,
    entry: &FullCatalogEntry,
    source: &str,
) -> ConvertError {
    let n_rows = entry
        .stats
        .as_ref()
        .map(|s| (s.row_end - s.row_start) as u32)
        .unwrap_or(0);
    ConvertError::ShardRead {
        row_start,
        n_rows,
        source: source.to_string(),
        inner: Box::new(inner),
    }
}

/// Apply `filter_shard` to one decoded shard, write the three
/// hyperslab slices, and bump the running offsets. Shared helper
/// between sequential and parallel paths so the on-disk byte layout
/// stays identical regardless of route.
#[allow(clippy::too_many_arguments)]
fn write_one_shard_to_prealloc(
    datasets: &TripletDatasets<'_>,
    indptr_local: Vec<i64>,
    indices_local: Vec<i32>,
    data_local: Vec<f32>,
    keep_mask_opt: Option<&[bool]>,
    shard_row_start: usize,
    nnz_offset: &mut u64,
    row_offset_kept: &mut u64,
) -> Result<(), ConvertError> {
    let (kept_indptr_tail, kept_indices_i32, kept_data_f32) = filter_shard(
        indptr_local,
        indices_local,
        data_local,
        keep_mask_opt,
        shard_row_start,
        *nnz_offset,
    );

    if !kept_indices_i32.is_empty() {
        datasets.indices.write_slice(
            ArrayView1::from(kept_indices_i32.as_slice()),
            ndarray::s![*nnz_offset as usize..*nnz_offset as usize + kept_indices_i32.len()],
        )?;
        datasets.data.write_slice(
            ArrayView1::from(kept_data_f32.as_slice()),
            ndarray::s![*nnz_offset as usize..*nnz_offset as usize + kept_data_f32.len()],
        )?;
    }
    if !kept_indptr_tail.is_empty() {
        let lo = (*row_offset_kept + 1) as usize;
        let hi = lo + kept_indptr_tail.len();
        datasets.indptr.write_slice(
            ArrayView1::from(kept_indptr_tail.as_slice()),
            ndarray::s![lo..hi],
        )?;
    }
    *row_offset_kept += kept_indptr_tail.len() as u64;
    *nnz_offset += kept_indices_i32.len() as u64;
    Ok(())
}

/// Per-shard working-set estimate for the export memory-budget
/// derate. Uses *exact* per-shard `nnz` and row count from catalog
/// stats — no density heuristic needed because every SCX shard
/// records these (`ShardStats::write_to` writes them at convert
/// time). Counts:
/// - `nnz × 4` for `indices` (i32),
/// - `nnz × 4` for `data` (f32),
/// - `(n_rows + 1) × 8` for `indptr` (i64),
/// - `nnz × 8` for transient codec scratch (matches the ingest
///   estimate's scratch line; the codec working set is bounded by
///   the encoded shard size, which is ≤ the decoded payload).
fn per_shard_export_bytes(stats: &ShardStats) -> u64 {
    let n_rows = stats.row_end.saturating_sub(stats.row_start);
    crate::stream::shard_working_set_bytes(stats.nnz, n_rows)
}

/// Test seam — exposes the per-shard export budget estimate so the
/// budget arithmetic in `derate_export_for_budget` can be anchored
/// against accidental regressions.
#[cfg(test)]
pub(crate) fn per_shard_export_bytes_for_test(stats: &ShardStats) -> u64 {
    per_shard_export_bytes(stats)
}

/// Worker → writer payload for the parallel export coordinator.
struct DecodedShard {
    /// Reorder key (matches the spawning shard index). Read via
    /// the channel envelope; the struct copy is kept for symmetry
    /// + diagnostics on failures.
    #[allow(dead_code)]
    shard_idx: u32,
    shard_row_start: usize,
    /// Kept for diagnostics + future filter optimisations.
    #[allow(dead_code)]
    n_rows: u32,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
}

/// Iterate every layer attached to `modality_id` and stream-write
/// each one under `parent/layers/{layer_name}`.
pub(crate) fn stream_layers_at(
    parent: &hdf5::Group,
    reader: &ScxReader,
    modality_id: u8,
    keep_mask_opt: Option<&[bool]>,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let names = reader.layer_names_for(modality_id);
    if names.is_empty() {
        return Ok(());
    }
    let n_vars = match modality_id {
        0 => reader.n_vars() as usize,
        _ => reader
            .modality_info(modality_id)
            .map(|info| info.n_vars as usize)
            .unwrap_or_else(|| reader.n_vars() as usize),
    };
    let layers_group = parent.create_group("layers")?;
    for layer_name in names {
        stream_csr_to_group_at(
            &layers_group,
            &layer_name,
            reader,
            modality_id,
            SectionType::LayerCsrShard,
            Some(&layer_name),
            n_vars,
            keep_mask_opt,
            opts,
            sink,
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_shards<'a>(
    reader: &'a ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
) -> Vec<&'a FullCatalogEntry> {
    match (section_type, layer_name) {
        (SectionType::CsrShard, None) => reader.catalog().csr_shards_for_modality(modality_id),
        (SectionType::LayerCsrShard, Some(name)) if modality_id == 0 => {
            // Legacy single-modality naming: `{layer}_shard_{idx}`.
            let prefix = format!("{name}_shard_");
            let mut shards: Vec<&FullCatalogEntry> = reader
                .catalog()
                .entries
                .iter()
                .filter(|e| {
                    e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix)
                })
                .collect();
            shards.sort_by_key(|e| {
                e.stats
                    .as_ref()
                    .map_or(u64::MAX, |s| s.major_start(SectionType::LayerCsrShard))
            });
            shards
        }
        (SectionType::LayerCsrShard, Some(name)) => reader
            .catalog()
            .layer_csr_shards_for_modality(modality_id, name),
        _ => Vec::new(),
    }
}

#[allow(clippy::type_complexity)] // (indptr, indices, data) matches every read_csr_shard_* return type.
fn read_shard_payload(
    reader: &ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    shard_idx: usize,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>), ConvertError> {
    let (ip, ix, dv) = match (section_type, layer_name) {
        (SectionType::CsrShard, _) => reader.read_csr_shard_for(modality_id, shard_idx)?,
        (SectionType::LayerCsrShard, Some(name)) if modality_id == 0 => {
            reader.read_layer_csr_shard(name, shard_idx)?
        }
        (SectionType::LayerCsrShard, Some(name)) => {
            reader.read_layer_csr_shard_for(modality_id, name, shard_idx)?
        }
        _ => {
            return Err(ConvertError::Other(format!(
                "unsupported shard read: section_type={section_type:?}, layer_name={layer_name:?}"
            )));
        }
    };
    Ok((ip, ix, dv))
}

/// Indptr-only counterpart of [`read_shard_payload`]. Decodes just the
/// row-pointer array, skipping indices/data — used by precompute paths
/// that only need per-row nnz counts.
fn read_shard_indptr(
    reader: &ScxReader,
    modality_id: u8,
    section_type: SectionType,
    layer_name: Option<&str>,
    shard_idx: usize,
) -> Result<Vec<i64>, ConvertError> {
    let ip = match (section_type, layer_name) {
        (SectionType::CsrShard, _) => reader.read_csr_shard_indptr_for(modality_id, shard_idx)?,
        (SectionType::LayerCsrShard, Some(name)) if modality_id == 0 => {
            reader.read_layer_csr_shard_indptr(name, shard_idx)?
        }
        (SectionType::LayerCsrShard, Some(name)) => {
            reader.read_layer_csr_shard_indptr_for(modality_id, name, shard_idx)?
        }
        _ => {
            return Err(ConvertError::Other(format!(
                "unsupported shard read: section_type={section_type:?}, layer_name={layer_name:?}"
            )));
        }
    };
    Ok(ip)
}

/// `entry.stats` must be present on every shard in v2 catalogs (the
/// writer auto-upgrades v1 → v2 on serialise). Missing stats here
/// means the file is corrupt or truncated.
fn require_stats(entry: &FullCatalogEntry) -> Result<&ShardStats, ConvertError> {
    entry.stats.as_ref().ok_or_else(|| {
        ConvertError::Other(format!(
            "shard '{}' has no catalog stats — file may be corrupt or truncated",
            entry.name
        ))
    })
}

fn total_rows_in_shards(shards: &[&FullCatalogEntry]) -> u64 {
    shards
        .iter()
        .filter_map(|e| e.stats.as_ref().map(|s| s.row_end - s.row_start))
        .sum()
}

fn count_kept_rows_in_shards(shards: &[&FullCatalogEntry], keep_mask: &[bool]) -> usize {
    let mut kept = 0usize;
    for entry in shards {
        if let Some(stats) = entry.stats.as_ref() {
            for r in stats.row_start as usize..stats.row_end as usize {
                if r < keep_mask.len() && keep_mask[r] {
                    kept += 1;
                }
            }
        }
    }
    kept
}

fn precompute_total_nnz(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
    keep_mask_opt: Option<&[bool]>,
    layer_name: Option<&str>,
    modality_id: u8,
) -> Result<u64, ConvertError> {
    match keep_mask_opt {
        None => {
            // Fast path: sum stats.nnz across shards (zero decode).
            let mut nnz = 0u64;
            for entry in shards {
                let stats = require_stats(entry)?;
                nnz = nnz.saturating_add(stats.nnz);
            }
            Ok(nnz)
        }
        Some(mask) => {
            // DV active: decode each shard's indptr once to count
            // kept nnz. Indices/data stay encoded — they're not
            // touched until the main write loop's full-shard read.
            let section_type = if layer_name.is_some() {
                SectionType::LayerCsrShard
            } else {
                SectionType::CsrShard
            };
            let mut nnz = 0u64;
            for (shard_idx, entry) in shards.iter().enumerate() {
                let stats = require_stats(entry)?;
                let row_start = stats.row_start as usize;
                let indptr_local =
                    read_shard_indptr(reader, modality_id, section_type, layer_name, shard_idx)?;
                let n_rows = indptr_local.len().saturating_sub(1);
                for r in 0..n_rows {
                    let global = row_start + r;
                    if global < mask.len() && mask[global] {
                        let start = indptr_local[r] as usize;
                        let end = indptr_local[r + 1] as usize;
                        nnz = nnz.saturating_add((end - start) as u64);
                    }
                }
            }
            Ok(nnz)
        }
    }
}

pub(crate) fn build_keep_mask(reader: &ScxReader) -> Result<Option<Vec<bool>>, ConvertError> {
    // Shared with the reader CSR filter, `scx compact`, and the pyscx obs
    // filter: see `ScxReader::deletion_keep_mask`.
    Ok(reader.deletion_keep_mask()?)
}

fn create_csr_triplet(
    group: &hdf5::Group,
    n_obs_kept: usize,
    n_vars: usize,
    total_nnz: usize,
) -> Result<(), ConvertError> {
    group
        .new_dataset::<i64>()
        .shape([n_obs_kept + 1])
        .create("indptr")?;
    group
        .new_dataset::<i32>()
        .shape([total_nnz])
        .create("indices")?;
    group
        .new_dataset::<f32>()
        .shape([total_nnz])
        .create("data")?;

    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("csr_matrix"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    let shape = [n_obs_kept as i64, n_vars as i64];
    group
        .new_attr::<i64>()
        .shape([2])
        .create("shape")?
        .write(&shape)?;
    Ok(())
}

/// Filter a single shard by `keep_mask_opt` and rebase its indptr to
/// the cumulative absolute nnz offset (the value scipy stores at
/// `indptr[i+1]`). Returns `(kept_indptr_tail, kept_indices,
/// kept_data)`.
///
/// Fast path: `keep_mask_opt.is_none()` — the input `indices` / `data`
/// vectors are returned unchanged (no allocation, no copy). Takes the
/// three Vecs by value so the fast path can transfer ownership
/// directly.
fn filter_shard(
    indptr_local: Vec<i64>,
    indices_local: Vec<i32>,
    data_local: Vec<f32>,
    keep_mask_opt: Option<&[bool]>,
    shard_row_start: usize,
    nnz_offset: u64,
) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
    match keep_mask_opt {
        None => {
            let kept_indptr_tail: Vec<i64> = indptr_local
                .iter()
                .skip(1)
                .map(|v| *v + nnz_offset as i64)
                .collect();
            (kept_indptr_tail, indices_local, data_local)
        }
        Some(mask) => {
            let n_rows = indptr_local.len().saturating_sub(1);
            let mut kept_indptr_tail: Vec<i64> = Vec::new();
            let mut kept_indices: Vec<i32> = Vec::new();
            let mut kept_data: Vec<f32> = Vec::new();
            let mut running_nnz: i64 = nnz_offset as i64;
            for r in 0..n_rows {
                let global = shard_row_start + r;
                if global >= mask.len() || !mask[global] {
                    continue;
                }
                let start = indptr_local[r] as usize;
                let end = indptr_local[r + 1] as usize;
                kept_indices.extend_from_slice(&indices_local[start..end]);
                kept_data.extend_from_slice(&data_local[start..end]);
                running_nnz += (end - start) as i64;
                kept_indptr_tail.push(running_nnz);
            }
            (kept_indptr_tail, kept_indices, kept_data)
        }
    }
}
