//! `scx sort` standalone engine.
//!
//! Globally reorders the obs (cell) axis of an already-built, codec-compressed
//! `.scx` by a sort key and writes a new file. The pure primitives (key
//! extraction, partition sizing, predicate-index feed, provenance) live in
//! [`crate::sort`]; this module is the I/O engine + strategy selector on top.
//!
//! ## The `new_pos` reduction (design note)
//!
//! A global row reorder is a random scatter over compressed CSR shards
//! (decoding one arbitrary row means decoding its whole shard), so a
//! permute-then-seek is O(n_shards²). A classic external partition sort
//! solves this by spilling decoded rows tagged with `(full_key,
//! source_row_id)`. We use a simpler, equivalent reduction: the obs **key**
//! set fits in RAM at the same memory class the predicate-index build already
//! requires (O(n_obs)), so we sort the keys once up front (`stable_argsort`)
//! to get the global order, then tag each spilled X row with a single integer
//! **`new_pos`** (its destination row). Everything downstream keys off
//! `new_pos`:
//!
//! - correctness for *any* key kind (categorical / numeric / composite /
//!   reverse) — the key comparison happened once, in pass 0;
//! - stability is inherent (the order comes from `stable_argsort`, ties broken
//!   by source row-id);
//! - parallel scatter is trivially safe (append order is irrelevant — a row's
//!   destination is fixed by `new_pos`);
//! - partitions are `new_pos` ranges, so each holds ≤ `P` rows **regardless of
//!   key skew** — an explicit pass-2 sub-split is unnecessary (a dominant
//!   category cannot blow a partition past budget).
//!
//! The binding constraint (the multi-TB X matrix) is still bounded to one
//! partition at a time; the O(n_obs) order array is the same memory class as
//! the predicate index it must build either way.
//!
//! ## Strategies
//!
//! All three compute the *same* `order` in pass 0 and share obs / obsm / layer
//! / index emission, differing only in how they gather X rows (so a/b/c are
//! byte-identical):
//! - **(a) in-memory** — decode all X into RAM, gather by `order`.
//! - **(b) K-pass-by-category** — single categorical key, low K: stream X once
//!   per category, zero spill.
//! - **(c) external partition sort** — spill X rows tagged by `new_pos`, emit
//!   per `new_pos`-range partition.
//!
//! ## Scope
//!
//! Local input only (cloud input deferred). obsm and layers are gathered
//! in-memory (the bounded-memory guarantee is for X). The CSC sidecar is
//! dropped; the CLI re-emits it post-write via `rebuild_csc_inplace` on
//! `--rebuild-csc`.
//!
//! **Output size is not guaranteed neutral.** The reorder re-encodes every X
//! shard with `--codec` (default `auto`). `scx1`-coded shards are size-neutral
//! under a row permutation (per-row independent index coding), but `zstd`-coded
//! shards and per-shard auto-codec re-selection shift X size a few percent in
//! either direction (regrouping which cells share a shard changes cross-row
//! compressibility). Observed: sorting the 149M-cell `drug.scx` (`mixed
//! scx1/zstd`, uint32) by `cell_type` grew the X matrix ~8%. Value encoding is
//! preserved (`x_value_encoding` widens only to fit the global max), so the
//! growth is purely codec/order-dependent. See docs/sharding.md and
//! docs/performance.md § Sort.
//!
//! Phase 5 lifted the earlier scope-outs: **multimodal** inputs reorder every
//! modality's X by the global obs order ([`sort_multimodal`], mirroring
//! `compact_multimodal`; per-modality X is gathered in-memory, the bounded
//! external path stays single-modality); **obsp** (obs×obs COO) is remapped
//! through the permutation via `compact::remap_obsp_coo` (varp passes through —
//! var axis untouched); and the **detection bitmap** is rebuilt per X shard
//! when `SortOptions.bitmap` is `Auto`/`Always` (default `Off` = drop). The
//! predicate index is skipped on the multimodal path (unimodal-only
//! engine-wide, as in compact).

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format_io::codec_select::select_codec;
use scx_format_io::header::FileHeader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{
    BitmapPolicy, BitmapShard, FullCatalogEntry, ScxReader, ShardHeader, SHARD_HEADER_SIZE,
};

use crate::error::{OpsError, Result};
use crate::flock::SharedFileLock;
use crate::helpers::encode_value;
use crate::sort::{
    partition_target_rows, rebuild_obs_predicate_index_streaming, sort_provenance_entry,
    stable_argsort, SortKeyExtractor, SortOptions, SortStrategy, SortSummary,
};

/// K-pass is only considered for a single categorical key with at most this
/// many distinct values (above which the external spill path wins).
const K_PASS_MAX_CARDINALITY: usize = 32;

/// Sort an SCX file: reorder the obs axis by `opts.by`, writing a new file.
///
/// Selects the execution strategy automatically from the input shape and
/// `opts.memory_budget`. See the module docs for the design.
pub fn sort(input: &Path, output: &Path, opts: &SortOptions) -> Result<SortSummary> {
    sort_with_strategy(input, output, opts, None)
}

/// Sort with an explicit strategy override. `None` auto-selects. The override
/// exists for the strategy-differential gate (§10) — all three strategies must
/// produce byte-identical output on a shared fixture.
pub fn sort_with_strategy(
    input: &Path,
    output: &Path,
    opts: &SortOptions,
    force_strategy: Option<SortStrategy>,
) -> Result<SortSummary> {
    let _lock = SharedFileLock::acquire(input)?;
    let reader = ScxReader::open(input)?;
    let in_header = reader.header().clone();

    if in_header.has_raw() {
        log::warn!(
            "scx sort: input {} carries an adata.raw matrix, which is not preserved \
             through sort — raw will be dropped from the output",
            input.display()
        );
    }

    let n_obs = in_header.n_obs as usize;
    let n_vars = in_header.n_vars;
    let n_vars_u32 = u32::try_from(n_vars).map_err(|_| {
        OpsError::InvalidInput(format!("scx sort: n_vars {n_vars} exceeds u32::MAX"))
    })?;

    // ----- Pass 0: compute the global order (shared by all strategies) -----
    log::info!("scx sort: pass 0 begin (n_obs={n_obs})");
    let keep_mask = reader.deletion_keep_mask()?;

    // Live rows only (deletions are materialized away, as compact does).
    let live_ids: Vec<u64> = match &keep_mask {
        Some(mask) => (0..n_obs as u64).filter(|&i| mask[i as usize]).collect(),
        None => (0..n_obs as u64).collect(),
    };

    // Pass 0a — compute the order from the sort-key columns ONLY. Reading
    // just `opts.by` (projected, assembled with the same dictionary-unify
    // as a full obs read) keeps the order computation off the unbudgeted
    // full-obs materialization that OOMs on atlas-scale sharded files.
    log::info!("scx sort: pass 0a reading sort-key columns {:?}", opts.by);
    let key_batch = reader.read_obs_keys(&opts.by)?;
    log::info!(
        "scx sort: pass 0a key batch read ({} rows, {} cols, key dtype {:?})",
        key_batch.num_rows(),
        key_batch.num_columns(),
        key_batch
            .schema()
            .fields()
            .first()
            .map(|f| f.data_type().clone())
    );
    let live_keys = match &keep_mask {
        Some(mask) => {
            let bool_arr = arrow::array::BooleanArray::from(mask.clone());
            arrow::compute::filter_record_batch(&key_batch, &bool_arr)?
        }
        None => key_batch,
    };
    let n_live = live_keys.num_rows();
    if n_live == 0 {
        return Err(OpsError::InvalidInput(
            "scx sort: input has no live rows to sort".to_string(),
        ));
    }

    let extractor = SortKeyExtractor::new(&live_keys.schema(), &opts.by, opts.reverse)?;
    let rows = extractor.rows(&live_keys)?;
    log::info!("scx sort: pass 0a key rows built; argsort over {n_live} rows");
    // Local indices into the live sequence, in sorted order (stable, ties by
    // source id). `live_keys` and `live_obs` (built below) are filtered from
    // the same row universe in the same shard-concatenated order, so these
    // local indices apply to both.
    let order_local = stable_argsort(&rows, 0);
    // Output row -> original (global) old row id.
    let order_old: Vec<u64> = order_local.iter().map(|&l| live_ids[l as usize]).collect();
    drop(rows);
    // `live_keys` (the projected sort-key columns) stays resident — it feeds the
    // X strategy selector / K-pass below (category enumeration, null detection),
    // replacing the full `live_obs` the in-memory path no longer always builds.

    // old row -> new position (-1 = deleted / absent). Drives the external
    // strategy's partition routing and the obsp remap; shared by all paths.
    let mut new_pos_of_old = vec![-1i64; n_obs];
    for (new, &old) in order_old.iter().enumerate() {
        new_pos_of_old[old as usize] = new as i64;
    }

    // Pass 0b — obs write strategy (SCX-SORT-OOM-BUG Part 2). The bounded
    // spill-scatter path applies only to a single-modality sort with a
    // `--memory-budget` on a sharded-obs input whose in-memory peak would
    // exceed the budget; everything else keeps the in-memory take path (which
    // also feeds the multimodal dispatch).
    //
    // The in-memory path's *peak* is well above one steady-state copy: `read_obs`
    // itself peaks at ~2× obs while concatenating shards, then `take_rows`
    // holds `obs_full` + `sorted_obs` co-resident, plus the three O(n_obs)
    // order arrays (8 B each). Comparing one steady-state estimate to the budget
    // (as a first cut did) routed atlas obs to the in-memory path and OOM-killed
    // it — so estimate the peak conservatively.
    let obs_spill = if !reader.is_multimodal() && reader.obs_metadata_shard_count() > 0 {
        match opts.memory_budget {
            Some(b) => {
                let steady = est_obs_bytes(&reader, n_live)?;
                let order_bytes = (n_live as u64).saturating_mul(24);
                let inmem_peak = steady.saturating_mul(2).saturating_add(order_bytes);
                log::info!(
                    "scx sort: obs steady-state ~{steady} B, in-memory peak ~{inmem_peak} B, \
                     budget {b} B -> {} path",
                    if inmem_peak > b { "spill" } else { "in-memory" }
                );
                inmem_peak > b
            }
            None => false,
        }
    } else {
        false
    };

    // In-memory path materializes the full sorted obs; the spill path builds it
    // shard-by-shard during the write below.
    let sorted_obs = if obs_spill {
        None
    } else {
        let obs_full = reader.read_obs()?;
        let live_obs = match &keep_mask {
            Some(mask) => {
                let bool_arr = arrow::array::BooleanArray::from(mask.clone());
                arrow::compute::filter_record_batch(&obs_full, &bool_arr)?
            }
            None => obs_full.clone(),
        };
        Some(take_rows(&live_obs, &order_local)?)
    };

    // Multimodal inputs reorder every modality's X by the same global obs
    // order (Phase 5 / T5.1); the single-modality engine below handles the
    // common case. Multimodal always takes the in-memory obs path.
    if reader.is_multimodal() {
        let sorted_obs = sorted_obs
            .as_ref()
            .expect("multimodal sort uses the in-memory obs path");
        return sort_multimodal(
            &reader,
            &in_header,
            output,
            input,
            &order_old,
            &new_pos_of_old,
            sorted_obs,
            n_live,
            opts,
        );
    }

    // ----- Output writer + header -----
    // Drop has_deletion_vectors (bit 5) and has_csc (bit 0); the sort applies
    // the deletion vector and drops the now-stale column-major sidecar.
    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);
    if in_header.has_csc() {
        log::warn!(
            "scx sort dropped CSC shards from {}: pass --rebuild-csc to restore the \
             column-major sidecar",
            input.display()
        );
    }
    let out_header = FileHeader {
        format_version: scx_format_io::rewrite_output_format_version(
            &[in_header.format_version],
            1,
        ),
        flags: out_flags,
        n_obs: n_live as u64,
        n_vars,
        shard_target_rows: opts.shard_target_rows,
        index_dtype: in_header.index_dtype,
        ..Default::default()
    };
    let mut writer = ScxWriter::new(output, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);

    // ----- obs (sorted, re-sharded) + var -----
    // In-memory: slice the materialized sorted obs. Spill: scatter input obs
    // rows to `new_pos`-range partitions, then emit sorted output shards (the
    // SpillDir is kept alive in `obs_spill_state` for the index rebuild).
    let mut obs_spill_state: Option<ObsSpillState> = None;
    let mut obs_partitions = 0usize;
    match &sorted_obs {
        Some(sorted_obs) => write_obs_sharded(&mut writer, sorted_obs, opts.shard_target_rows)?,
        None => {
            let state = prepare_obs_spill(&reader, &new_pos_of_old, n_live, opts)?;
            obs_partitions = state.n_parts;
            log::info!(
                "scx sort: obs write pass begin ({} partitions)",
                state.n_parts
            );
            for (out_idx, item) in state.reader().enumerate() {
                let (batch, offset) = item?;
                let n = batch.num_rows() as u64;
                writer.write_obs_shard(out_idx as u32, offset, n, n_live as u64, &batch)?;
            }
            log::info!("scx sort: obs write pass done");
            obs_spill_state = Some(state);
        }
    }
    let var = reader.read_var()?;
    writer.write_var(&var)?;
    log::info!("scx sort: var written; starting X gather");

    // ----- X: strategy-specific gather -----
    let value_encoding = x_value_encoding(&reader)?;
    let total_nnz: u64 = reader
        .catalog()
        .shards_sorted()
        .iter()
        .filter_map(|e| e.stats.as_ref().map(|s| s.nnz))
        .sum();
    let density = if n_obs > 0 && n_vars > 0 {
        (total_nnz as f64 / (n_obs as f64 * n_vars as f64)).clamp(1e-9, 1.0)
    } else {
        1.0
    };
    // Estimate peak resident bytes for the in-memory strategy: X (nnz·8 +
    // indptr) plus the layers and obsm it also gathers whole — so the selector
    // does not pick `InMemory` when layers/obsm push total RSS over the budget.
    let layer_nnz: u64 = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::LayerCsrShard)
        .filter_map(|e| e.stats.as_ref().map(|s| s.nnz))
        .sum();
    let obsm_bytes: u64 = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::ObsmEmbedding | SectionType::ObsmEmbeddingShard
            )
        })
        .map(|e| e.length)
        .sum();
    let est_x_bytes = total_nnz
        .saturating_add(layer_nnz)
        .saturating_mul(8)
        .saturating_add(n_obs as u64 * 8)
        .saturating_add(obsm_bytes);

    // Leading-key cardinality + single-categorical-key flag for selector / K-pass.
    let leading_categorical = !is_numeric(
        live_keys
            .schema()
            .field_with_name(&opts.by[0])
            .map(|f| f.data_type().clone())
            .unwrap_or(DataType::Utf8),
    );
    let single_key = opts.by.len() == 1;
    // K-pass derives its emit order from the non-null category enumeration and
    // cannot place null-key rows; only the in-memory / external paths (which
    // use the full `stable_argsort` order) handle nulls. Detect nulls in the
    // leading key so the selector avoids K-pass and a forced K-pass errors.
    let leading_key_has_nulls = single_key && leading_categorical && {
        let col = live_keys.column_by_name(&opts.by[0]);
        col.map(|c| c.null_count() > 0).unwrap_or(false)
    };

    let mut spill_bytes = 0u64;
    let mut partitions = 1usize;

    // Build the category map only if K-pass is a candidate (avoids the scan
    // when an explicit non-K-pass strategy is forced or selected).
    let categories = if single_key && leading_categorical {
        Some(distinct_categories(&live_keys, &opts.by[0], opts.reverse)?)
    } else {
        None
    };
    let k = categories.as_ref().map(|c| c.len()).unwrap_or(usize::MAX);

    let strategy = force_strategy.unwrap_or_else(|| {
        select_strategy(
            opts,
            est_x_bytes,
            single_key && leading_categorical && !leading_key_has_nulls,
            k,
        )
    });

    let mut x_emitter = CsrEmitter::new(
        EmitTarget::X,
        None,
        opts.shard_target_rows,
        opts.codec,
        value_encoding,
        n_vars_u32,
        opts.bitmap,
    );
    match strategy {
        SortStrategy::InMemory => {
            emit_x_in_memory(&reader, &mut writer, &mut x_emitter, &order_old)?;
        }
        SortStrategy::KPassByCategory => {
            if leading_key_has_nulls {
                return Err(OpsError::InvalidInput(
                    "K-pass strategy cannot sort a key column containing nulls (it would drop \
                     null-key rows); use the in-memory or external strategy"
                        .to_string(),
                ));
            }
            let cats = categories.as_ref().ok_or_else(|| {
                OpsError::InvalidInput(
                    "K-pass strategy requires a single categorical sort key".to_string(),
                )
            })?;
            let cat_of_old = category_of_old(&live_keys, &live_ids, &opts.by[0], cats, n_obs)?;
            emit_x_kpass(
                &reader,
                &mut writer,
                &mut x_emitter,
                &cat_of_old,
                cats.len(),
            )?;
        }
        SortStrategy::ExternalPartition => {
            let p = partition_target_rows(
                opts.memory_budget,
                n_vars as usize,
                density,
                opts.shard_target_rows,
            );
            // Refuse if a single output shard's rows cannot fit the budget
            // (T4.7 — no silent cap).
            if let Some(budget) = opts.memory_budget {
                let per_shard =
                    (opts.shard_target_rows as f64 * n_vars as f64 * density * 16.0) as u64;
                if per_shard > budget {
                    return Err(OpsError::InvalidInput(format!(
                        "scx sort: --memory-budget {budget} too small for one output shard \
                         (~{per_shard} bytes for {} rows); raise the budget or lower --shard-size",
                        opts.shard_target_rows
                    )));
                }
            }
            let (sb, np) = emit_x_external(
                &reader,
                &mut writer,
                &mut x_emitter,
                &new_pos_of_old,
                p,
                n_live,
                opts.temp_dir.as_deref(),
            )?;
            spill_bytes = sb;
            partitions = np;
            log::info!(
                "scx sort: external partition sort spilled {spill_bytes} bytes across \
                 {partitions} partitions (~{p} rows each)"
            );
        }
        other => {
            return Err(OpsError::InvalidInput(format!(
                "scx sort engine does not run strategy {other:?} (build-time forms use \
                 convert / merge)"
            )));
        }
    }
    x_emitter.finish(&mut writer)?;
    let output_shard_row_ranges = x_emitter.ranges.clone();

    // ----- layers (in-memory gather by the same order) -----
    emit_layers_in_memory(&reader, &mut writer, &order_old, opts)?;

    // ----- obsm (in-memory take) -----
    let mut obsm: Vec<_> = reader.read_all_obsm()?.into_iter().collect();
    obsm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsm {
        let reordered = take_rows(batch, &order_old)?;
        writer.write_obsm(name, &reordered)?;
    }

    // ----- uns / varm / varp passthrough; obsp dropped (Phase 5 remap) -----
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }
    let mut varm: Vec<_> = reader.read_all_varm()?.into_iter().collect();
    varm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varm {
        writer.write_varm(name, batch)?;
    }
    let mut varp: Vec<_> = reader.read_all_varp()?.into_iter().collect();
    varp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varp {
        writer.write_varp(name, batch)?;
    }
    // obsp (obs×obs COO): remap both endpoints through the sort permutation
    // (T5.3). `new_pos_of_old` is exactly the old→new map `remap_obsp_coo`
    // expects (-1 drops edges touching a deleted obs; dims collapse to n_live).
    let mut obsp: Vec<_> = reader.read_all_obsp()?.into_iter().collect();
    obsp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsp {
        let remapped = crate::compact::remap_obsp_coo(batch, &new_pos_of_old)?;
        writer.write_obsp(name, &remapped)?;
    }

    // ----- predicate index (always (re)built; sort key auto-added) -----
    let index_result = match (&sorted_obs, &obs_spill_state) {
        (Some(sorted_obs), _) => rebuild_obs_predicate_index_streaming(
            &mut writer,
            sorted_obs.schema(),
            std::iter::once(Ok::<_, scx_engine::EngineError>((sorted_obs.clone(), 0u64))),
            &var,
            &output_shard_row_ranges,
            n_vars as usize,
            &opts.by,
            &opts.index_options,
        )?,
        // Spill path: replay the spill (bounded, ~2× pass-2 CPU, no re-scatter)
        // to feed the index the same sorted obs shards the write pass emitted.
        (None, Some(state)) => rebuild_obs_predicate_index_streaming(
            &mut writer,
            state.out_schema.clone(),
            state.reader(),
            &var,
            &output_shard_row_ranges,
            n_vars as usize,
            &opts.by,
            &opts.index_options,
        )?,
        (None, None) => unreachable!("obs path produced neither sorted_obs nor a spill state"),
    };
    let indexed_columns = index_result.obs_indexed_columns.clone();

    // ----- provenance -----
    let mut prov = reader
        .read_provenance()
        .map(|p| p.operations)
        .unwrap_or_default();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    prov.push(sort_provenance_entry(
        &opts.by,
        opts.reverse,
        opts.shard_target_rows,
        &indexed_columns,
        ts,
    ));
    writer.write_provenance(prov)?;

    writer.finish()?;

    Ok(SortSummary {
        n_obs: n_live as u64,
        n_output_shards: output_shard_row_ranges.len() as u64,
        strategy,
        spill_bytes,
        partitions,
        indexed_columns,
        obs_spilled: obs_spill_state.is_some(),
        obs_partitions,
    })
}

// ---------------------------------------------------------------------------
// Multimodal — one global obs order applied to every modality's X
// ---------------------------------------------------------------------------

/// Sort a multimodal file: reorder every modality's X (+ layers + per-modality
/// obsm/varm) by the single global obs order, mirroring `compact_multimodal`
/// with a permutation in place of the keep-mask. The obs axis is globally
/// shared (one obs table, one `n_obs`), so `order_old` / `new_pos_of_old`
/// apply identically to every modality. Per-modality X is gathered in-memory
/// (`read_all_csr_shards_for`); the bounded-memory external path stays
/// single-modality. The predicate index is skipped (unimodal-only engine-wide,
/// as in compact).
#[allow(clippy::too_many_arguments)]
fn sort_multimodal(
    reader: &ScxReader,
    in_header: &FileHeader,
    output: &Path,
    input: &Path,
    order_old: &[u64],
    new_pos_of_old: &[i64],
    sorted_obs: &RecordBatch,
    n_live: usize,
    opts: &SortOptions,
) -> Result<SortSummary> {
    let table = reader
        .modality_table()
        .ok_or_else(|| {
            OpsError::InvalidInput("scx sort: multimodal file has no modality table".to_string())
        })?
        .clone();

    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);
    if in_header.has_csc() || table.entries.iter().any(|i| i.flags.has_csc()) {
        log::warn!(
            "scx sort dropped CSC shards from {}: pass --rebuild-csc to restore the \
             column-major sidecar",
            input.display()
        );
    }
    let max_n_vars = table.entries.iter().map(|i| i.n_vars).max().unwrap_or(0);
    let out_header = FileHeader {
        format_version: scx_format_io::rewrite_output_format_version(
            &[in_header.format_version],
            2,
        ),
        flags: out_flags,
        n_obs: n_live as u64,
        n_vars: max_n_vars,
        shard_target_rows: opts.shard_target_rows,
        index_dtype: in_header.index_dtype,
        ..Default::default()
    };
    let mut writer = ScxWriter::new(output, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);

    write_obs_sharded(&mut writer, sorted_obs, opts.shard_target_rows)?;

    let mut total_x_shards = 0u64;
    for (idx, info) in table.entries.iter().enumerate() {
        let in_id = (idx + 1) as u8;
        let default_codec = CodecId::from_u8(info.default_codec_id)
            .ok_or(OpsError::UnknownCodec(info.default_codec_id))?;
        let default_ve = ValueEncoding::from_u8(info.default_value_encoding)
            .ok_or(OpsError::UnknownValueEncoding(info.default_value_encoding))?;
        let out_id = writer.add_modality(
            &info.name,
            info.modality_type,
            default_codec,
            default_ve,
            false,
        )?;
        writer.set_modality_n_vars(out_id, info.n_vars)?;
        writer.write_var_for(out_id, &reader.read_var_for(in_id)?)?;

        // X — in-memory gather, reordered by the global order.
        let ve = entry_value_encoding(
            reader,
            reader
                .catalog()
                .csr_shards_for_modality(in_id)
                .first()
                .copied(),
        )?;
        let csr = reader.read_all_csr_shards_for(in_id)?;
        let mut emitter = CsrEmitter::new(
            EmitTarget::X,
            Some(out_id),
            opts.shard_target_rows,
            opts.codec,
            ve,
            info.n_vars as u32,
            opts.bitmap,
        );
        for &old in order_old {
            let s = csr.indptr[old as usize] as usize;
            let e = csr.indptr[old as usize + 1] as usize;
            emitter.push_row(&mut writer, &csr.indices[s..e], &csr.data[s..e])?;
        }
        emitter.finish(&mut writer)?;
        total_x_shards += emitter.ranges.len() as u64;

        // Per-modality obsm (obs-axis → reorder).
        let obsm_prefix = format!("obsm/{}/", info.name);
        for key in crate::compact::discover_modality_keys(
            reader,
            in_id,
            &obsm_prefix,
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        ) {
            let batch = reader.read_obsm_for(in_id, &key)?;
            writer.write_obsm_for(out_id, &key, &take_rows(&batch, order_old)?)?;
        }

        // Per-modality varm (var-axis → unchanged, one full-coverage shard).
        let varm_prefix = format!("varm/{}/", info.name);
        for key in crate::compact::discover_modality_keys(
            reader,
            in_id,
            &varm_prefix,
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        ) {
            let batch = reader.read_varm_for(in_id, &key)?;
            let n = batch.num_rows() as u64;
            writer.write_varm_shard_for(out_id, &key, 0, 0, n, n, &batch)?;
        }

        // Per-modality uns.
        if let Ok(uns) = reader.read_uns_for(in_id) {
            writer.write_uns_for(out_id, &uns)?;
        }

        // Per-modality layers (obs-axis → reorder; in-memory gather).
        for layer_name in modality_layer_names(reader, in_id, &info.name) {
            let lve = entry_value_encoding(
                reader,
                reader
                    .catalog()
                    .layer_csr_shards_for_modality(in_id, &layer_name)
                    .first()
                    .copied(),
            )?;
            let (l_indptr, l_indices, l_data) =
                assemble_modality_layer(reader, in_id, &layer_name)?;
            let mut em = CsrEmitter::new(
                EmitTarget::Layer(layer_name.clone()),
                Some(out_id),
                opts.shard_target_rows,
                opts.codec,
                lve,
                0,
                BitmapPolicy::Off,
            );
            for &old in order_old {
                let s = l_indptr[old as usize] as usize;
                let e = l_indptr[old as usize + 1] as usize;
                em.push_row(&mut writer, &l_indices[s..e], &l_data[s..e])?;
            }
            em.finish(&mut writer)?;
        }
    }

    // ----- global sections (modality 0), via modality-0 key discovery -----
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "obsm/",
        SectionType::ObsmEmbedding,
        SectionType::ObsmEmbeddingShard,
    ) {
        let batch = reader.read_obsm(&key)?;
        writer.write_obsm(&key, &take_rows(&batch, order_old)?)?;
    }
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "varm/",
        SectionType::VarmEmbedding,
        SectionType::VarmEmbeddingShard,
    ) {
        writer.write_varm(&key, &reader.read_varm(&key)?)?;
    }
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "varp/",
        SectionType::VarpEmbedding,
        SectionType::VarpEmbeddingShard,
    ) {
        writer.write_varp(&key, &reader.read_varp(&key)?)?;
    }
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "obsp/",
        SectionType::ObspEmbedding,
        SectionType::ObspEmbeddingShard,
    ) {
        let batch = reader.read_obsp(&key)?;
        writer.write_obsp(
            &key,
            &crate::compact::remap_obsp_coo(&batch, new_pos_of_old)?,
        )?;
    }
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Provenance (predicate index skipped — multimodal indexes are unimodal-only).
    let mut prov = reader
        .read_provenance()
        .map(|p| p.operations)
        .unwrap_or_default();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    prov.push(sort_provenance_entry(
        &opts.by,
        opts.reverse,
        opts.shard_target_rows,
        &[],
        ts,
    ));
    writer.write_provenance(prov)?;
    writer.finish()?;

    Ok(SortSummary {
        n_obs: n_live as u64,
        n_output_shards: total_x_shards,
        strategy: SortStrategy::InMemory,
        spill_bytes: 0,
        partitions: 1,
        indexed_columns: Vec::new(),
        obs_spilled: false,
        obs_partitions: 0,
    })
}

/// Distinct layer names for a modality (from `layer/{modality}/{layer}/shard_*`).
fn modality_layer_names(reader: &ScxReader, modality_id: u8, modality_name: &str) -> Vec<String> {
    let prefix = format!("layer/{modality_name}/");
    let mut names = std::collections::BTreeSet::new();
    for entry in &reader.catalog().entries {
        if entry.section_type == SectionType::LayerCsrShard
            && entry.modality_id == modality_id
            && entry.name.starts_with(&prefix)
        {
            if let Some(rem) = entry.name.strip_prefix(&prefix) {
                if let Some(pos) = rem.find("/shard_") {
                    names.insert(rem[..pos].to_string());
                }
            }
        }
    }
    names.into_iter().collect()
}

/// Assemble a modality's layer into one in-memory CSR `(indptr, indices, data)`
/// by concatenating its shards in row order.
#[allow(clippy::type_complexity)]
fn assemble_modality_layer(
    reader: &ScxReader,
    modality_id: u8,
    layer_name: &str,
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let mut indptr = vec![0i64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for entry in reader
        .catalog()
        .layer_csr_shards_for_modality(modality_id, layer_name)
    {
        let (sh_indptr, sh_indices, sh_data) = reader.read_shard_from_entry(entry)?;
        let base = *indptr.last().unwrap();
        for &v in sh_indptr.iter().skip(1) {
            indptr.push(base + v);
        }
        indices.extend_from_slice(&sh_indices);
        data.extend_from_slice(&sh_data);
    }
    Ok((indptr, indices, data))
}

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

fn select_strategy(
    opts: &SortOptions,
    est_x_bytes: u64,
    single_categorical: bool,
    k: usize,
) -> SortStrategy {
    match opts.memory_budget {
        // No budget set: the caller did not ask us to bound memory, so use the
        // simplest path. Pass --memory-budget to force the bounded engine.
        None => SortStrategy::InMemory,
        Some(budget) => {
            if est_x_bytes <= budget {
                SortStrategy::InMemory
            } else if single_categorical && k > 0 && k <= K_PASS_MAX_CARDINALITY {
                // K passes over the (local) compressed input beat writing +
                // re-reading a multi-TB decoded spill at low cardinality.
                SortStrategy::KPassByCategory
            } else {
                SortStrategy::ExternalPartition
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CSR re-accumulation emitter (factored from compact.rs:247-326 / :347-431)
// ---------------------------------------------------------------------------

enum EmitTarget {
    X,
    Layer(String),
}

/// Accumulates re-ordered rows and flushes them as CSR (or layer) shards;
/// `modality_id = Some(id)` routes to the per-modality writers. For X targets
/// with a non-`Off` `bitmap` policy it also emits a detection-bitmap sidecar
/// per shard.
struct CsrEmitter {
    target: EmitTarget,
    modality_id: Option<u8>,
    shard_target: u64,
    codec: CodecSelection,
    value_encoding: ValueEncoding,
    n_vars: u32,
    bitmap: BitmapPolicy,
    acc_indptr: Vec<u64>,
    acc_indices: Vec<u32>,
    acc_values: Vec<u8>,
    acc_row_count: u64,
    emitted_rows: u64,
    shard_idx: u32,
    ranges: Vec<(u64, u64)>,
}

impl CsrEmitter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        target: EmitTarget,
        modality_id: Option<u8>,
        shard_target: u32,
        codec: CodecSelection,
        value_encoding: ValueEncoding,
        n_vars: u32,
        bitmap: BitmapPolicy,
    ) -> Self {
        Self {
            target,
            modality_id,
            shard_target: shard_target.max(1) as u64,
            codec,
            value_encoding,
            n_vars,
            bitmap,
            acc_indptr: vec![0],
            acc_indices: Vec::new(),
            acc_values: Vec::new(),
            acc_row_count: 0,
            emitted_rows: 0,
            shard_idx: 0,
            ranges: Vec::new(),
        }
    }

    fn push_row(&mut self, writer: &mut ScxWriter, indices: &[i32], data: &[f32]) -> Result<()> {
        for (k, &col) in indices.iter().enumerate() {
            self.acc_indices.push(col as u32);
            encode_value(&mut self.acc_values, data[k], self.value_encoding)?;
        }
        let prev = *self.acc_indptr.last().unwrap();
        self.acc_indptr.push(prev + indices.len() as u64);
        self.acc_row_count += 1;
        if self.acc_row_count >= self.shard_target {
            self.flush(writer)?;
        }
        Ok(())
    }

    fn flush(&mut self, writer: &mut ScxWriter) -> Result<()> {
        if self.acc_row_count == 0 {
            return Ok(());
        }
        let codec = resolve_codec(self.codec, &self.acc_values, self.value_encoding);
        let row_start = self.emitted_rows;
        let n_rows = self.acc_row_count;
        let nnz = self.acc_indices.len();
        match (&self.target, self.modality_id) {
            (EmitTarget::X, None) => writer.write_csr_shard(
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
            )?,
            (EmitTarget::X, Some(id)) => writer.write_csr_shard_for(
                id,
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
            )?,
            (EmitTarget::Layer(name), None) => writer.write_layer_csr_shard(
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
                name,
                self.shard_idx,
            )?,
            (EmitTarget::Layer(name), Some(id)) => writer.write_layer_csr_shard_for(
                id,
                name,
                self.shard_idx,
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
            )?,
        }
        // Detection bitmap (X only) — one sidecar per X shard, written in shard
        // order so its index aligns with the CSR shard (matches convert).
        if matches!(self.target, EmitTarget::X)
            && bitmap_should_build(self.bitmap, n_rows, nnz, self.n_vars)
        {
            let shard = BitmapShard::build_from_csr(
                row_start,
                n_rows as u32,
                self.n_vars,
                &self.acc_indptr,
                &self.acc_indices,
            );
            match self.modality_id {
                None => writer.write_bitmap_shard(&shard)?,
                Some(id) => writer.write_bitmap_shard_for(id, &shard)?,
            }
        }
        self.emitted_rows += n_rows;
        self.ranges.push((row_start, self.emitted_rows));
        self.acc_indptr = vec![0];
        self.acc_indices.clear();
        self.acc_values.clear();
        self.acc_row_count = 0;
        self.shard_idx += 1;
        Ok(())
    }

    fn finish(&mut self, writer: &mut ScxWriter) -> Result<()> {
        self.flush(writer)
    }
}

fn resolve_codec(sel: CodecSelection, values: &[u8], enc: ValueEncoding) -> CodecId {
    match sel {
        CodecSelection::Auto => select_codec(values, enc),
        CodecSelection::Explicit(c) => c,
    }
}

/// Whether to emit a detection bitmap for a shard under `policy`. `auto`
/// mirrors convert's density + n_vars gates (the codec-compressed-size gate is
/// convert-only — sort lacks the encoded size cheaply at flush time).
fn bitmap_should_build(policy: BitmapPolicy, n_rows: u64, nnz: usize, n_vars: u32) -> bool {
    match policy {
        BitmapPolicy::Off => false,
        BitmapPolicy::Always => true,
        BitmapPolicy::Auto => {
            if n_vars == 0 || n_vars > 1_000_000 || n_rows == 0 {
                return false;
            }
            let density = nnz as f64 / (n_rows as f64 * n_vars as f64);
            density <= 0.30
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy (a) — in-memory
// ---------------------------------------------------------------------------

fn emit_x_in_memory(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    emitter: &mut CsrEmitter,
    order_old: &[u64],
) -> Result<()> {
    let csr = reader.read_all_csr_shards()?;
    for &old in order_old {
        let s = csr.indptr[old as usize] as usize;
        let e = csr.indptr[old as usize + 1] as usize;
        emitter.push_row(writer, &csr.indices[s..e], &csr.data[s..e])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy (b) — K-pass-by-category (single categorical key, zero spill)
// ---------------------------------------------------------------------------

fn emit_x_kpass(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    emitter: &mut CsrEmitter,
    cat_of_old: &[i32],
    n_categories: usize,
) -> Result<()> {
    let shards = reader.catalog().shards_sorted();
    for cat in 0..n_categories as i32 {
        for shard in &shards {
            let (indptr, indices, data) = reader.read_shard_from_entry(shard)?;
            let row_start = shard.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            for local in 0..indptr.len() - 1 {
                let global = row_start as usize + local;
                if cat_of_old[global] == cat {
                    let s = indptr[local] as usize;
                    let e = indptr[local + 1] as usize;
                    emitter.push_row(writer, &indices[s..e], &data[s..e])?;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy (c) — external partition sort (spill, bounded memory)
// ---------------------------------------------------------------------------

/// A spill session directory removed on drop (success or error).
struct SpillDir {
    path: PathBuf,
}

impl SpillDir {
    fn create(base: Option<&Path>) -> Result<Self> {
        let base = base
            .map(|p| p.to_path_buf())
            .unwrap_or_else(std::env::temp_dir);
        std::fs::create_dir_all(&base)?;
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        // Use `create_dir` (not `create_dir_all`) so a name collision — e.g. a
        // stale dir left by a SIGKILL'd prior run that reused this pid — fails
        // rather than silently reusing leftover `part_*.bin`; retry with a
        // bumped counter until we get a fresh, exclusively-created directory.
        for attempt in 0..1024u32 {
            let path = base.join(format!(
                "scx-sort-{}-{}-{}",
                std::process::id(),
                nanos,
                attempt
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e.into()),
            }
        }
        Err(OpsError::InvalidInput(
            "scx sort: could not create a unique spill directory".to_string(),
        ))
    }

    fn partition_file(&self, p: usize) -> PathBuf {
        self.path.join(format!("part_{p}.bin"))
    }
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Pass 1 (scatter) + pass 2 (load/sort/emit). Returns `(spill_bytes,
/// n_partitions)`. Partitions are `new_pos` ranges of width `p`, so each holds
/// at most `p` rows regardless of key skew (no sub-split needed).
fn emit_x_external(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    emitter: &mut CsrEmitter,
    new_pos_of_old: &[i64],
    p: usize,
    n_live: usize,
    temp_dir: Option<&Path>,
) -> Result<(u64, usize)> {
    let mut p = p.max(1);
    let mut n_parts = n_live.div_ceil(p);
    // Pass 1 opens one spill file per partition simultaneously; cap the count so
    // an atlas-scale `n_live` with a small `p` cannot exhaust file descriptors
    // (EMFILE). Widening `p` keeps each partition ≈ one budget's worth of rows.
    const MAX_PARTITIONS: usize = 512;
    if n_parts > MAX_PARTITIONS {
        p = n_live.div_ceil(MAX_PARTITIONS);
        n_parts = n_live.div_ceil(p);
    }
    let spill = SpillDir::create(temp_dir)?;

    // ----- Pass 1: scatter decoded rows to per-partition spill files -----
    let mut part_writers: Vec<BufWriter<File>> = (0..n_parts)
        .map(|i| Ok(BufWriter::new(File::create(spill.partition_file(i))?)))
        .collect::<Result<_>>()?;
    let mut spill_bytes = 0u64;

    let shards = reader.catalog().shards_sorted();
    for shard in &shards {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard)?;
        let row_start = shard.stats.as_ref().map(|s| s.row_start).unwrap_or(0) as usize;
        for local in 0..indptr.len() - 1 {
            let global = row_start + local;
            let np = new_pos_of_old[global];
            if np < 0 {
                continue; // deleted row
            }
            let np = np as u64;
            let part = (np as usize) / p;
            let s = indptr[local] as usize;
            let e = indptr[local + 1] as usize;
            spill_bytes +=
                write_spill_row(&mut part_writers[part], np, &indices[s..e], &data[s..e])?;
        }
    }
    for w in &mut part_writers {
        w.flush()?;
    }
    drop(part_writers);

    // ----- Pass 2: per partition, load + sort by new_pos + emit -----
    // Stream-parse the spill straight into the row Vec (one buffered reader)
    // rather than reading the whole file into a separate byte buffer first —
    // that keeps peak RAM to the parsed partition, not raw bytes + parsed.
    for part in 0..n_parts {
        let mut rows = parse_spill_file(&spill.partition_file(part))?;
        rows.sort_by_key(|r| r.0);
        for (_np, indices, data) in &rows {
            emitter.push_row(writer, indices, data)?;
        }
    }

    Ok((spill_bytes, n_parts))
}

/// Write one spill record: `new_pos:u64, nnz:u32, indices:i32×nnz,
/// values:f32×nnz`. Returns the bytes written.
fn write_spill_row(
    w: &mut BufWriter<File>,
    new_pos: u64,
    indices: &[i32],
    data: &[f32],
) -> Result<u64> {
    let nnz = indices.len() as u32;
    w.write_all(&new_pos.to_le_bytes())?;
    w.write_all(&nnz.to_le_bytes())?;
    for &idx in indices {
        w.write_all(&idx.to_le_bytes())?;
    }
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(8 + 4 + nnz as u64 * 8)
}

/// Stream a partition spill file into `(new_pos, indices, values)` rows via a
/// buffered reader — peak RAM is the parsed partition plus one record's scratch
/// buffers, not a whole-file byte copy on top of the parsed rows.
#[allow(clippy::type_complexity)]
fn parse_spill_file(path: &Path) -> Result<Vec<(u64, Vec<i32>, Vec<f32>)>> {
    let mut r = BufReader::new(File::open(path)?);
    let mut out = Vec::new();
    let mut hdr = [0u8; 12];
    loop {
        // A clean EOF at a record boundary ends the stream; a partial read
        // inside a record is a truncated spill.
        match r.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let np = u64::from_le_bytes(hdr[0..8].try_into().unwrap());
        let nnz = u32::from_le_bytes(hdr[8..12].try_into().unwrap()) as usize;
        let mut ibuf = vec![0u8; nnz * 4];
        r.read_exact(&mut ibuf)?;
        let mut vbuf = vec![0u8; nnz * 4];
        r.read_exact(&mut vbuf)?;
        let indices: Vec<i32> = ibuf
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let data: Vec<f32> = vbuf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        out.push((np, indices, data));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layers (in-memory gather, all strategies)
// ---------------------------------------------------------------------------

fn emit_layers_in_memory(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    order_old: &[u64],
    opts: &SortOptions,
) -> Result<()> {
    for layer_name in reader.layer_names() {
        let ve = layer_value_encoding(reader, &layer_name)?;
        let layer = reader.read_layer(&layer_name)?;
        // Layers carry no detection bitmap (X-only, matching convert).
        let mut emitter = CsrEmitter::new(
            EmitTarget::Layer(layer_name.clone()),
            None,
            opts.shard_target_rows,
            opts.codec,
            ve,
            0,
            BitmapPolicy::Off,
        );
        for &old in order_old {
            let s = layer.indptr[old as usize] as usize;
            let e = layer.indptr[old as usize + 1] as usize;
            emitter.push_row(writer, &layer.indices[s..e], &layer.data[s..e])?;
        }
        emitter.finish(writer)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gather `batch` rows by global index list (arrow `take` per column).
fn take_rows(batch: &RecordBatch, idx: &[u64]) -> Result<RecordBatch> {
    let idx_arr = UInt64Array::from(idx.to_vec());
    let cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c, &idx_arr, None))
        .collect::<std::result::Result<_, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), cols)?)
}

/// Write a sorted obs batch as `shard_target_rows`-sized `ObsMetadataShard`
/// sections (atlas-friendly; `write_obs_shard` upcasts `Utf8 → LargeUtf8`
/// per shard so no shard hits the Arrow IPC 2 GB ceiling).
fn write_obs_sharded(writer: &mut ScxWriter, obs: &RecordBatch, shard_target: u32) -> Result<()> {
    let total = obs.num_rows() as u64;
    let st = shard_target.max(1) as usize;
    let mut shard_idx = 0u32;
    let mut start = 0usize;
    while start < obs.num_rows() {
        let len = (obs.num_rows() - start).min(st);
        let slice = obs.slice(start, len);
        writer.write_obs_shard(shard_idx, start as u64, len as u64, total, &slice)?;
        start += len;
        shard_idx += 1;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Memory-bounded obs write (spill-scatter)
//
// A global obs reorder is a random scatter, so the bounded path mirrors the X
// external strategy: pass 1 streams input obs shards and scatters their live
// rows to `new_pos`-range spill partitions; pass 2 reads each partition back,
// sorts by `new_pos`, and emits `shard_target`-sized output obs shards in
// global order. Dictionary columns are decoded to their value type before
// spilling (one uniform plain schema across all input shards — no cross-shard
// dictionary reconciliation) and re-encoded per output shard at emit time
// (categorical dtype preserved; the reader narrows the key + unifies the vocab
// on `read_obs`). Used only for single-modality sorts with a `--memory-budget`.
// ---------------------------------------------------------------------------

/// Reserved synthetic column carrying each spilled row's destination row index
/// so a partition can be sorted into global order in pass 2.
const OBS_NEW_POS_COL: &str = "__scx_sort_new_pos";

/// Estimate the decoded in-memory size of the full obs table from one shard's
/// per-row footprint × the live row count. Conservative (a small shard
/// amortizes dictionary overhead over few rows, biasing toward the spill path,
/// which is always safe).
fn est_obs_bytes(reader: &ScxReader, n_live: usize) -> Result<u64> {
    if n_live == 0 {
        return Ok(0);
    }
    let s0 = reader.read_obs_shard(0)?;
    let rows = s0.num_rows().max(1) as u64;
    let bytes: usize = s0.columns().iter().map(|c| c.get_array_memory_size()).sum();
    Ok((bytes as u64 / rows).saturating_mul(n_live as u64))
}

/// Rows per obs spill partition from the budget and per-row footprint, floored
/// at one output shard. When `budget < per_row` the integer division yields 0;
/// the `.max(floor)` floor is what guarantees a partition is always at least one
/// output shard (the caller's per-shard-fits-budget guard runs separately).
fn obs_partition_rows(budget: u64, obs_bytes_per_row: u64, shard_target: u32) -> usize {
    let floor = shard_target.max(1) as usize;
    let per_row = obs_bytes_per_row.max(1);
    ((budget / per_row) as usize).max(floor)
}

/// Widen a string/binary value type to its 64-bit-offset variant. A spilled
/// obs partition concatenates up to one budget's worth of rows (hundreds of
/// thousands), so a narrow `Utf8`/`Binary` (i32 offsets) column can exceed
/// `i32::MAX` total bytes and raise `Offset overflow error`. The full
/// `read_obs` assembler avoids this by upcasting before concat; the spill path
/// must do the same. `write_obs_shard` re-narrows per output shard on write and
/// the reader narrows again on read, so this only affects the spill/concat.
fn largen_value_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Utf8 => DataType::LargeUtf8,
        DataType::Binary => DataType::LargeBinary,
        other => other.clone(),
    }
}

/// Build the spill schema (obs with every dictionary column decoded to its
/// value type, string types widened to 64-bit offsets, plus the trailing
/// [`OBS_NEW_POS_COL`]), the output schema (obs with the original dictionary
/// columns re-encoded as `Dictionary(Int32, v)`), and the set of column names
/// that were categorical. Derived from one decoded shard. The schemas
/// intentionally carry only the column fields, not `shard_schema.metadata()`
/// (the per-shard cover stamps): the spill is an internal format that never
/// feeds the assembler, and `write_obs_shard` re-stamps the output shards.
fn obs_spill_schemas(shard_schema: &Schema) -> Result<(SchemaRef, SchemaRef, HashSet<String>)> {
    let mut plain_fields: Vec<Field> = Vec::with_capacity(shard_schema.fields().len());
    let mut out_fields: Vec<Field> = Vec::with_capacity(shard_schema.fields().len());
    let mut categorical: HashSet<String> = HashSet::new();
    for f in shard_schema.fields() {
        if f.name() == OBS_NEW_POS_COL {
            return Err(OpsError::InvalidInput(format!(
                "scx sort: obs already has a column named '{OBS_NEW_POS_COL}' (reserved)"
            )));
        }
        match f.data_type() {
            DataType::Dictionary(_, value) => {
                categorical.insert(f.name().clone());
                let value = largen_value_type(value);
                plain_fields.push(
                    Field::new(f.name(), value.clone(), f.is_nullable())
                        .with_metadata(f.metadata().clone()),
                );
                out_fields.push(
                    Field::new(
                        f.name(),
                        DataType::Dictionary(Box::new(DataType::Int32), Box::new(value)),
                        f.is_nullable(),
                    )
                    .with_metadata(f.metadata().clone()),
                );
            }
            other => {
                let dt = largen_value_type(other);
                plain_fields.push(
                    Field::new(f.name(), dt.clone(), f.is_nullable())
                        .with_metadata(f.metadata().clone()),
                );
                out_fields.push(
                    Field::new(f.name(), dt, f.is_nullable()).with_metadata(f.metadata().clone()),
                );
            }
        }
    }
    let out_schema = Arc::new(Schema::new(out_fields));
    let mut spill_fields = plain_fields;
    spill_fields.push(Field::new(OBS_NEW_POS_COL, DataType::UInt64, false));
    let spill_schema = Arc::new(Schema::new(spill_fields));
    Ok((spill_schema, out_schema, categorical))
}

/// Align one input obs shard to the plain (dictionary-decoded) column types,
/// casting dictionary columns to their value type and any wide/narrow mismatch
/// to the target. Returns the plain columns (no `new_pos`).
fn decode_obs_shard_to_plain(batch: &RecordBatch, plain_fields: &[Field]) -> Result<Vec<ArrayRef>> {
    // The spill schema is derived from shard 0 and applied positionally to every
    // shard; a malformed file whose shard has fewer columns would panic on
    // `batch.column(i)`. Reject it instead (the spill path does not run the
    // assembler's cover validation). Uniform-schema sharded obs — the normal
    // case — passes; heterogeneous dict-vs-plain reconciliation across shards is
    // a deferred follow-up (F3 in SCX-SORT-OOM-BUG).
    if batch.num_columns() < plain_fields.len() {
        return Err(OpsError::InvalidInput(format!(
            "scx sort: obs shard has {} columns, expected at least {}",
            batch.num_columns(),
            plain_fields.len()
        )));
    }
    let mut cols = Vec::with_capacity(plain_fields.len());
    for (i, f) in plain_fields.iter().enumerate() {
        let c = batch.column(i);
        if c.data_type() == f.data_type() {
            cols.push(c.clone());
        } else {
            cols.push(arrow::compute::cast(c, f.data_type())?);
        }
    }
    Ok(cols)
}

/// Pass 1 — scatter every live obs row to its `new_pos`-range partition spill
/// file (Arrow IPC stream, uniform plain schema). `p` rows per partition,
/// `n_parts` partitions.
fn scatter_obs_to_spill(
    reader: &ScxReader,
    new_pos_of_old: &[i64],
    p: usize,
    n_parts: usize,
    spill_schema: &SchemaRef,
    spill: &SpillDir,
) -> Result<()> {
    // Plain fields without the trailing new_pos column.
    let plain_fields: Vec<Field> = spill_schema.fields()[..spill_schema.fields().len() - 1]
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    // Opens `n_parts` spill files at once. The caller caps `n_parts` at 512
    // (MAX_PARTITIONS) so this stays well under a typical `ulimit -n` (1024+);
    // the X external path uses the same bound.
    let mut writers: Vec<arrow::ipc::writer::StreamWriter<BufWriter<File>>> = (0..n_parts)
        .map(|i| {
            let f = BufWriter::new(File::create(spill.partition_file(i))?);
            Ok(arrow::ipc::writer::StreamWriter::try_new(f, spill_schema)?)
        })
        .collect::<Result<_>>()?;

    let mut cursor = 0usize; // global input obs row
    for res in reader.obs_shards() {
        let batch = res?;
        let n = batch.num_rows();
        // The obs shards must cover exactly `new_pos_of_old` rows; a malformed
        // file whose shard rows sum past `n_obs` would otherwise index out of
        // bounds below. Reject rather than panic.
        if cursor + n > new_pos_of_old.len() {
            return Err(OpsError::InvalidInput(format!(
                "scx sort: obs shards cover more rows than n_obs ({} > {})",
                cursor + n,
                new_pos_of_old.len()
            )));
        }
        let plain_cols = decode_obs_shard_to_plain(&batch, &plain_fields)?;
        // Group this shard's live rows by destination partition.
        let mut part_rows: Vec<Vec<u64>> = vec![Vec::new(); n_parts];
        let mut part_pos: Vec<Vec<u64>> = vec![Vec::new(); n_parts];
        for local in 0..n {
            let np = new_pos_of_old[cursor + local];
            if np < 0 {
                continue; // deleted row
            }
            let np = np as u64;
            // `part = new_pos / p` is < n_parts by construction (n_parts =
            // n_live.div_ceil(p) and new_pos < n_live); guard anyway so a stale
            // `new_pos_of_old` can never write past the partition vectors.
            let part = (np as usize) / p;
            if part >= n_parts {
                return Err(OpsError::InvalidInput(format!(
                    "scx sort: new_pos {np} maps to partition {part} >= n_parts {n_parts}"
                )));
            }
            part_rows[part].push(local as u64);
            part_pos[part].push(np);
        }
        for part in 0..n_parts {
            if part_rows[part].is_empty() {
                continue;
            }
            let idx = UInt64Array::from(std::mem::take(&mut part_rows[part]));
            let mut sub: Vec<ArrayRef> = plain_cols
                .iter()
                .map(|c| arrow::compute::take(c, &idx, None))
                .collect::<std::result::Result<_, _>>()?;
            sub.push(Arc::new(UInt64Array::from(std::mem::take(&mut part_pos[part]))) as ArrayRef);
            let batch = RecordBatch::try_new(spill_schema.clone(), sub)?;
            writers[part].write(&batch)?;
        }
        cursor += n;
    }
    for w in writers.iter_mut() {
        w.finish()?;
    }
    Ok(())
}

/// Pass 2 — yields `shard_target`-sized output obs shards (dictionary columns
/// re-encoded) in global `new_pos` order, reading one partition spill at a
/// time. Drives both the obs-shard write and (re-constructed) the predicate
/// index rebuild. Items are `EngineError`-typed so it composes with
/// [`rebuild_obs_predicate_index_streaming`].
struct ObsScatterReader {
    partition_files: Vec<PathBuf>,
    next_part: usize,
    pending: Option<RecordBatch>, // globally-ordered plain rows not yet emitted
    spill_schema: SchemaRef,      // plain + new_pos (the on-spill schema)
    plain_schema: SchemaRef,      // plain, no new_pos (pending / concat schema)
    out_schema: SchemaRef,        // re-encoded (dictionary) output schema
    categorical: HashSet<String>,
    shard_target: usize,
    emitted: u64,
}

impl ObsScatterReader {
    fn new(
        partition_files: Vec<PathBuf>,
        spill_schema: SchemaRef,
        out_schema: SchemaRef,
        categorical: HashSet<String>,
        shard_target: u32,
    ) -> Self {
        // plain schema = spill schema minus the trailing new_pos column.
        let plain_fields: Vec<Field> = spill_schema.fields()[..spill_schema.fields().len() - 1]
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        Self {
            partition_files,
            next_part: 0,
            pending: None,
            plain_schema: Arc::new(Schema::new(plain_fields)),
            spill_schema,
            out_schema,
            categorical,
            shard_target: shard_target.max(1) as usize,
            emitted: 0,
        }
    }

    /// Read one partition spill, sort it by `new_pos`, and return its rows in
    /// plain (no `new_pos`) form. `Ok(None)` for an empty partition.
    fn load_partition(
        &self,
        part: usize,
    ) -> std::result::Result<Option<RecordBatch>, scx_engine::EngineError> {
        let f = BufReader::new(File::open(&self.partition_files[part])?);
        let rdr = arrow::ipc::reader::StreamReader::try_new(f, None)?;
        let batches: Vec<RecordBatch> = rdr.collect::<std::result::Result<_, _>>()?;
        if batches.is_empty() {
            return Ok(None);
        }
        let with_pos = arrow::compute::concat_batches(&self.spill_schema, &batches)?;
        let pos_idx = with_pos.schema().index_of(OBS_NEW_POS_COL).map_err(|_| {
            arrow::error::ArrowError::InvalidArgumentError(format!(
                "obs spill partition missing '{OBS_NEW_POS_COL}'"
            ))
        })?;
        let pos = with_pos
            .column(pos_idx)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                arrow::error::ArrowError::InvalidArgumentError(format!(
                    "obs spill column '{OBS_NEW_POS_COL}' is not UInt64"
                ))
            })?;
        // new_pos is a unique per-partition key; a plain sort suffices.
        let mut order: Vec<u64> = (0..with_pos.num_rows() as u64).collect();
        order.sort_by_key(|&i| pos.value(i as usize));
        let idx = UInt64Array::from(order);
        let cols: Vec<ArrayRef> = (0..with_pos.num_columns())
            .filter(|&i| i != pos_idx)
            .map(|i| arrow::compute::take(with_pos.column(i), &idx, None))
            .collect::<std::result::Result<_, _>>()?;
        Ok(Some(RecordBatch::try_new(self.plain_schema.clone(), cols)?))
    }

    /// Re-encode the categorical columns of a plain shard back to
    /// `Dictionary(Int32, value)`, producing the output obs shard.
    fn reencode(
        &self,
        plain: &RecordBatch,
    ) -> std::result::Result<RecordBatch, scx_engine::EngineError> {
        let mut cols = Vec::with_capacity(plain.num_columns());
        for (i, f) in plain.schema().fields().iter().enumerate() {
            let c = plain.column(i);
            if self.categorical.contains(f.name()) {
                let dt = DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(f.data_type().clone()),
                );
                cols.push(arrow::compute::cast(c, &dt)?);
            } else {
                cols.push(c.clone());
            }
        }
        Ok(RecordBatch::try_new(self.out_schema.clone(), cols)?)
    }

    fn next_shard(
        &mut self,
    ) -> std::result::Result<Option<(RecordBatch, u64)>, scx_engine::EngineError> {
        // Fill `pending` to at least one output shard (or exhaust partitions).
        while self.pending.as_ref().map(|b| b.num_rows()).unwrap_or(0) < self.shard_target
            && self.next_part < self.partition_files.len()
        {
            let part = self.next_part;
            self.next_part += 1;
            if let Some(batch) = self.load_partition(part)? {
                self.pending = Some(match self.pending.take() {
                    None => batch,
                    Some(prev) => {
                        arrow::compute::concat_batches(&self.plain_schema, &[prev, batch])?
                    }
                });
            }
        }
        let pending = match self.pending.take() {
            Some(b) if b.num_rows() > 0 => b,
            _ => return Ok(None),
        };
        let have = pending.num_rows();
        let take_n = have.min(self.shard_target);
        let shard = pending.slice(0, take_n);
        if have > take_n {
            self.pending = Some(pending.slice(take_n, have - take_n));
        }
        let out = self.reencode(&shard)?;
        let offset = self.emitted;
        self.emitted += take_n as u64;
        Ok(Some((out, offset)))
    }
}

impl Iterator for ObsScatterReader {
    type Item = std::result::Result<(RecordBatch, u64), scx_engine::EngineError>;
    fn next(&mut self) -> Option<Self::Item> {
        self.next_shard().transpose()
    }
}

/// Holds the obs spill for the duration of the sort (SpillDir RAII removes it
/// on drop) plus the parameters to construct an [`ObsScatterReader`] — once for
/// the write pass, once for the predicate-index rebuild.
struct ObsSpillState {
    _spill: SpillDir,
    partition_files: Vec<PathBuf>,
    spill_schema: SchemaRef,
    out_schema: SchemaRef,
    categorical: HashSet<String>,
    shard_target: u32,
    n_parts: usize,
}

impl ObsSpillState {
    fn reader(&self) -> ObsScatterReader {
        ObsScatterReader::new(
            self.partition_files.clone(),
            self.spill_schema.clone(),
            self.out_schema.clone(),
            self.categorical.clone(),
            self.shard_target,
        )
    }
}

/// Build the obs spill: size partitions from the budget, scatter all live obs
/// rows to per-partition spill files, and return the state needed to emit and
/// index the sorted output shards. Caller must have established `obs_spill`
/// (single-modality, sharded obs, budget set).
fn prepare_obs_spill(
    reader: &ScxReader,
    new_pos_of_old: &[i64],
    n_live: usize,
    opts: &SortOptions,
) -> Result<ObsSpillState> {
    let budget = opts
        .memory_budget
        .expect("obs spill requires a memory budget");
    let s0 = reader.read_obs_shard(0)?;
    let bytes: usize = s0.columns().iter().map(|c| c.get_array_memory_size()).sum();
    let bytes_per_row = (bytes as u64 / s0.num_rows().max(1) as u64).max(1);

    // No silent cap: one output obs shard must fit the budget (mirrors the X
    // external path's per-shard refuse guard).
    let one_shard = bytes_per_row.saturating_mul(opts.shard_target_rows.max(1) as u64);
    if one_shard > budget {
        return Err(OpsError::InvalidInput(format!(
            "scx sort: --memory-budget {budget} too small for one obs shard \
             (~{one_shard} bytes for {} rows); raise the budget or lower --shard-size",
            opts.shard_target_rows
        )));
    }

    let mut p = obs_partition_rows(budget, bytes_per_row, opts.shard_target_rows);
    let mut n_parts = n_live.div_ceil(p);
    // Cap simultaneously-open spill files (EMFILE), widening partitions to fit.
    const MAX_PARTITIONS: usize = 512;
    if n_parts > MAX_PARTITIONS {
        p = n_live.div_ceil(MAX_PARTITIONS);
        n_parts = n_live.div_ceil(p);
    }

    let (spill_schema, out_schema, categorical) = obs_spill_schemas(&s0.schema())?;
    let spill = SpillDir::create(opts.temp_dir.as_deref())?;
    scatter_obs_to_spill(reader, new_pos_of_old, p, n_parts, &spill_schema, &spill)?;
    let partition_files = (0..n_parts).map(|i| spill.partition_file(i)).collect();
    log::info!("scx sort: obs spill-scatter across {n_parts} partitions (~{p} rows each)");

    Ok(ObsSpillState {
        _spill: spill,
        partition_files,
        spill_schema,
        out_schema,
        categorical,
        shard_target: opts.shard_target_rows,
        n_parts,
    })
}

fn is_numeric(dt: DataType) -> bool {
    use DataType::*;
    matches!(
        dt,
        Int8 | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float16
            | Float32
            | Float64
    )
}

/// Distinct values of a categorical/string-like column in sorted (or reversed)
/// order. Nulls excluded (routed to the leading partition under `nulls_first`).
fn distinct_categories(obs: &RecordBatch, name: &str, reverse: bool) -> Result<Vec<String>> {
    use arrow::array::LargeStringArray;
    let col = obs.column_by_name(name).ok_or_else(|| {
        OpsError::InvalidInput(format!("sort key column '{name}' missing from obs"))
    })?;
    // Decode to `LargeUtf8` (i64 offsets): the leading key is the full obs
    // column (e.g. 149 M rows), so a narrow `Utf8` decode overflows i32 offsets
    // at >2 GB of total string bytes (`Offset overflow error`).
    let utf8 = arrow::compute::cast(col, &DataType::LargeUtf8)?;
    let arr = utf8
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| {
            OpsError::InvalidInput(format!("sort key '{name}' is not categorical/string-like"))
        })?;
    let mut set = std::collections::BTreeSet::new();
    for i in 0..arr.len() {
        if !arr.is_null(i) {
            set.insert(arr.value(i).to_string());
        }
    }
    let mut cats: Vec<String> = set.into_iter().collect();
    if reverse {
        cats.reverse();
    }
    Ok(cats)
}

/// Map global old row id -> category index (-1 = deleted / null). `cats` is the
/// sorted category order; the leading key column is read from `live_obs`.
fn category_of_old(
    live_obs: &RecordBatch,
    live_ids: &[u64],
    name: &str,
    cats: &[String],
    n_obs: usize,
) -> Result<Vec<i32>> {
    use arrow::array::LargeStringArray;
    let col = live_obs.column_by_name(name).ok_or_else(|| {
        OpsError::InvalidInput(format!("sort key column '{name}' missing from obs"))
    })?;
    // `LargeUtf8` (i64 offsets): the key spans the full obs, so a narrow `Utf8`
    // decode overflows i32 offsets at >2 GB of total string bytes.
    let utf8 = arrow::compute::cast(col, &DataType::LargeUtf8)?;
    let arr = utf8.as_any().downcast_ref::<LargeStringArray>().unwrap();
    let index: std::collections::HashMap<&str, i32> = cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as i32))
        .collect();
    let mut out = vec![-1i32; n_obs];
    for local in 0..arr.len() {
        if arr.is_null(local) {
            continue;
        }
        if let Some(&ci) = index.get(arr.value(local)) {
            out[live_ids[local] as usize] = ci;
        }
    }
    Ok(out)
}

/// Value encoding wide enough to re-encode the whole sorted X output. Reorder
/// mixes rows across shards, so a single shard's encoding can be too narrow
/// (e.g. the first shard is `Uint8` but a later row exceeds 255). We probe the
/// first shard for the float/integer *kind* and widen the integer width to the
/// max value across all shard stats — cheap (catalog stats + one header read),
/// and avoids a spurious `ValueOutOfRange` mid-sort.
fn x_value_encoding(reader: &ScxReader) -> Result<ValueEncoding> {
    let shards = reader.catalog().shards_sorted();
    widest_value_encoding(reader, &shards)
}

/// Value encoding wide enough for a layer's whole sorted output (see
/// [`x_value_encoding`]).
fn layer_value_encoding(reader: &ScxReader, layer_name: &str) -> Result<ValueEncoding> {
    let prefix = format!("{layer_name}_shard_");
    let shards: Vec<&FullCatalogEntry> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix))
        .collect();
    widest_value_encoding(reader, &shards)
}

/// Pick the narrowest encoding that covers every shard in `shards`: float wins
/// outright; otherwise the integer width that fits the max `value_max`.
fn widest_value_encoding(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
) -> Result<ValueEncoding> {
    let Some(first) = shards.first() else {
        return Ok(ValueEncoding::Uint8);
    };
    let base = entry_value_encoding(reader, Some(*first))?;
    if matches!(base, ValueEncoding::Float32 | ValueEncoding::Float16) {
        return Ok(ValueEncoding::Float32);
    }
    let max_val = shards
        .iter()
        .filter_map(|e| e.stats.as_ref().map(|s| s.value_max))
        .max()
        .unwrap_or(0);
    Ok(if max_val <= u8::MAX as u32 {
        ValueEncoding::Uint8
    } else if max_val <= u16::MAX as u32 {
        ValueEncoding::Uint16
    } else {
        ValueEncoding::Uint32
    })
}

fn entry_value_encoding(
    reader: &ScxReader,
    entry: Option<&FullCatalogEntry>,
) -> Result<ValueEncoding> {
    match entry {
        Some(e) => {
            let section = reader.section_bytes(e)?;
            if section.len() < SHARD_HEADER_SIZE {
                return Err(OpsError::InvalidInput(format!(
                    "scx sort: truncated shard header in section '{}'",
                    e.name
                )));
            }
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))
        }
        None => Ok(ValueEncoding::Uint8),
    }
}

#[cfg(test)]
#[path = "sort_engine_tests.rs"]
mod tests;
