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
//! Lifted the earlier scope-outs: **multimodal** inputs reorder every
//! modality's X by the global obs order ([`sort_multimodal`], mirroring
//! `compact_multimodal`; per-modality X is gathered in-memory, the bounded
//! external path stays single-modality); **obsp** (obs×obs COO) is remapped
//! through the permutation via `compact::remap_obsp_coo` (varp passes through —
//! var axis untouched); and the **detection bitmap** is rebuilt per X shard
//! when `SortOptions.bitmap` is `Auto`/`Always` (default `Off` = drop). The
//! predicate index is skipped on the multimodal path (unimodal-only
//! engine-wide, as in compact).

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use rayon::prelude::*;
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::ResolvedCodec;

use crate::codec_intent::{framing_for_rewrite, seed_codec};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{
    encode_one_shard, BitmapPolicy, BitmapShard, FullCatalogEntry, PreEncodedSection, ScxReader,
    ShardHeader, SHARD_HEADER_SIZE,
};

use crate::encode_budget::ENCODE_PHASE_MULTIPLE;
use crate::error::{OpsError, Result};
use crate::flock::SharedFileLock;
use crate::helpers::encode_values;
use crate::sort::{
    cap_spill_partitions, partition_target_rows, rebuild_obs_predicate_index_streaming,
    sort_provenance_entry, stable_argsort, SortKeyExtractor, SortOptions, SortStrategy,
    SortSummary,
};

/// K-pass is only considered for a single categorical key with at most this
/// many distinct values (above which the external spill path wins).
const K_PASS_MAX_CARDINALITY: usize = 32;

/// F1: packing-size estimate per non-zero for the byte-budget group planner —
/// in-memory `i32` index (4 B) + `f32` value (4 B). This sizes packing
/// decisions, NOT bytes-on-disk (the codec + narrowest-int width determine the
/// real shard size). Public so `scx convert --group-target-bytes`
/// sizes byte-mode shards identically to `scx sort`.
pub const GROUP_BYTES_PER_NNZ: u64 = 8;

/// Bytes one nonzero adds to a grouped block's **cut accumulator** — a `u32`
/// index element plus the value bytes `encode_value` pushes at the file-wide
/// encoding.
///
/// This is the unit `--group-write-block-bytes` / `block_byte_cap` is
/// denominated in, and it is deliberately *not*
/// [`GROUP_BYTES_PER_NNZ`]: the accumulator tracks the narrowed bytes while
/// the gather buffers hold `i32 + f32`. Both the cap's budget clamp and
/// `emit_x_in_memory_grouped_fast`'s cut loop call this, so the cap and the
/// cut points cannot be denominated differently — which is what let a narrow
/// encoding admit 1.6x the nonzeros a budget could hold.
fn block_cut_bytes_per_nnz(value_encoding: ValueEncoding) -> u64 {
    let value_width: u64 = match value_encoding {
        ValueEncoding::Uint8 => 1,
        ValueEncoding::Uint16 | ValueEncoding::Float16 => 2,
        ValueEncoding::Uint32 | ValueEncoding::Float32 => 4,
    };
    4 + value_width // 4 = u32 index element
}

/// Effective per-block byte cap for the grouped-write sub-flush: the caller's
/// `--group-write-block-bytes` (or the 256 MB default), clamped so that one
/// block's whole live phase fits `--memory-budget`.
///
/// Without the clamp a small budget paired with the larger default cap would
/// let a block silently exceed the budget, and the M1 guard's
/// downgrade-to-warning would be misleading. `0` keeps the sub-flush disabled
/// (the guard then hard-errors) and is passed through untouched.
///
/// The clamp converts the budget **into the cap's own unit**. The cap counts
/// [`block_cut_bytes_per_nnz`] — the emitter's narrowed accumulator — while a
/// block in flight costs `ENCODE_PHASE_MULTIPLE * GROUP_BYTES_PER_NNZ`
/// per nonzero, so the nonzeros a cap admits (and therefore the memory it
/// commits) depend on the value width. At `Float32` the two agree and this is
/// `budget / ENCODE_PHASE_MULTIPLE`, exactly the prior arithmetic. At
/// `Uint8` they do not: 5 accumulator bytes per nnz against 48 in memory, so
/// the old `budget / 6` admitted 1.6x the nonzeros the budget could hold and
/// [`grouped_fast_concurrency`]'s `concurrency x per_block <= budget` was false
/// by that factor with no concurrency left to cut. Clamping to the raw budget
/// was worse still — at f32 with a 256 MiB budget the cap was 256 MiB and
/// `per_block` 1.5 GiB.
///
/// Same shape as the dense slab cap in `scx-convert`: a cost model and the cap
/// that feeds it must be derived from the same figure, or the cap hands the
/// model rows it cannot afford. Extracted from its single call site so
/// `grouped_block_byte_cap_keeps_one_block_inside_the_budget` can assert that
/// invariant across every value encoding, which is not reachable from a caller
/// that needs a real file.
fn grouped_block_byte_cap(
    cap: u64,
    memory_budget: Option<u64>,
    value_encoding: ValueEncoding,
) -> u64 {
    match (cap, memory_budget) {
        (0, _) => 0,
        (cap, Some(budget)) => {
            let phase_per_nnz = ENCODE_PHASE_MULTIPLE * GROUP_BYTES_PER_NNZ;
            let admissible =
                budget.saturating_mul(block_cut_bytes_per_nnz(value_encoding)) / phase_per_nnz;
            cap.min(admissible).max(1)
        }
        (cap, None) => cap,
    }
}

/// F1: default oversize threshold as a multiple of the per-shard target when
/// `--group-max-bytes` is not supplied. Public for the same reason as
/// [`GROUP_BYTES_PER_NNZ`].
pub const GROUP_MAX_BYTES_MULTIPLE: u64 = 4;

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

    // ----- 1D: shuffle-mode option validation -----
    // Shuffle replaces the *order source*, so every other order source is a
    // contradiction rather than a modifier. Reject instead of silently
    // preferring one: a user who wrote both has a wrong mental model, and
    // ignoring half of it would produce a plausible file with the wrong layout.
    if let Some(seed) = opts.shuffle {
        if !opts.by.is_empty() {
            return Err(OpsError::InvalidInput(
                "scx sort: --shuffle and --by are mutually exclusive order sources; \
                 --shuffle permutes rows randomly and has no sort key"
                    .to_string(),
            ));
        }
        if opts.group_by.is_some() {
            return Err(OpsError::InvalidInput(
                "scx sort: --shuffle and --group-by are mutually exclusive; grouping \
                 clusters rows by label and shuffling scatters them"
                    .to_string(),
            ));
        }
        if opts.reverse {
            return Err(OpsError::InvalidInput(
                "scx sort: --reverse is meaningless with --shuffle (there is no key to \
                 reverse); drop it, or change --seed for a different permutation"
                    .to_string(),
            ));
        }
        log::info!("scx sort: shuffle mode, seed {seed}");

        // A global shuffle is the exact inverse of what a sort does to a
        // predicate index: sorting collapses each category's shard_ranges to
        // one contiguous run, shuffling scatters every category across every
        // shard. Still allowed — a training file can legitimately want both —
        // but the user should not discover it after the rewrite.
        //
        // The input's *existing* index matters as much as the request: the
        // rebuild also picks up auto-detected columns, so a plain
        // `scx sort --shuffle in.scx out.scx` on an indexed input still emits
        // an index. Gating on the request alone left the warning silent in
        // exactly the common case (observed on `pbmc10k_auto.scx`, which
        // re-indexed `n_counts` with no index flags passed).
        let input_has_obs_index = reader
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::ObsPredicateIndex);
        if input_has_obs_index
            || !opts.index_options.index_obs.is_empty()
            || opts.index_options.index_preset.is_some()
        {
            log::warn!(
                "scx sort --shuffle: building an obs predicate index on a shuffled file — a \
                 random permutation maximally scatters each value's shard ranges, so the \
                 index will be larger and far less selective than on a sorted (or even an \
                 unsorted) file. This is expected, not a bug."
            );
        }

        // Warn *before* the rewrite, not after: on an atlas-scale file the user
        // would otherwise learn about the size growth hours later. Only when the
        // output codec is `auto` — an explicit `--codec` is the user having
        // already made this call.
        //
        // NARROWED: this warning used to lead with "`auto` RE-SELECTS per shard
        // and can flip to a bulkier codec — measured 1.86-2.09x, and it is the
        // dominant term", and told the user to pin the input's own codec. That
        // WAS the derived-file codec bug (`FramingConfig::default()` meant
        // `fast`, so `auto` never adopted ShufDeltaZstd on the rewrite). Now that
        // `auto` is genuinely adaptive here, the flip is gone and pinning the
        // input codec is the wrong advice. What survives is the real, much
        // smaller effect: a permutation costs cross-row redundancy for codecs
        // whose compression spans rows — measured 6-12% for zstd, under 1% for
        // lz4/shufdelta — so only warn when the dominant input codec is one of
        // those, and do not recommend pinning anything.
        if opts.codec.decode_target.is_some() && opts.codec.explicit_codec.is_none() {
            let (cross_row, total, dominant) = cross_row_coded_shard_counts(&reader)?;
            let dominant_is_cross_row_sensitive =
                matches!(dominant, Some(CodecId::Zstd) | Some(CodecId::Pcodec));
            if total > 0 && cross_row * 2 > total && dominant_is_cross_row_sensitive {
                // This reaches Python too — pyscx installs `pyo3_log`, so a
                // `pyscx.shuffle` caller sees this exact string.
                let name = dominant.map(codec_cli_name).unwrap_or("zstd");
                log::warn!(
                    "scx sort --shuffle: {cross_row}/{total} X shards are `{name}`-coded, whose \
                     compression spans rows, so a random permutation genuinely costs some \
                     cross-row redundancy — measured 6-12% for zstd, under 1% for \
                     lz4/shufdelta. This is inherent to shuffling, not a codec regression: \
                     the output codec is `auto`, which re-runs the same adaptive selection \
                     `scx convert` uses. See docs/sharding.md."
                );
            }
        }

        // The output's shard geometry is what a training loader's batch
        // composition is quantised by, so a shuffle that silently re-shards is
        // changing the very thing the user ran it to control. `--shard-size`
        // defaults to DEFAULT_SHARD_TARGET_ROWS (inherited from `sort`), which
        // on a file written with a different shard size is a second, unasked-for
        // change. This is the same trap that fabricated a 5.97x throughput
        // ratio in 1D's own benchmark before the arm threaded the input's
        // geometry through.
        if in_header.shard_target_rows != 0 && opts.shard_target_rows != in_header.shard_target_rows
        {
            log::warn!(
                "scx sort --shuffle: output shard size {} differs from the input's {} — the \
                 rewrite will re-shard as well as reorder, which changes how many cells share \
                 a shard and therefore what a `shard_group_size=1` batch contains. Pass \
                 `--shard-size {}` (Python: `shard_size={}`) to reorder only.",
                opts.shard_target_rows,
                in_header.shard_target_rows,
                in_header.shard_target_rows,
                in_header.shard_target_rows
            );
        }
    }

    // ----- F1: grouped-sharding option validation + normalization -----
    if opts.reference.is_some() && opts.group_by.is_none() {
        return Err(OpsError::InvalidInput(
            "scx sort: --reference requires --group-by".to_string(),
        ));
    }
    if opts.group_by.is_some() && reader.is_multimodal() {
        return Err(OpsError::InvalidInput(
            "scx sort: --group-by is not supported on multimodal inputs (v1); \
             sort a single modality or drop --group-by"
                .to_string(),
        ));
    }
    // Force `group_by` to be the leading sort key and force ascending order
    // (reference rows must sort first). Shadow `opts` with the normalized clone
    // so every downstream pass (order, predicate index, write) sees the same
    // `by`.
    let normalized_opts;
    let opts = if let Some(group_col) = opts.group_by.clone() {
        let mut o = opts.clone();
        if o.by.first() != Some(&group_col) {
            o.by.retain(|c| c != &group_col);
            o.by.insert(0, group_col);
        }
        if o.reverse {
            log::warn!(
                "scx sort: --reverse is ignored with --group-by (reference rows must sort first)"
            );
            o.reverse = false;
        }
        normalized_opts = o;
        &normalized_opts
    } else {
        opts
    };

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
    // F1: also fetch the reference column when it is a separate obs column, so
    // the synthetic reference-first key can be built from the same filtered
    // batch.
    //
    // 1D: shuffle mode has no key, so it skips this read entirely — a random
    // permutation is a function of `(seed, n_live)` alone. `live_keys` is then
    // an empty batch that only the key-dependent selector inputs below consult,
    // and each of those is guarded. Skipping is not merely an optimisation:
    // `read_obs_keys(&[])` yields a zero-*row* batch, which would trip the
    // "no live rows" guard on every shuffle.
    let live_keys = if opts.shuffle.is_some() {
        RecordBatch::new_empty(Arc::new(Schema::empty()))
    } else {
        let mut read_cols = opts.by.clone();
        if let Some(crate::sort::ReferenceSpec::Column(col)) = &opts.reference {
            if !read_cols.iter().any(|c| c == col) {
                read_cols.push(col.clone());
            }
        }
        log::info!("scx sort: pass 0a reading sort-key columns {:?}", read_cols);
        let key_batch = reader.read_obs_keys(&read_cols)?;
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
        match &keep_mask {
            Some(mask) => {
                let bool_arr = arrow::array::BooleanArray::from(mask.clone());
                arrow::compute::filter_record_batch(&key_batch, &bool_arr)?
            }
            None => key_batch,
        }
    };
    // In shuffle mode the key batch is empty, so the live row count comes from
    // the deletion-filtered id list instead. Both sources count the same rows.
    let n_live = if opts.shuffle.is_some() {
        live_ids.len()
    } else {
        live_keys.num_rows()
    };
    if n_live == 0 {
        return Err(OpsError::InvalidInput(
            "scx sort: input has no live rows to sort".to_string(),
        ));
    }

    // Total nnz + density (from catalog stats) drive both the in-memory/external
    // strategy sizing below and the grouped-shard memory-budget guard (M1), so
    // compute them before the grouped plan is built.
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

    // ----- F1: grouped order + plan (or plain global order) -----
    // Grouped sorts delegate the reference-first / group-by order + label /
    // reference computation to the shared `compute_grouped_order` (also used by
    // `scx convert --group-by`), so both paths produce a byte-identical grouped
    // layout. `live_keys` (the projected sort-key columns) stays resident and
    // un-augmented — it feeds the X strategy selector / K-pass below (category
    // enumeration, null detection) and `compute_grouped_order` injects the
    // synthetic reference-first key on its own clone.
    // F6: effective per-shard block cap for the grouped-write sub-flush; see
    // `grouped_block_byte_cap`. The file-wide value encoding is read first
    // because the cap is denominated in the emitter's narrowed accumulation
    // unit, and converting a budget into that unit depends on the width.
    let value_encoding = x_value_encoding(&reader)?;
    let block_byte_cap: u64 = grouped_block_byte_cap(
        opts.group_write_block_bytes
            .unwrap_or(DEFAULT_GROUP_WRITE_BLOCK_BYTES),
        opts.memory_budget,
        value_encoding,
    );
    let mut grouped_reference_labels: Vec<String> = Vec::new();
    let (order_local, group_plan): (Vec<u64>, Option<crate::group_plan::GroupPlan>) =
        if let Some(group_col) = &opts.group_by {
            // `opts.by` is normalized to `[group_col, <secondary...>]`.
            let secondary: Vec<String> = opts.by.iter().skip(1).cloned().collect();
            let go =
                compute_grouped_order(&live_keys, group_col, &secondary, opts.reference.as_ref())?;
            log::info!("scx sort: pass 0a grouped order built; argsort over {n_live} rows");
            // Byte-budget pre-scan (skipped in row-count mode); needs emission
            // order in global old-row ids for the shard indptr lookup.
            let (per_row_nnz, target_units, bytes_per_nnz) = match opts.group_target_bytes {
                Some(tb) => {
                    let order_old_tmp: Vec<u64> =
                        go.perm.iter().map(|&l| live_ids[l as usize]).collect();
                    (
                        prescan_per_row_nnz(&reader, &order_old_tmp, n_obs)?,
                        tb.max(1),
                        GROUP_BYTES_PER_NNZ,
                    )
                }
                None => (Vec::new(), opts.shard_target_rows.max(1) as u64, 0u64),
            };
            let max_units = opts
                .group_max_bytes
                .unwrap_or_else(|| target_units.saturating_mul(GROUP_MAX_BYTES_MULTIPLE));
            let plan = crate::group_plan::plan_group_shards(
                &go.group_of_new,
                &go.ref_of_new,
                &go.labels,
                &per_row_nnz,
                target_units,
                bytes_per_nnz,
                max_units,
            );
            log::info!(
                "scx sort: group plan -> {} shards, {} records, reference_shard={:?}",
                plan.n_shards,
                plan.records.len(),
                plan.reference_shard
            );
            // M1: bound the largest grouped shard against `--memory-budget`.
            // F6 Phase 0 relaxed the never-split contract: with the block-level
            // sub-flush enabled (the default), an oversized group is split across
            // shards and bounded at one block, so exceeding the budget is a
            // warning, not a hard error. Only when the sub-flush is explicitly
            // disabled (`group_write_block_bytes == Some(0)`) does the emitter
            // still buffer the whole group — then refuse loudly.
            if let Some(budget) = opts.memory_budget {
                // Use exact per-row nnz for the footprint: reuse the byte-mode
                // prescan if present, else prescan now (row-count mode). A
                // file-wide average-density estimate could pass a group that is
                // much denser than average and still OOM (codex P2), so when a
                // budget is set we always size the guard from real nnz.
                let guard_prescan: Vec<u64>;
                let guard_nnz: &[u64] = if !per_row_nnz.is_empty() {
                    &per_row_nnz
                } else {
                    let order_old_tmp: Vec<u64> =
                        go.perm.iter().map(|&l| live_ids[l as usize]).collect();
                    guard_prescan = prescan_per_row_nnz(&reader, &order_old_tmp, n_obs)?;
                    &guard_prescan
                };
                let (max_bytes, shard_idx, label) =
                    max_grouped_shard_footprint(&plan, guard_nnz, n_vars as usize, density);
                if max_bytes > budget {
                    if block_byte_cap > 0 {
                        // `block_byte_cap` is already clamped to `budget` above, so
                        // each sub-flushed block stays within the budget.
                        log::warn!(
                            "scx sort: grouped shard {shard_idx} (dominated by group {label:?}) \
                             would need ~{max_bytes} bytes if buffered whole, exceeding \
                             --memory-budget {budget}; the group will be sub-flushed across \
                             multiple shards at the {block_byte_cap}-byte block cap \
                             (min of --group-write-block-bytes and --memory-budget)"
                        );
                    } else {
                        return Err(OpsError::InvalidInput(format!(
                            "scx sort: grouped shard {shard_idx} (dominated by group {label:?}) \
                             needs ~{max_bytes} bytes to buffer but --memory-budget is {budget} \
                             and the block sub-flush is disabled (--group-write-block-bytes 0); \
                             raise --memory-budget / --group-target-bytes, enable the sub-flush, \
                             or drop --group-by"
                        )));
                    }
                }
            }
            grouped_reference_labels = go.reference_labels;
            (go.perm, Some(plan))
        } else if let Some(seed) = opts.shuffle {
            // 1D: the third pass-0 producer. Same shape and same `Vec<u64>`
            // memory as `stable_argsort`, so every downstream consumer —
            // in-memory take, spill-scatter routing, obsp remap, all three X
            // emitters — is untouched.
            log::info!("scx sort: pass 0a seeded permutation over {n_live} live rows");
            (crate::shuffle_order::seeded_permutation(n_live, seed), None)
        } else {
            let extractor = SortKeyExtractor::new(&live_keys.schema(), &opts.by, opts.reverse)?;
            let rows = extractor.rows(&live_keys)?;
            log::info!("scx sort: pass 0a key rows built; argsort over {n_live} rows");
            // Local indices into the live sequence, in sorted order (stable, ties
            // by source id). `live_keys` and `live_obs` are filtered from the same
            // row universe in the same shard-concatenated order, so these local
            // indices apply to both.
            (stable_argsort(&rows, 0), None)
        };
    // Output row -> original (global) old row id.
    let order_old: Vec<u64> = order_local.iter().map(|&l| live_ids[l as usize]).collect();

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
    // order; the single-modality engine below handles the
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
    // Preserve row-group framing: a v4 (framed) input yields a v4 output whose
    // re-encoded shards are all framed (via `set_framing` below), so sorting a
    // default file no longer silently downgrades it to unframed v3.
    let output_framed = in_header.format_version >= CURRENT_FORMAT_VERSION;
    let out_header = FileHeader {
        format_version: if output_framed {
            CURRENT_FORMAT_VERSION
        } else {
            scx_format_io::rewrite_output_format_version(&[in_header.format_version], 1)
        },
        flags: out_flags,
        n_obs: n_live as u64,
        n_vars,
        shard_target_rows: opts.shard_target_rows,
        index_dtype: in_header.index_dtype,
        ..Default::default()
    };
    let mut writer = ScxWriter::new(output, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);
    writer.set_framing(framing_for_rewrite(opts.codec, output_framed, "the input")?);

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
    //
    // 1D: shuffle mode has no key at all, so all three of these are "not
    // applicable" rather than false-by-accident. The guard is load-bearing:
    // `opts.by[0]` would panic on the empty `by`, and it runs unconditionally.
    // With `single_key = false` the selector routes InMemory / ExternalPartition
    // and never considers K-pass — which is correct, since K-pass emits in
    // category order and cannot express an arbitrary permutation.
    let single_key = opts.shuffle.is_none() && opts.by.len() == 1;
    let leading_categorical = opts.shuffle.is_none()
        && !is_numeric(
            live_keys
                .schema()
                .field_with_name(&opts.by[0])
                .map(|f| f.data_type().clone())
                .unwrap_or(DataType::Utf8),
        );
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

    let mut strategy = force_strategy.unwrap_or_else(|| {
        select_strategy(
            opts,
            est_x_bytes,
            single_key && leading_categorical && !leading_key_has_nulls,
            k,
        )
    });

    // F1: K-pass emits in category order and bypasses the planner's
    // reference-first global order, so grouped sorts use the global-order paths.
    // Route to the budgeted external path (memory-bounded), since K-pass is only
    // ever selected when a budget forced it.
    if group_plan.is_some() && strategy == SortStrategy::KPassByCategory {
        log::info!(
            "scx sort: --group-by selected; routing K-pass -> ExternalPartition for global order"
        );
        strategy = SortStrategy::ExternalPartition;
    }

    // `block_byte_cap` (budget-clamped) was computed with the group plan above.

    // In-memory grouped writes take the parallel fast path, which
    // bypasses the row-by-row `CsrEmitter`. It reproduces the emitter's exact
    // shard boundaries (planned breaks + the Phase-0 block sub-flush) so its
    // output is byte-identical, but gathers + encodes blocks in parallel via
    // rayon. Gated to `InMemory` + grouped; `SCX_SORT_NO_INMEM_FAST` forces the
    // legacy emitter path (byte-parity testing / operational safety).
    let use_fast_path =
        strategy == SortStrategy::InMemory && std::env::var_os("SCX_SORT_NO_INMEM_FAST").is_none();
    let output_shard_row_ranges = if let (true, Some(plan)) = (use_fast_path, group_plan.as_ref()) {
        emit_x_in_memory_grouped_fast(
            &reader,
            &mut writer,
            &order_old,
            plan,
            value_encoding,
            n_vars_u32,
            opts.codec,
            opts.bitmap,
            block_byte_cap,
            opts.memory_budget,
        )?
    } else {
        let mut x_emitter = CsrEmitter::new(
            EmitTarget::X,
            None,
            opts.shard_target_rows,
            opts.codec,
            value_encoding,
            n_vars_u32,
            opts.bitmap,
            block_byte_cap,
        );
        // F1: in grouped mode the planner's shard starts are authoritative (the
        // legacy fixed-size cap is disabled inside the emitter).
        if let Some(plan) = &group_plan {
            x_emitter.set_group_breaks(plan.shard_starts.clone());
        }
        match strategy {
            SortStrategy::InMemory => {
                emit_x_in_memory(&reader, &mut writer, &mut x_emitter, &order_old)?;
            }
            SortStrategy::KPassByCategory => {
                // 1D: auto-selection can never land here in shuffle mode
                // (`single_key` is false), but `force_strategy` can. Refuse
                // loudly rather than fall through to the `categories` unwrap
                // below with a message about a categorical key that does not
                // exist: `emit_x_kpass` scans the input once per category and
                // is structurally incapable of emitting an arbitrary
                // permutation, so this is a capability gap, not a config error.
                if opts.shuffle.is_some() {
                    return Err(OpsError::InvalidInput(
                        "scx sort: the K-pass strategy cannot express a random permutation \
                         (it emits rows grouped by category, in category order); --shuffle \
                         runs on the in-memory or external-partition strategy"
                            .to_string(),
                    ));
                }
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
                // Per-row decoded footprint (CSR ≈ 16 B/nnz, matching
                // `partition_target_rows`). `ceil` so a fractional estimate on sparse
                // data never *undercounts* the partition footprint the budget guard
                // re-validates after widening.
                let per_row = (n_vars as f64 * density * 16.0).max(1.0).ceil() as u64;
                // Refuse if a single output shard's rows cannot fit the budget
                // (T4.7 — no silent cap). Kept in f64 until after multiplying so the
                // fractional per-row estimate is not truncated before scaling.
                if let Some(budget) = opts.memory_budget {
                    let per_shard =
                        (opts.shard_target_rows.max(1) as f64 * n_vars as f64 * density * 16.0)
                            as u64;
                    if per_shard > budget {
                        return Err(OpsError::InvalidInput(format!(
                            "scx sort: --memory-budget {budget} too small for one output shard \
                         (~{per_shard} bytes for {} rows); raise the budget or lower --shard-size",
                            opts.shard_target_rows
                        )));
                    }
                }
                // Cap simultaneously-open spill files (EMFILE) and re-validate that a
                // widened partition — which pass 2 loads whole into RAM — still fits
                // the budget (the per-shard guard above only bounds one output shard).
                let (p, _n_parts) = cap_spill_partitions(p, n_live, per_row, opts.memory_budget)?;
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
        x_emitter.ranges.clone()
    };

    // ----- F1/F6: write the GroupIndex sidecar (reconciled to emitted shards) -----
    if let (Some(plan), Some(group_col)) = (&group_plan, &opts.group_by) {
        // F6 Phase 0: the block-level sub-flush may split a single planner record
        // (one (label, role) run) across several emitted output shards. Reconcile
        // by intersecting each planner record with the actual emitter shard
        // ranges and emitting one `GroupRecord` per (record ∩ shard) — carrying
        // the true emitted shard index and clipped `[row_start, row_stop)`. This
        // keeps each *record* within one shard (the read side's `shard` contract)
        // while allowing a *label* to span multiple records. When no sub-flush
        // occurred, every record maps to exactly one shard and the output is
        // identical to `plan.records` (byte-identical sidecar — no regression).
        let mut reconciled: Vec<crate::group_plan::GroupRecord> =
            Vec::with_capacity(plan.records.len());
        for r in &plan.records {
            let mut covered = false;
            for (shard_idx, &(s, e)) in output_shard_row_ranges.iter().enumerate() {
                let start = r.row_start.max(s);
                let stop = r.row_stop.min(e);
                if start < stop {
                    reconciled.push(crate::group_plan::GroupRecord {
                        label: r.label.clone(),
                        shard: shard_idx as u32,
                        row_start: start,
                        row_stop: stop,
                        role: r.role,
                    });
                    covered = true;
                }
            }
            // A non-empty planner record that overlaps no emitted shard means the
            // emitter and planner disagree on the row count — real corruption.
            if !covered && r.row_start < r.row_stop {
                return Err(OpsError::InvalidInput(format!(
                    "scx sort: group record range [{}, {}) for label {:?} escapes every emitted \
                     X shard range; refusing to write an inconsistent group index",
                    r.row_start, r.row_stop, r.label
                )));
            }
        }
        let reference_shard = reconciled
            .iter()
            .find(|r| r.role == crate::group_plan::Role::Reference)
            .map(|r| r.shard);
        // This plan is only a vehicle for `to_sidecar_json`, which serializes
        // exactly `{group_by, reference_shard, reference_labels, records}` —
        // `shard_starts`/`n_shards` are NOT serialized. The planner's original
        // `shard_starts` is stale after a sub-flush (fewer entries than emitted
        // shards), so leave it empty rather than carry an inconsistent value;
        // `n_shards` is set to the true emitted-shard count for completeness.
        let reconciled_plan = crate::group_plan::GroupPlan {
            shard_starts: Vec::new(),
            records: reconciled,
            reference_shard,
            n_shards: output_shard_row_ranges.len() as u32,
        };
        // reference_labels computed alongside the order in `compute_grouped_order`
        // (explicit set for `Labels`, else the distinct labels that ended up
        // reference for `ReferenceSpec::Column`).
        let payload = reconciled_plan.to_sidecar_json(group_col, &grouped_reference_labels);
        let bytes = serde_json::to_vec(&payload).map_err(|e| {
            OpsError::InvalidInput(format!("scx sort: failed to serialize group index: {e}"))
        })?;
        writer.write_group_index(&bytes)?;
    }

    // ----- layers (in-memory gather by the same order) -----
    emit_layers_in_memory(&reader, &mut writer, &order_old, opts)?;

    // ----- obsm (in-memory take) -----
    // All four mapping families are re-emitted as row-shards, at the same
    // `shard_target_rows` the X and obs shards above use. Sort applies a global
    // permutation, so the input's shard boundaries are meaningless here and
    // there is nothing to preserve; what matters is that the output is
    // bounded-readable, which one collapsed section is not.
    let mut obsm: Vec<_> = reader.read_all_obsm()?.into_iter().collect();
    obsm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsm {
        let reordered = take_rows(batch, &order_old)?;
        crate::rewrite_helpers::write_obsm_sharded(
            &mut writer,
            name,
            &reordered,
            opts.shard_target_rows,
        )?;
    }

    // ----- uns / varm / varp passthrough (var axis: neither deleted nor permuted) -----
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }
    let mut varm: Vec<_> = reader.read_all_varm()?.into_iter().collect();
    varm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varm {
        crate::rewrite_helpers::write_varm_sharded(
            &mut writer,
            name,
            batch,
            opts.shard_target_rows,
        )?;
    }
    let mut varp: Vec<_> = reader.read_all_varp()?.into_iter().collect();
    varp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varp {
        crate::rewrite_helpers::write_varp_sharded(
            &mut writer,
            name,
            batch,
            opts.shard_target_rows,
        )?;
    }
    // obsp (obs×obs COO): remap both endpoints through the sort permutation
    // (T5.3). `new_pos_of_old` is exactly the old→new map `remap_obsp_coo`
    // expects (-1 drops edges touching a deleted obs; dims collapse to n_live).
    let mut obsp: Vec<_> = reader.read_all_obsp()?.into_iter().collect();
    obsp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsp {
        let remapped = crate::compact::remap_obsp_coo(batch, &new_pos_of_old)?;
        crate::rewrite_helpers::write_obsp_sharded(
            &mut writer,
            name,
            &remapped,
            opts.shard_target_rows,
        )?;
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
            state.layout.out_schema.clone(),
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
        grouping_provenance(opts),
        opts.shuffle,
    ));
    writer.write_provenance(prov)?;

    crate::carry::audit_staged(crate::carry::RewriteOp::Sort, &[reader.catalog()], &writer)?;
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
    // Preserve framing (see the single-modality path).
    let output_framed = in_header.format_version >= CURRENT_FORMAT_VERSION;
    let out_header = FileHeader {
        format_version: if output_framed {
            CURRENT_FORMAT_VERSION
        } else {
            scx_format_io::rewrite_output_format_version(&[in_header.format_version], 2)
        },
        flags: out_flags,
        n_obs: n_live as u64,
        n_vars: max_n_vars,
        shard_target_rows: opts.shard_target_rows,
        index_dtype: in_header.index_dtype,
        ..Default::default()
    };
    let mut writer = ScxWriter::new(output, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);
    writer.set_framing(framing_for_rewrite(opts.codec, output_framed, "the input")?);

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
        // Scan ALL of this modality's shards for the widest encoding (SCX-004):
        // sort reorders rows across shards, so a first-shard-only encoding would
        // truncate a later float/wider-int shard.
        let ve = widest_value_encoding(reader, &reader.catalog().csr_shards_for_modality(in_id))?;
        let csr = reader.read_all_csr_shards_for(in_id)?;
        let mut emitter = CsrEmitter::new(
            EmitTarget::X,
            Some(out_id),
            opts.shard_target_rows,
            opts.codec,
            ve,
            info.n_vars as u32,
            opts.bitmap,
            opts.group_write_block_bytes
                .unwrap_or(DEFAULT_GROUP_WRITE_BLOCK_BYTES),
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
            let meta = scx_format_io::DenseShardMetadata::new(0, 0, n, n);
            writer.write_varm_shard_for(out_id, &key, meta, &batch)?;
        }

        // Per-modality uns.
        if let Ok(uns) = reader.read_uns_for(in_id) {
            writer.write_uns_for(out_id, &uns)?;
        }

        // Per-modality layers (obs-axis → reorder; in-memory gather).
        for layer_name in modality_layer_names(reader, in_id, &info.name) {
            // All-shard widest encoding, same rationale as X above (SCX-004).
            let lve = widest_value_encoding(
                reader,
                &reader
                    .catalog()
                    .layer_csr_shards_for_modality(in_id, &layer_name),
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
                opts.group_write_block_bytes
                    .unwrap_or(DEFAULT_GROUP_WRITE_BLOCK_BYTES),
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
        crate::rewrite_helpers::write_obsm_sharded(
            &mut writer,
            &key,
            &take_rows(&batch, order_old)?,
            opts.shard_target_rows,
        )?;
    }
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "varm/",
        SectionType::VarmEmbedding,
        SectionType::VarmEmbeddingShard,
    ) {
        crate::rewrite_helpers::write_varm_sharded(
            &mut writer,
            &key,
            &reader.read_varm(&key)?,
            opts.shard_target_rows,
        )?;
    }
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "varp/",
        SectionType::VarpEmbedding,
        SectionType::VarpEmbeddingShard,
    ) {
        crate::rewrite_helpers::write_varp_sharded(
            &mut writer,
            &key,
            &reader.read_varp(&key)?,
            opts.shard_target_rows,
        )?;
    }
    for key in crate::compact::discover_modality_keys(
        reader,
        0,
        "obsp/",
        SectionType::ObspEmbedding,
        SectionType::ObspEmbeddingShard,
    ) {
        let batch = reader.read_obsp(&key)?;
        crate::rewrite_helpers::write_obsp_sharded(
            &mut writer,
            &key,
            &crate::compact::remap_obsp_coo(&batch, new_pos_of_old)?,
            opts.shard_target_rows,
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
        grouping_provenance(opts),
        opts.shuffle,
    ));
    writer.write_provenance(prov)?;
    crate::carry::audit_staged(crate::carry::RewriteOp::Sort, &[reader.catalog()], &writer)?;
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

/// F6 Phase 0 — default byte cap on the emitter's grouped-mode accumulation
/// buffer (256 MB). Large enough to amortize codec overhead, small enough to
/// bound grouped-write peak RSS at one block per group instead of one whole
/// group. Overridable via `SortOptions.group_write_block_bytes` /
/// `--group-write-block-bytes`.
const DEFAULT_GROUP_WRITE_BLOCK_BYTES: u64 = 256 * 1024 * 1024;

/// Accumulates re-ordered rows and flushes them as CSR (or layer) shards;
/// `modality_id = Some(id)` routes to the per-modality writers. For X targets
/// with a non-`Off` `bitmap` policy it also emits a detection-bitmap sidecar
/// per shard.
struct CsrEmitter {
    target: EmitTarget,
    modality_id: Option<u8>,
    shard_target: u64,
    codec: ResolvedCodec,
    value_encoding: ValueEncoding,
    n_vars: u32,
    bitmap: BitmapPolicy,
    /// F6 Phase 0: byte cap on the accumulation buffer in grouped mode. `0`
    /// disables the sub-flush (legacy never-split-across-shards behaviour).
    block_byte_cap: u64,
    acc_indptr: Vec<u64>,
    acc_indices: Vec<u32>,
    acc_values: Vec<u8>,
    acc_row_count: u64,
    emitted_rows: u64,
    shard_idx: u32,
    ranges: Vec<(u64, u64)>,
    /// F1: sorted emit-row indices at which to seal a shard *before* pushing
    /// that row (the planner's authoritative shard starts). `None` => legacy
    /// fixed-size behaviour (byte-identical to pre-F1). `Some(_)` => planned-break
    /// (grouped) mode: the legacy `shard_target` size cap is disabled and all
    /// breaks come from here — even when the vec is empty (a plan that collapses
    /// to a single shard, e.g. one oversized group), which must NOT fall back to
    /// the row cap or it would split the group.
    group_breaks: Option<Vec<u64>>,
    break_cursor: usize,
}

impl CsrEmitter {
    #[allow(clippy::too_many_arguments)]
    fn new(
        target: EmitTarget,
        modality_id: Option<u8>,
        shard_target: u32,
        codec: ResolvedCodec,
        value_encoding: ValueEncoding,
        n_vars: u32,
        bitmap: BitmapPolicy,
        block_byte_cap: u64,
    ) -> Self {
        Self {
            target,
            modality_id,
            shard_target: shard_target.max(1) as u64,
            codec,
            value_encoding,
            n_vars,
            bitmap,
            block_byte_cap,
            acc_indptr: vec![0],
            acc_indices: Vec::new(),
            acc_values: Vec::new(),
            acc_row_count: 0,
            emitted_rows: 0,
            shard_idx: 0,
            ranges: Vec::new(),
            group_breaks: None,
            break_cursor: 0,
        }
    }

    /// F1: install the planner's shard-break offsets. Must be called before the
    /// first `push_row`. Switches the emitter into planned-break mode (legacy
    /// size cap disabled), even when `breaks` is empty (single-shard plan).
    fn set_group_breaks(&mut self, breaks: Vec<u64>) {
        self.group_breaks = Some(breaks);
        self.break_cursor = 0;
    }

    fn push_row(&mut self, writer: &mut ScxWriter, indices: &[i32], data: &[f32]) -> Result<()> {
        // F1: seal at a planned group-shard boundary *before* accumulating this
        // row. `emitted_rows + acc_row_count` is the global index of the row
        // about to be pushed.
        if let Some(breaks) = &self.group_breaks {
            if self.break_cursor < breaks.len()
                && self.emitted_rows + self.acc_row_count == breaks[self.break_cursor]
            {
                self.flush(writer)?;
                self.break_cursor += 1;
            }
        }
        // Every caller passes a CSR row's two parallel arrays, so these agree.
        // Worth stating: the per-nonzero loop this replaced indexed `data[k]`
        // from an `indices` walk, which would have panicked on a short `data`
        // and ignored a long one. The batch call encodes all of `data`, so a
        // future caller with mismatched lengths would write a shard whose
        // values and indices disagree -- silently, without this.
        // `assert_eq!`, not `debug_assert_eq!`. The loop this replaced indexed
        // `data[k]` from an `indices` walk, so a short `data` panicked in
        // release too; guarding only in debug would have turned a loud failure
        // into a shard whose values and indices disagree on disk. One length
        // comparison per row is not measurable against the encode.
        assert_eq!(
            indices.len(),
            data.len(),
            "push_row needs one value per index (CSR row invariant)"
        );
        // One call per row, not per nonzero -- see `helpers::encode_values`.
        // The accumulator's byte layout is unchanged, which matters here beyond
        // speed: `block_cut_bytes_per_nnz` prices these exact bytes, and
        // `emit_x_in_memory_grouped_fast` reproduces this loop's cut points.
        self.acc_indices
            .extend(indices.iter().map(|&col| col as u32));
        encode_values(&mut self.acc_values, data, self.value_encoding)?;
        let prev = *self.acc_indptr.last().unwrap();
        self.acc_indptr.push(prev + indices.len() as u64);
        self.acc_row_count += 1;
        // Legacy fixed-size cap ONLY in non-grouped mode (the planner never
        // splits a group, so grouped breaks are authoritative — including the
        // single-shard case where the break list is empty).
        if self.group_breaks.is_none() && self.acc_row_count >= self.shard_target {
            self.flush(writer)?;
        }
        // F6 Phase 0: block-level sub-flush in grouped mode. When the accumulated
        // (encoded) CSR of the current shard reaches `block_byte_cap`, seal it as
        // a standalone shard *within* the current group — this is what bounds
        // grouped-write peak RSS at one block instead of one whole (possibly
        // 100K+-cell) group. The planned group-break logic above is untouched:
        // it keys off the global row index and `break_cursor`, neither of which a
        // sub-flush perturbs (a sub-flush advances `emitted_rows` and resets
        // `acc_row_count`, so `emitted_rows + acc_row_count` still equals the next
        // planned break). An oversized group thus spans multiple output shards.
        if self.group_breaks.is_some() && self.block_byte_cap > 0 && self.acc_row_count > 0 {
            let acc_bytes = (self.acc_indices.len() * 4 + self.acc_values.len()) as u64;
            if acc_bytes >= self.block_byte_cap {
                self.flush(writer)?;
            }
        }
        Ok(())
    }

    fn flush(&mut self, writer: &mut ScxWriter) -> Result<()> {
        if self.acc_row_count == 0 {
            return Ok(());
        }
        let codec = seed_codec(self.codec, &self.acc_values, self.value_encoding);
        let row_start = self.emitted_rows;
        let n_rows = self.acc_row_count;
        let nnz = self.acc_indices.len();
        let shard = scx_format_io::ShardBuffers::new(
            &self.acc_indptr,
            &self.acc_indices,
            &self.acc_values,
            codec,
            self.value_encoding,
        );
        match (&self.target, self.modality_id) {
            (EmitTarget::X, None) => writer.write_csr_shard(
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
            )?,
            (EmitTarget::X, Some(id)) => writer.write_csr_shard_for(id, row_start, shard)?,
            (EmitTarget::Layer(name), None) => {
                writer.write_layer_csr_shard(name, self.shard_idx, row_start, shard)?
            }
            (EmitTarget::Layer(name), Some(id)) => {
                writer.write_layer_csr_shard_for(id, name, self.shard_idx, row_start, shard)?
            }
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

/// Cap the grouped-write fast path's parallel `concurrency` so its **total** peak
/// transient honors `--memory-budget` (H1).
///
/// Each in-flight block holds gather buffers (u32 index + f32 value =
/// [`GROUP_BYTES_PER_NNZ`] bytes per nnz), the encoder's own copy of the
/// values, and the framed encode's live buffers — `ENCODE_PHASE_MULTIPLE`
/// times the gather cost, over the `nnz ≈ block_byte_cap / per_nnz_bytes` a
/// full block holds at the cap. We cap `concurrency` to `budget / per_block`
/// (min 1), so `concurrency × per_block ≤ budget`; concurrency 1 gives parity
/// with the one-block-at-a-time `CsrEmitter`.
///
/// That inequality holds because the **caller's** `block_byte_cap` is clamped
/// to `budget * block_cut_bytes_per_nnz / (ENCODE_PHASE_MULTIPLE *
/// GROUP_BYTES_PER_NNZ)` (`sort_engine.rs`'s `block_byte_cap` binding), so a
/// block filled **to** the cap has a whole phase that fits the budget and the
/// `.max(1)` floor is not the branch that breaks it. Two earlier spellings of that clamp did
/// not hold: clamping to the raw budget made it false in the ordinary f32 case
/// (cap 256 MiB → `per_block` 1.5 GiB against a 256 MiB budget), and clamping
/// to `budget / ENCODE_PHASE_MULTIPLE` made it false by
/// `GROUP_BYTES_PER_NNZ / block_cut_bytes_per_nnz` — 1.0 at f32 but **1.6 at
/// `Uint8`**, because the cap counts narrowed accumulator bytes and the gather
/// holds `i32 + f32`. Converting the budget through the cap's own unit closes
/// that: at f32 the arithmetic is unchanged, and at a narrower encoding the cap
/// shrinks by exactly the overshoot.
///
/// ⚠️ **One residual, which predates this model, and it is why the paragraph
/// above says "filled to the cap" rather than "any block".** A whole row is
/// admitted before the cap is tested, so one pathologically deep row can
/// overshoot `nnz_per_block` — and therefore the budget — by its own length.
/// `.max(1)` cannot honour a budget smaller than one block's phase either:
/// there is no concurrency below one. So this is a much better estimate than
/// the gather-only model it replaced, and still not a proven ceiling.
///
/// Conservative in the other direction on one axis, symmetrically with
/// `scx-convert`'s `ENCODE_TRANSIENT_MULTIPLE`: the `4x` encode share assumes
/// the dual-candidate `rayon::join`, which only `--codec auto`/`compact`
/// incur, and it is charged whatever `writer.framing()` actually carries.
///
/// The encode side is `4×` and not the `~1×` this used to charge, because this
/// path takes `writer.framing()`, which under `--codec auto` carries a
/// `decode_target`: `encode_shard_adaptive` dual-encodes the block, and
/// `encode_shard_framed` holds every row group's encoded bytes alongside the
/// streams assembled from them. `scx-convert/src/budget.rs` carries the
/// derivation and the measured figures; `ENCODE_PHASE_MULTIPLE`'s doc
/// says why the constant is declared in this crate rather than shared.
///
/// What `per_block` still does **not** price, named rather than left implicit:
/// the block's `local_indptr` (8 B/row, small beside a block's nnz) and the
/// optional `BitmapShard`. Neither is the encode term, and neither was priced
/// before either.
///
/// `None` budget keeps the full rayon-thread concurrency (the user opted out of
/// budgeting). When a budget **is** set but the sub-flush is disabled
/// (`block_byte_cap == 0`, i.e. `--group-write-block-bytes 0`), each block is a
/// whole shard whose byte size isn't known here, so concurrency falls back to 1
/// (the one-block-at-a-time `CsrEmitter` bound) rather than being left uncapped —
/// the `InMemory` strategy gate guarantees a single shard fits the budget, so
/// peak stays ~≤ 2× budget. (`per_nnz_bytes` is `4 + value_width` ≥ 5, never 0;
/// the guard is defensive.)
fn grouped_fast_concurrency(
    threads: usize,
    block_byte_cap: u64,
    per_nnz_bytes: u64,
    memory_budget: Option<u64>,
) -> usize {
    match memory_budget {
        // Sub-flush enabled: cap concurrency so `concurrency × per-block ≤ budget`.
        Some(budget) if block_byte_cap > 0 && per_nnz_bytes > 0 => {
            let nnz_per_block = (block_byte_cap / per_nnz_bytes).max(1);
            let per_block = (ENCODE_PHASE_MULTIPLE * GROUP_BYTES_PER_NNZ)
                .saturating_mul(nnz_per_block)
                .max(1);
            // Compare in u64 before narrowing to avoid truncation on 32-bit usize.
            let by_budget = (budget / per_block).max(1).min(threads as u64);
            by_budget as usize
        }
        // Budget set, sub-flush disabled (whole-shard blocks): bound to one at a time.
        Some(_) => 1,
        None => threads,
    }
}

/// In-memory grouped-write fast path (X only).
///
/// Reads the source CSR once, computes the output shard/block boundaries up
/// front, then gathers + encodes each block **in parallel** via
/// [`encode_one_shard`] (with `opts.value_encoding` fixed), writing the pre-encoded bytes (and
/// detection bitmaps) sequentially in block order. It bypasses the row-by-row
/// [`CsrEmitter::push_row`] copy and parallelizes the encode, yet is
/// **byte-identical** to the emitter path because it:
///   - reproduces the emitter's exact cut points — the planner's `shard_starts`
///     breaks plus the block sub-flush at `block_byte_cap` (same
///     `indices*4 + values` accumulation rule);
///   - encodes with the same file-wide `value_encoding` (override, not the
///     per-shard auto-detect), the same codec (`Auto → select_codec` via
///     `ModalityType::Rna`, else the explicit id), the same `index_dtype`
///     (from `n_vars`, matching `write_shard_inner`), the same `framing`, and
///     the same `X_shard_{idx}` naming;
///   - builds detection bitmaps with the sort-side [`bitmap_should_build`] gate.
///
/// Memory is bounded: blocks are encoded + written in chunks of `concurrency`,
/// so at most ~`concurrency` blocks' gather + encoded buffers are in flight on
/// top of the resident source CSR — O(concurrency × block cap), independent of
/// the block count (it does NOT buffer the whole encoded matrix). `block_byte_cap`
/// is clamped to `--memory-budget / ENCODE_PHASE_MULTIPLE` upstream,
/// which is what bounds a *single* block's whole-phase transient — clamping it
/// to the raw budget did not (see [`grouped_fast_concurrency`]). The **total** peak is `concurrency × per-block transient`, so when a
/// `memory_budget` is set `concurrency` is additionally capped to
/// `budget / per-block transient` (see [`grouped_fast_concurrency`]) — otherwise a
/// many-core host would run `rayon::current_num_threads()` blocks in flight and
/// blow past the budget by that factor (the F6 OOM fix would be defeated by
/// default). With no budget the concurrency is only bounded by the rayon pool
/// (`RAYON_NUM_THREADS`). Note the budget bounds the *transient*: total peak is
/// the resident source CSR (≤ budget under the `InMemory` gate) plus the
/// transient (≤ budget), i.e. up to ~2× budget — the documented `CsrEmitter`
/// parity floor, not a per-call ceiling. Returns the emitted `[row_start,
/// row_stop)` ranges (what `CsrEmitter::ranges` would hold) for the GroupIndex
/// write-back.
#[allow(clippy::too_many_arguments)]
fn emit_x_in_memory_grouped_fast(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    order_old: &[u64],
    plan: &crate::group_plan::GroupPlan,
    value_encoding: ValueEncoding,
    n_vars_u32: u32,
    codec: ResolvedCodec,
    bitmap: BitmapPolicy,
    block_byte_cap: u64,
    memory_budget: Option<u64>,
) -> Result<Vec<(u64, u64)>> {
    let csr = reader.read_all_csr_shards()?;
    let n_obs = order_old.len();
    let shard_starts = &plan.shard_starts;

    // Matches how `acc_values` grows in `CsrEmitter` (`encode_value` pushes
    // this many bytes per nnz), so the block-cap cut points below are
    // identical — and it is the same function the cap's budget clamp uses.
    let per_nnz_bytes = block_cut_bytes_per_nnz(value_encoding);

    // 1. Compute block row-ranges, reproducing `CsrEmitter::push_row` exactly: a
    //    planned break at emit-row `p` seals the block ending at `p`; the
    //    block-cap sub-flush seals the block ending at `p+1` once the accumulated
    //    `indices.len()*4 + values.len()` reaches `block_byte_cap`.
    // Each block records `(row_start, row_stop, nnz)`; the nnz (the running
    // `cum_nnz` at the cut) lets the encode closure pre-size its buffers exactly.
    let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
    let mut block_start: usize = 0;
    let mut cum_nnz: u64 = 0;
    let mut break_cursor: usize = 0;
    for (p, &old_row) in order_old.iter().enumerate() {
        if break_cursor < shard_starts.len() && p as u64 == shard_starts[break_cursor] {
            if p > block_start {
                blocks.push((block_start, p, cum_nnz as usize));
                block_start = p;
                cum_nnz = 0;
            }
            break_cursor += 1;
        }
        let old = old_row as usize;
        let nnz_p = (csr.indptr[old + 1] - csr.indptr[old]) as u64;
        cum_nnz += nnz_p;
        if block_byte_cap > 0 && cum_nnz * per_nnz_bytes >= block_byte_cap && p + 1 > block_start {
            blocks.push((block_start, p + 1, cum_nnz as usize));
            block_start = p + 1;
            cum_nnz = 0;
        }
    }
    if n_obs > block_start {
        blocks.push((block_start, n_obs, cum_nnz as usize));
    }

    // 2. Encode + write in **bounded parallel chunks**. Each chunk of up to
    //    `concurrency` blocks is gathered + encoded in parallel (rayon), then
    //    written sequentially (ScxWriter is single-threaded for section ordering)
    //    before the next chunk is encoded — so at most `concurrency` blocks'
    //    gather + encoded buffers are ever in flight, instead of materializing
    //    every encoded block up front. This keeps peak memory O(concurrency ×
    //    block cap) on top of the resident source CSR, independent of block count.
    //    Output is byte-identical to the serial `CsrEmitter`: same blocks, same
    //    order, same `X_shard_{global_idx}` naming.
    let explicit_codec = codec.explicit_codec;
    // CSR index dtype from the minor (var) axis bound, matching `write_shard_inner`.
    let index_dtype: u8 = if (n_vars_u32 as u64).saturating_sub(1) <= u16::MAX as u64 {
        0
    } else {
        1
    };
    let framing = writer.framing();
    // Total peak is `concurrency × per-block transient`, so cap `concurrency` by
    // the budget (H1). Without this, a many-core host runs
    // `rayon::current_num_threads()` blocks in flight and overshoots the budget by
    // that factor even though each block is individually budget-clamped.
    let concurrency = grouped_fast_concurrency(
        rayon::current_num_threads().max(1),
        block_byte_cap,
        per_nnz_bytes,
        memory_budget,
    );

    let mut ranges: Vec<(u64, u64)> = Vec::with_capacity(blocks.len());
    let mut base: usize = 0;
    for chunk in blocks.chunks(concurrency) {
        let encoded: Vec<(PreEncodedSection, Option<BitmapShard>)> = chunk
            .par_iter()
            .enumerate()
            .map(
                |(j, &(r0, r1, block_nnz))| -> Result<(PreEncodedSection, Option<BitmapShard>)> {
                    // Global block index → stable `X_shard_{idx}` naming across chunks.
                    let idx = base + j;
                    let n_rows = r1 - r0;
                    let mut local_indptr: Vec<u64> = Vec::with_capacity(n_rows + 1);
                    local_indptr.push(0);
                    let mut local_indices: Vec<u32> = Vec::with_capacity(block_nnz);
                    let mut local_values: Vec<f32> = Vec::with_capacity(block_nnz);
                    for &op in &order_old[r0..r1] {
                        let o = op as usize;
                        let s = csr.indptr[o] as usize;
                        let e = csr.indptr[o + 1] as usize;
                        local_indices.extend(csr.indices[s..e].iter().map(|&c| c as u32));
                        local_values.extend_from_slice(&csr.data[s..e]);
                        local_indptr.push(local_indices.len() as u64);
                    }
                    let nnz = local_indices.len();
                    let mut enc_opts = scx_format_io::EncodeShardOptions::new(
                        format!("X_shard_{idx}"),
                        SectionType::CsrShard,
                        n_vars_u32 as u64,
                        r0 as u64,
                        index_dtype,
                    );
                    enc_opts.explicit_codec = explicit_codec;
                    enc_opts.framing = framing;
                    enc_opts.value_encoding = Some(value_encoding);
                    let section =
                        encode_one_shard(&local_indptr, &local_indices, &local_values, &enc_opts)?;
                    let bm = if bitmap_should_build(bitmap, n_rows as u64, nnz, n_vars_u32) {
                        Some(BitmapShard::build_from_csr(
                            r0 as u64,
                            n_rows as u32,
                            n_vars_u32,
                            &local_indptr,
                            &local_indices,
                        ))
                    } else {
                        None
                    };
                    Ok((section, bm))
                },
            )
            .collect::<Result<Vec<_>>>()?;

        for (&(r0, r1, _), (section, bm)) in chunk.iter().zip(encoded) {
            writer.write_preencoded_shard(section)?;
            if let Some(bm) = bm {
                writer.write_bitmap_shard(&bm)?;
            }
            ranges.push((r0 as u64, r1 as u64));
        }
        base += chunk.len();
    }
    Ok(ranges)
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
    // Partition sizing + the EMFILE cap and budget re-validation are owned by the
    // caller (`cap_spill_partitions`); `p` arrives already widened.
    let p = p.max(1);
    let n_parts = n_live.div_ceil(p);
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
            opts.group_write_block_bytes
                .unwrap_or(DEFAULT_GROUP_WRITE_BLOCK_BYTES),
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
// global order. Used only for single-modality sorts with a `--memory-budget`.
//
// ## Categoricals: spill the codes, never the values
//
// A dictionary column cannot be concatenated across shards without first
// reconciling the vocabularies, so this path used to decode every categorical
// to its value type on the way in and re-encode it with
// `cast(col, Dictionary(Int32, V))` on the way out. Arrow's plain -> dictionary
// packing builds the vocabulary from the rows of the array it is handed, which
// made the re-encode lossy three ways: a declared-but-unused `pd.Categorical`
// level vanished (no row references it, so no shard emits it, and no reader can
// recover it); the declared order was replaced by a per-shard first-occurrence
// order while the `scx.categorical.ordered` stamp still claimed the column was
// ordered; and every output shard declared a different list. A boolean
// categorical did not survive at all — arrow has no boolean dictionary packing
// (`Unsupported output type for dictionary packing: Boolean`), so the sort
// failed outright.
//
// Nothing is decoded now. Pass 0 ([`build_obs_vocabulary_template`]) folds every
// input shard into a 0-row batch holding the **union of the declared values**,
// through the same `scx_format_io` pipeline `read_obs()` uses, so the vocabulary
// and key width are the ones the in-memory path would have written. Pass 1
// remaps each shard's keys onto that union and spills the `Int32` **codes** —
// 4 B/row instead of the value width. Pass 2 rebuilds the dictionary from the
// codes and the union's values `Arc`, which every output shard then shares, as
// the in-memory path's `RecordBatch::slice` does. The result is byte-identical
// obs sections on the two paths, which is what `obs_spill_obs_matches_in_memory`
// pins.
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
///
/// The per-row figure the caller passes is the *decoded* shard's
/// `get_array_memory_size()` per row, which since the codes change over-states
/// what a spilled row costs (a categorical spills as 4 B, not as its value).
/// Deliberately left alone: over-estimating only makes partitions smaller and
/// more numerous, which `cap_spill_partitions` already bounds, and the same
/// figure is what the per-shard refuse guard is denominated in.
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
/// Categorical columns do not need it — they spill as `Int32` codes.
fn largen_value_type(dt: &DataType) -> DataType {
    match dt {
        DataType::Utf8 => DataType::LargeUtf8,
        DataType::Binary => DataType::LargeBinary,
        other => other.clone(),
    }
}

/// Pass 0 — fold every input obs shard into a **0-row** batch carrying, per
/// categorical column, the union of every shard's *declared* values.
///
/// Runs the exported read-side pipeline pairwise —
/// [`scx_format_io::widen_metadata_batch_for_concat`] →
/// [`scx_format_io::reconcile_and_share_metadata_batches`] →
/// [`scx_format_io::concat_prepared_metadata_batches`] — which is exactly what
/// `read_obs()` computes, so the union order (first occurrence over shards in
/// index order, declared order within a shard) and the minimal key width match
/// the in-memory path's output rather than merely resembling it. Interning the
/// *declared* values is also what keeps a level no row uses.
///
/// Folding pairwise and slicing the accumulator back to zero rows keeps this
/// O(union vocabulary) rather than O(n_shards × vocabulary) — the bound is the
/// point of the path this feeds.
///
/// A shard is sliced to zero rows before folding **when every one of
/// `categorical`'s columns is already a dictionary in it**: an arrow slice of a
/// `DictionaryArray` keeps the whole values array, so the declared vocabulary
/// survives and the fold costs O(vocabulary) instead of O(shard rows). A shard
/// that stores one of those columns plain is folded at full width, because
/// `reconcile_dictionary_representations` promotes it by reading its *rows* and
/// a 0-row slice would contribute nothing. That is the shape an `append` from a
/// plain-obs source onto a dictionary base leaves.
///
/// This costs one extra pass over the obs sections (the scatter reads them
/// again); obs is the small axis next to X, and the alternative — holding every
/// shard's vocabulary at once — is the bound this path exists to avoid.
///
/// One asymmetry survives, unchanged from before: which columns are categorical
/// is decided from shard 0's schema, so a file whose *first* shard stores a
/// column plain and a later shard stores it as a dictionary still writes that
/// column plain — the deferred F3 case.
fn build_obs_vocabulary_template(
    reader: &ScxReader,
    categorical: &HashSet<String>,
) -> Result<RecordBatch> {
    let mut acc: Option<RecordBatch> = None;
    for res in reader.obs_shards() {
        let shard = res?;
        let all_dict = categorical.iter().all(|name| {
            shard
                .column_by_name(name)
                .is_some_and(|c| matches!(c.data_type(), DataType::Dictionary(_, _)))
        });
        let wide = scx_format_io::widen_metadata_batch_for_concat(&shard)?;
        let wide = if all_dict { wide.slice(0, 0) } else { wide };
        acc = Some(match acc.take() {
            None => wide,
            Some(prev) => {
                let prepared = scx_format_io::reconcile_and_share_metadata_batches(vec![
                    scx_format_io::widen_metadata_batch_for_concat(&prev)?,
                    wide,
                ])?;
                scx_format_io::concat_prepared_metadata_batches(&prepared)?.slice(0, 0)
            }
        });
    }
    let acc = acc.ok_or_else(|| {
        OpsError::InvalidInput("scx sort: obs spill requires at least one obs shard".to_string())
    })?;
    // Normalise: a single-shard input never reached a concat above, so it is
    // still `Int32`-keyed and wide. Running it through the same tail gives the
    // minimal key width and narrow value type on every path.
    let prepared = scx_format_io::reconcile_and_share_metadata_batches(vec![
        scx_format_io::widen_metadata_batch_for_concat(&acc)?,
    ])?;
    Ok(scx_format_io::concat_prepared_metadata_batches(&prepared)?.slice(0, 0))
}

/// Which obs columns a schema declares as categorical.
fn categorical_obs_columns(schema: &Schema) -> HashSet<String> {
    schema
        .fields()
        .iter()
        .filter(|f| matches!(f.data_type(), DataType::Dictionary(_, _)))
        .map(|f| f.name().clone())
        .collect()
}

/// The declared values array of a `Dictionary`-typed column, whatever its key
/// width. `None` for a non-dictionary column.
fn dictionary_values(col: &ArrayRef) -> Option<ArrayRef> {
    use arrow::array::DictionaryArray;
    use arrow::datatypes::{
        Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
    };
    macro_rules! try_key {
        ($($k:ty),*) => {$(
            if let Some(d) = col.as_any().downcast_ref::<DictionaryArray<$k>>() {
                return Some(d.values().clone());
            }
        )*};
    }
    if !matches!(col.data_type(), DataType::Dictionary(_, _)) {
        return None;
    }
    try_key!(
        Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type
    );
    None
}

/// Everything the two spill passes need to agree on: the on-spill schema, the
/// dictionary-typed output schema, which columns are categorical, and the
/// pass-0 union vocabulary they are keyed against.
///
/// One value rather than six loose ones because they are only ever correct
/// together — the codes pass 1 writes index the vocabulary pass 2 rebuilds
/// against — and because the field list all three consumers walk used to be
/// re-derived from the spill schema in three separate places.
struct ObsSpillLayout {
    /// Spill columns plus the trailing [`OBS_NEW_POS_COL`].
    spill_schema: SchemaRef,
    /// Spill columns without `new_pos` — the unit every consumer iterates.
    spill_fields: Vec<Field>,
    /// The output obs shard schema: dictionary-typed for the categoricals.
    out_schema: SchemaRef,
    /// Column names that are `Dictionary`-typed in shard 0's schema.
    categorical: HashSet<String>,
    /// The pass-0 union as an `Int32`-keyed 0-row batch — the operand each
    /// shard is remapped onto. `None` when nothing is categorical.
    template_wide: Option<RecordBatch>,
    /// Per categorical column, the union values array every output shard
    /// shares. Empty when nothing is categorical.
    vocabulary: HashMap<String, ArrayRef>,
}

/// Build the spill layout: categorical columns spill as their `Int32` codes,
/// every other string type is widened to 64-bit offsets.
///
/// `categorical` is [`categorical_obs_columns`] over shard 0's schema — the one
/// source of truth for which columns this path treats as dictionaries, shared
/// with [`build_obs_vocabulary_template`] so the fold and the layout cannot
/// disagree. Each categorical column's **output** field — type, key width and
/// field metadata, the `scx.categorical.ordered` stamp included — is taken from
/// `template`, the pass-0 union, so the emitted shards declare what the
/// in-memory path would have declared.
/// The output schema keeps `shard_schema`'s metadata (the pandas index
/// envelope); `write_obs_shard` overwrites the four per-shard stamp keys, so
/// carrying shard 0's stale values through is harmless and carrying the rest is
/// what the in-memory path does.
fn obs_spill_layout(
    shard_schema: &Schema,
    categorical: HashSet<String>,
    template: Option<RecordBatch>,
) -> Result<ObsSpillLayout> {
    let mut spill_fields: Vec<Field> = Vec::with_capacity(shard_schema.fields().len());
    let mut out_fields: Vec<Field> = Vec::with_capacity(shard_schema.fields().len());
    for f in shard_schema.fields() {
        if f.name() == OBS_NEW_POS_COL {
            return Err(OpsError::InvalidInput(format!(
                "scx sort: obs already has a column named '{OBS_NEW_POS_COL}' (reserved)"
            )));
        }
        if categorical.contains(f.name()) {
            let unified = template
                .as_ref()
                .and_then(|t| {
                    t.schema()
                        .column_with_name(f.name())
                        .map(|(_, tf)| tf.clone())
                })
                .ok_or_else(|| {
                    OpsError::InvalidInput(format!(
                        "scx sort: obs column '{}' is categorical but missing from the \
                         spill vocabulary template",
                        f.name()
                    ))
                })?;
            spill_fields.push(Field::new(f.name(), DataType::Int32, true));
            out_fields.push(unified);
        } else {
            let dt = largen_value_type(f.data_type());
            spill_fields.push(
                Field::new(f.name(), dt.clone(), f.is_nullable())
                    .with_metadata(f.metadata().clone()),
            );
            out_fields.push(
                Field::new(f.name(), dt, f.is_nullable()).with_metadata(f.metadata().clone()),
            );
        }
    }
    let out_schema = Arc::new(Schema::new_with_metadata(
        out_fields,
        shard_schema.metadata().clone(),
    ));
    let mut with_pos = spill_fields.clone();
    with_pos.push(Field::new(OBS_NEW_POS_COL, DataType::UInt64, false));
    let spill_schema = Arc::new(Schema::new(with_pos));

    let vocabulary: HashMap<String, ArrayRef> = match &template {
        Some(t) => categorical
            .iter()
            .filter_map(|name| {
                t.column_by_name(name)
                    .and_then(dictionary_values)
                    .map(|v| (name.clone(), v))
            })
            .collect(),
        None => HashMap::new(),
    };
    let template_wide = match &template {
        Some(t) => Some(scx_format_io::widen_metadata_batch_for_concat(t)?),
        None => None,
    };
    Ok(ObsSpillLayout {
        spill_schema,
        spill_fields,
        out_schema,
        categorical,
        template_wide,
        vocabulary,
    })
}

/// Align one input obs shard to the spill schema: categorical columns become
/// their `Int32` code against the pass-0 union vocabulary, everything else is
/// cast to the widened plain type. Returns the columns in `spill_fields` order,
/// without `new_pos`.
///
/// Columns are matched **by name**, not by position. The previous positional
/// match silently dropped a shard's extra trailing columns and, on a shard whose
/// columns were reordered, wrote the wrong data under the right name.
fn align_obs_shard_for_spill(
    batch: &RecordBatch,
    layout: &ObsSpillLayout,
) -> Result<Vec<ArrayRef>> {
    use arrow::array::DictionaryArray;
    use arrow::datatypes::Int32Type;

    let categorical = &layout.categorical;
    let wide = scx_format_io::widen_metadata_batch_for_concat(batch)?;
    let aligned = match layout.template_wide.as_ref() {
        Some(t) if !categorical.is_empty() => {
            let prepared =
                scx_format_io::reconcile_and_share_metadata_batches(vec![t.clone(), wide])?;
            // The template was folded from these same shards, so remapping a
            // shard onto it cannot introduce a value the union lacks. If it
            // did, the codes below would index past the values array every
            // other output shard shares — say so rather than write it.
            for name in categorical {
                let before = t
                    .column_by_name(name)
                    .and_then(dictionary_values)
                    .map(|v| v.len());
                let after = prepared[0]
                    .column_by_name(name)
                    .and_then(dictionary_values)
                    .map(|v| v.len());
                if before != after {
                    return Err(OpsError::InvalidInput(format!(
                        "scx sort: obs column '{name}' gained categories while spilling \
                         ({before:?} -> {after:?}); the file changed under the sort"
                    )));
                }
            }
            prepared.into_iter().nth(1).expect("two prepared batches")
        }
        _ => wide,
    };

    let mut cols = Vec::with_capacity(layout.spill_fields.len());
    for f in &layout.spill_fields {
        let c = aligned.column_by_name(f.name()).ok_or_else(|| {
            OpsError::InvalidInput(format!(
                "scx sort: obs shard is missing column '{}'",
                f.name()
            ))
        })?;
        if categorical.contains(f.name()) {
            let d = c
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| {
                    OpsError::InvalidInput(format!(
                        "scx sort: obs column '{}' should be Dictionary(Int32, _) after \
                         reconciliation, found {:?}",
                        f.name(),
                        c.data_type()
                    ))
                })?;
            cols.push(Arc::new(d.keys().clone()) as ArrayRef);
        } else if c.data_type() == f.data_type() {
            cols.push(c.clone());
        } else {
            cols.push(arrow::compute::cast(c, f.data_type())?);
        }
    }
    Ok(cols)
}

/// Pass 1 — scatter every live obs row to its `new_pos`-range partition spill
/// file (Arrow IPC stream, uniform schema). `p` rows per partition,
/// `n_parts` partitions.
fn scatter_obs_to_spill(
    reader: &ScxReader,
    new_pos_of_old: &[i64],
    p: usize,
    n_parts: usize,
    layout: &ObsSpillLayout,
    spill: &SpillDir,
) -> Result<()> {
    let spill_schema = &layout.spill_schema;
    // Opens `n_parts` spill files at once. The caller caps `n_parts` via
    // `cap_spill_partitions` (`MAX_SPILL_PARTITIONS` in `sort.rs`) so this stays
    // well under a typical `ulimit -n` (1024+); the X external path uses the
    // same bound.
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
        let aligned_cols = align_obs_shard_for_spill(&batch, layout)?;
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
            let mut sub: Vec<ArrayRef> = aligned_cols
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

/// Pass 2 — yields `shard_target`-sized output obs shards (categorical columns
/// rebuilt from their codes) in global `new_pos` order, reading one partition
/// spill at a time. Drives both the obs-shard write and (re-constructed) the
/// predicate index rebuild. Items are `EngineError`-typed so it composes with
/// [`rebuild_obs_predicate_index_streaming`].
struct ObsScatterReader {
    partition_files: Vec<PathBuf>,
    next_part: usize,
    pending: Option<RecordBatch>, // globally-ordered spilled rows not yet emitted
    spill_schema: SchemaRef,      // codes/plain + new_pos (the on-spill schema)
    coded_schema: SchemaRef,      // codes/plain, no new_pos (pending / concat schema)
    out_schema: SchemaRef,        // dictionary-typed output schema
    /// Per categorical column, the union vocabulary every output shard shares.
    vocabulary: HashMap<String, ArrayRef>,
    shard_target: usize,
    emitted: u64,
}

impl ObsScatterReader {
    fn new(partition_files: Vec<PathBuf>, layout: &ObsSpillLayout, shard_target: u32) -> Self {
        Self {
            partition_files,
            next_part: 0,
            pending: None,
            coded_schema: Arc::new(Schema::new(layout.spill_fields.clone())),
            spill_schema: layout.spill_schema.clone(),
            out_schema: layout.out_schema.clone(),
            vocabulary: layout.vocabulary.clone(),
            shard_target: shard_target.max(1) as usize,
            emitted: 0,
        }
    }

    /// Read one partition spill, sort it by `new_pos`, and return its rows
    /// without the `new_pos` column. `Ok(None)` for an empty partition.
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
        Ok(Some(RecordBatch::try_new(self.coded_schema.clone(), cols)?))
    }

    /// Rebuild the categorical columns of a spilled shard from their `Int32`
    /// codes and the shared union vocabulary, producing the output obs shard.
    ///
    /// Nothing is packed: the values array is the pass-0 union, `Arc`-cloned, so
    /// every output shard declares the same levels in the same order — unused
    /// ones included — and the written bytes match the in-memory path's, whose
    /// `RecordBatch::slice` shares one values array the same way.
    fn reencode(
        &self,
        coded: &RecordBatch,
    ) -> std::result::Result<RecordBatch, scx_engine::EngineError> {
        let mut cols = Vec::with_capacity(coded.num_columns());
        for (i, f) in self.out_schema.fields().iter().enumerate() {
            let c = coded.column(i);
            match (f.data_type(), self.vocabulary.get(f.name())) {
                (DataType::Dictionary(key_type, _), Some(values)) => {
                    cols.push(build_dictionary(c, key_type, values.clone())?)
                }
                _ => cols.push(c.clone()),
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
                        arrow::compute::concat_batches(&self.coded_schema, &[prev, batch])?
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

/// Assemble a `DictionaryArray` of the requested key width from `Int32` codes
/// and a values array, without repacking either.
fn build_dictionary(
    codes: &ArrayRef,
    key_type: &DataType,
    values: ArrayRef,
) -> std::result::Result<ArrayRef, arrow::error::ArrowError> {
    use arrow::array::{DictionaryArray, Int16Array, Int32Array, Int8Array};
    use arrow::datatypes::{Int16Type, Int32Type, Int8Type};

    macro_rules! build {
        ($dt:expr, $arr:ty, $key:ty) => {{
            let narrowed = arrow::compute::cast(codes, $dt)?;
            let keys = narrowed
                .as_any()
                .downcast_ref::<$arr>()
                .ok_or_else(|| {
                    arrow::error::ArrowError::CastError(format!(
                        "obs spill: dictionary keys did not cast to {:?}",
                        $dt
                    ))
                })?
                .clone();
            Ok(Arc::new(DictionaryArray::<$key>::try_new(keys, values)?) as ArrayRef)
        }};
    }
    match key_type {
        DataType::Int8 => build!(&DataType::Int8, Int8Array, Int8Type),
        DataType::Int16 => build!(&DataType::Int16, Int16Array, Int16Type),
        DataType::Int32 => build!(&DataType::Int32, Int32Array, Int32Type),
        other => Err(arrow::error::ArrowError::NotYetImplemented(format!(
            "obs spill: unsupported dictionary key type {other:?}"
        ))),
    }
}

/// Holds the obs spill for the duration of the sort (SpillDir RAII removes it
/// on drop) plus the parameters to construct an [`ObsScatterReader`] — once for
/// the write pass, once for the predicate-index rebuild.
struct ObsSpillState {
    _spill: SpillDir,
    partition_files: Vec<PathBuf>,
    layout: ObsSpillLayout,
    shard_target: u32,
    n_parts: usize,
}

impl ObsSpillState {
    fn reader(&self) -> ObsScatterReader {
        ObsScatterReader::new(
            self.partition_files.clone(),
            &self.layout,
            self.shard_target,
        )
    }
}

/// Build the obs spill: fold the declared-value union, size partitions from the
/// budget, scatter all live obs rows to per-partition spill files, and return
/// the state needed to emit and index the sorted output shards. Caller must have
/// established `obs_spill` (single-modality, sharded obs, budget set).
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

    // Cap simultaneously-open spill files (EMFILE), widening partitions to fit,
    // and re-validate that a widened partition — which pass 2 loads whole into
    // RAM — still fits the budget (the per-shard guard above bounds one shard).
    let p0 = obs_partition_rows(budget, bytes_per_row, opts.shard_target_rows);
    let (p, n_parts) = cap_spill_partitions(p0, n_live, bytes_per_row, Some(budget))?;

    // Pass 0 — only when shard 0 declares a categorical; a plain-obs file pays
    // no extra read of the obs sections.
    let categorical = categorical_obs_columns(&s0.schema());
    let template = if categorical.is_empty() {
        None
    } else {
        Some(build_obs_vocabulary_template(reader, &categorical)?)
    };
    let layout = obs_spill_layout(&s0.schema(), categorical, template)?;

    let spill = SpillDir::create(opts.temp_dir.as_deref())?;
    scatter_obs_to_spill(reader, new_pos_of_old, p, n_parts, &layout, &spill)?;
    let partition_files = (0..n_parts).map(|i| spill.partition_file(i)).collect();
    log::info!("scx sort: obs spill-scatter across {n_parts} partitions (~{p} rows each)");

    Ok(ObsSpillState {
        _spill: spill,
        partition_files,
        layout,
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

// ---------------------------------------------------------------------------
// F1 — grouped-sharding helpers
// ---------------------------------------------------------------------------

/// Build the grouping provenance payload for a grouped sort (`None` when
/// `--group-by` was not used).
fn grouping_provenance(opts: &SortOptions) -> Option<serde_json::Value> {
    let group_by = opts.group_by.as_ref()?;
    let reference = match &opts.reference {
        Some(crate::sort::ReferenceSpec::Labels(l)) => serde_json::json!({ "labels": l }),
        Some(crate::sort::ReferenceSpec::Column(c)) => serde_json::json!({ "column": c }),
        None => serde_json::Value::Null,
    };
    Some(serde_json::json!({
        "group_by": group_by,
        "reference": reference,
        "group_target_bytes": opts.group_target_bytes,
        "group_max_bytes": opts.group_max_bytes,
    }))
}

/// Build the per-(live)row reference mask for a grouped sort. `live_keys` holds
/// the projected sort-key columns (group_by leading) plus, for
/// [`ReferenceSpec::Column`], the boolean reference column. Returns `None` when
/// no reference is configured (clustering only, no reference shard).
fn build_reference_mask(
    live_keys: &RecordBatch,
    group_col: &str,
    reference: &crate::sort::ReferenceSpec,
) -> Result<Vec<bool>> {
    use crate::sort::ReferenceSpec;
    let n = live_keys.num_rows();
    match reference {
        ReferenceSpec::Labels(set) => {
            use arrow::array::LargeStringArray;
            let col = live_keys.column_by_name(group_col).ok_or_else(|| {
                OpsError::InvalidInput(format!("--group-by column '{group_col}' missing from obs"))
            })?;
            let utf8 = arrow::compute::cast(col, &DataType::LargeUtf8)?;
            let arr = utf8
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    OpsError::InvalidInput(format!(
                        "--group-by column '{group_col}' is not categorical/string-like"
                    ))
                })?;
            let want: std::collections::HashSet<&str> = set.iter().map(String::as_str).collect();
            Ok((0..n)
                .map(|i| !arr.is_null(i) && want.contains(arr.value(i)))
                .collect())
        }
        ReferenceSpec::Column(col_name) => {
            use arrow::array::BooleanArray;
            let col = live_keys.column_by_name(col_name).ok_or_else(|| {
                OpsError::InvalidInput(format!("--reference column '{col_name}' missing from obs"))
            })?;
            let casted = arrow::compute::cast(col, &DataType::Boolean)?;
            let arr = casted
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    OpsError::InvalidInput(format!(
                        "--reference column '{col_name}' is not boolean-castable"
                    ))
                })?;
            Ok((0..n).map(|i| !arr.is_null(i) && arr.value(i)).collect())
        }
    }
}

/// Inject the synthetic reference-first key `__scx_ref__ = !is_reference` into
/// `live_keys` and return the extended batch + the extractor's `by` list
/// (`["__scx_ref__", <opts.by...>]`). Reference rows get `false`, sorting first
/// under ascending order.
fn inject_reference_first_key(
    live_keys: &RecordBatch,
    by: &[String],
    ref_mask: &[bool],
) -> Result<(RecordBatch, Vec<String>)> {
    use arrow::array::BooleanArray;
    let synth = BooleanArray::from(ref_mask.iter().map(|&r| !r).collect::<Vec<bool>>());
    let mut fields: Vec<Field> = live_keys
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    fields.insert(0, Field::new("__scx_ref__", DataType::Boolean, false));
    let mut columns: Vec<ArrayRef> = live_keys.columns().to_vec();
    columns.insert(0, std::sync::Arc::new(synth));
    let schema = std::sync::Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(schema, columns)?;
    let mut extractor_by = Vec::with_capacity(by.len() + 1);
    extractor_by.push("__scx_ref__".to_string());
    extractor_by.extend(by.iter().cloned());
    Ok((batch, extractor_by))
}

/// Map a local obs-row index -> category index (-1 = null), using the sorted
/// `cats` order. Local analogue of [`category_of_old`] for callers that work
/// purely in the obs-batch row space (no global old-id remap), e.g.
/// [`compute_grouped_order`].
fn category_of_local(obs: &RecordBatch, name: &str, cats: &[String]) -> Result<Vec<i32>> {
    use arrow::array::LargeStringArray;
    let col = obs.column_by_name(name).ok_or_else(|| {
        OpsError::InvalidInput(format!("sort key column '{name}' missing from obs"))
    })?;
    // `LargeUtf8` (i64 offsets): the key can span the full obs, so a narrow
    // `Utf8` decode overflows i32 offsets at >2 GB of total string bytes.
    let utf8 = arrow::compute::cast(col, &DataType::LargeUtf8)?;
    let arr = utf8
        .as_any()
        .downcast_ref::<LargeStringArray>()
        .ok_or_else(|| {
            OpsError::InvalidInput(format!("sort key '{name}' is not categorical/string-like"))
        })?;
    let index: std::collections::HashMap<&str, i32> = cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as i32))
        .collect();
    let mut out = vec![-1i32; arr.len()];
    for (i, slot) in out.iter_mut().enumerate() {
        if arr.is_null(i) {
            continue;
        }
        if let Some(&ci) = index.get(arr.value(i)) {
            *slot = ci;
        }
    }
    Ok(out)
}

/// The grouped emission order + per-row group/reference metadata produced by
/// [`compute_grouped_order`]. Shared by `scx sort` (F1) and `scx convert
/// --group-by` so both produce a byte-identical grouped layout.
pub struct GroupedOrder {
    /// `perm[emission_row] = obs_row` — the reference-first / group-by order
    /// (indices into the obs `RecordBatch` passed to `compute_grouped_order`).
    pub perm: Vec<u64>,
    /// Per-emission-row group id (index into `labels`).
    pub group_of_new: Vec<i32>,
    /// Per-emission-row reference flag.
    pub ref_of_new: Vec<bool>,
    /// Label table; `labels[group_of_new[e]]` is row `e`'s group label. A
    /// trailing `"__ungrouped__"` is appended iff some row's `group_by` is null.
    pub labels: Vec<String>,
    /// Distinct reference labels for the sidecar `reference_labels` field.
    pub reference_labels: Vec<String>,
}

/// Compute the grouped (reference-first, then `group_by` label, then
/// `secondary_by`) emission order over an obs `RecordBatch`, plus the per-row
/// group id / reference flag, label table, and reference labels.
///
/// Operates purely on the obs batch (the caller must have already filtered out
/// deleted rows and projected the sort-key columns + any reference column), so
/// it is reusable by both the standalone sort engine and convert-time grouping.
/// `per_row_nnz` is intentionally *not* computed here — the caller pairs the
/// result with [`crate::group_plan::plan_group_shards`], supplying per-row nnz
/// from its own source (SCX shards for sort, the h5ad indptr for convert).
pub fn compute_grouped_order(
    obs: &RecordBatch,
    group_by: &str,
    secondary_by: &[String],
    reference: Option<&crate::sort::ReferenceSpec>,
) -> Result<GroupedOrder> {
    // Reference mask in local obs-row order (None => clustering only).
    let ref_mask: Option<Vec<bool>> = match reference {
        Some(r) => Some(build_reference_mask(obs, group_by, r)?),
        None => None,
    };
    // Sort keys: `group_by` leading, then any secondary keys.
    let mut by = Vec::with_capacity(secondary_by.len() + 1);
    by.push(group_by.to_string());
    by.extend(secondary_by.iter().cloned());
    // Inject the synthetic reference-first key when a reference is configured.
    // Ascending order is forced (reference rows sort first); `--reverse` has no
    // meaning in grouped mode.
    let (keys, extractor_by) = match &ref_mask {
        Some(mask) => inject_reference_first_key(obs, &by, mask)?,
        None => (obs.clone(), by.clone()),
    };
    let extractor = SortKeyExtractor::new(&keys.schema(), &extractor_by, false)?;
    let rows = extractor.rows(&keys)?;
    let perm = stable_argsort(&rows, 0);
    drop(rows);

    // Group ids in emission order.
    let cats = distinct_categories(obs, group_by, false)?;
    let cat_of_local = category_of_local(obs, group_by, &cats)?;
    let mut labels = cats;
    let ungrouped_idx = labels.len() as i32;
    let mut used_ungrouped = false;
    let group_of_new: Vec<i32> = perm
        .iter()
        .map(|&l| {
            let c = cat_of_local[l as usize];
            if c < 0 {
                used_ungrouped = true;
                ungrouped_idx
            } else {
                c
            }
        })
        .collect();
    if used_ungrouped {
        labels.push("__ungrouped__".to_string());
    }

    // Reference flags in emission order.
    let ref_of_new: Vec<bool> = match &ref_mask {
        Some(mask) => perm.iter().map(|&l| mask[l as usize]).collect(),
        None => vec![false; perm.len()],
    };

    // A reference spec that matches zero rows is almost always a typo (wrong
    // label or a non-boolean/absent column): the archive ends up clustered-only
    // with no reference shard and no error. Warn so the mistake is visible.
    if reference.is_some() && !ref_of_new.iter().any(|&r| r) {
        log::warn!(
            "scx sort: --reference matched no rows; the output will be grouped but have no \
             reference shard (check the label set or the boolean column name)"
        );
    }

    // Reference labels for the sidecar: the explicit set for `Labels`, else the
    // distinct group labels that ended up reference (covers `Column`). Matches
    // the post-plan derivation the sort engine historically used — a reference
    // record exists for exactly these labels, since groups are never split.
    let reference_labels: Vec<String> = match reference {
        Some(crate::sort::ReferenceSpec::Labels(set)) => set.clone(),
        Some(crate::sort::ReferenceSpec::Column(_)) => {
            let mut seen = std::collections::BTreeSet::new();
            for e in 0..group_of_new.len() {
                if ref_of_new[e] {
                    seen.insert(labels[group_of_new[e] as usize].clone());
                }
            }
            seen.into_iter().collect()
        }
        None => Vec::new(),
    };

    Ok(GroupedOrder {
        perm,
        group_of_new,
        ref_of_new,
        labels,
        reference_labels,
    })
}

/// F1.3b — per-row nnz in emission order via an indptr-only pre-scan. For each
/// X CSR shard, decode only its indptr (cheap: ~8 B/row), fill a global
/// old-row nnz array, then reorder by `order_old`. Used by the byte-budget
/// group planner.
fn prescan_per_row_nnz(reader: &ScxReader, order_old: &[u64], n_obs: usize) -> Result<Vec<u64>> {
    let catalog_version = reader.catalog().catalog_version;
    let mut nnz_old = vec![0u64; n_obs];
    for entry in reader.catalog().shards_sorted() {
        let row_start = entry.stats.as_ref().map(|s| s.row_start).ok_or_else(|| {
            OpsError::InvalidInput(format!(
                "scx sort: X shard '{}' has no stats for the group nnz pre-scan",
                entry.name
            ))
        })?;
        let section = reader.section_bytes(entry)?;
        let indptr = scx_format_io::decode_shard_indptr_bytes(section, entry, catalog_version)?;
        for l in 0..indptr.len().saturating_sub(1) {
            let global = row_start as usize + l;
            if global < n_obs {
                nnz_old[global] = (indptr[l + 1] - indptr[l]) as u64;
            }
        }
    }
    Ok(order_old.iter().map(|&old| nnz_old[old as usize]).collect())
}

/// Largest decoded-RAM footprint the grouped emitter must buffer for any single
/// output shard, plus the shard index and the label of its dominant group (for
/// the error message). A group is never split, so the emitter accumulates every
/// row of a shard before flushing — an oversized group's whole footprint is
/// resident at once (M1).
///
/// Footprint uses the same ~16 B/nnz (+8 B/row indptr) accounting as the
/// non-grouped budget guard (`partition_target_rows` / the per-shard guard) and
/// convert's `per_worker_bytes`, so all budget checks reason in one currency.
/// `per_row_nnz` (emission order, indexed by global output row) is supplied in
/// byte-budget mode; when empty (row-count mode) the per-row nnz is estimated
/// from `n_vars * density`. Returns `(0, 0, "")` for an empty plan.
fn max_grouped_shard_footprint(
    plan: &crate::group_plan::GroupPlan,
    per_row_nnz: &[u64],
    n_vars: usize,
    density: f64,
) -> (u64, u32, String) {
    // Per-row decoded footprint estimate for row-count mode (matches
    // `partition_target_rows`'s 16 B/nnz assumption).
    let per_row_est = (n_vars as f64 * density * 16.0).max(1.0).ceil() as u64;
    let byte_mode = !per_row_nnz.is_empty();

    // Accumulate footprint per shard, tracking the dominant (largest) record's
    // label within each shard so the message can name the culprit group.
    let mut per_shard: std::collections::HashMap<u32, (u64, u64, String)> =
        std::collections::HashMap::new();
    for r in &plan.records {
        let rows = r.row_stop - r.row_start;
        let footprint = if byte_mode {
            let nnz: u64 = per_row_nnz[r.row_start as usize..r.row_stop as usize]
                .iter()
                .sum();
            nnz.saturating_mul(16)
                .saturating_add(rows.saturating_mul(8))
        } else {
            rows.saturating_mul(per_row_est)
        };
        let e = per_shard.entry(r.shard).or_insert((0, 0, String::new()));
        e.0 = e.0.saturating_add(footprint);
        if footprint >= e.1 {
            e.1 = footprint;
            e.2 = r.label.clone();
        }
    }

    per_shard
        .into_iter()
        .map(|(shard, (total, _dom_bytes, label))| (total, shard, label))
        .max_by_key(|&(total, _, _)| total)
        .unwrap_or((0, 0, String::new()))
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

/// `(cross_row_coded_shards, total_x_shards)` — how much of X is stored under a
/// codec whose compression spans rows.
///
/// `Scx1` codes each row's gene indices independently of row order, so a
/// permutation just relocates identically-sized per-row blocks (measured
/// size-neutral on `tabula_sapiens_100k`); `None` is incompressible by
/// definition. Everything else — `Zstd`, `ShufDeltaZstd`, `Lz4Shuffle`,
/// `Pcodec` — compresses the shard's byte stream as a whole, so *which* rows
/// share a shard changes the output size. `Pcodec` counts here because it
/// models the value sequence, and a permutation reshuffles that sequence even
/// though its index arrays go through zstd.
///
/// Reads one shard header per shard; on the local mmap reader `section_bytes`
/// is a zero-copy slice, the same access `widest_value_encoding` already makes
/// for every shard on this path.
fn cross_row_coded_shard_counts(reader: &ScxReader) -> Result<(usize, usize, Option<CodecId>)> {
    use std::collections::HashMap;

    let shards = reader.catalog().shards_sorted();
    let mut cross_row = 0usize;
    let mut histogram: HashMap<u8, usize> = HashMap::new();
    for entry in &shards {
        let section = reader.section_bytes(entry)?;
        if section.len() < SHARD_HEADER_SIZE {
            continue;
        }
        let sh = ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
        *histogram.entry(sh.codec_id).or_insert(0) += 1;
        match CodecId::from_u8(sh.codec_id) {
            Some(CodecId::None) | Some(CodecId::Scx1) => {}
            _ => cross_row += 1,
        }
    }
    // Most common codec, so the warning can name the pin that preserves size.
    let dominant = histogram
        .iter()
        .max_by_key(|(_, n)| **n)
        .and_then(|(id, _)| CodecId::from_u8(*id));
    Ok((cross_row, shards.len(), dominant))
}

/// The `--codec` spelling for a [`CodecId`], so a warning can name a flag value
/// the user can actually paste. Mirrors `CodecId::parse_cli`'s accepted names.
fn codec_cli_name(c: CodecId) -> &'static str {
    match c {
        CodecId::None => "none",
        CodecId::Scx1 => "scx1",
        CodecId::Zstd => "zstd",
        CodecId::Lz4Shuffle => "lz4",
        CodecId::Pcodec => "pcodec",
        CodecId::ShufDeltaZstd => "shufdelta",
    }
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
///
/// The float/integer *kind* must be probed across EVERY shard, not just the
/// first (SCX-004): a `Uint8` first shard followed by a `Float32` shard would
/// otherwise pick an integer encoding and truncate the float shard's
/// fractional values (e.g. `1.5 → 1`).
fn widest_value_encoding(
    reader: &ScxReader,
    shards: &[&FullCatalogEntry],
) -> Result<ValueEncoding> {
    if shards.is_empty() {
        return Ok(ValueEncoding::Uint8);
    }
    for entry in shards {
        let enc = entry_value_encoding(reader, Some(*entry))?;
        if matches!(enc, ValueEncoding::Float32 | ValueEncoding::Float16) {
            return Ok(ValueEncoding::Float32);
        }
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
