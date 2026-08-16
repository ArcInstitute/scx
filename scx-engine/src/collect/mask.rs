//! Per-shard masking: the row-set fast path, the legacy full-decode
//! fallback, and [`compute_mask`], the fork between them.
//!
//! This is the crate's most correctness-sensitive branch. Both arms must
//! produce the same keep masks; `tests/rowset_differential.rs` is the oracle
//! that holds them to it, and `QueryPipeline::rowset_pushdown(false)` is how a
//! caller pins the legacy arm.

use arrow::array::{Array, RecordBatch};

use crate::error::Result;
use crate::index::index_covers_all_obs;
use crate::pipeline::QueryPipeline;
use crate::predicate::{eval_rowset, evaluate, Predicate, RowSetCtx};
use crate::reader::SectionReader;
use crate::rowset::{RowRange, RowSet};

use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::{DeletionVectors, SectionType};

use super::plan::ExecutionPlan;
use super::retry::par_map_with_shard_retry;

/// Per-shard row range + local keep mask produced by `plan_and_mask`.
pub(crate) struct ShardInfo {
    pub(crate) shard_idx: usize,
    pub(crate) row_start: u64,
    #[allow(dead_code)] // retained for debugging and future use
    pub(crate) row_end: u64,
    pub(crate) local_keep_mask: Vec<bool>,
}

/// Obs metadata shard ranges from the catalog: `(shard_idx, row_start,
/// row_end)`, sorted by `shard_idx`. Returns `None` if any obs shard entry
/// lacks row-range stats (files written before stats were stamped on
/// metadata shards) — the caller then falls back to streaming every shard.
pub(crate) fn obs_shard_ranges_from_catalog(
    catalog: &scx_format_io::FullCatalog,
) -> Option<Vec<(u32, u64, u64)>> {
    let mut ranges: Vec<(u32, u64, u64)> = Vec::new();
    for e in &catalog.entries {
        if e.section_type != SectionType::ObsMetadataShard {
            continue;
        }
        let idx: u32 = match e
            .name
            .strip_prefix("obs_metadata/shard_")
            .and_then(|s| s.parse().ok())
        {
            Some(i) => i,
            None => continue,
        };
        // Any obs shard without stats -> bail to the full-stream fallback.
        let stats = e.stats.as_ref()?;
        ranges.push((idx, stats.row_start, stats.row_end));
    }
    ranges.sort_by_key(|(idx, _, _)| *idx);
    Some(ranges)
}

/// Read `row_start` from a per-shard obs batch's stamped schema metadata.
/// `n_rows` is taken from the batch itself (always exact).
fn obs_shard_row_start(batch: &RecordBatch) -> Result<u64> {
    batch
        .schema_ref()
        .metadata()
        .get("row_start")
        .and_then(|s| s.parse::<u64>().ok())
        .ok_or_else(|| {
            crate::error::EngineError::Generic(
                "obs metadata shard missing stamped `row_start` schema metadata".to_string(),
            )
        })
}

/// The per-shard masking output of [`compute_mask`]: everything `materialize`
/// needs to decode X and filter obs/var, computed without decoding any X shard.
pub(crate) struct MaskResult {
    /// Full obs batch — populated ONLY on the legacy single-section path.
    pub(crate) legacy_obs: Option<RecordBatch>,
    pub(crate) obs_shard_ranges: Vec<(u32, u64, u64)>,
    pub(crate) shard_infos: Vec<ShardInfo>,
    /// Pre-limit Level-2 match count (Σ keep-mask trues over ALL matching
    /// candidate shards). Used by `count()`; equals `csr.n_rows()` after an
    /// unlimited materialise. NOT reduced when `shard_infos` is capped for a
    /// limited query (the row-set path keeps it exact and full).
    pub(crate) matched_rows: usize,
}

/// Build the per-shard keep masks. Prefers the **row-set fast path** — resolve
/// indexed obs predicates straight from the predicate index, decoding no obs
/// shard for the indexed part — and falls back to the legacy full-decode path
/// for non-indexed / residual-only predicates, legacy single-section obs, a
/// stale index, or when the pushdown kill-switch is set.
pub(crate) fn compute_mask(
    pipeline: &QueryPipeline,
    plan: &ExecutionPlan,
    sorted_shards: &[&FullCatalogEntry],
    n_obs: usize,
) -> Result<MaskResult> {
    if let Some(m) = try_rowset_mask(pipeline, plan, sorted_shards, n_obs)? {
        return Ok(m);
    }
    build_legacy_mask(pipeline, plan, sorted_shards, n_obs)
}

/// Flatten the implicit top-level AND across an obs-predicate list (multiple
/// `filter_obs` calls are AND-ed) and across nested [`Predicate::And`] nodes,
/// collecting the leaf conjuncts.
fn flatten_and<'a>(pred: &'a Predicate, out: &mut Vec<&'a Predicate>) {
    match pred {
        Predicate::And(a, b) => {
            flatten_and(a, out);
            flatten_and(b, out);
        }
        other => out.push(other),
    }
}

/// Partition the AND-combined obs predicates into an exactly-indexed row-set and
/// a residual subtree. Conjuncts resolvable from the index (`eval_rowset`
/// `Some`) are intersected into the row-set; the rest become residual.
/// Returns `(None, ...)` when nothing resolves — the caller then uses the
/// legacy full-decode path, unchanged (no regression).
fn partition_obs_predicates(
    preds: &[Predicate],
    ctx: &RowSetCtx,
) -> (Option<RowSet>, Vec<Predicate>) {
    let mut conjuncts: Vec<&Predicate> = Vec::new();
    for p in preds {
        flatten_and(p, &mut conjuncts);
    }
    let mut indexed: Option<RowSet> = None;
    let mut residual: Vec<Predicate> = Vec::new();
    for c in conjuncts {
        match eval_rowset(c, ctx) {
            Some(rs) => {
                indexed = Some(match indexed {
                    None => rs,
                    Some(prev) => prev.intersect(&rs),
                });
            }
            None => residual.push(c.clone()),
        }
    }
    (indexed, residual)
}

/// The CSR/output shard row-range table the predicate index was built against:
/// `(shard_id, row_start, row_end)` where `shard_id` is the shard's position in
/// `shards_sorted()` (== its index in `output_shard_row_ranges` at build time).
/// Returns `None` if any shard lacks row-range stats — the index then can't be
/// mapped to global rows safely, so the caller falls back to the legacy path.
///
/// ⚠️ **The position `i` here equals the index's `shard_id` only at
/// `modality_id == 0`.** `sorted_shards` comes from [`super::plan::scan_shards`], which is
/// modality-scoped; on a multimodal file each modality's shards independently
/// tile `[0, n_obs)`, so modality 2's third shard is also position 2 and would
/// silently adopt modality 1's row-set.
///
/// What prevents that is [`super::plan::build_plan`], which forces `obs_predicate_index` to
/// `None` for every `modality_id != 0` pipeline — so this function is not
/// reachable on a modality-scoped query, whatever the file contains. That guard
/// is the load-bearing one; index absence is **not**. See
/// [`crate::index::derive_shard_column_stats`] for the write-side half and for
/// why "multimodal files carry no index" is a convention of the high-level
/// writers rather than an enforced invariant.
fn csr_shard_ranges_table(sorted_shards: &[&FullCatalogEntry]) -> Option<Vec<(u32, u64, u64)>> {
    let mut table = Vec::with_capacity(sorted_shards.len());
    for (i, entry) in sorted_shards.iter().enumerate() {
        let stats = entry.stats.as_ref()?;
        table.push((i as u32, stats.row_start, stats.row_end));
    }
    Some(table)
}

/// Global-row coverage of the surviving candidate shards (post Level-1 catalog
/// pruning). Intersecting the indexed row-set with this drops rows in shards
/// proven non-matching, keeping `matched_rows` exact.
fn candidate_shard_coverage(plan: &ExecutionPlan, sorted_shards: &[&FullCatalogEntry]) -> RowSet {
    let ranges: Vec<RowRange> = plan
        .candidate_shards
        .iter()
        .filter_map(|sc| {
            sorted_shards[sc.shard_idx]
                .stats
                .as_ref()
                .map(|s| RowRange {
                    start: s.row_start,
                    end: s.row_end,
                })
        })
        .collect();
    RowSet::from_ranges(ranges)
}

/// The set of globally-deleted obs rows as a [`RowSet`], read directly from
/// the v2 global deletion bitmap (`deletions[0]`).
fn deletion_rowset(dv: &DeletionVectors, n_obs: u64) -> RowSet {
    let mut ranges: Vec<RowRange> = Vec::new();
    if let Some(bitmap) = dv.global_deleted() {
        // Bitmap iterates ascending; coalesce consecutive deleted rows.
        let mut run: Option<(u64, u64)> = None;
        for row in bitmap.iter() {
            let g = row as u64;
            if g >= n_obs {
                break;
            }
            match run {
                Some((s, e)) if g == e => run = Some((s, g + 1)),
                Some((s, e)) => {
                    ranges.push(RowRange { start: s, end: e });
                    run = Some((g, g + 1));
                }
                None => run = Some((g, g + 1)),
            }
        }
        if let Some((s, e)) = run {
            ranges.push(RowRange { start: s, end: e });
        }
    }
    RowSet::from_ranges(ranges)
}

/// Coalesce a per-shard boolean keep mask into global [`RowRange`]s.
fn mask_to_ranges(mask: &[bool], row_start: u64) -> Vec<RowRange> {
    let mut out = Vec::new();
    let mut start: Option<u64> = None;
    for (i, &keep) in mask.iter().enumerate() {
        if keep {
            if start.is_none() {
                start = Some(row_start + i as u64);
            }
        } else if let Some(s) = start.take() {
            out.push(RowRange {
                start: s,
                end: row_start + i as u64,
            });
        }
    }
    if let Some(s) = start {
        out.push(RowRange {
            start: s,
            end: row_start + mask.len() as u64,
        });
    }
    out
}

/// The obs metadata shards spanned by `rs`, paired with their `row_start`.
/// `obs_shard_ranges` is sorted by `shard_idx` (== ascending `row_start`), so
/// each rs range is located with a binary search and a short forward walk.
fn obs_shards_spanned(rs: &RowSet, obs_shard_ranges: &[(u32, u64, u64)]) -> Vec<(u32, u64)> {
    let mut out: Vec<(u32, u64)> = Vec::new();
    let mut last: Option<usize> = None;
    for r in rs.ranges() {
        let mut i = obs_shard_ranges.partition_point(|(_, _, rend)| *rend <= r.start);
        while i < obs_shard_ranges.len() {
            let (idx, rstart, _) = obs_shard_ranges[i];
            if rstart >= r.end {
                break;
            }
            if last != Some(i) {
                out.push((idx, rstart));
                last = Some(i);
            }
            i += 1;
        }
    }
    out
}

/// Apply the residual (non-indexed) predicates by decoding ONLY the obs shards
/// that `rs` spans, evaluating the residual over them, and intersecting the
/// result back into `rs`. No obs shard outside `rs`'s span is read.
fn apply_residual(
    reader: &dyn SectionReader,
    residual: &[Predicate],
    rs: &RowSet,
    obs_shard_ranges: &[(u32, u64, u64)],
) -> Result<RowSet> {
    let needed = obs_shards_spanned(rs, obs_shard_ranges);
    let per_shard: Vec<Vec<RowRange>> = par_map_with_shard_retry(&needed, |&(idx, row_start)| {
        let batch = reader.read_obs_shard(idx)?;
        let mut local = vec![true; batch.num_rows()];
        for pred in residual {
            let arr = evaluate(pred, &batch)?;
            for (i, m) in local.iter_mut().enumerate() {
                if *m {
                    // `evaluate` has already coalesced any surviving UNKNOWN to
                    // false, so `is_valid` is belt-and-braces, not the null
                    // policy — that lives in `evaluate` (three-valued Kleene,
                    // UNKNOWN → not matched only at the top level).
                    *m = arr.is_valid(i) && arr.value(i);
                }
            }
        }
        Ok(mask_to_ranges(&local, row_start))
    })?;
    let mut all: Vec<RowRange> = Vec::new();
    for ranges in per_shard {
        all.extend(ranges);
    }
    Ok(rs.intersect(&RowSet::from_ranges(all)))
}

/// Build per-shard keep masks from the (final, exact) row-set, iterating
/// candidate shards ascending. With a `limit` the walk stops once the
/// cumulative kept count reaches it, so only the first K candidate shards get
/// masks — the `limit(N)` short-circuit. Uses a forward cursor over the sorted
/// row-set so the whole pass is O(candidate shards + row-set ranges).
fn shard_infos_from_rowset(
    plan: &ExecutionPlan,
    sorted_shards: &[&FullCatalogEntry],
    rs: &RowSet,
    limit: Option<usize>,
) -> Vec<ShardInfo> {
    let ranges = rs.ranges();
    let mut shard_infos: Vec<ShardInfo> = Vec::new();
    let mut cum = 0usize;
    let mut ri = 0usize; // forward cursor over `ranges`
    for sc in &plan.candidate_shards {
        if let Some(limit) = limit {
            if cum >= limit {
                break;
            }
        }
        let Some(stats) = sorted_shards[sc.shard_idx].stats.as_ref() else {
            continue;
        };
        let shard_start = stats.row_start;
        let shard_end = stats.row_end;
        let n_shard_rows = (shard_end - shard_start) as usize;

        // Skip ranges that end at/before this shard starts (fully consumed by
        // an earlier shard or sitting in a Level-1-pruned gap).
        while ri < ranges.len() && ranges[ri].end <= shard_start {
            ri += 1;
        }

        let mut local_mask = vec![false; n_shard_rows];
        let mut kept = 0usize;
        let mut j = ri;
        while j < ranges.len() && ranges[j].start < shard_end {
            let lo = ranges[j].start.max(shard_start);
            let hi = ranges[j].end.min(shard_end);
            local_mask[(lo - shard_start) as usize..(hi - shard_start) as usize].fill(true);
            kept += (hi - lo) as usize;
            // Only step the local cursor past ranges fully inside this shard; a
            // range extending beyond `shard_end` is reprocessed for the next.
            if ranges[j].end <= shard_end {
                j += 1;
            } else {
                break;
            }
        }

        if kept > 0 {
            cum += kept;
            shard_infos.push(ShardInfo {
                shard_idx: sc.shard_idx,
                row_start: shard_start,
                row_end: shard_end,
                local_keep_mask: local_mask,
            });
        }
    }
    shard_infos
}

/// Attempt the row-set fast path. Returns `Ok(None)` (so the caller uses the
/// legacy full-decode path) when it does not apply: kill-switch set, no obs
/// shards (legacy single-section file), no obs predicate index, no obs-shard
/// range table, a stale index, or no obs predicate resolvable from the index.
fn try_rowset_mask(
    pipeline: &QueryPipeline,
    plan: &ExecutionPlan,
    sorted_shards: &[&FullCatalogEntry],
    n_obs: usize,
) -> Result<Option<MaskResult>> {
    if !pipeline.rowset_pushdown_enabled() {
        return Ok(None);
    }
    let reader = pipeline.reader();
    // Only the row-sharded (atlas) path has obs shards + a range table.
    if reader.obs_metadata_shard_count() == 0 {
        return Ok(None);
    }
    let Some(index) = plan.obs_predicate_index.as_ref() else {
        return Ok(None);
    };
    // The predicate index is keyed to the CSR/output shard ranges (see
    // `RowSetCtx`), so the index→global mapping uses the catalog's CSR shard
    // ranges. The obs-metadata-shard ranges (separate `shard_target_rows`
    // chunking) are used only by `materialize` to read filtered obs.
    let Some(csr_shard_ranges) = csr_shard_ranges_table(sorted_shards) else {
        return Ok(None);
    };
    let Some(obs_shard_ranges) = obs_shard_ranges_from_catalog(reader.catalog()) else {
        return Ok(None);
    };
    // Staleness guard: a file-scope index that doesn't cover all obs rows (e.g.
    // after `append`) must not drive row selection.
    if !index_covers_all_obs(index, &csr_shard_ranges, n_obs as u64) {
        return Ok(None);
    }

    let ctx = RowSetCtx {
        index,
        shard_row_ranges: &csr_shard_ranges,
        n_obs: n_obs as u64,
        category_dicts: &plan.category_dicts,
    };
    let (indexed, residual) = partition_obs_predicates(&plan.obs_predicates, &ctx);
    // Nothing resolved from the index (including a no-filter query) → legacy
    // path, unchanged.
    let Some(mut rs) = indexed else {
        return Ok(None);
    };

    // Restrict to surviving candidate shards, then subtract deletions.
    rs = rs.intersect(&candidate_shard_coverage(plan, sorted_shards));
    if let Some(ref dv) = plan.deletion_vectors {
        rs = rs.difference(&deletion_rowset(dv, n_obs as u64));
    }

    // Residual predicates: decode only the obs shards `rs` spans.
    if !residual.is_empty() {
        rs = apply_residual(reader, &residual, &rs, &obs_shard_ranges)?;
    }

    // Full match count (used by `count()`), with no X decode.
    let matched_rows = rs.cardinality() as usize;
    let shard_infos = shard_infos_from_rowset(plan, sorted_shards, &rs, plan.limit);

    Ok(Some(MaskResult {
        legacy_obs: None,
        obs_shard_ranges,
        shard_infos,
        matched_rows,
    }))
}

/// Legacy full-decode masking path (pre-row-set behavior). Decodes the obs
/// shards overlapping the candidate CSR shards, evaluates ALL obs predicates as
/// boolean masks, applies deletion vectors, and builds per-shard keep masks.
fn build_legacy_mask(
    pipeline: &QueryPipeline,
    plan: &ExecutionPlan,
    sorted_shards: &[&FullCatalogEntry],
    n_obs: usize,
) -> Result<MaskResult> {
    let reader = pipeline.reader();
    let mut obs_mask = vec![true; n_obs];
    let mut legacy_obs: Option<RecordBatch> = None;
    let mut obs_shard_ranges: Vec<(u32, u64, u64)> = Vec::new();

    if reader.obs_metadata_shard_count() > 0 {
        // Catalog row ranges (cheap) let us skip decoding obs shards that
        // don't overlap any surviving CSR shard. Absent (pre-stats files)
        // -> stream every shard (bounded memory, no I/O skip).
        let catalog_ranges = obs_shard_ranges_from_catalog(reader.catalog());

        // Candidate CSR shard row ranges, sorted by row_start, for the
        // overlap test.
        let mut candidate_ranges: Vec<(u64, u64)> = plan
            .candidate_shards
            .iter()
            .filter_map(|sc| {
                sorted_shards[sc.shard_idx]
                    .stats
                    .as_ref()
                    .map(|s| (s.row_start, s.row_end))
            })
            .collect();
        candidate_ranges.sort_unstable_by_key(|(s, _)| *s);

        // Which obs shards to actually decode, each paired with its
        // catalog row_start when known (avoids a metadata parse).
        let needed: Vec<(u32, Option<u64>)> = match &catalog_ranges {
            Some(ranges) => ranges
                .iter()
                .filter(|(_, rs, re)| range_overlaps_any(*rs, *re, &candidate_ranges))
                .map(|(idx, rs, _)| (*idx, Some(*rs)))
                .collect(),
            None => (0..reader.obs_metadata_shard_count() as u32)
                .map(|idx| (idx, None))
                .collect(),
        };

        // Evaluate predicates per shard in parallel; each worker reads its
        // shard, produces a local keep mask, then drops the batch. Peak
        // string memory is bounded by ~(rayon width × one shard).
        let per_shard: Vec<(u64, Vec<bool>)> =
            par_map_with_shard_retry(&needed, |&(idx, cat_row_start)| {
                let batch = reader.read_obs_shard(idx)?;
                let row_start = match cat_row_start {
                    Some(rs) => rs,
                    None => obs_shard_row_start(&batch)?,
                };
                let mut local = vec![true; batch.num_rows()];
                for pred in &plan.obs_predicates {
                    let arr = evaluate(pred, &batch)?;
                    for (i, m) in local.iter_mut().enumerate() {
                        if *m {
                            *m = arr.is_valid(i) && arr.value(i);
                        }
                    }
                }
                Ok((row_start, local))
            })?;

        // Scatter local masks into the global mask. Obs shards are
        // disjoint and contiguous, so writes never overlap.
        for (row_start, local) in &per_shard {
            let base = *row_start as usize;
            if base >= n_obs {
                continue;
            }
            let end = (base + local.len()).min(n_obs);
            obs_mask[base..end].copy_from_slice(&local[..end - base]);
        }

        // Ranges for `materialize`: catalog ranges are the complete,
        // ordered set; otherwise we read every shard so the read results
        // cover all of obs — reconstruct from them.
        obs_shard_ranges = match catalog_ranges {
            Some(ranges) => ranges,
            None => {
                let mut r: Vec<(u32, u64, u64)> = needed
                    .iter()
                    .zip(per_shard.iter())
                    .map(|((idx, _), (rs, local))| (*idx, *rs, *rs + local.len() as u64))
                    .collect();
                r.sort_by_key(|(idx, _, _)| *idx);
                r
            }
        };
    } else {
        // Legacy single-section obs: read once and evaluate over the full
        // batch (these files are not atlas-scale).
        let obs_batch = reader.read_obs()?;
        for pred in &plan.obs_predicates {
            let mask_array = evaluate(pred, &obs_batch)?;
            for (i, m) in obs_mask.iter_mut().enumerate() {
                if *m {
                    *m = mask_array.is_valid(i) && mask_array.value(i);
                }
            }
        }
        legacy_obs = Some(obs_batch);
    }

    // Step 4: Apply deletion vectors — exclude deleted cells. v2 stores global
    // obs rows directly, so no shard-local translation is needed.
    if let Some(ref dv) = plan.deletion_vectors {
        if let Some(bitmap) = dv.global_deleted() {
            for row in bitmap.iter() {
                if (row as usize) < n_obs {
                    obs_mask[row as usize] = false;
                }
            }
        }
    }

    // Step 5: Map matching cells to shards — build per-shard keep masks.
    let mut shard_infos: Vec<ShardInfo> = Vec::new();
    for sc in &plan.candidate_shards {
        let entry = sorted_shards[sc.shard_idx];
        if let Some(ref stats) = entry.stats {
            let n_shard_rows = (stats.row_end - stats.row_start) as usize;
            let mut local_mask = Vec::with_capacity(n_shard_rows);
            let mut any_match = false;
            for local_row in 0..n_shard_rows {
                let global_row = stats.row_start as usize + local_row;
                let keep = global_row < n_obs && obs_mask[global_row];
                if keep {
                    any_match = true;
                }
                local_mask.push(keep);
            }
            if any_match {
                shard_infos.push(ShardInfo {
                    shard_idx: sc.shard_idx,
                    row_start: stats.row_start,
                    row_end: stats.row_end,
                    local_keep_mask: local_mask,
                });
            }
        }
    }

    // Pre-limit Level-2 match count, from the keep masks alone — no X decode.
    let matched_rows: usize = shard_infos
        .iter()
        .map(|si| si.local_keep_mask.iter().filter(|&&k| k).count())
        .sum();

    Ok(MaskResult {
        legacy_obs,
        obs_shard_ranges,
        shard_infos,
        matched_rows,
    })
}

/// True if `[rs, re)` overlaps any range in `sorted`.
///
/// **Precondition:** `sorted` must be **non-overlapping** and sorted by
/// `start`. For non-overlapping ranges this also makes the `end` field
/// monotonically non-decreasing, which is what licenses the
/// `partition_point` on `end` below — on overlapping ranges that binary
/// search can step past a range that actually overlaps. CSR shard row
/// ranges (the only caller) are contiguous and disjoint, so they satisfy
/// this. Do not reuse this helper for arbitrary, possibly-overlapping
/// ranges without a linear scan instead.
fn range_overlaps_any(rs: u64, re: u64, sorted: &[(u64, u64)]) -> bool {
    // Find the first range whose end is > rs, then check it starts < re.
    let i = sorted.partition_point(|(_, e)| *e <= rs);
    sorted.get(i).is_some_and(|(s, _)| *s < re)
}
