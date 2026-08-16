// Pipeline execution engine — collect.rs
//
// Ties together pushdown, decode, projection, filtering, and fused operations
// to execute a QueryPipeline. Called by `QueryPipeline::collect()`.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use arrow::array::{Array, RecordBatch, UInt32Array};
use arrow::compute;
use rayon::prelude::*;
use scx_sparse::ScxCsr;

use crate::error::{EngineError, Result};
use crate::fused_ops::apply_fused_ops;
use crate::index::{index_covers_all_obs, IndexedColumn, PredicateIndex};
use crate::pipeline::{QueryPipeline, QueryResult};
use crate::predicate::{eval_rowset, evaluate, Predicate, RowSetCtx};
use crate::projection::{decode_shard_projected, project_var};
use crate::pushdown::{prune_shards_by_catalog_with_dict, CategoryDictionaries, ShardCandidate};
use crate::reader::SectionReader;
use crate::rowset::{RowRange, RowSet};

use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::{assemble_filtered_metadata, DeletionVectors, SectionType};

/// Env var (diagnostic only) that forces the legacy full-decode obs path,
/// disabling row-set predicate pushdown.
///
/// Read **once per pipeline**, at construction, into
/// [`QueryPipeline::rowset_pushdown`](crate::QueryPipeline::rowset_pushdown)'s
/// backing field — not once per query. That field is the only thing
/// [`compute_mask`] consults, so nothing has to mutate the process environment
/// to exercise the legacy path; `tests/rowset_differential.rs` sets the field.
pub(crate) fn rowset_pushdown_disabled_by_env() -> bool {
    std::env::var_os("SCX_DISABLE_ROWSET_PUSHDOWN").is_some()
}

// ============================================================================
// F0. Resilient parallel shard map
// ============================================================================

/// Is `e` worth a second attempt? Transient I/O failures are — on the cloud
/// path every `CloudError` (timeout, download-failed, rate-limited) is erased
/// to [`EngineError::IoError`] at the `CloudSectionReader` boundary, and
/// `Generic` covers other transient surfaces. Deterministic decode failures
/// (`FormatError` / `ArrowError` / `CsrError` / `SchemaError`) are NOT: a retry
/// would just re-fail and double the work on genuinely corrupt input.
fn is_retryable_engine_err(e: &EngineError) -> bool {
    matches!(e, EngineError::IoError(_) | EngineError::Generic(_))
}

/// Run `f` over `items` in parallel, tolerating transient per-item failures.
///
/// Unlike `items.par_iter().map(f).collect::<Result<Vec<_>>>()`, pass 1 does
/// NOT short-circuit on the first `Err`: it runs `f` on every item via rayon
/// and keeps each result paired with its input index. If everything succeeded,
/// the results are returned in input order.
///
/// If some items failed:
/// - A non-retryable failure ([`is_retryable_engine_err`] == false, i.e. a
///   deterministic decode error) is returned immediately — this preserves the
///   prior fast-fail on corrupt input.
/// - Otherwise the failed items are retried once, **sequentially**. Lower
///   concurrency on the retry pass gives a transient/congestion window time to
///   clear, and each call still gets a fresh per-request retry budget from the
///   cloud `RetryingStore`. The call fails only if an item is still unrecovered
///   after the retry pass, and the error then names the offending indices
///   rather than surfacing just the first error.
///
/// This keeps a single shard GET exhausting its per-request retry budget from
/// aborting a whole atlas-scale query via `?`-propagation. `f` may be invoked
/// up to twice per item, so it must be idempotent.
fn par_map_with_shard_retry<I, T, F>(items: &[I], f: F) -> Result<Vec<T>>
where
    I: Sync,
    T: Send,
    F: Fn(&I) -> Result<T> + Sync,
{
    // Pass 1: parallel, order-preserving (slice `par_iter` is indexed), keep
    // every Result so a single failure doesn't discard the other shards' work.
    let pass1: Vec<Result<T>> = items.par_iter().map(&f).collect();

    let mut slots: Vec<Option<T>> = Vec::with_capacity(items.len());
    let mut failed: Vec<usize> = Vec::new();
    let mut first_nonretryable: Option<EngineError> = None;
    for (i, r) in pass1.into_iter().enumerate() {
        match r {
            Ok(v) => slots.push(Some(v)),
            Err(e) => {
                if !is_retryable_engine_err(&e) && first_nonretryable.is_none() {
                    first_nonretryable = Some(e);
                }
                slots.push(None);
                failed.push(i);
            }
        }
    }

    // A deterministic decode failure is not worth a second attempt — fail fast.
    if let Some(e) = first_nonretryable {
        return Err(e);
    }

    if !failed.is_empty() {
        let mut last_err: Option<EngineError> = None;
        let mut unrecovered: Vec<usize> = Vec::new();
        for &i in &failed {
            match f(&items[i]) {
                Ok(v) => slots[i] = Some(v),
                Err(e) => {
                    last_err = Some(e);
                    unrecovered.push(i);
                }
            }
        }
        if !unrecovered.is_empty() {
            return Err(EngineError::Generic(format!(
                "shard read failed after retry for {} of {} item(s) (indices {:?}): {}",
                unrecovered.len(),
                items.len(),
                unrecovered,
                last_err
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string()),
            )));
        }
    }

    Ok(slots
        .into_iter()
        .map(|s| s.expect("every slot filled by pass 1 success or retry recovery"))
        .collect())
}

// ============================================================================
// F1. Execution plan
// ============================================================================

/// Internal execution plan built from a QueryPipeline.
struct ExecutionPlan {
    candidate_shards: Vec<ShardCandidate>,
    obs_predicates: Vec<Predicate>,
    /// Used for var-level filtering during execution (see `execute()`).
    var_predicates: Vec<Predicate>,
    gene_indices: Option<Vec<u32>>,
    normalize: Option<f64>,
    log1p: bool,
    limit: Option<usize>,
    deletion_vectors: Option<DeletionVectors>,
    /// The deserialized obs predicate index, retained for **row-set pushdown**
    /// (row selection), not just the category dictionaries used for
    /// catalog-level shard pruning. `None` when the file has no obs predicate
    /// index. See [`try_rowset_mask`].
    obs_predicate_index: Option<PredicateIndex>,
    /// Per-column category vocabularies and whether each may be trusted as
    /// complete. Built once in [`build_plan`] and used by **both** pushdown
    /// levels — Level-1 for catalog pruning, Level-2 for row-set resolution —
    /// so the two cannot end up trusting the index to different degrees.
    category_dicts: CategoryDictionaries,
}

/// Build an execution plan from a QueryPipeline.
///
/// Runs catalog-level shard pruning and loads predicate indexes.
fn build_plan(pipeline: &QueryPipeline) -> Result<ExecutionPlan> {
    let catalog = pipeline.reader().catalog();

    // Load the obs predicate index (C5) — needed for category dictionaries.
    // The var predicate index is not consumed by execution (no var-level
    // pushdown is wired into the query path), so it is not read here.
    let obs_predicate_index = match pipeline.reader().read_obs_predicate_index_bytes()? {
        Some(bytes) => Some(PredicateIndex::read_from(&mut Cursor::new(bytes))?),
        None => None,
    };

    // Build category dictionaries from predicate index for catalog-level pruning.
    // Maps column_name_hash → sorted list of category values, so Utf8 predicate
    // values can be resolved to CategoryBitset bit positions.
    // Completeness is judged over exactly the shards the pruner will iterate —
    // `prune_shards_by_catalog_with_dict` scans the same modality below — so
    // the claim "every shard being pruned was covered by this vocabulary's
    // build" is checked against those shards and no others. (The index's own
    // `shard_id` space does not enter into it: the test is per-entry, "does
    // this shard carry a `CategoryBitset` for this column hash".)
    let pruned_shards = scan_shards(catalog, pipeline.modality_id());
    let category_dicts = build_category_dicts(&obs_predicate_index, &pruned_shards);
    let dicts_ref = if category_dicts.is_empty() {
        None
    } else {
        Some(&category_dicts)
    };

    // Catalog-level shard pruning (B1), now with category dictionary support,
    // scoped to the pipeline's modality (== shards_sorted() for the default
    // single-modality axis).
    let candidate_shards = prune_shards_by_catalog_with_dict(
        catalog,
        pipeline.obs_predicates(),
        pipeline.deletion_vectors().as_ref(),
        dicts_ref,
        pipeline.modality_id(),
    );

    // Level-2 row-set pushdown keys `ShardRange.shard_id` to the flattened
    // (all-modality) shard order at write time; those ids do not match a
    // per-modality enumeration. Disable the fast path for modality-scoped
    // queries so they fall back to modality-scoped Level-1 pruning + a global
    // obs residual scan (correct, just less pruned). The category dictionaries
    // above are position-based vocab (shard-id-independent) so Level-1 pruning
    // keeps working. See `docs/multimodal.md` § 3.4.
    let obs_predicate_index = if pipeline.modality_id() == 0 {
        obs_predicate_index
    } else {
        None
    };

    Ok(ExecutionPlan {
        candidate_shards,
        obs_predicates: pipeline.obs_predicates().to_vec(),
        var_predicates: pipeline.var_predicates().to_vec(),
        gene_indices: pipeline.gene_indices().cloned(),
        normalize: pipeline.normalize_target_sum(),
        log1p: pipeline.log1p(),
        limit: pipeline.limit_value(),
        deletion_vectors: pipeline.deletion_vectors().clone(),
        obs_predicate_index,
        category_dicts,
    })
}

/// Build category dictionaries from a predicate index.
///
/// For each categorical column in the index, creates a mapping from
/// `column_name_hash` to the sorted list of category values. The position
/// in this list corresponds to the bit position in `CategoryBitset`.
///
/// Each dictionary also carries whether its vocabulary may be trusted as the
/// column's **complete** value set — see [`BitsetCoverage::covers`].
/// Level-1 needs that bit before it may read a dictionary miss as proof the
/// value exists in no shard.
///
/// `pruned_shards` must be the shard list the caller will hand to
/// [`prune_shards_by_catalog_with_dict`] — the same modality. Completeness is a
/// statement about *those* shards, so judging it over any other set would
/// license pruning shards nothing was checked against. Note this is **not**
/// the index's `shard_id` space: the evidence is per-catalog-entry (see
/// [`collect_bitset_coverage`]) and never resolves a shard id.
fn build_category_dicts(
    index: &Option<PredicateIndex>,
    pruned_shards: &[&FullCatalogEntry],
) -> CategoryDictionaries {
    let mut dicts = CategoryDictionaries::new();
    let index = match index {
        Some(idx) => idx,
        None => return dicts,
    };
    let coverage = collect_bitset_coverage(pruned_shards);
    for col in &index.columns {
        if let IndexedColumn::Categorical(cat) = col {
            let hash = scx_format_io::column_name_hash(&cat.column_name);
            let values: Vec<String> = cat.entries.iter().map(|e| e.value.clone()).collect();
            let complete = coverage
                .get(&hash)
                .is_some_and(|cov| cov.covers(pruned_shards.len(), values.len()));
            // CategoricalIndex entries are already sorted by BTreeMap in build_indexes
            dicts.insert(hash, values, complete);
        }
    }
    dicts
}

/// What the catalog says about one column's per-shard `CategoryBitset`s,
/// accumulated in a **single** pass over the shard entries.
///
/// One pass matters. The obvious shape — ask "is this column covered?" once per
/// indexed column, each asking walking every shard's whole `column_stats`
/// vector — is O(shards × columns²), and `build_plan` runs it on every query,
/// including a `collect` or `count` with no obs predicate at all. A shard may
/// carry up to `u8::MAX` stats, so at atlas shard counts that is tens of
/// millions of hash comparisons to answer a question about a handful of
/// columns. Gathering the evidence once and answering per column from the map
/// is O(shards × stats) regardless of how many columns are indexed.
#[derive(Debug, Default)]
struct BitsetCoverage {
    /// **Distinct shards** carrying a `CategoryBitset` for this column — not
    /// the number of such stat records. The two differ exactly when one shard
    /// carries the column twice, and counting records would then let that
    /// shard's surplus pay for another shard's absence.
    shards: usize,
    /// Byte length of the first bitset seen.
    len: usize,
    /// Cleared once two shards disagree about that length.
    consistent: bool,
    /// Set when one shard carried this column more than once. A malformed
    /// catalog, not extra evidence: nothing says which of the two bitsets the
    /// dictionary's bit positions belong to.
    duplicated: bool,
}

impl BitsetCoverage {
    /// Record one shard's bitset for this column. Call **at most once per
    /// shard** — [`collect_bitset_coverage`] routes a repeat to
    /// [`Self::mark_duplicated`] instead, which is what keeps `shards` a shard
    /// count rather than a record count.
    fn observe(&mut self, len: usize) {
        if self.shards == 0 {
            self.len = len;
            self.consistent = true;
        } else if self.len != len {
            self.consistent = false;
        }
        self.shards += 1;
    }

    fn mark_duplicated(&mut self) {
        self.duplicated = true;
    }

    /// Whether this column's vocabulary of `n_values` may be trusted as the
    /// complete value set over `n_shards` shards.
    ///
    /// Four conditions, and each rules out a way the catalog and the index
    /// section can disagree:
    ///
    /// - **Every shard carries a bitset.** `derive_shard_column_stats` emits one
    ///   for *every* shard of an indexed categorical — including an all-zero one
    ///   where the column has no values there — so a missing bitset means the
    ///   build that produced this vocabulary never saw that shard. This is what
    ///   `append` without `--index-obs` leaves behind: the appended CSR shards
    ///   carry no `column_stats` at all.
    /// - **It is a `CategoryBitset`, not just *some* stat with this hash.**
    ///   `ColumnStat::column_name_hash` answers for `MinMax` too, so testing the
    ///   hash alone lets a numeric stat license a categorical vocabulary.
    /// - **Its length is the one this vocabulary implies**, consistently across
    ///   shards. `derive_shard_column_stats` sizes every bitset
    ///   `entries.len().div_ceil(8)` from the same entry list the dictionary
    ///   comes from, so a disagreement means the stats and the index section
    ///   were produced by different builds — and then bit *i* does not mean
    ///   entry *i*.
    /// - **No shard carries it twice.** `shards` counts distinct shards, so the
    ///   count alone cannot distinguish "both shards covered" from "one shard
    ///   covered twice, the other not at all" — and it is the uncovered shard
    ///   that the vocabulary would then be claiming to describe. A repeat is
    ///   also unresolvable on its own terms: nothing says which of the two
    ///   bitsets the dictionary's bit positions belong to.
    ///
    /// The last three matter at **Level-2** especially. Level-1 declines a
    /// mismatched bitset per shard (`pushdown::bitset_matches_dictionary`) and
    /// never prunes a shard that has no stats at all, but Level-2 does not look
    /// at bitsets: it asks only whether the column is usable and then treats
    /// `categorical_eq` as exact. Folding these conditions in here is what makes
    /// a mismatched, wrong-variant or unevenly-covered column residual at
    /// Level-2 rather than authoritative.
    ///
    /// An empty shard list is never complete: with nothing to check against, a
    /// coverage claim would be vacuous.
    fn covers(&self, n_shards: usize, n_values: usize) -> bool {
        n_shards > 0
            && self.consistent
            && !self.duplicated
            && self.shards == n_shards
            && self.len == n_values.div_ceil(8)
    }
}

/// One pass over `pruned_shards`, recording each column's `CategoryBitset`
/// coverage. See [`BitsetCoverage`] for why this is a pass rather than a query.
///
/// `seen_here` is what keeps [`BitsetCoverage::shards`] a count of *shards*
/// rather than of stat records: a column met twice within one shard is recorded
/// as duplicated instead of counted twice.
fn collect_bitset_coverage(pruned_shards: &[&FullCatalogEntry]) -> HashMap<u64, BitsetCoverage> {
    let mut coverage: HashMap<u64, BitsetCoverage> = HashMap::new();
    let mut seen_here: HashSet<u64> = HashSet::new();
    for entry in pruned_shards {
        let Some(stats) = entry.stats.as_ref() else {
            continue;
        };
        seen_here.clear();
        for cs in &stats.column_stats {
            if let scx_format_io::catalog::ColumnStat::CategoryBitset {
                column_name_hash,
                bitset,
            } = cs
            {
                let cov = coverage.entry(*column_name_hash).or_default();
                if seen_here.insert(*column_name_hash) {
                    cov.observe(bitset.len());
                } else {
                    cov.mark_duplicated();
                }
            }
        }
    }
    coverage
}

// ============================================================================
// F3. Row filtering within shards
// ============================================================================

/// Extract only the rows where `keep_mask[row]` is true from decoded CSR data.
///
/// Returns new `(indptr, indices, data)` arrays for the filtered subset.
/// Operates on pre-decoded scipy-compatible types (i64/i32/f32).
pub fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    keep_mask: &[bool],
) -> Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
    let n_rows = indptr.len().saturating_sub(1);
    // A mismatch used to be a `debug_assert!` followed by
    // `min(keep_mask.len(), n_rows)`, which means release builds — the ones
    // users run — silently dropped every row past the shorter of the two and
    // returned a truncated query result. The assertion documented the invariant
    // and then the next line worked around it.
    //
    // Same failure the reader-side deletion filters had: the mask and the CSR
    // come from independent places, and the direction that does *not* panic is
    // the dangerous one, because a wrong answer looks like an answer. Reject
    // both directions instead of clamping.
    if keep_mask.len() != n_rows {
        return Err(EngineError::FormatError(
            scx_format_io::ScxError::InvalidCatalog(format!(
                "filter_csr_rows: keep mask covers {} rows but the decoded CSR has {} \
             (truncated or corrupt shard)",
                keep_mask.len(),
                n_rows,
            )),
        ));
    }
    let mask_len = n_rows;

    let mut new_indptr = Vec::with_capacity(mask_len + 1);
    new_indptr.push(0i64);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    for row in 0..mask_len {
        if !keep_mask[row] {
            continue;
        }
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        new_indices.extend_from_slice(&indices[start..end]);
        new_data.extend_from_slice(&data[start..end]);
        let prev = *new_indptr.last().unwrap();
        new_indptr.push(prev + (end - start) as i64);
    }

    Ok((new_indptr, new_indices, new_data))
}

// ============================================================================
// F2. Pipeline execution
// ============================================================================

// Query execution is split into two halves (CLI2): `plan_and_mask` does the
// cheap work — (1) build plan / catalog-level pruning, (2) read obs metadata,
// (3) evaluate obs predicates → boolean mask, (4) apply deletion vectors,
// (5) map cells to shards / build keep masks, (6) var predicates + gene
// projection — without decoding any X shard. `materialize` does the expensive
// rest — (7) parallel shard decode, (8) assemble CSR, (9) fused normalize+log1p,
// (10) apply limit, (11) filter obs/var, (12) return QueryResult. `execute`
// runs both; `count` runs only the first and returns the match count.

/// Per-shard row range + local keep mask produced by `plan_and_mask`.
struct ShardInfo {
    shard_idx: usize,
    row_start: u64,
    #[allow(dead_code)] // retained for debugging and future use
    row_end: u64,
    local_keep_mask: Vec<bool>,
}

/// Result of the cheap planning + masking half of a query (CLI2): catalog
/// shard elimination, obs/var predicate evaluation, per-shard row-keep masks,
/// and gene projection — everything computable **without decoding any X
/// shard**. Consumed by [`materialize`] (full result) or summarised by
/// [`count`] (matched-row count only).
struct PlanAndMask {
    plan: ExecutionPlan,
    /// Full obs batch — populated ONLY on the legacy single-section path.
    /// On the row-sharded (atlas-scale) path obs is never materialised in
    /// full; `materialize` rebuilds just the matching rows from
    /// `obs_shard_ranges`. See [`plan_and_mask`].
    legacy_obs: Option<RecordBatch>,
    /// `(shard_idx, row_start, row_end)` for every obs metadata shard, in
    /// `shard_idx` order. Empty on the legacy single-section path. Lets
    /// `materialize` map matching global rows to the few obs shards that
    /// contain them and read only those.
    obs_shard_ranges: Vec<(u32, u64, u64)>,
    var_batch: RecordBatch,
    shard_infos: Vec<ShardInfo>,
    effective_gene_indices: Option<Vec<u32>>,
    n_output_cols: usize,
    /// Restores the caller's requested gene order on the materialized output
    /// (F4). `None` when the request was already ascending-unique.
    reorder: Option<ColumnReorder>,
    skipped_shards: usize,
    total_shards: usize,
    candidate_shard_rows: usize,
    /// Pre-limit Level-2 match count = Σ `local_keep_mask` trues. Equals
    /// `csr.n_rows()` after materialisation, but needs no decode.
    matched_rows: usize,
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

// ============================================================================
// F1b. Per-shard masking — row-set fast path + legacy full-decode fallback
// ============================================================================

/// The per-shard masking output of [`compute_mask`]: everything `materialize`
/// needs to decode X and filter obs/var, computed without decoding any X shard.
struct MaskResult {
    /// Full obs batch — populated ONLY on the legacy single-section path.
    legacy_obs: Option<RecordBatch>,
    obs_shard_ranges: Vec<(u32, u64, u64)>,
    shard_infos: Vec<ShardInfo>,
    /// Pre-limit Level-2 match count (Σ keep-mask trues over ALL matching
    /// candidate shards). Used by `count()`; equals `csr.n_rows()` after an
    /// unlimited materialise. NOT reduced when `shard_infos` is capped for a
    /// limited query (the row-set path keeps it exact and full).
    matched_rows: usize,
}

/// Build the per-shard keep masks. Prefers the **row-set fast path** — resolve
/// indexed obs predicates straight from the predicate index, decoding no obs
/// shard for the indexed part — and falls back to the legacy full-decode path
/// for non-indexed / residual-only predicates, legacy single-section obs, a
/// stale index, or when the pushdown kill-switch is set.
fn compute_mask(
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

/// The CSR shard list a modality-scoped pipeline scans, sorted by `row_start`.
///
/// This is the single source of truth for the shard set every enumerate-position
/// consumer (`build_plan` pruning, the shard-range table, candidate coverage, the
/// deletion rowset, and `materialize`'s decode) walks, so `shard_idx` numbering
/// stays internally consistent. On a single-modality / v1 file `modality_id == 0`
/// and this is order-and-set identical to the legacy `catalog.shards_sorted()`
/// (both filter `CsrShard` and sort by `row_start`; v1 entries carry
/// `modality_id = 0`) — the default path is byte-for-byte unchanged. See
/// `docs/multimodal.md` § 3.4.
pub(crate) fn scan_shards(
    catalog: &scx_format_io::FullCatalog,
    modality_id: u8,
) -> Vec<&FullCatalogEntry> {
    catalog.csr_shards_for_modality(modality_id)
}

/// The CSR/output shard row-range table the predicate index was built against:
/// `(shard_id, row_start, row_end)` where `shard_id` is the shard's position in
/// `shards_sorted()` (== its index in `output_shard_row_ranges` at build time).
/// Returns `None` if any shard lacks row-range stats — the index then can't be
/// mapped to global rows safely, so the caller falls back to the legacy path.
///
/// ⚠️ **The position `i` here equals the index's `shard_id` only at
/// `modality_id == 0`.** `sorted_shards` comes from [`scan_shards`], which is
/// modality-scoped; on a multimodal file each modality's shards independently
/// tile `[0, n_obs)`, so modality 2's third shard is also position 2 and would
/// silently adopt modality 1's row-set.
///
/// What prevents that is [`build_plan`], which forces `obs_predicate_index` to
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

/// Maps the engine's sorted-projection column layout to the caller's requested
/// gene order (F4). The decode merge-scan requires an ascending gene set, but
/// the *output* columns should follow the order the caller passed to
/// `select_genes`. Present only when those differ.
struct ColumnReorder {
    /// `sorted_to_output[k]` = output column index for the gene at sorted
    /// position `k`. Used to relabel decoded CSR column indices in place.
    sorted_to_output: Vec<u32>,
    /// `output_to_sorted[j]` = sorted position of output column `j`. Used as
    /// arrow `take` indices to reorder the projected `var` rows.
    output_to_sorted: Vec<u32>,
}

/// Resolved gene-column projection: the ascending index set fed to the decode
/// path, the output column count, and the optional reorder back to the
/// caller's requested order.
struct GeneProjection {
    /// Sorted, unique gene indices for `decode_shard_projected` / `project_var`
    /// (unchanged fast path). `None` = no projection (all genes).
    decode_indices: Option<Vec<u32>>,
    n_output_cols: usize,
    /// `None` when the request was already ascending-unique (output == decode
    /// order) — the common case, byte-identical to the pre-F4 behavior.
    reorder: Option<ColumnReorder>,
}

/// Resolve the effective gene-column projection (explicit gene indices
/// intersected with any var predicates) and the resulting output column count.
/// Operates on already-read var metadata — no obs/X shard decode. Shared by the
/// normal `plan_and_mask` path and its empty-candidate short-circuit.
fn resolve_gene_projection(
    plan: &ExecutionPlan,
    var_batch: &RecordBatch,
    n_vars: usize,
) -> Result<GeneProjection> {
    let mut effective_gene_indices = plan.gene_indices.clone();

    if !plan.var_predicates.is_empty() {
        let var_mask = {
            let mut mask = vec![true; var_batch.num_rows()];
            for pred in &plan.var_predicates {
                let mask_array = evaluate(pred, var_batch)?;
                for (i, m) in mask.iter_mut().enumerate() {
                    if *m {
                        *m = mask_array.is_valid(i) && mask_array.value(i);
                    }
                }
            }
            mask
        };

        // Convert var mask to gene indices
        let var_gene_indices: Vec<u32> = var_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i as u32) } else { None })
            .collect();

        // Intersect with explicit gene_indices if both present
        effective_gene_indices = Some(match effective_gene_indices {
            Some(explicit) => {
                let var_set: std::collections::HashSet<u32> =
                    var_gene_indices.iter().copied().collect();
                explicit
                    .into_iter()
                    .filter(|g| var_set.contains(g))
                    .collect()
            }
            None => var_gene_indices,
        });
    }

    // F4: preserve the caller's requested gene order. The decode merge-scan
    // needs an ascending unique set, but output columns should follow the
    // requested order. Build both and a reorder that maps sorted-column
    // positions to output positions; the reorder is `None` (and the path is
    // byte-identical to pre-F4) when the request is already ascending-unique.
    let Some(output_genes) = effective_gene_indices else {
        return Ok(GeneProjection {
            decode_indices: None,
            n_output_cols: n_vars,
            reorder: None,
        });
    };

    // Dedup preserving first-occurrence order (duplicates collapse, matching
    // the documented limitation).
    let mut seen = std::collections::HashSet::new();
    let output_genes: Vec<u32> = output_genes
        .into_iter()
        .filter(|g| seen.insert(*g))
        .collect();

    let mut sorted = output_genes.clone();
    sorted.sort_unstable(); // already unique (deduped above)
    let n_output_cols = output_genes.len();

    let reorder = if output_genes == sorted {
        None
    } else {
        let mut sorted_to_output = vec![0u32; sorted.len()];
        let mut output_to_sorted = vec![0u32; output_genes.len()];
        for (j, &g) in output_genes.iter().enumerate() {
            let k = sorted
                .binary_search(&g)
                .expect("output gene must be present in its own sorted set");
            sorted_to_output[k] = j as u32;
            output_to_sorted[j] = k as u32;
        }
        Some(ColumnReorder {
            sorted_to_output,
            output_to_sorted,
        })
    };

    Ok(GeneProjection {
        decode_indices: Some(sorted),
        n_output_cols,
        reorder,
    })
}

/// Cheap half of query execution: catalog pruning + obs/var predicate
/// evaluation + per-shard row-keep masks + gene projection. No X-shard decode.
fn plan_and_mask(pipeline: &QueryPipeline) -> Result<PlanAndMask> {
    let plan = build_plan(pipeline)?;
    let reader = pipeline.reader();
    let modality_id = pipeline.modality_id();
    // Per-modality X width (header().n_vars is the file-wide max, not the
    // modality's — docs/format.md § 13.4).
    let n_vars = pipeline.n_vars();

    // Sorted CSR-shard view, computed once and reused across this function
    // (was recomputed — filter+sort+alloc — ≥4× per query). OE6. Scoped to the
    // pipeline's modality (== shards_sorted() for the single-modality default).
    let sorted_shards = scan_shards(reader.catalog(), modality_id);
    debug_assert!(
        modality_id == 0
            || reader
                .catalog()
                .modality_csr_ranges_tile_obs(modality_id, reader.header().n_obs),
        "modality {modality_id} CSR shards must tile [0, n_obs)"
    );
    let total_shards = sorted_shards.len();
    let skipped_shards = total_shards - plan.candidate_shards.len();

    // Sum of rows in candidate shards (post Level 1
    // catalog-stats pruning, pre Level 2 PredicateIndex / row-evaluator
    // narrowing). Lets the CLI report "{matched_rows} of {candidate_rows}
    // candidate-shard rows matched" so users can see Level 2 is doing
    // work even when Level 1 skipped zero shards.
    let candidate_shard_rows: usize = plan
        .candidate_shards
        .iter()
        .filter_map(|sc| {
            sorted_shards[sc.shard_idx]
                .stats
                .as_ref()
                .map(|s| (s.row_end - s.row_start) as usize)
        })
        .sum();

    // No candidate shards survived catalog-level pruning — e.g. an indexed
    // equality whose value is absent from every shard (typo / stale id), or a
    // numeric equality outside every shard's [min, max]. The result is empty,
    // so short-circuit: skip ALL obs/X shard reads. Peak RSS stays at the
    // (MB-scale) predicate-index read regardless of file size, and we report
    // every shard skipped. Var is still read (cheap) for the output schema.
    if plan.candidate_shards.is_empty() {
        let var_batch = reader.read_var_for(modality_id)?;
        let GeneProjection {
            decode_indices: effective_gene_indices,
            n_output_cols,
            reorder,
        } = resolve_gene_projection(&plan, &var_batch, n_vars)?;
        // The empty result still needs the obs schema. On the row-sharded path
        // `materialize_filtered_obs` derives it from a single shard (bounded);
        // on the legacy single-section path there are NO obs shards, so we must
        // hand `materialize` the full obs batch to take the (legacy) empty
        // branch. Legacy files are not atlas-scale, so this read is cheap.
        let legacy_obs = if reader.obs_metadata_shard_count() == 0 {
            Some(reader.read_obs()?)
        } else {
            None
        };
        return Ok(PlanAndMask {
            plan,
            legacy_obs,
            obs_shard_ranges: Vec::new(),
            var_batch,
            shard_infos: Vec::new(),
            effective_gene_indices,
            n_output_cols,
            reorder,
            skipped_shards, // == total_shards (no shard survived)
            total_shards,
            candidate_shard_rows, // == 0
            matched_rows: 0,
        });
    }

    // Steps 2-5: produce the per-shard keep masks. `compute_mask` prefers the
    // row-set fast path — resolving indexed obs predicates directly from the
    // predicate index without decoding any obs shard — and falls back to the
    // legacy full-decode path otherwise (non-indexed / residual-only predicates,
    // legacy single-section obs, a stale index, or the pushdown kill-switch).
    let n_obs = reader.header().n_obs as usize;
    let MaskResult {
        legacy_obs,
        obs_shard_ranges,
        shard_infos,
        matched_rows,
    } = compute_mask(pipeline, &plan, &sorted_shards, n_obs)?;

    // Step 6: Handle var predicates and gene projection (assembled via trait impl)
    let var_batch = reader.read_var_for(modality_id)?;
    let GeneProjection {
        decode_indices: effective_gene_indices,
        n_output_cols,
        reorder,
    } = resolve_gene_projection(&plan, &var_batch, n_vars)?;

    Ok(PlanAndMask {
        plan,
        legacy_obs,
        obs_shard_ranges,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        reorder,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    })
}

/// Execute a query, materialising the full [`QueryResult`].
///
/// Borrows the pipeline — execution is read-only, so it is repeatable and a
/// failure leaves the pipeline intact. See
/// [`QueryPipeline::collect_ref`](crate::pipeline::QueryPipeline::collect_ref).
pub fn execute(pipeline: &QueryPipeline) -> Result<QueryResult> {
    let pm = plan_and_mask(pipeline)?;
    materialize(pipeline, pm)
}

/// Count matching rows without decoding `X` (CLI2). Ignores `limit` entirely,
/// so `--count --limit` cannot misreport (CLI6).
pub fn count(pipeline: &QueryPipeline) -> Result<crate::pipeline::CountResult> {
    let pm = plan_and_mask(pipeline)?;
    Ok(crate::pipeline::CountResult {
        matched_rows: pm.matched_rows,
        skipped_shards: pm.skipped_shards,
        total_shards: pm.total_shards,
        candidate_shard_rows: pm.candidate_shard_rows,
    })
}

/// Whether any row matches, without decoding `X` and ignoring `limit`. On the
/// indexed-only row-set path this needs no obs/X shard decode (the answer comes
/// from the row-set cardinality); with residual predicates it decodes only the
/// narrowed obs shards, like [`count`]. Computes the full match count rather
/// than stopping at the first match — cheap on the indexed path, and a true
/// early-exit is a deferred refinement.
pub fn exists(pipeline: &QueryPipeline) -> Result<bool> {
    Ok(plan_and_mask(pipeline)?.matched_rows > 0)
}

/// Rebuild the filtered obs metadata for a row-sharded file by reading only
/// the obs shards that contain matching rows (the whole point of the OOM
/// fix). `obs_shard_ranges` is `(shard_idx, row_start, row_end)` and
/// `matching_global_rows` is ascending. Peak memory is bounded by the
/// result size, not the file size.
pub(crate) fn materialize_filtered_obs(
    reader: &dyn crate::reader::SectionReader,
    obs_shard_ranges: &[(u32, u64, u64)],
    matching_global_rows: &[u32],
) -> Result<RecordBatch> {
    // View sorted by row_start so we can walk it with a single cursor as we
    // scan the (ascending) matching rows. For a contiguous cover shard_idx
    // order already equals row_start order, but sort defensively.
    let mut by_start: Vec<(u32, u64, u64)> = obs_shard_ranges.to_vec();
    by_start.sort_by_key(|(_, rs, _)| *rs);

    // Group matching rows into (position-in-`by_start`, local row indices),
    // preserving ascending order so the concatenation matches the CSR order.
    let mut groups: Vec<(usize, Vec<u32>)> = Vec::new();
    let mut cur = 0usize;
    for &g in matching_global_rows {
        let g = g as u64;
        while cur < by_start.len() && g >= by_start[cur].2 {
            cur += 1;
        }
        let Some(&(_, rs, _)) = by_start.get(cur) else {
            break; // out of range (shouldn't happen for a valid cover)
        };
        let local = (g - rs) as u32;
        match groups.last_mut() {
            Some((bi, locals)) if *bi == cur => locals.push(local),
            _ => groups.push((cur, vec![local])),
        }
    }

    // Read each needed obs shard and take its matching local rows. For an
    // empty result, feed a single 0-row shard so the assembler produces the
    // canonical schema (avoids the cloud `read_obs_schema` full-assembly).
    //
    // `read_obs_shard(0)` is safe here: `materialize_filtered_obs` is only
    // reached on the row-sharded path, where `obs_shard_ranges` is
    // non-empty — which requires `obs_metadata_shard_count() > 0`, so shard
    // 0 always exists. A future caller that reaches this with empty
    // `obs_shard_ranges` would get a "section not found" error instead.
    let filtered_batches: Vec<RecordBatch> = if groups.is_empty() {
        vec![reader.read_obs_shard(0)?.slice(0, 0)]
    } else {
        par_map_with_shard_retry(&groups, |(bi, locals)| {
            let (shard_idx, _, _) = by_start[*bi];
            let batch = reader.read_obs_shard(shard_idx)?;
            // `par_map_with_shard_retry` may invoke this closure twice (on a
            // transient-failure retry), so it borrows `locals` and clones per
            // call rather than consuming the owned Vec. `UInt32Array::from`
            // still takes the clone's allocation via `Buffer::from_vec`.
            let take_indices = UInt32Array::from(locals.clone());
            let columns: Vec<_> = batch
                .columns()
                .iter()
                .map(|col| compute::take(col.as_ref(), &take_indices, None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(RecordBatch::try_new(batch.schema(), columns)?)
        })?
    };

    // `template_schema` is only consulted when `batches` is empty; we always
    // pass at least one batch, so the first batch's schema is a safe filler.
    let template_schema = filtered_batches[0].schema();
    Ok(assemble_filtered_metadata(
        &template_schema,
        filtered_batches,
    )?)
}

/// Expensive half of query execution: decode matching shards, assemble the CSR
/// matrix, apply fused ops + limit, and filter obs/var metadata to the result.
fn materialize(pipeline: &QueryPipeline, pm: PlanAndMask) -> Result<QueryResult> {
    let PlanAndMask {
        plan,
        legacy_obs,
        obs_shard_ranges,
        var_batch,
        shard_infos,
        effective_gene_indices,
        n_output_cols,
        reorder,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
    } = pm;

    let reader = pipeline.reader();
    let sorted_shards = scan_shards(reader.catalog(), pipeline.modality_id());

    // Step 7: Parallel shard decode with optional projection.
    //
    // Push `limit` into shard selection. `shard_infos` are ascending by
    // `row_start` and each carries its keep mask, so the per-shard matched-row
    // count is known without decoding X. We therefore only decode the first K
    // shards whose cumulative matches reach the limit. This bounds peak decode +
    // CSR memory to those K shards even when a low-selectivity predicate matches
    // rows in every candidate shard — otherwise the entire matched X would be
    // materialised here before Step 10 truncates it to `limit`.
    let decode_count = match plan.limit {
        Some(limit) => {
            let mut cum = 0usize;
            let mut k = 0usize;
            for si in &shard_infos {
                if cum >= limit {
                    break;
                }
                cum += si.local_keep_mask.iter().filter(|&&keep| keep).count();
                k += 1;
            }
            k
        }
        None => shard_infos.len(),
    };

    // Each shard produces (indptr, indices, data) filtered to matching rows
    let shard_results: Vec<(Vec<i64>, Vec<i32>, Vec<f32>)> =
        par_map_with_shard_retry(&shard_infos[..decode_count], |si| {
            let entry = sorted_shards[si.shard_idx];

            // Decode shard (with or without projection)
            let (indptr, indices, data) = if let Some(ref gi) = effective_gene_indices {
                decode_shard_projected(reader, entry, gi)?
            } else {
                reader.read_shard_from_entry(entry)?
            };

            // Filter to matching rows within the shard
            let (filtered_indptr, filtered_indices, filtered_data) =
                filter_csr_rows(&indptr, &indices, &data, &si.local_keep_mask)?;

            Ok((filtered_indptr, filtered_indices, filtered_data))
        })?;

    // Max `value_max` over the shards actually decoded into `csr`. Integer
    // encodings record the true max; float encodings record 0. A reader
    // compares this against `scx_codec::F32_MAX_EXACT_INT` to fail loud on the
    // silent u32→f32 decode loss before returning X.
    let max_value = shard_infos[..decode_count]
        .iter()
        .filter_map(|si| sorted_shards[si.shard_idx].stats.as_ref())
        .map(|s| s.value_max)
        .max()
        .unwrap_or(0);

    // Step 8: Assemble CSR from per-shard results
    let mut merged_indptr: Vec<i64> = Vec::new();
    let mut merged_indices: Vec<i32> = Vec::new();
    let mut merged_data: Vec<f32> = Vec::new();
    let mut cumulative_nnz: i64 = 0;

    for (i, (indptr, indices, data)) in shard_results.iter().enumerate() {
        if i == 0 {
            merged_indptr.extend_from_slice(indptr);
        } else {
            // Skip leading 0 and offset by cumulative nnz
            for &v in &indptr[1..] {
                merged_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += indptr.last().copied().unwrap_or(0);
        merged_indices.extend_from_slice(indices);
        merged_data.extend_from_slice(data);
    }

    // Handle empty result case
    if merged_indptr.is_empty() {
        merged_indptr.push(0);
    }

    let n_rows = merged_indptr.len() - 1;
    let mut csr = ScxCsr::new_unchecked(
        (n_rows, n_output_cols),
        merged_indptr,
        merged_indices,
        merged_data,
    );

    // Step 9: Apply fused normalize+log1p
    apply_fused_ops(&mut csr, plan.normalize, plan.log1p);

    // `matched_rows` (pre-limit Level-2 count) was computed in `plan_and_mask`
    // from the keep masks. With no limit we decode every candidate shard, so it
    // must equal `csr.n_rows()` here — asserting that keeps the mask-only
    // `count()` path honest against the decode path without re-deriving it. With
    // a limit we decode only the prefix of shards needed to fill it (Step 7), so
    // the assembled CSR holds only those rows (>= limit, <= matched_rows) until
    // Step 10 truncates.
    debug_assert!(
        plan.limit.is_some() || matched_rows == csr.n_rows(),
        "unlimited query: matched_rows ({matched_rows}) must equal assembled rows ({})",
        csr.n_rows(),
    );

    // Step 10: Apply limit
    if let Some(limit) = plan.limit {
        if limit < csr.n_rows() {
            csr = csr.row_slice(0, limit)?;
        }
    }

    // Step 11: Filter obs metadata to matching rows
    // Build the list of global row indices that made it into the output.
    // `shard_infos` come from `candidate_shards`, which derive from
    // `catalog.shards_sorted()` (ascending `row_start`), so this list is
    // globally ascending and aligns row-for-row with the assembled CSR.
    //
    // Only the decoded prefix (`shard_infos[..decode_count]`) contributes rows
    // to the assembled CSR; for a `limit`ed query the trailing shards were
    // never decoded. The prefix already holds >= `limit` matches, so building
    // over it (then truncating below) yields the same first-`limit` rows while
    // avoiding an all-shards alloc on the low-selectivity full-scan path. With
    // no limit `decode_count == shard_infos.len()`, so this is unchanged.
    let mut matching_global_rows: Vec<u32> = Vec::new();
    for si in &shard_infos[..decode_count] {
        for (local_row, &keep) in si.local_keep_mask.iter().enumerate() {
            if keep {
                matching_global_rows.push((si.row_start as usize + local_row) as u32);
            }
        }
    }

    // Apply limit to matching rows BEFORE choosing obs shards, so a small
    // `limit` reads only the obs shard(s) covering the first N matches.
    if let Some(limit) = plan.limit {
        matching_global_rows.truncate(limit);
    }
    debug_assert!(
        matching_global_rows.windows(2).all(|w| w[0] <= w[1]),
        "matching_global_rows must be ascending to align with the assembled CSR"
    );

    let filtered_obs = if let Some(ref obs_batch) = legacy_obs {
        // Legacy single-section obs: take directly from the full batch.
        let take_indices = UInt32Array::from(matching_global_rows);
        let columns: Vec<_> = obs_batch
            .columns()
            .iter()
            .map(|col| compute::take(col.as_ref(), &take_indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        RecordBatch::try_new(obs_batch.schema(), columns)?
    } else {
        // Row-sharded obs: read only the shards that contain matching rows.
        materialize_filtered_obs(reader, &obs_shard_ranges, &matching_global_rows)?
    };

    // Step 11b: Filter var metadata to projected genes (in ascending order;
    // Step 11c restores the requested order).
    let filtered_var = if let Some(ref gi) = effective_gene_indices {
        project_var(&var_batch, gi)?
    } else {
        var_batch
    };

    // Step 11c: Restore the caller's requested gene order (F4). Decode +
    // `project_var` produce columns in ascending gene-index order; relabel the
    // CSR column indices and reorder the var rows to the requested order. The
    // reorder is `None` (skipped) when the request was already ascending-unique,
    // so the common path is unchanged. Per-row CSR indices may become
    // non-ascending here — correct for terminal output and matches anndata's
    // `adata[:, names]`; scipy tolerates unsorted indices.
    let filtered_var = if let Some(ref reorder) = reorder {
        for idx in csr.indices.iter_mut() {
            *idx = reorder.sorted_to_output[*idx as usize] as i32;
        }
        let take_idx = UInt32Array::from(reorder.output_to_sorted.clone());
        let columns: Vec<_> = filtered_var
            .columns()
            .iter()
            .map(|col| compute::take(col.as_ref(), &take_idx, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        RecordBatch::try_new(filtered_var.schema(), columns)?
    } else {
        filtered_var
    };

    // Step 12: Return QueryResult
    Ok(QueryResult {
        x: csr,
        obs: filtered_obs,
        var: filtered_var,
        skipped_shards,
        total_shards,
        candidate_shard_rows,
        matched_rows,
        max_value,
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
    }

    fn sample_obs(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
        ]);
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let types: Vec<&str> = (0..n)
            .map(|i| match i % 3 {
                0 => "T cell",
                1 => "B cell",
                _ => "NK cell",
            })
            .collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(types)),
            ],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> RecordBatch {
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
        let path = dir.path().join("test.scx");
        let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_obs {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            let (c0, c1) = if col0 < col1 {
                (col0, col1)
            } else {
                (col1, col0)
            };
            indices.push(c0 as u32);
            indices.push(c1 as u32);
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
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();
        path
    }

    // -----------------------------------------------------------------------
    // F3 Tests: filter_csr_rows
    // -----------------------------------------------------------------------

    #[test]
    fn filter_alternating_rows() {
        // 4-row CSR: keep rows 0 and 2
        let indptr = vec![0i64, 2, 5, 7, 10];
        let indices = vec![0i32, 1, 0, 1, 2, 1, 3, 0, 2, 3];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let keep = vec![true, false, true, false];

        let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();
        assert_eq!(new_ip, vec![0, 2, 4]);
        assert_eq!(new_idx, vec![0, 1, 1, 3]);
        assert_eq!(new_data, vec![1.0, 2.0, 6.0, 7.0]);
    }

    #[test]
    fn filter_keep_all() {
        let indptr = vec![0i64, 2, 5];
        let indices = vec![0i32, 1, 0, 1, 2];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let keep = vec![true, true];

        let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();
        assert_eq!(new_ip, vec![0, 2, 5]);
        assert_eq!(new_idx, indices);
        assert_eq!(new_data, data);
    }

    #[test]
    fn filter_keep_none() {
        let indptr = vec![0i64, 2, 5];
        let indices = vec![0i32, 1, 0, 1, 2];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let keep = vec![false, false];

        let (new_ip, new_idx, new_data) = filter_csr_rows(&indptr, &indices, &data, &keep).unwrap();
        assert_eq!(new_ip, vec![0]);
        assert!(new_idx.is_empty());
        assert!(new_data.is_empty());
    }

    /// A keep mask that disagrees with the CSR row count is a corrupt or
    /// truncated file, not a request to guess.
    ///
    /// This used to be a `debug_assert!` followed by `min(keep_mask.len(),
    /// n_rows)`, so in release — the builds users run — the *shorter* case
    /// silently dropped every row past the end of the mask and returned a
    /// truncated query result. That direction never panicked, which is exactly
    /// why it needed a test rather than an assertion: a wrong answer looks like
    /// an answer.
    #[test]
    fn filter_rejects_a_mask_that_disagrees_with_the_csr() {
        let indptr = vec![0i64, 2, 5, 7];
        let indices = vec![0i32, 1, 0, 1, 2, 1, 3];
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

        // Shorter than the CSR: the silent-truncation direction. Pre-fix this
        // returned Ok with 1 row where 3 were present.
        let short = vec![true];
        let err = filter_csr_rows(&indptr, &indices, &data, &short)
            .expect_err("a short mask must error, not truncate");
        let msg = err.to_string();
        assert!(
            msg.contains('1') && msg.contains('3'),
            "must report both counts: {msg}"
        );

        // Longer than the CSR: would have indexed past the end of indptr.
        let long = vec![true, true, true, true, true];
        assert!(filter_csr_rows(&indptr, &indices, &data, &long).is_err());

        // Control: an exactly-matching mask still filters.
        let ok = vec![true, false, true];
        let (ip, _, _) = filter_csr_rows(&indptr, &indices, &data, &ok).unwrap();
        assert_eq!(ip.len(), 3, "two kept rows");
    }

    // -----------------------------------------------------------------------
    // F2 Tests: Pipeline execution via QueryPipeline::collect()
    // -----------------------------------------------------------------------

    #[test]
    fn collect_no_predicates_returns_all() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        assert_eq!(result.x.n_rows(), 12);
        assert_eq!(result.x.n_cols(), 5);
        assert_eq!(result.obs.num_rows(), 12);
        assert_eq!(result.var.num_rows(), 5);
        assert_eq!(result.total_shards, 1);
    }

    #[test]
    fn collect_filter_obs_returns_subset() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();
        // cells 0, 3, 6, 9 are "T cell" (i % 3 == 0)
        assert_eq!(result.x.n_rows(), 4);
        assert_eq!(result.obs.num_rows(), 4);
        // Check that filtered obs matches
        let cell_types = result
            .obs
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..cell_types.len() {
            assert_eq!(cell_types.value(i), "T cell");
        }
    }

    #[test]
    fn collect_gene_projection() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(vec![0, 3, 7])
            .collect()
            .unwrap();
        assert_eq!(result.x.n_cols(), 3);
        assert_eq!(result.var.num_rows(), 3);
        // Check var metadata has correct gene IDs
        let gene_ids = result
            .var
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_0");
        assert_eq!(gene_ids.value(1), "gene_3");
        assert_eq!(gene_ids.value(2), "gene_7");
    }

    #[test]
    fn collect_gene_projection_preserves_requested_order() {
        // F4: select_genes must return columns in the caller's requested order,
        // not ascending index order.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);

        let requested = vec![7u32, 3, 0];
        let proj = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(requested.clone())
            .collect()
            .unwrap();
        assert_eq!(proj.x.n_cols(), 3);
        assert_eq!(proj.var.num_rows(), 3);

        // var rows follow the requested order.
        let gene_ids = proj
            .var
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_7");
        assert_eq!(gene_ids.value(1), "gene_3");
        assert_eq!(gene_ids.value(2), "gene_0");

        // X columns match: projected column j equals full column requested[j].
        let full = QueryPipeline::open(&path).unwrap().collect().unwrap();
        let dense_full = full.x.to_dense().unwrap(); // row-major, 12×10
        let dense_proj = proj.x.to_dense().unwrap(); // row-major, 12×3
        for r in 0..12 {
            for (j, &g) in requested.iter().enumerate() {
                assert_eq!(
                    dense_proj[r * 3 + j],
                    dense_full[r * 10 + g as usize],
                    "row {r} col {j} (gene {g})"
                );
            }
        }
    }

    #[test]
    fn collect_gene_projection_ascending_unchanged() {
        // An already-ascending-unique request takes the no-reorder fast path:
        // output columns are exactly the requested genes, in order, matching the
        // corresponding columns of the full (unprojected) matrix.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let requested = vec![0u32, 3, 7];
        let proj = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(requested.clone())
            .collect()
            .unwrap();
        assert_eq!(proj.x.n_cols(), 3);

        let full = QueryPipeline::open(&path).unwrap().collect().unwrap();
        let dense_full = full.x.to_dense().unwrap(); // 12×10
        let dense_proj = proj.x.to_dense().unwrap(); // 12×3
        for r in 0..12 {
            for (j, &g) in requested.iter().enumerate() {
                assert_eq!(dense_proj[r * 3 + j], dense_full[r * 10 + g as usize]);
            }
        }
    }

    #[test]
    fn collect_gene_projection_order_with_var_predicate() {
        // PR #242 review: select_genes order must survive intersection with a
        // var predicate that drops one of the requested genes.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(vec![7, 3, 0]) // non-ascending
            .filter_var("gene_id != 'gene_3'")
            .unwrap()
            .collect()
            .unwrap();
        // gene_3 removed; the survivors keep the requested order [7, 0].
        assert_eq!(result.x.n_cols(), 2);
        let gene_ids = result
            .var
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(gene_ids.value(0), "gene_7");
        assert_eq!(gene_ids.value(1), "gene_0");
    }

    #[test]
    fn collect_with_normalize_log1p() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 6, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .with_normalize(1e4)
            .with_log1p()
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 6);
        // Values should be transformed (no longer raw integers)
        // Each row had 2 non-zero values, after normalize+log1p they should be ln(v/sum*1e4 + 1)
        for row in 0..result.x.n_rows() {
            let start = result.x.indptr[row] as usize;
            let end = result.x.indptr[row + 1] as usize;
            for i in start..end {
                assert!(
                    result.x.data[i] > 0.0,
                    "fused ops should produce positive values"
                );
                assert!(
                    result.x.data[i] < 20.0,
                    "ln(10001) ≈ 9.21, values should be reasonable"
                );
            }
        }
    }

    #[test]
    fn collect_empty_result() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // Filter for a cell_type that doesn't exist — but the column exists
        // so the predicate is valid. All cells are T cell, B cell, or NK cell.
        // Use cell_id which is unique to force empty result.
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_id == 'nonexistent'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 0);
        assert_eq!(result.obs.num_rows(), 0);
    }

    #[test]
    fn collect_with_limit() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .limit(3)
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 3);
        assert_eq!(result.obs.num_rows(), 3);
        // matched_rows must reflect the
        // pre-limit Level-2 match count (12 rows match the no-predicate
        // pipeline), not the post-limit returned count.
        assert_eq!(result.matched_rows, 12);
    }

    #[test]
    fn collect_limit_exceeds_total() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 6, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .limit(100)
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 6);
        assert_eq!(result.obs.num_rows(), 6);
        // No truncation occurred — matched_rows must equal returned rows.
        assert_eq!(result.matched_rows, result.x.n_rows());
    }

    #[test]
    fn count_matches_collect_matched_rows() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // No predicate: every cell matches.
        let c = QueryPipeline::open(&path).unwrap().count().unwrap();
        assert_eq!(c.matched_rows, 12);
        let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        assert_eq!(c.matched_rows, result.matched_rows);
        assert_eq!(c.total_shards, result.total_shards);
        assert_eq!(c.skipped_shards, result.skipped_shards);
        assert_eq!(c.candidate_shard_rows, result.candidate_shard_rows);
    }

    #[test]
    fn collect_reports_shard_value_max() {
        // `write_test_file` writes Uint8 counts (row+1)%256 / (row+2)%256 over
        // 12 rows, so the on-disk max is 13 (row 11). collect() must surface it
        // via QueryResult::max_value for the F4 decode-loss guard.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        assert_eq!(result.max_value, 13);
    }

    #[test]
    fn count_with_predicate_matches_collect() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // count() and collect().matched_rows must agree for the same predicate,
        // even though count() never decodes X.
        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap();
        let c = pipeline.count().unwrap();
        let result = pipeline.collect().unwrap();
        assert_eq!(c.matched_rows, result.matched_rows);
        assert_eq!(c.matched_rows, result.x.n_rows());
    }

    #[test]
    fn count_ignores_limit() {
        // CLI6: count() reports the true match count regardless of `limit`.
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let no_limit = QueryPipeline::open(&path).unwrap().count().unwrap();
        let with_limit = QueryPipeline::open(&path)
            .unwrap()
            .limit(3)
            .count()
            .unwrap();
        assert_eq!(no_limit.matched_rows, 12);
        assert_eq!(
            with_limit.matched_rows, no_limit.matched_rows,
            "limit must not change the counted match total"
        );
    }

    #[test]
    fn count_empty_result() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let c = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_id == 'nonexistent'")
            .unwrap()
            .count()
            .unwrap();
        assert_eq!(c.matched_rows, 0);
    }

    #[test]
    fn collect_multiple_filter_obs_and_semantics() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        // cell_type == 'T cell' AND cell_id == 'cell_0' → only cell_0
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .filter_obs("cell_id == 'cell_0'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.x.n_rows(), 1);
        assert_eq!(result.obs.num_rows(), 1);
        let cell_ids = result
            .obs
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(cell_ids.value(0), "cell_0");
    }

    #[test]
    fn collect_obs_row_count_matches_x() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 5);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'B cell'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(result.obs.num_rows(), result.x.n_rows());
    }

    #[test]
    fn collect_var_row_count_matches_x_cols() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(vec![1, 5, 9])
            .collect()
            .unwrap();
        assert_eq!(result.var.num_rows(), result.x.n_cols());
    }

    #[test]
    fn collect_full_pipeline() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 12, 10);
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .select_genes(vec![0, 2, 4, 6, 8])
            .with_normalize(1e4)
            .with_log1p()
            .collect()
            .unwrap();
        // T cell: indices 0, 3, 6, 9 → 4 cells
        assert_eq!(result.x.n_rows(), 4);
        assert_eq!(result.x.n_cols(), 5); // 5 projected genes
        assert_eq!(result.obs.num_rows(), 4);
        assert_eq!(result.var.num_rows(), 5);
    }

    // -----------------------------------------------------------------------
    // FILTER-OBS-OOM follow-ups: full-scan limit-pushdown (#1) and no-match
    // short-circuit (#2). Both run `plan_and_mask` + `materialize` against a
    // *borrowed* pipeline so the per-shard X-decode counter
    // (`read_shard_from_entry`) can be inspected after materialisation.
    // -----------------------------------------------------------------------

    const MS_SHARDS: usize = 4;
    const MS_ROWS_PER_SHARD: usize = 3;

    /// 4 CSR + 4 obs shards of 3 rows each (12 rows). `cell_type` is indexed
    /// and cycles T/B/NK within every shard, so `'T cell'` appears in EVERY
    /// shard (0 % catalog skip → the low-selectivity full-scan path), while a
    /// value like `'Z'` is absent from the predicate index entirely.
    fn write_indexed_multishard_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        let n_obs = (MS_SHARDS * MS_ROWS_PER_SHARD) as u64;
        let n_vars = 4usize;
        let path = dir.path().join("indexed_multishard.scx");

        let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
        let types: Vec<&str> = (0..n_obs as usize)
            .map(|i| match i % MS_ROWS_PER_SHARD {
                0 => "T cell",
                1 => "B cell",
                _ => "NK cell",
            })
            .collect();
        let obs_schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, false),
        ]);
        let obs = RecordBatch::try_new(
            Arc::new(obs_schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(types)),
            ],
        )
        .unwrap();
        let var = sample_var(n_vars);

        let mut header = sample_header(n_obs, n_vars as u64, n_obs * 2);
        header.shard_target_rows = MS_ROWS_PER_SHARD as u32;
        let mut writer = ScxWriter::new(&path, header).unwrap();

        // Sharded obs aligned with the CSR shards.
        for s in 0..MS_SHARDS {
            let rs = (s * MS_ROWS_PER_SHARD) as u64;
            let slice = obs.slice(rs as usize, MS_ROWS_PER_SHARD);
            writer
                .write_obs_shard(s as u32, rs, MS_ROWS_PER_SHARD as u64, n_obs, &slice)
                .unwrap();
        }
        writer.write_var(&var).unwrap();

        // CSR shards: 1 nnz/row.
        let mut csr_row_ranges = Vec::new();
        for s in 0..MS_SHARDS {
            let rs = (s * MS_ROWS_PER_SHARD) as u64;
            let indptr: Vec<u64> = (0..=MS_ROWS_PER_SHARD as u64).collect();
            let indices: Vec<u32> = (0..MS_ROWS_PER_SHARD as u32)
                .map(|i| i % n_vars as u32)
                .collect();
            let values: Vec<u8> = vec![1u8; MS_ROWS_PER_SHARD];
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    rs,
                )
                .unwrap();
            csr_row_ranges.push((rs, rs + MS_ROWS_PER_SHARD as u64));
        }

        let opts = crate::ConversionPredicateIndexOptions {
            index_obs: vec!["cell_type".to_string()],
            index_var: Vec::new(),
            index_preset: None,
            index_auto_threshold: 1000,
        };
        crate::build_and_write_conversion_predicate_indexes(
            &mut writer,
            &obs,
            &var,
            &csr_row_ranges,
            n_vars,
            &opts,
        )
        .unwrap();
        writer.finish().unwrap();
        path
    }

    // Only the `debug_assertions`-gated assertions reference this, so it is
    // dead code in release builds.
    #[cfg(debug_assertions)]
    fn x_decode_count(pipeline: &QueryPipeline) -> u64 {
        pipeline
            .reader()
            .as_any()
            .downcast_ref::<scx_format_io::reader::ScxReader>()
            .expect("local reader")
            .debug_counts()
            .read_shard_from_entry
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Item #1: a low-selectivity predicate matches a row in every shard
    /// (0 % skip), so without limit-pushdown `materialize` would decode all 4
    /// X shards. With `.limit(1)` the first shard alone fills the budget, so
    /// only ONE X shard is decoded.
    #[test]
    fn limit_pushdown_bounds_full_scan_x_decode() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_indexed_multishard_file(&dir);

        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .limit(1);
        let pm = plan_and_mask(&pipeline).unwrap();
        // No catalog skip: 'T cell' is present in every shard.
        assert_eq!(pm.skipped_shards, 0);
        assert_eq!(pm.matched_rows, MS_SHARDS); // one T cell per shard
        let r = materialize(&pipeline, pm).unwrap();

        assert_eq!(r.x.n_rows(), 1);
        assert_eq!(r.obs.num_rows(), 1);
        assert_eq!(r.matched_rows, MS_SHARDS, "pre-limit Level-2 count");
        // The X-decode counter is only incremented under `debug_assertions`
        // (compiled away in release), so assert it only when present.
        #[cfg(debug_assertions)]
        assert_eq!(
            x_decode_count(&pipeline),
            1,
            "limit(1) must decode only the first X shard, not all {MS_SHARDS}"
        );
    }

    /// The over-fix guard, end to end: a file written by the normal conversion
    /// path must still be recognised as carrying a complete vocabulary, and a
    /// filter naming a value the file does not contain must still short-circuit
    /// without reading a single obs shard.
    ///
    /// Gating the short-circuit on completeness is only worth doing if
    /// completeness is the ordinary case. If this goes red, the guard has
    /// turned every miss on every file into a full obs scan.
    #[test]
    fn a_converted_file_carries_a_complete_vocabulary_and_still_short_circuits() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_indexed_multishard_file(&dir);

        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'Z'")
            .unwrap();
        let plan = build_plan(&pipeline).unwrap();
        let dict = plan
            .category_dicts
            .get(&scx_format_io::column_name_hash("cell_type"))
            .expect("cell_type is indexed");
        assert!(
            dict.complete,
            "every CSR shard of a freshly converted file carries the column's \
             CategoryBitset, so its vocabulary is the complete value set"
        );

        let pm = plan_and_mask(&pipeline).unwrap();
        assert_eq!(
            pm.skipped_shards, MS_SHARDS,
            "'Z' is absent from a complete vocabulary → prune every shard"
        );
        assert_eq!(pm.matched_rows, 0);
    }

    /// The catalog-derived signal itself.
    ///
    /// `derive_shard_column_stats` emits a `CategoryBitset` for *every* shard of
    /// an indexed categorical column — an all-zero one where the column has no
    /// values in that shard — so a shard without one was never seen by the
    /// build that produced the vocabulary. That is what `append` without
    /// `--index-obs` leaves behind.
    #[test]
    fn a_shard_without_the_columns_bitset_makes_the_vocabulary_incomplete() {
        use scx_format_io::catalog::{ColumnStat, FullCatalogEntry, ShardStats};
        use scx_format_io::section::SectionType;

        let hash = scx_format_io::column_name_hash("cell_type");
        let shard = |column_stats: Vec<ColumnStat>| FullCatalogEntry {
            name: "X_shard".to_string(),
            offset: 4352,
            length: 10,
            section_type: SectionType::CsrShard,
            checksum: [0; 32],
            modality_id: 0,
            stats: Some(ShardStats {
                row_start: 0,
                row_end: 10,
                col_start: 0,
                col_end: 0,
                nnz: 1,
                value_min: 1,
                value_max: 1,
                value_sum: 1,
                n_indexed_columns: column_stats.len() as u8,
                column_stats,
            }),
        };
        let bitset = || ColumnStat::CategoryBitset {
            column_name_hash: hash,
            bitset: vec![0b0000_0001],
        };

        let indexed = shard(vec![bitset()]);
        let appended = shard(Vec::new());
        // A shard carrying stats for some *other* indexed column is just as
        // uncovered for this one.
        let other_column = shard(vec![ColumnStat::CategoryBitset {
            column_name_hash: scx_format_io::column_name_hash("tissue"),
            bitset: vec![0b0000_0001],
        }]);
        // `ColumnStat::column_name_hash` answers for `MinMax` too, so a numeric
        // stat under this column's hash satisfies a hash-only test — and would
        // let a numeric column license a categorical vocabulary.
        let numeric_stat = shard(vec![ColumnStat::MinMax {
            column_name_hash: hash,
            min: 0.0,
            max: 1.0,
        }]);
        // Sized for a nine-value vocabulary: a different index build.
        let wrong_len = shard(vec![ColumnStat::CategoryBitset {
            column_name_hash: hash,
            bitset: vec![0b0000_0001, 0b0000_0000],
        }]);

        // One value → one byte, which is what `bitset()` carries.
        let complete = |shards: &[&FullCatalogEntry], n_values: usize| {
            collect_bitset_coverage(shards)
                .get(&hash)
                .is_some_and(|c| c.covers(shards.len(), n_values))
        };

        assert!(complete(&[&indexed, &indexed], 1));
        assert!(
            !complete(&[&indexed, &appended], 1),
            "the appended shard was never seen by the index build"
        );
        assert!(!complete(&[&other_column], 1));
        assert!(
            !complete(&[&numeric_stat], 1),
            "a MinMax stat is not evidence that a categorical vocabulary is complete"
        );
        assert!(
            !complete(&[&wrong_len], 1),
            "a bitset sized for another vocabulary means the stats and the index \
             section came from different builds"
        );
        assert!(
            !complete(&[&indexed, &wrong_len], 1),
            "one disagreeing shard is enough — bit i no longer means entry i"
        );
        assert!(
            !complete(&[&indexed], 9),
            "nine values need two bytes; this shard carries one"
        );
        assert!(
            !complete(&[], 1),
            "with nothing to check against, a coverage claim is vacuous"
        );

        // Counting stat *records* rather than distinct shards lets one shard's
        // surplus pay for another shard's absence. Two bitsets under the hash on
        // shard A and none on shard B still totals two — and shard B, which
        // nothing covers, is what the vocabulary would then be claiming to
        // describe. `&[&indexed, &indexed]` above does not catch this: it
        // repeats an entry *reference*, which is two shards each carrying one.
        let doubled = shard(vec![bitset(), bitset()]);
        assert!(
            !complete(&[&doubled, &appended], 1),
            "a duplicate bitset in one shard must not stand in for a shard that \
             carries none"
        );
        assert!(
            !complete(&[&doubled, &indexed], 1),
            "two bitsets for one column in a single shard is a malformed \
             catalog, not extra evidence"
        );
    }

    /// Proof that *which* modality's shards you scan is load-bearing.
    ///
    /// `csr_shards_for_modality(0)` filters to modality 0 — it is **not** the
    /// flattened all-modality list, though the test that appears to prove it
    /// (`csr_shards_for_modality_0_matches_shards_sorted_single_modality`) is a
    /// single-modality fixture. On a multimodal catalog the two lists give
    /// opposite verdicts, which is what this pins.
    ///
    /// ⚠️ **It does not guard `build_plan`'s call site.** It scans the modality
    /// itself rather than going through `build_plan`, so reverting that call to
    /// a hardcoded `0` would leave this green. Closing that needs a multimodal
    /// fixture file with a predicate index; what this test buys is that the
    /// argument matters at all, so the revert would be a behaviour change
    /// rather than a no-op.
    #[test]
    fn completeness_follows_the_queried_modality_not_modality_zero() {
        use scx_format_io::catalog::{ColumnStat, FullCatalog, FullCatalogEntry, ShardStats};
        use scx_format_io::section::SectionType;

        let hash = scx_format_io::column_name_hash("cell_type");
        let shard =
            |modality_id: u8, row_start: u64, column_stats: Vec<ColumnStat>| FullCatalogEntry {
                name: format!("X_shard_m{modality_id}_{row_start}"),
                offset: 4352 + row_start,
                length: 10,
                section_type: SectionType::CsrShard,
                checksum: [0; 32],
                modality_id,
                stats: Some(ShardStats {
                    row_start,
                    row_end: row_start + 10,
                    col_start: 0,
                    col_end: 0,
                    nnz: 1,
                    value_min: 1,
                    value_max: 1,
                    value_sum: 1,
                    n_indexed_columns: column_stats.len() as u8,
                    column_stats,
                }),
            };
        let bitset = vec![ColumnStat::CategoryBitset {
            column_name_hash: hash,
            bitset: vec![0b0000_0001],
        }];

        // Modality 0 is fully indexed; modality 1's shards carry no stats.
        let catalog = FullCatalog {
            catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            n_obs: 20,
            entries: vec![
                shard(0, 0, bitset.clone()),
                shard(0, 10, bitset.clone()),
                shard(1, 0, Vec::new()),
                shard(1, 10, Vec::new()),
            ],
            data_generation: 0,
            csc_build_generation: 0,
        };

        let covers = |modality_id: u8| {
            let shards = scan_shards(&catalog, modality_id);
            collect_bitset_coverage(&shards)
                .get(&hash)
                .is_some_and(|c| c.covers(shards.len(), 1))
        };

        assert!(covers(0), "modality 0's shards are all indexed");
        assert!(
            !covers(1),
            "a modality-1 query must judge completeness over modality 1's \
             shards — reading modality 0's would license pruning shards nothing \
             was checked against"
        );
    }

    /// Item #1 control: with no limit the full-scan path decodes every
    /// candidate shard and the pre-limit count equals the assembled rows.
    #[test]
    fn no_limit_full_scan_decodes_all_shards() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_indexed_multishard_file(&dir);

        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap();
        let pm = plan_and_mask(&pipeline).unwrap();
        let r = materialize(&pipeline, pm).unwrap();

        assert_eq!(r.x.n_rows(), MS_SHARDS);
        // Debug-only counter (see `limit_pushdown_bounds_full_scan_x_decode`).
        #[cfg(debug_assertions)]
        assert_eq!(
            x_decode_count(&pipeline),
            MS_SHARDS as u64,
            "unlimited query decodes every candidate shard"
        );
    }

    /// Item #2: an equality whose value is absent from the (indexed) column's
    /// global dictionary must skip every shard and touch NO obs/X metadata.
    #[test]
    fn no_match_indexed_value_short_circuits_without_reads() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_indexed_multishard_file(&dir);

        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'Z'")
            .unwrap();
        let pm = plan_and_mask(&pipeline).unwrap();
        assert_eq!(pm.total_shards, MS_SHARDS);
        assert_eq!(
            pm.skipped_shards, MS_SHARDS,
            "an absent indexed value skips every shard"
        );
        assert_eq!(pm.matched_rows, 0);
        let r = materialize(&pipeline, pm).unwrap();
        assert_eq!(r.x.n_rows(), 0);
        assert_eq!(r.obs.num_rows(), 0);
        assert_eq!(r.skipped_shards, MS_SHARDS);

        // Read counters are only incremented under `debug_assertions` (compiled
        // away in release), so inspect them only when present.
        #[cfg(debug_assertions)]
        {
            let reader = pipeline
                .reader()
                .as_any()
                .downcast_ref::<scx_format_io::reader::ScxReader>()
                .unwrap();
            let counts = reader.debug_counts();
            use std::sync::atomic::Ordering::Relaxed;
            assert_eq!(
                counts.read_obs.load(Relaxed),
                0,
                "short-circuit must not materialise the full obs table"
            );
            // The full obs scan is avoided entirely. `materialize` reads at most
            // one obs shard as the 0-row schema template for the empty result (a
            // bounded, MB-scale read) — never the whole file. `count()` skips
            // materialise and reads zero (asserted in the integration test).
            assert!(
                counts.read_obs_shard.load(Relaxed) < MS_SHARDS as u64,
                "short-circuit must not scan all obs shards"
            );
            assert_eq!(
                counts.read_shard_from_entry.load(Relaxed),
                0,
                "short-circuit must not decode any X shard"
            );
        }
    }
}
