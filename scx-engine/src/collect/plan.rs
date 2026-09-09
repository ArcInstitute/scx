//! The execution plan: Level-1 (catalog) shard pruning and the category
//! dictionaries both pushdown levels share.
//!
//! Level 1 answers "can this whole shard be excluded?" from the `column_stats`
//! on each catalog entry. Level 2 — resolving *which rows* match from the
//! predicate index — lives in [`super::mask`].

use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use crate::error::Result;
use crate::index::{IndexedColumn, PredicateIndex};
use crate::pipeline::QueryPipeline;
use crate::predicate::Predicate;
use crate::pushdown::{prune_shards_by_catalog_with_dict, CategoryDictionaries, ShardCandidate};

use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::DeletionVectors;

/// Internal execution plan built from a QueryPipeline.
pub(crate) struct ExecutionPlan {
    pub(crate) candidate_shards: Vec<ShardCandidate>,
    pub(crate) obs_predicates: Vec<Predicate>,
    /// Used for var-level filtering during execution (see `execute()`).
    pub(crate) var_predicates: Vec<Predicate>,
    pub(crate) gene_indices: Option<Vec<u32>>,
    pub(crate) normalize: Option<f64>,
    pub(crate) log1p: bool,
    pub(crate) limit: Option<usize>,
    pub(crate) deletion_vectors: Option<DeletionVectors>,
    /// The deserialized obs predicate index, retained for **row-set pushdown**
    /// (row selection), not just the category dictionaries used for
    /// catalog-level shard pruning. `None` when the file has no obs predicate
    /// index. See `mask::try_rowset_mask`.
    pub(crate) obs_predicate_index: Option<PredicateIndex>,
    /// Per-column category vocabularies and whether each may be trusted as
    /// complete. Built once in [`build_plan`] and used by **both** pushdown
    /// levels — Level-1 for catalog pruning, Level-2 for row-set resolution —
    /// so the two cannot end up trusting the index to different degrees.
    pub(crate) category_dicts: CategoryDictionaries,
}

/// Build an execution plan from a QueryPipeline.
///
/// Runs catalog-level shard pruning and loads predicate indexes.
///
/// `sorted_shards` is the caller's already-derived
/// [`scan_shards`] list for `pipeline`'s modality; taking it rather than
/// re-deriving it is what makes the `shard_idx` positions in the returned
/// `candidate_shards` indices into a list the caller still holds.
pub(crate) fn build_plan(
    pipeline: &QueryPipeline,
    sorted_shards: &[&FullCatalogEntry],
) -> Result<ExecutionPlan> {
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
    // it is handed the same `sorted_shards` slice below, so the two cannot
    // disagree about which shards those are — so the claim "every shard being
    // pruned was covered by this vocabulary's build" is checked against those
    // shards and no others. (The index's own
    // `shard_id` space does not enter into it: the test is per-entry, "does
    // this shard carry a `CategoryBitset` for this column hash".)
    let category_dicts = build_category_dicts(&obs_predicate_index, sorted_shards);
    let dicts_ref = if category_dicts.is_empty() {
        None
    } else {
        Some(&category_dicts)
    };

    // Catalog-level shard pruning (B1), now with category dictionary support,
    // over the caller's shard list — already scoped to the pipeline's modality
    // (== shards_sorted() for the default single-modality axis).
    let candidate_shards = prune_shards_by_catalog_with_dict(
        sorted_shards,
        pipeline.obs_predicates(),
        pipeline.deletion_vectors().as_ref(),
        dicts_ref,
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
pub(crate) struct BitsetCoverage {
    /// **Distinct shards** carrying a `CategoryBitset` for this column — not
    /// the number of such stat records. The two differ exactly when one shard
    /// carries the column twice, and counting records would then let that
    /// shard's surplus pay for another shard's absence.
    pub(crate) shards: usize,
    /// Byte length of the first bitset seen.
    pub(crate) len: usize,
    /// Cleared once two shards disagree about that length.
    pub(crate) consistent: bool,
    /// Set when one shard carried this column more than once. A malformed
    /// catalog, not extra evidence: nothing says which of the two bitsets the
    /// dictionary's bit positions belong to.
    pub(crate) duplicated: bool,
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
    pub(crate) fn covers(&self, n_shards: usize, n_values: usize) -> bool {
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
pub(crate) fn collect_bitset_coverage(
    pruned_shards: &[&FullCatalogEntry],
) -> HashMap<u64, BitsetCoverage> {
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
///
/// Takes the **positions** rather than a `modality_id` so that the filter and
/// sort behind them run once per pipeline
/// ([`QueryPipeline::csr_shard_positions`](crate::QueryPipeline::csr_shard_positions))
/// and every consumer provably walks the same list. `positions` must come from
/// `FullCatalog::csr_shard_indices` on this same catalog — it is the ordering
/// rule `csr_shards_for_modality` is itself expressed in terms of.
///
/// Only [`plan_and_mask`](super::execute::plan_and_mask) resolves the whole
/// list, because the pruner and the masker each take a slice of it. Everything
/// that needs one shard at a time goes through
/// [`QueryPipeline::csr_shard_entry`](crate::QueryPipeline::csr_shard_entry)
/// instead of allocating this vector again.
pub(crate) fn scan_shards<'a>(
    catalog: &'a scx_format_io::FullCatalog,
    positions: &[usize],
) -> Vec<&'a FullCatalogEntry> {
    // Positions from a *different* catalog would index out of range here (or,
    // worse, land on an entry of another section type). The only producer is
    // `QueryPipeline::csr_shard_positions`, which derives them from the same
    // reader every caller passes — so this states the coupling rather than
    // defending against a reachable input.
    debug_assert!(
        positions.iter().all(|&i| catalog
            .entries
            .get(i)
            .is_some_and(|e| e.section_type == scx_format_io::section::SectionType::CsrShard)),
        "shard positions do not index this catalog's CSR shards"
    );
    positions.iter().map(|&i| &catalog.entries[i]).collect()
}
