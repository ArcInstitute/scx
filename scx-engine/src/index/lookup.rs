//! Query-time lookups against a loaded predicate index.
//!
//! These consume the index for row *selection* (the row-set fast path in
//! `collect::mask`), not the catalog-level shard *pruning* that
//! `pushdown::prune_shards_by_catalog_with_dict` does from `column_stats`.

use super::{IndexKind, IndexedColumn, PredicateIndex, ShardRange};

/// Whether `index` covers every obs row, i.e. is safe to use for row-set
/// pushdown on this file.
///
/// The file-scope predicate index can go **stale** after `append`, which adds
/// new obs shards at the tail without rewriting the index (CLAUDE.md, cloud
/// query notes). A stale index would silently miss the appended rows on the
/// row-set fast path. This guard detects that case: the index covers all of obs
/// only when the maximum global row it references reaches `n_obs`.
///
/// Conservative by design: if the trailing obs rows happen to be null in *every*
/// indexed column they aren't referenced by any entry, so a fresh index can also
/// report `< n_obs` and we fall back to the full obs scan. That fallback is
/// correct (just slower) — atlas builds populate the indexed columns, so the
/// fast path applies in practice.
pub fn index_covers_all_obs(
    index: &PredicateIndex,
    obs_shard_ranges: &[(u32, u64, u64)],
    n_obs: u64,
) -> bool {
    if index.columns.is_empty() {
        return false;
    }
    index.max_covered_global_row(obs_shard_ranges) >= n_obs
}

impl PredicateIndex {
    // ------------------------------------------------------------------------
    // Query-time lookups (row-set pushdown). These consume the index for row
    // *selection*, not just the catalog-level shard *pruning* that
    // `build_category_dicts` / `prune_shards_by_catalog_with_dict` do today.
    // ------------------------------------------------------------------------

    /// The shard ranges containing `value` in the indexed categorical column
    /// `column`, or `None` if `column` is not an indexed categorical.
    ///
    /// - `None` => column not indexed as a categorical → the caller treats the
    ///   predicate as residual (decode + mask).
    /// - `Some(&[])` => column is indexed but `value` is absent → the predicate
    ///   matches no rows (an exact, empty row-set). This is *not* residual.
    ///
    /// `CategoricalIndex::entries` is sorted lexicographically by value
    /// (`build_categorical_index` builds it from a `BTreeMap`), so this is a
    /// binary search.
    pub fn categorical_eq(&self, column: &str, value: &str) -> Option<&[ShardRange]> {
        for col in &self.columns {
            if let IndexedColumn::Categorical(cat) = col {
                if cat.column_name == column {
                    return Some(
                        match cat
                            .entries
                            .binary_search_by(|e| e.value.as_str().cmp(value))
                        {
                            Ok(pos) => &cat.entries[pos].shard_ranges,
                            Err(_) => &[],
                        },
                    );
                }
            }
        }
        None
    }

    /// Which kind of index (if any) covers `column`.
    pub fn indexed_kind(&self, column: &str) -> Option<IndexKind> {
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) if cat.column_name == column => {
                    return Some(IndexKind::Categorical)
                }
                IndexedColumn::Numeric(num) if num.column_name == column => {
                    return Some(IndexKind::Numeric)
                }
                _ => {}
            }
        }
        None
    }

    /// The maximum global row covered by any shard range in this index, mapped
    /// through `obs_shard_ranges` (`(shard_idx, row_start, row_end)` sorted by
    /// `shard_idx`). Used by [`index_covers_all_obs`].
    fn max_covered_global_row(&self, obs_shard_ranges: &[(u32, u64, u64)]) -> u64 {
        let mut max_end = 0u64;
        let mut consider = |shard_id: u32, row_end: u32| {
            if let Ok(pos) = obs_shard_ranges.binary_search_by_key(&shard_id, |(idx, _, _)| *idx) {
                let (_, shard_row_start, shard_row_end) = obs_shard_ranges[pos];
                let g = (shard_row_start + row_end as u64).min(shard_row_end);
                if g > max_end {
                    max_end = g;
                }
            }
        };
        for col in &self.columns {
            match col {
                IndexedColumn::Categorical(cat) => {
                    for entry in &cat.entries {
                        for sr in &entry.shard_ranges {
                            consider(sr.shard_id, sr.row_end);
                        }
                    }
                }
                IndexedColumn::Numeric(num) => {
                    for leaf in &num.leaf_pages {
                        for e in &leaf.entries {
                            consider(e.shard_id, e.row_end);
                        }
                    }
                }
            }
        }
        max_end
    }
}
