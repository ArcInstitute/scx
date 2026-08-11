// Catalog-level and index-level shard pruning for predicate pushdown.
//
// Level 1 (catalog-level): Use per-shard ShardStats (docs/format.md (Dual Catalog)) to skip
// shards whose column stats prove no rows can match the predicate.
//
// Level 2 (index-level): Use PredicateIndex (docs/format.md (Predicate Indexes)) to narrow row
// ranges within shards. Implemented in Phase C.

use std::collections::HashMap;
use std::ops::Range;

use scx_format_io::catalog::{ColumnStat, FullCatalog};
use scx_format_io::column_name_hash;
use scx_format_io::DeletionVectors;

use crate::predicate::{Predicate, ScalarValue};

/// One column's category vocabulary, as recorded by the obs predicate index,
/// paired with whether that vocabulary may be trusted as the column's
/// **complete** value set over the shards being pruned.
///
/// The two are kept together deliberately. Level-1 makes a *global* claim from
/// this *local* artifact — "the predicate value is absent from `values`, so it
/// exists in no shard, so prune everything" — and that claim is only sound when
/// the index that produced `values` covered every shard the pruner will test.
/// Handing a caller the vocabulary without the bit that says whether it is
/// complete is how that inference came to be made unconditionally.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CategoryDictionary {
    /// Category values, sorted ascending. A value's position is its bit index
    /// in [`ColumnStat::CategoryBitset`] — the convention
    /// `scx_engine::derive_shard_column_stats` writes.
    pub values: Vec<String>,
    /// Whether `values` is the complete value set across every shard that
    /// Level-1 may prune. See [`CategoryDictionaries::insert`].
    pub complete: bool,
}

/// Dictionaries keyed by `column_name_hash`, used to resolve `Utf8` predicate
/// values to [`ColumnStat::CategoryBitset`] bit positions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CategoryDictionaries {
    by_column: HashMap<u64, CategoryDictionary>,
}

impl CategoryDictionaries {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record `values` for `column_name_hash`.
    ///
    /// `complete` must be `true` only when every shard the caller will pass to
    /// [`prune_shards_by_catalog_with_dict`] was covered by the index build
    /// that produced `values`. There is no default and no setter: a caller
    /// cannot supply a vocabulary without stating what it is worth.
    ///
    /// An **empty** vocabulary is never complete, whatever the caller says. It
    /// cannot tell "this column genuinely has no values" apart from "this build
    /// recorded none of them", so it proves nothing about any value — and the
    /// second reading is not hypothetical. A file written before integer-valued
    /// categoricals were classified carries an entry-less `CategoricalIndex`
    /// for such a column, and `derive_shard_column_stats` gives every shard a
    /// zero-length `CategoryBitset` for it, so every shard *is* covered and the
    /// coverage signal alone would call it complete. `batch == '1'` on that
    /// file would then prune every shard and return nothing, with no error and
    /// no warning — the type mismatch that would reject a string literal
    /// against an integer column lives in the residual evaluator, which never
    /// runs once Level-1 has short-circuited.
    ///
    /// Refusing costs a full obs scan on an all-null indexed column. Allowing
    /// costs a silent zero-row answer on a file full of data.
    pub fn insert(&mut self, column_name_hash: u64, values: Vec<String>, complete: bool) {
        let complete = complete && !values.is_empty();
        self.by_column
            .insert(column_name_hash, CategoryDictionary { values, complete });
    }

    pub fn get(&self, column_name_hash: &u64) -> Option<&CategoryDictionary> {
        self.by_column.get(column_name_hash)
    }

    pub fn is_empty(&self) -> bool {
        self.by_column.is_empty()
    }

    /// Whether a *categorical* predicate on `column` may be resolved from the
    /// index at all. `true` for a column with no dictionary here: it is not an
    /// indexed categorical, so the caller's own lookup will decline it.
    /// `false` only for a column that IS indexed and whose vocabulary cannot be
    /// trusted as complete.
    pub fn vocabulary_is_usable(&self, column: &str) -> bool {
        self.by_column
            .get(&column_name_hash(column))
            .is_none_or(|d| d.complete)
    }
}

/// A shard that *may* contain matching rows after catalog-level pruning.
#[derive(Debug, Clone)]
pub struct ShardCandidate {
    /// Index into `FullCatalog.shards_sorted()` result.
    pub shard_idx: usize,
    /// Row mask within shard (None = all rows are candidates).
    pub row_mask: Option<Vec<Range<u32>>>,
}

/// Determine which shards may contain matching rows based on catalog-level
/// statistics. Shards that definitely don't match are excluded.
///
/// This is the "cheap" level 1 pushdown (docs/api.md (Query engine, optimizations)) that avoids reading
/// any shard data. When `n_indexed_columns == 0` (Phase 1 files),
/// all shards pass through to post-read filtering.
///
/// If `deletion_vectors` is provided, fully-deleted shards are excluded
/// and partially-deleted shards retain deletion info.
pub fn prune_shards_by_catalog(
    catalog: &FullCatalog,
    predicates: &[Predicate],
    deletion_vectors: Option<&DeletionVectors>,
) -> Vec<ShardCandidate> {
    prune_shards_by_catalog_with_dict(catalog, predicates, deletion_vectors, None, 0)
}

/// Like `prune_shards_by_catalog`, but accepts an optional category dictionary
/// for resolving `Utf8` predicate values against `CategoryBitset` column stats,
/// and a `modality_id` scoping the shard list. `modality_id == 0` is the global
/// / single-modality axis (identical to the legacy `shards_sorted()` view);
/// `>= 1` prunes only that modality's CSR shards. `shard_idx` in the returned
/// candidates indexes the modality-scoped shard list — callers MUST index the
/// same [`crate::collect::scan_shards`] list downstream.
pub fn prune_shards_by_catalog_with_dict(
    catalog: &FullCatalog,
    predicates: &[Predicate],
    deletion_vectors: Option<&DeletionVectors>,
    category_dicts: Option<&CategoryDictionaries>,
    modality_id: u8,
) -> Vec<ShardCandidate> {
    let sorted_shards = crate::collect::scan_shards(catalog, modality_id);

    let mut candidates = Vec::with_capacity(sorted_shards.len());

    for (shard_idx, entry) in sorted_shards.iter().enumerate() {
        let stats = match &entry.stats {
            Some(s) => s,
            // No stats → can't prune, include as candidate
            None => {
                candidates.push(ShardCandidate {
                    shard_idx,
                    row_mask: None,
                });
                continue;
            }
        };

        // B3: Check deletion vectors — if entire shard is deleted, skip it.
        // v2 deletions are global obs rows, so count those falling in the
        // shard's [row_start, row_end).
        if let Some(dv) = deletion_vectors {
            let shard_n_rows = stats.row_end - stats.row_start;
            if dv.deleted_in_range(stats.row_start, stats.row_end) >= shard_n_rows {
                // All rows deleted → skip this shard entirely
                continue;
            }
        }

        // Check each predicate against column stats
        let mut excluded = false;

        if !stats.column_stats.is_empty() {
            for pred in predicates {
                if can_exclude_shard(pred, &stats.column_stats, category_dicts) {
                    excluded = true;
                    break;
                }
            }
        }
        // When n_indexed_columns == 0, no column stats → all shards pass through

        if !excluded {
            candidates.push(ShardCandidate {
                shard_idx,
                row_mask: None,
            });
        }
    }

    candidates
}

/// Check if a single predicate can definitively exclude a shard based on
/// its column stats. Returns true if the shard can be skipped.
fn can_exclude_shard(
    predicate: &Predicate,
    column_stats: &[ColumnStat],
    category_dicts: Option<&CategoryDictionaries>,
) -> bool {
    match predicate {
        Predicate::Eq(col, val) => {
            let hash = column_name_hash(col);
            for cs in column_stats {
                if cs.column_name_hash() != hash {
                    continue;
                }
                match cs {
                    ColumnStat::MinMax { min, max, .. } => {
                        // For numeric equality: skip if value outside [min, max].
                        // Eq is provably safe from false-prunes under monotonic
                        // f64 rounding, but guard it uniformly with the Gt/Ge/
                        // Lt/Le/In branches for defense-in-depth (SCX-003).
                        if let Some((v, is_int)) = scalar_to_f64_typed(val) {
                            if prune_exact(v, *min, is_int)
                                && prune_exact(v, *max, is_int)
                                && (v < *min || v > *max)
                            {
                                return true;
                            }
                        }
                    }
                    ColumnStat::CategoryBitset {
                        column_name_hash: cnh,
                        bitset,
                    } => {
                        // Resolve the predicate value to a bit index.
                        //
                        // Only a `Utf8` value can be resolved, and only
                        // through the category dictionary. An integer
                        // literal is a **value**, never a bit index: the
                        // bitset is positional over the sorted category
                        // list, so reading `batch == 2` as "bit 2" prunes on
                        // a numeric coincidence. Worse, a column indexed
                        // by an earlier version carries an entry-less
                        // `CategoricalIndex`, whose derived bitset is
                        // zero-length — so every shard took the
                        // out-of-range → "absent" branch below and the query
                        // returned nothing at all.
                        let dict = category_dicts.and_then(|dicts| dicts.get(cnh));
                        if !bitset_matches_dictionary(bitset, dict) {
                            continue;
                        }
                        let bit_index = match val {
                            ScalarValue::Utf8(s) => {
                                // Look up string value in category dictionary.
                                match dict {
                                    Some(dict) => match dict.values.binary_search(s) {
                                        Ok(idx) => Some(idx),
                                        // Absent from the dictionary. That is
                                        // proof the value exists in NO shard —
                                        // and so licenses excluding every one
                                        // of them, which is what lets a
                                        // no-match equality short-circuit
                                        // instead of scanning all obs
                                        // metadata — but only when the
                                        // vocabulary is the column's *complete*
                                        // value set. Absent from a partial
                                        // vocabulary means nothing at all, and
                                        // returning `true` there is a silent
                                        // zero-row answer on a file where every
                                        // row matches.
                                        Err(_) => return dict.complete,
                                    },
                                    // No dictionary for this column (not indexed)
                                    // → cannot resolve the value; stay
                                    // conservative and do not exclude.
                                    None => None,
                                }
                            }
                            _ => None,
                        };
                        if let Some(idx) = bit_index {
                            let byte_idx = idx / 8;
                            let bit_idx = idx % 8;
                            if byte_idx < bitset.len() {
                                if bitset[byte_idx] & (1 << bit_idx) == 0 {
                                    return true; // Category not present in shard
                                }
                            } else {
                                return true; // Index out of bitset range
                            }
                        }
                    }
                }
            }
            false
        }
        Predicate::Gt(col, val) | Predicate::Ge(col, val) => {
            let hash = column_name_hash(col);
            if let Some((v, is_int)) = scalar_to_f64_typed(val) {
                for cs in column_stats {
                    if cs.column_name_hash() != hash {
                        continue;
                    }
                    if let ColumnStat::MinMax { max, .. } = cs {
                        // Gt: skip if col_max <= v; Ge: skip if col_max < v
                        // Refuse to prune when the f64 comparison is not exact
                        // for an integer predicate (SCX-003): a bound rounded
                        // inward could false-prune a matching shard.
                        if !prune_exact(v, *max, is_int) {
                            continue;
                        }
                        let skip = match predicate {
                            Predicate::Gt(..) => *max <= v,
                            Predicate::Ge(..) => *max < v,
                            _ => unreachable!(),
                        };
                        if skip {
                            return true;
                        }
                    }
                }
            }
            false
        }
        Predicate::Lt(col, val) | Predicate::Le(col, val) => {
            let hash = column_name_hash(col);
            if let Some((v, is_int)) = scalar_to_f64_typed(val) {
                for cs in column_stats {
                    if cs.column_name_hash() != hash {
                        continue;
                    }
                    if let ColumnStat::MinMax { min, .. } = cs {
                        // Lt: skip if col_min >= v; Le: skip if col_min > v
                        // See SCX-003 note on the Gt/Ge branch above.
                        if !prune_exact(v, *min, is_int) {
                            continue;
                        }
                        let skip = match predicate {
                            Predicate::Lt(..) => *min >= v,
                            Predicate::Le(..) => *min > v,
                            _ => unreachable!(),
                        };
                        if skip {
                            return true;
                        }
                    }
                }
            }
            false
        }
        Predicate::Ne(_, _) => {
            // Ne can't prune: even if the only value in the shard equals the
            // predicate value, there could be nulls/other rows. Conservative.
            false
        }
        Predicate::In(col, values) => {
            // For In: exclude shard only if ALL values can be excluded.
            // For numeric MinMax: exclude if NONE of the In values overlap [min, max].
            let hash = column_name_hash(col);
            for cs in column_stats {
                if cs.column_name_hash() != hash {
                    continue;
                }
                match cs {
                    ColumnStat::MinMax { min, max, .. } => {
                        // A value is only counted as "outside" when its f64
                        // comparison against both bounds is exact (SCX-003);
                        // otherwise it is treated as possibly-inside so the
                        // shard is not pruned. Require at least one convertible
                        // (numeric) value so an all-non-numeric `In` list does
                        // not prune via a vacuously-true `all()`.
                        let numeric: Vec<(f64, bool)> =
                            values.iter().filter_map(scalar_to_f64_typed).collect();
                        let all_outside = !numeric.is_empty()
                            && numeric.iter().all(|&(v, is_int)| {
                                prune_exact(v, *min, is_int)
                                    && prune_exact(v, *max, is_int)
                                    && (v < *min || v > *max)
                            });
                        if all_outside {
                            return true;
                        }
                    }
                    ColumnStat::CategoryBitset {
                        column_name_hash: cnh,
                        bitset,
                    } => {
                        // For category In: exclude if none of the values have
                        // their bit set. As in the `Eq` arm, only a `Utf8`
                        // member resolves — an integer member is a value, not
                        // an ordinal, and treating it as one lets an
                        // entry-less legacy index exclude every shard.
                        let dict = category_dicts.and_then(|dicts| dicts.get(cnh));
                        if !bitset_matches_dictionary(bitset, dict) {
                            continue;
                        }
                        let all_absent = values.iter().all(|v| {
                            let resolved = match v {
                                ScalarValue::Utf8(s) => dict.map(|d| d.values.binary_search(s)),
                                _ => None,
                            };
                            let bit_index = match resolved {
                                Some(Ok(idx)) => Some(idx),
                                // Absent from the dictionary. Under a
                                // *complete* vocabulary that is proof the
                                // member is in no shard, so it counts as
                                // absent here — the same inference the `Eq`
                                // arm makes, which is why the two arms used to
                                // disagree about one predicate.
                                Some(Err(_)) => {
                                    return dict.is_some_and(|d| d.complete);
                                }
                                // No dictionary, or a non-`Utf8` member: a
                                // numeric literal against a string bitset is
                                // unresolvable, not provably absent.
                                None => None,
                            };
                            match bit_index {
                                Some(idx) => {
                                    let byte_idx = idx / 8;
                                    let bit_idx = idx % 8;
                                    if byte_idx < bitset.len() {
                                        bitset[byte_idx] & (1 << bit_idx) == 0
                                    } else {
                                        true // out of range = absent
                                    }
                                }
                                None => false, // can't resolve → conservative
                            }
                        });
                        if all_absent {
                            return true;
                        }
                    }
                }
            }
            false
        }
        Predicate::And(left, right) => {
            // AND: if EITHER side excludes the shard, the whole AND excludes it
            can_exclude_shard(left, column_stats, category_dicts)
                || can_exclude_shard(right, column_stats, category_dicts)
        }
        Predicate::Or(left, right) => {
            // OR: both sides must exclude the shard to prune it
            can_exclude_shard(left, column_stats, category_dicts)
                && can_exclude_shard(right, column_stats, category_dicts)
        }
        Predicate::Not(inner) => {
            // NOT is hard to push down in general. Conservative: don't prune.
            // Exception: Not(Eq) on numeric with value == only value in range
            // is too niche. Just skip.
            let _ = inner;
            let _ = category_dicts;
            false
        }
    }
}

/// Whether a shard's `CategoryBitset` and the dictionary agree about how many
/// categories the column has, i.e. whether they plausibly came from the same
/// index build.
///
/// `derive_shard_column_stats` sizes every bitset as `entries.len().div_ceil(8)`
/// from the same entry list `build_category_dicts` turns into `values`, so a
/// disagreement means the catalog's stats and the index section were produced
/// by different builds — and the bit positions then mean different things in
/// each. One comparison, and it turns a silently mis-resolved ordinal into a
/// skipped stat.
///
/// This is defence in depth, not proof: two builds whose vocabularies happen to
/// have the same length are indistinguishable here. What it does cover is the
/// case a length check *can* cover, cheaply.
///
/// `None` (no dictionary for this column) passes: the caller then resolves no
/// bit index and prunes nothing, so there is nothing to guard.
fn bitset_matches_dictionary(bitset: &[u8], dict: Option<&CategoryDictionary>) -> bool {
    match dict {
        Some(d) => bitset.len() == d.values.len().div_ceil(8),
        None => true,
    }
}

/// Integers with magnitude below 2^53 are exactly representable in f64, so
/// f64-based min/max pruning is exact for them. At or above 2^53, an i64/u64
/// value may have been rounded when the stored bound was derived from the
/// column (`value as f64`), or when the predicate value is widened here, so
/// pruning could drop a shard that actually matches. See SCX-003.
const F64_EXACT_INT_LIMIT: f64 = 9_007_199_254_740_992.0; // 2^53

/// Convert a ScalarValue to (f64, is_integer). The `is_integer` flag records
/// whether the value originated from an integer predicate, which determines
/// whether the `F64_EXACT_INT_LIMIT` pruning guard applies.
fn scalar_to_f64_typed(val: &ScalarValue) -> Option<(f64, bool)> {
    match val {
        ScalarValue::Int64(v) => Some((*v as f64, true)),
        ScalarValue::Float64(v) => Some((*v, false)),
        _ => None,
    }
}

/// Whether an f64 comparison between a predicate value `v` and a stored `bound`
/// is exact enough to prune a shard. Float predicates are always exact (both
/// sides are genuine f64); integer predicates are exact only when both operands
/// are below the exact-representable limit. See SCX-003.
fn prune_exact(v: f64, bound: f64, is_int: bool) -> bool {
    !is_int || (v.abs() < F64_EXACT_INT_LIMIT && bound.abs() < F64_EXACT_INT_LIMIT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;
    use scx_format_io::catalog::{FullCatalog, FullCatalogEntry, ShardStats};
    use scx_format_io::section::SectionType;
    use scx_format_io::DeletionVectors;

    /// Build a catalog with N CSR shards, each having the given column stats.
    fn catalog_with_shards(shard_specs: Vec<(u64, u64, Vec<ColumnStat>)>) -> FullCatalog {
        let mut entries = Vec::new();
        // Add obs/var (non-shard) entries
        entries.push(FullCatalogEntry {
            name: "obs".to_string(),
            offset: 4352,
            length: 1000,
            section_type: SectionType::ObsMetadata,
            checksum: [0; 32],
            modality_id: 0,
            stats: None,
        });
        entries.push(FullCatalogEntry {
            name: "var".to_string(),
            offset: 5352,
            length: 500,
            section_type: SectionType::VarMetadata,
            checksum: [0; 32],
            modality_id: 0,
            stats: None,
        });

        for (i, (row_start, row_end, column_stats)) in shard_specs.into_iter().enumerate() {
            entries.push(FullCatalogEntry {
                name: format!("X_shard_{i}"),
                offset: 10000 + (i as u64) * 10000,
                length: 5000,
                section_type: SectionType::CsrShard,
                checksum: [0; 32],
                modality_id: 0,
                stats: Some(ShardStats {
                    row_start,
                    row_end,
                    col_start: 0,
                    col_end: 0,
                    nnz: 1000,
                    value_min: 1,
                    value_max: 255,
                    value_sum: 10000,
                    n_indexed_columns: column_stats.len() as u8,
                    column_stats,
                }),
            });
        }

        FullCatalog {
            catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            n_obs: entries
                .iter()
                .filter_map(|e| e.stats.as_ref())
                .map(|s| s.row_end)
                .max()
                .unwrap_or(0),
            entries,
            data_generation: 0,
            csc_build_generation: 0,
        }
    }

    fn minmax_stat(col: &str, min: f64, max: f64) -> ColumnStat {
        ColumnStat::MinMax {
            column_name_hash: column_name_hash(col),
            min,
            max,
        }
    }

    fn category_bitset_stat(col: &str, bitset: Vec<u8>) -> ColumnStat {
        ColumnStat::CategoryBitset {
            column_name_hash: column_name_hash(col),
            bitset,
        }
    }

    // -----------------------------------------------------------------------
    // B1 Tests
    // -----------------------------------------------------------------------

    #[test]
    fn no_predicates_all_shards_pass() {
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![minmax_stat("n_genes", 100.0, 500.0)]),
            (100, 200, vec![minmax_stat("n_genes", 200.0, 800.0)]),
            (200, 300, vec![minmax_stat("n_genes", 50.0, 300.0)]),
        ]);
        let candidates = prune_shards_by_catalog(&catalog, &[], None);
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn numeric_minmax_excludes_shard() {
        // Predicate: n_genes > 600
        // Shard 0: max=500 → excluded (max <= 600)
        // Shard 1: max=800 → included
        // Shard 2: max=300 → excluded (max <= 600)
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![minmax_stat("n_genes", 100.0, 500.0)]),
            (100, 200, vec![minmax_stat("n_genes", 200.0, 800.0)]),
            (200, 300, vec![minmax_stat("n_genes", 50.0, 300.0)]),
        ]);
        let preds = vec![Predicate::Gt(
            "n_genes".to_string(),
            ScalarValue::Int64(600),
        )];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].shard_idx, 1);
    }

    #[test]
    fn int64_above_f64_exact_limit_does_not_false_prune() {
        // SCX-003 regression. A shard containing 2^53 + 1 records a max of
        // 2^53 (rounded to even) in its f64 stat. The predicate `col > 2^53`
        // must NOT prune this shard, because row 2^53 + 1 genuinely matches.
        let two_pow_53: i64 = 1 << 53;
        let rounded_max = two_pow_53 as f64; // (2^53 + 1) as f64 == 2^53
        let stats = vec![minmax_stat("big_id", 0.0, rounded_max)];

        // Gt just above the exact limit: pruning would be a false negative.
        assert!(
            !can_exclude_shard(
                &Predicate::Gt("big_id".to_string(), ScalarValue::Int64(two_pow_53)),
                &stats,
                None,
            ),
            "Gt(2^53) must not prune a shard whose true max is 2^53 + 1"
        );

        // In with a single value above the limit likewise must not prune.
        assert!(
            !can_exclude_shard(
                &Predicate::In(
                    "big_id".to_string(),
                    vec![ScalarValue::Int64(two_pow_53 + 10)],
                ),
                &stats,
                None,
            ),
            "In([> 2^53]) must not prune when the bound may be rounded"
        );

        // Eq above the limit must not prune either: a shard whose only value
        // is 2^53 + 1 stores max == 2^53, and Eq(2^53 + 1) rounds to 2^53, so
        // an unguarded `v > max` check could not exclude it here — but the
        // guard keeps Eq uniform with the other branches.
        assert!(
            !can_exclude_shard(
                &Predicate::Eq("big_id".to_string(), ScalarValue::Int64(two_pow_53 + 1)),
                &[minmax_stat("big_id", rounded_max, rounded_max)],
                None,
            ),
            "Eq(2^53 + 1) must not prune a shard whose true value is 2^53 + 1"
        );

        // Sanity: small integers below the limit still prune exactly.
        let small = vec![minmax_stat("n_genes", 100.0, 500.0)];
        assert!(can_exclude_shard(
            &Predicate::Gt("n_genes".to_string(), ScalarValue::Int64(600)),
            &small,
            None,
        ));
        // Eq below the limit still prunes when genuinely outside [min, max].
        assert!(can_exclude_shard(
            &Predicate::Eq("n_genes".to_string(), ScalarValue::Int64(600)),
            &small,
            None,
        ));
    }

    #[test]
    fn numeric_lt_excludes_shard() {
        // Predicate: n_genes < 100
        // Shard 0: min=100 → excluded (min >= 100)
        // Shard 1: min=200 → excluded (min >= 100)
        // Shard 2: min=50 → included
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![minmax_stat("n_genes", 100.0, 500.0)]),
            (100, 200, vec![minmax_stat("n_genes", 200.0, 800.0)]),
            (200, 300, vec![minmax_stat("n_genes", 50.0, 300.0)]),
        ]);
        let preds = vec![Predicate::Lt(
            "n_genes".to_string(),
            ScalarValue::Int64(100),
        )];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].shard_idx, 2);
    }

    /// The global category dictionary a real query carries: sorted
    /// values keyed by column-name hash, exactly the shape
    /// `CategoricalIndex::entries` (BTreeMap-sorted) produces.
    /// A dictionary the pruner may trust as the column's complete value set —
    /// what `build_category_dicts` produces when every shard carries the
    /// column's `CategoryBitset`.
    fn category_dicts(col: &str, values: &[&str]) -> CategoryDictionaries {
        category_dicts_with_completeness(col, values, true)
    }

    fn category_dicts_with_completeness(
        col: &str,
        values: &[&str],
        complete: bool,
    ) -> CategoryDictionaries {
        let mut m = CategoryDictionaries::new();
        m.insert(
            column_name_hash(col),
            values.iter().map(|s| s.to_string()).collect(),
            complete,
        );
        m
    }

    /// Bitset pruning as it is actually reached: a `Utf8` literal
    /// resolved to its ordinal through the category dictionary.
    ///
    /// This is the only way `can_exclude_shard` ever sees a
    /// `CategoryBitset` from `parse_predicate`, because a categorical
    /// column is `Utf8` / `Dictionary(_, Utf8)` and rejects a bare
    /// integer literal.
    #[test]
    fn category_bitset_excludes_shard_via_the_dictionary() {
        // Categories sorted: ["B cell", "NK cell", "T cell"] → bits 0,1,2.
        // Shard 0 holds {B cell, T cell} = bits 0,2 → 0b101
        // Shard 1 holds {NK cell}        = bit  1   → 0b010
        let catalog = catalog_with_shards(vec![
            (
                0,
                100,
                vec![category_bitset_stat("cell_type", vec![0b0000_0101])],
            ),
            (
                100,
                200,
                vec![category_bitset_stat("cell_type", vec![0b0000_0010])],
            ),
        ]);
        let dicts = category_dicts("cell_type", &["B cell", "NK cell", "T cell"]);
        let preds = vec![Predicate::Eq(
            "cell_type".to_string(),
            ScalarValue::Utf8("T cell".to_string()),
        )];
        let candidates = prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&dicts), 0);
        assert_eq!(candidates.len(), 1, "only shard 0 holds 'T cell'");
        assert_eq!(candidates[0].shard_idx, 0);

        // A value absent from the (complete) dictionary excludes every shard.
        let preds = vec![Predicate::Eq(
            "cell_type".to_string(),
            ScalarValue::Utf8("Nope".to_string()),
        )];
        assert!(
            prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&dicts), 0).is_empty()
        );
    }

    /// The dictionary miss is a claim about the whole file, so it may only be
    /// made when the vocabulary covers the whole file.
    ///
    /// `Err(_) => return true` reads "this value is in no shard" out of "this
    /// value is not in the dictionary I happen to hold". Sound while the
    /// vocabulary is the column's complete value set, and a silent zero-row
    /// answer the moment it is not.
    #[test]
    fn absent_value_prunes_only_when_complete() {
        let catalog = catalog_with_shards(vec![
            (
                0,
                100,
                vec![category_bitset_stat("cell_type", vec![0b0000_0101])],
            ),
            (
                100,
                200,
                vec![category_bitset_stat("cell_type", vec![0b0000_0010])],
            ),
        ]);
        let preds = vec![Predicate::Eq(
            "cell_type".to_string(),
            ScalarValue::Utf8("Nope".to_string()),
        )];
        let values = &["B cell", "NK cell", "T cell"];

        let complete = category_dicts_with_completeness("cell_type", values, true);
        assert!(
            prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&complete), 0)
                .is_empty(),
            "a complete vocabulary proves 'Nope' is in no shard"
        );

        let partial = category_dicts_with_completeness("cell_type", values, false);
        let candidates =
            prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&partial), 0);
        assert_eq!(
            candidates.len(),
            2,
            "an incomplete vocabulary proves nothing about a value it does not \
             list — every shard must stay a candidate"
        );
    }

    /// The same inference in the `In` arm, which used to refuse it outright:
    /// one predicate, two code paths, opposite trust assumptions. Both now key
    /// off the same bit.
    #[test]
    fn in_absent_member_prunes_only_when_complete() {
        // Shard 0 = {B cell, T cell} (bits 0,2); shard 1 = {NK cell} (bit 1).
        let catalog = catalog_with_shards(vec![
            (
                0,
                100,
                vec![category_bitset_stat("cell_type", vec![0b0000_0101])],
            ),
            (
                100,
                200,
                vec![category_bitset_stat("cell_type", vec![0b0000_0010])],
            ),
        ]);
        let values = &["B cell", "NK cell", "T cell"];
        // 'T cell' lives only in shard 0; 'Nope' lives nowhere.
        let preds = vec![Predicate::In(
            "cell_type".to_string(),
            vec![
                ScalarValue::Utf8("T cell".to_string()),
                ScalarValue::Utf8("Nope".to_string()),
            ],
        )];

        let complete = category_dicts_with_completeness("cell_type", values, true);
        let candidates =
            prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&complete), 0);
        assert_eq!(
            candidates.len(),
            1,
            "'Nope' is provably absent under a complete vocabulary, so shard 1 \
             is excluded on 'T cell' alone"
        );
        assert_eq!(candidates[0].shard_idx, 0);

        let partial = category_dicts_with_completeness("cell_type", values, false);
        assert_eq!(
            prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&partial), 0).len(),
            2,
            "'Nope' is merely unresolvable under a partial vocabulary — it could \
             be the value shard 1 holds"
        );
    }

    /// A bitset sized for a different vocabulary means the catalog's stats and
    /// the index section came from different builds, so bit `i` does not mean
    /// dictionary entry `i`. Resolving an ordinal against it prunes on a
    /// coincidence.
    #[test]
    fn bitset_length_mismatch_is_not_trusted() {
        // Nine categories need two bytes; this shard carries one, so it was
        // written against some other vocabulary.
        let catalog = catalog_with_shards(vec![(
            0,
            100,
            vec![category_bitset_stat("cell_type", vec![0b0000_0001])],
        )]);
        let dicts = category_dicts(
            "cell_type",
            &["a", "b", "c", "d", "e", "f", "g", "h", "i"], // 9 → 2 bytes
        );
        for pred in [
            Predicate::Eq(
                "cell_type".to_string(),
                ScalarValue::Utf8("i".to_string()), // bit 8: absent from a 1-byte bitset
            ),
            Predicate::In(
                "cell_type".to_string(),
                vec![ScalarValue::Utf8("i".to_string())],
            ),
        ] {
            assert_eq!(
                prune_shards_by_catalog_with_dict(
                    &catalog,
                    std::slice::from_ref(&pred),
                    None,
                    Some(&dicts),
                    0
                )
                .len(),
                1,
                "{pred} must not prune against a bitset from another build"
            );
        }
    }

    /// Why `append` without `--index-obs` still answers correctly, pinned.
    ///
    /// The appended CSR shard carries no `column_stats`, and a shard with no
    /// stats is never pruned — so a category value that exists only in the
    /// appended rows is found by the scan even though the file-scope index has
    /// never heard of it. That is structural, not luck, but nothing tested it,
    /// and it is the load-bearing reason the stale-index case was survivable.
    #[test]
    fn appended_shard_without_column_stats_is_never_pruned() {
        let catalog = catalog_with_shards(vec![
            // Pre-existing shard, indexed.
            (
                0,
                100,
                vec![category_bitset_stat("cell_type", vec![0b0000_0011])],
            ),
            // Appended shard: no column stats at all.
            (100, 200, vec![]),
        ]);
        // The index predates the append, so it lists neither the appended rows
        // nor their new category — and, because one shard has no bitset, it is
        // no longer a complete vocabulary.
        let dicts = category_dicts_with_completeness("cell_type", &["B cell", "T cell"], false);
        let preds = vec![Predicate::Eq(
            "cell_type".to_string(),
            ScalarValue::Utf8("Appended cell".to_string()),
        )];
        let candidates = prune_shards_by_catalog_with_dict(&catalog, &preds, None, Some(&dicts), 0);
        assert!(
            candidates.iter().any(|c| c.shard_idx == 1),
            "the appended shard must survive pruning or its rows are unreachable"
        );
    }

    /// The legacy-file regression again, through the door the integer-literal
    /// guard does not cover: a **string** literal.
    ///
    /// A file written before integer-valued categoricals were classified
    /// carries an entry-less `CategoricalIndex` for such a column, which
    /// `derive_shard_column_stats` turns into a zero-length `CategoryBitset` on
    /// every shard — so every shard *is* covered, and the vocabulary looks
    /// complete. `batch == '1'` then resolves through the dictionary rather
    /// than being rejected as an ordinal: `binary_search` on an empty vector
    /// misses, and the miss is read as proof of absence everywhere.
    ///
    /// The type mismatch that would reject `'1'` against an integer column
    /// lives in the residual evaluator, which never runs — Level-1 prunes
    /// first and the query short-circuits to zero rows with no error and no
    /// warning.
    ///
    /// An empty vocabulary cannot distinguish "this column genuinely has no
    /// values" from "this build recorded none of them", so it licenses
    /// nothing. The cost of refusing is a full scan on an all-null indexed
    /// column; the cost of allowing it is this.
    #[test]
    fn an_empty_vocabulary_never_proves_absence() {
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![category_bitset_stat("batch", vec![])]),
            (100, 200, vec![category_bitset_stat("batch", vec![])]),
        ]);
        // Every shard carries the column's bitset, so the catalog signal alone
        // would call this vocabulary complete.
        let dicts = category_dicts_with_completeness("batch", &[], true);
        for pred in [
            Predicate::Eq("batch".to_string(), ScalarValue::Utf8("1".to_string())),
            Predicate::In(
                "batch".to_string(),
                vec![ScalarValue::Utf8("1".to_string())],
            ),
        ] {
            assert_eq!(
                prune_shards_by_catalog_with_dict(
                    &catalog,
                    std::slice::from_ref(&pred),
                    None,
                    Some(&dicts),
                    0
                )
                .len(),
                2,
                "{pred} must scan both shards, not silently return nothing"
            );
        }
    }

    /// An integer literal is a **value**, never a bit index into the
    /// category bitset. Reading it as an ordinal prunes on a numeric
    /// coincidence: here `batch == 2` would resolve to bit 2, which
    /// shard 1 does not have set, and drop the shard that holds it.
    ///
    /// Unreachable before integer literals validated against
    /// integer-valued categoricals — and the reason that
    /// fix could not ship on its own.
    #[test]
    fn an_integer_literal_is_never_a_category_ordinal() {
        let catalog = catalog_with_shards(vec![
            (
                0,
                100,
                vec![category_bitset_stat("batch", vec![0b0000_0101])],
            ),
            (
                100,
                200,
                vec![category_bitset_stat("batch", vec![0b0000_0010])],
            ),
        ]);
        let preds = vec![Predicate::Eq("batch".to_string(), ScalarValue::Int64(2))];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        assert_eq!(
            candidates.len(),
            2,
            "an unresolvable literal must not exclude any shard"
        );
    }

    /// The legacy-file regression. Every SCX file written before the
    /// classification fix indexed an integer categorical as an entry-less
    /// `CategoricalIndex`, which `derive_shard_column_stats` turns into
    /// a **zero-length** `CategoryBitset` on every shard. Reading an
    /// integer literal as an ordinal then took the
    /// `byte_idx >= bitset.len()` → "absent" branch on every shard, so
    /// `batch == 3` came back empty with no error and no warning.
    #[test]
    fn an_empty_category_bitset_never_prunes_a_shard() {
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![category_bitset_stat("batch", vec![])]),
            (100, 200, vec![category_bitset_stat("batch", vec![])]),
        ]);
        for pred in [
            Predicate::Eq("batch".to_string(), ScalarValue::Int64(3)),
            Predicate::In(
                "batch".to_string(),
                vec![ScalarValue::Int64(1), ScalarValue::Int64(2)],
            ),
        ] {
            let candidates = prune_shards_by_catalog(&catalog, std::slice::from_ref(&pred), None);
            assert_eq!(
                candidates.len(),
                2,
                "{pred} must scan both shards, not silently return nothing"
            );
        }
    }

    #[test]
    fn all_pass_predicate_includes_all() {
        // Predicate on column that doesn't have stats → all shards pass
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![minmax_stat("n_genes", 100.0, 500.0)]),
            (100, 200, vec![minmax_stat("n_genes", 200.0, 800.0)]),
        ]);
        let preds = vec![Predicate::Eq(
            "other_column".to_string(),
            ScalarValue::Utf8("foo".to_string()),
        )];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        assert_eq!(candidates.len(), 2);
    }

    #[test]
    fn multiple_predicates_and_semantics() {
        // Predicate: n_genes > 100 AND n_genes < 400
        // Shard 0: min=100, max=500 → passes Gt check (max>100, min<400)
        // Shard 1: min=200, max=800 → excluded by Lt check (min >= 400? no, min=200)
        //          Actually: Shard 1 max=800 passes Gt(100) and min=200 passes Lt(400) → included
        // Shard 2: min=500, max=900 → excluded by Lt(400) check (min=500 >= 400)
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![minmax_stat("n_genes", 100.0, 500.0)]),
            (100, 200, vec![minmax_stat("n_genes", 200.0, 800.0)]),
            (200, 300, vec![minmax_stat("n_genes", 500.0, 900.0)]),
        ]);
        let preds = vec![
            Predicate::Gt("n_genes".to_string(), ScalarValue::Int64(100)),
            Predicate::Lt("n_genes".to_string(), ScalarValue::Int64(400)),
        ];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].shard_idx, 0);
        assert_eq!(candidates[1].shard_idx, 1);
    }

    #[test]
    fn phase1_no_column_stats_all_pass() {
        // Phase 1 files: n_indexed_columns == 0, empty column_stats
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![]),
            (100, 200, vec![]),
            (200, 300, vec![]),
        ]);
        let preds = vec![Predicate::Gt(
            "n_genes".to_string(),
            ScalarValue::Int64(600),
        )];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        // All shards should pass through (no stats to prune on)
        assert_eq!(candidates.len(), 3);
    }

    // -----------------------------------------------------------------------
    // B3 Tests: Deletion vector integration
    // -----------------------------------------------------------------------

    #[test]
    fn fully_deleted_shard_excluded() {
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![]),   // 100 rows
            (100, 200, vec![]), // 100 rows
        ]);
        // Mark all 100 rows in shard 0 as deleted
        let mut bm = RoaringBitmap::new();
        for i in 0..100u32 {
            bm.insert(i);
        }
        let mut dv = DeletionVectors::new();
        // Shard 0 has row_start 0, so local rows == global obs rows.
        dv.deletions.insert(0, bm);
        let candidates = prune_shards_by_catalog(&catalog, &[], Some(&dv));
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].shard_idx, 1);
    }

    #[test]
    fn partially_deleted_shard_included() {
        let catalog = catalog_with_shards(vec![
            (0, 100, vec![]),   // 100 rows
            (100, 200, vec![]), // 100 rows
        ]);
        // Mark 50 out of 100 rows in shard 0 as deleted
        let mut bm = RoaringBitmap::new();
        for i in 0..50u32 {
            bm.insert(i);
        }
        let mut dv = DeletionVectors::new();
        // Shard 0 has row_start 0, so local rows == global obs rows.
        dv.deletions.insert(0, bm);
        let candidates = prune_shards_by_catalog(&catalog, &[], Some(&dv));
        // Both shards included: shard 0 is only partially deleted
        assert_eq!(candidates.len(), 2);
    }
}
