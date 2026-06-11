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

/// Dictionary mapping column_name_hash → sorted list of category values.
/// Used to resolve Utf8 predicate values to CategoryBitset bit positions.
pub type CategoryDictionaries = HashMap<u64, Vec<String>>;

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
    prune_shards_by_catalog_with_dict(catalog, predicates, deletion_vectors, None)
}

/// Like `prune_shards_by_catalog`, but accepts an optional category dictionary
/// for resolving `Utf8` predicate values against `CategoryBitset` column stats.
pub fn prune_shards_by_catalog_with_dict(
    catalog: &FullCatalog,
    predicates: &[Predicate],
    deletion_vectors: Option<&DeletionVectors>,
    category_dicts: Option<&CategoryDictionaries>,
) -> Vec<ShardCandidate> {
    let sorted_shards = catalog.shards_sorted();

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

        // B3: Check deletion vectors — if entire shard is deleted, skip it
        if let Some(dv) = deletion_vectors {
            let shard_n_rows = stats.row_end - stats.row_start;
            if let Some(bitmap) = dv.shards.get(&(shard_idx as u32)) {
                if bitmap.len() >= shard_n_rows {
                    // All rows deleted → skip this shard entirely
                    continue;
                }
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
                        // For numeric equality: skip if value outside [min, max]
                        if let Some(v) = scalar_to_f64(val) {
                            if v < *min || v > *max {
                                return true;
                            }
                        }
                    }
                    ColumnStat::CategoryBitset {
                        column_name_hash: cnh,
                        bitset,
                    } => {
                        // Resolve the predicate value to a bit index.
                        let bit_index = match val {
                            ScalarValue::Int64(idx) => Some(*idx as usize),
                            ScalarValue::Utf8(s) => {
                                // Look up string value in category dictionary.
                                match category_dicts.and_then(|dicts| dicts.get(cnh)) {
                                    Some(values) => match values.binary_search(s) {
                                        Ok(idx) => Some(idx),
                                        // The dictionary is present (this column
                                        // is indexed), so it is the complete
                                        // global value set. A value absent from
                                        // it exists in NO shard → exclude every
                                        // shard. This is what lets a no-match
                                        // equality short-circuit instead of
                                        // scanning all obs metadata.
                                        Err(_) => return true,
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
            if let Some(v) = scalar_to_f64(val) {
                for cs in column_stats {
                    if cs.column_name_hash() != hash {
                        continue;
                    }
                    if let ColumnStat::MinMax { max, .. } = cs {
                        // Gt: skip if col_max <= v; Ge: skip if col_max < v
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
            if let Some(v) = scalar_to_f64(val) {
                for cs in column_stats {
                    if cs.column_name_hash() != hash {
                        continue;
                    }
                    if let ColumnStat::MinMax { min, .. } = cs {
                        // Lt: skip if col_min >= v; Le: skip if col_min > v
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
                        let all_outside = values
                            .iter()
                            .filter_map(scalar_to_f64_ref)
                            .all(|v| v < *min || v > *max);
                        if all_outside {
                            return true;
                        }
                    }
                    ColumnStat::CategoryBitset {
                        column_name_hash: cnh,
                        bitset,
                    } => {
                        // For category In: exclude if none of the values have their bit set
                        let all_absent = values.iter().all(|v| {
                            let bit_index = match v {
                                ScalarValue::Int64(idx) => Some(*idx as usize),
                                ScalarValue::Utf8(s) => category_dicts
                                    .and_then(|dicts| dicts.get(cnh))
                                    .and_then(|vals| vals.binary_search(s).ok()),
                                _ => None,
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

/// Convert a ScalarValue to f64 for numeric comparisons.
fn scalar_to_f64(val: &ScalarValue) -> Option<f64> {
    match val {
        ScalarValue::Int64(v) => Some(*v as f64),
        ScalarValue::Float64(v) => Some(*v),
        _ => None,
    }
}

/// Same as scalar_to_f64 but takes a reference (for use in iterator chains).
fn scalar_to_f64_ref(val: &ScalarValue) -> Option<f64> {
    scalar_to_f64(val)
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

    #[test]
    fn category_bitset_excludes_shard() {
        // Category bitset: Shard 0 has categories {0, 2} (bits 0,2 set)
        //                   Shard 1 has categories {1, 3} (bits 1,3 set)
        // Predicate: category index == 2 → Shard 1 excluded
        let catalog = catalog_with_shards(vec![
            (
                0,
                100,
                vec![category_bitset_stat("cell_type", vec![0b0000_0101])],
            ),
            (
                100,
                200,
                vec![category_bitset_stat("cell_type", vec![0b0000_1010])],
            ),
        ]);
        let preds = vec![Predicate::Eq(
            "cell_type".to_string(),
            ScalarValue::Int64(2),
        )];
        let candidates = prune_shards_by_catalog(&catalog, &preds, None);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].shard_idx, 0);
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
        dv.shards.insert(0, bm);
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
        dv.shards.insert(0, bm);
        let candidates = prune_shards_by_catalog(&catalog, &[], Some(&dv));
        // Both shards included: shard 0 is only partially deleted
        assert_eq!(candidates.len(), 2);
    }
}
