//! Property-based test for predicate-index query equivalence
//!
//! Strategy: build a categorical (Utf8) and a numeric (Float32) column
//! from random data, build a `PredicateIndex`, then check that
//! `prune_rows_by_index` agrees with a brute-force linear scan.
//!
//! Invariants:
//! - **Categorical equality**: precise. The index's row set must equal
//!   the brute-force set exactly. The index stores per-value shard
//!   ranges and never returns false positives for `Eq`/`In`.
//! - **Numeric range**: conservative superset. Leaf pages store
//!   `(min_value, max_value, row_range)` over contiguous sorted entries
//!   within one shard, so the index may return rows whose value is
//!   *outside* the predicate range — the engine post-filters. The
//!   invariant is therefore "index hits ⊇ brute-force hits".
//! - Across both, the index must never miss a row the brute-force scan
//!   accepts. That's the "no false negatives" half — the load-bearing
//!   correctness condition the engine relies on.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use proptest::prelude::*;
use scx_engine::{
    build_indexes, prune_rows_by_index, Predicate, PredicateIndex, ScalarValue,
};

const CATEGORIES: &[&str] = &["T cell", "B cell", "NK cell", "monocyte"];

// =========================================================================
// Strategies
// =========================================================================

fn arb_records(min_rows: usize, max_rows: usize) -> impl Strategy<Value = RecordBatch> {
    (min_rows..=max_rows).prop_flat_map(|n_rows| {
        let cat_strat =
            prop::collection::vec(0usize..CATEGORIES.len(), n_rows).prop_map(|categories| {
                let strs: Vec<&str> = categories.iter().map(|&i| CATEGORIES[i]).collect();
                StringArray::from(strs)
            });
        let num_strat = prop::collection::vec(-100.0f32..100.0, n_rows).prop_map(Float32Array::from);
        (Just(n_rows), cat_strat, num_strat).prop_map(|(n_rows, cat, num)| {
            let schema = Schema::new(vec![
                Field::new("cell_type", DataType::Utf8, false),
                Field::new("score", DataType::Float32, false),
            ]);
            let _ = n_rows;
            RecordBatch::try_new(
                Arc::new(schema),
                vec![Arc::new(cat) as ArrayRef, Arc::new(num) as ArrayRef],
            )
            .unwrap()
        })
    })
}

/// Partition n_rows into between 1 and 4 contiguous shards.
fn shard_row_ranges(n_rows: u64, n_shards: usize) -> Vec<(u64, u64)> {
    if n_shards == 0 || n_rows == 0 {
        return vec![(0, n_rows)];
    }
    let n_shards = n_shards.min(n_rows as usize).max(1);
    let per = n_rows.div_ceil(n_shards as u64);
    (0..n_shards as u64)
        .map(|i| {
            let start = i * per;
            let end = ((i + 1) * per).min(n_rows);
            (start, end)
        })
        .filter(|(s, e)| s < e)
        .collect()
}

// =========================================================================
// Brute-force scans
// =========================================================================

fn brute_force_eq_cat(batch: &RecordBatch, col_name: &str, target: &str) -> BTreeSet<u64> {
    let col = batch
        .schema()
        .index_of(col_name)
        .ok()
        .map(|i| batch.column(i))
        .unwrap();
    let arr = col
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column must be StringArray");
    let mut hits = BTreeSet::new();
    for row in 0..arr.len() {
        if !arr.is_null(row) && arr.value(row) == target {
            hits.insert(row as u64);
        }
    }
    hits
}

fn brute_force_numeric_range(
    batch: &RecordBatch,
    col_name: &str,
    lo: Option<(f32, bool)>, // (val, inclusive)
    hi: Option<(f32, bool)>,
) -> BTreeSet<u64> {
    let col = batch
        .schema()
        .index_of(col_name)
        .ok()
        .map(|i| batch.column(i))
        .unwrap();
    let arr = col
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("column must be Float32Array");
    let mut hits = BTreeSet::new();
    for row in 0..arr.len() {
        if arr.is_null(row) {
            continue;
        }
        let v = arr.value(row);
        let lo_ok = match lo {
            Some((bound, true)) => v >= bound,
            Some((bound, false)) => v > bound,
            None => true,
        };
        let hi_ok = match hi {
            Some((bound, true)) => v <= bound,
            Some((bound, false)) => v < bound,
            None => true,
        };
        if lo_ok && hi_ok {
            hits.insert(row as u64);
        }
    }
    hits
}

/// Translate per-shard `Range<u32>` results from `prune_rows_by_index`
/// back into a global-row set via the shard row-range map.
fn collect_index_hits(
    index: &PredicateIndex,
    predicate: &Predicate,
    shard_ranges: &[(u64, u64)],
) -> Option<BTreeSet<u64>> {
    let mut any_indexed = false;
    let mut hits: BTreeSet<u64> = BTreeSet::new();
    for (shard_idx, &(row_start, _row_end)) in shard_ranges.iter().enumerate() {
        match prune_rows_by_index(index, std::slice::from_ref(predicate), shard_idx) {
            Some(ranges) => {
                any_indexed = true;
                for r in ranges {
                    for local in r.start..r.end {
                        hits.insert(row_start + local as u64);
                    }
                }
            }
            None => {
                // Column not indexed at all — bail out; the property
                // only applies when the index can narrow.
                return None;
            }
        }
    }
    if !any_indexed {
        // All shards returned None (treated above) → not indexed.
        return None;
    }
    Some(hits)
}

// =========================================================================
// Property tests
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Categorical equality lookup agrees with brute-force scan for
    /// every value present in the column.
    #[test]
    fn categorical_eq_agrees_with_brute_force(
        batch in arb_records(20, 200),
        n_shards in 1usize..=4,
    ) {
        let shards = shard_row_ranges(batch.num_rows() as u64, n_shards);
        let index = build_indexes(
            &batch,
            &shards,
            &["cell_type".to_string()],
        ).expect("build_indexes");

        for target in CATEGORIES {
            let pred = Predicate::Eq(
                "cell_type".to_string(),
                ScalarValue::Utf8((*target).to_string()),
            );
            let bf = brute_force_eq_cat(&batch, "cell_type", target);
            if let Some(idx_hits) = collect_index_hits(&index, &pred, &shards) {
                prop_assert_eq!(idx_hits, bf,
                    "categorical Eq mismatch for value {:?}", target);
            }
        }
    }

    /// Numeric range lookup agrees with brute-force scan for a sample
    /// of random bounds.
    #[test]
    fn numeric_range_agrees_with_brute_force(
        batch in arb_records(20, 200),
        n_shards in 1usize..=4,
        lo_val in -120.0f32..120.0,
        hi_val in -120.0f32..120.0,
        lo_inclusive in any::<bool>(),
        hi_inclusive in any::<bool>(),
        use_lo in any::<bool>(),
        use_hi in any::<bool>(),
    ) {
        let shards = shard_row_ranges(batch.num_rows() as u64, n_shards);
        let index = build_indexes(
            &batch,
            &shards,
            &["score".to_string()],
        ).expect("build_indexes");

        // One-sided range: use Gt/Ge or Lt/Le. Two-sided ranges
        // require predicate composition the index can already handle
        // via Predicate::And; tested separately below.
        if use_lo && !use_hi {
            let pred = if lo_inclusive {
                Predicate::Ge("score".to_string(), ScalarValue::Float64(lo_val as f64))
            } else {
                Predicate::Gt("score".to_string(), ScalarValue::Float64(lo_val as f64))
            };
            let bf = brute_force_numeric_range(
                &batch, "score", Some((lo_val, lo_inclusive)), None);
            if let Some(idx_hits) = collect_index_hits(&index, &pred, &shards) {
                prop_assert!(bf.is_subset(&idx_hits),
                    "numeric one-sided lo: brute-force hits not subset of index hits \
                     (lo={} inclusive={}): missing rows = {:?}",
                    lo_val, lo_inclusive,
                    bf.difference(&idx_hits).copied().collect::<Vec<_>>());
            }
        } else if use_hi && !use_lo {
            let pred = if hi_inclusive {
                Predicate::Le("score".to_string(), ScalarValue::Float64(hi_val as f64))
            } else {
                Predicate::Lt("score".to_string(), ScalarValue::Float64(hi_val as f64))
            };
            let bf = brute_force_numeric_range(
                &batch, "score", None, Some((hi_val, hi_inclusive)));
            if let Some(idx_hits) = collect_index_hits(&index, &pred, &shards) {
                prop_assert!(bf.is_subset(&idx_hits),
                    "numeric one-sided hi: brute-force hits not subset of index hits \
                     (hi={} inclusive={}): missing rows = {:?}",
                    hi_val, hi_inclusive,
                    bf.difference(&idx_hits).copied().collect::<Vec<_>>());
            }
        }
    }

    /// Categorical AND numeric: intersection of the two single-column
    /// queries must equal the brute-force AND.
    #[test]
    fn categorical_and_numeric_agree_with_brute_force(
        batch in arb_records(40, 200),
        n_shards in 1usize..=3,
        threshold in -100.0f32..100.0,
        cat_idx in 0..CATEGORIES.len(),
    ) {
        let shards = shard_row_ranges(batch.num_rows() as u64, n_shards);
        let index = build_indexes(
            &batch,
            &shards,
            &["cell_type".to_string(), "score".to_string()],
        ).expect("build_indexes");

        let cat = CATEGORIES[cat_idx];
        let pred = Predicate::And(
            Box::new(Predicate::Eq(
                "cell_type".to_string(),
                ScalarValue::Utf8(cat.to_string()),
            )),
            Box::new(Predicate::Ge(
                "score".to_string(),
                ScalarValue::Float64(threshold as f64),
            )),
        );

        let bf_cat = brute_force_eq_cat(&batch, "cell_type", cat);
        let bf_num = brute_force_numeric_range(
            &batch, "score", Some((threshold, true)), None);
        let bf_intersect: BTreeSet<u64> = bf_cat.intersection(&bf_num).copied().collect();

        if let Some(idx_hits) = collect_index_hits(&index, &pred, &shards) {
            // Conservative superset: the numeric half over-approximates.
            prop_assert!(bf_intersect.is_subset(&idx_hits),
                "AND: brute-force intersection not subset of index hits");
            // Upper bound: cannot exceed the categorical (exact) hits.
            prop_assert!(idx_hits.is_subset(&bf_cat),
                "AND: index hits exceed the precise categorical hits");
        }
    }
}
