//! Phase-4 gate tests (SCX-SORT-SPEC §10/§13) for the standalone `scx sort`
//! engine. Pure SCX; built on the Phase-0 fixtures in `crate::test_utils`.

use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::path::Path;

use arrow::array::{Array, Int64Array, StringArray};
use arrow::datatypes::DataType;
use scx_engine::index::PredicateIndex;
use scx_format_io::ScxReader;

use super::{sort, sort_with_strategy};
use crate::sort::{SortOptions, SortStrategy};
use crate::test_utils::{
    fixture_composite, fixture_deletion, fixture_numeric, fixture_plain, fixture_skewed,
};

// --- helpers ---------------------------------------------------------------

fn opts(by: &[&str]) -> SortOptions {
    SortOptions {
        by: by.iter().map(|s| s.to_string()).collect(),
        // Small shards so tiny fixtures still exercise multi-shard re-sharding.
        shard_target_rows: 2,
        ..Default::default()
    }
}

/// (cell_ids, per-row sparse `(col, value)`) in output row order.
fn content(path: &Path) -> (Vec<String>, Vec<Vec<(i32, f32)>>) {
    let r = ScxReader::open(path).unwrap();
    let ids = str_col(&r.read_obs().unwrap(), "cell_id");
    let csr = r.read_all_csr_shards().unwrap();
    let mut rows = Vec::new();
    for i in 0..csr.shape.0 {
        let s = csr.indptr[i] as usize;
        let e = csr.indptr[i + 1] as usize;
        rows.push((s..e).map(|j| (csr.indices[j], csr.data[j])).collect());
    }
    (ids, rows)
}

fn str_col(batch: &arrow::array::RecordBatch, name: &str) -> Vec<String> {
    let col = batch.column_by_name(name).unwrap();
    let utf8 = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let arr = utf8.as_any().downcast_ref::<StringArray>().unwrap();
    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
}

fn col_of(path: &Path, name: &str) -> Vec<String> {
    str_col(&ScxReader::open(path).unwrap().read_obs().unwrap(), name)
}

fn is_sorted_asc(v: &[String]) -> bool {
    v.windows(2).all(|w| w[0] <= w[1])
}

// --- T4.1/round-trip: in-memory default ------------------------------------

#[test]
fn round_trip_equivalence_categorical() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    // No budget set → in-memory fast path.
    assert_eq!(summary.strategy, SortStrategy::InMemory);
    assert!(summary.indexed_columns.iter().any(|c| c == "cell_type"));

    let (in_ids, in_rows) = content(&inp);
    let (out_ids, out_rows) = content(&out);
    assert_eq!(in_ids.len(), out_ids.len());

    // Same set of cells, X content preserved per cell.
    let in_map: HashMap<&String, &Vec<(i32, f32)>> = in_ids.iter().zip(&in_rows).collect();
    for (id, row) in out_ids.iter().zip(&out_rows) {
        assert_eq!(in_map[id], row, "X row for {id} must survive the reorder");
    }
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}

// --- Strategy differential: a/b/c byte-identical (content) -----------------

#[test]
fn strategy_differential_identical() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64); // forces the external path to actually partition

    let mut results = Vec::new();
    for strat in [
        SortStrategy::InMemory,
        SortStrategy::KPassByCategory,
        SortStrategy::ExternalPartition,
    ] {
        let out = dir.path().join(format!("out_{strat:?}.scx"));
        let summary = sort_with_strategy(&inp, &out, &o, Some(strat)).unwrap();
        assert_eq!(summary.strategy, strat);
        results.push(content(&out));
    }
    assert_eq!(results[0], results[1], "in-memory vs K-pass must match");
    assert_eq!(results[1], results[2], "K-pass vs external must match");
}

// --- Determinism (modulo provenance timestamp) -----------------------------

#[test]
fn deterministic_output() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let a = dir.path().join("a.scx");
    let b = dir.path().join("b.scx");
    sort(&inp, &a, &opts(&["cell_type"])).unwrap();
    sort(&inp, &b, &opts(&["cell_type"])).unwrap();
    assert_eq!(content(&a), content(&b));
    assert_eq!(col_of(&a, "cell_type"), col_of(&b, "cell_type"));
}

// --- Stability under the external scatter path -----------------------------

#[test]
fn stable_tie_order_external() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition)).unwrap();

    // Input position of each cell.
    let (in_ids, _) = content(&inp);
    let pos: HashMap<&String, usize> = in_ids.iter().enumerate().map(|(i, s)| (s, i)).collect();

    let out_ids = col_of(&out, "cell_id");
    let out_types = col_of(&out, "cell_type");
    // Within each equal-key run, input positions must be strictly increasing
    // (stability: equal keys keep their original global order).
    for w in 0..out_ids.len().saturating_sub(1) {
        if out_types[w] == out_types[w + 1] {
            assert!(
                pos[&out_ids[w]] < pos[&out_ids[w + 1]],
                "tie order broke at output rows {w}/{}",
                w + 1
            );
        }
    }
}

// --- Skew: bounded partitions, no sub-split needed (new_pos design) --------

#[test]
fn skew_partitions_bounded() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_skewed(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64); // P == 2 rows → 10 partitions over 20 rows

    let summary =
        sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition)).unwrap();
    assert!(summary.partitions > 1, "skew must span multiple partitions");
    assert!(summary.spill_bytes > 0, "external path must spill");
    // new_pos-range partitions are inherently balanced regardless of the
    // dominant category: 20 rows / 2-per-partition.
    assert_eq!(summary.partitions, 10);
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}

// --- Spill-size telemetry + refuse-on-overflow (T4.7) ----------------------

#[test]
fn spill_size_matches_formula() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.memory_budget = Some(64);
    let summary =
        sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition)).unwrap();
    // 12 rows × (8 new_pos + 4 nnz + 2 nnz × 8 bytes) = 12 × 28.
    assert_eq!(summary.spill_bytes, 12 * 28);
}

#[test]
fn external_refuses_when_budget_below_one_shard() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    // per-shard bytes = 2 rows × 8 vars × 0.25 density × 16 = 64; budget below it.
    o.memory_budget = Some(32);
    let err = sort_with_strategy(&inp, &out, &o, Some(SortStrategy::ExternalPartition));
    assert!(err.is_err(), "must refuse a budget too small for one shard");
}

// --- Index correctness: contiguous shard ranges ----------------------------

#[test]
fn index_ranges_contiguous() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    let r = ScxReader::open(&out).unwrap();
    let bytes = r
        .read_obs_predicate_index_bytes()
        .unwrap()
        .expect("sort must (re)build the obs predicate index for the sort key");
    let index = PredicateIndex::read_from(&mut Cursor::new(bytes)).unwrap();

    let ranges = index
        .categorical_eq("cell_type", "B cell")
        .expect("cell_type must be an indexed categorical");
    assert!(!ranges.is_empty(), "B cell must occupy some shard range");
    // After sort, the category is a contiguous block → its shard ids form a
    // consecutive run.
    let mut shard_ids: Vec<u32> = ranges.iter().map(|r| r.shard_id).collect();
    shard_ids.sort_unstable();
    shard_ids.dedup();
    for w in shard_ids.windows(2) {
        assert_eq!(w[1], w[0] + 1, "B cell shard ids must be contiguous");
    }
}

// --- Composite + reverse + numeric keys ------------------------------------

#[test]
fn composite_key_lexicographic() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_composite(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["cell_type", "donor"])).unwrap();

    let types = col_of(&out, "cell_type");
    let donors = col_of(&out, "donor");
    assert!(is_sorted_asc(&types));
    // Within each cell_type block, donor is non-decreasing.
    for w in 0..types.len().saturating_sub(1) {
        if types[w] == types[w + 1] {
            assert!(donors[w] <= donors[w + 1], "donor order within a cell_type");
        }
    }
}

#[test]
fn reverse_key_descending() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_plain(&dir);
    let out = dir.path().join("out.scx");
    let mut o = opts(&["cell_type"]);
    o.reverse = true;
    sort(&inp, &out, &o).unwrap();
    let types = col_of(&out, "cell_type");
    assert!(
        types.windows(2).all(|w| w[0] >= w[1]),
        "reverse → descending"
    );
}

#[test]
fn numeric_key_ascending() {
    let dir = tempfile::tempdir().unwrap();
    let inp = fixture_numeric(&dir);
    let out = dir.path().join("out.scx");
    sort(&inp, &out, &opts(&["n_genes"])).unwrap();

    let r = ScxReader::open(&out).unwrap();
    let obs = r.read_obs().unwrap();
    let col = obs.column_by_name("n_genes").unwrap();
    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
    let vals: Vec<i64> = (0..arr.len()).map(|i| arr.value(i)).collect();
    assert!(vals.windows(2).all(|w| w[0] <= w[1]), "numeric ascending");
}

// --- Deletion vector input → dense, deletion-free --------------------------

#[test]
fn deletion_input_materialized_away() {
    let dir = tempfile::tempdir().unwrap();
    let (inp, n_deleted) = fixture_deletion(&dir);
    let out = dir.path().join("out.scx");
    let summary = sort(&inp, &out, &opts(&["cell_type"])).unwrap();

    assert_eq!(summary.n_obs, (12 - n_deleted) as u64);

    let r = ScxReader::open(&out).unwrap();
    // Output carries no live deletions.
    let clean = r
        .deletion_keep_mask()
        .unwrap()
        .map(|m| m.iter().all(|&k| k))
        .unwrap_or(true);
    assert!(clean, "sorted output must be deletion-free");

    // Deleted cells (rows 1,3,5) are gone; everyone else survives.
    let out_ids: HashSet<String> = col_of(&out, "cell_id").into_iter().collect();
    let expected: HashSet<String> = (0..12)
        .filter(|i| ![1usize, 3, 5].contains(i))
        .map(|i| format!("cell_{i}"))
        .collect();
    assert_eq!(out_ids, expected);
    assert!(is_sorted_asc(&col_of(&out, "cell_type")));
}
