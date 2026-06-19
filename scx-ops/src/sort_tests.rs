//! Shared sort-core gate tests: key ordering, reverse, composite
//! leading-key precedence, stability, partition sizing, categorical
//! grouping, and numeric quantile boundaries.

use super::*;
use arrow::array::{ArrayRef, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

fn utf8_batch(name: &str, vals: Vec<&str>) -> RecordBatch {
    let schema = Schema::new(vec![Field::new(name, DataType::Utf8, true)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(StringArray::from(vals))]).unwrap()
}

fn i64_batch(name: &str, vals: Vec<i64>) -> RecordBatch {
    let schema = Schema::new(vec![Field::new(name, DataType::Int64, true)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(Int64Array::from(vals))]).unwrap()
}

fn argsort(batch: &RecordBatch, by: &[&str], reverse: bool) -> Vec<u64> {
    let by: Vec<String> = by.iter().map(|s| s.to_string()).collect();
    let ex = SortKeyExtractor::new(&batch.schema(), &by, reverse).unwrap();
    let rows = ex.rows(batch).unwrap();
    stable_argsort(&rows, 0)
}

// --- T1.2 / T1.3: key ordering, reverse, composite, stability ---

#[test]
fn categorical_key_ordering() {
    let batch = utf8_batch("cell_type", vec!["B", "A", "B", "A"]);
    // ascending A < B; ties keep source order
    assert_eq!(argsort(&batch, &["cell_type"], false), vec![1, 3, 0, 2]);
}

#[test]
fn numeric_key_ordering_with_ties() {
    let batch = i64_batch("n_genes", vec![5, 1, 3, 1]);
    // ascending: 1@1, 1@3, 3@2, 5@0 — equal 1s keep ascending source id
    assert_eq!(argsort(&batch, &["n_genes"], false), vec![1, 3, 2, 0]);
}

#[test]
fn reverse_flips_order_but_keeps_stability() {
    let batch = i64_batch("n_genes", vec![5, 1, 3, 1]);
    // descending: 5@0, 3@2, then equal 1s still in ascending source order
    assert_eq!(argsort(&batch, &["n_genes"], true), vec![0, 2, 1, 3]);
}

#[test]
fn composite_leading_key_precedence() {
    let schema = Schema::new(vec![
        Field::new("cell_type", DataType::Utf8, true),
        Field::new("donor", DataType::Utf8, true),
    ]);
    let ct = StringArray::from(vec!["A", "A", "B", "B"]);
    let donor = StringArray::from(vec!["d2", "d1", "d1", "d2"]);
    let batch =
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(ct), Arc::new(donor)]).unwrap();
    // leading key A before B; within each, donor ascending
    assert_eq!(
        argsort(&batch, &["cell_type", "donor"], false),
        vec![1, 0, 2, 3]
    );
}

#[test]
fn stability_all_equal_keys_is_identity() {
    let batch = utf8_batch("cell_type", vec!["X", "X", "X", "X"]);
    assert_eq!(argsort(&batch, &["cell_type"], false), vec![0, 1, 2, 3]);
}

#[test]
fn cmp_keyed_breaks_ties_by_source_id() {
    let batch = utf8_batch("cell_type", vec!["X", "X"]);
    let ex = SortKeyExtractor::new(&batch.schema(), &["cell_type".to_string()], false).unwrap();
    let rows = ex.rows(&batch).unwrap();
    let r0 = rows.row(0);
    let r1 = rows.row(1);
    // equal keys → tie broken by id
    assert_eq!(cmp_keyed((&r0, 5), (&r1, 2)), Ordering::Greater);
    assert_eq!(cmp_keyed((&r0, 1), (&r1, 2)), Ordering::Less);
    assert_eq!(cmp_keyed((&r0, 7), (&r1, 7)), Ordering::Equal);
}

#[test]
fn missing_key_column_errors() {
    let batch = utf8_batch("cell_type", vec!["A"]);
    let err = SortKeyExtractor::new(&batch.schema(), &["nope".to_string()], false);
    assert!(matches!(err, Err(OpsError::InvalidInput(_))));
    let empty = SortKeyExtractor::new(&batch.schema(), &[], false);
    assert!(matches!(empty, Err(OpsError::InvalidInput(_))));
}

// --- T1.4: partition sizing, categorical grouping, numeric quantiles ---

#[test]
fn partition_target_rows_formula() {
    // No budget → one shard's worth.
    assert_eq!(partition_target_rows(None, 1000, 0.1, 16384), 16384);
    // Budget large enough to exceed the floor: per_row = 1000*0.1*16 = 1600.
    assert_eq!(
        partition_target_rows(Some(100_000_000), 1000, 0.1, 16384),
        62_500
    );
    // Budget below the floor → floored at shard_target_rows.
    assert_eq!(
        partition_target_rows(Some(1_000_000), 1000, 0.1, 16384),
        16384
    );
}

#[test]
fn categorical_partitions_group_and_isolate_dominant() {
    let dir = tempfile::tempdir().unwrap();
    let path = crate::test_utils::fixture_skewed(&dir);
    let reader = scx_format_io::ScxReader::open(&path).unwrap();
    let obs = reader.read_obs().unwrap();
    let col: ArrayRef = obs.column_by_name("cell_type").unwrap().clone();

    // target 5: B(1)+NK(1) group; dominant T cell(18) alone (> target).
    let plan = leading_key_partitions(&col, 5, false).unwrap();
    match plan {
        PartitionPlan::Categorical { groups } => {
            assert_eq!(groups.len(), 2);
            assert_eq!(groups[0], vec!["B cell".to_string(), "NK cell".to_string()]);
            assert_eq!(groups[1], vec!["T cell".to_string()]);
        }
        other => panic!("expected categorical plan, got {other:?}"),
    }
}

#[test]
fn categorical_partitions_enumerate_when_under_target() {
    let col: ArrayRef = Arc::new(StringArray::from(vec!["B", "A", "C", "A", "B"]));
    // target 1: each distinct value its own group, ascending.
    let plan = leading_key_partitions(&col, 1, false).unwrap();
    assert_eq!(
        plan,
        PartitionPlan::Categorical {
            groups: vec![
                vec!["A".to_string()],
                vec!["B".to_string()],
                vec!["C".to_string()],
            ]
        }
    );
    assert_eq!(plan.n_partitions(), 3);
}

#[test]
fn numeric_histogram_bins_values() {
    let hist = numeric_histogram([0.5f64, 1.5, 2.5].into_iter(), 0.0, 3.0, 3);
    assert_eq!(hist, vec![(1.0, 1), (2.0, 1), (3.0, 1)]);
    // value at max lands in the last bin
    let hist2 = numeric_histogram([3.0f64].into_iter(), 0.0, 3.0, 3);
    assert_eq!(hist2[2].1, 1);
}

#[test]
fn numeric_cut_points_split_by_target() {
    let hist = vec![(1.0, 10), (2.0, 10), (3.0, 10)];
    // target 10 → cut after bin0 and bin1, none after the final bin
    assert_eq!(numeric_cut_points(&hist, 10), vec![1.0, 2.0]);
    // target larger than total → single partition, no cuts
    assert_eq!(numeric_cut_points(&hist, 1000), Vec::<f64>::new());
}

#[test]
fn numeric_leading_key_partitions_from_column() {
    let col: ArrayRef = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8]));
    let plan = leading_key_partitions(&col, 2, false).unwrap();
    match plan {
        PartitionPlan::Numeric { cut_points } => {
            // 8 values, target 2 → ~4 partitions → ~3 cut points
            assert!(cut_points.len() >= 2, "got {cut_points:?}");
            assert!(cut_points.windows(2).all(|w| w[0] < w[1]), "ascending");
        }
        other => panic!("expected numeric plan, got {other:?}"),
    }
}

// --- T1.6: provenance ---

#[test]
fn provenance_entry_shape() {
    let entry = sort_provenance_entry(
        &["cell_type".to_string()],
        true,
        16384,
        &["cell_type".to_string()],
        1_700_000_000,
    );
    assert_eq!(entry.action, "sort");
    assert_eq!(entry.timestamp, 1_700_000_000);
    let v: serde_json::Value = serde_json::from_str(&entry.params_json).unwrap();
    assert_eq!(v["reverse"], serde_json::json!(true));
    assert_eq!(v["shard_size"], serde_json::json!(16384));
    assert_eq!(v["by"], serde_json::json!(["cell_type"]));
    assert_eq!(
        v["predicate_index"]["obs_columns"],
        serde_json::json!(["cell_type"])
    );
}
