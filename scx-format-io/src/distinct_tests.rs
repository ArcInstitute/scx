use super::*;
use arrow::array::{DictionaryArray, Int32Array, LargeStringArray, StringArray};
use arrow::datatypes::Int32Type;
use std::sync::Arc;

fn utf8(vals: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(StringArray::from(vals))
}

fn large_utf8(vals: Vec<Option<&str>>) -> ArrayRef {
    Arc::new(LargeStringArray::from(vals))
}

fn dict(vals: Vec<Option<&str>>) -> ArrayRef {
    let arr: DictionaryArray<Int32Type> = vals.into_iter().collect();
    Arc::new(arr)
}

#[test]
fn utf8_single_shard() {
    let mut acc = DistinctAccumulator::new("c", None, false);
    acc.push(&utf8(vec![Some("a"), Some("b"), Some("a"), Some("c")]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["a", "b", "c"]);
    assert!(!more);
}

#[test]
fn large_utf8_single_shard() {
    let mut acc = DistinctAccumulator::new("c", None, false);
    acc.push(&large_utf8(vec![Some("x"), Some("x"), Some("y")]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["x", "y"]);
    assert!(!more);
}

#[test]
fn dict_fast_path_single_shard() {
    let mut acc = DistinctAccumulator::new("c", None, false);
    acc.push(&dict(vec![Some("t"), Some("b"), Some("t"), Some("nk")]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["t", "b", "nk"]);
    assert!(!more);
}

#[test]
fn nulls_excluded() {
    let mut acc = DistinctAccumulator::new("c", None, false);
    acc.push(&utf8(vec![Some("a"), None, Some("b"), None]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["a", "b"]);
    assert!(!more);
}

#[test]
fn multi_shard_union_mixed_encoding() {
    // Shard 0 dictionary-encoded, shard 1 plain Utf8 (the from_anndata vs
    // append encoding split). Union should dedup across both.
    let mut acc = DistinctAccumulator::new("c", None, false);
    acc.push(&dict(vec![Some("a"), Some("b")])).unwrap();
    acc.push(&utf8(vec![Some("b"), Some("c")])).unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["a", "b", "c"]);
    assert!(!more);
}

#[test]
fn limit_first_n_with_has_more() {
    let mut acc = DistinctAccumulator::new("c", Some(2), false);
    acc.push(&utf8(vec![Some("a"), Some("b"), Some("c"), Some("d")]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["a", "b"]);
    assert!(more);
}

#[test]
fn limit_exact_no_more() {
    let mut acc = DistinctAccumulator::new("c", Some(3), false);
    acc.push(&utf8(vec![Some("a"), Some("b"), Some("a"), Some("c")]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["a", "b", "c"]);
    assert!(!more);
}

#[test]
fn done_short_circuits_across_shards() {
    let mut acc = DistinctAccumulator::new("c", Some(2), false);
    acc.push(&utf8(vec![Some("a"), Some("b"), Some("c")]))
        .unwrap();
    // Overflow value "c" already seen → caller can stop before the next shard.
    assert!(acc.done());
}

#[test]
fn sort_orders_and_truncates() {
    let mut acc = DistinctAccumulator::new("c", Some(2), true);
    acc.push(&utf8(vec![
        Some("delta"),
        Some("alpha"),
        Some("charlie"),
        Some("bravo"),
    ]))
    .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["alpha", "bravo"]);
    assert!(more);
}

#[test]
fn sort_no_limit_full_sorted() {
    let mut acc = DistinctAccumulator::new("c", None, true);
    acc.push(&utf8(vec![Some("z"), Some("a"), Some("m")]))
        .unwrap();
    let (vals, more) = acc.finish();
    assert_eq!(vals, vec!["a", "m", "z"]);
    assert!(!more);
}

#[test]
fn dict_superset_includes_unreferenced_entries() {
    // A non-compact dictionary: values carries "ghost" but no key references
    // it. The fast path scans values, so "ghost" is (intentionally) surfaced.
    let values = StringArray::from(vec!["a", "b", "ghost"]);
    let keys = Int32Array::from(vec![0, 1, 0]); // never indexes 2
    let arr = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap();
    let mut acc = DistinctAccumulator::new("c", None, false);
    acc.push(&(Arc::new(arr) as ArrayRef)).unwrap();
    let (vals, _) = acc.finish();
    assert!(vals.contains(&"ghost".to_string()));
}

#[test]
fn non_string_dtype_rejected() {
    let mut acc = DistinctAccumulator::new("counts", None, false);
    let err = acc
        .push(&(Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef))
        .unwrap_err();
    assert!(matches!(err, ScxError::UnsupportedColumnType { .. }));
}
