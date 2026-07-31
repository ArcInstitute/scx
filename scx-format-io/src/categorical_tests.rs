//! Unit tests for [`GlobalCategoryAccum`], driven by synthetic Arrow arrays so
//! they need no `.scx` fixture. File-level round trips live in
//! `reader_tests.rs`.

use super::*;

use std::sync::Arc;

use arrow::array::{DictionaryArray, Int8Array, LargeStringArray, StringArray};
use arrow::datatypes::Int8Type;

/// Plain `Utf8` shard from `Option<&str>` rows.
fn utf8(rows: &[Option<&str>]) -> ArrayRef {
    Arc::new(StringArray::from(rows.to_vec()))
}

/// `Dictionary(Int8, Utf8)` shard from explicit keys + a declared vocabulary,
/// so a test can build a *non-compact* dictionary (unreferenced entries) and a
/// null key independently.
fn dict(keys: &[Option<i8>], values: &[&str]) -> ArrayRef {
    let keys = Int8Array::from(keys.to_vec());
    let values: ArrayRef = Arc::new(StringArray::from(values.to_vec()));
    Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap())
}

fn run(shards: Vec<ArrayRef>) -> (Vec<i32>, Vec<String>) {
    let mut acc = GlobalCategoryAccum::new("col", 0);
    for s in &shards {
        acc.push(s).unwrap();
    }
    acc.finish()
}

/// Decode back to strings so assertions read as data, not as codes.
fn decode<'a>(codes: &[i32], categories: &'a [String]) -> Vec<Option<&'a str>> {
    codes
        .iter()
        .map(|&c| {
            if c < 0 {
                None
            } else {
                Some(categories[c as usize].as_str())
            }
        })
        .collect()
}

#[test]
fn plain_utf8_single_shard_codes_and_first_seen_order() {
    let (codes, cats) = run(vec![utf8(&[Some("b"), Some("a"), Some("b")])]);
    assert_eq!(cats, vec!["b", "a"], "first-seen order, not sorted");
    assert_eq!(codes, vec![0, 1, 0]);
}

/// The load-bearing case: shards with **disjoint** vocabularies must fold into
/// one global vocabulary with codes remapped per shard. A shared-vocab fixture
/// cannot detect a broken unify.
#[test]
fn disjoint_per_shard_vocabularies_unify() {
    let (codes, cats) = run(vec![
        utf8(&[Some("a"), Some("b")]),
        utf8(&[Some("c"), Some("a")]),
        utf8(&[Some("d")]),
    ]);
    assert_eq!(cats, vec!["a", "b", "c", "d"]);
    assert_eq!(
        decode(&codes, &cats),
        vec![Some("a"), Some("b"), Some("c"), Some("a"), Some("d")]
    );
}

/// Per-shard dictionaries have *local* codes: the same local key means different
/// values in different shards. If the local→global remap were skipped, codes
/// would be shaped correctly and wrong.
#[test]
fn dictionary_shards_with_colliding_local_codes_are_remapped() {
    // Shard 0: local 0 = "x". Shard 1: local 0 = "y".
    let (codes, cats) = run(vec![
        dict(&[Some(0), Some(1)], &["x", "z"]),
        dict(&[Some(0), Some(1)], &["y", "x"]),
    ]);
    assert_eq!(cats, vec!["x", "z", "y"]);
    assert_eq!(
        decode(&codes, &cats),
        vec![Some("x"), Some("z"), Some("y"), Some("x")],
        "local code 0 must map to 'x' in shard 0 and 'y' in shard 1"
    );
}

/// A file grown by `append` carries `Dictionary` base shards and plain `Utf8`
/// appended shards. Both must fold into one vocabulary.
#[test]
fn mixed_dictionary_and_plain_shards_share_one_vocabulary() {
    let (codes, cats) = run(vec![
        dict(&[Some(0), Some(1)], &["a", "b"]),
        utf8(&[Some("b"), Some("c")]),
        dict(&[Some(0)], &["c"]),
    ]);
    assert_eq!(cats, vec!["a", "b", "c"]);
    assert_eq!(
        decode(&codes, &cats),
        vec![Some("a"), Some("b"), Some("b"), Some("c"), Some("c")]
    );
    // "b" and "c" must each have exactly one code despite arriving via both
    // representations.
    assert_eq!(cats.len(), 3, "no duplicate categories across encodings");
}

#[test]
fn nulls_become_minus_one_in_both_representations() {
    let (codes, cats) = run(vec![
        utf8(&[Some("a"), None, Some("b")]),
        dict(&[Some(0), None], &["c"]),
    ]);
    assert_eq!(cats, vec!["a", "b", "c"]);
    assert_eq!(codes, vec![0, -1, 1, 2, -1]);
}

/// A literal `"NaN"` string is a real category, distinct from a missing value.
/// The training-internal `CategoryDict` conflates these by design; this accessor
/// must not, because pandas does not.
#[test]
fn literal_nan_string_is_distinct_from_null() {
    let (codes, cats) = run(vec![utf8(&[Some("NaN"), None, Some("NaN")])]);
    assert_eq!(cats, vec!["NaN"], "no synthetic level is appended for null");
    assert_eq!(codes, vec![0, -1, 0]);
    assert!(
        !cats.iter().any(|c| c.is_empty()),
        "null must not materialise as an empty-string category"
    );
}

/// An empty string is likewise a real category, not a null.
#[test]
fn empty_string_is_a_category_not_a_null() {
    let (codes, cats) = run(vec![utf8(&[Some(""), None])]);
    assert_eq!(cats, vec![""]);
    assert_eq!(codes, vec![0, -1]);
}

/// Unreferenced dictionary entries are retained, matching pandas keeping unused
/// levels and matching `DistinctAccumulator`'s documented superset behaviour.
#[test]
fn unreferenced_dictionary_entries_are_kept_as_categories() {
    // Only local 0 ("a") is referenced; "unused" is declared but never used.
    let (codes, cats) = run(vec![dict(&[Some(0), Some(0)], &["a", "unused"])]);
    assert_eq!(cats, vec!["a", "unused"]);
    assert_eq!(codes, vec![0, 0]);
}

#[test]
fn large_utf8_is_accepted() {
    let arr: ArrayRef = Arc::new(LargeStringArray::from(vec![Some("a"), None, Some("a")]));
    let (codes, cats) = run(vec![arr]);
    assert_eq!(cats, vec!["a"]);
    assert_eq!(codes, vec![0, -1, 0]);
}

#[test]
fn non_string_dtype_is_rejected() {
    let arr: ArrayRef = Arc::new(Int8Array::from(vec![1i8, 2]));
    let mut acc = GlobalCategoryAccum::new("n_genes", 0);
    let err = acc
        .push(&arr)
        .expect_err("numeric columns must be rejected");
    let msg = err.to_string();
    assert!(msg.contains("n_genes"), "error must name the column: {msg}");
    assert!(
        msg.contains("string/categorical"),
        "error must say what is supported: {msg}"
    );
}

#[test]
fn n_rows_tracks_folded_rows() {
    let mut acc = GlobalCategoryAccum::new("col", 0);
    assert_eq!(acc.n_rows(), 0);
    acc.push(&utf8(&[Some("a"), Some("b")])).unwrap();
    assert_eq!(acc.n_rows(), 2);
    acc.push(&dict(&[Some(0)], &["c"])).unwrap();
    assert_eq!(acc.n_rows(), 3);
}

#[test]
fn empty_shard_contributes_nothing() {
    let (codes, cats) = run(vec![utf8(&[]), utf8(&[Some("a")]), utf8(&[])]);
    assert_eq!(cats, vec!["a"]);
    assert_eq!(codes, vec![0]);
}
