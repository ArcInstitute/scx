//! Tests for the empty-match category diagnostic (dogfood E1).

use std::sync::Arc;

use arrow::array::{ArrayRef, DictionaryArray, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};

use super::*;
use crate::parse_predicate;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// obs with the three shapes the diagnostic must handle: a plain `Utf8`
/// column, a dictionary-encoded categorical (the common on-disk form), and a
/// numeric column that must be ignored.
fn obs_batch() -> RecordBatch {
    let ethnicity: ArrayRef = Arc::new(StringArray::from(vec![
        "European American",
        "Asian",
        "unknown",
        "European American",
    ]));
    let keys = Int32Array::from(vec![0, 1, 2, 0]);
    let values = StringArray::from(vec!["B cell", "NK cell", "T cell"]);
    let cell_type: ArrayRef =
        Arc::new(DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values)).unwrap());
    let score: ArrayRef = Arc::new(Float64Array::from(vec![0.1, 0.2, 0.3, 0.4]));

    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("self_reported_ethnicity", DataType::Utf8, true),
            Field::new(
                "cell_type",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new("score", DataType::Float64, true),
        ])),
        vec![ethnicity, cell_type, score],
    )
    .unwrap()
}

fn cats(values: &[&str]) -> BTreeSet<String> {
    values.iter().map(|s| s.to_string()).collect()
}

// ---------------------------------------------------------------------------
// Term collection
// ---------------------------------------------------------------------------

#[test]
fn collects_string_equality_through_and_or_not() {
    let schema = obs_batch().schema();
    let pred = parse_predicate(
        "(self_reported_ethnicity == 'European' and cell_type == 'B cell') \
         or not (cell_type == 'Q cell')",
        &schema,
        "obs",
    )
    .unwrap();
    let mut terms = Vec::new();
    string_equality_terms(&pred, &mut terms);
    assert_eq!(
        terms,
        vec![
            (
                "self_reported_ethnicity".to_string(),
                "European".to_string()
            ),
            ("cell_type".to_string(), "B cell".to_string()),
            ("cell_type".to_string(), "Q cell".to_string()),
        ]
    );
}

#[test]
fn collects_every_member_of_an_in_list() {
    let schema = obs_batch().schema();
    let pred = parse_predicate("cell_type in ['B cell', 'Q cell']", &schema, "obs").unwrap();
    let mut terms = Vec::new();
    string_equality_terms(&pred, &mut terms);
    assert_eq!(
        terms,
        vec![
            ("cell_type".to_string(), "B cell".to_string()),
            ("cell_type".to_string(), "Q cell".to_string()),
        ]
    );
}

/// `Ne` and the numeric comparisons are deliberately not collected: a typo'd
/// `!=` returns *every* row, so it can never be why a query matched zero, and
/// this module only ever runs on a zero-row result. Pin the omission so it reads
/// as a decision rather than a gap.
#[test]
fn ignores_ne_and_numeric_comparisons() {
    let schema = obs_batch().schema();
    for expr in [
        "self_reported_ethnicity != 'European'",
        "score > 0.5",
        "score == 0.99",
    ] {
        let pred = parse_predicate(expr, &schema, "obs").unwrap();
        let mut terms = Vec::new();
        string_equality_terms(&pred, &mut terms);
        assert!(
            terms.is_empty(),
            "{expr} must not be collected, got {terms:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Category extraction
// ---------------------------------------------------------------------------

#[test]
fn reads_categories_from_utf8_and_dictionary_columns() {
    let batch = obs_batch();
    assert_eq!(
        column_categories(&batch, "self_reported_ethnicity").unwrap(),
        cats(&["Asian", "European American", "unknown"])
    );
    // The dictionary's *values* child, so a category no surviving row uses is
    // still known — "T cell" is in the vocabulary but keys only reference 0..2.
    assert_eq!(
        column_categories(&batch, "cell_type").unwrap(),
        cats(&["B cell", "NK cell", "T cell"])
    );
}

#[test]
fn numeric_and_absent_columns_have_no_category_set() {
    let batch = obs_batch();
    assert!(
        column_categories(&batch, "score").is_none(),
        "a numeric column has no category set to be missing from"
    );
    assert!(column_categories(&batch, "nope").is_none());
}

// ---------------------------------------------------------------------------
// Closest-match heuristic
// ---------------------------------------------------------------------------

/// The report's own case. `'European'` vs `'European American'` scores ~0.53 on
/// normalized Levenshtein — below the 0.6 bar the column-name suggester uses —
/// so a pure-Levenshtein rule would print no suggestion at all for the one
/// example we know a real user hit. Hence the prefix rule.
#[test]
fn prefix_match_wins_where_levenshtein_would_give_up() {
    let categories = cats(&["European American", "Asian", "unknown"]);
    assert!(
        strsim::normalized_levenshtein("European", "European American") < 0.6,
        "anti-vacuous: this pair must be BELOW the Levenshtein bar, or the \
         prefix rule is not what makes the suggestion appear"
    );
    assert_eq!(
        closest_category("European", &categories),
        Some("European American".to_string())
    );
}

#[test]
fn prefix_match_is_case_insensitive_and_prefers_the_shortest_extension() {
    let categories = cats(&["B cell", "B cell, IgG-negative", "T cell"]);
    assert_eq!(
        closest_category("b cell", &categories),
        Some("B cell".to_string()),
        "the least presumptuous guess is the shortest extension"
    );
}

#[test]
fn levenshtein_catches_a_transposition_with_no_shared_prefix() {
    let categories = cats(&["monocyte", "lymphocyte"]);
    assert_eq!(
        closest_category("moncyote", &categories),
        Some("monocyte".to_string())
    );
}

/// An unrelated string must NOT be dressed up as a suggestion — a wrong "did
/// you mean" is worse than none, because it sends the user to re-run a query
/// that will also return nothing.
#[test]
fn no_suggestion_when_nothing_is_close() {
    let categories = cats(&["European American", "Asian", "unknown"]);
    assert_eq!(closest_category("zzzzzzzzzz", &categories), None);
}

// ---------------------------------------------------------------------------
// Index-sourced vocabulary
// ---------------------------------------------------------------------------

#[test]
fn categories_come_from_a_covering_index_without_touching_obs() {
    use crate::index::{CategoricalEntry, CategoricalIndex};

    let index = PredicateIndex {
        version: 1,
        columns: vec![IndexedColumn::Categorical(CategoricalIndex {
            column_name: "cell_type".to_string(),
            entries: vec![
                CategoricalEntry {
                    value: "B cell".to_string(),
                    shard_ranges: vec![],
                },
                CategoricalEntry {
                    value: "T cell".to_string(),
                    shard_ranges: vec![],
                },
            ],
        })],
    };
    assert_eq!(
        categories_from_index(&index, "cell_type").unwrap(),
        cats(&["B cell", "T cell"])
    );
    assert!(
        categories_from_index(&index, "self_reported_ethnicity").is_none(),
        "an uncovered column must fall through to the obs read, not report an \
         empty vocabulary (which would call every value a miss)"
    );
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

#[test]
fn render_names_the_value_the_column_the_count_and_the_suggestion() {
    let miss = CategoryMiss {
        column: "self_reported_ethnicity".to_string(),
        value: "European".to_string(),
        n_categories: 22,
        closest: Some("European American".to_string()),
    };
    let s = miss.render();
    for needle in [
        "European",
        "self_reported_ethnicity",
        "22 known categories",
        "European American",
    ] {
        assert!(s.contains(needle), "render() must mention {needle:?}: {s}");
    }
}

#[test]
fn render_singularises_one_category_and_omits_an_absent_suggestion() {
    let miss = CategoryMiss {
        column: "disease".to_string(),
        value: "zzz".to_string(),
        n_categories: 1,
        closest: None,
    };
    let s = miss.render();
    assert!(
        s.contains("1 known category"),
        "must not say '1 categories': {s}"
    );
    assert!(
        !s.contains("closest"),
        "no suggestion means no `closest match` clause: {s}"
    );
}

// ---------------------------------------------------------------------------
// End-to-end through a real file + SectionReader
// ---------------------------------------------------------------------------

/// Write a minimal SCX file carrying [`obs_batch`] so `diagnose_category_misses`
/// runs against a real `SectionReader` (the obs-read branch — this file has no
/// predicate index).
fn write_fixture(dir: &tempfile::TempDir) -> std::path::PathBuf {
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;

    let path = dir.path().join("cats.scx");
    let header = FileHeader::new_single_modality(4, 3, 0, 16384, 0, 0);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&obs_batch()).unwrap();
    let var = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
        vec![Arc::new(StringArray::from(vec!["g0", "g1", "g2"])) as ArrayRef],
    )
    .unwrap();
    writer.write_var(&var).unwrap();
    writer
        .write_csr_shard(
            &[0, 1, 2, 3, 4],
            &[0, 1, 2, 0],
            &[1u8, 2, 3, 4],
            scx_codec::CodecId::None,
            scx_codec::ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();
    path
}

#[test]
fn diagnoses_a_miss_and_stays_silent_on_a_hit() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir);
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let schema = obs_batch().schema();

    // The report's exact predicate: a real column, a value that is not one of
    // its categories.
    let miss = parse_predicate("self_reported_ethnicity == 'European'", &schema, "obs").unwrap();
    let found = diagnose_category_misses(&reader, std::slice::from_ref(&miss)).unwrap();
    assert_eq!(found.len(), 1, "expected one miss, got {found:?}");
    assert_eq!(found[0].column, "self_reported_ethnicity");
    assert_eq!(found[0].value, "European");
    assert_eq!(found[0].n_categories, 3);
    assert_eq!(found[0].closest.as_deref(), Some("European American"));

    // Control — a value that IS a category must produce nothing. Without this,
    // a function that flagged everything would pass the assertion above.
    let hit = parse_predicate(
        "self_reported_ethnicity == 'European American'",
        &schema,
        "obs",
    )
    .unwrap();
    assert!(
        diagnose_category_misses(&reader, std::slice::from_ref(&hit))
            .unwrap()
            .is_empty(),
        "a predicate naming a real category must not be reported as a miss"
    );

    // A numeric predicate that matches nothing is a genuine zero, not a typo.
    let numeric = parse_predicate("score == 99.0", &schema, "obs").unwrap();
    assert!(
        diagnose_category_misses(&reader, std::slice::from_ref(&numeric))
            .unwrap()
            .is_empty(),
        "a numeric column has no category set; reporting a miss would be a lie"
    );

    // No predicates at all → no work, no misses.
    assert!(diagnose_category_misses(&reader, &[]).unwrap().is_empty());
}

#[test]
fn diagnoses_a_dictionary_encoded_column() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir);
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let schema = obs_batch().schema();

    let pred = parse_predicate("cell_type == 'B cel'", &schema, "obs").unwrap();
    let found = diagnose_category_misses(&reader, std::slice::from_ref(&pred)).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].n_categories, 3, "the dictionary has 3 values");
    assert_eq!(found[0].closest.as_deref(), Some("B cell"));
}

/// Every miss in a compound predicate is reported, and each appears once even
/// when the same term is written twice.
#[test]
fn reports_each_distinct_miss_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(&dir);
    let reader = scx_format_io::reader::ScxReader::open(&path).unwrap();
    let schema = obs_batch().schema();

    let pred = parse_predicate(
        "self_reported_ethnicity == 'European' and cell_type == 'Q cell' \
         and self_reported_ethnicity == 'European'",
        &schema,
        "obs",
    )
    .unwrap();
    let found = diagnose_category_misses(&reader, std::slice::from_ref(&pred)).unwrap();
    let mut names: Vec<(&str, &str)> = found
        .iter()
        .map(|m| (m.column.as_str(), m.value.as_str()))
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        vec![
            ("cell_type", "Q cell"),
            ("self_reported_ethnicity", "European"),
        ]
    );
}
