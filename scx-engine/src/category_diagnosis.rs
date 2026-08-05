//! Why did this predicate match nothing? (dogfood E1)
//!
//! A string equality against a categorical obs column that matches **no**
//! category is indistinguishable, in the query output, from a predicate that is
//! simply satisfied by zero cells:
//!
//! ```text
//! scx query census.scx --filter "self_reported_ethnicity == 'European'" --count
//! Pushdown: 0/31 shards eliminated by catalog stats (Level 1); 0 of 500000 …
//! 0
//! ```
//!
//! The real categories were `'European American'`, `'Asian'`, `'unknown'`, … —
//! so that `0` is a typo, not a finding, and it is the single most likely way to
//! silently mis-slice an atlas. SCX holds the complete value set at query time
//! (dictionary-encoded obs, plus a categorical predicate index when present), so
//! it can tell the two apart rather than leaving the user to load obs in pandas.
//!
//! This module answers the question only when it is worth asking — the caller
//! runs it after a query has already matched **zero** rows, so the obs read it
//! may perform is off the hot path. See [`diagnose_category_misses`].

use std::collections::BTreeSet;

use arrow::array::{Array, AsArray};
use arrow::datatypes::DataType;

use crate::error::Result;
use crate::index::{IndexedColumn, PredicateIndex};
use crate::predicate::{Predicate, ScalarValue};
use crate::reader::SectionReader;

/// A string literal in an equality/`in` predicate that is not among its
/// column's categories.
#[derive(Debug, Clone, PartialEq)]
pub struct CategoryMiss {
    /// The obs column the predicate tested.
    pub column: String,
    /// The literal that matched no category.
    pub value: String,
    /// Size of the column's **declared** vocabulary, so the user can see at a
    /// glance whether they mistyped one of 22 or one of 11,000.
    ///
    /// The declared set, not the observed one — it can exceed a pandas
    /// `nunique()` on the same column, because a categorical carries categories
    /// no surviving row uses (`self_reported_ethnicity` on `census_500k`
    /// declares 37 and uses 22). That is deliberate and load-bearing: a value in
    /// the vocabulary with zero rows is a *genuine* empty result, not a typo, so
    /// it must not be reported here at all. The word "known" in [`Self::render`]
    /// marks which set this counts.
    pub n_categories: usize,
    /// Nearest existing category, when one is close enough to be a plausible
    /// typo of `value`. `None` when nothing is close — an unrelated string
    /// should not be dressed up as a suggestion.
    pub closest: Option<String>,
}

impl CategoryMiss {
    /// One-line rendering for a CLI note / warning.
    pub fn render(&self) -> String {
        let mut s = format!(
            "no category {:?} in '{}' ({} known categor{})",
            self.value,
            self.column,
            self.n_categories,
            if self.n_categories == 1 { "y" } else { "ies" }
        );
        if let Some(ref c) = self.closest {
            s.push_str(&format!("; closest match {c:?}"));
        }
        s
    }
}

/// Collect every `(column, literal)` pair that a **string equality** in
/// `predicates` tests, walking `and` / `or` / `not`.
///
/// `Eq` and `In` only. A typo'd `Ne` mis-slices too, but by returning *every*
/// row rather than none — it can never be the reason a query matched zero, which
/// is the only condition under which this module runs. Diagnosing it needs a
/// different trigger, so promising it here would be a lie.
fn string_equality_terms(pred: &Predicate, out: &mut Vec<(String, String)>) {
    match pred {
        Predicate::Eq(col, ScalarValue::Utf8(v)) => out.push((col.clone(), v.clone())),
        Predicate::In(col, values) => {
            for v in values {
                if let ScalarValue::Utf8(s) = v {
                    out.push((col.clone(), s.clone()));
                }
            }
        }
        Predicate::And(a, b) | Predicate::Or(a, b) => {
            string_equality_terms(a, out);
            string_equality_terms(b, out);
        }
        Predicate::Not(inner) => string_equality_terms(inner, out),
        _ => {}
    }
}

/// Nearest category to `value`, or `None` when nothing is close enough.
///
/// Same normalized-Levenshtein-at-0.6 rule the column-name suggestions use
/// (`index::best_match`), so a mistyped *value* and a mistyped *column* get
/// consistent treatment. One addition: a candidate that contains `value` as a
/// prefix wins outright. That is the report's own case — `'European'` vs
/// `'European American'` scores only 0.53 on normalized Levenshtein (the length
/// gap dominates) and would otherwise be dropped, even though it is obviously
/// the intended value.
fn closest_category(value: &str, categories: &BTreeSet<String>) -> Option<String> {
    let lower = value.to_lowercase();
    let mut prefix_hits: Vec<&String> = categories
        .iter()
        .filter(|c| c.to_lowercase().starts_with(&lower))
        .collect();
    if !prefix_hits.is_empty() {
        // Shortest prefix extension is the least presumptuous guess.
        prefix_hits.sort_by_key(|c| (c.len(), c.as_str()));
        return Some(prefix_hits[0].clone());
    }
    categories
        .iter()
        .map(|c| (c, strsim::normalized_levenshtein(value, c)))
        .filter(|(_, s)| *s >= 0.6)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(c, _)| c.clone())
}

/// Distinct values of an obs column, for the string-ish types a categorical
/// predicate can target. `None` for any other dtype — a numeric column has no
/// "category set" to be missing from, and this diagnostic must stay silent
/// rather than invent one.
fn column_categories(batch: &arrow::array::RecordBatch, column: &str) -> Option<BTreeSet<String>> {
    let col = batch.column_by_name(column)?;
    let mut out = BTreeSet::new();
    match col.data_type() {
        DataType::Utf8 => {
            for v in col.as_string::<i32>().iter().flatten() {
                out.insert(v.to_string());
            }
        }
        DataType::LargeUtf8 => {
            for v in col.as_string::<i64>().iter().flatten() {
                out.insert(v.to_string());
            }
        }
        // Dictionary-encoded (the common on-disk shape for a categorical): read
        // the *values* child, not the decoded rows — the whole vocabulary is
        // right there, including categories no surviving row uses.
        DataType::Dictionary(key, value) if matches!(**value, DataType::Utf8) => {
            let values = match **key {
                DataType::Int8 => col.as_dictionary::<arrow::datatypes::Int8Type>().values(),
                DataType::Int16 => col.as_dictionary::<arrow::datatypes::Int16Type>().values(),
                DataType::Int32 => col.as_dictionary::<arrow::datatypes::Int32Type>().values(),
                DataType::Int64 => col.as_dictionary::<arrow::datatypes::Int64Type>().values(),
                DataType::UInt8 => col.as_dictionary::<arrow::datatypes::UInt8Type>().values(),
                DataType::UInt16 => col.as_dictionary::<arrow::datatypes::UInt16Type>().values(),
                DataType::UInt32 => col.as_dictionary::<arrow::datatypes::UInt32Type>().values(),
                DataType::UInt64 => col.as_dictionary::<arrow::datatypes::UInt64Type>().values(),
                _ => return None,
            };
            for v in values.as_string::<i32>().iter().flatten() {
                out.insert(v.to_string());
            }
        }
        _ => return None,
    }
    Some(out)
}

/// Categories of `column` as recorded in a categorical predicate index, if it
/// covers that column. Free relative to an obs read — the index already holds
/// the sorted value set.
fn categories_from_index(index: &PredicateIndex, column: &str) -> Option<BTreeSet<String>> {
    index.columns.iter().find_map(|c| match c {
        IndexedColumn::Categorical(cat) if cat.column_name == column => {
            Some(cat.entries.iter().map(|e| e.value.clone()).collect())
        }
        _ => None,
    })
}

/// Explain a zero-row result: which string literals in `predicates` name a
/// category that does not exist.
///
/// **Call this only when a query matched zero rows.** It may read the full obs
/// table (~0.5 s / 500k × 28 cols), which is fine as a one-off explanation of an
/// empty result and not fine per query.
///
/// Prefers a categorical predicate index when one covers the column, and reads
/// obs at most once for whatever it does not cover — so an unindexed column
/// (the report's `self_reported_ethnicity`, which no preset includes) is still
/// diagnosed. Columns that are absent or non-string are skipped silently: a
/// missing column already fails earlier with its own "Did you mean" error, and a
/// numeric column has no category set.
///
/// Returns misses in predicate order, deduplicated.
pub fn diagnose_category_misses(
    reader: &dyn SectionReader,
    predicates: &[Predicate],
) -> Result<Vec<CategoryMiss>> {
    let mut terms: Vec<(String, String)> = Vec::new();
    for p in predicates {
        string_equality_terms(p, &mut terms);
    }
    terms.dedup();
    if terms.is_empty() {
        return Ok(Vec::new());
    }

    let index = match reader.read_obs_predicate_index_bytes()? {
        Some(bytes) => PredicateIndex::read_from(&mut std::io::Cursor::new(bytes.as_slice())).ok(),
        None => None,
    };

    // Resolve each distinct column's vocabulary once, from the index when it
    // covers the column and from obs otherwise. `obs` is read lazily so a fully
    // indexed predicate set costs no obs decode at all.
    let mut vocab: std::collections::HashMap<String, Option<BTreeSet<String>>> =
        std::collections::HashMap::new();
    let mut obs: Option<arrow::array::RecordBatch> = None;
    for (col, _) in &terms {
        if vocab.contains_key(col) {
            continue;
        }
        let from_index = index.as_ref().and_then(|i| categories_from_index(i, col));
        let cats = match from_index {
            Some(c) => Some(c),
            None => {
                if obs.is_none() {
                    obs = Some(reader.read_obs()?);
                }
                column_categories(obs.as_ref().expect("read above"), col)
            }
        };
        vocab.insert(col.clone(), cats);
    }

    let mut out = Vec::new();
    for (col, value) in terms {
        let Some(Some(cats)) = vocab.get(&col) else {
            continue; // absent or non-string column
        };
        if cats.contains(&value) {
            continue;
        }
        let miss = CategoryMiss {
            closest: closest_category(&value, cats),
            n_categories: cats.len(),
            column: col,
            value,
        };
        if !out.contains(&miss) {
            out.push(miss);
        }
    }
    Ok(out)
}

#[cfg(test)]
#[path = "category_diagnosis_tests.rs"]
mod tests;
