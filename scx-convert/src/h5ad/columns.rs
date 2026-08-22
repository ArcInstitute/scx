//! The dataframe export's layout pre-pass.
//!
//! HDF5 datasets are pre-allocated, so the streaming writer has to know each
//! column's final shape and dtype *before* it writes a byte. This module is
//! that decision: which columns need anndata's nullable representation, which
//! are dictionary-encoded in **any** shard (and so must be declared
//! categorical for **all** of them), and — only when a row filter is active —
//! which categories the kept rows actually reference.
//!
//! The write pass itself is [`super::column_stream`].

use super::categorical::{
    cat_class_mismatch_err, cat_value_class_eq, local_categorical_view, CatValues,
};
use super::column_stream::parse_shard_row_start;
use crate::pipeline::ConvertError;
use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, FieldRef, Schema};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Cross-shard pre-pass result for the streaming dataframe writer. Both
/// signals must be known before any HDF5 dataset is allocated, so they
/// are computed together in one decode pass over the metadata shards.
pub(crate) struct ColumnExportLayout {
    /// Aligned with `schema.fields()`: `true` ⇔ the field is an `Int32` /
    /// `Int64` / `Utf8` / `LargeUtf8` column that actually contains ≥1 null
    /// across the shards (needs anndata's nullable group encoding).
    pub needs_nullable: Vec<bool>,
    /// Aligned with `schema.fields()`: `Some(field)` ⇔ the column is a
    /// `Dictionary` in *at least one* shard (first such shard field seen,
    /// carrying its value type + categorical metadata). Used to build the
    /// unified export schema so a column that is categorical in any shard
    /// is exported as an h5ad categorical even when shard 0 is plain.
    pub dict_fields: Vec<Option<FieldRef>>,
    /// Aligned with `schema.fields()`: `Some(set)` ⇔ a row filter is active
    /// **and** the column is categorical, so the exported vocabulary is
    /// restricted to the category values at least one kept row references.
    /// `None` means "keep every declared category" — the unfiltered case.
    ///
    /// Populated by a second decode pass ([`scan_used_categories`]) that runs
    /// **only** when a keep mask is passed, because it is the one signal that
    /// cannot be derived from declared dictionaries alone. Without a filter the
    /// exporter never prunes, so the pass never runs and an ordinary export
    /// still costs exactly one pre-scan.
    pub used_categories: Vec<Option<UsedCategories>>,
}

impl ColumnExportLayout {
    /// The layout of a frame with no nullable columns and no categoricals:
    /// every column a plain dataset. Only meaningful when the caller already
    /// knows that — tests that drive [`write_dataframe_group_from_shards`]
    /// directly with a hand-built frame, where scanning would just restate the
    /// fixture — which is why it is `#[cfg(test)]`: production callers must go
    /// through [`scan_column_export_layout`], whose answer is measured.
    #[cfg(test)]
    pub fn all_plain(n_fields: usize) -> Self {
        ColumnExportLayout {
            needs_nullable: vec![false; n_fields],
            dict_fields: vec![None; n_fields],
            used_categories: (0..n_fields).map(|_| None).collect(),
        }
    }
}

/// The category values a filtered export actually keeps, per value class.
/// Mirrors [`CatValues`] / [`super::categorical::CatAccum`]; floats are keyed
/// by bit pattern for
/// the same reason (category values are exact labels, never computed).
pub(crate) enum UsedCategories {
    Str(HashSet<String>),
    Int(HashSet<i64>),
    Float(HashSet<u64>),
}

/// Pre-scan metadata shards to decide (a) which integer / string columns
/// need anndata's nullable group encoding, and (b) which columns are a
/// `Dictionary` in any shard (so the unified export schema can declare
/// them categorical — see
/// [`super::column_stream::write_dataframe_group_from_shards`]).
///
/// The streaming writer must allocate each HDF5 dataset (plain vs.
/// nullable group vs. categorical group) before it sees any shard data,
/// so neither signal can be derived from the shard-0 schema alone: Arrow
/// field nullability is set unconditionally by pandas → Arrow, and an
/// append-grown axis can mix `Dictionary` and plain shards for the same
/// column. The cost is one decode pass over the metadata-only shards.
///
/// Unlike the previous `scan_nullable_columns`, this pass does **not**
/// early-exit: a column cannot be proven "plain in every shard" (and thus
/// not a categorical) until every shard has been inspected. It is still a
/// single pass — no new pass is introduced — but it always runs to
/// completion. For obs/var metadata (rows, not X) the cost is small.
///
/// Mirrors `scx_format_io::reconcile_dictionary_representations` on the
/// read side by defending against shards that disagree on field count /
/// name (positional capture would otherwise mis-assign columns) or on a
/// categorical column's value **class** (`Dictionary(_, Utf8)` in one
/// shard vs `Dictionary(_, Int64)` in another is genuine corruption).
///
/// `make_shards` is a *factory*, not an iterator, because a filtered export
/// needs a second pass ([`scan_used_categories`]) that cannot be planned until
/// this one has decided which columns are categorical. `keep_mask` is `None`
/// for var and for an unfiltered obs export, and that is the case that still
/// costs exactly one pass.
pub(crate) fn scan_column_export_layout<I, F>(
    make_shards: F,
    schema: &Schema,
    keep_mask: Option<&[bool]>,
) -> Result<ColumnExportLayout, ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format_io::error::ScxError>>,
    F: Fn() -> I,
{
    let shards = make_shards();
    let n_fields = schema.fields().len();
    let eligible: Vec<bool> = schema
        .fields()
        .iter()
        .map(|f| {
            matches!(
                f.data_type(),
                DataType::Int32 | DataType::Int64 | DataType::Utf8 | DataType::LargeUtf8
            )
        })
        .collect();
    let mut needs_nullable = vec![false; n_fields];
    let mut dict_fields: Vec<Option<FieldRef>> = vec![None; n_fields];
    for batch_result in shards {
        let batch = batch_result?;
        let batch_schema = batch.schema();

        // Defend against producers that emit shards disagreeing on column
        // count / name (positional capture below would otherwise silently
        // mis-assign columns). Mirrors the read-side reconcile guard.
        if batch.num_columns() != n_fields {
            return Err(ConvertError::Other(format!(
                "shard schema mismatch: dataframe schema has {n_fields} columns but a shard \
                 has {} (columns must match by name and order across shards)",
                batch.num_columns()
            )));
        }
        for (i, field) in schema.fields().iter().enumerate() {
            let shard_field = batch_schema.field(i);
            if shard_field.name() != field.name() {
                return Err(ConvertError::Other(format!(
                    "shard schema mismatch: column {i} is '{}' in the dataframe schema but '{}' \
                     in a shard (columns must match by name and order across shards)",
                    field.name(),
                    shard_field.name(),
                )));
            }

            if eligible[i] && !needs_nullable[i] && batch.column(i).null_count() > 0 {
                needs_nullable[i] = true;
            }

            if let DataType::Dictionary(_, value_type) = shard_field.data_type() {
                match &dict_fields[i] {
                    None => dict_fields[i] = Some(batch_schema.fields()[i].clone()),
                    Some(seen) => {
                        // Two shards declare this column categorical with
                        // different value classes → corruption, not an
                        // append-grown layout we can reconcile.
                        let seen_vt = match seen.data_type() {
                            DataType::Dictionary(_, v) => v.as_ref(),
                            _ => unreachable!("dict_fields only stores Dictionary fields"),
                        };
                        if !cat_value_class_eq(seen_vt, value_type.as_ref()) {
                            return Err(ConvertError::Other(format!(
                                "shard schema mismatch: categorical column '{}' is a dictionary \
                                 of {seen_vt:?} in one shard but {:?} in another",
                                field.name(),
                                value_type.as_ref(),
                            )));
                        }
                    }
                }
            }
        }
    }
    let used_categories = match keep_mask {
        // Nothing is filtered out, so nothing is pruned: every declared
        // category is exported, and the second pass is not run at all.
        None => (0..n_fields).map(|_| None).collect(),
        Some(mask) => scan_used_categories(make_shards(), schema, &dict_fields, mask)?,
    };

    Ok(ColumnExportLayout {
        needs_nullable,
        dict_fields,
        used_categories,
    })
}

/// Second decode pass over the metadata shards, run **only** for a filtered
/// export: collect, per categorical column, the category values at least one
/// *kept* row references.
///
/// This is the one signal the declared dictionaries cannot supply. The
/// exporter's rule is anndata's `remove_unused_categories`-on-subset rule —
/// a filtered frame drops the levels its surviving rows no longer use, an
/// unfiltered one keeps the vocabulary the user declared — and deciding it
/// per category requires looking at rows, before any code is written to the
/// pre-allocated `codes` dataset.
///
/// It cannot be fused into [`scan_column_export_layout`]'s pass: which columns
/// are categorical is only known once *every* shard has been inspected (a
/// column plain in shard 0 and a `Dictionary` in shard 5 is categorical), and
/// interning every plain column on the chance it might be promoted would mean
/// interning the obs index — one entry per cell.
fn scan_used_categories<I>(
    shards: I,
    schema: &Schema,
    dict_fields: &[Option<FieldRef>],
    keep_mask: &[bool],
) -> Result<Vec<Option<UsedCategories>>, ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format_io::error::ScxError>>,
{
    let n_fields = schema.fields().len();
    let mut used: Vec<Option<UsedCategories>> = (0..n_fields)
        .map(|i| {
            dict_fields[i].as_ref().map(|f| match f.data_type() {
                DataType::Dictionary(_, v) => match v.as_ref() {
                    DataType::Float32 | DataType::Float64 => UsedCategories::Float(HashSet::new()),
                    DataType::Utf8 | DataType::LargeUtf8 => UsedCategories::Str(HashSet::new()),
                    _ => UsedCategories::Int(HashSet::new()),
                },
                // `dict_fields` only ever stores `Dictionary` fields.
                _ => UsedCategories::Str(HashSet::new()),
            })
        })
        .collect();
    if used.iter().all(|u| u.is_none()) {
        return Ok(used);
    }

    let mut cumulative_rows: usize = 0;
    for batch_result in shards {
        let batch = batch_result?;
        let n_shard_rows = batch.num_rows();
        // Same offset resolution as the write pass; a disagreement between the
        // stamp and the running count is that pass's error to raise, and this
        // one never reaches it because the export aborts first.
        let row_start = parse_shard_row_start(&batch).unwrap_or(cumulative_rows);
        for (i, slot) in used.iter_mut().enumerate() {
            let Some(slot) = slot.as_mut() else { continue };
            let name = schema.fields()[i].name();
            let (local_codes, local_values) = local_categorical_view(batch.column(i), name)?;
            for (r, &code) in local_codes.iter().enumerate().take(n_shard_rows) {
                if code < 0 {
                    continue;
                }
                let global_row = row_start + r;
                if !keep_mask.get(global_row).copied().unwrap_or(false) {
                    continue;
                }
                let idx = code as usize;
                match (&mut *slot, &local_values) {
                    (UsedCategories::Str(s), CatValues::Str(vals)) => {
                        let v = vals[idx].as_str();
                        if !s.contains(v) {
                            s.insert(v.to_string());
                        }
                    }
                    (UsedCategories::Int(s), CatValues::Int(vals)) => {
                        s.insert(vals[idx]);
                    }
                    (UsedCategories::Float(s), CatValues::Float(vals)) => {
                        s.insert(vals[idx].to_bits());
                    }
                    _ => return Err(cat_class_mismatch_err(name)),
                }
            }
        }
        cumulative_rows += n_shard_rows;
    }
    Ok(used)
}

/// Build the unified export schema: the shard-0 `schema` with every column
/// that is a `Dictionary` in *any* shard ([`ColumnExportLayout::dict_fields`])
/// re-declared as that dictionary type, so a categorical column is exported as
/// an h5ad categorical even when shard 0 happened to be plain (the reverse of
/// the append-grown layout). The declared column **name** and **nullability**
/// are preserved from the shard-0 schema; the `data_type` comes from the
/// captured dictionary field; categorical metadata (`CATEGORICAL_ORDERED_KEY`,
/// etc.) is the union of both (the captured dict field wins on conflict) so the
/// `ordered` bit survives whichever shard carried it. Columns that are plain in
/// every shard are returned unchanged, so homogeneous files produce an
/// identical schema (no behavior change).
pub(crate) fn build_unified_export_schema(schema: &Schema, layout: &ColumnExportLayout) -> Schema {
    let fields: Vec<FieldRef> = schema
        .fields()
        .iter()
        .enumerate()
        .map(
            |(i, schema_field)| match layout.dict_fields.get(i).and_then(|o| o.as_ref()) {
                None => schema_field.clone(),
                Some(dict_field) => {
                    let mut metadata: HashMap<String, String> = schema_field.metadata().clone();
                    for (k, v) in dict_field.metadata() {
                        metadata.insert(k.clone(), v.clone());
                    }
                    Arc::new(
                        Field::new(
                            schema_field.name(),
                            dict_field.data_type().clone(),
                            schema_field.is_nullable(),
                        )
                        .with_metadata(metadata),
                    )
                }
            },
        )
        .collect();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}
