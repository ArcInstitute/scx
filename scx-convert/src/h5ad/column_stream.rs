//! The dataframe column writer: nine encodings, one implementation each.
//!
//! [`write_dataframe_group_from_shards`] is the only dataframe-column writer
//! in the crate. [`write_dataframe_group_at`] is a whole-batch *driver* over
//! it — an assembled frame is one shard — which is what keeps the nine
//! encodings from existing twice. They did exist twice until ORG-11.16-1, and
//! they had already drifted on categorical vocabulary (review 11.1).
//!
//! Reads its layout from [`super::columns`] and its categorical decode from
//! [`super::categorical`].

use super::categorical::{
    cardinality_err, cat_class_mismatch_err, cat_value_class_eq, downcast_err,
    is_supported_cat_value_type, local_categorical_view, CatAccum, CatValues,
};
use super::columns::{
    build_unified_export_schema, scan_column_export_layout, ColumnExportLayout, UsedCategories,
};
use crate::h5_write_util::vlu;
use crate::pipeline::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};
use crate::CATEGORICAL_ORDERED_KEY;
use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, LargeStringArray,
    RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use hdf5::types::VarLenUnicode;
use ndarray::ArrayView1;

/// Resolve a categorical column's pandas `ordered` bit from the Arrow
/// `Field::metadata` the h5ad reader stamps ([`CATEGORICAL_ORDERED_KEY`]).
/// Non-categorical or metadata-less fields resolve to `false`.
fn field_ordered(field: &Field) -> bool {
    field
        .metadata()
        .get(CATEGORICAL_ORDERED_KEY)
        .map(|v| v == "true")
        .unwrap_or(false)
}

/// Whole-batch driver over [`write_dataframe_group_from_shards`]: a dataframe
/// already assembled in memory is one shard with no row filter.
///
/// This is the *only* difference between the two export directions. Both used
/// to carry a full implementation of the nine column encodings, selected at
/// runtime by whether the source had `ObsMetadataShard` sections, and they had
/// already drifted apart twice — once on `LargeUtf8` handling, and once on the
/// categorical vocabulary, which is review §11.1.
pub(crate) fn write_dataframe_group_at(
    parent: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    write_dataframe_group_filtered_at(parent, name, batch, None, sink)
}

/// [`write_dataframe_group_at`] with a row filter: the legacy single-section
/// obs path under a deletion vector.
///
/// The mask is passed *through* rather than applied to `batch` first, so the
/// unsharded path prunes unused categories on exactly the rule the sharded one
/// uses. Filtering first would hide the filter from the writer, which is how
/// the two came to disagree: `arrow`'s `filter` keeps a `DictionaryArray`'s
/// full dictionary, so a legacy file exported every declared category while a
/// sharded one exported only the used ones.
pub(crate) fn write_dataframe_group_filtered_at(
    parent: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
    keep_mask_opt: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let schema = batch.schema();
    let n_rows_kept = match keep_mask_opt {
        None => batch.num_rows(),
        Some(mask) => mask.iter().take(batch.num_rows()).filter(|&&b| b).count(),
    };
    let one_shard = || {
        std::iter::once::<Result<RecordBatch, scx_format_io::error::ScxError>>(Ok(batch.clone()))
    };
    let layout = scan_column_export_layout(one_shard, schema.as_ref(), keep_mask_opt)?;
    let unified = build_unified_export_schema(schema.as_ref(), &layout);
    write_dataframe_group_from_shards(
        parent,
        name,
        &unified,
        one_shard(),
        n_rows_kept,
        keep_mask_opt,
        layout,
        sink,
    )
}

/// Write the dataframe-level encoding attrs (`encoding-type="dataframe"`,
/// `encoding-version="0.2.0"`), resolve the pandas index field, and write the
/// `_index` attr. Prologue of [`write_dataframe_group_from_shards`], and so of
/// both its drivers — anndata.read_h5ad requires these on every dataframe
/// group, even an empty one.
///
/// Returns `(index_field_name, on_disk_index)` for the caller's per-column
/// loop. The index field is probed as: (1) the `pandas` schema metadata's
/// `index_columns` (the authoritative source — `pyarrow.Table.from_pandas`
/// stamps it; covers named and unnamed indexes), else (2) a literal
/// `__index_level_0__` / `_index` field, else (3) `schema.field(0)`.
/// pyarrow's canonical `__index_level_0__` (unnamed pandas index) is renamed
/// to anndata's `_index` literal on disk; named indexes keep their original
/// name.
///
/// Step (2) exists because step (3)'s premise — "field 0 already IS the index",
/// true of the CLI convert path — is **false for a file whose obs was rewritten
/// in place**. There, field order is whatever the caller's DataFrame had, and
/// pyarrow puts the index last. Files written before the `unify_dict_columns`
/// metadata fix carry no envelope at all, so without (2) their export silently
/// renamed every cell to the value of the first string column. (3) survives
/// only for genuinely envelope-less, index-field-less CLI output.
fn write_dataframe_header(
    group: &hdf5::Group,
    schema: &Schema,
) -> Result<(Option<String>, String), ConvertError> {
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("dataframe"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;

    let index_field_name: Option<String> = scx_format_io::resolve_index_columns(schema)
        .into_iter()
        .find(|n| schema.field_with_name(n).is_ok())
        .or_else(|| schema.fields().first().map(|f| f.name().clone()));

    let on_disk_index: String = match index_field_name.as_deref() {
        Some("__index_level_0__") => "_index".to_string(),
        Some(n) => n.to_string(),
        None => "_index".to_string(),
    };

    if !schema.fields().is_empty() {
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&vlu(&on_disk_index))?;
    }

    Ok((index_field_name, on_disk_index))
}

/// Write the `column-order` attr. anndata.read_h5ad requires it on every
/// dataframe group even when there are no non-index columns (it raises
/// `KeyError: "...can't locate attribute: 'column-order'"` otherwise), so a
/// length-0 array is written for the no-columns case — matching anndata's own
/// emission. SCX's reader handles the empty-attr case in
/// `read_dataframe_group`'s fallback branch. Shared by both dataframe writers.
fn write_column_order_attr(
    group: &hdf5::Group,
    col_order: &[VarLenUnicode],
) -> Result<(), ConvertError> {
    group
        .new_attr::<VarLenUnicode>()
        .shape(col_order.len())
        .create("column-order")?
        .write_raw(col_order)?;
    Ok(())
}

/// Streaming counterpart of [`write_dataframe_group_at`]. Pre-allocates
/// one HDF5 dataset per column at fixed size `n_rows_kept`, then drains
/// `shards` and hyperslab-writes the kept-row slice of each column.
///
/// Mirrors the pre-allocate-then-hyperslab pattern in
/// `stream_write::create_csr_triplet` + `stream_csr_to_group_at`
/// (the `/X` and `/layers/{name}` path). Peak RSS per column is bounded
/// to one shard's worth — atlas-scale obs no longer needs to live in
/// memory at once.
///
/// `schema` is taken from `ScxReader::read_obs_schema_logical_lossy()`
/// (resp. var). It carries the same `pandas` index metadata as a
/// `read_obs()` batch, so index resolution mirrors `write_dataframe_body`.
/// Per-shard batches may carry `LargeUtf8` / `Dictionary(_, LargeUtf8)`
/// even when the schema says narrow — the runtime dispatch accepts both.
///
/// `keep_mask_opt` is the global (length `n_obs`) deletion-vector keep
/// mask; `None` means no filtering. The mask is indexed by global row,
/// so this is consumed only on the obs axis (var has no DV).
///
/// `layout` is [`scan_column_export_layout`]'s result for the same shards and
/// the same `keep_mask_opt`, taken **by value** because the categorical
/// vocabularies move into the per-column writers. Its `needs_nullable` decides
/// plain-dataset vs. nullable-group per integer / string column (float columns
/// ignore it — always plain datasets with `NaN` at nulls), and its
/// `used_categories` decides which declared categories survive. Both must be
/// known before any HDF5 dataset is allocated, which is why they are scanned
/// rather than discovered while writing.
///
/// `schema` is the unified export schema from
/// [`build_unified_export_schema`]: a column that is a `Dictionary` in any
/// shard is declared categorical here even when shard 0 was plain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_dataframe_group_from_shards<I>(
    parent: &hdf5::Group,
    name: &str,
    schema: &Schema,
    shards: I,
    n_rows_kept: usize,
    keep_mask_opt: Option<&[bool]>,
    mut layout: ColumnExportLayout,
    sink: &mut WarningSink,
) -> Result<(), ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format_io::error::ScxError>>,
{
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;

    // Dataframe-level encoding attrs + pandas index resolution + `_index`
    // attr — shared prologue with the eager `write_dataframe_body`.
    let (index_field_name, on_disk_index) = write_dataframe_header(&group, schema)?;

    // Pre-allocate column writers and assemble `column-order` in schema
    // order (index excluded). Unsupported types are warn-and-skipped
    // inside `create_column_writer` (no dataset created) and must
    // also stay out of `column-order` so anndata's reader doesn't
    // look up a missing dataset.
    let mut col_order: Vec<VarLenUnicode> =
        Vec::with_capacity(schema.fields().len().saturating_sub(1));
    let mut col_writers: Vec<(usize, ColumnStreamWriter)> = Vec::new();
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let is_index = Some(field.name()) == index_field_name.as_ref();
        let on_disk_name: &str = if is_index {
            on_disk_index.as_str()
        } else {
            field.name()
        };
        // The index column is forced to a plain dataset (anndata requires
        // `_index` to be a plain dataset, never a nullable group).
        let want_nullable =
            !is_index && layout.needs_nullable.get(col_idx).copied().unwrap_or(false);
        let used = layout
            .used_categories
            .get_mut(col_idx)
            .and_then(|slot| slot.take());
        let writer = create_column_writer(
            &group,
            on_disk_name,
            field,
            n_rows_kept,
            want_nullable,
            used,
        )?;
        if matches!(writer, ColumnStreamWriter::Unsupported) {
            sink.emit(ConvertWarning::UnsupportedExportColumn {
                column: format!("{name}/{}", field.name()),
                dtype: format!("{:?}", field.data_type()),
            });
        } else if !is_index {
            col_order.push(vlu(field.name()));
        }
        col_writers.push((col_idx, writer));
    }

    write_column_order_attr(&group, &col_order)?;

    // One accumulator per column writer, positionally aligned with
    // `col_writers`; see `append_shard_to_column`.
    let mut coerced_nulls: Vec<u64> = vec![0; col_writers.len()];

    // Drain shards. Each shard's stamped `row_start` schema metadata
    // (set by the writer via `stamp_dense_shard_metadata`) gives its
    // global row offset; the cumulative shard row count is verified
    // against it for defense-in-depth against producers that might
    // emit shards out of order.
    let mut cumulative_rows: usize = 0;
    for batch_result in shards {
        let batch = batch_result?;
        let n_shard_rows = batch.num_rows();

        // Validate the shard's schema against the declared dataframe
        // schema before touching any HDF5 dataset. A producer that
        // emits the right column *count* but reordered or retyped
        // columns would otherwise write values into the wrong
        // hyperslab — the streaming path is part of the trust boundary
        // for pipeline-generated files, so this fails loudly instead.
        validate_shard_schema(schema, &batch, name)?;

        // Prefer the stamped `row_start` over cumulative counting so
        // out-of-order producers fail loudly instead of writing into
        // wrong hyperslab offsets. Legacy shards lack the stamp and
        // fall back to `cumulative_rows` — the next cross-check is a
        // no-op for them, but v2 sharded obs always stamps it.
        let row_start_global = parse_shard_row_start(&batch).unwrap_or(cumulative_rows);
        if row_start_global != cumulative_rows {
            return Err(ConvertError::Other(format!(
                "shard '{name}' row_start {row_start_global} does not match cumulative \
                 row count {cumulative_rows} — shards must arrive in order",
            )));
        }

        // Kept-row local indices for this shard.
        let kept_local: Vec<usize> = match keep_mask_opt {
            None => (0..n_shard_rows).collect(),
            Some(mask) => {
                let upper = row_start_global + n_shard_rows;
                if mask.len() < upper {
                    return Err(ConvertError::Other(format!(
                        "keep_mask length {} < shard upper row {upper} for '{name}' \
                         (catalog/header drift)",
                        mask.len(),
                    )));
                }
                (0..n_shard_rows)
                    .filter(|&i| mask[row_start_global + i])
                    .collect()
            }
        };

        // Called even for a shard with no kept rows: nothing is written, but a
        // categorical column still contributes its declared vocabulary, which
        // is a property of the column and not of which rows survive.
        for (slot, (col_idx, writer)) in coerced_nulls.iter_mut().zip(col_writers.iter_mut()) {
            let array = batch.column(*col_idx);
            let field_name = schema.fields()[*col_idx].name();
            append_shard_to_column(writer, array, &kept_local, field_name, slot)?;
        }

        cumulative_rows += n_shard_rows;
    }

    // Surface the one residual null-coercion case: a column written as a plain
    // dataset that nonetheless carried nulls. In practice that is only ever the
    // pandas index (`_index`), which anndata requires to be a plain dataset and
    // so can never use a nullable group; indexes virtually never carry nulls,
    // but when one does its null became `0` / `""` and that must be visible.
    for (count, (col_idx, _)) in coerced_nulls.iter().zip(col_writers.iter()) {
        if *count == 0 {
            continue;
        }
        let field = &schema.fields()[*col_idx];
        sink.emit(ConvertWarning::CoercedNulls {
            column: format!("{name}/{}", field.name()),
            dtype: format!("{:?}", field.data_type()),
            count: *count,
        });
    }

    // Validate every non-skipped column filled its pre-allocated
    // dataset exactly — a partial fill would leave default-initialised
    // trailing rows that look valid but encode incorrect data.
    for (col_idx, writer) in &col_writers {
        let field_name = schema.fields()[*col_idx].name();
        let written = column_writer_offset(writer);
        if let Some(written) = written {
            if written != n_rows_kept {
                return Err(ConvertError::Other(format!(
                    "column '{field_name}' wrote {written} rows but dataframe was \
                     pre-allocated to {n_rows_kept}",
                )));
            }
        }
    }

    // Finalize categorical writers (write `categories` + attrs).
    for (_, writer) in &col_writers {
        finalize_column_writer(writer)?;
    }

    Ok(())
}

/// Parse `row_start` from a shard's stamped schema metadata. Returns
/// `None` for legacy or non-stamped batches (callers fall back to
/// cumulative counting in that case).
pub(super) fn parse_shard_row_start(batch: &RecordBatch) -> Option<usize> {
    batch
        .schema_ref()
        .metadata()
        .get("row_start")
        .and_then(|s| s.parse::<u64>().ok())
        .map(|v| v as usize)
}

/// Number of rows written by a column streaming writer so far. Returns
/// `None` for `Unsupported` (no dataset to validate).
fn column_writer_offset(writer: &ColumnStreamWriter) -> Option<usize> {
    match writer {
        ColumnStreamWriter::Int32 { offset, .. }
        | ColumnStreamWriter::Int64 { offset, .. }
        | ColumnStreamWriter::Float32 { offset, .. }
        | ColumnStreamWriter::Float64 { offset, .. }
        | ColumnStreamWriter::Utf8 { offset, .. }
        | ColumnStreamWriter::Boolean { offset, .. }
        | ColumnStreamWriter::Categorical { offset, .. }
        | ColumnStreamWriter::Nullable { offset, .. } => Some(*offset),
        ColumnStreamWriter::Unsupported => None,
    }
}

/// Validate a per-shard `RecordBatch` against the declared dataframe
/// `schema`: same column count, same field names in the same order,
/// and logically-compatible data types. This is the streaming
/// counterpart of the assembly-time validation in
/// `assemble_sharded_metadata` — `obs_shards()` yields raw shards
/// lazily and skips that assembly, so without this a reordered or
/// retyped shard would silently land in the wrong HDF5 column.
fn validate_shard_schema(
    schema: &Schema,
    batch: &RecordBatch,
    name: &str,
) -> Result<(), ConvertError> {
    if batch.num_columns() != schema.fields().len() {
        return Err(ConvertError::Other(format!(
            "shard schema mismatch for '{name}': schema has {} fields, batch has {}",
            schema.fields().len(),
            batch.num_columns()
        )));
    }
    let batch_schema = batch.schema();
    for (i, field) in schema.fields().iter().enumerate() {
        let shard_field = batch_schema.field(i);
        if shard_field.name() != field.name() {
            return Err(ConvertError::Other(format!(
                "shard schema mismatch for '{name}': column {i} is '{}' in the dataframe \
                 schema but '{}' in the shard (columns must match by name and order)",
                field.name(),
                shard_field.name(),
            )));
        }
        if !logical_type_compatible(field.data_type(), shard_field.data_type()) {
            return Err(ConvertError::Other(format!(
                "shard schema mismatch for '{name}': column '{}' is {:?} in the dataframe \
                 schema but {:?} in the shard",
                field.name(),
                field.data_type(),
                shard_field.data_type(),
            )));
        }
    }
    Ok(())
}

/// True when a per-shard column type is interchangeable with the
/// declared (unified) schema type for the purpose of streaming export.
/// Exact equality always passes; additionally `Utf8`/`LargeUtf8` are
/// treated as equivalent. For a **categorical** column (schema type
/// `Dictionary(_, V)`), three further forms are accepted, generic over the
/// value class V (string AND numeric — §3.2):
///   * another `Dictionary(_, V')` shard whose value **class** matches V
///     (key width is irrelevant — base shards may be `Int8`-keyed, etc.);
///   * a **plain** shard whose type matches V's value class — this is the
///     append-grown layout (`append` decodes the dictionary to its value
///     type), folded back into the categorical by [`local_categorical_view`].
///
/// A plain shard whose value class differs from V (e.g. plain `Int64` under
/// a `Dictionary(_, Utf8)` column) still fails — the genuine-corruption case.
fn logical_type_compatible(schema_dt: &DataType, shard_dt: &DataType) -> bool {
    fn is_string_like(t: &DataType) -> bool {
        matches!(t, DataType::Utf8 | DataType::LargeUtf8)
    }
    fn dict_value_type(t: &DataType) -> Option<&DataType> {
        match t {
            DataType::Dictionary(_, v) => Some(v.as_ref()),
            _ => None,
        }
    }
    if schema_dt == shard_dt || (is_string_like(schema_dt) && is_string_like(shard_dt)) {
        return true;
    }
    match (dict_value_type(schema_dt), dict_value_type(shard_dt)) {
        // dict schema vs dict shard: value classes must match (any key width).
        (Some(sv), Some(dv)) => cat_value_class_eq(sv, dv),
        // dict schema vs plain shard: plain type must be the dict's value class.
        (Some(sv), None) => cat_value_class_eq(sv, shard_dt),
        _ => false,
    }
}

/// Per-column streaming writer state. Pre-allocated HDF5 datasets +
/// running offset (for hyperslab writes) + categorical accumulator
/// (for `Dictionary(_, Utf8)` columns).
enum ColumnStreamWriter {
    Int32 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Int64 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Float32 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Float64 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Utf8 {
        ds: hdf5::Dataset,
        offset: usize,
    },
    Boolean {
        // Group is created with `encoding-type` / `encoding-version`
        // attributes up front; kept here so its lifetime extends
        // through the streaming loop (the nested datasets borrow it).
        #[allow(dead_code)]
        group: hdf5::Group,
        values_ds: hdf5::Dataset,
        mask_ds: hdf5::Dataset,
        offset: usize,
    },
    Categorical {
        group: hdf5::Group,
        codes_ds: hdf5::Dataset,
        offset: usize,
        // Cross-shard global category vocabulary. The variant (string /
        // integer / float) is fixed at writer creation from the column's
        // dictionary value type; `append_shard_to_column` folds each shard's
        // **declared** categories in (declared order preserved), and
        // `finalize_column_writer` writes a `categories` dataset of the
        // matching HDF5 dtype.
        accum: CatAccum,
        // `Some` ⇔ a row filter is active, and then only these category
        // values enter the vocabulary. Computed by `scan_used_categories`
        // before any code was written, because a category dropped after the
        // fact would need every already-written code renumbered.
        used: Option<UsedCategories>,
        // pandas `ordered` bit, resolved from the source field metadata
        // ([`CATEGORICAL_ORDERED_KEY`]) at writer creation and emitted at
        // finalize (the `Field` is gone by then).
        ordered: bool,
    },
    /// anndata `nullable-integer` / `nullable-string-array` group: a
    /// `values` dataset (nulls filled with `0` / `""`) + a boolean `mask`
    /// (`mask[i] == true` ⇔ null). Allocated only when a pre-scan found
    /// the column actually contains nulls, so null-free columns stay
    /// plain datasets. `kind` selects the Arrow downcast / value dtype.
    Nullable {
        kind: NullableKind,
        // Held so the group outlives the borrowed `values_ds`/`mask_ds`.
        #[allow(dead_code)]
        group: hdf5::Group,
        values_ds: hdf5::Dataset,
        mask_ds: hdf5::Dataset,
        offset: usize,
    },
    Unsupported,
}

/// Value dtype of a [`ColumnStreamWriter::Nullable`] column.
#[derive(Clone, Copy)]
enum NullableKind {
    Int32,
    Int64,
    String,
}

/// Pre-allocate an anndata nullable group (`values` + `mask` datasets +
/// encoding attrs). Generic over the HDF5 value element type so it serves both
/// `nullable-integer` (i32/i64) and `nullable-string-array` (`VarLenUnicode`).
fn create_nullable_writer<T: hdf5::H5Type>(
    group: &hdf5::Group,
    on_disk_name: &str,
    n_rows_kept: usize,
    encoding_type: &str,
    kind: NullableKind,
) -> Result<ColumnStreamWriter, ConvertError> {
    let g = group.create_group(on_disk_name)?;
    let values_ds = g.new_dataset::<T>().shape([n_rows_kept]).create("values")?;
    let mask_ds = g
        .new_dataset::<bool>()
        .shape([n_rows_kept])
        .create("mask")?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu(encoding_type))?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    Ok(ColumnStreamWriter::Nullable {
        kind,
        group: g,
        values_ds,
        mask_ds,
        offset: 0,
    })
}

fn create_column_writer(
    group: &hdf5::Group,
    on_disk_name: &str,
    field: &Field,
    n_rows_kept: usize,
    want_nullable: bool,
    used: Option<UsedCategories>,
) -> Result<ColumnStreamWriter, ConvertError> {
    match field.data_type() {
        DataType::Int32 => {
            if want_nullable {
                return create_nullable_writer::<i32>(
                    group,
                    on_disk_name,
                    n_rows_kept,
                    "nullable-integer",
                    NullableKind::Int32,
                );
            }
            let ds = group
                .new_dataset::<i32>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Int32 { ds, offset: 0 })
        }
        DataType::Int64 => {
            if want_nullable {
                return create_nullable_writer::<i64>(
                    group,
                    on_disk_name,
                    n_rows_kept,
                    "nullable-integer",
                    NullableKind::Int64,
                );
            }
            let ds = group
                .new_dataset::<i64>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Int64 { ds, offset: 0 })
        }
        DataType::Float32 => {
            let ds = group
                .new_dataset::<f32>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Float32 { ds, offset: 0 })
        }
        DataType::Float64 => {
            let ds = group
                .new_dataset::<f64>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Float64 { ds, offset: 0 })
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            if want_nullable {
                return create_nullable_writer::<VarLenUnicode>(
                    group,
                    on_disk_name,
                    n_rows_kept,
                    "nullable-string-array",
                    NullableKind::String,
                );
            }
            let ds = group
                .new_dataset::<VarLenUnicode>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Utf8 { ds, offset: 0 })
        }
        DataType::Boolean => {
            // nullable-boolean v0.1.0: group with `values` + `mask`.
            let bool_group = group.create_group(on_disk_name)?;
            let values_ds = bool_group
                .new_dataset::<bool>()
                .shape([n_rows_kept])
                .create("values")?;
            let mask_ds = bool_group
                .new_dataset::<bool>()
                .shape([n_rows_kept])
                .create("mask")?;
            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("nullable-boolean"))?;
            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.1.0"))?;
            Ok(ColumnStreamWriter::Boolean {
                group: bool_group,
                values_ds,
                mask_ds,
                offset: 0,
            })
        }
        DataType::Dictionary(_, value_type) if is_supported_cat_value_type(value_type.as_ref()) => {
            let cat_group = group.create_group(on_disk_name)?;
            let codes_ds = cat_group
                .new_dataset::<i32>()
                .shape([n_rows_kept])
                .create("codes")?;
            Ok(ColumnStreamWriter::Categorical {
                group: cat_group,
                codes_ds,
                offset: 0,
                accum: CatAccum::new(value_type.as_ref()),
                used,
                ordered: field_ordered(field),
            })
        }
        _ => {
            // Mirrors `write_column_to_hdf5`'s warn-and-skip arm so the
            // streaming and eager paths behave identically on
            // unsupported types. The `UnsupportedExportColumn` warning
            // is emitted by the caller (which holds the `WarningSink`).
            Ok(ColumnStreamWriter::Unsupported)
        }
    }
}

/// `coerced_nulls` accumulates the nulls this column had to flatten to `0` /
/// `""` because it is written as a plain dataset. That only happens on the
/// pandas index (`_index`), which anndata requires to be a plain dataset and so
/// can never carry a nullable group: every other integer / string column with
/// nulls got a `Nullable` writer from the pre-scan. The caller turns a non-zero
/// total into one [`ConvertWarning::CoercedNulls`] per column.
fn append_shard_to_column(
    writer: &mut ColumnStreamWriter,
    array: &dyn Array,
    kept_local: &[usize],
    name: &str,
    coerced_nulls: &mut u64,
) -> Result<(), ConvertError> {
    if kept_local.is_empty() {
        // A shard every row of which was filtered out contributes no values —
        // and an empty hyperslab selection is not a write HDF5 accepts. Its
        // *declared* categories still belong in the vocabulary though, since a
        // category list is declared by the column, not implied by its rows. If
        // a row filter is active they are pruned by `used` like any other, so
        // this cannot resurrect a category no kept row references.
        if let ColumnStreamWriter::Categorical { accum, used, .. } = writer {
            let (_, local_values) = local_categorical_view(array, name)?;
            intern_declared_categories(accum, used.as_ref(), &local_values, name)?;
        }
        return Ok(());
    }
    match writer {
        ColumnStreamWriter::Int32 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            let values: Vec<i32> = kept_local
                .iter()
                .map(|&i| {
                    if arr.is_valid(i) {
                        arr.value(i)
                    } else {
                        *coerced_nulls += 1;
                        0
                    }
                })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Int64 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(name, "Int64"))?;
            let values: Vec<i64> = kept_local
                .iter()
                .map(|&i| {
                    if arr.is_valid(i) {
                        arr.value(i)
                    } else {
                        *coerced_nulls += 1;
                        0
                    }
                })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Float32 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(name, "Float32"))?;
            let values: Vec<f32> = kept_local
                .iter()
                .map(|&i| {
                    if arr.is_valid(i) {
                        arr.value(i)
                    } else {
                        f32::NAN
                    }
                })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Float64 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(name, "Float64"))?;
            let values: Vec<f64> = kept_local
                .iter()
                .map(|&i| {
                    if arr.is_valid(i) {
                        arr.value(i)
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Utf8 { ds, offset } => {
            let values: Vec<VarLenUnicode> = match array.data_type() {
                DataType::Utf8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| downcast_err(name, "Utf8"))?;
                    kept_local
                        .iter()
                        .map(|&i| {
                            if arr.is_valid(i) {
                                vlu(arr.value(i))
                            } else {
                                *coerced_nulls += 1;
                                vlu("")
                            }
                        })
                        .collect()
                }
                DataType::LargeUtf8 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .ok_or_else(|| downcast_err(name, "LargeUtf8"))?;
                    kept_local
                        .iter()
                        .map(|&i| {
                            if arr.is_valid(i) {
                                vlu(arr.value(i))
                            } else {
                                *coerced_nulls += 1;
                                vlu("")
                            }
                        })
                        .collect()
                }
                other => {
                    return Err(ConvertError::Other(format!(
                        "column '{name}': expected Utf8/LargeUtf8, got {other:?}"
                    )));
                }
            };
            ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Boolean {
            values_ds,
            mask_ds,
            offset,
            ..
        } => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(name, "Boolean"))?;
            let values: Vec<bool> = kept_local
                .iter()
                .map(|&i| arr.is_valid(i) && arr.value(i))
                .collect();
            let mask: Vec<bool> = kept_local.iter().map(|&i| !arr.is_valid(i)).collect();
            values_ds.write_slice(
                ArrayView1::from(values.as_slice()),
                ndarray::s![*offset..*offset + values.len()],
            )?;
            mask_ds.write_slice(
                ArrayView1::from(mask.as_slice()),
                ndarray::s![*offset..*offset + mask.len()],
            )?;
            *offset += values.len();
        }
        ColumnStreamWriter::Categorical {
            codes_ds,
            offset,
            accum,
            used,
            ..
        } => {
            // Accept both a `Dictionary(_, V)` shard (base) and a plain `V`
            // shard (appended): the helper normalizes both to local codes +
            // distinct values, generic over the value class (§3.3).
            let (local_codes, local_values) = local_categorical_view(array, name)?;
            let local_to_global =
                intern_declared_categories(accum, used.as_ref(), &local_values, name)?;

            let mut kept_codes: Vec<i32> = Vec::with_capacity(kept_local.len());
            for &i in kept_local {
                let lc = local_codes[i];
                if lc < 0 {
                    kept_codes.push(-1);
                    continue;
                }
                match local_to_global[lc as usize] {
                    Some(g) => kept_codes.push(g),
                    // Unreachable by construction: `used` is exactly the set of
                    // categories the *kept* rows reference, computed from the
                    // same mask over the same shards. Reaching it would mean
                    // the pre-scan and the write pass disagreed about which
                    // rows survive, which must fail loudly rather than write a
                    // code into a vocabulary that does not contain it.
                    None => {
                        return Err(ConvertError::Other(format!(
                            "column '{name}': kept row {i} references category {lc}, which the \
                             pre-scan recorded as referenced by no kept row (the keep mask used \
                             for the vocabulary scan disagrees with the one used for the write)"
                        )))
                    }
                }
            }
            codes_ds.write_slice(
                ArrayView1::from(kept_codes.as_slice()),
                ndarray::s![*offset..*offset + kept_codes.len()],
            )?;
            *offset += kept_codes.len();
        }
        ColumnStreamWriter::Nullable {
            kind,
            values_ds,
            mask_ds,
            offset,
            ..
        } => {
            let mask: Vec<bool> = kept_local.iter().map(|&i| !array.is_valid(i)).collect();
            match kind {
                NullableKind::Int32 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .ok_or_else(|| downcast_err(name, "Int32"))?;
                    let values: Vec<i32> = kept_local
                        .iter()
                        .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                        .collect();
                    values_ds.write_slice(
                        ArrayView1::from(values.as_slice()),
                        ndarray::s![*offset..*offset + values.len()],
                    )?;
                }
                NullableKind::Int64 => {
                    let arr = array
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| downcast_err(name, "Int64"))?;
                    let values: Vec<i64> = kept_local
                        .iter()
                        .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                        .collect();
                    values_ds.write_slice(
                        ArrayView1::from(values.as_slice()),
                        ndarray::s![*offset..*offset + values.len()],
                    )?;
                }
                NullableKind::String => {
                    // Per-shard arrays may be Utf8 or LargeUtf8 (mirrors the
                    // plain `Utf8` writer arm).
                    let values: Vec<VarLenUnicode> = match array.data_type() {
                        DataType::Utf8 => {
                            let arr = array
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| downcast_err(name, "Utf8"))?;
                            kept_local
                                .iter()
                                .map(|&i| {
                                    if arr.is_valid(i) {
                                        vlu(arr.value(i))
                                    } else {
                                        vlu("")
                                    }
                                })
                                .collect()
                        }
                        DataType::LargeUtf8 => {
                            let arr = array
                                .as_any()
                                .downcast_ref::<LargeStringArray>()
                                .ok_or_else(|| downcast_err(name, "LargeUtf8"))?;
                            kept_local
                                .iter()
                                .map(|&i| {
                                    if arr.is_valid(i) {
                                        vlu(arr.value(i))
                                    } else {
                                        vlu("")
                                    }
                                })
                                .collect()
                        }
                        other => {
                            return Err(ConvertError::Other(format!(
                                "column '{name}': expected Utf8/LargeUtf8, got {other:?}"
                            )));
                        }
                    };
                    values_ds.write_slice(
                        ArrayView1::from(values.as_slice()),
                        ndarray::s![*offset..*offset + values.len()],
                    )?;
                }
            }
            mask_ds.write_slice(
                ArrayView1::from(mask.as_slice()),
                ndarray::s![*offset..*offset + mask.len()],
            )?;
            *offset += mask.len();
        }
        ColumnStreamWriter::Unsupported => {
            // Warning emitted once at writer creation; per-shard
            // append is a no-op (mirrors the eager path's skip).
        }
    }
    Ok(())
}

/// Fold one shard's **declared** category vocabulary into the cross-shard
/// accumulator, in declared order, returning `local_to_global[c]` — the global
/// code for local category `c`, or `None` when that category is pruned.
///
/// Declared order, not appearance order, is the whole point (§11.1): a
/// `pd.Categorical`'s category list is chosen by the user, is what `ordered=True`
/// makes every comparison mean, and may name levels no row uses. Interning it
/// wholesale is what preserves both properties. For a shard that arrives as a
/// plain `V` array rather than a `Dictionary` there is no declared order —
/// `local_categorical_view` supplies first-appearance-within-the-shard, which
/// is all the shard carries.
///
/// A category is skipped only when `used` says no kept row references it, so an
/// unfiltered export prunes nothing. Lookups borrow (`&str`, `i64`, `u64`) and
/// allocate only on insert, so a re-declared vocabulary costs a hash per
/// category per shard and no allocation.
///
/// The two key projections are free functions rather than closures because a
/// closure returning a reference borrowed from its argument does not infer the
/// `for<'a>` bound the `HashMap::get` / `HashSet::contains` calls need.
fn intern_declared_categories(
    accum: &mut CatAccum,
    used: Option<&UsedCategories>,
    local_values: &CatValues,
    name: &str,
) -> Result<Vec<Option<i32>>, ConvertError> {
    /// Shared body: walk the declared values in order, skip the pruned ones,
    /// and assign each survivor its global code (existing or freshly issued).
    macro_rules! fold {
        ($vals:expr, $dict:expr, $order:expr, $used:expr, $key:expr, $store:expr) => {{
            let vals = $vals;
            let mut out: Vec<Option<i32>> = Vec::with_capacity(vals.len());
            for v in vals.iter() {
                let k = $key(v);
                if let Some(u) = $used {
                    if !u.contains(k) {
                        out.push(None);
                        continue;
                    }
                }
                let g = match $dict.get(k) {
                    Some(&g) => g,
                    None => {
                        // Defensive: real categoricals stay far below i32::MAX,
                        // but the on-disk `codes` dataset is i32.
                        let g: i32 = $order.len().try_into().map_err(|_| cardinality_err(name))?;
                        $dict.insert($store(v), g);
                        $order.push($store(v));
                        g
                    }
                };
                out.push(Some(g));
            }
            out
        }};
    }

    // The accumulator variant is fixed from the column's declared value type
    // at writer creation; every shard must present the same class, and so must
    // the pre-scan's `used` set.
    Ok(match (accum, local_values) {
        (CatAccum::Str { dict, order }, CatValues::Str(vals)) => {
            let used = match used {
                None => None,
                Some(UsedCategories::Str(s)) => Some(s),
                Some(_) => return Err(cat_class_mismatch_err(name)),
            };
            fn key(v: &VarLenUnicode) -> &str {
                v.as_str()
            }
            fn store(v: &VarLenUnicode) -> String {
                v.as_str().to_string()
            }
            fold!(vals, dict, order, used, key, store)
        }
        (CatAccum::Int { dict, order }, CatValues::Int(vals)) => {
            let used = match used {
                None => None,
                Some(UsedCategories::Int(s)) => Some(s),
                Some(_) => return Err(cat_class_mismatch_err(name)),
            };
            fn key(v: &i64) -> &i64 {
                v
            }
            fn store(v: &i64) -> i64 {
                *v
            }
            fold!(vals, dict, order, used, key, store)
        }
        (CatAccum::Float { dict, order }, CatValues::Float(vals)) => {
            let used = match used {
                None => None,
                Some(UsedCategories::Float(s)) => Some(s),
                Some(_) => return Err(cat_class_mismatch_err(name)),
            };
            // Floats key on the bit pattern in both the accumulator and the
            // used-set, so the borrowed key is a `u64` held in a local.
            let mut out: Vec<Option<i32>> = Vec::with_capacity(vals.len());
            for v in vals.iter() {
                let k = v.to_bits();
                if let Some(u) = used {
                    if !u.contains(&k) {
                        out.push(None);
                        continue;
                    }
                }
                let g = match dict.get(&k) {
                    Some(&g) => g,
                    None => {
                        let g: i32 = order.len().try_into().map_err(|_| cardinality_err(name))?;
                        dict.insert(k, g);
                        order.push(*v);
                        g
                    }
                };
                out.push(Some(g));
            }
            out
        }
        _ => return Err(cat_class_mismatch_err(name)),
    })
}

fn finalize_column_writer(writer: &ColumnStreamWriter) -> Result<(), ConvertError> {
    if let ColumnStreamWriter::Categorical {
        group,
        accum,
        ordered,
        ..
    } = writer
    {
        // Emit the `categories` dataset with the dtype matching the source
        // category class so numeric-keyed categoricals round-trip (anndata
        // reconstructs the numeric `pd.Categorical`); strings stay
        // `VarLenUnicode`.
        match accum {
            CatAccum::Str { order, .. } => {
                let cats: Vec<VarLenUnicode> = order.iter().map(|s| vlu(s)).collect();
                group
                    .new_dataset::<VarLenUnicode>()
                    .shape([cats.len()])
                    .create("categories")?
                    .write(&cats)?;
            }
            CatAccum::Int { order, .. } => {
                group
                    .new_dataset::<i64>()
                    .shape([order.len()])
                    .create("categories")?
                    .write(order.as_slice())?;
            }
            CatAccum::Float { order, .. } => {
                group
                    .new_dataset::<f64>()
                    .shape([order.len()])
                    .create("categories")?
                    .write(order.as_slice())?;
            }
        }
        group
            .new_attr::<VarLenUnicode>()
            .create("encoding-type")?
            .write_scalar(&vlu("categorical"))?;
        group
            .new_attr::<VarLenUnicode>()
            .create("encoding-version")?
            .write_scalar(&vlu("0.2.0"))?;
        group
            .new_attr::<bool>()
            .create("ordered")?
            .write_scalar(ordered)?;
    }
    Ok(())
}
