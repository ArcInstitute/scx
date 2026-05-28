// Write scx file back to h5ad format

use std::collections::HashMap;
use std::path::Path;

use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow::datatypes::{
    DataType, Field, Int16Type, Int32Type, Int64Type, Int8Type, Schema, UInt16Type, UInt32Type,
    UInt64Type, UInt8Type,
};
use hdf5::types::VarLenUnicode;
use ndarray::ArrayView1;

/// Convert &str to VarLenUnicode, validating no NUL bytes are present.
fn vlu(s: &str) -> VarLenUnicode {
    s.parse::<VarLenUnicode>().unwrap_or_else(|_| {
        // Strip NUL bytes rather than panicking (finding 8.2).
        let cleaned: String = s.chars().filter(|&c| c != '\0').collect();
        cleaned
            .parse::<VarLenUnicode>()
            .expect("cleaned string should have no NUL bytes")
    })
}

use scx_format::reader::ScxReader;

use super::pipeline::ConvertError;

/// Write an SCX file to h5ad format.
pub fn write_scx_to_h5ad(scx_path: &Path, h5ad_path: &Path) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let file = hdf5::File::create(h5ad_path)?;

    // Honor deletion vectors on every leg (X, obs, layers). The CSR
    // and layer readers filter by DV directly; obs goes through the
    // shared streaming-or-eager dispatcher which applies the same
    // global keep mask. Pre-fix, this path silently dropped DV
    // semantics — both /X and obs were written unfiltered.
    let keep_mask = crate::h5ad_stream_write::build_keep_mask(&reader)?;

    // Read the full CSR matrix (DV-filtered when active)
    let csr = reader.read_all_csr_shards_filtered()?;
    let n_obs = csr.shape.0;
    let n_vars = csr.shape.1;

    // Write X as CSR group
    write_sparse_group(
        &file,
        "X",
        &csr.indptr,
        &csr.indices,
        &csr.data,
        n_obs,
        n_vars,
    )?;

    // Write obs / var via the shared dispatcher so legacy + sharded
    // sources both flow through one code path and the DV keep mask
    // is honored.
    let root = file.as_group()?;
    crate::h5ad_stream_write::write_obs_streaming_or_eager(&root, &reader, keep_mask.as_deref())?;
    crate::h5ad_stream_write::write_var_streaming_or_eager(&root, &reader)?;

    // Write obsm (DV-filtered when active — obs-axis rows must match
    // /X and /obs).
    if let Ok(obsm_map) = reader.read_all_obsm() {
        if !obsm_map.is_empty() {
            let obsm_group = file.create_group("obsm")?;
            for (name, batch) in &obsm_map {
                let filtered = match keep_mask.as_deref() {
                    Some(mask) => {
                        crate::h5ad_stream_write::filter_record_batch_by_mask(batch, mask)?
                    }
                    None => batch.clone(),
                };
                write_obsm_entry(&obsm_group, name, &filtered)?;
            }
        }
    }

    // Write uns
    if let Ok(uns) = reader.read_uns() {
        let uns_group = file.create_group("uns")?;
        write_uns_entries(&uns_group, &uns)?;
    }

    // Write layers (DV-filtered when active — layers share X's
    // row count by AnnData invariant)
    let layer_names = reader.layer_names();
    if !layer_names.is_empty() {
        let layers_group = file.create_group("layers")?;
        for layer_name in &layer_names {
            if let Ok(layer_csr) = reader.read_layer_filtered(layer_name) {
                let lg = layers_group.create_group(layer_name)?;
                write_sparse_arrays(
                    &lg,
                    &layer_csr.indptr,
                    &layer_csr.indices,
                    &layer_csr.data,
                    layer_csr.shape.0,
                    layer_csr.shape.1,
                )?;
            }
        }
    }

    Ok(())
}

fn write_sparse_group(
    file: &hdf5::File,
    name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    let group = file.create_group(name)?;
    write_sparse_arrays(&group, indptr, indices, data, n_obs, n_vars)
}

// Module-internal helpers for the h5mu writer (Phase D.2). All
// take a parent `hdf5::Group` instead of the root `hdf5::File` so
// per-modality blocks under `/mod/{name}/…` can reuse the same
// emitters as `/X`, `/obs`, `/var`, `/obsm/…`, `/uns/…`.

pub(super) fn write_sparse_group_at(
    parent: &hdf5::Group,
    name: &str,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    let group = parent.create_group(name)?;
    write_sparse_arrays(&group, indptr, indices, data, n_obs, n_vars)
}

pub(super) fn write_dataframe_group_at(
    parent: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
) -> Result<(), ConvertError> {
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;
    write_dataframe_body(&group, batch)
}

/// Shared body for `write_dataframe_group_at`. Caller is responsible
/// for opening or creating `group`. Resolves the pandas index from
/// schema metadata, renames pyarrow's `__index_level_0__` to anndata's
/// `_index` literal on disk (named indexes keep their original name),
/// excludes the index column from `column-order`, and always writes
/// `column-order` (length-0 OK) since anndata.read_h5ad requires the
/// attribute to be present.
fn write_dataframe_body(
    group: &hdf5::Group,
    batch: &arrow::array::RecordBatch,
) -> Result<(), ConvertError> {
    let schema = batch.schema();
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("dataframe"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;

    // Resolve the schema field that holds the pandas index. Probe order:
    //   1. The `pandas` schema metadata's `index_columns` (the
    //      authoritative source — `pyarrow.Table.from_pandas` stamps
    //      this; covers both named and unnamed indexes).
    //   2. Fallback: `schema.field(0)` — preserves the CLI path's
    //      behaviour where obs/var come from disk without pandas
    //      metadata, and the first field already IS the index dataset.
    let pandas_idx_cols = scx_format::pandas_index_columns(schema.as_ref());
    let index_field_name: Option<String> = pandas_idx_cols
        .into_iter()
        .find(|n| schema.field_with_name(n).is_ok())
        .or_else(|| schema.fields().first().map(|f| f.name().clone()));

    // Rename pyarrow's canonical `__index_level_0__` (unnamed pandas
    // index) to anndata's `_index` literal on disk. Named indexes keep
    // their original name.
    let on_disk_index: &str = match index_field_name.as_deref() {
        Some("__index_level_0__") => "_index",
        Some(n) => n,
        None => "_index",
    };

    if !schema.fields().is_empty() {
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&vlu(on_disk_index))?;
    }

    let mut col_order: Vec<VarLenUnicode> =
        Vec::with_capacity(batch.num_columns().saturating_sub(1));
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(col_idx);
        if Some(field.name()) == index_field_name.as_ref() {
            // Index column → write under the anndata on-disk name and
            // exclude from `column-order` (matches anndata convention).
            write_column_to_hdf5(group, on_disk_index, col, field.data_type())?;
        } else if write_column_to_hdf5(group, field.name(), col, field.data_type())? {
            // Only list the column in `column-order` when a dataset
            // was actually created — unsupported types are
            // warn-and-skipped and must not appear in the index.
            col_order.push(vlu(field.name()));
        }
    }

    // anndata.read_h5ad requires `column-order` to be present on every
    // dataframe group, even when the dataframe has no non-index columns
    // (it will raise `KeyError: "...can't locate attribute:
    // 'column-order'"` otherwise). Write it unconditionally — a
    // length-0 array for the no-columns case matches anndata's own
    // emission. SCX's own reader handles the empty-attr case in
    // `read_dataframe_group`'s fallback branch.
    group
        .new_attr::<VarLenUnicode>()
        .shape(col_order.len())
        .create("column-order")?
        .write_raw(&col_order)?;

    Ok(())
}

pub(super) fn write_obsm_entry_at(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
) -> Result<(), ConvertError> {
    write_obsm_entry(obsm_group, name, batch)
}

pub(super) fn write_uns_entries_at(
    group: &hdf5::Group,
    value: &serde_json::Value,
) -> Result<(), ConvertError> {
    write_uns_entries(group, value)
}

fn write_sparse_arrays(
    group: &hdf5::Group,
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<(), ConvertError> {
    // Write arrays
    group
        .new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")?
        .write(indptr)?;
    group
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")?
        .write(indices)?;
    group
        .new_dataset::<f32>()
        .shape([data.len()])
        .create("data")?
        .write(data)?;

    // Set attributes
    let encoding_type = vlu("csr_matrix");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&encoding_type)?;

    let encoding_version = vlu("0.1.0");
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&encoding_version)?;

    let shape = [n_obs as i64, n_vars as i64];
    group
        .new_attr::<i64>()
        .shape([2])
        .create("shape")?
        .write(&shape)?;

    Ok(())
}

fn downcast_err(name: &str, expected: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': expected {expected} array but downcast failed"
    ))
}

/// Extract codes (promoted to i32, -1 for null) and string
/// categories from a categorical column. Handles all integer key
/// widths Arrow / pandas uses (Int8/16/32/64, UInt8/16/32/64). The
/// SCX → h5ad writer emits i32 codes uniformly so anndata's reader
/// doesn't need to dispatch on key width.
fn dict_codes_and_categories_i32(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<i32>, Vec<VarLenUnicode>), ConvertError> {
    macro_rules! extract {
        ($t:ty, $label:literal) => {{
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<$t>>()
                .ok_or_else(|| downcast_err(name, $label))?;
            let codes: Vec<i32> = dict
                .keys()
                .iter()
                .map(|v| match v {
                    Some(k) => k as i32,
                    None => -1,
                })
                .collect();
            let values_arr = dict.values().as_string::<i32>();
            let cats: Vec<VarLenUnicode> = (0..values_arr.len())
                .map(|i| vlu(values_arr.value(i)))
                .collect();
            Ok::<_, ConvertError>((codes, cats))
        }};
    }
    let key_type = match array.data_type() {
        DataType::Dictionary(k, _) => k.as_ref(),
        _ => {
            return Err(ConvertError::Other(format!(
                "column '{name}': expected Dictionary, got {:?}",
                array.data_type()
            )))
        }
    };
    match key_type {
        DataType::Int8 => extract!(Int8Type, "Dictionary<Int8, Utf8>"),
        DataType::Int16 => extract!(Int16Type, "Dictionary<Int16, Utf8>"),
        DataType::Int32 => extract!(Int32Type, "Dictionary<Int32, Utf8>"),
        DataType::Int64 => extract!(Int64Type, "Dictionary<Int64, Utf8>"),
        DataType::UInt8 => extract!(UInt8Type, "Dictionary<UInt8, Utf8>"),
        DataType::UInt16 => extract!(UInt16Type, "Dictionary<UInt16, Utf8>"),
        DataType::UInt32 => extract!(UInt32Type, "Dictionary<UInt32, Utf8>"),
        DataType::UInt64 => extract!(UInt64Type, "Dictionary<UInt64, Utf8>"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': unsupported categorical key type {other:?}"
        ))),
    }
}

/// Write one Arrow column into `group/name`. Returns `Ok(true)` on a
/// supported type (dataset created), `Ok(false)` when the type is
/// not yet supported and the column was warn-and-skipped. The caller
/// uses the boolean to decide whether to add `name` to `column-order`
/// — adding a name without a backing dataset breaks
/// `anndata.read_h5ad`'s lookup.
fn write_column_to_hdf5(
    group: &hdf5::Group,
    name: &str,
    array: &dyn Array,
    dtype: &DataType,
) -> Result<bool, ConvertError> {
    match dtype {
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            let values: Vec<i32> = arr.iter().map(|v| v.unwrap_or(0)).collect();
            group
                .new_dataset::<i32>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(name, "Int64"))?;
            let values: Vec<i64> = arr.iter().map(|v| v.unwrap_or(0)).collect();
            group
                .new_dataset::<i64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(name, "Float32"))?;
            let values: Vec<f32> = arr.iter().map(|v| v.unwrap_or(0.0)).collect();
            group
                .new_dataset::<f32>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Float64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(name, "Float64"))?;
            let values: Vec<f64> = arr.iter().map(|v| v.unwrap_or(0.0)).collect();
            group
                .new_dataset::<f64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .ok_or_else(|| downcast_err(name, "Utf8"))?;
            let values: Vec<VarLenUnicode> = arr.iter().map(|v| vlu(v.unwrap_or(""))).collect();
            group
                .new_dataset::<VarLenUnicode>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Boolean => {
            let arr = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(name, "Boolean"))?;

            // anndata's only registered IOSpec for h5py boolean
            // columns is `nullable-boolean` v0.1.0 — a *group* with
            // `values` and `mask` datasets. The legacy
            // flat-u8-with-encoding-type-boolean shape isn't
            // registered at all, so `anndata.read_h5ad` raises on
            // it. The group form additionally preserves Arrow's
            // per-element validity bits in the mask
            // (`mask[i] == 1` ⇔ row is null).
            let bool_group = group.create_group(name)?;

            // anndata + pandas's BooleanArray reader is strict: the
            // values dataset must have native HDF5 boolean dtype,
            // not u8. Plain u8 trips
            // `TypeError: values should be boolean numpy array`.
            let values: Vec<bool> = (0..arr.len())
                .map(|i| arr.is_valid(i) && arr.value(i))
                .collect();
            let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();

            bool_group
                .new_dataset::<bool>()
                .shape([values.len()])
                .create("values")?
                .write(&values)?;
            bool_group
                .new_dataset::<bool>()
                .shape([mask.len()])
                .create("mask")?
                .write(&mask)?;

            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("nullable-boolean"))?;
            bool_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.1.0"))?;
        }
        DataType::Dictionary(_key_type, value_type) if **value_type == DataType::Utf8 => {
            // Modern anndata categorical (encoding-version 0.2.0):
            // write as a group with `codes` + `categories` as
            // separate datasets, NOT as a `categories` attribute on
            // the codes dataset. The legacy attribute form overflows
            // HDF5's ~64 KB object-header limit on high-cardinality
            // categoricals (census-scale `cell_type` / `donor_id`):
            // `H5Acreate2(): object header message is too large`.
            //
            // Key types: pandas/Arrow picks the narrowest integer
            // type that fits the cardinality (Int8 for <128
            // categories, Int16 for <32K, Int32 above). We promote
            // every input to i32 on disk so the SCX → h5ad output
            // is uniform; anndata reads any width on the round-trip.
            let (codes, cats) = dict_codes_and_categories_i32(array, name)?;

            let cat_group = group.create_group(name)?;
            cat_group
                .new_dataset::<i32>()
                .shape([codes.len()])
                .create("codes")?
                .write(&codes)?;
            cat_group
                .new_dataset::<VarLenUnicode>()
                .shape([cats.len()])
                .create("categories")?
                .write(&cats)?;

            cat_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("categorical"))?;
            cat_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.2.0"))?;
            // `ordered=false` matches scipy / pandas default. Arrow's
            // DictionaryArray doesn't carry an `ordered` bit so we
            // never have richer information to forward. Native HDF5
            // bool — anndata's categorical reader expects
            // `H5T_NATIVE_HBOOL_8`, not u8.
            cat_group
                .new_attr::<bool>()
                .create("ordered")?
                .write_scalar(&false)?;
        }
        _ => {
            eprintln!("warning: skipping column '{name}' with unsupported type {dtype:?}");
            return Ok(false);
        }
    }
    Ok(true)
}

fn write_obsm_entry(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &RecordBatch,
) -> Result<(), ConvertError> {
    let n_rows = batch.num_rows();
    let n_cols = batch.num_columns();

    // Flatten to 2D f32 array (finding 8.11: handle non-Float32 columns).
    let mut flat = vec![0.0f32; n_rows * n_cols];
    for col_idx in 0..n_cols {
        let col = batch.column(col_idx);
        if let Some(arr) = col.as_any().downcast_ref::<Float32Array>() {
            for row_idx in 0..n_rows {
                flat[row_idx * n_cols + col_idx] = arr.value(row_idx);
            }
        } else if let Some(arr) = col.as_any().downcast_ref::<Float64Array>() {
            for row_idx in 0..n_rows {
                flat[row_idx * n_cols + col_idx] = arr.value(row_idx) as f32;
            }
        } else {
            return Err(ConvertError::Other(format!(
                "obsm '{name}' column {col_idx}: expected Float32 or Float64 array, got {:?}",
                col.data_type()
            )));
        }
    }

    let ds = obsm_group
        .new_dataset::<f32>()
        .shape([n_rows, n_cols])
        .create(name)?;

    // Write using ndarray
    let nd_array = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ConvertError::Other(format!("ndarray shape error: {e}")))?;
    ds.write(&nd_array)?;

    Ok(())
}

/// Streaming counterpart of [`write_dataframe_group_at`]. Pre-allocates
/// one HDF5 dataset per column at fixed size `n_rows_kept`, then drains
/// `shards` and hyperslab-writes the kept-row slice of each column.
///
/// Mirrors the pre-allocate-then-hyperslab pattern in
/// `h5ad_stream_write::create_csr_triplet` + `stream_csr_to_group_at`
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
pub(super) fn write_dataframe_group_streaming<I>(
    parent: &hdf5::Group,
    name: &str,
    schema: &Schema,
    shards: I,
    n_rows_kept: usize,
    keep_mask_opt: Option<&[bool]>,
) -> Result<(), ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format::error::ScxError>>,
{
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;

    // Dataframe-level encoding attrs. Same shape as `write_dataframe_body`
    // — anndata.read_h5ad requires these even on empty obs.
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu("dataframe"))?;
    group
        .new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.2.0"))?;

    // Resolve the pandas index field (same probe order as
    // `write_dataframe_body`: pandas metadata first, else field(0)).
    let pandas_idx_cols = scx_format::pandas_index_columns(schema);
    let index_field_name: Option<String> = pandas_idx_cols
        .into_iter()
        .find(|n| schema.field_with_name(n).is_ok())
        .or_else(|| schema.fields().first().map(|f| f.name().clone()));

    let on_disk_index: &str = match index_field_name.as_deref() {
        Some("__index_level_0__") => "_index",
        Some(n) => n,
        None => "_index",
    };

    if !schema.fields().is_empty() {
        group
            .new_attr::<VarLenUnicode>()
            .create("_index")?
            .write_scalar(&vlu(on_disk_index))?;
    }

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
            on_disk_index
        } else {
            field.name()
        };
        let writer = create_column_writer(&group, on_disk_name, field, n_rows_kept)?;
        if !is_index && !matches!(writer, ColumnStreamWriter::Unsupported) {
            col_order.push(vlu(field.name()));
        }
        col_writers.push((col_idx, writer));
    }

    // anndata requires `column-order` even when empty (matches
    // `write_dataframe_body`).
    group
        .new_attr::<VarLenUnicode>()
        .shape(col_order.len())
        .create("column-order")?
        .write_raw(&col_order)?;

    // Drain shards. Each shard's stamped `row_start` schema metadata
    // (set by the writer via `stamp_dense_shard_metadata`) gives its
    // global row offset; the cumulative shard row count is verified
    // against it for defense-in-depth against producers that might
    // emit shards out of order.
    let mut cumulative_rows: usize = 0;
    for batch_result in shards {
        let batch = batch_result?;
        let n_shard_rows = batch.num_rows();

        if batch.num_columns() != schema.fields().len() {
            return Err(ConvertError::Other(format!(
                "shard schema mismatch for '{name}': schema has {} fields, batch has {}",
                schema.fields().len(),
                batch.num_columns()
            )));
        }

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

        if !kept_local.is_empty() {
            for (col_idx, writer) in col_writers.iter_mut() {
                let array = batch.column(*col_idx);
                let field_name = schema.fields()[*col_idx].name();
                append_shard_to_column(writer, array, &kept_local, field_name)?;
            }
        }

        cumulative_rows += n_shard_rows;
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
fn parse_shard_row_start(batch: &RecordBatch) -> Option<usize> {
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
        | ColumnStreamWriter::Categorical { offset, .. } => Some(*offset),
        ColumnStreamWriter::Unsupported => None,
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
        // `dict` keys come from the per-shard dictionary values; the
        // insertion order is preserved by walking `cat_order` at finalize.
        dict: HashMap<String, i32>,
        cat_order: Vec<String>,
    },
    Unsupported,
}

fn create_column_writer(
    group: &hdf5::Group,
    on_disk_name: &str,
    field: &Field,
    n_rows_kept: usize,
) -> Result<ColumnStreamWriter, ConvertError> {
    match field.data_type() {
        DataType::Int32 => {
            let ds = group
                .new_dataset::<i32>()
                .shape([n_rows_kept])
                .create(on_disk_name)?;
            Ok(ColumnStreamWriter::Int32 { ds, offset: 0 })
        }
        DataType::Int64 => {
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
        DataType::Dictionary(_, value_type)
            if matches!(value_type.as_ref(), DataType::Utf8 | DataType::LargeUtf8) =>
        {
            let cat_group = group.create_group(on_disk_name)?;
            let codes_ds = cat_group
                .new_dataset::<i32>()
                .shape([n_rows_kept])
                .create("codes")?;
            Ok(ColumnStreamWriter::Categorical {
                group: cat_group,
                codes_ds,
                offset: 0,
                dict: HashMap::new(),
                cat_order: Vec::new(),
            })
        }
        other => {
            // Mirrors `write_column_to_hdf5`'s warn-and-skip arm so the
            // streaming and eager paths behave identically on
            // unsupported types.
            eprintln!(
                "warning: skipping column '{}' with unsupported type {other:?}",
                field.name()
            );
            Ok(ColumnStreamWriter::Unsupported)
        }
    }
}

fn append_shard_to_column(
    writer: &mut ColumnStreamWriter,
    array: &dyn Array,
    kept_local: &[usize],
    name: &str,
) -> Result<(), ConvertError> {
    match writer {
        ColumnStreamWriter::Int32 { ds, offset } => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            let values: Vec<i32> = kept_local
                .iter()
                .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
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
                .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0 })
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
                .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0.0 })
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
                .map(|&i| if arr.is_valid(i) { arr.value(i) } else { 0.0 })
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
            dict,
            cat_order,
            ..
        } => {
            let (local_codes, local_values) = dict_local_codes_and_string_values(array, name)?;
            // Build local→global remap (extends `dict` / `cat_order`
            // for any value not seen before).
            let mut remap: Vec<i32> = Vec::with_capacity(local_values.len());
            for v in local_values {
                let g = match dict.get(&v) {
                    Some(&g) => g,
                    None => {
                        // Cap at i32::MAX. Practical categorical
                        // cardinalities (cell_type, donor_id) stay
                        // well below this; saturating is defensive.
                        let g: i32 = cat_order.len().try_into().map_err(|_| {
                            ConvertError::Other(format!(
                                "column '{name}': categorical cardinality exceeds i32::MAX"
                            ))
                        })?;
                        dict.insert(v.clone(), g);
                        cat_order.push(v);
                        g
                    }
                };
                remap.push(g);
            }
            let kept_codes: Vec<i32> = kept_local
                .iter()
                .map(|&i| {
                    let lc = local_codes[i];
                    if lc < 0 {
                        -1
                    } else {
                        remap[lc as usize]
                    }
                })
                .collect();
            codes_ds.write_slice(
                ArrayView1::from(kept_codes.as_slice()),
                ndarray::s![*offset..*offset + kept_codes.len()],
            )?;
            *offset += kept_codes.len();
        }
        ColumnStreamWriter::Unsupported => {
            // Warning emitted once at writer creation; per-shard
            // append is a no-op (mirrors the eager path's skip).
        }
    }
    Ok(())
}

fn finalize_column_writer(writer: &ColumnStreamWriter) -> Result<(), ConvertError> {
    if let ColumnStreamWriter::Categorical {
        group, cat_order, ..
    } = writer
    {
        let cats: Vec<VarLenUnicode> = cat_order.iter().map(|s| vlu(s)).collect();
        group
            .new_dataset::<VarLenUnicode>()
            .shape([cats.len()])
            .create("categories")?
            .write(&cats)?;
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
            .write_scalar(&false)?;
    }
    Ok(())
}

/// Streaming counterpart of [`dict_codes_and_categories_i32`]: extracts
/// local codes (promoted to i32, -1 for null) and the local dictionary
/// values as owned `Vec<String>` so callers can fold values into a
/// running global dictionary across shards. Accepts both
/// `Dictionary(_, Utf8)` and `Dictionary(_, LargeUtf8)` value types so
/// shard runtime arrays match whatever the per-shard downcast left.
fn dict_local_codes_and_string_values(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<i32>, Vec<String>), ConvertError> {
    macro_rules! extract {
        ($t:ty, $label:literal) => {{
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<$t>>()
                .ok_or_else(|| downcast_err(name, $label))?;
            let codes: Vec<i32> = dict
                .keys()
                .iter()
                .map(|v| match v {
                    Some(k) => k as i32,
                    None => -1,
                })
                .collect();
            let values: Vec<String> = match dict.values().data_type() {
                DataType::Utf8 => {
                    let arr = dict.values().as_string::<i32>();
                    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
                }
                DataType::LargeUtf8 => {
                    let arr = dict.values().as_string::<i64>();
                    (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
                }
                other => {
                    return Err(ConvertError::Other(format!(
                        "column '{name}': unsupported dictionary value type {other:?}"
                    )));
                }
            };
            Ok::<_, ConvertError>((codes, values))
        }};
    }
    let key_type = match array.data_type() {
        DataType::Dictionary(k, _) => k.as_ref(),
        _ => {
            return Err(ConvertError::Other(format!(
                "column '{name}': expected Dictionary, got {:?}",
                array.data_type()
            )));
        }
    };
    match key_type {
        DataType::Int8 => extract!(Int8Type, "Dictionary<Int8, _>"),
        DataType::Int16 => extract!(Int16Type, "Dictionary<Int16, _>"),
        DataType::Int32 => extract!(Int32Type, "Dictionary<Int32, _>"),
        DataType::Int64 => extract!(Int64Type, "Dictionary<Int64, _>"),
        DataType::UInt8 => extract!(UInt8Type, "Dictionary<UInt8, _>"),
        DataType::UInt16 => extract!(UInt16Type, "Dictionary<UInt16, _>"),
        DataType::UInt32 => extract!(UInt32Type, "Dictionary<UInt32, _>"),
        DataType::UInt64 => extract!(UInt64Type, "Dictionary<UInt64, _>"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': unsupported categorical key type {other:?}"
        ))),
    }
}

fn write_uns_entries(group: &hdf5::Group, value: &serde_json::Value) -> Result<(), ConvertError> {
    if let serde_json::Value::Object(map) = value {
        for (key, val) in map {
            write_uns_value(group, key, val)?;
        }
    }
    Ok(())
}

fn write_uns_value(
    group: &hdf5::Group,
    name: &str,
    value: &serde_json::Value,
) -> Result<(), ConvertError> {
    match value {
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                group
                    .new_dataset::<i64>()
                    .shape(())
                    .create(name)?
                    .write_scalar(&i)?;
            } else if let Some(f) = n.as_f64() {
                group
                    .new_dataset::<f64>()
                    .shape(())
                    .create(name)?
                    .write_scalar(&f)?;
            }
        }
        serde_json::Value::String(s) => {
            let v = vlu(s);
            group
                .new_dataset::<VarLenUnicode>()
                .shape(())
                .create(name)?
                .write_scalar(&v)?;
        }
        serde_json::Value::Bool(b) => {
            group
                .new_dataset::<bool>()
                .shape(())
                .create(name)?
                .write_scalar(b)?;
        }
        serde_json::Value::Array(arr) => {
            // Try as array of numbers
            if arr.iter().all(|v| v.is_i64()) {
                let data: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                group
                    .new_dataset::<i64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if arr.iter().all(|v| v.is_number()) {
                let data: Vec<f64> = arr.iter().filter_map(|v| v.as_f64()).collect();
                group
                    .new_dataset::<f64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if arr.iter().all(|v| v.is_string()) {
                let data: Vec<VarLenUnicode> =
                    arr.iter().filter_map(|v| v.as_str()).map(vlu).collect();
                group
                    .new_dataset::<VarLenUnicode>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            }
        }
        serde_json::Value::Object(_) => {
            let subgroup = group.create_group(name)?;
            write_uns_entries(&subgroup, value)?;
        }
        serde_json::Value::Null => {}
    }
    Ok(())
}
