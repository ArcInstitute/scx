// Write scx file back to h5ad format

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    Array, AsArray, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, LargeStringArray, RecordBatch, StringArray,
};
use arrow::datatypes::{
    DataType, Field, FieldRef, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    Schema, UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use hdf5::types::VarLenUnicode;
use ndarray::ArrayView1;

use crate::h5_write_util::vlu;

use scx_format_io::reader::ScxReader;

use crate::pipeline::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};
use crate::CATEGORICAL_ORDERED_KEY;

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

/// Write an SCX file to h5ad format.
pub fn write_scx_to_h5ad(
    scx_path: &Path,
    h5ad_path: &Path,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let reader = ScxReader::open(scx_path)?;
    let file = hdf5::File::create(h5ad_path)?;

    // Honor deletion vectors on every leg (X, obs, layers). The CSR
    // and layer readers filter by DV directly; obs goes through the
    // shared streaming-or-eager dispatcher which applies the same
    // global keep mask. Pre-fix, this path silently dropped DV
    // semantics — both /X and obs were written unfiltered.
    let keep_mask = super::stream_write::build_keep_mask(&reader)?;

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
    super::stream_write::write_obs_streaming_or_eager(&root, &reader, keep_mask.as_deref(), sink)?;
    super::stream_write::write_var_streaming_or_eager(&root, &reader, sink)?;

    // Write obsm (DV-filtered when active — obs-axis rows must match
    // /X and /obs). `read_all_obsm` returns Ok(empty) when absent, so a
    // propagated error means genuine corruption — never swallow it (SCX-009).
    let obsm_map = reader.read_all_obsm()?;
    if !obsm_map.is_empty() {
        let obsm_group = file.create_group("obsm")?;
        for (name, batch) in &obsm_map {
            let filtered = match keep_mask.as_deref() {
                Some(mask) => super::stream_write::filter_record_batch_by_mask(batch, mask)?,
                None => batch.clone(),
            };
            write_obsm_entry(&obsm_group, name, &filtered)?;
        }
    }

    // Write uns. Absence is a clean `SectionNotFound`; any other error
    // (e.g. malformed JSON) is corruption and must abort (SCX-009).
    match reader.read_uns() {
        Ok(uns) => {
            let uns_group = file.create_group("uns")?;
            write_uns_entries(&uns_group, &uns)?;
        }
        Err(scx_format_io::error::ScxError::SectionNotFound(_)) => {}
        Err(e) => return Err(e.into()),
    }

    // Write layers (DV-filtered when active — layers share X's
    // row count by AnnData invariant). `layer_names()` already enumerates the
    // layers present, so a read failure here is corruption, not absence — it
    // must abort rather than silently omit the layer (SCX-009).
    let layer_names = reader.layer_names();
    if !layer_names.is_empty() {
        let layers_group = file.create_group("layers")?;
        for layer_name in &layer_names {
            let layer_csr = reader.read_layer_filtered(layer_name)?;
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

    // Write /raw (DV-filtered on the obs axis like /X).
    write_raw_to_h5ad(&root, &reader, keep_mask.as_deref(), sink)?;

    // Write obsp / varp pairwise matrices (COO → csr_matrix groups). obsp is
    // square on the obs axis, so deletion vectors filter BOTH axes; varp lives
    // on the var axis and is never obs-deleted. Both readers return Ok(empty)
    // on absence, so errors are corruption and propagate (SCX-009).
    let obsp = reader.read_all_obsp()?;
    write_pairwise_group(&root, "obsp", &obsp, keep_mask.as_deref())?;
    let varp = reader.read_all_varp()?;
    write_pairwise_group(&root, "varp", &varp, None)?;

    Ok(())
}

/// Write the `adata.raw` group (`raw/X` + `raw/var`) into an output
/// h5ad if the SCX file carries a raw matrix. Raw shares X's obs axis,
/// so the same deletion-vector keep mask is applied to its rows. Shared
/// by the eager and streaming SCX→h5ad export paths.
pub(crate) fn write_raw_to_h5ad(
    root: &hdf5::Group,
    reader: &ScxReader,
    keep_mask: Option<&[bool]>,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    if !reader.has_raw() {
        return Ok(());
    }
    let raw = reader.read_all_raw_csr_shards()?;
    let raw_n_vars = raw.shape.1;
    let (indptr, indices, data, n_obs) = match keep_mask {
        Some(mask) => filter_csr_rows(&raw.indptr, &raw.indices, &raw.data, mask),
        None => (raw.indptr, raw.indices, raw.data, raw.shape.0),
    };

    let raw_group = root.create_group("raw")?;
    write_sparse_group_at(&raw_group, "X", &indptr, &indices, &data, n_obs, raw_n_vars)?;
    let raw_var = reader.read_raw_var()?;
    write_dataframe_group_at(&raw_group, "var", &raw_var, sink)?;
    Ok(())
}

/// Subset CSR rows by a boolean obs keep-mask, returning new
/// `(indptr, indices, data, n_kept_rows)`. Used to apply deletion
/// vectors to the raw matrix on export (raw shares the obs axis).
fn filter_csr_rows(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    mask: &[bool],
) -> (Vec<i64>, Vec<i32>, Vec<f32>, usize) {
    let n_rows = indptr.len().saturating_sub(1).min(mask.len());
    let mut out_indptr = vec![0i64];
    let mut out_indices = Vec::new();
    let mut out_data = Vec::new();
    for (row, &keep) in mask.iter().enumerate().take(n_rows) {
        if keep {
            let s = indptr[row] as usize;
            let e = indptr[row + 1] as usize;
            out_indices.extend_from_slice(&indices[s..e]);
            out_data.extend_from_slice(&data[s..e]);
            out_indptr.push(out_indices.len() as i64);
        }
    }
    let n = out_indptr.len() - 1;
    (out_indptr, out_indices, out_data, n)
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

pub(crate) fn write_sparse_group_at(
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

pub(crate) fn write_dataframe_group_at(
    parent: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let group = parent.group(name).or_else(|_| parent.create_group(name))?;
    write_dataframe_body(&group, name, batch, sink)
}

/// Shared body for `write_dataframe_group_at`. Caller is responsible
/// for opening or creating `group`. Resolves the pandas index from
/// schema metadata, renames pyarrow's `__index_level_0__` to anndata's
/// `_index` literal on disk (named indexes keep their original name),
/// excludes the index column from `column-order`, and always writes
/// `column-order` (length-0 OK) since anndata.read_h5ad requires the
/// attribute to be present.
/// Write the dataframe-level encoding attrs (`encoding-type="dataframe"`,
/// `encoding-version="0.2.0"`), resolve the pandas index field, and write the
/// `_index` attr. Shared prologue for the eager [`write_dataframe_body`] and
/// streaming [`write_dataframe_group_streaming`] writers — anndata.read_h5ad
/// requires these on every dataframe group, even an empty one.
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

fn write_dataframe_body(
    group: &hdf5::Group,
    df_name: &str,
    batch: &arrow::array::RecordBatch,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let schema = batch.schema();
    let (index_field_name, on_disk_index) = write_dataframe_header(group, schema.as_ref())?;

    let mut col_order: Vec<VarLenUnicode> =
        Vec::with_capacity(batch.num_columns().saturating_sub(1));
    for (col_idx, field) in schema.fields().iter().enumerate() {
        let col = batch.column(col_idx);
        if Some(field.name()) == index_field_name.as_ref() {
            // Index column → write under the anndata on-disk name and
            // exclude from `column-order` (matches anndata convention).
            // anndata requires `_index` to be a plain dataset, so the
            // nullable-group encoding is disabled for it.
            write_column_to_hdf5(group, df_name, &on_disk_index, col, field, false, sink)?;
        } else if write_column_to_hdf5(group, df_name, field.name(), col, field, true, sink)? {
            // Only list the column in `column-order` when a dataset
            // was actually created — unsupported types are
            // warn-and-skipped and must not appear in the index.
            col_order.push(vlu(field.name()));
        }
    }

    write_column_order_attr(group, &col_order)?;
    Ok(())
}

pub(crate) fn write_obsm_entry_at(
    obsm_group: &hdf5::Group,
    name: &str,
    batch: &arrow::array::RecordBatch,
) -> Result<(), ConvertError> {
    write_obsm_entry(obsm_group, name, batch)
}

pub(crate) fn write_uns_entries_at(
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

/// Read a COO column (`row` / `col`) as `i64`, accepting both the v1
/// (`Int32`) and v2 (`Int64`) coordinate widths the pairwise readers emit.
fn coo_coord_column(batch: &RecordBatch, name: &str) -> Result<Vec<i64>, ConvertError> {
    let col = batch.column_by_name(name).ok_or_else(|| {
        ConvertError::Other(format!("pairwise COO batch missing '{name}' column"))
    })?;
    if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        Ok(a.values().iter().map(|&v| v as i64).collect())
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Ok(a.values().to_vec())
    } else {
        Err(ConvertError::Other(format!(
            "pairwise COO column '{name}' is neither Int32 nor Int64"
        )))
    }
}

/// CSR arrays (`indptr`, `indices`, `data`) plus the square dimension `n`.
type CooCsr = (Vec<i64>, Vec<i32>, Vec<f32>, usize);

/// Convert a pairwise COO `RecordBatch` (`row`, `col`, `data: Float32` +
/// `n_rows` / `n_cols` schema metadata) into CSR arrays for an h5ad
/// `csr_matrix` group. When `keep` is `Some`, the matrix is filtered on
/// **both** axes by the obs keep-mask (pairwise matrices are square on the
/// obs axis) and indices are remapped into the compacted space; `varp`
/// passes `None`. Output is sorted by `(row, col)` and is `n × n`.
fn coo_batch_to_csr(batch: &RecordBatch, keep: Option<&[bool]>) -> Result<CooCsr, ConvertError> {
    let meta = batch.schema_ref().metadata();
    let n_rows: usize = meta
        .get("n_rows")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| ConvertError::Other("pairwise COO batch missing n_rows metadata".into()))?;

    let rows = coo_coord_column(batch, "row")?;
    let cols = coo_coord_column(batch, "col")?;
    let data = batch
        .column_by_name("data")
        .and_then(|c| c.as_any().downcast_ref::<Float32Array>())
        .ok_or_else(|| ConvertError::Other("pairwise COO batch missing Float32 'data'".into()))?;

    // Build the old→new index remap (identity when no deletions). `remap[i]`
    // is the compacted index of kept row/col `i`, or -1 when dropped.
    let (remap, n_out): (Option<Vec<i64>>, usize) = match keep {
        Some(mask) => {
            let mut remap = vec![-1i64; n_rows];
            let mut next = 0i64;
            for (i, slot) in remap.iter_mut().enumerate().take(n_rows.min(mask.len())) {
                if mask[i] {
                    *slot = next;
                    next += 1;
                }
            }
            (Some(remap), next as usize)
        }
        None => (None, n_rows),
    };

    // CSR `indices` are i32 (scipy zero-copy). The largest column index emitted
    // is `n_out - 1`; reject axes too wide to represent rather than silently
    // wrapping the `c as i32` narrowing below. Checked here (before the
    // `indptr` allocation) so an oversized dimension fails fast.
    if n_out > i32::MAX as usize {
        return Err(ConvertError::Other(format!(
            "pairwise matrix dimension {n_out} exceeds the i32 CSR index limit ({})",
            i32::MAX
        )));
    }

    // Filter + remap into (row, col, value) triples.
    let mut triples: Vec<(i64, i64, f32)> = Vec::with_capacity(rows.len());
    for k in 0..rows.len() {
        let (r, c) = (rows[k], cols[k]);
        let (nr, nc) = match &remap {
            Some(remap) => {
                let (Some(&nr), Some(&nc)) = (remap.get(r as usize), remap.get(c as usize)) else {
                    continue;
                };
                if nr < 0 || nc < 0 {
                    continue;
                }
                (nr, nc)
            }
            // No remap (varp / no deletions): coordinates index `indptr`
            // directly, so an out-of-range coord from a malformed COO section
            // would panic. Reject it as a conversion error instead.
            None => {
                if r < 0 || r >= n_rows as i64 || c < 0 || c >= n_rows as i64 {
                    return Err(ConvertError::Other(format!(
                        "pairwise COO coordinate ({r}, {c}) out of bounds for \
                         {n_rows}x{n_rows} matrix"
                    )));
                }
                (r, c)
            }
        };
        triples.push((nr, nc, data.value(k)));
    }

    // Canonical CSR: sort by (row, col), then build indptr.
    triples.sort_by_key(|&(r, c, _)| (r, c));
    let mut indptr = vec![0i64; n_out + 1];
    let mut indices = Vec::with_capacity(triples.len());
    let mut values = Vec::with_capacity(triples.len());
    for &(r, c, v) in &triples {
        indptr[r as usize + 1] += 1;
        indices.push(c as i32);
        values.push(v);
    }
    for i in 0..n_out {
        indptr[i + 1] += indptr[i];
    }

    Ok((indptr, indices, values, n_out))
}

/// Write `obsp` / `varp` pairwise matrices into an h5ad group (one
/// `csr_matrix` subgroup per key). `keep` filters both axes by the obs
/// keep-mask (pass `Some` for `obsp` under deletion vectors, `None` for
/// `varp`, which lives on the var axis and is never obs-deleted).
pub(crate) fn write_pairwise_group(
    parent: &hdf5::Group,
    group_name: &str,
    entries: &HashMap<String, RecordBatch>,
    keep: Option<&[bool]>,
) -> Result<(), ConvertError> {
    if entries.is_empty() {
        return Ok(());
    }
    let group = parent.create_group(group_name)?;
    for (name, batch) in entries {
        let (indptr, indices, data, n) = coo_batch_to_csr(batch, keep)?;
        let sub = group.create_group(name)?;
        write_sparse_arrays(&sub, &indptr, &indices, &data, n, n)?;
    }
    Ok(())
}

fn downcast_err(name: &str, expected: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': expected {expected} array but downcast failed"
    ))
}

/// Extract categorical codes promoted to i32 (-1 for null). Handles all
/// integer key widths Arrow / pandas uses (Int8/16/32/64, UInt8/16/32/64).
/// The SCX → h5ad writer emits i32 codes uniformly so anndata's reader
/// doesn't need to dispatch on key width.
fn dict_codes_i32(array: &dyn Array, name: &str) -> Result<Vec<i32>, ConvertError> {
    macro_rules! codes {
        ($t:ty, $label:literal) => {{
            let dict = array
                .as_any()
                .downcast_ref::<DictionaryArray<$t>>()
                .ok_or_else(|| downcast_err(name, $label))?;
            Ok(dict
                .keys()
                .iter()
                .map(|v| match v {
                    Some(k) => k as i32,
                    None => -1,
                })
                .collect())
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
        DataType::Int8 => codes!(Int8Type, "Dictionary<Int8, _>"),
        DataType::Int16 => codes!(Int16Type, "Dictionary<Int16, _>"),
        DataType::Int32 => codes!(Int32Type, "Dictionary<Int32, _>"),
        DataType::Int64 => codes!(Int64Type, "Dictionary<Int64, _>"),
        DataType::UInt8 => codes!(UInt8Type, "Dictionary<UInt8, _>"),
        DataType::UInt16 => codes!(UInt16Type, "Dictionary<UInt16, _>"),
        DataType::UInt32 => codes!(UInt32Type, "Dictionary<UInt32, _>"),
        DataType::UInt64 => codes!(UInt64Type, "Dictionary<UInt64, _>"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': unsupported categorical key type {other:?}"
        ))),
    }
}

/// Categorical category payload, carrying the source value class so the
/// h5ad writer emits a `categories` dataset of the matching HDF5 dtype.
/// String categories are the common case; integer/float categories let
/// numeric-keyed categoricals (integer cluster labels, dose levels)
/// round-trip instead of being dropped.
enum CatValues {
    Str(Vec<VarLenUnicode>),
    Int(Vec<i64>),
    Float(Vec<f64>),
}

/// Extract a categorical's category values, dispatching on the dictionary
/// value type (independent of the key width). Integer/unsigned widths
/// normalize to `i64`, floats to `f64`. Categories never carry nulls (the
/// codes carry NA via `-1`), so `value(i)` is always valid.
fn dict_category_values(array: &dyn Array, name: &str) -> Result<CatValues, ConvertError> {
    // `as_any_dictionary` panics on a non-dictionary array; callers only reach
    // here inside a `Dictionary(_, _)` match arm, but use the fallible form so
    // a wrong call surfaces a structured error instead of a panic.
    let dict = array.as_any_dictionary_opt().ok_or_else(|| {
        ConvertError::Other(format!(
            "column '{name}': expected Dictionary, got {:?}",
            array.data_type()
        ))
    })?;
    let values = dict.values();
    // Categories never carry nulls (the codes carry NA via `-1`). Iterate the
    // backing buffer / string array directly rather than the bounds-checked
    // `value(i)`.
    macro_rules! ints {
        ($t:ty) => {{
            let a = values.as_primitive::<$t>();
            let mut out = Vec::with_capacity(a.len());
            for &v in a.values().iter() {
                out.push(cat_int_to_i64(v, name)?);
            }
            CatValues::Int(out)
        }};
    }
    macro_rules! floats {
        ($t:ty) => {{
            let a = values.as_primitive::<$t>();
            CatValues::Float(a.values().iter().map(|&v| v as f64).collect())
        }};
    }
    Ok(match values.data_type() {
        DataType::Utf8 => {
            let a = values.as_string::<i32>();
            CatValues::Str(a.iter().map(|v| vlu(v.unwrap_or(""))).collect())
        }
        DataType::LargeUtf8 => {
            let a = values.as_string::<i64>();
            CatValues::Str(a.iter().map(|v| vlu(v.unwrap_or(""))).collect())
        }
        DataType::Int8 => ints!(Int8Type),
        DataType::Int16 => ints!(Int16Type),
        DataType::Int32 => ints!(Int32Type),
        DataType::Int64 => ints!(Int64Type),
        DataType::UInt8 => ints!(UInt8Type),
        DataType::UInt16 => ints!(UInt16Type),
        DataType::UInt32 => ints!(UInt32Type),
        DataType::UInt64 => ints!(UInt64Type),
        DataType::Float32 => floats!(Float32Type),
        DataType::Float64 => floats!(Float64Type),
        other => {
            return Err(ConvertError::Other(format!(
                "column '{name}': unsupported dictionary value type {other:?}"
            )));
        }
    })
}

fn cardinality_err(name: &str) -> ConvertError {
    ConvertError::Other(format!(
        "column '{name}': categorical cardinality exceeds i32::MAX"
    ))
}

/// Normalize a categorical integer value to `i64`, **rejecting** an unsigned
/// value above `i64::MAX` rather than silently wrapping it to a negative label
/// (which would corrupt the category on round-trip). Signed widths and unsigned
/// widths ≤ 32 bits always fit, so only `UInt64` can actually fail; the checked
/// form is applied uniformly across widths (and on both the dictionary and the
/// plain-shard paths) so the two stay symmetric. The h5ad integer-categorical
/// `categories` dataset is `i64`, so out-of-range unsigned labels are
/// genuinely unrepresentable and must error rather than mis-encode.
fn cat_int_to_i64<T>(v: T, name: &str) -> Result<i64, ConvertError>
where
    T: TryInto<i64> + std::fmt::Display + Copy,
{
    v.try_into().map_err(|_| {
        ConvertError::Other(format!(
            "column '{name}': categorical integer value {v} exceeds the i64 range \
             of the h5ad integer-categorical encoding"
        ))
    })
}

/// Build a per-shard local categorical view `(local_codes, local_values)` for
/// the streaming categorical writer, accepting **both** a `Dictionary(_, V)`
/// array (the existing append-grown base shards) and a **plain `V`** array
/// (appended shards, where `append` decoded the dictionary to its value type
/// before writing — see `scx-ops::unify_dict_columns`). For the plain case it
/// builds a local first-seen dedup: `local_codes[row]` is the local index of
/// that row's value (or `-1` when the row is null, keyed on *validity* — a
/// genuine empty-string category is distinct from a null), and `local_values`
/// lists the distinct values in first-seen order.
///
/// Generic over the value class (string / integer / float) so numeric
/// categoricals reconcile exactly as strings do — matching
/// `reconcile_dictionary_representations` on the read side. The downstream
/// `remap!` then folds the result into the cross-shard `CatAccum` identically
/// for both representations (a value-class mismatch vs the accumulator — e.g. a
/// plain `Int64` shard under a `Dictionary(_, Utf8)` column — is rejected by the
/// `remap!` match's catch-all arm; §3.2's validator relax rejects it earlier).
fn local_categorical_view(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<i32>, CatValues), ConvertError> {
    if matches!(array.data_type(), DataType::Dictionary(_, _)) {
        return Ok((
            dict_codes_i32(array, name)?,
            dict_category_values(array, name)?,
        ));
    }

    let n = array.len();
    let mut local_codes = vec![-1i32; n];
    macro_rules! intern_int {
        ($t:ty) => {{
            let a = array.as_primitive::<$t>();
            let mut order: Vec<i64> = Vec::new();
            let mut seen: HashMap<i64, i32> = HashMap::new();
            for i in 0..n {
                if a.is_valid(i) {
                    let v = cat_int_to_i64(a.value(i), name)?;
                    let code = match seen.get(&v) {
                        Some(&c) => c,
                        None => {
                            let c: i32 =
                                order.len().try_into().map_err(|_| cardinality_err(name))?;
                            seen.insert(v, c);
                            order.push(v);
                            c
                        }
                    };
                    local_codes[i] = code;
                }
            }
            Ok((local_codes, CatValues::Int(order)))
        }};
    }
    macro_rules! intern_float {
        ($t:ty) => {{
            let a = array.as_primitive::<$t>();
            let mut order: Vec<f64> = Vec::new();
            let mut seen: HashMap<u64, i32> = HashMap::new();
            for i in 0..n {
                if a.is_valid(i) {
                    let v = a.value(i) as f64;
                    let k = v.to_bits();
                    let code = match seen.get(&k) {
                        Some(&c) => c,
                        None => {
                            let c: i32 =
                                order.len().try_into().map_err(|_| cardinality_err(name))?;
                            seen.insert(k, c);
                            order.push(v);
                            c
                        }
                    };
                    local_codes[i] = code;
                }
            }
            Ok((local_codes, CatValues::Float(order)))
        }};
    }
    macro_rules! intern_str {
        ($a:expr) => {{
            let a = $a;
            let mut order: Vec<VarLenUnicode> = Vec::new();
            // Key by `&str` borrowed from `a` (valid for this block) to avoid
            // allocating a `String` per distinct category in this hot path.
            let mut seen: HashMap<&str, i32> = HashMap::new();
            for i in 0..n {
                if a.is_valid(i) {
                    let v = a.value(i);
                    let code = match seen.get(v) {
                        Some(&c) => c,
                        None => {
                            let c: i32 =
                                order.len().try_into().map_err(|_| cardinality_err(name))?;
                            seen.insert(v, c);
                            order.push(vlu(v));
                            c
                        }
                    };
                    local_codes[i] = code;
                }
            }
            Ok((local_codes, CatValues::Str(order)))
        }};
    }
    match array.data_type() {
        DataType::Utf8 => intern_str!(array.as_string::<i32>()),
        DataType::LargeUtf8 => intern_str!(array.as_string::<i64>()),
        DataType::Int8 => intern_int!(Int8Type),
        DataType::Int16 => intern_int!(Int16Type),
        DataType::Int32 => intern_int!(Int32Type),
        DataType::Int64 => intern_int!(Int64Type),
        DataType::UInt8 => intern_int!(UInt8Type),
        DataType::UInt16 => intern_int!(UInt16Type),
        DataType::UInt32 => intern_int!(UInt32Type),
        DataType::UInt64 => intern_int!(UInt64Type),
        DataType::Float32 => intern_float!(Float32Type),
        DataType::Float64 => intern_float!(Float64Type),
        other => Err(ConvertError::Other(format!(
            "column '{name}': categorical column has unsupported plain shard type {other:?}"
        ))),
    }
}

/// Whether the streaming/eager categorical writers can preserve a
/// dictionary with this value type. Mirrors the dtypes [`dict_category_values`]
/// handles (string + every integer / unsigned / float width); anything else
/// (e.g. `Dictionary(_, Boolean)`) takes the warn-and-skip path.
fn is_supported_cat_value_type(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

/// Cross-shard global categorical accumulator for the streaming exporter.
/// One variant per category value class (matching [`CatValues`]); the `dict`
/// maps a category to its global code and `order` preserves insertion order
/// for the final `categories` dataset. Floats are deduped by bit pattern
/// (`f64::to_bits`) — categorical category values are exact (cluster labels,
/// dose levels), and pandas does not emit `NaN` categories.
enum CatAccum {
    Str {
        dict: HashMap<String, i32>,
        order: Vec<String>,
    },
    Int {
        dict: HashMap<i64, i32>,
        order: Vec<i64>,
    },
    Float {
        dict: HashMap<u64, i32>,
        order: Vec<f64>,
    },
}

impl CatAccum {
    /// Pick the accumulator variant for a dictionary value type. The caller
    /// only reaches this for types accepted by [`is_supported_cat_value_type`];
    /// non-numeric, non-string types fall back to `Str` (unreachable in
    /// practice).
    fn new(value_type: &DataType) -> Self {
        match value_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64 => CatAccum::Int {
                dict: HashMap::new(),
                order: Vec::new(),
            },
            DataType::Float32 | DataType::Float64 => CatAccum::Float {
                dict: HashMap::new(),
                order: Vec::new(),
            },
            _ => CatAccum::Str {
                dict: HashMap::new(),
                order: Vec::new(),
            },
        }
    }
}

/// Collect a `Utf8` / `LargeUtf8` string column into `(values, mask)`:
/// `values[i]` is the string with null positions filled with `""`, and
/// `mask[i] == true` ⇔ row `i` is null. Dispatches on the concrete array
/// type so both narrow (`StringArray`) and wide (`LargeStringArray`)
/// offsets are handled, mirroring the streaming string writer's
/// `Utf8 | LargeUtf8` arm.
fn string_values_and_mask(
    array: &dyn Array,
    name: &str,
) -> Result<(Vec<VarLenUnicode>, Vec<bool>), ConvertError> {
    macro_rules! collect {
        ($t:ty, $label:literal) => {{
            let arr = array
                .as_any()
                .downcast_ref::<$t>()
                .ok_or_else(|| downcast_err(name, $label))?;
            let values: Vec<VarLenUnicode> = (0..arr.len())
                .map(|i| {
                    if arr.is_valid(i) {
                        vlu(arr.value(i))
                    } else {
                        vlu("")
                    }
                })
                .collect();
            let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();
            Ok((values, mask))
        }};
    }
    match array.data_type() {
        DataType::Utf8 => collect!(arrow::array::StringArray, "Utf8"),
        DataType::LargeUtf8 => collect!(LargeStringArray, "LargeUtf8"),
        other => Err(ConvertError::Other(format!(
            "column '{name}': expected Utf8/LargeUtf8, got {other:?}"
        ))),
    }
}

/// Write one Arrow column into `group/name`. Returns `Ok(true)` on a
/// supported type (dataset created), `Ok(false)` when the type is
/// not yet supported and the column was warn-and-skipped. The caller
/// uses the boolean to decide whether to add `name` to `column-order`
/// — adding a name without a backing dataset breaks
/// `anndata.read_h5ad`'s lookup.
///
/// `allow_nullable_group` is `false` for the pandas index column
/// (`_index`), which anndata requires to be a plain dataset, never a
/// nullable group. For every other column, integer / string columns
/// that actually contain nulls are written using anndata's
/// `nullable-integer` / `nullable-string-array` group encodings
/// (`values` + `mask`), preserving null state; null-free columns stay
/// plain datasets (byte-identical to the pre-fix output). Floats are
/// always plain datasets with `NaN` at null positions — anndata has no
/// `nullable-float` encoding, so `NaN` is the canonical missing-float
/// representation.
fn write_column_to_hdf5(
    group: &hdf5::Group,
    df_name: &str,
    name: &str,
    array: &dyn Array,
    field: &Field,
    allow_nullable_group: bool,
    sink: &mut WarningSink,
) -> Result<bool, ConvertError> {
    // `dtype` selects the encoding; `ordered` (categorical only) is
    // resolved from the field's `scx.categorical.ordered` metadata.
    let dtype = field.data_type();
    let ordered = field_ordered(field);
    match dtype {
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(name, "Int32"))?;
            if allow_nullable_group && arr.null_count() > 0 {
                let values: Vec<i32> = (0..arr.len())
                    .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                    .collect();
                let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();
                write_nullable_group(group, name, &values, &mask, "nullable-integer")?;
            } else {
                warn_index_coerced_nulls(sink, df_name, name, "Int32", arr, allow_nullable_group);
                let values: Vec<i32> = arr.iter().map(|v| v.unwrap_or(0)).collect();
                group
                    .new_dataset::<i32>()
                    .shape([values.len()])
                    .create(name)?
                    .write(&values)?;
            }
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(name, "Int64"))?;
            if allow_nullable_group && arr.null_count() > 0 {
                let values: Vec<i64> = (0..arr.len())
                    .map(|i| if arr.is_valid(i) { arr.value(i) } else { 0 })
                    .collect();
                let mask: Vec<bool> = (0..arr.len()).map(|i| !arr.is_valid(i)).collect();
                write_nullable_group(group, name, &values, &mask, "nullable-integer")?;
            } else {
                warn_index_coerced_nulls(sink, df_name, name, "Int64", arr, allow_nullable_group);
                let values: Vec<i64> = arr.iter().map(|v| v.unwrap_or(0)).collect();
                group
                    .new_dataset::<i64>()
                    .shape([values.len()])
                    .create(name)?
                    .write(&values)?;
            }
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(name, "Float32"))?;
            // NaN is anndata's canonical missing-float value — lossless,
            // unlike the prior `0.0` coercion.
            let values: Vec<f32> = arr.iter().map(|v| v.unwrap_or(f32::NAN)).collect();
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
            let values: Vec<f64> = arr.iter().map(|v| v.unwrap_or(f64::NAN)).collect();
            group
                .new_dataset::<f64>()
                .shape([values.len()])
                .create(name)?
                .write(&values)?;
        }
        DataType::Utf8 | DataType::LargeUtf8 => {
            // Handle both narrow (`StringArray`, i32 offsets) and wide
            // (`LargeStringArray`, i64 offsets) string columns; the eager
            // assembled-obs path can legitimately carry either, matching
            // the streaming writer's `Utf8 | LargeUtf8` handling.
            let (values, mask) = string_values_and_mask(array, name)?;
            let null_count = mask.iter().filter(|&&m| m).count();
            if allow_nullable_group && null_count > 0 {
                write_nullable_group(group, name, &values, &mask, "nullable-string-array")?;
            } else {
                warn_index_coerced_nulls(sink, df_name, name, "Utf8", array, allow_nullable_group);
                group
                    .new_dataset::<VarLenUnicode>()
                    .shape([values.len()])
                    .create(name)?
                    .write(&values)?;
            }
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
        DataType::Dictionary(_key_type, _value_type) => {
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
            //
            // Categories keep their source class (string / integer /
            // float): anndata reconstructs a `pd.Categorical` from a
            // numeric `categories` dataset, so integer-/float-keyed
            // categoricals round-trip instead of being dropped.
            // Extract categories + codes *before* creating the group so an
            // unsupported value type OR key width leaves no partial group
            // behind. Both fall through to the same warn-and-skip (an
            // unsupported key type previously aborted the whole export via
            // `?` — now it skips the column like an unsupported value type),
            // and the warning carries the underlying error so a dropped
            // column is debuggable.
            macro_rules! skip_unsupported {
                ($e:expr) => {{
                    sink.emit(ConvertWarning::UnsupportedExportColumn {
                        column: format!("{df_name}/{name}"),
                        dtype: format!("{dtype:?} ({})", $e),
                    });
                    return Ok(false);
                }};
            }
            let cats = match dict_category_values(array, name) {
                Ok(c) => c,
                Err(e) => skip_unsupported!(e),
            };
            let codes = match dict_codes_i32(array, name) {
                Ok(c) => c,
                Err(e) => skip_unsupported!(e),
            };

            let cat_group = group.create_group(name)?;
            cat_group
                .new_dataset::<i32>()
                .shape([codes.len()])
                .create("codes")?
                .write(&codes)?;
            match cats {
                CatValues::Str(cats) => {
                    cat_group
                        .new_dataset::<VarLenUnicode>()
                        .shape([cats.len()])
                        .create("categories")?
                        .write(&cats)?;
                }
                CatValues::Int(cats) => {
                    cat_group
                        .new_dataset::<i64>()
                        .shape([cats.len()])
                        .create("categories")?
                        .write(&cats)?;
                }
                CatValues::Float(cats) => {
                    cat_group
                        .new_dataset::<f64>()
                        .shape([cats.len()])
                        .create("categories")?
                        .write(&cats)?;
                }
            }

            cat_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("categorical"))?;
            cat_group
                .new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.2.0"))?;
            // The pandas `ordered` bit is carried in the Arrow field
            // metadata (`scx.categorical.ordered`) the h5ad reader stamps;
            // `ordered` is resolved from it by the caller. Native HDF5
            // bool — anndata's categorical reader expects
            // `H5T_NATIVE_HBOOL_8`, not u8.
            cat_group
                .new_attr::<bool>()
                .create("ordered")?
                .write_scalar(&ordered)?;
        }
        _ => {
            sink.emit(ConvertWarning::UnsupportedExportColumn {
                column: format!("{df_name}/{name}"),
                dtype: format!("{dtype:?}"),
            });
            return Ok(false);
        }
    }
    Ok(true)
}

/// Write an anndata nullable group (`encoding-type` ∈ {`nullable-integer`,
/// `nullable-string-array`}, version `0.1.0`): a subgroup with a `values`
/// dataset (null positions filled with `0` / `""`) and a boolean `mask`
/// dataset (`mask[i] == true` ⇔ null). Byte-for-byte the shape anndata's
/// `write_nullable` emits and `_read_nullable` consumes. Generic over the
/// HDF5 element type so the same helper serves integer and string values.
fn write_nullable_group<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    values: &[T],
    mask: &[bool],
    encoding_type: &str,
) -> Result<(), ConvertError> {
    let g = group.create_group(name)?;
    g.new_dataset::<T>()
        .shape([values.len()])
        .create("values")?
        .write(values)?;
    g.new_dataset::<bool>()
        .shape([mask.len()])
        .create("mask")?
        .write(mask)?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-type")?
        .write_scalar(&vlu(encoding_type))?;
    g.new_attr::<VarLenUnicode>()
        .create("encoding-version")?
        .write_scalar(&vlu("0.1.0"))?;
    Ok(())
}

/// Surface the one residual null-coercion case: the pandas index column
/// (`_index`), which anndata requires to be a plain dataset and so can
/// never use a nullable group. Indexes virtually never carry nulls, but
/// if one does its null becomes `0` / `""` — emit
/// [`ConvertWarning::CoercedNulls`] so that is visible. No-op for
/// non-index columns (`allow_nullable_group == true`) or when there are
/// no nulls.
fn warn_index_coerced_nulls(
    sink: &mut WarningSink,
    df_name: &str,
    col_name: &str,
    dtype: &str,
    array: &dyn Array,
    allow_nullable_group: bool,
) {
    if !allow_nullable_group && array.null_count() > 0 {
        sink.emit(ConvertWarning::CoercedNulls {
            column: format!("{df_name}/{col_name}"),
            dtype: dtype.to_string(),
            count: array.null_count() as u64,
        });
    }
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
}

/// Pre-scan metadata shards to decide (a) which integer / string columns
/// need anndata's nullable group encoding, and (b) which columns are a
/// `Dictionary` in any shard (so the unified export schema can declare
/// them categorical — see [`write_dataframe_group_streaming`]).
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
pub(crate) fn scan_column_export_layout<I>(
    shards: I,
    schema: &Schema,
) -> Result<ColumnExportLayout, ConvertError>
where
    I: IntoIterator<Item = Result<RecordBatch, scx_format_io::error::ScxError>>,
{
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
    Ok(ColumnExportLayout {
        needs_nullable,
        dict_fields,
    })
}

/// Two categorical value types belong to the same value *class* for export
/// reconciliation: `Utf8`/`LargeUtf8` are interchangeable; every other type
/// must match exactly. Used to reject a dictionary whose value class differs
/// across shards (corruption) while allowing the harmless narrow/wide string
/// difference the per-shard batches legitimately carry.
///
/// Numeric widths are required to match **exactly** here (e.g. a plain `Int32`
/// shard under a `Dictionary(_, Int64)` column is rejected), which is
/// intentionally stricter than the read-side `reconcile_dictionary_representations`,
/// where `arrow::compute::cast` would coerce integer widths. The real
/// `append`/`from_anndata` flow never produces a width-mismatched layout —
/// `scx-ops::unify_dict_columns` preserves the exact value type `V` — so the only
/// way to hit the difference is a hand-crafted third-party file, where failing
/// loudly on export is preferable to a silent width coercion.
fn cat_value_class_eq(a: &DataType, b: &DataType) -> bool {
    fn is_string_like(t: &DataType) -> bool {
        matches!(t, DataType::Utf8 | DataType::LargeUtf8)
    }
    a == b || (is_string_like(a) && is_string_like(b))
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
/// `needs_nullable` is aligned with `schema.fields()`: when
/// `needs_nullable[i]` is `true`, the integer / string column at field
/// `i` is written with anndata's `nullable-integer` /
/// `nullable-string-array` group encoding (it contains nulls); otherwise
/// it is written as a plain dataset. Computed up front by
/// [`scan_column_export_layout`] because the HDF5 datasets must be
/// allocated before any shard is seen. Float columns ignore this flag
/// (always plain datasets with `NaN` at nulls).
///
/// `schema` is the unified export schema from
/// [`build_unified_export_schema`]: a column that is a `Dictionary` in any
/// shard is declared categorical here even when shard 0 was plain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_dataframe_group_streaming<I>(
    parent: &hdf5::Group,
    name: &str,
    schema: &Schema,
    shards: I,
    n_rows_kept: usize,
    keep_mask_opt: Option<&[bool]>,
    needs_nullable: &[bool],
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
        let want_nullable = !is_index && needs_nullable.get(col_idx).copied().unwrap_or(false);
        let writer = create_column_writer(&group, on_disk_name, field, n_rows_kept, want_nullable)?;
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
        // local categories in (insertion order preserved), and
        // `finalize_column_writer` writes a `categories` dataset of the
        // matching HDF5 dtype.
        accum: CatAccum,
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
/// encoding attrs) for the streaming path. Generic over the HDF5 value
/// element type so it serves both `nullable-integer` (i32/i64) and
/// `nullable-string-array` (`VarLenUnicode`). The streaming sibling of
/// [`write_nullable_group`].
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
            accum,
            ..
        } => {
            // Accept both a `Dictionary(_, V)` shard (base) and a plain `V`
            // shard (appended): the helper normalizes both to local codes +
            // distinct values, generic over the value class (§3.3).
            let (local_codes, local_values) = local_categorical_view(array, name)?;

            // C10: intern only the dictionary values actually referenced by
            // `kept_local` rows, so categories present only in
            // deletion-dropped rows don't enter the global vocabulary. The
            // local→global map is filled lazily as kept codes are visited.
            // `key`/`store` adapt the shared remap to the value class: the
            // dedup-map key (`String` / `i64` / `u64`-bits) and the stored
            // category order value. Cardinality is capped at i32::MAX
            // (defensive — real categoricals stay well below).
            macro_rules! remap {
                ($vals:expr, $dict:expr, $order:expr, $key:expr, $store:expr) => {{
                    let vals = $vals;
                    let mut local_to_global: Vec<Option<i32>> = vec![None; vals.len()];
                    let mut kept_codes: Vec<i32> = Vec::with_capacity(kept_local.len());
                    for &i in kept_local {
                        let lc = local_codes[i];
                        if lc < 0 {
                            kept_codes.push(-1);
                            continue;
                        }
                        let lc_idx = lc as usize;
                        let g = match local_to_global[lc_idx] {
                            Some(g) => g,
                            None => {
                                let v = &vals[lc_idx];
                                let k = $key(v);
                                let g = match $dict.get(&k) {
                                    Some(&g) => g,
                                    None => {
                                        let g: i32 = $order.len().try_into().map_err(|_| {
                                            ConvertError::Other(format!(
                                                "column '{name}': categorical cardinality exceeds i32::MAX"
                                            ))
                                        })?;
                                        $dict.insert(k, g);
                                        $order.push($store(v));
                                        g
                                    }
                                };
                                local_to_global[lc_idx] = Some(g);
                                g
                            }
                        };
                        kept_codes.push(g);
                    }
                    kept_codes
                }};
            }

            let kept_codes = match (&mut *accum, local_values) {
                (CatAccum::Str { dict, order }, CatValues::Str(vals)) => {
                    let svals: Vec<String> = vals.iter().map(|v| v.as_str().to_string()).collect();
                    remap!(svals, dict, order, |v: &String| v.clone(), |v: &String| v
                        .clone())
                }
                (CatAccum::Int { dict, order }, CatValues::Int(vals)) => {
                    remap!(vals, dict, order, |v: &i64| *v, |v: &i64| *v)
                }
                (CatAccum::Float { dict, order }, CatValues::Float(vals)) => {
                    remap!(vals, dict, order, |v: &f64| v.to_bits(), |v: &f64| *v)
                }
                _ => {
                    // The accumulator variant is fixed from the column's
                    // declared value type at writer creation; every shard
                    // must present the same class. A mismatch means a
                    // malformed file (shards disagree on dtype).
                    return Err(ConvertError::Other(format!(
                        "column '{name}': categorical value type changed across shards"
                    )));
                }
            };
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
            // 2-D numeric arrays (e.g. color/contrast matrices) — mirror the
            // nested-JSON shape produced by `read_uns_entry`'s 2-D arm so they
            // round-trip rather than being silently dropped. Ragged / empty /
            // non-numeric nested arrays fall through to the no-op skip (a true
            // HDF5 round-trip is always rectangular, so ragged only arises from
            // synthetic JSON).
            if !arr.is_empty() && arr.iter().all(|v| v.is_array()) {
                write_uns_2d_array(group, name, arr)?;
            } else if arr.iter().all(|v| v.is_i64()) {
                let data: Vec<i64> = arr.iter().filter_map(|v| v.as_i64()).collect();
                group
                    .new_dataset::<i64>()
                    .shape([data.len()])
                    .create(name)?
                    .write(&data)?;
            } else if !arr.is_empty() && arr.iter().all(|v| v.is_u64()) {
                // Unsigned values above i64::MAX (C3): keep them exact instead
                // of coercing to f64 via the `is_number` arm below.
                let data: Vec<u64> = arr.iter().filter_map(|v| v.as_u64()).collect();
                group
                    .new_dataset::<u64>()
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
            } else if !arr.is_empty() && arr.iter().all(|v| v.is_boolean()) {
                // 1-D boolean (C3): the scalar bool arm above confirms the
                // `bool` H5Type; without this arm bool vectors are dropped.
                let data: Vec<bool> = arr.iter().filter_map(|v| v.as_bool()).collect();
                group
                    .new_dataset::<bool>()
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
        serde_json::Value::Object(map) => {
            // A tagged `__scx_type__` envelope (emitted by `read_uns_entry`'s
            // `uns_ndarray_envelope` for NaN/Inf-carrying floats and rank ≥ 3
            // arrays) decodes back to a native HDF5 dataset. Anything else —
            // a plain dict, or a pyscx-only envelope (recarray / tuple /
            // pandas.*) — falls through to the generic subgroup recursion.
            if !try_write_uns_envelope(group, name, map)? {
                let subgroup = group.create_group(name)?;
                write_uns_entries(&subgroup, value)?;
            }
        }
        serde_json::Value::Null => {
            // anndata represents a Python `None` uns value as an `h5py.Empty`
            // dataset: an HDF5 null dataspace (0 elements) tagged
            // `encoding-type="null"`. Emit the same encoding so the value
            // round-trips as `None` rather than being silently dropped (and
            // read back as a bogus `0.0`, which breaks e.g. scanpy's
            // `uns['log1p']['base']`). Matches the null-dataspace detection
            // in `read_uns_entry`.
            let ds = group
                .new_dataset::<f32>()
                .shape(hdf5::Extents::null())
                .create(name)?;
            ds.new_attr::<VarLenUnicode>()
                .create("encoding-type")?
                .write_scalar(&vlu("null"))?;
            ds.new_attr::<VarLenUnicode>()
                .create("encoding-version")?
                .write_scalar(&vlu("0.1.0"))?;
        }
    }
    Ok(())
}

/// Decode a tagged `__scx_type__` `uns` envelope back to a native HDF5
/// dataset. Handles the numeric `ndarray` (`encoding == "base64le"`) and
/// `scalar` envelopes emitted by `read_uns_entry::uns_ndarray_envelope`
/// (the inverse of pyscx's `encode_ndarray_tagged` / `encode_np_scalar_tagged`).
/// Returns `Ok(false)` for a non-envelope object or any other tag/encoding so
/// the caller falls back to the generic subgroup recursion (no regression for
/// pyscx-only `json` / `recarray` / `tuple` / `pandas.*` envelopes).
fn try_write_uns_envelope(
    group: &hdf5::Group,
    name: &str,
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<bool, ConvertError> {
    let tag = match map
        .get(crate::h5ad::read::SCX_UNS_TYPE_KEY)
        .and_then(|v| v.as_str())
    {
        Some(t) => t,
        None => return Ok(false),
    };
    let (dtype, shape): (&str, Vec<usize>) = match tag {
        "ndarray" => {
            if map.get("encoding").and_then(|v| v.as_str()) != Some("base64le") {
                return Ok(false); // json-encoded string/object array — not ours
            }
            let dtype = match map.get("dtype").and_then(|v| v.as_str()) {
                Some(d) => d,
                None => return Ok(false),
            };
            // Strict: every dimension must be a valid non-negative integer.
            // A `null` / negative / fractional dim would otherwise be silently
            // dropped, producing a wrong-rank dataset with reinterpreted bytes.
            // Fall back to the generic subgroup write instead (preserves the
            // raw envelope, never corrupts).
            let shape: Vec<usize> = match map.get("shape").and_then(|v| v.as_array()) {
                Some(a) => match a
                    .iter()
                    .map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<usize>>>()
                {
                    Some(s) => s,
                    None => return Ok(false),
                },
                None => return Ok(false),
            };
            (dtype, shape)
        }
        // A `scalar` envelope is a 0-d value (empty shape).
        "scalar" => match map.get("dtype").and_then(|v| v.as_str()) {
            Some(d) => (d, Vec::new()),
            None => return Ok(false),
        },
        _ => return Ok(false),
    };

    let b64 = match map.get("data").and_then(|v| v.as_str()) {
        Some(d) => d,
        None => return Ok(false),
    };
    use base64::Engine;
    let bytes = match base64::engine::general_purpose::STANDARD.decode(b64) {
        Ok(b) => b,
        // Corrupt base64 — fall back rather than abort the whole export.
        Err(_) => return Ok(false),
    };

    // Returns false (→ generic fallback) for any dtype/order/itemsize we can't
    // faithfully decode; Err only on a genuine HDF5 write failure.
    write_uns_envelope_dataset(group, name, dtype, &shape, &bytes)
}

/// Reinterpret little-endian `bytes` as `dtype` and write a native HDF5
/// dataset of the given `shape` (empty `shape` → a 0-d scalar). Inverse of
/// the `to_le_bytes` packing in `uns_ndarray_envelope`.
///
/// Returns `Ok(true)` when the dataset was written, `Ok(false)` when the
/// envelope can't be faithfully decoded (unsupported byte order / dtype /
/// itemsize, or a byte count that isn't a multiple of itemsize) so the caller
/// falls back to writing the raw envelope as a subgroup — never aborting the
/// export and never reinterpreting bytes at the wrong width. `Err` is reserved
/// for a genuine HDF5 write failure.
fn write_uns_envelope_dataset(
    group: &hdf5::Group,
    name: &str,
    dtype: &str,
    shape: &[usize],
    bytes: &[u8],
) -> Result<bool, ConvertError> {
    // numpy `dtype.str`: byte-order char, kind char, then itemsize.
    let mut chars = dtype.chars();
    let order = chars.next();
    let kind = chars.next();
    let itemsize: usize = chars.as_str().parse().unwrap_or(0);
    // We only emit little-endian (`<`) or single-byte (`|`); `=` is native
    // (LE on supported hosts). Anything else (e.g. a big-endian `>f4` from a
    // pyscx-tagged AnnData uns) falls back to the generic subgroup write rather
    // than risking a byte-swapped misread or aborting the export.
    if !matches!(order, Some('<') | Some('|') | Some('=')) {
        return Ok(false);
    }

    macro_rules! emit {
        ($t:ty, $w:expr, $conv:expr) => {{
            let chunks = bytes.chunks_exact($w);
            if !chunks.remainder().is_empty() {
                // Truncated / corrupt payload — fall back instead of writing a
                // dataset from a partial buffer.
                return Ok(false);
            }
            let vals: Vec<$t> = chunks.map($conv).collect();
            write_typed_uns_dataset::<$t>(group, name, shape, vals)?;
            Ok(true)
        }};
    }

    match (kind, itemsize) {
        (Some('f'), 2) => emit!(half::f16, 2, |c: &[u8]| half::f16::from_le_bytes([
            c[0], c[1]
        ])),
        (Some('f'), 4) => emit!(f32, 4, |c: &[u8]| f32::from_le_bytes(c.try_into().unwrap())),
        (Some('f'), 8) => emit!(f64, 8, |c: &[u8]| f64::from_le_bytes(c.try_into().unwrap())),
        (Some('i'), 1) => emit!(i8, 1, |c: &[u8]| c[0] as i8),
        (Some('i'), 2) => emit!(i16, 2, |c: &[u8]| i16::from_le_bytes(c.try_into().unwrap())),
        (Some('i'), 4) => emit!(i32, 4, |c: &[u8]| i32::from_le_bytes(c.try_into().unwrap())),
        (Some('i'), 8) => emit!(i64, 8, |c: &[u8]| i64::from_le_bytes(c.try_into().unwrap())),
        (Some('u'), 1) => emit!(u8, 1, |c: &[u8]| c[0]),
        (Some('u'), 2) => emit!(u16, 2, |c: &[u8]| u16::from_le_bytes(c.try_into().unwrap())),
        (Some('u'), 4) => emit!(u32, 4, |c: &[u8]| u32::from_le_bytes(c.try_into().unwrap())),
        (Some('u'), 8) => emit!(u64, 8, |c: &[u8]| u64::from_le_bytes(c.try_into().unwrap())),
        (Some('b'), 1) => emit!(bool, 1, |c: &[u8]| c[0] != 0),
        // Unsupported kind/itemsize (datetime, complex, …) — fall back.
        _ => Ok(false),
    }
}

/// Write `vals` as a 0-d scalar (empty `shape`) or an N-D HDF5 dataset.
fn write_typed_uns_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    shape: &[usize],
    vals: Vec<T>,
) -> Result<(), ConvertError> {
    if shape.is_empty() {
        if vals.len() != 1 {
            return Err(ConvertError::Other(format!(
                "uns envelope '{name}': scalar expected 1 element, got {}",
                vals.len()
            )));
        }
        group
            .new_dataset::<T>()
            .shape(())
            .create(name)?
            .write_scalar(&vals[0])?;
    } else {
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), vals)
            .map_err(|e| ConvertError::Other(format!("uns envelope '{name}': shape: {e}")))?;
        group
            .new_dataset::<T>()
            .shape(shape)
            .create(name)?
            .write(&arr)?;
    }
    Ok(())
}

/// Write a 2-D numeric `uns` array (nested JSON `[[..],[..]]`) as a rectangular
/// HDF5 dataset, mirroring the dtype set of `read_uns_entry`'s 2-D arm
/// (Integer/Unsigned/Float/Boolean). Ragged, empty, or non-numeric inputs are
/// skipped (no-op) rather than errored, matching the lenient behavior of the
/// surrounding scalar/1-D arms. The caller guarantees every element is an array.
fn write_uns_2d_array(
    group: &hdf5::Group,
    name: &str,
    arr: &[serde_json::Value],
) -> Result<(), ConvertError> {
    let rows: Vec<&Vec<serde_json::Value>> = arr.iter().filter_map(|v| v.as_array()).collect();
    let n_rows = rows.len();
    let n_cols = rows[0].len();
    // Rectangular and non-degenerate, else skip.
    if n_cols == 0 || rows.iter().any(|r| r.len() != n_cols) {
        return Ok(());
    }
    let cells = || rows.iter().flat_map(|r| r.iter());

    // Detect element dtype over all cells, mirroring the read-path ordering:
    // i64 first (catches negatives + small ints), then u64 (large unsigned),
    // then f64, then bool. Anything else (mixed / non-numeric) is skipped.
    if cells().all(|v| v.is_i64()) {
        let flat: Vec<i64> = cells().filter_map(|v| v.as_i64()).collect();
        write_2d_dataset::<i64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_u64()) {
        let flat: Vec<u64> = cells().filter_map(|v| v.as_u64()).collect();
        write_2d_dataset::<u64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_number()) {
        let flat: Vec<f64> = cells().filter_map(|v| v.as_f64()).collect();
        write_2d_dataset::<f64>(group, name, n_rows, n_cols, flat)
    } else if cells().all(|v| v.is_boolean()) {
        let flat: Vec<bool> = cells().filter_map(|v| v.as_bool()).collect();
        write_2d_dataset::<bool>(group, name, n_rows, n_cols, flat)
    } else {
        Ok(())
    }
}

/// Create and write a rectangular `n_rows × n_cols` HDF5 dataset from a
/// row-major flattened buffer. Shared by the `write_uns_2d_array` type arms;
/// mirrors the `ndarray::Array2::from_shape_vec` idiom in `write_obsm_entry`.
fn write_2d_dataset<T: hdf5::H5Type>(
    group: &hdf5::Group,
    name: &str,
    n_rows: usize,
    n_cols: usize,
    flat: Vec<T>,
) -> Result<(), ConvertError> {
    let nd = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat)
        .map_err(|e| ConvertError::Other(format!("ndarray shape error: {e}")))?;
    group
        .new_dataset::<T>()
        .shape([n_rows, n_cols])
        .create(name)?
        .write(&nd)?;
    Ok(())
}

#[cfg(test)]
mod pairwise_tests {
    use super::*;
    use arrow::array::{Float32Array, Int32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    /// Build a COO pairwise `RecordBatch` (`row`/`col` Int32, `data` f32 +
    /// `n_rows`/`n_cols` metadata) like the obsp/varp readers emit.
    fn coo(n: usize, triples: &[(i32, i32, f32)]) -> RecordBatch {
        let rows: Vec<i32> = triples.iter().map(|t| t.0).collect();
        let cols: Vec<i32> = triples.iter().map(|t| t.1).collect();
        let data: Vec<f32> = triples.iter().map(|t| t.2).collect();
        let schema = Schema::new_with_metadata(
            vec![
                Field::new("row", DataType::Int32, false),
                Field::new("col", DataType::Int32, false),
                Field::new("data", DataType::Float32, false),
            ],
            HashMap::from([
                ("n_rows".to_string(), n.to_string()),
                ("n_cols".to_string(), n.to_string()),
            ]),
        );
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(rows)),
                Arc::new(Int32Array::from(cols)),
                Arc::new(Float32Array::from(data)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn coo_to_csr_identity_no_mask() {
        // 4×4: (0,0)=1 (0,2)=2 (2,1)=3 (3,3)=4 — deliberately unsorted input.
        let batch = coo(4, &[(0, 2, 2.0), (3, 3, 4.0), (0, 0, 1.0), (2, 1, 3.0)]);
        let (indptr, indices, data, n) = coo_batch_to_csr(&batch, None).unwrap();
        assert_eq!(n, 4);
        assert_eq!(indptr, vec![0, 2, 2, 3, 4]);
        assert_eq!(indices, vec![0, 2, 1, 3]); // row0 cols sorted, then row2, row3
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn coo_to_csr_keep_mask_filters_both_axes() {
        // Drop index 1 on both axes: keep = [T,F,T,T] → remap 0→0, 2→1, 3→2.
        let batch = coo(4, &[(0, 0, 1.0), (0, 2, 2.0), (2, 1, 3.0), (3, 3, 4.0)]);
        let keep = [true, false, true, true];
        let (indptr, indices, data, n) = coo_batch_to_csr(&batch, Some(&keep)).unwrap();
        // (2,1) is dropped (col 1 removed); the rest remap into a 3×3 matrix.
        assert_eq!(n, 3);
        assert_eq!(indptr, vec![0, 2, 2, 3]);
        assert_eq!(indices, vec![0, 1, 2]); // row0: (0,0)->0,(0,2)->1 ; row2: (3,3)->2
        assert_eq!(data, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn coo_to_csr_out_of_bounds_coord_errors_no_mask() {
        // Malformed COO: a coordinate >= n_rows would index `indptr` out of
        // bounds. The no-mask path must reject it rather than panic.
        let batch = coo(3, &[(0, 0, 1.0), (5, 1, 2.0)]);
        let err = coo_batch_to_csr(&batch, None).unwrap_err();
        assert!(
            err.to_string().contains("out of bounds"),
            "expected out-of-bounds error, got: {err}"
        );
    }

    #[test]
    fn coo_to_csr_rejects_dimension_exceeding_i32() {
        // A pairwise axis wider than i32::MAX cannot be represented in the i32
        // CSR `indices`; reject rather than wrap. Empty triples so the guard
        // fires before the large `indptr` allocation (cheap test).
        let batch = coo(i32::MAX as usize + 1, &[]);
        let err = coo_batch_to_csr(&batch, None).unwrap_err();
        assert!(
            err.to_string().contains("i32"),
            "expected i32 index-limit error, got: {err}"
        );
    }
}

#[cfg(test)]
mod uns_envelope_tests {
    use super::*;

    /// A malformed or non-decodable `__scx_type__` envelope must fall back to
    /// the generic subgroup write (preserving the raw fields) rather than
    /// aborting the whole h5ad export — Antigravity #1 (bad shape) + Codex P2
    /// (unsupported byte order). `write_uns_value` must return `Ok` and create
    /// a subgroup for each case.
    #[test]
    fn malformed_envelopes_fall_back_to_subgroup() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("t.h5")).unwrap();
        let uns = file.create_group("uns").unwrap();

        let env = |extra: &[(&str, serde_json::Value)]| {
            let mut m = serde_json::Map::new();
            m.insert("__scx_type__".into(), "ndarray".into());
            m.insert("encoding".into(), "base64le".into());
            m.insert("dtype".into(), "<f8".into());
            m.insert("shape".into(), serde_json::json!([2]));
            m.insert("data".into(), serde_json::json!("AAAAAAAA8D8AAAAAAAAAQA==")); // [1.0, 2.0]
            for (k, v) in extra {
                m.insert((*k).into(), v.clone());
            }
            serde_json::Value::Object(m)
        };

        // (1) missing `data` → fallback.
        let mut no_data = env(&[]);
        no_data.as_object_mut().unwrap().remove("data");
        write_uns_value(&uns, "no_data", &no_data).unwrap();

        // (2) non-integer shape dim → fallback (no wrong-rank dataset).
        write_uns_value(
            &uns,
            "bad_shape",
            &env(&[("shape", serde_json::json!([2, null]))]),
        )
        .unwrap();

        // (3) big-endian byte order → fallback (no abort, no byte-swap misread).
        write_uns_value(&uns, "big_endian", &env(&[("dtype", ">f8".into())])).unwrap();

        // Each fell back to a subgroup that preserved the raw envelope fields.
        for key in ["no_data", "bad_shape", "big_endian"] {
            let g = uns
                .group(key)
                .unwrap_or_else(|_| panic!("'{key}' should fall back to a subgroup"));
            assert!(
                g.dataset("dtype").is_ok(),
                "'{key}' fallback should preserve the raw envelope fields"
            );
        }

        // A well-formed little-endian envelope still writes a real dataset
        // (not a subgroup), confirming the fallback is scoped to the bad cases.
        write_uns_value(&uns, "good", &env(&[])).unwrap();
        assert!(
            uns.dataset("good").is_ok(),
            "valid envelope should write a dataset"
        );
        assert!(
            uns.group("good").is_err(),
            "valid envelope must not be a subgroup"
        );
    }
}
