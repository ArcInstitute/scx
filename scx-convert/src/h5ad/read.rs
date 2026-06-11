// Read h5ad file components: X matrix, obs, var, obsm, uns, layers

use std::collections::HashMap;

use arrow::array::{
    ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use hdf5::types::TypeDescriptor;
use std::sync::Arc;

use super::csc_transpose::csc_to_csr;
use crate::detect::{detect_matrix_format, detect_matrix_format_at, MatrixFormat};
use crate::pipeline::ConvertError;
use crate::warnings::{ConvertWarning, WarningSink};
use crate::CATEGORICAL_ORDERED_KEY;

/// CSR matrix arrays + shape: (indptr, indices, data, n_obs, n_vars)
type CsrArrays = (Vec<i64>, Vec<i32>, Vec<f32>, usize, usize);

/// Path-based wrapper around [`read_h5ad_x_shape`]. Opens the file and
/// returns `(n_obs, n_vars, x_format)`; lets consumers that don't link
/// the `hdf5` crate directly (e.g. pyscx) check shape without taking on
/// the dep.
pub fn read_h5ad_x_shape_from_path(
    path: &std::path::Path,
    sink: &mut WarningSink,
) -> Result<(usize, usize, &'static str), ConvertError> {
    let file = hdf5::File::open(path)?;
    read_h5ad_x_shape(&file, sink)
}

/// Bundle of metadata returned by [`read_h5ad_metadata_from_path`]. obs
/// and var come back as Arrow RecordBatches (caller picks the
/// presentation); uns as a JSON tree or `None` when the file has no
/// `/uns` group.
pub struct H5adMetadataParts {
    pub obs: RecordBatch,
    pub var: RecordBatch,
    pub uns: Option<serde_json::Value>,
    pub n_obs: usize,
    pub n_vars: usize,
    pub x_format: &'static str,
}

/// Single-shot pure-Rust h5ad metadata read: opens the file, reads
/// obs / var / uns / X shape, and returns everything as Arrow + JSON.
/// X data, obsm, varm, obsp, varp, and layers are not touched. Used by
/// `pyscx.read_h5ad_metadata` to avoid the obsm-materialisation OOM
/// that `anndata.read_h5ad(path, backed="r")` triggers.
pub fn read_h5ad_metadata_from_path(
    path: &std::path::Path,
    strict_uns: bool,
    sink: &mut WarningSink,
) -> Result<H5adMetadataParts, ConvertError> {
    let file = hdf5::File::open(path)?;
    let (n_obs, n_vars, x_format) = read_h5ad_x_shape(&file, sink)?;
    let obs = read_dataframe_group(&file, "obs", sink)?;
    let var = read_dataframe_group(&file, "var", sink)?;
    let uns = if file.group("uns").is_ok() {
        Some(read_uns(&file, strict_uns, sink)?)
    } else {
        None
    };
    Ok(H5adMetadataParts {
        obs,
        var,
        uns,
        n_obs,
        n_vars,
        x_format,
    })
}

/// Read just the `/X` shape and storage format from an h5ad file without
/// loading any matrix data. Used by lightweight metadata readers
/// (e.g. pyscx.read_h5ad_metadata) that need `(n_obs, n_vars)` to validate
/// caller-supplied overrides without paying for an obsm-materialising
/// anndata.read_h5ad call.
///
/// Returns `(n_obs, n_vars, x_format)` where `x_format` is the lowercase
/// string `"csr"`, `"csc"`, or `"dense"`. Errors with
/// `ConvertError::FormatMismatch` for non-h5ad inputs.
pub fn read_h5ad_x_shape(
    file: &hdf5::File,
    sink: &mut WarningSink,
) -> Result<(usize, usize, &'static str), ConvertError> {
    let matrix_format = detect_matrix_format(file, sink)?;
    let (n_obs, n_vars) = match matrix_format {
        MatrixFormat::Csr | MatrixFormat::Csc => {
            let group = file.group("X")?;
            let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
            if shape.len() != 2 {
                return Err(ConvertError::Other(format!(
                    "X group shape attribute must have length 2, got {}",
                    shape.len()
                )));
            }
            (shape[0] as usize, shape[1] as usize)
        }
        MatrixFormat::Dense => {
            let ds = file.dataset("X")?;
            let shape = ds.shape();
            if shape.len() != 2 {
                return Err(ConvertError::Other(format!(
                    "dense X must be 2D, got {}-D",
                    shape.len()
                )));
            }
            (shape[0], shape[1])
        }
    };
    let fmt_str = match matrix_format {
        MatrixFormat::Csr => "csr",
        MatrixFormat::Csc => "csc",
        MatrixFormat::Dense => "dense",
    };
    Ok((n_obs, n_vars, fmt_str))
}

/// Read the X matrix from an h5ad file, returning CSR arrays and shape.
pub fn read_x_matrix(file: &hdf5::File, format: MatrixFormat) -> Result<CsrArrays, ConvertError> {
    read_x_matrix_at(file, "X", format)
}

/// Read a sparse / dense matrix at an arbitrary group path (used by
/// the h5mu pipeline for per-modality X at `mod/{name}/X`).
pub fn read_x_matrix_at(
    file: &hdf5::File,
    path: &str,
    format: MatrixFormat,
) -> Result<CsrArrays, ConvertError> {
    match format {
        MatrixFormat::Csr => read_sparse_matrix(file, path, false),
        MatrixFormat::Csc => read_sparse_matrix(file, path, true),
        MatrixFormat::Dense => read_dense_matrix(file, path),
    }
}

/// Read the AnnData `/raw` group (`raw/X` + `raw/var`) if present.
///
/// Returns the raw CSR arrays (always obs×raw_n_vars, canonicalized) and
/// the raw var `RecordBatch`, or `None` when the file has no usable
/// `/raw/X`. `raw.X` has its OWN var axis — `raw_n_vars` is typically
/// larger than `n_vars` because `.raw` is captured before HVG subsetting
/// — so it is stored as a dedicated raw section family rather than a
/// layer. Reuses the same CSR/CSC/dense detectors and readers as `/X`.
/// The caller asserts `raw.n_obs == n_obs` (raw shares the obs axis).
pub fn read_raw_group(
    file: &hdf5::File,
    sink: &mut WarningSink,
) -> Result<Option<(CsrArrays, RecordBatch)>, ConvertError> {
    // No `/raw` group, or no `/raw/X` payload → nothing to ingest.
    if file.group("raw").is_err() {
        return Ok(None);
    }
    if file.group("raw/X").is_err() && file.dataset("raw/X").is_err() {
        return Ok(None);
    }

    let format = detect_matrix_format_at(file, "raw/X", sink)?;
    let (indptr, indices, mut data, n_obs, raw_n_vars) = read_x_matrix_at(file, "raw/X", format)?;

    // Canonicalize defensively: the CSR reader drops explicit zeros but
    // does not sort/dedup, so a messy raw CSR (unsorted columns, dup
    // coords) is made canonical before it is sharded and encoded. Mirrors
    // the X CSC path. `canonicalize_csr` short-circuits on already-canonical
    // input (the common case for a scipy-written raw matrix).
    let mut u_indptr: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
    let mut u_indices: Vec<u32> = indices.iter().map(|&v| v as u32).collect();
    scx_sparse::canonicalize_csr(&mut u_indptr, &mut u_indices, &mut data);
    let indptr: Vec<i64> = u_indptr.iter().map(|&v| v as i64).collect();
    let indices: Vec<i32> = u_indices.iter().map(|&v| v as i32).collect();

    let raw_var = read_dataframe_group(file, "raw/var", sink)?;

    Ok(Some(((indptr, indices, data, n_obs, raw_n_vars), raw_var)))
}

/// Read a sparse group's required 2-D `shape` attribute as
/// `(n_obs, n_vars)`. Errors (rather than panicking) on a missing or
/// non-length-2 attribute. Shared by the eager and streaming X readers
/// and the per-layer reader so the shape parse lives in one place.
pub(crate) fn read_shape_2d(
    group: &hdf5::Group,
    path: &str,
) -> Result<(usize, usize), ConvertError> {
    let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "expected 2D shape attr on '{path}', got {}-D",
            shape.len()
        )));
    }
    Ok((shape[0] as usize, shape[1] as usize))
}

/// Read a sparse matrix group (CSR or CSC) and return as CSR.
fn read_sparse_matrix(
    file: &hdf5::File,
    group_name: &str,
    is_csc: bool,
) -> Result<CsrArrays, ConvertError> {
    let group = file.group(group_name)?;

    // Read shape from group attribute
    let (n_obs, n_vars) = read_shape_2d(&group, group_name)?;

    // Read indptr with runtime dtype handling
    let indptr_ds = group.dataset("indptr")?;
    let indptr = read_i64_dataset(&indptr_ds)?;

    // Read indices
    let indices_ds = group.dataset("indices")?;
    let indices = read_i32_dataset(&indices_ds)?;

    // Read data
    let data_ds = group.dataset("data")?;
    let data = read_f32_dataset(&data_ds)?;

    if is_csc {
        // CSC shape is [n_obs, n_vars] but indptr length = n_vars + 1.
        // The transpose output is row-major and column-sorted; canonicalize
        // it through the shared scx-sparse entry point so a messy source CSC
        // (duplicate row indices within a column → duplicate columns within a
        // row) is summed, not passed through, and explicit zeros are dropped.
        // The i64/i32 → u64/u32 casts are lossless (a valid CSR has
        // non-negative indptr and `0 <= col < n_vars`).
        let (csr_indptr, csr_indices, csr_data) =
            csc_to_csr(&indptr, &indices, &data, n_obs, n_vars)?;
        let mut u_indptr: Vec<u64> = csr_indptr.iter().map(|&v| v as u64).collect();
        let mut u_indices: Vec<u32> = csr_indices.iter().map(|&v| v as u32).collect();
        let mut u_data = csr_data;
        scx_sparse::canonicalize_csr(&mut u_indptr, &mut u_indices, &mut u_data);
        let csr_indptr: Vec<i64> = u_indptr.iter().map(|&v| v as i64).collect();
        let csr_indices: Vec<i32> = u_indices.iter().map(|&v| v as i32).collect();
        Ok((csr_indptr, csr_indices, u_data, n_obs, n_vars))
    } else {
        // C1: a CSC matrix misdetected as CSR has indptr length n_vars+1, not
        // n_obs+1, which would index out of bounds in drop_explicit_zeros (a
        // panic when n_vars < n_obs) or silently transpose. The shared
        // classifier refines the diagnostic; the gate is the CSR length
        // invariant (square matrices satisfy n_obs+1 and are valid here).
        if indptr.len() != n_obs + 1 {
            let hint = if scx_sparse::validate_sparse_layout((n_obs, n_vars), indptr.len(), None)
                == scx_sparse::SparseLayout::Csc
            {
                "this is a CSC matrix missing its encoding-type attribute"
            } else {
                "unexpected indptr length for the declared shape"
            };
            return Err(ConvertError::Other(format!(
                "CSR matrix '{group_name}' has indptr length {} but expected n_obs+1 = {} \
                 (shape [{n_obs}, {n_vars}]); {hint}",
                indptr.len(),
                n_obs + 1
            )));
        }
        Ok(drop_explicit_zeros(indptr, indices, data, n_obs, n_vars))
    }
}

/// Read a dense X dataset and convert to CSR.
fn read_dense_matrix(file: &hdf5::File, dataset_name: &str) -> Result<CsrArrays, ConvertError> {
    let ds = file.dataset(dataset_name)?;
    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "dense matrix '{dataset_name}' must be 2D, got {}-D",
            shape.len()
        )));
    }
    let n_obs = shape[0];
    let n_vars = shape[1];

    // Read 2D dataset as ndarray then flatten to Vec<f32>.
    // ndarray 0.16 renamed `into_raw_vec` → `into_raw_vec_and_offset`,
    // which returns `(Vec<T>, Option<usize>)`. The offset is `None`
    // here because `read_2d` returns an owned, contiguous Array2 that
    // hasn't been view-sliced.
    let nd: ndarray::Array2<f32> = ds.read_2d()?;
    let (flat, _offset) = nd.into_raw_vec_and_offset();
    let csr = scx_sparse::dense_to_csr(&flat, n_obs, n_vars)
        .map_err(|e| ConvertError::Other(format!("CSR conversion error: {e}")))?;

    Ok((csr.indptr, csr.indices, csr.data, n_obs, n_vars))
}

/// Per-target conversion policy for [`read_dataset_as`].
///
/// The three former typed readers (`read_i64_dataset` / `read_i32_dataset`
/// / `read_f32_dataset`) shared an identical dtype-dispatch skeleton and
/// differed only in how each *source* value maps onto the *target* type.
/// That per-target rule now lives here: every integer width except `u64`
/// is widened losslessly to `i64` and routed through [`Self::from_i64`];
/// `u64` (the one width that can exceed `i64::MAX`) goes through
/// [`Self::from_u64`]; floats go through [`Self::from_f64`] (rejected by
/// the integer targets). The contract per impl is byte-for-byte the same
/// as the readers it replaces.
pub(crate) trait HdfNumericTarget: Sized {
    /// Name used in `UnsupportedDtype` / `IndexOverflow` messages.
    const NAME: &'static str;
    /// Convert a signed source value (all integer widths except `u64`).
    fn from_i64(v: i64, path: &str, src: &'static str) -> Result<Self, ConvertError>;
    /// Convert an unsigned 64-bit source value.
    fn from_u64(v: u64, path: &str, src: &'static str) -> Result<Self, ConvertError>;
    /// Convert a floating-point source value.
    fn from_f64(v: f64, path: &str, src: &'static str) -> Result<Self, ConvertError>;
}

#[inline]
fn index_overflow<T>(
    path: &str,
    src: &'static str,
    target: &'static str,
    value: impl std::fmt::Display,
) -> Result<T, ConvertError> {
    Err(ConvertError::IndexOverflow {
        path: path.to_string(),
        source_dtype: src,
        target,
        value: value.to_string(),
    })
}

#[inline]
fn float_rejected<T>(path: &str, target: &'static str) -> Result<T, ConvertError> {
    Err(ConvertError::UnsupportedDtype(format!(
        "dataset '{path}': float dtype cannot be read as {target}"
    )))
}

impl HdfNumericTarget for i64 {
    const NAME: &'static str = "i64";
    fn from_i64(v: i64, _path: &str, _src: &'static str) -> Result<Self, ConvertError> {
        // Every signed/narrow-unsigned source widens losslessly to i64.
        Ok(v)
    }
    fn from_u64(v: u64, path: &str, src: &'static str) -> Result<Self, ConvertError> {
        // u64 > i64::MAX would silently truncate CSR `indptr` — fail loud.
        if v > i64::MAX as u64 {
            return index_overflow(path, src, "i64", v);
        }
        Ok(v as i64)
    }
    fn from_f64(_v: f64, path: &str, _src: &'static str) -> Result<Self, ConvertError> {
        float_rejected(path, "i64")
    }
}

impl HdfNumericTarget for i32 {
    const NAME: &'static str = "i32";
    fn from_i64(v: i64, path: &str, src: &'static str) -> Result<Self, ConvertError> {
        // Narrow widths (i8/i16/u8/u16) always fit; i64/u32 are range-checked.
        if v < i32::MIN as i64 || v > i32::MAX as i64 {
            return index_overflow(path, src, "i32", v);
        }
        Ok(v as i32)
    }
    fn from_u64(v: u64, path: &str, src: &'static str) -> Result<Self, ConvertError> {
        if v > i32::MAX as u64 {
            return index_overflow(path, src, "i32", v);
        }
        Ok(v as i32)
    }
    fn from_f64(_v: f64, path: &str, _src: &'static str) -> Result<Self, ConvertError> {
        float_rejected(path, "i32")
    }
}

impl HdfNumericTarget for f32 {
    const NAME: &'static str = "f32";
    fn from_i64(v: i64, _path: &str, _src: &'static str) -> Result<Self, ConvertError> {
        // Casts from i64 may lose precision above 2^24 — documented behaviour.
        Ok(v as f32)
    }
    fn from_u64(v: u64, _path: &str, _src: &'static str) -> Result<Self, ConvertError> {
        Ok(v as f32)
    }
    fn from_f64(v: f64, _path: &str, _src: &'static str) -> Result<Self, ConvertError> {
        // f32 source arrives widened to f64; the f32→f64→f32 round-trip is
        // lossless (incl. NaN/Inf), so f32 output is unchanged.
        Ok(v as f32)
    }
}

/// Read an HDF5 1-D dataset and convert every element to `T`, applying
/// `T`'s [`HdfNumericTarget`] policy. The dtype-dispatch skeleton lives
/// here once; per-target overflow / float-rejection rules live in the
/// trait impls. Replaces the former `read_{i64,i32,f32}_dataset` trio
/// (kept below as thin wrappers for call-site stability).
pub(crate) fn read_dataset_as<T: HdfNumericTarget>(
    ds: &hdf5::Dataset,
) -> Result<Vec<T>, ConvertError> {
    use crate::hdf_dtype::HdfNumericDtype;
    let path = ds.name();
    let desc = ds.dtype()?.to_descriptor()?;
    let dt = HdfNumericDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "dataset '{path}': dtype {desc:?} cannot be read as {}",
            T::NAME
        ))
    })?;
    let src = dt.name();
    match dt {
        HdfNumericDtype::I8 => ds
            .read_1d::<i8>()?
            .into_iter()
            .map(|v| T::from_i64(i64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::I16 => ds
            .read_1d::<i16>()?
            .into_iter()
            .map(|v| T::from_i64(i64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::I32 => ds
            .read_1d::<i32>()?
            .into_iter()
            .map(|v| T::from_i64(i64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::I64 => ds
            .read_1d::<i64>()?
            .into_iter()
            .map(|v| T::from_i64(v, &path, src))
            .collect(),
        HdfNumericDtype::U8 => ds
            .read_1d::<u8>()?
            .into_iter()
            .map(|v| T::from_i64(i64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::U16 => ds
            .read_1d::<u16>()?
            .into_iter()
            .map(|v| T::from_i64(i64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::U32 => ds
            .read_1d::<u32>()?
            .into_iter()
            .map(|v| T::from_i64(i64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::U64 => ds
            .read_1d::<u64>()?
            .into_iter()
            .map(|v| T::from_u64(v, &path, src))
            .collect(),
        HdfNumericDtype::F32 => ds
            .read_1d::<f32>()?
            .into_iter()
            .map(|v| T::from_f64(f64::from(v), &path, src))
            .collect(),
        HdfNumericDtype::F64 => ds
            .read_1d::<f64>()?
            .into_iter()
            .map(|v| T::from_f64(v, &path, src))
            .collect(),
    }
}

/// Read a dataset as `Vec<i64>` (CSR `indptr`). `u64` values exceeding
/// `i64::MAX` fail with [`ConvertError::IndexOverflow`]; float source
/// dtypes are rejected. Thin wrapper over [`read_dataset_as`].
pub(crate) fn read_i64_dataset(ds: &hdf5::Dataset) -> Result<Vec<i64>, ConvertError> {
    read_dataset_as::<i64>(ds)
}

/// Read a dataset as `Vec<i32>` (CSR `indices`). Narrowing casts from
/// `i64` / `u32` / `u64` are range-checked (overflow →
/// [`ConvertError::IndexOverflow`]); floats are rejected. Thin wrapper
/// over [`read_dataset_as`].
pub(crate) fn read_i32_dataset(ds: &hdf5::Dataset) -> Result<Vec<i32>, ConvertError> {
    read_dataset_as::<i32>(ds)
}

/// Read a dataset as `Vec<f32>` (CSR `data`). Accepts every numeric
/// width; `i64` / `u64` casts may lose precision above 2^24 (documented
/// behaviour). Thin wrapper over [`read_dataset_as`].
pub(crate) fn read_f32_dataset(ds: &hdf5::Dataset) -> Result<Vec<f32>, ConvertError> {
    read_dataset_as::<f32>(ds)
}

/// Read a DataFrame group (obs or var) from an h5ad file as an Arrow RecordBatch.
///
/// Columns that cannot be read (unsupported encoding-type, read error) are
/// skipped and surfaced via `sink.emit(ConvertWarning::SkippedColumn { .. })`
/// so the loss reaches the Python `warnings.warn` channel and the per-category
/// provenance counter — never silently dropped.
pub fn read_dataframe_group(
    file: &hdf5::File,
    group_name: &str,
    sink: &mut WarningSink,
) -> Result<RecordBatch, ConvertError> {
    let group = file.group(group_name)?;

    // Get the index column name from the _index attribute
    let index_col_name: Option<String> = group
        .attr("_index")
        .ok()
        .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.to_string());

    // Use column-order attribute if present (h5ad spec), otherwise fall back to member_names.
    //
    // pandas / anndata write an *empty* `column-order` for dataframes
    // with no columns (only the index) as a length-0 `float64` array
    // (numpy's default empty-array dtype) — see e.g. pbmc10k.h5ad's
    // `/obs/@column-order`. hdf5-rust has no f64 → VarLenUnicode
    // conversion kernel, so reading the typed payload would crash with
    // the opaque `HDF5 error: no conversion paths found`. Short-circuit
    // the empty case before the typed read.
    let member_names: Vec<String> = if let Ok(attr) = group.attr("column-order") {
        if attr.shape().iter().product::<usize>() == 0 {
            Vec::new()
        } else {
            let ordered: Vec<hdf5::types::VarLenUnicode> = attr.read_1d()?.to_vec();
            ordered.iter().map(|s| s.to_string()).collect()
        }
    } else {
        group.member_names()?
    };

    let mut fields: Vec<Field> = Vec::new();
    let mut arrays: Vec<ArrayRef> = Vec::new();

    for name in &member_names {
        // Skip internal members
        if name.starts_with("__") {
            continue;
        }

        // Dataset form: covers numeric columns, strings, booleans,
        // and the legacy attribute-form categorical (codes dataset
        // + `categories` attribute) handled by
        // [`read_categorical_column`].
        if let Ok(ds) = group.dataset(name) {
            match read_column_to_arrow(&group, &ds, name) {
                Ok((field, array)) => {
                    fields.push(field);
                    arrays.push(array);
                }
                Err(e) => {
                    sink.emit(ConvertWarning::SkippedColumn {
                        group: group_name.to_string(),
                        name: name.clone(),
                        reason: e.to_string(),
                    });
                }
            }
            continue;
        }

        // Group form: modern anndata column encodings — categorical
        // (encoding-version 0.2.0, `codes` + `categories` datasets)
        // or nullable-boolean (encoding-version 0.1.0, `values` +
        // `mask` datasets). Both bypass HDF5's 64 KB attribute /
        // header limits and are anndata's canonical reading shape.
        if let Ok(subgroup) = group.group(name) {
            let enc = subgroup
                .attr("encoding-type")
                .ok()
                .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                .map(|v| v.to_string())
                .unwrap_or_default();
            match enc.as_str() {
                "categorical" => {
                    match read_categorical_group(&subgroup, name) {
                        Ok((field, array)) => {
                            fields.push(field);
                            arrays.push(array);
                        }
                        Err(e) => {
                            sink.emit(ConvertWarning::SkippedColumn {
                                group: group_name.to_string(),
                                name: name.clone(),
                                reason: format!("categorical group: {e}"),
                            });
                        }
                    }
                    continue;
                }
                "nullable-boolean" => {
                    match read_nullable_boolean_group(&subgroup, name) {
                        Ok((field, array)) => {
                            fields.push(field);
                            arrays.push(array);
                        }
                        Err(e) => {
                            sink.emit(ConvertWarning::SkippedColumn {
                                group: group_name.to_string(),
                                name: name.clone(),
                                reason: format!("nullable-boolean group: {e}"),
                            });
                        }
                    }
                    continue;
                }
                "nullable-integer" => {
                    match read_nullable_integer_group(&subgroup, name) {
                        Ok((field, array)) => {
                            fields.push(field);
                            arrays.push(array);
                        }
                        Err(e) => {
                            sink.emit(ConvertWarning::SkippedColumn {
                                group: group_name.to_string(),
                                name: name.clone(),
                                reason: format!("nullable-integer group: {e}"),
                            });
                        }
                    }
                    continue;
                }
                "nullable-float" => {
                    match read_nullable_float_group(&subgroup, name) {
                        Ok((field, array)) => {
                            fields.push(field);
                            arrays.push(array);
                        }
                        Err(e) => {
                            sink.emit(ConvertWarning::SkippedColumn {
                                group: group_name.to_string(),
                                name: name.clone(),
                                reason: format!("nullable-float group: {e}"),
                            });
                        }
                    }
                    continue;
                }
                "nullable-string-array" => {
                    match read_nullable_string_group(&subgroup, name) {
                        Ok((field, array)) => {
                            fields.push(field);
                            arrays.push(array);
                        }
                        Err(e) => {
                            sink.emit(ConvertWarning::SkippedColumn {
                                group: group_name.to_string(),
                                name: name.clone(),
                                reason: format!("nullable-string-array group: {e}"),
                            });
                        }
                    }
                    continue;
                }
                "" => {
                    // No `encoding-type` attribute: a genuine non-column
                    // nested group (e.g. a nested uns dict). anndata
                    // always stamps `encoding-type` on real dataframe
                    // columns, so leave these out silently.
                }
                other => {
                    // A column-order member carrying an `encoding-type`
                    // we don't decode (some future / exotic anndata
                    // encoding). Don't drop it silently — that's how the
                    // nullable-integer regression slipped through.
                    sink.emit(ConvertWarning::SkippedColumn {
                        group: group_name.to_string(),
                        name: name.clone(),
                        reason: format!("unsupported encoding-type '{other}'"),
                    });
                }
            }
        }
    }

    // Inject the pandas index column (referenced by the `_index` HDF5
    // attribute) into the RecordBatch. `column-order` excludes it by
    // anndata convention, so the loop above never visited it — yet
    // downstream consumers (`scx_format_io::pandas_index_columns`, used
    // by `pyscx.open(...).to_anndata()` and by
    // `scx-convert/src/h5ad/write.rs::write_dataframe_body`) rely on
    // the resulting schema's `pandas` metadata envelope to identify
    // which column is the index. Without this block, obs_names /
    // var_names silently default to integer-positional strings.
    //
    // Rename the literal `_index` (anndata's on-disk sentinel for an
    // unnamed pandas index) to `__index_level_0__` (pyarrow's
    // canonical name). The inverse rename lives in
    // `scx-convert/src/h5ad/write.rs::write_dataframe_body`'s
    // `Some("__index_level_0__") => "_index"` arm — together they
    // round-trip an unnamed pandas index byte-equivalent through SCX.
    let mut injected_index_field_name: Option<String> = None;
    if let Some(ref idx_name) = index_col_name {
        let on_arrow_name = if idx_name == "_index" {
            "__index_level_0__".to_string()
        } else {
            idx_name.clone()
        };
        if let Some(existing_pos) = fields.iter().position(|f| f.name() == idx_name) {
            // The index dataset already got picked up by the main loop
            // (e.g. when `column-order` is absent and we fell back to
            // `member_names`, which includes the `_index` dataset).
            // Rename it in-place so downstream consumers see the
            // canonical `__index_level_0__` instead of the on-disk
            // `_index` sentinel.
            if &on_arrow_name != idx_name {
                let old = fields[existing_pos].clone();
                fields[existing_pos] = Field::new(
                    on_arrow_name.clone(),
                    old.data_type().clone(),
                    old.is_nullable(),
                )
                .with_metadata(old.metadata().clone());
            }
            injected_index_field_name = Some(on_arrow_name);
        } else {
            let read_result: Option<Result<(Field, ArrayRef), ConvertError>> =
                if let Ok(ds) = group.dataset(idx_name) {
                    Some(read_column_to_arrow(&group, &ds, &on_arrow_name))
                } else if let Ok(subgroup) = group.group(idx_name) {
                    // The pandas index is unusual-but-legal as a
                    // categorical / nullable-boolean subgroup; preserve
                    // it the same way regular columns are handled.
                    let enc = subgroup
                        .attr("encoding-type")
                        .ok()
                        .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                        .map(|v| v.to_string())
                        .unwrap_or_default();
                    match enc.as_str() {
                        "categorical" => Some(read_categorical_group(&subgroup, &on_arrow_name)),
                        "nullable-boolean" => {
                            Some(read_nullable_boolean_group(&subgroup, &on_arrow_name))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
            if let Some(res) = read_result {
                let (field, array) = res?;
                fields.push(field);
                arrays.push(array);
                injected_index_field_name = Some(on_arrow_name);
            }
        }
    }

    if fields.is_empty() {
        // Truly empty dataframe — no columns and no readable index.
        // Preserve the historic single-column-of-utf8 shape so
        // downstream consumers that expect *something* don't crash.
        let schema = Schema::new(vec![Field::new("_index", DataType::Utf8, true)]);
        let empty_arr: ArrayRef = Arc::new(StringArray::from(Vec::<&str>::new()));
        return Ok(RecordBatch::try_new(Arc::new(schema), vec![empty_arr])?);
    }

    let mut metadata: HashMap<String, String> = HashMap::new();
    if let Some(ref idx_field_name) = injected_index_field_name {
        metadata.insert(
            "pandas".to_string(),
            serde_json::json!({"index_columns": [idx_field_name]}).to_string(),
        );
    }
    let schema = Schema::new(fields).with_metadata(metadata);
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

/// Read a single column dataset to an Arrow array.
fn read_column_to_arrow(
    group: &hdf5::Group,
    ds: &hdf5::Dataset,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    use crate::hdf_dtype::HdfNumericDtype;

    // Check for categorical encoding
    let is_categorical = ds
        .attr("encoding-type")
        .ok()
        .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
        .map(|v| v.as_str() == "categorical")
        .unwrap_or(false);

    if is_categorical {
        return read_categorical_column(group, ds, name);
    }

    let desc = ds.dtype()?.to_descriptor()?;

    // Non-numeric descriptors first — `HdfNumericDtype` only covers
    // integer / float widths. Boolean and string columns route here.
    match &desc {
        TypeDescriptor::Boolean => {
            let data: Vec<bool> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(BooleanArray::from(data));
            return Ok((Field::new(name, DataType::Boolean, true), array));
        }
        TypeDescriptor::VarLenUnicode
        | TypeDescriptor::VarLenAscii
        | TypeDescriptor::FixedUnicode(_)
        | TypeDescriptor::FixedAscii(_) => {
            let data: Vec<hdf5::types::VarLenUnicode> = ds.read_1d()?.to_vec();
            let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
            let array: ArrayRef = Arc::new(StringArray::from(
                strings.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ));
            return Ok((Field::new(name, DataType::Utf8, true), array));
        }
        _ => {}
    }

    let dt = HdfNumericDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!("column '{name}': unsupported HDF5 type: {desc:?}"))
    })?;

    match dt {
        HdfNumericDtype::I8 => {
            let data: Vec<i8> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(Int32Array::from(
                data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Int32, true), array))
        }
        HdfNumericDtype::I16 => {
            let data: Vec<i16> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(Int32Array::from(
                data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Int32, true), array))
        }
        HdfNumericDtype::I32 => {
            let data: Vec<i32> = ds.read_1d()?.to_vec();
            Ok((
                Field::new(name, DataType::Int32, true),
                Arc::new(Int32Array::from(data)),
            ))
        }
        HdfNumericDtype::I64 => {
            let data: Vec<i64> = ds.read_1d()?.to_vec();
            Ok((
                Field::new(name, DataType::Int64, true),
                Arc::new(Int64Array::from(data)),
            ))
        }
        // u8 columns may carry the anndata `boolean` encoding-type
        // attribute — preserved from the pre-refactor path.
        HdfNumericDtype::U8 => {
            let is_bool = ds
                .attr("encoding-type")
                .ok()
                .and_then(|attr| attr.read_scalar::<hdf5::types::VarLenUnicode>().ok())
                .map(|v| v.as_str() == "boolean")
                .unwrap_or(false);
            let data: Vec<u8> = ds.read_1d()?.to_vec();
            if is_bool {
                let array: ArrayRef = Arc::new(BooleanArray::from(
                    data.iter().map(|&v| v != 0).collect::<Vec<_>>(),
                ));
                Ok((Field::new(name, DataType::Boolean, true), array))
            } else {
                let array: ArrayRef = Arc::new(Int32Array::from(
                    data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
                ));
                Ok((Field::new(name, DataType::Int32, true), array))
            }
        }
        HdfNumericDtype::U16 => {
            let data: Vec<u16> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(Int32Array::from(
                data.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Int32, true), array))
        }
        HdfNumericDtype::U32 => {
            let data: Vec<u32> = ds.read_1d()?.to_vec();
            let array: ArrayRef = Arc::new(Int64Array::from(
                data.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Int64, true), array))
        }
        // u64 → i64 may overflow. Range-check loudly instead of
        // silently truncating — same precedent as `read_i64_dataset`
        // for CSR indptr.
        HdfNumericDtype::U64 => {
            let data: Vec<u64> = ds.read_1d()?.to_vec();
            if let Some(&v) = data.iter().find(|&&v| v > i64::MAX as u64) {
                return Err(ConvertError::IndexOverflow {
                    path: ds.name(),
                    source_dtype: dt.name(),
                    target: "i64",
                    value: v.to_string(),
                });
            }
            let array: ArrayRef = Arc::new(Int64Array::from(
                data.iter().map(|&v| v as i64).collect::<Vec<_>>(),
            ));
            Ok((Field::new(name, DataType::Int64, true), array))
        }
        HdfNumericDtype::F32 => {
            let data: Vec<f32> = ds.read_1d()?.to_vec();
            Ok((
                Field::new(name, DataType::Float32, true),
                Arc::new(Float32Array::from(data)),
            ))
        }
        HdfNumericDtype::F64 => {
            let data: Vec<f64> = ds.read_1d()?.to_vec();
            Ok((
                Field::new(name, DataType::Float64, true),
                Arc::new(Float64Array::from(data)),
            ))
        }
    }
}

/// Read a modern anndata categorical (encoding-version 0.2.0):
/// a subgroup containing `codes` and `categories` as separate
/// datasets plus an `ordered` attribute. The dataset form lets the
/// categories payload grow past HDF5's 64 KB object-header limit,
/// which the legacy attribute form (read by
/// [`read_categorical_column`]) cannot.
fn read_categorical_group(
    cat_group: &hdf5::Group,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    let codes_ds = cat_group.dataset("codes")?;
    let codes: Vec<i32> = read_i32_dataset(&codes_ds)?;

    let cats_ds = cat_group.dataset("categories")?;
    let cats_raw: Vec<hdf5::types::VarLenUnicode> = cats_ds.read_1d()?.to_vec();
    let categories: Vec<String> = cats_raw.iter().map(|s| s.to_string()).collect();

    let keys = Int32Array::from(
        codes
            .iter()
            .map(|&c| if c < 0 { None } else { Some(c) })
            .collect::<Vec<Option<i32>>>(),
    );
    let values = StringArray::from(categories.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values))?;

    // Carry the pandas `ordered` bit (anndata stores it as a scalar bool
    // attribute on the categorical group) in Arrow field metadata so the
    // h5ad writer can re-emit it; absent → false (pandas default).
    let ordered = cat_group
        .attr("ordered")
        .ok()
        .and_then(|a| a.read_scalar::<bool>().ok())
        .unwrap_or(false);
    let field = Field::new(
        name,
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        true,
    )
    .with_metadata(HashMap::from([(
        CATEGORICAL_ORDERED_KEY.to_string(),
        ordered.to_string(),
    )]));
    Ok((field, Arc::new(dict)))
}

/// Read anndata's `nullable-boolean` group form (encoding-version
/// 0.1.0): a subgroup with `values` (u8) and `mask` (u8) datasets.
/// `mask[i] == 1` marks the row as null; the resulting Arrow
/// `BooleanArray` carries the corresponding validity bit.
fn read_nullable_boolean_group(
    bool_group: &hdf5::Group,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    // anndata writes `values` and `mask` as native HDF5 boolean
    // dtype. hdf5-rust exposes the same via `bool` (truthful read +
    // write with H5T_NATIVE_HBOOL_8). Accept either bool or u8 on
    // read so older files written with the legacy u8 shape still
    // round-trip.
    let values: Vec<bool> = read_bool_or_u8(&bool_group.dataset("values")?)?;
    let mask: Vec<bool> = read_bool_or_u8(&bool_group.dataset("mask")?)?;
    if mask.len() != values.len() {
        return Err(ConvertError::Other(format!(
            "nullable-boolean '{name}': values len {} != mask len {}",
            values.len(),
            mask.len()
        )));
    }
    let arr = BooleanArray::from(
        values
            .iter()
            .zip(mask.iter())
            .map(|(v, m)| if *m { None } else { Some(*v) })
            .collect::<Vec<_>>(),
    );
    Ok((Field::new(name, DataType::Boolean, true), Arc::new(arr)))
}

/// Read anndata's `nullable-integer` group form: a subgroup with
/// `values` (integer dataset) and `mask` (bool/u8) datasets, written
/// for pandas nullable integer dtypes (`Int8`..`Int64`,
/// `UInt8`..`UInt64`). `mask[i] == true` marks the row as null. The
/// resulting Arrow array carries the corresponding validity bit; the
/// integer width is mapped to the same Arrow type
/// [`read_column_to_arrow`] uses for the equivalent plain column, so
/// downstream handling is identical to a non-nullable integer column.
fn read_nullable_integer_group(
    int_group: &hdf5::Group,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    use crate::hdf_dtype::HdfNumericDtype;

    let values_ds = int_group.dataset("values")?;
    let mask = read_bool_or_u8(&int_group.dataset("mask")?)?;
    let desc = values_ds.dtype()?.to_descriptor()?;
    let dt = HdfNumericDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "nullable-integer '{name}': unsupported values dtype: {desc:?}"
        ))
    })?;

    // Build `Vec<Option<T>>` from values + mask, validating lengths.
    macro_rules! opt_vec {
        ($read_ty:ty, $cast_ty:ty) => {{
            let values: Vec<$read_ty> = values_ds.read_1d()?.to_vec();
            if values.len() != mask.len() {
                return Err(ConvertError::Other(format!(
                    "nullable-integer '{name}': values len {} != mask len {}",
                    values.len(),
                    mask.len()
                )));
            }
            values
                .iter()
                .zip(mask.iter())
                .map(|(v, m)| if *m { None } else { Some(*v as $cast_ty) })
                .collect::<Vec<Option<$cast_ty>>>()
        }};
    }

    let (field, array): (Field, ArrayRef) = match dt {
        HdfNumericDtype::I8 => (
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(opt_vec!(i8, i32))),
        ),
        HdfNumericDtype::I16 => (
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(opt_vec!(i16, i32))),
        ),
        HdfNumericDtype::I32 => (
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(opt_vec!(i32, i32))),
        ),
        HdfNumericDtype::U8 => (
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(opt_vec!(u8, i32))),
        ),
        HdfNumericDtype::U16 => (
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(opt_vec!(u16, i32))),
        ),
        HdfNumericDtype::I64 => (
            Field::new(name, DataType::Int64, true),
            Arc::new(Int64Array::from(opt_vec!(i64, i64))),
        ),
        HdfNumericDtype::U32 => (
            Field::new(name, DataType::Int64, true),
            Arc::new(Int64Array::from(opt_vec!(u32, i64))),
        ),
        // u64 → i64 may overflow. Range-check loudly instead of silently
        // truncating — same precedent as the plain-column reader.
        HdfNumericDtype::U64 => {
            let values: Vec<u64> = values_ds.read_1d()?.to_vec();
            if values.len() != mask.len() {
                return Err(ConvertError::Other(format!(
                    "nullable-integer '{name}': values len {} != mask len {}",
                    values.len(),
                    mask.len()
                )));
            }
            if let Some(&v) = values
                .iter()
                .zip(mask.iter())
                .filter(|(_, m)| !**m)
                .map(|(v, _)| v)
                .find(|&&v| v > i64::MAX as u64)
            {
                return Err(ConvertError::IndexOverflow {
                    path: values_ds.name(),
                    source_dtype: dt.name(),
                    target: "i64",
                    value: v.to_string(),
                });
            }
            let opt = values
                .iter()
                .zip(mask.iter())
                .map(|(v, m)| if *m { None } else { Some(*v as i64) })
                .collect::<Vec<Option<i64>>>();
            (
                Field::new(name, DataType::Int64, true),
                Arc::new(Int64Array::from(opt)),
            )
        }
        HdfNumericDtype::F32 | HdfNumericDtype::F64 => {
            return Err(ConvertError::UnsupportedDtype(format!(
                "nullable-integer '{name}': values dtype is float ({}); expected integer",
                dt.name()
            )));
        }
    };
    Ok((field, array))
}

/// Read anndata's `nullable-float` group form: a subgroup with `values`
/// (float dataset) and `mask` (bool/u8) datasets, written for pandas
/// nullable float dtypes (`Float32` / `Float64`). `mask[i] == true`
/// marks the row as null. Mirrors [`read_nullable_integer_group`] for
/// floating-point widths.
fn read_nullable_float_group(
    float_group: &hdf5::Group,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    use crate::hdf_dtype::HdfNumericDtype;

    let values_ds = float_group.dataset("values")?;
    let mask = read_bool_or_u8(&float_group.dataset("mask")?)?;
    let desc = values_ds.dtype()?.to_descriptor()?;
    let dt = HdfNumericDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "nullable-float '{name}': unsupported values dtype: {desc:?}"
        ))
    })?;

    macro_rules! opt_vec {
        ($read_ty:ty) => {{
            let values: Vec<$read_ty> = values_ds.read_1d()?.to_vec();
            if values.len() != mask.len() {
                return Err(ConvertError::Other(format!(
                    "nullable-float '{name}': values len {} != mask len {}",
                    values.len(),
                    mask.len()
                )));
            }
            values
                .iter()
                .zip(mask.iter())
                .map(|(v, m)| if *m { None } else { Some(*v) })
                .collect::<Vec<Option<$read_ty>>>()
        }};
    }

    let (field, array): (Field, ArrayRef) = match dt {
        HdfNumericDtype::F32 => (
            Field::new(name, DataType::Float32, true),
            Arc::new(Float32Array::from(opt_vec!(f32))),
        ),
        HdfNumericDtype::F64 => (
            Field::new(name, DataType::Float64, true),
            Arc::new(Float64Array::from(opt_vec!(f64))),
        ),
        other => {
            return Err(ConvertError::UnsupportedDtype(format!(
                "nullable-float '{name}': values dtype is {}; expected float",
                other.name()
            )));
        }
    };
    Ok((field, array))
}

/// Read anndata's `nullable-string-array` group form (encoding-version
/// 0.1.0): a subgroup with a variable-length-UTF8 `values` dataset (null
/// positions filled with `""`) and a bool/u8 `mask` (`mask[i] == true` ⇔
/// null). Produces an Arrow `Utf8` array carrying the corresponding
/// validity bits, so it round-trips with the writer's
/// `nullable-string-array` output. Mirrors [`read_nullable_integer_group`].
fn read_nullable_string_group(
    str_group: &hdf5::Group,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    let values_ds = str_group.dataset("values")?;
    let mask = read_bool_or_u8(&str_group.dataset("mask")?)?;
    let values: Vec<hdf5::types::VarLenUnicode> = values_ds.read_1d()?.to_vec();
    if values.len() != mask.len() {
        return Err(ConvertError::Other(format!(
            "nullable-string-array '{name}': values len {} != mask len {}",
            values.len(),
            mask.len()
        )));
    }
    let strings: Vec<String> = values.iter().map(|s| s.to_string()).collect();
    let arr = StringArray::from(
        strings
            .iter()
            .zip(mask.iter())
            .map(|(v, m)| if *m { None } else { Some(v.as_str()) })
            .collect::<Vec<Option<&str>>>(),
    );
    Ok((Field::new(name, DataType::Utf8, true), Arc::new(arr)))
}

/// Read a 1D dataset that may be encoded as native HDF5 boolean
/// (`H5T_NATIVE_HBOOL_8`) or as u8. Used for `nullable-boolean`
/// values/mask datasets which different writers encode differently.
fn read_bool_or_u8(ds: &hdf5::Dataset) -> Result<Vec<bool>, ConvertError> {
    if let Ok(v) = ds.read_1d::<bool>() {
        return Ok(v.to_vec());
    }
    let v: Vec<u8> = ds.read_1d::<u8>()?.to_vec();
    Ok(v.into_iter().map(|x| x != 0).collect())
}

/// Read a categorical column (integer codes + string categories).
fn read_categorical_column(
    group: &hdf5::Group,
    ds: &hdf5::Dataset,
    name: &str,
) -> Result<(Field, ArrayRef), ConvertError> {
    // Read the integer codes
    let codes: Vec<i32> = read_i32_dataset(ds)?;

    // Try to find categories: first check "categories" attribute on the dataset
    let categories: Vec<String> = if let Ok(cats_attr) = ds.attr("categories") {
        // Categories stored as attribute
        let cats: Vec<hdf5::types::VarLenUnicode> = cats_attr.read_1d()?.to_vec();
        cats.iter().map(|s| s.to_string()).collect()
    } else if let Ok(cats_ds) = group.dataset(&format!("__categories/{name}")) {
        // Old-style: categories in __categories subgroup
        let cats: Vec<hdf5::types::VarLenUnicode> = cats_ds.read_1d()?.to_vec();
        cats.iter().map(|s| s.to_string()).collect()
    } else {
        // Fallback: treat codes as strings
        return Ok((
            Field::new(name, DataType::Int32, true),
            Arc::new(Int32Array::from(codes)),
        ));
    };

    // Build a DictionaryArray.
    // In h5ad, code = -1 means missing/NA. Convert to null entries.
    let keys = Int32Array::from(
        codes
            .iter()
            .map(|&c| if c < 0 { None } else { Some(c) })
            .collect::<Vec<Option<i32>>>(),
    );
    let values = StringArray::from(categories.iter().map(|s| s.as_str()).collect::<Vec<_>>());
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values))?;

    // Carry the `ordered` bit (legacy attr-form stores it on the dataset)
    // in Arrow field metadata; absent → false (pandas default).
    let ordered = ds
        .attr("ordered")
        .ok()
        .and_then(|a| a.read_scalar::<bool>().ok())
        .unwrap_or(false);
    let field = Field::new(
        name,
        DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
        true,
    )
    .with_metadata(HashMap::from([(
        CATEGORICAL_ORDERED_KEY.to_string(),
        ordered.to_string(),
    )]));

    Ok((field, Arc::new(dict)))
}

/// Remove explicit zeros from a CSR matrix.
///
/// Some h5ad files store explicit zeros in their sparse representation.
/// The SCX Rice codec requires values >= 1, so we must drop these entries.
fn drop_explicit_zeros(
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
    n_obs: usize,
    n_vars: usize,
) -> CsrArrays {
    // Fast path: no zeros present
    if data.iter().all(|&v| v != 0.0) {
        return (indptr, indices, data, n_obs, n_vars);
    }

    let mut new_indptr = Vec::with_capacity(indptr.len());
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();
    new_indptr.push(0i64);

    for row in 0..n_obs {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for i in start..end {
            if data[i] != 0.0 {
                new_indices.push(indices[i]);
                new_data.push(data[i]);
            }
        }
        new_indptr.push(new_data.len() as i64);
    }

    (new_indptr, new_indices, new_data, n_obs, n_vars)
}

/// Read obsm embeddings from h5ad file (root `/obsm`).
///
/// Kept for callers that need the non-streaming, fully-materialised
/// view (the streaming pipeline now goes through
/// [`list_dense_mapping_shapes`] + [`read_dense_mapping_shard`]). The
/// `_at` form is also exposed for h5mu per-modality paths.
#[allow(dead_code)]
pub fn read_obsm(
    file: &hdf5::File,
    sink: &mut WarningSink,
) -> Result<HashMap<String, RecordBatch>, ConvertError> {
    read_obsm_at(file, "obsm", sink)
}

/// Read varm embeddings from h5ad file (root `/varm`). See [`read_obsm`].
#[allow(dead_code)]
pub fn read_varm(
    file: &hdf5::File,
    sink: &mut WarningSink,
) -> Result<HashMap<String, RecordBatch>, ConvertError> {
    read_obsm_at(file, "varm", sink)
}

/// Read an obsm group at an arbitrary path (e.g. `mod/rna/obsm` for an
/// h5mu file's per-modality embeddings). Returns the same
/// `name -> RecordBatch` map as `read_obsm`. Missing groups return an
/// empty map rather than an error so callers don't need to special-case
/// modalities without obsm.
pub fn read_obsm_at(
    file: &hdf5::File,
    path: &str,
    sink: &mut WarningSink,
) -> Result<HashMap<String, RecordBatch>, ConvertError> {
    let mut result = HashMap::new();
    let obsm_group = match file.group(path) {
        Ok(g) => g,
        Err(_) => return Ok(result),
    };
    let member_names = obsm_group.member_names()?;
    for name in &member_names {
        match read_obsm_entry(&obsm_group, name) {
            Ok(batch) => {
                result.insert(name.clone(), batch);
            }
            Err(e) => {
                sink.emit(ConvertWarning::SkippedObsm {
                    name: format!("{path}/{name}"),
                    reason: e.to_string(),
                });
            }
        }
    }
    Ok(result)
}

fn read_obsm_entry(obsm_group: &hdf5::Group, name: &str) -> Result<RecordBatch, ConvertError> {
    let ds = obsm_group.dataset(name)?;
    let shape = ds.shape();

    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "obsm/{name} is not 2D (shape: {shape:?})"
        )));
    }

    let n_rows = shape[0];
    let n_cols = shape[1];

    // Read as f32 2D
    let flat: Vec<f32> = read_f32_dataset(&ds)?;

    // Build RecordBatch with numbered columns
    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n_cols);

    for col in 0..n_cols {
        let col_data: Vec<f32> = (0..n_rows).map(|row| flat[row * n_cols + col]).collect();
        fields.push(Field::new(format!("{col}"), DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from(col_data)));
    }

    let schema = Schema::new(fields);
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

// ---------------------------------------------------------------------------
// Streaming readers for obsm / varm / obsp / varp.
//
// The non-streaming readers above (`read_obsm`, `read_varm`) materialise the
// full dense matrix in one allocation before column-splitting, which dominates
// peak RSS for inputs with large embeddings (n_obs × k × 4 B per matrix). The
// helpers below expose a per-shard API the pipeline drives one row-range at a
// time, bounding peak memory to `shard_rows × k × 4 B` (dense) or
// `shard_rows × density × n_cols × 16 B` (sparse) per matrix.
// ---------------------------------------------------------------------------

/// Describes a dense obsm/varm dataset without reading any value bytes.
/// Returned by [`list_dense_mapping_shapes`] so the pipeline can pre-plan
/// the per-key shard schedule from h5py metadata only. The per-shard
/// read in [`read_dense_mapping_shard`] re-derives the column count from
/// the dataset shape, so only the row count is carried here.
#[derive(Debug, Clone)]
pub struct DenseMappingInfo {
    pub name: String,
    pub n_rows: usize,
}

/// Walk `<group_path>` in `file` and return each member's name + shape.
/// Members that are not 2D datasets are skipped silently. Missing group →
/// empty vec.
pub fn list_dense_mapping_shapes(
    file: &hdf5::File,
    group_path: &str,
) -> Result<Vec<DenseMappingInfo>, ConvertError> {
    let mut result = Vec::new();
    let group = match file.group(group_path) {
        Ok(g) => g,
        Err(_) => return Ok(result),
    };
    let names = group.member_names()?;
    for name in &names {
        let ds = match group.dataset(name) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let shape = ds.shape();
        if shape.len() != 2 {
            continue;
        }
        result.push(DenseMappingInfo {
            name: name.clone(),
            n_rows: shape[0],
        });
    }
    Ok(result)
}

/// Read a row-range `[row_start, row_end)` of `<group_path>/<name>` as
/// an Arrow `RecordBatch`. The returned batch has `n_cols` columns named
/// `"0", "1", ..., "{n_cols-1}"`, mirroring [`read_obsm_entry`].
///
/// Peak memory: `(row_end - row_start) × n_cols × 4 B` for the f32 slab
/// plus an equivalent amount during column transpose.
pub fn read_dense_mapping_shard(
    file: &hdf5::File,
    group_path: &str,
    name: &str,
    row_start: usize,
    row_end: usize,
) -> Result<RecordBatch, ConvertError> {
    let group = file.group(group_path)?;
    let ds = group.dataset(name)?;
    let shape = ds.shape();
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "{group_path}/{name} is not 2D (shape: {shape:?})"
        )));
    }
    let n_cols = shape[1];

    // Hyperslab read of the row range; widen to f32 if necessary.
    use super::dense_stream::{read_dense_slab_f32, DenseDtype};
    let desc = ds.dtype()?.to_descriptor()?;
    let dtype = DenseDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "{group_path}/{name}: dtype {desc:?} cannot be read as f32"
        ))
    })?;
    let flat: Vec<f32> = read_dense_slab_f32(&ds, dtype, row_start, row_end)?;

    let n_local = row_end - row_start;
    let mut fields = Vec::with_capacity(n_cols);
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(n_cols);
    for col in 0..n_cols {
        let col_data: Vec<f32> = (0..n_local).map(|row| flat[row * n_cols + col]).collect();
        fields.push(Field::new(format!("{col}"), DataType::Float32, false));
        arrays.push(Arc::new(Float32Array::from(col_data)));
    }
    let schema = Schema::new(fields);
    Ok(RecordBatch::try_new(Arc::new(schema), arrays)?)
}

/// Describes a sparse obsp/varp pairwise matrix without reading the
/// indptr/indices/data datasets. Carries the open `indptr` array so the
/// pipeline can chunk the per-row CSR slice ranges without rereading
/// `indptr` once per shard.
pub struct SparseMappingInfo {
    pub name: String,
    pub n_rows: usize,
    pub n_cols: usize,
    /// Eagerly-loaded indptr (length = n_rows + 1). `n_rows + 1` int64s
    /// — `(10⁶ + 1) × 8 B = 8 MB` for a million-row obsp, which is
    /// already in the noise relative to the per-shard nnz reads.
    pub indptr: Vec<i64>,
}

/// Walk `<group_path>` in `file` and return per-member shape + indptr.
/// Members that aren't CSR sparse groups are silently skipped.
pub fn list_sparse_mapping_shapes(
    file: &hdf5::File,
    group_path: &str,
) -> Result<Vec<SparseMappingInfo>, ConvertError> {
    let mut result = Vec::new();
    let parent = match file.group(group_path) {
        Ok(g) => g,
        Err(_) => return Ok(result),
    };
    let names = parent.member_names()?;
    for name in &names {
        let sub = match parent.group(name) {
            Ok(g) => g,
            Err(_) => continue,
        };
        // Only CSR is read here. A *square* CSC pairwise matrix has
        // `indptr.len() == n_cols + 1 == n_rows + 1`, so it would otherwise
        // slip past the indptr-length check below and be mis-read as CSR
        // (silently transposed). Skip any group whose `encoding-type` is
        // present and not `csr_matrix` — the caller surfaces the skip as a
        // `DroppedObsp` warning. Groups with no `encoding-type` attr fall
        // through to the structural checks (legacy / attr-less CSR).
        if let Some(enc) = sub
            .attr("encoding-type")
            .ok()
            .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
            .map(|v| v.to_string())
        {
            if enc != "csr_matrix" {
                continue;
            }
        }
        let shape_attr: Vec<i64> = match sub.attr("shape").and_then(|a| a.read_1d::<i64>()) {
            Ok(arr) => arr.to_vec(),
            Err(_) => continue,
        };
        if shape_attr.len() != 2 {
            continue;
        }
        let n_rows = shape_attr[0] as usize;
        let n_cols = shape_attr[1] as usize;
        let indptr_ds = match sub.dataset("indptr") {
            Ok(d) => d,
            Err(_) => continue,
        };
        let indptr = match read_i64_dataset(&indptr_ds) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if indptr.len() != n_rows + 1 {
            continue;
        }
        result.push(SparseMappingInfo {
            name: name.clone(),
            n_rows,
            n_cols,
            indptr,
        });
    }
    Ok(result)
}

/// Read a row-range `[row_start, row_end)` of `<group_path>/<name>` as
/// an Arrow `RecordBatch` in COO format (`row: Int32`, `col: Int32`,
/// `data: Float32` + schema metadata `n_rows` / `n_cols` of the logical
/// matrix). `row` values are global, not shard-local — the reader-side
/// concatenation does not need to apply an offset.
///
/// `info.indptr` must be the cached full-indptr from
/// [`list_sparse_mapping_shapes`] for this same key; the function only
/// reads the indices/data slices `[indptr[row_start], indptr[row_end])`.
pub fn read_sparse_mapping_shard(
    file: &hdf5::File,
    group_path: &str,
    info: &SparseMappingInfo,
    row_start: usize,
    row_end: usize,
) -> Result<RecordBatch, ConvertError> {
    use super::stream::{read_slice_f32, read_slice_i32};

    let parent = file.group(group_path)?;
    let sub = parent.group(&info.name)?;
    let nnz_start = info.indptr[row_start] as usize;
    let nnz_end = info.indptr[row_end] as usize;

    let cols = if nnz_end > nnz_start {
        let indices_ds = sub.dataset("indices")?;
        read_slice_i32(&indices_ds, nnz_start, nnz_end)?
    } else {
        Vec::new()
    };
    let data = if nnz_end > nnz_start {
        let data_ds = sub.dataset("data")?;
        read_slice_f32(&data_ds, nnz_start, nnz_end)?
    } else {
        Vec::new()
    };
    let mut rows: Vec<i32> = Vec::with_capacity(nnz_end - nnz_start);
    for r in row_start..row_end {
        let r_lo = info.indptr[r] as usize;
        let r_hi = info.indptr[r + 1] as usize;
        let r_i32 = i32::try_from(r).map_err(|_| {
            ConvertError::Other(format!(
                "{group_path}/{}: row index {r} exceeds i32::MAX (Arrow COO uses i32 row indices)",
                info.name
            ))
        })?;
        for _ in r_lo..r_hi {
            rows.push(r_i32);
        }
    }

    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), info.n_rows.to_string()),
            ("n_cols".to_string(), info.n_cols.to_string()),
        ]),
    ));
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(rows)),
            Arc::new(Int32Array::from(cols)),
            Arc::new(Float32Array::from(data)),
        ],
    )?)
}

/// Read uns (unstructured) section from h5ad file as JSON.
///
/// `strict_uns = false` (default) skips unsupported keys with a
/// [`ConvertWarning::SkippedUnsKey`] emission. `strict_uns = true`
/// aborts on the first unsupported key.
pub fn read_uns(
    file: &hdf5::File,
    strict_uns: bool,
    sink: &mut WarningSink,
) -> Result<serde_json::Value, ConvertError> {
    let uns_group = file.group("uns")?;
    let member_names = uns_group.member_names()?;

    let mut map = serde_json::Map::new();

    for name in &member_names {
        match read_uns_entry(&uns_group, name, strict_uns, sink) {
            Ok(value) => {
                map.insert(name.clone(), value);
            }
            Err(e) => {
                if strict_uns {
                    return Err(e);
                }
                sink.emit(ConvertWarning::SkippedUnsKey {
                    key: name.clone(),
                    reason: e.to_string(),
                });
            }
        }
    }

    Ok(serde_json::Value::Object(map))
}

fn read_uns_entry(
    group: &hdf5::Group,
    name: &str,
    strict_uns: bool,
    sink: &mut WarningSink,
) -> Result<serde_json::Value, ConvertError> {
    // Try reading as dataset first
    if let Ok(ds) = group.dataset(name) {
        let desc = ds.dtype()?.to_descriptor()?;
        let shape = ds.shape();

        // Scalar
        if shape.is_empty() || (shape.len() == 1 && shape[0] == 1) {
            return match desc {
                TypeDescriptor::Integer(_) => {
                    let v: i64 = ds.read_scalar()?;
                    Ok(serde_json::Value::Number(v.into()))
                }
                TypeDescriptor::Unsigned(_) => {
                    let v: u64 = ds.read_scalar()?;
                    Ok(serde_json::Value::Number(v.into()))
                }
                TypeDescriptor::Float(_) => {
                    let v: f64 = ds.read_scalar()?;
                    Ok(serde_json::json!(v))
                }
                TypeDescriptor::Boolean => {
                    let v: bool = ds.read_scalar()?;
                    Ok(serde_json::Value::Bool(v))
                }
                TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
                    let v: hdf5::types::VarLenUnicode = ds.read_scalar()?;
                    Ok(serde_json::Value::String(v.to_string()))
                }
                TypeDescriptor::FixedUnicode(_) | TypeDescriptor::FixedAscii(_) => {
                    let v: hdf5::types::VarLenUnicode = ds.read_scalar()?;
                    Ok(serde_json::Value::String(v.to_string()))
                }
                _ => Err(ConvertError::Other(format!(
                    "unsupported uns scalar type: {desc:?}"
                ))),
            };
        }

        // 1D array
        if shape.len() == 1 {
            return match desc {
                TypeDescriptor::Integer(_) => {
                    let data: Vec<i64> = ds.read_1d()?.to_vec();
                    Ok(serde_json::json!(data))
                }
                // C3: anndata stores uint/bool uns vectors too — the scalar arm
                // already handles these types, so the 1-D arm must as well or
                // they get dropped (lenient) / abort (strict_uns).
                TypeDescriptor::Unsigned(_) => {
                    let data: Vec<u64> = ds.read_1d()?.to_vec();
                    Ok(serde_json::json!(data))
                }
                TypeDescriptor::Float(_) => {
                    let data: Vec<f64> = ds.read_1d()?.to_vec();
                    Ok(serde_json::json!(data))
                }
                TypeDescriptor::Boolean => {
                    let data: Vec<bool> = ds.read_1d()?.to_vec();
                    Ok(serde_json::json!(data))
                }
                TypeDescriptor::VarLenUnicode | TypeDescriptor::VarLenAscii => {
                    let data: Vec<hdf5::types::VarLenUnicode> = ds.read_1d()?.to_vec();
                    let strings: Vec<String> = data.iter().map(|s| s.to_string()).collect();
                    Ok(serde_json::json!(strings))
                }
                _ => Err(ConvertError::Other(format!(
                    "unsupported uns array type: {desc:?}"
                ))),
            };
        }

        // C4: 2-D numeric arrays (e.g. color/contrast matrices) round-trip as
        // nested JSON arrays. Higher ranks and non-numeric 2-D arrays fall
        // through to the error → skip/warn (lenient) or abort (strict_uns) path.
        if shape.len() == 2 {
            // Row-major nested JSON arrays (`[[..], [..]]`). Built from the
            // concrete element type so `serde_json::json!` resolves Serialize
            // through serde_json itself (scx-convert has no direct `serde` dep).
            return match desc {
                TypeDescriptor::Integer(_) => {
                    let rows: Vec<Vec<i64>> = ds
                        .read_2d::<i64>()?
                        .outer_iter()
                        .map(|r| r.to_vec())
                        .collect();
                    Ok(serde_json::json!(rows))
                }
                TypeDescriptor::Unsigned(_) => {
                    let rows: Vec<Vec<u64>> = ds
                        .read_2d::<u64>()?
                        .outer_iter()
                        .map(|r| r.to_vec())
                        .collect();
                    Ok(serde_json::json!(rows))
                }
                TypeDescriptor::Float(_) => {
                    let rows: Vec<Vec<f64>> = ds
                        .read_2d::<f64>()?
                        .outer_iter()
                        .map(|r| r.to_vec())
                        .collect();
                    Ok(serde_json::json!(rows))
                }
                TypeDescriptor::Boolean => {
                    let rows: Vec<Vec<bool>> = ds
                        .read_2d::<bool>()?
                        .outer_iter()
                        .map(|r| r.to_vec())
                        .collect();
                    Ok(serde_json::json!(rows))
                }
                _ => Err(ConvertError::Other(format!(
                    "unsupported 2-D uns array type: {desc:?}"
                ))),
            };
        }

        return Err(ConvertError::Other(format!(
            "unsupported uns dataset shape: {shape:?}"
        )));
    }

    // Try reading as subgroup → recurse
    if let Ok(subgroup) = group.group(name) {
        // A pandas DataFrame in uns (`encoding-type == "dataframe"`) is
        // preserved by the generic recurse below as a nested dict of
        // per-column values + `_index`, but column order (carried only in
        // the group's `column-order` attribute) and per-column categorical
        // dtypes are NOT reconstructed. Surface that structure loss so it
        // is never silent. The data still round-trips as a dict.
        let enc = subgroup
            .attr("encoding-type")
            .ok()
            .and_then(|a| a.read_scalar::<hdf5::types::VarLenUnicode>().ok())
            .map(|v| v.to_string())
            .unwrap_or_default();
        if enc == "dataframe" {
            sink.emit(ConvertWarning::FlattenedUnsDataframe {
                key: name.to_string(),
            });
        }

        let sub_members = subgroup.member_names()?;
        let mut sub_map = serde_json::Map::new();
        for sub_name in &sub_members {
            match read_uns_entry(&subgroup, sub_name, strict_uns, sink) {
                Ok(v) => {
                    sub_map.insert(sub_name.clone(), v);
                }
                Err(e) => {
                    if strict_uns {
                        return Err(e);
                    }
                    sink.emit(ConvertWarning::SkippedUnsKey {
                        key: format!("{name}/{sub_name}"),
                        reason: e.to_string(),
                    });
                }
            }
        }
        return Ok(serde_json::Value::Object(sub_map));
    }

    Err(ConvertError::Other(format!(
        "uns/{name}: not a dataset or group"
    )))
}

/// Read layers from h5ad file (root `/layers`).
/// Returns a map of layer_name → (indptr, indices, data, n_obs, n_vars).
pub fn read_layers(
    file: &hdf5::File,
    sink: &mut WarningSink,
) -> Result<HashMap<String, CsrArrays>, ConvertError> {
    read_layers_at(file, "layers", sink)
}

/// Read layers from a layer group at an arbitrary path (e.g.
/// `mod/rna/layers` for an h5mu file's per-modality layers). Missing
/// groups return an empty map.
pub fn read_layers_at(
    file: &hdf5::File,
    path: &str,
    sink: &mut WarningSink,
) -> Result<HashMap<String, CsrArrays>, ConvertError> {
    let mut result = HashMap::new();
    let layers_group = match file.group(path) {
        Ok(g) => g,
        Err(_) => return Ok(result),
    };
    let member_names = layers_group.member_names()?;
    for name in &member_names {
        match read_layer_entry(file, name, sink) {
            Ok(data) => {
                result.insert(name.clone(), data);
            }
            Err(e) => {
                sink.emit(ConvertWarning::LayerSkipped {
                    name: format!("{path}/{name}"),
                    reason: e.to_string(),
                });
            }
        }
    }
    Ok(result)
}

fn read_layer_entry(
    file: &hdf5::File,
    name: &str,
    sink: &mut WarningSink,
) -> Result<CsrArrays, ConvertError> {
    // Each layer is like X: a sparse group (CSR or CSC) or a dense
    // dataset. Route through the shared detector + matrix reader — the
    // same path `X` and `raw/X` use — so layers get CSC auto-detection
    // (from shape, not just an explicit `encoding-type` attr),
    // `canonicalize_csr`, and the indptr-length validation, instead of a
    // private inline encoding-type read.
    let path = format!("layers/{name}");
    let format = detect_matrix_format_at(file, &path, sink)?;
    read_x_matrix_at(file, &path, format)
}

#[cfg(test)]
mod read_dataset_as_tests {
    use super::*;

    /// Write a 1-D dataset of element type `T` at `/d` in a fresh temp
    /// HDF5 file. Returns the `TempDir` (keeps the file on disk for the
    /// test's lifetime) and the open file handle.
    fn ds_file<T: hdf5::H5Type>(values: &[T]) -> (tempfile::TempDir, hdf5::File) {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("t.h5")).unwrap();
        file.new_dataset::<T>()
            .shape([values.len()])
            .create("d")
            .unwrap()
            .write(values)
            .unwrap();
        (dir, file)
    }

    #[test]
    fn i64_widens_every_integer_width() {
        let (_d, f) = ds_file::<u32>(&[0u32, 1, 4_000_000_000]);
        let got = read_dataset_as::<i64>(&f.dataset("d").unwrap()).unwrap();
        assert_eq!(got, vec![0i64, 1, 4_000_000_000]);
    }

    #[test]
    fn i64_rejects_u64_over_i64_max() {
        let (_d, f) = ds_file::<u64>(&[1u64, u64::MAX]);
        let err = read_dataset_as::<i64>(&f.dataset("d").unwrap()).unwrap_err();
        assert!(matches!(
            err,
            ConvertError::IndexOverflow { target: "i64", .. }
        ));
    }

    #[test]
    fn int_targets_reject_float_source() {
        let (_d, f) = ds_file::<f32>(&[1.0f32, 2.0]);
        assert!(matches!(
            read_dataset_as::<i64>(&f.dataset("d").unwrap()).unwrap_err(),
            ConvertError::UnsupportedDtype(_)
        ));
        assert!(matches!(
            read_dataset_as::<i32>(&f.dataset("d").unwrap()).unwrap_err(),
            ConvertError::UnsupportedDtype(_)
        ));
    }

    #[test]
    fn i32_range_checks_narrowing_casts() {
        // In-range i64 source succeeds (incl. negatives).
        let (_d, f) = ds_file::<i64>(&[-5i64, 100, i32::MAX as i64]);
        assert_eq!(
            read_dataset_as::<i32>(&f.dataset("d").unwrap()).unwrap(),
            vec![-5i32, 100, i32::MAX]
        );
        // Out-of-range i64 overflows.
        let (_d2, f2) = ds_file::<i64>(&[(i32::MAX as i64) + 1]);
        assert!(matches!(
            read_dataset_as::<i32>(&f2.dataset("d").unwrap()).unwrap_err(),
            ConvertError::IndexOverflow { target: "i32", .. }
        ));
    }

    #[test]
    fn i32_rejects_u32_over_i32_max() {
        let (_d, f) = ds_file::<u32>(&[u32::MAX]);
        assert!(matches!(
            read_dataset_as::<i32>(&f.dataset("d").unwrap()).unwrap_err(),
            ConvertError::IndexOverflow {
                target: "i32",
                source_dtype: "u32",
                ..
            }
        ));
    }

    #[test]
    fn f32_accepts_every_numeric_width() {
        let (_d, f) = ds_file::<f64>(&[1.5f64, -2.25, 0.0]);
        assert_eq!(
            read_dataset_as::<f32>(&f.dataset("d").unwrap()).unwrap(),
            vec![1.5f32, -2.25, 0.0]
        );
        let (_d2, f2) = ds_file::<i64>(&[7i64, -8]);
        assert_eq!(
            read_dataset_as::<f32>(&f2.dataset("d").unwrap()).unwrap(),
            vec![7.0f32, -8.0]
        );
    }
}
