// Streaming reader for h5ad sparse matrices.
//
// Hands back successive row-range shards of an on-disk CSR matrix
// (`X` or `/layers/{name}`) without ever materialising the full
// `indices` or `data` arrays. The full `indptr` is loaded eagerly —
// (n_obs + 1) × 8 bytes, ~80 MB at 10M cells, dominant resident cost
// at census-100M scale.

use hdf5::types::{IntSize, TypeDescriptor, VarLenUnicode};
use ndarray::s;

use super::detect::MatrixFormat;
use super::h5ad_read::read_i64_dataset;
use super::pipeline::ConvertError;

/// A single shard's worth of CSR rows read from an h5ad file.
///
/// `indptr` is shard-local (first element is always 0, length is
/// `n_rows + 1`). `indices` and `values` are concatenated across the
/// shard's rows.
pub struct CsrShardSlice {
    pub row_start: usize,
    pub n_rows: usize,
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

/// Streaming reader over an h5ad sparse matrix group (`X` or
/// `/layers/{name}`). Open once via [`open_x_streaming`] /
/// [`open_layer_streaming`], then call [`XStreamReader::next_shard`]
/// in a loop until it yields `None`.
pub struct XStreamReader {
    pub n_obs: usize,
    pub n_vars: usize,
    /// Eagerly loaded — `(n_obs + 1) × 8` bytes.
    indptr: Vec<i64>,
    /// Open HDF5 dataset handles; sliced per-shard, never read whole.
    indices_ds: hdf5::Dataset,
    data_ds: hdf5::Dataset,
    cursor: usize,
}

/// Open a sparse matrix group for streaming row-range reads.
///
/// Refuses CSC-on-disk and dense matrices with
/// [`ConvertError::StreamingUnsupported`]. The caller must opt out of
/// streaming (`--stream=false` / non-streaming `from_anndata` path) for
/// those formats.
pub fn open_x_streaming(
    file: &hdf5::File,
    group_path: &str,
    format: MatrixFormat,
) -> Result<XStreamReader, ConvertError> {
    if matches!(format, MatrixFormat::Dense) {
        return Err(ConvertError::StreamingUnsupported(
            "dense X cannot stream; pass --stream=false".into(),
        ));
    }
    if matches!(format, MatrixFormat::Csc) {
        return Err(ConvertError::StreamingUnsupported(
            "CSC-on-disk h5ad cannot stream; pass --stream=false or \
             pre-convert to CSR"
                .into(),
        ));
    }

    let group = file.group(group_path)?;

    let shape: Vec<i64> = group.attr("shape")?.read_1d()?.to_vec();
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "expected 2D shape attr on '{group_path}', got {}-D",
            shape.len()
        )));
    }
    let n_obs = shape[0] as usize;
    let n_vars = shape[1] as usize;

    // Best-effort encoding-type sanity check. Newer h5ad files set
    // this attribute; older files omit it — in that case we trust the
    // caller's `format` argument.
    if let Ok(attr) = group.attr("encoding-type") {
        if let Ok(enc) = attr.read_scalar::<VarLenUnicode>() {
            let enc_s = enc.as_str();
            if enc_s == "csc_matrix" {
                return Err(ConvertError::StreamingUnsupported(
                    "CSC-on-disk h5ad cannot stream; pass --stream=false or \
                     pre-convert to CSR"
                        .into(),
                ));
            }
            if enc_s != "csr_matrix" {
                return Err(ConvertError::StreamingUnsupported(format!(
                    "unsupported encoding-type '{enc_s}' for streaming"
                )));
            }
        }
    }

    let indptr_ds = group.dataset("indptr")?;
    let indptr = read_i64_dataset(&indptr_ds)?;
    if indptr.len() != n_obs + 1 {
        return Err(ConvertError::Other(format!(
            "indptr length {} != n_obs + 1 ({})",
            indptr.len(),
            n_obs + 1
        )));
    }

    let indices_ds = group.dataset("indices")?;
    let data_ds = group.dataset("data")?;

    Ok(XStreamReader {
        n_obs,
        n_vars,
        indptr,
        indices_ds,
        data_ds,
        cursor: 0,
    })
}

/// Open `/layers/{layer_name}` for streaming reads. h5ad stores
/// layers with the same CSR layout as `X`.
pub fn open_layer_streaming(
    file: &hdf5::File,
    layer_name: &str,
) -> Result<XStreamReader, ConvertError> {
    open_x_streaming(file, &format!("layers/{layer_name}"), MatrixFormat::Csr)
}

impl XStreamReader {
    /// Pull the next shard of up to `target_rows` rows. Returns
    /// `None` once all rows have been emitted.
    pub fn next_shard(
        &mut self,
        target_rows: usize,
    ) -> Option<Result<CsrShardSlice, ConvertError>> {
        if self.cursor >= self.n_obs {
            return None;
        }
        if target_rows == 0 {
            return Some(Err(ConvertError::Other("target_rows must be > 0".into())));
        }
        Some(self.read_next_shard(target_rows))
    }

    fn read_next_shard(&mut self, target_rows: usize) -> Result<CsrShardSlice, ConvertError> {
        let row_start = self.cursor;
        let row_end = (row_start + target_rows).min(self.n_obs);
        let n_rows = row_end - row_start;

        let base = self.indptr[row_start];
        let end_val = self.indptr[row_end];
        let nnz_start = usize::try_from(base)
            .map_err(|_| ConvertError::Other(format!("negative indptr base {base}")))?;
        let nnz_end = usize::try_from(end_val)
            .map_err(|_| ConvertError::Other(format!("negative indptr end {end_val}")))?;
        if nnz_end < nnz_start {
            return Err(ConvertError::Other(format!(
                "indptr non-monotonic across shard: base={base}, end={end_val}"
            )));
        }

        // Read the indices / data slices. Empty shards (nnz_start ==
        // nnz_end) skip the hdf5 read entirely — `read_slice_1d` on
        // an empty range is not well-defined across hdf5-rust
        // versions, so we short-circuit.
        let (shard_indices_i32, shard_values) = if nnz_start == nnz_end {
            (Vec::<i32>::new(), Vec::<f32>::new())
        } else {
            let i = read_slice_i32(&self.indices_ds, nnz_start, nnz_end)?;
            let v = read_slice_f32(&self.data_ds, nnz_start, nnz_end)?;
            (i, v)
        };

        // Validate the on-disk i64 indptr slice + i32 indices before
        // any dtype coercion. Phase 2 hoisted this helper into
        // scx-sparse; both this streaming reader and the in-memory
        // pyscx path share it.
        scx_sparse::validate_csr_arrays(
            &self.indptr[row_start..=row_end],
            &shard_indices_i32,
            self.n_vars as u64,
        )
        .map_err(|e| ConvertError::Other(format!("shard validation failed: {e}")))?;

        // Rebase the shard-local indptr to start at 0. Validation
        // above guarantees monotonicity, so `(v - base)` is non-
        // negative and `as u64` is lossless.
        let mut shard_indptr: Vec<u64> = Vec::with_capacity(n_rows + 1);
        for &v in &self.indptr[row_start..=row_end] {
            shard_indptr.push((v - base) as u64);
        }

        let shard_indices: Vec<u32> = shard_indices_i32.into_iter().map(|v| v as u32).collect();

        self.cursor = row_end;
        Ok(CsrShardSlice {
            row_start,
            n_rows,
            indptr: shard_indptr,
            indices: shard_indices,
            values: shard_values,
        })
    }
}

/// Slice-read variant of `read_i32_dataset` from `h5ad_read.rs`.
/// Dispatches on the on-disk dtype (i32 / i64 / u32 supported) and
/// applies the same range-validation checks per element.
fn read_slice_i32(ds: &hdf5::Dataset, start: usize, end: usize) -> Result<Vec<i32>, ConvertError> {
    let desc = ds.dtype()?.to_descriptor()?;
    let sel = s![start..end];
    match desc {
        TypeDescriptor::Integer(IntSize::U4) => {
            let (data, _) = ds.read_slice_1d::<i32, _>(sel)?.into_raw_vec_and_offset();
            Ok(data)
        }
        TypeDescriptor::Integer(IntSize::U8) => {
            let (data, _) = ds.read_slice_1d::<i64, _>(sel)?.into_raw_vec_and_offset();
            if let Some(&v) = data
                .iter()
                .find(|&&v| v < i32::MIN as i64 || v > i32::MAX as i64)
            {
                return Err(ConvertError::Other(format!(
                    "i64 index value {v} out of i32 range"
                )));
            }
            Ok(data.into_iter().map(|v| v as i32).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            let (data, _) = ds.read_slice_1d::<u32, _>(sel)?.into_raw_vec_and_offset();
            if let Some(&v) = data.iter().find(|&&v| v > i32::MAX as u32) {
                return Err(ConvertError::Other(format!(
                    "u32 index value {v} exceeds i32::MAX"
                )));
            }
            Ok(data.into_iter().map(|v| v as i32).collect())
        }
        other => Err(ConvertError::UnsupportedDtype(format!(
            "indices dtype {other:?} not supported by streaming reader"
        ))),
    }
}

/// Slice-read variant of `read_f32_dataset` from `h5ad_read.rs`.
fn read_slice_f32(ds: &hdf5::Dataset, start: usize, end: usize) -> Result<Vec<f32>, ConvertError> {
    let desc = ds.dtype()?.to_descriptor()?;
    let sel = s![start..end];
    match desc {
        TypeDescriptor::Float(hdf5::types::FloatSize::U4) => {
            let (data, _) = ds.read_slice_1d::<f32, _>(sel)?.into_raw_vec_and_offset();
            Ok(data)
        }
        TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
            let (data, _) = ds.read_slice_1d::<f64, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        TypeDescriptor::Integer(IntSize::U4) => {
            let (data, _) = ds.read_slice_1d::<i32, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            let (data, _) = ds.read_slice_1d::<u32, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(|v| v as f32).collect())
        }
        other => Err(ConvertError::UnsupportedDtype(format!(
            "data dtype {other:?} not supported by streaming reader"
        ))),
    }
}
