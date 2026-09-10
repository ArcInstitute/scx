// Streaming reader for h5ad sparse matrices.
//
// Hands back successive row-range shards of an on-disk CSR matrix
// (`X` or `/layers/{name}`) without ever materialising the full
// `indices` or `data` arrays. The full `indptr` is loaded eagerly —
// (n_obs + 1) × 8 bytes, ~80 MB at 10M cells, dominant resident cost
// at census-100M scale.

use hdf5::types::VarLenUnicode;
use ndarray::s;

use super::read::{read_i64_dataset, read_shape_2d};
use crate::detect::MatrixFormat;
use crate::pipeline::ConvertError;
use crate::stream::{CsrShardStream, IndexedCsrShardStream, StreamedCsrShard};
use crate::warnings::{ConvertWarning, WarningSink};

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
#[derive(Debug)]
pub struct XStreamReader {
    pub n_obs: usize,
    pub n_vars: usize,
    /// Source matrix label used by [`CsrShardStream::source_matrix_name`]
    /// and propagated onto each emitted [`StreamedCsrShard`]. Examples:
    /// `"X"`, `"layers/spliced"`.
    pub source_name: String,
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
    sink: &mut WarningSink,
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

    let (n_obs, n_vars) = read_shape_2d(&group, group_path)?;

    // Best-effort encoding-type sanity check. Newer h5ad files set
    // this attribute; older files omit it — in that case we trust the
    // caller's `format` argument and emit a warning so the conversion
    // record reflects the inference.
    match group.attr("encoding-type") {
        Ok(attr) => {
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
        Err(_) => {
            sink.emit(ConvertWarning::InferredEncoding {
                path: group_path.to_string(),
                inferred: "csr_matrix (no encoding-type attr)".into(),
            });
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
        source_name: group_path.to_string(),
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
    sink: &mut WarningSink,
) -> Result<XStreamReader, ConvertError> {
    open_x_streaming(
        file,
        &format!("layers/{layer_name}"),
        MatrixFormat::Csr,
        sink,
    )
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

        let indptr_slice = &self.indptr[row_start..=row_end];
        let (nnz_start, nnz_end) =
            scx_sparse::shard_nnz_bounds(indptr_slice, self.indices_ds.shape()[0])
                .map_err(|e| ConvertError::Other(format!("shard validation failed: {e}")))?;

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

        // Validate (monotonic indptr + column bound) + rebase + cast
        // through the shared scx-sparse helper. Both this streaming
        // reader and the eager pipeline path share it.
        let (shard_indptr, shard_indices) =
            scx_sparse::rebase_csr_shard(indptr_slice, &shard_indices_i32, self.n_vars as u64)
                .map_err(|e| ConvertError::Other(format!("shard validation failed: {e}")))?;

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

impl XStreamReader {
    /// Read rows `[row_start, row_start + n_rows)` without
    /// mutating internal state. Used by the parallel coordinator from
    /// worker threads (libhdf5 serialises overlapping reads internally
    /// under `--enable-threadsafe`; the runtime check in
    /// `crate::pipeline::hdf5_is_threadsafe` gates the parallel path).
    fn read_range_inner(
        &self,
        row_start: u64,
        n_rows: u32,
    ) -> Result<StreamedCsrShard, ConvertError> {
        let n_obs_u64 = self.n_obs as u64;
        if row_start.saturating_add(n_rows as u64) > n_obs_u64 {
            return Err(ConvertError::Other(format!(
                "read_range out of bounds: row_start={row_start}, n_rows={n_rows}, n_obs={n_obs_u64}"
            )));
        }
        let row_start_usize = row_start as usize;
        let row_end = row_start_usize + n_rows as usize;

        let indptr_slice = &self.indptr[row_start_usize..=row_end];
        let (nnz_start, nnz_end) =
            scx_sparse::shard_nnz_bounds(indptr_slice, self.indices_ds.shape()[0])
                .map_err(|e| ConvertError::Other(format!("shard validation failed: {e}")))?;

        let (shard_indices_i32, shard_values) = if nnz_start == nnz_end {
            (Vec::<i32>::new(), Vec::<f32>::new())
        } else {
            let i = read_slice_i32(&self.indices_ds, nnz_start, nnz_end)?;
            let v = read_slice_f32(&self.data_ds, nnz_start, nnz_end)?;
            (i, v)
        };

        // Validate + rebase + cast through the shared scx-sparse helper.
        let (shard_indptr, shard_indices) =
            scx_sparse::rebase_csr_shard(indptr_slice, &shard_indices_i32, self.n_vars as u64)
                .map_err(|e| ConvertError::Other(format!("shard validation failed: {e}")))?;

        let n_cols = u32::try_from(self.n_vars)
            .map_err(|_| ConvertError::Other(format!("n_vars {} exceeds u32::MAX", self.n_vars)))?;
        Ok(StreamedCsrShard {
            row_start,
            n_rows,
            n_cols,
            indptr: shard_indptr,
            indices: shard_indices,
            values: shard_values,
            source_name: Some(self.source_name.clone()),
            duplicates_merged: 0,
        })
    }
}

impl CsrShardStream for XStreamReader {
    fn n_obs(&self) -> u64 {
        self.n_obs as u64
    }

    fn n_vars(&self) -> u64 {
        self.n_vars as u64
    }

    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }

    fn next_csr_shard(
        &mut self,
        target_rows: usize,
    ) -> Result<Option<StreamedCsrShard>, ConvertError> {
        match self.next_shard(target_rows) {
            None => Ok(None),
            Some(Err(e)) => Err(e),
            Some(Ok(slice)) => {
                // Inherent `next_shard` already validated row counts
                // against `n_obs`; the casts to u32 are bounded by the
                // `n_vars` u32-fit check in the writer coordinator.
                let n_rows = u32::try_from(slice.n_rows).map_err(|_| {
                    ConvertError::Other(format!(
                        "shard row count {} exceeds u32::MAX",
                        slice.n_rows
                    ))
                })?;
                let n_cols = u32::try_from(self.n_vars).map_err(|_| {
                    ConvertError::Other(format!("n_vars {} exceeds u32::MAX", self.n_vars))
                })?;
                Ok(Some(StreamedCsrShard {
                    row_start: slice.row_start as u64,
                    n_rows,
                    n_cols,
                    indptr: slice.indptr,
                    indices: slice.indices,
                    values: slice.values,
                    source_name: Some(self.source_name.clone()),
                    duplicates_merged: 0,
                }))
            }
        }
    }

    fn as_indexed(&self) -> Option<&dyn IndexedCsrShardStream> {
        Some(self)
    }
}

impl IndexedCsrShardStream for XStreamReader {
    fn n_obs(&self) -> u64 {
        self.n_obs as u64
    }

    fn n_vars(&self) -> u64 {
        self.n_vars as u64
    }

    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }

    fn read_range(&self, row_start: u64, n_rows: u32) -> Result<StreamedCsrShard, ConvertError> {
        self.read_range_inner(row_start, n_rows)
    }

    /// C7: exact per-worker working set from the resident `indptr` — the
    /// largest nnz of any `shard_target_rows`-row window — instead of the
    /// density-ceiling default that under-estimates dense-stored-as-CSR
    /// inputs and over-spawns workers into OOM. No I/O: `indptr` is loaded
    /// eagerly at open.
    /// Resident and loaded eagerly at open, so the permuted adapter can price
    /// a reordered shard from real per-row nnz.
    fn source_row_indptr(&self) -> Option<&[i64]> {
        Some(&self.indptr)
    }

    fn per_worker_bytes(
        &self,
        shard_target_rows: u32,
        _modality_type: scx_format_io::modality::ModalityType,
    ) -> u64 {
        let n_obs = self.indptr.len().saturating_sub(1);
        if n_obs == 0 {
            return 1;
        }
        let t = (shard_target_rows.max(1) as usize).min(n_obs);
        let mut max_nnz: u64 = 0;
        let mut start = 0usize;
        while start < n_obs {
            let end = (start + t).min(n_obs);
            let nnz = self.indptr[end].saturating_sub(self.indptr[start]).max(0) as u64;
            max_nnz = max_nnz.max(nnz);
            start = end;
        }
        crate::stream::shard_working_set_bytes(max_nnz, t as u64)
    }
}

/// Slice-read variant of `read_i32_dataset` from `read.rs`.
/// Accepts every integer width; widens narrow source values and
/// range-checks narrowing casts (`i64` / `u32` / `u64`). Overflow
/// returns [`ConvertError::IndexOverflow`] — silent truncation of CSR
/// `indices` would corrupt the on-disk sparse layout. Float source
/// dtypes are rejected.
pub(crate) fn read_slice_i32(
    ds: &hdf5::Dataset,
    start: usize,
    end: usize,
) -> Result<Vec<i32>, ConvertError> {
    use crate::hdf_dtype::HdfNumericDtype;
    let path = ds.name();
    let desc = ds.dtype()?.to_descriptor()?;
    let dt = HdfNumericDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "dataset '{path}': dtype {desc:?} cannot be read as i32"
        ))
    })?;
    let sel = s![start..end];
    match dt {
        HdfNumericDtype::I8 => {
            let (data, _) = ds.read_slice_1d::<i8, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(i32::from).collect())
        }
        HdfNumericDtype::I16 => {
            let (data, _) = ds.read_slice_1d::<i16, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(i32::from).collect())
        }
        HdfNumericDtype::I32 => {
            let (data, _) = ds.read_slice_1d::<i32, _>(sel)?.into_raw_vec_and_offset();
            Ok(data)
        }
        HdfNumericDtype::I64 => {
            let (data, _) = ds.read_slice_1d::<i64, _>(sel)?.into_raw_vec_and_offset();
            if let Some(&v) = data
                .iter()
                .find(|&&v| v < i32::MIN as i64 || v > i32::MAX as i64)
            {
                return Err(ConvertError::IndexOverflow {
                    path,
                    source_dtype: dt.name(),
                    target: "i32",
                    value: v.to_string(),
                });
            }
            Ok(data.into_iter().map(|v| v as i32).collect())
        }
        HdfNumericDtype::U8 => {
            let (data, _) = ds.read_slice_1d::<u8, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(i32::from).collect())
        }
        HdfNumericDtype::U16 => {
            let (data, _) = ds.read_slice_1d::<u16, _>(sel)?.into_raw_vec_and_offset();
            Ok(data.into_iter().map(i32::from).collect())
        }
        HdfNumericDtype::U32 => {
            let (data, _) = ds.read_slice_1d::<u32, _>(sel)?.into_raw_vec_and_offset();
            if let Some(&v) = data.iter().find(|&&v| v > i32::MAX as u32) {
                return Err(ConvertError::IndexOverflow {
                    path,
                    source_dtype: dt.name(),
                    target: "i32",
                    value: v.to_string(),
                });
            }
            Ok(data.into_iter().map(|v| v as i32).collect())
        }
        HdfNumericDtype::U64 => {
            let (data, _) = ds.read_slice_1d::<u64, _>(sel)?.into_raw_vec_and_offset();
            if let Some(&v) = data.iter().find(|&&v| v > i32::MAX as u64) {
                return Err(ConvertError::IndexOverflow {
                    path,
                    source_dtype: dt.name(),
                    target: "i32",
                    value: v.to_string(),
                });
            }
            Ok(data.into_iter().map(|v| v as i32).collect())
        }
        HdfNumericDtype::F16 | HdfNumericDtype::F32 | HdfNumericDtype::F64 => {
            Err(ConvertError::UnsupportedDtype(format!(
                "dataset '{path}': float dtype {desc:?} cannot be read as i32"
            )))
        }
    }
}

/// Slice-read variant of `read_f32_dataset` from `read.rs`.
/// Accepts every numeric width; casts signed and unsigned integers
/// and `f64` to `f32`. Casts from `i64` / `u64` may lose precision
/// for values above 2^24 — documented behaviour.
pub(crate) fn read_slice_f32(
    ds: &hdf5::Dataset,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, ConvertError> {
    use crate::hdf_dtype::HdfNumericDtype;
    let path = ds.name();
    let desc = ds.dtype()?.to_descriptor()?;
    let dt = HdfNumericDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "dataset '{path}': dtype {desc:?} cannot be read as f32"
        ))
    })?;
    let sel = s![start..end];
    macro_rules! read_cast {
        ($t:ty) => {{
            let (data, _) = ds.read_slice_1d::<$t, _>(sel)?.into_raw_vec_and_offset();
            data.into_iter().map(|v| v as f32).collect()
        }};
    }
    Ok(match dt {
        HdfNumericDtype::F16 => {
            // `half::f16` has no `as f32` cast; widen via `to_f32()`.
            let (data, _) = ds
                .read_slice_1d::<half::f16, _>(sel)?
                .into_raw_vec_and_offset();
            data.into_iter().map(|v| v.to_f32()).collect()
        }
        HdfNumericDtype::F32 => {
            let (data, _) = ds.read_slice_1d::<f32, _>(sel)?.into_raw_vec_and_offset();
            data
        }
        HdfNumericDtype::F64 => read_cast!(f64),
        HdfNumericDtype::I8 => read_cast!(i8),
        HdfNumericDtype::I16 => read_cast!(i16),
        HdfNumericDtype::I32 => read_cast!(i32),
        HdfNumericDtype::I64 => read_cast!(i64),
        HdfNumericDtype::U8 => read_cast!(u8),
        HdfNumericDtype::U16 => read_cast!(u16),
        HdfNumericDtype::U32 => read_cast!(u32),
        HdfNumericDtype::U64 => read_cast!(u64),
    })
}
