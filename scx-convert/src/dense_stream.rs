// Streaming reader for dense h5ad `/X` (or `/layers/<name>`) datasets.
//
// Implements [`CsrShardStream`] over a 2D HDF5 dataset by slab-reading
// `target_rows` rows at a time and sparsifying each slab into a
// [`StreamedCsrShard`].
//
// Peak memory per shard is bounded by
// `slab_rows × n_vars × sizeof(source_dtype)` for the dense buffer,
// plus the sparsified CSR working set. `ConvertOptions::memory_budget`
// caps `slab_rows` independently of `shard_target_rows` so dense
// inputs with very large `n_vars` don't exceed the budget. When the
// budget is smaller than a single dense row, `open_dense_streaming`
// returns an actionable error rather than silently disabling the cap.

use hdf5::types::{FloatSize, IntSize, TypeDescriptor};
use ndarray::s;

use super::pipeline::{ConvertError, ConvertOptions};
use super::stream::{CsrShardStream, IndexedCsrShardStream, StreamedCsrShard};
use super::warnings::WarningSink;

/// Streaming reader over an h5ad dense matrix dataset. Open via
/// [`open_dense_streaming`] or [`open_dense_layer_streaming`], then
/// drive through `CsrShardStream::next_csr_shard`.
#[derive(Debug)]
pub struct DenseXStreamReader {
    pub n_obs: u64,
    pub n_vars: u64,
    /// `"X"` or `"layers/{name}"` for diagnostics. Matches the
    /// [`CsrShardStream::source_matrix_name`] return value.
    pub source_name: String,
    /// On-disk dataset handle. Sliced per-shard, never read whole.
    dataset: hdf5::Dataset,
    /// On-disk numeric dtype; drives the per-slab cast to f32.
    dtype: DenseDtype,
    /// `dense_zero_epsilon` snapshot from [`ConvertOptions`].
    /// `0.0` means equality-to-zero filtering (matches scipy).
    zero_eps: f32,
    cursor: u64,
    /// Budget-derived ceiling on rows per slab. Actual emitted slab
    /// size is `min(target_rows_from_coordinator, max_slab_rows, n_obs - cursor)`.
    max_slab_rows: usize,
}

/// On-disk numeric dtype for a 2D dense HDF5 dataset. Drives the
/// per-slab cast to `f32` in [`read_dense_slab_f32`].
#[derive(Debug, Clone, Copy)]
pub(crate) enum DenseDtype {
    F32,
    F64,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
}

impl DenseDtype {
    pub(crate) fn from_descriptor(desc: &TypeDescriptor) -> Result<Self, ConvertError> {
        Ok(match desc {
            TypeDescriptor::Float(FloatSize::U4) => Self::F32,
            TypeDescriptor::Float(FloatSize::U8) => Self::F64,
            TypeDescriptor::Integer(IntSize::U8) => Self::I64,
            TypeDescriptor::Integer(IntSize::U4) => Self::I32,
            TypeDescriptor::Integer(IntSize::U2) => Self::I16,
            TypeDescriptor::Integer(IntSize::U1) => Self::I8,
            TypeDescriptor::Unsigned(IntSize::U8) => Self::U64,
            TypeDescriptor::Unsigned(IntSize::U4) => Self::U32,
            TypeDescriptor::Unsigned(IntSize::U2) => Self::U16,
            TypeDescriptor::Unsigned(IntSize::U1) => Self::U8,
            other => {
                return Err(ConvertError::UnsupportedDtype(format!(
                    "dense /X dtype {other:?} not supported by streaming reader"
                )));
            }
        })
    }

    pub(crate) fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }
}

/// Read rows `[row_start, row_end)` of a 2D dense HDF5 dataset as a
/// row-major `Vec<f32>` of length `(row_end - row_start) * n_vars`.
/// Dispatches on `dtype` and casts non-`f32` source values to `f32`.
/// Used by [`DenseXStreamReader::read_slab_f32`] and the codec-sampling
/// path for dense h5mu modalities.
pub(crate) fn read_dense_slab_f32(
    ds: &hdf5::Dataset,
    dtype: DenseDtype,
    row_start: usize,
    row_end: usize,
) -> Result<Vec<f32>, ConvertError> {
    let sel = s![row_start..row_end, ..];
    macro_rules! read_and_cast {
        ($t:ty) => {{
            let (data, _) = ds.read_slice_2d::<$t, _>(sel)?.into_raw_vec_and_offset();
            data.into_iter().map(|v| v as f32).collect()
        }};
    }
    let slab: Vec<f32> = match dtype {
        DenseDtype::F32 => {
            let (data, _) = ds.read_slice_2d::<f32, _>(sel)?.into_raw_vec_and_offset();
            data
        }
        DenseDtype::F64 => read_and_cast!(f64),
        DenseDtype::I64 => read_and_cast!(i64),
        DenseDtype::I32 => read_and_cast!(i32),
        DenseDtype::I16 => read_and_cast!(i16),
        DenseDtype::I8 => read_and_cast!(i8),
        DenseDtype::U64 => read_and_cast!(u64),
        DenseDtype::U32 => read_and_cast!(u32),
        DenseDtype::U16 => read_and_cast!(u16),
        DenseDtype::U8 => read_and_cast!(u8),
    };
    Ok(slab)
}

/// Open a dense matrix dataset for streaming row-range reads.
///
/// `path` is typically `"X"` for the primary matrix or
/// `"layers/<name>"` for a layer. The dtype must be a numeric scalar
/// type (no compound / string / bool); unsupported types return
/// [`ConvertError::UnsupportedDtype`].
pub fn open_dense_streaming(
    file: &hdf5::File,
    path: &str,
    opts: &ConvertOptions,
    _sink: &mut WarningSink,
) -> Result<DenseXStreamReader, ConvertError> {
    let dataset = file.dataset(path)?;
    let shape = dataset.shape();
    if shape.len() != 2 {
        return Err(ConvertError::Other(format!(
            "dense matrix at '{path}' must be 2D, got {}-D",
            shape.len()
        )));
    }
    let n_obs = shape[0] as u64;
    let n_vars = shape[1] as u64;

    let dtype = DenseDtype::from_descriptor(&dataset.dtype()?.to_descriptor()?)?;

    // `memory_budget / (n_vars * sizeof(dtype)) / 4` — the `/4`
    // reserves headroom for the sparsified output, the encoder queue,
    // and per-shard sort scratch. A budget too small to fit one
    // 4×-reserved row would silently lose the cap and risk OOM —
    // reject with an actionable error instead, mirroring the
    // `csc_stream::open_csc_streaming` precedent.
    let max_slab_rows = match opts.memory_budget {
        None => usize::MAX,
        Some(budget) => {
            let row_bytes = (n_vars as usize).saturating_mul(dtype.size_bytes());
            let min_required = row_bytes.saturating_mul(4);
            match (budget as usize).checked_div(row_bytes) {
                None | Some(0) => {
                    return Err(ConvertError::Other(format!(
                        "memory_budget {budget} bytes too small for dense streaming of \
                         {n_vars} vars × {dtype:?}; need at least {min_required} bytes \
                         (≈ 4 × row_bytes for slab + sparsified output + encoder headroom)"
                    )));
                }
                Some(rows) => (rows / 4).max(1),
            }
        }
    };

    Ok(DenseXStreamReader {
        n_obs,
        n_vars,
        source_name: path.to_string(),
        dataset,
        dtype,
        zero_eps: opts.dense_zero_epsilon,
        cursor: 0,
        max_slab_rows,
    })
}

/// Open `/layers/{layer_name}` as a dense streaming reader. h5ad
/// layers share the same dtype rules as `X`.
pub fn open_dense_layer_streaming(
    file: &hdf5::File,
    layer_name: &str,
    opts: &ConvertOptions,
    sink: &mut WarningSink,
) -> Result<DenseXStreamReader, ConvertError> {
    open_dense_streaming(file, &format!("layers/{layer_name}"), opts, sink)
}

impl DenseXStreamReader {
    /// Read rows `[row_start, row_start + n_rows)` and
    /// sparsify into a `StreamedCsrShard` without mutating internal
    /// state. Subject to the same `max_slab_rows` budget cap as
    /// `next_csr_shard`; callers (the parallel coordinator) must size
    /// their ranges accordingly via [`crate::pipeline::compute_shard_row_ranges`]
    /// with `target_rows ≤ max_slab_rows`.
    fn read_range_inner(
        &self,
        row_start: u64,
        n_rows: u32,
    ) -> Result<StreamedCsrShard, ConvertError> {
        if row_start.saturating_add(n_rows as u64) > self.n_obs {
            return Err(ConvertError::Other(format!(
                "dense read_range out of bounds: row_start={row_start}, n_rows={n_rows}, n_obs={}",
                self.n_obs
            )));
        }
        if (n_rows as usize) > self.max_slab_rows {
            return Err(ConvertError::Other(format!(
                "dense read_range slab_rows={n_rows} exceeds max_slab_rows={} (lower --shard-size or raise --memory-budget)",
                self.max_slab_rows
            )));
        }
        let row_start_usize = row_start as usize;
        let row_end = row_start_usize + n_rows as usize;
        let n_vars = self.n_vars as usize;
        let slab_rows = n_rows as usize;

        let flat = self.read_slab_f32(row_start_usize, row_end, n_vars)?;

        let mut indptr: Vec<u64> = Vec::with_capacity(slab_rows + 1);
        let mut indices: Vec<u32> = Vec::with_capacity(slab_rows * n_vars / 32 + 1);
        let mut values: Vec<f32> = Vec::with_capacity(slab_rows * n_vars / 32 + 1);

        indptr.push(0);
        let eps = self.zero_eps;
        if eps == 0.0 {
            for row in 0..slab_rows {
                let base = row * n_vars;
                for col in 0..n_vars {
                    let v = flat[base + col];
                    if v != 0.0 {
                        indices.push(col as u32);
                        values.push(v);
                    }
                }
                indptr.push(values.len() as u64);
            }
        } else {
            for row in 0..slab_rows {
                let base = row * n_vars;
                for col in 0..n_vars {
                    let v = flat[base + col];
                    if v.abs() > eps {
                        indices.push(col as u32);
                        values.push(v);
                    }
                }
                indptr.push(values.len() as u64);
            }
        }

        Ok(StreamedCsrShard {
            row_start,
            n_rows,
            n_cols: self.n_vars as u32,
            indptr,
            indices,
            values,
            source_name: Some(self.source_name.clone()),
            duplicates_merged: 0,
        })
    }
}

impl CsrShardStream for DenseXStreamReader {
    fn n_obs(&self) -> u64 {
        self.n_obs
    }
    fn n_vars(&self) -> u64 {
        self.n_vars
    }
    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }

    fn next_csr_shard(
        &mut self,
        target_rows: usize,
    ) -> Result<Option<StreamedCsrShard>, ConvertError> {
        if self.cursor >= self.n_obs {
            return Ok(None);
        }
        if target_rows == 0 {
            return Err(ConvertError::Other("target_rows must be > 0".into()));
        }

        let remaining = (self.n_obs - self.cursor) as usize;
        let slab_rows = target_rows.min(self.max_slab_rows).min(remaining);
        let shard = self.read_range_inner(self.cursor, slab_rows as u32)?;
        self.cursor += slab_rows as u64;
        Ok(Some(shard))
    }

    fn as_indexed(&self) -> Option<&dyn IndexedCsrShardStream> {
        Some(self)
    }
}

impl IndexedCsrShardStream for DenseXStreamReader {
    fn n_obs(&self) -> u64 {
        self.n_obs
    }
    fn n_vars(&self) -> u64 {
        self.n_vars
    }
    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }
    fn read_range(&self, row_start: u64, n_rows: u32) -> Result<StreamedCsrShard, ConvertError> {
        self.read_range_inner(row_start, n_rows)
    }
}

impl DenseXStreamReader {
    /// Read rows `[row_start, row_end)` of the dense dataset and cast
    /// to a row-major `Vec<f32>` of length `slab_rows * n_vars`.
    /// hdf5-metno returns the slab in the caller's requested layout
    /// regardless of on-disk storage order, so column-major datasets
    /// transparently transpose during the read (perf hit but not a
    /// correctness gap).
    fn read_slab_f32(
        &self,
        row_start: usize,
        row_end: usize,
        n_vars: usize,
    ) -> Result<Vec<f32>, ConvertError> {
        let slab = read_dense_slab_f32(&self.dataset, self.dtype, row_start, row_end)?;
        let expected = (row_end - row_start) * n_vars;
        if slab.len() != expected {
            return Err(ConvertError::Other(format!(
                "dense slab read returned {} elements, expected {}",
                slab.len(),
                expected
            )));
        }
        Ok(slab)
    }
}
