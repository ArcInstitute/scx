// Streaming reader for dense h5ad `/X` (or `/layers/<name>`) datasets.
//
// Implements [`CsrShardStream`] over a 2D HDF5 dataset by slab-reading
// `target_rows` rows at a time and sparsifying each slab into a
// [`StreamedCsrShard`].
//
// Peak memory for one `read_range` is `slab_rows × n_vars ×
// crate::budget::dense_peak_bytes_per_elem(dtype)` — 12 B/element, and
// deliberately NOT keyed to `sizeof(source_dtype)`: the resident slab is f32
// whatever the input was, and the sparsified output does not depend on the
// source width at all. Sizing by the source width is the bug the `/4` here used
// to have (it under-reserved 3x for `u8`), so do not reintroduce that shape.
// `IngestOptions::memory_budget` caps `slab_rows` independently of
// `shard_target_rows` so dense inputs with very large `n_vars` stay inside
// their share of the budget.
//
// ⚠️ That bound covers the READER phase only. The worker calling it also holds
// the encoded shard alongside the raw CSR — see the `enforced: false` note on
// the ingest rows in `crate::budget::ALLOCATION_TABLE`. When the
// budget is smaller than a single dense row, `open_dense_streaming`
// returns an actionable error rather than silently disabling the cap.

use ndarray::s;

use crate::pipeline::{ConvertError, IngestOptions};
use crate::stream::{CsrShardStream, IndexedCsrShardStream, StreamedCsrShard};
use crate::warnings::WarningSink;

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
    /// `dense_zero_epsilon` snapshot from [`IngestOptions`].
    /// `0.0` means equality-to-zero filtering (matches scipy).
    zero_eps: f32,
    cursor: u64,
    /// Budget-derived ceiling on rows per slab. Actual emitted slab
    /// size is `min(target_rows_from_coordinator, max_slab_rows, n_obs - cursor)`.
    max_slab_rows: usize,
}

/// On-disk numeric dtype for a 2D dense HDF5 dataset. Aliased to the
/// shared `HdfNumericDtype` so the sparse readers and slice readers
/// dispatch on the same surface. The `read_dense_slab_f32` macro below
/// still matches every variant exhaustively.
pub(crate) use crate::hdf_dtype::HdfNumericDtype as DenseDtype;

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
        DenseDtype::F16 => {
            // `half::f16` has no `as f32` cast; widen via `to_f32()`.
            let (data, _) = ds
                .read_slice_2d::<half::f16, _>(sel)?
                .into_raw_vec_and_offset();
            data.into_iter().map(|v| v.to_f32()).collect()
        }
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
    opts: &IngestOptions,
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

    let desc = dataset.dtype()?.to_descriptor()?;
    let dtype = DenseDtype::from_descriptor(&desc).map_err(|_| {
        ConvertError::UnsupportedDtype(format!(
            "dense /X dtype {desc:?} not supported by streaming reader"
        ))
    })?;

    // One slab may claim `budget::SHARD_BUDGET_SHARE` of the budget; how many
    // rows that is comes from the table's cost model. Both the refusal
    // predicate and the number the message advertises come from the same pair
    // of functions, so they cannot drift -- they used to: the guard tested
    // `budget / row_bytes == 0` while the message promised `4 x row_bytes`, so
    // a budget of two rows passed a check claiming to need four.
    let dtype_bytes = dtype.size_bytes() as u64;
    let max_slab_rows = match opts.memory_budget {
        None => usize::MAX,
        Some(budget) => match crate::budget::dense_max_slab_rows(budget, n_vars, dtype_bytes) {
            Err(()) => {
                let min_required = crate::budget::dense_min_budget(n_vars, dtype_bytes);
                return Err(ConvertError::Other(format!(
                    "memory_budget {budget} bytes too small for dense streaming of \
                     {n_vars} vars × {dtype:?}; need at least {min_required} bytes \
                     (one slab row at {} B/element, which is \
                     {}/{} of the budget)",
                    crate::budget::dense_peak_bytes_per_elem(dtype_bytes),
                    crate::budget::SHARD_BUDGET_SHARE.numerator(),
                    crate::budget::SHARD_BUDGET_SHARE.denominator(),
                )));
            }
            Ok(rows) => usize::try_from(rows).unwrap_or(usize::MAX).max(1),
        },
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
    opts: &IngestOptions,
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

        // Count first, then reserve exactly.
        //
        // These used to start at `slab_rows * n_vars / 32` and grow by
        // doubling, which is what made this function's true peak unbounded by
        // anything the budget knew about: doubling overshoots the final length
        // by up to 2x, and while a realloc is in flight both the old and new
        // buffers are live, so a dense slab could reach ~24 B/element against
        // the 12 the allocation table budgets for
        // (`budget::DENSE_SPARSIFY_BYTES_PER_ELEM`). One extra linear scan of
        // `flat` buys an exact allocation, and it also deletes up to five
        // realloc+memcpy rounds on the way up.
        //
        // The predicate below must stay identical to the two retain predicates
        // in the loops that follow; `dense_sparsify_allocates_exactly` pins
        // that they agree.
        let eps = self.zero_eps;
        let nnz = if eps == 0.0 {
            flat.iter().filter(|v| **v != 0.0).count()
        } else {
            flat.iter().filter(|v| v.is_nan() || v.abs() > eps).count()
        };
        let mut indptr: Vec<u64> = Vec::with_capacity(slab_rows + 1);
        let mut indices: Vec<u32> = Vec::with_capacity(nnz);
        let mut values: Vec<f32> = Vec::with_capacity(nnz);

        indptr.push(0);
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
                    // Retain NaN like scipy/anndata (the eps==0 branch above
                    // keeps it via `v != 0.0`); `NaN.abs() > eps` is false, so
                    // without the explicit guard NaN would be silently dropped.
                    if v.is_nan() || v.abs() > eps {
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

    fn max_slab_rows(&self) -> Option<u32> {
        if self.max_slab_rows == usize::MAX {
            None
        } else {
            Some(u32::try_from(self.max_slab_rows).unwrap_or(u32::MAX))
        }
    }

    fn per_worker_bytes(
        &self,
        shard_target_rows: u32,
        _modality_type: scx_format_io::modality::ModalityType,
    ) -> u64 {
        // The same cost model that sized the cap in `open_dense_streaming`.
        //
        // ⚠️ This deliberately does NOT apply a further multiplier. It used to
        // multiply by 2 "so the dispatcher knows fits-or-doesn't", but
        // `shard_target_rows` reaching here has already been clamped to
        // `max_slab_rows`, which was itself produced by taking a quarter of the
        // budget -- so the ×2 charged the same reserve a second time. The
        // arithmetic worked out to `per_worker ≈ budget/2`, hence
        // `outstanding_max = 2`, `granted_threads = 1`, and every budgeted
        // dense convert silently taking the sequential coordinator (§11.5).
        crate::budget::dense_slab_bytes(
            shard_target_rows as u64,
            self.n_vars,
            self.dtype.size_bytes() as u64,
        )
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
