//! Device-resident CSC shard view + source trait (parallel hierarchy to the
//! CSR `GpuShardSource`).
//!
//! CSC sidecars store gene-major data (`docs/format.md` section 5,
//! `b"SCXS"` shard header with `shard_type = 1`). Each on-disk shard
//! covers a contiguous column range `[col_start, col_end)` × all rows
//! (column-only sharding; multi-shard via `--csc-cols-per-shard`).
//!
//! The view + source defined here are the GPU-side mirror of
//! `scx-format::shard_source::ColumnShardSource`. Built for the G4 v3 DE
//! pipeline (`pdex_ref_gpu_chunked_v3_csc`) — see
//! `scx-accel/src/diffexp_gpu.rs` for the consumer.
//!
//! Current implementation is **synchronous**: each `for_each_gpu_csc_shard`
//! iteration calls `read_csc_shard` then `htod_copy` for the three CSC
//! arrays. No pinned host staging, no dedicated copy stream, no worker
//! pipelining. This matches `RawGpuShardSource`'s pre-G3 shape and is the
//! correct baseline for G4.3 — pipelining is a separate optimization once
//! the dense-elimination perf gain is verified.

use cudarc::driver::{CudaSlice, CudaView};

use scx_format::shard_source::ColumnShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;

/// Borrowed CSC view backed by a device-resident CSC shard slot.
///
/// Column-major: `col_indptr` indexes into `row_indices` / `data` per
/// column. `row_indices` stores **global** row (cell) ids — the cell
/// numbering is consistent across all shards in the file. This matches the
/// on-disk encoding documented in `docs/format.md` § 5 (`shard_type = 1`).
///
/// All fields are exact-sized: `col_indptr.len() == n_cols_in_shard + 1`,
/// `row_indices.len() == data.len() == nnz`. Callers can pass `&col_indptr`,
/// `&row_indices`, `&data` directly to kernel launches.
pub struct GpuCscShardView<'a> {
    /// `[n_cols_in_shard + 1]` cumulative offsets (CSC-style).
    pub col_indptr: CudaView<'a, i64>,
    /// `[nnz]` global row (cell) indices.
    pub row_indices: CudaView<'a, i32>,
    /// `[nnz]` values.
    pub data: CudaView<'a, f32>,
    /// First global column id covered by this shard (inclusive).
    pub col_start: usize,
    /// One-past-last global column id (exclusive); `col_end - col_start` is
    /// the shard's column count.
    pub col_end: usize,
    /// Cell count (n_obs) — same across all shards in a file.
    pub n_obs: usize,
}

impl<'a> GpuCscShardView<'a> {
    /// Number of CSC columns held by this shard.
    pub fn n_cols(&self) -> usize {
        self.col_end - self.col_start
    }

    /// nnz from the view's data length (matches `col_indptr[n_cols]`).
    pub fn nnz(&self) -> usize {
        self.data.len()
    }
}

/// Sequence of GPU-resident CSC shards.
///
/// Parallel hierarchy to [`crate::gpu_shard_source::GpuShardSource`]; CSC is
/// not part of that trait because the slot layouts are incompatible (CSR
/// stores row-major indptr + col indices; CSC stores col-major indptr +
/// row indices).
pub trait GpuCscShardSource {
    /// Number of CSC shards in this source.
    fn n_csc_shards(&self) -> usize;

    /// Total observation count (rows). Same across all shards by CSC's
    /// column-only sharding invariant.
    fn n_obs(&self) -> usize;

    /// Variable count (columns) across all shards combined.
    fn n_vars(&self) -> usize;

    /// `(n_obs, n_vars)`.
    fn shape(&self) -> (usize, usize) {
        (self.n_obs(), self.n_vars())
    }

    /// Iterate over each non-empty CSC shard, invoking the callback with
    /// `(shard_idx, &GpuCscShardView)` positioned at the live shard.
    ///
    /// The view is invalidated when the iterator advances to the next
    /// shard (device slots are mutated in place).
    fn for_each_gpu_csc_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>;
}

/// Adapter from a CPU-side [`ColumnShardSource`] to [`GpuCscShardSource`].
///
/// Synchronous baseline: per shard, decode → `htod_copy` → callback. No
/// pipelining or pinned staging. Used by `pdex_ref_gpu_chunked_v3_csc`.
///
/// Reusable device slots are grow-only: `col_indptr` to fit
/// `n_cols_in_shard + 1` i64 entries; `row_indices` / `data` to fit `nnz`
/// per shard.
pub struct RawGpuCscShardSource<'a> {
    dev: &'a GpuDevice,
    source: &'a dyn ColumnShardSource,
    col_indptr: CudaSlice<i64>,
    row_indices: CudaSlice<i32>,
    data: CudaSlice<f32>,
    col_indptr_cap: usize,
    nnz_cap: usize,
    n_obs: usize,
    n_vars: usize,
}

impl<'a> RawGpuCscShardSource<'a> {
    /// Construct with lazy-grow staging.
    pub fn new(dev: &'a GpuDevice, source: &'a dyn ColumnShardSource) -> Result<Self, GpuError> {
        let (n_obs, n_vars) = source.shape();
        // Seed with size-1 slots; grow on first shard.
        let col_indptr = dev.alloc_zeros::<i64>(1)?;
        let row_indices = dev.alloc_zeros::<i32>(1)?;
        let data = dev.alloc_zeros::<f32>(1)?;
        Ok(Self {
            dev,
            source,
            col_indptr,
            row_indices,
            data,
            col_indptr_cap: 1,
            nnz_cap: 1,
            n_obs,
            n_vars,
        })
    }

    fn ensure_capacity(&mut self, col_indptr_len: usize, nnz: usize) -> Result<(), GpuError> {
        if col_indptr_len > self.col_indptr_cap {
            let new_cap = col_indptr_len.next_power_of_two().max(col_indptr_len);
            self.col_indptr = self.dev.alloc_zeros::<i64>(new_cap)?;
            self.col_indptr_cap = new_cap;
        }
        if nnz > self.nnz_cap {
            let new_cap = nnz.next_power_of_two().max(nnz);
            self.row_indices = self.dev.alloc_zeros::<i32>(new_cap)?;
            self.data = self.dev.alloc_zeros::<f32>(new_cap)?;
            self.nnz_cap = new_cap;
        }
        Ok(())
    }
}

impl<'a> GpuCscShardSource for RawGpuCscShardSource<'a> {
    fn n_csc_shards(&self) -> usize {
        self.source.n_csc_shards()
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn n_vars(&self) -> usize {
        self.n_vars
    }

    fn for_each_gpu_csc_shard<F>(&mut self, mut f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        let n = self.source.n_csc_shards();
        for idx in 0..n {
            let shard = self
                .source
                .read_csc_shard(idx)
                .map_err(|e| GpuError::CudaError(format!("read_csc_shard({idx}) failed: {e}")))?;
            let nnz = shard.data.len();
            let col_indptr_len = shard.indptr.len();
            let n_cols = shard.shape.1;
            if nnz == 0 || n_cols == 0 {
                continue;
            }
            let (col_start, col_end) = match self.source.csc_shard_col_range(idx) {
                Some((s, e)) => (s as usize, e as usize),
                None => {
                    return Err(GpuError::CudaError(format!(
                        "csc_shard_col_range({idx}) returned None"
                    )));
                }
            };

            self.ensure_capacity(col_indptr_len, nnz)?;

            // htod copies; the dev.stream() default-stream-ordering is
            // sufficient for the synchronous callback shape.
            let stream = self.dev.stream();
            stream
                .memcpy_htod(
                    &shard.indptr,
                    &mut self.col_indptr.slice_mut(..col_indptr_len),
                )
                .map_err(|e| GpuError::CudaError(format!("htod col_indptr: {e}")))?;
            stream
                .memcpy_htod(&shard.indices, &mut self.row_indices.slice_mut(..nnz))
                .map_err(|e| GpuError::CudaError(format!("htod row_indices: {e}")))?;
            stream
                .memcpy_htod(&shard.data, &mut self.data.slice_mut(..nnz))
                .map_err(|e| GpuError::CudaError(format!("htod data: {e}")))?;

            let view = GpuCscShardView {
                col_indptr: self.col_indptr.slice(..col_indptr_len),
                row_indices: self.row_indices.slice(..nnz),
                data: self.data.slice(..nnz),
                col_start,
                col_end,
                n_obs: self.n_obs,
            };
            f(idx, &view)?;
        }
        Ok(())
    }
}
