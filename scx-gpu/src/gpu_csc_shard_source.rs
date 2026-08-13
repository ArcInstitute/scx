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
//! `scx-accel/src/diffexp/gpu.rs` for the consumer.
//!
//! [`RawGpuCscShardSource`] mirrors [`crate::gpu_shard_source::RawGpuShardSource`]'s
//! G3-shaped pipelining: a 2-slot pinned host ring, a dedicated copy
//! stream, a scoped worker thread that pre-decodes the next shard, and
//! event-driven handshake between the copy and compute streams. The
//! synchronous baseline from G4.3 has been replaced because it
//! serialised decode + H→D + compute and regressed wall time at
//! scale (smartseq2 / tabula) where it dominates the per-shard
//! overhead.

use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::safe::{CudaEvent, CudaSlice, CudaStream, CudaView};

use scx_format_io::shard_source::ColumnShardSource;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_validate::validate_csc_shard_for_gpu;
use crate::staging::PinnedCscSlot;

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

    /// Iterate over CSC shards whose `[col_start, col_end)` overlaps
    /// `col_range`, skipping all others **before** decode + upload.
    ///
    /// Default impl falls back to a full iteration with the overlap
    /// filter applied inside the callback (i.e., still decodes +
    /// uploads every shard but ignores non-overlapping ones).
    /// Implementors that can answer "does this shard overlap?" via
    /// cheap catalog metadata (e.g. [`RawGpuCscShardSource`]) override
    /// this to skip the decode entirely — material at scale (smartseq2
    /// / tabula) where most shards don't overlap a given gene chunk.
    fn for_each_gpu_csc_shard_in_range<F>(
        &mut self,
        col_range: Range<u32>,
        mut f: F,
    ) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        self.for_each_gpu_csc_shard(|idx, view| {
            if (view.col_end as u32) <= col_range.start || (view.col_start as u32) >= col_range.end
            {
                return Ok(());
            }
            f(idx, view)
        })
    }
}

/// Adapter from a CPU-side [`ColumnShardSource`] to [`GpuCscShardSource`]
/// with G3-shaped pipelining.
///
/// Owns the staging pipeline:
/// - A **2-slot ring** of [`PinnedCscSlot`]s. Two are required to fix the
///   same host-side reuse race the CSR `RawGpuShardSource` documents: with
///   pinned memory the H→D copy issued on `copy_stream` is truly
///   asynchronous, so the next iteration's `stage()` would otherwise
///   overwrite the pinned source buffer while the prior shard's DMA was
///   still in flight. The ring pings between the two slots; before
///   re-staging, the loop host-waits on the captured copy-stream event for
///   that slot.
/// - Three reusable device CSC buffers (`col_indptr`, `row_indices`,
///   `data`) with grow-only capacity tracking.
/// - A dedicated `copy_stream` and per-shard upload events for the
///   device-side handshake (compute waits on copy; next copy waits on
///   compute reads of the prior shard's device buffers).
///
/// The decode loop runs on a scoped worker thread that pre-decodes the
/// next shard while the main thread processes the current one.
///
/// CSC kernels in this pipeline (`csc_shard_pseudobulk_kernel`,
/// `csc_shard_to_gene_major_kernel`) are custom — no cuSPARSE descriptor
/// cache is needed.
///
/// Every shard is run through [`validate_csc_shard_for_gpu`] before it is
/// staged, the column-major counterpart of what the CSR pipeline has always
/// done. Until review §8.3 this path validated nothing at all, so the
/// CSC-direct DE route — the default whenever a sidecar exists — fed NaN,
/// duplicate `(cell, gene)` pairs and out-of-range cell ids straight to the
/// kernels.
pub struct RawGpuCscShardSource<'a> {
    dev: &'a GpuDevice,
    source: &'a (dyn ColumnShardSource + Sync),
    /// Shard indices already validated by this adapter, so a source iterated
    /// once per gene chunk pays the O(nnz) scans once per shard rather than
    /// once per chunk. Sound because `source` is a shared borrow held for the
    /// adapter's whole lifetime: the bytes behind a given shard index cannot
    /// change underneath it.
    validated: Vec<bool>,
    pinned: [PinnedCscSlot; 2],
    /// Per-pinned-slot copy-stream events captured immediately after the
    /// slot's most recent `upload_to`. Host-waited on before the slot is
    /// reused for the next `stage()` so the CPU never overwrites a buffer
    /// whose DMA is still in flight.
    pinned_events: [Option<CudaEvent>; 2],
    col_indptr: CudaSlice<i64>,
    row_indices: CudaSlice<i32>,
    data: CudaSlice<f32>,
    col_indptr_cap: usize,
    nnz_cap: usize,
    copy_stream: Arc<CudaStream>,
    n_obs: usize,
    n_vars: usize,
}

impl<'a> RawGpuCscShardSource<'a> {
    /// Construct with lazy-grow staging buffers and a dedicated copy
    /// stream when more than one shard is present.
    ///
    /// Deliberately does **not** pre-size from per-shard catalog stats:
    /// the trait doesn't expose them and the grow-on-demand path is
    /// cheap (one-time alloc per buffer, amortised across all shards).
    pub fn new(
        dev: &'a GpuDevice,
        source: &'a (dyn ColumnShardSource + Sync),
    ) -> Result<Self, GpuError> {
        let (n_obs, n_vars) = source.shape();
        let ctx = dev.context();
        let pinned = [PinnedCscSlot::new(ctx, 1, 1), PinnedCscSlot::new(ctx, 1, 1)];
        let col_indptr = dev.alloc_zeros::<i64>(1)?;
        let row_indices = dev.alloc_zeros::<i32>(1)?;
        let data = dev.alloc_zeros::<f32>(1)?;

        // Dedicated copy stream — used only when n_csc_shards > 1
        // (single-shard fast path reuses the compute stream).
        let copy_stream = if source.n_csc_shards() > 1 {
            dev.context()
                .new_stream()
                .map_err(|e| GpuError::StreamError(format!("new_stream: {e}")))?
        } else {
            dev.stream().clone()
        };

        Ok(Self {
            dev,
            source,
            validated: vec![false; source.n_csc_shards()],
            pinned,
            pinned_events: [None, None],
            col_indptr,
            row_indices,
            data,
            col_indptr_cap: 1,
            nnz_cap: 1,
            copy_stream,
            n_obs,
            n_vars,
        })
    }

    /// Number of shards this adapter has validated so far. Test-only: lets the
    /// caching test assert that a second pass over the same source re-scans
    /// nothing, and that a pass did not skip a shard it staged.
    #[cfg(test)]
    pub(crate) fn validated_count(&self) -> usize {
        self.validated.iter().filter(|v| **v).count()
    }

    /// Grow device buffers to fit a shard's `col_indptr_len` / `nnz`.
    /// No-op when current capacity already suffices.
    fn ensure_device_capacity(
        &mut self,
        col_indptr_len: usize,
        nnz: usize,
    ) -> Result<(), GpuError> {
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

    /// Shared driver for both trait methods. `indices` is the list of
    /// shard indices to process in order — either `0..n_csc_shards()`
    /// (for `for_each_gpu_csc_shard`) or the pre-filtered subset (for
    /// `for_each_gpu_csc_shard_in_range`).
    ///
    /// Single-shard fast path is taken when `indices.len() == 1` to
    /// avoid spinning up a worker thread + dedicated copy stream when
    /// there's nothing to overlap.
    fn run_shards<F>(&mut self, indices: Vec<usize>, mut f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        if indices.is_empty() {
            return Ok(());
        }

        // Single-shard fast path (no worker, no copy stream, no ring).
        // Safe because there is no successor `stage()` that could race
        // the in-flight DMA — once `f` returns, the caller's next host
        // action implicitly orders against the compute stream and the
        // pinned buffer is free to reuse.
        if indices.len() == 1 {
            let i = indices[0];
            let shard = self
                .source
                .read_csc_shard(i)
                .map_err(|e| GpuError::CudaError(format!("read_csc_shard({i}) failed: {e}")))?;
            let col_indptr_len = shard.indptr.len();
            let nnz = shard.data.len();
            let n_cols = shard.shape.1;
            if nnz == 0 || n_cols == 0 {
                return Ok(());
            }
            let (col_start, col_end) = self.source.csc_shard_col_range(i).ok_or_else(|| {
                GpuError::CudaError(format!("csc_shard_col_range({i}) returned None"))
            })?;
            let col_start = col_start as usize;
            let col_end = col_end as usize;

            if !self.validated.get(i).copied().unwrap_or(false) {
                validate_csc_shard_for_gpu(&shard, self.n_obs, col_start)?;
                if let Some(v) = self.validated.get_mut(i) {
                    *v = true;
                }
            }

            self.ensure_device_capacity(col_indptr_len, nnz)?;
            self.pinned[0].stage(&shard)?;
            self.pinned[0].upload_to(
                self.dev.stream(),
                &mut self.col_indptr,
                &mut self.row_indices,
                &mut self.data,
                col_indptr_len,
                nnz,
            )?;

            let view = GpuCscShardView {
                col_indptr: self.col_indptr.slice(..col_indptr_len),
                row_indices: self.row_indices.slice(..nnz),
                data: self.data.slice(..nnz),
                col_start,
                col_end,
                n_obs: self.n_obs,
            };
            return f(i, &view);
        }

        // Multi-shard pipelined path. Borrow split: hoist references to
        // inner fields up-front so the scoped thread closure can
        // capture `source` without going through `&mut self`.
        let source = self.source;
        let dev = self.dev;
        let pinned = &mut self.pinned;
        let pinned_events = &mut self.pinned_events;
        let copy_stream = &self.copy_stream;
        let compute_stream = dev.stream();
        let col_indptr_buf = &mut self.col_indptr;
        let row_indices_buf = &mut self.row_indices;
        let data_buf = &mut self.data;
        let col_indptr_cap = &mut self.col_indptr_cap;
        let nnz_cap = &mut self.nnz_cap;
        let n_obs = self.n_obs;
        let validated = &mut self.validated;

        // Local grow helper — same logic as `ensure_device_capacity`
        // but operates on the hoisted field references.
        let ensure_capacity_local = |col_indptr_buf: &mut CudaSlice<i64>,
                                     row_indices_buf: &mut CudaSlice<i32>,
                                     data_buf: &mut CudaSlice<f32>,
                                     col_indptr_cap: &mut usize,
                                     nnz_cap: &mut usize,
                                     col_indptr_len: usize,
                                     nnz: usize|
         -> Result<(), GpuError> {
            if col_indptr_len > *col_indptr_cap {
                let new_cap = col_indptr_len.next_power_of_two().max(col_indptr_len);
                *col_indptr_buf = dev.alloc_zeros::<i64>(new_cap)?;
                *col_indptr_cap = new_cap;
            }
            if nnz > *nnz_cap {
                let new_cap = nnz.next_power_of_two().max(nnz);
                *row_indices_buf = dev.alloc_zeros::<i32>(new_cap)?;
                *data_buf = dev.alloc_zeros::<f32>(new_cap)?;
                *nnz_cap = new_cap;
            }
            Ok(())
        };

        let scope_result = std::thread::scope(|scope| -> Result<(), GpuError> {
            use std::sync::mpsc;
            type Msg = Result<(usize, scx_sparse::ScxCsc, u32, u32), scx_format_io::ScxError>;
            let (tx, rx) = mpsc::sync_channel::<Msg>(1);

            let indices_for_worker = indices.clone();
            scope.spawn(move || {
                for &i in indices_for_worker.iter() {
                    let csc = match source.read_csc_shard(i) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            break;
                        }
                    };
                    let range = match source.csc_shard_col_range(i) {
                        Some(r) => r,
                        None => {
                            let _ = tx.send(Err(scx_format_io::ScxError::InvalidCatalog(format!(
                                "csc_shard_col_range({i}) returned None"
                            ))));
                            break;
                        }
                    };
                    if tx.send(Ok((i, csc, range.0, range.1))).is_err() {
                        break;
                    }
                }
            });

            let mut pinned_idx: usize = 0;
            while let Ok(msg) = rx.recv() {
                let (i, csc, col_start_u32, col_end_u32) = match msg {
                    Ok(v) => v,
                    Err(e) => {
                        return Err(GpuError::CudaError(format!(
                            "CSC read error during streaming decode: {e}"
                        )))
                    }
                };
                let col_indptr_len = csc.indptr.len();
                let nnz = csc.data.len();
                let n_cols = csc.shape.1;
                if nnz == 0 || n_cols == 0 {
                    continue;
                }
                let col_start = col_start_u32 as usize;
                let col_end = col_end_u32 as usize;

                if !validated.get(i).copied().unwrap_or(false) {
                    validate_csc_shard_for_gpu(&csc, n_obs, col_start)?;
                    if let Some(v) = validated.get_mut(i) {
                        *v = true;
                    }
                }

                // Host-side gate: if this pinned slot still has an
                // outstanding copy-stream event from a previous shard,
                // we must wait for that DMA to drain on the host before
                // overwriting the pinned buffer. The device-side gates
                // below only order device streams against each other;
                // they do not prevent the CPU from racing the DMA's
                // source memory.
                if let Some(evt) = pinned_events[pinned_idx].take() {
                    evt.synchronize()
                        .map_err(|e| GpuError::CudaError(format!("pinned event sync: {e}")))?;
                }

                ensure_capacity_local(
                    col_indptr_buf,
                    row_indices_buf,
                    data_buf,
                    col_indptr_cap,
                    nnz_cap,
                    col_indptr_len,
                    nnz,
                )?;

                pinned[pinned_idx].stage(&csc)?;
                pinned[pinned_idx].upload_to(
                    copy_stream,
                    col_indptr_buf,
                    row_indices_buf,
                    data_buf,
                    col_indptr_len,
                    nnz,
                )?;
                // Device-side gate (copy → compute): kernel reads must
                // observe the just-uploaded shard data.
                let upload_event = copy_stream
                    .record_event(None)
                    .map_err(|e| GpuError::CudaError(format!("record upload event: {e}")))?;
                compute_stream
                    .wait(&upload_event)
                    .map_err(|e| GpuError::CudaError(format!("compute wait: {e}")))?;
                // Stash the same event against this pinned slot so the
                // next iteration that recycles it can host-wait.
                pinned_events[pinned_idx] = Some(upload_event);

                let view = GpuCscShardView {
                    col_indptr: col_indptr_buf.slice(..col_indptr_len),
                    row_indices: row_indices_buf.slice(..nnz),
                    data: data_buf.slice(..nnz),
                    col_start,
                    col_end,
                    n_obs,
                };
                f(i, &view)?;

                // Device-side gate (compute → copy): the next shard's
                // upload (which writes into the device buffers) must
                // wait for the current shard's compute reads to finish.
                let compute_event = compute_stream
                    .record_event(None)
                    .map_err(|e| GpuError::CudaError(format!("record compute event: {e}")))?;
                copy_stream
                    .wait(&compute_event)
                    .map_err(|e| GpuError::CudaError(format!("copy wait: {e}")))?;

                pinned_idx ^= 1;
            }
            Ok(())
        });

        // Drain any remaining pinned events so the caller may safely
        // mutate or drop the pinned host buffers immediately after this
        // function returns. Cheap in the common case — by the time we
        // reach this point the DMAs are typically already complete.
        for evt in self.pinned_events.iter_mut() {
            if let Some(e) = evt.take() {
                e.synchronize()
                    .map_err(|err| GpuError::CudaError(format!("pinned drain sync: {err}")))?;
            }
        }

        scope_result
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

    fn for_each_gpu_csc_shard<F>(&mut self, f: F) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        let indices: Vec<usize> = (0..self.source.n_csc_shards()).collect();
        self.run_shards(indices, f)
    }

    fn for_each_gpu_csc_shard_in_range<F>(
        &mut self,
        col_range: Range<u32>,
        f: F,
    ) -> Result<(), GpuError>
    where
        F: FnMut(usize, &GpuCscShardView<'_>) -> Result<(), GpuError>,
    {
        // Pre-filter shard indices using the cheap O(1) catalog lookup,
        // so non-overlapping shards never get decoded or uploaded. At
        // smartseq2 / tabula scale with many CSC shards (13+) and small
        // gene chunks, most shards skip — eliminating decode + H→D for
        // them collapses cache thrashing (default `cache_shards=4`)
        // and the per-chunk fixed cost.
        let mut indices: Vec<usize> = Vec::new();
        for i in 0..self.source.n_csc_shards() {
            if let Some((s, e)) = self.source.csc_shard_col_range(i) {
                if e > col_range.start && s < col_range.end {
                    indices.push(i);
                }
            }
        }
        self.run_shards(indices, f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::ScxCsc;

    /// In-memory `ColumnShardSource` stub: one `ScxCsc` per shard, each
    /// covering a precomputed `(col_start, col_end)` global column
    /// range. `read_csc_columns` is unimplemented because the pipelined
    /// driver only calls `read_csc_shard` + `csc_shard_col_range`.
    struct InMemoryCscSource {
        shards: Vec<ScxCsc>,
        ranges: Vec<(u32, u32)>,
        n_obs: usize,
        n_vars: usize,
    }

    impl ColumnShardSource for InMemoryCscSource {
        fn n_csc_shards(&self) -> usize {
            self.shards.len()
        }
        fn n_obs(&self) -> usize {
            self.n_obs
        }
        fn n_vars(&self) -> usize {
            self.n_vars
        }
        fn read_csc_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsc> {
            Ok(self.shards[shard_idx].clone())
        }
        fn read_csc_columns(
            &self,
            _col_range: std::ops::Range<u32>,
        ) -> scx_format_io::Result<ScxCsc> {
            unimplemented!("test stub does not implement read_csc_columns")
        }
        fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
            self.ranges.get(shard_idx).copied()
        }
    }

    /// Build a tagged CSC shard with one nonzero per column at row
    /// `(col_local % n_obs)` and value `base + col_local`.
    fn make_csc(n_obs: usize, n_cols: usize, base: f32) -> ScxCsc {
        let mut indptr = vec![0i64];
        let mut indices = Vec::with_capacity(n_cols);
        let mut data = Vec::with_capacity(n_cols);
        for c in 0..n_cols {
            indices.push((c % n_obs) as i32);
            data.push(base + c as f32);
            indptr.push(indices.len() as i64);
        }
        ScxCsc::new_unchecked((n_obs, n_cols), indptr, indices, data)
    }

    /// `n_cols` columns holding rows `0..per_col` in strictly increasing
    /// order — enough nonzeros per column that a duplicate can be planted.
    fn make_dense_csc(n_obs: usize, n_cols: usize, per_col: usize) -> ScxCsc {
        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for c in 0..n_cols {
            for r in 0..per_col {
                indices.push(r as i32);
                data.push((c * per_col + r) as f32 + 1.0);
            }
            indptr.push(indices.len() as i64);
        }
        ScxCsc::new_unchecked((n_obs, n_cols), indptr, indices, data)
    }

    /// §8.3: the CSC staging path validated nothing, so a malformed sidecar
    /// reached the kernels — a NaN corrupting the radix sort, a duplicate
    /// racing one `slab` cell, an out-of-range row as an OOB device read.
    ///
    /// Each fixture must be rejected **before** anything is staged: the
    /// callback never fires, and the shard is not recorded as validated.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_csc_source_rejects_malformed_shards_before_staging() {
        let dev = require_gpu!();

        let nan = {
            let mut s = make_dense_csc(8, 3, 4);
            s.data[5] = f32::NAN;
            s
        };
        let duplicate = {
            let mut s = make_dense_csc(8, 3, 4);
            s.indices[5] = s.indices[4];
            s
        };
        let out_of_range = {
            let mut s = make_dense_csc(8, 3, 4);
            s.indices[5] = 8; // == n_obs
            s
        };

        for (label, shard) in [
            ("non-finite", nan),
            ("duplicate row", duplicate),
            ("out-of-range row", out_of_range),
        ] {
            let src = InMemoryCscSource {
                shards: vec![shard],
                ranges: vec![(0, 3)],
                n_obs: 8,
                n_vars: 3,
            };
            let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
            let mut calls = 0usize;
            let err = gpu
                .for_each_gpu_csc_shard(|_, _| {
                    calls += 1;
                    Ok(())
                })
                .unwrap_err();
            assert!(
                matches!(err, GpuError::InvalidShard(_)),
                "{label}: expected InvalidShard, got {err:?}"
            );
            assert_eq!(calls, 0, "{label}: must be rejected before staging");
            assert_eq!(
                gpu.validated_count(),
                0,
                "{label}: a rejected shard must not be recorded as validated"
            );
        }
    }

    /// Every shard the source stages is validated — including on the
    /// multi-shard pipelined path, which is a separate call site from the
    /// single-shard fast path and was the one §8.3's absent check hid in.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_csc_source_validates_every_staged_shard() {
        let dev = require_gpu!();
        let src = InMemoryCscSource {
            shards: vec![
                make_dense_csc(8, 3, 4),
                make_dense_csc(8, 4, 4),
                make_dense_csc(8, 3, 4),
            ],
            ranges: vec![(0, 3), (3, 7), (7, 10)],
            n_obs: 8,
            n_vars: 10,
        };
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
        gpu.for_each_gpu_csc_shard(|_, _| Ok(())).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(
            gpu.validated_count(),
            3,
            "every staged shard must be scanned"
        );

        // A second pass re-yields all three; the per-shard record is what keeps
        // the O(nnz) scans from repeating once per gene chunk.
        gpu.for_each_gpu_csc_shard(|_, _| Ok(())).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(gpu.validated_count(), 3);
    }

    /// A malformed shard in the middle of a multi-shard source aborts the
    /// iteration rather than being staged.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_csc_source_rejects_malformed_shard_on_the_pipelined_path() {
        let dev = require_gpu!();
        let bad = {
            let mut s = make_dense_csc(8, 4, 4);
            s.data[2] = f32::NAN;
            s
        };
        let src = InMemoryCscSource {
            shards: vec![make_dense_csc(8, 3, 4), bad, make_dense_csc(8, 3, 4)],
            ranges: vec![(0, 3), (3, 7), (7, 10)],
            n_obs: 8,
            n_vars: 10,
        };
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();
        let mut seen: Vec<usize> = Vec::new();
        let err = gpu
            .for_each_gpu_csc_shard(|idx, _| {
                seen.push(idx);
                Ok(())
            })
            .unwrap_err();
        assert!(
            matches!(err, GpuError::InvalidShard(_)),
            "expected InvalidShard, got {err:?}"
        );
        assert_eq!(seen, vec![0], "iteration must stop at the malformed shard");
    }

    /// Pipelined source yields the staged shards verbatim — dtoh of
    /// the view's `data` returns each shard's tagged values.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_raw_csc_source_round_trip() {
        let dev = require_gpu!();
        let shards = vec![make_csc(8, 3, 1.0), make_csc(8, 4, 100.0)];
        let src = InMemoryCscSource {
            shards,
            ranges: vec![(0, 3), (3, 7)],
            n_obs: 8,
            n_vars: 7,
        };
        let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();

        let mut seen: Vec<f32> = Vec::new();
        let mut col_ranges: Vec<(usize, usize)> = Vec::new();
        gpu.for_each_gpu_csc_shard(|_idx, view| {
            col_ranges.push((view.col_start, view.col_end));
            let mut host = vec![0.0f32; view.data.len()];
            dev.stream()
                .memcpy_dtoh(&view.data, &mut host)
                .map_err(|e| GpuError::CudaError(format!("dtoh: {e}")))?;
            seen.extend(host);
            Ok(())
        })
        .unwrap();
        dev.synchronize().unwrap();

        // Shard 0 = [1, 2, 3] cols, shard 1 = [100, 101, 102, 103] cols.
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0]);
        assert_eq!(col_ranges, vec![(0, 3), (3, 7)]);
    }

    /// Regression test for the pinned-host-reuse race that the 2-slot
    /// pinned ring + host-side event sync fixes.
    ///
    /// Mirrors `test_multi_shard_pinned_no_corruption` in
    /// `gpu_shard_source.rs`. Pre-fix, `pinned.stage(&csc)` for shard
    /// `i+1` could overwrite the pinned source buffer while shard
    /// `i`'s `memcpy_htod_async` was still in flight on `copy_stream`.
    /// The device-side event handshake only orders streams against
    /// each other — it doesn't host-block the CPU writer. With pinned
    /// memory the H→D copy is truly async, so the corruption is
    /// observable as cross-shard bleed.
    ///
    /// The test stresses the bug by:
    ///   - Using ≥3 shards so the ring cycles at least once.
    ///   - Sizing each shard to ~10⁴ nnz so the DMA isn't trivially short.
    ///   - Using a **device-only** consumer (per-shard `memcpy_dtod`
    ///     into a private capture buffer on the compute stream) — no
    ///     per-callback host-blocking dtoh that would mask the race.
    ///   - Repeating across a small outer loop to amplify the window.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn test_multi_shard_csc_pinned_no_corruption() {
        use cudarc::driver::safe::CudaSlice;

        let dev = require_gpu!();
        let n_obs = 32usize;
        let cols_per_shard = 10_000usize; // ~10k cols × 1 nnz/col per shard
        let n_shards = 5usize;

        // Build per-shard CSCs with a per-shard tag value: shard `s`'s
        // data is `[s*1e6 + col]`. Any cross-shard bleed shows up
        // numerically.
        let make_tagged_csc = |shard: usize| -> ScxCsc {
            let mut indptr = Vec::with_capacity(cols_per_shard + 1);
            let mut indices = Vec::with_capacity(cols_per_shard);
            let mut data = Vec::with_capacity(cols_per_shard);
            indptr.push(0i64);
            for c in 0..cols_per_shard {
                indices.push((c % n_obs) as i32);
                data.push(shard as f32 * 1.0e6 + c as f32);
                indptr.push(indices.len() as i64);
            }
            ScxCsc::new_unchecked((n_obs, cols_per_shard), indptr, indices, data)
        };

        for repeat in 0..5 {
            let shards: Vec<ScxCsc> = (0..n_shards).map(make_tagged_csc).collect();
            let ranges: Vec<(u32, u32)> = (0..n_shards)
                .map(|s| {
                    (
                        (s * cols_per_shard) as u32,
                        ((s + 1) * cols_per_shard) as u32,
                    )
                })
                .collect();
            let src = InMemoryCscSource {
                shards,
                ranges,
                n_obs,
                n_vars: n_shards * cols_per_shard,
            };
            let mut gpu = RawGpuCscShardSource::new(&dev, &src).unwrap();

            // Per-shard device capture buffers. Filled via dtod on the
            // compute stream inside each callback — strictly
            // device-side, so no host-blocking mask of the race.
            let mut captures: Vec<CudaSlice<f32>> = (0..n_shards)
                .map(|_| dev.alloc_zeros::<f32>(cols_per_shard).unwrap())
                .collect();

            // Track which shard idx fills which capture (the source
            // yields in order, so capture index == shard index, but
            // assert via the view's col_start).
            gpu.for_each_gpu_csc_shard(|_idx, view| {
                assert_eq!(view.data.len(), cols_per_shard, "shard nnz mismatch");
                let shard = view.col_start / cols_per_shard;
                dev.stream()
                    .memcpy_dtod(&view.data, &mut captures[shard])
                    .map_err(|e| GpuError::CudaError(format!("dtod: {e}")))?;
                Ok(())
            })
            .unwrap();
            // Single boundary sync — everything queued on
            // compute_stream (the per-shard dtod copies) must complete
            // before we read the captures back to the host.
            dev.synchronize().unwrap();

            for (idx, capture) in captures.iter().enumerate() {
                let mut host = vec![0.0f32; cols_per_shard];
                dev.stream().memcpy_dtoh(capture, &mut host).unwrap();
                dev.synchronize().unwrap();
                for (c, &v) in host.iter().enumerate() {
                    let expected = idx as f32 * 1.0e6 + c as f32;
                    assert!(
                        (v - expected).abs() < 0.5,
                        "repeat {repeat}, shard {idx}, col {c}: got {v}, expected {expected} \
                         (pinned-host-reuse race?)"
                    );
                }
            }
        }
    }
}
