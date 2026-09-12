//! The combined device CSR that every GPU decode path fills, and the one
//! placement that fills it.
//!
//! Six decode loops end the same way: allocate `nnz`-sized `indices`/`data`
//! buffers, host-assemble the tiny global `indptr`, then for each row-group
//! (or, in `gpu_csr_assemble`, each shard) copy its decoded buffers into
//! `combined[base .. base + len]` at a running offset. That last step was
//! written out five times for indices and five for values, so anything it
//! ought to check had to be checked five times over — and the crate's own
//! policy (`gpu_shard_source.rs`: return `InvalidShard` rather than rely on a
//! `debug_assert!` that vanishes in release) was met by none of them.
//!
//! [`CombinedCsr`] is that step written once. The arithmetic it enforces lives
//! CUDA-free next door in [`crate::csr_placement`], where a CPU host can test
//! it; this file is the thin device layer over it.
//!
//! # What the placement check is for
//!
//! Not the decoded lengths, mostly. Four of the six loops size a group's output
//! by allocating it (`alloc_zeros(g_nnz)`), and their inputs are already
//! length-exact — `decompress_frame` rejects a zstd frame that decompresses to
//! the wrong size, and `nvcomp::batch_decompress_concat` reads back nvcomp's
//! per-chunk `actual` and rejects a mismatch. A length check there is
//! tautological. It has content on exactly two paths: framed Scx1, where
//! `forbp_decode_gpu` sizes its output from the bitstream's own per-row
//! varints, and `gpu_csr_assemble`, which compares a shard's decode against the
//! catalog's declared stats. Both already checked.
//!
//! What **all six** lacked is the bounds check. `CudaSlice::slice_mut` is
//! `try_slice_mut(bounds).unwrap()`, so a `base + len` past the end of the
//! combined buffer panicked inside a library rather than returning
//! `InvalidShard`. Both operands descend from unauthenticated block-index and
//! shard-header fields. That upgrade is the reason to have one placement rather
//! than five, and it now reaches **all six** loops: framed Scx1 and framed
//! ShufDeltaZstd-sequential in `shard_decode.rs`, the pipelined and both nvcomp
//! loops in `shufdelta_gpu.rs`, and the per-shard concat in
//! `gpu_csr_assemble.rs`.

use cudarc::driver::safe::CudaSlice;

use scx_format_io::shard::clamped_reserve;

use crate::csr_placement::{check_coverage, check_placement, Placement};
use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::shard_decode::GpuCsr;

/// A device CSR under construction: the two `nnz`-sized device buffers, the
/// host-side global `indptr`, and a tally of what has actually been placed.
pub(crate) struct CombinedCsr {
    /// Column indices, `nnz` elements. Filled group by group.
    indices: CudaSlice<i32>,
    /// Values, `nnz` elements. Filled group by group.
    data: CudaSlice<f32>,
    /// The global row pointer, assembled on the host and uploaded by
    /// [`CombinedCsr::finish`]. Public to the crate because the callers hand it
    /// straight to `prescan_framed_group_indptr`, which appends each group's
    /// rebased tail.
    pub(crate) indptr: Vec<i64>,
    nnz: usize,
    n_rows: usize,
    /// `(base, len)` of every unit placed into `indices`, in arrival order.
    /// `finish` sorts and walks these to prove an exact tiling — a running sum
    /// would accept an assembly that overlaps in one place and leaves a hole in
    /// another. Kept separate from `placed_values` because the cross-shard
    /// nvcomp path fills the two in different passes over different chunks, so
    /// a dropped values chunk leaves indices fully tiled.
    placed_indices: Vec<(usize, usize)>,
    placed_values: Vec<(usize, usize)>,
}

impl CombinedCsr {
    /// Allocate the combined buffers for an `n_rows × ?` CSR with `nnz`
    /// non-zeros.
    ///
    /// `indptr_encoded_bytes` is the encoded indptr sub-stream the host decode
    /// will draw from. `n_rows` is unauthenticated header data, so the host
    /// reservation goes through [`clamped_reserve`] rather than
    /// `Vec::with_capacity`, which calls `handle_alloc_error` and *aborts* —
    /// a ~45 KB block index can declare 134M rows. The device buffers keep the
    /// exact `nnz`: `alloc_zeros` surfaces an over-large request as a `Result`.
    pub(crate) fn new(
        dev: &GpuDevice,
        nnz: usize,
        n_rows: usize,
        indptr_encoded_bytes: usize,
    ) -> Result<Self, GpuError> {
        let mut indptr: Vec<i64> =
            Vec::with_capacity(clamped_reserve(n_rows + 1, indptr_encoded_bytes, 8));
        indptr.push(0);
        Self::with_indptr(dev, nnz, n_rows, indptr)
    }

    /// Allocate the combined buffers around an `indptr` the caller has already
    /// assembled, complete with its leading `0`.
    ///
    /// The cross-shard nvcomp path builds its global `indptr` during the
    /// pre-scan that flattens every shard's row groups, before it knows the
    /// totals this builder needs — so it hands the finished vector over rather
    /// than reserving a second one. It needs no clamp: that vector grew by
    /// `push` as real decoded data arrived, never by reserving against a
    /// declared count.
    pub(crate) fn with_indptr(
        dev: &GpuDevice,
        nnz: usize,
        n_rows: usize,
        indptr: Vec<i64>,
    ) -> Result<Self, GpuError> {
        Ok(Self {
            indices: dev.alloc_zeros::<i32>(nnz)?,
            data: dev.alloc_zeros::<f32>(nnz)?,
            indptr,
            nnz,
            n_rows,
            placed_indices: Vec::new(),
            placed_values: Vec::new(),
        })
    }

    /// Copy one unit's decoded **indices** into `[base, base + len)`.
    pub(crate) fn place_indices(
        &mut self,
        dev: &GpuDevice,
        at: Placement<'_>,
        src: &CudaSlice<i32>,
    ) -> Result<(), GpuError> {
        let Placement {
            base,
            len,
            op,
            index,
        } = at;
        check_placement(at, src.len(), len, self.nnz)?;
        // The check ran; a zero-length unit has nothing to copy. Placing one is
        // still recorded, so a caller that hands over an empty unit keeps its
        // length checked instead of skipping the call entirely.
        if len == 0 {
            self.placed_indices.push((base, 0));
            return Ok(());
        }
        let mut dst = self
            .indices
            .try_slice_mut(base..base + len)
            .ok_or_else(|| {
                GpuError::InvalidShard(format!(
                    "{op} {index} indices: could not view combined[{base}..{}]",
                    base + len
                ))
            })?;
        dev.stream()
            .memcpy_dtod(src, &mut dst)
            .map_err(|e| GpuError::CudaError(format!("dtod indices ({op} {index}): {e}")))?;
        self.placed_indices.push((base, len));
        Ok(())
    }

    /// Copy one unit's decoded **values** into `[base, base + len)`. Sibling of
    /// [`CombinedCsr::place_indices`]; separate because the cross-shard nvcomp
    /// path frees its indices plane buffer before allocating the values one and
    /// so cannot hold both borrows at once.
    pub(crate) fn place_values(
        &mut self,
        dev: &GpuDevice,
        at: Placement<'_>,
        src: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        let Placement {
            base,
            len,
            op,
            index,
        } = at;
        check_placement(at, len, src.len(), self.nnz)?;
        // The check ran; a zero-length unit has nothing to copy. Placing one is
        // still recorded, so a caller that hands over an empty unit keeps its
        // length checked instead of skipping the call entirely.
        if len == 0 {
            self.placed_values.push((base, 0));
            return Ok(());
        }
        let mut dst = self.data.try_slice_mut(base..base + len).ok_or_else(|| {
            GpuError::InvalidShard(format!(
                "{op} {index} values: could not view combined[{base}..{}]",
                base + len
            ))
        })?;
        dev.stream()
            .memcpy_dtod(src, &mut dst)
            .map_err(|e| GpuError::CudaError(format!("dtod data ({op} {index}): {e}")))?;
        self.placed_values.push((base, len));
        Ok(())
    }

    /// Place both halves of one unit at the same offset.
    pub(crate) fn place(
        &mut self,
        dev: &GpuDevice,
        at: Placement<'_>,
        indices: &CudaSlice<i32>,
        data: &CudaSlice<f32>,
    ) -> Result<(), GpuError> {
        self.place_indices(dev, at, indices)?;
        self.place_values(dev, at, data)
    }

    /// Upload the assembled `indptr` and hand back the finished [`GpuCsr`].
    ///
    /// Checks first that the units actually placed cover the whole matrix.
    /// Without it a decode that skipped a unit returns a full-shaped result
    /// whose gap is filled with `alloc_zeros`'s zeros — the right shape, the
    /// wrong matrix, and nothing downstream able to tell. This is the same
    /// class as the streaming PCA drive's row-coverage check; here it is
    /// defence in depth rather than a known gap, since the framed paths also
    /// check `Σ span.nnz` against the header before they start.
    ///
    /// Synchronizes: the per-unit decode kernels and the `memcpy_dtod` copies
    /// are queued async on `dev.stream()`, and a downstream cuPy consumer
    /// adopting these buffers must not race them.
    pub(crate) fn finish(
        self,
        dev: &GpuDevice,
        n_cols: usize,
        what: &str,
    ) -> Result<GpuCsr, GpuError> {
        self.finish_with_indptr(dev, n_cols, what)
            .map(|(csr, _indptr)| csr)
    }

    /// [`finish`](Self::finish), also handing back the assembled **host**
    /// indptr instead of dropping it.
    ///
    /// The host vector is built either by `CombinedCsr::with_indptr`'s caller or
    /// by the framed paths' `prescan_*_group_indptr`, uploaded once here, and
    /// then — before this existed — discarded. `gpu_csr_assemble`'s multi-shard
    /// loop wanted exactly that vector and used to recover it with a per-shard
    /// `dtoh_copy`, which on pageable host memory is a **host-synchronous**
    /// copy: one full pipeline drain per shard, purely to read back something
    /// the host had just computed.
    pub(crate) fn finish_with_indptr(
        mut self,
        dev: &GpuDevice,
        n_cols: usize,
        what: &str,
    ) -> Result<(GpuCsr, Vec<i64>), GpuError> {
        check_coverage(
            &mut self.placed_indices,
            &mut self.placed_values,
            self.nnz,
            what,
        )?;
        let d_indptr = dev.htod_copy(&self.indptr)?;
        dev.synchronize()?;
        let csr = GpuCsr::new(
            d_indptr,
            self.indices,
            self.data,
            (self.n_rows, n_cols),
            what,
        )?;
        Ok((csr, self.indptr))
    }
}
