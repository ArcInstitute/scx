//! Sort-on-convert support: a permuted
//! row-gather adapter over the streaming X/layer readers.
//!
//! [`PermutedCsrReader`] wraps any [`IndexedCsrShardStream`] plus a
//! permutation `perm[output_row] = source_row_id`, and presents the same
//! [`CsrShardStream`] / [`IndexedCsrShardStream`] interface — so the existing
//! `run_streaming_writer_coordinator` drives it unchanged (parallel when
//! libhdf5 is threadsafe, sequential otherwise) and the non-sort convert path
//! stays byte-identical.
//!
//! `read_range(out_start, n)` returns output rows `[out_start, out_start+n)` in
//! sorted order: it coalesces the requested source ids into contiguous runs
//! (so each underlying HDF5 read stays a contiguous slice), reads each run via
//! the inner reader, and scatters the rows back into output order. Peak memory
//! is one output shard plus the shared `perm`. A fully-shuffled categorical
//! sort degrades to ~one-row runs (correct and bounded-memory, but more, smaller
//! reads) — the honest convert-time cost; the random-access source still beats
//! the standalone decode+spill engine.

use std::sync::Arc;

use arrow::array::{RecordBatch, UInt64Array};
use scx_format_io::modality::ModalityType;

use crate::pipeline::ConvertError;
use crate::stream::{CsrShardStream, IndexedCsrShardStream, StreamedCsrShard};

/// Compute the obs-axis sort permutation `perm[output_row] = source_row_id`
/// using the shared `scx_ops::sort` core.
pub(crate) fn compute_sort_perm(
    obs: &RecordBatch,
    by: &[String],
    reverse: bool,
) -> Result<Vec<u64>, ConvertError> {
    let schema = obs.schema();
    let extractor = scx_ops::sort::SortKeyExtractor::new(&schema, by, reverse)
        .map_err(|e| ConvertError::Other(format!("sort key error: {e}")))?;
    let rows = extractor
        .rows(obs)
        .map_err(|e| ConvertError::Other(format!("sort key error: {e}")))?;
    Ok(scx_ops::sort::stable_argsort(&rows, 0))
}

/// Reorder every column of `batch` by `perm` (arrow `take`). Used to permute
/// obs (and obsm) rows into sorted order.
pub(crate) fn take_record_batch(
    batch: &RecordBatch,
    perm: &[u64],
) -> Result<RecordBatch, ConvertError> {
    let indices = UInt64Array::from(perm.to_vec());
    let mut cols = Vec::with_capacity(batch.num_columns());
    for col in batch.columns() {
        cols.push(arrow::compute::take(col, &indices, None)?);
    }
    Ok(RecordBatch::try_new(batch.schema(), cols)?)
}

/// Permuted-row gather adapter over a streaming CSR/dense reader.
pub(crate) struct PermutedCsrReader {
    inner: Box<dyn IndexedCsrShardStream>,
    /// `perm[output_row] = source_row_id`; length == `n_obs`.
    perm: Arc<Vec<u64>>,
    /// Cursor over output rows for the sequential [`CsrShardStream`] path.
    cursor: u64,
    source_name: String,
}

impl PermutedCsrReader {
    pub(crate) fn new(inner: Box<dyn IndexedCsrShardStream>, perm: Arc<Vec<u64>>) -> Self {
        let source_name = inner.source_matrix_name().to_string();
        Self {
            inner,
            perm,
            cursor: 0,
            source_name,
        }
    }

    /// Gather output rows `[out_start, out_start + n_rows)` in output order.
    fn gather(&self, out_start: u64, n_rows: u32) -> Result<StreamedCsrShard, ConvertError> {
        let n = n_rows as usize;
        let start = out_start as usize;
        let want = &self.perm[start..start + n];

        // (output-local index, source id), sorted by source id so we can
        // coalesce contiguous source runs into single inner reads.
        let mut order: Vec<(usize, u64)> = want.iter().copied().enumerate().collect();
        order.sort_unstable_by_key(|&(_, src)| src);

        // Cap each inner read at the reader's slab limit: a long contiguous
        // source run (an already-sorted block, or a large categorical mapping
        // to a near-`n_obs` run) must not exceed a dense reader's max hyperslab
        // — otherwise `DenseXStreamReader::read_range` hard-errors. The run
        // shards together still hold only one output block's worth of rows, so
        // peak memory stays ~one output shard.
        let max_run = self
            .inner
            .max_slab_rows()
            .map(|m| m.max(1) as usize)
            .unwrap_or(usize::MAX);

        let mut shards: Vec<StreamedCsrShard> = Vec::new();
        // For each output-local row: (index into `shards`, row index within it).
        let mut row_sources: Vec<(usize, usize)> = vec![(0, 0); n];
        let mut row_lengths: Vec<usize> = vec![0; n];
        let mut n_cols = self.inner.n_vars() as u32;

        let mut i = 0;
        while i < order.len() {
            let run_start_src = order[i].1;
            // extend the run while source ids stay strictly consecutive, but
            // never beyond the inner reader's slab cap
            let mut j = i + 1;
            while j < order.len() && order[j].1 == order[j - 1].1 + 1 && (j - i) < max_run {
                j += 1;
            }
            let run_len = (j - i) as u32;
            let shard = self.inner.read_range(run_start_src, run_len)?;
            n_cols = shard.n_cols;
            let shard_idx = shards.len();
            for k in 0..run_len as usize {
                let out_local = order[i + k].0;
                row_sources[out_local] = (shard_idx, k);
                row_lengths[out_local] = (shard.indptr[k + 1] - shard.indptr[k]) as usize;
            }
            shards.push(shard);
            i = j;
        }

        // Pre-allocate the flat output and copy each row directly from its
        // source shard (no per-row Vec allocations).
        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u64);
        let mut total_nnz = 0usize;
        for &len in &row_lengths {
            total_nnz += len;
            indptr.push(total_nnz as u64);
        }
        let mut indices = vec![0u32; total_nnz];
        let mut values = vec![0.0f32; total_nnz];
        for r in 0..n {
            let (s_idx, row_idx) = row_sources[r];
            let shard = &shards[s_idx];
            let lo = shard.indptr[row_idx] as usize;
            let hi = shard.indptr[row_idx + 1] as usize;
            let out_lo = indptr[r] as usize;
            let out_hi = indptr[r + 1] as usize;
            indices[out_lo..out_hi].copy_from_slice(&shard.indices[lo..hi]);
            values[out_lo..out_hi].copy_from_slice(&shard.values[lo..hi]);
        }

        Ok(StreamedCsrShard {
            row_start: out_start,
            n_rows,
            n_cols,
            indptr,
            indices,
            values,
            source_name: Some(self.source_name.clone()),
            duplicates_merged: 0,
        })
    }
}

impl CsrShardStream for PermutedCsrReader {
    fn n_obs(&self) -> u64 {
        self.perm.len() as u64
    }
    fn n_vars(&self) -> u64 {
        self.inner.n_vars()
    }
    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }
    fn next_csr_shard(
        &mut self,
        target_rows: usize,
    ) -> Result<Option<StreamedCsrShard>, ConvertError> {
        let total = self.perm.len() as u64;
        if self.cursor >= total {
            return Ok(None);
        }
        let n = (total - self.cursor).min(target_rows.max(1) as u64) as u32;
        let shard = self.gather(self.cursor, n)?;
        self.cursor += n as u64;
        Ok(Some(shard))
    }
    fn as_indexed(&self) -> Option<&dyn IndexedCsrShardStream> {
        Some(self)
    }
}

impl IndexedCsrShardStream for PermutedCsrReader {
    fn n_obs(&self) -> u64 {
        self.perm.len() as u64
    }
    fn n_vars(&self) -> u64 {
        self.inner.n_vars()
    }
    fn source_matrix_name(&self) -> &str {
        &self.source_name
    }
    fn read_range(&self, row_start: u64, n_rows: u32) -> Result<StreamedCsrShard, ConvertError> {
        self.gather(row_start, n_rows)
    }
    fn max_slab_rows(&self) -> Option<u32> {
        self.inner.max_slab_rows()
    }
    fn per_worker_bytes(&self, shard_target_rows: u32, modality_type: ModalityType) -> u64 {
        self.inner
            .per_worker_bytes(shard_target_rows, modality_type)
    }
}
