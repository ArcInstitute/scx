//! Sort-on-convert support (SCX-SORT-SPEC §5, Phase 2): a permuted
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

        let mut row_indices: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut row_values: Vec<Vec<f32>> = vec![Vec::new(); n];
        let mut n_cols = self.inner.n_vars() as u32;

        let mut i = 0;
        while i < order.len() {
            let run_start_src = order[i].1;
            // extend the run while source ids stay strictly consecutive
            let mut j = i + 1;
            while j < order.len() && order[j].1 == order[j - 1].1 + 1 {
                j += 1;
            }
            let run_len = (j - i) as u32;
            let shard = self.inner.read_range(run_start_src, run_len)?;
            n_cols = shard.n_cols;
            for k in 0..run_len as usize {
                let out_local = order[i + k].0;
                let lo = shard.indptr[k] as usize;
                let hi = shard.indptr[k + 1] as usize;
                row_indices[out_local] = shard.indices[lo..hi].to_vec();
                row_values[out_local] = shard.values[lo..hi].to_vec();
            }
            i = j;
        }

        let mut indptr = Vec::with_capacity(n + 1);
        indptr.push(0u64);
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..n {
            indices.extend_from_slice(&row_indices[r]);
            values.extend_from_slice(&row_values[r]);
            indptr.push(indices.len() as u64);
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
