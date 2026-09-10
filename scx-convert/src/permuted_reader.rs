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
    /// Price the shard this reader will actually **emit**, not the source
    /// windows the inner reader would have read.
    ///
    /// Delegating was an under-count with no bound: the CSR override scans
    /// source-aligned windows `[0, t), [t, 2t), …` and takes their maximum nnz,
    /// but a permutation can collect rows that were spread across those windows
    /// into one output shard. A categorical sort that groups the deepest cells
    /// together does exactly that, and the estimate the derate then works from
    /// describes a partition that is never read.
    ///
    /// So when the inner reader can hand out its `indptr`
    /// ([`IndexedCsrShardStream::source_row_indptr`]) this walks the *output*
    /// rows through `perm` and takes a **sliding** window maximum. Sliding, not
    /// aligned, because this method is asked for one width and serves two
    /// partitions: the fixed path's aligned `[0, t), [t, 2t), …` (for which
    /// sliding is exact at the maximum) and the grouped path's variable
    /// group-aligned ranges, for which the caller passes only the largest row
    /// count — any contiguous range of at most `shard_target_rows` rows is
    /// covered by some window of exactly that width, so the sliding maximum
    /// bounds every one of them. One O(n_obs) pass over resident memory, no I/O.
    ///
    /// Readers that cannot answer (`None`) still delegate; that is the dense
    /// reader, whose own override sizes a slab and is permutation-independent.
    ///
    /// The **gather copy** is deliberately not an added term. Inside
    /// [`Self::gather`] the source runs and the assembled output are both live
    /// (~16 B/nnz), but the runs drop at the end of `gather` and only the
    /// output reaches the encode — so that peak sits under the 48 B/nnz whole
    /// phase [`crate::budget::shard_working_set_bytes`] already charges, and
    /// adding it would double-count the way §11.5 did.
    fn per_worker_bytes(&self, shard_target_rows: u32, modality_type: ModalityType) -> u64 {
        let Some(indptr) = self.inner.source_row_indptr() else {
            return self
                .inner
                .per_worker_bytes(shard_target_rows, modality_type);
        };
        let n = self.perm.len();
        if n == 0 || indptr.len() < 2 {
            return self
                .inner
                .per_worker_bytes(shard_target_rows, modality_type);
        }
        let t = (shard_target_rows.max(1) as usize).min(n);
        let row_nnz = |out_row: usize| -> u64 {
            let src = self.perm[out_row] as usize;
            match (indptr.get(src), indptr.get(src + 1)) {
                (Some(&lo), Some(&hi)) => hi.saturating_sub(lo).max(0) as u64,
                _ => 0,
            }
        };
        let mut window: u64 = (0..t).map(row_nnz).sum();
        let mut max_nnz = window;
        for i in t..n {
            window = window
                .saturating_add(row_nnz(i))
                .saturating_sub(row_nnz(i - t));
            max_nnz = max_nnz.max(window);
        }
        crate::budget::shard_working_set_bytes(max_nnz, t as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Inner reader with a resident `indptr` and nothing else — enough to
    /// exercise both estimators without libhdf5. `read_range` is never called.
    struct IndptrOnlyReader {
        indptr: Vec<i64>,
        n_vars: u64,
    }

    impl IndexedCsrShardStream for IndptrOnlyReader {
        fn n_obs(&self) -> u64 {
            (self.indptr.len() - 1) as u64
        }
        fn n_vars(&self) -> u64 {
            self.n_vars
        }
        fn source_matrix_name(&self) -> &str {
            "X"
        }
        fn read_range(
            &self,
            _row_start: u64,
            _n_rows: u32,
        ) -> Result<StreamedCsrShard, ConvertError> {
            unreachable!("the estimator does no I/O")
        }
        fn source_row_indptr(&self) -> Option<&[i64]> {
            Some(&self.indptr)
        }
        fn per_worker_bytes(&self, shard_target_rows: u32, _m: ModalityType) -> u64 {
            // The same source-aligned window scan `XStreamReader` runs.
            let n_obs = self.indptr.len() - 1;
            let t = (shard_target_rows.max(1) as usize).min(n_obs);
            let mut max_nnz = 0u64;
            let mut start = 0usize;
            while start < n_obs {
                let end = (start + t).min(n_obs);
                max_nnz = max_nnz.max((self.indptr[end] - self.indptr[start]) as u64);
                start = end;
            }
            crate::budget::shard_working_set_bytes(max_nnz, t as u64)
        }
    }

    /// Four rows of 1, 9, 1, 9 nnz. Source-aligned 2-row windows both hold 10,
    /// so the delegated estimate says 10 — but the permutation that puts the
    /// two deep rows in one output shard emits 18.
    fn deep_rows_spread_across_windows() -> IndptrOnlyReader {
        IndptrOnlyReader {
            indptr: vec![0, 1, 10, 11, 20],
            n_vars: 32,
        }
    }

    #[test]
    fn permuted_estimate_prices_the_output_shard_not_the_source_windows() {
        let inner = deep_rows_spread_across_windows();
        let delegated = inner.per_worker_bytes(2, ModalityType::Rna);
        assert_eq!(delegated, crate::budget::shard_working_set_bytes(10, 2));

        // perm groups the deep rows (source 1 and 3) into output rows [0, 2).
        let reader = PermutedCsrReader::new(Box::new(inner), Arc::new(vec![1, 3, 0, 2]));
        let priced = reader.per_worker_bytes(2, ModalityType::Rna);
        assert_eq!(
            priced,
            crate::budget::shard_working_set_bytes(18, 2),
            "the permuted estimate must price the 18-nnz output shard"
        );
        assert!(
            priced > delegated,
            "delegating under-priced this permutation ({delegated} < {priced}), which is the \
             defect: the derate would have spawned workers against a partition never read"
        );
    }

    #[test]
    fn permuted_estimate_bounds_every_grouped_range_of_that_width() {
        // The grouped path passes only the largest range's row count, so the
        // estimate must cover any contiguous output range of that width — the
        // sliding maximum, not the aligned one. Here output rows are ordered
        // 1, 3, 0, 2 (nnz 9, 9, 1, 1): the aligned 2-row windows are 18 and 2,
        // and the worst *sliding* window is also 18, so a 3-row range (19) is
        // priced at width 3.
        let reader = PermutedCsrReader::new(
            Box::new(deep_rows_spread_across_windows()),
            Arc::new(vec![1, 3, 0, 2]),
        );
        assert_eq!(
            reader.per_worker_bytes(3, ModalityType::Rna),
            crate::budget::shard_working_set_bytes(19, 3)
        );
    }

    #[test]
    fn a_reader_without_a_resident_indptr_still_delegates() {
        struct NoIndptr(u64);
        impl IndexedCsrShardStream for NoIndptr {
            fn n_obs(&self) -> u64 {
                4
            }
            fn n_vars(&self) -> u64 {
                self.0
            }
            fn source_matrix_name(&self) -> &str {
                "X"
            }
            fn read_range(
                &self,
                _row_start: u64,
                _n_rows: u32,
            ) -> Result<StreamedCsrShard, ConvertError> {
                unreachable!()
            }
        }
        let inner = NoIndptr(1000);
        let expected = inner.per_worker_bytes(2, ModalityType::Rna);
        let reader = PermutedCsrReader::new(Box::new(inner), Arc::new(vec![3, 2, 1, 0]));
        assert_eq!(reader.per_worker_bytes(2, ModalityType::Rna), expected);
    }
}
