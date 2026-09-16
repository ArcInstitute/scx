//! [`BackedDenseReader`]: on-demand row gather over a dense row-sharded
//! mapping (`obsm/<name>`), the dense counterpart of [`BackedCsrReader`].

use super::*;

// ---------------------------------------------------------------------------
// BackedDenseReader — on-demand row-gather for dense row-sharded mappings
// ---------------------------------------------------------------------------

/// A decoded dense shard (`obsm/<name>_shard_<i>`): one Arrow
/// `RecordBatch` of `n_cols` primitive columns (one per embedding
/// dimension), `n_shard_rows` long. Cached as `Arc<DenseShard>`.
struct DenseShard {
    batch: arrow::array::RecordBatch,
}

/// Per-shard catalog row retained by [`BackedDenseReader`]. Dense analog
/// of [`ShardEntryLite`] — no `nnz` (dense shards aren't CSR).
#[derive(Debug, Clone, Copy)]
struct DenseShardEntryLite {
    offset: u64,
    length: u64,
    section_type: SectionType,
    modality_id: u8,
}

impl DenseShardEntryLite {
    fn into_transient_full_entry(self) -> FullCatalogEntry {
        FullCatalogEntry {
            name: String::new(),
            offset: self.offset,
            length: self.length,
            section_type: self.section_type,
            checksum: [0u8; 32],
            modality_id: self.modality_id,
            stats: None,
        }
    }
}

impl SizeHint for DenseShard {
    /// Arrow's own accounting rather than the CSR component formula — a
    /// `RecordBatch` of `n_cols` primitive arrays has buffer padding and
    /// validity bitmaps the component model does not see.
    fn size_bytes(&self) -> usize {
        self.batch.get_array_memory_size()
    }
}

/// On-demand row-gather reader for a dense row-sharded mapping
/// (`obsm/<name>`). The dense counterpart of [`BackedCsrReader`]:
/// `read_row_indices` decodes only the touched `ObsmEmbeddingShard`s
/// (bounded LRU + singleflight) and gathers the requested rows, so
/// per-call memory is `O(batch × n_cols)` and independent of file size.
///
/// # Fork safety
///
/// Like [`BackedCsrReader`], all of `cache` / `in_flight` / `metrics`
/// are **per-instance** — never global. A forked DataLoader worker that
/// constructs its own `BackedDenseReader` gets fresh `Mutex`es, so the
/// fork-deadlock contract (`pyscx/tests/test_fork_deadlock.py`) holds.
pub struct BackedDenseReader {
    reader: ScxReader,
    index: BackedCsrIndex,
    /// Logical mapping name (e.g. `"X_pca"`), used by [`Self::read_all`].
    name: String,
    n_rows: usize,
    n_cols: usize,
    dtype: arrow::datatypes::DataType,
    /// Canonical output schema (per-shard metadata stripped).
    schema: Arc<arrow::datatypes::Schema>,
    /// Per-shard catalog rows, ordered by `row_start`.
    sorted_entries: Vec<DenseShardEntryLite>,
    /// Decoded-shard cache + singleflight, keyed by shard index. Before this
    /// was shared, the reader carried `cache` / `in_flight` / `metrics` as
    /// three separate fields and its own copy of the rendezvous loop.
    cache: ShardCache<usize, DenseShard>,
    prefetch_count: usize,
}

impl BackedDenseReader {
    /// `Ok(())` unless this reader is watching its file and the file has
    /// changed since it was opened. See [`ScxReader::check_fresh`].
    ///
    /// Section reads are already covered — they funnel through
    /// `ScxReader::section_bytes`. This exists for the answers that never
    /// touch a section: the cached `shape` / shard-count scalars a caller
    /// reads straight off the index.
    pub fn check_fresh(&self) -> Result<()> {
        self.reader.check_fresh()
    }

    /// Open a backed dense reader over `obsm/<name>` with a count-only
    /// cache cap (`cache_shards` decoded shards; 0 = no cache).
    pub fn new_obsm(reader: ScxReader, name: &str, cache_shards: usize) -> Result<Self> {
        Self::new_obsm_with_byte_budget(reader, name, cache_shards, usize::MAX)
    }

    /// Open a backed dense reader over `obsm/<name>` with both a count
    /// cap and a byte cap on the decoded-shard LRU.
    pub fn new_obsm_with_byte_budget(
        reader: ScxReader,
        name: &str,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Result<Self> {
        let layout = reader.row_sharded_mapping_layout(
            "obsm",
            name,
            SectionType::ObsmEmbeddingShard,
            SectionType::ObsmEmbedding,
            crate::reader::LegacyRowCount::BatchRows,
        )?;
        Ok(Self::from_layout(
            reader,
            name,
            layout,
            cache_shards,
            bytes_budget,
        ))
    }

    fn from_layout(
        reader: ScxReader,
        name: &str,
        layout: crate::reader::MappingLayout,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Self {
        let ranges: Vec<(u64, u64)> = layout
            .entries
            .iter()
            .map(|e| (e.row_start, e.row_start + e.n_shard_rows))
            .collect();
        let index = BackedCsrIndex::from_ranges(&ranges);
        let sorted_entries: Vec<DenseShardEntryLite> = layout
            .entries
            .iter()
            .map(|e| DenseShardEntryLite {
                offset: e.offset,
                length: e.length,
                section_type: e.section_type,
                modality_id: e.modality_id,
            })
            .collect();
        let schema = Arc::new(arrow::datatypes::Schema::new(layout.fields));
        let prefetch_count = cache_shards.max(2);
        BackedDenseReader {
            reader,
            index,
            name: name.to_string(),
            n_rows: layout.n_rows as usize,
            n_cols: layout.n_cols,
            dtype: layout.dtype,
            schema,
            sorted_entries,
            cache: ShardCache::inline(cache_shards, bytes_budget),
            prefetch_count,
        }
    }

    /// Shape of the full mapping `(n_rows, n_cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rows, self.n_cols)
    }

    /// Number of rows (obs).
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Embedding dimensionality.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Representative on-disk dtype (column 0).
    pub fn dtype(&self) -> &arrow::datatypes::DataType {
        &self.dtype
    }

    /// Canonical output schema (per-shard metadata stripped).
    pub fn schema(&self) -> &Arc<arrow::datatypes::Schema> {
        &self.schema
    }

    /// Access the underlying shard index.
    pub fn index(&self) -> &BackedCsrIndex {
        &self.index
    }

    /// Enable cache-behaviour counters (test/diagnostic use).
    ///
    /// **Idempotent, and that is a change.** This used to install a fresh
    /// `CacheMetrics` on every call, so a second call reset the counters to
    /// zero and orphaned the handle the first caller was holding. It now
    /// returns the same handle every time, with counters that accumulate for
    /// the life of the reader — matching `BackedCsrReader`, which has always
    /// behaved this way via the shared cache's `OnceLock`.
    ///
    /// A caller that wants a fresh measurement window must therefore snapshot
    /// the counters and subtract, rather than re-enabling. Nothing in-tree did
    /// the latter, but this is public API and the old behaviour was reachable.
    pub fn enable_metrics(&mut self) -> Arc<CacheMetrics> {
        self.cache.enable_metrics()
    }

    /// True if `shard_idx` is currently cached.
    pub fn cache_contains(&self, shard_idx: usize) -> bool {
        self.cache.contains_key(shard_idx)
    }

    fn shard_count(&self) -> usize {
        self.sorted_entries.len()
    }

    /// Decode + cache one dense shard, returning a shared `Arc`. The
    /// singleflight, the cache-hit fast path and the metrics accounting all
    /// live in [`ShardCache::get_or_decode`]; what is left here is the part
    /// that is actually dense-specific.
    fn read_shard_cached_arc(&self, shard_idx: usize) -> Result<Arc<DenseShard>> {
        // See `BackedCsrReader::read_shard_cached_arc`: a cache hit bypasses
        // `section_bytes` entirely.
        self.check_fresh()?;
        self.cache
            .get_or_decode(shard_idx, || self.decode_shard(shard_idx))
    }

    fn decode_shard(&self, shard_idx: usize) -> Result<Arc<DenseShard>> {
        let lite = *self
            .sorted_entries
            .get(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.shard_count(),
            })?;
        let batch = self
            .reader
            .read_dense_mapping_entry(&lite.into_transient_full_entry())?;
        let shard = Arc::new(DenseShard { batch });

        #[cfg(unix)]
        {
            use memmap2::Advice;
            let n_shards = self.shard_count();
            let end = std::cmp::min(shard_idx + 1 + self.prefetch_count, n_shards);
            for prefetch_idx in (shard_idx + 1)..end {
                if let Some(entry) = self.sorted_entries.get(prefetch_idx) {
                    let _ = self.reader.mmap_ref().advise_range(
                        Advice::WillNeed,
                        entry.offset as usize,
                        entry.length as usize,
                    );
                }
            }
        }

        Ok(shard)
    }

    /// Pre-decode the requested shards into the LRU (sequential — the
    /// dense gather is bounded by the row batch, not zstd decode, so the
    /// CSR parallel-warm machinery isn't replicated here).
    fn warm_shards(&self, shard_indices: &[usize]) -> Result<()> {
        if !self.cache.has_cache() {
            return Ok(());
        }
        let mut seen: HashSet<usize> = HashSet::with_capacity(shard_indices.len());
        for &idx in shard_indices {
            if seen.insert(idx) {
                self.read_shard_cached_arc(idx)?;
            }
        }
        Ok(())
    }

    /// Empty `(0, n_cols)` batch with the canonical schema.
    fn empty_batch(&self) -> Result<arrow::array::RecordBatch> {
        Ok(arrow::array::RecordBatch::new_empty(Arc::clone(
            &self.schema,
        )))
    }

    /// Gather `[start, end)` rows as a dense `RecordBatch`.
    pub fn read_rows_range(&self, start: u64, end: u64) -> Result<arrow::array::RecordBatch> {
        if start >= end {
            return self.empty_batch();
        }
        let indices: Vec<u64> = (start..end).collect();
        self.read_row_indices(&indices)
    }

    /// Gather the given global row indices as a dense `RecordBatch`, in
    /// request order, decoding only the touched shards. Dtype-preserving.
    ///
    /// Errors if any requested row falls outside the mapping's row range
    /// (the caller — `ScxBackedObsmDataset` — always passes in-range
    /// global rows, so a miss signals a logic bug rather than a silent
    /// drop).
    pub fn read_row_indices(&self, indices: &[u64]) -> Result<arrow::array::RecordBatch> {
        use arrow::array::{ArrayRef, UInt32Array};

        if indices.is_empty() {
            return self.empty_batch();
        }

        let mut sorted_pairs: Vec<(u64, usize)> =
            indices.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        sorted_pairs.sort_by_key(|&(r, _)| r);

        let shard_indices = self
            .index
            .shards_for_indices(&sorted_pairs.iter().map(|&(r, _)| r).collect::<Vec<_>>());
        self.warm_shards(&shard_indices)?;

        let mut sub_batches: Vec<arrow::array::RecordBatch> =
            Vec::with_capacity(shard_indices.len());
        // Original request position of each output row, in concat order.
        let mut orig_order: Vec<usize> = Vec::with_capacity(indices.len());

        for &shard_idx in &shard_indices {
            let shard = self.read_shard_cached_arc(shard_idx)?;
            let (s_start, s_end) =
                self.index
                    .shard_range(shard_idx)
                    .ok_or(ScxError::ShardIndexOutOfBounds {
                        index: shard_idx,
                        count: self.index.n_shards(),
                    })?;

            let mut locals: Vec<u32> = Vec::new();
            for &(row, orig_idx) in &sorted_pairs {
                if row >= s_start && row < s_end {
                    locals.push((row - s_start) as u32);
                    orig_order.push(orig_idx);
                }
            }
            if locals.is_empty() {
                continue;
            }
            let idx_arr = UInt32Array::from(locals);
            let cols: Vec<ArrayRef> = shard
                .batch
                .columns()
                .iter()
                .map(|c| arrow::compute::take(c, &idx_arr, None))
                .collect::<std::result::Result<_, _>>()
                .map_err(ScxError::Arrow)?;
            sub_batches.push(
                arrow::array::RecordBatch::try_new(Arc::clone(&self.schema), cols)
                    .map_err(ScxError::Arrow)?,
            );
        }

        if orig_order.len() != indices.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "obsm/{}: dense row gather resolved {} of {} requested rows; \
                 {} fell outside the mapping's {}-row range",
                self.name,
                orig_order.len(),
                indices.len(),
                indices.len() - orig_order.len(),
                self.n_rows,
            )));
        }

        let concatenated = arrow::compute::concat_batches(&self.schema, sub_batches.iter())
            .map_err(ScxError::Arrow)?;

        // Reorder concat rows into request order via the inverse permutation.
        let mut perm = vec![0u32; orig_order.len()];
        for (concat_pos, &orig_idx) in orig_order.iter().enumerate() {
            perm[orig_idx] = concat_pos as u32;
        }
        let perm_arr = UInt32Array::from(perm);
        let final_cols: Vec<ArrayRef> = concatenated
            .columns()
            .iter()
            .map(|c| arrow::compute::take(c, &perm_arr, None))
            .collect::<std::result::Result<_, _>>()
            .map_err(ScxError::Arrow)?;
        arrow::array::RecordBatch::try_new(Arc::clone(&self.schema), final_cols)
            .map_err(ScxError::Arrow)
    }

    /// Read the full mapping. Delegates to [`ScxReader::read_obsm`],
    /// which already concatenates + validates the shard cover and
    /// preserves dtype.
    pub fn read_all(&self) -> Result<arrow::array::RecordBatch> {
        self.reader.read_obsm(&self.name)
    }
}
