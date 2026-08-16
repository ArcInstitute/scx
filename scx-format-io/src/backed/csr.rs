//! [`BackedCsrReader`]: on-demand row access to X, a layer, or a modality's
//! CSR shards.
//!
//! The native aggregation kernels built on it live in [`super::aggregate`].

use super::index::ShardEntryLite;
use super::*;

// ---------------------------------------------------------------------------
// BackedCsrReader
// ---------------------------------------------------------------------------

/// On-demand CSR reader with optional shard caching.
///
/// Wraps an [`ScxReader`] with a [`BackedCsrIndex`] for efficient row-range
/// lookups and an optional LRU cache for decoded shards.
///
/// # Thread Safety
///
/// `ScxReader` contains a `memmap2::Mmap` which is `Send + Sync`.
/// The reader itself is safe to share immutably.  Only the LRU cache
/// requires interior mutability via [`Mutex`].
///
/// # Fork safety
///
/// **Invariant**: the shard cache lives in a [`SharedShardCache`] held via
/// `Arc`, owned by this reader (or shared with sibling readers of the same
/// multi-file run instance), *never* a process-global `OnceCell` / `static` /
/// `lazy_static`. Per-instance state is the contract that keeps this type
/// fork-safe: a forked child that constructs its own reader (e.g. via
/// `pyscx.TrainingDataset` lazy construction inside a `DataLoader` worker's
/// `__iter__`) gets a fresh `Mutex` that can't be inherited from the parent in
/// a poisoned-locked state. Sharing the cache across readers within one
/// instance keeps the contract; only a global would break it, and the
/// fork-mode regression test (`pyscx/tests/test_fork_safety.py`) catches the
/// resulting hang on the first cache access in the child.
pub struct BackedCsrReader {
    pub(super) reader: ScxReader,
    pub(super) index: BackedCsrIndex,
    pub(super) n_vars: usize,
    pub(super) n_obs: usize,
    /// If set, this reader targets a specific layer rather than X.
    pub(super) layer_name: Option<String>,
    /// Pre-sorted lightweight catalog rows for X shards. Drops the
    /// per-shard `String` name and 32-byte BLAKE3 checksum that the
    /// read path never consumes. See [`ShardEntryLite`] for the
    /// field set and per-shard footprint.
    pub(super) x_sorted_entries: Vec<ShardEntryLite>,
    /// Pre-sorted lightweight catalog rows for layer shards (empty
    /// when this reader targets X). Same `ShardEntryLite` shape as
    /// `x_sorted_entries`.
    pub(super) sorted_entries: Vec<ShardEntryLite>,
    /// Decoded-shard cache + singleflight, keyed `(file_id, shard_id)`. Held via
    /// `Arc` so several readers of one run can share one bounded budget; a
    /// standalone reader gets its own with `file_id = 0` (behavior unchanged).
    /// See the "Fork safety" section above.
    shard_cache: Arc<SharedShardCache>,
    /// This reader's id within its `shard_cache` namespace. `0` for a
    /// standalone reader; assigned by the multi-reader engine otherwise.
    file_id: u32,
    /// Number of shards to prefetch with `MADV_WILLNEED` after a cache miss.
    prefetch_count: usize,
    /// Configured count cap on the LRU (0 = no cache). Mirrored here so
    /// [`Self::warm_shards`] can chunk parallel decodes without locking the
    /// cache. Only read under `cfg(feature = "parallel")`.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    cache_shards: usize,
    /// Per-reader gate for the codec-agnostic **block-index** (row-group)
    /// scattered gather path (see [`Self::block_index_eligible`]). A framed (v2)
    /// shard takes the group-level block-index path so a framed training file
    /// gets random-access decode. Defaults to the `SCX_SCATTER_BLOCK_INDEX` env
    /// value for every constructor. The loader's per-dataset
    /// `scatter_block_index=False` opt-out flips this too (via
    /// [`Self::set_scatter_block_index`]) so the off-switch disables both the L1
    /// gather and the L2 prefetch skip, not just the prefetch.
    scatter_block_index: bool,
    /// Lazy per-shard "is framed?" memo for [`Self::shard_is_framed`], filled on
    /// first probe (lock-free `AtomicU8`: 0 = unknown, 1 = framed, 2 = unframed).
    /// `block_index_eligible` is called per prefetch-candidate + per gather group
    /// on the training hot path; caching the 76-byte header parse avoids
    /// re-reading it every batch for shards whose framing never changes.
    framed_cache: OnceLock<Vec<AtomicU8>>,
    /// Optional pool for this reader's own parallel decode
    /// ([`Self::warm_shards`]). `None` — the default for every constructor —
    /// keeps dispatch on rayon's global registry, so `scx-accel`, `scx-ops`,
    /// `scx-engine` and `scx-cli` are unaffected. See [`Self::set_cpu_pool`].
    ///
    /// `cfg`-gated because `rayon` is an optional dependency: without the
    /// `parallel` feature there is no `rayon::ThreadPool` to name, and
    /// `warm_shards` is a sequential loop anyway.
    #[cfg(feature = "parallel")]
    cpu_pool: Option<Arc<rayon::ThreadPool>>,
}

impl BackedCsrReader {
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

    /// Create a new backed reader for X shards with a count-only cache cap.
    ///
    /// `cache_shards`: number of decoded shards to cache (0 = no cache).
    /// Equivalent to [`Self::new_with_byte_budget`] with `bytes_budget =
    /// usize::MAX` — no byte cap.
    ///
    /// # Not modality-scoped
    ///
    /// This indexes **every** `CsrShard` in the catalog regardless of
    /// `modality_id`, and takes `n_vars` from the file header (the max across
    /// modalities). On a multimodal file that means one index over overlapping
    /// row ranges — each modality independently tiles `[0, n_obs)` — so
    /// `read_all` returns `Σ modalities` rows for an `n_obs`-cell file. Use
    /// [`Self::for_modality`] for per-modality access; every production caller
    /// does. Pinned by
    /// `backed_csr_reader_new_on_a_multimodal_file_folds_every_modality`.
    pub fn new(reader: ScxReader, cache_shards: usize) -> Self {
        Self::new_with_byte_budget(reader, cache_shards, usize::MAX)
    }

    /// Create a new backed reader for X shards with both a count cap and a
    /// byte cap on the LRU. Inserts evict oldest entries until both caps
    /// are satisfied (see `cache::WeightedLruCache::put_with_budget`).
    ///
    /// `cache_shards`: count cap on cached decoded shards (0 = no cache).
    /// `bytes_budget`: byte cap on cumulative decoded shard bytes; pass
    /// `usize::MAX` for count-only behavior.
    pub fn new_with_byte_budget(
        reader: ScxReader,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Self {
        // Derive both the row-range index and the lightweight per-shard
        // table from a single `CatalogView` pass — avoids the double
        // scan + per-entry clone the old `FullCatalog::shards_sorted()`
        // path performed.
        let view = CatalogView::from_full(reader.catalog());
        let sorted = view.csr_shards_sorted();
        let index = BackedCsrIndex::from_view_sorted(&sorted);
        let x_sorted_entries: Vec<ShardEntryLite> = sorted
            .iter()
            .map(|e| ShardEntryLite::from_view_entry(e))
            .collect();
        let n_vars = reader.n_vars() as usize;
        let n_obs = reader.n_obs() as usize;
        let prefetch_count = cache_shards.max(2);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: None,
            x_sorted_entries,
            sorted_entries: Vec::new(),
            shard_cache: SharedShardCache::new(cache_shards, bytes_budget),
            file_id: 0,
            prefetch_count,
            cache_shards,
            scatter_block_index: scatter_block_index_enabled(),
            framed_cache: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
        }
    }

    /// Create an X-shard reader that shares an existing [`SharedShardCache`]
    /// under `file_id`, for a multi-file run where several readers draw against
    /// one bounded decoded-shard budget (the Phase 1 prefetch engine). The
    /// reader's own `cache_shards` (warm chunking cap) mirrors the shared cap.
    pub fn with_shared_cache(
        reader: ScxReader,
        file_id: u32,
        shard_cache: Arc<SharedShardCache>,
    ) -> Self {
        let view = CatalogView::from_full(reader.catalog());
        let sorted = view.csr_shards_sorted();
        let index = BackedCsrIndex::from_view_sorted(&sorted);
        let x_sorted_entries: Vec<ShardEntryLite> = sorted
            .iter()
            .map(|e| ShardEntryLite::from_view_entry(e))
            .collect();
        let n_vars = reader.n_vars() as usize;
        let n_obs = reader.n_obs() as usize;
        let cache_shards = shard_cache.cache_shards;
        let prefetch_count = cache_shards.max(2);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: None,
            x_sorted_entries,
            sorted_entries: Vec::new(),
            shard_cache,
            file_id,
            prefetch_count,
            cache_shards,
            scatter_block_index: scatter_block_index_enabled(),
            framed_cache: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
        }
    }

    /// Phase B.5 / D.4: create a backed CSR reader scoped to a
    /// specific modality. Filters CSR shards by
    /// `(SectionType::CsrShard, modality_id)` so each modality gets
    /// its own LRU cache and shard range — matching the
    /// `BackedCscReader::for_modality` shape. `n_vars` is taken from
    /// the modality's table entry (per-modality `n_vars`), not the
    /// file-wide `header.n_vars` (which is the max across modalities
    /// on multimodal v2 files).
    pub fn for_modality(reader: ScxReader, modality_id: u8, cache_shards: usize) -> Self {
        let view = CatalogView::from_full(reader.catalog());
        let sorted = view.csr_shards_for_modality(modality_id);
        let index = BackedCsrIndex::from_view_sorted(&sorted);
        let x_sorted_entries: Vec<ShardEntryLite> = sorted
            .iter()
            .map(|e| ShardEntryLite::from_view_entry(e))
            .collect();
        let n_vars = match reader.modality_info(modality_id) {
            Some(info) => info.n_vars as usize,
            None => reader.n_vars() as usize,
        };
        let n_obs = reader.n_obs() as usize;
        let prefetch_count = cache_shards.max(2);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: None,
            x_sorted_entries,
            sorted_entries: Vec::new(),
            shard_cache: SharedShardCache::new(cache_shards, usize::MAX),
            file_id: 0,
            prefetch_count,
            cache_shards,
            scatter_block_index: scatter_block_index_enabled(),
            framed_cache: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
        }
    }

    /// Create a new backed reader for a specific layer's shards.
    ///
    /// `layer_name`: the layer name (e.g., `"raw"`).
    /// `cache_shards`: number of decoded shards to cache (0 = no cache).
    pub fn new_for_layer(reader: ScxReader, layer_name: &str, cache_shards: usize) -> Self {
        Self::new_for_layer_with_byte_budget(reader, layer_name, cache_shards, usize::MAX)
    }

    /// Create a layer reader with both count and byte caps. See
    /// [`Self::new_with_byte_budget`].
    pub fn new_for_layer_with_byte_budget(
        reader: ScxReader,
        layer_name: &str,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Self {
        // Single `CatalogView` pass produces both the layer table
        // (filtered by `LayerCsrShard` + name prefix) and the X table
        // (kept alongside so layer-mode readers can still serve X reads
        // through their `shard_entry` dispatch).
        let view = CatalogView::from_full(reader.catalog());
        let prefix = format!("{layer_name}_shard_");
        let sorted_layer = view.layer_csr_shards_sorted_with_prefix(&prefix);
        let index = BackedCsrIndex::from_view_sorted(&sorted_layer);
        let sorted_entries: Vec<ShardEntryLite> = sorted_layer
            .iter()
            .map(|e| ShardEntryLite::from_view_entry(e))
            .collect();
        let sorted_x = view.csr_shards_sorted();
        let x_sorted_entries: Vec<ShardEntryLite> = sorted_x
            .iter()
            .map(|e| ShardEntryLite::from_view_entry(e))
            .collect();

        let n_vars = reader.n_vars() as usize;
        // n_obs for layers is the same as for X — layer shards cover the same rows.
        let n_obs = reader.n_obs() as usize;
        let prefetch_count = cache_shards.max(2);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: Some(layer_name.to_string()),
            x_sorted_entries,
            sorted_entries,
            shard_cache: SharedShardCache::new(cache_shards, bytes_budget),
            file_id: 0,
            prefetch_count,
            cache_shards,
            scatter_block_index: scatter_block_index_enabled(),
            framed_cache: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
        }
    }

    /// Enable cache-behavior metrics on this reader. Returns a cloneable
    /// `Arc<CacheMetrics>` so callers (e.g. `IndexPlanIter`) can sample
    /// counters without going through the cache lock. Subsequent calls
    /// rebind to a fresh metrics handle (intended for opt-in setup, not
    /// runtime toggling).
    pub fn enable_metrics(&mut self) -> Arc<CacheMetrics> {
        self.shard_cache.enable_metrics()
    }

    /// Borrow this reader's metrics handle, if metrics are enabled.
    pub fn metrics(&self) -> Option<&Arc<CacheMetrics>> {
        self.shard_cache.metrics()
    }

    /// Peek the LRU for `shard_idx` without touching recency or decoding.
    /// Returns `false` when no cache is configured.
    pub fn cache_contains(&self, shard_idx: usize) -> bool {
        self.shard_cache.contains(self.file_id, shard_idx)
    }

    /// Whether a request-group of `group_len` rows landing in `shard_idx` would
    /// Cheap "is this shard row-group framed (v2)?" probe. Reads only the
    /// 76-byte shard header from the mmap (no payload decode, no catalog linear
    /// scan — resolves the section via the retained [`ShardEntryLite`]), and
    /// tests `shard_format_version > DEFAULT_WRITE_SHARD_FORMAT_VERSION`. Used by
    /// [`Self::block_index_eligible`] to gate the block-index path **per shard**:
    /// a v4 file may mix v2-framed and legacy v1 shards, so a file-level version
    /// check would misclassify. Returns `false` on any missing entry / header
    /// parse error (caller then falls back to the full-shard path). The header
    /// parse is memoized per shard in `framed_cache` (a shard's framing is
    /// immutable for the reader's lifetime), so repeated hot-path probes cost one
    /// atomic load.
    fn shard_is_framed(&self, shard_idx: usize) -> bool {
        let cache = self
            .framed_cache
            .get_or_init(|| (0..self.shard_count()).map(|_| AtomicU8::new(0)).collect());
        // 0 = unknown, 1 = framed, 2 = unframed. Out-of-range shard_idx (should
        // not happen) falls through to a direct (uncached) compute.
        if let Some(slot) = cache.get(shard_idx) {
            match slot.load(Ordering::Relaxed) {
                1 => return true,
                2 => return false,
                _ => {}
            }
            let framed = self.compute_shard_is_framed(shard_idx);
            slot.store(if framed { 1 } else { 2 }, Ordering::Relaxed);
            return framed;
        }
        self.compute_shard_is_framed(shard_idx)
    }

    /// Uncached header read backing [`Self::shard_is_framed`].
    fn compute_shard_is_framed(&self, shard_idx: usize) -> bool {
        let Some(&lite) = self.shard_entry(shard_idx) else {
            return false;
        };
        let entry = lite.into_transient_full_entry();
        self.reader
            .read_shard_header(&entry)
            .map(|h| h.shard_format_version > crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION)
            .unwrap_or(false)
    }

    /// Override the per-reader block-index gate (default from
    /// `SCX_SCATTER_BLOCK_INDEX`). The loader's per-dataset
    /// `scatter_block_index=False` calls this so the off-switch disables the L1
    /// gather adoption too, not just the L2 prefetch skip.
    pub fn set_scatter_block_index(&mut self, enabled: bool) {
        self.scatter_block_index = enabled;
    }

    /// Run this reader's parallel shard warm ([`Self::warm_shards`]) on `pool`
    /// instead of rayon's global registry.
    ///
    /// Exists for callers that may be running in a **forked** process. `fork()`
    /// duplicates rayon's global registry as a data structure but not its
    /// worker threads, so a `par_*` dispatched from the child parks forever in
    /// `LockLatch::wait_and_reset`. A pool built *after* the fork has live
    /// threads and does not. `scx-loader` passes `scx_loader::pool::cpu_pool()`
    /// here; see its module docs for why that pool is keyed on the PID.
    ///
    /// Unset by default, which keeps every other consumer (`scx-accel`,
    /// `scx-ops`, `scx-engine`, `scx-cli`) on the global pool exactly as before
    /// — they do not fork, and they compose with their own global-pool parallel
    /// regions.
    #[cfg(feature = "parallel")]
    pub fn set_cpu_pool(&mut self, pool: Arc<rayon::ThreadPool>) {
        self.cpu_pool = Some(pool);
    }

    /// Whether a shard request-group should be served by the block-index
    /// (row-group) scattered path rather than a full-shard decode. Eligible when
    /// the process-global `SCX_SCATTER_BLOCK_INDEX` kill-switch is on **and** this
    /// reader's per-reader `scatter_block_index` gate is on, the shard is not
    /// already decoded in the LRU, the requested rows are a small fraction of the
    /// shard (`group_len * ROW_RANGE_WINDOW_DIVISOR < shard_rows`), **and** the
    /// shard is actually row-group framed. This is the single source of truth for
    /// the block-index-vs-full-shard decision — it covers both the L1
    /// `read_rows_with` gather and the L2 plan-prefetch warm-skip, so the gather
    /// choice and the prefetch skip never drift. The framing header read is
    /// ordered last so it only runs for cost-eligible, uncached candidates.
    pub fn block_index_eligible(&self, shard_idx: usize, group_len: usize) -> bool {
        scatter_block_index_enabled()
            && self.scatter_block_index
            && !self.shard_cache.contains(self.file_id, shard_idx)
            && self
                .index
                .shard_range(shard_idx)
                .is_some_and(|(s, e)| (group_len as u64) * ROW_RANGE_WINDOW_DIVISOR < (e - s))
            && self.shard_is_framed(shard_idx)
    }

    /// True if any CSR shard is row-group framed (`shard_format_version >= 2`),
    /// i.e. the scattered block-index fast path can fire on at least one shard.
    /// An all-unframed (legacy v1) file full-shard-decodes every scattered
    /// gather regardless of `scatter_block_index` — callers use this at open
    /// time to warn that the fast path is inert. Early-returns on the first
    /// framed shard; reads only shard headers (no payload), memoized per shard
    /// via `framed_cache`.
    pub fn any_shard_framed(&self) -> bool {
        (0..self.shard_count()).any(|i| self.shard_is_framed(i))
    }

    /// Live count cap on the decoded-shard LRU. `0` means no cache was
    /// installed and the cached read APIs decode on every call. Reflects any
    /// growth from [`Self::ensure_cache_capacity`] (not just the open-time
    /// `cache_shards`). Multi-pass kernels (streaming Wilcoxon, out-of-core
    /// PCA) use this to detect cache-too-small footguns and warn before paying
    /// the silent perf cliff.
    pub fn cache_capacity(&self) -> usize {
        self.shard_cache.capacity()
    }

    /// Reserve the decoded-shard LRU for a multi-pass op so it can hold its
    /// whole shard working set, returning the resulting count cap.
    ///
    /// Raises the count cap toward `min_shards` (never shrinks) and sets the
    /// byte budget to `max_cache_bytes` — the op's RAM ceiling, which bounds
    /// memory even on a count-only-opened reader (whose budget starts
    /// unbounded). An out-of-core matrix larger than `max_cache_bytes` still
    /// evicts, so the returned cap is honored only up to what the bytes allow;
    /// callers warn when it can't hold all shards. No-op (returns `0`) when no
    /// cache was installed at open time. Used by out-of-core PCA to turn the
    /// open-time `cache_shards` default into a working-set-sized cache governed
    /// by the op's memory budget; the byte budget persists for later reads on
    /// this reader.
    pub fn ensure_cache_capacity(&self, min_shards: usize, max_cache_bytes: usize) -> usize {
        self.shard_cache.reserve_for(min_shards, max_cache_bytes)
    }

    /// Peek the singleflight table for `shard_idx`. Returns `false` when no
    /// cache (and therefore no singleflight) is configured.
    pub fn in_flight_contains(&self, shard_idx: usize) -> bool {
        self.shard_cache.in_flight_contains(self.file_id, shard_idx)
    }

    /// Shape of the full matrix `(n_obs, n_vars)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_vars)
    }

    /// Number of variables (columns).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Access the underlying shard index.
    pub fn index(&self) -> &BackedCsrIndex {
        &self.index
    }

    /// Get the lightweight catalog row for a shard by index.
    ///
    /// Uses `sorted_entries` for layer readers, `x_sorted_entries` for X readers.
    fn shard_entry(&self, shard_idx: usize) -> Option<&ShardEntryLite> {
        if self.layer_name.is_some() {
            self.sorted_entries.get(shard_idx)
        } else {
            self.x_sorted_entries.get(shard_idx)
        }
    }

    /// Total number of shards.
    pub(super) fn shard_count(&self) -> usize {
        if self.layer_name.is_some() {
            self.sorted_entries.len()
        } else {
            self.x_sorted_entries.len()
        }
    }

    /// Read rows `[start, end)` as a scipy-compatible `ScxCsr`.
    ///
    /// Decompresses only overlapping shards.  Takes `&self` — cache
    /// mutation is handled via interior mutability (Mutex).
    pub fn read_rows(&self, start: u64, end: u64) -> Result<ScxCsr> {
        if start >= end {
            return Ok(ScxCsr::new_unchecked(
                (0, self.n_vars),
                vec![0],
                vec![],
                vec![],
            ));
        }

        let shard_indices = self.index.shards_for_range(start, end);
        if shard_indices.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (0, self.n_vars),
                vec![0],
                vec![],
                vec![],
            ));
        }

        // Plan each overlapping shard. A genuinely small window of an *uncached*
        // framed shard is decoded directly via the block-index row-range path
        // (O(window); no full-shard decode, no cache pollution). Everything else
        // — large windows, cached shards, unframed shards, repeated/sequential
        // access like the training loader — takes the full-decode + cache path, so
        // only those shards are pre-warmed. Tuple: (shard_idx, local_start,
        // local_end, use_row_range).
        const ROW_RANGE_WINDOW_DIVISOR: u64 = 4;
        let mut plans: Vec<(usize, usize, usize, bool)> = Vec::with_capacity(shard_indices.len());
        let mut full_shards: Vec<usize> = Vec::new();
        // Plan each shard's read strategy. The shared cache is keyed by
        // `(file_id, shard)`, so membership is queried per shard rather than
        // under one held lock; the planning set (shards a row range touches)
        // is small, and `warm_shards` re-locks anyway.
        {
            for &shard_idx in &shard_indices {
                let (s_start, s_end) =
                    self.index
                        .shard_range(shard_idx)
                        .ok_or(ScxError::ShardIndexOutOfBounds {
                            index: shard_idx,
                            count: self.index.n_shards(),
                        })?;
                let local_start = (start.max(s_start) - s_start) as usize;
                let local_end = (end.min(s_end) - s_start) as usize;
                let window = (local_end - local_start) as u64;
                let shard_rows = s_end - s_start;
                let cached = self.shard_cache.contains(self.file_id, shard_idx);
                let use_row_range = !cached && window * ROW_RANGE_WINDOW_DIVISOR < shard_rows;
                if !use_row_range {
                    full_shards.push(shard_idx);
                }
                plans.push((shard_idx, local_start, local_end, use_row_range));
            }
        }

        // Pre-decode the cold full-path shards in parallel (no-op if all cached).
        self.warm_shards(&full_shards)?;

        let mut slices = Vec::with_capacity(plans.len());
        for (shard_idx, local_start, local_end, use_row_range) in plans {
            if use_row_range {
                if let Some(sliced) = self.try_row_range_slice(shard_idx, local_start, local_end)? {
                    slices.push(sliced);
                    continue;
                }
                // Unframed shard — fall through to the full-decode path.
            }
            let shard_csr = self.read_shard_cached_arc(shard_idx)?;
            let sliced = shard_csr
                .row_slice(local_start, local_end)
                .map_err(|e| ScxError::Io(std::io::Error::other(e)))?;
            slices.push(sliced);
        }

        Ok(scx_sparse::concatenate_csr(&slices, self.n_vars)?)
    }

    /// Decode just rows `[local_start, local_end)` of shard `shard_idx` directly
    /// from its row-group block index (O(window)), returning `None` when the
    /// shard is not row-group framed so the caller can fall back to a full-shard
    /// decode. Byte-identical to decoding the whole shard and slicing.
    fn try_row_range_slice(
        &self,
        shard_idx: usize,
        local_start: usize,
        local_end: usize,
    ) -> Result<Option<ScxCsr>> {
        // Recover the *real* catalog entry for this shard. The `ShardEntryLite`
        // table drops the name/checksum, but `full_entry_at_offset` restores the
        // full entry the block-index decode needs. `(offset, section_type)`
        // uniquely identifies the section.
        let Some((offset, section_type)) = self
            .shard_entry(shard_idx)
            .map(|l| (l.offset, l.section_type))
        else {
            return Ok(None);
        };
        let Some(entry) = self.reader.full_entry_at_offset(offset, section_type) else {
            return Ok(None);
        };
        let n = local_end - local_start;
        match self
            .reader
            .decode_block_index_row_runs(entry, &[(local_start, n)])?
        {
            Some(mut runs) => {
                let (indptr, indices, data) = runs.pop().ok_or_else(|| {
                    ScxError::Io(std::io::Error::other(
                        "block-index row-range decode returned no runs",
                    ))
                })?;
                Ok(Some(ScxCsr::new_unchecked(
                    (n, self.n_vars),
                    indptr,
                    indices,
                    data,
                )))
            }
            None => Ok(None),
        }
    }

    /// Read specific row indices as a scipy-compatible `ScxCsr`.
    ///
    /// Decompresses only shards containing requested rows.
    pub fn read_row_indices(&self, indices: &[u64]) -> Result<ScxCsr> {
        if indices.is_empty() {
            return Ok(ScxCsr::new_unchecked(
                (0, self.n_vars),
                vec![0],
                vec![],
                vec![],
            ));
        }

        // Sort the indices but keep track of the original ordering
        // so we output rows in the requested order.
        let mut sorted_pairs: Vec<(u64, usize)> =
            indices.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        sorted_pairs.sort_by_key(|&(r, _)| r);

        // Group by shard
        let shard_indices = self
            .index
            .shards_for_indices(&sorted_pairs.iter().map(|&(r, _)| r).collect::<Vec<_>>());

        // Pre-decode cold shards in parallel before the per-shard gather.
        // No-op when every shard is already cached.
        self.warm_shards(&shard_indices)?;

        // For each shard, extract the needed rows
        let mut row_csrs: Vec<(usize, ScxCsr)> = Vec::new();

        for &shard_idx in &shard_indices {
            let shard_csr = self.read_shard_cached_arc(shard_idx)?;
            let (s_start, s_end) =
                self.index
                    .shard_range(shard_idx)
                    .ok_or(ScxError::ShardIndexOutOfBounds {
                        index: shard_idx,
                        count: self.index.n_shards(),
                    })?;

            // Extract individual rows from this shard
            for &(row, orig_idx) in &sorted_pairs {
                if row >= s_start && row < s_end {
                    let local_row = (row - s_start) as usize;
                    let sliced = shard_csr
                        .row_slice(local_row, local_row + 1)
                        .map_err(|e| ScxError::Io(std::io::Error::other(e)))?;
                    row_csrs.push((orig_idx, sliced));
                }
            }
        }

        // Sort by original index to preserve requested ordering
        row_csrs.sort_by_key(|&(idx, _)| idx);

        let ordered: Vec<ScxCsr> = row_csrs.into_iter().map(|(_, csr)| csr).collect();
        Ok(scx_sparse::concatenate_csr(&ordered, self.n_vars)?)
    }

    /// Read specific row indices, invoking `scatter` once per row with
    /// zero-copy `(indices, data)` views into the decoded shard.
    ///
    /// For each request `rows[i]`, calls `scatter(i, indices, data)` where
    /// `(indices, data)` are slices into the cached shard's CSR for that row.
    /// Each touched shard is decoded once via the LRU cache; scatter calls
    /// fire in shard-grouped (sorted-by-row) order, but the `i` argument is
    /// the original position in `rows`, so callers can write to a dense
    /// output buffer indexed by request order.
    ///
    /// Allocates no intermediate `ScxCsr` and does no per-row `row_slice`
    /// — the per-shard request sub-slice is found via binary search on the
    /// shard ranges (O(R log S) total grouping cost), avoiding the
    /// `read_row_indices` inner-scan pattern. Use this for dense-gather
    /// hot paths (ML training, paired-batch readers) where the consumer
    /// owns the dense output. Callers that need a `ScxCsr` (scipy interop)
    /// should keep using [`Self::read_row_indices`].
    ///
    /// `rows` may contain duplicates; each occurrence triggers one
    /// `scatter` call. Empty `rows` is a no-op.
    ///
    /// **Out-of-range semantics differ from [`Self::read_row_indices`]**:
    /// `read_row_indices` filters via `shards_for_indices` and silently
    /// drops rows that fall outside every shard, whereas `read_rows_with`
    /// walks shards directly and returns an error on the first row outside
    /// any shard range. Callers that need silent-skip semantics must
    /// pre-filter `rows`.
    pub fn read_rows_with<F>(&self, rows: &[u64], mut scatter: F) -> Result<()>
    where
        F: FnMut(usize, &[i32], &[f32]) -> Result<()>,
    {
        if rows.is_empty() {
            return Ok(());
        }

        // (row, orig_pos) sorted by row so duplicates / requests for the
        // same shard are contiguous and the shard decode happens once.
        let mut sorted_pairs: Vec<(u64, usize)> =
            rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        sorted_pairs.sort_by_key(|&(r, _)| r);

        // Plan each shard's request group: sidecar (row-range) vs full-shard
        // decode — same policy as the contiguous `read_rows` planner. Use the
        // sidecar only when the shard isn't already decoded AND the requested
        // rows are a small fraction of the shard (so O(rows) block-index decode
        // beats one full 16k-row shard decode). Scattered cell-set gather hits
        // the block-index path; sequential / cached reads keep the full-shard path.
        //
        // `shard_for_row` (not `shards_for_indices`) preserves the out-of-range
        // error semantics. Planning before warming is what lets the `!cached`
        // test mean something — we then warm ONLY the full-path shards, so
        // block-index-group shards stay undecoded and the row-range path is taken.
        // (start, end, shard_idx, s_start, use_block_index)
        let mut groups: Vec<(usize, usize, usize, u64, bool)> = Vec::new();
        let mut full_shards: Vec<usize> = Vec::new();
        let mut start = 0;
        while start < sorted_pairs.len() {
            let row = sorted_pairs[start].0;
            let shard_idx = self.index.shard_for_row(row).ok_or_else(|| {
                ScxError::Io(std::io::Error::other(format!(
                    "row index {row} out of range (n_obs={})",
                    self.n_obs
                )))
            })?;
            let (s_start, s_end) =
                self.index
                    .shard_range(shard_idx)
                    .ok_or(ScxError::ShardIndexOutOfBounds {
                        index: shard_idx,
                        count: self.index.n_shards(),
                    })?;

            // First request index in `sorted_pairs` whose row >= s_end.
            let group_len = sorted_pairs[start..].partition_point(|&(r, _)| r < s_end);
            let end = start + group_len;

            // Scattered-vs-full-shard via the shared predicate (also used by the
            // plan-prefetch skip, so the two never drift). As of L2, the prefetch
            // engines leave block-index-eligible cold sparse shards un-warmed, so an
            // eligible predicate returns true here and the O(rows) path is taken;
            // dense/cached/unframed groups still go full-shard.
            let use_block_index = self.block_index_eligible(shard_idx, group_len);
            if !use_block_index {
                full_shards.push(shard_idx);
            }
            groups.push((start, end, shard_idx, s_start, use_block_index));
            start = end;
        }

        // Pre-decode the cold full-path shards in parallel (no-op if all cached
        // or if every group took the block-index path).
        self.warm_shards(&full_shards)?;

        for (start, end, shard_idx, s_start, use_block_index) in groups {
            let group = &sorted_pairs[start..end];

            // Strategy: on an eligible (sparse, cache-cold, framed) group, decode
            // only the touched row-groups via the codec-agnostic block index;
            // otherwise fall back to a full-shard decode.
            let mut handled = false;
            if use_block_index {
                handled =
                    self.scatter_group_via_block_index(shard_idx, s_start, group, &mut scatter)?;
            }

            if let Some(m) = self.metrics() {
                if handled {
                    m.block_index_groups.fetch_add(1, Ordering::Relaxed);
                } else {
                    m.full_shard_groups.fetch_add(1, Ordering::Relaxed);
                }
            }

            if !handled {
                // Full-shard fallback: decode once (cached), slice each row.
                let shard_csr = self.read_shard_cached_arc(shard_idx)?;
                for &(row, orig_pos) in group {
                    let local = (row - s_start) as usize;
                    let lo = shard_csr.indptr[local] as usize;
                    let hi = shard_csr.indptr[local + 1] as usize;
                    scatter(
                        orig_pos,
                        &shard_csr.indices[lo..hi],
                        &shard_csr.data[lo..hi],
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Scatter one shard's request group directly from the block index, for
    /// **row-group-framed (v2)** shards: coalesce the request into consecutive
    /// runs and decode only the touched row-groups via the block index
    /// ([`ScxReader::decode_block_index_row_runs`]). Returns `Ok(false)` (nothing
    /// scattered) for a non-framed shard so the caller falls back to full-shard
    /// decode. Output is byte-identical to a full decode + slice.
    fn scatter_group_via_block_index<F>(
        &self,
        shard_idx: usize,
        s_start: u64,
        group: &[(u64, usize)],
        scatter: &mut F,
    ) -> Result<bool>
    where
        F: FnMut(usize, &[i32], &[f32]) -> Result<()>,
    {
        // Same consecutive-run coalescing as the sidecar path (absorbs gaps and
        // duplicate requests); each run is (run_start_local, run_len).
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut run_min = (group[0].0 - s_start) as usize;
        let mut run_max = run_min;
        for &(row, _) in &group[1..] {
            let local = (row - s_start) as usize;
            if local <= run_max + 1 {
                run_max = run_max.max(local);
            } else {
                runs.push((run_min, run_max - run_min + 1));
                run_min = local;
                run_max = local;
            }
        }
        runs.push((run_min, run_max - run_min + 1));

        let Some((offset, section_type)) = self
            .shard_entry(shard_idx)
            .map(|l| (l.offset, l.section_type))
        else {
            return Ok(false);
        };
        let Some(entry) = self.reader.full_entry_at_offset(offset, section_type) else {
            return Ok(false);
        };

        // `None` ⇒ shard is not row-group-framed ⇒ caller falls back; nothing
        // scattered yet (all-or-nothing, same contract as the sidecar path).
        let decoded = match self.reader.decode_block_index_row_runs(entry, &runs)? {
            Some(d) => d,
            None => return Ok(false),
        };

        let mut gi = 0usize;
        for (&(run_start, _run_len), (indptr, indices, data)) in runs.iter().zip(&decoded) {
            while gi < group.len() {
                let (row, orig_pos) = group[gi];
                let in_run = (row - s_start) as usize - run_start;
                if in_run >= indptr.len() - 1 {
                    break;
                }
                let lo = indptr[in_run] as usize;
                let hi = indptr[in_run + 1] as usize;
                scatter(orig_pos, &indices[lo..hi], &data[lo..hi])?;
                gi += 1;
            }
        }
        debug_assert_eq!(gi, group.len(), "block-index scatter left rows unscattered");
        Ok(true)
    }

    /// Read all rows — materializes the full matrix.
    ///
    /// Used by `to_memory()` on the Python side.
    pub fn read_all(&self) -> Result<ScxCsr> {
        match &self.layer_name {
            None => self.reader.read_all_csr_shards(),
            Some(name) => self.reader.read_layer(name),
        }
    }

    /// Phase 5b: modality-scoped X read for bitmap fallbacks. Mirrors
    /// `read_all()` for the X path but honours `self.modality_id()` so
    /// per-modality `BackedCsrReader`s on multimodal files do not fold
    /// in rows from other modalities. Layer readers route through the
    /// global layer path (bitmaps are X-only, so this method is never
    /// called on layer readers in practice).
    #[cfg(feature = "deletion-vectors")]
    fn read_all_modality_scoped(&self) -> Result<ScxCsr> {
        if let Some(name) = &self.layer_name {
            return self.reader.read_layer(name);
        }
        let modality_id = self.modality_id();
        if modality_id == 0 {
            self.reader.read_all_csr_shards()
        } else {
            self.reader.read_all_csr_shards_for(modality_id)
        }
    }

    /// Phase 5b: which modality this backed reader is scoped to. Used
    /// by [`Self::gene_detection_counts`] / [`Self::cells_expressing_gene`]
    /// to look up the matching bitmap sidecar shards.
    ///
    /// Returns `0` for unimodal files and for the global X path on
    /// multimodal files; `for_modality(reader, id, ...)` returns `id`.
    pub fn modality_id(&self) -> u8 {
        self.x_sorted_entries
            .first()
            .map(|e| e.modality_id)
            .unwrap_or(0)
    }

    /// Phase 5b: are per-modality detection bitmap shards present on
    /// disk for every CSR shard? Fast path predicate for
    /// [`Self::gene_detection_counts`] — if `false`, the helpers fall
    /// back to a CSR scan.
    #[cfg(feature = "deletion-vectors")]
    pub fn has_full_bitmap_coverage(&self) -> bool {
        if !self.reader.header().has_bitmap() {
            return false;
        }
        let modality_id = self.modality_id();
        self.reader.bitmap_shard_count(modality_id) == self.x_sorted_entries.len()
            && !self.x_sorted_entries.is_empty()
    }

    /// Phase 5b: per-gene detection counts across the entire shard
    /// range this reader covers. Length is `self.n_vars()`.
    ///
    /// Fast path: when every CSR shard has a matching bitmap sidecar,
    /// sum `RoaringBitmap::len()` per gene. Otherwise falls back to a
    /// CSR scan via `read_all().to_dense()` — the slow path is the
    /// reason the auto-policy / `--bitmap=always` flag exists.
    #[cfg(feature = "deletion-vectors")]
    pub fn gene_detection_counts(&self) -> Result<Vec<u64>> {
        let n_vars = self.n_vars;
        let mut out = vec![0u64; n_vars];
        if self.has_full_bitmap_coverage() {
            let modality_id = self.modality_id();
            for shard_idx in 0..self.x_sorted_entries.len() {
                let shard = self.reader.read_bitmap_shard_for(modality_id, shard_idx)?;
                for (&gene_id, bm) in &shard.genes {
                    let g = gene_id as usize;
                    if g < n_vars {
                        out[g] = out[g].saturating_add(bm.len());
                    }
                }
            }
            return Ok(out);
        }
        // Fallback: scan CSR. Counts the distinct rows per column.
        // Modality-scoped so per-modality readers don't fold in rows
        // from other modalities on multimodal files.
        let csr = self.read_all_modality_scoped()?;
        for row in 0..csr.indptr.len().saturating_sub(1) {
            let lo = csr.indptr[row] as usize;
            let hi = csr.indptr[row + 1] as usize;
            for &col in &csr.indices[lo..hi] {
                let c = col as usize;
                if c < n_vars {
                    out[c] = out[c].saturating_add(1);
                }
            }
        }
        Ok(out)
    }

    /// Phase 5b: global row indices of cells with `gene_idx > 0`.
    /// Uses bitmap sidecars when present; otherwise scans CSR.
    #[cfg(feature = "deletion-vectors")]
    pub fn cells_expressing_gene(&self, gene_idx: u32) -> Result<Vec<u32>> {
        if (gene_idx as usize) >= self.n_vars {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "gene_idx {gene_idx} out of range (n_vars = {})",
                    self.n_vars
                ),
            )));
        }
        let mut out: Vec<u32> = Vec::new();
        if self.has_full_bitmap_coverage() {
            let modality_id = self.modality_id();
            for shard_idx in 0..self.x_sorted_entries.len() {
                let shard = self.reader.read_bitmap_shard_for(modality_id, shard_idx)?;
                if let Some(bm) = shard.cells_expressing(gene_idx) {
                    let row_start = shard.row_start as u32;
                    for local in bm {
                        out.push(row_start.saturating_add(local));
                    }
                }
            }
            return Ok(out);
        }
        // Fallback: scan CSR for the gene column. Modality-scoped so
        // per-modality readers don't pick up rows from other modalities
        // on multimodal files.
        let csr = self.read_all_modality_scoped()?;
        for row in 0..csr.indptr.len().saturating_sub(1) {
            let lo = csr.indptr[row] as usize;
            let hi = csr.indptr[row + 1] as usize;
            if csr.indices[lo..hi].contains(&(gene_idx as i32)) {
                out.push(row as u32);
            }
        }
        Ok(out)
    }

    /// Read a single decoded shard without caching.
    ///
    /// Use this for sequential streaming workloads (aggregation, col_sums,
    /// row_sums, etc.) where each shard is visited exactly once.  Avoids
    /// the ~640 MB-per-shard LRU cache overhead that is dead weight during
    /// sequential access.
    pub fn read_shard_uncached(&self, shard_idx: usize) -> Result<ScxCsr> {
        // Resolve the shard via the lite-entry table (X or layer) rather
        // than `reader.read_csr_shard(shard_idx)`, which is keyed on the
        // global CSR shard index. For `BackedCsrReader::for_modality(...)`
        // the local `shard_idx` maps to a filtered subset of the catalog,
        // so the lite-entry path correctly addresses the per-modality
        // shard at its real catalog offset.
        let lite = self
            .shard_entry(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.shard_count(),
            })?;
        let (indptr, indices, data) = self
            .reader
            .read_shard_from_entry(&lite.into_transient_full_entry())?;

        // Release page cache for the just-decoded shard. This is safe because
        // the mmap is read-only and the data has been copied into owned Vecs.
        #[cfg(unix)]
        {
            use memmap2::UncheckedAdvice;
            if let Some(entry) = self.shard_entry(shard_idx) {
                unsafe {
                    let _ = self.reader.mmap_ref().unchecked_advise_range(
                        UncheckedAdvice::DontNeed,
                        entry.offset as usize,
                        entry.length as usize,
                    );
                }
            }
        }

        let n_rows = indptr.len().saturating_sub(1);
        self.check_decoded_shard_rows(shard_idx, n_rows)?;
        self.check_decoded_shard_minor(shard_idx)?;
        Ok(ScxCsr::new_unchecked(
            (n_rows, self.n_vars),
            indptr,
            indices,
            data,
        ))
    }

    /// Reject a shard whose decoded row count disagrees with the row range the
    /// catalog assigned it.
    ///
    /// The two numbers come from independent places and nothing used to compare
    /// them. `n_rows` is derived from the decoded indptr, whose length the shard
    /// header governs — and that header sits in the section payload, which the
    /// fast read path deliberately does not re-hash (`read_shard_from_entry`
    /// documents the omission). The row range comes from the catalog stats,
    /// which the catalog's BLAKE3 *does* cover. Truncation, a partial write, or
    /// a hostile file separates them.
    ///
    /// It matters because every caller addresses a row as
    /// `row - shard_row_start`, taking the offset from the catalog and applying
    /// it to the decoded indptr: `read_rows_with`'s full-shard fallback,
    /// `read_rows`, `read_shard_cached*`, and the `ShardSource` impl. A shard
    /// whose stats claim 8 rows but decodes to 4 panicked with an
    /// index-out-of-bounds *inside the reader*.
    ///
    /// Both decode entry points call this — [`Self::decode_shard`] (the cached
    /// path) and [`Self::read_shard_uncached`] (the streaming one). They build
    /// their `ScxCsr` separately, so a check on one alone would leave the other
    /// panicking.
    ///
    /// The range comes from `self.index`, **not** from the shard's catalog
    /// entry: `ShardEntryLite::into_transient_full_entry` sets `stats: None`,
    /// because the lite entry deliberately drops the row range and every row
    /// lookup goes through [`BackedCsrIndex`].
    /// Reconcile a shard header's declared minor extent against this reader's
    /// authenticated column width, before the decoded CSR is handed out.
    ///
    /// The shared full-entry seam (`decode_shard_bytes`) does this against the
    /// catalog's stats — but the backed reader passes a *transient* entry with
    /// `stats: None` by design, so that check no-ops here and this path was the
    /// one place a payload could still widen its own `n_minor`, smuggle an index
    /// past the codec's bound, and produce an `ScxCsr` whose `shape.1` is
    /// `self.n_vars`. Rust densify then rejects it, but the default read hands
    /// those triples straight to `scipy.sparse.csr_matrix`, which accepts them —
    /// and `.toarray()` misplaces the value into another row.
    ///
    /// O(1): the codec has already bounded every index against the header's
    /// `n_minor`, so confirming that number equals the real width is enough to
    /// know the bound it enforced was the right one.
    fn check_decoded_shard_minor(&self, shard_idx: usize) -> Result<()> {
        let Some(lite) = self.shard_entry(shard_idx) else {
            return Ok(());
        };
        let sh = self
            .reader
            .read_shard_header(&lite.into_transient_full_entry())?;
        crate::shard_decode::reconcile_declared_minor(
            sh.n_minor,
            self.n_vars as u64,
            &format!("CSR shard {shard_idx}"),
        )
    }

    fn check_decoded_shard_rows(&self, shard_idx: usize, n_rows: usize) -> Result<()> {
        let Some((row_start, row_end)) = self.index.shard_range(shard_idx) else {
            return Ok(());
        };
        let expected = row_end.saturating_sub(row_start) as usize;
        if expected != n_rows {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {shard_idx} covers rows {row_start}..{row_end} ({expected} rows) \
                 per the catalog, but decoded {n_rows} rows (truncated or corrupt file)"
            )));
        }
        Ok(())
    }

    /// Read and optionally cache a single decoded shard (owned clone).
    ///
    /// Clones the decoded buffers out of the Arc so the caller can mutate.
    /// Streaming callers that only read should prefer
    /// [`Self::read_shard_cached_arc`] to avoid the clone.
    ///
    /// Public so that downstream crates (e.g. `scx-accel`) can iterate
    /// shards directly for streaming operations like SpMM.
    pub fn read_shard_cached(&self, shard_idx: usize) -> Result<ScxCsr> {
        Ok((*self.read_shard_cached_arc(shard_idx)?).clone())
    }

    /// Read and optionally cache a single decoded shard, returning a shared
    /// `Arc<ScxCsr>` — zero-copy clone from the cache. Use this for streaming
    /// read-only passes (SpMM, row/col sums, PCA shard iteration).
    ///
    /// Concurrent calls for the same `shard_idx` are deduplicated by a
    /// per-instance singleflight table: the first thread to claim the slot
    /// becomes the leader and decodes; peers wait on the slot's Condvar
    /// until the leader signals, then re-read from the cache. If the
    /// leader's decode fails (or panics — see `cache::LeaderGuard`), waiters
    /// loop and the next claimant becomes a fresh leader.
    pub fn read_shard_cached_arc(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        // A cache hit never reaches `section_bytes`, so the freshness check
        // has to be here too — otherwise a shard read once before a mutation
        // keeps being served from the LRU afterwards, which is the *most*
        // likely way to read stale data, not the least.
        self.check_fresh()?;
        // The shared cache owns the hit/singleflight/insert orchestration,
        // keyed by `(file_id, shard_idx)`; this reader owns the decode (it is
        // mmap/catalog-specific).
        let shard_cache = Arc::clone(&self.shard_cache);
        shard_cache.get_or_decode((self.file_id, shard_idx), || self.decode_shard(shard_idx))
    }

    /// Decode shard `shard_idx` from the underlying reader and trigger
    /// best-effort `MADV_WILLNEED` for upcoming sequential shards. Does **not**
    /// touch the cache — the [`SharedShardCache`] inserts the result under the
    /// budget on the singleflight leader path.
    fn decode_shard(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        // Same per-modality-correct dispatch as `read_shard_uncached`:
        // address shards by their catalog offset via `shard_entry`, not by
        // global CSR index.
        let lite = self
            .shard_entry(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.shard_count(),
            })?;
        let (indptr, indices, data) = self
            .reader
            .read_shard_from_entry(&lite.into_transient_full_entry())?;
        let n_rows = indptr.len().saturating_sub(1);

        self.check_decoded_shard_rows(shard_idx, n_rows)?;
        self.check_decoded_shard_minor(shard_idx)?;

        let csr = Arc::new(ScxCsr::new_unchecked(
            (n_rows, self.n_vars),
            indptr,
            indices,
            data,
        ));

        // Prefetch upcoming shards so the kernel starts paging them in.
        #[cfg(unix)]
        {
            use memmap2::Advice;
            let n_shards = self.shard_count();
            let end = std::cmp::min(shard_idx + 1 + self.prefetch_count, n_shards);
            for prefetch_idx in (shard_idx + 1)..end {
                if let Some(entry) = self.shard_entry(prefetch_idx) {
                    let _ = self.reader.mmap_ref().advise_range(
                        Advice::WillNeed,
                        entry.offset as usize,
                        entry.length as usize,
                    );
                }
            }
        }

        Ok(csr)
    }

    /// Decode the requested shards into the LRU cache, in parallel when the
    /// `parallel` feature is on and the configured `cache_shards > 1`.
    ///
    /// Used by the multi-shard read paths (`read_rows`, `read_row_indices`,
    /// `read_rows_with`) to overlap zstd decode of cold shards across rayon
    /// threads before the per-shard gather loop runs. After this returns,
    /// up to `cache_shards` of the requested shards have been decoded and
    /// are either in the LRU or were just served to a peer leader via the
    /// singleflight table. Any tail beyond `cache_shards` is left for the
    /// per-shard gather loop to decode sequentially — warming further would
    /// just thrash the LRU (also capped at `cache_shards`) and force the
    /// gather loop to re-decode the evicted prefix.
    ///
    /// No-op for shards already cached or in flight via the singleflight
    /// table — the up-front filter avoids paying the singleflight Condvar
    /// wait twice for the same shard inside one call.
    ///
    /// Peak transient RAM during decode is bounded by
    /// `cache_shards × decoded shard size` — matches the steady-state cap
    /// the LRU already enforces.
    pub(super) fn warm_shards(&self, shard_indices: &[usize]) -> Result<()> {
        // No cache configured → nothing to warm; the per-shard reader path
        // will decode without caching.
        if !self.shard_cache.has_cache() || shard_indices.is_empty() {
            return Ok(());
        }

        // Dedup to the not-yet-cached / not-in-flight shards for this reader's
        // `file_id`. `filter_misses` snapshots membership under the same
        // `in_flight` → `cache` lock order as `get_or_decode`, so the late
        // arrivals it can't see are still caught by the singleflight + cache
        // fast path inside `read_shard_cached_arc`.
        #[cfg_attr(not(feature = "parallel"), allow(unused_mut))]
        let mut misses = self.shard_cache.filter_misses(self.file_id, shard_indices);

        if misses.is_empty() {
            return Ok(());
        }

        // Sequential body: parallel feature off, single-slot cache, or only
        // one shard to decode. `read_shard_cached_arc` handles the
        // singleflight + insert + metrics + MADV_WILLNEED prefetch.
        #[cfg(not(feature = "parallel"))]
        {
            for &idx in &misses {
                self.read_shard_cached_arc(idx)?;
            }
            // Tail expression, not `return`: with `parallel` off the block
            // below is cfg'd away and this one is the function body's tail, so
            // `return` is what clippy calls needless — a lint only the new
            // parallel-off CI lane can see.
            Ok(())
        }

        #[cfg(feature = "parallel")]
        {
            if self.cache_shards <= 1 || misses.len() == 1 {
                for &idx in &misses {
                    self.read_shard_cached_arc(idx)?;
                }
                return Ok(());
            }

            // Cap warm at `cache_shards`. Warming more would just thrash
            // the LRU (the cache is also capped at `cache_shards`),
            // forcing the gather loop to re-decode any prefix evicted by
            // later warm work. Tail shards beyond the cap are decoded
            // sequentially by the per-shard gather loop — same behavior
            // as the pre-`warm_shards` code for those shards, so we
            // gracefully degrade to the prior throughput on wide reads
            // instead of doubling decode work.
            if misses.len() > self.cache_shards {
                misses.truncate(self.cache_shards);
            }

            // `install` on the per-reader pool when one was set, else the
            // global registry as before. This is the fork guard: a caller that
            // may be running in a forked child (the ML loader) sets a pool
            // built after the fork, whose worker threads actually exist. See
            // `set_cpu_pool`.
            let decode_all = || -> Result<()> {
                misses
                    .par_iter()
                    .try_for_each(|&idx| self.warm_one_shard(idx))
            };
            match self.cpu_pool.as_ref() {
                Some(pool) => pool.install(decode_all)?,
                None => decode_all()?,
            }
            Ok(())
        }
    }

    /// One shard of [`Self::warm_shards`]' parallel body.
    ///
    /// Split out only so the test below has a place to observe *which* pool the
    /// decode ran on; inlining it back would make that unobservable.
    #[cfg(feature = "parallel")]
    fn warm_one_shard(&self, idx: usize) -> Result<()> {
        #[cfg(test)]
        tests::note_warm_thread();
        self.read_shard_cached_arc(idx).map(|_| ())
    }
}

// ---------------------------------------------------------------------------
// ShardSource impl
// ---------------------------------------------------------------------------

impl crate::shard_source::ShardSource for BackedCsrReader {
    fn n_shards(&self) -> usize {
        self.index.n_shards()
    }

    fn n_obs(&self) -> usize {
        self.n_obs
    }

    fn n_vars(&self) -> usize {
        self.n_vars
    }

    fn read_shard(&self, shard_idx: usize) -> Result<ScxCsr> {
        self.read_shard_uncached(shard_idx)
    }

    /// Cached override: serve from the decoded-shard LRU so multi-pass
    /// kernels (out-of-core PCA) decode each shard once instead of
    /// re-decoding from disk on every pass. Mirrors the DE streaming path,
    /// which already goes through this cache.
    fn read_shard_arc(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        self.read_shard_cached_arc(shard_idx)
    }

    /// Expose the LRU's configured count cap so multi-pass callers can warn
    /// when it is smaller than the shard count (evict-and-re-decode cliff).
    fn shard_cache_capacity(&self) -> Option<usize> {
        Some(self.cache_capacity())
    }

    /// O(1) override: shard row counts live in `BackedCsrIndex`, no decode needed.
    fn max_shard_rows(&self) -> Result<usize> {
        let idx = &self.index;
        Ok((0..idx.n_shards())
            .filter_map(|i| idx.shard_range(i))
            .map(|(s, e)| (e - s) as usize)
            .max()
            .unwrap_or(0))
    }

    /// O(n_shards) over already-loaded catalog metadata — no decode, no I/O.
    ///
    /// Returns `None` when the catalog carries no shard statistics: `nnz` is
    /// then 0 for every entry (`ShardEntryLite::from_view_entry` maps a missing
    /// `stats` block to 0), and reporting `max_nnz = 0` would be an
    /// *under*-estimate, which the hint contract forbids. An genuinely all-empty
    /// matrix is indistinguishable here and also declines — harmless, since the
    /// caller then grows buffers on demand exactly as it did before.
    fn shard_size_hint(&self) -> Option<crate::ShardSizeHint> {
        let max_nnz = (0..self.shard_count())
            .filter_map(|i| self.shard_entry(i))
            .map(|e| e.nnz)
            .max()
            .unwrap_or(0);
        if max_nnz == 0 {
            return None;
        }
        Some(crate::ShardSizeHint {
            max_rows: <Self as crate::ShardSource>::max_shard_rows(self).ok()?,
            max_nnz: max_nnz as usize,
        })
    }

    // col_means_and_sum_sq: not overridden here. Generic callers (`S:
    // ShardSource`) get the trait default, which iterates read_shard_arc() and
    // so shares the LRU cache. The inherent `BackedCsrReader::col_means_and_sum_sq`
    // (used by concrete-typed callers, which Rust resolves to the inherent
    // method) is likewise cache-backed — both paths warm/reuse the same LRU.
}

// `total_variance_from_col_sq` moved to `scx_sparse::total_variance_from_col_sq`
// (pure statistics, not format I/O).
