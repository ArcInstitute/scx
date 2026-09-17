//! [`BackedCsrReader`]: on-demand row access to X, a layer, or a modality's
//! CSR shards.
//!
//! The native aggregation kernels built on it live in [`super::aggregate`].

use super::index::ShardEntryLite;
use super::*;
use crate::reader::{assemble_row_run, FramedShardLayout};

// ---------------------------------------------------------------------------
// BackedCsrReader
// ---------------------------------------------------------------------------

/// One shard's slice of a sorted row request: `sorted[start..end]` all fall in
/// shard `shard_idx`, whose first global row is `s_start`. `use_block_index` is
/// [`BackedCsrReader::block_index_eligible`]'s verdict, taken at planning time.
#[derive(Clone, Copy, Debug)]
struct RowGroup {
    start: usize,
    end: usize,
    shard_idx: usize,
    s_start: u64,
    use_block_index: bool,
}

/// One **row group** of a scattered gather: `sorted[start..end]` all fall in
/// row group `g` of shard `shard_idx`, whose first global row is `s_start` and
/// whose group begins at shard-local row `group_row_start`.
///
/// A [`RowGroup`] is one shard's slice of the request; a `GroupRun` is one row
/// group's slice of that. The split is what lets the gather decode across
/// shards in parallel: the flat run list is the whole gather's decode work in
/// one ascending sequence, and the scatter reads straight out of each decoded
/// group with no intermediate CSR.
#[derive(Clone, Copy, Debug)]
struct GroupRun {
    start: usize,
    end: usize,
    shard_idx: usize,
    s_start: u64,
    g: usize,
    group_row_start: usize,
}

impl GroupRun {
    fn new(rg: &RowGroup, layout: &FramedShardLayout, g: usize, start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            shard_idx: rg.shard_idx,
            s_start: rg.s_start,
            g,
            group_row_start: layout.span(g).row_start as usize,
        }
    }
}

/// One shard's window of a contiguous row-range read: local rows
/// `[local_start, local_end)` land at output rows starting at `out_row`, with
/// exactly `nnz` nonzeros (filled in by the indptr-only prescan).
#[derive(Clone, Copy, Debug)]
struct RangePlan {
    shard_idx: usize,
    local_start: usize,
    local_end: usize,
    out_row: usize,
    nnz: usize,
    use_row_range: bool,
}

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
    /// [`Self::warm_shards`] can chunk parallel decodes and [`Self::read_rows`]
    /// can tell a cache-sized range from a bulk one without locking the cache.
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
    /// Per-reader gate for retaining decoded row groups in the shard LRU
    /// (OPT-FORMATIO-1). Defaults to `SCX_ROW_GROUP_CACHE`; a test-only setter
    /// overrides it. With it off, a framed scattered read
    /// decodes its touched groups and drops them, as before the row-group
    /// entries existed.
    row_group_cache: bool,
    /// Lazy per-shard framing memo for [`Self::framed_layout`]: `None` once a
    /// shard is known to be unframed (legacy v1), `Some` the resolved
    /// [`FramedShardLayout`] otherwise. `block_index_eligible` probes it per
    /// prefetch candidate + per gather group on the training hot path, and the
    /// block-index decode re-uses the resolved layout instead of re-reading the
    /// header and re-parsing the block index every call. Only a successful
    /// resolve is stored, so an I/O error is retried next time.
    framed_layouts: OnceLock<Vec<OnceLock<Option<Arc<FramedShardLayout>>>>>,
    /// Memo for [`Self::stored_value_encoding`]: the widest value encoding
    /// across this reader's shard family, folded from every shard header on
    /// first request. Immutable for the reader's lifetime, like
    /// `framed_layouts`; only a successful fold is stored, so an I/O error is
    /// retried next time.
    stored_encoding: OnceLock<Option<scx_codec::ValueEncoding>>,
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
    /// Test-only barrier invoked at the top of [`Self::decode_shard`], i.e.
    /// **after** the singleflight leader has published the shard in the
    /// in-flight table and **before** the decode runs. That window is the only
    /// place a test can hold a shard in-flight deterministically, which is what
    /// `scx-loader`'s prefetch-skip counters need in order to be pinned by a
    /// predicate rather than by a sleep.
    ///
    /// Behind the opt-in `test-hooks` feature, so the field, the `Option` check
    /// and the call site all vanish from a default build.
    #[cfg(feature = "test-hooks")]
    decode_barrier: Option<Arc<dyn Fn(usize) + Send + Sync>>,
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

    /// The byte budget a **count-only** reader actually runs with.
    ///
    /// `cache_shards` bounds whole shards by count; a decoded row group has no
    /// count cap (the LRU holds them by bytes alone, see
    /// [`SharedShardCache`]), so a `usize::MAX` budget would let a long-lived
    /// reader's row groups grow without limit. Instead the count cap is
    /// converted into the bytes it already implies: `cache_shards × the
    /// largest decoded shard` (from the catalog's per-shard `nnz` + row range,
    /// sized by the same model the LRU charges). That never binds before the
    /// count cap for whole shards — every shard is at most the largest — so
    /// whole-shard behaviour is unchanged, and row groups are bounded at "the
    /// memory `cache_shards` whole shards could take".
    ///
    /// A finite `bytes_budget` is returned unchanged. Without a cache
    /// (`cache_shards == 0`), or when any shard lacks stats (no `nnz` to size
    /// by), or on an empty family, the budget stays `usize::MAX` and the LRU
    /// stays whole-shard-only ([`ShardCache::caches_groups`] is false).
    fn count_only_byte_budget(
        cache_shards: usize,
        bytes_budget: usize,
        sorted: &[&CatalogViewEntry],
    ) -> usize {
        if bytes_budget != usize::MAX || cache_shards == 0 {
            return bytes_budget;
        }
        let mut max_shard_bytes = 0usize;
        for e in sorted {
            let Some(stats) = e.stats.as_ref() else {
                return usize::MAX;
            };
            let rows = stats.major_end.saturating_sub(stats.major_start) as usize;
            let nnz = stats.nnz as usize;
            max_shard_bytes = max_shard_bytes.max(csr_component_bytes(rows + 1, nnz, nnz));
        }
        if max_shard_bytes == 0 {
            return usize::MAX;
        }
        cache_shards.saturating_mul(max_shard_bytes)
    }

    /// Create a new backed reader for X shards with both a count cap and a
    /// byte cap on the LRU. Inserts evict oldest entries until both caps
    /// are satisfied (see `cache::WeightedLruCache::put_with_budget`).
    ///
    /// `cache_shards`: count cap on cached decoded shards (0 = no cache).
    /// `bytes_budget`: byte cap on cumulative decoded shard bytes **and** the
    /// decoded row groups a framed scattered read retains; pass `usize::MAX`
    /// for count-only behavior on whole shards, which
    /// `count_only_byte_budget` turns into the bytes that count implies
    /// so row groups stay bounded too.
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
        let bytes_budget = Self::count_only_byte_budget(cache_shards, bytes_budget, &sorted);
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
            row_group_cache: row_group_cache_enabled(),
            framed_layouts: OnceLock::new(),
            stored_encoding: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
            #[cfg(feature = "test-hooks")]
            decode_barrier: None,
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
            row_group_cache: row_group_cache_enabled(),
            framed_layouts: OnceLock::new(),
            stored_encoding: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
            #[cfg(feature = "test-hooks")]
            decode_barrier: None,
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
        let bytes_budget = Self::count_only_byte_budget(cache_shards, usize::MAX, &sorted);
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
            row_group_cache: row_group_cache_enabled(),
            framed_layouts: OnceLock::new(),
            stored_encoding: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
            #[cfg(feature = "test-hooks")]
            decode_barrier: None,
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
        let bytes_budget = Self::count_only_byte_budget(cache_shards, bytes_budget, &sorted_layer);
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
            row_group_cache: row_group_cache_enabled(),
            framed_layouts: OnceLock::new(),
            stored_encoding: OnceLock::new(),
            #[cfg(feature = "parallel")]
            cpu_pool: None,
            #[cfg(feature = "test-hooks")]
            decode_barrier: None,
        }
    }

    /// Enable cache-behavior metrics on this reader. Returns a cloneable
    /// `Arc<CacheMetrics>` so callers (e.g. `IndexPlanIter`) can sample
    /// counters without going through the cache lock.
    ///
    /// **Idempotent.** Repeat calls return the same accumulating handle, not a
    /// fresh one — `ShardCache::enable_metrics` installs the counters through a
    /// `OnceLock`. A caller that wants a fresh measurement window must snapshot
    /// the counters and subtract; re-enabling does not reset them. Same
    /// contract as [`BackedCscReader::enable_metrics`] and
    /// [`BackedDenseReader::enable_metrics`].
    ///
    /// This doc comment previously said the opposite — that later calls "rebind
    /// to a fresh metrics handle". That was never true of this wrapper: it has
    /// delegated to the shared cache's `OnceLock` since the shared cache
    /// existed. It went unnoticed until the CSC and dense readers were given
    /// the same contract explicitly and the two descriptions collided.
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

    /// The resolved framing layout of `shard_idx`, or `None` if the shard is
    /// not row-group framed (legacy v1) — or on any missing entry / header
    /// parse error, in which case the caller falls back to the full-shard path
    /// and that path reports the error properly. A v4 file may mix v2-framed
    /// and legacy v1 shards, so this is **per shard**, never a file-level
    /// version check.
    ///
    /// Memoized per shard in `framed_layouts` (a shard's framing is immutable
    /// for the reader's lifetime), so the hot-path probes in
    /// [`Self::block_index_eligible`] cost one `OnceLock` load, and the
    /// block-index decode itself re-uses the header scalars, sub-stream
    /// ranges and resolved block index instead of re-deriving them per call.
    /// Resolves the shard's real catalog entry once (`full_entry_at_offset` is
    /// a linear scan — that used to run on every scattered decode) so the
    /// `stats`-authenticated minor extent is reconciled, exactly as the
    /// whole-shard decode does. Only `Ok` is memoized; an `Err` is retried.
    pub(crate) fn framed_layout(&self, shard_idx: usize) -> Option<Arc<FramedShardLayout>> {
        let memo = self
            .framed_layouts
            .get_or_init(|| (0..self.shard_count()).map(|_| OnceLock::new()).collect());
        let Some(slot) = memo.get(shard_idx) else {
            // Out-of-range shard_idx (should not happen): direct, uncached.
            return self.resolve_framed_layout(shard_idx).ok().flatten();
        };
        if let Some(resolved) = slot.get() {
            return resolved.clone();
        }
        match self.resolve_framed_layout(shard_idx) {
            Ok(resolved) => {
                // A peer may have raced us to `set`; either value is the same
                // fact, and after `set` the slot is always populated.
                let _ = slot.set(resolved);
                slot.get().cloned().flatten()
            }
            Err(_) => None,
        }
    }

    /// Uncached resolve backing [`Self::framed_layout`].
    fn resolve_framed_layout(&self, shard_idx: usize) -> Result<Option<Arc<FramedShardLayout>>> {
        let Some(&lite) = self.shard_entry(shard_idx) else {
            return Ok(None);
        };
        // The real entry carries `stats` (the authenticated minor extent) and
        // the name error messages cite; the transient one is the fallback the
        // header probe always had.
        let transient;
        let entry = match self
            .reader
            .full_entry_at_offset(lite.offset, lite.section_type)
        {
            Some(e) => e,
            None => {
                transient = lite.into_transient_full_entry();
                &transient
            }
        };
        Ok(self.reader.framed_shard_layout(entry)?.map(Arc::new))
    }

    /// Cheap "is this shard row-group framed (v2)?" probe — the memoized
    /// [`Self::framed_layout`] answer. Used by [`Self::block_index_eligible`]
    /// to gate the block-index path per shard.
    fn shard_is_framed(&self, shard_idx: usize) -> bool {
        self.framed_layout(shard_idx).is_some()
    }

    /// Override the per-reader block-index gate (default from
    /// `SCX_SCATTER_BLOCK_INDEX`). The loader's per-dataset
    /// `scatter_block_index=False` calls this so the off-switch disables the L1
    /// gather adoption too, not just the L2 prefetch skip.
    pub fn set_scatter_block_index(&mut self, enabled: bool) {
        self.scatter_block_index = enabled;
    }

    /// The live byte budget of the shard LRU this reader draws on — shared by
    /// whole shards and row groups. `usize::MAX` for a count-only cache that
    /// could not be sized (see `count_only_byte_budget`), `0` when no
    /// cache is installed. The L2 prefetcher sizes a row-group warm against
    /// this.
    pub fn cache_bytes_budget(&self) -> usize {
        self.shard_cache.bytes_budget()
    }

    /// Bytes currently resident in the shard LRU — whole shards and row groups
    /// together, the quantity the byte budget bounds.
    pub fn cache_bytes_used(&self) -> usize {
        self.shard_cache.bytes_used()
    }

    /// Whether this reader retains decoded row groups: the per-reader gate is
    /// on **and** the cache admits them (exists, finite budget).
    fn retains_row_groups(&self) -> bool {
        self.row_group_cache && self.shard_cache.caches_groups()
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

    /// Install the test-only decode barrier described on
    /// [`Self::decode_barrier`]. `f` is called with the shard index at the top
    /// of every singleflight-leader decode; parking inside it holds that shard
    /// in the in-flight table for as long as the test needs.
    ///
    /// Takes `&mut self` deliberately: like [`Self::set_cpu_pool`], it must be
    /// set before the reader is shared behind an `Arc`.
    #[cfg(feature = "test-hooks")]
    pub fn set_decode_barrier(&mut self, f: Arc<dyn Fn(usize) + Send + Sync>) {
        self.decode_barrier = Some(f);
    }

    /// Whether a shard request-group should be served by the block-index
    /// (row-group) scattered path rather than a full-shard decode. Eligible when
    /// the process-global `SCX_SCATTER_BLOCK_INDEX` kill-switch is on **and** this
    /// reader's per-reader `scatter_block_index` gate is on, the shard is not
    /// already decoded **whole** in the LRU, the requested rows are a small
    /// fraction of the shard (`group_len * ROW_RANGE_WINDOW_DIVISOR <
    /// shard_rows`), **and** the shard is actually row-group framed. This is the
    /// single source of truth for the block-index-vs-full-shard decision — it
    /// covers both the L1 `read_rows_with` gather and the L2 plan-prefetch
    /// decision, so the gather choice and the prefetch never drift. The framing
    /// probe is ordered last so it only runs for cost-eligible, uncached
    /// candidates.
    ///
    /// Only the *whole-shard* entry disqualifies: a shard whose row groups are
    /// resident stays eligible, and the block-index path then serves those
    /// groups from the LRU (`row_group_hits`). A resident whole shard is the
    /// cheapest source for any of its rows — slicing it beats decoding groups —
    /// which is why that clause stays now that the group path caches too.
    pub fn block_index_eligible(&self, shard_idx: usize, group_len: usize) -> bool {
        self.block_index_route_enabled()
            && !self.shard_cache.contains(self.file_id, shard_idx)
            && self
                .index
                .shard_range(shard_idx)
                .is_some_and(|(s, e)| (group_len as u64) * ROW_RANGE_WINDOW_DIVISOR < (e - s))
            && self.shard_is_framed(shard_idx)
    }

    /// The two static gates on the row-group route: the process-wide
    /// `SCX_SCATTER_BLOCK_INDEX` switch and this reader's `scatter_block_index`.
    /// With either off no gather can take the route, so nothing that only
    /// serves it — the row-group admission sizing in particular — should
    /// resolve a framing layout on its account.
    fn block_index_route_enabled(&self) -> bool {
        scatter_block_index_enabled() && self.scatter_block_index
    }

    /// Decoded size of shard `shard_idx` as the LRU would charge it
    /// (`(rows + 1) × 8 + nnz × 8`), from the catalog's per-shard stats — `0`
    /// when the shard is unknown or has no stats. What a whole-shard warm or
    /// full-shard gather of that shard puts into the shared budget; the plan
    /// engine counts it alongside the plan's row-group bytes.
    pub fn shard_decoded_bytes(&self, shard_idx: usize) -> usize {
        let Some(lite) = self.shard_entry(shard_idx) else {
            return 0;
        };
        let Some((s, e)) = self.index.shard_range(shard_idx) else {
            return 0;
        };
        let nnz = lite.nnz as usize;
        csr_component_bytes((e - s) as usize + 1, nnz, nnz)
    }

    /// True if any CSR shard is row-group framed (`shard_format_version >= 2`),
    /// i.e. the scattered block-index fast path can fire on at least one shard.
    /// An all-unframed (legacy v1) file full-shard-decodes every scattered
    /// gather regardless of `scatter_block_index` — callers use this at open
    /// time to warn that the fast path is inert. Early-returns on the first
    /// framed shard; reads only shard headers and block indexes (no payload),
    /// memoized per shard via `framed_layout`.
    pub fn any_shard_framed(&self) -> bool {
        // Delegated so there is exactly one answer to this question, reachable
        // from a bare `ScxReader` too — see `ScxReader::any_csr_shard_framed`.
        self.reader.any_csr_shard_framed()
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

    /// The path the wrapped reader was opened from.
    ///
    /// Exists for callers that must be able to *reopen* this file later — a
    /// bounded reader registry that evicts handles to reclaim the parsed
    /// catalog has nothing else to reopen from once the handle is gone.
    pub fn path(&self) -> &std::path::Path {
        self.reader.path()
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
    /// Decompresses only overlapping shards and assembles the result **once**:
    /// an indptr-only prescan of every overlapping shard sizes the output
    /// exactly, then each shard's window is copied straight into it — no
    /// per-shard `row_slice`, no `concatenate_csr`, so the peak is the result
    /// plus decoded shards, not twice the result.
    ///
    /// A range whose full-path shards fit the LRU (`≤ cache_shards`) is warmed
    /// once and copied out of the cache, as before. A **bulk** range — more
    /// full shards than the LRU holds, e.g. `X[:]` — would only thrash it, so
    /// resident shards are copied from the cache and every other shard is
    /// decoded *uncached*, in parallel chunks of `cache_shards`, copied out and
    /// dropped. The LRU keeps exactly the entries it had (nothing inserted,
    /// nothing evicted — `read_all` also adds nothing), though the resident
    /// shards this read copies are promoted to most-recently-used, as any hit
    /// is.
    /// Peak = result + up to `cache_shards` shards decoding in flight, **on top
    /// of** whatever the LRU already holds (itself capped at `cache_shards`) —
    /// so at most `2 × cache_shards` decoded shards beside the result when the
    /// cache is full going in, `cache_shards` when it is empty. The LRU is the
    /// caller's resident budget; this read's own transient never exceeds it.
    ///
    /// `end` must not exceed `n_obs`: a range past the last shard is an error
    /// (the pre-PR-C path returned a shorter matrix; the pre-sized assembly
    /// would otherwise emit a non-monotone `indptr`). `start >= end` is the
    /// empty matrix. Takes `&self` — cache mutation is handled via interior
    /// mutability (Mutex).
    pub fn read_rows(&self, start: u64, end: u64) -> Result<ScxCsr> {
        if start >= end {
            return Ok(Self::empty_csr(self.n_vars));
        }
        if end > self.n_obs as u64 {
            return Err(ScxError::Io(std::io::Error::other(format!(
                "row range {start}..{end} out of range (n_obs={})",
                self.n_obs
            ))));
        }

        // No early return on an empty `shard_indices`: `start < end <= n_obs`
        // holds here, so a range no shard covers is a catalog gap and must
        // fail the tiling check below, not come back as an empty matrix.
        let shard_indices = self.index.shards_for_range(start, end);

        // Plan each overlapping shard. A genuinely small window of an *uncached*
        // framed shard is decoded directly via the block-index row-range path
        // (O(window); no full-shard decode — only the touched row groups enter
        // the LRU, under the same byte budget). Everything else
        // — large windows, cached shards, unframed shards, repeated/sequential
        // access like the training loader — takes the full-decode + cache path.
        // The shared cache is keyed by `(file_id, shard)`, so membership is
        // queried per shard rather than under one held lock; the planning set
        // (shards a row range touches) is small, and `warm_shards` re-locks
        // anyway.
        let mut plans: Vec<RangePlan> = Vec::with_capacity(shard_indices.len());
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
            plans.push(RangePlan {
                shard_idx,
                local_start,
                local_end,
                out_row: (start.max(s_start) - start) as usize,
                nnz: 0,
                use_row_range,
            });
        }

        // The plans must tile `[start, end)` exactly, in order: a gap leaves
        // output rows with no shard (a non-monotone `indptr`), an overlap writes
        // one window twice. Checked positionally — each window must begin where
        // the previous ended and the last must end at `end` — because a gap and
        // an overlap of equal size sum to the right length. Either means the
        // catalog does not tile the row axis: a corrupt file, or a reader built
        // with `new` on a multimodal file whose modalities each tile
        // `[0, n_obs)` (use `for_modality` there).
        let n_rows = (end - start) as usize;
        let mut cursor = 0usize;
        for plan in &plans {
            if plan.out_row != cursor {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {} covers output rows {}.. of {start}..{end} but the previous \
                     shard ended at {cursor}: the catalog does not tile the row axis (on a \
                     multimodal file open the reader with `for_modality`)",
                    plan.shard_idx, plan.out_row
                )));
            }
            cursor += plan.local_end - plan.local_start;
        }
        if cursor != n_rows {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shards cover {cursor} of the {n_rows} rows in {start}..{end}: the catalog \
                 does not tile the row axis (on a multimodal file open the reader with \
                 `for_modality`)"
            )));
        }

        // Phase 1 — exact sizing. The indptr-only decode is O(rows) per shard
        // (the indices/data streams are never touched) and is checked against
        // the catalog's row count so a lying shard header cannot drive an
        // out-of-bounds index below.
        let mut indptr = vec![0i64; n_rows + 1];
        let mut running: i64 = 0;
        for plan in plans.iter_mut() {
            let ip = self.shard_indptr(plan.shard_idx)?;
            let base = ip[plan.local_start];
            for k in 0..(plan.local_end - plan.local_start) {
                indptr[plan.out_row + k + 1] = running + (ip[plan.local_start + k + 1] - base);
            }
            plan.nnz = (ip[plan.local_end] - base) as usize;
            running += plan.nnz as i64;
        }
        let total_nnz = running as usize;
        let mut indices = vec![0i32; total_nnz];
        let mut data = vec![0f32; total_nnz];

        // Row-group admission for this read, decided once over every row-range
        // window it will decode: a read can have two edge windows (the first
        // and last shard of the range) plus whole shards between, and two
        // windows that fit the budget one at a time but not together would
        // evict each other on every repeat of the same read (review on #528).
        // Sized from the block index, no decode; `None` layouts (unframed)
        // contribute nothing and take the full-shard path below.
        let admit_windows: Admit = Admit::from(
            !self.retains_row_groups() || {
                let bytes = plans
                    .iter()
                    .filter(|p| p.use_row_range)
                    .fold(0usize, |acc, p| {
                        acc.saturating_add(self.window_group_bytes(
                            p.shard_idx,
                            p.local_start,
                            p.local_end,
                        ))
                    });
                bytes <= self.shard_cache.bytes_budget()
            },
        );

        // Phase 2 — copy each window into its pre-carved slot. Row-range plans
        // first (they retain only their touched row groups, never the whole
        // shard; a `None` means the shard is not framed and joins the full-path
        // chunks), then full-path plans in chunks of `cache_shards`.
        let mut full_plans: Vec<usize> = Vec::with_capacity(plans.len());
        for (i, plan) in plans.iter().enumerate() {
            if plan.use_row_range {
                if let Some(run) = self.try_row_range_slice(
                    plan.shard_idx,
                    plan.local_start,
                    plan.local_end,
                    &admit_windows,
                )? {
                    Self::copy_window(
                        &run,
                        0,
                        run.n_rows(),
                        plan,
                        &indptr,
                        &mut indices,
                        &mut data,
                    )?;
                    continue;
                }
            }
            full_plans.push(i);
        }

        let chunk = self.cache_shards.max(1);
        if full_plans.len() <= chunk {
            // Fits the LRU: warm once (parallel, no-op if cached), copy out.
            let shards: Vec<usize> = full_plans.iter().map(|&i| plans[i].shard_idx).collect();
            self.warm_shards(&shards)?;
            for &i in &full_plans {
                let plan = &plans[i];
                let shard_csr = self.read_shard_cached_arc(plan.shard_idx)?;
                Self::copy_window(
                    &shard_csr,
                    plan.local_start,
                    plan.local_end,
                    plan,
                    &indptr,
                    &mut indices,
                    &mut data,
                )?;
            }
        } else {
            // Bulk range: the LRU cannot hold it, so do not run it through the
            // LRU. Resident shards are copied from the cache (a hit each, no
            // eviction); the rest are decoded uncached in parallel chunks of
            // `cache_shards`, copied into their windows and dropped.
            let (resident, cold): (Vec<usize>, Vec<usize>) = full_plans
                .iter()
                .copied()
                .partition(|&i| self.shard_cache.contains(self.file_id, plans[i].shard_idx));
            for &i in &resident {
                let plan = &plans[i];
                let shard_csr = self.read_shard_cached_arc(plan.shard_idx)?;
                Self::copy_window(
                    &shard_csr,
                    plan.local_start,
                    plan.local_end,
                    plan,
                    &indptr,
                    &mut indices,
                    &mut data,
                )?;
            }
            for cold_chunk in cold.chunks(chunk) {
                let shards: Vec<usize> = cold_chunk.iter().map(|&i| plans[i].shard_idx).collect();
                let decoded = self.decode_shards_uncached(&shards)?;
                for (&i, shard_csr) in cold_chunk.iter().zip(&decoded) {
                    let plan = &plans[i];
                    Self::copy_window(
                        shard_csr,
                        plan.local_start,
                        plan.local_end,
                        plan,
                        &indptr,
                        &mut indices,
                        &mut data,
                    )?;
                }
            }
        }

        Ok(ScxCsr::new_unchecked(
            (n_rows, self.n_vars),
            indptr,
            indices,
            data,
        ))
    }

    /// Decode `shard_indices` without touching the LRU — in parallel on the
    /// reader's pool (or rayon's global registry) when the `parallel` feature
    /// is on and there is more than one shard, else sequentially. Peak is
    /// `shard_indices.len()` decoded shards **in addition to** the LRU's
    /// resident entries; callers chunk to `cache_shards` accordingly.
    fn decode_shards_uncached(&self, shard_indices: &[usize]) -> Result<Vec<ScxCsr>> {
        #[cfg(feature = "parallel")]
        {
            if shard_indices.len() > 1 {
                let decode_all = || -> Result<Vec<ScxCsr>> {
                    shard_indices
                        .par_iter()
                        .map(|&idx| self.read_shard_uncached(idx))
                        .collect()
                };
                return match self.cpu_pool.as_ref() {
                    Some(pool) => pool.install(decode_all),
                    None => decode_all(),
                };
            }
        }
        shard_indices
            .iter()
            .map(|&idx| self.read_shard_uncached(idx))
            .collect()
    }

    fn empty_csr(n_vars: usize) -> ScxCsr {
        ScxCsr::new_unchecked((0, n_vars), vec![0], vec![], vec![])
    }

    /// Copy rows `[local_start, local_end)` of `csr` into `plan`'s window of the
    /// output, checking that the decoded shard holds exactly the nonzeros the
    /// indptr-only prescan counted for that window.
    fn copy_window(
        csr: &ScxCsr,
        local_start: usize,
        local_end: usize,
        plan: &RangePlan,
        indptr: &[i64],
        indices: &mut [i32],
        data: &mut [f32],
    ) -> Result<()> {
        let lo = csr.indptr[local_start] as usize;
        let hi = csr.indptr[local_end] as usize;
        if hi - lo != plan.nnz {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {}: the indptr-only prescan counted {} nonzeros in rows {}..{} but \
                 the decoded shard holds {} (corrupt or inconsistent shard)",
                plan.shard_idx,
                plan.nnz,
                plan.local_start,
                plan.local_end,
                hi - lo
            )));
        }
        let dst = indptr[plan.out_row] as usize;
        indices[dst..dst + plan.nnz].copy_from_slice(&csr.indices[lo..hi]);
        data[dst..dst + plan.nnz].copy_from_slice(&csr.data[lo..hi]);
        Ok(())
    }

    /// Indptr-only decode of shard `shard_idx` — every codec, framed or not —
    /// checked against the catalog's row count. Never touches the LRU, so a
    /// prescan between planning and warming cannot change which groups
    /// [`Self::block_index_eligible`] admits.
    fn shard_indptr(&self, shard_idx: usize) -> Result<Vec<i64>> {
        let lite = self
            .shard_entry(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.shard_count(),
            })?;
        let ip = self
            .reader
            .read_shard_indptr_from_entry(&lite.into_transient_full_entry())?;
        self.check_decoded_shard_rows(shard_idx, ip.len().saturating_sub(1))?;
        Ok(ip)
    }

    /// Decode just rows `[local_start, local_end)` of shard `shard_idx` directly
    /// from its row-group block index (O(window)), returning `None` when the
    /// shard is not row-group framed so the caller can fall back to a full-shard
    /// decode. Byte-identical to decoding the whole shard and slicing. The
    /// touched groups go through the row-group LRU ([`Self::row_group`]), so a
    /// repeated small window is served without a decode.
    fn try_row_range_slice(
        &self,
        shard_idx: usize,
        local_start: usize,
        local_end: usize,
        admit: &Admit,
    ) -> Result<Option<ScxCsr>> {
        let Some(layout) = self.framed_layout(shard_idx) else {
            return Ok(None);
        };
        // A cache hit never reaches `section_bytes`, so the freshness check
        // has to be here too — see `read_shard_cached_arc`.
        self.check_fresh()?;
        let n = local_end - local_start;
        layout.check_run(local_start, n)?;
        // `admit` is `read_rows`' verdict over every window of the read — row
        // groups are fixed-height, not fixed-nnz, so "a quarter of the rows"
        // is not "a quarter of the bytes", and one window's fit says nothing
        // about the read's other window.
        let (indptr, indices, data) = assemble_row_run(&layout, local_start, n, |g| {
            self.row_group(shard_idx, &layout, g, admit)
        })?;
        Ok(Some(ScxCsr::new_unchecked(
            (n, self.n_vars),
            indptr,
            indices,
            data,
        )))
    }

    /// Row group `g` of framed shard `shard_idx`. A resident group is a hit
    /// either way. On a miss, an **admitted** lookup decodes once across
    /// concurrent callers (the same singleflight as whole shards) and inserts
    /// under the byte budget; a **non-admitted** one — the caller saying this
    /// read's working set does not fit, see
    /// [`Self::gather_row_groups_fit_budget`] — decodes independently, with no
    /// insert and no singleflight slot, and counts as a miss. With row-group
    /// retention off (the `SCX_ROW_GROUP_CACHE` gate, a count-only cache, or
    /// no cache) it decodes uncached and touches no counter.
    fn row_group(
        &self,
        shard_idx: usize,
        layout: &FramedShardLayout,
        g: usize,
        admit: &Admit,
    ) -> Result<Arc<ScxCsr>> {
        self.row_group_inner(shard_idx, layout, g, admit, /*probed_miss*/ false)
    }

    /// [`Self::row_group`], with `probed_miss` saying whether the caller has
    /// already asked the cache for this key and been told no.
    ///
    /// One body rather than two near-copies (review on #540): the only
    /// difference is that a caller which has just probed does not need the
    /// non-admitted branch to take the LRU mutex a second time — which on the
    /// miss-heavy scattered path is nearly every run. The admitted branch always
    /// goes through `get_or_decode`, whose own probe is part of the
    /// single-flight protocol and cannot be skipped.
    fn row_group_inner(
        &self,
        shard_idx: usize,
        layout: &FramedShardLayout,
        g: usize,
        admit: &Admit,
        probed_miss: bool,
    ) -> Result<Arc<ScxCsr>> {
        if !self.retains_row_groups() {
            return self.reader.decode_framed_row_group(layout, g).map(Arc::new);
        }
        let key = CacheKey::Group(self.file_id, shard_idx, g);
        let admitted = admit.admits(self.file_id, shard_idx, g);
        self.charge_verdict(admit, shard_idx, layout, g);
        if admitted {
            let shard_cache = Arc::clone(&self.shard_cache);
            return shard_cache.get_or_decode(key, || {
                self.reader.decode_framed_row_group(layout, g).map(Arc::new)
            });
        }
        // Not admitted: a resident group is still a hit, but a miss decodes
        // uncached — no insert, and no singleflight either (a slot whose
        // leader inserts nothing would make every waiter re-decode in turn;
        // concurrent non-admitted decodes of one group run independently
        // instead, which is what "not retained" means).
        if !probed_miss {
            if let Some(rg) = self.shard_cache.get_cached(key) {
                return Ok(rg);
            }
        }
        self.shard_cache.note_uncached_miss(CacheKind::RowGroup);
        self.reader.decode_framed_row_group(layout, g).map(Arc::new)
    }

    /// Record what the verdict decided about one row-group **lookup**.
    ///
    /// Once per **group lookup** — the same unit `row_group_hits` /
    /// `row_group_misses` count, i.e. each time this reader is asked for a
    /// group, not each time a row is requested (one consultation serves every
    /// row of its run). What survived the byte budget on top of that is
    /// `row_group_bytes_inserted`.
    ///
    /// ⚠️ Extracted so the resident fast path in `decode_group_runs` charges it
    /// too. That path bypasses [`Self::row_group`], and while the accounting
    /// lived inline there a hit contributed to NEITHER counter — on a
    /// 0.99-hit-rate gather the pair described 391 cold lookups instead of
    /// ~38,900, silently, and reruns would not have been comparable with the
    /// committed capture. Found by review on #540 (codex and Cursor Agent,
    /// independently).
    fn charge_verdict(
        &self,
        admit: &Admit,
        shard_idx: usize,
        layout: &FramedShardLayout,
        g: usize,
    ) {
        if let Some(m) = self.metrics() {
            let bytes = layout.group_bytes(g) as u64;
            if admit.admits(self.file_id, shard_idx, g) {
                m.admitted_group_bytes.fetch_add(bytes, Ordering::Relaxed);
            } else {
                m.rejected_group_bytes.fetch_add(bytes, Ordering::Relaxed);
            }
        }
    }

    /// Whether the row groups a gather's block-index groups touch all fit the
    /// LRU's byte budget — the admission rule `scatter_groups` applies once per
    /// gather. `true` when the reader retains no row groups (nothing to decide)
    /// and when nothing takes the block-index path.
    fn gather_row_groups_fit_budget(
        &self,
        sorted_pairs: &[(u64, usize)],
        groups: &[RowGroup],
    ) -> bool {
        if !self.retains_row_groups() {
            return true;
        }
        let mut planned = 0usize;
        for g in groups {
            if !g.use_block_index {
                // The shards this gather takes WHOLE count too, exactly as the
                // plan-level rule counts them (`PrefetchEngine::plan_footprint`).
                // They share one budget with the row groups, and
                // `read_shard_cached_arc` inserts them whether or not row groups
                // are admitted — so a mixed gather whose groups fit on their own
                // but not beside its whole shards would evict one with the other
                // on every repeat. Counted unconditionally, resident or not, for
                // the same reason the plan rule does: residency is not stable
                // between the decision and the read.
                planned = planned.saturating_add(self.shard_decoded_bytes(g.shard_idx));
                continue;
            }
            let Some(layout) = self.framed_layout(g.shard_idx) else {
                continue;
            };
            // `sorted_pairs` is row-sorted, so the group's rows are too — no
            // copy, no re-sort.
            let locals = sorted_pairs[g.start..g.end]
                .iter()
                .map(|&(r, _)| (r - g.s_start) as usize);
            planned = planned.saturating_add(Self::planned_bytes_sorted(&layout, locals));
        }
        // Against the WHOLE budget, not against its free bytes. A resident
        // entry is displaceable — `evict_bytes_for` makes room — so it is not a
        // claim on the budget, and subtracting `bytes_used()` would refuse every
        // gather once the cache reached its budget, which is the steady state of
        // a correctly sized cache. Falsified rather than assumed: see
        // `a_full_cache_does_not_refuse_a_small_gather`, and note that the
        // subtraction also reddens the pre-existing
        // `row_group_lru_evicts_to_fit_the_budget`.
        planned <= self.shard_cache.bytes_budget()
    }

    /// Bytes of the row groups the contiguous shard-local window
    /// `[local_start, local_end)` touches, from the block index; `0` for an
    /// unframed shard or an empty window.
    fn window_group_bytes(&self, shard_idx: usize, local_start: usize, local_end: usize) -> usize {
        if local_end <= local_start {
            return 0;
        }
        let Some(layout) = self.framed_layout(shard_idx) else {
            return 0;
        };
        if local_end > layout.n_major {
            return 0;
        }
        let (g0, g1) = (
            layout.find_group(local_start),
            layout.find_group(local_end - 1),
        );
        (g0..=g1).fold(0usize, |acc, g| acc.saturating_add(layout.group_bytes(g)))
    }

    /// Bytes of the distinct row groups the **ascending** shard-local `locals`
    /// touch, sized from the block index (`FramedShardLayout::group_bytes`).
    /// Consecutive rows in one group are counted once; no allocation.
    fn planned_bytes_sorted(
        layout: &FramedShardLayout,
        locals: impl Iterator<Item = usize>,
    ) -> usize {
        let mut planned = 0usize;
        let mut last: Option<usize> = None;
        for local in locals {
            let g = layout.find_group(local);
            if last != Some(g) {
                planned = planned.saturating_add(layout.group_bytes(g));
                last = Some(g);
            }
        }
        planned
    }

    /// Bytes the row groups `rows` (global row ids in `shard_idx`) touch would
    /// occupy in the LRU, sized from the block index **without decoding**.
    /// `0` when the shard is unframed, out of range, or this reader does not
    /// retain row groups — i.e. when a warm could not retain anything. The L2
    /// prefetcher sums this over a plan and warms only a plan that fits its
    /// share of the budget; see `scx-loader`'s plan engine.
    pub fn planned_row_group_bytes(&self, shard_idx: usize, rows: &[u64]) -> usize {
        // Expressed as a fold over `touched_row_groups` rather than as its own
        // walk of the block index. Two walks are how a sizing decision and an
        // admission decision come to disagree about a group boundary: the
        // prefetcher sizes a plan with this, and the reuse-signal verdict names
        // the very same groups as cache keys with that. One walk, two answers.
        self.touched_row_groups(shard_idx, rows)
            .into_iter()
            .fold(0usize, |acc, (_, bytes)| acc.saturating_add(bytes))
    }

    /// Distinct row groups the `rows` of `shard_idx` touch — ascending,
    /// deduplicated, each paired with the decoded byte size the block index
    /// stamps for it (`FramedShardLayout::group_bytes`, the same model the LRU
    /// charges once the group is decoded). No decode, no LRU touch.
    ///
    /// Empty under exactly the gates [`Self::planned_row_group_bytes`] applies
    /// — the block-index route statically or per-reader off, row-group
    /// retention off, the shard unframed or out of range — so a caller cannot
    /// derive cache keys for groups no gather would ever retain. That is the
    /// point of returning the keys and the bytes together: a caller deciding
    /// *which* groups to admit and a caller deciding *whether* the plan fits
    /// are answering two questions about one list.
    ///
    /// `rows` need not be sorted and may repeat. An out-of-shard row is
    /// **clamped** into the shard by `find_group`'s saturating search rather
    /// than rejected, which is `touched_groups`' documented contract; a caller
    /// that needs the error checks its rows first, as
    /// [`Self::warm_row_groups`] does.
    pub fn touched_row_groups(&self, shard_idx: usize, rows: &[u64]) -> Vec<(usize, usize)> {
        // Both gates before `framed_layout`, for the reason review on #528
        // gave: with the route statically off (the cell-set loader's default)
        // no gather can retain a group, and resolving every touched shard's
        // layout in order to name nothing would be new work on that path.
        if !self.block_index_route_enabled() || !self.retains_row_groups() {
            return Vec::new();
        }
        let Some(layout) = self.framed_layout(shard_idx) else {
            return Vec::new();
        };
        let Some((s_start, _)) = self.index.shard_range(shard_idx) else {
            return Vec::new();
        };
        let mut groups = Self::touched_groups(&layout, s_start, rows);
        groups.dedup();
        groups
            .into_iter()
            // `find_group` saturates to `spans.len() - 1`, which underflows to
            // `0` on a shard with no spans at all (`n_major == 0` tiles
            // `[0, 0)` with none). Indexing `spans[0]` there would panic, and a
            // reader returns rather than panics on malformed input. Filtering
            // is not reachable through `shard_range`, whose empty shard maps no
            // row — but this is a `pub` fn and the bound is cheap.
            .filter(|&g| g < layout.spans.len())
            .map(|g| (g, layout.group_bytes(g)))
            .collect()
    }

    /// Decode the row groups `rows` (global row ids in `shard_idx`) touch into
    /// the LRU ahead of a gather, so the gather serves them as
    /// `row_group_hits`. Returns the number of distinct groups fetched (hit or
    /// decoded). `Ok(0)` when the shard is unframed or this reader does not
    /// retain row groups — nothing to warm into. Rows outside the shard are an
    /// error, as on the gather.
    pub fn warm_row_groups(&self, shard_idx: usize, rows: &[u64]) -> Result<usize> {
        if !self.block_index_route_enabled() || !self.retains_row_groups() || rows.is_empty() {
            return Ok(0);
        }
        let Some(layout) = self.framed_layout(shard_idx) else {
            return Ok(0);
        };
        let (s_start, s_end) =
            self.index
                .shard_range(shard_idx)
                .ok_or(ScxError::ShardIndexOutOfBounds {
                    index: shard_idx,
                    count: self.index.n_shards(),
                })?;
        for &row in rows {
            if row < s_start || row >= s_end {
                return Err(ScxError::Io(std::io::Error::other(format!(
                    "row {row} is outside shard {shard_idx} ({s_start}..{s_end})"
                ))));
            }
            layout.check_run((row - s_start) as usize, 1)?;
        }
        self.check_fresh()?;
        let mut groups = Self::touched_groups(&layout, s_start, rows);
        groups.dedup();
        // Always admitted: the caller (the L2 prefetcher) has already checked
        // the plan fits its share of the budget before asking for a warm.
        for &g in &groups {
            self.row_group(shard_idx, &layout, g, &Admit::All)?;
        }
        Ok(groups.len())
    }

    /// Sorted (not deduplicated) group indices of `rows`, each clamped into
    /// the shard by `find_group`'s saturating search — callers that need an
    /// error for an out-of-shard row check before calling.
    fn touched_groups(layout: &FramedShardLayout, s_start: u64, rows: &[u64]) -> Vec<usize> {
        let mut groups: Vec<usize> = rows
            .iter()
            .map(|&row| layout.find_group(row.saturating_sub(s_start) as usize))
            .collect();
        groups.sort_unstable();
        groups
    }

    /// Read specific row indices as a scipy-compatible `ScxCsr`, in request
    /// order.
    ///
    /// `rows` may contain duplicates (each occurrence is its own output row)
    /// and need not be sorted. Every row must be in range: an out-of-range row
    /// is an error, the same contract as [`Self::read_rows_with`] (it used to
    /// be dropped silently, so a caller could get fewer rows than it asked for).
    ///
    /// Two passes, one allocation of the result: an indptr-only prescan of each
    /// touched shard gives the exact per-row lengths — hence an exact
    /// request-order `indptr` — and the [`Self::read_rows_with`] scatter then
    /// copies every row straight into its window. Each touched shard is
    /// decoded once; a sparse request group on a row-group-framed shard that
    /// is not resident whole is decoded by row group through the block index
    /// (see [`Self::block_index_eligible`]), and the touched groups are
    /// retained in the LRU under the shared byte budget, so repeated small
    /// gathers over one region are served from cache rather than re-decoding
    /// groups or paying a full-shard decode each. Peak memory is the result
    /// plus the shard cache (whole shards and row groups, one budget), plus up
    /// to `cache_shards` shards decoding in flight while [`Self::warm_shards`]
    /// fills that cache (a full LRU is evicted only as each new shard lands, so
    /// at most `2 × cache_shards` decoded shards sit beside the result) and one
    /// row group decoding — not a per-row `ScxCsr` per requested row and a
    /// second copy of the result, as before.
    pub fn read_row_indices(&self, rows: &[u64]) -> Result<ScxCsr> {
        if rows.is_empty() {
            return Ok(Self::empty_csr(self.n_vars));
        }

        let sorted = Self::sort_rows(rows);
        let groups = self.plan_row_groups(&sorted)?;

        // Phase 1 — exact per-row lengths in request order, then prefix-sum.
        let mut indptr = vec![0i64; rows.len() + 1];
        for g in &groups {
            let ip = self.shard_indptr(g.shard_idx)?;
            for &(row, pos) in &sorted[g.start..g.end] {
                let local = (row - g.s_start) as usize;
                indptr[pos + 1] = ip[local + 1] - ip[local];
            }
        }
        for i in 1..indptr.len() {
            indptr[i] += indptr[i - 1];
        }
        let nnz = indptr[rows.len()] as usize;

        // Phase 2 — scatter each row into its pre-carved window.
        let mut indices = vec![0i32; nnz];
        let mut data = vec![0f32; nnz];
        let mut fired = 0usize;
        let mut copied = 0usize;
        self.scatter_groups(&sorted, &groups, None, |pos, idx, val| {
            let lo = indptr[pos] as usize;
            let hi = indptr[pos + 1] as usize;
            if idx.len() != hi - lo || val.len() != hi - lo {
                return Err(ScxError::InvalidCatalog(format!(
                    "row {} (request position {pos}): the indptr-only prescan counted {} \
                     nonzeros but the decoded shard holds {} indices / {} values",
                    rows[pos],
                    hi - lo,
                    idx.len(),
                    val.len()
                )));
            }
            indices[lo..hi].copy_from_slice(idx);
            data[lo..hi].copy_from_slice(val);
            fired += 1;
            copied += idx.len();
            Ok(())
        })?;
        if fired != rows.len() || copied != nnz {
            return Err(ScxError::InvalidCatalog(format!(
                "row gather scattered {fired} of {} rows and {copied} of {nnz} nonzeros",
                rows.len()
            )));
        }

        Ok(ScxCsr::new_unchecked(
            (rows.len(), self.n_vars),
            indptr,
            indices,
            data,
        ))
    }

    /// Read specific row indices, invoking `scatter` once per row with
    /// zero-copy `(indices, data)` views into the decoded shard.
    ///
    /// For each request `rows[i]`, calls `scatter(i, indices, data)` where
    /// `(indices, data)` are slices into the cached shard's CSR for that row.
    /// Each touched shard is decoded once via the LRU cache. The `i` argument
    /// is the original position in `rows`, so callers write to a dense output
    /// buffer indexed by request order — **which is the only ordering
    /// guarantee**. Scatter calls do NOT fire sorted by row: see
    /// [`Self::read_rows_with_admission`] for the per-pass order and why.
    ///
    /// Allocates no intermediate `ScxCsr` and does no per-row `row_slice`
    /// — the per-shard request sub-slice is found via binary search on the
    /// shard ranges (O(R log S) total grouping cost). Use this for dense-gather
    /// hot paths (ML training, paired-batch readers) where the consumer
    /// owns the dense output. Callers that need a `ScxCsr` (scipy interop)
    /// should use [`Self::read_row_indices`], which is built on the same
    /// planner and scatter.
    ///
    /// `rows` may contain duplicates; each occurrence triggers one
    /// `scatter` call. Empty `rows` is a no-op. A row outside every shard
    /// range is an error (shared with `read_row_indices`).
    pub fn read_rows_with<F>(&self, rows: &[u64], scatter: F) -> Result<()>
    where
        F: FnMut(usize, &[i32], &[f32]) -> Result<()>,
    {
        self.read_rows_with_admission(rows, None, scatter)
    }

    /// [`Self::read_rows_with`] with the row-group admission decided by the
    /// caller. `None` decides per gather (this call's groups must fit the
    /// whole byte budget); `Some(admit)` is a verdict taken over a larger
    /// working set — `scx-loader`'s plan engine decides once per plan, over
    /// every gather the plan will make and against the plan's share of the
    /// budget, so the L1 gathers and the L2 warm cannot disagree and a plan of
    /// many individually-fitting gathers whose union does not fit cannot churn
    /// the LRU. A resident group is served as a hit either way;
    /// [`Admit::None`] only stops misses from being inserted, and
    /// [`Admit::Groups`] stops all but the named keys.
    /// ⚠️ **Scatter order is per PASS, not sorted by row.** `scatter_groups`
    /// serves every full-shard fallback first and every block-index row group
    /// second, so on a MIXED request a later full-shard row fires before an
    /// earlier block-index one. Every in-tree consumer addresses its output by
    /// the `orig_pos` this hands it (`index_plan`, `sparse_cellset`,
    /// `read_row_indices`) and is unaffected; a caller that assumed monotonic
    /// row order across a mixed gather is not. The contiguous `read_rows` path
    /// has always been range-first / full-second, so this makes the two
    /// consistent rather than introducing the split — but it IS a change to
    /// what the pre-split docs on this method promised. Review on #540.
    pub fn read_rows_with_admission<F>(
        &self,
        rows: &[u64],
        admit_row_groups: Option<&Admit>,
        scatter: F,
    ) -> Result<()>
    where
        F: FnMut(usize, &[i32], &[f32]) -> Result<()>,
    {
        if rows.is_empty() {
            return Ok(());
        }
        let sorted = Self::sort_rows(rows);
        let groups = self.plan_row_groups(&sorted)?;
        self.scatter_groups(&sorted, &groups, admit_row_groups, scatter)
    }

    /// `(row, orig_pos)` sorted by row so duplicates / requests for the same
    /// shard are contiguous and each shard decode happens once. Stable, so
    /// duplicates keep request order among themselves.
    fn sort_rows(rows: &[u64]) -> Vec<(u64, usize)> {
        let mut sorted_pairs: Vec<(u64, usize)> =
            rows.iter().enumerate().map(|(i, &r)| (r, i)).collect();
        sorted_pairs.sort_by_key(|&(r, _)| r);
        sorted_pairs
    }

    /// Group a sorted request by shard and decide each group's decode
    /// strategy: block-index row groups vs full-shard decode — same policy as
    /// the contiguous `read_rows` planner. The block-index path is used only
    /// when the shard isn't already decoded AND the requested rows are a small
    /// fraction of the shard (so O(rows) block-index decode beats one full
    /// 16k-row shard decode). Scattered cell-set gather hits the block-index
    /// path; sequential / cached reads keep the full-shard path.
    ///
    /// `shard_for_row` (not `shards_for_indices`) is what makes an
    /// out-of-range row an error rather than a dropped row. Planning happens
    /// **before** any warming — that is what lets the `!cached` test in
    /// [`Self::block_index_eligible`] mean something: [`Self::scatter_groups`]
    /// then warms ONLY the full-path shards, so block-index-group shards stay
    /// undecoded and the row-group path is taken.
    fn plan_row_groups(&self, sorted_pairs: &[(u64, usize)]) -> Result<Vec<RowGroup>> {
        let mut groups: Vec<RowGroup> = Vec::new();
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
            groups.push(RowGroup {
                start,
                end,
                shard_idx,
                s_start,
                use_block_index,
            });
            start = end;
        }
        Ok(groups)
    }

    /// Warm the full-path shards of `groups` (in parallel, capped at
    /// `cache_shards`), then scatter every group: eligible groups through the
    /// block index, the rest from the cached full-shard decode. Metrics count
    /// one `block_index_groups` / `full_shard_groups` per group.
    fn scatter_groups<F>(
        &self,
        sorted_pairs: &[(u64, usize)],
        groups: &[RowGroup],
        admit_row_groups: Option<&Admit>,
        mut scatter: F,
    ) -> Result<()>
    where
        F: FnMut(usize, &[i32], &[f32]) -> Result<()>,
    {
        // Pre-decode the cold full-path shards in parallel (no-op if all cached
        // or if every group took the block-index path).
        let full_shards: Vec<usize> = groups
            .iter()
            .filter(|g| !g.use_block_index)
            .map(|g| g.shard_idx)
            .collect();
        self.warm_shards(&full_shards)?;

        // Row-group admission: retained groups must fit, or a working set
        // larger than the cache inserts each group and evicts it before the
        // next gather could hit it — a scan larger than the cache, which an
        // LRU makes strictly worse (eviction churn + resident bytes for zero
        // hits; measured at 0 hits on tabula_sapiens_100k under
        // `read_scattered`'s 2-shard budget). A caller that sees a larger
        // working set than this one call — the plan loaders, whose plan spans
        // several gathers and whose prefetcher holds `lookahead` plans' warms
        // alongside — decides once per plan and passes the verdict in
        // (`read_rows_with_admission`); a standalone gather decides for itself
        // against the whole budget. Sized from the block index, no decode.
        let own_verdict: Admit;
        let admit_groups: &Admit = match admit_row_groups {
            Some(a) => a,
            None => {
                own_verdict = Admit::from(self.gather_row_groups_fit_budget(sorted_pairs, groups));
                &own_verdict
            }
        };

        // Pass 1 — classify, and serve every full-shard fallback in place.
        //
        // The block-index groups are only *collected* here, into one flat
        // ascending list of the distinct row groups the whole gather will
        // decode. Pass 2 then decodes them a chunk at a time in parallel.
        // Splitting the passes is what makes the parallelism gather-wide:
        // a §6-shaped 512-row random plan touches ~480 groups spread across
        // shards, and decoding them one shard's worth at a time — which is all
        // the per-shard walk could ever overlap — leaves most of that serial.
        let mut runs: Vec<GroupRun> = Vec::new();
        for g in groups {
            let group = &sorted_pairs[g.start..g.end];

            // Strategy: on an eligible (sparse, cache-cold, framed) group, decode
            // only the touched row-groups via the codec-agnostic block index;
            // otherwise fall back to a full-shard decode.
            let mut handled = false;
            if g.use_block_index {
                handled = self.collect_group_runs(g, group, &mut runs)?;
            }

            // Counted per SHARD REQUEST GROUP, exactly as before the split —
            // `block_index_adoption_rate` floors and the codec sweep's
            // `block_index_groups == 2` then `== 4` are built on that meaning,
            // not on the number of row groups decoded.
            if let Some(m) = self.metrics() {
                if handled {
                    m.block_index_groups.fetch_add(1, Ordering::Relaxed);
                } else {
                    m.full_shard_groups.fetch_add(1, Ordering::Relaxed);
                }
            }

            if !handled {
                // Full-shard fallback: decode once (cached), slice each row.
                let shard_csr = self.read_shard_cached_arc(g.shard_idx)?;
                for &(row, orig_pos) in group {
                    let local = (row - g.s_start) as usize;
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

        // Pass 2 — decode the block-index runs in chunks, scatter each chunk.
        self.scatter_group_runs(sorted_pairs, &runs, admit_groups, &mut scatter)
    }

    /// Split one shard request group into its distinct row groups, appending a
    /// [`GroupRun`] per group, and validate the request against the shard.
    ///
    /// `Ok(false)` — nothing appended — for a shard that is not row-group
    /// framed, so the caller falls back to a full-shard decode. All-or-nothing,
    /// the same contract the per-shard walk had.
    fn collect_group_runs(
        &self,
        g: &RowGroup,
        group: &[(u64, usize)],
        out: &mut Vec<GroupRun>,
    ) -> Result<bool> {
        let Some(layout) = self.framed_layout(g.shard_idx) else {
            return Ok(false);
        };
        // A cache hit never reaches `section_bytes`, so the freshness check has
        // to be here too — see `read_shard_cached_arc`.
        self.check_fresh()?;
        // Validate the request against the shard's own row count before any
        // group indexing — `group` is built from `ShardStats` ranges, which are
        // not otherwise checked against the header. Rows are ascending and
        // `>= s_start` by construction (`plan_row_groups`), so the last one is
        // the only bound that can fail.
        if let Some(&(max_row, _)) = group.last() {
            layout.check_run((max_row - g.s_start) as usize, 1)?;
        }

        // Rows are ascending within the request group and `find_group` is
        // monotone, so each row group's rows are a contiguous run — one pass,
        // no sort, no map.
        let mut run_start = 0usize;
        let mut current: Option<usize> = None;
        for (i, &(row, _)) in group.iter().enumerate() {
            let rg = layout.find_group((row - g.s_start) as usize);
            match current {
                Some(cur) if cur == rg => {}
                Some(cur) => {
                    out.push(GroupRun::new(
                        g,
                        &layout,
                        cur,
                        g.start + run_start,
                        g.start + i,
                    ));
                    run_start = i;
                    current = Some(rg);
                }
                None => current = Some(rg),
            }
        }
        if let Some(cur) = current {
            out.push(GroupRun::new(
                g,
                &layout,
                cur,
                g.start + run_start,
                g.start + group.len(),
            ));
        }
        Ok(true)
    }

    /// Decode `runs` a chunk at a time — in parallel on the reader's pool when
    /// the chunk holds more than one group — and scatter each chunk's rows
    /// before the next is decoded.
    ///
    /// **Chunked, not all-at-once.** A gather whose verdict is [`Admit::All`]
    /// could hold everything, since admission has just certified the footprint
    /// fits the budget — but a non-admitted gather is by definition *over* it,
    /// and that is exactly the gather with the most groups to overlap. Holding
    /// one chunk bounds peak at `chunk × max group bytes` in both cases.
    ///
    /// Decoded groups are held here rather than left to the LRU because a
    /// non-admitted [`Self::row_group`] inserts nothing: pre-decoding into a
    /// cache that will not keep them would simply be decoding each group twice.
    fn scatter_group_runs<F>(
        &self,
        sorted_pairs: &[(u64, usize)],
        runs: &[GroupRun],
        admit: &Admit,
        scatter: &mut F,
    ) -> Result<()>
    where
        F: FnMut(usize, &[i32], &[f32]) -> Result<()>,
    {
        if runs.is_empty() {
            return Ok(());
        }
        let chunk = self.group_decode_chunk();
        for window in runs.chunks(chunk) {
            let decoded = self.decode_group_runs(window, admit)?;
            for (run, rg) in window.iter().zip(decoded.iter()) {
                for &(row, orig_pos) in &sorted_pairs[run.start..run.end] {
                    let in_group = (row - run.s_start) as usize - run.group_row_start;
                    let lo = rg.indptr[in_group] as usize;
                    let hi = rg.indptr[in_group + 1] as usize;
                    scatter(orig_pos, &rg.indices[lo..hi], &rg.data[lo..hi])?;
                }
            }
        }
        Ok(())
    }

    /// One chunk's decodes. Parallel on the reader's pool (or rayon's registry)
    /// when the `parallel` feature is on and the chunk holds more than one
    /// group, else the serial loop verbatim.
    ///
    /// The pool is the reader's own, never a fresh one: a caller that may be
    /// running in a forked child (the ML loader) sets a pool built after the
    /// fork, whose worker threads actually exist. Same contract as
    /// [`Self::warm_shards`].
    fn decode_group_runs(&self, window: &[GroupRun], admit: &Admit) -> Result<Vec<Arc<ScxCsr>>> {
        // ⚠️ **Serve residents serially; parallelise only the misses.**
        //
        // A cache hit is a mutex lookup, and routing every run through rayon
        // regardless cost a MEASURED regression on the high-hit-rate path: on
        // `index_plan` / tabula the row-group hit rate is 0.9899 (≈38,500 hits
        // against 391 misses) and the random-plan scenario read
        // `gather_latency_ms_p50` 27.96 → 29.17 ms (0.952×, 1 of 12 rounds won,
        // p = 0.006) and p99 29.91 → 31.84 ms (p = 0.039) against the serial
        // arm — a reliable loss, in the phase's own committed A/B, found by
        // review reading that JSON.
        //
        // ⚠️ The miss-heavy `read_scattered` win (2.39-2.79×) was measured
        // BEFORE this split and has not been re-captured against it. The split
        // should be neutral there — almost every run is a miss, and
        // `row_group_inner(probed_miss = true)` keeps the probe from taking the LRU
        // mutex twice — but "should be neutral" is an argument, not a
        // measurement, and this comment does not claim otherwise.
        //
        // The probe is free on a miss: `get_cached` counts a hit and touches
        // recency, and counts nothing when absent, so the subsequent
        // `row_group_inner(probed_miss = true)` still records the miss exactly
        // once — without asking the cache again, which is why the flag exists.
        let mut out: Vec<Option<Arc<ScxCsr>>> = Vec::with_capacity(window.len());
        let mut misses: Vec<usize> = Vec::with_capacity(window.len());
        for (i, run) in window.iter().enumerate() {
            match self.resident_group(run) {
                Some(rg) => {
                    // A hit is still a lookup the verdict decided, and the pair
                    // is documented per lookup — so charge it here rather than
                    // leaving the fast path invisible to it.
                    if let Some(layout) = self.framed_layout(run.shard_idx) {
                        self.charge_verdict(admit, run.shard_idx, &layout, run.g);
                    }
                    out.push(Some(rg));
                }
                None => {
                    out.push(None);
                    misses.push(i);
                }
            }
        }

        #[cfg(feature = "parallel")]
        if misses.len() > 1 {
            let decode_misses = || -> Result<Vec<Arc<ScxCsr>>> {
                misses
                    .par_iter()
                    .map(|&i| self.decode_one_group_run(&window[i], admit))
                    .collect()
            };
            let decoded = match self.cpu_pool.as_ref() {
                Some(pool) => pool.install(decode_misses),
                None => decode_misses(),
            }?;
            if let Some(m) = self.metrics() {
                // Counts the groups actually DECODED in parallel, not the
                // window's length: a chunk of cache hits overlaps nothing.
                m.parallel_group_decodes
                    .fetch_add(misses.len() as u64, Ordering::Relaxed);
            }
            for (&i, rg) in misses.iter().zip(decoded) {
                out[i] = Some(rg);
            }
            return Ok(out.into_iter().map(|rg| rg.expect("filled")).collect());
        }

        for &i in &misses {
            out[i] = Some(self.decode_one_group_run(&window[i], admit)?);
        }
        Ok(out.into_iter().map(|rg| rg.expect("filled")).collect())
    }

    /// The run's row group if it is already resident, without decoding.
    ///
    /// `None` when this reader retains no row groups — there is no cache to
    /// probe, so every run is a "miss" and decodes, which is what the
    /// `SCX_ROW_GROUP_CACHE=0` arm measures.
    fn resident_group(&self, run: &GroupRun) -> Option<Arc<ScxCsr>> {
        if !self.retains_row_groups() {
            return None;
        }
        self.shard_cache
            .get_cached(CacheKey::Group(self.file_id, run.shard_idx, run.g))
    }

    /// One run's row group, through the LRU ([`Self::row_group`]) so an
    /// admitted key is retained and single-flighted exactly as before.
    fn decode_one_group_run(&self, run: &GroupRun, admit: &Admit) -> Result<Arc<ScxCsr>> {
        let layout = self
            .framed_layout(run.shard_idx)
            .expect("collect_group_runs resolved this layout");
        self.row_group_inner(
            run.shard_idx,
            &layout,
            run.g,
            admit,
            /*probed_miss*/ true,
        )
    }

    /// Groups decoded together in one chunk — the reader's pool width, so the
    /// bound on peak is one decode per worker and no worker idles inside a
    /// chunk.
    fn group_decode_chunk(&self) -> usize {
        // The A/B arm first: `SCX_ROW_GROUP_SERIAL_DECODE=1` forces one group per
        // chunk and therefore the serial path, which is the pre-change regime.
        if row_group_serial_decode_enabled() {
            return 1;
        }
        #[cfg(feature = "parallel")]
        {
            match self.cpu_pool.as_ref() {
                Some(pool) => pool.current_num_threads(),
                None => rayon::current_num_threads(),
            }
            .max(1)
        }
        #[cfg(not(feature = "parallel"))]
        1
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

    /// The `cache_shards` this reader was constructed with — the *requested*
    /// count cap on the decoded-shard LRU, so `0` means "no cache" (the cache
    /// itself clamps its capacity to ≥ 1 internally). Read back by the pyscx
    /// handles' `cache_shards` getter. There is no setter: the count is fixed
    /// when the reader is built, and a `to_anndata(backed=True)` builds its `X`
    /// reader and each layer's reader with the same count.
    pub fn cache_shards(&self) -> usize {
        self.cache_shards
    }

    /// The value encoding that describes this reader's shard family — `X`, the
    /// layer, or the modality it was scoped to, since it walks the same
    /// [`Self::shard_entry`] table every read does.
    ///
    /// Reads each shard's 76-byte header (no payload decode), folds with
    /// [`scx_codec::ValueEncoding::widest`] — a uniform family reports its own
    /// encoding, a mixed one the widest (any float ⇒ `Float32`, else the
    /// widest integer) — and memoises the answer. `Ok(None)` when the family
    /// has no shards. The encoding lives only in the shard header, not in the
    /// catalog stats (`ShardEntryLite` drops the `value_*` fields, and a
    /// `value_max` of 0 cannot tell a float shard from an all-zero one), so
    /// this is the one place a caller can learn whether the stored values are
    /// integer counts without decoding anything.
    pub fn stored_value_encoding(&self) -> Result<Option<scx_codec::ValueEncoding>> {
        // Before the memo, not after: the first fold reaches `section_bytes`,
        // which checks freshness itself, but a memo hit touches no section —
        // and an `append` can mix a wider encoding into a file whose warm memo
        // would otherwise keep answering the old one while `shape` refuses.
        self.check_fresh()?;
        if let Some(memo) = self.stored_encoding.get() {
            return Ok(*memo);
        }
        let mut encs = Vec::with_capacity(self.shard_count());
        for shard_idx in 0..self.shard_count() {
            let Some(&lite) = self.shard_entry(shard_idx) else {
                continue;
            };
            let entry = lite.into_transient_full_entry();
            let header = self.reader.read_shard_header(&entry)?;
            let enc = scx_codec::ValueEncoding::from_u8(header.value_encoding)
                .ok_or(ScxError::UnknownValueEncoding(header.value_encoding))?;
            encs.push(enc);
        }
        let widest = scx_codec::ValueEncoding::widest(&encs);
        Ok(*self.stored_encoding.get_or_init(|| widest))
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
        shard_cache.get_or_decode(CacheKey::Shard(self.file_id, shard_idx), || {
            self.decode_shard(shard_idx)
        })
    }

    /// Decode shard `shard_idx` from the underlying reader and trigger
    /// best-effort `MADV_WILLNEED` for upcoming sequential shards. Does **not**
    /// touch the cache — the [`SharedShardCache`] inserts the result under the
    /// budget on the singleflight leader path.
    fn decode_shard(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        // The singleflight leader has already published `(file_id, shard_idx)`
        // in the in-flight table by the time this runs — see
        // `SharedShardCache::get_or_decode`. Compiled out without `test-hooks`.
        #[cfg(feature = "test-hooks")]
        if let Some(barrier) = self.decode_barrier.as_ref() {
            barrier(shard_idx);
        }
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
    /// gather loop to re-decode the evicted prefix. (`read_rows` warms only a
    /// range that fits the LRU; a bulk range bypasses the cache — see
    /// [`Self::decode_shards_uncached`].)
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

// Test-only surface. Kept in its own `impl` at the END of the file: the I-ORG-1
// dedup guards scan production code with `sed '/#\[cfg(test)\]/,$d'`, so a
// `#[cfg(test)]` placed ahead of a production call site they count (the
// `get_or_decode` calls) truncates what they see and fails the guard — which
// is exactly what happened on #528 round 2.
#[cfg(test)]
impl BackedCsrReader {
    /// Override of the per-reader row-group retention gate (default from
    /// `SCX_ROW_GROUP_CACHE`, which is also the same-build A/B switch). Off, a
    /// framed scattered read decodes its touched groups and drops them — the
    /// pre-OPT-FORMATIO-1 behaviour — and none of the `row_group_*` counters
    /// move. No production caller changes it, so it is not API.
    pub(crate) fn set_row_group_cache(&mut self, enabled: bool) {
        self.row_group_cache = enabled;
    }
}
