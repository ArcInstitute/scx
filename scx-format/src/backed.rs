//! Backed (on-demand) CSR access for SCX files.
//!
//! Provides [`BackedCsrIndex`] for O(log n) shard lookups and
//! [`BackedCsrReader`] for on-demand shard decoding with optional LRU caching.
//! Used by pyscx's backed mode to implement AnnData-compatible lazy access.

use std::collections::{HashMap, HashSet};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use lru::LruCache;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scx_sparse::{ScxCsc, ScxCsr};

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::error::{Result, ScxError};
use crate::reader::ScxReader;
use crate::section::SectionType;

// ---------------------------------------------------------------------------
// BackedCsrIndex
// ---------------------------------------------------------------------------

/// Precomputed shard index for O(log n) row-range lookups.
///
/// Built once from a [`FullCatalog`] at open time.  Each entry stores
/// `row_start`, `row_end`, and a `sorted_shard_idx` — the position of the
/// shard after sort-by-`row_start` and filter-by-section-type, **not** the
/// original index in the catalog's `entries` vec. This distinction matters
/// because `BackedCsrReader::read_shard_cached` indexes
/// `self.sorted_entries[sorted_shard_idx]`, which is already in that sorted
/// order.
#[derive(Debug, Clone, Copy)]
struct ShardRange {
    row_start: u64,
    row_end: u64,
    /// Position in the sorted-and-filtered shard list (not the raw catalog
    /// entry index).
    sorted_shard_idx: usize,
}

/// Precomputed shard index for O(log n) row-range lookups.
///
/// Built once from a [`FullCatalog`] at open time.
#[derive(Debug, Clone)]
pub struct BackedCsrIndex {
    /// Sorted by `row_start`.
    shard_ranges: Vec<ShardRange>,
}

impl BackedCsrIndex {
    /// Build from a [`FullCatalog`].
    ///
    /// Extracts CSR shard entries, sorts by `row_start`, and records their
    /// position in the sorted order (which is the index used by
    /// `ScxReader::read_csr_shard`).
    pub fn from_catalog(catalog: &FullCatalog) -> Self {
        Self::from_catalog_filtered(catalog, SectionType::CsrShard, None)
    }

    /// Build from a [`FullCatalog`] for a specific layer.
    ///
    /// Extracts `LayerCsrShard` entries whose name starts with
    /// `"{layer_name}_shard_"`, sorts by `row_start`, and records their
    /// position in sorted order.
    pub fn from_catalog_layer(catalog: &FullCatalog, layer_name: &str) -> Self {
        let prefix = format!("{layer_name}_shard_");
        Self::from_catalog_filtered(catalog, SectionType::LayerCsrShard, Some(&prefix))
    }

    /// Internal: build from catalog filtering by section type and optional name prefix.
    fn from_catalog_filtered(
        catalog: &FullCatalog,
        section_type: SectionType,
        name_prefix: Option<&str>,
    ) -> Self {
        let mut shard_entries: Vec<ShardRange> = catalog
            .entries
            .iter()
            .filter(|e| {
                e.section_type == section_type && name_prefix.is_none_or(|p| e.name.starts_with(p))
            })
            .filter_map(|e| {
                e.stats.as_ref().map(|s| ShardRange {
                    row_start: s.row_start,
                    row_end: s.row_end,
                    sorted_shard_idx: 0, // filled below
                })
            })
            .collect();

        // Sort by row_start (deterministic ordering)
        shard_entries.sort_by_key(|r| r.row_start);

        // Assign sorted indices
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }

        BackedCsrIndex {
            shard_ranges: shard_entries,
        }
    }

    /// Number of shards in the index.
    pub fn n_shards(&self) -> usize {
        self.shard_ranges.len()
    }

    /// Find all shard indices that overlap `[row_start, row_end)`.
    ///
    /// Uses binary search — O(log n) in the number of shards.
    /// Returns a `Vec<usize>` of sorted shard indices.
    pub fn shards_for_range(&self, row_start: u64, row_end: u64) -> Vec<usize> {
        if row_start >= row_end || self.shard_ranges.is_empty() {
            return Vec::new();
        }

        // Binary search: find the first shard whose row_end > row_start.
        // A shard (s_start, s_end) overlaps [row_start, row_end) iff
        //   s_start < row_end  AND  s_end > row_start
        //
        // We scan from the first candidate shard onwards.
        let first = self
            .shard_ranges
            .partition_point(|r| r.row_end <= row_start);

        let mut result = Vec::new();
        for r in &self.shard_ranges[first..] {
            if r.row_start >= row_end {
                break; // no more overlapping shards
            }
            result.push(r.sorted_shard_idx);
        }
        result
    }

    /// Find all shard indices needed for a set of row indices.
    ///
    /// Sorts indices, deduplicates, then uses range lookups.
    pub fn shards_for_indices(&self, rows: &[u64]) -> Vec<usize> {
        if rows.is_empty() {
            return Vec::new();
        }

        let mut sorted_rows = rows.to_vec();
        sorted_rows.sort_unstable();
        sorted_rows.dedup();

        let mut result = Vec::new();
        let mut last_shard: Option<usize> = None;

        for &row in &sorted_rows {
            // Find the shard containing this row: shard where row_start <= row < row_end
            let pos = self.shard_ranges.partition_point(|r| r.row_start <= row);
            if pos == 0 {
                continue; // row is before all shards
            }
            let r = self.shard_ranges[pos - 1];
            if row >= r.row_start && row < r.row_end && last_shard != Some(r.sorted_shard_idx) {
                result.push(r.sorted_shard_idx);
                last_shard = Some(r.sorted_shard_idx);
            }
        }
        result
    }

    /// Get the shard range `(row_start, row_end)` for a given shard index.
    ///
    /// O(1) — shard indices are assigned sequentially during construction,
    /// so `shard_idx` is the position in the sorted `shard_ranges` vec.
    pub fn shard_range(&self, shard_idx: usize) -> Option<(u64, u64)> {
        self.shard_ranges
            .get(shard_idx)
            .map(|r| (r.row_start, r.row_end))
    }

    /// Find the shard index containing a single row, or `None` if the row
    /// falls outside every shard's range.
    ///
    /// O(log n) over `shard_ranges` via `partition_point`. Equivalent to a
    /// `shards_for_indices(&[row])` call without the sort/dedup overhead —
    /// useful when sorting plans by shard locality on a per-row basis.
    pub fn shard_for_row(&self, row: u64) -> Option<usize> {
        let pos = self.shard_ranges.partition_point(|r| r.row_start <= row);
        if pos == 0 {
            return None;
        }
        let r = &self.shard_ranges[pos - 1];
        if row >= r.row_start && row < r.row_end {
            Some(r.sorted_shard_idx)
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregation operation enum (internal)
// ---------------------------------------------------------------------------

/// Internal enum for masked column aggregation dispatch.
#[derive(Clone, Copy)]
enum AggOp {
    Sum,
    Nnz,
    Max,
    Min,
}

// ---------------------------------------------------------------------------
// Cache instrumentation: metrics + singleflight + byte-budgeted eviction
// ---------------------------------------------------------------------------

/// Atomic counters for shard-cache behavior. Opt-in via
/// [`BackedCsrReader::enable_metrics`]; cloning the returned `Arc` lets a
/// caller (e.g. `IndexPlanIter`) sample without touching the cache lock.
///
/// All counters use `Ordering::Relaxed` — the values are statistical and not
/// used for synchronization.
#[derive(Default, Debug)]
pub struct CacheMetrics {
    /// `read_shard_cached_arc` calls served from the LRU without decode.
    pub hits: AtomicU64,
    /// Calls that fell through to decode (became leader of a singleflight slot).
    pub misses: AtomicU64,
    /// Entries dropped by `WeightedLruCache::put_with_budget` to fit a new
    /// entry (count or byte cap; both share this counter).
    pub evictions: AtomicU64,
    /// Cumulative bytes inserted into the cache (estimated decoded size).
    pub bytes_inserted: AtomicU64,
    /// Calls that found a peer leader already decoding the same shard and
    /// waited on its Condvar instead of redecoding.
    pub duplicate_waiters: AtomicU64,
    /// High-water mark of `WeightedLruCache.bytes_used` since
    /// [`BackedCsrReader::enable_metrics`]. Maintained via `fetch_max` on
    /// every successful `put_with_budget`. Lets callers see whether the
    /// byte cap was actually exercised, vs. just configured generously.
    pub peak_bytes_in_cache: AtomicU64,
}

/// Per-shard rendezvous slot used by the singleflight in
/// `BackedCsrReader::read_shard_cached_arc`. The leader (first thread to
/// claim a slot) decodes; followers (peers that find the slot already in
/// `in_flight`) wait on the Condvar until `state == true`.
struct InFlightSlot {
    /// `false` while the leader is decoding; `true` once the leader has
    /// finished (success or fail) and waiters can wake.
    state: Mutex<bool>,
    cv: Condvar,
}

impl InFlightSlot {
    fn new() -> Self {
        InFlightSlot {
            state: Mutex::new(false),
            cv: Condvar::new(),
        }
    }
}

/// RAII cleanup for a singleflight leader. Held by the thread that claimed
/// an `InFlightSlot`; on drop — including via panic unwinding — it marks the
/// slot done, wakes every waiter, and removes the entry from the in-flight
/// table. Without this, a panic inside `decode_and_cache` would strand
/// peers on the Condvar forever.
struct LeaderGuard<'a> {
    in_flight: &'a Mutex<HashMap<usize, Arc<InFlightSlot>>>,
    slot: Arc<InFlightSlot>,
    key: usize,
}

impl Drop for LeaderGuard<'_> {
    fn drop(&mut self) {
        {
            let mut state = self.slot.state.lock().unwrap();
            *state = true;
        }
        self.slot.cv.notify_all();
        self.in_flight.lock().unwrap().remove(&self.key);
    }
}

/// LRU cache with both a count cap and a byte cap. Evicts oldest entries
/// until both caps are satisfied for a new insertion.
///
/// Bytes are estimated from the decoded `ScxCsr` (`indptr.len()*8 +
/// indices.len()*4 + data.len()*4`) — matches the per-component model used
/// by `IndexPlanLoader`'s memory-budget auto-tune.
struct CacheEntry {
    csr: Arc<ScxCsr>,
    bytes: usize,
}

struct WeightedLruCache {
    inner: LruCache<usize, CacheEntry>,
    /// Hard byte cap. `usize::MAX` means count-only behavior (compatible
    /// with `BackedCsrReader::new`).
    bytes_budget: usize,
    /// Cumulative bytes currently in `inner`.
    bytes_used: usize,
    /// Optional metrics handle (cloned from BackedCsrReader on construction).
    metrics: Option<Arc<CacheMetrics>>,
}

impl WeightedLruCache {
    fn new(cache_shards: usize, bytes_budget: usize) -> Self {
        let cap = NonZeroUsize::new(cache_shards).unwrap();
        WeightedLruCache {
            inner: LruCache::new(cap),
            bytes_budget,
            bytes_used: 0,
            metrics: None,
        }
    }

    fn estimate_bytes(csr: &ScxCsr) -> usize {
        csr.indptr
            .len()
            .saturating_mul(8)
            .saturating_add(csr.indices.len().saturating_mul(4))
            .saturating_add(csr.data.len().saturating_mul(4))
    }

    fn get(&mut self, key: &usize) -> Option<Arc<ScxCsr>> {
        self.inner.get(key).map(|e| Arc::clone(&e.csr))
    }

    fn contains(&self, key: &usize) -> bool {
        self.inner.contains(key)
    }

    /// Insert `csr` under `key`, evicting oldest entries until both the
    /// count cap (enforced by the inner `LruCache`) and the byte cap are
    /// satisfied. If a new entry on its own exceeds `bytes_budget`, all
    /// other entries are evicted and the new one is still inserted (the
    /// alternative — refusing to cache — would defeat the cache for any
    /// outsized shard).
    fn put_with_budget(&mut self, key: usize, csr: Arc<ScxCsr>) {
        let bytes = Self::estimate_bytes(&csr);

        // Evict by byte budget first. The LruCache's count cap is handled
        // by `LruCache::put` returning the displaced entry, which we
        // account for below. `saturating_add` keeps the comparison sound
        // even if a degenerate decoded shard pushes the sum past `usize`.
        while self.bytes_used.saturating_add(bytes) > self.bytes_budget && !self.inner.is_empty() {
            if let Some((_, evicted)) = self.inner.pop_lru() {
                self.bytes_used = self.bytes_used.saturating_sub(evicted.bytes);
                if let Some(m) = &self.metrics {
                    m.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }

        // Distinguish a same-key replacement from a true count-cap eviction.
        // `LruCache::put` returns the displaced entry in both cases; only
        // the latter should bump the eviction counter.
        let was_replace = self.inner.contains(&key);
        let entry = CacheEntry { csr, bytes };
        if let Some(displaced) = self.inner.put(key, entry) {
            self.bytes_used = self.bytes_used.saturating_sub(displaced.bytes);
            if !was_replace {
                if let Some(m) = &self.metrics {
                    m.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.bytes_used = self.bytes_used.saturating_add(bytes);
        if let Some(m) = &self.metrics {
            m.bytes_inserted.fetch_add(bytes as u64, Ordering::Relaxed);
            // High-water gauge: record the post-insert level so callers can
            // tell whether the byte cap was actually exercised. `fetch_max`
            // is monotonic so concurrent inserts converge correctly even
            // without a lock around the load.
            m.peak_bytes_in_cache
                .fetch_max(self.bytes_used as u64, Ordering::Relaxed);
        }
    }
}

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
/// **Invariant**: the shard `cache` is
/// per-`BackedCsrReader` instance, *not* a process-global `OnceCell` /
/// `static` / `lazy_static`. Per-instance state is the contract that
/// keeps this type fork-safe: a forked child that constructs its own
/// `BackedCsrReader` (e.g. via `pyscx.TrainingDataset` lazy construction
/// inside a `DataLoader` worker's `__iter__`) gets a fresh `Mutex` that
/// has no chance of being inherited from the parent in a poisoned-locked
/// state. If a future edit moves any of these fields into a global, the
/// fork-mode regression test (`pyscx/tests/test_fork_deadlock.py`) will
/// catch the resulting hang on the first cache access in the child.
pub struct BackedCsrReader {
    reader: ScxReader,
    index: BackedCsrIndex,
    n_vars: usize,
    n_obs: usize,
    /// If set, this reader targets a specific layer rather than X.
    layer_name: Option<String>,
    /// Pre-sorted catalog entries for X shards.
    x_sorted_entries: Vec<FullCatalogEntry>,
    /// Pre-sorted catalog entries for layer shards (empty for X shards).
    sorted_entries: Vec<FullCatalogEntry>,
    /// Per-instance shard cache. **Must remain per-instance** — see the
    /// "Fork safety" section in the type doc above.
    cache: Option<Mutex<WeightedLruCache>>,
    /// Per-instance singleflight table for in-flight shard decodes. Same
    /// fork-safety contract as `cache`: per-instance, never global.
    /// Holds an `Arc<InFlightSlot>` per shard currently being decoded by a
    /// peer; followers clone the Arc, drop the table lock, and wait on the
    /// slot's Condvar instead of redecoding.
    in_flight: Option<Mutex<HashMap<usize, Arc<InFlightSlot>>>>,
    /// Optional cache-behavior counters. Enable via [`Self::enable_metrics`].
    metrics: Option<Arc<CacheMetrics>>,
    /// Number of shards to prefetch with `MADV_WILLNEED` after a cache miss.
    prefetch_count: usize,
    /// Configured count cap on the LRU (0 = no cache). Mirrored here so
    /// [`Self::warm_shards`] can chunk parallel decodes without locking the
    /// cache. Only read under `cfg(feature = "parallel")`.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    cache_shards: usize,
}

impl BackedCsrReader {
    /// Create a new backed reader for X shards with a count-only cache cap.
    ///
    /// `cache_shards`: number of decoded shards to cache (0 = no cache).
    /// Equivalent to [`Self::new_with_byte_budget`] with `bytes_budget =
    /// usize::MAX` — no byte cap.
    pub fn new(reader: ScxReader, cache_shards: usize) -> Self {
        Self::new_with_byte_budget(reader, cache_shards, usize::MAX)
    }

    /// Create a new backed reader for X shards with both a count cap and a
    /// byte cap on the LRU. Inserts evict oldest entries until both caps
    /// are satisfied (see [`WeightedLruCache::put_with_budget`]).
    ///
    /// `cache_shards`: count cap on cached decoded shards (0 = no cache).
    /// `bytes_budget`: byte cap on cumulative decoded shard bytes; pass
    /// `usize::MAX` for count-only behavior.
    pub fn new_with_byte_budget(
        reader: ScxReader,
        cache_shards: usize,
        bytes_budget: usize,
    ) -> Self {
        let index = BackedCsrIndex::from_catalog(reader.catalog());
        let n_vars = reader.n_vars() as usize;
        let n_obs = reader.n_obs() as usize;
        let cache = Self::make_cache(cache_shards, bytes_budget);
        let in_flight = if cache.is_some() {
            Some(Mutex::new(HashMap::new()))
        } else {
            None
        };
        let x_sorted_entries = reader
            .catalog()
            .shards_sorted()
            .into_iter()
            .cloned()
            .collect();
        let prefetch_count = cache_shards.max(2);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: None,
            x_sorted_entries,
            sorted_entries: Vec::new(),
            cache,
            in_flight,
            metrics: None,
            prefetch_count,
            cache_shards,
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
        let index = BackedCsrIndex::from_catalog_layer(reader.catalog(), layer_name);
        let n_vars = reader.n_vars() as usize;
        // n_obs for layers is the same as for X — layer shards cover the same rows.
        let n_obs = reader.n_obs() as usize;
        let cache = Self::make_cache(cache_shards, bytes_budget);
        let in_flight = if cache.is_some() {
            Some(Mutex::new(HashMap::new()))
        } else {
            None
        };

        // Pre-compute sorted layer shard entries to avoid re-scanning catalog on every access
        let prefix = format!("{layer_name}_shard_");
        let mut sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix))
            .cloned()
            .collect();
        sorted_entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));

        let x_sorted_entries = reader
            .catalog()
            .shards_sorted()
            .into_iter()
            .cloned()
            .collect();
        let prefetch_count = cache_shards.max(2);
        BackedCsrReader {
            reader,
            index,
            n_vars,
            n_obs,
            layer_name: Some(layer_name.to_string()),
            x_sorted_entries,
            sorted_entries,
            cache,
            in_flight,
            metrics: None,
            prefetch_count,
            cache_shards,
        }
    }

    fn make_cache(cache_shards: usize, bytes_budget: usize) -> Option<Mutex<WeightedLruCache>> {
        if cache_shards > 0 {
            Some(Mutex::new(WeightedLruCache::new(
                cache_shards,
                bytes_budget,
            )))
        } else {
            None
        }
    }

    /// Enable cache-behavior metrics on this reader. Returns a cloneable
    /// `Arc<CacheMetrics>` so callers (e.g. `IndexPlanIter`) can sample
    /// counters without going through the cache lock. Subsequent calls
    /// rebind to a fresh metrics handle (intended for opt-in setup, not
    /// runtime toggling).
    pub fn enable_metrics(&mut self) -> Arc<CacheMetrics> {
        let m = Arc::new(CacheMetrics::default());
        self.metrics = Some(Arc::clone(&m));
        if let Some(ref cache_mutex) = self.cache {
            let mut c = cache_mutex.lock().unwrap();
            c.metrics = Some(Arc::clone(&m));
        }
        m
    }

    /// Borrow this reader's metrics handle, if metrics are enabled.
    pub fn metrics(&self) -> Option<&Arc<CacheMetrics>> {
        self.metrics.as_ref()
    }

    /// Peek the LRU for `shard_idx` without touching recency or decoding.
    /// Returns `false` when no cache is configured.
    pub fn cache_contains(&self, shard_idx: usize) -> bool {
        match &self.cache {
            Some(m) => m.lock().unwrap().contains(&shard_idx),
            None => false,
        }
    }

    /// Peek the singleflight table for `shard_idx`. Returns `false` when no
    /// cache (and therefore no singleflight) is configured.
    pub fn in_flight_contains(&self, shard_idx: usize) -> bool {
        match &self.in_flight {
            Some(m) => m.lock().unwrap().contains_key(&shard_idx),
            None => false,
        }
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

    /// Get the catalog entry for a shard by index.
    ///
    /// Uses `sorted_entries` for layer readers, `x_sorted_entries` for X readers.
    fn shard_entry(&self, shard_idx: usize) -> Option<&FullCatalogEntry> {
        if self.layer_name.is_some() {
            self.sorted_entries.get(shard_idx)
        } else {
            self.x_sorted_entries.get(shard_idx)
        }
    }

    /// Total number of shards.
    fn shard_count(&self) -> usize {
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

        // Pre-decode cold shards in parallel before the per-shard gather.
        // No-op when every shard is already cached.
        self.warm_shards(&shard_indices)?;

        let mut slices = Vec::with_capacity(shard_indices.len());
        for &shard_idx in &shard_indices {
            let shard_csr = self.read_shard_cached_arc(shard_idx)?;

            // Compute local row range within this shard
            let (s_start, s_end) =
                self.index
                    .shard_range(shard_idx)
                    .ok_or(ScxError::ShardIndexOutOfBounds {
                        index: shard_idx,
                        count: self.index.n_shards(),
                    })?;

            let local_start = (start.max(s_start) - s_start) as usize;
            let local_end = (end.min(s_end) - s_start) as usize;

            let sliced = shard_csr
                .row_slice(local_start, local_end)
                .map_err(|e| ScxError::Io(std::io::Error::other(e)))?;
            slices.push(sliced);
        }

        concatenate_csr(&slices, self.n_vars)
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
        concatenate_csr(&ordered, self.n_vars)
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

        // Pre-decode cold shards in parallel before the per-shard gather.
        // `shards_for_indices` silently drops out-of-range rows (vs. the
        // in-loop `shard_for_row` below which raises) — that's the right
        // behavior here: warm what's resolvable, defer the proper error
        // to the gather loop so out-of-range error semantics are preserved.
        let warm_rows: Vec<u64> = sorted_pairs.iter().map(|&(r, _)| r).collect();
        let prewarm_shards = self.index.shards_for_indices(&warm_rows);
        self.warm_shards(&prewarm_shards)?;

        // Walk by shard. partition_point finds each shard's request
        // sub-slice in O(log R) instead of the O(R) inner scan that
        // `read_row_indices` does for every touched shard.
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

            let shard_csr = self.read_shard_cached_arc(shard_idx)?;

            for &(row, orig_pos) in &sorted_pairs[start..end] {
                let local = (row - s_start) as usize;
                let lo = shard_csr.indptr[local] as usize;
                let hi = shard_csr.indptr[local + 1] as usize;
                scatter(
                    orig_pos,
                    &shard_csr.indices[lo..hi],
                    &shard_csr.data[lo..hi],
                )?;
            }

            start = end;
        }

        Ok(())
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

    /// Read a single decoded shard without caching.
    ///
    /// Use this for sequential streaming workloads (aggregation, col_sums,
    /// row_sums, etc.) where each shard is visited exactly once.  Avoids
    /// the ~640 MB-per-shard LRU cache overhead that is dead weight during
    /// sequential access.
    pub fn read_shard_uncached(&self, shard_idx: usize) -> Result<ScxCsr> {
        let (indptr, indices, data) = match &self.layer_name {
            None => self.reader.read_csr_shard(shard_idx)?,
            Some(_) => {
                let entry =
                    self.sorted_entries
                        .get(shard_idx)
                        .ok_or(ScxError::ShardIndexOutOfBounds {
                            index: shard_idx,
                            count: self.sorted_entries.len(),
                        })?;
                self.reader.read_shard_from_entry(entry)?
            }
        };

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
        Ok(ScxCsr::new_unchecked(
            (n_rows, self.n_vars),
            indptr,
            indices,
            data,
        ))
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
    /// leader's decode fails (or panics — see [`LeaderGuard`]), waiters
    /// loop and the next claimant becomes a fresh leader.
    pub fn read_shard_cached_arc(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        loop {
            // Cache hit fast path.
            if let Some(ref cache_mutex) = self.cache {
                let mut cache = cache_mutex.lock().unwrap();
                if let Some(cached) = cache.get(&shard_idx) {
                    if let Some(m) = &self.metrics {
                        m.hits.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(cached);
                }
            }

            // Singleflight: either claim leadership or wait on a peer leader.
            // `_guard` (when present) drives the post-decode signal + remove
            // via its `Drop`, so panic in `decode_and_cache` still wakes
            // waiters and clears the in-flight slot.
            let _guard: Option<LeaderGuard> = match &self.in_flight {
                Some(in_flight_mutex) => {
                    let mut in_flight = in_flight_mutex.lock().unwrap();
                    if let Some(existing) = in_flight.get(&shard_idx) {
                        let slot = Arc::clone(existing);
                        drop(in_flight);
                        if let Some(m) = &self.metrics {
                            m.duplicate_waiters.fetch_add(1, Ordering::Relaxed);
                        }
                        // Wait for the leader to finish (success, fail, or
                        // panic — the LeaderGuard's Drop wakes us in all
                        // three cases).
                        let mut state = slot.state.lock().unwrap();
                        while !*state {
                            state = slot.cv.wait(state).unwrap();
                        }
                        drop(state);
                        // Loop back to re-check the cache. On a leader-success
                        // path we hit the fast path; on leader-error the next
                        // iteration claims a fresh slot as the new leader.
                        continue;
                    }
                    // Re-check the cache while holding `in_flight`. Closes
                    // the race where a peer leader finished between our
                    // initial cache miss and our acquisition of `in_flight`:
                    // by then the leader has populated the cache *and* its
                    // `LeaderGuard::drop` has removed the slot, so an
                    // unguarded check here would re-decode unnecessarily.
                    // The leader's `decode_and_cache` inserts before its
                    // guard drops, so any post-removal observer sees the
                    // entry once it acquires `in_flight`.
                    if let Some(ref cache_mutex) = self.cache {
                        let mut cache = cache_mutex.lock().unwrap();
                        if let Some(cached) = cache.get(&shard_idx) {
                            if let Some(m) = &self.metrics {
                                m.hits.fetch_add(1, Ordering::Relaxed);
                            }
                            return Ok(cached);
                        }
                    }
                    let slot = Arc::new(InFlightSlot::new());
                    in_flight.insert(shard_idx, Arc::clone(&slot));
                    Some(LeaderGuard {
                        in_flight: in_flight_mutex,
                        slot,
                        key: shard_idx,
                    })
                }
                None => None,
            };

            if let Some(m) = &self.metrics {
                m.misses.fetch_add(1, Ordering::Relaxed);
            }

            // Leader path. `_guard` drops here on every exit (Ok, Err, or
            // panic), signalling waiters and clearing the in-flight slot.
            return self.decode_and_cache(shard_idx);
        }
    }

    /// Decode shard `shard_idx` from the underlying reader, insert into the
    /// cache (if configured), and trigger best-effort `MADV_WILLNEED` for
    /// upcoming sequential shards. Caller is responsible for singleflight /
    /// metrics bookkeeping around this call.
    fn decode_and_cache(&self, shard_idx: usize) -> Result<Arc<ScxCsr>> {
        let (indptr, indices, data) = match &self.layer_name {
            None => self.reader.read_csr_shard(shard_idx)?,
            Some(_) => {
                let entry =
                    self.sorted_entries
                        .get(shard_idx)
                        .ok_or(ScxError::ShardIndexOutOfBounds {
                            index: shard_idx,
                            count: self.sorted_entries.len(),
                        })?;
                self.reader.read_shard_from_entry(entry)?
            }
        };
        let n_rows = indptr.len().saturating_sub(1);
        let csr = Arc::new(ScxCsr::new_unchecked(
            (n_rows, self.n_vars),
            indptr,
            indices,
            data,
        ));

        if let Some(ref cache_mutex) = self.cache {
            let mut cache = cache_mutex.lock().unwrap();
            cache.put_with_budget(shard_idx, Arc::clone(&csr));
        }

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
    fn warm_shards(&self, shard_indices: &[usize]) -> Result<()> {
        // No cache configured → nothing to warm; the per-shard reader path
        // will decode without caching.
        let (Some(cache_mutex), Some(in_flight_mutex)) = (&self.cache, &self.in_flight) else {
            return Ok(());
        };

        if shard_indices.is_empty() {
            return Ok(());
        }

        // Build the unique miss list under one lock acquisition each. We
        // snapshot membership; a peer thread can race in between, but the
        // singleflight + cache fast path inside `read_shard_cached_arc`
        // catches every late-arriving entry so correctness is preserved —
        // we only over- or under-filter for parallelism.
        //
        // Lock order: `in_flight` → `cache`, matching the miss path of
        // `read_shard_cached_arc` (which acquires `in_flight`, then
        // re-checks `cache` while still holding it). Inverting here would
        // deadlock concurrent readers: one thread in `warm_shards` holding
        // `cache` and waiting for `in_flight`, another in
        // `read_shard_cached_arc` holding `in_flight` and waiting for
        // `cache`.
        let mut seen: HashSet<usize> = HashSet::with_capacity(shard_indices.len());
        let mut misses: Vec<usize> = Vec::with_capacity(shard_indices.len());
        {
            let in_flight = in_flight_mutex.lock().unwrap();
            let cache = cache_mutex.lock().unwrap();
            for &idx in shard_indices {
                if !seen.insert(idx) {
                    continue;
                }
                if cache.contains(&idx) || in_flight.contains_key(&idx) {
                    continue;
                }
                misses.push(idx);
            }
        }

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
            return Ok(());
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

            misses
                .par_iter()
                .try_for_each(|&idx| self.read_shard_cached_arc(idx).map(|_| ()))?;
            Ok(())
        }
    }

    // --- Native shard-by-shard aggregation ---
    //
    // These methods compute statistics without materializing the full
    // concatenated CSR. Peak memory = one decoded shard at a time
    // (plus the output vector).

    /// Compute per-row sums without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sums from each shard's
    /// CSR arrays, and concatenates the results.
    pub fn row_sums(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_sums = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            all_sums.extend(csr.row_sums());
        }
        Ok(all_sums)
    }

    /// Compute per-column sums without materializing the full matrix.
    ///
    /// Iterates shards, accumulates column sums into a single `n_vars`-length vector.
    pub fn col_sums(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut sums = vec![0.0f64; self.n_vars];
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            let partial = csr.col_sums();
            for (s, p) in sums.iter_mut().zip(partial.iter()) {
                *s += p;
            }
        }
        Ok(sums)
    }

    /// Compute per-row NNZ counts without materializing the full matrix.
    pub fn row_nnz(&self) -> Result<Vec<i64>> {
        let n_shards = self.index.n_shards();
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            all_nnz.extend(csr.row_nnz());
        }
        Ok(all_nnz)
    }

    /// Compute per-row NNZ and sums in a single shard scan.
    ///
    /// Avoids the double I/O of calling `row_nnz()` + `row_sums()` separately.
    /// Used by `filter_cells` when both `min_genes` and `min_counts` are specified.
    pub fn row_nnz_and_sums(&self) -> Result<(Vec<i64>, Vec<f64>)> {
        let n_shards = self.index.n_shards();
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        let mut all_sums = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            for row in 0..csr.n_rows() {
                let start = csr.indptr[row] as usize;
                let end = csr.indptr[row + 1] as usize;
                all_nnz.push((end - start) as i64);
                all_sums.push(csr.data[start..end].iter().map(|&v| v as f64).sum());
            }
        }
        Ok((all_nnz, all_sums))
    }

    /// Compute per-column NNZ counts without materializing the full matrix.
    pub fn col_nnz(&self) -> Result<Vec<i64>> {
        let n_shards = self.index.n_shards();
        let mut counts = vec![0i64; self.n_vars];
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            let partial = csr.col_nnz();
            for (c, p) in counts.iter_mut().zip(partial.iter()) {
                *c += p;
            }
        }
        Ok(counts)
    }

    /// Total NNZ across all shards.
    ///
    /// Reads shard statistics directly from the catalog — no shard decode
    /// required. Returns the sum of `stats.nnz` across every shard the
    /// reader covers (X shards when `layer_name.is_none()`, layer shards
    /// otherwise). Shards without a `stats` block contribute 0.
    pub fn total_nnz(&self) -> Result<usize> {
        let entries = if self.layer_name.is_none() {
            &self.x_sorted_entries
        } else {
            &self.sorted_entries
        };
        let total: u64 = entries
            .iter()
            .filter_map(|e| e.stats.as_ref().map(|s| s.nnz))
            .sum();
        Ok(total as usize)
    }

    /// Compute per-row sum of squared values without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sum-of-squares from each shard's
    /// CSR arrays, and concatenates the results. Used for scalar variance:
    /// `Var(X) = E[X²] - (E[X])²`.
    pub fn row_sum_of_squares(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_sq = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            all_sq.extend(csr.row_sum_of_squares());
        }
        Ok(all_sq)
    }

    // --- Variance ---

    /// Streaming per-row variance without materializing the full matrix.
    ///
    /// Each shard independently computes row variances (one row = one shard's row).
    pub fn row_var(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_var = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            all_var.extend(csr.row_var());
        }
        Ok(all_var)
    }

    /// Streaming per-column variance (population variance, ddof=0).
    ///
    /// Two-pass algorithm:
    ///   1. Compute column means via `col_sums() / n_obs`
    ///   2. Stream shards, accumulating `(x - mean)²` for stored values
    ///   3. Add zero-entry contributions: `(n_obs - col_nnz) * mean²`
    pub fn col_var(&self) -> Result<Vec<f64>> {
        let n_obs = self.n_obs;
        if n_obs == 0 {
            return Ok(vec![0.0f64; self.n_vars]);
        }

        // Pass 1: column means
        let col_sums = self.col_sums()?;
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_obs as f64).collect();

        // Pass 2: accumulate (val - mean)² for stored entries
        let n_shards = self.index.n_shards();
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            let partial = csr.col_var_partial(&col_means);
            for (s, p) in sq_devs.iter_mut().zip(partial.iter()) {
                *s += p;
            }
            let nnz = csr.col_nnz();
            for (c, &n) in col_nnz.iter_mut().zip(nnz.iter()) {
                *c += n as usize;
            }
        }

        // Add contribution from implicit zeros: (n_obs - col_nnz[c]) * mean[c]²
        let mut variances = vec![0.0f64; self.n_vars];
        for c in 0..self.n_vars {
            let n_zeros = n_obs - col_nnz[c];
            let total_sq_dev = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
            variances[c] = total_sq_dev / n_obs as f64;
        }

        Ok(variances)
    }

    // --- Max / Min ---

    /// Streaming per-row max without materializing the full matrix.
    pub fn row_max(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_max = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            all_max.extend(csr.row_max());
        }
        Ok(all_max)
    }

    /// Streaming per-column max without materializing the full matrix.
    ///
    /// Merges per-shard column maxes. Accounts for implicit zeros:
    /// if any column has fewer stored entries than `n_obs`, the max is
    /// at least 0.0.
    pub fn col_max(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut maxes = vec![f64::NEG_INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            // Get per-column max within this shard (using n_rows of shard, not global n_obs)
            // We need the raw stored max, so we pass n_rows = shard.n_rows()
            // But we want the global implicit-zero correction at the end,
            // so we track NNZ ourselves and compute raw stored max.
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let c = col as usize;
                let v = val as f64;
                maxes[c] = maxes[c].max(v);
                col_nnz[c] += 1;
            }
        }

        // Apply implicit-zero correction at the global level
        for c in 0..self.n_vars {
            if col_nnz[c] < self.n_obs {
                if maxes[c] == f64::NEG_INFINITY {
                    maxes[c] = 0.0;
                } else {
                    maxes[c] = maxes[c].max(0.0);
                }
            }
        }
        Ok(maxes)
    }

    /// Streaming per-row min without materializing the full matrix.
    pub fn row_min(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut all_min = Vec::with_capacity(self.n_obs);
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            all_min.extend(csr.row_min());
        }
        Ok(all_min)
    }

    /// Streaming per-column min without materializing the full matrix.
    pub fn col_min(&self) -> Result<Vec<f64>> {
        let n_shards = self.index.n_shards();
        let mut mins = vec![f64::INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let c = col as usize;
                let v = val as f64;
                mins[c] = mins[c].min(v);
                col_nnz[c] += 1;
            }
        }

        for c in 0..self.n_vars {
            if col_nnz[c] < self.n_obs {
                if mins[c] == f64::INFINITY {
                    mins[c] = 0.0;
                } else {
                    mins[c] = mins[c].min(0.0);
                }
            }
        }
        Ok(mins)
    }

    // --- Masked column aggregation (deletion-vector aware) ---
    //
    // These variants accept a `kept_rows` set and only aggregate values from
    // rows in that set. Used when deletion vectors are present.

    /// Column sums considering only the kept rows.
    ///
    /// Iterates shards, intersects with the kept set, and accumulates.
    pub fn col_sums_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Sum)
    }

    /// Column NNZ considering only the kept rows.
    pub fn col_nnz_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Nnz)
    }

    /// Column max considering only the kept rows.
    pub fn col_max_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Max)
    }

    /// Column min considering only the kept rows.
    pub fn col_min_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Min)
    }

    /// Column variance considering only the kept rows.
    pub fn col_var_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        let n_kept = kept_rows.len();
        if n_kept == 0 {
            return Ok(vec![0.0f64; self.n_vars]);
        }

        // Pass 1: column sums over kept rows → means
        let col_sums = self.col_sums_masked(kept_rows)?;
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_kept as f64).collect();

        // Pass 2: accumulate (val - mean)² for stored entries in kept rows
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        let n_shards = self.index.n_shards();
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                Some(r) => r,
                None => continue,
            };

            // Binary-search to find the sub-slice of kept_rows within [s_start, s_end).
            // kept_rows is sorted by construction (see compute_kept_to_global).
            let lo = kept_rows.partition_point(|&r| r < s_start);
            let hi = kept_rows.partition_point(|&r| r < s_end);
            for &global_row in &kept_rows[lo..hi] {
                let local_row = (global_row - s_start) as usize;
                let row_start = csr.indptr[local_row] as usize;
                let row_end = csr.indptr[local_row + 1] as usize;
                for j in row_start..row_end {
                    let c = csr.indices[j] as usize;
                    let diff = csr.data[j] as f64 - col_means[c];
                    sq_devs[c] += diff * diff;
                    col_nnz[c] += 1;
                }
            }
        }

        // Add zero-entry contributions
        let mut variances = vec![0.0f64; self.n_vars];
        for c in 0..self.n_vars {
            let n_zeros = n_kept - col_nnz[c];
            let total = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
            variances[c] = total / n_kept as f64;
        }
        Ok(variances)
    }

    /// Internal: masked column aggregation over kept rows.
    fn col_aggregate_masked(&self, kept_rows: &[u64], op: AggOp) -> Result<Vec<f64>> {
        let n_kept = kept_rows.len();
        let mut result = match op {
            AggOp::Sum | AggOp::Nnz => vec![0.0f64; self.n_vars],
            AggOp::Max => vec![f64::NEG_INFINITY; self.n_vars],
            AggOp::Min => vec![f64::INFINITY; self.n_vars],
        };
        let mut col_nnz = vec![0usize; self.n_vars];

        let n_shards = self.index.n_shards();
        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                Some(r) => r,
                None => continue,
            };

            // Binary-search to find the sub-slice of kept_rows within [s_start, s_end).
            // kept_rows is sorted by construction (see compute_kept_to_global).
            let lo = kept_rows.partition_point(|&r| r < s_start);
            let hi = kept_rows.partition_point(|&r| r < s_end);
            for &global_row in &kept_rows[lo..hi] {
                let local_row = (global_row - s_start) as usize;
                let row_start = csr.indptr[local_row] as usize;
                let row_end = csr.indptr[local_row + 1] as usize;
                for j in row_start..row_end {
                    let c = csr.indices[j] as usize;
                    let v = csr.data[j] as f64;
                    match op {
                        AggOp::Sum => result[c] += v,
                        AggOp::Nnz => result[c] += 1.0,
                        AggOp::Max => result[c] = result[c].max(v),
                        AggOp::Min => result[c] = result[c].min(v),
                    }
                    col_nnz[c] += 1;
                }
            }
        }

        // Handle implicit zeros for max/min
        match op {
            AggOp::Max => {
                for c in 0..self.n_vars {
                    if col_nnz[c] < n_kept {
                        if result[c] == f64::NEG_INFINITY {
                            result[c] = 0.0;
                        } else {
                            result[c] = result[c].max(0.0);
                        }
                    }
                }
            }
            AggOp::Min => {
                for c in 0..self.n_vars {
                    if col_nnz[c] < n_kept {
                        if result[c] == f64::INFINITY {
                            result[c] = 0.0;
                        } else {
                            result[c] = result[c].min(0.0);
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(result)
    }

    // --- PCA statistics ---

    /// Compute column means and column sum-of-squares in a single pass.
    ///
    /// Streams through all shards once, accumulating per-column sums and
    /// sum-of-squares. If `zero_center` is true, returns the column means
    /// (sums / n_obs); otherwise returns `None` for means.
    ///
    /// Used by both CPU and GPU PCA to avoid a separate data pass for
    /// variance computation.
    ///
    /// Returns `(means, col_sum_sq)` where:
    /// - `means`: `Some(Vec<f64>)` of length `n_vars` if `zero_center`, else `None`
    /// - `col_sum_sq`: `Vec<f64>` of length `n_vars` — per-column Σ x²
    pub fn col_means_and_sum_sq(&self, zero_center: bool) -> Result<(Option<Vec<f64>>, Vec<f64>)> {
        let n_vars = self.n_vars;
        let n_shards = self.index.n_shards();
        let mut col_sums = vec![0.0f64; n_vars];
        let mut col_sum_sq = vec![0.0f64; n_vars];

        for shard_idx in 0..n_shards {
            let csr = self.read_shard_uncached(shard_idx)?;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let v = val as f64;
                col_sums[col as usize] += v;
                col_sum_sq[col as usize] += v * v;
            }
        }

        let means = if zero_center {
            let n_obs = self.n_obs as f64;
            Some(col_sums.iter().map(|s| s / n_obs).collect())
        } else {
            None
        };

        Ok((means, col_sum_sq))
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

    /// O(1) override: shard row counts live in `BackedCsrIndex`, no decode needed.
    fn max_shard_rows(&self) -> Result<usize> {
        let idx = &self.index;
        Ok((0..idx.n_shards())
            .filter_map(|i| idx.shard_range(i))
            .map(|(s, e)| (e - s) as usize)
            .max()
            .unwrap_or(0))
    }

    // col_means_and_sum_sq: use the default trait impl which iterates
    // read_shard() — functionally identical to the inherent method above.
}

/// Total variance from pre-computed column sum-of-squares.
///
/// Uses the identity: `Var(X_j) = (Σ x²_j − n·μ_j²) / (n−1)`.
/// Sums variance contributions across all columns to get total variance.
///
/// - `col_sum_sq`: per-column sum-of-squares (from [`BackedCsrReader::col_means_and_sum_sq`])
/// - `means`: column means (if centering was applied)
/// - `n_obs`: number of observations
pub fn total_variance_from_col_sq(col_sum_sq: &[f64], means: Option<&[f64]>, n_obs: usize) -> f64 {
    let total = if let Some(mu) = means {
        col_sum_sq
            .iter()
            .zip(mu.iter())
            .map(|(&sq, &m)| sq - n_obs as f64 * m * m)
            .sum::<f64>()
    } else {
        col_sum_sq.iter().sum::<f64>()
    };
    total / (n_obs as f64 - 1.0).max(1.0)
}

// ---------------------------------------------------------------------------
// BackedCscIndex — column-major counterpart to BackedCsrIndex
// ---------------------------------------------------------------------------

/// Per-shard column range for a CSC sidecar.
///
/// Mirrors `ShardRange` (CSR) but the major axis is columns. The
/// on-disk fields are still `row_start` / `row_end` in `ShardStats`
/// (axis-overload — for `CscShard` entries those fields hold
/// `col_start` / `col_end`); we read them via
/// `ShardStats::major_start()` / `major_end()`.
#[derive(Debug, Clone, Copy)]
struct CscShardRange {
    col_start: u64,
    col_end: u64,
    /// Position in the sorted-and-filtered CSC shard list. Equal to the
    /// shard index used by `ScxReader::read_csc_shard`.
    sorted_shard_idx: usize,
}

/// Precomputed column-shard index for O(log n) col-range lookups.
///
/// Built once from a [`FullCatalog`] at open time. Filters
/// `SectionType::CscShard` only — `LayerCscShard` does not exist in
/// the current format version (deferred until layer-level CSC
/// support is needed).
#[derive(Debug, Clone)]
pub struct BackedCscIndex {
    /// Sorted by `col_start`.
    shard_ranges: Vec<CscShardRange>,
}

impl BackedCscIndex {
    /// Build from a [`FullCatalog`].
    ///
    /// Extracts CSC shard entries, sorts by `col_start`, and records
    /// each shard's position in sorted order (the index used by
    /// `ScxReader::read_csc_shard` after `csc_shards_sorted()`).
    pub fn from_catalog(catalog: &FullCatalog) -> Self {
        let mut shard_entries: Vec<CscShardRange> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard)
            .filter_map(|e| {
                e.stats.as_ref().map(|s| CscShardRange {
                    col_start: s.major_start(SectionType::CscShard),
                    col_end: s.major_end(SectionType::CscShard),
                    sorted_shard_idx: 0,
                })
            })
            .collect();

        shard_entries.sort_by_key(|r| r.col_start);
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }
        BackedCscIndex {
            shard_ranges: shard_entries,
        }
    }

    /// Number of CSC shards.
    pub fn n_shards(&self) -> usize {
        self.shard_ranges.len()
    }

    /// Find the CSC shard containing `col`, or `None` if `col` falls
    /// outside every shard's `[col_start, col_end)`.
    pub fn shard_for_col(&self, col: u64) -> Option<usize> {
        let pos = self.shard_ranges.partition_point(|r| r.col_start <= col);
        if pos == 0 {
            return None;
        }
        let r = &self.shard_ranges[pos - 1];
        if col >= r.col_start && col < r.col_end {
            Some(r.sorted_shard_idx)
        } else {
            None
        }
    }

    /// All shard indices whose `[col_start, col_end)` intersects
    /// `[c_lo, c_hi)`. Returned in ascending order.
    pub fn shards_for_col_range(&self, c_lo: u64, c_hi: u64) -> Vec<usize> {
        if c_lo >= c_hi || self.shard_ranges.is_empty() {
            return Vec::new();
        }
        let first = self.shard_ranges.partition_point(|r| r.col_end <= c_lo);
        let mut result = Vec::new();
        for r in &self.shard_ranges[first..] {
            if r.col_start >= c_hi {
                break;
            }
            result.push(r.sorted_shard_idx);
        }
        result
    }

    /// Get the shard column range `(col_start, col_end)`.
    pub fn shard_col_range(&self, shard_idx: usize) -> Option<(u64, u64)> {
        self.shard_ranges
            .get(shard_idx)
            .map(|r| (r.col_start, r.col_end))
    }
}

// ---------------------------------------------------------------------------
// BackedCscReader — column-major counterpart to BackedCsrReader
// ---------------------------------------------------------------------------

/// On-demand CSC reader with optional decoded-shard caching.
///
/// Wraps an [`ScxReader`] with a [`BackedCscIndex`] for O(log n)
/// column-range lookups and an optional count-only LRU cache for
/// decoded `ScxCsc` shards. Implements [`crate::ColumnShardSource`].
///
/// The cache is intentionally simpler than `BackedCsrReader`'s:
/// count-only LRU, no byte budget, no singleflight. CSC analytical
/// kernels (Phase F: DE on gene chunks, projected `col_*`) tend to
/// access shards in column-range order with limited reuse, so the CSR
/// reader's heavier machinery isn't a fit yet. If benchmarks later show
/// contention on the same shard from multiple threads, the singleflight
/// pattern can be ported over.
pub struct BackedCscReader {
    reader: ScxReader,
    index: BackedCscIndex,
    n_obs: usize,
    n_vars: usize,
    /// Sorted CSC shard catalog entries (catalog index == sorted shard
    /// index). Pre-cached at construction so we don't re-scan the
    /// catalog on every read.
    sorted_entries: Vec<FullCatalogEntry>,
    /// Optional count-only LRU cache (`None` ⇒ no caching).
    cache: Option<Mutex<CscCache>>,
    /// Optional metrics handle.
    metrics: Option<Arc<CacheMetrics>>,
}

/// Minimal count-only LRU cache for decoded CSC shards. Counterpart to
/// `WeightedLruCache` for CSR; we don't yet need a byte budget here.
struct CscCache {
    inner: LruCache<usize, Arc<ScxCsc>>,
    metrics: Option<Arc<CacheMetrics>>,
}

impl CscCache {
    fn new(cap: usize) -> Self {
        let cap = NonZeroUsize::new(cap).unwrap();
        CscCache {
            inner: LruCache::new(cap),
            metrics: None,
        }
    }

    fn get(&mut self, key: &usize) -> Option<Arc<ScxCsc>> {
        self.inner.get(key).cloned()
    }

    fn put(&mut self, key: usize, value: Arc<ScxCsc>) {
        let was_replace = self.inner.contains(&key);
        if let Some(_displaced) = self.inner.put(key, value) {
            if !was_replace {
                if let Some(m) = &self.metrics {
                    m.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        // CSC cache is count-only (no byte budget); `bytes_inserted` /
        // `peak_bytes_in_cache` on the shared `CacheMetrics` are
        // intentionally left at zero here. Mirror their CSR-side
        // semantics if a byte budget is added later.
    }
}

impl BackedCscReader {
    /// Create a new backed CSC reader from an [`ScxReader`].
    ///
    /// `cache_shards`: number of decoded CSC shards to cache (0 = no
    /// cache). The underlying file must have CSC sidecar shards
    /// (`reader.header().has_csc()`); otherwise `read_csc_shard` will
    /// always return an out-of-bounds error.
    pub fn new(reader: ScxReader, cache_shards: usize) -> Result<Self> {
        let index = BackedCscIndex::from_catalog(reader.catalog());
        let n_obs = reader.n_obs() as usize;
        let n_vars = reader.n_vars() as usize;
        let sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .csc_shards_sorted()
            .into_iter()
            .cloned()
            .collect();
        let cache = if cache_shards > 0 {
            Some(Mutex::new(CscCache::new(cache_shards)))
        } else {
            None
        };
        Ok(BackedCscReader {
            reader,
            index,
            n_obs,
            n_vars,
            sorted_entries,
            cache,
            metrics: None,
        })
    }

    /// Borrow the CSC shard index.
    pub fn index(&self) -> &BackedCscIndex {
        &self.index
    }

    /// Enable shard-cache metrics. Returns a cloneable
    /// `Arc<CacheMetrics>` so callers can sample counters without going
    /// through the cache lock. The returned handle's `hits` /
    /// `misses` / `evictions` reflect CSC-side activity; CSR metrics
    /// live on `BackedCsrReader::enable_metrics()` separately.
    pub fn enable_metrics(&mut self) -> Arc<CacheMetrics> {
        let m = Arc::new(CacheMetrics::default());
        self.metrics = Some(Arc::clone(&m));
        if let Some(ref cache_mutex) = self.cache {
            let mut c = cache_mutex.lock().unwrap();
            c.metrics = Some(Arc::clone(&m));
        }
        m
    }

    /// Borrow the metrics handle, if enabled.
    pub fn metrics(&self) -> Option<&Arc<CacheMetrics>> {
        self.metrics.as_ref()
    }

    /// Number of CSC shards.
    pub fn n_shards(&self) -> usize {
        self.index.n_shards()
    }

    /// Total number of observations (rows). CSC shards span the full
    /// row axis, so this equals the file's `n_obs`.
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Number of variables (columns).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Read and decode a CSC shard without consulting the cache.
    /// Useful for one-shot streaming passes where caching would only
    /// add overhead.
    pub fn read_shard_uncached(&self, shard_idx: usize) -> Result<ScxCsc> {
        let entry = self
            .sorted_entries
            .get(shard_idx)
            .ok_or(ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.sorted_entries.len(),
            })?;
        let (indptr, indices, data) = self.reader.read_shard_from_entry(entry)?;
        let n_cols_in_shard = indptr.len().saturating_sub(1);
        Ok(ScxCsc::new_unchecked(
            (self.n_obs, n_cols_in_shard),
            indptr,
            indices,
            data,
        ))
    }

    /// Read and optionally cache a CSC shard, returning a shared
    /// `Arc<ScxCsc>`. On cache hit increments `metrics.hits`; on miss
    /// (or no cache) increments `metrics.misses` and decodes.
    pub fn read_shard_cached(&self, shard_idx: usize) -> Result<Arc<ScxCsc>> {
        if let Some(ref cache_mutex) = self.cache {
            let mut cache = cache_mutex.lock().unwrap();
            if let Some(cached) = cache.get(&shard_idx) {
                if let Some(m) = &self.metrics {
                    m.hits.fetch_add(1, Ordering::Relaxed);
                }
                return Ok(cached);
            }
        }
        if let Some(m) = &self.metrics {
            m.misses.fetch_add(1, Ordering::Relaxed);
        }
        let csc = Arc::new(self.read_shard_uncached(shard_idx)?);
        if let Some(ref cache_mutex) = self.cache {
            let mut cache = cache_mutex.lock().unwrap();
            cache.put(shard_idx, Arc::clone(&csc));
        }
        Ok(csc)
    }

    /// Read a contiguous column slice across CSC shards.
    ///
    /// Skips shards whose `[col_start, col_end)` does not intersect
    /// `col_range`; `col_slice`s partial-overlap shards post-decode.
    /// Cached: each overlapping shard is fetched via
    /// [`Self::read_shard_cached`].
    pub fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> Result<ScxCsc> {
        let c_lo = col_range.start as u64;
        let c_hi = col_range.end as u64;
        if c_lo >= c_hi {
            return Ok(ScxCsc::new_unchecked(
                (self.n_obs, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }
        let shard_indices = self.index.shards_for_col_range(c_lo, c_hi);

        let mut decoded: Vec<ScxCsc> = Vec::with_capacity(shard_indices.len());
        for shard_idx in shard_indices {
            let csc = self.read_shard_cached(shard_idx)?;
            let (shard_lo, shard_hi) = self.index.shard_col_range(shard_idx).ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "BackedCscIndex missing range for shard {shard_idx}"
                ))
            })?;
            let lo_in_shard = c_lo.saturating_sub(shard_lo) as usize;
            let hi_in_shard = (c_hi.min(shard_hi).saturating_sub(shard_lo)) as usize;
            let sliced = if lo_in_shard == 0 && hi_in_shard == csc.n_cols() {
                (*csc).clone()
            } else {
                csc.col_slice(lo_in_shard, hi_in_shard).map_err(|e| {
                    ScxError::InvalidCatalog(format!(
                        "BackedCscReader col_slice failed for shard {shard_idx}: {e}"
                    ))
                })?
            };
            decoded.push(sliced);
        }
        concatenate_csc_along_cols(decoded, self.n_obs)
    }

    /// Read an arbitrary sorted column subset by collapsing it to
    /// contiguous runs and concatenating per-run `read_csc_columns`
    /// results. Mirrors `ScxReader::read_csc_columns_subset` but uses
    /// the cached shard reads.
    pub fn read_csc_columns_subset(&self, cols: &[u32]) -> Result<ScxCsc> {
        if cols.is_empty() {
            return Ok(ScxCsc::new_unchecked(
                (self.n_obs, 0),
                vec![0],
                Vec::new(),
                Vec::new(),
            ));
        }
        let mut runs: Vec<ScxCsc> = Vec::new();
        let mut run_start = cols[0];
        let mut run_end = cols[0] + 1;
        for &c in &cols[1..] {
            if c == run_end {
                run_end = c + 1;
            } else {
                runs.push(self.read_csc_columns(run_start..run_end)?);
                run_start = c;
                run_end = c + 1;
            }
        }
        runs.push(self.read_csc_columns(run_start..run_end)?);
        if runs.len() == 1 {
            return Ok(runs.pop().unwrap());
        }
        concatenate_csc_along_cols(runs, self.n_obs)
    }
}

/// Concatenate a list of CSC parts along the column axis.
/// Internal helper — the same logic also lives in `reader.rs` as a
/// private free function. Local copy avoids cross-module visibility
/// changes.
fn concatenate_csc_along_cols(parts: Vec<ScxCsc>, n_rows: usize) -> Result<ScxCsc> {
    if parts.is_empty() {
        return Ok(ScxCsc::new_unchecked(
            (n_rows, 0),
            vec![0],
            Vec::new(),
            Vec::new(),
        ));
    }
    let total_cols: usize = parts.iter().map(|p| p.n_cols()).sum();
    let total_nnz: usize = parts.iter().map(|p| p.nnz()).sum();
    let mut indptr = Vec::with_capacity(total_cols + 1);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);
    indptr.push(0i64);
    let mut cum_nnz: i64 = 0;
    for part in parts {
        let part_n_cols = part.n_cols();
        for i in 1..=part_n_cols {
            indptr.push(part.indptr[i] + cum_nnz);
        }
        cum_nnz += part.indptr[part_n_cols];
        indices.extend_from_slice(&part.indices);
        data.extend_from_slice(&part.data);
    }
    Ok(ScxCsc::new_unchecked(
        (n_rows, total_cols),
        indptr,
        indices,
        data,
    ))
}

impl crate::shard_source::ColumnShardSource for BackedCscReader {
    fn n_csc_shards(&self) -> usize {
        BackedCscReader::n_shards(self)
    }

    fn n_obs(&self) -> usize {
        BackedCscReader::n_obs(self)
    }

    fn n_vars(&self) -> usize {
        BackedCscReader::n_vars(self)
    }

    fn read_csc_shard(&self, shard_idx: usize) -> Result<ScxCsc> {
        // Owned clone — trait return type is `ScxCsc`, not `Arc<_>`.
        Ok((*self.read_shard_cached(shard_idx)?).clone())
    }

    fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> Result<ScxCsc> {
        BackedCscReader::read_csc_columns(self, col_range)
    }

    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
        let (lo, hi) = self.index.shard_col_range(shard_idx)?;
        let lo = u32::try_from(lo).ok()?;
        let hi = u32::try_from(hi).ok()?;
        Some((lo, hi))
    }
}

// ---------------------------------------------------------------------------
// CSR concatenation helper
// ---------------------------------------------------------------------------

/// Merge multiple `ScxCsr` values into one, rebasing indptr.
///
/// All CSRs must have the same number of columns (`n_vars`).
pub fn concatenate_csr(csrs: &[ScxCsr], n_vars: usize) -> Result<ScxCsr> {
    if csrs.is_empty() {
        return Ok(ScxCsr::new_unchecked((0, n_vars), vec![0], vec![], vec![]));
    }

    if csrs.len() == 1 {
        return Ok(csrs[0].clone());
    }

    // Pre-compute total sizes for allocation
    let total_rows: usize = csrs.iter().map(|c| c.n_rows()).sum();
    let total_nnz: usize = csrs.iter().map(|c| c.nnz()).sum();

    let mut merged_indptr = Vec::with_capacity(total_rows + 1);
    let mut merged_indices = Vec::with_capacity(total_nnz);
    let mut merged_data = Vec::with_capacity(total_nnz);
    let mut cumulative_nnz: i64 = 0;

    for (i, csr) in csrs.iter().enumerate() {
        if i == 0 {
            merged_indptr.extend_from_slice(&csr.indptr);
        } else {
            // Skip first element (0) and offset by cumulative nnz
            for &v in &csr.indptr[1..] {
                merged_indptr.push(v + cumulative_nnz);
            }
        }
        cumulative_nnz += *csr.indptr.last().unwrap_or(&0);
        merged_indices.extend_from_slice(&csr.indices);
        merged_data.extend_from_slice(&csr.data);
    }

    Ok(ScxCsr::new_unchecked(
        (total_rows, n_vars),
        merged_indptr,
        merged_indices,
        merged_data,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::{FileHeader, MAGIC};
    use crate::writer::ScxWriter;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field, Schema};
    use scx_codec::{CodecId, ValueEncoding};
    use std::sync::Arc;
    use tempfile::TempDir;

    // --- Test helpers (same as reader.rs test helpers) ---

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: crate::header::CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
        }
    }

    fn sample_obs(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> arrow::array::RecordBatch {
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..n_rows {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        (indptr, indices, values)
    }

    /// Write a test file with the specified number of shards and return
    /// a `BackedCsrReader` along with the reference full CSR.
    fn write_test_file_and_open(
        dir: &TempDir,
        n_obs: usize,
        n_vars: usize,
        n_shards: usize,
        cache_shards: usize,
    ) -> (BackedCsrReader, ScxCsr) {
        let path = dir.path().join("test.scx");
        let total_nnz = n_obs * 2;
        let header = sample_header(n_obs as u64, n_vars as u64, total_nnz as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();

        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        let rows_per_shard = n_obs / n_shards;
        for s in 0..n_shards {
            let shard_rows = if s == n_shards - 1 {
                n_obs - rows_per_shard * s
            } else {
                rows_per_shard
            };
            let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
            writer
                .write_csr_shard(
                    &indptr,
                    &indices,
                    &values,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    (s * rows_per_shard) as u64,
                )
                .unwrap();
        }

        writer.finish().unwrap();

        let reader = ScxReader::open(&path).unwrap();
        let full_csr = {
            let r2 = ScxReader::open(&path).unwrap();
            r2.read_all_csr_shards().unwrap()
        };
        let backed = BackedCsrReader::new(reader, cache_shards);
        (backed, full_csr)
    }

    // -----------------------------------------------------------------------
    // BackedCsrIndex tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_index_from_catalog() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        assert_eq!(backed.index().n_shards(), 4);
    }

    #[test]
    fn test_shards_for_range_single_shard() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // 4 shards of 3 rows each: [0,3), [3,6), [6,9), [9,12)
        let shards = backed.index().shards_for_range(0, 3);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_shards_for_range_two_shards() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // Spans shard 0 [0,3) and shard 1 [3,6)
        let shards = backed.index().shards_for_range(1, 5);
        assert_eq!(shards, vec![0, 1]);
    }

    #[test]
    fn test_shards_for_range_all_shards() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_range(0, 12);
        assert_eq!(shards, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_shards_for_range_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_range(5, 5);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_shards_for_range_at_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // Exact shard boundary: [3,6) should be shard 1 only
        let shards = backed.index().shards_for_range(3, 6);
        assert_eq!(shards, vec![1]);
    }

    #[test]
    fn test_shards_for_indices_scattered() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        // Row 0 in shard 0, row 5 in shard 1, row 11 in shard 3
        let shards = backed.index().shards_for_indices(&[0, 5, 11]);
        assert_eq!(shards, vec![0, 1, 3]);
    }

    #[test]
    fn test_shards_for_indices_all_one_shard() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_indices(&[0, 1, 2]);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_shards_for_indices_duplicates() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_indices(&[0, 0, 1, 1]);
        assert_eq!(shards, vec![0]);
    }

    #[test]
    fn test_shards_for_indices_unsorted() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);
        let shards = backed.index().shards_for_indices(&[11, 0, 5]);
        assert_eq!(shards, vec![0, 1, 3]);
    }

    // -----------------------------------------------------------------------
    // BackedCsrReader::read_rows tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_rows_full_range() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_rows(0, 12).unwrap();
        assert_eq!(result.shape, full.shape);
        assert_eq!(result.indptr, full.indptr);
        assert_eq!(result.indices, full.indices);
        assert_eq!(result.data, full.data);
    }

    #[test]
    fn test_read_rows_single_row() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        for row in 0..12 {
            let backed_row = backed.read_rows(row, row + 1).unwrap();
            let full_row = full.row_slice(row as usize, row as usize + 1).unwrap();
            assert_eq!(
                backed_row.indptr, full_row.indptr,
                "indptr mismatch at row {row}"
            );
            assert_eq!(
                backed_row.indices, full_row.indices,
                "indices mismatch at row {row}"
            );
            assert_eq!(backed_row.data, full_row.data, "data mismatch at row {row}");
        }
    }

    #[test]
    fn test_read_rows_cross_shard_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // Spans shards 0 and 1 (rows [2,5))
        let result = backed.read_rows(2, 5).unwrap();
        let expected = full.row_slice(2, 5).unwrap();
        assert_eq!(result.shape, expected.shape);
        assert_eq!(result.indptr, expected.indptr);
        assert_eq!(result.indices, expected.indices);
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_read_rows_every_contiguous_range() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // Test every possible contiguous range
        for start in 0..12u64 {
            for end in (start + 1)..=12 {
                let result = backed.read_rows(start, end).unwrap();
                let expected = full.row_slice(start as usize, end as usize).unwrap();
                assert_eq!(
                    result.indptr, expected.indptr,
                    "indptr mismatch at [{start}, {end})"
                );
                assert_eq!(
                    result.indices, expected.indices,
                    "indices mismatch at [{start}, {end})"
                );
                assert_eq!(
                    result.data, expected.data,
                    "data mismatch at [{start}, {end})"
                );
            }
        }
    }

    #[test]
    fn test_read_rows_empty_range() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_rows(5, 5).unwrap();
        assert_eq!(result.n_rows(), 0);
        assert_eq!(result.nnz(), 0);
    }

    // -----------------------------------------------------------------------
    // BackedCsrReader::read_row_indices tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_read_row_indices_scattered() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let indices = [0u64, 5, 11];
        let result = backed.read_row_indices(&indices).unwrap();
        assert_eq!(result.n_rows(), 3);

        // Each row should match the corresponding row from the full CSR
        for (i, &row) in indices.iter().enumerate() {
            let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
            let actual = result.row_slice(i, i + 1).unwrap();
            assert_eq!(actual.indices, expected.indices, "row {row} indices");
            assert_eq!(actual.data, expected.data, "row {row} data");
        }
    }

    #[test]
    fn test_read_row_indices_empty() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_row_indices(&[]).unwrap();
        assert_eq!(result.n_rows(), 0);
    }

    // -----------------------------------------------------------------------
    // BackedCsrReader::read_rows_with tests
    // -----------------------------------------------------------------------

    /// Helper: gather rows via `read_rows_with` into per-request `(indices,
    /// data)` clones in caller order.
    fn gather_with(backed: &BackedCsrReader, rows: &[u64]) -> Vec<(Vec<i32>, Vec<f32>)> {
        let mut out: Vec<(Vec<i32>, Vec<f32>)> = vec![Default::default(); rows.len()];
        backed
            .read_rows_with(rows, |i, idx, data| {
                out[i] = (idx.to_vec(), data.to_vec());
                Ok(())
            })
            .unwrap();
        out
    }

    #[test]
    fn test_read_rows_with_matches_read_row_indices_scattered() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let indices = [0u64, 5, 11];
        let dense = gather_with(&backed, &indices);

        for (i, &row) in indices.iter().enumerate() {
            let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
            assert_eq!(dense[i].0, expected.indices, "row {row} indices");
            assert_eq!(dense[i].1, expected.data, "row {row} data");
        }
    }

    #[test]
    fn test_read_rows_with_empty_no_scatter_calls() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let mut count = 0;
        backed
            .read_rows_with(&[], |_, _, _| {
                count += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_read_rows_with_duplicates_call_scatter_per_occurrence() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // Same row twice; scatter must fire twice with identical content.
        let indices = [3u64, 3];
        let dense = gather_with(&backed, &indices);

        let expected = full.row_slice(3, 4).unwrap();
        assert_eq!(dense[0].0, expected.indices);
        assert_eq!(dense[0].1, expected.data);
        assert_eq!(dense[1].0, expected.indices);
        assert_eq!(dense[1].1, expected.data);
    }

    #[test]
    fn test_read_rows_with_unsorted_caller_order() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // Mixed shard order — the public scatter callback receives `i`
        // matching the input position, regardless of internal sort.
        let indices = [11u64, 0, 7, 4, 3];
        let dense = gather_with(&backed, &indices);

        for (i, &row) in indices.iter().enumerate() {
            let expected = full.row_slice(row as usize, row as usize + 1).unwrap();
            assert_eq!(dense[i].0, expected.indices, "i={i} row={row} indices");
            assert_eq!(dense[i].1, expected.data, "i={i} row={row} data");
        }
    }

    #[test]
    fn test_read_rows_with_out_of_range_errors() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        let result = backed.read_rows_with(&[5, 99], |_, _, _| Ok(()));
        assert!(result.is_err(), "OOR row should surface as error");
    }

    // -----------------------------------------------------------------------
    // LRU cache behavior tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cache_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 0);

        // Should still work without cache
        let result = backed.read_rows(0, 12).unwrap();
        assert_eq!(result.indptr, full.indptr);
    }

    #[test]
    fn test_cache_hit() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);

        // First read (cold)
        let r1 = backed.read_rows(0, 3).unwrap();
        // Second read (cached)
        let r2 = backed.read_rows(0, 3).unwrap();
        assert_eq!(r1.indptr, r2.indptr);
        assert_eq!(r1.indices, r2.indices);
        assert_eq!(r1.data, r2.data);
    }

    #[test]
    fn test_cache_eviction() {
        let dir = tempfile::tempdir().unwrap();
        // Cache can hold 2 shards, file has 4
        let (backed, full) = write_test_file_and_open(&dir, 12, 10, 4, 2);

        // Access shards 0, 1, 2 — shard 0 should be evicted
        let _ = backed.read_rows(0, 3).unwrap(); // shard 0
        let _ = backed.read_rows(3, 6).unwrap(); // shard 1
        let _ = backed.read_rows(6, 9).unwrap(); // shard 2 (evicts shard 0)

        // Re-read shard 0 — should still produce correct results
        let result = backed.read_rows(0, 3).unwrap();
        let expected = full.row_slice(0, 3).unwrap();
        assert_eq!(result.indptr, expected.indptr);
        assert_eq!(result.indices, expected.indices);
        assert_eq!(result.data, expected.data);
    }

    // -----------------------------------------------------------------------
    // warm_shards / parallel cold-shard decode tests
    // -----------------------------------------------------------------------

    /// Build a reader on the same on-disk file as `seq_backed` but with
    /// `cache_shards = 1` (forces the sequential body inside `warm_shards`
    /// even when the `parallel` feature is on). Used to verify that the
    /// parallel decode path produces byte-identical output to the
    /// sequential one.
    fn open_with_cache_shards(dir: &TempDir, cache_shards: usize) -> BackedCsrReader {
        let path = dir.path().join("test.scx");
        let reader = ScxReader::open(&path).unwrap();
        BackedCsrReader::new(reader, cache_shards)
    }

    #[test]
    fn test_warm_shards_parity_parallel_vs_sequential() {
        // 8 shards, scattered indices that touch ≥ 3 of them. Compare
        // parallel-decode output against a sequential control.
        let dir = tempfile::tempdir().unwrap();
        let (par, _full) = write_test_file_and_open(&dir, 32, 10, 8, 8);
        let seq = open_with_cache_shards(&dir, 1);

        let indices: [u64; 7] = [0, 5, 11, 18, 24, 27, 31];
        let par_out = par.read_row_indices(&indices).unwrap();
        let seq_out = seq.read_row_indices(&indices).unwrap();

        assert_eq!(par_out.indptr, seq_out.indptr);
        assert_eq!(par_out.indices, seq_out.indices);
        assert_eq!(par_out.data, seq_out.data);

        // Same comparison for read_rows (range) and read_rows_with (gather).
        let par_range = par.read_rows(2, 30).unwrap();
        let seq_range = seq.read_rows(2, 30).unwrap();
        assert_eq!(par_range.indptr, seq_range.indptr);
        assert_eq!(par_range.indices, seq_range.indices);
        assert_eq!(par_range.data, seq_range.data);

        let par_gather = gather_with(&par, &indices);
        let seq_gather = gather_with(&seq, &indices);
        assert_eq!(par_gather, seq_gather);
    }

    #[test]
    fn test_warm_shards_no_misses_is_noop() {
        // Pre-warm every touched shard, then call read_row_indices on the
        // same set. The metrics `misses` counter should stay at the
        // pre-warm value because warm_shards filters cache hits up front.
        let dir = tempfile::tempdir().unwrap();
        let (mut backed, _) = write_test_file_and_open(&dir, 32, 10, 8, 8);
        let metrics = backed.enable_metrics();

        // Pre-warm shards 0, 1, 3 directly.
        for s in [0usize, 1, 3] {
            backed.read_shard_cached_arc(s).unwrap();
        }
        let baseline_misses = metrics.misses.load(Ordering::Relaxed);
        assert_eq!(baseline_misses, 3, "pre-warm should miss exactly 3 times");

        // Indices 0, 5, 13 land on shards 0, 1, 3 respectively.
        let _ = backed.read_row_indices(&[0u64, 5, 13]).unwrap();

        let after_misses = metrics.misses.load(Ordering::Relaxed);
        assert_eq!(
            after_misses, baseline_misses,
            "fully cached read should not register any new miss",
        );
    }

    #[test]
    fn test_warm_shards_concurrent_dedup_via_singleflight() {
        // N threads each call read_row_indices for the same set of cold
        // shards. The singleflight table must dedupe so total misses ==
        // n_unique_shards regardless of N.
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let (mut backed, _) = write_test_file_and_open(&dir, 64, 10, 8, 8);
        let metrics = backed.enable_metrics();
        let backed = Arc::new(backed);

        // Indices touching 4 unique shards: 0, 1, 5, 7.
        let indices: Vec<u64> = vec![0, 12, 41, 60];

        let n_threads = 8;
        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let b = Arc::clone(&backed);
            let ix = indices.clone();
            handles.push(thread::spawn(move || {
                b.read_row_indices(&ix).unwrap();
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let n_unique_shards: u64 = 4;
        let misses = metrics.misses.load(Ordering::Relaxed);
        assert!(
            misses <= n_unique_shards,
            "expected ≤ {n_unique_shards} misses across {n_threads} concurrent readers, \
             got {misses} — singleflight broke under parallel warm_shards",
        );
        // And at least one — every shard had to be decoded once.
        assert!(misses >= 1, "expected ≥ 1 miss");

        // Singleflight slots must be empty after all threads finish.
        for s in [0usize, 1, 5, 7] {
            assert!(
                !backed.in_flight_contains(s),
                "in_flight slot for shard {s} should be cleared after join",
            );
        }
    }

    #[test]
    fn test_warm_shards_propagates_oor_error() {
        // Calling warm_shards directly with an out-of-range shard_idx
        // should surface a ShardIndexOutOfBounds error from the underlying
        // read path, and must not strand the singleflight table.
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 32, 10, 4, 4);

        // Shard 99 doesn't exist — file has 4 shards.
        let result = backed.warm_shards(&[0usize, 1, 99]);
        assert!(result.is_err(), "OOR shard idx must error");

        // Singleflight cleanup: every leader's LeaderGuard::Drop must have
        // removed its slot regardless of error.
        for s in 0..4usize {
            assert!(
                !backed.in_flight_contains(s),
                "in_flight slot for shard {s} should be cleared after error",
            );
        }
        assert!(!backed.in_flight_contains(99));
    }

    #[test]
    fn test_warm_shards_no_cache_is_noop() {
        // cache_shards = 0 disables the cache entirely — warm_shards must
        // be a no-op (the per-shard read path will decode without caching).
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 0);

        backed.warm_shards(&[0usize, 1, 2, 3]).unwrap();
        // And the multi-shard reader path still works.
        let result = backed.read_rows(0, 12).unwrap();
        assert_eq!(result.shape.0, 12);
    }

    #[test]
    fn test_warm_shards_no_deadlock_with_concurrent_direct_decoders() {
        // Regression for the AB/BA lock-order inversion between
        // `warm_shards` and `read_shard_cached_arc`'s miss path. We spawn
        // two pools against the same `BackedCsrReader`:
        //   - "warmers": call `read_rows`, which funnels through `warm_shards`
        //   - "direct":  call `read_shard_cached_arc` directly
        // Both target overlapping cold shards, so the two lock-acquisition
        // paths run concurrently. Pre-fix this deadlocks probabilistically;
        // post-fix every thread joins. If the deadlock returns, the test
        // hangs and CI's job timeout fails the run.
        //
        // `cache_shards = 8` < n_shards = 16 also exercises the P2 truncation
        // branch (warm caps at cache_shards instead of decoding the full
        // miss list and thrashing the LRU).
        use std::sync::Arc;
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 128, 10, 16, 8);
        let backed = Arc::new(backed);

        let n_iters = 64;
        let n_warmers = 4;
        let n_direct = 4;
        let target_shards: Vec<usize> = (0..16).collect();

        let mut handles = Vec::with_capacity(n_warmers + n_direct);
        for _ in 0..n_warmers {
            let b = Arc::clone(&backed);
            handles.push(thread::spawn(move || {
                for _ in 0..n_iters {
                    let _ = b.read_rows(0, 128).unwrap();
                }
            }));
        }
        for _ in 0..n_direct {
            let b = Arc::clone(&backed);
            let shards = target_shards.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..n_iters {
                    for &s in &shards {
                        let _ = b.read_shard_cached_arc(s).unwrap();
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    }

    // -----------------------------------------------------------------------
    // CSR concatenation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_concatenate_empty() {
        let result = concatenate_csr(&[], 5).unwrap();
        assert_eq!(result.n_rows(), 0);
        assert_eq!(result.nnz(), 0);
        assert_eq!(result.shape.1, 5);
    }

    #[test]
    fn test_concatenate_single() {
        let csr = ScxCsr::new_unchecked((2, 5), vec![0, 2, 3], vec![1, 3, 2], vec![5.0, 10.0, 2.0]);
        let result = concatenate_csr(&[csr.clone()], 5).unwrap();
        assert_eq!(result.indptr, csr.indptr);
        assert_eq!(result.indices, csr.indices);
        assert_eq!(result.data, csr.data);
    }

    #[test]
    fn test_concatenate_two() {
        let csr1 = ScxCsr::new_unchecked((1, 5), vec![0, 2], vec![1, 3], vec![5.0, 10.0]);

        let csr2 = ScxCsr::new_unchecked((1, 5), vec![0, 1], vec![2], vec![7.0]);

        let result = concatenate_csr(&[csr1, csr2], 5).unwrap();
        assert_eq!(result.shape, (2, 5));
        assert_eq!(result.indptr, vec![0, 2, 3]);
        assert_eq!(result.indices, vec![1, 3, 2]);
        assert_eq!(result.data, vec![5.0, 10.0, 7.0]);
    }

    #[test]
    fn test_concatenate_empty_plus_nonempty() {
        let empty = ScxCsr::new_unchecked((0, 5), vec![0], vec![], vec![]);
        let nonempty = ScxCsr::new_unchecked((1, 5), vec![0, 2], vec![1, 3], vec![5.0, 10.0]);

        let result = concatenate_csr(&[empty, nonempty], 5).unwrap();
        assert_eq!(result.shape, (1, 5));
        assert_eq!(result.indptr, vec![0, 2]);
        assert_eq!(result.indices, vec![1, 3]);
        assert_eq!(result.data, vec![5.0, 10.0]);
    }

    #[test]
    fn test_shape_accessors() {
        let dir = tempfile::tempdir().unwrap();
        let (backed, _) = write_test_file_and_open(&dir, 12, 10, 4, 4);
        assert_eq!(backed.shape(), (12, 10));
        assert_eq!(backed.n_obs(), 12);
        assert_eq!(backed.n_vars(), 10);
    }

    // -----------------------------------------------------------------------
    // BackedCscReader tests
    // -----------------------------------------------------------------------

    /// Build a CSC arrays for a column range from a row-major dense
    /// reference. Returns `(indptr_u64, indices_u32, values_u8)`.
    fn csc_arrays_for_range(
        dense: &[u8],
        n_rows: usize,
        n_cols: usize,
        col_start: usize,
        col_end: usize,
    ) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr: Vec<u64> = vec![0];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for col in col_start..col_end {
            for row in 0..n_rows {
                let v = dense[row * n_cols + col];
                if v != 0 {
                    indices.push(row as u32);
                    values.push(v);
                }
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, values)
    }

    /// Write a CSR + 4-shard-CSC test file with `cols_per_csc_shard`
    /// columns per CSC shard. Returns the path and the dense reference
    /// matrix (row-major u8).
    fn write_csc_test_file(
        dir: &TempDir,
        n_obs: usize,
        n_vars: usize,
        cols_per_csc_shard: usize,
    ) -> (std::path::PathBuf, Vec<u8>) {
        let path = dir.path().join("with_csc.scx");
        let header = sample_header(n_obs as u64, n_vars as u64, 0);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();

        // Build a deterministic row-major dense matrix; pick a nnz
        // pattern that distributes values across all columns.
        let mut dense = vec![0u8; n_obs * n_vars];
        for r in 0..n_obs {
            for c in 0..n_vars {
                if (r + c) % 3 == 0 {
                    dense[r * n_vars + c] = ((r * 7 + c * 11) % 200 + 1) as u8;
                }
            }
        }

        // CSR shard built from the dense matrix.
        let mut indptr_csr = vec![0u64];
        let mut indices_csr = Vec::new();
        let mut values_csr = Vec::new();
        for r in 0..n_obs {
            for c in 0..n_vars {
                let v = dense[r * n_vars + c];
                if v != 0 {
                    indices_csr.push(c as u32);
                    values_csr.push(v);
                }
            }
            indptr_csr.push(indices_csr.len() as u64);
        }
        writer
            .write_csr_shard(
                &indptr_csr,
                &indices_csr,
                &values_csr,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // CSC sidecar split into shards by column.
        let mut col_start = 0usize;
        while col_start < n_vars {
            let col_end = (col_start + cols_per_csc_shard).min(n_vars);
            let (ip, ix, vb) = csc_arrays_for_range(&dense, n_obs, n_vars, col_start, col_end);
            writer
                .write_csc_shard(
                    &ip,
                    &ix,
                    &vb,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    col_start as u64,
                )
                .unwrap();
            col_start = col_end;
        }
        writer.finish().unwrap();
        (path, dense)
    }

    #[test]
    fn backed_csc_index_basic() {
        // 12 rows × 10 cols, 3 cols per CSC shard → 4 shards: [0,3), [3,6), [6,9), [9,10).
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_csc_test_file(&dir, 12, 10, 3);
        let reader = ScxReader::open(&path).unwrap();
        let csc = BackedCscReader::new(reader, 0).unwrap();

        assert_eq!(csc.n_shards(), 4);
        assert_eq!(csc.n_obs(), 12);
        assert_eq!(csc.n_vars(), 10);

        // Per-shard ranges via the index.
        let idx = csc.index();
        assert_eq!(idx.shard_col_range(0), Some((0, 3)));
        assert_eq!(idx.shard_col_range(1), Some((3, 6)));
        assert_eq!(idx.shard_col_range(2), Some((6, 9)));
        assert_eq!(idx.shard_col_range(3), Some((9, 10)));
        assert_eq!(idx.shard_col_range(99), None);

        // shards_for_col_range: cover shards 1+2 only.
        assert_eq!(idx.shards_for_col_range(4, 8), vec![1, 2]);
        assert_eq!(idx.shards_for_col_range(0, 10), vec![0, 1, 2, 3]);
        // Half-open boundary exclusion.
        assert_eq!(idx.shards_for_col_range(0, 3), vec![0]);
        // Past the end / empty / inverted.
        assert!(idx.shards_for_col_range(100, 200).is_empty());
        assert!(idx.shards_for_col_range(5, 5).is_empty());

        // shard_for_col single lookups.
        assert_eq!(idx.shard_for_col(0), Some(0));
        assert_eq!(idx.shard_for_col(2), Some(0));
        assert_eq!(idx.shard_for_col(3), Some(1));
        assert_eq!(idx.shard_for_col(9), Some(3));
        assert_eq!(idx.shard_for_col(10), None);
    }

    #[test]
    fn backed_csc_read_csc_columns_correctness() {
        // 8 rows × 12 cols, 4 cols per CSC shard → 3 shards.
        let dir = tempfile::tempdir().unwrap();
        let (path, dense) = write_csc_test_file(&dir, 8, 12, 4);
        let reader = ScxReader::open(&path).unwrap();
        let csc = BackedCscReader::new(reader, 4).unwrap();

        // Helper: dense slice as f32 for column range [c_lo, c_hi).
        let dense_slice = |c_lo: usize, c_hi: usize| -> Vec<f32> {
            let cols = c_hi - c_lo;
            let mut out = vec![0f32; 8 * cols];
            for r in 0..8 {
                for (oc, sc) in (c_lo..c_hi).enumerate() {
                    out[r * cols + oc] = dense[r * 12 + sc] as f32;
                }
            }
            out
        };

        for &(c_lo, c_hi) in &[(0u32, 12u32), (1, 5), (5, 11), (4, 8), (0, 0)] {
            let got = csc.read_csc_columns(c_lo..c_hi).unwrap();
            assert_eq!(got.shape, (8, (c_hi - c_lo) as usize));
            assert_eq!(
                got.to_dense().unwrap(),
                dense_slice(c_lo as usize, c_hi as usize),
                "mismatch on cols [{c_lo}..{c_hi})"
            );
        }
    }

    #[test]
    fn backed_csc_read_csc_columns_skip_count_metric() {
        // 8 rows × 12 cols, 4 cols per CSC shard → 3 shards.
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_csc_test_file(&dir, 8, 12, 4);
        let reader = ScxReader::open(&path).unwrap();
        let mut csc = BackedCscReader::new(reader, 4).unwrap();
        let metrics = csc.enable_metrics();

        // Query that overlaps shards 0 and 1 (cols [2..6) crosses the
        // 0..4 / 4..8 boundary). Shard 2 is skipped.
        let _ = csc.read_csc_columns(2..6).unwrap();
        let misses_after = metrics.misses.load(Ordering::Relaxed);
        assert_eq!(misses_after, 2, "exactly 2 shard decodes expected");
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 0);

        // Reissue same range — both shards now in cache → 0 new misses.
        let _ = csc.read_csc_columns(2..6).unwrap();
        assert_eq!(
            metrics.misses.load(Ordering::Relaxed),
            misses_after,
            "no new decodes; both shards served from cache"
        );
        assert_eq!(metrics.hits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn backed_csc_read_csc_columns_subset() {
        let dir = tempfile::tempdir().unwrap();
        let (path, dense) = write_csc_test_file(&dir, 8, 12, 4);
        let reader = ScxReader::open(&path).unwrap();
        let csc = BackedCscReader::new(reader, 4).unwrap();

        // Non-contiguous subset.
        let cols = [0u32, 1, 5, 6, 11];
        let got = csc.read_csc_columns_subset(&cols).unwrap();
        assert_eq!(got.shape, (8, cols.len()));
        let got_dense = got.to_dense().unwrap();
        for (oc, &sc) in cols.iter().enumerate() {
            for r in 0..8 {
                assert_eq!(
                    got_dense[r * cols.len() + oc],
                    dense[r * 12 + sc as usize] as f32,
                    "subset mismatch at row {r} col {sc}"
                );
            }
        }
    }

    #[test]
    fn backed_csc_column_shard_source_trait_dispatch() {
        // Confirm the trait impl delegates correctly.
        let dir = tempfile::tempdir().unwrap();
        let (path, _) = write_csc_test_file(&dir, 8, 12, 4);
        let reader = ScxReader::open(&path).unwrap();
        let csc = BackedCscReader::new(reader, 0).unwrap();
        let trait_obj: &dyn crate::ColumnShardSource = &csc;
        assert_eq!(trait_obj.n_csc_shards(), 3);
        assert_eq!(trait_obj.n_obs(), 8);
        assert_eq!(trait_obj.n_vars(), 12);
        assert_eq!(trait_obj.shape(), (8, 12));
        assert_eq!(trait_obj.csc_shard_col_range(0), Some((0, 4)));
        assert_eq!(trait_obj.csc_shard_col_range(2), Some((8, 12)));
        assert_eq!(trait_obj.csc_shard_col_range(99), None);
        let s0 = trait_obj.read_csc_shard(0).unwrap();
        assert_eq!(s0.n_cols(), 4);
    }
}
