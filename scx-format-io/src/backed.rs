//! Backed (on-demand) CSR access for SCX files.
//!
//! Provides [`BackedCsrIndex`] for O(log n) shard lookups and
//! [`BackedCsrReader`] for on-demand shard decoding with optional LRU caching.
//! Used by pyscx's backed mode to implement AnnData-compatible lazy access.

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use lru::LruCache;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scx_sparse::{ScxCsc, ScxCsr};

use crate::catalog::{FullCatalog, FullCatalogEntry};
use crate::catalog_view::{CatalogView, CatalogViewEntry};
use crate::error::{Result, ScxError};
use crate::prefetch;
use crate::reader::ScxReader;
use crate::section::SectionType;

/// Process-wide switch for the codec-agnostic **block-index** scattered read
/// path (F5 Phase 1). Default on; `SCX_SCATTER_BLOCK_INDEX=0` (or `false`)
/// disables the row-group path so a framed shard falls back to full-shard
/// decode. Read once per process.
///
/// Exposed so callers (e.g. the loader's unframed-file preflight warning) can
/// gate on the same process-global switch that `block_index_eligible` uses —
/// with the path globally disabled, reframing can't enable the fast path, so
/// there is nothing to warn about.
pub fn scatter_block_index_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SCX_SCATTER_BLOCK_INDEX")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

/// A shard request-group takes the O(rows) block-index path only when the
/// requested rows are a small fraction of the shard — `group_len * DIVISOR <
/// shard_rows`. Shared by `read_rows_with`'s `use_block_index` decision and the
/// plan-prefetch skip ([`BackedCsrReader::block_index_eligible`]) so the two can
/// never drift.
pub const ROW_RANGE_WINDOW_DIVISOR: u64 = 4;

// ---------------------------------------------------------------------------
// ShardEntryLite — internal per-shard row
// ---------------------------------------------------------------------------

/// Per-shard catalog row retained by `BackedCsrReader` for read-path
/// dispatch. Drops the `String` name and 32-byte BLAKE3 `checksum`
/// from `FullCatalogEntry`, plus the `value_*` / `col_*` /
/// `column_stats` fields the read path never reads. The remaining
/// 25 bytes (with alignment padding to 32) carry exactly what
/// `read_shard_from_entry`, the MADV_WILLNEED prefetch, and
/// `total_nnz` consume.
///
/// The `to_anndata_backed` worker amplification — N+3 reader opens
/// across W workers × thousands of catalog entries — used to clone
/// ~250-byte `FullCatalogEntry` values per shard. With `ShardEntryLite`
/// the per-shard retained footprint drops ~8× and the per-entry
/// `String` / `Vec<ColumnStat>` allocations disappear from the
/// construction path entirely.
#[derive(Debug, Clone, Copy)]
struct ShardEntryLite {
    offset: u64,
    length: u64,
    /// `nnz` for `total_nnz()` aggregation. Stored even though
    /// `read_shard_from_entry` doesn't need it — it's cheaper to
    /// retain the `u64` than to re-walk the catalog when total_nnz
    /// is called.
    nnz: u64,
    section_type: SectionType,
    modality_id: u8,
}

impl ShardEntryLite {
    /// Build from a `CatalogView` entry whose `stats` is `Some`. The
    /// view's `ShardStatsLite` already carries the dispatched major
    /// axis range; we only need `nnz` here, since row-range lookups
    /// go through `BackedCsrIndex`.
    fn from_view_entry(e: &CatalogViewEntry) -> Self {
        Self {
            offset: e.offset,
            length: e.length,
            nnz: e.stats.as_ref().map_or(0, |s| s.nnz),
            section_type: e.section_type,
            modality_id: e.modality_id,
        }
    }

    /// Synthesise a transient `FullCatalogEntry` for the few reader
    /// APIs (`ScxReader::read_shard_from_entry`,
    /// `ScxReader::section_bytes`) that still take it. `String::new()`
    /// is heap-free and `[0u8; 32]` is a stack array — total cost is
    /// a small stack copy per call, with no allocation. Used by
    /// `read_shard_uncached` and `decode_and_cache` to bridge into
    /// the existing reader API without paying for retained
    /// `FullCatalogEntry` clones.
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

    /// Phase B.5 / D.4: build from a [`FullCatalog`] for a specific
    /// modality. Mirrors `BackedCscIndex::from_catalog_for_modality`
    /// — filters CSR entries by `(SectionType::CsrShard, modality_id)`
    /// and indexes the per-modality position-to-row mapping.
    pub fn from_catalog_for_modality(catalog: &FullCatalog, modality_id: u8) -> Self {
        let mut shard_entries: Vec<ShardRange> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CsrShard && e.modality_id == modality_id)
            .filter_map(|e| {
                e.stats.as_ref().map(|s| ShardRange {
                    row_start: s.row_start,
                    row_end: s.row_end,
                    sorted_shard_idx: 0,
                })
            })
            .collect();
        shard_entries.sort_by_key(|r| r.row_start);
        for (i, entry) in shard_entries.iter_mut().enumerate() {
            entry.sorted_shard_idx = i;
        }
        BackedCsrIndex {
            shard_ranges: shard_entries,
        }
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

    /// Build the row-range index directly from a pre-sorted
    /// `&[&CatalogViewEntry]`. The caller is responsible for sorting
    /// by `stats.major_start`; we just zip the row range pair into
    /// `ShardRange` and stamp the sequential `sorted_shard_idx`. This
    /// is the function that pairs with the `ShardEntryLite::from_view_entry`
    /// builder — both index and lightweight entry list are produced
    /// in a single pass over the catalog view.
    fn from_view_sorted(sorted_view_entries: &[&CatalogViewEntry]) -> Self {
        let shard_ranges: Vec<ShardRange> = sorted_view_entries
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                e.stats.as_ref().map(|s| ShardRange {
                    row_start: s.major_start,
                    row_end: s.major_end,
                    sorted_shard_idx: i,
                })
            })
            .collect();
        BackedCsrIndex { shard_ranges }
    }

    /// Build the index directly from pre-sorted `(row_start, row_end)`
    /// pairs. Used by [`BackedDenseReader`], whose obsm shard catalog
    /// entries carry no `stats` block (row ranges come from each shard's
    /// Arrow schema metadata, not the catalog stats the `from_catalog*` /
    /// `from_view_sorted` constructors read). The caller must pass the
    /// ranges already sorted by `row_start`; `sorted_shard_idx` is
    /// stamped sequentially to index the caller's sorted shard-entry list.
    pub fn from_ranges(ranges: &[(u64, u64)]) -> Self {
        let shard_ranges = ranges
            .iter()
            .enumerate()
            .map(|(i, &(row_start, row_end))| ShardRange {
                row_start,
                row_end,
                sorted_shard_idx: i,
            })
            .collect();
        BackedCsrIndex { shard_ranges }
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
    /// `read_rows_with` shard request-groups served by a full-shard decode.
    /// Covers the planned `!use_block_index` case (cached/dense group) and the
    /// `use_block_index` group that fell back because the shard was unframed.
    pub full_shard_groups: AtomicU64,
    /// `read_rows_with` shard request-groups served by the codec-agnostic
    /// **row-group block-index** path (F5 Phase 1) — a framed (v2) shard decoded
    /// only in its touched groups. The block-index adoption signal, symmetric
    /// with `full_shard_groups`.
    pub block_index_groups: AtomicU64,
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
struct LeaderGuard<'a, K: Eq + Hash> {
    in_flight: &'a Mutex<HashMap<K, Arc<InFlightSlot>>>,
    slot: Arc<InFlightSlot>,
    key: K,
}

impl<K: Eq + Hash> Drop for LeaderGuard<'_, K> {
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
    inner: LruCache<(u32, usize), CacheEntry>,
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
        // Clamp to ≥1: `cache_shards` flows from user-facing Python constructors,
        // and `NonZeroUsize::new(0)` would panic. A 1-shard cache is the minimum
        // sensible budget (the byte budget still bounds memory independently).
        let cap = NonZeroUsize::new(cache_shards.max(1)).unwrap();
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

    fn get(&mut self, key: &(u32, usize)) -> Option<Arc<ScxCsr>> {
        self.inner.get(key).map(|e| Arc::clone(&e.csr))
    }

    fn contains(&self, key: &(u32, usize)) -> bool {
        self.inner.contains(key)
    }

    /// Insert `csr` under `key`, evicting oldest entries until both the
    /// count cap (enforced by the inner `LruCache`) and the byte cap are
    /// satisfied. If a new entry on its own exceeds `bytes_budget`, all
    /// other entries are evicted and the new one is still inserted (the
    /// alternative — refusing to cache — would defeat the cache for any
    /// outsized shard).
    fn put_with_budget(&mut self, key: (u32, usize), csr: Arc<ScxCsr>) {
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

        // Use `push`, not `put`: `LruCache::put` returns `Some` only on a
        // same-key *replacement* and `None` when a new key evicts the LRU, so a
        // count-cap eviction would go uncounted AND its bytes never subtracted
        // (inflating `bytes_used` / `peak_bytes_in_cache`). `push` returns the
        // displaced `(key, entry)` in BOTH cases; a returned key != the inserted
        // key is a genuine eviction.
        let entry = CacheEntry { csr, bytes };
        if let Some((evicted_key, displaced)) = self.inner.push(key, entry) {
            self.bytes_used = self.bytes_used.saturating_sub(displaced.bytes);
            if evicted_key != key {
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

    /// Current count cap.
    fn capacity(&self) -> usize {
        self.inner.cap().get()
    }

    /// Reserve the cache for a multi-pass op: raise the count cap to `min_cap`
    /// (never shrink) and set the byte budget to `byte_budget` — the op's
    /// authoritative RAM ceiling — evicting LRU entries to fit. Setting (rather
    /// than only raising) the byte budget is deliberate: the common
    /// count-only-opened reader starts at `usize::MAX` (unbounded), and an
    /// out-of-core matrix must stay bounded, so the op's budget governs.
    fn reserve_for(&mut self, min_cap: usize, byte_budget: usize) {
        if let Some(cap) = NonZeroUsize::new(min_cap) {
            if min_cap > self.inner.cap().get() {
                self.inner.resize(cap);
            }
        }
        self.bytes_budget = byte_budget;
        while self.bytes_used > self.bytes_budget && !self.inner.is_empty() {
            if let Some((_, evicted)) = self.inner.pop_lru() {
                self.bytes_used = self.bytes_used.saturating_sub(evicted.bytes);
                if let Some(m) = &self.metrics {
                    m.evictions.fetch_add(1, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SharedShardCache
// ---------------------------------------------------------------------------

/// Singleflight table: `(file_id, shard_id) → in-flight decode slot`.
type InFlightTable = Mutex<HashMap<(u32, usize), Arc<InFlightSlot>>>;

/// Decoded-shard cache + singleflight table, keyed by `(file_id, shard_id)` so
/// it can be **shared across several `BackedCsrReader`s** that together back one
/// multi-file run (the Phase 1 multi-reader prefetch engine). A standalone
/// reader gets its own `SharedShardCache` with `file_id = 0`, so keying
/// `(0, shard)` is isomorphic to the old per-reader `shard` key — single-reader
/// behavior is unchanged.
///
/// # Fork safety
///
/// **Invariant**: a `SharedShardCache` is owned (via `Arc`) by the reader(s) of
/// one run *instance*, never a process-global `OnceCell` / `static` /
/// `lazy_static`. A forked child constructs its own cache post-fork (e.g. a
/// `DataLoader` worker building readers inside `__iter__`), so it never inherits
/// a poisoned-locked `Mutex`. Sharing across readers within one instance keeps
/// that contract — the fork-mode regression test
/// (`pyscx/tests/test_fork_safety.py`) catches any regression.
pub struct SharedShardCache {
    /// `None` when `cache_shards == 0` (no caching; every read decodes).
    cache: Option<Mutex<WeightedLruCache>>,
    /// Singleflight table for in-flight decodes, same fork-safety contract as
    /// `cache`. Present iff `cache` is.
    in_flight: Option<InFlightTable>,
    /// Opt-in counters, shared by every reader on this cache.
    metrics: OnceLock<Arc<CacheMetrics>>,
    /// Open-time count cap, mirrored for `warm_shards` chunking without locking.
    cache_shards: usize,
}

impl SharedShardCache {
    /// Build a shared cache with a count cap (`cache_shards`, 0 = no cache) and
    /// byte budget. The budget governs *all* readers sharing this cache.
    pub fn new(cache_shards: usize, bytes_budget: usize) -> Arc<Self> {
        let (cache, in_flight) = if cache_shards > 0 {
            (
                Some(Mutex::new(WeightedLruCache::new(
                    cache_shards,
                    bytes_budget,
                ))),
                Some(Mutex::new(HashMap::new())),
            )
        } else {
            (None, None)
        };
        Arc::new(SharedShardCache {
            cache,
            in_flight,
            metrics: OnceLock::new(),
            cache_shards,
        })
    }

    fn has_cache(&self) -> bool {
        self.cache.is_some()
    }

    fn contains(&self, fid: u32, shard: usize) -> bool {
        match &self.cache {
            Some(m) => m.lock().unwrap().contains(&(fid, shard)),
            None => false,
        }
    }

    fn in_flight_contains(&self, fid: u32, shard: usize) -> bool {
        match &self.in_flight {
            Some(m) => m.lock().unwrap().contains_key(&(fid, shard)),
            None => false,
        }
    }

    fn capacity(&self) -> usize {
        match &self.cache {
            Some(m) => m.lock().unwrap().capacity(),
            None => 0,
        }
    }

    fn reserve_for(&self, min_shards: usize, max_cache_bytes: usize) -> usize {
        match &self.cache {
            Some(m) => {
                let mut cache = m.lock().unwrap();
                cache.reserve_for(min_shards, max_cache_bytes);
                cache.capacity()
            }
            None => 0,
        }
    }

    fn metrics(&self) -> Option<&Arc<CacheMetrics>> {
        self.metrics.get()
    }

    /// Enable cache-behavior metrics. Idempotent: the first call installs the
    /// counters (also wiring them into the LRU for eviction/byte stats);
    /// later calls return the same handle.
    fn enable_metrics(&self) -> Arc<CacheMetrics> {
        let m = self
            .metrics
            .get_or_init(|| Arc::new(CacheMetrics::default()))
            .clone();
        if let Some(ref cache_mutex) = self.cache {
            let mut c = cache_mutex.lock().unwrap();
            if c.metrics.is_none() {
                c.metrics = Some(Arc::clone(&m));
            }
        }
        m
    }

    /// Dedup `shard_indices` (local ids) to those neither cached nor in flight
    /// for `fid`. Lock order `in_flight` → `cache` matches [`Self::get_or_decode`]
    /// so concurrent callers can't deadlock.
    fn filter_misses(&self, fid: u32, shard_indices: &[usize]) -> Vec<usize> {
        let (Some(cache_mutex), Some(in_flight_mutex)) = (&self.cache, &self.in_flight) else {
            return Vec::new();
        };
        let mut seen: HashSet<usize> = HashSet::with_capacity(shard_indices.len());
        let mut misses: Vec<usize> = Vec::with_capacity(shard_indices.len());
        let in_flight = in_flight_mutex.lock().unwrap();
        let cache = cache_mutex.lock().unwrap();
        for &idx in shard_indices {
            if !seen.insert(idx) {
                continue;
            }
            if cache.contains(&(fid, idx)) || in_flight.contains_key(&(fid, idx)) {
                continue;
            }
            misses.push(idx);
        }
        misses
    }

    /// Return the cached `(fid, shard)` CSR, or run `decode` exactly once across
    /// concurrent callers (singleflight) and cache the result under the budget.
    /// `decode` produces the decoded shard; the caller (the reader) owns the
    /// decode because it is reader/mmap-specific. Mirrors the former
    /// `BackedCsrReader::read_shard_cached_arc` loop, keyed by `(fid, shard)`.
    fn get_or_decode(
        &self,
        fid: u32,
        shard: usize,
        decode: impl FnOnce() -> Result<Arc<ScxCsr>>,
    ) -> Result<Arc<ScxCsr>> {
        let key = (fid, shard);
        loop {
            // Cache hit fast path.
            if let Some(ref cache_mutex) = self.cache {
                let mut cache = cache_mutex.lock().unwrap();
                if let Some(cached) = cache.get(&key) {
                    if let Some(m) = self.metrics.get() {
                        m.hits.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(cached);
                }
            }

            // Singleflight: claim leadership or wait on a peer leader. `_guard`
            // (when present) signals waiters + removes the slot on drop, so a
            // panic in `decode` still wakes peers.
            let _guard: Option<LeaderGuard<(u32, usize)>> = match &self.in_flight {
                Some(in_flight_mutex) => {
                    let mut in_flight = in_flight_mutex.lock().unwrap();
                    if let Some(existing) = in_flight.get(&key) {
                        let slot = Arc::clone(existing);
                        drop(in_flight);
                        if let Some(m) = self.metrics.get() {
                            m.duplicate_waiters.fetch_add(1, Ordering::Relaxed);
                        }
                        let mut state = slot.state.lock().unwrap();
                        while !*state {
                            state = slot.cv.wait(state).unwrap();
                        }
                        drop(state);
                        // Re-check the cache; on leader-success we hit the fast
                        // path, on leader-error we claim a fresh slot.
                        continue;
                    }
                    // Re-check the cache while holding `in_flight` to close the
                    // race where a peer leader finished between our miss and our
                    // acquisition of `in_flight`.
                    if let Some(ref cache_mutex) = self.cache {
                        let mut cache = cache_mutex.lock().unwrap();
                        if let Some(cached) = cache.get(&key) {
                            if let Some(m) = self.metrics.get() {
                                m.hits.fetch_add(1, Ordering::Relaxed);
                            }
                            return Ok(cached);
                        }
                    }
                    let slot = Arc::new(InFlightSlot::new());
                    in_flight.insert(key, Arc::clone(&slot));
                    Some(LeaderGuard {
                        in_flight: in_flight_mutex,
                        slot,
                        key,
                    })
                }
                None => None,
            };

            if let Some(m) = self.metrics.get() {
                m.misses.fetch_add(1, Ordering::Relaxed);
            }

            // Leader path. Decode (reader-owned), then insert under the budget
            // before `_guard` drops — a post-removal observer sees the entry
            // once it re-acquires `in_flight`.
            let csr = decode()?;
            if let Some(ref cache_mutex) = self.cache {
                let mut cache = cache_mutex.lock().unwrap();
                cache.put_with_budget(key, Arc::clone(&csr));
            }
            return Ok(csr);
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
    reader: ScxReader,
    index: BackedCsrIndex,
    n_vars: usize,
    n_obs: usize,
    /// If set, this reader targets a specific layer rather than X.
    layer_name: Option<String>,
    /// Pre-sorted lightweight catalog rows for X shards. Drops the
    /// per-shard `String` name and 32-byte BLAKE3 checksum that the
    /// read path never consumes. See [`ShardEntryLite`] for the
    /// field set and per-shard footprint.
    x_sorted_entries: Vec<ShardEntryLite>,
    /// Pre-sorted lightweight catalog rows for layer shards (empty
    /// when this reader targets X). Same `ShardEntryLite` shape as
    /// `x_sorted_entries`.
    sorted_entries: Vec<ShardEntryLite>,
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
    /// leader's decode fails (or panics — see [`LeaderGuard`]), waiters
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
        shard_cache.get_or_decode(self.file_id, shard_idx, || self.decode_shard(shard_idx))
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
    fn warm_shards(&self, shard_indices: &[usize]) -> Result<()> {
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

    // --- Native shard-by-shard aggregation ---
    //
    // These methods compute statistics without materializing the full
    // concatenated CSR. Peak memory is `SCX_ACCEL_PREFETCH_DEPTH` decoded
    // shards (default 4) plus the output vector — the ordered decode-prefetch
    // pipeline keeps that many in flight so decode overlaps the reduction.
    // It was one shard before Phase 4.2, and still is whenever the pipeline
    // declines to engage (depth 1, a single shard, a one-thread rayon pool, or
    // a caller that is itself a rayon worker).
    //
    // The bound is **per call**, and the depth knob is process-global: N
    // concurrent callers hold N x depth shards. `pyscx.accel.col_*` release the
    // GIL, so that is reachable from Python threads.

    /// Compute per-row sums without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sums from each shard's
    /// CSR arrays, and concatenates the results.
    pub fn row_sums(&self) -> Result<Vec<f64>> {
        let mut all_sums = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_sums.extend(csr.row_sums());
                Ok(())
            },
        )?;
        Ok(all_sums)
    }

    /// Compute per-column sums without materializing the full matrix.
    ///
    /// Iterates shards, accumulates column sums into a single `n_vars`-length vector.
    pub fn col_sums(&self) -> Result<Vec<f64>> {
        let mut sums = vec![0.0f64; self.n_vars];
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let partial = csr.col_sums();
                for (s, p) in sums.iter_mut().zip(partial.iter()) {
                    *s += p;
                }
                Ok(())
            },
        )?;
        Ok(sums)
    }

    /// Compute per-row NNZ counts without materializing the full matrix.
    pub fn row_nnz(&self) -> Result<Vec<i64>> {
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_nnz.extend(csr.row_nnz());
                Ok(())
            },
        )?;
        Ok(all_nnz)
    }

    /// Compute per-row NNZ and sums in a single shard scan.
    ///
    /// Avoids the double I/O of calling `row_nnz()` + `row_sums()` separately.
    /// Used by `filter_cells` when both `min_genes` and `min_counts` are specified.
    pub fn row_nnz_and_sums(&self) -> Result<(Vec<i64>, Vec<f64>)> {
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        let mut all_sums = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                for row in 0..csr.n_rows() {
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    all_nnz.push((end - start) as i64);
                    all_sums.push(csr.data[start..end].iter().map(|&v| v as f64).sum());
                }
                Ok(())
            },
        )?;
        Ok((all_nnz, all_sums))
    }

    /// Compute per-column sums and per-column NNZ in a single shard scan.
    ///
    /// The column-axis twin of [`Self::row_nnz_and_sums`]: avoids the second
    /// full decode of every shard that calling `col_sums()` and `col_nnz()`
    /// separately incurs. Used by `calculate_qc_metrics`' gene axis and by
    /// `filter_genes` when both a cell and a count threshold are given.
    ///
    /// Bit-identical to the two separate calls — same per-shard visit order,
    /// same left-to-right f64 accumulation.
    pub fn col_sums_and_nnz(&self) -> Result<(Vec<f64>, Vec<u32>)> {
        let mut sums = vec![0.0f64; self.n_vars];
        let mut counts = vec![0u32; self.n_vars];
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let (partial_sums, partial_nnz) = csr.col_sums_and_nnz();
                for (s, p) in sums.iter_mut().zip(partial_sums.iter()) {
                    *s += p;
                }
                for (c, p) in counts.iter_mut().zip(partial_nnz.iter()) {
                    *c = c.saturating_add(*p);
                }
                Ok(())
            },
        )?;
        Ok((sums, counts))
    }

    /// Compute per-column NNZ counts without materializing the full matrix.
    pub fn col_nnz(&self) -> Result<Vec<u32>> {
        let mut counts = vec![0u32; self.n_vars];
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let partial = csr.col_nnz();
                for (c, p) in counts.iter_mut().zip(partial.iter()) {
                    *c = c.saturating_add(*p);
                }
                Ok(())
            },
        )?;
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
        // `ShardEntryLite` carries `nnz` directly — no `.stats`
        // indirection per shard.
        let total: u64 = entries.iter().map(|e| e.nnz).sum();
        Ok(total as usize)
    }

    /// Largest stored value across this reader's shards, **when the catalog
    /// can prove one** — `None` when it cannot.
    ///
    /// Walks the catalog only (O(shards), no payload reads, no decode), which
    /// is the whole point: it answers "how big do the values get?" without
    /// touching the data. `ShardStats::value_max` is exact for the integer
    /// value encodings and is written as `0` for `Float32`/`Float16`, where
    /// a `u32` field cannot represent the statistic — so:
    ///
    /// * `Some(m)` with `m > 0` — every contributing shard is integer-encoded
    ///   and `m` is the true maximum.
    /// * `Some(0)` — the matrix is empty (`nnz == 0`), so `0` *is* the maximum
    ///   and the catalog proves it.
    /// * `None` — a matrix whose maximum the catalog cannot bound: the shards
    ///   are float-encoded, or they carry no stats (or, vanishingly, the `nnz`
    ///   lookup itself failed). Those are indistinguishable from here, so
    ///   callers must not report any one as the cause; `None` means *unknown*,
    ///   and a caller that needs a real answer has to stream
    ///   ([`Self::col_max`]) or ask the user.
    ///
    /// Covers X shards when this reader targets X, layer shards otherwise —
    /// the same entry set as [`Self::total_nnz`].
    pub fn catalog_int_value_max(&self) -> Option<u32> {
        let want_layer = self.layer_name.is_some();
        let modality = self.modality_id();
        // Hoisted out of the filter: otherwise every catalog entry pays a
        // `format!` allocation just to be compared against.
        let layer_prefix = self.layer_name.as_ref().map(|n| format!("{n}_shard_"));
        let max = self
            .reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.modality_id == modality
                    && if want_layer {
                        e.section_type == SectionType::LayerCsrShard
                            && layer_prefix
                                .as_ref()
                                .is_some_and(|prefix| e.name.starts_with(prefix))
                    } else {
                        e.section_type == SectionType::CsrShard
                    }
            })
            .filter_map(|e| e.stats.as_ref())
            .map(|s| s.value_max)
            .max()
            .unwrap_or(0);
        if max > 0 {
            return Some(max);
        }
        // `value_max == 0` is ambiguous between "float-encoded / no stats" and
        // "there are no values". `nnz` disambiguates: an empty matrix really
        // does max to 0, and saying so beats making the caller refuse.
        match self.total_nnz() {
            Ok(0) => Some(0),
            _ => None,
        }
    }

    /// Compute per-row sum of squared values without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sum-of-squares from each shard's
    /// CSR arrays, and concatenates the results. Used for scalar variance:
    /// `Var(X) = E[X²] - (E[X])²`.
    pub fn row_sum_of_squares(&self) -> Result<Vec<f64>> {
        let mut all_sq = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_sq.extend(csr.row_sum_of_squares());
                Ok(())
            },
        )?;
        Ok(all_sq)
    }

    // --- Variance ---

    /// Streaming per-row variance without materializing the full matrix.
    ///
    /// Each shard independently computes row variances (one row = one shard's row).
    pub fn row_var(&self) -> Result<Vec<f64>> {
        let mut all_var = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_var.extend(csr.row_var());
                Ok(())
            },
        )?;
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
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let partial = csr.col_var_partial(&col_means);
                for (s, p) in sq_devs.iter_mut().zip(partial.iter()) {
                    *s += p;
                }
                let nnz = csr.col_nnz();
                for (c, &n) in col_nnz.iter_mut().zip(nnz.iter()) {
                    *c += n as usize;
                }
                Ok(())
            },
        )?;

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
        let mut all_max = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_max.extend(csr.row_max());
                Ok(())
            },
        )?;
        Ok(all_max)
    }

    /// Streaming per-column max without materializing the full matrix.
    ///
    /// Merges per-shard column maxes. Accounts for implicit zeros:
    /// if any column has fewer stored entries than `n_obs`, the max is
    /// at least 0.0.
    pub fn col_max(&self) -> Result<Vec<f64>> {
        let mut maxes = vec![f64::NEG_INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
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
                Ok(())
            },
        )?;

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
        let mut all_min = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_min.extend(csr.row_min());
                Ok(())
            },
        )?;
        Ok(all_min)
    }

    /// Streaming per-column min without materializing the full matrix.
    pub fn col_min(&self) -> Result<Vec<f64>> {
        let mut mins = vec![f64::INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                    let c = col as usize;
                    let v = val as f64;
                    mins[c] = mins[c].min(v);
                    col_nnz[c] += 1;
                }
                Ok(())
            },
        )?;

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

    /// Column sums and column NNZ over the kept rows, in a single shard scan.
    ///
    /// Deletion-aware twin of [`Self::col_sums_and_nnz`]. Bit-identical to
    /// `col_sums_masked()` + `col_nnz_masked()`, which walk the same rows in
    /// the same order — this just decodes each shard once instead of twice.
    pub fn col_sums_and_nnz_masked(&self, kept_rows: &[u64]) -> Result<(Vec<f64>, Vec<u32>)> {
        let mut sums = vec![0.0f64; self.n_vars];
        let mut counts = vec![0u32; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> Result<()> {
                let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                    Some(r) => r,
                    None => return Ok(()),
                };
                // `kept_rows` is sorted by construction (see compute_kept_to_global),
                // so the shard's slice is a binary-search range.
                let lo = kept_rows.partition_point(|&r| r < s_start);
                let hi = kept_rows.partition_point(|&r| r < s_end);
                for &global_row in &kept_rows[lo..hi] {
                    let local_row = (global_row - s_start) as usize;
                    let row_start = csr.indptr[local_row] as usize;
                    let row_end = csr.indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        let c = csr.indices[j] as usize;
                        sums[c] += csr.data[j] as f64;
                        counts[c] += 1;
                    }
                }
                Ok(())
            },
        )?;
        Ok((sums, counts))
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

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> Result<()> {
                let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                    Some(r) => r,
                    None => return Ok(()),
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
                Ok(())
            },
        )?;

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

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> Result<()> {
                let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                    Some(r) => r,
                    None => return Ok(()),
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
                Ok(())
            },
        )?;

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
            // Serve from the decoded-shard cache so this pass shares the LRU
            // with the streaming SpMM passes (multi-pass out-of-core PCA reuses
            // each shard instead of re-decoding it here).
            let csr = self.read_shard_cached_arc(shard_idx)?;
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
    /// Build from a [`FullCatalog`], scoped to global / single-modality
    /// CSC shards (`modality_id == 0`). Backwards-compatible entry
    /// point for v1 files and single-modality v2 files.
    pub fn from_catalog(catalog: &FullCatalog) -> Self {
        Self::from_catalog_for_modality(catalog, 0)
    }

    /// Build from a [`FullCatalog`], scoped to a specific modality.
    /// Filters CSC shards by `(SectionType::CscShard, modality_id)`,
    /// sorts by `col_start`, and records each shard's position in
    /// sorted order. v2 multimodal files use this constructor with
    /// `modality_id >= 1`.
    pub fn from_catalog_for_modality(catalog: &FullCatalog, modality_id: u8) -> Self {
        let mut shard_entries: Vec<CscShardRange> = catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::CscShard && e.modality_id == modality_id)
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

    /// Build from a [`FullCatalog`], scoped to a layer's CSC shards
    /// for a given modality. Filters by
    /// `(SectionType::LayerCscShard, modality_id, layer_name)`. The
    /// `layer_name` matches by substring `/{layer_name}/` in the
    /// section name (mirrors `FullCatalog::layer_csc_shards_for_modality`).
    pub fn from_catalog_for_layer(
        catalog: &FullCatalog,
        modality_id: u8,
        layer_name: &str,
    ) -> Self {
        let needle = format!("/{layer_name}/");
        let mut shard_entries: Vec<CscShardRange> = catalog
            .entries
            .iter()
            .filter(|e| {
                e.section_type == SectionType::LayerCscShard
                    && e.modality_id == modality_id
                    && e.name.contains(&needle)
            })
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
/// Reject a CSC sidecar that was built against an earlier generation of
/// the CSR data (the freshness guard introduced with `catalog_version`
/// v4). A sidecar is fresh iff `csc_build_generation == data_generation`;
/// CSR-mutating writers bump `data_generation`, and only a CSC (re)build
/// advances `csc_build_generation` to match.
///
/// The check is skipped when `csc_entries` is empty (no sidecar to
/// validate) and is a no-op for v1–v3 files (both counters default to
/// `0`, so `0 == 0`), so existing valid sidecars are never rejected.
fn check_csc_sidecar_fresh(catalog: &FullCatalog, csc_entries: &[FullCatalogEntry]) -> Result<()> {
    if csc_entries.is_empty() {
        return Ok(());
    }
    if catalog.csc_build_generation != catalog.data_generation {
        return Err(ScxError::StaleCscSidecar {
            built_generation: catalog.csc_build_generation,
            data_generation: catalog.data_generation,
        });
    }
    Ok(())
}

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
        // `push` (not `put`) so a count-cap eviction is counted: `put` returns
        // `None` when a new key evicts the LRU, so the eviction would be missed.
        // A returned key != the inserted key is a genuine eviction.
        if let Some((evicted_key, _displaced)) = self.inner.push(key, value) {
            if evicted_key != key {
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

    /// Create a new backed CSC reader from an [`ScxReader`], scoped
    /// to the global / single-modality CSC shards
    /// (`modality_id == 0`). Convenience wrapper around
    /// [`Self::for_modality`] kept for backward compatibility with
    /// v1 / single-modality v2 callers.
    ///
    /// `cache_shards`: number of decoded CSC shards to cache (0 = no
    /// cache). The underlying file must have CSC sidecar shards
    /// (`reader.header().has_csc()`); otherwise `read_csc_shard` will
    /// always return an out-of-bounds error.
    pub fn new(reader: ScxReader, cache_shards: usize) -> Result<Self> {
        Self::for_modality(reader, 0, cache_shards)
    }

    /// Create a backed CSC reader scoped to a specific modality.
    /// Filters CSC shards by `(SectionType::CscShard, modality_id)`
    /// so each modality gets its own LRU cache and shard range —
    /// avoids cache thrashing under interleaved access patterns
    /// (e.g. totalVI training touching RNA + ADT in the same step).
    pub fn for_modality(reader: ScxReader, modality_id: u8, cache_shards: usize) -> Result<Self> {
        let index = BackedCscIndex::from_catalog_for_modality(reader.catalog(), modality_id);
        let n_obs = reader.n_obs() as usize;
        // For multimodal files the per-modality `n_vars` lives on
        // the modality table; the file-level `header.n_vars` is the
        // primary modality's count or aggregate. Prefer the modality
        // info when available.
        let n_vars = match reader.modality_info(modality_id) {
            Some(info) => info.n_vars as usize,
            None => reader.n_vars() as usize,
        };
        let sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .csc_shards_for_modality(modality_id)
            .into_iter()
            .cloned()
            .collect();
        check_csc_sidecar_fresh(reader.catalog(), &sorted_entries)?;
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

    /// Create a backed CSC reader scoped to a layer's CSC sidecar
    /// for a given modality. Filters by
    /// `(SectionType::LayerCscShard, modality_id, layer_name)`.
    pub fn for_layer(
        reader: ScxReader,
        modality_id: u8,
        layer_name: &str,
        cache_shards: usize,
    ) -> Result<Self> {
        let index =
            BackedCscIndex::from_catalog_for_layer(reader.catalog(), modality_id, layer_name);
        let n_obs = reader.n_obs() as usize;
        let n_vars = match reader.modality_info(modality_id) {
            Some(info) => info.n_vars as usize,
            None => reader.n_vars() as usize,
        };
        let sorted_entries: Vec<FullCatalogEntry> = reader
            .catalog()
            .layer_csc_shards_for_modality(modality_id, layer_name)
            .into_iter()
            .cloned()
            .collect();
        check_csc_sidecar_fresh(reader.catalog(), &sorted_entries)?;
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
        // See `BackedCsrReader::read_shard_cached_arc`: a cache hit bypasses
        // `section_bytes` entirely.
        self.check_fresh()?;
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
            let (shard_lo, shard_hi) = self.index.shard_col_range(shard_idx).ok_or_else(|| {
                ScxError::InvalidCatalog(format!(
                    "BackedCscIndex missing range for shard {shard_idx}"
                ))
            })?;
            let lo_in_shard = c_lo.saturating_sub(shard_lo) as usize;
            let hi_in_shard = (c_hi.min(shard_hi).saturating_sub(shard_lo)) as usize;

            // A row-group-framed (v2) CSC shard is column-group indexed,
            // so a gene-subset read decodes only the touched column-groups via the
            // block index (major axis = columns) instead of full-decoding the whole
            // shard through the LRU cache. Non-framed shards fall back to the cached
            // full-decode + `col_slice` path below.
            if hi_in_shard > lo_in_shard {
                if let Some(entry) = self.sorted_entries.get(shard_idx) {
                    let header = self.reader.read_shard_header(entry)?;
                    if header.shard_format_version
                        > crate::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION
                    {
                        let run = (lo_in_shard, hi_in_shard - lo_in_shard);
                        if let Some(mut runs) =
                            self.reader.decode_block_index_row_runs(entry, &[run])?
                        {
                            if let Some((indptr, indices, data)) = runs.pop() {
                                decoded.push(ScxCsc::new_unchecked(
                                    (self.n_obs, hi_in_shard - lo_in_shard),
                                    indptr,
                                    indices,
                                    data,
                                ));
                                continue;
                            }
                        }
                    }
                }
            }

            let csc = self.read_shard_cached(shard_idx)?;
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

// `concatenate_csr` moved to `scx_sparse::concatenate_csr` (operates purely on
// `ScxCsr`, not format I/O).

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

/// LRU cache of decoded dense shards with both a count cap and a byte
/// cap. Dense analog of [`WeightedLruCache`]; bytes are measured exactly
/// via `RecordBatch::get_array_memory_size()` rather than the CSR
/// component formula.
struct DenseLruCache {
    inner: LruCache<usize, DenseCacheEntry>,
    bytes_budget: usize,
    bytes_used: usize,
    metrics: Option<Arc<CacheMetrics>>,
}

struct DenseCacheEntry {
    shard: Arc<DenseShard>,
    bytes: usize,
}

impl DenseLruCache {
    fn new(cache_shards: usize, bytes_budget: usize) -> Self {
        // Clamp to ≥1 — see `WeightedLruCache::new`; `NonZeroUsize::new(0)` panics.
        let cap = NonZeroUsize::new(cache_shards.max(1)).unwrap();
        DenseLruCache {
            inner: LruCache::new(cap),
            bytes_budget,
            bytes_used: 0,
            metrics: None,
        }
    }

    fn estimate_bytes(shard: &DenseShard) -> usize {
        shard.batch.get_array_memory_size()
    }

    fn get(&mut self, key: &usize) -> Option<Arc<DenseShard>> {
        self.inner.get(key).map(|e| Arc::clone(&e.shard))
    }

    fn contains(&self, key: &usize) -> bool {
        self.inner.contains(key)
    }

    fn put_with_budget(&mut self, key: usize, shard: Arc<DenseShard>) {
        let bytes = Self::estimate_bytes(&shard);
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
        // `push` (not `put`): see `WeightedLruCache::put_with_budget` — `put`
        // returns `None` on a count-cap eviction, so the eviction would go
        // uncounted and its bytes never subtracted. `push` returns the displaced
        // entry; a returned key != the inserted key is a genuine eviction.
        let entry = DenseCacheEntry { shard, bytes };
        if let Some((evicted_key, displaced)) = self.inner.push(key, entry) {
            self.bytes_used = self.bytes_used.saturating_sub(displaced.bytes);
            if evicted_key != key {
                if let Some(m) = &self.metrics {
                    m.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.bytes_used = self.bytes_used.saturating_add(bytes);
        if let Some(m) = &self.metrics {
            m.bytes_inserted.fetch_add(bytes as u64, Ordering::Relaxed);
            m.peak_bytes_in_cache
                .fetch_max(self.bytes_used as u64, Ordering::Relaxed);
        }
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
    cache: Option<Mutex<DenseLruCache>>,
    in_flight: Option<Mutex<HashMap<usize, Arc<InFlightSlot>>>>,
    metrics: Option<Arc<CacheMetrics>>,
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
        let layout = reader.dense_mapping_layout(
            "obsm",
            name,
            SectionType::ObsmEmbeddingShard,
            SectionType::ObsmEmbedding,
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
        layout: crate::reader::DenseMappingLayout,
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
        let cache = if cache_shards > 0 {
            Some(Mutex::new(DenseLruCache::new(cache_shards, bytes_budget)))
        } else {
            None
        };
        let in_flight = if cache.is_some() {
            Some(Mutex::new(HashMap::new()))
        } else {
            None
        };
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
            cache,
            in_flight,
            metrics: None,
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
    pub fn enable_metrics(&mut self) -> Arc<CacheMetrics> {
        let m = Arc::new(CacheMetrics::default());
        if let Some(ref cache_mutex) = self.cache {
            cache_mutex.lock().unwrap().metrics = Some(Arc::clone(&m));
        }
        self.metrics = Some(Arc::clone(&m));
        m
    }

    /// True if `shard_idx` is currently cached.
    pub fn cache_contains(&self, shard_idx: usize) -> bool {
        match &self.cache {
            Some(m) => m.lock().unwrap().contains(&shard_idx),
            None => false,
        }
    }

    fn shard_count(&self) -> usize {
        self.sorted_entries.len()
    }

    /// Decode + cache one dense shard, returning a shared `Arc`. Same
    /// singleflight contract as [`BackedCsrReader::read_shard_cached_arc`].
    fn read_shard_cached_arc(&self, shard_idx: usize) -> Result<Arc<DenseShard>> {
        // See `BackedCsrReader::read_shard_cached_arc`: a cache hit bypasses
        // `section_bytes` entirely.
        self.check_fresh()?;
        loop {
            if let Some(ref cache_mutex) = self.cache {
                let mut cache = cache_mutex.lock().unwrap();
                if let Some(cached) = cache.get(&shard_idx) {
                    if let Some(m) = &self.metrics {
                        m.hits.fetch_add(1, Ordering::Relaxed);
                    }
                    return Ok(cached);
                }
            }

            let _guard: Option<LeaderGuard<usize>> = match &self.in_flight {
                Some(in_flight_mutex) => {
                    let mut in_flight = in_flight_mutex.lock().unwrap();
                    if let Some(existing) = in_flight.get(&shard_idx) {
                        let slot = Arc::clone(existing);
                        drop(in_flight);
                        if let Some(m) = &self.metrics {
                            m.duplicate_waiters.fetch_add(1, Ordering::Relaxed);
                        }
                        let mut state = slot.state.lock().unwrap();
                        while !*state {
                            state = slot.cv.wait(state).unwrap();
                        }
                        drop(state);
                        continue;
                    }
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

            return self.decode_and_cache(shard_idx);
        }
    }

    fn decode_and_cache(&self, shard_idx: usize) -> Result<Arc<DenseShard>> {
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

        if let Some(ref cache_mutex) = self.cache {
            let mut cache = cache_mutex.lock().unwrap();
            cache.put_with_budget(shard_idx, Arc::clone(&shard));
        }

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
        if self.cache.is_none() {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "backed_tests.rs"]
mod tests;
