//! The decoded-shard cache: one byte-budgeted LRU and one singleflight,
//! shared by all three readers.
//!
//! There used to be three of each. [`WeightedLruCache`] is generic over the
//! payload via [`SizeHint`], and [`ShardCache::get_or_decode`] is the only
//! place that takes both the `in_flight` and `cache` locks — in that order,
//! which is what makes the lock ordering a property of the code rather than a
//! contract three transcriptions had to honour separately.

use super::*;

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

/// Decoded-payload size in bytes, as the cache's byte budget accounts for it.
///
/// The three caches measured their payloads three different ways in three
/// different places. This is that measure, once, next to the type it describes:
/// a CSR/CSC shard by the per-component model `IndexPlanLoader`'s memory-budget
/// auto-tune also uses, a dense shard by Arrow's own accounting.
pub trait SizeHint {
    fn size_bytes(&self) -> usize;
}

/// `indptr.len()*8 + indices.len()*4 + data.len()*4`.
fn csr_component_bytes(indptr: usize, indices: usize, data: usize) -> usize {
    indptr
        .saturating_mul(8)
        .saturating_add(indices.saturating_mul(4))
        .saturating_add(data.saturating_mul(4))
}

impl SizeHint for ScxCsr {
    fn size_bytes(&self) -> usize {
        csr_component_bytes(self.indptr.len(), self.indices.len(), self.data.len())
    }
}

impl SizeHint for ScxCsc {
    fn size_bytes(&self) -> usize {
        // Same three `Vec<i64>` / `Vec<i32>` / `Vec<f32>` components as
        // `ScxCsr`, just column-major, so the same model applies.
        csr_component_bytes(self.indptr.len(), self.indices.len(), self.data.len())
    }
}

struct CacheEntry<V> {
    value: Arc<V>,
    bytes: usize,
}

/// LRU cache with both a count cap and a byte cap. Evicts oldest entries
/// until both caps are satisfied for a new insertion.
///
/// Generic over the decoded payload: `ScxCsr` for X/layer shards, `ScxCsc` for
/// the gene-major sidecar, `DenseShard` for an `obsm` embedding. Bytes come
/// from [`SizeHint`], which is the only thing that differed between the three
/// hand-rolled caches this replaced.
///
/// `K: Copy` is load-bearing, not incidental: `put_with_budget` compares the
/// key `LruCache::push` hands back against the key it inserted, which is how a
/// genuine eviction is told apart from a same-key replacement.
pub(super) struct WeightedLruCache<K: Eq + Hash + Copy, V: SizeHint> {
    inner: LruCache<K, CacheEntry<V>>,
    /// Hard byte cap. `usize::MAX` means count-only behavior (compatible
    /// with `BackedCsrReader::new`).
    bytes_budget: usize,
    /// Cumulative bytes currently in `inner`.
    bytes_used: usize,
    /// Optional metrics handle (cloned from the owning reader on construction).
    pub(super) metrics: Option<Arc<CacheMetrics>>,
}

impl<K: Eq + Hash + Copy, V: SizeHint> WeightedLruCache<K, V> {
    pub(super) fn new(cache_shards: usize, bytes_budget: usize) -> Self {
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

    pub(super) fn get(&mut self, key: &K) -> Option<Arc<V>> {
        self.inner.get(key).map(|e| Arc::clone(&e.value))
    }

    pub(super) fn contains(&self, key: &K) -> bool {
        self.inner.contains(key)
    }

    /// Insert `value` under `key`, evicting oldest entries until both the
    /// count cap (enforced by the inner `LruCache`) and the byte cap are
    /// satisfied. If a new entry on its own exceeds `bytes_budget`, all
    /// other entries are evicted and the new one is still inserted (the
    /// alternative — refusing to cache — would defeat the cache for any
    /// outsized shard).
    pub(super) fn put_with_budget(&mut self, key: K, value: Arc<V>) {
        let bytes = value.size_bytes();

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
        let entry = CacheEntry { value, bytes };
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
    pub(super) fn capacity(&self) -> usize {
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
// ShardCache
// ---------------------------------------------------------------------------

/// Singleflight table: `key → in-flight decode slot`.
type InFlightTable<K> = Mutex<HashMap<K, Arc<InFlightSlot>>>;

/// Decoded-shard cache + singleflight table.
///
/// The CSR instantiation is keyed by `(file_id, shard_id)` so it can be
/// **shared across several `BackedCsrReader`s** that together back one
/// multi-file run (the Phase 1 multi-reader prefetch engine). A standalone
/// reader gets its own cache with `file_id = 0`, so keying `(0, shard)` is
/// isomorphic to a per-reader `shard` key — single-reader behavior is
/// unchanged. The CSC and dense readers key by plain `shard_idx`; they are
/// per-reader and have no second file to disambiguate.
///
/// # Fork safety
///
/// **Invariant**: a `ShardCache` is owned (via `Arc` for CSR, inline for CSC
/// and dense) by the reader(s) of one run *instance*, never a process-global
/// `OnceCell` / `static` / `lazy_static`. A forked child constructs its own
/// cache post-fork (e.g. a `DataLoader` worker building readers inside
/// `__iter__`), so it never inherits a poisoned-locked `Mutex`. Sharing across
/// readers within one instance keeps that contract — the fork-mode regression
/// test (`pyscx/tests/test_fork_safety.py`) catches any regression.
pub struct ShardCache<K: Eq + Hash + Copy, V: SizeHint> {
    /// `None` when `cache_shards == 0` (no caching; every read decodes).
    cache: Option<Mutex<WeightedLruCache<K, V>>>,
    /// Singleflight table for in-flight decodes, same fork-safety contract as
    /// `cache`. Present iff `cache` is.
    in_flight: Option<InFlightTable<K>>,
    /// Opt-in counters, shared by every reader on this cache.
    metrics: OnceLock<Arc<CacheMetrics>>,
    /// Open-time count cap, mirrored for `warm_shards` chunking without locking.
    pub(super) cache_shards: usize,
}

/// The CSR instantiation, and the name every caller outside this crate uses.
///
/// Kept as an alias rather than renaming the type: `SharedShardCache::new` is
/// called by `scx-loader`'s plan engine and the name is re-exported at the
/// crate root, so generalising the struct must not become a cross-crate API
/// change.
pub type SharedShardCache = ShardCache<(u32, usize), ScxCsr>;

impl<K: Eq + Hash + Copy, V: SizeHint> ShardCache<K, V> {
    /// Build a shared cache with a count cap (`cache_shards`, 0 = no cache) and
    /// byte budget. The budget governs *all* readers sharing this cache.
    pub fn new(cache_shards: usize, bytes_budget: usize) -> Arc<Self> {
        Arc::new(Self::inline(cache_shards, bytes_budget))
    }

    /// Same cache, un-`Arc`ed, for the readers that own theirs outright rather
    /// than sharing it across a multi-file run.
    pub(super) fn inline(cache_shards: usize, bytes_budget: usize) -> Self {
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
        ShardCache {
            cache,
            in_flight,
            metrics: OnceLock::new(),
            cache_shards,
        }
    }

    pub(super) fn has_cache(&self) -> bool {
        self.cache.is_some()
    }

    pub(super) fn contains_key(&self, key: K) -> bool {
        match &self.cache {
            Some(m) => m.lock().unwrap().contains(&key),
            None => false,
        }
    }

    fn in_flight_contains_key(&self, key: K) -> bool {
        match &self.in_flight {
            Some(m) => m.lock().unwrap().contains_key(&key),
            None => false,
        }
    }

    pub(super) fn capacity(&self) -> usize {
        match &self.cache {
            Some(m) => m.lock().unwrap().capacity(),
            None => 0,
        }
    }

    pub(super) fn reserve_for(&self, min_shards: usize, max_cache_bytes: usize) -> usize {
        match &self.cache {
            Some(m) => {
                let mut cache = m.lock().unwrap();
                cache.reserve_for(min_shards, max_cache_bytes);
                cache.capacity()
            }
            None => 0,
        }
    }

    pub(super) fn metrics(&self) -> Option<&Arc<CacheMetrics>> {
        self.metrics.get()
    }

    /// Enable cache-behavior metrics. Idempotent: the first call installs the
    /// counters (also wiring them into the LRU for eviction/byte stats);
    /// later calls return the same handle.
    pub(super) fn enable_metrics(&self) -> Arc<CacheMetrics> {
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

    /// Return the cached value for `key`, or run `decode` exactly once across
    /// concurrent callers (singleflight) and cache the result under the budget.
    /// `decode` produces the decoded shard; the caller (the reader) owns the
    /// decode because it is reader/mmap-specific.
    ///
    /// This is the *only* place that takes both locks, and it takes them
    /// `in_flight` → `cache`. That used to be a contract three transcriptions
    /// had to honour independently.
    pub(super) fn get_or_decode(
        &self,
        key: K,
        decode: impl FnOnce() -> Result<Arc<V>>,
    ) -> Result<Arc<V>> {
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
            let _guard: Option<LeaderGuard<K>> = match &self.in_flight {
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
            let value = decode()?;
            if let Some(ref cache_mutex) = self.cache {
                let mut cache = cache_mutex.lock().unwrap();
                cache.put_with_budget(key, Arc::clone(&value));
            }
            return Ok(value);
        }
    }
}

/// The `(file_id, shard_id)`-keyed helpers, which only the CSR reader has a use
/// for: it is the one instantiation whose key has two components, because it is
/// the one that can be shared across the readers of a multi-file run.
impl SharedShardCache {
    pub(super) fn contains(&self, fid: u32, shard: usize) -> bool {
        self.contains_key((fid, shard))
    }

    pub(super) fn in_flight_contains(&self, fid: u32, shard: usize) -> bool {
        self.in_flight_contains_key((fid, shard))
    }

    /// Dedup `shard_indices` (local ids) to those neither cached nor in flight
    /// for `fid`. Lock order `in_flight` → `cache` matches
    /// [`ShardCache::get_or_decode`] so concurrent callers can't deadlock.
    pub(super) fn filter_misses(&self, fid: u32, shard_indices: &[usize]) -> Vec<usize> {
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
}
