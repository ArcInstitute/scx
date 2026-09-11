//! The decoded-shard cache: one byte-budgeted LRU and one singleflight,
//! shared by all three readers.
//!
//! There used to be three of each. [`WeightedLruCache`] is generic over the
//! payload via [`SizeHint`], and [`ShardCache::get_or_decode`] is the only
//! place that takes both the `in_flight` and `cache` locks — in that order,
//! which is what makes the lock ordering a property of the code rather than a
//! contract three transcriptions had to honour separately.
//!
//! The CSR instantiation ([`SharedShardCache`]) holds **two kinds of entry**
//! under one budget and one LRU order: whole decoded shards
//! ([`CacheKey::Shard`]) and decoded **row groups** of a framed shard
//! ([`CacheKey::Group`]). A scattered gather over a framed file never decodes a
//! whole shard, so before the row-group entries existed that path could not
//! populate the cache at all and re-decoded the same groups every batch
//! (OPT-FORMATIO-1). The `cache_shards` count cap applies to whole shards only
//! — a row group is 1/64 of a shard at the default geometry, and the byte
//! budget is what bounds them. Each kind reports through its own set of
//! counters on [`CacheMetrics`], so `hits`/`misses` keep meaning "whole-shard
//! LRU" for every caller that read them before.

use super::*;

// ---------------------------------------------------------------------------
// Cache instrumentation: metrics + singleflight + byte-budgeted eviction
// ---------------------------------------------------------------------------

/// Which kind of entry a cache key names. Selects the counter set a hit,
/// miss, eviction or insert is attributed to — see [`CacheMetrics::counters`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheKind {
    /// A whole decoded shard (CSR, CSC, or dense). Bounded by the
    /// `cache_shards` count cap **and** the byte budget.
    Shard,
    /// One decoded row group of a row-group-framed CSR shard. Bounded by the
    /// byte budget only.
    RowGroup,
}

/// Key of the CSR instantiation. `Shard` is the pre-existing `(file_id,
/// shard_idx)` key; `Group` adds the row-group index within the shard's
/// block index. `file_id` is what lets several readers of a multi-file run
/// share one cache without colliding on shard `0`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CacheKey {
    Shard(u32, usize),
    Group(u32, usize, usize),
}

/// A cache key that knows which [`CacheKind`] it names. Blanket-free on
/// purpose: the CSC and dense readers key by plain `shard_idx`, and every
/// one of their entries is a whole shard.
pub trait CacheKeyKind: Eq + Hash + Copy {
    fn kind(&self) -> CacheKind;
}

impl CacheKeyKind for usize {
    fn kind(&self) -> CacheKind {
        CacheKind::Shard
    }
}

impl CacheKeyKind for CacheKey {
    fn kind(&self) -> CacheKind {
        match self {
            CacheKey::Shard(..) => CacheKind::Shard,
            CacheKey::Group(..) => CacheKind::RowGroup,
        }
    }
}

/// Atomic counters for shard-cache behavior. Opt-in via
/// [`BackedCsrReader::enable_metrics`]; cloning the returned `Arc` lets a
/// caller (e.g. `IndexPlanIter`) sample without touching the cache lock.
///
/// All counters use `Ordering::Relaxed` — the values are statistical and not
/// used for synchronization.
///
/// Two counter sets share the struct. `hits` … `duplicate_waiters` describe the
/// **whole-shard** entries; `row_group_*` describe the decoded **row-group**
/// entries a framed scattered read retains. `peak_bytes_in_cache` is the one
/// gauge over both, since they share the budget — so `peak ≤ bytes_inserted +
/// row_group_bytes_inserted`, not `≤ bytes_inserted`.
#[derive(Default, Debug)]
pub struct CacheMetrics {
    /// `read_shard_cached_arc` calls served from the LRU without decode.
    pub hits: AtomicU64,
    /// Calls that fell through to decode (became leader of a singleflight slot).
    pub misses: AtomicU64,
    /// Whole-shard entries dropped by `WeightedLruCache::put_with_budget` to
    /// fit a new entry (count or byte cap; both share this counter).
    pub evictions: AtomicU64,
    /// Cumulative whole-shard bytes inserted into the cache (estimated decoded
    /// size).
    pub bytes_inserted: AtomicU64,
    /// Calls that found a peer leader already decoding the same shard and
    /// waited on its Condvar instead of redecoding.
    pub duplicate_waiters: AtomicU64,
    /// High-water mark of `WeightedLruCache.bytes_used` — whole shards **and**
    /// row groups together — since [`BackedCsrReader::enable_metrics`].
    /// Maintained via `fetch_max` on every successful `put_with_budget`. Lets
    /// callers see whether the byte cap was actually exercised, vs. just
    /// configured generously.
    pub peak_bytes_in_cache: AtomicU64,
    /// `read_rows_with` shard request-groups served by a full-shard decode.
    /// Covers the planned `!use_block_index` case (cached/dense group) and the
    /// `use_block_index` group that fell back because the shard was unframed.
    pub full_shard_groups: AtomicU64,
    /// `read_rows_with` shard request-groups served by the codec-agnostic
    /// **row-group block-index** path (F5 Phase 1) — a framed (v2) shard decoded
    /// only in its touched groups, whether those groups were decoded on this
    /// call or served from the row-group LRU. The block-index adoption signal,
    /// symmetric with `full_shard_groups`.
    pub block_index_groups: AtomicU64,
    /// Row-group lookups served from the LRU without decode.
    pub row_group_hits: AtomicU64,
    /// Row-group lookups that decoded (singleflight leader).
    pub row_group_misses: AtomicU64,
    /// Row-group entries dropped to fit a new entry under the byte budget.
    pub row_group_evictions: AtomicU64,
    /// Cumulative row-group bytes inserted.
    pub row_group_bytes_inserted: AtomicU64,
    /// Row-group lookups that waited on a peer's in-flight decode.
    pub row_group_duplicate_waiters: AtomicU64,
}

/// The per-kind view of [`CacheMetrics`] the cache bumps through, so the six
/// sites that count never spell the kind dispatch themselves.
pub(super) struct CounterRefs<'a> {
    pub(super) hits: &'a AtomicU64,
    pub(super) misses: &'a AtomicU64,
    pub(super) evictions: &'a AtomicU64,
    pub(super) bytes_inserted: &'a AtomicU64,
    pub(super) duplicate_waiters: &'a AtomicU64,
}

impl CacheMetrics {
    /// The counter set for entries of `kind`.
    pub(super) fn counters(&self, kind: CacheKind) -> CounterRefs<'_> {
        match kind {
            CacheKind::Shard => CounterRefs {
                hits: &self.hits,
                misses: &self.misses,
                evictions: &self.evictions,
                bytes_inserted: &self.bytes_inserted,
                duplicate_waiters: &self.duplicate_waiters,
            },
            CacheKind::RowGroup => CounterRefs {
                hits: &self.row_group_hits,
                misses: &self.row_group_misses,
                evictions: &self.row_group_evictions,
                bytes_inserted: &self.row_group_bytes_inserted,
                duplicate_waiters: &self.row_group_duplicate_waiters,
            },
        }
    }
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
///
/// `pub(crate)`: the row-group path sizes a group from its `RowGroupSpan`
/// (`n_rows + 1`, `nnz`, `nnz`) *before* decoding it, and must agree with what
/// the cache will charge once it is decoded.
pub(crate) fn csr_component_bytes(indptr: usize, indices: usize, data: usize) -> usize {
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

/// LRU cache with a whole-shard count cap and a byte cap over every entry.
/// Evicts oldest entries until both caps are satisfied for a new insertion.
///
/// Generic over the decoded payload: `ScxCsr` for X/layer shards and their
/// row groups, `ScxCsc` for the gene-major sidecar, `DenseShard` for an `obsm`
/// embedding. Bytes come from [`SizeHint`], which is the only thing that
/// differed between the three hand-rolled caches this replaced.
///
/// The inner `LruCache` is **unbounded**: the count cap is enforced here, on
/// [`CacheKind::Shard`] entries only, by evicting the least-recently-used
/// *shard* (not the LRU entry of any kind) when a new shard would exceed
/// `shard_cap`. Row groups are bounded by bytes alone. `K: Copy` is
/// load-bearing: keys are copied out of an iterator borrow before the entry
/// they name is popped.
pub(super) struct WeightedLruCache<K: CacheKeyKind, V: SizeHint> {
    inner: LruCache<K, CacheEntry<V>>,
    /// Count cap on [`CacheKind::Shard`] entries (`cache_shards`, ≥ 1).
    shard_cap: usize,
    /// Live number of [`CacheKind::Shard`] entries in `inner`.
    n_shard_entries: usize,
    /// Hard byte cap over every entry. `usize::MAX` means count-only behavior
    /// (compatible with `BackedCsrReader::new`).
    bytes_budget: usize,
    /// Cumulative bytes currently in `inner`.
    bytes_used: usize,
    /// Optional metrics handle (cloned from the owning reader on construction).
    pub(super) metrics: Option<Arc<CacheMetrics>>,
}

impl<K: CacheKeyKind, V: SizeHint> WeightedLruCache<K, V> {
    pub(super) fn new(cache_shards: usize, bytes_budget: usize) -> Self {
        // Clamp to ≥1: `cache_shards` flows from user-facing Python constructors.
        // A 1-shard cache is the minimum sensible budget (the byte budget still
        // bounds memory independently). `LruCache::unbounded` allocates
        // nothing up front — `LruCache::new(cap)` preallocated `cap` buckets.
        WeightedLruCache {
            inner: LruCache::unbounded(),
            shard_cap: cache_shards.max(1),
            n_shard_entries: 0,
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

    /// Account for an entry that has just left `inner`.
    fn note_evicted(&mut self, key: K, entry: CacheEntry<V>) {
        self.bytes_used = self.bytes_used.saturating_sub(entry.bytes);
        if key.kind() == CacheKind::Shard {
            self.n_shard_entries = self.n_shard_entries.saturating_sub(1);
        }
        if let Some(m) = &self.metrics {
            m.counters(key.kind())
                .evictions
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Pop least-recently-used entries of any kind until `bytes_used +
    /// incoming` fits the budget or the cache is empty.
    fn evict_bytes_for(&mut self, incoming: usize) {
        // `saturating_add` keeps the comparison sound even if a degenerate
        // decoded shard pushes the sum past `usize`.
        while self.bytes_used.saturating_add(incoming) > self.bytes_budget && !self.inner.is_empty()
        {
            match self.inner.pop_lru() {
                Some((k, e)) => self.note_evicted(k, e),
                None => break,
            }
        }
    }

    /// Pop the least-recently-used **shard** entry (row groups are skipped over
    /// and keep their recency). `false` when no shard entry is resident.
    fn evict_oldest_shard(&mut self) -> bool {
        // `iter()` walks MRU → LRU; `rev()` gives the oldest first.
        let victim = self
            .inner
            .iter()
            .rev()
            .find(|(k, _)| k.kind() == CacheKind::Shard)
            .map(|(k, _)| *k);
        match victim {
            Some(k) => match self.inner.pop_entry(&k) {
                Some((k, e)) => {
                    self.note_evicted(k, e);
                    true
                }
                None => false,
            },
            None => false,
        }
    }

    /// Insert `value` under `key`, evicting oldest entries until both the
    /// whole-shard count cap and the byte cap are satisfied. If a new entry on
    /// its own exceeds `bytes_budget`, all other entries are evicted and the
    /// new one is still inserted (the alternative — refusing to cache — would
    /// defeat the cache for any outsized shard).
    pub(super) fn put_with_budget(&mut self, key: K, value: Arc<V>) {
        let bytes = value.size_bytes();
        let kind = key.kind();

        // Evict by byte budget first, oldest entry of either kind.
        self.evict_bytes_for(bytes);

        // Then the whole-shard count cap — only a *new* shard key grows the
        // count; a same-key replacement does not.
        if kind == CacheKind::Shard && !self.inner.contains(&key) {
            while self.n_shard_entries >= self.shard_cap {
                if !self.evict_oldest_shard() {
                    break;
                }
            }
        }

        // The inner cache is unbounded, so `push` returns `Some` only for a
        // same-key *replacement*; a returned key != the inserted key would be
        // a genuine eviction, kept accounted for defensively.
        let entry = CacheEntry { value, bytes };
        match self.inner.push(key, entry) {
            Some((displaced_key, displaced)) if displaced_key != key => {
                self.note_evicted(displaced_key, displaced);
                if kind == CacheKind::Shard {
                    self.n_shard_entries += 1;
                }
            }
            Some((_, displaced)) => {
                // Replacement: the old bytes leave, the shard count is unchanged.
                self.bytes_used = self.bytes_used.saturating_sub(displaced.bytes);
            }
            None => {
                if kind == CacheKind::Shard {
                    self.n_shard_entries += 1;
                }
            }
        }
        self.bytes_used = self.bytes_used.saturating_add(bytes);
        if let Some(m) = &self.metrics {
            m.counters(kind)
                .bytes_inserted
                .fetch_add(bytes as u64, Ordering::Relaxed);
            // High-water gauge: record the post-insert level so callers can
            // tell whether the byte cap was actually exercised. `fetch_max`
            // is monotonic so concurrent inserts converge correctly even
            // without a lock around the load.
            m.peak_bytes_in_cache
                .fetch_max(self.bytes_used as u64, Ordering::Relaxed);
        }
    }

    /// Current whole-shard count cap.
    pub(super) fn capacity(&self) -> usize {
        self.shard_cap
    }

    /// Bytes currently resident, both kinds.
    pub(super) fn bytes_used(&self) -> usize {
        self.bytes_used
    }

    /// The byte budget (`usize::MAX` = count-only).
    pub(super) fn bytes_budget(&self) -> usize {
        self.bytes_budget
    }

    /// Reserve the cache for a multi-pass op: raise the whole-shard count cap
    /// to `min_cap` (never shrink) and set the byte budget to `byte_budget` —
    /// the op's authoritative RAM ceiling — evicting LRU entries to fit.
    /// Setting (rather than only raising) the byte budget is deliberate: the
    /// common count-only-opened reader starts at `usize::MAX` (unbounded), and
    /// an out-of-core matrix must stay bounded, so the op's budget governs.
    fn reserve_for(&mut self, min_cap: usize, byte_budget: usize) {
        if min_cap > self.shard_cap {
            self.shard_cap = min_cap;
        }
        self.bytes_budget = byte_budget;
        self.evict_bytes_for(0);
    }
}

// ---------------------------------------------------------------------------
// ShardCache
// ---------------------------------------------------------------------------

/// Singleflight table: `key → in-flight decode slot`.
type InFlightTable<K> = Mutex<HashMap<K, Arc<InFlightSlot>>>;

/// Decoded-shard cache + singleflight table.
///
/// The CSR instantiation is keyed by [`CacheKey`] — `(file_id, shard_id)` for
/// a whole shard, plus the row-group index for a decoded row group — so it
/// can be **shared across several `BackedCsrReader`s** that together back one
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
pub struct ShardCache<K: CacheKeyKind, V: SizeHint> {
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
/// change. Nothing outside this module spells the key type.
pub type SharedShardCache = ShardCache<CacheKey, ScxCsr>;

impl<K: CacheKeyKind, V: SizeHint> ShardCache<K, V> {
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

    /// Bytes currently resident (both kinds); `0` without a cache.
    pub(super) fn bytes_used(&self) -> usize {
        match &self.cache {
            Some(m) => m.lock().unwrap().bytes_used(),
            None => 0,
        }
    }

    /// The live byte budget. `usize::MAX` for a count-only cache, `0` when no
    /// cache is installed.
    pub(super) fn bytes_budget(&self) -> usize {
        match &self.cache {
            Some(m) => m.lock().unwrap().bytes_budget(),
            None => 0,
        }
    }

    /// Whether this cache admits [`CacheKind::RowGroup`] entries: it must
    /// exist **and** have a finite byte budget. A count-only cache has no
    /// bound a row group would respect (the count cap is per whole shard), so
    /// it stays whole-shard-only rather than growing without limit.
    pub(super) fn caches_groups(&self) -> bool {
        match &self.cache {
            Some(m) => m.lock().unwrap().bytes_budget() != usize::MAX,
            None => false,
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

    /// The cached value for `key`, if resident — a hit (counted, recency
    /// touched), never a decode. The bypass half of the CSR reader's row-group
    /// admission: a gather whose working set does not fit the budget still
    /// serves whatever is already resident, then decodes the rest uncached
    /// (see [`Self::note_uncached_miss`]) instead of inserting and evicting.
    pub(super) fn get_cached(&self, key: K) -> Option<Arc<V>> {
        let cache_mutex = self.cache.as_ref()?;
        let hit = cache_mutex.lock().unwrap().get(&key);
        if hit.is_some() {
            if let Some(m) = self.metrics.get() {
                m.counters(key.kind()).hits.fetch_add(1, Ordering::Relaxed);
            }
        }
        hit
    }

    /// Count a decode the caller ran **outside** the cache — neither inserted
    /// nor singleflighted — so a working set that is over budget still reads as
    /// `misses` growing while `bytes_inserted` stays flat.
    pub(super) fn note_uncached_miss(&self, kind: CacheKind) {
        if let Some(m) = self.metrics.get() {
            m.counters(kind).misses.fetch_add(1, Ordering::Relaxed);
        }
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
        let kind = key.kind();
        loop {
            // Cache hit fast path.
            if let Some(ref cache_mutex) = self.cache {
                let mut cache = cache_mutex.lock().unwrap();
                if let Some(cached) = cache.get(&key) {
                    if let Some(m) = self.metrics.get() {
                        m.counters(kind).hits.fetch_add(1, Ordering::Relaxed);
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
                            m.counters(kind)
                                .duplicate_waiters
                                .fetch_add(1, Ordering::Relaxed);
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
                                m.counters(kind).hits.fetch_add(1, Ordering::Relaxed);
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
                m.counters(kind).misses.fetch_add(1, Ordering::Relaxed);
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
/// for: it is the one instantiation whose key has a file component, because it
/// is the one that can be shared across the readers of a multi-file run. All
/// of them address **whole-shard** entries; row groups have no membership
/// query — their consumers go straight through [`ShardCache::get_or_decode`].
impl SharedShardCache {
    pub(super) fn contains(&self, fid: u32, shard: usize) -> bool {
        self.contains_key(CacheKey::Shard(fid, shard))
    }

    pub(super) fn in_flight_contains(&self, fid: u32, shard: usize) -> bool {
        self.in_flight_contains_key(CacheKey::Shard(fid, shard))
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
            let key = CacheKey::Shard(fid, idx);
            if cache.contains(&key) || in_flight.contains_key(&key) {
                continue;
            }
            misses.push(idx);
        }
        misses
    }
}
