//! A bounded, lazily-opened `file_id → reader` registry for large manifests.
//!
//! # What this bounds, and what it does not
//!
//! [`crate::sparse_cellset::SparseCellSetLoader`] is handed a manifest of file
//! paths and, before this module, opened every one of them in its constructor
//! and held them all for the dataset's lifetime. Consumer manifests reach tens
//! of thousands of files, so the natural worry is file descriptors — and that
//! worry is **wrong**, which is worth stating here because it is the first
//! thing a reader of this module will assume.
//!
//! `ScxReader::open` mmaps the file and lets the `File` drop, so an unwatched
//! reader holds **zero descriptors and one mapping**; `scx-loader` never calls
//! `ScxReader::watching`, so every reader on this path is unwatched. Measured
//! on the real constructor: a 5,000-file manifest constructs and gathers under
//! `ulimit -n 1024` with the process's descriptor count flat.
//!
//! What an open reader does cost is **resident memory**, and almost all of it
//! is the parsed `FullCatalog` — one owned entry per catalog entry, with an
//! owned name and stats. Measured per open reader:
//!
//! ```text
//! ScxReader alone                       91.8 kB
//! + BackedCsrReader wrapper            104.4 kB   (tabula_sapiens_100k)
//!                                      120.6 kB   (census_1m)
//! ```
//!
//! At 26,453 files — a real consumer manifest size — that is ~2.8-3.2 GB per
//! process, before multiplying by DataLoader workers and ranks. `reader_limit`
//! brings it down 66-76x; the capture is
//! `benchmarks/scripts/measure_reader_registry_rss.py` and its rows are under
//! `results/raw/phase2_reader_registry/`.
//!
//! The consequence for this module is counter-intuitive and load-bearing:
//! **eviction must drop the catalog, and a reopen must re-parse it.** The
//! obvious optimisation is to retain `Arc<FullCatalog>` across an eviction and
//! reopen through `ScxReader::open_with_shared_catalog`, which exists precisely
//! to skip that parse — but the catalog is 91% of what we are trying to
//! reclaim, so retaining it would make a reopen cheap and the feature
//! pointless. What a vacated slot keeps is the path, the row counts, the shard
//! index, and the file's identity.
//!
//! # Why the registry is its own `Arc` and never reaches the engine
//!
//! [`crate::plan_engine::PrefetchEngine`] owns the tokio runtime, and
//! `crate::runtime` documents two deterministic failures caused by a spawned
//! task holding an owner further up: an already-started `spawn_blocking` cannot
//! be aborted, so such a task can release the final reference and drop the
//! runtime from one of its own threads. `spawn_prefetches` therefore clones the
//! reader `Arc` *outside* the closure. The same rule applies here: a spawned
//! closure may capture a **lease**, never the registry and never the engine.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use scx_format_io::freshness::FileIdentity;
use scx_format_io::{BackedCsrIndex, BackedCsrReader, ScxReader, SharedShardCache};

use crate::error::{LoaderError, Result};

/// Registry counters, reported through `SparseCellSetDataset.cache_metrics()`.
///
/// Separate from `scx_format_io::CacheMetrics`, which counts the decoded-shard
/// cache: that struct is shared by five Python surfaces including the
/// single-file `IndexPlanDataset`, and putting reader counters on it would give
/// that class four keys that are structurally always zero.
#[derive(Debug, Default)]
pub struct ReaderMetrics {
    /// `ScxReader::open` calls **the registry** has made: the handles the
    /// manifest scan handed it, plus every reopen since.
    ///
    /// Deliberately not the constructor's opens. The scan opens every file
    /// once whatever the limit — `n_vars`, the catalog shard stats and the
    /// CSR-range validation all come from the catalog — and releases the ones
    /// it is not keeping, so counting those here would report a fixed
    /// `n_files` on every dataset and say nothing. What this answers is the
    /// question a caller tuning `reader_limit` actually has: how much
    /// *reopening* the steady state is costing. At `reader_limit = None` it is
    /// the manifest size and never moves again.
    pub opens: AtomicU64,
    /// Handles closed to stay within `reader_limit`.
    pub evictions: AtomicU64,
    /// Currently-open handles.
    pub resident: AtomicU64,
    /// The largest `resident` ever reached. This is the number to check against
    /// `reader_limit`, which bounds *idle-retainable* handles rather than live
    /// ones — see [`ReaderRegistry::lease`].
    pub hwm: AtomicU64,
}

impl ReaderMetrics {
    fn note_open(&self) {
        self.opens.fetch_add(1, Ordering::Relaxed);
        let now = self.resident.fetch_add(1, Ordering::Relaxed) + 1;
        self.hwm.fetch_max(now, Ordering::Relaxed);
    }

    fn note_evict(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
        self.resident.fetch_sub(1, Ordering::Relaxed);
    }
}

/// What survives a handle eviction: everything needed to plan against the file
/// and to reopen it, and nothing that costs per-file kilobytes.
///
/// Built by the constructor's manifest scan and handed here as-is. There used
/// to be a second, field-identical `ScannedFile` that `from_scan` repacked into
/// this one — a round-trip that could not diverge without becoming a bug.
/// **Review on #536 (Cursor Agent, Antigravity).**
pub(crate) struct FileSlot {
    pub(crate) path: PathBuf,
    pub(crate) n_obs: u64,
    /// The shard index, retained for **every** file whatever the limit.
    ///
    /// The first version retained it only when the registry could evict, on the
    /// reasoning that a permanently-resident handle answers index queries
    /// itself. That was false economy: the fallback path takes the registry
    /// mutex, and `bucket_plan_rows` asks `shard_for_row` **once per row**, so
    /// the default path would have acquired a lock per row of every plan to
    /// save `n_shards x 24 B` per file — about 1.5 kB on census_1m's 62 shards,
    /// against the ~121 kB the handle itself costs.
    pub(crate) index: BackedCsrIndex,
    /// Stamped at first open, compared on every reopen.
    ///
    /// `Some` for every slot of a registry that holds a reopen recipe — which
    /// is every path-built loader, limited or not — and `None` only for
    /// [`ReaderRegistry::from_open`], which has no recipe and so cannot reopen
    /// at all. An earlier version stamped only when a limit was set, on the
    /// reasoning that an unlimited registry never evicts and so never reopens.
    /// That was true and still a hole: it left the check depending on *when*
    /// eviction happens to run, and `lease` now refuses a reopen it cannot
    /// verify rather than falling through. Do not "save the stat" here.
    pub(crate) identity: Option<FileIdentity>,
}

/// One currently-open handle, with the tick at which it was last leased.
struct OpenHandle {
    reader: Arc<BackedCsrReader>,
    last_used: u64,
}

struct OpenHandles {
    by_file: HashMap<u32, OpenHandle>,
    /// Monotonic recency counter. A counter rather than an intrusive list
    /// because eviction is O(open handles) at most once per reopen, and a
    /// reopen costs an `ScxReader::open` (0.09–20 ms measured per file) that
    /// dwarfs a scan of at most `reader_limit` integers.
    tick: u64,
}

/// How to rebuild a handle for a `file_id` after its slot was vacated.
///
/// `PrefetchEngine::from_scx_readers` applies three `&mut` setters between
/// `BackedCsrReader::with_shared_cache` and the `Arc` — the block-index gate,
/// the shared rayon pool, and metrics — because `BackedCsrReader` offers no
/// interior mutability for them and that window is the only one there is. A
/// reopen must replay all three; a second copy of that sequence is how a
/// reopened reader silently loses its route or stops counting.
struct OpenRecipe {
    shared: Arc<SharedShardCache>,
    scatter_block_index: bool,
}

/// Attach a freshly opened `ScxReader` to the shared cache as `file_id`.
///
/// **The one definition of the pre-`Arc` sequence.** `BackedCsrReader` exposes
/// no interior mutability for the block-index gate, the shared rayon pool or
/// metrics, so the window between `with_shared_cache` and the `Arc` is the only
/// place they can be set. Every caller that produces a reader for this engine
/// goes through here — the constructor's manifest scan and the registry's
/// reopen alike — because a second copy of these four lines is precisely how a
/// reopened reader would silently lose its route or stop counting.
pub(crate) fn wrap_reader(
    reader: ScxReader,
    file_id: u32,
    shared: &Arc<SharedShardCache>,
    scatter_block_index: bool,
) -> Arc<BackedCsrReader> {
    let mut backed = BackedCsrReader::with_shared_cache(reader, file_id, Arc::clone(shared));
    backed.set_scatter_block_index(scatter_block_index);
    // One pool for every reader, not one each: `cpu_pool()` is process-wide,
    // so an N-file manifest does not spawn N pools. Same fork rationale as
    // `IndexPlanLoader` — see `crate::pool`.
    backed.set_cpu_pool(crate::pool::cpu_pool());
    // Idempotent on the shared cache, so doing it per reader installs one
    // aggregate handle rather than N.
    backed.enable_metrics();
    Arc::new(backed)
}

/// A `file_id → reader` map that keeps at most `limit` readers resident.
///
/// `file_id` is the manifest position and is **never** reassigned to a
/// different path for the registry's lifetime. That is not a style rule: the
/// decoded-shard cache is keyed `CacheKey::Shard(file_id, shard)` /
/// `CacheKey::Group(file_id, shard, group)` with no path and no generation
/// counter, and those entries outlive a handle eviction in the shared cache —
/// so a `file_id` pointed at a second file would serve the first file's
/// decoded bytes as hits.
pub(crate) struct ReaderRegistry {
    slots: Vec<FileSlot>,
    open: Mutex<OpenHandles>,
    /// `None` = open everything and never evict, which is today's behaviour and
    /// the default.
    limit: Option<usize>,
    /// `None` when the registry was seeded from already-open readers and can
    /// therefore never need to reopen.
    recipe: Option<OpenRecipe>,
    metrics: Arc<ReaderMetrics>,
    /// Memoized answer to "does any file here have a framed CSR shard".
    any_framed: OnceLock<bool>,
}

impl ReaderRegistry {
    /// Registry over readers that are already open, retaining all of them.
    ///
    /// Used by `PrefetchEngine::new` — `IndexPlanLoader` (one file) and the
    /// `SparseCellSetLoader::new(Vec<ScxReader>, …)` entry point. Nothing is
    /// ever evicted, so no slot carries an identity — there is no reopen for it
    /// to check. The shard index is cloned out of each reader all the same, so
    /// the per-row bucketing path is lock-free here too.
    pub(crate) fn from_open(readers: Vec<Arc<BackedCsrReader>>) -> Arc<Self> {
        let metrics = Arc::new(ReaderMetrics::default());
        let mut by_file = HashMap::with_capacity(readers.len());
        let mut slots = Vec::with_capacity(readers.len());
        for (fid, reader) in readers.into_iter().enumerate() {
            slots.push(FileSlot {
                path: reader.path().to_path_buf(),
                n_obs: reader.n_obs() as u64,
                index: reader.index().clone(),
                identity: None,
            });
            metrics.note_open();
            by_file.insert(
                fid as u32,
                OpenHandle {
                    reader,
                    last_used: fid as u64,
                },
            );
        }
        let tick = slots.len() as u64;
        Arc::new(Self {
            slots,
            open: Mutex::new(OpenHandles { by_file, tick }),
            limit: None,
            recipe: None,
            metrics,
            any_framed: OnceLock::new(),
        })
    }

    /// Registry built from a manifest scan.
    ///
    /// `retained` are the handles the scan chose to keep — at most `limit`, or
    /// all of them when `limit` is `None`, which is the default and is today's
    /// behaviour: nothing is ever evicted and nothing is ever reopened.
    pub(crate) fn from_scan(
        slots: Vec<FileSlot>,
        retained: Vec<(u32, Arc<BackedCsrReader>)>,
        limit: Option<usize>,
        shared: Arc<SharedShardCache>,
        scatter_block_index: bool,
    ) -> Arc<Self> {
        let metrics = Arc::new(ReaderMetrics::default());
        let mut by_file = HashMap::with_capacity(retained.len());
        let mut tick = 0u64;
        for (fid, reader) in retained {
            metrics.note_open();
            by_file.insert(
                fid,
                OpenHandle {
                    reader,
                    last_used: tick,
                },
            );
            tick += 1;
        }
        Arc::new(Self {
            slots,
            open: Mutex::new(OpenHandles { by_file, tick }),
            limit,
            recipe: Some(OpenRecipe {
                shared,
                scatter_block_index,
            }),
            metrics,
            any_framed: OnceLock::new(),
        })
    }

    pub(crate) fn n_files(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn metrics(&self) -> Arc<ReaderMetrics> {
        Arc::clone(&self.metrics)
    }

    /// An owned handle, valid until the caller drops it.
    ///
    /// Owned rather than borrowed because eviction is the whole point and a
    /// borrow could not survive it: `read_rows_with_admission` holds its
    /// receiver across a parallel shard decode and can block on a peer thread's
    /// single-flight decode, and the prefetcher's `spawn_blocking` tasks are
    /// unabortable once started and outlive the iterator that spawned them.
    /// Unmapping a file under either of those is a use-after-free of the mmap.
    pub(crate) fn lease(&self, file_id: u32) -> Result<Arc<BackedCsrReader>> {
        let idx = file_id as usize;
        if idx >= self.slots.len() {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "plan file_id {file_id} out of range (n_files={})",
                    self.slots.len()
                ),
            });
        }
        let mut open = self.open.lock().expect("reader registry mutex poisoned");
        open.tick += 1;
        let tick = open.tick;
        if let Some(h) = open.by_file.get_mut(&file_id) {
            let hit = Arc::clone(&h.reader);
            h.last_used = tick;
            // Trim on a hit too, not only on a miss. Evicting only on a miss
            // makes residency a ratchet: a plan wide enough to pin more handles
            // than the limit leaves every one of them resident for as long as
            // later leases happen to hit, so `reader_limit` would describe the
            // steady state of some manifests and not others. It costs nothing
            // when already within the cap — `evict_down_to` returns on its
            // first length test. Caught by `a_leased_handle_is_never_evicted`.
            self.trim(&mut open);
            return Ok(hit);
        }
        let recipe = self
            .recipe
            .as_ref()
            .ok_or_else(|| LoaderError::ConfigError {
                reason: format!(
                    "reader for file_id {file_id} is not open and this registry cannot reopen \
                 (it was seeded from already-open readers)"
                ),
            })?;
        // Make room *before* opening, so the peak is `limit` rather than
        // `limit + 1`: an open that pushed over the cap and then trimmed back
        // would still have had the extra catalog resident, and residency is
        // exactly what this bounds. When every resident handle is leased this
        // evicts nothing and the open goes over the cap anyway — the soft-cap
        // case, which `ReaderMetrics::hwm` is there to make visible.
        self.evict_down_to(
            &mut open,
            self.limit.unwrap_or(usize::MAX).saturating_sub(1),
        );

        // Opened with the registry lock held, deliberately. It serialises
        // concurrent reopens, which is the point: two threads missing on the
        // same `file_id` would otherwise both parse its catalog and one of the
        // two would be thrown away. Nothing reached from here re-enters the
        // registry — `wrap_reader` only touches the reader and the process-wide
        // rayon pool — so there is no lock-order cycle to worry about. The
        // contention this costs is nil on today's callers: the gather leases on
        // one consumer thread, and prefetch tasks are handed their `Arc`s
        // before they are spawned rather than leasing for themselves.
        let slot = &self.slots[idx];
        let reader = ScxReader::open(&slot.path).map_err(|e| LoaderError::ConfigError {
            reason: format!("failed to reopen {}: {e}", slot.path.display()),
        })?;
        // The reopen landed on *a* file at that path; make it prove it is the
        // same one. Deliberately not `ScxReader::watching()`, which would
        // retain a descriptor per reader for the reader's whole life and add a
        // `stat` + `pread` to every section read — a behaviour change on the
        // default path, to answer a question only a reopen asks.
        //
        // A missing stamp is an error, not a skipped check. Today it is
        // unreachable (a registry with a recipe is built by the manifest scan,
        // which stamps every slot it can reopen), and the point of refusing
        // rather than falling through is to keep it unreachable: a future
        // constructor that forgot to stamp would otherwise serve a replaced
        // file silently, which is the exact failure this check exists for.
        let stamped = slot
            .identity
            .as_ref()
            .ok_or_else(|| LoaderError::ConfigError {
                reason: format!(
                    "cannot reopen {} — this registry can reopen but slot {file_id} carries no \
                     identity to verify the file against",
                    slot.path.display()
                ),
            })?;
        let now = FileIdentity::stamp(&slot.path, reader.header())?;
        stamped.ensure_same(&slot.path, &now)?;
        let wrapped = wrap_reader(reader, file_id, &recipe.shared, recipe.scatter_block_index);
        self.metrics.note_open();
        open.by_file.insert(
            file_id,
            OpenHandle {
                reader: Arc::clone(&wrapped),
                last_used: tick,
            },
        );
        Ok(wrapped)
    }

    /// Evict down to `self.limit`; a no-op when there is no limit.
    fn trim(&self, open: &mut OpenHandles) {
        if let Some(limit) = self.limit {
            self.evict_down_to(open, limit);
        }
    }

    /// Close least-recently-leased handles until at most `target` remain.
    ///
    /// Skips any handle with an outstanding lease. `Arc::strong_count == 1`
    /// means the registry's own entry is the only owner, and reading it under
    /// this lock is sound in the direction that matters: a lease can only be
    /// *obtained* through `lease`, which takes the same lock, so while it is
    /// held the count can fall but never rise. A count that is stale-high
    /// therefore costs a missed eviction, never a premature one.
    ///
    /// The consequence is that `reader_limit` bounds handles the registry is
    /// free to drop, not handles in existence. A plan that leases more files
    /// than the limit at once exceeds it rather than blocking — blocking here
    /// would deadlock against a caller that already holds leases from the same
    /// plan. `ReaderMetrics::hwm` reports what actually happened.
    fn evict_down_to(&self, open: &mut OpenHandles, target: usize) {
        while open.by_file.len() > target {
            let victim = open
                .by_file
                .iter()
                .filter(|(_, h)| Arc::strong_count(&h.reader) == 1)
                .min_by_key(|(_, h)| h.last_used)
                .map(|(fid, _)| *fid);
            let Some(fid) = victim else { return };
            open.by_file.remove(&fid);
            self.metrics.note_evict();
        }
    }

    /// `n_obs` for a file, without opening it.
    pub(crate) fn n_obs(&self, file_id: u32) -> Option<u64> {
        self.slots.get(file_id as usize).map(|s| s.n_obs)
    }

    /// A file's shard index. Lock-free — this is on the per-row bucketing path.
    ///
    /// Out-of-range `file_id` yields `None` without an error, which is what plan
    /// bucketing wants: the gather's own validation is the thing that reports a
    /// bad id, and it runs later.
    fn index(&self, file_id: u32) -> Option<&BackedCsrIndex> {
        self.slots.get(file_id as usize).map(|s| &s.index)
    }

    /// Which shard holds `row`, or `None` if the file has no such row.
    pub(crate) fn shard_for_row(&self, file_id: u32, row: u64) -> Option<usize> {
        self.index(file_id)?.shard_for_row(row)
    }

    /// CSR shard count for a file.
    pub(crate) fn n_shards(&self, file_id: u32) -> usize {
        self.index(file_id).map(|ix| ix.n_shards()).unwrap_or(0)
    }

    /// How many distinct shards of `file_id` the given rows touch.
    pub(crate) fn shards_touched(&self, file_id: u32, rows: &[u64]) -> usize {
        self.index(file_id)
            .map(|ix| ix.shards_for_indices(rows).len())
            .unwrap_or(0)
    }

    /// True if any file in the set has a row-group-framed CSR shard.
    ///
    /// **Answered from the resident handles first, and only then by reopening
    /// the rest.** At `reader_limit = None` every file is resident, so the walk
    /// opens nothing at all — which is the default and the overwhelmingly
    /// common case. Under a bound it costs a reopen only for files the scan
    /// already closed, and only until the first framed one.
    ///
    /// That ordering is the whole fix. Leasing every slot in turn — the first
    /// version — meant a bounded 26k-file all-unframed manifest reopened and
    /// re-parsed 26k catalogs the scan had closed seconds earlier, to answer a
    /// constructor warning. **Review on #536 (Cursor Agent, Antigravity).**
    ///
    /// The predicate itself stays on `BackedCsrReader`, where the block-index
    /// resolution it depends on lives; re-deriving "is this framed" from a
    /// shard header in this crate would be a second answer free to drift from
    /// the one the gather actually routes on.
    ///
    /// The one caller is the constructor's "you asked for a route that can
    /// never fire" warning, and `should_warn_unframed_scatter` short-circuits
    /// on `scatter_block_index` before reaching here, so the default path never
    /// pays for this at all.
    pub(crate) fn any_shard_framed(&self) -> bool {
        *self.any_framed.get_or_init(|| {
            // Resident first. Collected under the lock and released before any
            // reopen, so this never holds the registry mutex across I/O.
            let resident: Vec<Arc<BackedCsrReader>> = {
                let open = self.open.lock().expect("reader registry mutex poisoned");
                open.by_file
                    .values()
                    .map(|h| Arc::clone(&h.reader))
                    .collect()
            };
            if resident.iter().any(|r| r.any_shard_framed()) {
                return true;
            }
            let seen: std::collections::HashSet<u32> = {
                let open = self.open.lock().expect("reader registry mutex poisoned");
                open.by_file.keys().copied().collect()
            };
            for fid in 0..self.slots.len() as u32 {
                if seen.contains(&fid) {
                    continue;
                }
                match self.lease(fid) {
                    Ok(r) if r.any_shard_framed() => return true,
                    // A file that cannot be reopened cannot be shown to be
                    // framed. This feeds a constructor *warning*; turning it
                    // into an error would fail a construction over a
                    // diagnostic, and the next real read reports it properly.
                    Ok(_) | Err(_) => continue,
                }
            }
            false
        })
    }
}

#[cfg(test)]
#[path = "reader_registry_tests.rs"]
mod tests;
