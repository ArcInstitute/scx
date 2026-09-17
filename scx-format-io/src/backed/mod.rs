//! Backed (on-demand) CSR access for SCX files.
//!
//! Provides [`BackedCsrIndex`] for O(log n) shard lookups and
//! [`BackedCsrReader`] for on-demand shard decoding with optional LRU caching.
//! Used by pyscx's backed mode to implement AnnData-compatible lazy access.
//!
//! # Layout
//!
//! | Module | What |
//! |---|---|
//! | `index` | the row-range → shard binary search, shared by the CSR and dense readers |
//! | `cache` | one byte-budgeted LRU and one singleflight, shared by all three readers |
//! | `csr` | [`BackedCsrReader`] — row access to X, a layer, or a modality |
//! | `aggregate` | the native shard-by-shard statistics kernels built on it |
//! | `csc` | [`BackedCscReader`] — the optional gene-major sidecar |
//! | `dense` | [`BackedDenseReader`] — row gather over an `obsm` mapping |
//! | `pairwise` | [`BackedPairwiseReader`] — bounded row ranges over an `obsp` COO graph |

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
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

mod aggregate;
mod cache;
mod csc;
mod csr;
mod dense;
mod index;
mod pairwise;

// `backed::<name>` is an import path in pyscx, rscx, scx-accel, scx-loader and
// this crate's own `writer_tests.rs`, and six of these are also flat-re-exported
// from `lib.rs`. The split has to keep every one of them resolving, so
// everything reachable at `backed::<name>` before is re-exported here at the
// same visibility.
pub(crate) use cache::csr_component_bytes;
pub use cache::{CacheMetrics, ShardCache, SharedShardCache, SizeHint};
// The key/kind types stay in `cache`: `SharedShardCache` names them, but no
// caller outside this module spells them (review on #528).
pub(super) use cache::{CacheKey, CacheKind};
pub use csc::{BackedCscIndex, BackedCscReader};
pub use csr::BackedCsrReader;
pub use dense::BackedDenseReader;
pub use index::BackedCsrIndex;
pub use pairwise::{BackedPairwiseReader, PairwiseRows};
// Shared with `crate::mapping_shards`, which needs a width-generic read of a
// COO batch's `row` column and should not hand-roll a second one.
pub(crate) use pairwise::coo_coord_borrow;

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

/// Process-wide switch for retaining decoded **row groups** in the shard LRU
/// (OPT-FORMATIO-1). Default on; `SCX_ROW_GROUP_CACHE=0` (or `false`) makes a
/// framed scattered read decode its touched groups and drop them, as it did
/// before the row-group entries existed — the same-build A/B arm for the
/// `read_scattered` capture. Read once per process.
pub fn row_group_cache_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("SCX_ROW_GROUP_CACHE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(true)
    })
}

/// Process-wide switch for the **reuse-signal** admission policy (W10).
///
/// Default `reuse`: a plan over its `budget / (lookahead + 1)` share keeps the
/// row groups another plan of the prefetch window also touches.
/// `SCX_ROW_GROUP_ADMIT=plan` restores the pre-W10 all-or-nothing rule — a plan
/// over its share retains nothing — which is the same-build A/B arm for the
/// capture, exactly as `SCX_ROW_GROUP_CACHE=0` is for the row-group LRU itself.
/// Read once per process.
///
/// Anything other than `plan` (case-insensitive) is `reuse`, including an unset
/// variable and a typo: a misspelled arm must not silently disable the shipped
/// policy in a capture that then reports it as the default.
pub fn row_group_admit_reuse_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        !std::env::var("SCX_ROW_GROUP_ADMIT").is_ok_and(|v| v.eq_ignore_ascii_case("plan"))
    })
}

/// A shard request-group takes the O(rows) block-index path only when the
/// requested rows are a small fraction of the shard — `group_len * DIVISOR <
/// shard_rows`. Shared by `read_rows_with`'s `use_block_index` decision and the
/// plan-prefetch skip ([`BackedCsrReader::block_index_eligible`]) so the two can
/// never drift.
pub const ROW_RANGE_WINDOW_DIVISOR: u64 = 4;

/// Process-wide switch for the **serial** row-group decode.
///
/// Default off: a block-index gather decodes its touched row groups one
/// pool-width chunk at a time. `SCX_ROW_GROUP_SERIAL_DECODE=1` makes every
/// chunk one group and so takes the serial path — the pre-change regime, and
/// the same-build A/B arm for the chunked parallel decode itself. Without it
/// that change is unmeasurable on one build: it is not gated by
/// `SCX_ROW_GROUP_ADMIT`, so it is identical in both admission arms and cancels
/// out of every ratio they produce.
///
/// A **boolean**, not a width. An earlier version took any positive integer,
/// which was a configuration surface no caller set to anything but `1` — and a
/// large value silently defeated the `chunk × max group bytes` peak bound the
/// chunking exists to keep. Anything other than `1` is off, so a typo cannot
/// serialise a capture that then reports itself as the default. Read once per
/// process.
pub fn row_group_serial_decode_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED
        .get_or_init(|| std::env::var("SCX_ROW_GROUP_SERIAL_DECODE").is_ok_and(|v| v.trim() == "1"))
}

/// `(file_id, shard_idx, row_group_idx)` — how the shard cache keys a decoded
/// row group, what [`BackedCsrReader::touched_row_groups`] names, and the unit
/// the loader's reuse signal counts.
pub type RowGroupKey = (u32, usize, usize);

/// What a read is allowed to **retain** in the shard LRU's row-group half.
///
/// A read's *output* never depends on this — retention is a cache policy, not
/// a correctness property. What it decides is whether a decoded row group is
/// inserted under the byte budget (and single-flighted across concurrent
/// callers) or decoded independently and dropped.
///
/// * [`Admit::All`] — retain every group this read touches. What a plan gets
///   when its whole footprint fits the budget share it was sized against.
/// * [`Admit::None`] — retain nothing. A working set larger than the cache is
///   the one pattern an LRU makes strictly worse: every group is inserted and
///   evicted before the next read could hit it, so residency is spent for zero
///   hits (measured at 0 hits, +350-500 MB and 2-4 % slower on
///   tabula_sapiens_100k).
/// * [`Admit::Groups`] — retain exactly the named keys. The reuse signal: a
///   plan over its share forfeits its cold tail but keeps the groups several
///   plans of the prefetch window touch (a control pool, shared neighbours, a
///   repeated pair member). Strictly between the other two, and the reason the
///   verdict is a set rather than a `bool`.
///
/// Keys are `(file_id, shard_idx, row_group_idx)` — the same triple the cache
/// is keyed on, and what [`BackedCsrReader::touched_row_groups`] names. `Arc`
/// because one verdict is consulted by every set of a cell-set plan and, on the
/// parallel decode path, from several threads at once.
#[derive(Debug, Clone, Default)]
pub enum Admit {
    #[default]
    All,
    None,
    Groups(Arc<HashSet<RowGroupKey>>),
}

impl Admit {
    /// Whether group `g` of shard `shard_idx` in file `file_id` may be retained.
    pub fn admits(&self, file_id: u32, shard_idx: usize, g: usize) -> bool {
        match self {
            Admit::All => true,
            Admit::None => false,
            Admit::Groups(keys) => keys.contains(&(file_id, shard_idx, g)),
        }
    }

    /// `Admit::Groups` of `keys`, collapsing an empty set to [`Admit::None`].
    ///
    /// The collapse is not cosmetic. An empty `Groups` and `None` decide every
    /// lookup identically, so keeping them distinct would let a counter or a
    /// test read "partial admission happened" off a verdict that admitted
    /// nothing — which is exactly the claim this phase has to be able to
    /// measure.
    pub fn groups(keys: HashSet<RowGroupKey>) -> Self {
        if keys.is_empty() {
            Admit::None
        } else {
            Admit::Groups(Arc::new(keys))
        }
    }
}

impl From<bool> for Admit {
    fn from(b: bool) -> Self {
        if b {
            Admit::All
        } else {
            Admit::None
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// One file, attached here rather than split per submodule — the precedent
// `reader/mod.rs` set. It reaches private items across all six modules through
// `use super::*` on the re-exports above, and `csr::warm_one_shard` calls back
// into it (`super::tests::note_warm_thread`), which only works while `tests` is
// a sibling of the submodules rather than a child of one of them.
#[cfg(test)]
#[path = "../backed_tests.rs"]
mod tests;
