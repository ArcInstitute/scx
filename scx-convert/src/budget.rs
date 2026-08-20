//! The one place that says what a `memory_budget` buys.
//!
//! Before this module the arithmetic lived in five files with constants that
//! disagreed, and the disagreement was not theoretical: the dense reader
//! divided the budget by 4 to size its slab (`h5ad/dense_stream.rs`) and then
//! multiplied that already-capped slab by 2 to estimate a worker's footprint,
//! spending the same reserve twice. The result was that **every** dense convert
//! under `--memory-budget` collapsed to a single thread (§11.5), while a
//! narrow-dtype dense convert simultaneously overshot the budget three-fold,
//! because the same `/4` was keyed to the source dtype width — a quantity the
//! resident buffers do not depend on.
//!
//! Two ideas were tangled in that one division, and separating them is the fix:
//!
//! * a **share** — what fraction of the budget one concurrent unit may claim,
//!   which is what leaves room for more than one worker; and
//! * a **cost model** — how many bytes a unit actually holds per element,
//!   which is what keeps the process inside the budget at all.
//!
//! [`Share`] is the first; the `*_BYTES_PER_*` constants are the second.
//! [`ALLOCATION_TABLE`] declares every reservation together with the phase it
//! is held in, so "what fraction of the budget does this phase claim" has a
//! written answer that a test can check.
//!
//! ⚠️ **What the table does not yet bound.** Every reservation here sizes one
//! *stage* — the reader's working set, the transpose's buffers — and the ingest
//! and export rows carry `enforced: false` because the worker that owns them
//! also holds the *encoded* shard at the same time. So the invariant proves the
//! declared stages fit, not that the process peak fits. Read `enforced` before
//! quoting a row as a guarantee; the flag is the difference between "we sized
//! this" and "nothing exceeds this".
//!
//! Deliberately **not** gated on `hdf5`, for the same reason `parallel_drain`
//! is not: this is integer arithmetic, and keeping it feature-free is what lets
//! `allocation_table_shares_sum_to_at_most_one_per_phase` (a `#[cfg(test)]`
//! item, so deliberately not an intra-doc link) run in the ordinary
//! `cargo test --workspace` job rather than only in the hdf5 lane. Its
//! production callers *are* hdf5-gated, so at default features the items here
//! are dead by construction while the tests still exercise them.

/// An exact rational share of a memory budget.
///
/// Integer, not `f64`: the invariant in [`ALLOCATION_TABLE`] must be exact, and
/// a design in which three phases each take a third must sum to exactly one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Share {
    num: u64,
    den: u64,
}

impl Share {
    pub(crate) const fn new(num: u64, den: u64) -> Self {
        assert!(num > 0 && den > 0 && num <= den, "share must be in (0, 1]");
        Self { num, den }
    }

    /// Bytes of `budget` this share may claim.
    pub(crate) const fn of(self, budget: u64) -> u64 {
        budget / self.den * self.num
    }

    /// The smallest budget admitting one whole `unit` under this share.
    ///
    /// The refusal predicate and the "need at least N bytes" message must come
    /// from this one function. They used to be written separately, and drifted:
    /// the guard tested `budget / row_bytes == 0` while the message advertised
    /// `4 x row_bytes`, so a budget of twice a row passed a check that claimed
    /// to require four.
    pub(crate) const fn min_budget_for(self, unit: u64) -> u64 {
        // Smallest `b` with `self.of(b) >= unit`. Since `of` floors the
        // division, that is `ceil(unit / num) * den` -- rounding *up*, not
        // `unit * den / num`, which truncates and can report a budget whose
        // own share is smaller than the unit it was supposed to admit.
        // `Share(3,4).min_budget_for(1)` was the case that caught it.
        let units = unit.saturating_add(self.num - 1) / self.num;
        units.saturating_mul(self.den)
    }

    pub(crate) const fn numerator(self) -> u64 {
        self.num
    }

    pub(crate) const fn denominator(self) -> u64 {
        self.den
    }

    /// How many units of this share fit in a budget — the ceiling the worker
    /// derate solves against.
    pub(crate) const fn max_concurrent(self) -> u64 {
        self.den / self.num
    }
}

// ---------------------------------------------------------------------------
// Cost models — bytes actually held, per element or per nonzero.
// ---------------------------------------------------------------------------

/// `i32` column index + `f32` value, the resident cost of one nonzero.
pub(crate) const PAYLOAD_BYTES_PER_NNZ: u64 = 8;

/// Rebuild / encode scratch, bounded by the payload it is built from.
pub(crate) const WORKING_SET_SCRATCH_MULTIPLE: u64 = 2;

/// `u64` shard-local indptr, one entry per row plus the terminator.
pub(crate) const INDPTR_BYTES_PER_ROW: u64 = 8;

/// Bytes per element held while a dense slab is sparsified.
///
/// `read_range_inner` holds, simultaneously: the f32 slab (4 B/elem), the
/// `Vec<u32>` indices and the `Vec<f32>` values. On a matrix stored dense but
/// read as sparse the worst case is every element nonzero, so indices and
/// values each reach one entry per element: `4 + 4 + 4`.
///
/// This is a **bound only because those two vectors are allocated at exact
/// capacity**. They used to start at `n/32` and grow by doubling, which reaches
/// 2x the final length — and transiently 3x, while a realloc holds both
/// buffers — putting the real peak at up to 24 B/elem. The counting pass that
/// makes this number true lives in `read_range_inner`; do not remove it
/// without changing this constant.
pub(crate) const DENSE_SPARSIFY_BYTES_PER_ELEM: u64 = 12;

/// Bytes per element held while a dense slab is read and cast to f32.
///
/// The f32 arm moves its `Vec` straight out of `read_slice_2d`, so it costs
/// only the 4 B/elem slab. Every other arm does `into_iter().map().collect()`,
/// and the source allocation lives until the iterator drops — both buffers are
/// live at once, hence `dtype_bytes + 4`.
const fn dense_read_bytes_per_elem(dtype_bytes: u64) -> u64 {
    dtype_bytes + 4
}

/// Peak bytes per source element held by one dense `read_range` call.
///
/// Read-and-cast and sparsify are **separate phases** — the source buffer is
/// dropped before the sparsify loop starts — so the peak is the larger of the
/// two, never their sum. For every dtype the reader supports (`dtype_bytes` at
/// most 8) the sparsify phase dominates, which is why the answer does not
/// depend on the source width. Expressing it as a `max` rather than hard-coding
/// 12 keeps it correct if a wider source dtype is ever added.
pub(crate) const fn dense_peak_bytes_per_elem(dtype_bytes: u64) -> u64 {
    let read = dense_read_bytes_per_elem(dtype_bytes);
    if read > DENSE_SPARSIFY_BYTES_PER_ELEM {
        read
    } else {
        DENSE_SPARSIFY_BYTES_PER_ELEM
    }
}

/// Per-modality density assumptions for the sparse readers' default
/// `per_worker_bytes`. Over-estimating here costs parallelism, so the values
/// err conservative — but note that "over-estimating is safe" is exactly the
/// reasoning that let §11.5 stand, so it is a reason to keep the numbers
/// plausible, not a licence to ignore them.
pub(crate) const PARALLEL_DENSITY_DEFAULT_DEN: u64 = 20; // ≈ 5 % RNA/general
pub(crate) const PARALLEL_DENSITY_ATAC_DEN: u64 = 10; // ≈ 10 % ATAC peak matrices

// ---------------------------------------------------------------------------
// Shares.
// ---------------------------------------------------------------------------

/// What one in-flight shard may claim.
///
/// The reciprocal is the concurrency this permits: at 1/4 the derate can grant
/// four outstanding shards (three workers plus one queue slot). Raising it
/// trades parallelism for larger shards; **lowering the reciprocal below 2
/// re-creates §11.5**, because a single outstanding shard is the sequential
/// coordinator.
///
/// It is deliberately a compile-time constant rather than something solved from
/// `reader_threads`: a thread-aware share would make `--reader-threads`, a pure
/// performance knob, change the on-disk shard layout.
pub(crate) const SHARD_BUDGET_SHARE: Share = Share::new(1, 4);

/// The CSC external transpose's pass-1 column-chunk read.
pub(crate) const CSC_COLUMN_CHUNK_SHARE: Share = Share::new(1, 2);

/// The CSC external transpose's pass-2 bucket record buffer.
pub(crate) const CSC_BUCKET_SHARE: Share = Share::new(1, 4);

// ---------------------------------------------------------------------------
// Working-set models.
// ---------------------------------------------------------------------------

/// Resident bytes for one CSR shard: payload, its scratch, and the indptr.
///
/// Single source for the ingest worker derate and the export-side per-shard
/// estimate, so the two cannot drift.
pub(crate) fn shard_working_set_bytes(nnz: u64, n_rows: u64) -> u64 {
    let payload = nnz.saturating_mul(PAYLOAD_BYTES_PER_NNZ);
    let indptr = n_rows
        .saturating_add(1)
        .saturating_mul(INDPTR_BYTES_PER_ROW);
    payload
        .saturating_mul(WORKING_SET_SCRATCH_MULTIPLE)
        .saturating_add(indptr)
        .max(1)
}

/// Resident bytes for one dense slab of `rows` x `n_vars` elements.
///
/// Includes the `u64` indptr. It is negligible at atlas `n_vars` and is not
/// negligible at `n_vars = 1`, where it is the dominant term — omitting it was
/// an under-count that happened to be invisible on every fixture in the suite.
pub(crate) fn dense_slab_bytes(rows: u64, n_vars: u64, dtype_bytes: u64) -> u64 {
    let elements = rows
        .saturating_mul(n_vars)
        .saturating_mul(dense_peak_bytes_per_elem(dtype_bytes));
    let indptr = rows.saturating_add(1).saturating_mul(INDPTR_BYTES_PER_ROW);
    elements.saturating_add(indptr).max(1)
}

/// The smallest `memory_budget` that admits one row of a dense slab.
pub(crate) fn dense_min_budget(n_vars: u64, dtype_bytes: u64) -> u64 {
    SHARD_BUDGET_SHARE.min_budget_for(dense_slab_bytes(1, n_vars, dtype_bytes))
}

/// Rows of a dense slab that fit this shard's share of `budget`.
///
/// `Err` when the budget cannot admit even one row; the caller turns that into
/// an actionable refusal quoting [`dense_min_budget`], so the predicate and the
/// advertised minimum are the same expression by construction.
pub(crate) fn dense_max_slab_rows(budget: u64, n_vars: u64, dtype_bytes: u64) -> Result<u64, ()> {
    // Solve `rows` from `dense_slab_bytes`, rather than dividing the share by a
    // one-row cost. The indptr's terminator makes the cost affine, not linear —
    // `rows x (n_vars x peak + 8) + 8` — so dividing by `dense_slab_bytes(1, …)`
    // charges the terminator once per row and leaves a whole row unused.
    let share = SHARD_BUDGET_SHARE.of(budget);
    let per_row = n_vars
        .saturating_mul(dense_peak_bytes_per_elem(dtype_bytes))
        .saturating_add(INDPTR_BYTES_PER_ROW)
        .max(1);
    let rows = share.saturating_sub(INDPTR_BYTES_PER_ROW) / per_row;
    if rows == 0 {
        return Err(());
    }
    Ok(rows)
}

/// Bytes the CSC sidecar builder may use: the caller's `memory_budget` capped
/// at the builder's own default, or that default when no budget was given.
///
/// `write_csc_sidecar` documents its bound as "`cols_per_shard` or
/// `memory_budget_bytes`, whichever is smaller", and `pipeline.rs` documented
/// the convert side as "or the memory budget, whichever is smaller" — but the
/// callers passed the 4 GiB default unconditionally, so `--memory-budget 512M
/// --csc always` could still let sidecar generation claim 4 GiB. The doc was
/// true of the callee and false of every caller.
pub(crate) fn csc_sidecar_bytes(memory_budget: Option<u64>) -> u64 {
    let default = scx_format_io::csc_sidecar::DEFAULT_CSC_MEMORY_BYTES as u64;
    memory_budget.map_or(default, |b| b.min(default))
}

// ---------------------------------------------------------------------------
// The declared table.
// ---------------------------------------------------------------------------

/// When a reservation's bytes are *held*.
///
/// The table below is a **declaration**: production code derives its shares
/// from the constants above, and the table records what each of those claims
/// costs so `allocation_table_shares_sum_to_at_most_one_per_phase` can check
/// that the concurrent ones fit. It is therefore read only from tests, which is
/// the intended shape rather than an oversight -- a reservation nobody wrote
/// down is exactly the state this module exists to end.
///
/// The distinction matters because reservations in different phases are not
/// concurrent and must not be summed. The CSC external transpose is the case
/// that forces this: it claims half the budget for a column chunk in pass 1 and
/// a quarter for bucket records in pass 2, and a naive "the fractions must sum
/// to at most one" test over those two would be asserting something nobody
/// claimed.
#[cfg_attr(not(test), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Dense h5ad ingest, steady state.
    DenseIngest,
    /// Sparse (CSR) h5ad ingest, steady state.
    SparseIngest,
    /// SCX -> h5ad/h5mu export, steady state.
    Export,
    /// CSC external transpose, pass 1: scan columns, spill to buckets.
    CscExternalColumnScan,
    /// CSC external transpose, pass 2: load one bucket, emit shards.
    CscExternalBucketDrain,
    /// CSC sidecar generation, after X is written.
    CscSidecar,
}

/// One declared claim on the budget.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) struct Reservation {
    pub name: &'static str,
    pub phase: Phase,
    pub share: Share,
    /// Concurrent copies of this reservation held in `phase`.
    pub multiplicity: u64,
    /// Where the claim is made, for the reader who wants the code.
    pub site: &'static str,
    /// `false` when the table declares a claim the code does not yet enforce.
    /// Such a row documents a known gap; it is not a promise.
    pub enforced: bool,
}

impl Reservation {
    /// `"1/4 x 4"` — the share and how many copies are held at once. Used in
    /// the invariant's failure message, which is also what keeps `site` and
    /// `share` load-bearing rather than decorative.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn share_str(&self) -> String {
        format!(
            "{}/{} x {}",
            self.share.numerator(),
            self.share.denominator(),
            self.multiplicity
        )
    }
}

/// Every declared claim on a `memory_budget`, by phase.
///
/// A reservation that is not in this table is a claim nobody wrote down —
/// which is the state this module exists to end.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const ALLOCATION_TABLE: &[Reservation] = &[
    Reservation {
        name: "dense slab, one in-flight shard",
        phase: Phase::DenseIngest,
        share: SHARD_BUDGET_SHARE,
        multiplicity: SHARD_BUDGET_SHARE.max_concurrent(),
        site: "h5ad/dense_stream.rs::open_dense_streaming + per_worker_bytes",
        // READER-PHASE ONLY, which is why this is not `enforced: true`.
        // `encode_one_shard_worker` holds the raw CSR across `encode_one_shard`
        // and `maybe_build_bitmap_shard`, so the encoded `PreEncodedSection`
        // (and any bitmap) is live *alongside* the slab this reservation sizes.
        // The derate therefore bounds the reader working set, not the whole
        // worker, and `outstanding x share <= 1` is a statement about the
        // former. Closing it means sizing from the maximum complete worker
        // phase and re-deriving the share, which trades away the parallelism
        // 11.5 just restored -- a measured change, not a footnote.
        enforced: false,
    },
    Reservation {
        name: "CSR shard working set, one in-flight shard",
        phase: Phase::SparseIngest,
        share: SHARD_BUDGET_SHARE,
        multiplicity: SHARD_BUDGET_SHARE.max_concurrent(),
        site: "h5ad/stream.rs::per_worker_bytes -> derate_threads_and_depth",
        // Same reader-vs-worker gap as the dense row above: the encoded output
        // overlaps the shard working set this sizes.
        enforced: false,
    },
    Reservation {
        name: "export shard working set, one in-flight shard",
        phase: Phase::Export,
        share: SHARD_BUDGET_SHARE,
        multiplicity: SHARD_BUDGET_SHARE.max_concurrent(),
        site: "h5ad/stream_write.rs::per_shard_export_bytes -> derate_threads_and_depth",
        // Same reader-vs-worker gap as the dense row above: the encoded output
        // overlaps the shard working set this sizes.
        enforced: false,
    },
    Reservation {
        name: "CSC sidecar transpose",
        phase: Phase::CscSidecar,
        share: Share::new(1, 1),
        multiplicity: 1,
        site: "pipeline.rs -> scx_format_io::csc_sidecar::write_csc_sidecar",
        // Bounds the emitted shard's column count. Whole budget rather than a
        // share: the sidecar is built after X is written, not alongside it.
        enforced: true,
    },
    Reservation {
        name: "CSC column chunk",
        phase: Phase::CscExternalColumnScan,
        share: CSC_COLUMN_CHUNK_SHARE,
        multiplicity: 1,
        site: "h5ad/csc_stream.rs::CscToCsrExternalTransposer::open",
        enforced: true,
    },
    Reservation {
        name: "CSC bucket spill writers",
        phase: Phase::CscExternalColumnScan,
        // n_buckets x 8 KiB of BufWriter, unaccounted: the bucket count is
        // derived from the *mean* nnz/row, so a right-skewed depth
        // distribution can put it far above this. That is §11.4, whose
        // prescribed fix -- a per-row nnz quantile from `col_indptr` -- is not
        // implementable as written: `col_indptr` is per *column*, buckets are
        // row ranges, and deriving a row distribution needs the full pass over
        // `indices` that this route exists to avoid. Declared here so the gap
        // is named rather than invisible.
        share: Share::new(1, 8),
        multiplicity: 1,
        site: "h5ad/csc_stream.rs (BufWriter per bucket)",
        enforced: false,
    },
    Reservation {
        name: "CSC bucket records",
        phase: Phase::CscExternalBucketDrain,
        share: CSC_BUCKET_SHARE,
        multiplicity: 1,
        site: "h5ad/csc_stream.rs::read_bucket",
        // Sized from the mean nnz/row, so a high-depth bucket overshoots.
        // §11.4's other half.
        enforced: false,
    },
];

#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
