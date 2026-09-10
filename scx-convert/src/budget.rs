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
//! ⚠️ **What the table does not bound.** Read `enforced` before quoting a row
//! as a guarantee; the flag is the difference between "we sized this" and
//! "nothing exceeds this". **Every row here is still `enforced: false` except
//! none** — all seven are unenforced, and this change does not alter that.
//! What it changes is the *estimate*: the two **ingest** rows now size the
//! whole worker phase (payload, the encoder's value copy and the framed
//! encode's buffers, via [`WORKER_PHASE_BYTES_PER_NNZ`]) instead of the reader
//! stage alone, and the **export** row is sized from its own decode model
//! rather than borrowing the ingest one. Widening a cost model is not proving
//! a ceiling; each row names what it still does not bound.
//!
//! # Why the encode term is charged, and how it is derived
//!
//! Until PR-42 the encode side rode along inside a
//! `WORKING_SET_SCRATCH_MULTIPLE = 2` documented as "rebuild / encode scratch,
//! bounded by the payload it is built from". Parallelising the encode broke
//! that bound in two composing ways:
//!
//! * **Per framed encode: ~2x one shard's encoded bytes.** Every group's
//!   encoded bytes are live at once, alongside the three sub-streams assembled
//!   from them. Previously it was the streams plus one group. (The streams are
//!   now sized exactly, which removes the old doubling overshoot and the
//!   double buffer during its final realloc — so ~2x is the honest figure, not
//!   2x on top of a previous 1x.)
//! * **x2 again under `codec="auto"`/`"compact"`/`compact-trial`.**
//!   `encode_shard_adaptive` runs the two candidates under `rayon::join`, so
//!   the two per-encode transients **overlap**: up to **~4x** the encoded
//!   shard. Serially they did not — one candidate collapsed to a single
//!   `EncodedShard` before the next began.
//!
//! In B/nnz, which is the unit the reservations below use: the compressing
//! integer codecs run 1-4 B/nnz on real count data, but the bound has to hold
//! at the worst *reachable* case, and that is not the typical one.
//! `select_codec_for_modality` picks `Lz4Shuffle` for non-binary integer ATAC
//! peak counts, and on incompressible data LZ4 approaches its ~8 B/nnz input,
//! so a dual-encoded ATAC shard transiently holds ~32 B/nnz. Charging
//! `ENCODE_TRANSIENT_MULTIPLE = 4` against the 8 B/nnz payload is exactly that
//! figure — the bound is set by what a compressor cannot beat, its input, not
//! by what one usually achieves. Adding the resident payload and the encoder's
//! own value copy gives **48 B/nnz**, where the old model charged 16.
//!
//! `CodecId::None` is the other 8 B/nnz encoding and it is **not** reachable
//! as a dual-encode candidate: it exists only as an explicit `--codec none`
//! force, and `resolve_codec` gives an explicit force `decode_target: None`
//! and `codec_trial: false`, so it single-encodes. It is inside the same
//! bound at half the term.
//!
//! Measured on `scx compact --codec auto`, two release worktrees, 3 runs each,
//! 16 cores:
//!
//! | fixture | wall | peak RSS |
//! |---|---|---|
//! | pbmc3k (1 shard) | 0.153 -> 0.083 s | 49 -> 78 MB (+59%) |
//! | smartseq2 (4 shards) | 11.656 -> 4.659 s | 1364 -> 2010 MB (**+47%**) |
//! | census_500k (31 shards) | 43.439 -> 18.553 s | 1678 -> 2030 MB (+21%) |
//!
//! The spread across fixtures is the thing to read, not the census number:
//! the added term is one shard's *encoded* bytes, so it grows with nnz per
//! shard and not with the file. smartseq2 is deep-sequenced — few cells, many
//! nnz each — so its shards are large and the term is nearly half its peak,
//! while census_500k's 31 thinner shards dilute it to a fifth. Size this from
//! the widest, deepest shard a caller can produce, not from a census average.
//!
//! All three are the serial-encode path — `compact` encodes one shard at a
//! time — so a parallel convert multiplies the term by its granted worker
//! count. Every arm's per-section digests were identical to `main`'s (63
//! sections on census_500k), so none of this is a fidelity question.
//!
//! **None of the shares moved, and that is the point.** Every phase here is
//! claimed to exactly 1 (share x multiplicity), so the term could not have
//! been added as a *row* — `allocation_table_shares_sum_to_at_most_one_per_phase`
//! would reject it. What widened is the **cost model** the share is applied
//! to, which leaves the invariant intact and needs no new `Phase`. The price
//! is paid in granted concurrency instead: `derate_threads_and_depth` solves
//! `budget / per_shard_bytes`, and `per_shard_bytes` is **3x** larger on the
//! sparse ingest path (16 -> 48 B/nnz) and 3.7x on the dense one
//! (12 -> 44 B/element), so a budgeted convert is granted correspondingly
//! fewer workers and some budgets that were accepted are now refused outright
//! by `ensure_shard_fits_budget`. That refusal is the correct answer — those
//! runs were over-committing — and it is opt-in, because `memory_budget`
//! defaults to `None` on every surface and `None` skips the derate entirely.
//!
//! **Export is not charged this.** It decodes; `h5ad/stream_write.rs` contains
//! no encode call, so it takes `shard_decode_working_set_bytes` (16 B/nnz).
//! Sharing the ingest model with it over-derated a budgeted export 3x for
//! memory it never holds, and over-estimating is not the safe direction —
//! `docs/conventions.md` records that it silently routes to the sequential
//! coordinator. That row is still `enforced: false` too: `filter_shard`
//! allocates past what the decode model charges (its own row says how).
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

/// Decoder scratch on the export path, as a multiple of the payload.
///
/// `stream_csr_to_group_at` decodes a shard and filters it before the
/// hyperslab write, so a second copy of the payload is live. This is the term
/// the old shared model called "codec scratch"; on a decode path it is the
/// *whole* non-payload cost, because nothing there encodes.
pub(crate) const DECODE_SCRATCH_MULTIPLE: u64 = 1;

/// The encoder's own copy of the values, as a multiple of the payload.
///
/// `encode_one_shard` materialises `shard_values_bytes` — every value
/// re-serialised to its on-disk width — and **borrows it across the whole
/// encode** (`scx-format-io/src/encoder.rs:153`, passed on at `:162`), so it
/// is live at the peak rather than before it. At most 4 B/nnz against the
/// payload's 8, i.e. half a payload; charged as a whole one because these are
/// integer multiples of [`PAYLOAD_BYTES_PER_NNZ`] and rounding down here is
/// rounding the bound in the unsafe direction.
///
/// ⚠️ This term is why the whole-phase figure is **6x** and not the 5x an
/// earlier derivation gave. That derivation read the old
/// `WORKING_SET_SCRATCH_MULTIPLE = 2` as "payload + encode" and replaced the
/// encode half with 4; it missed that the 2 also had to cover this copy, and
/// at the worst reachable case (incompressible integers) 5x under-covers by
/// exactly this 4 B/nnz. Do not "simplify" it back.
pub(crate) const ENCODER_VALUE_COPY_MULTIPLE: u64 = 1;

/// The framed encode's live buffers, as a multiple of the payload they encode.
///
/// Derived, not a safety factor:
///
/// * `encode_shard_framed` holds **every** row group's encoded bytes
///   (`encoded: Vec<EncodedShard>`) *and* the three concatenated sub-streams
///   assembled from them, reserved to their exact final size, at the same
///   moment — **2x** the encoded shard.
/// * `encode_shard_adaptive` runs two candidate codecs under `rayon::join`
///   whenever `decode_target`/`trial` is set, which `codec="auto"` does for
///   every integer shard — so two of those transients **overlap**: **4x**.
/// * One nonzero encodes to at most its input: an index (<= 4 B) plus a value
///   (<= 4 B) is `PAYLOAD_BYTES_PER_NNZ`, and every codec here is either a
///   compressor or a 1:1 copy. The per-group header (a few bytes per 256 rows)
///   is the epsilon this rounds over.
///
/// So `4 x payload` bounds the encode phase for all six codecs. Before the
/// row groups and the candidates went parallel this was `1x`, folded into a
/// `WORKING_SET_SCRATCH_MULTIPLE = 2` that read "rebuild / encode scratch,
/// bounded by the payload it is built from" — a bound that stopped holding.
/// See `scx-format-io/src/encoder.rs`'s `rayon::join` comment.
///
/// ⚠️ **Charged unconditionally, including to jobs that provably
/// single-encode.** The `x2` above is the dual-candidate overlap, which only
/// `codec="auto"` / `"compact"` / `compact-trial` incur: `resolve_codec` gives
/// the `fast` profile and every explicit codec `decode_target: None`, and an
/// unframed output falls back to one encode. `run_streaming_writer_coordinator`
/// asks `indexed.per_worker_bytes(...)` and passes none of `opts.codec`,
/// `codec_trial` or `decode_target`, so a `--codec zstd` convert is charged
/// ~2x the encode buffers it will hold. That is the **safe** direction for
/// memory but not a free one: it costs those jobs threads, and it raises the
/// smallest budget `ensure_shard_fits_budget` accepts. Closing it means
/// threading the resolved encode plan through
/// [`crate::stream::IndexedCsrShardStream::per_worker_bytes`] (four impls and
/// two call sites) and splitting this into single- and dual-candidate costs;
/// deliberately not done here.
pub(crate) const ENCODE_TRANSIENT_MULTIPLE: u64 = 4;

/// Bytes one nonzero costs across a worker's **whole** phase: the resident
/// payload, the encoder's value copy, and the framed encode's buffers.
///
/// `1 + 1 + 4 = 6`, i.e. 48 B/nnz where the old model charged 16. This is the
/// figure the two ingest reservations are sized from: the payload alone bounds
/// the reader stage and not the worker, which is what
/// `derate_threads_and_depth` actually has to fit. It is a much better
/// estimate and **not** a ceiling — `ALLOCATION_TABLE`'s sparse-ingest row
/// enumerates the terms it does not bound.
///
/// What is deliberately still outside it: the optional `BitmapShard`, and (on
/// the dense path) nothing — `dense_slab_bytes` covers the slab and the
/// sparsified vectors, and the encode term is added to it separately.
pub(crate) const WORKER_PHASE_BYTES_PER_NNZ: u64 =
    PAYLOAD_BYTES_PER_NNZ * (1 + ENCODER_VALUE_COPY_MULTIPLE + ENCODE_TRANSIENT_MULTIPLE);

/// The whole-phase cost must exceed the payload-plus-one-scratch model this
/// replaced, or charging the encode achieved nothing.
///
/// A compile-time assertion rather than a unit test: it holds in every build,
/// including one compiled without tests, and clippy rightly rejects an
/// `assert!` on two constants inside a `#[test]`.
const _: () = assert!(WORKER_PHASE_BYTES_PER_NNZ > PAYLOAD_BYTES_PER_NNZ * 2);

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

/// Resident bytes for one CSR shard across a worker's **whole** phase: the
/// payload, the reader's rebuild scratch, the framed encode's buffers, and the
/// indptr — [`WORKER_PHASE_BYTES_PER_NNZ`] per nonzero.
///
/// **Ingest only.** It was the single source for the ingest derate and the
/// export-side per-shard estimate until the encode charge landed; export holds
/// no encode buffers, so a shared model over-derated a budgeted export
/// threefold. The export side is [`shard_decode_working_set_bytes`], and the
/// two are deliberately separate rather than one function two phases share.
pub(crate) fn shard_working_set_bytes(nnz: u64, n_rows: u64) -> u64 {
    let indptr = n_rows
        .saturating_add(1)
        .saturating_mul(INDPTR_BYTES_PER_ROW);
    nnz.saturating_mul(WORKER_PHASE_BYTES_PER_NNZ)
        .saturating_add(indptr)
        .max(1)
}

/// Resident bytes for one CSR shard on a **decode** path: the payload and the
/// decoder's scratch, plus the indptr — no encode term.
///
/// This is the export estimate, and it is deliberately *not*
/// [`shard_working_set_bytes`]. `h5ad/stream_write.rs` contains no encode call
/// at all: it decodes a shard into a `DecodedShard { indptr, indices, data }`
/// and hyperslab-writes it. Charging it the framed encode's 4x would derate a
/// budgeted export threefold for memory it never holds — and
/// `docs/conventions.md` is explicit that over-estimating is not the safe
/// direction, because it silently routes to the sequential coordinator.
///
/// The two directions shared one function until the encode charge landed, on
/// the argument that a single model cannot drift. That argument only holds
/// while the two phases are the same; they are not, and the shared model was
/// wrong for one of them.
pub(crate) fn shard_decode_working_set_bytes(nnz: u64, n_rows: u64) -> u64 {
    let payload = nnz.saturating_mul(PAYLOAD_BYTES_PER_NNZ);
    let indptr = n_rows
        .saturating_add(1)
        .saturating_mul(INDPTR_BYTES_PER_ROW);
    payload
        .saturating_mul(1 + DECODE_SCRATCH_MULTIPLE)
        .saturating_add(indptr)
        .max(1)
}

/// The sparse readers' default per-worker estimate: a density guess over the
/// shard's element count, at the whole-worker-phase cost per nonzero.
///
/// Lives here rather than at the call site because
/// `docs/conventions.md` promises every constant and fraction in the derate
/// comes from this file, and a bare `16` in `stream.rs` was making that false.
pub(crate) fn estimated_worker_bytes(shard_target_rows: u64, n_vars: u64, density_den: u64) -> u64 {
    shard_target_rows
        .saturating_mul(n_vars)
        .saturating_mul(WORKER_PHASE_BYTES_PER_NNZ)
        / density_den.max(1)
}

/// Bytes one dense *source element* costs across a worker's whole phase: the
/// slab and its sparsified vectors, plus the framed encode's buffers for the
/// nonzero that element may become.
///
/// The nnz bound is the element count — the same every-element-nonzero worst
/// case [`DENSE_SPARSIFY_BYTES_PER_ELEM`] already assumes, so the two terms
/// are consistent rather than one being pessimistic against the other.
///
/// It returns 44 at every width this crate supports today, so `dtype_bytes`
/// looks inert — three reviewers have now suggested dropping it. It is not:
/// [`dense_peak_bytes_per_elem`] is `max(dtype_bytes + 4, 12)`, constant only
/// because every dtype the reader accepts is at most 8 bytes wide. Hard-coding
/// 44 would bake that ceiling in silently, which is the property that function
/// deliberately expresses as a `max` rather than as the literal 12. Both
/// callers ([`dense_slab_bytes`], [`dense_max_slab_rows`]) already take
/// `dtype_bytes` from the reader, so threading it costs nothing.
///
/// Keeping it a `const fn` rather than a `const` also keeps CI's "both dense
/// budget sites derive from one table entry" guard working: it counts
/// `crate::budget::dense_*(` **calls** in `h5ad/dense_stream.rs` and fails
/// below four, and a constant is not a call.
///
/// ⚠️ **Both dense sites must use this, not the slab alone.** Charging the
/// encode term in `per_worker_bytes` while sizing the slab cap from the slab
/// alone re-creates §11.5 from the other direction: the cap yields rows whose
/// whole-phase cost then exceeds the share, `outstanding_max` falls to 1, and
/// a budgeted dense convert silently takes the sequential coordinator. Caught
/// by `dense_convert_under_a_memory_budget_stays_parallel`, which is why that
/// test exists. Deriving the cap from this figure keeps
/// `per_worker_bytes(max_slab_rows) ≈ share` by construction, so the budget
/// buys smaller shards rather than fewer workers.
pub(crate) const fn dense_worker_phase_bytes_per_elem(dtype_bytes: u64) -> u64 {
    dense_peak_bytes_per_elem(dtype_bytes) + PAYLOAD_BYTES_PER_NNZ * ENCODE_TRANSIENT_MULTIPLE
}

/// Resident bytes for one dense slab of `rows` x `n_vars` elements.
///
/// Includes the `u64` indptr. It is negligible at atlas `n_vars` and is not
/// negligible at `n_vars = 1`, where it is the dominant term — omitting it was
/// an under-count that happened to be invisible on every fixture in the suite.
pub(crate) fn dense_slab_bytes(rows: u64, n_vars: u64, dtype_bytes: u64) -> u64 {
    let elements = rows
        .saturating_mul(n_vars)
        .saturating_mul(dense_worker_phase_bytes_per_elem(dtype_bytes));
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
        .saturating_mul(dense_worker_phase_bytes_per_elem(dtype_bytes))
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
/// `memory_budget_bytes`, whichever is smaller", and the pipeline documented
/// the convert side as "or the memory budget, whichever is smaller" — but the
/// callers passed the 4 GiB default unconditionally, so `--memory-budget 512M
/// --csc always` could still let sidecar generation claim 4 GiB. The doc was
/// true of the callee and false of every caller.
pub fn csc_sidecar_bytes(memory_budget: Option<u64>) -> u64 {
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
        name: "dense slab + encode transient, one in-flight shard",
        phase: Phase::DenseIngest,
        share: SHARD_BUDGET_SHARE,
        multiplicity: SHARD_BUDGET_SHARE.max_concurrent(),
        site: "h5ad/dense_stream.rs::open_dense_streaming + per_worker_bytes",
        // Sized from the whole worker phase — `dense_worker_phase_bytes_per_elem`
        // covers the slab, its sparsified vectors and the framed encode, and
        // the slab cap derives from the same figure so the two cannot drift.
        // That is a 3.7x better estimate than the slab alone, and it is still
        // **not a proof**, which is why this stays `enforced: false`. See the
        // sparse row below for the list of terms it does not bound.
        enforced: false,
    },
    Reservation {
        name: "CSR shard working set + encode transient, one in-flight shard",
        phase: Phase::SparseIngest,
        share: SHARD_BUDGET_SHARE,
        multiplicity: SHARD_BUDGET_SHARE.max_concurrent(),
        site: "h5ad/stream.rs::per_worker_bytes -> derate_threads_and_depth",
        // `shard_working_set_bytes` charges `WORKER_PHASE_BYTES_PER_NNZ`, so
        // the encoded output no longer overlaps the share unpriced — 48 B/nnz
        // against the 16 this used to claim.
        //
        // Still `enforced: false`, and the honest reason is that
        // `encoded <= payload` is a good estimate rather than a guarantee.
        // What it does not bound:
        //
        //   * **frame expansion.** Zstd and LZ4 can emit slightly more than
        //     their input on incompressible data; the model assumes they
        //     cannot.
        //   * **intra-codec planes.** A codec holds its raw input, its
        //     shuffled/delta plane and its compressed output at once, so one
        //     candidate's internal peak can exceed its own output.
        //   * **the indptr.** Each candidate encodes it too; the model prices
        //     the encode per *nnz* only, so a shard with `nnz = 0` is charged
        //     one indptr and nothing else while both candidates still
        //     allocate.
        //   * **the bitmap.** `maybe_build_bitmap_shard` builds a `BTreeMap`
        //     of `RoaringBitmap`s while the raw shard and the
        //     `PreEncodedSection` are both live.
        //   * **readers without an exact nnz.** `MaterializedCsrStream` and
        //     anything else on the trait default takes the 5%/10% density
        //     guess, which a dense-stored-as-CSR matrix under-estimates.
        //     (`PermutedCsrReader` used to belong here for a different
        //     reason — it delegated to the CSR override, whose source-aligned
        //     window maximum does not describe the shard a permutation emits.
        //     It now walks `perm` through the inner `indptr` instead, so a
        //     sort- or group-on-convert prices the shard it will write.)
        //
        // Closing it means codec-specific worst cases plus an exact override
        // for the density-guess readers, with allocation tests for the
        // all-zero, shallow, incompressible and bitmap-enabled shapes. Until
        // then this is a much better *size*, not a ceiling.
        enforced: false,
    },
    Reservation {
        name: "export shard decode working set, one in-flight shard",
        phase: Phase::Export,
        share: SHARD_BUDGET_SHARE,
        multiplicity: SHARD_BUDGET_SHARE.max_concurrent(),
        site: "h5ad/stream_write.rs::per_shard_export_bytes -> derate_threads_and_depth",
        // `shard_decode_working_set_bytes`: the payload plus the decoder's
        // scratch, sized from `ShardStats.nnz` — the catalog's exact count,
        // not a density guess — with no encode term, because
        // `stream_write.rs` contains no encode call at all. It briefly shared
        // the ingest model, which over-charged it 3x for an encoder it never
        // runs; over-estimating is not the safe direction here, since
        // `docs/conventions.md` records that it silently routes to the
        // sequential coordinator.
        //
        // Still `enforced: false`, and an intermediate revision of this PR
        // wrongly said otherwise. `filter_shard` allocates beyond what this
        // charges, on both of its paths:
        //
        //   * no filter — a `kept_indptr_tail` of `n_rows` entries while
        //     `indptr_local` is still live, i.e. two indptrs, not one. On a
        //     shard with `nnz = 0` that is the whole cost and the model
        //     charges half of it.
        //   * masked — `kept_indices` and `kept_data` start at `Vec::new()`
        //     and grow by doubling while the originals are live, so their
        //     capacity can exceed their length and both copies coexist.
        //
        // Closing it means charging both indptrs and capacity-aware filtered
        // buffers, with all-zero, sub-1-nnz/row and masked cases measured.
        enforced: false,
    },
    Reservation {
        name: "CSC sidecar transpose",
        phase: Phase::CscSidecar,
        share: Share::new(1, 1),
        multiplicity: 1,
        site: "pipeline/shards.rs + h5mu/pipeline.rs -> csc_sidecar::write_csc_sidecar",
        // Whole budget rather than a share: the sidecar is built after X is
        // written, not alongside it.
        //
        // NOT enforced, and the first version of this row wrongly said it was.
        // The parameter sizes the transpose CHUNK only: `compute_chunk_cols`
        // reserves 12 B per potential entry while the chunk it returns is
        // 8 B/entry, and `write_csc_sidecar` then builds full-length
        // `csc_indptr_u64` / `csc_indices_u32` / raw value copies while that
        // chunk is still live, before the writer allocates encoded streams.
        // `scx-ops/src/build_csc.rs` is worse: it collects every source shard
        // into a `Vec<ScxCsr>` and holds it across the transpose loop. So the
        // budget controls column/shard sizing, not a memory ceiling.
        enforced: false,
    },
    Reservation {
        name: "CSC column chunk",
        phase: Phase::CscExternalColumnScan,
        share: CSC_COLUMN_CHUNK_SHARE,
        multiplicity: 1,
        site: "h5ad/csc_stream.rs::CscToCsrExternalTransposer::open",
        // NOT enforced. The chunk loop starts `col_end` at `col_start + 1` and
        // only then tests the budget, so column `col_start` is always read
        // whole: a single column with `8 x nnz` over the share blows it, which
        // on a large atlas is an ordinary ubiquitous gene rather than a corner
        // case. The `.max(PAYLOAD_BYTES_PER_NNZ * 8)` floor also exceeds the
        // share for any accepted budget under 128 bytes. Bounding it means
        // splitting or refusing an over-budget first column — a behaviour
        // change in the external transpose, which is §11.4's territory (see
        // ORG-11.16-5b), not a flag flip.
        enforced: false,
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
