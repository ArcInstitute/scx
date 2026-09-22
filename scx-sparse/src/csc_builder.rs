// One-pass, push-based CSR → CSC transpose with bucketed spill.
//
// # Why this exists
//
// `transpose::streaming_csr_to_csc_iter_with_cap` answers a column chunk at a
// time by rescanning **every** nonzero of **every** source shard, twice, per
// chunk — `2 * nnz * n_chunks` in total. On census_1m (1.40 B nnz, 173 chunks)
// that is 4.8e11 index comparisons, and it also forces its caller to hold the
// whole decoded CSR alive for the iterator's lifetime, because it borrows
// `&[ScxCsr]`.
//
// This builder inverts the control flow. It is a **sink**: shards are pushed
// into it in row order, each nonzero is routed once into a contiguous column
// bucket, and the CSC shards come out at `finish()`. Total work is `2 * nnz`,
// the caller holds one decoded shard at a time, and the resident set is a
// counter the caller's budget actually bounds.
//
// The push shape is required, not preferred: the rewrite ops (`sort`,
// `compact`) emit each output CSR shard exactly once and cannot be re-passed,
// so a sink is the only interface that can build a sidecar in the same pass
// they already make.
//
// # No sort, ever
//
// `scx-convert`'s external CSC→CSR transpose (`h5ad/csc_stream.rs`) buckets by
// row and must `sort_unstable_by_key` + `coalesce_sorted_coo` each bucket. Two
// things let this builder skip that, and both are worth stating because losing
// either silently breaks byte-identity:
//
//   1. **The input is already canonical.** That is the same contract
//      `write_csc_sidecar` states; `csc_stream` reads arbitrary h5ad and has to
//      implement scipy's `sum_duplicates()`. Coalescing here would *move* bytes
//      relative to the predecessor, not clean them up.
//   2. **The bucket key is the output's major axis, and arrival orders the
//      minor one.** Rows arrive globally ascending, so arrival order restricted
//      to any single column is ascending in row — which is exactly what the
//      counting scatter needs, and exactly what the predecessor produced.
//
// # Shard boundaries are unchanged
//
// Bucketing decouples the *memory block* from the *output shard*, so shard
// width becomes a free choice. This builder deliberately does not exercise that
// freedom: `shard_cols` still comes from `compute_chunk_cols_with_cap`, byte for
// byte the predecessor's rule, so the emitted layout is identical at every
// scale. Narrower shards are better for the reader — an unframed CSC shard
// decodes whole (`BackedCscReader` takes the block-index scattered path only on
// framed shards), so widening them would transfer this op's cost onto every
// gene-chunk read — and choosing a width is a separate decision that needs its
// own read-side measurement.

use std::collections::VecDeque;
use std::io::Read;

use crate::csr::{CsrError, ScxCsr};
use crate::transpose::{compute_chunk_cols_with_cap, TransposeError};

/// Bytes of a spill block handed to the store in one call.
///
/// Whole blocks, never single records, is what keeps a `SpillStore` down to one
/// open descriptor: at census_1m the builder makes ~11,600 `append` calls, so
/// open-append-close per call costs nothing. The EMFILE trap in
/// `scx-convert/src/h5ad/csc_stream.rs` — one `BufWriter<File>` per bucket, all
/// held for the whole pass, 11,922 of them at 50M cells — cannot arise here.
pub const DEFAULT_BLOCK_BYTES: usize = 1 << 20;

/// Buckets to aim for. Over-provisioning costs only row-block headers;
/// under-provisioning raises the emit's resident set.
pub const DEFAULT_TARGET_BUCKETS: usize = 64;

/// Hard ceiling on the bucket count.
///
/// Bounded by header overhead and per-bucket block slack (`n_buckets *
/// block_bytes`), **not** by `RLIMIT_NOFILE` — see [`DEFAULT_BLOCK_BYTES`].
/// It is a compile-time constant on purpose: a probed descriptor limit would
/// make the emitted layout host-dependent, and the layout is pinned by a
/// golden.
pub const MAX_BUCKETS: usize = 256;

/// `u32` row + `f32` value: the payload cost of one nonzero in a spill record.
pub const SPILL_BYTES_PER_NNZ: usize = 8;

/// Bytes of a row block's header: `u32` row + `u32` count.
const ROW_BLOCK_HEADER_BYTES: usize = 8;

#[derive(Debug, thiserror::Error)]
pub enum CscBuilderError {
    #[error("shape mismatch: shard n_cols {shard_cols} != expected {expected_cols}")]
    ShapeMismatch {
        shard_cols: usize,
        expected_cols: usize,
    },
    #[error("push_shard(row_start = {got}) out of order: {expected} rows have been pushed so far")]
    RowStartMismatch { expected: u64, got: u64 },
    #[error("pushed {pushed} rows, builder was constructed for {declared}")]
    RowCountMismatch { pushed: u64, declared: usize },
    #[error("n_rows {0} exceeds i32::MAX: CSC row indices are i32 on disk")]
    RowCountOverflow(usize),
    #[error("n_cols {0} exceeds u32::MAX")]
    ColCountOverflow(usize),
    #[error("column index {col} is outside the matrix's {n_cols} columns")]
    ColumnOutOfRange { col: usize, n_cols: usize },
    #[error(
        "this builder may not spill, and staging needs more than {budget} bytes; \
         supply a spill store or raise the budget"
    )]
    SpillRefused { budget: usize },
    #[error("bucket {bucket} spill is truncated: {detail}")]
    SpillCorrupt { bucket: usize, detail: String },
    #[error("spill I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Transpose(#[from] TransposeError),
    #[error(transparent)]
    Csr(#[from] CsrError),
}

// ---------------------------------------------------------------------------
// Spill stores
// ---------------------------------------------------------------------------

/// Where a bucket's overflow goes.
///
/// The builder hands over whole blocks in stream order and reads each bucket
/// back exactly once, sequentially. An implementation may therefore keep at
/// most one handle open at a time, and must create backing storage **lazily**:
/// a bucket that never overflows must never produce a file, because
/// `reader` returning `Ok(None)` for it is what keeps an all-in-memory build
/// free of disk entirely.
pub trait SpillStore: Send {
    /// Append `block` to `bucket`'s stream. Calls for one bucket arrive in
    /// stream order.
    fn append(&mut self, bucket: usize, block: &[u8]) -> std::io::Result<()>;

    /// Everything appended to `bucket`, in append order. `Ok(None)` iff the
    /// bucket never spilled.
    fn reader(&self, bucket: usize) -> std::io::Result<Option<Box<dyn Read + '_>>>;

    /// Bytes spilled for `bucket`, for reporting.
    fn spilled_bytes(&self, bucket: usize) -> u64;

    /// Whether this store can accept a block at all.
    ///
    /// A capability, not an error: the builder refuses *before* it has
    /// anything to lose, and with a message about the budget rather than one
    /// about I/O.
    fn can_spill(&self) -> bool {
        true
    }
}

/// A store that refuses to spill.
///
/// For callers with nowhere to put a temp file that would rather fail than
/// silently touch disk. Exceeding the staging budget is
/// [`CscBuilderError::SpillRefused`], not a file.
#[derive(Debug, Default)]
pub struct NoSpillStore;

impl SpillStore for NoSpillStore {
    fn append(&mut self, _bucket: usize, _block: &[u8]) -> std::io::Result<()> {
        Err(std::io::Error::other("this builder may not spill"))
    }
    fn can_spill(&self) -> bool {
        false
    }
    fn reader(&self, _bucket: usize) -> std::io::Result<Option<Box<dyn Read + '_>>> {
        Ok(None)
    }
    fn spilled_bytes(&self, _bucket: usize) -> u64 {
        0
    }
}

/// An in-RAM store, for tests.
///
/// Its point is that it makes the spill path *deterministically reachable*
/// without disk: the same input under three `spill_after_bytes` settings —
/// never, always, and a third of the way — must produce identical output, and
/// only the third exercises a column whose rows span the disk→RAM seam.
#[derive(Debug, Default)]
pub struct MemSpillStore {
    buckets: Vec<Vec<u8>>,
}

impl MemSpillStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SpillStore for MemSpillStore {
    fn append(&mut self, bucket: usize, block: &[u8]) -> std::io::Result<()> {
        if self.buckets.len() <= bucket {
            self.buckets.resize_with(bucket + 1, Vec::new);
        }
        self.buckets[bucket].extend_from_slice(block);
        Ok(())
    }
    fn reader(&self, bucket: usize) -> std::io::Result<Option<Box<dyn Read + '_>>> {
        match self.buckets.get(bucket) {
            Some(b) if !b.is_empty() => Ok(Some(Box::new(b.as_slice()))),
            _ => Ok(None),
        }
    }
    fn spilled_bytes(&self, bucket: usize) -> u64 {
        self.buckets.get(bucket).map_or(0, |b| b.len() as u64)
    }
}

// ---------------------------------------------------------------------------
// Configuration and plan
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CscBuilderConfig {
    /// Hard cap on an emitted shard's column width. `0` or `usize::MAX` mean
    /// "no cap", matching `compute_chunk_cols_with_cap`'s `max_cols`.
    pub cols_per_shard: usize,
    /// Feeds `compute_chunk_cols_with_cap`, i.e. it sizes the **shard**. Kept
    /// separate from `spill_after_bytes` because the two bound different
    /// things, and conflating them is what tied layout to the writer's budget.
    pub memory_bytes: usize,
    /// Ceiling on staged bucket bytes during the push phase. The realized
    /// bound is this plus `n_buckets * block_bytes` of block slack.
    pub spill_after_bytes: usize,
    /// Buckets to aim for; capped at [`MAX_BUCKETS`] and at the shard count.
    pub target_buckets: usize,
    /// Staging block size, and the unit handed to the store.
    pub block_bytes: usize,
}

impl Default for CscBuilderConfig {
    fn default() -> Self {
        Self {
            cols_per_shard: 5000,
            memory_bytes: 4 * 1024 * 1024 * 1024,
            spill_after_bytes: 2 * 1024 * 1024 * 1024,
            target_buckets: DEFAULT_TARGET_BUCKETS,
            block_bytes: DEFAULT_BLOCK_BYTES,
        }
    }
}

/// One emitted CSC shard's extent, known exactly before any bytes are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CscShardSpec {
    pub col_start: usize,
    pub col_end: usize,
    pub nnz: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CscBuilderStats {
    pub nnz: u64,
    pub n_buckets: usize,
    pub peak_in_memory_bytes: u64,
    pub spilled_bytes: u64,
    /// First column whose emitted rows were **not** strictly increasing, i.e.
    /// the source carried a duplicate `(row, col)`.
    ///
    /// Reported rather than repaired: coalescing would move bytes relative to
    /// the predecessor, which accepts such input and emits both entries. But
    /// `scx-gpu`'s `validate_csc` rejects a non-strict sidecar under
    /// `checks.sorted`, so failing at build time beats failing at DE time and
    /// the caller is given what it needs to warn.
    pub first_non_strict_column: Option<usize>,
}

// ---------------------------------------------------------------------------
// Bucket layout
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct Layout {
    shard_cols: usize,
    n_shards: usize,
    n_buckets: usize,
    shards_per_bucket: usize,
    bucket_cols: usize,
}

impl Layout {
    fn plan(n_rows: usize, n_cols: usize, cfg: &CscBuilderConfig) -> Result<Self, TransposeError> {
        // UNCHANGED from the predecessor: this is the shard-width rule, and
        // keeping it is what makes the output byte-identical.
        let shard_cols = compute_chunk_cols_with_cap(n_rows, cfg.memory_bytes, cfg.cols_per_shard)?;
        let n_shards = n_cols.div_ceil(shard_cols);
        let target = cfg.target_buckets.clamp(1, MAX_BUCKETS);
        let n_buckets = target.min(n_shards);
        let shards_per_bucket = if n_buckets == 0 {
            0
        } else {
            n_shards.div_ceil(n_buckets)
        };
        let bucket_cols = shards_per_bucket.saturating_mul(shard_cols);
        Ok(Self {
            shard_cols,
            n_shards,
            n_buckets,
            shards_per_bucket,
            bucket_cols,
        })
    }

    #[inline]
    fn bucket_of(&self, col: usize) -> usize {
        // `bucket_cols` is 0 only when there are no buckets, and then no
        // column exists to route.
        debug_assert!(self.bucket_cols > 0);
        (col / self.bucket_cols).min(self.n_buckets - 1)
    }

    fn shard_range(&self, shard: usize, n_cols: usize) -> (usize, usize) {
        let lo = shard.saturating_mul(self.shard_cols).min(n_cols);
        let hi = lo.saturating_add(self.shard_cols).min(n_cols);
        (lo, hi)
    }

    /// The half-open shard range bucket `b` owns.
    fn bucket_shards(&self, bucket: usize) -> (usize, usize) {
        let lo = bucket * self.shards_per_bucket;
        let hi = ((bucket + 1) * self.shards_per_bucket).min(self.n_shards);
        (lo, hi)
    }
}

// ---------------------------------------------------------------------------
// Staging
// ---------------------------------------------------------------------------

/// One bucket's in-RAM staging area: whole sealed blocks plus the block being
/// filled.
///
/// A block is sealed only at a **row-block boundary**, so every sealed block —
/// and hence every byte handed to the store — contains whole row blocks. That
/// is what lets the open row block's count field be patched in place, and what
/// makes a bucket's spill stream parseable by a single sequential walk.
#[derive(Debug, Default)]
struct Bucket {
    sealed: Vec<Vec<u8>>,
    /// Running total of `sealed`, so choosing a spill victim is O(n_buckets)
    /// rather than O(n_buckets x blocks).
    sealed_bytes: usize,
    cur: Vec<u8>,
    /// Offset of the open row block's header within `cur`, and its running
    /// count.
    open: Option<(usize, u32)>,
    last_row: Option<u32>,
}

impl Bucket {
    #[inline]
    fn push(&mut self, row: u32, col: u32, value: f32) {
        if self.open.is_none() || self.last_row != Some(row) {
            self.close_row();
            let at = self.cur.len();
            self.cur.extend_from_slice(&row.to_le_bytes());
            self.cur.extend_from_slice(&0u32.to_le_bytes());
            self.open = Some((at, 0));
            self.last_row = Some(row);
        }
        self.cur.extend_from_slice(&col.to_le_bytes());
        self.cur.extend_from_slice(&value.to_le_bytes());
        if let Some((_, n)) = self.open.as_mut() {
            *n += 1;
        }
    }

    /// Patch the open row block's count. A block with `n == 0` is never
    /// written: a row that contributes nothing to this bucket must cost
    /// nothing, which is precisely why the row is an explicit field rather
    /// than inferred from block position.
    #[inline]
    fn close_row(&mut self) {
        if let Some((at, n)) = self.open.take() {
            debug_assert!(n > 0, "an open row block always holds at least one nonzero");
            self.cur[at + 4..at + 8].copy_from_slice(&n.to_le_bytes());
        }
        self.last_row = None;
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

pub struct CscBuilder {
    n_rows: usize,
    n_cols: usize,
    cfg: CscBuilderConfig,
    layout: Layout,
    store: Box<dyn SpillStore>,
    buckets: Vec<Bucket>,
    col_counts: Vec<u64>,
    /// Buckets written to by the row currently being pushed, and the
    /// `global_row + 1` stamp that dedups them without a hash set.
    touched: Vec<usize>,
    touch_stamp: Vec<u64>,
    rows_pushed: u64,
    nnz: u64,
    in_memory_bytes: usize,
    peak_in_memory_bytes: usize,
}

impl CscBuilder {
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        cfg: CscBuilderConfig,
        store: Box<dyn SpillStore>,
    ) -> Result<Self, CscBuilderError> {
        // The predecessor casts `(row_offset + row) as i32` and lets it wrap
        // negative; `csc_sidecar`'s debug assert catches it in tests and
        // release builds "trust the contract", after which `validate_csc`
        // rejects the file at the GPU or a CPU kernel's `row >= n_obs` guard
        // silently drops the value. This is a deliberate non-identity: a named
        // error instead of silent corruption.
        if n_rows > i32::MAX as usize {
            return Err(CscBuilderError::RowCountOverflow(n_rows));
        }
        if n_cols > u32::MAX as usize {
            return Err(CscBuilderError::ColCountOverflow(n_cols));
        }
        let layout = Layout::plan(n_rows, n_cols, &cfg)?;
        let mut buckets = Vec::new();
        buckets.resize_with(layout.n_buckets, Bucket::default);
        Ok(Self {
            n_rows,
            n_cols,
            cfg,
            layout,
            store,
            buckets,
            col_counts: vec![0u64; n_cols],
            touched: Vec::with_capacity(layout.n_buckets),
            touch_stamp: vec![0u64; layout.n_buckets],
            rows_pushed: 0,
            nnz: 0,
            in_memory_bytes: 0,
            peak_in_memory_bytes: 0,
        })
    }

    /// A builder that may not spill; see [`NoSpillStore`].
    pub fn in_memory(
        n_rows: usize,
        n_cols: usize,
        cfg: CscBuilderConfig,
    ) -> Result<Self, CscBuilderError> {
        Self::new(n_rows, n_cols, cfg, Box::new(NoSpillStore))
    }

    /// Route one CSR shard's nonzeros into their column buckets.
    ///
    /// `row_start` must equal [`Self::rows_pushed`]: shards arrive in ascending
    /// row order and tile `[0, n_rows)`. The predecessor derived the global row
    /// from *slice position* and had no way to notice a permuted or repeated
    /// shard; checking it here is what makes the sink safe for the rewrite ops,
    /// where a row permutation would otherwise shift the whole row axis
    /// silently.
    ///
    /// `csr` must already be canonical — per-row indices strictly increasing,
    /// duplicates summed, explicit zeros dropped. This sink does not
    /// canonicalize, exactly as `write_csc_sidecar` does not.
    ///
    /// Note the one thing a push sink cannot reproduce: the predecessor
    /// validated every shard's `n_cols` up front, before emitting anything.
    /// Here a mismatch can only be reported at the offending push, after
    /// earlier shards have been consumed.
    pub fn push_shard(&mut self, row_start: u64, csr: &ScxCsr) -> Result<(), CscBuilderError> {
        if csr.shape.1 != self.n_cols {
            return Err(CscBuilderError::ShapeMismatch {
                shard_cols: csr.shape.1,
                expected_cols: self.n_cols,
            });
        }
        if row_start != self.rows_pushed {
            return Err(CscBuilderError::RowStartMismatch {
                expected: self.rows_pushed,
                got: row_start,
            });
        }
        let n_shard_rows = csr.n_rows();
        if n_shard_rows == 0 {
            return Ok(());
        }
        // A shard whose columns all lie outside `[0, n_cols)` cannot happen on
        // a validated `ScxCsr`, but `new_unchecked` does not check it and a
        // panic in a reader is against the conventions.
        let base = self.rows_pushed;
        for row in 0..n_shard_rows {
            let global_row = (base + row as u64) as u32;
            let start = csr.indptr[row] as usize;
            let end = csr.indptr[row + 1] as usize;
            for j in start..end {
                let col = csr.indices[j] as usize;
                if col >= self.n_cols {
                    return Err(CscBuilderError::ColumnOutOfRange {
                        col,
                        n_cols: self.n_cols,
                    });
                }
                self.col_counts[col] += 1;
                let b = self.layout.bucket_of(col);
                // Only the buckets this row actually wrote to can have crossed
                // the block threshold. Revisiting all of them per row, and
                // re-summing each one's sealed list to do it, is
                // O(n_rows x n_buckets x blocks) — 1e6 x 64 x growing at
                // census scale, which would cost more than the transpose it
                // replaces. A row-tagged stamp keeps it O(touched).
                if self.touch_stamp[b] != global_row as u64 + 1 {
                    self.touch_stamp[b] = global_row as u64 + 1;
                    self.touched.push(b);
                }
                let before = self.buckets[b].cur.len();
                self.buckets[b].push(global_row, col as u32, csr.data[j]);
                self.in_memory_bytes += self.buckets[b].cur.len() - before;
            }
            self.seal_and_maybe_spill()?;
        }
        self.rows_pushed += n_shard_rows as u64;
        self.nnz += csr.nnz() as u64;
        Ok(())
    }

    /// Seal any bucket whose open block has reached `block_bytes`, then spill
    /// the largest buckets until the staged total is back under
    /// `spill_after_bytes`.
    fn seal_and_maybe_spill(&mut self) -> Result<(), CscBuilderError> {
        let block_bytes = self.cfg.block_bytes.max(1);
        for &i in &self.touched {
            let bucket = &mut self.buckets[i];
            if bucket.cur.len() >= block_bytes {
                // Seal only at a row-block boundary: close the open row first,
                // so every byte handed to the store holds whole row blocks and
                // the count field is never patched after it has left.
                bucket.close_row();
                let sealed = std::mem::take(&mut bucket.cur);
                bucket.sealed_bytes += sealed.len();
                bucket.sealed.push(sealed);
            }
        }
        self.touched.clear();
        self.peak_in_memory_bytes = self.peak_in_memory_bytes.max(self.in_memory_bytes);

        while self.in_memory_bytes > self.cfg.spill_after_bytes {
            // Spill the largest bucket, chosen against one *global* counter. A
            // per-bucket share would mis-size under column-nnz skew, which is
            // the defect `scx_convert::budget`'s `CSC_BUCKET_SHARE` row already
            // documents for the other transpose: `nnz_per_obs` is a mean, so a
            // high-depth bucket loads well past its share.
            let Some((v, freed)) = self
                .buckets
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.sealed_bytes))
                .max_by_key(|&(_, n)| n)
                .filter(|&(_, n)| n > 0)
            else {
                // Nothing sealed anywhere: every bucket holds only a partial
                // block. That is the declared slack — the realized bound is
                // `spill_after_bytes + n_buckets * block_bytes` — so there is
                // nothing further to give.
                break;
            };
            if !self.store.can_spill() {
                return Err(CscBuilderError::SpillRefused {
                    budget: self.cfg.spill_after_bytes,
                });
            }
            for block in std::mem::take(&mut self.buckets[v].sealed) {
                self.store.append(v, &block)?;
            }
            self.buckets[v].sealed_bytes = 0;
            self.in_memory_bytes -= freed;
        }
        Ok(())
    }

    pub fn rows_pushed(&self) -> u64 {
        self.rows_pushed
    }

    pub fn nnz_pushed(&self) -> u64 {
        self.nnz
    }

    /// Exact per-column nnz so far.
    pub fn column_counts(&self) -> &[u64] {
        &self.col_counts
    }

    pub fn finish(mut self) -> Result<CscEmitter, CscBuilderError> {
        if self.rows_pushed != self.n_rows as u64 {
            return Err(CscBuilderError::RowCountMismatch {
                pushed: self.rows_pushed,
                declared: self.n_rows,
            });
        }
        for b in &mut self.buckets {
            b.close_row();
        }
        let spilled_bytes = (0..self.layout.n_buckets)
            .map(|b| self.store.spilled_bytes(b))
            .sum();

        // The plan, exact, before a byte is read back.
        let mut plan = Vec::with_capacity(self.layout.n_shards);
        for s in 0..self.layout.n_shards {
            let (lo, hi) = self.layout.shard_range(s, self.n_cols);
            plan.push(CscShardSpec {
                col_start: lo,
                col_end: hi,
                nnz: self.col_counts[lo..hi].iter().sum(),
            });
        }

        Ok(CscEmitter {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            layout: self.layout,
            store: self.store,
            buckets: self.buckets,
            col_counts: self.col_counts,
            plan,
            next_bucket: 0,
            pending: VecDeque::new(),
            stats: CscBuilderStats {
                nnz: self.nnz,
                n_buckets: self.layout.n_buckets,
                peak_in_memory_bytes: self.peak_in_memory_bytes as u64,
                spilled_bytes,
                first_non_strict_column: None,
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

/// One built shard, at on-disk widths.
struct Built {
    col_start: u64,
    indptr: Vec<u64>,
    indices: Vec<u32>,
    data: Vec<f32>,
}

pub struct CscEmitter {
    n_rows: usize,
    n_cols: usize,
    layout: Layout,
    store: Box<dyn SpillStore>,
    buckets: Vec<Bucket>,
    col_counts: Vec<u64>,
    plan: Vec<CscShardSpec>,
    next_bucket: usize,
    pending: VecDeque<Built>,
    stats: CscBuilderStats,
}

impl CscEmitter {
    /// Every shard this emitter will produce, in order, with exact nnz.
    pub fn plan(&self) -> &[CscShardSpec] {
        &self.plan
    }

    pub fn stats(&self) -> &CscBuilderStats {
        &self.stats
    }

    /// Fill `indptr`/`indices`/`data` with the next shard at **on-disk**
    /// widths, returning its `col_start`, or `None` when the plan is drained.
    ///
    /// On-disk widths rather than [`crate::CscArrays`]' `i64`/`i32`: the
    /// predecessor's caller rebuilt `csc_indptr_u64` and `csc_indices_u32` from
    /// every chunk, two full-length copies per shard that the allocation table
    /// names as a reason the CSC budget row could not be enforced.
    ///
    /// The caller's buffers are swapped in, not appended to, so their previous
    /// allocations are recycled rather than reused — the saving here is the
    /// copy, not the allocation.
    pub fn next_shard_into(
        &mut self,
        indptr: &mut Vec<u64>,
        indices: &mut Vec<u32>,
        data: &mut Vec<f32>,
    ) -> Result<Option<u64>, CscBuilderError> {
        while self.pending.is_empty() {
            if self.next_bucket >= self.layout.n_buckets {
                return Ok(None);
            }
            self.build_bucket(self.next_bucket)?;
            self.next_bucket += 1;
        }
        let built = self.pending.pop_front().expect("non-empty");
        *indptr = built.indptr;
        *indices = built.indices;
        *data = built.data;
        Ok(Some(built.col_start))
    }

    /// Build every shard bucket `b` covers, in one pass over its records.
    ///
    /// One pass rather than one per shard: each shard's columns are a disjoint
    /// slice of the bucket's range, so a single walk fills them all without any
    /// re-read of the spill.
    fn build_bucket(&mut self, b: usize) -> Result<(), CscBuilderError> {
        let (shard_lo, shard_hi) = self.layout.bucket_shards(b);
        if shard_lo >= shard_hi {
            return Ok(());
        }
        let bucket_col_lo = self.plan[shard_lo].col_start;
        let bucket_col_hi = self.plan[shard_hi - 1].col_end;

        // One prefix sum per shard, from counts that are already exact.
        let mut built: Vec<Built> = Vec::with_capacity(shard_hi - shard_lo);
        for s in shard_lo..shard_hi {
            let spec = self.plan[s];
            let width = spec.col_end - spec.col_start;
            let mut indptr = Vec::with_capacity(width + 1);
            indptr.push(0u64);
            let mut cumsum = 0u64;
            for c in spec.col_start..spec.col_end {
                cumsum += self.col_counts[c];
                indptr.push(cumsum);
            }
            let nnz = cumsum as usize;
            built.push(Built {
                col_start: spec.col_start as u64,
                indptr,
                indices: vec![0u32; nnz],
                data: vec![0.0f32; nnz],
            });
        }
        // Per-column write cursors, flat across the whole bucket range.
        let mut cursor = vec![0u64; bucket_col_hi - bucket_col_lo];
        let shard_cols = self.layout.shard_cols;

        {
            let plan = &self.plan;
            let built = &mut built;
            let cursor = &mut cursor;
            let mut scatter = |row: u32, payload: &[u8]| {
                let (recs, tail) = payload.as_chunks::<SPILL_BYTES_PER_NNZ>();
                debug_assert!(tail.is_empty(), "a row block's payload is 8 B per nonzero");
                for rec in recs {
                    let col = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as usize;
                    let val = f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
                    // Which shard owns this column. `shard_cols` is uniform, so
                    // this is a divide, not a search; the `.min` guards the
                    // `usize::MAX` width the zero-row case produces.
                    let s = (col / shard_cols).min(plan.len() - 1);
                    debug_assert!(s >= shard_lo && s < shard_hi);
                    let target = &mut built[s - shard_lo];
                    let lc = col - plan[s].col_start;
                    let flat = col - bucket_col_lo;
                    let dest = (target.indptr[lc] + cursor[flat]) as usize;
                    target.indices[dest] = row;
                    target.data[dest] = val;
                    cursor[flat] += 1;
                }
            };
            drain_bucket(&*self.store, &mut self.buckets[b], b, &mut scatter)?;
        }

        for (i, t) in built.iter().enumerate() {
            let s = shard_lo + i;
            let spec = self.plan[s];
            debug_assert_eq!(
                t.indices.len(),
                spec.nnz as usize,
                "shard {s} filled {} of {} slots",
                t.indices.len(),
                spec.nnz
            );
            for lc in 0..(spec.col_end - spec.col_start) {
                let flat = spec.col_start + lc - bucket_col_lo;
                debug_assert_eq!(
                    cursor[flat],
                    t.indptr[lc + 1] - t.indptr[lc],
                    "column {} filled {} of {} slots",
                    spec.col_start + lc,
                    cursor[flat],
                    t.indptr[lc + 1] - t.indptr[lc]
                );
                let (a, z) = (t.indptr[lc] as usize, t.indptr[lc + 1] as usize);
                if let Some(off) = t.indices[a..z].windows(2).position(|w| w[0] >= w[1]) {
                    let col = spec.col_start + lc;
                    let cur = self.stats.first_non_strict_column;
                    if cur.is_none_or(|c| col < c) {
                        self.stats.first_non_strict_column = Some(col);
                    }
                    debug_assert!(
                        false,
                        "column {col} rows are not strictly increasing at {off}: \
                         the source carried a duplicate (row, col)"
                    );
                }
            }
        }

        self.pending.extend(built);
        Ok(())
    }

    /// Number of rows every emitted shard spans.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Total columns across the plan.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }
}

/// Walk `bucket`'s records in append order: the spilled prefix, then the RAM
/// tail. The seam between them is an append-order seam, so a column whose rows
/// span it still comes out ascending.
fn drain_bucket(
    store: &dyn SpillStore,
    bucket: &mut Bucket,
    index: usize,
    f: &mut dyn FnMut(u32, &[u8]),
) -> Result<(), CscBuilderError> {
    if let Some(rd) = store.reader(index)? {
        let mut rd = std::io::BufReader::with_capacity(256 * 1024, rd);
        let mut header = [0u8; ROW_BLOCK_HEADER_BYTES];
        let mut payload: Vec<u8> = Vec::new();
        loop {
            match read_full(&mut rd, &mut header)? {
                0 => break,
                n if n < ROW_BLOCK_HEADER_BYTES => {
                    return Err(CscBuilderError::SpillCorrupt {
                        bucket: index,
                        detail: format!("{n}-byte partial row-block header at end of stream"),
                    })
                }
                _ => {}
            }
            let row = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
            let n = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;
            payload.resize(n * SPILL_BYTES_PER_NNZ, 0);
            let got = read_full(&mut rd, &mut payload)?;
            if got != payload.len() {
                return Err(CscBuilderError::SpillCorrupt {
                    bucket: index,
                    detail: format!(
                        "row {row} declares {n} nonzeros ({} bytes) but only {got} remain",
                        payload.len()
                    ),
                });
            }
            f(row, &payload);
        }
    }
    // The RAM tail: sealed blocks first, then the block still being filled.
    // Both hold whole row blocks, so both parse as slices with no copy.
    bucket.close_row();
    let sealed = std::mem::take(&mut bucket.sealed);
    for block in &sealed {
        walk_blocks(block, index, f)?;
    }
    let cur = std::mem::take(&mut bucket.cur);
    walk_blocks(&cur, index, f)
}

fn walk_blocks(
    block: &[u8],
    index: usize,
    f: &mut dyn FnMut(u32, &[u8]),
) -> Result<(), CscBuilderError> {
    let mut at = 0usize;
    while at < block.len() {
        if at + ROW_BLOCK_HEADER_BYTES > block.len() {
            return Err(CscBuilderError::SpillCorrupt {
                bucket: index,
                detail: "partial row-block header in a staged block".to_string(),
            });
        }
        let row = u32::from_le_bytes(block[at..at + 4].try_into().expect("4 bytes"));
        let n = u32::from_le_bytes(block[at + 4..at + 8].try_into().expect("4 bytes")) as usize;
        let lo = at + ROW_BLOCK_HEADER_BYTES;
        let hi = lo + n * SPILL_BYTES_PER_NNZ;
        if hi > block.len() {
            return Err(CscBuilderError::SpillCorrupt {
                bucket: index,
                detail: format!("row {row} declares {n} nonzeros past the end of a staged block"),
            });
        }
        f(row, &block[lo..hi]);
        at = hi;
    }
    Ok(())
}

/// `Read::read_exact` that reports a short final read instead of erroring, so a
/// clean end of stream is distinguishable from a truncated one.
fn read_full(rd: &mut impl Read, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut filled = 0usize;
    while filled < buf.len() {
        match rd.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(filled)
}

#[cfg(test)]
#[path = "csc_builder_tests.rs"]
mod tests;
