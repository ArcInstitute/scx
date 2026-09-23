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
/// Bounded by header overhead and per-bucket block slack
/// (`2 * n_buckets * block_capacity`, the expression `CscBuilderConfig`
/// documents and `staged_bytes_never_exceed_the_declared_bound` asserts),
/// **not** by `RLIMIT_NOFILE` — see [`DEFAULT_BLOCK_BYTES`].
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
pub trait SpillStore: Send + Sync {
    /// Append `block` to `bucket`'s stream. Calls for one bucket arrive in
    /// stream order.
    fn append(&mut self, bucket: usize, block: &[u8]) -> std::io::Result<()>;

    /// [`Self::append`] each of `blocks`, in order.
    ///
    /// Every spill hands over all of a bucket's sealed blocks at once, so a
    /// store whose append has a per-call cost — a file store opening and
    /// closing its file, which on a network filesystem is the dominant cost
    /// of a spill — can pay it once per spill instead of once per block.
    fn append_all(&mut self, bucket: usize, blocks: &[Vec<u8>]) -> std::io::Result<()> {
        for block in blocks {
            self.append(bucket, block)?;
        }
        Ok(())
    }

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
    /// Ceiling on staged bucket bytes during the push phase.
    ///
    /// The realized bound is this plus **`2 * n_buckets * block_capacity`**,
    /// where `block_capacity` is `block_bytes` plus one maximal row block.
    /// Two blocks per bucket, because only *sealed* capacity is counted
    /// against this ceiling: every bucket also holds one live `cur` block that
    /// the counter cannot see, and the bucket being sealed holds its freshly
    /// sealed block as well before the spill loop gets to release it. The
    /// bound is deliberately the loose form — `n_buckets + 1` blocks is the
    /// tight one — so it stays true whichever bucket the cut lands in.
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

/// How columns map to shards and to buckets.
///
/// Buckets are uniform `bucket_cols`-wide column ranges and never straddle a
/// shard boundary, in one of two shapes:
///
/// * **coarse** — a bucket holds `shards_per_bucket >= 1` whole shards. The
///   only shape before the parallel push, and still the one used whenever
///   there are at least `target_buckets` shards.
/// * **fine** — a shard holds `buckets_per_shard > 1` buckets. Used, with the
///   `parallel` feature, when there are fewer shards than `target_buckets`:
///   the push runs one task per bucket and the emit drains a shard's buckets
///   in parallel, so a 13-shard layout (61k genes at 5,000 columns a shard)
///   otherwise caps both at 13 tasks, unevenly loaded. `bucket_cols` must
///   divide `shard_cols` so no bucket straddles a boundary; a shard width with
///   no divisor in range falls back to coarse.
///
/// Neither shape reaches the emitted bytes: `shard_cols` is the one rule that
/// does, and it is computed first and never changed by the bucket choice.
#[derive(Debug, Clone, Copy)]
struct Layout {
    shard_cols: usize,
    n_shards: usize,
    n_buckets: usize,
    shards_per_bucket: usize,
    buckets_per_shard: usize,
    bucket_cols: usize,
}

impl Layout {
    fn plan(n_rows: usize, n_cols: usize, cfg: &CscBuilderConfig) -> Result<Self, TransposeError> {
        // UNCHANGED from the predecessor: this is the shard-width rule, and
        // keeping it is what makes the output byte-identical.
        let shard_cols = compute_chunk_cols_with_cap(n_rows, cfg.memory_bytes, cfg.cols_per_shard)?;
        let n_shards = n_cols.div_ceil(shard_cols);
        let target = cfg.target_buckets.clamp(1, MAX_BUCKETS);
        let shards_per_bucket = {
            let want = target.min(n_shards);
            if want == 0 {
                0
            } else {
                n_shards.div_ceil(want)
            }
        };
        // Recomputed from `shards_per_bucket`, NOT left at `target.min(n_shards)`.
        // Rounding the shards-per-bucket up means fewer buckets are actually
        // reachable: at `n_shards = 65, target = 64` it is 2, so `bucket_of`
        // tops out at bucket 32 and buckets 33..64 can never be routed to.
        // Keeping the higher count allocated one live block per phantom bucket
        // (31 MiB there, 6 MiB at the 173-shard census layout) and charged it
        // against the staging budget, and `bucket_shards` handed those indices
        // an inverted range.
        let n_buckets = if shards_per_bucket == 0 {
            0
        } else {
            n_shards.div_ceil(shards_per_bucket)
        };
        let bucket_cols = shards_per_bucket.saturating_mul(shard_cols);
        let coarse = Self {
            shard_cols,
            n_shards,
            n_buckets,
            shards_per_bucket,
            buckets_per_shard: 1,
            bucket_cols,
        };
        if !cfg!(feature = "parallel") || n_shards == 0 || n_shards >= target {
            return Ok(coarse);
        }
        // Fine: split each shard into `d` equal buckets, `d` the largest
        // divisor of the shard width within `target / n_shards`. One shard
        // has no interior boundary to respect, so any width works there.
        let per_shard = target / n_shards;
        let width = shard_cols.min(n_cols);
        let bucket_cols = if n_shards == 1 {
            width.div_ceil(per_shard)
        } else {
            match (2..=per_shard).rev().find(|d| shard_cols % d == 0) {
                Some(d) => shard_cols / d,
                None => return Ok(coarse),
            }
        };
        if bucket_cols == 0 || bucket_cols >= width {
            return Ok(coarse);
        }
        let n_buckets = n_cols.div_ceil(bucket_cols);
        let buckets_per_shard = if n_shards == 1 {
            n_buckets
        } else {
            shard_cols / bucket_cols
        };
        Ok(Self {
            shard_cols,
            n_shards,
            n_buckets,
            shards_per_bucket: 1,
            buckets_per_shard,
            bucket_cols,
        })
    }

    /// Emit units: a coarse bucket (its whole shards), or a fine shard (its
    /// buckets). Each is `(shard range, bucket range)`, half-open.
    fn n_groups(&self) -> usize {
        if self.buckets_per_shard > 1 {
            self.n_shards
        } else {
            self.n_buckets
        }
    }

    fn group(&self, g: usize) -> ((usize, usize), (usize, usize)) {
        if self.buckets_per_shard > 1 {
            let lo = g * self.buckets_per_shard;
            (
                (g, g + 1),
                (lo, (lo + self.buckets_per_shard).min(self.n_buckets)),
            )
        } else {
            (self.bucket_shards(g), (g, g + 1))
        }
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
    /// Running total of `sealed`'s **capacity**, so choosing a spill victim is
    /// O(n_buckets) rather than O(n_buckets x blocks).
    sealed_bytes: usize,
    cur: Vec<u8>,
    /// Offset of the open row block's header within `cur`, and its running
    /// count.
    open: Option<(usize, u32)>,
    last_row: Option<u32>,
}

impl Bucket {
    /// A fresh block, pre-sized so it never reallocates.
    ///
    /// Load-bearing for the budget, not a micro-optimisation: a `Vec` grown by
    /// `extend_from_slice` doubles, so a block sealed at `block_bytes` of
    /// *length* can hold up to twice that in *capacity* — and capacity is what
    /// the process is charged for. Counting length made the enforced bound up
    /// to 2x looser than it declared, which measurement caught as a 1.10x
    /// peak regression on a dataset small enough never to spill.
    fn with_capacity(cap: usize) -> Self {
        Self {
            sealed: Vec::new(),
            sealed_bytes: 0,
            cur: Vec::with_capacity(cap),
            open: None,
            last_row: None,
        }
    }

    /// Append one nonzero to the open row block, starting one if needed.
    ///
    /// A row may span several blocks: `seal_bucket` calls `close_row`, so the
    /// next push after a seal opens a fresh block even mid-row. That is what
    /// bounds a **duplicate-bearing** row — `block_capacity` reserves one
    /// record per column the bucket owns, which is exact for canonical input,
    /// but `ScxCsr::new` does not reject duplicate coordinates and
    /// `run_build_csc` deliberately accepts non-canonical pre-v3 input.
    ///
    /// Two row blocks carrying the same `row` are fine: the reader walks them
    /// in append order, which is still scan order, so the per-column scatter
    /// order is unchanged.
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

    /// [`Self::push`] for a run of one row's nonzeros, `cols[k]` / `vals[k]`,
    /// returning how many it took.
    ///
    /// Takes records up to and including the one that brings `cur` to
    /// `block_bytes` — exactly where [`Self::push`]'s caller would seal — so a
    /// caller that seals whenever `cur.len() >= block_bytes` after each call
    /// produces the same bytes, in the same blocks, as one record at a time.
    /// The difference is one resize per run instead of four `extend`s per
    /// record, which is what made the per-record path the parallel push's
    /// bottleneck.
    #[cfg(feature = "parallel")]
    #[inline]
    fn push_run(&mut self, row: u32, cols: &[i32], vals: &[f32], block_bytes: usize) -> usize {
        debug_assert!(!cols.is_empty() && cols.len() == vals.len());
        if self.open.is_none() || self.last_row != Some(row) {
            self.close_row();
            let at = self.cur.len();
            self.cur.extend_from_slice(&row.to_le_bytes());
            self.cur.extend_from_slice(&0u32.to_le_bytes());
            self.open = Some((at, 0));
            self.last_row = Some(row);
        }
        // Records until the one that reaches `block_bytes`; at least one,
        // since the seal test only ever runs after a record.
        let room = block_bytes.saturating_sub(self.cur.len());
        let take = cols.len().min(room.div_ceil(SPILL_BYTES_PER_NNZ).max(1));
        let old = self.cur.len();
        self.cur.resize(old + take * SPILL_BYTES_PER_NNZ, 0);
        for ((rec, &col), &val) in self.cur[old..]
            .as_chunks_mut::<SPILL_BYTES_PER_NNZ>()
            .0
            .iter_mut()
            .zip(&cols[..take])
            .zip(&vals[..take])
        {
            rec[..4].copy_from_slice(&(col as u32).to_le_bytes());
            rec[4..].copy_from_slice(&val.to_le_bytes());
        }
        if let Some((_, n)) = self.open.as_mut() {
            *n += take as u32;
        }
        take
    }

    /// Seal the open block and start a fresh one of `block_capacity`, closing
    /// the row block first so every byte handed to the store holds whole row
    /// blocks and no count field is patched after it has left. Returns the
    /// capacity the sealed block is charged at.
    ///
    /// Charges what was actually allocated, not `block_capacity`. The two are
    /// equal for canonical input, and sealing inside the row keeps them close
    /// even for duplicate-bearing input, but a single row block whose
    /// header-plus-payload exceeds the reserve can still grow the vec once.
    fn seal(&mut self, block_capacity: usize) -> usize {
        self.close_row();
        let sealed = std::mem::replace(&mut self.cur, Vec::with_capacity(block_capacity));
        let grew = sealed.capacity();
        self.sealed_bytes += grew;
        self.sealed.push(sealed);
        grew
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
    /// Bytes every staging block is allocated at, and therefore exactly what
    /// each one costs. `block_bytes` plus the largest single row block, so a
    /// block sealed at `block_bytes` can always absorb the row that crossed
    /// the threshold without reallocating.
    block_capacity: usize,
    layout: Layout,
    store: Box<dyn SpillStore>,
    buckets: Vec<Bucket>,
    col_counts: Vec<u64>,
    /// Buckets written to by the row currently being pushed, and the
    /// `global_row + 1` stamp that dedups them without a hash set.
    rows_pushed: u64,
    nnz: u64,
    in_memory_bytes: usize,
    peak_in_memory_bytes: usize,
    /// Shards with fewer nonzeros than this are pushed serially; see
    /// [`PARALLEL_PUSH_MIN_NNZ`]. A field rather than the constant so the
    /// tests can force either path on the same input.
    #[cfg_attr(not(feature = "parallel"), allow(dead_code))]
    parallel_min_nnz: usize,
}

/// Below this many nonzeros a shard is routed serially: the parallel push
/// scans every row once per bucket, which on a small shard costs more in
/// task overhead than it saves.
pub const PARALLEL_PUSH_MIN_NNZ: usize = 1 << 16;

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
        // The widest row block a bucket can ever hold: one header plus one
        // nonzero for every column the bucket owns.
        let max_row_block =
            ROW_BLOCK_HEADER_BYTES + SPILL_BYTES_PER_NNZ * layout.bucket_cols.min(n_cols.max(1));
        let block_capacity = cfg.block_bytes.max(1).saturating_add(max_row_block);
        let mut buckets = Vec::new();
        buckets.resize_with(layout.n_buckets, || Bucket::with_capacity(block_capacity));
        // Every bucket holds one live block from the start; charge it.
        let in_memory_bytes = layout.n_buckets.saturating_mul(block_capacity);
        Ok(Self {
            n_rows,
            n_cols,
            cfg,
            block_capacity,
            layout,
            store,
            buckets,
            col_counts: vec![0u64; n_cols],
            rows_pushed: 0,
            nnz: 0,
            in_memory_bytes,
            peak_in_memory_bytes: in_memory_bytes,
            parallel_min_nnz: PARALLEL_PUSH_MIN_NNZ,
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
    /// `row_start` must equal the number of rows pushed so far: shards arrive
    /// in ascending row order and tile `[0, n_rows)`. The predecessor derived the global row
    /// from *slice position* and had no way to notice a permuted or repeated
    /// shard; checking it here is what makes the sink safe for the rewrite ops,
    /// where a row permutation would otherwise shift the whole row axis
    /// silently.
    ///
    /// `csr` need **not** be canonical, and this sink does not canonicalize —
    /// exactly as `write_csc_sidecar` does not. A duplicate `(row, col)` pair
    /// is preserved as two nonzeros, adjacent and in `j` order, and a row
    /// whose `indices` are unsorted is routed in `j` order too; neither can
    /// disturb any *column's* internal order, because scan order restricted to
    /// one column is what the emit reproduces either way. Coalescing here
    /// would move bytes that the predecessor left alone, and `run_build_csc`
    /// deliberately accepts pre-v3 input that has them. What a duplicate does
    /// cost is a sidecar that `scx-gpu`'s `validate_csc` will reject under
    /// `checks.sorted`, so the first column carrying one is reported as
    /// [`CscBuilderStats::first_non_strict_column`] for the caller to warn on.
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
        #[cfg(feature = "parallel")]
        if self.layout.n_buckets > 1
            && csr.nnz() >= self.parallel_min_nnz
            && rows_route_by_search(csr, self.n_cols)
        {
            self.push_shard_parallel(csr)?;
            self.rows_pushed += n_shard_rows as u64;
            self.nnz += csr.nnz() as u64;
            return Ok(());
        }
        // A shard whose columns all lie outside `[0, n_cols)` cannot happen on
        // a validated `ScxCsr`, but `new_unchecked` does not check it and a
        // panic in a reader is against the conventions.
        let base = self.rows_pushed;
        let block_bytes = self.cfg.block_bytes.max(1);
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
                self.buckets[b].push(global_row, col as u32, csr.data[j]);
                // Seal AND SPILL here, at the push that crossed the
                // threshold — this is the only place either happens.
                //
                // An earlier revision also swept every bucket the row had
                // touched once the row ended, which is what the `touched` /
                // `touch_stamp` bookkeeping existed for. That sweep was dead
                // by the time this check moved inside the row: `push` is the
                // only thing that grows `cur`, and every push that takes a
                // bucket to `block_bytes` seals it right here, so no bucket
                // can still be at or over the threshold when the row ends.
                // Its spill call was dead for the same reason —
                // `in_memory_bytes` only rises in `seal_bucket`, and every
                // seal is followed by the spill below. Deleting it takes a
                // branch and a `Vec` push off the per-nonzero path, which at
                // census_1m ran 1.4e9 times.
                //
                // Sealing alone would not be enough: it just moves bytes from
                // `cur` into `sealed`. A duplicate-heavy row sealed hundreds
                // of blocks and held every one until the row ended, measured
                // at 45,760 bytes against a 416-byte bound. `close_row`
                // inside `seal_bucket` is what splits the row block.
                if self.buckets[b].cur.len() >= block_bytes {
                    self.seal_bucket(b);
                    self.spill_until_under_budget()?;
                }
            }
        }
        self.rows_pushed += n_shard_rows as u64;
        self.nnz += csr.nnz() as u64;
        Ok(())
    }

    /// Seal a bucket's open block, closing the row block first so every byte
    /// handed to the store holds whole row blocks and no count field is
    /// patched after it has left.
    fn seal_bucket(&mut self, i: usize) {
        // The retained block keeps its real capacity and a fresh one replaces
        // it, so the bucket's cost rises by exactly the retained block.
        self.in_memory_bytes += self.buckets[i].seal(self.block_capacity);
    }

    /// Spill sealed blocks, largest bucket first, until the staged total is
    /// back under `spill_after_bytes`.
    ///
    /// Called from `push_shard`, at the push that sealed a block. That it
    /// runs *inside* the row is what bounds a duplicate-heavy row: sealing
    /// alone only moves bytes from `cur` into `sealed`, so a single row with
    /// thousands of copies of one coordinate sealed hundreds of blocks and
    /// held every one of them until the row ended — measured at 45,760 bytes
    /// against a 416-byte bound before this split.
    fn spill_until_under_budget(&mut self) -> Result<(), CscBuilderError> {
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
                // block, which is the declared slack
                // (`spill_after_bytes + 2 * n_buckets * block_capacity`, the
                // expression `CscBuilderConfig` documents and
                // `staged_bytes_never_exceed_the_declared_bound` asserts), so
                // there is nothing further to give. This `break` means "no
                // sealed block left to spill", not "the bound holds".
                break;
            };
            if !self.store.can_spill() {
                return Err(CscBuilderError::SpillRefused {
                    budget: self.cfg.spill_after_bytes,
                });
            }
            self.store
                .append_all(v, &std::mem::take(&mut self.buckets[v].sealed))?;
            self.buckets[v].sealed_bytes = 0;
            self.in_memory_bytes -= freed;
        }
        Ok(())
    }

    /// [`Self::push_shard`]'s routing, one task per bucket.
    ///
    /// # Same bytes
    ///
    /// Each task walks every row of the shard in order and copies the row's
    /// entries in its bucket's column range, found by binary search — which is
    /// why [`rows_route_by_search`] must hold. A bucket therefore receives
    /// exactly the records, in exactly the order, the serial scan gives it, and
    /// it seals at the same records, because the seal test is the bucket's own
    /// block length. Everything a bucket holds is thus identical; what differs
    /// is only *when* sealed blocks go to the store, and a spill never changes
    /// the emitted bytes (the drain reads a bucket's spill stream, then its
    /// sealed blocks, then `cur`, and every spill takes a prefix of them).
    ///
    /// # Same bound
    ///
    /// The serial push spills the *largest* bucket at the seal that crossed
    /// the ceiling. Buckets here are owned by different tasks, so a task
    /// that crosses the ceiling spills its **own** sealed blocks instead —
    /// always including the one it just sealed. Every byte over the ceiling
    /// is then a block some task has sealed and is about to spill, at most
    /// one per bucket at any instant, which is inside the
    /// `2 * n_buckets * block_capacity` slack [`CscBuilderConfig`] declares
    /// (the other `n_buckets` blocks being the live `cur` ones). The victim
    /// choice moves which bytes reach disk, never how many may stay in RAM.
    #[cfg(feature = "parallel")]
    fn push_shard_parallel(&mut self, csr: &ScxCsr) -> Result<(), CscBuilderError> {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Mutex;

        // `n_rows <= i32::MAX` was checked at construction, so no global row
        // overflows `u32`.
        let base = self.rows_pushed as u32;
        let n_rows = csr.n_rows();
        let n_cols = self.n_cols;
        let n_buckets = self.layout.n_buckets;
        let bucket_cols = self.layout.bucket_cols;
        let block_bytes = self.cfg.block_bytes.max(1);
        let block_capacity = self.block_capacity;
        let ceiling = self.cfg.spill_after_bytes;
        let in_memory = AtomicUsize::new(self.in_memory_bytes);
        let peak = AtomicUsize::new(self.peak_in_memory_bytes);
        let store = Mutex::new(std::mem::replace(&mut self.store, Box::new(NoSpillStore)));
        // `bucket_of` clamps to the last bucket, and `Layout::plan` sizes the
        // bucket count so the last one's range ends at or past `n_cols`; the
        // `bucket_cols`-wide chunks of `col_counts` are therefore exactly the
        // buckets' column ranges.
        debug_assert_eq!(n_cols.div_ceil(bucket_cols), n_buckets);

        let result = self
            .buckets
            .par_iter_mut()
            .zip(self.col_counts.par_chunks_mut(bucket_cols))
            .enumerate()
            .try_for_each(|(b, (bucket, counts))| -> Result<(), CscBuilderError> {
                let lo = b * bucket_cols;
                let hi = if b + 1 == n_buckets {
                    n_cols
                } else {
                    lo + bucket_cols
                };
                for row in 0..n_rows {
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    let idx = &csr.indices[start..end];
                    let a = start + idx.partition_point(|&c| (c as usize) < lo);
                    let z = start + idx.partition_point(|&c| (c as usize) < hi);
                    for &col in &csr.indices[a..z] {
                        counts[col as usize - lo] += 1;
                    }
                    let mut j = a;
                    while j < z {
                        j += bucket.push_run(
                            base + row as u32,
                            &csr.indices[j..z],
                            &csr.data[j..z],
                            block_bytes,
                        );
                        if bucket.cur.len() < block_bytes {
                            continue;
                        }
                        let grew = bucket.seal(block_capacity);
                        let now = in_memory.fetch_add(grew, Ordering::Relaxed) + grew;
                        peak.fetch_max(now, Ordering::Relaxed);
                        if now <= ceiling {
                            continue;
                        }
                        let mut store = store.lock().unwrap_or_else(|p| p.into_inner());
                        if !store.can_spill() {
                            return Err(CscBuilderError::SpillRefused { budget: ceiling });
                        }
                        store.append_all(b, &std::mem::take(&mut bucket.sealed))?;
                        in_memory.fetch_sub(bucket.sealed_bytes, Ordering::Relaxed);
                        bucket.sealed_bytes = 0;
                    }
                }
                Ok(())
            });

        self.store = store.into_inner().unwrap_or_else(|p| p.into_inner());
        self.in_memory_bytes = in_memory.into_inner();
        self.peak_in_memory_bytes = peak.into_inner();
        result
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
            next_group: 0,
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

/// Whether every row of `csr` can be routed by binary search: indices
/// non-decreasing within each row and inside `[0, n_cols)`.
///
/// Canonical CSR always passes. A row with unsorted indices, or an index out
/// of range, sends the whole shard down the serial path, which routes in `j`
/// order and reports the out-of-range column exactly as it always has.
/// Non-decreasing rather than strictly increasing: a duplicate `(row, col)` is
/// adjacent in a sorted row, so a search keeps both copies in `j` order, as
/// the serial scan does.
fn rows_route_by_search(csr: &ScxCsr, n_cols: usize) -> bool {
    let row_ok = |row: usize| {
        let idx = &csr.indices[csr.indptr[row] as usize..csr.indptr[row + 1] as usize];
        match (idx.first(), idx.last()) {
            (Some(&first), Some(&last)) => {
                first >= 0 && (last as usize) < n_cols && idx.windows(2).all(|w| w[0] <= w[1])
            }
            _ => true,
        }
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..csr.n_rows())
            .into_par_iter()
            .with_min_len(1024)
            .all(row_ok)
    }
    #[cfg(not(feature = "parallel"))]
    (0..csr.n_rows()).all(row_ok)
}

// ---------------------------------------------------------------------------
// The emit contract
// ---------------------------------------------------------------------------

/// Something that yields a file's CSC shards, left to right, at on-disk widths.
///
/// Two implementations, because a caller that already holds the whole CSR and
/// one that can only be pushed to have genuinely different best answers:
///
/// * [`ResidentCscSource`] — the CSR is already in RAM (an eager h5ad ingest,
///   a `from_anndata` handing over GIL-owned copies, an R data frame). Routing
///   it through buckets would be a **second** copy at 8 B/nnz, roughly doubling
///   the peak of an ingest path that nothing gates. It scatters straight out of
///   the resident shards instead.
/// * [`CscEmitter`] — the source is a stream. Buckets are what buy the single
///   pass, and the spill is what bounds the memory.
///
/// One planner, one `indptr` construction and one writer loop above them; only
/// the record source differs. A differential test pins the two equal.
pub trait CscShardSource {
    /// Every shard this source will produce, in order, with exact nnz.
    fn plan(&self) -> &[CscShardSpec];

    /// Rows every emitted shard spans.
    fn n_rows(&self) -> usize;

    /// Fill the buffers with the next shard at **on-disk** widths and return
    /// its `col_start`, or `None` when the plan is drained.
    ///
    /// On-disk widths rather than [`CscArrays`]' `i64`/`i32`: the predecessor's
    /// caller rebuilt `csc_indptr_u64` and `csc_indices_u32` from every chunk,
    /// two full-length copies per shard that the allocation table names as a
    /// reason the CSC budget row could not be enforced.
    fn next_shard_into(
        &mut self,
        indptr: &mut Vec<u64>,
        indices: &mut Vec<u32>,
        data: &mut Vec<f32>,
    ) -> Result<Option<u64>, CscBuilderError>;

    /// The next shards, up to `max_nnz` nonzeros of them — always at least
    /// one while any remain, empty once drained. `whole_groups` lets a source
    /// that builds shards in groups hand out every shard of the groups it
    /// built, even past `max_nnz`: faster, since more shards encode at once,
    /// and bounded only by the group size. A source that cannot build ahead
    /// cheaply yields one at a time, which is this default.
    fn next_batch(
        &mut self,
        max_nnz: u64,
        whole_groups: bool,
    ) -> Result<Vec<CscShardArrays>, CscBuilderError> {
        let _ = (max_nnz, whole_groups);
        let mut a = CscShardArrays::default();
        Ok(
            match self.next_shard_into(&mut a.indptr, &mut a.indices, &mut a.data)? {
                Some(col_start) => {
                    a.col_start = col_start;
                    vec![a]
                }
                None => Vec::new(),
            },
        )
    }

    fn stats(&self) -> CscBuilderStats;
}

/// Emit CSC shards from a CSR that is already entirely resident.
///
/// This is the predecessor's `transpose_column_chunk` with the count pass
/// **hoisted out of the per-shard loop**: `nnz + n_shards * nnz` instead of
/// `2 * n_shards * nnz`, same bytes, same peak. It keeps the eager callers off
/// the bucket path, where their already-resident CSR would be copied a second
/// time.
pub struct ResidentCscSource<'a> {
    shards: &'a [ScxCsr],
    n_rows: usize,
    col_counts: Vec<u64>,
    plan: Vec<CscShardSpec>,
    next: usize,
    nnz: u64,
    /// Every row of every shard passes [`rows_route_by_search`], so a shard's
    /// column range is found by binary search per row instead of by testing
    /// every nonzero — `n_rows * log` per CSC shard rather than `nnz`, which
    /// at census_500k's 87 shards is the difference between reading the
    /// matrix once and reading it 87 times.
    sorted: bool,
}

impl<'a> ResidentCscSource<'a> {
    pub fn new(
        shards: &'a [ScxCsr],
        n_rows: usize,
        n_cols: usize,
        cols_per_shard: usize,
        memory_bytes: usize,
    ) -> Result<Self, CscBuilderError> {
        if n_rows > i32::MAX as usize {
            return Err(CscBuilderError::RowCountOverflow(n_rows));
        }
        for shard in shards {
            if shard.shape.1 != n_cols {
                return Err(CscBuilderError::ShapeMismatch {
                    shard_cols: shard.shape.1,
                    expected_cols: n_cols,
                });
            }
        }
        // Identical to the builder's, and to the predecessor's.
        let shard_cols = compute_chunk_cols_with_cap(n_rows, memory_bytes, cols_per_shard)?;

        // The one count pass. The predecessor ran this per chunk, range-testing
        // every nonzero each time, which is half of its `2 * nnz * n_chunks`.
        let mut col_counts = vec![0u64; n_cols];
        let mut nnz = 0u64;
        for shard in shards {
            for &col in &shard.indices {
                let col = col as usize;
                if col >= n_cols {
                    return Err(CscBuilderError::ColumnOutOfRange { col, n_cols });
                }
                col_counts[col] += 1;
                nnz += 1;
            }
        }

        let n_shards = n_cols.div_ceil(shard_cols);
        let mut plan = Vec::with_capacity(n_shards);
        for s in 0..n_shards {
            let lo = s.saturating_mul(shard_cols).min(n_cols);
            let hi = lo.saturating_add(shard_cols).min(n_cols);
            plan.push(CscShardSpec {
                col_start: lo,
                col_end: hi,
                nnz: col_counts[lo..hi].iter().sum(),
            });
        }
        let sorted = shards.iter().all(|s| rows_route_by_search(s, n_cols));
        Ok(Self {
            shards,
            n_rows,
            col_counts,
            plan,
            next: 0,
            nnz,
            sorted,
        })
    }

    /// Scatter shard `spec`'s columns out of the resident CSR.
    ///
    /// Shards in order, rows in order, positions in order — the scan order the
    /// predecessor's comment names, and the reason each column comes out
    /// strictly increasing in row without a sort. A sorted row's range is a
    /// contiguous run of it, so searching for the run visits the same entries
    /// in the same order the full scan would keep.
    fn fill(&self, spec: CscShardSpec) -> CscShardArrays {
        let (c0, c1) = (spec.col_start, spec.col_end);
        let mut indptr = Vec::with_capacity(c1 - c0 + 1);
        indptr.push(0u64);
        let mut cumsum = 0u64;
        for c in c0..c1 {
            cumsum += self.col_counts[c];
            indptr.push(cumsum);
        }
        let nnz = cumsum as usize;
        let mut indices = vec![0u32; nnz];
        let mut data = vec![0.0f32; nnz];
        let mut cursor = vec![0u64; c1 - c0];
        let mut row_offset: usize = 0;
        for shard in self.shards {
            let shard_n_rows = shard.n_rows();
            for row in 0..shard_n_rows {
                let start = shard.indptr[row] as usize;
                let end = shard.indptr[row + 1] as usize;
                let (a, z) = if self.sorted {
                    let idx = &shard.indices[start..end];
                    (
                        start + idx.partition_point(|&c| (c as usize) < c0),
                        start + idx.partition_point(|&c| (c as usize) < c1),
                    )
                } else {
                    (start, end)
                };
                for j in a..z {
                    let col = shard.indices[j] as usize;
                    if col >= c0 && col < c1 {
                        let lc = col - c0;
                        let dest = (indptr[lc] + cursor[lc]) as usize;
                        indices[dest] = (row_offset + row) as u32;
                        data[dest] = shard.data[j];
                        cursor[lc] += 1;
                    }
                }
            }
            row_offset += shard_n_rows;
        }
        CscShardArrays {
            col_start: c0 as u64,
            indptr,
            indices,
            data,
        }
    }
}

impl CscShardSource for ResidentCscSource<'_> {
    fn plan(&self) -> &[CscShardSpec] {
        &self.plan
    }

    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn next_shard_into(
        &mut self,
        indptr: &mut Vec<u64>,
        indices: &mut Vec<u32>,
        data: &mut Vec<f32>,
    ) -> Result<Option<u64>, CscBuilderError> {
        let Some(&spec) = self.plan.get(self.next) else {
            return Ok(None);
        };
        self.next += 1;
        let a = self.fill(spec);
        *indptr = a.indptr;
        *indices = a.indices;
        *data = a.data;
        Ok(Some(a.col_start))
    }

    /// Shards are independent reads of the resident CSR, so a batch fills them
    /// in parallel with `parallel`. A resident source has no groups, so
    /// `whole_groups` changes nothing.
    fn next_batch(
        &mut self,
        max_nnz: u64,
        _whole_groups: bool,
    ) -> Result<Vec<CscShardArrays>, CscBuilderError> {
        let (mut n, mut nnz) = (0usize, 0u64);
        while let Some(spec) = self.plan.get(self.next + n) {
            if n > 0 && nnz + spec.nnz > max_nnz {
                break;
            }
            nnz += spec.nnz;
            n += 1;
        }
        let specs = &self.plan[self.next..self.next + n];
        #[cfg(feature = "parallel")]
        let out = {
            use rayon::prelude::*;
            specs.par_iter().map(|&spec| self.fill(spec)).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let out = specs.iter().map(|&spec| self.fill(spec)).collect();
        self.next += n;
        Ok(out)
    }

    fn stats(&self) -> CscBuilderStats {
        CscBuilderStats {
            nnz: self.nnz,
            n_buckets: 0,
            peak_in_memory_bytes: 0,
            spilled_bytes: 0,
            first_non_strict_column: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

/// One emitted CSC shard, at on-disk widths.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CscShardArrays {
    pub col_start: u64,
    pub indptr: Vec<u64>,
    pub indices: Vec<u32>,
    pub data: Vec<f32>,
}

pub struct CscEmitter {
    n_rows: usize,
    n_cols: usize,
    layout: Layout,
    store: Box<dyn SpillStore>,
    buckets: Vec<Bucket>,
    col_counts: Vec<u64>,
    plan: Vec<CscShardSpec>,
    next_group: usize,
    pending: VecDeque<CscShardArrays>,
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

    /// See [`CscShardSource::next_shard_into`].
    ///
    /// The caller's buffers are moved into, not appended to, so their previous
    /// allocations are released rather than reused — what this saves is the
    /// width conversion, not the allocation.
    pub fn next_shard_into(
        &mut self,
        indptr: &mut Vec<u64>,
        indices: &mut Vec<u32>,
        data: &mut Vec<f32>,
    ) -> Result<Option<u64>, CscBuilderError> {
        while self.pending.is_empty() {
            if self.next_group >= self.layout.n_groups() {
                return Ok(None);
            }
            self.build_groups(1)?;
        }
        let built = self.pending.pop_front().expect("non-empty");
        *indptr = built.indptr;
        *indices = built.indices;
        *data = built.data;
        Ok(Some(built.col_start))
    }

    /// The next shards, up to `max_nnz` nonzeros of them (always at least one
    /// group's worth), built in parallel; empty once the plan is drained.
    ///
    /// Whole emit groups (see [`Layout::group`]) are taken while their exact
    /// planned nnz fits. Groups own disjoint buckets, so they build
    /// concurrently; the shards come back in column order.
    pub fn next_batch(
        &mut self,
        max_nnz: u64,
        whole_groups: bool,
    ) -> Result<Vec<CscShardArrays>, CscBuilderError> {
        if self.pending.is_empty() {
            let n_groups = self.layout.n_groups();
            let (mut n, mut nnz) = (0usize, 0u64);
            while self.next_group + n < n_groups {
                let ((lo, hi), _) = self.layout.group(self.next_group + n);
                let group_nnz: u64 = self.plan[lo..hi.max(lo)].iter().map(|s| s.nnz).sum();
                if n > 0 && nnz + group_nnz > max_nnz {
                    break;
                }
                nnz += group_nnz;
                n += 1;
            }
            if n == 0 {
                return Ok(Vec::new());
            }
            self.build_groups(n)?;
        }
        if whole_groups {
            return Ok(self.pending.drain(..).collect());
        }
        // Hand out whole shards up to `max_nnz` (at least one), not the whole
        // of what was built: a coarse group is several shards, and a batch
        // that could not be smaller than a group would make `max_nnz` a
        // floor-of-a-group rather than a cap. The rest wait in `pending`.
        let mut take = 0usize;
        let mut nnz = 0u64;
        for a in &self.pending {
            let n = a.indices.len() as u64;
            if take > 0 && nnz + n > max_nnz {
                break;
            }
            nnz += n;
            take += 1;
        }
        Ok(self.pending.drain(..take).collect())
    }

    /// Build the next `n` groups into `pending`, concurrently with `parallel`.
    fn build_groups(&mut self, n: usize) -> Result<(), CscBuilderError> {
        let first = self.next_group;
        let ctx = GroupCtx {
            layout: &self.layout,
            plan: &self.plan,
            col_counts: &self.col_counts,
            store: &*self.store,
            n_cols: self.n_cols,
        };
        // Consecutive groups own consecutive, disjoint bucket ranges.
        let mut jobs: Vec<(usize, &mut [Bucket])> = Vec::with_capacity(n);
        let mut rest: &mut [Bucket] = &mut self.buckets;
        let mut consumed = 0usize;
        for g in first..first + n {
            let (_, (lo, hi)) = ctx.layout.group(g);
            let (_, tail) = std::mem::take(&mut rest).split_at_mut(lo - consumed);
            let (mine, tail) = tail.split_at_mut(hi - lo);
            rest = tail;
            consumed = hi;
            jobs.push((g, mine));
        }
        #[cfg(feature = "parallel")]
        let results: Vec<_> = {
            use rayon::prelude::*;
            jobs.into_par_iter()
                .map(|(g, buckets)| build_group(&ctx, g, buckets))
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let results: Vec<_> = jobs
            .into_iter()
            .map(|(g, buckets)| build_group(&ctx, g, buckets))
            .collect();
        // In group order, so the first error — and the pending order — are
        // what one group at a time would give.
        for r in results {
            let (built, first_non_strict) = r?;
            if let Some(col) = first_non_strict {
                if self.stats.first_non_strict_column.is_none_or(|c| col < c) {
                    self.stats.first_non_strict_column = Some(col);
                }
            }
            self.pending.extend(built);
        }
        self.next_group += n;
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

/// What every group build of a batch reads.
struct GroupCtx<'a> {
    layout: &'a Layout,
    plan: &'a [CscShardSpec],
    col_counts: &'a [u64],
    store: &'a dyn SpillStore,
    n_cols: usize,
}

/// Build the shards of emit group `g` (see [`Layout::group`]), in one pass
/// over each of its buckets' records.
///
/// One pass per bucket rather than per shard: a bucket's columns are a
/// union of contiguous per-shard column ranges, and a contiguous column
/// range of a CSC shard is a contiguous slice of its `indices` / `data`.
/// So each bucket owns disjoint slices of the group's shards, which is what
/// lets a fine group drain its buckets in parallel with no locking and no
/// change to where any record lands.
fn build_group(
    ctx: &GroupCtx<'_>,
    g: usize,
    buckets: &mut [Bucket],
) -> Result<(Vec<CscShardArrays>, Option<usize>), CscBuilderError> {
    let ((shard_lo, shard_hi), (bucket_lo, bucket_hi)) = ctx.layout.group(g);
    debug_assert_eq!(buckets.len(), bucket_hi - bucket_lo);
    if shard_lo >= shard_hi {
        return Ok((Vec::new(), None));
    }

    // One prefix sum per shard, from counts that are already exact.
    let mut built: Vec<CscShardArrays> = Vec::with_capacity(shard_hi - shard_lo);
    for s in shard_lo..shard_hi {
        let spec = ctx.plan[s];
        let width = spec.col_end - spec.col_start;
        let mut indptr = Vec::with_capacity(width + 1);
        indptr.push(0u64);
        let mut cumsum = 0u64;
        for c in spec.col_start..spec.col_end {
            cumsum += ctx.col_counts[c];
            indptr.push(cumsum);
        }
        let nnz = cumsum as usize;
        built.push(CscShardArrays {
            col_start: spec.col_start as u64,
            indptr,
            indices: vec![0u32; nnz],
            data: vec![0.0f32; nnz],
        });
    }

    // Carve every shard's arrays into the per-bucket segments that cover
    // them. Buckets and shards both run left to right, so each segment is
    // the front of what is left of its shard.
    let bucket_cols = ctx.layout.bucket_cols;
    let n_cols = ctx.n_cols;
    let mut jobs: Vec<DrainJob<'_>> = Vec::with_capacity(bucket_hi - bucket_lo);
    {
        // What is left of each shard, as a segment spanning it whole.
        let mut rest: Vec<Segment<'_>> = built
            .iter_mut()
            .map(|t| Segment {
                col_lo: t.col_start as usize,
                indptr: t.indptr.as_slice(),
                indices: t.indices.as_mut_slice(),
                data: t.data.as_mut_slice(),
            })
            .collect();
        let group_lo = ctx.plan[shard_lo].col_start;
        let group_hi = ctx.plan[shard_hi - 1].col_end;
        for (b, bucket) in (bucket_lo..bucket_hi).zip(buckets.iter_mut()) {
            let c0 = (b * bucket_cols).max(group_lo);
            let c1 = if b + 1 == ctx.layout.n_buckets {
                n_cols
            } else {
                ((b + 1) * bucket_cols).min(group_hi)
            };
            let mut segments = Vec::new();
            for shard in rest.iter_mut() {
                let shard_end = shard.col_lo + shard.indptr.len() - 1;
                let (lo, hi) = (c0.max(shard.col_lo), c1.min(shard_end));
                if lo >= hi {
                    continue;
                }
                let (l0, l1) = (lo - shard.col_lo, hi - shard.col_lo);
                let n = (shard.indptr[l1] - shard.indptr[l0]) as usize;
                let (ix, ix_rest) = std::mem::take(&mut shard.indices).split_at_mut(n);
                let (dv, dv_rest) = std::mem::take(&mut shard.data).split_at_mut(n);
                segments.push(Segment {
                    col_lo: lo,
                    indptr: &shard.indptr[l0..=l1],
                    indices: ix,
                    data: dv,
                });
                // The shard's remainder now starts where this segment ends.
                shard.col_lo = hi;
                shard.indptr = &shard.indptr[l1..];
                shard.indices = ix_rest;
                shard.data = dv_rest;
            }
            jobs.push(DrainJob {
                index: b,
                col_lo: c0,
                col_hi: c1,
                bucket,
                segments,
            });
        }
    }

    let store = ctx.store;
    let shard_cols = ctx.layout.shard_cols;
    let col_counts = ctx.col_counts;
    #[cfg(feature = "parallel")]
    if jobs.len() > 1 {
        use rayon::prelude::*;
        jobs.into_par_iter()
            .try_for_each(|job| job.drain(store, shard_cols, col_counts))?;
    } else {
        for job in jobs {
            job.drain(store, shard_cols, col_counts)?;
        }
    }
    #[cfg(not(feature = "parallel"))]
    for job in jobs {
        job.drain(store, shard_cols, col_counts)?;
    }

    let mut first_non_strict: Option<usize> = None;
    for (i, t) in built.iter().enumerate() {
        let s = shard_lo + i;
        let spec = ctx.plan[s];
        debug_assert_eq!(
            t.indices.len(),
            spec.nnz as usize,
            "shard {s} filled {} of {} slots",
            t.indices.len(),
            spec.nnz
        );
        // Reported in BOTH profiles, never panicked in either.
        //
        // A non-strict column means the *source* carried a duplicate
        // `(row, col)`; it is not this builder's bug, and `run_build_csc`
        // accepts such input by design. A `debug_assert!(false)` here made
        // the contract depend on the build profile — panic in debug,
        // silently emit in release — which is worse than either. The bytes
        // match the predecessor's (both entries, in `j` order) and the
        // caller gets the column so it can warn; `validate_csc` is what
        // rejects such a sidecar at the GPU.
        let non_strict = |lc: &usize| {
            let (a, z) = (t.indptr[*lc] as usize, t.indptr[*lc + 1] as usize);
            t.indices[a..z].windows(2).any(|w| w[0] >= w[1])
        };
        let width = spec.col_end - spec.col_start;
        #[cfg(feature = "parallel")]
        let first = {
            use rayon::prelude::*;
            (0..width)
                .into_par_iter()
                .with_min_len(64)
                .find_first(non_strict)
        };
        #[cfg(not(feature = "parallel"))]
        let first = (0..width).find(non_strict);
        if let Some(lc) = first {
            let col = spec.col_start + lc;
            if first_non_strict.is_none_or(|c| col < c) {
                first_non_strict = Some(col);
            }
        }
    }
    Ok((built, first_non_strict))
}

impl CscShardSource for CscEmitter {
    fn plan(&self) -> &[CscShardSpec] {
        CscEmitter::plan(self)
    }
    fn n_rows(&self) -> usize {
        CscEmitter::n_rows(self)
    }
    fn next_shard_into(
        &mut self,
        indptr: &mut Vec<u64>,
        indices: &mut Vec<u32>,
        data: &mut Vec<f32>,
    ) -> Result<Option<u64>, CscBuilderError> {
        CscEmitter::next_shard_into(self, indptr, indices, data)
    }
    fn next_batch(
        &mut self,
        max_nnz: u64,
        whole_groups: bool,
    ) -> Result<Vec<CscShardArrays>, CscBuilderError> {
        CscEmitter::next_batch(self, max_nnz, whole_groups)
    }
    fn stats(&self) -> CscBuilderStats {
        CscEmitter::stats(self).clone()
    }
}

/// A contiguous column range of one shard being built, and the slices of its
/// arrays that range owns. `indptr` is the shard's own, restricted to the
/// range, so `indptr[k] - indptr[0]` is an offset into `indices` / `data`.
struct Segment<'a> {
    col_lo: usize,
    indptr: &'a [u64],
    indices: &'a mut [u32],
    data: &'a mut [f32],
}

/// One bucket to drain into the segments it covers.
struct DrainJob<'a> {
    index: usize,
    col_lo: usize,
    col_hi: usize,
    bucket: &'a mut Bucket,
    segments: Vec<Segment<'a>>,
}

impl DrainJob<'_> {
    fn drain(
        mut self,
        store: &dyn SpillStore,
        shard_cols: usize,
        col_counts: &[u64],
    ) -> Result<(), CscBuilderError> {
        let (b, col_lo, col_hi) = (self.index, self.col_lo, self.col_hi);
        // The widest row block this bucket's writer could have produced: one
        // record per column it owns. A duplicate-bearing source row can exceed
        // it, so allow the whole bucket's nnz as the ceiling rather than the
        // column count — the point is to reject a corrupt length, not to
        // second-guess a legal one.
        let max_records = col_counts[col_lo..col_hi].iter().sum::<u64>().max(1) as usize;
        // Per-column write cursors, flat across the bucket's range.
        let mut cursor = vec![0u64; col_hi - col_lo];
        // Segments are whole shards (coarse) or one part of one shard (fine),
        // so a column's segment is its shard's offset from the first one.
        // `shard_cols` is uniform, so this is a divide, not a search.
        let first_shard = col_lo / shard_cols;
        let segments = &mut self.segments;
        let mut scatter = |row: u32, payload: &[u8]| -> Result<(), CscBuilderError> {
            let (recs, tail) = payload.as_chunks::<SPILL_BYTES_PER_NNZ>();
            debug_assert!(tail.is_empty(), "a row block's payload is 8 B per nonzero");
            for rec in recs {
                let col = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]) as usize;
                let val = f32::from_le_bytes([rec[4], rec[5], rec[6], rec[7]]);
                // Fail at the FIRST bad record rather than noting it and
                // reading on. Recording the column and returning from this one
                // block left `drain_bucket` walking the rest, so a later record
                // — or two individually legal blocks whose totals exceed the
                // planned slots — could index past `indices` and panic before
                // the note was ever inspected.
                if col < col_lo || col >= col_hi {
                    return Err(CscBuilderError::SpillCorrupt {
                        bucket: b,
                        detail: format!(
                            "column {col} is outside this bucket's range [{col_lo}, {col_hi})"
                        ),
                    });
                }
                let seg = &mut segments[col / shard_cols - first_shard];
                let lc = col - seg.col_lo;
                let flat = col - col_lo;
                let slots = seg.indptr[lc + 1] - seg.indptr[lc];
                // The planned slot count for this column is exact, so a stream
                // carrying more records for it than were counted during the
                // push is corrupt — and would otherwise write into the next
                // column's slots, or past the end.
                if cursor[flat] >= slots {
                    return Err(CscBuilderError::SpillCorrupt {
                        bucket: b,
                        detail: format!(
                            "column {col} carries more records than the {slots} counted \
                             for it during the push"
                        ),
                    });
                }
                let dest = (seg.indptr[lc] - seg.indptr[0] + cursor[flat]) as usize;
                seg.indices[dest] = row;
                seg.data[dest] = val;
                cursor[flat] += 1;
            }
            Ok(())
        };
        drain_bucket(store, self.bucket, b, max_records, &mut scatter)?;
        debug_assert!(
            (col_lo..col_hi).all(|c| cursor[c - col_lo] == col_counts[c]),
            "bucket {b} left a column short of its counted slots"
        );
        Ok(())
    }
}

/// What a bucket walk hands each row block: the global row, then its packed
/// `(cols, vals)` payload. Fallible because the scatter validates every column
/// it is handed — a corrupt spill stream must surface as an error, not as an
/// out-of-range index into the shard arrays.
type RowBlockSink<'a> = &'a mut dyn FnMut(u32, &[u8]) -> Result<(), CscBuilderError>;

/// Walk `bucket`'s records in append order: the spilled prefix, then the RAM
/// tail. The seam between them is an append-order seam, so a column whose rows
/// span it still comes out ascending.
fn drain_bucket(
    store: &dyn SpillStore,
    bucket: &mut Bucket,
    index: usize,
    max_records: usize,
    f: RowBlockSink<'_>,
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
            // Bound the declared count before allocating for it. `n` comes
            // straight off disk, so a truncated or corrupt spill can name a
            // count whose `resize` is an OOM abort rather than an error the
            // caller can see. `max_records` is the widest row block the writer
            // could have produced for this bucket.
            if n > max_records {
                return Err(CscBuilderError::SpillCorrupt {
                    bucket: index,
                    detail: format!(
                        "row {row} declares {n} nonzeros, more than the {max_records} \
                         this bucket can hold"
                    ),
                });
            }
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
            f(row, &payload)?;
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

fn walk_blocks(block: &[u8], index: usize, f: RowBlockSink<'_>) -> Result<(), CscBuilderError> {
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
        f(row, &block[lo..hi])?;
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
