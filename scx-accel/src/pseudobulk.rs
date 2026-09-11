//! Pseudobulk aggregation for differential expression.
//!
//! Aggregates single-cell counts into pseudobulk samples by grouping cells
//! according to metadata columns (e.g., `["perturbation", "donor"]`).
//! The resulting count matrix is fed to `pydeseq2` on the Python side for
//! negative binomial GLM testing.
//!
//! Supports both streaming (shard-by-shard over any CSR `ShardSource`) and
//! in-memory (`ScxCsr`) paths. Streaming callers holding a subset SCX handle
//! must pass that handle's *view* (`as_shard_source()`), not the reader
//! underneath it — `obs_groups` is indexed by visible cell.
//!
//! # Parallelism and bit-identity
//!
//! Every CSR path scatters in parallel by partitioning the **output** across
//! rayon workers — one group's row per task while the largest group holds at
//! most two pool-shares of the nonzeros (and always when the rows are not
//! canonical), a contiguous column block of every group's row otherwise — and
//! merging nothing, so for each `(group, gene)` the f64 sum is formed from the same
//! f32 operands in the same ascending-row order as a serial loop, on any
//! thread count. `RAYON_NUM_THREADS` sizes the pool; `SCX_ACCEL_NUM_THREADS`
//! caps the column-block count (and a CSC run's column tasks) as it caps PCA's
//! blocks, while the group partition and the mean divide are one task per group
//! row on the ambient pool. Neither knob can move a bit. The dense path
//! partitions by group the same way.

use std::collections::HashMap;

use rayon::prelude::*;
use scx_format_io::ShardSource;

use crate::pca::colblocks;
use crate::Result;

/// Aggregation method for pseudobulk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Sum of counts per group (default for DESeq2).
    Sum,
    /// Mean of counts per group.
    Mean,
}

impl AggregationMethod {
    /// Parse the user-facing `aggr_method=` string — the single vocabulary +
    /// error text for every binding (the message is pinned by the pyscx test
    /// suite, which surfaces it as `RuntimeError`).
    pub fn parse(aggr_method: &str) -> std::result::Result<Self, crate::error::InvalidArgument> {
        match aggr_method {
            "sum" => Ok(AggregationMethod::Sum),
            "mean" => Ok(AggregationMethod::Mean),
            other => Err(crate::error::InvalidArgument(format!(
                "unsupported aggr_method '{other}': use 'sum' or 'mean'"
            ))),
        }
    }
}

/// pdex-style pseudobulk mode controlling per-cell and per-group-mean transforms.
///
/// Encodes the four `(geometric_mean × is_log1p)` combinations from
/// `pdex._math.pseudobulk`. The per-cell transform `f(x)` is applied to each
/// cell's expression value before averaging; the per-group-mean transform
/// `g(y)` is applied to the resulting mean. Both `f(0) = 0` and `g(0) = 0`
/// hold for all four modes, so CSR aggregation only visits non-zero entries.
///
/// | mode             | `geometric_mean` | `is_log1p` | `f(x)`    | `g(y)`     |
/// |------------------|------------------|------------|-----------|------------|
/// | `ArithRaw`       | false            | false      | `x`       | `y`        |
/// | `ArithLog1pExpand` | false          | true       | `expm1(x)`| `y`        |
/// | `GeomRaw`        | true             | false      | `log1p(x)`| `expm1(y)` |
/// | `GeomLog1p`      | true             | true       | `x`       | `expm1(y)` |
///
/// The output mean is always in **natural (count) space**, matching pdex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomMeanMode {
    /// `geometric_mean=False, is_log1p=False`: arithmetic mean of raw counts.
    ArithRaw,
    /// `geometric_mean=False, is_log1p=True`: arithmetic mean of `expm1(X)`.
    ArithLog1pExpand,
    /// `geometric_mean=True, is_log1p=False`: `expm1(mean(log1p(X)))`.
    GeomRaw,
    /// `geometric_mean=True, is_log1p=True`: `expm1(mean(X))`.
    GeomLog1p,
}

impl GeomMeanMode {
    /// Per-cell-value transform `f(x)`. All four modes satisfy `f(0) = 0`,
    /// so sparse-aggregation paths can skip explicit zeros.
    #[inline]
    pub fn pre(self, x: f64) -> f64 {
        match self {
            Self::ArithRaw | Self::GeomLog1p => x,
            Self::ArithLog1pExpand => x.exp_m1(),
            Self::GeomRaw => x.ln_1p(),
        }
    }

    /// Per-(group, gene) mean transform `g(y)`. Applied after dividing the
    /// sum of `f(x_i)` by the group's cell count.
    #[inline]
    pub fn post(self, y: f64) -> f64 {
        match self {
            Self::ArithRaw | Self::ArithLog1pExpand => y,
            Self::GeomRaw | Self::GeomLog1p => y.exp_m1(),
        }
    }

    /// Convenience constructor from the `(geometric_mean, is_log1p)` pair
    /// pdex's Python API exposes.
    #[inline]
    pub fn from_flags(geometric_mean: bool, is_log1p: bool) -> Self {
        match (geometric_mean, is_log1p) {
            (false, false) => Self::ArithRaw,
            (false, true) => Self::ArithLog1pExpand,
            (true, false) => Self::GeomRaw,
            (true, true) => Self::GeomLog1p,
        }
    }

    /// The count-space **arithmetic** counterpart of this mode, preserving its
    /// `is_log1p` interpretation. This is what pdex's `cpm_bulk` uses for the
    /// `cpm_filter` decision — a per-gene arithmetic mean in count space,
    /// independent of whether the reported mean is geometric.
    #[inline]
    pub fn arith(self) -> Self {
        match self {
            Self::ArithRaw | Self::GeomRaw => Self::ArithRaw,
            Self::ArithLog1pExpand | Self::GeomLog1p => Self::ArithLog1pExpand,
        }
    }
}

/// Result of pseudobulk aggregation.
#[derive(Debug, Clone)]
pub struct PseudobulkResult {
    /// Aggregated count matrix `[n_groups × n_vars]`, row-major.
    pub counts: Vec<f64>,
    /// Group labels: `group_labels[i]` is a Vec of column values for group `i`.
    /// E.g., for groupby `["perturbation", "donor"]`, `group_labels[0]` might be
    /// `["drug_A", "donor_1"]`.
    pub group_labels: Vec<Vec<String>>,
    /// Column names from groupby (e.g., `["perturbation", "donor"]`).
    pub groupby_columns: Vec<String>,
    /// Number of cells contributing to each group.
    pub cell_counts: Vec<usize>,
    /// Gene names.
    pub gene_names: Vec<String>,
    /// Number of groups.
    pub n_groups: usize,
    /// Number of variables (genes).
    pub n_vars: usize,
}

/// Build a group-key → group-index mapping from per-cell obs column vectors.
///
/// The group key is the cell's tuple of values across all groupby columns
/// (e.g. `("drug_A", "donor_1")`). Strings are interned once per column,
/// giving each distinct label a `u32` id; the per-cell hot path then builds
/// a `Vec<u32>` of ids and looks it up by borrowed slice, so lookups of
/// already-seen groups allocate nothing. This replaces the old `"\x1F"`
/// joined-string scheme (which collided on values containing the separator)
/// without re-introducing the per-cell `String` allocations that scheme paid.
///
/// Returns:
/// - `cell_to_group`: group index for each cell (length = n_obs)
/// - `group_labels`: per-group label vectors (each Vec has len = n_groupby_cols)
/// - ordered deterministically (sorted lexicographically by the label tuple)
pub fn build_group_mapping(
    obs_groups: &[Vec<String>],
    n_obs: usize,
) -> (Vec<usize>, Vec<Vec<String>>) {
    let n_cols = obs_groups.len();

    // Per-column string interners. Keyed by `&str` borrowed from obs_groups;
    // the backing Vec<String>s live for the full call, so the borrow is sound.
    let mut col_interners: Vec<HashMap<&str, u32>> = (0..n_cols).map(|_| HashMap::new()).collect();
    let mut col_vocab: Vec<Vec<&str>> = vec![Vec::new(); n_cols];

    // Group table: `Vec<u32>` of interned ids -> group index.
    let mut key_to_index: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut group_label_ids: Vec<Vec<u32>> = Vec::new();
    let mut cell_to_group = Vec::with_capacity(n_obs);

    // Reused per-cell buffer — avoids n_obs * n_cols * u32 reallocations.
    let mut key_buf: Vec<u32> = Vec::with_capacity(n_cols);

    // Parallel-index across `n_cols` columns — idiomatic `for cell in ...` is
    // clearer than the clippy-suggested iterator chain over the first column.
    #[allow(clippy::needless_range_loop)]
    for cell in 0..n_obs {
        key_buf.clear();
        for col_idx in 0..n_cols {
            let s: &str = obs_groups[col_idx][cell].as_str();
            let id = match col_interners[col_idx].get(s) {
                Some(&id) => id,
                None => {
                    let id = col_vocab[col_idx].len() as u32;
                    col_interners[col_idx].insert(s, id);
                    col_vocab[col_idx].push(s);
                    id
                }
            };
            key_buf.push(id);
        }

        // Borrowed-slice lookup: `Vec<u32>: Borrow<[u32]>`, so we hit without
        // allocating a key. Only new groups pay a single clone on insert.
        let group_idx = if let Some(&idx) = key_to_index.get(key_buf.as_slice()) {
            idx
        } else {
            let idx = group_label_ids.len();
            key_to_index.insert(key_buf.clone(), idx);
            group_label_ids.push(key_buf.clone());
            idx
        };

        cell_to_group.push(group_idx);
    }

    // Materialize owned group labels from interned ids (once per unique group).
    let group_labels: Vec<Vec<String>> = group_label_ids
        .iter()
        .map(|ids| {
            ids.iter()
                .enumerate()
                .map(|(col_idx, &id)| col_vocab[col_idx][id as usize].to_string())
                .collect()
        })
        .collect();

    // Sort groups deterministically by their label tuple (lexicographic).
    let mut sorted_indices: Vec<usize> = (0..group_labels.len()).collect();
    sorted_indices.sort_by(|&a, &b| group_labels[a].cmp(&group_labels[b]));

    // Build remapping: old index → new index.
    let mut remap = vec![0usize; group_labels.len()];
    for (new_idx, &old_idx) in sorted_indices.iter().enumerate() {
        remap[old_idx] = new_idx;
    }

    // Apply remapping.
    let sorted_labels: Vec<Vec<String>> = sorted_indices
        .iter()
        .map(|&old| group_labels[old].clone())
        .collect();
    let remapped_cells: Vec<usize> = cell_to_group.iter().map(|&old| remap[old]).collect();

    (remapped_cells, sorted_labels)
}

// ──────────────────────────────────────────────────────────────────────────
// The CSR scatter kernel.
//
// Every CSR path below accumulates `counts[g * n_vars + col] += data[j]` for
// each nonzero, cells in ascending row order. Written serially that is one
// dependent load-add-store per nonzero on the calling thread while the pool
// idles — memory-bound at ~1.4 ns per nonzero when the few group rows fit in
// L1, several times that once `counts` is tens of megabytes. The kernels here
// partition the **output** across rayon workers and never merge, so the
// partition cannot reach the result: for a fixed `(g, col)` every worker adds
// the same operands in the same ascending row order the serial loop did, on
// any thread count. Same thesis as `pca::colblocks`, whose planner this reuses.
//
// Two partitions, chosen per chunk (a shard on the streaming path, the whole
// matrix on the in-memory ones) by `choose_partition`:
//
// * `ByGroup` — each worker owns one group's row and walks that group's cells
//   ascending, visiting each row's nonzeros in **stored** order. Reads every
//   nonzero exactly once, needs no sortedness (so it is where a non-canonical
//   scipy matrix goes — `aggregate_pseudobulk`'s in-memory arm passes
//   `csr_matrix(x)` through without `sort_indices()`), and is bit-identical to
//   the serial loop even with duplicate columns. Its wall is bounded below by
//   the largest group's share of the chunk, so it is taken when that share is
//   at most `GROUP_PARTITION_SLACK` pool-shares.
// * `ColumnBlocks(n)` — each worker owns a contiguous column range of every
//   group's row. Indifferent to group skew (a control arm at half the cells) and
//   to cells sorted by group (a 4 096-row shard of a sorted file holds two or
//   three groups), which is what the group partition cannot handle. Requires
//   canonical rows. The obvious implementation — every block binary-searching
//   its window in every row — was measured slower than the serial loop on
//   100-nonzero rows: two dependent cache-missing search chains per row per
//   block cost more than the eight adds they find. So the windows are planned
//   **once**, in a parallel pass over rows that reads each row a single time
//   (`RowSplits`), and the block workers then read only their own windows —
//   two sequential passes over `indices`, one over `data`, no search. Even so
//   a block must own `MIN_BLOCK_WINDOW` nonzeros of the average row to be
//   worth a separate reader: twelve slivers of a 100-nonzero row touch more
//   cache lines than the serial loop streams, and that loop is already
//   memory-bound when two group rows fit in L1. Short-row chunks get fewer
//   blocks, down to the serial walk, and lose nothing to the machinery they
//   cannot use.
//
// Load balance for the column blocks comes from a per-column nnz histogram
// (`weights`) that the block workers bump for their own columns as they go —
// L1-resident, disjoint, free — and that accumulates across the chunks of one
// call, so chunk `k` is planned from chunks `0..k`. A separate `O(nnz)` weights
// pre-pass, which is what PCA does, would cost a fifth of the serial scatter it
// is meant to replace. Chunk 0 sees all-zero weights, for which `plan_blocks`
// falls back to even column counts.
// ──────────────────────────────────────────────────────────────────────────

/// Rows of a CSR matrix borrowed as three slices — one shard, or the whole
/// in-memory matrix. `indptr` has `n_rows + 1` entries.
#[derive(Clone, Copy)]
struct CsrRows<'a> {
    indptr: &'a [i64],
    indices: &'a [i32],
    data: &'a [f32],
}

impl<'a> CsrRows<'a> {
    fn of(csr: &'a scx_sparse::ScxCsr) -> Self {
        Self {
            indptr: &csr.indptr,
            indices: &csr.indices,
            data: &csr.data,
        }
    }

    fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    /// Nonzeros in these rows.
    fn nnz(&self) -> usize {
        match self.indptr {
            [first, .., last] => (*last - *first) as usize,
            _ => 0,
        }
    }

    #[inline]
    fn row(&self, r: usize) -> (&'a [i32], &'a [f32]) {
        let s = self.indptr[r] as usize;
        let e = self.indptr[r + 1] as usize;
        (&self.indices[s..e], &self.data[s..e])
    }
}

/// How one chunk of rows is split across workers. See the module comment
/// above.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScatterPartition {
    /// Exactly this many contiguous column blocks; requires canonical rows.
    ColumnBlocks(usize),
    /// One worker per group row; any row order.
    ByGroup,
}

/// Below this many nonzeros a chunk is scattered as one block: the rayon
/// dispatch would cost more than the work, and one block is the same
/// arithmetic in the same order, so the threshold is invisible in the result.
const MIN_NNZ_FOR_BLOCKS: usize = 4096;

/// The group partition is taken while the largest group's nonzeros are at
/// most this many pool-shares of the chunk's: its wall is bounded below by
/// that group, and the column blocks cost roughly two shares (a planning pass
/// and a scatter pass) regardless of how the cells are grouped.
const GROUP_PARTITION_SLACK: usize = 2;

/// A column block must own at least this many nonzeros of the average row to
/// be worth a separate reader. Twelve blocks each reading a 32-byte sliver of
/// a 100-nonzero row touch two to three times the cache lines the serial loop
/// streams, and that loop is already memory-bound when the few group rows fit
/// in L1 — measured 1.65× *slower* than serial on 2 000 genes at 5 % density
/// with two groups. So a short-row chunk gets fewer blocks, down to one (the
/// serial walk), and pays nothing for the parallel machinery it cannot use.
const MIN_BLOCK_WINDOW: usize = 64;

/// Whether every row's column indices are strictly increasing (sorted, no
/// duplicates) — the canonical form. `O(nnz)` reads spread over the pool;
/// allocates nothing.
fn rows_strictly_increasing_par(rows: CsrRows<'_>) -> bool {
    (0..rows.n_rows())
        .into_par_iter()
        .all(|r| rows.row(r).0.windows(2).all(|w| w[0] < w[1]))
}

/// Pick the partition for one chunk. `cell_to_group` is indexed by local row.
///
/// One block when there is too little work to split; the group partition when
/// its largest group is within [`GROUP_PARTITION_SLACK`] pool-shares of the
/// chunk (an `O(rows)` sum over `indptr`, no nonzero touched); otherwise as
/// many column blocks as the rows are long enough to feed
/// ([`MIN_BLOCK_WINDOW`]) when the rows are canonical, and the group partition
/// again when they are not — a non-canonical row cannot be windowed. The
/// canonical check is the one pass this can add; an SCX-decoded shard is
/// canonical by the writer contract, but `ShardSource` is a trait and a wrong
/// assumption here would silently drop nonzeros, so the streaming path pays
/// for it too when it gets this far.
fn choose_partition(
    rows: CsrRows<'_>,
    cell_to_group: &[usize],
    n_groups: usize,
    n_vars: usize,
) -> ScatterPartition {
    let nnz = rows.nnz();
    let n_rows = rows.n_rows();
    if nnz < MIN_NNZ_FOR_BLOCKS || n_rows == 0 || n_groups == 0 {
        return ScatterPartition::ColumnBlocks(1);
    }
    let pool = colblocks::block_count(n_vars);
    let mut nnz_by_group = vec![0usize; n_groups];
    for (r, &g) in cell_to_group.iter().enumerate() {
        nnz_by_group[g] += (rows.indptr[r + 1] - rows.indptr[r]) as usize;
    }
    let largest = nnz_by_group.iter().copied().max().unwrap_or(0);
    if largest * pool <= GROUP_PARTITION_SLACK * nnz {
        return ScatterPartition::ByGroup;
    }
    let n_blocks = (nnz / n_rows / MIN_BLOCK_WINDOW).clamp(1, pool);
    if n_blocks == 1 {
        return ScatterPartition::ColumnBlocks(1);
    }
    if rows_strictly_increasing_par(rows) {
        ScatterPartition::ColumnBlocks(n_blocks)
    } else {
        ScatterPartition::ByGroup
    }
}

/// Accumulate `rows` into `counts` (`[n_groups × n_vars]`, row-major).
///
/// `cell_to_group` is indexed by **local** row of `rows`. `weights` is the
/// cumulative per-column nnz histogram (`n_vars` entries) the column-block
/// partition plans from and adds to, when a later chunk will plan from it;
/// `None` (the in-memory paths — one dispatch, nothing plans after it) gets even
/// blocks and skips the per-nonzero bump. `ByGroup` leaves it alone either way.
fn scatter_rows(
    rows: CsrRows<'_>,
    cell_to_group: &[usize],
    counts: &mut [f64],
    n_vars: usize,
    n_groups: usize,
    part: ScatterPartition,
    weights: Option<&mut [u64]>,
) {
    debug_assert_eq!(cell_to_group.len(), rows.n_rows());
    debug_assert_eq!(counts.len(), n_groups * n_vars);
    if let Some(w) = &weights {
        debug_assert_eq!(w.len(), n_vars);
    }
    if rows.n_rows() == 0 || n_groups == 0 || n_vars == 0 {
        return;
    }
    match part {
        ScatterPartition::ByGroup => {
            scatter_by_group(rows, cell_to_group, counts, n_vars, n_groups)
        }
        ScatterPartition::ColumnBlocks(n_blocks) => {
            let n_blocks = n_blocks.clamp(1, n_vars);
            if n_blocks == 1 {
                let mut rows_g: Vec<&mut [f64]> = counts.chunks_mut(n_vars).collect();
                scatter_whole_rows(rows, cell_to_group, &mut rows_g);
            } else {
                scatter_column_blocks(
                    rows,
                    cell_to_group,
                    counts,
                    n_vars,
                    n_groups,
                    n_blocks,
                    weights,
                );
            }
        }
    }
}

/// One block spanning every column: the serial walk, into per-group rows. It
/// leaves the weights histogram alone — the bump would be a second store per
/// nonzero on a loop that is memory-bound already, for a plan only a later
/// multi-block chunk could use, and such a chunk plans from even splits fine.
fn scatter_whole_rows(rows: CsrRows<'_>, cell_to_group: &[usize], rows_g: &mut [&mut [f64]]) {
    for (r, &g) in cell_to_group.iter().enumerate() {
        let (idx, dat) = rows.row(r);
        let dst = &mut *rows_g[g];
        for (&c, &v) in idx.iter().zip(dat) {
            dst[c as usize] += v as f64;
        }
    }
}

/// Per-row block windows for the column-block kernel: row `r`'s nonzeros in
/// block `b` are positions `[at(r, b), at(r, b + 1))` of that row's slice.
/// Planned in one parallel pass that reads each row once (the row is then
/// L1-resident for its `n_blocks − 1` boundary searches), so the block workers
/// never search.
struct RowSplits {
    stride: usize,
    at: Vec<u32>,
}

impl RowSplits {
    fn plan(rows: CsrRows<'_>, blocks: &[std::ops::Range<usize>]) -> Self {
        let n_blocks = blocks.len();
        let stride = n_blocks + 1;
        let mut at = vec![0u32; rows.n_rows() * stride];
        at.par_chunks_mut(stride).enumerate().for_each(|(r, s)| {
            let (idx, _) = rows.row(r);
            let mut pos = 0usize;
            for (b, block) in blocks.iter().enumerate().skip(1) {
                pos += idx[pos..].partition_point(|&c| (c as usize) < block.start);
                s[b] = pos as u32;
            }
            s[n_blocks] = idx.len() as u32;
        });
        Self { stride, at }
    }

    #[inline]
    fn window(&self, r: usize, b: usize) -> (usize, usize) {
        let base = r * self.stride + b;
        (self.at[base] as usize, self.at[base + 1] as usize)
    }
}

/// Column blocks: plan the row windows, carve every group's row into its block
/// pieces, and let each block worker walk every row taking only its window.
fn scatter_column_blocks(
    rows: CsrRows<'_>,
    cell_to_group: &[usize],
    counts: &mut [f64],
    n_vars: usize,
    n_groups: usize,
    n_blocks: usize,
    weights: Option<&mut [u64]>,
) {
    // No histogram: even column counts (`plan_blocks` on all-zero weights).
    let even;
    let blocks = match &weights {
        Some(w) => colblocks::plan_blocks(w, n_blocks),
        None => {
            even = vec![0u64; n_vars];
            colblocks::plan_blocks(&even, n_blocks)
        }
    };
    let splits = RowSplits::plan(rows, &blocks);
    // Regroup the `[n_groups][n_blocks]` pieces as `[n_blocks][n_groups]` so
    // each worker holds one disjoint column window of every group's row.
    let mut per_block: Vec<Vec<&mut [f64]>> = (0..n_blocks)
        .map(|_| Vec::with_capacity(n_groups))
        .collect();
    for row in counts.chunks_mut(n_vars) {
        let mut rest = row;
        for (b, block) in blocks.iter().enumerate() {
            let (piece, tail) = rest.split_at_mut(block.len());
            per_block[b].push(piece);
            rest = tail;
        }
    }
    let weight_parts: Vec<Option<&mut [u64]>> = match weights {
        Some(w) => colblocks::split_by_blocks(w, &blocks, 1)
            .into_iter()
            .map(Some)
            .collect(),
        None => (0..n_blocks).map(|_| None).collect(),
    };
    blocks
        .par_iter()
        .enumerate()
        .zip(per_block)
        .zip(weight_parts)
        .for_each(|(((b, block), mut rows_g), w)| {
            let a = block.start;
            // Two copies of the window loop rather than a per-nonzero branch on
            // whether a histogram is being kept.
            match w {
                Some(w) => {
                    for (r, &g) in cell_to_group.iter().enumerate() {
                        let (lo, hi) = splits.window(r, b);
                        if lo == hi {
                            continue;
                        }
                        let (idx, dat) = rows.row(r);
                        let dst = &mut *rows_g[g];
                        for (&c, &v) in idx[lo..hi].iter().zip(&dat[lo..hi]) {
                            let c = c as usize - a;
                            dst[c] += v as f64;
                            w[c] += 1;
                        }
                    }
                }
                None => {
                    for (r, &g) in cell_to_group.iter().enumerate() {
                        let (lo, hi) = splits.window(r, b);
                        if lo == hi {
                            continue;
                        }
                        let (idx, dat) = rows.row(r);
                        let dst = &mut *rows_g[g];
                        for (&c, &v) in idx[lo..hi].iter().zip(&dat[lo..hi]) {
                            dst[c as usize - a] += v as f64;
                        }
                    }
                }
            }
        });
}

/// One worker per group row. Cells are bucketed by group with a counting sort
/// (two flat buffers, not one `Vec` per group — a 10k-group file streams 600
/// shards) and visited ascending within the group; each row's nonzeros are
/// taken in stored order, so this is exact for unsorted and duplicated
/// columns alike.
fn scatter_by_group(
    rows: CsrRows<'_>,
    cell_to_group: &[usize],
    counts: &mut [f64],
    n_vars: usize,
    n_groups: usize,
) {
    let mut offsets = vec![0usize; n_groups + 1];
    for &g in cell_to_group {
        offsets[g + 1] += 1;
    }
    for g in 0..n_groups {
        offsets[g + 1] += offsets[g];
    }
    let mut cursor = offsets[..n_groups].to_vec();
    let mut cells = vec![0u32; cell_to_group.len()];
    for (r, &g) in cell_to_group.iter().enumerate() {
        cells[cursor[g]] = r as u32;
        cursor[g] += 1;
    }
    // One task per group row, scheduled by the ambient pool. Groups differ in
    // size, so fine tasks balance where a fixed number of coarse ones does not:
    // capping this at `block_count` tasks (contiguous runs of group rows) was
    // measured 40 % slower on the streaming arms of the crate bench — eleven
    // unequal tasks per 4 096-row shard against sixty-four rayon can steal.
    // `SCX_ACCEL_NUM_THREADS` therefore caps the *block* splits (column blocks,
    // a CSC run's columns), not this one; `RAYON_NUM_THREADS` sizes the pool.
    counts
        .par_chunks_mut(n_vars)
        .enumerate()
        .for_each(|(g, dst)| {
            for &r in &cells[offsets[g]..offsets[g + 1]] {
                let (idx, dat) = rows.row(r as usize);
                for (&c, &v) in idx.iter().zip(dat) {
                    dst[c as usize] += v as f64;
                }
            }
        });
}

/// Scatter a whole in-memory matrix: one partition decision, one dispatch, no
/// histogram (nothing plans after it, so the blocks are even and the kernel
/// skips the per-nonzero bump). A whole matrix always holds every group, so
/// the per-shard routing the streaming path needs (a shard of a group-sorted
/// file holds two groups) has nothing to buy here, and one dispatch beats a
/// sequence of row-chunk waves.
fn scatter_matrix(
    rows: CsrRows<'_>,
    cell_to_group: &[usize],
    counts: &mut [f64],
    n_vars: usize,
    n_groups: usize,
) {
    let part = choose_partition(rows, cell_to_group, n_groups, n_vars);
    scatter_rows(rows, cell_to_group, counts, n_vars, n_groups, part, None);
}

/// Divide every group's row by its cell count, in place. One division per
/// element, so the split across workers cannot move a bit.
pub(crate) fn apply_mean(counts: &mut [f64], cell_counts: &[usize], n_vars: usize) {
    if n_vars == 0 || cell_counts.is_empty() {
        return;
    }
    debug_assert_eq!(counts.len(), cell_counts.len() * n_vars);
    counts
        .par_chunks_mut(n_vars)
        .zip(cell_counts.par_iter())
        .for_each(|(row, &cc)| {
            if cc > 0 {
                let cc = cc as f64;
                for v in row {
                    *v /= cc;
                }
            }
        });
}

/// Streaming pseudobulk aggregation over a CSR [`ShardSource`].
///
/// Iterates shards one at a time, accumulating per-group sums without
/// materializing the full matrix.
///
/// Generic over the source rather than taking a `BackedCsrReader`, because
/// `obs_groups` is indexed by *visible* cell: a caller holding a subset SCX
/// handle must pass that handle's view (`as_shard_source()`), not the reader
/// underneath it, or the group labels line up against the wrong rows.
///
/// Shards are consumed in order on the calling thread (the offset-dependent
/// `global_row` cursor needs that); within a shard the scatter is partitioned
/// across the rayon pool by output — one group's row per worker, or a column
/// block of every row when the groups are too few or too skewed for that —
/// which is bit-identical to the serial loop on any thread count; see the
/// kernel comment above. Nothing about the result depends on
/// `RAYON_NUM_THREADS` or `SCX_ACCEL_NUM_THREADS`; they bound speed only.
///
/// # Arguments
/// * `source` — CSR shard source for shard-by-shard iteration.
/// * `obs_groups` — Per-cell group labels for each groupby column.
///   `obs_groups[col_idx][cell_idx]` is the label for cell `cell_idx` in column `col_idx`.
/// * `groupby_columns` — Column names from obs (e.g., `["perturbation", "donor"]`).
/// * `gene_names` — Gene names (length = n_vars).
/// * `method` — Aggregation method (Sum or Mean).
/// * `min_cells_per_group` — Groups with fewer cells are excluded from the result.
pub fn pseudobulk_aggregate<S: ShardSource + Sync>(
    source: &S,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    // Accumulate counts and cell counts.
    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];

    // Count cells per group.
    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }

    // Stream shards with ordered decode-prefetch (2.1). Counts land in
    // per-group bins keyed by the global cell index, so shards must be consumed
    // in order for the `global_row` cursor to map cells correctly — StableOrder
    // is the only valid mode. Pseudobulk is a **single pass**, so use the
    // *uncached* prefetch variant: `read_shard` (not the LRU `read_shard_arc`),
    // preserving the pre-2.1 `read_shard_uncached` behaviour so this pass does
    // not warm/evict the shared shard cache (review feedback).
    //
    // The per-shard scatter runs *inside* the consume closure, partitioned
    // across the pool — the sanctioned shape (the driver must not be called
    // from a rayon region, but may spawn one). `weights` carries the column
    // histogram across shards so each shard's blocks are planned from the ones
    // before it.
    let mut weights = vec![0u64; n_vars];
    let mut global_row = 0usize;
    crate::prefetch::for_each_shard_ordered_uncached(
        source,
        crate::prefetch::prefetch_depth(),
        |shard_idx, shard_csr| {
            let _r = scx_format_io::reduction_guard();
            let shard_n_rows = shard_csr.n_rows();
            if global_row + shard_n_rows > n_obs {
                return Err(crate::AccelError::ShapeError(format!(
                    "pseudobulk: shard {shard_idx} carries rows {global_row}..{} but n_obs = {n_obs}",
                    global_row + shard_n_rows
                )));
            }
            let rows = CsrRows::of(&shard_csr);
            let groups = &cell_to_group[global_row..global_row + shard_n_rows];
            let part = choose_partition(rows, groups, n_groups, n_vars);
            scatter_rows(
                rows,
                groups,
                &mut counts,
                n_vars,
                n_groups,
                part,
                Some(&mut weights),
            );
            global_row += shard_n_rows;
            Ok(())
        },
    )?;
    if global_row != n_obs {
        return Err(crate::AccelError::ShapeError(format!(
            "pseudobulk: shards covered {global_row} rows but n_obs = {n_obs}; every cell \
             was counted in `cell_counts`, so a mean over these sums would be wrong"
        )));
    }

    if method == AggregationMethod::Mean {
        apply_mean(&mut counts, &cell_counts, n_vars);
    }

    // Filter by min_cells_per_group.
    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// In-memory pseudobulk aggregation from a pre-loaded `ScxCsr`.
///
/// Same algorithm as `pseudobulk_aggregate()` but operates on a single
/// already-decoded CSR matrix instead of streaming shards. The matrix need
/// not be canonical: unsorted or duplicated column indices (a scipy CSR
/// handed through without `sort_indices()` / `sum_duplicates()`) take the
/// order-preserving group partition and give the same bits the serial loop
/// would — a duplicate coordinate contributes both of its values.
pub fn pseudobulk_aggregate_inmemory(
    csr: &scx_sparse::ScxCsr,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = csr.shape;

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];

    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }

    scatter_matrix(
        CsrRows::of(csr),
        &cell_to_group,
        &mut counts,
        n_vars,
        n_groups,
    );

    if method == AggregationMethod::Mean {
        apply_mean(&mut counts, &cell_counts, n_vars);
    }

    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// In-memory pseudobulk aggregation from borrowed CSR slices.
///
/// Same algorithm as `pseudobulk_aggregate_inmemory()` but operates on
/// borrowed slices (`&[i64]`, `&[i32]`, `&[f32]`) instead of requiring
/// an owning `ScxCsr`. This enables zero-copy aggregation from numpy
/// arrays via `PyReadonlyArray1` without cloning the data.
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_aggregate_from_slices(
    shape: (usize, usize),
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;
    validate_csr_slices(indptr, indices, data, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let mut counts = vec![0.0f64; n_groups * n_vars];
    let mut cell_counts = vec![0usize; n_groups];

    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }

    scatter_matrix(
        CsrRows {
            indptr,
            indices,
            data,
        },
        &cell_to_group,
        &mut counts,
        n_vars,
        n_groups,
    );

    if method == AggregationMethod::Mean {
        apply_mean(&mut counts, &cell_counts, n_vars);
    }

    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// In-memory pseudobulk aggregation from a row-major dense `f32` matrix.
///
/// Avoids the `scipy.sparse.csr_matrix(dense_array)` densification round-trip
/// that the CSR-based paths pay when the caller's `X` is already dense.
/// Builds the same `PseudobulkResult` shape as `pseudobulk_aggregate_inmemory`
/// / `pseudobulk_aggregate_from_slices`.
///
/// At Replogle scale (n_obs ≈ 24K cells × n_vars ≈ 18K genes) the dense path
/// is ~50–100× faster than going through `scipy.sparse.csr_matrix`, because
/// the CSR conversion scans every f32 looking for non-zeros and materialises
/// 1.7 GB of `(indices, data)` arrays just to be summed back into a dense
/// per-group means matrix. The dense path skips that intermediate altogether.
///
/// Parallelisation is over groups (each thread writes to its own contiguous
/// row of `means`, so no shared-accumulator contention or thread-local
/// `n_groups × n_vars` blow-up — the per-thread allocator pattern would have
/// allocated ~11 GB across 32 threads for typical Replogle shapes).
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_aggregate_dense(
    data: &[f32],
    shape: (usize, usize),
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    method: AggregationMethod,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;
    if data.len() != n_obs * n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "data length {} != n_obs ({}) × n_vars ({}) = {}",
            data.len(),
            n_obs,
            n_vars,
            n_obs * n_vars,
        )));
    }

    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();

    // Invert cell_to_group → per-group list of cell row indices. One pass,
    // O(n_obs) time + O(n_obs + n_groups) memory.
    let mut cells_by_group: Vec<Vec<u32>> = vec![Vec::new(); n_groups];
    for (cell, &g) in cell_to_group.iter().enumerate() {
        cells_by_group[g].push(cell as u32);
    }
    let cell_counts: Vec<usize> = cells_by_group.iter().map(|c| c.len()).collect();

    // Allocate the result `[n_groups × n_vars]` matrix once, then have each
    // group sum (and optionally mean-normalise) its own cells into its
    // dedicated row in parallel. No shared mutable state across threads:
    // each thread owns a disjoint row range. Folding the divide into the
    // same loop avoids a sequential `n_groups × n_vars` post-pass (~432M
    // divisions at Replogle scale).
    let mut counts = vec![0.0f64; n_groups * n_vars];
    let want_mean = method == AggregationMethod::Mean;
    counts
        .par_chunks_mut(n_vars)
        .zip(cells_by_group.par_iter())
        .with_min_len(1)
        .for_each(|(dst, cells)| {
            for &cell in cells {
                let src = &data[cell as usize * n_vars..(cell as usize + 1) * n_vars];
                for (d, &s) in dst.iter_mut().zip(src.iter()) {
                    *d += s as f64;
                }
            }
            if want_mean && !cells.is_empty() {
                let cc = cells.len() as f64;
                for d in dst.iter_mut() {
                    *d /= cc;
                }
            }
        });

    filter_and_build_result(
        counts,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// Shape-check borrowed CSR slices before the kernel walks them — the
/// invariants `ScxCsr::new` enforces: `indptr` has `n_obs + 1` non-decreasing
/// entries from `0` to exactly `indices.len()`, `indices` / `data` have one
/// entry per nonzero, and every column index lies in `0..n_vars`. The kernel
/// indexes rows on rayon workers, so a malformed triple would otherwise panic
/// there — or silently drop leading / trailing nonzeros, or a short `data`'s
/// tail through `zip` — instead of erroring.
fn validate_csr_slices(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_obs: usize,
    n_vars: usize,
) -> Result<()> {
    let shape_err = |msg: String| Err(crate::AccelError::ShapeError(msg));
    if indptr.len() != n_obs + 1 {
        return shape_err(format!(
            "indptr has {} entries but n_obs + 1 = {}",
            indptr.len(),
            n_obs + 1
        ));
    }
    if indices.len() != data.len() {
        return shape_err(format!(
            "indices has {} entries but data has {}",
            indices.len(),
            data.len()
        ));
    }
    if indptr[0] != 0 {
        return shape_err(format!(
            "indptr[0] = {} but a CSR's first offset is 0",
            indptr[0]
        ));
    }
    if let Some(r) = indptr.windows(2).position(|w| w[1] < w[0]) {
        return shape_err(format!(
            "indptr is not non-decreasing at row {r}: {} > {}",
            indptr[r],
            indptr[r + 1]
        ));
    }
    let last = indptr[n_obs];
    if last as u64 != indices.len() as u64 {
        return shape_err(format!(
            "indptr ends at {last} but indices has {} entries",
            indices.len()
        ));
    }
    // One read pass over `indices`, spread over the pool; the serial search for
    // the message runs only once the check has failed.
    if !indices.par_iter().all(|&c| c >= 0 && (c as usize) < n_vars) {
        let k = indices
            .iter()
            .position(|&c| c < 0 || (c as usize) >= n_vars)
            .unwrap_or(0);
        return shape_err(format!(
            "indices[{k}] = {} is out of range for n_vars = {n_vars}",
            indices[k]
        ));
    }
    Ok(())
}

/// Validate common inputs for both streaming and in-memory paths.
fn validate_inputs(
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    n_obs: usize,
    n_vars: usize,
) -> Result<()> {
    if obs_groups.is_empty() {
        return Err(crate::AccelError::InvalidInput(
            "obs_groups must not be empty".to_string(),
        ));
    }
    if obs_groups.len() != groupby_columns.len() {
        return Err(crate::AccelError::InvalidInput(format!(
            "obs_groups has {} columns but groupby_columns has {}",
            obs_groups.len(),
            groupby_columns.len()
        )));
    }
    for (i, col) in obs_groups.iter().enumerate() {
        if col.len() != n_obs {
            return Err(crate::AccelError::InvalidInput(format!(
                "obs_groups[{}] has {} entries but n_obs = {}",
                i,
                col.len(),
                n_obs
            )));
        }
    }
    if gene_names.len() != n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "gene_names has {} entries but n_vars = {}",
            gene_names.len(),
            n_vars
        )));
    }
    Ok(())
}

/// Filter groups by min_cells and build the final `PseudobulkResult`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_and_build_result(
    counts: Vec<f64>,
    group_labels: Vec<Vec<String>>,
    groupby_columns: &[String],
    cell_counts: Vec<usize>,
    gene_names: &[String],
    n_groups: usize,
    n_vars: usize,
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    // Identify groups that pass the filter.
    let kept: Vec<usize> = (0..n_groups)
        .filter(|&g| cell_counts[g] >= min_cells_per_group)
        .collect();

    if kept.len() == n_groups {
        // No filtering needed.
        return Ok(PseudobulkResult {
            counts,
            group_labels,
            groupby_columns: groupby_columns.to_vec(),
            cell_counts,
            gene_names: gene_names.to_vec(),
            n_groups,
            n_vars,
        });
    }

    let new_n_groups = kept.len();
    let mut new_counts = Vec::with_capacity(new_n_groups * n_vars);
    let mut new_labels = Vec::with_capacity(new_n_groups);
    let mut new_cell_counts = Vec::with_capacity(new_n_groups);

    for &g in &kept {
        new_counts.extend_from_slice(&counts[g * n_vars..(g + 1) * n_vars]);
        new_labels.push(group_labels[g].clone());
        new_cell_counts.push(cell_counts[g]);
    }

    Ok(PseudobulkResult {
        counts: new_counts,
        group_labels: new_labels,
        groupby_columns: groupby_columns.to_vec(),
        cell_counts: new_cell_counts,
        gene_names: gene_names.to_vec(),
        n_groups: new_n_groups,
        n_vars,
    })
}

// ──────────────────────────────────────────────────────────────────────────
// GPU pseudobulk means.
//
// GPU-accelerate ONLY the aggregation (the cell-count-scaling step): produce
// per-group means `[n_groups × n_vars]` f64 by wrapping the scx-gpu DE
// pseudobulk primitives, then reuse `filter_and_build_result` so the
// `PseudobulkResult` is identical to the CPU `pseudobulk_aggregate*` output
// (same lexicographic group ordering, min-cells filtering, and cell counts).
// The five bulk metrics run downstream on the host. GPU sums accumulate in f64
// with the identity pre-transform, matching the CPU f32→f64 accumulate.
// ──────────────────────────────────────────────────────────────────────────

/// Sorted group mapping shared by the GPU means paths: per-cell group id (i32,
/// in the same lexicographic order as [`build_group_mapping`]), the sorted
/// group labels, and per-group cell counts.
/// Classify a `gpu_pseudobulk_means_csr` failure.
///
/// That call's shard pass guards the visible-row coverage its device kernel
/// depends on, and reports a plan disagreement as `GpuError::InvalidShard` —
/// the only `InvalidShard` it can produce. Blanket-wrapping every `GpuError` in
/// [`AccelError::LinAlg`] relabelled that as a device failure, which is the
/// asymmetry the GPU DE passes grew `GpuRowCursor` to avoid: a coverage error is
/// a `ShapeError` on both backends and on both halves of the check.
#[cfg(feature = "gpu")]
fn pseudobulk_gpu_err(context: &str, e: scx_gpu::GpuError) -> crate::AccelError {
    match e {
        scx_gpu::GpuError::InvalidShard(msg) => crate::AccelError::ShapeError(msg),
        other => crate::AccelError::LinAlg(format!("{context}: {other}")),
    }
}

#[cfg(all(test, feature = "gpu"))]
mod gpu_err_tests {
    use super::pseudobulk_gpu_err;

    /// A coverage error must keep its classification across the crate boundary,
    /// and a device error must keep its context prefix. Runs on a CPU host: it
    /// is a match on an error value, with no device in it.
    #[test]
    fn a_coverage_error_stays_a_shape_error_and_others_stay_linalg() {
        let shape = pseudobulk_gpu_err(
            "ctx",
            scx_gpu::GpuError::InvalidShard("cover 4 rows but n_obs = 6".into()),
        );
        assert!(
            matches!(shape, crate::AccelError::ShapeError(ref m) if m == "cover 4 rows but n_obs = 6"),
            "{shape:?}"
        );
        // Unprefixed on purpose: the message is the cursor's own and already
        // names the pass, so re-prefixing would double it.

        let other = pseudobulk_gpu_err("ctx", scx_gpu::GpuError::OutOfMemory("vram".into()));
        assert!(
            matches!(other, crate::AccelError::LinAlg(ref m) if m.starts_with("ctx: ")),
            "{other:?}"
        );
    }
}

#[cfg(feature = "gpu")]
fn gpu_group_plan(
    obs_groups: &[Vec<String>],
    n_obs: usize,
) -> (Vec<i32>, Vec<Vec<String>>, Vec<usize>) {
    let (cell_to_group, group_labels) = build_group_mapping(obs_groups, n_obs);
    let n_groups = group_labels.len();
    let mut cell_counts = vec![0usize; n_groups];
    for &g in &cell_to_group {
        cell_counts[g] += 1;
    }
    let cell_to_group_i32 = cell_to_group.iter().map(|&g| g as i32).collect();
    (cell_to_group_i32, group_labels, cell_counts)
}

/// GPU pseudobulk **means** over a CSR [`ShardSource`]. Mirrors
/// [`pseudobulk_aggregate`] with `AggregationMethod::Mean`, but streams the
/// shards in-VRAM and folds them with the DE pseudobulk kernel on the GPU.
///
/// Generic for the same reason as [`pseudobulk_aggregate`]: `obs_groups` is
/// per *visible* cell, so a subset SCX handle must hand over its view.
#[cfg(feature = "gpu")]
pub fn pseudobulk_means_gpu_streaming<S: ShardSource + Sync>(
    dev: &scx_gpu::GpuDevice,
    source: &S,
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let n_obs = source.n_obs();
    let n_vars = source.n_vars();
    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels, cell_counts) = gpu_group_plan(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let means = scx_gpu::gpu_pseudobulk_means_csr(
        dev,
        source,
        &cell_to_group,
        n_groups,
        n_vars,
        &cell_counts,
    )
    .map_err(|e| pseudobulk_gpu_err("GPU pseudobulk means (streaming)", e))?;

    filter_and_build_result(
        means,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// GPU pseudobulk means from in-memory CSR slices (scipy CSR). Mirrors
/// [`pseudobulk_aggregate_from_slices`] with `Mean`; wraps the borrowed CSR in
/// a single-shard source for the streaming kernel.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn pseudobulk_means_gpu_from_slices(
    dev: &scx_gpu::GpuDevice,
    shape: (usize, usize),
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;
    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels, cell_counts) = gpu_group_plan(obs_groups, n_obs);
    let n_groups = group_labels.len();

    let csr =
        scx_sparse::ScxCsr::new_unchecked(shape, indptr.to_vec(), indices.to_vec(), data.to_vec());
    let source = scx_format_io::shard_source::SingleShardSource { csr: &csr };

    let means = scx_gpu::gpu_pseudobulk_means_csr(
        dev,
        &source,
        &cell_to_group,
        n_groups,
        n_vars,
        &cell_counts,
    )
    .map_err(|e| pseudobulk_gpu_err("GPU pseudobulk means (csr)", e))?;

    filter_and_build_result(
        means,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

/// GPU pseudobulk means from a dense row-major `[n_obs × n_vars]` f32 matrix
/// (in-memory dense `X` or an `obsm` embedding). Mirrors
/// [`pseudobulk_aggregate_dense`] with `Mean`.
#[cfg(feature = "gpu")]
pub fn pseudobulk_means_gpu_dense(
    dev: &scx_gpu::GpuDevice,
    data: &[f32],
    shape: (usize, usize),
    obs_groups: &[Vec<String>],
    groupby_columns: &[String],
    gene_names: &[String],
    min_cells_per_group: usize,
) -> Result<PseudobulkResult> {
    let (n_obs, n_vars) = shape;
    if data.len() != n_obs * n_vars {
        return Err(crate::AccelError::InvalidInput(format!(
            "data length {} != n_obs ({}) × n_vars ({})",
            data.len(),
            n_obs,
            n_vars
        )));
    }
    validate_inputs(obs_groups, groupby_columns, gene_names, n_obs, n_vars)?;

    let (cell_to_group, group_labels, cell_counts) = gpu_group_plan(obs_groups, n_obs);
    let n_groups = group_labels.len();

    // Concatenated per-group cell lists + prefix offsets that the dense kernel
    // consumes (built in the sorted group order from `gpu_group_plan`).
    let mut group_offsets = vec![0i32; n_groups + 1];
    for g in 0..n_groups {
        group_offsets[g + 1] = group_offsets[g] + cell_counts[g] as i32;
    }
    let mut cursor: Vec<i32> = group_offsets[..n_groups].to_vec();
    let mut all_group_cells = vec![0i32; n_obs];
    for (cell, &g) in cell_to_group.iter().enumerate() {
        let g = g as usize;
        all_group_cells[cursor[g] as usize] = cell as i32;
        cursor[g] += 1;
    }

    let means = scx_gpu::gpu_pseudobulk_means_dense(
        dev,
        data,
        n_obs,
        n_vars,
        &all_group_cells,
        &group_offsets,
        n_groups,
        &cell_counts,
    )
    .map_err(|e| crate::AccelError::LinAlg(format!("GPU pseudobulk means (dense): {e}")))?;

    filter_and_build_result(
        means,
        group_labels,
        groupby_columns,
        cell_counts,
        gene_names,
        n_groups,
        n_vars,
        min_cells_per_group,
    )
}

#[cfg(test)]
#[path = "pseudobulk_tests.rs"]
mod tests;
