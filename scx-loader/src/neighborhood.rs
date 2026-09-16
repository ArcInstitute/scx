//! Neighbourhood plan builders — W7 steps 1 and 2.
//!
//! A neighbourhood *is* a cell set with the centre role-tagged, so the gather
//! primitive R6 needs already exists. What was missing is the plan. These two
//! builders turn a stored `obsp` graph, or `obsm["spatial"]` coordinates, into
//! the same `(file_ids, rows, role_tags, set_offsets)` plans
//! `SparseCellSetDataset::gather` / `iter_with_plans` already consume — which
//! makes R6 an R2 workload and hands it every R2 improvement for free.
//!
//! # Roles, and the order the rest of a set comes in
//!
//! Position 0 of a set is the **centre**, `role_tag = 0`; every other member is
//! a neighbour, `role_tag = 1`. That maps onto Nicheformer's cell + context
//! tokens and onto the per-set kernels in [`crate::tokenize`].
//!
//! ⚠️ **The two builders order the neighbours differently, and neither is
//! wrong.** The graph builder emits them **column-ascending** — whether or not
//! `k` selected them — because a stored graph's columns are the only order it
//! carries and re-sorting by weight would make *which* edges `k` picked also
//! change *where* they land. The coordinate builder emits them **nearest
//! first**, because it computed the distances and that is the order it
//! computed them in. So position 1 is the nearest neighbour on the coordinate
//! path and merely the lowest-numbered one on the graph path. A consumer that
//! cares about proximity ordering must not assume the graph path gives it.
//!
//! # Row space is physical, and deletions are dropped, not renumbered
//!
//! `SparseCellSetPlan::rows` are physical file rows — the loader has no logical
//! row space — so these builders work physically throughout. A caller-supplied
//! keep mask (`true` = the row survives) is applied as a graph decision, not a
//! coordinate transform:
//!
//! - a **deleted centre yields no set at all**, so the plan list is shorter
//!   than `n_obs` and [`NeighborhoodPlans::centers`] says which centres
//!   survived;
//! - a **deleted neighbour is dropped** from every set it appears in.
//!
//! ⚠️ What "dropped" costs a set depends on whether `k` is in play, and the two
//! are different answers rather than an inconsistency:
//!
//! - **Without `k`** — every stored edge, or a radius query — a set is simply
//!   short by however many of its neighbours are gone. Nothing replaces them.
//! - **With `k`**, deleted rows are **not candidates**, so the `k` best of the
//!   *live* neighbours are taken: a centre whose nearest neighbour is deleted
//!   gets its next one instead and the set is still `k` wide. That is what
//!   asking for `k` neighbours means. The alternative — select `k` in physical
//!   space and then delete from the selection — returns sets of varying width
//!   for no stated benefit. A set is short under `k` only when the live
//!   population runs out.
//!
//! That is the same rule `filter_coo_obsp_by_kept_rows` applies on the
//! `to_anndata` path (drop the edge if either endpoint is gone); the difference
//! is that nothing here renumbers, because the gather wants physical rows.
//!
//! # `file_id` is a manifest position, not a file identity
//!
//! Every emitted row carries the `file_id` the caller supplies, and that number
//! means "this file's index in the `SparseCellSetDataset` manifest you will
//! gather with". Build plans from file A and feed them to a dataset whose
//! manifest puts A third, and you will gather the *first* file's rows with no
//! error anywhere. There is no default that could guess it, so it is required.
//!
//! # Why not `scx_accel::neighbors::cpu::build_knn_graph`
//!
//! It is reachable — `scx-loader` already depends on `scx-accel` — and it is
//! deliberately not used. It dispatches to approximate HNSW above
//! `EXACT_KNN_NOBS_THRESHOLD = 5_000` rows, has no radius mode, and builds a
//! full connectivity CSR on the way. A builder whose acceptance gate is
//! equality against a numpy reference cannot be approximate, and a spatial
//! sample is above that threshold roughly always.
//!
//! # Parallelism
//!
//! Both builders are serial. The per-centre search is embarrassingly parallel,
//! but the benchmark arms report `plan_build_s` apart from the gather, so
//! parallelising it is a change to make when a measurement asks for it rather
//! than on the way past.

use crate::error::{LoaderError, Result};
use crate::sparse_cellset::SparseCellSetPlan;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use scx_format_io::{BackedDenseReader, BackedPairwiseReader, PairwiseRows, ScxReader};

/// Which end of a stored graph's weights `k` keeps.
///
/// There is no safe default per *key*, and inferring one from the key's name
/// would be worse than asking: scanpy writes `obsp["connectivities"]` where a
/// larger weight means *closer* and `obsp["distances"]` where a larger weight
/// means *farther*, so the same `k` over the same neighbourhood picks opposite
/// ends of it. A `k` against a distance graph under [`WeightOrder::Desc`]
/// returns that cell's **farthest** stored neighbours — structurally valid, and
/// the opposite of what was asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightOrder {
    /// Keep the largest weights — a connectivity / affinity / similarity graph.
    Desc,
    /// Keep the smallest weights — a distance graph.
    Asc,
}

/// Role tag of a set's centre cell.
pub const ROLE_CENTER: i32 = 0;
/// Role tag of a neighbour cell.
pub const ROLE_NEIGHBOR: i32 = 1;

/// Run-constant knobs shared by both builders.
#[derive(Clone, Copy, Debug)]
pub struct NeighborhoodConfig {
    /// Emit the centre at position 0 with [`ROLE_CENTER`]. With `false` a set
    /// is its neighbours only, and a centre whose neighbours are all gone
    /// yields an empty set rather than a singleton.
    pub include_center: bool,
    /// Position of this file in the `SparseCellSetDataset` manifest. See the
    /// module docs — this is not a file identity.
    pub file_id: u32,
}

/// The built plans, one per surviving centre, plus the centres themselves.
///
/// `centers` is parallel to `plans` and is what tells a caller which centres
/// were dropped: a deleted centre produces neither entry.
#[derive(Clone, Debug, Default)]
pub struct NeighborhoodPlans {
    pub plans: Vec<SparseCellSetPlan>,
    pub centers: Vec<u64>,
}

impl NeighborhoodPlans {
    /// Number of sets built.
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    /// True when no centre survived.
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
    }

    fn push_set(&mut self, center: u64, rows: Vec<u64>, cfg: NeighborhoodConfig) {
        let n = rows.len();
        let mut role_tags = Vec::with_capacity(n);
        if cfg.include_center && n > 0 {
            role_tags.push(ROLE_CENTER);
            role_tags.extend(std::iter::repeat_n(ROLE_NEIGHBOR, n - 1));
        } else {
            role_tags.extend(std::iter::repeat_n(ROLE_NEIGHBOR, n));
        }
        self.plans.push(SparseCellSetPlan {
            file_ids: vec![cfg.file_id; n],
            rows,
            role_tags,
            set_offsets: vec![0, n as i64],
        });
        self.centers.push(center);
    }
}

fn kept(keep: Option<&[bool]>, row: u64) -> bool {
    match keep {
        Some(mask) => mask.get(row as usize).copied().unwrap_or(false),
        None => true,
    }
}

// ---------------------------------------------------------------------------
// Step 1 — graph-driven plans
// ---------------------------------------------------------------------------

/// Append one chunk's worth of graph-driven plans.
///
/// `rows` is a physical row range of an `obsp` graph as
/// [`scx_format_io::BackedPairwiseReader::read_rows_range`] returns it. With
/// `k = None` every stored edge becomes a neighbour, in the graph's own
/// column-ascending order; with `k = Some(k)` the `k` edges at the
/// `weight_order` end are taken, ties broken by column ascending — a declared
/// rule, since a stored graph carries no order of its own.
///
/// ⚠️ `weight_order` is not cosmetic and has no safe per-key default. Scanpy's
/// `obsp["connectivities"]` is an affinity (larger = closer,
/// [`WeightOrder::Desc`]); its `obsp["distances"]` is a metric (larger =
/// farther, [`WeightOrder::Asc`]). The wrong one against a distance graph
/// returns each cell's `k` **farthest** stored neighbours — a structurally
/// valid plan that answers the opposite question. Inferring it from the key's
/// name would be worse than asking, because the name is the caller's.
///
/// Non-finite weights sort **last** whatever the order, so `k` never takes one
/// over a finite competitor: a distance graph's `inf` for "not connected" is
/// never chosen as a near neighbour.
///
/// A self-loop (an edge from a row to itself) is dropped when the centre is
/// already being emitted, rather than letting the centre appear twice and be
/// silently double-weighted by whatever consumes the set.
pub fn plans_from_graph_chunk(
    rows: &PairwiseRows,
    keep: Option<&[bool]>,
    k: Option<usize>,
    weight_order: WeightOrder,
    cfg: NeighborhoodConfig,
    out: &mut NeighborhoodPlans,
) {
    let mut scratch: Vec<(f32, i64)> = Vec::new();
    for i in 0..rows.n_rows as usize {
        let center = rows.row_start + i as u64;
        if !kept(keep, center) {
            continue;
        }
        let (cols, vals) = rows.row(i);
        scratch.clear();
        for (&c, &v) in cols.iter().zip(vals) {
            // `BackedPairwiseReader` refuses a negative column, but this
            // function is public and takes any `PairwiseRows`: `c as u64` on a
            // negative turns it into a row index above 2^63 that `kept(None, _)`
            // waves through and the gather then fails on, far from here.
            if c < 0 {
                continue;
            }
            let c_u = c as u64;
            if !kept(keep, c_u) {
                continue;
            }
            if c_u == center && cfg.include_center {
                continue;
            }
            scratch.push((v, c));
        }
        if let Some(k) = k {
            scratch.sort_by(|a, b| cmp_weight(*a, *b, weight_order));
            scratch.truncate(k);
            // Restore column order among the chosen edges, so which edges `k`
            // picked does not also change the order they are emitted in. Note
            // this makes the NEIGHBOURS ascending, not the set: the centre is
            // pushed first and its row may be larger than any of them.
            scratch.sort_by_key(|&(_, c)| c);
        }
        let mut set: Vec<u64> = Vec::with_capacity(scratch.len() + 1);
        if cfg.include_center {
            set.push(center);
        }
        set.extend(scratch.iter().map(|&(_, c)| c as u64));
        out.push_set(center, set, cfg);
    }
}

// ---------------------------------------------------------------------------
// Step 2 — coordinate-driven plans over a uniform grid
// ---------------------------------------------------------------------------

/// How the coordinate builder chooses neighbours.
#[derive(Clone, Copy, Debug)]
pub enum CoordQuery {
    /// The `k` nearest kept points, excluding the centre.
    Knn(usize),
    /// Every kept point within `radius` (inclusive), excluding the centre.
    Radius(f32),
}

/// Highest coordinate dimensionality the grid will accept.
///
/// Not a tuning knob — a structural bound. `Grid::walk_ring` enumerates the
/// `(2r+1)^d` offset tuples of a Chebyshev shell and rejects the out-of-range
/// ones at the leaf, so `d` sits in an exponent. At `d = 3` ring 1 is 27 visits;
/// at `d = 50` — an ordinary `obsm["X_pca"]`, and a plausible typo for
/// `obsm["spatial"]` — it is 3^50 and the call never returns. Spatial
/// coordinates are 2-D (Visium, Slide-seq) or 3-D (Xenium, MERFISH); anything
/// wider is a different problem and belongs on the graph path, over a kNN
/// somebody else built.
pub const MAX_COORD_DIMS: usize = 3;

/// Upper bound on grid cells, as a multiple of the point count. A grid finer
/// than this buys nothing and costs a bucket array; the cell size is widened
/// until the bound holds.
const MAX_CELLS_PER_POINT: usize = 4;

/// Build coordinate-driven plans from a flat row-major `coords` array of
/// `n × d` values.
///
/// A per-file uniform grid, bucketed by counting sort, then brute force within
/// the centre's cell and the rings around it: O(n) to build, no on-disk index,
/// and no approximation. Ties are ordered **(squared distance ascending, then
/// row ascending)**, which is a declared rule rather than an inherited one.
///
/// Non-finite coordinates are **refused**, not clipped. The tokenisation
/// kernels clip a NaN value to zero because zero is a meaningful expression
/// level; a NaN coordinate has no defensible grid cell, and bucketing it
/// somewhere would move a cell into a neighbourhood it is not in.
pub fn plans_from_coords(
    coords: &[f32],
    d: usize,
    keep: Option<&[bool]>,
    query: CoordQuery,
    cfg: NeighborhoodConfig,
) -> Result<NeighborhoodPlans> {
    if d == 0 || d > MAX_COORD_DIMS {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "plans_from_coords: coordinate dimensionality must be 1..={MAX_COORD_DIMS}, got \
                 {d}. The grid search is exponential in d — at d=50 it does not return — and \
                 spatial coordinates are 2-D or 3-D. To build neighbourhoods in a wide embedding, \
                 write a kNN graph into obsp (scanpy: sc.pp.neighbors(use_rep=...)) and use the \
                 graph builder instead."
            ),
        });
    }
    if !coords.len().is_multiple_of(d) {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "plans_from_coords: coords len {} is not a multiple of d={d}",
                coords.len()
            ),
        });
    }
    let n = coords.len() / d;
    if let CoordQuery::Radius(r) = query {
        if !(r.is_finite() && r > 0.0) {
            return Err(LoaderError::ConfigError {
                reason: format!("plans_from_coords: radius must be finite and > 0, got {r}"),
            });
        }
    }
    if let CoordQuery::Knn(k) = query {
        if k == 0 {
            return Err(LoaderError::ConfigError {
                reason: "plans_from_coords: k must be >= 1".into(),
            });
        }
    }
    if let Some(mask) = keep {
        if mask.len() != n {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "plans_from_coords: keep mask has {} entries but there are {n} points",
                    mask.len()
                ),
            });
        }
    }

    let members: Vec<u64> = (0..n as u64).filter(|&r| kept(keep, r)).collect();
    let mut out = NeighborhoodPlans::default();
    if members.is_empty() {
        return Ok(out);
    }
    for &r in &members {
        for j in 0..d {
            let v = coords[r as usize * d + j];
            if !v.is_finite() {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "plans_from_coords: row {r} dimension {j} is {v}, which has no grid cell; \
                         drop or impute the point before building plans"
                    ),
                });
            }
        }
    }

    let grid = Grid::build(coords, d, &members, query);
    let mut cand: Vec<(f32, u64)> = Vec::new();
    for &center in &members {
        grid.neighbors_of(coords, d, center, query, &mut cand);
        let mut set: Vec<u64> = Vec::with_capacity(cand.len() + 1);
        if cfg.include_center {
            set.push(center);
        }
        set.extend(cand.iter().map(|&(_, r)| r));
        out.push_set(center, set, cfg);
    }
    Ok(out)
}

/// A uniform grid over the kept points, stored CSR-style: `starts` indexes
/// `members_by_cell`, so the bucketing is a counting sort with no per-cell
/// allocation.
struct Grid {
    /// Per-dimension bbox minimum.
    origin: Vec<f32>,
    /// Per-dimension cell count (>= 1).
    dims: Vec<usize>,
    cell: f32,
    starts: Vec<u32>,
    members_by_cell: Vec<u64>,
}

impl Grid {
    fn build(coords: &[f32], d: usize, members: &[u64], query: CoordQuery) -> Grid {
        let mut lo = vec![f32::INFINITY; d];
        let mut hi = vec![f32::NEG_INFINITY; d];
        for &r in members {
            for j in 0..d {
                let v = coords[r as usize * d + j];
                lo[j] = lo[j].min(v);
                hi[j] = hi[j].max(v);
            }
        }
        let extent: Vec<f32> = (0..d).map(|j| (hi[j] - lo[j]).max(0.0)).collect();
        let mut cell = match query {
            // One cell per radius: every point within `radius` is then in the
            // centre's cell or one of the 3^d - 1 around it.
            CoordQuery::Radius(r) => r,
            // Aim for ~k+1 points per cell, so the first ring usually settles
            // the answer. A degenerate extent falls through to the guard below.
            CoordQuery::Knn(k) => {
                let volume: f64 = extent
                    .iter()
                    .map(|&e| e.max(f32::MIN_POSITIVE) as f64)
                    .product();
                let per_cell = (k + 1) as f64;
                (volume * per_cell / members.len() as f64).powf(1.0 / d as f64) as f32
            }
        };
        if !(cell.is_finite() && cell > 0.0) {
            // Every point identical, or an unusable estimate: one cell, brute
            // force. Correct, and the benchmark arm asserts against it rather
            // than timing it by accident.
            cell = 1.0;
        }
        // Widen until the bucket array is bounded. A finer grid than this
        // cannot pay for itself.
        let max_cells = members.len().saturating_mul(MAX_CELLS_PER_POINT).max(1);
        let mut dims;
        loop {
            dims = (0..d)
                .map(|j| {
                    ((extent[j] / cell).floor() as usize)
                        .saturating_add(1)
                        .max(1)
                })
                .collect::<Vec<usize>>();
            let total = dims.iter().try_fold(1usize, |a, &b| a.checked_mul(b));
            match total {
                Some(t) if t <= max_cells => break,
                _ => cell *= 2.0,
            }
        }
        let total_cells: usize = dims.iter().product();

        let cell_of = |r: u64| -> usize {
            let mut idx = 0usize;
            for j in 0..d {
                let c = (((coords[r as usize * d + j] - lo[j]) / cell).floor() as usize)
                    .min(dims[j] - 1);
                idx = idx * dims[j] + c;
            }
            idx
        };
        let mut starts = vec![0u32; total_cells + 1];
        for &r in members {
            starts[cell_of(r) + 1] += 1;
        }
        for i in 0..total_cells {
            starts[i + 1] += starts[i];
        }
        let mut cursor = starts[..total_cells].to_vec();
        let mut members_by_cell = vec![0u64; members.len()];
        // `members` is ascending, so each cell's bucket comes out ascending
        // too — which is what makes the row tiebreak below cheap.
        for &r in members {
            let c = cell_of(r);
            members_by_cell[cursor[c] as usize] = r;
            cursor[c] += 1;
        }
        Grid {
            origin: lo,
            dims,
            cell,
            starts,
            members_by_cell,
        }
    }

    fn coord_cell(&self, coords: &[f32], d: usize, r: u64) -> Vec<usize> {
        (0..d)
            .map(|j| {
                (((coords[r as usize * d + j] - self.origin[j]) / self.cell).floor() as usize)
                    .min(self.dims[j] - 1)
            })
            .collect()
    }

    fn max_ring(&self) -> usize {
        self.dims
            .iter()
            .map(|&n| n.saturating_sub(1))
            .max()
            .unwrap_or(0)
    }

    /// Fill `out` with the query's neighbours of `center`, ordered
    /// (squared distance ascending, row ascending), excluding `center` itself.
    fn neighbors_of(
        &self,
        coords: &[f32],
        d: usize,
        center: u64,
        query: CoordQuery,
        out: &mut Vec<(f32, u64)>,
    ) {
        out.clear();
        let home = self.coord_cell(coords, d, center);
        let max_ring = self.max_ring();
        let (want, radius2) = match query {
            CoordQuery::Knn(k) => (Some(k), f32::INFINITY),
            CoordQuery::Radius(r) => (None, r * r),
        };
        let mut ring = 0usize;
        loop {
            self.scan_ring(coords, d, center, &home, ring, radius2, out);
            match want {
                // Radius mode: cell size == radius, so ring 1 is exhaustive.
                None => {
                    if ring >= 1 {
                        break;
                    }
                }
                Some(k) => {
                    // Everything there is, is already in hand — no ring can add
                    // to it. Without this, `k` larger than the population means
                    // `out.len() >= k` is never true and every centre scans the
                    // whole bounding box to `max_ring`.
                    if out.len() + 1 >= self.members_by_cell.len() {
                        break;
                    }
                    // Anything outside the rings scanned so far is at least
                    // `ring * cell` from the centre, wherever in its own cell
                    // the centre sits. Once we hold `k` candidates no further
                    // than that, no later ring can displace one.
                    if out.len() >= k {
                        // Sorted in place rather than into a copy: the final
                        // sort below wants this order anyway, and a per-ring
                        // clone of the candidate list is the one allocation
                        // this loop could not justify.
                        out.sort_by(cmp_candidate);
                        let bound = ring as f32 * self.cell;
                        if out[k - 1].0 <= bound * bound {
                            break;
                        }
                    }
                    if ring >= max_ring {
                        break;
                    }
                }
            }
            ring += 1;
        }
        out.sort_by(cmp_candidate);
        if let Some(k) = want {
            out.truncate(k);
        }
    }

    /// Scan every cell at Chebyshev distance exactly `ring` from `home`,
    /// appending in-range candidates.
    #[allow(clippy::too_many_arguments)]
    fn scan_ring(
        &self,
        coords: &[f32],
        d: usize,
        center: u64,
        home: &[usize],
        ring: usize,
        radius2: f32,
        out: &mut Vec<(f32, u64)>,
    ) {
        let mut offsets = vec![0isize; d];
        self.walk_ring(0, ring, home, d, &mut offsets, &mut |cell_idx| {
            let lo = self.starts[cell_idx] as usize;
            let hi = self.starts[cell_idx + 1] as usize;
            for &r in &self.members_by_cell[lo..hi] {
                if r == center {
                    continue;
                }
                let mut d2 = 0.0f32;
                for j in 0..d {
                    let diff = coords[r as usize * d + j] - coords[center as usize * d + j];
                    d2 += diff * diff;
                }
                if d2 <= radius2 {
                    out.push((d2, r));
                }
            }
        });
    }

    /// Enumerate the cells whose offset from `home` has Chebyshev norm exactly
    /// `ring`, calling `visit` with each one's flat index.
    fn walk_ring(
        &self,
        dim: usize,
        ring: usize,
        home: &[usize],
        d: usize,
        offsets: &mut Vec<isize>,
        visit: &mut impl FnMut(usize),
    ) {
        if dim == d {
            // A ring is the shell, not the ball: at least one axis must be at
            // the extreme, or this cell was already scanned at a smaller ring.
            if ring > 0 && !offsets.iter().any(|o| o.unsigned_abs() == ring) {
                return;
            }
            let mut idx = 0usize;
            for j in 0..d {
                let c = home[j] as isize + offsets[j];
                if c < 0 || c as usize >= self.dims[j] {
                    return;
                }
                idx = idx * self.dims[j] + c as usize;
            }
            visit(idx);
            return;
        }
        let r = ring as isize;
        for off in -r..=r {
            offsets[dim] = off;
            self.walk_ring(dim + 1, ring, home, d, offsets, visit);
        }
        offsets[dim] = 0;
    }
}

/// One set ended at `total`; record the boundary.
fn rows_len_checked(set_offsets: &mut Vec<i64>, total: usize) {
    set_offsets.push(total as i64);
}

/// (squared distance ascending, row ascending) — the declared tie order.
///
/// `total_cmp` rather than `partial_cmp(..).unwrap_or(Equal)`: the latter is not
/// a total order in the presence of a NaN (`NaN == 1.0` and `NaN == 2.0` while
/// `1.0 != 2.0`), and `slice::sort_by` is entitled to panic on a comparator that
/// is not one. Coordinates are refused if non-finite, so this cannot fire here
/// today — it is written this way so that it still cannot if that ever changes.
fn cmp_candidate(a: &(f32, u64), b: &(f32, u64)) -> std::cmp::Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// Order two `(weight, column)` graph edges: **non-finite weights last**, then
/// by weight in the requested direction, then column ascending.
///
/// Non-finite last, and not refused, because a distance graph legitimately
/// carries `inf` for "not connected" and a NaN is a property of someone else's
/// upstream computation. Sorting them to the end means `k` can never select one
/// over a finite competitor, which is the outcome a caller wants; refusing the
/// whole build over one edge is not.
///
/// A total order in every case, which `partial_cmp(..).unwrap_or(Equal)` is not
/// — and `slice::sort_by` may panic on a comparator that is not one.
fn cmp_weight(a: (f32, i64), b: (f32, i64), order: WeightOrder) -> std::cmp::Ordering {
    match (a.0.is_finite(), b.0.is_finite()) {
        (true, false) => return std::cmp::Ordering::Less,
        (false, true) => return std::cmp::Ordering::Greater,
        (false, false) => return a.1.cmp(&b.1),
        (true, true) => {}
    }
    let by_weight = match order {
        WeightOrder::Desc => b.0.total_cmp(&a.0),
        WeightOrder::Asc => a.0.total_cmp(&b.0),
    };
    by_weight.then(a.1.cmp(&b.1))
}

// ---------------------------------------------------------------------------
// Batching
// ---------------------------------------------------------------------------

/// Concatenate **single-set** plans into batch plans of `sets_per_batch` sets.
///
/// Every input must be one set — `set_offsets == [0, rows.len()]`, which is
/// what both builders emit — so `sets_per_batch` counts what its name says.
/// An earlier version accepted multi-set inputs and re-based each set, which
/// meant the parameter silently chunked *plans* rather than sets: re-batching
/// an already-batched list at `sets_per_batch = 2` produced outputs of six
/// sets each. Nothing needed that, and a knob that means two different things
/// depending on its input is worse than one that refuses.
///
/// `shuffle_seed` shuffles the *set* order (never a set's members) with
/// `ChaCha8Rng`, so an epoch is reproducible across runs and machines. The
/// last batch is short rather than dropped.
pub fn batch_plans(
    plans: &[SparseCellSetPlan],
    sets_per_batch: usize,
    shuffle_seed: Option<u64>,
) -> Result<Vec<SparseCellSetPlan>> {
    if sets_per_batch == 0 {
        return Err(LoaderError::ConfigError {
            reason: "batch_plans: sets_per_batch must be >= 1".into(),
        });
    }
    // One set per input, so `sets_per_batch` counts sets. Also what makes the
    // rebase below correct: it appends exactly one offset per input.
    for (i, p) in plans.iter().enumerate() {
        let off = &p.set_offsets;
        if off.as_slice() != [0, p.rows.len() as i64] {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "batch_plans: plan {i} has set_offsets {:?} over {} rows, so it is not a \
                     single set. Every input must be one set ([0, rows.len()]) — which is what \
                     the neighbourhood builders emit — so that sets_per_batch counts sets. To \
                     re-batch an already-batched list, rebuild the single-set plans instead.",
                    off,
                    p.rows.len()
                ),
            });
        }
        if p.file_ids.len() != p.rows.len() || p.role_tags.len() != p.rows.len() {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "batch_plans: plan {i} has {} rows but {} file_ids and {} role_tags",
                    p.rows.len(),
                    p.file_ids.len(),
                    p.role_tags.len()
                ),
            });
        }
    }
    let mut order: Vec<usize> = (0..plans.len()).collect();
    if let Some(seed) = shuffle_seed {
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    }
    let mut out = Vec::with_capacity(plans.len().div_ceil(sets_per_batch));
    for chunk in order.chunks(sets_per_batch) {
        let mut file_ids = Vec::new();
        let mut rows = Vec::new();
        let mut role_tags = Vec::new();
        let mut set_offsets = vec![0i64];
        for &i in chunk {
            let p = &plans[i];
            // Each input is expected to be a single set; a multi-set input is
            // re-based set by set rather than collapsed into one, so batching
            // an already-batched list is idempotent in shape.
            file_ids.extend_from_slice(&p.file_ids);
            rows.extend_from_slice(&p.rows);
            role_tags.extend_from_slice(&p.role_tags);
            rows_len_checked(&mut set_offsets, rows.len());
        }
        out.push(SparseCellSetPlan {
            file_ids,
            rows,
            role_tags,
            set_offsets,
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "neighborhood_tests.rs"]
mod tests;

// ---------------------------------------------------------------------------
// Drivers — open a file, stream it, build
// ---------------------------------------------------------------------------

/// Rows of the graph read per `read_rows_range` call.
///
/// The chunk bounds the CSR view held at once; the *plans* built from it are
/// retained for the whole build, so this caps the transient, not the result.
pub const DEFAULT_GRAPH_CHUNK_ROWS: u64 = 65_536;

/// Build graph-driven plans from `obsp/<key>` of the file at `path`.
///
/// Streams the graph one row range at a time — peak memory is one obsp shard
/// plus one chunk's non-zeros, plus the accumulated plans. Note the caveat in
/// [`scx_format_io::BackedPairwiseReader`]: a *legacy unsharded* obsp (what
/// `scx sort` emits) has to be decoded whole whatever the chunk size.
#[allow(clippy::too_many_arguments)]
pub fn build_graph_plans(
    path: &std::path::Path,
    key: &str,
    drop_deleted: bool,
    k: Option<usize>,
    weight_order: WeightOrder,
    cfg: NeighborhoodConfig,
    chunk_rows: u64,
) -> Result<NeighborhoodPlans> {
    if chunk_rows == 0 {
        return Err(LoaderError::ConfigError {
            reason: "build_graph_plans: chunk_rows must be >= 1".into(),
        });
    }
    // ONE open, and the keep mask comes off the same reader as the graph.
    // Taking a caller-supplied mask (or reading it through a second
    // `ScxReader::open`) means two snapshots of one path, which can disagree if
    // the file is replaced between them — and the disagreement is a plan naming
    // rows that no longer exist, with nothing to catch it.
    let scx = ScxReader::open(path)?;
    let keep: Option<Vec<bool>> = if drop_deleted {
        scx.deletion_keep_mask()?
    } else {
        None
    };
    let reader = BackedPairwiseReader::new_obsp(scx, key)?;
    let n_rows = reader.n_rows();
    if let Some(mask) = keep.as_deref() {
        if mask.len() as u64 != n_rows {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "build_graph_plans: the file's deletion keep mask has {} entries but \
                     obsp/{key} has {n_rows} rows",
                    mask.len()
                ),
            });
        }
    }
    let mut out = NeighborhoodPlans::default();
    let mut start = 0u64;
    while start < n_rows {
        let end = (start + chunk_rows).min(n_rows);
        let rows = reader.read_rows_range(start, end)?;
        plans_from_graph_chunk(&rows, keep.as_deref(), k, weight_order, cfg, &mut out);
        start = end;
    }
    Ok(out)
}

/// Read `obsm/<key>` as a flat row-major `n x d` `f32` array.
///
/// Accepts the dtypes a coordinate matrix actually arrives in: scanpy writes
/// Visium's `obsm["spatial"]` as **int64** pixel coordinates, and an h5ad
/// round-trip can leave float64, so narrowing to the loader's f32 contract is
/// done here rather than refused.
pub fn read_coords(path: &std::path::Path, key: &str) -> Result<(Vec<f32>, usize)> {
    read_coords_from(ScxReader::open(path)?, key)
}

/// [`read_coords`] over a reader the caller already holds.
pub fn read_coords_from(scx: ScxReader, key: &str) -> Result<(Vec<f32>, usize)> {
    let reader = BackedDenseReader::new_obsm(scx, key, 1)?;
    let (n_rows, d) = reader.shape();
    // Checked HERE, from the layout, before anything is decoded. Left to
    // `plans_from_coords` the refusal still happened, but only after the whole
    // n x d array had been read and allocated — which on an atlas-scale file
    // and a mistyped `obsm_key="X_pca"` is the cost the refusal exists to
    // avoid.
    if d == 0 || d > MAX_COORD_DIMS {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "obsm/{key} has {d} columns; coordinates must be 1..={MAX_COORD_DIMS}-D. The \
                 grid search is exponential in the dimensionality. To build neighbourhoods in a \
                 wide embedding, write a kNN graph into obsp (scanpy: \
                 sc.pp.neighbors(use_rep=...)) and use the graph builder instead."
            ),
        });
    }
    let batch = reader.read_rows_range(0, n_rows as u64)?;
    let mut out = vec![0.0f32; n_rows * d];
    for (j, col) in batch.columns().iter().enumerate() {
        let vals = coord_column_f32(col.as_ref(), key, j)?;
        if vals.len() != n_rows {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "obsm/{key} column {j} has {} values but the mapping has {n_rows} rows",
                    vals.len()
                ),
            });
        }
        for (i, v) in vals.into_iter().enumerate() {
            out[i * d + j] = v;
        }
    }
    Ok((out, d))
}

fn coord_column_f32(col: &dyn arrow::array::Array, key: &str, j: usize) -> Result<Vec<f32>> {
    use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array, UInt32Array};
    // Before the dtype dispatch: `values()` hands back the raw buffer, so a
    // null slot reads as whatever is in it. The COO decoder got this check in
    // round 1 and the coordinate path did not, which is worse here than there
    // — a bogus-but-finite coordinate sails past the `is_finite` guard and
    // lands the cell in a neighbourhood it is not in, while a non-finite bit
    // pattern raises "NaN coordinate" and names the wrong defect.
    if col.null_count() > 0 {
        return Err(LoaderError::ConfigError {
            reason: format!(
                "obsm/{key} column {j} has {} null entries; a coordinate has no meaning when \
                 it is missing, so the row must be dropped or imputed before building plans",
                col.null_count()
            ),
        });
    }
    macro_rules! try_as {
        ($ty:ty) => {
            if let Some(a) = col.as_any().downcast_ref::<$ty>() {
                return Ok(a.values().iter().map(|&v| v as f32).collect());
            }
        };
    }
    try_as!(Float32Array);
    try_as!(Float64Array);
    try_as!(Int64Array);
    try_as!(Int32Array);
    try_as!(UInt32Array);
    Err(LoaderError::ConfigError {
        reason: format!(
            "obsm/{key} column {j} has dtype {:?}, which is not a coordinate type",
            col.data_type()
        ),
    })
}

/// Build coordinate-driven plans from `obsm/<key>` of the file at `path`.
pub fn build_coord_plans(
    path: &std::path::Path,
    key: &str,
    drop_deleted: bool,
    query: CoordQuery,
    cfg: NeighborhoodConfig,
) -> Result<NeighborhoodPlans> {
    // One open, for the same reason `build_graph_plans` takes one.
    let scx = ScxReader::open(path)?;
    let keep: Option<Vec<bool>> = if drop_deleted {
        scx.deletion_keep_mask()?
    } else {
        None
    };
    let (coords, d) = read_coords_from(scx, key)?;
    plans_from_coords(&coords, d, keep.as_deref(), query, cfg)
}
