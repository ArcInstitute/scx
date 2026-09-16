//! Neighbourhood plan builders — W7 steps 1 and 2.
//!
//! A neighbourhood *is* a cell set with the centre role-tagged, so the gather
//! primitive R6 needs already exists. What was missing is the plan. These two
//! builders turn a stored `obsp` graph, or `obsm["spatial"]` coordinates, into
//! the same `(file_ids, rows, role_tags, set_offsets)` plans
//! `SparseCellSetDataset::gather` / `iter_with_plans` already consume — which
//! makes R6 an R2 workload and hands it every R2 improvement for free.
//!
//! # Roles
//!
//! Position 0 of a set is the **centre**, `role_tag = 0`; every other member is
//! a neighbour, `role_tag = 1`. That maps onto Nicheformer's cell + context
//! tokens and onto the per-set kernels in [`crate::tokenize`].
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
//! - a **deleted neighbour is dropped** from every set it appears in, leaving a
//!   shorter set rather than a backfilled one.
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
/// column-ascending order; with `k = Some(k)` the `k` heaviest edges are taken,
/// **weight descending, ties broken by column ascending** — a declared rule,
/// since a stored graph carries no order of its own.
///
/// A self-loop (an edge from a row to itself) is dropped when the centre is
/// already being emitted, rather than letting the centre appear twice and be
/// silently double-weighted by whatever consumes the set.
pub fn plans_from_graph_chunk(
    rows: &PairwiseRows,
    keep: Option<&[bool]>,
    k: Option<usize>,
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
            // Weight descending, then column ascending. `sort_by` is stable, so
            // sorting on the key directly rather than reversing keeps the tie
            // order explicit instead of implied by the sort's stability.
            scratch.sort_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.1.cmp(&b.1))
            });
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
    if d == 0 {
        return Err(LoaderError::ConfigError {
            reason: "plans_from_coords: coordinate dimensionality must be >= 1".into(),
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

/// (squared distance ascending, row ascending) — the declared tie order.
fn cmp_candidate(a: &(f32, u64), b: &(f32, u64)) -> std::cmp::Ordering {
    a.0.partial_cmp(&b.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.1.cmp(&b.1))
}

// ---------------------------------------------------------------------------
// Batching
// ---------------------------------------------------------------------------

/// Concatenate single-set plans into batch plans of `sets_per_batch` sets.
///
/// Overlapping neighbourhoods — a cell that is a neighbour of many centres —
/// are the reuse signal a later admission policy can act on, and batching is
/// what puts overlapping sets in the same plan where the shard cache can see
/// them. `shuffle_seed` shuffles the *set* order (never a set's members) with
/// `ChaCha8Rng`, so an epoch is reproducible across runs and machines.
///
/// The last batch is short rather than dropped.
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
    // Every input's offsets must be a true indptr, because the rebase below adds
    // a running base to them. A plan starting at a non-zero offset would be
    // rebased to the wrong place and produce a batch whose sets are silently
    // shifted — the builders here always emit `[0, n]`, but this is public and
    // takes whatever a caller has.
    for (i, p) in plans.iter().enumerate() {
        let off = &p.set_offsets;
        if off.first() != Some(&0) || off.last() != Some(&(p.rows.len() as i64)) {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "batch_plans: plan {i} has set_offsets {:?} over {} rows; each plan's \
                     set_offsets must start at 0 and end at its row count",
                    off,
                    p.rows.len()
                ),
            });
        }
        if off.windows(2).any(|w| w[1] < w[0]) {
            return Err(LoaderError::ConfigError {
                reason: format!("batch_plans: plan {i} has non-monotonic set_offsets {off:?}"),
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
            let base = rows.len() as i64;
            file_ids.extend_from_slice(&p.file_ids);
            rows.extend_from_slice(&p.rows);
            role_tags.extend_from_slice(&p.role_tags);
            for &off in p.set_offsets.iter().skip(1) {
                set_offsets.push(base + off);
            }
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
pub fn build_graph_plans(
    path: &std::path::Path,
    key: &str,
    keep: Option<&[bool]>,
    k: Option<usize>,
    cfg: NeighborhoodConfig,
    chunk_rows: u64,
) -> Result<NeighborhoodPlans> {
    if chunk_rows == 0 {
        return Err(LoaderError::ConfigError {
            reason: "build_graph_plans: chunk_rows must be >= 1".into(),
        });
    }
    let reader = BackedPairwiseReader::new_obsp(ScxReader::open(path)?, key)?;
    let n_rows = reader.n_rows();
    if let Some(mask) = keep {
        if mask.len() as u64 != n_rows {
            return Err(LoaderError::ConfigError {
                reason: format!(
                    "build_graph_plans: keep mask has {} entries but obsp/{key} has {n_rows} rows",
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
        plans_from_graph_chunk(&rows, keep, k, cfg, &mut out);
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
    let reader = BackedDenseReader::new_obsm(ScxReader::open(path)?, key, 1)?;
    let (n_rows, d) = reader.shape();
    if d == 0 {
        return Err(LoaderError::ConfigError {
            reason: format!("obsm/{key} has no columns, so it carries no coordinates"),
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
    keep: Option<&[bool]>,
    query: CoordQuery,
    cfg: NeighborhoodConfig,
) -> Result<NeighborhoodPlans> {
    let (coords, d) = read_coords(path, key)?;
    plans_from_coords(&coords, d, keep, query, cfg)
}
