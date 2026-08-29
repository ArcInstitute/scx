//! Streaming Highly Variable Genes (HVG) kernels.
//!
//! Provides shard-by-shard streaming computation of per-gene mean/variance
//! and clipped-square-sum statistics for the seurat_v3 HVG algorithm.
//! These operate through the [`ShardSource`] trait, so they work on both
//! raw backed data and lazy-transformed data without materialization.

use crate::error::Result;
use scx_format_io::ShardSource;

/// Per-gene mean and variance statistics.
#[derive(Debug, Clone)]
pub struct HvgStats {
    /// Per-gene mean expression (length = n_vars).
    pub means: Vec<f64>,
    /// Per-gene variance with Bessel's correction (length = n_vars).
    pub variances: Vec<f64>,
}

/// Reject non-finite values at the HVG accelerator boundary.
///
/// Thin wrapper over the shared [`crate::finite::ensure_finite_values`]
/// primitive so every storage route (CSR here, CSC in [`crate::csc`]) enforces
/// the same guard. (The GPU HVG path does not yet enforce this on-device;
/// tracked as a follow-on.)
fn ensure_finite_hvg_data(data: &[f32]) -> Result<()> {
    crate::finite::ensure_finite_values(data, "HVG")
}

/// Streaming per-column `(Σ t(x), Σ t(x)²)` moments over all shards, where
/// `t` is a per-value transform (identity for raw moments, `expm1(scale·x)`
/// for the seurat count-space moments).
///
/// Uses the 2.1 decode-prefetch primitives ([`crate::prefetch::accumulate_shards`]):
/// order-stable + bit-exact by default, budgeted-parallel under
/// `SCX_ACCEL_REDUCTION_MODE=parallel`. The accumulator is two `n_vars`-length
/// f64 vectors, independent of the global row offset, so shard order does not
/// affect correctness (only float summation order in the parallel mode). The
/// finiteness guard is enforced per shard, matching the pre-2.1 loop.
fn accumulate_col_moments<S, T>(
    source: &S,
    n_vars: usize,
    transform: T,
) -> Result<(Vec<f64>, Vec<f64>)>
where
    S: ShardSource + Sync,
    T: Fn(f64) -> f64 + Sync,
{
    // Bound concurrent per-worker accumulators (each `2 · n_vars · f64`) to the
    // shared CPU budget; the ordered default holds exactly one accumulator.
    let per_acc = 2u64.saturating_mul(n_vars as u64).saturating_mul(8);
    let workers = crate::mem_budget::clamp_prefetch_depth(
        rayon::current_num_threads(),
        per_acc,
        crate::mem_budget::de_memory_budget(),
    );
    crate::prefetch::accumulate_shards(
        source,
        workers,
        || (vec![0.0f64; n_vars], vec![0.0f64; n_vars]),
        |acc: &mut (Vec<f64>, Vec<f64>), _idx, csr| {
            let _r = scx_format_io::reduction_guard();
            ensure_finite_hvg_data(&csr.data)?;
            let (col_sum, col_sum_sq) = acc;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let c = col as usize;
                let v = transform(val as f64);
                col_sum[c] += v;
                col_sum_sq[c] += v * v;
            }
            Ok(())
        },
        |mut a: (Vec<f64>, Vec<f64>), b: (Vec<f64>, Vec<f64>)| {
            for (x, y) in a.0.iter_mut().zip(b.0.iter()) {
                *x += *y;
            }
            for (x, y) in a.1.iter_mut().zip(b.1.iter()) {
                *x += *y;
            }
            a
        },
    )
}

/// Single-pass streaming mean and variance per column.
///
/// Accumulates per-column sum and sum-of-squares in f64, then computes:
///   mean = sum / n
///   var  = (sum_sq - n * mean²) / (n - 1)   (Bessel's correction, ddof=1)
///
/// This matches scanpy's `correction=1` parameter in `mean_var()`.
/// Memory: O(n_vars) for two accumulator vectors.
///
/// # Numerical stability
///
/// The two-pass sum-of-squares formula (`sum_sq - n * mean^2`) can suffer from
/// catastrophic cancellation when values are large relative to the variance.
/// Accumulating f32 sparse values into f64 provides sufficient headroom for
/// typical scRNA-seq data (counts 0-100, up to ~10M cells). Negative variances
/// from numerical noise are clamped to zero.
///
/// If this is ever needed for data with much larger magnitudes or tighter
/// variance, Welford's online algorithm would provide better numerical
/// stability at the cost of a branch per nonzero element.
///
/// Returns zero means and zero variances when `n_obs == 0`.
pub fn streaming_mean_var<S: ShardSource + Sync>(source: &S) -> Result<HvgStats> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();

    // Early return for empty source: avoid division by zero.
    if n_obs == 0 {
        return Ok(HvgStats {
            means: vec![0.0; n_vars],
            variances: vec![0.0; n_vars],
        });
    }

    // Decode-prefetched, order-stable per-column reduction (2.1). Accumulator
    // is `(col_sum, col_sum_sq)`, independent of the global row offset, so both
    // the ordered (bit-exact default) and budgeted-parallel modes are valid.
    let (col_sum, col_sum_sq) = accumulate_col_moments(source, n_vars, |v| v)?;

    let m = scx_sparse::finalize_column_moments(&col_sum, &col_sum_sq, n_obs);
    m.warn_if_unstable("streaming_mean_var");
    Ok(HvgStats {
        means: m.means,
        variances: m.variances,
    })
}

/// Single-pass streaming mean/variance per column on `expm1(scale · value)`.
///
/// scanpy's `seurat` HVG flavor un-`log1p`s the matrix before computing
/// moments: `x *= ln(base)` (identity when the stored base is natural log /
/// `None`, i.e. `scale = 1.0`), then `expm1`. Because `expm1(0) == 0`, implicit
/// and stored zeros contribute nothing, so the count-space moments stream from
/// the sparse nonzeros exactly like [`streaming_mean_var`].
///
/// `scale` is `ln(base)` for a log1p base of `base`, or `1.0` for natural-log /
/// no recorded base. Bessel's correction (ddof=1) matches scanpy's
/// `correction=1`.
pub fn streaming_mean_var_expm1<S: ShardSource + Sync>(source: &S, scale: f64) -> Result<HvgStats> {
    let n_vars = source.n_vars();
    let n_obs = source.n_obs();

    if n_obs == 0 {
        return Ok(HvgStats {
            means: vec![0.0; n_vars],
            variances: vec![0.0; n_vars],
        });
    }

    // Count-space moments: `expm1(scale · v)` un-logs before accumulating.
    // `expm1(0) == 0`, so implicit/stored zeros contribute nothing (2.1
    // decode-prefetched, order-stable by default).
    let (col_sum, col_sum_sq) = accumulate_col_moments(source, n_vars, |v| (v * scale).exp_m1())?;

    let m = scx_sparse::finalize_column_moments(&col_sum, &col_sum_sq, n_obs);
    m.warn_if_unstable("streaming_mean_var_expm1");
    Ok(HvgStats {
        means: m.means,
        variances: m.variances,
    })
}

/// Z-score-normalize log-dispersions within equal-width mean-expression bins —
/// the binning half of the scanpy `seurat` HVG flavor (its moments half is
/// [`streaming_mean_var_expm1`]). A faithful port of the pandas reference
/// (scanpy `_get_disp_stats` / `_postprocess_dispersions_seurat`, previously
/// shipped as pyscx's `_hvg_helpers.py`):
///
/// 1. **Binning** = `pd.cut(log_means, bins=n_bins)`: equal-width edges over
///    `[nanmin, nanmax]`, computed exactly as `np.linspace` does
///    (`i·step + mn`, last edge set to `mx`), then the leftmost edge widened
///    by `0.001·(mx − mn)`; when `mn == mx` both ends are widened first by
///    `0.001·|v|` (or `0.001` when `v == 0`) and no post-adjustment happens.
///    Intervals are **right-closed**, so a gene exactly on an interior edge
///    lands in the bin to its left (searchsorted `side="left"`).
/// 2. Per-bin **NaN-skipping mean** (`avg`) and **ddof=1 std** (`dev`) of the
///    dispersions.
/// 3. **Singleton-bin rule**, a scanpy quirk pinned by a past P0 fix
///    (commit 87f1d937): a bin whose `dev` is NaN gets `dev = avg; avg = 0`,
///    so its gene's normalized dispersion is EXACTLY `1.0` — not `0`, which
///    the tempting `dev = 1` would give.
/// 4. Per-gene `(dispersion − avg) / dev`, with **NaN preserved as NaN** (the
///    caller floors NaN to `−inf` for selection, matching scanpy's
///    `nan_to_num(nan=-inf)`).
///
/// Degenerate inputs return all-NaN rather than erroring: an empty input
/// yields an empty vec, and `n_bins == 0` or all-NaN `log_means` yield NaN
/// per gene (the pandas reference raises on those; callers validate
/// `n_bins ≥ 1` at their own boundary, and the accel finiteness guard
/// upstream keeps means finite).
pub fn binned_dispersion_norm(
    log_means: &[f64],
    log_dispersions: &[f64],
    n_bins: usize,
) -> Vec<f64> {
    let n = log_means.len();
    debug_assert_eq!(n, log_dispersions.len());
    // `checked_add` closes the one arithmetic wrap (`usize::MAX + 1`); the
    // per-bin vectors below still allocate O(n_bins), so callers bound n_bins
    // to something meaningful — pyscx rejects > 2^20 at its boundary. An
    // unbounded direct caller risks the ordinary infallible-alloc abort any
    // `vec![0; huge]` carries, which is not a contract this kernel can lift.
    let Some(n_edges) = n_bins.checked_add(1) else {
        return vec![f64::NAN; n];
    };
    if n == 0 || n_bins == 0 {
        return vec![f64::NAN; n];
    }

    // --- 1. pd.cut bin edges ------------------------------------------------
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for &v in log_means {
        if v.is_nan() {
            continue;
        }
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if !mn.is_finite() || !mx.is_finite() {
        return vec![f64::NAN; n];
    }
    let mut edges = Vec::with_capacity(n_edges);
    if mn == mx {
        // pandas widens a zero-width range on both ends *before* binning, and
        // then skips the leftmost-edge adjustment below.
        let widen = |v: f64| {
            if v == 0.0 {
                0.001
            } else {
                0.001 * v.abs()
            }
        };
        let (lo, hi) = (mn - widen(mn), mx + widen(mx));
        let step = (hi - lo) / n_bins as f64;
        for i in 0..n_bins {
            edges.push(i as f64 * step + lo);
        }
        edges.push(hi);
    } else {
        let step = (mx - mn) / n_bins as f64;
        for i in 0..n_bins {
            edges.push(i as f64 * step + mn);
        }
        edges.push(mx);
        // Right-closed intervals leave `mn` itself outside `(e0, e1]`; pandas
        // pulls the leftmost edge down so the minimum is included.
        edges[0] -= 0.001 * (mx - mn);
    }

    // searchsorted(edges, x, side="left"): the first edge ≥ x; bin = idx − 1.
    // idx == 0 (x at/below the widened leftmost edge) and idx past the last
    // edge are both "no bin" — unreachable for finite x in [mn, mx], kept for
    // exact pandas parity.
    let bin_of = |x: f64| -> Option<usize> {
        if x.is_nan() {
            return None;
        }
        let idx = edges.partition_point(|e| *e < x);
        if idx == 0 || idx == edges.len() {
            return None;
        }
        Some(idx - 1)
    };
    let bins: Vec<Option<usize>> = log_means.iter().map(|&m| bin_of(m)).collect();

    // --- 2. per-bin NaN-skipping mean + ddof=1 std (two-pass) ---------------
    let mut cnt = vec![0usize; n_bins];
    let mut sum = vec![0f64; n_bins];
    for (b, &d) in bins.iter().zip(log_dispersions) {
        if let Some(b) = *b {
            if !d.is_nan() {
                cnt[b] += 1;
                sum[b] += d;
            }
        }
    }
    let mut avg: Vec<f64> = (0..n_bins)
        .map(|b| {
            if cnt[b] > 0 {
                sum[b] / cnt[b] as f64
            } else {
                f64::NAN
            }
        })
        .collect();
    let mut sq = vec![0f64; n_bins];
    for (b, &d) in bins.iter().zip(log_dispersions) {
        if let Some(b) = *b {
            if !d.is_nan() {
                let r = d - avg[b];
                sq[b] += r * r;
            }
        }
    }
    let mut dev: Vec<f64> = (0..n_bins)
        .map(|b| {
            if cnt[b] > 1 {
                (sq[b] / (cnt[b] - 1) as f64).sqrt()
            } else {
                f64::NAN
            }
        })
        .collect();

    // --- 3. singleton-bin rule ----------------------------------------------
    // A bin with zero non-NaN members also lands here (avg is NaN, so the
    // division below still yields NaN for its genes — same as pandas).
    for b in 0..n_bins {
        if dev[b].is_nan() {
            dev[b] = avg[b];
            avg[b] = 0.0;
        }
    }

    // --- 4. map back per gene, NaN preserved ---------------------------------
    bins.iter()
        .zip(log_dispersions)
        .map(|(b, &d)| match b {
            Some(b) => (d - avg[*b]) / dev[*b],
            None => f64::NAN,
        })
        .collect()
}

/// Single-pass streaming clipped accumulation for seurat_v3 normalized variance.
///
/// For each nonzero value in the matrix, clips it to `min(val, clip_val[col])`,
/// then accumulates both the clipped sum and the clipped-squared sum per column.
///
/// Returns `(batch_counts_sum, squared_batch_counts_sum)` — both `Vec<f64>` of
/// length `n_vars`.
///
/// Rejects non-finite input (NaN/Inf) at the accelerator boundary via
/// [`ensure_finite_hvg_data`], mirroring [`streaming_mean_var`]: a NaN clips to
/// `clip_val[c]` (Rust's `f64::min` returns the non-NaN operand) and would
/// silently poison the clipped sums otherwise.
///
/// Memory: O(n_vars).
pub fn streaming_clip_square_sum<S: ShardSource + Sync>(
    source: &S,
    clip_val: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    let n_vars = source.n_vars();
    debug_assert_eq!(clip_val.len(), n_vars);

    // Per-column clipped `(Σ v, Σ v²)`, offset-independent → 2.1 prefetched,
    // order-stable by default.
    let per_acc = 2u64.saturating_mul(n_vars as u64).saturating_mul(8);
    let workers = crate::mem_budget::clamp_prefetch_depth(
        rayon::current_num_threads(),
        per_acc,
        crate::mem_budget::de_memory_budget(),
    );
    crate::prefetch::accumulate_shards(
        source,
        workers,
        || (vec![0.0f64; n_vars], vec![0.0f64; n_vars]),
        |acc: &mut (Vec<f64>, Vec<f64>), _idx, csr| {
            let _r = scx_format_io::reduction_guard();
            ensure_finite_hvg_data(&csr.data)?;
            let (bcs, sbcs) = acc;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let c = col as usize;
                let v = (val as f64).min(clip_val[c]);
                bcs[c] += v;
                sbcs[c] += v * v;
            }
            Ok(())
        },
        |mut a: (Vec<f64>, Vec<f64>), b: (Vec<f64>, Vec<f64>)| {
            for (x, y) in a.0.iter_mut().zip(b.0.iter()) {
                *x += *y;
            }
            for (x, y) in a.1.iter_mut().zip(b.1.iter()) {
                *x += *y;
            }
            a
        },
    )
}

/// Per-batch and global mean/variance from a single streaming pass.
#[derive(Debug, Clone)]
pub struct BatchedHvgStats {
    /// Per-batch mean and variance (length = `n_batches`).
    pub per_batch: Vec<HvgStats>,
    /// Global mean and variance (aggregated from all batches).
    pub global: HvgStats,
    /// Number of cells in each batch.
    pub batch_counts: Vec<usize>,
}

/// Single-pass streaming mean and variance per column **for multiple batches**.
///
/// Iterates through all shards once, accumulating per-batch sum and sum-of-squares
/// in f64. Also derives global statistics from the per-batch accumulators (no
/// extra pass needed). This reduces multi-batch HVG from 1 + 2N passes to 2 total.
///
/// `cell_batch` maps each visible cell (in shard-iteration order) to a batch index.
/// Use `-1` for cells that should be excluded from all batches.
///
/// Memory: O(n_vars * n_batches) for per-batch accumulators.
pub fn streaming_mean_var_batched<S: ShardSource + Sync>(
    source: &S,
    cell_batch: &[i32],
    n_batches: usize,
) -> Result<BatchedHvgStats> {
    let n_vars = source.n_vars();

    let mut batch_sum = vec![vec![0.0f64; n_vars]; n_batches];
    let mut batch_sum_sq = vec![vec![0.0f64; n_vars]; n_batches];
    let mut batch_count = vec![0usize; n_batches];

    // Ordered decode-prefetch (2.1): the cell→batch mapping is keyed by the
    // global cell index, so shards must be consumed in order — StableOrder is
    // the only valid mode here (`for_each_shard_ordered`, never the reordered
    // parallel reduction). The `cell_offset` cursor advances per delivered shard.
    let mut cell_offset = 0usize;
    crate::prefetch::for_each_shard_ordered(
        source,
        crate::prefetch::prefetch_depth(),
        |_idx, csr| {
            let _r = scx_format_io::reduction_guard();
            ensure_finite_hvg_data(&csr.data)?;
            let n_rows = csr.n_rows();

            for row in 0..n_rows {
                let cell_idx = cell_offset + row;
                let b = cell_batch[cell_idx];
                if b < 0 {
                    continue;
                }
                let b = b as usize;
                batch_count[b] += 1;

                let start = csr.indptr[row] as usize;
                let end = csr.indptr[row + 1] as usize;
                for j in start..end {
                    let c = csr.indices[j] as usize;
                    let v = csr.data[j] as f64;
                    batch_sum[b][c] += v;
                    batch_sum_sq[b][c] += v * v;
                }
            }
            cell_offset += n_rows;
            Ok(())
        },
    )?;

    // Compute per-batch means and variances.
    let mut per_batch = Vec::with_capacity(n_batches);
    for b in 0..n_batches {
        // `finalize_column_moments` returns zeros for `n == 0`, so the
        // empty-batch case needs no separate arm here.
        let m =
            scx_sparse::finalize_column_moments(&batch_sum[b], &batch_sum_sq[b], batch_count[b]);
        m.warn_if_unstable(&format!("streaming_mean_var_batched[batch {b}]"));
        per_batch.push(HvgStats {
            means: m.means,
            variances: m.variances,
        });
    }

    // Derive global stats from per-batch accumulators (no second pass over the
    // data). This is exact, not an approximation: raw moments are additive, so
    // the global Σx and Σx² are simply the sums of the per-batch Σx / Σx², and
    // the global mean/variance computed from them equal the pooled (single-pass
    // over all cells) result exactly. It does, however, inherit the same
    // near-constant-gene catastrophic-cancellation sensitivity as the
    // `Σx² − n·mean²` variance path (see `streaming_mean_var_with_device`
    // accuracy caveat); the per-batch partial sums do not worsen it.
    let total_n: usize = batch_count.iter().sum();
    let mut global_sum = vec![0.0f64; n_vars];
    let mut global_sum_sq = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        global_sum[j] = batch_sum.iter().map(|bs| bs[j]).sum();
        global_sum_sq[j] = batch_sum_sq.iter().map(|bs| bs[j]).sum();
    }
    let global = scx_sparse::finalize_column_moments(&global_sum, &global_sum_sq, total_n);
    global.warn_if_unstable("streaming_mean_var_batched[global]");

    Ok(BatchedHvgStats {
        per_batch,
        global: HvgStats {
            means: global.means,
            variances: global.variances,
        },
        batch_counts: batch_count,
    })
}

/// Single-pass streaming clipped accumulation for **multiple batches**.
///
/// For each nonzero value, looks up the cell's batch, clips by that batch's
/// `clip_val`, and accumulates per-batch clipped sums and squared sums.
///
/// `clip_vals[batch][gene]` is the clip threshold for each batch/gene pair.
///
/// Rejects non-finite input (NaN/Inf) at the accelerator boundary via
/// [`ensure_finite_hvg_data`], mirroring [`streaming_mean_var_batched`].
///
/// Memory: O(n_vars * n_batches).
pub fn streaming_clip_square_sum_batched<S: ShardSource + Sync>(
    source: &S,
    cell_batch: &[i32],
    n_batches: usize,
    clip_vals: &[Vec<f64>],
) -> Result<Vec<(Vec<f64>, Vec<f64>)>> {
    let n_vars = source.n_vars();
    debug_assert_eq!(clip_vals.len(), n_batches);

    let mut batch_bcs = vec![vec![0.0f64; n_vars]; n_batches];
    let mut batch_sbcs = vec![vec![0.0f64; n_vars]; n_batches];

    // Ordered decode-prefetch (2.1): batch mapping is global-cell-index keyed,
    // so the `cell_offset` cursor requires in-order shard delivery.
    let mut cell_offset = 0usize;
    crate::prefetch::for_each_shard_ordered(
        source,
        crate::prefetch::prefetch_depth(),
        |_idx, csr| {
            let _r = scx_format_io::reduction_guard();
            ensure_finite_hvg_data(&csr.data)?;
            let n_rows = csr.n_rows();

            for row in 0..n_rows {
                let cell_idx = cell_offset + row;
                let b = cell_batch[cell_idx];
                if b < 0 {
                    continue;
                }
                let b = b as usize;

                let start = csr.indptr[row] as usize;
                let end = csr.indptr[row + 1] as usize;
                for j in start..end {
                    let c = csr.indices[j] as usize;
                    let v = (csr.data[j] as f64).min(clip_vals[b][c]);
                    batch_bcs[b][c] += v;
                    batch_sbcs[b][c] += v * v;
                }
            }
            cell_offset += n_rows;
            Ok(())
        },
    )?;

    Ok(batch_bcs.into_iter().zip(batch_sbcs).collect())
}

/// A simple in-memory `ShardSource` for testing.
///
/// Lives at module scope rather than inside `mod tests` so the sibling
/// `moments_golden` module can reuse it — the crate already carries five
/// in-memory `ShardSource` doubles (`gene_score_tests`, `pflog_tests`,
/// `fused::gpu`, and two in `pca::cpu`) and a sixth is not an improvement.
#[cfg(test)]
struct InMemorySource {
    shards: Vec<scx_sparse::ScxCsr>,
    n_obs: usize,
    n_vars: usize,
}

#[cfg(test)]
impl ShardSource for InMemorySource {
    fn n_shards(&self) -> usize {
        self.shards.len()
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<scx_sparse::ScxCsr> {
        Ok(self.shards[shard_idx].clone())
    }
}

/// Golden per-column moment values for every finalize site in this crate,
/// pinned before Phase 7a's unification so the adoption is provably
/// behaviour-preserving rather than preserving-by-intent.
#[cfg(test)]
#[path = "moments_golden_tests.rs"]
mod moments_golden;

/// Golden values for [`binned_dispersion_norm`], pinned from the pandas
/// reference it replaces (pyscx `_hvg_helpers.py`) so the ORG-10.16-5 port is
/// provably behaviour-preserving.
#[cfg(test)]
#[path = "binned_dispersion_tests.rs"]
mod binned_dispersion_golden;

#[cfg(test)]
mod tests {
    use super::*;
    use scx_sparse::ScxCsr;

    fn make_test_source() -> InMemorySource {
        // 4 rows x 3 cols, split into 2 shards of 2 rows each:
        //   [[1, 0, 3],
        //    [0, 2, 0],
        //    [4, 0, 0],
        //    [0, 5, 6]]
        let shard0 = ScxCsr::new_unchecked(
            (2, 3),
            vec![0, 2, 3],       // indptr
            vec![0, 2, 1],       // indices
            vec![1.0, 3.0, 2.0], // data
        );
        let shard1 = ScxCsr::new_unchecked(
            (2, 3),
            vec![0, 1, 3],       // indptr
            vec![0, 1, 2],       // indices
            vec![4.0, 5.0, 6.0], // data
        );
        InMemorySource {
            shards: vec![shard0, shard1],
            n_obs: 4,
            n_vars: 3,
        }
    }

    #[test]
    fn test_streaming_mean_var() {
        let source = make_test_source();
        let stats = streaming_mean_var(&source).unwrap();

        // Column values: col0=[1,0,4,0], col1=[0,2,0,5], col2=[3,0,0,6]
        // Means:  col0=5/4=1.25, col1=7/4=1.75, col2=9/4=2.25
        let expected_means = [1.25, 1.75, 2.25];
        for (got, exp) in stats.means.iter().zip(expected_means.iter()) {
            assert!((got - exp).abs() < 1e-10, "mean: got {got}, expected {exp}");
        }

        // Variances (Bessel, ddof=1):
        //   col0: ((1-1.25)^2 + (0-1.25)^2 + (4-1.25)^2 + (0-1.25)^2) / 3
        //       = (0.0625 + 1.5625 + 7.5625 + 1.5625) / 3 = 10.75 / 3 ≈ 3.5833
        //   col1: ((0-1.75)^2 + (2-1.75)^2 + (0-1.75)^2 + (5-1.75)^2) / 3
        //       = (3.0625 + 0.0625 + 3.0625 + 10.5625) / 3 = 16.75 / 3 ≈ 5.5833
        //   col2: ((3-2.25)^2 + (0-2.25)^2 + (0-2.25)^2 + (6-2.25)^2) / 3
        //       = (0.5625 + 5.0625 + 5.0625 + 14.0625) / 3 = 24.75 / 3 = 8.25
        let expected_vars = [10.75 / 3.0, 16.75 / 3.0, 24.75 / 3.0];
        for (got, exp) in stats.variances.iter().zip(expected_vars.iter()) {
            assert!((got - exp).abs() < 1e-10, "var: got {got}, expected {exp}");
        }
    }

    #[test]
    fn test_streaming_mean_var_rejects_non_finite() {
        // 3.2: finiteness is a contract at the HVG accelerator entry. A NaN/Inf
        // in the working set is rejected, not silently summarised into garbage.
        let shard = ScxCsr::new_unchecked(
            (2, 2),
            vec![0, 2, 3],
            vec![0, 1, 0],
            vec![1.0, f32::NAN, 2.0],
        );
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 2,
            n_vars: 2,
        };
        let err = streaming_mean_var(&source).unwrap_err();
        assert!(
            matches!(err, crate::error::AccelError::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );

        // The batched entry enforces the same contract.
        let shard =
            ScxCsr::new_unchecked((2, 2), vec![0, 1, 2], vec![0, 1], vec![f32::INFINITY, 2.0]);
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 2,
            n_vars: 2,
        };
        let err = streaming_mean_var_batched(&source, &[0, 0], 1).unwrap_err();
        assert!(matches!(err, crate::error::AccelError::InvalidInput(_)));
    }

    #[test]
    fn test_streaming_clip_square_sum() {
        let source = make_test_source();
        // clip_val = [2.0, 3.0, 4.0] per column
        let clip_val = [2.0, 3.0, 4.0];

        let (bcs, sbcs) = streaming_clip_square_sum(&source, &clip_val).unwrap();

        // Nonzeros: (col0,1.0), (col2,3.0), (col1,2.0), (col0,4.0), (col1,5.0), (col2,6.0)
        // After clipping:
        //   col0: min(1,2)=1, min(4,2)=2  → sum=3, sq_sum=1+4=5
        //   col1: min(2,3)=2, min(5,3)=3  → sum=5, sq_sum=4+9=13
        //   col2: min(3,4)=3, min(6,4)=4  → sum=7, sq_sum=9+16=25
        assert!((bcs[0] - 3.0).abs() < 1e-10);
        assert!((bcs[1] - 5.0).abs() < 1e-10);
        assert!((bcs[2] - 7.0).abs() < 1e-10);
        assert!((sbcs[0] - 5.0).abs() < 1e-10);
        assert!((sbcs[1] - 13.0).abs() < 1e-10);
        assert!((sbcs[2] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_streaming_clip_square_sum_rejects_non_finite() {
        // 3.2: the clipped-sum (seurat_v3 second pass) helpers enforce the same
        // finiteness contract as the mean/var entries. Without it a NaN clips to
        // clip_val (Rust's f64::min returns the non-NaN operand) and silently
        // poisons the clipped sums.
        let shard = ScxCsr::new_unchecked(
            (2, 2),
            vec![0, 2, 3],
            vec![0, 1, 0],
            vec![1.0, f32::NAN, 2.0],
        );
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 2,
            n_vars: 2,
        };
        let err = streaming_clip_square_sum(&source, &[10.0, 10.0]).unwrap_err();
        assert!(
            matches!(err, crate::error::AccelError::InvalidInput(_)),
            "expected InvalidInput, got {err:?}"
        );

        // The batched entry enforces the same contract (Inf case).
        let shard =
            ScxCsr::new_unchecked((2, 2), vec![0, 1, 2], vec![0, 1], vec![f32::INFINITY, 2.0]);
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 2,
            n_vars: 2,
        };
        let err = streaming_clip_square_sum_batched(&source, &[0, 0], 1, &[vec![10.0, 10.0]])
            .unwrap_err();
        assert!(matches!(err, crate::error::AccelError::InvalidInput(_)));
    }

    #[test]
    fn test_streaming_mean_var_single_obs() {
        // Edge case: 1 row → denom = max(0, 1) = 1
        let shard = ScxCsr::new_unchecked((1, 2), vec![0, 2], vec![0, 1], vec![3.0, 7.0]);
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 1,
            n_vars: 2,
        };
        let stats = streaming_mean_var(&source).unwrap();
        assert!((stats.means[0] - 3.0).abs() < 1e-10);
        assert!((stats.means[1] - 7.0).abs() < 1e-10);
        // With only 1 obs, var = 0 (numerically: sum_sq - n*mean^2 = 9 - 9 = 0)
        assert!((stats.variances[0]).abs() < 1e-10);
        assert!((stats.variances[1]).abs() < 1e-10);
    }

    #[test]
    fn test_empty_source() {
        let shard = ScxCsr::new_unchecked((0, 3), vec![0], vec![], vec![]);
        let source = InMemorySource {
            shards: vec![shard],
            n_obs: 0,
            n_vars: 3,
        };
        // Should not panic on n=0; returns deterministic zeros
        let stats = streaming_mean_var(&source).unwrap();
        assert!(stats.means.iter().all(|&v| v == 0.0));
        assert!(stats.variances.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_streaming_mean_var_batched() {
        let source = make_test_source();
        // Rows: 0,1 in batch 0; rows 2,3 in batch 1
        let cell_batch = [0i32, 0, 1, 1];
        let result = streaming_mean_var_batched(&source, &cell_batch, 2).unwrap();

        assert_eq!(result.batch_counts, vec![2, 2]);

        // Batch 0: rows 0,1 → col0=[1,0], col1=[0,2], col2=[3,0]
        // Means: [0.5, 1.0, 1.5]
        let b0 = &result.per_batch[0];
        assert!((b0.means[0] - 0.5).abs() < 1e-10);
        assert!((b0.means[1] - 1.0).abs() < 1e-10);
        assert!((b0.means[2] - 1.5).abs() < 1e-10);

        // Batch 1: rows 2,3 → col0=[4,0], col1=[0,5], col2=[0,6]
        // Means: [2.0, 2.5, 3.0]
        let b1 = &result.per_batch[1];
        assert!((b1.means[0] - 2.0).abs() < 1e-10);
        assert!((b1.means[1] - 2.5).abs() < 1e-10);
        assert!((b1.means[2] - 3.0).abs() < 1e-10);

        // Global should match the non-batched result
        let global_ref = streaming_mean_var(&source).unwrap();
        for j in 0..3 {
            assert!(
                (result.global.means[j] - global_ref.means[j]).abs() < 1e-10,
                "global mean[{j}]: {} vs {}",
                result.global.means[j],
                global_ref.means[j],
            );
            assert!(
                (result.global.variances[j] - global_ref.variances[j]).abs() < 1e-10,
                "global var[{j}]: {} vs {}",
                result.global.variances[j],
                global_ref.variances[j],
            );
        }
    }

    #[test]
    fn test_streaming_clip_square_sum_batched() {
        let source = make_test_source();
        // Rows: 0,1 in batch 0; rows 2,3 in batch 1
        let cell_batch = [0i32, 0, 1, 1];
        let clip_vals = vec![
            vec![2.0, 3.0, 4.0], // batch 0 clip values
            vec![3.0, 4.0, 5.0], // batch 1 clip values
        ];

        let result =
            streaming_clip_square_sum_batched(&source, &cell_batch, 2, &clip_vals).unwrap();

        // Batch 0 nonzeros: (row0: col0=1, col2=3), (row1: col1=2)
        // Clipped by batch 0 clip_vals [2.0, 3.0, 4.0]:
        //   col0: min(1,2)=1 → sum=1, sq=1
        //   col1: min(2,3)=2 → sum=2, sq=4
        //   col2: min(3,4)=3 → sum=3, sq=9
        let (bcs0, sbcs0) = &result[0];
        assert!((bcs0[0] - 1.0).abs() < 1e-10);
        assert!((bcs0[1] - 2.0).abs() < 1e-10);
        assert!((bcs0[2] - 3.0).abs() < 1e-10);
        assert!((sbcs0[0] - 1.0).abs() < 1e-10);
        assert!((sbcs0[1] - 4.0).abs() < 1e-10);
        assert!((sbcs0[2] - 9.0).abs() < 1e-10);

        // Batch 1 nonzeros: (row2: col0=4), (row3: col1=5, col2=6)
        // Clipped by batch 1 clip_vals [3.0, 4.0, 5.0]:
        //   col0: min(4,3)=3 → sum=3, sq=9
        //   col1: min(5,4)=4 → sum=4, sq=16
        //   col2: min(6,5)=5 → sum=5, sq=25
        let (bcs1, sbcs1) = &result[1];
        assert!((bcs1[0] - 3.0).abs() < 1e-10);
        assert!((bcs1[1] - 4.0).abs() < 1e-10);
        assert!((bcs1[2] - 5.0).abs() < 1e-10);
        assert!((sbcs1[0] - 9.0).abs() < 1e-10);
        assert!((sbcs1[1] - 16.0).abs() < 1e-10);
        assert!((sbcs1[2] - 25.0).abs() < 1e-10);
    }
}
