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

    let n = n_obs as f64;
    let denom = (n - 1.0).max(1.0); // avoid division by zero for n <= 1
    let mut means = vec![0.0f64; n_vars];
    let mut variances = vec![0.0f64; n_vars];

    for j in 0..n_vars {
        let mean = col_sum[j] / n;
        means[j] = mean;
        // Var = (sum_sq - n * mean^2) / (n - 1)
        variances[j] = (col_sum_sq[j] - n * mean * mean) / denom;
        // Clamp to zero (numerical noise can produce tiny negatives)
        if variances[j] < 0.0 {
            variances[j] = 0.0;
        }
    }

    Ok(HvgStats { means, variances })
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

    let n = n_obs as f64;
    let denom = (n - 1.0).max(1.0);
    let mut means = vec![0.0f64; n_vars];
    let mut variances = vec![0.0f64; n_vars];
    for j in 0..n_vars {
        let mean = col_sum[j] / n;
        means[j] = mean;
        variances[j] = (col_sum_sq[j] - n * mean * mean) / denom;
        if variances[j] < 0.0 {
            variances[j] = 0.0;
        }
    }

    Ok(HvgStats { means, variances })
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
        let n = batch_count[b] as f64;
        let mut means = vec![0.0f64; n_vars];
        let mut variances = vec![0.0f64; n_vars];
        if batch_count[b] > 0 {
            let denom = (n - 1.0).max(1.0);
            for j in 0..n_vars {
                let mean = batch_sum[b][j] / n;
                means[j] = mean;
                variances[j] = ((batch_sum_sq[b][j] - n * mean * mean) / denom).max(0.0);
            }
        }
        per_batch.push(HvgStats { means, variances });
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
    let total_f = total_n as f64;
    let mut global_means = vec![0.0f64; n_vars];
    let mut global_variances = vec![0.0f64; n_vars];
    if total_n > 0 {
        let denom = (total_f - 1.0).max(1.0);
        for j in 0..n_vars {
            let global_sum: f64 = batch_sum.iter().map(|bs| bs[j]).sum();
            let global_sum_sq: f64 = batch_sum_sq.iter().map(|bs| bs[j]).sum();
            let mean = global_sum / total_f;
            global_means[j] = mean;
            global_variances[j] = ((global_sum_sq - total_f * mean * mean) / denom).max(0.0);
        }
    }

    Ok(BatchedHvgStats {
        per_batch,
        global: HvgStats {
            means: global_means,
            variances: global_variances,
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
