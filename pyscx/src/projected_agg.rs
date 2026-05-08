// Column-projected streaming aggregation functions.
//
// These compose BackedCsrReader::read_shard_uncached() (scx-format)
// with project_csr() (scx-engine) to support aggregation on column subsets
// without materializing the full matrix.
//
// Lives in pyscx because it depends on both scx-format and scx-engine,
// which cannot depend on each other.

use scx_engine::projection::project_csr;
use scx_format::BackedCsrReader;

type Result<T> = std::result::Result<T, scx_format::ScxError>;

// ---------------------------------------------------------------------------
// Column-axis aggregation (axis=0), projected
// ---------------------------------------------------------------------------

/// Streaming column sums restricted to a subset of columns.
///
/// For each shard: decode → project_csr(col_indices) → accumulate sums.
/// Returns `Vec<f64>` of length `col_indices.len()`.
pub fn col_sums_projected(reader: &BackedCsrReader, col_indices: &[u32]) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                sums[projected.indices[j] as usize] += projected.data[j] as f64;
            }
        }
    }
    Ok(sums)
}

/// Streaming per-column NNZ restricted to a subset of columns.
pub fn col_nnz_projected(reader: &BackedCsrReader, col_indices: &[u32]) -> Result<Vec<i64>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0i64; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                counts[projected.indices[j] as usize] += 1;
            }
        }
    }
    Ok(counts)
}

/// Streaming per-column max restricted to a subset of columns.
///
/// Accounts for implicit zeros: if col_nnz < n_obs, max is at least 0.0.
pub fn col_max_projected(
    reader: &BackedCsrReader,
    col_indices: &[u32],
    n_obs: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut maxes = vec![f64::NEG_INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                let v = projected.data[j] as f64;
                maxes[c] = maxes[c].max(v);
                col_nnz[c] += 1;
            }
        }
    }

    // Implicit zeros: if column has fewer stored entries than n_obs
    for c in 0..n_proj {
        if col_nnz[c] < n_obs {
            if maxes[c] == f64::NEG_INFINITY {
                maxes[c] = 0.0;
            } else {
                maxes[c] = maxes[c].max(0.0);
            }
        }
    }
    Ok(maxes)
}

/// Streaming per-column min restricted to a subset of columns.
///
/// Accounts for implicit zeros: if col_nnz < n_obs, min is at most 0.0.
pub fn col_min_projected(
    reader: &BackedCsrReader,
    col_indices: &[u32],
    n_obs: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut mins = vec![f64::INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                let v = projected.data[j] as f64;
                mins[c] = mins[c].min(v);
                col_nnz[c] += 1;
            }
        }
    }

    for c in 0..n_proj {
        if col_nnz[c] < n_obs {
            if mins[c] == f64::INFINITY {
                mins[c] = 0.0;
            } else {
                mins[c] = mins[c].min(0.0);
            }
        }
    }
    Ok(mins)
}

/// Streaming per-column variance restricted to a subset of columns.
///
/// Two-pass algorithm:
///   1. Compute projected column means via col_sums_projected / n_obs
///   2. Stream shards, project, accumulate (x - mean)²
///   3. Add zero-entry contributions: (n_obs - col_nnz) * mean²
pub fn col_var_projected(
    reader: &BackedCsrReader,
    col_indices: &[u32],
    n_obs: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    if n_obs == 0 {
        return Ok(vec![0.0f64; n_proj]);
    }

    // Pass 1: column means
    let col_sums = col_sums_projected(reader, col_indices)?;
    let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_obs as f64).collect();

    // Pass 2: accumulate (val - mean)² for stored entries
    let mut sq_devs = vec![0.0f64; n_proj];
    let mut col_nnz = vec![0usize; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                let diff = projected.data[j] as f64 - col_means[c];
                sq_devs[c] += diff * diff;
                col_nnz[c] += 1;
            }
        }
    }

    // Add contribution from implicit zeros
    let mut variances = vec![0.0f64; n_proj];
    for c in 0..n_proj {
        debug_assert!(
            col_nnz[c] <= n_obs,
            "col_nnz[{}] = {} exceeds n_obs = {}",
            c,
            col_nnz[c],
            n_obs
        );
        let n_zeros = n_obs - col_nnz[c];
        let total_sq_dev = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
        variances[c] = total_sq_dev / n_obs as f64;
    }
    Ok(variances)
}

// ---------------------------------------------------------------------------
// Row-axis aggregation (axis=1), projected
// ---------------------------------------------------------------------------

/// Row sums restricted to a subset of columns.
///
/// For each shard: decode → project_csr → accumulate per-row sums.
/// Returns `Vec<f64>` of length n_obs (all rows in the backing store).
pub fn row_sums_projected(reader: &BackedCsrReader, col_indices: &[u32]) -> Result<Vec<f64>> {
    let n_obs = reader.shape().0;
    let mut sums = vec![0.0f64; n_obs];
    let mut global_row = 0usize;

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            sums[global_row + row] = projected.data[s..e].iter().map(|&v| v as f64).sum();
        }
        global_row += csr.n_rows();
    }
    Ok(sums)
}

/// Row NNZ restricted to a subset of columns.
pub fn row_nnz_projected(reader: &BackedCsrReader, col_indices: &[u32]) -> Result<Vec<i64>> {
    let n_obs = reader.shape().0;
    let mut counts = vec![0i64; n_obs];
    let mut global_row = 0usize;

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            counts[global_row + row] = (e - s) as i64;
        }
        global_row += csr.n_rows();
    }
    Ok(counts)
}

/// Fused row NNZ + sums restricted to a subset of columns.
///
/// Computes both row NNZ and row sums in a single shard scan, avoiding the
/// double I/O of calling `row_nnz_projected()` + `row_sums_projected()`.
/// Used by `filter_cells` when both `min_genes` and `min_counts` are specified
/// and a column projection is active.
pub fn row_nnz_and_sums_projected(
    reader: &BackedCsrReader,
    col_indices: &[u32],
) -> Result<(Vec<i64>, Vec<f64>)> {
    let n_obs = reader.shape().0;
    let mut counts = vec![0i64; n_obs];
    let mut sums = vec![0.0f64; n_obs];
    let mut global_row = 0usize;

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            counts[global_row + row] = (e - s) as i64;
            sums[global_row + row] = projected.data[s..e].iter().map(|&v| v as f64).sum();
        }
        global_row += csr.n_rows();
    }
    Ok((counts, sums))
}

// ---------------------------------------------------------------------------
// Masked + projected aggregation (deletion vector + column subset)
// ---------------------------------------------------------------------------

/// Column sums restricted to a subset of columns AND kept rows.
///
/// For each shard: decode → project_csr → iterate only kept rows → accumulate.
pub fn col_sums_masked_projected(
    reader: &BackedCsrReader,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let (s_start, s_end) = match reader.index().shard_range(shard_idx) {
            Some(r) => r,
            None => continue,
        };

        let projected = project_csr(&csr, col_indices);

        // Binary-search to find the sub-slice of kept_rows within [s_start, s_end)
        let lo = kept_rows.partition_point(|&r| r < s_start);
        let hi = kept_rows.partition_point(|&r| r < s_end);
        for &global_row in &kept_rows[lo..hi] {
            let local_row = (global_row - s_start) as usize;
            let s = projected.indptr[local_row] as usize;
            let e = projected.indptr[local_row + 1] as usize;
            for j in s..e {
                sums[projected.indices[j] as usize] += projected.data[j] as f64;
            }
        }
    }
    Ok(sums)
}

/// Column NNZ restricted to a subset of columns AND kept rows.
pub fn col_nnz_masked_projected(
    reader: &BackedCsrReader,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<Vec<i64>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0i64; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let (s_start, s_end) = match reader.index().shard_range(shard_idx) {
            Some(r) => r,
            None => continue,
        };

        let projected = project_csr(&csr, col_indices);

        let lo = kept_rows.partition_point(|&r| r < s_start);
        let hi = kept_rows.partition_point(|&r| r < s_end);
        for &global_row in &kept_rows[lo..hi] {
            let local_row = (global_row - s_start) as usize;
            let s = projected.indptr[local_row] as usize;
            let e = projected.indptr[local_row + 1] as usize;
            for j in s..e {
                counts[projected.indices[j] as usize] += 1;
            }
        }
    }
    Ok(counts)
}

/// Column max restricted to a subset of columns AND kept rows.
pub fn col_max_masked_projected(
    reader: &BackedCsrReader,
    kept_rows: &[u64],
    col_indices: &[u32],
    n_kept: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut maxes = vec![f64::NEG_INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let (s_start, s_end) = match reader.index().shard_range(shard_idx) {
            Some(r) => r,
            None => continue,
        };

        let projected = project_csr(&csr, col_indices);

        let lo = kept_rows.partition_point(|&r| r < s_start);
        let hi = kept_rows.partition_point(|&r| r < s_end);
        for &global_row in &kept_rows[lo..hi] {
            let local_row = (global_row - s_start) as usize;
            let s = projected.indptr[local_row] as usize;
            let e = projected.indptr[local_row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                let v = projected.data[j] as f64;
                maxes[c] = maxes[c].max(v);
                col_nnz[c] += 1;
            }
        }
    }

    for c in 0..n_proj {
        if col_nnz[c] < n_kept {
            if maxes[c] == f64::NEG_INFINITY {
                maxes[c] = 0.0;
            } else {
                maxes[c] = maxes[c].max(0.0);
            }
        }
    }
    Ok(maxes)
}

/// Column min restricted to a subset of columns AND kept rows.
pub fn col_min_masked_projected(
    reader: &BackedCsrReader,
    kept_rows: &[u64],
    col_indices: &[u32],
    n_kept: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut mins = vec![f64::INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let (s_start, s_end) = match reader.index().shard_range(shard_idx) {
            Some(r) => r,
            None => continue,
        };

        let projected = project_csr(&csr, col_indices);

        let lo = kept_rows.partition_point(|&r| r < s_start);
        let hi = kept_rows.partition_point(|&r| r < s_end);
        for &global_row in &kept_rows[lo..hi] {
            let local_row = (global_row - s_start) as usize;
            let s = projected.indptr[local_row] as usize;
            let e = projected.indptr[local_row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                let v = projected.data[j] as f64;
                mins[c] = mins[c].min(v);
                col_nnz[c] += 1;
            }
        }
    }

    for c in 0..n_proj {
        if col_nnz[c] < n_kept {
            if mins[c] == f64::INFINITY {
                mins[c] = 0.0;
            } else {
                mins[c] = mins[c].min(0.0);
            }
        }
    }
    Ok(mins)
}

/// Column variance restricted to a subset of columns AND kept rows.
pub fn col_var_masked_projected(
    reader: &BackedCsrReader,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let n_kept = kept_rows.len();
    if n_kept == 0 {
        return Ok(vec![0.0f64; n_proj]);
    }

    // Pass 1: column means over kept rows
    let col_sums = col_sums_masked_projected(reader, kept_rows, col_indices)?;
    let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_kept as f64).collect();

    // Pass 2: accumulate (val - mean)² for stored entries in kept rows
    let mut sq_devs = vec![0.0f64; n_proj];
    let mut col_nnz = vec![0usize; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let (s_start, s_end) = match reader.index().shard_range(shard_idx) {
            Some(r) => r,
            None => continue,
        };

        let projected = project_csr(&csr, col_indices);

        let lo = kept_rows.partition_point(|&r| r < s_start);
        let hi = kept_rows.partition_point(|&r| r < s_end);
        for &global_row in &kept_rows[lo..hi] {
            let local_row = (global_row - s_start) as usize;
            let s = projected.indptr[local_row] as usize;
            let e = projected.indptr[local_row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                let diff = projected.data[j] as f64 - col_means[c];
                sq_devs[c] += diff * diff;
                col_nnz[c] += 1;
            }
        }
    }

    // Add zero-entry contributions
    let mut variances = vec![0.0f64; n_proj];
    for c in 0..n_proj {
        debug_assert!(
            col_nnz[c] <= n_kept,
            "col_nnz[{}] = {} exceeds n_kept = {}",
            c,
            col_nnz[c],
            n_kept
        );
        let n_zeros = n_kept - col_nnz[c];
        let total = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
        variances[c] = total / n_kept as f64;
    }
    Ok(variances)
}

// ---------------------------------------------------------------------------
// CSC twins
//
// Each `_csc` twin reads `col_indices` directly via the
// `ColumnShardSource` trait — one CSC slab per contiguous run of
// requested columns, no per-shard CSR decode + project_csr round trip.
// ---------------------------------------------------------------------------

use scx_format::ColumnShardSource;

/// Helper: walk `col_indices` in sorted contiguous-run order, calling
/// `f(local_col_in_run, output_col_idx, csc_run)` for each output
/// column. `csc_run` is the CSC slab covering one run; `local_col_in_run`
/// is the column within that run, and `output_col_idx` is the position
/// of that column in the user-facing `col_indices` order.
fn walk_csc_runs(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
    mut f: impl FnMut(usize, usize, &scx_sparse::ScxCsc),
) -> Result<()> {
    if col_indices.is_empty() {
        return Ok(());
    }
    let mut sorted_with_pos: Vec<(u32, usize)> = col_indices
        .iter()
        .copied()
        .enumerate()
        .map(|(i, c)| (c, i))
        .collect();
    sorted_with_pos.sort_by_key(|(c, _)| *c);

    let mut i = 0;
    while i < sorted_with_pos.len() {
        let mut j = i + 1;
        while j < sorted_with_pos.len() && sorted_with_pos[j].0 == sorted_with_pos[j - 1].0 + 1 {
            j += 1;
        }
        let run_start = sorted_with_pos[i].0;
        let run_end = sorted_with_pos[j - 1].0 + 1;
        let csc_run = source.read_csc_columns(run_start..run_end)?;
        for (local_col, &(_, output_col)) in sorted_with_pos[i..j].iter().enumerate() {
            f(local_col, output_col, &csc_run);
        }
        i = j;
    }
    Ok(())
}

/// CSC twin of [`col_sums_projected`].
pub fn col_sums_projected_csc(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        let mut acc = 0.0f64;
        for &v in &csc.data[s..e] {
            acc += v as f64;
        }
        sums[output_col] += acc;
    })?;
    Ok(sums)
}

/// CSC twin of [`col_nnz_projected`].
pub fn col_nnz_projected_csc(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
) -> Result<Vec<i64>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0i64; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        counts[output_col] += (e - s) as i64;
    })?;
    Ok(counts)
}

/// CSC twin of [`col_max_projected`]. `n_obs` accounts for implicit
/// zeros (columns whose nnz < n_obs include 0.0 in their domain).
pub fn col_max_projected_csc(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
    n_obs: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut maxes = vec![f64::NEG_INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        let mut m = maxes[output_col];
        for &v in &csc.data[s..e] {
            let v = v as f64;
            if v > m {
                m = v;
            }
        }
        maxes[output_col] = m;
        col_nnz[output_col] += e - s;
    })?;
    for c in 0..n_proj {
        if col_nnz[c] < n_obs {
            if maxes[c] == f64::NEG_INFINITY {
                maxes[c] = 0.0;
            } else {
                maxes[c] = maxes[c].max(0.0);
            }
        }
    }
    Ok(maxes)
}

/// CSC twin of [`col_min_projected`].
pub fn col_min_projected_csc(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
    n_obs: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut mins = vec![f64::INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        let mut m = mins[output_col];
        for &v in &csc.data[s..e] {
            let v = v as f64;
            if v < m {
                m = v;
            }
        }
        mins[output_col] = m;
        col_nnz[output_col] += e - s;
    })?;
    for c in 0..n_proj {
        if col_nnz[c] < n_obs {
            if mins[c] == f64::INFINITY {
                mins[c] = 0.0;
            } else {
                mins[c] = mins[c].min(0.0);
            }
        }
    }
    Ok(mins)
}

/// CSC twin of [`col_var_projected`]. Single-pass: tracks `sum_x` and
/// `sum_x²` per column, computes the variance using
/// `var = (sum_x² - n·mean²) / n` with implicit-zero correction
/// (`n_zeros = n_obs - col_nnz`). Numerically equivalent to the
/// two-pass CSR formulation within f64 epsilon.
pub fn col_var_projected_csc(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
    n_obs: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    if n_obs == 0 {
        return Ok(vec![0.0f64; n_proj]);
    }
    let mut sum_x = vec![0.0f64; n_proj];
    let mut sum_x2 = vec![0.0f64; n_proj];
    let mut col_nnz = vec![0usize; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        let mut sx = 0.0f64;
        let mut sx2 = 0.0f64;
        for &v in &csc.data[s..e] {
            let v = v as f64;
            sx += v;
            sx2 += v * v;
        }
        sum_x[output_col] += sx;
        sum_x2[output_col] += sx2;
        col_nnz[output_col] += e - s;
    })?;
    // var = E[X²] - (E[X])²; the implicit-zero entries contribute 0 to
    // both sum_x and sum_x², so the formula is just a population mean
    // and second moment over n_obs.
    let n = n_obs as f64;
    let mut variances = vec![0.0f64; n_proj];
    for c in 0..n_proj {
        let mean = sum_x[c] / n;
        let var = sum_x2[c] / n - mean * mean;
        variances[c] = if var < 0.0 { 0.0 } else { var };
    }
    Ok(variances)
}

/// CSC twin of [`col_sums_masked_projected`].
///
/// The CSC reader can't pre-filter rows, so we rebuild a `kept_set`
/// (BTreeSet) for fast O(log n) row lookups. Two-pointer merge over a
/// sorted `kept_rows` slice is asymptotically faster, but a straight
/// `partition_point` per row works fine and avoids the bookkeeping.
///
/// Currently unreachable from the public pyfunctions: row deletion
/// vectors disqualify the CSC capability gate. Kept here for symmetry
/// and so future consumers that handle deletion vectors themselves can
/// reach for the CSC path.
#[allow(dead_code)]
pub fn col_sums_masked_projected_csc(
    source: &dyn ColumnShardSource,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        for k in s..e {
            let row = csc.indices[k] as u64;
            if kept_rows.binary_search(&row).is_ok() {
                sums[output_col] += csc.data[k] as f64;
            }
        }
    })?;
    Ok(sums)
}

/// CSC twin of [`col_nnz_masked_projected`]. See
/// [`col_sums_masked_projected_csc`] for the reachability note.
#[allow(dead_code)]
pub fn col_nnz_masked_projected_csc(
    source: &dyn ColumnShardSource,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<Vec<i64>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0i64; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        for k in s..e {
            let row = csc.indices[k] as u64;
            if kept_rows.binary_search(&row).is_ok() {
                counts[output_col] += 1;
            }
        }
    })?;
    Ok(counts)
}

/// CSC twin of [`col_max_masked_projected`]. See
/// [`col_sums_masked_projected_csc`] for the reachability note.
#[allow(dead_code)]
pub fn col_max_masked_projected_csc(
    source: &dyn ColumnShardSource,
    kept_rows: &[u64],
    col_indices: &[u32],
    n_kept: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut maxes = vec![f64::NEG_INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        for k in s..e {
            let row = csc.indices[k] as u64;
            if kept_rows.binary_search(&row).is_ok() {
                let v = csc.data[k] as f64;
                if v > maxes[output_col] {
                    maxes[output_col] = v;
                }
                col_nnz[output_col] += 1;
            }
        }
    })?;
    for c in 0..n_proj {
        if col_nnz[c] < n_kept {
            if maxes[c] == f64::NEG_INFINITY {
                maxes[c] = 0.0;
            } else {
                maxes[c] = maxes[c].max(0.0);
            }
        }
    }
    Ok(maxes)
}

/// CSC twin of [`col_min_masked_projected`]. See
/// [`col_sums_masked_projected_csc`] for the reachability note.
#[allow(dead_code)]
pub fn col_min_masked_projected_csc(
    source: &dyn ColumnShardSource,
    kept_rows: &[u64],
    col_indices: &[u32],
    n_kept: usize,
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let mut mins = vec![f64::INFINITY; n_proj];
    let mut col_nnz = vec![0usize; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        for k in s..e {
            let row = csc.indices[k] as u64;
            if kept_rows.binary_search(&row).is_ok() {
                let v = csc.data[k] as f64;
                if v < mins[output_col] {
                    mins[output_col] = v;
                }
                col_nnz[output_col] += 1;
            }
        }
    })?;
    for c in 0..n_proj {
        if col_nnz[c] < n_kept {
            if mins[c] == f64::INFINITY {
                mins[c] = 0.0;
            } else {
                mins[c] = mins[c].min(0.0);
            }
        }
    }
    Ok(mins)
}

/// CSC twin of [`col_var_masked_projected`].
///
/// Single-pass over kept rows. Maintains `sum_x` / `sum_x²` per
/// column over kept rows only; implicit-zero entries within
/// `kept_rows` contribute zero to both sums, so the population
/// variance over `n_kept` rows is computed directly.
///
/// See [`col_sums_masked_projected_csc`] for the reachability note.
#[allow(dead_code)]
pub fn col_var_masked_projected_csc(
    source: &dyn ColumnShardSource,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<Vec<f64>> {
    let n_proj = col_indices.len();
    let n_kept = kept_rows.len();
    if n_kept == 0 {
        return Ok(vec![0.0f64; n_proj]);
    }
    let mut sum_x = vec![0.0f64; n_proj];
    let mut sum_x2 = vec![0.0f64; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        for k in s..e {
            let row = csc.indices[k] as u64;
            if kept_rows.binary_search(&row).is_ok() {
                let v = csc.data[k] as f64;
                sum_x[output_col] += v;
                sum_x2[output_col] += v * v;
            }
        }
    })?;
    let n = n_kept as f64;
    let mut variances = vec![0.0f64; n_proj];
    for c in 0..n_proj {
        let mean = sum_x[c] / n;
        let var = sum_x2[c] / n - mean * mean;
        variances[c] = if var < 0.0 { 0.0 } else { var };
    }
    Ok(variances)
}
