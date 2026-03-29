// Column-projected streaming aggregation functions.
//
// These compose BackedCsrReader::read_shard_cached() (scx-format)
// with project_csr() (scx-engine) to support aggregation on column subsets
// without materializing the full matrix.
//
// Lives in pyscx because it depends on both scx-format and scx-engine,
// which cannot depend on each other. See Phase4-ACC-ALL.md §3.2.

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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let csr = reader.read_shard_cached(shard_idx)?;
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
        let n_zeros = n_kept - col_nnz[c];
        let total = sq_devs[c] + n_zeros as f64 * col_means[c] * col_means[c];
        variances[c] = total / n_kept as f64;
    }
    Ok(variances)
}
