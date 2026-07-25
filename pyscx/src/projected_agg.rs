// Column-projected streaming aggregation functions.
//
// These compose BackedCsrReader::read_shard_uncached() (scx-format)
// with project_csr() (scx-engine) to support aggregation on column subsets
// without materializing the full matrix.
//
// Lives in pyscx because it depends on both scx-format and scx-engine,
// which cannot depend on each other.

use scx_engine::projection::project_csr;
use scx_format_io::BackedCsrReader;

type Result<T> = std::result::Result<T, scx_format_io::ScxError>;

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
pub fn col_nnz_projected(reader: &BackedCsrReader, col_indices: &[u32]) -> Result<Vec<u32>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0u32; n_proj];

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

/// Fused per-column sums + NNZ restricted to a subset of columns.
///
/// Single shard scan replacing `col_sums_projected()` + `col_nnz_projected()`.
/// Bit-identical to those two calls: same projection, same visit order, same
/// left-to-right f64 accumulation.
pub fn col_sums_and_nnz_projected(
    reader: &BackedCsrReader,
    col_indices: &[u32],
) -> Result<(Vec<f64>, Vec<u32>)> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];
    let mut counts = vec![0u32; n_proj];

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            for j in s..e {
                let c = projected.indices[j] as usize;
                sums[c] += projected.data[j] as f64;
                counts[c] += 1;
            }
        }
    }
    Ok((sums, counts))
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

/// Per-row sum and sum-of-squares over a column subset, computed in a single
/// shard pass (B6). Row and scalar variance over the projected columns derive
/// from these without materializing the projected submatrix, replacing the
/// previous `to_memory()` fallback. (Projected per-row nnz is served by
/// [`row_nnz_projected`].)
///
/// Variance over the projected columns (implicit zeros included, denominator
/// `n_proj = col_indices.len()`):
///   `var[i] = sumsq[i] / n_proj - (sum[i] / n_proj)^2`.
pub struct ProjectedRowStats {
    pub sums: Vec<f64>,
    pub sumsq: Vec<f64>,
}

pub fn row_stats_projected(
    reader: &BackedCsrReader,
    col_indices: &[u32],
) -> Result<ProjectedRowStats> {
    let n_obs = reader.shape().0;
    let mut sums = vec![0.0f64; n_obs];
    let mut sumsq = vec![0.0f64; n_obs];
    let mut global_row = 0usize;

    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let projected = project_csr(&csr, col_indices);
        for row in 0..projected.n_rows() {
            let s = projected.indptr[row] as usize;
            let e = projected.indptr[row + 1] as usize;
            let g = global_row + row;
            let mut sm = 0.0f64;
            let mut sq = 0.0f64;
            for &v in &projected.data[s..e] {
                let v = v as f64;
                sm += v;
                sq += v * v;
            }
            sums[g] = sm;
            sumsq[g] = sq;
        }
        global_row += csr.n_rows();
    }
    Ok(ProjectedRowStats { sums, sumsq })
}

// ---------------------------------------------------------------------------
// Fused QC row pass
// ---------------------------------------------------------------------------

/// Every per-cell quantity `calculate_qc_metrics` publishes, from one pass.
///
/// All vectors are **global**-length (`reader.shape().0`) and un-deleted;
/// deletion remapping is the caller's job (`filter_row_results`), matching the
/// convention of the other row-axis kernels here.
pub struct QcRowStats {
    /// Per-cell nnz over the visible columns (`n_genes_by_counts`).
    pub nnz: Vec<i64>,
    /// Per-cell sum over the visible columns (`total_counts`).
    pub sums: Vec<f64>,
    /// Per-cell sum over each `qc_var` gene subset, in the order the masks were
    /// supplied (`total_counts_<v>`). Empty when no `qc_var` was requested.
    pub qc_sums: Vec<Vec<f64>>,
}

impl QcRowStats {
    /// All-zero accumulator for `n_obs` cells and `n_qc` gene subsets.
    ///
    /// Exposed so the lazy-transform twin
    /// (`ScxLazyTransformedDataset::streaming_qc_row_pass`) can drive the same
    /// accumulator with its own shard loop.
    pub(crate) fn zeroed(n_obs: usize, n_qc: usize) -> Self {
        Self {
            nnz: vec![0i64; n_obs],
            sums: vec![0.0f64; n_obs],
            qc_sums: vec![vec![0.0f64; n_obs]; n_qc],
        }
    }
}

/// Accumulate one already-visible-space CSR shard into `out`.
///
/// `qc_bits[c]` is a bitmask over the requested `qc_var`s: bit *k* set means
/// visible column `c` belongs to subset *k*. Most columns belong to none, so
/// the hot path is a single load and a zero test.
///
/// Visits nonzeros in ascending column order within each row — the same order
/// the per-statistic kernels use — so the f64 sums are bit-identical to them.
pub(crate) fn accumulate_qc_rows_into(
    csr: &scx_sparse::ScxCsr,
    global_row: usize,
    qc_bits: &[u64],
    out: &mut QcRowStats,
) {
    // No subsets requested (or none representable) → skip the mask lookup
    // entirely; the sum loop is then identical to `row_sums_projected`'s.
    let plain = qc_bits.is_empty() || out.qc_sums.is_empty();
    for row in 0..csr.n_rows() {
        let s = csr.indptr[row] as usize;
        let e = csr.indptr[row + 1] as usize;
        let g = global_row + row;
        out.nnz[g] = (e - s) as i64;

        let mut total = 0.0f64;
        if plain {
            for &v in &csr.data[s..e] {
                total += v as f64;
            }
        } else {
            for j in s..e {
                let v = csr.data[j] as f64;
                total += v;
                let mut bits = qc_bits[csr.indices[j] as usize];
                while bits != 0 {
                    let k = bits.trailing_zeros() as usize;
                    out.qc_sums[k][g] += v;
                    bits &= bits - 1;
                }
            }
        }
        out.sums[g] = total;
    }
}

/// One-pass per-cell QC statistics: row nnz, row sums, and per-`qc_var` subset
/// sums over the visible column set.
///
/// Replaces the previous `row_sums` + `row_nnz` + one `row_sums_projected` per
/// `qc_var` — i.e. `2 + n_qc_vars` full shard decodes collapse into one.
///
/// `col_indices` is the on-disk index of each visible column (`None` when the
/// visible axis is the on-disk axis). `qc_bits` is indexed by **visible**
/// column and must therefore have length `col_indices.len()` (or `n_vars` when
/// `col_indices` is `None`); pass an empty slice for no `qc_var`s.
///
/// `n_qc` is the number of subsets in **this** pass, passed explicitly rather
/// than derived from the bits so an all-false mask still yields its (all-zero)
/// output row and the caller's indexing stays aligned. At most 64 per pass —
/// a caller with more subsets runs one pass per 64 (`resolve_qc_masks` does),
/// and the limit is enforced here rather than assumed.
pub fn qc_row_pass(
    reader: &BackedCsrReader,
    col_indices: Option<&[u32]>,
    qc_bits: &[u64],
    n_qc: usize,
) -> Result<QcRowStats> {
    let (n_obs, n_vars) = reader.shape();
    ensure_qc_pass_args(n_qc, qc_bits, col_indices.map_or(n_vars, |c| c.len()));

    let mut out = QcRowStats::zeroed(n_obs, n_qc);
    let mut global_row = 0usize;
    for shard_idx in 0..reader.index().n_shards() {
        let csr = reader.read_shard_uncached(shard_idx)?;
        let n_rows = csr.n_rows();
        match col_indices {
            Some(cols) => {
                accumulate_qc_rows_into(&project_csr(&csr, cols), global_row, qc_bits, &mut out)
            }
            None => accumulate_qc_rows_into(&csr, global_row, qc_bits, &mut out),
        }
        global_row += n_rows;
    }
    ensure_full_row_coverage(global_row, n_obs)?;
    Ok(out)
}

/// Per-column nnz over the visible axis, for any (projection, keep-mask) pair.
///
/// The 4-way dispatch had three copies — `ScxBackedSparseDataset::col_nnz_raw`,
/// the lazy `filter_genes` arm, and `_ComparisonResult` (which had *no*
/// projection arm at all, and so returned physical-width counts under a
/// projection). One copy, so a projection cannot be forgotten in a fourth.
pub(crate) fn col_nnz_for(
    reader: &BackedCsrReader,
    cols: Option<&[u32]>,
    kept: Option<&[u64]>,
) -> Result<Vec<u32>> {
    match (cols, kept) {
        (Some(cols), Some(kept)) => col_nnz_masked_projected(reader, kept, cols),
        (Some(cols), None) => col_nnz_projected(reader, cols),
        (None, Some(kept)) => Ok(reader
            .col_nnz_masked(kept)?
            .iter()
            .map(|&v| v as u32)
            .collect()),
        (None, None) => reader.col_nnz(),
    }
}

/// Per-row nnz over the visible columns. Global-length: the caller applies the
/// keep-mask, because row vectors index the global row axis.
pub(crate) fn row_nnz_for(reader: &BackedCsrReader, cols: Option<&[u32]>) -> Result<Vec<i64>> {
    match cols {
        Some(cols) => row_nnz_projected(reader, cols),
        None => reader.row_nnz(),
    }
}

/// Total stored entries inside the visible window.
pub(crate) fn total_nnz_for(
    reader: &BackedCsrReader,
    cols: Option<&[u32]>,
    kept: Option<&[u64]>,
) -> Result<usize> {
    match (cols, kept) {
        // Sum as u64: projected nnz can exceed u32::MAX at atlas scale.
        (Some(_), _) => Ok(col_nnz_for(reader, cols, kept)?
            .iter()
            .map(|&v| v as u64)
            .sum::<u64>() as usize),
        (None, Some(kept)) => {
            let all_nnz = reader.row_nnz()?;
            Ok(kept.iter().map(|&g| all_nnz[g as usize]).sum::<i64>() as usize)
        }
        (None, None) => reader.total_nnz(),
    }
}

/// Enforce the per-pass subset contract. Shared by the backed and lazy row
/// passes.
///
/// These are **caller-contract** violations, not data conditions — the Python
/// entry point (`resolve_qc_masks`) chunks at 64 and sizes the mask to the
/// visible axis, so neither can fire from a supported path. They are release-
/// active `assert!`s rather than `debug_assert!`s because an oversized `n_qc`
/// would shift past the mask width and *silently drop subsets* in the shipped
/// `.so`, which is precisely the failure mode this module exists to prevent.
pub(crate) fn ensure_qc_pass_args(n_qc: usize, qc_bits: &[u64], n_visible: usize) {
    assert!(
        n_qc <= 64,
        "qc_row_pass accepts at most 64 subsets per pass, got {n_qc}; \
         callers with more must run one pass per 64"
    );
    assert!(
        qc_bits.is_empty() || qc_bits.len() == n_visible,
        "qc_var bitmask has {} entries but the visible axis has {n_visible} columns",
        qc_bits.len()
    );
}

/// The row pass writes by index into `n_obs`-sized vectors from a running
/// counter, so shards that don't tile `[0, n_obs)` would publish zeros as real
/// QC values. The pre-fusion kernels built their output with `extend`, which
/// surfaced that corruption as a length mismatch — keep the loudness. Unlike
/// the argument checks above this *is* a data condition (a corrupt or
/// hand-edited catalog), so it returns an error rather than panicking.
pub(crate) fn ensure_full_row_coverage(seen: usize, n_obs: usize) -> Result<()> {
    if seen != n_obs {
        return Err(scx_format_io::ScxError::InvalidCatalog(format!(
            "shards cover {seen} rows but the header declares {n_obs}; refusing to \
             publish QC metrics from a partially-covered row axis"
        )));
    }
    Ok(())
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
) -> Result<Vec<u32>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0u32; n_proj];

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

/// Fused column sums + NNZ restricted to a subset of columns AND kept rows.
///
/// Single shard scan replacing `col_sums_masked_projected()` +
/// `col_nnz_masked_projected()`; bit-identical to both.
pub fn col_sums_and_nnz_masked_projected(
    reader: &BackedCsrReader,
    kept_rows: &[u64],
    col_indices: &[u32],
) -> Result<(Vec<f64>, Vec<u32>)> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];
    let mut counts = vec![0u32; n_proj];

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
                sums[c] += projected.data[j] as f64;
                counts[c] += 1;
            }
        }
    }
    Ok((sums, counts))
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

use scx_format_io::ColumnShardSource;

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
) -> Result<Vec<u32>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0u32; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        counts[output_col] += (e - s) as u32;
    })?;
    Ok(counts)
}

/// Fused CSC twin of [`col_sums_and_nnz_projected`].
///
/// One `walk_csc_runs` sweep of the sidecar instead of the two that
/// `col_sums_projected_csc` + `col_nnz_projected_csc` perform; bit-identical.
pub fn col_sums_and_nnz_projected_csc(
    source: &dyn ColumnShardSource,
    col_indices: &[u32],
) -> Result<(Vec<f64>, Vec<u32>)> {
    let n_proj = col_indices.len();
    let mut sums = vec![0.0f64; n_proj];
    let mut counts = vec![0u32; n_proj];
    walk_csc_runs(source, col_indices, |local_col, output_col, csc| {
        let s = csc.indptr[local_col] as usize;
        let e = csc.indptr[local_col + 1] as usize;
        let mut acc = 0.0f64;
        for &v in &csc.data[s..e] {
            acc += v as f64;
        }
        sums[output_col] += acc;
        counts[output_col] += (e - s) as u32;
    })?;
    Ok((sums, counts))
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
) -> Result<Vec<u32>> {
    let n_proj = col_indices.len();
    let mut counts = vec![0u32; n_proj];
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
