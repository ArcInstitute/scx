// LazyShardSource — ShardSource / ColumnShardSource for streaming pipelines.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::sync::Arc;

use scx_format_io::{BackedCscReader, BackedCsrReader};
use scx_sparse::{ScxCsc, ScxCsr};

use super::*;

/// Shard source that applies lazy transforms per-shard.
///
/// Enables streaming PCA (and other shard-by-shard algorithms) on
/// lazy-transformed data without materializing the full matrix.
pub(crate) struct LazyShardSource {
    backed: Arc<BackedCsrReader>,
    /// Optional CSC sidecar reader. Populated when the underlying file
    /// has CSC shards AND the open path requests CSC capability.
    /// `None` ⇒ this `LazyShardSource` cannot serve `ColumnShardSource`
    /// methods (they will return an error).
    backed_csc: Option<Arc<BackedCscReader>>,
    transforms: Vec<Transform>,
    kept_to_global: Option<Arc<Vec<u64>>>,
    col_projection: Option<Arc<Vec<u32>>>,
    shape_val: (usize, usize),
}

impl LazyShardSource {
    /// Create a shard source with an optional pre-existing kept-to-global mapping.
    ///
    /// Pass `None` for `kept_to_global` to signal "all rows kept" — this skips
    /// the deletion-vector filtering path in `read_shard` and avoids allocating
    /// a full identity range vector.
    pub(crate) fn new(
        backed: Arc<BackedCsrReader>,
        transforms: Vec<Transform>,
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_obs: usize,
        n_vars: usize,
    ) -> Self {
        LazyShardSource {
            backed,
            backed_csc: None,
            transforms,
            kept_to_global,
            col_projection,
            shape_val: (n_obs, n_vars),
        }
    }

    /// Create a shard source with both CSR and CSC backings.
    ///
    /// Used by callers that want CSC-capable streaming. The CSC reader
    /// must already be constructed (typically by the caller after
    /// inspecting `header.has_csc()`).
    pub(crate) fn new_with_csc(
        backed: Arc<BackedCsrReader>,
        backed_csc: Option<Arc<BackedCscReader>>,
        transforms: Vec<Transform>,
        kept_to_global: Option<Arc<Vec<u64>>>,
        col_projection: Option<Arc<Vec<u32>>>,
        n_obs: usize,
        n_vars: usize,
    ) -> Self {
        LazyShardSource {
            backed,
            backed_csc,
            transforms,
            kept_to_global,
            col_projection,
            shape_val: (n_obs, n_vars),
        }
    }

    /// Returns `true` if this lazy source can serve CSC reads:
    /// CSC sidecar present, all transforms column-local, no row
    /// deletion vector active.
    ///
    /// Predicate used by `ScxLazyTransformedDataset::as_column_source()`
    /// (the analog to `ScxBackedSparseDataset::as_column_source` for
    /// the lazy-transformed wrapper). `#[allow(dead_code)]` until the
    /// CSC consumers in `pyscx::accel` reach for it.
    #[allow(dead_code)]
    pub(crate) fn supports_csc(&self) -> bool {
        self.backed_csc.is_some()
            && self.transforms.iter().all(Transform::is_column_local)
            && self.kept_to_global.is_none()
    }
}

impl scx_format_io::ShardSource for LazyShardSource {
    fn n_shards(&self) -> usize {
        self.backed.index().n_shards()
    }

    fn n_obs(&self) -> usize {
        self.shape_val.0
    }

    fn n_vars(&self) -> usize {
        match &self.col_projection {
            Some(cols) => cols.len(),
            None => self.shape_val.1,
        }
    }

    /// Row counts are unchanged by transforms and column projection — delegate
    /// to the wrapped reader's O(1) implementation.
    fn max_shard_rows(&self) -> scx_format_io::Result<usize> {
        self.backed.max_shard_rows()
    }

    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        let (s_start, _) = self.backed.index().shard_range(shard_idx).ok_or_else(|| {
            scx_format_io::ScxError::ShardIndexOutOfBounds {
                index: shard_idx,
                count: self.backed.index().n_shards(),
            }
        })?;
        let global_row = s_start as usize;

        let mut csr = self.backed.read_shard_uncached(shard_idx)?;
        apply_transforms_to_csr(&self.transforms, &mut csr, global_row);

        // Apply column projection (remap column indices to projected space)
        if let Some(ref cols) = self.col_projection {
            csr = scx_engine::projection::project_csr(&csr, cols);
        }

        // Apply deletion vector: filter to only kept rows within this shard
        if let Some(ref kept) = self.kept_to_global {
            let (s_start, s_end) = self.backed.index().shard_range(shard_idx).unwrap();
            let lo = kept.partition_point(|&r| r < s_start);
            let hi = kept.partition_point(|&r| r < s_end);
            if hi > lo {
                let local_rows: Vec<usize> = kept[lo..hi]
                    .iter()
                    .map(|&g| (g - s_start) as usize)
                    .collect();
                csr = extract_local_rows(&csr, &local_rows);
            } else {
                // No kept rows in this shard — return empty
                let n_projected = self
                    .col_projection
                    .as_ref()
                    .map_or(self.shape_val.1, |c| c.len());
                csr = ScxCsr::new_unchecked((0, n_projected), vec![0], vec![], vec![]);
            }
        }

        Ok(csr)
    }

    // col_means_and_sum_sq: use the default trait impl which iterates
    // read_shard() — transforms and col_projection are applied per-shard.
}

/// Apply column-local transforms in-place on a decoded CSC shard.
///
/// The capability gate at `ScxBackedSparseDataset::as_column_source`
/// usually filters out non-column-local transforms before this path
/// runs. As a defense in depth, this helper returns an error rather
/// than silently producing wrong results if a non-column-local
/// transform sneaks through (e.g., a future caller that bypasses the
/// gate).
fn apply_transforms_to_csc(
    transforms: &[Transform],
    csc: &mut ScxCsc,
) -> scx_format_io::Result<()> {
    for transform in transforms {
        match transform {
            Transform::Log1p => {
                for v in &mut csc.data {
                    *v = v.ln_1p();
                }
            }
            Transform::Scale { factor } => {
                for v in &mut csc.data {
                    *v = (*v as f64 * *factor) as f32;
                }
            }
            Transform::NormalizeTotal { .. } | Transform::RowScale { .. } => {
                return Err(scx_format_io::ScxError::Io(std::io::Error::other(
                    "CSC unavailable: chain contains a non-column-local transform \
                     (NormalizeTotal or RowScale). Use prefer_format='csr' or remove \
                     the transform.",
                )));
            }
        }
    }
    Ok(())
}

impl scx_format_io::ColumnShardSource for LazyShardSource {
    fn n_csc_shards(&self) -> usize {
        match &self.backed_csc {
            Some(b) => b.n_shards(),
            None => 0,
        }
    }

    fn n_obs(&self) -> usize {
        self.shape_val.0
    }

    fn n_vars(&self) -> usize {
        match &self.col_projection {
            Some(cols) => cols.len(),
            None => self.shape_val.1,
        }
    }

    fn read_csc_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsc> {
        if self.kept_to_global.is_some() {
            return Err(scx_format_io::ScxError::Io(std::io::Error::other(
                "CSC unavailable: row deletion vector is active",
            )));
        }
        let backed = self.backed_csc.as_ref().ok_or_else(|| {
            scx_format_io::ScxError::Io(std::io::Error::other(
                "CSC unavailable: file has no CSC sidecar (open with CSC enabled)",
            ))
        })?;
        let mut csc = (*backed.read_shard_cached(shard_idx)?).clone();
        apply_transforms_to_csc(&self.transforms, &mut csc)?;
        if let Some(ref proj) = self.col_projection {
            // `proj` is sorted/dedup'd GLOBAL column IDs, but `csc` is a
            // shard slab whose own column space is `0..shard_n_cols`.
            // Filter `proj` to entries inside this shard's global range,
            // remap to shard-local, then project.
            let (g_lo, g_hi) =
                scx_format_io::ColumnShardSource::csc_shard_col_range(backed.as_ref(), shard_idx)
                    .ok_or_else(|| {
                    scx_format_io::ScxError::Io(std::io::Error::other(
                        "CSC unavailable: missing shard col range for projection remap",
                    ))
                })?;
            let p_lo = proj.partition_point(|&g| g < g_lo);
            let p_hi = proj.partition_point(|&g| g < g_hi);
            let local: Vec<u32> = proj[p_lo..p_hi].iter().map(|&g| g - g_lo).collect();
            csc = scx_engine::projection::project_csc(&csc, &local);
        }
        Ok(csc)
    }

    fn read_csc_columns(&self, col_range: std::ops::Range<u32>) -> scx_format_io::Result<ScxCsc> {
        if self.kept_to_global.is_some() {
            return Err(scx_format_io::ScxError::Io(std::io::Error::other(
                "CSC unavailable: row deletion vector is active",
            )));
        }
        let backed = self.backed_csc.as_ref().ok_or_else(|| {
            scx_format_io::ScxError::Io(std::io::Error::other(
                "CSC unavailable: file has no CSC sidecar (open with CSC enabled)",
            ))
        })?;

        // When a column projection is active, the user-facing column
        // axis is the projected one. Translate the projected range into
        // the underlying global range, fetch via the inner reader, and
        // re-project the result to the projected axis.
        let mut csc = match &self.col_projection {
            Some(proj) => {
                let lo = col_range.start as usize;
                let hi = (col_range.end as usize).min(proj.len());
                if lo >= hi {
                    // Empty range — return an empty CSC sized to the
                    // projected n_vars window.
                    return Ok(ScxCsc::new_unchecked(
                        (self.shape_val.0, 0),
                        vec![0],
                        Vec::new(),
                        Vec::new(),
                    ));
                }
                let global_subset = &proj[lo..hi];
                backed.read_csc_columns_subset(global_subset)?
            }
            None => backed.read_csc_columns(col_range)?,
        };

        apply_transforms_to_csc(&self.transforms, &mut csc)?;
        Ok(csc)
    }

    fn csc_shard_col_range(&self, shard_idx: usize) -> Option<(u32, u32)> {
        let backed = self.backed_csc.as_ref()?;
        let (g_lo, g_hi) =
            scx_format_io::ColumnShardSource::csc_shard_col_range(backed.as_ref(), shard_idx)?;
        match &self.col_projection {
            Some(proj) => {
                // Map the inner shard's global range [g_lo, g_hi) onto
                // the projected axis. Consumers iterating shards see
                // ranges in the same axis as `n_vars()` (projected).
                let p_lo = proj.partition_point(|&g| g < g_lo) as u32;
                let p_hi = proj.partition_point(|&g| g < g_hi) as u32;
                Some((p_lo, p_hi))
            }
            None => Some((g_lo, g_hi)),
        }
    }
}

/// Extract specific rows from a CSR by local (within-shard) row indices.
fn extract_local_rows(csr: &ScxCsr, local_rows: &[usize]) -> ScxCsr {
    let n_cols = csr.n_cols();
    let mut indptr = Vec::with_capacity(local_rows.len() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();

    indptr.push(0i64);
    for &local_row in local_rows {
        let s = csr.indptr[local_row] as usize;
        let e = csr.indptr[local_row + 1] as usize;
        indices.extend_from_slice(&csr.indices[s..e]);
        data.extend_from_slice(&csr.data[s..e]);
        indptr.push(indices.len() as i64);
    }

    ScxCsr::new_unchecked((local_rows.len(), n_cols), indptr, indices, data)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Concatenate multiple ScxCsr slices into one.
pub(crate) fn concatenate_csr_vec(slices: &[ScxCsr], n_cols: usize) -> ScxCsr {
    if slices.is_empty() {
        return ScxCsr::new_unchecked((0, n_cols), vec![0], vec![], vec![]);
    }
    if slices.len() == 1 {
        return slices[0].clone();
    }

    let total_rows: usize = slices.iter().map(|s| s.n_rows()).sum();
    let total_nnz: usize = slices.iter().map(|s| s.nnz()).sum();

    let mut indptr = Vec::with_capacity(total_rows + 1);
    let mut indices = Vec::with_capacity(total_nnz);
    let mut data = Vec::with_capacity(total_nnz);

    indptr.push(0i64);
    let mut offset = 0i64;

    for csr in slices {
        for row in 0..csr.n_rows() {
            let s = csr.indptr[row] as usize;
            let e = csr.indptr[row + 1] as usize;
            indices.extend_from_slice(&csr.indices[s..e]);
            data.extend_from_slice(&csr.data[s..e]);
            offset += (e - s) as i64;
            indptr.push(offset);
        }
    }

    ScxCsr::new_unchecked((total_rows, n_cols), indptr, indices, data)
}

/// Extract specific rows from a CSR by global row indices.
pub(crate) fn extract_rows(csr: &ScxCsr, indices: &[u64]) -> ScxCsr {
    let n_rows = indices.len();
    let n_cols = csr.shape.1;
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut new_indices = Vec::new();
    let mut new_data = Vec::new();

    indptr.push(0i64);
    for &g_row in indices {
        let r = g_row as usize;
        let s = csr.indptr[r] as usize;
        let e = csr.indptr[r + 1] as usize;
        new_indices.extend_from_slice(&csr.indices[s..e]);
        new_data.extend_from_slice(&csr.data[s..e]);
        indptr.push(new_indices.len() as i64);
    }

    ScxCsr::new_unchecked((n_rows, n_cols), indptr, new_indices, new_data)
}
