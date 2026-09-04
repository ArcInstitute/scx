//! Streaming + visible-space aggregation for `ScxLazyTransformedDataset`
//! — a second inherent impl split out of `dataset.rs` (ORG-10.16-6).

// ScxLazyTransformedDataset — PyO3 class for lazy per-row transforms.
//
// Extracted from the former pyscx/src/lazy_transform.rs (T5.7).

use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use scx_format_io::prefetch;
use scx_sparse::ScxCsr;

use crate::backed::detached;
use crate::convert::csr_to_scipy;

use super::*;

impl ScxLazyTransformedDataset {
    // --- Streaming aggregation through transforms ---
    //
    // These methods stream shard-by-shard, applying lazy transforms in-place,
    // then accumulating per-column or per-row statistics. All column-axis
    // methods return vectors in **physical column space** (length =
    // backed.shape().1). Callers must post-filter via apply_col_projection_to_vec()
    // when col_projection is active. See the doc comment on that method for
    // rationale.

    /// Stream all shards, apply transforms, compute per-row sums over **every
    /// on-disk column**, ignoring `col_projection`.
    ///
    /// The `_physical` suffix is load-bearing. Row statistics — unlike column
    /// statistics — are *not* independent of the projection: a row sum over the
    /// physical axis includes genes the caller cannot see. Anything user-facing
    /// wants [`Self::streaming_row_sums`], which honors the projection; this
    /// kernel exists only as that method's no-projection arm. Reaching for the
    /// physical variant by accident is the §9.18 bug class (six call sites once
    /// summed hidden genes into `X.sum(axis=1)` and `filter_cells` thresholds).
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_row_sums_physical(&self) -> Result<Vec<f64>, String> {
        let n_obs_global = self.backed.shape().0;
        let mut sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;

        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                self.apply_transforms(&mut csr, global_row);
                for row in 0..csr.n_rows() {
                    let s = csr.indptr[row] as usize;
                    let e = csr.indptr[row + 1] as usize;
                    sums[global_row + row] = csr.data[s..e].iter().map(|&v| v as f64).sum();
                }
                global_row += csr.n_rows();
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(sums)
    }

    /// Stream all shards, apply transforms, compute per-row NNZ and sums over
    /// the **visible** columns in a single pass.
    ///
    /// Avoids the double I/O of `row_nnz_raw()` + `streaming_row_sums()`. Used
    /// by `filter_cells` when both `min_genes` and `min_counts` are given.
    ///
    /// Delegates to [`Self::streaming_qc_row_pass`] with an empty subset mask —
    /// that kernel's `plain` branch *is* this computation, so there is one row
    /// walk to keep correct rather than two. NNZ comes from the (projected)
    /// `indptr`, which the value-wise transforms leave untouched.
    ///
    /// Returns global-length vectors (NOT filtered through deletion vectors).
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_row_nnz_and_sums(&self) -> Result<(Vec<i64>, Vec<f64>), String> {
        let stats = self.streaming_qc_row_pass(&[], 0)?;
        Ok((stats.nnz, stats.sums))
    }

    /// Stream all shards, apply transforms, project to visible columns, compute per-row sums.
    ///
    /// **Scanpy compatibility:** After `filter_genes()`, scanpy's `normalize_total`
    /// sums only over the visible (kept) gene set because `adata.X` is already sliced.
    /// In SCX backed mode, `adata.X` is still the full-width matrix with a
    /// `col_projection` mask. This method applies transforms to the full-width shard
    /// first (so prior `NormalizeTotal` transforms divide by the correct whole-row
    /// denominator), then calls `project_csr` to restrict to the projected gene subset
    /// before summing each row.
    ///
    /// Falls back to [`Self::streaming_row_sums_physical`] when no
    /// `col_projection` is active.
    ///
    /// **This is the row-sum kernel callers want.** It owns the unqualified
    /// name deliberately: the projection-blind variant is
    /// `streaming_row_sums_physical`, so picking the wrong one now requires
    /// typing a suffix that says what it does.
    ///
    /// Returns a global-length vector (`n_obs_global`), NOT filtered through
    /// deletion vectors.
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_row_sums(&self) -> Result<Vec<f64>, String> {
        match self.col_projection.clone() {
            Some(cols) => self.streaming_row_sums_for_cols(&cols),
            None => self.streaming_row_sums_physical(),
        }
    }

    /// Stream all shards, apply transforms to the **full-width** shard, then
    /// restrict to `cols` before summing each row.
    ///
    /// `cols` are indices into the underlying reader's column space (on-disk
    /// columns), NOT the visible axis — a caller holding visible-space indices
    /// must compose them through `col_projection` first.
    ///
    /// Transforms run before the projection so a prior `NormalizeTotal`
    /// divides by the correct whole-row denominator; see
    /// [`Self::streaming_row_sums`].
    ///
    /// Its only caller is [`Self::streaming_row_sums`], which passes
    /// `self.col_projection` — already on-disk indices by construction, so the
    /// composition caveat above does not apply there. Any *new* caller holding
    /// visible-space indices must compose them itself; handing visible indices
    /// straight to a reader is exactly the bug class this module's QC pass
    /// exists to avoid.
    ///
    /// Returns a global-length vector (`n_obs_global`), NOT filtered through
    /// deletion vectors.
    pub(crate) fn streaming_row_sums_for_cols(&self, cols: &[u32]) -> Result<Vec<f64>, String> {
        let n_obs_global = self.backed.shape().0;
        let mut sums = vec![0.0f64; n_obs_global];
        let mut global_row = 0usize;
        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                self.apply_transforms(&mut csr, global_row);
                let projected = scx_engine::projection::project_csr(&csr, cols);
                for row in 0..projected.n_rows() {
                    let s = projected.indptr[row] as usize;
                    let e = projected.indptr[row + 1] as usize;
                    sums[global_row + row] = projected.data[s..e].iter().map(|&v| v as f64).sum();
                }
                global_row += csr.n_rows();
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(sums)
    }

    /// Stream all shards, apply transforms, compute per-column sums.
    ///
    /// Returns a vector of length `backed.shape().1` (physical column count),
    /// NOT `shape_val.1` (projected). Callers must apply
    /// `apply_col_projection_to_vec()` before exposing to Python.
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_col_sums(&self) -> Result<Vec<f64>, String> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
        let mut sums = vec![0.0f64; n_vars];
        let mut global_row = 0usize;

        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                self.apply_transforms(&mut csr, global_row);
                for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                    sums[col as usize] += val as f64;
                }
                global_row += csr.n_rows();
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(sums)
    }

    /// Stream all shards, apply transforms, compute per-column variance (pop, ddof=0).
    ///
    /// Two-pass: first compute means via col_sums, then accumulate (x-mean)².
    ///
    /// Returns a vector of length `backed.shape().1` (physical column count).
    /// Callers must apply `apply_col_projection_to_vec()` before exposing to
    /// Python. This is correct because per-column variance is independent —
    /// computing var for filtered-out columns is wasted work but doesn't affect
    /// the values for kept columns.
    pub(crate) fn streaming_col_var(&self) -> Result<Vec<f64>, String> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
        let n_obs = self.shape_val.0;
        if n_obs == 0 {
            return Ok(vec![0.0f64; n_vars]);
        }

        // Pass 1: column means through transforms
        let col_sums_filtered = if self.kept_to_global.is_some() {
            self.streaming_col_sums_masked()?
        } else {
            self.streaming_col_sums()?
        };
        let col_means: Vec<f64> = col_sums_filtered
            .iter()
            .map(|&s| s / n_obs as f64)
            .collect();

        // Pass 2: accumulate (val - mean)² for stored entries
        let mut sq_devs = vec![0.0f64; n_vars];
        let mut col_nnz = vec![0usize; n_vars];
        let mut global_row = 0usize;

        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                self.apply_transforms(&mut csr, global_row);

                if let Some(ref kept) = self.kept_to_global {
                    let (s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                        Some(r) => r,
                        None => {
                            global_row += csr.n_rows();
                            return Ok(());
                        }
                    };
                    let lo = kept.partition_point(|&r| r < s_start);
                    let hi = kept.partition_point(|&r| r < s_end);
                    for &g_row in &kept[lo..hi] {
                        let local = (g_row - s_start) as usize;
                        let s = csr.indptr[local] as usize;
                        let e = csr.indptr[local + 1] as usize;
                        for j in s..e {
                            let c = csr.indices[j] as usize;
                            let diff = csr.data[j] as f64 - col_means[c];
                            sq_devs[c] += diff * diff;
                            col_nnz[c] += 1;
                        }
                    }
                } else {
                    for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                        let c = col as usize;
                        let diff = val as f64 - col_means[c];
                        sq_devs[c] += diff * diff;
                        col_nnz[c] += 1;
                    }
                }

                global_row += csr.n_rows();
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;

        // Add contribution from implicit zeros
        scx_sparse::finalize_implicit_zero_variance(&sq_devs, &col_nnz, &col_means, n_obs)
            .map_err(|e| e.to_string())
    }

    /// Stream all shards, apply transforms, compute masked column sums (deletion-aware).
    ///
    /// Like `streaming_col_sums()`, returns a vector of length `backed.shape().1`
    /// (physical column count). Callers must apply `apply_col_projection_to_vec()`.
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn streaming_col_sums_masked(&self) -> Result<Vec<f64>, String> {
        let n_vars = self.backed.shape().1; // physical width, intentionally
        let mut sums = vec![0.0f64; n_vars];
        let mut global_row = 0usize;

        let kept = match &self.kept_to_global {
            Some(k) => k,
            None => return self.streaming_col_sums(),
        };

        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                self.apply_transforms(&mut csr, global_row);

                let (s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                    Some(r) => r,
                    None => {
                        global_row += csr.n_rows();
                        return Ok(());
                    }
                };
                let lo = kept.partition_point(|&r| r < s_start);
                let hi = kept.partition_point(|&r| r < s_end);
                for &g_row in &kept[lo..hi] {
                    let local = (g_row - s_start) as usize;
                    let s = csr.indptr[local] as usize;
                    let e = csr.indptr[local + 1] as usize;
                    for j in s..e {
                        sums[csr.indices[j] as usize] += csr.data[j] as f64;
                    }
                }

                global_row += csr.n_rows();
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        Ok(sums)
    }

    /// Fused per-cell QC pass through the transform chain: row nnz, row sums
    /// and per-`qc_var` subset sums over the visible columns, in one scan.
    ///
    /// Lazy twin of [`crate::projected_agg::qc_row_pass`]. Transforms are
    /// applied to the **full-width** shard before projection so a prior
    /// `NormalizeTotal` divides by the denominator it was configured with; see
    /// [`Self::streaming_row_sums`].
    ///
    /// nnz is counted **after** `apply_transforms`, from the projected
    /// `indptr`. That is equivalent to counting it before: every [`Transform`]
    /// variant rewrites `csr.data` only and none prunes entries, so the
    /// sparsity pattern — and therefore the count — is unchanged. A future
    /// transform that *does* change the pattern would have to revisit this and
    /// count pre-transform. [`Self::streaming_row_nnz_and_sums`] delegates here
    /// with an empty mask, so it inherits the same reasoning.
    ///
    /// Returns global-length vectors (NOT filtered through deletion vectors).
    pub(crate) fn streaming_qc_row_pass(
        &self,
        qc_bits: &[u64],
        n_qc: usize,
    ) -> Result<crate::projected_agg::QcRowStats, String> {
        let cols = self.col_projection.clone();
        let n_visible = cols.as_deref().map_or(self.backed.shape().1, |c| c.len());
        crate::projected_agg::ensure_qc_pass_args(n_qc, qc_bits, n_visible);
        let mut out = crate::projected_agg::QcRowStats::zeroed(self.backed.shape().0, n_qc);
        let mut global_row = 0usize;
        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                let n_rows = csr.n_rows();
                self.apply_transforms(&mut csr, global_row);
                match cols.as_deref() {
                    Some(c) => crate::projected_agg::accumulate_qc_rows_into(
                        &scx_engine::projection::project_csr(&csr, c),
                        global_row,
                        qc_bits,
                        &mut out,
                    ),
                    None => crate::projected_agg::accumulate_qc_rows_into(
                        &csr, global_row, qc_bits, &mut out,
                    ),
                }
                global_row += n_rows;
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;
        crate::projected_agg::ensure_full_row_coverage(global_row, self.backed.shape().0)
            .map_err(|e| e.to_string())?;
        Ok(out)
    }

    // --- Visible-space aggregation API ---
    //
    // The `*_raw` family is *the* aggregation surface for this type, mirroring
    // `ScxBackedSparseDataset`'s. Every one of them speaks the **visible** axis:
    // column vectors come back at `shape_val.1`, row vectors are global-length
    // but summed only over projected columns (apply `filter_row_results` for the
    // visible rows). Python entry points and accel ops should call these, never
    // the `streaming_*` kernels underneath — that indirection is what keeps a
    // caller from silently picking the physical-width variant (§9.18).

    /// Per-row sums over the visible columns, through the transform chain.
    /// Global-length; caller applies [`Self::filter_row_results`].
    ///
    /// Twin of `ScxBackedSparseDataset::row_sums_raw`.
    pub(crate) fn row_sums_raw(&self) -> Result<Vec<f64>, String> {
        self.streaming_row_sums()
    }

    /// Per-row nnz over the visible columns. Global-length; caller applies
    /// [`Self::filter_row_results`].
    ///
    /// Reads the underlying backed reader rather than streaming through the
    /// transform chain: nnz counts stored entries, and every [`Transform`]
    /// rewrites values only. Twin of `ScxBackedSparseDataset::row_nnz_raw`.
    pub(crate) fn row_nnz_raw(&self) -> Result<Vec<i64>, String> {
        match &self.col_projection {
            Some(cols) => crate::projected_agg::row_nnz_projected(&self.backed, cols)
                .map_err(|e| e.to_string()),
            None => self.backed.row_nnz().map_err(|e| e.to_string()),
        }
    }

    /// Fused per-row nnz + sums over the visible columns, in one scan.
    /// Global-length; caller applies [`Self::filter_row_results`].
    ///
    /// The row-axis counterpart of [`Self::col_sums_and_nnz_raw`].
    pub(crate) fn row_nnz_and_sums_raw(&self) -> Result<(Vec<i64>, Vec<f64>), String> {
        self.streaming_row_nnz_and_sums()
    }

    /// Per-column sums through the transform chain, honoring column projection
    /// and keep-mask. Length = `shape_val.1`.
    ///
    /// The visible-space wrapper over the physical-width `streaming_col_sums*`
    /// kernels, mirroring `ScxBackedSparseDataset::col_sums_raw`. Prefer
    /// [`Self::col_sums_and_nnz_raw`] when the caller also needs nnz — that is
    /// one scan instead of two.
    pub(crate) fn col_sums_raw(&self) -> Result<Vec<f64>, String> {
        let physical = if self.kept_to_global.is_some() {
            self.streaming_col_sums_masked()?
        } else {
            self.streaming_col_sums()?
        };
        Ok(self.apply_col_projection_to_vec(physical))
    }

    /// Fused per-column sums + nnz through the transform chain, honoring
    /// column projection and keep-mask. Length = `shape_val.1`.
    ///
    /// One scan producing both statistics; the column-axis counterpart of
    /// [`Self::streaming_qc_row_pass`]. NNZ counts stored entries, which the
    /// value-wise transforms leave untouched, so it matches the raw reader's
    /// `col_nnz`.
    ///
    /// The kept-row walk is inlined rather than delegating to
    /// [`scx_format_io::BackedCsrReader::col_sums_and_nnz_masked`] because the
    /// transform chain has to run on each decoded shard *before* the values are
    /// accumulated — the backed kernel reads the untransformed shard.
    pub(crate) fn col_sums_and_nnz_raw(&self) -> Result<(Vec<f64>, Vec<u32>), String> {
        let n_vars = self.backed.shape().1; // physical width, projected below
        let mut sums = vec![0.0f64; n_vars];
        let mut counts = vec![0u32; n_vars];
        let mut global_row = 0usize;

        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                let n_rows = csr.n_rows();
                self.apply_transforms(&mut csr, global_row);

                match &self.kept_to_global {
                    None => {
                        for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                            let c = col as usize;
                            sums[c] += val as f64;
                            counts[c] += 1;
                        }
                    }
                    Some(kept) => {
                        let (s_start, s_end) = match self.backed.index().shard_range(shard_idx) {
                            Some(r) => r,
                            None => {
                                global_row += n_rows;
                                return Ok(());
                            }
                        };
                        let lo = kept.partition_point(|&r| r < s_start);
                        let hi = kept.partition_point(|&r| r < s_end);
                        for &g_row in &kept[lo..hi] {
                            let local = (g_row - s_start) as usize;
                            let s = csr.indptr[local] as usize;
                            let e = csr.indptr[local + 1] as usize;
                            for j in s..e {
                                let c = csr.indices[j] as usize;
                                sums[c] += csr.data[j] as f64;
                                counts[c] += 1;
                            }
                        }
                    }
                }
                global_row += n_rows;
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;

        let sums = self.apply_col_projection_to_vec(sums);
        let counts = match &self.col_projection {
            Some(cols) => cols.iter().map(|&c| counts[c as usize]).collect(),
            None => counts,
        };
        Ok((sums, counts))
    }

    /// Materialize the full matrix with all transforms applied.
    ///
    /// Public within the crate so ScxComparisonResult can call it.
    ///
    /// Pure-Rust (returns `Result<_, String>`, no `PyErr`) so callers can run
    /// it through `detached` with the GIL released.
    pub(crate) fn materialize_csr(&self) -> Result<ScxCsr, String> {
        if self.col_projection.is_some() {
            // A projected handle assembles shard by shard: `LazyShardSource`
            // decodes one shard, applies the transforms, projects the columns
            // and drops deleted rows while it is still one shard wide, and the
            // narrow pieces are concatenated — peak = 2× the *projected* result
            // + one shard, never the whole matrix (the unprojected path below
            // holds every transformed shard before filtering). Sequential on
            // purpose: decoding in parallel would hold every shard at once.
            use scx_format_io::ShardSource;
            let source = self.as_shard_source();
            let n_vars = source.n_vars();
            let mut pieces = Vec::with_capacity(source.n_shards());
            for shard_idx in 0..source.n_shards() {
                pieces.push(source.read_shard(shard_idx).map_err(|e| e.to_string())?);
            }
            return Ok(concatenate_csr_vec(&pieces, n_vars));
        }

        let mut global_row = 0usize;
        let mut all_slices = Vec::new();

        prefetch::for_each_shard_ordered_uncached(
            &*self.backed,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> scx_format_io::Result<()> {
                // The uncached read hands back a fresh refcount-1 Arc, so this
                // unwraps for free; the transforms below need owned buffers.
                let mut csr = Arc::try_unwrap(csr).unwrap_or_else(|s| (*s).clone());
                self.apply_transforms(&mut csr, global_row);
                global_row += csr.n_rows();
                all_slices.push(csr);
                Ok(())
            },
        )
        .map_err(|e| e.to_string())?;

        // Concatenate all shards
        let full = concatenate_csr_vec(&all_slices, self.backed.n_vars());

        // Apply deletion vector filtering
        let full = if let Some(ref kept) = self.kept_to_global {
            extract_rows(&full, kept)
        } else {
            full
        };

        // Apply column projection
        Ok(self.apply_col_projection(full))
    }

    /// Materialize the full matrix as a scipy CSR, callable from other modules.
    ///
    /// Shard decode + transform runs off the GIL (`detached`); only the scipy
    /// object is built on the GIL. This is the single hot path behind every
    /// `to_memory`-based method (`multiply`, `power`, `std`, row/scalar `var`, …).
    pub(crate) fn to_memory_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let csr = detached(py, || self.materialize_csr()).map_err(PyRuntimeError::new_err)?;
        csr_to_scipy(py, csr)
    }
}
