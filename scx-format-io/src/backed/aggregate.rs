//! Native shard-by-shard aggregation over a [`BackedCsrReader`].
//!
//! A second `impl BackedCsrReader` block rather than a separate type: these are
//! reader methods, and splitting them out is about file size, not about
//! introducing a boundary. They compute row/column statistics without
//! materialising the concatenated CSR, so peak memory is the prefetch depth in
//! decoded shards plus the output vector.

use super::*;

// ---------------------------------------------------------------------------
// Aggregation operation enum (internal)
// ---------------------------------------------------------------------------

/// Internal enum for masked column aggregation dispatch.
#[derive(Clone, Copy)]
enum AggOp {
    Sum,
    Nnz,
    Max,
    Min,
}

impl BackedCsrReader {
    // --- Native shard-by-shard aggregation ---
    //
    // These methods compute statistics without materializing the full
    // concatenated CSR. Peak memory is `SCX_ACCEL_PREFETCH_DEPTH` decoded
    // shards (default 4) plus the output vector — the ordered decode-prefetch
    // pipeline keeps that many in flight so decode overlaps the reduction.
    // It was one shard before Phase 4.2, and still is whenever the pipeline
    // declines to engage (depth 1, a single shard, a one-thread rayon pool, or
    // a caller that is itself a rayon worker).
    //
    // The bound is **per call**, and the depth knob is process-global: N
    // concurrent callers hold N x depth shards. `pyscx.accel.col_*` release the
    // GIL, so that is reachable from Python threads.

    /// Compute per-row sums without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sums from each shard's
    /// CSR arrays, and concatenates the results.
    pub fn row_sums(&self) -> Result<Vec<f64>> {
        let mut all_sums = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_sums.extend(csr.row_sums());
                Ok(())
            },
        )?;
        Ok(all_sums)
    }

    /// Compute per-column sums without materializing the full matrix.
    ///
    /// Iterates shards, accumulates column sums into a single `n_vars`-length vector.
    pub fn col_sums(&self) -> Result<Vec<f64>> {
        let mut sums = vec![0.0f64; self.n_vars];
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let partial = csr.col_sums();
                for (s, p) in sums.iter_mut().zip(partial.iter()) {
                    *s += p;
                }
                Ok(())
            },
        )?;
        Ok(sums)
    }

    /// Compute per-row NNZ counts without materializing the full matrix.
    pub fn row_nnz(&self) -> Result<Vec<i64>> {
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_nnz.extend(csr.row_nnz());
                Ok(())
            },
        )?;
        Ok(all_nnz)
    }

    /// Compute per-row NNZ and sums in a single shard scan.
    ///
    /// Avoids the double I/O of calling `row_nnz()` + `row_sums()` separately.
    /// Used by `filter_cells` when both `min_genes` and `min_counts` are specified.
    pub fn row_nnz_and_sums(&self) -> Result<(Vec<i64>, Vec<f64>)> {
        let mut all_nnz = Vec::with_capacity(self.n_obs);
        let mut all_sums = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                for row in 0..csr.n_rows() {
                    let start = csr.indptr[row] as usize;
                    let end = csr.indptr[row + 1] as usize;
                    all_nnz.push((end - start) as i64);
                    all_sums.push(csr.data[start..end].iter().map(|&v| v as f64).sum());
                }
                Ok(())
            },
        )?;
        Ok((all_nnz, all_sums))
    }

    /// Compute per-column sums and per-column NNZ in a single shard scan.
    ///
    /// The column-axis twin of [`Self::row_nnz_and_sums`]: avoids the second
    /// full decode of every shard that calling `col_sums()` and `col_nnz()`
    /// separately incurs. Used by `calculate_qc_metrics`' gene axis and by
    /// `filter_genes` when both a cell and a count threshold are given.
    ///
    /// Bit-identical to the two separate calls — same per-shard visit order,
    /// same left-to-right f64 accumulation.
    pub fn col_sums_and_nnz(&self) -> Result<(Vec<f64>, Vec<u32>)> {
        let mut sums = vec![0.0f64; self.n_vars];
        let mut counts = vec![0u32; self.n_vars];
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let (partial_sums, partial_nnz) = csr.col_sums_and_nnz();
                for (s, p) in sums.iter_mut().zip(partial_sums.iter()) {
                    *s += p;
                }
                for (c, p) in counts.iter_mut().zip(partial_nnz.iter()) {
                    *c = c.saturating_add(*p);
                }
                Ok(())
            },
        )?;
        Ok((sums, counts))
    }

    /// Compute per-column NNZ counts without materializing the full matrix.
    pub fn col_nnz(&self) -> Result<Vec<u32>> {
        let mut counts = vec![0u32; self.n_vars];
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let partial = csr.col_nnz();
                for (c, p) in counts.iter_mut().zip(partial.iter()) {
                    *c = c.saturating_add(*p);
                }
                Ok(())
            },
        )?;
        Ok(counts)
    }

    /// Total NNZ across all shards.
    ///
    /// Reads shard statistics directly from the catalog — no shard decode
    /// required. Returns the sum of `stats.nnz` across every shard the
    /// reader covers (X shards when `layer_name.is_none()`, layer shards
    /// otherwise). Shards without a `stats` block contribute 0.
    pub fn total_nnz(&self) -> Result<usize> {
        let entries = if self.layer_name.is_none() {
            &self.x_sorted_entries
        } else {
            &self.sorted_entries
        };
        // `ShardEntryLite` carries `nnz` directly — no `.stats`
        // indirection per shard.
        let total: u64 = entries.iter().map(|e| e.nnz).sum();
        Ok(total as usize)
    }

    /// Largest stored value across this reader's shards, **when the catalog
    /// can prove one** — `None` when it cannot.
    ///
    /// Walks the catalog only (O(shards), no payload reads, no decode), which
    /// is the whole point: it answers "how big do the values get?" without
    /// touching the data. `ShardStats::value_max` is exact for the integer
    /// value encodings and is written as `0` for `Float32`/`Float16`, where
    /// a `u32` field cannot represent the statistic — so:
    ///
    /// * `Some(m)` with `m > 0` — every contributing shard is integer-encoded
    ///   and `m` is the true maximum.
    /// * `Some(0)` — the matrix is empty (`nnz == 0`), so `0` *is* the maximum
    ///   and the catalog proves it.
    /// * `None` — a matrix whose maximum the catalog cannot bound: the shards
    ///   are float-encoded, or they carry no stats (or, vanishingly, the `nnz`
    ///   lookup itself failed). Those are indistinguishable from here, so
    ///   callers must not report any one as the cause; `None` means *unknown*,
    ///   and a caller that needs a real answer has to stream
    ///   ([`Self::col_max`]) or ask the user.
    ///
    /// Covers X shards when this reader targets X, layer shards otherwise —
    /// the same entry set as [`Self::total_nnz`].
    pub fn catalog_int_value_max(&self) -> Option<u32> {
        // An unscoped reader over overlapping ranges has no one maximum to
        // report. This narrows the catalog by `self.modality_id()`, which there
        // is whichever modality sorted first — so it answered that modality's
        // maximum as the file's, and a *falsely small* one whenever another
        // modality holds the larger value. `None` is already this method's
        // documented "unknown", so the refusal needs no new shape: a caller
        // that needs a real number streams `col_max`.
        if self.row_addressing_ambiguous {
            return None;
        }
        let want_layer = self.layer_name.is_some();
        let modality = self.modality_id();
        // Hoisted out of the filter: otherwise every catalog entry pays a
        // `format!` allocation just to be compared against.
        let layer_prefix = self.layer_name.as_ref().map(|n| format!("{n}_shard_"));
        let max = self
            .reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                e.modality_id == modality
                    && if want_layer {
                        e.section_type == SectionType::LayerCsrShard
                            && layer_prefix
                                .as_ref()
                                .is_some_and(|prefix| e.name.starts_with(prefix))
                    } else {
                        e.section_type == SectionType::CsrShard
                    }
            })
            .filter_map(|e| e.stats.as_ref())
            .map(|s| s.value_max)
            .max()
            .unwrap_or(0);
        if max > 0 {
            return Some(max);
        }
        // `value_max == 0` is ambiguous between "float-encoded / no stats" and
        // "there are no values". `nnz` disambiguates: an empty matrix really
        // does max to 0, and saying so beats making the caller refuse.
        match self.total_nnz() {
            Ok(0) => Some(0),
            _ => None,
        }
    }

    /// Compute per-row sum of squared values without materializing the full matrix.
    ///
    /// Iterates shards in order, computes row sum-of-squares from each shard's
    /// CSR arrays, and concatenates the results. Used for scalar variance:
    /// `Var(X) = E[X²] - (E[X])²`.
    pub fn row_sum_of_squares(&self) -> Result<Vec<f64>> {
        let mut all_sq = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_sq.extend(csr.row_sum_of_squares());
                Ok(())
            },
        )?;
        Ok(all_sq)
    }

    // --- Variance ---

    /// Streaming per-row variance without materializing the full matrix.
    ///
    /// Each shard independently computes row variances (one row = one shard's row).
    pub fn row_var(&self) -> Result<Vec<f64>> {
        let mut all_var = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_var.extend(csr.row_var()?);
                Ok(())
            },
        )?;
        Ok(all_var)
    }

    /// Streaming per-column variance (population variance, ddof=0).
    ///
    /// Two-pass algorithm:
    ///   1. Compute column means via `col_sums() / n_obs`
    ///   2. Stream shards, accumulating `(x - mean)²` for stored values
    ///   3. Add zero-entry contributions: `(n_obs - col_nnz) * mean²`
    pub fn col_var(&self) -> Result<Vec<f64>> {
        let n_obs = self.n_obs;
        if n_obs == 0 {
            // Paired with the CSC kernels, which validate their tallies before
            // this short-circuit. A 0-row file holding stored entries is
            // non-canonical, but reaching that verdict here costs a full walk
            // the CSR path would otherwise skip; `finalize_implicit_zero_variance`
            // is where the count is checked, and it is never reached. Documented
            // rather than silently divergent — see docs/api/rust-format-io.md.
            return Ok(vec![0.0f64; self.n_vars]);
        }

        // Pass 1: column means
        let col_sums = self.col_sums()?;
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_obs as f64).collect();

        // Pass 2: accumulate (val - mean)² for stored entries
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                let partial = csr.col_var_partial(&col_means);
                for (s, p) in sq_devs.iter_mut().zip(partial.iter()) {
                    *s += p;
                }
                let nnz = csr.col_nnz();
                for (c, &n) in col_nnz.iter_mut().zip(nnz.iter()) {
                    *c += n as usize;
                }
                Ok(())
            },
        )?;

        // Add contribution from implicit zeros: (n_obs - col_nnz[c]) * mean[c]²
        Ok(scx_sparse::finalize_implicit_zero_variance(
            &sq_devs, &col_nnz, &col_means, n_obs,
        )?)
    }

    // --- Max / Min ---

    /// Streaming per-row max without materializing the full matrix.
    pub fn row_max(&self) -> Result<Vec<f64>> {
        let mut all_max = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_max.extend(csr.row_max()?);
                Ok(())
            },
        )?;
        Ok(all_max)
    }

    /// Streaming per-column max without materializing the full matrix.
    ///
    /// Merges per-shard column maxes. Accounts for implicit zeros:
    /// if any column has fewer stored entries than `n_obs`, the max is
    /// at least 0.0.
    pub fn col_max(&self) -> Result<Vec<f64>> {
        let mut maxes = vec![f64::NEG_INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                // Get per-column max within this shard (using n_rows of shard, not global n_obs)
                // We need the raw stored max, so we pass n_rows = shard.n_rows()
                // But we want the global implicit-zero correction at the end,
                // so we track NNZ ourselves and compute raw stored max.
                for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                    let c = col as usize;
                    let v = val as f64;
                    maxes[c] = maxes[c].max(v);
                    col_nnz[c] += 1;
                }
                Ok(())
            },
        )?;

        // Apply implicit-zero correction at the global level
        for c in 0..self.n_vars {
            if scx_sparse::implicit_zero_count(self.n_obs, col_nnz[c])? > 0 {
                if maxes[c] == f64::NEG_INFINITY {
                    maxes[c] = 0.0;
                } else {
                    maxes[c] = maxes[c].max(0.0);
                }
            }
        }
        Ok(maxes)
    }

    /// Streaming per-row min without materializing the full matrix.
    pub fn row_min(&self) -> Result<Vec<f64>> {
        let mut all_min = Vec::with_capacity(self.n_obs);
        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                all_min.extend(csr.row_min()?);
                Ok(())
            },
        )?;
        Ok(all_min)
    }

    /// Streaming per-column min without materializing the full matrix.
    pub fn col_min(&self) -> Result<Vec<f64>> {
        let mut mins = vec![f64::INFINITY; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        prefetch::for_each_shard_ordered_uncached(
            self,
            prefetch::prefetch_depth(),
            |_shard_idx, csr| -> Result<()> {
                for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                    let c = col as usize;
                    let v = val as f64;
                    mins[c] = mins[c].min(v);
                    col_nnz[c] += 1;
                }
                Ok(())
            },
        )?;

        for c in 0..self.n_vars {
            if scx_sparse::implicit_zero_count(self.n_obs, col_nnz[c])? > 0 {
                if mins[c] == f64::INFINITY {
                    mins[c] = 0.0;
                } else {
                    mins[c] = mins[c].min(0.0);
                }
            }
        }
        Ok(mins)
    }

    // --- Masked column aggregation (deletion-vector aware) ---
    //
    // These variants accept a `kept_rows` set and only aggregate values from
    // rows in that set. Used when deletion vectors are present.

    /// Column sums considering only the kept rows.
    ///
    /// Iterates shards, intersects with the kept set, and accumulates.
    pub fn col_sums_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Sum)
    }

    /// Column NNZ considering only the kept rows.
    pub fn col_nnz_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Nnz)
    }

    /// Column sums and column NNZ over the kept rows, in a single shard scan.
    ///
    /// Deletion-aware twin of [`Self::col_sums_and_nnz`]. Bit-identical to
    /// `col_sums_masked()` + `col_nnz_masked()`, which walk the same rows in
    /// the same order — this just decodes each shard once instead of twice.
    pub fn col_sums_and_nnz_masked(&self, kept_rows: &[u64]) -> Result<(Vec<f64>, Vec<u32>)> {
        // A stale file must raise, and the plan below can be empty: with no
        // kept row in any shard there is no `read_shard` left to perform the
        // freshness check a watching reader relies on, so the call would answer
        // zeros from an obsolete mapping while `shape` on the same handle
        // raises. Check up front, where it is unconditional. (This also closes
        // the pre-existing `n_kept == 0` early return below, which never read
        // either.)
        self.check_fresh()?;

        let mut sums = vec![0.0f64; self.n_vars];
        let mut counts = vec![0u32; self.n_vars];

        // Skip shards no kept row falls in: the closure below already
        // computes that (`lo == hi`), but only after paying for the decode.
        prefetch::for_each_shard_ordered_uncached_selected(
            self,
            &self.index.shards_with_kept_rows(kept_rows),
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> Result<()> {
                let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                    Some(r) => r,
                    None => return Ok(()),
                };
                // `kept_rows` is sorted by construction (see compute_kept_to_global),
                // so the shard's slice is a binary-search range.
                let lo = kept_rows.partition_point(|&r| r < s_start);
                let hi = kept_rows.partition_point(|&r| r < s_end);
                for &global_row in &kept_rows[lo..hi] {
                    let local_row = (global_row - s_start) as usize;
                    let row_start = csr.indptr[local_row] as usize;
                    let row_end = csr.indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        let c = csr.indices[j] as usize;
                        sums[c] += csr.data[j] as f64;
                        counts[c] += 1;
                    }
                }
                Ok(())
            },
        )?;
        Ok((sums, counts))
    }

    /// Column max considering only the kept rows.
    pub fn col_max_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Max)
    }

    /// Column min considering only the kept rows.
    pub fn col_min_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        self.col_aggregate_masked(kept_rows, AggOp::Min)
    }

    /// Column variance considering only the kept rows.
    pub fn col_var_masked(&self, kept_rows: &[u64]) -> Result<Vec<f64>> {
        // A stale file must raise, and the plan below can be empty: with no
        // kept row in any shard there is no `read_shard` left to perform the
        // freshness check a watching reader relies on, so the call would answer
        // zeros from an obsolete mapping while `shape` on the same handle
        // raises. Check up front, where it is unconditional. (This also closes
        // the pre-existing `n_kept == 0` early return below, which never read
        // either.)
        self.check_fresh()?;

        let n_kept = kept_rows.len();
        if n_kept == 0 {
            return Ok(vec![0.0f64; self.n_vars]);
        }

        // Pass 1: column sums over kept rows → means
        let col_sums = self.col_sums_masked(kept_rows)?;
        let col_means: Vec<f64> = col_sums.iter().map(|&s| s / n_kept as f64).collect();

        // Pass 2: accumulate (val - mean)² for stored entries in kept rows
        let mut sq_devs = vec![0.0f64; self.n_vars];
        let mut col_nnz = vec![0usize; self.n_vars];

        // See `col_sums_and_nnz_masked`: skip the shards this kept set empties.
        prefetch::for_each_shard_ordered_uncached_selected(
            self,
            &self.index.shards_with_kept_rows(kept_rows),
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> Result<()> {
                let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                    Some(r) => r,
                    None => return Ok(()),
                };

                // Binary-search to find the sub-slice of kept_rows within [s_start, s_end).
                // kept_rows is sorted by construction (see compute_kept_to_global).
                let lo = kept_rows.partition_point(|&r| r < s_start);
                let hi = kept_rows.partition_point(|&r| r < s_end);
                for &global_row in &kept_rows[lo..hi] {
                    let local_row = (global_row - s_start) as usize;
                    let row_start = csr.indptr[local_row] as usize;
                    let row_end = csr.indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        let c = csr.indices[j] as usize;
                        let diff = csr.data[j] as f64 - col_means[c];
                        sq_devs[c] += diff * diff;
                        col_nnz[c] += 1;
                    }
                }
                Ok(())
            },
        )?;

        // Add zero-entry contributions
        Ok(scx_sparse::finalize_implicit_zero_variance(
            &sq_devs, &col_nnz, &col_means, n_kept,
        )?)
    }

    /// Internal: masked column aggregation over kept rows.
    fn col_aggregate_masked(&self, kept_rows: &[u64], op: AggOp) -> Result<Vec<f64>> {
        // A stale file must raise, and the plan below can be empty: with no
        // kept row in any shard there is no `read_shard` left to perform the
        // freshness check a watching reader relies on, so the call would answer
        // zeros from an obsolete mapping while `shape` on the same handle
        // raises. Check up front, where it is unconditional. (This also closes
        // the pre-existing `n_kept == 0` early return below, which never read
        // either.)
        self.check_fresh()?;

        let n_kept = kept_rows.len();
        let mut result = match op {
            AggOp::Sum | AggOp::Nnz => vec![0.0f64; self.n_vars],
            AggOp::Max => vec![f64::NEG_INFINITY; self.n_vars],
            AggOp::Min => vec![f64::INFINITY; self.n_vars],
        };
        let mut col_nnz = vec![0usize; self.n_vars];

        // See `col_sums_and_nnz_masked`: skip the shards this kept set empties.
        prefetch::for_each_shard_ordered_uncached_selected(
            self,
            &self.index.shards_with_kept_rows(kept_rows),
            prefetch::prefetch_depth(),
            |shard_idx, csr| -> Result<()> {
                let (s_start, s_end) = match self.index.shard_range(shard_idx) {
                    Some(r) => r,
                    None => return Ok(()),
                };

                // Binary-search to find the sub-slice of kept_rows within [s_start, s_end).
                // kept_rows is sorted by construction (see compute_kept_to_global).
                let lo = kept_rows.partition_point(|&r| r < s_start);
                let hi = kept_rows.partition_point(|&r| r < s_end);
                for &global_row in &kept_rows[lo..hi] {
                    let local_row = (global_row - s_start) as usize;
                    let row_start = csr.indptr[local_row] as usize;
                    let row_end = csr.indptr[local_row + 1] as usize;
                    for j in row_start..row_end {
                        let c = csr.indices[j] as usize;
                        let v = csr.data[j] as f64;
                        match op {
                            AggOp::Sum => result[c] += v,
                            AggOp::Nnz => result[c] += 1.0,
                            AggOp::Max => result[c] = result[c].max(v),
                            AggOp::Min => result[c] = result[c].min(v),
                        }
                        col_nnz[c] += 1;
                    }
                }
                Ok(())
            },
        )?;

        // Handle implicit zeros for max/min
        match op {
            AggOp::Max => {
                for c in 0..self.n_vars {
                    if scx_sparse::implicit_zero_count(n_kept, col_nnz[c])? > 0 {
                        if result[c] == f64::NEG_INFINITY {
                            result[c] = 0.0;
                        } else {
                            result[c] = result[c].max(0.0);
                        }
                    }
                }
            }
            AggOp::Min => {
                for c in 0..self.n_vars {
                    if scx_sparse::implicit_zero_count(n_kept, col_nnz[c])? > 0 {
                        if result[c] == f64::INFINITY {
                            result[c] = 0.0;
                        } else {
                            result[c] = result[c].min(0.0);
                        }
                    }
                }
            }
            _ => {}
        }

        Ok(result)
    }

    // --- PCA statistics ---

    /// Compute column means and column sum-of-squares in a single pass.
    ///
    /// Streams through all shards once, accumulating per-column sums and
    /// sum-of-squares. If `zero_center` is true, returns the column means
    /// (sums / n_obs); otherwise returns `None` for means.
    ///
    /// Used by both CPU and GPU PCA to avoid a separate data pass for
    /// variance computation.
    ///
    /// Returns `(means, col_sum_sq)` where:
    /// - `means`: `Some(Vec<f64>)` of length `n_vars` if `zero_center`, else `None`
    /// - `col_sum_sq`: `Vec<f64>` of length `n_vars` — per-column Σ x²
    pub fn col_means_and_sum_sq(&self, zero_center: bool) -> Result<(Option<Vec<f64>>, Vec<f64>)> {
        // Guarded here rather than by `ShardSource`: this is an *inherent*
        // method, and a concrete-typed call resolves to it rather than to the
        // trait, so it walked every shard of an unscoped multimodal reader and
        // accumulated both modalities into one `n_vars`-wide vector, dividing
        // by one modality's `n_obs`. Measured on a two-modality fixture with
        // equal declared widths: means of `[12.5, 0, 0, 0, 22.5]`, a matrix
        // neither modality has, returned as `Ok`. (With *differing* declared
        // widths an unrelated shard-header check happens to fire first, which
        // is why the fixture that pins this declares them equal.)
        self.ensure_row_addressable("BackedCsrReader::col_means_and_sum_sq")?;

        let n_vars = self.n_vars;
        let n_shards = self.index.n_shards();
        let mut col_sums = vec![0.0f64; n_vars];
        let mut col_sum_sq = vec![0.0f64; n_vars];

        for shard_idx in 0..n_shards {
            // Serve from the decoded-shard cache so this pass shares the LRU
            // with the streaming SpMM passes (multi-pass out-of-core PCA reuses
            // each shard instead of re-decoding it here).
            let csr = self.read_shard_cached_arc(shard_idx)?;
            for (&col, &val) in csr.indices.iter().zip(csr.data.iter()) {
                let v = val as f64;
                col_sums[col as usize] += v;
                col_sum_sq[col as usize] += v * v;
            }
        }

        let means = if zero_center {
            let n_obs = self.n_obs as f64;
            Some(col_sums.iter().map(|s| s / n_obs).collect())
        } else {
            None
        };

        Ok((means, col_sum_sq))
    }
}
