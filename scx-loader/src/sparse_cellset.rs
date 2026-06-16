//! Native sparse cell-set loader.
//!
//! Consumes state3's role-tagged, multi-file cell-set plans and gathers them as
//! sparse CSR through the [`crate::plan_engine::PrefetchEngine`], emitting the
//! SCX-DATA-LOADER §4.4 batch contract. Each plan item is **one batch** of cell
//! sets, delimited by `set_offsets`; each set's rows are gathered via one
//! `read_rows_with` per reader (single-file fast path) into per-set CSR builders.
//!
//! Output is **raw-local CSR by default** — state3 applies its `local→global`
//! remap, coalesce, clip, and downsampling in Python (downsampling has no Rust
//! primitive). An **optional** per-file `local→global` remap (off by default)
//! can emit global-vocab CSR for callers that want it (§4.3 item 3).

use std::collections::HashMap;
use std::sync::Arc;

use scx_format_io::ScxReader;

use crate::error::{LoaderError, Result};
use crate::plan_engine::PrefetchEngine;

/// One batch of cell sets to gather. Rows are flat across all sets in the
/// batch; `set_offsets` (length `n_sets + 1`) delimits each set's row range.
/// On the common path every row of a set shares a `file_id`.
#[derive(Clone, Debug)]
pub struct SparseCellSetPlan {
    pub file_ids: Vec<u32>,
    pub rows: Vec<u64>,
    pub role_tags: Vec<i32>,
    pub set_offsets: Vec<i64>,
}

/// Gathered sparse batch — the §4.4 contract. One output row per plan row, in
/// plan order; `set_offsets` carries over from the plan.
#[derive(Clone, Debug)]
pub struct SparseCellSetBatch {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub data: Vec<f32>,
    pub shape: (usize, usize),
    pub cell_indices: Vec<u64>,
    pub file_ids: Vec<u32>,
    pub set_offsets: Vec<i64>,
    pub role_tags: Vec<i32>,
}

/// Drives the prefetch engine to gather sparse cell-set batches.
pub struct SparseCellSetLoader {
    engine: Arc<PrefetchEngine>,
    /// Optional per-`file_id` `local→global` table (`-1` = gene absent). When
    /// set, gathered indices are remapped into the global vocab; otherwise
    /// indices are raw-local (the default — state3 remaps in Python).
    remap: Option<Vec<Vec<i32>>>,
    normalize: bool,
    log1p: bool,
    target_sum: f64,
    /// CSR column count: max per-file `n_vars` (raw-local) or global vocab size
    /// (remapped). Raw-local indices are file-local — consumers disambiguate via
    /// `file_ids`; `n_cols` is a nominal upper bound.
    n_cols: usize,
}

impl SparseCellSetLoader {
    /// Build a loader over `scx_readers` (one per `file_id`, in slice order),
    /// sharing one decoded-shard budget. `remap`/`n_global_genes` enable global
    /// remap (`n_global_genes` falls back to the tables' max+1 when `None`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scx_readers: Vec<ScxReader>,
        cache_shards: usize,
        bytes_budget: usize,
        lookahead: usize,
        remap: Option<Vec<Vec<i32>>>,
        n_global_genes: Option<usize>,
        normalize: bool,
        log1p: bool,
        target_sum: f64,
    ) -> Result<Arc<Self>> {
        if let Some(tables) = &remap {
            if tables.len() != scx_readers.len() {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "remap has {} tables but there are {} files",
                        tables.len(),
                        scx_readers.len()
                    ),
                });
            }
        }
        let n_cols = match (&remap, n_global_genes) {
            (Some(_), Some(g)) => g,
            (Some(tables), None) => tables
                .iter()
                .flat_map(|t| t.iter().copied())
                .filter(|&g| g >= 0)
                .max()
                .map(|g| g as usize + 1)
                .unwrap_or(0),
            (None, _) => scx_readers
                .iter()
                .map(|r| r.n_vars() as usize)
                .max()
                .unwrap_or(0),
        };
        let engine =
            PrefetchEngine::from_scx_readers(scx_readers, cache_shards, bytes_budget, lookahead);
        Ok(Arc::new(SparseCellSetLoader {
            engine,
            remap,
            normalize,
            log1p,
            target_sum,
            n_cols,
        }))
    }

    /// Number of readers (`file_id` range).
    pub fn n_files(&self) -> usize {
        self.engine.n_readers()
    }

    /// CSR column count of emitted batches.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Stream `plans` (each one batch) into gathered §4.4 batches, pipelining
    /// shard prefetch via the engine. Boxed so a PyO3 wrapper can hold it
    /// (the engine iterator is generic over closures).
    pub fn iter_with_plans<I>(
        self: Arc<Self>,
        plans: I,
        lookahead: usize,
    ) -> Box<dyn Iterator<Item = Result<SparseCellSetBatch>> + Send + Sync>
    where
        I: Iterator<Item = Result<SparseCellSetPlan>> + Send + 'static,
    {
        let engine = Arc::clone(&self.engine);
        let loader = Arc::clone(&self);
        let iter = engine.iter_with_plans(
            plans,
            lookahead,
            |plan: &SparseCellSetPlan| {
                plan.file_ids
                    .iter()
                    .copied()
                    .zip(plan.rows.iter().copied())
                    .collect()
            },
            move |eng: &PrefetchEngine, plan: &SparseCellSetPlan| loader.gather(eng, plan),
        );
        Box::new(iter)
    }

    /// Gather one batch of cell sets into the §4.4 contract. The `process`
    /// callback for the engine (shards already warmed).
    pub fn gather(
        &self,
        engine: &PrefetchEngine,
        plan: &SparseCellSetPlan,
    ) -> Result<SparseCellSetBatch> {
        let total_rows = plan.rows.len();
        if plan.file_ids.len() != total_rows || plan.role_tags.len() != total_rows {
            return Err(LoaderError::ConfigError {
                reason: "plan file_ids/rows/role_tags length mismatch".into(),
            });
        }

        // Plans arrive straight from (untrusted) Python. Validate structure up
        // front so a malformed plan returns a clean error instead of panicking
        // on an unchecked slice/index deep in the gather (mirrors the sibling
        // `IndexPlanLoader`, which validates row indices before reading).
        let n_files = self.n_files();
        for &fid in &plan.file_ids {
            if fid as usize >= n_files {
                return Err(LoaderError::ConfigError {
                    reason: format!("plan file_id {fid} out of range (n_files={n_files})"),
                });
            }
        }
        // `set_offsets` must be non-decreasing and bounded by `total_rows`, so
        // every `[lo..hi]` slice below is in range.
        let mut prev: i64 = 0;
        for (k, &off) in plan.set_offsets.iter().enumerate() {
            if off < 0 || off as usize > total_rows {
                return Err(LoaderError::ConfigError {
                    reason: format!("plan set_offsets[{k}]={off} out of range [0, {total_rows}]"),
                });
            }
            if k > 0 && off < prev {
                return Err(LoaderError::ConfigError {
                    reason: format!(
                        "plan set_offsets must be non-decreasing (set_offsets[{k}]={off} < {prev})"
                    ),
                });
            }
            prev = off;
        }
        // Row indices must be in range for their file, surfaced as `IndexError`
        // (consistent with `Experiment.gather_rows_sparse` and the pair loader).
        for (&fid, &row) in plan.file_ids.iter().zip(plan.rows.iter()) {
            let n_obs = engine.reader(fid).n_obs();
            if row as usize >= n_obs {
                return Err(LoaderError::IndexOutOfRange { idx: row, n_obs });
            }
        }

        let n_sets = plan.set_offsets.len().saturating_sub(1);

        let mut indptr: Vec<i64> = Vec::with_capacity(total_rows + 1);
        indptr.push(0);
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        let mut cell_indices: Vec<u64> = Vec::with_capacity(total_rows);
        let mut out_file_ids: Vec<u32> = Vec::with_capacity(total_rows);
        let mut role_tags: Vec<i32> = Vec::with_capacity(total_rows);

        for s in 0..n_sets {
            let lo = plan.set_offsets[s] as usize;
            let hi = plan.set_offsets[s + 1] as usize;
            if hi <= lo {
                continue; // empty set — boundary only, no rows
            }
            let set_fids = &plan.file_ids[lo..hi];
            let set_rows = &plan.rows[lo..hi];
            let n = hi - lo;

            // Per-row gathered (and transformed) CSR, placed in set order.
            let mut per_row: Vec<Option<(Vec<i32>, Vec<f32>)>> = (0..n).map(|_| None).collect();

            let uniform = set_fids.iter().all(|&f| f == set_fids[0]);
            if uniform {
                // --- single-file fast path (the only current configuration) ---
                let fid = set_fids[0];
                let reader = engine.reader(fid);
                reader
                    .read_rows_with(set_rows, |orig_pos, idx, dat| {
                        per_row[orig_pos] = Some(self.transform_row(fid, idx, dat));
                        Ok(())
                    })
                    .map_err(LoaderError::FormatError)?;
            } else {
                // --- cross-file insurance path (requires global remap) ---
                if self.remap.is_none() {
                    return Err(LoaderError::ConfigError {
                        reason: "cross-file cell set requires global-vocab remap tables \
                                 (raw-local indices from different files are not comparable)"
                            .into(),
                    });
                }
                let mut by_file: HashMap<u32, Vec<(usize, u64)>> = HashMap::new();
                for (j, (&f, &r)) in set_fids.iter().zip(set_rows.iter()).enumerate() {
                    by_file.entry(f).or_default().push((j, r));
                }
                for (f, items) in by_file {
                    let reader = engine.reader(f);
                    let rs: Vec<u64> = items.iter().map(|&(_, r)| r).collect();
                    reader
                        .read_rows_with(&rs, |orig_pos, idx, dat| {
                            let within_set_pos = items[orig_pos].0;
                            per_row[within_set_pos] = Some(self.transform_row(f, idx, dat));
                            Ok(())
                        })
                        .map_err(LoaderError::FormatError)?;
                }
            }

            for (j, row) in per_row.into_iter().enumerate() {
                let (ridx, rdat) = row.ok_or_else(|| {
                    LoaderError::ChannelError(format!(
                        "row {} (set {s}) was not scattered",
                        set_rows[j]
                    ))
                })?;
                indices.extend_from_slice(&ridx);
                data.extend_from_slice(&rdat);
                indptr.push(indices.len() as i64);
                cell_indices.push(set_rows[j]);
                out_file_ids.push(set_fids[j]);
                role_tags.push(plan.role_tags[lo + j]);
            }
        }

        Ok(SparseCellSetBatch {
            indptr,
            indices,
            data,
            shape: (cell_indices.len(), self.n_cols),
            cell_indices,
            file_ids: out_file_ids,
            set_offsets: plan.set_offsets.clone(),
            role_tags,
        })
    }

    /// Apply the optional remap + value-only transforms to one gathered row.
    fn transform_row(&self, fid: u32, idx: &[i32], dat: &[f32]) -> (Vec<i32>, Vec<f32>) {
        let (out_idx, mut out_dat) = match &self.remap {
            Some(tables) => remap_row(idx, dat, &tables[fid as usize]),
            None => (idx.to_vec(), dat.to_vec()),
        };
        if self.normalize || self.log1p {
            apply_sparse_transforms(&mut out_dat, self.normalize, self.log1p, self.target_sum);
        }
        (out_idx, out_dat)
    }
}

/// Map local gene ids to global via `local_to_global` (`-1` = drop), then sort
/// by global id and coalesce duplicates by summing — matching state3's
/// `local_to_global` + `_coalesce_gene_counts` (`dataset.py:381-391`), so the
/// output CSR row stays canonical (sorted, unique).
fn remap_row(indices: &[i32], data: &[f32], local_to_global: &[i32]) -> (Vec<i32>, Vec<f32>) {
    let mut pairs: Vec<(i32, f32)> = Vec::with_capacity(indices.len());
    for (&col, &val) in indices.iter().zip(data.iter()) {
        let g = if col >= 0 {
            local_to_global.get(col as usize).copied().unwrap_or(-1)
        } else {
            -1
        };
        if g >= 0 {
            pairs.push((g, val));
        }
    }
    pairs.sort_by_key(|&(g, _)| g);
    let mut out_idx: Vec<i32> = Vec::with_capacity(pairs.len());
    let mut out_dat: Vec<f32> = Vec::with_capacity(pairs.len());
    for (g, v) in pairs {
        if out_idx.last() == Some(&g) {
            *out_dat.last_mut().unwrap() += v;
        } else {
            out_idx.push(g);
            out_dat.push(v);
        }
    }
    (out_idx, out_dat)
}

/// Value-only, zero-preserving sparse transforms on a row's `data`, delegating
/// to the canonical dense helpers so the result is **bit-identical** to the
/// dense path on the stored nonzeros (SCX-DATA-LOADER §0). The dense transforms
/// are zero-preserving (`0 * factor = 0`; `ln(1+0) = 0`) and `normalize` derives
/// its scale from the row sum — which, over a sparse row's nonzeros, equals the
/// full row sum since the absent zeros contribute nothing. `normalize` rounds
/// `(v as f64 * factor) as f32` per element (not an f32 scale multiply), exactly
/// as [`normalize_dense_row`]; `log1p` is `ln_1p`, exactly as [`log1p_dense_row`].
fn apply_sparse_transforms(data: &mut [f32], normalize: bool, log1p: bool, target_sum: f64) {
    if normalize {
        crate::normalize::normalize_dense_row(data, target_sum);
    }
    if log1p {
        crate::normalize::log1p_dense_row(data);
    }
}

#[cfg(test)]
#[path = "sparse_cellset_tests.rs"]
mod tests;
