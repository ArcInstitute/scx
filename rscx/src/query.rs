// Phase C: Query Pipeline — R bindings for scx-engine::QueryPipeline
//
// Wraps scx-engine::QueryPipeline and QueryResult for R.
// Uses the Option<T>::take() pattern to handle Rust move semantics
// in R (which has no ownership model). Each builder method takes
// &mut self, .take()s the inner pipeline, applies the operation,
// and returns a new RQueryPipeline. After collect() the pipeline
// is consumed and further calls return an error.

use arrow::array::RecordBatch;
use std::path::PathBuf;

use extendr_api::prelude::*;
use scx_engine::pipeline::{QueryPipeline, QueryResult};
use scx_engine::GroupShardHandle;

// ---------------------------------------------------------------------------
// RQueryPipeline
// ---------------------------------------------------------------------------

/// R query pipeline — pipe-friendly with |>.
///
/// Construct via `ScxExperiment$query()`. Call `$filter_obs()`,
/// `$select_genes()`, `$with_normalize()`, `$with_log1p()`, `$limit()`,
/// then `$collect()` to execute.
///
/// After `$collect()` the pipeline is consumed and further calls will
/// raise an error.
#[extendr]
pub struct RQueryPipeline {
    inner: Option<QueryPipeline>,
}

impl RQueryPipeline {
    /// Create from a file path (called by ScxExperiment$query()).
    pub fn from_path(path: &str) -> Result<Self> {
        let pipeline = QueryPipeline::open(path).map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            inner: Some(pipeline),
        })
    }

    /// Take the inner pipeline, returning an error if already consumed.
    fn take_inner(&mut self) -> Result<QueryPipeline> {
        self.inner
            .take()
            .ok_or_else(|| Error::Other("Pipeline already consumed by collect()".into()))
    }
}

#[extendr]
impl RQueryPipeline {
    /// Filter observations by predicate expression.
    /// Returns a new RQueryPipeline for pipe chaining.
    ///
    /// Note: takes &mut self because extendr doesn't support consuming-self.
    /// Internally .take()s the pipeline and wraps the result in a new object.
    ///
    /// Example predicates:
    ///   "cell_type == 'T cell'"
    ///   "tissue == 'lung' and cell_type == 'B cell'"
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err`: a fallible
    /// `#[extendr]` method would otherwise `unwrap()`-panic in extendr 0.8.0,
    /// masking the real message behind "User function panicked". See B3/B7.
    fn filter_obs(&mut self, expr: &str) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryPipeline> {
            let p = self.take_inner()?;
            let p = p
                .filter_obs(expr)
                .map_err(|e| Error::Other(e.to_string()))?;
            Ok(RQueryPipeline { inner: Some(p) })
        })())
    }

    /// Filter variables by predicate expression.
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn filter_var(&mut self, expr: &str) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryPipeline> {
            let p = self.take_inner()?;
            let p = p
                .filter_var(expr)
                .map_err(|e| Error::Other(e.to_string()))?;
            Ok(RQueryPipeline { inner: Some(p) })
        })())
    }

    /// Select specific gene indices for projection.
    /// Note: indices arrive as 0-based i32 (the R-facing `select_genes()` wrapper
    /// takes 1-based indices and subtracts 1; the extendr wrapper coerces to
    /// integer), converted to u32 internally. Negative indices raise an error.
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn select_genes(&mut self, indices: Vec<i32>) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryPipeline> {
            let p = self.take_inner()?;
            let u32_indices: Vec<u32> = indices
                .into_iter()
                .map(|i| {
                    if i < 0 {
                        Err(Error::Other(format!("negative gene index: {}", i)))
                    } else {
                        Ok(i as u32)
                    }
                })
                .collect::<Result<Vec<u32>>>()?;
            let p = p.select_genes(u32_indices);
            Ok(RQueryPipeline { inner: Some(p) })
        })())
    }

    /// Enable total-count normalization.
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn with_normalize(&mut self, target_sum: f64) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryPipeline> {
            let p = self.take_inner()?;
            let p = p.with_normalize(target_sum);
            Ok(RQueryPipeline { inner: Some(p) })
        })())
    }

    /// Enable log1p transformation.
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn with_log1p(&mut self) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryPipeline> {
            let p = self.take_inner()?;
            let p = p.with_log1p();
            Ok(RQueryPipeline { inner: Some(p) })
        })())
    }

    /// Limit the number of returned cells.
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn limit(&mut self, n: i32) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryPipeline> {
            let p = self.take_inner()?;
            if n < 0 {
                return Err(Error::Other(format!("negative limit: {}", n)));
            }
            let p = p.limit(n as usize);
            Ok(RQueryPipeline { inner: Some(p) })
        })())
    }

    /// Execute the pipeline and return an RQueryResult.
    /// The pipeline is consumed — further calls will error.
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn collect(&mut self) -> Robj {
        crate::util::throw_on_err((|| -> Result<RQueryResult> {
            let p = self.take_inner()?;
            let result = p.collect().map_err(|e| Error::Other(e.to_string()))?;
            Ok(RQueryResult::from_result(result))
        })())
    }

    /// Count matching cells without decoding the matrix — runs only the
    /// plan + obs-mask half (no CSR shard payload is decoded), mirroring
    /// pyscx's `query().count()`. Returns the true Level-2 match count (any
    /// `limit()` is intentionally *not* applied) and does **not** consume the
    /// pipeline. Returns f64 (R has no i64) to stay safe past 2^31 cells.
    ///
    /// Returns `Robj` and throws via `throw_on_err` (see `filter_obs`).
    fn count(&self) -> Robj {
        crate::util::throw_on_err((|| -> Result<Robj> {
            let p = self
                .inner
                .as_ref()
                .ok_or_else(|| Error::Other("pipeline already consumed".into()))?;
            let c = p.count().map_err(|e| Error::Other(e.to_string()))?;
            Ok(Robj::from(c.matched_rows as f64))
        })())
    }
}

// ---------------------------------------------------------------------------
// RQueryResult
// ---------------------------------------------------------------------------

/// Result from executing a query pipeline via `$collect()`.
///
/// Provides `$to_dgcmatrix()` for sparse matrix conversion, plus
/// metadata getters that remain accessible after conversion.
#[extendr]
pub struct RQueryResult {
    result: Option<QueryResult>,
    cached_obs: RecordBatch,
    cached_var: RecordBatch,
    cached_n_obs: usize,
    cached_n_vars: usize,
    cached_nnz: usize,
    cached_skipped_shards: usize,
    cached_total_shards: usize,
}

impl RQueryResult {
    /// Construct from a QueryResult, caching metadata and dimension values.
    pub fn from_result(r: QueryResult) -> Self {
        Self {
            cached_obs: r.obs.clone(),
            cached_var: r.var.clone(),
            cached_n_obs: r.x.n_rows(),
            cached_n_vars: r.x.n_cols(),
            cached_nnz: r.x.nnz(),
            cached_skipped_shards: r.skipped_shards,
            cached_total_shards: r.total_shards,
            result: Some(r),
        }
    }

    /// Take the inner result, returning an error if already consumed.
    fn take_result(&mut self) -> Result<QueryResult> {
        self.result
            .take()
            .ok_or_else(|| Error::Other("QueryResult already consumed".into()))
    }

    /// Peek the decode-loss `value_max` without consuming the result, so the
    /// guard can fire *before* `take_result()`. That keeps a tripped guard
    /// non-destructive: the caller can retry the same object with
    /// `allow_lossy = TRUE`. Returns 0 once consumed (the guard then passes and
    /// `take_result()` surfaces the "already consumed" error).
    fn peek_max_value(&self) -> u32 {
        self.result.as_ref().map(|r| r.max_value).unwrap_or(0)
    }
}

#[allow(clippy::wrong_self_convention)]
#[extendr]
impl RQueryResult {
    /// Convert to a dgCMatrix (Matrix package sparse matrix).
    /// Consumes the inner data — further to_dgcmatrix() calls will error.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3/B7).
    fn to_dgcmatrix(&mut self, allow_lossy: Option<bool>) -> Robj {
        // Guard before consuming so a tripped guard leaves the object reusable.
        let res =
            crate::guard::guard_decode_loss(self.peek_max_value(), allow_lossy.unwrap_or(false))
                .and_then(|()| self.take_result())
                .and_then(|r| crate::interop::csr_to_dgcmatrix(&r.x));
        crate::util::throw_on_err(res)
    }

    /// Convert to a Seurat v5 object.
    /// Consumes the inner data — requires Seurat >= 5.0.0.
    ///
    /// Returns `Robj` (not `Result`) and throws a clean R error via
    /// `throw_on_err` on failure: a fallible `#[extendr]` method would
    /// otherwise `unwrap()`-panic in extendr 0.8.0, masking the real message
    /// (missing-`Seurat` etc.) behind "User function panicked". See B7.
    fn to_seurat(&mut self, allow_lossy: Option<bool>) -> Robj {
        // Guard before consuming so a tripped guard leaves the object reusable.
        let res =
            crate::guard::guard_decode_loss(self.peek_max_value(), allow_lossy.unwrap_or(false))
                .and_then(|()| self.take_result())
                .and_then(|r| crate::interop::to_seurat_v5(&r));
        crate::util::throw_on_err(res)
    }

    /// Convert to a SingleCellExperiment object.
    /// Consumes the inner data — requires SingleCellExperiment package.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see
    /// `to_seurat` above and B7).
    fn to_sce(&mut self, allow_lossy: Option<bool>) -> Robj {
        // Guard before consuming so a tripped guard leaves the object reusable.
        let res =
            crate::guard::guard_decode_loss(self.peek_max_value(), allow_lossy.unwrap_or(false))
                .and_then(|()| self.take_result())
                .and_then(|r| crate::interop::to_sce(&r));
        crate::util::throw_on_err(res)
    }

    /// Read obs metadata as an R data.frame from the query result.
    /// Accessible even after to_dgcmatrix()/to_seurat()/to_sce() consume the matrix data.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3/B7).
    fn obs(&self) -> Robj {
        crate::util::throw_on_err(crate::interop::record_batch_to_dataframe(&self.cached_obs))
    }

    /// Read var metadata as an R data.frame from the query result.
    /// Accessible even after to_dgcmatrix()/to_seurat()/to_sce() consume the matrix data.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3/B7).
    fn var(&self) -> Robj {
        crate::util::throw_on_err(crate::interop::record_batch_to_dataframe(&self.cached_var))
    }

    /// Use f64 for n_obs/n_vars/nnz to avoid i32 overflow on large datasets.
    /// R has no unsigned or i64 integer type; f64 safely represents up to 2^53.
    fn n_obs(&self) -> Robj {
        Robj::from(self.cached_n_obs as f64)
    }
    fn n_vars(&self) -> Robj {
        Robj::from(self.cached_n_vars as f64)
    }
    fn nnz(&self) -> Robj {
        Robj::from(self.cached_nnz as f64)
    }
    fn skipped_shards(&self) -> i32 {
        self.cached_skipped_shards as i32
    }
    fn total_shards(&self) -> i32 {
        self.cached_total_shards as i32
    }
}

// ---------------------------------------------------------------------------
// RGroupShardHandle (F2 grouped reads)
// ---------------------------------------------------------------------------

/// One non-reference shard's grouped contents (F2), with deferred I/O. Returned
/// in a list by `ScxExperiment$iter_group_shards()`. Re-opens the file per read
/// (like `query()`); grouped-read parity, not the pyscx shared-pipeline path.
#[extendr]
pub struct RGroupShardHandle {
    path: PathBuf,
    handle: GroupShardHandle,
}

impl RGroupShardHandle {
    /// Construct from a path + engine handle (kept off the `#[extendr]` surface).
    pub fn new(path: PathBuf, handle: GroupShardHandle) -> Self {
        Self { path, handle }
    }

    fn open_pipeline(&self) -> Result<QueryPipeline> {
        QueryPipeline::open(&self.path).map_err(|e| Error::Other(e.to_string()))
    }

    /// Fallible body of `read_group` (one label within this shard).
    fn read_group_impl(&self, label: &str) -> Result<RQueryResult> {
        let (start, stop) = self.handle.range(label).ok_or_else(|| {
            Error::Other(format!(
                "label '{label}' is not in shard {}",
                self.handle.shard_index
            ))
        })?;
        let qr = self
            .open_pipeline()?
            .read_row_range(start, stop)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(RQueryResult::from_result(qr))
    }

    /// Fallible body of `to_query_result` (the whole shard's rows).
    fn to_query_result_impl(&self) -> Result<RQueryResult> {
        let qr = self
            .open_pipeline()?
            .read_row_range(self.handle.global_start, self.handle.global_stop)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(RQueryResult::from_result(qr))
    }

    /// Fallible body of `groups` (a data.frame of label / local start / stop).
    fn groups_impl(&self) -> Result<Robj> {
        let labels: Vec<String> = self
            .handle
            .groups
            .iter()
            .map(|(l, _, _)| l.clone())
            .collect();
        let starts: Vec<f64> = self
            .handle
            .groups
            .iter()
            .map(|(_, s, _)| *s as f64)
            .collect();
        let stops: Vec<f64> = self
            .handle
            .groups
            .iter()
            .map(|(_, _, e)| *e as f64)
            .collect();
        let n = labels.len() as i32;
        let list = List::from_pairs(vec![
            ("label", Robj::from(labels)),
            ("start", Robj::from(starts)),
            ("stop", Robj::from(stops)),
        ]);
        let robj: Robj = list.into();
        R!("{ x <- {{robj}}; class(x) <- 'data.frame'; attr(x, 'row.names') <- seq_len({{n}}); x }")
            .map_err(|e| Error::Other(format!("data.frame construction failed: {}", e)))
    }
}

#[extendr]
impl RGroupShardHandle {
    /// This shard's index in the grouped layout.
    fn shard_index(&self) -> i32 {
        self.handle.shard_index as i32
    }

    /// First global output row (inclusive). f64 to avoid i32 overflow.
    fn global_start(&self) -> Robj {
        Robj::from(self.handle.global_start as f64)
    }

    /// One past the last global output row (exclusive). f64.
    fn global_stop(&self) -> Robj {
        Robj::from(self.handle.global_stop as f64)
    }

    /// Labels present in this shard (character vector).
    fn labels(&self) -> Vec<String> {
        self.handle
            .groups
            .iter()
            .map(|(l, _, _)| l.clone())
            .collect()
    }

    /// Per-label shard-local `[start, stop)` ranges as a data.frame
    /// (`label`, `start`, `stop`). Returns `Robj` + `throw_on_err`.
    fn groups(&self) -> Robj {
        crate::util::throw_on_err(self.groups_impl())
    }

    /// Read just the cells of `label` within this shard as an `RQueryResult`.
    /// Unknown label → clean R `stop()`.
    fn read_group(&self, label: &str) -> Robj {
        crate::util::throw_on_err(self.read_group_impl(label))
    }

    /// Read this shard's full row range as an `RQueryResult`.
    fn to_query_result(&self) -> Robj {
        crate::util::throw_on_err(self.to_query_result_impl())
    }
}

// ---------------------------------------------------------------------------
// Sub-module registration for extendr
// ---------------------------------------------------------------------------

extendr_module! {
    mod query;
    impl RQueryPipeline;
    impl RQueryResult;
    impl RGroupShardHandle;
}
