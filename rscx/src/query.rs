// Phase C: Query Pipeline — R bindings for scx-engine::QueryPipeline
//
// Wraps scx-engine::QueryPipeline and QueryResult for R.
// Uses the Option<T>::take() pattern to handle Rust move semantics
// in R (which has no ownership model). Each builder method takes
// &mut self, .take()s the inner pipeline, applies the operation,
// and returns a new RQueryPipeline. After collect() the pipeline
// is consumed and further calls return an error.

use arrow::array::RecordBatch;
use extendr_api::prelude::*;
use scx_engine::pipeline::{QueryPipeline, QueryResult};

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
    /// Note: indices are i32 from R (no unsigned int), converted to u32 internally.
    /// Negative indices will raise an error.
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
}

#[allow(clippy::wrong_self_convention)]
#[extendr]
impl RQueryResult {
    /// Convert to a dgCMatrix (Matrix package sparse matrix).
    /// Consumes the inner data — further to_dgcmatrix() calls will error.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see B3/B7).
    fn to_dgcmatrix(&mut self) -> Robj {
        crate::util::throw_on_err(
            self.take_result()
                .and_then(|r| crate::interop::csr_to_dgcmatrix(&r.x)),
        )
    }

    /// Convert to a Seurat v5 object.
    /// Consumes the inner data — requires Seurat >= 5.0.0.
    ///
    /// Returns `Robj` (not `Result`) and throws a clean R error via
    /// `throw_on_err` on failure: a fallible `#[extendr]` method would
    /// otherwise `unwrap()`-panic in extendr 0.8.0, masking the real message
    /// (missing-`Seurat` etc.) behind "User function panicked". See B7.
    fn to_seurat(&mut self) -> Robj {
        crate::util::throw_on_err(
            self.take_result()
                .and_then(|r| crate::interop::to_seurat_v5(&r)),
        )
    }

    /// Convert to a SingleCellExperiment object.
    /// Consumes the inner data — requires SingleCellExperiment package.
    ///
    /// Returns `Robj` and throws a clean R error via `throw_on_err` (see
    /// `to_seurat` above and B7).
    fn to_sce(&mut self) -> Robj {
        crate::util::throw_on_err(self.take_result().and_then(|r| crate::interop::to_sce(&r)))
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
// Sub-module registration for extendr
// ---------------------------------------------------------------------------

extendr_module! {
    mod query;
    impl RQueryPipeline;
    impl RQueryResult;
}
