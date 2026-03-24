// Phase C: Query Pipeline — R bindings for scx-engine::QueryPipeline
//
// Wraps scx-engine::QueryPipeline and QueryResult for R.
// Uses the Option<T>::take() pattern to handle Rust move semantics
// in R (which has no ownership model). Each builder method takes
// &mut self, .take()s the inner pipeline, applies the operation,
// and returns a new RQueryPipeline. After collect() the pipeline
// is consumed and further calls return an error.

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
        let pipeline = QueryPipeline::open(path)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(Self {
            inner: Some(pipeline),
        })
    }

    /// Take the inner pipeline, returning an error if already consumed.
    fn take_inner(&mut self) -> Result<QueryPipeline> {
        self.inner.take().ok_or_else(|| {
            Error::Other("Pipeline already consumed by collect()".into())
        })
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
    fn filter_obs(&mut self, expr: &str) -> Result<RQueryPipeline> {
        let p = self.take_inner()?;
        let p = p
            .filter_obs(expr)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(RQueryPipeline { inner: Some(p) })
    }

    /// Filter variables by predicate expression.
    fn filter_var(&mut self, expr: &str) -> Result<RQueryPipeline> {
        let p = self.take_inner()?;
        let p = p
            .filter_var(expr)
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(RQueryPipeline { inner: Some(p) })
    }

    /// Select specific gene indices for projection.
    /// Note: indices are i32 from R (no unsigned int), converted to u32 internally.
    /// Negative indices will raise an error.
    fn select_genes(&mut self, indices: Vec<i32>) -> Result<RQueryPipeline> {
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
    }

    /// Enable total-count normalization.
    fn with_normalize(&mut self, target_sum: f64) -> Result<RQueryPipeline> {
        let p = self.take_inner()?;
        let p = p.with_normalize(target_sum);
        Ok(RQueryPipeline { inner: Some(p) })
    }

    /// Enable log1p transformation.
    fn with_log1p(&mut self) -> Result<RQueryPipeline> {
        let p = self.take_inner()?;
        let p = p.with_log1p();
        Ok(RQueryPipeline { inner: Some(p) })
    }

    /// Limit the number of returned cells.
    fn limit(&mut self, n: i32) -> Result<RQueryPipeline> {
        let p = self.take_inner()?;
        if n < 0 {
            return Err(Error::Other(format!("negative limit: {}", n)));
        }
        let p = p.limit(n as usize);
        Ok(RQueryPipeline { inner: Some(p) })
    }

    /// Execute the pipeline and return an RQueryResult.
    /// The pipeline is consumed — further calls will error.
    fn collect(&mut self) -> Result<RQueryResult> {
        let p = self.take_inner()?;
        let result = p
            .collect()
            .map_err(|e| Error::Other(e.to_string()))?;
        Ok(RQueryResult::from_result(result))
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
    cached_n_obs: usize,
    cached_n_vars: usize,
    cached_nnz: usize,
    cached_skipped_shards: usize,
    cached_total_shards: usize,
}

impl RQueryResult {
    /// Construct from a QueryResult, caching dimension values.
    pub fn from_result(r: QueryResult) -> Self {
        Self {
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
        self.result.take().ok_or_else(|| {
            Error::Other("QueryResult already consumed".into())
        })
    }
}

#[allow(clippy::wrong_self_convention)]
#[extendr]
impl RQueryResult {
    /// Convert to a dgCMatrix (Matrix package sparse matrix).
    /// Consumes the inner data — further to_dgcmatrix() calls will error.
    fn to_dgcmatrix(&mut self) -> Result<Robj> {
        let r = self.take_result()?;
        crate::interop::csr_to_dgcmatrix(&r.x)
    }

    /// Convert to a Seurat v5 object.
    /// Consumes the inner data — stub for Phase D.
    fn to_seurat(&mut self) -> Result<Robj> {
        let _r = self.take_result()?;
        Err(Error::Other(
            "to_seurat() not yet implemented (Phase D)".into(),
        ))
    }

    /// Convert to a SingleCellExperiment object.
    /// Consumes the inner data — stub for Phase D.
    fn to_sce(&mut self) -> Result<Robj> {
        let _r = self.take_result()?;
        Err(Error::Other(
            "to_sce() not yet implemented (Phase D)".into(),
        ))
    }

    /// Read obs metadata as an R data.frame from the query result.
    fn obs(&mut self) -> Result<Robj> {
        let r = self
            .result
            .as_ref()
            .ok_or_else(|| Error::Other("QueryResult already consumed".into()))?;
        crate::interop::record_batch_to_dataframe(&r.obs)
    }

    /// Read var metadata as an R data.frame from the query result.
    fn var(&mut self) -> Result<Robj> {
        let r = self
            .result
            .as_ref()
            .ok_or_else(|| Error::Other("QueryResult already consumed".into()))?;
        crate::interop::record_batch_to_dataframe(&r.var)
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
