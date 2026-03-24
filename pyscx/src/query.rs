// Query pipeline Python bindings
//
// Wraps scx-engine::QueryPipeline and QueryResult for Python.
// See Phase2-Step7.md §B1–B2 for design details.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_engine::pipeline::{QueryPipeline, QueryResult};
use scx_engine::EngineError;

use crate::anndata;

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

/// Convert an EngineError to a Python exception.
///
/// SchemaError and PredicateParseError → ValueError (validation errors).
/// All other variants → RuntimeError.
fn engine_to_pyerr(e: EngineError) -> PyErr {
    match &e {
        EngineError::SchemaError { .. } | EngineError::PredicateParseError { .. } => {
            PyValueError::new_err(e.to_string())
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

// ---------------------------------------------------------------------------
// PyQueryPipeline
// ---------------------------------------------------------------------------

/// Lazy query pipeline builder.
///
/// Construct via `PyExperiment.query()`. Call `.filter_obs()`,
/// `.select_genes()`, `.with_normalize()`, `.with_log1p()`, `.limit()`,
/// then `.collect()` to execute.
///
/// After `.collect()` the pipeline is consumed and further calls will
/// raise RuntimeError.
#[pyclass]
pub struct PyQueryPipeline {
    pipeline: Option<QueryPipeline>,
}

impl PyQueryPipeline {
    /// Create from an already-opened QueryPipeline (Rust-only).
    pub fn from_pipeline(pipeline: QueryPipeline) -> Self {
        Self {
            pipeline: Some(pipeline),
        }
    }

    /// Take the inner pipeline, returning an error if already consumed.
    fn take_pipeline(&mut self) -> PyResult<QueryPipeline> {
        self.pipeline
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("Pipeline already consumed by collect()"))
    }

    /// Store a pipeline back after a builder step.
    fn put_pipeline(&mut self, p: QueryPipeline) {
        self.pipeline = Some(p);
    }
}

#[pymethods]
impl PyQueryPipeline {
    /// Filter observations (cells) by a predicate expression.
    ///
    /// The predicate is validated against the obs schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    /// Returns self for method chaining (finding 9.5).
    ///
    /// Example:
    ///     pipeline.filter_obs("cell_type == 'T cell' and tissue == 'lung'").collect()
    fn filter_obs<'py>(slf: Bound<'py, Self>, expr: &str) -> PyResult<Bound<'py, Self>> {
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            let p = p.filter_obs(expr).map_err(engine_to_pyerr)?;
            inner.put_pipeline(p);
        }
        Ok(slf)
    }

    /// Filter variables (genes) by a predicate expression.
    ///
    /// The predicate is validated against the var schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    /// Returns self for method chaining (finding 9.5).
    fn filter_var<'py>(slf: Bound<'py, Self>, expr: &str) -> PyResult<Bound<'py, Self>> {
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            let p = p.filter_var(expr).map_err(engine_to_pyerr)?;
            inner.put_pipeline(p);
        }
        Ok(slf)
    }

    /// Select specific gene indices for projection.
    /// Returns self for method chaining (finding 9.5).
    ///
    /// Example:
    ///     pipeline.select_genes([0, 1, 2, 100, 200]).collect()
    fn select_genes(slf: Bound<'_, Self>, indices: Vec<u32>) -> PyResult<Bound<'_, Self>> {
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            let p = p.select_genes(indices);
            inner.put_pipeline(p);
        }
        Ok(slf)
    }

    /// Enable total-count normalization with the given target sum.
    /// Returns self for method chaining (finding 9.5).
    ///
    /// Example:
    ///     pipeline.with_normalize(target_sum=1e4).with_log1p().collect()
    #[pyo3(signature = (target_sum=1e4))]
    fn with_normalize(slf: Bound<'_, Self>, target_sum: f64) -> PyResult<Bound<'_, Self>> {
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            let p = p.with_normalize(target_sum);
            inner.put_pipeline(p);
        }
        Ok(slf)
    }

    /// Enable log1p transformation.
    /// Returns self for method chaining (finding 9.5).
    fn with_log1p(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            let p = p.with_log1p();
            inner.put_pipeline(p);
        }
        Ok(slf)
    }

    /// Limit the number of returned cells.
    /// Returns self for method chaining (finding 9.5).
    fn limit(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            let p = p.limit(n);
            inner.put_pipeline(p);
        }
        Ok(slf)
    }

    /// Execute the pipeline and return a `PyQueryResult`.
    ///
    /// This is where all I/O and computation occurs. The pipeline is
    /// consumed — further method calls will raise RuntimeError.
    /// The GIL is released during execution.
    fn collect(&mut self, py: Python<'_>) -> PyResult<PyQueryResult> {
        let pipeline = self.take_pipeline()?;
        // Note: pipeline is NOT put back — collect() consumes it
        let result = py
            .allow_threads(|| pipeline.collect())
            .map_err(engine_to_pyerr)?;
        Ok(PyQueryResult::from_result(result))
    }

    /// Convenience: execute the pipeline and return the number of matching cells.
    fn count(&mut self, py: Python<'_>) -> PyResult<usize> {
        let result = self.collect(py)?;
        Ok(result.cached_n_obs)
    }

    fn __repr__(&self) -> String {
        match &self.pipeline {
            Some(p) => format!("{:?}", p),
            None => "PyQueryPipeline(consumed)".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// PyQueryResult
// ---------------------------------------------------------------------------

/// Result of executing a query pipeline via `.collect()`.
///
/// Provides `.to_anndata()` and `.to_csr()` for conversion, plus
/// metadata getters that remain accessible after conversion.
#[pyclass]
pub struct PyQueryResult {
    result: Option<QueryResult>,
    // Cached values so getters work after to_anndata()/to_csr() consumes data
    cached_n_obs: usize,
    cached_n_vars: usize,
    cached_nnz: usize,
    cached_skipped_shards: usize,
    cached_total_shards: usize,
}

impl PyQueryResult {
    /// Construct from a QueryResult, caching dimension values.
    pub fn from_result(r: QueryResult) -> Self {
        let n_obs = r.x.n_rows();
        let n_vars = r.x.n_cols();
        let nnz = r.x.nnz();
        let skipped = r.skipped_shards;
        let total = r.total_shards;
        Self {
            result: Some(r),
            cached_n_obs: n_obs,
            cached_n_vars: n_vars,
            cached_nnz: nnz,
            cached_skipped_shards: skipped,
            cached_total_shards: total,
        }
    }

    /// Take the inner result, returning an error if already consumed.
    fn take_result(&mut self) -> PyResult<QueryResult> {
        self.result.take().ok_or_else(|| {
            PyRuntimeError::new_err("QueryResult already consumed by to_anndata() or to_csr()")
        })
    }
}

#[pymethods]
impl PyQueryResult {
    /// Convert the query result to an AnnData object.
    ///
    /// Uses zero-copy for the CSR matrix (moves Vec to numpy).
    /// After this call, to_anndata()/to_csr() cannot be called again,
    /// but getters (n_obs, n_vars, etc.) still work.
    #[allow(clippy::wrong_self_convention)]
    fn to_anndata<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = self.take_result()?;

        let anndata_mod = py.import("anndata")?;

        // X — zero-copy CSR → scipy
        let x = anndata::csr_to_scipy(py, result.x)?;

        // obs → pandas DataFrame
        let obs_table = anndata::record_batch_to_pyarrow(py, &result.obs)?;
        let obs_df = anndata::pyarrow_table_to_pandas(&obs_table)?;

        // var → pandas DataFrame
        let var_table = anndata::record_batch_to_pyarrow(py, &result.var)?;
        let var_df = anndata::pyarrow_table_to_pandas(&var_table)?;

        // Build AnnData
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("X", x)?;
        kwargs.set_item("obs", obs_df)?;
        kwargs.set_item("var", var_df)?;

        let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
        Ok(adata)
    }

    /// Return just the scipy CSR matrix without building full AnnData.
    ///
    /// Also consumes the inner data for zero-copy transfer.
    #[allow(clippy::wrong_self_convention)]
    fn to_csr<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = self.take_result()?;
        anndata::csr_to_scipy(py, result.x)
    }

    /// Number of observations (cells) in the result.
    #[getter]
    fn n_obs(&self) -> usize {
        self.cached_n_obs
    }

    /// Number of variables (genes) in the result.
    #[getter]
    fn n_vars(&self) -> usize {
        self.cached_n_vars
    }

    /// Total number of non-zero entries.
    #[getter]
    fn nnz(&self) -> usize {
        self.cached_nnz
    }

    /// Number of shards skipped by predicate pushdown.
    #[getter]
    fn skipped_shards(&self) -> usize {
        self.cached_skipped_shards
    }

    /// Total number of shards in the file.
    #[getter]
    fn total_shards(&self) -> usize {
        self.cached_total_shards
    }

    fn __repr__(&self) -> String {
        format!(
            "PyQueryResult(n_obs={}, n_vars={}, nnz={}, skipped={}/{}{})",
            self.cached_n_obs,
            self.cached_n_vars,
            self.cached_nnz,
            self.cached_skipped_shards,
            self.cached_total_shards,
            if self.result.is_none() {
                ", consumed"
            } else {
                ""
            },
        )
    }
}
