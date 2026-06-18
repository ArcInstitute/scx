// Query pipeline Python bindings
//
// Wraps scx-engine::QueryPipeline and QueryResult for Python.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_engine::pipeline::{QueryPipeline, QueryResult};
use scx_engine::EngineError;

use crate::convert;

/// A gene selector accepted by [`PyQueryPipeline::select_genes`]: either a
/// positional integer index or a gene name resolved against `var`.
enum GeneRef {
    Index(u32),
    Name(String),
}

/// Parse one `select_genes` element into a [`GeneRef`].
///
/// Integer-like first via `__index__`, so numpy integer scalars (from a
/// `np.array([...])` / `list(np.array(...))` selector) resolve as indices
/// instead of falling through to the name path with a confusing `KeyError`.
/// Python `str` has no `__index__`, so gene names are unaffected.
fn parse_gene_ref(obj: &Bound<'_, PyAny>) -> PyResult<GeneRef> {
    if let Ok(idx) = obj.call_method0("__index__") {
        let v: i64 = idx.extract()?;
        if !(0..=u32::MAX as i64).contains(&v) {
            return Err(PyValueError::new_err(format!(
                "gene index {v} out of range (expected 0..={})",
                u32::MAX
            )));
        }
        return Ok(GeneRef::Index(v as u32));
    }
    if let Ok(name) = obj.extract::<String>() {
        return Ok(GeneRef::Name(name));
    }
    Err(PyValueError::new_err(
        "select_genes entries must be integer indices (including numpy ints) \
         or gene-name strings",
    ))
}

// ---------------------------------------------------------------------------
// Error conversion
// ---------------------------------------------------------------------------

/// Convert an EngineError to a Python exception.
///
/// SchemaError and PredicateParseError → ValueError (validation errors).
/// All other variants → RuntimeError.
pub(crate) fn engine_to_pyerr(e: EngineError) -> PyErr {
    match &e {
        EngineError::SchemaError { .. } | EngineError::PredicateParseError { .. } => {
            PyValueError::new_err(e.to_string())
        }
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// Build an `anndata.AnnData` from a collected `QueryResult` (X + obs +
/// var). Shared by `PyQueryResult.to_anndata()` and the module-level
/// `pyscx.read_cloud(...)` helper.
pub(crate) fn query_result_to_anndata<'py>(
    py: Python<'py>,
    result: QueryResult,
) -> PyResult<Bound<'py, PyAny>> {
    let anndata_mod = py.import("anndata")?;

    // X — zero-copy CSR → scipy
    let x = convert::csr_to_scipy(py, result.x)?;

    // obs → pandas DataFrame
    let obs_table = convert::record_batch_to_pyarrow(py, &result.obs)?;
    let obs_df = convert::pyarrow_table_to_pandas(&obs_table)?;

    // var → pandas DataFrame
    let var_table = convert::record_batch_to_pyarrow(py, &result.var)?;
    let var_df = convert::pyarrow_table_to_pandas(&var_table)?;

    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("X", x)?;
    kwargs.set_item("obs", obs_df)?;
    kwargs.set_item("var", var_df)?;

    let adata = anndata_mod.call_method("AnnData", (), Some(&kwargs))?;
    Ok(adata)
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

    /// Select specific genes for projection, by integer index or gene name
    /// (or a mix). The output columns follow the requested order, matching
    /// `adata[:, gene_list]`; duplicates collapse to first occurrence. Names
    /// are resolved against `var` (raises `KeyError` if not found).
    /// Returns self for method chaining (finding 9.5).
    ///
    /// Example:
    ///     pipeline.select_genes([0, 1, 2, 100, 200]).collect()
    ///     pipeline.select_genes(["MS4A1", "CD79A", "CD3D"]).collect()
    fn select_genes<'py>(
        slf: Bound<'py, Self>,
        genes: Vec<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        // Parse each selector (int index — incl. numpy ints — or gene name).
        let genes: Vec<GeneRef> = genes
            .iter()
            .map(parse_gene_ref)
            .collect::<PyResult<Vec<_>>>()?;
        {
            let mut inner = slf.borrow_mut();
            let p = inner.take_pipeline()?;
            // Read var only when at least one name needs resolving.
            let var_batch = if genes.iter().any(|g| matches!(g, GeneRef::Name(_))) {
                Some(p.reader().read_var().map_err(engine_to_pyerr)?)
            } else {
                None
            };
            let mut indices: Vec<u32> = Vec::with_capacity(genes.len());
            for g in genes {
                match g {
                    GeneRef::Index(i) => indices.push(i),
                    GeneRef::Name(name) => {
                        let vb = var_batch.as_ref().expect("var read when names present");
                        match crate::experiment::lookup_gene_in_batch(vb, &name) {
                            Some(idx) => indices.push(idx),
                            None => {
                                // p is untouched (select_genes not yet called);
                                // restore it so the pipeline stays usable.
                                inner.put_pipeline(p);
                                return Err(PyKeyError::new_err(format!(
                                    "gene name '{name}' not found in var index"
                                )));
                            }
                        }
                    }
                }
            }
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
        let result = py.detach(|| pipeline.collect()).map_err(engine_to_pyerr)?;
        Ok(PyQueryResult::from_result(result))
    }

    /// Number of matching cells, **without decoding X** and ignoring `limit`.
    ///
    /// Runs only the planning + masking half of the engine; on the row-set fast
    /// path (indexed predicates) this needs no obs/X shard decode. Borrows the
    /// pipeline — it stays usable for a later `collect()`. The GIL is released
    /// during execution.
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Pipeline already consumed by collect()"))?;
        let result = py.detach(|| pipeline.count()).map_err(engine_to_pyerr)?;
        Ok(result.matched_rows)
    }

    /// Whether any cell matches, **without decoding X** and ignoring `limit`.
    ///
    /// On the row-set fast path this needs no obs/X shard decode; with residual
    /// (non-indexed) predicates it decodes only the narrowed obs shards. Borrows
    /// the pipeline — it stays usable for a later `collect()`.
    fn exists(&self, py: Python<'_>) -> PyResult<bool> {
        let pipeline = self
            .pipeline
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Pipeline already consumed by collect()"))?;
        py.detach(|| pipeline.exists()).map_err(engine_to_pyerr)
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
        query_result_to_anndata(py, result)
    }

    /// Return just the scipy CSR matrix without building full AnnData.
    ///
    /// Also consumes the inner data for zero-copy transfer.
    #[allow(clippy::wrong_self_convention)]
    fn to_csr<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let result = self.take_result()?;
        convert::csr_to_scipy(py, result.x)
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
