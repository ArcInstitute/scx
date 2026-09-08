// Query pipeline Python bindings
//
// Wraps scx-engine::QueryPipeline and QueryResult for Python.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_engine::pipeline::{QueryPipeline, QueryResult, TypedQueryResult};
use scx_engine::EngineError;
use scx_sparse::ValueDtype;

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
        // Unknown modality name — matches the `KeyError` convention used across
        // pyscx's other `modality=` kwargs (`read_uns`, `detection_counts`, …).
        EngineError::UnknownModality { .. } => PyKeyError::new_err(e.to_string()),
        // Multimodal file queried without a modality → a value/usage error.
        EngineError::ModalityRequired { .. } => PyValueError::new_err(e.to_string()),
        // An operation refusing an input it cannot serve faithfully (e.g. a
        // dtype-selected collect over a fused transform) is a usage error.
        EngineError::UnsupportedRewrite { .. } => PyValueError::new_err(e.to_string()),
        // The fail-loud cast gate surfaces through the engine as
        // `ScxError::Codec`. Same reasoning as `typed_read_to_pyerr` on the
        // eager path: a refused narrow is a bad request, not a runtime fault,
        // and the two paths must agree — a `uint16` request that raises
        // `ValueError` eagerly cannot raise `RuntimeError` through a query.
        EngineError::FormatError(scx_format_io::ScxError::Codec(ce)) => {
            PyValueError::new_err(ce.to_string())
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
    // f32 by construction — this door has no dtype kwarg (`read_group`,
    // `read_reference`, the group-shard readers, `read_cloud`), so the only
    // possible loss is the decode that already happened.
    // Not `guard_failure_to_pyerr`: that message offers a wider `data_dtype`
    // and `allow_lossy=True`, and this door (`read_group`, `read_reference`, the
    // group-shard readers, `read_cloud`) has neither kwarg. Advertising a remedy
    // the signature does not carry is worse than the bare fact.
    convert::guard_decode_loss_f32_only(result.max_value)?;
    let x = convert::csr_to_scipy(py, result.x)?;
    anndata_from_parts(py, x, &result.obs, &result.var)
}

/// Assemble an `anndata.AnnData` from an already-materialized `X` plus the
/// result's obs / var batches.
///
/// Split out so both matrix representations share it, and so the obs/var half
/// stops being duplicated per dtype arm.
pub(crate) fn anndata_from_parts<'py>(
    py: Python<'py>,
    x: Bound<'py, PyAny>,
    obs: &arrow::array::RecordBatch,
    var: &arrow::array::RecordBatch,
) -> PyResult<Bound<'py, PyAny>> {
    let anndata_mod = crate::pyimport::import_module(py, "anndata")?;

    let obs_table = convert::record_batch_to_pyarrow(py, obs)?;
    let obs_df = convert::pyarrow_table_to_pandas(&obs_table)?;
    let var_table = convert::record_batch_to_pyarrow(py, var)?;
    let var_df = convert::pyarrow_table_to_pandas(&var_table)?;

    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item("X", x)?;
    kwargs.set_item("obs", obs_df)?;
    kwargs.set_item("var", var_df)?;

    anndata_mod.call_method("AnnData", (), Some(&kwargs))
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
/// A failed builder step leaves the pipeline unchanged and still usable —
/// a bad predicate or an unknown gene name does not invalidate the object.
/// Only a *successful* `.collect()` consumes it; after that every method
/// raises RuntimeError.
#[pyclass]
pub struct PyQueryPipeline {
    /// `None` iff a **successful** `collect()` took the pipeline.
    ///
    /// Nothing else may empty this field. Builder steps mutate in place via
    /// `QueryPipeline::*_mut`, which leaves the pipeline untouched on error,
    /// and `collect()` runs on a borrow and takes only once it has succeeded.
    /// That is what makes [`CONSUMED_MSG`] true wherever it is raised: the
    /// previous take-then-rebuild shape dropped the pipeline inside the
    /// engine's consuming builder on any error, so a predicate typo left the
    /// object permanently dead while blaming a `collect()` that never ran.
    pipeline: Option<QueryPipeline>,
}

/// Raised when a method is called on a pipeline that a successful `collect()`
/// already consumed. Shared by every accessor so the three former copies of
/// this string cannot drift.
const CONSUMED_MSG: &str = "pipeline was consumed by a successful collect(); \
                            call Experiment.query() to start a new pipeline";

impl PyQueryPipeline {
    /// Create from an already-opened QueryPipeline (Rust-only).
    pub fn from_pipeline(pipeline: QueryPipeline) -> Self {
        Self {
            pipeline: Some(pipeline),
        }
    }

    /// Borrow the inner pipeline, or raise if a successful `collect()` took it.
    fn pipeline_ref(&self) -> PyResult<&QueryPipeline> {
        self.pipeline
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(CONSUMED_MSG))
    }

    /// Mutably borrow the inner pipeline for an in-place builder step.
    fn pipeline_mut(&mut self) -> PyResult<&mut QueryPipeline> {
        self.pipeline
            .as_mut()
            .ok_or_else(|| PyRuntimeError::new_err(CONSUMED_MSG))
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
            inner
                .pipeline_mut()?
                .filter_obs_mut(expr)
                .map_err(engine_to_pyerr)?;
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
            inner
                .pipeline_mut()?
                .filter_var_mut(expr)
                .map_err(engine_to_pyerr)?;
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
            let p = inner.pipeline_mut()?;
            // Hoisted so only one shared reborrow of `p` is live across the
            // `read_var_for` call; `read_var_for` returns an owned RecordBatch,
            // so the reborrow ends with the statement and `p` is free for the
            // mutating `select_genes_mut` below.
            let modality_id = p.modality_id();
            // Read var only when at least one name needs resolving. Scope to the
            // pipeline's modality so names resolve against the modality's var
            // (a multimodal file has no global `var` section).
            let var_batch = if genes.iter().any(|g| matches!(g, GeneRef::Name(_))) {
                Some(
                    p.reader()
                        .read_var_for(modality_id)
                        .map_err(engine_to_pyerr)?,
                )
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
                            // Nothing was moved out of `inner`, so returning
                            // here leaves the pipeline untouched and usable.
                            None => {
                                return Err(PyKeyError::new_err(format!(
                                    "gene name '{name}' not found in var index"
                                )))
                            }
                        }
                    }
                }
            }
            p.select_genes_mut(indices);
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
            inner.pipeline_mut()?.with_normalize_mut(target_sum);
        }
        Ok(slf)
    }

    /// Enable log1p transformation.
    /// Returns self for method chaining (finding 9.5).
    fn with_log1p(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        {
            let mut inner = slf.borrow_mut();
            inner.pipeline_mut()?.with_log1p_mut();
        }
        Ok(slf)
    }

    /// Limit the number of returned cells.
    /// Returns self for method chaining (finding 9.5).
    fn limit(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        {
            let mut inner = slf.borrow_mut();
            inner.pipeline_mut()?.limit_mut(n);
        }
        Ok(slf)
    }

    /// Execute the pipeline and return a `PyQueryResult`.
    ///
    /// This is where all I/O and computation occurs. A *successful* collect
    /// consumes the pipeline — further method calls then raise RuntimeError.
    /// A failed one does not: the pipeline stays usable so the caller can fix
    /// the offending step (e.g. an out-of-range gene index, or a `data_dtype`
    /// too narrow for the file) and re-collect. The GIL is released during
    /// execution.
    ///
    /// Args:
    ///     data_dtype: dtype to **decode** `X` at, rather than the default
    ///         `float32`. Because this call is the decode, it is the only place
    ///         a dtype can make the read lossless: a count above 2²⁴ read into
    ///         `"uint32"` / `"int64"` / `"float64"` comes back exactly, where
    ///         naming the dtype on the following `to_anndata()` / `to_csr()`
    ///         only casts values that already rounded (and fails loud). A
    ///         narrower dtype than the file's values still refuses. Refused
    ///         with `with_normalize()` / `with_log1p()`, which replace the
    ///         counts with floating-point values.
    ///     index_dtype: dtype for the CSR column indices (`"int16"` /
    ///         `"int32"` / `"int64"`). Only meaningful alongside `data_dtype`;
    ///         scipy upcasts `int16` back to `int32` on construction.
    ///     allow_lossy: accept a narrowing decode that loses data. The gate
    ///         paired with `data_dtype` — without it here, an intentional lossy
    ///         narrow would be unreachable at decode time.
    ///
    /// Keyword-only. `container=` is deliberately absent: it cannot lose
    /// anything, so it stays on `to_anndata()` / `to_csr()` where the
    /// presentation choice belongs.
    #[pyo3(signature = (*, data_dtype=None, index_dtype=None, allow_lossy=false))]
    fn collect(
        &mut self,
        py: Python<'_>,
        data_dtype: Option<&str>,
        index_dtype: Option<&str>,
        allow_lossy: bool,
    ) -> PyResult<PyQueryResult> {
        // Run on a borrow (`collect_ref` / `collect_typed`) rather than moving
        // the pipeline into the engine, so an error leaves it recoverable —
        // including a refused cast, which now makes `collect` itself able to
        // fail. Consuming on success is a deliberate binding-layer policy, not a
        // Rust-ownership artifact.
        // Parse **every** supplied kwarg before dispatching. Parsing inside one
        // arm meant `collect(index_dtype="not_a_dtype")` was silently accepted
        // on the other — the same silent-argument-loss class this PR refuses at
        // materialize time, on the very door that refusal points at.
        let requested = convert::parse_value_dtype_opt(data_dtype)?;
        let requested_index = index_dtype
            .map(|_| convert::parse_index_dtype(index_dtype))
            .transpose()?;

        // An explicit `index_dtype` needs the typed assembly too — it is what
        // narrows the index buffer — so it routes there even at `float32`, where
        // the values are identical either way.
        let typed = match (requested, requested_index) {
            (None | Some(scx_sparse::ValueDtype::F32), None) => None,
            (dtype, index) => Some(scx_sparse::MaterializePlan {
                container: scx_sparse::Container::Csr,
                data_dtype: dtype.unwrap_or(scx_sparse::ValueDtype::F32),
                index_dtype: index.unwrap_or(scx_sparse::IndexDtype::I32),
                allow_lossy,
            }),
        };

        let mut result = match typed {
            // Nothing to narrow: today's f32 decode, unchanged and
            // byte-identical, and the door the fused transforms need.
            None => {
                let pipeline = self.pipeline_ref()?;
                let r = py
                    .detach(|| pipeline.collect_ref())
                    .map_err(engine_to_pyerr)?;
                PyQueryResult::from_result(r)
            }
            Some(plan) => {
                let pipeline = self.pipeline_ref()?;
                let r = py
                    .detach(|| pipeline.collect_typed(&plan))
                    .map_err(engine_to_pyerr)?;
                PyQueryResult::from_typed_result(r)
            }
        };
        // `allow_lossy` is a property of the read the caller asked for, so it
        // travels with the result. Dropping it meant `collect(allow_lossy=True)`
        // then `to_csr()` was refused with a message telling the caller to pass
        // the flag they had just passed.
        result.collected_allow_lossy = allow_lossy;
        self.pipeline = None;
        Ok(result)
    }

    /// Number of matching cells, **without decoding X** and ignoring `limit`.
    ///
    /// Runs only the planning + masking half of the engine; on the row-set fast
    /// path (indexed predicates) this needs no obs/X shard decode. Borrows the
    /// pipeline — it stays usable for a later `collect()`. The GIL is released
    /// during execution.
    fn count(&self, py: Python<'_>) -> PyResult<usize> {
        let pipeline = self.pipeline_ref()?;
        let result = py.detach(|| pipeline.count()).map_err(engine_to_pyerr)?;
        Ok(result.matched_rows)
    }

    /// Whether any cell matches, **without decoding X** and ignoring `limit`.
    ///
    /// On the row-set fast path this needs no obs/X shard decode; with residual
    /// (non-indexed) predicates it decodes only the narrowed obs shards. Borrows
    /// the pipeline — it stays usable for a later `collect()`.
    fn exists(&self, py: Python<'_>) -> PyResult<bool> {
        let pipeline = self.pipeline_ref()?;
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
/// What `collect()` produced — exactly one arm, taken once.
///
/// One `Option<Collected>` rather than an `Option` per arm: the bug class this
/// field has already had is "more than one thing can empty it", and
/// `take_collected`'s single "already consumed" message is only truthful while
/// exactly one field can be emptied.
enum Collected {
    /// Default `collect()` — the untouched f32 result. Byte-identical path.
    F32(QueryResult),
    /// `collect(data_dtype=…)` — decoded at the caller's dtype by the engine.
    Typed(TypedQueryResult),
}

impl Collected {
    fn shape(&self) -> (usize, usize) {
        match self {
            Collected::F32(r) => (r.x.n_rows(), r.x.n_cols()),
            Collected::Typed(r) => (r.x.n_rows(), r.x.n_cols()),
        }
    }

    fn nnz(&self) -> usize {
        match self {
            Collected::F32(r) => r.x.nnz(),
            Collected::Typed(r) => r.x.nnz(),
        }
    }

    fn skipped_shards(&self) -> usize {
        match self {
            Collected::F32(r) => r.skipped_shards,
            Collected::Typed(r) => r.skipped_shards,
        }
    }

    fn total_shards(&self) -> usize {
        match self {
            Collected::F32(r) => r.total_shards,
            Collected::Typed(r) => r.total_shards,
        }
    }

    fn max_value(&self) -> u32 {
        match self {
            Collected::F32(r) => r.max_value,
            Collected::Typed(r) => r.max_value,
        }
    }

    /// The dtype the values are **already** stored at — step 1 of the
    /// decode-loss rule, and the meaning of a later `data_dtype=None`.
    fn dtype(&self) -> ValueDtype {
        match self {
            Collected::F32(_) => ValueDtype::F32,
            Collected::Typed(r) => r.x.values.dtype(),
        }
    }

    /// The index dtype the column indices are already stored at.
    /// Whether the values were decoded by the typed assembler.
    ///
    /// **Not** `dtype() != F32`: `collect(index_dtype=…)` alone takes the typed
    /// route at `F32` values, so the value dtype stopped being a proxy for the
    /// arm the moment that became reachable.
    fn is_typed(&self) -> bool {
        matches!(self, Collected::Typed(_))
    }

    fn index_dtype(&self) -> scx_sparse::IndexDtype {
        match self {
            Collected::F32(_) => scx_sparse::IndexDtype::I32,
            Collected::Typed(r) => r.x.indices.dtype(),
        }
    }
}

#[pyclass]
pub struct PyQueryResult {
    /// `None` iff a **successful** to_anndata()/to_csr() took it.
    collected: Option<Collected>,
    // Cached values so getters work after to_anndata()/to_csr() consumes data
    cached_n_obs: usize,
    cached_n_vars: usize,
    cached_nnz: usize,
    cached_skipped_shards: usize,
    cached_total_shards: usize,
    /// The dtype X was decoded at, kept for `__repr__` after consumption.
    cached_dtype: ValueDtype,
    /// Whether `collect()` was given `allow_lossy=True`. Carried so a later
    /// materialize does not re-demand an opt-in the caller already made.
    collected_allow_lossy: bool,
}

impl PyQueryResult {
    /// Construct from a default (f32) QueryResult, caching dimension values.
    pub fn from_result(r: QueryResult) -> Self {
        Self::from_collected(Collected::F32(r))
    }

    /// Construct from a dtype-selected result.
    pub fn from_typed_result(r: TypedQueryResult) -> Self {
        Self::from_collected(Collected::Typed(r))
    }

    fn from_collected(c: Collected) -> Self {
        let (n_obs, n_vars) = c.shape();
        Self {
            cached_n_obs: n_obs,
            cached_n_vars: n_vars,
            cached_nnz: c.nnz(),
            cached_skipped_shards: c.skipped_shards(),
            cached_total_shards: c.total_shards(),
            cached_dtype: c.dtype(),
            collected_allow_lossy: false,
            collected: Some(c),
        }
    }

    /// Take the inner result, returning an error if already consumed.
    fn take_collected(&mut self) -> PyResult<Collected> {
        self.collected.take().ok_or_else(|| {
            PyRuntimeError::new_err("QueryResult already consumed by to_anndata() or to_csr()")
        })
    }

    /// Peek the decode-loss `value_max` without consuming the result, so the
    /// guard can fire *before* `take_result()`. That keeps a tripped guard
    /// non-destructive: the caller can retry the same object with
    /// `allow_lossy=True`, which is exactly what the guard's message tells them
    /// to do. Mirrors `rscx`'s `RQueryResult::peek_max_value`. Returns 0 once
    /// consumed (the guard then passes and `take_result()` surfaces the
    /// "already consumed" error).
    fn peek_max_value(&self) -> u32 {
        self.collected
            .as_ref()
            .map(Collected::max_value)
            .unwrap_or(0)
    }

    /// Guard the request against **both** loss steps, then take the result.
    ///
    /// Guarding before the take is what keeps a tripped guard non-destructive,
    /// so the `allow_lossy=True` retry the message recommends works on the same
    /// object. Both `to_anndata` and `to_csr` route through here: the defect
    /// being fixed was four hand-copied guard sites, and two of them were these.
    fn guarded_take(
        &mut self,
        requested: Option<ValueDtype>,
        requested_index: Option<scx_sparse::IndexDtype>,
        container: scx_sparse::Container,
        allow_lossy: bool,
    ) -> PyResult<Collected> {
        let Some(collected) = self.collected.as_ref() else {
            // Already consumed: let the take below say so rather than guarding
            // a result that is not there (peek_max_value returns 0).
            return self.take_collected();
        };
        let assembled = collected.dtype();
        let requested = requested.unwrap_or(assembled);

        // A typed result's buffers are handed to numpy as they are, so an
        // `index_dtype=` here would be accepted and ignored. Refuse instead —
        // and refuse before the value-dtype check only in the sense that both
        // are reported the same way.
        // Skipped for dense output, which carries no column indices at all — a
        // width the caller cannot observe is not worth refusing over.
        if collected.is_typed() && container == scx_sparse::Container::Csr {
            if let Some(idx) = requested_index {
                if idx != collected.index_dtype() {
                    return Err(PyValueError::new_err(format!(
                        "this result was collected with `index_dtype=\"{have}\"`; \
                         `index_dtype=\"{want}\"` here would be ignored, because a \
                         dtype-selected result's indices are already narrowed. Pass it to \
                         `.collect(index_dtype=...)` instead.",
                        have = collected.index_dtype().numpy_name(),
                        want = idx.numpy_name(),
                    )));
                }
            }
        }
        if collected.is_typed() && requested != assembled {
            // A typed result would need an assembled→requested cast across the
            // full dtype matrix, which nothing else here needs and which is not
            // what the caller means: they want a different *decode*, or a numpy
            // cast of what they have. `allow_lossy` does not unlock it — it is a
            // wrong-knob error, not a loss question.
            return Err(PyValueError::new_err(format!(
                "this result was collected as {assembled}; `data_dtype=\"{requested}\"` here \
                 would re-cast it. Pass the dtype to `.collect(data_dtype=...)`, or cast the \
                 returned matrix yourself (`X.astype(np.{requested})`).",
                assembled = assembled.numpy_name(),
                requested = requested.numpy_name(),
            )));
        }

        // Step 1 is skipped on the typed arm: it ran pre-decode inside
        // `collect_typed`, and re-raising it here would report a decode that
        // already happened and cannot be undone.
        if !collected.is_typed() {
            convert::materialize_guard(
                self.peek_max_value(),
                assembled,
                requested,
                allow_lossy || self.collected_allow_lossy,
            )
            .map_err(convert::guard_failure_to_pyerr)?;
        }
        self.take_collected()
    }

    /// Materialize X from whichever arm the result carries, plus its obs / var.
    ///
    /// The f32 arm takes the untouched `csr_to_scipy_typed` path (zero-copy for
    /// the default plan, post-assembly cast otherwise). The typed arm's values
    /// are already at the requested dtype, so they move straight into numpy —
    /// `container` is the only thing left to honour.
    fn materialize_x<'py>(
        &self,
        py: Python<'py>,
        collected: Collected,
        plan: &scx_sparse::MaterializePlan,
    ) -> PyResult<(
        Bound<'py, PyAny>,
        arrow::array::RecordBatch,
        arrow::array::RecordBatch,
    )> {
        match collected {
            Collected::F32(r) => {
                let x = convert::csr_to_scipy_typed(py, r.x, plan)?;
                Ok((x, r.obs, r.var))
            }
            Collected::Typed(r) => {
                let x = match plan.container {
                    scx_sparse::Container::Csr => convert::typed_csr_to_scipy(py, r.x)?,
                    scx_sparse::Container::Dense => {
                        // `n_rows × n_cols` allocation plus a full scatter — no
                        // Python touched, so it does not need the GIL.
                        let dense = py
                            .detach(|| scx_format_io::scatter_typed_csr_to_dense(&r.x))
                            .map_err(|e| PyValueError::new_err(e.to_string()))?;
                        convert::typed_dense_to_numpy(py, dense)?
                    }
                };
                Ok((x, r.obs, r.var))
            }
        }
    }
}

#[pymethods]
impl PyQueryResult {
    /// Convert the query result to an AnnData object.
    ///
    /// Uses zero-copy for the CSR matrix (moves Vec to numpy).
    /// After this call, to_anndata()/to_csr() cannot be called again,
    /// but getters (n_obs, n_vars, etc.) still work.
    #[pyo3(signature = (container="csr", data_dtype=None, index_dtype=None, allow_lossy=false))]
    #[allow(clippy::wrong_self_convention)]
    fn to_anndata<'py>(
        &mut self,
        py: Python<'py>,
        container: &str,
        data_dtype: Option<&str>,
        index_dtype: Option<&str>,
        allow_lossy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let requested = convert::parse_value_dtype_opt(data_dtype)?;
        let requested_index = index_dtype
            .map(|_| convert::parse_index_dtype(index_dtype))
            .transpose()?;
        let plan = convert::build_plan_with_default(
            py,
            container,
            data_dtype,
            index_dtype,
            allow_lossy,
            self.cached_dtype,
        )?;
        let collected =
            self.guarded_take(requested, requested_index, plan.container, allow_lossy)?;
        let (x, obs, var) = self.materialize_x(py, collected, &plan)?;
        anndata_from_parts(py, x, &obs, &var)
    }

    /// Return just the scipy CSR matrix (or dense array) without building full
    /// AnnData. Also consumes the inner data for zero-copy transfer.
    #[pyo3(signature = (container="csr", data_dtype=None, index_dtype=None, allow_lossy=false))]
    #[allow(clippy::wrong_self_convention)]
    fn to_csr<'py>(
        &mut self,
        py: Python<'py>,
        container: &str,
        data_dtype: Option<&str>,
        index_dtype: Option<&str>,
        allow_lossy: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let requested = convert::parse_value_dtype_opt(data_dtype)?;
        let requested_index = index_dtype
            .map(|_| convert::parse_index_dtype(index_dtype))
            .transpose()?;
        let plan = convert::build_plan_with_default(
            py,
            container,
            data_dtype,
            index_dtype,
            allow_lossy,
            self.cached_dtype,
        )?;
        let collected =
            self.guarded_take(requested, requested_index, plan.container, allow_lossy)?;
        let (x, _obs, _var) = self.materialize_x(py, collected, &plan)?;
        Ok(x)
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
            "PyQueryResult(n_obs={}, n_vars={}, nnz={}, dtype={}, skipped={}/{}{})",
            self.cached_n_obs,
            self.cached_n_vars,
            self.cached_nnz,
            self.cached_dtype.numpy_name(),
            self.cached_skipped_shards,
            self.cached_total_shards,
            if self.collected.is_none() {
                ", consumed"
            } else {
                ""
            },
        )
    }
}
