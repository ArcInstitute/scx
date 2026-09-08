// QueryPipeline builder and QueryResult types.
//
// The pipeline is lazy: it stores configuration but performs no I/O
// until `.collect()` is called. Schema errors are raised immediately
// at construction time (docs/api.md (Query engine, lazy evaluation)).

use std::path::Path;

use arrow::datatypes::Schema;
use scx_format_io::ScxReader;
use scx_sparse::{MaterializePlan, ScxCsr};

use scx_format_io::DeletionVectors;

use crate::error::Result;
use crate::predicate::{parse_predicate, Predicate};
use crate::reader::SectionReader;

/// Configuration for normalize-total operation.
#[derive(Debug, Clone)]
pub struct NormalizeConfig {
    pub target_sum: f64,
}

/// The matrix shapes a [`QueryResult`] can carry.
///
/// Two implementations — [`ScxCsr`] (the default `f32` collect) and
/// [`scx_sparse::TypedCsr`] (the dtype-selected collect) — consumed by this
/// module's shared `Debug` and by the bindings, which cache a result's
/// dimensions before handing the matrix out.
pub trait QueryMatrix {
    /// `(n_rows, n_cols)`.
    fn shape(&self) -> (usize, usize);
    /// Number of stored non-zeros.
    fn nnz(&self) -> usize;
}

impl QueryMatrix for ScxCsr {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn nnz(&self) -> usize {
        ScxCsr::nnz(self)
    }
}

impl QueryMatrix for scx_sparse::TypedCsr {
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    fn nnz(&self) -> usize {
        scx_sparse::TypedCsr::nnz(self)
    }
}

/// A [`QueryResult`] whose `X` was decoded at a caller-chosen dtype rather than
/// `f32` — what [`QueryPipeline::collect_typed`] returns.
pub type TypedQueryResult = QueryResult<scx_sparse::TypedCsr>;

/// Result of a query pipeline execution.
///
/// Generic in the matrix so the dtype-selected collect reuses every metadata
/// field, with [`ScxCsr`] defaulted so existing callers keep spelling it
/// `QueryResult`.
pub struct QueryResult<X = ScxCsr> {
    /// The expression matrix (filtered + projected).
    pub x: X,
    /// Observation metadata for matching cells.
    pub obs: arrow::array::RecordBatch,
    /// Variable/gene metadata for projected genes.
    pub var: arrow::array::RecordBatch,
    /// Number of shards skipped by Level 1 (catalog-stats) pushdown.
    pub skipped_shards: usize,
    /// Total number of shards in the file.
    pub total_shards: usize,
    /// Sum of rows in the candidate shards that survived Level 1 pushdown,
    /// before Level 2 (PredicateIndex / row-evaluator) narrowing. Compare
    /// against `result.x.n_rows()` to see how much Level 2 trimmed.
    pub candidate_shard_rows: usize,
    /// Number of rows that matched the obs predicate, *before* `--limit`
    /// truncation. `result.x.n_rows()` is the post-limit returned count;
    /// `matched_rows` is the true Level 2 match count. They're equal when
    /// no limit was applied (or the limit exceeded the match count).
    pub matched_rows: usize,
    /// Maximum `ShardStats::value_max` over the shards that survived Level 1
    /// pushdown (i.e. that were decoded into `x`). `0` when no integer-encoded
    /// shard contributed (float-encoded shards record `value_max = 0`). A
    /// reader compares this against [`scx_codec::F32_MAX_EXACT_INT`] to fail
    /// loud on the silent `u32 → f32` decode loss before returning `x`.
    pub max_value: u32,
}

/// Result of a count-only query ([`QueryPipeline::count`]): the matched-row
/// count plus pushdown statistics, computed **without decoding the X matrix**
/// (CLI2) and **without applying `limit`** (CLI6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountResult {
    /// Number of rows matching the obs predicate (Level 2). This is the true
    /// match count — `limit` is never applied on the count path.
    pub matched_rows: usize,
    /// Number of shards skipped by Level 1 (catalog-stats) pushdown.
    pub skipped_shards: usize,
    /// Total number of shards in the file.
    pub total_shards: usize,
    /// Sum of rows in candidate shards that survived Level 1 pushdown, before
    /// Level 2 narrowing.
    pub candidate_shard_rows: usize,
}

impl<X: QueryMatrix> std::fmt::Debug for QueryResult<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryResult")
            .field("x_shape", &self.x.shape())
            .field("obs_rows", &self.obs.num_rows())
            .field("var_rows", &self.var.num_rows())
            .field("skipped_shards", &self.skipped_shards)
            .field("total_shards", &self.total_shards)
            .field("candidate_shard_rows", &self.candidate_shard_rows)
            .field("matched_rows", &self.matched_rows)
            .finish()
    }
}

/// A lazy query pipeline builder for SCX files.
///
/// Stores configuration but performs no I/O until `.collect()` is called.
/// Schema errors (unknown columns, type mismatches) are raised immediately.
pub struct QueryPipeline {
    reader: Box<dyn SectionReader>,
    /// Modality scope for X / var / gene resolution. `0` = global /
    /// single-modality axis (the default, back-compatible). `>= 1` scopes
    /// the pipeline to one modality of a multimodal file (see
    /// `docs/multimodal.md` § 3.4). Obs predicate evaluation is always
    /// global — obs is shared across modalities.
    modality_id: u8,
    /// Cached per-modality variable count. For `modality_id == 0` this is
    /// `header().n_vars`; for a modality it is `modality_info(id).n_vars`
    /// (the file-wide header carries the **max** across modalities and must
    /// not be used as the X width — see docs/format.md § 13.4).
    n_vars: usize,
    obs_schema: Schema,
    var_schema: Schema,
    obs_predicates: Vec<Predicate>,
    var_predicates: Vec<Predicate>,
    gene_indices: Option<Vec<u32>>,
    normalize: Option<NormalizeConfig>,
    log1p: bool,
    limit: Option<usize>,
    deletion_vectors: Option<DeletionVectors>,
    /// Whether row-set predicate pushdown may run for this pipeline. `false`
    /// forces the legacy full-decode obs path. Defaults from the
    /// `SCX_DISABLE_ROWSET_PUSHDOWN` environment variable, read **once here**
    /// rather than per query — see [`rowset_pushdown`](Self::rowset_pushdown).
    rowset_pushdown: bool,
    /// Lazily-parsed `group_index` sidecar (F2 grouped reads). Parsed at most
    /// once per pipeline by [`require_grouped`](Self::require_grouped); shared
    /// by all grouped-read calls on this pipeline.
    group_index: std::sync::OnceLock<crate::group::GroupIndex>,
}

impl std::fmt::Debug for QueryPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryPipeline")
            .field("obs_schema_fields", &self.obs_schema.fields().len())
            .field("var_schema_fields", &self.var_schema.fields().len())
            .field("obs_predicates", &self.obs_predicates.len())
            .field("var_predicates", &self.var_predicates.len())
            .field("gene_indices", &self.gene_indices)
            .field("normalize", &self.normalize)
            .field("log1p", &self.log1p)
            .field("limit", &self.limit)
            .finish()
    }
}

impl QueryPipeline {
    /// Open a local SCX file and create a new query pipeline.
    ///
    /// Reads and caches the obs and var schemas for eager validation.
    /// If deletion vectors are present, they are loaded automatically.
    /// Convenience wrapper around [`from_reader`](Self::from_reader)
    /// for the local-file case.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_reader(Box::new(ScxReader::open(path)?))
    }

    /// Open a local SCX file scoped to a single modality.
    ///
    /// `modality_id == 0` is the global / single-modality axis (identical
    /// to [`open`](Self::open)); `>= 1` scopes X / var / gene resolution to
    /// that modality of a multimodal file. See
    /// [`from_reader_for_modality`](Self::from_reader_for_modality).
    pub fn open_for_modality(path: impl AsRef<Path>, modality_id: u8) -> Result<Self> {
        Self::from_reader_for_modality(Box::new(ScxReader::open(path)?), modality_id)
    }

    /// Create a new query pipeline from any `SectionReader`.
    ///
    /// This is the entry point for cloud-backed pipelines: hand it a
    /// `scx_cloud::CloudSectionReader` (or any other backend) to drive
    /// the same query engine against remote storage.
    ///
    /// Scopes to the global / single-modality axis (`modality_id = 0`).
    pub fn from_reader(reader: Box<dyn SectionReader>) -> Result<Self> {
        Self::from_reader_for_modality(reader, 0)
    }

    /// Create a modality-scoped query pipeline from any `SectionReader`.
    ///
    /// `modality_id == 0` is the global / single-modality axis. `>= 1`
    /// scopes X assembly, var, `filter_var`, and `select_genes` to one
    /// modality of a multimodal file; `filter_obs` still evaluates against
    /// the shared global obs axis. The modality is fixed here (not a
    /// builder setter) because `filter_var` / `select_genes` validate
    /// against the var schema immediately.
    ///
    /// Deletion vectors (v2 global-obs bitmaps) apply identically to every
    /// modality, so a modality-scoped query on a file with deletions is fully
    /// supported — the deleted cells are dropped from the queried modality's X.
    ///
    /// Errors:
    ///  - [`EngineError::UnknownModality`](crate::error::EngineError::UnknownModality)
    ///    if `modality_id` is not registered.
    pub fn from_reader_for_modality(
        reader: Box<dyn SectionReader>,
        modality_id: u8,
    ) -> Result<Self> {
        // Validate the modality id against the file's modality table. id 0 is
        // always valid (global / single-modality). Validate a registered
        // modality by table membership (ids are 1-based) so a genuine I/O error
        // from `modality_n_vars` below propagates as-is rather than being masked
        // as `UnknownModality`.
        if modality_id != 0 && u32::from(modality_id) > reader.n_modalities() {
            return Err(crate::error::EngineError::UnknownModality {
                requested: modality_id.to_string(),
                available: reader.modality_names(),
            });
        }

        // Cache schemas for eager validation. obs is global; var is
        // modality-scoped.
        let obs_schema = reader.read_obs_schema()?;
        let var_schema = reader.read_var_schema_for(modality_id)?;
        let n_vars = reader.modality_n_vars(modality_id)? as usize;

        // Load deletion vectors if present. v2 bitmaps are global obs rows, so
        // they apply identically to any modality's CSR shards (each tiles
        // `[0, n_obs)`) — no per-modality remapping needed.
        let deletion_vectors = reader.read_deletion_vectors()?;

        Ok(Self {
            reader,
            modality_id,
            n_vars,
            obs_schema,
            var_schema,
            obs_predicates: Vec::new(),
            var_predicates: Vec::new(),
            gene_indices: None,
            normalize: None,
            log1p: false,
            limit: None,
            deletion_vectors,
            rowset_pushdown: !crate::collect::rowset_pushdown_disabled_by_env(),
            group_index: std::sync::OnceLock::new(),
        })
    }

    /// The modality this pipeline is scoped to (`0` = global /
    /// single-modality).
    pub fn modality_id(&self) -> u8 {
        self.modality_id
    }

    // -- Builders ----------------------------------------------------------
    //
    // Each builder comes in two forms: an in-place `*_mut(&mut self)` primitive
    // that carries the implementation, and the fluent consuming form that
    // delegates to it. The `_mut` forms exist for bindings that own the
    // pipeline behind an `Option` (pyscx `PyQueryPipeline`, rscx
    // `RQueryPipeline`): the consuming form moves `self` into the call and
    // drops it on `Err`, so a failed step there leaves the binding with a
    // permanently empty `Option`. Mutating in place cannot lose the pipeline —
    // a rejected predicate leaves it unchanged and still usable. Keep `_mut`
    // as the primitive; the reverse direction would need a dummy
    // `QueryPipeline` to `mem::replace` with.

    /// Filter observations (cells) by a predicate expression, in place.
    ///
    /// The `&mut` form of [`filter_obs`](Self::filter_obs). On a parse/schema
    /// error the pipeline is **unchanged and still usable** — nothing is
    /// half-applied.
    pub fn filter_obs_mut(&mut self, expr: &str) -> Result<()> {
        let pred = parse_predicate(expr, &self.obs_schema, "obs")?;
        self.obs_predicates.push(pred);
        Ok(())
    }

    /// Filter observations (cells) by a predicate expression.
    ///
    /// The predicate is validated against the obs schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    pub fn filter_obs(mut self, expr: &str) -> Result<Self> {
        self.filter_obs_mut(expr)?;
        Ok(self)
    }

    /// Filter variables (genes) by a predicate expression, in place.
    ///
    /// The `&mut` form of [`filter_var`](Self::filter_var); see
    /// [`filter_obs_mut`](Self::filter_obs_mut) for why it exists.
    pub fn filter_var_mut(&mut self, expr: &str) -> Result<()> {
        let pred = parse_predicate(expr, &self.var_schema, "var")?;
        self.var_predicates.push(pred);
        Ok(())
    }

    /// Filter variables (genes) by a predicate expression.
    ///
    /// The predicate is validated against the var schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    pub fn filter_var(mut self, expr: &str) -> Result<Self> {
        self.filter_var_mut(expr)?;
        Ok(self)
    }

    /// Select specific gene indices for projection, in place.
    pub fn select_genes_mut(&mut self, gene_indices: Vec<u32>) {
        self.gene_indices = Some(gene_indices);
    }

    /// Select specific gene indices for projection.
    ///
    /// Out-of-range indices are handled at collect time.
    pub fn select_genes(mut self, gene_indices: Vec<u32>) -> Self {
        self.select_genes_mut(gene_indices);
        self
    }

    /// Enable normalize-total with the given target sum, in place.
    pub fn with_normalize_mut(&mut self, target_sum: f64) {
        self.normalize = Some(NormalizeConfig { target_sum });
    }

    /// Enable normalize-total with the given target sum.
    pub fn with_normalize(mut self, target_sum: f64) -> Self {
        self.with_normalize_mut(target_sum);
        self
    }

    /// Enable log1p transformation, in place.
    pub fn with_log1p_mut(&mut self) {
        self.log1p = true;
    }

    /// Enable log1p transformation.
    pub fn with_log1p(mut self) -> Self {
        self.with_log1p_mut();
        self
    }

    /// Limit the number of returned cells, in place.
    pub fn limit_mut(&mut self, n: usize) {
        self.limit = Some(n);
    }

    /// Limit the number of returned cells.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit_mut(n);
        self
    }

    /// Enable or disable row-set predicate pushdown for this pipeline, in
    /// place.
    ///
    /// The `&mut` form of [`rowset_pushdown`](Self::rowset_pushdown).
    pub fn rowset_pushdown_mut(&mut self, enabled: bool) {
        self.rowset_pushdown = enabled;
    }

    /// Enable or disable row-set predicate pushdown for this pipeline.
    ///
    /// **Diagnostic knob, not a semantic one.** Both paths must return the same
    /// rows for every predicate — that equivalence is what
    /// `tests/rowset_differential.rs` asserts, and it is the reason this exists:
    /// the oracle runs each generated query on both paths and compares. Turning
    /// pushdown off makes queries slower, never more correct.
    ///
    /// Defaults to enabled unless `SCX_DISABLE_ROWSET_PUSHDOWN` is set in the
    /// environment, which remains the field's only other source. Prefer this
    /// setter in tests: the environment is process-global, so a test that
    /// toggles it races every other test in the same binary, and cargo runs a
    /// test binary's tests on multiple threads by default.
    pub fn rowset_pushdown(mut self, enabled: bool) -> Self {
        self.rowset_pushdown_mut(enabled);
        self
    }

    /// Whether row-set predicate pushdown may run for this pipeline.
    pub(crate) fn rowset_pushdown_enabled(&self) -> bool {
        self.rowset_pushdown
    }

    /// Execute the pipeline and return the query result **without consuming
    /// it**.
    ///
    /// Execution only ever reads the pipeline (`plan_and_mask(&self)` +
    /// `materialize(&self, …)`), so a failed or repeated run is harmless.
    /// Bindings that hand a long-lived pipeline object to a user call this so
    /// that a failed `collect()` cannot destroy the caller's object; consuming
    /// on success is then a binding-layer policy rather than a consequence of
    /// Rust ownership.
    pub fn collect_ref(&self) -> Result<QueryResult> {
        crate::collect::execute(self)
    }

    /// Execute the pipeline and return the query result.
    ///
    /// This is where all I/O and computation occurs.
    /// Delegates to `collect::execute()` which implements the full
    /// pipeline: pushdown → decode → projection → filter → fused ops.
    pub fn collect(self) -> Result<QueryResult> {
        self.collect_ref()
    }

    /// Execute the pipeline, decoding `X` **at `mplan`'s dtype** instead of
    /// `f32`.
    ///
    /// The reason this exists rather than being a kwarg on the materialization
    /// step: [`collect_ref`](Self::collect_ref) *is* the decode, so a dtype
    /// named after it can only cast values that already rounded. A count above
    /// 2²⁴ is exact through this door and unreachable through the other.
    ///
    /// Borrows, like `collect_ref`, so a refused cast leaves the pipeline usable.
    /// Ask [`typed_collect_supported`](Self::typed_collect_supported) first — a
    /// fused transform is refused here rather than silently served from the f32
    /// route, because *which guard ran* is the thing a caller is relying on.
    pub fn collect_typed(&self, mplan: &MaterializePlan) -> Result<TypedQueryResult> {
        crate::collect::execute_typed(self, mplan)
    }

    /// Whether [`collect_typed`](Self::collect_typed) can serve `mplan`.
    ///
    /// `false` for a pipeline carrying `with_normalize` / `with_log1p` (the
    /// transform replaces the stored counts with floats, so no dtype makes the
    /// read exact) and for a non-CSR container (the typed assembly produces a
    /// CSR; a dense request is a presentation step on top). Both cases belong on
    /// the `f32` route, which guards on `f32` — correctly, because that is what
    /// it produced.
    pub fn typed_collect_supported(&self, mplan: &MaterializePlan) -> bool {
        mplan.container == scx_sparse::Container::Csr && self.normalize.is_none() && !self.log1p
    }

    /// Count matching rows without decoding the X matrix (CLI2).
    ///
    /// Runs only the planning + masking half (`pushdown → obs/var predicate
    /// evaluation → row-keep masks`); no CSR shard payload is decoded and no
    /// fused transform runs. `limit` is **not** applied — the returned count is
    /// the true Level-2 match count, so `--count --limit` cannot misreport
    /// (CLI6). Takes `&self` so the same pipeline can still be `collect`ed.
    pub fn count(&self) -> Result<CountResult> {
        crate::collect::count(self)
    }

    /// Whether any row matches the obs/var predicates, without decoding the X
    /// matrix and ignoring `limit`.
    ///
    /// On the row-set fast path (indexed-only predicates) this needs **no
    /// obs/X shard decode** — the answer comes from the index-derived row-set.
    /// With residual (non-indexed) predicates it decodes only the obs shards the
    /// indexed part narrowed to, like [`Self::count`]. Currently equivalent to
    /// `count()? > 0` (it computes the full match count rather than stopping at
    /// the first match).
    pub fn exists(&self) -> Result<bool> {
        crate::collect::exists(self)
    }

    // -- F2: grouped reads (over the `group_index` sidecar) -----------------

    /// Load the group index, or `Err(EngineError::NotGrouped)` if the archive
    /// was not written with `--group-by`.
    ///
    /// The sidecar is parsed at most once per pipeline and cached; repeated
    /// grouped reads share the same `GroupIndex`.
    pub fn require_grouped(&self) -> Result<&crate::group::GroupIndex> {
        if let Some(gi) = self.group_index.get() {
            return Ok(gi);
        }
        let gi = crate::group::GroupIndex::open(self.reader.as_ref())?;
        // Idempotent on a race: a concurrent caller may have set it first; the
        // value is identical either way, so discard our copy on a lost race.
        let _ = self.group_index.set(gi);
        Ok(self.group_index.get().expect("group_index populated above"))
    }

    /// Read exactly the rows of `label` in the `group_by` column (Route B —
    /// direct global row-range slice). Grouped archive only.
    ///
    /// `Err(EngineError::UnknownGroupLabel { suggestions })` on a miss, carrying
    /// `strsim` close matches (difflib-equivalent).
    pub fn read_group(&self, label: &str) -> Result<QueryResult> {
        let gi = self.require_grouped()?;
        // F6 Phase 0: a group may span multiple records (block sub-flush split it
        // across shards). `label_range` unions them into the group's contiguous
        // `[start, stop)` span so `read_row_range` returns every cell of the
        // group regardless of how many shards it occupies.
        match gi.label_range(label) {
            Some((start, stop)) => self.read_row_range(start, stop),
            None => Err(crate::error::EngineError::UnknownGroupLabel {
                label: label.to_string(),
                suggestions: gi.close_matches(label, 5),
            }),
        }
    }

    /// Read the full reference region (the contiguous leading range spanning
    /// every reference record). `Ok(None)` if the archive has no reference
    /// rows. A reference set spanning several leading shards is fully returned.
    pub fn read_reference(&self) -> Result<Option<QueryResult>> {
        let gi = self.require_grouped()?;
        match gi.reference_range() {
            Some((start, stop)) => Ok(Some(self.read_row_range(start, stop)?)),
            None => Ok(None),
        }
    }

    /// Distinct group labels present (for discovery / error messages).
    pub fn group_labels(&self) -> Result<Vec<String>> {
        Ok(self.require_grouped()?.labels())
    }

    /// One handle per non-reference shard (global range + label→local-slice
    /// map). The caller streams each shard via [`Self::read_row_range`] over the
    /// handle's `[global_start, global_stop)`, keeping ~one shard resident.
    pub fn iter_group_shards(&self) -> Result<Vec<crate::group::GroupShardHandle>> {
        Ok(self.require_grouped()?.shard_handles())
    }

    /// Read a contiguous global output-row range `[start, stop)` as a
    /// `QueryResult` (the Route B seam). Decodes only the CSR shards covering
    /// the range; `skipped_shards` reflects the pruning. obs is sliced to the
    /// range and var passed through, so `.to_anndata()` matches every other read
    /// path.
    ///
    /// Deletion vectors are honored: if the archive gained a deletion vector
    /// after the grouped sort (e.g. a later `mark_deleted`), rows marked deleted
    /// in `[start, stop)` are dropped from both `X` and `obs`, preserving
    /// equivalence with `query().collect().to_anndata()`.
    ///
    /// **Raw read.** This method (and the grouped `read_group` / `read_reference`
    /// / `iter_group_shards` built on it) ignores the pipeline *builder* state —
    /// `filter_obs` / `filter_var` predicates, `select_genes`, `with_normalize`,
    /// `with_log1p`, and `limit` are NOT applied. It returns the raw cells of the
    /// range (minus deletions). The pyscx grouped-read methods are structurally
    /// safe (each opens a fresh pipeline with no builder state); Rust callers who
    /// need projection/transforms must post-process or use `collect()`.
    ///
    /// On a row-sharded file (atlas scale) obs metadata is read shard-scoped —
    /// only the `ObsMetadataShard`s overlapping `[start, stop)` are decoded, so
    /// peak obs memory is bounded by the touched shards, not the whole table.
    /// Legacy single-section obs (and pre-stats files) fall back to a full read
    /// + slice.
    pub fn read_row_range(&self, start: u64, stop: u64) -> Result<QueryResult> {
        use scx_format_io::catalog::FullCatalogEntry;

        let n_obs = self.reader.header().n_obs;
        let stop = stop.max(start);
        if stop > n_obs {
            return Err(crate::error::EngineError::Generic(format!(
                "read_row_range: stop {stop} exceeds n_obs {n_obs}"
            )));
        }
        let range_len = (stop - start) as usize;
        let n_vars = self.n_vars;

        // Modality-aware keep-mask for deletion vectors (None when the archive
        // is clean, which is the case for a freshly written grouped output).
        // The global (whole-cell) bitmap always applies; a scoped
        // (`modality_id >= 1`) bitmap, when present, additionally drops that
        // modality's rows. Today only the global bitmap is populated, so this
        // equals the whole-cell mask for every modality.
        let keep_mask: Option<Vec<bool>> = self
            .deletion_vectors
            .as_ref()
            .map(|dv| dv.build_keep_mask(n_obs as usize, self.modality_id));

        let csr_shards: Vec<&FullCatalogEntry> =
            crate::collect::scan_shards(self.reader.catalog(), self.modality_id);
        debug_assert!(
            self.modality_id == 0
                || self
                    .reader
                    .catalog()
                    .modality_csr_ranges_tile_obs(self.modality_id, n_obs),
            "modality {} CSR shards must tile [0, n_obs)",
            self.modality_id
        );
        let total_shards = csr_shards.len();

        let mut merged_indptr: Vec<i64> = vec![0];
        let mut merged_indices: Vec<i32> = Vec::new();
        let mut merged_data: Vec<f32> = Vec::new();
        let mut candidate_shard_rows = 0usize;
        let mut touched = 0usize;
        let mut max_value = 0u32; // max ShardStats::value_max over touched shards
        let mut covered = 0u64; // rows of [start, stop) actually decoded
        let mut kept = 0usize; // rows surviving the deletion filter
                               // Per-range keep flags, in ascending global-row order (aligned with the
                               // obs slice). Empty when there are no deletion vectors.
        let mut keep_local: Vec<bool> = Vec::new();

        for e in &csr_shards {
            let Some(stats) = e.stats.as_ref() else {
                continue;
            };
            let (rs, re) = (stats.row_start, stats.row_end);
            if re <= start || rs >= stop {
                continue; // no overlap
            }
            touched += 1;
            max_value = max_value.max(stats.value_max);
            candidate_shard_rows += (re - rs) as usize;
            let (indptr, indices, data) = self.reader.read_shard_from_entry(e)?;
            let lo = start.max(rs);
            let hi = stop.min(re);
            for g in lo..hi {
                covered += 1;
                let keep = keep_mask.as_ref().map(|m| m[g as usize]).unwrap_or(true);
                if keep_mask.is_some() {
                    keep_local.push(keep);
                }
                if !keep {
                    continue;
                }
                let l = (g - rs) as usize;
                let s = indptr[l] as usize;
                let en = indptr[l + 1] as usize;
                merged_indices.extend_from_slice(&indices[s..en]);
                merged_data.extend_from_slice(&data[s..en]);
                let prev = *merged_indptr.last().unwrap();
                merged_indptr.push(prev + (en - s) as i64);
                kept += 1;
            }
        }

        // Coverage guard: every row in [start, stop) must be backed by a shard
        // (no gap, no double-count). Protects the `new_unchecked` below from a
        // silently-corrupt CSR on a bad/out-of-cover range.
        if covered != stop - start {
            return Err(crate::error::EngineError::Generic(format!(
                "read_row_range: range [{start}, {stop}) is not fully covered by CSR shards \
                 (covered {covered} of {range_len} rows)"
            )));
        }

        let x = ScxCsr::new_unchecked((kept, n_vars), merged_indptr, merged_indices, merged_data);

        // obs: the kept global rows of [start, stop), ascending. With no
        // deletion vector that's the whole contiguous range; otherwise only the
        // rows the keep-mask retained.
        let matching_global_rows: Vec<u32> = if keep_mask.is_some() {
            keep_local
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| {
                    if k {
                        Some(start as u32 + i as u32)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            (start..stop).map(|g| g as u32).collect()
        };

        // Prefer reading only the obs shards overlapping [start, stop) on a
        // row-sharded file. Fall back to a full read + slice for legacy
        // single-section obs (or pre-stats shards that lack row ranges).
        let obs = if self.reader.obs_metadata_shard_count() > 0 {
            match crate::collect::obs_shard_ranges_from_catalog(self.reader.catalog()) {
                Some(ranges) if !ranges.is_empty() => crate::collect::materialize_filtered_obs(
                    self.reader.as_ref(),
                    &ranges,
                    &matching_global_rows,
                )?,
                _ => {
                    self.read_obs_range_full(start, range_len, &keep_local, keep_mask.is_some())?
                }
            }
        } else {
            self.read_obs_range_full(start, range_len, &keep_local, keep_mask.is_some())?
        };
        let var = self.reader.read_var_for(self.modality_id)?;

        Ok(QueryResult {
            x,
            obs,
            var,
            skipped_shards: total_shards.saturating_sub(touched),
            total_shards,
            candidate_shard_rows,
            matched_rows: kept,
            max_value,
        })
    }

    /// Legacy obs path for `read_row_range`: read the full obs table, slice the
    /// `[start, start+range_len)` window, then (if the archive has deletions)
    /// keep only the rows flagged in `keep_local`. Used for single-section obs
    /// and pre-stats sharded files where shard-scoped reads aren't available.
    fn read_obs_range_full(
        &self,
        start: u64,
        range_len: usize,
        keep_local: &[bool],
        has_deletions: bool,
    ) -> Result<arrow::array::RecordBatch> {
        use arrow::array::{RecordBatch, UInt32Array};
        let obs_full = self.reader.read_obs()?;
        let obs_range = obs_full.slice(start as usize, range_len);
        if has_deletions {
            let idx: Vec<u32> = keep_local
                .iter()
                .enumerate()
                .filter_map(|(i, &k)| if k { Some(i as u32) } else { None })
                .collect();
            let take = UInt32Array::from(idx);
            let cols = obs_range
                .columns()
                .iter()
                .map(|c| arrow::compute::take(c.as_ref(), &take, None))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(RecordBatch::try_new(obs_range.schema(), cols)?)
        } else {
            Ok(obs_range)
        }
    }

    // -- Accessors for testing and the `collect` modules --

    /// Access the cached obs schema.
    pub fn obs_schema(&self) -> &Schema {
        &self.obs_schema
    }

    /// Access the cached var schema.
    pub fn var_schema(&self) -> &Schema {
        &self.var_schema
    }

    /// The per-modality variable count (X width) this pipeline assembles to.
    /// For `modality_id == 0` this is `header().n_vars`; for a modality it is
    /// `modality_info(id).n_vars`.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Access the accumulated obs predicates.
    pub fn obs_predicates(&self) -> &[Predicate] {
        &self.obs_predicates
    }

    /// Access the accumulated var predicates.
    pub fn var_predicates(&self) -> &[Predicate] {
        &self.var_predicates
    }

    /// Access the underlying reader as a `SectionReader` trait object.
    pub fn reader(&self) -> &dyn SectionReader {
        self.reader.as_ref()
    }

    /// Downcast accessor for callers that need the local `ScxReader`
    /// API (`scx subset` reads `uns`, layer names, value encoding,
    /// etc.). Returns `None` for cloud-backed pipelines.
    pub fn local_reader(&self) -> Option<&ScxReader> {
        self.reader.as_any().downcast_ref::<ScxReader>()
    }

    /// Access the gene indices for projection.
    pub(crate) fn gene_indices(&self) -> Option<&Vec<u32>> {
        self.gene_indices.as_ref()
    }

    /// Access the normalize target sum.
    pub(crate) fn normalize_target_sum(&self) -> Option<f64> {
        self.normalize.as_ref().map(|n| n.target_sum)
    }

    /// Check if log1p is enabled.
    pub(crate) fn log1p(&self) -> bool {
        self.log1p
    }

    /// Access the limit value.
    pub(crate) fn limit_value(&self) -> Option<usize> {
        self.limit
    }

    /// Access the loaded deletion vectors.
    pub(crate) fn deletion_vectors(&self) -> &Option<DeletionVectors> {
        &self.deletion_vectors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::EngineError;
    use arrow::array::Array;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format_io::header::FileHeader;
    use scx_format_io::writer::ScxWriter;
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
    }

    fn sample_obs(n: usize) -> arrow::array::RecordBatch {
        let schema = Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
        ]);
        let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
        let types: Vec<&str> = (0..n)
            .map(|i| match i % 3 {
                0 => "T cell",
                1 => "B cell",
                _ => "NK cell",
            })
            .collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(
                    ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(types)),
            ],
        )
        .unwrap()
    }

    fn sample_var(n: usize) -> arrow::array::RecordBatch {
        let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
        let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
        arrow::array::RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            ))],
        )
        .unwrap()
    }

    fn sample_shard_data(n_rows: usize, n_vars: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for row in 0..n_rows {
            let col0 = (row * 2) % n_vars;
            let col1 = (row * 2 + 1) % n_vars;
            indices.push(col0 as u32);
            indices.push(col1 as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        (indptr, indices, values)
    }

    fn write_test_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> std::path::PathBuf {
        let path = dir.path().join("test.scx");
        let header = sample_header(n_obs as u64, n_vars as u64, (n_obs * 2) as u64);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        writer.write_var(&sample_var(n_vars)).unwrap();
        let (indptr, indices, values) = sample_shard_data(n_obs, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        writer.finish().unwrap();
        path
    }

    #[test]
    fn pipeline_open_succeeds() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path).unwrap();
        assert_eq!(pipeline.obs_schema().fields().len(), 2);
        assert_eq!(pipeline.var_schema().fields().len(), 1);
    }

    #[test]
    fn filter_obs_schema_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path).unwrap();
        let err = pipeline.filter_obs("nonexistent == 'x'").unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
    }

    /// The `_mut` builder leaves the pipeline usable after a rejected
    /// predicate — the property the consuming form structurally cannot offer
    /// (it drops `self` on `Err`). pyscx/rscx hold the pipeline behind an
    /// `Option`, so without this a predicate typo emptied it permanently.
    #[test]
    fn filter_obs_mut_error_leaves_pipeline_usable() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let mut pipeline = QueryPipeline::open(&path).unwrap();

        let err = pipeline.filter_obs_mut("nonexistent == 'x'").unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
        assert_eq!(
            pipeline.obs_predicates().len(),
            0,
            "a rejected predicate must not half-apply"
        );

        pipeline.filter_obs_mut("cell_type == 'T cell'").unwrap();
        assert_eq!(pipeline.obs_predicates().len(), 1);
        // cell_type cycles T/B/NK, so 10 rows => rows 0, 3, 6, 9.
        assert_eq!(pipeline.collect().unwrap().x.n_rows(), 4);
    }

    #[test]
    fn filter_var_mut_error_leaves_pipeline_usable() {
        let dir = tempfile::tempdir().unwrap();
        // n_vars > 2 * n_obs so `sample_shard_data`'s column pairs never wrap:
        // a wrapped row is unsorted and `project_csr_row` rejects it.
        let path = write_test_file(&dir, 10, 20);
        let mut pipeline = QueryPipeline::open(&path).unwrap();

        let err = pipeline.filter_var_mut("nonexistent == 'x'").unwrap_err();
        assert!(matches!(err, EngineError::SchemaError { .. }));
        assert_eq!(pipeline.var_predicates().len(), 0);

        pipeline.filter_var_mut("gene_id == 'gene_1'").unwrap();
        assert_eq!(pipeline.var_predicates().len(), 1);
        assert_eq!(pipeline.collect().unwrap().x.n_cols(), 1);
    }

    /// The `_mut` family and the fluent consuming family build the same
    /// pipeline — the delegation is a refactor, not a second implementation.
    #[test]
    fn mut_builders_match_consuming() {
        let dir = tempfile::tempdir().unwrap();
        // Wide enough that no row's column pair wraps — see
        // `filter_var_mut_error_leaves_pipeline_usable`.
        let path = write_test_file(&dir, 10, 20);

        let mut a = QueryPipeline::open(&path).unwrap();
        a.filter_obs_mut("cell_type == 'T cell'").unwrap();
        a.select_genes_mut(vec![0, 1, 2]);
        a.with_normalize_mut(1e4);
        a.with_log1p_mut();
        a.limit_mut(5);

        let b = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .select_genes(vec![0, 1, 2])
            .with_normalize(1e4)
            .with_log1p()
            .limit(5);

        assert_eq!(a.obs_predicates().len(), b.obs_predicates().len());
        assert_eq!(a.gene_indices(), b.gene_indices());
        assert_eq!(a.normalize_target_sum(), b.normalize_target_sum());
        assert_eq!(a.log1p(), b.log1p());
        assert_eq!(a.limit_value(), b.limit_value());

        let (ra, rb) = (a.collect().unwrap(), b.collect().unwrap());
        assert_eq!(ra.x.shape, rb.x.shape);
        assert_eq!(ra.x.indptr, rb.x.indptr);
        assert_eq!(ra.x.indices, rb.x.indices);
        assert_eq!(ra.x.data, rb.x.data);
    }

    /// `collect_ref` borrows: it is repeatable, and the pipeline survives for a
    /// later consuming `collect()`. This is what lets a binding run a collect
    /// that might fail without destroying the caller's object.
    #[test]
    fn collect_ref_does_not_consume() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap();

        let a = pipeline.collect_ref().unwrap();
        let b = pipeline.collect_ref().unwrap();
        assert_eq!(a.x.shape, b.x.shape);
        assert_eq!(a.x.data, b.x.data);

        let c = pipeline.collect().unwrap();
        assert_eq!(a.x.shape, c.x.shape);
        assert_eq!(a.x.data, c.x.data);
    }

    #[test]
    fn filter_obs_accumulates() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .filter_obs("cell_id == 'cell_0'")
            .unwrap();
        assert_eq!(pipeline.obs_predicates().len(), 2);
    }

    #[test]
    fn pipeline_chaining() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .select_genes(vec![0, 1, 2])
            .with_normalize(1e4)
            .with_log1p()
            .limit(5);
        // Verify all settings applied (collect not yet implemented)
        assert_eq!(pipeline.obs_predicates().len(), 1);
    }

    #[test]
    fn collect_returns_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path).unwrap();
        let result = pipeline.collect().unwrap();
        assert_eq!(result.x.n_rows(), 10);
        assert_eq!(result.x.n_cols(), 5);
    }

    // -------------------------------------------------------------------
    // Multimodal-aware predicate pushdown. A two-modality fixture with
    // DIFFERENT n_vars per modality
    // (rna=5, adt=3) exercises the per-modality X width + global obs mask.
    // -------------------------------------------------------------------

    /// Write a 2-modality file (rna: n_vars=5, adt: n_vars=3) over a shared
    /// `n_obs` obs axis. Each modality has one CSR shard covering [0, n_obs).
    /// Row `r` in rna expresses gene `r % 5` (value r+1); in adt gene `r % 3`
    /// (value 100). Returns `(path, rna_id, adt_id)`.
    fn write_multimodal_test_file(
        dir: &tempfile::TempDir,
        n_obs: usize,
    ) -> (std::path::PathBuf, u8, u8) {
        use scx_format_io::modality::ModalityType;
        let path = dir.path().join("mm.scx");
        // header n_vars is the file-wide max across modalities.
        let header = sample_header(n_obs as u64, 5, 0);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        writer.write_obs(&sample_obs(n_obs)).unwrap();
        let rna_id = writer
            .add_modality(
                "rna",
                ModalityType::Rna,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        let adt_id = writer
            .add_modality(
                "adt",
                ModalityType::Protein,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        writer.write_var_for(rna_id, &sample_var(5)).unwrap();
        writer.write_var_for(adt_id, &sample_var(3)).unwrap();
        writer.set_modality_n_vars(rna_id, 5).unwrap();
        writer.set_modality_n_vars(adt_id, 3).unwrap();

        let mut rna_indptr = vec![0u64];
        let mut rna_indices = Vec::new();
        let mut rna_values = Vec::new();
        let mut adt_indptr = vec![0u64];
        let mut adt_indices = Vec::new();
        let mut adt_values = Vec::new();
        for r in 0..n_obs {
            rna_indices.push((r % 5) as u32);
            rna_values.push(((r + 1) % 256) as u8);
            rna_indptr.push(rna_indptr.last().unwrap() + 1);
            adt_indices.push((r % 3) as u32);
            adt_values.push(100u8);
            adt_indptr.push(adt_indptr.last().unwrap() + 1);
        }
        let rna_shard = scx_format_io::ShardBuffers::new(
            &rna_indptr,
            &rna_indices,
            &rna_values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(rna_id, 0, rna_shard).unwrap();
        let adt_shard = scx_format_io::ShardBuffers::new(
            &adt_indptr,
            &adt_indices,
            &adt_values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(adt_id, 0, adt_shard).unwrap();
        writer.finish().unwrap();
        (path, rna_id, adt_id)
    }

    /// Invariant 1: on a single-modality file, `open` and `open_for_modality(0)`
    /// are identical; the default path is unchanged.
    #[test]
    fn single_modality_parity_modality_zero() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let a = QueryPipeline::open(&path).unwrap().collect().unwrap();
        let b = QueryPipeline::open_for_modality(&path, 0)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(a.x.n_rows(), b.x.n_rows());
        assert_eq!(a.x.n_cols(), b.x.n_cols());
        assert_eq!(a.x.indices, b.x.indices);
        assert_eq!(a.x.data, b.x.data);
        assert_eq!(a.total_shards, b.total_shards);
    }

    /// Invariant 2: X width comes from the modality, not header().n_vars (max).
    #[test]
    fn multimodal_per_modality_width() {
        let dir = tempfile::tempdir().unwrap();
        let (path, rna_id, adt_id) = write_multimodal_test_file(&dir, 9);

        let rna = QueryPipeline::open_for_modality(&path, rna_id)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(rna.x.n_rows(), 9);
        assert_eq!(rna.x.n_cols(), 5, "rna width is its own n_vars");
        assert_eq!(rna.var.num_rows(), 5);

        let adt = QueryPipeline::open_for_modality(&path, adt_id)
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(adt.x.n_rows(), 9);
        assert_eq!(
            adt.x.n_cols(),
            3,
            "adt width is its own n_vars, not header max 5"
        );
        assert_eq!(adt.var.num_rows(), 3);
    }

    /// Invariant 3: the obs predicate is global; querying either modality with
    /// the same filter yields the same obs rows (but each modality's own X).
    #[test]
    fn multimodal_global_obs_mask_applied_per_modality() {
        let dir = tempfile::tempdir().unwrap();
        let (path, rna_id, adt_id) = write_multimodal_test_file(&dir, 9);
        // cell_type cycles T/B/NK → "T cell" selects rows 0, 3, 6.
        let rna = QueryPipeline::open_for_modality(&path, rna_id)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();
        let adt = QueryPipeline::open_for_modality(&path, adt_id)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(rna.x.n_rows(), 3);
        assert_eq!(adt.x.n_rows(), 3);
        // Same obs rows regardless of modality.
        let rna_ids = rna
            .obs
            .column_by_name("cell_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let adt_ids = adt
            .obs
            .column_by_name("cell_id")
            .unwrap()
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let rna_vec: Vec<&str> = (0..rna_ids.len()).map(|i| rna_ids.value(i)).collect();
        let adt_vec: Vec<&str> = (0..adt_ids.len()).map(|i| adt_ids.value(i)).collect();
        assert_eq!(rna_vec, vec!["cell_0", "cell_3", "cell_6"]);
        assert_eq!(rna_vec, adt_vec, "obs rows identical across modalities");
        // But widths differ.
        assert_eq!(rna.x.n_cols(), 5);
        assert_eq!(adt.x.n_cols(), 3);
    }

    /// Invariant 6: gene projection resolves in the modality's var index space.
    #[test]
    fn multimodal_gene_projection_in_modality_space() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _rna_id, adt_id) = write_multimodal_test_file(&dir, 9);
        // adt has only 3 genes; select the last two.
        let adt = QueryPipeline::open_for_modality(&path, adt_id)
            .unwrap()
            .select_genes(vec![1, 2])
            .collect()
            .unwrap();
        assert_eq!(adt.x.n_cols(), 2);
        assert_eq!(adt.var.num_rows(), 2);
    }

    /// Unknown modality id → `UnknownModality`.
    #[test]
    fn unknown_modality_errors() {
        let dir = tempfile::tempdir().unwrap();
        let (path, _rna_id, _adt_id) = write_multimodal_test_file(&dir, 9);
        let err = QueryPipeline::open_for_modality(&path, 99).unwrap_err();
        assert!(
            matches!(err, EngineError::UnknownModality { .. }),
            "expected UnknownModality, got {err:?}"
        );
    }

    /// count()/exists() are modality-scoped (obs mask global, count identical
    /// across modalities for a global predicate).
    #[test]
    fn multimodal_count_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let (path, rna_id, adt_id) = write_multimodal_test_file(&dir, 9);
        let rna_count = QueryPipeline::open_for_modality(&path, rna_id)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .count()
            .unwrap();
        let adt_count = QueryPipeline::open_for_modality(&path, adt_id)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .count()
            .unwrap();
        assert_eq!(rna_count.matched_rows, 3);
        assert_eq!(adt_count.matched_rows, 3);
    }
}
