// QueryPipeline builder and QueryResult types.
//
// The pipeline is lazy: it stores configuration but performs no I/O
// until `.collect()` is called. Schema errors are raised immediately
// at construction time (docs/api.md (Query engine, lazy evaluation)).

use std::path::Path;

use arrow::datatypes::Schema;
use scx_format_io::ScxReader;
use scx_sparse::ScxCsr;

use scx_format_io::DeletionVectors;

use crate::error::Result;
use crate::predicate::{parse_predicate, Predicate};
use crate::reader::SectionReader;

/// Configuration for normalize-total operation.
#[derive(Debug, Clone)]
pub struct NormalizeConfig {
    pub target_sum: f64,
}

/// Result of a query pipeline execution.
pub struct QueryResult {
    /// The expression matrix (filtered + projected).
    pub x: ScxCsr,
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

impl std::fmt::Debug for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryResult")
            .field("x_shape", &self.x.shape)
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
    obs_schema: Schema,
    var_schema: Schema,
    obs_predicates: Vec<Predicate>,
    var_predicates: Vec<Predicate>,
    gene_indices: Option<Vec<u32>>,
    normalize: Option<NormalizeConfig>,
    log1p: bool,
    limit: Option<usize>,
    deletion_vectors: Option<DeletionVectors>,
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

    /// Create a new query pipeline from any `SectionReader`.
    ///
    /// This is the entry point for cloud-backed pipelines: hand it a
    /// `scx_cloud::CloudSectionReader` (or any other backend) to drive
    /// the same query engine against remote storage.
    pub fn from_reader(reader: Box<dyn SectionReader>) -> Result<Self> {
        // Cache schemas for eager validation
        let obs_schema = reader.read_obs_schema()?;
        let var_schema = reader.read_var_schema()?;

        // Load deletion vectors if present
        let deletion_vectors = reader.read_deletion_vectors()?;

        Ok(Self {
            reader,
            obs_schema,
            var_schema,
            obs_predicates: Vec::new(),
            var_predicates: Vec::new(),
            gene_indices: None,
            normalize: None,
            log1p: false,
            limit: None,
            deletion_vectors,
        })
    }

    /// Filter observations (cells) by a predicate expression.
    ///
    /// The predicate is validated against the obs schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    pub fn filter_obs(mut self, expr: &str) -> Result<Self> {
        let pred = parse_predicate(expr, &self.obs_schema, "obs")?;
        self.obs_predicates.push(pred);
        Ok(self)
    }

    /// Filter variables (genes) by a predicate expression.
    ///
    /// The predicate is validated against the var schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    pub fn filter_var(mut self, expr: &str) -> Result<Self> {
        let pred = parse_predicate(expr, &self.var_schema, "var")?;
        self.var_predicates.push(pred);
        Ok(self)
    }

    /// Select specific gene indices for projection.
    ///
    /// Out-of-range indices are handled at collect time.
    pub fn select_genes(mut self, gene_indices: Vec<u32>) -> Self {
        self.gene_indices = Some(gene_indices);
        self
    }

    /// Enable normalize-total with the given target sum.
    pub fn with_normalize(mut self, target_sum: f64) -> Self {
        self.normalize = Some(NormalizeConfig { target_sum });
        self
    }

    /// Enable log1p transformation.
    pub fn with_log1p(mut self) -> Self {
        self.log1p = true;
        self
    }

    /// Limit the number of returned cells.
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    /// Execute the pipeline and return the query result.
    ///
    /// This is where all I/O and computation occurs.
    /// Delegates to `collect::execute()` which implements the full
    /// pipeline: pushdown → decode → projection → filter → fused ops.
    pub fn collect(self) -> Result<QueryResult> {
        crate::collect::execute(self)
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
    pub fn require_grouped(&self) -> Result<crate::group::GroupIndex> {
        crate::group::GroupIndex::open(self.reader.as_ref())
    }

    /// Read exactly the rows of `label` in the `group_by` column (Route B —
    /// direct global row-range slice). Grouped archive only.
    ///
    /// `Err(EngineError::UnknownGroupLabel { suggestions })` on a miss, carrying
    /// `strsim` close matches (difflib-equivalent).
    pub fn read_group(&self, label: &str) -> Result<QueryResult> {
        let gi = self.require_grouped()?;
        match gi.record(label) {
            Some(rec) => self.read_row_range(rec.row_start, rec.row_stop),
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
    /// v1 note: obs metadata is read in full and sliced (cheap Arrow slice); the
    /// expensive X decode is what the range restricts.
    pub fn read_row_range(&self, start: u64, stop: u64) -> Result<QueryResult> {
        use scx_format_io::catalog::FullCatalogEntry;
        let stop = stop.max(start);
        let n = (stop - start) as usize;
        let n_vars = self.reader.header().n_vars as usize;
        let csr_shards: Vec<&FullCatalogEntry> = self.reader.catalog().shards_sorted();
        let total_shards = csr_shards.len();

        let mut merged_indptr: Vec<i64> = vec![0];
        let mut merged_indices: Vec<i32> = Vec::new();
        let mut merged_data: Vec<f32> = Vec::new();
        let mut candidate_shard_rows = 0usize;
        let mut touched = 0usize;

        for e in &csr_shards {
            let Some(stats) = e.stats.as_ref() else {
                continue;
            };
            let (rs, re) = (stats.row_start, stats.row_end);
            if re <= start || rs >= stop {
                continue; // no overlap
            }
            touched += 1;
            candidate_shard_rows += (re - rs) as usize;
            let (indptr, indices, data) = self.reader.read_shard_from_entry(e)?;
            let lo = start.max(rs);
            let hi = stop.min(re);
            for g in lo..hi {
                let l = (g - rs) as usize;
                let s = indptr[l] as usize;
                let en = indptr[l + 1] as usize;
                merged_indices.extend_from_slice(&indices[s..en]);
                merged_data.extend_from_slice(&data[s..en]);
                let prev = *merged_indptr.last().unwrap();
                merged_indptr.push(prev + (en - s) as i64);
            }
        }

        let x = ScxCsr::new_unchecked((n, n_vars), merged_indptr, merged_indices, merged_data);
        let obs_full = self.reader.read_obs()?;
        let obs = obs_full.slice(start as usize, n);
        let var = self.reader.read_var()?;

        Ok(QueryResult {
            x,
            obs,
            var,
            skipped_shards: total_shards.saturating_sub(touched),
            total_shards,
            candidate_shard_rows,
            matched_rows: n,
        })
    }

    // -- Accessors for testing and collect.rs --

    /// Access the cached obs schema.
    pub fn obs_schema(&self) -> &Schema {
        &self.obs_schema
    }

    /// Access the cached var schema.
    pub fn var_schema(&self) -> &Schema {
        &self.var_schema
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
}
