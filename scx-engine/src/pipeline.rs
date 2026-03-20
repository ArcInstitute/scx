// QueryPipeline builder and QueryResult types.
//
// The pipeline is lazy: it stores configuration but performs no I/O
// until `.collect()` is called. Schema errors are raised immediately
// at construction time (SPEC §7.1).

use std::path::Path;

use arrow::datatypes::Schema;
use scx_format::ScxReader;
use scx_sparse::ScxCsr;

use scx_format::DeletionVectors;

use crate::error::{EngineError, Result};
use crate::predicate::{parse_predicate, Predicate};

/// Configuration for normalize-total operation.
#[derive(Debug, Clone)]
pub struct NormalizeConfig {
    pub target_sum: f64,
}

/// Result of a query pipeline execution.
///
/// See SPEC.md §6.1 for the in-memory data model.
pub struct QueryResult {
    /// The expression matrix (filtered + projected).
    pub x: ScxCsr,
    /// Observation metadata for matching cells.
    pub obs: arrow::array::RecordBatch,
    /// Variable/gene metadata for projected genes.
    pub var: arrow::array::RecordBatch,
    /// Number of shards skipped by predicate pushdown.
    pub skipped_shards: usize,
    /// Total number of shards in the file.
    pub total_shards: usize,
}

impl std::fmt::Debug for QueryResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryResult")
            .field("x_shape", &self.x.shape)
            .field("obs_rows", &self.obs.num_rows())
            .field("var_rows", &self.var.num_rows())
            .field("skipped_shards", &self.skipped_shards)
            .field("total_shards", &self.total_shards)
            .finish()
    }
}

/// A lazy query pipeline builder for SCX files.
///
/// Stores configuration but performs no I/O until `.collect()` is called.
/// Schema errors (unknown columns, type mismatches) are raised immediately.
pub struct QueryPipeline {
    reader: ScxReader,
    obs_schema: Schema,
    var_schema: Schema,
    obs_predicates: Vec<Predicate>,
    var_predicates: Vec<Predicate>,
    gene_indices: Option<Vec<u32>>,
    normalize: Option<NormalizeConfig>,
    log1p: bool,
    limit: Option<usize>,
    #[allow(dead_code)] // loaded at open(); used by Phase F execution
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
    /// Open an SCX file and create a new query pipeline.
    ///
    /// Reads and caches the obs and var schemas for eager validation.
    /// If deletion vectors are present, they are loaded automatically.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let reader = ScxReader::open(path)?;

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
        let pred = parse_predicate(expr, &self.obs_schema)?;
        self.obs_predicates.push(pred);
        Ok(self)
    }

    /// Filter variables (genes) by a predicate expression.
    ///
    /// The predicate is validated against the var schema immediately.
    /// Multiple calls accumulate predicates with AND semantics.
    pub fn filter_var(mut self, expr: &str) -> Result<Self> {
        let pred = parse_predicate(expr, &self.var_schema)?;
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
    /// Currently a placeholder — full implementation in Phase F.
    pub fn collect(self) -> Result<QueryResult> {
        // Phase F will implement the full execution pipeline.
        // For now, return EmptyPipeline to indicate this is not yet implemented.
        Err(EngineError::EmptyPipeline)
    }

    // -- Accessors for testing --

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

    /// Access the underlying reader.
    pub fn reader(&self) -> &ScxReader {
        &self.reader
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType, Field};
    use scx_codec::{CodecId, ValueEncoding};
    use scx_format::writer::ScxWriter;
    use scx_format::header::FileHeader;
    use std::sync::Arc;

    fn sample_header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
        FileHeader {
            magic: scx_format::MAGIC,
            format_version: 1,
            header_length: 256,
            flags: 0,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards: 0,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 0,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 0,
            root_catalog_length: 0,
            full_catalog_offset: 0,
            full_catalog_length: 0,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            reserved: [0u8; 132],
        }
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
    fn collect_placeholder_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_test_file(&dir, 10, 5);
        let pipeline = QueryPipeline::open(&path).unwrap();
        let err = pipeline.collect().unwrap_err();
        assert!(matches!(err, EngineError::EmptyPipeline));
    }
}
