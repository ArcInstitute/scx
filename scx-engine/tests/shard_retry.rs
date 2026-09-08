//! Regression tests for the engine-side per-shard retry isolation in
//! `collect::retry::par_map_with_shard_retry`.
//!
//! At atlas scale a query fans thousands of shard reads out over rayon. The
//! pre-fix `collect::<Result<_>>()?` made a *single* shard read that exhausted
//! its per-request retry budget (during a transient/congestion window) abort
//! the entire query via `?`-propagation. The helper now tolerates a transient
//! per-shard failure by retrying only the failed shards, and fails the query
//! only if a shard is still unrecovered.
//!
//! These tests wrap a real on-disk sharded `ScxReader` in a fault-injecting
//! `SectionReader` and assert:
//!   * a transient obs-shard failure is recovered (same rows as no-fault),
//!   * a persistent shard failure still fails the query, naming the shard,
//!   * a deterministic (non-retryable) decode error is NOT retried (fast-fail),
//!   * a transient CSR X-shard decode failure (`read_shard_from_entry`) is
//!     likewise recovered.

use std::any::Any;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::error::EngineError;
use scx_engine::{
    build_and_write_conversion_predicate_indexes, ConversionPredicateIndexOptions, QueryPipeline,
    SectionReader,
};
use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::header::FileHeader;
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{DeletionVectors, FullCatalog};
use tempfile::TempDir;

const N_VARS: usize = 4;
const N_OBS: u64 = 8;

fn header(n_obs: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, N_VARS as u64, 0, 16384, 0, 0)
}

fn full_obs() -> RecordBatch {
    let cell_id: Vec<String> = (0..N_OBS).map(|i| format!("cell_{i}")).collect();
    let cell_type: Vec<&str> = vec!["A", "A", "A", "A", "B", "B", "B", "B"];
    let donor: Vec<&str> = vec!["d1", "d1", "d2", "d2", "d1", "d1", "d2", "d2"];
    let n_counts: Vec<i64> = vec![10, 11, 12, 13, 1000, 1001, 1002, 1003];
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("donor", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int64, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                cell_id.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_type)),
            Arc::new(StringArray::from(donor)),
            Arc::new(Int64Array::from(n_counts)),
        ],
    )
    .unwrap()
}

fn sample_var() -> RecordBatch {
    let ids: Vec<String> = (0..N_VARS).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// A trivial CSR shard of `n_rows` rows × N_VARS cols (1 nnz/row).
fn write_csr_shard(writer: &mut ScxWriter, n_rows: usize, row_start: u64) {
    let indptr: Vec<u64> = (0..=n_rows as u64).collect();
    let indices: Vec<u32> = (0..n_rows as u32).map(|i| i % N_VARS as u32).collect();
    let values: Vec<u8> = vec![1u8; n_rows];
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
            row_start,
        )
        .unwrap();
}

/// Sharded obs (4 shards of 2 rows) + CSR shards (4 rows each) + a predicate
/// index over `cell_type` / `donor` / `n_counts`. Mirrors the fixture in
/// `obs_shard_streaming.rs`.
fn build_sharded_indexed_file(dir: &TempDir) -> std::path::PathBuf {
    let path = dir.path().join("sharded_indexed.scx");
    let obs = full_obs();
    let var = sample_var();
    let mut writer = ScxWriter::new(&path, header(N_OBS)).unwrap();

    let obs_splits = [(0u64, 2u64), (2, 2), (4, 2), (6, 2)];
    for (i, (row_start, n)) in obs_splits.iter().enumerate() {
        let slice = obs.slice(*row_start as usize, *n as usize);
        writer
            .write_obs_shard(i as u32, *row_start, *n, N_OBS, &slice)
            .unwrap();
    }
    writer.write_var(&var).unwrap();

    write_csr_shard(&mut writer, 4, 0);
    write_csr_shard(&mut writer, 4, 4);

    let opts = ConversionPredicateIndexOptions {
        index_obs: vec![
            "cell_type".to_string(),
            "donor".to_string(),
            "n_counts".to_string(),
        ],
        index_var: Vec::new(),
        index_preset: None,
        index_auto_threshold: 1000,
    };
    let csr_row_ranges = [(0u64, 4u64), (4u64, 8u64)];
    build_and_write_conversion_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        N_VARS,
        &opts,
    )
    .unwrap();
    writer.finish().unwrap();
    path
}

/// What kind of synthetic failure the injector returns.
#[derive(Clone, Copy)]
enum Fault {
    /// Transient I/O error — the engine retries it (mirrors the cloud
    /// `CloudError` → `EngineError::IoError` erasure).
    Transient,
    /// Deterministic decode error — the engine must NOT retry it.
    NonRetryable,
}

/// Wraps a real `ScxReader` and fails the first `obs_fails` reads of a target
/// obs shard (with `usize::MAX` meaning "always fail"). Every other method
/// delegates straight through.
struct FaultInjectingReader {
    inner: ScxReader,
    target_obs_shard: u32,
    obs_fails_remaining: AtomicUsize,
    /// Number of `read_shard_from_entry` (CSR X-shard decode) calls to fail,
    /// regardless of which entry — exercises the Site 3 fan-out.
    x_fails_remaining: AtomicUsize,
    fault: Fault,
}

impl FaultInjectingReader {
    fn new(inner: ScxReader, target_obs_shard: u32, obs_fails: usize, fault: Fault) -> Self {
        Self {
            inner,
            target_obs_shard,
            obs_fails_remaining: AtomicUsize::new(obs_fails),
            x_fails_remaining: AtomicUsize::new(0),
            fault,
        }
    }

    /// Fail the first `n` CSR X-shard decode reads (`read_shard_from_entry`).
    fn with_x_fails(mut self, n: usize) -> Self {
        self.x_fails_remaining = AtomicUsize::new(n);
        self
    }

    /// Consume one failure credit from `counter` if any remain; returns true
    /// when a synthetic failure should be injected for this call.
    fn take_failure(counter: &AtomicUsize) -> bool {
        counter
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| {
                if n > 0 {
                    Some(n - 1)
                } else {
                    None
                }
            })
            .is_ok()
    }

    fn make_err(&self) -> EngineError {
        match self.fault {
            Fault::Transient => EngineError::IoError(std::io::Error::other(
                "synthetic transient shard read failure",
            )),
            Fault::NonRetryable => EngineError::SchemaError {
                column: "synthetic".to_string(),
                reason: "deterministic decode failure".to_string(),
            },
        }
    }
}

// Delegate every method through the `SectionReader` trait explicitly:
// `ScxReader` also has inherent methods with the same names but different
// signatures, so unqualified `self.inner.method()` would resolve to the wrong
// one.
impl SectionReader for FaultInjectingReader {
    fn header(&self) -> &FileHeader {
        SectionReader::header(&self.inner)
    }
    fn catalog(&self) -> &FullCatalog {
        SectionReader::catalog(&self.inner)
    }
    fn read_obs_schema(&self) -> scx_engine::Result<Schema> {
        SectionReader::read_obs_schema(&self.inner)
    }
    fn read_var_schema(&self) -> scx_engine::Result<Schema> {
        SectionReader::read_var_schema(&self.inner)
    }
    fn read_obs(&self) -> scx_engine::Result<RecordBatch> {
        SectionReader::read_obs(&self.inner)
    }
    fn obs_metadata_shard_count(&self) -> usize {
        SectionReader::obs_metadata_shard_count(&self.inner)
    }
    fn read_obs_shard(&self, shard_idx: u32) -> scx_engine::Result<RecordBatch> {
        if shard_idx == self.target_obs_shard && Self::take_failure(&self.obs_fails_remaining) {
            return Err(self.make_err());
        }
        SectionReader::read_obs_shard(&self.inner, shard_idx)
    }
    fn read_var(&self) -> scx_engine::Result<RecordBatch> {
        SectionReader::read_var(&self.inner)
    }
    fn read_obs_predicate_index_bytes(&self) -> scx_engine::Result<Option<Vec<u8>>> {
        SectionReader::read_obs_predicate_index_bytes(&self.inner)
    }
    fn read_var_predicate_index_bytes(&self) -> scx_engine::Result<Option<Vec<u8>>> {
        SectionReader::read_var_predicate_index_bytes(&self.inner)
    }
    fn read_group_index_bytes(&self) -> scx_engine::Result<Option<Vec<u8>>> {
        SectionReader::read_group_index_bytes(&self.inner)
    }
    fn read_deletion_vectors(&self) -> scx_engine::Result<Option<DeletionVectors>> {
        SectionReader::read_deletion_vectors(&self.inner)
    }
    fn read_shard_from_entry(
        &self,
        entry: &FullCatalogEntry,
    ) -> scx_engine::Result<(Vec<i64>, Vec<i32>, Vec<f32>)> {
        if Self::take_failure(&self.x_fails_remaining) {
            return Err(self.make_err());
        }
        SectionReader::read_shard_from_entry(&self.inner, entry)
    }
    /// Draws on the **same** failure credit as the f32 twin, so a scenario
    /// written for one decode shape exercises the retry path of whichever one
    /// the pipeline actually took.
    fn read_shard_from_entry_native(
        &self,
        entry: &FullCatalogEntry,
    ) -> scx_engine::Result<(Vec<i64>, Vec<u32>, scx_codec::ShardValuesNative)> {
        if Self::take_failure(&self.x_fails_remaining) {
            return Err(self.make_err());
        }
        SectionReader::read_shard_from_entry_native(&self.inner, entry)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn n_counts(obs: &RecordBatch) -> Vec<i64> {
    let idx = obs.schema().index_of("n_counts").unwrap();
    let arr = arrow::compute::cast(obs.column(idx), &DataType::Int64).unwrap();
    let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
    (0..a.len()).map(|i| a.value(i)).collect()
}

#[test]
fn query_survives_transient_single_shard_failure() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    // Reference rows (no fault). "B" lives in obs shards 2 and 3.
    let reference = n_counts(
        &QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'B'")
            .unwrap()
            .collect()
            .unwrap()
            .obs,
    );
    assert_eq!(reference, vec![1000, 1001, 1002, 1003]);

    // Obs shard 3 fails ONCE then succeeds on the engine's retry pass.
    let inner = ScxReader::open(&path).unwrap();
    let faulty = FaultInjectingReader::new(inner, 3, 1, Fault::Transient);
    let result = QueryPipeline::from_reader(Box::new(faulty))
        .unwrap()
        .filter_obs("cell_type == 'B'")
        .unwrap()
        .collect()
        .expect("a single transient shard failure must be recovered, not fatal");

    assert_eq!(
        n_counts(&result.obs),
        reference,
        "recovered query must match the no-fault result"
    );
}

#[test]
fn query_fails_on_persistent_shard_failure() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    // Obs shard 3 always fails (more credits than the engine will ever spend).
    let inner = ScxReader::open(&path).unwrap();
    let faulty = FaultInjectingReader::new(inner, 3, usize::MAX, Fault::Transient);
    let err = QueryPipeline::from_reader(Box::new(faulty))
        .unwrap()
        .filter_obs("cell_type == 'B'")
        .unwrap()
        .collect()
        .expect_err("a permanently failing shard must fail the query");

    let msg = err.to_string();
    assert!(
        msg.contains("shard read failed after retry"),
        "error should report the unrecovered shard(s), got: {msg}"
    );
}

#[test]
fn deterministic_decode_error_is_not_retried() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    // A non-retryable (decode) error on obs shard 3: the engine must fail fast
    // and read that shard exactly once (no second attempt).
    let inner = ScxReader::open(&path).unwrap();
    let faulty = Box::new(FaultInjectingReader::new(
        inner,
        3,
        usize::MAX,
        Fault::NonRetryable,
    ));
    let err = QueryPipeline::from_reader(faulty)
        .unwrap()
        .filter_obs("cell_type == 'B'")
        .unwrap()
        .collect()
        .expect_err("a deterministic decode error must propagate");

    // The SchemaError surfaces unchanged (not wrapped in the retry-exhaustion
    // message), proving it was returned immediately without a retry pass.
    let msg = err.to_string();
    assert!(
        msg.contains("deterministic decode failure"),
        "non-retryable error should fast-fail unchanged, got: {msg}"
    );
    assert!(
        !msg.contains("shard read failed after retry"),
        "non-retryable error must not go through the retry pass, got: {msg}"
    );
}

#[test]
fn query_survives_transient_csr_shard_failure() {
    // The CSR X-shard decode fan-out (`read_shard_from_entry`, collect::execute Site
    // 3) is wrapped by the same per-shard retry as the obs path. A transient
    // decode failure must be recovered, not abort the materialize.
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    let reference = n_counts(
        &QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'B'")
            .unwrap()
            .collect()
            .unwrap()
            .obs,
    );

    // Fail the first CSR shard decode once; the engine's retry pass re-reads it.
    let inner = ScxReader::open(&path).unwrap();
    let faulty = FaultInjectingReader::new(inner, 0, 0, Fault::Transient).with_x_fails(1);
    let result = QueryPipeline::from_reader(Box::new(faulty))
        .unwrap()
        .filter_obs("cell_type == 'B'")
        .unwrap()
        .collect()
        .expect("a transient CSR-shard decode failure must be recovered");

    assert_eq!(
        n_counts(&result.obs),
        reference,
        "recovered query must match the no-fault result"
    );
    assert_eq!(
        result.x.n_rows(),
        reference.len(),
        "all matched rows materialized"
    );
}

/// The **native** shard decode retries too.
///
/// `read_shard_from_entry_native` draws on the same failure credit as its f32
/// twin, but nothing exercised that: the dtype-selected collect is the only
/// caller, and no test here used it, so the retry path on the new decode was
/// implemented and unverified.
#[test]
fn typed_query_survives_a_transient_native_shard_failure() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    let reference = n_counts(
        &QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'B'")
            .unwrap()
            .collect()
            .unwrap()
            .obs,
    );

    let mplan = scx_sparse::MaterializePlan {
        container: scx_sparse::Container::Csr,
        data_dtype: scx_sparse::ValueDtype::F64,
        index_dtype: scx_sparse::IndexDtype::I32,
        allow_lossy: false,
    };

    let inner = ScxReader::open(&path).unwrap();
    let faulty = FaultInjectingReader::new(inner, 0, 0, Fault::Transient).with_x_fails(1);
    let result = QueryPipeline::from_reader(Box::new(faulty))
        .unwrap()
        .filter_obs("cell_type == 'B'")
        .unwrap()
        .collect_typed(&mplan)
        .expect("a transient native shard decode failure must be recovered");

    assert_eq!(n_counts(&result.obs), reference);
    assert_eq!(result.x.n_rows(), reference.len());
    assert_eq!(result.x.values.dtype(), scx_sparse::ValueDtype::F64);
}

/// Every shard pruned at Level-1 means **no shard is read at all** — asserted
/// by making every read fail.
///
/// `pushdown_skip.rs` covers "one of two shards is skipped". Nothing covered
/// the all-pruned case, where the query must return an empty result without
/// touching X: the candidate list is empty, so the decode loop never runs and
/// the retry machinery above never gets a chance to mask a read that should
/// not have happened.
///
/// The fault injector is used here as a *detector* rather than as a fault: with
/// `usize::MAX` X-shard failures armed, any decode at all surfaces as an error,
/// so `collect()` succeeding is direct evidence that none occurred. A counter
/// would prove the same thing and would also have to be believed; an injected
/// failure cannot be silently zero.
///
/// `cell_type == 'Ghost'` is absent from the file's complete `cell_type`
/// vocabulary, so both shards' `CategoryBitset` prune.
#[test]
fn a_predicate_matching_no_shard_reads_no_shard() {
    let dir = TempDir::new().unwrap();
    let path = build_sharded_indexed_file(&dir);

    let armed = |expr: &str| {
        let inner = ScxReader::open(&path).unwrap();
        let reader =
            FaultInjectingReader::new(inner, 0, 0, Fault::NonRetryable).with_x_fails(usize::MAX);
        QueryPipeline::from_reader(Box::new(reader))
            .unwrap()
            .filter_obs(expr)
            .unwrap()
    };

    let r = armed("cell_type == 'Ghost'")
        .collect()
        .expect("every shard is pruned, so no X shard may be decoded");
    assert_eq!(r.total_shards, 2);
    assert_eq!(r.skipped_shards, 2, "both shards must be pruned");
    assert_eq!(r.matched_rows, 0);
    assert_eq!(r.x.n_rows(), 0);
    assert_eq!(r.obs.num_rows(), 0, "obs must be filtered to nothing too");

    // count() / exists() take their own paths to the same conclusion.
    assert_eq!(
        armed("cell_type == 'Ghost'").count().unwrap().matched_rows,
        0
    );
    assert!(!armed("cell_type == 'Ghost'").exists().unwrap());

    // Control, in the same armed configuration: a predicate that DOES match
    // must hit the injector. Without this the assertions above pass just as
    // well against an engine that reads nothing ever.
    assert!(
        armed("cell_type == 'A'").collect().is_err(),
        "a matching predicate must decode a shard and so must trip the injector \
         — otherwise the no-read assertions above are vacuous"
    );
}
