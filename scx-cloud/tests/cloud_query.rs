//! End-to-end Phase 7 cloud-query tests.
//!
//! Verify that `QueryPipeline::from_reader(CloudSectionReader)` returns
//! the same `QueryResult` as the local `QueryPipeline::open(path)` for
//! each of the three on-disk layouts that `scx_cloud::open_cloud`
//! resolves:
//!   - exploded `.scxd/` directory (one object per section)
//!   - cloud-optimized packed `.scx` (front catalog)
//!   - non-cloud-optimized packed `.scx` (EOF catalog)
//!
//! Uses `object_store::LocalFileSystem` (resolved automatically when
//! `scx_cloud::open_cloud` is handed a filesystem path) so the tests
//! run without any cloud credentials.

use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{DictionaryArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use scx_cloud::{cloud_optimize, explode, CloudSectionReader};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{QueryPipeline, SectionReader};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

fn test_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
}

fn sample_obs(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T_cell",
            1 => "B_cell",
            _ => "NK_cell",
        })
        .collect();
    RecordBatch::try_new(
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

fn sample_var(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    RecordBatch::try_new(
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
        let (c0, c1) = if col0 < col1 {
            (col0, col1)
        } else {
            (col1, col0)
        };
        indices.push(c0 as u32);
        indices.push(c1 as u32);
        values.push(((row + 1) % 256) as u8);
        values.push(((row + 2) % 256) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

/// Write a multi-shard test SCX file in the given tempdir. Returns its path.
fn write_test_scx(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> PathBuf {
    let path = dir.path().join("test.scx");
    let header = test_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();
    path
}

/// Write a Phase 2 sharded-obs test SCX: obs is emitted as multiple
/// `ObsMetadataShard` sections instead of one `ObsMetadata`. Used to
/// verify the cloud query path assembles sharded obs and returns the
/// same `QueryResult` as the local in-process path.
fn write_sharded_test_scx(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> PathBuf {
    let path = dir.path().join("sharded.scx");
    let header = test_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = sample_obs(n_obs);
    let obs_rows_per_shard = 40usize;
    let mut shard_idx = 0u32;
    let mut off = 0usize;
    while off < n_obs {
        let len = std::cmp::min(obs_rows_per_shard, n_obs - off);
        let chunk = obs.slice(off, len);
        writer
            .write_obs_shard(shard_idx, off as u64, len as u64, n_obs as u64, &chunk)
            .unwrap();
        off += len;
        shard_idx += 1;
    }
    writer.write_var(&sample_var(n_vars)).unwrap();

    let rows_per_shard = 50;
    let mut row_offset = 0;
    while row_offset < n_obs {
        let shard_rows = std::cmp::min(rows_per_shard, n_obs - row_offset);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_offset as u64,
            )
            .unwrap();
        row_offset += shard_rows;
    }
    writer.finish().unwrap();
    path
}

fn build_cloud_pipeline(rt: &Arc<tokio::runtime::Runtime>, url: &str) -> QueryPipeline {
    let reader = rt.block_on(scx_cloud::open_cloud(url)).unwrap();
    let adapter = CloudSectionReader::new(Arc::new(reader), Arc::clone(rt));
    QueryPipeline::from_reader(Box::new(adapter)).unwrap()
}

fn assert_results_match(local: &scx_engine::QueryResult, cloud: &scx_engine::QueryResult) {
    assert_eq!(local.x.n_rows(), cloud.x.n_rows(), "x.n_rows mismatch");
    assert_eq!(local.x.n_cols(), cloud.x.n_cols(), "x.n_cols mismatch");
    assert_eq!(local.x.indptr, cloud.x.indptr, "indptr mismatch");
    assert_eq!(local.x.indices, cloud.x.indices, "indices mismatch");
    assert_eq!(local.x.data, cloud.x.data, "data mismatch");
    assert_eq!(
        local.obs.num_rows(),
        cloud.obs.num_rows(),
        "obs n_rows mismatch"
    );
    assert_eq!(
        local.var.num_rows(),
        cloud.var.num_rows(),
        "var n_rows mismatch"
    );
}

/// Exploded `.scxd/` selective query returns the same result as the
/// local in-process query.
#[test]
fn cloud_query_exploded_matches_local_filter_obs() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 100, 50);

    let exploded_path = dir.path().join("test.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
    assert!(local.x.n_rows() > 0, "filter should keep at least one cell");
}

/// Cloud-optimized packed `.scx` (front catalog) returns the same result
/// as the local in-process query for the same predicate.
#[test]
fn cloud_query_cloud_optimized_packed_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 100, 50);

    let optimized_path = dir.path().join("optimized.scx");
    cloud_optimize(&scx_path, &optimized_path).unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'B_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &optimized_path.to_string_lossy())
        .filter_obs("cell_type == 'B_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
}

/// Non-cloud-optimized packed `.scx` (EOF catalog) returns the same
/// result. Slower in practice (extra HEAD + EOF range read) but
/// functionally identical.
#[test]
fn cloud_query_plain_packed_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 100, 50);

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'NK_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &scx_path.to_string_lossy())
        .filter_obs("cell_type == 'NK_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
}

/// Gene projection via `select_genes` produces the same column subset
/// in both code paths.
#[test]
fn cloud_query_select_genes_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 80, 30);
    let exploded_path = dir.path().join("test.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let gene_indices: Vec<u32> = vec![0, 5, 7, 12, 20];

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .select_genes(gene_indices.clone())
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .select_genes(gene_indices)
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
    assert_eq!(cloud.x.n_cols(), 5);
}

/// `total_shards` and `skipped_shards` accounting works on the cloud
/// path. Predicate indexes are absent on these test fixtures, so
/// skipped_shards is 0 — the assertion is that the COUNT matches
/// (i.e. shard pushdown gives the same answer regardless of backend).
#[test]
fn cloud_query_reports_total_shards() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx(&dir, 200, 20);
    let exploded_path = dir.path().join("test.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_eq!(cloud.total_shards, local.total_shards);
    assert_eq!(cloud.skipped_shards, local.skipped_shards);
    assert_eq!(cloud.total_shards, 4); // 200 cells / 50 per shard
}

// ---------------------------------------------------------------------------
// Fix 3 regression: cloud `read_obs_schema` must narrow
// `Dictionary(Int32, LargeUtf8)` to `Dictionary(Int32, Utf8)` so the schema
// matches what the local `ScxReader::read_obs_schema` returns. Before the
// fix the cloud decoder only narrowed top-level `LargeUtf8` / `LargeBinary`,
// not the value type inside a Dictionary, causing divergence for any
// categorical column in obs.
// ---------------------------------------------------------------------------

/// Write a test SCX file whose obs contains a `Dictionary(Int32, Utf8)`
/// `cell_type` column. The writer upcasts narrow Utf8 → LargeUtf8 (including
/// inside dictionaries) before serialising, so the on-disk obs is
/// `Dictionary(Int32, LargeUtf8)` — exactly the case Fix 3 targets.
fn write_test_scx_with_dictionary_obs(
    dir: &tempfile::TempDir,
    n_obs: usize,
    n_vars: usize,
) -> PathBuf {
    let path = dir.path().join("test_dict.scx");
    let header = test_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let type_strings: Vec<&str> = (0..n_obs)
        .map(|i| match i % 3 {
            0 => "T_cell",
            1 => "B_cell",
            _ => "NK_cell",
        })
        .collect();
    let dict: DictionaryArray<Int32Type> = type_strings.iter().copied().collect();
    let dict_dt = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", dict_dt, false),
    ]);
    let obs = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(dict),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
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

/// Local and cloud obs schemas must agree for a file whose obs holds a
/// `Dictionary(Int32, Utf8)` column (which the writer widens to
/// `Dictionary(Int32, LargeUtf8)` on disk). Pre-fix the cloud reader
/// returned `Dictionary(Int32, LargeUtf8)` while the local reader
/// returned `Dictionary(Int32, Utf8)`.
#[test]
fn cloud_read_obs_schema_narrows_dictionary_large_utf8() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_test_scx_with_dictionary_obs(&dir, 60, 20);

    // Local schema via ScxReader.
    let local_reader = ScxReader::open(&scx_path).unwrap();
    let local_schema = local_reader.read_obs_schema().unwrap();

    // Cloud schema via CloudReader through CloudSectionReader.
    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud_reader = rt
        .block_on(scx_cloud::open_cloud(&scx_path.to_string_lossy()))
        .unwrap();
    let cloud_adapter = CloudSectionReader::new(Arc::new(cloud_reader), Arc::clone(&rt));
    let cloud_schema = cloud_adapter.read_obs_schema().unwrap();

    let expected_dict = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));

    // Both schemas must narrow LargeUtf8 → Utf8 inside the dict.
    let local_cell_type_dt = local_schema
        .field_with_name("cell_type")
        .unwrap()
        .data_type();
    let cloud_cell_type_dt = cloud_schema
        .field_with_name("cell_type")
        .unwrap()
        .data_type();
    assert_eq!(
        local_cell_type_dt, &expected_dict,
        "local schema should narrow dict to Utf8 value type"
    );
    assert_eq!(
        cloud_cell_type_dt, &expected_dict,
        "cloud schema must narrow dict to Utf8 value type (Fix 3)"
    );
    // Field-for-field equality across the rest of the schema.
    assert_eq!(local_schema.fields().len(), cloud_schema.fields().len());
    for (lf, cf) in local_schema
        .fields()
        .iter()
        .zip(cloud_schema.fields().iter())
    {
        assert_eq!(lf.name(), cf.name());
        assert_eq!(lf.data_type(), cf.data_type());
    }
}

/// Sharded-obs exploded `.scxd/` query matches the local in-process
/// query — verifies the cloud reader assembles `ObsMetadataShard`
/// sections rather than failing on the missing single `obs` section.
#[test]
fn cloud_query_sharded_obs_exploded_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_sharded_test_scx(&dir, 100, 50);
    let exploded_path = dir.path().join("sharded.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &exploded_path.to_string_lossy())
        .filter_obs("cell_type == 'T_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
    assert!(local.x.n_rows() > 0, "filter should keep at least one cell");
}

/// Sharded-obs cloud-optimized packed `.scx` (range reads by offset)
/// matches the local query.
#[test]
fn cloud_query_sharded_obs_packed_matches_local() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_sharded_test_scx(&dir, 100, 50);
    let optimized_path = dir.path().join("sharded_optimized.scx");
    cloud_optimize(&scx_path, &optimized_path).unwrap();

    let local = QueryPipeline::open(&scx_path)
        .unwrap()
        .filter_obs("cell_type == 'B_cell'")
        .unwrap()
        .collect()
        .unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    let cloud = build_cloud_pipeline(&rt, &optimized_path.to_string_lossy())
        .filter_obs("cell_type == 'B_cell'")
        .unwrap()
        .collect()
        .unwrap();

    assert_results_match(&local, &cloud);
    assert!(local.x.n_rows() > 0, "filter should keep at least one cell");
}

// ---------------------------------------------------------------------------
// Modality-scoped cloud queries. A 2-modality file (rna: 8 vars,
// adt: 3 vars) where BOTH modalities tile [0, n_obs) (single shard each), so
// the engine's per-modality tiling debug_assert holds. Verifies the cloud
// per-modality var + shard routing matches the local `open_for_modality`
// result across exploded + both packed layouts (the exploded arm also guards
// the `var/{name}` filename-collision fix in explode.rs).
// ---------------------------------------------------------------------------

fn write_multimodal_scx(
    dir: &tempfile::TempDir,
    n_obs: usize,
    rna_vars: usize,
    adt_vars: usize,
) -> PathBuf {
    use scx_format_io::modality::ModalityType;
    let path = dir.path().join("mm.scx");
    let header = test_header(n_obs as u64, rna_vars.max(adt_vars) as u64);
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
    writer.write_var_for(rna_id, &sample_var(rna_vars)).unwrap();
    writer.write_var_for(adt_id, &sample_var(adt_vars)).unwrap();
    writer.set_modality_n_vars(rna_id, rna_vars as u64).unwrap();
    writer.set_modality_n_vars(adt_id, adt_vars as u64).unwrap();

    for (id, m_vars) in [(rna_id, rna_vars), (adt_id, adt_vars)] {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..n_obs {
            indices.push((r % m_vars) as u32);
            values.push(((r + 1) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
        writer
            .write_csr_shard_for(
                id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

fn build_cloud_pipeline_for_modality(
    rt: &Arc<tokio::runtime::Runtime>,
    url: &str,
    modality_id: u8,
) -> QueryPipeline {
    let reader = rt.block_on(scx_cloud::open_cloud(url)).unwrap();
    let adapter = CloudSectionReader::new(Arc::new(reader), Arc::clone(rt));
    QueryPipeline::from_reader_for_modality(Box::new(adapter), modality_id).unwrap()
}

#[test]
fn cloud_modality_query_matches_local_all_layouts() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_multimodal_scx(&dir, 60, 8, 3);

    let exploded_path = dir.path().join("mm.scxd");
    explode(&scx_path, &exploded_path).unwrap();
    let optimized_path = dir.path().join("mm_opt.scx");
    cloud_optimize(&scx_path, &optimized_path).unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());

    // rna is modality 1, adt is modality 2.
    for (modality_id, expect_vars) in [(1u8, 8usize), (2u8, 3usize)] {
        let local = QueryPipeline::open_for_modality(&scx_path, modality_id)
            .unwrap()
            .filter_obs("cell_type == 'T_cell'")
            .unwrap()
            .collect()
            .unwrap();
        assert_eq!(local.x.n_cols(), expect_vars, "local per-modality width");

        for url in [
            exploded_path.to_string_lossy().to_string(),
            optimized_path.to_string_lossy().to_string(),
            scx_path.to_string_lossy().to_string(),
        ] {
            let cloud = build_cloud_pipeline_for_modality(&rt, &url, modality_id)
                .filter_obs("cell_type == 'T_cell'")
                .unwrap()
                .collect()
                .unwrap();
            assert_eq!(
                cloud.x.n_cols(),
                expect_vars,
                "cloud modality {modality_id} width for {url}"
            );
            assert_results_match(&local, &cloud);
            assert!(cloud.x.n_rows() > 0, "filter should keep cells ({url})");
        }
    }
}

#[test]
fn cloud_modality_accessors_and_unknown_modality() {
    let dir = tempfile::tempdir().unwrap();
    let scx_path = write_multimodal_scx(&dir, 30, 8, 3);
    let exploded_path = dir.path().join("mm.scxd");
    explode(&scx_path, &exploded_path).unwrap();

    let rt = Arc::new(tokio::runtime::Runtime::new().unwrap());
    for url in [
        exploded_path.to_string_lossy().to_string(),
        scx_path.to_string_lossy().to_string(),
    ] {
        let reader = rt.block_on(scx_cloud::open_cloud(&url)).unwrap();
        let adapter = CloudSectionReader::new(Arc::new(reader), Arc::clone(&rt));
        assert!(adapter.is_multimodal(), "{url}");
        assert_eq!(adapter.n_modalities(), 2, "{url}");
        let mut names = adapter.modality_names();
        names.sort();
        assert_eq!(names, vec!["adt".to_string(), "rna".to_string()], "{url}");
        assert_eq!(adapter.modality_id_by_name("rna"), Some(1), "{url}");
        assert_eq!(adapter.modality_id_by_name("nope"), None, "{url}");

        // Unknown modality id → engine UnknownModality at construction.
        let reader2 = rt.block_on(scx_cloud::open_cloud(&url)).unwrap();
        let adapter2 = CloudSectionReader::new(Arc::new(reader2), Arc::clone(&rt));
        let err = QueryPipeline::from_reader_for_modality(Box::new(adapter2), 99)
            .err()
            .expect("unknown modality should error");
        assert!(
            matches!(err, scx_engine::EngineError::UnknownModality { .. }),
            "expected UnknownModality, got {err:?} ({url})"
        );
    }
}
