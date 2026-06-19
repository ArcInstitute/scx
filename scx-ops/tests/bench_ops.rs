/// Benchmark tests for scx-ops file operations (Phase 2 Step 3).
///
/// These tests are marked `#[ignore]` so they don't run in normal `cargo test`.
/// Run with: `cargo test -p scx-ops bench_ops_ --release -- --nocapture --ignored`
///
/// Each test prints a JSON line with benchmark results that the Python
/// script (`benchmarks/scripts/benchmark_ops.py`) parses into a report.
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_ops::AppendOptions;
use std::sync::Arc;
use std::time::Instant;
use tempfile::TempDir;

fn sample_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        n_obs,
        n_vars,
        shard_target_rows: 16384,
        index_dtype: if n_vars <= 65535 { 0 } else { 1 }, // u16 or u32
        ..Default::default()
    }
}

fn sample_obs(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn sample_var(n: usize) -> arrow::array::RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
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

fn write_bench_file(
    dir: &TempDir,
    filename: &str,
    n_obs: usize,
    n_vars: usize,
) -> std::path::PathBuf {
    let path = dir.path().join(filename);
    let header = sample_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    // Write in shards of 16384
    let shard_size = 16384;
    let mut row_start = 0usize;
    while row_start < n_obs {
        let shard_rows = std::cmp::min(shard_size, n_obs - row_start);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
        row_start += shard_rows;
    }

    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp: 1710000000,
            action: "convert".to_string(),
            tool: "bench".to_string(),
            params_json: "{}".to_string(),
            input_checksums: vec![],
        }])
        .unwrap();

    writer.finish().unwrap();
    path
}

const BENCH_CELLS: usize = 10_000;
const BENCH_GENES: usize = 20_000;

#[test]
#[ignore]
fn bench_ops_append_10k() {
    let dir = tempfile::tempdir().unwrap();

    // Create fresh file with 10K cells for size comparison
    let fresh_path = write_bench_file(&dir, "fresh_20k.scx", BENCH_CELLS * 2, BENCH_GENES);
    let fresh_size = std::fs::metadata(&fresh_path).unwrap().len();

    // Create base file with 10K cells
    let path = write_bench_file(&dir, "base.scx", BENCH_CELLS, BENCH_GENES);
    let base_size = std::fs::metadata(&path).unwrap().len();

    // Append 10K cells and measure time
    let new_obs = sample_obs(BENCH_CELLS);
    let (indptr, indices, values) = sample_shard_data(BENCH_CELLS, BENCH_GENES);

    let start = Instant::now();
    scx_ops::append(
        &path,
        &new_obs,
        &indptr,
        &indices,
        &values,
        ValueEncoding::Uint8,
        &AppendOptions::default(),
    )
    .unwrap();
    let elapsed = start.elapsed();

    let append_size = std::fs::metadata(&path).unwrap().len();

    // Verify correctness
    let reader = ScxReader::open(&path).unwrap();
    assert_eq!(reader.n_obs(), (BENCH_CELLS * 2) as u64);

    let bloat_pct = if fresh_size > 0 {
        ((append_size as f64 - fresh_size as f64) / fresh_size as f64) * 100.0
    } else {
        0.0
    };

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "append_10k",
            "time_ms": elapsed.as_millis() as f64,
            "base_size_bytes": base_size,
            "fresh_size_bytes": fresh_size,
            "append_size_bytes": append_size,
            "bloat_pct": bloat_pct,
            "n_cells_appended": BENCH_CELLS,
        })
    );
}

#[test]
#[ignore]
fn bench_ops_compact_after_3_appends() {
    let dir = tempfile::tempdir().unwrap();

    // Create base file
    let path = write_bench_file(&dir, "compact_base.scx", BENCH_CELLS, BENCH_GENES);

    // Append 3 times (3333 cells each ≈ 10K total)
    let append_size = 3334;
    for i in 0..3 {
        let rows = if i == 2 { 3332 } else { append_size };
        let new_obs = sample_obs(rows);
        let (indptr, indices, values) = sample_shard_data(rows, BENCH_GENES);
        scx_ops::append(
            &path,
            &new_obs,
            &indptr,
            &indices,
            &values,
            ValueEncoding::Uint8,
            &AppendOptions::default(),
        )
        .unwrap();
    }

    let after_appends = std::fs::metadata(&path).unwrap().len();

    // Compact
    let compact_path = dir.path().join("compacted.scx");
    let start = Instant::now();
    scx_ops::compact(&path, &compact_path).unwrap();
    let compact_elapsed = start.elapsed();

    let after_compact = std::fs::metadata(&compact_path).unwrap().len();

    // Fresh equivalent: write 20K cells from scratch
    let fresh_path = write_bench_file(&dir, "fresh_20k.scx", BENCH_CELLS * 2, BENCH_GENES);
    let fresh_size = std::fs::metadata(&fresh_path).unwrap().len();

    let compact_vs_fresh = if fresh_size > 0 {
        after_compact as f64 / fresh_size as f64
    } else {
        0.0
    };

    // Verify correctness
    let reader = ScxReader::open(&compact_path).unwrap();
    assert_eq!(reader.n_obs(), (BENCH_CELLS * 2) as u64);

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "compact_after_3_appends",
            "compact_time_ms": compact_elapsed.as_millis() as f64,
            "after_appends_bytes": after_appends,
            "after_compact_bytes": after_compact,
            "fresh_size_bytes": fresh_size,
            "compact_vs_fresh": compact_vs_fresh,
        })
    );
}

#[test]
#[ignore]
fn bench_ops_dv_read_overhead() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bench_file(&dir, "dv_bench.scx", BENCH_CELLS, BENCH_GENES);

    // Warmup: read once
    {
        let reader = ScxReader::open(&path).unwrap();
        let _ = reader.read_all_csr_shards().unwrap();
    }

    // Measure read without DVs (average of 5 runs)
    let mut no_dv_times = Vec::new();
    for _ in 0..5 {
        let reader = ScxReader::open(&path).unwrap();
        let start = Instant::now();
        let _ = reader.read_all_csr_shards().unwrap();
        no_dv_times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let avg_no_dv = no_dv_times.iter().sum::<f64>() / no_dv_times.len() as f64;

    // Mark 10% as deleted
    let delete_indices: Vec<u64> = (0..BENCH_CELLS as u64).step_by(10).collect();
    scx_ops::mark_deleted(&path, &delete_indices).unwrap();

    // Warmup
    {
        let reader = ScxReader::open(&path).unwrap();
        let _ = reader.read_all_csr_shards_filtered().unwrap();
    }

    // Measure read with DVs (average of 5 runs)
    let mut dv_times = Vec::new();
    for _ in 0..5 {
        let reader = ScxReader::open(&path).unwrap();
        let start = Instant::now();
        let csr = reader.read_all_csr_shards_filtered().unwrap();
        dv_times.push(start.elapsed().as_secs_f64() * 1000.0);
        assert_eq!(
            csr.shape.0,
            BENCH_CELLS - delete_indices.len(),
            "filtered read should exclude deleted cells"
        );
    }
    let avg_dv = dv_times.iter().sum::<f64>() / dv_times.len() as f64;

    let overhead_pct = if avg_no_dv > 0.0 {
        ((avg_dv - avg_no_dv) / avg_no_dv) * 100.0
    } else {
        0.0
    };

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "dv_read_overhead",
            "read_no_dv_ms": avg_no_dv,
            "read_dv_ms": avg_dv,
            "overhead_pct": overhead_pct,
            "deleted_fraction": 0.1,
            "n_deleted": delete_indices.len(),
        })
    );
}

#[test]
#[ignore]
fn bench_ops_merge_throughput() {
    let dir = tempfile::tempdir().unwrap();

    // Create 3 input files
    let path1 = write_bench_file(&dir, "merge1.scx", BENCH_CELLS, BENCH_GENES);
    let path2 = write_bench_file(&dir, "merge2.scx", BENCH_CELLS, BENCH_GENES);
    let path3 = write_bench_file(&dir, "merge3.scx", BENCH_CELLS, BENCH_GENES);

    let size1 = std::fs::metadata(&path1).unwrap().len();
    let size2 = std::fs::metadata(&path2).unwrap().len();
    let size3 = std::fs::metadata(&path3).unwrap().len();
    let total_input_bytes = size1 + size2 + size3;

    let output = dir.path().join("merged.scx");

    let start = Instant::now();
    scx_ops::merge(
        &[path1.as_path(), path2.as_path(), path3.as_path()],
        &output,
    )
    .unwrap();
    let elapsed = start.elapsed();

    // Verify correctness
    let reader = ScxReader::open(&output).unwrap();
    assert_eq!(reader.n_obs(), (BENCH_CELLS * 3) as u64);

    let throughput_mb_s = if elapsed.as_secs_f64() > 0.0 {
        (total_input_bytes as f64 / (1024.0 * 1024.0)) / elapsed.as_secs_f64()
    } else {
        0.0
    };

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "merge_throughput",
            "time_ms": elapsed.as_millis() as f64,
            "total_input_mb": total_input_bytes as f64 / (1024.0 * 1024.0),
            "output_size_bytes": std::fs::metadata(&output).unwrap().len(),
            "throughput_mb_s": throughput_mb_s,
            "n_inputs": 3,
        })
    );
}

/// Write a categorical-key bench file: `cell_type` cycles over `k`
/// categories so every input CSR shard contains every category (worst-case
/// locality before sort). `shard_size` controls the shard count.
fn write_categorical_bench_file(
    dir: &TempDir,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    k: usize,
    shard_size: usize,
) -> std::path::PathBuf {
    use arrow::datatypes::Field;
    let path = dir.path().join(name);
    let mut header = sample_header(n_obs as u64, n_vars as u64);
    header.shard_target_rows = shard_size as u32;
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let types: Vec<String> = (0..n_obs).map(|i| format!("ct_{}", i % k)).collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let obs = arrow::array::RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                types.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();

    let mut row_start = 0usize;
    while row_start < n_obs {
        let shard_rows = std::cmp::min(shard_size, n_obs - row_start);
        let (indptr, indices, values) = sample_shard_data(shard_rows, n_vars);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
        row_start += shard_rows;
    }
    writer.finish().unwrap();
    path
}

/// Count the distinct CSR shards that hold at least one row of `target`
/// (the X-shard fetch count for an "all cells of category X" scan).
fn shards_touching_category(path: &std::path::Path, target: &str) -> usize {
    use arrow::array::Array;
    let r = ScxReader::open(path).unwrap();
    let obs = r.read_obs().unwrap();
    let col = obs.column_by_name("cell_type").unwrap();
    let utf8 = arrow::compute::cast(col, &DataType::Utf8).unwrap();
    let ct = utf8.as_any().downcast_ref::<StringArray>().unwrap();

    let mut ranges: Vec<(u64, u64)> = r
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format_io::section::SectionType::CsrShard)
        .filter_map(|e| e.stats.as_ref().map(|s| (s.row_start, s.row_end)))
        .collect();
    ranges.sort_by_key(|(s, _)| *s);

    ranges
        .iter()
        .filter(|(s, e)| {
            (*s..*e).any(|i| !ct.is_null(i as usize) && ct.value(i as usize) == target)
        })
        .count()
}

/// Locality benchmark (SCX-SORT-SPEC §10 / §13 T4 gate): X-shard fetch count
/// for an "all cells of one category" scan *before* vs *after* `scx sort`.
/// Expectation: O(shards) → O(few) once a category is physically contiguous.
#[test]
#[ignore]
fn bench_ops_sort_locality() {
    let dir = tempfile::tempdir().unwrap();
    let (n_obs, k, shard_size) = (BENCH_CELLS * 4, 8usize, 4096usize);
    let path =
        write_categorical_bench_file(&dir, "sort_locality.scx", n_obs, BENCH_GENES, k, shard_size);

    let target = "ct_3";
    let before = shards_touching_category(&path, target);

    // Sort by cell_type (in-memory; no budget) and re-shard at the same size.
    let sorted = dir.path().join("sort_locality.sorted.scx");
    let opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        shard_target_rows: shard_size as u32,
        ..Default::default()
    };
    let start = Instant::now();
    let summary = scx_ops::sort(&path, &sorted, &opts).unwrap();
    let elapsed = start.elapsed();

    let after = shards_touching_category(&sorted, target);
    let ratio = before as f64 / after.max(1) as f64;

    assert!(
        after <= before,
        "sort must not increase the per-category shard fetch count"
    );

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "sort_locality",
            "n_obs": n_obs as u64,
            "k_categories": k,
            "shard_size": shard_size,
            "shards_before": before,
            "shards_after": after,
            "locality_ratio": ratio,
            "sort_time_ms": elapsed.as_millis() as f64,
            "n_output_shards": summary.n_output_shards,
        })
    );
}
