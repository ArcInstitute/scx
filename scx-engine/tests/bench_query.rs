/// Benchmark tests for scx-engine query engine (Phase 2 Step 4, Phase H).
///
/// These tests are marked `#[ignore]` so they don't run in normal `cargo test`.
/// Run with: `cargo test -p scx-engine bench_query_ --release -- --nocapture --ignored`
///
/// Each test prints a JSON line with benchmark results that the Python
/// script (`benchmarks/scripts/benchmark_query.py`) parses into a report.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{build_indexes, QueryPipeline};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::writer::ScxWriter;
use tempfile::TempDir;

// ============================================================================
// Benchmark fixture: 100K cells × 30K genes, 7 shards
// ============================================================================

const BENCH_CELLS: usize = 100_000;
const BENCH_GENES: usize = 30_000;
const ROWS_PER_SHARD: usize = 15_000; // ~7 shards for 100K cells
const NNZ_PER_ROW: usize = 5;

/// 7 cell types with non-uniform distribution across shards.
/// This ensures some cell types are absent from some shards, enabling pushdown.
///
/// Distribution:
///   Shard 0 (rows     0-14999): T cell, B cell, NK cell (no Monocyte, Dendritic, Macrophage, Epithelial)
///   Shard 1 (rows 15000-29999): B cell, NK cell, Monocyte (no T cell)
///   Shard 2 (rows 30000-44999): T cell, Monocyte, Dendritic (no B cell, NK cell)
///   Shard 3 (rows 45000-59999): Macrophage, Epithelial, NK cell (no T cell, B cell, Monocyte)
///   Shard 4 (rows 60000-74999): T cell, B cell, Dendritic (no NK cell, Monocyte)
///   Shard 5 (rows 75000-89999): Monocyte, Macrophage, Epithelial (no T cell, B cell, NK cell)
///   Shard 6 (rows 90000-99999): T cell, B cell, NK cell, Monocyte (all common types)
const CELL_TYPES: [&str; 7] = [
    "T cell",
    "B cell",
    "NK cell",
    "Monocyte",
    "Dendritic",
    "Macrophage",
    "Epithelial",
];

fn cell_type_for_row(i: usize) -> &'static str {
    let shard = i / ROWS_PER_SHARD;
    let local = i % ROWS_PER_SHARD;
    match shard {
        0 => match local % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        },
        1 => match local % 3 {
            0 => "B cell",
            1 => "NK cell",
            _ => "Monocyte",
        },
        2 => match local % 3 {
            0 => "T cell",
            1 => "Monocyte",
            _ => "Dendritic",
        },
        3 => match local % 3 {
            0 => "Macrophage",
            1 => "Epithelial",
            _ => "NK cell",
        },
        4 => match local % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "Dendritic",
        },
        5 => match local % 3 {
            0 => "Monocyte",
            1 => "Macrophage",
            _ => "Epithelial",
        },
        // Shard 6 (remaining rows): all common types
        _ => match local % 4 {
            0 => "T cell",
            1 => "B cell",
            2 => "NK cell",
            _ => "Monocyte",
        },
    }
}

fn n_genes_for_row(i: usize) -> i32 {
    100 + ((i * 4907) % 4901) as i32
}

fn make_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs,
        n_vars,
        nnz: 0,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: ROWS_PER_SHARD as u32,
        codec_id: 0,
        index_dtype: if n_vars <= 65535 { 0 } else { 1 },
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

fn build_bench_obs(n: usize) -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
        Field::new("n_genes", DataType::Int32, false),
    ]);
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let types: Vec<&str> = (0..n).map(cell_type_for_row).collect();
    let n_genes: Vec<i32> = (0..n).map(n_genes_for_row).collect();

    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
            Arc::new(Int32Array::from(n_genes)),
        ],
    )
    .unwrap()
}

fn build_var(n: usize) -> RecordBatch {
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

fn shard_data(
    n_rows: usize,
    n_vars: usize,
    row_offset: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::with_capacity(n_rows * NNZ_PER_ROW);
    let mut values = Vec::with_capacity(n_rows * NNZ_PER_ROW);

    for row in 0..n_rows {
        let global = row_offset + row;
        // Generate NNZ_PER_ROW non-zero entries with distinct sorted column indices
        let mut cols: Vec<u32> = (0..NNZ_PER_ROW)
            .map(|k| ((global * 7 + k * 3571) % n_vars) as u32)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for &c in &cols {
            indices.push(c);
            values.push(((global + c as usize + 1) % 255 + 1) as u8);
        }
        indptr.push(indptr.last().unwrap() + cols.len() as u64);
    }

    (indptr, indices, values)
}

/// Write the benchmark fixture file with predicate index and per-shard column stats.
fn write_bench_file(dir: &TempDir) -> PathBuf {
    use scx_format::catalog::{column_name_hash, ColumnStat};
    use std::collections::BTreeSet;

    let path = dir.path().join("bench_query.scx");
    let header = make_header(BENCH_CELLS as u64, BENCH_GENES as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = build_bench_obs(BENCH_CELLS);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&build_var(BENCH_GENES)).unwrap();

    // Collect the sorted set of all cell types for the global dictionary.
    // Bit position i in the CategoryBitset = the i-th value in this sorted list.
    let mut all_types: BTreeSet<&str> = BTreeSet::new();
    for &ct in &CELL_TYPES {
        all_types.insert(ct);
    }
    let sorted_types: Vec<&str> = all_types.into_iter().collect();
    let cell_type_hash = column_name_hash("cell_type");

    // Write shards, computing CategoryBitset column stats for each
    let mut row_start = 0usize;
    let mut shard_row_ranges: Vec<(u64, u64)> = Vec::new();
    while row_start < BENCH_CELLS {
        let shard_rows = std::cmp::min(ROWS_PER_SHARD, BENCH_CELLS - row_start);
        let (indptr, indices, values) = shard_data(shard_rows, BENCH_GENES, row_start);
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

        // Compute which cell types are present in this shard's row range
        let mut present: BTreeSet<&str> = BTreeSet::new();
        for i in row_start..(row_start + shard_rows) {
            present.insert(cell_type_for_row(i));
        }

        // Build the CategoryBitset: bit i set if sorted_types[i] is present
        let n_bytes = (sorted_types.len() + 7) / 8;
        let mut bitset = vec![0u8; n_bytes];
        for (i, &ct) in sorted_types.iter().enumerate() {
            if present.contains(ct) {
                bitset[i / 8] |= 1 << (i % 8);
            }
        }

        writer.set_shard_column_stats(vec![ColumnStat::CategoryBitset {
            column_name_hash: cell_type_hash,
            bitset,
        }]);

        shard_row_ranges.push((row_start as u64, (row_start + shard_rows) as u64));
        row_start += shard_rows;
    }

    // Build and write predicate index on cell_type
    let pred_index =
        build_indexes(&obs, &shard_row_ranges, &["cell_type".to_string()]).unwrap();
    let mut index_bytes = Vec::new();
    pred_index.write_to(&mut index_bytes).unwrap();
    writer.write_obs_predicate_index(&index_bytes).unwrap();

    writer.finish().unwrap();
    path
}

/// Compute percentile from sorted values (0-based, linear interpolation).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64)
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&sorted, 50.0)
}

// ============================================================================
// H2. Shard skip rate benchmark
// ============================================================================

#[test]
#[ignore]
fn bench_query_shard_skip_rate() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bench_file(&dir);

    let mut results = Vec::new();

    for &cell_type in &CELL_TYPES {
        let expr = format!("cell_type == '{cell_type}'");
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs(&expr)
            .unwrap()
            .collect()
            .unwrap();

        let skip_rate = if result.total_shards > 0 {
            result.skipped_shards as f64 / result.total_shards as f64 * 100.0
        } else {
            0.0
        };

        // Count expected cells for this type
        let expected_cells: usize = (0..BENCH_CELLS)
            .filter(|&i| cell_type_for_row(i) == cell_type)
            .count();

        results.push(serde_json::json!({
            "cell_type": cell_type,
            "total_shards": result.total_shards,
            "skipped_shards": result.skipped_shards,
            "skip_rate_pct": skip_rate,
            "returned_cells": result.x.n_rows(),
            "expected_cells": expected_cells,
        }));
    }

    // Compute average skip rate across cell types
    let avg_skip_rate: f64 = results
        .iter()
        .map(|r| r["skip_rate_pct"].as_f64().unwrap())
        .sum::<f64>()
        / results.len() as f64;

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "shard_skip_rate",
            "per_cell_type": results,
            "avg_skip_rate_pct": avg_skip_rate,
            "pass": avg_skip_rate >= 50.0,
        })
    );
}

// ============================================================================
// H3. Query latency benchmark
// ============================================================================

#[test]
#[ignore]
fn bench_query_latency() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bench_file(&dir);
    let n_runs = 10;

    // Warmup
    let _ = QueryPipeline::open(&path)
        .unwrap()
        .filter_obs("cell_type == 'Dendritic'")
        .unwrap()
        .collect()
        .unwrap();

    // Selective query: Dendritic cells (present in only 2 shards)
    let mut selective_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'Dendritic'")
            .unwrap()
            .collect()
            .unwrap();
        selective_times.push(start.elapsed().as_secs_f64());
        assert!(result.x.n_rows() > 0);
    }

    // Broad query: T cell (present in many shards)
    let mut broad_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'T cell'")
            .unwrap()
            .collect()
            .unwrap();
        broad_times.push(start.elapsed().as_secs_f64());
        assert!(result.x.n_rows() > 0);
    }

    let mut sel_sorted = selective_times.clone();
    sel_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut broad_sorted = broad_times.clone();
    broad_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "query_latency",
            "selective_query": {
                "predicate": "cell_type == 'Dendritic'",
                "n_runs": n_runs,
                "median_s": median(&selective_times),
                "p95_s": percentile(&sel_sorted, 95.0),
                "p99_s": percentile(&sel_sorted, 99.0),
                "min_s": sel_sorted[0],
                "max_s": sel_sorted[sel_sorted.len() - 1],
                "target_s": 5.0,
                "pass": median(&selective_times) < 5.0,
            },
            "broad_query": {
                "predicate": "cell_type == 'T cell'",
                "n_runs": n_runs,
                "median_s": median(&broad_times),
                "p95_s": percentile(&broad_sorted, 95.0),
                "p99_s": percentile(&broad_sorted, 99.0),
                "min_s": broad_sorted[0],
                "max_s": broad_sorted[broad_sorted.len() - 1],
                "target_s": 15.0,
                "pass": median(&broad_times) < 15.0,
            },
        })
    );
}

// ============================================================================
// H4. Query vs full-read comparison
// ============================================================================

#[test]
#[ignore]
fn bench_query_vs_full_read() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bench_file(&dir);
    let n_runs = 5;

    // Warmup
    let _ = QueryPipeline::open(&path).unwrap().collect().unwrap();

    // Method 1: SCX query with pushdown
    let mut query_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let result = QueryPipeline::open(&path)
            .unwrap()
            .filter_obs("cell_type == 'Dendritic'")
            .unwrap()
            .collect()
            .unwrap();
        query_times.push(start.elapsed().as_secs_f64());
        assert!(result.x.n_rows() > 0);
    }

    // Method 2: Read everything then filter in-memory (simulates AnnData pattern)
    let mut full_read_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        // Read all data (no predicates)
        let full_result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        // Manually filter obs to find matching cells
        let types_col = full_result
            .obs
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let _matching: Vec<usize> = (0..types_col.len())
            .filter(|&i| types_col.value(i) == "Dendritic")
            .collect();
        full_read_times.push(start.elapsed().as_secs_f64());
    }

    let query_median = median(&query_times);
    let full_median = median(&full_read_times);
    let speedup = if query_median > 0.0 {
        full_median / query_median
    } else {
        0.0
    };

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "query_vs_full_read",
            "predicate": "cell_type == 'Dendritic'",
            "n_runs": n_runs,
            "query_median_s": query_median,
            "full_read_median_s": full_median,
            "speedup": speedup,
            "query_wins": query_median < full_median,
        })
    );
}

// ============================================================================
// H5. Gene projection speedup
// ============================================================================

#[test]
#[ignore]
fn bench_query_gene_projection() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_bench_file(&dir);
    let n_runs = 5;

    // Select 2000 HVG indices (evenly spaced across 30K genes)
    let n_hvg = 2000;
    let hvg_indices: Vec<u32> = (0..n_hvg)
        .map(|i| (i as u64 * BENCH_GENES as u64 / n_hvg as u64) as u32)
        .collect();

    // Warmup
    let _ = QueryPipeline::open(&path).unwrap().collect().unwrap();

    // All genes
    let mut all_genes_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let result = QueryPipeline::open(&path).unwrap().collect().unwrap();
        all_genes_times.push(start.elapsed().as_secs_f64());
        assert_eq!(result.x.n_cols(), BENCH_GENES);
    }

    // Projected genes
    let mut proj_times = Vec::new();
    for _ in 0..n_runs {
        let start = Instant::now();
        let result = QueryPipeline::open(&path)
            .unwrap()
            .select_genes(hvg_indices.clone())
            .collect()
            .unwrap();
        proj_times.push(start.elapsed().as_secs_f64());
        assert_eq!(result.x.n_cols(), n_hvg);
    }

    let all_median = median(&all_genes_times);
    let proj_median = median(&proj_times);
    let speedup = if proj_median > 0.0 {
        all_median / proj_median
    } else {
        0.0
    };
    let data_reduction = BENCH_GENES as f64 / n_hvg as f64;

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "gene_projection",
            "n_genes_total": BENCH_GENES,
            "n_genes_projected": n_hvg,
            "data_reduction_ratio": data_reduction,
            "n_runs": n_runs,
            "all_genes_median_s": all_median,
            "projected_median_s": proj_median,
            "speedup": speedup,
        })
    );
}

// ============================================================================
// H6. Fused vs sequential normalize+log1p (CSR, isolated compute)
// ============================================================================

const FUSED_NNZ_PER_ROW: usize = 200;

/// Like `shard_data()` but with higher NNZ per row for fused ops benchmark.
fn shard_data_dense(
    n_rows: usize,
    n_vars: usize,
    row_offset: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let mut indices = Vec::with_capacity(n_rows * FUSED_NNZ_PER_ROW);
    let mut values = Vec::with_capacity(n_rows * FUSED_NNZ_PER_ROW);

    for row in 0..n_rows {
        let global = row_offset + row;
        let mut cols: Vec<u32> = (0..FUSED_NNZ_PER_ROW)
            .map(|k| ((global * 7 + k * 149) % n_vars) as u32)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for &c in &cols {
            indices.push(c);
            values.push(((global + c as usize + 1) % 255 + 1) as u8);
        }
        indptr.push(indptr.last().unwrap() + cols.len() as u64);
    }

    (indptr, indices, values)
}

/// Write a benchmark fixture with ~200 NNZ/row for fused ops benchmark.
/// Total CSR data: 100K × 200 × 12 bytes ≈ 230 MB (exceeds L3 cache).
fn write_fused_bench_file(dir: &TempDir) -> PathBuf {
    let path = dir.path().join("bench_fused.scx");
    let header = make_header(BENCH_CELLS as u64, BENCH_GENES as u64);
    let mut writer = ScxWriter::new(&path, header).unwrap();

    let obs = build_bench_obs(BENCH_CELLS);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&build_var(BENCH_GENES)).unwrap();

    let mut row_start = 0usize;
    while row_start < BENCH_CELLS {
        let shard_rows = std::cmp::min(ROWS_PER_SHARD, BENCH_CELLS - row_start);
        let (indptr, indices, values) =
            shard_data_dense(shard_rows, BENCH_GENES, row_start);
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

#[test]
#[ignore]
fn bench_query_fused_ops() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fused_bench_file(&dir);
    let n_runs = 5;

    // Read once (not timed) — isolate compute from I/O
    let base_result = QueryPipeline::open(&path).unwrap().collect().unwrap();
    let total_nnz = base_result.x.data.len();
    let data_mb = (total_nnz * 4 + (base_result.x.indptr.len() * 8) + (total_nnz * 4)) as f64
        / (1024.0 * 1024.0);

    // Warmup: apply fused once to fault in pages
    {
        let mut warmup = base_result.x.clone();
        scx_engine::apply_fused_ops(&mut warmup, Some(1e4), true);
    }

    // Fused: single-pass normalize+log1p on in-memory CSR
    let mut fused_times = Vec::new();
    for _ in 0..n_runs {
        let mut csr = base_result.x.clone();
        let start = Instant::now();
        scx_engine::apply_fused_ops(&mut csr, Some(1e4), true);
        fused_times.push(start.elapsed().as_secs_f64());
    }

    // Sequential: normalize pass, then log1p pass on in-memory CSR
    let mut sequential_times = Vec::new();
    for _ in 0..n_runs {
        let mut csr = base_result.x.clone();
        let start = Instant::now();
        scx_engine::apply_fused_ops(&mut csr, Some(1e4), false); // normalize only
        scx_engine::apply_fused_ops(&mut csr, None, true); // log1p only
        sequential_times.push(start.elapsed().as_secs_f64());
    }

    let fused_median = median(&fused_times);
    let sequential_median = median(&sequential_times);
    let speedup = if fused_median > 0.0 {
        sequential_median / fused_median
    } else {
        0.0
    };

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "fused_ops",
            "n_rows": BENCH_CELLS,
            "nnz_per_row": FUSED_NNZ_PER_ROW,
            "total_nnz": total_nnz,
            "data_mb": format!("{data_mb:.1}"),
            "n_runs": n_runs,
            "fused_median_s": fused_median,
            "sequential_median_s": sequential_median,
            "speedup": speedup,
            "target_speedup": 1.3,
            "pass": speedup >= 1.0,
        })
    );
}

// ============================================================================
// H6b. Fused vs sequential normalize+log1p (dense rows, loader hot path)
// ============================================================================

/// Normalize a dense row to target sum (benchmark-local copy of scx-loader fn).
fn bench_normalize_dense_row(row: &mut [f32], target_sum: f64) {
    let row_sum: f64 = row.iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in row.iter_mut() {
            *v = (*v as f64 * factor) as f32;
        }
    }
}

/// Apply ln(x+1) to a dense row (benchmark-local copy of scx-loader fn).
fn bench_log1p_dense_row(row: &mut [f32]) {
    for v in row.iter_mut() {
        *v = (*v + 1.0).ln();
    }
}

/// Fused normalize+log1p on a dense row (benchmark-local copy of scx-loader fn).
fn bench_fused_normalize_log1p_dense(row: &mut [f32], target_sum: f64) {
    let row_sum: f64 = row.iter().map(|&v| v as f64).sum();
    if row_sum > 0.0 {
        let factor = target_sum / row_sum;
        for v in row.iter_mut() {
            *v = ((*v as f64 * factor) as f32 + 1.0).ln();
        }
    }
}

#[test]
#[ignore]
fn bench_query_fused_ops_dense() {
    let n_rows = 1024;
    let n_cols = 30_000;
    let n_runs = 5;
    let data_mb = (n_rows * n_cols * 4) as f64 / (1024.0 * 1024.0);

    // Generate realistic dense data: sparse-like with ~200 non-zeros per row,
    // rest zeros. Mimics post-scatter output in the loader.
    let mut base_data = vec![0.0f32; n_rows * n_cols];
    for row in 0..n_rows {
        for k in 0..200 {
            let col = (row * 7 + k * 149) % n_cols;
            base_data[row * n_cols + col] = ((row + col + 1) % 255 + 1) as f32;
        }
    }

    // Warmup
    {
        let mut warmup = base_data.clone();
        for row in warmup.chunks_mut(n_cols) {
            bench_fused_normalize_log1p_dense(row, 1e4);
        }
    }

    // Fused: single pass per row
    let mut fused_times = Vec::new();
    for _ in 0..n_runs {
        let mut data = base_data.clone();
        let start = Instant::now();
        for row in data.chunks_mut(n_cols) {
            bench_fused_normalize_log1p_dense(row, 1e4);
        }
        fused_times.push(start.elapsed().as_secs_f64());
    }

    // Sequential: normalize all rows, then log1p all rows (two full passes)
    let mut sequential_times = Vec::new();
    for _ in 0..n_runs {
        let mut data = base_data.clone();
        let start = Instant::now();
        for row in data.chunks_mut(n_cols) {
            bench_normalize_dense_row(row, 1e4);
        }
        for row in data.chunks_mut(n_cols) {
            bench_log1p_dense_row(row);
        }
        sequential_times.push(start.elapsed().as_secs_f64());
    }

    let fused_median = median(&fused_times);
    let sequential_median = median(&sequential_times);
    let speedup = if fused_median > 0.0 {
        sequential_median / fused_median
    } else {
        0.0
    };

    println!(
        "{}",
        serde_json::json!({
            "benchmark": "fused_ops_dense",
            "n_rows": n_rows,
            "n_cols": n_cols,
            "data_mb": format!("{data_mb:.1}"),
            "n_runs": n_runs,
            "fused_median_s": fused_median,
            "sequential_median_s": sequential_median,
            "speedup": speedup,
            "target_speedup": 1.5,
            "pass": speedup >= 1.0,
        })
    );
}
