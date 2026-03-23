// scx benchmark — Run read/query benchmarks on SCX files.

use std::path::Path;
use std::time::Instant;

use arrow::array::Array;
use scx_engine::QueryPipeline;
use scx_format::reader::ScxReader;

pub fn run_benchmark(
    file: &Path,
    compare_h5ad: Option<&Path>,
    runs: usize,
    json: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = ScxReader::open(file)?;
    let header = reader.header();
    let file_size = std::fs::metadata(file)?.len();

    let n_obs = header.n_obs;
    let n_vars = header.n_vars;
    let n_shards = header.n_csr_shards;
    let nnz = header.nnz;
    drop(reader);

    // --- Read benchmark ---
    let mut read_times_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let rdr = ScxReader::open(file)?;
        let _csr = rdr.read_all_csr_shards()?;
        let elapsed = start.elapsed();
        read_times_ms.push(elapsed.as_secs_f64() * 1000.0);
    }
    read_times_ms.sort_by(|a, b| a.total_cmp(b));
    let read_median = read_times_ms[read_times_ms.len() / 2];
    let read_min = read_times_ms[0];
    let read_max = *read_times_ms.last().unwrap();

    // --- Query benchmark ---
    // Find a categorical column to query on
    let rdr = ScxReader::open(file)?;
    let obs_schema = rdr.read_obs_schema()?;
    drop(rdr);

    let cat_col = obs_schema
        .fields()
        .iter()
        .find(|f| matches!(f.data_type(), arrow::datatypes::DataType::Utf8));

    let mut query_median_ms = None;
    let mut query_skip_rate = None;

    if let Some(field) = cat_col {
        let col_name = field.name();
        // Read a sample value to build predicates from
        let rdr = ScxReader::open(file)?;
        let obs = rdr.read_obs()?;
        drop(rdr);

        if let Some(col) = obs.column_by_name(col_name) {
            if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                if str_arr.len() > 0 {
                    let sample_val = str_arr.value(0);
                    let pred = format!("{col_name} == '{sample_val}'");

                    let mut query_times_ms = Vec::with_capacity(runs);
                    let mut skip_rate = 0.0;

                    for _ in 0..runs {
                        let start = Instant::now();
                        let result = QueryPipeline::open(file)?.filter_obs(&pred)?.collect()?;
                        let elapsed = start.elapsed();
                        query_times_ms.push(elapsed.as_secs_f64() * 1000.0);
                        if result.total_shards > 0 {
                            skip_rate =
                                result.skipped_shards as f64 / result.total_shards as f64 * 100.0;
                        }
                    }
                    query_times_ms.sort_by(|a, b| a.total_cmp(b));
                    query_median_ms = Some(query_times_ms[query_times_ms.len() / 2]);
                    query_skip_rate = Some(skip_rate);
                }
            }
        }
    }

    // --- H5AD comparison ---
    let mut h5ad_median_ms = None;
    if let Some(h5ad_path) = compare_h5ad {
        h5ad_median_ms = benchmark_h5ad(h5ad_path, runs);
    }

    // --- Output ---
    if json {
        let mut obj = serde_json::json!({
            "file": file.display().to_string(),
            "file_size_bytes": file_size,
            "n_obs": n_obs,
            "n_vars": n_vars,
            "n_shards": n_shards,
            "nnz": nnz,
            "runs": runs,
            "read_median_ms": read_median,
            "read_min_ms": read_min,
            "read_max_ms": read_max,
        });

        if let Some(qm) = query_median_ms {
            obj["query_median_ms"] = serde_json::json!(qm);
        }
        if let Some(sr) = query_skip_rate {
            obj["query_skip_rate_pct"] = serde_json::json!(sr);
        }
        if let Some(hm) = h5ad_median_ms {
            obj["h5ad_read_median_ms"] = serde_json::json!(hm);
            obj["speedup"] = serde_json::json!(hm / read_median);
        }

        println!("{}", serde_json::to_string_pretty(&obj)?);
    } else {
        println!("SCX Benchmark Results");
        println!("=====================");
        println!();
        println!(
            "File: {} ({:.1} MB)",
            file.display(),
            file_size as f64 / 1024.0 / 1024.0
        );
        println!(
            "Data: {} cells x {} genes, {} nnz, {} shards",
            n_obs, n_vars, nnz, n_shards
        );
        println!();
        println!("Read benchmark ({} runs):", runs);
        println!(
            "  Median: {:.1} ms | Min: {:.1} ms | Max: {:.1} ms",
            read_median, read_min, read_max
        );

        if let Some(qm) = query_median_ms {
            println!();
            println!("Query benchmark ({} runs):", runs);
            println!("  Median: {:.1} ms", qm);
            if let Some(sr) = query_skip_rate {
                println!("  Shard skip rate: {:.1}%", sr);
            }
        }

        if let Some(hm) = h5ad_median_ms {
            println!();
            println!("H5AD comparison:");
            println!("  H5AD read median: {:.1} ms", hm);
            println!("  SCX speedup: {:.1}x", hm / read_median);
        }
    }

    Ok(())
}

/// Benchmark h5ad read using Python subprocess.
/// Returns None if Python or anndata is not available.
fn benchmark_h5ad(path: &Path, runs: usize) -> Option<f64> {
    let script = format!(
        r#"
import time
import anndata
times = []
for _ in range({runs}):
    t = time.perf_counter()
    adata = anndata.read_h5ad("{path}")
    _ = adata.X
    times.append((time.perf_counter() - t) * 1000)
times.sort()
print(times[len(times) // 2])
"#,
        runs = runs,
        path = path.display(),
    );

    let output = std::process::Command::new("python3")
        .args(["-c", &script])
        .output()
        .ok()?;

    if !output.status.success() {
        eprintln!(
            "Note: h5ad comparison skipped (Python/anndata not available)\n  stderr: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<f64>().ok()
}
