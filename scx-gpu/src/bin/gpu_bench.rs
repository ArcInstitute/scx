//! GPU microbenchmark binary for SCX codec decode performance.
//!
//! Benchmarks GPU vs CPU performance for:
//!   #1  Rice decode, FOR-BP decode, full shard decode
//!   #4  Sparse-to-dense conversion (with/without HVG projection)
//!   #5  Multi-shard parallel decode scaling
//!
//! Outputs JSON lines to stdout; progress/status to stderr.
//!
//! Usage: cargo run --release -p scx-gpu --bin gpu_bench

use std::time::Instant;

use serde::Serialize;

use scx_codec::forbp::{forbp_decode, forbp_encode};
use scx_codec::rice::{rice_decode, rice_encode, B_VAL};
use scx_codec::{encode_shard, CodecId, EncodedShardRef, ValueEncoding};
use scx_gpu::test_utils::{build_test_shard, build_test_shard_with_metadata};
use scx_gpu::{
    decode_shard_gpu, decode_shard_gpu_with_metadata, forbp_decode_gpu, rice_decode_gpu,
    sparse_to_dense_gpu, GpuDevice,
};

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct BenchResult {
    benchmark: String,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_values: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_rows: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_cols: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nnz: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_shards: Option<usize>,
    cpu_median_us: f64,
    gpu_median_us: f64,
    speedup: f64,
    cpu_min_us: f64,
    cpu_max_us: f64,
    gpu_min_us: f64,
    gpu_max_us: f64,
}

/// Result of the sidecar-driven decode benchmark (Task 4.4a): GPU decode with
/// the CPU prescan vs driven by the decode sidecar, plus the host↔device
/// transfer accounting that quantifies the remaining FOR-BP host-fallback share
/// (the 4.4b decision input).
#[derive(Serialize)]
struct SidecarBenchResult {
    benchmark: String,
    label: String,
    n_rows: usize,
    nnz: usize,
    /// GPU decode median with the CPU prescan (no sidecar).
    prescan_median_us: f64,
    /// GPU decode median driven by the decode sidecar.
    sidecar_median_us: f64,
    /// prescan / sidecar — the prescan-elimination speedup.
    speedup: f64,
    host_uploaded_bytes: u64,
    device_decoded_bytes: u64,
    fully_device_decoded: bool,
    any_forbp_host_fallback: bool,
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

struct TimingResult {
    median_us: f64,
    min_us: f64,
    max_us: f64,
}

fn time_fn<F: FnMut()>(warmup: usize, iters: usize, mut f: F) -> TimingResult {
    for _ in 0..warmup {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        times.push(t.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    TimingResult {
        median_us: times[times.len() / 2],
        min_us: times[0],
        max_us: times[times.len() - 1],
    }
}

// ---------------------------------------------------------------------------
// Data generation (from scx-codec/benches/codec_bench.rs)
// ---------------------------------------------------------------------------

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Generate realistic CSR data. Returns (indptr, indices, values_u8, n_rows, nnz).
fn generate_shard(
    n_rows: usize,
    avg_nnz_per_row: usize,
    n_vars: u32,
) -> (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize) {
    let mut state: u64 = 0xDEAD_BEEF_CAFE_BABE;

    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::new();
    let mut values = Vec::new();
    indptr.push(0u64);

    for _ in 0..n_rows {
        let nnz = ((avg_nnz_per_row as u64 / 2)
            + xorshift(&mut state) % (avg_nnz_per_row as u64 + 1)) as usize;
        let nnz = nnz.min(n_vars as usize);

        let mut row_indices: Vec<u32> = Vec::with_capacity(nnz);
        let mut prev = 0u32;
        for _ in 0..nnz {
            let gap = 1 + (xorshift(&mut state) % ((n_vars as u64 / nnz.max(1) as u64).max(1)));
            prev = prev.saturating_add(gap as u32).min(n_vars - 1);
            row_indices.push(prev);
        }
        row_indices.sort_unstable();
        row_indices.dedup();

        let actual_nnz = row_indices.len();
        indptr.push(indptr.last().unwrap() + actual_nnz as u64);
        indices.extend_from_slice(&row_indices);

        for _ in 0..actual_nnz {
            let v = match xorshift(&mut state) % 100 {
                0..=60 => 1u8,
                61..=80 => 2,
                81..=90 => 3,
                91..=95 => (4 + xorshift(&mut state) % 4) as u8,
                _ => (8 + xorshift(&mut state) % 20) as u8,
            };
            values.push(v);
        }
    }

    let nnz = *indptr.last().unwrap() as usize;
    (indptr, indices, values, n_rows, nnz)
}

/// Generate UMI-like values for Rice benchmarking.
fn generate_umi_values(n: usize) -> Vec<u32> {
    let mut state: u64 = 0xCAFE_BABE_1234_5678;
    (0..n)
        .map(|_| match xorshift(&mut state) % 100 {
            0..=60 => 1,
            61..=80 => 2,
            81..=90 => 3,
            91..=95 => 4 + (xorshift(&mut state) % 4) as u32,
            _ => 8 + (xorshift(&mut state) % 20) as u32,
        })
        .collect()
}

/// CPU sparse-to-dense conversion (single-threaded row scatter).
fn cpu_sparse_to_dense(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_rows: usize,
    n_cols: usize,
) -> Vec<f32> {
    let mut dense = vec![0.0f32; n_rows * n_cols];
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for j in start..end {
            let col = indices[j] as usize;
            dense[row * n_cols + col] = data[j];
        }
    }
    dense
}

/// CPU sparse-to-dense with HVG projection.
fn cpu_sparse_to_dense_hvg(
    indptr: &[i64],
    indices: &[i32],
    data: &[f32],
    n_rows: usize,
    hvg_map: &[u32],
    n_output_cols: usize,
) -> Vec<f32> {
    let mut dense = vec![0.0f32; n_rows * n_output_cols];
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for j in start..end {
            let col = indices[j] as usize;
            if col < hvg_map.len() {
                let out_col = hvg_map[col];
                if out_col != 0xFFFF_FFFF {
                    dense[row * n_output_cols + out_col as usize] = data[j];
                }
            }
        }
    }
    dense
}

// ---------------------------------------------------------------------------
// Benchmark #1a: Rice decode
// ---------------------------------------------------------------------------

fn bench_rice_decode(dev: &GpuDevice) -> Vec<BenchResult> {
    let sizes = [
        (1_000, "1K"),
        (10_000, "10K"),
        (100_000, "100K"),
        (1_000_000, "1M"),
    ];
    let mut results = Vec::new();

    for &(n_values, label) in &sizes {
        eprintln!("  Rice {label}: {n_values} values...");
        let values = generate_umi_values(n_values);
        let encoded = rice_encode(&values, B_VAL).unwrap();

        let cpu = time_fn(3, 10, || {
            let _ = rice_decode(&encoded, n_values, B_VAL).unwrap();
        });

        let gpu = time_fn(3, 10, || {
            let _ = rice_decode_gpu(dev, &encoded, n_values, B_VAL).unwrap();
            dev.synchronize().unwrap();
        });

        let speedup = cpu.median_us / gpu.median_us;
        eprintln!(
            "    CPU: {:.0} us  GPU: {:.0} us  speedup: {:.1}x",
            cpu.median_us, gpu.median_us, speedup
        );

        results.push(BenchResult {
            benchmark: "rice_decode".into(),
            label: label.into(),
            n_values: Some(n_values),
            n_rows: None,
            n_cols: None,
            nnz: None,
            n_shards: None,
            cpu_median_us: cpu.median_us,
            gpu_median_us: gpu.median_us,
            speedup,
            cpu_min_us: cpu.min_us,
            cpu_max_us: cpu.max_us,
            gpu_min_us: gpu.min_us,
            gpu_max_us: gpu.max_us,
        });
    }
    results
}

// ---------------------------------------------------------------------------
// Benchmark #1b: FOR-BP decode
// ---------------------------------------------------------------------------

fn bench_forbp_decode(dev: &GpuDevice) -> Vec<BenchResult> {
    let configs = [
        (128, 200, "128r_200nnz"),
        (2048, 500, "2048r_500nnz"),
        (16384, 2000, "16384r_2000nnz"),
    ];
    let mut results = Vec::new();

    for &(n_rows, avg_nnz, label) in &configs {
        eprintln!("  FOR-BP {label}...");
        let (indptr, indices, _, n_rows_actual, nnz) = generate_shard(n_rows, avg_nnz, 30000);
        let row_lengths: Vec<usize> = indptr.windows(2).map(|w| (w[1] - w[0]) as usize).collect();
        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();

        let cpu = time_fn(3, 10, || {
            let _ = forbp_decode(&encoded, n_rows_actual, true).unwrap();
        });

        let gpu = time_fn(3, 10, || {
            let _ = forbp_decode_gpu(dev, &encoded, n_rows_actual, true).unwrap();
            dev.synchronize().unwrap();
        });

        let speedup = cpu.median_us / gpu.median_us;
        eprintln!(
            "    CPU: {:.0} us  GPU: {:.0} us  speedup: {:.1}x",
            cpu.median_us, gpu.median_us, speedup
        );

        results.push(BenchResult {
            benchmark: "forbp_decode".into(),
            label: label.into(),
            n_values: None,
            n_rows: Some(n_rows_actual),
            n_cols: None,
            nnz: Some(nnz),
            n_shards: None,
            cpu_median_us: cpu.median_us,
            gpu_median_us: gpu.median_us,
            speedup,
            cpu_min_us: cpu.min_us,
            cpu_max_us: cpu.max_us,
            gpu_min_us: gpu.min_us,
            gpu_max_us: gpu.max_us,
        });
    }
    results
}

// ---------------------------------------------------------------------------
// Benchmark #1c: Full shard decode
// ---------------------------------------------------------------------------

fn bench_shard_decode(dev: &GpuDevice) -> Vec<BenchResult> {
    let configs = [(2048, 500, "2048r_500nnz"), (16384, 2000, "16384r_2000nnz")];
    let n_vars: u32 = 30000;
    let mut results = Vec::new();

    for &(n_rows, avg_nnz, label) in &configs {
        eprintln!("  Shard decode {label}...");
        let (indptr, indices, values, n_rows_actual, nnz) = generate_shard(n_rows, avg_nnz, n_vars);
        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            n_vars,
        );

        // Prepare CPU decode path via EncodedShardRef
        let index_dtype_u16 = n_vars <= 65535;
        let encoded = encode_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            index_dtype_u16,
        )
        .unwrap();

        let cpu = time_fn(3, 10, || {
            let encoded_ref = EncodedShardRef {
                indptr_bytes: &encoded.indptr_bytes,
                indices_bytes: &encoded.indices_bytes,
                values_bytes: &encoded.values_bytes,
            };
            let _ = scx_codec::decode_shard_scipy(
                &encoded_ref,
                CodecId::Scx1,
                ValueEncoding::Uint8,
                n_rows_actual,
                nnz,
                index_dtype_u16,
            )
            .unwrap();
        });

        let gpu = time_fn(3, 10, || {
            let _ = decode_shard_gpu(dev, &shard_bytes).unwrap();
            dev.synchronize().unwrap();
        });

        let speedup = cpu.median_us / gpu.median_us;
        eprintln!(
            "    CPU: {:.0} us  GPU: {:.0} us  speedup: {:.1}x",
            cpu.median_us, gpu.median_us, speedup
        );

        results.push(BenchResult {
            benchmark: "shard_decode".into(),
            label: label.into(),
            n_values: None,
            n_rows: Some(n_rows_actual),
            n_cols: Some(n_vars as usize),
            nnz: Some(nnz),
            n_shards: None,
            cpu_median_us: cpu.median_us,
            gpu_median_us: gpu.median_us,
            speedup,
            cpu_min_us: cpu.min_us,
            cpu_max_us: cpu.max_us,
            gpu_min_us: gpu.min_us,
            gpu_max_us: gpu.max_us,
        });
    }
    results
}

// ---------------------------------------------------------------------------
// Benchmark #4: Sparse-to-dense GPU vs CPU
// ---------------------------------------------------------------------------

fn bench_sparse_to_dense(dev: &GpuDevice) -> Vec<BenchResult> {
    let configs: &[(usize, u32, usize, &str)] = &[
        (256, 30000, 200, "256r_30Kc"),
        (2048, 30000, 500, "2048r_30Kc"),
        (16384, 30000, 2000, "16384r_30Kc"),
    ];
    let mut results = Vec::new();

    for &(n_rows, n_vars, avg_nnz, label) in configs {
        eprintln!("  Sparse→dense {label}...");
        let (indptr, indices, values, n_rows_actual, nnz) = generate_shard(n_rows, avg_nnz, n_vars);
        let n_cols = n_vars as usize;

        // Build CPU CSR arrays (scipy types)
        let indptr_i64: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let indices_i32: Vec<i32> = indices.iter().map(|&v| v as i32).collect();
        let data_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();

        // CPU sparse-to-dense
        let cpu = time_fn(2, 5, || {
            let _ =
                cpu_sparse_to_dense(&indptr_i64, &indices_i32, &data_f32, n_rows_actual, n_cols);
        });

        // GPU: decode shard → sparse_to_dense
        let shard_bytes = build_test_shard(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            n_vars,
        );
        let gpu_csr = decode_shard_gpu(dev, &shard_bytes).unwrap();
        dev.synchronize().unwrap();

        let gpu = time_fn(2, 5, || {
            let _ = sparse_to_dense_gpu(dev, &gpu_csr, None, n_cols).unwrap();
            dev.synchronize().unwrap();
        });

        let speedup = cpu.median_us / gpu.median_us;
        eprintln!(
            "    CPU: {:.0} us  GPU: {:.0} us  speedup: {:.1}x",
            cpu.median_us, gpu.median_us, speedup
        );

        results.push(BenchResult {
            benchmark: "sparse_to_dense".into(),
            label: label.into(),
            n_values: None,
            n_rows: Some(n_rows_actual),
            n_cols: Some(n_cols),
            nnz: Some(nnz),
            n_shards: None,
            cpu_median_us: cpu.median_us,
            gpu_median_us: gpu.median_us,
            speedup,
            cpu_min_us: cpu.min_us,
            cpu_max_us: cpu.max_us,
            gpu_min_us: gpu.min_us,
            gpu_max_us: gpu.max_us,
        });

        // HVG projection sub-benchmark on the largest case
        if n_rows == 16384 {
            let n_hvg = 2000;
            eprintln!("  Sparse→dense HVG ({n_hvg} genes)...");

            // Build HVG map: select every 15th gene (2000 out of 30000)
            let mut hvg_map = vec![0xFFFF_FFFFu32; n_cols];
            let mut out_col = 0u32;
            for i in (0..n_cols).step_by(n_cols / n_hvg) {
                if (out_col as usize) < n_hvg {
                    hvg_map[i] = out_col;
                    out_col += 1;
                }
            }
            let n_output_cols = out_col as usize;

            let cpu_hvg = time_fn(2, 5, || {
                let _ = cpu_sparse_to_dense_hvg(
                    &indptr_i64,
                    &indices_i32,
                    &data_f32,
                    n_rows_actual,
                    &hvg_map,
                    n_output_cols,
                );
            });

            let d_hvg_map = dev.htod_copy(&hvg_map).unwrap();
            let gpu_hvg = time_fn(2, 5, || {
                let _ =
                    sparse_to_dense_gpu(dev, &gpu_csr, Some(&d_hvg_map), n_output_cols).unwrap();
                dev.synchronize().unwrap();
            });

            let speedup_hvg = cpu_hvg.median_us / gpu_hvg.median_us;
            eprintln!(
                "    CPU: {:.0} us  GPU: {:.0} us  speedup: {:.1}x",
                cpu_hvg.median_us, gpu_hvg.median_us, speedup_hvg
            );

            results.push(BenchResult {
                benchmark: "sparse_to_dense_hvg".into(),
                label: format!("16384r_{n_hvg}hvg"),
                n_values: None,
                n_rows: Some(n_rows_actual),
                n_cols: Some(n_output_cols),
                nnz: Some(nnz),
                n_shards: None,
                cpu_median_us: cpu_hvg.median_us,
                gpu_median_us: gpu_hvg.median_us,
                speedup: speedup_hvg,
                cpu_min_us: cpu_hvg.min_us,
                cpu_max_us: cpu_hvg.max_us,
                gpu_min_us: gpu_hvg.min_us,
                gpu_max_us: gpu_hvg.max_us,
            });
        }
    }
    results
}

// ---------------------------------------------------------------------------
// Benchmark #5: Multi-shard parallel decode
// ---------------------------------------------------------------------------

fn bench_multi_shard(dev: &GpuDevice) -> Vec<BenchResult> {
    eprintln!("  Generating shard for multi-shard test...");
    let (indptr, indices, values, _, _) = generate_shard(2048, 500, 30000);
    let shard_bytes = build_test_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Scx1,
        ValueEncoding::Uint8,
        30000,
    );

    // Single-shard baseline (CPU)
    let index_dtype_u16 = true;
    let encoded = encode_shard(
        &indptr,
        &indices,
        &values,
        CodecId::Scx1,
        ValueEncoding::Uint8,
        index_dtype_u16,
    )
    .unwrap();
    let n_rows = indptr.len() - 1;
    let nnz = *indptr.last().unwrap() as usize;

    let cpu_single = time_fn(3, 10, || {
        let encoded_ref = EncodedShardRef {
            indptr_bytes: &encoded.indptr_bytes,
            indices_bytes: &encoded.indices_bytes,
            values_bytes: &encoded.values_bytes,
        };
        let _ = scx_codec::decode_shard_scipy(
            &encoded_ref,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            n_rows,
            nnz,
            index_dtype_u16,
        )
        .unwrap();
    });

    let shard_counts = [1, 2, 4, 8, 16];
    let mut results = Vec::new();

    for &n_shards in &shard_counts {
        eprintln!("  Multi-shard: {n_shards} shards...");

        let gpu = time_fn(2, 5, || {
            for _ in 0..n_shards {
                let _ = decode_shard_gpu(dev, &shard_bytes).unwrap();
            }
            dev.synchronize().unwrap();
        });

        let cpu_total_us = cpu_single.median_us * n_shards as f64;
        let speedup = cpu_total_us / gpu.median_us;
        let per_shard_us = gpu.median_us / n_shards as f64;
        let throughput = 1e6 / per_shard_us;

        eprintln!(
            "    GPU total: {:.0} us  per-shard: {:.0} us  throughput: {:.0} shards/s  speedup vs CPU: {:.1}x",
            gpu.median_us, per_shard_us, throughput, speedup
        );

        results.push(BenchResult {
            benchmark: "multi_shard_decode".into(),
            label: format!("{n_shards}_shards"),
            n_values: None,
            n_rows: Some(n_rows),
            n_cols: Some(30000),
            nnz: Some(nnz),
            n_shards: Some(n_shards),
            cpu_median_us: cpu_total_us,
            gpu_median_us: gpu.median_us,
            speedup,
            cpu_min_us: cpu_single.min_us * n_shards as f64,
            cpu_max_us: cpu_single.max_us * n_shards as f64,
            gpu_min_us: gpu.min_us,
            gpu_max_us: gpu.max_us,
        });
    }
    results
}

// ---------------------------------------------------------------------------
// Benchmark #6: Sidecar-driven decode (Task 4.4a)
// ---------------------------------------------------------------------------

/// Compare GPU shard decode driven by the decode sidecar (no CPU prescan) vs the
/// prescan path, and report the host↔device transfer stats. The dense config
/// exposes the residual FOR-BP host-fallback (the Task 4.4b gap): its `indices`
/// still upload from the host, so `fully_device_decoded` is false and
/// `host_uploaded_bytes` includes `nnz*4`; the sparse config decodes fully on
/// the device (only the tiny indptr uploaded).
fn bench_sidecar_decode(dev: &GpuDevice) -> Vec<SidecarBenchResult> {
    // (n_rows, n_vars, avg_nnz, label). As of Task 4.4b both decode fully on the
    // device — the dense config (>=128 nnz) goes through the BitPacker4x kernel
    // instead of the old FOR-BP host-fallback, so it now reports
    // fully_device_decoded=true with bytes_uploaded ≈ indptr only.
    let configs: &[(usize, u32, usize, &str)] = &[
        (16384, 30000, 50, "16384r_sparse50"),  // <128 nnz
        (16384, 30000, 256, "16384r_dense256"), // >=128 nnz → BitPacker4x kernel
    ];
    let mut results = Vec::new();

    for &(n_rows, n_vars, avg_nnz, label) in configs {
        eprintln!("  Sidecar decode {label}...");
        let (indptr, indices, values, n_rows_actual, nnz) = generate_shard(n_rows, avg_nnz, n_vars);
        let (shard_bytes, meta) = build_test_shard_with_metadata(
            &indptr,
            &indices,
            &values,
            CodecId::Scx1,
            ValueEncoding::Uint8,
            n_vars,
        );
        let meta = meta.expect("Scx1 shard emits decode metadata");

        // No-sidecar (CPU prescan) path vs sidecar-driven path.
        let prescan = time_fn(2, 10, || {
            let _ = decode_shard_gpu_with_metadata(dev, &shard_bytes, None).unwrap();
            dev.synchronize().unwrap();
        });
        let sidecar = time_fn(2, 10, || {
            let _ = decode_shard_gpu_with_metadata(dev, &shard_bytes, Some(&meta)).unwrap();
            dev.synchronize().unwrap();
        });
        // Capture the transfer stats once (deterministic; cheap relative to timing).
        let (_, stats) = decode_shard_gpu_with_metadata(dev, &shard_bytes, Some(&meta)).unwrap();
        dev.synchronize().unwrap();

        let speedup = prescan.median_us / sidecar.median_us;
        eprintln!(
            "    prescan: {:.0} us  sidecar: {:.0} us  speedup: {:.2}x  uploaded: {} B  device: {} B  fully_device={}",
            prescan.median_us,
            sidecar.median_us,
            speedup,
            stats.host_uploaded_bytes,
            stats.device_decoded_bytes,
            stats.fully_device_decoded
        );

        results.push(SidecarBenchResult {
            benchmark: "sidecar_decode".into(),
            label: label.into(),
            n_rows: n_rows_actual,
            nnz,
            prescan_median_us: prescan.median_us,
            sidecar_median_us: sidecar.median_us,
            speedup,
            host_uploaded_bytes: stats.host_uploaded_bytes,
            device_decoded_bytes: stats.device_decoded_bytes,
            fully_device_decoded: stats.fully_device_decoded,
            any_forbp_host_fallback: stats.any_forbp_host_fallback,
        });
    }
    results
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("=== SCX GPU Microbenchmarks ===");

    let dev = match GpuDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("ERROR: Cannot create GPU device: {e}");
            eprintln!("GPU benchmarks require a CUDA-capable GPU.");
            std::process::exit(1);
        }
    };

    eprintln!("GPU device 0 initialized\n");

    let mut all_results = Vec::new();

    eprintln!("--- Benchmark #1a: Rice decode GPU vs CPU ---");
    all_results.extend(bench_rice_decode(&dev));

    eprintln!("\n--- Benchmark #1b: FOR-BP decode GPU vs CPU ---");
    all_results.extend(bench_forbp_decode(&dev));

    eprintln!("\n--- Benchmark #1c: Full shard decode GPU vs CPU ---");
    all_results.extend(bench_shard_decode(&dev));

    eprintln!("\n--- Benchmark #4: Sparse-to-dense GPU vs CPU ---");
    all_results.extend(bench_sparse_to_dense(&dev));

    eprintln!("\n--- Benchmark #5: Multi-shard parallel decode ---");
    all_results.extend(bench_multi_shard(&dev));

    eprintln!("\n--- Benchmark #6: Sidecar-driven decode (Task 4.4a) ---");
    let sidecar_results = bench_sidecar_decode(&dev);

    eprintln!("\n=== Done ===");

    // Output JSON lines to stdout
    for r in &all_results {
        println!("{}", serde_json::to_string(r).unwrap());
    }
    for r in &sidecar_results {
        println!("{}", serde_json::to_string(r).unwrap());
    }
}
