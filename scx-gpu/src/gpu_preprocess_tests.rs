use super::*;

/// CPU reference: fused normalize+log1p for a CSR matrix.
/// Matches scx-engine's implementation exactly (f64 intermediates, f32 output).
fn cpu_fused_normalize_log1p(indptr: &[i64], data: &mut [f32], target_sum: f64) {
    let n_rows = indptr.len() - 1;
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
        if row_sum > 0.0 {
            let factor = target_sum / row_sum;
            for v in &mut data[start..end] {
                *v = ((*v as f64 * factor) as f32).ln_1p();
            }
        }
    }
}

/// CPU reference: normalize-only.
fn cpu_normalize(indptr: &[i64], data: &mut [f32], target_sum: f64) {
    let n_rows = indptr.len() - 1;
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let row_sum: f64 = data[start..end].iter().map(|&v| v as f64).sum();
        if row_sum > 0.0 {
            let factor = target_sum / row_sum;
            for v in &mut data[start..end] {
                *v = (*v as f64 * factor) as f32;
            }
        }
    }
}

/// CPU reference: log1p-only.
fn cpu_log1p(indptr: &[i64], data: &mut [f32]) {
    let n_rows = indptr.len() - 1;
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        for v in &mut data[start..end] {
            *v = v.ln_1p();
        }
    }
}

#[test]
fn test_gpu_normalize_log1p_matches_cpu() {
    let dev = require_gpu!();

    // 3-row CSR:
    // row 0: [5.0, 10.0] at indices [1, 3]   row_sum = 15
    // row 1: [1.0, 3.0, 7.0] at indices [0, 2, 4]   row_sum = 11
    // row 2: [2.0] at index [2]   row_sum = 2
    let indptr: Vec<i64> = vec![0, 2, 5, 6];
    let data: Vec<f32> = vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0];
    let target_sum = 10_000.0f32;
    let n_rows = 3;

    // GPU path
    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
    dev.synchronize().unwrap();
    let gpu_result = dev.dtoh_copy(&d_data).unwrap();

    // CPU reference
    let mut cpu_data = data.clone();
    cpu_fused_normalize_log1p(&indptr, &mut cpu_data, target_sum as f64);

    // Compare with relative tolerance (GPU f32 vs CPU f64 intermediates)
    assert_eq!(gpu_result.len(), cpu_data.len());
    for i in 0..gpu_result.len() {
        let diff = (gpu_result[i] - cpu_data[i]).abs();
        let denom = cpu_data[i].abs().max(1e-10);
        assert!(
            diff / denom < 1e-5,
            "normalize_log1p mismatch at [{}]: gpu={}, cpu={}, rel_err={}",
            i,
            gpu_result[i],
            cpu_data[i],
            diff / denom
        );
    }
}

#[test]
fn test_gpu_normalize_only() {
    let dev = require_gpu!();

    let indptr: Vec<i64> = vec![0, 3, 5];
    let data: Vec<f32> = vec![1.0, 4.0, 5.0, 2.0, 8.0];
    let target_sum = 1.0f32;
    let n_rows = 2;

    // GPU path
    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_normalize(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
    dev.synchronize().unwrap();
    let gpu_result = dev.dtoh_copy(&d_data).unwrap();

    // CPU reference
    let mut cpu_data = data.clone();
    cpu_normalize(&indptr, &mut cpu_data, target_sum as f64);

    for i in 0..gpu_result.len() {
        assert!(
            (gpu_result[i] - cpu_data[i]).abs() < 1e-6,
            "normalize mismatch at [{}]: gpu={}, cpu={}",
            i,
            gpu_result[i],
            cpu_data[i]
        );
    }

    // Verify row sums
    let row0_sum: f32 = gpu_result[0..3].iter().sum();
    let row1_sum: f32 = gpu_result[3..5].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5, "row 0 sum = {row0_sum}");
    assert!((row1_sum - 1.0).abs() < 1e-5, "row 1 sum = {row1_sum}");
}

#[test]
fn test_gpu_log1p_only() {
    let dev = require_gpu!();

    let indptr: Vec<i64> = vec![0, 2, 4];
    let data: Vec<f32> = vec![0.0, 5.0, 10.0, 100.0];
    let n_rows = 2;

    // GPU path
    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_log1p(&dev, &d_indptr, &mut d_data, n_rows).unwrap();
    dev.synchronize().unwrap();
    let gpu_result = dev.dtoh_copy(&d_data).unwrap();

    // CPU reference
    let mut cpu_data = data.clone();
    cpu_log1p(&indptr, &mut cpu_data);

    for i in 0..gpu_result.len() {
        assert!(
            (gpu_result[i] - cpu_data[i]).abs() < 1e-6,
            "log1p mismatch at [{}]: gpu={}, cpu={}",
            i,
            gpu_result[i],
            cpu_data[i]
        );
    }
}

#[test]
fn test_gpu_normalize_log1p_empty_row() {
    let dev = require_gpu!();

    // Row 0 is empty, row 1 has data
    let indptr: Vec<i64> = vec![0, 0, 3];
    let data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let target_sum = 1e4f32;
    let n_rows = 2;

    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
    dev.synchronize().unwrap();
    let gpu_result = dev.dtoh_copy(&d_data).unwrap();

    // CPU reference
    let mut cpu_data = data.clone();
    cpu_fused_normalize_log1p(&indptr, &mut cpu_data, target_sum as f64);

    for i in 0..gpu_result.len() {
        let diff = (gpu_result[i] - cpu_data[i]).abs();
        assert!(
            diff < 1e-4,
            "empty_row mismatch at [{}]: gpu={}, cpu={}",
            i,
            gpu_result[i],
            cpu_data[i]
        );
    }
}

#[test]
fn test_gpu_normalize_log1p_zero_rows() {
    let dev = require_gpu!();

    // Zero rows should be a no-op
    let indptr: Vec<i64> = vec![0];
    let data: Vec<f32> = vec![];

    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, 0, 1e4).unwrap();
    // Should not panic
}

#[test]
fn test_gpu_apply_fused_ops_dispatch() {
    let dev = require_gpu!();

    let indptr: Vec<i64> = vec![0, 2, 4];
    let data: Vec<f32> = vec![5.0, 10.0, 3.0, 7.0];
    let n_rows = 2;

    // Test all four dispatch paths

    // 1. Both normalize + log1p
    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data, n_rows, Some(1e4), true, None).unwrap();
    dev.synchronize().unwrap();
    let result_both = dev.dtoh_copy(&d_data).unwrap();

    let mut cpu_both = data.clone();
    cpu_fused_normalize_log1p(&indptr, &mut cpu_both, 1e4);
    for i in 0..result_both.len() {
        assert!(
            (result_both[i] - cpu_both[i]).abs() < 1e-4,
            "both mismatch at {i}"
        );
    }

    // 2. Normalize only
    let mut d_data2 = dev.htod_copy(&data).unwrap();
    gpu_apply_fused_ops(
        &dev,
        &d_indptr,
        &mut d_data2,
        n_rows,
        Some(1.0),
        false,
        None,
    )
    .unwrap();
    dev.synchronize().unwrap();
    let result_norm = dev.dtoh_copy(&d_data2).unwrap();
    let row0_sum: f32 = result_norm[0..2].iter().sum();
    assert!((row0_sum - 1.0).abs() < 1e-5);

    // 3. Log1p only
    let mut d_data3 = dev.htod_copy(&data).unwrap();
    gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data3, n_rows, None, true, None).unwrap();
    dev.synchronize().unwrap();
    let result_log = dev.dtoh_copy(&d_data3).unwrap();
    assert!((result_log[0] - 5.0f32.ln_1p()).abs() < 1e-6);

    // 4. No-op
    let mut d_data4 = dev.htod_copy(&data).unwrap();
    gpu_apply_fused_ops(&dev, &d_indptr, &mut d_data4, n_rows, None, false, None).unwrap();
    dev.synchronize().unwrap();
    let result_noop = dev.dtoh_copy(&d_data4).unwrap();
    assert_eq!(result_noop, data);
}

#[test]
fn test_gpu_normalize_log1p_large() {
    let dev = require_gpu!();

    // Generate a larger CSR matrix to stress-test the kernel
    let n_rows = 500;
    let nnz_per_row = 10;
    let mut indptr = vec![0i64];
    let mut data = Vec::new();
    let mut state: u64 = 0xCAFE_BABE;

    for _row in 0..n_rows {
        for _col in 0..nnz_per_row {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // UMI-like count values (1-20)
            data.push(((state % 20) + 1) as f32);
        }
        indptr.push(data.len() as i64);
    }

    let target_sum = 1e4f32;

    // GPU
    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_normalize_log1p(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
    dev.synchronize().unwrap();
    let gpu_result = dev.dtoh_copy(&d_data).unwrap();

    // CPU
    let mut cpu_data = data.clone();
    cpu_fused_normalize_log1p(&indptr, &mut cpu_data, target_sum as f64);

    // Compare
    let mut max_rel_err = 0.0f64;
    for i in 0..gpu_result.len() {
        let diff = (gpu_result[i] as f64 - cpu_data[i] as f64).abs();
        let denom = (cpu_data[i] as f64).abs().max(1e-10);
        let rel = diff / denom;
        if rel > max_rel_err {
            max_rel_err = rel;
        }
    }
    assert!(
        max_rel_err < 1e-5,
        "max relative error = {max_rel_err} (threshold: 1e-5)"
    );
}

// -------------------------------------------------------------------
// Task 5.1 — `gpu_preprocess_to_csr` streaming driver tests
// -------------------------------------------------------------------

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use scx_sparse::ScxCsr;

struct InMemorySource {
    shards: Vec<ScxCsr>,
    n_obs: usize,
    n_vars: usize,
}

impl ShardSource for InMemorySource {
    fn n_shards(&self) -> usize {
        self.shards.len()
    }
    fn n_obs(&self) -> usize {
        self.n_obs
    }
    fn n_vars(&self) -> usize {
        self.n_vars
    }
    fn read_shard(&self, shard_idx: usize) -> scx_format_io::Result<ScxCsr> {
        Ok(self.shards[shard_idx].clone())
    }
}

/// Build a random CSR with ~`density` fraction of nonzeros and strictly
/// positive f32 values in `[0.5, 20.5)` (UMI-like).
fn random_pos_csr(n_rows: usize, n_cols: usize, density: f32, seed: u64) -> ScxCsr {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indptr: Vec<i64> = Vec::with_capacity(n_rows + 1);
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    indptr.push(0);
    for _ in 0..n_rows {
        for c in 0..n_cols {
            if rng.gen_bool(density as f64) {
                indices.push(c as i32);
                data.push(rng.gen_range(0.5..20.5));
            }
        }
        indptr.push(indices.len() as i64);
    }
    ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data)
}

fn split_into_shards(csr: &ScxCsr, n_shards: usize) -> Vec<ScxCsr> {
    let (n_rows, n_cols) = (csr.n_rows(), csr.n_cols());
    let rows_per = n_rows.div_ceil(n_shards);
    let mut out = Vec::new();
    let mut row_start = 0;
    while row_start < n_rows {
        let row_end = (row_start + rows_per).min(n_rows);
        let p0 = csr.indptr[row_start] as usize;
        let p1 = csr.indptr[row_end] as usize;
        let shard_indptr: Vec<i64> = csr.indptr[row_start..=row_end]
            .iter()
            .map(|&p| p - csr.indptr[row_start])
            .collect();
        let shard_indices = csr.indices[p0..p1].to_vec();
        let shard_data = csr.data[p0..p1].to_vec();
        out.push(ScxCsr::new_unchecked(
            (row_end - row_start, n_cols),
            shard_indptr,
            shard_indices,
            shard_data,
        ));
        row_start = row_end;
    }
    out
}

#[test]
fn test_gpu_preprocess_to_csr_normalize_matches_cpu() {
    let dev = require_gpu!();
    let n_rows = 500;
    let n_cols = 100;
    let target_sum = 1e4f32;
    let csr = random_pos_csr(n_rows, n_cols, 0.1, 11);
    let shards = split_into_shards(&csr, 4);
    let source = InMemorySource {
        shards,
        n_obs: n_rows,
        n_vars: n_cols,
    };

    let result = gpu_preprocess_to_csr(&dev, &source, Some(target_sum), false, None)
        .expect("gpu preprocess");
    assert_eq!(result.n_rows(), n_rows);
    assert_eq!(result.n_cols(), n_cols);
    assert_eq!(result.nnz(), csr.nnz());
    assert_eq!(result.indices, csr.indices);
    assert_eq!(result.indptr, csr.indptr);

    // CPU reference: apply normalize on the full concatenated CSR.
    let mut cpu_data = csr.data.clone();
    cpu_normalize(&csr.indptr, &mut cpu_data, target_sum as f64);

    // Elementwise rel-err < 1e-5.
    let mut max_rel_err = 0.0f64;
    for i in 0..cpu_data.len() {
        let diff = (result.data[i] as f64 - cpu_data[i] as f64).abs();
        let denom = (cpu_data[i] as f64).abs().max(1e-10);
        max_rel_err = max_rel_err.max(diff / denom);
    }
    assert!(max_rel_err < 1e-5, "normalize max rel err = {max_rel_err}");

    // Per-row sums equal target_sum (within 1e-3 rel).
    for r in 0..n_rows {
        let p0 = result.indptr[r] as usize;
        let p1 = result.indptr[r + 1] as usize;
        if p0 == p1 {
            continue;
        }
        let row_sum: f32 = result.data[p0..p1].iter().sum();
        let rel = (row_sum - target_sum).abs() / target_sum;
        assert!(rel < 1e-3, "row {r} sum {row_sum} != {target_sum}");
    }
}

#[test]
fn test_gpu_preprocess_to_csr_log1p_matches_cpu() {
    let dev = require_gpu!();
    let n_rows = 500;
    let n_cols = 100;
    let csr = random_pos_csr(n_rows, n_cols, 0.1, 22);
    let shards = split_into_shards(&csr, 4);
    let source = InMemorySource {
        shards,
        n_obs: n_rows,
        n_vars: n_cols,
    };

    let result = gpu_preprocess_to_csr(&dev, &source, None, true, None).expect("gpu preprocess");
    assert_eq!(result.indices, csr.indices);
    assert_eq!(result.indptr, csr.indptr);

    let mut cpu_data = csr.data.clone();
    cpu_log1p(&csr.indptr, &mut cpu_data);

    for i in 0..cpu_data.len() {
        let diff = (result.data[i] - cpu_data[i]).abs();
        assert!(diff < 1e-6, "log1p mismatch at [{i}]: {diff}");
    }
}

#[test]
fn test_gpu_preprocess_to_csr_fused_matches_cpu() {
    let dev = require_gpu!();
    let n_rows = 500;
    let n_cols = 100;
    let target_sum = 1e4f32;
    let csr = random_pos_csr(n_rows, n_cols, 0.1, 33);
    let shards = split_into_shards(&csr, 4);
    let source = InMemorySource {
        shards,
        n_obs: n_rows,
        n_vars: n_cols,
    };

    let result =
        gpu_preprocess_to_csr(&dev, &source, Some(target_sum), true, None).expect("gpu preprocess");
    assert_eq!(result.indices, csr.indices);
    assert_eq!(result.indptr, csr.indptr);

    let mut cpu_data = csr.data.clone();
    cpu_fused_normalize_log1p(&csr.indptr, &mut cpu_data, target_sum as f64);

    let mut max_rel_err = 0.0f64;
    for i in 0..cpu_data.len() {
        let diff = (result.data[i] as f64 - cpu_data[i] as f64).abs();
        let denom = (cpu_data[i] as f64).abs().max(1e-10);
        max_rel_err = max_rel_err.max(diff / denom);
    }
    // GPU f32 vs CPU f64 intermediates — same tolerance as the fused kernel test above.
    assert!(
        max_rel_err < 1e-4,
        "fused max rel err = {max_rel_err} (threshold 1e-4)"
    );
}

#[test]
fn test_gpu_preprocess_to_csr_single_shard() {
    // Exercises the single-shard fast path of `RawGpuShardSource`.
    let dev = require_gpu!();
    let n_rows = 100;
    let n_cols = 40;
    let target_sum = 1.0f32;
    let csr = random_pos_csr(n_rows, n_cols, 0.1, 44);
    let source = InMemorySource {
        shards: vec![csr.clone()],
        n_obs: n_rows,
        n_vars: n_cols,
    };

    let result = gpu_preprocess_to_csr(&dev, &source, Some(target_sum), false, None)
        .expect("gpu preprocess");
    assert_eq!(result.n_rows(), n_rows);
    assert_eq!(result.n_cols(), n_cols);
    // Per-row sums should be ~1.0.
    for r in 0..n_rows {
        let p0 = result.indptr[r] as usize;
        let p1 = result.indptr[r + 1] as usize;
        if p0 == p1 {
            continue;
        }
        let row_sum: f32 = result.data[p0..p1].iter().sum();
        assert!((row_sum - 1.0).abs() < 1e-4, "row {r} sum = {row_sum}");
    }
}

// -------------------------------------------------------------------
// row_scale (item 3l): per-row explicit-factor scaling on device
// -------------------------------------------------------------------

/// CPU reference: per-row explicit-factor scale (matches `row_scale_kernel`).
fn cpu_row_scale(indptr: &[i64], data: &mut [f32], factors: &[f32]) {
    let n_rows = indptr.len() - 1;
    for row in 0..n_rows {
        let start = indptr[row] as usize;
        let end = indptr[row + 1] as usize;
        let f = factors[row];
        for v in &mut data[start..end] {
            *v *= f;
        }
    }
}

/// Standalone `gpu_row_scale` (single matrix, in-place) vs CPU per-row
/// multiply. row_scale uses plain f32 multiply on both paths, so this is a
/// tight bound.
#[test]
fn test_gpu_row_scale_matches_cpu() {
    let dev = require_gpu!();
    let indptr: Vec<i64> = vec![0, 2, 5, 6];
    let data: Vec<f32> = vec![5.0, 10.0, 1.0, 3.0, 7.0, 2.0];
    let factors: Vec<f32> = vec![2.0, 0.5, 3.0];
    let n_rows = 3;

    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    let d_factors = dev.htod_copy(&factors).unwrap();
    gpu_row_scale(&dev, &d_indptr, &mut d_data, n_rows, &d_factors).unwrap();
    dev.synchronize().unwrap();
    let gpu_result = dev.dtoh_copy(&d_data).unwrap();

    let mut cpu_data = data.clone();
    cpu_row_scale(&indptr, &mut cpu_data, &factors);

    for i in 0..gpu_result.len() {
        assert!(
            (gpu_result[i] - cpu_data[i]).abs() < 1e-5,
            "row_scale mismatch at {i}: gpu={}, cpu={}",
            gpu_result[i],
            cpu_data[i]
        );
    }
}

/// `gpu_preprocess_to_csr` row_scale-only over a multi-shard source — the
/// concatenated result must equal the source scaled per global row,
/// validating the per-shard `global_row_offset` accumulation.
#[test]
fn test_gpu_preprocess_to_csr_row_scale_only_multishard() {
    let dev = require_gpu!();
    let n_rows = 400;
    let n_cols = 80;
    let csr = random_pos_csr(n_rows, n_cols, 0.1, 0x5EED_1234);
    let factors: Vec<f32> = (0..n_rows).map(|r| 0.25 + (r % 16) as f32 * 0.25).collect();
    let shards = split_into_shards(&csr, 4);
    let source = InMemorySource {
        shards,
        n_obs: n_rows,
        n_vars: n_cols,
    };

    let result =
        gpu_preprocess_to_csr(&dev, &source, None, false, Some(&factors)).expect("gpu preprocess");

    let mut cpu_data = csr.data.clone();
    cpu_row_scale(&csr.indptr, &mut cpu_data, &factors);

    assert_eq!(result.data.len(), cpu_data.len());
    for i in 0..cpu_data.len() {
        let diff = (result.data[i] as f64 - cpu_data[i] as f64).abs();
        let denom = (cpu_data[i] as f64).abs().max(1e-10);
        assert!(diff / denom < 1e-5, "row_scale mismatch at nnz {i}");
    }
}

/// Full chain normalize → log1p → row_scale on a multi-shard source vs a
/// CPU reference applying the transforms in the same canonical order.
#[test]
fn test_gpu_preprocess_to_csr_normalize_log1p_row_scale_multishard() {
    let dev = require_gpu!();
    let n_rows = 500;
    let n_cols = 100;
    let target_sum = 1e4f32;
    let csr = random_pos_csr(n_rows, n_cols, 0.1, 0xBEEF_F00D);
    let factors: Vec<f32> = (0..n_rows).map(|r| 0.5 + (r % 7) as f32 * 0.3).collect();
    let shards = split_into_shards(&csr, 4);
    let source = InMemorySource {
        shards,
        n_obs: n_rows,
        n_vars: n_cols,
    };

    let result = gpu_preprocess_to_csr(&dev, &source, Some(target_sum), true, Some(&factors))
        .expect("gpu preprocess");

    // CPU reference: fused normalize+log1p, then per-row scale (global row).
    let mut cpu_data = csr.data.clone();
    cpu_fused_normalize_log1p(&csr.indptr, &mut cpu_data, target_sum as f64);
    cpu_row_scale(&csr.indptr, &mut cpu_data, &factors);

    assert_eq!(result.data.len(), cpu_data.len());
    let mut max_rel = 0.0f64;
    for i in 0..cpu_data.len() {
        let diff = (result.data[i] as f64 - cpu_data[i] as f64).abs();
        let denom = (cpu_data[i] as f64).abs().max(1e-10);
        max_rel = max_rel.max(diff / denom);
    }
    assert!(
        max_rel < 1e-5,
        "max relative error = {max_rel} (threshold: 1e-5)"
    );
}

// -------------------------------------------------------------------
// Phase 2.1 — nonzero-parallel kernels: tier selection + parity
// -------------------------------------------------------------------

/// The tier selector is pure and correctness-neutral; verify its
/// mean-density boundaries without a GPU.
#[test]
fn test_choose_row_tier_boundaries() {
    // Empty / degenerate.
    assert_eq!(choose_row_tier(0, 0), RowTier::Thread);
    // mean == 0 (all-empty rows) → Thread.
    assert_eq!(choose_row_tier(100, 0), RowTier::Thread);
    // mean exactly at the thread ceiling → Thread.
    assert_eq!(
        choose_row_tier(10, 10 * TIER_THREAD_MAX_MEAN_NNZ),
        RowTier::Thread
    );
    // Just above the thread ceiling → Warp.
    assert_eq!(
        choose_row_tier(10, 10 * (TIER_THREAD_MAX_MEAN_NNZ + 1)),
        RowTier::Warp
    );
    // Mid-range → Warp.
    assert_eq!(choose_row_tier(1000, 1000 * 64), RowTier::Warp);
    // Just below the block floor → Warp.
    assert_eq!(
        choose_row_tier(10, 10 * (TIER_BLOCK_MIN_MEAN_NNZ - 1)),
        RowTier::Warp
    );
    // At/above the block floor → Block.
    assert_eq!(
        choose_row_tier(10, 10 * TIER_BLOCK_MIN_MEAN_NNZ),
        RowTier::Block
    );
    assert_eq!(choose_row_tier(10, 10 * 4096), RowTier::Block);
}

/// Build a CSR (`indptr`, `data`) with an explicit per-row nnz. `indices`
/// are irrelevant to the normalize/log1p kernels (they only touch
/// `indptr` + `data`), so they're omitted.
fn csr_with_row_nnz(row_nnz: &[usize], seed: u64) -> (Vec<i64>, Vec<f32>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indptr = vec![0i64];
    let mut data = Vec::new();
    for &k in row_nnz {
        for _ in 0..k {
            data.push(rng.gen_range(0.5..20.5));
        }
        indptr.push(data.len() as i64);
    }
    (indptr, data)
}

/// Run `normalize` (optionally fused with log1p) through a forced tier and
/// return the transformed `data`.
fn run_normalize_tier(
    dev: &GpuDevice,
    indptr: &[i64],
    data: &[f32],
    n_rows: usize,
    target_sum: f32,
    do_log1p: bool,
    tier: RowTier,
) -> Vec<f32> {
    let d_indptr = dev.htod_copy(indptr).unwrap();
    let mut d_data = dev.htod_copy(data).unwrap();
    {
        let iv = d_indptr.slice(..);
        let mut dv = d_data.slice_mut(..);
        apply_fused_ops_with_tier(
            dev,
            &iv,
            &mut dv,
            n_rows,
            Some(target_sum),
            do_log1p,
            None,
            tier,
        )
        .unwrap();
    }
    dev.synchronize().unwrap();
    dev.dtoh_copy(&d_data).unwrap()
}

/// Every tier (Thread / Warp / Block) must match the CPU reference on the
/// same input, including empty and single-nonzero rows. Forced tiers let us
/// exercise the warp/block kernels on a small deterministic matrix.
#[test]
fn test_gpu_normalize_tiers_match_cpu() {
    let dev = require_gpu!();
    // Mix of empty, tiny, medium, and dense rows.
    let row_nnz = [0usize, 1, 5, 33, 64, 200, 1, 0, 129];
    let n_rows = row_nnz.len();
    let (indptr, data) = csr_with_row_nnz(&row_nnz, 0xA11CE);
    let target_sum = 1e4f32;

    let mut cpu = data.clone();
    cpu_normalize(&indptr, &mut cpu, target_sum as f64);

    for tier in [RowTier::Thread, RowTier::Warp, RowTier::Block] {
        let gpu = run_normalize_tier(&dev, &indptr, &data, n_rows, target_sum, false, tier);
        let mut max_rel = 0.0f64;
        for i in 0..cpu.len() {
            let diff = (gpu[i] as f64 - cpu[i] as f64).abs();
            let denom = (cpu[i] as f64).abs().max(1e-10);
            max_rel = max_rel.max(diff / denom);
        }
        // f32-accumulation bound: the Thread tier sums a row's nonzeros
        // sequentially in f32, whose drift vs the f64 CPU reference grows
        // with row nnz (~nnz·2^-24, here ~1e-5 for the 200-nnz row). The
        // warp/block tiers tree-reduce and stay tighter. A real logic error
        // would yield O(1) diffs, not 5e-5.
        assert!(
            max_rel < 5e-5,
            "normalize tier {tier:?} max rel err = {max_rel}"
        );
    }
}

/// Same, for the fused normalize+log1p kernels (f32 vs f64 → 1e-4).
#[test]
fn test_gpu_normalize_log1p_tiers_match_cpu() {
    let dev = require_gpu!();
    let row_nnz = [0usize, 1, 7, 50, 64, 300, 2, 0, 257];
    let n_rows = row_nnz.len();
    let (indptr, data) = csr_with_row_nnz(&row_nnz, 0xBEE5);
    let target_sum = 1e4f32;

    let mut cpu = data.clone();
    cpu_fused_normalize_log1p(&indptr, &mut cpu, target_sum as f64);

    for tier in [RowTier::Thread, RowTier::Warp, RowTier::Block] {
        let gpu = run_normalize_tier(&dev, &indptr, &data, n_rows, target_sum, true, tier);
        let mut max_rel = 0.0f64;
        for i in 0..cpu.len() {
            let diff = (gpu[i] as f64 - cpu[i] as f64).abs();
            let denom = (cpu[i] as f64).abs().max(1e-10);
            max_rel = max_rel.max(diff / denom);
        }
        assert!(
            max_rel < 1e-4,
            "normalize_log1p tier {tier:?} max rel err = {max_rel}"
        );
    }
}

/// Heavy intra-shard skew (a few very dense rows among many sparse/empty)
/// stresses the cooperative reductions' empty/zero-sum guards and the
/// strided write loops. Both nonzero-parallel tiers must still match CPU.
#[test]
fn test_gpu_normalize_skewed_rows() {
    let dev = require_gpu!();
    let mut row_nnz = vec![0usize; 200];
    for r in (0..200).step_by(3) {
        row_nnz[r] = 2;
    }
    row_nnz[50] = 800;
    row_nnz[123] = 1500;
    let n_rows = row_nnz.len();
    let (indptr, data) = csr_with_row_nnz(&row_nnz, 0x5EED_BEEF);
    let target_sum = 1e4f32;

    let mut cpu = data.clone();
    cpu_normalize(&indptr, &mut cpu, target_sum as f64);

    for tier in [RowTier::Warp, RowTier::Block] {
        let gpu = run_normalize_tier(&dev, &indptr, &data, n_rows, target_sum, false, tier);
        for i in 0..cpu.len() {
            let diff = (gpu[i] as f64 - cpu[i] as f64).abs();
            let denom = (cpu[i] as f64).abs().max(1e-10);
            assert!(
                diff / denom < 1e-5,
                "skewed tier {tier:?} mismatch at nnz {i}"
            );
        }
    }
}

/// Auto-dispatch (`gpu_normalize` → `choose_row_tier`) must land in the
/// expected tier for the warp- and block-density regimes and match CPU.
#[test]
fn test_gpu_normalize_auto_dispatch_warp_and_block() {
    let dev = require_gpu!();
    let target_sum = 1e4f32;

    // Warp regime: 256 rows × mean ~64 nnz.
    let warp_nnz = vec![64usize; 256];
    assert_eq!(
        choose_row_tier(256, 256 * 64),
        RowTier::Warp,
        "expected warp regime"
    );
    // Block regime: 64 rows × mean ~600 nnz.
    let block_nnz = vec![600usize; 64];
    assert_eq!(
        choose_row_tier(64, 64 * 600),
        RowTier::Block,
        "expected block regime"
    );

    for (label, row_nnz, n_rows) in [("warp", warp_nnz, 256usize), ("block", block_nnz, 64usize)] {
        let (indptr, data) = csr_with_row_nnz(&row_nnz, 0xD15A);
        let d_indptr = dev.htod_copy(&indptr).unwrap();
        let mut d_data = dev.htod_copy(&data).unwrap();
        gpu_normalize(&dev, &d_indptr, &mut d_data, n_rows, target_sum).unwrap();
        dev.synchronize().unwrap();
        let gpu = dev.dtoh_copy(&d_data).unwrap();

        let mut cpu = data.clone();
        cpu_normalize(&indptr, &mut cpu, target_sum as f64);
        let mut max_rel = 0.0f64;
        for i in 0..cpu.len() {
            let diff = (gpu[i] as f64 - cpu[i] as f64).abs();
            let denom = (cpu[i] as f64).abs().max(1e-10);
            max_rel = max_rel.max(diff / denom);
        }
        assert!(max_rel < 1e-5, "auto {label} max rel err = {max_rel}");
    }
}

/// The log1p-only path now launches the one-thread-per-nonzero kernel.
/// Exercise it on a larger matrix (the small `test_gpu_log1p_only` above
/// already covers it through the unchanged public API).
#[test]
fn test_gpu_log1p_nnz_large() {
    let dev = require_gpu!();
    let row_nnz = vec![37usize; 1000];
    let n_rows = row_nnz.len();
    let (indptr, data) = csr_with_row_nnz(&row_nnz, 0x106_1A6E);

    let d_indptr = dev.htod_copy(&indptr).unwrap();
    let mut d_data = dev.htod_copy(&data).unwrap();
    gpu_log1p(&dev, &d_indptr, &mut d_data, n_rows).unwrap();
    dev.synchronize().unwrap();
    let gpu = dev.dtoh_copy(&d_data).unwrap();

    let mut cpu = data.clone();
    cpu_log1p(&indptr, &mut cpu);
    for i in 0..cpu.len() {
        assert!(
            (gpu[i] - cpu[i]).abs() < 1e-6,
            "log1p_nnz mismatch at {i}: gpu={}, cpu={}",
            gpu[i],
            cpu[i]
        );
    }
}
