//! Device-resident UMAP fuzzy-simplicial-set construction (V3 plan Phase 2.4).
//!
//! [`gpu_fuzzy_simplicial_set_device`] builds the symmetrized fuzzy connectivity
//! graph entirely on the GPU from a raw [`DeviceKnnGraph`], producing a
//! [`DeviceFuzzyGraph`] (device CSR) that GPU UMAP consumes without the previous
//! host round-trip through `scx_accel::neighbors::compute_connectivities`.
//!
//! Stages (see `kernels/fuzzy_simplicial_set.cu` for the kernel-level detail):
//!
//! 1. `fuzzy_membership_kernel` — per-row self-filter + `sqrt` + bandwidth
//!    (sigma) search + membership strengths, emitting a forward CSR `M` with
//!    exactly `k` column-sorted entries per row.
//! 2. [`cusparse::csr_transpose`] — `T = Mᵀ` (cuSPARSE owns the hub-safe scatter
//!    + per-row sort), so the merge sees two column-sorted streams.
//! 3. `fuzzy_union_count_kernel` — per-row merge-count of `union(M[i], T[i])`;
//!    the row counts are exclusive-scanned on the host (`n_obs` ints only — no
//!    edge data round-trips) into the output `i64` row pointers.
//! 4. `fuzzy_union_fill_kernel` — per-row merge writing `conn = a + b - a*b`.
//!
//! The result's [`DeviceFuzzyGraph::to_host`] matches
//! `scx_accel::neighbors::compute_connectivities` within f32 tolerance.

use cudarc::driver::safe::LaunchConfig;
use cudarc::driver::PushKernelArg;

use crate::cusparse::{csr_transpose, CusparseHandle};
use crate::device::GpuDevice;
use crate::device_resident::{DeviceFuzzyGraph, DeviceKnnGraph};
use crate::error::GpuError;

/// Compiled PTX for the fuzzy-simplicial-set kernels (produced by build.rs).
const FUZZY_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/fuzzy_simplicial_set.ptx"));

/// Must match `FUZZY_MAX_K` in `kernels/fuzzy_simplicial_set.cu`.
const FUZZY_MAX_K: usize = 256;

/// Build a device-resident UMAP fuzzy simplicial set from a raw kNN graph.
///
/// `knn` is the raw CAGRA output (`search_k`-wide rows, L2-squared distances,
/// self-hits included); `n_neighbors` is the post-self-filter neighbor count
/// (`k`). The returned [`DeviceFuzzyGraph`] is the symmetrized connectivity CSR
/// (`indptr` `i64`, `indices` `i32`, `data` `f32`), equivalent to running
/// `scx_accel::neighbors::compute_connectivities` on `knn.to_host()`.
///
/// # Errors
///
/// Returns [`GpuError`] on kernel / cuSPARSE failure or when `n_neighbors`
/// exceeds the kernel's per-row scratch bound (`FUZZY_MAX_K`).
pub fn gpu_fuzzy_simplicial_set_device(
    dev: &GpuDevice,
    knn: &DeviceKnnGraph,
    n_neighbors: usize,
) -> Result<DeviceFuzzyGraph, GpuError> {
    let n_obs = knn.n_obs();
    let search_k = knn.search_k();
    let k = n_neighbors;

    if k == 0 || k > FUZZY_MAX_K {
        return Err(GpuError::ShapeMismatch {
            expected: format!("1 <= n_neighbors <= {FUZZY_MAX_K}"),
            got: format!("{k}"),
        });
    }
    if n_obs == 0 {
        return DeviceFuzzyGraph::new(
            dev.alloc_zeros::<i64>(1)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<f32>(0)?,
            0,
            0,
        );
    }

    let module = dev.load_module_cached(FUZZY_PTX)?;
    let n_obs_i = n_obs as i32;
    let search_k_i = search_k as i32;
    let k_i = k as i32;

    let threads: u32 = 256;
    let grid = (n_obs as u32).div_ceil(threads);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // --- Stage 1: forward CSR M (k column-sorted entries per row). ---
    let nnz_m = n_obs * k;
    let mut col_m = dev.alloc_zeros::<i32>(nnz_m)?;
    let mut val_m = dev.alloc_zeros::<f32>(nnz_m)?;
    {
        let func = module
            .load_function("fuzzy_membership_kernel")
            .map_err(|e| GpuError::KernelLaunchFailed(format!("load fuzzy_membership: {e}")))?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(knn.indices())
                .arg(knn.distances())
                .arg(&n_obs_i)
                .arg(&search_k_i)
                .arg(&k_i)
                .arg(&mut col_m)
                .arg(&mut val_m)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("fuzzy_membership_kernel: {e}")))?;
    }

    // --- Stage 2: transpose T = Mᵀ via cuSPARSE (column-sorted rows). ---
    // M's row pointers are regular (i*k); build them once on the host.
    let indptr_m: Vec<i32> = (0..=n_obs).map(|r| (r * k) as i32).collect();
    let d_indptr_m = dev.htod_copy(&indptr_m)?;
    let handle = CusparseHandle::new()?;
    let (t_indptr, t_indices, t_data) = csr_transpose(
        &handle,
        dev,
        dev.stream(),
        &d_indptr_m,
        &col_m,
        &val_m,
        n_obs,
        n_obs,
        nnz_m,
    )?;

    // --- Stage 3: merge-count union(M[i], T[i]) → host exclusive scan. ---
    let mut row_nnz = dev.alloc_zeros::<i32>(n_obs)?;
    {
        let func = module
            .load_function("fuzzy_union_count_kernel")
            .map_err(|e| GpuError::KernelLaunchFailed(format!("load fuzzy_union_count: {e}")))?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&col_m)
                .arg(&val_m)
                .arg(&t_indptr)
                .arg(&t_indices)
                .arg(&t_data)
                .arg(&n_obs_i)
                .arg(&k_i)
                .arg(&mut row_nnz)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("fuzzy_union_count_kernel: {e}")))?;
    }
    dev.synchronize()?;
    let row_nnz_host = dev.dtoh_copy(&row_nnz)?;

    let mut out_indptr_host = Vec::with_capacity(n_obs + 1);
    let mut acc: i64 = 0;
    out_indptr_host.push(0i64);
    for &cnt in &row_nnz_host {
        acc += cnt as i64;
        out_indptr_host.push(acc);
    }
    let total_nnz = acc as usize;
    let d_out_indptr = dev.htod_copy(&out_indptr_host)?;

    // --- Stage 4: fill the symmetric CSR (conn = a + b - a*b). ---
    let mut out_indices = dev.alloc_zeros::<i32>(total_nnz)?;
    let mut out_data = dev.alloc_zeros::<f32>(total_nnz)?;
    if total_nnz > 0 {
        let func = module
            .load_function("fuzzy_union_fill_kernel")
            .map_err(|e| GpuError::KernelLaunchFailed(format!("load fuzzy_union_fill: {e}")))?;
        unsafe {
            dev.stream()
                .launch_builder(&func)
                .arg(&col_m)
                .arg(&val_m)
                .arg(&t_indptr)
                .arg(&t_indices)
                .arg(&t_data)
                .arg(&d_out_indptr)
                .arg(&n_obs_i)
                .arg(&k_i)
                .arg(&mut out_indices)
                .arg(&mut out_data)
                .launch(cfg)
        }
        .map_err(|e| GpuError::KernelLaunchFailed(format!("fuzzy_union_fill_kernel: {e}")))?;
        dev.synchronize()?;
    }

    DeviceFuzzyGraph::new(d_out_indptr, out_indices, out_data, n_obs, total_nnz)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU reference for the fuzzy simplicial set, operating on the same raw
    /// CAGRA buffers the device kernels consume. Mirrors
    /// `DeviceKnnGraph::to_host` (self-filter + sqrt + pad) followed by
    /// `scx_accel::neighbors::compute_connectivities` — duplicated here so the
    /// `scx-gpu` test has no upward dependency on `scx-accel`.
    fn cpu_fuzzy_reference(
        knn_indices: &[u32],
        knn_dist_sq: &[f32],
        n_obs: usize,
        search_k: usize,
        k: usize,
    ) -> (Vec<i64>, Vec<i32>, Vec<f32>) {
        use std::collections::{HashMap, HashSet};

        // Per-row clean kNN (self-filter + sqrt + pad), as f64.
        let mut nbr = vec![0usize; n_obs * k];
        let mut dist = vec![0f64; n_obs * k];
        for i in 0..n_obs {
            let base = i * search_k;
            let mut count = 0;
            for j in 0..search_k {
                if count >= k {
                    break;
                }
                let nb = knn_indices[base + j] as usize;
                if nb == i {
                    continue;
                }
                nbr[i * k + count] = nb;
                let dsq = knn_dist_sq[base + j].max(0.0) as f64;
                dist[i * k + count] = dsq.sqrt();
                count += 1;
            }
            while count < k {
                nbr[i * k + count] = i;
                dist[i * k + count] = 0.0;
                count += 1;
            }
        }

        let target = (k as f64).ln() / std::f64::consts::LN_2;
        let find_sigma = |row: &[f64], rho: f64| -> f64 {
            let mut lo = 1e-10_f64;
            let mut hi = 1000.0_f64;
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                let sum: f64 = row.iter().map(|&d| (-(d - rho).max(0.0) / mid).exp()).sum();
                if sum > target {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            0.5 * (lo + hi)
        };

        let mut mu_map: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..n_obs {
            let row = &dist[i * k..i * k + k];
            let rho = row[0];
            let sigma = find_sigma(row, rho);
            for t in 0..k {
                let j = nbr[i * k + t];
                let d = row[t];
                let strength = if d <= rho || sigma <= 1e-10 {
                    1.0
                } else {
                    (-(d - rho) / sigma).exp()
                };
                mu_map.insert((i, j), strength);
            }
        }

        let mut sym: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_obs];
        let mut seen = HashSet::<(usize, usize)>::new();
        for &(i, j) in mu_map.keys() {
            if !seen.insert((i, j)) {
                continue;
            }
            seen.insert((j, i));
            let mu_ij = mu_map.get(&(i, j)).copied().unwrap_or(0.0);
            let mu_ji = mu_map.get(&(j, i)).copied().unwrap_or(0.0);
            let conn = mu_ij + mu_ji - mu_ij * mu_ji;
            if conn > 0.0 {
                sym[i].push((j, conn));
                if i != j {
                    sym[j].push((i, conn));
                }
            }
        }
        for row in &mut sym {
            row.sort_by_key(|&(c, _)| c);
        }

        let mut indptr = vec![0i64];
        let mut indices = Vec::new();
        let mut data = Vec::new();
        for row in &sym {
            for &(c, v) in row {
                indices.push(c as i32);
                data.push(v as f32);
            }
            indptr.push(indices.len() as i64);
        }
        (indptr, indices, data)
    }

    /// A small synthetic kNN graph: two tight clusters of points on a line, so
    /// neighbor sets and distances are deterministic and the symmetric pattern
    /// is non-trivial (cross-cluster asymmetry exercises the fuzzy union).
    fn synthetic_knn(n_obs: usize, k: usize) -> (Vec<u32>, Vec<f32>, usize) {
        // search_k = k + 1 (CAGRA returns self in the first slot).
        let search_k = k + 1;
        let mut indices = vec![0u32; n_obs * search_k];
        let mut dist_sq = vec![0f32; n_obs * search_k];
        // 1-D coordinates: cluster A at 0,1,2,..., cluster B offset by 1000.
        let coord = |i: usize| -> f32 {
            let half = n_obs / 2;
            if i < half {
                i as f32
            } else {
                1000.0 + (i - half) as f32
            }
        };
        for i in 0..n_obs {
            // Rank all other points by distance, take self + nearest k.
            let mut order: Vec<usize> = (0..n_obs).collect();
            order.sort_by(|&a, &b| {
                let da = (coord(a) - coord(i)).abs();
                let db = (coord(b) - coord(i)).abs();
                da.partial_cmp(&db).unwrap().then(a.cmp(&b))
            });
            for s in 0..search_k {
                let nb = order[s.min(n_obs - 1)];
                indices[i * search_k + s] = nb as u32;
                let d = coord(nb) - coord(i);
                dist_sq[i * search_k + s] = d * d;
            }
        }
        (indices, dist_sq, search_k)
    }

    #[test]
    fn test_gpu_fuzzy_matches_cpu_reference() {
        let dev = require_gpu!();
        let n_obs = 60;
        let k = 8;
        let (indices, dist_sq, search_k) = synthetic_knn(n_obs, k);

        let d_idx = dev.htod_copy(&indices).unwrap();
        let d_dist = dev.htod_copy(&dist_sq).unwrap();
        let knn = DeviceKnnGraph::new(d_idx, d_dist, n_obs, search_k, k).unwrap();

        let graph = gpu_fuzzy_simplicial_set_device(&dev, &knn, k).unwrap();
        let (g_indptr, g_indices, g_data) = graph.to_host(&dev).unwrap();

        let (c_indptr, c_indices, c_data) =
            cpu_fuzzy_reference(&indices, &dist_sq, n_obs, search_k, k);

        // Identical symmetric sparsity pattern.
        assert_eq!(g_indptr, c_indptr, "fuzzy graph indptr (pattern) mismatch");
        assert_eq!(
            g_indices, c_indices,
            "fuzzy graph column-index pattern mismatch"
        );
        // Values within f32 tolerance (device f32 sigma search vs host f64).
        assert_eq!(g_data.len(), c_data.len());
        for (idx, (a, b)) in g_data.iter().zip(c_data.iter()).enumerate() {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "conn[{idx}] device {a} vs cpu {b}"
            );
        }
    }

    #[test]
    fn test_gpu_fuzzy_symmetric() {
        let dev = require_gpu!();
        let n_obs = 40;
        let k = 6;
        let (indices, dist_sq, search_k) = synthetic_knn(n_obs, k);
        let d_idx = dev.htod_copy(&indices).unwrap();
        let d_dist = dev.htod_copy(&dist_sq).unwrap();
        let knn = DeviceKnnGraph::new(d_idx, d_dist, n_obs, search_k, k).unwrap();

        let graph = gpu_fuzzy_simplicial_set_device(&dev, &knn, k).unwrap();
        let (indptr, cols, data) = graph.to_host(&dev).unwrap();

        // Build a dense lookup and assert conn(i,j) == conn(j,i).
        let mut dense = std::collections::HashMap::<(i32, i32), f32>::new();
        for i in 0..n_obs {
            for p in indptr[i] as usize..indptr[i + 1] as usize {
                dense.insert((i as i32, cols[p]), data[p]);
            }
        }
        for (&(i, j), &v) in &dense {
            let vt = dense.get(&(j, i)).copied().unwrap_or(-1.0);
            assert!(
                (v - vt).abs() <= 1e-6,
                "asymmetry at ({i},{j}): {v} vs {vt}"
            );
        }
    }
}
