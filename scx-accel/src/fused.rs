//! Device-resident fused GPU pipelines (V3 plan Phase 2.3).
//!
//! These run multiple GPU accelerators back-to-back while keeping intermediate
//! results resident on the device, eliminating host round-trips. The first such
//! pipeline is [`pca_then_knn_gpu`]: GPU PCA produces a device-resident
//! embedding that feeds straight into CAGRA kNN, so the embedding never makes
//! the GPU → host → GPU trip the separate `pca` + `neighbors` calls incur.
//!
//! Gated entirely behind the `gpu` feature.

#![cfg(feature = "gpu")]

use scx_format::ShardSource;

use crate::error::{AccelError, Result};
use crate::neighbors::{build_knn_csr, compute_connectivities, KnnResult};
use crate::pca::PcaResult;
use crate::umap::{random_init, spectral_init, UmapResult};

/// Run GPU PCA then GPU CAGRA kNN with the embedding kept **device-resident**
/// between the two stages.
///
/// PCA is computed via the device-returning entry points
/// ([`scx_gpu::gpu_randomized_pca_device`] / [`scx_gpu::gpu_covariance_pca_device`],
/// selected by `use_covariance`); the resulting [`scx_gpu::DeviceEmbedding`]
/// is handed directly to [`scx_gpu::gpu_knn_cagra_device`] as the CAGRA
/// dataset. The kNN neighbor indices/distances are downloaded once and the
/// fuzzy connectivities are built on the CPU (matching
/// [`crate::build_knn_graph_gpu`]); the embedding is downloaded once for the
/// returned [`PcaResult`].
///
/// Returns `(PcaResult, KnnResult)` — byte-for-byte equivalent to running
/// [`crate::randomized_pca_gpu`] (or [`crate::covariance_pca_gpu`]) followed by
/// [`crate::build_knn_graph_gpu`] on its embedding, minus the intermediate host
/// round-trip of the embedding.
///
/// The fuzzy-graph step stays on the host until V3 Phase 2.4 adds the
/// device-resident fuzzy-simplicial-set kernel.
#[allow(clippy::too_many_arguments)]
pub fn pca_then_knn_gpu<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: scx_gpu::QrMethod,
    use_covariance: bool,
    n_neighbors: usize,
    tuning: scx_gpu::GpuPcaTuning,
) -> Result<(PcaResult, KnnResult)> {
    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;

    // Stage 1: GPU PCA → device-resident embedding (stays on the GPU).
    let pca_dev = if use_covariance {
        scx_gpu::gpu_covariance_pca_device(&dev, source, n_components, zero_center)
    } else {
        scx_gpu::gpu_randomized_pca_device(
            &dev,
            source,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            seed,
            qr_method,
            tuning,
        )
    }
    .map_err(|e| AccelError::LinAlg(format!("GPU PCA failed: {e}")))?;

    let n_obs = pca_dev.n_obs;

    if n_neighbors == 0 || n_neighbors > n_obs {
        return Err(AccelError::InvalidInput(format!(
            "n_neighbors ({n_neighbors}) must be in [1, n_obs ({n_obs})]"
        )));
    }

    // Stage 2: CAGRA kNN reads the embedding directly off the device — no
    // GPU → host → GPU round-trip.
    let knn_dev = scx_gpu::gpu_knn_cagra_device(&dev, &pca_dev.embeddings, n_neighbors)
        .map_err(|e| AccelError::InvalidInput(format!("GPU kNN (CAGRA): {e}")))?;
    let gpu_knn = knn_dev
        .to_host(&dev)
        .map_err(|e| AccelError::InvalidInput(format!("GPU kNN download: {e}")))?;

    // Build the KnnResult (fuzzy graph on the host — same as build_knn_graph_gpu).
    let indices: Vec<usize> = gpu_knn.indices.iter().map(|&idx| idx as usize).collect();
    let distances: Vec<f64> = gpu_knn.distances.iter().map(|&d| d as f64).collect();
    let (dist_indptr, dist_indices, dist_data) =
        build_knn_csr(&indices, &distances, n_obs, n_neighbors);
    let (conn_indptr, conn_indices, conn_data) =
        compute_connectivities(&indices, &distances, n_obs, n_neighbors);
    let knn = KnnResult {
        indices,
        distances,
        conn_indptr,
        conn_indices,
        conn_data,
        dist_indptr,
        dist_indices,
        dist_data,
        n_neighbors,
        n_obs,
    };

    // Build the PcaResult — download the embedding once (for obsm["X_pca"]).
    let emb_f32 = pca_dev
        .embeddings
        .to_host(&dev)
        .map_err(|e| AccelError::LinAlg(format!("PCA embedding download: {e}")))?;
    let embeddings: Vec<f64> = emb_f32.iter().map(|&v| v as f64).collect();
    let components: Vec<f64> = pca_dev.components.iter().map(|&v| v as f64).collect();
    let pca = PcaResult {
        embeddings,
        components,
        variance_explained: pca_dev.variance_explained,
        variance_ratio: pca_dev.variance_ratio,
        mean: pca_dev.mean,
        n_components: pca_dev.n_components,
        n_obs: pca_dev.n_obs,
        n_vars: pca_dev.n_vars,
        graph_replayed: Some(pca_dev.graph_replayed),
    };

    Ok((pca, knn))
}

/// Run GPU PCA → CAGRA kNN → fuzzy graph → UMAP with the embedding **and** the
/// connectivity graph kept device-resident across the whole chain (V3 Phase 2.4).
///
/// Extends [`pca_then_knn_gpu`] with the device fuzzy-simplicial-set kernel
/// ([`scx_gpu::gpu_fuzzy_simplicial_set_device`]) and device-resident UMAP
/// ([`scx_gpu::gpu_umap_from_device_graph`]): the symmetrized connectivity CSR
/// is built on-device (replacing the host `compute_connectivities`) and feeds
/// UMAP directly, so the graph is downloaded **once** — for the returned
/// `obsp` matrices and the host spectral init — rather than symmetrized on the
/// CPU and re-uploaded for SGD.
///
/// Returns `(PcaResult, KnnResult, UmapResult)`. The `KnnResult` carries the
/// device-built connectivities (`conn_*`) and the raw kNN distance CSR
/// (`dist_*`); UMAP is seeded by a host spectral init computed from those
/// connectivities (random fallback for tiny inputs).
#[allow(clippy::too_many_arguments)]
pub fn pca_then_knn_umap_gpu<S: ShardSource + Sync>(
    device_id: usize,
    source: &S,
    n_components: usize,
    n_oversamples: usize,
    n_power_iterations: usize,
    zero_center: bool,
    seed: u64,
    qr_method: scx_gpu::QrMethod,
    use_covariance: bool,
    n_neighbors: usize,
    umap_n_components: usize,
    n_epochs: usize,
    min_dist: f64,
    spread: f64,
    negative_sample_rate: usize,
    umap_learning_rate: f64,
    umap_seed: u64,
    tuning: scx_gpu::GpuPcaTuning,
) -> Result<(PcaResult, KnnResult, UmapResult)> {
    let dev = scx_gpu::GpuDevice::new(device_id)
        .map_err(|e| AccelError::LinAlg(format!("GPU init failed: {e}")))?;

    // Stage 1: GPU PCA → device-resident embedding.
    let pca_dev = if use_covariance {
        scx_gpu::gpu_covariance_pca_device(&dev, source, n_components, zero_center)
    } else {
        scx_gpu::gpu_randomized_pca_device(
            &dev,
            source,
            n_components,
            n_oversamples,
            n_power_iterations,
            zero_center,
            seed,
            qr_method,
            tuning,
        )
    }
    .map_err(|e| AccelError::LinAlg(format!("GPU PCA failed: {e}")))?;

    let n_obs = pca_dev.n_obs;
    if n_neighbors == 0 || n_neighbors > n_obs {
        return Err(AccelError::InvalidInput(format!(
            "n_neighbors ({n_neighbors}) must be in [1, n_obs ({n_obs})]"
        )));
    }

    // Stage 2: CAGRA kNN reads the embedding directly off the device.
    let knn_dev = scx_gpu::gpu_knn_cagra_device(&dev, &pca_dev.embeddings, n_neighbors)
        .map_err(|e| AccelError::InvalidInput(format!("GPU kNN (CAGRA): {e}")))?;

    // Stage 3: device fuzzy simplicial set (replaces host compute_connectivities).
    let fuzzy = scx_gpu::gpu_fuzzy_simplicial_set_device(&dev, &knn_dev, n_neighbors)
        .map_err(|e| AccelError::InvalidInput(format!("GPU fuzzy graph: {e}")))?;

    // Stage 4: single download — raw kNN (for indices/distances + dist CSR) and
    // the symmetric connectivities (for obsp + spectral init).
    let gpu_knn = knn_dev
        .to_host(&dev)
        .map_err(|e| AccelError::InvalidInput(format!("GPU kNN download: {e}")))?;
    let (conn_indptr, conn_indices, conn_data_f32) = fuzzy
        .to_host(&dev)
        .map_err(|e| AccelError::InvalidInput(format!("fuzzy graph download: {e}")))?;
    let conn_data: Vec<f64> = conn_data_f32.iter().map(|&v| v as f64).collect();

    let indices: Vec<usize> = gpu_knn.indices.iter().map(|&idx| idx as usize).collect();
    let distances: Vec<f64> = gpu_knn.distances.iter().map(|&d| d as f64).collect();
    let (dist_indptr, dist_indices, dist_data) =
        build_knn_csr(&indices, &distances, n_obs, n_neighbors);
    let knn = KnnResult {
        indices,
        distances,
        conn_indptr,
        conn_indices,
        conn_data,
        dist_indptr,
        dist_indices,
        dist_data,
        n_neighbors,
        n_obs,
    };

    // Stage 5: host spectral init from the connectivities (random fallback).
    let init_f64 = spectral_init(
        &knn.conn_indptr,
        &knn.conn_indices,
        &knn.conn_data,
        n_obs,
        umap_n_components,
        umap_seed,
    )
    .unwrap_or_else(|_| random_init(n_obs, umap_n_components, umap_seed));
    let init_f32: Vec<f32> = init_f64.iter().map(|&v| v as f32).collect();

    // Stage 6: device UMAP consumes the device fuzzy graph (no re-upload); only
    // the final coordinates are downloaded.
    let gpu_umap = scx_gpu::gpu_umap_from_device_graph(
        &dev,
        &fuzzy,
        umap_n_components,
        n_epochs,
        min_dist as f32,
        spread as f32,
        negative_sample_rate,
        umap_learning_rate as f32,
        umap_seed,
        Some(&init_f32),
    )
    .map_err(|e| AccelError::InvalidInput(format!("GPU UMAP: {e}")))?;
    let umap = UmapResult {
        embeddings: gpu_umap.embedding.iter().map(|&v| v as f64).collect(),
        n_obs,
        n_components: umap_n_components,
    };

    // Stage 7: PcaResult — download the embedding once (for obsm["X_pca"]).
    let emb_f32 = pca_dev
        .embeddings
        .to_host(&dev)
        .map_err(|e| AccelError::LinAlg(format!("PCA embedding download: {e}")))?;
    let embeddings: Vec<f64> = emb_f32.iter().map(|&v| v as f64).collect();
    let components: Vec<f64> = pca_dev.components.iter().map(|&v| v as f64).collect();
    let pca = PcaResult {
        embeddings,
        components,
        variance_explained: pca_dev.variance_explained,
        variance_ratio: pca_dev.variance_ratio,
        mean: pca_dev.mean,
        n_components: pca_dev.n_components,
        n_obs: pca_dev.n_obs,
        n_vars: pca_dev.n_vars,
        graph_replayed: Some(pca_dev.graph_replayed),
    };

    Ok((pca, knn, umap))
}

#[cfg(test)]
mod tests {
    use super::*;
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
        fn read_shard(&self, i: usize) -> scx_format::Result<ScxCsr> {
            Ok(self.shards[i].clone())
        }
    }

    fn random_two_shard_source(n_rows: usize, n_cols: usize, seed: u64) -> InMemorySource {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indptr: Vec<i64> = vec![0];
        let mut indices: Vec<i32> = Vec::new();
        let mut data: Vec<f32> = Vec::new();
        for _ in 0..n_rows {
            for c in 0..n_cols {
                if rng.gen_bool(0.1) {
                    indices.push(c as i32);
                    data.push(rng.gen_range(-1.0..1.0));
                }
            }
            indptr.push(indices.len() as i64);
        }
        let csr = ScxCsr::new_unchecked((n_rows, n_cols), indptr, indices, data);
        let mid = n_rows / 2;
        let p_mid = csr.indptr[mid] as usize;
        let shard0 = ScxCsr::new_unchecked(
            (mid, n_cols),
            csr.indptr[0..=mid].to_vec(),
            csr.indices[0..p_mid].to_vec(),
            csr.data[0..p_mid].to_vec(),
        );
        let shard1 = ScxCsr::new_unchecked(
            (n_rows - mid, n_cols),
            csr.indptr[mid..=n_rows]
                .iter()
                .map(|&p| p - csr.indptr[mid])
                .collect(),
            csr.indices[p_mid..].to_vec(),
            csr.data[p_mid..].to_vec(),
        );
        InMemorySource {
            shards: vec![shard0, shard1],
            n_obs: n_rows,
            n_vars: n_cols,
        }
    }

    /// The fused `pca_then_knn_gpu` must produce the same `PcaResult` and
    /// `KnnResult` as running `randomized_pca_gpu` then `build_knn_graph_gpu` on
    /// its embedding — the fused path only removes the host round-trip of the
    /// embedding between the two stages.
    #[test]
    fn test_pca_then_knn_gpu_matches_sequential() {
        if !crate::gpu_available() || !crate::cuvs_available() {
            eprintln!("GPU + cuVS not available — skipping fused PCA→kNN parity test");
            return;
        }

        let n_rows = 400;
        let n_cols = 60;
        let k = 12;
        let n_neighbors = 10;
        let source = random_two_shard_source(n_rows, n_cols, 7);

        // Fused device-resident path.
        let (pca, knn) = pca_then_knn_gpu(
            0,
            &source,
            k,
            10,
            2,
            true,
            7,
            scx_gpu::QrMethod::Householder,
            false, // randomized
            n_neighbors,
            scx_gpu::GpuPcaTuning::default(),
        )
        .unwrap();

        // (1) PCA parity: a *separate* GPU randomized-PCA run. Two GPU PCA runs
        //     differ only at f32 noise level (run-to-run SpMM/QR nondeterminism),
        //     so compare with tolerance, not exact equality.
        let pca_ref = crate::randomized_pca_gpu(
            0,
            &source,
            k,
            10,
            2,
            true,
            7,
            scx_gpu::QrMethod::Householder,
            scx_gpu::GpuPcaTuning::default(),
        )
        .unwrap();
        assert_eq!(pca.embeddings.len(), pca_ref.embeddings.len());
        for (a, b) in pca.embeddings.iter().zip(pca_ref.embeddings.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "fused PCA {a} vs ref {b}"
            );
        }
        assert_eq!(pca.components.len(), pca_ref.components.len());
        for (a, b) in pca.components.iter().zip(pca_ref.components.iter()) {
            assert!((a - b).abs() <= 1e-3, "fused PCA loading {a} vs ref {b}");
        }

        // (2) kNN handoff correctness: run CAGRA on the fused run's OWN embedding
        //     (the exact f32 values the device-resident kNN saw), isolating the
        //     handoff from PCA nondeterminism. The two CAGRA builds may still
        //     break ties between equidistant neighbors in a different order, so
        //     compare per-row neighbor *sets*, not element order.
        let fused_emb_f32: Vec<f32> = pca.embeddings.iter().map(|&v| v as f32).collect();
        let knn_ref =
            crate::build_knn_graph_gpu(0, &fused_emb_f32, n_rows, k, n_neighbors).unwrap();
        assert_eq!(knn.n_obs, n_rows);
        assert_eq!(knn.n_neighbors, n_neighbors);
        assert_eq!(knn.indices.len(), knn_ref.indices.len());
        for i in 0..n_rows {
            let lo = i * n_neighbors;
            let hi = lo + n_neighbors;
            let mut a = knn.indices[lo..hi].to_vec();
            let mut b = knn_ref.indices[lo..hi].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "row {i}: fused vs ref neighbor sets differ");
        }
    }

    /// The fused `pca_then_knn_umap_gpu` must produce a `PcaResult` / `KnnResult`
    /// consistent with `pca_then_knn_gpu` (same device PCA + CAGRA), and a valid
    /// `UmapResult` (correct shape, all finite) seeded by the device fuzzy graph.
    #[test]
    fn test_pca_then_knn_umap_gpu_matches_sequential() {
        if !crate::gpu_available() || !crate::cuvs_available() {
            eprintln!("GPU + cuVS not available — skipping fused PCA→kNN→UMAP test");
            return;
        }

        let n_rows = 400;
        let n_cols = 60;
        let k = 12;
        let n_neighbors = 10;
        let umap_comps = 2;
        let source = random_two_shard_source(n_rows, n_cols, 11);

        let (pca, knn, umap) = pca_then_knn_umap_gpu(
            0,
            &source,
            k,
            10,
            2,
            true,
            11,
            scx_gpu::QrMethod::Householder,
            false, // randomized PCA
            n_neighbors,
            umap_comps,
            50,  // n_epochs (low for test speed)
            0.1, // min_dist
            1.0, // spread
            5,   // negative_sample_rate
            1.0, // umap_learning_rate
            11,  // umap_seed
            scx_gpu::GpuPcaTuning::default(),
        )
        .unwrap();

        // PCA parity vs a separate device randomized-PCA run (f32 noise tol).
        let pca_ref = crate::randomized_pca_gpu(
            0,
            &source,
            k,
            10,
            2,
            true,
            11,
            scx_gpu::QrMethod::Householder,
            scx_gpu::GpuPcaTuning::default(),
        )
        .unwrap();
        assert_eq!(pca.embeddings.len(), pca_ref.embeddings.len());
        for (a, b) in pca.embeddings.iter().zip(pca_ref.embeddings.iter()) {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "fused PCA {a} vs ref {b}"
            );
        }

        // kNN handoff: neighbor sets match CAGRA on the fused run's own embedding.
        let fused_emb_f32: Vec<f32> = pca.embeddings.iter().map(|&v| v as f32).collect();
        let knn_ref =
            crate::build_knn_graph_gpu(0, &fused_emb_f32, n_rows, k, n_neighbors).unwrap();
        for i in 0..n_rows {
            let lo = i * n_neighbors;
            let hi = lo + n_neighbors;
            let mut a = knn.indices[lo..hi].to_vec();
            let mut b = knn_ref.indices[lo..hi].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            assert_eq!(a, b, "row {i}: fused vs ref neighbor sets differ");
        }

        // Connectivities are present and symmetric in shape.
        assert_eq!(knn.conn_indptr.len(), n_rows + 1);
        assert!(
            *knn.conn_indptr.last().unwrap() > 0,
            "device fuzzy graph produced no edges"
        );

        // UMAP embedding: correct shape, all finite.
        assert_eq!(umap.n_obs, n_rows);
        assert_eq!(umap.n_components, umap_comps);
        assert_eq!(umap.embeddings.len(), n_rows * umap_comps);
        for &v in &umap.embeddings {
            assert!(v.is_finite(), "UMAP embedding value not finite: {v}");
        }
    }
}
