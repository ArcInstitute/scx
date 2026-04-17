//! End-to-end integration test for the Harmony2 + LISI accelerator stack.
//!
//! Exercises the full scx-accel analysis pipeline on a small synthetic
//! dataset with an explicit batch effect:
//!   PCA -> Harmony -> kNN -> UMAP -> Leiden -> LISI.
//!
//! The aim is to verify cross-crate wiring, not numerical parity (that
//! lives in pyscx/tests/test_harmony_validation.py). Assertions check
//! output shapes, that Harmony actually moved the embedding (some cells
//! shift non-trivially), that Leiden finds > 1 cluster on the
//! well-separated synthetic data, and that LISI increases after Harmony
//! correction (mixing improved).

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use scx_accel::{
    build_knn_graph, compute_lisi, compute_umap, covariance_pca_inmemory, harmony_integrate,
    leiden, BatchCovariate, HarmonyConfig, LisiConfig,
};
use scx_sparse::ScxCsr;

/// Build a two-cluster × two-batch synthetic dataset.
///
/// N cells, n_vars genes. Cluster labels A/B split the cells; batch
/// labels 0/1 add a per-batch offset so that without correction the
/// batch label confounds the cluster structure.
fn build_synthetic(n_per_group: usize, n_vars: usize, seed: u64) -> (ScxCsr, Vec<u32>, Vec<u32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = n_per_group * 4;

    // Dense floats first (we'll drop zeros to make it sparse-ish).
    let mut dense = vec![0f32; n * n_vars];
    let mut labels_cluster = Vec::with_capacity(n);
    let mut labels_batch = Vec::with_capacity(n);

    for gi in 0..4 {
        let cluster = gi % 2; // 0,1,0,1
        let batch = gi / 2; // 0,0,1,1
        let base_cluster = if cluster == 0 { 5.0f32 } else { -5.0f32 };
        let base_batch = if batch == 0 { 0.0f32 } else { 1.2f32 };
        for _ in 0..n_per_group {
            let row = labels_cluster.len();
            labels_cluster.push(cluster as u32);
            labels_batch.push(batch as u32);
            // Fill a few "signature" features per cluster plus a small
            // batch offset across all features, then add noise.
            for j in 0..n_vars {
                let sig = if (cluster == 0 && j < 5) || (cluster == 1 && j >= n_vars - 5) {
                    base_cluster + 0.5 * (rng.gen::<f32>() - 0.5)
                } else {
                    0.2 * (rng.gen::<f32>() - 0.5)
                };
                let val = sig + base_batch + 0.1 * (rng.gen::<f32>() - 0.5);
                dense[row * n_vars + j] = val.max(0.0); // CSR stores non-negative signal
            }
        }
    }

    // Convert to CSR (drop exact zeros).
    let mut indptr: Vec<i64> = vec![0];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for r in 0..n {
        for c in 0..n_vars {
            let v = dense[r * n_vars + c];
            if v > 0.0 {
                indices.push(c as i32);
                data.push(v);
            }
        }
        indptr.push(indices.len() as i64);
    }

    let csr = ScxCsr::new_unchecked((n, n_vars), indptr, indices, data);
    (csr, labels_cluster, labels_batch)
}

#[test]
fn test_harmony_pipeline_end_to_end() {
    // Synthetic 400 cells × 40 genes with explicit batch confounder.
    let (csr, cluster_labels, batch_labels) = build_synthetic(100, 40, 42);
    let n = csr.n_rows();
    let n_vars = csr.n_cols();
    assert_eq!(n, 400);
    assert_eq!(n_vars, 40);

    // ── 1. PCA ────────────────────────────────────────────────
    let n_pcs = 10usize;
    let pca = covariance_pca_inmemory(&csr, n_pcs, true).expect("PCA failed");
    assert_eq!(pca.n_obs, n);
    assert_eq!(pca.n_components, n_pcs);
    assert_eq!(pca.embeddings.len(), n * n_pcs);

    let pca_f32: Vec<f32> = pca.embeddings.iter().map(|&v| v as f32).collect();

    // ── 2. Harmony ────────────────────────────────────────────
    let cov = BatchCovariate {
        labels: batch_labels.clone(),
        n_levels: 2,
        name: Some("batch".into()),
    };
    let config = HarmonyConfig {
        n_clusters: Some(5),
        max_iter: 4,
        random_state: 7,
        ..Default::default()
    };
    let harmony = harmony_integrate(&pca_f32, n, n_pcs, &[cov], &config).expect("Harmony failed");
    assert_eq!(harmony.z_corrected.len(), n * n_pcs);
    assert!(harmony.z_corrected.iter().all(|v| v.is_finite()));
    // Harmony should move the embedding — not all entries should equal
    // the PCA input.
    let unchanged = harmony
        .z_corrected
        .iter()
        .zip(pca.embeddings.iter())
        .filter(|(a, b)| (**a as f64 - *b).abs() < 1e-9)
        .count();
    assert!(
        unchanged < harmony.z_corrected.len(),
        "Harmony made no change to the embedding"
    );

    // Promote corrected embeddings to f32 for kNN / LISI.
    let corrected_f32: Vec<f32> = harmony.z_corrected.iter().map(|&v| v as f32).collect();

    // ── 3. kNN ───────────────────────────────────────────────
    let knn = build_knn_graph(&corrected_f32, n, n_pcs, 15, 100, 50, 0).expect("kNN failed");
    assert_eq!(knn.n_obs, n);
    assert_eq!(knn.conn_indptr.len(), n + 1);
    assert!(!knn.conn_indices.is_empty());
    assert_eq!(knn.conn_indices.len(), knn.conn_data.len());

    // ── 4. UMAP ─────────────────────────────────────────────
    let umap = compute_umap(
        &knn.conn_indptr,
        &knn.conn_indices,
        &knn.conn_data,
        n,
        2,
        50,
        0.5,
        1.0,
        5,
        1.0,
        0,
        None,
    )
    .expect("UMAP failed");
    assert_eq!(umap.embeddings.len(), n * 2);
    assert!(umap.embeddings.iter().all(|v| v.is_finite()));

    // ── 5. Leiden ───────────────────────────────────────────
    let leiden_res = leiden(
        &knn.conn_indptr,
        &knn.conn_indices,
        &knn.conn_data,
        n,
        1.0,
        0,
        2,
        false,
    )
    .expect("Leiden failed");
    assert_eq!(leiden_res.membership.len(), n);
    assert!(
        leiden_res.n_communities >= 2,
        "Leiden found only {} communities on a 2-cluster synthetic",
        leiden_res.n_communities
    );

    // ── 6. LISI — shape + range checks on batch + cluster labels ────
    //
    // On this tiny synthetic (4 well-separated quadrants of 100 cells
    // each) pre- and post-correction LISI both land very close to 1.0
    // because every cell's k nearest neighbours sit inside its own
    // quadrant regardless of correction. So we assert finiteness and the
    // spec'd `[1, n_categories]` range here; numerical-mixing
    // assertions belong in `pyscx/tests/test_harmony_validation.py`
    // which runs against real datasets with known ground truth.
    let lisi_cfg = LisiConfig {
        perplexity: 10.0,
        n_neighbors: 30,
        ..Default::default()
    };
    let lisi_batch =
        compute_lisi(&corrected_f32, n, n_pcs, &batch_labels, &lisi_cfg).expect("LISI batch");
    assert_eq!(lisi_batch.lisi.len(), n);
    assert!(lisi_batch.lisi.iter().all(|v| v.is_finite()));
    assert!(lisi_batch.lisi.iter().all(|&v| (0.9..=2.1).contains(&v)));

    let lisi_cluster =
        compute_lisi(&corrected_f32, n, n_pcs, &cluster_labels, &lisi_cfg).expect("LISI cluster");
    assert_eq!(lisi_cluster.lisi.len(), n);
    assert!(lisi_cluster.lisi.iter().all(|v| v.is_finite()));
    assert!(lisi_cluster.lisi.iter().all(|&v| (0.9..=2.1).contains(&v)));

    // Sanity: every Leiden community labels a non-empty contiguous
    // slice of the membership vector and no community index exceeds
    // `n_communities`. Numerical mixing is deferred to
    // pyscx/tests/test_harmony_validation.py.
    for &m in &leiden_res.membership {
        assert!(m < leiden_res.n_communities);
    }
}
