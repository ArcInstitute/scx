"""Before/after wall + peak RSS for task 2.5 (connectivity + Leiden).

Times pyscx.accel.neighbors (build_knn_graph -> compute_connectivities) and
pyscx.accel.leiden on a synthetic PCA embedding. Run once with the pre-2.5 .so
(baseline) and again after rebuilding with the 2.5 changes.
"""

import resource
import time

import anndata as ad
import numpy as np

import pyscx

N = 50_000
D = 50
rng = np.random.default_rng(0)
# 8 gaussian blobs in 50-D so Leiden has real structure.
centers = rng.normal(0, 10, size=(8, D))
labels = rng.integers(0, 8, size=N)
X = centers[labels] + rng.normal(0, 1.0, size=(N, D))
adata = ad.AnnData(X=np.zeros((N, 1), dtype=np.float32))
adata.obsm["X_pca"] = X.astype(np.float32)

t0 = time.perf_counter()
pyscx.accel.neighbors(adata, n_neighbors=15, use_rep="X_pca", device="cpu")
t_neighbors = time.perf_counter() - t0

t0 = time.perf_counter()
pyscx.accel.leiden(adata, resolution=1.0, random_state=42, device="cpu")
t_leiden = time.perf_counter() - t0

peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
n_clusters = len(set(adata.obs["leiden"].tolist()))
print(f"n_obs={N} d={D}")
print(f"neighbors wall: {t_neighbors*1000:.1f} ms")
print(f"leiden wall:    {t_leiden*1000:.1f} ms")
print(f"peak RSS:       {peak_mb:.0f} MB")
print(f"n_clusters:     {n_clusters}")
