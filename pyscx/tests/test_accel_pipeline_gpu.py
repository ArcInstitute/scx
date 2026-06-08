"""Full GPU pipeline integration test — Phase 7.5.

Runs the end-to-end pyscx.accel pipeline on GPU:
    normalize_total → log1p → highly_variable_genes → pca → neighbors → umap → leiden

Gated on `pyscx.accel.gpu_info() is not None` — silently skipped on CPU-only
hosts. Serves as a smoke test that every GPU dispatch wires up correctly and
writes to the expected AnnData slots. Not a parity test — per-op correctness
is covered by test_accel_pca_gpu.py and the Rust-level unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _gpu_available() -> bool:
    try:
        import pyscx

        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — skipping GPU pipeline integration test",
)


@gpu_only
def test_full_gpu_pipeline(tmp_path):
    """normalize → log1p → HVG → PCA → kNN → UMAP → Leiden on SCX-backed data.

    Asserts that:
    * Every op completes without raising.
    * `adata.obsm["X_pca"]`, `adata.obsm["X_umap"]` are populated.
    * `adata.obs["leiden"]` exists with > 1 unique label.
    * Result shapes match expectations.
    """
    import anndata
    import pyscx

    # Synthetic stand-in for pbmc3k. Keeps runtime tight while still exercising
    # multi-shard streaming (default shard size).
    rng = np.random.default_rng(2026)
    n_obs, n_vars = 2_700, 5_000
    dense = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    mask = rng.random((n_obs, n_vars)) > 0.05
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata_src = anndata.AnnData(X=x)

    # Write to SCX so the pipeline exercises backed/lazy paths.
    scx_path = str(tmp_path / "pipeline.scx")
    pyscx.from_anndata(adata_src, scx_path)
    adata = pyscx.open(scx_path).to_anndata()

    # 1. normalize_total — GPU eager; materializes X to scipy CSR.
    pyscx.accel.normalize_total(adata, target_sum=1e4, device="gpu")
    # 2. log1p — fusion-marker path re-runs fused normalize+log1p over the
    #    original backed source.
    pyscx.accel.log1p(adata, device="gpu")

    # 3. HVG — narrow to single-batch seurat_v3 so the GPU dispatch fires.
    pyscx.accel.highly_variable_genes(
        adata, n_top_genes=500, flavor="seurat_v3", device="gpu",
    )
    assert "highly_variable" in adata.var
    assert adata.var["highly_variable"].sum() == 500

    # 4. PCA — in-memory `device="gpu"` routes to rapids-singlecell (the native
    #    in-VRAM covariance core was removed in Phase 3.2).
    pyscx.accel.pca(adata, n_comps=30, device="gpu", random_state=0)
    assert adata.obsm["X_pca"].shape == (n_obs, 30)
    assert "PCs" in adata.varm
    assert "pca" in adata.uns

    # 5. kNN.
    pyscx.accel.neighbors(adata, n_neighbors=15, device="gpu")
    assert "connectivities" in adata.obsp
    assert "distances" in adata.obsp

    # 6. UMAP.
    pyscx.accel.umap(adata, device="gpu")
    assert "X_umap" in adata.obsm
    assert adata.obsm["X_umap"].shape == (n_obs, 2)

    # 7. Leiden — Rust-native path wins regardless of device setting; still
    # must populate adata.obs["leiden"].
    pyscx.accel.leiden(adata, resolution=1.0, device="gpu")
    assert "leiden" in adata.obs
    n_clusters = int(adata.obs["leiden"].astype(str).nunique())
    assert n_clusters > 1, f"Leiden produced only {n_clusters} cluster"
    assert n_clusters < n_obs, "Leiden produced as many clusters as cells"
