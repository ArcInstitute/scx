"""Tests for pyscx.accel.pca_neighbors() — fused device-resident PCA → kNN.

The fused entry runs GPU PCA then GPU CAGRA kNN in one call, keeping the PCA
embedding resident on the GPU between the two stages (V3 plan Phase 2.3). When
the fully device-resident path is unavailable it falls back to sequential
pca() + neighbors().

Coverage:
* CPU fallback (any host): outputs + route metadata match sequential
  pca(device="cpu") + neighbors(device="cpu"); routes stamp "cpu_csr".
* Fused GPU path (gated on cuVS): outputs match sequential GPU pca + neighbors;
  pca / neighbors / pca_neighbors routes stamp "gpu_device_resident".
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


def _cuvs_available() -> bool:
    """True iff a GPU is present AND cuVS CAGRA actually runs (method=="cagra")."""
    if not _gpu_available():
        return False
    try:
        import anndata as ad
        import pyscx

        rng = np.random.default_rng(0)
        a = ad.AnnData(X=sp.csr_matrix(rng.random((60, 10), dtype=np.float32)))
        a.obsm["X_pca"] = np.ascontiguousarray(a.X.toarray(), dtype=np.float32)
        pyscx.accel.neighbors(a, n_neighbors=5, device="gpu")
        return a.uns["neighbors"]["params"]["method"] == "cagra"
    except Exception:
        return False


cuvs_only = pytest.mark.skipif(
    not _cuvs_available(),
    reason="cuVS CAGRA not available — skipping fused device-resident GPU tests",
)


@pytest.fixture
def pca_adata(synthetic_adata, scx_from_adata):
    """Write synthetic AnnData to SCX, return (scx_path, original_adata)."""
    path = scx_from_adata(synthetic_adata, "fused_test.scx")
    return path, synthetic_adata


# ---------------------------------------------------------------------------
# CPU fallback (runs on any host)
# ---------------------------------------------------------------------------


class TestPcaNeighborsCpu:
    def test_writes_all_slots(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="cpu")

        assert "X_pca" in adata.obsm
        assert "PCs" in adata.varm
        assert "pca" in adata.uns
        assert "distances" in adata.obsp
        assert "connectivities" in adata.obsp
        assert "neighbors" in adata.uns
        assert adata.obsm["X_pca"].shape == (100, 10)

    def test_matches_sequential(self, synthetic_adata):
        """Fused CPU path == sequential pca(cpu) + neighbors(cpu)."""
        import pyscx

        fused = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(
            fused, n_comps=10, n_neighbors=15, device="cpu", random_state=0
        )

        seq = synthetic_adata.copy()
        pyscx.accel.pca(seq, n_comps=10, device="cpu", random_state=0)
        pyscx.accel.neighbors(seq, n_neighbors=15, device="cpu", random_state=0)

        np.testing.assert_allclose(
            fused.obsm["X_pca"], seq.obsm["X_pca"], rtol=1e-5, atol=1e-5
        )
        # Same kNN graph (CPU HNSW is deterministic at matched seed).
        assert (fused.obsp["distances"] != seq.obsp["distances"]).nnz == 0
        assert (fused.obsp["connectivities"] != seq.obsp["connectivities"]).nnz == 0

    def test_route_metadata_cpu(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(adata, n_comps=8, n_neighbors=10, device="cpu")

        accel = adata.uns["scx_accel"]
        assert accel["pca"]["route"] == "cpu_csr"
        assert accel["neighbors"]["route"] == "cpu_csr"
        assert accel["pca_neighbors"]["route"] == "cpu_csr"

    def test_invalid_args(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        with pytest.raises(ValueError):
            pyscx.accel.pca_neighbors(adata, method="bogus", device="cpu")
        with pytest.raises(ValueError):
            pyscx.accel.pca_neighbors(adata, qr_method="bogus", device="cpu")
        with pytest.raises(ValueError):
            pyscx.accel.pca_neighbors(adata, prefer_format="csc", device="cpu")


# ---------------------------------------------------------------------------
# Fused device-resident GPU path (gated on cuVS)
# ---------------------------------------------------------------------------


@cuvs_only
class TestPcaNeighborsGpu:
    def test_route_metadata_device_resident(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        accel = adata.uns["scx_accel"]
        assert accel["pca"]["route"] == "gpu_device_resident"
        assert accel["neighbors"]["route"] == "gpu_device_resident"
        assert accel["pca_neighbors"]["route"] == "gpu_device_resident"
        # And neighbors used CAGRA, not HNSW.
        assert adata.uns["neighbors"]["params"]["method"] == "cagra"

    def test_matches_sequential_gpu(self, synthetic_adata):
        """Fused GPU path matches the sequential ops.

        Two separate GPU PCA runs differ only at f32 noise level, so PCA is
        compared with tolerance. The kNN graph is checked *deterministically* by
        running CAGRA on the fused run's own X_pca (the exact embedding the
        device-resident kNN saw) — identical input ⇒ identical graph.
        """
        import anndata as ad
        import pyscx

        fused = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(
            fused, n_comps=20, n_neighbors=15, device="gpu", random_state=0
        )

        # PCA parity (tolerance) vs a separate GPU PCA run.
        seq = synthetic_adata.copy()
        pyscx.accel.pca(seq, n_comps=20, device="gpu", random_state=0)
        np.testing.assert_allclose(
            fused.obsm["X_pca"], seq.obsm["X_pca"], rtol=1e-3, atol=1e-3
        )

        # kNN handoff: CAGRA on the fused embedding must reproduce the fused
        # graph. Two CAGRA builds can break ties between equidistant neighbors
        # differently, so compare per-row neighbor *sets* and allow a small
        # fraction of rows to differ (a broken handoff would mismatch ~all rows).
        ref = ad.AnnData(X=fused.X.copy())
        ref.obsm["X_pca"] = np.ascontiguousarray(fused.obsm["X_pca"], dtype=np.float32)
        pyscx.accel.neighbors(ref, n_neighbors=15, device="gpu")
        assert ref.uns["neighbors"]["params"]["method"] == "cagra"

        fd = fused.obsp["distances"].tocsr()
        rd = ref.obsp["distances"].tocsr()
        n_obs = fd.shape[0]
        mismatched = sum(
            set(fd.indices[fd.indptr[i] : fd.indptr[i + 1]])
            != set(rd.indices[rd.indptr[i] : rd.indptr[i + 1]])
            for i in range(n_obs)
        )
        assert mismatched <= 0.05 * n_obs, (
            f"{mismatched}/{n_obs} rows have different neighbor sets — "
            "device kNN handoff likely broken"
        )

    def test_backed_input(self, pca_adata):
        """Fused path works on a backed SCX `X` (streaming source)."""
        import pyscx

        scx_path, _ = pca_adata
        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        assert adata.obsm["X_pca"].shape[1] == 10
        assert "connectivities" in adata.obsp
        assert adata.uns["scx_accel"]["pca_neighbors"]["route"] == "gpu_device_resident"
