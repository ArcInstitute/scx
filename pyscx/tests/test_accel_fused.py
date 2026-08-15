"""Tests for the fused accelerators `pyscx.accel.pca_neighbors[_umap]`.

Routing after native in-VRAM GPU paths were removed:

* **CPU** (any host): outputs + route metadata match sequential
  `pca(device="cpu")` + `neighbors(device="cpu")` [+ `umap(device="cpu")`];
  routes stamp `cpu_csr` (and `cpu_dense` for the umap stage/summary).
* **In-memory `device="gpu"` with rapids-singlecell present:** routes to the
  rapids pipeline — every stage + the summary stamp `rapids_singlecell_gpu`
  (`pca_neighbors`: `rsc.pp.pca`+`rsc.pp.neighbors`; `pca_neighbors_umap`:
  +`rsc.tl.umap`).
* **`pca_neighbors` device-resident native path** (`gpu_device_resident`): kept
  for backed/lazy `X` and for in-memory `X` under `SCX_FORCE_NATIVE_GPU=1` —
  the PCA embedding stays GPU-resident and feeds straight into CAGRA. PCA on
  GPU is always randomized (the in-VRAM covariance core was removed in 3.2).
* **`pca_neighbors_umap` has no native device-resident path** (native UMAP SGD +
  device fuzzy graph were removed in 3.1): in-VRAM it runs the rapids pipeline;
  backed/lazy (or rapids-absent) inputs fall back to sequential
  pca → neighbors → umap (umap stage uses cuML → CPU).
"""

from __future__ import annotations

import os

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


def _probe_fused_route(env: dict[str, str] | None = None) -> str | None:
    """Run `pca_neighbors(device="gpu")` on a tiny in-memory AnnData and return
    the stamped `pca_neighbors` route (or None on any failure).

    Used to detect, at collection time, which GPU regime the host is in —
    behavioral detection is more robust than reimplementing the internal
    rapids / cuVS probes (which are not exposed to Python).
    """
    saved: dict[str, str | None] = {}
    if env:
        for k, v in env.items():
            saved[k] = os.environ.get(k)
            os.environ[k] = v
    try:
        import anndata as ad
        import pyscx

        rng = np.random.default_rng(0)
        a = ad.AnnData(X=sp.csr_matrix(rng.random((60, 12), dtype=np.float32)))
        pyscx.accel.pca_neighbors(a, n_comps=5, n_neighbors=5, device="gpu")
        return a.uns["scx_accel"]["pca_neighbors"]["route"]
    except Exception:
        return None
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# In-VRAM rapids pipeline available (in-memory `device="gpu"` → rapids).
_RAPIDS_FUSED = _gpu_available() and _probe_fused_route() == "rapids_singlecell_gpu"
# Native device-resident CAGRA fused path available (force-native bypasses
# rapids; backed/lazy inputs use the same native path). cuVS-gated.
_DEVICE_RESIDENT = (
    _gpu_available()
    and _probe_fused_route({"SCX_FORCE_NATIVE_GPU": "1"}) == "gpu_device_resident"
)

rapids_fused_only = pytest.mark.skipif(
    not _RAPIDS_FUSED,
    reason="rapids-singlecell in-VRAM pipeline not available — skipping rapids fused tests",
)
device_resident_only = pytest.mark.skipif(
    not _DEVICE_RESIDENT,
    reason="native device-resident CAGRA (cuVS) not available — skipping device-resident fused tests",
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
# In-memory `device="gpu"` → rapids-singlecell pipeline (Phase 1.4)
# ---------------------------------------------------------------------------


@rapids_fused_only
class TestPcaNeighborsRapids:
    def test_route_metadata_rapids(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        accel = adata.uns["scx_accel"]
        assert accel["pca"]["route"] == "rapids_singlecell_gpu"
        assert accel["neighbors"]["route"] == "rapids_singlecell_gpu"
        assert accel["pca_neighbors"]["route"] == "rapids_singlecell_gpu"

    def test_writes_all_slots(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        assert adata.obsm["X_pca"].shape == (100, 10)
        assert "connectivities" in adata.obsp
        assert adata.obsp["connectivities"].nnz > 0


# ---------------------------------------------------------------------------
# Native device-resident fused PCA → CAGRA kNN (gated on cuVS).
# Reached for backed/lazy `X`, or in-memory `X` under SCX_FORCE_NATIVE_GPU=1.
# ---------------------------------------------------------------------------


@device_resident_only
class TestPcaNeighborsDeviceResident:
    def test_route_metadata_force_native(self, synthetic_adata, monkeypatch):
        import pyscx

        monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        accel = adata.uns["scx_accel"]
        assert accel["pca"]["route"] == "gpu_device_resident"
        assert accel["neighbors"]["route"] == "gpu_device_resident"
        assert accel["pca_neighbors"]["route"] == "gpu_device_resident"
        # The device-resident kNN stage uses CAGRA, not HNSW.
        assert adata.uns["neighbors"]["params"]["method"] == "cagra"

    def test_matches_sequential_pca(self, synthetic_adata, monkeypatch):
        """Fused device-resident PCA matches a separate force-native GPU PCA run,
        and the kNN graph is valid.

        Two GPU randomized-PCA runs differ only at f32 noise level, so PCA is
        compared with tolerance. The native standalone `neighbors(device="gpu")`
        no longer produces a CAGRA graph (Phase 3.3 routes it to CPU HNSW /
        rapids), so the fused CAGRA graph cannot be reproduced standalone — we
        assert it is non-degenerate instead.
        """
        import pyscx

        monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
        fused = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(
            fused, n_comps=20, n_neighbors=15, device="gpu", random_state=0
        )

        seq = synthetic_adata.copy()
        pyscx.accel.pca(seq, n_comps=20, device="gpu", random_state=0)
        np.testing.assert_allclose(
            fused.obsm["X_pca"], seq.obsm["X_pca"], rtol=1e-3, atol=1e-3
        )

        # kNN graph sanity: symmetric connectivities with the expected fan-out.
        conn = fused.obsp["connectivities"].tocsr()
        assert conn.shape == (100, 100)
        assert conn.nnz > 0
        assert fused.uns["neighbors"]["params"]["method"] == "cagra"

    def test_backed_input(self, pca_adata):
        """Fused path works on a backed SCX `X` (streaming source); backed never
        routes to rapids, so it takes the native device-resident path."""
        import pyscx

        scx_path, _ = pca_adata
        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        assert adata.obsm["X_pca"].shape[1] == 10
        assert "connectivities" in adata.obsp
        assert adata.uns["scx_accel"]["pca_neighbors"]["route"] == "gpu_device_resident"

    def test_lazy_transform_input(self, pca_adata):
        """Fused path works on a lazy `ScxLazyTransformedDataset` `X`.

        Exercises the `ScxLazyTransformedDataset` dispatch branch in fused.rs.
        The CPU-device normalize/log1p setup keeps `X` lazy (a GPU-device
        normalize would eager-materialize), so the fused GPU entry consumes the
        lazy shard source directly — and lazy inputs take the native
        device-resident path (never rapids).
        """
        import pyscx

        scx_path, _ = pca_adata
        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4, device="cpu")
        pyscx.accel.log1p(adata, device="cpu")
        assert type(adata.X).__name__ == "ScxLazyTransformedDataset"

        pyscx.accel.pca_neighbors(adata, n_comps=10, n_neighbors=15, device="gpu")

        assert adata.obsm["X_pca"].shape[1] == 10
        assert "connectivities" in adata.obsp
        assert adata.uns["scx_accel"]["pca_neighbors"]["route"] == "gpu_device_resident"

    @pytest.mark.parametrize("method", ["randomized", "auto"])
    def test_pca_method_routes(self, synthetic_adata, monkeypatch, method):
        """GPU PCA always resolves to randomized (the in-VRAM covariance core
        was removed in Phase 3.2), so every method keeps the device-resident
        route under force-native."""
        import pyscx

        monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(
            adata, n_comps=10, n_neighbors=15, device="gpu", method=method
        )

        assert adata.obsm["X_pca"].shape == (100, 10)
        assert "connectivities" in adata.obsp
        assert adata.uns["scx_accel"]["pca_neighbors"]["route"] == "gpu_device_resident"

    def test_fused_randomized_records_tuning_metadata(self, synthetic_adata, monkeypatch):
        """The native device-resident path propagates the PCA tuning knobs onto
        the stamped route, exactly as standalone `pca()` does.

        For the randomized core, `stamp_fused_route` records the math-mode and
        SpMM-policy defaults. PCA reports `graph_replay` as `None`: SpMM-segment
        capture was removed in 3.4, so it has no capture decision to make. The
        key is still stamped — `route.rs` emits it for every op — so the contract
        is "present and None", not "absent".
        """
        import pyscx

        monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors(
            adata, n_comps=10, n_neighbors=15, device="gpu", method="randomized"
        )

        pca = adata.uns["scx_accel"]["pca"]
        assert pca["route"] == "gpu_device_resident"
        assert pca["math_mode"] == "strict_fp32"
        assert pca["spmm_policy"] == "default"
        # PCA makes no CUDA-graph capture decision, but the key is still
        # stamped, so assert presence *and* value — `.get(...) is None` would
        # also pass if the key silently stopped being emitted. `resident_csr` is
        # the residency decision PCA does make.
        assert "graph_replay" in pca
        assert pca["graph_replay"] is None
        assert pca["resident_csr"] in (True, False)

    def test_non_default_use_rep_falls_back_to_sequential(self, synthetic_adata, monkeypatch):
        """A non-default `use_rep` must NOT take the fused device-resident path.

        The fused path always runs kNN on the freshly-computed PCA embedding, so
        honoring `obsm[use_rep]` requires the sequential path. Regression guard
        for the silently-wrong-result bug: with `use_rep != "X_pca"` the route
        must not be `gpu_device_resident`, and the recorded kNN representation
        must be the one requested.
        """
        import pyscx

        monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
        adata = synthetic_adata.copy()
        # Provide the custom representation the sequential neighbors() will read.
        adata.obsm["X_custom"] = np.ascontiguousarray(
            adata.obsm["X_pca"], dtype=np.float32
        )

        pyscx.accel.pca_neighbors(
            adata, n_comps=10, n_neighbors=15, device="gpu", use_rep="X_custom"
        )

        assert adata.uns["scx_accel"]["pca_neighbors"]["route"] != "gpu_device_resident"
        assert adata.uns["neighbors"]["params"]["use_rep"] == "X_custom"
        assert "connectivities" in adata.obsp


# ---------------------------------------------------------------------------
# Fused PCA → kNN → UMAP — CPU
# ---------------------------------------------------------------------------


class TestPcaNeighborsUmapCpu:
    def test_writes_all_slots(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=10, n_neighbors=15, n_components=2, device="cpu"
        )

        assert "X_pca" in adata.obsm
        assert "distances" in adata.obsp
        assert "connectivities" in adata.obsp
        assert "neighbors" in adata.uns
        assert "X_umap" in adata.obsm
        assert adata.obsm["X_pca"].shape == (100, 10)
        assert adata.obsm["X_umap"].shape == (100, 2)
        assert np.isfinite(adata.obsm["X_umap"]).all()

    def test_matches_sequential(self, synthetic_adata):
        """Fused CPU path == sequential pca + neighbors + umap (PCA + graph)."""
        import pyscx

        fused = synthetic_adata.copy()
        pyscx.accel.pca_neighbors_umap(
            fused, n_comps=10, n_neighbors=15, n_components=2, device="cpu", random_state=0
        )

        seq = synthetic_adata.copy()
        pyscx.accel.pca(seq, n_comps=10, device="cpu", random_state=0)
        pyscx.accel.neighbors(seq, n_neighbors=15, device="cpu", random_state=0)
        pyscx.accel.umap(seq, n_components=2, device="cpu", random_state=0)

        np.testing.assert_allclose(
            fused.obsm["X_pca"], seq.obsm["X_pca"], rtol=1e-5, atol=1e-5
        )
        assert (fused.obsp["connectivities"] != seq.obsp["connectivities"]).nnz == 0
        # UMAP is the same deterministic CPU SGD at a matched seed.
        np.testing.assert_allclose(
            fused.obsm["X_umap"], seq.obsm["X_umap"], rtol=1e-4, atol=1e-4
        )

    def test_route_metadata_cpu(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=8, n_neighbors=10, n_components=2, device="cpu"
        )

        accel = adata.uns["scx_accel"]
        assert accel["pca"]["route"] == "cpu_csr"
        assert accel["neighbors"]["route"] == "cpu_csr"
        # The umap stage stamps the dense CPU route; the summary mirrors it.
        assert accel["umap"]["route"] == "cpu_dense"
        assert accel["pca_neighbors_umap"]["route"] == "cpu_dense"

    def test_invalid_args(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        with pytest.raises(ValueError):
            pyscx.accel.pca_neighbors_umap(adata, method="bogus", device="cpu")
        with pytest.raises(ValueError):
            pyscx.accel.pca_neighbors_umap(adata, qr_method="bogus", device="cpu")
        with pytest.raises(ValueError):
            pyscx.accel.pca_neighbors_umap(adata, prefer_format="csc", device="cpu")


# ---------------------------------------------------------------------------
# Fused PCA → kNN → UMAP — in-memory `device="gpu"` → rapids pipeline (Phase 1.4)
# ---------------------------------------------------------------------------


@rapids_fused_only
class TestPcaNeighborsUmapRapids:
    def test_route_metadata_rapids(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=10, n_neighbors=15, n_components=2, device="gpu"
        )

        accel = adata.uns["scx_accel"]
        assert accel["pca"]["route"] == "rapids_singlecell_gpu"
        assert accel["neighbors"]["route"] == "rapids_singlecell_gpu"
        assert accel["umap"]["route"] == "rapids_singlecell_gpu"
        assert accel["pca_neighbors_umap"]["route"] == "rapids_singlecell_gpu"

    def test_writes_valid_embedding(self, synthetic_adata):
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=20, n_neighbors=15, n_components=2, device="gpu"
        )

        assert adata.obsm["X_pca"].shape == (100, 20)
        assert "connectivities" in adata.obsp
        assert adata.obsp["connectivities"].nnz > 0
        assert adata.obsm["X_umap"].shape == (100, 2)
        assert np.isfinite(adata.obsm["X_umap"]).all()

    def test_all_zero_genes_raise_scx_hint(self):
        """F11: rapids PCA rejects all-zero genes. The fused GPU pipeline must
        translate rapids' bare ValueError into an scx-level message pointing at
        `filter_genes(min_cells=1)`, then succeed once the genes are filtered.
        """
        import anndata
        import pyscx

        rng = np.random.default_rng(0)
        n_obs, n_vars = 120, 40
        dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
        # Force the last 8 genes to zero across all cells.
        dense[:, -8:] = 0.0
        adata = anndata.AnnData(X=sp.csr_matrix(dense))
        pyscx.accel.normalize_total(adata, target_sum=1e4, device="gpu")
        pyscx.accel.log1p(adata, device="gpu")

        with pytest.raises(ValueError, match=r"filter_genes\(adata, min_cells=1\)"):
            pyscx.accel.pca_neighbors_umap(
                adata, n_comps=10, n_neighbors=15, n_components=2, device="gpu"
            )

        # The translated error must leave X host-resident (not stranded as a cupy
        # matrix), so the recovery path below works — the run/run_fused error path
        # restores the host AnnData even on failure.
        assert sp.issparse(adata.X) and not str(type(adata.X)).startswith(
            "<class 'cupy"
        )

        # After filtering the all-zero genes, the fused pipeline runs cleanly.
        pyscx.accel.filter_genes(adata, min_cells=1)
        assert adata.n_vars == n_vars - 8
        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=10, n_neighbors=15, n_components=2, device="gpu"
        )
        assert adata.obsm["X_umap"].shape == (n_obs, 2)

    def test_pca_all_zero_genes_raise_scx_hint(self):
        """F11: the standalone `pca(device="gpu")` rapids path shares the same
        translation — an all-zero-gene matrix raises the scx hint, not the raw
        rapids ValueError.
        """
        import anndata
        import pyscx

        rng = np.random.default_rng(1)
        dense = rng.integers(0, 50, size=(80, 30)).astype(np.float32)
        dense[:, -5:] = 0.0
        adata = anndata.AnnData(X=sp.csr_matrix(dense))
        pyscx.accel.normalize_total(adata, target_sum=1e4, device="gpu")
        pyscx.accel.log1p(adata, device="gpu")

        with pytest.raises(ValueError, match=r"filter_genes\(adata, min_cells=1\)"):
            pyscx.accel.pca(adata, n_comps=10, device="gpu")

    def test_large_n_neighbors_completes(self):
        """Large `n_neighbors` must complete without error and produce a valid
        embedding. The old `FUZZY_MAX_K` per-row scratch cap (and the fused
        device-resident UMAP path it guarded) was removed in Phase 3 — in-VRAM
        the call now runs the full rapids pipeline, which handles large k. The
        route is never `gpu_device_resident` (no native fused UMAP path exists).
        """
        import anndata
        import pyscx

        rng = np.random.default_rng(0)
        n_obs, n_vars = 400, 50
        dense = rng.integers(0, 200, size=(n_obs, n_vars)).astype(np.float32)
        dense[rng.random((n_obs, n_vars)) > 0.3] = 0
        adata = anndata.AnnData(X=sp.csr_matrix(dense))

        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=10, n_neighbors=300, n_components=2, device="gpu"
        )

        assert (
            adata.uns["scx_accel"]["pca_neighbors_umap"]["route"]
            != "gpu_device_resident"
        )
        assert adata.obsm["X_umap"].shape == (n_obs, 2)
        assert np.isfinite(adata.obsm["X_umap"]).all()


# ---------------------------------------------------------------------------
# Fused PCA → kNN → UMAP — backed/lazy `X`: sequential fallback.
# Native device-resident UMAP was removed in 3.1, and backed inputs never route
# to rapids, so the summary must NOT be `gpu_device_resident`.
# ---------------------------------------------------------------------------


@device_resident_only
class TestPcaNeighborsUmapSequentialFallback:
    def test_backed_input_falls_back(self, pca_adata):
        """Fused PCA→kNN→UMAP on a backed SCX `X` falls back to the sequential
        path (no native device-resident UMAP), but still produces a valid
        embedding."""
        import pyscx

        scx_path, _ = pca_adata
        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.pca_neighbors_umap(
            adata, n_comps=10, n_neighbors=15, n_components=2, device="gpu"
        )

        assert adata.obsm["X_pca"].shape[1] == 10
        assert adata.obsm["X_umap"].shape == (adata.n_obs, 2)
        assert np.isfinite(adata.obsm["X_umap"]).all()
        assert (
            adata.uns["scx_accel"]["pca_neighbors_umap"]["route"]
            != "gpu_device_resident"
        )


def test_fused_keeps_the_completed_pca_stage_when_knn_fails(pca_adata):
    """A fused op stamps one key per stage, and the keys are independent.

    `pca_neighbors` runs PCA, then kNN. If kNN raises, PCA still *ran* — its
    stamp is correct and must survive — while `neighbors` and the `pca_neighbors`
    summary must not appear. That is the per-stage granularity the
    present-iff-completed contract buys.
    """
    import pyscx

    path, _ = pca_adata
    adata = pyscx.open(str(path)).to_anndata(backed=True)

    with pytest.raises(RuntimeError, match="exceeds n_obs"):
        pyscx.accel.pca_neighbors(
            adata, n_comps=5, n_neighbors=10_000, device="cpu"
        )

    stamps = adata.uns.get("scx_accel", {})
    assert "pca" in stamps, "the PCA stage completed; its stamp must survive"
    assert "neighbors" not in stamps
    assert "pca_neighbors" not in stamps
