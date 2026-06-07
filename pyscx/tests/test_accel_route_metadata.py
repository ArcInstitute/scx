"""Accelerator execution-route metadata on adata.uns["scx_accel"].

Every DE call records which route it took (and why it fell back) on
``adata.uns["scx_accel"][<op>]``. These tests pin the CPU-side contract — they
run without a GPU, so they assert the CPU routes and the device-fallback reason
(`user_forced_cpu`). The GPU-route assertions (`gpu_csc_v3` vs `gpu_csr_v3`)
live in the GPU parity suites which run on Chimera.
"""

from __future__ import annotations

import numpy as np  # noqa: E402
import pytest

import scipy.sparse as sp  # noqa: E402

import pyscx  # noqa: E402

from _pdex_fixtures import REFERENCE, _make_adata  # noqa: E402

GROUPBY = "target"


def _route(adata, op):
    return adata.uns["scx_accel"][op]


def _gpu_available() -> bool:
    """True iff pyscx was built with `--features gpu` AND a CUDA device is
    visible. Mirrors the canonical check in ``test_accel_pca_gpu.py``
    (`pyscx.accel.gpu_info()` returns None on CPU-only builds/hosts)."""
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


gpu_only = pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — skipping GPU route-metadata tests",
)


def _random_count_adata(n_obs: int, n_vars: int, density: float, seed: int):
    """Synthetic Poisson-like sparse CSR AnnData (mirrors the PCA GPU suite)."""
    import anndata

    rng = np.random.default_rng(seed)
    dense = rng.poisson(2.0, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > density] = 0
    return anndata.AnnData(X=sp.csr_matrix(dense))


def test_pdex_ref_cpu_dense_route():
    adata = _make_adata()  # dense numpy X
    pytest.importorskip("polars")
    pyscx.accel.pdex_ref(adata, GROUPBY, reference=REFERENCE, device="cpu")
    info = _route(adata, "pdex_ref")
    assert info["route"] == "cpu_dense"
    assert info["fallback_reason"] == "user_forced_cpu"


def test_pdex_ref_cpu_csr_route():
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)  # in-memory CSR
    pytest.importorskip("polars")
    pyscx.accel.pdex_ref(adata, GROUPBY, reference=REFERENCE, device="cpu")
    info = _route(adata, "pdex_ref")
    assert info["route"] == "cpu_csr"
    assert info["fallback_reason"] == "user_forced_cpu"
    assert info["csc_available"] is False


def test_rank_genes_groups_records_route_both_places():
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)
    pyscx.accel.rank_genes_groups(adata, GROUPBY, reference="rest", device="cpu")
    # Mirrored onto the scanpy-style dict and the unified scx_accel lookup.
    assert adata.uns["rank_genes_groups"]["scx_accel_route"] == "cpu_csr"
    info = _route(adata, "rank_genes_groups")
    assert info["route"] == "cpu_csr"
    assert info["fallback_reason"] == "user_forced_cpu"


def test_rank_genes_groups_df_records_route():
    # rank_genes_groups_df returns a polars DataFrame, so it requires polars
    # (the optional `[eval]` extra). Skip cleanly where it isn't installed
    # (e.g. the cloud/hdf5 CI job), matching the other polars-returning tests.
    pytest.importorskip("polars")
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)
    pyscx.accel.rank_genes_groups_df(adata, GROUPBY, reference="rest", device="cpu")
    info = _route(adata, "rank_genes_groups_df")
    assert info["route"] == "cpu_csr"


def test_scx_accel_dict_accumulates_ops():
    """Running two ops on the same adata leaves both route entries intact."""
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)
    pytest.importorskip("polars")
    pyscx.accel.pdex_ref(adata, GROUPBY, reference=REFERENCE, device="cpu")
    pyscx.accel.rank_genes_groups(adata, GROUPBY, reference="rest", device="cpu")
    assert "pdex_ref" in adata.uns["scx_accel"]
    assert "rank_genes_groups" in adata.uns["scx_accel"]


def test_pca_cpu_route_omits_tuning_metadata():
    """Task 2.5: on the CPU route, the math-mode / SpMM-policy / graph-replay
    fields are recorded as None (they are GPU-only knobs)."""
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)
    pyscx.accel.pca(adata, n_comps=5, device="cpu")
    info = _route(adata, "pca")
    assert info["route"] == "cpu_csr"
    assert info["math_mode"] is None
    assert info["spmm_policy"] is None
    assert info["graph_replay"] is None


def test_pca_rejects_invalid_spmm_policy():
    """Task 2.5: spmm_policy is validated on every device path."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="spmm_policy"):
        pyscx.accel.pca(adata, n_comps=5, device="cpu", spmm_policy="bogus")


@gpu_only
def test_pca_gpu_randomized_populates_tuning_metadata():
    """Task 2.5: the randomized GPU PCA route records the math-mode /
    SpMM-policy / graph-replay fields the CPU route leaves None. Defaults are
    strict_fp32 + heuristic SpMM; capture is opt-in/off so graph_replay is
    False (not None). De-risks the deferred 2.7 gate wiring."""
    adata = _random_count_adata(n_obs=400, n_vars=120, density=0.1, seed=3)
    pyscx.accel.pca(adata, n_comps=5, device="gpu", method="randomized")
    info = _route(adata, "pca")
    assert info["route"] == "gpu_csr"
    assert info["math_mode"] == "strict_fp32"
    assert info["spmm_policy"] == "default"
    assert info["graph_replay"] is False


@gpu_only
def test_pca_gpu_randomized_records_tuned_knobs():
    """Task 2.5: explicit `allow_tf32` / `spmm_policy` kwargs are reflected in
    the recorded metadata on the randomized GPU route."""
    adata = _random_count_adata(n_obs=400, n_vars=120, density=0.1, seed=4)
    pyscx.accel.pca(
        adata,
        n_comps=5,
        device="gpu",
        method="randomized",
        allow_tf32=True,
        spmm_policy="deterministic",
    )
    info = _route(adata, "pca")
    assert info["route"] == "gpu_csr"
    assert info["math_mode"] == "allow_tf32"
    assert info["spmm_policy"] == "deterministic"
