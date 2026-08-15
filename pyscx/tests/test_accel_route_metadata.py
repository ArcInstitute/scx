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
    pyscx.accel.pdex_ref(adata, GROUPBY, reference=REFERENCE, device="cpu")
    info = _route(adata, "pdex_ref")
    assert info["route"] == "cpu_dense"
    assert info["fallback_reason"] == "user_forced_cpu"


def test_pdex_ref_cpu_csr_route():
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)  # in-memory CSR
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
    # No polars guard: `rank_genes_groups_df` defaults to pandas (F6), which is a
    # hard dependency via anndata, so this runs everywhere pyscx imports.
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)
    pyscx.accel.rank_genes_groups_df(adata, GROUPBY, reference="rest", device="cpu")
    info = _route(adata, "rank_genes_groups_df")
    assert info["route"] == "cpu_csr"


def test_scx_accel_dict_accumulates_ops():
    """Running two ops on the same adata leaves both route entries intact."""
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)
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


def test_pca_rejects_invalid_spmm_policy():
    """Task 2.5: spmm_policy is validated on every device path."""
    adata = _make_adata()
    with pytest.raises(ValueError, match="spmm_policy"):
        pyscx.accel.pca(adata, n_comps=5, device="cpu", spmm_policy="bogus")


@gpu_only
def test_pca_gpu_randomized_populates_tuning_metadata(monkeypatch):
    """Task 2.5: the randomized GPU PCA route records the math-mode /
    SpMM-policy fields the CPU route leaves None. Defaults are strict_fp32 +
    heuristic SpMM. De-risks the deferred 2.7 gate wiring.

    PCA stamps no `graph_replay`: SpMM-segment capture was removed, so there is
    no capture decision to report. `harmony_integrate` is the op that does.

    `SCX_FORCE_NATIVE_GPU=1` is required, not cosmetic: `math_mode` /
    `spmm_policy` are stamped only by the **native** randomized
    route, and an in-memory `X` with rapids-singlecell installed routes to
    `rapids_singlecell_gpu` instead. Without the pin this asserted a route that
    cannot occur on a rapids host — observed failing on an H100 with
    `'rapids_singlecell_gpu' == 'gpu_csr'`. Same pin as `test_accel_pca_gpu.py`.
    """
    monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
    adata = _random_count_adata(n_obs=400, n_vars=120, density=0.1, seed=3)
    pyscx.accel.pca(adata, n_comps=5, device="gpu", method="randomized")
    info = _route(adata, "pca")
    assert info["route"] == "gpu_csr"
    assert info["math_mode"] == "strict_fp32"
    assert info["spmm_policy"] == "default"
    assert info.get("graph_replay") is None


@gpu_only
def test_pca_gpu_randomized_records_tuned_knobs(monkeypatch):
    """Task 2.5: explicit `allow_tf32` / `spmm_policy` kwargs are reflected in
    the recorded metadata on the randomized GPU route.

    Pinned to the native route for the same reason as the test above.

    Scope note: this asserts the metadata *echo* only. That the streaming arm
    of that route actually applies the requested SpMM algorithm — the §8.11
    defect — is covered by `test_gpu_pca_resident.py`, because this fixture is
    ~38 KB and always takes the device-resident loop, which honoured the policy
    all along.
    """
    monkeypatch.setenv("SCX_FORCE_NATIVE_GPU", "1")
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


# ---------------------------------------------------------------------------
# The stamp means "this op completed"
# ---------------------------------------------------------------------------
#
# Thirteen ops stamp their route *before* dispatch, so the route is known while
# a long backed run is still in flight. The cost, until `RouteStamp`, was that a
# raise left the stamp behind and `uns["scx_accel"]` claimed an op had run when
# it hadn't. Found by dogfooding v0.11.8: `normalize_total` raised on a backed
# view, `pca` then happily ran on the un-normalized counts, and the metadata
# asserted the normalize had happened on `cpu_csr`.
#
# The contract these pin: `uns["scx_accel"][op]` is present **iff** the op
# completed — and a failing re-run *restores* the entry an earlier successful
# run left, rather than deleting it.
#
# Each trigger below raises strictly *after* its op's stamp. That is the whole
# point, so it is worth being explicit about where: `calculate_qc_metrics`
# rejects `prefer_format="csc"` on a scipy X only once it has seen `adata.X`;
# `pca` and `neighbors` fail inside the scx-accel kernels. (A pre-stamp raise —
# `neighbors(use_rep="X_missing")`, say — would pass these tests vacuously.)


def _scx_accel(adata):
    return adata.uns.get("scx_accel", {})


def test_failing_op_does_not_create_the_container():
    adata = _random_count_adata(n_obs=60, n_vars=25, density=0.4, seed=11)
    assert "scx_accel" not in adata.uns

    with pytest.raises(RuntimeError, match="requires adata.X"):
        pyscx.accel.calculate_qc_metrics(adata, prefer_format="csc")

    assert "scx_accel" not in adata.uns, (
        "a failing first accel op must leave uns exactly as it found it"
    )


def test_failing_op_restores_the_previous_stamp():
    """Restore, don't delete: a bad re-run must not erase a good stamp.

    600 genes because the *successful* call delegates to scanpy on a scipy X,
    and scanpy's default `percent_top=[50, 100, 200, 500]` requires at least
    500 columns.
    """
    adata = _random_count_adata(n_obs=60, n_vars=600, density=0.1, seed=12)
    pyscx.accel.calculate_qc_metrics(adata)
    good = dict(_route(adata, "calculate_qc_metrics"))

    with pytest.raises(RuntimeError, match="requires adata.X"):
        pyscx.accel.calculate_qc_metrics(adata, prefer_format="csc")

    assert dict(_route(adata, "calculate_qc_metrics")) == good


def test_failing_hvg_leaves_no_stamp():
    adata = _random_count_adata(n_obs=60, n_vars=25, density=0.4, seed=13)
    with pytest.raises(KeyError):
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=5, flavor="seurat_v3", batch_key="does_not_exist"
        )
    assert "highly_variable_genes" not in _scx_accel(adata)


def test_failing_pca_leaves_no_stamp():
    adata = _random_count_adata(n_obs=100, n_vars=50, density=0.4, seed=14)
    with pytest.raises(RuntimeError, match="exceeds matrix rank bound"):
        pyscx.accel.pca(adata, n_comps=200, device="cpu")
    assert "pca" not in _scx_accel(adata)


def test_failing_pca_restores_the_earlier_pca_stamp():
    """Covers `stamp_pca_route`, which stamps pre-dispatch and re-stamps after."""
    adata = _random_count_adata(n_obs=100, n_vars=50, density=0.4, seed=15)
    pyscx.accel.pca(adata, n_comps=5, device="cpu")
    good = dict(_route(adata, "pca"))

    with pytest.raises(RuntimeError, match="exceeds matrix rank bound"):
        pyscx.accel.pca(adata, n_comps=200, device="cpu")

    assert dict(_route(adata, "pca")) == good


def test_failing_neighbors_leaves_no_stamp():
    adata = _random_count_adata(n_obs=100, n_vars=50, density=0.4, seed=16)
    pyscx.accel.pca(adata, n_comps=5, device="cpu")
    with pytest.raises(RuntimeError, match="exceeds n_obs"):
        pyscx.accel.neighbors(adata, n_neighbors=1000, use_rep="X_pca", device="cpu")
    assert "neighbors" not in _scx_accel(adata)


def test_a_neighbours_failure_does_not_disturb_other_ops_stamps():
    adata = _random_count_adata(n_obs=100, n_vars=50, density=0.4, seed=17)
    pyscx.accel.pca(adata, n_comps=5, device="cpu")

    with pytest.raises(RuntimeError, match="exceeds n_obs"):
        pyscx.accel.neighbors(adata, n_neighbors=1000, use_rep="X_pca", device="cpu")

    assert "pca" in _scx_accel(adata), "an unrelated op's stamp must survive"
    assert "neighbors" not in _scx_accel(adata)
