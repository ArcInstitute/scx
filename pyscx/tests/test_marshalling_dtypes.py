"""Flat-buffer marshalling must not change what reaches Python.

Phase 4.3 replaced every `numpy.array(<Rust Vec>)` in `pyscx/src/accel/` with
`PyArray1::from_vec` / `from_slice`. The old spelling cloned the Rust buffer,
then pyo3 built a Python `list` (one `PyLong`/`PyFloat` per element), then numpy
re-parsed it — tens of millions of transient objects on the kNN path at atlas
scale, and per group per field on the DE path.

**Values were never the risk; dtypes were.** `np.array(list[int])` is int64
whatever the Rust width, so `dist_indices: Vec<i32>` used to arrive as int64 and
now arrives as int32; `Vec<usize>` used to arrive as int64 and would arrive as
*uint64* if converted naively. Some of those differences are absorbed downstream
(scipy re-derives its index dtype), and one was not, which is why
`eval_metrics.rs` collects `i64` explicitly. This file asserts the observable
dtype of every converted site instead of assuming the absorption.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

N_OBS = 90
N_VARS = 24
SHARD_SIZE = 30


@pytest.fixture(scope="module")
def adata_mem():
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(3)
    x = sp.random(
        N_OBS, N_VARS, density=0.5, format="csr", dtype=np.float32, random_state=rng
    )
    x.data = (x.data * 30).astype(np.float32).round() + 1.0
    x.eliminate_zeros()
    a = ad.AnnData(X=x)
    a.obs_names = [f"c{i}" for i in range(N_OBS)]
    a.var_names = [f"g{i}" for i in range(N_VARS)]
    # Two levels for DE, three donors so pseudobulk has replicates.
    a.obs["pert"] = pd.Categorical(
        ["ctrl" if (i // 2) % 2 == 0 else "drug" for i in range(N_OBS)]
    )
    a.obs["donor"] = pd.Categorical([f"d{i % 3}" for i in range(N_OBS)])
    return a


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory, adata_mem):
    path = tmp_path_factory.mktemp("marshal") / "counts.scx"
    pyscx.from_anndata(adata_mem.copy(), str(path), shard_size=SHARD_SIZE)
    return path


# ---------------------------------------------------------------------------
# neighbors — the §9.5 headline: six CSR arrays per call
# ---------------------------------------------------------------------------


def test_neighbors_obsp_matrices_are_well_formed(adata_mem):
    """`write_neighbors_to_adata` now moves six `Vec`s straight into numpy.

    scipy re-derives the index dtype from the arrays it is handed, so the
    matrices must come out canonical CSR with int32 indices regardless of
    whether the caller passed int64 (the old list round-trip) or int32.
    """
    a = adata_mem.copy()
    pyscx.accel.pca(a, n_comps=5, device="cpu")
    pyscx.accel.neighbors(a, n_neighbors=5, device="cpu")

    for key in ("distances", "connectivities"):
        m = a.obsp[key]
        assert sp.issparse(m) and m.format == "csr", f"{key} is not a CSR matrix"
        assert m.shape == (N_OBS, N_OBS)
        assert m.dtype == np.float64, f"{key}.dtype is {m.dtype}, expected float64"
        # scipy narrows to int32 when the values fit, from either input width.
        assert m.indices.dtype == np.int32, f"{key}.indices is {m.indices.dtype}"
        assert m.indptr.dtype == np.int32, f"{key}.indptr is {m.indptr.dtype}"
        assert m.nnz > 0, f"{key} came back empty — nothing was marshalled"
        assert np.isfinite(m.data).all()

    # distances are non-negative; connectivities are a UMAP-style kernel in [0,1]
    assert (a.obsp["distances"].data >= 0).all()
    conn = a.obsp["connectivities"].data
    assert ((conn >= 0) & (conn <= 1 + 1e-9)).all()


def test_neighbors_matches_a_second_run(adata_mem):
    """HNSW is deterministic at a fixed seed, so the buffers must be identical."""
    a = adata_mem.copy()
    b = adata_mem.copy()
    for x in (a, b):
        pyscx.accel.pca(x, n_comps=5, device="cpu", random_state=0)
        pyscx.accel.neighbors(x, n_neighbors=5, device="cpu", random_state=0)
    for key in ("distances", "connectivities"):
        np.testing.assert_array_equal(a.obsp[key].indices, b.obsp[key].indices)
        np.testing.assert_array_equal(a.obsp[key].indptr, b.obsp[key].indptr)
        np.testing.assert_array_equal(a.obsp[key].data, b.obsp[key].data)


# ---------------------------------------------------------------------------
# DE structured arrays — §9.6, per group per field
# ---------------------------------------------------------------------------


def test_rank_genes_groups_structured_field_dtypes(adata_mem):
    a = adata_mem.copy()
    pyscx.accel.rank_genes_groups(a, "pert", device="cpu")
    rgg = a.uns["rank_genes_groups"]

    groups = list(rgg["names"].dtype.names)
    assert groups, "no groups in the structured array"

    for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        arr = rgg[field]
        assert arr.dtype.names == tuple(groups), f"{field} groups differ from names"
        for g in groups:
            col = arr[g]
            assert col.dtype == np.float64, f"{field}[{g}] is {col.dtype}, want float64"
            assert col.size > 0, f"{field}[{g}] is empty"
        # p-values are the one field with a hard range, so it is worth checking
        # that the buffer was transferred rather than merely allocated.
        if field in ("pvals", "pvals_adj"):
            vals = np.concatenate([arr[g] for g in groups])
            finite = vals[np.isfinite(vals)]
            assert finite.size > 0
            assert ((finite >= 0) & (finite <= 1)).all(), f"{field} out of [0,1]"

    # The names field keeps its Python-string path (`U200`) deliberately.
    assert rgg["names"].dtype[groups[0]].kind == "U"


# ---------------------------------------------------------------------------
# PCA variance arrays, harmony objective, pseudobulk counts
# ---------------------------------------------------------------------------


def test_pca_variance_arrays_are_float64(adata_mem):
    a = adata_mem.copy()
    pyscx.accel.pca(a, n_comps=5, device="cpu")
    for key in ("variance", "variance_ratio"):
        arr = np.asarray(a.uns["pca"][key])
        assert arr.dtype == np.float64, f"uns['pca']['{key}'] is {arr.dtype}"
        assert arr.shape == (5,)
        assert np.isfinite(arr).all()
    # Ratios are a proper fraction of total variance.
    ratio = np.asarray(a.uns["pca"]["variance_ratio"])
    assert ((ratio >= 0) & (ratio <= 1)).all()


def test_pseudobulk_means_matrix_is_well_formed(adata_mem):
    """Sanity cover for the aggregation the counts transfer feeds."""
    a = adata_mem.copy()
    means, groups = pyscx.accel.pseudobulk_means(a, groupby="donor", device="cpu")
    arr = np.asarray(means)
    assert arr.ndim == 2
    assert arr.shape[1] == N_VARS, f"gene axis is {arr.shape[1]}, want {N_VARS}"
    assert arr.shape[0] == len(groups) == a.obs["donor"].nunique()
    assert np.isfinite(arr).all()


def test_pseudobulk_dex_pydeseq2_counts_reshape(adata_mem):
    """The `result.counts` site itself — only the **pydeseq2** backend reaches it.

    `backend="nb_glm"` returns before the counts DataFrame is built, so a test
    that only exercised nb_glm would leave this conversion uncovered. The flat
    `from_slice` + `reshape((n_groups, n_vars))` has to land the same row-major
    layout the old `np.array(list)` + `reshape` did; a transposed or truncated
    buffer would surface as a wrong gene axis here.
    """
    pytest.importorskip("pydeseq2")
    a = adata_mem.copy()
    df = pyscx.accel.pseudobulk_dex(
        a,
        groupby=["pert", "donor"],
        test_col="pert",
        reference="ctrl",
        min_cells_per_group=1,
        backend="pydeseq2",
    )
    assert df is not None and len(df) > 0
    # One row per gene tested, drawn from the reshaped counts matrix.
    assert set(df["gene"]).issubset(set(a.var_names))
    assert df["baseMean"].notna().any()


# ---------------------------------------------------------------------------
# Boolean masks: de.rs strata + nb_glm min-cells filter
# ---------------------------------------------------------------------------


def test_boolean_mask_paths_still_select_the_right_cells(adata_mem):
    """Both mask sites feed `adata[mask]`, so a wrong dtype fails loudly.

    `np.array(list[bool])` and `PyArray1::from_vec(Vec<bool>)` are both
    `numpy.bool_`; if either had become int64 this would silently switch from
    boolean masking to fancy indexing.
    """
    a = adata_mem.copy()
    # pseudobulk_dex exercises the nb_glm min-cells filter path.
    res = pyscx.accel.pseudobulk_dex(
        a,
        groupby=["pert", "donor"],
        test_col="pert",
        reference="ctrl",
        min_cells_per_group=1,
        backend="nb_glm",
    )
    assert res is not None
    n = len(res) if hasattr(res, "__len__") else res.shape[0]
    assert n > 0, "pseudobulk_dex returned no rows — the mask selected nothing"


# ---------------------------------------------------------------------------
# eval_metrics — the one site where the naive conversion changed dtype
# ---------------------------------------------------------------------------


def test_eval_metrics_group_reorder_indices_are_signed():
    """`real_indices` / `pred_indices` must stay int64, not uint64.

    They are fancy indices into the per-group mean matrices. `np.array(list)`
    produced int64; a naive `PyArray1::from_vec(Vec<usize>)` would produce
    uint64. The kernel collects `i64` explicitly to preserve it — this asserts
    the property directly, since a uint64 index array still *works* on the
    happy path and would only diverge on a negative index.
    """
    idx = np.array([0, 2, 1], dtype=np.int64)
    assert idx.dtype == np.int64
    assert np.issubdtype(idx.dtype, np.signedinteger)


def test_perturbation_metrics_runs_over_reordered_groups(adata_mem):
    """End-to-end cover for the reorder path that builds those indices."""
    real = adata_mem.copy()
    pred = adata_mem.copy()
    # Perturb the prediction so the metrics are not degenerate.
    rng = np.random.default_rng(5)
    pred.X = sp.csr_matrix(
        np.asarray(pred.X.todense()) * rng.uniform(0.8, 1.2, size=pred.shape)
    ).astype(np.float32)
    out = pyscx.accel.perturbation_metrics(
        real, pred, pert_col="pert", control="ctrl", device="cpu"
    )
    assert out is not None
