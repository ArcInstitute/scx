"""CSC dispatch on `ScxLazyTransformedDataset`.

Verifies the column-local capability gate on lazy datasets:

  - `Log1p` is column-local → `prefer_format="csc"` succeeds and
    matches the CSR equivalent.
  - `NormalizeTotal` is row-local → `prefer_format="csc"` raises
    `RuntimeError` (no silent fallback).
  - `prefer_format="csr"` works on both transform chains.

Sibling to `test_csc_dispatch.py`. The point of this file is the
*lazy* path specifically — `test_csc_dispatch.py` exercises the
backed path with the same assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(11)
    mat = sp.random(40, 16, density=0.4, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 50).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(40)]
    adata.obs["group"] = ["A"] * 20 + ["B"] * 20
    adata.var["gene_id"] = [f"g{i}" for i in range(16)]
    return adata


def _open_with_csc(path, adata):
    import pyscx

    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=4)
    return pyscx.open(str(path)).to_anndata(backed=True)


# ---------------------------------------------------------------------------
# Log1p path: column-local; CSC dispatch must succeed and match CSR.
# ---------------------------------------------------------------------------


def test_lazy_log1p_csc_col_sums_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)

    csc_sums = pyscx.accel.col_sums(a_csc.X, prefer_format="csc")
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.sum(axis=0).astype(np.float64)
    np.testing.assert_allclose(csc_sums, expected, atol=1e-5)


def test_lazy_log1p_csc_col_var_matches_materialized(small_adata, tmp_path):
    """`col_var` on a lazy log1p chain via CSC matches the
    materialized log1p variance."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)

    csc_var = pyscx.accel.col_var(a_csc.X, prefer_format="csc")
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.var(axis=0).astype(np.float64)
    np.testing.assert_allclose(csc_var, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# NormalizeTotal path: row-local; CSC dispatch must raise without
# silent fallback.
# ---------------------------------------------------------------------------


def test_lazy_normalize_total_csc_raises(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.normalize_total(a_csc)

    with pytest.raises(RuntimeError, match="CSC|column-local"):
        pyscx.accel.col_sums(a_csc.X, prefer_format="csc")


def test_lazy_normalize_total_csr_works(small_adata, tmp_path):
    """`prefer_format="csr"` is unaffected by NormalizeTotal."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    # Pin target_sum explicitly: the streaming-vs-materialized equality below is
    # magnitude-sensitive at atol=1e-5, and the default is now None→median (a
    # small target on integer counts), which tips two genes over the tolerance.
    pyscx.accel.normalize_total(a_csc, target_sum=1e4)

    # CSR path on a lazy NormalizeTotal chain currently goes through
    # the dunder methods on the lazy wrapper (the col_sums pyfunction
    # only supports backed datasets on its CSR path — documented
    # limitation in pyscx/src/accel/col_aggs.rs). Use the dunder
    # interface for the CSR check.
    csr_sums = np.asarray(a_csc.X.sum(axis=0)).ravel()
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.sum(axis=0).astype(np.float64)
    np.testing.assert_allclose(csr_sums, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Compound chains: log1p preserves CSC capability; appending
# normalize_total breaks it.
# ---------------------------------------------------------------------------


def test_lazy_log1p_then_normalize_total_csc_raises(small_adata, tmp_path):
    """A column-local op followed by a row-local op disqualifies the
    chain — the gate must reject CSC dispatch."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)
    # CSC dispatch should still work after log1p alone.
    _ = pyscx.accel.col_sums(a_csc.X, prefer_format="csc")
    pyscx.accel.normalize_total(a_csc)
    with pytest.raises(RuntimeError, match="CSC|column-local"):
        pyscx.accel.col_sums(a_csc.X, prefer_format="csc")


# ---------------------------------------------------------------------------
# Column projection × CSC: HVG must accumulate against the projected
# axis. Regression for the bug where LazyShardSource::read_csc_shard
# applied global col_projection IDs against shard-local column space.
# ---------------------------------------------------------------------------


def test_lazy_csc_with_col_projection_hvg_matches_csr(small_adata, tmp_path):
    """HVG via prefer_format='csc' on a column-projected lazy dataset
    must match the CSR equivalent.

    The CSC HVG kernels (`streaming_mean_var_csc`,
    `streaming_clip_square_sum_csc`) iterate per CSC shard via
    `read_csc_shard`. Before the fix, `LazyShardSource::read_csc_shard`
    fed `project_csc` global file column IDs against a shard-local
    column space — out-of-shard projection entries were silently dropped
    and survivors mapped to wrong shard-local positions, corrupting the
    means/variances written to `adata.var`.

    The fixture has 16 genes with `csc_cols_per_shard=4` → 4 CSC shards.
    `filter_genes(min_cells=3)` yields a projection spanning multiple
    shards. We compare HVG outputs against the CSR path on the same
    chain.
    """
    import pyscx

    a_csr = _open_with_csc(tmp_path / "with_csc_csr.scx", small_adata)
    a_csc = _open_with_csc(tmp_path / "with_csc_csc.scx", small_adata)

    pyscx.accel.log1p(a_csr)
    pyscx.accel.log1p(a_csc)

    pyscx.accel.filter_genes(a_csr, min_cells=3)
    pyscx.accel.filter_genes(a_csc, min_cells=3)

    n_proj = a_csc.X.shape[1]
    # The bug only fires when the projection spans at least two CSC
    # shards. csc_cols_per_shard=4, so we need projected indices that
    # fall in different shards. With min_cells=3 on the seeded fixture,
    # the projection covers multiple shards by construction; assert as
    # a guard against fixture drift.
    assert n_proj >= 4, (
        f"projection too narrow ({n_proj}); test won't exercise multi-shard "
        f"col_projection. Adjust min_cells or fixture seed."
    )

    n_top = max(2, min(5, n_proj - 1))
    pyscx.accel.highly_variable_genes(
        a_csr,
        n_top_genes=n_top,
        flavor="seurat_v3",
        prefer_format="csr",
    )
    pyscx.accel.highly_variable_genes(
        a_csc,
        n_top_genes=n_top,
        flavor="seurat_v3",
        prefer_format="csc",
    )

    csr_means = np.asarray(a_csr.var["means"], dtype=np.float64)
    csc_means = np.asarray(a_csc.var["means"], dtype=np.float64)
    csr_var = np.asarray(a_csr.var["variances"], dtype=np.float64)
    csc_var = np.asarray(a_csc.var["variances"], dtype=np.float64)

    np.testing.assert_allclose(
        csc_means, csr_means, atol=1e-5,
        err_msg="CSC means diverge from CSR after col_projection — "
                "LazyShardSource::read_csc_shard projection remap regression",
    )
    np.testing.assert_allclose(
        csc_var, csr_var, atol=1e-5,
        err_msg="CSC variances diverge from CSR after col_projection — "
                "LazyShardSource::read_csc_shard projection remap regression",
    )
