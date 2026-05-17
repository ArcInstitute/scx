"""Phase 5b: conversion-time detection bitmaps from Python."""

from __future__ import annotations

import numpy as np
import pytest

import pyscx


def test_detection_counts_matches_numpy(synthetic_adata, tmp_dir):
    """`bitmap="always"` writes per-shard bitmap sidecars, and
    `PyExperiment.detection_counts()` returns the same per-gene
    detection counts as a numpy `(X > 0).sum(axis=0)` reference."""
    out = tmp_dir / "with_bitmap.scx"
    pyscx.from_anndata(synthetic_adata, str(out), bitmap="always")
    exp = pyscx.open(str(out))
    got = np.asarray(exp.detection_counts())
    x = synthetic_adata.X
    expected = np.asarray((x != 0).sum(axis=0)).ravel()
    assert got.dtype == np.int64
    assert got.shape == expected.shape
    assert np.array_equal(got, expected.astype(np.int64))


def test_cells_expressing_gene_by_index_and_name(synthetic_adata, tmp_dir):
    """`cells_expressing(int)` and `cells_expressing("gene_name")`
    return the same set of cell indices."""
    out = tmp_dir / "with_bitmap.scx"
    pyscx.from_anndata(synthetic_adata, str(out), bitmap="always")
    exp = pyscx.open(str(out))
    # Pick the first gene that has any cells expressing it.
    x = synthetic_adata.X.toarray() if hasattr(synthetic_adata.X, "toarray") else synthetic_adata.X
    counts = (x != 0).sum(axis=0)
    g = int(np.argmax(counts))
    by_idx = np.asarray(exp.cells_expressing(g))
    name = str(synthetic_adata.var.index[g])
    by_name = np.asarray(exp.cells_expressing(name))
    assert np.array_equal(np.sort(by_idx), np.sort(by_name))
    # And against numpy.
    expected = np.flatnonzero(x[:, g] != 0).astype(np.uint32)
    assert np.array_equal(np.sort(by_idx), np.sort(expected))


def test_bitmap_off_default_no_section(synthetic_adata, tmp_dir):
    """Default `bitmap` is off; `detection_counts` still works via
    the CSR fallback path."""
    out = tmp_dir / "no_bitmap.scx"
    pyscx.from_anndata(synthetic_adata, str(out))
    exp = pyscx.open(str(out))
    got = np.asarray(exp.detection_counts())
    x = synthetic_adata.X
    expected = np.asarray((x != 0).sum(axis=0)).ravel().astype(np.int64)
    assert np.array_equal(got, expected)


def test_cells_expressing_rejects_unknown_gene(synthetic_adata, tmp_dir):
    out = tmp_dir / "no_bitmap.scx"
    pyscx.from_anndata(synthetic_adata, str(out))
    exp = pyscx.open(str(out))
    with pytest.raises(KeyError, match="not found"):
        exp.cells_expressing("nonexistent_gene_xyz")


def _index_only_var_adata(index_name):
    """Build a small AnnData where gene names live only in `var.index`.

    The synthetic fixture in conftest happens to duplicate the index
    into a `gene_id` column, which masks the pandas-metadata resolver
    bug. This builder produces a var DataFrame whose only string column
    is the index (named or unnamed), so the resolver must consult the
    Arrow `pandas` schema metadata (or fall back to `__index_level_0__`)
    to find the gene name.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    n_obs, n_vars = 12, 5
    rng = np.random.default_rng(7)
    dense = rng.integers(0, 10, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.4] = 0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    gene_names = [f"GENE_{i}" for i in range(n_vars)]
    var = pd.DataFrame(
        index=pd.Index(gene_names, name=index_name),
    )
    return anndata.AnnData(X=x, obs=obs, var=var)


def test_cells_expressing_resolves_named_index(tmp_dir):
    """`var.index.name = "gene_symbols"`, no other string column: the
    resolver must consult the Arrow `pandas` metadata's
    `index_columns` to find the gene."""
    adata = _index_only_var_adata(index_name="gene_symbols")
    out = tmp_dir / "named_index.scx"
    pyscx.from_anndata(adata, str(out), bitmap="always")
    exp = pyscx.open(str(out))
    target = "GENE_2"
    by_name = np.asarray(exp.cells_expressing(target))
    g = list(adata.var.index).index(target)
    by_idx = np.asarray(exp.cells_expressing(g))
    expected = np.flatnonzero(adata.X.toarray()[:, g] != 0).astype(np.uint32)
    assert np.array_equal(np.sort(by_name), np.sort(by_idx))
    assert np.array_equal(np.sort(by_name), np.sort(expected))


def test_cells_expressing_resolves_unnamed_index(tmp_dir):
    """`var.index` has no `name`, no other string column: the resolver
    must fall back to the canonical pyarrow `__index_level_0__` column."""
    adata = _index_only_var_adata(index_name=None)
    out = tmp_dir / "unnamed_index.scx"
    pyscx.from_anndata(adata, str(out), bitmap="always")
    exp = pyscx.open(str(out))
    target = "GENE_1"
    by_name = np.asarray(exp.cells_expressing(target))
    g = list(adata.var.index).index(target)
    expected = np.flatnonzero(adata.X.toarray()[:, g] != 0).astype(np.uint32)
    assert np.array_equal(np.sort(by_name), np.sort(expected))
