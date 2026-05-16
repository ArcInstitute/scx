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
