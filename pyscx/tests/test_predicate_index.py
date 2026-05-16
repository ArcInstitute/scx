"""Phase 5a: conversion-time predicate indexes from Python."""

from __future__ import annotations

import pytest

import pyscx


def test_from_anndata_index_obs_writes_predicate_index(synthetic_adata, tmp_dir):
    """`index_obs=[...]` on from_anndata produces an obs predicate
    index section on disk that the read side can decode."""
    out = tmp_dir / "indexed.scx"
    pyscx.from_anndata(synthetic_adata, str(out), index_obs=["batch"])
    # A filter on `batch` should round-trip identically to the
    # unindexed result. (Index correctness is exercised more
    # thoroughly in scx-convert::tests::convert_with_index_obs_writes_predicate_index;
    # this test just confirms the kwarg plumbing reaches the writer.)
    exp = pyscx.open(str(out))
    n_b = int((synthetic_adata.obs["batch"] == "B").sum())
    got = exp.query().filter_obs('batch == "B"').count()
    assert got == n_b


def test_from_anndata_unknown_forced_column_raises(synthetic_adata, tmp_dir):
    out = tmp_dir / "bad.scx"
    with pytest.raises(ValueError, match="nonexistent_column"):
        pyscx.from_anndata(
            synthetic_adata, str(out), index_obs=["nonexistent_column"]
        )


def test_from_anndata_unknown_preset_raises(synthetic_adata, tmp_dir):
    out = tmp_dir / "bad.scx"
    with pytest.raises(ValueError, match="not_a_real_preset"):
        pyscx.from_anndata(
            synthetic_adata, str(out), index_preset="not_a_real_preset"
        )
