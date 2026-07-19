"""`pyscx.append` / `pyscx.append_from_anndata` modality routing.

Pins the `modality=` keyword (the Python equivalent of `scx append
--modality NAME`): required on multimodal targets, rejected on
single-modality files, and resolved by name against both the target and
the source reader.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import scipy.sparse as sp


def _make_mudata(n_obs, rna_n_vars=20, adt_n_vars=6, seed=0):
    mudata = pytest.importorskip("mudata")
    anndata = pytest.importorskip("anndata")

    rng = np.random.default_rng(seed)
    rna_dense = rng.poisson(lam=0.4, size=(n_obs, rna_n_vars)).astype(np.float32)
    adt_dense = rng.poisson(lam=0.4, size=(n_obs, adt_n_vars)).astype(np.float32)
    rna_ad = anndata.AnnData(X=sp.csr_matrix(rna_dense))
    rna_ad.var_names = [f"g{i}" for i in range(rna_n_vars)]
    adt_ad = anndata.AnnData(X=sp.csr_matrix(adt_dense))
    adt_ad.var_names = [f"a{i}" for i in range(adt_n_vars)]
    mu = mudata.MuData({"rna": rna_ad, "adt": adt_ad})
    mu.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return mu


def _make_single_modality(path, n_obs=16, n_vars=20, seed=1):
    import anndata as ad
    import pyscx

    rng = np.random.default_rng(seed)
    dense = rng.poisson(lam=0.4, size=(n_obs, n_vars)).astype(np.float32)
    adata = ad.AnnData(X=sp.csr_matrix(dense))
    adata.var_names = [f"g{i}" for i in range(n_vars)]
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    pyscx.from_anndata(adata, path)
    return adata


# ---------------------------------------------------------------------------
# Single-modality files
# ---------------------------------------------------------------------------


def test_append_modality_on_single_modality_rejected(tmp_path):
    import pyscx

    target = str(tmp_path / "target.scx")
    source = str(tmp_path / "source.scx")
    _make_single_modality(target)
    _make_single_modality(source, seed=2)

    with pytest.raises(ValueError, match="single-modality"):
        pyscx.append(target, source, modality="rna")


def test_append_single_modality_no_modality_still_works(tmp_path):
    """Backward-compat: omitting `modality` on single-modality files
    keeps the legacy global-axis append."""
    import pyscx

    target = str(tmp_path / "target.scx")
    source = str(tmp_path / "source.scx")
    _make_single_modality(target, n_obs=16)
    _make_single_modality(source, n_obs=10, seed=2)

    pyscx.append(target, source)
    assert pyscx.open(target).n_obs == 26


# ---------------------------------------------------------------------------
# Multimodal files
# ---------------------------------------------------------------------------


def test_append_multimodal_requires_modality(tmp_path):
    import pyscx

    mudata = pytest.importorskip("mudata")  # noqa: F841 — gate on mudata
    target = str(tmp_path / "target.scx")
    source = str(tmp_path / "source.scx")
    pyscx.from_mudata(_make_mudata(32), target)
    pyscx.from_mudata(_make_mudata(8, seed=3), source)

    with pytest.raises(ValueError, match="multimodal"):
        pyscx.append(target, source)


def test_append_multimodal_unknown_modality(tmp_path):
    import pyscx

    pytest.importorskip("mudata")
    target = str(tmp_path / "target.scx")
    source = str(tmp_path / "source.scx")
    pyscx.from_mudata(_make_mudata(32), target)
    pyscx.from_mudata(_make_mudata(8, seed=3), source)

    with pytest.raises(ValueError, match="no modality named 'nope'"):
        pyscx.append(target, source, modality="nope")


def test_append_into_modality_rejected(tmp_path):
    """Multimodal append is deferred (review finding #3): it would leave sibling
    modalities under-covering the global obs axis → an unreadable file. It must
    be rejected before touching the target."""
    import pyscx

    pytest.importorskip("mudata")
    target = str(tmp_path / "target.scx")
    source = str(tmp_path / "source.scx")
    pyscx.from_mudata(_make_mudata(32), target)
    pyscx.from_mudata(_make_mudata(8, seed=3), source)

    with pytest.raises(ValueError, match="multimodal"):
        pyscx.append(target, source, modality="rna")

    # Target left untouched.
    out = pyscx.open(target)
    assert out.n_obs == 32


# ---------------------------------------------------------------------------
# append_from_anndata
# ---------------------------------------------------------------------------


def test_append_from_anndata_into_modality_rejected(tmp_path):
    """Multimodal append is deferred — rejected for the anndata path too."""
    import anndata as ad
    import pyscx

    pytest.importorskip("mudata")
    target = str(tmp_path / "target.scx")
    pyscx.from_mudata(_make_mudata(32), target)

    # AnnData with the rna modality's n_vars (20).
    rng = np.random.default_rng(7)
    dense = rng.poisson(lam=0.4, size=(5, 20)).astype(np.float32)
    new = ad.AnnData(X=sp.csr_matrix(dense))
    new.var_names = [f"g{i}" for i in range(20)]
    new.obs_names = [f"new_{i}" for i in range(5)]

    with pytest.raises(ValueError, match="multimodal"):
        pyscx.append_from_anndata(target, new, modality="rna")

    out = pyscx.open(target)
    assert out.n_obs == 32


def test_append_from_anndata_modality_nvars_mismatch(tmp_path):
    """pyscx validates the AnnData's n_vars against the resolved modality
    before calling into scx-ops, so an n_vars mismatch still surfaces its own
    error (this check runs ahead of the core multimodal-append reject)."""
    import anndata as ad
    import pyscx

    pytest.importorskip("mudata")
    target = str(tmp_path / "target.scx")
    pyscx.from_mudata(_make_mudata(32), target)

    # 6 vars matches adt, not rna -> mismatch against modality="rna" (20 vars).
    rng = np.random.default_rng(7)
    dense = rng.poisson(lam=0.4, size=(5, 6)).astype(np.float32)
    new = ad.AnnData(X=sp.csr_matrix(dense))
    new.var_names = [f"x{i}" for i in range(6)]

    with pytest.raises(ValueError, match="n_vars mismatch"):
        pyscx.append_from_anndata(target, new, modality="rna")
