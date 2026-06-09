"""Tests for in-place metadata replacement: pyscx.set_uns / modify_metadata.

These wrap scx_ops::set_uns / modify_metadata — replace SCX metadata sections
(uns / obs / var / obsm / varm) without re-encoding X.
"""

import numpy as np
import pandas as pd
import pytest

import pyscx


def test_set_uns_round_trip(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)

    new_uns = {"method": "updated", "k": 7, "labels": ["a", "b", "c"]}
    pyscx.set_uns(path, new_uns)

    uns = pyscx.open(path).to_anndata().uns
    assert uns["method"] == "updated"
    assert int(uns["k"]) == 7
    assert list(uns["labels"]) == ["a", "b", "c"]
    # Matrix untouched — file still opens and has the same shape.
    exp = pyscx.open(path)
    assert exp.n_obs == synthetic_adata.n_obs
    assert exp.n_vars == synthetic_adata.n_vars


def test_modify_metadata_obs_replace(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    new_obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "donor": ["donor_Z"] * n,
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    pyscx.modify_metadata(path, obs=new_obs)

    obs = pyscx.open(path).to_anndata().obs
    assert len(obs) == n
    assert "donor" in obs.columns
    assert obs["donor"].iloc[0] == "donor_Z"


def test_modify_metadata_obs_wrong_shape_raises(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    bad_obs = pd.DataFrame({"x": list(range(n - 1))})  # one row short
    with pytest.raises(ValueError):
        pyscx.modify_metadata(path, obs=bad_obs)


def test_modify_metadata_obs_index_rebuild(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n = synthetic_adata.n_obs

    new_obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n)],
            "batch": pd.Categorical(np.random.choice(["A", "B"], size=n)),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    # Request a predicate index over `batch` while replacing obs.
    pyscx.modify_metadata(path, obs=new_obs, index_obs=["batch"])

    # File reopens and the new obs is present.
    obs = pyscx.open(path).to_anndata().obs
    assert "batch" in obs.columns
    assert len(obs) == n


def test_modify_metadata_varm_replace(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    n_vars = synthetic_adata.n_vars

    loadings = np.random.randn(n_vars, 4).astype(np.float32)
    pyscx.modify_metadata(path, varm={"PCs": loadings})

    adata = pyscx.open(path).to_anndata()
    assert "PCs" in adata.varm
    assert adata.varm["PCs"].shape == (n_vars, 4)


def test_set_uns_then_rollback(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    original = dict(pyscx.open(path).to_anndata().uns)

    pyscx.set_uns(path, {"state": "mutated"})
    assert pyscx.open(path).to_anndata().uns["state"] == "mutated"

    pyscx.rollback(path)
    restored = pyscx.open(path).to_anndata().uns
    assert "state" not in restored
    # Original keys are back.
    assert restored.get("species") == original.get("species")


def test_modify_metadata_empty_patch_raises(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata)
    with pytest.raises(ValueError):
        pyscx.modify_metadata(path)
