"""Tests for `pyscx.optimize(..., shard_obs=...)` — migrating a legacy
single-section obs table to the sharded `ObsMetadataShard` layout."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


def _single_section_scx(path, n_obs, shard_size):
    """Write a single-section (legacy) obs `.scx`.

    `force_legacy_metadata=True` opts out of from_anndata's automatic obs
    sharding, so obs is one `ObsMetadata` section regardless of `n_obs` —
    exactly the legacy layout `optimize --shard-obs` migrates.
    """
    anndata = pytest.importorskip("anndata")
    adata = anndata.AnnData(
        X=sp.random(n_obs, 100, density=0.3, format="csr", dtype=np.float32),
        obs=pd.DataFrame(
            {"cell_type": (["A", "B", "C"] * n_obs)[:n_obs]},
            index=[f"c{i}" for i in range(n_obs)],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(100)]),
    )
    pyscx.from_anndata(
        adata, path, shard_size=shard_size, force_legacy_metadata=True
    )


def test_optimize_shard_obs_always_shards_single_section():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.scx")
        dst = os.path.join(tmp, "out.scx")
        # n_obs (40) > shard_size (8): "always" must produce multiple obs shards.
        _single_section_scx(src, n_obs=40, shard_size=8)
        assert pyscx.open(src).obs_metadata_shard_count == 0  # legacy layout

        pyscx.optimize(src, dst, shard_obs="always")

        exp = pyscx.open(dst)
        assert exp.obs_metadata_shard_count > 0
        assert exp.n_obs == 40
        # obs round-trips row-for-row across the shard boundaries.
        obs = exp.read_obs()
        assert obs.shape[0] == 40
        assert list(obs.index[:3]) == ["c0", "c1", "c2"]


def test_optimize_shard_obs_auto_shards_above_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.scx")
        dst = os.path.join(tmp, "out.scx")
        # n_obs (40) > shard_size (8) → auto shards (the from_anndata threshold).
        _single_section_scx(src, n_obs=40, shard_size=8)

        pyscx.optimize(src, dst, shard_obs="auto")

        assert pyscx.open(dst).obs_metadata_shard_count > 0


def test_optimize_shard_obs_off_keeps_single_section():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.scx")
        dst = os.path.join(tmp, "out.scx")
        _single_section_scx(src, n_obs=40, shard_size=8)

        pyscx.optimize(src, dst, shard_obs="off")

        assert pyscx.open(dst).obs_metadata_shard_count == 0


def test_optimize_default_shard_obs_is_auto():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.scx")
        dst = os.path.join(tmp, "out.scx")
        # Default kwarg (auto) shards a large single-section obs.
        _single_section_scx(src, n_obs=40, shard_size=8)
        pyscx.optimize(src, dst)
        assert pyscx.open(dst).obs_metadata_shard_count > 0


def test_optimize_invalid_shard_obs_raises():
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.scx")
        _single_section_scx(src, n_obs=16, shard_size=8)
        with pytest.raises(ValueError):
            pyscx.optimize(src, os.path.join(tmp, "out.scx"), shard_obs="sometimes")
