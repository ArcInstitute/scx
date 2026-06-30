"""Tests for the obs-only bindings: Experiment.read_obs / distinct_values
(and CloudExperiment parity over file:// URLs)."""

import numpy as np
import pandas as pd
import pytest

import pyscx


# ---------------------------------------------------------------------------
# read_obs
# ---------------------------------------------------------------------------


def test_read_obs_full_shape(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata, "ro_full.scx")
    exp = pyscx.open(path)
    obs = exp.read_obs()
    assert isinstance(obs, pd.DataFrame)
    assert obs.shape[0] == exp.n_obs
    # Every obs_keys() column is present (index column excluded from obs_keys).
    for col in exp.obs_keys():
        assert col in obs.columns


def test_read_obs_columns_projection(synthetic_adata, scx_from_adata):
    path = scx_from_adata(synthetic_adata, "ro_proj.scx")
    exp = pyscx.open(path)
    obs = exp.read_obs(columns=["batch"])
    assert list(obs.columns) == ["batch"]
    assert obs.shape[0] == exp.n_obs


def test_read_obs_matches_to_anndata(query_adata, tmp_dir):
    # Multi-shard obs (120 obs, shard_size=40 → 3 shards): read_obs must
    # round-trip against the to_anndata() obs baseline.
    path = str(tmp_dir / "ro_sharded.scx")
    pyscx.from_anndata(query_adata, path, shard_size=40)
    exp = pyscx.open(path)
    assert exp.obs_keys()  # sanity

    obs = exp.read_obs()
    ref = exp.to_anndata().obs
    assert obs.shape[0] == ref.shape[0] == 120
    # cell_type values agree row-for-row across the shard boundaries.
    assert (
        obs["cell_type"].astype(str).tolist()
        == ref["cell_type"].astype(str).tolist()
    )


# ---------------------------------------------------------------------------
# distinct_values
# ---------------------------------------------------------------------------


def test_distinct_values_categorical(query_adata, tmp_dir):
    path = str(tmp_dir / "dv_cat.scx")
    pyscx.from_anndata(query_adata, path, shard_size=40)  # 3 obs shards
    exp = pyscx.open(path)

    vals, has_more = exp.distinct_values("cell_type")
    assert isinstance(vals, list) and isinstance(has_more, bool)
    assert set(vals) == {"T cell", "B cell", "NK cell"}
    assert not has_more


def test_distinct_values_sorted(query_adata, tmp_dir):
    path = str(tmp_dir / "dv_sort.scx")
    pyscx.from_anndata(query_adata, path, shard_size=40)
    exp = pyscx.open(path)

    vals, has_more = exp.distinct_values("cell_type", sort=True)
    assert vals == ["B cell", "NK cell", "T cell"]
    assert not has_more


def test_distinct_values_limit_has_more(query_adata, tmp_dir):
    path = str(tmp_dir / "dv_limit.scx")
    pyscx.from_anndata(query_adata, path, shard_size=40)
    exp = pyscx.open(path)

    vals, has_more = exp.distinct_values("cell_type", limit=2)
    assert len(vals) == 2
    assert has_more

    vals, has_more = exp.distinct_values("cell_type", limit=3)
    assert len(vals) == 3
    assert not has_more


def test_distinct_values_nulls_excluded(tmp_dir):
    import anndata

    n = 6
    obs = pd.DataFrame(
        {"grp": pd.Categorical(["a", None, "b", "a", None, "b"])},
        index=[f"c{i}" for i in range(n)],
    )
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = str(tmp_dir / "dv_null.scx")
    pyscx.from_anndata(adata, path)
    exp = pyscx.open(path)

    vals, _ = exp.distinct_values("grp", sort=True)
    assert vals == ["a", "b"]  # None excluded


def test_distinct_values_unknown_column_raises(query_adata, scx_from_adata):
    path = scx_from_adata(query_adata, "dv_badcol.scx")
    exp = pyscx.open(path)
    with pytest.raises(Exception):
        exp.distinct_values("not_a_column")


def test_distinct_values_non_string_column_raises(synthetic_adata, scx_from_adata):
    # 'highly_variable' is a var column; use an obs numeric column instead.
    # synthetic_adata obs has no numeric column, so add one.
    import anndata
    import scipy.sparse as sp

    n = 8
    obs = pd.DataFrame(
        {"n_counts": np.arange(n, dtype=np.int64)},
        index=[f"c{i}" for i in range(n)],
    )
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.ones((n, 3), dtype=np.float32)),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
    )
    path = scx_from_adata(adata, "dv_numeric.scx")
    exp = pyscx.open(path)
    with pytest.raises(ValueError):
        exp.distinct_values("n_counts")


# ---------------------------------------------------------------------------
# CloudExperiment parity (file:// URL — no network / credentials)
# ---------------------------------------------------------------------------

open_cloud = getattr(pyscx, "open_cloud", None)
needs_cloud = pytest.mark.skipif(
    open_cloud is None, reason="pyscx built without the cloud feature"
)


@needs_cloud
def test_cloud_read_obs_and_distinct(query_adata, tmp_dir):
    path = str(tmp_dir / "cloud_obs.scx")
    pyscx.from_anndata(query_adata, path, shard_size=40)
    exp = pyscx.open_cloud("file://" + path)

    obs = exp.read_obs(columns=["cell_type"])
    assert list(obs.columns) == ["cell_type"]
    assert obs.shape[0] == 120

    vals, has_more = exp.distinct_values("cell_type", sort=True)
    assert vals == ["B cell", "NK cell", "T cell"]
    assert not has_more

    vals, has_more = exp.distinct_values("cell_type", limit=2)
    assert len(vals) == 2 and has_more
