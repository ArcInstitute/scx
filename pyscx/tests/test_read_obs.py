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


def test_read_obs_columns_preserves_index(synthetic_adata, scx_from_adata):
    # Regression: read_obs(columns=...) must keep the cell-barcode index, not
    # silently fall back to a RangeIndex (parity with unprojected read_obs()).
    path = scx_from_adata(synthetic_adata, "ro_idx.scx")
    exp = pyscx.open(path)
    full = exp.read_obs()
    proj = exp.read_obs(columns=["batch"])
    assert list(proj.index) == list(full.index)
    assert list(proj.index[:2]) == ["cell_0", "cell_1"]
    assert list(proj.columns) == ["batch"]  # index not surfaced as a column


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
# Row space: logical by default since 0.17, physical under logical=False
# ---------------------------------------------------------------------------


def _deleted_file(synthetic_adata, scx_from_adata, name, deleted=(3, 7, 50)):
    """A 100-cell file with three rows logically deleted."""
    path = scx_from_adata(synthetic_adata, name)
    pyscx.mark_deleted(path, list(deleted))
    exp = pyscx.open(path)
    assert exp.has_deletions and exp.n_obs == 97 and exp.n_obs_physical == 100, "premise"
    return path, exp


def test_read_obs_default_is_logical_and_matches_the_backed_obs(
    synthetic_adata, scx_from_adata
):
    path, exp = _deleted_file(synthetic_adata, scx_from_adata, "ro_logical.scx")

    obs = exp.read_obs()
    assert len(obs) == exp.n_obs == 97
    backed = exp.to_anndata(backed=True).obs
    pd.testing.assert_frame_equal(obs, backed)
    # The deleted barcodes are gone, the survivors keep their order.
    full_index = pyscx.open(path).read_obs(logical=False).index
    assert list(obs.index) == [b for i, b in enumerate(full_index) if i not in (3, 7, 50)]


def test_read_obs_logical_false_is_the_physical_table(synthetic_adata, scx_from_adata):
    _, exp = _deleted_file(synthetic_adata, scx_from_adata, "ro_physical.scx")

    physical = exp.read_obs(logical=False)
    assert len(physical) == exp.n_obs_physical == 100
    # The deleted rows are still there, in their physical positions.
    assert list(physical.index[:4]) == list(synthetic_adata.obs_names[:4])


def test_read_obs_columns_projection_is_logical_too(synthetic_adata, scx_from_adata):
    _, exp = _deleted_file(synthetic_adata, scx_from_adata, "ro_proj_logical.scx")

    proj = exp.read_obs(columns=["batch"])
    full = exp.read_obs()
    assert len(proj) == 97
    assert list(proj.index) == list(full.index)
    # Values agree (the projected path may dictionary-encode; compare values).
    assert proj["batch"].astype(str).tolist() == full["batch"].astype(str).tolist()

    phys_proj = exp.read_obs(columns=["batch"], logical=False)
    assert len(phys_proj) == 100


def test_read_obs_without_deletions_is_unchanged_by_logical(synthetic_adata, scx_from_adata):
    # Byte-identical contract on a file with nothing deleted: both spaces are
    # the same frame, and the logical read costs no copy.
    path = scx_from_adata(synthetic_adata, "ro_nodel.scx")
    exp = pyscx.open(path)
    assert not exp.has_deletions
    pd.testing.assert_frame_equal(exp.read_obs(), exp.read_obs(logical=False))
    pd.testing.assert_frame_equal(
        exp.read_obs(columns=["batch"]), exp.read_obs(columns=["batch"], logical=False)
    )


def test_obs_categorical_row_space_follows_read_obs(synthetic_adata, scx_from_adata):
    _, exp = _deleted_file(synthetic_adata, scx_from_adata, "ro_codes.scx")

    codes, cats = exp.obs_categorical("batch")
    assert len(codes) == exp.n_obs == 97
    decoded = [None if c < 0 else cats[c] for c in codes]
    assert decoded == exp.read_obs()["batch"].astype(str).tolist()

    codes_p, cats_p = exp.obs_categorical("batch", logical=False)
    assert len(codes_p) == 100
    assert cats_p == cats, "the row filter never drops a level"
    decoded_p = [None if c < 0 else cats_p[c] for c in codes_p]
    assert decoded_p == exp.read_obs(logical=False)["batch"].astype(str).tolist()

    many = exp.obs_categorical_many(["batch", "cell_id"])
    assert [len(c) for c, _ in many] == [97, 97]
    many_p = exp.obs_categorical_many(["batch"], logical=False)
    assert len(many_p[0][0]) == 100


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
    # Index preserved on the cloud projected path too (parity with local).
    full = exp.read_obs()
    assert list(obs.index) == list(full.index)

    vals, has_more = exp.distinct_values("cell_type", sort=True)
    assert vals == ["B cell", "NK cell", "T cell"]
    assert not has_more

    vals, has_more = exp.distinct_values("cell_type", limit=2)
    assert len(vals) == 2 and has_more


@needs_cloud
def test_cloud_read_obs_is_logical_like_the_local_handle(synthetic_adata, scx_from_adata):
    path, local = _deleted_file(synthetic_adata, scx_from_adata, "cloud_logical.scx")
    cloud = pyscx.open_cloud("file://" + path)

    assert cloud.n_obs == local.n_obs == 97
    assert cloud.n_obs_physical == 100
    assert cloud.shape == (97, 50)
    assert "97" in repr(cloud)

    obs = cloud.read_obs()
    assert len(obs) == 97
    assert list(obs.index) == list(local.read_obs().index)
    assert len(cloud.read_obs(logical=False)) == 100
    proj = cloud.read_obs(columns=["batch"])
    assert len(proj) == 97 and list(proj.columns) == ["batch"]
    assert len(cloud.read_obs(columns=["batch"], logical=False)) == 100

    codes, cats = cloud.obs_categorical("batch")
    assert len(codes) == 97
    local_codes, local_cats = local.obs_categorical("batch")
    assert [cats[c] for c in codes] == [local_cats[c] for c in local_codes]
    assert len(cloud.obs_categorical("batch", logical=False)[0]) == 100
    assert len(cloud.obs_categorical_many(["batch"])[0][0]) == 97
