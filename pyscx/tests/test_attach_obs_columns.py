"""Tests for `pyscx.attach_obs_columns`.

The DataFrame twin of `obs_import`, on the same `attach_external_obs` seam:
key-joined by default, with an explicit `positional=True` mode for frames
computed in-process from the file's own `read_obs()` output — the one shape a
key join structurally cannot serve (a merged atlas whose obs index is fully
duplicated with no unique column).

The join tests deliberately scramble the frame's row order relative to the
target, exactly as `test_obs_import.py` does: a positional assumption behind a
key join would land every value on the wrong cell while still producing a
correctly-shaped column.
"""

import json

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


def _fixture(tmp_path, name="t.scx", obs=None, index_obs=None):
    """A small SCX file. `obs` defaults to four unique barcodes."""
    if obs is None:
        bc = ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]
        obs = pd.DataFrame(index=bc)
    n = len(obs)
    X = sparse.csr_matrix(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    path = tmp_path / name
    pyscx.from_anndata(
        anndata.AnnData(X=X, obs=obs, var=var), str(path), index_obs=index_obs
    )
    return path


def _obs(path):
    return pyscx.open(str(path)).read_obs()


# ---------------------------------------------------------------------------
# Key join
# ---------------------------------------------------------------------------


def test_joins_by_df_index_by_default(tmp_path):
    scx = _fixture(tmp_path)
    # Out of order, covering three of four cells: a positional attach would
    # land every score on the wrong cell.
    df = pd.DataFrame(
        {"score": [0.30, 0.10, 0.90]}, index=["AAAT-1", "AAAC-1", "AAAG-1"]
    )

    r = pyscx.attach_obs_columns(str(scx), df)
    assert r["n_obs"] == 4
    assert r["n_matched"] == 3
    assert r["obs_key_column"] == "obs_names"
    assert r["obs_columns_added"] == ["score"]

    obs = _obs(scx)
    by_cell = dict(zip(obs.index, obs["score"]))
    assert by_cell["AAAT-1"] == pytest.approx(0.30)
    assert by_cell["AAAC-1"] == pytest.approx(0.10)
    assert by_cell["AAAG-1"] == pytest.approx(0.90)
    assert pd.isna(by_cell["AAAA-1"]), "an uncovered row is null, never 0.0"


def test_named_key_column_is_consumed_not_reimported(tmp_path):
    # `key=` names the same column on BOTH sides, exactly as rscx's
    # `scx_attach_obs(key_columns=)` — so the target carries `bc` too.
    obs = pd.DataFrame(
        {"bc": ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]},
        index=[f"c{i}" for i in range(4)],
    )
    scx = _fixture(tmp_path, obs=obs)
    df = pd.DataFrame(
        {
            "bc": ["AAAG-1", "AAAC-1", "AAAT-1", "AAAA-1"],
            "grp": ["a", "b", "c", "d"],
        }
    )
    r = pyscx.attach_obs_columns(str(scx), df, key=["bc"])
    assert r["n_matched"] == 4
    obs = _obs(scx)
    assert list(obs.columns) == ["bc", "grp"], "the key column is not re-imported"
    assert dict(zip(obs["bc"], obs["grp"]))["AAAG-1"] == "a"


def test_composite_key(tmp_path):
    obs = pd.DataFrame(
        {"sample_id": ["s1", "s1", "s2", "s2"], "barcode": ["A", "B", "A", "B"]},
        index=["c0", "c1", "c2", "c3"],
    )
    scx = _fixture(tmp_path, obs=obs)
    df = pd.DataFrame(
        {
            "sample_id": ["s2", "s1", "s2", "s1"],
            "barcode": ["A", "A", "B", "B"],
            "score": [2.0, 0.0, 3.0, 1.0],
        }
    )
    r = pyscx.attach_obs_columns(str(scx), df, key=["sample_id", "barcode"])
    assert r["n_matched"] == 4
    got = _obs(scx)["score"].tolist()
    assert got == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_key_mode_on_duplicated_index_names_the_fix(tmp_path):
    scx = _fixture(tmp_path, obs=pd.DataFrame(index=["dup"] * 4))
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]}, index=["dup"] * 4)
    with pytest.raises(ValueError, match="duplicates"):
        pyscx.attach_obs_columns(str(scx), df)


# ---------------------------------------------------------------------------
# Positional
# ---------------------------------------------------------------------------


def test_positional_works_where_the_key_join_cannot(tmp_path):
    scx = _fixture(tmp_path, obs=pd.DataFrame(index=["dup"] * 4))
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})

    r = pyscx.attach_obs_columns(str(scx), df, positional=True)
    assert r["n_matched"] == 4
    assert r["obs_key_column"] == "<positional>"
    got = _obs(scx)["score"].tolist()
    assert got == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_positional_and_key_are_mutually_exclusive(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="mutually exclusive"):
        pyscx.attach_obs_columns(str(scx), df, key=["bc"], positional=True)


def test_positional_row_count_mismatch_names_physical_space(tmp_path):
    scx = _fixture(tmp_path)  # n_obs = 4
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="physical"):
        pyscx.attach_obs_columns(str(scx), df, positional=True)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_a_dict_is_rejected_naming_the_function(tmp_path):
    scx = _fixture(tmp_path)
    with pytest.raises(TypeError, match="attach_obs_columns"):
        pyscx.attach_obs_columns(str(scx), {"score": [1, 2, 3, 4]})


def test_uns_without_uns_key_is_rejected(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="uns_key"):
        pyscx.attach_obs_columns(str(scx), df, positional=True, uns={"m": 1})
    with pytest.raises(ValueError, match="uns_key"):
        pyscx.attach_obs_columns(str(scx), df, positional=True, uns_key="m")


# ---------------------------------------------------------------------------
# uns merge, rollback, dry run
# ---------------------------------------------------------------------------


def test_uns_merges_one_key_and_one_rollback_undoes_both(tmp_path):
    scx = _fixture(tmp_path)
    pyscx.set_uns(str(scx), {"existing": "kept"})
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})

    pyscx.attach_obs_columns(
        str(scx), df, positional=True, uns={"method": "majority"}, uns_key="consensus"
    )
    exp = pyscx.open(str(scx))
    uns = exp.read_uns()
    assert uns["consensus"]["method"] == "majority"
    assert uns["existing"] == "kept", "other uns keys survive the one-key merge"
    assert "score" in exp.read_obs().columns

    pyscx.rollback(str(scx))
    exp = pyscx.open(str(scx))
    assert "score" not in exp.read_obs().columns
    uns = exp.read_uns() or {}
    assert "consensus" not in uns, "one rollback undoes obs and uns together"


def test_dry_run_writes_nothing_and_reports_the_join(tmp_path):
    scx = _fixture(tmp_path)
    before = scx.read_bytes()
    df = pd.DataFrame({"score": [0.30, 0.10]}, index=["AAAT-1", "AAAC-1"])

    r = pyscx.attach_obs_columns(str(scx), df, dry_run=True)
    assert r["dry_run"] is True
    assert r["n_matched"] == 2
    assert "key_diagnosis" in r, "a key-mode dry run carries the diagnosis"
    assert before == scx.read_bytes()
    assert "score" not in _obs(scx).columns


# ---------------------------------------------------------------------------
# Predicate index and provenance
# ---------------------------------------------------------------------------


def test_positional_pure_add_keeps_the_predicate_index(tmp_path):
    obs = pd.DataFrame(
        {"grp": pd.Categorical(["A", "A", "B", "B"])},
        index=[f"c{i}" for i in range(4)],
    )
    scx = _fixture(tmp_path, obs=obs, index_obs=["grp"])

    def sections():
        return [name for name, _ in pyscx.open(str(scx)).validate()]

    assert "obs_predicate_index" in sections(), "fixture must start indexed"

    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    r = pyscx.attach_obs_columns(str(scx), df, positional=True)
    assert r["obs_index_dropped"] is False

    assert "obs_predicate_index" in sections()
    q = pyscx.open(str(scx)).query()
    q.filter_obs("grp == 'A'")
    assert q.collect().n_obs == 2, "pushdown still answers after a pure add"


def test_provenance_records_the_attach(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    pyscx.attach_obs_columns(str(scx), df, positional=True)

    entry = pyscx.open(str(scx)).provenance()[-1]
    assert entry["action"] == "attach_obs_columns"
    params = json.loads(entry["params_json"])
    assert params["obs_key_column"] == "<positional>"
    assert params["n_matched"] == 4
    assert params["source_file"] == "<DataFrame>"


def test_an_experiment_handle_is_accepted_and_reloaded(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})

    pyscx.attach_obs_columns(exp, df, positional=True)
    # The handle was reloaded by the wrapper, so it answers without raising.
    assert "score" in exp.read_obs().columns
