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


def test_positional_row_count_mismatch_names_the_file_count(tmp_path):
    scx = _fixture(tmp_path)  # n_obs = 4, no deletions
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match=r"3 rows.*n_obs = 4.*no logical deletions"):
        pyscx.attach_obs_columns(str(scx), df, positional=True)


# On a file with deletions a positional frame is accepted in either row space,
# told apart by length: `read_obs()` (live rows, since 0.17) or
# `read_obs(logical=False)` (every physical row).


def _deleted_fixture(tmp_path):
    scx = _fixture(tmp_path)  # AAAC-1, AAAG-1, AAAT-1, AAAA-1
    pyscx.mark_deleted(str(scx), [1])
    exp = pyscx.open(str(scx))
    assert exp.n_obs == 3 and exp.n_obs_physical == 4, "premise"
    return scx, exp


def test_positional_accepts_a_frame_computed_from_read_obs_on_a_deleted_file(tmp_path):
    # The docstring's own example: a column computed row-for-row from
    # `read_obs()` — which is the live frame now — lands on the live rows.
    scx, exp = _deleted_fixture(tmp_path)
    obs = exp.read_obs()
    scores = pd.DataFrame({"my_score": [10.0 * (i + 1) for i in range(len(obs))]})

    r = pyscx.attach_obs_columns(str(scx), scores, positional=True)
    assert r["n_obs"] == 4, "the physical axis the attach was built over"
    assert r["n_matched"] == 3
    assert r["n_target_rows_absent"] == 1, "the deleted row"

    backed = pyscx.open(str(scx)).to_anndata(backed=True).obs
    assert backed["my_score"].tolist() == [10.0, 20.0, 30.0]
    assert list(backed.index) == ["AAAC-1", "AAAT-1", "AAAA-1"]
    physical = pyscx.open(str(scx)).read_obs(logical=False)
    assert physical["my_score"].tolist()[0] == 10.0
    assert pd.isna(physical["my_score"].iloc[1]), "the deleted row is null, not a value"
    assert physical["my_score"].tolist()[2:] == [20.0, 30.0]


def test_positional_still_accepts_a_physical_length_frame_on_a_deleted_file(tmp_path):
    scx, exp = _deleted_fixture(tmp_path)
    scores = pd.DataFrame({"my_score": [1.0, 2.0, 3.0, 4.0]})

    r = pyscx.attach_obs_columns(str(scx), scores, positional=True)
    assert r["n_matched"] == 4 and r["n_target_rows_absent"] == 0
    physical = pyscx.open(str(scx)).read_obs(logical=False)
    assert physical["my_score"].tolist() == [1.0, 2.0, 3.0, 4.0], "the deleted row is written"
    assert pyscx.open(str(scx)).read_obs()["my_score"].tolist() == [1.0, 3.0, 4.0]


def test_positional_neither_length_names_both_counts_on_a_deleted_file(tmp_path):
    scx, _ = _deleted_fixture(tmp_path)
    for n in (2, 5):
        df = pd.DataFrame({"score": [0.1] * n})
        with pytest.raises(ValueError) as e:
            pyscx.attach_obs_columns(str(scx), df, positional=True)
        msg = str(e.value)
        for needle in (f"{n} rows", "n_obs = 3", "n_obs_physical = 4", "read_obs()",
                       "read_obs(logical=False)"):
            assert needle in msg, f"{needle!r} missing from: {msg}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_a_dict_is_rejected_naming_the_function(tmp_path):
    scx = _fixture(tmp_path)
    with pytest.raises(TypeError, match="attach_obs_columns"):
        pyscx.attach_obs_columns(str(scx), {"score": [1, 2, 3, 4]})


def test_uns_key_without_uns_is_rejected(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="uns_key"):
        pyscx.attach_obs_columns(str(scx), df, positional=True, uns_key="m")


def test_uns_without_uns_key_must_be_a_dict(tmp_path):
    """Without `uns_key` the payload is spread at top level, so it has to have
    top-level keys. Checked on the Python object: under the tagged format a
    tuple normalises to a JSON *object* (an `__scx_type__` envelope), and a
    post-hoc object check would have sprayed envelope fields into uns."""
    scx = _fixture(tmp_path)
    before = scx.read_bytes()
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    for bad in [(1, 2), [1, 2], np.array([1.0, 2.0]), "s"]:
        with pytest.raises(ValueError, match="must be a dict"):
            pyscx.attach_obs_columns(str(scx), df, positional=True, uns=bad)
    assert scx.read_bytes() == before
    assert "__scx_type__" not in (pyscx.open(str(scx)).read_uns() or {})


# ---------------------------------------------------------------------------
# uns merge, rollback, dry run
# ---------------------------------------------------------------------------


def test_uns_without_uns_key_merges_top_level_keys_and_one_rollback_undoes_all(tmp_path):
    """Two obs columns + three uns keys, one commit, one rollback — the shape
    that used to force a whole-obs `modify_metadata(obs=, uns=)` re-encode."""
    scx = _fixture(tmp_path)
    pyscx.set_uns(str(scx), {"existing": "kept", "nested": {"a": [1, 2]}})
    before = pyscx.open(str(scx)).read_uns()
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4], "flag": [True, False, True, False]})

    pyscx.attach_obs_columns(
        str(scx),
        df,
        positional=True,
        uns={"de": {"method": "wilcoxon"}, "de_params": [1, 2.5], "note": "s"},
    )
    exp = pyscx.open(str(scx))
    uns = exp.read_uns()
    assert uns["de"] == {"method": "wilcoxon"}
    assert list(uns["de_params"]) == [1, 2.5]
    assert uns["note"] == "s"
    for k, v in before.items():
        assert uns[k] == v, f"pre-existing uns[{k!r}] changed"
    obs = exp.read_obs()
    assert "score" in obs.columns and "flag" in obs.columns

    pyscx.rollback(str(scx))
    exp = pyscx.open(str(scx))
    obs = exp.read_obs()
    assert "score" not in obs.columns and "flag" not in obs.columns
    assert exp.read_uns() == before, "one rollback undoes columns and every uns key"


def test_uns_collision_names_the_key(tmp_path):
    scx = _fixture(tmp_path)
    pyscx.set_uns(str(scx), {"existing": "kept"})
    df = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="'existing'"):
        pyscx.attach_obs_columns(
            str(scx), df, positional=True, uns={"fresh": 1, "existing": "new"}
        )
    pyscx.attach_obs_columns(
        str(scx), df, positional=True, uns={"fresh": 1, "existing": "new"}, overwrite=True
    )
    uns = pyscx.open(str(scx)).read_uns()
    assert uns["existing"] == "new" and uns["fresh"] == 1


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


def test_positional_ignores_a_named_pandas_index(tmp_path):
    """Round-1 finding (Cursor/codex/Antigravity): pyarrow materializes a NAMED
    index under its own name, not `__index_level_0__` — it must still be
    dropped, or it collides with the target's same-named column."""
    obs = pd.DataFrame(
        {"barcode": ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]},
        index=[f"c{i}" for i in range(4)],
    )
    scx = _fixture(tmp_path, obs=obs)
    df = pd.DataFrame(
        {"score": [0.1, 0.2, 0.3, 0.4]},
        index=pd.Index(["x", "y", "z", "w"], name="barcode"),
    )

    r = pyscx.attach_obs_columns(str(scx), df, positional=True)
    assert r["obs_columns_added"] == ["score"]
    got = _obs(scx)
    assert got["score"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert got["barcode"].tolist() == ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"], (
        "the df's named index must be ignored, not attached over the target's column"
    )


def test_key_none_with_a_named_index_joins_against_the_target_obs_index(tmp_path):
    """key=None resolves each side independently (as obs_import does): the
    source uses its index whatever its name; the target uses its own obs
    index — not a column named after the source index."""
    scx = _fixture(tmp_path)  # target obs index: AAAC-1 …
    df = pd.DataFrame(
        {"grp": ["c", "a", "g"]},
        index=pd.Index(["AAAT-1", "AAAC-1", "AAAG-1"], name="my_cells"),
    )

    r = pyscx.attach_obs_columns(str(scx), df)
    assert r["n_matched"] == 3
    got = _obs(scx)
    assert "my_cells" not in got.columns, "the named source index is consumed, not attached"
    by_cell = dict(zip(got.index, got["grp"]))
    assert by_cell["AAAT-1"] == "c" and by_cell["AAAC-1"] == "a"


def test_status_column_matching_an_annotation_name_is_rejected(tmp_path):
    """Round-2 finding (codex): the collision check compares planned names
    against the OLD schema only, so this used to write TWO 'score' columns."""
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [0.1, 0.2]}, index=["AAAT-1", "AAAC-1"])
    with pytest.raises(ValueError, match="same name"):
        pyscx.attach_obs_columns(str(scx), df, status_column="score")
