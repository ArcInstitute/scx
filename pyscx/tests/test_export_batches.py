"""Tests for `pyscx.export_batches`.

The export half of the doublet workflow. The loop itself is trivial; what this
helper earns its place with is the key check, so most of these are about
refusing to write a file whose rows a tool could not tell apart.

The motivating measurement is real: on a CELLxGENE-derived atlas, one batch in
1,086 had a duplicated obs index and it held 15.5% of the file's cells. Without
the check that surfaces much later — as a duplicate-key error at import, or as
scores landing on the wrong cell.
"""

import pathlib

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


def _fixture(tmp_path, obs, name="atlas.scx"):
    n = len(obs)
    X = sparse.csr_matrix(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    path = tmp_path / name
    pyscx.from_anndata(anndata.AnnData(X=X, obs=obs, var=var), str(path))
    return path


def _clean_obs():
    """Two donors, globally unique barcodes — the ordinary case."""
    return pd.DataFrame(
        {"donor_id": ["d1", "d1", "d2", "d2"], "cell_uid": ["u0", "u1", "u2", "u3"]},
        index=["AAAC-1", "AAAG-1", "AAAT-2", "AAAA-2"],
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_exports_one_h5ad_per_batch(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    out = tmp_path / "batches"

    r = pyscx.export_batches(str(scx), str(out), batch_key="donor_id")

    assert r["n_batches"] == 2
    assert r["n_cells_exported"] == 4
    assert r["key_is_globally_unique"]
    assert {b["batch"] for b in r["batches"]} == {"d1", "d2"}

    for b in r["batches"]:
        assert pathlib.Path(b["path"]).exists()
        assert b["key_unique_within_batch"]
        ad = anndata.read_h5ad(b["path"])
        assert ad.n_obs == b["n_cells"] == 2
        # The cells in each file really are that donor's.
        assert set(ad.obs["donor_id"]) == {b["batch"]}


def test_the_join_key_survives_into_the_exported_file(tmp_path):
    """Whatever key the import will use has to be present in what the tool sees."""
    scx = _fixture(tmp_path, _clean_obs())
    out = tmp_path / "batches"

    r = pyscx.export_batches(str(scx), str(out), batch_key="donor_id", key="cell_uid")
    assert r["key"] == "cell_uid"

    seen = set()
    for b in r["batches"]:
        ad = anndata.read_h5ad(b["path"])
        assert "cell_uid" in ad.obs.columns, "the key must reach the tool"
        seen |= set(ad.obs["cell_uid"])
    assert seen == {"u0", "u1", "u2", "u3"}


def test_key_defaults_to_what_the_importer_would_resolve(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")

    # The asymmetry this used to work around is gone: `diagnose_obs_key` now
    # ranks the obs index first among usable keys, so its *suggestion* — not
    # just its `resolved_key` — is what the export picks. Both are asserted,
    # because agreement between the two is the property that matters: the key
    # the export hands to a tool has to be one the import will resolve to.
    diag = pyscx.diagnose_obs_key(str(scx))
    assert r["key"] == diag["resolved_key"] == diag["suggestion"] == "obs_names"
    assert r["key_is_obs_index"] is True
    # And it is a name `obs_import` accepts — a display name the resolver
    # rejected would just be a new dead end.
    assert "cell_uid" in diag["unique_columns"], "the other unique column is still offered"


def test_batches_can_be_restricted(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(
        str(scx), str(tmp_path / "b"), batch_key="donor_id", batches=["d2"]
    )
    assert r["n_batches"] == 1
    assert r["batches"][0]["batch"] == "d2"


def test_round_trips_through_doublet_import(tmp_path):
    """Export, score each batch, concatenate, import once."""
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")
    assert r["key_is_globally_unique"], "else the import below needs a composite key"

    frames = []
    for i, b in enumerate(sorted(r["batches"], key=lambda x: x["batch"])):
        ad = anndata.read_h5ad(b["path"])
        frames.append(
            pd.DataFrame(
                {
                    "doublet_score": [0.1 * (i + 1)] * ad.n_obs,
                    "predicted_doublet": [i == 1] * ad.n_obs,
                },
                index=ad.obs_names,
            )
        )
    # Concatenated and imported ONCE: overwrite replaces rather than merges, so
    # importing per batch would keep only the last.
    calls = tmp_path / "calls.csv"
    pd.concat(frames).to_csv(calls)

    res = pyscx.doublet_import(str(scx), str(calls), tool="scrublet")
    assert res["n_matched"] == 4
    obs = pyscx.open(str(scx)).read_obs()
    assert obs["scrublet_status"].tolist() == ["present"] * 4


def test_exports_only_the_live_cells_of_a_deleted_file(tmp_path):
    """Pinned before `read_obs()` flipped to logical rows (pyscx 0.17).

    `export_batches` builds each batch's mask in the physical row space that
    `to_h5ad(obs_mask=)` requires, so it reads obs with `logical=False`. What a
    tool receives must not change across that flip: exactly the batch's live
    cells, never a deleted one. (The per-batch `n_cells` counts physical rows
    and is documented as an overcount on a file with deletions; it is not
    pinned here.)
    """
    scx = _fixture(tmp_path, _clean_obs())
    pyscx.mark_deleted(str(scx), [1])  # AAAG-1, a d1 cell
    exp = pyscx.open(str(scx))
    assert exp.n_obs == 3 and exp.n_obs_physical == 4, "premise"

    r = pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")

    assert r["n_batches"] == 2
    by_batch = {b["batch"]: b for b in r["batches"]}
    d1 = anndata.read_h5ad(by_batch["d1"]["path"])
    d2 = anndata.read_h5ad(by_batch["d2"]["path"])
    assert list(d1.obs_names) == ["AAAC-1"], "the deleted d1 cell is not exported"
    assert list(d2.obs_names) == ["AAAT-2", "AAAA-2"]
    assert set(d1.obs["donor_id"]) == {"d1"} and set(d2.obs["donor_id"]) == {"d2"}


# ---------------------------------------------------------------------------
# The key guard
# ---------------------------------------------------------------------------


def _ambiguous_obs():
    """Batch d2's two cells share a barcode — the census_1m shape in miniature."""
    return pd.DataFrame(
        {"donor_id": ["d1", "d1", "d2", "d2"], "cell_uid": ["u0", "u1", "u2", "u3"]},
        index=["AAAC-1", "AAAG-1", "DUP", "DUP"],
    )


def test_auto_resolution_prefers_the_index_when_it_is_unique(tmp_path):
    """A unique obs index wins over any other unique column.

    The index is what the tools hand back (`adata.obs_names` /
    `colnames(sce)`), so picking it makes the tool's identity and the import's
    join key the same column — no `key=` to remember at import time.
    """
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")

    # Reported as `obs_names`, not the physical `__index_level_0__`: that is a
    # pyarrow serialization name, and `read_obs()` hands the field back as the
    # frame's unnamed index, so `obs["__index_level_0__"]` is a KeyError.
    assert r["key"] == "obs_names"
    assert r["key_is_obs_index"] is True
    assert r["n_batches"] == 2


def test_a_unique_column_cannot_rescue_a_duplicated_index(tmp_path):
    """Previously this exported happily and was asserted to.

    Resolution picks `cell_uid` (the only unique column), but the exported
    h5ad still identifies its rows by the duplicated index, so batch d2 hands
    a tool two cells it cannot tell apart. Checking the resolved key alone
    certified exactly the wrong-join this helper exists to prevent.
    """
    scx = _fixture(tmp_path, _ambiguous_obs())

    with pytest.raises(ValueError) as e:
        pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")
    assert "obs_names" in str(e.value)


def test_an_explicitly_duplicated_key_refuses_before_writing(tmp_path):
    # The realistic shape of this failure: the caller names the key because
    # they know what the tool will emit, and it happens to be duplicated.
    scx = _fixture(tmp_path, _ambiguous_obs())
    out = tmp_path / "batches"

    with pytest.raises(ValueError) as e:
        pyscx.export_batches(
            str(scx), str(out), batch_key="donor_id", key="__index_level_0__"
        )
    msg = str(e.value)
    assert "d2" in msg, msg
    assert "not unique within" in msg, msg
    # Nothing half-written: every batch is planned before any is exported.
    assert not out.exists() or not list(out.glob("*.h5ad"))


def test_the_refusal_names_a_key_that_would_work(tmp_path):
    scx = _fixture(tmp_path, _ambiguous_obs())
    with pytest.raises(ValueError) as e:
        pyscx.export_batches(
            str(scx), str(tmp_path / "b"), batch_key="donor_id",
            key="__index_level_0__",
        )
    assert "diagnose_obs_key" in str(e.value)

    # `cell_uid` is unique, but naming it does NOT rescue the export: the h5ad
    # still identifies rows by the duplicated index, which is what a tool reads
    # back. Both identities have to work, not either one.
    with pytest.raises(ValueError) as e2:
        pyscx.export_batches(
            str(scx), str(tmp_path / "b2"), batch_key="donor_id", key="cell_uid",
        )
    assert "obs_names" in str(e2.value)


def test_a_file_with_no_usable_key_at_all_says_so(tmp_path):
    obs = pd.DataFrame({"donor_id": ["d1", "d1", "d2", "d2"]},
                       index=["AAAC-1", "AAAG-1", "DUP", "DUP"])
    scx = _fixture(tmp_path, obs)
    with pytest.raises(ValueError) as e:
        pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")
    msg = str(e.value)
    # The per-batch guard still fires, but its remedy ("pass a unique key")
    # would be unfollowable here, so the file-level diagnosis is appended.
    assert "not unique within" in msg, msg
    assert "no usable join key" in msg, msg


def test_skip_exports_the_good_batches_and_records_why(tmp_path):
    scx = _fixture(tmp_path, _ambiguous_obs())
    r = pyscx.export_batches(
        str(scx), str(tmp_path / "b"), batch_key="donor_id",
        key="__index_level_0__", on_ambiguous_key="skip",
    )

    assert r["n_batches"] == 1
    assert r["n_cells_exported"] == 2
    good = [b for b in r["batches"] if b["path"]]
    bad = [b for b in r["batches"] if not b["path"]]
    assert good[0]["batch"] == "d1"
    assert bad[0]["batch"] == "d2"
    # Not a silent omission.
    assert "not unique" in bad[0]["skipped_reason"]


def test_warn_exports_anyway(tmp_path):
    scx = _fixture(tmp_path, _ambiguous_obs())
    r = pyscx.export_batches(
        str(scx), str(tmp_path / "b"), batch_key="donor_id",
        key="__index_level_0__", on_ambiguous_key="warn",
    )
    assert r["n_batches"] == 2
    assert not [b for b in r["batches"] if b["batch"] == "d2"][0][
        "key_unique_within_batch"
    ]


def test_a_key_unique_only_within_batches_is_reported_not_refused(tmp_path):
    """Per-batch uniqueness is enough to export; global uniqueness is not.

    The distinction decides how the results come back: unique everywhere means
    concatenate and import once, unique only per batch means a composite key.
    """
    obs = pd.DataFrame(
        {"donor_id": ["d1", "d1", "d2", "d2"]},
        # The same two barcodes in both donors — the ordinary multi-library shape.
        index=["AAAC", "AAAG", "AAAC", "AAAG"],
    )
    scx = _fixture(tmp_path, obs)

    r = pyscx.export_batches(
        str(scx), str(tmp_path / "b"), batch_key="donor_id",
        key="__index_level_0__",
    )
    assert r["n_batches"] == 2
    assert all(b["key_unique_within_batch"] for b in r["batches"])
    assert not r["key_is_globally_unique"], (
        "the caller has to know a plain concatenate-and-import would collide"
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_a_missing_batch_key_errors_naming_the_columns(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    with pytest.raises(ValueError) as e:
        pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="nope")
    assert "donor_id" in str(e.value)


def test_a_missing_key_column_errors(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    with pytest.raises(ValueError, match="neither an obs column nor the obs index"):
        pyscx.export_batches(
            str(scx), str(tmp_path / "b"), batch_key="donor_id", key="nope"
        )


def test_a_batch_value_that_matches_nothing_errors(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    with pytest.raises(ValueError, match="matches no rows"):
        pyscx.export_batches(
            str(scx), str(tmp_path / "b"), batch_key="donor_id", batches=["d9"]
        )


def test_a_bad_on_ambiguous_key_errors(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    with pytest.raises(ValueError, match="on_ambiguous_key"):
        pyscx.export_batches(
            str(scx), str(tmp_path / "b"), batch_key="donor_id",
            on_ambiguous_key="bogus",
        )


def test_existing_files_are_not_clobbered_without_overwrite(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    out = tmp_path / "b"
    pyscx.export_batches(str(scx), str(out), batch_key="donor_id")

    with pytest.raises(FileExistsError):
        pyscx.export_batches(str(scx), str(out), batch_key="donor_id")

    r = pyscx.export_batches(
        str(scx), str(out), batch_key="donor_id", overwrite=True
    )
    assert r["n_batches"] == 2


def test_accepts_pathlib_and_an_experiment_handle(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(
        pathlib.Path(scx), pathlib.Path(tmp_path / "b"), batch_key="donor_id"
    )
    assert r["n_batches"] == 2

    exp = pyscx.open(str(scx))
    r = pyscx.export_batches(
        exp, str(tmp_path / "b2"), batch_key="donor_id"
    )
    assert r["n_batches"] == 2


# ---------------------------------------------------------------------------
# The exported identity, not just the resolved key
# ---------------------------------------------------------------------------


def test_the_report_says_whether_the_key_is_the_index(tmp_path):
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(str(scx), str(tmp_path / "out"), batch_key="donor_id")
    # Clean fixture: auto-resolution lands on the index, so a tool's obs_names
    # and the import's join key are the same thing.
    assert r["key_is_obs_index"] is True
    assert all(b["obs_names_unique_within_batch"] for b in r["batches"])


def test_skip_reports_which_identity_collided(tmp_path):
    obs = pd.DataFrame(
        {"donor_id": ["d1", "d1", "d2", "d2"],
         "cell_uid": ["u0", "u1", "u2", "u3"]},
        index=["A-1", "B-1", "DUP", "DUP"],
    )
    scx = _fixture(tmp_path, obs)

    r = pyscx.export_batches(str(scx), str(tmp_path / "out"),
                             batch_key="donor_id", on_ambiguous_key="skip")

    by_batch = {b["batch"]: b for b in r["batches"]}
    assert by_batch["d1"]["path"] is not None
    assert by_batch["d2"]["path"] is None
    assert "obs_names" in by_batch["d2"]["skipped_reason"]


def test_a_batch_label_cannot_escape_the_output_directory(tmp_path):
    """Batch labels are obs data, not identifiers: `../x` would otherwise write
    outside out_dir."""
    obs = pd.DataFrame(
        {"donor_id": ["../evil", "../evil", "ok/slash", "ok/slash"]},
        index=[f"c{i}" for i in range(4)],
    )
    scx = _fixture(tmp_path, obs)
    out = tmp_path / "out"

    r = pyscx.export_batches(str(scx), str(out), batch_key="donor_id")

    for b in r["batches"]:
        written = pathlib.Path(b["path"]).resolve()
        assert written.parent == out.resolve(), written


def test_labels_that_sanitise_alike_get_distinct_files(tmp_path):
    """`batch/1` and `batch_1` both sanitise to `batch_1`.

    Left unchecked that is a FileExistsError about a file this very call just
    wrote (overwrite=False) or, worse, one batch silently overwriting the
    other's export (overwrite=True) — the second batch's cells would never
    reach a tool and the first's results would be gone.
    """
    obs = pd.DataFrame(
        {"donor_id": ["batch/1", "batch/1", "batch_1", "batch_1"]},
        index=[f"c{i}" for i in range(4)],
    )
    scx = _fixture(tmp_path, obs)

    r = pyscx.export_batches(str(scx), str(tmp_path / "out"), batch_key="donor_id")

    paths = [b["path"] for b in r["batches"]]
    assert len(set(paths)) == len(paths) == 2, paths
    assert r["n_cells_exported"] == 4
    # Each file really holds its own batch's cells.
    for b in r["batches"]:
        assert anndata.read_h5ad(b["path"]).n_obs == 2


def test_a_named_index_is_also_preferred_when_unique(tmp_path):
    """The preference must cover anndata's `_index` spelling, not only
    pyarrow's `__index_level_0__`."""
    scx = _fixture(tmp_path, _clean_obs())
    r = pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")
    assert r["key_is_obs_index"] is True
