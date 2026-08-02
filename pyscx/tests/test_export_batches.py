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
    # Same resolution the importer performs, so the two cannot disagree.
    assert r["key"] == pyscx.diagnose_obs_key(str(scx))["suggestion"]


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


# ---------------------------------------------------------------------------
# The key guard
# ---------------------------------------------------------------------------


def _ambiguous_obs():
    """Batch d2's two cells share a barcode — the census_1m shape in miniature."""
    return pd.DataFrame(
        {"donor_id": ["d1", "d1", "d2", "d2"], "cell_uid": ["u0", "u1", "u2", "u3"]},
        index=["AAAC-1", "AAAG-1", "DUP", "DUP"],
    )


def test_auto_resolution_prefers_a_key_that_actually_works(tmp_path):
    """Given a duplicated index and a unique column, it picks the unique one.

    This is why the helper resolves through `diagnose_obs_key`'s *suggestion*
    rather than the plain fallback order: the fallback would land on the index,
    which is exactly the column that cannot be joined on.
    """
    scx = _fixture(tmp_path, _ambiguous_obs())
    r = pyscx.export_batches(str(scx), str(tmp_path / "b"), batch_key="donor_id")

    assert r["key"] == "cell_uid"
    assert r["n_batches"] == 2
    assert all(b["key_unique_within_batch"] for b in r["batches"])


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

    # And the key it points at does work.
    r = pyscx.export_batches(
        str(scx), str(tmp_path / "b2"), batch_key="donor_id", key="cell_uid",
    )
    assert r["n_batches"] == 2
    assert all(b["key_unique_within_batch"] for b in r["batches"])


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
