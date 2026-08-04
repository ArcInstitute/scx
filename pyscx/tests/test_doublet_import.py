"""Tests for `pyscx.doublet_import`.

The generic importer is covered by `test_obs_import.py`; what is new here is
the per-tool column mapping and the value derivation. So the fixtures are
written the way the tools actually write them — R's `write.csv` with a quoted
header, an unnamed index column and bare `NA`; pandas' `to_csv` with `True` /
`False` — and every fixture is deliberately out of order relative to the SCX
file, because a positional import would produce a correctly-shaped column with
every score on the wrong cell.
"""

import pathlib

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")

BARCODES = ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]


def _fixture(tmp_path, name="t.scx", obs=None):
    """A small SCX file. `obs` defaults to four unique barcodes."""
    if obs is None:
        obs = pd.DataFrame(index=BARCODES)
    n = len(obs)
    X = sparse.csr_matrix(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    path = tmp_path / name
    pyscx.from_anndata(anndata.AnnData(X=X, obs=obs, var=var), str(path))
    return path


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def _calls(obs, column):
    """Read a nullable boolean obs column as a list of True/False/None."""
    return [None if pd.isna(v) else bool(v) for v in obs[column]]


# ---------------------------------------------------------------------------
# One test per tool profile
# ---------------------------------------------------------------------------


def test_scdblfinder_profile(tmp_path):
    scx = _fixture(tmp_path)
    # What `write.csv(as.data.frame(colData(sce)[, ...]))` produces. Out of
    # order, and covering three of four cells.
    csv = _write(
        tmp_path,
        "sce.csv",
        '"","scDblFinder.score","scDblFinder.class","scDblFinder.mostLikelyOrigin"\n'
        '"AAAT-1",0.30,"singlet",NA\n'
        '"AAAC-1",0.10,"singlet",NA\n'
        '"AAAG-1",0.90,"doublet","1+2"\n',
    )

    r = pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder")
    assert r["tool"] == "scdblfinder"
    assert r["key_added"] == "scdblfinder"
    assert r["score_source_column"] == "scDblFinder.score"
    assert r["call_source_column"] == "scDblFinder.class"
    assert r["canonical_columns"] == ["scdblfinder_score", "scdblfinder_predicted"]
    assert r["n_matched"] == 3

    obs = pyscx.open(str(scx)).read_obs()
    # File order, not CSV order.
    assert obs["scdblfinder_score"].tolist()[:3] == pytest.approx([0.10, 0.90, 0.30])
    assert pd.isna(obs["scdblfinder_score"].iloc[3]), "uncovered cell must be null"
    assert _calls(obs, "scdblfinder_predicted") == [False, True, False, None]
    assert obs["scdblfinder_status"].tolist() == [
        "present",
        "present",
        "present",
        "absent",
    ]
    # Every other source column survives, prefixed and unchanged.
    assert "scdblfinder_scDblFinder.mostLikelyOrigin" in obs.columns


def test_scrublet_profile(tmp_path):
    scx = _fixture(tmp_path)
    obs_out = pd.DataFrame(
        {"doublet_score": [0.71, 0.03], "predicted_doublet": [True, False]},
        index=["AAAG-1", "AAAC-1"],
    )
    csv = tmp_path / "scrublet.csv"
    obs_out.to_csv(csv)  # the real shape: a leading unnamed index column

    r = pyscx.doublet_import(str(scx), str(csv), tool="scrublet")
    assert r["renamed_index_column"]
    assert r["canonical_columns"] == ["scrublet_score", "scrublet_predicted"]

    obs = pyscx.open(str(scx)).read_obs()
    assert obs["scrublet_score"].tolist()[:2] == pytest.approx([0.03, 0.71])
    assert _calls(obs, "scrublet_predicted") == [False, True, None, None]


def test_doubletfinder_profile_matches_by_prefix(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "seurat.csv",
        "barcode,pANN_0.25_0.09_87,DF.classifications_0.25_0.09_87\n"
        "AAAC-1,0.11,Singlet\n"
        "AAAG-1,0.87,Doublet\n",
    )

    r = pyscx.doublet_import(str(scx), str(csv), tool="doubletfinder")
    assert r["score_source_column"] == "pANN_0.25_0.09_87"
    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "doubletfinder_predicted") == [False, True, None, None]


def test_doubletdetection_profile_reads_a_numeric_label(tmp_path):
    scx = _fixture(tmp_path)
    # NaN is what the classifier emits for a cell it never converged on. That
    # must stay a null, not become a singlet call.
    csv = _write(
        tmp_path,
        "dd.csv",
        "barcode,doublet_score,doublet_label\n"
        "AAAC-1,1.2,0\n"
        "AAAG-1,8.4,1\n"
        "AAAT-1,3.0,NaN\n",
    )

    pyscx.doublet_import(str(scx), str(csv), tool="doubletdetection")
    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "doubletdetection_predicted") == [False, True, None, None]


def test_solo_profile(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "solo.csv",
        "barcode,softmax_score,prediction\nAAAC-1,0.1,singlet\nAAAG-1,0.9,doublet\n",
    )
    r = pyscx.doublet_import(str(scx), str(csv), tool="solo")
    assert r["score_source_column"] == "softmax_score"
    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "solo_predicted") == [False, True, None, None]


def test_scds_profile_omits_the_call_column(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "scds.csv",
        "barcode,cxds_score,bcds_score,hybrid_score\nAAAC-1,0.4,0.2,0.31\n",
    )

    r = pyscx.doublet_import(str(scx), str(csv), tool="scds")
    assert r["score_source_column"] == "hybrid_score"
    assert r["call_source_column"] is None
    assert r["canonical_columns"] == ["scds_score"]

    obs = pyscx.open(str(scx)).read_obs()
    assert "scds_score" in obs.columns
    # Never thresholded into existence: choosing a cutoff is a scientific
    # decision the importer does not own.
    assert "scds_predicted" not in obs.columns
    assert "scds_cxds_score" in obs.columns


def test_generic_profile_takes_both_columns_from_the_caller(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "mine.csv",
        "barcode,my_score,my_call\nAAAC-1,0.5,yes\nAAAG-1,0.9,no\n",
    )
    pyscx.doublet_import(
        str(scx),
        str(csv),
        tool="generic",
        key_added="dbl",
        score_column="my_score",
        call_column="my_call",
        call_true="yes",
        call_false="no",
    )
    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "dbl_predicted") == [True, False, None, None]


# ---------------------------------------------------------------------------
# Cross-tool behaviour
# ---------------------------------------------------------------------------


def test_two_tools_coexist_and_are_comparable(tmp_path):
    """The payoff: consensus is reading two columns, not re-running anything."""
    scx = _fixture(tmp_path)
    a = _write(
        tmp_path,
        "a.csv",
        "barcode,scDblFinder.score,scDblFinder.class\n"
        + "".join(
            f"{b},{s},{c}\n"
            for b, s, c in zip(BARCODES, [0.1, 0.9, 0.2, 0.3],
                               ["singlet", "doublet", "singlet", "singlet"])
        ),
    )
    b = _write(
        tmp_path,
        "b.csv",
        "barcode,doublet_score,predicted_doublet\n"
        + "".join(
            f"{bc},{s},{c}\n"
            for bc, s, c in zip(BARCODES, [0.05, 0.80, 0.10, 0.15],
                                ["False", "True", "False", "False"])
        ),
    )

    pyscx.doublet_import(str(scx), str(a), tool="scdblfinder")
    pyscx.doublet_import(str(scx), str(b), tool="scrublet")

    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "scdblfinder_predicted") == [False, True, False, False]
    assert _calls(obs, "scrublet_predicted") == [False, True, False, False]
    # Canonical names, so the agreement is computable without knowing the tools.
    agree = (obs["scdblfinder_predicted"] == obs["scrublet_predicted"]).all()
    assert agree


def test_key_added_lets_the_same_tool_run_twice(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    )
    pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder", key_added="run1")
    pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder", key_added="run2")

    obs = pyscx.open(str(scx)).read_obs()
    assert {"run1_score", "run2_score", "run1_predicted", "run2_predicted"} <= set(
        obs.columns
    )


def test_composite_key_across_libraries(tmp_path):
    bc = ["AAAC-1", "AAAG-1", "AAAC-1", "AAAG-1"]
    obs = pd.DataFrame(
        {"sample_id": ["A", "A", "B", "B"], "barcode": bc},
        index=[f"{s}_{b}" for s, b in zip(["A", "A", "B", "B"], bc)],
    )
    scx = _fixture(tmp_path, obs=obs)
    csv = _write(
        tmp_path,
        "sce.csv",
        "sample_id,barcode,scDblFinder.score,scDblFinder.class\n"
        "A,AAAC-1,0.1,singlet\n"
        "A,AAAG-1,0.2,singlet\n"
        "B,AAAC-1,0.9,doublet\n"
        "B,AAAG-1,0.8,doublet\n",
    )

    r = pyscx.doublet_import(
        str(scx), str(csv), tool="scdblfinder", key=["sample_id", "barcode"]
    )
    assert r["n_matched"] == 4
    obs_back = pyscx.open(str(scx)).read_obs()
    assert _calls(obs_back, "scdblfinder_predicted") == [False, False, True, True]


# ---------------------------------------------------------------------------
# Refusals — clean exceptions, never panics
# ---------------------------------------------------------------------------


def test_missing_expected_column_errors_naming_present_columns(tmp_path):
    scx = _fixture(tmp_path)
    # A scrublet-shaped file handed to the scDblFinder profile.
    csv = _write(
        tmp_path,
        "wrong.csv",
        "barcode,doublet_score,predicted_doublet\nAAAC-1,0.03,False\n",
    )
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder")
    msg = str(e.value)
    assert "scDblFinder.score" in msg, msg
    assert "doublet_score" in msg, msg


def test_ambiguous_doubletfinder_pann_columns_error(tmp_path):
    scx = _fixture(tmp_path)
    # DoubletFinder run twice with different pK leaves both columns behind.
    csv = _write(
        tmp_path,
        "two.csv",
        "barcode,pANN_0.25_0.09_87,pANN_0.25_0.30_54,DF.classifications_0.25_0.09_87\n"
        "AAAC-1,0.1,0.2,Singlet\n",
    )
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(csv), tool="doubletfinder")
    msg = str(e.value)
    assert "pANN_0.25_0.09_87" in msg and "pANN_0.25_0.30_54" in msg, msg

    # Naming one resolves it, which is what the message says to do.
    r = pyscx.doublet_import(
        str(scx), str(csv), tool="doubletfinder", score_column="pANN_0.25_0.30_54"
    )
    assert r["score_source_column"] == "pANN_0.25_0.30_54"


def test_unknown_call_token_errors_naming_the_value(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\n"
        "AAAC-1,0.02,singlet\n"
        "AAAG-1,0.51,ambiguous\n",
    )
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder")
    msg = str(e.value)
    assert "ambiguous" in msg, msg
    assert "doublet" in msg and "singlet" in msg, msg

    # The escape hatch the message points at.
    pyscx.doublet_import(
        str(scx),
        str(csv),
        tool="scdblfinder",
        call_true="ambiguous",
        call_false="singlet",
    )
    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "scdblfinder_predicted") == [False, True, None, None]


def test_unknown_tool_errors_listing_the_valid_set(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "x.csv", "barcode,doublet_score\nAAAC-1,0.5\n")
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(csv), tool="scDblFinderr")
    msg = str(e.value)
    assert "scdblfinder" in msg and "doubletdetection" in msg, msg


def test_an_h5ad_is_read_rather_than_refused_on_its_extension(tmp_path):
    # Phase 5 made h5ad a real source, so the blanket deferral is gone. This
    # file is not HDF5 at all, so it fails as unreadable rather than as an
    # unsupported format.
    scx = _fixture(tmp_path)
    h5 = _write(tmp_path, "scrublet_out.h5ad", "not really hdf5")
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(h5), tool="scrublet")
    msg = str(e.value)
    assert "as HDF5" in msg, msg
    assert "not implemented" not in msg, f"the Phase-4 deferral must be gone: {msg}"


def test_a_rejected_import_leaves_the_file_byte_identical(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "wrong.csv",
        "barcode,doublet_score,predicted_doublet\nAAAC-1,0.03,False\n",
    )
    before = pathlib.Path(scx).read_bytes()
    with pytest.raises(ValueError):
        pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder")
    assert pathlib.Path(scx).read_bytes() == before


def test_second_import_errors_and_overwrite_replaces(tmp_path):
    scx = _fixture(tmp_path)
    a = _write(
        tmp_path,
        "a.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.1,singlet\n",
    )
    b = _write(
        tmp_path,
        "b.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAT-1,0.9,doublet\n",
    )

    pyscx.doublet_import(str(scx), str(a), tool="scdblfinder")
    with pytest.raises(ValueError, match="already exists"):
        pyscx.doublet_import(str(scx), str(b), tool="scdblfinder")

    pyscx.doublet_import(str(scx), str(b), tool="scdblfinder", overwrite=True)
    obs = pyscx.open(str(scx)).read_obs()
    # Batch A is gone: overwrite REPLACES. Importing per-batch tables one after
    # another keeps only the last -- concatenate and import once.
    assert pd.isna(obs["scdblfinder_score"].iloc[0])
    assert obs["scdblfinder_score"].iloc[2] == pytest.approx(0.9)


def test_dry_run_writes_nothing_and_carries_a_key_diagnosis(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    )
    before = pathlib.Path(scx).read_bytes()

    r = pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder", dry_run=True)
    assert r["dry_run"]
    assert r["n_matched"] == 1
    assert "key_diagnosis" in r
    assert pathlib.Path(scx).read_bytes() == before


def test_rollback_undoes_the_import(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    )
    pyscx.doublet_import(str(scx), str(csv), tool="scdblfinder")
    assert "scdblfinder_score" in pyscx.open(str(scx)).read_obs().columns

    pyscx.rollback(str(scx))
    obs = pyscx.open(str(scx)).read_obs()
    assert "scdblfinder_score" not in obs.columns
    assert "scdblfinder_status" not in obs.columns


def test_accepts_pathlib_and_an_experiment_handle(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "sce.csv",
        "barcode,scDblFinder.score,scDblFinder.class\nAAAC-1,0.5,doublet\n",
    )
    r = pyscx.doublet_import(
        pathlib.Path(scx), pathlib.Path(csv), tool="scdblfinder", dry_run=True
    )
    assert r["n_matched"] == 1

    exp = pyscx.open(str(scx))
    r = pyscx.doublet_import(exp, str(csv), tool="scdblfinder", dry_run=True)
    assert r["n_matched"] == 1


def test_doublet_tools_matches_what_the_importer_accepts(tmp_path):
    tools = pyscx.doublet_tools()
    assert "scdblfinder" in tools and "generic" in tools
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "x.csv", "barcode,s\nAAAC-1,0.5\n")
    for t in tools:
        # Every advertised name must reach the profile table. `generic` fails
        # for a different reason (no score column), which is still not "unknown
        # tool".
        try:
            pyscx.doublet_import(
                str(scx), str(csv), tool=t, score_column="s", dry_run=True
            )
        except ValueError as e:
            assert "unknown doublet tool" not in str(e), (t, str(e))


# ---------------------------------------------------------------------------
# h5ad sources (Phase 5)
#
# The scanpy-resident tools write their results back into an h5ad in place, so
# these use files `anndata` actually wrote rather than hand-built ones. The
# assertion that matters is *equivalence*: the h5ad route must be a way to the
# same place as the CSV route, not a second implementation with its own quirks.
# ---------------------------------------------------------------------------


def _scrublet_shaped_h5ad(tmp_path, name="scrublet_out.h5ad", categorical=False):
    """What `sc.pp.scrublet(adata)` leaves behind: results in obs, in place."""
    obs = pd.DataFrame(
        {
            "doublet_score": [0.03, 0.71, 0.05],
            "predicted_doublet": [False, True, False],
        },
        index=["AAAC-1", "AAAG-1", "AAAT-1"],
    )
    if categorical:
        # pandas stores this as a Categorical, which lands as an arrow
        # dictionary -- the shape the derivation had to learn to see through.
        obs["scDblFinder.class"] = pd.Categorical(["singlet", "doublet", "singlet"])
    X = sparse.csr_matrix(np.arange(len(obs) * 3, dtype=np.float32).reshape(len(obs), 3))
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    p = tmp_path / name
    anndata.AnnData(X=X, obs=obs, var=var).write_h5ad(p)
    return p


def test_h5ad_source_imports_without_a_csv_detour(tmp_path):
    scx = _fixture(tmp_path)
    h5 = _scrublet_shaped_h5ad(tmp_path)

    r = pyscx.doublet_import(str(scx), str(h5), tool="scrublet")
    assert r["format"] == "h5ad"
    assert r["delimiter"] is None, "there is no delimiter in an HDF5 file"
    assert r["score_source_column"] == "doublet_score"
    assert r["n_matched"] == 3

    obs = pyscx.open(str(scx)).read_obs()
    assert obs["scrublet_score"].tolist()[:3] == pytest.approx([0.03, 0.71, 0.05])
    assert _calls(obs, "scrublet_predicted") == [False, True, False, None]
    assert obs["scrublet_status"].tolist()[3] == "absent"


def test_h5ad_and_csv_paths_agree_exactly(tmp_path):
    """The exit criterion: two routes, one result."""
    h5 = _scrublet_shaped_h5ad(tmp_path)
    # The same values, via the CSV route the user had to take before.
    src = anndata.read_h5ad(h5).obs[["doublet_score", "predicted_doublet"]]
    csv = tmp_path / "calls.csv"
    src.to_csv(csv)

    from_h5ad = _fixture(tmp_path, name="a.scx")
    from_csv = _fixture(tmp_path, name="b.scx")
    r_h5 = pyscx.doublet_import(str(from_h5ad), str(h5), tool="scrublet")
    r_csv = pyscx.doublet_import(str(from_csv), str(csv), tool="scrublet")

    assert r_h5["n_matched"] == r_csv["n_matched"]
    assert r_h5["canonical_columns"] == r_csv["canonical_columns"]
    assert r_h5["score_source_column"] == r_csv["score_source_column"]

    a = pyscx.open(str(from_h5ad)).read_obs()
    b = pyscx.open(str(from_csv)).read_obs()
    # `Series.equals` rather than `==`: an uncovered cell is NaN on both sides,
    # and `nan == nan` is False, which would fail a comparison that is in fact
    # holding. Positional NaN equality is exactly the semantics wanted here.
    for col in ["scrublet_score", "scrublet_predicted", "scrublet_status"]:
        assert a[col].equals(b[col]), (
            f"{col} differs between the h5ad and CSV routes:\n"
            f"  h5ad: {a[col].tolist()}\n  csv : {b[col].tolist()}"
        )


def test_h5ad_categorical_class_column_derives(tmp_path):
    """A pandas Categorical arrives dictionary-encoded; it must still derive."""
    scx = _fixture(tmp_path)
    h5 = _scrublet_shaped_h5ad(tmp_path, name="sce.h5ad", categorical=True)

    r = pyscx.doublet_import(
        str(scx), str(h5), tool="scdblfinder", score_column="doublet_score"
    )
    assert r["call_source_column"] == "scDblFinder.class"
    obs = pyscx.open(str(scx)).read_obs()
    assert _calls(obs, "scdblfinder_predicted") == [False, True, False, None]


def test_h5ad_uns_key_is_opt_in_and_nested(tmp_path):
    scx = _fixture(tmp_path)
    h5 = _scrublet_shaped_h5ad(tmp_path)
    ad = anndata.read_h5ad(h5)
    ad.uns["scrublet"] = {"threshold": 0.35}
    ad.write_h5ad(h5)

    # Nothing by default: /uns routinely holds things worth not importing.
    r = pyscx.doublet_import(str(scx), str(h5), tool="scrublet")
    assert r["uns_keys_imported"] == []

    pyscx.rollback(str(scx))
    r = pyscx.doublet_import(
        str(scx), str(h5), tool="scrublet", uns_keys=["scrublet"]
    )
    assert r["uns_keys_imported"] == ["scrublet"]


def test_a_missing_uns_key_errors_naming_what_is_present(tmp_path):
    scx = _fixture(tmp_path)
    h5 = _scrublet_shaped_h5ad(tmp_path)
    ad = anndata.read_h5ad(h5)
    ad.uns["scrublet"] = {"threshold": 0.35}
    ad.write_h5ad(h5)

    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(
            str(scx), str(h5), tool="scrublet", uns_keys=["scdblfinder"]
        )
    msg = str(e.value)
    assert "scdblfinder" in msg and "scrublet" in msg, msg


def test_uns_keys_on_a_csv_source_error_rather_than_being_ignored(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,doublet_score\nAAAC-1,0.5\n")
    with pytest.raises(ValueError, match="carries no uns"):
        pyscx.doublet_import(str(scx), str(csv), tool="scrublet", uns_keys=["x"])


def test_obs_import_also_reads_an_h5ad(tmp_path):
    """The generic importer gained the same route, not just the wrapper."""
    scx = _fixture(tmp_path)
    h5 = _scrublet_shaped_h5ad(tmp_path)

    r = pyscx.obs_import(str(scx), str(h5))
    assert r["format"] == "h5ad"
    obs = pyscx.open(str(scx)).read_obs()
    assert obs["doublet_score"].tolist()[:3] == pytest.approx([0.03, 0.71, 0.05])


def test_h5mu_source_is_refused_with_the_modality_route(tmp_path):
    scx = _fixture(tmp_path)
    fake = _write(tmp_path, "atlas.h5mu", "not really hdf5")
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(fake), tool="scrublet")
    assert "--modality" in str(e.value)


# ---------------------------------------------------------------------------
# B3: a declared-but-absent call column must not degrade silently
# ---------------------------------------------------------------------------


def test_a_declared_call_column_absent_warns(tmp_path):
    """The dogfood repro: a scrublet-shaped table imported as doubletdetection.

    It still imports (score-only is valid and loses no data), but it must say so
    — the failure otherwise surfaces much later in `doublet_consensus`.
    """
    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n"
        "AAAC-1,0.10,True\nAAAG-1,0.20,False\n"
        "AAAT-1,0.30,False\nAAAA-1,0.40,False\n",
    )

    with pytest.warns(UserWarning, match="doublet_label"):
        r = pyscx.doublet_import(str(path), str(table), tool="doubletdetection")

    assert r["call_source_column"] is None
    assert r["call_column_status"] == "declared_but_absent"
    assert r["expected_call_columns"] == ["doublet_label"]
    assert r["canonical_columns"] == ["doubletdetection_score"]

    obs = pyscx.open(str(path)).read_obs()
    assert "doubletdetection_score" in obs.columns
    assert "doubletdetection_predicted" not in obs.columns
    # No data lost — the unmatched column survives verbatim.
    assert "doubletdetection_predicted_doublet" in obs.columns


def test_the_warning_names_the_column_the_table_actually_has(tmp_path):
    """A near-miss is usually the wrong `tool=`, so the message should point at
    the real column rather than leave the user to diff two vocabularies."""
    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n"
        "AAAC-1,0.10,True\nAAAG-1,0.20,False\n"
        "AAAT-1,0.30,False\nAAAA-1,0.40,False\n",
    )
    with pytest.warns(UserWarning, match=r'call_column="predicted_doublet"'):
        pyscx.doublet_import(str(path), str(table), tool="doubletdetection")


def test_a_score_only_profile_does_not_warn(tmp_path):
    """Guards against over-warning: scds declares no call column, so its absence
    is by design and a warning there would make the real one worthless."""
    import warnings

    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "scds.csv",
        "barcode,hybrid_score\nAAAC-1,0.10\nAAAG-1,0.20\n"
        "AAAT-1,0.30\nAAAA-1,0.40\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = pyscx.doublet_import(str(path), str(table), tool="scds")
    assert r["call_column_status"] == "not_declared"
    assert r["expected_call_columns"] == []


def test_the_uns_record_says_why_there_is_no_call_column(tmp_path):
    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n"
        "AAAC-1,0.10,True\nAAAG-1,0.20,False\n"
        "AAAT-1,0.30,False\nAAAA-1,0.40,False\n",
    )
    with pytest.warns(UserWarning):
        pyscx.doublet_import(str(path), str(table), tool="doubletdetection")
    rec = pyscx.open(str(path)).read_uns()["doubletdetection"]
    assert rec["call_column_status"] == "declared_but_absent"
    assert rec["expected_call_columns"] == ["doublet_label"]


def test_call_column_recovers_the_call_and_clears_the_status(tmp_path):
    """The documented remedy must actually work and must stop warning."""
    import warnings

    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n"
        "AAAC-1,0.10,True\nAAAG-1,0.20,False\n"
        "AAAT-1,0.30,False\nAAAA-1,0.40,False\n",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = pyscx.doublet_import(
            str(path), str(table), tool="doubletdetection",
            call_column="predicted_doublet",
        )
    assert r["call_column_status"] == "resolved"
    assert "doubletdetection_predicted" in r["canonical_columns"]


def test_consensus_diagnoses_a_declared_but_unmatched_call_column(tmp_path):
    """The whole B3 loop end to end.

    The two halves are individually plausible and only wrong together: the import
    quietly drops the call, and the consensus then blames the tool's profile. This
    is the test that catches that pair.
    """
    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "dd.csv",
        "barcode,doublet_score,predicted_doublet\n"
        "AAAC-1,0.10,True\nAAAG-1,0.20,False\n"
        "AAAT-1,0.30,False\nAAAA-1,0.40,False\n",
    )
    with pytest.warns(UserWarning):
        pyscx.doublet_import(str(path), str(table), tool="doubletdetection")

    with pytest.raises(ValueError) as e:
        pyscx.doublet_consensus(str(path), keys=["doubletdetection"],
                                method="majority")
    msg = str(e.value)
    # Names the real cause and the real fix...
    assert "doublet_label" in msg
    assert "declared_but_absent" in msg
    # ...and no longer asserts the false one.
    assert "emits no call column" not in msg


def test_consensus_still_says_a_score_only_tool_emits_no_call(tmp_path):
    """The other branch: for scds the old wording was correct, so keep it."""
    path = _fixture(tmp_path)
    table = _write(
        tmp_path,
        "scds.csv",
        "barcode,hybrid_score\nAAAC-1,0.10\nAAAG-1,0.20\n"
        "AAAT-1,0.30\nAAAA-1,0.40\n",
    )
    pyscx.doublet_import(str(path), str(table), tool="scds")
    with pytest.raises(ValueError, match="emits no call column"):
        pyscx.doublet_consensus(str(path), keys=["scds"], method="majority")


def test_doublet_profiles_matches_the_docs_table():
    """The per-tool table in docs/scanpy.md must not drift from the profiles.

    D2: that table not existing anywhere user-facing is what turned a
    call-column name mismatch into a silent score-only import — the only way to
    learn `doubletdetection` wants `doublet_label` was to read the Rust source.
    A hand-maintained table would drift, so pin it.
    """
    import pathlib
    import re

    profiles = pyscx.doublet_profiles()
    assert set(profiles) == set(pyscx.doublet_tools())

    doc = (
        pathlib.Path(__file__).resolve().parents[2] / "docs" / "scanpy.md"
    ).read_text()
    section = doc.split("### The per-tool column table", 1)[1].split("###", 1)[0]
    rows = {
        m.group(1): m.group(0)
        for m in re.finditer(r"^\| `([a-z]+)` \|.*$", section, re.M)
    }
    assert set(rows) == set(profiles), "every tool needs a documented row"

    for tool, spec in profiles.items():
        row = rows[tool]
        for col in spec["score_columns"] + spec["call_columns"]:
            assert col in row, f"{tool}: {col} missing from the docs row"
        for pre in (spec["score_prefix"], spec["call_prefix"]):
            if pre:
                assert pre in row, f"{tool}: prefix {pre} missing from the docs row"
        # The load-bearing column: whether `<K>_predicted` can exist at all.
        # Check the LAST cell specifically — a substring search over the whole
        # row would match the "no" inside scds's "*(none)*" call cell and pass
        # for the wrong reason.
        last_cell = [c.strip() for c in row.strip().strip("|").split("|")][-1]
        expected = "yes" if spec["emits_call"] else "no"
        assert expected in last_cell.lower(), (
            f"{tool}: the emits-a-call cell is {last_cell!r}, must say {expected!r}"
        )
