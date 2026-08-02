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


def test_h5ad_source_errors_with_the_convert_hint(tmp_path):
    scx = _fixture(tmp_path)
    h5 = _write(tmp_path, "scrublet_out.h5ad", "not really hdf5")
    with pytest.raises(ValueError) as e:
        pyscx.doublet_import(str(scx), str(h5), tool="scrublet")
    assert "to_csv" in str(e.value)


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
