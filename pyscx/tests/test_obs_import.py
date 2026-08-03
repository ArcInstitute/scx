"""Tests for `pyscx.obs_import` and `pyscx.diagnose_obs_key`.

The import's only real risk is the join. A doublet caller run per library hands
back rows in its own order, so these fixtures deliberately scramble the table's
row order relative to the SCX target: a positional import would put every score
on the wrong cell and still produce a correctly-shaped column.

The second risk is the key itself. On a merged atlas the obvious candidate is
often not unique, so the failure has to name a column that *would* work rather
than leaving the caller to guess.
"""

import pathlib

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


def _fixture(tmp_path, name="t.scx", obs=None):
    """A small SCX file. `obs` defaults to four unique barcodes."""
    if obs is None:
        bc = ["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"]
        obs = pd.DataFrame(index=bc)
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


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_joins_by_key_not_position(tmp_path):
    scx = _fixture(tmp_path)
    # Out of order, and covering only three of four cells.
    csv = _write(
        tmp_path,
        "calls.csv",
        "barcode,score,call\nAAAT-1,0.30,singlet\nAAAC-1,0.10,singlet\nAAAG-1,0.90,doublet\n",
    )

    r = pyscx.obs_import(str(scx), str(csv), status_column="dbl_status")
    assert r["n_obs"] == 4
    assert r["n_matched"] == 3
    assert r["n_target_rows_absent"] == 1
    assert r["n_rows_in_source"] == 3
    assert r["delimiter"] == ","
    assert not r["dry_run"]

    obs = pyscx.open(str(scx)).read_obs()
    # File order, not CSV order.
    assert obs["score"].tolist()[:3] == [0.10, 0.90, 0.30]
    assert pd.isna(obs["score"].iloc[3]), "the uncovered cell must be null, not 0.0"
    assert obs["dbl_status"].tolist() == ["present", "present", "present", "absent"]


def test_accepts_pathlib_and_an_experiment_handle(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")

    # pathlib.Path for both arguments.
    r = pyscx.obs_import(pathlib.Path(scx), pathlib.Path(csv), dry_run=True)
    assert r["n_matched"] == 1

    # An open Experiment for the target.
    exp = pyscx.open(str(scx))
    r = pyscx.obs_import(exp, str(csv), dry_run=True)
    assert r["n_matched"] == 1


def test_composite_key_disambiguates_repeated_barcodes(tmp_path):
    # A genuine multi-library merge: the barcode is its own obs column and
    # repeats across libraries, so only sample_id+barcode is unique.
    bc = ["AAAC-1", "AAAG-1", "AAAC-1", "AAAG-1"]
    obs = pd.DataFrame(
        {"sample_id": ["A", "A", "B", "B"], "barcode": bc},
        index=[f"{s}_{b}" for s, b in zip(["A", "A", "B", "B"], bc)],
    )
    scx = _fixture(tmp_path, obs=obs)
    csv = _write(
        tmp_path,
        "calls.csv",
        "sample_id,barcode,score\nA,AAAC-1,1\nA,AAAG-1,2\nB,AAAC-1,3\nB,AAAG-1,4\n",
    )

    # The bare index is duplicated, so the single-column path must refuse.
    with pytest.raises(ValueError):
        pyscx.obs_import(str(scx), str(csv), key="barcode")

    # A list builds the composite, and both sides are fused identically.
    r = pyscx.obs_import(str(scx), str(csv), key=["sample_id", "barcode"])
    assert r["n_matched"] == 4
    assert r["obs_key_column"] == "sample_id,barcode"
    obs_back = pyscx.open(str(scx)).read_obs()
    assert obs_back["score"].tolist() == [1, 2, 3, 4]


def test_column_selection_rename_and_prefix(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "calls.csv",
        "barcode,score,call,extra\nAAAC-1,0.5,singlet,9\n",
    )
    r = pyscx.obs_import(
        str(scx),
        str(csv),
        columns=["score", "call"],
        rename={"score": "doublet_score"},
        prefix="scdbl_",
    )
    assert r["obs_columns_added"] == ["scdbl_doublet_score", "scdbl_call"]
    obs = pyscx.open(str(scx)).read_obs()
    assert "scdbl_doublet_score" in obs.columns
    assert "scdbl_extra" not in obs.columns


# ---------------------------------------------------------------------------
# Real-world file shapes
# ---------------------------------------------------------------------------


def test_pandas_obs_to_csv_round_trips(tmp_path):
    """The shape `adata.obs.to_csv()` produces: a leading unnamed index column."""
    scx = _fixture(tmp_path)
    obs = pd.DataFrame(
        {"doublet_score": [0.03, 0.71], "predicted_doublet": [False, True]},
        index=["AAAC-1", "AAAG-1"],
    )
    csv = tmp_path / "scrublet.csv"
    obs.to_csv(csv)

    r = pyscx.obs_import(str(scx), str(csv))
    assert r["renamed_index_column"], "the unnamed index column must become the key"
    assert r["n_matched"] == 2
    back = pyscx.open(str(scx)).read_obs()
    assert back["doublet_score"].tolist()[:2] == [0.03, 0.71]


def test_r_style_na_keeps_the_score_numeric(tmp_path):
    """R's `write.csv` writes NA; the score must stay numeric, not become str."""
    scx = _fixture(tmp_path)
    csv = _write(
        tmp_path,
        "sce.csv",
        '"","scDblFinder.score","scDblFinder.mostLikelyOrigin"\n'
        '"AAAC-1",0.02,NA\n'
        '"AAAG-1",0.98,"1+2"\n',
    )
    pyscx.obs_import(str(scx), str(csv))
    obs = pyscx.open(str(scx)).read_obs()
    assert obs["scDblFinder.score"].dtype.kind == "f", obs["scDblFinder.score"].dtype
    assert obs["scDblFinder.score"].tolist()[:2] == [0.02, 0.98]


def test_tsv_is_detected_from_the_extension(tmp_path):
    scx = _fixture(tmp_path)
    tsv = _write(tmp_path, "calls.tsv", "barcode\tscore\nAAAC-1\t0.5\n")
    r = pyscx.obs_import(str(scx), str(tsv))
    assert r["delimiter"] == "\t"
    assert r["n_matched"] == 1


# ---------------------------------------------------------------------------
# Errors surface as clean Python exceptions, never panics
# ---------------------------------------------------------------------------


def test_dry_run_writes_nothing_and_reports_the_real_count(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\nAAAG-1,0.6\n")
    before = pathlib.Path(scx).read_bytes()

    r = pyscx.obs_import(str(scx), str(csv), dry_run=True)
    assert r["dry_run"]
    assert r["n_matched"] == 2
    assert "key_diagnosis" in r
    assert pathlib.Path(scx).read_bytes() == before
    assert "score" not in pyscx.open(str(scx)).read_obs().columns


def test_second_import_errors_and_overwrite_replaces(tmp_path):
    scx = _fixture(tmp_path)
    a = _write(tmp_path, "a.csv", "barcode,score\nAAAC-1,1\nAAAG-1,2\n")
    b = _write(tmp_path, "b.csv", "barcode,score\nAAAT-1,3\nAAAA-1,4\n")

    pyscx.obs_import(str(scx), str(a))
    with pytest.raises(ValueError, match="already exists"):
        pyscx.obs_import(str(scx), str(b))

    pyscx.obs_import(str(scx), str(b), overwrite=True)
    obs = pyscx.open(str(scx)).read_obs()
    # Batch A's values are gone: overwrite REPLACES. Importing per-batch tables
    # one after another keeps only the last -- concatenate and import once.
    assert pd.isna(obs["score"].iloc[0]) and pd.isna(obs["score"].iloc[1])
    assert obs["score"].tolist()[2:] == [3, 4]


def test_zero_overlap_raises_naming_both_sides(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nWRONG-1,0.5\n")
    with pytest.raises(ValueError) as e:
        pyscx.obs_import(str(scx), str(csv))
    msg = str(e.value)
    assert "AAAC-1" in msg and "WRONG-1" in msg, msg


def test_duplicated_key_names_a_column_that_would_work(tmp_path):
    """The Phase-0 atlas shape: the obvious key is duplicated, one column isn't."""
    obs = pd.DataFrame(
        {"cell_uid": ["u0", "u1", "u2", "u3"]},
        index=["dup", "dup", "x", "y"],
    )
    scx = _fixture(tmp_path, obs=obs)
    csv = _write(tmp_path, "calls.csv", "barcode,score\ndup,0.5\n")

    with pytest.raises(ValueError) as e:
        pyscx.obs_import(str(scx), str(csv))
    assert "cell_uid" in str(e.value), str(e.value)


def test_missing_key_column_and_bad_enums_raise_value_error(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")

    with pytest.raises(ValueError, match="nope"):
        pyscx.obs_import(str(scx), str(csv), key="nope")
    with pytest.raises(ValueError, match="on_missing_rows"):
        pyscx.obs_import(str(scx), str(csv), on_missing_rows="bogus")
    with pytest.raises(ValueError, match="on_extra_rows"):
        pyscx.obs_import(str(scx), str(csv), on_extra_rows="bogus")
    with pytest.raises(ValueError, match="delimiter"):
        pyscx.obs_import(str(scx), str(csv), delimiter=";;")


def test_a_missing_table_raises_file_not_found(tmp_path):
    scx = _fixture(tmp_path)
    with pytest.raises(FileNotFoundError):
        pyscx.obs_import(str(scx), str(tmp_path / "nope.csv"))


def test_rollback_undoes_the_import(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")
    pyscx.obs_import(str(scx), str(csv))
    assert "score" in pyscx.open(str(scx)).read_obs().columns

    pyscx.rollback(str(scx))
    assert "score" not in pyscx.open(str(scx)).read_obs().columns


# ---------------------------------------------------------------------------
# Key diagnosis
# ---------------------------------------------------------------------------


def test_diagnose_obs_key_reports_unique_columns(tmp_path):
    obs = pd.DataFrame(
        {"cell_uid": ["u0", "u1", "u2", "u3"], "batch": ["a", "a", "b", "b"]},
        index=["dup", "dup", "x", "y"],
    )
    scx = _fixture(tmp_path, obs=obs)

    d = pyscx.diagnose_obs_key(str(scx))
    assert d["n_obs"] == 4
    assert "cell_uid" in d["unique_columns"]
    assert d["suggestion"] == "cell_uid"
    assert "cell_uid" in d["summary"]


def test_an_integer_key_column_can_be_joined_on(tmp_path, synthetic_adata):
    """`diagnose_obs_key` suggests the only unique column; the join must accept it.

    On the merged CELLxGENE-derived atlas this feature exists for, the obs
    index is duplicated and `soma_joinid` — an Int64 — is the *only* unique
    column. The diagnosis names it, so refusing it as "not a string column"
    sent users down a dead end with no alternative key on that file. Both
    sides fuse through the same cast, so an integer key joins exactly.
    """
    import numpy as np
    import pandas as pd

    import anndata as ad
    import scipy.sparse as sp

    # Built explicitly rather than from the shared fixture, which already has a
    # unique `cell_id` — the point here is a file where the ONLY unique column
    # is the integer one, which is the measured census_1m shape.
    n = 6
    obs = pd.DataFrame(
        {"soma_joinid": np.arange(n, dtype=np.int64),
         "donor": ["d1"] * n},
        index=["DUP"] * n,
    )
    X = sp.csr_matrix(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    path = str(tmp_path / "atlas.scx")
    pyscx.from_anndata(
        ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=list("abc"))), path
    )

    diag = pyscx.diagnose_obs_key(path)
    assert diag["unique_columns"] == ["soma_joinid"]
    assert diag["suggestion"] == "soma_joinid"

    table = tmp_path / "calls.csv"
    pd.DataFrame({
        "soma_joinid": np.arange(n, dtype=np.int64),
        "score": np.linspace(0, 1, n),
    }).to_csv(table, index=False)

    # The key the diagnosis pointed at must actually work.
    r = pyscx.obs_import(path, str(table), key="soma_joinid")
    assert r["n_matched"] == n

    back = pyscx.open(path).read_obs()
    assert np.allclose(back["score"].astype(float), np.linspace(0, 1, n))


def test_a_float_key_column_is_still_refused(tmp_path, synthetic_adata):
    """Integers are exact through the text form both sides fuse on; floats are
    not guaranteed to be, so a float key could silently half-match."""
    import numpy as np
    import pandas as pd

    n = synthetic_adata.n_obs
    adata = synthetic_adata.copy()
    adata.obs["ratio"] = np.linspace(0.1, 0.9, n).astype(np.float64)
    path = str(tmp_path / "f.scx")
    pyscx.from_anndata(adata, path)

    table = tmp_path / "t.csv"
    pd.DataFrame({"ratio": np.linspace(0.1, 0.9, n), "score": np.zeros(n)}).to_csv(
        table, index=False
    )
    with pytest.raises(ValueError, match="cannot be a join key|floats are refused"):
        pyscx.obs_import(path, str(table), key="ratio")
