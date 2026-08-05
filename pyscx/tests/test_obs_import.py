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


def test_on_missing_rows_accepts_null_as_well_as_zero(tmp_path, synthetic_adata):
    """The help says uncovered rows are left NULL, so `null` has to be a token
    a user can actually pass. `zero` stays accepted — it is the name the
    policy enum carries and what earlier callers wrote."""
    import pandas as pd

    n = synthetic_adata.n_obs
    for token in ("null", "zero"):
        path = str(tmp_path / f"{token}.scx")
        pyscx.from_anndata(synthetic_adata, path)
        table = tmp_path / f"{token}.csv"
        pd.DataFrame({
            "cell_id": list(synthetic_adata.obs["cell_id"])[: n // 2],
            "score": [0.5] * (n // 2),
        }).to_csv(table, index=False)

        r = pyscx.obs_import(path, str(table), key="cell_id",
                             on_missing_rows=token)
        assert r["n_matched"] == n // 2
        back = pyscx.open(path).read_obs()
        # Uncovered rows are NULL, which is what both spellings mean.
        assert back["score"].isna().sum() == n - n // 2


# ---------------------------------------------------------------------------
# The obs index is addressable, under the name the tooling prints (F1)
# ---------------------------------------------------------------------------


def test_the_obs_index_is_reported_as_obs_names(tmp_path):
    """`__index_level_0__` is pyarrow's serialization name, not a column.

    `read_obs()` hands that field back as the frame's *unnamed index*, so
    following a diagnostic that named it literally — `obs["__index_level_0__"]`
    — is a KeyError. Every field the diagnosis returns has to be a name the
    import accepts.
    """
    obs = pd.DataFrame({"donor": ["d1", "d1", "d2", "d2"]},
                       index=["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"])
    scx = _fixture(tmp_path, obs=obs)

    d = pyscx.diagnose_obs_key(str(scx))
    blob = repr(d)
    assert "__index_level_0__" not in blob, blob
    assert d["resolved_key"] == "obs_names"
    assert d["suggestion"] == "obs_names"
    assert "obs_names" in d["summary"]


def test_obs_names_is_accepted_as_a_key(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv",
                 "barcode,score\nAAAT-1,0.30\nAAAC-1,0.10\n")

    r = pyscx.obs_import(str(scx), str(csv), key="obs_names",
                         source_key="barcode")
    assert r["n_matched"] == 2
    assert r["obs_key_column"] == "obs_names", "reported back in the same vocabulary"

    # File order (AAAC, AAAG, AAAT, AAAA), not CSV order.
    obs = pyscx.open(str(scx)).read_obs()
    assert obs["score"].iloc[0] == pytest.approx(0.10)
    assert obs["score"].iloc[2] == pytest.approx(0.30)
    assert pd.isna(obs["score"].iloc[1])


def test_the_suggested_key_always_joins(tmp_path):
    """The invariant behind F1 and F2: whatever the diagnosis names, works.

    Same property `test_an_integer_key_column_can_be_joined_on` pins for the
    Int64 case — a suggestion the join then refuses is worse than no suggestion.
    """
    scx = _fixture(tmp_path)
    suggestion = pyscx.diagnose_obs_key(str(scx))["suggestion"]
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")
    r = pyscx.obs_import(str(scx), str(csv), key=suggestion, source_key="barcode")
    assert r["n_matched"] == 1


def test_the_physical_index_name_still_works(tmp_path):
    """Back-compat: anything that hard-coded the old spelling keeps running."""
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")
    r = pyscx.obs_import(str(scx), str(csv), key="__index_level_0__",
                         source_key="barcode")
    assert r["n_matched"] == 1


# ---------------------------------------------------------------------------
# A float is never offered as a key (F2)
# ---------------------------------------------------------------------------


def test_a_unique_float_column_is_not_suggested(tmp_path):
    """The shape a file takes after two doublet imports on a merged atlas.

    The obs index repeats and the only per-row-unique columns are float scores —
    which `obs_import` refuses (see `test_a_float_key_column_is_still_refused`),
    so naming one is a guaranteed dead end.
    """
    obs = pd.DataFrame(
        {"scrublet_score": np.array([0.11, 0.22, 0.33, 0.44], dtype=np.float32),
         "donor": ["d1", "d1", "d2", "d2"]},
        index=["DUP"] * 4,
    )
    scx = _fixture(tmp_path, obs=obs)

    d = pyscx.diagnose_obs_key(str(scx))
    assert d["suggestion"] != "scrublet_score"
    assert "scrublet_score" not in d["unique_columns"]
    # Set aside and named, not silently dropped: "nothing is unique" is false.
    assert d["unusable_unique_columns"] == ["scrublet_score"]
    assert "floats are refused" in d["summary"]


# ---------------------------------------------------------------------------
# Per-side key names (F3)
# ---------------------------------------------------------------------------


def _merged_atlas(tmp_path):
    """Two libraries whose barcodes collide; identity is (sample_id, barcode)."""
    obs = pd.DataFrame(
        {"sample_id": ["s1", "s1", "s2", "s2"]},
        index=["AAAC-1", "AAAG-1", "AAAC-1", "AAAG-1"],
    )
    return _fixture(tmp_path, obs=obs)


def test_a_composite_key_can_name_different_columns_per_side(tmp_path):
    """The reported failure: target identity is (sample_id, obs-index), the
    tool's output is keyed (sample_id, barcode). Before `source_key=` this
    needed a rename in pandas first."""
    scx = _merged_atlas(tmp_path)
    csv = _write(
        tmp_path,
        "ml.csv",
        "sample_id,barcode,score\n"
        "s2,AAAG-1,0.4\ns1,AAAC-1,0.1\ns2,AAAC-1,0.3\ns1,AAAG-1,0.2\n",
    )

    r = pyscx.obs_import(str(scx), str(csv),
                         key=["sample_id", "obs_names"],
                         source_key=["sample_id", "barcode"])
    assert r["n_matched"] == 4

    obs = pyscx.open(str(scx)).read_obs()
    # File order, not CSV order — proving the composite joined by key.
    assert obs["score"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_source_key_needs_key(tmp_path):
    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")
    with pytest.raises(ValueError, match="source_key= needs key="):
        pyscx.obs_import(str(scx), str(csv), source_key="barcode")


def test_source_key_must_have_the_same_arity_as_key(tmp_path):
    scx = _merged_atlas(tmp_path)
    csv = _write(tmp_path, "ml.csv", "sample_id,barcode,score\ns1,AAAC-1,0.1\n")
    with pytest.raises(ValueError, match="pair up positionally"):
        pyscx.obs_import(str(scx), str(csv),
                         key=["sample_id", "obs_names"], source_key=["barcode"])


def test_a_single_key_can_differ_across_sides(tmp_path):
    """Not just composites — a named single key could not differ either."""
    obs = pd.DataFrame({"cell_uid": ["u0", "u1", "u2", "u3"]},
                       index=["AAAC-1", "AAAG-1", "AAAT-1", "AAAA-1"])
    scx = _fixture(tmp_path, obs=obs)
    csv = _write(tmp_path, "calls.csv", "tool_cell,score\nu2,0.3\nu0,0.1\n")

    r = pyscx.obs_import(str(scx), str(csv), key="cell_uid",
                         source_key="tool_cell")
    assert r["n_matched"] == 2
    obs_back = pyscx.open(str(scx)).read_obs()
    assert obs_back["score"].tolist()[:3] == pytest.approx([0.1, np.nan, 0.3],
                                                          nan_ok=True)


# ---------------------------------------------------------------------------
# on_missing_rows spelling (F4)
# ---------------------------------------------------------------------------


def test_on_missing_rows_defaults_to_null(tmp_path):
    """The default spells what actually happens. "zero" stays an accepted alias
    — it names the shared policy enum, whose `zero` is literal only on
    `cellbender_import`, where a missing *matrix* row really is zeros."""
    # The default lives on the pyo3 function; the Python wrapper forwards
    # **kwargs, so `inspect.signature` on it would not see the parameter.
    assert 'on_missing_rows="null"' in pyscx._obs_import_native.__text_signature__
    assert 'on_missing_rows="null"' in pyscx._doublet_import_native.__text_signature__
    # ...and cellbender keeps "zero", where a missing matrix row really IS zeros:
    # an unmatched target row gets an all-zero CSR row, not a null.
    assert ('on_missing_rows="zero"'
            in pyscx.cellbender_import.__text_signature__)

    scx = _fixture(tmp_path)
    csv = _write(tmp_path, "calls.csv", "barcode,score\nAAAC-1,0.5\n")
    r = pyscx.obs_import(str(scx), str(csv), on_missing_rows="zero")
    assert r["n_matched"] == 1
    obs = pyscx.open(str(scx)).read_obs()
    assert pd.isna(obs["score"].iloc[1]), "'zero' still means null, not 0.0"


def test_source_key_arity_is_checked_in_both_directions(tmp_path):
    """A longer `source_key` must error, not silently drop the extras.

    Raised in review as a possible truncation-by-zip hazard. The arity check
    already covers it — `source.len() != target.len()` is direction-agnostic —
    but only the shorter case was pinned, so the guard could have been narrowed
    later without a test noticing.
    """
    scx = _merged_atlas(tmp_path)
    csv = _write(tmp_path, "ml.csv",
                 "sample_id,barcode,score\ns1,AAAC-1,0.1\n")
    # source longer than key
    with pytest.raises(ValueError, match="pair up positionally"):
        pyscx.obs_import(str(scx), str(csv), key="sample_id",
                         source_key=["sample_id", "barcode"])
    # source shorter than key
    with pytest.raises(ValueError, match="pair up positionally"):
        pyscx.obs_import(str(scx), str(csv),
                         key=["sample_id", "obs_names"], source_key=["barcode"])


def test_obs_names_resolves_on_the_source_side_too(tmp_path):
    """`obs_names` must work as a *source* key, not just a target key.

    The docs say every name the diagnosis reports is paste-able into `key=`.
    That was only true target-side: the table readers did a literal schema
    lookup, so a tool table whose key is its own unnamed index — written by a
    plain `DataFrame.to_csv()`, which the reader renames to `_index` — could not
    be named without a `source_key=`.
    """
    scx = _fixture(tmp_path)
    # No `barcode` header: the index column is unnamed, exactly what
    # `df.to_csv()` produces for an index-keyed tool output.
    csv = _write(tmp_path, "calls.csv", ",score\nAAAT-1,0.30\nAAAC-1,0.10\n")

    r = pyscx.obs_import(str(scx), str(csv), key="obs_names")
    assert r["n_matched"] == 2
    obs = pyscx.open(str(scx)).read_obs()
    assert obs["score"].iloc[0] == pytest.approx(0.10)
    assert obs["score"].iloc[2] == pytest.approx(0.30)
