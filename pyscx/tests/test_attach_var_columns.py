"""Tests for `pyscx.attach_var_columns` and `pyscx.diagnose_var_key`.

The var-axis twin of `test_attach_obs_columns.py`, on the same
`attach_external_var` seam: key-joined by default, with an explicit
`positional=True` mode for frames computed in-process from the file's own
`read_var()` output.

The join tests deliberately scramble the frame's row order relative to the
target: a positional assumption behind a key join would land every value on the
wrong gene while still producing a correctly-shaped column.
"""

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


def _fixture(tmp_path, name="t.scx", var=None, index_var=None, n_obs=4):
    """A small SCX file. `var` defaults to four gene ids plus a symbol column."""
    if var is None:
        var = pd.DataFrame(
            {"gene_symbol": ["TP53", "MYC", "EGFR", "KRAS"]},
            index=[f"ENSG{i}" for i in range(4)],
        )
    n_vars = len(var)
    X = sparse.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    path = tmp_path / name
    pyscx.from_anndata(
        anndata.AnnData(X=X, obs=obs, var=var), str(path), index_var=index_var
    )
    return path


def _var(path):
    return pyscx.open(str(path)).read_var()


# ---------------------------------------------------------------------------
# Key join
# ---------------------------------------------------------------------------


def test_joins_by_df_index_by_default(tmp_path):
    scx = _fixture(tmp_path)
    # Out of order, covering three of four genes: a positional attach would
    # land every score on the wrong gene.
    df = pd.DataFrame(
        {"peak_score": [0.3, 0.1, 0.2]}, index=["ENSG2", "ENSG0", "ENSG1"]
    )
    r = pyscx.attach_var_columns(str(scx), df)
    assert (r["n_matched"], r["n_target_rows_absent"]) == (3, 1)
    assert r["var_key_column"] == "var_names"

    got = _var(scx)
    assert got.loc["ENSG0", "peak_score"] == pytest.approx(0.1)
    assert got.loc["ENSG1", "peak_score"] == pytest.approx(0.2)
    assert got.loc["ENSG2", "peak_score"] == pytest.approx(0.3)
    assert pd.isna(got.loc["ENSG3", "peak_score"]), (
        "a gene the frame does not cover must be null, never a fabricated 0.0"
    )
    assert list(got["gene_symbol"]) == ["TP53", "MYC", "EGFR", "KRAS"], (
        "the pre-existing var columns must survive untouched"
    )


def test_a_named_column_key_joins_on_the_same_name_on_both_sides(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame(
        {"gene_symbol": ["KRAS", "TP53"], "is_driver": [True, True]}
    )
    r = pyscx.attach_var_columns(str(scx), df, key="gene_symbol")
    assert r["n_matched"] == 2
    got = _var(scx)
    assert bool(got.loc["ENSG0", "is_driver"]) is True
    assert bool(got.loc["ENSG3", "is_driver"]) is True
    assert pd.isna(got.loc["ENSG1", "is_driver"])
    assert "gene_symbol" not in r["var_columns_added"], (
        "the key column is consumed by the join, not re-imported"
    )


def test_var_names_names_the_var_index(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=list(_var(scx).index))
    r = pyscx.attach_var_columns(str(scx), df, key="var_names")
    assert r["n_matched"] == 4
    assert r["var_key_column"] == "var_names", (
        "the physical field name must never leak to the caller"
    )


def test_a_composite_key_disambiguates_repeated_symbols(tmp_path):
    var = pd.DataFrame(
        {"gene_symbol": ["DUP", "DUP", "X"]}, index=["e1", "e2", "e3"]
    )
    scx = _fixture(tmp_path, var=var)
    df = pd.DataFrame(
        {
            "gene_symbol": ["DUP", "X", "DUP"],
            "var_names": ["e2", "e3", "e1"],
            "score": [2.0, 3.0, 1.0],
        }
    )
    r = pyscx.attach_var_columns(str(scx), df, key=["gene_symbol", "var_names"])
    assert r["n_matched"] == 3
    got = _var(scx)
    assert list(got["score"]) == [1.0, 2.0, 3.0]


def test_a_duplicated_var_index_names_the_column_that_would_work(tmp_path):
    var = pd.DataFrame({"gene_id": ["e1", "e2"]}, index=["DUP", "DUP"])
    scx = _fixture(tmp_path, var=var)
    df = pd.DataFrame({"score": [1.0, 2.0]}, index=["DUP", "DUP"])
    with pytest.raises(ValueError) as e:
        pyscx.attach_var_columns(str(scx), df)
    msg = str(e.value)
    assert "duplicates" in msg
    assert "gene_id" in msg, f"the diagnosis must name a usable key: {msg}"


# ---------------------------------------------------------------------------
# Positional
# ---------------------------------------------------------------------------


def test_positional_lands_row_for_row(tmp_path):
    scx = _fixture(tmp_path)
    v = _var(scx)
    v["is_hvg"] = [True, False, True, False]
    r = pyscx.attach_var_columns(str(scx), v[["is_hvg"]], positional=True)
    assert r["n_matched"] == 4
    assert r["var_key_column"] == "<positional>"
    assert list(_var(scx)["is_hvg"]) == [True, False, True, False]


def test_positional_and_key_are_mutually_exclusive(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="mutually exclusive"):
        pyscx.attach_var_columns(str(scx), df, key="var_names", positional=True)


def test_a_positional_frame_of_the_wrong_length_names_n_vars(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(ValueError) as e:
        pyscx.attach_var_columns(str(scx), df, positional=True)
    msg = str(e.value)
    assert "n_vars = 4" in msg and "3 rows" in msg, msg


def test_a_positional_frame_reordered_after_read_var_is_refused(tmp_path):
    scx = _fixture(tmp_path)
    v = _var(scx)
    v["score"] = [1.0, 2.0, 3.0, 4.0]
    # Right length, right labels, wrong order — every value on the wrong gene.
    shuffled = v.sort_values("gene_symbol")
    with pytest.raises(ValueError) as e:
        pyscx.attach_var_columns(str(scx), shuffled[["score"]], positional=True)
    assert "different order" in str(e.value)
    assert "score" not in _var(scx).columns, "and nothing was written"


def test_a_status_column_is_refused_under_positional(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3, 4]})
    with pytest.raises(ValueError, match="status_column is meaningless"):
        pyscx.attach_var_columns(str(scx), df, positional=True, status_column="s")


# ---------------------------------------------------------------------------
# Dtypes and categoricals
# ---------------------------------------------------------------------------


def test_a_categorical_column_survives_with_its_order_and_unused_levels(tmp_path):
    scx = _fixture(tmp_path)
    cats = pd.Categorical(
        ["promoter", "enhancer", "promoter", "enhancer"],
        categories=["promoter", "enhancer", "intergenic"],
        ordered=True,
    )
    df = pd.DataFrame({"peak_class": cats}, index=list(_var(scx).index))
    pyscx.attach_var_columns(str(scx), df)

    col = _var(scx)["peak_class"]
    assert isinstance(col.dtype, pd.CategoricalDtype), col.dtype
    assert list(col.cat.categories) == ["promoter", "enhancer", "intergenic"], (
        "declared order and the unused level must both survive"
    )
    assert col.cat.ordered is True
    assert list(col) == ["promoter", "enhancer", "promoter", "enhancer"]


def test_to_anndata_sees_the_new_columns(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=list(_var(scx).index))
    pyscx.attach_var_columns(str(scx), df)
    ad = pyscx.open(str(scx)).to_anndata()
    assert list(ad.var["score"]) == [1.0, 2.0, 3.0, 4.0]
    assert ad.shape == (4, 4)


# ---------------------------------------------------------------------------
# Guards, uns, rollback, handles
# ---------------------------------------------------------------------------


def test_overwrite_replaces_and_does_not_merge(tmp_path):
    scx = _fixture(tmp_path)
    idx = list(_var(scx).index)
    first = pd.DataFrame({"score": [1.0, 1.0]}, index=idx[:2])
    pyscx.attach_var_columns(str(scx), first)
    second = pd.DataFrame({"score": [2.0, 2.0]}, index=idx[2:])
    with pytest.raises(ValueError, match="overwrite=true"):
        pyscx.attach_var_columns(str(scx), second)
    pyscx.attach_var_columns(str(scx), second, overwrite=True)
    got = _var(scx)["score"]
    assert pd.isna(got.iloc[0]) and pd.isna(got.iloc[1]), (
        "overwrite REPLACES; concatenate-first is the contract"
    )
    assert list(got.iloc[2:]) == [2.0, 2.0]


def test_uns_rides_in_the_same_commit_and_rollback_undoes_both(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=list(_var(scx).index))
    pyscx.attach_var_columns(str(scx), df, uns={"peaks": {"n": 4}})
    assert pyscx.open(str(scx)).read_uns()["peaks"] == {"n": 4}

    pyscx.rollback(str(scx))
    assert "score" not in _var(scx).columns
    # The fixture had no uns section at all, so rolling back the attach
    # restores its absence rather than an empty dict.
    assert pyscx.open(str(scx)).read_uns() is None


def test_uns_key_nests_and_needs_a_payload(tmp_path):
    scx = _fixture(tmp_path)
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=list(_var(scx).index))
    pyscx.attach_var_columns(str(scx), df, uns=[1, 2, 3], uns_key="k")
    assert pyscx.open(str(scx)).read_uns()["k"] == [1, 2, 3]
    with pytest.raises(ValueError, match="needs a uns= payload"):
        pyscx.attach_var_columns(str(scx), df, uns_key="k2", overwrite=True)


def test_a_dict_is_refused_naming_the_function(tmp_path):
    scx = _fixture(tmp_path)
    with pytest.raises(TypeError) as e:
        pyscx.attach_var_columns(str(scx), {"score": [1, 2, 3, 4]})
    assert "attach_var_columns" in str(e.value)


def test_an_empty_frame_is_refused(tmp_path):
    scx = _fixture(tmp_path)
    with pytest.raises(ValueError, match="no rows"):
        pyscx.attach_var_columns(str(scx), pd.DataFrame({"a": []}))


def test_dry_run_writes_nothing_and_reports_the_match_count(tmp_path):
    scx = _fixture(tmp_path)
    before = scx.read_bytes()
    df = pd.DataFrame({"score": [1.0, 2.0]}, index=list(_var(scx).index)[:2])
    r = pyscx.attach_var_columns(str(scx), df, dry_run=True)
    assert (r["n_matched"], r["n_target_rows_absent"]) == (2, 2)
    assert "key_diagnosis" in r
    assert scx.read_bytes() == before


def test_an_experiment_handle_is_accepted_and_reloaded(tmp_path):
    scx = _fixture(tmp_path)
    exp = pyscx.open(str(scx))
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=list(exp.read_var().index))
    pyscx.attach_var_columns(exp, df)
    # The handle must be usable afterwards — the wrapper reloads it rather than
    # leaving it pointing at a catalog that is no longer current.
    assert "score" in exp.read_var().columns


def test_a_multimodal_target_is_refused(tmp_path):
    mudata = pytest.importorskip("mudata")
    rna = anndata.AnnData(
        X=sparse.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    atac = anndata.AnnData(
        X=sparse.csr_matrix(np.ones((3, 2), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1", "c2"]),
        var=pd.DataFrame(index=["p0", "p1"]),
    )
    path = tmp_path / "mm.scx"
    pyscx.from_mudata(mudata.MuData({"rna": rna, "atac": atac}), str(path))
    df = pd.DataFrame({"score": [1.0, 2.0]}, index=["g0", "g1"])
    with pytest.raises(ValueError) as e:
        pyscx.attach_var_columns(str(path), df)
    assert "multimodal" in str(e.value).lower()


# ---------------------------------------------------------------------------
# The predicate index
# ---------------------------------------------------------------------------


def test_a_pure_add_keeps_the_var_index_and_an_overwrite_rebuilds_it(tmp_path):
    var = pd.DataFrame(
        {"feature_type": ["Gene Expression", "Peaks", "Gene Expression", "Peaks"]},
        index=[f"ENSG{i}" for i in range(4)],
    )
    scx = _fixture(tmp_path, var=var, index_var=["feature_type"])

    add = pd.DataFrame({"score": [1.0, 2.0, 3.0, 4.0]}, index=list(var.index))
    r = pyscx.attach_var_columns(str(scx), add)
    assert r["var_index_rebuilt"] is False, "a pure add keeps the index valid"

    over = pd.DataFrame(
        {"feature_type": ["Antibody Capture"] * 4}, index=list(var.index)
    )
    r = pyscx.attach_var_columns(str(scx), over, overwrite=True)
    assert r["var_index_rebuilt"] is True
    assert r["var_columns_not_carried"] == []
    # And the values really changed on the file.
    assert set(_var(scx)["feature_type"]) == {"Antibody Capture"}


# ---------------------------------------------------------------------------
# diagnose_var_key
# ---------------------------------------------------------------------------


def test_diagnose_var_key_ranks_usable_keys_and_sets_aside_floats(tmp_path):
    var = pd.DataFrame(
        {"gene_id": ["e1", "e2", "e3"], "gc": [0.1, 0.2, 0.3]},
        index=["DUP", "DUP", "X"],
    )
    scx = _fixture(tmp_path, var=var, n_obs=3)
    d = pyscx.diagnose_var_key(str(scx))
    assert d["n_vars"] == 3, "the row-count key must name the var axis"
    assert d["unique_columns"] == ["gene_id"]
    assert d["suggestion"] == "gene_id"
    assert d["unusable_unique_columns"] == ["gc"], (
        "a unique float must be reported apart, never offered as a key"
    )
    assert "var columns that ARE unique" in d["summary"]
