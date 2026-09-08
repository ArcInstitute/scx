"""Tests for `pyscx.var_import` — a var annotation table, or an h5ad's `/var`.

The var-axis twin of `test_obs_import.py`. Two things are worth pinning beyond
the obs suite's coverage: `key=None` resolves through the **gene** fallbacks
rather than the barcode ones (the two lists are disjoint, so a shared list
would be visible immediately), and the source-file plumbing — column selection,
rename, prefix, delimiter sniffing — reaches the var path at all.
"""

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


def _fixture(tmp_path, name="t.scx", var=None, n_obs=3):
    if var is None:
        # `var_names` holds the Ensembl ids, as a real file does, with the
        # symbols in a column. That matters for the `key=None` test below: auto
        # resolution runs on each side independently, so the two sides must be
        # naming the same identifier for it to be a test of the *fallback list*
        # rather than of a cross-identifier mismatch.
        var = pd.DataFrame(
            {"gene_symbol": ["TP53", "MYC", "EGFR"]},
            index=["ENSG1", "ENSG2", "ENSG3"],
        )
    n_vars = len(var)
    X = sparse.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    path = tmp_path / name
    pyscx.from_anndata(anndata.AnnData(X=X, obs=obs, var=var), str(path))
    return path


def _table(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def _var(path):
    return pyscx.open(str(path)).read_var()


def test_imports_a_csv_joined_on_the_gene_id_fallback(tmp_path):
    scx = _fixture(tmp_path)
    # Deliberately out of order, and carrying a `barcode` column: if the var
    # axis resolved through the OBS fallback list it would key on that and
    # every gene would miss.
    csv = _table(
        tmp_path,
        "ann.csv",
        "barcode,gene_id,peak_score\nnot_a_key,ENSG3,0.3\nalso_not,ENSG1,0.1\n",
    )
    r = pyscx.var_import(str(scx), str(csv))
    assert r["source_key_columns"] == ["gene_id"], (
        "the SOURCE side must resolve through the gene fallbacks, not the "
        "barcode ones"
    )
    assert r["var_key_column"] == "var_names"
    assert (r["n_matched"], r["n_target_rows_absent"]) == (2, 1)
    assert r["n_rows_in_source"] == 2
    assert r["format"] == "table"
    assert r["delimiter"] == ","

    got = _var(scx)
    assert got.loc["ENSG1", "peak_score"] == pytest.approx(0.1)
    assert got.loc["ENSG3", "peak_score"] == pytest.approx(0.3)
    assert pd.isna(got.loc["ENSG2", "peak_score"])


def test_var_names_keys_on_the_var_index(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "var_names,score\nENSG2,2.0\nENSG1,1.0\n")
    r = pyscx.var_import(str(scx), str(csv), key="var_names")
    assert r["n_matched"] == 2
    assert r["var_key_column"] == "var_names"
    got = _var(scx)
    assert got.loc["ENSG1", "score"] == pytest.approx(1.0)
    assert got.loc["ENSG2", "score"] == pytest.approx(2.0)


def test_source_key_pairs_positionally_with_key(tmp_path):
    scx = _fixture(tmp_path)
    # The table spells the key differently on its own side.
    csv = _table(tmp_path, "a.csv", "symbol,score\nMYC,2.0\nTP53,1.0\n")
    r = pyscx.var_import(
        str(scx), str(csv), key="gene_symbol", source_key="symbol"
    )
    assert r["n_matched"] == 2
    assert r["source_key_columns"] == ["symbol"]
    got = _var(scx)
    assert got.loc["ENSG1", "score"] == pytest.approx(1.0)


def test_source_key_without_key_is_refused_with_the_remedy(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "symbol,score\nTP53,1.0\n")
    with pytest.raises(ValueError) as e:
        pyscx.var_import(str(scx), str(csv), source_key="symbol")
    assert "source_key= needs key=" in str(e.value)
    assert "var_names" in str(e.value)


def test_columns_rename_and_prefix_apply(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(
        tmp_path,
        "a.csv",
        "gene_id,keep,drop_me\nENSG1,1.0,9.0\nENSG2,2.0,9.0\nENSG3,3.0,9.0\n",
    )
    r = pyscx.var_import(
        str(scx),
        str(csv),
        columns=["keep"],
        rename={"keep": "kept"},
        prefix="pk_",
    )
    assert r["var_columns_added"] == ["pk_kept"]
    got = _var(scx)
    assert "pk_kept" in got.columns
    assert "drop_me" not in got.columns


def test_a_tsv_is_sniffed_from_its_extension(tmp_path):
    scx = _fixture(tmp_path)
    tsv = _table(tmp_path, "a.tsv", "gene_id\tscore\nENSG1\t1.0\n")
    r = pyscx.var_import(str(scx), str(tsv))
    assert r["delimiter"] == "\t"
    assert r["n_matched"] == 1


def test_a_status_column_marks_uncovered_genes_absent(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "gene_id,score\nENSG1,1.0\n")
    pyscx.var_import(str(scx), str(csv), status_column="ann_status")
    got = _var(scx)
    assert list(got["ann_status"]) == ["present", "absent", "absent"]


def test_zero_overlap_names_the_columns_that_would_work(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "gene_id,score\nNOPE,1.0\n")
    with pytest.raises(ValueError) as e:
        pyscx.var_import(str(scx), str(csv))
    msg = str(e.value)
    assert "no target row key matched" in msg
    assert "var columns that ARE unique" in msg, msg


def test_uns_keys_are_refused_for_a_delimited_table(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "gene_id,score\nENSG1,1.0\n")
    with pytest.raises(ValueError, match="carries no uns"):
        pyscx.var_import(str(scx), str(csv), uns_keys=["peaks"])


def test_dry_run_leaves_the_file_byte_identical(tmp_path):
    scx = _fixture(tmp_path)
    before = scx.read_bytes()
    csv = _table(tmp_path, "a.csv", "gene_id,score\nENSG1,1.0\n")
    r = pyscx.var_import(str(scx), str(csv), dry_run=True)
    assert r["n_matched"] == 1
    assert r["dry_run"] is True
    assert "key_diagnosis" in r
    assert scx.read_bytes() == before


def test_rollback_undoes_the_import(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "gene_id,score\nENSG1,1.0\n")
    pyscx.var_import(str(scx), str(csv))
    assert "score" in _var(scx).columns
    pyscx.rollback(str(scx))
    assert "score" not in _var(scx).columns


def test_a_second_import_errors_and_overwrite_replaces(tmp_path):
    scx = _fixture(tmp_path)
    a = _table(tmp_path, "a.csv", "gene_id,score\nENSG1,1.0\n")
    b = _table(tmp_path, "b.csv", "gene_id,score\nENSG2,2.0\n")
    pyscx.var_import(str(scx), str(a))
    with pytest.raises(ValueError, match="overwrite=true"):
        pyscx.var_import(str(scx), str(b))
    pyscx.var_import(str(scx), str(b), overwrite=True)
    got = _var(scx)["score"]
    assert pd.isna(got.loc["ENSG1"]), "overwrite replaces, it does not merge"
    assert got.loc["ENSG2"] == pytest.approx(2.0)


def test_the_experiment_handle_form_is_accepted(tmp_path):
    scx = _fixture(tmp_path)
    csv = _table(tmp_path, "a.csv", "gene_id,score\nENSG1,1.0\n")
    exp = pyscx.open(str(scx))
    pyscx.var_import(exp, str(csv))
    assert "score" in exp.read_var().columns


def test_an_h5mu_source_is_refused_naming_the_extraction_route(tmp_path):
    scx = _fixture(tmp_path)
    fake = _table(tmp_path, "x.h5mu", "not hdf5")
    with pytest.raises(ValueError) as e:
        pyscx.var_import(str(scx), str(fake))
    msg = str(e.value)
    assert "/mod/<modality>/var" in msg
    assert "subset --modality" in msg


@pytest.mark.skipif(not pyscx._HAS_HDF5, reason="needs an hdf5 build")
def test_imports_var_from_an_h5ad_written_by_anndata(tmp_path):
    scx = _fixture(tmp_path)
    # A real anndata file, so the `/var` layout is the one on disk in the wild
    # rather than one assembled here.
    src = anndata.AnnData(
        X=sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(
            {
                "gene_id": ["ENSG3", "ENSG1", "ENSG2"],
                "symbol_norm": ["EGFR_n", "TP53_n", "MYC_n"],
            },
            index=["EGFR", "TP53", "MYC"],
        ),
    )
    h5 = tmp_path / "src.h5ad"
    src.write_h5ad(h5)

    r = pyscx.var_import(str(scx), str(h5), key="var_names", source_key="gene_id")
    assert r["format"] == "h5ad"
    assert r["delimiter"] is None
    assert r["n_matched"] == 3
    got = _var(scx)
    assert got.loc["ENSG1", "symbol_norm"] == "TP53_n"
    assert got.loc["ENSG2", "symbol_norm"] == "MYC_n"
    assert got.loc["ENSG3", "symbol_norm"] == "EGFR_n"
    assert "__index_level_0__" not in got.columns, (
        "the source h5ad's own index must not land as a column"
    )


@pytest.mark.skipif(not pyscx._HAS_HDF5, reason="needs an hdf5 build")
def test_an_h5ad_var_import_can_carry_uns_keys(tmp_path):
    scx = _fixture(tmp_path)
    src = anndata.AnnData(
        X=sparse.csr_matrix(np.ones((2, 3), dtype=np.float32)),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(
            {
                "gene_id": ["ENSG1", "ENSG2", "ENSG3"],
                "peak_width": [100, 200, 300],
            },
            index=["a", "b", "c"],
        ),
        uns={"peak_params": {"width": 500}},
    )
    h5 = tmp_path / "src.h5ad"
    src.write_h5ad(h5)

    r = pyscx.var_import(
        str(scx), str(h5), key="var_names", source_key="gene_id",
        uns_keys=["peak_params"]
    )
    assert r["uns_keys_imported"] == ["peak_params"]
    assert r["var_columns_added"] == ["peak_width"]
    assert pyscx.open(str(scx)).read_uns()["peak_params"] == {"width": 500}
    assert list(_var(scx)["peak_width"]) == [100, 200, 300]
