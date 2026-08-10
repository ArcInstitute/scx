"""Phase 5a: conversion-time predicate indexes from Python."""

from __future__ import annotations

import pytest

import pyscx


def test_from_anndata_index_obs_writes_predicate_index(synthetic_adata, tmp_dir):
    """`index_obs=[...]` on from_anndata produces an obs predicate
    index section on disk that the read side can decode."""
    out = tmp_dir / "indexed.scx"
    pyscx.from_anndata(synthetic_adata, str(out), index_obs=["batch"])
    # A filter on `batch` should round-trip identically to the
    # unindexed result. (Index correctness is exercised more
    # thoroughly in scx-convert::tests::convert_with_index_obs_writes_predicate_index;
    # this test just confirms the kwarg plumbing reaches the writer.)
    exp = pyscx.open(str(out))
    n_b = int((synthetic_adata.obs["batch"] == "B").sum())
    got = exp.query().filter_obs('batch == "B"').count()
    assert got == n_b


def test_from_anndata_unknown_forced_column_raises(synthetic_adata, tmp_dir):
    out = tmp_dir / "bad.scx"
    with pytest.raises(ValueError, match="nonexistent_column"):
        pyscx.from_anndata(
            synthetic_adata, str(out), index_obs=["nonexistent_column"]
        )


def test_from_anndata_unknown_preset_raises(synthetic_adata, tmp_dir):
    out = tmp_dir / "bad.scx"
    with pytest.raises(ValueError, match="not_a_real_preset"):
        pyscx.from_anndata(
            synthetic_adata, str(out), index_preset="not_a_real_preset"
        )


# ---------------------------------------------------------------------------
# Non-string categorical value types (review §5.6)
#
# pandas writes every Categorical as an Arrow dictionary regardless of what
# the categories are, so `pd.Categorical([1, 2, 3])` arrives as
# `Dictionary(_, Int64)`. That used to be classified as *categorical*, whose
# builder can only read string values out of a dictionary — so the column was
# indexed with zero entries and no warning, and both `batch == 1` and
# `batch == '1'` failed with a type mismatch.
# ---------------------------------------------------------------------------


def _adata_with_typed_categoricals():
    import anndata
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    n = 400
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(np.repeat([1, 2, 3, 4], n // 4)),
            "score": pd.Categorical(np.repeat([1.5, 2.5], n // 2)),
            "flag": pd.Categorical(np.repeat([True, False], n // 2)),
            "cell_type": pd.Categorical(np.repeat(["A", "B"], n // 2)),
        },
        index=[f"cell_{i}" for i in range(n)],
    )
    X = sp.random(n, 20, density=0.2, format="csr", dtype="float32", random_state=0)
    return anndata.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(20)]))


@pytest.mark.parametrize(
    "expr, column, expected_value",
    [
        ("batch == 3", "batch", 3),
        ("batch in [1, 2]", "batch", None),
        ("score > 2.0", "score", None),
        ("flag == true", "flag", True),
    ],
)
def test_non_string_categoricals_are_queryable(
    tmp_dir, expr, column, expected_value
):
    """A numeric- or boolean-valued categorical must be filterable, and
    must return the *right* rows — not zero rows, and not every row."""
    adata = _adata_with_typed_categoricals()
    out = tmp_dir / "typed_cats.scx"
    pyscx.from_anndata(adata, str(out), index_obs=["batch", "cell_type"])

    col = adata.obs[column]
    if expr == "batch in [1, 2]":
        truth = int(col.isin([1, 2]).sum())
    elif expr == "score > 2.0":
        truth = int((col.astype(float) > 2.0).sum())
    else:
        truth = int((col == expected_value).sum())

    assert truth > 0, "fixture must not make the expected count vacuously zero"
    assert truth < adata.n_obs, "fixture must not make every row match"
    assert pyscx.open(str(out)).query().filter_obs(expr).count() == truth


def test_integer_categorical_rejects_a_quoted_literal(tmp_dir):
    """The wrong guess has to say which two types disagree."""
    adata = _adata_with_typed_categoricals()
    out = tmp_dir / "typed_cats_q.scx"
    pyscx.from_anndata(adata, str(out), index_obs=["batch"])
    with pytest.raises(Exception, match="integer.*string|string.*integer"):
        pyscx.open(str(out)).query().filter_obs("batch == '3'").count()


def test_forced_index_on_a_boolean_categorical_errors(tmp_dir):
    """A boolean categorical is a dictionary-encoded Boolean and no more
    indexable than the plain column. It used to write an empty index and
    report success."""
    adata = _adata_with_typed_categoricals()
    with pytest.raises(ValueError, match="flag"):
        pyscx.from_anndata(adata, str(tmp_dir / "boolidx.scx"), index_obs=["flag"])


def test_boolean_categorical_survives_a_row_sharded_write(tmp_dir):
    """A `Dictionary(_, Boolean)` obs column used to make `read_obs()` fail
    outright on a row-sharded file — arrow cannot re-encode a Boolean array
    into a dictionary, and the sharded-metadata assembler round-tripped
    through exactly that. The write side never complained."""
    adata = _adata_with_typed_categoricals()
    out = tmp_dir / "bool_sharded.scx"
    pyscx.from_anndata(adata, str(out), shard_size=100)
    obs = pyscx.open(str(out)).read_obs()
    assert len(obs) == adata.n_obs
    assert list(obs["flag"].astype(bool)) == list(adata.obs["flag"].astype(bool))
