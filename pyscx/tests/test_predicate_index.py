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
# Non-string categorical value types
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


def test_boolean_categorical_survives_append_onto_a_sharded_file(tmp_dir):
    """The mixed dictionary/plain shape `append` creates.

    `append` raw-copies the base `Dictionary(_, Boolean)` shards while the
    appended obs shard lands as plain `Boolean`, so the assembler has to
    reconcile the two — and arrow cannot pack a `Boolean` into a dictionary.
    Fixing the key-widening and dedup paths alone left this third site
    raising the same error on `read_obs()` after an ordinary append, which is
    also what `to_anndata()` and a filtered collect go through.
    """
    import pandas as pd

    adata = _adata_with_typed_categoricals()
    base, addition = tmp_dir / "ap_base.scx", tmp_dir / "ap_add.scx"
    pyscx.from_anndata(adata, str(base), shard_size=100)

    second = _adata_with_typed_categoricals()
    second.obs.index = [f"cell_{i + adata.n_obs}" for i in range(second.n_obs)]
    pyscx.from_anndata(second, str(addition), shard_size=100)

    assert len(pyscx.open(str(base)).read_obs()) == adata.n_obs
    pyscx.append(str(base), str(addition))

    obs = pyscx.open(str(base)).read_obs()
    assert len(obs) == adata.n_obs + second.n_obs
    # The column must still be a categorical, not silently degraded to plain
    # bool by the append — a dtype that changes under append is its own bug.
    assert isinstance(obs["flag"].dtype, pd.CategoricalDtype)
    expected = list(adata.obs["flag"].astype(bool)) + list(second.obs["flag"].astype(bool))
    assert list(obs["flag"].astype(bool)) == expected


@pytest.mark.parametrize(
    "levels, ordered",
    [([True], False), ([True], True), ([True, False], False)],
)
def test_append_does_not_invent_a_boolean_category(tmp_dir, levels, ordered):
    """Append must not widen a boolean categorical's declared vocabulary.

    Encoding the plain (appended) shard as a dictionary is what makes the
    mixed representation readable at all, but installing ``[False, True]``
    unconditionally *invents* a category: a column declaring only ``[True]``
    came back as ``CategoricalDtype(categories=[True, False])``. That is
    observable in ``.cat.categories``, in dtype equality, in
    ``groupby(observed=False)``, and in any categorical encoder — while the
    row values, which an earlier version of this test checked alone, are
    untouched.
    """
    import anndata
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    def make(n, offset):
        flag = pd.Categorical([levels[0]] * n, categories=levels, ordered=ordered)
        obs = pd.DataFrame({"flag": flag}, index=[f"c{offset + i}" for i in range(n)])
        X = sp.random(n, 5, density=0.2, format="csr", dtype="float32", random_state=0)
        return anndata.AnnData(X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(5)]))

    base, addition = tmp_dir / "inv_base.scx", tmp_dir / "inv_add.scx"
    pyscx.from_anndata(make(200, 0), str(base), shard_size=50)
    pyscx.from_anndata(make(100, 200), str(addition), shard_size=50)

    before = pyscx.open(str(base)).read_obs()["flag"]
    assert list(before.cat.categories) == levels
    pyscx.append(str(base), str(addition))
    after = pyscx.open(str(base)).read_obs()["flag"]

    assert list(after.cat.categories) == levels, (
        f"append widened the declared vocabulary {levels} -> "
        f"{list(after.cat.categories)}"
    )
    assert after.cat.ordered == ordered
    assert len(after) == 300
    assert bool((after.astype(bool) == levels[0]).all())
