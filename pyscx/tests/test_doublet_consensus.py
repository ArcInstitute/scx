"""Tests for `pyscx.doublet_consensus`.

The last step of the doublet interop: N tools' canonical obs columns in, one
consensus call out. Nothing here knows what a doublet is — it is arithmetic
over `<key>_predicted` / `<key>_score`.

The property most of these fixtures exist to defend is the null one. A tool
that never saw a cell does not vote on it, and a cell nobody voted on must come
out `null`, not `False` — those are different facts, and the importer went out
of its way to keep them apart (it refuses to fabricate a `0.0` for an uncovered
cell). A naive `sum(...) >= k` over a nullable column destroys the distinction
silently, which is why the all-null row and the partial-coverage cases are
pinned from several directions.
"""

import numpy as np
import pytest

import pyscx

anndata = pytest.importorskip("anndata")
sparse = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

T, F, N = True, False, None


def _adata(calls, scores=None, extra=None, n_vars=3):
    """Build an AnnData whose obs already carries canonical tool columns.

    `calls` is `{key: [True/False/None per cell]}` — None meaning that tool
    never saw the cell, exactly as `doublet_import` leaves it. `scores` is the
    same shape with floats and `None`.
    """
    source = calls or scores or extra
    n = len(next(iter(source.values())))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n)])
    for key, values in (calls or {}).items():
        obs[f"{key}_predicted"] = pd.array(values, dtype="boolean")
    for key, values in (scores or {}).items():
        obs[f"{key}_score"] = pd.array(values, dtype="Float32").astype("float32")
    for name, values in (extra or {}).items():
        obs[name] = values
    X = sparse.csr_matrix(np.arange(n * n_vars, dtype=np.float32).reshape(n, n_vars))
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _scx(tmp_path, adata, name="atlas.scx"):
    path = tmp_path / name
    pyscx.from_anndata(adata, str(path))
    return str(path)


def _obs(path):
    return pyscx.open(path).read_obs()


def _calls(series):
    """A call column as plain `True`/`False`/`None`.

    Nulls surface as `pd.NA` from a nullable column and as `None` from an
    object one depending on which side of the file round trip they came from,
    and `pd.NA != None`, so comparisons go through here rather than asserting
    on whichever spelling a given path happens to produce.
    """
    return [None if pd.isna(v) else bool(v) for v in series]


# ---------------------------------------------------------------------------
# The voting rules
# ---------------------------------------------------------------------------


def test_majority_needs_more_than_half():
    # Rows: 2/3 call, 1/3 calls, 3/3 call, 0/3 call.
    a = _adata({
        "one":   [T, T, T, F],
        "two":   [T, F, T, F],
        "three": [F, F, T, F],
    })
    r = pyscx.doublet_consensus(a, keys=["one", "two", "three"])

    assert _calls(a.obs["doublet_predicted"]) == [T, F, T, F]
    assert list(a.obs["doublet_n_tools_calling"]) == [2, 1, 3, 0]
    assert list(a.obs["doublet_n_tools_voting"]) == [3, 3, 3, 3]
    assert r["method"] == "majority"
    assert r["n_predicted_doublet"] == 2
    assert r["n_predicted_singlet"] == 2
    assert r["n_no_vote"] == 0


def test_an_even_split_is_not_a_majority():
    # Two tools disagreeing is a real disagreement, and False is the honest
    # answer: it is NOT the same as nobody having looked, which is null.
    a = _adata({"one": [T, T], "two": [F, T]})
    pyscx.doublet_consensus(a, keys=["one", "two"])

    assert a.obs["doublet_predicted"].iloc[0] is not pd.NA
    assert bool(a.obs["doublet_predicted"].iloc[0]) is False
    assert bool(a.obs["doublet_predicted"].iloc[1]) is True
    assert list(a.obs["doublet_n_tools_voting"]) == [2, 2]


def test_any_and_all():
    a = _adata({"one": [T, T, F], "two": [T, F, F]})

    pyscx.doublet_consensus(a, keys=["one", "two"], method="any",
                            key_added="dbl_any")
    pyscx.doublet_consensus(a, keys=["one", "two"], method="all",
                            key_added="dbl_all")

    assert list(a.obs["dbl_any_predicted"]) == [T, T, F]
    assert list(a.obs["dbl_all_predicted"]) == [T, F, F]


def test_all_does_not_count_a_non_voter_against_the_call():
    # "all" means every tool that *looked* agreed, not every tool named. Cell 1
    # was seen by `one` alone, which called it — so it is a doublet. Requiring
    # len(keys) agreements instead would make partial coverage silently
    # unanimous-proof: no cell any tool skipped could ever be called.
    a = _adata({"one": [T, T], "two": [T, N]})
    pyscx.doublet_consensus(a, keys=["one", "two"], method="all")

    assert list(a.obs["doublet_n_tools_voting"]) == [2, 1]
    assert _calls(a.obs["doublet_predicted"]) == [T, T]


def test_majority_over_a_partial_voter_uses_only_the_voters():
    # `two` never saw cells 1 and 2. On those rows the majority is decided by
    # `one` alone — a non-voter must not be counted as a "singlet" vote, which
    # would make cell 1 a 1-of-2 minority instead of a 1-of-1 majority.
    a = _adata({"one": [T, T, F], "two": [F, N, N]})
    pyscx.doublet_consensus(a, keys=["one", "two"])

    assert list(a.obs["doublet_n_tools_voting"]) == [2, 1, 1]
    assert _calls(a.obs["doublet_predicted"]) == [F, T, F]


# ---------------------------------------------------------------------------
# The null rule — the property the whole design exists to preserve
# ---------------------------------------------------------------------------


def test_a_row_no_tool_voted_on_is_null_not_false():
    a = _adata({"one": [T, N, F], "two": [F, N, F]})
    r = pyscx.doublet_consensus(a, keys=["one", "two"])

    call = a.obs["doublet_predicted"]
    assert call.isna().tolist() == [False, True, False]
    # The uncovered row is null, and its neighbours are a real False — the
    # distinction a `sum(...) >= k` would have flattened.
    assert bool(call.iloc[2]) is False
    assert r["n_no_vote"] == 1
    # Row 0 is an even split -> singlet; row 2 is unanimous singlet.
    assert r["n_predicted_singlet"] == 2


def test_n_tools_voting_is_what_makes_a_zero_readable():
    # `n_tools_calling == 0` alone is ambiguous: every tool said singlet, or no
    # tool looked? `n_tools_voting` resolves it, which is why it is written.
    a = _adata({"one": [F, N], "two": [F, N]})
    pyscx.doublet_consensus(a, keys=["one", "two"])

    assert list(a.obs["doublet_n_tools_calling"]) == [0, 0]
    assert list(a.obs["doublet_n_tools_voting"]) == [2, 0]
    assert a.obs["doublet_predicted"].isna().tolist() == [False, True]


def test_every_row_unvoted_is_all_null():
    a = _adata({"one": [N, N], "two": [N, N]})
    r = pyscx.doublet_consensus(a, keys=["one", "two"], method="any")

    assert a.obs["doublet_predicted"].isna().all()
    assert r["n_no_vote"] == 2
    assert r["n_predicted_doublet"] == 0
    assert r["n_predicted_singlet"] == 0


@pytest.mark.parametrize("method", ["majority", "any", "all"])
def test_no_vote_is_null_under_every_call_method(method):
    a = _adata({"one": [T, N], "two": [T, N]})
    pyscx.doublet_consensus(a, keys=["one", "two"], method=method)
    assert a.obs["doublet_predicted"].isna().tolist() == [False, True]


# ---------------------------------------------------------------------------
# mean_rank
# ---------------------------------------------------------------------------


def test_mean_rank_ranks_within_each_tool_then_averages():
    # `two` covers only the last three cells. Ranking per tool rather than
    # pooling raw scores is what keeps its 0..1 contribution comparable to
    # `one`'s despite the different coverage and scale.
    a = _adata(
        calls={},
        scores={
            "one": [0.1, 0.2, 0.3, 0.9],
            "two": [None, 900.0, 100.0, 500.0],
        },
    )
    r = pyscx.doublet_consensus(a, keys=["one", "two"], method="mean_rank",
                                quantile=0.25)

    assert "doublet_score" in a.obs
    assert a.obs["doublet_score"].dtype == np.float32
    # Cell 0 was scored by `one` only, so its mean rank is that tool's alone.
    assert not np.isnan(a.obs["doublet_score"].iloc[0])
    assert list(a.obs["doublet_n_tools_voting"]) == [1, 2, 2, 2]
    # A quarter of four assessed cells is one call.
    assert int(a.obs["doublet_predicted"].sum()) == 1
    assert r["quantile"] == 0.25
    assert r["threshold"] is not None


def test_mean_rank_leaves_an_unscored_cell_null():
    a = _adata(calls={}, scores={"one": [0.1, 0.9, None]})
    pyscx.doublet_consensus(a, keys=["one"], method="mean_rank", quantile=0.5)

    assert np.isnan(a.obs["doublet_score"].iloc[2])
    assert a.obs["doublet_predicted"].isna().tolist() == [False, False, True]
    assert list(a.obs["doublet_n_tools_voting"]) == [1, 1, 0]


def test_mean_rank_requires_a_quantile():
    a = _adata(calls={}, scores={"one": [0.1, 0.9]})
    with pytest.raises(ValueError) as ei:
        pyscx.doublet_consensus(a, keys=["one"], method="mean_rank")
    msg = str(ei.value)
    # Refuses to invent a cutoff, and names the alternative.
    assert "quantile" in msg
    assert "majority" in msg


def test_quantile_is_rejected_for_a_call_method():
    a = _adata({"one": [T, F]})
    with pytest.raises(ValueError, match="mean_rank"):
        pyscx.doublet_consensus(a, keys=["one"], quantile=0.1)


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_quantile_must_be_a_proper_fraction(bad):
    a = _adata(calls={}, scores={"one": [0.1, 0.9]})
    with pytest.raises(ValueError, match="between 0 and 1"):
        pyscx.doublet_consensus(a, keys=["one"], method="mean_rank",
                                quantile=bad)


def test_mean_rank_uses_a_score_only_tool_that_cannot_vote_on_calls():
    # scds emits a score and no call. It is an error under "majority" and fine
    # here, which is the whole reason mean_rank exists.
    a = _adata({"one": [T, F]}, scores={"one": [0.9, 0.1], "scds": [0.8, 0.2]})
    r = pyscx.doublet_consensus(a, keys=["one", "scds"], method="mean_rank",
                                quantile=0.5)

    assert list(a.obs["doublet_n_tools_voting"]) == [2, 2]
    # `n_tools_calling` stays a fact about the tools' own calls: scds has none.
    assert list(a.obs["doublet_n_tools_calling"]) == [1, 0]
    assert r["per_key"]["scds"]["n_voting"] == 2
    assert "predicted_column" not in r["per_key"]["scds"]


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_score_only_tool_cannot_vote_on_a_call():
    a = _adata({"one": [T, F]}, scores={"scds": [0.9, 0.1]})
    with pytest.raises(ValueError) as ei:
        pyscx.doublet_consensus(a, keys=["one", "scds"])
    msg = str(ei.value)
    assert "scds" in msg
    # Dropping a voter silently would change the answer, so the message has to
    # offer the three real options.
    assert "mean_rank" in msg
    assert "call_column" in msg


def test_an_unknown_key_names_what_is_available():
    a = _adata({"scrublet": [T, F]})
    with pytest.raises(ValueError) as ei:
        pyscx.doublet_consensus(a, keys=["solo"])
    msg = str(ei.value)
    assert "solo_predicted" in msg
    assert "scrublet" in msg


def test_a_native_call_column_is_refused_by_value():
    # Pointing `keys` at a tool's own column rather than the canonical one:
    # "doublet"/"singlet" strings. Treating a non-empty string as truthy would
    # call every single cell a doublet.
    a = _adata({}, extra={"scdblfinder_predicted": ["doublet", "singlet"]})
    with pytest.raises(ValueError) as ei:
        pyscx.doublet_consensus(a, keys=["scdblfinder"])
    assert "doublet" in str(ei.value)
    assert "key_added" in str(ei.value)


def test_a_non_binary_numeric_call_column_is_refused():
    # 0/1 is accepted (matching the importer's own coercion); an arbitrary
    # float is a score wearing a call's name and must not be thresholded here.
    ok = _adata({}, extra={"t_predicted": np.array([1, 0, 1], dtype=np.int64)})
    pyscx.doublet_consensus(ok, keys=["t"], method="any")
    assert list(ok.obs["doublet_predicted"]) == [T, F, T]

    bad = _adata({}, extra={"t_predicted": np.array([0.3, 0.9, 0.1])})
    with pytest.raises(ValueError, match="not a doublet call"):
        pyscx.doublet_consensus(bad, keys=["t"], method="any")


def test_an_empty_or_repeated_keys_list_is_refused():
    a = _adata({"one": [T, F]})
    with pytest.raises(ValueError, match="at least one"):
        pyscx.doublet_consensus(a, keys=[])
    with pytest.raises(ValueError, match="twice"):
        pyscx.doublet_consensus(a, keys=["one", "one"])


def test_key_added_may_not_be_one_of_the_inputs():
    a = _adata({"doublet": [T, F]})
    with pytest.raises(ValueError, match="own output"):
        pyscx.doublet_consensus(a, keys=["doublet"], key_added="doublet")


def test_an_unknown_method_is_refused():
    a = _adata({"one": [T, F]})
    with pytest.raises(ValueError, match="method must be one of"):
        pyscx.doublet_consensus(a, keys=["one"], method="vote")


def test_a_second_run_will_not_silently_clobber():
    a = _adata({"one": [T, F], "two": [T, T]})
    pyscx.doublet_consensus(a, keys=["one"], method="any")
    with pytest.raises(ValueError, match="overwrite=True"):
        pyscx.doublet_consensus(a, keys=["two"], method="any")

    pyscx.doublet_consensus(a, keys=["two"], method="any", overwrite=True)
    assert list(a.obs["doublet_predicted"]) == [T, T]


# ---------------------------------------------------------------------------
# Key discovery
# ---------------------------------------------------------------------------


def test_keys_are_discovered_when_not_named():
    a = _adata({"scdblfinder": [T, F], "scrublet": [T, T]})
    r = pyscx.doublet_consensus(a, method="any")

    assert r["keys"] == ["scdblfinder", "scrublet"]
    assert list(a.obs["doublet_n_tools_voting"]) == [2, 2]


def test_discovery_ignores_its_own_previous_output():
    a = _adata({"one": [T, F]})
    pyscx.doublet_consensus(a, method="any")
    # A second pass must not treat `doublet_predicted` as a third tool.
    r = pyscx.doublet_consensus(a, method="any", overwrite=True)
    assert r["keys"] == ["one"]


def test_discovery_skips_another_consensus_key():
    """B1: `keys=None` must not count a *different* consensus as a voting tool.

    The dogfood repro: three callers where A and C agree on cells 0-4 and B
    disagrees. An earlier 2-tool consensus voting as a 4th tool drives
    majority-of-4 to False on exactly the cells 2 of 3 real callers call
    doublets.
    """
    a = _adata(
        {
            "scrublet": [T, T, T, T, T, F],
            "scdblfinder": [F, F, F, F, F, F],
            "doubletdetection": [T, T, T, T, T, F],
        }
    )
    pyscx.doublet_consensus(a, keys=["scrublet", "scdblfinder"],
                            method="majority", key_added="early")
    assert int(a.obs["early_predicted"].fillna(False).sum()) == 0  # even split

    with pytest.warns(UserWarning, match="early"):
        auto = pyscx.doublet_consensus(a, method="majority", key_added="auto")
    expl = pyscx.doublet_consensus(
        a, keys=["scrublet", "scdblfinder", "doubletdetection"],
        method="majority", key_added="expl",
    )

    assert auto["keys"] == ["doubletdetection", "scdblfinder", "scrublet"]
    assert auto["keys_excluded"] == ["early"]
    assert set(a.obs["auto_n_tools_voting"]) == {3}
    assert auto["n_predicted_doublet"] == expl["n_predicted_doublet"] == 5
    assert list(a.obs["auto_predicted"]) == list(a.obs["expl_predicted"])


def test_discovery_skips_a_mean_rank_consensus_score():
    """A prior `mean_rank` consensus writes `<K>_score`, so a `_predicted`-only
    exclusion would miss it. This is the case the obs `_n_tools_voting` marker
    catches and the uns record alone would not."""
    a = _adata(
        {"one": [T, F], "two": [F, T]},
        scores={"one": [0.9, 0.1], "two": [0.2, 0.8]},
    )
    pyscx.doublet_consensus(a, method="mean_rank", quantile=0.5, key_added="mr")
    assert "mr_score" in a.obs.columns

    with pytest.warns(UserWarning, match="mr"):
        r = pyscx.doublet_consensus(
            a, method="mean_rank", quantile=0.5, key_added="mr2"
        )
    assert r["keys"] == ["one", "two"]
    assert r["keys_excluded"] == ["mr"]


def test_discovery_survives_a_dropped_uns():
    """The exclusion must not depend on uns: another op can drop or rewrite it,
    and files written before the fix have no record at all."""
    a = _adata({"one": [T, F], "two": [F, T]})
    pyscx.doublet_consensus(a, keys=["one", "two"], method="any", key_added="c1")
    a.uns.clear()

    with pytest.warns(UserWarning, match="c1"):
        r = pyscx.doublet_consensus(a, method="any", key_added="c2")
    assert r["keys"] == ["one", "two"]
    assert r["keys_excluded"] == ["c1"]


def test_a_tool_named_like_a_consensus_is_still_discovered():
    """`doublet_import(key_added="foo_consensus")` writes `uns["foo_consensus"]`,
    which must not be mistaken for a consensus record over key `foo`."""
    a = _adata({"foo": [T, F], "bar": [F, T]})
    a.uns["foo_consensus"] = {"tool": "scrublet", "source_call_column": "x"}
    r = pyscx.doublet_consensus(a, method="any")
    assert r["keys"] == ["bar", "foo"]
    assert r["keys_excluded"] == []


def test_a_file_carrying_only_consensus_columns_says_why():
    a = _adata({"one": [T, F]})
    pyscx.doublet_consensus(a, keys=["one"], method="any", key_added="c1")
    del a.obs["one_predicted"]
    with pytest.raises(ValueError, match="previous consensus outputs"):
        pyscx.doublet_consensus(a, method="any", key_added="c2")


def test_an_explicit_consensus_key_is_allowed_but_warned():
    """Combining two consensus panels is coherent, so allow it — but say what it
    costs, and record the unusual choice in the file."""
    a = _adata({"one": [T, F], "two": [F, T]})
    pyscx.doublet_consensus(a, keys=["one", "two"], method="any", key_added="c1")

    with pytest.warns(UserWarning, match="itself a consensus over"):
        r = pyscx.doublet_consensus(a, keys=["one", "c1"], method="any",
                                    key_added="c2")
    assert r["keys"] == ["one", "c1"]
    assert r["keys_that_are_consensus"] == ["c1"]
    assert set(a.obs["c2_n_tools_voting"]) == {2}


def test_discovery_with_nothing_to_discover_says_so():
    a = _adata({}, extra={"donor": ["d1", "d2"]})
    with pytest.raises(ValueError, match="nothing to reach a consensus over"):
        pyscx.doublet_consensus(a)


def test_mean_rank_discovers_by_score_column():
    a = _adata({"one": [T, F]}, scores={"one": [0.9, 0.1], "scds": [0.8, 0.2]})
    r = pyscx.doublet_consensus(a, method="mean_rank", quantile=0.5)
    assert r["keys"] == ["one", "scds"]


# ---------------------------------------------------------------------------
# Against a real SCX file
# ---------------------------------------------------------------------------


def test_writes_to_an_scx_file_in_place(tmp_path):
    path = _scx(tmp_path, _adata({"one": [T, F, N, T], "two": [T, F, N, F]}))

    r = pyscx.doublet_consensus(path, keys=["one", "two"])

    obs = _obs(path)
    assert _calls(obs["doublet_predicted"]) == [T, F, N, F]
    assert list(obs["doublet_n_tools_voting"]) == [2, 2, 0, 2]
    assert r["n_no_vote"] == 1
    # The record lands in uns under the wrapper's key.
    uns = pyscx.open(path).read_uns()
    assert uns["doublet_consensus"]["method"] == "majority"
    assert uns["doublet_consensus"]["keys"] == ["one", "two"]


def test_the_matrix_and_other_obs_columns_survive(tmp_path):
    a = _adata({"one": [T, F, T]}, extra={"donor": ["d1", "d1", "d2"]})
    path = _scx(tmp_path, a)
    before = pyscx.open(path).to_anndata()

    pyscx.doublet_consensus(path, keys=["one"], method="any")

    after = pyscx.open(path).to_anndata()
    assert np.array_equal(before.X.toarray(), after.X.toarray())
    assert list(after.obs["donor"]) == ["d1", "d1", "d2"]
    assert list(after.obs_names) == list(before.obs_names)
    assert list(after.var_names) == list(before.var_names)


def test_one_rollback_undoes_both_obs_and_uns(tmp_path):
    # obs and uns go in a single commit precisely so this is one step. Two
    # commits would leave a file that had been half-rolled-back.
    path = _scx(tmp_path, _adata({"one": [T, F]}))

    pyscx.doublet_consensus(path, keys=["one"], method="any")
    assert "doublet_predicted" in _obs(path).columns

    pyscx.rollback(path)

    obs = _obs(path)
    assert "doublet_predicted" not in obs.columns
    assert "doublet_n_tools_voting" not in obs.columns
    uns = pyscx.open(path).read_uns()
    assert not uns or "doublet_consensus" not in uns


def test_an_existing_uns_key_is_preserved(tmp_path):
    a = _adata({"one": [T, F]})
    a.uns["pipeline"] = {"step": "qc"}
    path = _scx(tmp_path, a)

    pyscx.doublet_consensus(path, keys=["one"], method="any")

    uns = pyscx.open(path).read_uns()
    assert uns["pipeline"] == {"step": "qc"}
    assert uns["doublet_consensus"]["n_obs"] == 2


def test_a_file_target_refuses_to_clobber_without_overwrite(tmp_path):
    path = _scx(tmp_path, _adata({"one": [T, F], "two": [T, T]}))
    pyscx.doublet_consensus(path, keys=["one"], method="any")

    with pytest.raises(ValueError, match="overwrite=True"):
        pyscx.doublet_consensus(path, keys=["two"], method="any")
    # The refused run left the file alone.
    assert list(_obs(path)["doublet_predicted"]) == [T, F]

    pyscx.doublet_consensus(path, keys=["two"], method="any", overwrite=True)
    assert list(_obs(path)["doublet_predicted"]) == [T, T]


def test_an_experiment_handle_works_as_a_target(tmp_path):
    path = _scx(tmp_path, _adata({"one": [T, F]}))
    pyscx.doublet_consensus(pyscx.open(path), keys=["one"], method="any")
    assert "doublet_predicted" in _obs(path).columns


def test_a_duplicated_obs_index_survives_the_round_trip(tmp_path):
    # A merged atlas's obs index is routinely duplicated — measured 10x over on
    # the CELLxGENE-derived file this work was validated against — and the
    # whole read_obs -> compute -> modify_metadata path has to stay positional
    # through it. (This pins the end-to-end property, not the bare-array
    # assignment specifically: pandas tolerates aligning on a duplicated index
    # while the two indexes compare equal, which they do here.)
    a = _adata({"one": [T, F, T, F]})
    a.obs_names = ["dup", "dup", "dup", "dup"]
    path = _scx(tmp_path, a)

    pyscx.doublet_consensus(path, keys=["one"], method="any")

    obs = _obs(path)
    assert list(obs["doublet_predicted"]) == [T, F, T, F]
    assert list(obs["one_predicted"]) == [T, F, T, F]


def test_deleted_rows_keep_their_place_in_the_physical_obs(tmp_path):
    # `read_obs` is the PHYSICAL row space and `modify_metadata` validates
    # against it, so a file with logical deletions must round trip without a
    # row-count error — and the surviving rows must keep their own values.
    path = _scx(tmp_path, _adata({"one": [T, F, T, F]}))
    pyscx.mark_deleted(path, [1])

    pyscx.doublet_consensus(path, keys=["one"], method="any")

    assert len(_obs(path)) == 4
    adata = pyscx.open(path).to_anndata()
    assert adata.n_obs == 3
    assert list(adata.obs["doublet_predicted"]) == [T, T, F]


def test_the_end_to_end_import_then_consensus_path(tmp_path):
    # The real shape of the workflow: two tools imported by `doublet_import`,
    # each covering part of the file, then a consensus over what they left.
    n = 6
    obs = pd.DataFrame({"donor": ["d1"] * n},
                       index=[f"cell_{i}" for i in range(n)])
    X = sparse.csr_matrix(np.arange(n * 3, dtype=np.float32).reshape(n, 3))
    path = str(tmp_path / "atlas.scx")
    pyscx.from_anndata(
        anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=list("abc"))), path
    )

    scdbl = tmp_path / "scdbl.csv"
    pd.DataFrame({
        "barcode": [f"cell_{i}" for i in range(4)],
        "scDblFinder.score": [0.9, 0.1, 0.8, 0.2],
        "scDblFinder.class": ["doublet", "singlet", "doublet", "singlet"],
    }).to_csv(scdbl, index=False)
    pyscx.doublet_import(path, str(scdbl), tool="scdblfinder")

    scrub = tmp_path / "scrub.csv"
    pd.DataFrame({
        "barcode": [f"cell_{i}" for i in range(2, 6)],
        "doublet_score": [0.7, 0.2, 0.3, 0.1],
        "predicted_doublet": ["True", "False", "False", "False"],
    }).to_csv(scrub, index=False)
    pyscx.doublet_import(path, str(scrub), tool="scrublet")

    r = pyscx.doublet_consensus(path, keys=["scdblfinder", "scrublet"],
                                method="any")

    obs_back = _obs(path)
    # cell_0/1: scdblfinder only. cell_2/3: both. cell_4/5: scrublet only.
    assert list(obs_back["doublet_n_tools_voting"]) == [1, 1, 2, 2, 1, 1]
    assert list(obs_back["doublet_predicted"]) == [T, F, T, F, F, F]
    assert r["n_no_vote"] == 0
    assert r["per_key"]["scdblfinder"]["n_voting"] == 4
    assert r["per_key"]["scrublet"]["n_voting"] == 4


def test_mean_rank_over_imported_scores_on_a_file(tmp_path):
    path = _scx(
        tmp_path,
        _adata(calls={}, scores={"one": [0.1, 0.5, 0.9, None],
                                 "two": [0.2, 0.4, 0.8, 0.99]}),
    )
    r = pyscx.doublet_consensus(path, method="mean_rank", quantile=0.5)

    obs = _obs(path)
    assert obs["doublet_score"].dtype == np.float32
    assert obs["doublet_predicted"].notna().all()
    assert r["threshold"] is not None
    assert r["columns_added"][0] == "doublet_score"


# ---------------------------------------------------------------------------
# The obs predicate index must survive the last step of the workflow
# ---------------------------------------------------------------------------


def test_consensus_on_a_file_keeps_the_obs_predicate_index(tmp_path):
    """The reported bug, end to end.

    `doublet_import` joins by key and knows which columns it writes, so it
    preserves the index. `doublet_consensus` replaces obs wholesale and used to
    take the index with it — silently reverting `query().filter_obs(...)` to a
    full obs scan on the LAST step of the documented workflow, with no warning
    and nothing in provenance to say it had happened.
    """
    a = _adata({"one": [T, F, T, F]},
               extra={"grp": pd.Categorical(["A", "B", "A", "B"])})
    path = str(tmp_path / "atlas.scx")
    pyscx.from_anndata(a, path, index_obs=["grp"])

    def sections():
        return [name for name, _ in pyscx.open(path).validate()]

    assert "obs_predicate_index" in sections()

    pyscx.doublet_consensus(path, keys=["one"], method="any")

    assert "obs_predicate_index" in sections()
    q = pyscx.open(path).query()
    q.filter_obs("grp == 'A'")
    assert q.collect().n_obs == 2


# ---------------------------------------------------------------------------
# Which seam the file write goes through
# ---------------------------------------------------------------------------


def test_consensus_default_path_writes_via_attach_not_modify_metadata(tmp_path):
    """A consensus is a pure column add computed from the file's own obs, so
    the default path takes the attach seam (positional, one-key uns merge) —
    not the whole-frame `modify_metadata` replace."""
    a = _adata({"one": [T, F, T, F]})
    path = _scx(tmp_path, a)

    pyscx.doublet_consensus(path, keys=["one"], method="any")

    entry = pyscx.open(path).provenance()[-1]
    assert entry["action"] == "attach_obs_columns", entry["action"]


def test_consensus_with_index_obs_still_routes_through_modify_metadata(tmp_path):
    """`index_obs` / `index_preset` request an index (re)build, which only
    `modify_metadata` can do in the same commit — the documented respec route."""
    a = _adata({"one": [T, F, T, F]},
               extra={"grp": pd.Categorical(["A", "B", "A", "B"])})
    path = _scx(tmp_path, a)

    pyscx.doublet_consensus(path, keys=["one"], method="any", index_obs=["grp"])

    entry = pyscx.open(path).provenance()[-1]
    assert entry["action"] == "modify_metadata", entry["action"]
    q = pyscx.open(path).query()
    q.filter_obs("grp == 'A'")
    assert q.collect().n_obs == 2, "the requested index must actually be built"


def test_consensus_overwrite_rerun_keeps_an_index_over_a_consensus_column(tmp_path):
    """Round-1 finding (Cursor Agent): the attach seam DROPS a predicate index
    covering an overwritten column, where the whole-frame route rebuilds it.
    A re-run that rewrites existing consensus columns must therefore take the
    modify_metadata route, or someone's index over `doublet_n_tools_calling`
    silently reverts filter_obs to a full scan."""
    a = _adata({"one": [T, F, T, F]})
    path = _scx(tmp_path, a)

    # Run 1: index a consensus column via the respec route.
    pyscx.doublet_consensus(path, keys=["one"], method="any",
                            index_obs=["doublet_n_tools_calling"])

    def sections():
        return [name for name, _ in pyscx.open(path).validate()]

    assert "obs_predicate_index" in sections()

    # Run 2: plain overwrite re-run — no index kwargs.
    pyscx.doublet_consensus(path, keys=["one"], method="any", overwrite=True)

    entry = pyscx.open(path).provenance()[-1]
    assert entry["action"] == "modify_metadata", (
        "an overwriting re-run must take the route that rebuilds the index"
    )
    assert "obs_predicate_index" in sections(), (
        "the index over the rewritten consensus column must survive the re-run"
    )
    # calls [T, F, T, F] under method="any": two cells have one tool calling.
    q = pyscx.open(path).query()
    q.filter_obs("doublet_n_tools_calling == 1")
    assert q.collect().n_obs == 2
