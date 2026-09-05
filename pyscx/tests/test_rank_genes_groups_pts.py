"""`rank_genes_groups(pts=, groups=, corr_method=)` and the `pct_nz_*` columns
of `rank_genes_groups_df` (REC-10, PR G).

`pts` is scanpy's "fraction of cells in the group with a nonzero value":
`uns[key]["pts"]` (every group) and `uns[key]["pts_rest"]` (1-vs-rest only)
as `genes × groups` DataFrames indexed by var name, over every gene whatever
`n_genes` says. The count is exact, so the bar against scanpy is 1e-12 — only
the division can differ, and it does not.

`groups=` is an *output* filter: "rest" keeps every other cell, so a group's
statistics are identical with or without it, and identical to scanpy's
`groups=`. Every test pins `device="cpu"`.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

PTS_ATOL = 1e-12
KEY = "rank_genes_groups"


def _rgg(adata, **kw):
    kw.setdefault("device", "cpu")
    pyscx.accel.rank_genes_groups(adata, "batch", **kw)
    return adata.uns[KEY]


def _manual_pts(adata, group_labels, groups, reference="rest"):
    """scanpy's definition, computed by hand: nonzero fraction per group, and
    the nonzero fraction over `X[~mask_g]` — every other cell, an unlabelled
    one included."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    nz = X != 0
    labels = np.asarray(group_labels, dtype=object)
    pts = {}
    pts_rest = {}
    for g in groups:
        in_g = labels == g
        pts[g] = nz[in_g].mean(axis=0)
        pts_rest[g] = nz[~in_g].mean(axis=0)
    pts = pd.DataFrame(pts, index=adata.var_names)
    pts_rest = pd.DataFrame(pts_rest, index=adata.var_names) if reference == "rest" else None
    return pts, pts_rest


def _assert_frames_equal_to(a: pd.DataFrame, b: pd.DataFrame, atol: float) -> None:
    assert list(a.index) == list(b.index)
    assert list(a.columns) == list(b.columns)
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), atol=atol, rtol=0)


# ---------------------------------------------------------------------------
# pts vs scanpy
# ---------------------------------------------------------------------------


def test_pts_and_pts_rest_match_scanpy_to_1e12(synthetic_adata):
    sc = pytest.importorskip("scanpy")
    adata_sc = synthetic_adata.copy()
    sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon", pts=True)
    rgg_sc = adata_sc.uns[KEY]

    rgg = _rgg(synthetic_adata.copy(), pts=True)
    for table in ("pts", "pts_rest"):
        assert isinstance(rgg[table], pd.DataFrame)
        assert rgg[table].dtypes.map(lambda d: d == np.float64).all()
        _assert_frames_equal_to(rgg[table], rgg_sc[table], PTS_ATOL)
    # Every gene, var order — never rank order, never truncated.
    assert list(rgg["pts"].index) == list(synthetic_adata.var_names)


def test_pct_nz_columns_of_rank_genes_groups_df_match_scanpy(synthetic_adata):
    sc = pytest.importorskip("scanpy")
    adata_sc = synthetic_adata.copy()
    sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon", pts=True)
    adata = synthetic_adata.copy()
    _rgg(adata, pts=True)

    for group in ("A", "B", "C"):
        want = sc.get.rank_genes_groups_df(adata_sc, group=group).set_index("names")
        got = pyscx.accel.rank_genes_groups_df(adata, group=group)
        assert list(got.columns) == [
            "names",
            "scores",
            "logfoldchanges",
            "pvals",
            "pvals_adj",
            "pct_nz_group",
            "pct_nz_reference",
        ]
        got = got.set_index("names")
        assert set(got.index) == set(want.index)
        for col in ("pct_nz_group", "pct_nz_reference"):
            np.testing.assert_allclose(
                got.loc[want.index, col].to_numpy(),
                want[col].to_numpy(),
                atol=PTS_ATOL,
                rtol=0,
            )

    # The all-groups extract keeps the `group` column in front.
    every = pyscx.accel.rank_genes_groups_df(adata)
    assert list(every.columns)[:2] == ["group", "names"]
    assert list(every.columns)[-2:] == ["pct_nz_group", "pct_nz_reference"]
    # ...and after the row filters the values still belong to their gene.
    filtered = pyscx.accel.rank_genes_groups_df(adata, group="A", pval_cutoff=0.5).set_index(
        "names"
    )
    np.testing.assert_array_equal(
        filtered["pct_nz_group"].to_numpy(),
        adata.uns[KEY]["pts"].loc[filtered.index, "A"].to_numpy(),
    )


def test_pairwise_reference_writes_pts_with_the_reference_column_and_no_pts_rest(
    synthetic_adata,
):
    sc = pytest.importorskip("scanpy")
    adata_sc = synthetic_adata.copy()
    sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon", reference="A", pts=True)
    rgg = _rgg(synthetic_adata.copy(), reference="A", pts=True)

    assert list(rgg["pts"].columns) == ["A", "B", "C"]  # category order, reference in place
    assert "pts_rest" not in rgg
    assert "pts_rest" not in adata_sc.uns[KEY]
    _assert_frames_equal_to(rgg["pts"], adata_sc.uns[KEY]["pts"], PTS_ATOL)

    adata = synthetic_adata.copy()
    _rgg(adata, reference="A", pts=True)
    df = pyscx.accel.rank_genes_groups_df(adata, group="B")
    assert "pct_nz_group" in df.columns
    assert "pct_nz_reference" not in df.columns  # scanpy has no fallback either


def test_pts_defaults_off_and_writes_neither_table(synthetic_adata):
    adata = synthetic_adata.copy()
    rgg = _rgg(adata)
    assert "pts" not in rgg and "pts_rest" not in rgg
    df = pyscx.accel.rank_genes_groups_df(adata, group="A")
    assert "pct_nz_group" not in df.columns and "pct_nz_reference" not in df.columns
    assert rgg["params"]["corr_method"] == "benjamini-hochberg"


def test_pts_covers_every_gene_even_when_n_genes_truncates(synthetic_adata):
    rgg = _rgg(synthetic_adata.copy(), pts=True, n_genes=5)
    assert len(rgg["names"]) == 5
    assert rgg["pts"].shape == (synthetic_adata.n_vars, 3)
    assert rgg["pts_rest"].shape == (synthetic_adata.n_vars, 3)


# ---------------------------------------------------------------------------
# pts is route-independent: backed, lazy, CSC-direct, dense
# ---------------------------------------------------------------------------


def test_pts_on_backed_lazy_csc_and_dense_equals_in_memory(synthetic_adata, tmp_path):
    ref = _rgg(synthetic_adata.copy(), pts=True)

    path = str(tmp_path / "pts.scx")
    pyscx.from_anndata(synthetic_adata, path, csc="always", csc_cols_per_shard=7)

    backed = pyscx.open(path).to_anndata(backed=True)
    got = _rgg(backed, pts=True)
    for table in ("pts", "pts_rest"):
        pd.testing.assert_frame_equal(got[table], ref[table])

    csc = pyscx.open(path).to_anndata(backed=True)
    got = _rgg(csc, pts=True, prefer_format="csc")
    assert csc.uns["scx_accel"][KEY]["route"] == "cpu_csc"
    for table in ("pts", "pts_rest"):
        pd.testing.assert_frame_equal(got[table], ref[table])

    lazy = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.log1p(lazy)
    assert type(lazy.X).__name__ == "ScxLazyTransformedDataset"
    got = _rgg(lazy, pts=True)
    for table in ("pts", "pts_rest"):
        pd.testing.assert_frame_equal(got[table], ref[table])

    dense = synthetic_adata.copy()
    dense.X = dense.X.toarray()
    got = _rgg(dense, pts=True)
    for table in ("pts", "pts_rest"):
        pd.testing.assert_frame_equal(got[table], ref[table])


def test_pts_counts_nonzero_not_positive_and_ignores_stored_zeros():
    """scanpy: `eliminate_zeros()` then `getnnz` — a stored zero is not
    expressing, a negative value is. Pinned against a hand count."""
    import anndata

    rng = np.random.default_rng(3)
    n_obs, n_vars = 40, 6
    dense = rng.normal(size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) < 0.5] = 0.0
    X = sp.csr_matrix(dense)
    # Plant explicit zeros in the stored data of a few entries.
    X.data[:5] = 0.0
    assert (X.data == 0).sum() >= 5
    labels = pd.Categorical(["g0", "g1"] * (n_obs // 2))
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame({"batch": labels}, index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)]),
    )
    rgg = _rgg(adata, pts=True)
    want, want_rest = _manual_pts(adata, adata.obs["batch"], ["g0", "g1"])
    _assert_frames_equal_to(rgg["pts"], want, 0.0)
    _assert_frames_equal_to(rgg["pts_rest"], want_rest, 0.0)
    # The negatives were counted: pts is larger than the positive-only fraction.
    pos_only = (adata.X.toarray() > 0)[np.asarray(labels) == "g0"].mean(axis=0)
    assert (rgg["pts"]["g0"].to_numpy() >= pos_only).all()
    assert (rgg["pts"]["g0"].to_numpy() > pos_only).any()


def test_pts_rest_with_unlabelled_cells_matches_scanpy(synthetic_adata):
    """scanpy's `pts_rest[g]` is over `X[~mask_g]`: every other cell, the
    NaN-labelled ones included. Since X9 the statistic uses that same pool, so
    the two agree with each other as well as with scanpy — see
    `test_pts_rest_and_the_statistic_share_one_pool` below."""
    sc = pytest.importorskip("scanpy")
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[:7] = None
    adata.obs["batch"] = pd.Categorical(labels, categories=["A", "B", "C"])
    # Plant a gene nonzero only in unlabelled cells: its pts is 0 for every
    # group while its pts_rest is not — the arm that tells the two pools apart.
    X = adata.X.tolil()
    X[:, 0] = 0
    X[:7, 0] = 3.0
    adata.X = X.tocsr()

    adata_sc = adata.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon", pts=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pyscx's unlabelled-cell warning
        rgg = _rgg(adata, pts=True)
    rgg_sc = adata_sc.uns[KEY]
    _assert_frames_equal_to(rgg["pts"], rgg_sc["pts"], PTS_ATOL)
    _assert_frames_equal_to(rgg["pts_rest"], rgg_sc["pts_rest"], PTS_ATOL)
    assert (rgg["pts"].iloc[0] == 0).all()
    assert (rgg["pts_rest"].iloc[0] > 0).all()
    want, want_rest = _manual_pts(adata, labels, ["A", "B", "C"])
    _assert_frames_equal_to(rgg["pts"], want, 0.0)
    _assert_frames_equal_to(rgg["pts_rest"], want_rest, 0.0)


def test_pts_rest_and_the_statistic_share_one_pool(synthetic_adata):
    """One dict, one reference population (X9).

    0.17 shipped `pts_rest` over `X[~mask_g]` (unlabelled included) while
    `scores` / `pvals` still used the labelled-only pool, so a single
    `uns["rank_genes_groups"]` described two different "rest"s with nothing
    saying so. The gene planted below is the sharpest statement of it: nonzero
    *only* in the unlabelled cells, so it is indistinguishable from an all-zero
    gene to anything that leaves them out.

    Asserted against a gene that really is all-zero on the same run, rather than
    against a number: the claim is that the two are distinguishable, and that is
    exactly what a shared pool buys.
    """
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[:7] = None
    adata.obs["batch"] = pd.Categorical(labels, categories=["A", "B", "C"])
    X = adata.X.tolil()
    X[:, 0] = 0
    X[:7, 0] = 3.0  # gene 0: nonzero only in the unlabelled cells
    X[:, 1] = 0  # gene 1: all zeros everywhere
    adata.X = X.tocsr()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rgg = _rgg(adata, pts=True)

    unlabelled_only, all_zero = adata.var_names[0], adata.var_names[1]
    # pts_rest already saw those cells before X9 …
    assert (rgg["pts_rest"].loc[unlabelled_only] > 0).all()
    assert (rgg["pts_rest"].loc[all_zero] == 0).all()
    # … and now the statistic does too, so the two genes are not the same
    # column any more.
    for group in rgg["names"].dtype.names:
        scores = dict(zip(rgg["names"][group], rgg["scores"][group]))
        assert scores[unlabelled_only] != scores[all_zero], (
            f"group {group}: a gene nonzero only in unlabelled cells scores the "
            f"same as an all-zero gene — `pts_rest` and `scores` are still "
            f"describing different reference populations"
        )


def _dup_adata():
    import anndata

    rng = np.random.default_rng(11)
    X = sp.csr_matrix(rng.poisson(1.0, size=(30, 4)).astype(np.float32))
    return anndata.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"batch": pd.Categorical(["A", "B", "C"] * 10)},
            index=[f"c{i}" for i in range(30)],
        ),
        var=pd.DataFrame(index=["dup", "dup", "g2", "g3"]),
    )


def test_pts_refuses_duplicate_var_names_on_write_and_on_extract():
    """AnnData permits duplicate `var_names`, but a `pts` table indexed by them
    cannot be joined by gene name without handing one gene's fraction to the
    other (scanpy's merge multiplies the rows instead). Both ends refuse:
    `rank_genes_groups(pts=True)` before writing, and the extractor when a
    table with a duplicated index is already in `uns` (scanpy-written, say).
    Without `pts` the duplicate names are as fine as they ever were."""
    adata = _dup_adata()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # anndata's non-unique var_names warning
        with pytest.raises(ValueError, match=r'unique var names.*"dup".*var_names_make_unique'):
            _rgg(adata, pts=True)
        # With the names coming from `adata.raw`, `var_names_make_unique()` would
        # not help — the message says so and names the way out.
        with_raw = adata.copy()
        with_raw.raw = with_raw
        with pytest.raises(ValueError, match=r"adata\.raw\.var_names.*use_raw=False"):
            _rgg(with_raw, pts=True)
        _rgg(adata)  # pts=False: unaffected
        assert len(pyscx.accel.rank_genes_groups_df(adata, group="A")) == 4
        adata.uns[KEY]["pts"] = pd.DataFrame(
            {g: np.zeros(4) for g in ("A", "B", "C")}, index=list(adata.var_names)
        )
        with pytest.raises(ValueError, match=r'\["pts"\] has a duplicated var name.*"dup"'):
            pyscx.accel.rank_genes_groups_df(adata, group="A")


def test_pct_nz_survives_gene_names_longer_than_200_characters(synthetic_adata):
    """`names` used to be a fixed `U200`, so a longer var name was truncated in
    the structured array and missed the full-name `pts` index — a silent NaN.
    It is object dtype now (scanpy's), which also means one long name does not
    widen every cell of every group the way a fitted `U{max}` would."""
    adata = synthetic_adata.copy()
    long_name = "G" * 250
    names = list(adata.var_names)
    names[3] = long_name
    adata.var_names = names
    _rgg(adata, pts=True)
    rgg = adata.uns[KEY]
    assert rgg["names"].dtype["A"].kind == "O"
    assert long_name in list(rgg["names"]["A"])
    df = pyscx.accel.rank_genes_groups_df(adata, group="A").set_index("names")
    assert not np.isnan(df.loc[long_name, "pct_nz_group"])
    assert df.loc[long_name, "pct_nz_group"] == rgg["pts"].loc[long_name, "A"]


def test_pct_nz_refuses_a_pts_rest_indexed_unlike_pts(synthetic_adata):
    adata = synthetic_adata.copy()
    _rgg(adata, pts=True)
    adata.uns[KEY]["pts_rest"] = adata.uns[KEY]["pts_rest"].iloc[::-1]
    with pytest.raises(ValueError, match="not indexed like"):
        pyscx.accel.rank_genes_groups_df(adata, group="A")


def test_pts_is_refused_with_stratify_by(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.obs["stratum"] = pd.Categorical(["s0", "s1"] * (adata.n_obs // 2))
    with pytest.raises(ValueError, match="pts=True cannot be combined with stratify_by"):
        pyscx.accel.rank_genes_groups(
            adata, "batch", stratify_by=["stratum"], min_cells_per_stratum=1, pts=True
        )


# ---------------------------------------------------------------------------
# groups=
# ---------------------------------------------------------------------------


def test_groups_restricts_and_reorders_the_output_without_changing_values(synthetic_adata):
    full = _rgg(synthetic_adata.copy(), pts=True)
    sub = _rgg(synthetic_adata.copy(), pts=True, groups=["B", "A"])
    assert sub["names"].dtype.names == ("B", "A")
    for field in ("names", "scores", "pvals", "pvals_adj", "logfoldchanges"):
        for g in ("B", "A"):
            np.testing.assert_array_equal(sub[field][g], full[field][g])
    # `pts` columns follow the request; `pts_rest` too (1-vs-rest).
    assert list(sub["pts"].columns) == ["B", "A"]
    pd.testing.assert_frame_equal(sub["pts"], full["pts"][["B", "A"]])
    pd.testing.assert_frame_equal(sub["pts_rest"], full["pts_rest"][["B", "A"]])
    # No `groups` key in params: scanpy writes none either.
    assert "groups" not in sub["params"]


def test_groups_with_a_named_reference_appends_it_to_pts_only(synthetic_adata):
    rgg = _rgg(synthetic_adata.copy(), reference="C", pts=True, groups=["A"])
    assert rgg["names"].dtype.names == ("A",)
    assert list(rgg["pts"].columns) == ["A", "C"]
    assert "pts_rest" not in rgg
    # Naming the reference in `groups` is not an error (scanpy drops it silently).
    again = _rgg(synthetic_adata.copy(), reference="C", pts=True, groups=["C", "A"])
    assert again["names"].dtype.names == ("A",)
    assert list(again["pts"].columns) == ["C", "A"]


def test_groups_subset_matches_scanpy_groups(synthetic_adata):
    """The 'rest' semantic: scanpy's `groups=["A"]` keeps B and C cells in
    rest, so its numbers for A are the unrestricted ones. So are ours."""
    sc = pytest.importorskip("scanpy")
    from test_accel import SCANPY_PVAL_ATOL, SCANPY_SCORE_ATOL

    adata_sc = synthetic_adata.copy()
    sc.tl.rank_genes_groups(adata_sc, "batch", method="wilcoxon", groups=["A"], pts=True)
    rgg_sc = adata_sc.uns[KEY]
    rgg = _rgg(synthetic_adata.copy(), groups=["A"], pts=True)
    assert rgg["names"].dtype.names == rgg_sc["names"].dtype.names == ("A",)
    sc_map = dict(zip(rgg_sc["names"]["A"], rgg_sc["scores"]["A"]))
    for gene, score in zip(rgg["names"]["A"], rgg["scores"]["A"]):
        assert abs(score - sc_map[gene]) <= SCANPY_SCORE_ATOL
    sc_p = dict(zip(rgg_sc["names"]["A"], rgg_sc["pvals"]["A"]))
    for gene, p in zip(rgg["names"]["A"], rgg["pvals"]["A"]):
        assert abs(p - sc_p[gene]) <= SCANPY_PVAL_ATOL
    _assert_frames_equal_to(rgg["pts"], rgg_sc["pts"], PTS_ATOL)
    _assert_frames_equal_to(rgg["pts_rest"], rgg_sc["pts_rest"], PTS_ATOL)


def _one_cell_in_group_a(adata, categories=("A", "B", "C")):
    """Move every A cell but one into B, so `A` has exactly one cell."""
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[labels == "A"] = "B"
    labels[0] = "A"
    adata.obs["batch"] = pd.Categorical(labels, categories=list(categories))
    return adata, labels


def test_groups_refuses_a_named_group_with_fewer_than_two_cells(synthetic_adata):
    """scanpy's rule, applied to the groups the caller named (and a named
    reference). The `groups=None` half is pinned by the sibling test below."""
    adata, labels = _one_cell_in_group_a(synthetic_adata.copy())
    with pytest.raises(ValueError, match="groups A since they only contain one sample"):
        _rgg(adata, groups=["A"])
    with pytest.raises(ValueError, match="groups A since they only contain one sample"):
        _rgg(adata, groups=["B"], reference="A")
    # A named group with two or more cells is fine — the guard is a floor, not
    # a filter, and naming only healthy groups must not trip it even though a
    # singleton level exists in the column.
    rgg = _rgg(adata, groups=["B"])
    assert rgg["names"].dtype.names == ("B",)
    # An empty category named in `groups` is refused the same way.
    adata.obs["batch"] = pd.Categorical(labels, categories=["A", "B", "C", "D"])
    with pytest.raises(ValueError, match="groups D since they only contain one sample"):
        _rgg(adata, groups=["D", "B"])


def test_default_path_refuses_a_singleton_group_too(synthetic_adata):
    """X8: the guard must not depend on how the caller spelled the request.

    `groups=None` is what every ordinary call does, and until this it was the
    *unguarded* branch: a one-cell group came back with finite, plausible
    z-scores and no warning, while `groups=["A"]` on the same input raised.
    NaN announces itself; `0.99` does not.
    """
    adata, _ = _one_cell_in_group_a(synthetic_adata.copy())
    with pytest.raises(ValueError, match="groups A since they only contain one sample"):
        _rgg(adata)
    # Same input, same op, named request — this arm already raised, and must
    # keep raising with the identical message.
    with pytest.raises(ValueError, match="groups A since they only contain one sample"):
        _rgg(adata.copy(), groups=["A"])


def test_default_path_accepts_a_two_cell_group(synthetic_adata):
    """The floor is two, not three: a two-cell group is a legal Wilcoxon."""
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[labels == "A"] = "B"
    labels[0] = "A"
    labels[1] = "A"  # exactly two A cells
    adata.obs["batch"] = pd.Categorical(labels, categories=["A", "B", "C"])
    rgg = _rgg(adata)
    assert set(rgg["names"].dtype.names) == {"A", "B", "C"}
    assert np.isfinite(np.asarray(rgg["scores"]["A"], dtype=float)).all()


def test_default_path_refuses_an_unused_category_and_names_the_remedy(synthetic_adata):
    """An unused level has zero cells, and `0 < 2`, so scanpy raises on it too:
    its `value_counts()` reports every category and `groups="all"` selects
    every category. The overwhelmingly common source is a subset that kept its
    parent's categories, so the message says how to drop them."""
    adata = synthetic_adata.copy()
    adata.obs["batch"] = pd.Categorical(
        adata.obs["batch"].astype(object).to_numpy(), categories=["A", "B", "C", "D"]
    )
    with pytest.raises(ValueError) as excinfo:
        _rgg(adata)
    assert "groups D since they only contain one sample" in str(excinfo.value)
    assert "remove_unused_categories()" in str(excinfo.value)


@pytest.mark.parametrize(
    "na", [np.nan, None, pd.NA], ids=["np.nan", "None", "pd.NA"]
)
def test_a_plain_string_column_with_a_missing_value_is_not_a_singleton_group(
    synthetic_adata, na
):
    """The guard must not fire on a non-categorical column carrying a missing value.

    Whether a cell is missing is asked of `pandas.isna`, never inferred from how
    the value prints — `astype("str")` renders `np.nan` as `"nan"`, `None` as
    `"None"` and `pd.NA` as `"<NA>"`. A spelling test caught only the first, so
    a single `None` failed the *whole* default call with "…groups None since
    they only contain one sample", and two of them minted a result row and a
    `pts` column scanpy never emits. scanpy never sees such a level: it runs
    `sanitize_anndata`, and `astype("category")` makes no missing value a
    category.

    Parametrised over all three spellings because one of them passing is
    exactly what hid this: the original test planted `np.nan` only.
    """
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[0] = na
    labels[1] = na  # two, so a phantom level would be a group rather than an error
    adata.obs["batch"] = labels  # plain object column, not a Categorical
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rgg = _rgg(adata, pts=True)
    assert set(rgg["names"].dtype.names) == {"A", "B", "C"}
    assert list(rgg["pts"].columns) == ["A", "B", "C"], (
        "a phantom missing-value level reached the pts frame"
    )


@pytest.mark.parametrize("na", [np.nan, None, pd.NA], ids=["np.nan", "None", "pd.NA"])
def test_a_single_missing_value_does_not_trip_the_singlet_guard(synthetic_adata, na):
    """One missing cell must not fail the run — the sharpest form of the above.

    With missingness read off the printed form, exactly one `None` produced
    `Could not calculate statistics for groups None since they only contain one
    sample.` while scanpy 1.12 succeeded on the same input. Asserted on the
    default path, under `groups=`, and with a named reference, because the
    guard's participating set differs in each.
    """
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[0] = na
    adata.obs["batch"] = labels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        assert set(_rgg(adata.copy())["names"].dtype.names) == {"A", "B", "C"}
        assert _rgg(adata.copy(), groups=["B"])["names"].dtype.names == ("B",)
        assert set(_rgg(adata.copy(), reference="A")["names"].dtype.names) == {"B", "C"}


@pytest.mark.parametrize("name", ["nan", "None", "<NA>", ""], ids=lambda s: repr(s))
def test_a_group_whose_name_looks_like_a_missing_value_is_still_a_group(
    synthetic_adata, name
):
    """The other direction: a real cluster named `"nan"` is not a missing value.

    A spelling denylist steals these. `pandas.isna` does not — the strings are
    present, so no cell is missing and every level survives, including under
    `pts` where a stolen level would silently vanish from the frame.
    """
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[labels == "C"] = name
    adata.obs["batch"] = labels
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # nothing is missing here
        rgg = _rgg(adata, pts=True)
    assert set(rgg["names"].dtype.names) == {"A", "B", name}
    assert set(rgg["pts"].columns) == {"A", "B", name}


def test_a_nullable_string_column_with_pd_na_is_handled(synthetic_adata):
    """pandas' nullable `string` dtype, not an object array holding `pd.NA`.

    It has no `.cat`, so it takes the same branch as an object column, and its
    missing value also prints as `"<NA>"`. Called out separately because the
    dtype is what a `pyarrow`-backed or `convert_dtypes()`-ed obs frame gives
    you, and the object-array arm above does not exercise it.
    """
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[0] = pd.NA
    adata.obs["batch"] = pd.array(labels, dtype="string")
    assert str(adata.obs["batch"].dtype) == "string"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rgg = _rgg(adata, pts=True)
    assert set(rgg["names"].dtype.names) == {"A", "B", "C"}
    assert list(rgg["pts"].columns) == ["A", "B", "C"]


def test_a_categorical_level_named_nan_is_told_apart_from_a_real_nan(synthetic_adata):
    """Both print as `"nan"` after `astype("str")`; only `isna` separates them.

    A categorical carrying a level literally called `"nan"` *and* genuine NaN
    cells is the case a spelling test cannot get right at all: it must either
    drop the real level or keep the missing cells. The level keeps its cells and
    the NaN rows are the ones the warning counts.
    """
    adata = synthetic_adata.copy()
    labels = adata.obs["batch"].astype(object).to_numpy()
    labels[labels == "C"] = "nan"
    labels[0] = np.nan  # a genuine missing value, printing the same way
    labels[1] = np.nan
    adata.obs["batch"] = pd.Categorical(labels, categories=["A", "B", "nan"])

    with pytest.warns(UserWarning, match=r"2 of \d+ cells have no group label"):
        rgg = _rgg(adata, pts=True)
    assert set(rgg["names"].dtype.names) == {"A", "B", "nan"}
    # The level is real, so it has cells; the two NaN rows are in nobody's pts.
    assert (rgg["pts"]["nan"] > 0).any()


def test_groups_errors(synthetic_adata):
    with pytest.raises(ValueError, match=r'"Z" is not a level.*available: \["A", "B", "C"\]'):
        _rgg(synthetic_adata.copy(), groups=["A", "Z"])
    with pytest.raises(ValueError, match="at least one group"):
        _rgg(synthetic_adata.copy(), groups=[])
    with pytest.raises(ValueError, match="more than once"):
        _rgg(synthetic_adata.copy(), groups=["A", "A"])
    with pytest.raises(ValueError, match="leaves no group to test"):
        _rgg(synthetic_adata.copy(), reference="A", groups=["A"])
    with pytest.raises(TypeError):
        _rgg(synthetic_adata.copy(), groups="A")


def test_groups_applies_to_the_stratified_frame(synthetic_adata):
    adata = synthetic_adata.copy()
    adata.obs["stratum"] = pd.Categorical(["s0", "s1"] * (adata.n_obs // 2))
    kw = dict(stratify_by=["stratum"], min_cells_per_stratum=1, device="cpu")
    full = pyscx.accel.rank_genes_groups(adata.copy(), "batch", **kw)
    sub = pyscx.accel.rank_genes_groups(adata.copy(), "batch", groups=["C"], **kw)
    assert set(sub["group"]) == {"C"}
    want = full[full["group"] == "C"].reset_index(drop=True)
    pd.testing.assert_frame_equal(sub.reset_index(drop=True), want)


def test_a_stratum_with_a_singleton_group_is_dropped_not_silently_scored(
    synthetic_adata,
):
    """Under `stratify_by` the guard rides the existing per-stratum policy.

    A per-stratum failure already warns and drops that stratum, and the singlet
    guard is now one such failure. Unused categories are not a problem here:
    anndata prunes them when it builds the subset view, so only a stratum that
    genuinely contains one cell of a group is affected — and the surviving
    strata must still come back, which is the half a bare `pytest.warns` would
    not check.
    """
    adata = synthetic_adata.copy()
    strata = np.array(["s0", "s1"] * (adata.n_obs // 2), dtype=object)
    labels = adata.obs["batch"].astype(object).to_numpy()
    # Make group "A" a singleton inside s0 only: every other s0 A-cell becomes
    # a B, so s1 keeps a healthy A.
    s0_a = [i for i in range(adata.n_obs) if strata[i] == "s0" and labels[i] == "A"]
    assert len(s0_a) >= 2, "fixture must give s0 more than one A cell"
    for i in s0_a[1:]:
        labels[i] = "B"
    adata.obs["batch"] = pd.Categorical(labels, categories=["A", "B", "C"])
    adata.obs["stratum"] = pd.Categorical(strata)

    with pytest.warns(UserWarning, match=r"DE failed for stratum \[s0\].*one sample"):
        out = pyscx.accel.rank_genes_groups(
            adata,
            "batch",
            stratify_by=["stratum"],
            min_cells_per_stratum=1,
            device="cpu",
        )
    assert set(out["stratum"]) == {"s1"}, "the healthy stratum was dropped too"


# ---------------------------------------------------------------------------
# corr_method
# ---------------------------------------------------------------------------


def test_corr_method_accepts_only_benjamini_hochberg(synthetic_adata):
    rgg = _rgg(synthetic_adata.copy(), corr_method="benjamini-hochberg")
    assert rgg["params"]["corr_method"] == "benjamini-hochberg"
    with pytest.raises(ValueError, match="benjamini-hochberg"):
        _rgg(synthetic_adata.copy(), corr_method="bonferroni")
    with pytest.raises(ValueError, match="benjamini-hochberg"):
        _rgg(synthetic_adata.copy(), corr_method="fdr_bh")


# ---------------------------------------------------------------------------
# X6: the `pts` frames survive an SCX round trip.
# ---------------------------------------------------------------------------


def test_pts_frames_survive_a_from_anndata_round_trip(synthetic_adata, tmp_path):
    """`from_anndata` used to *refuse* the object it had just told pyscx to make.

    PR G made `rank_genes_groups(pts=True)` write two `pandas.DataFrame`s into
    `uns`, which every pyscx uns writer then rejected — scanpy's own `pts=True`
    output included. The documented workaround was to delete the two keys
    before writing. This is that workaround becoming unnecessary.

    `assert_frame_equal`, not `==`: comparing two frames with `==` yields a
    *frame*, so an `assert` on it raises rather than comparing.
    """
    adata = synthetic_adata.copy()
    _rgg(adata, pts=True)

    path = str(tmp_path / "pts.scx")
    pyscx.from_anndata(adata, path)
    back = pyscx.open(path).to_anndata()

    for key in ("pts", "pts_rest"):
        got = back.uns[KEY][key]
        assert isinstance(got, pd.DataFrame), f"{key} came back as {type(got).__name__}"
        pd.testing.assert_frame_equal(got, adata.uns[KEY][key], check_dtype=True)

    # Column order is the group order, carried explicitly rather than by JSON
    # object order, and the index is the var names `.loc` lookups need.
    assert list(back.uns[KEY]["pts"].columns) == list(adata.uns[KEY]["pts"].columns)
    assert list(back.uns[KEY]["pts"].index) == list(adata.var_names)


def test_round_tripped_pts_still_drives_the_scanpy_consumers(synthetic_adata, tmp_path):
    """The two scanpy functions that are the *reason* `pts` is a frame.

    `sc.tl.filter_rank_genes_groups` does `uns[key]["pts"][group].loc[var_names]`
    and `sc.get.rank_genes_groups_df` melts and merges the tables — both break
    on a dict. Running them after the round trip is what proves the envelope
    restored a frame rather than something frame-shaped.
    """
    sc = pytest.importorskip("scanpy")

    adata = synthetic_adata.copy()
    _rgg(adata, pts=True)
    path = str(tmp_path / "pts_consumers.scx")
    pyscx.from_anndata(adata, path)
    back = pyscx.open(path).to_anndata()

    group = str(back.uns[KEY]["pts"].columns[0])

    sc_df = sc.get.rank_genes_groups_df(back, group=group)
    assert {"pct_nz_group", "pct_nz_reference"} <= set(sc_df.columns)

    sc.tl.filter_rank_genes_groups(back, key=KEY, min_in_group_fraction=0.1)
    assert f"{KEY}_filtered" in back.uns

    # pyscx's own extractor reads the same two frames back out.
    px_df = pyscx.accel.rank_genes_groups_df(back, group=group)
    assert {"pct_nz_group", "pct_nz_reference"} <= set(px_df.columns)
