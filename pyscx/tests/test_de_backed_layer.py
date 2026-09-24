"""DE with `layer=` / a presentation-ordered gene axis, on a **backed** file.

`test_de_use_raw_layer.py` covers `use_raw=` and `layer=` on in-memory scipy
AnnData only — every fixture in it is `sp.random(...)`. So nothing covered the
case the kwarg exists for: a file opened backed, whose raw counts live in a
layer because `X` has been normalised. On such a file both DE entry points
raised, because `adata.layers[name]` is `ScxBackedLayerDataset` — a separate
`#[pyclass]` — and every backed dispatch site cast `ScxBackedSparseDataset`:

    rank_genes_groups(layer=): TypeError: ScxBackedLayerDataset is a lazy handle
        onto an SCX file and does not convert to a numpy array implicitly
    pdex_ref(layer=):          TypeError: ScxBackedLayerDataset.max() got an
        unexpected keyword argument 'out'

(`rank_genes_groups` fell past the sparse arm — `scipy.sparse.issparse(handle)`
is `False` — into the dense arm's `np.asarray`; `pdex_ref` died earlier, in
`is_log1p` auto-detection's `np.max`.)
`docs/scanpy/accel-differential-expression.md` documented the kwarg for both
with no such restriction.

The second half is a silent one. `rank_genes_groups_df` used the prologue
without the `preserve_var_order` guard, so on a presentation-ordered backed `X`
it computed over the sorted on-disk projection and labelled the result from the
request-ordered `adata.var`. Measured on a fixture where only `g2` separates the
groups, asking for `["g2", "g0"]`:

    target feature   fold_change            target feature   fold_change
    b      g0        9.000000e+10     vs     b      g2        9.000000e+10
    b      g2        1.000000e+00            b      g0        1.000000e+00
    (presentation-ordered: wrong)            (sorted request: correct)

`g2`'s effect reported under `g0`'s name, with no warning. The other streaming
accel ops had refused that input since they started streaming the handle's view;
DE had not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

anndata = pytest.importorskip("anndata")


@pytest.fixture
def paths(tmp_path):
    """A backed file whose `X` and `counts` layer are *different* matrices.

    Deliberately not a scalar multiple of each other: Wilcoxon ranks are
    invariant under a monotone transform, so `counts = 2 * X` would make "the
    layer result equals the layer oracle" pass even if the op silently read
    `X`. The premise test below asserts the two really do differ.
    """
    import pyscx

    rng = np.random.default_rng(7)
    n_obs, n_vars = 40, 6
    x = rng.integers(0, 40, size=(n_obs, n_vars)).astype(np.float32)
    counts = rng.integers(0, 40, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(
        {"grp": ["a"] * 20 + ["b"] * 20},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])

    adata = anndata.AnnData(X=sp.csr_matrix(x), obs=obs.copy(), var=var.copy())
    adata.layers["counts"] = sp.csr_matrix(counts)
    plain = str(tmp_path / "de_layer.scx")
    pyscx.from_anndata(adata, plain)

    with_csc = str(tmp_path / "de_layer_csc.scx")
    pyscx.from_anndata(adata, with_csc, csc="always", csc_cols_per_shard=3)

    # The oracle: an in-memory AnnData whose X *is* the counts layer.
    oracle = anndata.AnnData(
        X=sp.csr_matrix(counts), obs=obs.copy(), var=var.copy()
    )
    return {"plain": plain, "csc": with_csc, "oracle": oracle}


def _rgg_scores(adata):
    res = adata.uns["rank_genes_groups"]
    names = np.asarray(res["names"].tolist())
    scores = np.asarray(res["scores"].tolist())
    return names, scores


# ---------------------------------------------------------------------------
# layer= on a backed file
# ---------------------------------------------------------------------------


def test_premise_x_and_the_layer_give_different_de(paths):
    """Without this, every equality below could pass while reading `X`."""
    import pyscx

    on_x = pyscx.open(paths["plain"]).to_anndata(backed=True)
    pyscx.accel.rank_genes_groups(on_x, "grp", device="cpu")
    oracle = paths["oracle"].copy()
    pyscx.accel.rank_genes_groups(oracle, "grp", device="cpu")

    _, x_scores = _rgg_scores(on_x)
    _, layer_scores = _rgg_scores(oracle)
    assert not np.allclose(x_scores, layer_scores), (
        "X and the counts layer produce the same DE — the fixture cannot "
        "distinguish an op that reads the wrong matrix"
    )


def test_rank_genes_groups_layer_on_a_backed_file(paths):
    import pyscx

    backed = pyscx.open(paths["plain"]).to_anndata(backed=True)
    assert type(backed.layers["counts"]).__name__ == "ScxBackedLayerDataset", (
        "premise: the layer is the handle class the dispatch used to miss"
    )
    pyscx.accel.rank_genes_groups(backed, "grp", layer="counts", device="cpu")

    oracle = paths["oracle"].copy()
    pyscx.accel.rank_genes_groups(oracle, "grp", device="cpu")

    b_names, b_scores = _rgg_scores(backed)
    o_names, o_scores = _rgg_scores(oracle)
    assert b_names.tolist() == o_names.tolist()
    np.testing.assert_allclose(b_scores, o_scores, rtol=1e-6, atol=1e-6)
    assert backed.uns["rank_genes_groups"]["params"]["layer"] == "counts"


def test_pdex_ref_layer_on_a_backed_file(paths):
    """`is_log1p` is left to auto-detect on purpose — that is where it died."""
    import pyscx

    backed = pyscx.open(paths["plain"]).to_anndata(backed=True)
    got = pyscx.accel.pdex_ref(backed, "grp", reference="a", layer="counts", device="cpu")

    oracle = paths["oracle"].copy()
    want = pyscx.accel.pdex_ref(oracle, "grp", reference="a", device="cpu")

    assert list(got["feature"]) == list(want["feature"])
    np.testing.assert_allclose(
        got["fold_change"].to_numpy(), want["fold_change"].to_numpy(), rtol=1e-6
    )
    assert backed.uns["scx_accel"]["pdex_ref"]["layer"] == "counts"


def test_a_layer_is_not_routed_through_x_s_csc_sidecar(paths):
    """A CSC sidecar belongs to `X`; reading it for a layer reads the wrong data.

    The sidecar makes `prefer_format="auto"` pick CSC-direct for `X`. A layer
    has no sidecar of its own, so the layer request must stay on CSR — and,
    critically, must not borrow `X`'s. The route stamp and the value equality
    together say which matrix was actually read.
    """
    import pyscx

    on_x = pyscx.open(paths["csc"]).to_anndata(backed=True)
    pyscx.accel.rank_genes_groups(on_x, "grp", prefer_format="auto", device="cpu")
    assert on_x.uns["scx_accel"]["rank_genes_groups"]["route"] == "cpu_csc_nnz", (
        "premise: the file's sidecar really does drive X to the CSC route"
    )

    on_layer = pyscx.open(paths["csc"]).to_anndata(backed=True)
    pyscx.accel.rank_genes_groups(
        on_layer, "grp", layer="counts", prefer_format="auto", device="cpu"
    )
    assert on_layer.uns["scx_accel"]["rank_genes_groups"]["route"] == "cpu_csr"

    oracle = paths["oracle"].copy()
    pyscx.accel.rank_genes_groups(oracle, "grp", device="cpu")
    np.testing.assert_allclose(
        _rgg_scores(on_layer)[1], _rgg_scores(oracle)[1], rtol=1e-6, atol=1e-6
    )


def test_layer_with_a_deletion_vector(paths, tmp_path):
    """`mark_deleted` + `layer=`: the op sees the visible rows of the layer."""
    import pyscx

    dropped = {"c0", "c1", "c20", "c21"}
    exp = pyscx.open(paths["plain"])
    keep = np.array(
        [c not in dropped for c in exp.read_obs().index], dtype=bool
    )
    exp.mark_deleted(~keep)

    backed = pyscx.open(paths["plain"]).to_anndata(backed=True)
    assert backed.n_obs == 36, "premise: the deletion is visible"
    pyscx.accel.rank_genes_groups(backed, "grp", layer="counts", device="cpu")

    oracle = paths["oracle"][keep].copy()
    pyscx.accel.rank_genes_groups(oracle, "grp", device="cpu")
    np.testing.assert_allclose(
        _rgg_scores(backed)[1], _rgg_scores(oracle)[1], rtol=1e-6, atol=1e-6
    )


def test_layer_under_a_sorted_gene_projection(paths):
    """A column-projected backed layer: fewer genes, still the right ones."""
    import pyscx

    backed = pyscx.open(paths["plain"]).to_anndata(
        backed=True, var_names=["g1", "g3", "g4"]
    )
    pyscx.accel.rank_genes_groups(backed, "grp", layer="counts", device="cpu")

    oracle = paths["oracle"][:, ["g1", "g3", "g4"]].copy()
    pyscx.accel.rank_genes_groups(oracle, "grp", device="cpu")
    assert _rgg_scores(backed)[0].tolist() == _rgg_scores(oracle)[0].tolist()
    np.testing.assert_allclose(
        _rgg_scores(backed)[1], _rgg_scores(oracle)[1], rtol=1e-6, atol=1e-6
    )


def test_unknown_layer_names_the_layer(paths):
    """The message the three sibling ops copied by name, finally pinned."""
    import pyscx

    backed = pyscx.open(paths["plain"]).to_anndata(backed=True)
    with pytest.raises(ValueError, match=r"layer 'nope' not found"):
        pyscx.accel.rank_genes_groups(backed, "grp", layer="nope", device="cpu")
    with pytest.raises(ValueError, match=r"layer 'nope' not found"):
        pyscx.accel.pdex_ref(backed, "grp", reference="a", layer="nope", device="cpu")


# ---------------------------------------------------------------------------
# Presentation-ordered gene axis
# ---------------------------------------------------------------------------


@pytest.fixture
def ordered(tmp_path):
    """Only `g2` separates the groups, so a mislabel is unambiguous."""
    import pyscx

    dense = np.zeros((20, 3), dtype=np.float32)
    dense[:, 0] = 1.0
    dense[:10, 1] = 50.0
    dense[10:, 2] = 90.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame({"grp": ["a"] * 10 + ["b"] * 10},
                         index=[f"c{i}" for i in range(20)]),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )
    adata.layers["counts"] = sp.csr_matrix(dense)
    path = str(tmp_path / "ordered.scx")
    pyscx.from_anndata(adata, path)
    return path


def _de_ops():
    """Every DE entry point, so a guard added to one and not the others is red."""
    import pyscx

    return [
        pytest.param(
            lambda a, **kw: pyscx.accel.rank_genes_groups(a, "grp", device="cpu", **kw),
            id="rank_genes_groups",
        ),
        pytest.param(
            lambda a, **kw: pyscx.accel.pdex_ref(
                a, "grp", reference="a", device="cpu", **kw
            ),
            id="pdex_ref",
        ),
        pytest.param(
            lambda a, **kw: pyscx.accel.rank_genes_groups_df(
                a, "grp", reference="a", device="cpu", **kw
            ),
            id="rank_genes_groups_df",
        ),
    ]


@pytest.mark.parametrize("op", _de_ops())
def test_a_presentation_ordered_x_is_refused(ordered, op):
    """`rank_genes_groups_df` is the one that used to return wrong labels."""
    import pyscx

    adata = pyscx.open(ordered).to_anndata(
        backed=True, var_names=["g2", "g0"], preserve_var_order=True
    )
    assert list(adata.var_names) == ["g2", "g0"], "premise: request order kept"
    with pytest.raises(RuntimeError, match=r"caller-requested order"):
        op(adata)


@pytest.mark.parametrize("op", _de_ops()[:2])
def test_a_presentation_ordered_layer_is_refused(ordered, op):
    """Materialising `X` disarms the `adata.X`-only guard; the layer is not."""
    import pyscx

    adata = pyscx.open(ordered).to_anndata(
        backed=True, var_names=["g2", "g0"], preserve_var_order=True
    )
    adata.X = adata.X.to_memory()
    with pytest.raises(RuntimeError, match=r"caller-requested order"):
        op(adata, layer="counts")


@pytest.mark.parametrize("op", _de_ops())
def test_a_sorted_request_is_still_allowed(ordered, op):
    """The accept side: a sorted projection is not a presentation order.

    Without this the guard could refuse every backed DE call and the two tests
    above would still pass.
    """
    import pyscx

    adata = pyscx.open(ordered).to_anndata(backed=True, var_names=["g0", "g2"])
    assert list(adata.var_names) == ["g0", "g2"], "premise: sorted request"
    op(adata)


def test_a_sorted_request_labels_the_right_gene(ordered):
    """Pins the value the mislabel got wrong, on all three entry points.

    `g2` is the only gene that separates the groups, so it must come back as
    group `b`'s top gene / largest effect. Refusing the presentation-ordered
    input is only useful if the input that *is* accepted is labelled correctly.
    """
    import pyscx

    def fresh():
        return pyscx.open(ordered).to_anndata(backed=True, var_names=["g0", "g2"])

    a = fresh()
    pyscx.accel.rank_genes_groups(a, "grp", device="cpu")
    names = a.uns["rank_genes_groups"]["names"]
    assert names.dtype.names == ("a", "b"), "premise: both groups reported"
    assert names["b"][0] == "g2"

    df = pyscx.accel.pdex_ref(fresh(), "grp", reference="a", device="cpu")
    best = df.iloc[df["log2_fold_change"].abs().to_numpy().argmax()]
    assert best["feature"] == "g2"

    out = pyscx.accel.rank_genes_groups_df(
        fresh(), "grp", reference="a", device="cpu"
    )
    top = out.iloc[out["abs_log2_fold_change"].to_numpy().argmax()]
    assert top["feature"] == "g2"


def test_the_documented_remedy_works_on_an_ordered_x(ordered):
    """Materialising the matrix the op reads must actually let it run.

    The refusal tells the caller to materialise "the matrix the op reads", and
    `docs/api/python-experiment.md` says the guard covers that matrix rather than `adata.X`. Both
    were false while the entry prologue ran the X-only check *before*
    `select_de_matrix` chose the layer: with a presentation-ordered `X`, the
    caller could materialise `counts`, ask for `layer="counts"`, and still be
    refused for a matrix the op was not going to read. The prologue is now the
    no-var-guard variant and `select_de_matrix` is the single guard.

    The accept side is the half the first version of this file missed: it pinned
    only the opposite arrangement (materialised `X` plus an *ordered* layer,
    which must still refuse).
    """
    import pyscx

    def ordered_handle():
        return pyscx.open(ordered).to_anndata(
            backed=True, var_names=["g2", "g0"], preserve_var_order=True
        )

    a = ordered_handle()
    a.layers["counts"] = a.layers["counts"].to_memory()
    pyscx.accel.rank_genes_groups(a, "grp", layer="counts", device="cpu")
    assert a.uns["rank_genes_groups"]["names"]["b"][0] == "g2", (
        "the materialised layer must also be labelled correctly, not merely "
        "accepted — its var order is the request order"
    )

    b = ordered_handle()
    b.layers["counts"] = b.layers["counts"].to_memory()
    got = pyscx.accel.pdex_ref(b, "grp", reference="a", layer="counts", device="cpu")
    best = got.iloc[got["log2_fold_change"].abs().to_numpy().argmax()]
    assert best["feature"] == "g2"

    # And the guard has not simply been deleted: reading the ordered `X`
    # itself, or an ordered layer, still refuses.
    with pytest.raises(RuntimeError, match=r"caller-requested order"):
        pyscx.accel.rank_genes_groups(ordered_handle(), "grp", device="cpu")
    still_ordered = ordered_handle()
    still_ordered.X = still_ordered.X.to_memory()
    with pytest.raises(RuntimeError, match=r"caller-requested order"):
        pyscx.accel.rank_genes_groups(
            still_ordered, "grp", layer="counts", device="cpu"
        )
