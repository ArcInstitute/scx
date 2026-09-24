"""Accelerators handed an anndata *view* must not crash, and must not gather X.

Companion to `test_accel_axis_view.py`. That file covers the *handle's* window
(`accel.subset_var` / `filter_cells`, which never produce an `AnnData` view).
This one covers plain anndata indexing — `adata[:, mask]` — which
`docs/scanpy/accelerators.md` advertises as supported on a backed `X`, and
which is the first thing a scanpy user reaches for after
`highly_variable_genes`.

Handed a view, an accelerator wants to write its result back. Every write on a
view goes through anndata's copy-on-write, and copy-on-write is
`adata.copy()` → `_subset(ref.X, idx).copy()` → the whole matrix in RAM. Two
failures came out of that, both found by dogfooding v0.11.8 on a 500k-cell
Census slice:

* **Loud**, and only after a prior op had already stamped `uns["scx_accel"]`:
  the route stamp is then a *nested* dict write, which does not trigger
  copy-on-write, so the object stayed a view and `adata.X = <lazy>` died inside
  anndata with `'ScxBackedSparseDataset' object does not support item
  assignment`. That is why every test here that reproduces it runs
  `highly_variable_genes` first — without it the op "worked", by accident.
* **Silent**, everywhere else: the op succeeded *because* copy-on-write had
  gathered X. Measured on census_500k, `normalize_total` on a 500k × 3k view
  took peak RSS from 1.8 GB to 10.3 GB — out-of-core in name only.

Both are fixed by rebuilding the view as an actual `AnnData` up front, keeping
the handle lazy (`accel::prepare_target`). So the assertions come in pairs: the
op must not raise, **and** `type(adata.X)` must still be an SCX handle.

Multi-shard fixtures throughout — a single-shard file cannot exhibit a
row-window bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

N_OBS = 120
N_VARS = 60
SHARD_SIZE = 25  # → 5 shards

ROW_KEEP = np.zeros(N_OBS, dtype=bool)
ROW_KEEP[np.arange(2, N_OBS, 2)] = True  # 59 cells, spans every shard

COL_KEEP = np.zeros(N_VARS, dtype=bool)
COL_KEEP[np.arange(0, N_VARS, 2)] = True  # 30 genes, ascending

SCX_X_TYPES = ("ScxBackedSparseDataset", "ScxLazyTransformedDataset")


@pytest.fixture(scope="module")
def counts():
    """Poisson counts with lognormal per-gene means.

    Not `sp.random`: seurat_v3 fits a loess of log-variance on log-mean, and a
    uniform-random matrix puts every gene at the same mean with no spread, so
    the fit is singular and HVG raises before it can exercise anything this
    file is about. A lognormal mean per gene gives a well-conditioned curve —
    and is what real counts look like.
    """
    rng = np.random.default_rng(11)
    means = rng.lognormal(mean=0.5, sigma=1.2, size=N_VARS).astype(np.float32)
    dense = rng.poisson(lam=np.broadcast_to(means, (N_OBS, N_VARS))).astype(np.float32)
    x = sp.csr_matrix(dense)
    x.eliminate_zeros()
    return x


@pytest.fixture(scope="module")
def obs_cols():
    pert = np.array(
        ["ctrl" if (i // 2) % 2 == 0 else "drug" for i in range(N_OBS)], dtype=object
    )
    donor = np.array([f"d{i % 3}" for i in range(N_OBS)], dtype=object)
    return pert, donor


@pytest.fixture(scope="module")
def var_names():
    return [f"g{i}" for i in range(N_VARS)]


def _build(counts, obs_cols, var_names):
    import anndata as ad
    import pandas as pd

    pert, donor = obs_cols
    adata = ad.AnnData(X=counts.copy())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.obs["pert"] = pd.Categorical(list(pert))
    adata.obs["donor"] = pd.Categorical(list(donor))
    adata.var_names = list(var_names)
    return adata


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory, counts, obs_cols, var_names):
    adata = _build(counts, obs_cols, var_names)
    path = tmp_path_factory.mktemp("accel_on_view") / "counts.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


def _backed(path):
    return pyscx.open(str(path)).to_anndata(backed=True)


def _seed(adata, needs):
    """Put `X_pca` / the kNN graph on the **parent**, before it is sliced.

    Seeding must not happen on the view. `pca` and `neighbors` are themselves
    write-back ops, so running them on the view de-views it, and the op under
    test would then receive an ordinary in-memory-backed AnnData — where
    writing `obs` / `obsm` / `obsp` cannot gather a lazy `X` no matter what.
    The test would pass whether or not the op had the prologue.

    That is not hypothetical: the first cut of this file seeded on the view,
    and with `compute_lisi`'s prologue deleted the whole parametrized test
    still passed 34/34.
    """
    if needs in ("pca", "neighbors"):
        pyscx.accel.pca(adata, n_comps=5, device="cpu")
    if needs == "neighbors":
        pyscx.accel.neighbors(adata, n_neighbors=5, use_rep="X_pca", device="cpu")


def _hvg_view(path, n_top_genes=10):
    """The exact dogfood repro shape: backed → HVG → gene-subset view.

    The prior HVG matters: it creates `uns["scx_accel"]`, which is what stops
    the later route stamp from accidentally triggering copy-on-write.
    """
    adata = _backed(path)
    pyscx.accel.highly_variable_genes(
        adata, n_top_genes=n_top_genes, flavor="seurat_v3", device="cpu"
    )
    view = adata[:, adata.var["highly_variable"].values]
    assert view.is_view, "fixture precondition: plain indexing must give a view"
    return adata, view


# --------------------------------------------------------------------------
# B1 — the reported crash
# --------------------------------------------------------------------------


def test_hvg_then_gene_subset_then_normalize_total(scx_path, counts):
    """The dogfood repro. Pre-fix: TypeError from anndata's view X-setter."""
    parent, view = _hvg_view(scx_path)
    hv = parent.var["highly_variable"].values

    pyscx.accel.normalize_total(view, target_sum=1e4, device="cpu")

    assert not view.is_view, "the op must leave the caller an actual AnnData"
    assert type(view.X).__name__ == "ScxLazyTransformedDataset"

    expected = counts[:, hv].toarray()
    totals = expected.sum(axis=1, keepdims=True)
    expected = expected / np.where(totals == 0, 1.0, totals) * 1e4
    np.testing.assert_allclose(view.X.to_memory().toarray(), expected, rtol=1e-5)


def test_hvg_then_gene_subset_then_log1p(scx_path):
    parent, view = _hvg_view(scx_path)
    pyscx.accel.normalize_total(view, target_sum=1e4, device="cpu")
    pyscx.accel.log1p(view, device="cpu")
    assert type(view.X).__name__ == "ScxLazyTransformedDataset"


def test_normalize_total_on_a_view_without_a_prior_stamp_stays_lazy(scx_path):
    """No prior HVG — the branch that 'worked' by gathering X.

    Pre-fix this passed with `type(X) is csr_matrix`: anndata's copy-on-write
    had materialized the matrix. That silent gather is the 1.8 → 10.3 GB
    regression, so the type assertion is the whole point of the test.
    """
    adata = _backed(scx_path)
    view = adata[:, COL_KEEP]
    assert view.is_view

    pyscx.accel.normalize_total(view, target_sum=1e4, device="cpu")

    assert type(view.X).__name__ == "ScxLazyTransformedDataset"


def test_pca_on_a_view_does_not_materialize_x(scx_path):
    adata = _backed(scx_path)
    view = adata[:, COL_KEEP]

    pyscx.accel.pca(view, n_comps=5, device="cpu")

    assert "X_pca" in view.obsm
    assert view.obsm["X_pca"].shape == (N_OBS, 5)
    assert type(view.X).__name__ == "ScxBackedSparseDataset"


def test_results_do_not_leak_into_the_parent(scx_path):
    """A nested `uns` write on a view lands in the *parent's* dict.

    `DictView` is a shallow copy, so `uns["scx_accel"]` is the same object the
    parent holds. Pre-fix, an op on a view stamped the parent and left the view
    with nothing.
    """
    adata = _backed(scx_path)
    view = adata[:, COL_KEEP]

    pyscx.accel.pca(view, n_comps=5, device="cpu")

    assert "pca" in view.uns.get("scx_accel", {}), "the stamp belongs on the view"
    assert "pca" not in adata.uns.get("scx_accel", {}), "parent must be untouched"
    assert "X_pca" not in adata.obsm
    assert adata.n_vars == N_VARS


def test_second_op_on_the_same_view_still_works(scx_path):
    """Once de-viewed, further ops are ordinary — and both stamps survive."""
    adata = _backed(scx_path)
    view = adata[:, COL_KEEP]

    pyscx.accel.normalize_total(view, target_sum=1e4, device="cpu")
    pyscx.accel.log1p(view, device="cpu")

    assert {"normalize_total", "log1p"} <= set(view.uns["scx_accel"])
    assert type(view.X).__name__ == "ScxLazyTransformedDataset"


@pytest.mark.parametrize(
    ("op", "needs"),
    [
        pytest.param(lambda a: pyscx.accel.normalize_total(a, device="cpu"), None, id="normalize_total"),
        pytest.param(lambda a: pyscx.accel.log1p(a, device="cpu"), None, id="log1p"),
        pytest.param(lambda a: pyscx.accel.calculate_qc_metrics(a), None, id="calculate_qc_metrics"),
        pytest.param(
            lambda a: pyscx.accel.highly_variable_genes(
                a, n_top_genes=5, flavor="seurat_v3", device="cpu"
            ),
            None,
            id="highly_variable_genes",
        ),
        pytest.param(lambda a: pyscx.accel.pca(a, n_comps=4, device="cpu"), None, id="pca"),
        pytest.param(
            lambda a: pyscx.accel.rank_genes_groups(
                a, groupby="pert", method="wilcoxon", device="cpu"
            ),
            None,
            id="rank_genes_groups",
        ),
        pytest.param(
            lambda a: pyscx.accel.pdex_ref(a, groupby="pert", reference="ctrl", device="cpu"),
            None,
            id="pdex_ref",
        ),
        pytest.param(lambda a: pyscx.accel.filter_genes(a, min_cells=1), None, id="filter_genes"),
        pytest.param(lambda a: pyscx.accel.filter_cells(a, min_genes=1), None, id="filter_cells"),
        pytest.param(
            lambda a: pyscx.accel.score_genes(a, gene_list=["g0", "g2", "g4"], device="cpu"),
            None,
            id="score_genes",
        ),
        pytest.param(lambda a: pyscx.accel.pflog(a, n_components=5), None, id="pflog"),
        pytest.param(
            lambda a: pyscx.accel.pseudobulk_means(a, "pert"), None, id="pseudobulk_means"
        ),
        # obsm / obsp consumers — prerequisites are seeded on the PARENT (see
        # `_seed`), so the op under test is the first thing to touch the view.
        pytest.param(
            lambda a: pyscx.accel.neighbors(a, n_neighbors=5, device="cpu"),
            "pca",
            id="neighbors",
        ),
        pytest.param(lambda a: pyscx.accel.umap(a, device="cpu"), "neighbors", id="umap"),
        pytest.param(lambda a: pyscx.accel.leiden(a, device="cpu"), "neighbors", id="leiden"),
        pytest.param(
            lambda a: pyscx.accel.harmony_integrate(a, "donor"), "pca", id="harmony_integrate"
        ),
        pytest.param(
            lambda a: pyscx.accel.compute_lisi(a, key="donor", perplexity=5.0),
            "neighbors",
            id="compute_lisi",
        ),
        pytest.param(
            lambda a: pyscx.accel.pseudobulk_dex(
                a, groupby=["pert", "donor"], test_col="pert", reference="ctrl"
            ),
            None,
            id="pseudobulk_dex",
        ),
    ],
)
@pytest.mark.parametrize("axis", ["var", "obs"])
def test_write_back_ops_accept_a_view_without_gathering(scx_path, op, needs, axis):
    """Locks the prologue's coverage list.

    An op missing the prologue shows up here either as a raise or as an X that
    stopped being an SCX handle. `compute_lisi` is in this list because it was
    the one write-back op the first cut of the prologue missed — it *returns*
    the LISI vector, which reads as a pure function, but also writes
    `obs["lisi_<key>"]`.
    """
    adata = _backed(scx_path)
    _seed(adata, needs)
    view = adata[:, COL_KEEP] if axis == "var" else adata[ROW_KEEP]
    assert view.is_view, "the op under test must be the first thing to touch the view"

    op(view)

    assert type(view.X).__name__ in SCX_X_TYPES, (
        f"X was gathered to {type(view.X).__name__}; the op is missing prepare_target"
    )


def test_composed_row_and_col_view(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    view = adata[ROW_KEEP, COL_KEEP]

    pyscx.accel.normalize_total(view, target_sum=1e4, device="cpu")

    assert type(view.X).__name__ == "ScxLazyTransformedDataset"
    expected = counts[ROW_KEEP, :][:, COL_KEEP].toarray()
    totals = expected.sum(axis=1, keepdims=True)
    expected = expected / np.where(totals == 0, 1.0, totals) * 1e4
    np.testing.assert_allclose(view.X.to_memory().toarray(), expected, rtol=1e-5)


def test_deview_emits_implicit_modification_warning(scx_path):
    """The notice is one-shot *per op, per process*.

    So this must use an op no other test in this file de-views, or it passes or
    fails on collection order. `pca_neighbors` is the only write-back op used
    nowhere else here — do not add it to the parametrize list above.
    """
    import anndata as ad

    adata = _backed(scx_path)
    view = adata[:, COL_KEEP]
    with pytest.warns(ad.ImplicitModificationWarning, match="No data was copied"):
        pyscx.accel.pca_neighbors(view, n_comps=5, n_neighbors=5, device="cpu")


# --------------------------------------------------------------------------
# Carve-outs: cases the prologue deliberately declines
# --------------------------------------------------------------------------


def test_plain_scipy_view_is_left_to_anndata(counts, obs_cols, var_names):
    """No SCX handle to protect → anndata's own copy-on-write is correct.

    The assertion that matters is the *parent's* `uns`: the route stamp must
    still go through a top-level `uns["scx_accel"] = …` so copy-on-write fires,
    rather than mutating the nested dict the view shares with its parent.
    """
    adata = _build(counts, obs_cols, var_names)
    view = adata[:, COL_KEEP]

    pyscx.accel.pca(view, n_comps=4, device="cpu")

    assert "X_pca" in view.obsm
    assert "scx_accel" not in adata.uns, "the parent must not be stamped"


def test_non_expressible_view_index_materializes_without_raising(scx_path):
    """A duplicated index has no window representation, so `view.X` is already
    scipy. The rebuild installs it — exactly what anndata would do. The
    contract being pinned is 'no TypeError', not 'stays lazy'."""
    adata = _backed(scx_path)
    view = adata[[2, 2, 7]]

    pyscx.accel.normalize_total(view, target_sum=1e4, device="cpu")

    assert view.n_obs == 3
    assert not view.is_view
