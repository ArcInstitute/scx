"""Streaming accelerators must run on the backed handle's *view*.

Companion to `test_pca_axis_view.py`, for the ops that *consume* an obs- or
var-aligned array rather than producing one: pseudobulk aggregation,
`rank_genes_groups`, `pdex_ref`, and `pseudobulk_means` (the perturbation
evaluation path).

These took the same wrong turn as `pca` — `&*backed.backed`, the file rather
than the window. Each is handed `groups` / `obs_groups` (one entry per
**visible** cell) and `gene_names` (one per **visible** gene). Measured against
`b4616630`, that produced two different failures:

* **Loud** wherever a length is checked against the source — a row subset gave
  `InvalidInput: groups length 59 != n_obs 120`, and a var subset gave
  `gene_names has 15 entries but n_vars = 30` from pseudobulk's
  `validate_inputs`. Still broken (`filter_cells` then `pseudobulk_means` is an
  ordinary thing to want), just not silent.
* **Silent** for `rank_genes_groups` / `pdex_ref` after a *var* subset, which
  is why `test_rank_genes_groups_after_var_subset` matters. Those kernels take
  `n_vars` from `gene_names.len()`, not from the source, so the guard passes —
  and then they chunk columns `0..15` of a 30-gene reader. The result came back
  fully formed, describing on-disk genes 0–14 under the names of visible genes
  {0, 2, 4, …}. Same index-space confusion as `pca`'s `mask_var` case.

The kernels are now generic over `ShardSource` and the dispatch hands them the
handle's view, so these work and agree with the materialized answer.

Multi-shard fixtures throughout — a single-shard file cannot exhibit a
row-window bug.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import pyscx

N_OBS = 120
N_VARS = 30
SHARD_SIZE = 25  # → 5 shards

# Not a prefix, and spans every shard, so a raw-reader read is distinguishable.
ROW_KEEP = np.zeros(N_OBS, dtype=bool)
ROW_KEEP[np.arange(2, N_OBS, 2)] = True  # 59 cells

COL_KEEP = np.zeros(N_VARS, dtype=bool)
COL_KEEP[np.arange(0, N_VARS, 2)] = True  # 15 genes, ascending


@pytest.fixture(scope="module")
def counts():
    rng = np.random.default_rng(7)
    x = sp.random(
        N_OBS, N_VARS, density=0.5, format="csr", dtype=np.float32, random_state=rng
    )
    x.data = (x.data * 30).astype(np.float32).round() + 1.0
    x.eliminate_zeros()
    return x


@pytest.fixture(scope="module")
def obs_cols():
    """`pert` (2 levels) for DE, plus `donor` so pseudobulk_dex has replicates.

    `pert` alternates on *pairs* rather than on parity so `ROW_KEEP` (every
    other cell) still spans both levels — with plain parity the subset is all
    `ctrl` and DE has nothing to contrast.
    """
    pert = np.array(
        ["ctrl" if (i // 2) % 2 == 0 else "drug" for i in range(N_OBS)], dtype=object
    )
    donor = np.array([f"d{i % 3}" for i in range(N_OBS)], dtype=object)
    return pert, donor


def _annotate(adata, pert, donor, var_names):
    import pandas as pd

    adata.obs["pert"] = pd.Categorical(list(pert))
    adata.obs["donor"] = pd.Categorical(list(donor))
    adata.var_names = list(var_names)
    return adata


@pytest.fixture(scope="module")
def var_names():
    return [f"g{i}" for i in range(N_VARS)]


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory, counts, obs_cols, var_names):
    import anndata as ad

    pert, donor = obs_cols
    adata = ad.AnnData(X=counts.copy())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    _annotate(adata, pert, donor, var_names)
    path = tmp_path_factory.mktemp("accel_axis_view") / "counts.scx"
    pyscx.from_anndata(adata, str(path), shard_size=SHARD_SIZE)
    return path


@pytest.fixture(scope="module")
def csc_path(tmp_path_factory, counts, obs_cols, var_names):
    import anndata as ad

    pert, donor = obs_cols
    adata = ad.AnnData(X=counts.copy())
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    _annotate(adata, pert, donor, var_names)
    path = tmp_path_factory.mktemp("accel_axis_view_csc") / "csc.scx"
    pyscx.from_anndata(
        adata, str(path), shard_size=SHARD_SIZE, csc="always", csc_cols_per_shard=8
    )
    return path


def _backed(path):
    return pyscx.open(str(path)).to_anndata(backed=True)


def _reference(counts, obs_cols, var_names, rows=None, cols=None):
    """The materialized equivalent of the same subset — the ground truth."""
    import anndata as ad

    pert, donor = obs_cols
    x = counts
    names = list(var_names)
    if rows is not None:
        x, pert, donor = x[rows, :], pert[rows], donor[rows]
    if cols is not None:
        x = x[:, cols]
        names = [n for n, k in zip(names, cols) if k]
    a = ad.AnnData(X=x.copy())
    return _annotate(a, pert, donor, names)


# ---------------------------------------------------------------------------
# pseudobulk_means — the eval-metrics aggregation path
# ---------------------------------------------------------------------------


def test_pseudobulk_means_after_row_subset(scx_path, counts, obs_cols, var_names):
    """Was `InvalidInput: obs_groups[0] length 59 != n_obs 120`."""
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    assert adata.n_obs == int(ROW_KEEP.sum())
    assert type(adata.X).__name__ == "ScxBackedSparseDataset"

    means, names = pyscx.accel.pseudobulk_means(adata, "pert", device="cpu")
    ref_means, ref_names = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names, rows=ROW_KEEP), "pert", device="cpu"
    )
    assert names == ref_names
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


def test_pseudobulk_means_after_var_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_var(adata, COL_KEEP)

    means, names = pyscx.accel.pseudobulk_means(adata, "pert", device="cpu")
    ref_means, ref_names = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names, cols=COL_KEEP), "pert", device="cpu"
    )
    assert means.shape == ref_means.shape == (2, int(COL_KEEP.sum()))
    assert names == ref_names
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


def test_pseudobulk_means_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)

    means, _ = pyscx.accel.pseudobulk_means(adata, "pert", device="cpu")
    ref_means, _ = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP), "pert", device="cpu"
    )
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


def test_pseudobulk_means_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    """The common path must be untouched by the source swap."""
    means, _ = pyscx.accel.pseudobulk_means(_backed(scx_path), "pert", device="cpu")
    ref_means, _ = pyscx.accel.pseudobulk_means(
        _reference(counts, obs_cols, var_names), "pert", device="cpu"
    )
    np.testing.assert_allclose(means, ref_means, rtol=1e-6)


# ---------------------------------------------------------------------------
# pseudobulk_dex — the `scx_accel::pseudobulk_aggregate` path
# ---------------------------------------------------------------------------


def _dex(adata):
    return pyscx.accel.pseudobulk_dex(
        adata,
        groupby=["pert", "donor"],
        test_col="pert",
        reference="ctrl",
        min_cells_per_group=1,
        backend="nb_glm",
    )


def _assert_dex_equal(got, ref, n_expected):
    """Compare the DE *statistics*, not just the gene labels.

    Gene names come from `adata.var_names`, so a name-only assertion is
    identical whichever physical columns were read — it cannot detect a wrong
    -column read, which is the whole failure mode under test.
    """
    assert len(got) == len(ref) == n_expected
    gene_col = "gene" if "gene" in got.columns else got.columns[0]
    assert list(got[gene_col]) == list(ref[gene_col])
    numeric = [
        c
        for c in got.columns
        if c != gene_col and got[c].dtype.kind in "fi" and ref[c].dtype.kind in "fi"
    ]
    assert numeric, f"expected numeric DE columns to compare, got {list(got.columns)}"
    for col in numeric:
        np.testing.assert_allclose(
            got[col].to_numpy().astype(np.float64),
            ref[col].to_numpy().astype(np.float64),
            rtol=1e-6,
            atol=1e-9,
            err_msg=f"pseudobulk_dex column {col!r} differs from the materialized run",
        )


def test_pseudobulk_dex_after_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)

    got = _dex(adata)
    ref = _dex(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP))
    _assert_dex_equal(got, ref, n_expected=int(COL_KEEP.sum()))


def test_pseudobulk_dex_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    _assert_dex_equal(
        _dex(_backed(scx_path)),
        _dex(_reference(counts, obs_cols, var_names)),
        n_expected=N_VARS,
    )


# ---------------------------------------------------------------------------
# rank_genes_groups (Wilcoxon) and pdex_ref
# ---------------------------------------------------------------------------


def _rgg(adata, **kw):
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", device="cpu", **kw
    )
    return pyscx.accel.rank_genes_groups_df(adata, group="drug")


def _assert_de_equal(got, ref):
    assert list(got["names"]) == list(ref["names"])
    np.testing.assert_allclose(
        got["scores"].to_numpy().astype(np.float64),
        ref["scores"].to_numpy().astype(np.float64),
        atol=1e-5,
    )


def test_rank_genes_groups_after_row_subset(scx_path, counts, obs_cols, var_names):
    """Was `InvalidInput: groups length 59 != n_obs 120`."""
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    _assert_de_equal(
        _rgg(adata), _rgg(_reference(counts, obs_cols, var_names, rows=ROW_KEEP))
    )


def test_rank_genes_groups_after_var_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_var(adata, COL_KEEP)
    got = _rgg(adata)
    assert len(got) == int(COL_KEEP.sum())
    _assert_de_equal(got, _rgg(_reference(counts, obs_cols, var_names, cols=COL_KEEP)))


def test_rank_genes_groups_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)
    _assert_de_equal(
        _rgg(adata),
        _rgg(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP)),
    )


def test_rank_genes_groups_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    _assert_de_equal(_rgg(_backed(scx_path)), _rgg(_reference(counts, obs_cols, var_names)))


def _pdex(adata):
    return pyscx.accel.pdex_ref(adata, "pert", reference="ctrl", device="cpu")


def test_pdex_ref_after_composed_subset(scx_path, counts, obs_cols, var_names):
    adata = _backed(scx_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)

    got = _pdex(adata)
    ref = _pdex(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP))

    assert list(got["feature"]) == list(ref["feature"])
    for col in ("target_mean", "ref_mean", "log2_fold_change", "p_value"):
        np.testing.assert_allclose(
            got[col].to_numpy().astype(np.float64),
            ref[col].to_numpy().astype(np.float64),
            atol=1e-6,
        )


def test_pdex_ref_unsubset_is_unchanged(scx_path, counts, obs_cols, var_names):
    got = _pdex(_backed(scx_path))
    ref = _pdex(_reference(counts, obs_cols, var_names))
    assert list(got["feature"]) == list(ref["feature"])
    np.testing.assert_allclose(
        got["log2_fold_change"].to_numpy().astype(np.float64),
        ref["log2_fold_change"].to_numpy().astype(np.float64),
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# preserve_var_order — must be refused, not silently permuted
# ---------------------------------------------------------------------------
#
# `as_shard_source()` emits columns in sorted on-disk order; under
# `preserve_var_order=True` `adata.var` (and so `gene_names` / the result
# columns) is in *request* order. Before these ops streamed the view, the
# widths disagreed and most of them failed loudly on a length guard. Streaming
# the view makes the widths *match*, which would turn that into a silent
# gene↔column permutation — so every op that consumes the view has to refuse a
# presentation-ordered handle, exactly as `pca` / HVG / `score_genes` do.


@pytest.fixture(scope="module")
def request_ordered(scx_path):
    """A backed handle whose gene axis is in caller-request order."""

    def _open():
        names = ["g9", "g2", "g10", "g5"]  # deliberately unsorted
        a = pyscx.open(str(scx_path)).to_anndata(
            backed=True, var_names=names, preserve_var_order=True
        )
        assert list(a.var_names) == names
        return a

    return _open


@pytest.mark.parametrize(
    "op",
    [
        pytest.param(
            lambda a: pyscx.accel.rank_genes_groups(
                a, groupby="pert", method="wilcoxon", device="cpu"
            ),
            id="rank_genes_groups",
        ),
        pytest.param(
            lambda a: pyscx.accel.pdex_ref(a, "pert", reference="ctrl", device="cpu"),
            id="pdex_ref",
        ),
        pytest.param(
            lambda a: pyscx.accel.pseudobulk_means(a, "pert", device="cpu"),
            id="pseudobulk_means",
        ),
        pytest.param(lambda a: _dex(a), id="pseudobulk_dex"),
    ],
)
def test_preserve_var_order_is_refused(request_ordered, op):
    with pytest.raises(RuntimeError, match="preserve_var_order"):
        op(request_ordered())


def test_sorted_projection_is_not_refused(scx_path, counts, obs_cols, var_names):
    """The guard must only catch a genuine request-order axis.

    `set_col_projection_ordered` leaves `col_presentation` unset when the
    permutation is the identity, so an ordinary ascending subset — including
    `var_names=` listed in sorted order — must still run.
    """
    a = pyscx.open(str(scx_path)).to_anndata(
        backed=True, var_names=["g2", "g5", "g9", "g10"], preserve_var_order=True
    )
    means, _ = pyscx.accel.pseudobulk_means(a, "pert", device="cpu")
    assert means.shape == (2, 4)

    b = _backed(scx_path)
    pyscx.accel.subset_var(b, COL_KEEP)
    pyscx.accel.pseudobulk_means(b, "pert", device="cpu")  # must not raise


# ---------------------------------------------------------------------------
# prefer_format="csc" on a subset — served, where it used to be refused
# ---------------------------------------------------------------------------
#
# Both of these asserted a `RuntimeError` naming its cause, and both causes
# were real while the backed handle's column source was the full-axis
# `BackedCscReader`: a row subset renumbers the live rows that the sidecar's
# `indices` do not know about, and a column subset left the kernel comparing
# `gene_names length 15 != source.n_vars() 30` (which is what the named
# refusal was introduced to replace). The handle now hands back a view that
# renumbers rows and remaps columns, so both are served — and the assertion
# that carries the weight is no longer the message but the answer.


def test_csc_on_row_subset_matches_the_reference(csc_path, counts, obs_cols, var_names):
    adata = _backed(csc_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    _assert_de_equal(
        _rgg(adata, prefer_format="csc"),
        _rgg(_reference(counts, obs_cols, var_names, rows=ROW_KEEP)),
    )
    assert _de_route(adata) == "cpu_csc_nnz"


def test_csc_on_var_subset_matches_the_reference(csc_path, counts, obs_cols, var_names):
    adata = _backed(csc_path)
    pyscx.accel.subset_var(adata, COL_KEEP)
    _assert_de_equal(
        _rgg(adata, prefer_format="csc"),
        _rgg(_reference(counts, obs_cols, var_names, cols=COL_KEEP)),
    )
    assert _de_route(adata) == "cpu_csc_nnz"


def test_csc_on_both_axes_matches_the_reference(csc_path, counts, obs_cols, var_names):
    """Both windows at once, which is the shape a QC-then-filter workflow
    leaves behind and the one neither refusal above could reach."""
    adata = _backed(csc_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)
    _assert_de_equal(
        _rgg(adata, prefer_format="csc"),
        _rgg(_reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP)),
    )
    assert _de_route(adata) == "cpu_csc_nnz"


def _project_genes(adata, how, counts):
    """Project the gene axis of a backed handle three ways, with no row filter
    and no transform chain."""
    if how == "subset_var":
        pyscx.accel.subset_var(adata, COL_KEEP)
    elif how == "filter_genes":
        detected = np.asarray((counts > 0).sum(axis=0)).ravel()
        pyscx.accel.filter_genes(adata, min_cells=int(np.median(detected)) + 1)
    else:
        adata = adata[:, COL_KEEP]
    return adata


@pytest.mark.parametrize("how", ["subset_var", "filter_genes", "slice"])
def test_auto_takes_csc_on_a_gene_only_projection(csc_path, counts, obs_cols, var_names, how):
    """At the default `prefer_format="auto"`, a backed handle whose only view is
    a gene projection takes `cpu_csc`.

    A regression pin, not the test for a fix. The behaviour arrived when the
    backed handle started serving its CSC reads through the same view a lazy
    one does, which remaps columns into the projected axis. Before that its
    column source was the full-axis sidecar reader, so the probe excluded a
    column projection by hand and `filter_genes` alone routed `cpu_csr` while
    `filter_genes + normalize_total + log1p` routed `cpu_csc`. Nothing asserted
    the gene-only case since, so this pins it for all three ways of making one.
    """
    adata = _project_genes(_backed(csc_path), how, counts)
    kept = [var_names.index(n) for n in adata.var_names]
    assert 0 < len(kept) < N_VARS, f"premise: {how} must drop some genes, kept {len(kept)}"
    assert adata.n_obs == N_OBS, "premise: no row filter"
    cols = np.zeros(N_VARS, dtype=bool)
    cols[kept] = True
    import warnings

    with warnings.catch_warnings():
        # `slice` hands the op an AnnData view, which it rebuilds in place.
        warnings.simplefilter("ignore")
        got = _rgg(adata)
    _assert_de_equal(got, _rgg(_reference(counts, obs_cols, var_names, cols=cols)))
    assert _de_route(adata) == "cpu_csc_nnz"


# ---------------------------------------------------------------------------
# GPU DE route: the `has_axis_view()` switch
# ---------------------------------------------------------------------------
#
# `has_axis_view()` decides **which** CSC source to hand over, and used to
# decide whether to hand one over at all. `GpuDeShardInput::Lazy` carries no
# CSC sidecar, so an unsubset handle goes through `Backed { csr, csc }` with
# the concrete readers to keep the CSC-direct `gpu_csc_v3` route —
# `CLAUDE.md` treats a silent CSC→CSR downgrade as a hard gate failure.
#
# A subset handle used to be forced onto `Lazy`, giving the route up, because
# `Backed`'s `csc` was read straight off `backed_csc` — the *file*, full row
# axis and full column axis — and neither of the other CSC gates protects this
# path (`as_column_source()` is bypassed, and `resolve_de_format`
# short-circuits to "csr" on GPU without consulting `csc_route_available`), so
# it would have run the CSC-direct kernel against on-disk columns under
# visible-width `gene_names`. `Backed` now takes trait objects, so a subset
# handle passes its own view on both sides — rows renumbered onto the live
# space, columns remapped into the projected one — and keeps the route.


def _gpu_available():
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


gpu_only = pytest.mark.skipif(not _gpu_available(), reason="CUDA GPU not available")


def _de_route(adata):
    return adata.uns["scx_accel"]["rank_genes_groups"]["route"]


@gpu_only
def test_gpu_de_keeps_csc_direct_when_unsubset(csc_path):
    adata = _backed(csc_path)
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", device="gpu"
    )
    assert _de_route(adata) == "gpu_csc_v3", (
        f"unsubset handle with a CSC sidecar must keep the CSC-direct route, "
        f"got {_de_route(adata)!r}"
    )


@gpu_only
@pytest.mark.parametrize(
    "subset",
    [
        pytest.param(lambda a: pyscx.accel.subset_obs(a, ROW_KEEP), id="rows"),
        pytest.param(lambda a: pyscx.accel.subset_var(a, COL_KEEP), id="cols"),
    ],
)
def test_gpu_de_keeps_csc_direct_when_subset(csc_path, subset):
    """This asserted `gpu_csr_v3` until `GpuDeShardInput::Backed` took trait
    objects. It was right then: the only CSC source a subset handle could
    offer was the full-axis reader. It now offers its own view, so the route
    survives the subset — and `test_gpu_de_subset_matches_cpu` below is what
    says the answer survives with it."""
    adata = _backed(csc_path)
    subset(adata)
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", device="gpu"
    )
    assert _de_route(adata) == "gpu_csc_v3", (
        f"a subset handle presents its own CSC view, so the CSC-direct route "
        f"must survive, got {_de_route(adata)!r}"
    )


@gpu_only
def test_gpu_de_subset_matches_cpu(csc_path, counts, obs_cols, var_names):
    """The route must also produce the right answer, not just be recorded.

    Load-bearing in a way it was not before: the CSC-direct GPU kernels index
    `cell_to_group[row_indices[e]]`, a per-row table the driver sizes from the
    caller's `groups`. On a subset handle that table is visible-length while
    the sidecar's rows are global, so this is the test that says the
    renumbering reached the device."""
    adata = _backed(csc_path)
    pyscx.accel.subset_obs(adata, ROW_KEEP)
    pyscx.accel.subset_var(adata, COL_KEEP)
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", device="gpu"
    )
    got = pyscx.accel.rank_genes_groups_df(adata, group="drug")
    ref = _rgg(
        _reference(counts, obs_cols, var_names, rows=ROW_KEEP, cols=COL_KEEP)
    )
    assert list(got["names"]) == list(ref["names"])
    np.testing.assert_allclose(
        got["scores"].to_numpy().astype(np.float64),
        ref["scores"].to_numpy().astype(np.float64),
        atol=1e-4,
    )


@gpu_only
def test_gpu_de_keeps_csc_direct_on_a_transformed_handle(csc_path):
    """The shape an actual pipeline presents on GPU.

    `normalize_total` turns `X` into an `ScxLazyTransformedDataset`, a
    different pyclass that never reaches `has_axis_view()` — it took the
    CSR-shaped `Lazy` input unconditionally, so `gpu_csc_v3` was unreachable
    for any transformed handle however many sidecars the file had. Compared
    against the CPU CSC route on the same window and chain, at the tolerance
    the other GPU DE tests in this file use.
    """
    def run(device):
        adata = _backed(csc_path)
        pyscx.accel.subset_obs(adata, ROW_KEEP)
        # `device="cpu"` on the transforms, deliberately. Left at the default
        # `"auto"` they route through rapids-singlecell on a GPU host, which
        # **materialises** `X` — so the handle reaching DE is an in-memory
        # scipy matrix, not a lazy SCX one, `GpuDeShardInput::Csr` is what the
        # dispatch picks, and the route comes back `gpu_csr_v3` without the
        # lazy arm this test exists for ever being reached. That is how the
        # first version of this test failed: it asserted the right thing about
        # a shape it never built.
        pyscx.accel.normalize_total(adata, target_sum=1e4, device="cpu")
        pyscx.accel.log1p(adata, device="cpu")
        assert type(adata.X).__name__ == "ScxLazyTransformedDataset", (
            f"premise: the chain must leave X lazy, got {type(adata.X).__name__}"
        )
        pyscx.accel.rank_genes_groups(
            adata, groupby="pert", method="wilcoxon", device=device
        )
        return _de_route(adata), pyscx.accel.rank_genes_groups_df(adata, group="drug")

    gpu_route, gpu_df = run("gpu")
    cpu_route, cpu_df = run("cpu")
    assert gpu_route == "gpu_csc_v3", (
        f"a transformed handle with a sidecar must reach the CSC-direct GPU "
        f"route, got {gpu_route!r}"
    )
    assert cpu_route == "cpu_csc_nnz", f"premise: the CPU side takes CSC too, got {cpu_route!r}"
    assert list(gpu_df["names"]) == list(cpu_df["names"])
    np.testing.assert_allclose(
        gpu_df["scores"].to_numpy().astype(np.float64),
        cpu_df["scores"].to_numpy().astype(np.float64),
        atol=1e-4,
    )


def test_csc_unsubset_still_works(csc_path, counts, obs_cols, var_names):
    """The new guard must not catch an ordinary CSC-direct run."""
    adata = _backed(csc_path)
    pyscx.accel.rank_genes_groups(
        adata, groupby="pert", method="wilcoxon", prefer_format="csc", device="cpu"
    )
    _assert_de_equal(
        pyscx.accel.rank_genes_groups_df(adata, group="drug"),
        _rgg(_reference(counts, obs_cols, var_names)),
    )
