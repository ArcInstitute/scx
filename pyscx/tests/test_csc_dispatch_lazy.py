"""CSC dispatch on `ScxLazyTransformedDataset`.

Verifies the capability gate on lazy datasets:

  - `Log1p` → `prefer_format="csc"` succeeds and matches the CSR
    equivalent.
  - `NormalizeTotal` is row-*indexed*, not column-local, and `csc`
    serves it anyway by looking up `row_sums[indices[k]]` — `indices`
    *is* the global row. Bit-identical to CSR.
  - A row deletion vector still disqualifies, and that one is a
    correctness barrier rather than conservatism: it renumbers the live
    rows while CSC `indices` stay global.
  - `prefer_format="csr"` works on every transform chain.

Sibling to `test_csc_dispatch.py`. The point of this file is the
*lazy* path specifically — `test_csc_dispatch.py` exercises the
backed path with the same assertions.
"""

from __future__ import annotations

import numpy as np
import pytest
import pandas as pd
import scipy.sparse as sp


@pytest.fixture
def small_adata():
    pytest.importorskip("anndata")
    import anndata as ad

    rng = np.random.default_rng(11)
    mat = sp.random(40, 16, density=0.4, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 50).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(40)]
    adata.obs["group"] = ["A"] * 20 + ["B"] * 20
    adata.var["gene_id"] = [f"g{i}" for i in range(16)]
    return adata


def _wide_adata():
    """60x40 counts — wide enough that a real gene cut still fits loess.

    `small_adata` is 40x16, which cannot be both narrowed (so a column
    projection exists at all) and left loess-viable for `seurat_v3`.
    """
    import anndata as ad

    rng = np.random.default_rng(11)
    mat = sp.random(60, 40, density=0.4, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 50).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(60)]
    adata.obs["group"] = ["A"] * 30 + ["B"] * 30
    adata.var["gene_id"] = [f"g{i}" for i in range(40)]
    return adata


def _open_with_csc(path, adata):
    import pyscx

    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=4)
    return pyscx.open(str(path)).to_anndata(backed=True)


# ---------------------------------------------------------------------------
# Log1p path: CSC dispatch must succeed and match CSR.
# ---------------------------------------------------------------------------


def test_lazy_log1p_csc_col_sums_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)

    csc_sums = pyscx.accel.col_sums(a_csc.X, prefer_format="csc")
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.sum(axis=0).astype(np.float64)
    np.testing.assert_allclose(csc_sums, expected, atol=1e-5)


def test_lazy_log1p_csc_col_var_matches_materialized(small_adata, tmp_path):
    """`col_var` on a lazy log1p chain via CSC matches the
    materialized log1p variance."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)

    csc_var = pyscx.accel.col_var(a_csc.X, prefer_format="csc")
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.var(axis=0).astype(np.float64)
    np.testing.assert_allclose(csc_var, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# NormalizeTotal path: row-*indexed*, not column-local — CSC dispatch
# serves it by looking the row up, and must agree with CSR exactly.
#
# These two tests previously asserted that `prefer_format="csc"` **raises**
# here. That was a faithful spec of the old gate, which tested
# `is_column_local()`: does an element's output depend only on its own
# column? For `NormalizeTotal` it does not. But that is the wrong question
# for a CSC reader, which knows each nonzero's global row because
# `ScxCsc::indices` *is* that row, and `NormalizeTotal` carries its
# `row_sums` vector with it. So the gate stopped asking about the chain at
# all: it now checks only for a sidecar and the absence of a row filter.
# ---------------------------------------------------------------------------


def test_lazy_normalize_total_csc_matches_csr(small_adata, tmp_path):
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.normalize_total(a_csc, target_sum=1e4)

    # `col_sums(prefer_format="csr")` has no lazy path (documented limitation
    # in col_aggs.rs), so the CSR reference is the dunder sum — the same one
    # `test_lazy_normalize_total_csr_works` below uses. Tolerance, not
    # equality: the two accumulate in different orders. Exactness of the
    # *transform* is pinned bit-for-bit in the Rust unit tests
    # (`normalize_total_is_bit_identical_across_layouts`).
    csc_sums = np.asarray(pyscx.accel.col_sums(a_csc.X, prefer_format="csc"))
    materialised = a_csc.X[:]
    dense = materialised.toarray() if sp.issparse(materialised) else np.asarray(materialised)
    np.testing.assert_allclose(csc_sums, dense.sum(axis=0).astype(np.float64), rtol=1e-6)


def test_lazy_normalize_total_csr_works(small_adata, tmp_path):
    """`prefer_format="csr"` is unaffected by NormalizeTotal."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    # Pin target_sum explicitly: the streaming-vs-materialized equality below is
    # magnitude-sensitive at atol=1e-5, and the default is now None→median (a
    # small target on integer counts), which tips two genes over the tolerance.
    pyscx.accel.normalize_total(a_csc, target_sum=1e4)

    # CSR path on a lazy NormalizeTotal chain currently goes through
    # the dunder methods on the lazy wrapper (the col_sums pyfunction
    # only supports backed datasets on its CSR path — documented
    # limitation in pyscx/src/accel/col_aggs.rs). Use the dunder
    # interface for the CSR check.
    csr_sums = np.asarray(a_csc.X.sum(axis=0)).ravel()
    materialised = a_csc.X[:].toarray() if sp.issparse(a_csc.X[:]) else np.asarray(a_csc.X[:])
    expected = materialised.sum(axis=0).astype(np.float64)
    np.testing.assert_allclose(csr_sums, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# Compound chains: log1p preserves CSC capability; appending
# normalize_total breaks it.
# ---------------------------------------------------------------------------


def test_lazy_log1p_then_normalize_total_stays_csc_capable(small_adata, tmp_path):
    """A mixed chain keeps CSC dispatch, and still agrees with CSR."""
    import pyscx

    a_csc = _open_with_csc(tmp_path / "with_csc.scx", small_adata)
    pyscx.accel.log1p(a_csc)
    after_log1p = np.asarray(pyscx.accel.col_sums(a_csc.X, prefer_format="csc"))

    pyscx.accel.normalize_total(a_csc, target_sum=1e4)
    csc_sums = np.asarray(pyscx.accel.col_sums(a_csc.X, prefer_format="csc"))
    materialised = a_csc.X[:]
    dense = materialised.toarray() if sp.issparse(materialised) else np.asarray(materialised)
    np.testing.assert_allclose(csc_sums, dense.sum(axis=0).astype(np.float64), rtol=1e-6)
    # Premise: appending normalize_total actually changed the values, so the
    # comparison above is not checking the chain against itself.
    assert not np.allclose(after_log1p, csc_sums)


def test_a_row_deletion_vector_still_disqualifies_csc(small_adata, tmp_path):
    """The one gate condition this change does *not* relax.

    CSC `indices` encode **global** row ids. A deletion vector renumbers the
    live rows, so a row-indexed transform would read the wrong entry of its
    per-row vector — unlike `normalize_total`, this is not a lookup away.
    """
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)
    exp = pyscx.open(str(path))
    mask = np.zeros(small_adata.n_obs, dtype=bool)
    mask[:2] = True
    exp.mark_deleted(mask)
    a = pyscx.open(str(path)).to_anndata(backed=True)
    # Premise: the deletion vector really is active on this handle, so the
    # refusal below is the row filter's doing and not a missing sidecar.
    assert a.n_obs == small_adata.n_obs - 2
    pyscx.accel.normalize_total(a, target_sum=1e4)
    # The message must name *this* cause. `build_csc` is the wrong repair
    # here — the file already has a sidecar — so the error must not offer it.
    with pytest.raises(RuntimeError, match="row deletion vector is active") as e:
        pyscx.accel.col_sums(a.X, prefer_format="csc")
    assert "build_csc" not in str(e.value), (
        "a sidecar-carrying file filtered by rows must not be told to build one"
    )


def test_a_missing_sidecar_names_the_other_cause(small_adata, tmp_path):
    """The complement: the same helper, the other branch.

    Without this, `csc_unavailable` could return the deletion-vector string
    unconditionally and the test above would still pass.
    """
    import pyscx

    path = tmp_path / "no_csc.scx"
    pyscx.from_anndata(small_adata, str(path))
    a = pyscx.open(str(path)).to_anndata(backed=True)
    pyscx.accel.normalize_total(a, target_sum=1e4)
    with pytest.raises(RuntimeError, match="no CSC sidecar") as e:
        pyscx.accel.col_sums(a.X, prefer_format="csc")
    assert "deletion vector" not in str(e.value)


# ---------------------------------------------------------------------------
# Column projection × CSC: HVG must accumulate against the projected
# axis. Regression for the bug where LazyShardSource::read_csc_shard
# applied global col_projection IDs against shard-local column space.
# ---------------------------------------------------------------------------


def test_lazy_csc_with_col_projection_hvg_matches_csr(small_adata, tmp_path):
    """HVG via prefer_format='csc' on a column-projected lazy dataset
    must match the CSR equivalent.

    The CSC HVG kernels (`streaming_mean_var_csc`,
    `streaming_clip_square_sum_csc`) iterate per CSC shard via
    `read_csc_shard`. Before the fix, `LazyShardSource::read_csc_shard`
    fed `project_csc` global file column IDs against a shard-local
    column space — out-of-shard projection entries were silently dropped
    and survivors mapped to wrong shard-local positions, corrupting the
    means/variances written to `adata.var`.

    The fixture has 16 genes with `csc_cols_per_shard=4` → 4 CSC shards.
    `filter_genes(min_cells=3)` yields a projection spanning multiple
    shards. We compare HVG outputs against the CSR path on the same
    chain.
    """
    import pyscx

    # Deliberately not `small_adata` (40x16): its least-detected gene appears in
    # 10 of 40 cells, so the original `min_cells=3` kept all 16 — and once
    # `filter_genes` learned to skip an all-kept mask outright, this test lost
    # the projection that is its entire subject. Narrowing 16 genes far enough
    # to cut leaves too few for the seurat_v3 loess fit ("Chernobyl! trL>n"), so
    # the fixture is widened here instead: 60x40 with `min_cells=22` keeps 34,
    # a strict subset spanning several `csc_cols_per_shard=4` shards and still
    # comfortably loess-viable.
    wide = _wide_adata()
    a_csr = _open_with_csc(tmp_path / "with_csc_csr.scx", wide)
    a_csc = _open_with_csc(tmp_path / "with_csc_csc.scx", wide)

    pyscx.accel.log1p(a_csr)
    pyscx.accel.log1p(a_csc)

    n_before = a_csc.n_vars
    pyscx.accel.filter_genes(a_csr, min_cells=22)
    pyscx.accel.filter_genes(a_csc, min_cells=22)

    n_proj = a_csc.X.shape[1]
    # The bug only fires when the projection spans at least two CSC shards.
    # `csc_cols_per_shard=4`, so the projection must be both a strict subset
    # (otherwise there is no projection at all) and wide enough to cross a
    # shard boundary.
    assert 4 <= n_proj < n_before, (
        f"projection is {n_proj}/{n_before}; the test needs a strict subset "
        "spanning multiple CSC shards. Adjust min_cells or the fixture seed."
    )

    n_top = max(2, min(5, n_proj - 1))
    pyscx.accel.highly_variable_genes(
        a_csr,
        n_top_genes=n_top,
        flavor="seurat_v3",
        prefer_format="csr",
    )
    pyscx.accel.highly_variable_genes(
        a_csc,
        n_top_genes=n_top,
        flavor="seurat_v3",
        prefer_format="csc",
    )

    csr_means = np.asarray(a_csr.var["means"], dtype=np.float64)
    csc_means = np.asarray(a_csc.var["means"], dtype=np.float64)
    csr_var = np.asarray(a_csr.var["variances"], dtype=np.float64)
    csc_var = np.asarray(a_csc.var["variances"], dtype=np.float64)

    np.testing.assert_allclose(
        csc_means, csr_means, atol=1e-5,
        err_msg="CSC means diverge from CSR after col_projection — "
                "LazyShardSource::read_csc_shard projection remap regression",
    )
    np.testing.assert_allclose(
        csc_var, csr_var, atol=1e-5,
        err_msg="CSC variances diverge from CSR after col_projection — "
                "LazyShardSource::read_csc_shard projection remap regression",
    )


# ---------------------------------------------------------------------------
# The payoff path: Wilcoxon DE on `normalize_total -> log1p`.
#
# This is the chain essentially every analyst runs before differential
# expression, and it is the one the old gate refused. Measured on census_1m
# (1M x 61,497, 50 groups): 1,270 s / 7,319 MB on cpu_csr against 226 s /
# 5,652 MB on cpu_csc. The speed is the reason to care; these tests are about
# the part that has to be true for the speed to be worth anything.
# ---------------------------------------------------------------------------


def _de_fixture():
    import anndata as ad

    rng = np.random.default_rng(7)
    mat = sp.random(120, 24, density=0.5, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 100).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(120)]
    adata.obs["grp"] = pd.Categorical(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    adata.var["gene_id"] = [f"g{i}" for i in range(24)]
    return adata


def _de_on(path, adata, prefer):
    import pyscx

    a = pyscx.open(str(path)).to_anndata(backed=True)
    a.obs["grp"] = adata.obs["grp"].to_numpy()
    a.obs["grp"] = pd.Categorical(a.obs["grp"])
    pyscx.accel.normalize_total(a, target_sum=1e4)
    pyscx.accel.log1p(a)
    pyscx.accel.rank_genes_groups(
        a, groupby="grp", method="wilcoxon", device="cpu", prefer_format=prefer,
    )
    route = (a.uns.get("scx_accel") or {}).get("rank_genes_groups", {}).get("route")
    return route, a.uns["rank_genes_groups"]


def test_de_on_the_normalize_log1p_chain_routes_csc_and_is_bit_identical(tmp_path):
    import pyscx

    adata = _de_fixture()
    path = tmp_path / "de_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=6)

    route_csr, res_csr = _de_on(path, adata, "csr")
    route_csc, res_csc = _de_on(path, adata, "csc")
    route_auto, _ = _de_on(path, adata, "auto")

    assert route_csr == "cpu_csr"
    assert route_csc == "cpu_csc"
    # `auto` is DE's default, so this is what an ordinary caller now gets.
    assert route_auto == "cpu_csc"

    for field in ("scores", "pvals", "logfoldchanges"):
        a = np.array([list(r) for r in res_csr[field]])
        b = np.array([list(r) for r in res_csc[field]])
        assert a.shape == b.shape
        # Bit-level, not `allclose`: these are element-wise maps feeding the
        # same kernel, so exact agreement is achievable and anything less
        # would mean one route is computing something different.
        assert np.array_equal(a.view(np.uint32 if a.dtype == np.float32 else np.uint64),
                              b.view(np.uint32 if b.dtype == np.float32 else np.uint64)), (
            f"{field} differs between routes; max|diff| = {np.nanmax(np.abs(a - b))}"
        )
    names_csr = np.array([list(r) for r in res_csr["names"]])
    names_csc = np.array([list(r) for r in res_csc["names"]])
    assert np.array_equal(names_csr, names_csc)


def test_de_without_a_sidecar_still_routes_csr(tmp_path):
    """Premise: the route above is the sidecar's doing, not the default."""
    import pyscx

    adata = _de_fixture()
    path = tmp_path / "de_nocsc.scx"
    pyscx.from_anndata(adata, str(path), csc="off")
    route, _ = _de_on(path, adata, "auto")
    assert route == "cpu_csr"


# ---------------------------------------------------------------------------
# Round-2 regressions.
#
# Both were found by review of the round-1 fix commit, and both are cases the
# non-negative fixtures above structurally cannot reach.
# ---------------------------------------------------------------------------


def _signed_adata():
    import anndata as ad

    # Row 0 cancels to a total of exactly 0, row 1 to -1. Every other row is
    # ordinary positive data, so only these two take the `sum > 0.0` false
    # branch — which on non-negative input is reachable only with all-zero
    # rows, where every candidate behaviour agrees.
    dense = np.array(
        [
            [0.5, -0.5, 0.0, 0.0],
            [-2.0, 1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    a = ad.AnnData(X=sp.csr_matrix(dense))
    a.obs["cell_id"] = [f"c{i}" for i in range(dense.shape[0])]
    a.var["gene_id"] = [f"g{i}" for i in range(dense.shape[1])]
    return a


def test_a_signed_row_reads_the_same_through_a_slice_and_a_fancy_index(tmp_path):
    """`X[:]` and `X[[...]]` are different code paths, and both fuse.

    `apply_transforms_to_csr` and `apply_transforms_per_row` each special-case
    a leading `normalize_total → log1p`. The round-1 fix repaired the first and
    left the second, so a row whose total is not positive came back with raw
    values through a fancy index and `ln_1p`-ed values through a slice — the
    same matrix answering differently depending on how it was addressed.
    """
    import pyscx

    adata = _signed_adata()
    path = tmp_path / "signed.scx"
    pyscx.from_anndata(adata, str(path))

    h = pyscx.open(str(path)).to_anndata(backed=True)
    pyscx.accel.normalize_total(h, target_sum=1e4)
    pyscx.accel.log1p(h)

    rows = [0, 1, 2, 3]
    sliced = h.X[:]
    fancy = h.X[rows]
    sliced = sliced.toarray() if sp.issparse(sliced) else np.asarray(sliced)
    fancy = fancy.toarray() if sp.issparse(fancy) else np.asarray(fancy)

    # Premise: the signed rows really did reach the non-positive branch. A
    # value of -2.0 has no real ln_1p, so a NaN here is the branch's signature.
    assert np.isnan(sliced[1]).any(), (
        "row 1 sums to -1 and holds -2.0; the slice path should have applied "
        f"ln_1p and produced a NaN. Got {sliced[1]}"
    )
    np.testing.assert_array_equal(
        np.isnan(sliced), np.isnan(fancy),
        err_msg="slice and fancy-index disagree on which cells are NaN",
    )
    np.testing.assert_array_equal(
        np.nan_to_num(sliced, nan=0.0), np.nan_to_num(fancy, nan=0.0),
        err_msg="slice and fancy-index disagree on a signed row's values",
    )


def test_a_projected_lazy_handle_addresses_the_projected_axis(tmp_path):
    """A lazy CSC source is already on the projected axis.

    The projected-axis consumers used to hand it back the *global* column ids,
    applying the projection twice: wrong genes when the ids stayed in range,
    and a panic in `walk_csc_runs` when they did not. Reachable before this PR
    with a `log1p`-only chain, so both chains are checked.
    """
    import pyscx

    rng = np.random.default_rng(5)
    mat = sp.random(60, 20, density=0.5, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 50).astype(np.float32).round() + 1.0
    import anndata as ad

    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(60)]
    adata.var["gene_id"] = [f"g{i}" for i in range(20)]
    path = tmp_path / "proj.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=5)

    # Non-identity: keep the second half, so a global id is never a valid
    # position in the projected axis and the two addressings cannot coincide.
    keep = np.arange(20) >= 10

    for chain in ("log1p", "normalize_then_log1p"):
        h = pyscx.open(str(path)).to_anndata(backed=True)[:, keep]
        if chain == "normalize_then_log1p":
            pyscx.accel.normalize_total(h, target_sum=1e4)
        pyscx.accel.log1p(h)

        csc_sums = np.asarray(pyscx.accel.col_sums(h.X, prefer_format="csc"))
        materialised = h.X[:]
        dense = (
            materialised.toarray() if sp.issparse(materialised) else np.asarray(materialised)
        )
        assert csc_sums.shape == (int(keep.sum()),), f"{chain}: wrong width"
        np.testing.assert_allclose(
            csc_sums, dense.sum(axis=0).astype(np.float64), rtol=1e-6,
            err_msg=f"{chain}: projected lazy CSC col_sums disagree with the matrix",
        )


def test_projected_lazy_qc_gene_axis_matches_the_csr_route(tmp_path):
    """The same double-projection, through `calculate_qc_metrics`."""
    import pyscx
    import anndata as ad

    rng = np.random.default_rng(6)
    mat = sp.random(50, 16, density=0.6, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 30).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(50)]
    adata.var["gene_id"] = [f"g{i}" for i in range(16)]
    path = tmp_path / "qc_proj.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=4)
    keep = np.arange(16) >= 8

    for chain in ("log1p", "normalize_then_log1p"):
        out = {}
        for prefer in ("csr", "csc"):
            h = pyscx.open(str(path)).to_anndata(backed=True)[:, keep]
            if chain == "normalize_then_log1p":
                pyscx.accel.normalize_total(h, target_sum=1e4)
            pyscx.accel.log1p(h)
            pyscx.accel.calculate_qc_metrics(h, prefer_format=prefer, inplace=True)
            out[prefer] = np.asarray(h.var["total_counts"], dtype=np.float64)
        np.testing.assert_allclose(
            out["csc"], out["csr"], rtol=1e-6,
            err_msg=f"{chain}: projected lazy QC gene axis disagrees with CSR",
        )


# ---------------------------------------------------------------------------
# Round-3 regression: pseudobulk's two coordinate spaces.
#
# `resolved_indices` addressed the CSC source and indexed `adata.var` at once.
# Those are different spaces on a projected *backed* handle — the reader is
# full-axis, `adata.var` is already subset — so a non-prefix mask raised, and a
# mask that kept every id below the visible width silently labelled file column
# `c` with visible gene `c`. Pre-existing; reachable through the very path the
# projection table in docs/scanpy.md recommends.
# ---------------------------------------------------------------------------


def _pseudobulk_adata():
    import anndata as ad
    import pandas as pd

    rng = np.random.default_rng(9)
    n_obs, n_vars = 120, 20
    mat = sp.random(n_obs, n_vars, density=0.6, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 40).astype(np.float32).round() + 1.0
    a = ad.AnnData(X=mat)
    a.obs["cell_id"] = [f"c{i}" for i in range(n_obs)]
    a.obs["cond"] = pd.Categorical(["ctrl"] * 60 + ["trt"] * 60)
    # Three donors inside each condition, so every condition has the >= 2
    # pseudobulk replicates the NB-GLM needs.
    a.obs["donor"] = pd.Categorical([f"d{i % 3}" for i in range(60)] * 2)
    a.var["gene_id"] = [f"g{i}" for i in range(n_vars)]
    return a


def _pb(path, keep, prefer, gene_indices=None):
    import pyscx

    h = pyscx.open(str(path)).to_anndata(backed=True)[:, keep]
    return pyscx.accel.pseudobulk_dex(
        h, groupby=["cond", "donor"], test_col="cond", reference="ctrl",
        prefer_format=prefer, gene_indices=gene_indices,
    ).set_index("gene")


def test_projected_backed_pseudobulk_reads_and_labels_the_visible_genes(tmp_path):
    """A non-prefix projection: raised `gene index 10 out of range` before.

    The mask keeps the second half, so every on-disk id is >= the visible
    width — which is exactly what made the conflated index vector fail loudly
    rather than quietly.
    """
    import pyscx

    adata = _pseudobulk_adata()
    path = tmp_path / "pb.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=5)
    keep = np.arange(20) >= 10

    csc = _pb(path, keep, "csc")
    csr = _pb(path, keep, "csr")

    assert list(csc.index) == list(csr.index), "CSC labelled different genes than CSR"
    # Premise: the projection really is non-prefix, so a conflated index
    # vector could not have survived by coincidence.
    assert list(csc.index) == [str(i) for i in range(10, 20)]
    np.testing.assert_allclose(
        csc["baseMean"].to_numpy(), csr["baseMean"].to_numpy(), rtol=1e-4,
        err_msg="CSC read different columns than CSR for the same visible genes",
    )


def test_gene_indices_on_a_projected_backed_handle_compose_through_the_projection(tmp_path):
    """The silent half: positions are *visible*, the reader is full-axis.

    `gene_indices=[0, 5]` after `[:, 10:]` means visible genes 10 and 15. The
    conflated vector sent 0 and 5 straight to the full-axis reader, returning
    file columns 0 and 5 under the names of genes 10 and 15 — no error, just
    the wrong numbers.
    """
    import pyscx

    adata = _pseudobulk_adata()
    path = tmp_path / "pb2.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=5)
    keep = np.arange(20) >= 10

    sub = _pb(path, keep, "csc", gene_indices=[0, 5])
    assert list(sub.index) == ["10", "15"]

    # The values must come from genes 10 and 15, not file columns 0 and 5.
    # Compared against a projection restricted to those same two genes, so both
    # runs aggregate the identical gene set and share their size factors.
    pair = np.zeros(20, dtype=bool)
    pair[[10, 15]] = True
    ref = _pb(path, pair, "csr")
    assert list(ref.index) == ["10", "15"]
    np.testing.assert_allclose(
        sub["baseMean"].to_numpy(), ref["baseMean"].to_numpy(), rtol=1e-4,
        err_msg="gene_indices were not composed through the column projection",
    )
