"""CSC dispatch on `ScxLazyTransformedDataset`.

Verifies the capability gate on lazy datasets:

  - `Log1p` → `prefer_format="csc"` succeeds and matches the CSR
    equivalent.
  - `NormalizeTotal` is row-*indexed*, not column-local, and `csc`
    serves it anyway by looking up `row_sums[indices[k]]` — `indices`
    *is* the global row. Bit-identical to CSR.
  - A row filter no longer disqualifies either. CSC `indices` stay
    global, but the read path renumbers them onto the live row space
    before the slab leaves the reader, so what a kernel receives is
    addressed the way it already assumed.
  - The only remaining disqualifier is a file with no CSC sidecar.
  - `prefer_format="csr"` works on every transform chain.

The row-filter tests compare **values**, not just the recorded route,
and that is deliberate: every CSC kernel skips a row it cannot place
(`row >= n_obs`), so a wrong renumbering drops nonzeros and mislabels
the rest without raising. A route assertion cannot see it.

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


def test_a_deletion_vector_no_longer_disqualifies_csc(small_adata, tmp_path):
    """The gate condition this change relaxes, and the one that used to be
    called a correctness barrier rather than conservatism.

    It *was* a barrier as written: CSC `indices` encode **global** row ids and
    a deletion vector renumbers the live rows, so a slab handed over unchanged
    would have scattered into the wrong output rows. The read path renumbers
    it instead. Compared against the CSR route, because the failure mode is a
    silently dropped row, not an error.
    """
    import pyscx

    path = tmp_path / "with_csc.scx"
    pyscx.from_anndata(small_adata, str(path), csc="always", csc_cols_per_shard=4)
    mask = np.zeros(small_adata.n_obs, dtype=bool)
    mask[:2] = True
    pyscx.open(str(path)).mark_deleted(mask)

    a = pyscx.open(str(path)).to_anndata(backed=True)
    # Premise: the deletion vector really is active on this handle, so what
    # follows is the row filter's doing.
    assert a.n_obs == small_adata.n_obs - 2
    pyscx.accel.normalize_total(a, target_sum=1e4)
    csc_sums = np.asarray(pyscx.accel.col_sums(a.X, prefer_format="csc"))

    # Against the live matrix computed in numpy, not against the CSR route:
    # `col_sums(prefer_format="csr")` does not accept a lazy handle at all
    # (it refuses and tells you to materialise), so the dense window is the
    # available oracle — and the stronger one, since it cannot be wrong the
    # same way both routes might be.
    dense = small_adata.X.toarray()[2:]
    scale = 1e4 / dense.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        csc_sums, (dense * scale).sum(axis=0).astype(np.float64), rtol=1e-5
    )


def test_a_missing_sidecar_is_now_the_only_cause(small_adata, tmp_path):
    """The complement, and after this change the whole of it.

    `csc_unavailable` used to take two booleans and pick between two causes.
    A row filter is no longer one of them, so the helper names the one that is
    left — and must still name it, rather than having been collapsed into
    something vaguer.
    """
    import pyscx

    path = tmp_path / "no_csc.scx"
    pyscx.from_anndata(small_adata, str(path))
    a = pyscx.open(str(path)).to_anndata(backed=True)
    pyscx.accel.normalize_total(a, target_sum=1e4)
    with pytest.raises(RuntimeError, match="no CSC sidecar") as e:
        pyscx.accel.col_sums(a.X, prefer_format="csc")
    assert "build_csc" in str(e.value), "the message must name the repair"
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


# ---------------------------------------------------------------------------
# Row filters: the shape every real pipeline has.
#
# `filter_cells` drops rows, which renumbers the live row axis while the CSC
# sidecar keeps addressing the file's. Serving that needs the slab's rows
# translated, and the translation is invisible if it is wrong: every CSC
# kernel skips a row it cannot place (`row >= n_obs`) and reads
# `groups[row]` / `cell_to_group[row]` for the ones it can, so a bad map
# silently drops nonzeros and mislabels the rest. Hence every test here
# compares values against the CSR route rather than asserting a route.
# ---------------------------------------------------------------------------


def _counts_adata(n_obs=120, n_vars=24, seed=5):
    import anndata as ad

    rng = np.random.default_rng(seed)
    mat = sp.random(n_obs, n_vars, density=0.5, format="csr", dtype=np.float32, random_state=rng)
    mat.data = (mat.data * 100).astype(np.float32).round() + 1.0
    adata = ad.AnnData(X=mat)
    adata.obs["cell_id"] = [f"c{i}" for i in range(n_obs)]
    adata.obs["grp"] = pd.Categorical(
        ["A"] * (n_obs // 3) + ["B"] * (n_obs // 3) + ["C"] * (n_obs - 2 * (n_obs // 3))
    )
    adata.var["gene_id"] = [f"g{i}" for i in range(n_vars)]
    return adata


def _min_genes_that_drops(adata):
    """A `filter_cells` threshold that provably drops at least one cell.

    Hard-coding one is the trap this avoids: at `min_genes=200` pbmc3k drops
    *nothing*, so a probe written there reports `cpu_csc` while never having
    reached the gate at all. The median genes-per-cell always drops some.
    """
    per_cell = np.diff(adata.X.indptr)
    thresh = int(np.median(per_cell))
    assert (per_cell < thresh).sum() > 0, "fixture cannot exercise a row filter"
    return thresh


def _dense_view(a):
    x = a.X[:]
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def _filtered_handle(path, adata, *, rows=True, cols=False, transforms=True, obs_grp=True):
    """Open `path` and apply the requested window and chain."""
    import pyscx

    a = pyscx.open(str(path)).to_anndata(backed=True)
    if obs_grp:
        a.obs["grp"] = pd.Categorical(adata.obs["grp"].to_numpy())
    if rows:
        pyscx.accel.filter_cells(a, min_genes=_min_genes_that_drops(adata))
        assert a.n_obs < adata.n_obs, "premise: the row filter dropped cells"
    if cols:
        # Sized from the *live* window's own per-gene counts. A threshold
        # taken from the unfiltered matrix is too high for a row-filtered
        # handle and drops every gene; a fixed fraction of `n_obs` drops
        # none. Both pass a bare `n_vars < adata.n_vars` premise — the first
        # at `n_vars == 0` — so assert both bounds.
        window = _dense_view(a)
        pyscx.accel.filter_genes(a, min_cells=int(np.median((window > 0).sum(axis=0))))
        assert 0 < a.n_vars < adata.n_vars, (
            f"premise: the gene filter must drop some genes and keep some; "
            f"kept {a.n_vars} of {adata.n_vars}"
        )
    if transforms:
        pyscx.accel.normalize_total(a, target_sum=1e4)
        pyscx.accel.log1p(a)
    return a


@pytest.mark.parametrize("cols", [False, True], ids=["rows", "rows+cols"])
@pytest.mark.parametrize("transforms", [False, True], ids=["raw", "norm_log1p"])
def test_a_row_filtered_handle_serves_csc_column_reductions(tmp_path, cols, transforms):
    """The four combinations of window and chain, on `col_sums` and `col_var`.

    The `raw` arms are the ones the unification adds: with no transform chain
    the handle is still an `ScxBackedSparseDataset`, whose column source used
    to be the full-axis sidecar reader and so had to refuse a window outright.

    The oracle is the materialised window, not the CSR route:
    `col_sums(prefer_format="csr")` refuses a lazy handle outright, so on the
    `norm_log1p` arms there is no CSR route to compare against — and numpy
    over `a.X[:]` cannot be wrong in the same way a shared kernel bug would
    make both routes wrong.
    """
    import pyscx

    adata = _counts_adata()
    path = tmp_path / "counts_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=6)

    a = _filtered_handle(path, adata, cols=cols, transforms=transforms)
    csc_sums = np.asarray(pyscx.accel.col_sums(a.X, prefer_format="csc"))
    csc_var = np.asarray(pyscx.accel.col_var(a.X, prefer_format="csc"))
    dense = _dense_view(a)

    assert csc_sums.shape == (a.n_vars,)
    np.testing.assert_allclose(csc_sums, dense.sum(axis=0).astype(np.float64), rtol=1e-5)
    np.testing.assert_allclose(csc_var, dense.var(axis=0).astype(np.float64), rtol=1e-4)

    if not transforms:
        # A backed handle also has a CSR route, so pin the two against each
        # other where both are available.
        b = _filtered_handle(path, adata, cols=cols, transforms=False)
        np.testing.assert_allclose(
            csc_sums, np.asarray(pyscx.accel.col_sums(b.X, prefer_format="csr")), rtol=1e-6
        )
        np.testing.assert_allclose(
            csc_var, np.asarray(pyscx.accel.col_var(b.X, prefer_format="csr")), rtol=1e-6
        )


def test_the_full_pipeline_shape_routes_csc_and_matches_csr(tmp_path):
    """`filter_cells` + `filter_genes` + `normalize_total` + `log1p` → DE.

    All four conditions at once, which is what `pipeline_ooc_constrained` and
    any ordinary scanpy-shaped workflow presents. Before this change the row
    filter alone forced `cpu_csr`; `auto` is DE's default, so this is what a
    caller who passes no `prefer_format` now gets.
    """
    import pyscx

    adata = _counts_adata()
    path = tmp_path / "pipeline_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=6)

    def de(prefer):
        a = _filtered_handle(path, adata, cols=True)
        pyscx.accel.rank_genes_groups(
            a, groupby="grp", method="wilcoxon", device="cpu", prefer_format=prefer
        )
        route = (a.uns.get("scx_accel") or {}).get("rank_genes_groups", {}).get("route")
        return route, a.uns["rank_genes_groups"]

    route_csr, res_csr = de("csr")
    route_csc, res_csc = de("csc")
    route_auto, res_auto = de("auto")

    assert route_csr == "cpu_csr"
    assert route_csc == "cpu_csc"
    assert route_auto == "cpu_csc"

    for field in ("scores", "pvals", "logfoldchanges"):
        a = np.array([list(r) for r in res_csr[field]])
        b = np.array([list(r) for r in res_csc[field]])
        assert a.shape == b.shape
        np.testing.assert_array_equal(a, b, err_msg=f"{field} differs between routes")
    assert np.array_equal(
        np.array([list(r) for r in res_csr["names"]]),
        np.array([list(r) for r in res_auto["names"]]),
    )


def test_a_row_filter_that_empties_whole_csr_shards_still_reads(tmp_path):
    """A window confined to one shard.

    On the CSR side `visible_shard_indices` skips the emptied shards. The CSC
    side has no equivalent — every column shard spans the whole row axis — so
    this is a test that the compaction handles a slab where most rows are gone
    rather than a test of shard skipping.
    """
    import pyscx

    adata = _counts_adata(n_obs=120)
    path = tmp_path / "shards_csc.scx"
    pyscx.from_anndata(adata, str(path), shard_size=20, csc="always", csc_cols_per_shard=6)
    assert pyscx.open(str(path)).shard_count > 1, "premise: more than one CSR shard"

    a = pyscx.open(str(path)).to_anndata(backed=True)
    # Keep only rows 0..15 — the first shard, partially.
    a = a[np.arange(16)]
    assert a.n_obs == 16
    pyscx.accel.normalize_total(a, target_sum=1e4)
    csc_sums = np.asarray(pyscx.accel.col_sums(a.X, prefer_format="csc"))

    dense = adata.X.toarray()[:16]
    scale = 1e4 / dense.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(
        csc_sums, (dense * scale).sum(axis=0).astype(np.float64), rtol=1e-5
    )


@pytest.mark.parametrize("cols", [False, True], ids=["rows", "rows+cols"])
def test_a_filtered_qc_gene_axis_matches_the_csr_route(tmp_path, cols):
    """`calculate_qc_metrics`' gene axis, which reads `col_sums_and_nnz`.

    Its per-gene `n_cells_by_counts` is an nnz count off `indptr` deltas, so a
    slab still carrying dropped rows' nonzeros inflates it — a wrong answer
    that no bounds check can catch.

    The `rows+cols` arm is also the one case where this op's *column* space
    changed: its backed arm used to pass `col_projection()`'s global ids to a
    full-axis reader, and now passes identity positions to a projected view.
    Passing the global ids to the view would apply the projection twice.
    """
    import pyscx

    adata = _counts_adata()
    path = tmp_path / "qc_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=6)

    def qc(prefer):
        a = _filtered_handle(path, adata, cols=cols, transforms=False)
        pyscx.accel.calculate_qc_metrics(a, inplace=True, prefer_format=prefer)
        return (
            np.asarray(a.var["total_counts"], dtype=np.float64),
            np.asarray(a.var["n_cells_by_counts"], dtype=np.float64),
            _dense_view(a),
        )

    csc_total, csc_cells, dense = qc("csc")
    csr_total, csr_cells, _ = qc("csr")
    np.testing.assert_allclose(csc_total, csr_total, rtol=1e-6)
    np.testing.assert_array_equal(csc_cells, csr_cells)
    np.testing.assert_allclose(csc_total, dense.sum(axis=0), rtol=1e-5)
    np.testing.assert_array_equal(csc_cells, (dense > 0).sum(axis=0))


def test_a_filtered_hvg_csc_route_matches_csr(tmp_path):
    """HVG's CSC mean/var kernel divides by `source.n_obs()`.

    So it is wrong in two directions at once on an uncompacted slab: the sums
    include dropped rows while the divisor counts only live ones.
    """
    import pyscx

    pytest.importorskip("skmisc")
    adata = _counts_adata(n_obs=200, n_vars=60, seed=9)
    path = tmp_path / "hvg_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=10)

    def hvg(prefer, filtered=True):
        a = _filtered_handle(path, adata, rows=filtered, transforms=False)
        pyscx.accel.highly_variable_genes(
            a, n_top_genes=20, flavor="seurat_v3", span=1.0, prefer_format=prefer, device="cpu"
        )
        return (
            np.asarray(a.var["highly_variable"].values),
            np.asarray(a.var["variances"], dtype=np.float64),
            a.uns["scx_accel"]["highly_variable_genes"],
        )

    csc_flags, csc_var, csc_info = hvg("csc")
    csr_flags, csr_var, csr_info = hvg("csr")
    np.testing.assert_array_equal(csc_flags, csr_flags)
    np.testing.assert_allclose(csc_var, csr_var, rtol=1e-6)

    # The route stamp, not just the numbers. `hvg_exec_info`'s `csc_available`
    # argument is a *route* input — `plan_hvg_route` picks `CpuCsc` from it —
    # so an attempt to report the real capability through it stamped `cpu_csc`
    # over the CSR kernel on every sidecar-carrying file, filtered or not.
    # Comparing variances cannot see that; this can.
    assert csc_info["route"] == "cpu_csc"
    assert csr_info["route"] == "cpu_csr", (
        f"the default CSR path must record cpu_csr, got {csr_info['route']!r}"
    )
    assert csr_info["csc_available"] is True, (
        "the file has a sidecar, and the CSR stamp must say so without claiming "
        "the CSC route ran"
    )

    # Same stamp, unfiltered — the shape where nothing declined CSC and the
    # capability is equally true.
    _, _, unfiltered_info = hvg("csr", filtered=False)
    assert unfiltered_info["route"] == "cpu_csr"
    assert unfiltered_info["csc_available"] is True


def test_a_filtered_pseudobulk_dex_matches_csr(tmp_path):
    """The CSC pseudobulk aggregation reads `cell_to_group[row]`, a
    **visible**-length vector.

    An uncompacted slab indexes it with a global row, so a cell's counts land
    in another cell's group — and under a non-prefix filter that is a
    different group, not a missing one. `pseudobulk_dex` is the only entry
    point onto that kernel, so the comparison runs through the whole op; the
    aggregation is what differs between the routes, the GLM is not.
    """
    import pyscx

    adata = _counts_adata(n_obs=150, n_vars=30, seed=3)
    # A replicate column, so the default NB-GLM backend has something to fit.
    adata.obs["donor"] = pd.Categorical([f"d{i % 2}" for i in range(adata.n_obs)])
    path = tmp_path / "pb_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=8)

    def pb(prefer):
        a = _filtered_handle(path, adata, cols=True, transforms=False)
        a.obs["donor"] = pd.Categorical(adata.obs["donor"].to_numpy()[: a.n_obs])
        return pyscx.accel.pseudobulk_dex(
            a,
            groupby=["grp", "donor"],
            test_col="grp",
            reference="A",
            min_cells_per_group=1,
            prefer_format=prefer,
        )

    csc = pb("csc")
    csr = pb("csr")
    assert len(csc) > 0, "premise: the op produced results to compare"
    assert list(csc.columns) == list(csr.columns)
    csc = csc.sort_values(list(csc.columns[:2])).reset_index(drop=True)
    csr = csr.sort_values(list(csr.columns[:2])).reset_index(drop=True)
    for col in csr.columns:
        if np.issubdtype(np.asarray(csr[col]).dtype, np.number):
            np.testing.assert_allclose(
                np.asarray(csc[col], dtype=np.float64),
                np.asarray(csr[col], dtype=np.float64),
                rtol=1e-5,
                err_msg=f"{col} differs between routes",
            )
        else:
            assert list(csc[col]) == list(csr[col]), col


def test_a_filtered_pdex_ref_matches_csr(tmp_path):
    """The second DE entry point, which shares the dense scatter but not the
    statistic."""
    import pyscx

    adata = _counts_adata()
    path = tmp_path / "pdex_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=6)

    def run(prefer):
        a = _filtered_handle(path, adata)
        res = pyscx.accel.pdex_ref(
            a, "grp", reference="A", device="cpu", prefer_format=prefer
        )
        route = (a.uns.get("scx_accel") or {}).get("pdex_ref", {}).get("route")
        return route, res

    route_csc, res_csc = run("csc")
    route_csr, res_csr = run("csr")
    assert route_csc == "cpu_csc"
    assert route_csr == "cpu_csr"
    for col in res_csr.columns:
        if np.issubdtype(np.asarray(res_csr[col]).dtype, np.number):
            np.testing.assert_allclose(
                np.asarray(res_csc[col], dtype=np.float64),
                np.asarray(res_csr[col], dtype=np.float64),
                rtol=1e-6,
                err_msg=f"{col} differs between routes",
            )
        else:
            assert list(res_csc[col]) == list(res_csr[col]), col


# ---------------------------------------------------------------------------
# The exact-nnz Wilcoxon kernel under a row filter.
#
# `SCX_ACCEL_WILCOXON_NNZ=1` swaps in a structurally different kernel that does
# not densify: it sorts each gene's stored values and derives the zero
# tie-block arithmetically, as `n_obs - (n_neg + n_pos)` pooled and
# `group_cell_counts[g] - nonzero_in_g[g]` per bucket. Both sides of those
# subtractions have to be live counts. If a slab still carried dropped rows'
# nonzeros the pooled count could exceed `n_obs`, which the kernel reports as
# a non-canonical sidecar — or, worse, not exceed it and quietly widen the
# rank block.
# ---------------------------------------------------------------------------

_FILTERED_NNZ_PROBE = r"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, pyscx

rng = np.random.default_rng(7)
mat = sp.random(120, 24, density=0.5, format="csr", dtype=np.float32, random_state=rng)
mat.data = (mat.data * 100).astype(np.float32).round() + 1.0
adata = ad.AnnData(X=mat)
adata.obs["cell_id"] = [f"c{i}" for i in range(120)]
adata.obs["grp"] = pd.Categorical(["A"] * 60 + ["B"] * 60)
adata.var["gene_id"] = [f"g{i}" for i in range(24)]

path = sys.argv[1]
pyscx.from_anndata(adata, path, csc="always", csc_cols_per_shard=6)
thresh = int(np.median(np.diff(adata.X.indptr)))


def run(prefer):
    a = pyscx.open(path).to_anndata(backed=True)
    a.obs["grp"] = pd.Categorical(adata.obs["grp"].to_numpy())
    pyscx.accel.filter_cells(a, min_genes=thresh)
    assert a.n_obs < 120, "premise: the row filter dropped cells"
    pyscx.accel.rank_genes_groups(
        a, "grp", device="cpu", prefer_format=prefer, reference="rest"
    )
    route = a.uns["scx_accel"]["rank_genes_groups"]["route"]
    scores = np.array([list(r) for r in a.uns["rank_genes_groups"]["scores"]])
    names = np.array([list(r) for r in a.uns["rank_genes_groups"]["names"]])
    return route, scores, names


route_csc, scores_csc, names_csc = run("csc")
route_csr, scores_csr, names_csr = run("csr")
print(route_csc)
print(route_csr)
print(int(np.array_equal(scores_csc, scores_csr)))
print(int(np.array_equal(names_csc, names_csr)))
"""


def test_the_exact_nnz_wilcoxon_kernel_serves_a_filtered_handle(tmp_path):
    """Subprocess, not tidiness: the gate is read through a `OnceLock`, so
    setting it inside an interpreter that has already taken the CSC path once
    has no effect."""
    pytest.importorskip("anndata")
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env["SCX_ACCEL_WILCOXON_NNZ"] = "1"
    out = subprocess.run(
        [sys.executable, "-c", _FILTERED_NNZ_PROBE, str(tmp_path / "nnz.scx")],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln for ln in out.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 4, out.stdout + out.stderr
    assert lines[0] == "cpu_csc_nnz", f"the gate must select the nnz kernel: {lines[0]}"
    assert lines[1] == "cpu_csr"
    assert lines[2] == "1", f"scores differ between the nnz CSC kernel and CSR\n{out.stderr}"
    assert lines[3] == "1", f"gene order differs between the nnz CSC kernel and CSR\n{out.stderr}"


# ---------------------------------------------------------------------------
# Presentation order on the CSC path: unchanged, and pinned so it stays that
# way rather than being assumed.
# ---------------------------------------------------------------------------


def test_the_csc_column_reductions_answer_in_sorted_projection_order(tmp_path):
    """`col_sums` on the CSC path reports sorted-projection order, not the
    caller's requested order — before this change and after it.

    Not an endorsement: `col_aggs` is one of the two CSC consumers that does
    not call `reject_presentation_ordered_source`, so a `preserve_var_order`
    handle has always been answered in sorted order here while the CSR path
    reorders. The unification does not touch that — the backed path passed the
    (sorted) `col_projection()` before and the view emits sorted-projection
    order now — and this test says so, so a future change to either has to
    decide deliberately.
    """
    import pyscx

    adata = _counts_adata(n_obs=60, n_vars=12, seed=2)
    path = tmp_path / "order_csc.scx"
    pyscx.from_anndata(adata, str(path), csc="always", csc_cols_per_shard=4)

    a = pyscx.open(str(path)).to_anndata(backed=True)
    requested = [5, 1, 9]
    a = a[:, requested]
    csc_sums = np.asarray(pyscx.accel.col_sums(a.X, prefer_format="csc"))

    dense = adata.X.toarray()
    np.testing.assert_allclose(
        csc_sums, dense[:, sorted(requested)].sum(axis=0).astype(np.float64), rtol=1e-6
    )


# ---------------------------------------------------------------------------
# Capability is not policy: `auto` must not follow a narrow row window into a
# scan CSC cannot prune.
#
# A CSC column shard spans the whole row axis, so a window confined to a few
# CSR shards still reads every physical cell of the columns it asks for, where
# the CSR path skips the emptied shards outright. `csc_preferred_for_auto` is
# the second question; `prefer_format="csc"` skips it.
# ---------------------------------------------------------------------------


def _multishard_csc_file(tmp_path, adata, name, shard_size=20):
    import pyscx

    path = tmp_path / name
    pyscx.from_anndata(
        adata, str(path), shard_size=shard_size, csc="always", csc_cols_per_shard=6
    )
    exp = pyscx.open(str(path))
    assert exp.shard_count >= 4, f"premise: need several CSR shards, got {exp.shard_count}"
    assert exp.has_csc
    return path


def test_a_narrow_row_window_leaves_auto_on_csr(tmp_path):
    """One shard's worth of rows out of six: `auto` stays CSR."""
    import pyscx

    adata = _counts_adata(n_obs=120, n_vars=24)
    path = _multishard_csc_file(tmp_path, adata, "narrow.scx")

    def de(prefer):
        a = pyscx.open(str(path)).to_anndata(backed=True)
        a.obs["grp"] = pd.Categorical(adata.obs["grp"].to_numpy())
        a = a[np.arange(18)]  # rows 0..17 — shard 0 and a sliver of shard 1
        assert a.n_obs == 18
        pyscx.accel.rank_genes_groups(
            a, groupby="grp", method="wilcoxon", device="cpu", prefer_format=prefer
        )
        return a.uns["scx_accel"]["rank_genes_groups"], a.uns["rank_genes_groups"]

    info_auto, res_auto = de("auto")
    assert info_auto["route"] == "cpu_csr", (
        f"a window spanning 2 of 6 shards must not auto-route to CSC, "
        f"got {info_auto['route']!r}"
    )
    # The route is only half of it. `csc_available` is documented as whether a
    # sidecar was *available* at dispatch, and the benchmark route gates read it
    # to tell a silent CSC→CSR fallback from a file that never had a sidecar.
    # Declining CSC on policy must not report the file as sidecar-less — which
    # is exactly what the first version of this policy did, because it expressed
    # the decision by removing CSC from the kernel input.
    assert info_auto["csc_available"] is True, (
        "the file has a sidecar; a policy decline must not claim otherwise"
    )

    # The capability is still there for a caller who asks for it, and it agrees.
    info_csc, res_csc = de("csc")
    assert info_csc["route"] == "cpu_csc"
    assert info_csc["csc_available"] is True
    for field in ("scores", "pvals", "logfoldchanges"):
        np.testing.assert_array_equal(
            np.array([list(r) for r in res_auto[field]]),
            np.array([list(r) for r in res_csc[field]]),
            err_msg=f"{field} differs between the CSR and CSC routes",
        )


def test_a_wide_row_window_still_auto_routes_csc(tmp_path):
    """The complement, and the premise for the test above.

    Without it, `csc_preferred_for_auto` could return `false` unconditionally
    and the narrow-window assertion would still pass — while the PR's whole
    payoff had quietly reverted. Every other cell, so every shard keeps
    survivors and the CSR path has nothing to skip.
    """
    import pyscx

    adata = _counts_adata(n_obs=120, n_vars=24)
    path = _multishard_csc_file(tmp_path, adata, "wide.scx")

    a = pyscx.open(str(path)).to_anndata(backed=True)
    a.obs["grp"] = pd.Categorical(adata.obs["grp"].to_numpy())
    a = a[np.arange(0, 120, 2)]
    assert a.n_obs == 60
    pyscx.accel.normalize_total(a, target_sum=1e4)
    pyscx.accel.log1p(a)
    pyscx.accel.rank_genes_groups(a, groupby="grp", method="wilcoxon", device="cpu")
    info = a.uns["scx_accel"]["rank_genes_groups"]
    assert info["route"] == "cpu_csc"
    assert info["csc_available"] is True
