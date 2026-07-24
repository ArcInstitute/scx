"""Regression tests: ``calculate_qc_metrics`` must honor a column projection.

Before this fix the default ``prefer_format="csr"`` route ignored
``col_projection`` entirely:

* per-cell ``total_counts`` summed genes hidden by the projection;
* the gene axis came back at *physical* width, misaligning against ``adata.var``;
* ``qc_var`` masks were read in the visible axis but applied to the underlying
  on-disk axis, so a projected dataset scored the wrong genes;
* on lazy data the ``qc_var`` subset sums were read **pre**-transform while
  ``total_counts`` was post-transform, making ``pct_counts_<v>`` a ratio of two
  different matrices.

The CSC route (``prefer_format="csc"``) already handled the projection, so these
tests also pin CSR/CSC agreement under a projection.
"""

import numpy as np
import pytest


@pytest.fixture
def qc_adata():
    """A 100 x 50 counts AnnData with **no layers and no obsm**.

    Deliberately leaner than the shared ``synthetic_adata`` fixture: both
    ``filter_genes`` and ``filter_cells`` currently trip anndata's
    aligned-mapping validation on a backed dataset that carries a layer /
    ``obsm`` entry, because they assign the sliced ``_var`` / ``_obs`` before the
    aligned members have been re-projected. Those are separate pre-existing
    defects; excluding the members keeps these tests measuring QC semantics.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.RandomState(42)
    n_obs, n_vars = 100, 50
    dense = rng.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random_sample((n_obs, n_vars)) > 0.3] = 0
    return anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(n_vars)]),
    )


@pytest.fixture
def scx_path(qc_adata, tmp_dir):
    """A 100 x 50 SCX file."""
    import pyscx

    path = str(tmp_dir / "qc_proj.scx")
    pyscx.from_anndata(qc_adata, path)
    return path


@pytest.fixture
def dense_ref(qc_adata):
    """The same matrix as a dense float64 array, for numpy references."""
    return np.asarray(qc_adata.X.todense(), dtype=np.float64)


# Deliberately non-contiguous and not starting at 0, so a visible-space index
# never coincides with its on-disk index.
GENE_SUBSET = [5, 8, 13, 21, 22, 30, 34, 41, 47, 49]


def _project(adata, subset):
    """Apply a column projection to a backed/lazy X and slice var to match."""
    adata.X.set_col_projection([int(c) for c in subset])
    # `_var` bypasses anndata's shape validation (X.shape.1 already changed).
    adata._var = adata.var.iloc[subset].copy()
    return adata


def _expect_row_totals(dense, subset):
    return dense[:, subset].sum(axis=1)


def _expect_row_nnz(dense, subset):
    return (dense[:, subset] != 0).sum(axis=1)


def _expect_pct(subset_sums, totals):
    return np.divide(
        subset_sums * 100.0, totals, out=np.zeros_like(totals), where=totals > 0
    )


# ---------------------------------------------------------------------------
# Backed, projection active
# ---------------------------------------------------------------------------


def test_backed_projection_cell_axis(scx_path, dense_ref):
    """total_counts / n_genes_by_counts cover only the visible genes."""
    import pyscx

    adata = _project(pyscx.open(scx_path).to_anndata(backed=True), GENE_SUBSET)
    pyscx.accel.calculate_qc_metrics(adata)

    np.testing.assert_allclose(
        adata.obs["total_counts"].to_numpy(dtype=np.float64),
        _expect_row_totals(dense_ref, GENE_SUBSET),
        rtol=1e-6,
        err_msg="total_counts must not include genes hidden by the projection",
    )
    np.testing.assert_array_equal(
        adata.obs["n_genes_by_counts"].to_numpy(dtype=np.int64),
        _expect_row_nnz(dense_ref, GENE_SUBSET),
    )


def test_backed_projection_gene_axis(scx_path, dense_ref):
    """The gene axis is visible-width and aligned with adata.var."""
    import pyscx

    adata = _project(pyscx.open(scx_path).to_anndata(backed=True), GENE_SUBSET)
    pyscx.accel.calculate_qc_metrics(adata)

    assert len(adata.var) == len(GENE_SUBSET)
    np.testing.assert_allclose(
        adata.var["total_counts"].to_numpy(dtype=np.float64),
        dense_ref[:, GENE_SUBSET].sum(axis=0),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        adata.var["n_cells_by_counts"].to_numpy(dtype=np.int64),
        (dense_ref[:, GENE_SUBSET] != 0).sum(axis=0),
    )


def test_backed_projection_qc_var_selects_visible_genes(scx_path, dense_ref):
    """qc_var mask positions index adata.var, not the on-disk gene axis.

    The mask marks visible positions 0 and 1, i.e. on-disk genes 5 and 8. Read
    without composing through the projection, positions 0/1 would select on-disk
    genes 0/1 — which are not even in the visible set.
    """
    import pyscx

    adata = _project(pyscx.open(scx_path).to_anndata(backed=True), GENE_SUBSET)
    mask = np.zeros(len(GENE_SUBSET), dtype=bool)
    mask[[0, 1]] = True
    adata.var["mt"] = mask

    pyscx.accel.calculate_qc_metrics(adata, qc_vars=["mt"])

    mt_ondisk = [GENE_SUBSET[0], GENE_SUBSET[1]]
    expected_subset = dense_ref[:, mt_ondisk].sum(axis=1)
    np.testing.assert_allclose(
        adata.obs["total_counts_mt"].to_numpy(dtype=np.float64),
        expected_subset,
        rtol=1e-6,
    )

    totals = _expect_row_totals(dense_ref, GENE_SUBSET)
    np.testing.assert_allclose(
        adata.obs["pct_counts_mt"].to_numpy(dtype=np.float64),
        _expect_pct(expected_subset, totals),
        rtol=1e-6,
    )

    # Sanity: the wrong (uncomposed) answer is genuinely different, so this
    # test would have failed before the fix.
    wrong = dense_ref[:, [0, 1]].sum(axis=1)
    assert not np.allclose(wrong, expected_subset)


def test_backed_projection_via_filter_genes(scx_path, dense_ref):
    """The realistic path: filter_genes sets the projection, then QC runs."""
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.filter_genes(adata, min_cells=25)
    kept = np.asarray((dense_ref != 0).sum(axis=0) >= 25).nonzero()[0]
    assert 0 < len(kept) < dense_ref.shape[1], "fixture must actually drop genes"

    pyscx.accel.calculate_qc_metrics(adata)

    np.testing.assert_allclose(
        adata.obs["total_counts"].to_numpy(dtype=np.float64),
        dense_ref[:, kept].sum(axis=1),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        adata.var["total_counts"].to_numpy(dtype=np.float64),
        dense_ref[:, kept].sum(axis=0),
        rtol=1e-6,
    )


def test_backed_projection_with_deletions(scx_path, dense_ref):
    """Projection and deletion vector compose on both axes."""
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.filter_cells(adata, min_counts=1500)
    keep_rows = np.asarray(dense_ref.sum(axis=1) >= 1500).nonzero()[0]
    assert 0 < len(keep_rows) < dense_ref.shape[0], "fixture must actually drop cells"

    _project(adata, GENE_SUBSET)
    pyscx.accel.calculate_qc_metrics(adata)

    sub = dense_ref[np.ix_(keep_rows, GENE_SUBSET)]
    np.testing.assert_allclose(
        adata.obs["total_counts"].to_numpy(dtype=np.float64),
        sub.sum(axis=1),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        adata.var["total_counts"].to_numpy(dtype=np.float64),
        sub.sum(axis=0),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        adata.var["n_cells_by_counts"].to_numpy(dtype=np.int64),
        (sub != 0).sum(axis=0),
    )


# ---------------------------------------------------------------------------
# Lazy (transform chain active)
# ---------------------------------------------------------------------------


def _normalized_ref(dense, target_sum=1e4):
    """`normalize_total` semantics: scale each row to `target_sum`.

    The denominator is the row sum of the columns **visible at the time
    normalize_total ran** — after `filter_genes` scanpy sums only the kept
    genes, and SCX matches that (`row_sums_projected`). So callers pass the
    already-subset matrix when a projection is active.
    """
    row_sums = dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return dense / row_sums * target_sum


def test_lazy_projection_cell_and_gene_axes(scx_path, dense_ref):
    """Lazy QC honors the projection on both axes, post-transform."""
    import pyscx

    # Project first: `normalize_total` wraps the backed X in a lazy dataset that
    # inherits `col_projection` (the lazy class exposes no Python setter).
    adata = _project(pyscx.open(scx_path).to_anndata(backed=True), GENE_SUBSET)
    pyscx.accel.normalize_total(adata, target_sum=1e4)

    pyscx.accel.calculate_qc_metrics(adata)

    # The projection was active when normalize_total ran, so it scaled each row
    # by the *visible* row sum; QC then sums that normalized visible submatrix.
    norm_sub = _normalized_ref(dense_ref[:, GENE_SUBSET])
    np.testing.assert_allclose(
        adata.obs["total_counts"].to_numpy(dtype=np.float64),
        norm_sub.sum(axis=1),
        rtol=1e-5,
    )
    assert len(adata.var) == len(GENE_SUBSET)
    np.testing.assert_allclose(
        adata.var["total_counts"].to_numpy(dtype=np.float64),
        norm_sub.sum(axis=0),
        rtol=1e-5,
    )


def test_lazy_qc_var_is_post_transform(scx_path, dense_ref):
    """pct_counts_<v> divides a transformed numerator by a transformed total."""
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    adata.var["mt"] = np.isin(np.arange(dense_ref.shape[1]), [3, 4, 5])

    pyscx.accel.calculate_qc_metrics(adata, qc_vars=["mt"])

    norm = _normalized_ref(dense_ref)
    expected_subset = norm[:, [3, 4, 5]].sum(axis=1)
    np.testing.assert_allclose(
        adata.obs["total_counts_mt"].to_numpy(dtype=np.float64),
        expected_subset,
        rtol=1e-5,
        err_msg="lazy qc_var sums must stream through the transform chain",
    )

    totals = adata.obs["total_counts"].to_numpy(dtype=np.float64)
    expected_pct = np.where(totals > 0, expected_subset / totals * 100.0, 0.0)
    np.testing.assert_allclose(
        adata.obs["pct_counts_mt"].to_numpy(dtype=np.float64),
        expected_pct,
        rtol=1e-5,
    )

    # The pre-transform numerator differs materially — the old behavior.
    raw_subset = dense_ref[:, [3, 4, 5]].sum(axis=1)
    assert not np.allclose(raw_subset, expected_subset)


def test_lazy_projection_qc_var_composed(scx_path, dense_ref):
    """Projection + transform + qc_var all compose."""
    import pyscx

    adata = _project(pyscx.open(scx_path).to_anndata(backed=True), GENE_SUBSET)
    pyscx.accel.normalize_total(adata, target_sum=1e4)
    mask = np.zeros(len(GENE_SUBSET), dtype=bool)
    mask[[2, 3]] = True
    adata.var["mt"] = mask

    pyscx.accel.calculate_qc_metrics(adata, qc_vars=["mt"])

    norm_sub = _normalized_ref(dense_ref[:, GENE_SUBSET])
    np.testing.assert_allclose(
        adata.obs["total_counts_mt"].to_numpy(dtype=np.float64),
        norm_sub[:, [2, 3]].sum(axis=1),
        rtol=1e-5,
    )


# ---------------------------------------------------------------------------
# Guards: unprojected behavior unchanged, CSR/CSC agreement under projection
# ---------------------------------------------------------------------------


def test_no_projection_unchanged(scx_path, dense_ref):
    """Without a projection the results are the plain full-matrix statistics."""
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    adata.var["mt"] = np.isin(np.arange(dense_ref.shape[1]), [0, 1, 2])
    pyscx.accel.calculate_qc_metrics(adata, qc_vars=["mt"])

    np.testing.assert_allclose(
        adata.obs["total_counts"].to_numpy(dtype=np.float64),
        dense_ref.sum(axis=1),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        adata.var["total_counts"].to_numpy(dtype=np.float64),
        dense_ref.sum(axis=0),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        adata.obs["total_counts_mt"].to_numpy(dtype=np.float64),
        dense_ref[:, [0, 1, 2]].sum(axis=1),
        rtol=1e-6,
    )


def test_csr_matches_csc_under_projection(synthetic_adata, tmp_dir, dense_ref):
    """The CSC gene-axis route already honored the projection; CSR now agrees."""
    import pyscx

    path = str(tmp_dir / "qc_proj_csc.scx")
    pyscx.from_anndata(synthetic_adata, path, csc="always")

    a_csr = _project(pyscx.open(path).to_anndata(backed=True), GENE_SUBSET)
    a_csc = _project(pyscx.open(path).to_anndata(backed=True), GENE_SUBSET)
    pyscx.accel.calculate_qc_metrics(a_csr)
    pyscx.accel.calculate_qc_metrics(a_csc, prefer_format="csc")

    for col in ("total_counts", "n_cells_by_counts"):
        np.testing.assert_allclose(
            a_csr.var[col].to_numpy(dtype=np.float64),
            a_csc.var[col].to_numpy(dtype=np.float64),
            rtol=1e-9,
            err_msg=f"var[{col}] must agree across prefer_format under a projection",
        )
