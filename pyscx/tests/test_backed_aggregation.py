"""Integration tests for backed mode aggregation operations.

Tests var(), max(), min() native Rust paths and deletion-vector-aware
column aggregation (sum/mean/getnnz/var/max/min with axis=0).
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_scx(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic_adata for backed tests."""
    import pyscx

    path = str(tmp_dir / "backed_agg.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def adata_backed(backed_scx):
    """Load the backed AnnData."""
    import pyscx

    return pyscx.open(backed_scx).to_anndata(backed=True)


@pytest.fixture
def adata_non_backed(backed_scx):
    """Load the full (non-backed) AnnData for comparison."""
    import pyscx

    return pyscx.open(backed_scx).to_anndata()


# --- Variance tests ---


def test_var_axis0(adata_backed, adata_non_backed):
    """var(axis=0) on backed matches scipy CSR var(axis=0)."""
    backed_var = np.asarray(adata_backed.X.var(axis=0)).flatten()
    scipy_var = np.asarray(adata_non_backed.X.toarray().var(axis=0)).flatten()
    # rtol=1e-5: Rust computes in f64 throughout; numpy accumulates f32 intermediates
    np.testing.assert_allclose(backed_var, scipy_var, rtol=1e-5)


def test_var_axis1(adata_backed, adata_non_backed):
    """var(axis=1) on backed matches scipy CSR var(axis=1)."""
    backed_var = np.asarray(adata_backed.X.var(axis=1)).flatten()
    scipy_var = np.asarray(adata_non_backed.X.toarray().var(axis=1)).flatten()
    np.testing.assert_allclose(backed_var, scipy_var, rtol=1e-5)


def test_var_no_axis(adata_backed, adata_non_backed):
    """var() (scalar) on backed matches numpy dense var()."""
    backed_var = adata_backed.X.var()
    numpy_var = adata_non_backed.X.toarray().var()
    np.testing.assert_allclose(backed_var, numpy_var, rtol=1e-5)


# --- getnnz dtype parity (B5) ---


@pytest.mark.parametrize("axis", [0, 1])
def test_getnnz_dtype_is_int32(adata_backed, adata_non_backed, axis):
    """getnnz returns a uniform int32 for both axes (was uint32 on axis=0,
    int64 on axis=1). scipy's own getnnz dtype is platform-inconsistent
    across axes, so we pin our output to int32 and check value parity."""
    backed_nnz = adata_backed.X.getnnz(axis=axis)
    scipy_nnz = adata_non_backed.X.getnnz(axis=axis)
    assert backed_nnz.dtype == np.int32
    np.testing.assert_array_equal(backed_nnz, scipy_nnz)


# --- Max tests ---


def test_max_axis0(adata_backed, adata_non_backed):
    """max(axis=0) on backed matches scipy CSR max(axis=0)."""
    backed_max = np.asarray(adata_backed.X.max(axis=0)).flatten()
    scipy_max = np.asarray(adata_non_backed.X.max(axis=0).toarray()).flatten()
    np.testing.assert_array_equal(backed_max, scipy_max)


def test_max_axis1(adata_backed, adata_non_backed):
    """max(axis=1) on backed matches scipy CSR max(axis=1)."""
    backed_max = np.asarray(adata_backed.X.max(axis=1)).flatten()
    scipy_max = np.asarray(adata_non_backed.X.max(axis=1).toarray()).flatten()
    np.testing.assert_array_equal(backed_max, scipy_max)


def test_max_no_axis(adata_backed, adata_non_backed):
    """max() on backed matches scipy CSR max()."""
    backed_max = adata_backed.X.max()
    scipy_max = adata_non_backed.X.max()
    assert backed_max == scipy_max


# --- Min tests ---


def test_min_axis0(adata_backed, adata_non_backed):
    """min(axis=0) on backed matches scipy CSR min(axis=0)."""
    backed_min = np.asarray(adata_backed.X.min(axis=0)).flatten()
    scipy_min = np.asarray(adata_non_backed.X.min(axis=0).toarray()).flatten()
    np.testing.assert_array_equal(backed_min, scipy_min)


def test_min_axis1(adata_backed, adata_non_backed):
    """min(axis=1) on backed matches scipy CSR min(axis=1)."""
    backed_min = np.asarray(adata_backed.X.min(axis=1)).flatten()
    scipy_min = np.asarray(adata_non_backed.X.min(axis=1).toarray()).flatten()
    np.testing.assert_array_equal(backed_min, scipy_min)


def test_min_no_axis(adata_backed, adata_non_backed):
    """min() on backed matches scipy CSR min()."""
    backed_min = adata_backed.X.min()
    scipy_min = adata_non_backed.X.min()
    assert backed_min == scipy_min


# --- Deletion vector tests ---


def _make_deleted_file(tmp_dir, name="del_agg.scx"):
    """Create an SCX file with deletion vectors and return (path, delete_mask)."""
    import anndata
    import pyscx

    np.random.seed(99)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)
    return path, delete_mask


def test_col_sums_with_deletions(tmp_dir):
    """Column sums with deletion vectors match filtered-matrix sums."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_col_sums.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_sums = np.asarray(adata_backed.X.sum(axis=0)).flatten()
    full_sums = np.asarray(adata_full.X.sum(axis=0)).flatten()

    np.testing.assert_allclose(backed_sums, full_sums, rtol=1e-6)


def test_col_nnz_with_deletions(tmp_dir):
    """Column NNZ with deletion vectors match filtered-matrix counts."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_col_nnz.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_nnz = np.asarray(adata_backed.X.getnnz(axis=0)).flatten()
    full_nnz = np.asarray(adata_full.X.getnnz(axis=0)).flatten()

    np.testing.assert_array_equal(backed_nnz, full_nnz)


def test_var_with_deletions(tmp_dir):
    """var(axis=0) with deletions matches materialized subset .var()."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_var.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_var = np.asarray(adata_backed.X.var(axis=0)).flatten()
    full_var = np.asarray(adata_full.X.toarray().var(axis=0)).flatten()

    np.testing.assert_allclose(backed_var, full_var, rtol=1e-5)



def test_var_no_axis_with_deletions(tmp_dir):
    """var() (scalar) with deletions matches materialized subset .var()."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_var_scalar.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_var = adata_backed.X.var()
    full_var = adata_full.X.toarray().var()

    np.testing.assert_allclose(backed_var, full_var, rtol=1e-5)


def test_max_with_deletions(tmp_dir):
    """max(axis=0) with deletions matches materialized subset max()."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_max.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_max = np.asarray(adata_backed.X.max(axis=0)).flatten()
    full_max = np.asarray(adata_full.X.max(axis=0).toarray()).flatten()

    np.testing.assert_array_equal(backed_max, full_max)


def test_min_with_deletions(tmp_dir):
    """min(axis=0) with deletions matches materialized subset min()."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_min.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_min = np.asarray(adata_backed.X.min(axis=0)).flatten()
    full_min = np.asarray(adata_full.X.min(axis=0).toarray()).flatten()

    np.testing.assert_array_equal(backed_min, full_min)


def test_mean_with_deletions(tmp_dir):
    """mean(axis=0) with deletions matches materialized subset mean()."""
    import pyscx

    path, delete_mask = _make_deleted_file(tmp_dir, "del_mean.scx")

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_mean = np.asarray(adata_backed.X.mean(axis=0)).flatten()
    full_mean = np.asarray(adata_full.X.mean(axis=0)).flatten()

    np.testing.assert_allclose(backed_mean, full_mean, rtol=1e-6)


def test_variance_preserves_nan_on_every_view(tmp_dir):
    """A NaN in X must stay NaN through `var()`, projected or not.

    Rust's `f64::max` *ignores* NaN and returns the other operand, so a
    `.max(0.0)` clamp silently reports a NaN variance as a real `0.0`. NaN in
    `X` is supported input — the dense h5ad streamer preserves it deliberately —
    so that is data loss, not tidying.

    This is a regression lock with a history: the clamp was introduced to stop
    an unrelated negative variance, using `.max(0.0)`. One review round fixed
    the unprojected scalar arm and left the two projected arms in the *same
    function* still eating NaN, which the next round caught. There was no test
    either time, which is why it shipped twice. Hence: every view, one test.
    """
    import anndata
    import pandas as pd
    import pyscx

    dense = np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32)
    adata = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    path = str(tmp_dir / "nan_var.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)

    # Unprojected: scalar, per-row and per-column.
    assert np.isnan(float(np.asarray(backed.X.var()))), "scalar var() ate the NaN"
    assert np.isnan(np.asarray(backed.X.var(axis=1)).ravel()[0]), "var(axis=1) ate the NaN"
    assert np.isnan(np.asarray(backed.X.var(axis=0)).ravel()[0]), "var(axis=0) ate the NaN"

    # Through a column projection — the arms that were missed the first time.
    view = backed[:, ["g0", "g1"]]
    assert np.isnan(float(np.asarray(view.X.var()))), "projected scalar var() ate the NaN"
    assert np.isnan(
        np.asarray(view.X.var(axis=1)).ravel()[0]
    ), "projected var(axis=1) ate the NaN"

    # The NaN-free row/column must still produce a real number on every view,
    # so the assertions above cannot pass by making everything NaN.
    assert np.isfinite(np.asarray(backed.X.var(axis=1)).ravel()[1])
    assert np.isfinite(np.asarray(view.X.var(axis=1)).ravel()[1])
