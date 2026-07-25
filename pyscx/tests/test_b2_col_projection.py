"""Test B2 fix: streaming column aggregation on lazy datasets with column projection."""
import numpy as np
import scipy.sparse as sp
import tempfile, shutil, os, pytest
import pyscx

def _make_scx(tmpdir, n_obs=100, n_vars=50):
    X = sp.random(n_obs, n_vars, density=0.3, random_state=42, format='csr', dtype=np.float32)
    X.data = np.ceil(X.data * 100).astype(np.float32)
    X.eliminate_zeros()
    path = os.path.join(tmpdir, "test.scx")
    import anndata, pandas as pd
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    pyscx.from_anndata(adata, path)
    return path

@pytest.fixture
def scx_path():
    tmpdir = tempfile.mkdtemp()
    path = _make_scx(tmpdir)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)

# The fixture's least-detected gene appears in 21 of 100 cells and its sparsest
# cell carries 7 genes, so the original `min_cells=3` / `min_genes=1` kept
# everything. That was still exercising an identity projection until
# `filter_genes` / `filter_cells` learned to skip an all-kept mask outright — at
# which point this whole file, whose subject *is* the column projection, stopped
# having one. Thresholds now cut roughly half, and the helpers assert it.
MIN_CELLS = 30
MIN_GENES = 15


def _filter_genes(adata, min_cells=MIN_CELLS):
    before = adata.n_vars
    pyscx.accel.filter_genes(adata, min_cells=min_cells)
    assert 0 < adata.n_vars < before, (
        f"min_cells={min_cells} kept {adata.n_vars}/{before} genes — this file "
        "tests the column projection, so the filter has to create one"
    )


def _filter_cells(adata):
    before = adata.n_obs
    pyscx.accel.filter_cells(adata, min_genes=MIN_GENES)
    assert 0 < adata.n_obs < before, (
        f"min_genes={MIN_GENES} kept {adata.n_obs}/{before} cells — the "
        "deletion-vector premise needs an actual cut"
    )


def test_sum_axis0_with_col_projection(scx_path):
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    _filter_genes(adata)
    n_proj = adata.X.shape[1]
    result = adata.X.sum(axis=0)
    assert np.asarray(result).shape == (1, n_proj)
    mat = adata.X.to_memory()
    np.testing.assert_allclose(np.asarray(result), np.asarray(mat.sum(axis=0)), rtol=1e-5)

def test_mean_axis0_with_col_projection(scx_path):
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    _filter_genes(adata)
    n_proj = adata.X.shape[1]
    result = adata.X.mean(axis=0)
    assert np.asarray(result).shape == (1, n_proj)
    mat = adata.X.to_memory()
    np.testing.assert_allclose(np.asarray(result), np.asarray(mat.mean(axis=0)), rtol=1e-5)

def test_var_axis0_with_col_projection(scx_path):
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    _filter_genes(adata)
    n_proj = adata.X.shape[1]
    result = adata.X.var(axis=0)
    assert np.asarray(result).shape == (1, n_proj)
    mat = adata.X.to_memory()
    expected = np.asarray(mat.toarray().var(axis=0)).reshape(1, -1)
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-4)

def test_both_deletion_and_col_projection(scx_path):
    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    _filter_cells(adata)
    # The cell cut already removed 45 % of the rows, so per-gene detection
    # counts here run 14–24 rather than 21–40 — MIN_CELLS would take the width
    # to zero, and anything below 15 would keep all 50.
    _filter_genes(adata, min_cells=20)
    n_obs, n_proj = adata.X.shape
    for method in ["sum", "mean", "var"]:
        r = getattr(adata.X, method)(axis=0)
        assert np.asarray(r).shape == (1, n_proj), f"{method} shape mismatch"
    mat = adata.X.to_memory()
    np.testing.assert_allclose(
        np.asarray(adata.X.sum(axis=0)), np.asarray(mat.sum(axis=0)), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(adata.X.mean(axis=0)), np.asarray(mat.mean(axis=0)), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(adata.X.var(axis=0)),
        np.asarray(mat.toarray().var(axis=0)).reshape(1, -1), rtol=1e-4)
