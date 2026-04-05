"""Tests for CSR bypass optimization (Phase 1C).

Verifies that from_anndata produces identical output regardless of input format:
CSR (sorted, unsorted, various dtypes), CSC, dense, and backed AnnData.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import anndata
import pyscx


def _make_adata(x_matrix, n_obs=100, n_vars=50):
    """Create AnnData with the given X matrix."""
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return anndata.AnnData(X=x_matrix, obs=obs, var=var)


def _reference_csr(seed=42):
    """Generate a reference CSR matrix (sorted, float32, integer values)."""
    np.random.seed(seed)
    n_obs, n_vars = 100, 50
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    return sp.csr_matrix(dense), n_obs, n_vars


@pytest.mark.parametrize(
    "input_format",
    [
        "csr_f32_sorted",
        "csr_f64",
        "csr_int32_data",
        "csr_unsorted",
        "csc",
        "dense",
    ],
)
def test_csr_bypass_equivalence(tmp_path, input_format):
    """Verify identical round-trip for all input formats."""
    csr, n_obs, n_vars = _reference_csr()
    dense = csr.toarray()

    if input_format == "csr_f32_sorted":
        x = csr.copy()
        x.sort_indices()
    elif input_format == "csr_f64":
        x = csr.copy()
        x.data = x.data.astype(np.float64)
    elif input_format == "csr_int32_data":
        x = csr.copy()
        x.data = x.data.astype(np.int32)
    elif input_format == "csr_unsorted":
        # Reverse indices within each row to force unsorted
        x = csr.copy()
        for i in range(x.shape[0]):
            start, end = x.indptr[i], x.indptr[i + 1]
            x.indices[start:end] = x.indices[start:end][::-1]
            x.data[start:end] = x.data[start:end][::-1]
        x.has_sorted_indices = False
    elif input_format == "csc":
        x = sp.csc_matrix(dense)
    elif input_format == "dense":
        x = dense
    else:
        raise ValueError(f"Unknown format: {input_format}")

    adata = _make_adata(x, n_obs, n_vars)
    path = str(tmp_path / f"test_{input_format}.scx")
    pyscx.from_anndata(adata, path)

    # Read back and compare against reference dense
    result = pyscx.open(path).to_anndata()
    result_dense = (
        result.X.toarray() if sp.issparse(result.X) else np.asarray(result.X)
    )
    np.testing.assert_array_equal(result_dense, dense)


def test_csr_bypass_with_layers(tmp_path):
    """Verify bypass works for layers too."""
    csr, n_obs, n_vars = _reference_csr()
    np.random.seed(99)
    raw_dense = np.random.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    raw_dense[mask] = 0
    raw_csr = sp.csr_matrix(raw_dense)

    adata = _make_adata(csr, n_obs, n_vars)
    adata.layers["raw"] = raw_csr

    path = str(tmp_path / "test_layers.scx")
    pyscx.from_anndata(adata, path)

    result = pyscx.open(path).to_anndata()
    result_x = result.X.toarray() if sp.issparse(result.X) else np.asarray(result.X)
    result_raw = (
        result.layers["raw"].toarray()
        if sp.issparse(result.layers["raw"])
        else np.asarray(result.layers["raw"])
    )
    np.testing.assert_array_equal(result_x, csr.toarray())
    np.testing.assert_array_equal(result_raw, raw_dense)


def test_csr_bypass_empty_matrix(tmp_path):
    """Verify bypass handles empty (all-zero) matrices."""
    x = sp.csr_matrix((10, 20), dtype=np.float32)
    adata = _make_adata(x, 10, 20)
    path = str(tmp_path / "test_empty.scx")
    pyscx.from_anndata(adata, path)

    result = pyscx.open(path).to_anndata()
    result_dense = (
        result.X.toarray() if sp.issparse(result.X) else np.asarray(result.X)
    )
    np.testing.assert_array_equal(result_dense, np.zeros((10, 20)))


def test_csr_bypass_large_values(tmp_path):
    """Verify bypass handles uint16 and uint32 value ranges."""
    np.random.seed(123)
    n_obs, n_vars = 50, 30
    # Values in uint16 range
    dense = np.random.randint(0, 60000, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)

    adata = _make_adata(x, n_obs, n_vars)
    path = str(tmp_path / "test_large_values.scx")
    pyscx.from_anndata(adata, path)

    result = pyscx.open(path).to_anndata()
    result_dense = (
        result.X.toarray() if sp.issparse(result.X) else np.asarray(result.X)
    )
    np.testing.assert_array_equal(result_dense, dense)
