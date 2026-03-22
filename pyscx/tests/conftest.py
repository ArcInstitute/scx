"""Shared pytest fixtures for pyscx tests."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def synthetic_adata():
    """Create a synthetic AnnData with 100 cells × 50 genes.

    - X: sparse CSR with integer UMI counts (uint8 range)
    - obs: cell_id (str) + batch (categorical)
    - var: gene_id (str) + highly_variable (bool)
    - obsm: X_pca (100×10 float)
    - uns: {"species": "human", "version": 2}
    - layers: {"raw": same shape as X with different values}
    """
    import anndata

    np.random.seed(42)
    n_obs, n_vars = 100, 50

    # Sparse integer counts
    density = 0.3
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > density
    dense[mask] = 0
    x = sp.csr_matrix(dense)

    # obs metadata
    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_obs)],
            "batch": pd.Categorical(
                np.random.choice(["A", "B", "C"], size=n_obs)
            ),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    # var metadata
    var = pd.DataFrame(
        {
            "gene_id": [f"gene_{i}" for i in range(n_vars)],
            "highly_variable": np.random.choice([True, False], size=n_vars),
        },
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    # obsm
    obsm = {"X_pca": np.random.randn(n_obs, 10).astype(np.float32)}

    # uns
    uns = {"species": "human", "version": 2}

    # layers (raw counts — different random)
    raw_dense = np.random.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
    raw_dense[mask] = 0
    layers = {"raw": sp.csr_matrix(raw_dense)}

    adata = anndata.AnnData(
        X=x, obs=obs, var=var, obsm=obsm, uns=uns, layers=layers
    )
    return adata


@pytest.fixture
def query_adata():
    """Create a synthetic AnnData with known obs metadata for query testing.

    - 120 cells × 40 genes
    - obs: cell_type (T cell / B cell / NK cell), tissue (lung / blood)
    - X: sparse integer counts (uint8 range)
    """
    import anndata

    np.random.seed(99)
    n_obs, n_vars = 120, 40

    # Sparse integer counts
    dense = np.random.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)

    # obs with cell_type and tissue columns
    cell_types = (["T cell"] * 40) + (["B cell"] * 40) + (["NK cell"] * 40)
    tissues = (["lung"] * 20 + ["blood"] * 20) * 3
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(cell_types),
            "tissue": pd.Categorical(tissues),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    # var metadata
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    return anndata.AnnData(X=x, obs=obs, var=var)


@pytest.fixture
def scx_from_adata(tmp_dir):
    """Helper fixture: write an AnnData to a temp SCX file, return (path, adata)."""
    import pyscx

    def _write(adata, name="test.scx"):
        path = str(tmp_dir / name)
        pyscx.from_anndata(adata, path)
        return path

    return _write

