"""Go/No-Go gate validation tests (Task 17.9).

Three criteria:
1. h5ad → scx → h5ad round-trip is bit-exact for integer counts
2. SCX file < 60% the size of h5ad for PBMC 3K
3. scx.open().to_anndata() → full scanpy pipeline (QC → PCA → Leiden → DE)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
try:
    from benchmarks.comprehensive.bench_env import DATA_DIR
except (RuntimeError, ModuleNotFoundError):
    # SCX_WORK_DIR not set or python-dotenv not installed — skip this module
    pytest.skip(
        "bench_env unavailable (SCX_WORK_DIR not set or python-dotenv missing)",
        allow_module_level=True,
    )

PBMC_H5AD = DATA_DIR / "pbmc3k.h5ad"


# =========================================================================
# Gate 1: Bit-exact round-trip for integer counts
# =========================================================================


def test_bit_exact_roundtrip_uint8(tmp_dir):
    """Verify integer counts survive h5ad → scx → h5ad round-trip exactly."""
    import anndata
    import pyscx

    np.random.seed(123)
    n_obs, n_vars = 500, 200

    # Create sparse integer matrix with uint8-range values
    dense = np.random.randint(0, 255, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    X = sp.csr_matrix(dense)

    import pandas as pd
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    # Write h5ad → SCX → read back
    h5ad_path = str(tmp_dir / "gate1.h5ad")
    scx_path = str(tmp_dir / "gate1.scx")
    adata.write_h5ad(h5ad_path)

    pyscx.from_anndata(adata, scx_path)
    adata2 = pyscx.open(scx_path).to_anndata()

    # Bit-exact comparison
    X_orig = sp.csr_matrix(adata.X)
    X_rt = sp.csr_matrix(adata2.X)
    assert X_orig.shape == X_rt.shape
    assert X_orig.nnz == X_rt.nnz
    np.testing.assert_array_equal(X_orig.indptr, X_rt.indptr)
    np.testing.assert_array_equal(X_orig.indices, X_rt.indices)
    # Compare as integers (original was integer-valued float32)
    np.testing.assert_array_equal(
        X_orig.data.astype(np.int32), X_rt.data.astype(np.int32)
    )


def test_bit_exact_roundtrip_uint16(tmp_dir):
    """Verify uint16-range integer counts survive round-trip exactly."""
    import anndata
    import pyscx

    np.random.seed(456)
    n_obs, n_vars = 200, 100

    dense = np.random.randint(0, 65535, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    X = sp.csr_matrix(dense)

    import pandas as pd
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    scx_path = str(tmp_dir / "gate1_u16.scx")
    pyscx.from_anndata(adata, scx_path)
    adata2 = pyscx.open(scx_path).to_anndata()

    X_orig = sp.csr_matrix(adata.X)
    X_rt = sp.csr_matrix(adata2.X)
    np.testing.assert_array_equal(
        X_orig.data.astype(np.int32), X_rt.data.astype(np.int32)
    )


# =========================================================================
# Gate 2: Compression ratio < 60% on PBMC 3K
# =========================================================================


@pytest.mark.skipif(
    not PBMC_H5AD.exists(),
    reason=f"PBMC 3K dataset not available at {PBMC_H5AD}",
)
def test_compression_ratio_pbmc3k(tmp_dir):
    """Verify SCX file < 60% the size of h5ad for PBMC 3K."""
    import anndata
    import pyscx

    adata = anndata.read_h5ad(str(PBMC_H5AD))
    scx_path = str(tmp_dir / "pbmc3k.scx")
    pyscx.from_anndata(adata, scx_path)

    h5ad_size = PBMC_H5AD.stat().st_size
    scx_size = os.path.getsize(scx_path)
    ratio = scx_size / h5ad_size

    print(f"\nPBMC 3K compression: h5ad={h5ad_size}, scx={scx_size}, ratio={ratio:.3f}")
    assert ratio < 0.60, f"Compression ratio {ratio:.3f} >= 0.60 (target: < 0.60)"


# =========================================================================
# Gate 3: Full scanpy pipeline (QC → PCA → Leiden → DE)
# =========================================================================


def test_scanpy_full_pipeline(synthetic_adata, tmp_dir):
    """Verify full scanpy pipeline: QC → PCA → Leiden → DE."""
    import pyscx
    import scanpy as sc

    adata = synthetic_adata
    scx_path = str(tmp_dir / "pipeline.scx")
    pyscx.from_anndata(adata, scx_path)

    # Load from SCX
    adata2 = pyscx.open(scx_path).to_anndata()

    # Full preprocessing pipeline
    sc.pp.filter_cells(adata2, min_genes=1)
    sc.pp.filter_genes(adata2, min_cells=1)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)
    sc.pp.pca(adata2, n_comps=min(10, adata2.n_vars - 1))
    sc.pp.neighbors(adata2, n_neighbors=10)
    sc.tl.leiden(adata2, flavor="igraph", n_iterations=2, directed=False)

    # Verify clustering worked
    assert "leiden" in adata2.obs.columns
    assert adata2.obs["leiden"].nunique() >= 1

    # Differential expression (DE)
    sc.tl.rank_genes_groups(adata2, groupby="leiden", method="wilcoxon")

    # Verify DE results exist and have p-values
    assert adata2.uns["rank_genes_groups"] is not None
    result = adata2.uns["rank_genes_groups"]
    assert "names" in result
    assert "pvals" in result

    # Check that p-values are valid (between 0 and 1)
    pvals = result["pvals"]
    # pvals is a structured array; check the first group
    first_group_key = list(pvals.dtype.names)[0]
    pval_arr = pvals[first_group_key]
    assert len(pval_arr) > 0
    assert np.all(pval_arr >= 0)
    assert np.all(pval_arr <= 1)


@pytest.mark.skipif(
    not PBMC_H5AD.exists(),
    reason=f"PBMC 3K dataset not available at {PBMC_H5AD}",
)
def test_scanpy_pipeline_pbmc3k(tmp_dir):
    """End-to-end: real PBMC 3K → SCX → to_anndata → scanpy QC → PCA → Leiden → DE."""
    import anndata
    import pyscx
    import scanpy as sc

    adata = anndata.read_h5ad(str(PBMC_H5AD))
    scx_path = str(tmp_dir / "pbmc3k_pipeline.scx")
    pyscx.from_anndata(adata, scx_path)

    adata2 = pyscx.open(scx_path).to_anndata()

    sc.pp.filter_cells(adata2, min_genes=200)
    sc.pp.filter_genes(adata2, min_cells=3)
    sc.pp.normalize_total(adata2, target_sum=1e4)
    sc.pp.log1p(adata2)
    sc.pp.highly_variable_genes(adata2, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata2 = adata2[:, adata2.var.highly_variable]
    sc.pp.scale(adata2, max_value=10)
    sc.tl.pca(adata2, svd_solver="arpack")
    sc.pp.neighbors(adata2, n_neighbors=10, n_pcs=40)
    sc.tl.leiden(adata2, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.rank_genes_groups(adata2, groupby="leiden", method="wilcoxon")

    assert "leiden" in adata2.obs.columns
    assert adata2.obs["leiden"].nunique() >= 2
    assert adata2.uns["rank_genes_groups"] is not None
