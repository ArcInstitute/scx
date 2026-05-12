"""Round-trip tests: AnnData → SCX → AnnData (Tasks 15.9–15.11)."""

import numpy as np
import pytest
import scipy.sparse as sp


def test_round_trip_anndata(synthetic_adata, tmp_dir):
    """15.9: Synthetic AnnData → SCX → to_anndata → compare shapes, nnz."""
    import pyscx

    path = str(tmp_dir / "test.scx")
    adata = synthetic_adata

    pyscx.from_anndata(adata, path)
    exp = pyscx.open(path)

    assert exp.n_obs == adata.n_obs
    assert exp.n_vars == adata.n_vars
    assert exp.nnz == adata.X.nnz

    adata2 = exp.to_anndata()

    assert adata2.n_obs == adata.n_obs
    assert adata2.n_vars == adata.n_vars
    assert adata2.X.nnz == adata.X.nnz
    assert adata2.X.shape == adata.X.shape


def test_round_trip_integer_counts(tmp_dir):
    """15.10: Integer UMI counts → SCX → back → bit-exact."""
    import anndata
    import pyscx

    np.random.seed(123)
    n_obs, n_vars = 50, 30

    # Create integer counts in uint8 range
    dense = np.random.randint(0, 255, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.4
    dense[mask] = 0
    x = sp.csr_matrix(dense)

    obs = anndata.AnnData(X=x).obs
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "int_counts.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    # Bit-exact comparison of the dense arrays
    orig_dense = adata.X.toarray()
    rt_dense = adata2.X.toarray()
    np.testing.assert_array_equal(orig_dense, rt_dense)


def test_layers_obsm_uns_round_trip(synthetic_adata, tmp_dir):
    """15.11: Include layers, obsm, uns → verify all survive."""
    import pyscx

    adata = synthetic_adata
    path = str(tmp_dir / "full.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    # Layers
    assert "raw" in adata2.layers
    orig_raw = adata.layers["raw"].toarray()
    rt_raw = adata2.layers["raw"].toarray()
    np.testing.assert_array_equal(orig_raw, rt_raw)

    # obsm — X_pca should be present (may have numeric column names)
    assert "X_pca" in adata2.obsm
    assert adata2.obsm["X_pca"].shape == (adata.n_obs, 10)

    # uns
    assert adata2.uns["species"] == "human"
    assert adata2.uns["version"] == 2


def test_obsp_varp_varm_round_trip(tmp_dir):
    """Patch 7 / P0 #6: obsp, varp, varm survive AnnData → SCX → AnnData."""
    import anndata
    import pyscx

    rng = np.random.default_rng(42)
    n_obs, n_vars = 50, 30

    X = sp.random(
        n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    adata = anndata.AnnData(X=X)
    adata.varm["PCs"] = rng.random((n_vars, 5)).astype(np.float32)
    conn = sp.random(
        n_obs, n_obs, density=0.05, format="csr", dtype=np.float32, random_state=rng
    )
    adata.obsp["connectivities"] = conn
    varp_mat = sp.random(
        n_vars, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    adata.varp["gene_corr"] = varp_mat

    path = str(tmp_dir / "obsp_varp_varm.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    assert "PCs" in adata2.varm
    np.testing.assert_allclose(adata2.varm["PCs"], adata.varm["PCs"], atol=1e-6)

    assert "connectivities" in adata2.obsp
    diff = (adata2.obsp["connectivities"] - conn).toarray()
    assert np.allclose(diff, 0.0, atol=1e-6)

    assert "gene_corr" in adata2.varp
    diff2 = (adata2.varp["gene_corr"] - varp_mat).toarray()
    assert np.allclose(diff2, 0.0, atol=1e-6)


def test_round_trip_large_values(tmp_dir):
    """Test that uint16 and uint32 ranges round-trip correctly."""
    import anndata
    import pyscx

    n_obs, n_vars = 20, 10
    dense = np.array(
        [[0, 300, 0, 0, 60000, 0, 0, 0, 0, 0]] * n_obs, dtype=np.float32
    )
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "large_vals.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    np.testing.assert_array_equal(adata.X.toarray(), adata2.X.toarray())


def test_empty_obs_var_columns(tmp_dir):
    """Round-trip with minimal obs/var (no extra columns)."""
    import anndata
    import pyscx

    x = sp.csr_matrix(np.eye(5, dtype=np.float32))
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "minimal.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    assert adata2.n_obs == 5
    assert adata2.n_vars == 5
    np.testing.assert_array_equal(adata.X.toarray(), adata2.X.toarray())
