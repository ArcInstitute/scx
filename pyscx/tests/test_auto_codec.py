"""Tests for auto-codec selection in pyscx."""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def umi_adata():
    """AnnData with small UMI-like counts (median ~2) → should select Scx1."""
    import anndata

    np.random.seed(123)
    n_obs, n_vars = 50, 30
    # Geometric-like distribution: small values (1-5 mostly)
    dense = np.random.geometric(p=0.5, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    return anndata.AnnData(X=sp.csr_matrix(dense))


@pytest.fixture
def large_value_adata():
    """AnnData with large values (uniform 100-1000) → should select Zstd."""
    import anndata

    np.random.seed(456)
    n_obs, n_vars = 50, 30
    dense = np.random.randint(100, 1000, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    return anndata.AnnData(X=sp.csr_matrix(dense))


def test_auto_codec_default_roundtrip(synthetic_adata, tmp_dir):
    """Default (auto) codec round-trips data correctly."""
    import pyscx

    path = str(tmp_dir / "auto.scx")
    pyscx.from_anndata(synthetic_adata, path)

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    # Check X values match
    x1 = synthetic_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_auto_codec_umi_roundtrip(umi_adata, tmp_dir):
    """Auto codec on UMI-like data round-trips correctly."""
    import pyscx

    path = str(tmp_dir / "umi_auto.scx")
    pyscx.from_anndata(umi_adata, path)

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    x1 = umi_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_auto_codec_large_values_roundtrip(large_value_adata, tmp_dir):
    """Auto codec on large-value data round-trips correctly."""
    import pyscx

    path = str(tmp_dir / "large_auto.scx")
    pyscx.from_anndata(large_value_adata, path)

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    x1 = large_value_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_explicit_codec_none(synthetic_adata, tmp_dir):
    """Explicit codec='none' works."""
    import pyscx

    path = str(tmp_dir / "none.scx")
    pyscx.from_anndata(synthetic_adata, path, codec="none")

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    x1 = synthetic_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_explicit_codec_scx1(synthetic_adata, tmp_dir):
    """Explicit codec='scx1' works."""
    import pyscx

    path = str(tmp_dir / "scx1.scx")
    pyscx.from_anndata(synthetic_adata, path, codec="scx1")

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    x1 = synthetic_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_explicit_codec_zstd(synthetic_adata, tmp_dir):
    """Explicit codec='zstd' works."""
    import pyscx

    path = str(tmp_dir / "zstd.scx")
    pyscx.from_anndata(synthetic_adata, path, codec="zstd")

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    x1 = synthetic_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_explicit_codec_auto(synthetic_adata, tmp_dir):
    """Explicit codec='auto' is accepted and works."""
    import pyscx

    path = str(tmp_dir / "explicit_auto.scx")
    pyscx.from_anndata(synthetic_adata, path, codec="auto")

    exp = pyscx.open(path)
    adata2 = exp.to_anndata()

    x1 = synthetic_adata.X.toarray()
    x2 = adata2.X.toarray()
    np.testing.assert_array_almost_equal(x1, x2)


def test_invalid_codec_raises(synthetic_adata, tmp_dir):
    """Invalid codec name raises an error."""
    import pyscx

    path = str(tmp_dir / "bad.scx")
    with pytest.raises(RuntimeError, match="Unknown codec"):
        pyscx.from_anndata(synthetic_adata, path, codec="invalid")
