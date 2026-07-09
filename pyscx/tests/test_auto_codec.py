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


def _read_x(path):
    import pyscx

    return pyscx.open(path).to_anndata().X.toarray()


def test_auto_v2_roundtrip_and_no_size_regression(tmp_dir):
    """`codec='auto_v2'` round-trips and never exceeds the plain-`auto` size
    (Phase C: per-shard it keeps the smaller of the heuristic vs ShufDeltaZstd,
    biased by `decode_target`)."""
    import os

    import anndata
    import pyscx

    # Larger counts (median > 8) → Zstd heuristic, which ShufDeltaZstd beats;
    # enough rows for the size delta to be meaningful.
    rng = np.random.default_rng(0)
    X = sp.random(3000, 1500, density=0.05, format="csr", random_state=0)
    X.data = np.round(X.data * 60 + 1).astype(np.float32)
    adata = anndata.AnnData(X=X)

    auto = str(tmp_dir / "auto.scx")
    av2 = str(tmp_dir / "av2.scx")
    pyscx.from_anndata(adata, auto, codec="auto", row_group_rows=256)
    pyscx.from_anndata(adata, av2, codec="auto_v2", decode_target="storage", row_group_rows=256)

    # Data round-trips identically.
    np.testing.assert_array_equal(_read_x(auto), X.toarray())
    np.testing.assert_array_equal(_read_x(av2), X.toarray())

    # No size regression vs auto (Storage picks the smaller-or-equal codec).
    assert os.path.getsize(av2) <= os.path.getsize(auto)


def test_auto_v2_cpu_matches_auto_size(tmp_dir):
    """`decode_target='cpu'` is conservative — the shard data matches plain
    `auto` (Rust proves byte-identity); the file differs only by the small
    codec-selection provenance stamp auto_v2 records."""
    import os

    import pyscx

    a = str(tmp_dir / "a.scx")
    c = str(tmp_dir / "c.scx")
    pyscx.from_anndata(large_value_adata_big(), a, codec="auto", row_group_rows=256)
    pyscx.from_anndata(large_value_adata_big(), c, codec="auto_v2", decode_target="cpu",
                       row_group_rows=256)
    np.testing.assert_array_equal(_read_x(a), _read_x(c))
    # Cpu never adopts ShufDeltaZstd, so the only file-size delta is the
    # provenance stamp (< 1 KB), never a codec change.
    assert abs(os.path.getsize(a) - os.path.getsize(c)) < 1024


def large_value_adata_big():
    import anndata

    X = sp.random(1000, 800, density=0.06, format="csr", random_state=7)
    X.data = np.round(X.data * 40 + 1).astype(np.float32)
    return anndata.AnnData(X=X)


def test_decode_target_requires_auto_v2(synthetic_adata, tmp_dir):
    """`decode_target` is only valid with `codec='auto_v2'`."""
    import pyscx

    with pytest.raises((ValueError, RuntimeError), match="auto_v2"):
        pyscx.from_anndata(synthetic_adata, str(tmp_dir / "x.scx"),
                           codec="auto", decode_target="gpu", row_group_rows=256)


def test_auto_v2_requires_framing(synthetic_adata, tmp_dir):
    """`codec='auto_v2'` requires row-group framing (row_group_rows > 0)."""
    import pyscx

    with pytest.raises((ValueError, RuntimeError), match="row_group_rows"):
        pyscx.from_anndata(synthetic_adata, str(tmp_dir / "x.scx"),
                           codec="auto_v2", row_group_rows=0)
