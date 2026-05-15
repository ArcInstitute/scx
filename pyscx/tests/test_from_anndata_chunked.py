"""Tests for the ``chunked=True`` path on pyscx.from_anndata().

The chunked path writes the first chunk with the native ``from_anndata`` and
appends remaining chunks with ``append_from_anndata``. ``X`` content is
byte-identical to a one-shot write; ``obsm``/``varm``/``obsp``/``varp``/
``layers`` are dropped with a UserWarning because the underlying append
primitive does not extend those aligned mappings.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp


def _x_equal(a, b):
    a = a.toarray() if sp.issparse(a) else np.asarray(a)
    b = b.toarray() if sp.issparse(b) else np.asarray(b)
    return np.array_equal(a, b)


def test_chunked_x_matches_one_shot(synthetic_adata, tmp_dir):
    """X is byte-identical between one-shot and chunked writes."""
    import pyscx

    one_shot = str(tmp_dir / "one_shot.scx")
    chunked = str(tmp_dir / "chunked.scx")

    pyscx.from_anndata(synthetic_adata, one_shot)
    with warnings.catch_warnings():
        # synthetic_adata carries obsm + layers; chunked drops them with a
        # UserWarning, which we ignore here — it's exercised separately.
        warnings.simplefilter("ignore", UserWarning)
        pyscx.from_anndata(synthetic_adata, chunked, chunked=True, n_chunks=5)

    a = pyscx.open(one_shot).to_anndata()
    b = pyscx.open(chunked).to_anndata()

    assert a.shape == b.shape
    assert _x_equal(a.X, b.X)


def test_chunked_default_n_chunks(synthetic_adata, tmp_dir):
    """``chunked=True`` with no ``n_chunks`` defaults to ~one-shard chunks."""
    import pyscx

    path = str(tmp_dir / "default.scx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pyscx.from_anndata(synthetic_adata, path, chunked=True)

    out = pyscx.open(path).to_anndata()
    assert out.n_obs == synthetic_adata.n_obs
    assert out.n_vars == synthetic_adata.n_vars
    assert _x_equal(out.X, synthetic_adata.X)


def test_chunked_n_chunks_one(synthetic_adata, tmp_dir):
    """``n_chunks=1`` reduces to a single from_anndata call."""
    import pyscx

    path = str(tmp_dir / "one.scx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pyscx.from_anndata(synthetic_adata, path, chunked=True, n_chunks=1)

    out = pyscx.open(path).to_anndata()
    assert _x_equal(out.X, synthetic_adata.X)


def test_chunked_more_chunks_than_rows_caps(synthetic_adata, tmp_dir):
    """``n_chunks > n_obs`` is capped to ``n_obs``."""
    import pyscx

    n_obs = synthetic_adata.n_obs
    path = str(tmp_dir / "many.scx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pyscx.from_anndata(
            synthetic_adata, path, chunked=True, n_chunks=n_obs + 50
        )

    out = pyscx.open(path).to_anndata()
    assert out.n_obs == n_obs
    assert _x_equal(out.X, synthetic_adata.X)


def test_chunked_invalid_n_chunks(synthetic_adata, tmp_dir):
    """Non-positive / non-int ``n_chunks`` raises ValueError."""
    import pyscx

    path = str(tmp_dir / "invalid.scx")
    for bad in (0, -3, 2.5):
        with pytest.raises(ValueError):
            pyscx.from_anndata(
                synthetic_adata, path, chunked=True, n_chunks=bad
            )


def test_chunked_warns_when_dropping_aligned_slots(synthetic_adata, tmp_dir):
    """When obsm/layers are non-empty, a UserWarning names what was dropped."""
    import pyscx

    path = str(tmp_dir / "warn.scx")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.from_anndata(synthetic_adata, path, chunked=True, n_chunks=4)

    user_warns = [
        str(w.message) for w in caught if issubclass(w.category, UserWarning)
    ]
    assert any(
        "obsm" in m and "layers" in m for m in user_warns
    ), f"expected obsm+layers in warning, got: {user_warns}"

    out = pyscx.open(path).to_anndata()
    assert out.n_obs == synthetic_adata.n_obs
    assert _x_equal(out.X, synthetic_adata.X)
    assert len(getattr(out, "obsm", {}) or {}) == 0
    assert len(getattr(out, "layers", {}) or {}) == 0


def test_chunked_from_backed_h5ad(synthetic_adata, tmp_dir):
    """Chunked write from a backed h5ad — X never fully materialized."""
    import anndata
    import pyscx

    h5ad_path = str(tmp_dir / "src.h5ad")
    synthetic_adata.write_h5ad(h5ad_path)

    backed = anndata.read_h5ad(h5ad_path, backed="r")
    try:
        out_path = str(tmp_dir / "from_backed.scx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            pyscx.from_anndata(backed, out_path, chunked=True, n_chunks=4)
    finally:
        if backed.isbacked:
            backed.file.close()

    out = pyscx.open(out_path).to_anndata()
    assert out.n_obs == synthetic_adata.n_obs
    assert _x_equal(out.X, synthetic_adata.X)


def test_chunked_false_default_preserves_obsm(synthetic_adata, tmp_dir):
    """``chunked=False`` (default) keeps obsm/layers — no warning, no drop."""
    import pyscx

    path = str(tmp_dir / "no_chunked.scx")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pyscx.from_anndata(synthetic_adata, path)

    user_warns = [w for w in caught if issubclass(w.category, UserWarning)]
    assert not any(
        "obsm" in str(w.message) or "layers" in str(w.message)
        for w in user_warns
    )

    out = pyscx.open(path).to_anndata()
    assert _x_equal(out.X, synthetic_adata.X)
    assert "X_pca" in (out.obsm or {})
    assert "raw" in (out.layers or {})
