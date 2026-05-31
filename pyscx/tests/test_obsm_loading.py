"""Tests for selective + lazy + backed ``obsm`` loading in ``to_anndata``.

Covers the three obsm loading phases (selective eager / lazy / backed
row-gather):

- Phase 1: ``to_anndata(obsm=[...])`` selective eager loading.
- Phase 2: ``to_anndata(obsm=[...], eager=False)`` lazy ``ScxLazyObsmMapping``.
- Phase 3: ``to_anndata(backed=True, obsm=[...])`` backed dense row-gather
  (``ScxBackedObsmDataset``).

All new behaviour is opt-in: ``obsm=None`` (default) stays byte-identical.
"""

import numpy as np
import pytest


def _adata_multi_obsm(n_obs=60, seed=7):
    """AnnData with two obsm keys of different widths/dtypes."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    n_vars = 12
    x = sp.random(
        n_obs, n_vars, density=0.3, format="csr", dtype=np.float32, random_state=seed
    )
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                rng.choice(["T cell", "B cell", "NK cell"], size=n_obs)
            ),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    adata.obsm["X_pca"] = rng.standard_normal((n_obs, 10)).astype(np.float32)
    adata.obsm["X_umap"] = rng.standard_normal((n_obs, 2)).astype(np.float32)
    return adata


@pytest.fixture
def multi_obsm_scx(tmp_dir):
    import pyscx

    adata = _adata_multi_obsm()
    path = str(tmp_dir / "multi_obsm.scx")
    pyscx.from_anndata(adata, path)
    return path, adata


# ---------------------------------------------------------------------------
# Phase 1 — selective eager obsm
# ---------------------------------------------------------------------------


def test_obsm_none_loads_all(multi_obsm_scx):
    """Default obsm=None loads every key (back-compat)."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata()
    assert set(out.obsm.keys()) == {"X_pca", "X_umap"}


def test_obsm_selective_loads_only_listed(multi_obsm_scx):
    """obsm=["X_pca"] loads only that key; values match the full load."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(obsm=["X_pca"])
    assert set(out.obsm.keys()) == {"X_pca"}
    np.testing.assert_array_equal(np.asarray(out.obsm["X_pca"]), adata.obsm["X_pca"])


def test_obsm_selective_empty_list(multi_obsm_scx):
    """obsm=[] loads no embeddings."""
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata(obsm=[])
    assert len(out.obsm) == 0


def test_obsm_unknown_key_raises(multi_obsm_scx):
    """A missing obsm key raises KeyError."""
    import pyscx

    path, _ = multi_obsm_scx
    with pytest.raises(KeyError):
        pyscx.open(path).to_anndata(obsm=["X_nope"])


def test_obsm_selective_eager_backed(multi_obsm_scx):
    """backed=True, eager=True, obsm=[...] → selective eager dict."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, eager=True, obsm=["X_umap"])
    assert set(out.obsm.keys()) == {"X_umap"}
    np.testing.assert_array_equal(np.asarray(out.obsm["X_umap"]), adata.obsm["X_umap"])


# ---------------------------------------------------------------------------
# Phase 2 — lazy obsm mapping (non-backed)
# ---------------------------------------------------------------------------


def test_obsm_default_is_eager_dict(multi_obsm_scx):
    """Default (obsm=None) keeps obsm eager — NOT a lazy bridge.

    Guards the default-behaviour invariant: obsm only goes lazy when the
    caller opts into selection.
    """
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata()
    assert type(out._obsm).__name__ != "ScxLazyObsmMapping"


def test_obsm_lazy_mapping_when_selected(multi_obsm_scx):
    """obsm=[...] + eager=False (default) → ScxLazyObsmMapping bridge."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(obsm=["X_pca"])
    assert type(out._obsm).__name__ == "ScxLazyObsmMapping"
    # Existence probes serve from the catalog key set.
    assert "X_pca" in out._obsm
    assert len(out._obsm) == 1
    # First access materialises and matches the source.
    np.testing.assert_array_equal(np.asarray(out.obsm["X_pca"]), adata.obsm["X_pca"])


def test_obsm_lazy_mapping_caches(multi_obsm_scx):
    """Repeated access of the same lazy key returns the cached object."""
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata(obsm=["X_pca"])
    a = out._obsm["X_pca"]
    b = out._obsm["X_pca"]
    assert a is b


# ---------------------------------------------------------------------------
# Phase 3 — backed dense row-gather
# ---------------------------------------------------------------------------


def test_obsm_backed_is_row_gather_dataset(multi_obsm_scx):
    """backed=True, obsm=[...] → ScxBackedObsmDataset per key, reachable
    through the public `adata.obsm[key]` access path (AnnData's obsm
    coercion accepts it via the CSRDataset registration)."""
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    assert type(out._obsm).__name__ == "ScxLazyObsmMapping"
    # Public access (what the dataloader uses) must return the backed dataset.
    ds = out.obsm["X_pca"]
    assert type(ds).__name__ == "ScxBackedObsmDataset"
    assert ds.shape == (60, 10)


def test_obsm_backed_row_gather_matches_full(multi_obsm_scx):
    """m[idx_arr] gathers the requested rows, equal to the full array."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    m = out.obsm["X_pca"]  # public access path

    idx = np.array([5, 0, 59, 23, 1], dtype=np.int64)
    got = m[idx]
    assert got.shape == (5, 10)
    np.testing.assert_array_equal(got, adata.obsm["X_pca"][idx])

    # Single int index → 1-D row vector.
    row = m[7]
    assert row.shape == (10,)
    np.testing.assert_array_equal(row, adata.obsm["X_pca"][7])

    # Slice.
    sl = m[10:15]
    np.testing.assert_array_equal(sl, adata.obsm["X_pca"][10:15])

    # Full materialise.
    np.testing.assert_array_equal(np.asarray(m), adata.obsm["X_pca"])


def test_obsm_backed_dtype_preserved(multi_obsm_scx):
    """Row-gather preserves the on-disk float32 dtype."""
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    m = out.obsm["X_pca"]
    assert m.dtype == np.dtype("float32")
    assert m[np.array([0, 1])].dtype == np.dtype("float32")


def test_obsm_backed_obs_filter_falls_back_to_eager(multi_obsm_scx):
    """backed + obsm + obs_filter falls back to eager obsm (documented V1).

    The result still carries correctly-sliced obsm values; it just isn't
    a backed row-gather dataset.
    """
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(
        backed=True, obsm=["X_pca"], obs_filter="cell_type == 'T cell'"
    )
    n_t = int((adata.obs["cell_type"] == "T cell").sum())
    assert out.n_obs == n_t
    assert "X_pca" in out.obsm
    assert np.asarray(out.obsm["X_pca"]).shape == (n_t, 10)
    # Eager fallback → not a backed dataset.
    assert type(out.obsm["X_pca"]).__name__ != "ScxBackedObsmDataset"


# ---------------------------------------------------------------------------
# Deletion-vector composition
# ---------------------------------------------------------------------------


def test_obsm_backed_with_deletions(tmp_dir):
    """Row-gather composes with a deletion vector (rows align with obs)."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    import pyscx

    rng = np.random.default_rng(3)
    n_obs, n_vars = 50, 8
    x = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32,
                  random_state=3)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    adata.obsm["X_emb"] = rng.standard_normal((n_obs, 6)).astype(np.float32)

    path = str(tmp_dir / "del.scx")
    pyscx.from_anndata(adata, path)

    # Drop the first 10 cells via a filtered subset to create a deletion
    # vector, written back out.
    keep = np.arange(10, n_obs)
    sub = adata[keep].copy()
    sub_path = str(tmp_dir / "del_sub.scx")
    pyscx.from_anndata(sub, sub_path)

    out = pyscx.open(sub_path).to_anndata(backed=True, obsm=["X_emb"])
    m = out.obsm["X_emb"]
    assert m.shape == (len(keep), 6)
    np.testing.assert_array_equal(np.asarray(m), adata.obsm["X_emb"][keep])
    # Row-gather of a subset aligns with the same rows of the source.
    sub_idx = np.array([0, 5, len(keep) - 1])
    np.testing.assert_array_equal(m[sub_idx], adata.obsm["X_emb"][keep][sub_idx])


# ---------------------------------------------------------------------------
# Backed indexing edge cases (boolean masks)
# ---------------------------------------------------------------------------


def test_obsm_backed_bool_mask_full_length(multi_obsm_scx):
    """A correctly-sized 1-D boolean mask gathers the True rows."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    m = out.obsm["X_pca"]
    mask = np.zeros(60, dtype=bool)
    mask[[3, 17, 58]] = True
    np.testing.assert_array_equal(m[mask], adata.obsm["X_pca"][mask])


def test_obsm_backed_bool_mask_wrong_length_raises(multi_obsm_scx):
    """A short 1-D boolean mask raises IndexError (numpy semantics) rather
    than silently gathering only its leading rows."""
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    m = out.obsm["X_pca"]
    with pytest.raises(IndexError):
        _ = m[np.array([True, False])]


def test_obsm_backed_array_protocol_copy(multi_obsm_scx):
    """`__array__` accepts numpy 2.0's `copy` kwarg: asarray/array work
    without a warning, and `copy=False` (no-copy demand) raises ValueError
    since a backed dataset materialises a fresh array."""
    import pyscx

    path, adata = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    m = out.obsm["X_pca"]

    np.testing.assert_array_equal(np.asarray(m), adata.obsm["X_pca"])
    np.testing.assert_array_equal(np.array(m), adata.obsm["X_pca"])
    # No-copy demand cannot be satisfied by a backed dataset.
    with pytest.raises(ValueError):
        m.__array__(copy=False)


def test_obsm_backed_bool_mask_2d_raises(multi_obsm_scx):
    """A 2-D boolean mask raises IndexError rather than collapsing to its
    first nonzero coordinate."""
    import pyscx

    path, _ = multi_obsm_scx
    out = pyscx.open(path).to_anndata(backed=True, obsm=["X_pca"])
    m = out.obsm["X_pca"]
    mask2d = np.zeros((60, 10), dtype=bool)
    mask2d[0, 0] = True
    with pytest.raises(IndexError):
        _ = m[mask2d]


# ---------------------------------------------------------------------------
# Positional back-compat (obsm appended at the end of the signature)
# ---------------------------------------------------------------------------


def test_to_anndata_positional_preserve_slots_still_binds(multi_obsm_scx):
    """`obsm` is appended after `memory_budget`, so a legacy positional call
    that set `preserve_slots=True` by position must keep working.

    Positional order: backed, cache_shards, var_names, obs_filter, layers,
    preserve_slots, ...
    """
    import pyscx

    path, adata = multi_obsm_scx
    # preserve_slots=True passed positionally (6th positional arg).
    out = pyscx.open(path).to_anndata(
        False, 4, None, "cell_type == 'T cell'", None, True
    )
    n_t = int((adata.obs["cell_type"] == "T cell").sum())
    assert out.n_obs == n_t
    # preserve_slots=True keeps obsm after the filter.
    assert "X_pca" in out.obsm
