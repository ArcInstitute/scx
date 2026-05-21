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


def test_obsp_round_trip_with_deletions(tmp_dir):
    """obsp is subset to kept rows/cols when a deletion vector is active.

    Regression for the bug where to_anndata() loaded obsp at the original
    n_obs × n_obs shape after mark_deleted(), causing AnnData's shape
    validator to reject the result.
    """
    import anndata
    import pyscx

    rng = np.random.default_rng(7)
    n_obs, n_vars = 60, 25

    X = sp.random(
        n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    conn = sp.random(
        n_obs, n_obs, density=0.05, format="csr", dtype=np.float32, random_state=rng
    )
    adata = anndata.AnnData(X=X)
    adata.obsp["connectivities"] = conn

    path = str(tmp_dir / "obsp_del.scx")
    pyscx.from_anndata(adata, path)

    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[[3, 7, 11, 42]] = True
    pyscx.open(path).mark_deleted(delete_mask)

    kept_mask = ~delete_mask
    n_kept = int(kept_mask.sum())
    expected = conn.tocsr()[kept_mask][:, kept_mask]

    # Non-backed path
    adata2 = pyscx.open(path).to_anndata()
    assert adata2.n_obs == n_kept
    assert adata2.obsp["connectivities"].shape == (n_kept, n_kept)
    diff = (adata2.obsp["connectivities"] - expected).toarray()
    assert np.allclose(diff, 0.0, atol=1e-6)

    # Backed path
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.n_obs == n_kept
    assert adata_backed.obsp["connectivities"].shape == (n_kept, n_kept)
    diff_b = (adata_backed.obsp["connectivities"] - expected).toarray()
    assert np.allclose(diff_b, 0.0, atol=1e-6)


def test_obsp_round_trip_with_deletions_and_obs_filter(tmp_dir):
    """Backed mode: obsp respects deletion vector + obs_filter composition."""
    import anndata
    import pandas as pd
    import pyscx

    rng = np.random.default_rng(13)
    n_obs, n_vars = 40, 15

    X = sp.random(
        n_obs, n_vars, density=0.15, format="csr", dtype=np.float32, random_state=rng
    )
    conn = sp.random(
        n_obs, n_obs, density=0.08, format="csr", dtype=np.float32, random_state=rng
    )
    obs = pd.DataFrame({"score": rng.random(n_obs).astype(np.float32)})
    adata = anndata.AnnData(X=X, obs=obs)
    adata.obsp["connectivities"] = conn

    path = str(tmp_dir / "obsp_del_filter.scx")
    pyscx.from_anndata(adata, path)

    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[[0, 5, 9]] = True
    pyscx.open(path).mark_deleted(delete_mask)

    # The reader exposes obs.score post-deletion; further filter via obs_filter.
    adata_filtered = pyscx.open(path).to_anndata(
        backed=True, obs_filter="score > 0.5"
    )
    n_final = adata_filtered.n_obs
    assert adata_filtered.obsp["connectivities"].shape == (n_final, n_final)


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


# ---------------------------------------------------------------------------
# `pyscx.from_h5ad → pyscx.to_h5ad` must preserve `var_names` / `obs_names`
# rather than silently swapping the pandas index with the first non-index
# column. The bug was in `scx-convert/src/h5ad_write.rs::write_dataframe_group_at`,
# which used `schema.field(0).name()` as the `_index` attribute instead of
# consulting the pandas `index_columns` schema metadata.
# ---------------------------------------------------------------------------


def test_pyscx_h5ad_roundtrip_preserves_unnamed_var_index(tmp_dir):
    """The pbmc10k-shaped case: `var.index.name = None`, multiple var
    columns. Round-trip must keep var_names as the gene symbols and must
    NOT introduce a `__index_level_0__` column."""
    import anndata as ad
    import pandas as pd
    import pyscx

    n_obs, n_vars = 6, 4
    x = sp.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    var = pd.DataFrame(
        {
            "gene_ids": [f"ENSG{i:07d}" for i in range(n_vars)],
            "feature_types": pd.Categorical(["Gene Expression"] * n_vars),
        },
        index=pd.Index([f"GENE_{i}" for i in range(n_vars)]),  # unnamed
    )
    obs = pd.DataFrame(index=pd.Index([f"CELL_{i}" for i in range(n_obs)]))
    src = ad.AnnData(X=x, obs=obs, var=var)

    src_path = tmp_dir / "src.h5ad"
    scx_path = tmp_dir / "src.scx"
    rt_path = tmp_dir / "rt.h5ad"
    src.write_h5ad(src_path)

    pyscx.from_h5ad(str(src_path), str(scx_path))
    pyscx.to_h5ad(str(scx_path), str(rt_path))

    rt = ad.read_h5ad(rt_path)
    assert list(rt.var_names) == list(src.var_names), (
        f"var_names corrupted: got {list(rt.var_names)[:3]}, expected "
        f"{list(src.var_names)[:3]}"
    )
    assert list(rt.var.columns) == ["gene_ids", "feature_types"], (
        f"var.columns drifted: {rt.var.columns.tolist()}"
    )
    assert rt.var.index.name is None
    assert "__index_level_0__" not in rt.var.columns


def test_pyscx_h5ad_roundtrip_preserves_named_var_index(tmp_dir):
    """Named pandas index (`var.index.name = 'gene_symbols'`): the
    round-trip must keep the index NAME as well as the values."""
    import anndata as ad
    import pandas as pd
    import pyscx

    n_obs, n_vars = 6, 4
    x = sp.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    var = pd.DataFrame(
        {"gene_ids": [f"ENSG{i:07d}" for i in range(n_vars)]},
        index=pd.Index(
            [f"SYMBOL_{i}" for i in range(n_vars)], name="gene_symbols"
        ),
    )
    obs = pd.DataFrame(index=pd.Index([f"CELL_{i}" for i in range(n_obs)]))
    src = ad.AnnData(X=x, obs=obs, var=var)

    src_path = tmp_dir / "src_named.h5ad"
    scx_path = tmp_dir / "src_named.scx"
    rt_path = tmp_dir / "rt_named.h5ad"
    src.write_h5ad(src_path)

    pyscx.from_h5ad(str(src_path), str(scx_path))
    pyscx.to_h5ad(str(scx_path), str(rt_path))

    rt = ad.read_h5ad(rt_path)
    assert list(rt.var_names) == list(src.var_names)
    assert rt.var.index.name == "gene_symbols", (
        f"named index lost: got {rt.var.index.name!r}"
    )
    assert list(rt.var.columns) == ["gene_ids"]
    assert "__index_level_0__" not in rt.var.columns


def test_pyscx_h5ad_roundtrip_preserves_index_only_var(tmp_dir):
    """Codex P1 (B2 follow-up): an AnnData with non-trivial var_names
    but zero var columns must survive a full
    `from_h5ad → to_h5ad → from_h5ad → to_anndata` loop without
    `var_names`/`obs_names` getting replaced by empty strings.

    Triggers the SCX-internal re-read path that hits
    `read_dataframe_group`'s empty-`column-order` fallback — the
    branch that pre-fix synthesised `vec![""; n_rows]` for the index.
    """
    import anndata as ad
    import pandas as pd
    import pyscx

    n_obs, n_vars = 6, 4
    x = sp.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    src = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=pd.Index([f"CELL_{i}" for i in range(n_obs)])),
        var=pd.DataFrame(index=pd.Index([f"GENE_{i}" for i in range(n_vars)])),
    )
    assert src.var.shape[1] == 0 and src.obs.shape[1] == 0

    src_h5ad = tmp_dir / "idx_only.h5ad"
    src_scx = tmp_dir / "idx_only.scx"
    rt_h5ad = tmp_dir / "idx_only_rt.h5ad"
    rt_scx = tmp_dir / "idx_only_rt.scx"
    src.write_h5ad(src_h5ad)

    # First leg: h5ad → scx → h5ad (exercises the writer fix from B2).
    pyscx.from_h5ad(str(src_h5ad), str(src_scx))
    pyscx.to_h5ad(str(src_scx), str(rt_h5ad))

    # Second leg: h5ad → scx via the CLI-style path that goes through
    # `read_dataframe_group`. This is the branch Codex flagged.
    pyscx.from_h5ad(str(rt_h5ad), str(rt_scx))
    rt = pyscx.open(str(rt_scx)).to_anndata()

    assert list(rt.var_names) == list(src.var_names), (
        f"var_names lost on round-trip: got {list(rt.var_names)}, "
        f"expected {list(src.var_names)}"
    )
    assert list(rt.obs_names) == list(src.obs_names), (
        f"obs_names lost on round-trip: got {list(rt.obs_names)}, "
        f"expected {list(src.obs_names)}"
    )


# ---------------------------------------------------------------------------
# Lazy-by-default obsp / varp / varm / layers
# (2026-05-20 BENCHMARK-REGRESSIONS, Cluster 1)
# ---------------------------------------------------------------------------


def _make_adata_with_aligned_slots(rng):
    """Synthetic AnnData populated with obsp / varp / varm / layers so
    the lazy-bridge round-trip exercises every slot."""
    import anndata
    import pandas as pd

    n_obs, n_vars = 40, 24
    X = sp.random(
        n_obs, n_vars, density=0.15, format="csr", dtype=np.float32, random_state=rng
    )
    obs = pd.DataFrame(
        {"score": rng.random(n_obs).astype(np.float32)},
        index=pd.Index([f"CELL_{i}" for i in range(n_obs)]),
    )
    var = pd.DataFrame(index=pd.Index([f"GENE_{i}" for i in range(n_vars)]))
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.varm["PCs"] = rng.random((n_vars, 5)).astype(np.float32)
    adata.obsp["connectivities"] = sp.random(
        n_obs, n_obs, density=0.08, format="csr", dtype=np.float32, random_state=rng
    )
    adata.obsp["distances"] = sp.random(
        n_obs, n_obs, density=0.05, format="csr", dtype=np.float32, random_state=rng
    )
    adata.varp["gene_corr"] = sp.random(
        n_vars, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    adata.layers["spliced"] = sp.random(
        n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )
    return adata


def test_to_anndata_obsp_is_lazy(tmp_dir):
    """`to_anndata()` (lazy default) returns the SCX lazy bridge in
    `_obsp` / `_varp` / `_varm` / `_layers` so peak RSS is not paid up
    front. First access materialises through AnnData's validation
    loop and gives identical values to the eager path."""
    import pyscx

    rng = np.random.default_rng(101)
    adata = _make_adata_with_aligned_slots(rng)
    path = str(tmp_dir / "lazy.scx")
    pyscx.from_anndata(adata, path)

    lazy = pyscx.open(path).to_anndata()

    assert type(lazy._obsp).__name__ == "ScxLazyPairwiseMapping"
    assert type(lazy._varp).__name__ == "ScxLazyPairwiseMapping"
    assert type(lazy._varm).__name__ == "ScxLazyVarmMapping"
    assert type(lazy._layers).__name__ == "ScxLazyLayersMapping"

    # Existence-only probes don't decode the section payload — verified
    # indirectly via __contains__ / __len__, which the lazy bridges
    # serve from the catalog-derived key set.
    assert "connectivities" in lazy._obsp
    assert "distances" in lazy._obsp
    assert len(lazy._obsp) == 2

    # First public access goes through AnnData's PairwiseArrays init,
    # which calls our bridge's items() per key. Result must match the
    # eagerly-loaded reference.
    ref = pyscx.open(path).to_anndata(eager=True)
    diff = (lazy.obsp["connectivities"] - ref.obsp["connectivities"]).toarray()
    assert np.allclose(diff, 0.0, atol=1e-6)

    # Second access of the same key returns the cached object (identity
    # preserved through the bridge's internal cache, even though the
    # outer PairwiseArrays wrapper is reconstructed by the descriptor).
    a = lazy.obsp["connectivities"]
    b = lazy.obsp["connectivities"]
    assert a is b


def test_to_anndata_eager_kwarg(tmp_dir):
    """`eager=True` substitutes a plain `dict` so the returned AnnData
    is fully detached from the SCX file handle and matches pre-fix
    behaviour."""
    import pyscx

    rng = np.random.default_rng(202)
    adata = _make_adata_with_aligned_slots(rng)
    path = str(tmp_dir / "eager.scx")
    pyscx.from_anndata(adata, path)

    eager = pyscx.open(path).to_anndata(eager=True)

    assert isinstance(eager._obsp, dict)
    assert isinstance(eager._varp, dict)
    assert isinstance(eager._varm, dict)
    assert isinstance(eager._layers, dict)
    assert set(eager._obsp.keys()) == {"connectivities", "distances"}
    assert set(eager._varp.keys()) == {"gene_corr"}
    assert set(eager._varm.keys()) == {"PCs"}
    assert set(eager._layers.keys()) == {"spliced"}

    # Values match the source AnnData regardless of laziness.
    diff = (eager.obsp["connectivities"] - adata.obsp["connectivities"]).toarray()
    assert np.allclose(diff, 0.0, atol=1e-6)


def test_to_anndata_obsp_lazy_with_deletions(tmp_dir):
    """Lazy obsp must apply the file's deletion vector on first access,
    yielding the same `(n_kept, n_kept)` matrix as the eager path."""
    import pyscx

    rng = np.random.default_rng(303)
    n_obs, n_vars = 50, 20
    X = sp.random(
        n_obs, n_vars, density=0.12, format="csr", dtype=np.float32, random_state=rng
    )
    conn = sp.random(
        n_obs, n_obs, density=0.06, format="csr", dtype=np.float32, random_state=rng
    )
    import anndata

    ad = anndata.AnnData(X=X)
    ad.obsp["connectivities"] = conn
    path = str(tmp_dir / "lazy_del.scx")
    pyscx.from_anndata(ad, path)

    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[[2, 9, 17, 33, 47]] = True
    pyscx.open(path).mark_deleted(delete_mask)

    kept_mask = ~delete_mask
    n_kept = int(kept_mask.sum())
    expected = conn.tocsr()[kept_mask][:, kept_mask]

    lazy = pyscx.open(path).to_anndata()
    assert type(lazy._obsp).__name__ == "ScxLazyPairwiseMapping"
    assert lazy.n_obs == n_kept

    got = lazy.obsp["connectivities"]
    assert got.shape == (n_kept, n_kept)
    diff = (got - expected).toarray()
    assert np.allclose(diff, 0.0, atol=1e-6)


def test_to_anndata_layers_lazy_non_backed(tmp_dir):
    """Non-backed `layers` are wrapped in `ScxLazyLayersMapping`; only
    the accessed layer is decoded."""
    import anndata
    import pyscx

    rng = np.random.default_rng(404)
    n_obs, n_vars = 30, 12
    X = sp.random(
        n_obs, n_vars, density=0.2, format="csr", dtype=np.float32, random_state=rng
    )
    ad = anndata.AnnData(X=X)
    ad.layers["spliced"] = sp.random(
        n_obs, n_vars, density=0.15, format="csr", dtype=np.float32, random_state=rng
    )
    ad.layers["unspliced"] = sp.random(
        n_obs, n_vars, density=0.1, format="csr", dtype=np.float32, random_state=rng
    )

    path = str(tmp_dir / "layers_lazy.scx")
    pyscx.from_anndata(ad, path)

    lazy = pyscx.open(path).to_anndata()
    assert type(lazy._layers).__name__ == "ScxLazyLayersMapping"
    assert set(lazy._layers) == {"spliced", "unspliced"}

    # Round-trip values per layer.
    diff_s = (lazy.layers["spliced"] - ad.layers["spliced"]).toarray()
    assert np.allclose(diff_s, 0.0, atol=1e-6)
    diff_u = (lazy.layers["unspliced"] - ad.layers["unspliced"]).toarray()
    assert np.allclose(diff_u, 0.0, atol=1e-6)


def test_to_anndata_backed_aligned_slots_lazy(tmp_dir):
    """Backed mode also returns lazy bridges for obsp/varp/varm; layers
    in backed mode continue to use the per-entry `ScxBackedLayerDataset`."""
    import pyscx

    rng = np.random.default_rng(505)
    adata = _make_adata_with_aligned_slots(rng)
    path = str(tmp_dir / "backed_lazy.scx")
    pyscx.from_anndata(adata, path)

    backed = pyscx.open(path).to_anndata(backed=True)
    assert type(backed._obsp).__name__ == "ScxLazyPairwiseMapping"
    assert type(backed._varp).__name__ == "ScxLazyPairwiseMapping"
    assert type(backed._varm).__name__ == "ScxLazyVarmMapping"

    diff = (backed.obsp["connectivities"] - adata.obsp["connectivities"]).toarray()
    assert np.allclose(diff, 0.0, atol=1e-6)
    np.testing.assert_allclose(
        backed.varm["PCs"], adata.varm["PCs"], atol=1e-6
    )


def test_to_anndata_values_items_are_lazy(tmp_dir):
    """`values()` and `items()` on the lazy bridges return per-entry
    iterators (not pre-materialised lists). Each `next()` step pulls
    exactly one entry through the cache."""
    import pyscx

    rng = np.random.default_rng(606)
    adata = _make_adata_with_aligned_slots(rng)
    path = str(tmp_dir / "iter.scx")
    pyscx.from_anndata(adata, path)

    obsp = pyscx.open(path).to_anndata()._obsp
    assert type(obsp).__name__ == "ScxLazyPairwiseMapping"
    # 2 obsp keys present, 0 materialised at construction time
    assert "0 materialized" in repr(obsp)

    vit = obsp.values()
    assert type(vit).__name__ == "ScxLazyValueIterator"
    assert iter(vit) is vit  # __iter__ returns self
    next(vit)
    assert "1 materialized" in repr(obsp)
    next(vit)
    assert "2 materialized" in repr(obsp)

    # items() yields (key, value) tuples lazily on a fresh bridge.
    obsp2 = pyscx.open(path).to_anndata()._obsp
    iit = obsp2.items()
    assert type(iit).__name__ == "ScxLazyItemIterator"
    assert "0 materialized" in repr(obsp2)
    k, v = next(iit)
    assert isinstance(k, str)
    assert v is not None
    assert "1 materialized" in repr(obsp2)

    # list() drains the iterator and matches len()
    obsp3 = pyscx.open(path).to_anndata()._obsp
    drained = list(obsp3.values())
    assert len(drained) == len(obsp3)


def test_to_anndata_get_propagates_non_keyerror(tmp_dir):
    """`get()` returns `default` only for missing keys; real keys still
    return the decoded value. (Decode-failure propagation is hard to
    reproduce without a corrupted fixture, so the present-key + missing-
    key paths are the testable contract.)"""
    import pyscx

    rng = np.random.default_rng(707)
    adata = _make_adata_with_aligned_slots(rng)
    path = str(tmp_dir / "get.scx")
    pyscx.from_anndata(adata, path)
    lazy = pyscx.open(path).to_anndata()

    # Missing key → default sentinel
    sentinel = object()
    assert lazy._obsp.get("does_not_exist", sentinel) is sentinel
    # Missing key, no default → None
    assert lazy._obsp.get("also_missing") is None
    # Present key → real value (not the default)
    v = lazy._obsp.get("connectivities", sentinel)
    assert v is not sentinel
    assert v is not None
