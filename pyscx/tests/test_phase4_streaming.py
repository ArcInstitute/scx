"""Phase 4 streaming `from_anndata` / `to_anndata` regression tests.

Covers:
- 4a/4b: one-key-at-a-time obsm/varm/obsp/varp streaming round-trip.
- 4c: `from_anndata` auto-shards obs/var when `n_obs > shard_target_rows`,
  and the `force_legacy_metadata=True` opt-out preserves the single-section
  layout.
- 4b warning: `MappingPeakFootprintHigh` fires when an individual obsm /
  obsp footprint exceeds `memory_budget`.
- 4d warning: `EagerAssemblyMemoryHigh` fires when `to_anndata()` estimated
  assembly exceeds the caller's `memory_budget`.
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _build_adata(n_obs, n_vars, *, with_obsm=True, with_varm=False,
                 with_obsp=False, with_varp=False):
    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"cell_id": [f"c{i}" for i in range(n_obs)]},
        index=[f"c{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"g{i}" for i in range(n_vars)]},
        index=[f"g{i}" for i in range(n_vars)],
    )
    obsm = {}
    if with_obsm:
        obsm["X_pca"] = rng.standard_normal((n_obs, 8)).astype(np.float32)
    varm = {}
    if with_varm:
        varm["W"] = rng.standard_normal((n_vars, 4)).astype(np.float32)
    obsp = {}
    if with_obsp:
        # symmetric tiny connectivity
        m = sp.random(n_obs, n_obs, density=0.05, format="csr", random_state=1,
                      dtype=np.float32)
        obsp["connectivities"] = m
    varp = {}
    if with_varp:
        m = sp.random(n_vars, n_vars, density=0.05, format="csr", random_state=2,
                      dtype=np.float32)
        varp["W"] = m
    adata = anndata.AnnData(X=x, obs=obs, var=var, obsm=obsm)
    if varm:
        for k, v in varm.items():
            adata.varm[k] = v
    if obsp:
        for k, v in obsp.items():
            adata.obsp[k] = v
    if varp:
        for k, v in varp.items():
            adata.varp[k] = v
    return adata


def test_from_anndata_shards_obs_when_rows_exceed_shard_size(tmp_path):
    """Phase 4c: n_obs > shard_size triggers ObsMetadataShard layout."""
    import pyscx

    n_obs = 17_000  # > default shard_size 16384
    adata = _build_adata(n_obs=n_obs, n_vars=20)
    out = str(tmp_path / "sharded_obs.scx")
    pyscx.from_anndata(adata, out, shard_size=16384)

    exp = pyscx.open(out)
    assert exp.n_obs == n_obs
    # Two shards: 16384 + 616 rows.
    assert exp.obs_metadata_shard_count == 2
    assert exp.var_metadata_shard_count == 0  # n_vars=20 ≤ shard_size
    ad = exp.to_anndata()
    assert ad.n_obs == n_obs
    assert ad.obs.index.tolist() == adata.obs.index.tolist()
    # X round-trip
    np.testing.assert_array_equal(
        ad.X.toarray(), adata.X.toarray()
    )


def test_from_anndata_force_legacy_metadata_keeps_single_section(tmp_path):
    """Phase 4c: force_legacy_metadata=True preserves single-section obs."""
    import pyscx

    n_obs = 17_000
    adata = _build_adata(n_obs=n_obs, n_vars=20)
    out = str(tmp_path / "legacy_obs.scx")
    pyscx.from_anndata(adata, out, shard_size=16384, force_legacy_metadata=True)

    exp = pyscx.open(out)
    assert exp.n_obs == n_obs
    # Hard assertion on the layout: no ObsMetadataShard sections emitted.
    assert exp.obs_metadata_shard_count == 0
    assert exp.var_metadata_shard_count == 0
    ad = exp.to_anndata()
    assert ad.n_obs == n_obs
    assert ad.obs.index.tolist() == adata.obs.index.tolist()


def test_from_anndata_streams_obsm_varm_obsp_varp_round_trip(tmp_path):
    """Phase 4a/4b: per-key streaming preserves obsm / varm / obsp / varp."""
    import pyscx

    adata = _build_adata(
        n_obs=100,
        n_vars=20,
        with_obsm=True,
        with_varm=True,
        with_obsp=True,
        with_varp=True,
    )
    out = str(tmp_path / "mappings.scx")
    pyscx.from_anndata(adata, out)

    # Suppress the obsm/varm/obsp/varp dropped-by-query-engine warning.
    ad = pyscx.open(out).to_anndata(eager=True)

    np.testing.assert_allclose(
        ad.obsm["X_pca"], adata.obsm["X_pca"], rtol=1e-5
    )
    np.testing.assert_allclose(
        ad.varm["W"], adata.varm["W"], rtol=1e-5
    )
    # obsp / varp round-trip nonzero pattern (values cast to f32)
    src_obsp = adata.obsp["connectivities"].tocoo()
    out_obsp = ad.obsp["connectivities"].tocoo()
    assert out_obsp.nnz == src_obsp.nnz
    src_varp = adata.varp["W"].tocoo()
    out_varp = ad.varp["W"].tocoo()
    assert out_varp.nnz == src_varp.nnz


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.int32, np.int64],
    ids=["f32", "f64", "i32", "i64"],
)
def test_from_anndata_obsm_dtype_round_trip(tmp_path, dtype):
    """All four `numpy_or_pandas_to_record_batch` fast-path dtypes
    must round-trip through SCX byte-identically. The `.tobytes()` +
    transpose path is sensitive to (a) byte ordering and (b) per-column
    extraction logic; either bug shows up as wrong values per cell.
    """
    import pyscx

    rng = np.random.default_rng(0)
    n_obs, n_vars = 40, 5
    adata = _build_adata(n_obs=n_obs, n_vars=n_vars, with_obsm=False)
    # Build a 2-D embedding of the parametrised dtype. Use small ints
    # so int32/int64 round-trip exactly under astype.
    if np.issubdtype(dtype, np.floating):
        embedding = rng.standard_normal((n_obs, 6)).astype(dtype)
    else:
        embedding = rng.integers(-1_000, 1_000, size=(n_obs, 6)).astype(dtype)
    adata.obsm["embed"] = embedding
    out = str(tmp_path / f"embed_{dtype.__name__}.scx")
    pyscx.from_anndata(adata, out)

    ad = pyscx.open(out).to_anndata(eager=True)
    got = ad.obsm["embed"]
    if np.issubdtype(dtype, np.floating):
        np.testing.assert_allclose(got, embedding, rtol=1e-6)
    else:
        # Integer dtypes must round-trip exactly through the fast path.
        np.testing.assert_array_equal(got.astype(dtype), embedding)


def test_mapping_peak_footprint_high_warning_fires(tmp_path):
    """Phase 4b: tiny `memory_budget` triggers MappingPeakFootprintHigh."""
    import pyscx

    adata = _build_adata(n_obs=200, n_vars=20, with_obsm=True)
    out = str(tmp_path / "small_budget.scx")
    # obsm X_pca is 200 * 8 * 4 = 6400 bytes; budget=1024 → triggers warning.
    with pytest.warns(UserWarning, match="mapping_peak_footprint_high"):
        pyscx.from_anndata(adata, out, memory_budget=1024)


def test_eager_assembly_memory_high_warning_fires(tmp_path):
    """Phase 4d: eager `to_anndata` warns when estimated bytes > budget."""
    import pyscx

    adata = _build_adata(n_obs=200, n_vars=20)
    out = str(tmp_path / "eager_budget.scx")
    pyscx.from_anndata(adata, out)

    # Tiny budget guaranteed to trip the estimate regardless of file size.
    with pytest.warns(UserWarning, match="eager_assembly_memory_high"):
        pyscx.open(out).to_anndata(memory_budget=64)


def test_eager_assembly_no_warning_under_budget(tmp_path):
    """Phase 4d: no warning when assembly fits the budget."""
    import pyscx

    adata = _build_adata(n_obs=50, n_vars=10)
    out = str(tmp_path / "fits_budget.scx")
    pyscx.from_anndata(adata, out)

    # Large budget; assembly cost is tiny → no warning emitted.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Only the eager-assembly warning class — other UserWarnings (e.g.
        # query-engine drop notice) can still fire elsewhere. Use the
        # narrower filterwarnings instead.
        warnings.resetwarnings()
        warnings.filterwarnings(
            "error",
            message=r".*eager_assembly_memory_high.*",
            category=UserWarning,
        )
        ad = pyscx.open(out).to_anndata(memory_budget=1 << 32)  # 4 GiB
        assert ad.n_obs == 50
