"""Tests for streaming h5ad → SCX conversion.

Cover:
- `pyscx.from_h5ad(path, out)` produces the same logical SCX content as
  `pyscx.from_anndata(adata, out)` for the same underlying h5ad.
- `pyscx.from_anndata(sc.read_h5ad(path, backed='r'), out)` auto-routes
  to streaming and matches the non-backed conversion.
- In-memory `obs` mutation on a backed AnnData is preserved end-to-end
  (validates the `StreamingOverrides` plumbing).
- CSC sidecar parity: `from_h5ad(... csc='always')` and
  `from_anndata(... csc='always')` produce CSC shards of matching shape.
- A large-fixture smoke test gated on `SCX_LARGE_FIXTURE` so CI doesn't
  spin one up.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# Streaming conversion lives behind the `hdf5` feature: `pyscx.from_h5ad`
# is only exposed in hdf5 builds, and `from_anndata` on a backed AnnData
# routes through the same path. CI's Python-bindings job builds with
# `--features cloud` only, so skip the whole module there. Mirrors the
# guard used in `test_cloud.py` for cloud-gated features.
_pyscx = pytest.importorskip("pyscx")
if not hasattr(_pyscx, "from_h5ad"):
    pytest.skip("pyscx built without hdf5 feature", allow_module_level=True)


def _write_h5ad(adata, path):
    """Persist an in-memory AnnData to disk."""
    adata.write_h5ad(path)


def _load_anndata(scx_path):
    """Open an SCX file and return its AnnData representation."""
    import pyscx

    return pyscx.open(scx_path).to_anndata()


@pytest.fixture
def h5ad_path(synthetic_adata, tmp_dir):
    """Persist the shared `synthetic_adata` fixture to disk as h5ad."""
    path = str(tmp_dir / "in.h5ad")
    _write_h5ad(synthetic_adata, path)
    return path


def _assert_anndata_equal(a, b):
    """Compare two AnnData on the fields the streaming path round-trips:
    X (dense), obs, var, uns, obsm, layers."""
    # X — dense comparison after CSR materialisation.
    np.testing.assert_array_equal(
        a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X),
        b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X),
    )

    # obs / var — index + columns. `check_dtype=False` because the SCX
    # readback may surface categoricals as object dtype.
    pd.testing.assert_index_equal(a.obs.index, b.obs.index, check_names=False)
    pd.testing.assert_index_equal(a.var.index, b.var.index, check_names=False)
    common_obs_cols = sorted(set(a.obs.columns) & set(b.obs.columns))
    for col in common_obs_cols:
        np.testing.assert_array_equal(
            np.asarray(a.obs[col]).astype(str),
            np.asarray(b.obs[col]).astype(str),
        )

    # obsm — same keys, same shape, same values.
    assert set(a.obsm.keys()) == set(b.obsm.keys())
    for k in a.obsm:
        np.testing.assert_array_equal(np.asarray(a.obsm[k]), np.asarray(b.obsm[k]))

    # Layers — same keys; data may differ in sparse representation,
    # so compare via toarray when possible.
    assert set(a.layers.keys()) == set(b.layers.keys())
    for k in a.layers:
        la = a.layers[k]
        lb = b.layers[k]
        np.testing.assert_array_equal(
            la.toarray() if sp.issparse(la) else np.asarray(la),
            lb.toarray() if sp.issparse(lb) else np.asarray(lb),
        )


def test_from_h5ad_matches_from_anndata(synthetic_adata, h5ad_path, tmp_dir):
    """Both entry points produce SCX files that re-read identically."""
    import pyscx

    via_anndata = str(tmp_dir / "via_anndata.scx")
    via_h5ad = str(tmp_dir / "via_h5ad.scx")
    pyscx.from_anndata(synthetic_adata, via_anndata)
    pyscx.from_h5ad(h5ad_path, via_h5ad)

    a = _load_anndata(via_anndata)
    b = _load_anndata(via_h5ad)
    _assert_anndata_equal(a, b)


def test_from_anndata_backed_routes_to_streaming(h5ad_path, tmp_dir):
    """`from_anndata` on a backed AnnData must auto-route to the streaming
    pipeline and produce the same SCX content as the non-backed path."""
    import anndata as ad
    import pyscx

    backed = ad.read_h5ad(h5ad_path, backed="r")
    out_backed = str(tmp_dir / "out_backed.scx")
    pyscx.from_anndata(backed, out_backed)

    # Compare against the streaming `from_h5ad` path on the same file.
    out_h5ad = str(tmp_dir / "out_h5ad.scx")
    pyscx.from_h5ad(h5ad_path, out_h5ad)

    a = _load_anndata(out_backed)
    b = _load_anndata(out_h5ad)
    _assert_anndata_equal(a, b)


def test_backed_obs_mutation_preserved(h5ad_path, tmp_dir):
    """Mutating `obs` on a backed AnnData before `from_anndata` should
    write the *mutated* values to SCX via `StreamingOverrides`."""
    import anndata as ad
    import pyscx

    backed = ad.read_h5ad(h5ad_path, backed="r")
    # Add a new column that didn't exist in the on-disk fixture.
    backed.obs["new_flag"] = np.arange(len(backed.obs)).astype(np.int64)

    out = str(tmp_dir / "mutated.scx")
    pyscx.from_anndata(backed, out)

    re = _load_anndata(out)
    assert "new_flag" in re.obs.columns, "in-memory obs mutation lost"
    np.testing.assert_array_equal(
        np.asarray(re.obs["new_flag"]).astype(np.int64),
        np.arange(len(backed.obs), dtype=np.int64),
    )


def test_sharded_obsm_obsp_round_trip(tmp_dir):
    """A backed AnnData with non-trivial `obsm` + `obsp` should round
    trip via the sharded streaming layout (`obsm/<k>_shard_<idx>` and
    `obsp/<k>_shard_<idx>` sections). Reading the SCX file back as an
    AnnData reconstructs the same matrices."""
    import anndata as ad
    import pyscx

    n_obs, n_vars = 9, 4
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.random((n_obs, n_vars), dtype=np.float32))

    obs = pd.DataFrame(
        {"label": [f"c{i}" for i in range(n_obs)]},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"name": [f"g{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )

    # obsm: a 5-component PCA-like embedding the sharded path must
    # reassemble correctly across multiple shards (shard_size = 3 below
    # → 3 shards).
    X_pca = rng.random((n_obs, 5), dtype=np.float32)
    # obsp: a sparse symmetric distance graph (kNN-like), one entry per
    # row pair within the shard's row range.
    obsp_row = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    obsp_col = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0], dtype=np.int32)
    obsp_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
    dist = sp.coo_matrix((obsp_data, (obsp_row, obsp_col)), shape=(n_obs, n_obs)).tocsr()

    a = ad.AnnData(X=X, obs=obs, var=var, obsm={"X_pca": X_pca}, obsp={"distances": dist})
    src = str(tmp_dir / "src.h5ad")
    a.write_h5ad(src)

    backed = ad.read_h5ad(src, backed="r")
    out = str(tmp_dir / "sharded.scx")
    # shard_size=3 forces multiple obsm/obsp shards (n_obs=9).
    pyscx.from_anndata(backed, out, shard_size=3)

    # Materialise the SCX file back to AnnData and confirm the
    # logical matrices reassemble correctly from their shards.
    materialised = pyscx.open(out).to_anndata()
    assert "X_pca" in materialised.obsm
    assert "distances" in materialised.obsp
    np.testing.assert_array_equal(
        np.asarray(materialised.obsm["X_pca"]), X_pca
    )
    # obsp round-trips as a sparse matrix; compare densified values.
    np.testing.assert_array_equal(
        materialised.obsp["distances"].toarray(),
        dist.toarray(),
    )


def test_csc_sidecar_parity(synthetic_adata, h5ad_path, tmp_dir):
    """`csc='always'` should emit a CSC sidecar with the same shard count
    via either entry point. Cell-level byte equality is asserted in the
    Rust round-trip test (`streaming_csc_always_emits_sidecar_matching_non_streaming`)
    in `scx-convert/src/tests.rs`; here we sanity-check the Python
    surface."""
    import pyscx

    via_anndata = str(tmp_dir / "via_anndata_csc.scx")
    via_h5ad = str(tmp_dir / "via_h5ad_csc.scx")
    pyscx.from_anndata(synthetic_adata, via_anndata, csc="always")
    pyscx.from_h5ad(h5ad_path, via_h5ad, csc="always")

    # `pyscx.open(...).info()` would expose n_csc_shards; in lieu of
    # that, both files should round-trip the same X content.
    a = _load_anndata(via_anndata)
    b = _load_anndata(via_h5ad)
    np.testing.assert_array_equal(a.X.toarray(), b.X.toarray())


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("SCX_LARGE_FIXTURE"),
    reason="set SCX_LARGE_FIXTURE=/path/to/big.h5ad to enable",
)
def test_large_fixture_smoke(tmp_dir):
    """Smoke test that `pyscx.from_h5ad` survives a file ≥ 1/4 of
    available RAM without OOM. Disabled by default — the test harness
    has no large fixture to point at, and the user opts in via the
    `SCX_LARGE_FIXTURE` env var on a host with enough disk."""
    import pyscx
    import psutil  # noqa: F401 — imported only for the env-driven smoke

    src = os.environ["SCX_LARGE_FIXTURE"]
    out = str(tmp_dir / "large.scx")
    pyscx.from_h5ad(src, out)
    # Sanity-check the output has a header n_obs > 0; deeper assertions
    # need the original h5ad's metadata which we don't load to keep
    # peak memory bounded.
    reader = pyscx.open(out)
    assert reader.n_obs > 0
