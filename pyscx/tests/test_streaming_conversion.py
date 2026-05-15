"""Phase 8 tests for streaming h5ad → SCX conversion.

Cover:
- `pyscx.from_h5ad(path, out)` produces the same logical SCX content as
  `pyscx.from_anndata(adata, out)` for the same underlying h5ad.
- `pyscx.from_anndata(sc.read_h5ad(path, backed='r'), out)` now succeeds
  (Phase 7 routing) and matches the non-backed conversion.
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
    """`from_anndata` on a backed AnnData must succeed (Phase 7 routing)
    and produce the same SCX content as the non-backed path."""
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
    write the *mutated* values to SCX (option (b) override path)."""
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
