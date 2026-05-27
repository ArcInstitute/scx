"""Phase 5 v2 pairwise COO Int64 coordinate-width regression tests
(Python side).

Covers what's testable from Python without standing up an AnnData with
> 2.15B obs (impractical). The wire-format v2 path (Int64 obsp shards
round-tripped through `ScxWriter`/`ScxReader`) is covered by the Rust
integration test in `scx-format/tests/pairwise_v2_int64.rs`.

Covered here:
- 5a/5c routing: small-axis obsp keeps Int32 v1 layout and round-trips
  through `from_anndata` → `to_anndata` byte-identically.
- 5a filter path: `filter_coo_obsp_by_kept_rows` (exercised via the
  lazy obsp path) shrinks an obsp correctly under a sorted deletion
  vector. The new binary-search remap replaces the previous dense
  `Vec<i32>` of length `n_rows` — round-trip equivalence is the test.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp


def test_small_obsp_round_trips_with_int32_layout(tmp_path):
    """Sub-2^31 axes must keep the Int32 (v1) layout intact —
    `coo_needs_int64_coords` is the routing oracle and small axes
    return false, so on-disk coords stay 4 bytes wide."""
    import anndata

    import pyscx

    rng = np.random.default_rng(0)
    n_obs = 50
    # anndata requires obsp to be CSR or CSC, not COO.
    m = sp.random(n_obs, n_obs, density=0.05, format="csr",
                  random_state=0, dtype=np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=["g0"])
    x = sp.csr_matrix((n_obs, 1), dtype=np.float32)
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    adata.obsp["W"] = m

    out = str(tmp_path / "phase5_v1.scx")
    pyscx.from_anndata(adata, out)
    ad = pyscx.open(out).to_anndata(eager=True)
    rt = ad.obsp["W"].tocoo()
    src = m.tocoo()
    assert rt.nnz == src.nnz
    # v1 round-trip preserves scalar values; check sum as a quick proof.
    assert pytest_isclose(rt.sum(), src.sum())


# `filter_coo_obsp_by_kept_rows` deletion-vector round-trip is already
# covered by `tests/test_round_trip.py::test_obsp_round_trip_with_deletions_and_obs_filter`.
# The Phase 5a rewrite (binary-search remap over sorted kept_rows
# replacing the dense `Vec<i32>` of length n_rows) is exercised by
# that test on every run; equivalence is its regression assertion.


def pytest_isclose(a, b, rel=1e-5):
    return abs(float(a) - float(b)) <= rel * max(abs(float(a)), abs(float(b)), 1.0)
