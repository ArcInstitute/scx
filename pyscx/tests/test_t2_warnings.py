"""T2.1 / T2.2: lossy conversion paths surface as warnings, never silent.

T2.1 (I-CONV-1): an unreadable obs/var column is skipped with a
`skipped_column` `UserWarning` (was a bare `eprintln!`), and the column is
absent from the converted output.

T2.2 (I-CONV-2): a *dense* obsp matrix is preserved on ingest (stored as
nonzero COO, reconstructed by `to_anndata`), and a CSC / unsupported obsp
is dropped with a `dropped_obsp` `UserWarning` instead of being silently
mis-ingested.

Note: the SCX → h5ad exporter does not (yet) write obsp/varp back to
`/obsp`, so the dense-obsp round-trip is verified through
`pyscx.open(...).to_anndata()` (which reconstructs obsp from the COO
shards), proving the dense matrix was preserved on ingest.
"""

from __future__ import annotations

import numpy as np
import pytest


def _base_adata(n_obs=20, n_vars=5):
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    x = sp.csr_matrix(rng.integers(0, 5, size=(n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(
        {"n_counts": np.arange(n_obs, dtype=np.int32)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return anndata.AnnData(X=x, obs=obs, var=var)


# --------------------------------------------------------------------------
# T2.1 — unreadable obs column → skipped_column warning + column dropped
# --------------------------------------------------------------------------
def test_unreadable_obs_column_warns_and_drops(tmp_dir):
    import h5py

    import pyscx

    src = _base_adata()
    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)

    # Inject a column group with an encoding-type the reader cannot decode,
    # and register it in `/obs` column-order so the reader visits it.
    with h5py.File(h5ad_in, "r+") as f:
        obs = f["obs"]
        weird = obs.create_group("weird")
        weird.attrs["encoding-type"] = "future-array-encoding"
        weird.attrs["encoding-version"] = "0.1.0"
        order = [
            c.decode() if isinstance(c, bytes) else str(c)
            for c in obs.attrs["column-order"]
        ]
        order.append("weird")
        del obs.attrs["column-order"]
        obs.attrs.create(
            "column-order",
            np.array(order, dtype=object),
            dtype=h5py.string_dtype(),
        )

    scx_path = str(tmp_dir / "out.scx")
    with pytest.warns(UserWarning, match="skipped_column"):
        pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert "weird" not in out.obs.columns, "unsupported column must be dropped"
    # The good column survives.
    assert "n_counts" in out.obs.columns


# --------------------------------------------------------------------------
# T2.2 — dense obsp preserved (as nonzero COO); CSC obsp dropped + warns
# --------------------------------------------------------------------------
def test_dense_obsp_preserved_on_ingest(tmp_dir):
    import pyscx

    n_obs = 20
    src = _base_adata(n_obs=n_obs)
    rng = np.random.default_rng(1)
    dense = rng.random((n_obs, n_obs)).astype(np.float32)
    dense[dense < 0.7] = 0.0  # mostly zeros → COO stores only nonzeros
    src.obsp["dense_pw"] = dense

    h5ad_in = str(tmp_dir / "dense_obsp.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "dense_obsp.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert "dense_pw" in out.obsp, "dense obsp must be preserved, not dropped"
    got = out.obsp["dense_pw"]
    if hasattr(got, "toarray"):
        got = got.toarray()
    np.testing.assert_allclose(np.asarray(got, dtype=np.float32), dense)


def test_csc_obsp_dropped_with_warning(tmp_dir):
    import scipy.sparse as sp

    import pyscx

    n_obs = 20
    src = _base_adata(n_obs=n_obs)
    rng = np.random.default_rng(2)
    csc = sp.random(
        n_obs, n_obs, density=0.1, format="csc", random_state=2, dtype=np.float32
    )
    src.obsp["csc_pw"] = csc

    h5ad_in = str(tmp_dir / "csc_obsp.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "csc_obsp.scx")

    with pytest.warns(UserWarning, match="dropped_obsp"):
        pyscx.from_h5ad(h5ad_in, scx_path)

    # The unsupported pairwise matrix is absent, not silently mis-ingested.
    out = pyscx.open(scx_path).to_anndata()
    assert "csc_pw" not in out.obsp
