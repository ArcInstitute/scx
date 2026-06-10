"""Item 2: `to_h5ad` re-emits obsp/varp pairwise matrices.

Previously the SCX→h5ad exporter dropped `/obsp` and `/varp` (they
round-tripped only via `to_anndata`). Now they are written back as
float32 `csr_matrix` groups, so a full `h5ad → scx → h5ad` (and
`from_anndata → to_h5ad`) preserves them. A dense `obsp` input is
preserved (re-exported as sparse, values identical).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def _base(n_obs=20, n_vars=6):
    import anndata

    rng = np.random.default_rng(0)
    x = sp.csr_matrix(rng.integers(0, 5, size=(n_obs, n_vars)).astype(np.float32))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    return anndata.AnnData(X=x, obs=obs, var=var)


def test_obsp_varp_round_trip_to_h5ad(tmp_dir):
    import anndata

    import pyscx

    n_obs, n_vars = 20, 6
    src = _base(n_obs, n_vars)
    w = sp.random(n_obs, n_obs, density=0.2, format="csr", random_state=1, dtype=np.float32)
    v = sp.random(n_vars, n_vars, density=0.3, format="csr", random_state=2, dtype=np.float32)
    src.obsp["W"] = w
    src.varp["V"] = v

    scx_path = str(tmp_dir / "pw.scx")
    pyscx.from_anndata(src, scx_path)
    h5ad_out = str(tmp_dir / "pw_out.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)

    out = anndata.read_h5ad(h5ad_out)
    assert "W" in out.obsp, "obsp must be re-emitted by to_h5ad"
    assert "V" in out.varp, "varp must be re-emitted by to_h5ad"
    np.testing.assert_allclose(out.obsp["W"].toarray(), w.toarray())
    np.testing.assert_allclose(out.varp["V"].toarray(), v.toarray())


def test_dense_obsp_round_trips_to_h5ad_as_sparse(tmp_dir):
    import anndata

    import pyscx

    n_obs = 16
    src = _base(n_obs, 4)
    rng = np.random.default_rng(3)
    dense = rng.random((n_obs, n_obs)).astype(np.float32)
    dense[dense < 0.6] = 0.0
    src.obsp["dense"] = dense  # dense ndarray obsp

    h5ad_in = str(tmp_dir / "din.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "d.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)
    h5ad_out = str(tmp_dir / "dout.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)

    out = anndata.read_h5ad(h5ad_out)
    assert "dense" in out.obsp
    got = out.obsp["dense"]
    if hasattr(got, "toarray"):
        got = got.toarray()
    np.testing.assert_allclose(np.asarray(got, dtype=np.float32), dense)
