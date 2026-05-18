"""Phase 8: `pyscx.to_h5mu` round-trip tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

# `mudata` is an optional dependency. CI's CPU-only Python lane does not
# install it; skip the entire module rather than error on collect.
pytest.importorskip("mudata")


def _dense(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


@pytest.fixture
def synthetic_cite_mu():
    """Tiny CITE-seq MuData (6 cells, 4 genes + 3 antibody features)."""
    import anndata
    import mudata

    n_obs = 6
    rna = anndata.AnnData(
        X=sp.csr_matrix(np.eye(n_obs, 4, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    adt = anndata.AnnData(
        X=sp.csr_matrix(np.eye(n_obs, 3, dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"a{i}" for i in range(3)]),
    )
    return mudata.MuData({"rna": rna, "adt": adt})


def test_to_h5mu_round_trip_streaming(tmp_dir, synthetic_cite_mu):
    """`pyscx.from_mudata → pyscx.to_h5mu → mudata.read_h5mu`."""
    import mudata
    import pyscx

    scx_path = str(tmp_dir / "cite.scx")
    h5mu_out = str(tmp_dir / "cite.h5mu")
    pyscx.from_mudata(synthetic_cite_mu, scx_path)
    pyscx.to_h5mu(scx_path, h5mu_out, stream=True)

    out = mudata.read_h5mu(h5mu_out)
    assert set(out.mod.keys()) == {"rna", "adt"}
    np.testing.assert_array_equal(
        _dense(synthetic_cite_mu.mod["rna"].X),
        _dense(out.mod["rna"].X),
    )
    np.testing.assert_array_equal(
        _dense(synthetic_cite_mu.mod["adt"].X),
        _dense(out.mod["adt"].X),
    )


def test_to_h5mu_streaming_matches_materialising(tmp_dir, synthetic_cite_mu):
    """`stream=False` and `stream=True` must produce equivalent X matrices."""
    import mudata
    import pyscx

    scx_path = str(tmp_dir / "cite.scx")
    h5mu_stream = str(tmp_dir / "stream.h5mu")
    h5mu_mat = str(tmp_dir / "mat.h5mu")
    pyscx.from_mudata(synthetic_cite_mu, scx_path)
    pyscx.to_h5mu(scx_path, h5mu_stream, stream=True)
    pyscx.to_h5mu(scx_path, h5mu_mat, stream=False)

    a = mudata.read_h5mu(h5mu_stream)
    b = mudata.read_h5mu(h5mu_mat)
    for modality in a.mod:
        np.testing.assert_array_equal(
            _dense(a.mod[modality].X),
            _dense(b.mod[modality].X),
            err_msg=f"modality {modality} diverged between stream / materialising",
        )
