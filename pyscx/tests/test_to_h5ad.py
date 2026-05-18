"""Phase 8: `pyscx.to_h5ad` round-trip tests.

Mirrors `pyscx.from_h5ad`. Streams by default — these tests confirm
the round-trip is bit-exact for integer X values and that the
extracted h5ad reads cleanly via `anndata.read_h5ad`.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dense_from_anndata_x(adata):
    """Return X as a dense float32 array, regardless of sparse/dense."""
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _simple_adata(n_obs=20, n_vars=12):
    """An AnnData fixture whose obs/var only use column dtypes the
    materialising h5ad_write supports today (`Int32`, `Int64`,
    `Float32`, `Float64`, `Utf8`, `Boolean`, `Dictionary<Int32,
    Utf8>`). Avoids `Dictionary<Int8, Utf8>` categoricals which the
    writer currently skips — that's a pre-existing limitation, not
    Phase 8 scope.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0.0
    x = sp.csr_matrix(dense)
    obs = pd.DataFrame(
        {"n_counts": np.arange(n_obs, dtype=np.int32)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"mean_expr": np.linspace(0.1, 1.0, n_vars, dtype=np.float32)},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=x, obs=obs, var=var)


def test_to_h5ad_round_trip_streaming(tmp_dir):
    """`pyscx.from_anndata → pyscx.to_h5ad → anndata.read_h5ad` round-trip."""
    import anndata
    import pyscx

    src = _simple_adata()
    scx_path = str(tmp_dir / "round_trip.scx")
    h5ad_out = str(tmp_dir / "round_trip.h5ad")
    pyscx.from_anndata(src, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_out, stream=True)

    out = anndata.read_h5ad(h5ad_out)
    np.testing.assert_array_equal(
        _dense_from_anndata_x(src),
        _dense_from_anndata_x(out),
    )
    # Whether the obs/var index round-trips to the same labels is an
    # `from_anndata` concern (pyscx writes the index as a regular
    # column); the Phase 8 contract is that `to_h5ad` preserves
    # whatever obs/var SCX has.
    assert out.n_obs == src.n_obs
    assert out.n_vars == src.n_vars


def test_to_h5ad_round_trip_materialising(tmp_dir):
    """`stream=False` falls back to the materialising path; output
    must match the streaming path."""
    import anndata
    import pyscx

    src = _simple_adata()
    scx_path = str(tmp_dir / "rt.scx")
    h5ad_stream = str(tmp_dir / "rt_stream.h5ad")
    h5ad_mat = str(tmp_dir / "rt_mat.h5ad")
    pyscx.from_anndata(src, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_stream, stream=True)
    pyscx.to_h5ad(scx_path, h5ad_mat, stream=False)

    stream = anndata.read_h5ad(h5ad_stream)
    mat = anndata.read_h5ad(h5ad_mat)
    np.testing.assert_array_equal(
        _dense_from_anndata_x(stream),
        _dense_from_anndata_x(mat),
    )


def test_to_h5ad_multimodal_without_modality_raises(tmp_dir):
    """Plain `to_h5ad` on a multimodal SCX file must raise; the
    user should pass `modality=...` or call `to_h5mu`."""
    import anndata
    import mudata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

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
    mu = mudata.MuData({"rna": rna, "adt": adt})
    scx_path = str(tmp_dir / "cite.scx")
    h5ad_out = str(tmp_dir / "cite.h5ad")
    pyscx.from_mudata(mu, scx_path)

    with pytest.raises(Exception):
        pyscx.to_h5ad(scx_path, h5ad_out)


def test_to_h5ad_modality_extract(tmp_dir):
    """Multimodal SCX → single-modality h5ad via `modality='rna'`."""
    import anndata
    import mudata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

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
    mu = mudata.MuData({"rna": rna, "adt": adt})
    scx_path = str(tmp_dir / "cite.scx")
    h5ad_out = str(tmp_dir / "rna.h5ad")
    pyscx.from_mudata(mu, scx_path)
    pyscx.to_h5ad(scx_path, h5ad_out, modality="rna", stream=True)

    out = anndata.read_h5ad(h5ad_out)
    assert out.n_vars == 4  # rna's gene count, not adt's
    np.testing.assert_array_equal(
        _dense_from_anndata_x(rna), _dense_from_anndata_x(out)
    )
