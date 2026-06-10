"""T1.2: `adata.raw` round-trips through SCX.

`adata.raw` has its OWN (wider) var axis. These tests confirm the file
ingest path (`pyscx.from_h5ad`) preserves it and that both
`pyscx.open(...).to_anndata()` reconstruction and `pyscx.to_h5ad` export
reproduce `adata.raw.X` and `adata.raw.var`.
"""

from __future__ import annotations

import numpy as np
import pytest


def _dense(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _adata_with_raw(n_obs=20, n_vars=12, raw_n_vars=30):
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0.0
    obs = pd.DataFrame(
        {"n_counts": np.arange(n_obs, dtype=np.int32)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"mean_expr": np.linspace(0.1, 1.0, n_vars, dtype=np.float32)},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    adata = anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)

    # Raw: WIDER var axis (raw_n_vars > n_vars), integer counts.
    raw_dense = rng.integers(0, 40, size=(n_obs, raw_n_vars)).astype(np.float32)
    raw_dense[rng.random((n_obs, raw_n_vars)) > 0.4] = 0.0
    raw_var = pd.DataFrame(index=[f"raw_gene_{i}" for i in range(raw_n_vars)])
    adata.raw = anndata.AnnData(X=sp.csr_matrix(raw_dense), var=raw_var)
    return adata, raw_dense


def test_raw_reconstructed_in_to_anndata(tmp_dir):
    import anndata  # noqa: F401
    import pyscx

    n_obs, n_vars, raw_n_vars = 20, 12, 30
    src, raw_dense = _adata_with_raw(n_obs, n_vars, raw_n_vars)

    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is not None, "adata.raw must be reconstructed"
    assert out.raw.shape == (n_obs, raw_n_vars)
    assert out.n_vars == n_vars, "main X var axis must be unchanged"
    np.testing.assert_array_equal(raw_dense, _dense(out.raw.X))
    assert list(out.raw.var_names) == [f"raw_gene_{i}" for i in range(raw_n_vars)]


def test_raw_round_trips_to_h5ad(tmp_dir):
    import anndata
    import pyscx

    n_obs, n_vars, raw_n_vars = 20, 12, 30
    src, raw_dense = _adata_with_raw(n_obs, n_vars, raw_n_vars)

    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    h5ad_out = str(tmp_dir / "out.h5ad")
    pyscx.to_h5ad(scx_path, h5ad_out)

    rt = anndata.read_h5ad(h5ad_out)
    assert rt.raw is not None, "raw must survive scx → h5ad"
    assert rt.raw.shape == (n_obs, raw_n_vars)
    np.testing.assert_array_equal(raw_dense, _dense(rt.raw.X))
    assert list(rt.raw.var_names) == [f"raw_gene_{i}" for i in range(raw_n_vars)]


def test_raw_dropped_with_warning_in_backed_mode(tmp_dir):
    import anndata  # noqa: F401
    import pyscx

    src, _ = _adata_with_raw(20, 12, 30)
    h5ad_in = str(tmp_dir / "in.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "raw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    # Backed mode does not reconstruct raw's obs axis → drop + warn
    # (human-readable DroppedRaw message), never silent.
    with pytest.warns(UserWarning, match="dropped_raw"):
        out = pyscx.open(scx_path).to_anndata(backed=True)
    assert out.raw is None


def test_no_raw_is_none(tmp_dir):
    import anndata
    import pandas as pd
    import scipy.sparse as sp
    import pyscx

    rng = np.random.default_rng(1)
    dense = rng.integers(0, 10, size=(8, 5)).astype(np.float32)
    src = anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )
    h5ad_in = str(tmp_dir / "noraw.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "noraw.scx")
    pyscx.from_h5ad(h5ad_in, scx_path)

    out = pyscx.open(scx_path).to_anndata()
    assert out.raw is None
