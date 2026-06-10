"""T2.4: a pandas DataFrame in `uns` flattens with a visible warning.

SCX has no DataFrame round-trip, so a `uns` DataFrame is preserved as a
nested dict (per-column values + `_index`) rather than reconstructed as a
DataFrame. That structure loss must NOT be silent: conversion emits a
`flattened_uns_dataframe` `UserWarning`. The column data still survives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _adata_with_uns_df():
    import anndata
    import scipy.sparse as sp

    rng = np.random.default_rng(0)
    x = sp.csr_matrix(rng.integers(0, 5, size=(6, 3)).astype(np.float32))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(6)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(3)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)
    adata.uns["scores_df"] = pd.DataFrame(
        {"a": [1, 2, 3], "b": ["x", "y", "z"]},
        index=["r0", "r1", "r2"],
    )
    return adata


def test_uns_dataframe_flattens_with_warning(tmp_dir):
    import anndata  # noqa: F401

    import pyscx

    src = _adata_with_uns_df()
    h5ad_in = str(tmp_dir / "uns_df.h5ad")
    src.write_h5ad(h5ad_in)
    scx_path = str(tmp_dir / "uns_df.scx")

    # No silent flatten — the structure loss is surfaced.
    with pytest.warns(UserWarning, match="flattened_uns_dataframe"):
        pyscx.from_h5ad(h5ad_in, scx_path)

    # The column data is preserved (as a nested dict, not a DataFrame).
    out = pyscx.open(scx_path).to_anndata()
    df = out.uns["scores_df"]
    assert isinstance(df, dict)
    assert list(df["a"]) == [1, 2, 3]
    assert list(df["b"]) == ["x", "y", "z"]
