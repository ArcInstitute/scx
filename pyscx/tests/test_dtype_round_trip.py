"""Regression: pandas-extension dtypes (Int64, boolean, Categorical) must
survive `pyscx.from_anndata` → `pyscx.open(...).to_anndata()`.

Guards against the dtype-coercion regression introduced when the
B1/B2-2026-05-20 fix unconditionally stripped the full pyarrow `pandas`
metadata envelope on read.
"""

import anndata
import numpy as np
import pandas as pd
import pyscx
import scipy.sparse as sp


def test_obs_extension_dtypes_round_trip(tmp_dir):
    n_obs, n_vars = 4, 2
    x = sp.csr_matrix(
        np.arange(n_obs * n_vars, dtype=np.float32).reshape(n_obs, n_vars)
    )
    obs = pd.DataFrame(
        {
            "count": pd.array([1, 2, pd.NA, 4], dtype="Int64"),
            "flag": pd.array([True, False, pd.NA, True], dtype="boolean"),
            "cat": pd.Categorical(["a", "b", "a", "c"]),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)

    path = str(tmp_dir / "round.scx")
    pyscx.from_anndata(adata, path)
    rt = pyscx.open(path).to_anndata()

    assert str(rt.obs["count"].dtype) == "Int64"
    assert str(rt.obs["flag"].dtype) == "boolean"
    assert isinstance(rt.obs["cat"].dtype, pd.CategoricalDtype)
    assert rt.obs.index.name is None
    assert list(rt.obs.index) == list(obs.index)
    # Values (including NA) preserved.
    pd.testing.assert_series_equal(rt.obs["count"], obs["count"], check_names=False)
    pd.testing.assert_series_equal(rt.obs["flag"], obs["flag"], check_names=False)
