"""Metadata tests: index and categorical round-trips (Tasks 15.14–15.15)."""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


def test_obs_var_index(tmp_dir):
    """15.14: Verify DataFrame index set correctly after round-trip."""
    import anndata
    import pyscx

    n_obs, n_vars = 20, 10
    x = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32)
    # Force integer values for bit-exact round-trip
    x.data[:] = np.round(x.data * 100).astype(np.float32)

    obs = pd.DataFrame(
        {"cell_id": [f"cell_{i}" for i in range(n_obs)]},
        index=[f"obs_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"var_{i}" for i in range(n_vars)],
    )

    adata = anndata.AnnData(X=x, obs=obs, var=var)
    path = str(tmp_dir / "index.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    # obs columns should survive
    assert "cell_id" in adata2.obs.columns
    assert list(adata2.obs["cell_id"]) == list(adata.obs["cell_id"])

    # var columns should survive
    assert "gene_id" in adata2.var.columns
    assert list(adata2.var["gene_id"]) == list(adata.var["gene_id"])


def test_categorical_round_trip(tmp_dir):
    """15.15: Categorical columns survive as pd.Categorical."""
    import anndata
    import pyscx

    n_obs, n_vars = 30, 10
    x = sp.random(n_obs, n_vars, density=0.3, format="csr", dtype=np.float32)
    x.data[:] = np.round(x.data * 50).astype(np.float32)

    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(["A", "B", "C"] * 10),
            "sample": pd.Categorical(
                [f"s{i % 5}" for i in range(n_obs)]
            ),
        },
        index=[f"cell_{i}" for i in range(n_obs)],
    )

    adata = anndata.AnnData(X=x, obs=obs)
    path = str(tmp_dir / "categorical.scx")
    pyscx.from_anndata(adata, path)
    adata2 = pyscx.open(path).to_anndata()

    # Check values match
    assert list(adata2.obs["batch"]) == list(adata.obs["batch"])
    assert list(adata2.obs["sample"]) == list(adata.obs["sample"])


def test_experiment_repr(synthetic_adata, tmp_dir):
    """15.16: Experiment repr is AnnData-style (T3.4)."""
    import pyscx

    path = str(tmp_dir / "repr.scx")
    pyscx.from_anndata(synthetic_adata, path)
    exp = pyscx.open(path)

    r = repr(exp)
    # AnnData-style header line; no Rusty `Py` prefix leaking through.
    assert r.startswith("Experiment object with n_obs × n_vars = 100 × 50")
    assert "PyExperiment" not in r
    # Codec/shard internals moved off the repr onto .info().
    assert "codec" in exp.info()


def test_experiment_shape(synthetic_adata, tmp_dir):
    """D1: Experiment.shape mirrors anndata.AnnData.shape == (n_obs, n_vars)."""
    import pyscx

    path = str(tmp_dir / "shape.scx")
    pyscx.from_anndata(synthetic_adata, path)
    exp = pyscx.open(path)

    assert exp.shape == (exp.n_obs, exp.n_vars) == (100, 50)


def test_validate(synthetic_adata, tmp_dir):
    """PyExperiment.validate() returns section results."""
    import pyscx

    path = str(tmp_dir / "validate.scx")
    pyscx.from_anndata(synthetic_adata, path)
    exp = pyscx.open(path)

    results = exp.validate()
    assert isinstance(results, list)
    assert len(results) > 0
    for name, passed in results:
        assert isinstance(name, str)
        assert passed is True


def test_validate_deep(synthetic_adata, tmp_dir):
    """deep=True adds decode-level (canonical-CSR) checks."""
    import pyscx

    path = str(tmp_dir / "validate_deep.scx")
    pyscx.from_anndata(synthetic_adata, path)

    shallow = pyscx.validate(path)
    deep = pyscx.validate(path, deep=True)

    # Deep mode is a superset: the checksum rows plus the new deep rows.
    assert len(deep) > len(shallow)
    deep_names = [name for name, _ in deep]
    assert any(name.startswith("canonical-csr ") for name in deep_names)

    for name, passed in deep:
        assert isinstance(name, str)
        assert passed is True

    # Experiment.validate(deep=True) agrees with the module-level function.
    exp = pyscx.open(path)
    assert exp.validate(deep=True) == deep
