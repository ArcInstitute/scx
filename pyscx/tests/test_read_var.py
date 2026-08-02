"""`Experiment.read_var()` — the var-axis mirror of `read_obs()`.

`read_obs` has existed for a long time; `read_var` did not. A user who wanted
just the gene table — to map Ensembl ids to `feature_name` when labelling
marker genes, say — had to call `to_anndata()` and materialize the whole
matrix to get at it. On a 500k-cell Census file that is a 1.4 s / 9.9 GB
detour for a 5 MB table (user-report F2).

These tests pin the symmetry with `read_obs`: same index, same projection
behaviour, X never touched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx

N_OBS, N_VARS = 40, 12


@pytest.fixture(scope="module")
def scx_path(tmp_path_factory):
    import anndata as ad

    rng = np.random.default_rng(5)
    adata = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, (N_OBS, N_VARS)).astype("float32")))
    adata.obs_names = [f"c{i}" for i in range(N_OBS)]
    adata.var_names = [f"ENSG{i:05d}" for i in range(N_VARS)]
    adata.var["feature_name"] = [f"GENE{i}" for i in range(N_VARS)]
    adata.var["feature_type"] = pd.Categorical(["protein_coding"] * N_VARS)
    adata.var["length"] = np.arange(N_VARS, dtype=np.int64)
    path = tmp_path_factory.mktemp("read_var") / "v.scx"
    pyscx.from_anndata(adata, str(path))
    return path


def test_read_var_matches_to_anndata_var(scx_path):
    """The whole point: same frame, without materializing X."""
    exp = pyscx.open(str(scx_path))
    var = exp.read_var()
    reference = exp.to_anndata().var

    assert list(var.index) == list(reference.index)
    assert set(var.columns) == set(reference.columns)
    for col in reference.columns:
        pd.testing.assert_series_equal(
            var[col], reference[col], check_dtype=False, check_categorical=False
        )


def test_read_var_index_is_the_gene_names(scx_path):
    var = pyscx.open(str(scx_path)).read_var()
    assert list(var.index) == [f"ENSG{i:05d}" for i in range(N_VARS)]
    assert len(var) == N_VARS


def test_read_var_columns_projects_and_keeps_the_index(scx_path):
    """Same contract as `read_obs(columns=...)`: the index survives."""
    var = pyscx.open(str(scx_path)).read_var(columns=["feature_name"])
    assert list(var.columns) == ["feature_name"]
    assert list(var.index) == [f"ENSG{i:05d}" for i in range(N_VARS)]
    assert list(var["feature_name"]) == [f"GENE{i}" for i in range(N_VARS)]


def test_read_var_columns_preserves_requested_order(scx_path):
    var = pyscx.open(str(scx_path)).read_var(columns=["length", "feature_name"])
    assert list(var.columns) == ["length", "feature_name"]


def test_read_var_unknown_column_names_what_is_available(scx_path):
    """A bare pyarrow KeyError would not say what the user could have asked for."""
    with pytest.raises(KeyError, match="feature_name"):
        pyscx.open(str(scx_path)).read_var(columns=["not_a_column"])


def test_read_var_agrees_with_var_keys(scx_path):
    """`columns` takes physical names — the ones `var_keys()` reports."""
    exp = pyscx.open(str(scx_path))
    keys = exp.var_keys()
    var = exp.read_var(columns=keys)
    assert set(var.columns) == set(keys)


def test_read_var_does_not_materialize_x(scx_path):
    """`read_var` must stay an obs/var-metadata read.

    Asserted behaviourally: reading var off a *backed* handle must leave `X`
    a lazy handle. If `read_var` ever routed through `to_anndata()` this flips.
    """
    exp = pyscx.open(str(scx_path))
    adata = exp.to_anndata(backed=True)
    exp.read_var()
    assert type(adata.X).__name__ == "ScxBackedSparseDataset"


def test_read_var_unknown_modality_raises(scx_path):
    with pytest.raises(KeyError, match="unknown modality"):
        pyscx.open(str(scx_path)).read_var(modality="nope")
