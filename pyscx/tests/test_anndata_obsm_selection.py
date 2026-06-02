"""Tests for selective obsm loading via ``PyExperiment.to_anndata(obsm=...)``.

Covers the WIRE-SHIM.md step-0 contract: ``None`` loads all keys
(byte-identical to prior behaviour), ``[]`` loads none, a list loads only the
requested keys (unknown key → ``KeyError``), and the §3.5.1 query-path rules
(non-empty selection under ``obs_filter`` + ``preserve_slots=False`` →
``ValueError``; ``[]`` suppresses the dropped-obsm warning).
"""

import warnings

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import pyscx


@pytest.fixture
def multi_obsm_adata():
    """AnnData with three obsm keys and a cell_type column for filtering."""
    rng = np.random.default_rng(7)
    n_obs, n_vars = 80, 30

    dense = rng.integers(0, 50, size=(n_obs, n_vars)).astype(np.float32)
    dense[rng.random((n_obs, n_vars)) > 0.3] = 0
    x = sp.csr_matrix(dense)

    cell_types = (["A"] * 40) + (["B"] * 40)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(cell_types)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_id": [f"gene_{i}" for i in range(n_vars)]},
        index=[f"gene_{i}" for i in range(n_vars)],
    )
    obsm = {
        "X_state": rng.standard_normal((n_obs, 8)).astype(np.float32),
        "X_hvg": rng.standard_normal((n_obs, 5)).astype(np.float32),
        "X_mse": rng.standard_normal((n_obs, 4)).astype(np.float32),
    }
    return anndata.AnnData(X=x, obs=obs, var=var, obsm=obsm)


@pytest.fixture
def scx_path(multi_obsm_adata, tmp_dir):
    path = str(tmp_dir / "multi_obsm.scx")
    pyscx.from_anndata(multi_obsm_adata, path)
    return path


# --- eager (no filter) path -------------------------------------------------


def test_obsm_none_loads_all_keys(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata()
    assert set(adata.obsm.keys()) == {"X_state", "X_hvg", "X_mse"}


def test_obsm_selection_loads_only_requested(scx_path, multi_obsm_adata):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(obsm=["X_state"])
    assert set(adata.obsm.keys()) == {"X_state"}
    # value parity with the full load
    full = exp.to_anndata().obsm["X_state"]
    np.testing.assert_allclose(adata.obsm["X_state"], full)
    np.testing.assert_allclose(adata.obsm["X_state"], multi_obsm_adata.obsm["X_state"])


def test_obsm_empty_loads_none(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(obsm=[])
    assert len(adata.obsm) == 0


def test_obsm_unknown_key_raises_keyerror(scx_path):
    exp = pyscx.open(scx_path)
    with pytest.raises(KeyError):
        exp.to_anndata(obsm=["does_not_exist"])


def test_obsm_unknown_keys_listed_in_keyerror(scx_path):
    # Plural message (WIRE-SHIM §3.3 shape): all missing keys reported at once.
    exp = pyscx.open(scx_path)
    with pytest.raises(KeyError) as excinfo:
        exp.to_anndata(obsm=["nope_a", "X_state", "nope_b"])
    msg = str(excinfo.value)
    assert "nope_a" in msg and "nope_b" in msg


def test_obsm_duplicate_keys_deduplicated(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(obsm=["X_state", "X_state"])
    assert set(adata.obsm.keys()) == {"X_state"}


def test_obsm_multiple_keys(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(obsm=["X_state", "X_mse"])
    assert set(adata.obsm.keys()) == {"X_state", "X_mse"}


# --- backed path ------------------------------------------------------------


def test_obsm_selection_backed(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(backed=True, obsm=["X_state"])
    assert set(adata.obsm.keys()) == {"X_state"}


def test_obsm_empty_backed(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(backed=True, obsm=[])
    assert len(adata.obsm) == 0


def test_obsm_selection_backed_with_filter(scx_path):
    # Selection composes with backed obs_filter row slicing (exercises the
    # binary-search positions path in to_anndata_backed_with_options).
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(
        backed=True, obs_filter="cell_type == 'A'", obsm=["X_state"]
    )
    assert set(adata.obsm.keys()) == {"X_state"}
    assert adata.n_obs == 40
    assert adata.obsm["X_state"].shape == (40, 8)


def test_modality_with_obsm_raises(scx_path):
    # The modality guard rejects any obsm= (including []) before resolving the
    # modality, so a single-modality file still triggers the ValueError.
    exp = pyscx.open(scx_path)
    with pytest.raises(ValueError, match="selective obsm"):
        exp.to_anndata(modality="rna", backed=True, obsm=[])


# --- query-engine path (obs_filter + preserve_slots=False), §3.5.1 ----------


def test_query_path_nonempty_obsm_raises_valueerror(scx_path):
    exp = pyscx.open(scx_path)
    with pytest.raises(ValueError, match="query engine"):
        exp.to_anndata(obs_filter="cell_type == 'A'", preserve_slots=False, obsm=["X_state"])


def test_query_path_empty_obsm_no_warning(scx_path):
    exp = pyscx.open(scx_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adata = exp.to_anndata(obs_filter="cell_type == 'A'", preserve_slots=False, obsm=[])
    assert len(adata.obsm) == 0
    assert not any("obsm" in str(w.message) for w in caught)


def test_query_path_obsm_none_warns(scx_path):
    exp = pyscx.open(scx_path)
    with pytest.warns(UserWarning, match="obsm"):
        exp.to_anndata(obs_filter="cell_type == 'A'", preserve_slots=False)


# --- preserve_slots=True path -----------------------------------------------


def test_preserve_slots_honours_obsm_selection(scx_path):
    exp = pyscx.open(scx_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adata = exp.to_anndata(
            obs_filter="cell_type == 'A'", preserve_slots=True, obsm=["X_state"]
        )
    assert set(adata.obsm.keys()) == {"X_state"}
    assert adata.n_obs == 40


# --- var_names projection path ----------------------------------------------


def test_var_names_honours_obsm_selection(scx_path):
    exp = pyscx.open(scx_path)
    adata = exp.to_anndata(var_names=["gene_0", "gene_1"], obsm=["X_hvg"])
    assert set(adata.obsm.keys()) == {"X_hvg"}
    assert adata.n_vars == 2
