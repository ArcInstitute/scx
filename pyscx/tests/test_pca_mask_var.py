"""Phase 1 §3.1 — PCA `mask_var` / highly_variable masking.

`pyscx.accel.pca` now accepts `mask_var` (a var-column name, a boolean array, or
None → auto-consume `adata.var['highly_variable']`), runs PCA on the selected
columns only, keeps `varm['PCs']` aligned to the full var axis (excluded vars
= 0), and records `uns['pca']['params']`. Works on in-memory, backed, and lazy X.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
import pytest

import pyscx


def _make_adata(n_obs=300, n_vars=100, n_keep=40, seed=0):
    rng = np.random.default_rng(seed)
    X = sp.random(n_obs, n_vars, density=0.3, random_state=seed,
                  format="csr", dtype=np.float32)
    X.data = np.ceil(X.data * 10).astype(np.float32)
    X.eliminate_zeros()
    mask = np.zeros(n_vars, dtype=bool)
    mask[rng.choice(n_vars, size=n_keep, replace=False)] = True
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    return adata, mask


def _sign_align(ref, other):
    out = np.array(other, dtype=np.float64, copy=True)
    for j in range(ref.shape[1]):
        if np.dot(ref[:, j], other[:, j]) < 0:
            out[:, j] = -out[:, j]
    return out


class TestMaskVarInMemory:
    def test_mask_equals_manual_subset(self):
        # PCA(mask_var=mask) must equal PCA on the manually column-subset matrix
        # (same scx solver, same data → identical up to per-PC sign).
        adata, mask = _make_adata()
        n_comps = 10

        a_mask = adata.copy()
        pyscx.accel.pca(a_mask, n_comps=n_comps, mask_var=mask)

        a_sub = adata[:, mask].copy()
        pyscx.accel.pca(a_sub, n_comps=n_comps)

        emb_ref = np.asarray(a_sub.obsm["X_pca"], dtype=np.float64)
        emb_mask = _sign_align(emb_ref, np.asarray(a_mask.obsm["X_pca"]))
        np.testing.assert_allclose(emb_mask, emb_ref, atol=1e-4)

        np.testing.assert_allclose(
            np.asarray(a_mask.uns["pca"]["variance_ratio"]),
            np.asarray(a_sub.uns["pca"]["variance_ratio"]),
            rtol=1e-4, atol=1e-6,
        )

    def test_varm_full_axis_with_zeros(self):
        adata, mask = _make_adata()
        n_comps = 10
        pyscx.accel.pca(adata, n_comps=n_comps, mask_var=mask)
        pcs = np.asarray(adata.varm["PCs"])
        assert pcs.shape == (adata.n_vars, n_comps)
        # Excluded vars are zero-filled; selected vars are (generally) not.
        assert np.allclose(pcs[~mask], 0.0)
        assert not np.allclose(pcs[mask], 0.0)

    def test_varm_masked_rows_match_subset(self):
        adata, mask = _make_adata()
        n_comps = 10
        a_mask = adata.copy()
        pyscx.accel.pca(a_mask, n_comps=n_comps, mask_var=mask)
        a_sub = adata[:, mask].copy()
        pyscx.accel.pca(a_sub, n_comps=n_comps)

        pcs_full = np.asarray(a_mask.varm["PCs"], dtype=np.float64)[mask]
        pcs_sub = np.asarray(a_sub.varm["PCs"], dtype=np.float64)
        # Sign-align by embedding sign is hard here; align PCs columns directly.
        pcs_aligned = _sign_align(pcs_sub, pcs_full)
        np.testing.assert_allclose(pcs_aligned, pcs_sub, atol=1e-4)

    def test_params_recorded(self):
        adata, mask = _make_adata()
        adata.var["highly_variable"] = mask
        pyscx.accel.pca(adata, n_comps=7, mask_var="highly_variable")
        params = adata.uns["pca"]["params"]
        assert bool(params["use_highly_variable"]) is True
        assert params["mask_var"] == "highly_variable"
        assert int(params["n_comps"]) == 7
        assert bool(params["zero_center"]) is True

    def test_auto_consumes_highly_variable(self):
        # mask_var=None auto-uses var['highly_variable'] → same as explicit mask.
        adata, mask = _make_adata()
        adata.var["highly_variable"] = mask

        a_auto = adata.copy()
        pyscx.accel.pca(a_auto, n_comps=8)  # no mask_var

        a_expl = adata.copy()
        pyscx.accel.pca(a_expl, n_comps=8, mask_var=mask)

        emb_ref = np.asarray(a_expl.obsm["X_pca"], dtype=np.float64)
        emb_auto = _sign_align(emb_ref, np.asarray(a_auto.obsm["X_pca"]))
        np.testing.assert_allclose(emb_auto, emb_ref, atol=1e-4)
        assert bool(a_auto.uns["pca"]["params"]["use_highly_variable"]) is True

    def test_no_mask_when_no_highly_variable(self):
        # Without a highly_variable column and no mask_var, PCA uses all genes.
        adata, _ = _make_adata()
        pyscx.accel.pca(adata, n_comps=10)
        pcs = np.asarray(adata.varm["PCs"])
        assert pcs.shape == (adata.n_vars, 10)
        # use_highly_variable recorded False.
        assert bool(adata.uns["pca"]["params"]["use_highly_variable"]) is False

    def test_scanpy_variance_ratio_parity(self):
        # Sanity: masked PCA computes real PCA — variance_ratio close to scanpy
        # run on the pre-subset matrix.
        sc = pytest.importorskip("scanpy")
        adata, mask = _make_adata()
        n_comps = 10
        pyscx.accel.pca(adata, n_comps=n_comps, mask_var=mask, zero_center=True)

        ref = adata[:, mask].copy()
        ref.X = ref.X.astype(np.float32)
        sc.pp.pca(ref, n_comps=n_comps, zero_center=True)
        np.testing.assert_allclose(
            np.asarray(adata.uns["pca"]["variance_ratio"]),
            np.asarray(ref.uns["pca"]["variance_ratio"]),
            rtol=0.05, atol=1e-3,
        )


class TestMaskVarErrors:
    def test_wrong_length_mask(self):
        adata, _ = _make_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pca(adata, mask_var=np.ones(adata.n_vars + 3, dtype=bool))

    def test_all_false_mask(self):
        adata, _ = _make_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pca(adata, mask_var=np.zeros(adata.n_vars, dtype=bool))

    def test_unknown_mask_column(self):
        adata, _ = _make_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pca(adata, mask_var="not_a_column")


class TestMaskVarBacked:
    def test_backed_mask_equals_subset(self):
        adata, mask = _make_adata()
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "t.scx")
        pyscx.from_anndata(adata, path)
        n_comps = 10

        a_backed = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.pca(a_backed, n_comps=n_comps, mask_var=mask)

        a_sub = adata[:, mask].copy()
        pyscx.accel.pca(a_sub, n_comps=n_comps)

        emb_ref = np.asarray(a_sub.obsm["X_pca"], dtype=np.float64)
        emb_backed = _sign_align(emb_ref, np.asarray(a_backed.obsm["X_pca"]))
        np.testing.assert_allclose(emb_backed, emb_ref, atol=1e-3)

        pcs = np.asarray(a_backed.varm["PCs"])
        assert pcs.shape == (adata.n_vars, n_comps)
        assert np.allclose(pcs[~mask], 0.0)
