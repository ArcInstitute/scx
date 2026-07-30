"""Phase 1 §3.2 — DE `use_raw` / `layer` data-selection contract.

`rank_genes_groups` and `pdex_ref` now select the analyzed matrix like scanpy:
`use_raw` → `adata.raw.X` (with `adata.raw.var` names), `layer` → a named layer,
mutually exclusive, `use_raw=None` auto-resolves. Previously both always analyzed
`adata.X` and recorded `use_raw=False`.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata
import pytest

import pyscx


def _adata_with_raw_and_layer():
    """AnnData whose X (log-normalized) differs from raw.X / layer['counts']."""
    import scanpy as sc

    n_obs, n_vars = 200, 30
    counts = sp.random(n_obs, n_vars, density=0.4, random_state=1,
                       format="csr", dtype=np.float32)
    counts.data = np.ceil(counts.data * 20).astype(np.float32)
    counts.eliminate_zeros()

    groups = ["A"] * (n_obs // 2) + ["B"] * (n_obs - n_obs // 2)
    obs = pd.DataFrame({"grp": groups}, index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])

    adata = anndata.AnnData(X=counts.copy(), obs=obs, var=var)
    adata.layers["counts"] = counts.copy()
    # Freeze raw counts from a SEPARATE AnnData so the in-place normalize below
    # cannot alias/mutate it (`adata.raw = adata` shares the X buffer).
    adata.raw = anndata.AnnData(X=counts.copy(), obs=obs.copy(), var=var.copy())
    # Process X so it differs from raw / the counts layer.
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _rgg_scores(adata):
    """Flatten uns['rank_genes_groups'] names→scores across all groups."""
    rgg = adata.uns["rank_genes_groups"]
    names, scores = rgg["names"], rgg["scores"]
    out = {}
    for grp in names.dtype.names:
        for nm, sc_ in zip(names[grp], scores[grp]):
            out[(grp, str(nm))] = float(sc_)
    return out


class TestPdexRefUseRawLayer:
    def test_use_raw_equals_layer_of_same_matrix(self):
        # raw.X and layers['counts'] hold the identical matrix → identical DE.
        ad = _adata_with_raw_and_layer()
        df_raw = pyscx.accel.pdex_ref(ad, "grp", reference="A", use_raw=True,
                                      is_log1p=False, output="pandas")
        df_layer = pyscx.accel.pdex_ref(ad, "grp", reference="A", layer="counts",
                                        is_log1p=False, output="pandas")
        pd.testing.assert_frame_equal(
            df_raw.reset_index(drop=True), df_layer.reset_index(drop=True)
        )

    def test_use_raw_differs_from_x(self):
        ad = _adata_with_raw_and_layer()
        df_x = pyscx.accel.pdex_ref(ad, "grp", reference="A", output="pandas")
        df_raw = pyscx.accel.pdex_ref(ad, "grp", reference="A", use_raw=True,
                                      is_log1p=False, output="pandas")
        # Some fold-change / p-value column must differ (X is log-space, raw is counts).
        num_cols = [c for c in df_x.columns if df_x[c].dtype.kind == "f"]
        assert any(
            not np.allclose(df_x[c].to_numpy(), df_raw[c].to_numpy(), equal_nan=True)
            for c in num_cols
        ), "use_raw=True did not change pdex_ref results vs X"

    def test_metadata_records_use_raw_and_layer(self):
        ad = _adata_with_raw_and_layer()
        pyscx.accel.pdex_ref(ad, "grp", reference="A", use_raw=True,
                             is_log1p=False, output="pandas")
        entry = ad.uns["scx_accel"]["pdex_ref"]
        assert entry["use_raw"] is True
        assert entry["layer"] is None

        ad2 = _adata_with_raw_and_layer()
        pyscx.accel.pdex_ref(ad2, "grp", reference="A", layer="counts",
                             is_log1p=False, output="pandas")
        entry2 = ad2.uns["scx_accel"]["pdex_ref"]
        assert entry2["use_raw"] is False
        assert entry2["layer"] == "counts"


class TestRankGenesGroupsUseRawLayer:
    def test_use_raw_equals_layer(self):
        ad = _adata_with_raw_and_layer()
        pyscx.accel.rank_genes_groups(ad, "grp", use_raw=True)
        raw_scores = _rgg_scores(ad)

        ad2 = _adata_with_raw_and_layer()
        pyscx.accel.rank_genes_groups(ad2, "grp", layer="counts")
        layer_scores = _rgg_scores(ad2)

        assert set(raw_scores) == set(layer_scores)
        for k in raw_scores:
            assert raw_scores[k] == pytest.approx(layer_scores[k], abs=1e-6)

    def test_params_record_use_raw(self):
        ad = _adata_with_raw_and_layer()
        pyscx.accel.rank_genes_groups(ad, "grp", use_raw=True)
        params = ad.uns["rank_genes_groups"]["params"]
        assert bool(params["use_raw"]) is True

        ad2 = _adata_with_raw_and_layer()
        pyscx.accel.rank_genes_groups(ad2, "grp", layer="counts")
        params2 = ad2.uns["rank_genes_groups"]["params"]
        assert bool(params2["use_raw"]) is False
        assert params2["layer"] == "counts"

    def test_default_uses_raw_when_present(self):
        # scanpy semantics: use_raw=None resolves to True when adata.raw exists.
        ad = _adata_with_raw_and_layer()
        pyscx.accel.rank_genes_groups(ad, "grp")  # no use_raw / layer
        assert bool(ad.uns["rank_genes_groups"]["params"]["use_raw"]) is True


class TestUseRawLayerErrors:
    def test_use_raw_and_layer_mutually_exclusive(self):
        ad = _adata_with_raw_and_layer()
        with pytest.raises(ValueError):
            pyscx.accel.rank_genes_groups(ad, "grp", use_raw=True, layer="counts")
        with pytest.raises(ValueError):
            pyscx.accel.pdex_ref(ad, "grp", reference="A", use_raw=True,
                                 layer="counts", output="pandas")

    def test_use_raw_true_without_raw_errors(self):
        n_obs, n_vars = 50, 10
        counts = sp.random(n_obs, n_vars, density=0.4, random_state=2,
                           format="csr", dtype=np.float32)
        obs = pd.DataFrame({"grp": ["A"] * 25 + ["B"] * 25},
                           index=[f"c{i}" for i in range(n_obs)])
        var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
        ad = anndata.AnnData(X=counts, obs=obs, var=var)  # no .raw
        with pytest.raises(ValueError):
            pyscx.accel.rank_genes_groups(ad, "grp", use_raw=True)

    def test_unknown_layer_errors(self):
        ad = _adata_with_raw_and_layer()
        with pytest.raises(ValueError):
            pyscx.accel.rank_genes_groups(ad, "grp", layer="does_not_exist")
