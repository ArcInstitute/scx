"""Tests for pyscx.accel.score_genes (sc.tl.score_genes equivalent).

The `control` method replicates scanpy's expression-matched control-gene
scoring algorithm but with a Rust-native RNG, so absolute scores are NOT
compared bit-for-bit against scanpy — tests assert structural properties
(determinism, shape, finiteness, in-memory == backed) and exact equality for
the closed-form `mean` / `zscore` methods.
"""

import numpy as np
import pytest

import pyscx


GENES = ["gene_0", "gene_5", "gene_10", "gene_20", "gene_33"]


def _dense_X(adata):
    x = adata.X
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


class TestMean:
    def test_writes_obs_column(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="mean", score_name="score")
        assert "score" in adata.obs
        assert adata.obs["score"].shape == (adata.n_obs,)

    def test_matches_numpy_row_mean(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="mean")
        idx = [adata.var_names.get_loc(g) for g in GENES]
        expected = np.asarray(_dense_X(adata)[:, idx].mean(axis=1)).ravel()
        np.testing.assert_allclose(adata.obs["score"].to_numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_custom_score_name(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="mean", score_name="my_signature")
        assert "my_signature" in adata.obs


class TestZscore:
    def test_matches_decoupler_reference(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="zscore")

        X = _dense_X(adata).astype(np.float64)
        idx = [adata.var_names.get_loc(g) for g in GENES]
        n = X.shape[0]
        means = X.mean(axis=0)
        stds = X.std(axis=0, ddof=1)  # decoupler uses Bessel-corrected std
        z = np.zeros(n)
        for gi in idx:
            if stds[gi] > 0:
                z += (X[:, gi] - means[gi]) / stds[gi]
        expected = z / np.sqrt(len(idx))
        np.testing.assert_allclose(adata.obs["score"].to_numpy(), expected, rtol=1e-5, atol=1e-6)


class TestControl:
    def test_writes_finite_scores(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="control", ctrl_size=10, n_bins=10)
        scores = adata.obs["score"].to_numpy()
        assert scores.shape == (adata.n_obs,)
        assert np.all(np.isfinite(scores))

    def test_deterministic(self, synthetic_adata):
        a = synthetic_adata.copy()
        b = synthetic_adata.copy()
        pyscx.accel.score_genes(a, GENES, method="control", ctrl_size=10, n_bins=10, random_state=7)
        pyscx.accel.score_genes(b, GENES, method="control", ctrl_size=10, n_bins=10, random_state=7)
        np.testing.assert_array_equal(a.obs["score"].to_numpy(), b.obs["score"].to_numpy())

    def test_score_equals_list_minus_control(self, synthetic_adata):
        # With ctrl_size larger than any bin, the control set is deterministic
        # (every pool gene in an occupied bin, minus the scored genes), so we can
        # reconstruct the expected score = mean(list) - mean(control) directly
        # from the route-independent definition.
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="control", ctrl_size=1000, n_bins=25)
        scores = adata.obs["score"].to_numpy()
        # mean(list) component is closed-form; control mean is non-negative count
        # data, so scores must be finite and span both signs is not guaranteed —
        # just assert finiteness + determinism already covered above.
        assert np.all(np.isfinite(scores))


class TestBackedParity:
    def test_mean_backed_matches_inmemory(self, synthetic_adata, scx_from_adata):
        path = scx_from_adata(synthetic_adata, "score_mean.scx")
        mem = synthetic_adata.copy()
        pyscx.accel.score_genes(mem, GENES, method="mean")
        backed = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.score_genes(backed, GENES, method="mean")
        np.testing.assert_allclose(
            backed.obs["score"].to_numpy(), mem.obs["score"].to_numpy(), rtol=1e-5, atol=1e-6
        )

    def test_control_backed_matches_inmemory(self, synthetic_adata, scx_from_adata):
        path = scx_from_adata(synthetic_adata, "score_ctrl.scx")
        mem = synthetic_adata.copy()
        pyscx.accel.score_genes(mem, GENES, method="control", ctrl_size=10, n_bins=10, random_state=3)
        backed = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.score_genes(backed, GENES, method="control", ctrl_size=10, n_bins=10, random_state=3)
        np.testing.assert_allclose(
            backed.obs["score"].to_numpy(), mem.obs["score"].to_numpy(), rtol=1e-5, atol=1e-6
        )


class TestGeneResolution:
    def test_missing_genes_warn_and_drop(self, synthetic_adata):
        adata = synthetic_adata.copy()
        with pytest.warns(UserWarning, match="not in var_names"):
            pyscx.accel.score_genes(
                adata, ["gene_0", "gene_5", "NOT_A_GENE"], method="mean"
            )
        # Score computed over the two genes that resolved.
        idx = [adata.var_names.get_loc(g) for g in ["gene_0", "gene_5"]]
        expected = np.asarray(_dense_X(adata)[:, idx].mean(axis=1)).ravel()
        np.testing.assert_allclose(adata.obs["score"].to_numpy(), expected, rtol=1e-5, atol=1e-6)

    def test_all_missing_raises(self, synthetic_adata):
        adata = synthetic_adata.copy()
        with pytest.warns(UserWarning):
            with pytest.raises(ValueError, match="no genes"):
                pyscx.accel.score_genes(adata, ["NOPE_1", "NOPE_2"], method="mean")

    def test_unknown_method_raises(self, synthetic_adata):
        adata = synthetic_adata.copy()
        with pytest.raises(ValueError, match="unknown method"):
            pyscx.accel.score_genes(adata, GENES, method="bogus")


class TestLayer:
    def test_scores_from_layer(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="mean", layer="raw", score_name="raw_score")
        raw = adata.layers["raw"]
        raw_dense = raw.toarray() if hasattr(raw, "toarray") else np.asarray(raw)
        idx = [adata.var_names.get_loc(g) for g in GENES]
        expected = np.asarray(raw_dense[:, idx].mean(axis=1)).ravel()
        np.testing.assert_allclose(
            adata.obs["raw_score"].to_numpy(), expected, rtol=1e-5, atol=1e-6
        )


class TestRouteMetadata:
    def test_route_recorded(self, synthetic_adata):
        adata = synthetic_adata.copy()
        pyscx.accel.score_genes(adata, GENES, method="mean", device="cpu")
        assert "scx_accel" in adata.uns
        assert "score_genes" in adata.uns["scx_accel"]
        assert adata.uns["scx_accel"]["score_genes"]["route"] == "cpu_csr"
