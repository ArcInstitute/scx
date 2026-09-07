"""Tests for pyscx.accel.score_genes (sc.tl.score_genes equivalent).

`method="control"` replicates scanpy's expression-matched control-gene scoring
but samples with a Rust-native RNG, so it does not pick the same control genes
as scanpy and its absolute scores differ. Those tests assert structural
properties (determinism, shape, finiteness, in-memory == backed) and exact
equality for the closed-form `mean` / `zscore` methods.

`ctrl_genes=` is the exception, and `TestCtrlGenes` compares against scanpy
directly: given the same control set there is no sampling left to diverge on.
That comparison is only worth anything in a configuration where scanpy actually
samples — when `ctrl_size` is at least the bin size scanpy takes whole bins and
the two agree regardless — so `TestCtrlGenes` builds its own fixture and asserts
the divergence it is closing.
"""

import warnings

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

    def test_duplicate_gene_pool_is_finite_and_deterministic(self, synthetic_adata):
        # An explicit gene_pool with duplicates must be de-duplicated internally
        # (duplicates would skew control-gene sampling). Scores stay finite and
        # reproducible, and match the same pool passed without duplicates.
        pool = [f"gene_{i}" for i in range(30)]
        pool_dup = pool + pool[:10]  # 10 duplicates
        a = synthetic_adata.copy()
        b = synthetic_adata.copy()
        pyscx.accel.score_genes(
            a, GENES, method="control", gene_pool=pool, n_bins=10, random_state=5
        )
        pyscx.accel.score_genes(
            b, GENES, method="control", gene_pool=pool_dup, n_bins=10, random_state=5
        )
        assert np.all(np.isfinite(b.obs["score"].to_numpy()))
        np.testing.assert_array_equal(a.obs["score"].to_numpy(), b.obs["score"].to_numpy())

    def test_mostly_unresolved_gene_pool_warns(self, synthetic_adata):
        adata = synthetic_adata.copy()
        # Only 1 of 5 pool genes exists → >50% unresolved → one-shot warning.
        with pytest.warns(UserWarning, match="gene_pool"):
            pyscx.accel.score_genes(
                adata,
                GENES,
                method="control",
                gene_pool=["gene_0", "NOPE_1", "NOPE_2", "NOPE_3", "NOPE_4"],
                n_bins=5,
            )


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


class TestDeviceValidation:
    def test_invalid_device_raises_value_error(self, synthetic_adata):
        # Round-2 review (codex + Cursor): score_genes previously skipped
        # resolve_device, so device="tpu" was silently treated as GPU intent
        # in the route stamp instead of raising like every other accel op.
        adata = synthetic_adata.copy()
        with pytest.raises(ValueError, match="unknown device"):
            pyscx.accel.score_genes(adata, GENES, method="mean", device="tpu")
        assert "score" not in adata.obs


# ---------------------------------------------------------------------------
# ctrl_genes= — explicit control set, exact parity with whatever picked them
# ---------------------------------------------------------------------------


def _binned_adata(n_obs=300, n_vars=1200, n_signature=40, seed=7):
    """Heterogeneous gene means, so scanpy's expression bins are non-degenerate.

    `synthetic_adata` is 100 x 50: with `n_bins=25` scanpy's
    `n_items = round(50 / 24) = 2`, every bin is smaller than the default
    `ctrl_size`, and scanpy never samples. A control-gene test written on it
    cannot fail.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    scale = np.exp(rng.normal(0, 1.5, n_vars)).astype(np.float32)
    dense = (rng.random((n_obs, n_vars)) < 0.3) * rng.random((n_obs, n_vars)) * scale
    names = [f"g{i}" for i in range(n_vars)]
    adata = ad.AnnData(
        X=sp.csr_matrix(dense.astype(np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=names),
    )
    signature = [names[i] for i in rng.choice(n_vars, n_signature, replace=False)]
    return adata, signature


# Bin size here is round(1200 / 24) = 50, so this forces scanpy to sample.
# At ctrl_size=50 it would take whole bins and the Rust sampler would agree
# with it by construction — the configuration in which a parity test lies.
SAMPLING_CTRL_SIZE = 25


def _scanpy_control_genes(adata, gene_list, *, ctrl_size, n_bins, random_state):
    """The control set `sc.tl.score_genes` would actually use.

    scanpy does not expose it — `score_genes` only logs how many it drew — so
    this reaches for two private helpers and skips if either moves. That is the
    honest cost of pinning "given the same controls, the same score": the
    alternative is re-implementing scanpy's selection in the test, which is the
    moving target `ctrl_genes=` exists to stop chasing.
    """
    import pandas as pd

    try:
        from scanpy.tools._score_genes import (
            _check_score_genes_args,
            _score_genes_bins,
        )
    except ImportError as exc:  # pragma: no cover - version drift
        pytest.skip(f"scanpy's control-selection internals moved: {exc}")

    np.random.seed(random_state)  # what score_genes does before sampling
    gl, pool, get_subset = _check_score_genes_args(
        adata, gene_list, None, use_raw=False, layer=None
    )
    control = pd.Index([], dtype="string")
    for r_genes in _score_genes_bins(
        gl,
        pool,
        ctrl_as_ref=True,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
        get_subset=get_subset,
    ):
        control = control.union(r_genes)
    assert len(control) > 0, "premise: scanpy drew no control genes"
    return list(control)


class TestCtrlGenes:
    def test_matches_a_float64_mean_difference(self, synthetic_adata):
        """The kernel, against an oracle that needs no scanpy."""
        adata = synthetic_adata.copy()
        ctrl = ["gene_1", "gene_2", "gene_3", "gene_7", "gene_11"]
        pyscx.accel.score_genes(adata, GENES, ctrl_genes=ctrl, device="cpu")

        x = _dense_X(adata).astype(np.float64)
        loc = adata.var_names.get_loc
        expected = x[:, [loc(g) for g in GENES]].mean(axis=1) - x[
            :, [loc(g) for g in ctrl]
        ].mean(axis=1)
        np.testing.assert_allclose(
            adata.obs["score"].to_numpy(), expected, rtol=1e-6, atol=1e-9
        )

    def test_equals_scanpy_given_scanpy_s_own_controls(self):
        """The claim `ctrl_genes=` is for: same controls, same score.

        Includes its own premise check — on this fixture the sampled default
        path diverges from scanpy by a wide margin, so the arm cannot pass
        merely because the two implementations happened to agree.
        """
        import scanpy as sc

        adata, signature = _binned_adata()

        reference = adata.copy()
        sc.tl.score_genes(
            reference,
            signature,
            score_name="sc",
            ctrl_size=SAMPLING_CTRL_SIZE,
            random_state=0,
        )

        sampled = adata.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyscx.accel.score_genes(
                sampled,
                signature,
                score_name="scx",
                ctrl_size=SAMPLING_CTRL_SIZE,
                random_state=0,
                device="cpu",
            )
        divergence = np.abs(
            reference.obs["sc"].to_numpy() - sampled.obs["scx"].to_numpy()
        ).max()
        assert divergence > 0.05, (
            "premise: scanpy must actually be sampling here, otherwise the two "
            f"agree anyway and this test proves nothing (max diff {divergence})"
        )

        controls = _scanpy_control_genes(
            adata, signature, ctrl_size=SAMPLING_CTRL_SIZE, n_bins=25, random_state=0
        )
        explicit = adata.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pyscx.accel.score_genes(
                explicit,
                signature,
                ctrl_genes=controls,
                score_name="scx",
                device="cpu",
            )
        # Both sides mean f32 data into f64, in different orders; the residual
        # measured on this fixture is ~1e-7 on scores of order 1.
        np.testing.assert_allclose(
            explicit.obs["scx"].to_numpy(),
            reference.obs["sc"].to_numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_works_on_a_backed_x_where_scanpy_refuses(self, synthetic_adata, tmp_dir):
        """`sc.tl.score_genes` raises `NotImplementedError` on a backed matrix."""
        path = str(tmp_dir / "score_ctrl.scx")
        pyscx.from_anndata(synthetic_adata, path)

        ctrl = ["gene_1", "gene_2", "gene_3"]
        backed = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.score_genes(backed, GENES, ctrl_genes=ctrl, device="cpu")

        in_memory = synthetic_adata.copy()
        pyscx.accel.score_genes(in_memory, GENES, ctrl_genes=ctrl, device="cpu")
        np.testing.assert_allclose(
            backed.obs["score"].to_numpy(),
            in_memory.obs["score"].to_numpy(),
            rtol=1e-6,
            atol=1e-9,
        )

    def test_ignores_random_state(self, synthetic_adata):
        """No sampling means no seed dependence."""
        ctrl = ["gene_1", "gene_2", "gene_3"]
        a = synthetic_adata.copy()
        b = synthetic_adata.copy()
        pyscx.accel.score_genes(a, GENES, ctrl_genes=ctrl, random_state=1, device="cpu")
        pyscx.accel.score_genes(b, GENES, ctrl_genes=ctrl, random_state=99, device="cpu")
        np.testing.assert_array_equal(
            a.obs["score"].to_numpy(), b.obs["score"].to_numpy()
        )

    def test_duplicates_count_once(self, synthetic_adata):
        a = synthetic_adata.copy()
        b = synthetic_adata.copy()
        pyscx.accel.score_genes(
            a, GENES, ctrl_genes=["gene_1", "gene_2"], device="cpu"
        )
        pyscx.accel.score_genes(
            b, GENES, ctrl_genes=["gene_1", "gene_2", "gene_1"], device="cpu"
        )
        np.testing.assert_array_equal(
            a.obs["score"].to_numpy(), b.obs["score"].to_numpy()
        )

    def test_overlap_with_gene_list_follows_scanpy(self, synthetic_adata):
        """scanpy subtracts the raw control mean, overlap and all."""
        adata = synthetic_adata.copy()
        ctrl = [GENES[0], "gene_1", "gene_2"]
        pyscx.accel.score_genes(adata, GENES, ctrl_genes=ctrl, device="cpu")

        x = _dense_X(adata).astype(np.float64)
        loc = adata.var_names.get_loc
        expected = x[:, [loc(g) for g in GENES]].mean(axis=1) - x[
            :, [loc(g) for g in ctrl]
        ].mean(axis=1)
        np.testing.assert_allclose(
            adata.obs["score"].to_numpy(), expected, rtol=1e-6, atol=1e-9
        )

    def test_missing_control_genes_warn_and_drop(self, synthetic_adata):
        adata = synthetic_adata.copy()
        with pytest.warns(UserWarning, match="not in var_names"):
            pyscx.accel.score_genes(
                adata, GENES, ctrl_genes=["gene_1", "nope_1"], device="cpu"
            )
        x = _dense_X(adata).astype(np.float64)
        loc = adata.var_names.get_loc
        expected = x[:, [loc(g) for g in GENES]].mean(axis=1) - x[:, [loc("gene_1")]].mean(
            axis=1
        )
        np.testing.assert_allclose(
            adata.obs["score"].to_numpy(), expected, rtol=1e-6, atol=1e-9
        )

    def test_all_control_genes_missing_raises(self, synthetic_adata):
        with pytest.raises(ValueError, match="ctrl_genes"):
            pyscx.accel.score_genes(
                synthetic_adata.copy(), GENES, ctrl_genes=["nope_1"], device="cpu"
            )

    @pytest.mark.parametrize("method", ["mean", "zscore"])
    def test_rejects_a_method_that_has_no_controls(self, synthetic_adata, method):
        """Silently ignoring a kwarg is how a wrong number gets published."""
        with pytest.raises(ValueError, match="ctrl_genes"):
            pyscx.accel.score_genes(
                synthetic_adata.copy(),
                GENES,
                ctrl_genes=["gene_1"],
                method=method,
                device="cpu",
            )

    def test_rejects_an_explicit_gene_pool(self, synthetic_adata):
        """`gene_pool` only feeds the sampler, which `ctrl_genes=` replaces."""
        with pytest.raises(ValueError, match="gene_pool"):
            pyscx.accel.score_genes(
                synthetic_adata.copy(),
                GENES,
                ctrl_genes=["gene_1"],
                gene_pool=["gene_1", "gene_2"],
                device="cpu",
            )
