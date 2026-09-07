"""Tests for pyscx.accel.pflog (PFlog v4 / shifted-log, Booeshaghi et al.).

v4 shifts **raw counts** by the matrix-wide Anscombe pseudocount ``1/(4α)`` (no
per-cell depth). `α` is estimated once from the matrix (`alpha=None`) or pinned
by the caller. The binding writes the per-cell baseline to `adata.obs`, stamps
the fit into `adata.uns["pflog"]`, writes an out-of-core PCA embedding to
`adata.obsm`, and (size-guarded) a dense layer. The shared `reference_dense`
helper is the source of truth; assertions hold to ~1e-5 (the f32 `delta` floor),
with PCA at a looser tolerance (randomized SVD on f32).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import anndata

import pyscx

# Deterministic pinned α for the equivalence/PCA/stream tests (α = 0.5 →
# pseudocount 1/(4α) = 0.5). `alpha=None` (estimation) is covered separately.
ALPHA = 0.5


def reference_dense(X, alpha):
    """v4 exact reference: z_ij = log(x_ij + 1/(4α)) − mean_j log(x_ik + 1/(4α)).

    Equivalent to `log1p(4α·x) − rowmean` (the folded `log(4α)` cancels under
    centering), which is exactly `delta + baseline` from the Rust kernel.
    """
    X = np.asarray(X, dtype=np.float64)
    pc = 1.0 / (4.0 * alpha)
    L = np.log(X + pc)
    return L - L.mean(axis=1, keepdims=True)


# The estimator's numerics are pinned Rust-side against counts simulated from a
# known dispersion (`scx-accel/src/pflog_reference_tests.rs`). A Python
# re-implementation of the MoM median used to live here and assert equality with
# it; it carried the same `var > mean` pre-filter as the code under test, so all
# it could confirm was that two copies of one bug agreed (§7.14). What is left
# below is the binding's job: that `alpha=None` routes to estimation and that the
# stamped `uns["pflog"]` fields are internally consistent.
MU_MIN = 1e-3


def small_adata():
    """6×4 count matrix with no empty rows; named vars."""
    X = np.array(
        [
            [0, 1, 3, 0],
            [2, 0, 0, 5],
            [1, 1, 1, 1],
            [4, 2, 0, 1],
            [0, 0, 6, 2],
            [3, 3, 1, 0],
        ],
        dtype=np.float32,
    )
    ad = anndata.AnnData(sp.csr_matrix(X))
    ad.var_names = [f"gene_{i}" for i in range(X.shape[1])]
    ad.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    return ad, X


def underdispersed_gene_adata():
    """6×3 counts whose first gene is constant, so `var == 0 < mean`.

    `small_adata`'s four genes are all over-dispersed, which makes any assertion
    about the size of the α pool pass whether or not the pre-fix `var > mean`
    filter is present. This fixture is the discriminating one: the pool is 3
    genes, the filtered pool would be 2, and the two give different α
    (0.379 vs 0.525).
    """
    X = np.array(
        [[2, 0, 4], [2, 5, 0], [2, 1, 7], [2, 3, 1], [2, 0, 5], [2, 6, 2]],
        dtype=np.float32,
    )
    ad = anndata.AnnData(sp.csr_matrix(X))
    ad.var_names = [f"gene_{i}" for i in range(X.shape[1])]
    ad.obs_names = [f"cell_{i}" for i in range(X.shape[0])]
    return ad, X


class TestBaseline:
    def test_writes_obs_baseline(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="baseline")
        assert "pflog_baseline" in ad.obs
        assert ad.obs["pflog_baseline"].shape == (ad.n_obs,)

    def test_custom_baseline_key(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="baseline", baseline_key="b")
        assert "b" in ad.obs


class TestDenseEquivalence:
    def test_matches_reference_default(self):
        ad, X = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="dense", layer_out="Z")
        Z = np.asarray(ad.layers["Z"])
        np.testing.assert_allclose(Z, reference_dense(X, ALPHA), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 4.0])
    def test_matches_reference_various_alpha(self, alpha):
        ad, X = small_adata()
        pyscx.accel.pflog(ad, alpha=alpha, store="dense", layer_out="Z")
        Z = np.asarray(ad.layers["Z"])
        np.testing.assert_allclose(Z, reference_dense(X, alpha), rtol=1e-5, atol=1e-5)

    def test_row_sums_are_zero(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="dense", layer_out="Z")
        Z = np.asarray(ad.layers["Z"])
        np.testing.assert_allclose(Z.sum(axis=1), 0.0, atol=1e-5)

    def test_dense_default_layer_name(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="dense")
        assert "pflog" in ad.layers


class TestAlphaEstimation:
    def test_alpha_none_estimates_and_stamps_uns(self):
        ad, X = small_adata()
        pyscx.accel.pflog(ad, store="baseline")  # alpha=None → estimate
        meta = ad.uns["pflog"]
        assert meta["version"] == "v4"
        assert meta["alpha_source"] == "estimated"
        assert not bool(meta["fell_back"])

        # `n_genes_used` is the size of the median pool, and the pool is every
        # gene with a mean above `mu_min` — nothing else is excluded. Counting
        # that here is a threshold on a column mean, not a second copy of the
        # estimator. On THIS fixture every gene is over-dispersed, so the
        # assertion holds with or without a dispersion filter; the discriminating
        # case is `test_pool_includes_underdispersed_genes` below.
        n_with_a_mean = int((np.asarray(X, dtype=np.float64).mean(axis=0) > MU_MIN).sum())
        assert n_with_a_mean > 0, "fixture has no gene above mu_min"
        assert int(meta["n_genes_used"]) == n_with_a_mean

        alpha = float(meta["alpha"])
        assert alpha > 0.0 and np.isfinite(alpha)
        np.testing.assert_allclose(
            float(meta["pseudocount"]), 1.0 / (4.0 * alpha), rtol=1e-12
        )

    def test_pool_includes_underdispersed_genes(self):
        """A constant gene has a mean, so it belongs in the α pool (§7.14).

        With the pre-fix `var > mean` filter this reports 2 pooled genes and a
        different α; the filter kept only the upper tail of the dispersion
        distribution and biased the estimate high.
        """
        ad, X = underdispersed_gene_adata()
        pyscx.accel.pflog(ad, store="baseline")  # alpha=None → estimate
        meta = ad.uns["pflog"]
        assert meta["alpha_source"] == "estimated"
        assert not bool(meta["fell_back"])
        assert int(meta["n_genes_used"]) == 3, (
            "the constant gene was dropped from the pool before the median"
        )
        alpha = float(meta["alpha"])
        assert alpha > 0.0 and np.isfinite(alpha)
        np.testing.assert_allclose(
            float(meta["pseudocount"]), 1.0 / (4.0 * alpha), rtol=1e-12
        )

    def test_pinned_alpha_stamps_source(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="baseline")
        meta = ad.uns["pflog"]
        assert meta["version"] == "v4"
        assert meta["alpha_source"] == "pinned"
        np.testing.assert_allclose(float(meta["alpha"]), ALPHA)
        np.testing.assert_allclose(float(meta["pseudocount"]), 1.0 / (4.0 * ALPHA))


class TestPCA:
    def test_writes_obsm_and_singular_values(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", n_components=3)
        assert ad.obsm["X_pflog_pca"].shape == (ad.n_obs, 3)
        assert "X_pflog_pca_singular_values" in ad.uns

    def test_singular_values_match_dense_svd(self):
        ad, X = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", n_components=4, n_power_iterations=4)
        ref = reference_dense(X, ALPHA)
        ref_c = ref - ref.mean(axis=0, keepdims=True)
        sv_ref = np.linalg.svd(ref_c, compute_uv=False)
        sv = np.asarray(ad.uns["X_pflog_pca_singular_values"])
        np.testing.assert_allclose(sv, sv_ref[: len(sv)], rtol=1e-3, atol=1e-4)

    def test_embedding_column_norms_equal_singular_values(self):
        # embeddings = U·Σ with U orthonormal → column 2-norms are the σ_i.
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", n_components=4, n_power_iterations=4)
        emb = np.asarray(ad.obsm["X_pflog_pca"])
        sv = np.asarray(ad.uns["X_pflog_pca_singular_values"])
        np.testing.assert_allclose(np.linalg.norm(emb, axis=0), sv, rtol=1e-3, atol=1e-4)

    def test_store_all_writes_everything(self):
        ad, X = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="all", n_components=3, layer_out="Z")
        assert "pflog_baseline" in ad.obs
        assert ad.obsm["X_pflog_pca"].shape == (ad.n_obs, 3)
        np.testing.assert_allclose(
            np.asarray(ad.layers["Z"]), reference_dense(X, ALPHA), rtol=1e-5, atol=1e-5
        )


class TestBackedParity:
    def test_backed_matches_inmemory(self, scx_from_adata):
        ad, _ = small_adata()
        path = scx_from_adata(ad, "pflog.scx")

        mem = ad.copy()
        pyscx.accel.pflog(mem, alpha=ALPHA, store="all", n_components=4, n_power_iterations=4)

        backed = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.pflog(backed, alpha=ALPHA, store="all", n_components=4, n_power_iterations=4)

        # Baseline is exact (rotation/sign-free) → tight tolerance.
        np.testing.assert_allclose(
            backed.obs["pflog_baseline"].to_numpy(),
            mem.obs["pflog_baseline"].to_numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        # Singular values are rotation/sign invariant.
        np.testing.assert_allclose(
            np.asarray(backed.uns["X_pflog_pca_singular_values"]),
            np.asarray(mem.uns["X_pflog_pca_singular_values"]),
            rtol=1e-4,
            atol=1e-5,
        )


class TestValidation:
    def test_nonpositive_alpha_raises(self):
        ad, _ = small_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pflog(ad, alpha=0.0)
        with pytest.raises(ValueError):
            pyscx.accel.pflog(ad, alpha=-1.0)

    def test_bad_store_raises(self):
        ad, _ = small_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pflog(ad, alpha=ALPHA, store="nonsense")

    def test_dense_guard_raises(self):
        ad, _ = small_adata()
        # 6×4 = 24 elements; guard at 10 must reject the dense path.
        with pytest.raises(RuntimeError):
            pyscx.accel.pflog(ad, alpha=ALPHA, store="dense", dense_max_elems=10)

    def test_empty_cell_ok(self):
        # v4 improvement: an empty cell is representable (baseline 0), not an error.
        X = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        ad = anndata.AnnData(sp.csr_matrix(X))
        ad.var_names = [f"gene_{i}" for i in range(4)]
        pyscx.accel.pflog(ad, alpha=ALPHA, store="baseline")
        b = ad.obs["pflog_baseline"].to_numpy()
        assert b[0] == 0.0
        assert np.all(np.isfinite(b))

    def test_raw_count_guard_rejects_transformed_lazy(self, scx_from_adata):
        ad, _ = small_adata()
        path = scx_from_adata(ad, "pflog_guard.scx")
        backed = pyscx.open(path).to_anndata(backed=True)
        # normalize_total turns X into a lazy-transformed dataset → must reject.
        pyscx.accel.normalize_total(backed, target_sum=1e4)
        with pytest.raises((ValueError, RuntimeError)):
            pyscx.accel.pflog(backed, alpha=ALPHA, store="baseline")


class TestRouteMetadata:
    def test_route_recorded(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", n_components=3)
        assert ad.uns["scx_accel"]["pflog"]["route"] == "cpu_csr"


# CodecId numeric values (scx-codec/src/dispatch.rs): Zstd=2, Pcodec=4.
_CODEC_ZSTD = 2
_CODEC_PCODEC = 4


class TestStreamToDisk:
    """`out=` streams the materialized transform to a new SCX file."""

    def test_delta_baseline_roundtrip(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "compact.scx")
        pyscx.accel.pflog(ad, alpha=ALPHA, store="dense", out=out, store_repr="delta_baseline")

        exp = pyscx.open(out)
        # Compact form: X is the sparse delta layer (Pcodec), baseline in obs.
        assert exp.codec_id == _CODEC_PCODEC
        re_ad = exp.to_anndata()
        assert "pflog_baseline" in re_ad.obs

        Z = pyscx.accel.pflog_reconstruct(re_ad)
        ref = reference_dense(X, ALPHA)
        np.testing.assert_allclose(Z, ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(Z.sum(axis=1), np.zeros(ad.n_obs), atol=1e-4)

    def test_delta_baseline_custom_alpha(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "compact_a.scx")
        pyscx.accel.pflog(ad, alpha=1.0, store="dense", out=out)  # default repr
        re_ad = pyscx.open(out).to_anndata()
        Z = pyscx.accel.pflog_reconstruct(re_ad)
        np.testing.assert_allclose(Z, reference_dense(X, 1.0), rtol=1e-5, atol=1e-5)

    def test_dense_roundtrip_multishard(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "dense.scx")
        # Tiny shard_size exercises the re-chunking / multi-shard path (6 rows / 2).
        pyscx.accel.pflog(
            ad, alpha=ALPHA, store="dense", out=out, store_repr="dense", shard_size=2
        )
        exp = pyscx.open(out)
        assert exp.codec_id == _CODEC_ZSTD
        assert exp.shard_count > 1
        re_ad = exp.to_anndata()
        Z = np.asarray(re_ad.X.toarray() if hasattr(re_ad.X, "toarray") else re_ad.X)
        np.testing.assert_allclose(Z, reference_dense(X, ALPHA), rtol=1e-5, atol=1e-5)

    def test_disk_path_bypasses_size_guard(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "bypass.scx")
        # 6×4 = 24 elements; a guard of 10 rejects the in-memory layer but the
        # streamed-to-disk path has no such guard.
        pyscx.accel.pflog(ad, alpha=ALPHA, store="dense", out=out, dense_max_elems=10)
        re_ad = pyscx.open(out).to_anndata()
        Z = pyscx.accel.pflog_reconstruct(re_ad)
        np.testing.assert_allclose(Z, reference_dense(X, ALPHA), rtol=1e-5, atol=1e-5)
        # Same guard still bites without out=.
        with pytest.raises(RuntimeError):
            pyscx.accel.pflog(ad.copy(), alpha=ALPHA, store="dense", dense_max_elems=10)

    def test_out_requires_dense_store(self, tmp_path):
        ad, _ = small_adata()
        out = str(tmp_path / "nope.scx")
        with pytest.raises(ValueError):
            pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", out=out)

    def test_bad_store_repr_raises(self, tmp_path):
        ad, _ = small_adata()
        out = str(tmp_path / "nope.scx")
        with pytest.raises(ValueError):
            pyscx.accel.pflog(ad, alpha=ALPHA, store="dense", out=out, store_repr="bogus")

    def test_backed_matches_inmemory_on_disk(self, tmp_path, scx_from_adata):
        ad, X = small_adata()
        src = scx_from_adata(ad, "pflog_src.scx")

        out_mem = str(tmp_path / "from_mem.scx")
        pyscx.accel.pflog(ad.copy(), alpha=ALPHA, store="dense", out=out_mem)

        backed = pyscx.open(src).to_anndata(backed=True)
        out_backed = str(tmp_path / "from_backed.scx")
        pyscx.accel.pflog(backed, alpha=ALPHA, store="dense", out=out_backed)

        Z_mem = pyscx.accel.pflog_reconstruct(pyscx.open(out_mem).to_anndata())
        Z_backed = pyscx.accel.pflog_reconstruct(pyscx.open(out_backed).to_anndata())
        np.testing.assert_allclose(Z_mem, Z_backed, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(Z_mem, reference_dense(X, ALPHA), rtol=1e-5, atol=1e-5)


class TestValidationHardening:
    """Review-driven guards: non-finite counts and non-finite alpha."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_count_rejected(self, bad):
        _, X = small_adata()
        Xb = X.copy()
        Xb[0, 1] = bad  # a non-finite nonzero
        ad = anndata.AnnData(sp.csr_matrix(Xb))
        ad.var_names = [f"gene_{i}" for i in range(Xb.shape[1])]
        with pytest.raises((ValueError, RuntimeError)):
            pyscx.accel.pflog(ad, alpha=ALPHA, store="baseline")

    @pytest.mark.parametrize("bad_alpha", [float("inf"), float("nan")])
    def test_non_finite_alpha_rejected(self, bad_alpha):
        ad, _ = small_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pflog(ad, alpha=bad_alpha)


class TestPcaOptions:
    """zero_center, layer, custom obsm_key, and store='all' + out= coverage."""

    def test_zero_center_false_finite(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", zero_center=False, n_components=3)
        emb = np.asarray(ad.obsm["X_pflog_pca"])
        assert emb.shape == (ad.n_obs, 3)
        assert np.all(np.isfinite(emb))

    def test_runs_on_named_layer(self):
        ad, X = small_adata()
        ad.layers["counts"] = ad.X.copy()
        pyscx.accel.pflog(ad, alpha=ALPHA, layer="counts", store="baseline")
        ref = reference_dense(X, ALPHA)
        assert np.all(np.isfinite(ad.obs["pflog_baseline"].to_numpy()))
        assert ref.shape == (ad.n_obs, ad.n_vars)

    def test_custom_obsm_key(self):
        ad, _ = small_adata()
        pyscx.accel.pflog(ad, alpha=ALPHA, store="pca", n_components=3, obsm_key="X_custom")
        assert "X_custom" in ad.obsm
        assert "X_custom_singular_values" in ad.uns

    def test_store_all_with_out(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "all.scx")
        pyscx.accel.pflog(ad, alpha=ALPHA, store="all", out=out, n_components=3)
        # In-memory adata gets PCA + baseline; the file is the compact form.
        assert ad.obsm["X_pflog_pca"].shape == (ad.n_obs, 3)
        assert "pflog_baseline" in ad.obs
        re_ad = pyscx.open(out).to_anndata()
        Z = pyscx.accel.pflog_reconstruct(re_ad)
        np.testing.assert_allclose(Z, reference_dense(X, ALPHA), rtol=1e-5, atol=1e-5)


class TestDeviceValidation:
    def test_invalid_device_raises_value_error(self):
        # Round-2 review (codex + Cursor): pflog previously skipped
        # resolve_device, so an invalid device string was silently treated as
        # GPU intent in the route stamp instead of raising like every other
        # accel op.
        ad, _ = small_adata()
        with pytest.raises(ValueError, match="unknown device"):
            pyscx.accel.pflog(ad, alpha=ALPHA, store="baseline", device="tpu")
        assert "pflog_baseline" not in ad.obs


def test_pflog_layer_on_a_backed_adata(tmp_dir):
    """`pflog(layer=)` on a backed file — the `ScxBackedLayerDataset` arm.

    Same hole as `score_genes` / `highly_variable_genes` / `calculate_qc_metrics`
    had: a backed layer is a distinct pyclass that a bare
    `cast::<ScxBackedSparseDataset>()` misses, so the handle reached
    `scipy.sparse.csr_matrix(...)` and raised.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    import pyscx
    from pyscx import ScxBackedLayerDataset

    rng = np.random.default_rng(4)
    n_obs, n_vars = 120, 60
    x = sp.csr_matrix(
        (rng.random((n_obs, n_vars)) < 0.3) * rng.integers(1, 40, (n_obs, n_vars)).astype(np.float32)
    )
    adata = ad.AnnData(
        X=x,
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(n_vars)]),
    )
    adata.layers["counts"] = x.copy()

    path = str(tmp_dir / "pflog_backed_layer.scx")
    pyscx.from_anndata(adata, path)
    backed = pyscx.open(path).to_anndata(backed=True)
    assert isinstance(backed.layers["counts"], ScxBackedLayerDataset), "premise"

    pyscx.accel.pflog(backed, layer="counts", store="baseline")
    assert "pflog_baseline" in backed.obs
    assert np.all(np.isfinite(backed.obs["pflog_baseline"].to_numpy()))
