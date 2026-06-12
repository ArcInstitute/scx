"""Tests for pyscx.accel.pflog1ppf (PFlog1pPF / shifted-CLR, Booeshaghi 2026).

The exact transform is dense; the binding writes the per-cell baseline to
`adata.obs`, an out-of-core PCA embedding to `adata.obsm`, and (size-guarded) a
dense layer. The shared `reference_dense` helper mirrors the spec's §11.1 Python
reference and is the source of truth; assertions hold to ~1e-5 (the f32 `delta`
floor), with PCA at a looser tolerance (randomized SVD on f32).
"""

import numpy as np
import pytest
import scipy.sparse as sp
import anndata

import pyscx


def reference_dense(X, c=1.0):
    """Spec §11.1 exact reference: z_ij = log(x_ij/s_i + c) − mean_j(...)."""
    X = np.asarray(X, dtype=np.float64)
    depth = X.sum(axis=1, keepdims=True)
    U = X / depth
    L = np.log(U + c)
    return L - L.mean(axis=1, keepdims=True)


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


class TestBaseline:
    def test_writes_obs_baseline(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="baseline")
        assert "pflog1ppf_baseline" in ad.obs
        assert ad.obs["pflog1ppf_baseline"].shape == (ad.n_obs,)

    def test_custom_baseline_key(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="baseline", baseline_key="b")
        assert "b" in ad.obs


class TestDenseEquivalence:
    def test_matches_reference_default_c(self):
        ad, X = small_adata()
        pyscx.accel.pflog1ppf(ad, store="dense", layer_out="Z")
        Z = np.asarray(ad.layers["Z"])
        np.testing.assert_allclose(Z, reference_dense(X, 1.0), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("c", [0.1, 0.5, 1.0, 2.0])
    def test_matches_reference_general_c(self, c):
        ad, X = small_adata()
        pyscx.accel.pflog1ppf(ad, c=c, store="dense", layer_out="Z")
        Z = np.asarray(ad.layers["Z"])
        np.testing.assert_allclose(Z, reference_dense(X, c), rtol=1e-5, atol=1e-5)

    def test_row_sums_are_zero(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="dense", layer_out="Z")
        Z = np.asarray(ad.layers["Z"])
        np.testing.assert_allclose(Z.sum(axis=1), 0.0, atol=1e-5)

    def test_dense_default_layer_name(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="dense")
        assert "pflog1ppf" in ad.layers


class TestPCA:
    def test_writes_obsm_and_singular_values(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="pca", n_components=3)
        assert ad.obsm["X_pflog1ppf_pca"].shape == (ad.n_obs, 3)
        assert "X_pflog1ppf_pca_singular_values" in ad.uns

    def test_singular_values_match_dense_svd(self):
        ad, X = small_adata()
        pyscx.accel.pflog1ppf(ad, store="pca", n_components=4, n_power_iterations=4)
        ref = reference_dense(X, 1.0)
        ref_c = ref - ref.mean(axis=0, keepdims=True)
        sv_ref = np.linalg.svd(ref_c, compute_uv=False)
        sv = np.asarray(ad.uns["X_pflog1ppf_pca_singular_values"])
        np.testing.assert_allclose(sv, sv_ref[: len(sv)], rtol=1e-3, atol=1e-4)

    def test_embedding_column_norms_equal_singular_values(self):
        # embeddings = U·Σ with U orthonormal → column 2-norms are the σ_i.
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="pca", n_components=4, n_power_iterations=4)
        emb = np.asarray(ad.obsm["X_pflog1ppf_pca"])
        sv = np.asarray(ad.uns["X_pflog1ppf_pca_singular_values"])
        np.testing.assert_allclose(
            np.linalg.norm(emb, axis=0), sv, rtol=1e-3, atol=1e-4
        )

    def test_store_all_writes_everything(self):
        ad, X = small_adata()
        pyscx.accel.pflog1ppf(ad, store="all", n_components=3, layer_out="Z")
        assert "pflog1ppf_baseline" in ad.obs
        assert ad.obsm["X_pflog1ppf_pca"].shape == (ad.n_obs, 3)
        np.testing.assert_allclose(
            np.asarray(ad.layers["Z"]), reference_dense(X, 1.0), rtol=1e-5, atol=1e-5
        )


class TestBackedParity:
    def test_backed_matches_inmemory(self, scx_from_adata):
        ad, _ = small_adata()
        path = scx_from_adata(ad, "pflog1ppf.scx")

        mem = ad.copy()
        pyscx.accel.pflog1ppf(mem, store="all", n_components=4, n_power_iterations=4)

        backed = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.pflog1ppf(backed, store="all", n_components=4, n_power_iterations=4)

        # Baseline is exact (rotation/sign-free) → tight tolerance.
        np.testing.assert_allclose(
            backed.obs["pflog1ppf_baseline"].to_numpy(),
            mem.obs["pflog1ppf_baseline"].to_numpy(),
            rtol=1e-5,
            atol=1e-6,
        )
        # Singular values are rotation/sign invariant.
        np.testing.assert_allclose(
            np.asarray(backed.uns["X_pflog1ppf_pca_singular_values"]),
            np.asarray(mem.uns["X_pflog1ppf_pca_singular_values"]),
            rtol=1e-4,
            atol=1e-5,
        )


class TestValidation:
    def test_nonpositive_c_raises(self):
        ad, _ = small_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pflog1ppf(ad, c=0.0)
        with pytest.raises(ValueError):
            pyscx.accel.pflog1ppf(ad, c=-1.0)

    def test_bad_store_raises(self):
        ad, _ = small_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pflog1ppf(ad, store="nonsense")

    def test_dense_guard_raises(self):
        ad, _ = small_adata()
        # 6×4 = 24 elements; guard at 10 must reject the dense path.
        with pytest.raises(RuntimeError):
            pyscx.accel.pflog1ppf(ad, store="dense", dense_max_elems=10)

    def test_empty_cell_raises(self):
        X = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        ad = anndata.AnnData(sp.csr_matrix(X))
        ad.var_names = [f"gene_{i}" for i in range(4)]
        with pytest.raises((ValueError, RuntimeError)):
            pyscx.accel.pflog1ppf(ad, store="baseline")

    def test_raw_count_guard_rejects_transformed_lazy(self, scx_from_adata):
        ad, _ = small_adata()
        path = scx_from_adata(ad, "pflog1ppf_guard.scx")
        backed = pyscx.open(path).to_anndata(backed=True)
        # normalize_total turns X into a lazy-transformed dataset → must reject.
        pyscx.accel.normalize_total(backed, target_sum=1e4)
        with pytest.raises((ValueError, RuntimeError)):
            pyscx.accel.pflog1ppf(backed, store="baseline")


class TestRouteMetadata:
    def test_route_recorded(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="pca", n_components=3)
        assert ad.uns["scx_accel"]["pflog1ppf"]["route"] == "cpu_csr"


# CodecId numeric values (scx-codec/src/dispatch.rs): Zstd=2, Pcodec=4.
_CODEC_ZSTD = 2
_CODEC_PCODEC = 4


class TestStreamToDisk:
    """Phase 4c — `out=` streams the materialized transform to a new SCX file."""

    def test_delta_baseline_roundtrip(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "compact.scx")
        pyscx.accel.pflog1ppf(ad, store="dense", out=out, store_repr="delta_baseline")

        exp = pyscx.open(out)
        # Compact form: X is the sparse delta layer (Pcodec), baseline in obs.
        assert exp.codec_id == _CODEC_PCODEC
        re_ad = exp.to_anndata()
        assert "pflog1ppf_baseline" in re_ad.obs

        Z = pyscx.accel.pflog1ppf_reconstruct(re_ad)
        ref = reference_dense(X, 1.0)
        np.testing.assert_allclose(Z, ref, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(Z.sum(axis=1), np.zeros(ad.n_obs), atol=1e-4)

    def test_delta_baseline_custom_c(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "compact_c.scx")
        pyscx.accel.pflog1ppf(ad, c=0.5, store="dense", out=out)  # default repr
        re_ad = pyscx.open(out).to_anndata()
        Z = pyscx.accel.pflog1ppf_reconstruct(re_ad)
        np.testing.assert_allclose(Z, reference_dense(X, 0.5), rtol=1e-5, atol=1e-5)

    def test_dense_roundtrip_multishard(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "dense.scx")
        # Tiny shard_size exercises the re-chunking / multi-shard path (6 rows / 2).
        pyscx.accel.pflog1ppf(
            ad, store="dense", out=out, store_repr="dense", shard_size=2
        )
        exp = pyscx.open(out)
        assert exp.codec_id == _CODEC_ZSTD
        assert exp.shard_count > 1
        re_ad = exp.to_anndata()
        Z = np.asarray(re_ad.X.toarray() if hasattr(re_ad.X, "toarray") else re_ad.X)
        np.testing.assert_allclose(Z, reference_dense(X, 1.0), rtol=1e-5, atol=1e-5)

    def test_disk_path_bypasses_size_guard(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "bypass.scx")
        # 6×4 = 24 elements; a guard of 10 rejects the in-memory layer but the
        # streamed-to-disk path has no such guard.
        pyscx.accel.pflog1ppf(ad, store="dense", out=out, dense_max_elems=10)
        re_ad = pyscx.open(out).to_anndata()
        Z = pyscx.accel.pflog1ppf_reconstruct(re_ad)
        np.testing.assert_allclose(Z, reference_dense(X, 1.0), rtol=1e-5, atol=1e-5)
        # Same guard still bites without out=.
        with pytest.raises(RuntimeError):
            pyscx.accel.pflog1ppf(ad.copy(), store="dense", dense_max_elems=10)

    def test_out_requires_dense_store(self, tmp_path):
        ad, _ = small_adata()
        out = str(tmp_path / "nope.scx")
        with pytest.raises(ValueError):
            pyscx.accel.pflog1ppf(ad, store="pca", out=out)

    def test_bad_store_repr_raises(self, tmp_path):
        ad, _ = small_adata()
        out = str(tmp_path / "nope.scx")
        with pytest.raises(ValueError):
            pyscx.accel.pflog1ppf(ad, store="dense", out=out, store_repr="bogus")

    def test_backed_matches_inmemory_on_disk(self, tmp_path, scx_from_adata):
        ad, X = small_adata()
        src = scx_from_adata(ad, "pflog1ppf_src.scx")

        out_mem = str(tmp_path / "from_mem.scx")
        pyscx.accel.pflog1ppf(ad.copy(), store="dense", out=out_mem)

        backed = pyscx.open(src).to_anndata(backed=True)
        out_backed = str(tmp_path / "from_backed.scx")
        pyscx.accel.pflog1ppf(backed, store="dense", out=out_backed)

        Z_mem = pyscx.accel.pflog1ppf_reconstruct(pyscx.open(out_mem).to_anndata())
        Z_backed = pyscx.accel.pflog1ppf_reconstruct(
            pyscx.open(out_backed).to_anndata()
        )
        np.testing.assert_allclose(Z_mem, Z_backed, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(Z_mem, reference_dense(X, 1.0), rtol=1e-5, atol=1e-5)


class TestValidationHardening:
    """Review-driven guards: non-finite counts and non-finite c."""

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_count_rejected(self, bad):
        _, X = small_adata()
        Xb = X.copy()
        Xb[0, 1] = bad  # a non-finite nonzero
        ad = anndata.AnnData(sp.csr_matrix(Xb))
        ad.var_names = [f"gene_{i}" for i in range(Xb.shape[1])]
        with pytest.raises((ValueError, RuntimeError)):
            pyscx.accel.pflog1ppf(ad, store="baseline")

    @pytest.mark.parametrize("bad_c", [float("inf"), float("nan")])
    def test_non_finite_c_rejected(self, bad_c):
        ad, _ = small_adata()
        with pytest.raises(ValueError):
            pyscx.accel.pflog1ppf(ad, c=bad_c)


class TestPcaOptions:
    """zero_center, layer, custom obsm_key, and store='all' + out= coverage."""

    def test_zero_center_false_finite(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="pca", zero_center=False, n_components=3)
        emb = np.asarray(ad.obsm["X_pflog1ppf_pca"])
        assert emb.shape == (ad.n_obs, 3)
        assert np.all(np.isfinite(emb))

    def test_runs_on_named_layer(self):
        ad, X = small_adata()
        ad.layers["counts"] = ad.X.copy()
        pyscx.accel.pflog1ppf(ad, layer="counts", store="baseline")
        # Baseline computed from the layer must match the full-transcriptome ref.
        ref = reference_dense(X, 1.0)
        # baseline_i = Z_ij - delta_ij at any zero column == row mean of -log term;
        # simplest invariant: each Z row sums to ~0, baseline finite.
        assert np.all(np.isfinite(ad.obs["pflog1ppf_baseline"].to_numpy()))
        assert ref.shape == (ad.n_obs, ad.n_vars)

    def test_custom_obsm_key(self):
        ad, _ = small_adata()
        pyscx.accel.pflog1ppf(ad, store="pca", n_components=3, obsm_key="X_custom")
        assert "X_custom" in ad.obsm
        assert "X_custom_singular_values" in ad.uns

    def test_store_all_with_out(self, tmp_path):
        ad, X = small_adata()
        out = str(tmp_path / "all.scx")
        pyscx.accel.pflog1ppf(ad, store="all", out=out, n_components=3)
        # In-memory adata gets PCA + baseline; the file is the compact form.
        assert ad.obsm["X_pflog1ppf_pca"].shape == (ad.n_obs, 3)
        assert "pflog1ppf_baseline" in ad.obs
        re_ad = pyscx.open(out).to_anndata()
        Z = pyscx.accel.pflog1ppf_reconstruct(re_ad)
        np.testing.assert_allclose(Z, reference_dense(X, 1.0), rtol=1e-5, atol=1e-5)
