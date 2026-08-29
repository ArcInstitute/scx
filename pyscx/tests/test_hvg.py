"""Tests for pyscx.accel.highly_variable_genes().

Tests streaming HVG on backed and lazy-transformed datasets,
with seurat_v3 and seurat flavors, subset support, and batch_key.
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def scx_path(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic data."""
    import pyscx

    path = str(tmp_dir / "hvg_test.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def scx_from_adata(tmp_dir):
    """Factory to create SCX files with custom names."""
    import pyscx

    def _create(adata, name):
        path = str(tmp_dir / name)
        pyscx.from_anndata(adata, path)
        return path

    return _create


# ---------------------------------------------------------------------------
# seurat_v3 flavor
# ---------------------------------------------------------------------------


class TestSeuratV3:
    """Tests for seurat_v3 flavor (raw count data)."""

    def test_basic_backed(self, scx_path):
        """HVG seurat_v3 on backed data sets expected var columns."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.highly_variable_genes(adata, n_top_genes=20, flavor="seurat_v3")

        assert "highly_variable" in adata.var.columns
        assert "means" in adata.var.columns
        assert "variances" in adata.var.columns
        assert "variances_norm" in adata.var.columns
        assert "highly_variable_rank" in adata.var.columns
        assert adata.var["highly_variable"].sum() == 20

    def test_overlap_with_scanpy(self, synthetic_adata, scx_from_adata):
        """HVG selection >80% overlap with scanpy on synthetic data."""
        import pyscx
        import scanpy as sc

        path = scx_from_adata(synthetic_adata, "hvg_overlap.scx")
        adata_scx = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.highly_variable_genes(
            adata_scx, n_top_genes=20, flavor="seurat_v3"
        )
        scx_hvg = set(adata_scx.var.index[adata_scx.var["highly_variable"]])

        adata_ref = synthetic_adata.copy()
        sc.pp.highly_variable_genes(adata_ref, n_top_genes=20, flavor="seurat_v3")
        ref_hvg = set(adata_ref.var.index[adata_ref.var["highly_variable"]])

        overlap = len(scx_hvg & ref_hvg) / max(len(ref_hvg), 1)
        assert overlap > 0.80, f"HVG overlap {overlap:.2f} < 0.80"

    def test_batch_column_parity_with_scanpy(self, scx_from_adata):
        """§2.2: multi-batch seurat_v3 publishes the per-gene MEDIAN rank (not
        selection order), writes `highly_variable_nbatches`, uses a true
        even-median, and sorts NaN ranks last — column parity with scanpy.

        Uses an even batch count (4) so `np.ma.median` averages the two middle
        ranks, exercising the even-median fix (half-integer ranks appear).
        """
        import anndata as ad
        import pandas as pd
        import pyscx
        import scanpy as sc

        pytest.importorskip("skmisc")
        rng = np.random.default_rng(11)
        n_per, n_vars, n_batches = 200, 80, 4
        blocks, batches = [], []
        for b in range(n_batches):
            means = rng.uniform(0.3, 4.0, size=n_vars) * (1.0 + 0.25 * b)
            blocks.append(
                rng.poisson(means[None, :], size=(n_per, n_vars)).astype(np.float32)
            )
            batches += [f"b{b}"] * n_per
        raw = ad.AnnData(X=np.vstack(blocks))
        raw.obs_names = [f"c{i}" for i in range(raw.n_obs)]
        raw.var_names = [f"g{j}" for j in range(n_vars)]
        raw.obs["batch"] = pd.Categorical(batches)

        path = scx_from_adata(raw.copy(), "hvg_v3_batch_parity.scx")
        a = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.highly_variable_genes(
            a, n_top_genes=25, flavor="seurat_v3", batch_key="batch"
        )

        ref = raw.copy()
        sc.pp.highly_variable_genes(
            ref, n_top_genes=25, flavor="seurat_v3", batch_key="batch"
        )

        # highly_variable_nbatches is now written (was omitted) and matches.
        assert "highly_variable_nbatches" in a.var.columns
        np.testing.assert_array_equal(
            a.var["highly_variable_nbatches"].values,
            ref.var["highly_variable_nbatches"].values,
        )
        # Published rank is the per-gene median (NaN-last), matching scanpy —
        # NOT the old 0..n_top-1 selection order.
        rk = np.asarray(a.var["highly_variable_rank"], dtype=float)
        np.testing.assert_array_equal(np.isnan(rk), np.isnan(ref.var["highly_variable_rank"].values.astype(float)))
        np.testing.assert_allclose(
            rk,
            np.asarray(ref.var["highly_variable_rank"], dtype=float),
            equal_nan=True,
        )
        finite_ranks = np.sort(rk[~np.isnan(rk)])
        assert not np.array_equal(
            finite_ranks, np.arange(finite_ranks.size, dtype=float)
        ), "highly_variable_rank is still the old selection-order sequence"
        assert np.any(finite_ranks != np.floor(finite_ranks)), (
            "even-median should produce at least one half-integer rank"
        )
        np.testing.assert_allclose(
            np.asarray(a.var["variances_norm"], dtype=float),
            np.asarray(ref.var["variances_norm"], dtype=float),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_array_equal(
            a.var["highly_variable"].values, ref.var["highly_variable"].values
        )

    def test_after_filter_genes(self, scx_path):
        """HVG works after filter_genes (with active col_projection)."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.filter_genes(adata, min_cells=1)
        n_vars_after = adata.n_vars
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=min(10, n_vars_after), flavor="seurat_v3"
        )
        assert "highly_variable" in adata.var.columns

    def test_subset_reduces_vars(self, scx_path):
        """subset=True reduces n_vars and updates var DataFrame."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        n_top = 15
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=n_top, flavor="seurat_v3", subset=True
        )
        assert adata.n_vars == n_top
        assert adata.var.shape[0] == n_top

    def test_subset_lazy(self, scx_path):
        """subset=True on lazy-transformed dataset stays lazy."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)

        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=10, flavor="seurat_v3", subset=True
        )
        assert adata.n_vars == 10
        # X should still be lazy-transformed (not materialized)
        assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)

    def test_batch_key(self, scx_path):
        """batch_key produces valid results."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=15, flavor="seurat_v3", batch_key="batch"
        )
        assert "highly_variable" in adata.var.columns
        assert adata.var["highly_variable"].sum() == 15


# ---------------------------------------------------------------------------
# seurat flavor
# ---------------------------------------------------------------------------


class TestSeurat:
    """Tests for default seurat flavor (log-normalized data)."""

    def test_basic_backed(self, scx_path):
        """HVG seurat on backed data sets expected var columns."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        pyscx.accel.log1p(adata)
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=20, flavor="seurat"
        )

        assert "highly_variable" in adata.var.columns
        assert "means" in adata.var.columns
        assert "dispersions" in adata.var.columns
        assert "dispersions_norm" in adata.var.columns
        assert adata.var["highly_variable"].sum() == 20

    def test_overlap_with_scanpy(self, synthetic_adata, scx_from_adata):
        """Seurat flavor selects reasonable genes (relaxed threshold for small data).

        On the 50-gene synthetic dataset, bin-based normalization is noisy
        due to very few genes per bin. Use n_bins=5 for more stable results.
        """
        import pyscx
        import scanpy as sc

        path = scx_from_adata(synthetic_adata, "hvg_seurat.scx")

        adata_scx = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_scx)
        pyscx.accel.log1p(adata_scx)
        pyscx.accel.highly_variable_genes(
            adata_scx, n_top_genes=20, flavor="seurat", n_bins=5
        )
        scx_hvg = set(adata_scx.var.index[adata_scx.var["highly_variable"]])

        adata_ref = synthetic_adata.copy()
        sc.pp.normalize_total(adata_ref)
        sc.pp.log1p(adata_ref)
        sc.pp.highly_variable_genes(adata_ref, n_top_genes=20, flavor="seurat", n_bins=5)
        ref_hvg = set(adata_ref.var.index[adata_ref.var["highly_variable"]])

        overlap = len(scx_hvg & ref_hvg) / max(len(ref_hvg), 1)
        # On the 50-gene synthetic dataset, bin-based normalization is noisy.
        # The seurat_v3 flavor achieves >99% on real data; seurat flavor is
        # more sensitive to binning details. Overlap >30% confirms reasonable gene selection.
        assert overlap > 0.30, f"Seurat HVG overlap {overlap:.2f} < 0.30"

    def test_column_parity_with_scanpy(self, scx_from_adata):
        """§2.1: every published seurat column matches scanpy 1.12 when both
        see IDENTICAL log-normalized data.

        The moments must be computed in count space (`expm1` first, honoring
        `uns['log1p']['base']`), `means` published as `log1p(count-mean)`,
        `dispersions` as `log(var/mean)`, and singleton bins normalized to
        exactly 1. We normalize+log1p once with scanpy and write THAT matrix to
        SCX so this isolates the HVG algorithm from the `normalize_total`
        default divergence (§3.5).
        """
        import anndata as ad
        import pyscx
        import scanpy as sc

        rng = np.random.default_rng(3)
        n_obs, n_vars = 400, 60
        X = rng.poisson(0.4, size=(n_obs, n_vars)).astype(np.float32)
        X[:, 0] = rng.poisson(8, size=n_obs)
        X[:, 5] = rng.poisson(5, size=n_obs)
        raw = ad.AnnData(X=X)
        raw.obs_names = [f"c{i}" for i in range(n_obs)]
        raw.var_names = [f"g{j}" for j in range(n_vars)]

        ref = raw.copy()
        sc.pp.normalize_total(ref, target_sum=1e4)
        sc.pp.log1p(ref)

        path = scx_from_adata(ref.copy(), "hvg_seurat_parity.scx")
        a = pyscx.open(path).to_anndata(backed=True)
        a.uns["log1p"] = {"base": None}
        pyscx.accel.highly_variable_genes(a, n_top_genes=15, flavor="seurat", n_bins=10)

        ref2 = ref.copy()
        sc.pp.highly_variable_genes(ref2, n_top_genes=15, flavor="seurat", n_bins=10)

        for col in ["means", "dispersions", "dispersions_norm"]:
            np.testing.assert_allclose(
                np.asarray(a.var[col], dtype=float),
                np.asarray(ref2.var[col], dtype=float),
                rtol=1e-5,
                atol=1e-5,
                equal_nan=True,
                err_msg=f"seurat column '{col}' diverges from scanpy",
            )
        np.testing.assert_array_equal(
            a.var["highly_variable"].values, ref2.var["highly_variable"].values
        )

    def test_non_default_log_base_parity(self, scx_from_adata):
        """§2.1: `expm1` un-log honors `uns['log1p']['base']`. With a base-2
        log1p, count-space moments (and all columns) still match scanpy."""
        import anndata as ad
        import pyscx
        import scanpy as sc

        rng = np.random.default_rng(7)
        n_obs, n_vars = 400, 50
        counts = rng.poisson(1.0, size=(n_obs, n_vars)).astype(np.float32)
        counts[:, 1] = rng.poisson(9, size=n_obs)
        # log1p with base 2: stored = log2(1 + normalized_count).
        norm = ad.AnnData(X=counts.copy())
        norm.obs_names = [f"c{i}" for i in range(n_obs)]
        norm.var_names = [f"g{j}" for j in range(n_vars)]
        sc.pp.normalize_total(norm, target_sum=1e4)
        sc.pp.log1p(norm, base=2)  # sets uns["log1p"]["base"] = 2

        path = scx_from_adata(norm.copy(), "hvg_seurat_base2.scx")
        a = pyscx.open(path).to_anndata(backed=True)
        a.uns["log1p"] = {"base": 2}
        pyscx.accel.highly_variable_genes(a, n_top_genes=12, flavor="seurat", n_bins=10)

        ref = norm.copy()
        sc.pp.highly_variable_genes(ref, n_top_genes=12, flavor="seurat", n_bins=10)

        for col in ["means", "dispersions", "dispersions_norm"]:
            np.testing.assert_allclose(
                np.asarray(a.var[col], dtype=float),
                np.asarray(ref.var[col], dtype=float),
                rtol=1e-4,
                atol=1e-4,
                equal_nan=True,
                err_msg=f"base-2 seurat column '{col}' diverges from scanpy",
            )
        np.testing.assert_array_equal(
            a.var["highly_variable"].values, ref.var["highly_variable"].values
        )


# ---------------------------------------------------------------------------
# Fallback and edge cases
# ---------------------------------------------------------------------------


class TestFallback:
    """Tests for non-SCX adata fallback to scanpy."""

    def test_fallback_scipy(self, synthetic_adata):
        """Non-SCX adata falls back to scanpy."""
        import pyscx

        adata = synthetic_adata.copy()
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=20, flavor="seurat_v3"
        )
        assert "highly_variable" in adata.var.columns

    def test_invalid_flavor(self, scx_path):
        """Invalid flavor raises ValueError."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        with pytest.raises(ValueError, match="Unsupported HVG flavor"):
            pyscx.accel.highly_variable_genes(
                adata, n_top_genes=20, flavor="invalid"
            )


# ---------------------------------------------------------------------------
# GPU-narrowing fallback warning content
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    try:
        import pyscx

        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


@pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — narrow-fallback warning is GPU-only",
)
def test_gpu_fallback_warning_includes_device_string(scx_path):
    """When the GPU HVG path is narrowed to CPU (non-`seurat_v3` flavor),
    the `UserWarning` must echo the user's `device` string so that an
    explicit `gpu:N` index isn't silently dropped without acknowledgment.

    `flavor="seurat"` is the remaining narrowing trigger: `batch_key` is
    now supported on the GPU `seurat_v3` path via per-batch kernels.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    pyscx.accel.normalize_total(adata)
    pyscx.accel.log1p(adata)

    with pytest.warns(UserWarning, match=r'device="gpu:0"'):
        pyscx.accel.highly_variable_genes(
            adata,
            n_top_genes=20,
            flavor="seurat",
            device="gpu:0",
        )


# ---------------------------------------------------------------------------
# seurat flavor — boundary errors (round-1 review, PR #478)
# ---------------------------------------------------------------------------


class TestSeuratBoundaryErrors:
    """The Rust binning kernel answers degenerate inputs with all-NaN, which
    the -inf selection floor would silently turn into ZERO genes selected —
    so the pyscx boundary must refuse them loudly instead, the way the pandas
    era did (it raised)."""

    def _adata(self, values=None):
        import anndata
        import numpy as np
        import pandas as pd
        from scipy import sparse

        X = np.arange(12, dtype=np.float32).reshape(4, 3) % 5 + 0.1
        if values is not None:
            X[0, 0] = values
        return anndata.AnnData(
            X=sparse.csr_matrix(X),
            obs=pd.DataFrame(index=[f"c{i}" for i in range(4)]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(3)]),
        )

    def test_n_bins_zero_raises(self):
        import pyscx

        with pytest.raises(ValueError, match="n_bins"):
            pyscx.accel.highly_variable_genes(
                self._adata(), n_top_genes=2, flavor="seurat", n_bins=0
            )

    def test_absurd_n_bins_raises_instead_of_aborting(self):
        """Pandas raised MemoryError and the interpreter survived; Rust's
        infallible per-bin allocations would abort the whole process."""
        import sys

        import pyscx

        with pytest.raises(ValueError, match="n_bins"):
            pyscx.accel.highly_variable_genes(
                self._adata(), n_top_genes=2, flavor="seurat",
                n_bins=sys.maxsize,
            )

    def test_raw_count_scale_values_raise_not_select_nothing(self):
        """expm1 overflows to Inf around x ≈ 709 — the raw-counts mix-up. One
        overflowing gene used to make every dispersions_norm NaN, selecting
        zero genes with no error (and subset=True would drop every gene)."""
        import pyscx

        with pytest.raises(ValueError, match="log-normalized"):
            pyscx.accel.highly_variable_genes(
                self._adata(values=800.0), n_top_genes=2, flavor="seurat"
            )
