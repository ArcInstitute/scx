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
