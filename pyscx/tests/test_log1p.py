"""Tests for pyscx.accel.log1p() — Step 4 of Phase 4d.

Tests verify:
1. log1p on backed ScxBackedSparseDataset creates ScxLazyTransformedDataset
2. log1p on existing ScxLazyTransformedDataset appends transform
3. log1p on scipy CSR delegates to scanpy
4. Output matches scanpy's log1p within tolerance
5. Chained normalize_total + log1p (fused optimization) matches scanpy pipeline
6. Type checks and shape preservation
"""

import numpy as np
import scipy.sparse as sp
import pytest
import tempfile
import os

import pyscx


def _make_test_scx(n_obs=100, n_vars=50, density=0.3, seed=42):
    """Create a temp SCX file with random integer sparse data."""
    X = sp.random(n_obs, n_vars, density=density, random_state=seed,
                  format='csr', dtype=np.float32)
    # Make integer-valued (like UMI counts)
    X.data = np.ceil(X.data * 100).astype(np.float32)
    X.eliminate_zeros()

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test.scx")

    import anndata
    import pandas as pd

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    pyscx.from_anndata(adata, path)
    return path, X


@pytest.fixture
def scx_file():
    """Create a test SCX file and return (path, reference_X)."""
    path, X = _make_test_scx()
    yield path, X


class TestLog1pBacked:
    """Test log1p on ScxBackedSparseDataset."""

    def test_creates_lazy_wrapper(self, scx_file):
        """log1p should replace X with ScxLazyTransformedDataset."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        assert type(adata.X).__name__ == 'ScxBackedSparseDataset'

        pyscx.accel.log1p(adata)

        assert type(adata.X).__name__ == 'ScxLazyTransformedDataset'

    def test_shape_preserved(self, scx_file):
        """Shape should be unchanged after log1p."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        orig_shape = adata.X.shape

        pyscx.accel.log1p(adata)

        assert adata.X.shape == orig_shape

    def test_repr_shows_transform(self, scx_file):
        """Repr should include Log1p."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.log1p(adata)

        r = repr(adata.X)
        assert 'Log1p' in r

    def test_matches_scanpy(self, scx_file):
        """Lazy log1p should match scanpy's log1p."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference: scanpy log1p on materialized data
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.log1p(adata_ref)
        ref_X = adata_ref.X

        # Lazy: pyscx.accel.log1p on backed data
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)

        # Materialize the lazy result
        lazy_X = adata.X.to_memory()

        np.testing.assert_allclose(
            lazy_X.toarray(),
            ref_X.toarray(),
            atol=1e-6,
            err_msg="log1p: lazy vs scanpy mismatch"
        )

    def test_getitem_sliced(self, scx_file):
        """X[5:15] should work on log1p'd data."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.log1p(adata_ref)
        ref_slice = adata_ref.X[5:15]

        # Lazy
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)
        lazy_slice = adata.X[5:15]

        np.testing.assert_allclose(
            lazy_slice.toarray(),
            ref_slice.toarray(),
            atol=1e-6,
            err_msg="sliced access after log1p mismatch"
        )

    def test_values_are_log1p(self, scx_file):
        """After log1p, each stored value should be ln(original + 1)."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)

        lazy_X = adata.X.to_memory()
        expected = X_ref.copy()
        expected.data = np.log1p(expected.data)

        np.testing.assert_allclose(
            lazy_X.toarray(),
            expected.toarray(),
            atol=1e-6,
            err_msg="log1p values should be ln(x + 1)"
        )

    def test_nnz_preserved(self, scx_file):
        """NNZ should be unchanged by log1p (nonzeros stay nonzero)."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        nnz_before = adata.X.getnnz(axis=0)

        pyscx.accel.log1p(adata)

        nnz_after = adata.X.getnnz(axis=0)

        np.testing.assert_array_equal(
            np.array(nnz_before), np.array(nnz_after),
            err_msg="NNZ should be unchanged by log1p"
        )


class TestLog1pChaining:
    """Test chaining log1p with normalize_total."""

    def test_append_to_existing_lazy(self, scx_file):
        """log1p on ScxLazyTransformedDataset should append transform."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        # First: normalize_total
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert type(adata.X).__name__ == 'ScxLazyTransformedDataset'

        # Second: log1p (the common pipeline pattern)
        pyscx.accel.log1p(adata)
        r = repr(adata.X)
        assert 'NormalizeTotal' in r
        assert 'Log1p' in r

    def test_chained_matches_scanpy_pipeline(self, scx_file):
        """Chained normalize_total + log1p should match scanpy pipeline."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference: scanpy pipeline on materialized data
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        ref_X = adata_ref.X

        # Lazy: pyscx.accel pipeline on backed data
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)

        # Materialize the lazy result
        lazy_X = adata.X.to_memory()

        np.testing.assert_allclose(
            lazy_X.toarray(),
            ref_X.toarray(),
            atol=1e-3,
            err_msg="chained normalize_total + log1p: lazy vs scanpy mismatch"
        )

    def test_fused_matches_sequential(self, scx_file):
        """Fused NormalizeTotal+Log1p should produce same results as separate."""
        path, X_ref = scx_file

        # Sequential: apply normalize_total, then log1p manually
        X_seq = X_ref.copy().toarray()
        row_sums = X_seq.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        X_seq = X_seq * (1e4 / row_sums)
        X_seq = np.log1p(X_seq)

        # Fused: use lazy pipeline (internally uses fused optimization)
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)
        lazy_X = adata.X.to_memory()

        np.testing.assert_allclose(
            lazy_X.toarray(),
            X_seq,
            atol=1e-3,
            err_msg="fused normalize+log1p should match sequential"
        )

    def test_double_log1p(self, scx_file):
        """Double log1p (unusual but valid) should chain correctly."""

        path, X_ref = scx_file

        # Reference: double log1p
        expected = X_ref.copy()
        expected.data = np.log1p(np.log1p(expected.data))

        # Lazy: double log1p
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)
        pyscx.accel.log1p(adata)

        r = repr(adata.X)
        assert r.count('Log1p') == 2

        lazy_X = adata.X.to_memory()
        np.testing.assert_allclose(
            lazy_X.toarray(),
            expected.toarray(),
            atol=1e-6,
            err_msg="double log1p mismatch"
        )


class TestLog1pScipy:
    """Test log1p fallback on regular scipy CSR."""

    def test_scipy_fallback(self, scx_file):
        """Should delegate to scanpy for scipy CSR data."""
        import scanpy as sc
        import anndata

        _, X_ref = scx_file

        # Direct scipy CSR in AnnData
        adata = anndata.AnnData(X=X_ref.copy())
        pyscx.accel.log1p(adata)

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.log1p(adata_ref)

        np.testing.assert_allclose(
            adata.X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-6,
            err_msg="scipy CSR fallback mismatch"
        )


class TestLog1pProperties:
    """Test that properties are correct on lazy-transformed data after log1p."""

    def test_issparse_properties(self, scx_file):
        """format, ndim, dtype should be correct after log1p."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)

        assert adata.X.format == 'csr'
        assert adata.X.ndim == 2
        assert str(adata.X.dtype) == 'float32'


class TestStreamingAggAfterLog1p:
    """Test streaming aggregation on log1p'd data."""

    def test_sum_axis1_after_log1p(self, scx_file):
        """sum(axis=1) on log1p'd data should match materialized."""
        path, X_ref = scx_file

        # Reference
        expected = X_ref.copy()
        expected.data = np.log1p(expected.data)
        ref_sums = np.array(expected.sum(axis=1)).flatten()

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)

        sums = np.array(adata.X.sum(axis=1)).flatten()

        np.testing.assert_allclose(
            sums, ref_sums, rtol=1e-5,
            err_msg="streaming sum(axis=1) after log1p mismatch"
        )

    def test_sum_axis0_after_log1p(self, scx_file):
        """sum(axis=0) on log1p'd data should match materialized."""
        path, X_ref = scx_file

        # Reference
        expected = X_ref.copy()
        expected.data = np.log1p(expected.data)
        ref_sums = np.array(expected.sum(axis=0)).flatten()

        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.log1p(adata)

        sums = np.array(adata.X.sum(axis=0)).flatten()

        np.testing.assert_allclose(
            sums, ref_sums, rtol=1e-5,
            err_msg="streaming sum(axis=0) after log1p mismatch"
        )

    def test_sum_after_chained_pipeline(self, scx_file):
        """sum(axis=0) on normalize+log1p should match scanpy."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        ref_sums = np.array(adata_ref.X.sum(axis=0)).flatten()

        # Lazy
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)
        lazy_sums = np.array(adata.X.sum(axis=0)).flatten()

        np.testing.assert_allclose(
            lazy_sums, ref_sums, rtol=1e-3,
            err_msg="streaming sum(axis=0) after normalize+log1p mismatch"
        )

    def test_var_axis0_after_chained_pipeline(self, scx_file):
        """var(axis=0) on normalize+log1p should match materialized."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file

        # Reference
        adata_ref = anndata.AnnData(X=X_ref.copy())
        sc.pp.normalize_total(adata_ref, target_sum=1e4)
        sc.pp.log1p(adata_ref)
        ref_var = np.array(adata_ref.X.toarray().var(axis=0)).flatten()

        # Lazy
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata, target_sum=1e4)
        pyscx.accel.log1p(adata)
        lazy_var = np.array(adata.X.var(axis=0)).flatten()

        np.testing.assert_allclose(
            lazy_var, ref_var, rtol=1e-3,
            err_msg="streaming var(axis=0) after normalize+log1p mismatch"
        )


class TestLog1pColProjectionRegression:
    """Regression tests for B1: log1p must preserve col_projection from backed datasets.

    Previously, log1p() passed None for col_projection when wrapping a backed
    dataset in a lazy transform, causing the user-visible shape to jump back
    to the full gene count after filter_genes() → log1p().
    """

    def test_filter_genes_then_log1p_preserves_shape(self, scx_file):
        """filter_genes() → log1p() should not change shape."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        original_n_vars = adata.shape[1]

        # filter_genes reduces columns
        pyscx.accel.filter_genes(adata, min_cells=1)
        filtered_shape = adata.shape
        assert filtered_shape[1] <= original_n_vars

        # log1p should NOT change the shape
        pyscx.accel.log1p(adata)
        assert adata.X.shape == filtered_shape, (
            f"log1p changed shape from {filtered_shape} to {adata.X.shape} "
            f"(B1 regression: col_projection was dropped)"
        )

    def test_filter_genes_then_log1p_shape_matches_var(self, scx_file):
        """After filter_genes() → log1p(), X.shape[1] must equal len(adata.var)."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.filter_genes(adata, min_cells=1)
        pyscx.accel.log1p(adata)

        assert adata.X.shape[1] == len(adata.var), (
            f"X.shape[1]={adata.X.shape[1]} != len(var)={len(adata.var)} "
            f"(B1 regression: col_projection dropped by log1p)"
        )

    def test_filter_genes_then_log1p_values_correct(self, scx_file):
        """Lazy filter_genes() → log1p() values should match materialized pipeline."""
        import scanpy as sc
        import anndata

        path, X_ref = scx_file
        n_obs, n_vars = X_ref.shape

        # Lazy pipeline
        adata = pyscx.open(path).to_anndata(backed=True)
        pyscx.accel.filter_genes(adata, min_cells=1)
        kept_genes = list(adata.var.index)
        pyscx.accel.log1p(adata)
        lazy_X = adata.X.to_memory()

        # Reference: materialize, subset columns, then log1p
        adata_ref = anndata.AnnData(X=X_ref.copy())
        adata_ref.var_names = [f"gene_{i}" for i in range(n_vars)]
        adata_ref = adata_ref[:, kept_genes].copy()
        sc.pp.log1p(adata_ref)

        np.testing.assert_allclose(
            lazy_X.toarray(),
            adata_ref.X.toarray(),
            atol=1e-6,
            err_msg="filter_genes + log1p: lazy vs materialized mismatch (B1 regression)"
        )

    def test_filter_genes_then_normalize_then_log1p(self, scx_file):
        """Full pipeline: filter_genes → normalize_total → log1p preserves shape."""
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        pyscx.accel.filter_genes(adata, min_cells=1)
        filtered_shape = adata.shape

        pyscx.accel.normalize_total(adata, target_sum=1e4)
        assert adata.X.shape == filtered_shape

        pyscx.accel.log1p(adata)
        assert adata.X.shape == filtered_shape, (
            "Shape changed after log1p in filter→normalize→log1p pipeline"
        )


def _gpu_available() -> bool:
    try:
        return pyscx.accel.gpu_info() is not None
    except Exception:
        return False


@pytest.mark.skipif(
    not _gpu_available(),
    reason="CUDA GPU not available — skipping GPU dispatch test",
)
def test_log1p_gpu_on_scipy_falls_back_to_cpu():
    """`log1p(device="gpu")` on a materialised scipy CSR (no fusion marker,
    no backed/lazy source) emits a `UserWarning` and produces output identical
    to `sc.pp.log1p`. The slow on-device-materialised path was retired —
    catastrophically slow per benchmarks (0.00–0.02× vs CPU)."""
    import anndata
    import scanpy as sc

    rng = np.random.default_rng(0)
    dense = rng.poisson(2.0, size=(200, 100)).astype(np.float32)
    dense[rng.random(dense.shape) > 0.3] = 0
    x = sp.csr_matrix(dense)

    a_gpu = anndata.AnnData(X=x.copy())
    a_ref = anndata.AnnData(X=x.copy())

    with pytest.warns(UserWarning, match="falls back to CPU"):
        pyscx.accel.log1p(a_gpu, device="gpu")

    sc.pp.log1p(a_ref)

    assert sp.issparse(a_gpu.X), "X should remain scipy sparse after fallback"
    np.testing.assert_allclose(
        a_gpu.X.toarray(), a_ref.X.toarray(), rtol=1e-6, atol=1e-7
    )
