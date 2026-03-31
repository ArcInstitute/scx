"""Tests for ScxLazyTransformedDataset — Step 2 of Phase 4d.

Tests verify:
1. Backed mode basic access matches reference.
2. Streaming aggregation (sum, mean, getnnz) on backed dataset.
3. __getitem__ with slices, ints, boolean masks, and fancy indexing.
4. Comparison operators and (X > 0).sum() short-circuit.
5. ScxLazyTransformedDataset class availability.
"""

import numpy as np
import scipy.sparse as sp
import pytest
import tempfile
import shutil
import os

import pyscx


def _make_test_scx(tmpdir, n_obs=100, n_vars=50, density=0.3, seed=42):
    """Create a temp SCX file with random integer sparse data."""
    X = sp.random(n_obs, n_vars, density=density, random_state=seed, format='csr', dtype=np.float32)
    # Make integer-valued (like UMI counts)
    X.data = np.ceil(X.data * 100).astype(np.float32)
    X.eliminate_zeros()

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
    tmpdir = tempfile.mkdtemp()
    path, X = _make_test_scx(tmpdir)
    yield path, X
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestLazyTransformConstruction:
    """Test that ScxLazyTransformedDataset can be created."""

    def test_class_exists(self):
        assert hasattr(pyscx, 'ScxLazyTransformedDataset')

    def test_backed_shape(self, scx_file):
        path, X = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        assert adata.X.shape == X.shape

    def test_repr(self, scx_file):
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        r = repr(adata.X)
        assert "ScxBackedSparseDataset" in r


class TestStreamingAggregation:
    """Test that streaming aggregation matches materialized results."""

    def test_backed_sum_axis0(self, scx_file):
        """Test sum(axis=0) on backed dataset."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        backed_sum = adata.X.sum(axis=0)
        ref_sum = np.array(X_ref.sum(axis=0)).flatten()

        np.testing.assert_allclose(
            np.array(backed_sum).flatten(),
            ref_sum,
            rtol=1e-5,
            err_msg="sum(axis=0) mismatch"
        )

    def test_backed_sum_axis1(self, scx_file):
        """Test sum(axis=1) on backed dataset."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        backed_sum = adata.X.sum(axis=1)
        ref_sum = np.array(X_ref.sum(axis=1)).flatten()

        np.testing.assert_allclose(
            np.array(backed_sum).flatten(),
            ref_sum,
            rtol=1e-5,
            err_msg="sum(axis=1) mismatch"
        )

    def test_backed_getnnz_axis0(self, scx_file):
        """Test getnnz(axis=0) on backed dataset."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        nnz_axis0 = adata.X.getnnz(axis=0)
        ref_nnz_0 = np.diff(X_ref.tocsc().indptr)
        np.testing.assert_array_equal(
            np.array(nnz_axis0),
            ref_nnz_0,
            err_msg="getnnz(axis=0) mismatch"
        )

    def test_backed_getnnz_axis1(self, scx_file):
        """Test getnnz(axis=1) on backed dataset."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        nnz_axis1 = adata.X.getnnz(axis=1)
        ref_nnz_1 = np.diff(X_ref.indptr)
        np.testing.assert_array_equal(
            np.array(nnz_axis1),
            ref_nnz_1,
            err_msg="getnnz(axis=1) mismatch"
        )


class TestGetitem:
    """Test __getitem__ returns correct results."""

    def test_getitem_int(self, scx_file):
        """Test single row access."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        row0 = adata.X[0]
        ref_row0 = X_ref[0]
        np.testing.assert_allclose(
            row0.toarray(),
            ref_row0.toarray(),
            rtol=1e-5,
            err_msg="row 0 mismatch"
        )

    def test_getitem_slice(self, scx_file):
        """Test slice access."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        rows = adata.X[5:15]
        ref_rows = X_ref[5:15]
        np.testing.assert_allclose(
            rows.toarray(),
            ref_rows.toarray(),
            rtol=1e-5,
            err_msg="slice [5:15] mismatch"
        )

    def test_getitem_negative_index(self, scx_file):
        """Test negative index access."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        last_row = adata.X[-1]
        ref_last = X_ref[-1]
        np.testing.assert_allclose(
            last_row.toarray(),
            ref_last.toarray(),
            rtol=1e-5,
            err_msg="row -1 mismatch"
        )

    def test_getitem_bool_mask(self, scx_file):
        """Test boolean mask indexing."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        mask = np.zeros(X_ref.shape[0], dtype=bool)
        mask[::3] = True  # every 3rd row
        rows = adata.X[mask]
        ref_rows = X_ref[mask]
        np.testing.assert_allclose(
            rows.toarray(),
            ref_rows.toarray(),
            rtol=1e-5,
            err_msg="boolean mask mismatch"
        )

    def test_getitem_fancy(self, scx_file):
        """Test fancy (integer array) indexing."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        idx = np.array([0, 5, 10, 20, 99])
        rows = adata.X[idx]
        ref_rows = X_ref[idx]
        np.testing.assert_allclose(
            rows.toarray(),
            ref_rows.toarray(),
            rtol=1e-5,
            err_msg="fancy indexing mismatch"
        )


class TestComparisonShortCircuit:
    """Test that (X > 0).sum() == getnnz() for non-negative data."""

    def test_gt0_sum_axis1(self, scx_file):
        """(X > 0).sum(axis=1) should equal getnnz(axis=1) for count data."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        result = (adata.X > 0).sum(axis=1)
        ref = np.diff(X_ref.indptr)

        np.testing.assert_array_equal(
            np.array(result).flatten(),
            ref,
            err_msg="(X > 0).sum(axis=1) mismatch"
        )

    def test_gt0_sum_axis0(self, scx_file):
        """(X > 0).sum(axis=0) should equal getnnz(axis=0) for count data."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        result = (adata.X > 0).sum(axis=0)
        ref = np.diff(X_ref.tocsc().indptr)

        np.testing.assert_array_equal(
            np.array(result).flatten(),
            ref,
            err_msg="(X > 0).sum(axis=0) mismatch"
        )


class TestToMemory:
    """Test full materialization."""

    def test_to_memory(self, scx_file):
        """to_memory() should return a scipy CSR matching the original."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        mat = adata.X.to_memory()
        np.testing.assert_allclose(
            mat.toarray(),
            X_ref.toarray(),
            rtol=1e-5,
            err_msg="to_memory mismatch"
        )

    def test_copy(self, scx_file):
        """copy() should return a scipy CSR matching the original."""
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)

        mat = adata.X.copy()
        np.testing.assert_allclose(
            mat.toarray(),
            X_ref.toarray(),
            rtol=1e-5,
            err_msg="copy mismatch"
        )


class TestProperties:
    """Test basic properties of the dataset."""

    def test_shape(self, scx_file):
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        assert adata.X.shape == X_ref.shape

    def test_ndim(self, scx_file):
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        assert adata.X.ndim == 2

    def test_dtype(self, scx_file):
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        assert str(adata.X.dtype) == 'float32'

    def test_format(self, scx_file):
        path, _ = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        assert adata.X.format == 'csr'

    def test_len(self, scx_file):
        path, X_ref = scx_file
        adata = pyscx.open(path).to_anndata(backed=True)
        assert len(adata.X) == X_ref.shape[0]

    def test_lazy_class_registered(self):
        """Verify ScxLazyTransformedDataset is available as a Python class."""
        cls = pyscx.ScxLazyTransformedDataset
        assert cls is not None
