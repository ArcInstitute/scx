"""Zero-copy verification tests (Task 15.13)."""

import numpy as np
import pytest


def test_csr_arrays_have_correct_types(synthetic_adata, tmp_dir):
    """15.13: Verify CSR data array is f32 and indices are i32 (zero-copy from Rust).

    Note: scipy's csr_matrix constructor may internally cast indptr from i64
    to i32 when nnz fits in int32 range. The data and indices arrays should
    preserve Rust dtypes via zero-copy (PyArray1::from_vec).
    """
    import pyscx

    adata = synthetic_adata
    path = str(tmp_dir / "zerocopy.scx")
    pyscx.from_anndata(adata, path)

    adata2 = pyscx.open(path).to_anndata()

    # data and indices preserve exact Rust dtypes
    assert adata2.X.data.dtype == np.float32, "data should be float32"
    assert adata2.X.indices.dtype == np.int32, "indices should be int32"

    # indptr: scipy may downcast from int64 → int32, but must be integer
    assert adata2.X.indptr.dtype in (np.int32, np.int64), "indptr should be integer"


def test_csr_dtypes_match_scipy(synthetic_adata, tmp_dir):
    """Verify CSR arrays have scipy-compatible dtypes."""
    import pyscx

    adata = synthetic_adata
    path = str(tmp_dir / "dtypes.scx")
    pyscx.from_anndata(adata, path)

    adata2 = pyscx.open(path).to_anndata()

    # scipy may normalize indptr to int32 for small matrices; both are valid
    assert adata2.X.indptr.dtype in (np.int32, np.int64)
    assert adata2.X.indices.dtype == np.int32
    assert adata2.X.data.dtype == np.float32
