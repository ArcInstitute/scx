"""Integration tests for SCX backed mode (Phase 5).

Tests the backed (lazy-loading) AnnData integration, validating:
- Round-trip correctness (backed X[:] matches non-backed)
- Row slicing, fancy indexing, boolean masks, column filtering
- to_memory() materializationfrom backed
- Layer backed access
- Deletion vector support
- anndata.abc.CSRDataset isinstance check
- Full scanpy pipeline on backed data
- Memory efficiency vs full materialization
- Cache configuration options
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_scx(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic_adata for backed tests."""
    import pyscx

    path = str(tmp_dir / "backed_test.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def adata_non_backed(backed_scx):
    """Load the full (non-backed) AnnData for comparison."""
    import pyscx

    return pyscx.open(backed_scx).to_anndata()


@pytest.fixture
def adata_backed(backed_scx):
    """Load the backed AnnData."""
    import pyscx

    return pyscx.open(backed_scx).to_anndata(backed=True)


def test_backed_roundtrip(adata_backed, adata_non_backed):
    """Backed X[:] matches non-backed X element-wise."""
    backed_full = adata_backed.X[:]
    assert isinstance(backed_full, sp.csr_matrix)
    assert backed_full.shape == adata_non_backed.X.shape

    # Element-wise comparison
    np.testing.assert_array_equal(
        backed_full.toarray(), adata_non_backed.X.toarray()
    )


def test_backed_row_slice(adata_backed, adata_non_backed):
    """X[10:20] matches non-backed equivalent."""
    backed_slice = adata_backed.X[10:20]
    expected = adata_non_backed.X[10:20]

    assert backed_slice.shape == expected.shape
    np.testing.assert_array_equal(backed_slice.toarray(), expected.toarray())


def test_backed_fancy_index(adata_backed, adata_non_backed):
    """X[[0, 5, 10]] returns correct rows."""
    indices = [0, 5, 10]
    backed_fancy = adata_backed.X[indices]
    expected = adata_non_backed.X[indices]

    assert backed_fancy.shape == expected.shape
    np.testing.assert_array_equal(backed_fancy.toarray(), expected.toarray())


def test_backed_boolean_mask(adata_backed, adata_non_backed):
    """X[mask] returns correct subset."""
    n_obs = adata_non_backed.X.shape[0]
    np.random.seed(42)
    mask = np.random.choice([True, False], size=n_obs)

    backed_masked = adata_backed.X[mask]
    expected = adata_non_backed.X[mask]

    assert backed_masked.shape == expected.shape
    np.testing.assert_array_equal(backed_masked.toarray(), expected.toarray())


def test_backed_column_slice(adata_backed, adata_non_backed):
    """X[10:20, :500] filters columns correctly."""
    # Note: adata_non_backed has 50 vars, so use :25
    backed_2d = adata_backed.X[10:20, :25]
    expected = adata_non_backed.X[10:20, :25]

    assert backed_2d.shape == expected.shape
    np.testing.assert_array_equal(backed_2d.toarray(), expected.toarray())


def test_backed_to_memory(adata_backed, adata_non_backed):
    """to_memory() matches full to_anndata()."""
    materialized = adata_backed.X.to_memory()
    assert isinstance(materialized, sp.csr_matrix)
    np.testing.assert_array_equal(
        materialized.toarray(), adata_non_backed.X.toarray()
    )


def test_backed_layers(backed_scx, adata_non_backed):
    """Layer backed access works and matches non-backed."""
    import pyscx

    adata = pyscx.open(backed_scx).to_anndata(backed=True)

    # Check that the layer exists and is backed
    assert "raw" in adata.layers
    layer_type = type(adata.layers["raw"]).__name__
    assert "Backed" in layer_type or "Dataset" in layer_type

    # Full slice should match non-backed
    backed_layer = adata.layers["raw"][:]
    expected_layer = adata_non_backed.layers["raw"]

    np.testing.assert_array_equal(
        backed_layer.toarray(), expected_layer.toarray()
    )

    # Row slice should match
    backed_layer_slice = adata.layers["raw"][5:15]
    expected_layer_slice = adata_non_backed.layers["raw"][5:15]
    np.testing.assert_array_equal(
        backed_layer_slice.toarray(), expected_layer_slice.toarray()
    )


def test_backed_with_deletions(tmp_dir):
    """Backed mode on file with mark_deleted() excludes deleted rows."""
    import anndata
    import pyscx

    np.random.seed(99)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / "backed_del.scx")
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True
    n_deleted = int(delete_mask.sum())

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)

    # Non-backed should exclude deleted rows
    adata_full = pyscx.open(path).to_anndata()
    assert adata_full.n_obs == n_obs - n_deleted

    # Backed should also exclude deleted rows
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.X.shape[0] == n_obs - n_deleted

    # Backed full read should match non-backed
    backed_full = adata_backed.X[:]
    np.testing.assert_array_equal(
        backed_full.toarray(), adata_full.X.toarray()
    )


def test_backed_obsm_with_deletions(tmp_dir):
    """obsm arrays are filtered by deletion vectors (shape matches obs)."""
    import anndata
    import pyscx

    np.random.seed(42)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    obsm = {"X_pca": np.random.randn(n_obs, 10).astype(np.float32)}
    adata = anndata.AnnData(X=x, obsm=obsm)

    path = str(tmp_dir / "backed_obsm_del.scx")
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True
    n_deleted = int(delete_mask.sum())
    n_kept = n_obs - n_deleted

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)

    # Non-backed: obs and obsm should have matching shapes
    adata_full = pyscx.open(path).to_anndata()
    assert adata_full.n_obs == n_kept
    assert adata_full.obsm["X_pca"].shape[0] == n_kept

    # Backed: obs and obsm should also have matching shapes
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    assert adata_backed.n_obs == n_kept
    assert adata_backed.obsm["X_pca"].shape[0] == n_kept

    # obsm values should match between backed and non-backed
    np.testing.assert_array_equal(
        adata_backed.obsm["X_pca"], adata_full.obsm["X_pca"]
    )


def test_backed_isinstance(adata_backed):
    """isinstance(adata.X, anndata.abc.CSRDataset) is True."""
    import anndata.abc

    assert isinstance(adata_backed.X, anndata.abc.CSRDataset)


def test_backed_scanpy_pipeline(backed_scx):
    """Full scanpy pipeline works with backed mode.

    Backed mode is read-only, so we subset → copy → preprocess.
    """
    import pyscx
    import scanpy as sc

    adata = pyscx.open(backed_scx).to_anndata(backed=True)

    # Subset (first 50 cells) and materialize
    adata_sub = adata[:50].copy()

    # Standard scanpy pipeline on materialized data
    sc.pp.normalize_total(adata_sub, target_sum=1e4)
    sc.pp.log1p(adata_sub)
    sc.pp.pca(adata_sub)

    # Verify PCA ran
    assert "X_pca" in adata_sub.obsm
    assert adata_sub.obsm["X_pca"].shape[0] == 50


def test_backed_memory(synthetic_adata, tmp_dir):
    """Backed mode peak memory << full materialization.

    We verify that the backed object itself is lightweight by checking
    that the X attribute does not hold a materialized array.
    """
    import sys

    import pyscx

    path = str(tmp_dir / "backed_mem.scx")
    pyscx.from_anndata(synthetic_adata, path)

    # Backed: X is a lightweight proxy object
    adata_backed = pyscx.open(path).to_anndata(backed=True)
    x_backed_size = sys.getsizeof(adata_backed.X)

    # Non-backed: X is a full scipy sparse matrix
    adata_full = pyscx.open(path).to_anndata()
    # The full X stores actual data arrays
    full_data_size = (
        adata_full.X.data.nbytes
        + adata_full.X.indices.nbytes
        + adata_full.X.indptr.nbytes
    )

    # Backed proxy should be much smaller than the actual data
    assert x_backed_size < full_data_size


def test_backed_cache_config(synthetic_adata, tmp_dir):
    """cache_shards=0 and cache_shards=16 both work."""
    import pyscx

    path = str(tmp_dir / "backed_cache.scx")
    pyscx.from_anndata(synthetic_adata, path)

    # No cache
    adata0 = pyscx.open(path).to_anndata(backed=True, cache_shards=0)
    slice0 = adata0.X[0:10]
    assert slice0.shape[0] == 10

    # Large cache
    adata16 = pyscx.open(path).to_anndata(backed=True, cache_shards=16)
    slice16 = adata16.X[0:10]
    assert slice16.shape[0] == 10

    # Both should return the same data
    np.testing.assert_array_equal(slice0.toarray(), slice16.toarray())


# ---------------------------------------------------------------------------
# is_backed_handle — the public "is this lazy?" predicate
# ---------------------------------------------------------------------------


def test_is_backed_handle_covers_all_four_handle_classes(tmp_dir):
    """The predicate covers every handle class that can appear on an AnnData —
    `X`, a layer, an aligned `obsm` store, and a lazily-transformed `X`.

    These drift silently otherwise: a fifth handle class would get its anndata
    seams registered (`HANDLE_CLASSES` in `pyscx/src/anndata_hooks.rs`) while
    every caller branching on this predicate quietly took the wrong arm.
    """
    import anndata
    import pyscx

    rng = np.random.RandomState(1)
    x = (rng.random_sample((60, 20)) * 100).astype(np.float32)
    x[rng.random_sample((60, 20)) > 0.5] = 0.0
    adata = anndata.AnnData(
        X=sp.csr_matrix(x),
        layers={"counts": sp.csr_matrix(x)},
        obsm={"X_emb": rng.random_sample((60, 4)).astype(np.float32)},
    )
    path = str(tmp_dir / "handles.scx")
    pyscx.from_anndata(adata, path)

    # `obsm=[…]` is what keeps an aligned store lazy; without it obsm
    # materializes to an ndarray and the third assertion below would be vacuous.
    backed = pyscx.open(path).to_anndata(backed=True, obsm=["X_emb"])
    seen = {
        type(backed.X).__name__,
        type(backed.layers["counts"]).__name__,
        type(backed.obsm["X_emb"]).__name__,
    }
    assert seen == {
        "ScxBackedSparseDataset",
        "ScxBackedLayerDataset",
        "ScxBackedObsmDataset",
    }, f"fixture no longer yields the three handle types, got {seen}"
    assert pyscx.is_backed_handle(backed.X)
    assert pyscx.is_backed_handle(backed.layers["counts"])
    assert pyscx.is_backed_handle(backed.obsm["X_emb"])

    # The fourth: stacking a lazy transform swaps X for a different class.
    transformed = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.normalize_total(transformed, target_sum=1e4)
    assert type(transformed.X).__name__ == "ScxLazyTransformedDataset"
    assert pyscx.is_backed_handle(transformed.X)


def test_is_backed_handle_false_for_in_memory_matrices(adata_non_backed):
    """A materialized matrix is not a handle — this is the whole point of the
    predicate, so the negative case matters as much as the positive."""
    import pyscx

    x = adata_non_backed.X
    assert not pyscx.is_backed_handle(x)
    assert not pyscx.is_backed_handle(x.toarray())
    assert not pyscx.is_backed_handle(None)
    assert not pyscx.is_backed_handle("not a matrix")


def test_issparse_is_false_on_the_handle_but_true_on_a_slice(adata_backed):
    """Pins the trap the predicate exists for.

    `scipy.sparse.issparse` cannot be made true for a handle — `sparray` /
    `spmatrix` are concrete classes, not ABCs, so there is no `register()`
    seam. The idiomatic `if sp.issparse(X)` guard therefore takes the *dense*
    arm for a handle, silently. Slicing yields genuine scipy sparse, so the
    reliable tests are: this predicate on the matrix, `issparse` on a slice.
    """
    import pyscx

    x = adata_backed.X
    assert pyscx.is_backed_handle(x)
    assert not sp.issparse(x)
    assert sp.issparse(x[0:5])
    # And the documented escape from a handle to something `issparse` accepts.
    assert sp.issparse(x.to_memory())


@pytest.mark.parametrize("which", ["X", "layer", "lazy"])
def test_asarray_on_a_sparse_handle_raises(backed_scx, which):
    """The other half of the trap, closed: `np.asarray` on a sparse handle
    used to return a 0-d object array that failed far away ("setting an array
    element with a sequence"). It now raises `TypeError` naming the explicit
    paths, because the alternative — decoding n_obs × n_vars silently — is
    the worse failure at atlas scale. Every entry point numpy uses goes
    through `__array__`, so `np.array`, a dtype request and `copy=False` all
    refuse the same way."""
    import pyscx

    adata = pyscx.open(backed_scx).to_anndata(backed=True)
    if which == "X":
        h = adata.X
    elif which == "layer":
        h = adata.layers["raw"]
    else:
        pyscx.accel.normalize_total(adata)
        h = adata.X
        assert isinstance(h, pyscx.ScxLazyTransformedDataset)

    for call in (
        lambda: np.asarray(h),
        lambda: np.array(h),
        lambda: np.asarray(h, dtype=np.float32),
        lambda: h.__array__(copy=False),
    ):
        with pytest.raises(TypeError, match="to_memory"):
            call()
    # The explicit paths still work.
    assert sp.issparse(h.to_memory())
    assert isinstance(h.toarray(), np.ndarray)


def test_cache_shards_is_readable_on_every_handle(synthetic_adata, tmp_dir):
    """`cache_shards` is fixed per `to_anndata` call — X and each layer get
    their own reader, built with the same count; the getter reads it back from
    the reader (0 is the uncached path, not clamped)."""
    import pyscx

    path = str(tmp_dir / "cache_shards_getter.scx")
    pyscx.from_anndata(synthetic_adata, path)

    adata = pyscx.open(path).to_anndata(backed=True, cache_shards=7, obsm=["X_pca"])
    assert adata.X.cache_shards == 7
    assert adata.layers["raw"].cache_shards == 7
    assert type(adata.obsm["X_pca"]).__name__ == "ScxBackedObsmDataset"
    assert adata.obsm["X_pca"].cache_shards == 7
    pyscx.accel.normalize_total(adata)
    assert isinstance(adata.X, pyscx.ScxLazyTransformedDataset)
    assert adata.X.cache_shards == 7

    adata0 = pyscx.open(path).to_anndata(backed=True, cache_shards=0)
    assert adata0.X.cache_shards == 0
    assert adata0.layers["raw"].cache_shards == 0


# ---------------------------------------------------------------------------
# The .pyi stubs vs the runtime classes
# ---------------------------------------------------------------------------

# Specials a caller can plausibly reach on a matrix-like handle. Anything in
# this set that a Rust class implements must be declared in the stub, because
# `__getattr__` does not reach special methods — Python looks them up on the
# type, so an omitted one is a type error on documented usage (`X[0:10]`).
_MATRIX_SPECIALS = frozenset(
    {
        "__getitem__",
        "__len__",
        "__array__",
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__truediv__",
        "__rtruediv__",
        "__matmul__",
        "__rmatmul__",
    }
)

_HANDLE_CLASSES_FOR_STUBS = (
    "ScxBackedSparseDataset",
    "ScxBackedLayerDataset",
    "ScxBackedObsmDataset",
    "ScxLazyTransformedDataset",
)


# Shared with `test_experiment_stub_coverage.py` -- one parse of the stub, not two.
from _stub_ast import stub_class_methods  # noqa: E402


@pytest.mark.parametrize("cls_name", _HANDLE_CLASSES_FOR_STUBS)
def test_handle_stubs_declare_every_runtime_special(cls_name):
    """Both directions, because each failure mode is silent in its own way.

    Missing a special the class really has → a type error on working code (the
    `X[0:10]` regression). Declaring one it does not have → a type checker
    green-lights a call that raises at runtime. The repo has no mypy gate, so
    this pytest is what pins it.
    """
    import pyscx

    cls = getattr(pyscx, cls_name)
    declared = set(stub_class_methods(cls_name))

    # `hasattr` is the wrong probe and was the bug in this test's first
    # version: `object` supplies `__eq__`, `__ne__` and all four ordering
    # slots, so `hasattr(cls, "__lt__")` is True for a class that implements no
    # comparison at all. That blinded the check in both directions at once --
    # it credited `ScxBackedObsmDataset` with comparisons it does not have
    # (letting `obsm > 0` type-check and then raise) while the real
    # `__eq__`/`__ne__` on the sparse classes went unnoticed. Count a special
    # as implemented only when it is not the inherited slot.
    runtime = {
        name
        for name in _MATRIX_SPECIALS
        if getattr(cls, name, None) is not None
        and getattr(cls, name, None) is not getattr(object, name, None)
    }
    stubbed = _MATRIX_SPECIALS & declared

    assert runtime - stubbed == set(), (
        f"{cls_name} implements {sorted(runtime - stubbed)} but __init__.pyi "
        "does not declare them; `__getattr__` does not cover special methods, "
        "so these are type errors on valid code"
    )
    assert stubbed - runtime == set(), (
        f"__init__.pyi declares {sorted(stubbed - runtime)} on {cls_name}, "
        "which the Rust class does not implement"
    )
