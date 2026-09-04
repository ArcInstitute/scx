"""Tests for non-materializing column __getitem__ on backed/lazy datasets.

When X[:, col_array] is called with all-rows selection and an array/mask
column selector, the result should be a new backed/lazy dataset with
col_projection set — NOT a materialized scipy matrix.  This keeps
subsequent aggregation on the f64 streaming path.
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def scx_path(synthetic_adata, tmp_dir):
    """Create an SCX file from synthetic data."""
    import pyscx

    path = str(tmp_dir / "col_proj_getitem.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def reference_dense(synthetic_adata):
    """Dense array of the original data for comparison."""
    return synthetic_adata.X.toarray()


# ---------------------------------------------------------------------------
# ScxBackedSparseDataset: X[:, array] returns backed dataset
# ---------------------------------------------------------------------------


class TestBackedColProjectionGetitem:
    """X[:, col] on ScxBackedSparseDataset returns a new backed dataset."""

    def test_bool_mask_returns_backed(self, scx_path):
        """X[:, bool_mask] returns ScxBackedSparseDataset, not scipy."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        mask = np.zeros(adata.shape[1], dtype=bool)
        mask[[0, 5, 10, 20]] = True

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (adata.shape[0], int(mask.sum()))

    def test_int_array_returns_backed(self, scx_path):
        """X[:, int_array] returns ScxBackedSparseDataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([2, 7, 15, 30, 42], dtype=np.int64)

        result = adata.X[:, cols]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (adata.shape[0], len(cols))

    def test_sum_axis0_uses_streaming(self, scx_path, reference_dense):
        """X[:, mask].sum(axis=0) uses f64 streaming, matches reference."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([1, 5, 10, 25, 40])
        projected = adata.X[:, cols]

        result = np.asarray(projected.sum(axis=0)).ravel()
        expected = reference_dense[:, cols].sum(axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_var_axis0_uses_streaming(self, scx_path, reference_dense):
        """X[:, mask].var(axis=0) uses f64 streaming, matches reference."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([3, 8, 15, 30])
        projected = adata.X[:, cols]

        result = np.asarray(projected.var(axis=0)).ravel()
        expected = reference_dense[:, cols].var(axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_getnnz_axis0_uses_streaming(self, scx_path, reference_dense):
        """X[:, mask].getnnz(axis=0) uses streaming, matches reference."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([0, 10, 20, 30, 40, 49])
        projected = adata.X[:, cols]

        result = np.asarray(projected.getnnz(axis=0))
        expected = (reference_dense[:, cols] != 0).sum(axis=0)
        np.testing.assert_array_equal(result, expected)

    def test_composition(self, scx_path, reference_dense):
        """X[:, mask1][:, mask2] composes projections correctly."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        # First projection: select 10 columns
        cols1 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        proj1 = adata.X[:, cols1]
        assert proj1.shape == (100, 10)

        # Second projection: select 3 of those 10
        cols2 = np.array([1, 4, 8])  # indices into the projected space
        proj2 = proj1[:, cols2]
        assert proj2.shape == (100, 3)

        # Should be equivalent to selecting [5, 20, 40] from original
        result = np.asarray(proj2.sum(axis=0)).ravel()
        expected = reference_dense[:, [5, 20, 40]].sum(axis=0)
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_to_memory_respects_projection(self, scx_path, reference_dense):
        """X[:, mask].to_memory() materializes only projected columns."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([2, 7, 15])
        projected = adata.X[:, cols]

        mat = projected.to_memory()
        assert sp.issparse(mat)
        assert mat.shape == (100, 3)
        np.testing.assert_allclose(
            mat.toarray(), reference_dense[:, cols], rtol=1e-6
        )


# ---------------------------------------------------------------------------
# ScxLazyTransformedDataset: X[:, array] stays lazy
# ---------------------------------------------------------------------------


class TestLazyColProjectionGetitem:
    """After normalize_total+log1p, X[:, col] stays lazy-transformed."""

    def test_lazy_mask_returns_lazy(self, scx_path):
        """X[:, bool_mask] on lazy dataset returns ScxLazyTransformedDataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        pyscx.accel.log1p(adata)

        mask = np.zeros(adata.shape[1], dtype=bool)
        mask[[0, 5, 10, 20]] = True

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)
        assert result.shape == (adata.shape[0], int(mask.sum()))

    def test_lazy_int_array_returns_lazy(self, scx_path):
        """X[:, int_array] on lazy dataset returns ScxLazyTransformedDataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        cols = np.array([2, 7, 15, 30, 42], dtype=np.int64)

        result = adata.X[:, cols]
        assert isinstance(result, pyscx.ScxLazyTransformedDataset)
        assert result.shape == (adata.shape[0], len(cols))

    def test_lazy_sum_axis0_correctness(self, scx_path):
        """X[:, cols].sum(axis=0) on lazy data matches full materialization."""
        import pyscx

        # Lazy path: normalize+log1p then project then sum
        adata_b = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_b)
        pyscx.accel.log1p(adata_b)
        cols = np.array([1, 5, 10, 25, 40])
        projected = adata_b.X[:, cols]
        result = np.asarray(projected.sum(axis=0)).ravel()

        # Reference: materialize full lazy data, then subset columns
        adata_b2 = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata_b2)
        pyscx.accel.log1p(adata_b2)
        full_mat = adata_b2.X[:].toarray()
        expected = full_mat[:, cols].sum(axis=0)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_lazy_composition(self, scx_path):
        """X[:, mask1][:, mask2] composes on lazy dataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)

        cols1 = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
        proj1 = adata.X[:, cols1]
        assert isinstance(proj1, pyscx.ScxLazyTransformedDataset)
        assert proj1.shape == (100, 10)

        cols2 = np.array([2, 5, 9])
        proj2 = proj1[:, cols2]
        assert isinstance(proj2, pyscx.ScxLazyTransformedDataset)
        assert proj2.shape == (100, 3)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases for column projection __getitem__."""

    def test_row_subset_still_materializes(self, scx_path):
        """X[0:10, cols] with row subset still returns scipy (not backed)."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        cols = np.array([1, 5, 10])

        result = adata.X[0:10, cols]
        assert sp.issparse(result)
        assert result.shape == (10, 3)

    def test_scalar_col_still_works(self, scx_path):
        """X[0, 5] still returns a scalar float."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        result = adata.X[0, 5]
        assert isinstance(result, (float, np.floating, int, np.integer))

    def test_empty_mask(self, scx_path):
        """X[:, empty_mask] returns a 0-column dataset."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        mask = np.zeros(adata.shape[1], dtype=bool)

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (100, 0)

    def test_full_mask(self, scx_path, reference_dense):
        """X[:, all_true_mask] returns all columns."""
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        mask = np.ones(adata.shape[1], dtype=bool)

        result = adata.X[:, mask]
        assert isinstance(result, pyscx.ScxBackedSparseDataset)
        assert result.shape == (100, 50)

        sums = np.asarray(result.sum(axis=0)).ravel()
        expected = reference_dense.sum(axis=0)
        np.testing.assert_allclose(sums, expected, rtol=1e-6)


# ---------------------------------------------------------------------------
# REC-7 (PR D): every column selector form projects, never the whole matrix
# ---------------------------------------------------------------------------


def _dense(m):
    return m.toarray() if hasattr(m, "toarray") else np.asarray(m)


class TestWidenedColumnSelectors:
    """`X[:, sel]` on a backed handle: int / list / range / slice / any-order
    ndarray all resolve without a decode. Unique selectors come back as a
    handle (a reordered one carries a presentation permutation); a selector
    with repeats materialises the projected unique columns and gathers — never
    `read_rows(0, n_obs)`."""

    def test_int_column_is_a_one_column_handle(self, scx_path, reference_dense):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        h = X[:, 5]
        assert isinstance(h, pyscx.ScxBackedSparseDataset)
        assert h.shape == (100, 1)
        np.testing.assert_allclose(_dense(h.to_memory()), reference_dense[:, [5]])
        last = X[:, -1]
        assert last.shape == (100, 1)
        np.testing.assert_allclose(_dense(last.to_memory()), reference_dense[:, [49]])

    def test_slice_and_range_are_handles(self, scx_path, reference_dense):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        h = X[:, 10:20]
        assert isinstance(h, pyscx.ScxBackedSparseDataset)
        assert h.shape == (100, 10)
        np.testing.assert_allclose(_dense(h.to_memory()), reference_dense[:, 10:20])
        r = X[:, range(5, 15)]
        assert isinstance(r, pyscx.ScxBackedSparseDataset)
        np.testing.assert_allclose(_dense(r.to_memory()), reference_dense[:, 5:15])
        stepped = X[:, 0:50:7]
        assert isinstance(stepped, pyscx.ScxBackedSparseDataset)
        np.testing.assert_allclose(_dense(stepped.to_memory()), reference_dense[:, 0:50:7])

    def test_list_in_request_order_is_a_handle(self, scx_path, reference_dense):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        h = X[:, [7, 2, 11]]
        assert isinstance(h, pyscx.ScxBackedSparseDataset)
        assert h.shape == (100, 3)
        np.testing.assert_allclose(_dense(h.to_memory()), reference_dense[:, [7, 2, 11]])
        sums = np.asarray(h.sum(axis=0)).ravel()
        np.testing.assert_allclose(sums, reference_dense[:, [7, 2, 11]].sum(axis=0), rtol=1e-6)

    def test_reversed_slice_is_a_presentation_ordered_handle(self, scx_path, reference_dense):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        h = X[:, ::-1]
        assert isinstance(h, pyscx.ScxBackedSparseDataset)
        assert h.shape == (100, 50)
        np.testing.assert_allclose(_dense(h.to_memory()), reference_dense[:, ::-1])

    def test_composition_through_a_presentation_order(self, scx_path, reference_dense):
        """The old fast path was gated off once a presentation order was
        active; composing must now go *through* it."""
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        h = X[:, [7, 2, 11]][:, [2, 0]]
        assert isinstance(h, pyscx.ScxBackedSparseDataset)
        assert h.shape == (100, 2)
        np.testing.assert_allclose(_dense(h.to_memory()), reference_dense[:, [11, 7]])

    def test_repeated_columns_materialise_the_projected_columns(self, scx_path, reference_dense):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        m = X[:, [3, 1, 3]]
        assert sp.issparse(m)
        assert m.shape == (100, 3)
        np.testing.assert_allclose(_dense(m), reference_dense[:, [3, 1, 3]])
        # Through an existing projection too.
        m2 = X[:, [7, 2, 11]][:, [2, 2, 0]]
        assert sp.issparse(m2)
        np.testing.assert_allclose(_dense(m2), reference_dense[:, [11, 11, 7]])

    def test_unsigned_and_empty_selectors(self, scx_path, reference_dense):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        h = X[:, np.array([1, 3], dtype=np.uint64)]
        assert isinstance(h, pyscx.ScxBackedSparseDataset)
        np.testing.assert_allclose(_dense(h.to_memory()), reference_dense[:, [1, 3]])
        empty = X[:, []]
        assert isinstance(empty, pyscx.ScxBackedSparseDataset)
        assert empty.shape == (100, 0)

    def test_invalid_column_selectors_raise_index_error(self, scx_path):
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        with pytest.raises(IndexError, match="1000000"):
            X[:, [0, 10**6]]
        with pytest.raises(IndexError, match="column index -51"):
            X[:, -51]
        with pytest.raises(IndexError, match="boolean column mask"):
            X[:, np.ones(3, dtype=bool)]
        with pytest.raises(IndexError, match="integer array or a boolean mask"):
            X[:, np.array([1.5])]
        with pytest.raises(IndexError, match="one-dimensional"):
            X[:, np.zeros((2, 2), dtype=np.int64)]

    def test_accel_refusal_names_the_handle_level_reorder(self, scx_path):
        """A reordered `X[:, cols]` assigned back to `adata.X` installs the same
        presentation permutation `preserve_var_order=True` does; the accel
        prologue refuses either way, and its message must name this route and
        the `to_memory()` escape rather than blaming a flag that was never set."""
        import anndata
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        order = [7, 2, 11]
        # anndata accepts the handle as X (it is registered as a CSRDataset);
        # the var frame is sliced by hand so the two axes agree.
        sub = anndata.AnnData(X=adata.X[:, order], obs=adata.obs, var=adata.var.iloc[order])
        assert isinstance(sub.X, pyscx.ScxBackedSparseDataset)
        assert sub.X.shape == (100, 3)
        with pytest.raises(RuntimeError, match="reordered through"):
            pyscx.accel.normalize_total(sub)
        with pytest.raises(RuntimeError, match="to_memory"):
            pyscx.accel.normalize_total(sub)
        # The escape the message names works.
        sub.X = sub.X.to_memory()
        pyscx.accel.normalize_total(sub)

    def test_full_column_slice_still_returns_the_matrix(self, scx_path, reference_dense):
        """`X[:, :]` is a copy in scipy; it stays the materialised row CSR."""
        import pyscx

        X = pyscx.open(scx_path).to_anndata(backed=True).X
        m = X[:, :]
        assert sp.issparse(m)
        np.testing.assert_allclose(_dense(m), reference_dense)


class TestLazyWidenedColumnSelectors:
    """The lazy handle has no presentation order: ascending-unique selectors
    project, anything else materialises the projected unique columns and
    gathers (never the whole matrix)."""

    @staticmethod
    def _lazy_and_full(scx_path):
        import pyscx

        adata = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(adata)
        pyscx.accel.log1p(adata)
        ref = pyscx.open(scx_path).to_anndata(backed=True)
        pyscx.accel.normalize_total(ref)
        pyscx.accel.log1p(ref)
        return adata.X, ref.X[:].toarray()

    def test_int_slice_and_list_project(self, scx_path):
        import pyscx

        X, full = self._lazy_and_full(scx_path)
        h = X[:, 5]
        assert isinstance(h, pyscx.ScxLazyTransformedDataset)
        assert h.shape == (100, 1)
        np.testing.assert_allclose(_dense(h.to_memory()), full[:, [5]], rtol=1e-6)
        s = X[:, 10:20]
        assert isinstance(s, pyscx.ScxLazyTransformedDataset)
        np.testing.assert_allclose(_dense(s.to_memory()), full[:, 10:20], rtol=1e-6)
        lst = X[:, [1, 3]]
        assert isinstance(lst, pyscx.ScxLazyTransformedDataset)
        np.testing.assert_allclose(_dense(lst.to_memory()), full[:, [1, 3]], rtol=1e-6)

    def test_reorder_and_repeats_materialise_projected_columns(self, scx_path):
        X, full = self._lazy_and_full(scx_path)
        m = X[:, [7, 2]]
        assert sp.issparse(m)
        np.testing.assert_allclose(_dense(m), full[:, [7, 2]], rtol=1e-6)
        d = X[:, [3, 1, 3]]
        assert sp.issparse(d)
        assert d.shape == (100, 3)
        np.testing.assert_allclose(_dense(d), full[:, [3, 1, 3]], rtol=1e-6)

    def test_invalid_selectors_raise_index_error(self, scx_path):
        X, _ = self._lazy_and_full(scx_path)
        with pytest.raises(IndexError, match="1000000"):
            X[:, [0, 10**6]]
        with pytest.raises(IndexError, match="integer array or a boolean mask"):
            X[:, np.array([1.5])]


class TestLayerColumnSelectors:
    """`adata.layers[k][:, cols]` keeps the layer wrapper (and its name)."""

    def test_layer_projection_keeps_the_wrapper(self, scx_path, synthetic_adata):
        import pyscx

        layer = pyscx.open(scx_path).to_anndata(backed=True).layers["raw"]
        ref = synthetic_adata.layers["raw"].toarray()
        h = layer[:, [1, 3]]
        assert isinstance(h, pyscx.ScxBackedLayerDataset)
        assert h.layer_name == "raw"
        assert h.shape == (100, 2)
        np.testing.assert_allclose(_dense(h.to_memory()), ref[:, [1, 3]])
        one = layer[:, 5]
        assert isinstance(one, pyscx.ScxBackedLayerDataset)
        np.testing.assert_allclose(_dense(one.to_memory()), ref[:, [5]])
        mask = np.zeros(50, dtype=bool)
        mask[[2, 4]] = True
        assert isinstance(layer[:, mask], pyscx.ScxBackedLayerDataset)
        # Rows still gather to scipy through the wrapper.
        assert sp.issparse(layer[0:5])
