"""Tests for ``to_anndata()`` integration.

Validates:
- var_names: gene projection at load time
- obs_filter: predicate-based cell filtering via query engine
- layers: selective layer loading
- Combined parameters
- Error cases
"""

import re

import numpy as np
import pytest


def test_var_names_projects_genes(synthetic_adata, scx_from_adata):
    """to_anndata(var_names=[...]) returns only the requested genes."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames.scx")
    target_genes = ["gene_0", "gene_5", "gene_10"]
    adata = pyscx.open(path).to_anndata(var_names=target_genes)

    assert adata.n_vars == len(target_genes)
    assert adata.n_obs == synthetic_adata.n_obs
    # var index should contain exactly the requested genes
    for gene in target_genes:
        assert gene in adata.var.index.tolist()


def test_var_names_data_correct(synthetic_adata, scx_from_adata):
    """var_names projection produces correct X data."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_data.scx")
    target_genes = ["gene_0", "gene_1"]

    # Load with var_names
    filtered = pyscx.open(path).to_anndata(var_names=target_genes)

    # Load full and manually slice
    full = pyscx.open(path).to_anndata()
    gene_mask = full.var.index.isin(target_genes)
    expected = full[:, gene_mask].copy()

    np.testing.assert_array_equal(
        filtered.X.toarray(), expected.X.toarray()
    )


def test_obs_filter_filters_cells(query_adata, scx_from_adata):
    """to_anndata(obs_filter=...) returns only matching cells."""
    import pyscx

    path = scx_from_adata(query_adata, "obsfilter.scx")
    adata = pyscx.open(path).to_anndata(obs_filter="cell_type == 'T cell'")

    assert adata.n_obs == 40  # 40 T cells in query_adata fixture
    assert all(adata.obs["cell_type"] == "T cell")


def test_obs_filter_matches_query_pipeline(query_adata, scx_from_adata):
    """obs_filter produces same result as the query pipeline."""
    import pyscx

    path = scx_from_adata(query_adata, "obsfilter_qp.scx")

    # to_anndata(obs_filter=...)
    adata_filter = pyscx.open(path).to_anndata(obs_filter="cell_type == 'B cell'")

    # query pipeline
    adata_qp = (
        pyscx.open(path)
        .query()
        .filter_obs("cell_type == 'B cell'")
        .collect()
        .to_anndata()
    )

    assert adata_filter.n_obs == adata_qp.n_obs
    np.testing.assert_array_equal(
        adata_filter.X.toarray(), adata_qp.X.toarray()
    )


def test_obs_filter_with_var_names(query_adata, scx_from_adata):
    """Combined obs_filter + var_names correctly filters both dimensions."""
    import pyscx

    path = scx_from_adata(query_adata, "combined.scx")
    adata = pyscx.open(path).to_anndata(
        obs_filter="cell_type == 'NK cell'",
        var_names=["gene_0", "gene_1", "gene_2"],
    )

    assert adata.n_obs == 40  # 40 NK cells
    assert adata.n_vars == 3
    assert all(adata.obs["cell_type"] == "NK cell")


def test_layers_selective(synthetic_adata, scx_from_adata):
    """to_anndata(layers=[...]) loads only requested layers."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "layers.scx")

    # Load only "raw" layer
    adata = pyscx.open(path).to_anndata(layers=["raw"])
    assert "raw" in adata.layers
    assert adata.n_obs == synthetic_adata.n_obs

    # Load with empty list — no layers
    adata_no_layers = pyscx.open(path).to_anndata(layers=[])
    assert len(adata_no_layers.layers) == 0


def test_layers_backed(synthetic_adata, scx_from_adata):
    """to_anndata(backed=True, layers=[...]) loads only requested layers."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "layers_backed.scx")

    # Load only "raw" layer in backed mode
    adata = pyscx.open(path).to_anndata(backed=True, layers=["raw"])
    assert "raw" in adata.layers

    # Load with empty list — no layers
    adata_no = pyscx.open(path).to_anndata(backed=True, layers=[])
    assert len(adata_no.layers) == 0


def test_obs_filter_backed(query_adata, scx_from_adata):
    """to_anndata(backed=True, obs_filter=...) filters cells in backed mode."""
    import pyscx

    path = scx_from_adata(query_adata, "obsfilter_backed.scx")
    adata = pyscx.open(path).to_anndata(backed=True, obs_filter="cell_type == 'T cell'")

    assert adata.n_obs == 40
    assert all(adata.obs["cell_type"] == "T cell")

    # X should be a backed dataset
    assert isinstance(adata.X, pyscx.ScxBackedSparseDataset)

    # Materialize and verify shape
    x_dense = adata.X.to_memory()
    assert x_dense.shape == (40, 40)


def test_var_names_backed(synthetic_adata, scx_from_adata):
    """backed=True + var_names produces correct shape and var."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_backed.scx")
    target_genes = ["gene_0", "gene_5", "gene_10"]
    adata = pyscx.open(path).to_anndata(backed=True, var_names=target_genes)

    assert adata.n_vars == len(target_genes)
    assert adata.n_obs == synthetic_adata.n_obs
    for gene in target_genes:
        assert gene in adata.var.index.tolist()

    # X should be a ScxBackedSparseDataset
    assert isinstance(adata.X, pyscx.ScxBackedSparseDataset)
    assert adata.X.shape == (synthetic_adata.n_obs, len(target_genes))

    # Materialize and check shape
    x_mat = adata.X.to_memory()
    assert x_mat.shape == (synthetic_adata.n_obs, len(target_genes))


def test_var_names_backed_data_matches_nonbacked(synthetic_adata, scx_from_adata):
    """backed + var_names produces same data as non-backed + var_names."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_backed_match.scx")
    target_genes = ["gene_0", "gene_1"]

    # Non-backed
    eager = pyscx.open(path).to_anndata(var_names=target_genes)
    # Backed
    backed = pyscx.open(path).to_anndata(backed=True, var_names=target_genes)
    backed_x = backed.X.to_memory()

    np.testing.assert_array_equal(backed_x.toarray(), eager.X.toarray())


def test_var_names_backed_combined_obs_filter(query_adata, scx_from_adata):
    """backed + var_names + obs_filter work together."""
    import pyscx

    path = scx_from_adata(query_adata, "backed_combined.scx")
    adata = pyscx.open(path).to_anndata(
        backed=True,
        var_names=["gene_0", "gene_1", "gene_2"],
        obs_filter="cell_type == 'T cell'",
    )

    assert adata.n_obs == 40
    assert adata.n_vars == 3
    assert all(adata.obs["cell_type"] == "T cell")
    assert isinstance(adata.X, pyscx.ScxBackedSparseDataset)
    assert adata.X.shape == (40, 3)


def test_var_names_none_found_raises(synthetic_adata, scx_from_adata):
    """var_names with no matching genes raises error."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "varnames_bad.scx")

    # strict_var_names defaults to True, so an unknown name raises KeyError
    # (before the all-unknown RuntimeError path).
    with pytest.raises((RuntimeError, ValueError, KeyError)):
        pyscx.open(path).to_anndata(var_names=["nonexistent_gene"])


def test_no_params_unchanged(synthetic_adata, scx_from_adata):
    """to_anndata() with no new params produces same result as before."""
    import pyscx

    path = scx_from_adata(synthetic_adata, "no_params.scx")

    adata = pyscx.open(path).to_anndata()
    assert adata.n_obs == synthetic_adata.n_obs
    assert adata.n_vars == synthetic_adata.n_vars
    assert "raw" in adata.layers


def test_to_anndata_deleted_rows_filters_layers(tmp_dir):
    """Eager to_anndata() applies deletion vectors to layers, not just X.

    Regression test for the case where mark_deleted() produced an SCX file
    that the eager reader could no longer load when layers were present
    (X had n_kept rows but layers still had n_obs rows -> AnnData rejected).
    Deletions span at least two shards to exercise per-shard mask handling.
    """
    import anndata
    import pyscx
    import scipy.sparse as sp

    np.random.seed(7)
    n_obs, n_vars = 60, 20
    dense_x = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    dense_x[np.random.random((n_obs, n_vars)) > 0.3] = 0
    dense_raw = np.random.randint(0, 100, size=(n_obs, n_vars)).astype(np.float32)
    dense_raw[np.random.random((n_obs, n_vars)) > 0.4] = 0

    adata = anndata.AnnData(
        X=sp.csr_matrix(dense_x),
        layers={"raw": sp.csr_matrix(dense_raw)},
    )

    path = str(tmp_dir / "layers_del.scx")
    # shard_size=20 -> 3 shards of 20 rows each
    pyscx.from_anndata(adata, path, shard_size=20)

    # Deletions span 3 shards (rows 1, 25, 55 -> shards 0, 1, 2)
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[1] = True
    delete_mask[25] = True
    delete_mask[55] = True
    n_kept = n_obs - int(delete_mask.sum())

    pyscx.open(path).mark_deleted(delete_mask)

    adata_loaded = pyscx.open(path).to_anndata()

    assert adata_loaded.n_obs == n_kept
    assert adata_loaded.X.shape == (n_kept, n_vars)
    assert "raw" in adata_loaded.layers
    assert adata_loaded.layers["raw"].shape == (n_kept, n_vars)

    # Layer values for the kept rows must match the original layer rows
    expected_raw = dense_raw[~delete_mask]
    np.testing.assert_array_equal(
        adata_loaded.layers["raw"].toarray(), expected_raw
    )


def _gene_symbol_adata():
    """AnnData with var.index = ENSG IDs and var['gene_symbol'] = symbols.

    Index values and symbol values are disjoint, so a query against a symbol
    only resolves through the non-index gene_symbol column.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    np.random.seed(123)
    n_obs, n_vars = 30, 8
    # Distinct integer values per (cell, gene) so we can verify column ordering
    dense = (np.arange(n_obs * n_vars).reshape(n_obs, n_vars) + 1).astype(np.float32)

    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(["A"] * 15 + ["B"] * 15)},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(
        {"gene_symbol": [f"SYM_{i}" for i in range(n_vars)]},
        index=[f"ENSG_{i:04d}" for i in range(n_vars)],
    )
    return anndata.AnnData(X=sp.csr_matrix(dense), obs=obs, var=var)


def test_var_names_symbol_column_eager(tmp_dir):
    """Eager to_anndata(var_names=[symbol]) resolves through non-index columns.

    Before the fix, the eager-no-obs_filter path matched only var.index, so
    gene symbols stored in var['gene_symbol'] were silently dropped (or all-
    missing → error). This asserts symbol-column resolution now works and
    that X / var stay aligned.
    """
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "varnames_symbol_eager.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_5", "SYM_2"]  # non-index, intentionally unsorted
    out = pyscx.open(path).to_anndata(var_names=requested)

    assert out.n_vars == 2
    assert out.X.shape == (adata.n_obs, 2)
    assert out.var.shape[0] == 2
    # Sorted by original column position -> SYM_2 then SYM_5
    assert list(out.var["gene_symbol"]) == ["SYM_2", "SYM_5"]
    assert list(out.var.index) == ["ENSG_0002", "ENSG_0005"]
    # X column values match the source columns at positions 2 and 5
    np.testing.assert_array_equal(
        out.X.toarray(), adata.X.toarray()[:, [2, 5]]
    )


def test_var_names_symbol_column_backed(tmp_dir):
    """Backed to_anndata(var_names=[symbol]) keeps X and var aligned.

    Before the fix, X was projected by resolved positional indices but var
    was filtered via var.index.isin(names). When names matched a non-index
    column, var came back empty and AnnData rejected the assembly. This
    asserts the backed path now slices var positionally.
    """
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "varnames_symbol_backed.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_5", "SYM_2"]
    out = pyscx.open(path).to_anndata(backed=True, var_names=requested)

    assert out.n_vars == 2
    assert out.X.shape == (adata.n_obs, 2)
    assert out.var.shape[0] == 2
    assert list(out.var["gene_symbol"]) == ["SYM_2", "SYM_5"]
    assert list(out.var.index) == ["ENSG_0002", "ENSG_0005"]

    # Materialize X and verify column data
    x_dense = out.X[:].toarray() if hasattr(out.X, "to_memory") else out.X.toarray()
    np.testing.assert_array_equal(x_dense, adata.X.toarray()[:, [2, 5]])


def test_var_names_symbol_column_query(tmp_dir):
    """obs_filter + var_names (query-engine path) resolves symbols correctly."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "varnames_symbol_query.scx")
    pyscx.from_anndata(adata, path)

    out = pyscx.open(path).to_anndata(
        obs_filter="cell_type == 'A'",
        var_names=["SYM_5", "SYM_2"],
    )
    assert out.n_obs == 15
    assert out.n_vars == 2
    assert list(out.var["gene_symbol"]) == ["SYM_2", "SYM_5"]
    assert list(out.var.index) == ["ENSG_0002", "ENSG_0005"]


def test_var_names_consistent_across_paths(tmp_dir):
    """Eager / backed / query paths return identical X and var for the same names.

    Locks in the alignment fix: regardless of which path resolves var_names,
    the resulting X column data and var rows must agree.
    """
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "varnames_consistent.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_5", "SYM_2"]

    eager = pyscx.open(path).to_anndata(var_names=requested)
    backed = pyscx.open(path).to_anndata(backed=True, var_names=requested)
    backed_x = (
        backed.X[:].toarray() if hasattr(backed.X, "to_memory") else backed.X.toarray()
    )
    query = pyscx.open(path).to_anndata(
        obs_filter="cell_type == 'A' or cell_type == 'B'", var_names=requested
    )

    np.testing.assert_array_equal(eager.X.toarray(), backed_x)
    np.testing.assert_array_equal(eager.X.toarray(), query.X.toarray())
    assert list(eager.var.index) == list(backed.var.index) == list(query.var.index)
    assert (
        list(eager.var["gene_symbol"])
        == list(backed.var["gene_symbol"])
        == list(query.var["gene_symbol"])
    )


def _expected_cols_for(full, requested):
    """Map requested symbols → their original column positions in `full`."""
    sym_to_pos = {s: i for i, s in enumerate(full.var["gene_symbol"])}
    return [sym_to_pos[s] for s in requested]


def test_preserve_var_order_eager(tmp_dir):
    """preserve_var_order=True returns the eager gene axis in request order."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "preserve_eager.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_5", "SYM_1", "SYM_3"]
    out = pyscx.open(path).to_anndata(var_names=requested, preserve_var_order=True)

    assert list(out.var["gene_symbol"]) == requested
    full = pyscx.open(path).to_anndata()
    np.testing.assert_array_equal(
        out.X.toarray(), full.X.toarray()[:, _expected_cols_for(full, requested)]
    )


def test_preserve_var_order_default_is_sorted(tmp_dir):
    """Without the flag, the gene axis stays in sorted original-column order."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "preserve_default.scx")
    pyscx.from_anndata(adata, path)

    out = pyscx.open(path).to_anndata(var_names=["SYM_5", "SYM_1", "SYM_3"])
    # SYM_i lives at original column i, so sorted order is SYM_1, SYM_3, SYM_5.
    assert list(out.var["gene_symbol"]) == ["SYM_1", "SYM_3", "SYM_5"]


def test_preserve_var_order_backed(tmp_dir):
    """preserve_var_order=True on the backed path: X and var both in request order."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "preserve_backed.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_6", "SYM_0", "SYM_4"]
    out = pyscx.open(path).to_anndata(
        backed=True, var_names=requested, preserve_var_order=True
    )

    assert list(out.var["gene_symbol"]) == requested
    backed_x = (
        out.X[:].toarray() if hasattr(out.X, "to_memory") else out.X.toarray()
    )
    full = pyscx.open(path).to_anndata()
    np.testing.assert_array_equal(
        backed_x, full.X.toarray()[:, _expected_cols_for(full, requested)]
    )


def test_preserve_var_order_backed_row_slice(tmp_dir):
    """Row slicing a presentation-ordered backed dataset preserves column order."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "preserve_backed_slice.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_7", "SYM_2", "SYM_5"]
    out = pyscx.open(path).to_anndata(
        backed=True, var_names=requested, preserve_var_order=True
    )
    full = pyscx.open(path).to_anndata()
    cols = _expected_cols_for(full, requested)
    # A contiguous row slice through the backed reader.
    np.testing.assert_array_equal(
        out.X[3:9].toarray(), full.X.toarray()[3:9][:, cols]
    )


def test_preserve_var_order_obs_filter(tmp_dir):
    """Query-engine path (obs_filter + var_names) honours request order."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "preserve_query.scx")
    pyscx.from_anndata(adata, path)

    requested = ["SYM_5", "SYM_2", "SYM_0"]
    out = pyscx.open(path).to_anndata(
        obs_filter="cell_type == 'A'", var_names=requested, preserve_var_order=True
    )
    assert list(out.var["gene_symbol"]) == requested
    # Column values must follow the same order (cell_type=='A' is the first 15 rows).
    full = pyscx.open(path).to_anndata()
    cols = _expected_cols_for(full, requested)
    np.testing.assert_array_equal(
        out.X.toarray(), full.X.toarray()[:15][:, cols]
    )


def test_preserve_var_order_dedup_first_wins(tmp_dir):
    """Duplicate names collapse to the first occurrence, order preserved."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "preserve_dedup.scx")
    pyscx.from_anndata(adata, path)

    out = pyscx.open(path).to_anndata(
        var_names=["SYM_3", "SYM_1", "SYM_3"], preserve_var_order=True
    )
    assert list(out.var["gene_symbol"]) == ["SYM_3", "SYM_1"]


def test_strict_var_names_default_raises_on_unknown(tmp_dir):
    """strict_var_names defaults to True: an unknown name raises KeyError."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "strict_default.scx")
    pyscx.from_anndata(adata, path)

    with pytest.raises(KeyError):
        pyscx.open(path).to_anndata(var_names=["SYM_1", "NOT_A_GENE"])
    # Backed path is strict too.
    with pytest.raises(KeyError):
        pyscx.open(path).to_anndata(backed=True, var_names=["SYM_1", "NOT_A_GENE"])


def test_strict_var_names_false_drops_unknown(tmp_dir):
    """strict_var_names=False restores the lenient silent-drop behaviour."""
    import pyscx

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "strict_lenient.scx")
    pyscx.from_anndata(adata, path)

    out = pyscx.open(path).to_anndata(
        var_names=["SYM_1", "NOT_A_GENE", "SYM_3"], strict_var_names=False
    )
    assert set(out.var["gene_symbol"]) == {"SYM_1", "SYM_3"}


@pytest.mark.parametrize(
    "op", ["normalize_total", "log1p", "score_genes", "calculate_qc_metrics"]
)
def test_accel_ops_reject_preserve_var_order(tmp_dir, op):
    """Accel ops that gather over the sorted projection must reject a
    presentation-ordered backed dataset rather than silently misalign."""
    import pyscx
    import pyscx.accel as accel

    adata = _gene_symbol_adata()
    path = str(tmp_dir / f"reject_{op}.scx")
    pyscx.from_anndata(adata, path)

    ad = pyscx.open(path).to_anndata(
        backed=True, var_names=["SYM_5", "SYM_1", "SYM_3"], preserve_var_order=True
    )
    with pytest.raises(RuntimeError, match="preserve_var_order"):
        if op == "normalize_total":
            accel.normalize_total(ad)
        elif op == "log1p":
            accel.log1p(ad)
        elif op == "score_genes":
            accel.score_genes(ad, ["SYM_5", "SYM_1"])
        else:
            # This guard is what lets calculate_qc_metrics index its per-column
            # qc_var bitmask by `adata.var` position: it holds only while the
            # visible axis is the sorted projection.
            accel.calculate_qc_metrics(ad)


def test_accel_ops_allow_default_order(tmp_dir):
    """Default (sorted) backed projection still works with accel ops."""
    import pyscx
    import pyscx.accel as accel

    adata = _gene_symbol_adata()
    path = str(tmp_dir / "allow_default.scx")
    pyscx.from_anndata(adata, path)

    ad = pyscx.open(path).to_anndata(backed=True, var_names=["SYM_5", "SYM_1", "SYM_3"])
    # No preserve_var_order → no presentation reorder → accel ops run fine.
    accel.normalize_total(ad)


class _DuckAnnData:
    """Minimal AnnData-like shim for duck-typed validation tests.

    Real ``anndata.AnnData`` rejects mismatched layer shapes at assignment
    time, so we have to bypass it to exercise pyscx's own shape validation.
    """

    def __init__(self, X, obs, var, layers):
        self.X = X
        self.obs = obs
        self.var = var
        self.obsm = {}
        self.uns = {}
        self.layers = layers


def test_from_anndata_bad_layer_shape_returns_value_error(tmp_dir):
    """from_anndata raises ValueError (not panics) on mismatched layer shape.

    Regression: a duck-typed AnnData-like with X.shape=(3,4) but
    layers['bad'].shape=(2,4) used to panic in Rust ('index out of bounds')
    while indexing the layer's indptr in the shard boundary loop.
    """
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    n_obs, n_vars = 3, 4
    fake = _DuckAnnData(
        X=sp.csr_matrix(np.zeros((n_obs, n_vars), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
        layers={
            "bad": sp.csr_matrix(np.zeros((n_obs - 1, n_vars), dtype=np.float32))
        },
    )

    path = str(tmp_dir / "bad_layer.scx")
    with pytest.raises(
        ValueError,
        match=r"Layer 'bad' has shape \(2, 4\), expected \(3, 4\)",
    ):
        pyscx.from_anndata(fake, path)


def _adata_with_uns(uns):
    """Build a minimal AnnData carrying the given `uns` payload."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    n_obs, n_vars = 2, 2
    a = anndata.AnnData(
        X=sp.csr_matrix(np.zeros((n_obs, n_vars), dtype=np.float32)),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_obs)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]),
    )
    a.uns.update(uns)
    return a


def test_from_anndata_uns_numpy_arrays_roundtrip(tmp_dir):
    """Numeric and object NumPy arrays in `uns` survive the JSON boundary
    *and* preserve dtype/shape under the default `uns_format="tagged"`.

    Regression for issue #4: prior code called json.dumps(adata.uns) which
    raised TypeError on NumPy arrays. STATE/State Designer store HVG names
    as np.ndarray(dtype=object) in adata.uns['X_hvg_var_names']; Scanpy
    writes structured arrays into adata.uns['rank_genes_groups'].

    Under `uns_format="tagged"` (default since the review-#10 fix) these
    return as real `np.ndarray` instances with original dtype and shape.
    """
    import pyscx

    rgg = {
        "names": np.array(
            [("g0", "g1"), ("g2", "g0")], dtype=[("A", "U4"), ("B", "U4")]
        ),
        # Object-dtype structured array — the actual shape of scanpy's
        # `rank_genes_groups["names"]`. Regression for B1: the encoder used to
        # `tobytes()` these, serializing object pointers and destroying the
        # strings on disk (silent data loss; reopen then raised ValueError).
        "names_obj": np.array(
            list(zip(["GENEA", "GENEB"], ["GENEC", "GENED"])),
            dtype=[("0", "O"), ("1", "O")],
        ),
        "pvals": np.array([1e-3, 1e-2], dtype=np.float32),
    }
    adata = _adata_with_uns({
        "X_hvg_var_names": np.array(["g0", "g1", "g2"], dtype=object),
        "rank_genes_groups": rgg,
        "numeric_arr": np.array([[1, 2], [3, 4]], dtype=np.int32),
    })

    path = str(tmp_dir / "uns_arrays.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()

    hvg = out.uns["X_hvg_var_names"]
    assert isinstance(hvg, np.ndarray)
    assert hvg.dtype == np.dtype("O")
    assert list(hvg) == ["g0", "g1", "g2"]

    num = out.uns["numeric_arr"]
    assert isinstance(num, np.ndarray)
    assert num.dtype == np.int32
    assert num.shape == (2, 2)
    assert np.array_equal(num, np.array([[1, 2], [3, 4]], dtype=np.int32))

    names = out.uns["rank_genes_groups"]["names"]
    assert isinstance(names, np.ndarray)
    assert names.dtype.names == ("A", "B")
    assert names["A"][0] == "g0" and names["B"][0] == "g1"
    assert names["A"][1] == "g2" and names["B"][1] == "g0"

    pvals = out.uns["rank_genes_groups"]["pvals"]
    assert isinstance(pvals, np.ndarray)
    assert pvals.dtype == np.float32
    assert np.allclose(pvals, [1e-3, 1e-2])

    names_obj = out.uns["rank_genes_groups"]["names_obj"]
    assert isinstance(names_obj, np.ndarray)
    assert names_obj.dtype.names == ("0", "1")
    assert names_obj.dtype["0"] == np.dtype("O")
    assert names_obj["0"][0] == "GENEA" and names_obj["1"][0] == "GENEC"
    assert names_obj["0"][1] == "GENEB" and names_obj["1"][1] == "GENED"


def test_from_anndata_uns_structured_object_with_subarray_field(tmp_dir):
    """PR #242 review (Codex P2): an object-bearing structured uns array with a
    *shaped* numeric subarray field must round-trip — the `fields_json` decode
    previously re-expanded the subarray dims and raised a broadcast error on read.
    Also confirms NaN in a numeric field returns as nan."""
    import pyscx

    rec = np.zeros(2, dtype=[("name", "O"), ("score", "f4", (3,))])
    rec["name"] = ["GENEA", "GENEB"]
    rec["score"] = [[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]]
    adata = _adata_with_uns({"rgg_sub": rec})

    path = str(tmp_dir / "uns_subarray.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata().uns["rgg_sub"]

    assert isinstance(out, np.ndarray)
    assert out.dtype.names == ("name", "score")
    assert out.dtype["score"].shape == (3,)
    assert list(out["name"]) == ["GENEA", "GENEB"]
    assert np.array_equal(out["score"][1], np.array([4.0, 5.0, 6.0], dtype="f4"))
    # NaN survives (JSON null → None → nan in a float field).
    assert out["score"][0][0] == 1.0 and np.isnan(out["score"][0][2])


def test_from_anndata_uns_numpy_scalars_roundtrip(tmp_dir):
    """NumPy scalars retain their dtype under tagged mode."""
    import pyscx

    adata = _adata_with_uns({
        "i": np.int64(42),
        "f": np.float32(2.5),
        "b": np.bool_(True),
        "u": np.uint32(7),
    })

    path = str(tmp_dir / "uns_scalars.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()

    assert out.uns["i"] == 42 and isinstance(out.uns["i"], np.int64)
    assert isinstance(out.uns["f"], np.float32)
    assert out.uns["f"] == np.float32(2.5)
    assert out.uns["b"] is np.True_ or bool(out.uns["b"]) is True
    assert isinstance(out.uns["b"], np.bool_)
    assert out.uns["u"] == 7 and isinstance(out.uns["u"], np.uint32)


def test_from_anndata_uns_nested_dicts_roundtrip(tmp_dir):
    """Nested dicts mix dict / list / tuple / NumPy and round-trip with types intact."""
    import pyscx

    adata = _adata_with_uns({
        "level1": {
            "level2": {
                "arr": np.array([1, 2, 3]),
                "tup": (np.int64(1), "two", 3.5),
                "list_of_dicts": [{"k": np.float32(0.5)}, {"k": np.float32(1.5)}],
            }
        }
    })

    path = str(tmp_dir / "uns_nested.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()

    inner = out.uns["level1"]["level2"]

    assert isinstance(inner["arr"], np.ndarray)
    assert np.array_equal(inner["arr"], np.array([1, 2, 3]))

    # Tuple survives as a tuple, including the inner np.int64.
    assert isinstance(inner["tup"], tuple)
    assert isinstance(inner["tup"][0], np.int64)
    assert inner["tup"][0] == 1
    assert inner["tup"][1] == "two"
    assert inner["tup"][2] == 3.5

    ks = [d["k"] for d in inner["list_of_dicts"]]
    assert all(isinstance(k, np.float32) for k in ks)
    assert ks == [np.float32(0.5), np.float32(1.5)]


def test_from_anndata_uns_pandas_categorical_roundtrip(tmp_dir):
    """pandas Index / Categorical / Series round-trip with name, codes,
    categories, and ordered preserved under tagged mode."""
    import pandas as pd
    import pyscx

    adata = _adata_with_uns({
        "idx": pd.Index(["x", "y", "z"], name="g"),
        "cat": pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True),
        "series": pd.Series([10, 20, 30], name="s"),
    })

    path = str(tmp_dir / "uns_pandas.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()

    idx = out.uns["idx"]
    assert isinstance(idx, pd.Index)
    assert list(idx) == ["x", "y", "z"]
    assert idx.name == "g"

    cat = out.uns["cat"]
    assert isinstance(cat, pd.Categorical)
    assert list(cat) == ["a", "b", "a"]
    assert list(cat.categories) == ["a", "b"]
    assert cat.ordered is True

    s = out.uns["series"]
    assert isinstance(s, pd.Series)
    assert list(s) == [10, 20, 30]
    assert s.name == "s"


def test_from_anndata_uns_bytes_raises(tmp_dir):
    """`bytes` are not JSON-serializable and should error explicitly in both modes."""
    import pyscx

    adata = _adata_with_uns({"k": b"hello"})
    path = str(tmp_dir / "uns_bytes.scx")
    with pytest.raises(ValueError, match=r"uns at uns\['k'\]: bytes are not JSON-serializable"):
        pyscx.from_anndata(adata, path)
    # Same behavior under plain mode.
    with pytest.raises(ValueError, match=r"uns at uns\['k'\]: bytes are not JSON-serializable"):
        pyscx.from_anndata(adata, str(tmp_dir / "uns_bytes_plain.scx"), uns_format="plain")


def test_from_anndata_uns_non_finite_python_float_raises(tmp_dir):
    """Top-level Python `float('nan')` still errors loudly with a key path.

    Tagged mode only preserves NaN/Inf when the value is inside an
    `np.ndarray` (base64-LE bytes); a raw Python scalar has no dtype to
    pin down, so we keep the explicit error for that case.
    """
    import pyscx

    adata = _adata_with_uns({"top": float("nan")})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['top'\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(adata, str(tmp_dir / "nan.scx"))


def test_from_anndata_uns_nan_in_array_roundtrips_under_tagged(tmp_dir):
    """NaN / +Inf / -Inf inside an `np.ndarray` round-trip bit-exact under
    the default tagged mode — the array bytes are stored as base64 so the
    IEEE-754 bit pattern survives the JSON envelope.

    Same payload still raises under `uns_format="plain"` because the
    legacy path goes through `.tolist()` → Python `float` → JSON number,
    and JSON has no representation for NaN/Inf.
    """
    import pyscx

    src = np.array([1.0, float("nan"), float("inf"), -float("inf"), 1.5], dtype=np.float64)
    adata = _adata_with_uns({"arr": src})

    path = str(tmp_dir / "nan_arr_tagged.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()
    rt = out.uns["arr"]
    assert isinstance(rt, np.ndarray)
    assert rt.dtype == np.float64
    # Bit-exact: identical bytes via view(uint64).
    assert np.array_equal(rt.view(np.uint64), src.view(np.uint64))

    # Under plain mode the same payload errors (NaN/Inf rejected).
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['arr'\]\[1\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(
            _adata_with_uns({"arr": src}),
            str(tmp_dir / "nan_arr_plain.scx"),
            uns_format="plain",
        )


def test_from_anndata_uns_unsupported_type_reports_path(tmp_dir):
    """Unsupported types (e.g. set) raise ValueError naming type and key path."""
    import pyscx

    adata = _adata_with_uns({"outer": {"inner": [1, 2, {3, 4}]}})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['outer'\]\['inner'\]\[2\]: cannot serialize set to JSON",
    ):
        pyscx.from_anndata(adata, str(tmp_dir / "unsupported.scx"))


def test_from_anndata_uns_self_referential_dict_raises(tmp_dir):
    """Cycles in `uns` raise ValueError instead of crashing the interpreter.

    Regression: prior to cycle detection a self-referential dict would
    recurse until Rust stack overflow and SIGABRT the Python process —
    a regression vs `json.dumps(check_circular=True)` which raised.
    """
    import pyscx

    cyclic = {}
    cyclic["self"] = cyclic
    adata = _adata_with_uns({"top": cyclic})

    with pytest.raises(ValueError, match=r"circular reference detected"):
        pyscx.from_anndata(adata, str(tmp_dir / "cyclic_dict.scx"))


def test_from_anndata_uns_indirect_cycle_raises(tmp_dir):
    """Indirect cycles (list referencing parent dict) also raise."""
    import pyscx

    parent = {}
    child = [1, 2, parent]
    parent["child"] = child
    adata = _adata_with_uns({"top": parent})

    with pytest.raises(ValueError, match=r"circular reference detected"):
        pyscx.from_anndata(adata, str(tmp_dir / "cyclic_indirect.scx"))


# --------------------------------------------------------------------------
# uns nesting depth
#
# Cycle detection is by object identity, which says nothing about a merely
# *deep* tree: `[[[[...]]]]` passes the cycle guard and then recurses until the
# Rust stack is gone. That is a SIGSEGV, not a Python exception, so nothing
# below can be written as a plain in-process assertion until the cap exists.
#
# A second, quieter half of the same defect needs no crash at all: serde_json's
# serializer has no depth limit while its parser stops at 127 levels, so an
# uncapped writer emits `uns` sections that no reader can ever take back.
# Measured before the fix, on this fixture: 125 nested lists round-tripped,
# 126 wrote successfully and then failed to read with "recursion limit
# exceeded"; in tagged mode, where each tuple costs two JSON levels, the
# boundary was 62 tuples.
# --------------------------------------------------------------------------


def _nested_lists(depth):
    """Exactly `depth` nested lists around a scalar leaf."""
    cur = "leaf"
    for _ in range(depth):
        cur = [cur]
    return cur


def _nested_tuples(depth):
    """Exactly `depth` nested tuples around a scalar leaf.

    The worst case for the cap: under `uns_format="tagged"` a tuple is written
    as `{"__scx_type__": "tuple", "data": [...]}`, so one Python container
    costs *two* JSON levels and the parser's ceiling is reached twice as fast.
    """
    cur = "leaf"
    for _ in range(depth):
        cur = (cur,)
    return cur


def _uns_depth(value):
    """Container levels in a decoded `uns` value."""
    if isinstance(value, dict):
        return 1 + max((_uns_depth(v) for v in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_uns_depth(v) for v in value), default=0)
    return 0


def _probe_max_uns_depth(tmp_dir):
    """The cap, read back out of the error message.

    Deliberately not a literal. `MAX_UNS_DEPTH` lives in Rust, and a test that
    hardcoded 60 here would go on asserting 60 after someone raised it —
    passing while pinning the wrong boundary.
    """
    import pyscx

    adata = _adata_with_uns({"probe": _nested_lists(500)})
    with pytest.raises(ValueError) as exc:
        pyscx.from_anndata(adata, str(tmp_dir / "probe.scx"))
    match = re.search(r"maximum of (\d+) levels", str(exc.value))
    assert match, f"depth error must name its limit, got: {exc.value}"
    return int(match.group(1))


def test_from_anndata_uns_deeply_nested_raises(tmp_dir):
    """Deep-but-acyclic `uns` raises ValueError instead of overflowing.

    Before the cap this wrote without complaint at this depth (the overflow
    needs ~20k+, and killed the interpreter rather than raising), leaving a
    file whose `uns` section no reader could parse.
    """
    import pyscx

    adata = _adata_with_uns({"top": {"bomb": _nested_lists(1000)}})

    with pytest.raises(ValueError, match=r"deeper than the maximum of \d+ levels"):
        pyscx.from_anndata(adata, str(tmp_dir / "deep.scx"))


def test_from_anndata_uns_depth_error_names_the_key_path(tmp_dir):
    """The error locates the offending key, not just the fact of a depth."""
    import pyscx

    adata = _adata_with_uns({"harmless": 1, "guilty": _nested_lists(1000)})

    with pytest.raises(ValueError, match=r"uns\['guilty'\]"):
        pyscx.from_anndata(adata, str(tmp_dir / "deep_path.scx"))


def test_uns_at_max_depth_round_trips_but_one_deeper_raises(tmp_dir):
    """The cap admits exactly what the reader can take back.

    The load-bearing test for the *value* of `MAX_UNS_DEPTH`: nested tuples in
    tagged mode are the 2x-amplifying worst case, so if the cap were set from
    the Python depth alone (the review suggested ~256) this write would
    succeed and the read would fail with "recursion limit exceeded".
    """
    import pyscx

    cap = _probe_max_uns_depth(tmp_dir)

    # The `uns` dict is itself container level 1, so the deepest legal payload
    # is `cap - 1` tuples below it.
    ok = _adata_with_uns({"deep": _nested_tuples(cap - 1)})
    path = str(tmp_dir / "at_cap.scx")
    pyscx.from_anndata(ok, path, uns_format="tagged")
    back = pyscx.open(path).to_anndata()
    assert _uns_depth(back.uns["deep"]) == cap - 1

    too_deep = _adata_with_uns({"deep": _nested_tuples(cap)})
    with pytest.raises(ValueError, match=r"deeper than the maximum"):
        pyscx.from_anndata(too_deep, str(tmp_dir / "over_cap.scx"), uns_format="tagged")


def test_from_anndata_uns_object_array_of_deep_list_raises(tmp_dir):
    """An object-dtype ndarray reaches a *different* walker, also capped.

    `encode_ndarray_tagged` hands object arrays to `pylist_to_string_json_array`,
    which never passes through `normalize_uns_value`'s guard — so a cap on the
    container recursion alone leaves this path overflowing. Verified pre-fix:
    this shape SIGSEGV'd at depth 50000 exactly like the plain-list one.
    """
    import pyscx

    arr = np.empty(1, dtype=object)
    arr[0] = _nested_lists(1000)
    adata = _adata_with_uns({"objarr": arr})

    with pytest.raises(ValueError, match=r"deeper than the maximum"):
        pyscx.from_anndata(adata, str(tmp_dir / "deep_objarr.scx"), uns_format="tagged")


def test_structured_array_object_field_holding_a_deep_list_raises(tmp_dir):
    """Third walker: `pylist_to_json_leaf`, reached only via a recarray.

    A structured dtype with an object field routes each field through
    `pylist_to_json_leaf` rather than `pylist_to_string_json_array`, so neither
    the container guard nor the object-ndarray test above covers it. Added
    after a review credited the test matrix with covering every walker — it
    did not, and this is one of the three it missed.
    """
    import pyscx

    arr = np.empty(1, dtype=[("names", "O"), ("score", "f4")])
    arr["names"][0] = _nested_lists(1000)
    adata = _adata_with_uns({"rec": arr})

    with pytest.raises(ValueError, match=r"deeper than the maximum"):
        pyscx.from_anndata(adata, str(tmp_dir / "deep_recarray.scx"), uns_format="tagged")


def test_deeply_nested_structured_dtype_raises(tmp_dir):
    """Fourth walker: `pytuple_descr_to_json`, over `dtype.descr`.

    numpy will happily nest sub-record dtypes arbitrarily deep, and the descr
    walk is recursive over that nesting — independent of how much *data* the
    array holds. A 1-element array is enough to blow it up.
    """
    import pyscx

    dtype = np.dtype([("leaf", "i4")])
    for i in range(80):
        dtype = np.dtype([(f"l{i}", dtype)])
    adata = _adata_with_uns({"nested_dt": np.zeros(1, dtype=dtype)})

    with pytest.raises(ValueError, match=r"deeper than the maximum"):
        pyscx.from_anndata(adata, str(tmp_dir / "deep_dtype.scx"), uns_format="tagged")


def test_set_uns_rejects_deep_nesting(tmp_dir):
    """The in-place uns surfaces are capped too.

    `set_uns` takes a raw user dict with no AnnData in between, so it reaches
    the same walker with the fewest steps of anything in the API.
    """
    import pyscx

    path = str(tmp_dir / "set_uns.scx")
    pyscx.from_anndata(_adata_with_uns({"ok": 1}), path)

    with pytest.raises(ValueError, match=r"deeper than the maximum"):
        pyscx.set_uns(path, {"bomb": _nested_lists(1000)})


def test_from_h5ad_deep_uns_group_chain_does_not_crash(tmp_dir):
    """An h5ad someone else wrote cannot take the process down.

    This arm involves no pyscx write path at all: `scx-convert`'s h5ad reader
    builds its JSON straight from the HDF5 group tree, so serde_json's parser
    limit never applies. Verified pre-fix: a 30000-deep `/uns` chain SIGSEGV'd
    `read_h5ad_metadata` before any conversion started.

    The depth error travels the ordinary `strict_uns` route, so the default is
    a skipped key with a warning rather than a refused file — one pathological
    key should not make an otherwise fine dataset unconvertible.
    """
    import anndata
    import h5py
    import pyscx
    import scipy.sparse as sp

    h5_path = str(tmp_dir / "deep_uns.h5ad")
    anndata.AnnData(
        X=sp.csr_matrix(np.zeros((2, 2), dtype=np.float32))
    ).write_h5ad(h5_path)
    with h5py.File(h5_path, "a") as f:
        group = f.require_group("uns").require_group("bomb")
        for i in range(500):
            group = group.require_group(f"g{i}")
        group.create_dataset("leaf", data=np.int64(1))

    cap = _probe_max_uns_depth(tmp_dir)

    meta = pyscx.read_h5ad_metadata(h5_path)
    assert _uns_depth(meta.uns) <= cap

    out = str(tmp_dir / "deep_uns.scx")
    pyscx.from_h5ad(h5_path, out)
    # The truncated tree must still be readable — the point of tying the
    # ingest cap to the same constant the writer uses.
    assert _uns_depth(pyscx.open(out).to_anndata().uns) <= cap


def test_from_anndata_uns_oversized_int_raises(tmp_dir):
    """Integers outside [i64::MIN, u64::MAX] raise ValueError with key path."""
    import pyscx

    too_big = (1 << 64) + 1  # > u64::MAX
    adata = _adata_with_uns({"big": too_big})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['big'\]: integer is too large for JSON",
    ):
        pyscx.from_anndata(adata, str(tmp_dir / "uns_big_int.scx"))


def test_from_anndata_uns_non_string_dict_keys_stringified(tmp_dir):
    """Non-string dict keys are stringified via str(k) (documented behavior)."""
    import pyscx

    adata = _adata_with_uns({"by_int": {1: "a", 2: "b"}})

    path = str(tmp_dir / "uns_int_keys.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()

    assert out.uns["by_int"] == {"1": "a", "2": "b"}


# ---------------------------------------------------------------------------
# Issue #10 (review): `uns_format="plain"` regression suite.
#
# The legacy lossy path is preserved as an explicit opt-in so downstream
# pipelines that depend on `list`-typed readback can keep working. These
# tests mirror the historical behavior asserted in the original tests, now
# pinned behind `uns_format="plain"`.
# ---------------------------------------------------------------------------


def test_from_anndata_uns_plain_mode_numpy_arrays_collapse_to_lists(tmp_dir):
    """Under `uns_format="plain"`, NumPy arrays collapse to nested lists."""
    import pyscx

    adata = _adata_with_uns({
        "X_hvg_var_names": np.array(["g0", "g1", "g2"], dtype=object),
        "numeric_arr": np.array([[1, 2], [3, 4]], dtype=np.int32),
    })
    path = str(tmp_dir / "uns_plain_arrays.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    out = pyscx.open(path).to_anndata()
    assert out.uns["X_hvg_var_names"] == ["g0", "g1", "g2"]
    assert out.uns["numeric_arr"] == [[1, 2], [3, 4]]


def test_from_anndata_uns_plain_mode_scalars_collapse(tmp_dir):
    """Under `plain`, NumPy scalars collapse to Python scalars (dtype lost)."""
    import pyscx

    adata = _adata_with_uns({
        "i": np.int64(42),
        "f": np.float32(2.5),
        "b": np.bool_(True),
    })
    path = str(tmp_dir / "uns_plain_scalars.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    out = pyscx.open(path).to_anndata()
    assert isinstance(out.uns["i"], int) and out.uns["i"] == 42
    assert isinstance(out.uns["f"], float) and abs(out.uns["f"] - 2.5) < 1e-6
    assert out.uns["b"] is True


def test_from_anndata_uns_plain_mode_tuple_becomes_list(tmp_dir):
    """Under `plain`, tuples collapse to lists."""
    import pyscx

    adata = _adata_with_uns({"tup": (1, "two", 3.5)})
    path = str(tmp_dir / "uns_plain_tuple.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    out = pyscx.open(path).to_anndata()
    assert out.uns["tup"] == [1, "two", 3.5]


def test_from_anndata_uns_plain_mode_pandas_collapses(tmp_dir):
    """Under `plain`, pandas Index/Categorical/Series collapse to lists."""
    import pandas as pd
    import pyscx

    adata = _adata_with_uns({
        "idx": pd.Index(["x", "y", "z"]),
        "cat": pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True),
        "series": pd.Series([10, 20, 30]),
    })
    path = str(tmp_dir / "uns_plain_pandas.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    out = pyscx.open(path).to_anndata()
    assert out.uns["idx"] == ["x", "y", "z"]
    assert out.uns["cat"] == ["a", "b", "a"]
    assert out.uns["series"] == [10, 20, 30]


def test_from_anndata_uns_plain_mode_nan_in_array_still_raises(tmp_dir):
    """Under `plain`, NaN inside an array still raises because the path
    goes through `.tolist()` → Python float → JSON."""
    import pyscx

    adata = _adata_with_uns({"arr": np.array([1.0, float("nan"), 3.0])})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['arr'\]\[1\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(
            adata, str(tmp_dir / "nan_arr_plain.scx"), uns_format="plain"
        )


def test_from_anndata_uns_invalid_format_raises(tmp_dir):
    """Unknown `uns_format` values raise ValueError."""
    import pyscx

    adata = _adata_with_uns({"a": 1})
    with pytest.raises(ValueError, match=r"invalid uns_format 'weird'"):
        pyscx.from_anndata(adata, str(tmp_dir / "bad.scx"), uns_format="weird")


# ---------------------------------------------------------------------------
# Issue #10 (review): tagged-mode-only round-trip cases.
# ---------------------------------------------------------------------------


def test_from_anndata_uns_tagged_float32_bit_exact(tmp_dir):
    """A `np.float32` array round-trips bit-exact (no f32→f64→f32 drift)."""
    import pyscx

    src = np.array(
        [1.1, 2.2, 3.3, np.float32("inf"), -np.float32("inf"), np.float32("nan")],
        dtype=np.float32,
    )
    adata = _adata_with_uns({"x": src})
    path = str(tmp_dir / "uns_f32_exact.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()
    rt = out.uns["x"]
    assert isinstance(rt, np.ndarray)
    assert rt.dtype == np.float32
    assert rt.shape == src.shape
    # Bit-exact via uint32 view.
    assert np.array_equal(rt.view(np.uint32), src.view(np.uint32))


def test_from_anndata_uns_tagged_reads_plain_file(tmp_dir):
    """A file written with `uns_format="plain"` reads correctly back —
    the auto-detecting reader passes plain JSON through unchanged."""
    import pyscx

    adata = _adata_with_uns({"a": [1, 2, 3], "b": "hi"})
    path = str(tmp_dir / "auto_plain.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    out = pyscx.open(path).to_anndata()
    assert out.uns["a"] == [1, 2, 3]
    assert out.uns["b"] == "hi"


def test_from_anndata_uns_tagged_unknown_marker_warns(tmp_dir):
    """A dict with an unknown `__scx_type__` tag triggers a warning on
    read and is returned verbatim (forward-compat for future schemas).

    The writer doesn't introspect user-supplied dict keys, so we can
    construct the future-tag scenario by handing the writer a plain
    Python dict that happens to look like an envelope.
    """
    import pyscx

    adata = _adata_with_uns({
        "future_marker": {"__scx_type__": "future_thing", "data": 1},
    })
    path = str(tmp_dir / "uns_future.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    with pytest.warns(UserWarning, match=r"unknown __scx_type__ tag 'future_thing'"):
        out = pyscx.open(path).to_anndata()
    assert out.uns["future_marker"] == {"__scx_type__": "future_thing", "data": 1}


def test_from_anndata_uns_known_tag_with_missing_keys_treated_as_plain_dict(tmp_dir):
    """A user dict whose `__scx_type__` matches a known tag (e.g. "ndarray")
    but is missing the envelope's required structural keys must round-trip
    verbatim as a plain dict, with **no** warning.

    Regression for the Codex review on PR #89: the previous read path
    unconditionally dispatched any string-valued `__scx_type__` to the
    matching envelope decoder, which then raised ValueError on missing
    keys. Legitimate user metadata that happens to use our sentinel key
    should be silently passed through.
    """
    import warnings as _warnings

    import pyscx

    # Each tag with a payload that is *not* a valid envelope (missing one or
    # more required keys per `envelope_required_keys`). plain mode is the
    # vehicle because the writer doesn't introspect user dict contents.
    payloads = {
        "fake_ndarray": {"__scx_type__": "ndarray", "foo": 1},
        "fake_scalar": {"__scx_type__": "scalar"},
        "fake_categorical": {"__scx_type__": "categorical", "codes": [0, 1]},
        "fake_index": {"__scx_type__": "pandas.Index", "data": [1, 2]},
    }
    adata = _adata_with_uns(payloads)
    path = str(tmp_dir / "uns_collision.scx")
    pyscx.from_anndata(adata, path, uns_format="plain")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # any warning fails the test
        out = pyscx.open(path).to_anndata()
    for key, original in payloads.items():
        assert out.uns[key] == original, f"plain-dict fallback failed for {key}"


# ---------------------------------------------------------------------------
# Issue #5: ensure_csr() must not mutate caller-owned CSR matrices by default.
# ---------------------------------------------------------------------------


def _unsorted_csr_adata():
    """AnnData whose X is a CSR with explicitly unsorted column indices."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    # Two rows; row 0 has indices [2, 0], row 1 has [3, 1] — both unsorted.
    data = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    indices = np.array([2, 0, 3, 1], dtype=np.int32)
    indptr = np.array([0, 2, 4], dtype=np.int64)
    x = sp.csr_matrix((data, indices, indptr), shape=(2, 4))
    assert x.has_sorted_indices is False  # fixture invariant
    return anndata.AnnData(
        X=x,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )


def test_from_anndata_does_not_mutate_unsorted_csr(tmp_dir):
    """Default in_place=False must leave adata.X.indices/data byte-identical."""
    import pyscx

    adata = _unsorted_csr_adata()
    indices_before = adata.X.indices.copy()
    data_before = adata.X.data.copy()
    indptr_before = adata.X.indptr.copy()

    path = str(tmp_dir / "no_mutate.scx")
    pyscx.from_anndata(adata, path)

    # Caller's CSR is unchanged.
    assert (adata.X.indices == indices_before).all()
    assert (adata.X.data == data_before).all()
    assert (adata.X.indptr == indptr_before).all()
    assert adata.X.has_sorted_indices is False

    # On-disk values must still be correct (writer sorts internally).
    rt = pyscx.open(path).to_anndata()
    expected = np.zeros((2, 4), dtype=np.float32)
    expected[0, 2] = 10.0
    expected[0, 0] = 20.0
    expected[1, 3] = 30.0
    expected[1, 1] = 40.0
    assert np.array_equal(rt.X.toarray(), expected)


def test_from_anndata_in_place_sorts_unsorted_csr(tmp_dir):
    """in_place=True opts in to the historical behavior: sort caller's CSR."""
    import pyscx

    adata = _unsorted_csr_adata()
    assert adata.X.has_sorted_indices is False

    path = str(tmp_dir / "in_place.scx")
    pyscx.from_anndata(adata, path, in_place=True)

    # Caller's CSR has been sorted in place.
    assert adata.X.has_sorted_indices is True
    # Row 0: [2,0] -> [0,2] with data [10,20] -> [20,10]
    assert adata.X.indices.tolist() == [0, 2, 1, 3]
    assert adata.X.data.tolist() == [20.0, 10.0, 40.0, 30.0]


def test_from_anndata_does_not_mutate_unsorted_layer(tmp_dir):
    """Layers with unsorted CSR indices must also be preserved by default."""
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    # X is sorted (so X is unchanged trivially); layer is the interesting case.
    x = sp.csr_matrix(np.eye(2, 4, dtype=np.float32))
    layer = sp.csr_matrix(
        (
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([2, 0, 3, 1], dtype=np.int32),
            np.array([0, 2, 4], dtype=np.int64),
        ),
        shape=(2, 4),
    )
    assert layer.has_sorted_indices is False

    adata = anndata.AnnData(
        X=x,
        obs=pd.DataFrame(index=["c0", "c1"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
        layers={"raw": layer},
    )
    indices_before = adata.layers["raw"].indices.copy()
    data_before = adata.layers["raw"].data.copy()

    path = str(tmp_dir / "layer_no_mutate.scx")
    pyscx.from_anndata(adata, path)

    assert (adata.layers["raw"].indices == indices_before).all()
    assert (adata.layers["raw"].data == data_before).all()
    assert adata.layers["raw"].has_sorted_indices is False


# ---------------------------------------------------------------------------
# Review issue #8: preserve_slots=True opt-in for non-backed obs_filter
# ---------------------------------------------------------------------------


def _adata_with_obsm_and_layer():
    """6×4 AnnData with cell_type column, X_umap obsm, and a `raw` layer."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    x = sp.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 4],
                [5, 0, 0, 0],
                [0, 6, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    raw = sp.csr_matrix(
        np.array(
            [
                [10, 0, 0, 0],
                [0, 20, 0, 0],
                [0, 0, 30, 0],
                [0, 0, 0, 40],
                [50, 0, 0, 0],
                [0, 60, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                ["T cell", "B cell", "T cell", "B cell", "T cell", "NK cell"]
            ),
        },
        index=[f"c{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    obsm = {"X_umap": np.arange(12, dtype=np.float32).reshape(6, 2)}
    return anndata.AnnData(X=x, obs=obs, var=var, obsm=obsm, layers={"raw": raw})


def test_obs_filter_preserve_slots_keeps_obsm_and_layers(tmp_dir):
    """preserve_slots=True with obs_filter materializes obsm + layers."""
    import warnings as warnings_mod

    import pyscx

    adata = _adata_with_obsm_and_layer()
    path = str(tmp_dir / "preserve_slots.scx")
    pyscx.from_anndata(adata, path)

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        result = pyscx.open(path).to_anndata(
            obs_filter="cell_type == 'T cell'", preserve_slots=True
        )

    # obs_filter selects rows 0, 2, 4 (the T cells)
    assert result.n_obs == 3
    assert result.n_vars == 4
    assert list(result.obs["cell_type"]) == ["T cell"] * 3

    # obsm and layers must be present and shape-aligned with filtered X
    assert "X_umap" in result.obsm
    assert result.obsm["X_umap"].shape == (3, 2)
    np.testing.assert_array_equal(
        result.obsm["X_umap"],
        np.array([[0, 1], [4, 5], [8, 9]], dtype=np.float32),
    )
    assert "raw" in result.layers
    assert result.layers["raw"].shape == (3, 4)
    expected_raw = np.array(
        [[10, 0, 0, 0], [0, 0, 30, 0], [50, 0, 0, 0]], dtype=np.float32
    )
    np.testing.assert_array_equal(result.layers["raw"].toarray(), expected_raw)

    # The "non-backed mode" warning must NOT fire when preserve_slots=True.
    assert not any(
        "non-backed mode" in str(w.message) for w in caught
    ), [str(w.message) for w in caught]


def test_obs_filter_preserve_slots_default_drops_with_warning(tmp_dir):
    """Default preserve_slots=False keeps query-engine path + warning."""
    import warnings as warnings_mod

    import pyscx

    adata = _adata_with_obsm_and_layer()
    path = str(tmp_dir / "preserve_slots_default.scx")
    pyscx.from_anndata(adata, path)

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        result = pyscx.open(path).to_anndata(obs_filter="cell_type == 'T cell'")

    # obsm and layers must be dropped
    assert len(result.obsm) == 0
    assert len(result.layers) == 0
    # Documented warning fires (matches "non-backed mode" + lists obsm/layers)
    matching = [w for w in caught if "non-backed mode" in str(w.message)]
    assert matching, [str(w.message) for w in caught]
    msg = str(matching[0].message)
    assert "obsm" in msg or "layers" in msg
    assert "preserve_slots" in msg


def test_obs_filter_preserve_slots_with_var_names(tmp_dir):
    """preserve_slots=True composes with var_names column projection."""
    import pyscx

    adata = _adata_with_obsm_and_layer()
    path = str(tmp_dir / "preserve_slots_varnames.scx")
    pyscx.from_anndata(adata, path)

    result = pyscx.open(path).to_anndata(
        obs_filter="cell_type == 'T cell'",
        var_names=["g0", "g2"],
        preserve_slots=True,
    )

    # Rows: T cells = 3. Cols: g0, g2 (sorted positional, matches existing
    # var_names contract).
    assert result.shape == (3, 2)
    assert list(result.var.index) == ["g0", "g2"]
    # Layers must be column-projected to the same gene set
    assert result.layers["raw"].shape == (3, 2)
    np.testing.assert_array_equal(
        result.layers["raw"].toarray(),
        np.array([[10, 0], [0, 30], [50, 0]], dtype=np.float32),
    )
    # X projection
    np.testing.assert_array_equal(
        result.X.toarray(),
        np.array([[1, 0], [0, 3], [5, 0]], dtype=np.float32),
    )
    # obsm is row-only and is preserved unchanged
    assert result.obsm["X_umap"].shape == (3, 2)


def test_obs_filter_preserve_slots_rejects_non_boolean_expression(tmp_dir):
    """preserve_slots=True must reject obs_filter exprs that don't yield a bool mask.

    Without this guard, a numeric expression like ``"n_counts"`` would be
    handed to AnnData's ``__getitem__`` as positional indices and silently
    reorder rows (or raise an opaque IndexError). The guard surfaces a
    clear PyValueError instead.
    """
    import anndata
    import pandas as pd
    import pyscx
    import pytest
    import scipy.sparse as sp

    x = sp.csr_matrix(np.eye(4, dtype=np.float32))
    obs = pd.DataFrame(
        {"n_counts": np.array([10, 20, 30, 40], dtype=np.int64)},
        index=[f"c{i}" for i in range(4)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    adata = anndata.AnnData(X=x, obs=obs, var=var)

    path = str(tmp_dir / "preserve_slots_non_bool.scx")
    pyscx.from_anndata(adata, path)

    with pytest.raises(ValueError, match="boolean mask"):
        pyscx.open(path).to_anndata(obs_filter="n_counts", preserve_slots=True)


# ---------------------------------------------------------------------------
# Review issue #8: predicate grammar parity between preserve_slots=True
# (pandas.eval) and preserve_slots=False (SCX predicate engine).
# ---------------------------------------------------------------------------


def _adata_for_grammar_parity():
    """8×2 AnnData with cell_type + integer n_counts for predicate testing."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    x = sp.csr_matrix(np.eye(8, 2, dtype=np.float32))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                [
                    "T cell",
                    "B cell",
                    "T cell",
                    "NK cell",
                    "B cell",
                    "T cell",
                    "NK cell",
                    "B cell",
                ]
            ),
            "n_counts": np.array([10, 30, 70, 90, 50, 20, 80, 40], dtype=np.int64),
        },
        index=[f"c{i}" for i in range(8)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(2)])
    return anndata.AnnData(X=x, obs=obs, var=var)


@pytest.mark.parametrize(
    "expr",
    [
        "cell_type == 'T cell'",
        "n_counts > 50",
        "n_counts >= 50",
        "n_counts > 30 and cell_type == 'T cell'",
        "cell_type in ['T cell', 'B cell']",
        "not (cell_type == 'NK cell')",
        "(n_counts > 50) or (cell_type == 'NK cell')",
    ],
)
def test_obs_filter_grammar_parity_common_ground(tmp_dir, expr):
    """Expressions accepted by both the SCX engine and pandas.eval must
    select identical row sets.

    The common-ground subset that users can rely on across both paths:
    comparison operators (==, !=, <, <=, >, >=), keyword-form boolean
    operators (`and` / `or` / `not`), `in [...]` against a bracket-delimited
    list literal, and parenthesised sub-expressions. Divergences are
    documented in docs/scanpy.md (see "Filter Expression Compatibility").

    NOTE: this fixture has **no missing values**, and that is load-bearing for
    the `!=` / `not (...)` cases. Those two operators parse in both grammars
    but do NOT select the same rows once a column has nulls — the engine
    follows SQL (UNKNOWN, so a NULL row does not match) while pandas is
    two-valued. `test_obs_filter_grammar_parity_with_null_categorical` is the
    null-bearing counterpart and deliberately omits them.
    """
    import warnings as warnings_mod

    import pyscx

    adata = _adata_for_grammar_parity()
    path = str(tmp_dir / "grammar_parity.scx")
    pyscx.from_anndata(adata, path)

    eager = pyscx.open(path).to_anndata(obs_filter=expr, preserve_slots=False)
    with warnings_mod.catch_warnings():
        # Suppress the pandas.eval grammar-shift warning emitted by the
        # preserve_slots=True path; the surrounding test is the parity
        # check itself, not the warning assertion.
        warnings_mod.simplefilter("ignore")
        preserved = pyscx.open(path).to_anndata(obs_filter=expr, preserve_slots=True)

    assert list(eager.obs.index) == list(preserved.obs.index), (
        f"obs index mismatch for {expr!r}: "
        f"engine={list(eager.obs.index)} pandas={list(preserved.obs.index)}"
    )


def _adata_for_null_parity():
    """8×2 AnnData whose `cell_type` is missing on rows c1 and c5.

    The shared grammar-parity fixture has no nulls, which is why it never
    caught the engine returning fewer rows than pandas for `or`: with every
    cell annotated, no operand is ever UNKNOWN. c1 is the load-bearing row —
    NULL `cell_type` *and* `n_counts` above every threshold below.
    """
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    x = sp.csr_matrix(np.eye(8, 2, dtype=np.float32))
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(
                [
                    "T cell",
                    None,
                    "T cell",
                    "NK cell",
                    "B cell",
                    None,
                    "NK cell",
                    "B cell",
                ]
            ),
            "n_counts": np.array([10, 90, 70, 30, 50, 20, 80, 40], dtype=np.int64),
        },
        index=[f"c{i}" for i in range(8)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(2)])
    return anndata.AnnData(X=x, obs=obs, var=var)


@pytest.mark.parametrize(
    "expr",
    [
        # The headline case: c1 has a NULL cell_type and n_counts=90, so the
        # right operand alone should select it.
        "(n_counts > 50) or (cell_type == 'NK cell')",
        "(cell_type == 'NK cell') or (n_counts > 50)",  # order-independent
        "cell_type == 'B cell' or n_counts > 60",
        # AND and IN over the same null-bearing column, for contrast.
        "n_counts > 30 and cell_type == 'T cell'",
        "cell_type in ['T cell', 'B cell']",
        "cell_type == 'T cell'",
    ],
)
def test_obs_filter_grammar_parity_with_null_categorical(tmp_dir, expr):
    """Engine and pandas must agree even when an operand column has nulls.

    Restricted to `and` / `or` / `in` / bare comparisons on purpose. `!=` and
    `not (...)` genuinely diverge on a NULL cell — the engine follows SQL
    (UNKNOWN, so the row does not match) while pandas is two-valued
    (`NaN != 'x'` is True, so it does) — and that divergence is documented in
    docs/scanpy.md rather than asserted away here.
    """
    import warnings as warnings_mod

    import pyscx

    adata = _adata_for_null_parity()
    path = str(tmp_dir / "null_parity.scx")
    pyscx.from_anndata(adata, path)

    eager = pyscx.open(path).to_anndata(obs_filter=expr, preserve_slots=False)
    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("ignore")
        preserved = pyscx.open(path).to_anndata(obs_filter=expr, preserve_slots=True)

    assert list(eager.obs.index) == list(preserved.obs.index), (
        f"obs index mismatch for {expr!r}: "
        f"engine={list(eager.obs.index)} pandas={list(preserved.obs.index)}"
    )


def test_obs_filter_or_includes_row_whose_other_operand_is_null(tmp_dir):
    """Ground truth, not just parity: `null OR true` is true.

    Stated against a hand-computed expected set so the check survives even if
    both paths were to regress together.
    """
    import pyscx

    adata = _adata_for_null_parity()
    path = str(tmp_dir / "null_or_truth.scx")
    pyscx.from_anndata(adata, path)

    got = list(
        pyscx.open(path)
        .to_anndata(obs_filter="cell_type == 'B cell' or n_counts > 60", preserve_slots=False)
        .obs.index
    )
    # n_counts > 60: c1 (90), c2 (70), c6 (80).  cell_type == 'B cell': c4, c7.
    # c1's cell_type is NULL, and it is the row the non-Kleene `or` dropped.
    assert got == ["c1", "c2", "c4", "c6", "c7"], (
        f"expected c1 (NULL cell_type, n_counts=90) to match via the right "
        f"operand; got {got}"
    )


@pytest.mark.parametrize(
    "expr,reason",
    [
        (
            "n_counts > 50 & cell_type == 'T cell'",
            "pandas.eval accepts `&` as bitwise-and; SCX requires `and`",
        ),
        (
            "cell_type in ('T cell', 'B cell')",
            "pandas.eval accepts tuple literals; SCX `in` requires `[...]`",
        ),
    ],
)
def test_obs_filter_grammar_divergence_scx_rejects(tmp_dir, expr, reason):
    """Expressions valid in pandas.eval but rejected by the SCX predicate
    engine. These are intentional divergences — the SCX grammar is the
    canonical form for preserve_slots=False / backed=True / cloud
    selective-pull. Users hitting these errors should rewrite the
    expression in the common-ground subset above.
    """
    import warnings as warnings_mod

    import pyscx

    adata = _adata_for_grammar_parity()
    path = str(tmp_dir / "grammar_divergence.scx")
    pyscx.from_anndata(adata, path)

    # SCX path raises; the precise error class is RuntimeError today
    # (predicate parse errors wrap through QueryPipeline).
    with pytest.raises((RuntimeError, ValueError)):
        pyscx.open(path).to_anndata(obs_filter=expr, preserve_slots=False)

    # pandas.eval path accepts the expression (the point of the divergence).
    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("ignore")
        result = pyscx.open(path).to_anndata(obs_filter=expr, preserve_slots=True)
    assert result.n_obs >= 0, reason


def test_obs_filter_preserve_slots_emits_pandas_eval_warning(tmp_dir):
    """preserve_slots=True must surface the pandas.eval grammar shift via
    warnings.warn so notebook users see it without reading docs."""
    import warnings as warnings_mod

    import pyscx

    adata = _adata_for_grammar_parity()
    path = str(tmp_dir / "grammar_warning.scx")
    pyscx.from_anndata(adata, path)

    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        pyscx.open(path).to_anndata(
            obs_filter="cell_type == 'T cell'", preserve_slots=True
        )

    matching = [w for w in caught if "pandas.eval" in str(w.message)]
    assert matching, [str(w.message) for w in caught]
    msg = str(matching[0].message)
    assert "preserve_slots" in msg
    assert "SCX predicate engine" in msg
