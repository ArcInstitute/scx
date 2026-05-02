"""Tests for improved to_anndata() integration (Phase4 Step 5).

Validates:
- var_names: gene projection at load time
- obs_filter: predicate-based cell filtering via query engine
- layers: selective layer loading
- Combined parameters
- Error cases
"""

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

    with pytest.raises((RuntimeError, ValueError)):
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
    """Numeric and object NumPy arrays in `uns` survive the JSON boundary.

    Regression for issue #4: prior code called json.dumps(adata.uns) which
    raised TypeError on NumPy arrays. STATE/State Designer store HVG names
    as np.ndarray(dtype=object) in adata.uns['X_hvg_var_names']; Scanpy
    writes structured arrays into adata.uns['rank_genes_groups'].
    """
    import pyscx

    rgg = {
        "names": np.array(
            [("g0", "g1"), ("g2", "g0")], dtype=[("A", "U4"), ("B", "U4")]
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

    assert out.uns["X_hvg_var_names"] == ["g0", "g1", "g2"]
    assert out.uns["numeric_arr"] == [[1, 2], [3, 4]]
    assert out.uns["rank_genes_groups"]["names"] == [["g0", "g1"], ["g2", "g0"]]
    assert len(out.uns["rank_genes_groups"]["pvals"]) == 2


def test_from_anndata_uns_numpy_scalars_roundtrip(tmp_dir):
    """NumPy scalars (np.int64, np.float32, np.bool_) collapse to Python scalars."""
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

    assert out.uns["i"] == 42 and isinstance(out.uns["i"], int)
    assert abs(out.uns["f"] - 2.5) < 1e-6 and isinstance(out.uns["f"], float)
    assert out.uns["b"] is True
    assert out.uns["u"] == 7 and isinstance(out.uns["u"], int)


def test_from_anndata_uns_nested_dicts_roundtrip(tmp_dir):
    """Nested dicts mixing dict / list / tuple / NumPy survive the round-trip."""
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
    assert inner["arr"] == [1, 2, 3]
    assert inner["tup"] == [1, "two", 3.5]
    assert [d["k"] for d in inner["list_of_dicts"]] == pytest.approx([0.5, 1.5])


def test_from_anndata_uns_pandas_categorical_roundtrip(tmp_dir):
    """pandas Index / Categorical / Series in `uns` collapse to lists."""
    import pandas as pd
    import pyscx

    adata = _adata_with_uns({
        "idx": pd.Index(["x", "y", "z"]),
        "cat": pd.Categorical(["a", "b", "a"], categories=["a", "b"], ordered=True),
        "series": pd.Series([10, 20, 30]),
    })

    path = str(tmp_dir / "uns_pandas.scx")
    pyscx.from_anndata(adata, path)
    out = pyscx.open(path).to_anndata()

    assert out.uns["idx"] == ["x", "y", "z"]
    assert out.uns["cat"] == ["a", "b", "a"]
    assert out.uns["series"] == [10, 20, 30]


def test_from_anndata_uns_bytes_raises(tmp_dir):
    """`bytes` are not JSON-serializable and should error explicitly."""
    import pyscx

    adata = _adata_with_uns({"k": b"hello"})
    path = str(tmp_dir / "uns_bytes.scx")
    with pytest.raises(ValueError, match=r"uns at uns\['k'\]: bytes are not JSON-serializable"):
        pyscx.from_anndata(adata, path)


def test_from_anndata_uns_non_finite_float_raises(tmp_dir):
    """NaN / inf in floats fail loudly with a key path, even when nested in arrays."""
    import pyscx

    # Top-level NaN
    adata = _adata_with_uns({"top": float("nan")})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['top'\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(adata, str(tmp_dir / "nan.scx"))

    # NaN inside a NumPy array — error path includes the index.
    adata2 = _adata_with_uns({"arr": np.array([1.0, float("nan"), 3.0])})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['arr'\]\[1\]: non-finite float \(NaN\) cannot be serialized",
    ):
        pyscx.from_anndata(adata2, str(tmp_dir / "nan_arr.scx"))


def test_from_anndata_uns_unsupported_type_reports_path(tmp_dir):
    """Unsupported types (e.g. set) raise ValueError naming type and key path."""
    import pyscx

    adata = _adata_with_uns({"outer": {"inner": [1, 2, {3, 4}]}})
    with pytest.raises(
        ValueError,
        match=r"uns at uns\['outer'\]\['inner'\]\[2\]: cannot serialize set to JSON",
    ):
        pyscx.from_anndata(adata, str(tmp_dir / "unsupported.scx"))
