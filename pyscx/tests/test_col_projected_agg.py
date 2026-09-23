"""Integration tests for column-projected streaming aggregation.

Tests that sum(), mean(), var(), getnnz(), max(), min() work correctly
when a col_projection is active on ScxBackedSparseDataset, streaming
through projected shards without materializing the full matrix.
"""

import numpy as np
import pytest
import scipy.sparse as sp


@pytest.fixture
def backed_projected_data(synthetic_adata, tmp_dir):
    """Create an SCX file and return backed adata with col_projection set."""
    import pyscx

    path = str(tmp_dir / "proj_agg.scx")
    pyscx.from_anndata(synthetic_adata, path)
    return path


@pytest.fixture
def col_subset():
    """A deterministic subset of column indices for projection."""
    np.random.seed(123)
    n_vars = 50
    # Select ~10 columns
    return sorted(np.random.choice(n_vars, size=10, replace=False).tolist())


# ---------------------------------------------------------------------------
# Basic column-projected aggregation (no deletions)
# ---------------------------------------------------------------------------


def test_col_sums_projected(backed_projected_data, col_subset):
    """col sums with projection matches materialized subset col sums."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    # Create a column-projected view
    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    # Streaming projected col sums
    projected_sums = np.asarray(backed_x.sum(axis=0)).flatten()

    # Reference: materialize and subset
    full_x = adata_full.X.toarray()
    expected_sums = full_x[:, col_subset].sum(axis=0)

    np.testing.assert_allclose(projected_sums, expected_sums, rtol=1e-6)


def test_col_nnz_projected(backed_projected_data, col_subset):
    """col nnz with projection matches materialized subset col nnz."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_nnz = np.asarray(backed_x.getnnz(axis=0)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_nnz = np.asarray(full_x.getnnz(axis=0)).flatten()

    np.testing.assert_array_equal(projected_nnz, expected_nnz)


def test_col_var_projected(backed_projected_data, col_subset):
    """col var with projection matches materialized subset col var."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_var = np.asarray(backed_x.var(axis=0)).flatten()

    full_x = adata_full.X.toarray()
    expected_var = full_x[:, col_subset].var(axis=0)

    np.testing.assert_allclose(projected_var, expected_var, rtol=1e-5)


def test_col_max_projected(backed_projected_data, col_subset):
    """col max with projection matches materialized subset col max."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_max = np.asarray(backed_x.max(axis=0)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_max = np.asarray(full_x.max(axis=0).toarray()).flatten()

    np.testing.assert_array_equal(projected_max, expected_max)


def test_col_min_projected(backed_projected_data, col_subset):
    """col min with projection matches materialized subset col min."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_min = np.asarray(backed_x.min(axis=0)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_min = np.asarray(full_x.min(axis=0).toarray()).flatten()

    np.testing.assert_array_equal(projected_min, expected_min)


def test_col_mean_projected(backed_projected_data, col_subset):
    """col mean with projection matches materialized subset col mean."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_mean = np.asarray(backed_x.mean(axis=0)).flatten()

    full_x = adata_full.X.toarray()
    expected_mean = full_x[:, col_subset].mean(axis=0)

    np.testing.assert_allclose(projected_mean, expected_mean, rtol=1e-6)


# ---------------------------------------------------------------------------
# Row-axis aggregation with column projection
# ---------------------------------------------------------------------------


def test_row_sums_projected(backed_projected_data, col_subset):
    """row sums with projection matches materialized subset row sums."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_row_sums = np.asarray(backed_x.sum(axis=1)).flatten()

    full_x = adata_full.X.toarray()
    expected_row_sums = full_x[:, col_subset].sum(axis=1)

    np.testing.assert_allclose(projected_row_sums, expected_row_sums, rtol=1e-6)


def test_row_nnz_projected(backed_projected_data, col_subset):
    """row nnz with projection matches materialized subset row nnz."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_row_nnz = np.asarray(backed_x.getnnz(axis=1)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_row_nnz = np.asarray(full_x.getnnz(axis=1)).flatten()

    np.testing.assert_array_equal(projected_row_nnz, expected_row_nnz)


def test_total_sum_projected(backed_projected_data, col_subset):
    """total sum (no axis) with projection matches materialized."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_total = backed_x.sum()

    full_x = adata_full.X.toarray()
    expected_total = full_x[:, col_subset].sum()

    np.testing.assert_allclose(projected_total, expected_total, rtol=1e-6)


def test_total_nnz_projected(backed_projected_data, col_subset):
    """total nnz (no axis) with projection matches materialized."""
    import pyscx

    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    adata_full = pyscx.open(backed_projected_data).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_total_nnz = backed_x.getnnz()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_total_nnz = full_x.getnnz()

    assert projected_total_nnz == expected_total_nnz


# ---------------------------------------------------------------------------
# Masked + projected aggregation (deletion vector + column subset)
# ---------------------------------------------------------------------------


def _make_deleted_projected_file(tmp_dir, name="del_proj.scx"):
    """Create an SCX file with deletion vectors for projected agg tests."""
    import anndata
    import pyscx

    np.random.seed(99)
    n_obs, n_vars = 100, 30
    dense = np.random.randint(0, 200, size=(n_obs, n_vars)).astype(np.float32)
    mask = np.random.random((n_obs, n_vars)) > 0.3
    dense[mask] = 0
    x = sp.csr_matrix(dense)
    adata = anndata.AnnData(X=x)

    path = str(tmp_dir / name)
    pyscx.from_anndata(adata, path)

    # Mark some cells as deleted
    delete_mask = np.zeros(n_obs, dtype=bool)
    delete_mask[0] = True
    delete_mask[5] = True
    delete_mask[n_obs - 1] = True

    exp = pyscx.open(path)
    exp.mark_deleted(delete_mask)
    return path, delete_mask, adata


def test_col_sums_masked_projected(tmp_dir):
    """Column sums with deletion + projection matches filtered materialized."""
    import pyscx

    path, delete_mask, orig_adata = _make_deleted_projected_file(
        tmp_dir, "del_proj_sums.scx"
    )
    col_subset = [0, 3, 7, 15, 20, 25]

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_sums = np.asarray(backed_x.sum(axis=0)).flatten()

    full_x = adata_full.X.toarray()
    expected_sums = full_x[:, col_subset].sum(axis=0)

    np.testing.assert_allclose(projected_sums, expected_sums, rtol=1e-6)


def test_col_nnz_masked_projected(tmp_dir):
    """Column NNZ with deletion + projection matches filtered materialized."""
    import pyscx

    path, delete_mask, orig_adata = _make_deleted_projected_file(
        tmp_dir, "del_proj_nnz.scx"
    )
    col_subset = [0, 3, 7, 15, 20, 25]

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_nnz = np.asarray(backed_x.getnnz(axis=0)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_nnz = np.asarray(full_x.getnnz(axis=0)).flatten()

    np.testing.assert_array_equal(projected_nnz, expected_nnz)


def test_col_var_masked_projected(tmp_dir):
    """Column var with deletion + projection matches filtered materialized."""
    import pyscx

    path, delete_mask, orig_adata = _make_deleted_projected_file(
        tmp_dir, "del_proj_var.scx"
    )
    col_subset = [0, 3, 7, 15, 20, 25]

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_var = np.asarray(backed_x.var(axis=0)).flatten()

    full_x = adata_full.X.toarray()
    expected_var = full_x[:, col_subset].var(axis=0)

    np.testing.assert_allclose(projected_var, expected_var, rtol=1e-5)


def test_col_max_masked_projected(tmp_dir):
    """Column max with deletion + projection matches filtered materialized."""
    import pyscx

    path, delete_mask, orig_adata = _make_deleted_projected_file(
        tmp_dir, "del_proj_max.scx"
    )
    col_subset = [0, 3, 7, 15, 20, 25]

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_max = np.asarray(backed_x.max(axis=0)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_max = np.asarray(full_x.max(axis=0).toarray()).flatten()

    np.testing.assert_array_equal(projected_max, expected_max)


def test_col_min_masked_projected(tmp_dir):
    """Column min with deletion + projection matches filtered materialized."""
    import pyscx

    path, delete_mask, orig_adata = _make_deleted_projected_file(
        tmp_dir, "del_proj_min.scx"
    )
    col_subset = [0, 3, 7, 15, 20, 25]

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_min = np.asarray(backed_x.min(axis=0)).flatten()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_min = np.asarray(full_x.min(axis=0).toarray()).flatten()

    np.testing.assert_array_equal(projected_min, expected_min)


def test_total_nnz_masked_projected(tmp_dir):
    """Total NNZ (no axis) with deletion + projection matches filtered materialized."""
    import pyscx

    path, delete_mask, orig_adata = _make_deleted_projected_file(
        tmp_dir, "del_proj_total_nnz.scx"
    )
    col_subset = [0, 3, 7, 15, 20, 25]

    adata_backed = pyscx.open(path).to_anndata(backed=True)
    adata_full = pyscx.open(path).to_anndata()

    backed_x = adata_backed.X
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_total_nnz = backed_x.getnnz()

    full_x = sp.csr_matrix(adata_full.X)[:, col_subset]
    expected_total_nnz = full_x.getnnz()

    assert projected_total_nnz == expected_total_nnz


# ---------------------------------------------------------------------------
# QC metrics integration test
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="scanpy uses np.count_nonzero() internally which doesn't dispatch "
    "to our backed getnnz(). Streaming projected aggregation works correctly "
    "when called directly (see other tests). Requires scanpy-side fix or "
    "monkey-patching of scanpy._utils.axis_nnz dispatch.",
    strict=False,
)
def test_qc_metrics_with_gene_subset(backed_projected_data):
    """sc.pp.calculate_qc_metrics with qc_vars works in backed mode.
    
    This is the critical integration test — it's the exact scanpy call
    that triggers col_projection + sum/getnnz on backed data.
    """
    import pyscx
    import scanpy as sc

    # Backed adata
    adata_backed = pyscx.open(backed_projected_data).to_anndata(backed=True)
    
    # Define a "mito" gene set (first 5 genes)
    adata_backed.var["mt"] = False
    adata_backed.var.iloc[:5, adata_backed.var.columns.get_loc("mt")] = True

    # This should NOT materialize — uses streaming projected agg
    sc.pp.calculate_qc_metrics(adata_backed, qc_vars=["mt"], inplace=True)

    # Non-backed reference
    adata_full = pyscx.open(backed_projected_data).to_anndata()
    adata_full.var["mt"] = False
    adata_full.var.iloc[:5, adata_full.var.columns.get_loc("mt")] = True
    sc.pp.calculate_qc_metrics(adata_full, qc_vars=["mt"], inplace=True)

    # Validate QC metrics match
    np.testing.assert_allclose(
        adata_backed.obs["total_counts"].values,
        adata_full.obs["total_counts"].values,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        adata_backed.obs["n_genes_by_counts"].values,
        adata_full.obs["n_genes_by_counts"].values,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        adata_backed.obs["pct_counts_mt"].values,
        adata_full.obs["pct_counts_mt"].values,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# B6: projected row / scalar variance (streaming, no materialization fallback)
# ---------------------------------------------------------------------------


def test_col_var_axis1_projected(backed_projected_data, col_subset):
    """Row var (axis=1) with projection matches the materialized subset, via
    the streaming row-stats reducer (no to_memory fallback)."""
    import pyscx

    backed_x = pyscx.open(backed_projected_data).to_anndata(backed=True).X
    full_x = pyscx.open(backed_projected_data).to_anndata().X.toarray()
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_var = np.asarray(backed_x.var(axis=1)).flatten()
    expected_var = full_x[:, col_subset].var(axis=1)
    np.testing.assert_allclose(projected_var, expected_var, rtol=1e-5)


def test_var_scalar_projected(backed_projected_data, col_subset):
    """Scalar var (axis=None) with projection matches the materialized subset."""
    import pyscx

    backed_x = pyscx.open(backed_projected_data).to_anndata(backed=True).X
    full_x = pyscx.open(backed_projected_data).to_anndata().X.toarray()
    backed_x.set_col_projection([int(c) for c in col_subset])

    projected_var = float(np.asarray(backed_x.var()))
    expected_var = float(full_x[:, col_subset].var())
    np.testing.assert_allclose(projected_var, expected_var, rtol=1e-5)


# ---------------------------------------------------------------------------
# `pyscx.accel.col_*` on an unprojected handle: the reader's own kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("masked", [False, True], ids=["whole", "deleted_rows"])
@pytest.mark.parametrize("op", ["col_sums", "col_nnz", "col_min", "col_max", "col_var"])
def test_accel_col_ops_agree_with_the_dunders_without_a_projection(tmp_dir, op, masked):
    """With no column projection, `pyscx.accel.col_min` / `col_max` / `col_var`
    run the reader's whole-axis kernels — the ones `X.min/max/var(axis=0)` use —
    and so agree with them **bit for bit**, on a whole handle and on one with a
    deletion vector.

    They used to pass an identity column list to the projected kernels, which
    run `project_csr` on every decoded shard: a copy of the whole matrix per
    pass, which made those three 7x slower than `col_sums` on tabula_sapiens_100k
    (6.6 s against 0.95 s after this change) and gave `col_var` a different
    summation order from `X.var(axis=0)` on the same handle.
    """
    import anndata
    import pyscx

    # Multi-shard, with non-integer values: on one shard, or on sums of small
    # integers, the two accumulation orders produce the same doubles and the
    # comparison below could not tell them apart.
    rng = np.random.default_rng(5)
    dense = rng.lognormal(size=(400, 30)).astype(np.float32)
    dense[rng.random((400, 30)) > 0.4] = 0
    path = str(tmp_dir / f"col_ops_{op}_{masked}.scx")
    pyscx.from_anndata(anndata.AnnData(X=sp.csr_matrix(dense)), path, shard_size=50)
    if masked:
        drop = np.zeros(400, dtype=bool)
        drop[[0, 77, 260, 399]] = True
        pyscx.open(path).mark_deleted(drop)
    assert pyscx.open(path).shard_count >= 8, "premise: a multi-shard file"
    x = pyscx.open(path).to_anndata(backed=True).X
    got = np.asarray(getattr(pyscx.accel, op)(x), dtype=np.float64)
    dunder = {
        "col_sums": lambda: x.sum(axis=0),
        "col_nnz": lambda: x.getnnz(axis=0),
        "col_min": lambda: x.min(axis=0),
        "col_max": lambda: x.max(axis=0),
        "col_var": lambda: x.var(axis=0),
    }[op]
    want = np.asarray(dunder(), dtype=np.float64).ravel()
    np.testing.assert_array_equal(got, want)

    dense = pyscx.open(path).to_anndata().X.toarray().astype(np.float64)
    ref = {
        "col_sums": lambda: dense.sum(axis=0),
        "col_nnz": lambda: (dense != 0).sum(axis=0),
        "col_min": lambda: dense.min(axis=0),
        "col_max": lambda: dense.max(axis=0),
        "col_var": lambda: dense.var(axis=0),
    }[op]()
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("op", ["col_sums", "col_nnz", "col_min", "col_max", "col_var"])
def test_csc_col_ops_on_a_projection_that_skips_whole_shards(tmp_dir, op):
    """The CSC walk splits a contiguous run of requested columns at CSC shard
    boundaries. A projection that takes the tail of one shard, skips the next
    two entirely and resumes mid-way through a fourth exercises the cases that
    split has to get right: a run that ends at a boundary, shards the
    projection empties (whose ranges collapse onto the next shard's start), and
    a run that starts part-way into a shard."""
    import anndata
    import pyscx

    rng = np.random.default_rng(11)
    dense = rng.lognormal(size=(120, 40)).astype(np.float32)
    dense[rng.random((120, 40)) > 0.5] = 0
    path = str(tmp_dir / f"skip_{op}.scx")
    pyscx.from_anndata(
        anndata.AnnData(X=sp.csr_matrix(dense)), path, csc="always", csc_cols_per_shard=8
    )
    cols = list(range(5, 8)) + list(range(28, 35)) + [39]  # shards 0, 3, 4; skips 1-2
    adata = pyscx.open(path).to_anndata(backed=True)
    pyscx.accel.subset_var(adata, cols)
    got = np.asarray(getattr(pyscx.accel, op)(adata.X, prefer_format="csc"), dtype=np.float64)
    csr = np.asarray(getattr(pyscx.accel, op)(adata.X, prefer_format="csr"), dtype=np.float64)
    sub = dense[:, cols].astype(np.float64)
    ref = {
        "col_sums": sub.sum(axis=0),
        "col_nnz": (sub != 0).sum(axis=0),
        "col_min": sub.min(axis=0),
        "col_max": sub.max(axis=0),
        "col_var": sub.var(axis=0),
    }[op]
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got, csr, rtol=1e-9, atol=1e-12)
