"""Query pipeline integration tests."""

import numpy as np
import pytest
import scipy.sparse as sp


# ───────────────────────────────────────────────────────────────────────
# Tests
# ───────────────────────────────────────────────────────────────────────


def test_filter_obs_cell_type(query_adata, scx_from_adata):
    """filter_obs with cell_type predicate returns correct subset."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    result = pyscx.open(path).query()
    result.filter_obs("cell_type == 'T cell'")
    qr = result.collect()

    assert qr.n_obs == 40  # 40 T cells in fixture
    assert qr.n_vars == 40


def test_collect_to_anndata(query_adata, scx_from_adata):
    """collect().to_anndata() returns valid AnnData with filtered obs."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'B cell'")
    adata = p.collect().to_anndata()

    assert adata.n_obs == 40
    assert adata.n_vars == 40
    assert sp.issparse(adata.X)
    # obs should only contain B cells
    assert all(adata.obs["cell_type"] == "B cell")


def test_select_genes(query_adata, scx_from_adata):
    """select_genes([0, 1, 2]) returns 3-column matrix."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.select_genes([0, 1, 2])
    result = p.collect()

    assert result.n_vars == 3
    assert result.n_obs == 120  # all cells


def test_select_genes_preserves_order_and_names(query_adata, scx_from_adata):
    """F4: select_genes returns columns in the requested order (not sorted),
    accepts gene names and mixed int/str, raises KeyError on unknown names, and
    collapses duplicates to first occurrence."""
    import pyscx

    path = scx_from_adata(query_adata, "order.scx")
    ref = query_adata.X.toarray()  # 120 × 40 reference

    # Non-ascending integer request → requested order preserved.
    req = [7, 3, 0, 20]
    adata = pyscx.open(path).query().select_genes(req).collect().to_anndata()
    assert list(adata.var["gene_id"]) == [f"gene_{i}" for i in req]
    assert np.array_equal(adata.X.toarray(), ref[:, req])

    # Gene names in a custom order.
    names = ["gene_5", "gene_2", "gene_30"]
    a2 = pyscx.open(path).query().select_genes(names).collect().to_anndata()
    assert list(a2.var["gene_id"]) == names
    assert np.array_equal(a2.X.toarray(), ref[:, [5, 2, 30]])

    # Mixed int + name, order preserved.
    a3 = pyscx.open(path).query().select_genes([1, "gene_9", 4]).collect().to_anndata()
    assert list(a3.var["gene_id"]) == ["gene_1", "gene_9", "gene_4"]
    assert np.array_equal(a3.X.toarray(), ref[:, [1, 9, 4]])

    # Unknown name → KeyError (eager, at select_genes).
    with pytest.raises(KeyError):
        pyscx.open(path).query().select_genes(["not_a_gene"])

    # Duplicates collapse to first occurrence.
    a4 = pyscx.open(path).query().select_genes([3, 3, 1]).collect().to_anndata()
    assert list(a4.var["gene_id"]) == ["gene_3", "gene_1"]

    # Empty selection → 0-gene result (no error).
    a5 = pyscx.open(path).query().select_genes([]).collect().to_anndata()
    assert a5.n_vars == 0 and a5.n_obs == 120


def test_select_genes_accepts_numpy_int_selectors(query_adata, scx_from_adata):
    """PR #242 review: numpy integer scalars (a `np.array([...])` or a list of
    numpy ints) resolve as indices, not falling through to the name path."""
    import pyscx

    path = scx_from_adata(query_adata, "npidx.scx")
    ref = query_adata.X.toarray()
    req = [7, 3, 0]

    a_arr = pyscx.open(path).query().select_genes(np.array(req)).collect().to_anndata()
    a_list = pyscx.open(path).query().select_genes([np.int64(i) for i in req]).collect().to_anndata()
    for a in (a_arr, a_list):
        assert list(a.var["gene_id"]) == [f"gene_{i}" for i in req]
        assert np.array_equal(a.X.toarray(), ref[:, req])


def test_normalize_log1p(query_adata, scx_from_adata):
    """with_normalize + with_log1p produces transformed values matching scanpy."""
    import pyscx
    import scanpy as sc

    path = scx_from_adata(query_adata, "query.scx")

    # SCX pipeline path
    p = pyscx.open(path).query()
    p.with_normalize(1e4)
    p.with_log1p()
    scx_adata = p.collect().to_anndata()

    # Scanpy reference path
    ref_adata = pyscx.open(path).to_anndata()
    sc.pp.normalize_total(ref_adata, target_sum=1e4)
    sc.pp.log1p(ref_adata)

    np.testing.assert_allclose(
        scx_adata.X.toarray(), ref_adata.X.toarray(), rtol=1e-5, atol=1e-5
    )


def test_count(query_adata, scx_from_adata):
    """count() returns number of matching cells."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'NK cell'")
    assert p.count() == 40


def test_count_does_not_consume_pipeline(query_adata, scx_from_adata):
    """count() borrows the pipeline (no X decode), so collect() still works."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'B cell'")
    # count() must not consume the pipeline.
    assert p.count() == 40
    qr = p.collect()
    assert qr.n_obs == 40


def test_count_ignores_limit(query_adata, scx_from_adata):
    """count() reports the full match count, ignoring limit (engine CLI6 contract).

    (Previously pyscx count() ran a full collect() and so reflected the limit;
    it now routes to the no-decode engine count, which ignores limit.)
    """
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")  # 40 T cells
    p.limit(5)
    assert p.count() == 40  # full match count, not 5


def test_exists_matches_count(query_adata, scx_from_adata):
    """exists() agrees with count() > 0 and is non-consuming."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")

    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    assert p.exists() is True
    # Non-consuming: count() still works afterward.
    assert p.count() == 40

    p2 = pyscx.open(path).query()
    p2.filter_obs("cell_type == 'macrophage'")  # absent
    assert p2.exists() is False
    assert p2.count() == 0


def test_count_matches_collect(query_adata, scx_from_adata):
    """count() equals collect().n_obs across a few predicates (limit ignored)."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    for expr, _ in [
        ("cell_type == 'T cell'", 40),
        ("tissue == 'lung'", 60),
        ("cell_type == 'B cell' and tissue == 'blood'", 20),
    ]:
        p = pyscx.open(path).query()
        p.filter_obs(expr)
        c = p.count()
        n = p.collect().n_obs
        assert c == n, f"count {c} != collect n_obs {n} for `{expr}`"


def test_limit(query_adata, scx_from_adata):
    """limit(5) returns at most 5 cells."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.limit(5)
    result = p.collect()	

    assert result.n_obs <= 5


def test_chained_predicates(query_adata, scx_from_adata):
    """Chained filter_obs calls apply AND semantics."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    p.filter_obs("tissue == 'lung'")
    result = p.collect()

    # 40 T cells, half lung half blood → 20 expected
    assert result.n_obs == 20


def test_invalid_column_raises_valueerror(query_adata, scx_from_adata):
    """Invalid column name raises ValueError (not RuntimeError).

    The assertion is deliberately narrow: an over-broad
    `(ValueError, RuntimeError)` here is what hid the pipeline-poisoning bug
    below, because the RuntimeError the *second* call raised looked acceptable.
    """
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises(ValueError):
        p.filter_obs("nonexistent_col == 'foo'")


def test_empty_result(query_adata, scx_from_adata):
    """Empty result (no matching cells) gives AnnData with 0 rows."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'macrophage'")  # no macrophages in fixture
    result = p.collect()

    assert result.n_obs == 0
    adata = result.to_anndata()
    assert adata.n_obs == 0


def test_to_csr(query_adata, scx_from_adata):
    """to_csr() returns scipy CSR without AnnData overhead."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    result = pyscx.open(path).query().collect()
    csr = result.to_csr()

    assert sp.issparse(csr)
    assert csr.shape == (120, 40)


def test_shard_stats_accessible(query_adata, scx_from_adata):
    """skipped_shards and total_shards are accessible on result."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    result = pyscx.open(path).query().collect()

    assert result.total_shards >= 1
    assert result.skipped_shards >= 0


def test_sequential_call_syntax(query_adata, scx_from_adata):
    """Sequential call syntax works (no method chaining needed)."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    p.with_log1p()
    result = p.collect()

    assert result.n_obs == 40


# ───────────────────────────────────────────────────────────────────────
# A failed builder step must not poison the pipeline
#
# Regression: each builder used to move the pipeline into the engine's
# consuming builder, which drops it on Err. A failed step therefore left the
# Python object permanently empty, and the *next* call raised
# "Pipeline already consumed by collect()" — blaming an operation that never
# ran. Builders now mutate in place, so an error changes nothing.
# ───────────────────────────────────────────────────────────────────────


def test_filter_obs_parse_error_leaves_pipeline_usable(query_adata, scx_from_adata):
    """A predicate typo raises, and the pipeline is still usable afterwards."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises(ValueError):
        p.filter_obs("cell_type ==== 'T cell'")

    # Same object, corrected predicate — and the rejected one did not
    # half-apply (40 T cells, not 0).
    p.filter_obs("cell_type == 'T cell'")
    assert p.count() == 40
    assert p.collect().n_obs == 40


def test_filter_obs_schema_error_leaves_pipeline_usable(query_adata, scx_from_adata):
    """An unknown obs column raises ValueError; the pipeline survives."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises(ValueError):
        p.filter_obs("nonexistent_col == 'foo'")

    p.filter_obs("tissue == 'lung'")
    assert p.collect().n_obs == 60


def test_filter_var_error_leaves_pipeline_usable(query_adata, scx_from_adata):
    """An unknown var column raises ValueError; the pipeline survives."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises(ValueError):
        p.filter_var("nonexistent_col == 1")

    p.filter_obs("cell_type == 'B cell'")
    assert p.collect().n_obs == 40


def test_select_genes_unknown_name_leaves_pipeline_usable(query_adata, scx_from_adata):
    """An unknown gene name raises KeyError; the pipeline survives.

    This branch used to restore the pipeline by hand while its siblings did
    not; pin the behaviour now that the restore is gone (nothing is taken out
    in the first place).
    """
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises(KeyError):
        p.select_genes(["not_a_gene"])

    p.select_genes([0, 1, 2])
    assert p.collect().n_vars == 3


def test_select_genes_bad_selector_leaves_pipeline_usable(query_adata, scx_from_adata):
    """Selectors rejected before the engine is reached also leave it usable."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()

    with pytest.raises((ValueError, TypeError)):
        p.select_genes([object()])
    with pytest.raises((ValueError, OverflowError)):
        p.select_genes([-1])

    p.select_genes(["gene_5", 2])
    assert p.collect().n_vars == 2


def test_failed_collect_does_not_consume_pipeline(query_adata, scx_from_adata):
    """A failed collect() leaves the pipeline usable so the caller can retry.

    Out-of-range gene indices are documented as "handled at collect time", so
    they fail inside the engine rather than in the builder — the reachable way
    to make collect() fail without an I/O fault.

    `BaseException` rather than `Exception` because an out-of-range index
    currently surfaces as a `pyo3_runtime.PanicException` from arrow's `take`
    (an unchecked index, separate pre-existing defect — `select_genes` should
    reject it or the projection should return an error). What is asserted here
    is only the property this test exists for: whatever collect() raises, it
    does not take the pipeline with it.
    """
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.select_genes([10**6])

    with pytest.raises(BaseException):  # noqa: B017,PT011 — see docstring
        p.collect()

    # Alive: the borrowing collect never took the pipeline.
    assert p.count() == 120
    p.select_genes([0, 1])  # overwrite the bad selection
    assert p.collect().n_vars == 2


def test_collect_consumes_pipeline_on_success(query_adata, scx_from_adata):
    """The contract we keep: a *successful* collect() consumes the pipeline,
    and every later call says so — accurately, now that nothing else can empty
    it."""
    import pyscx

    path = scx_from_adata(query_adata, "query.scx")
    p = pyscx.open(path).query()
    p.filter_obs("cell_type == 'T cell'")
    assert p.collect().n_obs == 40
    assert repr(p) == "PyQueryPipeline(consumed)"

    for call in (
        lambda: p.filter_obs("tissue == 'lung'"),
        lambda: p.filter_var("gene_id == 'gene_1'"),
        lambda: p.select_genes([0]),
        lambda: p.with_normalize(),
        lambda: p.with_log1p(),
        lambda: p.limit(5),
        lambda: p.count(),
        lambda: p.exists(),
        lambda: p.collect(),
    ):
        with pytest.raises(RuntimeError, match="consumed by a successful collect"):
            call()
