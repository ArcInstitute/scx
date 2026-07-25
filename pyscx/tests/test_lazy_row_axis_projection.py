"""Every array-protocol reduction must speak the **visible** axis.

Phase 4.0a, root cause (b) of §9.18. ``ScxLazyTransformedDataset`` had two
row-axis kernels — ``streaming_row_sums()`` (physical width) and
``streaming_row_sums_projected()`` (correct) — and the projection-blind one had
the obvious name. Six callers picked it, so under an active ``col_projection``:

* ``X.sum(axis=1)`` / ``X.sum()`` summed genes the caller cannot see;
* ``X.mean(axis=1)`` / ``X.mean()`` divided that physical-width numerator by the
  *visible* column count, wrong twice over;
* ``filter_cells`` on a lazy ``X`` thresholded cells on totals that included the
  genes ``filter_genes`` had already removed.

Measured before the fix: 29 529 where the visible-gene total was 10 000.

The matrix below is {backed, lazy} × {projection, none} × {deletions, none}
against a **pure-numpy oracle** built from the source dense matrix — not against
another SCX kernel, so there is a real oracle. The backed arms are expected to
pass on both sides of the fix; they pin its blast radius.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

N_OBS, N_VARS = 60, 40
TARGET_SUM = 1e4

# Non-contiguous and not starting at 0, so a visible-space index never
# coincides with its on-disk index.
GENE_SUBSET = [3, 7, 11, 12, 19, 25, 28, 33, 37, 39]

# Backed arms accumulate the stored f32 values in f64 walking ascending columns;
# numpy sums the same values pairwise. Same inputs, different association only.
RTOL_UNTRANSFORMED = 1e-12
# Lazy arms additionally carry `normalize_total`'s rescale, which lands back in
# f32 storage — so the oracle and the kernel diverge at f32 epsilon (~1e-8
# relative, measured). Still four orders tighter than the ~3x error the
# projection-blind kernel produced.
RTOL_TRANSFORMED = 1e-6


@pytest.fixture
def dense_src():
    """``(N_OBS, N_VARS)`` float32 counts, ~40 % dense.

    Floats, not integers: an all-integer fixture makes an equality assertion
    satisfiable by *any* accumulation order, which would make the comparison
    vacuous. Column ``GENE_SUBSET[0]`` is forced nonzero in every row so no row
    is empty within the projection — that keeps ``normalize_total``'s zero-sum
    guard out of the oracle.
    """
    rng = np.random.RandomState(20260725)
    dense = (rng.random_sample((N_OBS, N_VARS)) * 1e3).astype(np.float32)
    dense[rng.random_sample((N_OBS, N_VARS)) > 0.4] = 0.0
    dense[:, GENE_SUBSET[0]] = (rng.random_sample(N_OBS) * 1e3 + 1.0).astype(np.float32)
    return dense


@pytest.fixture
def scx_path(dense_src, tmp_dir):
    import anndata
    import pandas as pd
    import pyscx
    import scipy.sparse as sp

    adata = anndata.AnnData(
        X=sp.csr_matrix(dense_src),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"gene_{j}" for j in range(N_VARS)]),
    )
    path = str(tmp_dir / "row_axis.scx")
    pyscx.from_anndata(adata, path)
    return path


def _build(scx_path, dense, kind, projection, deletions):
    """Return ``(X, M, rtol)`` — the SCX handle, its pure-numpy oracle, and the
    tolerance that flavor earns.

    ``M`` is float64 of shape ``(n_visible_obs, n_visible_vars)``: the source
    matrix restricted to the visible axes, with ``normalize_total`` replicated
    in numpy for the lazy flavors.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    cols = list(GENE_SUBSET) if projection else list(range(N_VARS))
    if projection:
        adata.X.set_col_projection([int(c) for c in cols])
        # `_var` bypasses anndata's shape validation (X width already changed).
        adata._var = adata.var.iloc[cols].copy()

    m = dense[:, cols].astype(np.float64)

    if kind == "lazy":
        pyscx.accel.normalize_total(adata, target_sum=TARGET_SUM)
        # scanpy semantics: the denominator is the *visible* row total.
        m = m * (TARGET_SUM / m.sum(axis=1))[:, None]

    if deletions:
        keep = np.ones(N_OBS, dtype=bool)
        keep[::3] = False
        pyscx.accel.subset_obs(adata, keep)
        m = m[keep]

    rtol = RTOL_TRANSFORMED if kind == "lazy" else RTOL_UNTRANSFORMED
    return adata.X, m, rtol


FLAVORS = [
    pytest.param(
        (kind, proj, dele),
        id=f"{kind}-{'proj' if proj else 'full'}-{'del' if dele else 'nodel'}",
    )
    for kind in ("backed", "lazy")
    for proj in (True, False)
    for dele in (True, False)
]


@pytest.fixture(params=FLAVORS)
def flavor(request, scx_path, dense_src):
    kind, projection, deletions = request.param
    return _build(scx_path, dense_src, kind, projection, deletions)


def _flat(value):
    return np.asarray(value, dtype=np.float64).ravel()


def test_shape_is_visible(flavor):
    x, m, _ = flavor
    assert x.shape == m.shape


def test_sum_axes(flavor):
    x, m, rtol = flavor
    assert_allclose(_flat(x.sum(axis=0)), m.sum(axis=0), rtol=rtol)
    assert_allclose(_flat(x.sum(axis=1)), m.sum(axis=1), rtol=rtol)
    assert_allclose(float(x.sum()), m.sum(), rtol=rtol)


def test_mean_axes(flavor):
    x, m, rtol = flavor
    assert_allclose(_flat(x.mean(axis=0)), m.mean(axis=0), rtol=rtol)
    assert_allclose(_flat(x.mean(axis=1)), m.mean(axis=1), rtol=rtol)
    assert_allclose(float(x.mean()), m.mean(), rtol=rtol)


def test_getnnz_axes(flavor):
    x, m, _ = flavor
    nz = m != 0
    assert_array_equal(_flat(x.getnnz(axis=0)), nz.sum(axis=0))
    assert_array_equal(_flat(x.getnnz(axis=1)), nz.sum(axis=1))
    assert int(x.getnnz()) == int(nz.sum())


def test_var_axes(flavor):
    x, m, rtol = flavor
    # Population variance (ddof=0), matching both implementations. `E[X²]-E[X]²`
    # cancels ~7 leading digits when the mean dominates, so this is looser than
    # the sum tolerance and carries an absolute floor scaled to the data.
    kw = dict(rtol=max(rtol, 1e-5), atol=1e-3)
    assert_allclose(_flat(x.var(axis=0)), m.var(axis=0), **kw)
    assert_allclose(_flat(x.var(axis=1)), m.var(axis=1), **kw)
    assert_allclose(float(x.var()), m.var(), **kw)


def test_filter_cells_after_filter_genes_and_normalize(scx_path, dense_src):
    """The pipeline that motivated the fix.

    ``filter_genes → normalize_total → filter_cells`` must threshold cells on
    totals over the *kept* genes. Before the fix the lazy ``filter_cells`` arm
    summed the removed genes back in, so cells were kept or dropped on a number
    the user could not observe anywhere.

    After ``normalize_total`` every visible row total is exactly ``TARGET_SUM``,
    so a ``min_counts`` just above it must drop **every** cell — while the
    physical-width total (which includes the filtered-out genes) is far larger
    and would keep them all.
    """
    import pyscx

    adata = pyscx.open(scx_path).to_anndata(backed=True)
    # Keep a strict subset of genes: only those detected in > 60 % of cells.
    detected = (dense_src != 0).sum(axis=0)
    threshold = int(np.median(detected)) + 1
    pyscx.accel.filter_genes(adata, min_cells=threshold)
    assert adata.n_vars < N_VARS, "fixture must actually drop genes"

    pyscx.accel.normalize_total(adata, target_sum=TARGET_SUM)
    visible_totals = np.asarray(adata.X.to_memory().sum(axis=1)).ravel()
    assert_allclose(visible_totals, TARGET_SUM, rtol=1e-5)

    pyscx.accel.filter_cells(adata, min_counts=TARGET_SUM * 1.01)
    assert adata.n_obs == 0, (
        "filter_cells thresholded on physical-width row totals: every visible "
        f"row sums to {TARGET_SUM}, so none should clear the bar"
    )
