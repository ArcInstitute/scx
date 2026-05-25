"""Numerical parity tests for `pyscx.accel.pdex_ref` against `pdex.pdex(mode="ref")`.

The pdex accelerator implements the same per-cell MWU + pseudobulk-geometric-mean
log fold change algorithm pdex emits, but in Rust with SCX-backed streaming and
gene chunking. These tests verify exact (to fp32 tolerance) parity between the
two implementations across the four (geometric_mean × is_log1p) combinations
on four input shapes: dense numpy, in-memory CSR, log1p-transformed, and an
SCX-backed file produced via `pyscx.from_anndata`.
"""

from __future__ import annotations

import numpy as np
import pytest

pl = pytest.importorskip("polars")
pdex_mod = pytest.importorskip("pdex")
pdex_fn = pdex_mod.pdex

import anndata as ad  # noqa: E402
import scipy.sparse as sp  # noqa: E402

import pyscx  # noqa: E402


SEED = 0
N_OBS = 90
N_VARS = 15
N_GROUPS = 3
REFERENCE = "non-targeting"


def _make_adata(seed: int = SEED) -> ad.AnnData:
    """Synthetic count-style AnnData with three groups (one is the reference).

    Group means differ deterministically so DE produces a non-trivial signal.
    """
    rng = np.random.default_rng(seed)
    # Cells per group, evenly split.
    per_group = N_OBS // N_GROUPS
    groups = np.repeat([REFERENCE, "ko_a", "ko_b"], per_group)
    # Per-group mean expression: shape (3, N_VARS). KO groups have a 2× shift
    # on a subset of genes so MWU has signal, leaving the rest near-equal.
    base = rng.uniform(0.5, 3.0, size=(N_GROUPS, N_VARS))
    base[1, : N_VARS // 2] *= 2.0  # ko_a perturbs first half
    base[2, N_VARS // 2 :] *= 2.5  # ko_b perturbs second half
    counts = np.zeros((N_OBS, N_VARS), dtype=np.float32)
    for c in range(N_OBS):
        g = (c // per_group)
        counts[c] = rng.poisson(base[g]).astype(np.float32)
    obs = {
        "target": groups,
    }
    import pandas as pd

    obs_df = pd.DataFrame(obs, index=[f"cell_{i}" for i in range(N_OBS)])
    var_df = pd.DataFrame(
        {"gene_id": [f"gene_{j}" for j in range(N_VARS)]},
        index=[f"gene_{j}" for j in range(N_VARS)],
    )
    adata = ad.AnnData(X=counts, obs=obs_df, var=var_df)
    return adata


def _normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (target, feature) and select only the parity-checked columns.

    pdex emits rows grouped by target; pdex_ref emits rows grouped by target.
    Within each group, both emit rows in `var_names` (== feature) order today.
    Sorting by (target, feature) makes the comparison robust to any future
    re-ordering on either side.
    """
    keep = [
        "target",
        "feature",
        "target_mean",
        "ref_mean",
        "target_membership",
        "ref_membership",
        "log2_fold_change",
        "percent_change",
        "p_value",
        "statistic",
        "fdr",
    ]
    df = df.select([c for c in keep if c in df.columns])
    return df.sort(["target", "feature"])


def _assert_frames_close(
    scx: pl.DataFrame, pdx: pl.DataFrame, atol: float = 1e-4, rtol: float = 1e-4
) -> None:
    scx = _normalize_frame(scx)
    pdx = _normalize_frame(pdx)
    assert scx.shape == pdx.shape, f"shape mismatch: {scx.shape} vs {pdx.shape}"

    # String columns must match exactly.
    for col in ("target", "feature"):
        assert (scx[col] == pdx[col]).all(), f"column '{col}' mismatch"

    # Integer membership columns must match exactly.
    for col in ("target_membership", "ref_membership"):
        if col in scx.columns and col in pdx.columns:
            scx_int = scx[col].cast(pl.Int64).to_numpy()
            pdx_int = pdx[col].cast(pl.Int64).to_numpy()
            np.testing.assert_array_equal(
                scx_int, pdx_int, err_msg=f"column '{col}' integer mismatch"
            )

    # Float columns must agree within tolerance.
    float_cols = (
        "target_mean",
        "ref_mean",
        "log2_fold_change",
        "percent_change",
        "p_value",
        "statistic",
        "fdr",
    )
    for col in float_cols:
        if col not in scx.columns or col not in pdx.columns:
            continue
        a = scx[col].cast(pl.Float64).to_numpy()
        b = pdx[col].cast(pl.Float64).to_numpy()

        # Non-finite entries must agree on finiteness.
        a_finite = np.isfinite(a)
        b_finite = np.isfinite(b)
        np.testing.assert_array_equal(
            a_finite, b_finite, err_msg=f"finite-mask mismatch in '{col}'"
        )

        np.testing.assert_allclose(
            a[a_finite],
            b[b_finite],
            atol=atol,
            rtol=rtol,
            err_msg=f"column '{col}' tolerance violation",
        )


@pytest.mark.parametrize(
    "geometric_mean,is_log1p,epsilon",
    [
        (True, False, 0.0),
        (False, False, 0.0),
        (True, True, 0.0),
        (False, True, 0.0),
        (True, False, 0.5),
    ],
)
def test_pdex_ref_parity_dense(geometric_mean: bool, is_log1p: bool, epsilon: float):
    """`pdex_ref` on a dense AnnData matches `pdex.pdex(mode="ref")`."""
    adata = _make_adata()
    if is_log1p:
        adata.X = np.log1p(adata.X)

    # Reference via pdex.
    pdx_df = pdex_fn(
        adata,
        groupby="target",
        mode="ref",
        reference=REFERENCE,
        geometric_mean=geometric_mean,
        is_log1p=is_log1p,
        epsilon=epsilon,
    )

    # SCX accelerator path. Pass is_log1p explicitly to bypass auto-detection.
    scx_df = pyscx.accel.pdex_ref(
        adata,
        "target",
        reference=REFERENCE,
        geometric_mean=geometric_mean,
        is_log1p=is_log1p,
        epsilon=epsilon,
    )

    _assert_frames_close(scx_df, pdx_df)


@pytest.mark.parametrize("geometric_mean,is_log1p", [(True, False), (False, True)])
def test_pdex_ref_parity_sparse(geometric_mean: bool, is_log1p: bool):
    """`pdex_ref` on an in-memory CSR matches `pdex.pdex(mode="ref")`."""
    adata = _make_adata()
    if is_log1p:
        adata.X = np.log1p(adata.X)
    adata.X = sp.csr_matrix(adata.X)

    pdx_df = pdex_fn(
        adata,
        groupby="target",
        mode="ref",
        reference=REFERENCE,
        geometric_mean=geometric_mean,
        is_log1p=is_log1p,
    )
    scx_df = pyscx.accel.pdex_ref(
        adata,
        "target",
        reference=REFERENCE,
        geometric_mean=geometric_mean,
        is_log1p=is_log1p,
        gene_chunk_size=4,  # force multi-chunk path
    )
    _assert_frames_close(scx_df, pdx_df)


def test_pdex_ref_parity_backed(tmp_path):
    """`pdex_ref` on an SCX-backed AnnData matches `pdex.pdex(mode="ref")`."""
    adata = _make_adata()
    adata.X = sp.csr_matrix(adata.X)

    # Reference DE on the in-memory copy.
    pdx_df = pdex_fn(
        adata,
        groupby="target",
        mode="ref",
        reference=REFERENCE,
        geometric_mean=True,
        is_log1p=False,
    )

    # Write to SCX, open backed, run pdex_ref.
    path = str(tmp_path / "fixture.scx")
    pyscx.from_anndata(adata, path)
    backed_adata = pyscx.open(path).to_anndata(backed=True)

    scx_df = pyscx.accel.pdex_ref(
        backed_adata,
        "target",
        reference=REFERENCE,
        geometric_mean=True,
        is_log1p=False,
        gene_chunk_size=4,
    )
    _assert_frames_close(scx_df, pdx_df)


def test_pdex_ref_schema_matches_pdex():
    """Column set and dtypes of `pdex_ref` are compatible with `DEResults`."""
    adata = _make_adata()
    scx_df = pyscx.accel.pdex_ref(
        adata,
        "target",
        reference=REFERENCE,
        geometric_mean=True,
        is_log1p=False,
    )
    required = {
        "target",
        "feature",
        "target_mean",
        "ref_mean",
        "target_membership",
        "ref_membership",
        "fold_change",
        "log2_fold_change",
        "percent_change",
        "p_value",
        "statistic",
        "fdr",
    }
    missing = required - set(scx_df.columns)
    assert not missing, f"pdex_ref missing columns: {missing}"

    # `fold_change` is pdex's deprecated alias; pdex_ref mirrors log2_fold_change.
    np.testing.assert_allclose(
        scx_df["fold_change"].to_numpy(),
        scx_df["log2_fold_change"].to_numpy(),
        err_msg="fold_change should mirror log2_fold_change exactly",
    )


def _csr_with_descending_indices(adata: ad.AnnData) -> sp.csr_matrix:
    """Return a `csr_matrix` carrying the same dense values as `adata.X`,
    but with each row's column indices in **descending** order — i.e.
    `has_sorted_indices == False`.

    Mirrors the pathological scipy CSR shape seen on real-world datasets
    like `pbmc10k.h5ad`, which surfaced the
    `scx_engine::project_csr_row` precondition bug.
    """
    base = adata.X
    if sp.issparse(base):
        base = base.tocsr().copy()
        base.sort_indices()  # canonicalise first so the reversal is deterministic
    else:
        base = sp.csr_matrix(base)

    indptr = base.indptr.astype(np.int64).copy()
    indices = base.indices.astype(np.int32).copy()
    data = base.data.astype(np.float32).copy()
    # Reverse each row's slice so indices walk col_max → col_min.
    for r in range(indptr.size - 1):
        lo, hi = int(indptr[r]), int(indptr[r + 1])
        if hi - lo > 1:
            indices[lo:hi] = indices[lo:hi][::-1]
            data[lo:hi] = data[lo:hi][::-1]
    out = sp.csr_matrix((data, indices, indptr), shape=base.shape, copy=False)
    # Tell scipy not to assume sortedness — has_sorted_indices is a cached
    # flag, so set it explicitly to mirror real h5ad-derived CSRs.
    out.has_sorted_indices = False
    return out


def test_pdex_ref_cpu_unsorted_scipy_csr_matches_sorted():
    """Regression test for `scx_engine::project_csr_row` precondition.

    Before this fix, an `adata` whose `X` is a `scipy.sparse.csr_matrix`
    with `has_sorted_indices=False` (the pbmc10k.h5ad case) made the CPU
    `pdex_ref` collapse every gene's U to `n_g·n_ref/2` and p-value to
    1.0 — because `project_csr_row` uses a monotonic merge-scan pointer
    that silently drops every column index following a larger one.

    The fix is at the pyscx CPU sparse dispatch boundary
    (`pyscx::accel::de::run_pdex_ref_inner`): we now route through
    `ensure_csr`, which calls `.sorted_indices()` when the input CSR is
    unsorted. This test fixes the value of two `pdex_ref` runs against
    each other — sorted and unsorted — and asserts every numeric column
    agrees bit-for-bit. Atomically catches the regression without
    needing pdex.
    """
    adata_sorted = _make_adata()
    # Ensure adata_sorted's X is a sorted CSR for a clean baseline.
    adata_sorted.X = sp.csr_matrix(adata_sorted.X)
    adata_sorted.X.sort_indices()
    assert adata_sorted.X.has_sorted_indices

    adata_unsorted = adata_sorted.copy()
    adata_unsorted.X = _csr_with_descending_indices(adata_sorted)
    assert not adata_unsorted.X.has_sorted_indices

    sorted_df = pyscx.accel.pdex_ref(
        adata_sorted, "target", reference=REFERENCE, device="cpu"
    )
    unsorted_df = pyscx.accel.pdex_ref(
        adata_unsorted, "target", reference=REFERENCE, device="cpu"
    )

    _assert_frames_close(sorted_df, unsorted_df, atol=0.0, rtol=0.0)

    # Sanity: the fixture has real signal — pdex_ref should NOT return
    # p=1.0 for every gene. Catches the original "trivial U everywhere"
    # failure mode directly, in case both runs regress simultaneously.
    p_vals = sorted_df["p_value"].to_numpy()
    n_nontrivial = int(np.sum(p_vals < 0.99))
    assert n_nontrivial > 0, (
        f"fixture lost DE signal: all {len(p_vals)} p-values ≥ 0.99 — either the "
        f"_make_adata fixture changed or pdex_ref CPU is broken in a different way"
    )

    # Caller's AnnData must not be mutated (ensure_csr called with in_place=False).
    assert not adata_unsorted.X.has_sorted_indices, (
        "ensure_csr(in_place=False) must not mutate caller's CSR — but "
        "has_sorted_indices flipped to True after pdex_ref. Check that "
        "the dispatch site passes `in_place=false`."
    )
