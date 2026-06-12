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

# Synthetic-fixture builder + constants live in a sibling module so that
# downstream regression tests can reuse them without paying for the
# polars/pdex importorskip side effects above.
from _pdex_fixtures import (  # noqa: E402
    REFERENCE,
    _csr_with_descending_indices,
    _make_adata,
)


def _normalize_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Sort by (target, feature) and select only the parity-checked columns.

    pdex emits rows grouped by target; pdex_ref emits rows grouped by target.
    Within each group, both emit rows in `var_names` (== feature) order today.
    Sorting by (target, feature) makes the comparison robust to any future
    re-ordering on either side.

    Column-name normalization: upstream `pdex` reports the log2 fold change in a
    column literally named ``fold_change`` (it has no ``log2_fold_change``
    column), whereas `pdex_ref` exposes the same value under the descriptive
    ``log2_fold_change`` name (and additionally mirrors it to ``fold_change``
    as a migration alias — see ``test_pdex_ref_schema_matches_pdex``). Alias
    ``fold_change`` → ``log2_fold_change`` when the latter is absent so both
    frames carry the canonical column and the parity check actually compares
    fold changes rather than silently dropping the column.
    """
    if "log2_fold_change" not in df.columns and "fold_change" in df.columns:
        df = df.with_columns(pl.col("fold_change").alias("log2_fold_change"))
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


def test_pdex_ref_output_pandas():
    """F6: pdex_ref supports output='pandas' — a pandas DataFrame with
    identical columns/values to the polars default; bad output errors."""
    import pandas as pd

    adata = _make_adata()

    df_pl = pyscx.accel.pdex_ref(adata, "target", reference=REFERENCE, output="polars")
    df_pd = pyscx.accel.pdex_ref(adata, "target", reference=REFERENCE, output="pandas")

    assert isinstance(df_pd, pd.DataFrame)
    # Same column names + order.
    assert list(df_pd.columns) == list(df_pl.columns)
    # Values identical to the polars result.
    pd.testing.assert_frame_equal(
        df_pd.reset_index(drop=True),
        df_pl.to_pandas().reset_index(drop=True),
    )
    # Unknown output value is rejected.
    with pytest.raises(ValueError):
        pyscx.accel.pdex_ref(adata, "target", reference=REFERENCE, output="bogus")
