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
