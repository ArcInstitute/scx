"""`calculate_qc_metrics` / `filter_genes` fused-pass equivalence and pass count.

The QC row axis (`total_counts`, `n_genes_by_counts`, every
`total_counts_<qc_var>`) and the QC gene axis (`total_counts`,
`n_cells_by_counts`) are each computed in a **single** shard scan. Previously
each statistic drove its own full decode of every shard: 4 + one-per-`qc_var`.

Two things are pinned here:

1. **Bit-identity.** The fused kernels must agree *exactly* with the
   per-statistic ones. The array-protocol dunders (``X.sum(axis=…)`` /
   ``X.getnnz(axis=…)``) still route through the unfused
   ``col_sums_raw`` / ``col_nnz_raw`` / ``row_sums_raw`` / ``row_nnz_raw``
   kernels, so they are an independent same-order reference — not a numpy
   re-implementation with different summation order.
2. **Pass count.** A decode-bucket assertion via ``cpu_profile_snapshot()``, so
   a future refactor that re-splits the passes fails loudly instead of silently
   costing a full extra decode per statistic.
"""

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest


N_OBS, N_VARS = 240, 60


@pytest.fixture
def fused_adata():
    """Counts AnnData with no layers/obsm (see test_qc_metrics_projection)."""
    import anndata
    import pandas as pd
    import scipy.sparse as sp

    rng = np.random.RandomState(7)
    # Non-integer values on purpose: with integer counts every partial sum is an
    # exact f64 and `assert_array_equal` would pass under ANY accumulation
    # order, making the bit-identity assertions vacuous.
    dense = (rng.random_sample((N_OBS, N_VARS)) * 1e3).astype(np.float32)
    dense[rng.random_sample((N_OBS, N_VARS)) > 0.35] = 0
    return anndata.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(N_OBS)]),
        var=pd.DataFrame(index=[f"g{j}" for j in range(N_VARS)]),
    )


@pytest.fixture
def multishard_path(fused_adata, tmp_dir):
    """Written with a small shard target so the file has several shards."""
    import pyscx

    path = str(tmp_dir / "qc_fused.scx")
    pyscx.from_anndata(fused_adata, path, shard_size=50)
    return path


# The fixture's smallest row sum is ~6365, so the original 4000 kept every one
# of the 240 cells. That was harmless until `filter_cells` learned to skip an
# all-kept mask entirely (a no-op filter used to install an identity
# `kept_to_global`, permanently closing the CSC capability gate): from then on
# the four `*deleted*` cases below collapsed onto their non-deleted twins and
# the deletion-aware fused kernels stopped being exercised at all. Above the
# median (~10639) so the cut is unambiguous, and `_open` asserts it happened.
DROP_CELLS_MIN_COUNTS = 10_000


def _open(path, *, project=None, drop_cells=None, normalize=False):
    import pyscx

    adata = pyscx.open(path).to_anndata(backed=True)
    if drop_cells is not None:
        before = adata.n_obs
        pyscx.accel.filter_cells(adata, min_counts=drop_cells)
        assert 0 < adata.n_obs < before, (
            f"min_counts={drop_cells} kept {adata.n_obs}/{before} cells — the "
            "deletion-vector cases need an actual cut to mean anything"
        )
    if project is not None:
        adata.X.set_col_projection([int(c) for c in project])
        adata._var = adata.var.iloc[project].copy()
    if normalize:
        pyscx.accel.normalize_total(adata, target_sum=1e4)
    return adata


PROJECTION = [3, 7, 11, 19, 23, 31, 44, 51, 58]

CASES = {
    "plain": {},
    "projected": {"project": PROJECTION},
    "deleted": {"drop_cells": DROP_CELLS_MIN_COUNTS},
    "projected+deleted": {"project": PROJECTION, "drop_cells": DROP_CELLS_MIN_COUNTS},
    "lazy": {"normalize": True},
    "lazy+deleted": {"drop_cells": DROP_CELLS_MIN_COUNTS, "normalize": True},
    "lazy+projected": {"project": PROJECTION, "normalize": True},
    "lazy+projected+deleted": {
        "project": PROJECTION,
        "drop_cells": DROP_CELLS_MIN_COUNTS,
        "normalize": True,
    },
}


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("n_qc", [0, 1, 3])
def test_fused_matches_unfused_reference(multishard_path, case, n_qc):
    """Fused QC output matches an independently computed reference.

    Backed cases use the array-protocol dunders, which still drive the unfused
    one-statistic-per-scan kernels in the same accumulation order — so equality
    is asserted **exactly**.

    Lazy cases use the materialized visible matrix instead. That was originally
    because the lazy `sum` / `getnnz` dunders called the *unprojected*
    `streaming_row_sums` and so were not a valid reference under a column
    projection; Phase 4.0a fixed that, but the materialized matrix is kept here
    deliberately — it is an oracle *independent* of the kernels under test,
    where the dunders now share the projection machinery with them. numpy's
    pairwise summation reorders the adds, so these compare at a tight tolerance
    rather than exactly.
    """
    import pyscx

    adata = _open(multishard_path, **CASES[case])
    n_visible = adata.n_vars
    is_lazy = bool(CASES[case].get("normalize"))

    if is_lazy:
        mat = np.asarray(adata.X[:, :].todense(), dtype=np.float64)
        ref_total = mat.sum(axis=1)
        ref_ngenes = (mat != 0).sum(axis=1).astype(np.int64)
        ref_gene_total = mat.sum(axis=0)
        ref_ncells = (mat != 0).sum(axis=0).astype(np.int64)
        compare = lambda a, b, **kw: np.testing.assert_allclose(  # noqa: E731
            a, b, rtol=1e-12, **kw
        )
    else:
        mat = None
        ref_total = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
        ref_ngenes = np.asarray(adata.X.getnnz(axis=1)).ravel().astype(np.int64)
        ref_gene_total = np.asarray(adata.X.sum(axis=0)).ravel().astype(np.float64)
        ref_ncells = np.asarray(adata.X.getnnz(axis=0)).ravel().astype(np.int64)
        compare = np.testing.assert_array_equal

    qc_vars = []
    for k in range(n_qc):
        mask = np.zeros(n_visible, dtype=bool)
        mask[k::4] = True
        adata.var[f"qc{k}"] = mask
        qc_vars.append(f"qc{k}")

    pyscx.accel.calculate_qc_metrics(adata, qc_vars=qc_vars or None)

    compare(adata.obs["total_counts"].to_numpy(dtype=np.float64), ref_total)
    np.testing.assert_array_equal(
        adata.obs["n_genes_by_counts"].to_numpy(dtype=np.int64), ref_ngenes
    )
    compare(adata.var["total_counts"].to_numpy(dtype=np.float64), ref_gene_total)
    np.testing.assert_array_equal(
        adata.var["n_cells_by_counts"].to_numpy(dtype=np.int64), ref_ncells
    )

    for qc_var in qc_vars:
        subset_cols = np.nonzero(adata.var[qc_var].to_numpy(dtype=bool))[0]
        if is_lazy:
            expected = mat[:, subset_cols].sum(axis=1)
        else:
            # Re-project the dataset to exactly the subset genes and read its
            # row sums through the same unfused kernel.
            ondisk = (
                [PROJECTION[i] for i in subset_cols]
                if "project" in CASES[case]
                else list(subset_cols)
            )
            ref_adata = _open(
                multishard_path,
                project=ondisk,
                drop_cells=CASES[case].get("drop_cells"),
            )
            expected = np.asarray(ref_adata.X.sum(axis=1)).ravel().astype(np.float64)
        compare(
            adata.obs[f"total_counts_{qc_var}"].to_numpy(dtype=np.float64),
            expected,
            err_msg=f"{qc_var} subset sum diverged from the reference",
        )


@pytest.mark.parametrize("case", ["plain", "projected", "lazy+projected"])
def test_partitioning_qc_vars_sum_to_total(multishard_path, case):
    """qc_var subsets that partition the gene axis must sum to total_counts.

    Independent of the reference kernels — it constrains the fused accumulator's
    bitmask dispatch (every nonzero contributes to exactly the subsets it
    belongs to, once).
    """
    import pyscx

    adata = _open(multishard_path, **CASES[case])
    n_visible = adata.n_vars
    for k in range(3):
        mask = np.zeros(n_visible, dtype=bool)
        mask[k::3] = True
        adata.var[f"part{k}"] = mask

    pyscx.accel.calculate_qc_metrics(adata, qc_vars=["part0", "part1", "part2"])

    parts = sum(
        adata.obs[f"total_counts_part{k}"].to_numpy(dtype=np.float64) for k in range(3)
    )
    np.testing.assert_allclose(
        parts, adata.obs["total_counts"].to_numpy(dtype=np.float64), rtol=1e-12
    )


def test_overlapping_qc_vars_double_count(multishard_path):
    """A gene in two subsets contributes to both (bitmask, not exclusive)."""
    import pyscx

    adata = _open(multishard_path)
    all_true = np.ones(adata.n_vars, dtype=bool)
    adata.var["a"] = all_true
    adata.var["b"] = all_true
    pyscx.accel.calculate_qc_metrics(adata, qc_vars=["a", "b"])

    total = adata.obs["total_counts"].to_numpy(dtype=np.float64)
    for name in ("a", "b"):
        np.testing.assert_allclose(
            adata.obs[f"total_counts_{name}"].to_numpy(dtype=np.float64),
            total,
            rtol=1e-12,
        )


def test_empty_qc_var_zero_fills_without_changing_the_column_set(multishard_path):
    """An all-false mask zero-fills, and publishes the same three columns.

    This used to assert `"log1p_total_counts_none" not in adata.obs.columns` —
    an empty mask "kept its historical shape" and skipped that one column. That
    made the output schema a function of the *data* (did any gene match?) rather
    than of the call, and it disagreed with scanpy, which writes
    `log1p(0) == 0.0` like any other value. A dataset with no annotated
    mitochondrial genes then produced a frame missing a column its siblings had.

    The advisory `emit_qc_advisories` raises for an empty mask is still the
    signal that something is wrong with the mask; a missing column is not.
    """
    import pyscx

    adata = _open(multishard_path)
    adata.var["none"] = np.zeros(adata.n_vars, dtype=bool)
    adata.var["some"] = np.arange(adata.n_vars) < 5

    with pytest.warns(UserWarning):
        pyscx.accel.calculate_qc_metrics(adata, qc_vars=["none", "some"], log1p=True)

    assert np.all(adata.obs["total_counts_none"].to_numpy() == 0.0)
    assert np.all(adata.obs["pct_counts_none"].to_numpy() == 0.0)
    assert np.all(adata.obs["log1p_total_counts_none"].to_numpy() == 0.0)
    assert "log1p_total_counts_some" in adata.obs.columns


def test_inplace_false_returns_frames_without_writing(multishard_path):
    """The `(obs_df, var_df)` return path survives the row-pass restructure."""
    import pyscx

    adata = _open(multishard_path)
    adata.var["mt"] = np.arange(adata.n_vars) < 6
    obs_df, var_df = pyscx.accel.calculate_qc_metrics(
        adata, qc_vars=["mt"], inplace=False
    )

    assert "total_counts" not in adata.obs.columns
    assert "total_counts" not in adata.var.columns
    assert list(obs_df.index) == list(adata.obs_names)
    assert list(var_df.index) == list(adata.var_names)

    ref_total = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
    np.testing.assert_array_equal(
        obs_df["total_counts"].to_numpy(dtype=np.float64), ref_total
    )
    subset = _open(multishard_path, project=list(range(6)))
    np.testing.assert_array_equal(
        obs_df["total_counts_mt"].to_numpy(dtype=np.float64),
        np.asarray(subset.X.sum(axis=1)).ravel().astype(np.float64),
    )


def test_many_qc_vars_chunk_rather_than_reject(multishard_path):
    """>64 subsets exceed the bitmask width, so the row pass repeats per 64.

    The pre-fusion implementation supported any number of `qc_vars` (one scan
    each), so rejecting past 64 would have been a capability regression.
    """
    import pyscx

    adata = _open(multishard_path)
    names = []
    for k in range(70):
        adata.var[f"m{k}"] = np.arange(adata.n_vars) % 70 == k
        names.append(f"m{k}")

    pyscx.accel.calculate_qc_metrics(adata, qc_vars=names)

    # Every subset published, and — since the 70 masks partition the gene axis —
    # they must sum back to total_counts.
    parts = np.zeros(adata.n_obs, dtype=np.float64)
    for name in names:
        assert f"total_counts_{name}" in adata.obs.columns
        parts += adata.obs[f"total_counts_{name}"].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(
        parts, adata.obs["total_counts"].to_numpy(dtype=np.float64), rtol=1e-12
    )


def test_qc_var_beyond_first_chunk_is_correct(multishard_path):
    """A subset landing in the 2nd chunk (index >= 64) gets the right sum."""
    import pyscx

    adata = _open(multishard_path)
    names = []
    for k in range(66):
        # Only subset 65 (2nd chunk) selects anything; the rest are empty.
        adata.var[f"s{k}"] = (
            np.arange(adata.n_vars) < 5 if k == 65 else np.zeros(adata.n_vars, bool)
        )
        names.append(f"s{k}")

    with pytest.warns(UserWarning):
        pyscx.accel.calculate_qc_metrics(adata, qc_vars=names)

    ref = _open(multishard_path, project=list(range(5)))
    np.testing.assert_allclose(
        adata.obs["total_counts_s65"].to_numpy(dtype=np.float64),
        np.asarray(ref.X.sum(axis=1)).ravel().astype(np.float64),
        rtol=1e-12,
    )
    assert np.all(adata.obs["total_counts_s0"].to_numpy() == 0.0)


# ---------------------------------------------------------------------------
# Pass count
# ---------------------------------------------------------------------------

_DECODE_PROBE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import pyscx

    path = sys.argv[1]
    n_qc = int(sys.argv[2])
    case = sys.argv[3]
    op = sys.argv[4]

    adata = pyscx.open(path).to_anndata(backed=True)
    n_shards = adata.X.n_shards if hasattr(adata.X, "n_shards") else None
    if "deleted" in case:
        # Must match DROP_CELLS_MIN_COUNTS above — a threshold that keeps every
        # cell is skipped outright, so the deleted cases would silently become
        # duplicates of the plain ones.
        before = adata.n_obs
        pyscx.accel.filter_cells(adata, min_counts=10_000)
        assert 0 < adata.n_obs < before, "deleted case kept every cell"
    if "projected" in case:
        cols = [3, 7, 11, 19, 23, 31, 44, 51, 58]
        adata.X.set_col_projection(cols)
        adata._var = adata.var.iloc[cols].copy()
    if "lazy" in case:
        pyscx.accel.normalize_total(adata, target_sum=1e4)
    qc_vars = []
    for k in range(n_qc):
        mask = np.zeros(adata.n_vars, dtype=bool)
        mask[k::4] = True
        adata.var[f"qc{k}"] = mask
        qc_vars.append(f"qc{k}")

    pyscx.accel.cpu_profile_reset()
    if op == "qc":
        pyscx.accel.calculate_qc_metrics(adata, qc_vars=qc_vars or None)
    else:
        pyscx.accel.filter_genes(adata, min_cells=1, min_counts=1.0)
    snap = pyscx.accel.cpu_profile_snapshot()
    decodes = snap["decode_scx1"]["count"] + snap["decode_generic"]["count"]
    print(json.dumps({"enabled": snap["enabled"], "decodes": decodes,
                      "n_shards": n_shards}))
    """
)


def _probe_decodes(path, n_qc, case="plain", op="qc"):
    import json

    env = dict(os.environ, SCX_CPU_PROFILE="1")
    out = subprocess.run(
        [sys.executable, "-c", _DECODE_PROBE, path, str(n_qc), case, op],
        capture_output=True,
        text=True,
        env=env,
    )
    if out.returncode != 0:
        # `check=True` would swallow stderr behind CalledProcessError.
        raise AssertionError(
            f"decode probe failed (case={case}, op={op}):\n{out.stderr[-2000:]}"
        )
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    "case", ["plain", "projected", "deleted", "lazy", "lazy+projected"]
)
def test_filter_genes_decode_count_is_one_pass(multishard_path, case):
    """filter_genes with both thresholds decodes each shard once, every route.

    The lazy arm is the one that regressed in review: it kept two scans after
    the backed arm was fused.
    """
    res = _probe_decodes(multishard_path, 0, case=case, op="filter_genes")
    assert res["enabled"]
    n_shards = res["n_shards"]
    assert res["decodes"] == n_shards, (
        f"{case}: expected 1 pass over {n_shards} shards, saw {res['decodes']}"
    )


@pytest.mark.parametrize(
    "case", ["plain", "projected", "deleted", "lazy", "lazy+projected"]
)
@pytest.mark.parametrize("n_qc", [0, 1, 3])
def test_decode_count_is_two_passes(multishard_path, n_qc, case):
    """QC decodes each shard exactly twice — one row pass, one column pass.

    Before the fusion this was `(2 + n_qc) x n_shards` for the row-side
    statistics plus `2 x n_shards` for the gene axis. Pinned on every fused
    route, not just backed-plain, so a re-split anywhere is caught.
    """
    res = _probe_decodes(multishard_path, n_qc, case=case)
    assert res["enabled"], "SCX_CPU_PROFILE=1 did not reach the child process"
    n_shards = res["n_shards"]
    assert n_shards and n_shards > 1, f"fixture must be multi-shard, got {n_shards}"
    assert res["decodes"] == 2 * n_shards, (
        f"{case}: expected 2 passes over {n_shards} shards, saw {res['decodes']} "
        f"decodes with n_qc={n_qc} — a statistic regained its own scan"
    )
