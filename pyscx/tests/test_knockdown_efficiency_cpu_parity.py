"""CPU regression tests for `pyscx.accel.knockdown_efficiency`.

`scx_accel::compute_knockdown_efficiency` looks up each cell's value at
its target gene's column via a binary search over the row's CSR indices
(see `csr_get_value` in `scx-accel/src/eval_metrics/knockdown.rs`). That
binary search silently returns 0.0 on an unsorted CSR whenever the
target column sits out-of-order in `indices`, collapsing every
perturbed cell's `KnockDownEfficiency` to `1.0 - 0.0 / (baseline + eps)
≈ 1.0` regardless of actual expression — a uniformly degenerate
failure mode.

Real h5ad files routinely arrive with `has_sorted_indices == False`.
The fix at `pyscx::accel::eval_metrics::knockdown_efficiency` routes
the extracted CSR through `pyscx::anndata::ensure_csr` before handing
its raw `indices` array to the scx-accel kernel.

This test mirrors `test_rank_genes_groups_cpu_parity.py` and
`test_pdex_ref_parity.py::test_pdex_ref_cpu_unsorted_scipy_csr_matches_sorted`,
substituting `knockdown_efficiency` for the DE call.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import pyscx

# Reuse the unsorted-CSR builder from the shared fixtures module. The
# knockdown test needs a bespoke AnnData (perturbation labels match gene
# names; values at the target gene are non-zero), so we don't reuse
# `_make_adata` directly.
from _pdex_fixtures import _csr_with_descending_indices


SEED = 0
N_CONTROL = 30
N_PERT_PER_GROUP = 15
N_VARS = 12
CONTROL_LABEL = "control"
# Perturbations target the first three genes by name. Each target gene's
# value in perturbed cells is suppressed (but still non-zero) so
# `csr_get_value` has something real to read — the regression manifests
# when binary_search drops the lookup, not when the underlying value is 0.
TARGET_GENES = ["gene_0", "gene_1", "gene_2"]


def _make_knockdown_adata(seed: int = SEED) -> ad.AnnData:
    """Synthetic AnnData where perturbation labels match gene names.

    - `N_CONTROL` control cells with moderate uniform expression (Poisson
      around 5.0 across all genes).
    - `N_PERT_PER_GROUP` cells per target gene; perturbed cells are
      identical to controls except the target gene's value is suppressed
      to a Poisson around 1.0 — non-zero, so the bug isn't masked by
      coincidental sparsity at the target column.
    """
    rng = np.random.default_rng(seed)
    n_obs = N_CONTROL + N_PERT_PER_GROUP * len(TARGET_GENES)
    gene_names = [f"gene_{j}" for j in range(N_VARS)]
    gene_idx = {g: i for i, g in enumerate(gene_names)}

    counts = rng.poisson(5.0, size=(n_obs, N_VARS)).astype(np.float32)
    pert_labels: list[str] = [CONTROL_LABEL] * N_CONTROL
    for tg in TARGET_GENES:
        pert_labels.extend([tg] * N_PERT_PER_GROUP)
    # Suppress the target-gene value for perturbed cells, but keep it
    # strictly positive so binary_search has a real value to return.
    for cell_idx, label in enumerate(pert_labels):
        if label == CONTROL_LABEL:
            continue
        # Sample 1 + Poisson(0.5) so the value is always ≥ 1, never 0.
        counts[cell_idx, gene_idx[label]] = 1.0 + float(rng.poisson(0.5))

    obs_df = pd.DataFrame(
        {"perturbation": pert_labels},
        index=[f"cell_{i}" for i in range(n_obs)],
    )
    var_df = pd.DataFrame({"gene_id": gene_names}, index=gene_names)
    return ad.AnnData(X=counts, obs=obs_df, var=var_df)


def test_knockdown_efficiency_cpu_unsorted_scipy_csr_matches_sorted():
    """`knockdown_efficiency` must produce identical per-cell outputs on
    an unsorted scipy CSR (`has_sorted_indices == False`) as on the
    canonical sorted equivalent.

    Pre-fix failure mode: `csr_get_value`'s `binary_search` over each
    row's `indices` returns 0.0 on the unsorted CSR for any target gene
    whose column index is positioned out-of-order, so
    `KnockDownEfficiency` collapses to ≈1.0 for every perturbed cell
    regardless of actual target-gene expression.
    """
    adata_sorted = _make_knockdown_adata()
    adata_sorted.X = sp.csr_matrix(adata_sorted.X)
    adata_sorted.X.sort_indices()
    assert adata_sorted.X.has_sorted_indices

    adata_unsorted = adata_sorted.copy()
    adata_unsorted.X = _csr_with_descending_indices(adata_sorted)
    assert not adata_unsorted.X.has_sorted_indices

    # knockdown_efficiency writes results into adata.obs; operate on
    # copies so the two runs don't clobber each other.
    a_sorted = adata_sorted.copy()
    a_unsorted = adata_unsorted.copy()

    pyscx.accel.knockdown_efficiency(
        a_sorted, pert_col="perturbation", control=CONTROL_LABEL
    )
    pyscx.accel.knockdown_efficiency(
        a_unsorted, pert_col="perturbation", control=CONTROL_LABEL
    )

    for col in ("KnockDownEfficiency", "KnockDownGeneFC"):
        s = a_sorted.obs[col].to_numpy()
        u = a_unsorted.obs[col].to_numpy()
        # Finite mask must match (control / non-matching cells are NaN
        # in both runs).
        s_fin = np.isfinite(s)
        u_fin = np.isfinite(u)
        np.testing.assert_array_equal(
            s_fin, u_fin, err_msg=f"finite-mask mismatch in obs[{col!r}]"
        )
        # Finite entries must be bit-for-bit equal — there is no
        # tolerance budget here: both runs go through the exact same
        # downstream f32 kernel; only the CSR ordering differs.
        np.testing.assert_allclose(
            u[s_fin],
            s[s_fin],
            atol=0.0,
            rtol=0.0,
            err_msg=f"obs[{col!r}] differs between sorted and unsorted CSR",
        )

    # Sanity: the sorted run must produce non-trivial efficiencies.
    # Pre-fix, the unsorted run collapsed to all 1.0 (KD = 1 - 0/baseline);
    # the sorted run is the floor of "real signal" — if it's also all
    # 1.0 (or all NaN) the fixture lost its target-gene signal and the
    # equality assertion above wouldn't catch the bug.
    s_eff = a_sorted.obs["KnockDownEfficiency"].to_numpy()
    s_eff_finite = s_eff[np.isfinite(s_eff)]
    assert s_eff_finite.size > 0, (
        "no finite KnockDownEfficiency values in sorted run — fixture "
        "lost its perturbed-cell mass"
    )
    # Suppressed-but-non-zero target values give KD strictly < 1.0
    # because we sampled `1 + Poisson(0.5)` rather than 0. Pre-fix
    # behaviour: all perturbed cells would read 1.0.
    n_strictly_below_one = int(np.sum(s_eff_finite < 0.99))
    assert n_strictly_below_one > 0, (
        f"sorted run produced no KnockDownEfficiency < 0.99 "
        f"({s_eff_finite.size} finite values) — either the fixture "
        f"changed or knockdown_efficiency CPU is broken in a different way"
    )

    # Caller's AnnData must not be mutated (ensure_csr called with in_place=False).
    # `a_unsorted` is the deep-copy that was actually handed to
    # `knockdown_efficiency` — that's the only object the function
    # could have mutated.
    assert not a_unsorted.X.has_sorted_indices, (
        "ensure_csr(in_place=False) mutated caller's CSR — "
        "has_sorted_indices flipped to True after knockdown_efficiency"
    )
