"""Shared synthetic AnnData builders for DE / knockdown CPU+GPU parity tests.

Factored out of `test_pdex_ref_parity.py` so that downstream test modules
(`test_rank_genes_groups_cpu_parity.py`, `test_pdex_ref_gpu_parity.py`,
`test_rank_genes_groups_gpu_parity.py`, `test_knockdown_efficiency_cpu_parity.py`,
...) can reuse the same 90-cell × 15-gene 3-group fixture without
transitively triggering `pytest.importorskip("polars")` and
`pytest.importorskip("pdex")` from that file's module top.

This module imports only the minimum the fixtures need (numpy, scipy.sparse,
anndata, pandas) — no polars, no pdex.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SEED = 0
N_OBS = 90
N_VARS = 15
N_GROUPS = 3
REFERENCE = "non-targeting"


def as_polars(df):
    """Coerce a DE frame to polars, whatever container it arrived in.

    The DE accelerators return **pandas** by default (F6) and polars only on
    `output="polars"`, while upstream `pdex` always returns polars. Parity
    helpers compare column-wise with `pl` casts, so exactly one side needs
    converting — this is that conversion, in one place. Polars is imported
    lazily so this module stays polars-free at import time (see the module
    docstring): only a caller that actually compares frames pays for it.
    """
    import polars as pl

    return df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)


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
        g = c // per_group
        counts[c] = rng.poisson(base[g]).astype(np.float32)
    obs_df = pd.DataFrame(
        {"target": groups}, index=[f"cell_{i}" for i in range(N_OBS)]
    )
    var_df = pd.DataFrame(
        {"gene_id": [f"gene_{j}" for j in range(N_VARS)]},
        index=[f"gene_{j}" for j in range(N_VARS)],
    )
    return ad.AnnData(X=counts, obs=obs_df, var=var_df)


def _make_adata_special_genes(seed: int = SEED) -> ad.AnnData:
    """`_make_adata` with two controlled genes for fold-change edge cases:

    * gene 0: zero in **every** group → `0/0` (must report 0.0, not NaN/inf).
    * gene 1: zero in the reference, positive in the test groups → one-sided
      zero (`±inf` with epsilon=0).

    The remaining genes keep the standard 3-group signal so FDR has a realistic
    universe to correct over.
    """
    adata = _make_adata(seed)
    X = np.asarray(adata.X, dtype=np.float32).copy()
    ref_mask = adata.obs["target"].to_numpy() == REFERENCE
    X[:, 0] = 0.0  # zero everywhere
    X[ref_mask, 1] = 0.0  # zero in reference
    X[~ref_mask, 1] = 5.0  # positive in test groups
    adata.X = X
    return adata


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
