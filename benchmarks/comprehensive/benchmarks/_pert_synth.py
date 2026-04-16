"""
Synthetic perturbation dataset generator — shared fixture for the
cell_eval_parity_perf benchmark.

Adapted from ``pyscx/tests/test_cell_eval_parity.py`` (``_make_cell_eval_adata``
and ``_make_raw_count_adata``). Scaled up for benchmark-sized cell counts
(100K–1M+) and cached on disk keyed by ``(n_obs, n_vars, n_perts, seed)`` so
generation runs at most once per (size, seed) combination.

Invariants (required by cell-eval + arc-bench):
  * ``obs`` column ``"perturbation"`` with control label ``"control"``.
  * First ``n_perts - 1`` gene names match the non-control perturbation names
    (so ``knockdown_efficiency`` and ``discrimination_score`` target-gene
    lookup can find them).
  * >= 20 cells per perturbation (stable pseudobulk).
  * Values in ``[0, 15)`` with a fractional component — passes
    ``cell_eval.utils.guess_is_lognorm``.
  * ``real`` and ``pred`` share ``var_names``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from benchmarks.comprehensive.config import DATA_DIR

SYNTH_ROOT = DATA_DIR / "synthetic"


def _cache_dir(n_obs: int, n_vars: int, n_perts: int, seed: int) -> Path:
    return SYNTH_ROOT / f"pert_{n_obs}_{n_vars}_{n_perts}_{seed}"


def _rebalance_n_obs(n_obs: int, n_perts: int) -> int:
    cells_per_pert = max(20, n_obs // n_perts)
    return cells_per_pert * n_perts


def _generate_paired(
    n_obs: int, n_vars: int, n_perts: int, seed: int,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Generate a fresh paired (real, pred) dataset. No caching."""
    rng = np.random.default_rng(seed)

    pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]
    gene_names = [f"gene_{i}" for i in range(n_vars)]
    for pname in pert_names[1:]:
        if pname not in gene_names:
            raise ValueError(
                f"Perturbation '{pname}' not in gene_names "
                f"(need n_vars >= n_perts - 1)"
            )

    cells_per_pert = n_obs // n_perts
    labels: list[str] = []
    for name in pert_names:
        labels.extend([name] * cells_per_pert)
    n_obs_actual = len(labels)

    base = rng.exponential(5.0, size=n_vars).astype(np.float32)

    deltas: dict[str, np.ndarray] = {}
    for name in pert_names[1:]:
        delta = rng.normal(0, 2, size=n_vars).astype(np.float32)
        gene_idx = gene_names.index(name)
        delta[gene_idx] = -base[gene_idx] * 0.7  # 70% knockdown
        deltas[name] = delta

    # Build real expression (raw counts). Work row-by-row to keep memory bounded.
    X_real = np.zeros((n_obs_actual, n_vars), dtype=np.float32)
    X_pred = np.zeros((n_obs_actual, n_vars), dtype=np.float32)
    for i, label in enumerate(labels):
        noise_r = rng.poisson(0.5, size=n_vars).astype(np.float32)
        noise_p = rng.poisson(0.5, size=n_vars).astype(np.float32)
        if label == "control":
            X_real[i] = np.maximum(base + noise_r, 0)
            X_pred[i] = np.maximum(base + noise_p, 0)
        else:
            pred_delta = deltas[label] + rng.normal(0, 0.5, size=n_vars).astype(np.float32)
            X_real[i] = np.maximum(base + deltas[label] + noise_r, 0)
            X_pred[i] = np.maximum(base + pred_delta + noise_p, 0)

    X_real = np.round(X_real)
    X_pred = np.round(X_pred)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=gene_names)

    adata_real = ad.AnnData(X=X_real, obs=obs.copy(), var=var.copy())
    adata_pred = ad.AnnData(X=X_pred, obs=obs.copy(), var=var.copy())

    sc.pp.normalize_total(adata_real)
    sc.pp.log1p(adata_real)
    sc.pp.normalize_total(adata_pred)
    sc.pp.log1p(adata_pred)

    adata_real.X = sp.csr_matrix(adata_real.X)
    adata_pred.X = sp.csr_matrix(adata_pred.X)
    return adata_real, adata_pred


def make_paired_adata(
    n_obs: int, n_vars: int = 2000, n_perts: int = 50, seed: int = 42,
    *, cache: bool = True,
) -> Tuple[ad.AnnData, ad.AnnData]:
    """Return paired (real, pred) AnnData at the requested scale.

    With ``cache=True`` (default), result is persisted under
    ``$SCX_DATA_DIR/synthetic/pert_{n_obs}_{n_vars}_{n_perts}_{seed}/``.
    Subsequent calls with the same key re-read the cache instead of
    regenerating (generation at 1M × 2K is several minutes).
    """
    n_obs = _rebalance_n_obs(n_obs, n_perts)

    if not cache:
        return _generate_paired(n_obs, n_vars, n_perts, seed)

    cdir = _cache_dir(n_obs, n_vars, n_perts, seed)
    real_path = cdir / "real.h5ad"
    pred_path = cdir / "pred.h5ad"
    if real_path.exists() and pred_path.exists():
        return ad.read_h5ad(real_path), ad.read_h5ad(pred_path)

    cdir.mkdir(parents=True, exist_ok=True)
    real, pred = _generate_paired(n_obs, n_vars, n_perts, seed)
    real.write_h5ad(real_path, compression="gzip")
    pred.write_h5ad(pred_path, compression="gzip")
    return real, pred


def make_raw_counts(
    n_obs: int, n_vars: int = 2000, n_perts: int = 50, seed: int = 42,
    *, cache: bool = True,
) -> ad.AnnData:
    """Raw-count AnnData (not log1p'd) for knockdown-efficiency benchmarks."""
    n_obs = _rebalance_n_obs(n_obs, n_perts)

    if cache:
        cdir = _cache_dir(n_obs, n_vars, n_perts, seed)
        raw_path = cdir / "raw.h5ad"
        if raw_path.exists():
            return ad.read_h5ad(raw_path)

    rng = np.random.default_rng(seed + 10_000)
    pert_names = ["control"] + [f"gene_{i}" for i in range(n_perts - 1)]
    gene_names = [f"gene_{i}" for i in range(n_vars)]
    cells_per_pert = n_obs // n_perts
    labels = [name for name in pert_names for _ in range(cells_per_pert)]

    base = rng.exponential(5.0, size=n_vars).astype(np.float32)
    deltas: dict[str, np.ndarray] = {}
    for name in pert_names[1:]:
        d = rng.normal(0, 1, size=n_vars).astype(np.float32)
        d[gene_names.index(name)] = -base[gene_names.index(name)] * 0.7
        deltas[name] = d

    X = np.zeros((len(labels), n_vars), dtype=np.float32)
    for i, label in enumerate(labels):
        noise = rng.poisson(0.5, size=n_vars).astype(np.float32)
        if label == "control":
            X[i] = np.maximum(base + noise, 0)
        else:
            X[i] = np.maximum(base + deltas[label] + noise, 0)
    X = np.round(X).astype(np.float32)

    obs = pd.DataFrame({"perturbation": labels})
    var = pd.DataFrame(index=gene_names)
    adata = ad.AnnData(X=sp.csr_matrix(X), obs=obs, var=var)

    if cache:
        cdir = _cache_dir(n_obs, n_vars, n_perts, seed)
        cdir.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(cdir / "raw.h5ad", compression="gzip")

    return adata


def materialize_dataset(dataset_name: str, params: dict) -> Path:
    """Ensure the synthetic h5ad exists for ``dataset_name``.

    The orchestrator's dataset-existence check gates on ``dataset.h5ad_path``.
    This helper generates the real-side h5ad on demand and returns its path
    (so callers can set it as the dataset's h5ad path without triggering
    an OS existence error before the benchmark runs).
    """
    real, _ = make_paired_adata(**params)
    # real.h5ad already written by make_paired_adata with cache=True.
    cdir = _cache_dir(
        _rebalance_n_obs(params["n_obs"], params.get("n_perts", 50)),
        params.get("n_vars", 2000),
        params.get("n_perts", 50),
        params.get("seed", 42),
    )
    return cdir / "real.h5ad"
