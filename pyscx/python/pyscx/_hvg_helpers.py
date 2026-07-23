"""Internal helpers for HVG computation called from Rust via PyO3.

This module is not part of the public API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def binned_dispersion_norm(
    log_means: np.ndarray,
    log_dispersions: np.ndarray,
    n_bins: int,
) -> list[float]:
    """Z-score normalize log-dispersions within mean-expression bins.

    Matches scanpy's seurat-flavor dispersion normalization:
    bins genes by ``pd.cut(log_means, bins=n_bins)``, then within each bin
    computes ``(log_disp - mean) / std`` (ddof=1).  NaN values (from
    zero-dispersion genes) are filled with 0.0.

    Parameters
    ----------
    log_means : ndarray of shape (n_vars,)
        ``log1p(mean_expression)`` per gene.
    log_dispersions : ndarray of shape (n_vars,)
        ``log(variance / mean)`` per gene. May contain NaN.
    n_bins : int
        Number of expression bins (scanpy default is 20).

    Returns
    -------
    list[float]
        Normalized dispersions, length ``n_vars``.
    """
    log_means = np.asarray(log_means, dtype=float)
    log_dispersions = np.asarray(log_dispersions, dtype=float)

    # Mirror scanpy's `_get_disp_stats` / `_postprocess_dispersions_seurat`
    # exactly (seurat flavor): bin by mean, aggregate per-bin avg=mean and
    # dev=std (ddof=1, NaN-skipping like pandas), then normalize
    # `(dispersion - avg) / dev`.
    df = pd.DataFrame({"means": log_means, "dispersions": log_dispersions})
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    stats = grouped.agg(avg="mean", dev="std")

    # Single-gene bins have NaN std. scanpy sets their normalized dispersion to
    # EXACTLY 1 via `dev = avg; avg = 0` (NOT `std = 1`, which would give 0).
    one_gene = stats["dev"].isnull()
    stats.loc[one_gene, "dev"] = stats.loc[one_gene, "avg"]
    stats.loc[one_gene, "avg"] = 0.0

    per_gene = stats.loc[df["mean_bin"]].set_index(df.index)
    disp_norm = (df["dispersions"] - per_gene["avg"]) / per_gene["dev"]
    # Leave NaN where scanpy would (zero-dispersion genes): the caller floors
    # NaN to -inf for selection, matching scanpy's `nan_to_num(nan=-inf)`.
    return disp_norm.to_numpy().tolist()
