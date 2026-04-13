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
    log_means = np.asarray(log_means)
    log_dispersions = np.asarray(log_dispersions)
    n = len(log_means)
    disp_norm = np.zeros(n)

    mean_bins = pd.cut(log_means, bins=n_bins)
    for b in mean_bins.categories:
        mask = np.asarray(mean_bins == b)
        if mask.sum() == 0:
            continue
        vals = log_dispersions[mask]
        avg = np.nanmean(vals)
        std = np.nanstd(vals, ddof=1)
        if np.isnan(std) or std == 0:
            std = 1.0
        disp_norm[mask] = (vals - avg) / std

    disp_norm[np.isnan(disp_norm)] = 0.0
    return disp_norm.tolist()
