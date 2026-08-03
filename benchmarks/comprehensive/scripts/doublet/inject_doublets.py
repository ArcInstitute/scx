"""Computational doublet injection — the benchmark's labelled truth.

Nothing on this cluster carries hashing or genotype doublet labels
(`pbmc3k`/`pbmc10k` obs is just `n_counts`; `tabula_sapiens_100k` has
`donor_id`, but those donors are separate libraries, so there are no
cross-donor doublets to label). Without labels, "scDblFinder called 4.7–7.1%"
is an observation rather than a measurement, which is exactly what acceptance
gate 7 objects to.

Injection closes that: sum the counts of two real cells and you know, exactly,
that the result is a doublet.

**What this truth is worth, and what it is not.** This is
``tasks/DOUBLET-DETECTION.md`` Category D, and that spec is blunt about the
limits: injected doublets are "controlled but may not reproduce all capture and
ambient RNA artifacts", and must not be used to tune production defaults
without also checking real labelled datasets. So every record here carries its
``evidence_type``, and results from injected truth must never be pooled with a
hashing- or genotype-labelled result. Summing two count vectors also models a
doublet as pure addition, which ignores that a real droplet has one capture
efficiency rather than two — heterotypic pairs are the honest part, homotypic
pairs much less so, which is why the two are labelled separately and reported
apart.

The output is an ordinary h5ad: real cells first in their original order, then
the injected rows. Row order matters to nothing downstream (every join in this
stack is by key), but keeping the real block intact means a diff against the
source is readable.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["InjectionSpec", "InjectionResult", "inject_doublets"]

# obs columns written by the injection. Named here so the benchmark module and
# the tests agree on the schema without restating strings.
TRUTH_LABEL = "truth_label"
TRUTH_KIND = "truth_kind"
TRUTH_SOURCE_A = "truth_source_a"
TRUTH_SOURCE_B = "truth_source_b"

DOUBLET = "doublet"
SINGLET = "singlet"


@dataclass(frozen=True)
class InjectionSpec:
    """How many doublets to inject, and from which pairs."""

    rate: float = 0.08
    """Injected doublets as a fraction of the *real* cell count."""

    seed: int = 0

    heterotypic_only: bool = False
    """Restrict pairs to different `cell_type_key` values.

    Off by default: a benchmark that only ever injects heterotypic pairs
    measures the easy half of the problem and reports it as the whole
    (`DOUBLET-DETECTION.md` makes the same point about species-mixing data).
    Both kinds are injected and labelled, so they can be scored apart.
    """

    max_source_cells: int | None = None
    """Optionally subsample the real cells first, to bound runtime."""


@dataclass(frozen=True)
class InjectionResult:
    path: Path
    n_real: int
    n_injected: int
    n_heterotypic: int
    n_homotypic: int
    category: str
    evidence_type: str
    spec: dict


def inject_doublets(
    adata,
    out_h5ad: Path,
    spec: InjectionSpec,
    *,
    cell_type_key: str | None = None,
):
    """Append synthetic doublets to *adata* and write the result to disk.

    Args:
        adata: source AnnData with **raw counts** in `X`.
        out_h5ad: destination.
        spec: injection parameters.
        cell_type_key: obs column used to classify a pair as heterotypic or
            homotypic. When absent every pair is recorded as `"unknown"` —
            legible, rather than silently labelling everything heterotypic.

    Returns:
        :class:`InjectionResult`.
    """
    import anndata as ad
    import pandas as pd
    import scipy.sparse as sp

    if not 0.0 < spec.rate < 1.0:
        raise ValueError(f"injection rate must be in (0, 1); got {spec.rate}")

    rng = np.random.default_rng(spec.seed)

    if spec.max_source_cells is not None and adata.n_obs > spec.max_source_cells:
        keep = rng.choice(adata.n_obs, size=spec.max_source_cells, replace=False)
        keep.sort()
        adata = adata[keep].copy()

    n_real = int(adata.n_obs)
    if n_real < 2:
        raise ValueError(f"need at least 2 cells to pair; got {n_real}")
    n_inject = max(1, int(round(n_real * spec.rate)))

    X = adata.X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    X = X.tocsr()

    types = None
    if cell_type_key is not None:
        if cell_type_key not in adata.obs.columns:
            raise KeyError(
                f"cell_type_key {cell_type_key!r} is not an obs column; obs "
                f"has {list(adata.obs.columns)}"
            )
        types = adata.obs[cell_type_key].astype(str).to_numpy()

    idx_a, idx_b = _sample_pairs(rng, n_real, n_inject, types,
                                 spec.heterotypic_only)

    # A doublet is the sum of the two cells' counts. Row-slicing a CSR by a
    # fancy index and adding is exact for integer counts held as f32 up to
    # 2**24, which every real UMI count is far below.
    synthetic = (X[idx_a] + X[idx_b]).tocsr()

    names = adata.obs_names.astype(str).to_numpy()
    synth_names = np.array(
        [f"injected_{i}_{names[a]}__{names[b]}"
         for i, (a, b) in enumerate(zip(idx_a, idx_b))],
        dtype=object,
    )

    if types is not None:
        kinds = np.where(types[idx_a] == types[idx_b], "homotypic", "heterotypic")
    else:
        kinds = np.full(n_inject, "unknown", dtype=object)

    # Real rows keep every original obs column; the truth columns are appended
    # to both blocks so the concatenated frame has one schema.
    real_obs = adata.obs.copy()
    real_obs[TRUTH_LABEL] = SINGLET
    real_obs[TRUTH_KIND] = "real"
    real_obs[TRUTH_SOURCE_A] = ""
    real_obs[TRUTH_SOURCE_B] = ""

    # Inherit the first parent's whole obs row, so batch/donor/cell-type stay
    # populated — a synthetic row with a NaN batch would be dropped by the
    # per-batch export and never scored at all.
    #
    # Taken as a positional row slice rather than column by column: casting
    # each column through `object` (the obvious way to index it) turns a
    # numeric column like `n_counts` into Python floats, `pd.concat` then makes
    # the merged column object-dtype, and anndata writes object columns as
    # variable-length strings — which fails with "Can't implicitly convert
    # non-string objects to strings" only at write time, on the first real
    # dataset that has a numeric obs column. `.iloc` preserves every dtype.
    synth_obs = adata.obs.iloc[idx_a].copy()
    synth_obs.index = pd.Index(synth_names, name=real_obs.index.name)
    synth_obs[TRUTH_LABEL] = DOUBLET
    synth_obs[TRUTH_KIND] = kinds
    synth_obs[TRUTH_SOURCE_A] = names[idx_a]
    synth_obs[TRUTH_SOURCE_B] = names[idx_b]

    obs = pd.concat([real_obs, synth_obs], axis=0)
    obs[TRUTH_LABEL] = obs[TRUTH_LABEL].astype(str)
    obs[TRUTH_KIND] = obs[TRUTH_KIND].astype(str)

    out = ad.AnnData(
        X=sp.vstack([X, synthetic]).tocsr(),
        obs=obs,
        var=adata.var.copy(),
    )
    out.obs_names = np.concatenate([names, synth_names.astype(str)])

    n_hetero = int((kinds == "heterotypic").sum())
    n_homo = int((kinds == "homotypic").sum())

    result = InjectionResult(
        path=Path(out_h5ad),
        n_real=n_real,
        n_injected=n_inject,
        n_heterotypic=n_hetero,
        n_homotypic=n_homo,
        category="injected",
        evidence_type=(
            "computational injection (DOUBLET-DETECTION.md Category D): exact "
            "labels, but does not reproduce capture or ambient-RNA artifacts; "
            "do not pool with hashing/genotype-labelled results"
        ),
        spec=asdict(spec),
    )
    out.uns["doublet_injection"] = {
        k: (str(v) if isinstance(v, Path) else v)
        for k, v in asdict(result).items()
    }

    Path(out_h5ad).parent.mkdir(parents=True, exist_ok=True)
    out.write_h5ad(out_h5ad)
    logger.info(
        "injected %d doublets (%d heterotypic, %d homotypic) into %d real "
        "cells -> %s", n_inject, n_hetero, n_homo, n_real, out_h5ad,
    )
    return result


def _sample_pairs(rng, n_real: int, n_inject: int, types, heterotypic_only: bool):
    """Sample `n_inject` ordered pairs of distinct row indices."""
    a = rng.choice(n_real, size=n_inject, replace=True)
    b = rng.choice(n_real, size=n_inject, replace=True)

    # Resample collisions rather than shifting the index: `b = (b + 1) % n`
    # would correlate the second parent with the first, and on a file sorted by
    # cell type that quietly makes almost every forced pair homotypic.
    for _ in range(64):
        bad = a == b
        if heterotypic_only and types is not None:
            bad |= types[a] == types[b]
        if not bad.any():
            break
        b = np.where(bad, rng.choice(n_real, size=n_inject, replace=True), b)
    else:
        bad = a == b
        if heterotypic_only and types is not None:
            bad |= types[a] == types[b]
        if bad.any():
            # Fail loud: silently emitting self-pairs would put rows labelled
            # "doublet" into the truth set that are just a scaled single cell,
            # and every accuracy number downstream would be quietly wrong.
            raise RuntimeError(
                f"could not sample {int(bad.sum())} distinct"
                f"{' heterotypic' if heterotypic_only else ''} pairs after 64 "
                "attempts — the dataset likely has too few cells or too few "
                "distinct cell types for this injection rate"
            )
    return a, b
