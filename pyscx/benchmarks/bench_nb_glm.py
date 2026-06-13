#!/usr/bin/env python
"""Standalone benchmark: Rust-native NB-GLM vs PyDESeq2 (spec §16, Phase 5).

Measures wall-clock and ranking concordance of the SCX NB-GLM pseudobulk DE
backend against PyDESeq2 (the real competitor, spec §1) on stratified synthetic
Perturb-seq data, plus the pure-fitter throughput (genes/sec) of
`pyscx.accel.nb_glm` on an already-pseudobulked matrix.

NOTE: This is a **standalone** bench. It is intentionally **not** wired into
`benchmarks/comprehensive/scripts/gate_candidate.py` — PyDESeq2's reference
implementation OOMs in the comprehensive correctness gate (192-core numba), so
the gate carries zero correctness rows for it. NB-GLM's ranking-parity guard
lives in `pyscx/tests/test_pdex_nb_glm.py` (deterministic, cheap). This script
exists purely for positioning numbers and is reported to `benchmarks/results/`.

For representative numbers, build pyscx in **release** first — a `maturin develop`
debug build runs the fitter ~10–15× slower:
    cd pyscx && ../.venv/bin/maturin develop --release --features hdf5

Usage:
    python pyscx/benchmarks/bench_nb_glm.py            # default grid
    python pyscx/benchmarks/bench_nb_glm.py --smoke    # one tiny size
    python pyscx/benchmarks/bench_nb_glm.py --no-pydeseq2
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import pyscx

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

REFERENCE = "control"

# (n_perts, n_donors, n_genes, cells_per_combo)
DEFAULT_SIZES = [
    (4, 4, 2_000, 40),
    (8, 4, 5_000, 40),
    (8, 6, 10_000, 50),
]
SMOKE_SIZES = [(3, 3, 500, 20)]


def make_perturb_adata(n_perts, n_donors, n_genes, cells_per, seed=0) -> ad.AnnData:
    """Stratified synthetic Perturb-seq counts: control + (n_perts-1) KO
    perturbations × n_donors donors (the replicate stratifier). Graded,
    gene-specific multiplicative effects on a near-uniform base so per-gene power
    is roughly constant (keeps NB-GLM vs PyDESeq2 rankings concordant)."""
    rng = np.random.default_rng(seed)
    perts = [REFERENCE] + [f"ko_{i}" for i in range(n_perts - 1)]
    donors = [f"d{j}" for j in range(n_donors)]

    base = rng.uniform(45.0, 55.0, size=n_genes)
    ramp = np.linspace(-1.5, 1.5, n_genes)
    effect = {REFERENCE: np.ones(n_genes)}
    for i, p in enumerate(perts[1:]):
        scale = 0.5 + 0.5 * ((i % 3) + 1)  # vary effect strength per pert
        sign = 1.0 if i % 2 == 0 else -1.0
        effect[p] = 2.0 ** (sign * scale * ramp)
    donor_factor = {d: rng.uniform(0.92, 1.08, size=n_genes) for d in donors}

    blocks, pert_labels, donor_labels = [], [], []
    for p in perts:
        for d in donors:
            mean = base * effect[p] * donor_factor[d]
            blocks.append(rng.poisson(mean[None, :], size=(cells_per, n_genes)).astype(np.float32))
            pert_labels.extend([p] * cells_per)
            donor_labels.extend([d] * cells_per)

    x = np.vstack(blocks)
    obs = pd.DataFrame(
        {"perturbation": pert_labels, "donor": donor_labels},
        index=[f"cell_{i}" for i in range(x.shape[0])],
    )
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    return ad.AnnData(X=x, obs=obs, var=var)


def _time(fn, reps: int) -> tuple[float, object]:
    """Median wall-clock over `reps` runs; returns (median_s, last_result)."""
    times, result = [], None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), result


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return None
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return None
    # Index [0] (not `.statistic`) for SciPy < 1.10 backward compatibility.
    return float(spearmanr(a[mask], b[mask])[0])


def bench_size(n_perts, n_donors, n_genes, cells_per, reps, use_pydeseq2) -> dict:
    adata = make_perturb_adata(n_perts, n_donors, n_genes, cells_per)
    kw = dict(
        groupby=["perturbation", "donor"],
        test_col="perturbation",
        reference=REFERENCE,
        min_cells_per_group=1,
    )

    nb_s, nb_df = _time(
        lambda: pyscx.accel.pseudobulk_dex(adata.copy(), backend="nb_glm", **kw), reps
    )

    row = {
        "n_perts": n_perts,
        "n_donors": n_donors,
        "n_genes": n_genes,
        "cells_per_combo": cells_per,
        "n_cells": int(adata.n_obs),
        "nb_glm_s": round(nb_s, 4),
        "pydeseq2_s": None,
        "speedup": None,
        "rho_log2fc": None,
        "rho_padj": None,
    }

    if use_pydeseq2:
        try:
            pdq_s, pdq_df = _time(
                lambda: pyscx.accel.pseudobulk_dex(adata.copy(), backend="pydeseq2", **kw),
                reps,
            )
            row["pydeseq2_s"] = round(pdq_s, 4)
            row["speedup"] = round(pdq_s / nb_s, 2) if nb_s > 0 else None
            merged = nb_df.merge(pdq_df, on=["gene", "target"], suffixes=("_nb", "_pdq"))
            if len(merged):
                row["rho_log2fc"] = _spearman(
                    merged["log2FoldChange_nb"].to_numpy(),
                    merged["log2FoldChange_pdq"].to_numpy(),
                )
                row["rho_padj"] = _spearman(
                    merged["padj_nb"].to_numpy(), merged["padj_pdq"].to_numpy()
                )
        except Exception as e:  # pydeseq2 missing or failed — record and continue
            row["pydeseq2_s"] = f"skipped: {type(e).__name__}: {e}"

    return row


def bench_direct_fitter(n_genes=20_000, n_samples=20, n_features=3, reps=3) -> dict:
    """Pure-fitter throughput of accel.nb_glm on an already-pseudobulked matrix
    (no aggregation / no per-target loop)."""
    rng = np.random.default_rng(0)
    half = n_samples // 2
    counts = np.empty((n_samples, n_genes), dtype=np.float64)
    for s in range(n_samples):
        level = rng.uniform(5.0, 105.0, size=n_genes)
        trt = 1.8 if s >= half else 1.0
        counts[s] = np.round(level * trt * rng.uniform(0.8, 1.2, size=n_genes))
    design = np.zeros((n_samples, n_features))
    design[:, 0] = 1.0
    design[half:, 1] = 1.0
    if n_features > 2:
        design[:, 2:] = rng.uniform(0.0, 1.0, size=(n_samples, n_features - 2))

    fit_s, _ = _time(lambda: pyscx.accel.nb_glm(counts, design, contrast=1), reps)
    return {
        "n_genes": n_genes,
        "n_samples": n_samples,
        "n_features": n_features,
        "fit_s": round(fit_s, 4),
        "genes_per_sec": round(n_genes / fit_s) if fit_s > 0 else None,
    }


def generate_report(sizes_rows: list[dict], fitter_rows: list[dict]) -> str:
    sysinfo = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    lines = [
        "# NB-GLM benchmark (Rust-native vs PyDESeq2)",
        "",
        "_Standalone bench — not part of the comprehensive regression gate (see"
        " `GPU-NB-GLM-SPEC.md` §22.6). Build pyscx with `maturin develop --release`"
        " for representative timings._",
        "",
        f"- platform: `{sysinfo['platform']}`",
        f"- cpu_count: {sysinfo['cpu_count']} | python {sysinfo['python']} | numpy {sysinfo['numpy']}",
        "",
        "## pseudobulk_dex: nb_glm vs pydeseq2",
        "",
        "| perts | donors | genes | cells | nb_glm (s) | pydeseq2 (s) | speedup | ρ(log2FC) | ρ(padj) |",
        "|------:|-------:|------:|------:|-----------:|-------------:|--------:|----------:|--------:|",
    ]
    for r in sizes_rows:
        def fmt(v, nd=3):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ("—" if v is None else str(v))
        lines.append(
            f"| {r['n_perts']} | {r['n_donors']} | {r['n_genes']} | {r['n_cells']} "
            f"| {fmt(r['nb_glm_s'])} | {fmt(r['pydeseq2_s'])} | {fmt(r['speedup'], 2)} "
            f"| {fmt(r['rho_log2fc'])} | {fmt(r['rho_padj'])} |"
        )
    lines += [
        "",
        "## Pure-fitter throughput (`accel.nb_glm`, pre-aggregated)",
        "",
    ]
    for f in fitter_rows:
        lines.append(
            f"- {f['n_genes']} genes × {f['n_samples']} samples × "
            f"{f['n_features']} features: **{f['fit_s']:.3f} s** "
            f"(~{f['genes_per_sec']:,} genes/sec)"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="One tiny size only")
    ap.add_argument("--no-pydeseq2", action="store_true", help="Skip the PyDESeq2 comparison")
    ap.add_argument("--reps", type=int, default=3, help="Repeats per timing (median)")
    ap.add_argument("--out-name", default="nb_glm_benchmark")
    args = ap.parse_args()

    sizes = SMOKE_SIZES if args.smoke else DEFAULT_SIZES
    use_pydeseq2 = not args.no_pydeseq2
    if use_pydeseq2:
        try:
            __import__("pydeseq2")
        except ImportError:
            print("pydeseq2 not installed — running nb_glm-only (pass --no-pydeseq2 to silence).")
            use_pydeseq2 = False

    rows = []
    for sz in sizes:
        print(f"size {sz} ...", flush=True)
        rows.append(bench_size(*sz, reps=args.reps, use_pydeseq2=use_pydeseq2))

    # Pure-fitter throughput: a small "typical pseudobulk" point plus the spec's
    # key perf target (30k × 200 × 5, §15) for a directly comparable number.
    fitter_dims = [(20_000, 20, 3)] if args.smoke else [(20_000, 20, 3), (30_000, 200, 5)]
    fitter_rows = [bench_direct_fitter(*d, reps=args.reps) for d in fitter_dims]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = generate_report(rows, fitter_rows)
    (RESULTS_DIR / f"{args.out_name}.md").write_text(md)
    (RESULTS_DIR / f"{args.out_name}.json").write_text(
        json.dumps({"pseudobulk_dex": rows, "direct_fitter": fitter_rows}, indent=2)
    )
    print(md)
    print(f"\nWrote {RESULTS_DIR / (args.out_name + '.md')} and .json")


if __name__ == "__main__":
    main()
