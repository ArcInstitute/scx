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


def make_perturb_adata_sparse(
    n_perts, n_donors, n_genes, cells_per, seed=0, density=0.08
) -> ad.AnnData:
    """Like `make_perturb_adata` but with a **sparse** CSR `X` at realistic
    single-cell density (~5–10% nonzero), via low per-cell Poisson means.

    Why this matters (Stage B B0 profiling): the dense `make_perturb_adata` makes
    pseudobulk **aggregation** ~62% of `pdex_nb_glm` wall-time, which Amdahl-caps
    the GPU-fit speedup at ~1.35× — an artifact of dense input. Real Perturb-seq is
    sparse; the per-gene **fit** cost is independent of input density (it runs on
    the dense aggregated pseudobulk), while aggregation scales with nnz. A sparse
    fixture is the representative workload for the GPU-vs-CPU sweep."""
    import scipy.sparse as sp

    rng = np.random.default_rng(seed)
    perts = [REFERENCE] + [f"ko_{i}" for i in range(n_perts - 1)]
    donors = [f"d{j}" for j in range(n_donors)]

    # Per-cell mean chosen so P(count>0) ≈ density (Poisson: 1 - e^-mean).
    mean_level = -np.log(max(1e-6, 1.0 - density))
    base = rng.uniform(0.5 * mean_level, 1.5 * mean_level, size=n_genes)
    ramp = np.linspace(-1.5, 1.5, n_genes)
    effect = {REFERENCE: np.ones(n_genes)}
    for i, p in enumerate(perts[1:]):
        scale = 0.5 + 0.5 * ((i % 3) + 1)
        sign = 1.0 if i % 2 == 0 else -1.0
        effect[p] = 2.0 ** (sign * scale * ramp)
    donor_factor = {d: rng.uniform(0.92, 1.08, size=n_genes) for d in donors}

    blocks, pert_labels, donor_labels = [], [], []
    for p in perts:
        for d in donors:
            mean = base * effect[p] * donor_factor[d]
            blk = rng.poisson(mean[None, :], size=(cells_per, n_genes)).astype(np.float32)
            blocks.append(sp.csr_matrix(blk))
            pert_labels.extend([p] * cells_per)
            donor_labels.extend([d] * cells_per)

    x = sp.vstack(blocks).tocsr()
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


# (n_perts, n_donors, n_genes, cells_per_combo) for the GPU many-target sweep.
# n_donors drives n_sub per target (= 2·n_donors: target reps + reference reps),
# kept at 3 → n_sub=6, the target many-target regime (n_sub ≈ 4–8). Many
# perturbations × ~18k genes is where the CPU sweep costs minutes.
GPU_SIZES = [
    (20, 3, 18_000, 30),
    (50, 3, 18_000, 30),
    (100, 3, 18_000, 30),
]
GPU_SMOKE_SIZES = [(6, 3, 2_000, 20)]


def bench_gpu_size(n_perts, n_donors, n_genes, cells_per, reps) -> dict:
    """GPU-vs-CPU wall-time + ranking concordance + route check for the
    many-target ``pdex_nb_glm`` sweep. The CPU baseline is the **saturated
    multi-core** fit (rayon over genes), not single-thread.

    Uses a **sparse** fixture (realistic ~8% density): the per-gene fit cost is
    density-independent (it runs on the dense aggregated pseudobulk), while
    aggregation scales with nnz, so a dense fixture would be aggregation-bound and
    understate the GPU-fit speedup."""
    adata = make_perturb_adata_sparse(n_perts, n_donors, n_genes, cells_per)

    def run(dev, a):
        return pyscx.accel.pdex_nb_glm(
            a, "perturbation", REFERENCE, stratify_by=["donor"],
            min_cells_per_group=1, device=dev,
        )

    # Per-phase profiles (populated only when SCX_NBGLM_PROFILE=1; harmless
    # zeros otherwise). Reset before each path so the snapshot isolates that
    # path's phases (the shared host-tail phases — wald_cooks/mtc/assembly —
    # are what cap the GPU speedup, since the GPU fit is a small fraction of wall).
    # NB: do NOT `adata.copy()` per rep — pdex_nb_glm only reads X/obs (it stamps
    # uns, harmless to repeat), and a per-rep deep copy of the sparse matrix
    # (~150MB) is a measurement artifact that inflates both paths' wall and
    # deflates the ratio (Stage B B0 finding). Measure the real per-call cost.
    def _profiled(dev):
        pyscx.accel.nb_glm_profile_reset()
        t, df = _time(lambda: run(dev, adata), reps)
        return t, df, pyscx.accel.nb_glm_profile_snapshot()

    cpu_s, cpu_df, cpu_prof = _profiled("cpu")
    gpu_s, gpu_df, gpu_prof = _profiled("gpu")

    # Route check (separate call so timing isn't perturbed by the uns write).
    a = adata.copy()
    run("gpu", a)
    route = a.uns["scx_accel"]["pdex_nb_glm"]["route"]

    rho = None
    try:
        merged = cpu_df.to_pandas().merge(
            gpu_df.to_pandas(), on=["target", "feature"], suffixes=("_cpu", "_gpu")
        )
        if len(merged):
            rho = _spearman(
                merged["log2_fold_change_cpu"].to_numpy(),
                merged["log2_fold_change_gpu"].to_numpy(),
            )
    except Exception:
        pass

    return {
        "n_perts": n_perts,
        "n_donors": n_donors,
        "n_sub_per_target": 2 * n_donors,
        "n_genes": n_genes,
        "n_cells": int(adata.n_obs),
        "cpu_s": round(cpu_s, 4),
        "gpu_s": round(gpu_s, 4),
        "speedup": round(cpu_s / gpu_s, 2) if gpu_s > 0 else None,
        "rho_log2fc": rho,
        "route": route,
        "nb_glm_route_gpu_correct": 1.0 if route.startswith("gpu_nb_glm") else 0.0,
        # Per-phase ms over `reps` runs (SCX_NBGLM_PROFILE=1). The GPU profile's
        # wald_cooks+mtc+assembly vs gpu_mle_fit+gpu_shrink_fit split is the
        # Stage-B decision signal.
        "cpu_profile_ms": {k: v["ms"] for k, v in cpu_prof.items()},
        "gpu_profile_ms": {k: v["ms"] for k, v in gpu_prof.items()},
    }


def generate_gpu_report(rows: list[dict]) -> str:
    lines = [
        "## GPU vs CPU: pdex_nb_glm many-target sweep",
        "",
        "_CPU baseline = saturated multi-core (rayon), same machine._",
        "",
        "| perts | n_sub | genes | cells | cpu (s) | gpu (s) | speedup | ρ(log2FC) | route |",
        "|------:|------:|------:|------:|--------:|--------:|--------:|----------:|:------|",
    ]
    for r in rows:
        def fmt(v, nd=3):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) else ("—" if v is None else str(v))
        lines.append(
            f"| {r['n_perts']} | {r['n_sub_per_target']} | {r['n_genes']} | {r['n_cells']} "
            f"| {fmt(r['cpu_s'])} | {fmt(r['gpu_s'])} | {fmt(r['speedup'], 2)} "
            f"| {fmt(r['rho_log2fc'])} | `{r['route']}` |"
        )

    # Per-phase breakdown (only meaningful under SCX_NBGLM_PROFILE=1). Surfaces
    # whether the GPU path is fit-bound or host-tail-bound (the Stage-B lever).
    phases = [
        "transpose", "cpu_mle_pass", "cpu_shrink_pass", "gpu_mle_fit",
        "gpu_trend_prior", "gpu_shrink_fit", "wald_cooks", "mtc", "assembly",
    ]
    if any(sum(r.get("gpu_profile_ms", {}).values()) > 0 for r in rows):
        lines += [
            "",
            "### Per-phase wall-time (ms, summed over reps; SCX_NBGLM_PROFILE)",
            "",
            "GPU path:",
            "",
            "| perts | " + " | ".join(phases) + " |",
            "|------:|" + "|".join(["---:"] * len(phases)) + "|",
        ]
        for r in rows:
            prof = r.get("gpu_profile_ms", {})
            cells = " | ".join(f"{prof.get(p, 0.0):.1f}" for p in phases)
            lines.append(f"| {r['n_perts']} | {cells} |")
        lines += ["", "CPU path:", "",
                  "| perts | " + " | ".join(phases) + " |",
                  "|------:|" + "|".join(["---:"] * len(phases)) + "|"]
        for r in rows:
            prof = r.get("cpu_profile_ms", {})
            cells = " | ".join(f"{prof.get(p, 0.0):.1f}" for p in phases)
            lines.append(f"| {r['n_perts']} | {cells} |")
    lines.append("")
    return "\n".join(lines)


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
        " benchmarks/README.md). Build pyscx with `maturin develop --release`"
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
    ap.add_argument(
        "--gpu",
        choices=["auto", "on", "off"],
        default="auto",
        help="Run the GPU-vs-CPU pdex_nb_glm sweep (auto: only when a GPU is present)",
    )
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

    # GPU-vs-CPU many-target sweep — the Stage-A go/no-go measurement.
    run_gpu = args.gpu == "on" or (args.gpu == "auto" and pyscx.accel.gpu_available())
    if args.gpu == "on" and not pyscx.accel.gpu_available():
        print("WARNING: --gpu=on but no CUDA GPU detected; the GPU sweep will fall back to CPU.")
    gpu_rows = []
    if run_gpu:
        try:
            __import__("polars")  # pdex_nb_glm emits the polars cell-eval schema
            gpu_sizes = GPU_SMOKE_SIZES if args.smoke else GPU_SIZES
            for sz in gpu_sizes:
                print(f"gpu size {sz} ...", flush=True)
                gpu_rows.append(bench_gpu_size(*sz, reps=args.reps))
        except ImportError:
            print("polars not installed — skipping the GPU pdex_nb_glm sweep.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    md = generate_report(rows, fitter_rows)
    if gpu_rows:
        md = md + "\n" + generate_gpu_report(gpu_rows)
    (RESULTS_DIR / f"{args.out_name}.md").write_text(md)
    (RESULTS_DIR / f"{args.out_name}.json").write_text(
        json.dumps(
            {"pseudobulk_dex": rows, "direct_fitter": fitter_rows, "gpu_vs_cpu": gpu_rows},
            indent=2,
        )
    )
    print(md)
    print(f"\nWrote {RESULTS_DIR / (args.out_name + '.md')} and .json")


if __name__ == "__main__":
    main()
