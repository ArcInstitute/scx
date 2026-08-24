"""Generate pinned cell-eval / arc-bench reference values for `scx-accel`.

Run it from the repo root, in the uv venv — never system Python:

    .venv/bin/python benchmarks/scripts/generate_eval_metrics_references.py

Paste the output into
`scx-accel/src/eval_metrics/cell_eval_reference_values.rs`, then `cargo fmt`.

## Why this script exists

`docs/scanpy.md` claims the perturbation-evaluation metrics are "numerically
equivalent to the Python references within the tolerances below", and cites
`pyscx/tests/test_cell_eval_parity.py` as the evidence. That file
`importorskip`s `cell_eval`, `arc_bench` and `polars`, and **none of the six
conda envs has them** — nor does CI, whose Python-bindings job installs
scanpy/pydeseq2/polars and not these. So the claim rested entirely on a local
run in a developer venv.

Across `scx-accel/src/eval_metrics/` exactly three external numbers were pinned
before this: sklearn's AMI/NMI/ARI in `clustering.rs`. Everything else was
hand-derived from *reading* the reference implementation, or an f32-vs-f64
comparison between two SCX code paths — the "sibling implementations agreeing"
shape ORG-7.21-4 exists to retire.

## Every value here was PRODUCED by cell-eval or arc-bench

  * bulk metrics      <- `cell_eval.metrics._anndata.{pearson_delta,mse,mae,mse_delta,mae_delta}`
  * discrimination    <- `cell_eval.metrics._anndata.discrimination_score`
  * e-distance        <- `cell_eval.metrics._anndata.edistance`
  * knockdown         <- `arc_bench.tools.normalize_transform.core.*`
  * the pseudobulk means the first two consume <- `PerturbationAnndataPair._bulk_anndata`

That last one matters: the means are taken from cell-eval's own bulk step rather
than recomputed here, so the Rust kernel is handed exactly the input cell-eval
scored. Recomputing them would compare two metrics over two slightly different
matrices and call the difference a tolerance.

## Two claims are deliberately NOT pinned

  * **Mixed-tie discrimination ranks.** `numpy.argsort`'s default kind is
    `quicksort`, which is not stable, so the reference's own tie behaviour is
    implementation-defined. `docs/scanpy.md` already carves this out and
    `test_cell_eval_parity.py` pins SCX's stable-argsort semantics as a
    deliberate *divergence*.
  * **`clustering_agreement`.** Leiden is stochastic; the documented bar is an
    aggregate `atol=0.15`, which is not a number worth freezing.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

from _rust_literals import f32_literal, f64_literal, rust_matrix

N_GENES = 8
CELLS_PER_PERT = 4
N_PERTS = 8  # plus the control group
CONTROL = "control"
SEED_REAL, SEED_PRED = 70201, 70202
# `compute_knockdown_efficiency`'s finite guard; arc-bench's default.
KD_EPS = 1e-9

PERTS = [CONTROL] + [f"gene_{i}" for i in range(N_PERTS)]
GENES = [f"gene_{i}" for i in range(N_GENES)]
LABELS = np.repeat(PERTS, CELLS_PER_PERT)
N_OBS = len(LABELS)


def cells(seed: int) -> np.ndarray:
    """Continuous per-cell values, stored f32.

    **Not** rounded to integers, and that is the point. A first version of this
    fixture used integer counts; the pseudobulk means then landed on multiples of
    `1/CELLS_PER_PERT` and two perturbations' L1 distances came out exactly equal
    (`29.25` twice). On a tie, cell-eval's rank depends on `numpy.argsort`'s
    default `quicksort`, which is **not stable**, so its answer there is
    implementation-defined — `docs/scanpy.md` says the discrimination claim is
    "not claimed on mixed ties" and `discrimination_cell_eval_tests.rs` pins
    SCX's stable-argsort semantics as a deliberate divergence. Pinning a tied
    fixture would have frozen the reference's arbitrary choice as the contract.
    `check_untied` below refuses to emit one.

    The values are emitted through `f32_literal`, so what Rust reads is bit-equal
    to what cell-eval scored.
    """
    rng = np.random.default_rng(seed)
    return rng.exponential(3.0, size=(N_OBS, N_GENES)).astype(np.float32)


def build_pair(real: np.ndarray, pred: np.ndarray):
    import anndata as ad
    import scipy.sparse as sp
    from cell_eval import PerturbationAnndataPair

    def wrap(x: np.ndarray):
        a = ad.AnnData(sp.csr_matrix(x))
        a.var_names = GENES
        a.obs["perturbation"] = LABELS
        a.obs_names = [f"cell_{i}" for i in range(N_OBS)]
        return a

    return PerturbationAnndataPair(
        real=wrap(real), pred=wrap(pred), pert_col="perturbation", control_pert=CONTROL
    ), wrap(real)


def main() -> int:
    warnings.filterwarnings("ignore")
    try:
        from importlib.metadata import version

        from arc_bench.tools.normalize_transform.core import (
            compute_control_baseline,
            compute_knockdown_efficiency,
            compute_log_deviation,
        )
        from cell_eval.metrics._anndata import (
            discrimination_score,
            edistance,
            mae,
            mae_delta,
            mse,
            mse_delta,
            pearson_delta,
        )
    except ImportError as e:
        raise SystemExit(
            f"// cannot reach cell-eval / arc-bench: {e}\n"
            f"// They are editable installs in the repo's uv venv; no conda env has\n"
            f"// them. Run this with .venv/bin/python from the repo root."
        ) from e

    print(
        f"// numpy {version('numpy')}, cell-eval {version('cell-eval')}, "
        f"arc-bench {version('arc-bench')}"
    )
    print(f"// generated by benchmarks/scripts/{os.path.basename(__file__)}")
    print()

    real, pred = cells(SEED_REAL), cells(SEED_PRED)
    pair, real_ad = build_pair(real, pred)

    # --- the pseudobulk means cell-eval scores, taken from cell-eval ---------
    names_r, means_r = pair._bulk_anndata(pair.real, "perturbation")
    names_p, means_p = pair._bulk_anndata(pair.pred, "perturbation")
    failures: list[str] = []
    if list(names_r) != list(names_p):
        failures.append("cell-eval's bulk row order differs between real and pred")
    order = [str(n) for n in names_r]
    if CONTROL not in order:
        failures.append(f"the control group {CONTROL!r} is not in cell-eval's bulk rows")
    ctrl_idx = order.index(CONTROL)
    scored = [p for p in order if p != CONTROL]

    print(f"pub const CE_N_GENES: usize = {N_GENES};")
    print(f"pub const CE_N_ROWS: usize = {len(order)};")
    print(f"pub const CE_CTRL_IDX: usize = {ctrl_idx};")
    print(f"pub const CE_N_OBS: usize = {N_OBS};")
    print(f"pub const CE_KD_EPS: f64 = {f64_literal(KD_EPS)};")
    print()
    print("/// Row order of the pseudobulk means, as cell-eval emitted it.")
    print(
        "pub const CE_ROW_NAMES: [&str; CE_N_ROWS] = ["
        + ", ".join(f'"{n}"' for n in order)
        + "];"
    )
    print("/// Gene names, in fixture column order.")
    print(
        "pub const CE_GENE_NAMES: [&str; CE_N_GENES] = ["
        + ", ".join(f'"{g}"' for g in GENES)
        + "];"
    )
    print("/// Per-cell perturbation labels, in fixture row order.")
    print(
        "pub const CE_CELL_LABELS: [&str; CE_N_OBS] = ["
        + ", ".join(f'"{n}"' for n in LABELS)
        + "];"
    )
    print()
    print(rust_matrix("CE_MEANS_REAL", np.asarray(means_r, dtype=np.float64)))
    print()
    print(rust_matrix("CE_MEANS_PRED", np.asarray(means_p, dtype=np.float64)))
    print()
    print(rust_matrix("CE_CELLS_REAL", real, ty="f32"))
    print()
    print(rust_matrix("CE_CELLS_PRED", pred, ty="f32"))
    print()

    def emit_1d(name: str, values, ty: str = "f64", n: str | None = None) -> None:
        lit = f32_literal if ty == "f32" else f64_literal
        body = ", ".join(lit(float(v)) for v in values)
        size = n or str(len(list(values)))
        print(f"pub const {name}: [{ty}; {size}] = [{body}];")

    # --- bulk metrics -------------------------------------------------------
    bulk = {
        "PEARSON_DELTA": pearson_delta(pair),
        "MSE": mse(pair),
        "MAE": mae(pair),
        "MSE_DELTA": mse_delta(pair),
        "MAE_DELTA": mae_delta(pair),
    }
    print("// --- cell-eval bulk metrics, in CE_SCORED_NAMES order ---")
    print(
        "pub const CE_SCORED_NAMES: [&str; CE_N_ROWS - 1] = ["
        + ", ".join(f'"{n}"' for n in scored)
        + "];"
    )
    for name, table in bulk.items():
        keys = {str(k) for k in table}
        if keys != set(scored):
            failures.append(
                f"{name}: cell-eval scored {sorted(keys)}, not the {len(scored)} "
                f"non-control groups"
            )
            continue
        emit_1d(f"CE_{name}", [table[k] for k in table], n="CE_N_ROWS - 1")
    print()

    # --- discrimination -----------------------------------------------------
    #
    # Refuse to pin a tied fixture: see `cells`. The distances are recomputed here
    # from the pinned means purely to detect the tie — the *scores* still come
    # from cell-eval.
    def check_untied(metric: str, excl: bool) -> None:
        er = np.array([means_r[i] - means_r[ctrl_idx] for i in range(len(order)) if i != ctrl_idx])
        ep = np.array([means_p[i] - means_p[ctrl_idx] for i in range(len(order)) if i != ctrl_idx])
        for p_i, target in enumerate(scored):
            keep = np.array([g != target for g in GENES]) if excl else np.ones(N_GENES, bool)
            a, b = er[:, keep], ep[p_i][keep]
            if metric == "l1":
                d = np.abs(a - b).sum(axis=1)
            elif metric == "l2":
                d = np.sqrt(((a - b) ** 2).sum(axis=1))
            else:
                na = np.linalg.norm(a, axis=1) * np.linalg.norm(b)
                d = 1.0 - np.where(na > 0, (a @ b) / np.where(na > 0, na, 1), 0.0)
            if int((d == d[p_i]).sum()) > 1:
                failures.append(
                    f"discrimination {metric}/excl={excl}: perturbation {target!r} "
                    f"ties with another at distance {d[p_i]!r}. On a tie cell-eval's "
                    f"rank comes from numpy's unstable quicksort, which docs/scanpy.md "
                    f"explicitly does not claim — reseed rather than pin it"
                )

    print("// --- cell-eval discrimination_score ---")
    for metric in ("l1", "l2", "cosine"):
        for excl in (True, False):
            check_untied("l2" if metric == "l2" else metric, excl)
            table = discrimination_score(
                pair, metric=metric, exclude_target_gene=excl
            )
            suffix = "EXCL" if excl else "ALL"
            emit_1d(
                f"CE_DISCRIM_{metric.upper()}_{suffix}",
                [table[k] for k in table],
                n="CE_N_ROWS - 1",
            )
    print()

    # --- e-distance ---------------------------------------------------------
    ed = float(edistance(pair))
    print("// --- cell-eval edistance: the real-vs-pred correlation, one scalar ---")
    print(f"pub const CE_EDISTANCE_CORR: f64 = {f64_literal(ed)};")
    print()

    # --- knockdown ----------------------------------------------------------
    baseline = np.asarray(
        compute_control_baseline(real_ad, "perturbation", CONTROL), dtype=np.float64
    )
    kd = np.asarray(
        compute_knockdown_efficiency(real_ad, baseline, "perturbation", CONTROL),
        dtype=np.float64,
    )
    import scanpy as sc

    logged = real_ad.copy()
    sc.pp.log1p(logged)
    logdev = np.asarray(
        compute_log_deviation(logged, np.log1p(baseline), "perturbation", CONTROL),
        dtype=np.float64,
    )
    print("// --- arc-bench knockdown, on the same fixture ---")
    emit_1d("CE_KD_BASELINE", baseline, n="CE_N_GENES")
    print("/// NaN where arc-bench reports no value: control cells, and cells whose")
    print("/// perturbation names no gene in the matrix.")
    print(
        "pub const CE_KD_IS_NAN: [bool; CE_N_OBS] = ["
        + ", ".join("true" if np.isnan(v) else "false" for v in kd)
        + "];"
    )
    emit_1d("CE_KD_EFFICIENCY", np.nan_to_num(kd, nan=0.0), n="CE_N_OBS")
    emit_1d("CE_KD_LOG_DEVIATION", np.nan_to_num(logdev, nan=0.0), n="CE_N_OBS")
    print()

    # --- premise checks -----------------------------------------------------
    if np.isnan(kd).all():
        failures.append("every knockdown value is NaN; the fixture scores nothing")
    n_real = int((~np.isnan(kd)).sum())
    if n_real < N_PERTS:
        failures.append(
            f"only {n_real} cells got a knockdown value; the perturbation labels "
            f"must name genes present in the matrix"
        )
    if not np.array_equal(np.isnan(kd), np.isnan(logdev)):
        failures.append("knockdown and log-deviation disagree on which cells are NaN")
    for metric in ("l1", "l2", "cosine"):
        a = discrimination_score(pair, metric=metric, exclude_target_gene=True)
        if len({round(float(v), 12) for v in a.values()}) < 2:
            failures.append(
                f"discrimination_score({metric}) is constant across perturbations, "
                f"so the pinned table would be satisfied by any implementation that "
                f"returns that constant"
            )
    if not np.isfinite(ed):
        failures.append(f"cell-eval's edistance correlation is {ed!r}")

    print("// Measured, at generation time:")
    print(f"//   {len(scored)} scored perturbations, control at row {ctrl_idx}")
    print(f"//   edistance real-vs-pred correlation: {ed:.12f}")
    print(f"//   {n_real} of {N_OBS} cells carry a knockdown value")
    print(
        "//   discrimination l1/excl spread: "
        + ", ".join(
            f"{float(v):.4f}"
            for v in discrimination_score(pair, metric="l1", exclude_target_gene=True).values()
        )
    )

    if failures:
        raise SystemExit(
            "// the fixture no longer supports the claims it is pinned for:\n  "
            + "\n  ".join(failures)
        )
    print("// every table above was produced by cell-eval or arc-bench.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
