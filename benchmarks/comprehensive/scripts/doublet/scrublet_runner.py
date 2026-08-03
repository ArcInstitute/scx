"""Scrublet runner for the SCX doublet-interop benchmark.

Reads a JSON config from stdin, runs ``sc.pp.scrublet`` on one exported
per-batch h5ad, writes a CSV of the tool's NATIVE output columns, and emits a
JSON run record on stdout. Same protocol as ``scdblfinder_runner.R``.

Runs under its own interpreter (see :mod:`._tool_env`), so nothing here may
import pyscx or the benchmark package — this file is executed by
``<scx-bench>/bin/python``, not by the benchmark's own process.

The input is the h5ad ``pyscx.export_batches`` wrote, which is the real entry
point for a scanpy-resident tool. The output columns keep scanpy's own
spellings (``doublet_score`` / ``predicted_doublet``) so the benchmark drives
the actual ``doublet_import(tool="scrublet")`` profile rather than
pre-canonicalising and testing nothing.

Config:
    h5ad_path   per-batch h5ad written by export_batches
    out_csv     where to write the per-cell table
    batch       label recorded in the run record (informational)
    seed        RNG seed
    expected_doublet_rate   optional float; None uses scanpy's default
"""

from __future__ import annotations

import json
import sys
import time


def _peak_rss_kb() -> int:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _emit(payload: dict) -> None:
    print(json.dumps(payload))
    sys.exit(0)


def _fail(msg: str) -> None:
    print(json.dumps({"error": msg}))
    sys.exit(1)


def main() -> None:
    raw = sys.stdin.read()
    if not raw.strip():
        _fail("No input received on stdin")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(f"JSON parse error: {exc}")

    for field in ("h5ad_path", "out_csv"):
        if not config.get(field):
            _fail(f"Missing '{field}' in config")

    seed = int(config.get("seed", 0))

    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scanpy as sc
    except ImportError as exc:
        _fail(f"scrublet runner needs scanpy/anndata/pandas: {exc}")

    np.random.seed(seed)

    t0 = time.perf_counter()
    try:
        adata = ad.read_h5ad(config["h5ad_path"])
    except Exception as exc:  # noqa: BLE001 — surfaced as JSON, not a traceback
        _fail(f"read_h5ad failed: {type(exc).__name__}: {exc}")
    read_s = time.perf_counter() - t0

    if adata.n_obs == 0:
        _fail(f"{config['h5ad_path']} has 0 cells")
    # The join key. Catching a duplicated barcode here costs one line; catching
    # it downstream means either a duplicate-key error at import or scores
    # landing on the wrong cell, which nothing downstream would notice.
    names = adata.obs_names.astype(str)
    if not names.is_unique:
        n_dup = int(len(names) - names.nunique())
        _fail(f"obs_names are not unique within this batch "
              f"({n_dup} duplicated of {len(names)})")

    kwargs: dict = {"random_state": seed}
    rate = config.get("expected_doublet_rate")
    if rate is not None:
        kwargs["expected_doublet_rate"] = float(rate)

    t0 = time.perf_counter()
    try:
        # threshold is left unset on purpose: scrublet's automatic
        # thresholding is the thing a baseline should be measured with, and it
        # is also what makes scikit-image a hard requirement of this env.
        sc.pp.scrublet(adata, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _fail(f"sc.pp.scrublet failed: {type(exc).__name__}: {exc}")
    tool_s = time.perf_counter() - t0

    missing = [c for c in ("doublet_score", "predicted_doublet")
               if c not in adata.obs.columns]
    if missing:
        _fail(f"scrublet emitted no {missing}; obs has "
              f"{list(adata.obs.columns)}")

    out = pd.DataFrame({
        "barcode": names,
        "doublet_score": adata.obs["doublet_score"].to_numpy(),
        # Written as True/False text, which the annotation reader infers as a
        # nullable Boolean; the scrublet profile reads it as the call column.
        "predicted_doublet": adata.obs["predicted_doublet"].astype(bool),
    })
    out.to_csv(config["out_csv"], index=False)

    n_called = int(out["predicted_doublet"].sum())
    threshold = adata.uns.get("scrublet", {}).get("threshold")

    _emit({
        "tool": "scrublet",
        "batch": config.get("batch"),
        "out_csv": config["out_csv"],
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_called": n_called,
        "called_rate": n_called / adata.n_obs,
        "score_column": "doublet_score",
        "call_column": "predicted_doublet",
        "threshold": float(threshold) if threshold is not None else None,
        "read_s": read_s,
        "tool_s": tool_s,
        "peak_rss_kb": _peak_rss_kb(),
        "seed": seed,
        "versions": {
            "scanpy": sc.__version__,
            "anndata": ad.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
    })


if __name__ == "__main__":
    main()
