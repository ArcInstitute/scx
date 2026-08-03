"""Solo (scVI) runner for the SCX doublet-interop benchmark.

Same stdin/stdout protocol as the other runners. Solo is deliberately
**optional**: ``tasks/DOUBLET-DETECTION.md`` keeps it "in a separate optional
benchmark environment", and no env on this cluster ships scvi-tools.

So the interesting behaviour here is the *absence* path. When scVI is missing
this exits 0 with ``{"available": false, "reason": ...}`` rather than raising,
because a benchmark that dies on an optional comparator is worse than one that
records why the comparator did not run. The caller turns that into a legible
skip in the result metadata; a silent omission would look identical to a tool
that ran and called nothing.

Config: as ``scrublet_runner.py``, plus
    max_epochs   optional int passed to SOLO.train
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


def _unavailable(reason: str) -> None:
    """Exit 0 — an absent optional tool is a skip, not a failure."""
    print(json.dumps({"tool": "solo", "available": False, "reason": reason}))
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

    try:
        import scvi
        from scvi.external import SOLO
    except ImportError as exc:
        _unavailable(
            f"scvi-tools is not installed in this interpreter "
            f"({sys.executable}): {exc}. Solo is an optional comparator; see "
            "tasks/DOUBLET-DETECTION.md § Baseline tools to compare."
        )

    for field in ("h5ad_path", "out_csv"):
        if not config.get(field):
            _fail(f"Missing '{field}' in config")

    seed = int(config.get("seed", 0))

    import anndata as ad
    import numpy as np
    import pandas as pd

    scvi.settings.seed = seed

    t0 = time.perf_counter()
    try:
        adata = ad.read_h5ad(config["h5ad_path"])
    except Exception as exc:  # noqa: BLE001
        _fail(f"read_h5ad failed: {type(exc).__name__}: {exc}")
    read_s = time.perf_counter() - t0

    if adata.n_obs == 0:
        _fail(f"{config['h5ad_path']} has 0 cells")
    names = adata.obs_names.astype(str)
    if not names.is_unique:
        n_dup = int(len(names) - names.nunique())
        _fail(f"obs_names are not unique within this batch "
              f"({n_dup} duplicated of {len(names)})")

    t0 = time.perf_counter()
    try:
        scvi.model.SCVI.setup_anndata(adata)
        vae = scvi.model.SCVI(adata)
        train_kwargs = {}
        if config.get("max_epochs") is not None:
            train_kwargs["max_epochs"] = int(config["max_epochs"])
        vae.train(**train_kwargs)
        solo = SOLO.from_scvi_model(vae)
        solo.train(**train_kwargs)
        pred = solo.predict()
    except Exception as exc:  # noqa: BLE001
        _fail(f"SOLO failed: {type(exc).__name__}: {exc}")
    tool_s = time.perf_counter() - t0

    # `predict()` returns a per-cell probability frame; the doublet column is
    # the score, and `predict(soft=False)` the label. Read both explicitly
    # rather than positionally so a column reorder upstream is an error here.
    if "doublet" not in pred.columns:
        _fail(f"SOLO.predict() has no 'doublet' column; got "
              f"{list(pred.columns)}")
    label = solo.predict(soft=False)

    out = pd.DataFrame({
        "barcode": names,
        "solo_score": np.asarray(pred["doublet"], dtype="float64"),
        "solo_label": np.asarray(label, dtype=object).astype(str),
    })
    out.to_csv(config["out_csv"], index=False)

    n_called = int((out["solo_label"] == "doublet").sum())

    _emit({
        "tool": "solo",
        "available": True,
        "batch": config.get("batch"),
        "out_csv": config["out_csv"],
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_called": n_called,
        "called_rate": n_called / adata.n_obs,
        "score_column": "solo_score",
        "call_column": "solo_label",
        "read_s": read_s,
        "tool_s": tool_s,
        "peak_rss_kb": _peak_rss_kb(),
        "seed": seed,
        "versions": {
            "scvi": scvi.__version__,
            "anndata": ad.__version__,
            "python": sys.version.split()[0],
        },
    })


if __name__ == "__main__":
    main()
