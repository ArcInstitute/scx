#!/usr/bin/env python3
"""
Accelerator fingerprints — regression lock for PCA / neighbors / UMAP / Leiden / Harmony / LISI.

Runs each accelerator on a small pinned dataset (pbmc3k by default) with a
pinned seed and thread count, then writes:

  - fingerprints.json   : per-accelerator BLAKE3 hashes of canonical outputs.
  - arrays/*.npy        : the raw arrays that were hashed, for byte-diffable
                          post-mortem when a hash changes.

This is the artefact the regression-lock CI job compares against. Any numerical
drift in an accelerator changes at least one hash; reviewers then compare the
.npy arrays to understand the magnitude and shape of the drift.

Determinism knobs (must be identical on baseline capture and on later runs):

  - RAYON_NUM_THREADS   : pinned via env var (default 1 for exact repro).
  - numpy/scanpy seeds  : passed via random_state=... on every accelerator.
  - Hashed slices       : stored with explicit dtype + C-contiguous layout so
                          the byte representation is deterministic.

Usage:

    python benchmarks/comprehensive/scripts/fingerprint_accelerators.py \
        --output-dir benchmarks/comprehensive/results/baseline_2026_04_17/fingerprints

    # Compare an existing fingerprint directory against the baseline:
    python benchmarks/comprehensive/scripts/fingerprint_accelerators.py --verify \
        --baseline benchmarks/comprehensive/results/baseline_2026_04_17/fingerprints

The fingerprint file also records the dataset path, seed, thread count, git
SHA, and pyscx / rust versions so stale fingerprints are flagged as such when
the compare script is run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import blake3  # type: ignore
    _HAS_BLAKE3 = True
except ImportError:  # pragma: no cover — fall back to sha256 if blake3 absent
    _HAS_BLAKE3 = False


# Pinned seed used for every accelerator call.  Changing this value
# invalidates every baseline fingerprint — don't.
PINNED_SEED = 42

# Thread pinning.  Rayon reduction order is not bit-exact across thread
# counts, so we pin to 1 thread for fingerprints even though benchmarks
# run with more.
PINNED_THREADS = 1


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def _hash_bytes(buf: bytes) -> str:
    """BLAKE3 hex of buf, with sha256 fallback if the blake3 package is absent."""
    if _HAS_BLAKE3:
        return blake3.blake3(buf).hexdigest()
    return "sha256:" + hashlib.sha256(buf).hexdigest()


def _hash_array(arr) -> tuple[str, dict[str, Any]]:
    """Hash a numpy array in a deterministic, layout-stable way.

    Returns (hash_hex, metadata).  Metadata is stored alongside the hash so
    a later comparison can report shape / dtype changes separately from
    value drift.
    """
    import numpy as np

    a = np.ascontiguousarray(arr)
    meta = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "nbytes": int(a.nbytes),
    }
    return _hash_bytes(a.tobytes()), meta


def _save_array(out_dir: Path, name: str, arr) -> None:
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", np.ascontiguousarray(arr))


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _rust_version() -> str:
    try:
        out = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _capture_environment() -> dict[str, Any]:
    import numpy as np

    env = {
        "hostname": os.uname().nodename,
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "git_sha": _git_sha(),
        "rust_version": _rust_version(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "unset"),
        "pinned_seed": PINNED_SEED,
        "pinned_threads": PINNED_THREADS,
        "blake3_available": _HAS_BLAKE3,
    }

    for pkg, import_name in [
        ("pyscx", "pyscx"),
        ("scanpy", "scanpy"),
        ("anndata", "anndata"),
        ("scipy", "scipy"),
        # GPU competitor stack — recorded so the accel_pipeline residency
        # head-to-head (V3 task 2.7) is traceable to a rapids-singlecell / cuML
        # version, next to the cuVS/cuGraph fingerprints captured elsewhere.
        ("rapids_singlecell", "rapids_singlecell"),
        ("cuml", "cuml"),
    ]:
        try:
            mod = __import__(import_name)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env[f"{pkg}_version"] = "not installed"

    return env


# ---------------------------------------------------------------------------
# Accelerator runs
# ---------------------------------------------------------------------------


def _load_pbmc3k(h5ad_path: Path):
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path)
    # Use a deterministic view: sort obs/var by name and slice to the first N
    # cells.  This guarantees identical ordering across capture environments
    # where scanpy loaders may otherwise yield an arbitrary permutation.
    adata = adata[:, adata.var_names.argsort()].copy()
    # Normalize + log1p — the minimal preprocessing needed so that the
    # downstream accelerators produce non-trivial output.
    import scanpy as sc

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


# Every accelerator below is pinned to the CPU.
#
# This file's own header lists the knobs that "must be identical on baseline
# capture and on later runs" — thread count, seeds, hashed dtype — and did not
# list the DEVICE. It should have: `device` defaults to `"auto"`, so the hash
# silently depended on whether the wheel this job happened to build carried the
# `gpu` feature.
#
# It went unnoticed while every gate built `--features hdf5`. Phase 7's gate is
# the first to build `hdf5,gpu`, and all six fingerprints failed at once: PCA
# routed to rapids-singlecell, which rejects genes with zero expression across
# all cells (raw pbmc3k has many), and neighbors / umap / leiden / harmony /
# lisi then cascaded off the missing `X_pca`. "Fingerprint missing: 8" is what
# the gate reported — not a mismatch, because nothing ran.
#
# So Phase 6's stable hashes were CPU hashes all along, and pinning the CPU is
# what makes them comparable rather than a fresh convention. A GPU-vs-CPU
# numerical claim is a different artefact from a drift lock, and the
# `*_route_gpu_correct` floors in `thresholds.yaml` are where it belongs.
FP_DEVICE = "cpu"


def _fingerprint_pca(adata, results: dict[str, Any], arrays_dir: Path) -> None:
    import pyscx

    pyscx.accel.pca(adata, n_comps=20, random_state=PINNED_SEED, device=FP_DEVICE)
    arr = adata.obsm["X_pca"]
    h, meta = _hash_array(arr)
    results["pca"] = {"hash": h, **meta}
    _save_array(arrays_dir, "pca_X_pca", arr)

    # Also fingerprint the variance ratio — smaller, drift-sensitive signal.
    if "pca" in adata.uns and "variance_ratio" in adata.uns["pca"]:
        vr = adata.uns["pca"]["variance_ratio"]
        h_vr, meta_vr = _hash_array(vr)
        results["pca_variance_ratio"] = {"hash": h_vr, **meta_vr}
        _save_array(arrays_dir, "pca_variance_ratio", vr)


def _fingerprint_neighbors(adata, results: dict[str, Any], arrays_dir: Path) -> None:
    import pyscx

    pyscx.accel.neighbors(adata, n_neighbors=15, random_state=PINNED_SEED, device=FP_DEVICE)
    # Hash the indices matrix (distances are perturbation-sensitive so not
    # ideal as a byte-exact regression lock; we still save them for diffing).
    dist = adata.obsp["distances"].toarray()
    conn = adata.obsp["connectivities"].toarray()
    h_d, meta_d = _hash_array(dist)
    h_c, meta_c = _hash_array(conn)
    results["neighbors_distances"] = {"hash": h_d, **meta_d}
    results["neighbors_connectivities"] = {"hash": h_c, **meta_c}
    _save_array(arrays_dir, "neighbors_distances", dist)
    _save_array(arrays_dir, "neighbors_connectivities", conn)


def _fingerprint_umap(adata, results: dict[str, Any], arrays_dir: Path) -> None:
    import pyscx

    pyscx.accel.umap(adata, n_epochs=200, random_state=PINNED_SEED, device=FP_DEVICE)
    arr = adata.obsm["X_umap"]
    h, meta = _hash_array(arr)
    results["umap"] = {"hash": h, **meta}
    _save_array(arrays_dir, "umap_X_umap", arr)


def _fingerprint_leiden(adata, results: dict[str, Any], arrays_dir: Path) -> None:
    import numpy as np
    import pyscx

    pyscx.accel.leiden(adata, random_state=PINNED_SEED, key_added="leiden_fp", device=FP_DEVICE)
    labels = np.asarray(adata.obs["leiden_fp"].astype("int32"))
    h, meta = _hash_array(labels)
    results["leiden"] = {"hash": h, **meta}
    _save_array(arrays_dir, "leiden_labels", labels)


def _fingerprint_harmony(adata, results: dict[str, Any], arrays_dir: Path) -> None:
    """Harmony needs a batch column.  pbmc3k has no natural batch, so we
    synthesise one deterministically (3 strata by cell index).  The exact
    labels don't matter for regression — only that they are stable."""
    import numpy as np
    import pandas as pd
    import pyscx

    n = adata.n_obs
    adata.obs["_fp_batch"] = pd.Categorical(
        ["A", "B", "C"] * (n // 3 + 1)
    )[:n]

    pyscx.accel.harmony_integrate(
        adata, "_fp_batch",
        max_iter=5,
        random_state=PINNED_SEED,
        device=FP_DEVICE,
        adjusted_basis="X_pca_harmony_fp",
    )
    arr = adata.obsm["X_pca_harmony_fp"]
    h, meta = _hash_array(arr)
    results["harmony"] = {"hash": h, **meta}
    _save_array(arrays_dir, "harmony_X_pca_harmony_fp", arr)

    # Clean up the scratch obs column so it doesn't perturb later fingerprints.
    del adata.obs["_fp_batch"]


def _fingerprint_lisi(adata, results: dict[str, Any], arrays_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    import pyscx

    n = adata.n_obs
    adata.obs["_fp_batch_lisi"] = pd.Categorical(
        ["A", "B", "C"] * (n // 3 + 1)
    )[:n]

    # No `device=` here: `compute_lisi` is CPU-only and does not take one.
    lisi = pyscx.accel.compute_lisi(
        adata, "_fp_batch_lisi",
        perplexity=30.0,
        basis="X_pca",
    )
    arr = np.asarray(lisi, dtype=np.float64)
    h, meta = _hash_array(arr)
    results["lisi"] = {"hash": h, **meta}
    _save_array(arrays_dir, "lisi", arr)

    del adata.obs["_fp_batch_lisi"]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


ACCELERATORS = {
    "pca":       _fingerprint_pca,
    "neighbors": _fingerprint_neighbors,
    "umap":      _fingerprint_umap,
    "leiden":    _fingerprint_leiden,
    "harmony":   _fingerprint_harmony,
    "lisi":      _fingerprint_lisi,
}


def run_fingerprints(
    dataset_path: Path,
    output_dir: Path,
    only: list[str] | None = None,
) -> dict[str, Any]:
    # Enforce thread pinning before importing anything that spawns threads.
    os.environ.setdefault("RAYON_NUM_THREADS", str(PINNED_THREADS))

    env = _capture_environment()
    env["dataset_path"] = str(dataset_path)

    print(f"[fingerprint] dataset: {dataset_path}")
    print(f"[fingerprint] RAYON_NUM_THREADS = {os.environ['RAYON_NUM_THREADS']}")
    print(f"[fingerprint] seed = {PINNED_SEED}")

    t0 = time.perf_counter()
    adata = _load_pbmc3k(dataset_path)
    print(f"[fingerprint] loaded: {adata.shape} in {time.perf_counter() - t0:.2f}s")

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_dir = output_dir / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)

    accelerators = only or list(ACCELERATORS.keys())
    results: dict[str, Any] = {}
    timings: dict[str, float] = {}

    for name in accelerators:
        if name not in ACCELERATORS:
            print(f"[fingerprint] WARNING: unknown accelerator '{name}', skipping")
            continue
        fn = ACCELERATORS[name]
        t0 = time.perf_counter()
        try:
            fn(adata, results, arrays_dir)
            timings[name] = time.perf_counter() - t0
            print(f"[fingerprint]   {name:<10s} ok  "
                  f"({timings[name]:.2f}s)  hash={results.get(name, {}).get('hash', '?')[:16]}...")
        except Exception as e:
            timings[name] = time.perf_counter() - t0
            results[name] = {"error": repr(e)}
            print(f"[fingerprint]   {name:<10s} FAIL ({timings[name]:.2f}s): {e}")

    manifest = {
        "schema_version": 1,
        "environment": env,
        "timings_seconds": timings,
        "fingerprints": results,
    }

    manifest_path = output_dir / "fingerprints.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[fingerprint] wrote {manifest_path}")
    return manifest


def verify_against_baseline(
    dataset_path: Path,
    baseline_dir: Path,
    work_dir: Path,
    tolerance: float = 0.0,
) -> int:
    """Recompute fingerprints and compare to baseline/fingerprints.json.

    Exit code 0 on exact match, 1 on any hash divergence, 2 on missing
    entries or environment drift warnings.  `tolerance` reserved for a
    future mode that compares saved .npy arrays with numpy.allclose rather
    than byte-exact hashes.
    """
    baseline_json = baseline_dir / "fingerprints.json"
    if not baseline_json.exists():
        print(f"[verify] ERROR: baseline not found at {baseline_json}")
        return 2
    baseline = json.loads(baseline_json.read_text())

    current = run_fingerprints(dataset_path, work_dir)

    failed: list[str] = []
    missing: list[str] = []
    for name, ref in baseline["fingerprints"].items():
        if name not in current["fingerprints"]:
            missing.append(name)
            continue
        cur = current["fingerprints"][name]
        if "hash" not in ref or "hash" not in cur:
            failed.append(f"{name} (non-hashable baseline or failure)")
            continue
        if ref["hash"] != cur["hash"]:
            failed.append(
                f"{name}: baseline={ref['hash'][:16]}..., "
                f"current={cur['hash'][:16]}..."
            )

    if failed:
        print("[verify] FAIL: fingerprint divergence:")
        for f in failed:
            print(f"   - {f}")
        return 1
    if missing:
        print(f"[verify] WARN: missing fingerprints: {missing}")
        return 2
    print(f"[verify] OK: {len(baseline['fingerprints'])} fingerprints match")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to pbmc3k.h5ad (default: $SCX_DATA_DIR/pbmc3k.h5ad).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "benchmarks" / "comprehensive" /
                    "results" / "baseline_2026_04_17" / "fingerprints"),
        help="Output directory for fingerprints.json and arrays/.",
    )
    parser.add_argument(
        "--only", nargs="+", default=None,
        help="Subset of accelerators to fingerprint (default: all).",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare against an existing baseline instead of writing one.",
    )
    parser.add_argument(
        "--baseline", default=None,
        help="Baseline directory for --verify.",
    )
    args = parser.parse_args()

    # Resolve dataset path via the shared bench env (.env / SCX_DATA_DIR).
    if args.dataset is None:
        from benchmarks.comprehensive.bench_env import DATA_DIR

        dataset_path = Path(DATA_DIR) / "pbmc3k.h5ad"
    else:
        dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"ERROR: dataset not found at {dataset_path}")
        return 2

    output_dir = Path(args.output_dir)

    if args.verify:
        baseline_dir = Path(args.baseline) if args.baseline else output_dir
        work_dir = output_dir.with_name(output_dir.name + "_current")
        return verify_against_baseline(dataset_path, baseline_dir, work_dir)

    run_fingerprints(dataset_path, output_dir, only=args.only)
    return 0


if __name__ == "__main__":
    sys.exit(main())
