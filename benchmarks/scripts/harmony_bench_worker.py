#!/usr/bin/env python3
"""Subprocess worker for Harmony benchmark timing + RSS measurement.

Runs one (impl, dataset, device) configuration in a fresh process so
ru_maxrss reflects only this workload. Samples RSS every 100 ms via a
background thread. Prints a single JSON object to stdout.

Not intended to be invoked directly — see benchmark_harmony.py.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import threading
import time
from pathlib import Path

import numpy as np


# ───────────────────────── RSS sampling ─────────────────────────

def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * resource.getpagesize() / (1024 * 1024)
    except OSError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class RssSampler:
    """Background thread sampling RSS at a fixed interval."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._t0: float | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._t0 = time.perf_counter()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        assert self._t0 is not None
        while not self._stop.is_set():
            elapsed = time.perf_counter() - self._t0
            self.samples.append(
                {"t": round(elapsed, 3), "rss_mb": round(_rss_mb(), 1)}
            )
            time.sleep(self.interval_ms / 1000)

    def downsampled(self, n: int = 200) -> list[dict]:
        """Return at most `n` evenly-spaced samples for the JSON."""
        if len(self.samples) <= n:
            return self.samples
        stride = len(self.samples) / n
        return [self.samples[int(i * stride)] for i in range(n)]


# ───────────────────── impl runners ─────────────────────

def _load_inputs(args) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pca = np.load(args.pca).astype(np.float32, copy=False)
    batch = np.load(args.batch)
    # Convert integer labels back to strings (R harmony + harmonypy expect
    # character/factor). If labels are already strings (allow_pickle=True),
    # respect them.
    if batch.dtype.kind in "iu":
        batch_str = [f"b{v}" for v in batch.tolist()]
    else:
        batch_str = [str(v) for v in batch.tolist()]
    return pca, batch, batch_str


def run_scx_accel_cpu(args, pca, batch, batch_str):
    import pyscx
    import anndata as ad
    import pandas as pd

    N, d = pca.shape
    adata = ad.AnnData(
        X=np.zeros((N, 1), dtype=np.float32),
        obs=pd.DataFrame({"batch": pd.Categorical(batch_str)}),
    )
    adata.obsm["X_pca"] = pca
    pyscx.accel.harmony_integrate(
        adata,
        "batch",
        n_clusters=args.n_clusters,
        theta=args.theta,
        max_iter=args.max_iter,
        random_state=args.seed,
        device="cpu",
        adjusted_basis="X_pca_harmony",
    )
    meta = adata.uns["harmony"]
    return {
        "z_corrected": np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float32),
        "n_iterations": int(meta["n_iterations"]),
        "converged": bool(meta["converged"]),
        "objective": list(meta["objective_harmony"]),
    }


def run_scx_accel_gpu(args, pca, batch, batch_str):
    import pyscx
    import anndata as ad
    import pandas as pd

    N, d = pca.shape
    adata = ad.AnnData(
        X=np.zeros((N, 1), dtype=np.float32),
        obs=pd.DataFrame({"batch": pd.Categorical(batch_str)}),
    )
    adata.obsm["X_pca"] = pca
    pyscx.accel.harmony_integrate(
        adata,
        "batch",
        n_clusters=args.n_clusters,
        theta=args.theta,
        max_iter=args.max_iter,
        random_state=args.seed,
        device="gpu",
        adjusted_basis="X_pca_harmony",
    )
    meta = adata.uns["harmony"]
    return {
        "z_corrected": np.asarray(adata.obsm["X_pca_harmony"], dtype=np.float32),
        "n_iterations": int(meta["n_iterations"]),
        "converged": bool(meta["converged"]),
        "objective": list(meta["objective_harmony"]),
    }


def run_harmonypy(args, pca, batch, batch_str):
    import harmonypy
    import pandas as pd

    meta = pd.DataFrame({"batch": batch_str})
    ho = harmonypy.run_harmony(
        pca.astype(np.float64),
        meta,
        vars_use=["batch"],
        theta=args.theta,
        nclust=args.n_clusters,
        max_iter_harmony=args.max_iter,
        random_state=args.seed,
    )
    # harmonypy's Z_corr is (d x N) — transpose back to (N x d).
    z_corr = np.asarray(ho.Z_corr.T, dtype=np.float32)
    return {
        "z_corrected": z_corr,
        "n_iterations": int(getattr(ho, "objective_harmony", [0])
                            .__len__()),
        "converged": bool(getattr(ho, "converged", False)),
        "objective": list(getattr(ho, "objective_harmony", [])),
    }


def run_r_harmony(args, pca, batch, batch_str, tmp_dir: Path):
    """Invoke R harmony via the rscx conda env as a child subprocess."""
    import subprocess
    pca_path = tmp_dir / "pca.npy"
    batch_path = tmp_dir / "batch.npy"
    out_prefix = tmp_dir / "out"
    np.save(pca_path, pca.astype(np.float32))
    np.save(batch_path, np.asarray(batch))  # int or str dtype preserved
    r_script = (
        Path(__file__).resolve().parent / "generate_harmony_reference.R"
    )
    rscript = "/home/nickyoungblut/miniforge3/envs/rscx/bin/Rscript"
    cmd = [
        rscript,
        str(r_script),
        "--pca", str(pca_path),
        "--batch", str(batch_path),
        "--out-prefix", str(out_prefix),
        "--theta", str(args.theta),
        "--nclust", str(args.n_clusters),
        "--max-iter", str(args.max_iter),
        "--seed", str(args.seed),
    ]
    subprocess.check_call(cmd, stderr=subprocess.STDOUT)
    ref = np.load(f"{out_prefix}.npz")
    return {
        "z_corrected": ref["Z_corr"].astype(np.float32),
        "n_iterations": int(ref["n_iterations"]),
        "converged": True,  # R returns on completion; no convergence flag
        "objective": ref["objective"].tolist(),
    }


IMPL_RUNNERS = {
    "scx_accel_cpu": run_scx_accel_cpu,
    "scx_accel_gpu": run_scx_accel_gpu,
    "harmonypy": run_harmonypy,
    "r_harmony": run_r_harmony,
}


# ───────────────────── main ─────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True, choices=sorted(IMPL_RUNNERS))
    ap.add_argument("--pca", required=True, help="Path to N x d .npy embedding")
    ap.add_argument("--batch", required=True, help="Path to N-vector .npy labels")
    ap.add_argument("--reference", help="Optional .npz from R harmony for per-PC correlation")
    ap.add_argument("--n-clusters", type=int, default=100)
    ap.add_argument("--theta", type=float, default=2.0)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rss-interval-ms", type=int, default=100)
    ap.add_argument("--tmp-dir", default="/tmp")
    args = ap.parse_args()

    gc.collect()
    sampler = RssSampler(interval_ms=args.rss_interval_ms)

    pca, batch, batch_str = _load_inputs(args)
    runner = IMPL_RUNNERS[args.impl]

    # Snapshot RSS *after* PCA + batch labels are loaded but *before* the
    # harmony runner begins. This gives a clean baseline so the eventual
    # `peak_rss_mb` can be split into the input-load floor (PCA cache,
    # interpreter, scanpy/anndata imports) and the harmony-only delta.
    # `docs/performance/accel-qc-de-integration.md` quotes `harmony_delta_rss_mb` rather than the
    # raw peak — the prior driver measured peak only, which conflated
    # PCA-load overhead (~N·d·4 bytes) with the algorithm's real footprint.
    #
    # Use *current* RSS (`_rss_mb`, /proc/self/statm) here, not
    # `_peak_rss_mb` (`ru_maxrss`). `ru_maxrss` is a process-lifetime
    # high-water mark, so if h5ad/scanpy import or PCA load briefly
    # peaked above harmony's working set, `peak_rss - baseline_peak`
    # would clip to ~0 and silently under-report the real algorithm
    # footprint.
    gc.collect()
    baseline_rss = _rss_mb()

    # Run + time (wall-clock + RSS sweep).
    err_msg: str | None = None
    out: dict | None = None
    sampler.start()
    t0 = time.perf_counter()
    try:
        if args.impl == "r_harmony":
            tmp = Path(args.tmp_dir) / f"harmony_bench_{int(t0)}"
            tmp.mkdir(parents=True, exist_ok=True)
            out = runner(args, pca, batch, batch_str, tmp)
        else:
            out = runner(args, pca, batch, batch_str)
        wall_s = time.perf_counter() - t0
    except MemoryError as e:
        wall_s = time.perf_counter() - t0
        err_msg = f"MemoryError: {e}"
    except Exception as e:  # noqa: BLE001
        wall_s = time.perf_counter() - t0
        err_msg = f"{type(e).__name__}: {e}"
    finally:
        sampler.stop()

    peak_rss = _peak_rss_mb()
    harmony_delta_rss = max(0.0, peak_rss - baseline_rss)

    result: dict = {
        "impl": args.impl,
        "n_obs": int(pca.shape[0]),
        "n_pcs": int(pca.shape[1]),
        "n_clusters": args.n_clusters,
        "theta": args.theta,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "wall_s": round(wall_s, 3),
        "peak_rss_mb": round(peak_rss, 1),
        "baseline_rss_mb": round(baseline_rss, 1),
        "harmony_delta_rss_mb": round(harmony_delta_rss, 1),
        "rss_timeseries": sampler.downsampled(),
        "ok": err_msg is None,
    }

    if err_msg:
        result["error"] = err_msg
    if out is not None:
        result["n_iterations"] = out["n_iterations"]
        result["converged"] = out["converged"]
        result["objective"] = out["objective"]
        # Per-PC Pearson correlation against the R reference if provided.
        if args.reference and Path(args.reference).exists():
            ref = np.load(args.reference)
            ref_z = ref["Z_corr"].astype(np.float32)
            got_z = out["z_corrected"]
            if ref_z.shape == got_z.shape:
                per_pc = []
                for pc in range(ref_z.shape[1]):
                    a = ref_z[:, pc].astype(np.float64)
                    b = got_z[:, pc].astype(np.float64)
                    if a.std() < 1e-12 or b.std() < 1e-12:
                        per_pc.append(None)
                        continue
                    per_pc.append(float(np.corrcoef(a, b)[0, 1]))
                result["per_pc_pearson_r"] = per_pc
            else:
                result["per_pc_pearson_r_note"] = (
                    f"shape mismatch: ref {ref_z.shape} vs run {got_z.shape}"
                )

    print(json.dumps(result))
    return 0 if err_msg is None else 1


if __name__ == "__main__":
    sys.exit(main())
