#!/usr/bin/env python
"""Phase-4 task 4.5 — GPU DE device-residency profile (§9.11 + §9.13).

Before 4.5 the GPU CSR DE routes ran a full `for_each_gpu_csr_shard` pass per
gene chunk, so cost was `n_gene_chunks x n_shards` host decodes and H->D
uploads. At census_500k (61,497 genes / a 500-gene chunk = 123 chunks over 31
shards) that was 1,005.6 s of host decode against a 1,019.7 s wall. Both CSR
kernels *also* strided every row's whole nonzero range per chunk, so the
compute side was quadratic too.

4.5 retains the shards on the device and binary-searches each row down to the
`[c0, c1)` window. This captures what that did.

What to read, and what not to
-----------------------------

* **host-decode is a sum over concurrent workers**, so it can exceed wall.
  The ratio against wall is the signal, not the absolute.
* **the `htod` bucket is not the device copy.** `gpu_shard_source.rs` records
  it around `PinnedCsrSlot::stage()` — a *host* memcpy into the pinned buffer —
  and the asynchronous `upload_to` happens after the timer closes.
* **`resident_csr` is the only proof residency engaged.** A silent fall back to
  streaming produces identical numbers, just slowly, so the route stamp is
  recorded per run and a run that did not engage is called out.
* **VRAM is the cost side of this change and the reason this script exists**
  rather than reusing `profile_gpu_staging.py`: residency trades ~6 GB of
  device memory (census_500k, 747M nnz x 8 B) for the decode. The existing
  harness records no VRAM at all. Sampled here with `nvidia-smi` in a
  subprocess — no pynvml/cupy dependency, so it works in every bench env.

Run under `sbatch` on a GPU node, in the `scx-bench-gpu` conda env::

    SCX_GPU_PROFILE=1 python benchmarks/scripts/profile_gpu_de_resident.py

Env overrides: GPU_DE_DATASETS, GPU_DE_OPS (subset of `pdex_ref,wilcoxon,hvg`),
GPU_DE_RUNS, GPU_DE_OUT (JSON path).

`hvg` is in the default op set as a **control**: 4.5 touches neither the HVG
kernels nor its two-pass structure (that is §9.14, deferred to 4.4), so it
should not move. An hvg speedup would mean something other than the intended
change is being measured.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.comprehensive.bench_env import DATA_DIR  # noqa: E402

DATASETS = tuple(
    d.strip()
    for d in os.environ.get(
        "GPU_DE_DATASETS", "tabula_sapiens_100k,census_500k,census_1m"
    ).split(",")
    if d.strip()
)
OPS = tuple(
    o.strip()
    for o in os.environ.get("GPU_DE_OPS", "pdex_ref,wilcoxon,hvg").split(",")
    if o.strip()
)
N_RUNS = int(os.environ.get("GPU_DE_RUNS", 3))
OUT_PATH = os.environ.get("GPU_DE_OUT", "")


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class VramSampler:
    """Poll `nvidia-smi` for this process's device memory while an op runs.

    Deliberately dependency-free: the bench envs disagree about pynvml, and a
    missing import after hours of queue time is a worse failure than a coarse
    200 ms sample. Reports the peak *device-wide* used memory on the visible
    GPU, so a co-scheduled job would inflate it — the capture script asks for a
    whole GPU, and the baseline sample taken before the op runs makes a
    pre-existing occupant visible rather than silently folded into the delta.
    """

    def __init__(self, index: int = 0, interval_s: float = 0.2) -> None:
        self.index = index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.available = self._query() is not None

    def _query(self) -> int | None:
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self.index}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode != 0:
                return None
            return int(out.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            return None

    def _loop(self) -> None:
        while not self._stop.is_set():
            v = self._query()
            if v is not None:
                self.samples.append(v)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "VramSampler":
        self.samples = []
        if self.available:
            baseline = self._query()
            if baseline is not None:
                self.samples.append(baseline)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def baseline_mb(self) -> float:
        return float(self.samples[0]) if self.samples else 0.0

    @property
    def peak_mb(self) -> float:
        return float(max(self.samples)) if self.samples else 0.0


def _flat(snap: dict) -> dict[str, float]:
    out: dict[str, float] = {"gpu_profile_enabled": 1.0 if snap.get("enabled") else 0.0}
    for bucket in (
        "host_decode_scx1",
        "host_decode_generic",
        "htod_scx1",
        "htod_generic",
        "gpu_decode",
        "compute",
    ):
        st = snap.get(bucket) or {}
        out[f"{bucket}_ms"] = float(st.get("ms", 0.0))
        out[f"{bucket}_count"] = float(st.get("count", 0) or st.get("shards", 0))
        out[f"{bucket}_bytes"] = float(st.get("bytes", 0))
    return out


def _run_op(op: str, scx_path: Path) -> dict:
    """Run one op and return its recorded accelerator route info."""
    import numpy as np
    import pyscx

    adata = pyscx.open(str(scx_path)).to_anndata(backed=True)
    if op == "hvg":
        pyscx.accel.highly_variable_genes(
            adata, n_top_genes=2000, flavor="seurat_v3", device="gpu"
        )
        return dict(adata.uns.get("scx_accel", {}).get("highly_variable_genes", {}))

    adata.obs["grp"] = np.where(np.arange(adata.n_obs) % 2 == 0, "a", "b")
    if op == "wilcoxon":
        pyscx.accel.rank_genes_groups(adata, "grp", device="gpu")
        return dict(adata.uns.get("scx_accel", {}).get("rank_genes_groups", {}))
    if op == "pdex_ref":
        pyscx.accel.pdex_ref(adata, "grp", reference="a", device="gpu")
        return dict(adata.uns.get("scx_accel", {}).get("pdex_ref", {}))
    raise ValueError(op)


def main() -> int:
    import pyscx

    resident = os.environ.get("SCX_GPU_DE_RESIDENT", "<unset (on)>")
    print(f"=== GPU DE residency profile (SCX_GPU_DE_RESIDENT={resident}) ===")
    print(f"datasets: {', '.join(DATASETS)}")
    print(f"ops     : {', '.join(OPS)}   runs: {N_RUNS}")
    print()

    rows: list[dict] = []
    for dataset in DATASETS:
        scx_path = DATA_DIR / f"{dataset}_auto.scx"
        if not scx_path.exists():
            print(f"  SKIP {dataset}: {scx_path} not found", flush=True)
            continue
        for op in OPS:
            print(f"  ... {dataset}/{op}", flush=True)
            walls: list[float] = []
            snaps: list[dict] = []
            vram: list[tuple[float, float]] = []
            info: dict = {}
            for _ in range(N_RUNS):
                pyscx.accel.gpu_profile_reset()
                sampler = VramSampler()
                t0 = time.perf_counter()
                try:
                    with sampler:
                        info = _run_op(op, scx_path)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"  FAIL {dataset}/{op}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    walls = []
                    break
                walls.append(time.perf_counter() - t0)
                snaps.append(_flat(pyscx.accel.gpu_profile_snapshot()))
                vram.append((sampler.baseline_mb, sampler.peak_mb))
            if not walls:
                continue

            # Sort (wall, snapshot, vram) together — indexing a separately
            # sorted `walls` would pair the median wall with another run's
            # buckets, which is how the 4.2 harness first got this wrong.
            paired = sorted(zip(walls, snaps, vram), key=lambda t: t[0])
            median, snap, (vram_base, vram_peak) = paired[len(paired) // 2]
            hd = snap["host_decode_scx1_ms"] + snap["host_decode_generic_ms"]
            htod = snap["htod_scx1_ms"] + snap["htod_generic_ms"]
            row = {
                "dataset": dataset,
                "op": op,
                "resident_env": resident,
                "route": info.get("route"),
                "resident_csr": info.get("resident_csr"),
                "chunk_size": info.get("chunk_size"),
                "shards_decoded": info.get("shards_decoded"),
                "wall_ms": median * 1000.0,
                "host_decode_ms": hd,
                "host_decode_over_wall": hd / (median * 1000.0) if median else 0.0,
                "htod_ms": htod,
                "compute_ms": snap["compute_ms"],
                "peak_rss_mb": _peak_rss_mb(),
                "vram_baseline_mb": vram_base,
                "vram_peak_mb": vram_peak,
                "vram_delta_mb": vram_peak - vram_base,
            }
            rows.append(row)
            flag = ""
            if op != "hvg":
                if row["resident_csr"] is True:
                    flag = " [resident]"
                elif row["route"] and str(row["route"]).startswith("gpu_csr"):
                    flag = " [STREAMING — residency declined]"
                elif row["route"]:
                    flag = f" [{row['route']}]"
            print(
                f"  {dataset:<22} {op:<9} wall {row['wall_ms']:9.1f} ms | "
                f"host-decode {hd:9.1f} ms "
                f"({row['host_decode_over_wall'] * 100:5.1f}% of wall) | "
                f"HTOD {htod:8.1f} ms | compute {snap['compute_ms']:8.1f} ms | "
                f"RSS {row['peak_rss_mb']:7.0f} MB | "
                f"VRAM +{row['vram_delta_mb']:7.0f} MB{flag}",
                flush=True,
            )

    print()
    print(json.dumps(rows, indent=2))
    if OUT_PATH:
        Path(OUT_PATH).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {OUT_PATH}")

    if not rows:
        print("NOTE: nothing captured — no dataset produced a usable run")
        return 1
    if not any(r["host_decode_ms"] for r in rows):
        print(
            "ERROR: every host-decode bucket is zero — the profiler was not "
            "enabled. Re-run with SCX_GPU_PROFILE=1; the table above measures "
            "nothing."
        )
        return 1
    # A DE row that took the CSR route but did not go resident is either an
    # environment leak or a VRAM refusal. Either way the arm is not measuring
    # what the run was for, so say so loudly rather than let a modest number
    # be read as "residency did not help".
    if os.environ.get("SCX_GPU_DE_RESIDENT") != "0":
        declined = [
            r
            for r in rows
            if r["op"] != "hvg"
            and str(r.get("route") or "").startswith("gpu_csr")
            and r.get("resident_csr") is not True
        ]
        if declined:
            print(
                "\nWARNING: residency was expected but declined on: "
                + ", ".join(f"{r['dataset']}/{r['op']}" for r in declined)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
