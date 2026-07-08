#!/usr/bin/env python3
"""
GPU codec-decode profiling driver — GPU ShufDeltaZstd decode, Phase 0 (tasks 0a/0c).

Profiles ``pyscx.open(path).to_gpu_anndata(device="gpu")`` across three codec
encodings of the *same* dataset:

  * ``scx1``          — in-VRAM device decode (BitPacker4x FOR-BP + Rice).
  * ``compact_trial`` — per-shard smaller of {heuristic winner, ShufDeltaZstd};
                        ShufDeltaZstd shards currently **host-bounce**.
  * ``shufdelta``     — uniform ShufDeltaZstd; host-bounces today.

The point is the **exit criterion** from the spec (§6): if the host-bounced
codec's ``to_gpu_anndata`` throughput is >= 90% of the Scx1 device-decode
throughput, a GPU ShufDeltaZstd kernel (Phase 1) is not justified; if < 90%,
proceed. This script measures that ratio on real fixtures and prints an
explicit go / no-go.

It also confirms the decode route per codec via
``adata.uns["scx_accel"]["to_gpu_anndata"]["transfer_mode"]``
(``scx_device_decode_gpu`` for Scx1 vs ``scx_device_handoff_streamed`` for
host-bounce) and, best-effort, does a byte-exact parity check of the device
CSR against a single host reference decode per dataset.

Fixtures are the pre-built ``{dataset}_{codec}.scx`` files under
``$SCX_DATA_DIR`` (defaults to ``$SCX_WORK_DIR/benchmarks/datasets``); no
conversion is done here. Must run on a GPU node (see the sibling
``slurm_bench_gpu_codec.sh`` wrapper).

Usage:
    python benchmarks/scripts/bench_gpu_codec.py \
        --datasets census_1m census_500k --n-runs 3 \
        --out-dir benchmarks/results/gpu_codec
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.comprehensive.bench_env import DATA_DIR  # noqa: E402

# codec-variant key -> on-disk filename suffix. Scx1 is the device-decode
# baseline the host-bounced codecs are measured against.
CODECS: dict[str, str] = {
    "scx1": "scx1",
    "compact_trial": "compact_trial",
    "shufdelta": "shufdelta",
}
BASELINE_CODEC = "scx1"

# 90% exit criterion (spec §6): host-bounce >= this fraction of Scx1 GPU
# throughput => Phase 1 not justified.
EXIT_RATIO = 0.90


def _fixture_path(dataset: str, codec_suffix: str) -> Path:
    return DATA_DIR / f"{dataset}_{codec_suffix}.scx"


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def _profile_snapshot(pyscx) -> dict | None:
    """Best-effort GPU/host per-stage profile snapshot (SCX_GPU_PROFILE=1).

    For host-bounced codecs the whole CPU decode lands in ``host_decode_generic``
    — useful supporting data for the 0c write-up. Returns None if unavailable.
    """
    try:
        return dict(pyscx.accel.gpu_profile_snapshot())
    except Exception:
        return None


def _profile_reset(pyscx) -> None:
    try:
        pyscx.accel.gpu_profile_reset()
    except Exception:
        pass


def _host_reference(pyscx, path: Path):
    """Decode the host-side scipy CSR once (sorted) for a parity baseline.

    Wrapped by the caller in try/except so an OOM on the largest tiers degrades
    to 'parity skipped' rather than aborting the throughput measurement.
    """
    x = pyscx.open(str(path)).to_anndata().X.tocsr()
    x.sort_indices()
    return x


def _parity(adata_gpu, host_x) -> bool:
    import numpy as np

    gpu_x = adata_gpu.X.get()  # cupyx CSR -> scipy CSR
    gpu_x.sort_indices()
    return (
        tuple(gpu_x.shape) == tuple(host_x.shape)
        and np.array_equal(gpu_x.indptr, host_x.indptr)
        and np.array_equal(gpu_x.indices, host_x.indices)
        and np.array_equal(gpu_x.data, host_x.data)
    )


def profile_codec(
    pyscx,
    dataset: str,
    codec: str,
    codec_suffix: str,
    n_runs: int,
    n_warmup: int,
    host_x,
    decode_variant: str = "pipeline",
) -> dict | None:
    """Profile ``to_gpu_anndata`` for one (dataset, codec). Returns a record.

    ``decode_variant`` selects the shufdelta GPU decode path via env vars read
    per-call by the Rust dispatch (so they can be flipped in-process):
      * ``"pipeline"``        — DEFAULT: Phase-1.5 parallel-zstd + multi-stream.
      * ``"nvcomp"``          — `SCX_SHUFDELTA_NVCOMP=1` → Phase-2/2.x full in-VRAM
                                (nvcomp GPU zstd; `scx_device_decode_gpu`, compressed
                                upload). For a uniform-shufdelta file this takes the
                                Phase-2.x **cross-shard batched** path (2 nvcomp calls
                                total).
      * ``"nvcomp_pershard"`` — `SCX_SHUFDELTA_NVCOMP=1` + `SCX_NVCOMP_NO_BATCH=1` →
                                the Phase-2 **per-shard** nvcomp loop (2 nvcomp calls
                                per shard). The A/B baseline for the batched path.
      * ``"sequential"``      — `SCX_SHUFDELTA_GPU_SEQUENTIAL=1` → Phase-1 sequential.
    Only meaningful for shufdelta / compact_trial codecs.
    """
    os.environ.pop("SCX_SHUFDELTA_GPU_SEQUENTIAL", None)
    os.environ.pop("SCX_SHUFDELTA_NVCOMP", None)
    os.environ.pop("SCX_NVCOMP_NO_BATCH", None)
    if decode_variant == "sequential":
        os.environ["SCX_SHUFDELTA_GPU_SEQUENTIAL"] = "1"
    elif decode_variant == "nvcomp":
        os.environ["SCX_SHUFDELTA_NVCOMP"] = "1"
    elif decode_variant == "nvcomp_pershard":
        os.environ["SCX_SHUFDELTA_NVCOMP"] = "1"
        os.environ["SCX_NVCOMP_NO_BATCH"] = "1"

    path = _fixture_path(dataset, codec_suffix)
    if not path.exists():
        print(f"  [skip] {dataset}/{codec}: fixture missing at {path}", file=sys.stderr)
        return None

    # Warm-up (PTX load, allocator warm).
    try:
        for _ in range(n_warmup):
            warm = pyscx.open(str(path)).to_gpu_anndata(device="gpu")
            del warm
            gc.collect()
    except Exception as exc:  # noqa: BLE001 — cupy/cuda errors are opaque
        print(f"  [skip] {dataset}/{codec}: warm-up failed ({exc})", file=sys.stderr)
        return {"dataset": dataset, "codec": codec, "error": f"warmup: {exc}"}

    walls: list[float] = []
    transfer_mode = None
    bytes_uploaded = 0
    n_shards_shufdelta_gpu = None
    nnz = 0
    n_obs = 0
    n_vars = 0
    parity_ok: bool | None = None
    profile = None

    for i in range(n_runs):
        gc.collect()
        _profile_reset(pyscx)
        t0 = time.perf_counter()
        adata_gpu = pyscx.open(str(path)).to_gpu_anndata(device="gpu")
        wall = time.perf_counter() - t0
        walls.append(wall)

        info = adata_gpu.uns["scx_accel"]["to_gpu_anndata"]
        transfer_mode = info.get("transfer_mode")
        bytes_uploaded = int(info.get("bytes_uploaded") or 0)
        # Phase-1 per-codec GPU routing: how many shards took the ShufDeltaZstd
        # GPU decode path (vs host-bounce). None on builds without the counter.
        n_shards_shufdelta_gpu = info.get("n_shards_shufdelta_gpu")
        nnz = int(adata_gpu.X.nnz)
        n_obs, n_vars = (int(adata_gpu.shape[0]), int(adata_gpu.shape[1]))
        if profile is None:
            profile = _profile_snapshot(pyscx)

        if host_x is not None and parity_ok is None:
            try:
                parity_ok = _parity(adata_gpu, host_x)
            except Exception as exc:  # noqa: BLE001
                print(f"    parity check errored: {exc}", file=sys.stderr)
                parity_ok = None

        print(
            f"  {dataset}/{codec}[{decode_variant}] run {i + 1}/{n_runs}: "
            f"wall={wall:.3f}s transfer_mode={transfer_mode} "
            f"bytes_uploaded={bytes_uploaded} n_shards_shufdelta_gpu={n_shards_shufdelta_gpu} "
            f"nnz={nnz} parity={parity_ok}"
        )
        del adata_gpu
        gc.collect()

    median_wall = _median(walls)
    throughput_nnz_s = nnz / median_wall if median_wall > 0 else float("nan")
    return {
        "dataset": dataset,
        "codec": codec,
        "decode_variant": decode_variant,
        "fixture": str(path),
        "file_size_bytes": path.stat().st_size,
        "n_obs": n_obs,
        "n_vars": n_vars,
        "nnz": nnz,
        "n_runs": n_runs,
        "walls_s": walls,
        "median_wall_s": median_wall,
        "throughput_nnz_per_s": throughput_nnz_s,
        "transfer_mode": transfer_mode,
        "bytes_uploaded": bytes_uploaded,
        "n_shards_shufdelta_gpu": n_shards_shufdelta_gpu,
        "parity_vs_host": parity_ok,
        "gpu_profile": profile,
    }


def analyze_dataset(records: list[dict]) -> dict:
    """Compute the host-bounce vs Scx1 throughput ratio + go/no-go verdict.

    Uses the pipelined (default) variant for the ratio; the sequential A/B
    records are kept for reporting but do not drive the verdict.
    """
    by_codec = {
        r["codec"]: r
        for r in records
        if r
        and "throughput_nnz_per_s" in r
        and r.get("decode_variant", "pipeline") == "pipeline"
    }
    base = by_codec.get(BASELINE_CODEC)
    out: dict = {"ratios": {}, "baseline_throughput_nnz_per_s": None, "go_phase1": None}
    if not base:
        out["note"] = f"no {BASELINE_CODEC} baseline — cannot compute ratio"
        return out
    base_tput = base["throughput_nnz_per_s"]
    out["baseline_throughput_nnz_per_s"] = base_tput
    go = False
    for codec, r in by_codec.items():
        if codec == BASELINE_CODEC:
            continue
        ratio = r["throughput_nnz_per_s"] / base_tput if base_tput else float("nan")
        out["ratios"][codec] = ratio
        # go if ANY host-bounced codec is below the exit ratio.
        if ratio < EXIT_RATIO:
            go = True
    out["go_phase1"] = go
    out["exit_ratio"] = EXIT_RATIO
    return out


def render_markdown(all_records: dict[str, list[dict]], analysis: dict[str, dict]) -> str:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# GPU codec-decode profiling (GPU ShufDeltaZstd decode, Phase 0)",
        "",
        f"**Generated**: {ts}",
        "",
        "`to_gpu_anndata(device=\"gpu\")` wall time and throughput per codec. "
        "Scx1 decodes in VRAM; compact_trial / shufdelta host-bounce today.",
        "",
    ]
    for dataset, records in all_records.items():
        lines += [
            f"## {dataset}",
            "",
            "| Codec | variant | file (GB) | nnz | median wall (s) | throughput (Mnnz/s) | transfer_mode | bytes_uploaded (MB) | shufdelta_gpu shards | parity |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in records:
            if not r or "throughput_nnz_per_s" not in r:
                err = (r or {}).get("error", "missing")
                lines.append(
                    f"| {(r or {}).get('codec', '?')} | {(r or {}).get('decode_variant', '?')} "
                    f"| — | — | — | — | {err} | — | — | — |"
                )
                continue
            lines.append(
                f"| {r['codec']} "
                f"| {r.get('decode_variant', 'pipeline')} "
                f"| {r['file_size_bytes'] / 1e9:.2f} "
                f"| {r['nnz']:,} "
                f"| {r['median_wall_s']:.3f} "
                f"| {r['throughput_nnz_per_s'] / 1e6:.1f} "
                f"| {r['transfer_mode']} "
                f"| {r['bytes_uploaded'] / 1e6:.1f} "
                f"| {r.get('n_shards_shufdelta_gpu')} "
                f"| {r['parity_vs_host']} |"
            )
        a = analysis.get(dataset, {})
        lines += ["", "**Exit-criterion ratios (host-bounce ÷ Scx1 GPU throughput):**", ""]
        for codec, ratio in a.get("ratios", {}).items():
            verdict = "< 0.90 → favors Phase 1" if ratio < EXIT_RATIO else ">= 0.90 → no-go"
            lines.append(f"- `{codec}`: {ratio:.3f}  ({verdict})")
        go = a.get("go_phase1")
        lines += [
            "",
            f"**{dataset} verdict:** "
            + ("**GO** (host-bounce < 90% of Scx1 GPU)" if go else "**NO-GO** (host-bounce >= 90%)")
            if go is not None
            else f"**{dataset} verdict:** inconclusive ({a.get('note', '')})",
            "",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["census_1m", "census_500k"])
    parser.add_argument("--codecs", nargs="+", default=list(CODECS.keys()))
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--n-warmup", type=int, default=1)
    parser.add_argument(
        "--no-parity",
        action="store_true",
        help="skip the host reference decode + byte-exact parity check",
    )
    parser.add_argument(
        "--compare-sequential",
        action="store_true",
        help="also run shufdelta/compact_trial through the Phase-1 sequential "
        "decode (SCX_SHUFDELTA_GPU_SEQUENTIAL=1) for an A/B vs the pipeline",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "gpu_codec",
    )
    args = parser.parse_args()

    # Enable the per-stage GPU/host profiler for supporting 0c data.
    os.environ.setdefault("SCX_GPU_PROFILE", "1")

    try:
        import pyscx
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: pyscx import failed: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        import cupy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cupy import failed (need a GPU node): {exc}", file=sys.stderr)
        sys.exit(1)

    all_records: dict[str, list[dict]] = {}
    analysis: dict[str, dict] = {}

    for dataset in args.datasets:
        print(f"\n=== {dataset} ===")
        # One host reference decode per dataset (all codecs encode identical
        # data); parity of every codec's GPU decode is checked against it.
        host_x = None
        if not args.no_parity:
            ref_path = _fixture_path(dataset, CODECS[BASELINE_CODEC])
            if ref_path.exists():
                try:
                    print(f"  host reference decode from {ref_path.name} ...")
                    host_x = _host_reference(pyscx, ref_path)
                except Exception as exc:  # noqa: BLE001 — MemoryError included
                    print(
                        f"  host reference decode skipped ({type(exc).__name__}: {exc}); "
                        "parity will be reported as None",
                        file=sys.stderr,
                    )
                    host_x = None

        records: list[dict] = []
        for codec in args.codecs:
            suffix = CODECS.get(codec, codec)
            # DEFAULT path (Phase-1.5 pipeline) — drives the go/no-go ratios.
            rec = profile_codec(
                pyscx, dataset, codec, suffix, args.n_runs, args.n_warmup, host_x
            )
            if rec is not None:
                records.append(rec)
            # Optional A/B for host-bounced codecs: Phase-2.x batched nvcomp
            # (device-decode + compressed upload), the Phase-2 per-shard nvcomp
            # baseline (the key Phase-2.x A/B), and Phase-1 sequential.
            if args.compare_sequential and codec != BASELINE_CODEC:
                for variant in ("nvcomp", "nvcomp_pershard", "sequential"):
                    alt = profile_codec(
                        pyscx,
                        dataset,
                        codec,
                        suffix,
                        args.n_runs,
                        args.n_warmup,
                        host_x,
                        decode_variant=variant,
                    )
                    if alt is not None:
                        records.append(alt)
        # Leave the env clean for the next dataset.
        os.environ.pop("SCX_SHUFDELTA_GPU_SEQUENTIAL", None)
        os.environ.pop("SCX_SHUFDELTA_NVCOMP", None)
        os.environ.pop("SCX_NVCOMP_NO_BATCH", None)
        del host_x
        gc.collect()

        all_records[dataset] = records
        analysis[dataset] = analyze_dataset(records)
        a = analysis[dataset]
        print(
            f"  {dataset} ratios={ {k: round(v, 3) for k, v in a.get('ratios', {}).items()} } "
            f"go_phase1={a.get('go_phase1')}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"gpu_codec_{stamp}.json"
    md_path = args.out_dir / f"gpu_codec_{stamp}.md"
    json_path.write_text(
        json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "exit_ratio": EXIT_RATIO,
                "records": all_records,
                "analysis": analysis,
            },
            indent=2,
            default=str,
        )
    )
    md_path.write_text(render_markdown(all_records, analysis))
    print(f"\nJSON:   {json_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
