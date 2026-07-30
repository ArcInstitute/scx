#!/bin/bash
# Phase-4 task 4.5 — resident-vs-streaming output equivalence at ATLAS SCALE.
#
# Everything else that checks 4.5's correctness runs on small fixtures:
# `test_gpu_de_resident.py` compares the two arms at 240 x 40, the kernel
# window-parity test at 5 x 12, and the CPU-vs-GPU parity suites at test scale.
# None of them reach the regime the change was built for — 61,497 genes over 123
# gene chunks and 31 shards, where residency retains ~6 GB in VRAM and the
# windowed kernels binary-search 500,000 rows per chunk.
#
# That gap matters because the plausible at-scale failure modes are exactly the
# ones a small fixture cannot express: index arithmetic that only overflows past
# 2^31, a `global_row` offset that only drifts across many shards, a VRAM
# pre-flight that only trips on a real matrix. The perf capture ran DE to
# completion at this scale but never compared its output to anything.
#
# So: same file, same groups, `SCX_GPU_DE_RESIDENT=0` vs default, compare the
# arrays. Statistics and p-values must be **exact** — they come from the
# single-writer gene-major scatter, which the windowing leaves bit-identical.
# Pseudobulk-derived means and fold changes are compared to tolerance, and the
# tolerance is *measured* rather than asserted: the streaming path is run twice
# and its own run-to-run spread (f64 atomicAdd ordering across rows is already
# nondeterministic) is the bar residency has to come in under. An absolute
# tolerance picked by hand would prove nothing.
#
# **Run alone** — see `_run_4_5_gpu_verify.sh`'s header on the shared `.so`.
#SBATCH --job-name=scx-4.5-scale-equiv
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=/home/nickyoungblut/scx-bench-4.5/scale_equiv_%j.out
#SBATCH --error=/home/nickyoungblut/scx-bench-4.5/scale_equiv_%j.out

set -uo pipefail
SCX_DIR=/home/nickyoungblut/dev/rust/scx
CONDA=/home/nickyoungblut/miniforge3
ENV="${CONDA}/envs/scx-bench-gpu"
OUT="/home/nickyoungblut/scx-bench-4.5/scale_equiv_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
unset VIRTUAL_ENV
# shellcheck disable=SC1091
source "${CONDA}/etc/profile.d/conda.sh"
conda activate scx-bench-gpu
export SCX_DISABLE_CUDA_GRAPHS=1
export NUMBA_NUM_THREADS=8

TARGET=/home/nickyoungblut/.cargo-target-45-scaleeq
rm -rf "${TARGET}"
( cd "${SCX_DIR}/pyscx" && VIRTUAL_ENV="${ENV}" CARGO_TARGET_DIR="${TARGET}" \
    "${ENV}/bin/maturin" develop --release --features hdf5,gpu ) 2>&1 | tail -2
cd "${SCX_DIR}" || exit 1
set -a; . ./.env; set +a

DATASET="${SCALE_EQUIV_DATASET:-census_500k}"

# Each arm runs in its own process: SCX_GPU_DE_RESIDENT is read once per
# process via OnceLock, so one interpreter cannot hold both.
ARM=$(cat <<'PY'
import json, os, sys
import numpy as np
import pyscx

path, op, dest = sys.argv[1], sys.argv[2], sys.argv[3]
adata = pyscx.open(path).to_anndata(backed=True)
adata.obs["grp"] = np.where(np.arange(adata.n_obs) % 2 == 0, "a", "b")

if op == "wilcoxon":
    pyscx.accel.rank_genes_groups(adata, "grp", device="gpu")
    info = adata.uns["scx_accel"]["rank_genes_groups"]
    r = adata.uns["rank_genes_groups"]
    payload = {
        "names":  [list(map(str, r["names"][f]))  for f in r["names"].dtype.names],
        "pvals":  [list(map(float, r["pvals"][f]))  for f in r["pvals"].dtype.names],
        "scores": [list(map(float, r["scores"][f])) for f in r["scores"].dtype.names],
        "lfc":    [list(map(float, r["logfoldchanges"][f]))
                   for f in r["logfoldchanges"].dtype.names],
    }
else:
    df = pyscx.accel.pdex_ref(adata, "grp", reference="a", device="gpu")
    info = adata.uns["scx_accel"]["pdex_ref"]
    payload = {
        "feature":     [str(v)   for v in df["feature"].to_list()],
        "p_value":     [float(v) for v in df["p_value"].to_list()],
        "statistic":   [float(v) for v in df["statistic"].to_list()],
        "target_mean": [float(v) for v in df["target_mean"].to_list()],
        "lfc":         [float(v) for v in df["log2_fold_change"].to_list()],
    }

payload["_route"] = info["route"]
payload["_resident"] = info["resident_csr"]
payload["_chunk"] = info["chunk_size"]
with open(dest, "w") as fh:
    json.dump(payload, fh)
print(f"{op}: route={info['route']} resident={info['resident_csr']} "
      f"chunk={info['chunk_size']}", flush=True)
PY
)

SCX="${SCX_DATA_DIR:-/large_storage/arcinfra/projects/scx/benchmarks/datasets}/${DATASET}_auto.scx"
echo "fixture: ${SCX}"

for op in wilcoxon pdex_ref; do
    echo ""
    echo "########## ${op} ##########"
    # Two streaming runs, so the comparison has a measured bar rather than a
    # hand-picked tolerance.
    SCX_GPU_DE_RESIDENT=0 python -c "${ARM}" "${SCX}" "${op}" "${OUT}/${op}_stream_a.json"
    SCX_GPU_DE_RESIDENT=0 python -c "${ARM}" "${SCX}" "${op}" "${OUT}/${op}_stream_b.json"
    python -c "${ARM}" "${SCX}" "${op}" "${OUT}/${op}_resident.json"
done

echo ""
echo "########## COMPARISON ##########"
python - "${OUT}" <<'PY'
import json, sys
from pathlib import Path
import numpy as np

out = Path(sys.argv[1])
failures = []

for op in ("wilcoxon", "pdex_ref"):
    a = json.loads((out / f"{op}_stream_a.json").read_text())
    b = json.loads((out / f"{op}_stream_b.json").read_text())
    r = json.loads((out / f"{op}_resident.json").read_text())

    print(f"\n=== {op}")
    print(f"  streaming route={a['_route']} resident={a['_resident']} chunk={a['_chunk']}")
    print(f"  resident  route={r['_route']} resident={r['_resident']} chunk={r['_chunk']}")

    # Premises. Without these the comparison proves nothing.
    if not str(a["_route"]).startswith("gpu_csr"):
        failures.append(f"{op}: streaming arm took {a['_route']}, not a CSR route")
    if r["_resident"] is not True:
        failures.append(f"{op}: residency did not engage on the resident arm")
    if a["_resident"] is not False:
        failures.append(f"{op}: SCX_GPU_DE_RESIDENT=0 did not disable residency")
    n_chunks = -(-61497 // int(r["_chunk"] or 1))
    print(f"  gene chunks: ~{n_chunks}")
    if n_chunks <= 1:
        failures.append(f"{op}: only one gene chunk — residency is a no-op here")

    exact_keys = {"names", "feature", "pvals", "scores", "p_value", "statistic"}
    for key in a:
        if key.startswith("_"):
            continue
        va, vb, vr = np.asarray(a[key]), np.asarray(b[key]), np.asarray(r[key])
        if va.dtype.kind in "UO":
            same = np.array_equal(va, vr)
            print(f"  {key:<12} exact string match: {same}")
            if not same:
                failures.append(f"{op}.{key}: string arrays differ")
            continue
        # nan-safe max abs difference
        def maxdiff(x, y):
            m = ~(np.isnan(x) & np.isnan(y))
            if not m.any():
                return 0.0
            d = np.abs(x[m] - y[m])
            return float(np.nanmax(d)) if d.size else 0.0
        stream_spread = maxdiff(va, vb)
        resident_delta = maxdiff(va, vr)
        if key in exact_keys:
            ok = resident_delta == 0.0
            note = "EXACT required"
        else:
            # The bar is the streaming path's own run-to-run spread, with a
            # floor so a coincidentally-identical pair of streaming runs does
            # not demand bit-identity from a differently-ordered reduction.
            bar = max(stream_spread, 1e-9 * float(np.nanmax(np.abs(va)) or 1.0))
            ok = resident_delta <= bar
            note = f"bar={bar:.3e} (streaming self-spread {stream_spread:.3e})"
        print(f"  {key:<12} resident-vs-streaming max|d|={resident_delta:.3e}  "
              f"{note}  {'OK' if ok else 'FAIL'}")
        if not ok:
            failures.append(
                f"{op}.{key}: |d|={resident_delta:.3e} exceeds {note}"
            )

print()
if failures:
    print("########## FAILURES ##########")
    for f in failures:
        print(" -", f)
    raise SystemExit(1)
print("all checks passed: resident output matches streaming at atlas scale")
PY

echo ""
echo "=== done; artifacts under ${OUT} ==="
