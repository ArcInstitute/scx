#!/bin/bash
# Orchestrate the doublet-interop benchmark, ONE DATASET AT A TIME.
#
# This is an *orchestrator*, not a job: it runs in the foreground (from any CPU
# host — see benchmarks/README.md) and calls `run_parallel.py` once per dataset,
# blocking until each finishes before starting the next. Do not sbatch it; it
# submits SLURM jobs itself, and wrapping it in a job would nest submissions.
#
# The one-at-a-time rule is load-bearing here, not a style preference
# (CLAUDE.local.md):
#
#   * Every job imports pyscx from the single editable `.so` in the repo, so a
#     concurrent `maturin develop` invalidates a job that never builds anything.
#   * Slurm will co-schedule two of your jobs on one node, which biases a
#     timing comparison unevenly across its arms — worse than noise, because it
#     looks like a result.
#
# Sizing overrides matter too. Left to itself the orchestrator sizes this
# benchmark at 15 minutes on `cpu_preemptible`; scDblFinder on a real donor
# takes longer than that, and preemptible jobs on this cluster can starve for
# most of a day. Hence the explicit `--partition cpu` and a generous timeout.
#
# Usage:
#   conda activate scx-bench
#   bash benchmarks/comprehensive/scripts/run_slurm_doublet.sh
#   bash benchmarks/comprehensive/scripts/run_slurm_doublet.sh tabula_sapiens_100k

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PARTITION="${DOUBLET_PARTITION:-cpu}"
# Minutes. scDblFinder is the long pole; 3 donors of tabula_sapiens_100k took
# well under an hour by hand in the Phase-4 run, so 8h is generous headroom
# rather than a measured need.
TIMEOUT="${DOUBLET_TIMEOUT:-480}"
# The `cpu` partition bills memory as CPU-equivalent (4 GB/CPU), so a large
# floor quietly raises the charge; 200 is the documented Tier-2 figure.
MEM_GB="${DOUBLET_MEM_GB:-200}"

# Cheapest first, so a harness bug surfaces on pbmc3k in minutes rather than
# after hours of scDblFinder on the atlas slice.
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  DATASETS=(pbmc3k pbmc10k tabula_sapiens_100k)
fi

cd "$REPO_ROOT" || exit 1

if [ -z "${CONDA_PREFIX:-}" ] || [[ "$CONDA_PREFIX" != *scx-bench* ]]; then
  echo "ERROR: run this from the scx-bench conda env (benchmarks/README.md)." >&2
  echo "  conda activate scx-bench" >&2
  exit 1
fi

# Preflight the thing about to be measured, rather than discovering a missing
# tool env after scDblFinder has already run. Costs a second; the alternative
# costs hours.
echo "=== tool environments ==="
python - <<'PY' || exit 1
import json, sys
from benchmarks.comprehensive.scripts.doublet._tool_env import probe_all
envs = probe_all()
for tool, rec in envs.items():
    state = "OK  " if rec.get("available") else "MISS"
    print(f"  {state} {tool:12s} {rec.get('env_name', '?'):14s} "
          f"{rec.get('versions', rec.get('reason', ''))}")
if not any(e.get("available") for e in envs.values()):
    sys.exit("no doublet tool is available in any environment")
if sum(1 for e in envs.values() if e.get("available")) < 2:
    print("  NOTE: fewer than two tools available — the pairwise agreement "
          "and score-rank arms will be empty this run.")
PY

rc_all=0
for ds in "${DATASETS[@]}"; do
  echo
  echo "=== $ds — started $(date) ==="
  python benchmarks/comprehensive/scripts/run_parallel.py \
    --benchmarks doublet_interop \
    --formats scx_auto \
    --datasets "$ds" \
    --partition "$PARTITION" \
    --timeout "$TIMEOUT" \
    --mem-gb "$MEM_GB" \
    --skip-smoke
  rc=$?
  echo "=== $ds — finished $(date), exit=$rc ==="
  # Keep going on failure: a dataset whose tool env is missing should not
  # strand the rest, which is the same reasoning as `--dependency=afterany`.
  [ $rc -ne 0 ] && rc_all=$rc
done

echo
echo "All ${#DATASETS[@]} dataset(s) done; worst exit=$rc_all"
exit $rc_all
