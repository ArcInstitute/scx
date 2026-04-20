#!/usr/bin/env bash
#
# On-demand regression gate: capture a candidate snapshot, then compare it
# against the canonical baseline under ``results/baselines/LATEST``. One
# command for the manual workflow — no cron, no scheduling.
#
# Usage:
#
#   # Default: small tier, candidate named candidate_<git-sha>
#   bash benchmarks/comprehensive/scripts/gate_candidate.sh
#
#   # Larger tier:
#   bash benchmarks/comprehensive/scripts/gate_candidate.sh --tier full
#
#   # Skip capture and re-use an existing candidate dir (useful when the
#   # capture already ran and you just want to replay the gate):
#   bash benchmarks/comprehensive/scripts/gate_candidate.sh \
#       --skip-capture --name candidate_abc1234
#
#   # Custom label (defaults to candidate_<git-sha>-<YYYYMMDD>):
#   bash benchmarks/comprehensive/scripts/gate_candidate.sh --name my_label
#
# The gate's exit code bubbles up: 0 = pass, 1 = regression / floor /
# fingerprint, 2 = missing inputs. `--justifications` and `--thresholds`
# auto-default to the committed repo paths; override via env if needed.

set -euo pipefail

# Resolve repo root (this script lives 3 dirs deep from the root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
COMPREHENSIVE="$REPO_ROOT/benchmarks/comprehensive"

# Defaults
TIER="small"
SKIP_CAPTURE=0
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
NAME="candidate_${GIT_SHA}_$(date +%Y%m%d)"
PYTHON="${PYTHON:-python}"
BASELINE=""   # empty → compare script picks results/baselines/LATEST
EXTRA_GATE_ARGS=()

usage() {
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier)         TIER="$2"; shift 2 ;;
        --name)         NAME="$2"; shift 2 ;;
        --baseline)     BASELINE="$2"; shift 2 ;;
        --skip-capture) SKIP_CAPTURE=1; shift ;;
        --python)       PYTHON="$2"; shift 2 ;;
        -h|--help)      usage; exit 0 ;;
        # Anything else is passed through to compare_against_baseline.py.
        *)              EXTRA_GATE_ARGS+=("$1"); shift ;;
    esac
done

CANDIDATE_DIR="$COMPREHENSIVE/results/$NAME"

echo "[gate_candidate] repo_root=$REPO_ROOT"
echo "[gate_candidate] candidate=$CANDIDATE_DIR (tier=$TIER)"

if [[ "$SKIP_CAPTURE" -eq 0 ]]; then
    echo "[gate_candidate] capturing baseline snapshot …"
    "$PYTHON" "$COMPREHENSIVE/scripts/capture_baseline.py" \
        --tier "$TIER" --name "$NAME"
else
    echo "[gate_candidate] --skip-capture: reusing existing $CANDIDATE_DIR"
    if [[ ! -d "$CANDIDATE_DIR" ]]; then
        echo "[gate_candidate] ERROR: $CANDIDATE_DIR does not exist" >&2
        exit 2
    fi
fi

echo "[gate_candidate] running gate …"
GATE_CMD=(
    "$PYTHON" "$COMPREHENSIVE/scripts/compare_against_baseline.py"
    --current "$CANDIDATE_DIR"
    --gate
)
if [[ -n "$BASELINE" ]]; then
    GATE_CMD+=(--baseline "$BASELINE")
fi
GATE_CMD+=("${EXTRA_GATE_ARGS[@]}")

set +e
"${GATE_CMD[@]}"
RC=$?
set -e

echo "[gate_candidate] gate exit=$RC (0=pass, 1=regression, 2=missing inputs)"
exit "$RC"
