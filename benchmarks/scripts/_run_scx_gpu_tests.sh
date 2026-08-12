#!/bin/bash
# Driver invoked by slurm_scx_gpu_tests.sh under sbatch --wrap.
# Kept separate so the wrap payload can be `bash <driver>` and we
# can use bash-isms (pipefail, arrays).
#
# Every GPU test in scx-gpu / scx-accel carries `#[ignore = "requires a CUDA
# GPU"]`, so a default `cargo test` reports it as ignored rather than as a pass
# that did nothing. This is the run that opts back in:
#
#   --include-ignored  select them
#   SCX_REQUIRE_GPU=1  a test that still cannot open a device FAILS here
#
# Without the env var the pair would just restore the old silence on a node
# whose driver broke: the tests would be selected, find no device, and return.
#
# --test-threads=1 --nocapture is what makes the per-test skip markers
# attributable: libtest prints `test <name> ... ` before the body runs, so a
# marker lands inside that test's own line.

set -uo pipefail

SCX_DIR="/home/nickyoungblut/dev/rust/scx"
cd "${SCX_DIR}"

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
export PATH=/usr/local/cuda/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
echo

# --- Preflight -------------------------------------------------------------
# Fail before spending the build rather than after: on a node with no visible
# device every gated test would panic, which is correct but reads as 204
# unrelated failures instead of one misrouted job.
if ! nvidia-smi -L 2>/dev/null | grep -q '^GPU '; then
    echo "PREFLIGHT FAILED: no CUDA device visible on $(hostname)." >&2
    echo "  This job must run on a GPU node with --gres=gpu:1." >&2
    exit 1
fi
echo "=== preflight: $(nvidia-smi -L | wc -l) device(s) visible ==="
echo

export SCX_REQUIRE_GPU=1
echo "SCX_REQUIRE_GPU=1 — a GPU test that cannot open a device is a FAILURE here,"
echo "not a skip. Optional-library skips (nvcomp, cuVS) are still allowed and are"
echo "listed in the summary below; set SCX_REQUIRE_NVCOMP=1 / SCX_REQUIRE_CUVS=1"
echo "to make those hard requirements too."
echo

LOG_DIR=$(mktemp -d)
trap 'rm -rf "${LOG_DIR}"' EXIT
STATUS=0

run_suite() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}.log"
    echo "=== ${name} ==="
    "$@" -- --include-ignored --nocapture --test-threads=1 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
        echo "!! ${name}: cargo test exited ${rc}"
        STATUS=1
    fi
    # `--include-ignored` selects everything, so a non-zero ignored count means
    # tests were filtered out by something other than us — a stale binary, or a
    # flag that did not take. Silence there is the failure mode this whole
    # change exists to remove, so treat it as one.
    local leftover
    leftover=$(grep -oE 'test result:.*; [0-9]+ ignored' "${log}" \
               | grep -oE '[0-9]+ ignored' | grep -v '^0 ignored' || true)
    if [[ -n "${leftover}" ]]; then
        echo "!! ${name}: tests still reported as ignored under --include-ignored:"
        echo "${leftover}" | sed 's/^/     /'
        STATUS=1
    fi
    echo
}

run_suite scx-gpu    cargo test -p scx-gpu --release
run_suite scx-accel  cargo test -p scx-accel --features gpu --release

# --- Summary ---------------------------------------------------------------
# What did NOT run, and why. With SCX_REQUIRE_GPU=1 the only survivors are
# optional-library gates, so this is the list of coverage this node did not
# provide — printed rather than inferred from a test that quietly vanished.
#
# The marker is NOT at the start of a line: under --nocapture --test-threads=1
# libtest writes `test <name> ... ` before the body runs, so the marker lands
# inside that test's own line. That is what makes it attributable — and it is
# why this must not be anchored with `^`, which would silently match nothing
# and report full coverage on a node that skipped half the suite.
echo "=== skipped on this node ==="
SKIPPED=$(grep -h 'SCX_GPU_TEST_SKIPPED' "${LOG_DIR}"/*.log 2>/dev/null \
    | awk -F' \\.\\.\\. SCX_GPU_TEST_SKIPPED: ' '{
          name = $1; sub(/^test /, "", name);
          reason = $2; sub(/ \(.*/, "", reason); sub(/^[^ ]+ . /, "", reason);
          printf "  %s  [%s]\n", name, reason
      }' | sort -u)
if [[ -n "${SKIPPED}" ]]; then
    echo "${SKIPPED}"
    echo
    echo "  $(printf '%s\n' "${SKIPPED}" | wc -l) test(s) skipped. Under SCX_REQUIRE_GPU=1 a device"
    echo "  gate cannot skip, so these are optional-library gates (nvcomp, cuVS)"
    echo "  — coverage this node did not provide, not tests that stopped existing."
else
    echo "(none — every GPU test executed)"
fi
echo

echo "=== done (status ${STATUS}) ==="
exit ${STATUS}
