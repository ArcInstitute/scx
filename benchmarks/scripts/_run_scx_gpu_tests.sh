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
# And require real kernels, not placeholders. `build.rs` writes comment-only PTX
# stubs when nvcc is absent, and until this series neither PATH nor CUDA_HOME was
# in its fingerprint -- so a CARGO_TARGET_DIR that once saw a CPU-only build
# replays the stub branch on a node that HAS nvcc. That is what killed the
# Phase 7 gate (job 2839940: CUDA_ERROR_INVALID_IMAGE, 82s, nothing measured).
#
# This job is the one place the GPU suites execute, so it is the last place that
# should run against stubs: without the variable the build succeeds, every test
# is selected, and 200+ of them fail at their first kernel launch with an opaque
# "named symbol not found" -- which reads as 200 broken kernels rather than one
# broken build. With it, the build fails in ~20 seconds and says why.
export SCX_GPU_REQUIRE_NVCC=1
# Default the free-VRAM gate ON, for the same reason SCX_REQUIRE_GPU is on: this
# job holds a whole H100 via --gres=gpu:1, so the ~10 GiB of headroom the >2^31
# indexing test needs is a property of the allocation, not a gamble. Left opt-in,
# the ONE test that can observe the 2^31 kernel overflow could quietly skip on a
# fragmented card and the suite would still go green — which is exactly the
# silence this harness exists to remove. Override with SCX_REQUIRE_LARGE_VRAM=0
# when running on a smaller or shared GPU.
export SCX_REQUIRE_LARGE_VRAM=${SCX_REQUIRE_LARGE_VRAM:-1}
echo "SCX_REQUIRE_GPU=1 — a GPU test that cannot open a device is a FAILURE here,"
echo "not a skip. SCX_REQUIRE_LARGE_VRAM=${SCX_REQUIRE_LARGE_VRAM} — likewise for the"
echo "tests that need a multi-GiB allocation to reach a 32-bit index boundary."
echo "Optional-library skips (nvcomp, cuVS) are still allowed and are listed in the"
echo "summary below; set SCX_REQUIRE_NVCOMP=1 / SCX_REQUIRE_CUVS=1 to make those"
echo "hard requirements too. All of these pass through from the submitting"
echo "environment."
echo

LOG_DIR=$(mktemp -d)
trap 'rm -rf "${LOG_DIR}"' EXIT
STATUS=0

run_suite() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}.log"

    # `--lib --tests`, NOT the default target set: `--include-ignored` is passed
    # through to *every* harness cargo starts, including the doctest one, where
    # a ```ignore fence means exactly the same flag. Three module-doc examples
    # here are illustrative pseudo-code fenced that way (cusolver, cusparse,
    # gpu_preprocess); selecting them makes rustdoc try to compile prose.
    echo "=== ${name} (lib + integration tests) ==="
    "$@" --lib --tests -- --include-ignored --nocapture --test-threads=1 2>&1 | tee "${log}"
    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
        echo "!! ${name}: cargo test exited ${rc}"
        STATUS=1
    fi

    # Doctests, with the default selection: no GPU doctest exists to opt into.
    echo "=== ${name} (doctests) ==="
    "$@" --doc 2>&1 | tail -5
    rc=${PIPESTATUS[0]}
    if [[ ${rc} -ne 0 ]]; then
        echo "!! ${name}: doctests exited ${rc}"
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
# `test_gate::tests::*` are the gate's OWN unit tests. They call `decline`
# directly to prove the skip path still prints and the strict path still
# panics, so they emit the marker while running perfectly happily on any host.
# Listing them as "coverage this node did not provide" would be false, and
# would put a line reading `[nvcomp not available]` in front of a reader for a
# test that has nothing to do with nvcomp. That module is CPU-only by
# construction — it tests the decision, never a device.
SKIPPED=$(grep -h 'SCX_GPU_TEST_SKIPPED' "${LOG_DIR}"/*.log 2>/dev/null \
    | grep -v 'test_gate::tests::' \
    | awk -F' \\.\\.\\. SCX_GPU_TEST_SKIPPED: ' '{
          name = $1; sub(/^test /, "", name);
          reason = $2; sub(/ \(.*/, "", reason); sub(/^[^ ]+ . /, "", reason);
          printf "  %s  [%s]\n", name, reason
      }' | sort -u)
if [[ -n "${SKIPPED}" ]]; then
    echo "${SKIPPED}"
    echo
    echo "  $(printf '%s\n' "${SKIPPED}" | wc -l) test(s) skipped. Under SCX_REQUIRE_GPU=1 a device"
    echo "  gate cannot skip, so these are coverage this node did not provide, not"
    echo "  tests that stopped existing."
    # The footer must read the ACTUAL value, not the default: under
    # SCX_REQUIRE_LARGE_VRAM=0 a vram_or_skip miss still prints the marker and
    # lands in this list, and asserting "these are optional-library gates" would
    # mislabel it as nvcomp/cuVS on the very override path this script documents.
    if [[ "${SCX_REQUIRE_LARGE_VRAM}" == "1" ]]; then
        echo "  SCX_REQUIRE_LARGE_VRAM=1, so a free-VRAM gate cannot skip either —"
        echo "  every entry above is an optional-library gate (nvcomp, cuVS)."
    else
        echo "  SCX_REQUIRE_LARGE_VRAM=${SCX_REQUIRE_LARGE_VRAM}, so an entry above may be a"
        echo "  free-VRAM skip as well as an optional-library one (nvcomp, cuVS)."
    fi
else
    echo "(none — every GPU test executed)"
fi
echo

echo "=== done (status ${STATUS}) ==="
exit ${STATUS}
