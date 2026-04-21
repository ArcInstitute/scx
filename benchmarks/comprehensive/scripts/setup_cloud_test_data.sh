#!/usr/bin/env bash
#
# One-time staging of cloud benchmark fixtures into the shared test bucket
# (gs://arc-ctc-nextflow/scx-test/ by default). Idempotent — skips any
# fixture whose cloud path already exists.
#
# Fixtures staged by this script:
#
#   small / medium: the pbmc3k / tabula_sapiens_100k / census_* datasets
#     used by the ``cloud_*`` benchmarks. These are typically auto-uploaded
#     by ``ensure_cloud_fixture`` on first benchmark run, but running this
#     script up front avoids paying the upload cost inside a timed run.
#
#   large-atlas: the 50 GB+ atlas consumed by the Phase F.5
#     ``cloud_large_atlas`` benchmark. That benchmark deliberately does
#     NOT auto-upload — a 50 GB upload is not a thing we want to trigger
#     implicitly from a bench invocation. Staging happens here.
#
# Prerequisites:
#   - ``gcloud`` authenticated for project c-tc-429521 OR
#     ``GOOGLE_APPLICATION_CREDENTIALS=~/.gcp/scx-bench.json``
#   - ``gsutil`` on PATH
#   - The local SCX files already materialized (run the small /
#     tabula_sapiens_100k / census conversion pipeline first)
#
# Usage:
#   source .env                       # loads SCX_WORK_DIR
#   bash benchmarks/comprehensive/scripts/setup_cloud_test_data.sh
#
# Dry-run (print gsutil commands without running them):
#   DRY_RUN=1 bash benchmarks/comprehensive/scripts/setup_cloud_test_data.sh

set -euo pipefail

BUCKET="${GCS_TEST_BUCKET:-gs://arc-ctc-nextflow/scx-test}"
DATA_DIR="${SCX_WORK_DIR:-/large_storage/arcinfra/projects/scx/benchmarks/datasets}"
DRY_RUN="${DRY_RUN:-0}"

# (dataset_name, local_dir_or_file, cloud_suffix) — cloud_suffix matches
# the ``_FORMAT_KEY_TO_CLOUD_SUFFIX`` map in config.py.
SMALL_FIXTURES=(
    "pbmc3k:pbmc3k_auto.scx:.scxd"
    "tabula_sapiens_100k:tabula_sapiens_100k_auto.scx:.scxd"
)

# Large atlas fixture for F.5. Adjust the local path once the atlas is
# materialized; leaving the placeholder here to fail loudly if run before
# the atlas exists locally.
LARGE_ATLAS_NAME="${LARGE_ATLAS_NAME:-census_10m}"
LARGE_ATLAS_LOCAL="${LARGE_ATLAS_LOCAL:-$DATA_DIR/${LARGE_ATLAS_NAME}_auto.scx}"

info() { echo "[setup_cloud_test_data] $*" >&2; }

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] $*"
    else
        info "exec: $*"
        "$@"
    fi
}

exists_on_gcs() {
    local url="$1"
    gsutil -q stat "$url" 2>/dev/null || gsutil -q ls "${url}**" 2>/dev/null >/dev/null
}

stage_scx_fixture() {
    local name="$1"
    local local_file="$2"
    local suffix="$3"
    local remote="${BUCKET}/${name}${suffix}/"

    if ! [[ -f "$local_file" ]]; then
        info "SKIP $name — local source missing ($local_file). Convert first."
        return 0
    fi

    if exists_on_gcs "$remote"; then
        # Compare local BLAKE3 against the cloud sidecar. When they match,
        # skip re-upload; when they differ or no sidecar exists, re-push.
        local local_hash remote_hash
        local_hash=$(python -c "
import sys
sys.path.insert(0, '$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")/..')
from benchmarks.comprehensive.cloud_fixtures import compute_blake3
from pathlib import Path
print(compute_blake3(Path(sys.argv[1])))
" "$local_file" 2>/dev/null || echo "")
        remote_hash=$(gsutil -q cat "${remote%/}.blake3" 2>/dev/null | tr -d '[:space:]' || echo "")
        if [[ -n "$local_hash" && -n "$remote_hash" && "$local_hash" == "$remote_hash" ]]; then
            info "SKIP $name — BLAKE3 matches cloud sidecar (${local_hash:0:12}…)"
            return 0
        fi
        info "REPUSH $name — digest drift or missing sidecar (local=${local_hash:0:12}…, remote=${remote_hash:0:12}…)"
    fi

    info "PUSH $name: $local_file → $remote"
    run_cmd python -c "
import sys, pyscx
pyscx.push(sys.argv[1], sys.argv[2])
" "$local_file" "$remote"
}

# ---------------------------------------------------------------------------
# Small / medium fixtures
# ---------------------------------------------------------------------------

for spec in "${SMALL_FIXTURES[@]}"; do
    IFS=':' read -r name rel_file suffix <<<"$spec"
    stage_scx_fixture "$name" "$DATA_DIR/$rel_file" "$suffix"
done

# ---------------------------------------------------------------------------
# F.5 large-atlas fixture
# ---------------------------------------------------------------------------

info "--- large-atlas fixture (Phase F.5) ---"
if ! [[ -f "$LARGE_ATLAS_LOCAL" ]]; then
    info "SKIP large-atlas — local file missing: $LARGE_ATLAS_LOCAL"
    info "      Override via LARGE_ATLAS_LOCAL=/path/to/atlas.scx"
else
    stage_scx_fixture "$LARGE_ATLAS_NAME" "$LARGE_ATLAS_LOCAL" ".scxd"
fi

info "done."
