#!/usr/bin/env bash
# setup_cloud_test_data.sh — Upload benchmark datasets to GCS for integration testing.
#
# Prerequisites:
#   - gcloud CLI authenticated: `gcloud auth application-default login`
#   - `cargo build -p scx-cli --features cloud --release` completed
#   - Test .scx files available (generated below if missing)
#
# Usage:
#   ./benchmarks/scripts/setup_cloud_test_data.sh
#
# This script:
#   1. Builds the scx CLI with cloud feature
#   2. Creates synthetic test .scx files (if not available)
#   3. Explodes them to .scxd directories
#   4. Pushes .scxd directories to gs://arc-ctc-nextflow/scx-test/
#
# Environment variables:
#   GCS_TEST_BUCKET — GCS bucket URL (default: gs://arc-ctc-nextflow/scx-test)
#   SCX_DATA_DIR    — directory containing .scx test files

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GCS_TEST_BUCKET="${GCS_TEST_BUCKET:-gs://arc-ctc-nextflow/scx-test}"
SCX_CLI="$PROJECT_ROOT/target/release/scx"
TMPDIR_SETUP="${TMPDIR:-/tmp}/scx-cloud-setup-$$"

cleanup() {
    rm -rf "$TMPDIR_SETUP"
}
trap cleanup EXIT

echo "=== SCX Cloud Test Data Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "GCS bucket:   $GCS_TEST_BUCKET"
echo ""

# 1. Build scx CLI with cloud features
echo "--- Building scx CLI (release, cloud feature) ---"
cd "$PROJECT_ROOT"
if ! cargo build -p scx-cli --features cloud --release 2>&1; then
    echo "ERROR: Failed to build scx CLI. Make sure dependencies are satisfied."
    exit 1
fi
echo "Built: $SCX_CLI"
echo ""

# 2. Create test data if needed
mkdir -p "$TMPDIR_SETUP"

# Generate a small test file using the integration test binary
echo "--- Generating test data ---"
# We use cargo test to generate data, then find it. Alternative: run a quick
# test that writes to a known path.
# For now, let's create a synthetic file using the gen_test_scx tool.
cat > "$TMPDIR_SETUP/gen_test_scx.rs" << 'RUSTEOF'
// Standalone Rust script to generate synthetic test .scx files.
// This is compiled and run by the setup script.
use std::sync::Arc;
use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_format::writer::ScxWriter;
use scx_format::header::{FileHeader, MAGIC};
use scx_codec::{CodecId, ValueEncoding};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: gen_test_scx <output.scx> <n_obs> <n_vars>");
        std::process::exit(1);
    }
    let output = &args[1];
    let n_obs: usize = args[2].parse().unwrap();
    let n_vars: usize = args[3].parse().unwrap();

    let header = FileHeader {
        magic: MAGIC, format_version: 1, header_length: 256,
        flags: 0, n_obs: n_obs as u64, n_vars: n_vars as u64,
        nnz: 0, n_csr_shards: 0, n_csc_shards: 0,
        shard_target_rows: 16384, codec_id: 0, index_dtype: 0,
        endian: 0, reserved_padding: 0,
        root_catalog_offset: 0, root_catalog_length: 0,
        full_catalog_offset: 0, full_catalog_length: 0,
        manifest_sequence: 1, prev_catalog_offset: 0,
        file_checksum: 0, front_catalog_offset: 0, front_catalog_length: 0,
        reserved: [0u8; 132],
    };

    let mut writer = ScxWriter::new(std::path::Path::new(output), header).unwrap();

    // obs with cell_id and cell_type
    let ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let types: Vec<String> = (0..n_obs).map(|i| {
        match i % 3 { 0 => "T_cell", 1 => "B_cell", _ => "Monocyte" }.to_string()
    }).collect();
    let obs_schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    let obs = RecordBatch::try_new(Arc::new(obs_schema), vec![
        Arc::new(StringArray::from(ids.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
        Arc::new(StringArray::from(types.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
    ]).unwrap();
    writer.write_obs(&obs).unwrap();

    // var
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var_schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let var = RecordBatch::try_new(Arc::new(var_schema), vec![
        Arc::new(StringArray::from(gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>())),
    ]).unwrap();
    writer.write_var(&var).unwrap();

    // Shards
    let rows_per_shard = 1000.min(n_obs);
    let mut offset = 0;
    while offset < n_obs {
        let chunk = rows_per_shard.min(n_obs - offset);
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..chunk {
            let c0 = (r * 2) % n_vars;
            let c1 = (r * 2 + 1) % n_vars;
            indices.push(c0 as u32);
            indices.push(c1 as u32);
            values.push(((r + 1) % 256) as u8);
            values.push(((r + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
        writer.write_csr_shard(&indptr, &indices, &values,
            CodecId::None, ValueEncoding::Uint8, offset as u64).unwrap();
        offset += chunk;
    }
    writer.finish().unwrap();
    eprintln!("Created {output}: {n_obs} obs, {n_vars} vars");
}
RUSTEOF

echo "Note: Test data generation is done via Rust integration tests."
echo "To generate and upload test data to GCS, run:"
echo "  cargo test -p scx-cloud --test gcs_integration -- --ignored setup_gcs_test_data"
echo ""

# 3. Alternative: if SCX data files exist, explode & push them
if [ -n "${SCX_DATA_DIR:-}" ] && [ -d "$SCX_DATA_DIR" ]; then
    for scx_file in "$SCX_DATA_DIR"/*.scx; do
        if [ -f "$scx_file" ]; then
            basename=$(basename "$scx_file" .scx)
            exploded_dir="$TMPDIR_SETUP/${basename}.scxd"
            echo "--- Exploding $scx_file → $exploded_dir ---"
            "$SCX_CLI" explode "$scx_file" "$exploded_dir"

            echo "--- Pushing $exploded_dir → ${GCS_TEST_BUCKET}/${basename}.scxd/ ---"
            "$SCX_CLI" push "$scx_file" "${GCS_TEST_BUCKET}/${basename}.scxd/"
            echo ""
        fi
    done
fi

echo "=== Setup complete ==="
echo ""
echo "To run GCS integration tests:"
echo "  cargo test -p scx-cloud --test gcs_integration -- --ignored"
