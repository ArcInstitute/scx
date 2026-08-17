//! The all-families fixture: one SCX file carrying every section family an op
//! can decide about.
//!
//! Shared between `section_carry.rs`, which asserts what each op does with each
//! family, and `testkit_against_real_ops.rs`, which asserts that the digest
//! harness still works when the file is this rich. Both need the same file for
//! the same reason: a fixture with only X and obs makes most of what a rewrite
//! op does invisible.
//!
//! Not every helper is used by every consumer, hence the blanket `dead_code`
//! allow — the alternative is per-item attributes that go stale.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float32Array, Int32Array, Int64Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;

pub const N_OBS: usize = 8;
pub const N_VARS: usize = 6;
/// Two X shards, so `compact` and `sort` have a row layout to actually change.
/// With one shard several ops are trivially identity and the test proves less.
pub const SHARD_ROWS: usize = 4;
pub const RAW_N_VARS: usize = 9;

/// Every family the format can carry, in one file.
///
/// Built by hand rather than by running an op, so it does not inherit any op's
/// idea of what a file contains — which is the thing under test.
pub fn fixture_all_families(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    let mut writer = ScxWriter::new(
        &path,
        FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, SHARD_ROWS as u32, 0, 0),
    )
    .unwrap();

    // --- obs / var -------------------------------------------------------
    let obs = obs_batch();
    writer.write_obs(&obs).unwrap();
    let var = var_batch(N_VARS, "gene");
    writer.write_var(&var).unwrap();

    // --- X, in two shards ------------------------------------------------
    let mut shard_ranges: Vec<(u64, u64)> = Vec::new();
    for (shard_idx, row_start) in (0..N_OBS).step_by(SHARD_ROWS).enumerate() {
        let (indptr, indices, values) = csr_rows(row_start, SHARD_ROWS, N_VARS, 1);
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
        shard_ranges.push((row_start as u64, (row_start + SHARD_ROWS) as u64));

        // --- detection bitmaps, keyed to this shard's local rows ---------
        let shard = scx_format_io::bitmap::BitmapShard::build_from_csr(
            row_start as u64,
            SHARD_ROWS as u32,
            N_VARS as u32,
            &indptr,
            &indices,
        );
        writer.write_bitmap_shard(&shard).unwrap();
        let _ = shard_idx;
    }

    // --- a layer ---------------------------------------------------------
    let (l_indptr, l_indices, l_values) = csr_rows(0, N_OBS, N_VARS, 3);
    writer
        .write_layer_csr_shard(
            &l_indptr,
            &l_indices,
            &l_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
            "spliced",
            0,
        )
        .unwrap();

    // --- obsm / varm -----------------------------------------------------
    writer.write_obsm("X_pca", &dense_embedding(N_OBS)).unwrap();
    writer.write_varm("PCs", &dense_embedding(N_VARS)).unwrap();

    // --- obsp / varp -----------------------------------------------------
    writer
        .write_obsp_shard_coo(
            "connectivities",
            0,
            0,
            N_OBS as u64,
            N_OBS as u64,
            &coo_batch(N_OBS),
        )
        .unwrap();
    writer.write_varp("gene_corr", &coo_i32(N_VARS)).unwrap();

    // --- uns -------------------------------------------------------------
    writer
        .write_uns(&serde_json::json!({ "carry_fixture": true }))
        .unwrap();

    // --- adata.raw (its own, wider var axis) -----------------------------
    writer.set_raw_n_vars(RAW_N_VARS as u64);
    let (r_indptr, r_indices, r_values) = csr_rows(0, N_OBS, RAW_N_VARS, 7);
    writer
        .write_raw_csr_shard(
            &r_indptr,
            &r_indices,
            &r_values,
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.write_raw_var(&var_batch(RAW_N_VARS, "raw")).unwrap();

    // --- the grouped-sort sidecar ----------------------------------------
    writer
        .write_group_index(
            serde_json::json!({
                "group_by": "cell_type",
                "reference_shard": 0,
                "reference_labels": ["T cell"],
                "records": [],
            })
            .to_string()
            .as_bytes(),
        )
        .unwrap();

    // --- predicate indexes ------------------------------------------------
    let opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["cell_type".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let (mut outcomes, mut cols) = (Vec::new(), Vec::new());
    let obs_bytes = scx_engine::build_obs_predicate_index_bytes(
        &obs,
        &shard_ranges,
        &opts,
        &mut outcomes,
        &mut cols,
    )
    .unwrap()
    .expect("the fixture's cell_type column must produce an obs predicate index");
    writer.write_obs_predicate_index(&obs_bytes).unwrap();

    let var_opts = scx_engine::PredicateIndexBuildOptions {
        forced_columns: vec!["gene_kind".to_string()],
        preset_columns: Vec::new(),
        auto_threshold: 0,
        high_cardinality_threshold: 100_000,
    };
    let (mut v_outcomes, mut v_cols) = (Vec::new(), Vec::new());
    let var_bytes = scx_engine::build_var_predicate_index_bytes(
        &var,
        &[(0, N_VARS as u64)],
        &var_opts,
        &mut v_outcomes,
        &mut v_cols,
    )
    .unwrap()
    .expect("the fixture's gene_kind column must produce a var predicate index");
    writer.write_var_predicate_index(&var_bytes).unwrap();

    writer.finish().unwrap();

    // --- deletion vectors, and the CSC sidecar ----------------------------
    // Both via their real ops: a hand-written deletion vector would not be
    // exercising the same section the ops read, and `build-csc` is the only
    // thing that writes a CSC sidecar.
    scx_ops::mark_deleted(&path, &[2, 5]).unwrap();
    path
}

fn obs_batch() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cell_id", DataType::Utf8, false),
            Field::new("cell_type", DataType::Utf8, true),
            Field::new("n_counts", DataType::UInt32, false),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..N_OBS).map(|i| format!("cell_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..N_OBS)
                    .map(|i| if i % 2 == 0 { "T cell" } else { "B cell" })
                    .collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(
                (0..N_OBS).map(|i| (100 + i) as u32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn var_batch(n: usize, prefix: &str) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("gene_id", DataType::Utf8, false),
            Field::new("gene_kind", DataType::Utf8, true),
        ])),
        vec![
            Arc::new(StringArray::from(
                (0..n).map(|i| format!("{prefix}_{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                (0..n)
                    .map(|i| if i % 3 == 0 { "mito" } else { "nuclear" })
                    .collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// Two nonzeros per row, deterministic, with `salt` separating X from the layer
/// and from raw so a mixed-up carry shows as wrong values rather than as a pass.
fn csr_rows(
    row_start: usize,
    n_rows: usize,
    n_cols: usize,
    salt: usize,
) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
    let mut indptr = vec![0u64];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for r in 0..n_rows {
        let row = row_start + r;
        let (c0, c1) = ((row * 2) % n_cols, (row * 2 + 1) % n_cols);
        let (lo, hi) = if c0 <= c1 { (c0, c1) } else { (c1, c0) };
        indices.push(lo as u32);
        indices.push(hi as u32);
        values.push(((row + salt) % 255 + 1) as u8);
        values.push(((row + salt + 1) % 255 + 1) as u8);
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values)
}

fn dense_embedding(n_rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("0", DataType::Float32, false),
            Field::new("1", DataType::Float32, false),
        ])),
        vec![
            Arc::new(Float32Array::from(
                (0..n_rows).map(|i| i as f32).collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n_rows).map(|i| -(i as f32)).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// obs×obs COO in the Int64 form `write_obsp_shard_coo` takes.
fn coo_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int64, false),
            Field::new("col", DataType::Int64, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())),
            Arc::new(Int64Array::from(
                (0..n as i64)
                    .map(|r| (r + 1) % n as i64)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// var×var COO in the Int32 form `write_varp` documents.
fn coo_i32(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new_with_metadata(
        vec![
            Field::new("row", DataType::Int32, false),
            Field::new("col", DataType::Int32, false),
            Field::new("data", DataType::Float32, false),
        ],
        HashMap::from([
            ("n_rows".to_string(), n.to_string()),
            ("n_cols".to_string(), n.to_string()),
        ]),
    ));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from((0..n as i32).collect::<Vec<_>>())),
            Arc::new(Int32Array::from(
                (0..n as i32)
                    .map(|r| (r + 1) % n as i32)
                    .collect::<Vec<_>>(),
            )),
            Arc::new(Float32Array::from(
                (0..n).map(|i| (i + 1) as f32).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}
