//! Shared fixture helpers for `IndexPlanLoader` integration tests.
//!
//! These mirror the inline fixture builders in `src/index_plan.rs::tests`
//! but live here so multiple integration tests can share them without
//! either duplicating the code or exposing test-only helpers from `lib.rs`.

#![allow(dead_code)]

use std::sync::Arc as StdArc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use scx_codec::{CodecId, ValueEncoding};
use scx_format::header::{FileHeader, MAGIC};
use scx_format::writer::ScxWriter;
use scx_loader::{IndexPlanLoader, LoaderConfig};

/// Build a multi-shard `.scx` fixture with one nonzero per row at column
/// `row % n_vars` carrying value `((row + 1) & 0xFF) as u8`. The
/// `(row, col, value)` tuple is recoverable from the row index alone, which
/// lets tests assert exact byte-level correctness without storing the dense
/// matrix separately.
pub fn write_multi_shard_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
) -> std::path::PathBuf {
    assert!(n_obs % n_shards == 0, "n_obs must divide n_shards");
    let rows_per_shard = n_obs / n_shards;

    let header = FileHeader {
        magic: MAGIC,
        format_version: 1,
        header_length: 256,
        flags: 0,
        n_obs: n_obs as u64,
        n_vars: n_vars as u64,
        nnz: n_obs as u64,
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: rows_per_shard as u32,
        codec_id: 0,
        index_dtype: 0,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        reserved: [0u8; 132],
    };
    let mut writer = ScxWriter::new(path, header).unwrap();

    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let obs = RecordBatch::try_new(
        StdArc::new(Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)])),
        vec![StdArc::new(StringArray::from(
            cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var = RecordBatch::try_new(
        StdArc::new(Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)])),
        vec![StdArc::new(StringArray::from(
            gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_var(&var).unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            let col = (row % n_vars) as u32;
            let val = ((row + 1) & 0xFF) as u8;
            indices.push(col);
            values.push(val);
            indptr.push(*indptr.last().unwrap() + 1);
        }
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
    }
    writer.finish().unwrap();
    path.to_path_buf()
}

/// Build an `IndexPlanLoader` against a fixture with default settings —
/// no normalization, no log1p, single obs column `"cell_id"`, 4 cache shards,
/// shard sort enabled, lookahead 4, plan-size cap 16384, and a generous
/// memory budget so auto-tuning does not interfere with correctness assertions.
pub fn open_loader(path: &std::path::Path, sort_by_shard: bool) -> IndexPlanLoader {
    let mut config = LoaderConfig::default();
    config.normalize = false;
    config.log1p = false;
    config.obs_columns = vec!["cell_id".to_string()];
    config.max_memory_mb = 1024;
    IndexPlanLoader::new(path, config, 4, sort_by_shard, 4, 16384).unwrap()
}

/// Same as `open_loader` but with a fully-saturating normalize+log1p config.
pub fn open_loader_normalized(
    path: &std::path::Path,
    sort_by_shard: bool,
    target_sum: f64,
) -> IndexPlanLoader {
    let mut config = LoaderConfig::default();
    config.normalize = true;
    config.log1p = true;
    config.target_sum = target_sum;
    config.obs_columns = vec!["cell_id".to_string()];
    config.max_memory_mb = 1024;
    IndexPlanLoader::new(path, config, 4, sort_by_shard, 4, 16384).unwrap()
}

/// HVG-projected loader. Validates HVG indices against `n_vars` at
/// construction.
pub fn open_loader_hvg(
    path: &std::path::Path,
    hvg: Vec<u32>,
    sort_by_shard: bool,
) -> IndexPlanLoader {
    let mut config = LoaderConfig::default();
    config.normalize = false;
    config.log1p = false;
    config.hvg_indices = Some(hvg);
    config.obs_columns = vec!["cell_id".to_string()];
    config.max_memory_mb = 1024;
    IndexPlanLoader::new(path, config, 4, sort_by_shard, 4, 16384).unwrap()
}
