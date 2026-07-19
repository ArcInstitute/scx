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
use scx_format_io::header::FileHeader;
use scx_format_io::writer::ScxWriter;
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

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        n_obs as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();

    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    let obs = RecordBatch::try_new(
        StdArc::new(Schema::new(vec![Field::new(
            "cell_id",
            DataType::Utf8,
            false,
        )])),
        vec![StdArc::new(StringArray::from(
            cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap();
    writer.write_obs(&obs).unwrap();

    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    let var = RecordBatch::try_new(
        StdArc::new(Schema::new(vec![Field::new(
            "gene_id",
            DataType::Utf8,
            false,
        )])),
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

/// Build a dense multi-shard `.scx` fixture with strictly-increasing,
/// collision-free columns (gaps up to 250) written with the given `codec`.
/// With `CodecId::Scx1` + dense integer rows the writer emits a per-row decode
/// sidecar per shard (within the 25% overhead budget); with `CodecId::None` it
/// emits none. Two calls with identical params produce byte-identical logical
/// data, enabling a sidecar-vs-full-shard parity check. Mirrors
/// `index_plan.rs::tests::write_dense_scx1_fixture`.
pub fn write_dense_scx1_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_shards: usize,
    nnz_per_row: usize,
    codec: CodecId,
) -> std::path::PathBuf {
    assert!(n_obs % n_shards == 0);
    let rows_per_shard = n_obs / n_shards;
    let n_vars = nnz_per_row * 251 + 16;
    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        (n_obs * nnz_per_row) as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();

    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    writer
        .write_obs(
            &RecordBatch::try_new(
                StdArc::new(Schema::new(vec![Field::new(
                    "cell_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![StdArc::new(StringArray::from(
                    cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    writer
        .write_var(
            &RecordBatch::try_new(
                StdArc::new(Schema::new(vec![Field::new(
                    "gene_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![StdArc::new(StringArray::from(
                    gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            let mut col = 0u32;
            for k in 0..nnz_per_row {
                col += 1 + ((row * 13 + k * 7) % 250) as u32;
                indices.push(col);
                values.push(1u8 + ((row + k) % 5) as u8);
            }
            indptr.push(*indptr.last().unwrap() + nnz_per_row as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                codec,
                ValueEncoding::Uint8,
                row_start as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path.to_path_buf()
}

/// Build a fixture where every row has nonzeros at the **same known** genes
/// (`2, 7, 33, 58`) with strictly-positive values. Because the columns are
/// fixed, a test can choose an HVG panel that captures some-but-not-all of a
/// row's mass and know exactly that the panel-local depth is strictly below the
/// full-transcriptome depth — the condition under which the L2 panel-local
/// normalize bug would diverge from the correct (full-depth) result. `n_vars`
/// is 64 (> the max gene id, 58).
pub fn write_known_multinnz_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_shards: usize,
) -> std::path::PathBuf {
    assert!(n_obs % n_shards == 0, "n_obs must divide n_shards");
    const GENES: [u32; 4] = [2, 7, 33, 58];
    let n_vars: usize = 64;
    let rows_per_shard = n_obs / n_shards;
    let nnz_per_row = GENES.len();

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        (n_obs * nnz_per_row) as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();

    let cell_ids: Vec<String> = (0..n_obs).map(|i| format!("cell_{i}")).collect();
    writer
        .write_obs(
            &RecordBatch::try_new(
                StdArc::new(Schema::new(vec![Field::new(
                    "cell_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![StdArc::new(StringArray::from(
                    cell_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();
    let gene_ids: Vec<String> = (0..n_vars).map(|i| format!("gene_{i}")).collect();
    writer
        .write_var(
            &RecordBatch::try_new(
                StdArc::new(Schema::new(vec![Field::new(
                    "gene_id",
                    DataType::Utf8,
                    false,
                )])),
                vec![StdArc::new(StringArray::from(
                    gene_ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                ))],
            )
            .unwrap(),
        )
        .unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            let row = row_start + local;
            for (k, &g) in GENES.iter().enumerate() {
                indices.push(g);
                // Strictly positive, gene- and row-dependent so panel and full
                // depth vary across rows.
                values.push(1u8 + ((row + k * 3) % 7) as u8);
            }
            indptr.push(*indptr.last().unwrap() + nnz_per_row as u64);
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
    open_loader_with_flags(path, true, true, target_sum, sort_by_shard)
}

/// Build an `IndexPlanLoader` with arbitrary `(normalize, log1p)` flags.
/// Lets per-test parity checks exercise all four combinations through the
/// public `process_plan` surface without per-test config boilerplate.
pub fn open_loader_with_flags(
    path: &std::path::Path,
    normalize: bool,
    log1p: bool,
    target_sum: f64,
    sort_by_shard: bool,
) -> IndexPlanLoader {
    let mut config = LoaderConfig::default();
    config.normalize = normalize;
    config.log1p = log1p;
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
