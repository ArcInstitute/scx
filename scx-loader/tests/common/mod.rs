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
    assert!(n_obs.is_multiple_of(n_shards), "n_obs must divide n_shards");
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
    assert!(n_obs.is_multiple_of(n_shards));
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

/// Gene count of [`write_known_multinnz_fixture`]'s output. Exported so a test
/// can range-check its HVG panel against the same number the fixture wrote,
/// rather than repeating the literal.
pub const KNOWN_MULTINNZ_N_VARS: u64 = 64;

/// Build a fixture where every row has nonzeros at the **same known** genes
/// (`2, 7, 33, 58`) with strictly-positive values. Because the columns are
/// fixed, a test can choose an HVG panel that captures some-but-not-all of a
/// row's mass and know exactly that the panel-local depth is strictly below the
/// full-transcriptome depth — the condition under which the L2 panel-local
/// normalize bug would diverge from the correct (full-depth) result. `n_vars`
/// is [`KNOWN_MULTINNZ_N_VARS`] (> the max gene id, 58).
pub fn write_known_multinnz_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_shards: usize,
) -> std::path::PathBuf {
    assert!(n_obs.is_multiple_of(n_shards), "n_obs must divide n_shards");
    const GENES: [u32; 4] = [2, 7, 33, 58];
    let n_vars: usize = KNOWN_MULTINNZ_N_VARS as usize;
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

/// Build a **deliberately malformed** single-modality fixture whose two CSR
/// shards claim overlapping global row ranges: shard 0 covers `[0, rows)` and
/// shard 1 covers `[rows - overlap, 2*rows - overlap)`.
///
/// This is the shape a merge / append / compact defect would leave behind. The
/// writer takes `row_start` from the caller and does not cross-check it against
/// previously written shards, which is what makes the fixture constructible —
/// and is also why the loader has to check for itself.
pub fn write_overlapping_shards_fixture(
    path: &std::path::Path,
    rows_per_shard: usize,
    n_vars: usize,
    overlap: usize,
) -> std::path::PathBuf {
    assert!(overlap > 0 && overlap <= rows_per_shard);
    let n_obs = 2 * rows_per_shard - overlap;

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        (2 * rows_per_shard) as u64,
        rows_per_shard as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&string_column("cell_id", n_obs)).unwrap();
    writer.write_var(&string_column("gene_id", n_vars)).unwrap();

    for row_start in [0usize, rows_per_shard - overlap] {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            indices.push(((row_start + local) % n_vars) as u32);
            values.push(1u8);
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

/// Build a **deliberately malformed** single-modality fixture whose CSR shards
/// leave a *gap*: `[0, gap_start)` and `[gap_start + gap_rows, n_obs)`, so the
/// rows in between belong to no shard at all.
///
/// The complement of [`write_overlapping_shards_fixture`]. An overlap-only
/// check passes this file and then silently loses those cells.
pub fn write_gapped_shards_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    gap_start: usize,
    gap_rows: usize,
) -> std::path::PathBuf {
    assert!(gap_rows > 0 && gap_start + gap_rows < n_obs);

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        (n_obs - gap_rows) as u64,
        gap_start.max(1) as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&string_column("cell_id", n_obs)).unwrap();
    writer.write_var(&string_column("gene_id", n_vars)).unwrap();

    for (row_start, rows) in [
        (0usize, gap_start),
        (gap_start + gap_rows, n_obs - gap_start - gap_rows),
    ] {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows {
            indices.push(((row_start + local) % n_vars) as u32);
            values.push(1u8);
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

/// Build a well-formed 2-modality fixture (`rna` / `adt`) over a shared obs
/// axis, each modality holding one CSR shard covering `[0, n_obs)`.
///
/// Legitimate on its own terms — but the two modalities' shard row ranges
/// necessarily overlap in the *flattened* catalog view, which is what a loader
/// opened without a `modality_id` would consume.
pub fn write_multimodal_fixture(
    path: &std::path::Path,
    n_obs: usize,
    rna_vars: usize,
    adt_vars: usize,
) -> std::path::PathBuf {
    use scx_format_io::modality::ModalityType;

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        rna_vars.max(adt_vars) as u64,
        (2 * n_obs) as u64,
        n_obs as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&string_column("cell_id", n_obs)).unwrap();

    let mut ids = Vec::new();
    for (name, kind, m_vars) in [
        ("adt", ModalityType::Protein, adt_vars),
        ("rna", ModalityType::Rna, rna_vars),
    ] {
        let id = writer
            .add_modality(name, kind, CodecId::None, ValueEncoding::Uint8, false)
            .unwrap();
        writer
            .write_var_for(id, &string_column("gene_id", m_vars))
            .unwrap();
        writer.set_modality_n_vars(id, m_vars as u64).unwrap();
        ids.push((id, m_vars));
    }

    for (id, m_vars) in ids {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for r in 0..n_obs {
            indices.push((r % m_vars) as u32);
            values.push(1u8);
            indptr.push(*indptr.last().unwrap() + 1);
        }
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer.write_csr_shard_for(id, 0, shard).unwrap();
    }
    writer.finish().unwrap();
    path.to_path_buf()
}

/// Build a **valid** file whose single modality is registered in the modality
/// table, so its only X is stamped `modality_id = 1` and modality 0 owns no
/// shards at all.
///
/// This is the shape `pyscx.from_mudata(MuData({"rna": adata}))` and a
/// single-modality h5mu ingest emit — as unambiguous as a legacy id-0 file, and
/// the case an "is modality 0 covered?" preflight silently rejects. Every other
/// "single modality" fixture here writes id-0 shards, which is exactly why that
/// regression got through.
pub fn write_single_registered_modality_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
) -> std::path::PathBuf {
    use scx_format_io::modality::ModalityType;
    assert!(n_obs.is_multiple_of(n_shards));
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
    writer.write_obs(&string_column("cell_id", n_obs)).unwrap();

    let rna = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer
        .write_var_for(rna, &string_column("gene_id", n_vars))
        .unwrap();
    writer.set_modality_n_vars(rna, n_vars as u64).unwrap();

    for s in 0..n_shards {
        let row_start = s * rows_per_shard;
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows_per_shard {
            indices.push(((row_start + local) % n_vars) as u32);
            values.push(1u8);
            indptr.push(*indptr.last().unwrap() + 1);
        }
        let shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer
            .write_csr_shard_for(rna, row_start as u64, shard)
            .unwrap();
    }
    writer.finish().unwrap();
    path.to_path_buf()
}

/// A 2-modality fixture where modality `adt` (id 1) tiles `[0, n_obs)` with one
/// clean shard and modality `rna` (id 2) is **malformed**: two shards whose row
/// ranges overlap, so that modality alone violates the exactly-once invariant.
///
/// Lets a test scope to a modality and still be wrong — the case
/// `ShardGroupIndex::build` cannot catch at `shard_group_size == 1`, because the
/// two overlapping shards never share a group.
pub fn write_multimodal_overlapping_fixture(
    path: &std::path::Path,
    n_obs: usize,
    n_vars: usize,
) -> std::path::PathBuf {
    use scx_format_io::modality::ModalityType;
    assert!(n_obs >= 4 && n_obs.is_multiple_of(2));

    let header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        (2 * n_obs) as u64,
        n_obs as u32,
        0,
        0,
    );
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&string_column("cell_id", n_obs)).unwrap();

    let one_row_shard = |row_start: usize, rows: usize| {
        let mut indptr = vec![0u64];
        let mut indices = Vec::new();
        let mut values = Vec::new();
        for local in 0..rows {
            indices.push(((row_start + local) % n_vars) as u32);
            values.push(1u8);
            indptr.push(*indptr.last().unwrap() + 1);
        }
        (indptr, indices, values)
    };

    // id 1 — clean: one shard covering [0, n_obs).
    let adt = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer
        .write_var_for(adt, &string_column("gene_id", n_vars))
        .unwrap();
    writer.set_modality_n_vars(adt, n_vars as u64).unwrap();
    let (indptr, indices, values) = one_row_shard(0, n_obs);
    let adt_shard = scx_format_io::ShardBuffers::new(
        &indptr,
        &indices,
        &values,
        CodecId::None,
        ValueEncoding::Uint8,
    );
    writer.write_csr_shard_for(adt, 0, adt_shard).unwrap();

    // id 2 — malformed: [0, n_obs/2 + 1) and [n_obs/2 - 1, n_obs), overlapping
    // on two rows.
    let rna = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::None,
            ValueEncoding::Uint8,
            false,
        )
        .unwrap();
    writer
        .write_var_for(rna, &string_column("gene_id", n_vars))
        .unwrap();
    writer.set_modality_n_vars(rna, n_vars as u64).unwrap();
    let half = n_obs / 2;
    for (row_start, rows) in [(0usize, half + 1), (half - 1, n_obs - half + 1)] {
        let (indptr, indices, values) = one_row_shard(row_start, rows);
        let rna_shard = scx_format_io::ShardBuffers::new(
            &indptr,
            &indices,
            &values,
            CodecId::None,
            ValueEncoding::Uint8,
        );
        writer
            .write_csr_shard_for(rna, row_start as u64, rna_shard)
            .unwrap();
    }

    writer.finish().unwrap();
    path.to_path_buf()
}

/// Single-column `RecordBatch` of `{prefix}_{i}` strings — the obs/var shape
/// every fixture here writes.
fn string_column(name: &str, n: usize) -> RecordBatch {
    let prefix = name.trim_end_matches("_id");
    let ids: Vec<String> = (0..n).map(|i| format!("{prefix}_{i}")).collect();
    RecordBatch::try_new(
        StdArc::new(Schema::new(vec![Field::new(name, DataType::Utf8, false)])),
        vec![StdArc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// Build an `IndexPlanLoader` against a fixture with default settings —
/// no normalization, no log1p, single obs column `"cell_id"`, 4 cache shards,
/// shard sort enabled, lookahead 4, plan-size cap 16384, and a generous
/// memory budget so auto-tuning does not interfere with correctness assertions.
pub fn open_loader(path: &std::path::Path, sort_by_shard: bool) -> IndexPlanLoader {
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
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
    let config = LoaderConfig {
        normalize,
        log1p,
        target_sum,
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    IndexPlanLoader::new(path, config, 4, sort_by_shard, 4, 16384).unwrap()
}

/// HVG-projected loader. Validates HVG indices against `n_vars` at
/// construction.
pub fn open_loader_hvg(
    path: &std::path::Path,
    hvg: Vec<u32>,
    sort_by_shard: bool,
) -> IndexPlanLoader {
    let config = LoaderConfig {
        normalize: false,
        log1p: false,
        hvg_indices: Some(hvg),
        obs_columns: vec!["cell_id".to_string()],
        max_memory_mb: 1024,
        ..Default::default()
    };
    IndexPlanLoader::new(path, config, 4, sort_by_shard, 4, 16384).unwrap()
}
