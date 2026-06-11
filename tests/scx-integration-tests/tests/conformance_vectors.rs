//! Extended conformance vectors beyond the codec-backward-compat
//! `golden_*.scx` corpus.
//!
//! Covers v1 layers/obsm/uns, CSC sidecar, predicate indexes, deletion
//! vectors, v2 multimodal (CITE-seq + partial CSC), Phase 5b bitmap, and
//! cloud-optimized + exploded layouts. Sidecar JSON captures the
//! observable shape (header summary, catalog summary, CSR triplet for
//! fixtures with X) so future readers can diff intentional format
//! changes.
//!
//! Mirrors the pattern in `golden_files.rs`:
//!   - `#[ignore]`-d `generate_conformance_vectors()` writes every
//!     fixture, its `.json` sidecar, and appends blake3 hashes to
//!     `MANIFEST.json` (preserving the existing 19 codec-compat
//!     entries).
//!   - Always-on validators re-open every fixture, recompute hashes,
//!     and assert the CSR / catalog summary against the sidecar.
//!
//! Regenerate with:
//!     cargo test -p scx-integration-tests --test conformance_vectors \
//!         generate_conformance_vectors -- --ignored

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float32Array, RecordBatch, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use roaring::RoaringBitmap;
use scx_codec::{CodecId, ValueEncoding};
use scx_engine::{
    build_obs_predicate_index_bytes, build_var_predicate_index_bytes, BuildOutcome,
    PredicateIndexBuildOptions,
};
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{BitmapShard, DeletionVectors, FullCatalog, ModalityType, SectionType};
use serde::{Deserialize, Serialize};

const SEED: u64 = 0xDEAD_BEEF_CAFE_1234;

// =========================================================================
// Deterministic PRNG (same as golden_files.rs)
// =========================================================================

struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

// =========================================================================
// Sidecar types
// =========================================================================

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct HeaderSummary {
    format_version: u32,
    n_obs: u64,
    n_vars: u64,
    nnz: u64,
    n_csr_shards: u32,
    n_csc_shards: u32,
    n_modalities: u32,
    has_deletion_vectors: bool,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Clone)]
struct CatalogSummaryEntry {
    section_type: String,
    name: String,
    modality_id: u8,
    length: u64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Default)]
struct ExpectedCsr {
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConformanceSidecar {
    fixture_name: String,
    header: HeaderSummary,
    catalog_summary: Vec<CatalogSummaryEntry>,
    /// Filled only for fixtures with a global X (skipped for cloud
    /// directory fixtures and pure-multimodal fixtures).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected_csr: Option<ExpectedCsr>,
    /// Free-form prose: what this fixture exercises and any
    /// compatibility caveats.
    compatibility_notes: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    algorithm: String,
    files: BTreeMap<String, String>,
}

// =========================================================================
// Paths
// =========================================================================

fn reference_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest).join("../reference_files")
}

fn manifest_path() -> PathBuf {
    reference_dir().join("MANIFEST.json")
}

// =========================================================================
// Shared fixture helpers
// =========================================================================

fn default_header(n_obs: u64, n_vars: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, 0, 16384, 0, 0)
}

fn make_obs(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("conf_cell_{i}")).collect();
    let cell_types: Vec<&str> = (0..n)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(cell_types)),
        ],
    )
    .unwrap()
}

fn make_var(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let total_counts: Vec<u32> = (0..n).map(|i| (i as u32 + 1) * 11).collect();
    let schema = Schema::new(vec![
        Field::new("gene_id", DataType::Utf8, false),
        Field::new("total_counts", DataType::UInt32, false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(UInt32Array::from(total_counts)),
        ],
    )
    .unwrap()
}

/// Generate sparse CSR with ~30% density, u16 values.
fn make_csr(n_rows: usize, n_vars: usize, seed_offset: u64) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut rng = Xorshift64::new(SEED.wrapping_add(seed_offset));
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut data = Vec::new();
    for _ in 0..n_rows {
        let mut row_cols: Vec<u32> = (0..n_vars as u32).filter(|_| rng.next() % 10 < 3).collect();
        row_cols.sort_unstable();
        for &c in &row_cols {
            indices.push(c);
            data.push(((rng.next() % 65000) + 1) as f32);
        }
        indptr.push(indptr.last().unwrap() + row_cols.len() as u64);
    }
    (indptr, indices, data)
}

fn encode_u16(values: &[f32]) -> Vec<u8> {
    ValueEncoding::Uint16.encode_f32_batch(values).unwrap()
}

fn write_minimal_into(writer: &mut ScxWriter, n_obs: usize, n_vars: usize, seed_offset: u64) {
    writer.write_obs(&make_obs(n_obs)).unwrap();
    writer.write_var(&make_var(n_vars)).unwrap();
    let (indptr, indices, data) = make_csr(n_obs, n_vars, seed_offset);
    let values_bytes = encode_u16(&data);
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();
}

// =========================================================================
// Per-fixture generators
// =========================================================================

fn generate_v1_minimal(out: &Path) {
    let header = default_header(40, 25);
    let mut writer = ScxWriter::new(out, header).unwrap();
    write_minimal_into(&mut writer, 40, 25, 1);
    writer.finish().unwrap();
}

fn generate_v1_csc(out: &Path) {
    let n_obs = 40;
    let n_vars = 25;
    let header = default_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(out, header).unwrap();
    writer.write_obs(&make_obs(n_obs)).unwrap();
    writer.write_var(&make_var(n_vars)).unwrap();
    let (indptr, indices, data) = make_csr(n_obs, n_vars, 2);
    let values_bytes = encode_u16(&data);
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    // CSC sidecar: transpose. Simple O(n_vars * n_obs) transpose for
    // the small fixture.
    let mut csc_indptr = vec![0u64];
    let mut csc_indices = Vec::new();
    let mut csc_data: Vec<f32> = Vec::new();
    for col in 0..n_vars as u32 {
        let mut col_rows: Vec<(u32, f32)> = Vec::new();
        for row in 0..n_obs {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            if let Some(pos) = indices[start..end].iter().position(|&c| c == col) {
                col_rows.push((row as u32, data[start + pos]));
            }
        }
        for (r, v) in &col_rows {
            csc_indices.push(*r);
            csc_data.push(*v);
        }
        csc_indptr.push(csc_indptr.last().unwrap() + col_rows.len() as u64);
    }
    let csc_values_bytes = encode_u16(&csc_data);
    writer
        .write_csc_shard(
            &csc_indptr,
            &csc_indices,
            &csc_values_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    writer.finish().unwrap();
}

fn generate_v1_layers_obsm_uns(out: &Path) {
    let n_obs = 30;
    let n_vars = 20;
    let header = default_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(out, header).unwrap();
    write_minimal_into(&mut writer, n_obs, n_vars, 3);

    // Layer shard: a second CSR for the "raw_counts" layer.
    let (l_indptr, l_indices, l_data) = make_csr(n_obs, n_vars, 31);
    let l_values_bytes = encode_u16(&l_data);
    writer
        .write_layer_csr_shard(
            &l_indptr,
            &l_indices,
            &l_values_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
            "raw_counts",
            0,
        )
        .unwrap();

    // obsm: 2-D PCA-like embedding.
    let pc1: Vec<f32> = (0..n_obs).map(|i| i as f32 * 0.05).collect();
    let pc2: Vec<f32> = (0..n_obs).map(|i| (n_obs - i) as f32 * -0.03).collect();
    let obsm_schema = Schema::new(vec![
        Field::new("PC1", DataType::Float32, false),
        Field::new("PC2", DataType::Float32, false),
    ]);
    let obsm = RecordBatch::try_new(
        Arc::new(obsm_schema),
        vec![
            Arc::new(Float32Array::from(pc1)),
            Arc::new(Float32Array::from(pc2)),
        ],
    )
    .unwrap();
    writer.write_obsm("X_pca", &obsm).unwrap();

    // uns: tagged-JSON blob.
    let uns = serde_json::json!({
        "description": "Phase 9 conformance fixture",
        "seed": SEED,
        "schema_version": "0.1",
    });
    writer.write_uns(&uns).unwrap();

    writer.finish().unwrap();
}

fn generate_v1_predicate_indexes(out: &Path) {
    let n_obs = 40;
    let n_vars = 25;
    let header = default_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(out, header).unwrap();
    let obs = make_obs(n_obs);
    let var = make_var(n_vars);
    writer.write_obs(&obs).unwrap();
    writer.write_var(&var).unwrap();
    let (indptr, indices, data) = make_csr(n_obs, n_vars, 4);
    let values_bytes = encode_u16(&data);
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    // Predicate index on obs.cell_type (categorical).
    let mut obs_outcomes: Vec<BuildOutcome> = Vec::new();
    let mut obs_indexed: Vec<String> = Vec::new();
    let obs_bytes = build_obs_predicate_index_bytes(
        &obs,
        &[(0, n_obs as u64)],
        &PredicateIndexBuildOptions {
            forced_columns: vec!["cell_type".to_string()],
            preset_columns: vec![],
            auto_threshold: 1000,
            high_cardinality_threshold: 100_000,
        },
        &mut obs_outcomes,
        &mut obs_indexed,
    )
    .unwrap();
    if let Some(bytes) = obs_bytes {
        writer.write_obs_predicate_index(&bytes).unwrap();
    }

    // Predicate index on var.total_counts (numeric).
    let mut var_outcomes: Vec<BuildOutcome> = Vec::new();
    let mut var_indexed: Vec<String> = Vec::new();
    let var_bytes = build_var_predicate_index_bytes(
        &var,
        &[(0, n_vars as u64)],
        &PredicateIndexBuildOptions {
            forced_columns: vec!["total_counts".to_string()],
            preset_columns: vec![],
            auto_threshold: 1000,
            high_cardinality_threshold: 100_000,
        },
        &mut var_outcomes,
        &mut var_indexed,
    )
    .unwrap();
    if let Some(bytes) = var_bytes {
        writer.write_var_predicate_index(&bytes).unwrap();
    }

    writer.finish().unwrap();
}

fn generate_v1_deletion_vectors(out: &Path) {
    // Seed the file with a single shard, then write a deletion vector
    // bitmap directly. Goes through ScxWriter rather than scx-ops so
    // the manifest chain stays single-catalog (deterministic across
    // platforms, no scx-ops shard layout depending on
    // shard_target_rows).
    let n_obs = 40;
    let n_vars = 25;
    let header = default_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(out, header).unwrap();
    write_minimal_into(&mut writer, n_obs, n_vars, 5);

    let mut bitmap = RoaringBitmap::new();
    // Delete a deterministic spread of rows.
    bitmap.insert(2);
    bitmap.insert(5);
    bitmap.insert(11);
    bitmap.insert(13);
    bitmap.insert(28);
    let mut dv = DeletionVectors::new();
    dv.insert(0, bitmap);
    writer.write_deletion_vectors(&dv).unwrap();

    writer.finish().unwrap();
}

fn generate_v2_multimodal_citeseq(out: &Path) {
    let n_obs = 30;
    let rna_n_vars = 25;
    let adt_n_vars = 8;
    // Header carries the union obs axis and a placeholder n_vars (=0
    // for multimodal files; per-modality n_vars lives in the modality
    // table).
    let header = default_header(n_obs as u64, 0);
    let mut writer = ScxWriter::new(out, header).unwrap();
    writer.write_obs(&make_obs(n_obs)).unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            false,
        )
        .unwrap();
    let adt_id = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            false,
        )
        .unwrap();

    // Per-modality var
    writer.write_var_for(rna_id, &make_var(rna_n_vars)).unwrap();
    writer.write_var_for(adt_id, &make_var(adt_n_vars)).unwrap();

    // Per-modality CSR
    let (r_indptr, r_indices, r_data) = make_csr(n_obs, rna_n_vars, 61);
    writer
        .write_csr_shard_for(
            rna_id,
            &r_indptr,
            &r_indices,
            &encode_u16(&r_data),
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    let (a_indptr, a_indices, a_data) = make_csr(n_obs, adt_n_vars, 62);
    writer
        .write_csr_shard_for(
            adt_id,
            &a_indptr,
            &a_indices,
            &encode_u16(&a_data),
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    writer.finish().unwrap();
}

fn generate_v2_multimodal_partial_csc(out: &Path) {
    let n_obs = 30;
    let rna_n_vars = 25;
    let adt_n_vars = 8;
    let header = default_header(n_obs as u64, 0);
    let mut writer = ScxWriter::new(out, header).unwrap();
    writer.write_obs(&make_obs(n_obs)).unwrap();

    let rna_id = writer
        .add_modality(
            "rna",
            ModalityType::Rna,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            false, // RNA: no CSC sidecar
        )
        .unwrap();
    let adt_id = writer
        .add_modality(
            "adt",
            ModalityType::Protein,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            true, // ADT: HAS_CSC
        )
        .unwrap();

    writer.write_var_for(rna_id, &make_var(rna_n_vars)).unwrap();
    writer.write_var_for(adt_id, &make_var(adt_n_vars)).unwrap();

    let (r_indptr, r_indices, r_data) = make_csr(n_obs, rna_n_vars, 71);
    writer
        .write_csr_shard_for(
            rna_id,
            &r_indptr,
            &r_indices,
            &encode_u16(&r_data),
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    let (a_indptr, a_indices, a_data) = make_csr(n_obs, adt_n_vars, 72);
    writer
        .write_csr_shard_for(
            adt_id,
            &a_indptr,
            &a_indices,
            &encode_u16(&a_data),
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    // Build ADT CSC by transposing.
    let mut csc_indptr = vec![0u64];
    let mut csc_indices = Vec::new();
    let mut csc_data: Vec<f32> = Vec::new();
    for col in 0..adt_n_vars as u32 {
        let mut col_rows: Vec<(u32, f32)> = Vec::new();
        for row in 0..n_obs {
            let start = a_indptr[row] as usize;
            let end = a_indptr[row + 1] as usize;
            if let Some(pos) = a_indices[start..end].iter().position(|&c| c == col) {
                col_rows.push((row as u32, a_data[start + pos]));
            }
        }
        for (r, v) in &col_rows {
            csc_indices.push(*r);
            csc_data.push(*v);
        }
        csc_indptr.push(csc_indptr.last().unwrap() + col_rows.len() as u64);
    }
    writer
        .write_csc_shard_for(
            adt_id,
            &csc_indptr,
            &csc_indices,
            &encode_u16(&csc_data),
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    writer.finish().unwrap();
}

fn generate_v2_bitmap(out: &Path) {
    let n_obs = 40;
    let n_vars = 25;
    let header = default_header(n_obs as u64, n_vars as u64);
    let mut writer = ScxWriter::new(out, header).unwrap();
    writer.write_obs(&make_obs(n_obs)).unwrap();
    writer.write_var(&make_var(n_vars)).unwrap();
    let (indptr, indices, data) = make_csr(n_obs, n_vars, 6);
    let values_bytes = encode_u16(&data);
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values_bytes,
            CodecId::Scx1,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();

    // Derive the bitmap directly from the shard.
    let bitmap_shard =
        BitmapShard::build_from_csr(0, n_obs as u32, n_vars as u32, &indptr, &indices);
    writer.write_bitmap_shard(&bitmap_shard).unwrap();

    writer.finish().unwrap();
}

// =========================================================================
// Catalog summary extraction
// =========================================================================

fn extract_catalog_summary(path: &Path) -> Vec<CatalogSummaryEntry> {
    use scx_format_io::catalog::FullCatalog;
    use scx_format_io::header::HEADER_SIZE;
    let bytes = fs::read(path).unwrap();
    let header = FileHeader::read_from(&mut std::io::Cursor::new(&bytes[..HEADER_SIZE])).unwrap();
    let fc_off = header.full_catalog_offset as usize;
    let fc_len = header.full_catalog_length as usize;
    let cat = FullCatalog::read_from(
        &mut std::io::Cursor::new(&bytes[fc_off..fc_off + fc_len]),
        fc_len,
        true,
    )
    .unwrap();
    cat.entries
        .iter()
        .map(|e| CatalogSummaryEntry {
            section_type: format!("{:?}", e.section_type),
            name: e.name.clone(),
            modality_id: e.modality_id,
            length: e.length,
        })
        .collect()
}

fn extract_header_summary(path: &Path) -> HeaderSummary {
    let reader = ScxReader::open(path).unwrap();
    let h = reader.header();
    HeaderSummary {
        format_version: h.format_version as u32,
        n_obs: h.n_obs,
        n_vars: h.n_vars,
        nnz: h.nnz,
        n_csr_shards: h.n_csr_shards,
        n_csc_shards: h.n_csc_shards,
        n_modalities: h.n_modalities,
        has_deletion_vectors: h.has_deletion_vectors(),
    }
}

fn extract_expected_csr(path: &Path) -> Option<ExpectedCsr> {
    let reader = ScxReader::open(path).unwrap();
    if reader.header().n_csr_shards == 0 {
        return None;
    }
    let csr = reader.read_all_csr_shards().ok()?;
    Some(ExpectedCsr {
        indptr: csr.indptr,
        indices: csr.indices,
        data: csr.data,
    })
}

// =========================================================================
// Hashing helpers
// =========================================================================

fn hash_file(path: &Path) -> String {
    let bytes = fs::read(path).unwrap();
    blake3::hash(&bytes).to_hex().to_string()
}

/// Recursively hash every file in `dir`, returning `(rel_path, hash)` tuples.
fn hash_directory(dir: &Path, prefix: &Path) -> Vec<(String, String)> {
    let mut entries: Vec<_> = fs::read_dir(dir).unwrap().filter_map(|e| e.ok()).collect();
    entries.sort_by_key(|e| e.file_name());
    let mut hashes = Vec::new();
    for entry in entries {
        let path = entry.path();
        let rel = path
            .strip_prefix(prefix)
            .unwrap()
            .to_string_lossy()
            .to_string();
        let ft = entry.file_type().unwrap();
        if ft.is_dir() {
            hashes.extend(hash_directory(&path, prefix));
        } else if ft.is_file() {
            let h = hash_file(&path);
            hashes.push((rel, h));
        }
    }
    hashes
}

// =========================================================================
// Generator (run with --ignored)
// =========================================================================

#[derive(Debug, Clone, Copy)]
enum FixtureKind {
    File,
    Directory,
}

struct Fixture {
    name: &'static str,
    rel_path: &'static str,
    kind: FixtureKind,
    notes: &'static str,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "v1_minimal",
        rel_path: "v1_minimal.scx",
        kind: FixtureKind::File,
        notes: "Smallest valid v1 file: obs + var + 1 CSR shard. No optional sections.",
    },
    Fixture {
        name: "v1_csc",
        rel_path: "v1_csc.scx",
        kind: FixtureKind::File,
        notes: "CSR + CSC sidecar (gene-major) for the same matrix.",
    },
    Fixture {
        name: "v1_layers_obsm_uns",
        rel_path: "v1_layers_obsm_uns.scx",
        kind: FixtureKind::File,
        notes: "v1 base + LayerCsrShard (raw_counts) + ObsmEmbedding (X_pca) + UnsBlob.",
    },
    Fixture {
        name: "v1_predicate_indexes",
        rel_path: "v1_predicate_indexes.scx",
        kind: FixtureKind::File,
        notes: "obs categorical predicate index (cell_type) + var numeric predicate index (total_counts).",
    },
    Fixture {
        name: "v1_deletion_vectors",
        rel_path: "v1_deletion_vectors.scx",
        kind: FixtureKind::File,
        notes: "v1 base + DeletionVectors section marking 5 logical row deletions.",
    },
    Fixture {
        name: "v2_multimodal_citeseq",
        rel_path: "v2_multimodal_citeseq.scx",
        kind: FixtureKind::File,
        notes: "v2 multimodal: RNA (Rna) + ADT (Protein) modalities, per-modality var + CSR.",
    },
    Fixture {
        name: "v2_multimodal_partial_csc",
        rel_path: "v2_multimodal_partial_csc.scx",
        kind: FixtureKind::File,
        notes: "v2 multimodal where only the ADT modality carries a CSC sidecar (Phase 6 prerequisite).",
    },
    Fixture {
        name: "v2_bitmap",
        rel_path: "v2_bitmap.scx",
        kind: FixtureKind::File,
        notes: "v1 base + BitmapShard (gene -> rows roaring) built from the CSR shard.",
    },
    Fixture {
        name: "cloud_optimized_reference",
        rel_path: "cloud_optimized_reference.scx",
        kind: FixtureKind::File,
        notes: "v1_minimal passed through scx_cloud::cloud_optimize: includes a front catalog.",
    },
    Fixture {
        name: "cloud_exploded_reference",
        rel_path: "cloud_exploded_reference.scxd",
        kind: FixtureKind::Directory,
        notes: "v1_minimal passed through scx_cloud::explode: one file per section, byte-identical.",
    },
];

fn generate_file_fixture(name: &str, path: &Path, dir: &Path) {
    match name {
        "v1_minimal" => generate_v1_minimal(path),
        "v1_csc" => generate_v1_csc(path),
        "v1_layers_obsm_uns" => generate_v1_layers_obsm_uns(path),
        "v1_predicate_indexes" => generate_v1_predicate_indexes(path),
        "v1_deletion_vectors" => generate_v1_deletion_vectors(path),
        "v2_multimodal_citeseq" => generate_v2_multimodal_citeseq(path),
        "v2_multimodal_partial_csc" => generate_v2_multimodal_partial_csc(path),
        "v2_bitmap" => generate_v2_bitmap(path),
        "cloud_optimized_reference" => {
            let base = dir.join("v1_minimal.scx");
            assert!(base.exists(), "v1_minimal.scx must be generated first");
            scx_cloud::cloud_optimize::cloud_optimize(&base, path).unwrap();
        }
        _ => panic!("unknown file fixture: {name}"),
    }
}

fn generate_directory_fixture(name: &str, path: &Path, dir: &Path) {
    if path.exists() {
        let _ = fs::remove_dir_all(path);
    }
    match name {
        "cloud_exploded_reference" => {
            let base = dir.join("v1_minimal.scx");
            assert!(base.exists(), "v1_minimal.scx must be generated first");
            scx_cloud::explode::explode(&base, path).unwrap();
        }
        _ => panic!("unknown directory fixture: {name}"),
    }
}

fn write_sidecar(fixture: &Fixture, dir: &Path) {
    let scx_path = dir.join(fixture.rel_path);
    // Cloud directory fixtures: load the catalog from `_catalog.bin`
    // and the header from `_header.bin`.
    let (header_summary, catalog_summary, expected_csr) = match fixture.kind {
        FixtureKind::File => {
            let h = extract_header_summary(&scx_path);
            let c = extract_catalog_summary(&scx_path);
            let csr = extract_expected_csr(&scx_path);
            (h, c, csr)
        }
        FixtureKind::Directory => {
            use scx_format_io::catalog::FullCatalog;
            use scx_format_io::header::HEADER_SIZE;
            let header_bytes = fs::read(scx_path.join("_header.bin")).unwrap();
            let header =
                FileHeader::read_from(&mut std::io::Cursor::new(&header_bytes[..HEADER_SIZE]))
                    .unwrap();
            let catalog_bytes = fs::read(scx_path.join("_catalog.bin")).unwrap();
            let cat = FullCatalog::read_from(
                &mut std::io::Cursor::new(&catalog_bytes),
                catalog_bytes.len(),
                true,
            )
            .unwrap();
            let h = HeaderSummary {
                format_version: header.format_version as u32,
                n_obs: header.n_obs,
                n_vars: header.n_vars,
                nnz: header.nnz,
                n_csr_shards: header.n_csr_shards,
                n_csc_shards: header.n_csc_shards,
                n_modalities: header.n_modalities,
                has_deletion_vectors: header.has_deletion_vectors(),
            };
            let c: Vec<CatalogSummaryEntry> = cat
                .entries
                .iter()
                .map(|e| CatalogSummaryEntry {
                    section_type: format!("{:?}", e.section_type),
                    name: e.name.clone(),
                    modality_id: e.modality_id,
                    length: e.length,
                })
                .collect();
            (h, c, None)
        }
    };

    let sidecar = ConformanceSidecar {
        fixture_name: fixture.name.to_string(),
        header: header_summary,
        catalog_summary,
        expected_csr,
        compatibility_notes: fixture.notes.to_string(),
    };
    let json = serde_json::to_string_pretty(&sidecar).unwrap();
    let sidecar_path = dir.join(format!("{}.json", fixture.name));
    fs::write(&sidecar_path, json).unwrap();
}

#[test]
#[ignore]
fn generate_conformance_vectors() {
    let dir = reference_dir();
    fs::create_dir_all(&dir).unwrap();

    // Generate file fixtures (order matters: v1_minimal must come first
    // because cloud_* derive from it).
    for fixture in FIXTURES {
        let target = dir.join(fixture.rel_path);
        match fixture.kind {
            FixtureKind::File => {
                if target.exists() {
                    let _ = fs::remove_file(&target);
                }
                generate_file_fixture(fixture.name, &target, &dir);
            }
            FixtureKind::Directory => {
                generate_directory_fixture(fixture.name, &target, &dir);
            }
        }
        write_sidecar(fixture, &dir);
        eprintln!("Generated {} ({:?})", fixture.rel_path, fixture.kind);
    }

    // Merge new hashes into MANIFEST.json without disturbing existing
    // entries (the 19 golden_*.scx hashes already there).
    let mp = manifest_path();
    let mut existing: Manifest = if mp.exists() {
        serde_json::from_str(&fs::read_to_string(&mp).unwrap()).unwrap()
    } else {
        Manifest {
            algorithm: "blake3".to_string(),
            files: BTreeMap::new(),
        }
    };
    for fixture in FIXTURES {
        let target = dir.join(fixture.rel_path);
        match fixture.kind {
            FixtureKind::File => {
                let h = hash_file(&target);
                existing.files.insert(fixture.rel_path.to_string(), h);
            }
            FixtureKind::Directory => {
                // Drop any previously recorded entries under this prefix
                // so renames don't leave orphan hashes.
                let prefix = format!("{}/", fixture.rel_path);
                existing.files.retain(|k, _| !k.starts_with(&prefix));
                for (rel, h) in hash_directory(&target, &target) {
                    existing.files.insert(format!("{prefix}{rel}"), h);
                }
            }
        }
    }
    let manifest_json = serde_json::to_string_pretty(&existing).unwrap();
    fs::write(&mp, manifest_json).unwrap();

    eprintln!(
        "Wrote MANIFEST.json with {} total entries",
        existing.files.len()
    );
}

// =========================================================================
// Always-on validators
// =========================================================================

fn load_sidecar(name: &str) -> Option<ConformanceSidecar> {
    let path = reference_dir().join(format!("{name}.json"));
    if !path.exists() {
        return None;
    }
    let content = fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Skip body if the fixture sidecar is missing (`--ignored` generator
/// hasn't run yet).
macro_rules! require_sidecar {
    ($fixture:expr) => {
        match load_sidecar($fixture.name) {
            Some(s) => s,
            None => {
                eprintln!(
                    "Skipping conformance check for {}: sidecar missing. \
                     Run: cargo test -p scx-integration-tests --test conformance_vectors \
                     generate_conformance_vectors -- --ignored",
                    $fixture.name
                );
                return;
            }
        }
    };
}

#[test]
fn test_conformance_files_reader_opens() {
    for fixture in FIXTURES {
        if !matches!(fixture.kind, FixtureKind::File) {
            continue;
        }
        let path = reference_dir().join(fixture.rel_path);
        if !path.exists() {
            continue;
        }
        let reader = ScxReader::open(&path)
            .unwrap_or_else(|e| panic!("failed to open {}: {e}", fixture.rel_path));
        let h = reader.header();
        // Sanity: format_version is current.
        assert_eq!(
            h.format_version, CURRENT_FORMAT_VERSION,
            "{} format_version differs",
            fixture.rel_path
        );
    }
}

#[test]
fn test_conformance_files_validate() {
    for fixture in FIXTURES {
        if !matches!(fixture.kind, FixtureKind::File) {
            continue;
        }
        let path = reference_dir().join(fixture.rel_path);
        if !path.exists() {
            continue;
        }
        let reader = ScxReader::open(&path).unwrap();
        let results = reader.validate().unwrap();
        for (section, ok) in &results {
            assert!(
                ok,
                "{}: validate failed for section '{section}'",
                fixture.rel_path
            );
        }
    }
}

#[test]
fn test_conformance_files_header_summary() {
    for fixture in FIXTURES {
        let sidecar = require_sidecar!(fixture);
        let path = reference_dir().join(fixture.rel_path);
        if !path.exists() {
            continue;
        }
        let actual = match fixture.kind {
            FixtureKind::File => extract_header_summary(&path),
            FixtureKind::Directory => {
                use scx_format_io::header::HEADER_SIZE;
                let header_bytes = fs::read(path.join("_header.bin")).unwrap();
                let header =
                    FileHeader::read_from(&mut std::io::Cursor::new(&header_bytes[..HEADER_SIZE]))
                        .unwrap();
                HeaderSummary {
                    format_version: header.format_version as u32,
                    n_obs: header.n_obs,
                    n_vars: header.n_vars,
                    nnz: header.nnz,
                    n_csr_shards: header.n_csr_shards,
                    n_csc_shards: header.n_csc_shards,
                    n_modalities: header.n_modalities,
                    has_deletion_vectors: header.has_deletion_vectors(),
                }
            }
        };
        assert_eq!(
            actual, sidecar.header,
            "{}: header summary diverged from sidecar",
            fixture.rel_path
        );
    }
}

#[test]
fn test_conformance_files_csr_match() {
    for fixture in FIXTURES {
        let sidecar = require_sidecar!(fixture);
        let path = reference_dir().join(fixture.rel_path);
        if !path.exists() {
            continue;
        }
        if !matches!(fixture.kind, FixtureKind::File) {
            continue;
        }
        let Some(expected) = &sidecar.expected_csr else {
            continue;
        };
        let actual = extract_expected_csr(&path).unwrap();
        assert_eq!(
            actual.indptr, expected.indptr,
            "{}: CSR indptr mismatch",
            fixture.rel_path
        );
        assert_eq!(
            actual.indices, expected.indices,
            "{}: CSR indices mismatch",
            fixture.rel_path
        );
        assert_eq!(
            actual.data, expected.data,
            "{}: CSR data mismatch",
            fixture.rel_path
        );
    }
}

#[test]
fn test_conformance_files_manifest_hashes() {
    let mp = manifest_path();
    if !mp.exists() {
        eprintln!("MANIFEST.json missing; skipping.");
        return;
    }
    let manifest: Manifest = serde_json::from_str(&fs::read_to_string(&mp).unwrap()).unwrap();
    let dir = reference_dir();

    // If the entire reference corpus is absent (fresh checkout before
    // generation), skip the whole test rather than running zero
    // assertions. Otherwise every fixture in FIXTURES must be present
    // AND must have a manifest entry — silent-skip would defeat the
    // frozen-reference guarantee.
    let any_present = FIXTURES.iter().any(|f| dir.join(f.rel_path).exists());
    if !any_present {
        eprintln!("no conformance fixtures present; skipping (run generate_conformance_vectors).");
        return;
    }

    for fixture in FIXTURES {
        let target = dir.join(fixture.rel_path);
        assert!(
            target.exists(),
            "{}: fixture file missing — regenerate via `cargo test -p scx-integration-tests \
             --test conformance_vectors generate_conformance_vectors -- --ignored`",
            fixture.rel_path
        );
        match fixture.kind {
            FixtureKind::File => {
                let expected_hash = manifest.files.get(fixture.rel_path).unwrap_or_else(|| {
                    panic!(
                        "{}: missing from MANIFEST.json — regenerate the corpus",
                        fixture.rel_path
                    )
                });
                let actual_hash = hash_file(&target);
                assert_eq!(
                    &actual_hash, expected_hash,
                    "{}: blake3 hash diverged from MANIFEST.json",
                    fixture.rel_path
                );
            }
            FixtureKind::Directory => {
                for (rel, actual_hash) in hash_directory(&target, &target) {
                    let key = format!("{}/{rel}", fixture.rel_path);
                    let expected_hash = manifest.files.get(&key).unwrap_or_else(|| {
                        panic!("{key}: missing from MANIFEST.json — regenerate the corpus")
                    });
                    assert_eq!(
                        &actual_hash, expected_hash,
                        "{}: blake3 hash diverged from MANIFEST.json",
                        key
                    );
                }
            }
        }
    }
}

#[test]
fn test_conformance_files_catalog_summary() {
    for fixture in FIXTURES {
        let sidecar = require_sidecar!(fixture);
        let path = reference_dir().join(fixture.rel_path);
        if !path.exists() {
            continue;
        }
        let actual = match fixture.kind {
            FixtureKind::File => extract_catalog_summary(&path),
            FixtureKind::Directory => {
                use scx_format_io::catalog::FullCatalog;
                let catalog_bytes = fs::read(path.join("_catalog.bin")).unwrap();
                let cat = FullCatalog::read_from(
                    &mut std::io::Cursor::new(&catalog_bytes),
                    catalog_bytes.len(),
                    true,
                )
                .unwrap();
                cat.entries
                    .iter()
                    .map(|e| CatalogSummaryEntry {
                        section_type: format!("{:?}", e.section_type),
                        name: e.name.clone(),
                        modality_id: e.modality_id,
                        length: e.length,
                    })
                    .collect()
            }
        };
        assert_eq!(
            actual, sidecar.catalog_summary,
            "{}: catalog summary diverged from sidecar",
            fixture.rel_path
        );
    }
}

/// Unknown future section-type IDs must be skipped with a warning
/// rather than failing the read. This is the catalog-level analogue of
/// `golden_files.rs::test_unknown_codec_rejected_gracefully`: rewrite
/// one catalog entry's `section_type` byte to 254 (an unused
/// discriminant — `SectionType::from_u8(254)` returns `None`), recompute
/// the catalog BLAKE3 checksum, and assert the reader opens the file
/// (per the warn-and-continue path at
/// `scx-format/src/catalog.rs::FullCatalog::read_from`).
#[test]
fn test_unknown_future_section_type_skipped() {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("unknown_section_type.scx");
    generate_v1_minimal(&path);

    let n_obs_expected = 40u64;
    let n_vars_expected = 25u64;

    let mut bytes = fs::read(&path).unwrap();
    let header = FileHeader::read_from(&mut std::io::Cursor::new(&bytes[..256])).unwrap();
    let fc_off = header.full_catalog_offset as usize;
    let fc_len = header.full_catalog_length as usize;
    let fc_end = fc_off + fc_len;

    // Parse the catalog (skip checksum since we're about to mutate).
    let catalog = FullCatalog::read_from(
        &mut std::io::Cursor::new(&bytes[fc_off..fc_end]),
        fc_len,
        false,
    )
    .expect("catalog must parse");

    // Target the CsrShard entry: rebranding it leaves obs/var (the
    // sections the reader actually needs to report shape) untouched,
    // and `n_obs`/`n_vars` come from the file header, not the catalog.
    let target_idx = catalog
        .entries
        .iter()
        .position(|e| e.section_type == SectionType::CsrShard)
        .expect("v1_minimal must contain a CsrShard entry");

    // Walk the v2 payload to locate the target entry's section_type
    // byte. Catalog payload layout (see `FullCatalog::write_to`):
    //   header (30 bytes): u16 catalog_version + u64 manifest_sequence
    //     + u64 prev_catalog_offset + u64 n_obs + u32 n_entries
    //   per v2 entry: u16 name_len + name + u64 offset + u64 length
    //     + u8 section_type + 32 checksum + u8 modality_id
    //     + u16 stats_len + stats
    let payload_end = fc_end - 32; // strip trailing BLAKE3 checksum
    let abs_section_type_offset = {
        let payload = &bytes[fc_off..payload_end];
        let mut cursor: usize = 30; // skip catalog header
        let mut found: Option<usize> = None;
        for entry_idx in 0..catalog.entries.len() {
            let name_len =
                u16::from_le_bytes(payload[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2 + name_len;
            cursor += 8 + 8; // offset + length
            let section_type_pos = cursor;
            cursor += 1; // section_type
            cursor += 32; // checksum
            cursor += 1; // modality_id
            let stats_len =
                u16::from_le_bytes(payload[cursor..cursor + 2].try_into().unwrap()) as usize;
            cursor += 2 + stats_len;

            if entry_idx == target_idx {
                found = Some(fc_off + section_type_pos);
                break;
            }
        }
        found.expect("walked target entry in catalog payload")
    };

    // Sanity: the byte we're about to patch must currently hold the
    // CsrShard discriminant. If this fails the layout walk drifted.
    assert_eq!(
        bytes[abs_section_type_offset],
        SectionType::CsrShard as u8,
        "section_type byte offset walk landed on the wrong byte"
    );

    bytes[abs_section_type_offset] = 254;

    // Recompute the catalog's trailing BLAKE3 checksum over the patched
    // payload — the reader rejects mismatched checksums before it ever
    // reaches the per-entry SectionType match.
    let new_checksum = blake3::hash(&bytes[fc_off..payload_end]);
    bytes[payload_end..fc_end].copy_from_slice(new_checksum.as_bytes());

    let patched_path = tmp.path().join("unknown_section_type_patched.scx");
    fs::write(&patched_path, &bytes).unwrap();

    // The reader must open the file; the unknown discriminant entry is
    // logged and skipped per the warn-and-continue path.
    let reader = ScxReader::open(&patched_path)
        .expect("reader must skip unknown section types (warn-and-continue invariant)");

    // obs/var sections are intact, and the file header is unchanged.
    assert_eq!(reader.n_obs(), n_obs_expected);
    assert_eq!(reader.n_vars(), n_vars_expected);
}
