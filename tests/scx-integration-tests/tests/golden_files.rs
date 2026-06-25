//! Golden file regression tests for SCX format.
//!
//! - The `generate_golden_files` test (#[ignore]) generates 11 golden SCX files,
//!   JSON sidecars, and a BLAKE3 manifest in `tests/reference_files/`.
//! - The remaining tests validate those golden files on every `cargo test` run.
//!
//! Valid codec × encoding combinations (11 total):
//!   None  × {u8, u16, u32, f32}  = 4
//!   Scx1  × {u8, u16, u32}       = 3  (Scx1 rejects float encodings)
//!   Zstd  × {u8, u16, u32, f32}  = 4
//!
//! Scx2 is intentionally NOT in the pinned golden set (its end-to-end
//! write→read path is covered by `scx2_writer_reader_round_trip` below, without
//! re-pinning bytes — the committed goldens predate a writer format-version bump
//! and must not be regenerated as a side effect of this change).

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{AsArray, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::header::FileHeader;
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const N_OBS: usize = 100;
const N_VARS: usize = 50;
const SEED: u64 = 0xDEAD_BEEF_CAFE_1234;

/// All valid (codec, encoding) combinations.
fn golden_combinations() -> Vec<(CodecId, ValueEncoding)> {
    vec![
        (CodecId::None, ValueEncoding::Uint8),
        (CodecId::None, ValueEncoding::Uint16),
        (CodecId::None, ValueEncoding::Uint32),
        (CodecId::None, ValueEncoding::Float32),
        (CodecId::Scx1, ValueEncoding::Uint8),
        (CodecId::Scx1, ValueEncoding::Uint16),
        (CodecId::Scx1, ValueEncoding::Uint32),
        // Scx1 × Float32 is invalid (FloatWithScx1 error)
        (CodecId::Zstd, ValueEncoding::Uint8),
        (CodecId::Zstd, ValueEncoding::Uint16),
        (CodecId::Zstd, ValueEncoding::Uint32),
        (CodecId::Zstd, ValueEncoding::Float32),
        // Sprint 2: LZ4+shuffle (Phase 2D) — works with any value encoding
        (CodecId::Lz4Shuffle, ValueEncoding::Uint8),
        (CodecId::Lz4Shuffle, ValueEncoding::Uint16),
        (CodecId::Lz4Shuffle, ValueEncoding::Uint32),
        (CodecId::Lz4Shuffle, ValueEncoding::Float32),
        // Sprint 3: Pcodec (Phase 3C) — works with any value encoding
        (CodecId::Pcodec, ValueEncoding::Uint8),
        (CodecId::Pcodec, ValueEncoding::Uint16),
        (CodecId::Pcodec, ValueEncoding::Uint32),
        (CodecId::Pcodec, ValueEncoding::Float32),
    ]
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (no rand dependency, platform-independent)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Sidecar types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
struct GoldenSidecar {
    codec: String,
    value_encoding: String,
    n_obs: usize,
    n_vars: usize,
    nnz: usize,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f32>,
    obs_cell_ids: Vec<String>,
    obs_cell_types: Vec<String>,
    var_gene_ids: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    algorithm: String,
    files: BTreeMap<String, String>,
}

// ---------------------------------------------------------------------------
// Naming helpers
// ---------------------------------------------------------------------------

fn codec_name(c: CodecId) -> &'static str {
    match c {
        CodecId::None => "none",
        CodecId::Scx1 => "scx1",
        CodecId::Scx2 => "scx2",
        CodecId::Zstd => "zstd",
        CodecId::Lz4Shuffle => "lz4shuffle",
        CodecId::Pcodec => "pcodec",
    }
}

fn encoding_name(e: ValueEncoding) -> &'static str {
    match e {
        ValueEncoding::Uint8 => "u8",
        ValueEncoding::Uint16 => "u16",
        ValueEncoding::Uint32 => "u32",
        ValueEncoding::Float32 => "f32",
        ValueEncoding::Float16 => "f16",
    }
}

fn golden_basename(codec: CodecId, encoding: ValueEncoding) -> String {
    format!("golden_{}_{}", codec_name(codec), encoding_name(encoding))
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/reference_files")
}

// ---------------------------------------------------------------------------
// Deterministic matrix generation
// ---------------------------------------------------------------------------

/// Generate a deterministic sparse CSR matrix.
///
/// Returns (indptr as u64, indices as u32, data as f32).
/// The sparsity pattern is identical across all encodings (same seed),
/// but values are clamped to the encoding's valid range.
fn generate_matrix(encoding: ValueEncoding) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut rng = Xorshift64::new(SEED);
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut data = Vec::new();

    for _row in 0..N_OBS {
        let mut row_nnz = 0u64;
        for col in 0..N_VARS {
            let r = rng.next();
            // ~30% density
            if r % 10 < 3 {
                indices.push(col as u32);
                let val = match encoding {
                    ValueEncoding::Uint8 => ((r >> 16) % 255 + 1) as f32,
                    ValueEncoding::Uint16 => ((r >> 16) % 60000 + 1) as f32,
                    ValueEncoding::Uint32 => ((r >> 16) % 100000 + 1) as f32,
                    ValueEncoding::Float32 => {
                        // Non-integer floats representable as exact f32
                        let raw = ((r >> 16) & 0xFFFF) as f32;
                        (raw / 10.0 + 0.1_f32).round() / 10.0 * 10.0
                        // Simpler: just use raw / 10.0 + 1.0
                    }
                    ValueEncoding::Float16 => ((r >> 16) % 255 + 1) as f32,
                };
                // For Float32, use a cleaner generation that avoids precision issues
                let val = if encoding == ValueEncoding::Float32 {
                    let raw = ((r >> 16) & 0xFFFF) as u32;
                    (raw % 10000 + 1) as f32 / 10.0
                } else {
                    val
                };
                data.push(val);
                row_nnz += 1;
            }
        }
        indptr.push(indptr.last().unwrap() + row_nnz);
    }

    (indptr, indices, data)
}

// ---------------------------------------------------------------------------
// Obs / Var helpers (matching integration_lifecycle.rs pattern)
// ---------------------------------------------------------------------------

fn make_obs() -> RecordBatch {
    let schema = Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("cell_type", DataType::Utf8, true),
    ]);
    let ids: Vec<String> = (0..N_OBS).map(|i| format!("golden_cell_{i}")).collect();
    let types: Vec<&str> = (0..N_OBS)
        .map(|i| match i % 3 {
            0 => "T cell",
            1 => "B cell",
            _ => "NK cell",
        })
        .collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(StringArray::from(
                ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(types)),
        ],
    )
    .unwrap()
}

fn make_var() -> RecordBatch {
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    let ids: Vec<String> = (0..N_VARS).map(|i| format!("gene_{i}")).collect();
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn make_header() -> FileHeader {
    // N_VARS=50 < 65535, so index_dtype (last arg) = 0 (u16 on-disk indices).
    FileHeader::new_single_modality(N_OBS as u64, N_VARS as u64, 0, 10_000, 0, 0)
}

/// ISC-24/25: end-to-end ScxWriter → ScxReader round-trip for the `scx2` codec
/// across integer encodings, without pinning bytes. Confirms the shard header
/// carries `codec_id = 5` (the reader resolves it via `from_u8` and decodes the
/// Rice-gap index stream) and that scx2 decodes identically to the source CSR
/// and to scx1 (which shares everything but the index stream).
#[test]
fn scx2_writer_reader_round_trip() {
    let tmp = std::env::temp_dir().join(format!("scx2_rt_{}", std::process::id()));
    fs::create_dir_all(&tmp).unwrap();
    let obs = make_obs();
    let var = make_var();

    for encoding in [
        ValueEncoding::Uint8,
        ValueEncoding::Uint16,
        ValueEncoding::Uint32,
    ] {
        let (indptr, indices, data_f32) = generate_matrix(encoding);
        let values_bytes = encoding.encode_f32_batch(&data_f32).unwrap();
        let exp_indptr: Vec<i64> = indptr.iter().map(|&v| v as i64).collect();
        let exp_indices: Vec<i32> = indices.iter().map(|&v| v as i32).collect();

        let mut decoded = Vec::new();
        for codec in [CodecId::Scx1, CodecId::Scx2] {
            let path = tmp.join(format!(
                "rt_{}_{}.scx",
                codec_name(codec),
                encoding_name(encoding)
            ));
            let mut w = ScxWriter::new(&path, make_header()).unwrap();
            w.write_obs(&obs).unwrap();
            w.write_var(&var).unwrap();
            w.write_csr_shard(&indptr, &indices, &values_bytes, codec, encoding, 0)
                .unwrap();
            w.finish().unwrap();

            let reader = ScxReader::open(&path).unwrap();
            let csr = reader.read_all_csr_shards().unwrap();
            assert_eq!(csr.indptr, exp_indptr, "{codec:?} {encoding:?} indptr");
            assert_eq!(csr.indices, exp_indices, "{codec:?} {encoding:?} indices");
            decoded.push(csr);
        }
        // scx2 decodes bit-identically to scx1 for the same matrix.
        assert_eq!(decoded[0].indptr, decoded[1].indptr, "{encoding:?} indptr");
        assert_eq!(
            decoded[0].indices, decoded[1].indices,
            "{encoding:?} indices"
        );
        assert_eq!(decoded[0].data, decoded[1].data, "{encoding:?} data");
    }
    fs::remove_dir_all(&tmp).ok();
}

// ---------------------------------------------------------------------------
// Generator (run with --ignored)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn generate_golden_files() {
    let dir = golden_dir();
    fs::create_dir_all(&dir).unwrap();

    let obs = make_obs();
    let var = make_var();

    let obs_cell_ids: Vec<String> = (0..N_OBS).map(|i| format!("golden_cell_{i}")).collect();
    let obs_cell_types: Vec<String> = (0..N_OBS)
        .map(|i| match i % 3 {
            0 => "T cell".to_string(),
            1 => "B cell".to_string(),
            _ => "NK cell".to_string(),
        })
        .collect();
    let var_gene_ids: Vec<String> = (0..N_VARS).map(|i| format!("gene_{i}")).collect();

    let mut manifest_files = BTreeMap::new();

    for (codec, encoding) in golden_combinations() {
        let basename = golden_basename(codec, encoding);
        let scx_path = dir.join(format!("{basename}.scx"));
        let json_path = dir.join(format!("{basename}.json"));

        // Generate matrix for this encoding
        let (indptr, indices, data_f32) = generate_matrix(encoding);
        let nnz = data_f32.len();

        // Encode values to raw bytes
        let values_bytes = encoding.encode_f32_batch(&data_f32).unwrap();

        // Write SCX file
        let header = make_header();
        let mut writer = ScxWriter::new(&scx_path, header).unwrap();
        writer.write_obs(&obs).unwrap();
        writer.write_var(&var).unwrap();
        writer
            .write_csr_shard(&indptr, &indices, &values_bytes, codec, encoding, 0)
            .unwrap();
        writer.finish().unwrap();

        // Read back to get the post-roundtrip f32 values (important for Float32)
        let reader = ScxReader::open(&scx_path).unwrap();
        let csr = reader.read_all_csr_shards().unwrap();

        // Write JSON sidecar using read-back values
        let sidecar = GoldenSidecar {
            codec: codec_name(codec).to_string(),
            value_encoding: encoding_name(encoding).to_string(),
            n_obs: N_OBS,
            n_vars: N_VARS,
            nnz,
            indptr: csr.indptr.clone(),
            indices: csr.indices.clone(),
            data: csr.data.clone(),
            obs_cell_ids: obs_cell_ids.clone(),
            obs_cell_types: obs_cell_types.clone(),
            var_gene_ids: var_gene_ids.clone(),
        };
        let json = serde_json::to_string_pretty(&sidecar).unwrap();
        fs::write(&json_path, &json).unwrap();

        // Compute BLAKE3 hash for manifest
        let file_bytes = fs::read(&scx_path).unwrap();
        let hash = blake3::hash(&file_bytes);
        manifest_files.insert(format!("{basename}.scx"), hash.to_hex().to_string());

        eprintln!(
            "Generated {basename}.scx ({nnz} nnz, {} bytes) + sidecar",
            file_bytes.len()
        );
    }

    // Write MANIFEST.json
    let manifest = Manifest {
        algorithm: "blake3".to_string(),
        files: manifest_files,
    };
    let manifest_json = serde_json::to_string_pretty(&manifest).unwrap();
    fs::write(dir.join("MANIFEST.json"), &manifest_json).unwrap();

    eprintln!(
        "Generated MANIFEST.json with {} entries",
        manifest.files.len()
    );
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Load all JSON sidecars from the golden directory.
fn load_sidecars() -> Vec<(String, GoldenSidecar)> {
    let dir = golden_dir();
    let mut sidecars = Vec::new();
    for (codec, encoding) in golden_combinations() {
        let basename = golden_basename(codec, encoding);
        let json_path = dir.join(format!("{basename}.json"));
        assert!(
            json_path.exists(),
            "Missing golden sidecar: {json_path:?}. Run: cargo test -p scx-integration-tests \
             --test golden_files generate_golden_files -- --ignored"
        );
        let content = fs::read_to_string(&json_path).unwrap();
        let sidecar: GoldenSidecar = serde_json::from_str(&content).unwrap();
        sidecars.push((basename, sidecar));
    }
    sidecars
}

fn scx_path_for(basename: &str) -> PathBuf {
    golden_dir().join(format!("{basename}.scx"))
}

// ---------------------------------------------------------------------------
// Validation tests (always run)
// ---------------------------------------------------------------------------

/// `scx validate` passes for every golden file (checksums intact).
#[test]
fn test_golden_files_validate() {
    for (basename, _) in load_sidecars() {
        let path = scx_path_for(&basename);
        let reader = ScxReader::open(&path).unwrap();
        let results = reader.validate().unwrap();
        for (section, passed) in &results {
            assert!(
                passed,
                "{basename}: checksum failed for section '{section}'"
            );
        }
    }
}

/// CSR arrays match the JSON sidecar exactly (element-by-element).
#[test]
fn test_golden_files_csr_match() {
    for (basename, sidecar) in load_sidecars() {
        let path = scx_path_for(&basename);
        let reader = ScxReader::open(&path).unwrap();
        let csr = reader.read_all_csr_shards().unwrap();

        assert_eq!(
            csr.shape,
            (sidecar.n_obs, sidecar.n_vars),
            "{basename}: shape mismatch"
        );
        assert_eq!(csr.data.len(), sidecar.nnz, "{basename}: nnz mismatch");
        assert_eq!(csr.indptr, sidecar.indptr, "{basename}: indptr mismatch");
        assert_eq!(csr.indices, sidecar.indices, "{basename}: indices mismatch");

        // f32 bit-exact comparison (lossless codecs)
        for (i, (got, expected)) in csr.data.iter().zip(sidecar.data.iter()).enumerate() {
            assert!(
                got.to_bits() == expected.to_bits(),
                "{basename}: data[{i}] mismatch: got {got} (bits {:08x}), expected {expected} (bits {:08x})",
                got.to_bits(),
                expected.to_bits(),
            );
        }
    }
}

/// obs/var metadata matches the sidecar.
#[test]
fn test_golden_files_metadata_match() {
    for (basename, sidecar) in load_sidecars() {
        let path = scx_path_for(&basename);
        let reader = ScxReader::open(&path).unwrap();

        // Check obs
        let obs = reader.read_obs().unwrap();
        let cell_ids = obs
            .column_by_name("cell_id")
            .expect("missing cell_id column")
            .as_string::<i32>();
        let cell_types = obs
            .column_by_name("cell_type")
            .expect("missing cell_type column")
            .as_string::<i32>();

        for i in 0..sidecar.n_obs {
            assert_eq!(
                cell_ids.value(i),
                sidecar.obs_cell_ids[i],
                "{basename}: obs cell_id[{i}] mismatch"
            );
            assert_eq!(
                cell_types.value(i),
                sidecar.obs_cell_types[i],
                "{basename}: obs cell_type[{i}] mismatch"
            );
        }

        // Check var
        let var = reader.read_var().unwrap();
        let gene_ids = var
            .column_by_name("gene_id")
            .expect("missing gene_id column")
            .as_string::<i32>();

        for i in 0..sidecar.n_vars {
            assert_eq!(
                gene_ids.value(i),
                sidecar.var_gene_ids[i],
                "{basename}: var gene_id[{i}] mismatch"
            );
        }
    }
}

/// BLAKE3 hashes in MANIFEST.json match the actual file hashes.
#[test]
fn test_manifest_hashes() {
    let dir = golden_dir();
    let manifest_path = dir.join("MANIFEST.json");
    assert!(
        manifest_path.exists(),
        "Missing MANIFEST.json. Run: cargo test -p scx-integration-tests \
         --test golden_files generate_golden_files -- --ignored"
    );

    let content = fs::read_to_string(&manifest_path).unwrap();
    let manifest: Manifest = serde_json::from_str(&content).unwrap();
    assert_eq!(manifest.algorithm, "blake3");

    for (filename, expected_hash) in &manifest.files {
        let file_path = dir.join(filename);
        assert!(
            file_path.exists(),
            "Manifest references missing file: {filename}"
        );
        let file_bytes = fs::read(&file_path).unwrap();
        let actual_hash = blake3::hash(&file_bytes).to_hex().to_string();
        assert_eq!(
            &actual_hash, expected_hash,
            "BLAKE3 hash mismatch for {filename}"
        );
    }

    // Verify the manifest contains an entry for every codec-compat
    // golden fixture. Added conformance vectors that share
    // this MANIFEST.json (also validated via `conformance_vectors.rs`),
    // so the total entry count may exceed `golden_combinations()`.
    for (codec, encoding) in golden_combinations() {
        let basename = golden_basename(codec, encoding);
        let key = format!("{basename}.scx");
        assert!(
            manifest.files.contains_key(&key),
            "Manifest missing golden entry: {key}"
        );
    }
}

// ---------------------------------------------------------------------------
// Backward compatibility tests
// ---------------------------------------------------------------------------

/// A reader that only knows codecs 0–2 should reject LZ4+shuffle files
/// with a clear `UnknownCodec` error, not a crash or silent corruption.
///
/// We simulate this by writing an LZ4+shuffle file, then patching the shard header's
/// codec_id byte to a hypothetical future codec (e.g., 99) and verifying the reader
/// produces a clear error.
#[test]
fn test_unknown_codec_rejected_gracefully() {
    // Write a valid Lz4Shuffle file
    let dir = tempfile::tempdir().unwrap();
    let scx_path = dir.path().join("test_unknown_codec.scx");

    let (indptr, indices, data_f32) = generate_matrix(ValueEncoding::Uint16);
    let values_bytes = ValueEncoding::Uint16.encode_f32_batch(&data_f32).unwrap();

    let header = make_header();
    let mut writer = ScxWriter::new(&scx_path, header).unwrap();
    writer.write_obs(&make_obs()).unwrap();
    writer.write_var(&make_var()).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &values_bytes,
            CodecId::Lz4Shuffle,
            ValueEncoding::Uint16,
            0,
        )
        .unwrap();
    writer.finish().unwrap();

    // Verify the file reads correctly with the current reader
    let reader = ScxReader::open(&scx_path).unwrap();
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape, (N_OBS, N_VARS));

    // Now patch the shard header's codec_id byte to 99 (unknown future codec)
    let mut file_bytes = fs::read(&scx_path).unwrap();

    // Find the shard header magic "SCXS" and patch codec_id (offset +6 from magic)
    let shard_magic = b"SCXS";
    let mut patched = false;
    for i in 0..file_bytes.len() - 4 {
        if &file_bytes[i..i + 4] == shard_magic {
            // Shard header layout: magic(4) + shard_format_version(1) + shard_type(1) + codec_id(1)
            file_bytes[i + 6] = 99; // patch codec_id to unknown value
            patched = true;
            break;
        }
    }
    assert!(patched, "Could not find shard header magic in file");

    // Write patched file
    let patched_path = dir.path().join("test_patched_codec.scx");
    fs::write(&patched_path, &file_bytes).unwrap();

    // The reader should open the file (header is fine) but fail on shard decode
    // with UnknownCodec error
    let reader2 = ScxReader::open(&patched_path).unwrap();
    let result = reader2.read_all_csr_shards();
    assert!(result.is_err(), "Expected error for unknown codec, got Ok");
    let err = result.unwrap_err();
    let err_msg = format!("{err}");
    assert!(
        err_msg.contains("unknown codec ID: 99"),
        "Expected 'unknown codec ID: 99' in error message, got: {err_msg}"
    );
}

/// Pre-LZ4 golden files (codecs 0–2) must still be readable after later
/// codec additions. This is implicitly tested by the above tests
/// (test_golden_files_validate, test_golden_files_csr_match, etc.) which
/// run on every `cargo test`. This test explicitly verifies that subset.
#[test]
fn test_phase0_golden_files_still_readable() {
    let phase0_combinations = vec![
        (CodecId::None, ValueEncoding::Uint8),
        (CodecId::None, ValueEncoding::Uint16),
        (CodecId::None, ValueEncoding::Uint32),
        (CodecId::None, ValueEncoding::Float32),
        (CodecId::Scx1, ValueEncoding::Uint8),
        (CodecId::Scx1, ValueEncoding::Uint16),
        (CodecId::Scx1, ValueEncoding::Uint32),
        (CodecId::Zstd, ValueEncoding::Uint8),
        (CodecId::Zstd, ValueEncoding::Uint16),
        (CodecId::Zstd, ValueEncoding::Uint32),
        (CodecId::Zstd, ValueEncoding::Float32),
    ];

    for (codec, encoding) in phase0_combinations {
        let basename = golden_basename(codec, encoding);
        let path = scx_path_for(&basename);
        assert!(path.exists(), "Phase 0 golden file missing: {path:?}");
        let reader = ScxReader::open(&path).unwrap();
        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(
            csr.shape,
            (N_OBS, N_VARS),
            "{basename}: shape mismatch in Phase 0 golden file"
        );
    }
}
