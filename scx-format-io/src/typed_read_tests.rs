//! In-assembly narrow reader tests.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use scx_codec::{checked_cast_indices, checked_cast_values, CodecId, ValueEncoding};
use scx_sparse::{Container, IndexBuffer, IndexDtype, MaterializePlan, ValueBuffer, ValueDtype};

use crate::encoder::{encode_one_shard, FramingConfig};
use crate::modality::ModalityType;
use crate::shard::ShardHeader;
use crate::shard_decode::{decode_shard_regions_native, decode_shard_regions_scipy};
use crate::writer::ScxWriter;
use crate::{FileHeader, ScxReader, SectionType};
use scx_codec::ShardValuesNative;

// -----------------------------------------------------------------------
// Test A — native decode == scipy decode, per codec, framed + unframed
// -----------------------------------------------------------------------

/// Assert a native-decoded shard matches the scipy-decoded shard value-for-value.
fn assert_native_matches_scipy(
    native: &(Vec<i64>, Vec<u32>, ShardValuesNative),
    scipy: &(Vec<i64>, Vec<i32>, Vec<f32>),
) {
    let (n_ip, n_ix, n_vals) = native;
    let (s_ip, s_ix, s_data) = scipy;
    assert_eq!(n_ip, s_ip, "indptr differ");
    let n_ix_i32: Vec<i32> = n_ix.iter().map(|&v| v as i32).collect();
    assert_eq!(&n_ix_i32, s_ix, "indices differ");
    let n_as_f32: Vec<f32> = match n_vals {
        ShardValuesNative::U32(v) => v.iter().map(|&x| x as f32).collect(),
        ShardValuesNative::F32(v) => v.clone(),
    };
    assert_eq!(&n_as_f32, s_data, "values differ");
}

/// `(native decode, scipy decode)` of the same shard, for the parity assertions.
type NativeAndScipy = (
    (Vec<i64>, Vec<u32>, ShardValuesNative),
    (Vec<i64>, Vec<i32>, Vec<f32>),
);

fn decode_both(codec: CodecId, values: &[f32], framing: Option<FramingConfig>) -> NativeAndScipy {
    let indptr = [0u64, 2, 2, 5, 7];
    let indices = [0u32, 3, 1, 4, 9, 2, 8];
    let n_cols: u32 = 16;

    let mut opts = crate::encoder::EncodeShardOptions::new(
        "X_shard_0",
        SectionType::CsrShard,
        n_cols as u64,
        0,
        0,
    );
    opts.explicit_codec = Some(codec);
    opts.framing = framing;
    let s = encode_one_shard(&indptr, &indices, values, &opts).expect("encode_one_shard");
    let sh = ShardHeader::read_from(&mut Cursor::new(&s.header_buf[..])).unwrap();

    let native = decode_shard_regions_native(
        &sh,
        &s.encoded.indptr_bytes,
        &s.encoded.indices_bytes,
        &s.encoded.values_bytes,
        &s.block_index_bytes,
    )
    .expect("native decode");
    let scipy = decode_shard_regions_scipy(
        &sh,
        &s.encoded.indptr_bytes,
        &s.encoded.indices_bytes,
        &s.encoded.values_bytes,
        &s.block_index_bytes,
    )
    .expect("scipy decode");
    (native, scipy)
}

#[test]
fn native_decode_matches_scipy_integer_codecs() {
    // Integer values (representable exactly in both u32 and f32).
    let values: Vec<f32> = vec![1.0, 4.0, 2.0, 5.0, 9.0, 3.0, 7.0];
    for codec in [
        CodecId::None,
        CodecId::Zstd,
        CodecId::Lz4Shuffle,
        CodecId::ShufDeltaZstd,
        CodecId::Scx1,
    ] {
        let (native, scipy) = decode_both(codec, &values, None);
        // Integer shards must decode to the U32 native variant.
        assert!(
            matches!(native.2, ShardValuesNative::U32(_)),
            "codec {codec:?} integer shard should be U32-native"
        );
        assert_native_matches_scipy(&native, &scipy);

        // Framed (v2) parity.
        let (nf, sf) = decode_both(
            codec,
            &values,
            Some(FramingConfig {
                row_group_rows: 2,
                target_nnz: None,
                trial: false,
                decode_target: None,
            }),
        );
        assert_native_matches_scipy(&nf, &sf);
    }
}

#[test]
fn native_decode_matches_scipy_float_codec() {
    // Non-integer values force a float encoding → F32 native variant.
    let values: Vec<f32> = vec![1.5, 4.25, 2.5, 5.75, 9.125, 3.0, 7.5];
    for codec in [CodecId::None, CodecId::Zstd, CodecId::Pcodec] {
        let (native, scipy) = decode_both(codec, &values, None);
        assert!(
            matches!(native.2, ShardValuesNative::F32(_)),
            "codec {codec:?} float shard should be F32-native"
        );
        assert_native_matches_scipy(&native, &scipy);
    }
}

// -----------------------------------------------------------------------
// Test B / C — full typed assembly through the writer+reader
// -----------------------------------------------------------------------

fn header(n_obs: u64, n_vars: u64, nnz: u64) -> FileHeader {
    FileHeader::new_single_modality(n_obs, n_vars, nnz, 16384, 0, 0)
}

fn obs_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("cell_{i}")).collect();
    let schema = Schema::new(vec![Field::new("cell_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

fn var_batch(n: usize) -> RecordBatch {
    let ids: Vec<String> = (0..n).map(|i| format!("gene_{i}")).collect();
    let schema = Schema::new(vec![Field::new("gene_id", DataType::Utf8, false)]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(StringArray::from(
            ids.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// A shard: (indptr, indices, raw value bytes, encoding, row_start).
type ShardSpec = (Vec<u64>, Vec<u32>, Vec<u8>, ValueEncoding, u64);

fn write_file(
    dir: &tempfile::TempDir,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    shards: &[ShardSpec],
) -> std::path::PathBuf {
    write_file_codec(dir, name, n_obs, n_vars, shards, CodecId::None)
}

fn write_file_codec(
    dir: &tempfile::TempDir,
    name: &str,
    n_obs: usize,
    n_vars: usize,
    shards: &[ShardSpec],
    codec: CodecId,
) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let total_nnz: u64 = shards.iter().map(|s| *s.0.last().unwrap()).sum();
    let hdr = header(n_obs as u64, n_vars as u64, total_nnz);
    let mut writer = ScxWriter::new(&path, hdr).unwrap();
    writer.write_obs(&obs_batch(n_obs)).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    for (indptr, indices, values, enc, row_start) in shards {
        writer
            .write_csr_shard(indptr, indices, values, codec, *enc, *row_start)
            .unwrap();
    }
    writer.finish().unwrap();
    path
}

/// Build a small Uint8 shard: `n_rows` rows, 2 nnz each.
fn u8_shard(n_rows: usize, n_vars: usize, row_start: u64, val_base: u8) -> ShardSpec {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        indices.push(((row * 2) % n_vars) as u32);
        indices.push(((row * 2 + 1) % n_vars) as u32);
        values.push(val_base.wrapping_add(row as u8).wrapping_add(1));
        values.push(val_base.wrapping_add(row as u8).wrapping_add(2));
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values, ValueEncoding::Uint8, row_start)
}

/// Build a small Uint16 shard: `n_rows` rows, 2 nnz each, LE value bytes.
fn u16_shard(n_rows: usize, n_vars: usize, row_start: u64, val_base: u16) -> ShardSpec {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values: Vec<u8> = Vec::new();
    for row in 0..n_rows {
        indices.push(((row * 2) % n_vars) as u32);
        indices.push(((row * 2 + 1) % n_vars) as u32);
        values.extend_from_slice(&(val_base + row as u16 + 1).to_le_bytes());
        values.extend_from_slice(&(val_base + row as u16 + 2).to_le_bytes());
        indptr.push(indptr.last().unwrap() + 2);
    }
    (indptr, indices, values, ValueEncoding::Uint16, row_start)
}

/// A per-modality spec for the multimodal writer.
type ModalitySpec = (&'static str, ModalityType, usize, Vec<ShardSpec>);

/// Write a v2 multimodal SCX file. Mirrors `mudata.rs::from_mudata_impl`'s writer
/// sequence (`add_modality` → `set_modality_n_vars` → `write_var_for` →
/// `write_csr_shard_for`), so `finish()` emits a `ModalityTable` and the reader
/// opens it as multimodal.
fn write_multimodal_file(
    dir: &tempfile::TempDir,
    name: &str,
    n_obs: usize,
    modalities: &[ModalitySpec],
    codec: CodecId,
    framing: Option<FramingConfig>,
) -> std::path::PathBuf {
    let path = dir.path().join(name);
    let total_nnz: u64 = modalities
        .iter()
        .flat_map(|(_, _, _, shards)| shards.iter())
        .map(|s| *s.0.last().unwrap())
        .sum();
    let max_n_vars = modalities
        .iter()
        .map(|(_, _, nv, _)| *nv)
        .max()
        .unwrap_or(0) as u64;
    let hdr = FileHeader::new_single_modality(n_obs as u64, max_n_vars, total_nnz, 16384, 0, 0);
    let mut writer = ScxWriter::new(&path, hdr).unwrap();
    writer.set_framing(framing);
    writer.write_obs(&obs_batch(n_obs)).unwrap();
    for (mname, mtype, n_vars, shards) in modalities {
        let enc = shards[0].3;
        let mid = writer
            .add_modality(mname, *mtype, codec, enc, false)
            .unwrap();
        writer.set_modality_n_vars(mid, *n_vars as u64).unwrap();
        writer.write_var_for(mid, &var_batch(*n_vars)).unwrap();
        for (indptr, indices, values, senc, row_start) in shards {
            let shard = crate::writer::ShardBuffers::new(indptr, indices, values, codec, *senc);
            writer.write_csr_shard_for(mid, *row_start, shard).unwrap();
        }
    }
    writer.finish().unwrap();
    path
}

/// Per-modality typed read equals the f32 sibling (`read_all_csr_shards_for`) cast
/// down, for `None` and a framed non-`None` codec — with the RNA modality narrowed
/// to `uint16` and a second (ATAC-like) modality to `uint8` in the same file.
#[test]
fn typed_per_modality_matches_f32_cast() {
    for (codec, framing) in [
        (CodecId::None, None),
        (
            CodecId::Scx1,
            Some(FramingConfig {
                row_group_rows: 2,
                target_nnz: None,
                trial: false,
                decode_target: None,
            }),
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let modalities: Vec<ModalitySpec> = vec![
            (
                "rna",
                ModalityType::Rna,
                10,
                vec![u16_shard(3, 10, 0, 300), u16_shard(3, 10, 3, 1000)],
            ),
            ("atac", ModalityType::Atac, 8, vec![u8_shard(6, 8, 0, 0)]),
        ];
        let path = write_multimodal_file(&dir, "mm.scx", 6, &modalities, codec, framing);
        let reader = ScxReader::open(&path).unwrap();
        assert!(reader.is_multimodal(), "codec {codec:?}: not multimodal");
        assert_eq!(reader.n_modalities(), 2);

        // Modality 1 (rna) → uint16.
        let rna_f32 = reader.read_all_csr_shards_for(1).unwrap();
        let rna_plan = MaterializePlan {
            container: Container::Csr,
            data_dtype: ValueDtype::U16,
            index_dtype: IndexDtype::I32,
            allow_lossy: false,
        };
        let rna_typed = reader.read_all_csr_shards_for_typed(1, &rna_plan).unwrap();
        assert_eq!(rna_typed.shape, rna_f32.shape, "codec {codec:?}: rna shape");
        assert_eq!(
            rna_typed.indptr, rna_f32.indptr,
            "codec {codec:?}: rna indptr"
        );
        let want_v: Vec<u16> = checked_cast_values(&rna_f32.data, false).unwrap();
        match &rna_typed.values {
            ValueBuffer::U16(v) => assert_eq!(v, &want_v, "codec {codec:?}: rna values"),
            _ => panic!("codec {codec:?}: wrong rna value arm"),
        }

        // Modality 2 (atac) → uint8, in the same read.
        let atac_f32 = reader.read_all_csr_shards_for(2).unwrap();
        let atac_plan = MaterializePlan {
            container: Container::Csr,
            data_dtype: ValueDtype::U8,
            index_dtype: IndexDtype::I32,
            allow_lossy: false,
        };
        let atac_typed = reader.read_all_csr_shards_for_typed(2, &atac_plan).unwrap();
        assert_eq!(
            atac_typed.shape, atac_f32.shape,
            "codec {codec:?}: atac shape"
        );
        let want_a: Vec<u8> = checked_cast_values(&atac_f32.data, false).unwrap();
        match &atac_typed.values {
            ValueBuffer::U8(v) => assert_eq!(v, &want_a, "codec {codec:?}: atac values"),
            _ => panic!("codec {codec:?}: wrong atac value arm"),
        }
    }
}

/// A per-modality read of a modality carrying a `value_max > 2²⁴` count reads exact
/// under `uint32` (G2), and fails loud into `uint16` without `allow_lossy`.
#[test]
fn typed_per_modality_exact_above_2pow24() {
    let big: u32 = (1 << 24) + 7; // 16_777_223, unrepresentable exactly in f32
    let mut vals = Vec::new();
    vals.extend_from_slice(&big.to_le_bytes());
    vals.extend_from_slice(&5u32.to_le_bytes());
    let big_shard: ShardSpec = (
        vec![0u64, 1, 2],
        vec![0u32, 1u32],
        vals,
        ValueEncoding::Uint32,
        0,
    );

    let dir = tempfile::tempdir().unwrap();
    let modalities: Vec<ModalitySpec> = vec![
        ("rna", ModalityType::Rna, 4, vec![u8_shard(2, 4, 0, 0)]),
        ("big", ModalityType::Custom, 4, vec![big_shard]),
    ];
    let path = write_multimodal_file(&dir, "mm_big.scx", 2, &modalities, CodecId::None, None);
    let reader = ScxReader::open(&path).unwrap();

    let plan_u32 = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U32,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let typed = reader.read_all_csr_shards_for_typed(2, &plan_u32).unwrap();
    match &typed.values {
        ValueBuffer::U32(v) => assert_eq!(v, &vec![big, 5]),
        _ => panic!("wrong arm"),
    }

    let plan_u16 = MaterializePlan {
        data_dtype: ValueDtype::U16,
        ..plan_u32
    };
    assert!(
        reader.read_all_csr_shards_for_typed(2, &plan_u16).is_err(),
        "value_max > 65535 must fail loud into uint16"
    );
}

/// A missing modality id (e.g. id 0 / global on a multimodal file) errors rather
/// than assembling a garbage shape — the typed reader needs `n_cols` up front.
#[test]
fn typed_per_modality_missing_modality_info_errors() {
    let dir = tempfile::tempdir().unwrap();
    let modalities: Vec<ModalitySpec> =
        vec![("rna", ModalityType::Rna, 4, vec![u8_shard(2, 4, 0, 0)])];
    let path = write_multimodal_file(&dir, "mm_one.scx", 2, &modalities, CodecId::None, None);
    let reader = ScxReader::open(&path).unwrap();
    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    // modality_id 0 is the global slot — no ModalityInfo → error.
    assert!(reader.read_all_csr_shards_for_typed(0, &plan).is_err());
}

#[test]
fn typed_assembly_matches_f32_cast_csr() {
    let dir = tempfile::tempdir().unwrap();
    // Two shards, 3 rows each.
    let shards = vec![u8_shard(3, 10, 0, 0), u8_shard(3, 10, 3, 100)];
    let path = write_file(&dir, "typed.scx", 6, 10, &shards);
    let reader = ScxReader::open(&path).unwrap();

    let f32_csr = reader.read_all_csr_shards().unwrap();

    // u16 values / i32 indices.
    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let typed = reader.read_all_csr_shards_typed(&plan).unwrap();
    assert_eq!(typed.shape, f32_csr.shape);
    assert_eq!(typed.indptr, f32_csr.indptr);
    let want_ix: Vec<i32> = checked_cast_indices(&f32_csr.indices, false).unwrap();
    let want_v: Vec<u16> = checked_cast_values(&f32_csr.data, false).unwrap();
    match (&typed.indices, &typed.values) {
        (IndexBuffer::I32(ix), ValueBuffer::U16(v)) => {
            assert_eq!(ix, &want_ix);
            assert_eq!(v, &want_v);
        }
        _ => panic!("wrong buffer arms"),
    }

    // f64 values / i64 indices.
    let plan2 = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::F64,
        index_dtype: IndexDtype::I64,
        allow_lossy: false,
    };
    let typed2 = reader.read_all_csr_shards_typed(&plan2).unwrap();
    let want_v64: Vec<f64> = checked_cast_values(&f32_csr.data, false).unwrap();
    let want_ix64: Vec<i64> = checked_cast_indices(&f32_csr.indices, false).unwrap();
    match (&typed2.indices, &typed2.values) {
        (IndexBuffer::I64(ix), ValueBuffer::F64(v)) => {
            assert_eq!(ix, &want_ix64);
            assert_eq!(v, &want_v64);
        }
        _ => panic!("wrong buffer arms"),
    }
}

/// End-to-end typed assembly (the reader + `assemble_shards_typed` +
/// `read_shard_from_entry_native`, not just the `decode_shard_regions_native`
/// primitive) across the real codecs — not only `CodecId::None`. Confirms the
/// assembler's per-shard native decode + cast + indptr rebasing is codec-agnostic.
#[test]
fn typed_assembly_matches_f32_cast_across_codecs() {
    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    for codec in [
        CodecId::None,
        CodecId::Zstd,
        CodecId::Scx1,
        CodecId::ShufDeltaZstd,
        CodecId::Lz4Shuffle,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let shards = vec![u8_shard(3, 10, 0, 0), u8_shard(3, 10, 3, 100)];
        let path = write_file_codec(&dir, "codec.scx", 6, 10, &shards, codec);
        let reader = ScxReader::open(&path).unwrap();

        let f32_csr = reader.read_all_csr_shards().unwrap();
        let typed = reader.read_all_csr_shards_typed(&plan).unwrap();
        assert_eq!(typed.shape, f32_csr.shape, "shape mismatch for {codec:?}");
        assert_eq!(
            typed.indptr, f32_csr.indptr,
            "indptr mismatch for {codec:?}"
        );
        let want_ix: Vec<i32> = checked_cast_indices(&f32_csr.indices, false).unwrap();
        let want_v: Vec<u16> = checked_cast_values(&f32_csr.data, false).unwrap();
        match (&typed.indices, &typed.values) {
            (IndexBuffer::I32(ix), ValueBuffer::U16(v)) => {
                assert_eq!(ix, &want_ix, "indices mismatch for {codec:?}");
                assert_eq!(v, &want_v, "values mismatch for {codec:?}");
            }
            _ => panic!("wrong buffer arms for {codec:?}"),
        }
    }
}

/// `index_dtype="int16"` narrowing: the reader produces an `IndexBuffer::I16`
/// arm, and a column index that overflows `i16` (≥ 32768) fails loud unless
/// `allow_lossy` (spec § 4 index-narrowing contract). This is the only coverage
/// of the I16 fill arm (scipy upcasts int16→int32, so a Python CSR read can't
/// observe it — see docs/api/python-experiment.md).
#[test]
fn typed_index_narrow_i16() {
    // In-range: n_vars small, indices < 32768 → I16 arm, exact.
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![u8_shard(4, 10, 0, 0)];
    let path = write_file(&dir, "i16_ok.scx", 4, 10, &shards);
    let reader = ScxReader::open(&path).unwrap();
    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I16,
        allow_lossy: false,
    };
    let typed = reader.read_all_csr_shards_typed(&plan).unwrap();
    let f32_csr = reader.read_all_csr_shards().unwrap();
    let want_ix: Vec<i16> = checked_cast_indices(&f32_csr.indices, false).unwrap();
    match &typed.indices {
        IndexBuffer::I16(ix) => assert_eq!(ix, &want_ix),
        _ => panic!("expected I16 index arm"),
    }

    // Out-of-range: a column index of 32768 (≥ i16::MAX+1) fails loud into i16.
    let dir2 = tempfile::tempdir().unwrap();
    let n_vars = 40_000usize;
    let indptr = vec![0u64, 1];
    let indices = vec![32_768u32]; // > i16::MAX (32767)
    let values = vec![7u8];
    let shard: ShardSpec = (indptr, indices, values, ValueEncoding::Uint8, 0);
    let path2 = write_file(&dir2, "i16_overflow.scx", 1, n_vars, &[shard]);
    let reader2 = ScxReader::open(&path2).unwrap();
    assert!(
        reader2.read_all_csr_shards_typed(&plan).is_err(),
        "column index 32768 must fail loud into int16 without allow_lossy"
    );
    // allow_lossy wraps without error.
    let plan_lossy = MaterializePlan {
        allow_lossy: true,
        ..plan
    };
    assert!(reader2.read_all_csr_shards_typed(&plan_lossy).is_ok());
}

#[test]
fn typed_dense_matches_f32_dense() {
    let dir = tempfile::tempdir().unwrap();
    let shards = vec![u8_shard(4, 8, 0, 0)];
    let path = write_file(&dir, "dense.scx", 4, 8, &shards);
    let reader = ScxReader::open(&path).unwrap();

    let f32_csr = reader.read_all_csr_shards().unwrap();
    let want_dense: Vec<u32> = {
        // Scatter the f32 CSR to dense u32 as the reference.
        let (n_rows, n_cols) = f32_csr.shape;
        let mut d = vec![0u32; n_rows * n_cols];
        for row in 0..n_rows {
            let s = f32_csr.indptr[row] as usize;
            let e = f32_csr.indptr[row + 1] as usize;
            for k in s..e {
                d[row * n_cols + f32_csr.indices[k] as usize] = f32_csr.data[k] as u32;
            }
        }
        d
    };

    let plan = MaterializePlan {
        container: Container::Dense,
        data_dtype: ValueDtype::U32,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let dense = reader.read_all_csr_shards_dense_typed(&plan).unwrap();
    assert_eq!(dense.shape, f32_csr.shape);
    match &dense.values {
        ValueBuffer::U32(v) => assert_eq!(v, &want_dense),
        _ => panic!("wrong arm"),
    }
}

#[test]
fn g2_exact_integer_read_above_2pow24() {
    // A single Uint32 shard carrying a count > 2²⁴ (unrepresentable exactly in f32).
    let big: u32 = (1 << 24) + 7; // 16_777_223
    let indptr = vec![0u64, 1, 2];
    let indices = vec![0u32, 1u32];
    let mut values = Vec::new();
    values.extend_from_slice(&big.to_le_bytes());
    values.extend_from_slice(&5u32.to_le_bytes());
    let shard: ShardSpec = (indptr, indices, values, ValueEncoding::Uint32, 0);

    let dir = tempfile::tempdir().unwrap();
    let path = write_file(&dir, "big.scx", 2, 4, &[shard]);
    let reader = ScxReader::open(&path).unwrap();

    // uint32: exact, lossless.
    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U32,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let typed = reader.read_all_csr_shards_typed(&plan).unwrap();
    match &typed.values {
        ValueBuffer::U32(v) => assert_eq!(v, &vec![big, 5]),
        _ => panic!("wrong arm"),
    }

    // uint16: value_max > 65535 → fail loud (O(1) guard) without allow_lossy.
    let plan_narrow = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    assert!(reader.read_all_csr_shards_typed(&plan_narrow).is_err());
}

#[cfg(feature = "deletion-vectors")]
#[test]
fn typed_read_applies_deletion_vectors() {
    use crate::deletion_vectors::DeletionVectors;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dv.scx");
    let shards = [u8_shard(6, 10, 0, 0)];
    let total_nnz: u64 = *shards[0].0.last().unwrap();
    let hdr = header(6, 10, total_nnz);
    let mut writer = ScxWriter::new(&path, hdr).unwrap();
    writer.write_obs(&obs_batch(6)).unwrap();
    writer.write_var(&var_batch(10)).unwrap();
    let (ip, ix, v, enc, rs) = &shards[0];
    writer
        .write_csr_shard(ip, ix, v, CodecId::None, *enc, *rs)
        .unwrap();
    // Delete global rows 1 and 4 (single shard, row_start 0 → local == global).
    let mut dv = DeletionVectors::new();
    dv.insert_global([1u32, 4]);
    writer.write_deletion_vectors(&dv).unwrap();
    writer.finish().unwrap();

    let reader = ScxReader::open(&path).unwrap();
    let f32_filtered = reader.read_all_csr_shards_filtered().unwrap();

    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let typed = reader.read_all_csr_shards_typed(&plan).unwrap();
    assert_eq!(typed.shape, f32_filtered.shape);
    assert_eq!(typed.indptr, f32_filtered.indptr);
    let want_v: Vec<u16> = checked_cast_values(&f32_filtered.data, false).unwrap();
    match &typed.values {
        ValueBuffer::U16(v) => assert_eq!(v, &want_v),
        _ => panic!("wrong arm"),
    }
}

/// The typed twin of `deletion_mask_longer_than_csr_errors` /
/// `deletion_mask_shorter_than_csr_errors`.
///
/// These two paths are each other's oracle in
/// `typed_read_applies_deletion_vectors`, which asserts the typed result equals
/// the f32 result. That agreement is only evidence if both sides reject the
/// same malformed input — a guard on one side alone would make them disagree
/// on exactly the files where the comparison matters.
#[cfg(feature = "deletion-vectors")]
#[test]
fn typed_deletion_mask_must_match_the_csr_row_count() {
    use crate::deletion_vectors::DeletionVectors;

    let dir = tempfile::tempdir().unwrap();
    // n_obs declared vs CSR rows actually written.
    let build = |name: &str, n_obs: usize, csr_rows: usize| -> std::path::PathBuf {
        let path = dir.path().join(name);
        let (ip, ix, v, enc, rs) = u8_shard(csr_rows, 10, 0, 0);
        let hdr = header(n_obs as u64, 10, *ip.last().unwrap());
        let mut writer = ScxWriter::new(&path, hdr).unwrap();
        writer.write_obs(&obs_batch(n_obs)).unwrap();
        writer.write_var(&var_batch(10)).unwrap();
        writer
            .write_csr_shard(&ip, &ix, &v, CodecId::None, enc, rs)
            .unwrap();
        let mut dv = DeletionVectors::new();
        dv.insert_global([0u32]);
        writer.write_deletion_vectors(&dv).unwrap();
        writer.finish().unwrap();
        path
    };

    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };

    for (name, n_obs, csr_rows, what) in [
        (
            "typed_mask_long.scx",
            10usize,
            4usize,
            "mask longer than CSR",
        ),
        ("typed_mask_short.scx", 4, 10, "mask shorter than CSR"),
    ] {
        let path = build(name, n_obs, csr_rows);
        let reader = ScxReader::open(&path).unwrap();
        let err = reader
            .read_all_csr_shards_typed(&plan)
            .expect_err(&format!("{what} must be rejected, not answered"));
        assert!(
            matches!(err, crate::error::ScxError::InvalidCatalog(_)),
            "{what}: expected InvalidCatalog, got {err:?}"
        );
    }

    // Control: agreeing counts still filter, so the guard is not rejecting
    // every file.
    let path = build("typed_mask_ok.scx", 6, 6);
    let reader = ScxReader::open(&path).unwrap();
    let typed = reader.read_all_csr_shards_typed(&plan).unwrap();
    assert_eq!(typed.shape.0, 5, "one of six rows was deleted");
}

/// The dense scatter is bounded by the **assembled matrix's** `n_cols` (the
/// file header's `n_vars`), while the decode seam validates against the
/// **shard header's** `n_minor`. On any file a writer produced those are equal,
/// so the seam covers the scatter — but they are two independent numbers, and a
/// corrupt shard claiming `n_minor > n_vars` passes the seam while carrying an
/// index the scatter cannot hold.
///
/// The gap matters at this one site because a violation here is **silent**.
/// The assertions below pin the corruption, not merely the error: pre-fix,
/// `dense[base + col]` with `col = 12` and `n_cols = 10` wrote two cells into
/// the *next* row and returned a plausible, wrong 3x10 matrix — no panic, no
/// error. Asserting only `is_err()` would pass against a build that panicked
/// instead, and would not characterise the bug at all.
#[test]
fn dense_scatter_rejects_an_index_the_seam_bound_let_through() {
    use scx_sparse::TypedCsr;

    // 3x10, but row 1 carries column 12 — beyond `n_cols`, and beyond anything
    // this matrix can hold.
    let csr = TypedCsr {
        shape: (3, 10),
        indptr: vec![0, 1, 3, 4],
        indices: IndexBuffer::I32(vec![0, 12, 3, 9]),
        values: ValueBuffer::U16(vec![11, 22, 33, 44]),
    };

    let err = super::scatter_typed_csr_to_dense(&csr)
        .expect_err("an index past n_cols must be rejected, not scattered");
    match err {
        crate::error::ScxError::ShardIndexOutOfRange {
            index,
            position,
            n_minor,
        } => assert_eq!((index, position, n_minor), (12, 1, 10)),
        other => panic!("expected ShardIndexOutOfRange, got {other:?}"),
    }

    // Characterise what the unguarded write would have done, so the failure this
    // guards is on the record rather than described: `base + col` for row 1 is
    // `10 + 12 = 22`, which lands in row 2 (cells 20..30) — a value silently
    // attributed to the wrong cell of the wrong row.
    let (row, col, n_cols) = (1usize, 12usize, 10usize);
    let flat = row * n_cols + col;
    assert_eq!(flat, 22);
    assert_eq!(flat / n_cols, 2, "row 1's value lands in row 2");

    // Control: the same matrix with an in-range index scatters normally, so the
    // guard is not rejecting every dense read.
    let ok = TypedCsr {
        shape: (3, 10),
        indptr: vec![0, 1, 3, 4],
        indices: IndexBuffer::I32(vec![0, 2, 3, 9]),
        values: ValueBuffer::U16(vec![11, 22, 33, 44]),
    };
    let dense = super::scatter_typed_csr_to_dense(&ok).expect("in-range indices scatter");
    match dense.values {
        ValueBuffer::U16(v) => {
            assert_eq!(v.len(), 30);
            assert_eq!(v[0], 11);
            assert_eq!(v[12], 22, "row 1, col 2");
            assert_eq!(v[29], 44, "row 2, col 9");
        }
        other => panic!("wrong arm: {other:?}"),
    }
}

// -----------------------------------------------------------------------
// Parallel typed assembly
// -----------------------------------------------------------------------

/// A shard whose rows carry a *varying* number of nonzeros, with distinct
/// nonzero values.
///
/// Both properties are load-bearing for the carve-up the parallel assembler
/// depends on: with a uniform nnz per row a shard's nnz is recoverable from its
/// row count, so a chunk sized from the wrong shard still lands correctly; and a
/// value of `0` is indistinguishable from an untouched slot in a freshly zeroed
/// buffer, which is exactly what a mis-sized chunk leaves behind. Same reasoning
/// as `irregular_row` on the f32 side.
///
/// `cfg`-gated with its only consumer: without `parallel` there is no `Parallel`
/// arm to compare against, so an ungated helper is dead code under `-D warnings`
/// on the `--no-default-features` clippy legs.
#[cfg(feature = "parallel")]
fn irregular_u8_shard(n_rows: usize, n_vars: usize, row_start: u64, val_base: u8) -> ShardSpec {
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for row in 0..n_rows {
        let global = row_start as usize + row;
        let nnz = (global % 4) + 1;
        let mut cols: Vec<u32> = (0..nnz)
            .map(|k| ((global + k * 3) % n_vars) as u32)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for (k, col) in cols.iter().enumerate() {
            indices.push(*col);
            values.push(val_base.wrapping_add((global * 7 + k * 3) as u8) | 1);
        }
        indptr.push(indptr.last().unwrap() + cols.len() as u64);
    }
    (indptr, indices, values, ValueEncoding::Uint8, row_start)
}

/// Fanning the typed assembly's writes across threads must not reorder or
/// overlap them: `Parallel` and `Sequential` produce byte-identical `indptr` /
/// `indices` / `values`.
///
/// Not a differential between two implementations — there is one, and the
/// strategy is its only parameter. What it pins is the `chunks_mut` carve-up,
/// which is the whole reason the loop can be fanned out at all: every shard
/// writes only the region its own `(n_rows, nnz)` claims, and those regions tile
/// the allocation exactly.
#[test]
#[cfg(feature = "parallel")]
fn typed_parallel_matches_sequential() {
    use crate::reader::RowMajorStrategy;

    let dir = tempfile::tempdir().unwrap();
    let n_vars = 24usize;
    // Five shards of *differing* row counts, so the indptr carve (shard 0 takes
    // n_rows + 1, the rest n_rows) cannot be satisfied by a uniform stride.
    let row_counts = [7usize, 3, 11, 5, 9];
    let mut shards = Vec::new();
    let mut row_start = 0u64;
    for (i, &rows) in row_counts.iter().enumerate() {
        shards.push(irregular_u8_shard(rows, n_vars, row_start, (i * 13) as u8));
        row_start += rows as u64;
    }
    let n_obs = row_start as usize;
    let path = write_file(&dir, "typed_par_seq.scx", n_obs, n_vars, &shards);

    let reader = ScxReader::open(&path).unwrap();
    let entries = reader.catalog().shards_sorted();
    assert_eq!(
        entries.len(),
        row_counts.len(),
        "fixture must be multi-shard"
    );

    // i64 indices + f64 values: the widest arms, so a chunk boundary computed in
    // elements rather than bytes (or vice versa) cannot cancel out.
    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::F64,
        index_dtype: IndexDtype::I64,
        allow_lossy: false,
    };

    let sequential = reader
        .assemble_shards_typed_with(&entries, n_vars, &plan, RowMajorStrategy::Sequential)
        .unwrap();
    let parallel = reader
        .assemble_shards_typed_with(&entries, n_vars, &plan, RowMajorStrategy::Parallel)
        .unwrap();

    assert_eq!(sequential.shape, parallel.shape);
    assert_eq!(sequential.indptr, parallel.indptr);
    match (&sequential.indices, &parallel.indices) {
        (IndexBuffer::I64(a), IndexBuffer::I64(b)) => assert_eq!(a, b, "indices differ"),
        _ => panic!("expected I64 index arms"),
    }
    match (&sequential.values, &parallel.values) {
        (ValueBuffer::F64(a), ValueBuffer::F64(b)) => assert_eq!(a, b, "values differ"),
        _ => panic!("expected F64 value arms"),
    }

    // And both agree with the f32 assembler, so "identical" is not "identically
    // wrong" — a carve that dropped the last shard would agree with itself.
    let f32_csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(parallel.indptr, f32_csr.indptr);
    let want_ix: Vec<i64> = checked_cast_indices(&f32_csr.indices, false).unwrap();
    let want_v: Vec<f64> = checked_cast_values(&f32_csr.data, false).unwrap();
    match (&parallel.indices, &parallel.values) {
        (IndexBuffer::I64(ix), ValueBuffer::F64(v)) => {
            assert_eq!(ix, &want_ix);
            assert_eq!(v, &want_v);
        }
        _ => panic!("wrong buffer arms"),
    }
    assert!(
        want_v.iter().all(|v| *v != 0.0),
        "fixture must have no zero values, or an unwritten slot reads as correct"
    );
}

/// The fan-out is bounded by bytes in flight, not by the pool's width.
///
/// Each task materializes one whole shard's native buffers, so an unbounded
/// `into_par_iter` over 59 shards on a 64-thread pool holds all 59 at once —
/// measured as +6.3 GiB of process high-water on a 2.65e9-nnz file, and zero at
/// 4 threads. The window is computed from the widest shard because rayon may
/// run any subset of a batch simultaneously.
#[test]
#[cfg(feature = "parallel")]
fn the_in_flight_window_is_sized_from_the_widest_shard() {
    use crate::typed_read::{in_flight_batches, native_shard_bytes, IN_FLIGHT_NATIVE_BUDGET_BYTES};

    // 4 B of index + 4 B of value per nonzero, plus this shard's own i64 indptr.
    assert_eq!(native_shard_bytes((10, 100)), 100 * 8 + 11 * 8);

    // Shards small enough that the budget does not bind: one batch.
    let small = vec![(1_000usize, 1_000usize); 8];
    let batches = in_flight_batches((0..8).collect::<Vec<_>>(), &small);
    assert_eq!(batches.len(), 1, "a small shard list must not be split");
    assert_eq!(batches[0].len(), 8);

    // One shard at just under half the budget: two at a time, so four batches of
    // two. (`- 8` leaves room for the shard's own one-element indptr, which
    // `native_shard_bytes` also counts — at exactly half it is one over and the
    // window collapses to 1, which is the boundary being pinned.)
    let half = ((IN_FLIGHT_NATIVE_BUDGET_BYTES / 2 - 8) / 8) as usize;
    let big = vec![(0usize, half); 8];
    let batches = in_flight_batches((0..8).collect::<Vec<_>>(), &big);
    assert_eq!(batches.len(), 4);
    assert!(batches.iter().all(|b| b.len() == 2), "{batches:?}");

    // The indices survive the regrouping, in order — they select the shard.
    let flat: Vec<usize> = batches.iter().flatten().map(|(i, _)| *i).collect();
    assert_eq!(flat, (0..8).collect::<Vec<_>>());

    // A single shard larger than the whole budget still gets decoded.
    let huge = vec![(0usize, IN_FLIGHT_NATIVE_BUDGET_BYTES as usize); 3];
    let batches = in_flight_batches((0..3).collect::<Vec<_>>(), &huge);
    assert_eq!(batches.len(), 3);
    assert!(batches.iter().all(|b| b.len() == 1));
}

// -----------------------------------------------------------------------
// The multimodal tripwire on the typed whole-matrix reads
// -----------------------------------------------------------------------

/// The typed twin of `read_all_csr_shards` has the same hazard and had no
/// guard: `shards_sorted()` is the flattened, all-modality list, so a
/// multimodal file assembled into one `TypedCsr` stacked both modalities'
/// tilings of `[0, n_obs)` — a 6-row result for a 3-cell file, at the
/// file-wide maximum width.
///
/// It is reachable from `to_anndata(data_dtype=…)`, which is why it matters
/// that it was not on the review's list of sites.
#[test]
fn typed_whole_matrix_reads_refuse_an_overlapping_shard_tiling() {
    let dir = tempfile::tempdir().unwrap();
    let modalities: Vec<ModalitySpec> = vec![
        ("rna", ModalityType::Rna, 10, vec![u16_shard(3, 10, 0, 300)]),
        (
            "atac",
            ModalityType::Atac,
            6,
            vec![u16_shard(3, 6, 0, 1000)],
        ),
    ];
    let path = write_multimodal_file(&dir, "mm_typed.scx", 3, &modalities, CodecId::None, None);
    let reader = ScxReader::open(&path).unwrap();
    assert!(reader.is_multimodal(), "fixture premise");

    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let err = reader
        .read_all_csr_shards_typed(&plan)
        .expect_err("a typed whole-matrix read of a two-modality file is not well defined");
    assert!(
        matches!(err, crate::ScxError::MultimodalRequiresModality { .. }),
        "expected MultimodalRequiresModality, got {err:?}"
    );

    // The dense twin builds on the CSR one, so it inherits the refusal rather
    // than scattering a doubled row axis into a zeroed buffer.
    let dense_plan = MaterializePlan {
        container: Container::Dense,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    let dense_err = reader
        .read_all_csr_shards_dense_typed(&dense_plan)
        .expect_err("the dense typed read must refuse what the CSR one refuses");
    assert!(
        matches!(
            dense_err,
            crate::ScxError::MultimodalRequiresModality { .. }
        ),
        "expected MultimodalRequiresModality, got {dense_err:?}"
    );

    // The scoped typed read is unaffected: it never touched the flat list.
    let rna_id = reader.modality_id("rna").unwrap();
    let rna = reader.read_all_csr_shards_for_typed(rna_id, &plan).unwrap();
    assert_eq!(rna.shape, (3, 10));
}

/// The accept side: a single-modality file still reads typed, at both
/// containers, so the guard above is not simply refusing everything.
#[test]
fn typed_whole_matrix_reads_accept_a_single_tiling() {
    let dir = tempfile::tempdir().unwrap();
    let modalities: Vec<ModalitySpec> = vec![(
        "rna",
        ModalityType::Rna,
        10,
        vec![u16_shard(3, 10, 0, 300), u16_shard(3, 10, 3, 1000)],
    )];
    let path = write_multimodal_file(&dir, "uni_typed.scx", 6, &modalities, CodecId::None, None);
    let reader = ScxReader::open(&path).unwrap();

    let plan = MaterializePlan {
        container: Container::Csr,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    assert_eq!(
        reader.read_all_csr_shards_typed(&plan).unwrap().shape,
        (6, 10)
    );
    let dense_plan = MaterializePlan {
        container: Container::Dense,
        data_dtype: ValueDtype::U16,
        index_dtype: IndexDtype::I32,
        allow_lossy: false,
    };
    assert_eq!(
        reader
            .read_all_csr_shards_dense_typed(&dense_plan)
            .unwrap()
            .shape,
        (6, 10)
    );
}
