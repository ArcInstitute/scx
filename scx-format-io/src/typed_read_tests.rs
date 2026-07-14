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

fn decode_both(
    codec: CodecId,
    values: &[f32],
    framing: Option<FramingConfig>,
) -> (
    (Vec<i64>, Vec<u32>, ShardValuesNative),
    (Vec<i64>, Vec<i32>, Vec<f32>),
) {
    let indptr = [0u64, 2, 2, 5, 7];
    let indices = [0u32, 3, 1, 4, 9, 2, 8];
    let n_cols: u32 = 16;

    let s = encode_one_shard(
        &indptr,
        &indices,
        values,
        Some(codec),
        0,
        n_cols,
        0,
        SectionType::CsrShard,
        ModalityType::Rna,
        "X_shard_0".to_string(),
        framing,
    )
    .expect("encode_one_shard");
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
    let path = dir.path().join(name);
    let total_nnz: u64 = shards.iter().map(|s| *s.0.last().unwrap()).sum();
    let hdr = header(n_obs as u64, n_vars as u64, total_nnz);
    let mut writer = ScxWriter::new(&path, hdr).unwrap();
    writer.write_obs(&obs_batch(n_obs)).unwrap();
    writer.write_var(&var_batch(n_vars)).unwrap();
    for (indptr, indices, values, enc, row_start) in shards {
        writer
            .write_csr_shard(indptr, indices, values, CodecId::None, *enc, *row_start)
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
    use roaring::RoaringBitmap;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dv.scx");
    let shards = vec![u8_shard(6, 10, 0, 0)];
    let total_nnz: u64 = *shards[0].0.last().unwrap();
    let hdr = header(6, 10, total_nnz);
    let mut writer = ScxWriter::new(&path, hdr).unwrap();
    writer.write_obs(&obs_batch(6)).unwrap();
    writer.write_var(&var_batch(10)).unwrap();
    let (ip, ix, v, enc, rs) = &shards[0];
    writer
        .write_csr_shard(ip, ix, v, CodecId::None, *enc, *rs)
        .unwrap();
    // Delete local rows 1 and 4 of shard 0.
    let mut dv = DeletionVectors::new();
    let mut bm = RoaringBitmap::new();
    bm.insert(1);
    bm.insert(4);
    dv.insert(0, bm);
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
