//! Property-based tests for codec round-trips (Task 17.8).
//!
//! Generates random CSR matrices and verifies that encode → decode
//! is bit-exact for every codec × value encoding combination.

use proptest::prelude::*;
use scx_codec::{decode_shard, encode_shard, CodecId, ValueEncoding};

use scx_codec::delta_golomb::{delta_golomb_decode, delta_golomb_encode};
use scx_codec::forbp::{forbp_decode, forbp_encode};
use scx_codec::rice::{rice_decode, rice_encode, B_VAL};

// =========================================================================
// Strategies
// =========================================================================

/// Generate a random non-zero u32 value in [1, max] for Rice encoding.
fn nonzero_val(max: u32) -> impl Strategy<Value = u32> {
    1..=max
}

/// Generate a random CSR matrix as (indptr, indices, values_u8).
/// Returns (indptr, indices, values_raw, n_rows, nnz, index_dtype_u16).
fn arb_csr(
    max_rows: usize,
    max_nnz_per_row: usize,
    encoding: ValueEncoding,
) -> impl Strategy<Value = (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize, bool)> {
    (1..=max_rows).prop_flat_map(move |n_rows| {
        prop::collection::vec(0..=max_nnz_per_row, n_rows).prop_flat_map(move |row_nnzs| {
            let total_nnz: usize = row_nnzs.iter().sum();
            let n_vars = 1000u32; // use u16 indices
            let index_u16 = n_vars <= 65535;

            // Generate sorted indices per row
            let idx_strats: Vec<_> = row_nnzs
                .iter()
                .map(|&nnz| {
                    if nnz == 0 {
                        Just(vec![]).boxed()
                    } else {
                        prop::collection::hash_set(0..n_vars, nnz)
                            .prop_map(|set| {
                                let mut v: Vec<u32> = set.into_iter().collect();
                                v.sort_unstable();
                                v
                            })
                            .boxed()
                    }
                })
                .collect();

            let val_max = match encoding {
                ValueEncoding::Uint8 => 255u32,
                ValueEncoding::Uint16 => 65535,
                ValueEncoding::Uint32 => 100_000,
                _ => 255,
            };

            (
                idx_strats,
                prop::collection::vec(nonzero_val(val_max), total_nnz),
            )
                .prop_map(move |(row_indices_vecs, values_u32)| {
                    let mut indptr = Vec::with_capacity(n_rows + 1);
                    let mut all_indices = Vec::new();
                    indptr.push(0u64);

                    for row_idx in &row_indices_vecs {
                        all_indices.extend_from_slice(row_idx);
                        indptr.push(indptr.last().unwrap() + row_idx.len() as u64);
                    }

                    let nnz = all_indices.len();
                    let values_raw = match encoding {
                        ValueEncoding::Uint8 => {
                            values_u32.iter().take(nnz).map(|&v| v as u8).collect()
                        }
                        ValueEncoding::Uint16 => {
                            let mut buf = Vec::with_capacity(nnz * 2);
                            for &v in values_u32.iter().take(nnz) {
                                buf.extend_from_slice(&(v as u16).to_le_bytes());
                            }
                            buf
                        }
                        ValueEncoding::Uint32 => {
                            let mut buf = Vec::with_capacity(nnz * 4);
                            for &v in values_u32.iter().take(nnz) {
                                buf.extend_from_slice(&v.to_le_bytes());
                            }
                            buf
                        }
                        _ => values_u32.iter().take(nnz).map(|&v| v as u8).collect(),
                    };

                    (indptr, all_indices, values_raw, n_rows, nnz, index_u16)
                })
        })
    })
}

// =========================================================================
// Rice codec property tests
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn rice_roundtrip(values in prop::collection::vec(1..10000u32, 1..500)) {
        let encoded = rice_encode(&values, B_VAL).unwrap();
        let decoded = rice_decode(&encoded, values.len(), B_VAL).unwrap();
        prop_assert_eq!(decoded, values);
    }
}

// =========================================================================
// Delta-Golomb codec property tests
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn delta_golomb_roundtrip(n_rows in 1..200usize) {
        // Generate random monotonic indptr
        let mut indptr = Vec::with_capacity(n_rows + 1);
        indptr.push(0u64);
        let mut rng_state = 0xDEAD_BEEFu64.wrapping_add(n_rows as u64);
        for _ in 0..n_rows {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let delta = (rng_state >> 33) % 500;
            indptr.push(indptr.last().unwrap() + delta);
        }
        let encoded = delta_golomb_encode(&indptr);
        let decoded = delta_golomb_decode(&encoded, indptr.len()).unwrap();
        prop_assert_eq!(decoded, indptr);
    }
}

// =========================================================================
// FOR-BP codec property tests
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn forbp_roundtrip(n_rows in 1..200usize) {
        let mut indices = Vec::new();
        let mut row_lengths = Vec::new();
        let mut rng_state = 0xCAFE_BABEu64.wrapping_add(n_rows as u64);

        for _ in 0..n_rows {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let nnz = ((rng_state >> 33) % 20) as usize;
            let mut prev = 0u32;
            for _ in 0..nnz {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                prev += ((rng_state >> 33) % 50 + 1) as u32;
                indices.push(prev);
            }
            row_lengths.push(nnz);
        }

        let encoded = forbp_encode(&indices, &row_lengths, true).unwrap();
        let (dec_indices, dec_row_lengths) = forbp_decode(&encoded, n_rows, true).unwrap();
        prop_assert_eq!(dec_indices, indices);
        prop_assert_eq!(dec_row_lengths, row_lengths);
    }
}

// =========================================================================
// Full dispatch round-trip property tests
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn dispatch_none_u8_roundtrip(
        data in arb_csr(50, 20, ValueEncoding::Uint8)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::None, ValueEncoding::Uint8, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::None, ValueEncoding::Uint8, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_scx1_u8_roundtrip(
        data in arb_csr(50, 20, ValueEncoding::Uint8)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Scx1, ValueEncoding::Uint8, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Uint8, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_scx1_u16_roundtrip(
        data in arb_csr(50, 20, ValueEncoding::Uint16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Scx1, ValueEncoding::Uint16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Uint16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_zstd_u32_roundtrip(
        data in arb_csr(50, 20, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Zstd, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Zstd, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    /// Regression guard for Phase 1B: mixed per-shard encodings through
    /// the full encode→decode roundtrip. Shard 1 uses uint8, shard 2 uses
    /// uint16 — both with Scx1 codec.
    #[test]
    fn dispatch_scx1_mixed_u8_u16_roundtrip(
        data_u8 in arb_csr(25, 20, ValueEncoding::Uint8),
        data_u16 in arb_csr(25, 20, ValueEncoding::Uint16),
    ) {
        // Shard 1: uint8 encoding
        let (ip1, ix1, v1, nr1, nnz1, u16_1) = data_u8;
        let enc1 = encode_shard(&ip1, &ix1, &v1, CodecId::Scx1, ValueEncoding::Uint8, u16_1).unwrap();
        let (d_ip1, d_ix1, d_v1) = decode_shard(&enc1, CodecId::Scx1, ValueEncoding::Uint8, nr1, nnz1, u16_1).unwrap();
        prop_assert_eq!(&d_ip1, &ip1);
        prop_assert_eq!(&d_ix1, &ix1);
        prop_assert_eq!(&d_v1, &v1);

        // Shard 2: uint16 encoding
        let (ip2, ix2, v2, nr2, nnz2, u16_2) = data_u16;
        let enc2 = encode_shard(&ip2, &ix2, &v2, CodecId::Scx1, ValueEncoding::Uint16, u16_2).unwrap();
        let (d_ip2, d_ix2, d_v2) = decode_shard(&enc2, CodecId::Scx1, ValueEncoding::Uint16, nr2, nnz2, u16_2).unwrap();
        prop_assert_eq!(&d_ip2, &ip2);
        prop_assert_eq!(&d_ix2, &ix2);
        prop_assert_eq!(&d_v2, &v2);
    }

    /// Mixed encoding with Zstd codec (uint8 + uint32).
    #[test]
    fn dispatch_zstd_mixed_u8_u32_roundtrip(
        data_u8 in arb_csr(25, 20, ValueEncoding::Uint8),
        data_u32 in arb_csr(25, 20, ValueEncoding::Uint32),
    ) {
        let (ip1, ix1, v1, nr1, nnz1, u16_1) = data_u8;
        let enc1 = encode_shard(&ip1, &ix1, &v1, CodecId::Zstd, ValueEncoding::Uint8, u16_1).unwrap();
        let (d_ip1, d_ix1, d_v1) = decode_shard(&enc1, CodecId::Zstd, ValueEncoding::Uint8, nr1, nnz1, u16_1).unwrap();
        prop_assert_eq!(&d_ip1, &ip1);
        prop_assert_eq!(&d_ix1, &ix1);
        prop_assert_eq!(&d_v1, &v1);

        let (ip2, ix2, v2, nr2, nnz2, u16_2) = data_u32;
        let enc2 = encode_shard(&ip2, &ix2, &v2, CodecId::Zstd, ValueEncoding::Uint32, u16_2).unwrap();
        let (d_ip2, d_ix2, d_v2) = decode_shard(&enc2, CodecId::Zstd, ValueEncoding::Uint32, nr2, nnz2, u16_2).unwrap();
        prop_assert_eq!(&d_ip2, &ip2);
        prop_assert_eq!(&d_ix2, &ix2);
        prop_assert_eq!(&d_v2, &v2);
    }
}
