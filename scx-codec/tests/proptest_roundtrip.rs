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

/// Column counts spanning the u16/u32 index-dtype boundary.
///
/// The writer picks `index_dtype_u16` per shard from `index_max_value <=
/// u16::MAX` where `index_max_value = n_vars - 1`
/// (`scx-format-io/src/writer.rs`), so the boundary sits at **65_537 columns**,
/// not 65_536: a 65_536-column matrix's largest index is 65_535, which still
/// fits u16. Both exact sides are sampled.
fn arb_n_vars() -> impl Strategy<Value = u32> {
    prop_oneof![
        Just(1000u32),    // the historical value
        Just(65_536),     // largest n_vars still using u16 indices
        Just(65_537),     // smallest n_vars needing u32 indices
        Just(200_000),    // comfortably u32
        1000u32..150_000, // and a spread across the boundary
    ]
}

/// Generate a random CSR matrix as (indptr, indices, values_u8).
/// Returns (indptr, indices, values_raw, n_rows, nnz, index_dtype_u16).
///
/// `n_vars` is sampled from [`arb_n_vars`] rather than fixed, so every property
/// built on this strategy sees both index dtypes. Per-row nnz is clamped to
/// `n_vars` because the index generator draws *distinct* columns.
fn arb_csr(
    max_rows: usize,
    max_nnz_per_row: usize,
    encoding: ValueEncoding,
) -> impl Strategy<Value = (Vec<u64>, Vec<u32>, Vec<u8>, usize, usize, bool)> {
    (1..=max_rows, arb_n_vars()).prop_flat_map(move |(n_rows, n_vars)| {
        let row_cap = max_nnz_per_row.min(n_vars as usize);
        prop::collection::vec(0..=row_cap, n_rows).prop_flat_map(move |row_nnzs| {
            let total_nnz: usize = row_nnzs.iter().sum();
            // Mirrors the writer: the *maximum index*, not the column count.
            let index_u16 = (n_vars as usize).saturating_sub(1) <= u16::MAX as usize;

            // Sorted distinct indices *by construction*, via strictly positive
            // gaps — never by rejection.
            //
            // The obvious `hash_set(0..n_vars, nnz)` draws `nnz` columns and
            // rejects the whole row if any two collide, so its cost is the
            // birthday problem: P(no collision) ≈ exp(-nnz² / 2·n_vars). That
            // is ~0.82 at the old nnz ≤ 20 / n_vars = 1000, and ~2e-9 once nnz
            // reaches 200 — every draw rejected, the property never runs. Gaps
            // of `1..=n_vars/nnz` keep the last index below `n_vars` while
            // costing one draw per element.
            let idx_strats: Vec<_> = row_nnzs
                .iter()
                .map(|&nnz| {
                    // `checked_div` rather than an `if nnz == 0` guard around
                    // `n_vars / nnz`: clippy 1.97's `manual_checked_ops` rejects
                    // the latter, and CI's toolchain is newer than the dev box's.
                    match (n_vars as usize).checked_div(nnz) {
                        None => Just(vec![]).boxed(),
                        Some(gap) => {
                            let max_gap = gap.max(1) as u32;
                            prop::collection::vec(1..=max_gap, nnz)
                                .prop_map(|gaps| {
                                    let mut v = Vec::with_capacity(gaps.len());
                                    let mut cur = 0u32;
                                    for g in gaps {
                                        cur += g;
                                        v.push(cur - 1);
                                    }
                                    v
                                })
                                .boxed()
                        }
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
                        ValueEncoding::Float32 => {
                            let mut buf = Vec::with_capacity(nnz * 4);
                            for &v in values_u32.iter().take(nnz) {
                                buf.extend_from_slice(&(v as f32).to_le_bytes());
                            }
                            buf
                        }
                        ValueEncoding::Float16 => {
                            let mut buf = Vec::with_capacity(nnz * 2);
                            for &v in values_u32.iter().take(nnz) {
                                let f = half::f16::from_f32(v as f32);
                                buf.extend_from_slice(&f.to_le_bytes());
                            }
                            buf
                        }
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
        let encoded = delta_golomb_encode(&indptr).unwrap();
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

    /// Large-row proptest exercising the BitPacker4x SIMD path (NNZ >= 128).
    #[test]
    fn forbp_roundtrip_large_rows(n_rows in 1..50usize) {
        let mut indices = Vec::new();
        let mut row_lengths = Vec::new();
        let mut rng_state = 0xDEAD_BEEFu64.wrapping_add(n_rows as u64);

        for _ in 0..n_rows {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // NNZ up to 300 — most rows will exceed the 128-value SIMD threshold
            let nnz = ((rng_state >> 33) % 300) as usize;
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

// =========================================================================
// Strategy coverage guard
//
// `arb_csr` used to hardcode `n_vars = 1000` and every caller passed
// `max_nnz_per_row = 20`, so across all 18 dispatch properties the u32-index
// branch and FOR-BP's BitPacker4x path (`nnz >= SIMD_THRESHOLD`) were reached
// exactly zero times. Widening the ranges is only half the fix — a range that
// never samples the far side is the same vacuum in a new costume. This asserts
// the generator actually reaches both sides of both boundaries.
// =========================================================================

#[test]
fn arb_csr_reaches_both_index_dtypes_and_both_sides_of_simd_threshold() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::{RngAlgorithm, TestRng, TestRunner};

    // 12 rows keeps each draw cheap — every draw still yields 12 per-row nnz
    // samples, so a few dozen draws give hundreds of SIMD-threshold
    // observations. Building the full 50-row strategy 256 times costs minutes
    // in a debug build, and this guard runs on every `cargo test`.
    let strat = arb_csr(12, 200, ValueEncoding::Uint32);
    // Each draw rejection-samples one index set per row, so the default
    // per-runner local-reject budget (1000) is spent long before the loop ends.
    // Deterministic RNG so the coverage assertion cannot flake.
    let cfg = ProptestConfig {
        max_local_rejects: 1 << 22,
        ..ProptestConfig::default()
    };
    let mut runner =
        TestRunner::new_with_rng(cfg, TestRng::deterministic_rng(RngAlgorithm::ChaCha));

    let (mut u16_idx_seen, mut u32_idx_seen) = (0usize, 0usize);
    let (mut below_simd, mut at_or_above_simd) = (0usize, 0usize);

    for _ in 0..64 {
        let (indptr, _, _, _, _, u16_idx) = strat.new_tree(&mut runner).unwrap().current();
        if u16_idx {
            u16_idx_seen += 1;
        } else {
            u32_idx_seen += 1;
        }
        for w in indptr.windows(2) {
            let row_nnz = (w[1] - w[0]) as usize;
            if row_nnz >= scx_codec::forbp::SIMD_THRESHOLD {
                at_or_above_simd += 1;
            } else {
                below_simd += 1;
            }
        }
    }

    // A tenth, not "at least one": each dispatch property runs only 30 cases,
    // so a branch reached once in 64 draws would still be absent from most
    // properties most runs. Both shares sit near a half by construction, so
    // this has wide margin while still failing if either branch becomes rare.
    let draws = u16_idx_seen + u32_idx_seen;
    assert!(
        u16_idx_seen * 10 >= draws && u32_idx_seen * 10 >= draws,
        "each index dtype must be a real share of draws, got u16={u16_idx_seen} u32={u32_idx_seen} of {draws}"
    );
    let rows = below_simd + at_or_above_simd;
    assert!(
        below_simd * 10 >= rows && at_or_above_simd * 10 >= rows,
        "per-row nnz must straddle SIMD_THRESHOLD={} in both directions, got below={below_simd} at_or_above={at_or_above_simd} of {rows}",
        scx_codec::forbp::SIMD_THRESHOLD
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn dispatch_none_u8_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint8)
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
        data in arb_csr(50, 200, ValueEncoding::Uint8)
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
        data in arb_csr(50, 200, ValueEncoding::Uint16)
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
        data in arb_csr(50, 200, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Zstd, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Zstd, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_lz4shuffle_u8_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint8)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Lz4Shuffle, ValueEncoding::Uint8, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Lz4Shuffle, ValueEncoding::Uint8, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_lz4shuffle_u16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Lz4Shuffle, ValueEncoding::Uint16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Lz4Shuffle, ValueEncoding::Uint16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_lz4shuffle_u32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Lz4Shuffle, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Lz4Shuffle, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    /// Phase 9: float32 roundtrip for the non-integer codecs. Closes
    /// the codec × encoding matrix for the canonical-matrix property.
    #[test]
    fn dispatch_lz4shuffle_f32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Lz4Shuffle, ValueEncoding::Float32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Lz4Shuffle, ValueEncoding::Float32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_zstd_f32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Zstd, ValueEncoding::Float32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Zstd, ValueEncoding::Float32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_none_f32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::None, ValueEncoding::Float32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::None, ValueEncoding::Float32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_pcodec_u8_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint8)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Pcodec, ValueEncoding::Uint8, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Pcodec, ValueEncoding::Uint8, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_pcodec_u16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Pcodec, ValueEncoding::Uint16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Pcodec, ValueEncoding::Uint16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_pcodec_u32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Pcodec, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Pcodec, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_pcodec_f32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Pcodec, ValueEncoding::Float32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Pcodec, ValueEncoding::Float32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    /// F5 ShufDeltaZstd: full codec × encoding matrix (u8/u16/u32 integer
    /// values through the shuffle+delta+zstd path, f32 through the zstd-only
    /// value path).
    #[test]
    fn dispatch_shufdelta_u8_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint8)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::ShufDeltaZstd, ValueEncoding::Uint8, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::ShufDeltaZstd, ValueEncoding::Uint8, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_shufdelta_u16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::ShufDeltaZstd, ValueEncoding::Uint16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::ShufDeltaZstd, ValueEncoding::Uint16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_shufdelta_u32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::ShufDeltaZstd, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::ShufDeltaZstd, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_shufdelta_f32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::ShufDeltaZstd, ValueEncoding::Float32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::ShufDeltaZstd, ValueEncoding::Float32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    /// Regression guard for Phase 1B: mixed per-shard encodings through
    /// the full encode→decode roundtrip. Shard 1 uses uint8, shard 2 uses
    /// uint16 — both with Scx1 codec.
    #[test]
    fn dispatch_scx1_mixed_u8_u16_roundtrip(
        data_u8 in arb_csr(25, 200, ValueEncoding::Uint8),
        data_u16 in arb_csr(25, 200, ValueEncoding::Uint16),
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
        data_u8 in arb_csr(25, 200, ValueEncoding::Uint8),
        data_u32 in arb_csr(25, 200, ValueEncoding::Uint32),
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

// =========================================================================
// Codec × value-encoding cells the matrix above never covered.
//
// `dispatch_scx1_u32` is the one the review names; the rest close cells that
// were simply absent (`None` at u16/u32, `Zstd` at u8/u16). `Float16` was
// passed by no property at all, which made `arb_csr`'s Float16 packer dead
// code and left the codecs' 2-byte-stride paths unexercised here.
// =========================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(30))]

    #[test]
    fn dispatch_scx1_u32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Scx1, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Scx1, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_none_u16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::None, ValueEncoding::Uint16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::None, ValueEncoding::Uint16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_none_u32_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint32)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::None, ValueEncoding::Uint32, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::None, ValueEncoding::Uint32, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_zstd_u8_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint8)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Zstd, ValueEncoding::Uint8, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Zstd, ValueEncoding::Uint8, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_zstd_u16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Uint16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Zstd, ValueEncoding::Uint16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Zstd, ValueEncoding::Uint16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    /// Float16 is lossy on encode (f32 → f16), so `arb_csr` already emits the
    /// narrowed 2-byte payload and the round-trip is byte-exact on that.
    #[test]
    fn dispatch_none_f16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::None, ValueEncoding::Float16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::None, ValueEncoding::Float16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_pcodec_f16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Pcodec, ValueEncoding::Float16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Pcodec, ValueEncoding::Float16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_zstd_f16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Zstd, ValueEncoding::Float16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Zstd, ValueEncoding::Float16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_lz4shuffle_f16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::Lz4Shuffle, ValueEncoding::Float16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::Lz4Shuffle, ValueEncoding::Float16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }

    #[test]
    fn dispatch_shufdelta_f16_roundtrip(
        data in arb_csr(50, 200, ValueEncoding::Float16)
    ) {
        let (indptr, indices, values, n_rows, nnz, u16_idx) = data;
        let encoded = encode_shard(&indptr, &indices, &values, CodecId::ShufDeltaZstd, ValueEncoding::Float16, u16_idx).unwrap();
        let (d_ip, d_ix, d_v) = decode_shard(&encoded, CodecId::ShufDeltaZstd, ValueEncoding::Float16, n_rows, nnz, u16_idx).unwrap();
        prop_assert_eq!(&d_ip, &indptr);
        prop_assert_eq!(&d_ix, &indices);
        prop_assert_eq!(&d_v, &values);
    }
}
