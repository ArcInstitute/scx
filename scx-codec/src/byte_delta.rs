//! Per-plane byte-delta filter for the ShufDeltaZstd codec (F5).
//!
//! These operate on the **plane-major** output of [`crate::shuffle::byte_shuffle`],
//! where a buffer of `n_planes × n_elems` bytes stores each byte-plane
//! contiguously (all byte-0s of every element, then all byte-1s, …). Within a
//! plane the delta runs along the element axis with wrapping `u8` arithmetic;
//! the first element of each plane is kept raw. Because CSR column indices are
//! sorted within a row and indptr is monotonic, after byte-shuffling each byte
//! lane, adjacent values differ by small predictable amounts, so the delta
//! turns each plane into a near-constant stream that zstd crushes.
//!
//! Applied to `indices`/`indptr` only — count `data` is effectively random, so
//! delta would *increase* entropy and is skipped (see the codec pipeline in
//! `dispatch.rs`).

/// In-place per-plane wrapping-`u8` delta along the element axis.
///
/// `buf` is plane-major (`buf[plane * n_elems + elem]`); `buf.len()` must equal
/// `n_planes * n_elems`. The first element of each plane is left raw; each later
/// element becomes `elem - prev` (`wrapping_sub`). A no-op for `n_elems <= 1`.
pub fn byte_delta_planes(buf: &mut [u8], n_planes: usize, n_elems: usize) {
    if n_elems <= 1 {
        return;
    }
    debug_assert_eq!(buf.len(), n_planes * n_elems);
    for p in 0..n_planes {
        let base = p * n_elems;
        // Walk high → low so each subtraction still reads the ORIGINAL previous
        // byte (matches numpy's `s[:, 1:] - s[:, :-1]` on the un-mutated array).
        for e in (1..n_elems).rev() {
            buf[base + e] = buf[base + e].wrapping_sub(buf[base + e - 1]);
        }
    }
}

/// Inverse of [`byte_delta_planes`]: in-place per-plane wrapping-`u8` cumulative
/// sum along the element axis. `buf.len()` must equal `n_planes * n_elems`.
///
/// On x86_64 this dispatches to the SSE2 kernel `simd::undelta_planes`
/// (crate-private, baseline-guaranteed, bit-identical); every other target uses the scalar
/// reference below.
pub fn byte_undelta_planes(buf: &mut [u8], n_planes: usize, n_elems: usize) {
    if n_elems <= 1 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    crate::simd::undelta_planes(buf, n_planes, n_elems);
    #[cfg(not(target_arch = "x86_64"))]
    byte_undelta_planes_scalar(buf, n_planes, n_elems);
}

/// Scalar reference for [`byte_undelta_planes`]: in-place per-plane wrapping-`u8`
/// cumulative sum. Kept as the correctness oracle for the SIMD kernel (see the
/// `simd == scalar` proptest) and used directly on non-x86_64 targets.
// On x86_64 the SIMD kernel always supersedes this in the lib build; it is still
// reached by the proptest oracle (cfg(test)) and by non-x86_64 targets.
#[cfg_attr(target_arch = "x86_64", allow(dead_code))]
pub(crate) fn byte_undelta_planes_scalar(buf: &mut [u8], n_planes: usize, n_elems: usize) {
    if n_elems <= 1 {
        return;
    }
    debug_assert_eq!(buf.len(), n_planes * n_elems);
    for p in 0..n_planes {
        let base = p * n_elems;
        for e in 1..n_elems {
            buf[base + e] = buf[base + e].wrapping_add(buf[base + e - 1]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn delta_undelta_roundtrip_small() {
        // 2 planes × 4 elems, plane-major.
        let orig: Vec<u8> = vec![
            10, 12, 12, 20, /* plane 0 */ 0, 255, 1, 3, /* plane 1 */
        ];
        let mut buf = orig.clone();
        byte_delta_planes(&mut buf, 2, 4);
        // plane 0 deltas: 10, +2, 0, +8  → [10, 2, 0, 8]
        assert_eq!(&buf[0..4], &[10, 2, 0, 8]);
        // plane 1 deltas: 0, 255, 2 (1-255 wraps), 2 (3-1)
        assert_eq!(&buf[4..8], &[0, 255, 2, 2]);
        byte_undelta_planes(&mut buf, 2, 4);
        assert_eq!(buf, orig);
    }

    #[test]
    fn single_element_plane_is_noop() {
        let orig: Vec<u8> = vec![7, 9, 11];
        let mut buf = orig.clone();
        byte_delta_planes(&mut buf, 3, 1);
        assert_eq!(buf, orig);
        byte_undelta_planes(&mut buf, 3, 1);
        assert_eq!(buf, orig);
    }

    #[test]
    fn empty_is_noop() {
        let mut buf: Vec<u8> = vec![];
        byte_delta_planes(&mut buf, 0, 0);
        byte_undelta_planes(&mut buf, 0, 0);
        assert!(buf.is_empty());
    }

    proptest! {
        #[test]
        fn delta_undelta_roundtrip(
            n_planes in 1usize..=8,
            n_elems in 0usize..=256,
            seed in any::<u64>(),
        ) {
            // Deterministic pseudo-random fill (Math.random-free).
            let mut state = seed;
            let mut buf: Vec<u8> = (0..n_planes * n_elems)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (state >> 33) as u8
                })
                .collect();
            let orig = buf.clone();
            byte_delta_planes(&mut buf, n_planes, n_elems);
            byte_undelta_planes(&mut buf, n_planes, n_elems);
            prop_assert_eq!(buf, orig);
        }

        /// Phase B: the SIMD `byte_undelta_planes` (dispatched on x86_64) must be
        /// bit-identical to the scalar reference across the real plane counts
        /// (2/4/8) and lengths that straddle the 16-byte SIMD block + tail.
        #[test]
        fn undelta_simd_matches_scalar(
            n_planes in prop::sample::select(vec![2usize, 4, 8]),
            n_elems in 0usize..=300,
            seed in any::<u64>(),
        ) {
            let mut state = seed;
            let buf: Vec<u8> = (0..n_planes * n_elems)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    (state >> 33) as u8
                })
                .collect();
            let mut simd_buf = buf.clone();
            let mut scalar_buf = buf;
            byte_undelta_planes(&mut simd_buf, n_planes, n_elems);
            byte_undelta_planes_scalar(&mut scalar_buf, n_planes, n_elems);
            prop_assert_eq!(simd_buf, scalar_buf);
        }
    }
}
