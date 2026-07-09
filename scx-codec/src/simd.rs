//! x86_64 SIMD kernels for the ShufDeltaZstd decode byte-transforms (Phase B).
//!
//! Two hot decode transforms — the per-plane wrapping-`u8` prefix scan
//! ([`crate::byte_delta::byte_undelta_planes`]) and the plane→element transpose
//! ([`crate::shuffle::byte_unshuffle`]) — are serial scalar and run at
//! ~0.4–0.7 GB/s, dominating ShufDeltaZstd CPU decode. These 128-bit kernels
//! replace the inner loops.
//!
//! **Baseline, not opt-in.** SSE2 is part of the x86_64 ABI baseline, so these
//! intrinsics are always available at runtime on any x86_64 CPU — no
//! `#[target_feature]` gate or `is_x86_feature_detected!` probe is required for
//! correctness. The whole module is `#[cfg(target_arch = "x86_64")]`; every
//! other architecture uses the scalar reference. Each kernel is a drop-in that
//! must be **bit-identical** to its scalar counterpart (enforced by
//! `simd == scalar` proptests in the sibling modules).
//!
//! A wider AVX2 (256-bit) variant is a future enhancement: it would sit behind
//! `is_x86_feature_detected!("avx2")` here, but 256-bit lanes complicate the
//! cross-lane byte shift (prefix scan) and the per-128-lane `unpack` (transpose)
//! for marginal gain, so the SSE2 kernels are the Phase-B implementation.
#![cfg(target_arch = "x86_64")]

use core::arch::x86_64::*;

/// In-place per-plane wrapping-`u8` inclusive prefix sum — SSE2 kernel matching
/// [`crate::byte_delta::byte_undelta_planes_scalar`] byte-for-byte.
///
/// `buf` is plane-major: plane `p` occupies `buf[p*n_elems .. (p+1)*n_elems]`.
/// Within each plane, computes `buf[e] = sum(buf[0..=e]) (mod 256)`. Caller
/// guarantees `buf.len() == n_planes * n_elems` and `n_elems >= 2`.
pub(crate) fn undelta_planes(buf: &mut [u8], n_planes: usize, n_elems: usize) {
    debug_assert_eq!(buf.len(), n_planes * n_elems);
    for p in 0..n_planes {
        let base = p * n_elems;
        // SAFETY: `base + n_elems <= buf.len()`; all `_mm_*` here are SSE2,
        // guaranteed on x86_64. Loads/stores are unaligned (`loadu`/`storeu`).
        unsafe { undelta_one_plane(&mut buf[base..base + n_elems]) };
    }
}

/// Prefix-sum a single contiguous plane in place (wrapping `u8`).
///
/// # Safety
/// x86_64 baseline (SSE2) only; `plane` must be a valid `&mut [u8]`.
#[inline]
unsafe fn undelta_one_plane(plane: &mut [u8]) {
    let n = plane.len();
    let ptr = plane.as_mut_ptr();
    let mut carry: u8 = 0;
    let mut e = 0usize;
    while e + 16 <= n {
        let p = ptr.add(e) as *mut __m128i;
        let mut x = _mm_loadu_si128(p as *const __m128i);
        // Log-step (Hillis–Steele) inclusive prefix sum over the 16 byte lanes.
        // `_mm_slli_si128` shifts the whole register left by N *bytes*, filling
        // zeros, so after shifts of 1,2,4,8 lane i holds sum(lanes 0..=i).
        x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
        x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
        // Fold in the running total from all previous blocks.
        x = _mm_add_epi8(x, _mm_set1_epi8(carry as i8));
        _mm_storeu_si128(p, x);
        // New running total = last lane of this block (read back from memory to
        // avoid an SSE4.1 `_mm_extract_epi8`).
        carry = *ptr.add(e + 15);
        e += 16;
    }
    // Scalar tail: continue the inclusive scan from `carry`.
    let mut prev = carry;
    while e < n {
        prev = prev.wrapping_add(*ptr.add(e));
        *ptr.add(e) = prev;
        e += 1;
    }
}

/// Plane→element transpose (byte-unshuffle) for `element_width ∈ {2,4,8}` —
/// SSE2 kernel matching [`crate::shuffle::byte_unshuffle_scalar`] byte-for-byte.
///
/// `input` is plane-major (`w` contiguous planes of `n = input.len()/w` bytes);
/// output is element-major (`out[i*w + b] = input[b*n + i]`). Caller guarantees
/// `input.len() % w == 0` and `w ∈ {2,4,8}`.
pub(crate) fn unshuffle(input: &[u8], w: usize) -> Vec<u8> {
    debug_assert!(w == 2 || w == 4 || w == 8);
    debug_assert_eq!(input.len() % w, 0);
    let n = input.len() / w;
    let mut out = vec![0u8; input.len()];
    // SAFETY: bounds established below (16-element blocks stay within each
    // plane and the output); all intrinsics are SSE2.
    unsafe {
        match w {
            2 => unshuffle_w2(input, &mut out, n),
            4 => unshuffle_w4(input, &mut out, n),
            8 => unshuffle_w8(input, &mut out, n),
            _ => unreachable!("unshuffle SIMD called with unsupported width"),
        }
    }
    out
}

/// Scalar interleave of `input[b*n + i]` → `out[i*w + b]` for elements
/// `[start, n)`. Used for the sub-16 tail of every width.
#[inline]
fn unshuffle_tail(input: &[u8], out: &mut [u8], n: usize, w: usize, start: usize) {
    for i in start..n {
        for b in 0..w {
            out[i * w + b] = input[b * n + i];
        }
    }
}

/// # Safety
/// SSE2 only. `input`/`out` sized for `n` elements of width 2.
#[inline]
unsafe fn unshuffle_w2(input: &[u8], out: &mut [u8], n: usize) {
    let p0 = input.as_ptr();
    let p1 = input.as_ptr().add(n);
    let o = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let a = _mm_loadu_si128(p0.add(i) as *const __m128i);
        let b = _mm_loadu_si128(p1.add(i) as *const __m128i);
        let lo = _mm_unpacklo_epi8(a, b); // elements i..i+8
        let hi = _mm_unpackhi_epi8(a, b); // elements i+8..i+16
        _mm_storeu_si128(o.add(i * 2) as *mut __m128i, lo);
        _mm_storeu_si128(o.add(i * 2 + 16) as *mut __m128i, hi);
        i += 16;
    }
    unshuffle_tail(input, out, n, 2, i);
}

/// # Safety
/// SSE2 only. `input`/`out` sized for `n` elements of width 4.
#[inline]
unsafe fn unshuffle_w4(input: &[u8], out: &mut [u8], n: usize) {
    let p0 = input.as_ptr();
    let p1 = input.as_ptr().add(n);
    let p2 = input.as_ptr().add(2 * n);
    let p3 = input.as_ptr().add(3 * n);
    let o = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let a = _mm_loadu_si128(p0.add(i) as *const __m128i);
        let b = _mm_loadu_si128(p1.add(i) as *const __m128i);
        let c = _mm_loadu_si128(p2.add(i) as *const __m128i);
        let d = _mm_loadu_si128(p3.add(i) as *const __m128i);
        let ab_lo = _mm_unpacklo_epi8(a, b);
        let ab_hi = _mm_unpackhi_epi8(a, b);
        let cd_lo = _mm_unpacklo_epi8(c, d);
        let cd_hi = _mm_unpackhi_epi8(c, d);
        // Interleave (a,b) with (c,d) at 16-bit granularity → [a,b,c,d] per elem.
        let o0 = _mm_unpacklo_epi16(ab_lo, cd_lo); // elements i..i+4
        let o1 = _mm_unpackhi_epi16(ab_lo, cd_lo); // i+4..i+8
        let o2 = _mm_unpacklo_epi16(ab_hi, cd_hi); // i+8..i+12
        let o3 = _mm_unpackhi_epi16(ab_hi, cd_hi); // i+12..i+16
        let base = i * 4;
        _mm_storeu_si128(o.add(base) as *mut __m128i, o0);
        _mm_storeu_si128(o.add(base + 16) as *mut __m128i, o1);
        _mm_storeu_si128(o.add(base + 32) as *mut __m128i, o2);
        _mm_storeu_si128(o.add(base + 48) as *mut __m128i, o3);
        i += 16;
    }
    unshuffle_tail(input, out, n, 4, i);
}

/// # Safety
/// SSE2 only. `input`/`out` sized for `n` elements of width 8.
#[inline]
unsafe fn unshuffle_w8(input: &[u8], out: &mut [u8], n: usize) {
    let pp: [*const u8; 8] = [
        input.as_ptr(),
        input.as_ptr().add(n),
        input.as_ptr().add(2 * n),
        input.as_ptr().add(3 * n),
        input.as_ptr().add(4 * n),
        input.as_ptr().add(5 * n),
        input.as_ptr().add(6 * n),
        input.as_ptr().add(7 * n),
    ];
    let o = out.as_mut_ptr();
    let mut i = 0usize;
    while i + 16 <= n {
        let a = _mm_loadu_si128(pp[0].add(i) as *const __m128i);
        let b = _mm_loadu_si128(pp[1].add(i) as *const __m128i);
        let c = _mm_loadu_si128(pp[2].add(i) as *const __m128i);
        let d = _mm_loadu_si128(pp[3].add(i) as *const __m128i);
        let e = _mm_loadu_si128(pp[4].add(i) as *const __m128i);
        let f = _mm_loadu_si128(pp[5].add(i) as *const __m128i);
        let g = _mm_loadu_si128(pp[6].add(i) as *const __m128i);
        let h = _mm_loadu_si128(pp[7].add(i) as *const __m128i);
        // Level 1: 8-bit interleave within each pair.
        let ab_lo = _mm_unpacklo_epi8(a, b);
        let ab_hi = _mm_unpackhi_epi8(a, b);
        let cd_lo = _mm_unpacklo_epi8(c, d);
        let cd_hi = _mm_unpackhi_epi8(c, d);
        let ef_lo = _mm_unpacklo_epi8(e, f);
        let ef_hi = _mm_unpackhi_epi8(e, f);
        let gh_lo = _mm_unpacklo_epi8(g, h);
        let gh_hi = _mm_unpackhi_epi8(g, h);
        // Level 2: 16-bit interleave → (a,b,c,d) and (e,f,g,h) quads.
        let abcd0 = _mm_unpacklo_epi16(ab_lo, cd_lo);
        let abcd1 = _mm_unpackhi_epi16(ab_lo, cd_lo);
        let abcd2 = _mm_unpacklo_epi16(ab_hi, cd_hi);
        let abcd3 = _mm_unpackhi_epi16(ab_hi, cd_hi);
        let efgh0 = _mm_unpacklo_epi16(ef_lo, gh_lo);
        let efgh1 = _mm_unpackhi_epi16(ef_lo, gh_lo);
        let efgh2 = _mm_unpacklo_epi16(ef_hi, gh_hi);
        let efgh3 = _mm_unpackhi_epi16(ef_hi, gh_hi);
        // Level 3: 32-bit interleave → full 8-byte elements [a,b,c,d,e,f,g,h].
        let base = i * 8;
        _mm_storeu_si128(
            o.add(base) as *mut __m128i,
            _mm_unpacklo_epi32(abcd0, efgh0),
        );
        _mm_storeu_si128(
            o.add(base + 16) as *mut __m128i,
            _mm_unpackhi_epi32(abcd0, efgh0),
        );
        _mm_storeu_si128(
            o.add(base + 32) as *mut __m128i,
            _mm_unpacklo_epi32(abcd1, efgh1),
        );
        _mm_storeu_si128(
            o.add(base + 48) as *mut __m128i,
            _mm_unpackhi_epi32(abcd1, efgh1),
        );
        _mm_storeu_si128(
            o.add(base + 64) as *mut __m128i,
            _mm_unpacklo_epi32(abcd2, efgh2),
        );
        _mm_storeu_si128(
            o.add(base + 80) as *mut __m128i,
            _mm_unpackhi_epi32(abcd2, efgh2),
        );
        _mm_storeu_si128(
            o.add(base + 96) as *mut __m128i,
            _mm_unpacklo_epi32(abcd3, efgh3),
        );
        _mm_storeu_si128(
            o.add(base + 112) as *mut __m128i,
            _mm_unpackhi_epi32(abcd3, efgh3),
        );
        i += 16;
    }
    unshuffle_tail(input, out, n, 8, i);
}
