//! Small shared helpers for the R bindings.

use std::collections::HashMap;

use extendr_api::prelude::*;

/// Surface a fallible binding result to R as a **clean** error condition.
///
/// extendr 0.8.0's default `Robj::from(Result<T, E>)` conversion (the path the
/// `#[extendr]` wrapper takes for a `-> Result<…>` method) calls `.unwrap()` on
/// `Err`, which panics; the macro's `catch_unwind` then re-surfaces that panic
/// to R as the opaque `Error: User function panicked: <fn>`, burying the real
/// message in stderr. `#[extendr]` methods that can fail should instead return
/// a plain `Robj` and route through this helper, which calls `Rf_error` via
/// `throw_r_error` so R sees a normal `stop()` carrying the actual message.
/// See B3 / B7 in the 2026-06-15 user report.
///
/// **Call only at the top `#[extendr]` boundary** (directly in the wrapped
/// method body, as the returned value). On the error path `throw_r_error`
/// invokes `Rf_error`, which `longjmp`s out to R's nearest context — skipping
/// every Rust destructor between here and the `.Call` entry. Invoking it deeper
/// in the stack (with live `CudaSlice`s, file handles, `Box`es, or other RAII
/// guards on the frames it jumps over) would leak or corrupt them. Keep the
/// fallible work in a `-> Result` inner fn and only pass its result here.
pub(crate) fn throw_on_err<T: Into<Robj>>(r: Result<T>) -> Robj {
    match r {
        Ok(v) => v.into(),
        Err(e) => throw_r_error(e.to_string()),
    }
}

/// Largest integer, 2^53, up to and *including* which every integer is exactly
/// representable as an `f64` (2^53 + 1 is the first that is not). Above it,
/// consecutive integers are indistinguishable as doubles, so a value that large
/// is a mistake rather than an addressable row — hence the guard is `v > this`,
/// which accepts 2^53 itself.
const MAX_EXACT_F64_INT: f64 = 9_007_199_254_740_992.0;

/// Why `v` cannot be used as a whole, non-negative index or count — or `None`
/// when it can.
///
/// `f64 as u64` / `as usize` **saturate** in Rust: `-1.0` and `NaN` both become
/// `0`, `1e300` becomes `u64::MAX`, and fractional values truncate. None of
/// that is an error, so an unvalidated cast turns a typo, an `NA` from an
/// upstream join, or a 1-based index handed to a 0-based method into a silent
/// read of row 0 — the wrong cells, no warning.
///
/// Deliberately pure (no `R_IsNA`/FFI) so the whole value-class matrix is
/// testable by `cargo test -p rscx` rather than through the R install loop.
/// R's `NA_real_` *is* a quiet NaN, so `is_nan()` catches it, but the two are
/// indistinguishable without R — hence "NA or NaN". extendr already rejects
/// `NA` for a **scalar** `f64` parameter before we are called; it does not
/// check inside a `Vec<f64>`, which is why the slice form below matters most.
fn reject_reason(v: f64, what: &str) -> Option<String> {
    // `{v:?}` and not `{v}`: `Display` for f64 never uses exponent notation, so
    // a stray `1e300` would render as a 301-digit error message.
    if v.is_nan() {
        Some(format!("{what} is NA or NaN"))
    } else if !v.is_finite() {
        Some(format!("non-finite {what}: {v:?}"))
    } else if v < 0.0 {
        // IEEE: `-0.0 < 0.0` is false, so `-0.0` is accepted and casts to 0 —
        // the right answer for a 0-based index. Do not "improve" this to
        // `is_sign_negative()`.
        Some(format!("negative {what}: {v:?}"))
    } else if v.fract() != 0.0 {
        Some(format!("non-integer {what}: {v:?}"))
    } else if v > MAX_EXACT_F64_INT {
        Some(format!("{what} too large to be an exact integer: {v:?}"))
    } else {
        None
    }
}

/// Convert one R double into a `u64` row index or bound.
///
/// `what` names the value in the error message; include "0-based" in it when
/// the method's contract is 0-based, so the number in the message is not read
/// as a 1-based index. This is the only sanctioned way to turn an R numeric
/// into an index — see `docs/conventions.md` § R Bindings.
pub(crate) fn r_whole_u64(v: f64, what: &str) -> Result<u64> {
    match reject_reason(v, what) {
        Some(msg) => Err(Error::Other(msg)),
        None => Ok(v as u64),
    }
}

/// Vector form of [`r_whole_u64`], naming the offending element by its 1-based
/// position — with 10k indices, the value alone does not locate the mistake.
///
/// Two passes so the happy path costs no per-element `format!`.
pub(crate) fn r_whole_u64_slice(vs: &[f64], what: &str) -> Result<Vec<u64>> {
    for (i, &v) in vs.iter().enumerate() {
        if let Some(msg) = reject_reason(v, what) {
            return Err(Error::Other(format!("{msg} (at position {})", i + 1)));
        }
    }
    Ok(vs.iter().map(|&v| v as u64).collect())
}

/// Convert one R double into a `usize` count.
///
/// This only guarantees the value is whole, non-negative and exact. A caller
/// that uses the result as an **allocation capacity** must clamp it to a sane
/// ceiling too: `cache_shards` reaches `LruCache::new`, which pre-allocates a
/// `HashMap` of that size, so `1e9` would try to reserve tens of gigabytes.
pub(crate) fn r_whole_usize(v: f64, what: &str) -> Result<usize> {
    let n = r_whole_u64(v, what)?;
    usize::try_from(n).map_err(|_| Error::Other(format!("{what} too large for this platform: {n}")))
}

/// Default LRU size, mirrored from the R-side `cache_shards = 128` defaults.
const DEFAULT_CACHE_SHARDS: usize = 128;

/// Clamp a requested shard-cache size into a range that cannot abort the
/// R session.
///
/// The lower clamp is the long-standing one: `NonZeroUsize::new(0)` would panic
/// and a zero-entry LRU defeats the cache. The **upper** clamp is the one that
/// matters — `cache_shards` reaches `LruCache::new`, which pre-allocates a
/// `HashMap` of that capacity, so `cache_shards = 1e9` reserves tens of
/// gigabytes and dies on allocation failure rather than raising an R error.
/// A cache bigger than the file has shards is useless, so the shard count is
/// the natural ceiling; never cap below the documented default, so a header
/// that under-counts cannot quietly degrade the common case.
pub(crate) fn clamp_cache_shards(requested: usize, n_shards: usize) -> usize {
    requested.clamp(1, n_shards.max(DEFAULT_CACHE_SHARDS))
}

/// Factorise a character vector into contiguous `u32` level codes plus the
/// first-seen level names — matching `pandas.factorize(sort=False)`.
///
/// Single source of truth for the accelerator (`accel.rs`) and Harmony
/// (`harmony.rs`) bindings. Callers that only need the level
/// *count* take `levels.len()`.
pub(crate) fn factorize_chars(labels: &[String]) -> (Vec<u32>, Vec<String>) {
    // Key the map with `&str` borrowed from the input (which outlives the map),
    // so only `levels` clones each unique label — one allocation per level
    // instead of two.
    let mut map: HashMap<&str, u32> = HashMap::new();
    let mut levels: Vec<String> = Vec::new();
    let mut codes: Vec<u32> = Vec::with_capacity(labels.len());
    for s in labels {
        let code = match map.get(s.as_str()) {
            Some(&c) => c,
            None => {
                let c = levels.len() as u32;
                map.insert(s.as_str(), c);
                levels.push(s.clone());
                c
            }
        };
        codes.push(code);
    }
    (codes, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err(v: f64) -> String {
        r_whole_u64(v, "row index").unwrap_err().to_string()
    }

    // `cast_nan_to_int` fires on `f64::NAN as u64` below. That cast is the
    // premise under test, not a mistake: the point is to pin, in the test
    // itself, that the bare cast this module exists to replace yields a
    // valid-looking index rather than failing.
    #[allow(clippy::cast_nan_to_int)]
    #[test]
    fn rejects_every_value_the_bare_cast_would_swallow() {
        // The premise: each of these casts to a *valid-looking* index today.
        assert_eq!(-1.0f64 as u64, 0);
        assert_eq!(f64::NAN as u64, 0);
        assert_eq!(1.9f64 as u64, 1);
        assert_eq!(1e300f64 as u64, u64::MAX);

        assert!(err(-1.0).contains("negative"), "{}", err(-1.0));
        assert!(err(f64::NAN).contains("NA or NaN"), "{}", err(f64::NAN));
        assert!(err(1.9).contains("non-integer"), "{}", err(1.9));
        assert!(err(f64::INFINITY).contains("non-finite"));
        assert!(err(f64::NEG_INFINITY).contains("non-finite"));
        assert!(err(MAX_EXACT_F64_INT + 2.0).contains("too large"));
    }

    #[test]
    fn accepts_whole_non_negative_values_including_the_boundaries() {
        assert_eq!(r_whole_u64(0.0, "x").unwrap(), 0);
        // IEEE: -0.0 < 0.0 is false, and -0.0 as u64 is 0 — correct for a
        // 0-based index, so this must not be rejected.
        assert_eq!(r_whole_u64(-0.0, "x").unwrap(), 0);
        assert_eq!(r_whole_u64(19.0, "x").unwrap(), 19);
        assert_eq!(r_whole_u64(MAX_EXACT_F64_INT, "x").unwrap(), 1u64 << 53);
        // Values above 2^31 are the entire reason these cross the boundary as
        // doubles rather than as R integers.
        assert_eq!(r_whole_u64(3e9, "x").unwrap(), 3_000_000_000);
    }

    #[test]
    fn slice_form_locates_the_offending_element_by_1_based_position() {
        let e = r_whole_u64_slice(&[0.0, 1.0, -2.0], "row index")
            .unwrap_err()
            .to_string();
        assert!(e.contains("negative"), "{e}");
        assert!(e.contains("at position 3"), "{e}");

        // First bad element wins, and a clean slice round-trips in order.
        let e2 = r_whole_u64_slice(&[f64::NAN, -1.0], "row index")
            .unwrap_err()
            .to_string();
        assert!(
            e2.contains("NA or NaN") && e2.contains("position 1"),
            "{e2}"
        );
        assert_eq!(r_whole_u64_slice(&[2.0, 0.0], "x").unwrap(), vec![2, 0]);
        assert!(r_whole_u64_slice(&[], "x").unwrap().is_empty());
    }

    #[test]
    fn message_names_the_value_compactly_not_as_a_301_digit_expansion() {
        // `Display` for f64 never uses exponent notation; `Debug` does.
        let e = err(1e300);
        assert!(e.contains("1e300"), "{e}");
        assert!(
            e.len() < 80,
            "message should stay short, got {} chars",
            e.len()
        );
    }

    #[test]
    fn cache_shards_is_clamped_on_both_ends() {
        assert_eq!(clamp_cache_shards(0, 4), 1);
        assert_eq!(clamp_cache_shards(64, 4), 64); // never below the default
        assert_eq!(clamp_cache_shards(128, 4), 128);
        // The hazard: an enormous request would pre-allocate a HashMap of that
        // capacity inside LruCache::new and abort the process.
        assert_eq!(clamp_cache_shards(1_000_000_000, 4), DEFAULT_CACHE_SHARDS);
        assert_eq!(clamp_cache_shards(usize::MAX, 4096), 4096);
    }

    #[test]
    fn count_form_shares_the_same_rejections() {
        assert_eq!(r_whole_usize(128.0, "cache_shards").unwrap(), 128);
        assert!(r_whole_usize(-1.0, "cache_shards")
            .unwrap_err()
            .to_string()
            .contains("negative"));
        assert!(r_whole_usize(f64::NAN, "cache_shards")
            .unwrap_err()
            .to_string()
            .contains("NA or NaN"));
    }
}

/// Refuse to open a backed / lazy handle over a file that carries deletion
/// vectors.
///
/// The backed handles are a *physical* row-space API: `read_rows(start, end)`
/// and `read_row_indices(idx)` take global row ids and bounds-check them against
/// the header's `n_obs`. Honouring deletions properly needs a kept→global index
/// translation on every read (pyscx has one; rscx does not), and filtering only
/// the whole-matrix `to_dgcmatrix()` would leave two different row spaces inside
/// one object — a partial fix that reads as a working one.
///
/// So this fails loud instead, with the two ways forward. It is a narrow case:
/// the file must actually have cells marked deleted, and every non-backed rscx
/// read path applies the mask.
pub fn reject_backed_on_deletions(reader: &scx_format_io::ScxReader, path: &str) -> Result<()> {
    let n_deleted = reader
        .read_deletion_vectors()
        .map_err(|e| Error::Other(e.to_string()))?
        .map(|dv| dv.total_deleted())
        .unwrap_or(0);
    if n_deleted == 0 {
        return Ok(());
    }
    Err(Error::Other(format!(
        "'{path}' has {n_deleted} logically deleted cell(s), which backed and lazy \
         handles cannot yet address (their row indices are physical, so they would \
         hand back deleted cells). Use scx_open(path)$x_matrix() / scx_query(), which \
         apply deletions, or run `scx compact` to materialize them away first."
    )))
}
