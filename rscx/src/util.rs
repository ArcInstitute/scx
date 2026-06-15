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
pub(crate) fn throw_on_err<T: Into<Robj>>(r: Result<T>) -> Robj {
    match r {
        Ok(v) => v.into(),
        Err(e) => throw_r_error(e.to_string()),
    }
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
