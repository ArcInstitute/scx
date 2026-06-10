//! Small shared helpers for the R bindings.

use std::collections::HashMap;

/// Factorise a character vector into contiguous `u32` level codes plus the
/// first-seen level names — matching `pandas.factorize(sort=False)`.
///
/// Single source of truth for the accelerator (`accel.rs`) and Harmony
/// (`harmony.rs`) bindings (I-ORG-1 / T4.9). Callers that only need the level
/// *count* take `levels.len()`.
pub(crate) fn factorize_chars(labels: &[String]) -> (Vec<u32>, Vec<String>) {
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut levels: Vec<String> = Vec::new();
    let mut codes: Vec<u32> = Vec::with_capacity(labels.len());
    for s in labels {
        let code = match map.get(s) {
            Some(&c) => c,
            None => {
                let c = levels.len() as u32;
                map.insert(s.clone(), c);
                levels.push(s.clone());
                c
            }
        };
        codes.push(code);
    }
    (codes, levels)
}
