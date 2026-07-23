//! Shared finiteness guard for accelerator streaming kernels.
//!
//! NaN/Inf cannot be summarised into a meaningful mean/variance and would
//! silently poison downstream selection (HVG) or corrupt clipped
//! accumulation (`f64::min` returns the non-NaN operand, so a NaN is
//! silently absorbed as the clip value). Finiteness is a contract at the
//! accelerator entry, validated where the streaming pass already touches
//! every nonzero — not at file ingest.
//!
//! This is the single primitive used by every **HVG** storage route (CSR and
//! CSC, in-memory and backed) so the guard cannot be enforced on one layout and
//! skipped on another. (Other accelerators — `diffexp`, `gene_score`, `pflog`
//! — still carry their own local finiteness checks; consolidating those is out
//! of scope here.)

use crate::error::{AccelError, Result};

/// Reject non-finite (NaN/±Inf) values at an accelerator boundary.
///
/// `context` names the operation for the error message (e.g. `"HVG"`).
/// Reports the first offending value and its index within `data`.
pub(crate) fn ensure_finite_values(data: &[f32], context: &str) -> Result<()> {
    if let Some(pos) = data.iter().position(|v| !v.is_finite()) {
        return Err(AccelError::InvalidInput(format!(
            "{context} input contains a non-finite value ({}) at nonzero index {pos}; \
             this accelerator requires finite input — filter/QC NaN and Inf before \
             computing variance",
            data[pos]
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_finite() {
        assert!(ensure_finite_values(&[0.0, 1.5, -3.0, 1e30], "HVG").is_ok());
        assert!(ensure_finite_values(&[], "HVG").is_ok());
    }

    #[test]
    fn rejects_non_finite() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let data = [1.0, bad, 2.0];
            let err = ensure_finite_values(&data, "HVG").unwrap_err();
            match err {
                AccelError::InvalidInput(msg) => {
                    assert!(msg.contains("non-finite"), "message: {msg}");
                    assert!(msg.contains("index 1"), "should report position: {msg}");
                }
                other => panic!("expected InvalidInput, got {other:?}"),
            }
        }
    }
}
