//! Uniform version gating for on-disk section types.
//!
//! Several versioned sections (`BitmapShard`, `DeletionVectors`,
//! `ModalityTable`, …) each read a version field in their `read_from`, but
//! historically validated it ad hoc — some wrapped the mismatch in
//! [`ScxError::Io`](crate::ScxError::Io), some in
//! [`ScxError::InvalidCatalog`](crate::ScxError::InvalidCatalog), and
//! `DeletionVectors` did not validate it at all (a future v2 layout would have
//! been silently misparsed — code-review finding F5).
//!
//! [`VersionedSection`] gives every section one place to declare its current
//! version and one shared [`check_version`](VersionedSection::check_version)
//! that returns the structured
//! [`ScxError::UnsupportedSectionVersion`](crate::ScxError::UnsupportedSectionVersion).
//! A section that bumps its layout updates `CURRENT_VERSION` and the regression
//! test asserting `check_version(CURRENT_VERSION + 1)` is rejected catches any
//! reader that forgot to gate.

use crate::error::{Result, ScxError};

/// A section type that carries an on-disk format version.
pub trait VersionedSection {
    /// Human-readable section name, used in the error message.
    const SECTION_NAME: &'static str;
    /// The version this build writes and is the only version it can read.
    const CURRENT_VERSION: u16;

    /// Reject a version this build does not understand.
    ///
    /// Forward-incompatible by design: an unknown (including newer) version is
    /// an error rather than a best-effort parse, so a layout change can never
    /// be silently misread.
    fn check_version(found: u16) -> Result<()> {
        if found != Self::CURRENT_VERSION {
            return Err(ScxError::UnsupportedSectionVersion {
                section: Self::SECTION_NAME,
                found,
                expected: Self::CURRENT_VERSION,
            });
        }
        Ok(())
    }
}
