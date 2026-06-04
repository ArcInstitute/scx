//! Bounds-checked view over a shard section's bytes.
//!
//! Shard-header offset/length fields are read raw from untrusted bytes: the
//! catalog BLAKE3 authenticates *catalog* bytes, not shard payloads, so a
//! corrupt or hostile shard can declare a region that points past the section.
//! The outer section extent is already validated once against the mmap in
//! [`ScxReader::section_bytes`](crate::reader::ScxReader::section_bytes)
//! (`checked_add` + mmap-len); the remaining hole is the *internal* offset
//! slicing in the shard decoder.
//!
//! [`ValidatedSection`] carries the "this `&[u8]` is the full, extent-checked
//! section" contract in the type and funnels every internal slice through one
//! checked extraction that returns [`ScxError::SectionOutOfBounds`] instead of
//! panicking. This is the structural form of code-review finding F1 (Phase 1,
//! Abstraction 1) — the surgical fix is to route the four
//! `&section[off..][..len]` slices through here, *not* to change
//! `ShardHeader::read_from`'s signature (the section is already extent-checked
//! by contract, so threading `section_len` would touch ~80–100 call sites for
//! no added safety).

use crate::error::{Result, ScxError};
use crate::shard::SHARD_HEADER_SIZE;

/// A shard section whose outer extent has already been validated against the
/// backing store (mmap or `object_store` range read). Construct via
/// [`ValidatedSection::new`] only where that contract holds.
#[derive(Clone, Copy)]
pub(crate) struct ValidatedSection<'a> {
    bytes: &'a [u8],
}

impl<'a> ValidatedSection<'a> {
    /// Wrap a section whose extent is already bounds-checked against the
    /// backing store. The decoder only ever receives such slices (from
    /// `section_bytes` / a sized range read).
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }

    /// The fixed-size shard-header prefix.
    ///
    /// A section shorter than [`SHARD_HEADER_SIZE`] would panic the raw
    /// `&section[..SHARD_HEADER_SIZE]` slice; reject it as out-of-bounds.
    pub(crate) fn header(&self) -> Result<&'a [u8]> {
        if self.bytes.len() < SHARD_HEADER_SIZE {
            return Err(ScxError::SectionOutOfBounds {
                offset: 0,
                length: SHARD_HEADER_SIZE as u64,
                file_size: self.bytes.len(),
            });
        }
        Ok(&self.bytes[..SHARD_HEADER_SIZE])
    }

    /// Bounds-checked extraction of `section[off..off + len]`.
    ///
    /// `off`/`len` come from the untrusted shard header, so the addition is
    /// done in `u64` to defeat `u32` overflow before comparing against the
    /// section length.
    pub(crate) fn subslice(&self, off: u32, len: u32) -> Result<&'a [u8]> {
        let start = off as usize;
        let end = (off as u64).checked_add(len as u64);
        match end {
            Some(end) if end <= self.bytes.len() as u64 => Ok(&self.bytes[start..end as usize]),
            _ => Err(ScxError::SectionOutOfBounds {
                offset: off as u64,
                length: len as u64,
                file_size: self.bytes.len(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_rejects_truncated_section() {
        let buf = vec![0u8; SHARD_HEADER_SIZE - 1];
        let vs = ValidatedSection::new(&buf);
        assert!(matches!(
            vs.header(),
            Err(ScxError::SectionOutOfBounds { .. })
        ));
    }

    #[test]
    fn header_returns_prefix() {
        let buf = vec![7u8; SHARD_HEADER_SIZE + 10];
        let vs = ValidatedSection::new(&buf);
        let h = vs.header().unwrap();
        assert_eq!(h.len(), SHARD_HEADER_SIZE);
    }

    #[test]
    fn subslice_bounds() {
        let buf = vec![0u8; 100];
        let vs = ValidatedSection::new(&buf);
        assert!(vs.subslice(0, 100).is_ok());
        assert!(vs.subslice(50, 50).is_ok());
        assert_eq!(vs.subslice(10, 5).unwrap().len(), 5);
        assert!(matches!(
            vs.subslice(50, 51),
            Err(ScxError::SectionOutOfBounds { .. })
        ));
        // offset + length overflow must not wrap to a small in-bounds value.
        assert!(matches!(
            vs.subslice(u32::MAX, u32::MAX),
            Err(ScxError::SectionOutOfBounds { .. })
        ));
    }
}
