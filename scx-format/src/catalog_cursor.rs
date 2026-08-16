//! One walk over the full catalog's entry list.
//!
//! ORG-4.9-1. Review §4.9's finding is that the same wire format was decoded
//! by more than one hand-written walk, and §4.2 is what that cost: they had
//! drifted on the order of two steps. [`FullCatalog::read_from`](crate::catalog::FullCatalog::read_from) decoded an
//! entry's `ShardStats` *before* resolving its `SectionType`, so a file
//! carrying a section type this reader does not know — with a stats blob
//! shorter than the current layout — failed to parse **at all**. Not the
//! entry: the catalog, which is the index to every section, so the whole file
//! became unopenable. `CatalogView::read_from_bytes`, decoding the identical
//! bytes, already resolved the type first and skipped cleanly.
//!
//! [`CatalogEntryCursor`] yields [`RawEntry`], which is deliberately *raw*:
//! `section_type_raw` is an unresolved `u8` and `name_bytes` / `stats_bytes`
//! are borrowed and undecoded. A consumer therefore cannot decode a stats
//! blob before it has decided it wants the entry — the ordering §4.2 is about
//! is a property of the shape, not of each consumer remembering it.
//!
//! # What each consumer still owns
//!
//! Materialisation, and only materialisation. `FullCatalog` allocates a
//! `String` name and a full [`ShardStats`](crate::catalog::ShardStats);
//! `CatalogView` keeps names only for entries that may be looked up by one
//! and collapses stats to three `u64`s. The rule the split encodes: **each
//! consumer validates exactly what it materialises.** That makes the UTF-8
//! policy uniform — an entry dropped for an unknown section type is never
//! decoded, so its name encoding cannot fail the catalog — where previously
//! `FullCatalog` validated every name including the ones it discarded and
//! `CatalogView` validated only the ones it kept.
//!
//! # Callers that need the length before they can slice
//!
//! Catalogs in the manifest chain do not record their own byte length, so
//! `scx info`'s history walk has to compute it. [`catalog_payload_len`]
//! drains a cursor to do that, rather than re-deriving the per-entry
//! arithmetic — which is what `scx-cli` did, with its own copy of
//! [`MIN_ENTRY_BYTES`], the v2 `modality_id` gate and the v4 trailer gate.

use byteorder::{LittleEndian, ReadBytesExt};

use crate::checksum::blake3_hash;
use crate::error::{validate_allocation, Result, ScxError};

/// Catalog payload header: `catalog_version: u16` + `manifest_sequence: u64`
/// + `prev_catalog_offset: u64` + `n_obs: u64` + `n_entries: u32`.
pub const CATALOG_PREAMBLE_LEN: usize = 2 + 8 + 8 + 8 + 4;

/// Trailing BLAKE3 over the whole payload.
pub const CATALOG_CHECKSUM_LEN: usize = 32;

/// Smallest serialized entry: `name_len(2)` + an empty name + `offset(8)` +
/// `length(8)` + `section_type(1)` + `checksum(32)` + `stats_len(2)`. v2
/// entries add a `modality_id` byte, so the v1 figure is the safe bound for
/// the up-front allocation guard.
pub const MIN_ENTRY_BYTES: usize = 53;

/// Fixed-width bytes between an entry's name and its `stats_len` prefix:
/// `offset(8)` + `length(8)` + `section_type(1)` + `checksum(32)`, plus the
/// `modality_id(1)` that v2 introduced.
pub const fn entry_fixed_bytes_after_name(catalog_version: u16) -> usize {
    8 + 8 + 1 + 32 + if catalog_version >= 2 { 1 } else { 0 }
}

/// The two v4 generation counters (`data_generation`, `csc_build_generation`)
/// that sit between the entry list and the trailing checksum. Zero bytes
/// below v4 — the counters did not exist, and both default to `0`.
pub const fn v4_trailer_len(catalog_version: u16) -> usize {
    if catalog_version >= 4 {
        16
    } else {
        0
    }
}

/// The catalog payload's fixed header, parsed once by the cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogPreamble {
    pub catalog_version: u16,
    pub manifest_sequence: u64,
    pub prev_catalog_offset: u64,
    pub n_obs: u64,
    /// Declared entry count. The number of entries a consumer *keeps* is
    /// smaller whenever the catalog carries a section type it cannot resolve.
    pub n_entries: usize,
}

/// One catalog entry, exactly as it sits on the wire.
///
/// Nothing here is decoded or validated beyond its length: the section type
/// is the raw discriminant, and the name and stats payloads are borrowed out
/// of the caller's buffer. That is the point — see the module docs.
#[derive(Debug, Clone, Copy)]
pub struct RawEntry<'a> {
    /// Section name, **not** UTF-8 validated. Validate at materialisation.
    pub name_bytes: &'a [u8],
    pub offset: u64,
    pub length: u64,
    /// Unresolved discriminant. `SectionType::from_u8` returning `None` means
    /// a future writer produced it; skip the entry, do not fail the catalog.
    pub section_type_raw: u8,
    /// The per-entry BLAKE3, exactly 32 bytes.
    pub checksum_bytes: &'a [u8],
    /// `0` on v1 catalogs, which have no such byte.
    pub modality_id: u8,
    /// The stats payload, undecoded. Empty when `stats_len == 0`. Its length
    /// is authoritative: a decoder must not read past it, and bytes it does
    /// not consume are forward-compat padding.
    pub stats_bytes: &'a [u8],
}

/// A single forward pass over a catalog payload's entry list.
///
/// Construct with [`Self::new`] (payload + trailing checksum, the shape both
/// `FullCatalog::read_from` and `CatalogView::read_from_bytes` receive) or
/// [`Self::over_payload`] (no trailing checksum, for callers still working
/// out where the catalog ends). Drain with [`Self::next_entry`], then take
/// the v4 counters with [`Self::finish`].
#[derive(Debug)]
pub struct CatalogEntryCursor<'a> {
    payload: &'a [u8],
    cur: &'a [u8],
    preamble: CatalogPreamble,
    remaining_entries: usize,
}

impl<'a> CatalogEntryCursor<'a> {
    /// Split the trailing 32-byte checksum off `payload_with_checksum`,
    /// optionally verify it, then parse the preamble.
    ///
    /// `verify_checksum == false` still consumes the checksum bytes without
    /// checking them — the trusted-source read path both parsers already had.
    pub fn new(payload_with_checksum: &'a [u8], verify_checksum: bool) -> Result<Self> {
        let total_len = payload_with_checksum.len();
        if total_len < CATALOG_CHECKSUM_LEN {
            return Err(ScxError::ChecksumMismatch {
                section: "full_catalog (too short)".to_string(),
            });
        }
        let payload_len = total_len - CATALOG_CHECKSUM_LEN;
        let (payload, expected_checksum) = payload_with_checksum.split_at(payload_len);

        if verify_checksum {
            let computed = blake3_hash(payload);
            if computed[..] != *expected_checksum {
                return Err(ScxError::ChecksumMismatch {
                    section: "full_catalog".to_string(),
                });
            }
        }

        Self::over_payload(payload)
    }

    /// Parse the preamble of a payload that carries no trailing checksum.
    ///
    /// `payload` may extend past the end of the catalog — the walk stops
    /// after `n_entries`, and [`Self::consumed`] then reports where that was.
    /// The up-front allocation guard is bounded by `payload.len()`, which is
    /// looser in that case but still rejects an `n_entries` that could not
    /// possibly fit.
    pub fn over_payload(payload: &'a [u8]) -> Result<Self> {
        let mut cur: &[u8] = payload;
        let catalog_version = cur.read_u16::<LittleEndian>()?;
        let manifest_sequence = cur.read_u64::<LittleEndian>()?;
        let prev_catalog_offset = cur.read_u64::<LittleEndian>()?;
        let n_obs = cur.read_u64::<LittleEndian>()?;
        let n_entries = cur.read_u32::<LittleEndian>()? as usize;

        // Reject a corrupt count before it drives a `Vec::with_capacity` or a
        // long walk: the entries cannot fit if even their minimum size
        // exceeds the payload.
        validate_allocation(n_entries.saturating_mul(MIN_ENTRY_BYTES), payload.len())?;

        Ok(Self {
            payload,
            cur,
            preamble: CatalogPreamble {
                catalog_version,
                manifest_sequence,
                prev_catalog_offset,
                n_obs,
                n_entries,
            },
            remaining_entries: n_entries,
        })
    }

    pub fn preamble(&self) -> &CatalogPreamble {
        &self.preamble
    }

    /// Bytes consumed so far, measured from the first byte of the payload.
    pub fn consumed(&self) -> usize {
        self.payload.len() - self.cur.len()
    }

    /// The next entry, or `None` once the declared count is exhausted.
    ///
    /// A malformed entry yields `Some(Err(_))` and leaves the cursor unusable
    /// — callers propagate rather than continuing.
    pub fn next_entry(&mut self) -> Option<Result<RawEntry<'a>>> {
        if self.remaining_entries == 0 {
            return None;
        }
        self.remaining_entries -= 1;
        Some(self.read_entry())
    }

    fn read_entry(&mut self) -> Result<RawEntry<'a>> {
        let name_len = self.cur.read_u16::<LittleEndian>()? as usize;
        let name_bytes = self.take(name_len, "catalog entry name truncated")?;

        let offset = self.cur.read_u64::<LittleEndian>()?;
        let length = self.cur.read_u64::<LittleEndian>()?;
        let section_type_raw = self.cur.read_u8()?;
        let checksum_bytes = self.take(32, "catalog entry checksum truncated")?;

        // v2 catalogs carry a `modality_id: u8` after the per-entry checksum;
        // v1 catalogs do not, and every v1 entry is the global modality.
        let modality_id = if self.preamble.catalog_version >= 2 {
            self.cur.read_u8()?
        } else {
            0u8
        };

        let stats_len = self.cur.read_u16::<LittleEndian>()? as usize;
        let stats_bytes = self.take(stats_len, "catalog entry stats payload truncated")?;

        Ok(RawEntry {
            name_bytes,
            offset,
            length,
            section_type_raw,
            checksum_bytes,
            modality_id,
            stats_bytes,
        })
    }

    /// Peel `n` bytes off the cursor as a borrowed sub-slice of the payload.
    ///
    /// `cur` is copied out of `self` first so the returned slice carries the
    /// payload's `'a`, not a reborrow of `&mut self`.
    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8]> {
        let cur: &'a [u8] = self.cur;
        let (head, rest) = cur
            .split_at_checked(n)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::UnexpectedEof, what))?;
        self.cur = rest;
        Ok(head)
    }

    /// The v4 generation counters that follow the entry list, or `(0, 0)` on
    /// v1–v3 catalogs, which lack the fields.
    ///
    /// `0 == 0` reads as "fresh" to the CSC-sidecar staleness guard, so a
    /// legacy file is never treated as stale. Errors if the entry list was
    /// not drained — the trailer only sits where it does once it has been.
    pub fn finish(mut self) -> Result<(u64, u64)> {
        if self.remaining_entries != 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "catalog trailer read with {} entries still unread",
                self.remaining_entries
            )));
        }
        if self.preamble.catalog_version >= 4 {
            let data_generation = self.cur.read_u64::<LittleEndian>()?;
            let csc_build_generation = self.cur.read_u64::<LittleEndian>()?;
            Ok((data_generation, csc_build_generation))
        } else {
            Ok((0, 0))
        }
    }
}

/// Byte length of the catalog starting at `bytes[0]`, from its first byte
/// through the v4 trailer — **excluding** the trailing 32-byte checksum.
///
/// For callers that must discover where a catalog ends before they can slice
/// it: prior catalogs in the manifest chain do not record their own length,
/// but a catalog is self-delimiting, since the preamble records `n_entries`
/// and every entry carries `name_len` / `stats_len` prefixes. `bytes` may run
/// past the end of the catalog (to EOF, typically); the walk stops at the
/// declared entry count.
pub fn catalog_payload_len(bytes: &[u8]) -> Result<usize> {
    let mut cursor = CatalogEntryCursor::over_payload(bytes)?;
    while let Some(entry) = cursor.next_entry() {
        entry?;
    }
    let consumed = cursor.consumed();
    let trailer = v4_trailer_len(cursor.preamble.catalog_version);
    // Not for the counters — for the bounds check. `finish` errors if the
    // trailer a v4 catalog declares runs past the end of `bytes`.
    cursor.finish()?;
    Ok(consumed + trailer)
}

#[cfg(test)]
#[path = "catalog_cursor_tests.rs"]
mod tests;
