// FileHeader struct + read/write (docs/format.md (File Header))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::error::{Result, ScxError};

/// Size of the file header in bytes.
pub const HEADER_SIZE: usize = 256;

/// Magic bytes identifying an SCX file.
pub const MAGIC: [u8; 4] = *b"SCX\x01";

/// Current format version produced by ScxWriter.
///
/// Readers accept versions 1..=CURRENT_FORMAT_VERSION.
/// The writer stamps this value in `finish()`.
///
/// v2 (current) carries three new fields between `front_catalog_length`
/// and the trailing `reserved` block: `n_modalities`,
/// `modality_table_offset`, `modality_table_length`. v1 readers reject
/// v2 files via the `format_version > CURRENT_FORMAT_VERSION` check;
/// v2 readers accept both versions and stamp the new fields as zero
/// when reading a v1 file (single-modality semantic equivalence).
pub const CURRENT_FORMAT_VERSION: u16 = 2;

/// Bitmask of currently-defined flag bits. Reserved bits (4 and 8..=31)
/// must be zero per the on-disk spec; `read_from` rejects any header
/// whose `flags & !KNOWN_FLAGS != 0` so future writers can't sneak
/// undefined bits past today's readers.
pub const KNOWN_FLAGS: u32 =
    (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 6) | (1 << 7);

/// The 256-byte file header that starts every SCX file.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic bytes: b"SCX\x01"
    pub magic: [u8; 4],
    /// Format version (currently 2)
    pub format_version: u16,
    /// Header length in bytes (always 256)
    pub header_length: u16,
    /// Bit-field flags.
    ///
    /// | Bit | Meaning                                              |
    /// |-----|------------------------------------------------------|
    /// |   0 | `has_csc`                                            |
    /// |   1 | `has_bitmap`                                         |
    /// |   2 | `has_obsm`                                           |
    /// |   3 | `has_obsp`                                           |
    /// |   4 | **reserved** — must be zero on write, ignored on read |
    /// |   5 | `has_deletion_vectors`                               |
    /// |   6 | `has_front_catalog`                                  |
    /// |   7 | `has_modalities` (v2; set when `n_modalities > 0`)    |
    /// | 8–31 | reserved for future use                              |
    pub flags: u32,
    /// Number of observations (rows)
    pub n_obs: u64,
    /// Number of variables (columns)
    pub n_vars: u64,
    /// Total number of non-zero entries
    pub nnz: u64,
    /// Number of CSR shards
    pub n_csr_shards: u32,
    /// Number of CSC shards
    pub n_csc_shards: u32,
    /// Target rows per shard
    pub shard_target_rows: u32,
    /// Codec ID (0=None, 1=Scx1, 2=Zstd)
    pub codec_id: u8,
    /// Index dtype (0=u16, 1=u32)
    pub index_dtype: u8,
    /// Endianness marker (must be 0 for little-endian)
    pub endian: u8,
    /// Reserved padding byte
    pub reserved_padding: u8,
    /// Offset of the root catalog
    pub root_catalog_offset: u64,
    /// Length of the root catalog
    pub root_catalog_length: u64,
    /// Offset of the full catalog
    pub full_catalog_offset: u64,
    /// Length of the full catalog
    pub full_catalog_length: u64,
    /// Manifest sequence number
    pub manifest_sequence: u64,
    /// Offset of previous catalog (for append)
    pub prev_catalog_offset: u64,
    /// Truncated BLAKE3 checksum of the file
    pub file_checksum: u64,
    /// Offset of the front catalog
    pub front_catalog_offset: u64,
    /// Length of the front catalog
    pub front_catalog_length: u64,
    /// Number of named modalities (v2). `0` for single-modality files
    /// (semantic equivalent of v1). Capped at 255 in practice; stored
    /// as u32 for 8-byte alignment of the next field.
    pub n_modalities: u32,
    /// Byte offset of the `ModalityTable` section, or `0` when
    /// `n_modalities == 0`.
    pub modality_table_offset: u64,
    /// Length of the `ModalityTable` section in bytes, or `0` when
    /// `n_modalities == 0`.
    pub modality_table_length: u64,
    /// Reserved bytes (must be zero). Shrunk from 132 → 112 in v2 to
    /// make room for the three modality-routing fields above
    /// (132 − 4 − 8 − 8 = 112).
    pub reserved: [u8; 112],
}

impl FileHeader {
    /// Write the header to a writer in little-endian byte order.
    ///
    /// The on-disk layout depends on `format_version`:
    /// - v1: ... `front_catalog_length`, then a 132-byte `reserved` tail.
    /// - v2: ... `front_catalog_length`, `n_modalities` (u32),
    ///   `modality_table_offset` (u64), `modality_table_length` (u64),
    ///   then a 112-byte `reserved` tail.
    ///
    /// Writers should always emit v2 going forward
    /// (`format_version = CURRENT_FORMAT_VERSION`).
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.magic)?;
        w.write_u16::<LittleEndian>(self.format_version)?;
        w.write_u16::<LittleEndian>(self.header_length)?;
        w.write_u32::<LittleEndian>(self.flags)?;
        w.write_u64::<LittleEndian>(self.n_obs)?;
        w.write_u64::<LittleEndian>(self.n_vars)?;
        w.write_u64::<LittleEndian>(self.nnz)?;
        w.write_u32::<LittleEndian>(self.n_csr_shards)?;
        w.write_u32::<LittleEndian>(self.n_csc_shards)?;
        w.write_u32::<LittleEndian>(self.shard_target_rows)?;
        w.write_u8(self.codec_id)?;
        w.write_u8(self.index_dtype)?;
        w.write_u8(self.endian)?;
        w.write_u8(self.reserved_padding)?;
        w.write_u64::<LittleEndian>(self.root_catalog_offset)?;
        w.write_u64::<LittleEndian>(self.root_catalog_length)?;
        w.write_u64::<LittleEndian>(self.full_catalog_offset)?;
        w.write_u64::<LittleEndian>(self.full_catalog_length)?;
        w.write_u64::<LittleEndian>(self.manifest_sequence)?;
        w.write_u64::<LittleEndian>(self.prev_catalog_offset)?;
        w.write_u64::<LittleEndian>(self.file_checksum)?;
        w.write_u64::<LittleEndian>(self.front_catalog_offset)?;
        w.write_u64::<LittleEndian>(self.front_catalog_length)?;
        if self.format_version >= 2 {
            // v2 layout: three modality-routing fields before reserved.
            w.write_u32::<LittleEndian>(self.n_modalities)?;
            w.write_u64::<LittleEndian>(self.modality_table_offset)?;
            w.write_u64::<LittleEndian>(self.modality_table_length)?;
            w.write_all(&self.reserved)?;
        } else {
            // v1 layout: reserved is 132 bytes. Writers should not emit
            // v1 going forward (CURRENT_FORMAT_VERSION = 2), but the
            // 132-byte tail is preserved here for symmetry with the
            // legacy on-disk shape: 20 leading zero bytes (where v2
            // placed the new fields) followed by `self.reserved` (112
            // bytes). The struct invariant for a v1 header must have
            // n_modalities/modality_table_offset/length = 0 — the
            // padding is a strict zero region.
            let zero_pad = [0u8; 20];
            w.write_all(&zero_pad)?;
            w.write_all(&self.reserved)?;
        }
        Ok(())
    }

    /// Read and validate a header from a reader.
    ///
    /// Branches on `format_version` after the version check:
    /// - v1: the 20 bytes after `front_catalog_length` are part of the
    ///   legacy 132-byte `reserved` block. They are strictly required
    ///   to be zero (v1 writers always wrote them as zero, so any
    ///   non-zero value here indicates corruption). The new modality
    ///   fields are stamped as zero in the returned struct.
    /// - v2: the three modality-routing fields (`n_modalities`,
    ///   `modality_table_offset`, `modality_table_length`) are parsed
    ///   from those 20 bytes, followed by 112 bytes of `reserved`.
    ///
    /// In either case, the cross-check
    /// `(n_modalities == 0) == (modality_table_offset == 0) ==
    /// (modality_table_length == 0)` must hold; disagreement raises
    /// `ScxError::InvalidCatalog`.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(ScxError::InvalidMagic);
        }

        let format_version = r.read_u16::<LittleEndian>()?;
        if format_version == 0 || format_version > CURRENT_FORMAT_VERSION {
            return Err(ScxError::UnsupportedVersion);
        }

        let header_length = r.read_u16::<LittleEndian>()?;
        if header_length as usize != HEADER_SIZE {
            return Err(ScxError::InvalidCatalog(format!(
                "header_length {} != HEADER_SIZE {}",
                header_length, HEADER_SIZE
            )));
        }
        let flags = r.read_u32::<LittleEndian>()?;
        if flags & !KNOWN_FLAGS != 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "reserved flag bits must be zero; got flags = {:#010x}",
                flags
            )));
        }
        let n_obs = r.read_u64::<LittleEndian>()?;
        let n_vars = r.read_u64::<LittleEndian>()?;
        let nnz = r.read_u64::<LittleEndian>()?;
        let n_csr_shards = r.read_u32::<LittleEndian>()?;
        let n_csc_shards = r.read_u32::<LittleEndian>()?;
        let shard_target_rows = r.read_u32::<LittleEndian>()?;
        let codec_id = r.read_u8()?;
        let index_dtype = r.read_u8()?;
        let endian = r.read_u8()?;
        if endian != 0 {
            return Err(ScxError::UnsupportedEndian);
        }
        let reserved_padding = r.read_u8()?;
        if reserved_padding != 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "reserved_padding must be zero; got {}",
                reserved_padding
            )));
        }
        let root_catalog_offset = r.read_u64::<LittleEndian>()?;
        let root_catalog_length = r.read_u64::<LittleEndian>()?;
        let full_catalog_offset = r.read_u64::<LittleEndian>()?;
        let full_catalog_length = r.read_u64::<LittleEndian>()?;
        let manifest_sequence = r.read_u64::<LittleEndian>()?;
        let prev_catalog_offset = r.read_u64::<LittleEndian>()?;
        let file_checksum = r.read_u64::<LittleEndian>()?;
        let front_catalog_offset = r.read_u64::<LittleEndian>()?;
        let front_catalog_length = r.read_u64::<LittleEndian>()?;

        let (n_modalities, modality_table_offset, modality_table_length) = if format_version >= 2 {
            let n_modalities = r.read_u32::<LittleEndian>()?;
            let modality_table_offset = r.read_u64::<LittleEndian>()?;
            let modality_table_length = r.read_u64::<LittleEndian>()?;
            (n_modalities, modality_table_offset, modality_table_length)
        } else {
            // v1: the next 20 bytes were part of the legacy 132-byte
            // reserved tail. They must be zero (defensive — v1 writers
            // always wrote them as zero).
            let mut leading_zero_pad = [0u8; 20];
            r.read_exact(&mut leading_zero_pad)?;
            if leading_zero_pad.iter().any(|&b| b != 0) {
                return Err(ScxError::InvalidCatalog(
                    "v1 header reserved bytes 0..20 must be zero".to_string(),
                ));
            }
            (0u32, 0u64, 0u64)
        };

        let mut reserved = [0u8; 112];
        r.read_exact(&mut reserved)?;
        if reserved.iter().any(|&b| b != 0) {
            return Err(ScxError::InvalidCatalog(
                "trailing reserved bytes must all be zero".to_string(),
            ));
        }

        // Cross-check: modality fields are all-zero or all-set.
        let any_set = n_modalities != 0 || modality_table_offset != 0 || modality_table_length != 0;
        let all_set = n_modalities != 0 && modality_table_offset != 0 && modality_table_length != 0;
        if any_set && !all_set {
            return Err(ScxError::InvalidCatalog(
                "modality fields must be all-zero or all-non-zero".to_string(),
            ));
        }

        Ok(FileHeader {
            magic,
            format_version,
            header_length,
            flags,
            n_obs,
            n_vars,
            nnz,
            n_csr_shards,
            n_csc_shards,
            shard_target_rows,
            codec_id,
            index_dtype,
            endian,
            reserved_padding,
            root_catalog_offset,
            root_catalog_length,
            full_catalog_offset,
            full_catalog_length,
            manifest_sequence,
            prev_catalog_offset,
            file_checksum,
            front_catalog_offset,
            front_catalog_length,
            n_modalities,
            modality_table_offset,
            modality_table_length,
            reserved,
        })
    }

    /// Returns true if the CSC flag (bit 0) is set.
    pub fn has_csc(&self) -> bool {
        self.flags & (1 << 0) != 0
    }

    /// Set the CSC flag (bit 0).
    pub fn set_csc(&mut self) {
        self.flags |= 1 << 0;
    }

    /// Clear the CSC flag (bit 0). Used by mutating ops (`append`,
    /// `compact`, `merge`, `subset`) when CSC sidecars are dropped
    /// from the output and the row layout no longer matches the
    /// previously-stored column-major shards.
    pub fn clear_csc(&mut self) {
        self.flags &= !(1 << 0);
    }

    /// Returns true if the bitmap flag (bit 1) is set.
    pub fn has_bitmap(&self) -> bool {
        self.flags & (1 << 1) != 0
    }

    /// Set the bitmap flag (bit 1).
    pub fn set_bitmap(&mut self) {
        self.flags |= 1 << 1;
    }

    /// Returns true if the obsm flag (bit 2) is set.
    pub fn has_obsm(&self) -> bool {
        self.flags & (1 << 2) != 0
    }

    /// Set the obsm flag (bit 2).
    pub fn set_obsm(&mut self) {
        self.flags |= 1 << 2;
    }

    /// Returns true if the obsp flag (bit 3) is set.
    pub fn has_obsp(&self) -> bool {
        self.flags & (1 << 3) != 0
    }

    /// Set the obsp flag (bit 3).
    pub fn set_obsp(&mut self) {
        self.flags |= 1 << 3;
    }

    /// Returns true if the deletion vectors flag (bit 5) is set.
    pub fn has_deletion_vectors(&self) -> bool {
        self.flags & (1 << 5) != 0
    }

    /// Set the deletion vectors flag (bit 5).
    pub fn set_deletion_vectors(&mut self) {
        self.flags |= 1 << 5;
    }

    /// Returns true if the has_front_catalog flag (bit 6) is set.
    pub fn has_front_catalog(&self) -> bool {
        self.flags & (1 << 6) != 0
    }

    /// Set the has_front_catalog flag (bit 6).
    pub fn set_front_catalog(&mut self) {
        self.flags |= 1 << 6;
    }

    /// Clear the has_front_catalog flag (bit 6).
    pub fn clear_front_catalog(&mut self) {
        self.flags &= !(1 << 6);
    }

    /// Returns true if the has_modalities flag (bit 7) is set. v2-only.
    /// Set when `n_modalities > 0`; provides a fast capability check
    /// without reading the modality table.
    pub fn has_modalities(&self) -> bool {
        self.flags & (1 << 7) != 0
    }

    /// Set the has_modalities flag (bit 7). v2-only.
    pub fn set_modalities(&mut self) {
        self.flags |= 1 << 7;
    }

    /// Clear the has_modalities flag (bit 7). v2-only.
    pub fn clear_modalities(&mut self) {
        self.flags &= !(1 << 7);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_header() -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: CURRENT_FORMAT_VERSION,
            header_length: 256,
            flags: 0,
            n_obs: 50_000,
            n_vars: 30_000,
            nnz: 10_000_000,
            n_csr_shards: 4,
            n_csc_shards: 0,
            shard_target_rows: 16384,
            codec_id: 1,
            index_dtype: 0,
            endian: 0,
            reserved_padding: 0,
            root_catalog_offset: 256,
            root_catalog_length: 4096,
            full_catalog_offset: 1_000_000,
            full_catalog_length: 8192,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            file_checksum: 0xDEAD_BEEF_CAFE_BABE,
            front_catalog_offset: 0,
            front_catalog_length: 0,
            n_modalities: 0,
            modality_table_offset: 0,
            modality_table_length: 0,
            reserved: [0u8; 112],
        }
    }

    #[test]
    fn header_round_trip() {
        let original = sample_header();
        let mut buf = Vec::new();
        original.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = FileHeader::read_from(&mut cursor).unwrap();

        assert_eq!(decoded.magic, original.magic);
        assert_eq!(decoded.format_version, original.format_version);
        assert_eq!(decoded.header_length, original.header_length);
        assert_eq!(decoded.flags, original.flags);
        assert_eq!(decoded.n_obs, original.n_obs);
        assert_eq!(decoded.n_vars, original.n_vars);
        assert_eq!(decoded.nnz, original.nnz);
        assert_eq!(decoded.n_csr_shards, original.n_csr_shards);
        assert_eq!(decoded.n_csc_shards, original.n_csc_shards);
        assert_eq!(decoded.shard_target_rows, original.shard_target_rows);
        assert_eq!(decoded.codec_id, original.codec_id);
        assert_eq!(decoded.index_dtype, original.index_dtype);
        assert_eq!(decoded.endian, original.endian);
        assert_eq!(decoded.root_catalog_offset, original.root_catalog_offset);
        assert_eq!(decoded.root_catalog_length, original.root_catalog_length);
        assert_eq!(decoded.full_catalog_offset, original.full_catalog_offset);
        assert_eq!(decoded.full_catalog_length, original.full_catalog_length);
        assert_eq!(decoded.manifest_sequence, original.manifest_sequence);
        assert_eq!(decoded.prev_catalog_offset, original.prev_catalog_offset);
        assert_eq!(decoded.file_checksum, original.file_checksum);
        assert_eq!(decoded.front_catalog_offset, original.front_catalog_offset);
        assert_eq!(decoded.front_catalog_length, original.front_catalog_length);
    }

    #[test]
    fn header_writes_exactly_256_bytes() {
        let header = sample_header();
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);
    }

    #[test]
    fn header_reserved_is_all_zeros() {
        let original = sample_header();
        let mut buf = Vec::new();
        original.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = FileHeader::read_from(&mut cursor).unwrap();
        assert_eq!(decoded.reserved, [0u8; 112]);
    }

    #[test]
    fn reject_bad_magic() {
        let mut header = sample_header();
        header.magic = *b"BAD!";
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        assert!(matches!(err, ScxError::InvalidMagic));
    }

    #[test]
    fn reject_bad_endian() {
        let mut header = sample_header();
        header.endian = 1;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        assert!(matches!(err, ScxError::UnsupportedEndian));
    }

    #[test]
    fn reject_bad_version() {
        // Anything beyond CURRENT_FORMAT_VERSION (currently 2) must be
        // rejected. Version 0 is also invalid.
        let mut header = sample_header();
        header.format_version = CURRENT_FORMAT_VERSION + 1;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        assert!(matches!(err, ScxError::UnsupportedVersion));
    }

    /// v2 reader on a v1 header buffer: parses cleanly with
    /// `n_modalities == 0` and zero modality-table offsets.
    #[test]
    fn v1_file_via_v2_reader() {
        // Hand-build a v1-shaped 256-byte header on disk:
        // identical prefix through `front_catalog_length`, then 132
        // bytes of zero-valued reserved tail.
        let mut header = sample_header();
        header.format_version = 1;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = FileHeader::read_from(&mut cursor).unwrap();
        assert_eq!(decoded.format_version, 1);
        assert_eq!(decoded.n_modalities, 0);
        assert_eq!(decoded.modality_table_offset, 0);
        assert_eq!(decoded.modality_table_length, 0);
        assert_eq!(decoded.reserved, [0u8; 112]);
    }

    /// v2 round-trip: write a header with non-zero modality fields,
    /// read back, and assert all four new fields preserved.
    #[test]
    fn v2_round_trip_modality_fields() {
        let mut header = sample_header();
        header.n_modalities = 3;
        header.modality_table_offset = 0x4000;
        header.modality_table_length = 0x200;
        header.set_modalities();

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = FileHeader::read_from(&mut cursor).unwrap();
        assert_eq!(decoded.format_version, 2);
        assert_eq!(decoded.n_modalities, 3);
        assert_eq!(decoded.modality_table_offset, 0x4000);
        assert_eq!(decoded.modality_table_length, 0x200);
        assert!(decoded.has_modalities());
    }

    /// Cross-check: modality fields must be all-zero or all-non-zero.
    /// Disagreement raises `InvalidCatalog`.
    #[test]
    fn v2_partial_modality_fields_rejected() {
        let mut header = sample_header();
        header.n_modalities = 3;
        header.modality_table_offset = 0; // partial — should be rejected
        header.modality_table_length = 0x200;

        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        assert!(matches!(err, ScxError::InvalidCatalog(_)));
    }

    #[test]
    fn header_rejects_bad_header_length() {
        let mut header = sample_header();
        header.header_length = 255;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        match err {
            ScxError::InvalidCatalog(msg) => {
                assert!(msg.contains("header_length"), "got: {msg}");
            }
            other => panic!("expected InvalidCatalog, got {other:?}"),
        }
    }

    #[test]
    fn header_rejects_nonzero_reserved_padding() {
        let mut header = sample_header();
        header.reserved_padding = 1;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        match err {
            ScxError::InvalidCatalog(msg) => {
                assert!(msg.contains("reserved_padding"), "got: {msg}");
            }
            other => panic!("expected InvalidCatalog, got {other:?}"),
        }
    }

    #[test]
    fn header_rejects_nonzero_reserved_bytes() {
        let mut header = sample_header();
        header.reserved[0] = 1;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        match err {
            ScxError::InvalidCatalog(msg) => {
                assert!(msg.contains("reserved"), "got: {msg}");
            }
            other => panic!("expected InvalidCatalog, got {other:?}"),
        }
    }

    #[test]
    fn header_rejects_reserved_flag_bits() {
        // Bit 4 is explicitly reserved per the flags doc table.
        let mut header = sample_header();
        header.flags = 1 << 4;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        match err {
            ScxError::InvalidCatalog(msg) => {
                assert!(msg.contains("flag"), "got: {msg}");
            }
            other => panic!("expected InvalidCatalog, got {other:?}"),
        }
    }

    #[test]
    fn header_accepts_all_known_flag_bits() {
        // Sanity: every accessor-defined bit must be in KNOWN_FLAGS — guards
        // against drift if a future maintainer adds a `set_*` method but
        // forgets to widen KNOWN_FLAGS.
        let mut header = sample_header();
        header.flags = KNOWN_FLAGS;
        // KNOWN_FLAGS includes the modalities bit (7), so the cross-check
        // also expects n_modalities/offset/length to be all-set.
        header.n_modalities = 1;
        header.modality_table_offset = 0x4000;
        header.modality_table_length = 0x80;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = FileHeader::read_from(&mut cursor).expect("KNOWN_FLAGS must validate");
        assert_eq!(decoded.flags, KNOWN_FLAGS);
    }

    /// has_modalities flag round-trip via setters.
    #[test]
    fn has_modalities_flag_accessors() {
        let mut header = sample_header();
        assert!(!header.has_modalities());
        header.set_modalities();
        assert!(header.has_modalities());
        // Other flag accessors unaffected.
        assert!(!header.has_csc());
        assert!(!header.has_obsm());
        header.clear_modalities();
        assert!(!header.has_modalities());
    }

    #[test]
    fn flag_accessors() {
        let mut header = sample_header();

        // No flags set
        assert!(!header.has_csc());
        assert!(!header.has_bitmap());
        assert!(!header.has_obsm());
        assert!(!header.has_obsp());
        assert!(!header.has_deletion_vectors());

        // Set bit 0 (CSC) via setter
        header.set_csc();
        assert!(header.has_csc());
        assert!(!header.has_bitmap());

        // Set bit 1 (bitmap) via setter
        header.flags = 0;
        header.set_bitmap();
        assert!(!header.has_csc());
        assert!(header.has_bitmap());

        // Set bit 2 (obsm) via setter
        header.flags = 0;
        header.set_obsm();
        assert!(header.has_obsm());
        assert!(!header.has_obsp());

        // Set bit 3 (obsp) via setter
        header.flags = 0;
        header.set_obsp();
        assert!(header.has_obsp());
        assert!(!header.has_obsm());

        // Set bit 5 (deletion vectors) via setter
        header.flags = 0;
        header.set_deletion_vectors();
        assert!(header.has_deletion_vectors());

        // Multiple flags via setters
        header.flags = 0;
        header.set_csc();
        header.set_obsm();
        header.set_deletion_vectors();
        assert!(header.has_csc());
        assert!(!header.has_bitmap());
        assert!(header.has_obsm());
        assert!(!header.has_obsp());
        assert!(header.has_deletion_vectors());
    }
}
