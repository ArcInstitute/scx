// FileHeader struct + read/write (SPEC §3.1)

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::error::{Result, ScxError};

/// Size of the file header in bytes.
pub const HEADER_SIZE: usize = 256;

/// Magic bytes identifying an SCX file.
pub const MAGIC: [u8; 4] = *b"SCX\x01";

/// The 256-byte file header that starts every SCX file.
#[derive(Debug, Clone)]
pub struct FileHeader {
    /// Magic bytes: b"SCX\x01"
    pub magic: [u8; 4],
    /// Format version (currently 1)
    pub format_version: u16,
    /// Header length in bytes (always 256)
    pub header_length: u16,
    /// Bit-field flags
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
    /// Reserved bytes (must be zero)
    pub reserved: [u8; 132],
}

impl FileHeader {
    /// Write the header to a writer in little-endian byte order.
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
        w.write_all(&self.reserved)?;
        Ok(())
    }

    /// Read and validate a header from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(ScxError::InvalidMagic);
        }

        let format_version = r.read_u16::<LittleEndian>()?;
        if format_version > 1 {
            return Err(ScxError::UnsupportedVersion);
        }

        let header_length = r.read_u16::<LittleEndian>()?;
        let flags = r.read_u32::<LittleEndian>()?;
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
        let root_catalog_offset = r.read_u64::<LittleEndian>()?;
        let root_catalog_length = r.read_u64::<LittleEndian>()?;
        let full_catalog_offset = r.read_u64::<LittleEndian>()?;
        let full_catalog_length = r.read_u64::<LittleEndian>()?;
        let manifest_sequence = r.read_u64::<LittleEndian>()?;
        let prev_catalog_offset = r.read_u64::<LittleEndian>()?;
        let file_checksum = r.read_u64::<LittleEndian>()?;
        let front_catalog_offset = r.read_u64::<LittleEndian>()?;
        let front_catalog_length = r.read_u64::<LittleEndian>()?;

        let mut reserved = [0u8; 132];
        r.read_exact(&mut reserved)?;

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
            reserved,
        })
    }

    /// Returns true if the CSC flag (bit 0) is set.
    pub fn has_csc(&self) -> bool {
        self.flags & (1 << 0) != 0
    }

    /// Returns true if the bitmap flag (bit 1) is set.
    pub fn has_bitmap(&self) -> bool {
        self.flags & (1 << 1) != 0
    }

    /// Returns true if the obsm flag (bit 2) is set.
    pub fn has_obsm(&self) -> bool {
        self.flags & (1 << 2) != 0
    }

    /// Returns true if the obsp flag (bit 3) is set.
    pub fn has_obsp(&self) -> bool {
        self.flags & (1 << 3) != 0
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_header() -> FileHeader {
        FileHeader {
            magic: MAGIC,
            format_version: 1,
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
            reserved: [0u8; 132],
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
        assert_eq!(decoded.reserved, [0u8; 132]);
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
        let mut header = sample_header();
        header.format_version = 2;
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = FileHeader::read_from(&mut cursor).unwrap_err();
        assert!(matches!(err, ScxError::UnsupportedVersion));
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

        // Set bit 0 (CSC)
        header.flags = 1 << 0;
        assert!(header.has_csc());
        assert!(!header.has_bitmap());

        // Set bit 1 (bitmap)
        header.flags = 1 << 1;
        assert!(!header.has_csc());
        assert!(header.has_bitmap());

        // Set bit 2 (obsm)
        header.flags = 1 << 2;
        assert!(header.has_obsm());

        // Set bit 3 (obsp)
        header.flags = 1 << 3;
        assert!(header.has_obsp());

        // Set bit 5 (deletion vectors)
        header.flags = 1 << 5;
        assert!(header.has_deletion_vectors());

        // Multiple flags
        header.flags = (1 << 0) | (1 << 2) | (1 << 5);
        assert!(header.has_csc());
        assert!(!header.has_bitmap());
        assert!(header.has_obsm());
        assert!(!header.has_obsp());
        assert!(header.has_deletion_vectors());
    }
}
