// ShardHeader + shard read/write (docs/format.md (CSR Shard))

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::error::{Result, ScxError};
use crate::section::SectionType;

/// Size of the shard header in bytes.
pub const SHARD_HEADER_SIZE: usize = 76;

/// Magic bytes identifying an SCX shard.
pub const SHARD_MAGIC: [u8; 4] = *b"SCXS";

/// Size of a single block index entry in bytes.
pub const BLOCK_INDEX_ENTRY_SIZE: usize = 22;

/// The 76-byte shard header that starts every CSR/CSC shard.
#[derive(Debug, Clone)]
pub struct ShardHeader {
    /// Magic bytes: b"SCXS"
    pub magic: [u8; 4],
    /// Shard format version
    pub shard_format_version: u8,
    /// Shard type (0=CSR, 1=CSC)
    pub shard_type: u8,
    /// Codec ID (0=None, 1=Scx1, 2=Zstd) — overrides file header
    pub codec_id: u8,
    /// Value encoding (0=u8, 1=u16, 2=u32, 3=f32, 4=f16)
    pub value_encoding: u8,
    /// Index dtype (0=u16, 1=u32)
    pub index_dtype: u8,
    /// Reserved flags
    pub reserved_flags: [u8; 3],
    /// Number of major-axis rows in this shard
    pub n_major: u32,
    /// Number of minor-axis columns
    pub n_minor: u32,
    /// Total non-zeros in this shard
    pub nnz: u64,
    /// Global major-axis index of the first major entry in this shard
    /// (not a byte offset).
    ///
    /// Axis-dependent semantics:
    /// - `CsrShard` / `LayerCsrShard` / `ObspCsrShard`: `row_start`
    ///   (global row index where this shard begins).
    /// - `CscShard`: `col_start` (global column index where this shard
    ///   begins).
    ///
    /// Use `ShardHeader::major_axis_start()` to get this field by its
    /// generic name.
    pub global_offset: u64,
    /// Relative offset to indptr data (from shard start)
    pub indptr_rel_offset: u32,
    /// Length of indptr data in bytes
    pub indptr_length: u32,
    /// Relative offset to indices data
    pub indices_rel_offset: u32,
    /// Length of indices data in bytes
    pub indices_length: u32,
    /// Relative offset to values data
    pub values_rel_offset: u32,
    /// Length of values data in bytes
    pub values_length: u32,
    /// Relative offset to block index
    pub block_index_rel_offset: u32,
    /// Length of block index in bytes
    pub block_index_length: u32,
    /// Truncated BLAKE3 checksum (8 bytes) of payload after header
    pub checksum: [u8; 8],
}

impl ShardHeader {
    /// Write the shard header to a writer in little-endian byte order.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.magic)?;
        w.write_u8(self.shard_format_version)?;
        w.write_u8(self.shard_type)?;
        w.write_u8(self.codec_id)?;
        w.write_u8(self.value_encoding)?;
        w.write_u8(self.index_dtype)?;
        w.write_all(&self.reserved_flags)?;
        w.write_u32::<LittleEndian>(self.n_major)?;
        w.write_u32::<LittleEndian>(self.n_minor)?;
        w.write_u64::<LittleEndian>(self.nnz)?;
        w.write_u64::<LittleEndian>(self.global_offset)?;
        w.write_u32::<LittleEndian>(self.indptr_rel_offset)?;
        w.write_u32::<LittleEndian>(self.indptr_length)?;
        w.write_u32::<LittleEndian>(self.indices_rel_offset)?;
        w.write_u32::<LittleEndian>(self.indices_length)?;
        w.write_u32::<LittleEndian>(self.values_rel_offset)?;
        w.write_u32::<LittleEndian>(self.values_length)?;
        w.write_u32::<LittleEndian>(self.block_index_rel_offset)?;
        w.write_u32::<LittleEndian>(self.block_index_length)?;
        w.write_all(&self.checksum)?;
        Ok(())
    }

    /// Read and validate a shard header from a reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != SHARD_MAGIC {
            return Err(ScxError::InvalidShardMagic);
        }

        let shard_format_version = r.read_u8()?;
        let shard_type = r.read_u8()?;
        let codec_id = r.read_u8()?;
        let value_encoding = r.read_u8()?;
        let index_dtype = r.read_u8()?;

        let mut reserved_flags = [0u8; 3];
        r.read_exact(&mut reserved_flags)?;

        let n_major = r.read_u32::<LittleEndian>()?;
        let n_minor = r.read_u32::<LittleEndian>()?;
        let nnz = r.read_u64::<LittleEndian>()?;
        let global_offset = r.read_u64::<LittleEndian>()?;
        let indptr_rel_offset = r.read_u32::<LittleEndian>()?;
        let indptr_length = r.read_u32::<LittleEndian>()?;
        let indices_rel_offset = r.read_u32::<LittleEndian>()?;
        let indices_length = r.read_u32::<LittleEndian>()?;
        let values_rel_offset = r.read_u32::<LittleEndian>()?;
        let values_length = r.read_u32::<LittleEndian>()?;
        let block_index_rel_offset = r.read_u32::<LittleEndian>()?;
        let block_index_length = r.read_u32::<LittleEndian>()?;

        let mut checksum = [0u8; 8];
        r.read_exact(&mut checksum)?;

        Ok(ShardHeader {
            magic,
            shard_format_version,
            shard_type,
            codec_id,
            value_encoding,
            index_dtype,
            reserved_flags,
            n_major,
            n_minor,
            nnz,
            global_offset,
            indptr_rel_offset,
            indptr_length,
            indices_rel_offset,
            indices_length,
            values_rel_offset,
            values_length,
            block_index_rel_offset,
            block_index_length,
            checksum,
        })
    }
}

/// Derive the on-disk `shard_type` byte from a `SectionType`.
///
/// Returns `1` for `CscShard` (column-major) and `0` for everything else
/// (CSR, layer CSR, obsp CSR). Used by writer call sites to label shards
/// correctly in their 76-byte headers.
pub fn derive_shard_type(section_type: SectionType) -> u8 {
    match section_type {
        SectionType::CscShard => 1,
        _ => 0,
    }
}

impl ShardHeader {
    /// Returns true if this shard is column-major (CSC).
    ///
    /// Honors both the on-disk `shard_type` byte (`1` = CSC) and the
    /// catalog `section_type` for forward/backward compatibility:
    /// pre-Phase-A files were written with `shard_type = 0` even for CSC
    /// shards (the catalog `section_type == CscShard` is what made them
    /// CSC). The catalog wins when the byte disagrees.
    pub fn is_csc(&self, catalog_section_type: SectionType) -> bool {
        self.shard_type == 1 || catalog_section_type == SectionType::CscShard
    }

    /// Strict v2 validation: when the catalog tags a shard as CSC,
    /// require `shard_type == 1`. Used by v2 read paths only.
    ///
    /// The `is_csc()` catalog-wins fallback survives on the v1 read
    /// path, where legacy files in the wild may carry the buggy
    /// `shard_type = 0` byte for CSC shards.
    ///
    /// On v2 reads, the writer is correct from day one
    /// (`derive_shard_type` returns `1` for `CscShard`), so any
    /// disagreement is corruption: surface a clear error rather than
    /// quietly accept it.
    pub fn validate_csc_strict(&self, catalog_section_type: SectionType) -> Result<()> {
        if catalog_section_type == SectionType::CscShard && self.shard_type != 1 {
            return Err(ScxError::InvalidShardType {
                expected: 1,
                got: self.shard_type,
                section_type: catalog_section_type as u8,
            });
        }
        Ok(())
    }

    /// Return the global major-axis start offset.
    ///
    /// This is `row_start` for CSR/LayerCsrShard/ObspCsrShard and
    /// `col_start` for CscShard. The on-disk field is named
    /// `global_offset` for axis-agnostic encoding.
    pub fn major_axis_start(&self) -> u64 {
        self.global_offset
    }
}

/// A single entry in the block index, pointing to a block of rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockIndexEntry {
    /// First row in this block
    pub row_start: u32,
    /// Number of rows in this block
    pub n_rows: u16,
    /// Byte offset into indptr stream for this block
    pub indptr_byte_offset: u32,
    /// Byte offset into indices stream for this block
    pub indices_byte_offset: u32,
    /// Byte offset into values stream for this block
    pub values_byte_offset: u32,
    /// Total non-zeros in this block
    pub nnz_in_block: u32,
}

impl BlockIndexEntry {
    /// Create a new BlockIndexEntry with validation against truncation.
    ///
    /// Returns `ScxError::BlockRowsOverflow` if `n_rows > u16::MAX`,
    /// or `ScxError::BlockNnzOverflow` if `nnz_in_block > u32::MAX`.
    pub fn new(
        row_start: u32,
        n_rows: u32,
        indptr_byte_offset: u32,
        indices_byte_offset: u32,
        values_byte_offset: u32,
        nnz_in_block: u64,
    ) -> Result<Self> {
        if n_rows > u16::MAX as u32 {
            return Err(crate::ScxError::BlockRowsOverflow(n_rows));
        }
        if nnz_in_block > u32::MAX as u64 {
            return Err(crate::ScxError::BlockNnzOverflow(nnz_in_block));
        }
        Ok(Self {
            row_start,
            n_rows: n_rows as u16,
            indptr_byte_offset,
            indices_byte_offset,
            values_byte_offset,
            nnz_in_block: nnz_in_block as u32,
        })
    }

    fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u32::<LittleEndian>(self.row_start)?;
        w.write_u16::<LittleEndian>(self.n_rows)?;
        w.write_u32::<LittleEndian>(self.indptr_byte_offset)?;
        w.write_u32::<LittleEndian>(self.indices_byte_offset)?;
        w.write_u32::<LittleEndian>(self.values_byte_offset)?;
        w.write_u32::<LittleEndian>(self.nnz_in_block)?;
        Ok(())
    }

    fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        Ok(BlockIndexEntry {
            row_start: r.read_u32::<LittleEndian>()?,
            n_rows: r.read_u16::<LittleEndian>()?,
            indptr_byte_offset: r.read_u32::<LittleEndian>()?,
            indices_byte_offset: r.read_u32::<LittleEndian>()?,
            values_byte_offset: r.read_u32::<LittleEndian>()?,
            nnz_in_block: r.read_u32::<LittleEndian>()?,
        })
    }
}

/// Block index for a shard, enabling random access to blocks of rows.
#[derive(Debug, Clone)]
pub struct BlockIndex {
    pub entries: Vec<BlockIndexEntry>,
}

impl BlockIndex {
    /// Write the block index: u32 count followed by entries.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u32::<LittleEndian>(self.entries.len() as u32)?;
        for entry in &self.entries {
            entry.write_to(w)?;
        }
        Ok(())
    }

    /// Read a block index: u32 count followed by that many entries.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let n_blocks = r.read_u32::<LittleEndian>()?;
        let mut entries = Vec::with_capacity(n_blocks as usize);
        for _ in 0..n_blocks {
            entries.push(BlockIndexEntry::read_from(r)?);
        }
        Ok(BlockIndex { entries })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn sample_shard_header() -> ShardHeader {
        ShardHeader {
            magic: SHARD_MAGIC,
            shard_format_version: 1,
            shard_type: 0,     // CSR
            codec_id: 1,       // Scx1
            value_encoding: 1, // u16
            index_dtype: 0,    // u16
            reserved_flags: [0; 3],
            n_major: 16384,
            n_minor: 30000,
            nnz: 2_500_000,
            global_offset: 4352,
            indptr_rel_offset: 76,
            indptr_length: 1000,
            indices_rel_offset: 1076,
            indices_length: 5_000_000,
            values_rel_offset: 5_001_076,
            values_length: 2_500_000,
            block_index_rel_offset: 7_501_076,
            block_index_length: 512,
            checksum: [0xAA, 0xBB, 0xCC, 0xDD, 0x11, 0x22, 0x33, 0x44],
        }
    }

    #[test]
    fn shard_header_round_trip() {
        let original = sample_shard_header();
        let mut buf = Vec::new();
        original.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let decoded = ShardHeader::read_from(&mut cursor).unwrap();

        assert_eq!(decoded.magic, original.magic);
        assert_eq!(decoded.shard_format_version, original.shard_format_version);
        assert_eq!(decoded.shard_type, original.shard_type);
        assert_eq!(decoded.codec_id, original.codec_id);
        assert_eq!(decoded.value_encoding, original.value_encoding);
        assert_eq!(decoded.index_dtype, original.index_dtype);
        assert_eq!(decoded.reserved_flags, original.reserved_flags);
        assert_eq!(decoded.n_major, original.n_major);
        assert_eq!(decoded.n_minor, original.n_minor);
        assert_eq!(decoded.nnz, original.nnz);
        assert_eq!(decoded.global_offset, original.global_offset);
        assert_eq!(decoded.indptr_rel_offset, original.indptr_rel_offset);
        assert_eq!(decoded.indptr_length, original.indptr_length);
        assert_eq!(decoded.indices_rel_offset, original.indices_rel_offset);
        assert_eq!(decoded.indices_length, original.indices_length);
        assert_eq!(decoded.values_rel_offset, original.values_rel_offset);
        assert_eq!(decoded.values_length, original.values_length);
        assert_eq!(
            decoded.block_index_rel_offset,
            original.block_index_rel_offset
        );
        assert_eq!(decoded.block_index_length, original.block_index_length);
        assert_eq!(decoded.checksum, original.checksum);
    }

    #[test]
    fn shard_header_writes_exactly_76_bytes() {
        let header = sample_shard_header();
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), SHARD_HEADER_SIZE);
    }

    #[test]
    fn reject_bad_shard_magic() {
        let mut header = sample_shard_header();
        header.magic = *b"BAD!";
        let mut buf = Vec::new();
        header.write_to(&mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let err = ShardHeader::read_from(&mut cursor).unwrap_err();
        assert!(matches!(err, ScxError::InvalidShardMagic));
    }

    #[test]
    fn block_index_round_trip() {
        let index = BlockIndex {
            entries: vec![
                BlockIndexEntry {
                    row_start: 0,
                    n_rows: 128,
                    indptr_byte_offset: 0,
                    indices_byte_offset: 0,
                    values_byte_offset: 0,
                    nnz_in_block: 50_000,
                },
                BlockIndexEntry {
                    row_start: 128,
                    n_rows: 128,
                    indptr_byte_offset: 512,
                    indices_byte_offset: 100_000,
                    values_byte_offset: 50_000,
                    nnz_in_block: 48_000,
                },
                BlockIndexEntry {
                    row_start: 256,
                    n_rows: 64,
                    indptr_byte_offset: 1024,
                    indices_byte_offset: 196_000,
                    values_byte_offset: 98_000,
                    nnz_in_block: 25_000,
                },
            ],
        };

        let mut buf = Vec::new();
        index.write_to(&mut buf).unwrap();

        // 4 bytes (count) + 3 * 22 bytes (entries) = 70 bytes
        assert_eq!(buf.len(), 4 + 3 * BLOCK_INDEX_ENTRY_SIZE);

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockIndex::read_from(&mut cursor).unwrap();

        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries, index.entries);
    }

    #[test]
    fn block_index_entry_new_valid() {
        let entry = BlockIndexEntry::new(0, 128, 0, 0, 0, 50_000).unwrap();
        assert_eq!(entry.n_rows, 128);
        assert_eq!(entry.nnz_in_block, 50_000);
    }

    #[test]
    fn block_index_entry_rejects_large_n_rows() {
        let err = BlockIndexEntry::new(0, 65536, 0, 0, 0, 100).unwrap_err();
        assert!(matches!(err, ScxError::BlockRowsOverflow(65536)));
    }

    #[test]
    fn block_index_entry_rejects_large_nnz() {
        let err = BlockIndexEntry::new(0, 128, 0, 0, 0, u32::MAX as u64 + 1).unwrap_err();
        assert!(matches!(err, ScxError::BlockNnzOverflow(_)));
    }

    #[test]
    fn derive_shard_type_csc_is_one() {
        assert_eq!(derive_shard_type(SectionType::CscShard), 1);
    }

    #[test]
    fn derive_shard_type_csr_variants_are_zero() {
        assert_eq!(derive_shard_type(SectionType::CsrShard), 0);
        assert_eq!(derive_shard_type(SectionType::LayerCsrShard), 0);
        assert_eq!(derive_shard_type(SectionType::ObspCsrShard), 0);
    }

    /// `validate_csc_strict` (the v2 strict path) accepts a correct
    /// CSC shard (byte == 1) and rejects shard_type != 1 with the
    /// `InvalidShardType` error.
    #[test]
    fn validate_csc_strict_accepts_byte_1() {
        let mut h = sample_shard_header();
        h.shard_type = 1;
        h.validate_csc_strict(SectionType::CscShard).unwrap();
    }

    #[test]
    fn validate_csc_strict_rejects_byte_0_for_csc() {
        let mut h = sample_shard_header();
        h.shard_type = 0;
        let err = h.validate_csc_strict(SectionType::CscShard).unwrap_err();
        match err {
            ScxError::InvalidShardType {
                expected,
                got,
                section_type,
            } => {
                assert_eq!(expected, 1);
                assert_eq!(got, 0);
                assert_eq!(section_type, SectionType::CscShard as u8);
            }
            other => panic!("expected InvalidShardType, got {other:?}"),
        }
    }

    #[test]
    fn validate_csc_strict_noop_for_non_csc() {
        let mut h = sample_shard_header();
        h.shard_type = 0;
        // CSR catalog → strict check is a no-op regardless of shard_type byte.
        h.validate_csc_strict(SectionType::CsrShard).unwrap();
    }

    #[test]
    fn shard_header_is_csc_byte_says_yes() {
        let mut h = sample_shard_header();
        h.shard_type = 1;
        // Even with a CsrShard catalog entry, byte=1 still flags CSC
        // (this branch is mostly defensive — the writer never produces
        // such a mismatch).
        assert!(h.is_csc(SectionType::CsrShard));
        assert!(h.is_csc(SectionType::CscShard));
    }

    #[test]
    fn shard_header_is_csc_catalog_says_yes_v1_legacy() {
        // Legacy v1 file: writer hardcoded shard_type=0 for CSC; the
        // catalog section_type is the source of truth.
        let mut h = sample_shard_header();
        h.shard_type = 0;
        assert!(h.is_csc(SectionType::CscShard));
        assert!(!h.is_csc(SectionType::CsrShard));
    }

    #[test]
    fn shard_header_major_axis_start_returns_global_offset() {
        let mut h = sample_shard_header();
        h.global_offset = 1234;
        assert_eq!(h.major_axis_start(), 1234);
    }

    #[test]
    fn block_index_empty() {
        let index = BlockIndex { entries: vec![] };

        let mut buf = Vec::new();
        index.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), 4); // just the count

        let mut cursor = Cursor::new(&buf);
        let decoded = BlockIndex::read_from(&mut cursor).unwrap();
        assert!(decoded.entries.is_empty());
    }
}
