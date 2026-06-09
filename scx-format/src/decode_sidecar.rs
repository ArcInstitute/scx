//! Decode-metadata sidecars for device/random-access shard decode.
//!
//! `DecodeSidecar` is intentionally metadata-only: it indexes the encoded
//! shard streams already stored in the corresponding CSR section. The first
//! shipped sidecar kind covers Scx1 CSR shards (Delta-Golomb indptr,
//! FOR-BP indices, Rice values), because those are the streams the GPU
//! decoder can consume without a host decompression round-trip.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use scx_codec::forbp::ForBpRowMetadata;
use scx_codec::rice::RiceBlockMetadata;
use scx_codec::{CodecId, Scx1DecodeMetadata, ValueEncoding};
use std::io::{Read, Write};

use crate::error::{validate_allocation, Result, ScxError};
use crate::section::SectionType;
use crate::versioned::VersionedSection;

/// Magic bytes at the start of every decode sidecar section.
pub const DECODE_SIDECAR_MAGIC: [u8; 4] = *b"SCXD";
/// Wire-format version for `DecodeSidecar`.
pub const DECODE_SIDECAR_VERSION: u16 = 1;
/// Sidecar kind: Scx1 CSR/LayerCSR decode metadata.
pub const DECODE_SIDECAR_KIND_SCX1_CSR: u8 = 1;
/// Maximum writer-side sidecar overhead before auto-emission is skipped.
pub const DEFAULT_DECODE_SIDECAR_MAX_OVERHEAD_RATIO: f64 = 0.25;

/// Per-row FOR-BP and value-stream positioning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeRowEntry {
    /// Number of stored values in this row.
    pub nnz: u32,
    /// Global value ordinal where this row starts within the shard.
    pub value_start: u64,
    /// FOR-BP `frame_min` for this row (`0` for empty rows).
    pub frame_min: u32,
    /// FOR-BP bit width for deltas in this row (`0` for empty/singleton rows).
    pub frame_bits: u8,
    /// `0` empty, `1` scalar bitpack, `2` BitPacker4x-compatible layout.
    pub index_packing: u8,
    /// Bit offset from the start of `indices_bytes` to this row's packed deltas.
    pub indices_bit_offset: u64,
}

/// Per-Rice-block metadata for value decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RiceBlockEntry {
    /// First value ordinal covered by this Rice block.
    pub value_start: u64,
    /// Number of values in this block.
    pub n_values: u16,
    /// Bit offset from the start of `values_bytes` to the block header byte.
    pub bit_offset: u64,
    /// Rice parameter stored in the low nibble of the block header.
    pub k: u8,
}

/// Decode metadata for one source CSR-like shard section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeSidecar {
    pub version: u16,
    pub kind: u8,
    pub target_section_type: SectionType,
    pub codec_id: u8,
    pub value_encoding: u8,
    pub index_dtype: u8,
    pub n_rows: u32,
    pub n_cols: u32,
    pub nnz: u64,
    pub major_start: u64,
    pub source_section_offset: u64,
    pub source_section_length: u64,
    pub source_section_checksum: [u8; 32],
    pub rows: Vec<DecodeRowEntry>,
    pub rice_blocks: Vec<RiceBlockEntry>,
}

impl VersionedSection for DecodeSidecar {
    const SECTION_NAME: &'static str = "decode sidecar";
    const CURRENT_VERSION: u16 = DECODE_SIDECAR_VERSION;
}

impl DecodeSidecar {
    /// Build an Scx1 decode sidecar from the **encoder-produced** metadata
    /// ([`scx_codec::Scx1DecodeMetadata`]) plus the encoded source section
    /// identity.
    ///
    /// The row/Rice-block offsets are exactly the ones the codec recorded while
    /// writing the bitstreams — the sidecar never re-derives codec internals, so
    /// it cannot drift from the actual on-disk layout. Returns `Ok(None)` for
    /// non-CSR targets or non-integer value encodings (no sidecar emitted).
    #[allow(clippy::too_many_arguments)]
    pub fn from_codec_metadata(
        meta: &Scx1DecodeMetadata,
        value_encoding: ValueEncoding,
        index_dtype: u8,
        n_cols: u32,
        major_start: u64,
        target_section_type: SectionType,
        source_section_offset: u64,
        source_section_length: u64,
        source_section_checksum: [u8; 32],
    ) -> Result<Option<Self>> {
        if !matches!(
            target_section_type,
            SectionType::CsrShard | SectionType::LayerCsrShard
        ) {
            return Ok(None);
        }
        if !value_encoding.is_integer() {
            return Ok(None);
        }

        let n_rows = meta.rows.len();
        if n_rows > u32::MAX as usize {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar n_rows {n_rows} exceeds u32::MAX"
            )));
        }

        let rows: Vec<DecodeRowEntry> = meta
            .rows
            .iter()
            .map(|r| DecodeRowEntry {
                nnz: r.nnz,
                value_start: r.value_start,
                frame_min: r.frame_min,
                frame_bits: r.frame_bits,
                index_packing: r.index_packing,
                indices_bit_offset: r.indices_bit_offset,
            })
            .collect();
        let rice_blocks: Vec<RiceBlockEntry> = meta
            .rice_blocks
            .iter()
            .map(|b| RiceBlockEntry {
                value_start: b.value_start,
                n_values: b.n_values,
                bit_offset: b.bit_offset,
                k: b.k,
            })
            .collect();
        let nnz: u64 = rows.iter().map(|r| r.nnz as u64).sum();

        Ok(Some(Self {
            version: DECODE_SIDECAR_VERSION,
            kind: DECODE_SIDECAR_KIND_SCX1_CSR,
            target_section_type,
            codec_id: CodecId::Scx1 as u8,
            value_encoding: value_encoding as u8,
            index_dtype,
            n_rows: n_rows as u32,
            n_cols,
            nnz,
            major_start,
            source_section_offset,
            source_section_length,
            source_section_checksum,
            rows,
            rice_blocks,
        }))
    }

    /// Reconstruct the codec-level decode metadata so the source shard can be
    /// decoded **through** the recorded offsets (`scx_codec::decode_scx1_with_metadata`)
    /// for the open-verify parity check.
    pub fn to_scx1_metadata(&self) -> Scx1DecodeMetadata {
        Scx1DecodeMetadata {
            rows: self
                .rows
                .iter()
                .map(|r| ForBpRowMetadata {
                    nnz: r.nnz,
                    value_start: r.value_start,
                    frame_min: r.frame_min,
                    frame_bits: r.frame_bits,
                    index_packing: r.index_packing,
                    indices_bit_offset: r.indices_bit_offset,
                })
                .collect(),
            rice_blocks: self
                .rice_blocks
                .iter()
                .map(|b| RiceBlockMetadata {
                    value_start: b.value_start,
                    n_values: b.n_values,
                    bit_offset: b.bit_offset,
                    k: b.k,
                })
                .collect(),
        }
    }

    pub fn with_source_offset(mut self, source_section_offset: u64) -> Self {
        self.source_section_offset = source_section_offset;
        self
    }

    pub fn estimated_encoded_size(&self) -> usize {
        4 + 2
            + 1
            + 1
            + 1
            + 1
            + 1
            + 2
            + 4
            + 4
            + 8
            + 8
            + 8
            + 8
            + 32
            + 4
            + 4
            + self.rows.len() * (4 + 8 + 4 + 1 + 1 + 2 + 8)
            + self.rice_blocks.len() * (8 + 2 + 1 + 1 + 8)
            + 32
    }

    pub fn overhead_ratio(&self) -> f64 {
        if self.source_section_length == 0 {
            return f64::INFINITY;
        }
        self.estimated_encoded_size() as f64 / self.source_section_length as f64
    }

    pub fn within_overhead_budget(&self, max_ratio: f64) -> bool {
        self.overhead_ratio() <= max_ratio
    }

    /// Serialise the sidecar, including the trailing BLAKE3-256 checksum.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut buf = Vec::with_capacity(self.estimated_encoded_size());
        buf.write_all(&DECODE_SIDECAR_MAGIC)?;
        buf.write_u16::<LittleEndian>(self.version)?;
        buf.write_u8(self.kind)?;
        buf.write_u8(self.target_section_type as u8)?;
        buf.write_u8(self.codec_id)?;
        buf.write_u8(self.value_encoding)?;
        buf.write_u8(self.index_dtype)?;
        buf.write_u16::<LittleEndian>(0)?; // reserved
        buf.write_u32::<LittleEndian>(self.n_rows)?;
        buf.write_u32::<LittleEndian>(self.n_cols)?;
        buf.write_u64::<LittleEndian>(self.nnz)?;
        buf.write_u64::<LittleEndian>(self.major_start)?;
        buf.write_u64::<LittleEndian>(self.source_section_offset)?;
        buf.write_u64::<LittleEndian>(self.source_section_length)?;
        buf.write_all(&self.source_section_checksum)?;
        buf.write_u32::<LittleEndian>(self.rows.len() as u32)?;
        buf.write_u32::<LittleEndian>(self.rice_blocks.len() as u32)?;
        for row in &self.rows {
            buf.write_u32::<LittleEndian>(row.nnz)?;
            buf.write_u64::<LittleEndian>(row.value_start)?;
            buf.write_u32::<LittleEndian>(row.frame_min)?;
            buf.write_u8(row.frame_bits)?;
            buf.write_u8(row.index_packing)?;
            buf.write_u16::<LittleEndian>(0)?;
            buf.write_u64::<LittleEndian>(row.indices_bit_offset)?;
        }
        for block in &self.rice_blocks {
            buf.write_u64::<LittleEndian>(block.value_start)?;
            buf.write_u16::<LittleEndian>(block.n_values)?;
            buf.write_u8(block.k)?;
            buf.write_u8(0)?;
            buf.write_u64::<LittleEndian>(block.bit_offset)?;
        }
        let checksum = blake3::hash(&buf);
        buf.extend_from_slice(checksum.as_bytes());
        // Guard against `estimated_encoded_size` (used for the overhead budget
        // and the write buffer reservation) drifting from the actual layout.
        debug_assert_eq!(
            buf.len(),
            self.estimated_encoded_size(),
            "decode sidecar estimated_encoded_size out of sync with write_to"
        );
        w.write_all(&buf)?;
        Ok(())
    }

    /// Read and validate a decode sidecar section.
    pub fn read_from<R: Read>(r: &mut R, section_len: usize) -> Result<Self> {
        const FIXED_PREFIX: usize =
            4 + 2 + 1 + 1 + 1 + 1 + 1 + 2 + 4 + 4 + 8 + 8 + 8 + 8 + 32 + 4 + 4;
        if section_len < FIXED_PREFIX + 32 {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar section too small: {section_len} bytes"
            )));
        }

        let mut all = vec![0u8; section_len];
        r.read_exact(&mut all)?;
        let payload_len = section_len - 32;
        let (payload, expected) = all.split_at(payload_len);
        let computed = blake3::hash(payload);
        if computed.as_bytes() != expected {
            return Err(ScxError::ChecksumMismatch {
                section: "decode sidecar".to_string(),
            });
        }

        let mut cur: &[u8] = payload;
        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if magic != DECODE_SIDECAR_MAGIC {
            return Err(ScxError::InvalidCatalog(format!(
                "bad decode sidecar magic: {magic:?}"
            )));
        }
        let version = cur.read_u16::<LittleEndian>()?;
        if version != DECODE_SIDECAR_VERSION {
            return Err(ScxError::UnsupportedSectionVersion {
                section: DecodeSidecar::SECTION_NAME,
                found: version,
                expected: DECODE_SIDECAR_VERSION,
            });
        }
        let kind = cur.read_u8()?;
        let section_type_raw = cur.read_u8()?;
        let target_section_type = SectionType::from_u8(section_type_raw)
            .ok_or(ScxError::UnknownSectionType(section_type_raw))?;
        let codec_id = cur.read_u8()?;
        let value_encoding = cur.read_u8()?;
        let index_dtype = cur.read_u8()?;
        let reserved = cur.read_u16::<LittleEndian>()?;
        if reserved != 0 {
            return Err(ScxError::InvalidCatalog(format!(
                "decode sidecar reserved field must be zero, got {reserved}"
            )));
        }
        let n_rows = cur.read_u32::<LittleEndian>()?;
        let n_cols = cur.read_u32::<LittleEndian>()?;
        let nnz = cur.read_u64::<LittleEndian>()?;
        let major_start = cur.read_u64::<LittleEndian>()?;
        let source_section_offset = cur.read_u64::<LittleEndian>()?;
        let source_section_length = cur.read_u64::<LittleEndian>()?;
        let mut source_section_checksum = [0u8; 32];
        cur.read_exact(&mut source_section_checksum)?;
        let n_row_entries = cur.read_u32::<LittleEndian>()? as usize;
        let n_rice_blocks = cur.read_u32::<LittleEndian>()? as usize;

        validate_allocation(
            n_row_entries.saturating_mul(28) + n_rice_blocks.saturating_mul(20),
            cur.len(),
        )?;

        let mut rows = Vec::with_capacity(n_row_entries);
        for _ in 0..n_row_entries {
            let nnz = cur.read_u32::<LittleEndian>()?;
            let value_start = cur.read_u64::<LittleEndian>()?;
            let frame_min = cur.read_u32::<LittleEndian>()?;
            let frame_bits = cur.read_u8()?;
            let index_packing = cur.read_u8()?;
            let reserved = cur.read_u16::<LittleEndian>()?;
            if reserved != 0 {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode row reserved field must be zero, got {reserved}"
                )));
            }
            let indices_bit_offset = cur.read_u64::<LittleEndian>()?;
            rows.push(DecodeRowEntry {
                nnz,
                value_start,
                frame_min,
                frame_bits,
                index_packing,
                indices_bit_offset,
            });
        }

        let mut rice_blocks = Vec::with_capacity(n_rice_blocks);
        for _ in 0..n_rice_blocks {
            let value_start = cur.read_u64::<LittleEndian>()?;
            let n_values = cur.read_u16::<LittleEndian>()?;
            let k = cur.read_u8()?;
            let reserved = cur.read_u8()?;
            if reserved != 0 {
                return Err(ScxError::InvalidCatalog(format!(
                    "decode Rice block reserved field must be zero, got {reserved}"
                )));
            }
            let bit_offset = cur.read_u64::<LittleEndian>()?;
            rice_blocks.push(RiceBlockEntry {
                value_start,
                n_values,
                bit_offset,
                k,
            });
        }

        validate_sidecar_shape(
            kind,
            codec_id,
            value_encoding,
            index_dtype,
            n_rows,
            nnz,
            &rows,
        )?;

        Ok(Self {
            version,
            kind,
            target_section_type,
            codec_id,
            value_encoding,
            index_dtype,
            n_rows,
            n_cols,
            nnz,
            major_start,
            source_section_offset,
            source_section_length,
            source_section_checksum,
            rows,
            rice_blocks,
        })
    }
}

fn validate_sidecar_shape(
    kind: u8,
    codec_id: u8,
    value_encoding: u8,
    index_dtype: u8,
    n_rows: u32,
    nnz: u64,
    rows: &[DecodeRowEntry],
) -> Result<()> {
    if kind != DECODE_SIDECAR_KIND_SCX1_CSR {
        return Err(ScxError::InvalidCatalog(format!(
            "unknown decode sidecar kind: {kind}"
        )));
    }
    if codec_id != CodecId::Scx1 as u8 {
        return Err(ScxError::InvalidCatalog(format!(
            "Scx1 decode sidecar must reference codec id 1, got {codec_id}"
        )));
    }
    let value_encoding_is_integer = ValueEncoding::from_u8(value_encoding)
        .map(|ve| ve.is_integer())
        .unwrap_or(false);
    if !value_encoding_is_integer {
        return Err(ScxError::InvalidCatalog(format!(
            "decode sidecar value encoding {value_encoding} is not an integer encoding"
        )));
    }
    if index_dtype > 1 {
        return Err(ScxError::InvalidCatalog(format!(
            "decode sidecar index_dtype must be 0 or 1, got {index_dtype}"
        )));
    }
    if rows.len() != n_rows as usize {
        return Err(ScxError::InvalidCatalog(format!(
            "decode sidecar row count {} != n_rows {n_rows}",
            rows.len()
        )));
    }
    let total_nnz: u64 = rows.iter().map(|r| r.nnz as u64).sum();
    if total_nnz != nnz {
        return Err(ScxError::InvalidCatalog(format!(
            "decode sidecar row nnz sum {total_nnz} != shard nnz {nnz}"
        )));
    }
    Ok(())
}
