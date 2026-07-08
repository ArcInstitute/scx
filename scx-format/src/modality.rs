// ModalityTable section + ModalityType / ModalityInfo.
//
// A v2 SCX file may carry one (and only one) `ModalityTable` section
// (`SectionType::ModalityTable = 15`). It is an ordered list of named
// modalities (RNA, ADT, ATAC, …), each with per-modality counts and
// codec hints. Catalog entries are routed to a modality via the new
// `FullCatalogEntry.modality_id: u8` field — `0` means "global" (the
// shared obs axis for all v1 files and single-modality v2 files); the
// numeric ids `1..=n_modalities` correspond positionally to the
// modalities listed in this table.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use std::io::{Read, Write};

use crate::checksum::blake3_hash;
use crate::error::{Result, ScxError};
use crate::versioned::VersionedSection;

/// Magic bytes identifying a `ModalityTable` section payload.
pub const MODALITY_TABLE_MAGIC: [u8; 4] = *b"MTBL";

/// Current `ModalityTable` payload-version. Bumped when the per-entry
/// or per-table on-disk schema changes.
pub const MODALITY_TABLE_VERSION: u16 = 1;

/// Maximum modality-name length in bytes (UTF-8). Names longer than
/// this are rejected by both writer and reader.
pub const MODALITY_NAME_MAX_BYTES: usize = 64;

/// Maximum number of modalities per file. The header carries
/// `n_modalities: u32` for alignment; in practice the cap is `u8::MAX`
/// because `FullCatalogEntry.modality_id` is a u8 and `0` is reserved
/// for "global".
pub const MAX_MODALITIES: u32 = 255;

/// Modality-type tag stored alongside each modality. Drives codec
/// auto-selection (Phase E) and downstream tools that need to dispatch
/// on biological modality (e.g. CITE-seq pipelines).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ModalityType {
    /// Gene-expression count matrix (RNA-seq, snRNA-seq, …).
    Rna = 0,
    /// Antibody-derived tags / surface protein abundance (ADT).
    Protein = 1,
    /// Open-chromatin peak counts (ATAC-seq).
    Atac = 2,
    /// Spatial transcriptomics — RNA modality + spatial obsm.
    Spatial = 3,
    /// DNA methylation (per-CpG or per-region counts).
    Methylation = 4,
    /// Custom / user-defined modality.
    Custom = 255,
}

impl ModalityType {
    /// Convert a raw u8 to a `ModalityType`. Returns `None` for
    /// unrecognised values; readers should surface that as a clear
    /// error rather than guessing.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Rna),
            1 => Some(Self::Protein),
            2 => Some(Self::Atac),
            3 => Some(Self::Spatial),
            4 => Some(Self::Methylation),
            255 => Some(Self::Custom),
            _ => None,
        }
    }

    /// Heuristically map a modality name to a `ModalityType`. Used when the
    /// caller has not supplied an explicit override; the recognised tokens
    /// follow the conventions adopted by scverse / 10x for CITE-seq and
    /// multiome files. This is the single source of truth shared by the
    /// h5mu conversion pipeline and the `pyscx` / `rscx` bindings — callers
    /// that fall through to it should record the inference (e.g.
    /// `ConvertWarning::ModalityTypeInferred`) so it is visible in provenance.
    pub fn infer_from_name(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("atac") || lower.contains("peak") || lower.contains("accessibility") {
            Self::Atac
        } else if lower.contains("adt")
            || lower.contains("protein")
            || lower.contains("antibody")
            || lower.contains("prot")
        {
            Self::Protein
        } else if lower.contains("spatial") {
            Self::Spatial
        } else if lower.contains("methyl") {
            Self::Methylation
        } else if lower == "rna" || lower == "gex" || lower.contains("expression") {
            Self::Rna
        } else {
            Self::Custom
        }
    }
}

/// Per-modality flag bits stored in `ModalityInfo.flags`. Provides
/// fast capability checks without scanning catalog entries for the
/// modality. Newtype around `u8` so the bit-set arithmetic stays
/// total-explicit at every call site (no `bitflags` macro
/// dependency).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ModalityFlags(pub u8);

impl ModalityFlags {
    /// Empty flag set.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Modality has at least one CSC sidecar shard.
    pub const HAS_CSC: u8 = 1 << 0;
    /// Modality has at least one obsm embedding.
    pub const HAS_OBSM: u8 = 1 << 1;
    /// Modality has at least one obsp graph.
    pub const HAS_OBSP: u8 = 1 << 2;
    /// Modality has at least one named layer.
    pub const HAS_LAYERS: u8 = 1 << 3;
    /// Modality has uns metadata.
    pub const HAS_UNS: u8 = 1 << 4;
    /// Phase 5b: modality has at least one detection-bitmap sidecar shard.
    pub const HAS_BITMAP: u8 = 1 << 5;

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits)
    }

    pub fn has_csc(self) -> bool {
        self.0 & Self::HAS_CSC != 0
    }
    pub fn has_obsm(self) -> bool {
        self.0 & Self::HAS_OBSM != 0
    }
    pub fn has_obsp(self) -> bool {
        self.0 & Self::HAS_OBSP != 0
    }
    pub fn has_layers(self) -> bool {
        self.0 & Self::HAS_LAYERS != 0
    }
    pub fn has_uns(self) -> bool {
        self.0 & Self::HAS_UNS != 0
    }
    pub fn has_bitmap(self) -> bool {
        self.0 & Self::HAS_BITMAP != 0
    }

    pub fn set_csc(&mut self) {
        self.0 |= Self::HAS_CSC;
    }
    pub fn set_obsm(&mut self) {
        self.0 |= Self::HAS_OBSM;
    }
    pub fn set_obsp(&mut self) {
        self.0 |= Self::HAS_OBSP;
    }
    pub fn set_layers(&mut self) {
        self.0 |= Self::HAS_LAYERS;
    }
    pub fn set_uns(&mut self) {
        self.0 |= Self::HAS_UNS;
    }
    pub fn set_bitmap(&mut self) {
        self.0 |= Self::HAS_BITMAP;
    }
}

/// Per-modality information block. Mirrors the on-disk per-modality
/// record in the `ModalityTable` section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModalityInfo {
    /// Modality name (UTF-8, ≤ `MODALITY_NAME_MAX_BYTES`, unique
    /// within a file). Used as the key in MuData round-trips.
    pub name: String,
    /// Modality biological type (drives codec defaults).
    pub modality_type: ModalityType,
    /// Default codec id for this modality's primary X shards.
    /// Stored as raw u8 to avoid a hard dependency on
    /// `scx_codec::CodecId` from this module.
    pub default_codec_id: u8,
    /// Default value-encoding for this modality's primary X shards.
    /// Stored as raw u8 (mirrors `ValueEncoding` in `scx_codec`).
    pub default_value_encoding: u8,
    /// Number of variables (genes / proteins / peaks / …) in this
    /// modality.
    pub n_vars: u64,
    /// Total non-zeros across all CSR shards of this modality.
    pub nnz: u64,
    /// Number of CSR shards belonging to this modality.
    pub n_csr_shards: u32,
    /// Number of CSC sidecar shards belonging to this modality.
    pub n_csc_shards: u32,
    /// Per-modality capability flags.
    pub flags: ModalityFlags,
}

/// Bytes consumed by a single modality record on disk, EXCLUDING the
/// variable-length name. The name is encoded as `u8 length` + that
/// many UTF-8 bytes.
///
/// Fixed prefix layout (after `name_length` + `name` bytes):
/// `modality_type(1) + default_codec_id(1) + default_value_encoding(1)
/// + reserved_flags(1) + n_vars(8) + nnz(8) + n_csr_shards(4)
/// + n_csc_shards(4) + flags(1) + reserved[7]`.
const MODALITY_RECORD_FIXED_BYTES: usize = 1 + 1 + 1 + 1 + 8 + 8 + 4 + 4 + 1 + 7;

/// In-memory representation of a `ModalityTable` section. The order of
/// `entries` is significant — modality_id `i + 1` (1-based) refers to
/// `entries[i]` (0-based).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModalityTable {
    pub entries: Vec<ModalityInfo>,
}

impl VersionedSection for ModalityTable {
    const SECTION_NAME: &'static str = "ModalityTable";
    const CURRENT_VERSION: u16 = MODALITY_TABLE_VERSION;
}

impl ModalityTable {
    /// Convenience constructor.
    pub fn new(entries: Vec<ModalityInfo>) -> Self {
        Self { entries }
    }

    /// Number of modalities in this table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Resolve a modality name to its 1-based `modality_id`. Returns
    /// `None` if the name is not registered.
    pub fn id_of(&self, name: &str) -> Option<u8> {
        self.entries
            .iter()
            .position(|e| e.name == name)
            .map(|idx| (idx + 1) as u8)
    }

    /// Look up modality info by 1-based `modality_id`. `id == 0`
    /// (global) returns `None`; ids beyond the table also return
    /// `None`.
    pub fn info_of(&self, modality_id: u8) -> Option<&ModalityInfo> {
        if modality_id == 0 {
            return None;
        }
        self.entries.get((modality_id - 1) as usize)
    }

    /// Validate a candidate modality name. Returns the name on
    /// success; raises `ScxError::InvalidCatalog` on empty / overlong
    /// strings or for non-UTF-8 (the latter cannot occur given a
    /// `&str` argument, but we also enforce no interior NULs).
    pub fn validate_name(name: &str) -> Result<&str> {
        if name.is_empty() {
            return Err(ScxError::InvalidCatalog(
                "modality name must be non-empty".to_string(),
            ));
        }
        if name.len() > MODALITY_NAME_MAX_BYTES {
            return Err(ScxError::InvalidCatalog(format!(
                "modality name '{name}' exceeds {MODALITY_NAME_MAX_BYTES} bytes ({} bytes)",
                name.len()
            )));
        }
        if name.bytes().any(|b| b == 0) {
            return Err(ScxError::InvalidCatalog(format!(
                "modality name '{name}' contains an embedded NUL byte"
            )));
        }
        Ok(name)
    }

    /// Serialize the modality table to its on-disk representation.
    /// Layout (LE throughout):
    ///   - magic "MTBL" (4 bytes)
    ///   - version u16
    ///   - n_modalities u16
    ///   - per modality:
    ///     u8 name_length, bytes name[name_length],
    ///     u8 modality_type, u8 default_codec_id,
    ///     u8 default_value_encoding, u8 reserved_flags = 0,
    ///     u64 n_vars, u64 nnz,
    ///     u32 n_csr_shards, u32 n_csc_shards,
    ///     u8 flags, u8 reserved[7]
    ///   - 4-byte BLAKE3-truncated checksum of all preceding bytes.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        if self.entries.len() > MAX_MODALITIES as usize {
            return Err(ScxError::InvalidCatalog(format!(
                "modality count {} exceeds MAX_MODALITIES ({MAX_MODALITIES})",
                self.entries.len()
            )));
        }

        // First serialize everything-but-checksum to an in-memory
        // buffer so we can hash exactly the bytes that hit disk.
        let mut buf = Vec::new();
        buf.write_all(&MODALITY_TABLE_MAGIC)?;
        buf.write_u16::<LittleEndian>(MODALITY_TABLE_VERSION)?;
        buf.write_u16::<LittleEndian>(self.entries.len() as u16)?;

        for entry in &self.entries {
            Self::validate_name(&entry.name)?;
            let name_bytes = entry.name.as_bytes();
            buf.write_u8(name_bytes.len() as u8)?;
            buf.write_all(name_bytes)?;
            buf.write_u8(entry.modality_type as u8)?;
            buf.write_u8(entry.default_codec_id)?;
            buf.write_u8(entry.default_value_encoding)?;
            buf.write_u8(0)?; // reserved_flags
            buf.write_u64::<LittleEndian>(entry.n_vars)?;
            buf.write_u64::<LittleEndian>(entry.nnz)?;
            buf.write_u32::<LittleEndian>(entry.n_csr_shards)?;
            buf.write_u32::<LittleEndian>(entry.n_csc_shards)?;
            buf.write_u8(entry.flags.bits())?;
            buf.write_all(&[0u8; 7])?; // reserved[7]
        }

        let full_hash = blake3_hash(&buf);
        let mut truncated = [0u8; 4];
        truncated.copy_from_slice(&full_hash[..4]);

        w.write_all(&buf)?;
        w.write_all(&truncated)?;
        Ok(())
    }

    /// Deserialize a `ModalityTable` from its on-disk bytes. The
    /// `total_len` is the section's full byte length (including the
    /// trailing 4-byte checksum) as recorded in the file header /
    /// catalog. Validates the magic, version, payload-checksum, and
    /// per-modality name uniqueness.
    pub fn read_from<R: Read>(r: &mut R, total_len: usize) -> Result<Self> {
        if total_len < 4 + 2 + 2 + 4 {
            return Err(ScxError::InvalidCatalog(format!(
                "ModalityTable section too short: {total_len} bytes (min {})",
                4 + 2 + 2 + 4
            )));
        }
        let mut all_bytes = vec![0u8; total_len];
        r.read_exact(&mut all_bytes)?;
        let payload_len = total_len - 4;
        let payload = &all_bytes[..payload_len];
        let on_disk_checksum = &all_bytes[payload_len..];
        let computed_full = blake3_hash(payload);
        if computed_full[..4] != *on_disk_checksum {
            return Err(ScxError::ChecksumMismatch {
                section: "modality_table".to_string(),
            });
        }

        let mut cur = std::io::Cursor::new(payload);

        let mut magic = [0u8; 4];
        cur.read_exact(&mut magic)?;
        if magic != MODALITY_TABLE_MAGIC {
            return Err(ScxError::InvalidCatalog(format!(
                "unexpected ModalityTable magic: {magic:?}"
            )));
        }

        let version = cur.read_u16::<LittleEndian>()?;
        Self::check_version(version)?;

        let n_modalities = cur.read_u16::<LittleEndian>()? as usize;
        if n_modalities > MAX_MODALITIES as usize {
            return Err(ScxError::InvalidCatalog(format!(
                "ModalityTable n_modalities {n_modalities} exceeds MAX_MODALITIES \
                 ({MAX_MODALITIES})"
            )));
        }

        let mut entries = Vec::with_capacity(n_modalities);
        for _ in 0..n_modalities {
            let name_len = cur.read_u8()? as usize;
            if name_len == 0 || name_len > MODALITY_NAME_MAX_BYTES {
                return Err(ScxError::InvalidCatalog(format!(
                    "ModalityTable entry has invalid name length {name_len}"
                )));
            }
            let mut name_bytes = vec![0u8; name_len];
            cur.read_exact(&mut name_bytes)?;
            let name = String::from_utf8(name_bytes)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            Self::validate_name(&name)?;
            let modality_type_raw = cur.read_u8()?;
            let modality_type = ModalityType::from_u8(modality_type_raw).ok_or_else(|| {
                ScxError::InvalidCatalog(format!("unknown modality_type {modality_type_raw}"))
            })?;
            let default_codec_id = cur.read_u8()?;
            let default_value_encoding = cur.read_u8()?;
            let _reserved_flags = cur.read_u8()?;
            let n_vars = cur.read_u64::<LittleEndian>()?;
            let nnz = cur.read_u64::<LittleEndian>()?;
            let n_csr_shards = cur.read_u32::<LittleEndian>()?;
            let n_csc_shards = cur.read_u32::<LittleEndian>()?;
            let flags_raw = cur.read_u8()?;
            let flags = ModalityFlags::from_bits_truncate(flags_raw);
            let mut reserved = [0u8; 7];
            cur.read_exact(&mut reserved)?;

            entries.push(ModalityInfo {
                name,
                modality_type,
                default_codec_id,
                default_value_encoding,
                n_vars,
                nnz,
                n_csr_shards,
                n_csc_shards,
                flags,
            });
        }

        // Reject duplicate names (case-sensitive).
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                if entries[i].name == entries[j].name {
                    return Err(ScxError::InvalidCatalog(format!(
                        "duplicate modality name '{}'",
                        entries[i].name
                    )));
                }
            }
        }

        Ok(Self { entries })
    }

    /// Compute the exact serialized byte length for this table (used
    /// by writers to position the catalog after the section).
    pub fn serialized_len(&self) -> usize {
        let mut len = 4 + 2 + 2; // magic + version + n_modalities
        for entry in &self.entries {
            len += 1 + entry.name.len() + MODALITY_RECORD_FIXED_BYTES;
        }
        len += 4; // truncated BLAKE3
        len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn check_version_rejects_future() {
        assert!(matches!(
            ModalityTable::check_version(MODALITY_TABLE_VERSION + 1),
            Err(ScxError::UnsupportedSectionVersion { .. })
        ));
        assert!(ModalityTable::check_version(MODALITY_TABLE_VERSION).is_ok());
    }

    #[test]
    fn infer_from_name_maps_known_tokens() {
        use ModalityType::*;
        // ATAC (incl. the `accessibility` token the old pyscx/rscx copies lacked)
        assert_eq!(ModalityType::infer_from_name("atac"), Atac);
        assert_eq!(ModalityType::infer_from_name("ATAC"), Atac);
        assert_eq!(ModalityType::infer_from_name("peaks"), Atac);
        assert_eq!(ModalityType::infer_from_name("accessibility"), Atac);
        // Protein (incl. the `prot` token the old copies lacked)
        assert_eq!(ModalityType::infer_from_name("adt"), Protein);
        assert_eq!(ModalityType::infer_from_name("protein"), Protein);
        assert_eq!(ModalityType::infer_from_name("antibody_capture"), Protein);
        assert_eq!(ModalityType::infer_from_name("prot"), Protein);
        // Spatial / Methylation
        assert_eq!(ModalityType::infer_from_name("spatial"), Spatial);
        assert_eq!(ModalityType::infer_from_name("methylation"), Methylation);
        // RNA
        assert_eq!(ModalityType::infer_from_name("rna"), Rna);
        assert_eq!(ModalityType::infer_from_name("gex"), Rna);
        assert_eq!(ModalityType::infer_from_name("gene_expression"), Rna);
        // Unrecognised → Custom
        assert_eq!(ModalityType::infer_from_name("mystery_assay"), Custom);
        assert_eq!(ModalityType::infer_from_name(""), Custom);
    }

    fn sample_info(name: &str, mtype: ModalityType, n_vars: u64) -> ModalityInfo {
        ModalityInfo {
            name: name.to_string(),
            modality_type: mtype,
            default_codec_id: 1, // Scx1
            default_value_encoding: 0,
            n_vars,
            nnz: n_vars * 100,
            n_csr_shards: 2,
            n_csc_shards: 0,
            flags: ModalityFlags::empty(),
        }
    }

    #[test]
    fn round_trip_empty_table() {
        let table = ModalityTable::new(vec![]);
        let mut buf = Vec::new();
        table.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), table.serialized_len());

        let mut cur = Cursor::new(&buf);
        let decoded = ModalityTable::read_from(&mut cur, buf.len()).unwrap();
        assert_eq!(decoded, table);
    }

    #[test]
    fn round_trip_three_modalities() {
        let table = ModalityTable::new(vec![
            sample_info("rna", ModalityType::Rna, 30_000),
            sample_info("adt", ModalityType::Protein, 200),
            sample_info("atac", ModalityType::Atac, 100_000),
        ]);
        let mut buf = Vec::new();
        table.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), table.serialized_len());

        let mut cur = Cursor::new(&buf);
        let decoded = ModalityTable::read_from(&mut cur, buf.len()).unwrap();
        assert_eq!(decoded, table);

        assert_eq!(decoded.id_of("adt"), Some(2));
        assert_eq!(
            decoded.info_of(2).unwrap().modality_type,
            ModalityType::Protein
        );
        assert_eq!(decoded.info_of(0), None);
        assert_eq!(decoded.info_of(99), None);
    }

    #[test]
    fn corrupt_checksum_rejected() {
        let table = ModalityTable::new(vec![sample_info("rna", ModalityType::Rna, 10)]);
        let mut buf = Vec::new();
        table.write_to(&mut buf).unwrap();
        // Flip a byte in the payload (not in the trailing 4-byte
        // checksum).
        buf[10] ^= 0xFF;
        let mut cur = Cursor::new(&buf);
        let err = ModalityTable::read_from(&mut cur, buf.len()).unwrap_err();
        assert!(matches!(err, ScxError::ChecksumMismatch { .. }));
    }

    #[test]
    fn empty_name_rejected() {
        let err = ModalityTable::validate_name("").unwrap_err();
        assert!(matches!(err, ScxError::InvalidCatalog(_)));
    }

    #[test]
    fn overlong_name_rejected() {
        let long = "a".repeat(MODALITY_NAME_MAX_BYTES + 1);
        let err = ModalityTable::validate_name(&long).unwrap_err();
        assert!(matches!(err, ScxError::InvalidCatalog(_)));
    }

    #[test]
    fn duplicate_name_rejected() {
        let table = ModalityTable::new(vec![
            sample_info("rna", ModalityType::Rna, 10),
            sample_info("rna", ModalityType::Atac, 20),
        ]);
        // Writer accepts (just emits the bytes); reader rejects.
        let mut buf = Vec::new();
        table.write_to(&mut buf).unwrap();
        let mut cur = Cursor::new(&buf);
        let err = ModalityTable::read_from(&mut cur, buf.len()).unwrap_err();
        match err {
            ScxError::InvalidCatalog(msg) => {
                assert!(msg.contains("duplicate modality name"));
            }
            other => panic!("expected InvalidCatalog, got {other:?}"),
        }
    }

    #[test]
    fn id_of_global_modality_is_none() {
        let table = ModalityTable::new(vec![sample_info("rna", ModalityType::Rna, 10)]);
        // 0 is reserved for "global" — there is no name for it.
        assert_eq!(table.id_of("global"), None);
        assert_eq!(table.info_of(0), None);
    }

    #[test]
    fn modality_type_round_trip() {
        for v in [0u8, 1, 2, 3, 4, 255] {
            let mt = ModalityType::from_u8(v).unwrap();
            assert_eq!(mt as u8, v);
        }
        assert_eq!(ModalityType::from_u8(99), None);
    }

    #[test]
    fn modality_flags_round_trip() {
        let mut f = ModalityFlags::empty();
        f.set_csc();
        f.set_layers();
        let bits = f.bits();
        let restored = ModalityFlags::from_bits_truncate(bits);
        assert_eq!(restored, f);
        assert!(restored.has_csc());
        assert!(restored.has_layers());
        assert!(!restored.has_obsm());
    }
}
