// Section alignment, padding, types (docs/format.md)

/// Section types in the SCX file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SectionType {
    ObsMetadata = 0,
    ObsIndex = 1,
    VarMetadata = 2,
    VarIndex = 3,
    CsrShard = 4,
    CscShard = 5,
    BitmapShard = 6,
    LayerCsrShard = 7,
    ObsmEmbedding = 8,
    ObspCsrShard = 9,
    UnsBlob = 10,
    Provenance = 11,
    DeletionVectors = 12,
    ObsPredicateIndex = 13,
    VarPredicateIndex = 14,
    /// v2: named, ordered list of modalities (CITE-seq, multiome, …).
    /// One per file. See `crate::modality::ModalityTable`.
    ModalityTable = 15,
    /// v2: per-modality CSC sidecar for a layer (the column-major
    /// counterpart of `LayerCsrShard = 7`).
    LayerCscShard = 16,
    /// Dense var × components embedding (Arrow IPC), mirror of `ObsmEmbedding`.
    /// Section name prefix: `varm/`.
    VarmEmbedding = 17,
    /// obs × obs pairwise sparse matrix (Arrow IPC, COO format).
    /// Section name prefix: `obsp/`. Schema: `row: Int32`, `col: Int32`,
    /// `data: Float32` + metadata `n_rows`, `n_cols`.
    ObspEmbedding = 18,
    /// var × var pairwise sparse matrix (Arrow IPC, COO format).
    /// Section name prefix: `varp/`. Same wire format as `ObspEmbedding`.
    VarpEmbedding = 19,
}

impl SectionType {
    /// Convert a raw u8 to a `SectionType`, returning `None` for unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::ObsMetadata),
            1 => Some(Self::ObsIndex),
            2 => Some(Self::VarMetadata),
            3 => Some(Self::VarIndex),
            4 => Some(Self::CsrShard),
            5 => Some(Self::CscShard),
            6 => Some(Self::BitmapShard),
            7 => Some(Self::LayerCsrShard),
            8 => Some(Self::ObsmEmbedding),
            9 => Some(Self::ObspCsrShard),
            10 => Some(Self::UnsBlob),
            11 => Some(Self::Provenance),
            12 => Some(Self::DeletionVectors),
            13 => Some(Self::ObsPredicateIndex),
            14 => Some(Self::VarPredicateIndex),
            15 => Some(Self::ModalityTable),
            16 => Some(Self::LayerCscShard),
            17 => Some(Self::VarmEmbedding),
            18 => Some(Self::ObspEmbedding),
            19 => Some(Self::VarpEmbedding),
            _ => None,
        }
    }

    /// Returns true if `v` is a known section type ID (0..=19).
    pub fn is_known(v: u8) -> bool {
        v <= 19
    }
}

/// Round `offset` up to the next 8-byte boundary.
pub fn align_to_8(offset: u64) -> u64 {
    (offset + 7) & !7
}

/// Pre-allocated zero buffer for 8-byte alignment padding. Stack-based,
/// avoids the heap allocation of `vec![0u8; pad]` for at most 7 bytes.
const ZERO_PAD: [u8; 7] = [0u8; 7];

/// Write zero-padding bytes to align `current_offset` to an 8-byte
/// boundary. Returns the number of padding bytes written (0–7).
///
/// Use this instead of `writer.write_all(&vec![0u8; pad])` to avoid
/// a heap allocation for ≤ 7 zero bytes.
pub fn write_alignment_padding<W: std::io::Write>(
    w: &mut W,
    current_offset: u64,
) -> std::io::Result<usize> {
    let pad = (8 - (current_offset % 8) as usize) % 8;
    if pad > 0 {
        w.write_all(&ZERO_PAD[..pad])?;
    }
    Ok(pad)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_type_from_u8_known() {
        assert_eq!(SectionType::from_u8(0), Some(SectionType::ObsMetadata));
        assert_eq!(SectionType::from_u8(4), Some(SectionType::CsrShard));
        assert_eq!(SectionType::from_u8(12), Some(SectionType::DeletionVectors));
        assert_eq!(
            SectionType::from_u8(13),
            Some(SectionType::ObsPredicateIndex)
        );
        assert_eq!(
            SectionType::from_u8(14),
            Some(SectionType::VarPredicateIndex)
        );
        assert_eq!(SectionType::from_u8(15), Some(SectionType::ModalityTable));
        assert_eq!(SectionType::from_u8(16), Some(SectionType::LayerCscShard));
        assert_eq!(SectionType::from_u8(17), Some(SectionType::VarmEmbedding));
        assert_eq!(SectionType::from_u8(18), Some(SectionType::ObspEmbedding));
        assert_eq!(SectionType::from_u8(19), Some(SectionType::VarpEmbedding));
    }

    #[test]
    fn section_type_from_u8_unknown() {
        assert_eq!(SectionType::from_u8(20), None);
        assert_eq!(SectionType::from_u8(255), None);
    }

    #[test]
    fn section_type_is_known() {
        for v in 0..=19 {
            assert!(SectionType::is_known(v));
        }
        assert!(!SectionType::is_known(20));
        assert!(!SectionType::is_known(255));
    }

    #[test]
    fn align_to_8_already_aligned() {
        assert_eq!(align_to_8(0), 0);
        assert_eq!(align_to_8(8), 8);
        assert_eq!(align_to_8(16), 16);
        assert_eq!(align_to_8(4096), 4096);
    }

    #[test]
    fn align_to_8_needs_padding() {
        assert_eq!(align_to_8(1), 8);
        assert_eq!(align_to_8(7), 8);
        assert_eq!(align_to_8(9), 16);
        assert_eq!(align_to_8(4097), 4104);
    }
}
