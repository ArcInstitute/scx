// Section alignment, padding, types (SPEC §3)

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
            _ => None,
        }
    }

    /// Returns true if `v` is a known section type ID (0..=12).
    pub fn is_known(v: u8) -> bool {
        v <= 12
    }
}

/// Round `offset` up to the next 8-byte boundary.
pub fn align_to_8(offset: u64) -> u64 {
    (offset + 7) & !7
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn section_type_from_u8_known() {
        assert_eq!(SectionType::from_u8(0), Some(SectionType::ObsMetadata));
        assert_eq!(SectionType::from_u8(4), Some(SectionType::CsrShard));
        assert_eq!(SectionType::from_u8(12), Some(SectionType::DeletionVectors));
    }

    #[test]
    fn section_type_from_u8_unknown() {
        assert_eq!(SectionType::from_u8(13), None);
        assert_eq!(SectionType::from_u8(255), None);
    }

    #[test]
    fn section_type_is_known() {
        for v in 0..=12 {
            assert!(SectionType::is_known(v));
        }
        assert!(!SectionType::is_known(13));
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
