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
    /// Row-shard of an `obsm/<name>` dense embedding (Arrow IPC).
    /// Section name: `obsm/<name>_shard_<idx>`. Same column schema as
    /// [`ObsmEmbedding`]; Arrow schema metadata carries `shard_idx` /
    /// `row_start` / `n_shard_rows` / `n_rows_total` so the reader can
    /// verify that the catalog entries form a contiguous, ordered cover
    /// of the logical matrix. Readers concatenate shards in `shard_idx`
    /// order to reconstruct the logical matrix; legacy single-section
    /// [`ObsmEmbedding`] files remain readable.
    ObsmEmbeddingShard = 20,
    /// Row-shard of a `varm/<name>` dense embedding (Arrow IPC). Mirror
    /// of [`ObsmEmbeddingShard`] for the `varm/` axis. Sharded along
    /// the var (gene) axis.
    VarmEmbeddingShard = 21,
    /// Row-shard of an `obsp/<name>` pairwise sparse matrix (Arrow IPC
    /// COO). Section name: `obsp/<name>_shard_<idx>`. Same column
    /// schema as [`ObspEmbedding`] (`row: Int32`, `col: Int32`,
    /// `data: Float32`) with `row` values stored as **global** indices
    /// (no shard-local renumbering). Readers concatenate shards in
    /// `shard_idx` order without offset application. Schema metadata
    /// carries `shard_idx` / `row_start` / `n_shard_rows` /
    /// `n_rows_total` / `n_rows` / `n_cols`.
    ObspEmbeddingShard = 22,
    /// Row-shard of a `varp/<name>` pairwise sparse matrix. Mirror of
    /// [`ObspEmbeddingShard`] for the `varp/` axis.
    VarpEmbeddingShard = 23,
    /// Row-shard of the obs metadata Arrow IPC batch.
    /// Section name: `obs_metadata/shard_<idx>`. Same payload schema as
    /// [`ObsMetadata`] (one Arrow IPC `RecordBatch`); Arrow schema
    /// metadata carries `shard_idx` / `row_start` / `n_shard_rows` /
    /// `n_rows_total` so the reader can verify the catalog entries form
    /// a contiguous, ordered cover of the logical obs table.
    ///
    /// Sharded obs lets atlas-scale merges and appends keep peak memory
    /// bounded to one shard at a time. Files written with sharded obs
    /// MUST NOT also write a single [`ObsMetadata`] section; the writer
    /// enforces this. Legacy single-section [`ObsMetadata`] files remain
    /// readable and are converted to a single shard on the first
    /// `scx append` that grows the file.
    ObsMetadataShard = 24,
    /// Row-shard of the var metadata Arrow IPC batch. Mirror of
    /// [`ObsMetadataShard`] for the var axis. Section name:
    /// `var_metadata/shard_<idx>`.
    VarMetadataShard = 25,
    // id 26 is reserved (formerly DecodeMetadataShard, the Scx1 decode sidecar,
    // removed once row-group framing became the default write layout). Legacy
    // files carrying section 26 are skipped via the unknown-section path and
    // full-decode to byte-identical output.
    /// Row-shard of the `adata.raw` count matrix (`raw/X`). Same row
    /// (obs) axis as the main `CsrShard` matrix but its OWN column
    /// (var) axis — `raw.n_vars` is typically larger than `n_vars`
    /// because `.raw` is captured before HVG subsetting. Section name:
    /// `raw/X_shard_<idx>`. Per-shard `index_dtype` is resolved against
    /// `raw_n_vars`, so a raw matrix with > 65535 genes uses u32 column
    /// indices even when the main matrix uses u16. Presence is signalled
    /// by the `has_raw` header flag.
    RawCsrShard = 27,
    /// The `adata.raw.var` DataFrame (Arrow IPC), companion to
    /// [`RawCsrShard`]. Section name: `raw/var`. Same payload schema as
    /// [`VarMetadata`] but describes the raw var axis.
    RawVarMetadata = 28,
    /// F1: condition/label-grouped sharding sidecar. One per file. Section
    /// name: `group_index`. JSON payload
    /// `{group_by, reference_shard, reference_labels, records[]}` where each
    /// record is `{label, shard, row_start, row_stop, role}` with `row_start`/
    /// `row_stop` as **global** output-row indices and `role` ∈
    /// `{"group","reference"}`. Written by `scx sort --group-by`; consumed by
    /// the grouped-read API (`read_group` / `read_reference` /
    /// `iter_group_shards`). Forward-compatible: pre-F1 readers skip it via the
    /// unknown-section path. Uses reserved id 29 (docs/format.md § section ids).
    GroupIndex = 29,
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
            20 => Some(Self::ObsmEmbeddingShard),
            21 => Some(Self::VarmEmbeddingShard),
            22 => Some(Self::ObspEmbeddingShard),
            23 => Some(Self::VarpEmbeddingShard),
            24 => Some(Self::ObsMetadataShard),
            25 => Some(Self::VarMetadataShard),
            // 26 reserved (formerly DecodeMetadataShard) — falls through to None
            // and is skipped by the catalog reader's unknown-section path.
            27 => Some(Self::RawCsrShard),
            28 => Some(Self::RawVarMetadata),
            29 => Some(Self::GroupIndex),
            _ => None,
        }
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
        assert_eq!(
            SectionType::from_u8(20),
            Some(SectionType::ObsmEmbeddingShard)
        );
        assert_eq!(
            SectionType::from_u8(21),
            Some(SectionType::VarmEmbeddingShard)
        );
        assert_eq!(
            SectionType::from_u8(22),
            Some(SectionType::ObspEmbeddingShard)
        );
        assert_eq!(
            SectionType::from_u8(23),
            Some(SectionType::VarpEmbeddingShard)
        );
        assert_eq!(
            SectionType::from_u8(24),
            Some(SectionType::ObsMetadataShard)
        );
        assert_eq!(
            SectionType::from_u8(25),
            Some(SectionType::VarMetadataShard)
        );
        // 26 reserved (formerly DecodeMetadataShard) — now unknown/None.
        assert_eq!(SectionType::from_u8(26), None);
        assert_eq!(SectionType::from_u8(27), Some(SectionType::RawCsrShard));
        assert_eq!(SectionType::from_u8(28), Some(SectionType::RawVarMetadata));
        assert_eq!(SectionType::from_u8(29), Some(SectionType::GroupIndex));
    }

    #[test]
    fn section_type_from_u8_unknown() {
        assert_eq!(SectionType::from_u8(30), None);
        assert_eq!(SectionType::from_u8(255), None);
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
