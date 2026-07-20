// Deletion vectors (docs/format.md (Deletion Vectors))
//
// v2 binary layout: u8 version (=2), u32 n_entries,
// then per-entry: u8 modality_id, [u8; 3] reserved, u32 bitmap_len,
// [u8; bitmap_len] roaring bitmap of GLOBAL obs row indices.
//
// v1 (legacy, read-only): u8 version (=1), u32 n_shards,
// then per-shard: u32 shard_id, u32 bitmap_len, [u8; bitmap_len] roaring bitmap
// of SHARD-LOCAL row indices. v1 is folded to a global bitmap at the reader
// boundary via `fold_v1_to_global`.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use roaring::RoaringBitmap;
use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::error::{Result, ScxError};
use crate::versioned::VersionedSection;

/// On-disk version of the deletion-vectors section (current writer version).
pub const DV_VERSION: u8 = 2;

/// The legacy v1 deletion-vectors layout — still read (and folded to the v2
/// global representation), never written.
pub const DV_V1: u8 = 1;

/// `modality_id` sentinel for a whole-cell (global) deletion — applies to
/// every modality. `>= 1` scopes a deletion to that modality only (wire-format
/// headroom; shipped writers only populate the global key).
pub const DV_GLOBAL: u8 = 0;

/// Deletion vectors section: logically deleted rows keyed by modality.
///
/// v2 stores a `BTreeMap<modality_id, bitmap>` where each bitmap holds
/// **global obs row indices**. `modality_id == 0` (`DV_GLOBAL`) is the
/// whole-cell delete applied to every modality; `>= 1` is reserved for
/// modality-scoped deletion. Because every modality's CSR shards independently
/// tile `[0, n_obs)`, a global bitmap applies identically to any modality — no
/// shard-positional index, no `partition_point`, no overlap hazard.
///
/// Legacy v1 files (per-shard, shard-local bitmaps) are parsed into a private
/// transient (`v1_shards`) by [`Self::read_from`] and converted to the global
/// representation by [`Self::fold_v1_to_global`] at the reader boundary, so all
/// downstream consumers only ever see the v2 shape.
#[derive(Debug, Clone)]
pub struct DeletionVectors {
    pub dv_version: u8,
    /// Deletions keyed by `modality_id` (`0` = all modalities). Bitmaps store
    /// GLOBAL obs row indices. Public so callers can construct/mutate state
    /// directly; write order on disk is BTreeMap key order (byte-stable).
    pub deletions: BTreeMap<u8, RoaringBitmap>,
    /// Transient holding pen for a freshly-read v1 file's per-shard bitmaps,
    /// keyed by the flattened `shards_sorted()` positional index. `None` for
    /// v2 files and after [`Self::fold_v1_to_global`] has run. Not serialized.
    v1_shards: Option<BTreeMap<u32, RoaringBitmap>>,
}

impl VersionedSection for DeletionVectors {
    const SECTION_NAME: &'static str = "deletion vectors";
    const CURRENT_VERSION: u16 = DV_VERSION as u16;

    /// Accept both the current v2 layout and the legacy v1 layout (folded to
    /// global on read); reject anything else. Overrides the default
    /// exact-match gate so v1 files remain readable across the `1 -> 2` bump.
    fn check_version(found: u16) -> Result<()> {
        if found != DV_V1 as u16 && found != Self::CURRENT_VERSION {
            return Err(ScxError::UnsupportedSectionVersion {
                section: Self::SECTION_NAME,
                found,
                expected: Self::CURRENT_VERSION,
            });
        }
        Ok(())
    }
}

impl DeletionVectors {
    /// Create an empty (v2) DeletionVectors.
    pub fn new() -> Self {
        Self {
            dv_version: DV_VERSION,
            deletions: BTreeMap::new(),
            v1_shards: None,
        }
    }

    /// Serialize to writer in the v2 layout. Entry iteration order is sorted
    /// by `modality_id` (BTreeMap invariant), so the byte stream is stable.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        debug_assert!(
            self.v1_shards.is_none(),
            "write_to called on an unfolded v1 DeletionVectors"
        );
        w.write_u8(DV_VERSION)?;
        w.write_u32::<LittleEndian>(self.deletions.len() as u32)?;
        for (&modality_id, bitmap) in &self.deletions {
            w.write_u8(modality_id)?;
            w.write_all(&[0u8; 3])?; // reserved
            let mut bitmap_bytes = Vec::new();
            bitmap
                .serialize_into(&mut bitmap_bytes)
                .map_err(std::io::Error::other)?;
            w.write_u32::<LittleEndian>(bitmap_bytes.len() as u32)?;
            w.write_all(&bitmap_bytes)?;
        }
        Ok(())
    }

    /// Deserialize from reader. Catalog-free: a v1 file is parsed into the
    /// private `v1_shards` transient (empty `deletions`) and must be folded via
    /// [`Self::fold_v1_to_global`] before its deletions are observable — done
    /// at the reader boundary. A v2 file populates `deletions` directly.
    ///
    /// `section_len` is the total byte length of the enclosing section (from
    /// the catalog entry); every on-disk length field is validated against it
    /// before allocating.
    pub fn read_from<R: Read>(r: &mut R, section_len: usize) -> Result<Self> {
        let dv_version = r.read_u8()?;
        Self::check_version(dv_version as u16)?;

        if dv_version == DV_V1 {
            return Self::read_v1(r, section_len);
        }

        let n_entries = r.read_u32::<LittleEndian>()? as usize;
        // Minimum bytes per entry: 1 (modality_id) + 3 (reserved) + 4 (bitmap_len).
        crate::error::validate_allocation(n_entries.saturating_mul(8), section_len)?;

        let mut deletions = BTreeMap::new();
        for _ in 0..n_entries {
            let modality_id = r.read_u8()?;
            let mut reserved = [0u8; 3];
            r.read_exact(&mut reserved)?;
            let bitmap_len = r.read_u32::<LittleEndian>()? as usize;
            crate::error::validate_allocation(bitmap_len, section_len)?;
            let mut bitmap_bytes = vec![0u8; bitmap_len];
            r.read_exact(&mut bitmap_bytes)?;
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_bytes[..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            // Duplicate modality_id → last-writer-wins (well-formed files never
            // emit duplicates; BTreeMap holds one entry per key).
            deletions.insert(modality_id, bitmap);
        }
        Ok(Self {
            dv_version,
            deletions,
            v1_shards: None,
        })
    }

    /// Parse the legacy v1 per-shard layout into the `v1_shards` transient.
    fn read_v1<R: Read>(r: &mut R, section_len: usize) -> Result<Self> {
        let n_shards = r.read_u32::<LittleEndian>()? as usize;
        // Minimum bytes per shard entry: 4 (shard_id) + 4 (bitmap_len) = 8.
        crate::error::validate_allocation(n_shards.saturating_mul(8), section_len)?;

        let mut shards = BTreeMap::new();
        for _ in 0..n_shards {
            let shard_id = r.read_u32::<LittleEndian>()?;
            let bitmap_len = r.read_u32::<LittleEndian>()? as usize;
            crate::error::validate_allocation(bitmap_len, section_len)?;
            let mut bitmap_bytes = vec![0u8; bitmap_len];
            r.read_exact(&mut bitmap_bytes)?;
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_bytes[..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            shards.insert(shard_id, bitmap);
        }
        Ok(Self {
            dv_version: DV_V1,
            deletions: BTreeMap::new(),
            v1_shards: Some(shards),
        })
    }

    /// Convert a freshly-read v1 file's per-shard (shard-local) bitmaps to the
    /// global v2 representation under `deletions[DV_GLOBAL]`, using the CSR
    /// shard order and `stats.row_start` from `catalog`. No-op for v2 files
    /// (`v1_shards` is `None`). Called at the reader boundary so downstream
    /// consumers only ever observe the v2 shape.
    pub fn fold_v1_to_global(&mut self, catalog: &crate::FullCatalog) {
        let Some(shards) = self.v1_shards.take() else {
            return;
        };
        let sorted = catalog.shards_sorted();
        let global = self.deletions.entry(DV_GLOBAL).or_default();
        for (shard_idx, entry) in sorted.iter().enumerate() {
            if let Some(stats) = entry.stats.as_ref() {
                if let Some(bitmap) = shards.get(&(shard_idx as u32)) {
                    for local_row in bitmap.iter() {
                        let global_row = stats.row_start + local_row as u64;
                        if global_row <= u32::MAX as u64 {
                            global.insert(global_row as u32);
                        }
                    }
                }
            }
        }
        if global.is_empty() {
            self.deletions.remove(&DV_GLOBAL);
        }
        self.dv_version = DV_VERSION;
    }

    /// Whether row `row` is deleted for every modality (the global bitmap).
    pub fn is_deleted_global(&self, row: u32) -> bool {
        self.deletions
            .get(&DV_GLOBAL)
            .is_some_and(|bm| bm.contains(row))
    }

    /// Whether row `row` is deleted for `modality_id` — global deletions plus
    /// any deletions scoped to that modality.
    pub fn is_deleted_in(&self, modality_id: u8, row: u32) -> bool {
        if self.is_deleted_global(row) {
            return true;
        }
        modality_id != DV_GLOBAL
            && self
                .deletions
                .get(&modality_id)
                .is_some_and(|bm| bm.contains(row))
    }

    /// The set of globally-deleted obs rows (`deletions[DV_GLOBAL]`), if any.
    pub fn global_deleted(&self) -> Option<&RoaringBitmap> {
        self.deletions.get(&DV_GLOBAL)
    }

    /// Count of globally-deleted obs rows in the half-open range
    /// `[start, end)` — used by shard pruning to detect a fully-deleted shard.
    pub fn deleted_in_range(&self, start: u64, end: u64) -> u64 {
        self.global_deleted().map_or(0, |bm| {
            let lo = start.min(u32::MAX as u64) as u32;
            let hi = end.min(u32::MAX as u64 + 1) as u32;
            bm.range(lo..hi).count() as u64
        })
    }

    /// Insert global obs rows into the whole-cell (`DV_GLOBAL`) bitmap.
    pub fn insert_global<I: IntoIterator<Item = u32>>(&mut self, rows: I) {
        let global = self.deletions.entry(DV_GLOBAL).or_default();
        for r in rows {
            global.insert(r);
        }
    }

    /// Total number of deleted cells — the global bitmap only. Drives the
    /// logical `n_obs` (physical − deleted); modality-scoped (`id >= 1`)
    /// deletions do not change the shared obs count.
    pub fn total_deleted(&self) -> u64 {
        self.deletions.get(&DV_GLOBAL).map_or(0, |bm| bm.len())
    }

    /// Build a per-row keep mask of length `n_obs` (`true` = retained) for
    /// `modality_id`. Global deletions always apply; a modality (`>= 1`) read
    /// additionally applies that modality's scoped deletions. v2 bitmaps are
    /// already global, so no catalog is needed (v1 files must be folded first).
    ///
    /// This is the single source of truth for deletion-vector row filtering
    /// shared by the reader, the h5ad/h5mu streaming export, `scx compact`, and
    /// the `pyscx` obs filter.
    pub fn build_keep_mask(&self, n_obs: usize, modality_id: u8) -> Vec<bool> {
        let mut keep = vec![true; n_obs];
        let mut apply = |bm: &RoaringBitmap| {
            for row in bm.iter() {
                if (row as usize) < n_obs {
                    keep[row as usize] = false;
                }
            }
        };
        if let Some(bm) = self.deletions.get(&DV_GLOBAL) {
            apply(bm);
        }
        if modality_id != DV_GLOBAL {
            if let Some(bm) = self.deletions.get(&modality_id) {
                apply(bm);
            }
        }
        keep
    }

    /// Convenience for the common whole-cell keep mask (`modality_id == 0`).
    pub fn build_keep_mask_global(&self, n_obs: usize) -> Vec<bool> {
        self.build_keep_mask(n_obs, DV_GLOBAL)
    }

    /// Merge another DeletionVectors into this one, OR-ing bitmaps per
    /// `modality_id` key.
    pub fn merge(&mut self, other: &DeletionVectors) {
        debug_assert!(
            self.v1_shards.is_none() && other.v1_shards.is_none(),
            "merge called on an unfolded v1 DeletionVectors"
        );
        for (&modality_id, other_bm) in &other.deletions {
            self.deletions
                .entry(modality_id)
                .and_modify(|bm| *bm |= other_bm)
                .or_insert_with(|| other_bm.clone());
        }
    }
}

impl Default for DeletionVectors {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FullCatalog, FullCatalogEntry, SectionType, ShardStats};

    /// Build a minimal CSR-shard catalog entry spanning `[row_start, row_end)`.
    fn csr_shard(name: &str, row_start: u64, row_end: u64) -> FullCatalogEntry {
        FullCatalogEntry {
            name: name.to_string(),
            offset: 0,
            length: 0,
            section_type: SectionType::CsrShard,
            checksum: [0u8; 32],
            modality_id: 0,
            stats: Some(ShardStats {
                row_start,
                row_end,
                col_start: 0,
                col_end: 0,
                nnz: 0,
                value_min: 0,
                value_max: 0,
                value_sum: 0,
                n_indexed_columns: 0,
                column_stats: Vec::new(),
            }),
        }
    }

    fn catalog_with(entries: Vec<FullCatalogEntry>, n_obs: u64) -> FullCatalog {
        FullCatalog {
            catalog_version: crate::CURRENT_CATALOG_VERSION,
            manifest_sequence: 1,
            prev_catalog_offset: 0,
            n_obs,
            entries,
            data_generation: 0,
            csc_build_generation: 0,
        }
    }

    /// Hand-serialize a v1 (legacy) deletion-vectors section body from a set
    /// of `(shard_id, [local_rows])`, for the v1→global fold tests.
    fn v1_bytes(shards: &[(u32, &[u32])]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.write_u8(DV_V1).unwrap();
        buf.write_u32::<LittleEndian>(shards.len() as u32).unwrap();
        for (shard_id, rows) in shards {
            buf.write_u32::<LittleEndian>(*shard_id).unwrap();
            let mut bm = RoaringBitmap::new();
            for &r in *rows {
                bm.insert(r);
            }
            let mut bm_bytes = Vec::new();
            bm.serialize_into(&mut bm_bytes).unwrap();
            buf.write_u32::<LittleEndian>(bm_bytes.len() as u32)
                .unwrap();
            buf.write_all(&bm_bytes).unwrap();
        }
        buf
    }

    #[test]
    fn build_keep_mask_global_and_scoped() {
        let mut dv = DeletionVectors::new();
        // Global deletes rows 1 and an out-of-range row 99 (ignored by bound).
        dv.deletions
            .insert(DV_GLOBAL, RoaringBitmap::from_iter([1u32, 99]));
        // Modality-2 scoped delete of row 3.
        dv.deletions.insert(2, RoaringBitmap::from_iter([3u32]));

        // Global / default read applies only the global bitmap.
        assert_eq!(
            dv.build_keep_mask_global(5),
            vec![true, false, true, true, true]
        );
        // Modality-1 read: no scoped deletions → same as global.
        assert_eq!(
            dv.build_keep_mask(5, 1),
            vec![true, false, true, true, true]
        );
        // Modality-2 read: global ∪ scoped.
        assert_eq!(
            dv.build_keep_mask(5, 2),
            vec![true, false, true, false, true]
        );
    }

    #[test]
    fn v1_read_then_fold_to_global() {
        // v1 layout: shard 0 = rows [0,3), shard 1 = rows [3,5). n_obs = 5.
        // shard 0 deletes local row 1 (global 1) + out-of-range 99;
        // shard 1 deletes local row 0 (global 3); shard 9 does not exist.
        let bytes = v1_bytes(&[(0, &[1, 99]), (1, &[0]), (9, &[0])]);
        let mut dv =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&bytes), bytes.len()).unwrap();
        // Before folding, a v1 file has no observable global deletions.
        assert_eq!(dv.dv_version, DV_V1);
        assert_eq!(dv.total_deleted(), 0);

        let catalog = catalog_with(
            vec![csr_shard("X_shard_0", 0, 3), csr_shard("X_shard_1", 3, 5)],
            5,
        );
        dv.fold_v1_to_global(&catalog);

        assert_eq!(dv.dv_version, DV_VERSION);
        // global 99 is beyond the shards' rows; still stored as a global row,
        // but the keep-mask ignores it (>= n_obs). Rows 1 and 3 deleted.
        assert_eq!(
            dv.build_keep_mask_global(5),
            vec![true, false, true, false, true]
        );
        // total_deleted counts the raw global bitmap (rows 1, 3, 99).
        assert_eq!(dv.total_deleted(), 3);
        // Folding is idempotent (no v1_shards left).
        dv.fold_v1_to_global(&catalog);
        assert_eq!(dv.total_deleted(), 3);
    }

    /// An unknown `dv_version` on the wire must be rejected.
    #[test]
    fn read_rejects_unknown_version() {
        let mut buf = Vec::new();
        buf.write_u8(DV_VERSION + 1).unwrap();
        buf.write_u32::<LittleEndian>(0).unwrap();
        let err =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), buf.len()).unwrap_err();
        assert!(
            matches!(err, crate::ScxError::UnsupportedSectionVersion { .. }),
            "expected UnsupportedSectionVersion, got {err:?}"
        );
    }

    #[test]
    fn check_version_accepts_v1_and_v2_rejects_future() {
        assert!(DeletionVectors::check_version(DV_V1 as u16).is_ok());
        assert!(DeletionVectors::check_version(DV_VERSION as u16).is_ok());
        assert!(matches!(
            DeletionVectors::check_version(DV_VERSION as u16 + 1),
            Err(crate::ScxError::UnsupportedSectionVersion { .. })
        ));
    }

    #[test]
    fn round_trip_v2_modality_keyed() {
        let mut dv = DeletionVectors::new();
        dv.deletions
            .insert(DV_GLOBAL, RoaringBitmap::from_iter([0u32, 5, 100]));
        dv.deletions.insert(3, RoaringBitmap::from_iter([42u32]));

        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();

        let decoded =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.dv_version, DV_VERSION);
        assert_eq!(decoded.deletions.len(), 2);
        assert!(decoded.is_deleted_global(0));
        assert!(decoded.is_deleted_global(5));
        assert!(decoded.is_deleted_global(100));
        assert!(!decoded.is_deleted_global(1));
        // modality 3 sees global ∪ its scoped rows.
        assert!(decoded.is_deleted_in(3, 42));
        assert!(decoded.is_deleted_in(3, 0));
        assert!(!decoded.is_deleted_in(1, 42));
        // total_deleted counts the global bitmap only (0, 5, 100).
        assert_eq!(decoded.total_deleted(), 3);
    }

    #[test]
    fn insert_global_and_deleted_in_range() {
        let mut dv = DeletionVectors::new();
        dv.insert_global([2u32, 4, 7, 9]);
        assert_eq!(dv.total_deleted(), 4);
        assert!(dv.is_deleted_global(4));
        assert!(!dv.is_deleted_global(5));
        // [4, 8): rows 4 and 7.
        assert_eq!(dv.deleted_in_range(4, 8), 2);
        // [0, 3): row 2.
        assert_eq!(dv.deleted_in_range(0, 3), 1);
        assert_eq!(dv.deleted_in_range(10, 20), 0);
    }

    #[test]
    fn merge_vectors() {
        let mut dv1 = DeletionVectors::new();
        dv1.deletions
            .insert(DV_GLOBAL, RoaringBitmap::from_iter([1u32, 3]));

        let mut dv2 = DeletionVectors::new();
        dv2.deletions
            .insert(DV_GLOBAL, RoaringBitmap::from_iter([3u32, 7]));
        dv2.deletions.insert(2, RoaringBitmap::from_iter([0u32]));

        dv1.merge(&dv2);
        assert_eq!(dv1.deletions.len(), 2);
        assert!(dv1.is_deleted_global(1));
        assert!(dv1.is_deleted_global(3));
        assert!(dv1.is_deleted_global(7));
        assert!(dv1.is_deleted_in(2, 0));
        assert_eq!(dv1.total_deleted(), 3); // global: 1, 3, 7
    }

    #[test]
    fn empty_round_trip() {
        let dv = DeletionVectors::new();
        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();
        let decoded =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.deletions.len(), 0);
        assert_eq!(decoded.total_deleted(), 0);
    }

    #[test]
    fn sorted_serialization_byte_stable() {
        // Insertion order != key order; serialization must be key-sorted.
        let mut dv = DeletionVectors::new();
        let b = RoaringBitmap::from_iter([1u32]);
        dv.deletions.insert(5, b.clone());
        dv.deletions.insert(2, b.clone());
        dv.deletions.insert(8, b);

        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();
        // Each v2 entry: u8 modality_id + [u8;3] reserved + u32 bitmap_len +
        // bitmap. Header is u8 version + u32 n_entries = 5 bytes.
        let mut offset = 5usize;
        let mut seen = Vec::new();
        for _ in 0..3 {
            seen.push(buf[offset]);
            let bm_len =
                u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap()) as usize;
            offset += 8 + bm_len;
        }
        assert_eq!(seen, vec![2, 5, 8]);
    }

    // -----------------------------------------------------------------------
    // Defensive allocation cap tests
    // -----------------------------------------------------------------------

    #[test]
    fn dv_rejects_oversized_n_entries() {
        let mut buf = Vec::new();
        buf.write_u8(DV_VERSION).unwrap();
        buf.write_u32::<LittleEndian>(u32::MAX).unwrap(); // n_entries = absurd

        let section_len = buf.len();
        let result = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), section_len);
        assert!(result.is_err(), "should reject oversized n_entries");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("allocation too large"),
            "error should mention allocation: {err_msg}"
        );
    }

    #[test]
    fn dv_rejects_oversized_bitmap_len() {
        let mut buf = Vec::new();
        buf.write_u8(DV_VERSION).unwrap();
        buf.write_u32::<LittleEndian>(1).unwrap(); // n_entries = 1
        buf.write_u8(0).unwrap(); // modality_id
        buf.write_all(&[0u8; 3]).unwrap(); // reserved
        buf.write_u32::<LittleEndian>(u32::MAX).unwrap(); // bitmap_len = absurd

        let section_len = buf.len();
        let result = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), section_len);
        assert!(result.is_err(), "should reject oversized bitmap_len");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("allocation too large"),
            "error should mention allocation: {err_msg}"
        );
    }

    #[test]
    fn v1_read_rejects_oversized_n_shards() {
        let mut buf = Vec::new();
        buf.write_u8(DV_V1).unwrap();
        buf.write_u32::<LittleEndian>(u32::MAX).unwrap(); // n_shards = absurd
        let section_len = buf.len();
        let result = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), section_len);
        assert!(result.is_err(), "should reject oversized v1 n_shards");
        assert!(format!("{}", result.unwrap_err()).contains("allocation too large"));
    }
}
