// Deletion vectors (docs/format.md (Deletion Vectors))
//
// Binary layout: u8 version, u32 n_shards,
// then per-shard: u32 shard_id, u32 bitmap_len, [u8; bitmap_len] roaring bitmap.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use roaring::RoaringBitmap;
use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::error::Result;
use crate::versioned::VersionedSection;

/// On-disk version of the deletion-vectors section. Increment only when an
/// existing field changes shape.
pub const DV_VERSION: u8 = 1;

/// Per-shard deletion bitmap — retained as a convenience struct for call
/// sites that construct deletions eagerly (tests, pack/unpack helpers).
/// Internally [`DeletionVectors`] keys by `shard_id` directly for O(1)
/// lookup; see finding M18.
#[derive(Debug, Clone)]
pub struct ShardDeletion {
    pub shard_id: u32,
    pub bitmap: RoaringBitmap,
}

/// Deletion vectors section: tracks logically deleted rows per shard.
///
/// Storage is a `BTreeMap<shard_id, bitmap>` — O(log n) lookup and canonical
/// sorted-by-key iteration on write. Files produced by the previous
/// `Vec<ShardDeletion>`-based implementation whose entries happened to be
/// pushed in ascending shard-id order are byte-identical under the new
/// writer; files whose entries were pushed out of order now serialize in
/// sorted order, producing a different byte sequence but equivalent semantic
/// content. Readers are order-agnostic, so old files of either shape still
/// decode correctly.
#[derive(Debug, Clone)]
pub struct DeletionVectors {
    pub dv_version: u8,
    /// Per-shard deletion bitmaps keyed by shard id.
    ///
    /// Public so callers can mutate freely (e.g. tests asserting specific
    /// state); write order on disk is BTreeMap iteration order (sorted by
    /// shard id).
    pub shards: BTreeMap<u32, RoaringBitmap>,
}

impl VersionedSection for DeletionVectors {
    const SECTION_NAME: &'static str = "deletion vectors";
    const CURRENT_VERSION: u16 = DV_VERSION as u16;
}

impl DeletionVectors {
    /// Create an empty DeletionVectors.
    pub fn new() -> Self {
        Self {
            dv_version: DV_VERSION,
            shards: BTreeMap::new(),
        }
    }

    /// Serialize to writer. Shard iteration order is sorted by `shard_id`
    /// (BTreeMap invariant) — see the struct-level docs on compat with the
    /// legacy `Vec<ShardDeletion>` writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(self.dv_version)?;
        w.write_u32::<LittleEndian>(self.shards.len() as u32)?;
        for (&shard_id, bitmap) in &self.shards {
            w.write_u32::<LittleEndian>(shard_id)?;
            let mut bitmap_bytes = Vec::new();
            bitmap
                .serialize_into(&mut bitmap_bytes)
                .map_err(std::io::Error::other)?;
            w.write_u32::<LittleEndian>(bitmap_bytes.len() as u32)?;
            w.write_all(&bitmap_bytes)?;
        }
        Ok(())
    }

    /// Deserialize from reader.
    ///
    /// `section_len` is the total byte length of the enclosing section
    /// (from the catalog entry). All on-disk length fields are validated
    /// against this bound before allocating, preventing a single
    /// malformed `u32` from requesting a multi-GB allocation.
    pub fn read_from<R: Read>(r: &mut R, section_len: usize) -> Result<Self> {
        let dv_version = r.read_u8()?;
        // Version gate (F5): previously read-and-ignored, so a future v2
        // layout would have been silently misparsed.
        Self::check_version(dv_version as u16)?;
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
            // Behavioral note vs. the legacy `Vec<ShardDeletion>` reader:
            // that reader preserved duplicate shard_ids in the vector, so
            // `is_deleted` OR'd them via linear scan. The BTreeMap can hold
            // only one entry per key, so a duplicate silently overwrites the
            // earlier one — last-writer-wins. A well-formed file produced by
            // any SCX writer never emits duplicates (writers dedupe by
            // shard_id before serialization), so this path only fires on
            // malformed or hand-crafted input.
            shards.insert(shard_id, bitmap);
        }
        Ok(Self { dv_version, shards })
    }

    /// Check if a specific row in a shard is deleted. O(log n_shards) via
    /// the BTreeMap key lookup, vs. the linear scan the Vec-backed version
    /// required (finding M18).
    pub fn is_deleted(&self, shard_id: u32, local_row: u32) -> bool {
        self.shards
            .get(&shard_id)
            .is_some_and(|bm| bm.contains(local_row))
    }

    /// Return the deletion bitmap for `shard_id`, if any.
    pub fn get(&self, shard_id: u32) -> Option<&RoaringBitmap> {
        self.shards.get(&shard_id)
    }

    /// Insert or overwrite the bitmap for a shard.
    pub fn insert(&mut self, shard_id: u32, bitmap: RoaringBitmap) {
        self.shards.insert(shard_id, bitmap);
    }

    /// Convenience constructor from a `Vec<ShardDeletion>` — kept for
    /// back-compat with call sites that already build the Vec form (e.g.
    /// pack-time serialization helpers in `scx-cloud`). Later entries for
    /// the same `shard_id` overwrite earlier ones.
    pub fn from_shard_deletions<I: IntoIterator<Item = ShardDeletion>>(it: I) -> Self {
        let mut dv = Self::new();
        for sd in it {
            dv.shards.insert(sd.shard_id, sd.bitmap);
        }
        dv
    }

    /// Total number of deleted rows across all shards.
    pub fn total_deleted(&self) -> u64 {
        self.shards.values().map(|bm| bm.len()).sum()
    }

    /// Merge another DeletionVectors into this one.
    /// For matching shard_ids, bitmaps are OR-merged. New shard_ids are inserted.
    pub fn merge(&mut self, other: &DeletionVectors) {
        for (&shard_id, other_bm) in &other.shards {
            self.shards
                .entry(shard_id)
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

    /// F5: an unknown `dv_version` on the wire must be rejected (it used to be
    /// read and ignored, so a future v2 layout would have been misparsed).
    #[test]
    fn read_rejects_unknown_version() {
        let mut dv = DeletionVectors::new();
        dv.dv_version = DV_VERSION + 1;
        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();
        let err =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), buf.len()).unwrap_err();
        assert!(
            matches!(err, crate::ScxError::UnsupportedSectionVersion { .. }),
            "expected UnsupportedSectionVersion, got {err:?}"
        );
    }

    #[test]
    fn check_version_rejects_future() {
        assert!(matches!(
            DeletionVectors::check_version(DV_VERSION as u16 + 1),
            Err(crate::ScxError::UnsupportedSectionVersion { .. })
        ));
        assert!(DeletionVectors::check_version(DV_VERSION as u16).is_ok());
    }

    #[test]
    fn round_trip() {
        let mut dv = DeletionVectors::new();
        let mut bm0 = RoaringBitmap::new();
        bm0.insert(0);
        bm0.insert(5);
        bm0.insert(100);
        dv.shards.insert(0, bm0);

        let mut bm2 = RoaringBitmap::new();
        bm2.insert(42);
        dv.shards.insert(2, bm2);

        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();

        let decoded =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.dv_version, 1);
        assert_eq!(decoded.shards.len(), 2);
        assert!(decoded.is_deleted(0, 0));
        assert!(decoded.is_deleted(0, 5));
        assert!(decoded.is_deleted(0, 100));
        assert!(!decoded.is_deleted(0, 1));
        assert!(decoded.is_deleted(2, 42));
        assert!(!decoded.is_deleted(1, 0));
        assert_eq!(decoded.total_deleted(), 4);
    }

    #[test]
    fn merge_vectors() {
        let mut dv1 = DeletionVectors::new();
        let mut bm = RoaringBitmap::new();
        bm.insert(1);
        bm.insert(3);
        dv1.shards.insert(0, bm);

        let mut dv2 = DeletionVectors::new();
        let mut bm2 = RoaringBitmap::new();
        bm2.insert(3);
        bm2.insert(7);
        dv2.shards.insert(0, bm2);
        let mut bm3 = RoaringBitmap::new();
        bm3.insert(0);
        dv2.shards.insert(1, bm3);

        dv1.merge(&dv2);
        assert_eq!(dv1.shards.len(), 2);
        assert!(dv1.is_deleted(0, 1));
        assert!(dv1.is_deleted(0, 3));
        assert!(dv1.is_deleted(0, 7));
        assert!(dv1.is_deleted(1, 0));
        assert_eq!(dv1.total_deleted(), 4);
    }

    #[test]
    fn empty_round_trip() {
        let dv = DeletionVectors::new();
        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();
        let decoded =
            DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.shards.len(), 0);
        assert_eq!(decoded.total_deleted(), 0);
    }

    #[test]
    fn sorted_serialization_byte_stable() {
        // Insertion order != key order; serialization must be key-sorted so
        // old-format files and new-format files are byte-identical.
        let mut dv = DeletionVectors::new();
        let mut b = RoaringBitmap::new();
        b.insert(1);
        dv.shards.insert(5, b.clone());
        dv.shards.insert(2, b.clone());
        dv.shards.insert(8, b);

        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();
        // The shard_id values serialize as u32 LE starting at offset 5 (1
        // byte version + 4 byte count), then interleaved with bitmap blobs.
        // Extract just the shard_id bytes by stepping through entries.
        let mut offset = 5usize;
        let mut seen = Vec::new();
        for _ in 0..3 {
            let sid = u32::from_le_bytes(buf[offset..offset + 4].try_into().unwrap());
            seen.push(sid);
            let bm_len =
                u32::from_le_bytes(buf[offset + 4..offset + 8].try_into().unwrap()) as usize;
            offset += 8 + bm_len;
        }
        assert_eq!(seen, vec![2, 5, 8]);
    }

    // -----------------------------------------------------------------------
    // Defensive allocation cap tests (Patch 9)
    // -----------------------------------------------------------------------

    #[test]
    fn dv_rejects_oversized_n_shards() {
        use byteorder::WriteBytesExt;
        let mut buf = Vec::new();
        buf.write_u8(1).unwrap(); // version
        buf.write_u32::<byteorder::LittleEndian>(u32::MAX).unwrap(); // n_shards = absurd

        let section_len = buf.len();
        let result = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), section_len);
        assert!(result.is_err(), "should reject oversized n_shards");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("allocation too large"),
            "error should mention allocation: {err_msg}"
        );
    }

    #[test]
    fn dv_rejects_oversized_bitmap_len() {
        use byteorder::WriteBytesExt;
        let mut buf = Vec::new();
        buf.write_u8(1).unwrap(); // version
        buf.write_u32::<byteorder::LittleEndian>(1).unwrap(); // n_shards = 1
        buf.write_u32::<byteorder::LittleEndian>(0).unwrap(); // shard_id = 0
        buf.write_u32::<byteorder::LittleEndian>(u32::MAX).unwrap(); // bitmap_len = absurd

        let section_len = buf.len();
        let result = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf), section_len);
        assert!(result.is_err(), "should reject oversized bitmap_len");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("allocation too large"),
            "error should mention allocation: {err_msg}"
        );
    }
}
