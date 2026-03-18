// Deletion vectors (SPEC §3.6.3)
//
// Binary layout: u8 version, u32 n_shards,
// then per-shard: u32 shard_id, u32 bitmap_len, [u8; bitmap_len] roaring bitmap.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use roaring::RoaringBitmap;
use std::io::{Read, Write};

use crate::error::Result;

/// Per-shard deletion bitmap.
#[derive(Debug, Clone)]
pub struct ShardDeletion {
    pub shard_id: u32,
    pub bitmap: RoaringBitmap,
}

/// Deletion vectors section: tracks logically deleted rows per shard.
#[derive(Debug, Clone)]
pub struct DeletionVectors {
    pub dv_version: u8,
    pub shards: Vec<ShardDeletion>,
}

impl DeletionVectors {
    /// Create an empty DeletionVectors.
    pub fn new() -> Self {
        Self {
            dv_version: 1,
            shards: Vec::new(),
        }
    }

    /// Serialize to writer.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_u8(self.dv_version)?;
        w.write_u32::<LittleEndian>(self.shards.len() as u32)?;
        for sd in &self.shards {
            w.write_u32::<LittleEndian>(sd.shard_id)?;
            let mut bitmap_bytes = Vec::new();
            sd.bitmap
                .serialize_into(&mut bitmap_bytes)
                .map_err(std::io::Error::other)?;
            w.write_u32::<LittleEndian>(bitmap_bytes.len() as u32)?;
            w.write_all(&bitmap_bytes)?;
        }
        Ok(())
    }

    /// Deserialize from reader.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self> {
        let dv_version = r.read_u8()?;
        let n_shards = r.read_u32::<LittleEndian>()? as usize;
        let mut shards = Vec::with_capacity(n_shards);
        for _ in 0..n_shards {
            let shard_id = r.read_u32::<LittleEndian>()?;
            let bitmap_len = r.read_u32::<LittleEndian>()? as usize;
            let mut bitmap_bytes = vec![0u8; bitmap_len];
            r.read_exact(&mut bitmap_bytes)?;
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_bytes[..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            shards.push(ShardDeletion { shard_id, bitmap });
        }
        Ok(Self { dv_version, shards })
    }

    /// Check if a specific row in a shard is deleted.
    pub fn is_deleted(&self, shard_id: u32, local_row: u32) -> bool {
        self.shards
            .iter()
            .find(|sd| sd.shard_id == shard_id)
            .is_some_and(|sd| sd.bitmap.contains(local_row))
    }

    /// Total number of deleted rows across all shards.
    pub fn total_deleted(&self) -> u64 {
        self.shards.iter().map(|sd| sd.bitmap.len()).sum()
    }

    /// Merge another DeletionVectors into this one.
    /// For matching shard_ids, bitmaps are OR-merged. New shard_ids are appended.
    pub fn merge(&mut self, other: &DeletionVectors) {
        for other_sd in &other.shards {
            if let Some(existing) = self
                .shards
                .iter_mut()
                .find(|sd| sd.shard_id == other_sd.shard_id)
            {
                existing.bitmap |= &other_sd.bitmap;
            } else {
                self.shards.push(other_sd.clone());
            }
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

    #[test]
    fn round_trip() {
        let mut dv = DeletionVectors::new();
        let mut bm0 = RoaringBitmap::new();
        bm0.insert(0);
        bm0.insert(5);
        bm0.insert(100);
        dv.shards.push(ShardDeletion {
            shard_id: 0,
            bitmap: bm0,
        });

        let mut bm2 = RoaringBitmap::new();
        bm2.insert(42);
        dv.shards.push(ShardDeletion {
            shard_id: 2,
            bitmap: bm2,
        });

        let mut buf = Vec::new();
        dv.write_to(&mut buf).unwrap();

        let decoded = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf)).unwrap();
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
        dv1.shards.push(ShardDeletion {
            shard_id: 0,
            bitmap: bm,
        });

        let mut dv2 = DeletionVectors::new();
        let mut bm2 = RoaringBitmap::new();
        bm2.insert(3);
        bm2.insert(7);
        dv2.shards.push(ShardDeletion {
            shard_id: 0,
            bitmap: bm2,
        });
        let mut bm3 = RoaringBitmap::new();
        bm3.insert(0);
        dv2.shards.push(ShardDeletion {
            shard_id: 1,
            bitmap: bm3,
        });

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
        let decoded = DeletionVectors::read_from(&mut std::io::Cursor::new(&buf)).unwrap();
        assert_eq!(decoded.shards.len(), 0);
        assert_eq!(decoded.total_deleted(), 0);
    }
}
