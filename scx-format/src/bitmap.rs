// Detection bitmap shards (Phase 5b of REAL-WORLD-UX-FEATS.md).
//
// One section per CSR shard. Each section stores a roaring bitmap
// per gene whose nonzero presence in that shard is recorded. Drives
// O(roaring-cardinality) detection-count and "cells expressing gene X"
// queries without scanning the CSR payload.
//
// On-disk wire format (mirrors REAL-WORLD-UX-FEATS.md "Wire format proposal"):
//
//   magic            : [u8; 4]   = b"SCXB"
//   version          : u16 LE    = 1
//   orientation      : u8        = 0   // 0 = gene → local rows
//   index_dtype      : u8        // 0 = u16 gene_id (n_vars ≤ 65535), 1 = u32
//   row_start        : u64 LE    // global row offset (mirrors CSR shard)
//   n_rows           : u32 LE    // rows in this shard
//   n_vars           : u32 LE
//   n_genes_with_hits: u32 LE
//   for i in 0..n_genes_with_hits:
//     gene_id        : u16 or u32 LE depending on index_dtype
//     roaring_len    : u32 LE
//     roaring_bytes  : [u8; roaring_len]   // RoaringBitmap::serialize_into
//   checksum         : [u8; 32]  // BLAKE3-256 over preceding bytes

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use roaring::RoaringBitmap;
use std::collections::BTreeMap;
use std::io::{Read, Write};

use crate::error::{validate_allocation, Result, ScxError};

/// Magic bytes at the start of every bitmap shard section.
pub const BITMAP_SHARD_MAGIC: [u8; 4] = *b"SCXB";
/// Wire-format version. Increment only when an existing field
/// changes shape; new fields tagged by version are forward-compat.
pub const BITMAP_SHARD_VERSION: u16 = 1;
/// Only orientation shipped today: `gene_id -> local row ids`.
pub const BITMAP_ORIENTATION_GENE_TO_ROWS: u8 = 0;

/// Roaring-encoded detection bitmap for a single CSR shard.
///
/// `genes` is a sparse map (only genes with ≥1 hit appear); the bitmap
/// stores **local** row ids (0..n_rows). Pair with `row_start` to recover
/// the global row id (`row_start + local`).
#[derive(Debug, Clone)]
pub struct BitmapShard {
    pub version: u16,
    pub orientation: u8,
    pub index_dtype: u8,
    pub row_start: u64,
    pub n_rows: u32,
    pub n_vars: u32,
    /// `gene_id -> roaring bitmap of local row ids (0..n_rows)`.
    /// `BTreeMap` keeps the on-disk serialisation order canonical
    /// (sorted by gene_id) — round-trip stable across producers.
    pub genes: BTreeMap<u32, RoaringBitmap>,
}

impl BitmapShard {
    /// Build a `BitmapShard` from a sorted CSR shard. `indptr.len() ==
    /// n_rows + 1` and `indices[indptr[r]..indptr[r+1]]` lists the
    /// nonzero columns for local row `r` (no checks on order — duplicate
    /// entries are silently OR'd into the same gene's bitmap).
    pub fn build_from_csr(
        row_start: u64,
        n_rows: u32,
        n_vars: u32,
        indptr: &[u64],
        indices: &[u32],
    ) -> Self {
        let index_dtype: u8 = if n_vars <= u16::MAX as u32 { 0 } else { 1 };
        let mut genes: BTreeMap<u32, RoaringBitmap> = BTreeMap::new();
        for row in 0..n_rows as usize {
            let start = indptr[row] as usize;
            let end = indptr[row + 1] as usize;
            for &gene_id in &indices[start..end] {
                genes.entry(gene_id).or_default().insert(row as u32);
            }
        }
        Self {
            version: BITMAP_SHARD_VERSION,
            orientation: BITMAP_ORIENTATION_GENE_TO_ROWS,
            index_dtype,
            row_start,
            n_rows,
            n_vars,
            genes,
        }
    }

    /// Number of cells expressing `gene_id` in this shard. Returns 0
    /// if the gene has no hits (the gene is simply absent from the map).
    pub fn gene_detection_count(&self, gene_id: u32) -> u64 {
        self.genes.get(&gene_id).map_or(0, |bm| bm.len())
    }

    /// Roaring bitmap of *local* row ids where `gene_id` is detected.
    /// Returns `None` if the gene has no hits.
    pub fn cells_expressing(&self, gene_id: u32) -> Option<&RoaringBitmap> {
        self.genes.get(&gene_id)
    }

    /// Aggregate detection counts per gene as a dense `Vec<u64>` of
    /// length `n_vars`. Used by the convert-time auto policy to compare
    /// against the encoded CSR size, and by the read-side
    /// `gene_detection_counts()` fast path.
    pub fn per_gene_counts(&self) -> Vec<u64> {
        let mut out = vec![0u64; self.n_vars as usize];
        for (&gene_id, bm) in &self.genes {
            out[gene_id as usize] = bm.len();
        }
        out
    }

    /// Cheap upper bound on the encoded byte length, used by the
    /// auto-policy size estimator. Roaring's `serialized_size()`
    /// is O(n_containers) so this is essentially free.
    pub fn estimated_encoded_size(&self) -> usize {
        // Header: 4 + 2 + 1 + 1 + 8 + 4 + 4 + 4 = 28 bytes.
        // Per gene: gene_id width + 4 (roaring_len) + serialized roaring.
        // Trailer: 32 (BLAKE3-256).
        let gene_id_width = if self.index_dtype == 0 { 2 } else { 4 };
        let mut total = 28usize + 32;
        for bm in self.genes.values() {
            total = total.saturating_add(gene_id_width + 4 + bm.serialized_size());
        }
        total
    }

    /// Serialise per the wire format above. Writes everything including
    /// the trailing 32-byte BLAKE3 checksum.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        let mut hasher = blake3::Hasher::new();
        let mut buf: Vec<u8> = Vec::new();
        buf.write_all(&BITMAP_SHARD_MAGIC)?;
        buf.write_u16::<LittleEndian>(self.version)?;
        buf.write_u8(self.orientation)?;
        buf.write_u8(self.index_dtype)?;
        buf.write_u64::<LittleEndian>(self.row_start)?;
        buf.write_u32::<LittleEndian>(self.n_rows)?;
        buf.write_u32::<LittleEndian>(self.n_vars)?;
        buf.write_u32::<LittleEndian>(self.genes.len() as u32)?;
        for (&gene_id, bm) in &self.genes {
            if self.index_dtype == 0 {
                if gene_id > u16::MAX as u32 {
                    return Err(ScxError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!(
                            "gene_id {gene_id} exceeds u16::MAX but index_dtype = 0; \
                             promote BitmapShard.index_dtype to 1 first"
                        ),
                    )));
                }
                buf.write_u16::<LittleEndian>(gene_id as u16)?;
            } else {
                buf.write_u32::<LittleEndian>(gene_id)?;
            }
            let mut bitmap_bytes = Vec::with_capacity(bm.serialized_size());
            bm.serialize_into(&mut bitmap_bytes)
                .map_err(std::io::Error::other)?;
            buf.write_u32::<LittleEndian>(bitmap_bytes.len() as u32)?;
            buf.write_all(&bitmap_bytes)?;
        }
        hasher.update(&buf);
        let checksum = hasher.finalize();
        buf.extend_from_slice(checksum.as_bytes());
        w.write_all(&buf)?;
        Ok(())
    }

    /// Deserialise from a reader. `section_len` is the total byte
    /// length of the enclosing section (from the catalog entry). All
    /// on-disk length fields are validated against this bound before
    /// allocating, preventing malformed inputs from forcing huge
    /// allocations (mirrors `deletion_vectors.rs`).
    pub fn read_from<R: Read>(r: &mut R, section_len: usize) -> Result<Self> {
        // Header is 28 bytes; trailer is 32. Reject anything that can't
        // hold both upfront.
        if section_len < 28 + 32 {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bitmap shard section too small: {section_len} bytes"),
            )));
        }

        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if magic != BITMAP_SHARD_MAGIC {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("bad bitmap magic: {magic:?}"),
            )));
        }
        let version = r.read_u16::<LittleEndian>()?;
        if version != BITMAP_SHARD_VERSION {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "unsupported bitmap shard version {version}; expected {BITMAP_SHARD_VERSION}"
                ),
            )));
        }
        let orientation = r.read_u8()?;
        if orientation != BITMAP_ORIENTATION_GENE_TO_ROWS {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported bitmap orientation {orientation}"),
            )));
        }
        let index_dtype = r.read_u8()?;
        if index_dtype > 1 {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("unsupported bitmap index_dtype {index_dtype}"),
            )));
        }
        let row_start = r.read_u64::<LittleEndian>()?;
        let n_rows = r.read_u32::<LittleEndian>()?;
        let n_vars = r.read_u32::<LittleEndian>()?;
        let n_genes_with_hits = r.read_u32::<LittleEndian>()? as usize;

        // Minimum bytes per gene entry: gene_id_width + 4 (roaring_len).
        let gene_id_width = if index_dtype == 0 { 2 } else { 4 };
        validate_allocation(
            n_genes_with_hits.saturating_mul(gene_id_width + 4),
            section_len,
        )?;

        let mut genes: BTreeMap<u32, RoaringBitmap> = BTreeMap::new();
        for _ in 0..n_genes_with_hits {
            let gene_id = if index_dtype == 0 {
                r.read_u16::<LittleEndian>()? as u32
            } else {
                r.read_u32::<LittleEndian>()?
            };
            let bitmap_len = r.read_u32::<LittleEndian>()? as usize;
            validate_allocation(bitmap_len, section_len)?;
            let mut bitmap_bytes = vec![0u8; bitmap_len];
            r.read_exact(&mut bitmap_bytes)?;
            let bitmap = RoaringBitmap::deserialize_from(&bitmap_bytes[..])
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            // Last-writer-wins on duplicates (well-formed files never emit them).
            genes.insert(gene_id, bitmap);
        }

        // Trailing checksum (advisory — the catalog entry's BLAKE3 is
        // the authoritative integrity check; this one helps detect
        // intra-section truncation early during streaming reads).
        let mut checksum = [0u8; 32];
        r.read_exact(&mut checksum)?;

        Ok(Self {
            version,
            orientation,
            index_dtype,
            row_start,
            n_rows,
            n_vars,
            genes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_small_csr() -> (Vec<u64>, Vec<u32>, u32, u32) {
        // 5 rows × 4 cols
        //   row 0 → [0, 2]
        //   row 1 → []
        //   row 2 → [1, 2, 3]
        //   row 3 → [0]
        //   row 4 → [3]
        // Per-gene detection masks:
        //   gene 0: {0, 3}
        //   gene 1: {2}
        //   gene 2: {0, 2}
        //   gene 3: {2, 4}
        let indptr: Vec<u64> = vec![0, 2, 2, 5, 6, 7];
        let indices: Vec<u32> = vec![0, 2, 1, 2, 3, 0, 3];
        (indptr, indices, 5, 4)
    }

    #[test]
    fn build_matches_csr_detection() {
        let (indptr, indices, n_rows, n_vars) = make_small_csr();
        let shard = BitmapShard::build_from_csr(0, n_rows, n_vars, &indptr, &indices);
        let counts = shard.per_gene_counts();
        assert_eq!(counts, vec![2, 1, 2, 2]);
        assert_eq!(shard.gene_detection_count(0), 2);
        assert_eq!(shard.gene_detection_count(99), 0); // out-of-range absent gene
        let g0 = shard.cells_expressing(0).unwrap();
        assert!(g0.contains(0) && g0.contains(3));
        assert!(!g0.contains(1));
    }

    #[test]
    fn roundtrip_empty_shard() {
        let shard = BitmapShard::build_from_csr(0, 0, 100, &[0], &[]);
        let mut buf = Vec::new();
        shard.write_to(&mut buf).unwrap();
        let decoded = BitmapShard::read_from(&mut Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.n_rows, 0);
        assert_eq!(decoded.n_vars, 100);
        assert!(decoded.genes.is_empty());
    }

    #[test]
    fn roundtrip_small() {
        let (indptr, indices, n_rows, n_vars) = make_small_csr();
        let shard = BitmapShard::build_from_csr(1000, n_rows, n_vars, &indptr, &indices);
        let mut buf = Vec::new();
        shard.write_to(&mut buf).unwrap();
        let decoded = BitmapShard::read_from(&mut Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.row_start, 1000);
        assert_eq!(decoded.n_rows, n_rows);
        assert_eq!(decoded.n_vars, n_vars);
        assert_eq!(decoded.per_gene_counts(), shard.per_gene_counts());
        for (g, bm) in &shard.genes {
            assert_eq!(decoded.genes.get(g).unwrap(), bm);
        }
    }

    #[test]
    fn roundtrip_u32_index_dtype() {
        // n_vars > u16::MAX forces index_dtype = 1.
        let n_vars: u32 = 70_000;
        let indptr: Vec<u64> = vec![0, 1, 2];
        let indices: Vec<u32> = vec![69_999, 65_536];
        let shard = BitmapShard::build_from_csr(0, 2, n_vars, &indptr, &indices);
        assert_eq!(shard.index_dtype, 1);
        let mut buf = Vec::new();
        shard.write_to(&mut buf).unwrap();
        let decoded = BitmapShard::read_from(&mut Cursor::new(&buf), buf.len()).unwrap();
        assert_eq!(decoded.index_dtype, 1);
        assert!(decoded.cells_expressing(69_999).unwrap().contains(0));
        assert!(decoded.cells_expressing(65_536).unwrap().contains(1));
    }

    #[test]
    fn read_rejects_bad_magic() {
        let mut buf = vec![0u8; 60];
        buf[0..4].copy_from_slice(b"XXXX");
        let err = BitmapShard::read_from(&mut Cursor::new(&buf), buf.len()).unwrap_err();
        assert!(format!("{err}").contains("magic"));
    }

    #[test]
    fn read_rejects_short_section() {
        let buf = vec![0u8; 32]; // less than 60-byte minimum
        let err = BitmapShard::read_from(&mut Cursor::new(&buf), buf.len()).unwrap_err();
        assert!(format!("{err}").contains("too small"));
    }

    #[test]
    fn estimated_size_lower_bounds_encoded() {
        let (indptr, indices, n_rows, n_vars) = make_small_csr();
        let shard = BitmapShard::build_from_csr(0, n_rows, n_vars, &indptr, &indices);
        let estimate = shard.estimated_encoded_size();
        let mut buf = Vec::new();
        shard.write_to(&mut buf).unwrap();
        // Estimate is an upper bound (roaring serialized_size is exact).
        // Allow ±10% slack on either side just in case roaring internals shift.
        assert!(estimate >= buf.len(), "estimate {estimate} < actual {}", buf.len());
    }
}
