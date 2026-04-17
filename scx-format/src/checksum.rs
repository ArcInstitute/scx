// BLAKE3 helpers (docs/format.md (Checksums))

/// Compute a full 32-byte BLAKE3 hash.
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

/// Compute BLAKE3 truncated to 64 bits (first 8 bytes).
/// Used for per-shard checksums.
pub fn blake3_truncated_64(data: &[u8]) -> [u8; 8] {
    let full = blake3::hash(data);
    let mut out = [0u8; 8];
    out.copy_from_slice(&full.as_bytes()[..8]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_deterministic() {
        let data = b"hello scx shard payload";
        let h1 = blake3_hash(data);
        let h2 = blake3_hash(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn truncated_matches_prefix() {
        let data = b"test payload";
        let full = blake3_hash(data);
        let trunc = blake3_truncated_64(data);
        assert_eq!(&full[..8], &trunc);
    }

    #[test]
    fn checksum_detects_corruption() {
        let data = b"shard payload bytes here";
        let original_checksum = blake3_truncated_64(data);

        let mut corrupted = data.to_vec();
        corrupted[5] ^= 0xFF; // flip one byte
        let corrupted_checksum = blake3_truncated_64(&corrupted);

        assert_ne!(original_checksum, corrupted_checksum);
    }

    #[test]
    fn empty_input() {
        // Should not panic on empty data
        let h = blake3_hash(b"");
        let t = blake3_truncated_64(b"");
        assert_eq!(&h[..8], &t);
    }
}
