//! Checksum and structural validation.
//!
//! Section checksums, the file checksum, and the canonical-CSR shape
//! rules a v3 writer promises.

use super::*;

impl ScxReader {
    /// Recompute the whole-file BLAKE3 checksum and compare it against the
    /// stored `header.file_checksum`.
    ///
    /// Hashes the 256-byte header (with the `file_checksum` field zeroed)
    /// followed by every byte from offset 256 to EOF. This is the same extent
    /// every writer uses — `ScxWriter::finish` (`scx-format-io`),
    /// `scx_ops::checksum::finalize_header_with_checksum` (rollback / append),
    /// and the cloud relayout — so a rolled-back file (whose header points at an
    /// older catalog while newer, now-superseded catalogs/sections remain
    /// trailing past the active catalog) still verifies. Hashing only up to
    /// `full_catalog_offset + full_catalog_length` would wrongly report such a
    /// file as corrupt.
    ///
    /// Covers the 256-byte header and the 4096-byte root catalog, which no
    /// per-section catalog-entry checksum protects — so a flipped byte in
    /// `n_vars`/`full_catalog_offset`/`flags` or the root catalog is caught here
    /// but not by [`validate`](Self::validate)'s per-section walk.
    ///
    /// O(file size); intended for the explicit `validate` path, not the hot
    /// `open` path. Never panics: the only slice (`[HEADER_SIZE..]`) is always
    /// in range because `open` already enforced `mmap.len() >= HEADER_SIZE`, and
    /// a corrupt header simply hashes to a non-matching value (returns `false`).
    pub fn verify_file_checksum(&self) -> Result<bool> {
        let stored = self.header.file_checksum;

        // `open` guarantees this, but guard anyway so the slice below is always
        // sound even if a caller constructs a reader some other way.
        if self.mmap.len() < HEADER_SIZE {
            return Ok(false);
        }

        // Re-serialize the header with the checksum field zeroed, exactly as
        // every writer does before hashing.
        let mut header_for_hash = self.header.clone();
        header_for_hash.file_checksum = 0;
        let mut header_bytes = Vec::with_capacity(HEADER_SIZE);
        header_for_hash.write_to(&mut header_bytes)?;

        let mut hasher = blake3::Hasher::new();
        hasher.update(&header_bytes); // header [0..256], checksum field zeroed
        hasher.update(&self.mmap[HEADER_SIZE..]); // body [256..EOF]
        let computed = crate::checksum::truncate_hash_to_u64(&hasher.finalize());
        Ok(computed == stored)
    }

    /// Validate all section checksums.
    ///
    /// Returns a list of `(section_name, passed)` pairs. The first entry is
    /// always `("file_checksum", _)` — the whole-file integrity check from
    /// [`verify_file_checksum`](Self::verify_file_checksum), covering the header
    /// and root catalog that no per-section checksum protects — followed by one
    /// entry per catalog section. If any essential section (obs, var, CsrShard)
    /// fails, returns `Err(ChecksumMismatch)`.
    ///
    /// A failing `file_checksum` is reported as a flag but is **not** treated as
    /// essential: the whole-file hash cannot localize the corrupted region, so
    /// promoting it to an error would force `Err` even when only a non-essential
    /// (e.g. rebuildable CSC sidecar) section is damaged — contradicting the
    /// granular per-section contract. Callers wanting strict whole-file
    /// integrity should inspect the `file_checksum` flag (the CLI `scx validate`
    /// does, and fails on it).
    pub fn validate(&self) -> Result<Vec<(String, bool)>> {
        let mut results = Vec::new();
        let mut essential_failed = false;

        // Whole-file integrity first: catches corruption in the 256-byte header
        // and root catalog that the per-section walk below cannot see (those
        // bytes are covered by no catalog-entry checksum). Reported as a flag,
        // not an essential error (see the doc note above).
        let file_ok = self.verify_file_checksum()?;
        results.push(("file_checksum".to_string(), file_ok));

        for entry in &self.full_catalog.entries {
            let slice = self.section_bytes(entry)?;
            let computed = blake3_hash(slice);
            let passed = computed == entry.checksum;

            if !passed && entry.section_type.is_essential() {
                // Covers legacy AND sharded obs/var, main X CSR, and raw
                // CSR/var — not just the three legacy types (SCX-008).
                essential_failed = true;
            }

            results.push((entry.name.clone(), passed));
        }

        if essential_failed {
            return Err(ScxError::ChecksumMismatch {
                section: "essential section(s)".to_string(),
            });
        }

        Ok(results)
    }

    /// Validate that a row-major sparse shard obeys the v3 canonical CSR invariant.
    pub fn validate_canonical_csr_entry(&self, entry: &FullCatalogEntry) -> Result<()> {
        if !Self::is_canonical_csr_section(entry.section_type) {
            return Err(ScxError::InvalidCatalog(format!(
                "entry {} is {:?}, not a canonical CSR shard type",
                entry.name, entry.section_type
            )));
        }
        let shard_header = self.read_shard_header(entry)?;
        let (indptr, indices, data) = self.read_shard_from_entry_verified(entry)?;
        Self::validate_decoded_canonical_csr(&entry.name, &shard_header, &indptr, &indices, &data)
    }

    /// Validate all row-major sparse shards against the v3 canonical CSR invariant.
    pub fn validate_canonical_csr_shards(&self) -> Vec<(String, bool)> {
        self.full_catalog
            .entries
            .iter()
            .filter(|entry| Self::is_canonical_csr_section(entry.section_type))
            .map(|entry| {
                (
                    entry.name.clone(),
                    self.validate_canonical_csr_entry(entry).is_ok(),
                )
            })
            .collect()
    }

    /// Which section types the v3 canonical-CSR invariant applies to.
    ///
    /// **Public because it must be the only such list.** `scx-cli`'s
    /// `validate --deep` loop kept its own copy that omitted `RawCsrShard`, so
    /// the CLI silently checked three of the four families — the exact gap
    /// SCX-008 added raw here to close, reopened one crate over. That is the
    /// same "one question answered in two places" shape the section-carry table
    /// exists to remove, and the answer is the same: one list, exported.
    ///
    /// Callers deciding whether to *emit* a canonical shard want this too: an
    /// op that stamps a version asserting the invariant must hold every one of
    /// these to it (see `scx_ops::rewrite_helpers::copy_csr_class_aux`).
    pub fn is_canonical_csr_section(section_type: SectionType) -> bool {
        // RawCsrShard is stored as canonical CSR too, so deep validation must
        // cover it — otherwise `.raw` corruption escapes the canonical pass
        // (SCX-008).
        matches!(
            section_type,
            SectionType::CsrShard
                | SectionType::LayerCsrShard
                | SectionType::ObspCsrShard
                | SectionType::RawCsrShard
        )
    }

    fn validate_decoded_canonical_csr(
        name: &str,
        header: &ShardHeader,
        indptr: &[i64],
        indices: &[i32],
        data: &[f32],
    ) -> Result<()> {
        let expected_indptr_len = header.n_major as usize + 1;
        if indptr.len() != expected_indptr_len {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indptr len {} != n_major + 1 {}",
                indptr.len(),
                expected_indptr_len
            )));
        }
        if indptr.first().copied() != Some(0) {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indptr must start at 0"
            )));
        }
        if indices.len() != data.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indices len {} != data len {}",
                indices.len(),
                data.len()
            )));
        }
        if header.nnz != indices.len() as u64 {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} header nnz {} != decoded nnz {}",
                header.nnz,
                indices.len()
            )));
        }

        for (row, window) in indptr.windows(2).enumerate() {
            let start_raw = window[0];
            let end_raw = window[1];
            if start_raw < 0 || end_raw < 0 || end_raw < start_raw {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {name} invalid indptr window at row {row}: {start_raw}..{end_raw}"
                )));
            }
            let start = usize::try_from(start_raw).map_err(|_| {
                ScxError::InvalidCatalog(format!(
                    "CSR shard {name} row {row} start overflows usize"
                ))
            })?;
            let end = usize::try_from(end_raw).map_err(|_| {
                ScxError::InvalidCatalog(format!("CSR shard {name} row {row} end overflows usize"))
            })?;
            if end > indices.len() {
                return Err(ScxError::InvalidCatalog(format!(
                    "CSR shard {name} row {row} end {end} exceeds decoded nnz {}",
                    indices.len()
                )));
            }

            let mut prev: Option<i32> = None;
            for pos in start..end {
                let idx = indices[pos];
                if idx < 0 || idx as u64 >= header.n_minor as u64 {
                    return Err(ScxError::InvalidCatalog(format!(
                        "CSR shard {name} row {row} index {idx} out of range [0, {})",
                        header.n_minor
                    )));
                }
                if let Some(prev_idx) = prev {
                    if idx <= prev_idx {
                        return Err(ScxError::InvalidCatalog(format!(
                        "CSR shard {name} row {row} indices are not strictly increasing: {idx} after {prev_idx}"
                    )));
                    }
                }
                if data[pos] == 0.0 {
                    return Err(ScxError::InvalidCatalog(format!(
                        "CSR shard {name} row {row} stores explicit zero at ordinal {pos}"
                    )));
                }
                prev = Some(idx);
            }
        }

        let last = *indptr.last().unwrap_or(&-1);
        if last < 0 || last as usize != indices.len() {
            return Err(ScxError::InvalidCatalog(format!(
                "CSR shard {name} indptr last {last} != decoded nnz {}",
                indices.len()
            )));
        }

        Ok(())
    }
}
