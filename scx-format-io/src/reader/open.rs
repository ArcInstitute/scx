//! Opening a file: the constructors and the freshness opt-in.
//!
//! Both constructors validate through the same helpers; see
//! [`ScxReader::open_with_shared_catalog`] for the one difference that
//! remains and why it is a rejection rather than a divergence.

use super::*;

impl ScxReader {
    /// Open an SCX file for reading.
    ///
    /// Validates the file header magic/version/endianness and the full
    /// catalog's trailing BLAKE3 checksum.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path, true)
    }

    /// Open an SCX file without verifying the catalog checksum.
    ///
    /// Skips the full catalog BLAKE3 verification for performance-sensitive
    /// paths where the file is trusted (e.g., `scx info`, repeated reads of
    /// a file that was already validated). The header magic/version/endianness
    /// are still checked.
    pub fn open_unchecked(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_inner(path, false)
    }

    /// `mmap` the file, echoing the offending path in any error — a bare
    /// "No such file or directory (os error 2)" forces the caller to guess
    /// which file failed, and the readers that hit this hardest open thousands
    /// of files per run.
    ///
    /// Shared by both constructors so neither can quietly lose the context;
    /// `open_with_shared_catalog` did, for exactly as long as it had its own
    /// copy of these four lines.
    /// Returns the mapping **and the identity of the inode it was taken
    /// from**, read through the very `File` that produced it.
    ///
    /// The identity has to come from here and nowhere else. Deriving it later
    /// from a second `std::fs::metadata(path)` leaves a window in which an
    /// atomic rename lands between the two, pairing the *new* file's inode with
    /// the *old* file's catalog — a hybrid stamp that a reopen of the new file
    /// then matches, so the reader silently changes files underneath a stable
    /// `file_id`. Found on #536 by codex - gpt-5.6-sol and confirmed
    /// independently by Cursor Agent and Antigravity.
    fn map_with_path_context(path: &Path) -> Result<(Mmap, crate::freshness::InodeIdentity)> {
        let file = File::open(path).map_err(|e| {
            ScxError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot open '{}': {}", path.display(), e),
            ))
        })?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                ScxError::Io(std::io::Error::new(
                    e.kind(),
                    format!("cannot mmap '{}': {}", path.display(), e),
                ))
            })?
        };

        // Start with Normal advice (kernel default heuristic). Access-pattern-
        // specific hints (Sequential, WillNeed, DontNeed) are issued at each
        // call site — see assemble_shards*() and BackedCsrReader.
        #[cfg(unix)]
        {
            use memmap2::Advice;
            let _ = mmap.advise(Advice::Normal);
        }

        let meta = file.metadata().map_err(|e| {
            ScxError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot stat '{}': {}", path.display(), e),
            ))
        })?;
        let inode = crate::freshness::InodeIdentity::of(&meta);

        // Check minimum file size (header + root catalog placeholder)
        if mmap.len() < HEADER_SIZE {
            return Err(ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "file too small: {} bytes (minimum {})",
                    mmap.len(),
                    HEADER_SIZE
                ),
            )));
        }
        Ok((mmap, inode))
    }

    /// Read the root catalog at offset 256.
    ///
    /// The cursor is bounded to the root-catalog region so a corrupt entry
    /// count cannot read past it into section-body bytes (SCX-007).
    /// `RootCatalog::read_from` also rejects an over-large count on its own,
    /// so this is defence in depth rather than the only line — but it is the
    /// line that survives a change to that parser, and having it in one place
    /// is why both constructors now get it.
    fn read_root_catalog(mmap: &Mmap) -> Result<RootCatalog> {
        let root_end = (HEADER_SIZE + crate::ROOT_CATALOG_MAX_SIZE).min(mmap.len());
        RootCatalog::read_from(&mut Cursor::new(&mmap[HEADER_SIZE..root_end]))
    }

    /// Parse the `ModalityTable` section if the header points at one. The
    /// pointer is `0/0` for single-modality files (legacy shape and v1 files
    /// alike), which yields `None`.
    ///
    /// Extracted because both constructors carried a byte-identical copy.
    fn parse_modality_table(mmap: &Mmap, header: &FileHeader) -> Result<Option<ModalityTable>> {
        if header.n_modalities == 0
            || header.modality_table_offset == 0
            || header.modality_table_length == 0
        {
            return Ok(None);
        }
        let mt_off = header.modality_table_offset as usize;
        let mt_len = header.modality_table_length as usize;
        let mt_end = mt_off
            .checked_add(mt_len)
            .ok_or(ScxError::SectionOutOfBounds {
                offset: header.modality_table_offset,
                length: header.modality_table_length,
                file_size: mmap.len(),
            })?;
        if mt_end > mmap.len() {
            return Err(ScxError::SectionOutOfBounds {
                offset: header.modality_table_offset,
                length: header.modality_table_length,
                file_size: mmap.len(),
            });
        }
        let table = ModalityTable::read_from(&mut Cursor::new(&mmap[mt_off..mt_end]), mt_len)?;
        // Cross-check header.n_modalities against the table's embedded count.
        // Disagreement is corruption, not a v1/v2 mismatch.
        if table.len() as u32 != header.n_modalities {
            return Err(ScxError::InvalidCatalog(format!(
                "header.n_modalities ({}) != ModalityTable.len() ({})",
                header.n_modalities,
                table.len()
            )));
        }
        Ok(Some(table))
    }

    /// Verify that a v1 catalog handed to [`Self::open_with_shared_catalog`]
    /// has already been through `reconcile_v1_csr_col_range`.
    ///
    /// The discriminator is the exact pair the reconciliation writes:
    /// `col_start = 0`, `col_end = n_vars`. v1 stats carry no column pair on
    /// disk, so `FullCatalog::read_from` leaves both at `0`; a row-major entry
    /// not carrying that pair on a file with `n_vars > 0` never went through
    /// the reconciliation, and its shard ranges would be wrong for any
    /// column-projected read.
    ///
    /// Checking the whole pair rather than `col_end != 0` matters: the looser
    /// form accepted any nonzero `col_end` and never looked at `col_start`, so
    /// it admitted catalogs carrying a column range that is not the one
    /// reconciliation produces — a weaker guarantee than the one this function
    /// documents.
    ///
    /// No-op for v2+, where the column pair is on disk and the reconciliation
    /// is itself a no-op.
    ///
    /// `ObspCsrShard` is checked against `n_vars` like the others, which is the
    /// *wrong* axis for an obs x obs graph — deliberately, and only here. See
    /// `FullCatalog::reconcile_v1_csr_col_range`: this asserts that the
    /// reconciliation ran, so it must expect exactly what that function writes,
    /// and that function keeps the legacy axis so legacy files stay readable.
    fn check_v1_catalog_reconciled(catalog: &FullCatalog, header: &FileHeader) -> Result<()> {
        if catalog.catalog_version >= 2 || header.n_vars == 0 {
            return Ok(());
        }
        for e in &catalog.entries {
            if !matches!(
                e.section_type,
                SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
            ) {
                continue;
            }
            if let Some(stats) = e.stats.as_ref() {
                if stats.col_start != 0 || stats.col_end != header.n_vars {
                    return Err(ScxError::InvalidCatalog(format!(
                        "shared catalog is v1 and entry '{}' has not been reconciled \
                         (col_start/col_end {}..{}, expected 0..{}); this path cannot \
                         reconcile a shared Arc<FullCatalog> — pass a catalog from \
                         ScxReader::open()",
                        e.name, stats.col_start, stats.col_end, header.n_vars,
                    )));
                }
            }
        }
        Ok(())
    }

    fn open_inner(path: impl AsRef<Path>, verify_catalog: bool) -> Result<Self> {
        let path = path.as_ref();
        let (mmap, inode) = Self::map_with_path_context(path)?;

        // Read and validate file header
        let header = FileHeader::read_from(&mut Cursor::new(&mmap[..HEADER_SIZE]))?;

        let root_catalog = Self::read_root_catalog(&mmap)?;

        // Read full catalog using header's offset and length
        let fc_offset = header.full_catalog_offset as usize;
        let fc_length = header.full_catalog_length as usize;
        let fc_end = fc_offset
            .checked_add(fc_length)
            .ok_or(ScxError::SectionOutOfBounds {
                offset: header.full_catalog_offset,
                length: header.full_catalog_length,
                file_size: mmap.len(),
            })?;
        if fc_end > mmap.len() {
            return Err(ScxError::SectionOutOfBounds {
                offset: header.full_catalog_offset,
                length: header.full_catalog_length,
                file_size: mmap.len(),
            });
        }
        let fc_slice = &mmap[fc_offset..fc_end];
        let mut full_catalog =
            FullCatalog::read_from(&mut Cursor::new(fc_slice), fc_length, verify_catalog)?;

        // v1 → v2 reconciliation: populate `col_start`/`col_end` for
        // row-major shard entries from `n_vars`. CSC entries are
        // already reconciled inside `FullCatalog::read_from`. No-op on
        // v2 catalogs.
        full_catalog.reconcile_v1_csr_col_range(header.n_vars);

        let modality_table = Self::parse_modality_table(&mmap, &header)?;

        Ok(ScxReader {
            mmap,
            inode,
            path: path.to_path_buf(),
            freshness: None,
            header,
            root_catalog,
            full_catalog: Arc::new(full_catalog),
            modality_table,
            debug_counts: ReaderDebugCounts::default(),
        })
    }

    /// Open an SCX file reusing an already-parsed `FullCatalog` from a
    /// sibling `ScxReader` against the same file. Skips
    /// `FullCatalog::read_from` entirely — the expensive part of
    /// `open()` for files with thousands of catalog entries.
    ///
    /// Worker-amplification use case: `to_anndata_backed` opens N+3
    /// `ScxReader` instances per call (main reader + X CSR + CSC
    /// sidecar + N backed layers). The catalog is identical
    /// bytes-for-bytes across all of them; sharing one parsed copy
    /// collapses the per-call parse cost from `(N+3) ×` to `1×` on
    /// the worker construction path that `cell-load-scx` /
    /// `state-scx` hit inside their DataLoader iterators.
    ///
    /// The header magic / version / endianness and the root catalog
    /// at offset 256 are still validated against the fresh mmap, and
    /// the fresh header's `manifest_sequence` is compared against the
    /// shared catalog's — any divergence means the file was mutated
    /// between opens (`scx-ops` append / compact / rollback bumps the
    /// sequence) and the shared catalog no longer describes this mmap.
    /// The function does not re-verify the trailing BLAKE3 catalog
    /// checksum: the source `ScxReader` already did that at its own
    /// open, and the sequence check covers the mutation case the
    /// checksum would otherwise have to catch.
    ///
    /// # v1 catalogs must arrive reconciled
    ///
    /// [`Self::open_inner`] finishes by calling
    /// `FullCatalog::reconcile_v1_csr_col_range`, which backfills
    /// `col_start` / `col_end` on v1 row-major entries. This path cannot: it
    /// receives an `Arc<FullCatalog>`, and the reconciliation takes
    /// `&mut self`.
    ///
    /// So it checks that the reconciliation has already happened, rather than
    /// assuming it or refusing v1 outright. Refusing outright is what this did
    /// first, and it was wrong in the direction that matters:
    /// `reconcile_v1_csr_col_range` does not bump `catalog_version`, so a
    /// *reconciled* v1 catalog still reports `1` — and every donor that comes
    /// straight from `ScxReader::open()` is exactly that. `pyscx`'s backed
    /// conversion opens `X` through this constructor unconditionally, so a
    /// blanket refusal turned an internal invariant into a user-visible v1
    /// read-compatibility break.
    ///
    /// # Fork safety
    ///
    /// `Arc<FullCatalog>` is `Send + Sync` and has no interior
    /// mutability. When pyscx's worker iterators fork after the
    /// parent has constructed a `BackedCsrReader` ladder, the
    /// shared catalog is COW-duplicated into each child — no locks,
    /// no mutexes, no shared mutable state.
    pub fn open_with_shared_catalog(
        path: impl AsRef<Path>,
        catalog: Arc<FullCatalog>,
    ) -> Result<Self> {
        let path = path.as_ref();

        let (mmap, inode) = Self::map_with_path_context(path)?;

        let header = FileHeader::read_from(&mut Cursor::new(&mmap[..HEADER_SIZE]))?;

        // See "# v1 catalogs must arrive reconciled" above.
        Self::check_v1_catalog_reconciled(&catalog, &header)?;
        if header.manifest_sequence != catalog.manifest_sequence {
            return Err(ScxError::InvalidCatalog(format!(
                "shared catalog manifest_sequence ({}) does not match file header ({}) — \
                 file was mutated between opens",
                catalog.manifest_sequence, header.manifest_sequence,
            )));
        }
        let root_catalog = Self::read_root_catalog(&mmap)?;

        // Re-parse the (small) modality table from this instance's mmap.
        // It's only present on multimodal v2 files and is small; the
        // parse cost is negligible vs the full catalog.
        let modality_table = Self::parse_modality_table(&mmap, &header)?;

        Ok(ScxReader {
            mmap,
            inode,
            path: path.to_path_buf(),
            freshness: None,
            header,
            root_catalog,
            full_catalog: catalog,
            modality_table,
            debug_counts: ReaderDebugCounts::default(),
        })
    }

    /// Opt this reader into detecting that its file changed on disk.
    ///
    /// Stamps the file's current identity (see [`crate::freshness`]) and makes
    /// every subsequent [`Self::section_bytes`] verify it first, so a read
    /// after an in-place or copy-out mutation raises
    /// [`ScxError::FileChangedOnDisk`] instead of quietly answering from the
    /// mapping of a file that no longer exists at that path.
    ///
    /// Deliberately a separate, explicit step rather than a variant of every
    /// constructor: the readers that must *not* watch (`scx-ops` brackets its
    /// own mutations with them, and the loader's hot path opens thousands)
    /// outnumber the ones that must, so opting in is greppable and opting out
    /// is the default.
    ///
    /// Costs one `open` + `stat` here, then two syscalls (`stat` + a 256-byte
    /// `pread`) per section read — see [`crate::freshness`] for why the stat
    /// alone will not do. Chains off any constructor:
    ///
    /// ```no_run
    /// # use scx_format_io::ScxReader;
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let reader = ScxReader::open("cells.scx")?.watching()?;
    /// # Ok(()) }
    /// ```
    pub fn watching(mut self) -> Result<Self> {
        self.freshness = Some(crate::freshness::FreshnessGuard::stamp(
            &self.path,
            &self.header,
        )?);
        Ok(self)
    }

    /// Whether this reader was opted into change detection via
    /// [`Self::watching`].
    pub fn is_watching(&self) -> bool {
        self.freshness.is_some()
    }

    /// `Ok(())` unless this is a watched reader whose file has changed since it
    /// was opened. Always `Ok(())` for an unwatched reader.
    ///
    /// [`Self::section_bytes`] calls this itself, so it covers every section
    /// read in the workspace from one site. Call it directly only for answers
    /// that never touch a section — `n_obs`, `shape`, `has_csc` and the rest
    /// come from the parsed header or catalog and would otherwise stay
    /// silently stale after an `append`.
    pub fn check_fresh(&self) -> Result<()> {
        match &self.freshness {
            Some(guard) => guard.check(),
            None => Ok(()),
        }
    }
}
