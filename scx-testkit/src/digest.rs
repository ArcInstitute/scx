//! Per-section content digest of an SCX file.

use std::collections::BTreeSet;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use scx_format_io::checksum::blake3_hash;
use scx_format_io::section::SectionType;
use scx_format_io::{Result, ScxError, ScxReader};
use serde::{Deserialize, Serialize};

/// Environment variable that rewrites goldens instead of asserting.
pub const BLESS_ENV: &str = "SCX_TESTKIT_BLESS";

/// How much of the file layout a comparison pins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strictness {
    /// Section identity, size and content — but not where the section landed.
    ///
    /// The default for a refactor A/B. Section offsets shift whenever anything
    /// earlier in the file changes size, including the excluded `Provenance`
    /// section, so pinning them would make the comparison depend on exactly the
    /// thing it is trying to ignore.
    Content,
    /// Also pins each section's byte offset, so a pure layout change fails.
    ///
    /// Use for a path whose contract is the layout itself — a byte-passthrough
    /// copy, or an in-place mutation claiming not to move the matrix.
    Layout,
}

/// One catalog section, hashed independently of where it landed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionDigest {
    /// Section name, e.g. `X_shard_0`.
    pub name: String,
    /// Raw section-type id.
    ///
    /// Stored raw rather than as a `SectionType` so the digest does not depend
    /// on this crate's view of the type table. Note it does **not** buy
    /// forward-compatibility: `FullCatalog::read_from` skips section types it
    /// does not recognise (`catalog.rs`, "skipping unknown section type"), so an
    /// unknown type never reaches this function at all.
    pub section_type: u16,
    /// `0` for the global / primary modality.
    pub modality_id: u8,
    /// Section length in bytes.
    pub length: u64,
    /// Byte offset — `Some` only under [`Strictness::Layout`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub offset: Option<u64>,
    /// BLAKE3 of the section bytes, hex-encoded.
    pub blake3: String,
    /// The entry's `ShardStats`, when it has them.
    ///
    /// **Load-bearing, not decoration.** These live in the catalog, not in the
    /// section payload, so a file whose shard bytes are byte-identical can still
    /// carry different `row_start` / `nnz` / per-column stats — and readers act
    /// on them: shards are ordered by `row_start`, export budgets from `nnz`,
    /// and predicate pushdown prunes on `column_stats`. §6.1 of the review is
    /// precisely that shape (stale `column_stats` survive an obs replacement, so
    /// a query silently returns nothing), and a digest blind to them could not
    /// have caught it.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub stats: Option<StatsDigest>,
}

/// A catalog entry's `ShardStats`, flattened for comparison.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatsDigest {
    pub row_start: u64,
    pub row_end: u64,
    pub col_start: u64,
    pub col_end: u64,
    pub nnz: u64,
    pub value_min: u32,
    pub value_max: u32,
    pub value_sum: u64,
    /// Kept alongside `column_stats` deliberately: the two drifting apart is a
    /// decode bug (`ShardStats::read_from` reads exactly this many), so the
    /// digest should catch a file where they disagree.
    pub n_indexed_columns: u8,
    /// One entry per column stat, `(column_name_hash, discriminant, payload)`.
    /// `MinMax` payloads are compared by bit pattern so a `-0.0`/`0.0` or NaN
    /// change is not silently equal.
    pub column_stats: Vec<String>,
}

/// The header fields a refactor must not change.
///
/// `file_checksum` is deliberately absent: it covers the provenance section,
/// so it differs between two runs of the same op and carries no information a
/// per-section hash does not already carry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeaderDigest {
    pub format_version: u16,
    pub flags: u32,
    pub n_obs: u64,
    pub n_vars: u64,
    pub nnz: u64,
    pub n_csr_shards: u32,
    pub n_csc_shards: u32,
    pub shard_target_rows: u32,
    pub codec_id: u8,
    pub index_dtype: u8,
    pub manifest_sequence: u64,
    pub n_modalities: u32,
}

/// A whole file, reduced to what a behaviour-preserving change must preserve.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileDigest {
    pub strictness: Strictness,
    /// Section types excluded from `sections`, by raw id.
    pub excluded_section_types: Vec<u16>,
    pub header: HeaderDigest,
    /// Catalog-level fields that live outside every section payload.
    pub catalog: CatalogDigest,
    /// Sorted by `(name, modality_id)`, so shard *write order* does not affect
    /// the digest — only shard *content and boundaries* do. Reordering which
    /// rows land in which shard changes `X_shard_0`'s bytes and is caught.
    pub sections: Vec<SectionDigest>,
}

/// Catalog-level scalars, none of which appear in any section's bytes.
///
/// **Not** covered: the 4096-byte root catalog at offset 256. Verified: it has
/// **zero** production readers workspace-wide — every writer rebuilds it from
/// the full catalog — so two files differing only there behave identically, and
/// digesting it would make an A/B fail on a difference nothing can observe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogDigest {
    pub catalog_version: u16,
    pub manifest_sequence: u64,
    pub n_obs: u64,
    /// Bumped by every in-place CSR-mutating writer; a rewrite that forgets to
    /// bump it is a real defect a payload hash cannot see.
    pub data_generation: u64,
    /// The `data_generation` a CSC sidecar was built against. A mismatch makes
    /// readers reject the sidecar, so it is behaviour, not bookkeeping.
    pub csc_build_generation: u64,
    /// BLAKE3 of the front catalog's bytes, when the file carries one.
    ///
    /// The front catalog is a **second parsed copy** of the catalog, not
    /// decoration: `scx-cloud`'s `CloudReader` reads it in preference to the
    /// EOF one when `has_front_catalog()` is set. A cloud-ready file can
    /// therefore have an intact EOF catalog, intact section payloads, and a
    /// stale front catalog — and `open_cloud` would see different bytes from
    /// every local reader. Hashed rather than parsed so a difference is caught
    /// whatever its shape.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub front_catalog_blake3: Option<String>,
    /// Header catalog pointers — `Some` only under [`Strictness::Layout`].
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub catalog_offset: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub front_catalog_offset: Option<u64>,
    /// The rollback chain's back-pointer — `Some` only under
    /// [`Strictness::Layout`].
    ///
    /// Excluded from `Content` on purpose: a fresh write and a rewrite of the
    /// same logical content legitimately point at different previous catalogs,
    /// and that pair must compare equal. But `scx_ops::rollback` walks this
    /// chain, so under `Layout` — whose job is physical identity — a file
    /// pointing rollback somewhere else is a real difference.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prev_catalog_offset: Option<u64>,
}

/// Section types excluded from a digest by default.
///
/// Only `Provenance`. Every mutating write path stamps `SystemTime::now()`
/// into it, so including it would make every comparison a coin flip on the
/// wall clock — see the module docs. Keep this list at one entry: each
/// addition is a region of the file the harness stops covering.
pub const DEFAULT_EXCLUDED: &[SectionType] = &[SectionType::Provenance];

/// Digest `path`, excluding [`DEFAULT_EXCLUDED`].
pub fn digest_file(path: &Path, strictness: Strictness) -> Result<FileDigest> {
    digest_file_excluding(path, strictness, DEFAULT_EXCLUDED)
}

/// Digest `path` with an explicit exclusion list.
///
/// Pass `&[]` to include provenance — used by the harness's own tests to prove
/// the default exclusion is narrow rather than a hole that would swallow a real
/// difference.
pub fn digest_file_excluding(
    path: &Path,
    strictness: Strictness,
    excluded: &[SectionType],
) -> Result<FileDigest> {
    let reader = ScxReader::open(path)?;
    let skip: BTreeSet<u16> = excluded.iter().map(|t| *t as u16).collect();

    let mut sections = Vec::new();
    for entry in &reader.catalog().entries {
        if skip.contains(&(entry.section_type as u16)) {
            continue;
        }
        let bytes = reader.section_bytes(entry)?;
        // Hash the bytes, not the catalog's stored checksum. They should agree
        // — but a writer that stamps the wrong checksum is exactly the kind of
        // defect a refactor can introduce, and a digest built from the stored
        // value would be blind to it in both arms equally.
        let computed = blake3_hash(bytes);
        if computed != entry.checksum {
            return Err(ScxError::ChecksumMismatch {
                section: entry.name.clone(),
            });
        }
        sections.push(SectionDigest {
            name: entry.name.clone(),
            section_type: entry.section_type as u16,
            modality_id: entry.modality_id,
            length: entry.length,
            offset: match strictness {
                Strictness::Content => None,
                Strictness::Layout => Some(entry.offset),
            },
            blake3: hex(&computed),
            stats: entry.stats.as_ref().map(stats_digest),
        });
    }
    sections.sort_by(|a, b| (&a.name, a.modality_id).cmp(&(&b.name, b.modality_id)));

    let h = reader.header();
    let c = reader.catalog();
    Ok(FileDigest {
        strictness,
        excluded_section_types: skip.into_iter().collect(),
        catalog: CatalogDigest {
            catalog_version: c.catalog_version,
            manifest_sequence: c.manifest_sequence,
            n_obs: c.n_obs,
            data_generation: c.data_generation,
            csc_build_generation: c.csc_build_generation,
            front_catalog_blake3: front_catalog_hash(path, h)?,
            prev_catalog_offset: match strictness {
                Strictness::Content => None,
                Strictness::Layout => Some(c.prev_catalog_offset),
            },
            catalog_offset: match strictness {
                Strictness::Content => None,
                Strictness::Layout => Some(h.full_catalog_offset),
            },
            front_catalog_offset: match strictness {
                Strictness::Content => None,
                Strictness::Layout => Some(h.front_catalog_offset),
            },
        },
        header: HeaderDigest {
            format_version: h.format_version,
            flags: h.flags,
            n_obs: h.n_obs,
            n_vars: h.n_vars,
            nnz: h.nnz,
            n_csr_shards: h.n_csr_shards,
            n_csc_shards: h.n_csc_shards,
            shard_target_rows: h.shard_target_rows,
            codec_id: h.codec_id,
            index_dtype: h.index_dtype,
            manifest_sequence: h.manifest_sequence,
            n_modalities: h.n_modalities,
        },
        sections,
    })
}

/// Assert two digests are equal, naming what differs.
///
/// # Panics
///
/// On any difference, with a per-section report: sections present on one side
/// only, and sections whose length / offset / content changed.
pub fn assert_digests_eq(actual: &FileDigest, expected: &FileDigest) {
    if actual == expected {
        return;
    }
    let mut report = Vec::new();

    if actual.strictness != expected.strictness {
        report.push(format!(
            "strictness: {:?} vs {:?} — the two digests are not comparable",
            actual.strictness, expected.strictness
        ));
    }
    if actual.excluded_section_types != expected.excluded_section_types {
        report.push(format!(
            "excluded section types: {:?} vs {:?} — the two digests cover \
             different parts of the file",
            actual.excluded_section_types, expected.excluded_section_types
        ));
    }
    if actual.header != expected.header {
        report.push(format!(
            "header: {:?}\n     expected: {:?}",
            actual.header, expected.header
        ));
    }
    if actual.catalog != expected.catalog {
        report.push(format!(
            "catalog: {:?}\n     expected: {:?}",
            actual.catalog, expected.catalog
        ));
    }

    let key = |s: &SectionDigest| (s.name.clone(), s.modality_id);
    let a: BTreeSet<_> = actual.sections.iter().map(key).collect();
    let e: BTreeSet<_> = expected.sections.iter().map(key).collect();
    for (n, m) in e.difference(&a) {
        report.push(format!("missing section: {n} (modality {m})"));
    }
    for (n, m) in a.difference(&e) {
        report.push(format!("unexpected section: {n} (modality {m})"));
    }
    for sa in &actual.sections {
        let Some(se) = expected
            .sections
            .iter()
            .find(|s| s.name == sa.name && s.modality_id == sa.modality_id)
        else {
            continue;
        };
        if sa == se {
            continue;
        }
        let mut what = Vec::new();
        if sa.length != se.length {
            what.push(format!("length {} vs {}", sa.length, se.length));
        }
        if sa.offset != se.offset {
            what.push(format!("offset {:?} vs {:?}", sa.offset, se.offset));
        }
        if sa.blake3 != se.blake3 {
            what.push(format!(
                "content {}… vs {}…",
                &sa.blake3[..16],
                &se.blake3[..16]
            ));
        }
        if sa.section_type != se.section_type {
            what.push(format!(
                "section type {} vs {}",
                sa.section_type, se.section_type
            ));
        }
        if sa.stats != se.stats {
            // Named separately from `content`: identical shard bytes with
            // different catalog stats is the §6.1 shape, and reporting it as a
            // generic mismatch would send the reader looking in the payload.
            what.push(format!("catalog stats {:?} vs {:?}", sa.stats, se.stats));
        }
        report.push(format!("{}: {}", sa.name, what.join(", ")));
    }

    panic!(
        "SCX output digest changed ({} difference(s)):\n  {}\n\n\
         If the change was intended, re-run with {BLESS_ENV}=1 to update the \
         golden and review the diff.",
        report.len(),
        report.join("\n  ")
    );
}

/// Assert `path`'s digest matches a checked-in golden JSON file.
///
/// With [`BLESS_ENV`]`=1` the golden is rewritten instead. A missing golden is
/// always written (and then fails, so a first run cannot pass by accident).
pub fn assert_matches_golden(path: &Path, golden: &Path, strictness: Strictness) -> Result<()> {
    let actual = digest_file(path, strictness)?;
    let bless = std::env::var(BLESS_ENV).as_deref() == Ok("1");

    if bless || !golden.exists() {
        if let Some(parent) = golden.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(golden, serde_json::to_string_pretty(&actual).unwrap())?;
        assert!(
            bless,
            "golden {} did not exist; it has been written. Review it and re-run \
             — a first run must not pass by writing its own expectation.",
            golden.display()
        );
        return Ok(());
    }

    let text = std::fs::read_to_string(golden)?;
    let expected: FileDigest = serde_json::from_str(&text).unwrap_or_else(|e| {
        panic!(
            "golden {} is not a FileDigest: {e}. Delete it and re-run with \
             {BLESS_ENV}=1 to regenerate.",
            golden.display()
        )
    });
    assert_digests_eq(&actual, &expected);
    Ok(())
}

/// Hash the front catalog's bytes, or `None` when the file has none.
fn front_catalog_hash(
    path: &Path,
    h: &scx_format_io::header::FileHeader,
) -> Result<Option<String>> {
    if !h.has_front_catalog() || h.front_catalog_offset == 0 || h.front_catalog_length == 0 {
        return Ok(None);
    }
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(h.front_catalog_offset))?;
    let mut buf = vec![0u8; h.front_catalog_length as usize];
    f.read_exact(&mut buf)?;
    Ok(Some(hex(&blake3_hash(&buf))))
}

/// Flatten a `ShardStats` for comparison.
///
/// `column_stats` becomes one string per stat rather than a structured value:
/// `ColumnStat` is not `Eq` (it carries `f64`), and comparing the bit pattern is
/// what makes a `0.0` → `-0.0` or a NaN change visible instead of silently equal.
fn stats_digest(s: &scx_format_io::catalog::ShardStats) -> StatsDigest {
    use scx_format_io::catalog::ColumnStat;
    StatsDigest {
        row_start: s.row_start,
        row_end: s.row_end,
        col_start: s.col_start,
        col_end: s.col_end,
        nnz: s.nnz,
        value_min: s.value_min,
        value_max: s.value_max,
        value_sum: s.value_sum,
        n_indexed_columns: s.n_indexed_columns,
        column_stats: s
            .column_stats
            .iter()
            .map(|cs| match cs {
                ColumnStat::MinMax {
                    column_name_hash,
                    min,
                    max,
                } => format!(
                    "minmax:{column_name_hash:016x}:{:016x}:{:016x}",
                    min.to_bits(),
                    max.to_bits()
                ),
                ColumnStat::CategoryBitset {
                    column_name_hash,
                    bitset,
                } => format!(
                    "bitset:{column_name_hash:016x}:{}",
                    hex(&blake3_hash(bitset))
                ),
            })
            .collect(),
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
#[path = "digest_tests.rs"]
mod tests;
