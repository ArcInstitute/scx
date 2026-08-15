//! Per-section content digest of an SCX file.

use std::collections::BTreeSet;
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
    /// Raw section-type id, so an unknown future type still digests rather
    /// than being silently dropped from the comparison.
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
    /// Sorted by `(name, modality_id)`, so shard *write order* does not affect
    /// the digest — only shard *content and boundaries* do. Reordering which
    /// rows land in which shard changes `X_shard_0`'s bytes and is caught.
    pub sections: Vec<SectionDigest>,
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
        });
    }
    sections.sort_by(|a, b| (&a.name, a.modality_id).cmp(&(&b.name, b.modality_id)));

    let h = reader.header();
    Ok(FileDigest {
        strictness,
        excluded_section_types: skip.into_iter().collect(),
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

fn hex(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
#[path = "digest_tests.rs"]
mod tests;
