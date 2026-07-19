//! Explode: convert a packed `.scx` file into an exploded `.scxd` directory.
//!
//! Implements docs/cloud.md (Exploded layout). Each section becomes a separate file, enabling
//! individual section uploads to cloud object stores.
//!
//! Output directory structure:
//!   experiment.scxd/
//!   ├── _catalog.bin        (full catalog, written last)
//!   ├── _header.bin         (256-byte file header)
//!   ├── obs.arrow           (obs metadata)
//!   ├── var.arrow           (var metadata)
//!   ├── X/
//!   │   ├── 000000.shard    (CSR shard 0, byte-identical to packed)
//!   │   └── ...
//!   ├── obsm/               (optional)
//!   ├── layers/             (optional)
//!   └── uns.json            (optional)

use std::io::Write;
use std::path::Path;

use scx_format_io::section::SectionType;

use crate::error::Result;

/// Explode a packed `.scx` file into a directory of individual section files.
///
/// The `_catalog.bin` is written last for atomic-publish semantics.
///
/// Streams each section's byte range from the source in bounded chunks — peak
/// memory is `O(chunk size)`, independent of the source-file size, so this
/// works on multi-TB atlases that don't fit in RAM.
pub fn explode(input: &Path, output_dir: &Path) -> Result<()> {
    // Read only the header + catalog (KB–MB), not the whole file.
    let (_header, full_catalog) = crate::streaming::read_header_and_catalog(input)?;
    let mut src = std::fs::File::open(input)?;
    // One reusable streaming buffer for every section copy (avoids re-allocating
    // per shard on files with thousands of shards).
    let mut buf = vec![0u8; crate::streaming::CHUNK_SIZE];

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Write _header.bin (raw 256-byte header, byte-identical to source).
    let header_bytes = crate::streaming::read_raw_header(input)?;
    std::fs::write(output_dir.join("_header.bin"), &header_bytes[..])?;

    // Stream each section to its mapped file path.
    for entry in &full_catalog.entries {
        let rel_path = section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
            crate::error::CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        let file_path = output_dir.join(&rel_path);

        // Create parent directories
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut out = std::io::BufWriter::new(std::fs::File::create(&file_path)?);
        crate::streaming::copy_section(&mut src, entry.offset, entry.length, &mut out, &mut buf)?;
        out.flush()?;
    }

    // Write _catalog.bin LAST (atomic-publish semantics)
    let mut catalog_bytes = Vec::new();
    full_catalog.write_to(&mut catalog_bytes)?;
    std::fs::write(output_dir.join("_catalog.bin"), &catalog_bytes)?;

    Ok(())
}

/// Map a catalog entry's section name + type to a relative file path.
///
/// Returns an error if a shard index cannot be parsed from the section name.
pub(crate) fn section_name_to_path(
    name: &str,
    section_type: SectionType,
) -> std::result::Result<String, String> {
    match section_type {
        SectionType::ObsMetadata => Ok("obs.arrow".to_string()),
        SectionType::ObsIndex => Ok("obs_index.arrow".to_string()),
        // Global var → "var.arrow"; per-modality var (written by
        // `ScxWriter::write_var_for` as "var/{modality_name}") → top-level
        // "var.{modality_name}.arrow" so modalities never collide on the single
        // "var.arrow" object. A top-level dot-separated name (rather than a
        // `var/{name}.arrow` subdir file) keeps it out of the `var/` sharded-var
        // directory, so a modality named with pure digits can't be mistaken for
        // a `var/{idx:06}.arrow` shard on the reverse map.
        SectionType::VarMetadata => {
            if let Some(mname) = name.strip_prefix("var/") {
                Ok(format!("var.{mname}.arrow"))
            } else {
                Ok("var.arrow".to_string())
            }
        }
        SectionType::VarIndex => Ok("var_index.arrow".to_string()),
        SectionType::CsrShard => {
            // Phase G.2: per-modality CSR shards are named
            // "X/{modality_name}/shard_{idx}" by the writer (see
            // writer.rs::write_csr_shard_for). They map to
            // "X/{modality_name}/{idx:06}.shard" on disk so each
            // modality lives in its own directory.
            if let Some(rest) = name.strip_prefix("X/") {
                if let Some(slash_pos) = rest.rfind("/shard_") {
                    let mname = &rest[..slash_pos];
                    let idx_str = &rest[slash_pos + "/shard_".len()..];
                    let idx: u32 = idx_str.parse().map_err(|_| {
                        format!(
                            "invalid per-modality CsrShard name: cannot parse index from '{name}'"
                        )
                    })?;
                    return Ok(format!("X/{mname}/{idx:06}.shard"));
                }
            }
            // Legacy / single-modality: "X_shard_N" → "X/NNNNNN.shard"
            let idx: u32 = name
                .strip_prefix("X_shard_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("invalid CsrShard name: expected 'X_shard_N' or 'X/{{modality}}/shard_N', got '{name}'")
                })?;
            Ok(format!("X/{idx:06}.shard"))
        }
        SectionType::CscShard => {
            // Phase G.2: per-modality CSC shards are named
            // "X_csc/{modality_name}/shard_{idx}" by the writer (see
            // writer.rs::write_csc_shard_for). Map to
            // "Xc/{modality_name}/{idx:06}.shard".
            if let Some(rest) = name.strip_prefix("X_csc/") {
                if let Some(slash_pos) = rest.rfind("/shard_") {
                    let mname = &rest[..slash_pos];
                    let idx_str = &rest[slash_pos + "/shard_".len()..];
                    let idx: u32 = idx_str.parse().map_err(|_| {
                        format!(
                            "invalid per-modality CscShard name: cannot parse index from '{name}'"
                        )
                    })?;
                    return Ok(format!("Xc/{mname}/{idx:06}.shard"));
                }
            }
            // Legacy: "X_csc_shard_N" → "Xc/NNNNNN.shard"
            let idx: u32 = name
                .strip_prefix("X_csc_shard_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("invalid CscShard name: expected 'X_csc_shard_N' or 'X_csc/{{modality}}/shard_N', got '{name}'")
                })?;
            Ok(format!("Xc/{idx:06}.shard"))
        }
        SectionType::ObsmEmbedding => {
            // "obsm/{name}" → "obsm/{name}.arrow"
            let obsm_name = name.strip_prefix("obsm/").unwrap_or(name);
            Ok(format!("obsm/{obsm_name}.arrow"))
        }
        SectionType::LayerCsrShard => {
            // "{layer_name}_shard_N" → "layers/{layer_name}/NNNNNN.shard"
            if let Some(pos) = name.rfind("_shard_") {
                let layer_name = &name[..pos];
                let idx: u32 = name[pos + 7..].parse().map_err(|_| {
                    format!("invalid LayerCsrShard name: cannot parse shard index from '{name}'")
                })?;
                Ok(format!("layers/{layer_name}/{idx:06}.shard"))
            } else {
                Ok(format!("layers/{name}.bin"))
            }
        }
        SectionType::ObspCsrShard => {
            // "obsp/{name}_shard_N" → "obsp/{name}/NNNNNN.shard"
            let inner = name.strip_prefix("obsp/").unwrap_or(name);
            if let Some(pos) = inner.rfind("_shard_") {
                let obsp_name = &inner[..pos];
                let idx: u32 = inner[pos + 7..].parse().map_err(|_| {
                    format!("invalid ObspCsrShard name: cannot parse shard index from '{name}'")
                })?;
                Ok(format!("obsp/{obsp_name}/{idx:06}.shard"))
            } else {
                Ok(format!("obsp/{inner}.bin"))
            }
        }
        SectionType::UnsBlob => Ok("uns.json".to_string()),
        SectionType::Provenance => Ok("_provenance.bin".to_string()),
        SectionType::DeletionVectors => Ok("_deletion_vectors.bin".to_string()),
        SectionType::ObsPredicateIndex => Ok("_obs_predicate_index.bin".to_string()),
        SectionType::VarPredicateIndex => Ok("_var_predicate_index.bin".to_string()),
        // Phase 2 sharded obs/var metadata: "obs_metadata/shard_N" →
        // "obs/NNNNNN.arrow" (own subdirectory so it never collides with
        // the single-section "obs.arrow"). Mirror for var.
        SectionType::ObsMetadataShard => {
            let idx: u32 = name
                .strip_prefix("obs_metadata/shard_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("invalid ObsMetadataShard name: expected 'obs_metadata/shard_N', got '{name}'")
                })?;
            Ok(format!("obs/{idx:06}.arrow"))
        }
        SectionType::VarMetadataShard => {
            let idx: u32 = name
                .strip_prefix("var_metadata/shard_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("invalid VarMetadataShard name: expected 'var_metadata/shard_N', got '{name}'")
                })?;
            Ok(format!("var/{idx:06}.arrow"))
        }
        // Phase G.2: modality table is a single global section, written
        // as a top-level file in the exploded layout. The pack reader
        // reattaches it via `path_to_section_name`.
        SectionType::ModalityTable => Ok("_modality_table.bin".to_string()),
        _ => Ok(format!("{name}.bin")),
    }
}

/// Reverse mapping: given a file path, determine section name and type.
#[allow(dead_code)]
pub(crate) fn path_to_section_name(rel_path: &str) -> Option<(String, SectionType)> {
    match rel_path {
        "obs.arrow" => Some(("obs".to_string(), SectionType::ObsMetadata)),
        "obs_index.arrow" => Some(("obs_index".to_string(), SectionType::ObsIndex)),
        "var.arrow" => Some(("var".to_string(), SectionType::VarMetadata)),
        "var_index.arrow" => Some(("var_index".to_string(), SectionType::VarIndex)),
        "uns.json" => Some(("uns".to_string(), SectionType::UnsBlob)),
        "_provenance.bin" => Some(("provenance".to_string(), SectionType::Provenance)),
        "_deletion_vectors.bin" => {
            Some(("deletion_vectors".to_string(), SectionType::DeletionVectors))
        }
        "_obs_predicate_index.bin" => Some((
            "obs_predicate_index".to_string(),
            SectionType::ObsPredicateIndex,
        )),
        "_var_predicate_index.bin" => Some((
            "var_predicate_index".to_string(),
            SectionType::VarPredicateIndex,
        )),
        "_modality_table.bin" => Some(("modality_table".to_string(), SectionType::ModalityTable)),
        _ if rel_path.starts_with("X/") && rel_path.ends_with(".shard") => {
            // Phase G.2: per-modality CSR shards live at
            // "X/{modality_name}/{idx:06}.shard"; the legacy
            // single-modality layout is "X/{idx:06}.shard".
            let inner = rel_path
                .strip_prefix("X/")
                .unwrap()
                .strip_suffix(".shard")
                .unwrap();
            let parts: Vec<&str> = inner.rsplitn(2, '/').collect();
            if parts.len() == 2 {
                // "rna/000042" → "X/rna/shard_42"
                let idx: u32 = parts[0].parse().ok()?;
                let mname = parts[1];
                return Some((format!("X/{mname}/shard_{idx}"), SectionType::CsrShard));
            }
            let idx: u32 = inner.parse().ok()?;
            Some((format!("X_shard_{idx}"), SectionType::CsrShard))
        }
        // Reverse of "X_csc_shard_N" → "Xc/NNNNNN.shard".
        _ if rel_path.starts_with("Xc/") && rel_path.ends_with(".shard") => {
            let inner = rel_path
                .strip_prefix("Xc/")
                .unwrap()
                .strip_suffix(".shard")
                .unwrap();
            let parts: Vec<&str> = inner.rsplitn(2, '/').collect();
            if parts.len() == 2 {
                let idx: u32 = parts[0].parse().ok()?;
                let mname = parts[1];
                return Some((format!("X_csc/{mname}/shard_{idx}"), SectionType::CscShard));
            }
            let idx: u32 = inner.parse().ok()?;
            Some((format!("X_csc_shard_{idx}"), SectionType::CscShard))
        }
        // "obs/NNNNNN.arrow" → "obs_metadata/shard_N" (Phase 2 sharded
        // obs). Mirror for var. Checked before the generic "obsm/"
        // arm; the distinct "obs/" / "var/" prefixes avoid collision.
        _ if rel_path.starts_with("obs/") && rel_path.ends_with(".arrow") => {
            let idx: u32 = rel_path
                .strip_prefix("obs/")
                .unwrap()
                .strip_suffix(".arrow")
                .unwrap()
                .parse()
                .ok()?;
            Some((
                format!("obs_metadata/shard_{idx}"),
                SectionType::ObsMetadataShard,
            ))
        }
        // "var/NNNNNN.arrow" → "var_metadata/shard_N" (Phase 2 sharded var).
        // The `var/` subdir holds ONLY sharded var; per-modality var lives at
        // top-level "var.{name}.arrow" (below), so this stem is always numeric.
        _ if rel_path.starts_with("var/") && rel_path.ends_with(".arrow") => {
            let idx: u32 = rel_path
                .strip_prefix("var/")
                .unwrap()
                .strip_suffix(".arrow")
                .unwrap()
                .parse()
                .ok()?;
            Some((
                format!("var_metadata/shard_{idx}"),
                SectionType::VarMetadataShard,
            ))
        }
        // "var.{modality_name}.arrow" → per-modality var "var/{name}". Global
        // "var.arrow" is exact-matched above, so the middle segment is always a
        // non-empty modality name (even a pure-digit one — no shard collision,
        // since sharded var lives under the `var/` subdir).
        _ if rel_path.starts_with("var.")
            && rel_path.ends_with(".arrow")
            && rel_path.len() > "var.".len() + ".arrow".len() =>
        {
            let mname = &rel_path["var.".len()..rel_path.len() - ".arrow".len()];
            Some((format!("var/{mname}"), SectionType::VarMetadata))
        }
        _ if rel_path.starts_with("obsm/") && rel_path.ends_with(".arrow") => {
            let name = rel_path
                .strip_prefix("obsm/")
                .unwrap()
                .strip_suffix(".arrow")
                .unwrap();
            Some((format!("obsm/{name}"), SectionType::ObsmEmbedding))
        }
        _ if rel_path.starts_with("layers/") && rel_path.ends_with(".shard") => {
            // "layers/{name}/{idx:06}.shard" → "{name}_shard_{idx}"
            let inner = rel_path.strip_prefix("layers/").unwrap();
            let parts: Vec<&str> = inner.splitn(2, '/').collect();
            if parts.len() == 2 {
                let layer_name = parts[0];
                let idx_str = parts[1].strip_suffix(".shard")?;
                let idx: u32 = idx_str.parse().ok()?;
                Some((
                    format!("{layer_name}_shard_{idx}"),
                    SectionType::LayerCsrShard,
                ))
            } else {
                None
            }
        }
        _ if rel_path.starts_with("obsp/") && rel_path.ends_with(".shard") => {
            // "obsp/{name}/{idx:06}.shard" → "obsp/{name}_shard_{idx}"
            let inner = rel_path.strip_prefix("obsp/").unwrap();
            let parts: Vec<&str> = inner.splitn(2, '/').collect();
            if parts.len() == 2 {
                let obsp_name = parts[0];
                let idx_str = parts[1].strip_suffix(".shard")?;
                let idx: u32 = idx_str.parse().ok()?;
                Some((
                    format!("obsp/{obsp_name}_shard_{idx}"),
                    SectionType::ObspCsrShard,
                ))
            } else {
                None
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_section_name_to_path_mapping() {
        assert_eq!(
            section_name_to_path("obs", SectionType::ObsMetadata).unwrap(),
            "obs.arrow"
        );
        assert_eq!(
            section_name_to_path("var", SectionType::VarMetadata).unwrap(),
            "var.arrow"
        );
        assert_eq!(
            section_name_to_path("X_shard_0", SectionType::CsrShard).unwrap(),
            "X/000000.shard"
        );
        assert_eq!(
            section_name_to_path("X_shard_42", SectionType::CsrShard).unwrap(),
            "X/000042.shard"
        );
        assert_eq!(
            section_name_to_path("obsm/X_pca", SectionType::ObsmEmbedding).unwrap(),
            "obsm/X_pca.arrow"
        );
        assert_eq!(
            section_name_to_path("raw_counts_shard_0", SectionType::LayerCsrShard).unwrap(),
            "layers/raw_counts/000000.shard"
        );
        assert_eq!(
            section_name_to_path("obsp/distances_shard_3", SectionType::ObspCsrShard).unwrap(),
            "obsp/distances/000003.shard"
        );
        assert_eq!(
            section_name_to_path("uns", SectionType::UnsBlob).unwrap(),
            "uns.json"
        );
        assert_eq!(
            section_name_to_path("provenance", SectionType::Provenance).unwrap(),
            "_provenance.bin"
        );
        assert_eq!(
            section_name_to_path("deletion_vectors", SectionType::DeletionVectors).unwrap(),
            "_deletion_vectors.bin"
        );
        assert_eq!(
            section_name_to_path("obs_predicate_index", SectionType::ObsPredicateIndex).unwrap(),
            "_obs_predicate_index.bin"
        );
        assert_eq!(
            section_name_to_path("obs_metadata/shard_0", SectionType::ObsMetadataShard).unwrap(),
            "obs/000000.arrow"
        );
        assert_eq!(
            section_name_to_path("var_metadata/shard_7", SectionType::VarMetadataShard).unwrap(),
            "var/000007.arrow"
        );
        // Per-modality var gets a distinct path (no collision with global var
        // or across modalities).
        assert_eq!(
            section_name_to_path("var/rna", SectionType::VarMetadata).unwrap(),
            "var.rna.arrow"
        );
        assert_eq!(
            section_name_to_path("var/adt", SectionType::VarMetadata).unwrap(),
            "var.adt.arrow"
        );
    }

    #[test]
    fn per_modality_var_path_round_trips() {
        // Forward then reverse must recover the per-modality var section name
        // and type. A pure-digit modality name (e.g. "000007") must round-trip
        // as a modality var, NOT be mistaken for a `var/{idx:06}.arrow` shard —
        // per-modality var lives at top-level "var.{name}.arrow", sharded var
        // under the "var/" subdir.
        for mname in ["rna", "adt", "atac", "000007"] {
            let name = format!("var/{mname}");
            let path = section_name_to_path(&name, SectionType::VarMetadata).unwrap();
            assert_eq!(path, format!("var.{mname}.arrow"));
            let (rev_name, rev_ty) = path_to_section_name(&path).unwrap();
            assert_eq!(rev_name, name);
            assert_eq!(rev_ty, SectionType::VarMetadata);
        }
        // Global var round-trips to the global section.
        let g = section_name_to_path("var", SectionType::VarMetadata).unwrap();
        assert_eq!(g, "var.arrow");
        assert_eq!(
            path_to_section_name(&g).unwrap(),
            ("var".to_string(), SectionType::VarMetadata)
        );
        // A "var/NNNNNN.arrow" (subdir) is always a sharded-var section.
        assert_eq!(
            path_to_section_name("var/000007.arrow").unwrap(),
            (
                "var_metadata/shard_7".to_string(),
                SectionType::VarMetadataShard
            )
        );
    }

    #[test]
    fn test_section_name_to_path_rejects_invalid_shard_index() {
        assert!(section_name_to_path("X_shard_abc", SectionType::CsrShard).is_err());
        assert!(section_name_to_path("X_shard_", SectionType::CsrShard).is_err());
        assert!(section_name_to_path("bad_name", SectionType::CsrShard).is_err());
        assert!(section_name_to_path("raw_counts_shard_xyz", SectionType::LayerCsrShard).is_err());
        assert!(section_name_to_path("obsp/dist_shard_xyz", SectionType::ObspCsrShard).is_err());
        // CSC shard names must follow X_csc_shard_N pattern.
        assert!(section_name_to_path("X_csc_shard_abc", SectionType::CscShard).is_err());
        assert!(section_name_to_path("X_shard_0", SectionType::CscShard).is_err());
    }

    #[test]
    fn test_csc_shard_path_mapping() {
        assert_eq!(
            section_name_to_path("X_csc_shard_0", SectionType::CscShard).unwrap(),
            "Xc/000000.shard"
        );
        assert_eq!(
            section_name_to_path("X_csc_shard_5", SectionType::CscShard).unwrap(),
            "Xc/000005.shard"
        );
    }

    #[test]
    fn test_path_to_section_name_roundtrip() {
        let cases: Vec<(&str, SectionType)> = vec![
            ("obs", SectionType::ObsMetadata),
            ("var", SectionType::VarMetadata),
            ("X_shard_0", SectionType::CsrShard),
            ("X_shard_42", SectionType::CsrShard),
            ("X_csc_shard_0", SectionType::CscShard),
            ("X_csc_shard_5", SectionType::CscShard),
            ("obsm/X_pca", SectionType::ObsmEmbedding),
            ("uns", SectionType::UnsBlob),
            ("provenance", SectionType::Provenance),
            ("deletion_vectors", SectionType::DeletionVectors),
            ("obs_predicate_index", SectionType::ObsPredicateIndex),
            ("var_predicate_index", SectionType::VarPredicateIndex),
            ("obs_metadata/shard_0", SectionType::ObsMetadataShard),
            ("obs_metadata/shard_13", SectionType::ObsMetadataShard),
            ("var_metadata/shard_0", SectionType::VarMetadataShard),
            ("var_metadata/shard_9", SectionType::VarMetadataShard),
        ];

        for (name, st) in cases {
            let path = section_name_to_path(name, st).unwrap();
            let (recovered_name, recovered_type) = path_to_section_name(&path)
                .unwrap_or_else(|| panic!("path_to_section_name failed for path: {path}"));
            assert_eq!(recovered_name, name, "name roundtrip failed for {path}");
            assert_eq!(recovered_type, st, "type roundtrip failed for {path}");
        }
    }
}
