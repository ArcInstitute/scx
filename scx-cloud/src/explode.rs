//! Explode: convert a packed `.scx` file into an exploded `.scxd` directory.
//!
//! Implements SPEC §12.5. Each section becomes a separate file, enabling
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

use std::io::Cursor;
use std::path::Path;

use scx_format::catalog::FullCatalog;
use scx_format::header::{FileHeader, HEADER_SIZE};
use scx_format::section::SectionType;

use crate::error::Result;

/// Explode a packed `.scx` file into a directory of individual section files.
///
/// The `_catalog.bin` is written last for atomic-publish semantics.
pub fn explode(input: &Path, output_dir: &Path) -> Result<()> {
    let input_data = std::fs::read(input)?;
    let header = FileHeader::read_from(&mut Cursor::new(&input_data[..HEADER_SIZE]))?;

    // Read full catalog
    let fc_offset = header.full_catalog_offset as usize;
    let fc_length = header.full_catalog_length as usize;
    let fc_end = fc_offset.checked_add(fc_length).ok_or_else(|| {
        crate::error::CloudError::SliceBoundsExceeded {
            offset: fc_offset,
            length: fc_length,
            data_len: input_data.len(),
        }
    })?;
    if fc_end > input_data.len() {
        return Err(crate::error::CloudError::SliceBoundsExceeded {
            offset: fc_offset,
            length: fc_length,
            data_len: input_data.len(),
        });
    }
    let full_catalog = FullCatalog::read_from(
        &mut Cursor::new(&input_data[fc_offset..fc_end]),
        fc_length,
        true,
    )?;

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Write _header.bin
    std::fs::write(output_dir.join("_header.bin"), &input_data[..HEADER_SIZE])?;

    // Write each section to its mapped file path
    for entry in &full_catalog.entries {
        let rel_path = section_name_to_path(&entry.name, entry.section_type).map_err(|e| {
            crate::error::CloudError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e))
        })?;
        let file_path = output_dir.join(&rel_path);

        // Create parent directories
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Copy section bytes verbatim
        let src_start = entry.offset as usize;
        let src_len = entry.length as usize;
        let src_end = src_start.checked_add(src_len).ok_or_else(|| {
            crate::error::CloudError::SliceBoundsExceeded {
                offset: src_start,
                length: src_len,
                data_len: input_data.len(),
            }
        })?;
        if src_end > input_data.len() {
            return Err(crate::error::CloudError::SliceBoundsExceeded {
                offset: src_start,
                length: src_len,
                data_len: input_data.len(),
            });
        }
        std::fs::write(&file_path, &input_data[src_start..src_end])?;
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
        SectionType::VarMetadata => Ok("var.arrow".to_string()),
        SectionType::VarIndex => Ok("var_index.arrow".to_string()),
        SectionType::CsrShard => {
            // "X_shard_N" → "X/NNNNNN.shard"
            let idx: u32 = name
                .strip_prefix("X_shard_")
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| {
                    format!("invalid CsrShard name: expected 'X_shard_N', got '{name}'")
                })?;
            Ok(format!("X/{idx:06}.shard"))
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
        _ if rel_path.starts_with("X/") && rel_path.ends_with(".shard") => {
            let idx_str = rel_path
                .strip_prefix("X/")
                .unwrap()
                .strip_suffix(".shard")
                .unwrap();
            let idx: u32 = idx_str.parse().ok()?;
            Some((format!("X_shard_{idx}"), SectionType::CsrShard))
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
    }

    #[test]
    fn test_section_name_to_path_rejects_invalid_shard_index() {
        assert!(section_name_to_path("X_shard_abc", SectionType::CsrShard).is_err());
        assert!(section_name_to_path("X_shard_", SectionType::CsrShard).is_err());
        assert!(section_name_to_path("bad_name", SectionType::CsrShard).is_err());
        assert!(section_name_to_path("raw_counts_shard_xyz", SectionType::LayerCsrShard).is_err());
        assert!(section_name_to_path("obsp/dist_shard_xyz", SectionType::ObspCsrShard).is_err());
    }

    #[test]
    fn test_path_to_section_name_roundtrip() {
        let cases: Vec<(&str, SectionType)> = vec![
            ("obs", SectionType::ObsMetadata),
            ("var", SectionType::VarMetadata),
            ("X_shard_0", SectionType::CsrShard),
            ("X_shard_42", SectionType::CsrShard),
            ("obsm/X_pca", SectionType::ObsmEmbedding),
            ("uns", SectionType::UnsBlob),
            ("provenance", SectionType::Provenance),
            ("deletion_vectors", SectionType::DeletionVectors),
            ("obs_predicate_index", SectionType::ObsPredicateIndex),
            ("var_predicate_index", SectionType::VarPredicateIndex),
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
