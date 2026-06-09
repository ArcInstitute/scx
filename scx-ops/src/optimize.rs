//! `scx optimize` — in-place upgrade of an existing single-modality SCX file:
//! re-encode every CSR shard (canonicalizing it) so the output carries decode
//! sidecars and legitimately claims the v3 canonical-CSR invariant, while
//! preserving the row layout, obs/var, obsm/varm/obsp/varp, uns, and predicate
//! indexes.
//!
//! Unlike `compact` (which targets deletion reclaim + re-sharding and keeps the
//! source `format_version`), `optimize` is a faithful 1:1 upgrade: it does not
//! apply deletions, does not change shard boundaries, and canonicalizes each
//! shard so the output is a real v3 file with sidecars. The CSC sidecar is
//! dropped (rerun `scx build-csc` / `--rebuild-csc`); a `decode/*` sidecar is
//! emitted for every Scx1 integer shard automatically by the encoder.

use std::path::Path;

use scx_format::encoder::encode_one_shard;
use scx_format::header::{FileHeader, CURRENT_FORMAT_VERSION, MAGIC};
use scx_format::modality::ModalityType;
use scx_format::section::SectionType;
use scx_format::writer::ScxWriter;
use scx_format::ScxReader;
use scx_sparse::canonicalize_csr;

use crate::error::{OpsError, Result};
use crate::rewrite_helpers::{append_provenance, copy_predicate_indices};

/// Re-encode + canonicalize every CSR shard of `input_path` into `output_path`,
/// emitting decode sidecars and stamping `format_version = 3`. Single-modality
/// only (multimodal files should use `scx compact`).
pub fn optimize(input_path: &Path, output_path: &Path) -> Result<()> {
    let reader = ScxReader::open(input_path)?;
    if reader.is_multimodal() {
        return Err(OpsError::InvalidInput(
            "scx optimize does not support multimodal files yet; use `scx compact`".into(),
        ));
    }

    let in_header = reader.header().clone();
    let index_dtype = in_header.index_dtype;
    let n_vars = u32::try_from(in_header.n_vars)
        .map_err(|_| OpsError::InvalidInput("n_vars exceeds u32".into()))?;

    // The CSC sidecar (bit 0) is dropped — re-canonicalizing may change nnz and
    // would leave the column-major sidecar referencing stale offsets. The
    // deletion-vector flag (bit 5) is preserved: optimize keeps every row.
    let had_csc = in_header.has_csc();
    if had_csc {
        log::warn!(
            "scx optimize dropped CSC shards from {}: rerun `scx build-csc` \
             to restore the column-major sidecar",
            input_path.display()
        );
    }
    let out_flags = in_header.flags & !(1 << 0);

    let out_header = FileHeader {
        magic: MAGIC,
        // We canonicalize every shard below, so the v3 invariant is real.
        format_version: CURRENT_FORMAT_VERSION,
        header_length: 256,
        flags: out_flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        nnz: 0, // set by finish
        n_csr_shards: 0,
        n_csc_shards: 0,
        shard_target_rows: in_header.shard_target_rows,
        // File-level codec hint; the real per-shard codec is auto-selected by
        // `encode_one_shard`. Preserve the source hint for `scx info`.
        codec_id: in_header.codec_id,
        index_dtype,
        endian: 0,
        reserved_padding: 0,
        root_catalog_offset: 0,
        root_catalog_length: 0,
        full_catalog_offset: 0,
        full_catalog_length: 0,
        manifest_sequence: 0,
        prev_catalog_offset: 0,
        file_checksum: 0,
        front_catalog_offset: 0,
        front_catalog_length: 0,
        n_modalities: 0,
        modality_table_offset: 0,
        modality_table_length: 0,
        reserved: [0u8; 112],
    };

    let mut writer = ScxWriter::new(output_path, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);

    // obs / var pass through unchanged (rows are preserved 1:1).
    writer.write_obs(&reader.read_obs()?)?;
    writer.write_var(&reader.read_var()?)?;

    // Re-encode every CSR shard (X first, then layers) preserving its name and
    // global row offset, canonicalizing the triplet so the output is canonical
    // and the encoder emits a decode sidecar for Scx1 integer shards.
    let mut x_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .collect();
    x_entries.sort_by_key(|e| e.stats.as_ref().map(|s| s.row_start).unwrap_or(0));
    let mut layer_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::LayerCsrShard)
        .collect();
    layer_entries.sort_by(|a, b| a.name.cmp(&b.name));

    for entry in x_entries.into_iter().chain(layer_entries.into_iter()) {
        let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        let (indptr_i64, indices_i32, values) = reader.read_shard_from_entry(entry)?;
        let mut indptr: Vec<u64> = indptr_i64.iter().map(|&v| v as u64).collect();
        let mut indices: Vec<u32> = indices_i32.iter().map(|&v| v as u32).collect();
        let mut values = values;
        canonicalize_csr(&mut indptr, &mut indices, &mut values);

        let pre = encode_one_shard(
            &indptr,
            &indices,
            &values,
            None, // auto-codec (Scx1 for low-median integer counts → emits sidecar)
            index_dtype,
            n_vars,
            row_start,
            entry.section_type,
            ModalityType::Rna,
            entry.name.clone(),
        )?;
        writer.write_preencoded_shard(pre)?;
    }

    // Auxiliary matrices pass through unchanged (rows/cols preserved). Sorted
    // for deterministic output.
    let mut obsm: Vec<_> = reader.read_all_obsm()?.into_iter().collect();
    obsm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsm {
        writer.write_obsm(name, batch)?;
    }
    let mut varm: Vec<_> = reader.read_all_varm()?.into_iter().collect();
    varm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varm {
        writer.write_varm(name, batch)?;
    }
    let mut obsp: Vec<_> = reader.read_all_obsp()?.into_iter().collect();
    obsp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsp {
        writer.write_obsp(name, batch)?;
    }
    let mut varp: Vec<_> = reader.read_all_varp()?.into_iter().collect();
    varp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varp {
        writer.write_varp(name, batch)?;
    }
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Predicate indexes reference obs row ranges; rows + shard boundaries are
    // preserved, so they stay valid — copy through. Then record provenance.
    copy_predicate_indices(&reader, &mut writer)
        .map_err(|e| OpsError::InvalidInput(format!("copy predicate indices: {e}")))?;
    append_provenance(
        &reader,
        &mut writer,
        "optimize",
        "{\"canonicalize\":true,\"sidecars\":true}",
    )
    .map_err(|e| OpsError::InvalidInput(format!("append provenance: {e}")))?;

    writer.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{sample_header, sample_obs, sample_var};
    use scx_codec::{CodecId, ValueEncoding};

    #[test]
    fn optimize_adds_sidecars_canonicalizes_and_upgrades_to_v3() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        // v2 input with a `None`-codec CSR shard (no decode sidecar). Rows are
        // dense (512 nnz) with moderate column gaps so the re-encoded Scx1
        // sidecar comfortably fits the 25% overhead budget; row 0 has ONE
        // duplicate column (non-canonical). u32 indices (n_vars > 65535).
        let n_obs = 40usize;
        let nnz_per_row = 512usize;
        let n_vars = 200_000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1; // u32 indices

        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let mut col = 0u32;
            for k in 0..nnz_per_row {
                if r == 0 && k == 1 {
                    // Duplicate the previous column → one non-canonical entry.
                    indices.push(*indices.last().unwrap());
                } else {
                    col += 1 + ((r * 13 + k * 7) % 250) as u32; // gaps → frame_bits ~8
                    indices.push(col);
                }
                values.push(1u8 + ((r + k) % 5) as u8); // small, non-zero → Scx1
            }
            indptr.push(indices.len() as u64);
        }
        let input_nnz = indices.len();
        {
            let mut w = ScxWriter::new(&input, header).unwrap();
            w.write_obs(&sample_obs(n_obs)).unwrap();
            w.write_var(&sample_var(n_vars)).unwrap();
            w.write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        // Input carries no decode sidecar (None codec).
        let in_reader = ScxReader::open(&input).unwrap();
        assert!(!in_reader
            .catalog()
            .entries
            .iter()
            .any(|e| e.section_type == SectionType::DecodeMetadataShard));

        optimize(&input, &output).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(out.header().format_version, 3, "optimize stamps v3");

        // A decode sidecar is emitted for the re-encoded (now Scx1) shard.
        let sidecars: Vec<_> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
            .cloned()
            .collect();
        assert_eq!(sidecars.len(), 1, "decode sidecar added");
        // Sidecar decode-parity + source identity (the deep-validate check).
        for e in &sidecars {
            out.validate_decode_sidecar_entry(e).unwrap();
        }
        // Every CSR shard is canonical post-optimize.
        assert!(out
            .validate_canonical_csr_shards()
            .iter()
            .all(|(_, ok)| *ok));

        // Canonicalization summed the single duplicate → nnz drops by exactly 1.
        let (out_indptr, out_indices, _values) = out.read_csr_shard(0).unwrap();
        assert_eq!(out_indices.len(), input_nnz - 1);
        assert_eq!(*out_indptr.last().unwrap() as usize, input_nnz - 1);
        assert_eq!(out_indptr.len(), n_obs + 1);

        // obs/var preserved.
        assert_eq!(out.read_obs().unwrap().num_rows(), n_obs);
        assert_eq!(out.read_var().unwrap().num_rows(), n_vars);
    }
}
