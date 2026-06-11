//! `scx optimize` — in-place upgrade of an existing single-modality SCX file:
//! re-encode every CSR shard (canonicalizing it) so the output carries decode
//! sidecars and legitimately claims the v3 canonical-CSR invariant, while
//! preserving the row layout, obs/var, obsm/varm/obsp/varp, uns, and predicate
//! indexes.
//!
//! Unlike `compact` (which targets deletion reclaim + re-sharding and keeps the
//! source `format_version`), `optimize` is a faithful 1:1 upgrade: it does not
//! apply deletions (the deletion-vector section is carried through unchanged),
//! does not change shard boundaries, and canonicalizes each shard so the output
//! is a real v3 file with sidecars. The CSC sidecar is dropped (rerun
//! `scx build-csc` / `--rebuild-csc`); a `decode/*` sidecar is emitted for every
//! Scx1 integer shard automatically by the encoder.

use std::path::Path;

use scx_codec::CodecId;
use scx_format_io::encoder::encode_one_shard;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;
use scx_sparse::canonicalize_csr;

use crate::error::{OpsError, Result};
use crate::rewrite_helpers::{append_provenance, copy_predicate_indices};

/// Re-encode + canonicalize every CSR shard of `input_path` into `output_path`,
/// emitting decode sidecars and stamping `format_version = 3`. Single-modality
/// only (multimodal files should use `scx compact`).
///
/// `codec` selects the per-shard codec passed to the encoder: `None` keeps the
/// auto-codec (Scx1 for low-median integer counts, else Zstd), while
/// `Some(CodecId::Scx1)` forces Scx1 on every integer shard — guaranteeing a
/// `decode/*` sidecar (and thus the `to_gpu_anndata` device-decode route) even
/// for high-median shards that auto would route to Zstd. Non-integer shards
/// fall back to Zstd regardless (handled inside `encode_one_shard`).
pub fn optimize(input_path: &Path, output_path: &Path, codec: Option<CodecId>) -> Result<()> {
    let reader = ScxReader::open(input_path)?;
    if reader.is_multimodal() {
        return Err(OpsError::InvalidInput(
            "scx optimize does not support multimodal files yet; use `scx compact`".into(),
        ));
    }

    let in_header = reader.header().clone();
    let index_dtype = in_header.index_dtype;

    // The CSC sidecar (bit 0) is dropped — re-canonicalizing may change nnz and
    // would leave the column-major sidecar referencing stale offsets. The
    // deletion-vector flag (bit 5) is cleared here and re-set below only if we
    // actually carry the `DeletionVectors` section through (otherwise the header
    // would claim deletions with no section, and the logically-deleted rows
    // would silently reappear).
    let had_csc = in_header.has_csc();
    if had_csc {
        log::warn!(
            "scx optimize dropped CSC shards from {}: rerun `scx build-csc` \
             to restore the column-major sidecar",
            input_path.display()
        );
    }
    let out_flags = in_header.flags & !(1 << 0) & !(1 << 5);

    // We canonicalize every shard below, so the default v3 invariant is real.
    let out_header = FileHeader {
        flags: out_flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        shard_target_rows: in_header.shard_target_rows,
        // File-level codec hint for `scx info`. When the caller forces a codec,
        // reflect it; otherwise preserve the source hint (the real per-shard
        // codec is auto-selected by `encode_one_shard`).
        codec_id: codec.map(|c| c as u8).unwrap_or(in_header.codec_id),
        index_dtype,
        ..Default::default()
    };

    let mut writer = ScxWriter::new(output_path, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);

    // obs / var pass through unchanged (rows are preserved 1:1).
    writer.write_obs(&reader.read_obs()?)?;
    writer.write_var(&reader.read_var()?)?;

    // Re-encode every CSR-backed shard (X first, then layers, then obs×obs
    // pairwise graphs) preserving its name and global row offset, canonicalizing
    // the triplet so the output is canonical and the encoder emits a decode
    // sidecar for Scx1 integer shards. `ObspCsrShard` is X-class CSR (minor axis
    // is `n_obs`, not `n_vars`), so it is re-encoded here too — dropping it would
    // silently lose the pairwise graph.
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
    let mut obsp_csr_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::ObspCsrShard)
        .collect();
    obsp_csr_entries.sort_by(|a, b| a.name.cmp(&b.name));

    for entry in x_entries
        .into_iter()
        .chain(layer_entries)
        .chain(obsp_csr_entries)
    {
        let row_start = entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        // Use the shard's own minor dimension rather than the file-level
        // `n_vars`: correct for `ObspCsrShard` (minor axis = `n_obs`) and robust
        // against any per-shard width difference.
        let n_minor = reader.read_shard_header(entry)?.n_minor;
        let (indptr_i64, indices_i32, mut values) = reader.read_shard_from_entry(entry)?;
        let mut indptr: Vec<u64> = indptr_i64.iter().map(|&v| v as u64).collect();
        let mut indices: Vec<u32> = indices_i32.iter().map(|&v| v as u32).collect();
        canonicalize_csr(&mut indptr, &mut indices, &mut values);

        let pre = encode_one_shard(
            &indptr,
            &indices,
            &values,
            codec, // None = auto-codec; Some(Scx1) forces sidecars on every integer shard
            index_dtype,
            n_minor,
            row_start,
            entry.section_type,
            ModalityType::Rna,
            entry.name.clone(),
        )?;
        writer.write_preencoded_shard(pre)?;
    }

    // Auxiliary matrices (obsm/varm + COO obsp/varp) are unchanged by optimize,
    // so they are copied byte-for-byte. Verbatim copy preserves any sharded
    // layout (e.g. `ObsmEmbeddingShard`) and avoids decoding large `obsp`/`varp`
    // graphs into one section (OOM / the 2 GB Arrow IPC ceiling). `ObspCsrShard`
    // is excluded — it is the CSR-backed obsp graph, re-encoded in the loop
    // above. Sorted by name for deterministic output.
    let mut aux_entries: Vec<_> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            matches!(
                e.section_type,
                SectionType::ObsmEmbedding
                    | SectionType::ObsmEmbeddingShard
                    | SectionType::VarmEmbedding
                    | SectionType::VarmEmbeddingShard
                    | SectionType::ObspEmbedding
                    | SectionType::ObspEmbeddingShard
                    | SectionType::VarpEmbedding
                    | SectionType::VarpEmbeddingShard
            )
        })
        .collect();
    aux_entries.sort_by(|a, b| a.name.cmp(&b.name));
    for entry in aux_entries {
        let bytes = reader.section_bytes(entry)?;
        writer.copy_section_verbatim(entry, bytes)?;
    }
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }

    // Deletion vectors are carried through unchanged (optimize does not apply
    // them — that is `compact`'s job). The section references obs row ranges /
    // shard ids, both preserved here, so it stays valid; `write_deletion_vectors`
    // re-sets the header flag (cleared above) so it is set iff the section exists.
    if let Some(dv) = reader.read_deletion_vectors()? {
        writer.write_deletion_vectors(&dv)?;
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

        optimize(&input, &output, None).unwrap();

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

    #[test]
    fn optimize_codec_scx1_forces_sidecar_on_high_median_shard() {
        // A high-median integer shard (all counts == 100) routes to Zstd under
        // the auto-codec → NO decode sidecar. `Some(Scx1)` must force Scx1 and
        // emit the sidecar, which `to_gpu_anndata`'s device-decode route needs.
        // Dense rows (512 nnz, moderate gaps, 200k vars) so the Scx1 decode
        // sidecar fits the 25% overhead budget — matching the dims the
        // auto-codec test uses; only the value magnitude differs.
        let n_obs = 40usize;
        let nnz_per_row = 512usize;
        let n_vars = 200_000usize;

        let build_input = |path: &Path| {
            let mut header = sample_header(n_obs as u64, n_vars as u64);
            header.format_version = 2;
            header.index_dtype = 1; // u32 indices
            let mut indptr = vec![0u64];
            let mut indices: Vec<u32> = Vec::new();
            let mut values: Vec<u8> = Vec::new();
            for r in 0..n_obs {
                let mut col = 0u32;
                for k in 0..nnz_per_row {
                    col += 1 + ((r * 13 + k * 7) % 250) as u32;
                    indices.push(col);
                    values.push(100u8); // median == 100 (> 8) → auto picks Zstd
                }
                indptr.push(indices.len() as u64);
            }
            let mut w = ScxWriter::new(path, header).unwrap();
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
        };

        let count_sidecars = |path: &Path| {
            ScxReader::open(path)
                .unwrap()
                .catalog()
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
                .count()
        };

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        build_input(&input);

        // Auto-codec: high-median shard → Zstd → no sidecar.
        let auto_out = dir.path().join("auto.scx");
        optimize(&input, &auto_out, None).unwrap();
        assert_eq!(
            count_sidecars(&auto_out),
            0,
            "auto-codec leaves the high-median shard sidecar-less"
        );

        // Forced Scx1: sidecar present + decode-parity holds.
        let scx1_out = dir.path().join("scx1.scx");
        optimize(&input, &scx1_out, Some(CodecId::Scx1)).unwrap();
        let out = ScxReader::open(&scx1_out).unwrap();
        let sidecars: Vec<_> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::DecodeMetadataShard)
            .cloned()
            .collect();
        assert_eq!(sidecars.len(), 1, "forced Scx1 emits a decode sidecar");
        for e in &sidecars {
            out.validate_decode_sidecar_entry(e).unwrap();
        }
        // The decoded matrix is unchanged by the codec choice.
        let (a_indptr, a_indices, a_values) = ScxReader::open(&auto_out)
            .unwrap()
            .read_csr_shard(0)
            .unwrap();
        let (s_indptr, s_indices, s_values) = out.read_csr_shard(0).unwrap();
        assert_eq!(a_indptr, s_indptr);
        assert_eq!(a_indices, s_indices);
        assert_eq!(a_values, s_values);
    }

    /// Build a small canonical Scx1-eligible CSR triplet (strictly-increasing
    /// per-row indices, small positive counts) with `n_obs` rows.
    fn small_csr(n_obs: usize) -> (Vec<u64>, Vec<u32>, Vec<u8>) {
        let mut indptr = vec![0u64];
        let mut indices: Vec<u32> = Vec::new();
        let mut values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let mut col = 0u32;
            for k in 0..16usize {
                col += 1 + ((r + k) % 7) as u32;
                indices.push(col);
                values.push(1u8 + ((r + k) % 4) as u8);
            }
            indptr.push(indices.len() as u64);
        }
        (indptr, indices, values)
    }

    #[test]
    fn optimize_preserves_deletion_vectors() {
        use roaring::RoaringBitmap;
        use scx_format_io::deletion_vectors::DeletionVectors;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 8usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
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
            // Mark local row 1 of shard 0 as logically deleted.
            let mut dv = DeletionVectors::new();
            let mut bm = RoaringBitmap::new();
            bm.insert(1);
            dv.insert(0, bm);
            w.write_deletion_vectors(&dv).unwrap();
            w.finish().unwrap();
        }

        optimize(&input, &output, None).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(out.header().format_version, 3);
        // The deletion-vector section is carried through (flag re-set + section
        // present), not dropped — so the deleted row is not silently resurrected.
        assert!(
            out.header().has_deletion_vectors(),
            "deletion-vector flag preserved"
        );
        let dv = out
            .read_deletion_vectors()
            .unwrap()
            .expect("deletion-vector section present after optimize");
        assert!(dv.is_deleted(0, 1));
        assert_eq!(dv.total_deleted(), 1);
        // optimize does NOT apply deletions — rows stay (logical deletion).
        assert_eq!(out.read_obs().unwrap().num_rows(), n_obs);
        let (out_indptr, _, _) = out.read_csr_shard(0).unwrap();
        assert_eq!(out_indptr.len(), n_obs + 1);
    }

    #[test]
    fn optimize_preserves_and_canonicalizes_obsp_csr_shard() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 8usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
        // Canonical obs×obs graph: each row links to two distinct, sorted
        // neighbors (indices < n_obs).
        let mut op_indptr = vec![0u64];
        let mut op_indices: Vec<u32> = Vec::new();
        let mut op_values: Vec<u8> = Vec::new();
        for r in 0..n_obs {
            let a = ((r + 1) % n_obs) as u32;
            let b = ((r + 3) % n_obs) as u32;
            let (lo, hi) = if a < b { (a, b) } else { (b, a) };
            op_indices.push(lo);
            op_values.push(1u8);
            op_indices.push(hi);
            op_values.push(1u8);
            op_indptr.push(op_indices.len() as u64);
        }
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
            w.write_obsp_shard(
                &op_indptr,
                &op_indices,
                &op_values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
                "connectivities",
                0,
            )
            .unwrap();
            w.finish().unwrap();
        }

        optimize(&input, &output, None).unwrap();

        let out = ScxReader::open(&output).unwrap();
        // The CSR-backed obsp graph survives (it would be silently dropped if
        // the re-encode loop only handled CsrShard/LayerCsrShard).
        let obsp_shards: Vec<_> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObspCsrShard)
            .collect();
        assert_eq!(obsp_shards.len(), 1, "obsp CSR shard preserved");
        // Every CSR-class shard (X + obsp) is canonical post-optimize.
        assert!(out
            .validate_canonical_csr_shards()
            .iter()
            .all(|(_, ok)| *ok));
    }

    #[test]
    fn optimize_copies_sharded_obsm_verbatim() {
        use arrow::array::{Float32Array, RecordBatch};
        use arrow::datatypes::{DataType, Field, Schema};
        use std::sync::Arc;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.scx");
        let output = dir.path().join("out.scx");

        let n_obs = 6usize;
        let n_vars = 1000usize;
        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 2;
        header.index_dtype = 1;
        let (indptr, indices, values) = small_csr(n_obs);
        let schema = Arc::new(Schema::new(vec![
            Field::new("pc1", DataType::Float32, false),
            Field::new("pc2", DataType::Float32, false),
        ]));
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
            // Two 3-row obsm shards (sharded layout to prove it is preserved).
            for shard_idx in 0u32..2 {
                let row_start = shard_idx as usize * 3;
                let rows: Vec<f32> = (row_start..row_start + 3).map(|r| r as f32).collect();
                let rows2: Vec<f32> = rows.iter().map(|r| r * 2.0).collect();
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Float32Array::from(rows)),
                        Arc::new(Float32Array::from(rows2)),
                    ],
                )
                .unwrap();
                w.write_obsm_shard(
                    "X_pca",
                    shard_idx,
                    row_start as u64,
                    batch.num_rows() as u64,
                    n_obs as u64,
                    &batch,
                )
                .unwrap();
            }
            w.finish().unwrap();
        }

        let in_reader = ScxReader::open(&input).unwrap();
        let in_shards: Vec<(String, Vec<u8>)> = in_reader
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsmEmbeddingShard)
            .map(|e| (e.name.clone(), in_reader.section_bytes(e).unwrap().to_vec()))
            .collect();
        assert_eq!(in_shards.len(), 2);
        drop(in_reader);

        optimize(&input, &output, None).unwrap();

        let out = ScxReader::open(&output).unwrap();
        let out_shards: Vec<(String, Vec<u8>)> = out
            .catalog()
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsmEmbeddingShard)
            .map(|e| (e.name.clone(), out.section_bytes(e).unwrap().to_vec()))
            .collect();
        // Sharded layout preserved (not flattened) and copied byte-for-byte.
        assert_eq!(out_shards.len(), 2, "sharded obsm preserved as shards");
        assert_eq!(in_shards, out_shards, "obsm shards copied verbatim");
        assert!(
            !out.catalog()
                .entries
                .iter()
                .any(|e| e.section_type == SectionType::ObsmEmbedding),
            "no flattened single-section obsm emitted"
        );
    }
}
