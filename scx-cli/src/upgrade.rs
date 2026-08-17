// scx upgrade — Upgrade an SCX file to the latest UNFRAMED format version (v3).
//
// Rewrites the file fully through the current ScxWriter, which produces the
// newest unframed version (DEFAULT_WRITE_FORMAT_VERSION). It intentionally does
// NOT add row-group framing, so it does not reach CURRENT_FORMAT_VERSION (v4) —
// use `scx optimize --row-group-rows N` for the framed v4 layout. If the file
// is already at the unframed target — or already *past* it, at v4 — this is a
// no-op. See SCX-013.

use std::path::Path;

use scx_format_io::header::{FileHeader, DEFAULT_WRITE_FORMAT_VERSION};
use scx_format_io::reader::ScxReader;
use scx_format_io::writer::ScxWriter;

pub fn run_upgrade(
    input: &Path,
    output: Option<&Path>,
    in_place: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Validate: either output or --in-place required
    if output.is_none() && !in_place {
        return Err("Specify an output path or use --in-place".into());
    }

    // 2. Open and read existing file
    let reader = ScxReader::open(input)?;
    let old_version = reader.header().format_version;

    // `scx upgrade` re-writes to the newest **unframed** version (v4 requires
    // row-group framing, which upgrade does not add).
    //
    // A file *newer* than that target is declined rather than rewritten. The
    // rewrite is not a no-op on such a file: it re-emits every shard unframed
    // and stamps v3, so a v4 input silently loses the row-group framing — and
    // with it the sub-shard random access v4 exists to provide — while printing
    // "Upgraded ... v4 -> v3". On `--in-place` that is unrecoverable: the rename
    // lands a wholly new file carrying no prior catalog, so `scx rollback` has
    // nothing to roll back to. There is deliberately no `--allow-downgrade`:
    // nothing asks for one, and `scx optimize` is the op that owns framing in
    // both directions.
    if old_version > DEFAULT_WRITE_FORMAT_VERSION {
        // The rollback clause is true only of the rename path; on the copy-out
        // form the input is never touched, so saying it there would be a wrong
        // rationale for a right refusal.
        let irreversible = if in_place {
            " and `--in-place` is not rollback-able"
        } else {
            ""
        };
        println!(
            "File is at format version {old_version}, which is newer than the version \
             `scx upgrade` targets (v{DEFAULT_WRITE_FORMAT_VERSION}, unframed). Rewriting \
             it here would strip row-group framing and its sub-shard random access{irreversible}. \
             Nothing to do. Use `scx optimize --row-group-rows N` to re-frame, or \
             `scx optimize --row-group-rows 0` if you genuinely need an unframed v3 \
             file for an older reader."
        );
        return Ok(());
    }
    if old_version == DEFAULT_WRITE_FORMAT_VERSION {
        println!(
            "File is already at format version {} (newest unframed version; \
             use `scx optimize --row-group-rows N` for framed v4). Nothing to do.",
            DEFAULT_WRITE_FORMAT_VERSION
        );
        return Ok(());
    }

    // Only now: nothing here is modality-aware, so a multimodal file that would
    // actually be rewritten must be refused. `csr_shards_sorted()` returns every
    // modality's shards, the loop below writes them through the single-modality
    // `write_csr_shard`, and the output header's `..Default::default()` zeroes
    // `n_modalities` and the modality-table pointers — a CITE-seq file would come
    // out nominally single-modality with overlapping global row ranges and no
    // per-modality var, on `--in-place` over the top of the original. Multimodal
    // only requires v2, so it reaches here.
    //
    // **After** the version gates, deliberately. Placing it first also turned a
    // multimodal *v3 or v4* file — which is never rewritten and was a harmless
    // exit-0 "nothing to do" — into a hard error, contradicting this module's
    // own contract and `docs/api.md`. Refuse what would be damaged, not what
    // would be left alone.
    if reader.is_multimodal() {
        return Err(format!(
            "scx upgrade does not support multimodal files yet: {} carries a modality \
             table, and this rewrite would flatten it into a single-modality file. \
             Extract a modality with `scx subset {} out.scx --modality NAME`, or use \
             `scx compact`, which handles multimodal inputs.",
            input.display(),
            input.display()
        )
        .into());
    }

    // 3. Determine output path
    let output_path = if in_place {
        // Write to temp file, then atomic rename
        let mut tmp = input.to_path_buf();
        tmp.set_extension("scx.upgrading");
        tmp
    } else {
        output.unwrap().to_path_buf()
    };

    // 4. Re-read and re-write using current writer
    let rewrite_result = rewrite_with_current_version(&reader, &output_path);
    if rewrite_result.is_err() && in_place {
        // Clean up orphaned temp file on failure
        let _ = std::fs::remove_file(&output_path);
    }
    rewrite_result?;

    let new_reader = ScxReader::open(&output_path)?;
    let new_version = new_reader.header().format_version;
    drop(new_reader);

    // 5. If in-place, atomic rename
    if in_place {
        std::fs::rename(&output_path, input)?;
        println!(
            "Upgraded {} from v{} \u{2192} v{} (in-place)",
            input.display(),
            old_version,
            new_version
        );
    } else {
        println!(
            "Upgraded {} \u{2192} {} (v{} \u{2192} v{})",
            input.display(),
            output_path.display(),
            old_version,
            new_version
        );
    }

    Ok(())
}

/// Re-read all sections from the reader and re-write them using the current
/// ScxWriter, which produces the latest format_version.
fn rewrite_with_current_version(
    reader: &ScxReader,
    output: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let in_header = reader.header();

    let csr_entries = reader.catalog().csr_shards_sorted();

    // Set up output header (canonical v3 upgrade: bump manifest, preserve
    // flags/codec/index dtype from the source; writer fills nnz + shard counts).
    let out_header = FileHeader {
        flags: in_header.flags,
        n_obs: in_header.n_obs,
        n_vars: in_header.n_vars,
        shard_target_rows: in_header.shard_target_rows,
        codec_id: in_header.codec_id,
        index_dtype: in_header.index_dtype,
        manifest_sequence: in_header.manifest_sequence + 1,
        ..Default::default()
    };

    // Create writer
    let mut writer = ScxWriter::new(output, out_header)?;

    // obs/var pass through 1:1, so a row-sharded input must come out
    // row-sharded rather than collapsed into one legacy section — see
    // `copy_obs_var_preserving_layout` for what that costs. An upgrade never
    // drops rows, which is also what licenses the deletion-vector carry in
    // `copy_auxiliary_sections` below.
    scx_ops::copy_obs_var_preserving_layout(reader, &mut writer)?;

    // Re-write CSR shards (per-shard codec), canonicalizing each one.
    //
    // The output header stamps `DEFAULT_WRITE_FORMAT_VERSION`, and v3's contract
    // *is* canonical row-major CSR: indices sorted within each row, duplicate
    // coordinates summed, no explicit zeros. A pre-v3 source carries no such
    // guarantee — that is exactly why `rewrite_output_format_version` refuses to
    // promote one — so re-emitting its rows unchanged under a v3 header would
    // make the file claim an invariant it does not hold, and `scx validate
    // --deep` would rightly reject it.
    //
    // The three rewriting ops split on this and only two got it right:
    // `optimize` canonicalizes and so legitimately stamps v3; `build_csc` does
    // not canonicalize and so clamps its output version to the source's
    // (SCX-005). `upgrade` did neither — it stamped v3 over whatever it was
    // handed. Clamping is not an option here, because producing v3 is the whole
    // point of the command, so it canonicalizes, like `optimize`.
    // Set when canonicalization actually changed a shard — see the CSC block.
    let mut csr_was_rewritten = false;
    for shard_entry in &csr_entries {
        let sh = reader.read_shard_header(shard_entry)?;
        let ve = crate::shard_utils::decode_value_encoding(sh.value_encoding)?;
        let ci = crate::shard_utils::decode_codec_id(sh.codec_id)?;

        let (indptr, indices, mut data) = reader.read_shard_from_entry(shard_entry)?;
        let shard_row_start = shard_entry.stats.as_ref().map(|s| s.row_start).unwrap_or(0);

        let mut indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let mut indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
        // Canonicalizing sums duplicate coordinates, so a value can outgrow the
        // width the source chose for the values it held. Re-encoding through the
        // old width either refuses outright (`Uint8` rejects 200 + 200 = 400,
        // failing on exactly the input this op exists to repair) or, for
        // `Float16`, silently yields infinity. Widen to what the result needs.
        let (ve, ci) = if !scx_sparse::is_canonical_csr(&indptr_u64, &indices_u32, &data) {
            scx_sparse::canonicalize_csr(&mut indptr_u64, &mut indices_u32, &mut data);
            csr_was_rewritten = true;
            let widened = scx_ops::encoding_for_canonicalized(ve, &data);
            // Widening can land on `Float32`, which `Scx1` cannot encode;
            // `write_csr_shard` passes the codec through unchanged, so the
            // downgrade has to happen here or the upgrade dies FloatWithScx1.
            (widened, scx_ops::codec_for_canonicalized(ci, widened))
        } else {
            (ve, ci)
        };
        let raw_values = ve.encode_f32_batch(&data)?;

        writer.write_csr_shard(
            &indptr_u64,
            &indices_u32,
            &raw_values,
            ci,
            ve,
            shard_row_start,
        )?;
    }

    // Re-write CSC shards (if present, per-shard codec). Sorted by
    // major_start() (= col_start for CSC entries via the axis-overload
    // in ShardStats; on-disk fields are unchanged).
    //
    // Unless canonicalizing above actually rewrote a shard. The sidecar is a
    // second representation of the same matrix, built against the *old* CSR:
    // summing a duplicate coordinate or dropping an explicit zero changes X's
    // structure and its nnz, so carrying the old CSC forward leaves the file
    // holding two matrices that disagree. That is not a stale-sidecar
    // annoyance — `write_csc_shard` stamps `csc_build_generation` from the
    // writer's `data_generation`, so the freshness guard would bless it, and a
    // consumer on the CSC path (`prefer_format="csc"`, GPU CSC-direct DE) would
    // silently read different numbers than one on CSR. `optimize` faces the
    // same choice and drops unconditionally.
    //
    // Dropped only when it would actually be wrong, because carrying CSC is the
    // one thing `upgrade` does that its siblings don't, and an already-canonical
    // input — every file any current writer produces — keeps it.
    let csc_entries = reader.catalog().csc_shards_sorted();
    if csr_was_rewritten && !csc_entries.is_empty() {
        log::warn!(
            "scx upgrade: canonicalizing the CSR matrix changed its structure, so the \
             CSC sidecar built against the old matrix is no longer a faithful second \
             view of it and has been dropped. Rerun `scx build-csc` to rebuild the \
             column-major sidecar against the upgraded file."
        );
    }
    let csc_entries: Vec<_> = if csr_was_rewritten {
        Vec::new()
    } else {
        csc_entries
    };

    for csc_entry in &csc_entries {
        let sh = reader.read_shard_header(csc_entry)?;
        let ve = crate::shard_utils::decode_value_encoding(sh.value_encoding)?;
        let ci = crate::shard_utils::decode_codec_id(sh.codec_id)?;

        let (indptr, indices, data) = reader.read_shard_from_entry(csc_entry)?;
        let col_start = csc_entry
            .stats
            .as_ref()
            .map(|s| s.major_start(csc_entry.section_type))
            .unwrap_or(0);

        let indices_u32: Vec<u32> = indices.iter().map(|&i| i as u32).collect();
        let indptr_u64: Vec<u64> = indptr.iter().map(|&v| v as u64).collect();
        let raw_values = ve.encode_f32_batch(&data)?;

        writer.write_csc_shard(&indptr_u64, &indices_u32, &raw_values, ci, ve, col_start)?;
    }

    // Copy auxiliary sections (layers, obsm, uns, predicate indices, deletion
    // vectors, provenance). `canonicalize = true` for the same reason the X loop
    // above canonicalizes: a layer CSR shard re-emitted verbatim under a v3
    // header would carry the same false claim. It also warns about the section
    // families this helper does not carry, which matters here because
    // `--in-place` renames over the target with no prior catalog to roll back to.
    scx_ops::copy_auxiliary_sections_canonicalizing(reader, &mut writer, "upgrade", "{}", true)?;

    writer.finish()?;
    // `upgrade` is the one op in the carry table whose call site lives outside
    // `scx-ops`, so the audit is wired here rather than inside the shared helper
    // above — which cannot see the finished output, and is shared with
    // `build-csc`, whose policy differs on the CSC sidecar.
    scx_ops::carry::audit_output(
        scx_ops::carry::RewriteOp::Upgrade,
        &[reader.catalog()],
        output,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::write_test_file;
    use scx_codec::{CodecId, ValueEncoding};
    use std::path::PathBuf;

    /// A format upgrade is a 1:1 rewrite, so the deletion-vector section has to
    /// come with it. `rewrite_with_current_version` copies `in_header.flags`
    /// forward, which *looks* like it preserves the flag — but the writer
    /// re-derives every flag from the sections actually written, so without the
    /// carry the deletions silently vanish and the cells come back.
    #[test]
    fn upgrade_carries_deletion_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 8, 5);
        scx_ops::mark_deleted(&input, &[2, 6]).unwrap();
        let output = dir.path().join("upgraded.scx");

        let reader = ScxReader::open(&input).unwrap();
        rewrite_with_current_version(&reader, &output).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert!(
            out.header().has_deletion_vectors(),
            "deletion-vector flag must survive an upgrade"
        );
        let dv = out
            .read_deletion_vectors()
            .unwrap()
            .expect("deletion-vector section present after upgrade");
        assert_eq!(dv.total_deleted(), 2);
        assert!(dv.is_deleted_global(2) && dv.is_deleted_global(6));
        // Carried, not applied — the rows are still physically there.
        assert_eq!(out.n_obs(), 8);
        assert_eq!(out.read_all_csr_shards_filtered().unwrap().shape.0, 6);
    }

    #[test]
    fn test_upgrade_preserves_data() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 8, 5);
        let output = dir.path().join("upgraded.scx");

        // Current version is 1 and file is at version 1, so this should no-op.
        // To test data preservation, we call rewrite_with_current_version directly.
        let reader = ScxReader::open(&input).unwrap();
        rewrite_with_current_version(&reader, &output).unwrap();

        // Verify output data matches input
        let orig_reader = ScxReader::open(&input).unwrap();
        let new_reader = ScxReader::open(&output).unwrap();

        let orig_hdr = orig_reader.header();
        let new_hdr = new_reader.header();
        assert_eq!(new_hdr.n_obs, orig_hdr.n_obs);
        assert_eq!(new_hdr.n_vars, orig_hdr.n_vars);
        assert_eq!(
            new_hdr.format_version,
            scx_format_io::DEFAULT_WRITE_FORMAT_VERSION
        );

        // Verify CSR data matches
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let new_csr = new_reader.read_all_csr_shards().unwrap();
        assert_eq!(new_csr.shape, orig_csr.shape);
        assert_eq!(new_csr.indptr, orig_csr.indptr);
        assert_eq!(new_csr.indices, orig_csr.indices);
        assert_eq!(new_csr.data, orig_csr.data);

        // Verify obs metadata
        let orig_obs = orig_reader.read_obs().unwrap();
        let new_obs = new_reader.read_obs().unwrap();
        assert_eq!(new_obs.num_rows(), orig_obs.num_rows());
        assert_eq!(new_obs.num_columns(), orig_obs.num_columns());

        // Verify var metadata
        let orig_var = orig_reader.read_var().unwrap();
        let new_var = new_reader.read_var().unwrap();
        assert_eq!(new_var.num_rows(), orig_var.num_rows());
    }

    #[test]
    fn test_upgrade_noop_current() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 4, 3);
        let output = dir.path().join("upgraded.scx");

        // File is at version 1 (current), should no-op
        let result = run_upgrade(&input, Some(output.as_path()), false);
        assert!(result.is_ok());
        // Output should NOT have been created (no-op)
        assert!(!output.exists(), "no-op upgrade should not create output");
    }

    #[test]
    fn test_upgrade_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 6, 4);

        // Read original data for comparison
        let orig_reader = ScxReader::open(&input).unwrap();
        let orig_csr = orig_reader.read_all_csr_shards().unwrap();
        let orig_n_obs = orig_reader.header().n_obs;
        drop(orig_reader);

        // rewrite_with_current_version directly, then atomic rename to simulate
        // an in-place upgrade (since version is already 1, run_upgrade would no-op)
        let tmp_path = dir.path().join("test.scx.upgrading");
        let reader = ScxReader::open(&input).unwrap();
        rewrite_with_current_version(&reader, &tmp_path).unwrap();
        drop(reader);
        std::fs::rename(&tmp_path, &input).unwrap();

        // Verify data preserved after in-place rewrite
        let reader = ScxReader::open(&input).unwrap();
        assert_eq!(reader.header().n_obs, orig_n_obs);
        assert_eq!(
            reader.header().format_version,
            scx_format_io::DEFAULT_WRITE_FORMAT_VERSION
        );

        let csr = reader.read_all_csr_shards().unwrap();
        assert_eq!(csr.shape, orig_csr.shape);
        assert_eq!(csr.indptr, orig_csr.indptr);
        assert_eq!(csr.indices, orig_csr.indices);
        assert_eq!(csr.data, orig_csr.data);
    }

    #[test]
    fn test_upgrade_no_output_no_inplace_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_test_file(&dir, 4, 3);

        let err = run_upgrade(&input, None, false);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("--in-place"));
    }

    /// write a file with CSR + multi-shard CSC, run
    /// rewrite_with_current_version, and confirm the output preserves
    /// has_csc, the CSC shard count, per-shard column ranges, and
    /// densified contents.
    #[test]
    fn test_upgrade_preserves_csc_multi_shard() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use scx_format_io::section::SectionType;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("with_csc.scx");

        // 4 rows × 6 cols dense reference (column-by-column nnz).
        let n_rows = 4usize;
        let n_cols = 6usize;
        #[rustfmt::skip]
        let dense: Vec<f32> = vec![
            // col: 0    1    2    3    4    5
                   1.0, 0.0, 0.0, 4.0, 0.0, 7.0,
                   0.0, 2.0, 5.0, 0.0, 0.0, 8.0,
                   0.0, 0.0, 0.0, 0.0, 6.0, 0.0,
                   3.0, 0.0, 0.0, 0.0, 0.0, 9.0,
        ];

        // Build CSC arrays for a column range from the dense reference.
        let csc_arrays = |col_start: usize, col_end: usize| -> (Vec<u64>, Vec<u32>, Vec<u8>) {
            let mut indptr: Vec<u64> = vec![0];
            let mut indices: Vec<u32> = Vec::new();
            let mut values: Vec<u8> = Vec::new();
            for col in col_start..col_end {
                for row in 0..n_rows {
                    let v = dense[row * n_cols + col];
                    if v != 0.0 {
                        indices.push(row as u32);
                        values.push(v as u8);
                    }
                }
                indptr.push(indices.len() as u64);
            }
            (indptr, indices, values)
        };

        let header = sample_header(n_rows as u64, n_cols as u64);
        let mut writer = ScxWriter::new(&input, header).unwrap();
        writer.write_obs(&sample_obs(n_rows)).unwrap();
        writer.write_var(&sample_var(n_cols)).unwrap();

        // Empty CSR shard for file invariants.
        let csr_indptr = vec![0u64; n_rows + 1];
        writer
            .write_csr_shard(
                &csr_indptr,
                &[],
                &[],
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();

        // 3 CSC shards: cols [0..2), [2..4), [4..6) — non-uniform on
        // purpose so the col_range preservation is meaningful.
        for (cs, ce) in [(0usize, 2usize), (2, 4), (4, 6)] {
            let (ip, ix, vb) = csc_arrays(cs, ce);
            writer
                .write_csc_shard(
                    &ip,
                    &ix,
                    &vb,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    cs as u64,
                )
                .unwrap();
        }
        writer.finish().unwrap();

        // Sanity: input has CSC.
        let in_reader = ScxReader::open(&input).unwrap();
        assert!(in_reader.header().has_csc());
        assert_eq!(in_reader.header().n_csc_shards, 3);
        let in_csc = in_reader.read_all_csc_shards().unwrap();
        assert_eq!(in_csc.to_dense().unwrap(), dense);

        // Rewrite through current writer.
        let output = dir.path().join("upgraded.scx");
        rewrite_with_current_version(&in_reader, &output).unwrap();
        drop(in_reader);

        // Output preserves CSC: flag, count, per-shard col ranges, contents.
        let out_reader = ScxReader::open(&output).unwrap();
        assert!(
            out_reader.header().has_csc(),
            "has_csc must survive upgrade"
        );
        assert_eq!(out_reader.header().n_csc_shards, 3);

        let out_csc_entries = out_reader.catalog().csc_shards_sorted();
        assert_eq!(out_csc_entries.len(), 3);
        let ranges: Vec<std::ops::Range<u64>> = out_csc_entries
            .iter()
            .map(|e| e.stats.as_ref().unwrap().col_range())
            .collect();
        assert_eq!(ranges, vec![0..2, 2..4, 4..6]);

        // On-disk shard_type byte must remain `1` for re-emitted CSC
        // shards (invariant survives rewrite).
        let out_data = std::fs::read(&output).unwrap();
        for entry in &out_csc_entries {
            let section = &out_data[entry.offset as usize..][..entry.length as usize];
            let sh = scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
                &section[..scx_format_io::shard::SHARD_HEADER_SIZE],
            ))
            .unwrap();
            assert_eq!(sh.shard_type, 1, "CSC shard byte must be 1 after upgrade");
            assert!(sh.is_csc(SectionType::CscShard));
        }

        // Densify and compare.
        let out_csc = out_reader.read_all_csc_shards().unwrap();
        assert_eq!(out_csc.shape, (n_rows, n_cols));
        assert_eq!(out_csc.to_dense().unwrap(), dense);
    }

    /// A real v4 file, built the way users get one: framed shards, sharded obs.
    /// `optimize` is the op that produces the framed layout, and
    /// `ObsShardPolicy::Always` gives us the sharded obs at test scale.
    fn write_framed_v4_file(dir: &tempfile::TempDir, n_obs: usize, n_vars: usize) -> PathBuf {
        let base = write_test_file(dir, n_obs, n_vars);
        let v4 = dir.path().join("framed_v4.scx");
        scx_ops::optimize_with_framing(
            &base,
            &v4,
            None,
            scx_format_io::ObsShardPolicy::Always,
            Some(scx_format_io::FramingConfig::default()),
        )
        .unwrap();

        let r = ScxReader::open(&v4).unwrap();
        assert_eq!(
            r.header().format_version,
            scx_format_io::header::CURRENT_FORMAT_VERSION,
            "fixture must actually be v4, or the test proves nothing"
        );
        assert!(
            r.obs_metadata_shard_count() > 0,
            "fixture must actually have sharded obs"
        );
        v4
    }

    /// `scx upgrade` targets the newest **unframed** version (v3). Handed
    /// something newer it must decline, not "upgrade" it downwards.
    ///
    /// The rewrite is not a no-op on a v4 input: it re-emits every shard
    /// unframed and stamps v3, silently destroying the sub-shard random access
    /// v4 exists to provide — and it collapses the sharded obs layout on the
    /// way past. Nothing in the CLI help or `docs/api.md` says it does either.
    #[test]
    fn upgrade_declines_a_framed_v4_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_framed_v4_file(&dir, 8, 5);
        let output = dir.path().join("upgraded.scx");

        run_upgrade(&input, Some(output.as_path()), false).unwrap();

        assert!(
            !output.exists(),
            "a file newer than the upgrade target must be left alone, not \
             rewritten downwards into a new file"
        );
        let r = ScxReader::open(&input).unwrap();
        assert_eq!(
            r.header().format_version,
            scx_format_io::header::CURRENT_FORMAT_VERSION
        );
    }

    /// The `--in-place` variant of the above, which is the irrecoverable one:
    /// it renames a wholly new file over the target carrying no prior catalog,
    /// so `scx rollback` cannot undo it.
    #[test]
    fn upgrade_in_place_leaves_a_v4_file_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let input = write_framed_v4_file(&dir, 8, 5);

        let before = ScxReader::open(&input).unwrap();
        let (version, obs_shards, csr_entries) = (
            before.header().format_version,
            before.obs_metadata_shard_count(),
            before.catalog().csr_shards_sorted().len(),
        );
        let framed_before = before
            .read_shard_header(before.catalog().csr_shards_sorted()[0])
            .unwrap()
            .shard_format_version;
        assert_eq!(framed_before, 2, "fixture's shards must actually be framed");
        drop(before);

        run_upgrade(&input, None, true).unwrap();

        let after = ScxReader::open(&input).unwrap();
        assert_eq!(after.header().format_version, version, "version changed");
        assert_eq!(
            after.obs_metadata_shard_count(),
            obs_shards,
            "obs layout changed"
        );
        assert_eq!(after.catalog().csr_shards_sorted().len(), csr_entries);
        assert_eq!(
            after
                .read_shard_header(after.catalog().csr_shards_sorted()[0])
                .unwrap()
                .shard_format_version,
            framed_before,
            "row-group framing was stripped in place, and this is the path \
             `scx rollback` cannot undo"
        );
    }

    /// A CSC sidecar is a second representation of the same matrix. If
    /// canonicalizing X changes its structure, the sidecar built against the old
    /// matrix is no longer a view of the new one — and carrying it forward means
    /// the file holds two matrices that disagree, with `write_csc_shard`
    /// stamping `csc_build_generation` from the writer's `data_generation` so
    /// the freshness guard blesses it. A reader on `prefer_format="csc"` or the
    /// GPU CSC-direct DE route would silently get different numbers than one on
    /// CSR.
    ///
    /// So it is dropped — but only when canonicalization actually changed
    /// something. Carrying CSC is the one thing `upgrade` does that `optimize`
    /// and `build-csc` do not, and an already-canonical input (every file any
    /// current writer produces) keeps it. `test_upgrade_preserves_csc_multi_shard`
    /// covers that side.
    #[test]
    fn upgrade_drops_csc_when_canonicalizing_rewrites_the_matrix() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use scx_format_io::section::SectionType;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("noncanonical_with_csc.scx");
        let (n_obs, n_vars) = (2usize, 4usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        // Row 0 repeats column 1 (2 + 3 = 5) — canonicalizing drops one nnz.
        w.write_csr_shard(
            &[0u64, 3, 4],
            &[1u32, 1, 3, 0],
            &[2u8, 3, 7, 4],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        // A CSC sidecar built against that pre-canonicalization matrix.
        w.write_csc_shard(
            &[0u64, 1, 3, 3, 4],
            &[1u32, 0, 0, 0],
            &[4u8, 2, 3, 7],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let src = ScxReader::open(&input).unwrap();
        assert!(src.header().has_csc(), "fixture must carry a CSC sidecar");
        drop(src);

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert!(
            !out.header().has_csc(),
            "a CSC sidecar that no longer matches the canonicalized CSR must be \
             dropped, not carried forward and stamped fresh"
        );
        assert_eq!(
            out.catalog()
                .entries
                .iter()
                .filter(|e| e.section_type == SectionType::CscShard)
                .count(),
            0,
            "no CSC section may survive"
        );
        // The CSR really was rewritten — otherwise the drop above is vacuous.
        let got = out.read_all_csr_shards().unwrap();
        assert_eq!(got.data, vec![5.0, 7.0, 4.0]);
    }

    /// Multimodal is v2, so it lands in the branch this command still rewrites —
    /// and nothing here is modality-aware: `csr_shards_sorted()` returns every
    /// modality's shards and the loop writes them through the single-modality
    /// API, while `..Default::default()` zeroes the modality table. `optimize`
    /// and `build_csc` both refuse; `upgrade` was the one that did not.
    #[test]
    fn upgrade_refuses_a_multimodal_file() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use scx_format_io::modality::ModalityType;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("legacy_mm.scx");
        let (n_obs, rna_vars, adt_vars) = (6usize, 4usize, 3usize);

        // The shared multimodal fixture stamps v3, which the version gate
        // declines before the multimodal guard is reached — so it would pass
        // this test while exercising nothing. v2 is the multimodal feature
        // floor, and a multimodal file assembled from pre-v3 sources carries it
        // (`rewrite_output_format_version` clamps into `[2, 3]`), so that is the
        // shape the guard actually has to catch.
        let mut header = sample_header(n_obs as u64, rna_vars.max(adt_vars) as u64);
        header.format_version = 2;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        let rna = w
            .add_modality(
                "rna",
                ModalityType::Rna,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        let adt = w
            .add_modality(
                "adt",
                ModalityType::Protein,
                CodecId::None,
                ValueEncoding::Uint8,
                false,
            )
            .unwrap();
        w.write_var_for(rna, &sample_var(rna_vars)).unwrap();
        w.write_var_for(adt, &sample_var(adt_vars)).unwrap();
        w.set_modality_n_vars(rna, rna_vars as u64).unwrap();
        w.set_modality_n_vars(adt, adt_vars as u64).unwrap();
        for (id, m_vars) in [(rna, rna_vars), (adt, adt_vars)] {
            let mut indptr = vec![0u64];
            let (mut indices, mut values) = (Vec::new(), Vec::new());
            for r in 0..n_obs {
                indices.push((r % m_vars) as u32);
                values.push(((r + 1) % 256) as u8);
                indptr.push(indptr.last().unwrap() + 1);
            }
            w.write_csr_shard_for(
                id,
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        }
        w.finish().unwrap();

        // Only meaningful if the fixture actually reaches the rewrite branch.
        let src = ScxReader::open(&input).unwrap();
        assert!(src.is_multimodal());
        let version = src.header().format_version;
        drop(src);

        let output = dir.path().join("upgraded.scx");
        let err = run_upgrade(&input, Some(output.as_path()), false)
            .expect_err("a multimodal file must be refused, not flattened");
        assert!(
            err.to_string().contains("multimodal"),
            "error should name the reason, got: {err}"
        );
        assert!(!output.exists(), "nothing may be written on refusal");

        // And if the version gate would have caught it anyway, this test proves
        // nothing — so pin that it would not have.
        assert!(
            version < DEFAULT_WRITE_FORMAT_VERSION,
            "fixture is v{version}, which the version gate already declines — the \
             multimodal guard would be untested"
        );
    }

    /// Canonicalizing **sums** duplicate coordinates, so a value can outgrow the
    /// width the source picked for the values it actually held. Re-encoding
    /// through the old width fails two ways, and the quiet one is worse:
    /// `Uint8::encode_f32` refuses `200 + 200 = 400` outright — so the upgrade
    /// errors on exactly the non-canonical input it exists to repair — while
    /// `Float16` is unchecked and `f16::from_f32` maps anything past 65504 to
    /// **infinity**, reporting success.
    #[test]
    fn upgrade_widens_the_value_encoding_when_canonicalizing_overflows_it() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("uint8_overflow.scx");
        let (n_obs, n_vars) = (1usize, 3usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        // Column 1 duplicated: 200 + 200 = 400, past what Uint8 can hold.
        w.write_csr_shard(
            &[0u64, 2],
            &[1u32, 1],
            &[200u8, 200],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false)
            .expect("a duplicate summing past the source width must not fail the upgrade");

        let out = ScxReader::open(&output).unwrap();
        let got = out.read_all_csr_shards().unwrap();
        assert_eq!(
            got.data,
            vec![400.0],
            "the summed value must survive intact, not saturate or error"
        );
    }

    /// Widening past `Uint32` lands on `Float32`, and `Scx1` encodes integers
    /// only — so the widened encoding has to drag the codec with it. The
    /// `encode_one_shard*` path does that downgrade itself
    /// (`scx-format-io/src/encoder.rs`), but `write_csr_shard` /
    /// `write_layer_csr_shard` take the codec they are given and hand it
    /// straight to `encode_shard_adaptive`, which returns `FloatWithScx1`.
    /// So the arm that exists to *avoid* aborting the upgrade would abort it.
    ///
    /// `2_500_000_000 + 2_500_000_000` is exact in f32 both before and after
    /// summing (each is a multiple of its binade's ULP), so the fixture tests
    /// the codec transition and nothing else.
    #[test]
    fn upgrade_pairs_a_widened_float_encoding_with_a_codec_that_can_hold_it() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("scx1_u32_overflow.scx");
        let (n_obs, n_vars) = (1usize, 3usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        let half = 2_500_000_000u32;
        let raw: Vec<u8> = [half, half].iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_csr_shard(
            &[0u64, 2],
            &[1u32, 1],
            &raw,
            CodecId::Scx1,
            ValueEncoding::Uint32,
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false).expect(
            "a Uint32 sum past 2³² must widen to Float32 AND drop Scx1, not fail FloatWithScx1",
        );

        let out = ScxReader::open(&output).unwrap();
        let entry = out.catalog().shards_sorted()[0];
        let sh = out.read_shard_header(entry).unwrap();
        assert_eq!(
            ValueEncoding::from_u8(sh.value_encoding),
            Some(ValueEncoding::Float32)
        );
        assert_ne!(
            CodecId::from_u8(sh.codec_id),
            Some(CodecId::Scx1),
            "Scx1 cannot encode a float value encoding"
        );
        assert_eq!(out.read_all_csr_shards().unwrap().data, vec![5e9]);
    }

    /// The layer twin of the above, and the more exposed of the two: the X loop
    /// only runs the ladder when `is_canonical_csr` says the shard needs
    /// rewriting, but `copy_layers` runs it on **every** layer shard whenever
    /// `canonicalize` is set — so a layer reaches the widening arm on inputs the
    /// X path never would.
    #[test]
    fn upgrade_pairs_a_widened_layer_encoding_with_a_codec_that_can_hold_it() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("scx1_layer_overflow.scx");
        let (n_obs, n_vars) = (1usize, 3usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        w.write_csr_shard(
            &[0u64, 1],
            &[0u32],
            &[1u8],
            CodecId::None,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
        let half = 2_500_000_000u32;
        let raw: Vec<u8> = [half, half].iter().flat_map(|v| v.to_le_bytes()).collect();
        w.write_layer_csr_shard(
            &[0u64, 2],
            &[1u32, 1],
            &raw,
            CodecId::Scx1,
            ValueEncoding::Uint32,
            0,
            "counts",
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false)
            .expect("a layer shard widened to Float32 must drop Scx1, not fail FloatWithScx1");

        let out = ScxReader::open(&output).unwrap();
        // The header, not just the decoded value: a decode-only assertion still
        // passes if the shard were written Float32 under some other wrong codec.
        let entry = out
            .catalog()
            .entries
            .iter()
            .find(|e| e.section_type == scx_format_io::section::SectionType::LayerCsrShard)
            .expect("the upgraded file must carry the layer shard");
        let sh = out.read_shard_header(entry).unwrap();
        assert_eq!(
            ValueEncoding::from_u8(sh.value_encoding),
            Some(ValueEncoding::Float32)
        );
        assert_ne!(CodecId::from_u8(sh.codec_id), Some(CodecId::Scx1));
        assert_eq!(out.read_layer("counts").unwrap().data, vec![5e9]);
    }

    /// `u32::MAX as f32` **is** 2³², so a decoded on-disk `u32::MAX` is
    /// indistinguishable from a canonicalized sum that reached 2³². Re-encoding
    /// it as `Uint32` saturates it back to exactly `u32::MAX`, which makes the
    /// rewrite lossless for that value; widening to `Float32` instead writes
    /// 2³², which is one larger and no longer fits `u32` at all — so a file that
    /// read fine as `uint32` stops doing so after an upgrade that was supposed
    /// to preserve it. The widening arm must therefore trigger strictly *above*
    /// 2³², not at it.
    ///
    /// The shard is non-canonical (unsorted indices) so the ladder actually
    /// runs; a canonical shard never reaches it on the X path.
    #[test]
    fn upgrade_does_not_widen_a_decoded_u32_max_into_a_value_u32_cannot_hold() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use scx_sparse::materialize::{MaterializePlan, ValueBuffer, ValueDtype};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("u32_max.scx");
        let (n_obs, n_vars) = (1usize, 3usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        let raw: Vec<u8> = [u32::MAX, 5u32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        w.write_csr_shard(
            &[0u64, 2],
            &[2u32, 1], // descending: forces canonicalization
            &raw,
            CodecId::None,
            ValueEncoding::Uint32,
            0,
        )
        .unwrap();
        w.finish().unwrap();

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false).unwrap();

        let out = ScxReader::open(&output).unwrap();
        let entry = out.catalog().shards_sorted()[0];
        let sh = out.read_shard_header(entry).unwrap();
        assert_eq!(
            ValueEncoding::from_u8(sh.value_encoding),
            Some(ValueEncoding::Uint32),
            "a format-valid Uint32 archive must not be rewritten as float"
        );

        let plan = MaterializePlan {
            data_dtype: ValueDtype::U32,
            ..MaterializePlan::default_csr_f32()
        };
        let typed = out
            .read_all_csr_shards_typed(&plan)
            .expect("the upgraded file must still read as uint32");
        match typed.values {
            ValueBuffer::U32(v) => assert!(
                v.contains(&u32::MAX),
                "u32::MAX did not survive the upgrade: {v:?}"
            ),
            other => panic!("expected a u32 buffer, got {other:?}"),
        }
    }

    /// The unit-level statement of the same rule, including the `Float16` arm —
    /// which cannot be exercised end-to-end here (no CLI path writes f16 counts)
    /// but is the one that fails *silently*.
    #[test]
    fn canonicalized_encoding_widens_only_when_the_sums_need_it() {
        use scx_codec::ValueEncoding::*;
        // Fits: unchanged.
        assert_eq!(scx_ops::encoding_for_canonicalized(Uint8, &[255.0]), Uint8);
        assert_eq!(
            scx_ops::encoding_for_canonicalized(Uint16, &[65535.0]),
            Uint16
        );
        assert_eq!(
            scx_ops::encoding_for_canonicalized(Float16, &[65504.0]),
            Float16
        );
        // Overflows: widened, and only one step where one step suffices.
        assert_eq!(scx_ops::encoding_for_canonicalized(Uint8, &[400.0]), Uint16);
        assert_eq!(
            scx_ops::encoding_for_canonicalized(Uint8, &[70000.0]),
            Uint32,
            "a Uint8 sum past u16 must go all the way to u32, not stop at u16"
        );
        assert_eq!(
            scx_ops::encoding_for_canonicalized(Float16, &[70000.0]),
            Float32,
            "f16 saturates to infinity past 65504 without complaining"
        );
    }

    /// Switching obs/var to the streaming writers must not quietly trade away
    /// the shard-cover validation `read_obs()` / `read_var()` performed on the
    /// way to assembling the axis.
    ///
    /// That check is what stands between a malformed sharded axis and a
    /// *plausible* output: the streaming writer re-derives each output shard's
    /// `row_start` from batch lengths, so a gapped input is not merely copied
    /// through — it is normalised into a well-formed file whose metadata now
    /// describes different matrix rows than it did. An error is the only
    /// acceptable answer, and it is the one the assembling path already gave.
    #[test]
    fn upgrade_rejects_a_gapped_obs_shard_cover() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("gapped.scx");
        let (n_obs, n_vars) = (6usize, 4usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        let obs = sample_obs(n_obs);
        // Shard 0 covers rows [0, 3). Shard 1 claims to start at row 4, leaving
        // row 3 described by nothing — the cover no longer tiles [0, n_obs).
        w.write_obs_shard(0, 0, 3, n_obs as u64, &obs.slice(0, 3))
            .unwrap();
        w.write_obs_shard(1, 4, 3, n_obs as u64, &obs.slice(3, 3))
            .unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();

        let mut indptr = vec![0u64];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in 0..n_obs {
            indices.push(((row * 2) % n_vars) as u32);
            values.push(((row + 1) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
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

        // The assembling path rejects this — that is the behaviour being kept.
        let src = ScxReader::open(&input).unwrap();
        assert!(
            src.read_obs().is_err(),
            "fixture must be malformed enough for read_obs() to reject it"
        );
        drop(src);

        let output = dir.path().join("upgraded.scx");
        let err = run_upgrade(&input, Some(output.as_path()), false)
            .expect_err("a gapped obs cover must not upgrade to a well-formed file");
        let msg = err.to_string();
        assert!(
            msg.contains("row_start") && msg.contains("obs_metadata"),
            "the error should name the axis and what was wrong, got: {msg}"
        );
    }

    /// A multimodal file at or past the target version is never rewritten, so
    /// there is nothing to protect it from — it must stay the exit-0 "nothing to
    /// do" it always was. Placing the refusal before the version gates turned
    /// that into a hard error for the common modern case, contradicting the
    /// module contract and `docs/api.md`.
    #[test]
    fn upgrade_leaves_an_at_target_multimodal_file_alone() {
        use crate::test_utils::write_multimodal_test_file;

        let dir = tempfile::tempdir().unwrap();
        let input = write_multimodal_test_file(&dir, 6, 4, 3);
        let src = ScxReader::open(&input).unwrap();
        assert!(src.is_multimodal());
        assert_eq!(
            src.header().format_version,
            DEFAULT_WRITE_FORMAT_VERSION,
            "fixture must be at the target version for this to test the ordering"
        );
        drop(src);

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false)
            .expect("an already-at-target multimodal file must be a no-op, not an error");
        assert!(!output.exists(), "a no-op writes nothing");
    }

    /// A shard whose stamp disagrees with the payload it carries. Walking the
    /// stamps alone proves they tile a range; it does not prove each shard holds
    /// the rows it claims, and the streaming writer derives the output range from
    /// the payload — so this turns a detectable input into a malformed output.
    #[test]
    fn upgrade_rejects_a_shard_whose_stamp_disagrees_with_its_payload() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("stamp_mismatch.scx");
        let (n_obs, n_vars) = (6usize, 4usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        let obs = sample_obs(n_obs);
        // Stamped 3 rows, carries 2. The stamps still tile [0, 6).
        w.write_obs_shard(0, 0, 3, n_obs as u64, &obs.slice(0, 2))
            .unwrap();
        w.write_obs_shard(1, 3, 3, n_obs as u64, &obs.slice(3, 3))
            .unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();
        let mut indptr = vec![0u64];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in 0..n_obs {
            indices.push(((row * 2) % n_vars) as u32);
            values.push(((row + 1) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 1);
        }
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

        let output = dir.path().join("upgraded.scx");
        let err = run_upgrade(&input, Some(output.as_path()), false)
            .expect_err("a shard whose stamp overstates its payload must be refused");
        let msg = err.to_string();
        assert!(
            msg.contains("n_shard_rows") && msg.contains("carries"),
            "the error should say the stamp and the payload disagree, got: {msg}"
        );
    }

    /// An upgrade stamps v3, and v3's contract *is* canonical CSR — indices
    /// sorted within each row, duplicate coordinates summed, no explicit zeros.
    /// A pre-v3 source carries no such guarantee, which is exactly why
    /// `rewrite_output_format_version` refuses to promote one.
    ///
    /// So the output must actually *be* canonical, not merely claim it. This
    /// feeds a v1 file whose single row has unsorted indices, a duplicate
    /// coordinate and an explicit zero, and asserts the upgraded file satisfies
    /// the invariant its own header advertises. The sharded-layout fixture next
    /// door cannot catch this — its rows are already canonical, so it passes
    /// whether or not anything canonicalizes.
    #[test]
    fn upgrade_canonicalizes_before_claiming_v3() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("legacy_noncanonical.scx");
        let (n_obs, n_vars) = (2usize, 6usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();
        w.write_obs(&sample_obs(n_obs)).unwrap();
        w.write_var(&sample_var(n_vars)).unwrap();

        // Row 0: descending indices, column 2 repeated (3 + 4 = 7), and an
        // explicit zero at column 5. Row 1: already canonical, so the test also
        // shows canonicalization leaves a well-formed row alone.
        let indptr: Vec<u64> = vec![0, 4, 5];
        let indices: Vec<u32> = vec![4, 2, 2, 5, 1];
        let values: Vec<u8> = vec![9, 3, 4, 0, 8];
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

        // The fixture really is non-canonical, or the assertion below is empty.
        let src = ScxReader::open(&input).unwrap();
        assert_eq!(src.header().format_version, 1);
        let s = src.read_all_csr_shards().unwrap();
        assert!(
            !scx_sparse::is_canonical_csr(
                &s.indptr.iter().map(|&v| v as u64).collect::<Vec<_>>(),
                &s.indices.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                &s.data
            ),
            "fixture must be non-canonical or this test cannot fail"
        );
        drop(src);

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(
            out.header().format_version,
            DEFAULT_WRITE_FORMAT_VERSION,
            "the upgrade must still have happened"
        );
        let got = out.read_all_csr_shards().unwrap();
        let (ip, ix): (Vec<u64>, Vec<u32>) = (
            got.indptr.iter().map(|&v| v as u64).collect(),
            got.indices.iter().map(|&v| v as u32).collect(),
        );
        assert!(
            scx_sparse::is_canonical_csr(&ip, &ix, &got.data),
            "a v3 header claims canonical CSR; the output must hold that invariant. \
             got indptr={ip:?} indices={ix:?} data={:?}",
            got.data
        );
        // Sorted, the duplicate summed, the explicit zero dropped — and the
        // values are the ones the input meant, not merely *a* canonical shape.
        assert_eq!(ix, vec![2, 4, 1]);
        assert_eq!(got.data, vec![7.0, 9.0, 8.0]);
        assert_eq!(ip, vec![0, 2, 3]);
    }

    /// obs and var pass through an upgrade 1:1, so a row-sharded input must
    /// come out row-sharded.
    ///
    /// `read_obs()` + `write_obs()` assembles every `ObsMetadataShard` into one
    /// in-memory batch and emits it as a single legacy section: peak RSS
    /// O(n_obs) — the OOM the sharded layout exists to prevent — and a silent
    /// loss of the row-sharded-obs precondition Level-2 row-set pushdown
    /// depends on. Same defect `build-csc` had.
    ///
    /// The fixture is stamped pre-v3 so it reaches the rewrite at all: sharded
    /// obs sets no version floor (`rewrite_output_format_version` clamps into
    /// `[feature_floor, 3]` and nothing raises the floor for section types
    /// 24/25), so a merge or append over pre-v3 inputs yields exactly this —
    /// an old file with sharded obs, which is what `scx upgrade` is *for*.
    #[test]
    fn upgrade_preserves_sharded_obs_and_var_layout() {
        use crate::test_utils::{sample_header, sample_obs, sample_var};
        use scx_format_io::section::SectionType;

        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("legacy_sharded.scx");
        let (n_obs, n_vars) = (6usize, 4usize);

        let mut header = sample_header(n_obs as u64, n_vars as u64);
        header.format_version = 1;
        let mut w = ScxWriter::new(&input, header).unwrap();

        let obs = sample_obs(n_obs);
        for shard_idx in 0u32..2 {
            let row_start = shard_idx as usize * 3;
            w.write_obs_shard(
                shard_idx,
                row_start as u64,
                3,
                n_obs as u64,
                &obs.slice(row_start, 3),
            )
            .unwrap();
        }
        let var = sample_var(n_vars);
        for shard_idx in 0u32..2 {
            let col_start = shard_idx as usize * 2;
            w.write_var_shard(
                shard_idx,
                col_start as u64,
                2,
                n_vars as u64,
                &var.slice(col_start, 2),
            )
            .unwrap();
        }

        let mut indptr = vec![0u64];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in 0..n_obs {
            indices.push(((row * 2) % n_vars) as u32);
            indices.push(((row * 2 + 1) % n_vars) as u32);
            values.push(((row + 1) % 256) as u8);
            values.push(((row + 2) % 256) as u8);
            indptr.push(indptr.last().unwrap() + 2);
        }
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

        // The gate has to let this through, or the test proves nothing.
        let src = ScxReader::open(&input).unwrap();
        assert_eq!(src.header().format_version, 1);
        assert_eq!(src.obs_metadata_shard_count(), 2);
        assert_eq!(src.var_metadata_shard_count(), 2);
        drop(src);

        let output = dir.path().join("upgraded.scx");
        run_upgrade(&input, Some(output.as_path()), false).unwrap();

        let out = ScxReader::open(&output).unwrap();
        assert_eq!(
            out.header().format_version,
            DEFAULT_WRITE_FORMAT_VERSION,
            "the upgrade itself must still have happened"
        );
        assert_eq!(
            out.obs_metadata_shard_count(),
            2,
            "sharded obs must stay sharded through an upgrade"
        );
        assert_eq!(
            out.var_metadata_shard_count(),
            2,
            "sharded var must stay sharded through an upgrade"
        );
        for collapsed in [SectionType::ObsMetadata, SectionType::VarMetadata] {
            assert!(
                !out.catalog()
                    .entries
                    .iter()
                    .any(|e| e.section_type == collapsed),
                "no collapsed single-section {collapsed:?} may be emitted"
            );
        }
        // ...and the content still round-trips.
        assert_eq!(out.read_obs().unwrap().num_rows(), n_obs);
        assert_eq!(out.read_var().unwrap().num_rows(), n_vars);
        assert_eq!(out.read_all_csr_shards().unwrap().shape, (n_obs, n_vars));
    }
}
