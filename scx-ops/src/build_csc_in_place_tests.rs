//! The in-place append: what it must not touch, what it must write, and how it
//! is undone.

use super::*;
use crate::test_utils::{sample_header, sample_obs, sample_var};
use scx_format_io::header::{CURRENT_FORMAT_VERSION, DEFAULT_WRITE_FORMAT_VERSION};

/// `n_obs x n_vars`, two nnz per row, split into `n_shards` row shards, at the
/// given format version — framed (v2 shards) iff `framed`.
fn write_input(
    path: &Path,
    n_obs: usize,
    n_vars: usize,
    n_shards: usize,
    framed: bool,
) -> std::path::PathBuf {
    let mut header = sample_header(n_obs as u64, n_vars as u64);
    header.format_version = if framed {
        CURRENT_FORMAT_VERSION
    } else {
        DEFAULT_WRITE_FORMAT_VERSION
    };
    let mut writer = ScxWriter::new(path, header).unwrap();
    if framed {
        writer.set_framing(Some(FramingConfig::default()));
    }
    writer.write_obs(&sample_obs(n_obs)).unwrap();
    writer.write_var(&sample_var(n_vars)).unwrap();
    let rows_per = n_obs.div_ceil(n_shards);
    for s in 0..n_shards {
        let (lo, hi) = (s * rows_per, ((s + 1) * rows_per).min(n_obs));
        let mut indptr = vec![0u64];
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        for row in lo..hi {
            let mut cols = [(row * 3) % n_vars, (row * 3 + 1) % n_vars];
            cols.sort_unstable();
            for (k, c) in cols.iter().enumerate() {
                indices.push(*c as u32);
                values.push(((row + k) % 200 + 1) as u8);
            }
            indptr.push(indices.len() as u64);
        }
        writer
            .write_csr_shard(
                &indptr,
                &indices,
                &values,
                CodecId::None,
                ValueEncoding::Uint8,
                lo as u64,
            )
            .unwrap();
    }
    writer.finish().unwrap();
    path.to_path_buf()
}

fn entries_of(path: &Path, t: SectionType) -> Vec<(String, u64, u64, [u8; 32])> {
    ScxReader::open(path)
        .unwrap()
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == t)
        .map(|e| (e.name.clone(), e.offset, e.length, e.checksum))
        .collect()
}

fn assert_csc_is_the_transpose(path: &Path) {
    let r = ScxReader::open(path).unwrap();
    let want = scx_sparse::transpose::csr_to_csc(&r.read_all_csr_shards().unwrap());
    let got = r.read_all_csc_shards().unwrap();
    assert_eq!(got.indptr, want.indptr);
    assert_eq!(got.indices, want.indices);
    assert_eq!(got.data, want.data);
}

fn shard_versions(path: &Path, t: SectionType) -> Vec<u8> {
    let r = ScxReader::open(path).unwrap();
    r.catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == t)
        .map(|e| r.read_shard_header(e).unwrap().shard_format_version)
        .collect()
}

/// The contract of an append, stated on the bytes: everything between the root
/// catalog and the old EOF is unchanged. Only the header, the 4096-byte root
/// catalog and the new tail may differ — so no CSR shard, obs/var section or
/// old catalog was rewritten.
#[test]
fn an_in_place_build_only_appends() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_input(&dir.path().join("in.scx"), 30, 9, 3, false);
    let before = std::fs::read(&path).unwrap();
    let before_reader = ScxReader::open(&path).unwrap();
    let (old_catalog_offset, old_seq, old_gen) = (
        before_reader.header().full_catalog_offset,
        before_reader.catalog().manifest_sequence,
        before_reader.catalog().data_generation,
    );
    let n_prov_before = before_reader
        .read_provenance()
        .map(|p| p.operations.len())
        .unwrap_or(0);
    drop(before_reader);
    let csr_before = entries_of(&path, SectionType::CsrShard);

    let outcome = crate::rebuild_csc_inplace(&path, 4, "1G", None, None).unwrap();
    assert_eq!(outcome, BuildCscOutcome::Built);

    let after = std::fs::read(&path).unwrap();
    let start = scx_format_io::writer::SECTIONS_START_OFFSET as usize;
    assert!(after.len() > before.len());
    assert!(
        after[start..before.len()] == before[start..],
        "an in-place build rewrote bytes it did not own"
    );
    assert_eq!(
        entries_of(&path, SectionType::CsrShard),
        csr_before,
        "every CSR entry keeps its offset, length and checksum"
    );
    assert_csc_is_the_transpose(&path);

    let r = ScxReader::open(&path).unwrap();
    assert!(r.header().has_csc());
    assert_eq!(r.header().n_csc_shards, 3, "9 columns at 4 per shard");
    assert_eq!(
        r.catalog().prev_catalog_offset,
        old_catalog_offset,
        "the pre-build catalog is the previous one — what rollback restores"
    );
    assert_eq!(r.catalog().manifest_sequence, old_seq + 1);
    assert_eq!(r.catalog().data_generation, old_gen, "X is untouched");
    assert_eq!(
        r.catalog().csc_build_generation,
        old_gen,
        "and the sidecar is fresh"
    );
    let prov = r.read_provenance().unwrap().operations;
    assert_eq!(
        prov.len(),
        n_prov_before + 1,
        "one provenance entry, appended"
    );
    let last = prov.last().unwrap();
    assert_eq!(last.action, "build-csc");
    assert!(last.params_json.contains("\"csc_cols_per_shard\":4"));
}

/// A framed v4 file stays v4 with its CSR framing, and the sidecar is framed
/// to match; an unframed v3 file gets an unframed sidecar.
#[test]
fn the_sidecar_follows_the_files_framing() {
    let dir = tempfile::tempdir().unwrap();
    let v4 = write_input(&dir.path().join("v4.scx"), 600, 11, 2, true);
    assert_eq!(shard_versions(&v4, SectionType::CsrShard), vec![2, 2]);
    let csr_before = entries_of(&v4, SectionType::CsrShard);
    // `None`, which used to strip the framing off the whole file.
    crate::rebuild_csc_inplace(&v4, 5, "1G", None, None).unwrap();
    let r = ScxReader::open(&v4).unwrap();
    assert_eq!(r.header().format_version, CURRENT_FORMAT_VERSION);
    assert_eq!(entries_of(&v4, SectionType::CsrShard), csr_before);
    let csc = shard_versions(&v4, SectionType::CscShard);
    assert!(
        !csc.is_empty() && csc.iter().all(|&v| v >= 2),
        "framed: {csc:?}"
    );
    assert_csc_is_the_transpose(&v4);

    let v3 = write_input(&dir.path().join("v3.scx"), 60, 11, 2, false);
    crate::rebuild_csc_inplace(&v3, 5, "1G", None, None).unwrap();
    assert_eq!(
        ScxReader::open(&v3).unwrap().header().format_version,
        DEFAULT_WRITE_FORMAT_VERSION
    );
    let csc = shard_versions(&v3, SectionType::CscShard);
    assert!(csc.iter().all(|&v| v == 1), "unframed: {csc:?}");
}

/// Framing a sidecar on a ≤ v3 file would need the CSR re-framed, which an
/// append cannot do. Refused before any byte is written.
#[test]
fn framing_an_unframed_file_is_refused_and_leaves_it_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let v3 = write_input(&dir.path().join("v3.scx"), 20, 6, 2, false);
    let before = std::fs::read(&v3).unwrap();
    let err = crate::rebuild_csc_inplace(&v3, 5, "1G", Some(FramingConfig::default()), None)
        .unwrap_err()
        .to_string();
    assert!(err.contains("scx optimize"), "names the way out: {err}");
    assert_eq!(std::fs::read(&v3).unwrap(), before);
}

/// Rebuilding replaces the sidecar: the old entries leave the catalog (their
/// bytes stay, orphaned), the new ones are named from zero without colliding,
/// and one rollback returns to the first sidecar.
#[test]
fn a_rebuild_replaces_the_sidecar_and_rolls_back_to_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_input(&dir.path().join("in.scx"), 40, 12, 4, false);
    crate::rebuild_csc_inplace(&path, 6, "1G", None, None).unwrap();
    let first = entries_of(&path, SectionType::CscShard);
    assert_eq!(first.len(), 2);

    crate::rebuild_csc_inplace(&path, 4, "1G", None, None).unwrap();
    let second = entries_of(&path, SectionType::CscShard);
    assert_eq!(second.len(), 3);
    let names: std::collections::BTreeSet<_> = second.iter().map(|e| e.0.clone()).collect();
    assert_eq!(names.len(), 3, "no duplicate section names");
    assert!(
        second.iter().all(|e| first.iter().all(|f| f.1 != e.1)),
        "the new sidecar is new bytes, not the old entries renamed"
    );
    assert_eq!(ScxReader::open(&path).unwrap().header().n_csc_shards, 3);
    assert_csc_is_the_transpose(&path);

    crate::rollback(&path).unwrap();
    assert_eq!(entries_of(&path, SectionType::CscShard), first);
    assert_eq!(ScxReader::open(&path).unwrap().header().n_csc_shards, 2);
    crate::rollback(&path).unwrap();
    let r = ScxReader::open(&path).unwrap();
    assert!(!r.header().has_csc(), "and the second rollback removes it");
}

/// A 0-row file carrying a stale sidecar loses it in place — and, being an
/// in-place op now, that too is undoable.
#[test]
fn an_empty_matrix_drops_a_stale_sidecar_in_place() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_with_csc.scx");
    let mut writer = ScxWriter::new(&path, sample_header(0, 2)).unwrap();
    writer.write_obs(&sample_obs(0)).unwrap();
    writer.write_var(&sample_var(2)).unwrap();
    writer
        .write_csc_shard(&[0, 0, 0], &[], &[], CodecId::None, ValueEncoding::Uint8, 0)
        .unwrap();
    writer.finish().unwrap();
    assert!(ScxReader::open(&path).unwrap().header().has_csc());

    let outcome = crate::rebuild_csc_inplace(&path, 5000, "1G", None, None).unwrap();
    assert_eq!(outcome, BuildCscOutcome::NoSidecar);
    let r = ScxReader::open(&path).unwrap();
    assert!(!r.header().has_csc());
    assert_eq!(r.header().n_csc_shards, 0);
    // Left as it was (1, stamped when the stale sidecar was written): the field
    // also keeps a layer sidecar fresh, so the drop does not zero it.
    assert_eq!(
        r.catalog().csc_build_generation,
        r.catalog().data_generation
    );
    drop(r);

    crate::rollback(&path).unwrap();
    assert!(ScxReader::open(&path).unwrap().header().has_csc());
}

/// An empty matrix with no sidecar is a true no-op: not even a new catalog.
#[test]
fn an_empty_matrix_without_a_sidecar_is_not_touched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.scx");
    let mut writer = ScxWriter::new(&path, sample_header(0, 3)).unwrap();
    writer.write_obs(&sample_obs(0)).unwrap();
    writer.write_var(&sample_var(3)).unwrap();
    writer.finish().unwrap();
    let before = std::fs::read(&path).unwrap();
    let outcome = crate::rebuild_csc_inplace(&path, 5000, "1G", None, None).unwrap();
    assert_eq!(outcome, BuildCscOutcome::NoSidecar);
    assert_eq!(std::fs::read(&path).unwrap(), before);
}

/// The copy-out form leaves its input byte-identical, leaves no staging file,
/// and its output is "input, then an append" — so rolling the output back
/// yields the input's catalog, sidecar-free.
#[test]
fn copy_out_is_a_copy_then_an_append() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_input(&dir.path().join("in.scx"), 30, 8, 3, false);
    let before = std::fs::read(&input).unwrap();
    let output = dir.path().join("out.scx");
    run_build_csc(&input, &output, "1G", false, 3, None, None).unwrap();

    assert_eq!(std::fs::read(&input).unwrap(), before, "input untouched");
    let out = std::fs::read(&output).unwrap();
    let start = scx_format_io::writer::SECTIONS_START_OFFSET as usize;
    assert!(out[start..before.len()] == before[start..]);
    assert_csc_is_the_transpose(&output);
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names.len(), 2, "no staging file survives: {names:?}");

    // An existing output is refused without `force`, and replaced with it.
    assert!(run_build_csc(&input, &output, "1G", false, 3, None, None).is_err());
    run_build_csc(&input, &output, "1G", true, 3, None, None).unwrap();

    crate::rollback(&output).unwrap();
    assert!(!ScxReader::open(&output).unwrap().header().has_csc());
}

/// A copy-out refusal happens against the input, before the copy.
#[test]
fn a_refused_copy_out_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let input = write_input(&dir.path().join("in.scx"), 30, 8, 3, false);
    let output = dir.path().join("out.scx");
    let err = run_build_csc(
        &input,
        &output,
        "1G",
        false,
        3,
        Some(FramingConfig::default()),
        None,
    );
    assert!(err.is_err());
    let names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        names,
        vec!["in.scx".to_string()],
        "no output, no staging: {names:?}"
    );
}

/// A failure between the first append and the commit leaves the file at its
/// original length, rather than a sidecar-sized orphan tail.
#[test]
fn truncate_on_error_restores_the_length_and_keeps_the_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_input(&dir.path().join("in.scx"), 10, 4, 1, false);
    let before = std::fs::read(&path).unwrap();
    let mut lock = FileLock::acquire_exclusive(&path).unwrap();
    let len = lock.seek(SeekFrom::End(0)).unwrap();

    let err = truncate_on_error(&mut lock, len, |lock| -> Result<(), BoxError> {
        lock.write_all(&[7u8; 4096])?;
        lock.flush()?;
        Err("injected".into())
    })
    .unwrap_err();
    assert_eq!(err.to_string(), "injected");
    drop(lock);
    assert_eq!(std::fs::read(&path).unwrap(), before);

    // And success keeps what was written.
    let mut lock = FileLock::acquire_exclusive(&path).unwrap();
    truncate_on_error(&mut lock, len, |lock| -> Result<(), BoxError> {
        lock.seek(SeekFrom::End(0))?;
        lock.write_all(&[1u8; 8])?;
        Ok(())
    })
    .unwrap();
    drop(lock);
    assert_eq!(std::fs::read(&path).unwrap().len(), before.len() + 8);
}
