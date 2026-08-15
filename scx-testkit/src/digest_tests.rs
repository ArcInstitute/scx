//! Premise tests for the digest harness.
//!
//! Every one of these was watched failing against a deliberately broken digest
//! before being trusted. A harness whose own tests have only ever been green is
//! the thing this crate exists to stop other people shipping.

use super::*;
use crate::fixtures::{mixed_codec_file, mixed_codec_file_with, FixtureOpts};
use scx_format_io::ScxReader;

fn write(dir: &tempfile::TempDir, name: &str, opts: &FixtureOpts) -> std::path::PathBuf {
    mixed_codec_file_with(&dir.path().join(name), opts).unwrap()
}

fn dig(p: &std::path::Path) -> FileDigest {
    digest_file(p, Strictness::Content).unwrap()
}

/// The fixture really does carry all three encoder paths.
///
/// Without this the four tests below could all pass against a file that is
/// three copies of the same unframed `CodecId::None` shard — i.e. against a
/// digest that pins no codec at all. This is the premise the rest rests on.
#[test]
fn fixture_covers_unframed_framed_and_float() {
    let dir = tempfile::tempdir().unwrap();
    let path = mixed_codec_file(&dir.path().join("f.scx")).unwrap();
    let reader = ScxReader::open(&path).unwrap();

    let mut codecs = Vec::new();
    let mut framed = 0;
    for entry in reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == scx_format_io::section::SectionType::CsrShard)
    {
        let bytes = reader.section_bytes(entry).unwrap();
        let h = scx_format_io::shard::ShardHeader::read_from(&mut &bytes[..]).unwrap();
        codecs.push(h.codec_id);
        if h.shard_format_version >= 2 {
            framed += 1;
        }
    }
    codecs.sort_unstable();
    assert!(
        codecs.contains(&(scx_codec::CodecId::None as u8)),
        "no unframed plain shard: {codecs:?}"
    );
    assert!(
        codecs.contains(&(scx_codec::CodecId::Scx1 as u8)),
        "no Scx1 integer shard: {codecs:?}"
    );
    assert!(
        codecs.contains(&(scx_codec::CodecId::Pcodec as u8)),
        "no float shard — an identity claim tested only on integer data is \
         vacuous: {codecs:?}"
    );
    assert_eq!(framed, 1, "exactly one shard should be row-group framed");

    // And the file is readable, so the digest is describing a valid file.
    let csr = reader.read_all_csr_shards().unwrap();
    assert_eq!(csr.shape.0, 24);
}

/// A digest of a file equals a digest of the same file. The trivial direction,
/// and the one that would break first if section ordering were unstable.
#[test]
fn digest_is_stable_for_one_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = mixed_codec_file(&dir.path().join("f.scx")).unwrap();
    assert_digests_eq(&dig(&path), &dig(&path));
}

/// Two independent writes of the same fixture agree — so a difference in a
/// later test is a real difference, not write nondeterminism.
#[test]
fn two_identical_writes_agree() {
    let dir = tempfile::tempdir().unwrap();
    let o = FixtureOpts::default();
    assert_digests_eq(
        &dig(&write(&dir, "a.scx", &o)),
        &dig(&write(&dir, "b.scx", &o)),
    );
}

/// Reordering which row range lands in which shard changes the digest.
///
/// This is the reorder-sensitivity requirement. A digest that sorted sections
/// by content rather than by name would call these two files equal — the three
/// shards hold the same rows overall, only differently assigned — and would
/// then be blind to exactly the failure mode a rewrite-loop refactor
/// introduces.
#[test]
fn digest_detects_a_reordered_shard_write() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(&dir, "a.scx", &FixtureOpts::default());
    let b = write(
        &dir,
        "b.scx",
        &FixtureOpts {
            shard_order: [1, 0, 2],
            ..Default::default()
        },
    );

    // Premise: both files still hold the same matrix, so nothing but the
    // shard assignment differs.
    let (ra, rb) = (ScxReader::open(&a).unwrap(), ScxReader::open(&b).unwrap());
    let (ca, cb) = (
        ra.read_all_csr_shards().unwrap(),
        rb.read_all_csr_shards().unwrap(),
    );
    assert_eq!(ca.shape, cb.shape);
    assert_eq!(ca.indptr, cb.indptr, "premise: same logical matrix");
    assert_eq!(ca.indices, cb.indices, "premise: same logical matrix");

    assert_ne!(
        dig(&a).sections,
        dig(&b).sections,
        "a reordered shard write must change the digest"
    );
}

/// Changing one shard's codec changes the digest, and the diff names that
/// shard rather than reporting that the files differ.
#[test]
fn digest_detects_a_perturbed_codec_choice() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(&dir, "a.scx", &FixtureOpts::default());
    let b = write(
        &dir,
        "b.scx",
        &FixtureOpts {
            framed_codec: scx_codec::CodecId::Zstd,
            ..Default::default()
        },
    );

    let (da, db) = (dig(&a), dig(&b));
    let changed: Vec<&str> = da
        .sections
        .iter()
        .zip(&db.sections)
        .filter(|(x, y)| x.blake3 != y.blake3)
        .map(|(x, _)| x.name.as_str())
        .collect();
    assert_eq!(
        changed,
        vec!["X_shard_1"],
        "only the re-coded shard should differ; got {changed:?}"
    );

    let msg = std::panic::catch_unwind(|| assert_digests_eq(&da, &db))
        .expect_err("differing digests must panic");
    let msg = msg
        .downcast_ref::<String>()
        .expect("panic payload should be a String");
    assert!(
        msg.contains("X_shard_1") && msg.contains("content"),
        "the failure must name the shard and what changed; got:\n{msg}"
    );
}

/// Flipping one mantissa bit of one f32 changes the digest.
///
/// The float shard is the same length either way, so this is the case a
/// length-only or shape-only comparison passes — which is what two of the
/// tree's existing "byte-identical" tests degrade to.
#[test]
fn digest_detects_a_one_bit_float_change() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(&dir, "a.scx", &FixtureOpts::default());
    let b = write(
        &dir,
        "b.scx",
        &FixtureOpts {
            perturb_float_bit: true,
            ..Default::default()
        },
    );

    let (da, db) = (dig(&a), dig(&b));
    let float_a = da.sections.iter().find(|s| s.name == "X_shard_2").unwrap();
    let float_b = db.sections.iter().find(|s| s.name == "X_shard_2").unwrap();
    assert_ne!(
        float_a.blake3, float_b.blake3,
        "a one-bit float change must change the digest"
    );
}

/// The `Provenance` exclusion is narrow: it hides the timestamp and nothing
/// else.
///
/// The second half is the one that matters. Without it, an exclusion list that
/// had quietly grown to cover half the file would still pass this test — and
/// the harness would be reporting "identical" about regions it no longer looks
/// at.
#[test]
fn digest_ignores_a_provenance_timestamp_and_nothing_more() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(&dir, "a.scx", &FixtureOpts::default());
    let b = write(
        &dir,
        "b.scx",
        &FixtureOpts {
            provenance_timestamp: 1_900_000_000,
            ..Default::default()
        },
    );

    assert_digests_eq(&dig(&a), &dig(&b));

    // Premise: the two files genuinely are different on disk. If they were not,
    // the assertion above would be vacuous.
    assert_ne!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "premise: the timestamp must actually reach the file"
    );

    // And the exclusion covers only provenance: include it and they differ.
    let inc = |p: &std::path::Path| digest_file_excluding(p, Strictness::Content, &[]).unwrap();
    assert_ne!(
        inc(&a).sections,
        inc(&b).sections,
        "including Provenance must surface the difference the default hides"
    );
}

/// `Strictness::Content` survives a section moving; `Strictness::Layout` does
/// not. Two knobs that would be indistinguishable if either were wired wrong.
#[test]
fn layout_strictness_pins_offsets_and_content_does_not() {
    let dir = tempfile::tempdir().unwrap();
    // A longer provenance string shifts every later section's offset without
    // changing any of their bytes.
    let a = write(&dir, "a.scx", &FixtureOpts::default());
    let b = write(
        &dir,
        "b.scx",
        &FixtureOpts {
            provenance_timestamp: 1_900_000_000,
            ..Default::default()
        },
    );
    assert_digests_eq(&dig(&a), &dig(&b));

    let la = digest_file(&a, Strictness::Layout).unwrap();
    assert!(
        la.sections.iter().all(|s| s.offset.is_some()),
        "Layout must record offsets"
    );
    assert!(
        dig(&a).sections.iter().all(|s| s.offset.is_none()),
        "Content must not record offsets"
    );

    // Digests taken at different strictness are refused rather than silently
    // compared on their common fields.
    let err = std::panic::catch_unwind(|| assert_digests_eq(&la, &dig(&b)))
        .expect_err("mixed strictness must panic");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("strictness"), "got:\n{msg}");
}

/// A corrupt section is an error, not a digest.
///
/// The digest hashes the section bytes and cross-checks the catalog's stored
/// checksum. Trusting the stored value would make the harness blind to a
/// writer that stamps the wrong one — a defect a rewrite refactor can plausibly
/// introduce, and one that would then be invisible in both arms of the A/B.
#[test]
fn a_section_whose_stored_checksum_lies_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = mixed_codec_file(&dir.path().join("f.scx")).unwrap();

    let (off, len) = {
        let r = ScxReader::open(&path).unwrap();
        let e = r
            .catalog()
            .entries
            .iter()
            .find(|e| e.name == "X_shard_0")
            .unwrap();
        (e.offset as usize, e.length as usize)
    };
    let mut bytes = std::fs::read(&path).unwrap();
    // Flip a byte inside the section payload, past the shard header, leaving
    // the catalog's stored checksum claiming the original content.
    bytes[off + len - 1] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let err = digest_file(&path, Strictness::Content).expect_err("must not digest a corrupt file");
    assert!(
        matches!(err, ScxError::ChecksumMismatch { ref section } if section == "X_shard_0"),
        "got {err:?}"
    );
}

/// A golden round-trips, a changed file fails against it, and a first run
/// cannot pass by writing its own expectation.
#[test]
fn golden_round_trips_and_a_first_run_cannot_pass() {
    let dir = tempfile::tempdir().unwrap();
    let path = mixed_codec_file(&dir.path().join("f.scx")).unwrap();
    let golden = dir.path().join("nested/f.digest.json");

    let first = std::panic::catch_unwind(|| {
        assert_matches_golden(&path, &golden, Strictness::Content).unwrap()
    })
    .expect_err("a missing golden must be written AND fail");
    let msg = first.downcast_ref::<String>().unwrap();
    assert!(msg.contains("did not exist"), "got:\n{msg}");
    assert!(golden.exists(), "the golden should have been written");

    // Now it passes.
    assert_matches_golden(&path, &golden, Strictness::Content).unwrap();

    // And a changed file does not.
    let changed = write(
        &dir,
        "changed.scx",
        &FixtureOpts {
            perturb_float_bit: true,
            ..Default::default()
        },
    );
    let err = std::panic::catch_unwind(|| {
        assert_matches_golden(&changed, &golden, Strictness::Content).unwrap()
    })
    .expect_err("a changed file must fail against its golden");
    let msg = err.downcast_ref::<String>().unwrap();
    assert!(msg.contains("X_shard_2"), "got:\n{msg}");
}
