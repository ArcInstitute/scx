// B4/P8 regression suite: `codec="auto"` must be genuinely adaptive on every
// derived-file op, not silently `fast`.
//
// The bug: every rewriting op built `FramingConfig::default()`, whose
// `decode_target: None` IS the `fast` profile, and `ScxWriter::write_shard_inner`
// ignored `decode_target` outright. So a `shufdelta` input was re-encoded with
// the `Scx1`/`Zstd` heuristic — roughly 2x bytes/nnz. `scx compact` GREW an
// 864 MiB file by 5.1% while claiming to reclaim 562 MB of orphaned bytes, and
// `scx sort --by` turned it into 1.7 GiB.
//
// Every `*_auto_adopts_*` test below fails before that fix, and the suite proves
// that itself rather than asserting it: `decode_target: None` — what the old code
// hardcoded — is exactly the `fast` profile, so each `*_fast_keeps_the_heuristic`
// test pins the pre-fix output (Zstd) while its `*_auto_adopts_*` twin pins the
// post-fix output (ShufDeltaZstd) for the same op on the same input. The pair is
// a permanent before/after, so a regression that reverts the threading turns the
// `auto` half red while the `fast` half stays green.
//
// Structure mirrors `framing_preservation.rs` (the sibling F-d suite).

use std::path::{Path, PathBuf};

use scx_codec::{CodecId, ValueEncoding};
use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::header::{FileHeader, CURRENT_FORMAT_VERSION};
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{FramingConfig, ResolvedCodec, ScxReader};

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

const N_ROWS: usize = 2048;
const NNZ_PER_ROW: usize = 60;
const N_VARS: u32 = 20_000;
/// Values up to 255 → floor median > 8 → the heuristic picks `Zstd`, and the
/// strided sorted indices make `ShufDeltaZstd` win by far more than
/// `ADOPT_MARGIN` (5%). Both halves matter: a low-median fixture would pick
/// `Scx1` and a random-index fixture would not favour delta coding, either of
/// which makes every assertion below vacuous. Pinned by
/// `adaptive_fixture_actually_favours_shufdelta`.
const MAX_COUNT: u32 = 255;

/// Port of the `#[cfg(test)]`-private `gen_int_shard` in
/// `scx-format-io/src/encoder.rs` (kept in sync by
/// `adaptive_fixture_actually_favours_shufdelta`, which asserts the same
/// heuristic-vs-adaptive outcome that file's unit tests do).
fn gen_int_shard(
    n_rows: usize,
    nnz: usize,
    n_cols: u32,
    max_count: u32,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::with_capacity(n_rows * nnz);
    let mut values = Vec::with_capacity(n_rows * nnz);
    indptr.push(0u64);
    let mut state: u64 = 0x1234_5678_9abc_def0;
    for _ in 0..n_rows {
        let stride = (n_cols as usize / nnz.max(1)).max(1);
        let mut col = 0usize;
        for _ in 0..nnz {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            col += 1 + (state as usize % stride);
            if col >= n_cols as usize {
                break;
            }
            indices.push(col as u32);
            values.push((1 + (state % max_count as u64) as u32) as f32);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

fn obs_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("cell_id", DataType::Utf8, false),
        Field::new("group", DataType::Utf8, false),
        Field::new("n_counts", DataType::Int32, false),
    ]));
    let ids: Vec<String> = (0..n).map(|i| format!("cell{i}")).collect();
    // Two values so `sort --by group` has something to order on.
    let groups: Vec<&str> = (0..n).map(|i| if i % 2 == 0 { "a" } else { "b" }).collect();
    let counts: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(groups)),
            Arc::new(Int32Array::from(counts)),
        ],
    )
    .unwrap()
}

fn var_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "gene_id",
        DataType::Utf8,
        false,
    )]));
    let ids: Vec<String> = (0..n).map(|i| format!("g{i}")).collect();
    RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(ids))]).unwrap()
}

/// Write a v4 framed input whose X shards are `Zstd` — i.e. exactly what the
/// heuristic picks, so any later `ShufDeltaZstd` in the output is proof the
/// adaptive path ran rather than an artefact of the input.
///
/// `framed = false` writes a legacy v3 unframed file instead.
fn write_input(path: &Path, framed: bool) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let (indptr, indices, values) = gen_int_shard(N_ROWS, NNZ_PER_ROW, N_VARS, MAX_COUNT);
    let mut header = FileHeader::new_single_modality(
        N_ROWS as u64,
        N_VARS as u64,
        0,
        N_ROWS as u32,
        0,
        1, // u32 index dtype (n_vars > 65535 is false here, but explicit is fine)
    );
    header.index_dtype = 0;
    if framed {
        header.format_version = CURRENT_FORMAT_VERSION;
    }
    let mut writer = ScxWriter::new(path, header).unwrap();
    if framed {
        // `decode_target: None` — write the input at the plain heuristic so the
        // fixture starts at Zstd. The ops under test supply their own framing.
        writer.set_framing(Some(FramingConfig::default()));
    }
    writer.write_obs(&obs_batch(N_ROWS)).unwrap();
    writer.write_var(&var_batch(N_VARS as usize)).unwrap();
    let raw = scx_codec::values_to_raw_bytes(&values, ValueEncoding::Uint8).unwrap();
    writer
        .write_csr_shard(
            &indptr,
            &indices,
            &raw,
            CodecId::Zstd,
            ValueEncoding::Uint8,
            0,
        )
        .unwrap();
    writer.finish().unwrap();
    (indptr, indices, values)
}

fn x_shards(reader: &ScxReader) -> Vec<&FullCatalogEntry> {
    reader
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CsrShard)
        .collect()
}

/// Every X shard's codec id, plus their total encoded bytes.
fn x_codecs_and_bytes(path: &Path) -> (Vec<u8>, u64) {
    let reader = ScxReader::open(path).unwrap();
    let shards = x_shards(&reader);
    assert!(!shards.is_empty(), "no X shards in {}", path.display());
    let codecs: Vec<u8> = shards
        .iter()
        .map(|e| reader.read_shard_header(e).unwrap().codec_id)
        .collect();
    let bytes = scx_format_io::total_shard_bytes(&shards);
    (codecs, bytes)
}

/// Assert every X shard uses `want`. The codec id is the exact mechanism under
/// test; `assert_smaller_than` covers the user-visible symptom separately.
fn assert_all_x_codec(path: &Path, want: CodecId, ctx: &str) {
    let (codecs, _) = x_codecs_and_bytes(path);
    for (i, c) in codecs.iter().enumerate() {
        assert_eq!(
            *c, want as u8,
            "{ctx}: X shard {i} is codec {c}, expected {:?} ({}). All: {codecs:?}",
            want, want as u8
        );
    }
}

fn intent(s: &str) -> ResolvedCodec {
    scx_format_io::resolve_codec(Some(s)).unwrap()
}

fn tmp() -> tempfile::TempDir {
    tempfile::tempdir().unwrap()
}

// ---------------------------------------------------------------------------
// Test 0 — the guard that keeps every assertion below meaningful
// ---------------------------------------------------------------------------

#[test]
fn adaptive_fixture_actually_favours_shufdelta() {
    // If this fails, the generator has drifted and every `*_auto_adopts_*` test
    // below is vacuously green: `auto` would be "adopting" nothing because the
    // heuristic already won. Assert the gap explicitly, both directions.
    let (indptr, indices, values) = gen_int_shard(N_ROWS, NNZ_PER_ROW, N_VARS, MAX_COUNT);
    let raw = scx_codec::values_to_raw_bytes(&values, ValueEncoding::Uint8).unwrap();
    let seed = scx_format_io::codec_select::select_codec(&raw, ValueEncoding::Uint8);
    assert_eq!(
        seed,
        CodecId::Zstd,
        "fixture must land on the Zstd heuristic (median > 8), got {seed:?}"
    );

    let framed = |dt| FramingConfig {
        row_group_rows: 256,
        target_nnz: None,
        trial: false,
        decode_target: dt,
    };
    let run = |dt| {
        scx_format_io::encode_shard_adaptive(
            &indptr,
            &indices,
            &raw,
            seed,
            ValueEncoding::Uint8,
            true,
            Some(framed(dt)),
        )
        .unwrap()
    };

    let (h_enc, _, _, h_codec) = run(None);
    let (s_enc, _, _, s_codec) = run(Some(scx_format_io::codec_select::DecodeTarget::Auto));
    assert_eq!(h_codec, CodecId::Zstd, "fast must keep the heuristic");
    assert_eq!(
        s_codec,
        CodecId::ShufDeltaZstd,
        "auto must adopt ShufDeltaZstd on this fixture"
    );

    let size = |e: &scx_codec::EncodedShard| {
        e.indptr_bytes.len() + e.indices_bytes.len() + e.values_bytes.len()
    };
    let (h, s) = (size(&h_enc), size(&s_enc));
    assert!(
        (s as f64) < (h as f64) * 0.95,
        "ShufDeltaZstd must beat the heuristic by more than ADOPT_MARGIN, \
         else `auto` legitimately declines to adopt: heuristic={h} shufdelta={s}"
    );
}

// ---------------------------------------------------------------------------
// compact
// ---------------------------------------------------------------------------

fn compact_with(input: &Path, out: &Path, codec: &str) {
    scx_ops::compact_with_options(
        input,
        out,
        &scx_ops::CompactOptions {
            codec: intent(codec),
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn compact_auto_adopts_shufdelta_and_shrinks_x() {
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let (in_codecs, in_bytes) = x_codecs_and_bytes(&inp);
    assert!(in_codecs.iter().all(|c| *c == CodecId::Zstd as u8));

    let out = d.path().join("out.scx");
    compact_with(&inp, &out, "auto");

    assert_all_x_codec(&out, CodecId::ShufDeltaZstd, "compact --codec auto");
    let (_, out_bytes) = x_codecs_and_bytes(&out);
    assert!(
        out_bytes < in_bytes,
        "compact must not grow X: {in_bytes} -> {out_bytes} bytes"
    );
}

#[test]
fn compact_fast_keeps_the_heuristic() {
    // Proves the axis is threaded rather than the adaptive path hardcoded.
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let out = d.path().join("out.scx");
    compact_with(&inp, &out, "fast");
    assert_all_x_codec(&out, CodecId::Zstd, "compact --codec fast");
}

#[test]
fn compact_explicit_codec_is_honoured_over_the_adaptive_pick() {
    // The writer-side adopt must never override an explicit force.
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let out = d.path().join("out.scx");
    compact_with(&inp, &out, "scx1");
    assert_all_x_codec(&out, CodecId::Scx1, "compact --codec scx1");
}

#[test]
fn compact_auto_degrades_on_an_unframed_input_but_compact_profile_errors() {
    let d = tmp();
    let inp = d.path().join("v3.scx");
    write_input(&inp, false);

    // `auto` documents a silent fallback to the single-encode heuristic.
    let out = d.path().join("auto.scx");
    compact_with(&inp, &out, "auto");
    assert_all_x_codec(&out, CodecId::Zstd, "compact auto on v3");

    // `compact` requires framing, so it must say so rather than quietly run
    // `fast` — the exact silent-degradation class this suite exists for.
    let out2 = d.path().join("compact.scx");
    let err = scx_ops::compact_with_options(
        &inp,
        &out2,
        &scx_ops::CompactOptions {
            codec: intent("compact"),
            ..Default::default()
        },
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("framed"), "names the requirement: {msg}");
    assert!(msg.contains("scx optimize"), "names a remedy: {msg}");
}

// ---------------------------------------------------------------------------
// merge
// ---------------------------------------------------------------------------

fn merge_with(inputs: &[PathBuf], out: &Path, codec: &str, force_slow_path: bool) {
    let refs: Vec<&Path> = inputs.iter().map(|p| p.as_path()).collect();
    scx_ops::merge_with_options(
        &refs,
        out,
        &scx_ops::MergeOptions {
            codec: intent(codec),
            // Merge's raw-copy fast path preserves the source codec
            // byte-for-byte under `auto`/`fast` (size-neutral and free), so it
            // must be bypassed to exercise the decode-encode path this bug lived
            // on. Precedent: streaming_merge_append.rs.
            assume_identical_var: force_slow_path,
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn merge_auto_adopts_shufdelta_on_the_re_encode_path() {
    let d = tmp();
    let a = d.path().join("a.scx");
    let b = d.path().join("b.scx");
    write_input(&a, true);
    write_input(&b, true);
    let out = d.path().join("out.scx");
    merge_with(&[a, b], &out, "auto", true);
    assert_all_x_codec(
        &out,
        CodecId::ShufDeltaZstd,
        "merge --codec auto (slow path)",
    );
}

#[test]
fn merge_fast_keeps_the_heuristic_on_the_re_encode_path() {
    let d = tmp();
    let a = d.path().join("a.scx");
    let b = d.path().join("b.scx");
    write_input(&a, true);
    write_input(&b, true);
    let out = d.path().join("out.scx");
    merge_with(&[a, b], &out, "fast", true);
    assert_all_x_codec(&out, CodecId::Zstd, "merge --codec fast (slow path)");
}

#[test]
fn merge_raw_copy_preserves_the_source_codec_under_auto() {
    // The fast path is *supposed* to keep the input's codec: concatenation is
    // size-neutral and re-deciding a codec the source already chose is pure
    // cost. Pinned so the B4 fix is not "corrected" into re-encoding here.
    let d = tmp();
    let a = d.path().join("a.scx");
    let b = d.path().join("b.scx");
    write_input(&a, true);
    write_input(&b, true);
    let out = d.path().join("out.scx");
    merge_with(&[a, b], &out, "auto", false);
    assert_all_x_codec(&out, CodecId::Zstd, "merge --codec auto (raw-copy path)");
}

// ---------------------------------------------------------------------------
// sort
// ---------------------------------------------------------------------------

fn sort_with(input: &Path, out: &Path, codec: &str) {
    let opts = scx_ops::SortOptions {
        by: vec!["group".to_string()],
        codec: intent(codec),
        ..Default::default()
    };
    scx_ops::sort_engine::sort(input, out, &opts).unwrap();
}

#[test]
fn sort_by_key_auto_adopts_shufdelta_and_does_not_grow_x() {
    // The measured symptom: `scx sort --by cell_type` on an 864 MiB shufdelta
    // atlas produced 1.7 GiB. A key sort groups like rows together, so it should
    // compress at least as well as the input, never ~2x worse.
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let (_, in_bytes) = x_codecs_and_bytes(&inp);

    let out = d.path().join("out.scx");
    sort_with(&inp, &out, "auto");

    assert_all_x_codec(&out, CodecId::ShufDeltaZstd, "sort --by --codec auto");
    let (_, out_bytes) = x_codecs_and_bytes(&out);
    assert!(
        out_bytes < in_bytes,
        "sort --by must not grow X: {in_bytes} -> {out_bytes} bytes"
    );
}

#[test]
fn sort_fast_keeps_the_heuristic() {
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let out = d.path().join("out.scx");
    sort_with(&inp, &out, "fast");
    assert_all_x_codec(&out, CodecId::Zstd, "sort --codec fast");
}

#[test]
fn sort_explicit_codec_is_honoured() {
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let out = d.path().join("out.scx");
    sort_with(&inp, &out, "scx1");
    assert_all_x_codec(&out, CodecId::Scx1, "sort --codec scx1");
}

// ---------------------------------------------------------------------------
// The preservation contract: build_csc must NOT re-select
// ---------------------------------------------------------------------------

#[test]
fn build_csc_preserves_per_shard_csr_codec() {
    // `build_csc` re-writes every CSR shard at the codec read off the source
    // header. `framing_for_file()` therefore passes `decode_target: None` on
    // purpose — see its doc comment. If someone "makes it consistent" with the
    // derived-file ops, the writer starts re-selecting and this fails.
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let (before, _) = x_codecs_and_bytes(&inp);

    let out = d.path().join("csc.scx");
    scx_ops::build_csc::run_build_csc(
        &inp,
        &out,
        "4G",
        false,
        5000,
        // Exactly what `scx-cli`'s `framing_for_file()` passes.
        Some(FramingConfig::default()),
    )
    .unwrap();

    let (after, _) = x_codecs_and_bytes(&out);
    assert_eq!(
        before, after,
        "build_csc must preserve each CSR shard's codec byte-for-byte"
    );
}

/// The hazard `framing_for_file()` exists to prevent — pinned so the
/// preservation test above is not vacuous.
///
/// If a caller hands the *rewrite* framing to a CSC rebuild, the writer
/// re-selects each shard's codec instead of honouring the source header. Both
/// `scx subset --rebuild-csc` and `scx convert --csc` did exactly that until it
/// was caught in review: harmless while `write_shard_inner` ignored
/// `decode_target`, a silent override once it honoured it. This test asserts the
/// override is real, so "preserve" and "re-select" cannot quietly become the
/// same thing.
#[test]
fn a_csc_rebuild_carrying_decode_target_does_override_the_codec() {
    let d = tmp();
    let inp = d.path().join("in.scx");
    write_input(&inp, true);
    let (before, _) = x_codecs_and_bytes(&inp);
    assert_eq!(
        before[0],
        CodecId::Zstd as u8,
        "fixture must start at the heuristic pick, else this proves nothing"
    );

    // What the buggy sites passed: the rewrite framing under `--codec auto`.
    let rewrite_framing = scx_ops::framing_for_rewrite(ResolvedCodec::AUTO, true, "the input")
        .unwrap()
        .expect("framed output");
    assert!(
        rewrite_framing.decode_target.is_some(),
        "the whole point of this fixture is that the rewrite framing carries it"
    );

    let out = d.path().join("csc_overridden.scx");
    scx_ops::build_csc::run_build_csc(&inp, &out, "4G", false, 5000, Some(rewrite_framing))
        .unwrap();

    let (after, _) = x_codecs_and_bytes(&out);
    assert_ne!(
        before, after,
        "a CSC rebuild given `decode_target: Some(_)` re-selects the CSR codec — \
         which is why every `rebuild_csc_inplace` caller must route through \
         `framing_for_file()` / `framing_preserving_codec()` instead"
    );
    assert_eq!(
        after[0],
        CodecId::ShufDeltaZstd as u8,
        "and it adopts the adaptive pick"
    );
}

/// The *other* way a CSC rebuild can damage the file it is extending: passing
/// `None`, which rewrites CSR + CSC unframed and strips row-group framing from
/// the X that was just written.
///
/// `framing_for_csc_rebuild` exists because this rule has been got wrong in both
/// directions — `subset`/`convert --csc` passed the rewrite framing (codec
/// override, pinned above), and pyscx's `sort`/`shuffle`/`from_anndata` passed
/// `None` (this downgrade). Asserting both halves keeps the helper from being
/// "simplified" back into either mistake.
#[test]
fn framing_for_csc_rebuild_preserves_v4_and_re_selects_nothing() {
    let d = tmp();
    let framed = d.path().join("framed.scx");
    write_input(&framed, true);
    let (before, _) = x_codecs_and_bytes(&framed);

    let chosen = scx_ops::framing_for_csc_rebuild(&framed)
        .expect("a v4 target must keep framing, not fall back to None");
    assert!(
        chosen.decode_target.is_none() && !chosen.trial,
        "it must not authorise codec re-selection"
    );

    let out = d.path().join("csc.scx");
    scx_ops::build_csc::run_build_csc(&framed, &out, "4G", false, 5000, Some(chosen)).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(
        reader.header().format_version,
        CURRENT_FORMAT_VERSION,
        "passing `None` here is what silently downgraded a framed file to v3"
    );
    let (after, _) = x_codecs_and_bytes(&out);
    assert_eq!(before, after, "and the CSR codecs are untouched");

    // An unframed input has nothing to preserve, so `None` is right there.
    let unframed = d.path().join("v3.scx");
    write_input(&unframed, false);
    assert!(scx_ops::framing_for_csc_rebuild(&unframed).is_none());
}

// ---------------------------------------------------------------------------
// The preservation contract, part 2: build_csc across SEVERAL CSR shards
// ---------------------------------------------------------------------------
//
// `build_csc_preserves_per_shard_csr_codec` above writes ONE CSR shard, so it
// cannot tell "each shard's own codec" from "shard 0's codec for every shard".
// Nor can any other build-csc test in the tree: the three named
// `..._multi_shard_...` in `build_csc.rs` are multi-*CSC*-shard over a single
// CSR shard, and the one genuinely multi-CSR-shard test
// (`scx-cli/tests/cli_ops_integration.rs::build_csc_carries_predicate_index_and_pushdown`)
// builds its shards with `subset --shard-size`, so they all share one codec and
// one encoding and it never compares X content.
//
// `op_output_identity.rs`'s two build_csc arms cover most of the rest, and this
// test is the local, in-words statement of what they pin numerically — except
// for one thing they cannot see. Measured, not assumed:
//
// * Its `build_csc` arm runs over `scx_testkit::fixtures::mixed_codec_file`,
//   whose three shards differ in both codec and encoding — so the entry-to-shard
//   pairing, each shard's `row_start` and each shard's codec/encoding survive
//   there. But that fixture always contains a `Float32` shard, so
//   `widest_value_encoding` answers `Float32` and the CSC pick short-circuits to
//   `(Float32, Pcodec)`: the integer arm is never evaluated on it.
// * Its `build_csc_indexed` arm *does* reach the integer arm — `compact` writes
//   that input's two X shards at `Scx1` and `Zstd` respectively, both `Uint8`
//   with `value_max <= 10`. So "the CSC sidecar takes the FIRST shard's codec"
//   is pinned cross-commit after all.
// * What neither arm pins is the **widest declared integer encoding across
//   shards** (SCX-004): `compact` picks one encoding for every output shard, so
//   that input's shards cannot disagree about it. A build-csc that dropped the
//   declared-encoding half of the scan and trusted `stats.value_max` alone
//   leaves both golden arms unmoved.
//
// Hence this fixture: all-integer so the integer arm runs, three shards with
// irregular row counts and pairwise-distinct codecs, per-shard-distinct
// payloads, and — the part only this test can see — one shard whose declared
// encoding is wider than any value in the whole file.

/// Row counts per shard. Irregular, with the one-row shard in the middle: equal
/// shard sizes make an off-by-one in a running row offset invisible.
const MS_SHARD_ROWS: [usize; 3] = [3, 1, 5];
const MS_N_VARS: usize = 12;

/// `(codec, declared encoding)` per shard. Shard 1 declares `Uint16` while
/// every value in the fixture fits in a `u8` — that is what makes the
/// declared-encoding half of the scan observable, since the `stats.value_max`
/// half alone would pick `Uint8`.
const MS_SPECS: [(CodecId, ValueEncoding); 3] = [
    (CodecId::None, ValueEncoding::Uint8),
    (CodecId::Zstd, ValueEncoding::Uint16),
    (CodecId::Scx1, ValueEncoding::Uint8),
];

/// Two nnz per row, at shard-distinct columns and shard-distinct values.
///
/// Values start at 1 and stay <= 200: never 0, because `Scx1`'s Rice arm
/// rejects zero values, and never > 255, because the whole point of shard 1's
/// `Uint16` declaration is that no *value* justifies it.
fn ms_shard_rows(shard: usize) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let base_col = shard * 3;
    let base_val = 1 + shard * 50;
    let mut indptr = vec![0u64];
    let mut indices = Vec::new();
    let mut values = Vec::new();
    for r in 0..MS_SHARD_ROWS[shard] {
        indices.push(((base_col + r) % MS_N_VARS) as u32);
        indices.push(((base_col + r + 1) % MS_N_VARS) as u32);
        values.push((base_val + 2 * r) as f32);
        values.push((base_val + 2 * r + 1) as f32);
        // Sorted column indices within the row (the v3 canonical invariant).
        let n = indices.len();
        if indices[n - 2] > indices[n - 1] {
            indices.swap(n - 2, n - 1);
            values.swap(n - 2, n - 1);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// Write the three-shard all-integer input described above.
fn write_multi_shard_integer_input(path: &Path) {
    let n_obs: usize = MS_SHARD_ROWS.iter().sum();
    let mut header =
        FileHeader::new_single_modality(n_obs as u64, MS_N_VARS as u64, 0, n_obs as u32, 0, 0);
    header.index_dtype = 0;
    let mut writer = ScxWriter::new(path, header).unwrap();
    writer.write_obs(&obs_batch(n_obs)).unwrap();
    writer.write_var(&var_batch(MS_N_VARS)).unwrap();

    let mut row_start = 0u64;
    for (shard, &(codec, encoding)) in MS_SPECS.iter().enumerate() {
        let (indptr, indices, values) = ms_shard_rows(shard);
        let raw = scx_codec::values_to_raw_bytes(&values, encoding).unwrap();
        writer
            .write_csr_shard(&indptr, &indices, &raw, codec, encoding, row_start)
            .unwrap();
        row_start += MS_SHARD_ROWS[shard] as u64;
    }
    writer.finish().unwrap();
}

/// Every X shard's `(codec_id, value_encoding, row_start)`, in catalog-sorted
/// order, plus each shard's decoded payload.
#[allow(clippy::type_complexity)]
fn x_shard_fingerprints(path: &Path) -> (Vec<(u8, u8, u64)>, Vec<(Vec<i64>, Vec<i32>, Vec<f32>)>) {
    let reader = ScxReader::open(path).unwrap();
    let entries = reader.catalog().csr_shards_sorted();
    let mut meta = Vec::new();
    let mut payloads = Vec::new();
    for e in &entries {
        let sh = reader.read_shard_header(e).unwrap();
        let row_start = e.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
        meta.push((sh.codec_id, sh.value_encoding, row_start));
        payloads.push(reader.read_shard_from_entry(e).unwrap());
    }
    (meta, payloads)
}

/// The CSC sidecar's `(codec_id, value_encoding)` — one pair, asserted equal
/// across every CSC shard so a single-shard layout is not silently assumed.
fn csc_codec_and_encoding(path: &Path) -> (u8, u8) {
    let reader = ScxReader::open(path).unwrap();
    let entries = reader.catalog().csc_shards_sorted();
    assert!(!entries.is_empty(), "no CSC shards in {}", path.display());
    let pairs: Vec<(u8, u8)> = entries
        .iter()
        .map(|e| {
            let sh = reader.read_shard_header(e).unwrap();
            (sh.codec_id, sh.value_encoding)
        })
        .collect();
    assert!(
        pairs.windows(2).all(|w| w[0] == w[1]),
        "build-csc picks ONE codec/encoding for the whole sidecar: {pairs:?}"
    );
    pairs[0]
}

#[test]
fn build_csc_over_shards_that_differ_keeps_each_shards_codec_and_widens_the_sidecar() {
    let d = tmp();
    let input = d.path().join("multi_int.scx");
    write_multi_shard_integer_input(&input);

    // --- premises: without these the assertions below are vacuous ----------
    let (in_meta, in_payloads) = x_shard_fingerprints(&input);
    assert_eq!(in_meta.len(), 3, "fixture must write three CSR shards");
    let in_codecs: Vec<u8> = in_meta.iter().map(|m| m.0).collect();
    assert!(
        in_codecs[0] != in_codecs[1]
            && in_codecs[1] != in_codecs[2]
            && in_codecs[0] != in_codecs[2],
        "the three shards' codecs must be pairwise distinct, else 'each shard's own codec' \
         and 'shard 0's codec for all' agree and this test proves nothing: {in_codecs:?}"
    );
    let in_encs: Vec<u8> = in_meta.iter().map(|m| m.1).collect();
    assert!(
        in_encs.windows(2).any(|w| w[0] != w[1]),
        "the shards' declared value encodings must differ: {in_encs:?}"
    );
    assert!(
        in_payloads[0].2 != in_payloads[1].2
            && in_payloads[1].2 != in_payloads[2].2
            && in_payloads[0].2 != in_payloads[2].2,
        "the shards' payloads must be pairwise distinct, else a reversed or constant \
         entry-to-shard pairing is invisible"
    );
    // The declared-encoding half of the scan is the ONLY thing that can widen
    // the sidecar past Uint8 on this fixture.
    let max_value = in_payloads
        .iter()
        .flat_map(|p| p.2.iter().copied())
        .fold(0.0f32, f32::max);
    assert!(
        max_value <= u8::MAX as f32,
        "every value must fit in a u8, so `stats.value_max` alone would pick Uint8 \
         (max seen: {max_value})"
    );
    let reader = ScxReader::open(&input).unwrap();
    assert!(
        !reader.header().has_csc(),
        "fixture must start without a sidecar"
    );
    let in_dense = reader.read_all_csr_shards().unwrap().to_dense().unwrap();
    drop(reader);

    // --- the op -----------------------------------------------------------
    let output = d.path().join("multi_int_csc.scx");
    scx_ops::build_csc::run_build_csc(&input, &output, "4G", false, 5000, None).unwrap();

    // 1 + 2 + 3. Each output shard carries its OWN codec, its OWN declared
    // encoding, its OWN row_start, and its OWN payload.
    let (out_meta, out_payloads) = x_shard_fingerprints(&output);
    assert_eq!(
        out_meta, in_meta,
        "build-csc re-emits one output CSR shard per input shard: each must keep that \
         shard's codec, declared value encoding and row_start"
    );
    assert_eq!(
        out_payloads, in_payloads,
        "and each output shard must decode to the input shard it was paired with"
    );

    // 4. The CSC sidecar takes the widest DECLARED integer encoding across
    //    shards (Uint16, from shard 1) and the FIRST shard's codec (None).
    let (csc_codec, csc_enc) = csc_codec_and_encoding(&output);
    assert_eq!(
        csc_enc,
        ValueEncoding::Uint16 as u8,
        "the CSC sidecar must be wide enough for every shard's DECLARED encoding \
         (SCX-004); shard 1 declares Uint16 while no value needs it"
    );
    assert_eq!(
        csc_codec, in_codecs[0],
        "the integer arm takes the first shard's codec, and shard 0's differs from \
         shards 1 and 2 so 'first' is distinguishable from 'last'"
    );

    // 5. Content round-trips, both ways.
    let out_reader = ScxReader::open(&output).unwrap();
    let out_dense = out_reader
        .read_all_csr_shards()
        .unwrap()
        .to_dense()
        .unwrap();
    assert_eq!(out_dense, in_dense, "build-csc must not change X");
    let csc_dense = out_reader
        .read_all_csc_shards()
        .unwrap()
        .to_dense()
        .unwrap();
    assert_eq!(
        csc_dense, in_dense,
        "the CSC transpose must match the CSR data"
    );
}

/// The other half of merge's raw-copy gate: `compact` must turn the fast path
/// OFF and actually re-encode.
///
/// Flagged twice in review as the missing twin. `raw_copy_csr_eligible` has unit
/// coverage, but only the `auto`-keeps-raw-copy side was pinned end to end —
/// so a change that disabled raw-copy for every profile, or enabled it for
/// `compact`, would have gone unnoticed here. Note this passes
/// `force_slow_path: false`: the point is that the *codec intent alone* decides,
/// with nothing else forcing the re-encode.
#[test]
fn merge_compact_disables_raw_copy_and_re_encodes() {
    let d = tmp();
    let a = d.path().join("a.scx");
    let b = d.path().join("b.scx");
    write_input(&a, true);
    write_input(&b, true);
    let (before, _) = x_codecs_and_bytes(&a);
    assert_eq!(
        before[0],
        CodecId::Zstd as u8,
        "fixture must start at the heuristic pick"
    );

    let out = d.path().join("out.scx");
    merge_with(&[a, b], &out, "compact", false);

    // Raw-copy would have carried Zstd through untouched, as the `auto` twin
    // asserts. `compact` adopts on ties, so every shard must have moved.
    assert_all_x_codec(
        &out,
        CodecId::ShufDeltaZstd,
        "merge --codec compact (must re-encode, not raw-copy)",
    );
}
