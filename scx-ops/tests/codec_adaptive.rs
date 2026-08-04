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
