//! Oracles for the row-group-framed encoder, [`encode_shard_framed`].
//!
//! # Why these exist
//!
//! Before this module, **nothing in the tree pinned the framed layout**. Three
//! things look like they do and none does:
//!
//! * `tests/scx-integration-tests/tests/op_output_identity.rs` reaches
//!   `encode_shard_framed` zero times for every arm but `optimize_framed`,
//!   which was added alongside this module. Its other arms run over fixtures
//!   that never call `ScxWriter::set_framing`, so `framing_for_rewrite` hands
//!   the writer `None` and every output is unframed.
//! * `scx-codec/tests/reference_vectors.rs` pins `scx_codec::encode_shard` —
//!   one group, no framing. It stays green through any change to *this*
//!   function, which makes it evidence of nothing here.
//! * `adaptive_codec_tests::from_bytes_matches_f32_path_byte_for_byte`
//!   compares `encode_one_shard` against `encode_one_shard_from_bytes`: two
//!   callers of this same framed encoder. A change to the concatenation moves
//!   both arms identically and the assertion still passes.
//!
//! So a layout change — a group emitted in a different order, an offset
//! rebased from the wrong base — produced a file that decoded *correctly*
//! (the block index is self-consistent) with different bytes, and no test in
//! the workspace said so. That is what the byte pin below is for.
//!
//! # The two oracles, and why both
//!
//! * [`framed_matches_an_independent_framer`] is a differential against
//!   `reference_framed`, a second implementation written the obvious way. It
//!   catches a wrong rebase, a dropped group, a mis-sized offset — anything
//!   where the two implementations disagree.
//!
//!   Every mutation tried so far reddens the pin as well, so this is not extra
//!   *detection*. What it adds is the ability to tell **which layer moved**:
//!   `reference_framed` calls the same `scx_codec::encode_shard`, so a change
//!   inside a codec moves the pin and leaves the differential green, while a
//!   change to the framing moves both. Without it, a future codec tweak fails
//!   six opaque hashes with nothing to say whether the framing is implicated.
//!   It also reports *which* sub-stream or entry field differs, where the pin
//!   reports one changed hex string.
//! * [`framed_layout_is_byte_pinned`] compares a BLAKE3 of the three
//!   sub-streams and the entry table against a checked-in constant. It catches
//!   the class the differential cannot: a change *both* implementations would
//!   make. A codec whose output bytes shift, a group-header field that starts
//!   being written differently, a `BlockIndexEntry` gaining a value — the
//!   reference framer calls the same `encode_shard` and records offsets the
//!   same way, so it moves with the code and agrees with it.
//!
//! What the pin turned out **not** to be needed for: a permutation of the
//! concatenation that recomputes its offsets to stay self-consistent. There is
//! no such thing. `resolve_block_index` derives each group's byte range as
//! `[offset[g], offset[g + 1])` and validates the offsets monotonic — the rule
//! `docs/format.md` § Block index already states — so the entry table's
//! offsets must ascend with its rows. Reversing the byte order and the entry
//! table together produces an **unreadable** file, not merely a different one:
//! measured, it reddens `decode_shard_regions_framed_matches_unframed` and
//! five backed-read tests. Worth restating here because "the index says where
//! each group is, so emit them in completion order" is the obvious wrong idea
//! about this function, and the spec sentence that forbids it is three files
//! away.
//!
//! The constants were blessed on `0383ea8f`, **before** the parallel encode
//! landed, so they describe the pre-existing layout rather than the new code's
//! opinion of it.

use super::*;

/// Deterministic integer CSR: `n_rows` rows of `nnz_per_row` sorted unique
/// columns, values cycling `1..=max_v` (never 0 — Scx1's Rice coder rejects a
/// zero count, which `zero_value_in_a_later_group_still_errors` uses on
/// purpose).
fn gen_csr(
    n_rows: usize,
    nnz_per_row: usize,
    n_cols: u32,
    max_v: u32,
) -> (Vec<u64>, Vec<u32>, Vec<f32>) {
    let mut indptr = Vec::with_capacity(n_rows + 1);
    let mut indices = Vec::with_capacity(n_rows * nnz_per_row);
    let mut values = Vec::with_capacity(n_rows * nnz_per_row);
    indptr.push(0u64);
    let stride = (n_cols as usize / nnz_per_row.max(1)).max(1);
    for row in 0..n_rows {
        // Sorted and unique per row — what the writer guarantees its encoders.
        let mut cols: Vec<u32> = (0..nnz_per_row)
            .map(|k| ((k * stride + row * 7) % n_cols as usize) as u32)
            .collect();
        cols.sort_unstable();
        cols.dedup();
        for (k, col) in cols.iter().enumerate() {
            indices.push(*col);
            values.push(((row + k) as u32 % max_v + 1) as f32);
        }
        indptr.push(indices.len() as u64);
    }
    (indptr, indices, values)
}

/// One entry, flattened, so a mismatch prints the numbers rather than a struct.
type FlatEntry = (u32, u16, u32, u32, u32, u32);

fn flatten(bi: &BlockIndex) -> Vec<FlatEntry> {
    bi.entries
        .iter()
        .map(|e| {
            (
                e.row_start,
                e.n_rows,
                e.indptr_byte_offset,
                e.indices_byte_offset,
                e.values_byte_offset,
                e.nnz_in_block,
            )
        })
        .collect()
}

/// An independent framer: the layout [`encode_shard_framed`] must produce,
/// written the obvious way — one [`encode_shard`] per group, concatenated in
/// group order, each group's offsets recorded before its bytes are appended.
///
/// Deliberately **does not** implement the `target_nnz` cap, so every caller
/// passes `target_nnz: None`. That is the only production setting (nothing in
/// the workspace sets it), and re-implementing the cap here would copy the
/// growth loop this is supposed to be independent of.
fn reference_framed(
    indptr: &[u64],
    indices: &[u32],
    values_bytes: &[u8],
    codec: CodecId,
    enc: ValueEncoding,
    u16_idx: bool,
    g: usize,
) -> (EncodedShard, Vec<FlatEntry>) {
    let w = enc.byte_width();
    let n_rows = indptr.len() - 1;
    let (mut ip, mut ix, mut vv) = (Vec::new(), Vec::new(), Vec::new());
    let mut entries = Vec::new();
    let mut r0 = 0usize;
    while r0 < n_rows {
        let r1 = (r0 + g).min(n_rows);
        let base = indptr[r0];
        let (s, e) = (base as usize, indptr[r1] as usize);
        let local: Vec<u64> = indptr[r0..=r1].iter().map(|&v| v - base).collect();
        let group = encode_shard(
            &local,
            &indices[s..e],
            &values_bytes[s * w..e * w],
            codec,
            enc,
            u16_idx,
        )
        .expect("reference group encode");
        entries.push((
            r0 as u32,
            (r1 - r0) as u16,
            ip.len() as u32,
            ix.len() as u32,
            vv.len() as u32,
            (indptr[r1] - base) as u32,
        ));
        ip.extend_from_slice(&group.indptr_bytes);
        ix.extend_from_slice(&group.indices_bytes);
        vv.extend_from_slice(&group.values_bytes);
        r0 = r1;
    }
    (
        EncodedShard {
            indptr_bytes: ip,
            indices_bytes: ix,
            values_bytes: vv,
        },
        entries,
    )
}

fn framing(g: u32) -> FramingConfig {
    FramingConfig {
        row_group_rows: g,
        target_nnz: None,
        trial: false,
        decode_target: None,
    }
}

/// Every `CodecId` a framed write can land on, with the u16-index variant on
/// one of them. `Pcodec` is the float path; `Lz4Shuffle` is not merely
/// user-forcible — `select_codec_for_modality` picks it automatically for
/// non-binary integer ATAC peak counts (`scx-format/src/codec_select.rs:255`),
/// which makes it a **dual-encode seed** under `codec="auto"` on any Multiome
/// or TEA-seq write.
///
/// `Float16` is here because a write path *can* produce it, contrary to an
/// earlier version of this comment: `detect_value_encoding` never returns it,
/// but `EncodeShardOptions::value_encoding` overrides that detection
/// (`encoder.rs`'s `unwrap_or_else`), and `scx-ops`' external-layer attach and
/// grouped sort both forward a pinned per-shard encoding through it unchanged.
/// It matters here for one concrete reason: `byte_width() == 2`, so it is the
/// only combination that exercises the `values_bytes[start * w_v..end * w_v]`
/// slicing at a width other than 1 or 4.
fn combos() -> Vec<(CodecId, ValueEncoding, bool)> {
    vec![
        (CodecId::None, ValueEncoding::Uint8, false),
        (CodecId::Scx1, ValueEncoding::Uint32, false),
        (CodecId::Zstd, ValueEncoding::Uint16, true),
        (CodecId::ShufDeltaZstd, ValueEncoding::Uint32, false),
        (CodecId::Lz4Shuffle, ValueEncoding::Uint32, false),
        (CodecId::Pcodec, ValueEncoding::Float32, false),
        (CodecId::Pcodec, ValueEncoding::Float16, false),
    ]
}

/// Differential oracle. Geometry: 37 rows at G = 8 → **5 groups, the last only
/// 5 rows**, so a uniform-group assumption or an off-by-one on the tail is
/// visible. The premise (more than one group) is asserted, because a
/// single-group fixture would make every assertion below vacuous — the shape
/// `mixed_codec_file` documents for its own framed shard.
#[test]
fn framed_matches_an_independent_framer() {
    let (indptr, indices, values) = gen_csr(37, 4, 50, 7);
    for (codec, enc, u16_idx) in combos() {
        let bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");
        let (got, bi) =
            encode_shard_framed(&indptr, &indices, &bytes, codec, enc, u16_idx, framing(8))
                .expect("framed encode");
        assert!(
            bi.entries.len() > 1,
            "{codec:?}/{enc:?}: fixture must span more than one row group, \
             else every assertion here is vacuous"
        );
        assert_eq!(bi.entries.len(), 5, "{codec:?}/{enc:?}: 37 rows at G=8");
        assert_eq!(
            bi.entries.last().unwrap().n_rows,
            5,
            "{codec:?}/{enc:?}: the tail group is short on purpose"
        );

        let (want, want_entries) =
            reference_framed(&indptr, &indices, &bytes, codec, enc, u16_idx, 8);
        assert_eq!(
            got.indptr_bytes, want.indptr_bytes,
            "{codec:?}/{enc:?}: indptr sub-stream"
        );
        assert_eq!(
            got.indices_bytes, want.indices_bytes,
            "{codec:?}/{enc:?}: indices sub-stream"
        );
        assert_eq!(
            got.values_bytes, want.values_bytes,
            "{codec:?}/{enc:?}: values sub-stream"
        );
        assert_eq!(
            flatten(&bi),
            want_entries,
            "{codec:?}/{enc:?}: block index entries"
        );
    }
}

/// BLAKE3 over the three sub-streams (each length-prefixed, so a byte moving
/// across a stream boundary cannot collide) and the entry table.
fn digest_framed(e: &EncodedShard, bi: &BlockIndex) -> String {
    let mut h = blake3::Hasher::new();
    for s in [&e.indptr_bytes, &e.indices_bytes, &e.values_bytes] {
        h.update(&(s.len() as u64).to_le_bytes());
        h.update(s);
    }
    for en in &bi.entries {
        h.update(&en.row_start.to_le_bytes());
        h.update(&en.n_rows.to_le_bytes());
        h.update(&en.indptr_byte_offset.to_le_bytes());
        h.update(&en.indices_byte_offset.to_le_bytes());
        h.update(&en.values_byte_offset.to_le_bytes());
        h.update(&en.nnz_in_block.to_le_bytes());
    }
    h.finalize().to_hex().to_string()
}

/// Byte pin: the one class the differential above cannot see, a change *both*
/// implementations would make (see this module's header).
///
/// Blessed on `0383ea8f` before the parallel encode existed. If a deliberate
/// format change moves these, that is a **wire-format change**: say so in the
/// commit, and check `docs/format.md` § Block index with it.
#[test]
fn framed_layout_is_byte_pinned() {
    // (codec, encoding, u16 indices, expected digest)
    let expected: &[(CodecId, ValueEncoding, bool, &str)] = &[
        (
            CodecId::None,
            ValueEncoding::Uint8,
            false,
            "52ae6205d5f03be17828443c441bff5d3a74b41cdc7942dbe56f7d677f45a258",
        ),
        (
            CodecId::Scx1,
            ValueEncoding::Uint32,
            false,
            "813dbfdc48773c968f435655a608530c5c9573ea5c17284e4963c62aca794e62",
        ),
        (
            CodecId::Zstd,
            ValueEncoding::Uint16,
            true,
            "fc8d014f2403807f0fe751c5799274666750dedce089726c241ace0b588f5935",
        ),
        (
            CodecId::ShufDeltaZstd,
            ValueEncoding::Uint32,
            false,
            "d560fd219dc806741d9ed92f9f16923e9e03b70963738166485be076752eab38",
        ),
        (
            CodecId::Lz4Shuffle,
            ValueEncoding::Uint32,
            false,
            "78b62589452e8728e19d2ae2a890d9cb0ccc65b83737e9662bc86e3c3f521023",
        ),
        (
            CodecId::Pcodec,
            ValueEncoding::Float32,
            false,
            "aa28644acc5bd13166205ed812fe9ef5bb230facd247ac59f890345dbbcb57ca",
        ),
        // Identical to the Float32 digest above, and that is the pin, not a
        // copy-paste slip: `PcodecCodec::encode` widens f16 to f32 and
        // compresses as f32 (`scx-codec/src/codecs/pcodec.rs:70-78`), and this
        // fixture's values are small integers, exactly representable in f16.
        // So the encoded stream *must* match — and if pcodec ever stops
        // widening, these two rows stop agreeing and this is what says so.
        (
            CodecId::Pcodec,
            ValueEncoding::Float16,
            false,
            "aa28644acc5bd13166205ed812fe9ef5bb230facd247ac59f890345dbbcb57ca",
        ),
    ];
    let (indptr, indices, values) = gen_csr(37, 4, 50, 7);
    for &(codec, enc, u16_idx, want) in expected {
        let bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");
        let (e, bi) =
            encode_shard_framed(&indptr, &indices, &bytes, codec, enc, u16_idx, framing(8))
                .expect("framed encode");
        assert!(bi.entries.len() >= 3, "{codec:?}: pin needs several groups");
        assert_eq!(
            digest_framed(&e, &bi),
            want,
            "{codec:?}/{enc:?}: framed layout changed"
        );
    }
}

/// The three concatenated sub-streams are sized exactly, not doubling-grown.
/// A `Vec` that grew by doubling reports a larger capacity than length (the
/// pre-change code measured 320 for a 194-byte indptr stream), so this pins
/// the pre-size rather than merely the contents.
///
/// Kept rather than left to the benchmark, which measures time and cannot see
/// an allocation: the exact sizing is now load-bearing for a **memory** claim.
/// Parallel groups mean the encoded groups and the assembled streams are live
/// together, and `scx-convert/src/budget.rs` prices that transient — a
/// doubling-grown stream would add its overshoot, plus both buffers during the
/// final realloc, on top. `Vec::with_capacity` is documented to allocate *at
/// least* the request, so in principle an allocator could round up and make
/// this fail; measured, it does not on any target this repo builds for, and a
/// failure here would be a signal worth reading rather than noise.
#[test]
fn framed_streams_are_sized_exactly() {
    let (indptr, indices, values) = gen_csr(37, 4, 50, 7);
    let enc = ValueEncoding::Uint32;
    let bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");
    let (e, bi) = encode_shard_framed(
        &indptr,
        &indices,
        &bytes,
        CodecId::Zstd,
        enc,
        false,
        framing(8),
    )
    .expect("framed encode");
    assert!(bi.entries.len() > 1, "premise: more than one group");
    for (name, s) in [
        ("indptr", &e.indptr_bytes),
        ("indices", &e.indices_bytes),
        ("values", &e.values_bytes),
    ] {
        assert_eq!(
            s.len(),
            s.capacity(),
            "{name} sub-stream is over-allocated ({} of {})",
            s.len(),
            s.capacity()
        );
    }
}

/// A group that cannot encode reports its error rather than being dropped or
/// swallowed — and the failing group is deliberately **not** the first, so a
/// concat pass that only ever looks at group 0 fails here.
///
/// Scx1 is the vehicle: its Rice coder rejects a zero count
/// (`scx-codec/src/rice.rs:39`), so a single 0 in group 1 fails that group's
/// encode and no other. The positive control below is what makes the negative
/// meaningful — without it this test would also pass if the fixture were
/// unencodable for some unrelated reason.
///
/// What this does **not** pin: *which* error surfaces when two groups fail.
/// The concat pass takes the first `Err` in group order (what the serial loop
/// returned), but `rice_encode` fails with a payload-free `BitStreamError` and
/// no second data-dependent encode error exists in `scx-codec`, so two failing
/// groups are indistinguishable. Order is deterministic by construction here,
/// not by assertion.
#[test]
fn zero_value_in_a_later_group_still_errors() {
    let (indptr, indices, mut values) = gen_csr(37, 4, 50, 7);
    let enc = ValueEncoding::Uint32;

    // Positive control: the same shard encodes cleanly with no zero.
    let ok_bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");
    let (_, bi) = encode_shard_framed(
        &indptr,
        &indices,
        &ok_bytes,
        CodecId::Scx1,
        enc,
        false,
        framing(8),
    )
    .expect("control: Scx1 encodes a zero-free shard");
    assert!(bi.entries.len() > 1, "premise: more than one group");

    // Row 10 is in group 1 (rows 8..16 at G=8), not group 0.
    let zero_at = indptr[10] as usize;
    assert!(
        zero_at >= indptr[8] as usize && zero_at < indptr[16] as usize,
        "premise: the zero must land in group 1, not group 0"
    );
    values[zero_at] = 0.0;
    let bad_bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");
    let err = encode_shard_framed(
        &indptr,
        &indices,
        &bad_bytes,
        CodecId::Scx1,
        enc,
        false,
        framing(8),
    )
    .expect_err("a zero count in group 1 must fail the shard");
    assert!(
        matches!(err, ScxError::Codec(_)),
        "expected the group's codec error, got {err:?}"
    );
}

/// The `target_nnz` cap's boundary rule, which nothing else in the workspace
/// exercises — no Rust caller sets it, and its only public surface is pyscx's
/// `row_group_target_nnz=` kwarg.
///
/// It is pinned here because it is the one part of the group split that is
/// *data*-dependent: the growth loop admits row `r1` only if the group's nnz
/// **through `r1`** stays within the cap, which means it reads one row past
/// the candidate boundary. Anything that derives boundaries per group in
/// isolation, or that peeks at `indptr[r1]` instead of `indptr[r1 + 1]`, gets
/// a different (still self-consistent, still decodable) split.
///
/// Geometry: 8 rows × 3 nnz, `row_group_rows = 8` so `g` never binds, and
/// `target_nnz = 7` so the cap alone decides — two rows per group, four
/// groups. The row-count split (one group of 8) and the mis-peeked split
/// (groups of 3) are both different, so the assertion distinguishes all three.
#[test]
fn target_nnz_cap_decides_the_group_boundaries() {
    let n_rows = 8usize;
    let per_row = 3usize;
    let indptr: Vec<u64> = (0..=n_rows).map(|r| (r * per_row) as u64).collect();
    let indices: Vec<u32> = (0..n_rows * per_row).map(|i| (i % 40) as u32).collect();
    let values: Vec<f32> = (0..n_rows * per_row).map(|i| (i % 5 + 1) as f32).collect();
    let enc = ValueEncoding::Uint8;
    let bytes = values_to_raw_bytes(&values, enc).expect("values_to_raw_bytes");

    let (_, bi) = encode_shard_framed(
        &indptr,
        &indices,
        &bytes,
        CodecId::None,
        enc,
        false,
        FramingConfig {
            row_group_rows: 8,
            target_nnz: Some(7),
            trial: false,
            decode_target: None,
        },
    )
    .expect("framed encode");

    let split: Vec<(u32, u16, u32)> = bi
        .entries
        .iter()
        .map(|e| (e.row_start, e.n_rows, e.nnz_in_block))
        .collect();
    assert_eq!(
        split,
        vec![(0, 2, 6), (2, 2, 6), (4, 2, 6), (6, 2, 6)],
        "the nnz cap must break every group at 2 rows / 6 nnz"
    );
}
