// Shared CSR→CSC sidecar writer.
//
// The transpose-and-write loop used to exist three times: here, inline in
// `scx-ops`' `run_build_csc` (which kept its own copy to drive a progress bar,
// with a NOTE asking the two be kept in sync), and again inside
// `ScxWriter::finish`'s finish-time auto-emit. [`emit_csc_shards`] is now the
// single definition, and it is written against
// [`scx_sparse::CscShardSource`] so the three callers differ only in where
// their records come from and what they want told about each shard.
//
// Two sources, both used:
//
//   * [`scx_sparse::ResidentCscSource`] for a caller that already holds the
//     whole CSR — the four `write_csc_sidecar` callers do, from an eager h5ad
//     ingest, a modality copy, GIL-owned `from_anndata` buffers or decoded R
//     values. Pushing those through the builder's buckets would be a *second*
//     copy at 8 B/nnz, roughly doubling the peak of an ingest path nothing
//     gates.
//   * [`scx_sparse::CscBuilder`] for a caller that decodes shard by shard and
//     must not hold them all, which is `run_build_csc`'s 14.7 GB.
//
// Callers still build their canonical `ScxCsr` shard(s) — the casting and
// canonicalization preamble differs per source, so it stays caller-side.

use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::{CscShardSource, ResidentCscSource, ScxCsr};

use crate::encoder::FramingConfig;
use crate::error::ScxError;
use crate::writer::ScxWriter;

/// Default per-chunk memory budget for the streaming CSC transpose (4 GiB).
/// Bounds the number of columns materialized at once independent of
/// `cols_per_shard`.
pub const DEFAULT_CSC_MEMORY_BYTES: usize = 4 * 1024 * 1024 * 1024;

/// Pick the value encoding and codec for a CSC sidecar built from `encs`.
///
/// The sidecar's encoding must cover **every** source shard, not just the
/// first (SCX-004): a `Uint8` first shard followed by a `Float32` shard would
/// truncate the float values. Two inputs decide it — the shards' *declared*
/// encodings and the largest integer value the catalog reports:
///
/// - any float shard ⇒ `(Float32, Pcodec)`. The codec is not the source's:
///   the first shard's codec may be integer-only (`Scx1`), which cannot
///   represent float values at all.
/// - otherwise the wider of the declared integer width and the width
///   `max_int_val` needs. Flooring on the declared width matters because a
///   shard with no `ShardStats` contributes nothing to `max_int_val` and would
///   otherwise be under-picked as `Uint8`.
///
/// Returns `None` only on the integer path with `first_codec: None`, i.e. no
/// source shards at all — a condition both callers already guard, and which
/// they report in their own words rather than sharing a message.
pub fn pick_csc_encoding(
    encs: &[ValueEncoding],
    max_int_val: u32,
    first_codec: Option<CodecId>,
) -> Option<(ValueEncoding, CodecId)> {
    let declared = ValueEncoding::widest_for_write(encs);
    // `widest_for_write` maps any float shard to `Float32` and never yields
    // `Float16`, so this one predicate is the whole `any_float` test.
    if matches!(declared, ValueEncoding::Float32) {
        return Some((ValueEncoding::Float32, CodecId::Pcodec));
    }
    let by_value = if max_int_val <= u8::MAX as u32 {
        ValueEncoding::Uint8
    } else if max_int_val <= u16::MAX as u32 {
        ValueEncoding::Uint16
    } else {
        ValueEncoding::Uint32
    };
    // Both are integer encodings here, so the float arm cannot fire.
    let enc = ValueEncoding::widest_for_write(&[declared, by_value]);
    Some((enc, first_codec?))
}

/// Configuration options for writing a CSC sidecar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CscSidecarOptions {
    /// Bounds each emitted shard's column count, together with
    /// `memory_budget_bytes` (whichever is smaller).
    pub cols_per_shard: usize,
    pub memory_budget_bytes: usize,
    /// - `None` → single-modality file; shards written via
    ///   [`ScxWriter::write_csc_shard`] (`X_csc_shard_*`).
    /// - `Some(id)` → multimodal file; shards written via
    ///   [`ScxWriter::write_csc_shard_for`] under modality `id`.
    pub modality_id: Option<u8>,
    /// When `Some`, the emitted CSC shards are row-group-framed (shard v2,
    /// column-major — the "row" axis is columns for CSC), enabling the
    /// scattered per-gene-group CSC reader. When `None`, unframed (v1). This
    /// overrides the writer's framing for the scope of the call and restores
    /// it afterward, so CSC framing no longer depends on hidden writer state
    /// (the caller need not `set_framing` first). The caller is responsible
    /// for the file `format_version` being v4 when framing (see
    /// `IngestOptions`/`from_anndata`).
    pub framing: Option<FramingConfig>,
}

impl Default for CscSidecarOptions {
    fn default() -> Self {
        Self {
            cols_per_shard: 5000,
            memory_budget_bytes: DEFAULT_CSC_MEMORY_BYTES,
            modality_id: None,
            framing: None,
        }
    }
}

/// Where an emitted shard is written, and under which encoding.
///
/// Separate from [`CscSidecarOptions`] because the push side and the emit side
/// have different owners on the `build-csc` path: there the caller drives the
/// shard walk and this drives the writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CscEmitOptions {
    pub value_encoding: ValueEncoding,
    pub codec_id: CodecId,
    /// - `None` → single-modality file; [`ScxWriter::write_csc_shard`].
    /// - `Some(id)` → [`ScxWriter::write_csc_shard_for`] under modality `id`.
    pub modality_id: Option<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CscSidecarStats {
    pub n_shards: u32,
    pub total_nnz: u64,
    /// Bytes the source spilled. Zero for a resident source, and the signal
    /// that separates "bounded" from "bounded by thrashing" for a streamed one.
    pub spill_bytes: u64,
    /// See [`scx_sparse::CscBuilderStats::first_non_strict_column`].
    pub first_non_strict_column: Option<usize>,
}

/// Drain a [`CscShardSource`] into the writer, one CSC section per shard.
///
/// `on_shard(index, col_start, nnz)` fires after each write. That callback is
/// the whole reason `run_build_csc` can stop carrying its own copy of this
/// loop: its NOTE said the copy existed "because it drives a progress bar per
/// chunk".
///
/// Framing is **not** scoped here. [`write_csc_sidecar`] scopes it around the
/// whole call, and `run_build_csc` sets it once for both its CSR re-emit and
/// its CSC shards; scoping it a second time inside the drain would restore the
/// writer's previous framing part-way through.
pub fn emit_csc_shards(
    writer: &mut ScxWriter,
    source: &mut dyn CscShardSource,
    opts: &CscEmitOptions,
    mut on_shard: impl FnMut(u32, u64, u64),
) -> Result<CscSidecarStats, ScxError> {
    let (mut indptr, mut indices, mut data) = (Vec::new(), Vec::new(), Vec::new());
    // One raw-value buffer for the whole drain. The predecessor allocated a
    // fresh one per chunk via `values_to_raw_bytes`.
    let mut raw_values: Vec<u8> = Vec::new();
    let mut stats = CscSidecarStats::default();

    loop {
        let Some(col_start) = source
            .next_shard_into(&mut indptr, &mut indices, &mut data)
            .map_err(|e| ScxError::CscTranspose(e.to_string()))?
        else {
            break;
        };
        raw_values.clear();
        opts.value_encoding
            .encode_f32_into(&mut raw_values, &data)?;

        match opts.modality_id {
            Some(mid) => {
                let shard = crate::writer::ShardBuffers::new(
                    &indptr,
                    &indices,
                    &raw_values,
                    opts.codec_id,
                    opts.value_encoding,
                );
                writer.write_csc_shard_for(mid, col_start, shard)?;
            }
            None => writer.write_csc_shard(
                &indptr,
                &indices,
                &raw_values,
                opts.codec_id,
                opts.value_encoding,
                col_start,
            )?,
        }
        on_shard(stats.n_shards, col_start, data.len() as u64);
        stats.n_shards += 1;
        stats.total_nnz += data.len() as u64;
    }

    let src = source.stats();
    stats.spill_bytes = src.spilled_bytes;
    stats.first_non_strict_column = src.first_non_strict_column;
    Ok(stats)
}

/// Transpose `csr_shards` and write the result as a CSC sidecar.
///
/// For a caller that already holds the whole CSR; see the module header for
/// why that does not go through [`scx_sparse::CscBuilder`]. See
/// [`CscSidecarOptions`] for what each option controls.
///
/// `csr_shards` must already be canonical — this helper does not
/// sort/dedup/drop-zeros (callers do so upstream when their source requires
/// it).
pub fn write_csc_sidecar(
    writer: &mut ScxWriter,
    csr_shards: &[ScxCsr],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    opts: CscSidecarOptions,
) -> Result<(), ScxError> {
    // Scope framing to this batch: override, write, restore. CSC framing is thus
    // explicit per-call rather than dependent on prior `set_framing` state.
    let prev_framing = writer.framing();
    writer.set_framing(opts.framing);
    let result = write_csc_sidecar_inner(
        writer,
        csr_shards,
        n_obs,
        n_vars,
        value_encoding,
        codec_id,
        &opts,
    );
    writer.set_framing(prev_framing);
    result.map(|_| ())
}

fn write_csc_sidecar_inner(
    writer: &mut ScxWriter,
    csr_shards: &[ScxCsr],
    n_obs: usize,
    n_vars: usize,
    value_encoding: ValueEncoding,
    codec_id: CodecId,
    opts: &CscSidecarOptions,
) -> Result<CscSidecarStats, ScxError> {
    let mut source = ResidentCscSource::new(
        csr_shards,
        n_obs,
        n_vars,
        opts.cols_per_shard,
        opts.memory_budget_bytes,
    )
    .map_err(|e| ScxError::CscTranspose(e.to_string()))?;
    let emit = CscEmitOptions {
        value_encoding,
        codec_id,
        modality_id: opts.modality_id,
    };
    emit_csc_shards(writer, &mut source, &emit, |_, _, _| {})
}

#[cfg(test)]
mod tests {
    use super::pick_csc_encoding;
    use scx_codec::value_encoding::values_to_raw_bytes;
    use scx_codec::{CodecId, ValueEncoding};

    /// SCX-004 itself: the encoding must cover every shard, so one float shard
    /// after an integer one takes the whole sidecar to `Float32` — and to
    /// `Pcodec`, *not* the source's codec, because `Scx1` is integer-only and
    /// cannot represent a float value at all. Passing `Scx1` here is the point
    /// of the case, not incidental.
    #[test]
    fn a_later_float_shard_widens_the_whole_sidecar_and_forces_pcodec() {
        let got = pick_csc_encoding(
            &[ValueEncoding::Uint8, ValueEncoding::Float32],
            200,
            Some(CodecId::Scx1),
        );
        assert_eq!(got, Some((ValueEncoding::Float32, CodecId::Pcodec)));
    }

    /// `ShardStats` is format-permitted to be absent, and `max_int_val` is
    /// folded only over the shards that have it. Flooring on the *declared*
    /// width is what stops a stats-less `Uint32` shard being encoded `Uint8`
    /// and truncated.
    #[test]
    fn a_wide_integer_shard_without_stats_is_not_under_picked() {
        let got = pick_csc_encoding(&[ValueEncoding::Uint32], 0, Some(CodecId::Scx1));
        assert_eq!(got, Some((ValueEncoding::Uint32, CodecId::Scx1)));
    }

    /// The other direction: a shard that declares `Uint8` but whose catalog
    /// reports a value no `Uint8` holds widens from the value side.
    #[test]
    fn a_value_wider_than_the_declared_encoding_widens_it() {
        let got = pick_csc_encoding(&[ValueEncoding::Uint8], 70_000, Some(CodecId::Zstd));
        assert_eq!(got, Some((ValueEncoding::Uint32, CodecId::Zstd)));
    }

    /// Write semantics, not `ValueEncoding::widest`'s reporting semantics: a
    /// uniform `Float16` family *reports* as `Float16` but must be *written*
    /// as `Float32`, which is the divergence the two functions exist to keep
    /// apart.
    #[test]
    fn an_all_float16_family_is_written_as_float32() {
        assert_eq!(
            ValueEncoding::widest(&[ValueEncoding::Float16, ValueEncoding::Float16]),
            Some(ValueEncoding::Float16),
        );
        let got = pick_csc_encoding(
            &[ValueEncoding::Float16, ValueEncoding::Float16],
            0,
            Some(CodecId::Scx1),
        );
        assert_eq!(got, Some((ValueEncoding::Float32, CodecId::Pcodec)));
    }

    /// `None` is reachable on exactly one path — integer, with no shard to
    /// take a codec from. The float path never needs one, so it answers even
    /// with an empty shard list.
    #[test]
    fn no_shards_yields_none_only_on_the_integer_path() {
        assert_eq!(pick_csc_encoding(&[], 0, None), None);
        assert_eq!(
            pick_csc_encoding(&[ValueEncoding::Float32], 0, None),
            Some((ValueEncoding::Float32, CodecId::Pcodec)),
        );
    }

    use super::*;
    use crate::header::FileHeader;
    use crate::reader::ScxReader;
    use crate::writer::ScxWriter;

    /// Round-trip a small 3×3 canonical CSR through `write_csc_sidecar` and
    /// read the CSC sidecar back, checking the column-major transpose.
    /// Every column must be covered by exactly one emitted shard, including
    /// columns no row touches.
    ///
    /// Two failure modes this guards, both silent. `BackedCscIndex`'s
    /// `shard_for_col` and `shards_for_col_range` binary-search a tiling they
    /// assume is sorted, contiguous and non-overlapping, so a skipped
    /// zero-nnz shard makes those columns answer `None` — a read that quietly
    /// returns nothing rather than erroring. And `from_catalog` **drops** any
    /// entry whose `stats` is absent, so a shard that reached disk without
    /// stats would be invisible to the reader while `n_csc_shards` still
    /// counted it.
    #[test]
    fn all_zero_column_runs_still_get_their_own_shards() {
        // 4 x 9. Columns 0-2 and 5-8 are entirely empty: a leading run, an
        // interior run and a trailing one, so a shard can be skipped at either
        // end or in the middle.
        let csr = ScxCsr::new(
            (4, 9),
            vec![0, 1, 2, 3, 4],
            vec![3, 4, 3, 4],
            vec![1.0, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sparse_cols.scx");
        let header = FileHeader::new_single_modality(4, 9, 4, 16384, CodecId::None as u8, 0);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        let raw = values_to_raw_bytes(&csr.data, ValueEncoding::Uint8).unwrap();
        writer
            .write_csr_shard(
                &csr.indptr.iter().map(|&v| v as u64).collect::<Vec<_>>(),
                &csr.indices.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                &raw,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        write_csc_sidecar(
            &mut writer,
            std::slice::from_ref(&csr),
            4,
            9,
            ValueEncoding::Uint8,
            CodecId::None,
            CscSidecarOptions {
                cols_per_shard: 2,
                ..Default::default()
            },
        )
        .unwrap();
        let final_path = writer.finish().unwrap();

        let reader = ScxReader::open(&final_path).unwrap();
        // ceil(9 / 2) = 5 shards, three of which hold no nonzeros at all.
        assert_eq!(reader.header().n_csc_shards, 5);
        let index = crate::backed::BackedCscIndex::from_catalog(reader.catalog());
        assert_eq!(
            index.n_shards(),
            reader.header().n_csc_shards as usize,
            "a CSC entry without stats is dropped by from_catalog and would be \
             invisible to every read"
        );
        for c in 0..9u64 {
            assert!(
                index.shard_for_col(c).is_some(),
                "column {c} is in no shard's range"
            );
        }
        // Contiguous, ascending, non-overlapping — what the binary searches
        // above assume.
        let mut expect_lo = 0u64;
        for s in 0..index.n_shards() {
            let (lo, hi) = index.shard_col_range(s).expect("range");
            assert_eq!(
                lo, expect_lo,
                "shard {s} starts at {lo}, expected {expect_lo}"
            );
            assert!(hi > lo, "shard {s} is empty-width");
            expect_lo = hi;
        }
        assert_eq!(expect_lo, 9);

        let csc = reader.read_all_csc_shards_for(0).unwrap();
        assert_eq!(csc.to_dense().unwrap(), csr.to_dense().unwrap());
    }

    #[test]
    fn write_csc_sidecar_round_trips_single_modality() {
        // CSR (3 rows × 3 cols):
        //   row0: (c0=1, c2=3)
        //   row1: (c1=2)
        //   row2: (c2=4)
        let csr = ScxCsr::new(
            (3, 3),
            vec![0, 2, 3, 4],
            vec![0, 2, 1, 2],
            vec![1.0, 3.0, 2.0, 4.0],
        )
        .unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("csc.scx");
        let header = FileHeader::new_single_modality(3, 3, 4, 16384, CodecId::None as u8, 0);
        let mut writer = ScxWriter::new(&path, header).unwrap();
        let raw = values_to_raw_bytes(&csr.data, ValueEncoding::Uint8).unwrap();
        writer
            .write_csr_shard(
                &csr.indptr.iter().map(|&v| v as u64).collect::<Vec<_>>(),
                &csr.indices.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                &raw,
                CodecId::None,
                ValueEncoding::Uint8,
                0,
            )
            .unwrap();
        write_csc_sidecar(
            &mut writer,
            std::slice::from_ref(&csr),
            3,
            3,
            ValueEncoding::Uint8,
            CodecId::None,
            CscSidecarOptions {
                cols_per_shard: 8,
                ..Default::default()
            },
        )
        .unwrap();
        let final_path = writer.finish().unwrap();

        let reader = ScxReader::open(&final_path).unwrap();
        assert!(reader.header().n_csc_shards >= 1);
        let csc = reader.read_all_csc_shards_for(0).unwrap();
        // Reconstruct dense from the CSC and compare to the CSR's dense form
        // (both row-major n_obs × n_vars).
        let dense_from_csc = csc.to_dense().unwrap();
        let dense_from_csr = csr.to_dense().unwrap();
        assert_eq!(dense_from_csc, dense_from_csr);
    }

    /// `write_csc_sidecar(..., Some(framing))` emits **framed (v2)** CSC shards
    /// (framed CSC sidecar producer). The framed sidecar round-trips to the same
    /// dense matrix and every emitted CSC shard reports `shard_format_version == 2`,
    /// while a `None` control emits v1. Also confirms scattered per-column-group
    /// reads over the produced framed sidecar match a full column-slice.
    #[test]
    fn write_csc_sidecar_frames_when_requested() {
        let csr = ScxCsr::new(
            (3, 4),
            vec![0, 2, 3, 5],
            vec![0, 3, 1, 2, 3],
            vec![1.0, 4.0, 2.0, 3.0, 5.0],
        )
        .unwrap();

        let produce = |name: &str, framing: Option<FramingConfig>| {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join(name);
            let mut header =
                FileHeader::new_single_modality(3, 4, 5, 16384, CodecId::None as u8, 0);
            if framing.is_some() {
                header.format_version = crate::header::CURRENT_FORMAT_VERSION;
            }
            let mut writer = ScxWriter::new(&path, header).unwrap();
            // Some CSR X shard must exist for a valid file.
            let raw = values_to_raw_bytes(&csr.data, ValueEncoding::Uint8).unwrap();
            writer
                .write_csr_shard(
                    &csr.indptr.iter().map(|&v| v as u64).collect::<Vec<_>>(),
                    &csr.indices.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    &raw,
                    CodecId::None,
                    ValueEncoding::Uint8,
                    0,
                )
                .unwrap();
            write_csc_sidecar(
                &mut writer,
                std::slice::from_ref(&csr),
                3,
                4,
                ValueEncoding::Uint8,
                CodecId::ShufDeltaZstd,
                CscSidecarOptions {
                    cols_per_shard: 8,
                    framing,
                    ..Default::default()
                },
            )
            .unwrap();
            // Framing is call-scoped: the writer's framing is restored afterward.
            assert!(writer.framing().is_none(), "framing must be restored");
            let final_path = writer.finish().unwrap();
            // Move the file out of the tempdir's lifetime by reading eagerly.
            let reader = ScxReader::open(&final_path).unwrap();
            let versions: Vec<u8> = reader
                .catalog()
                .csc_shards_sorted()
                .iter()
                .map(|e| reader.read_shard_header(e).unwrap().shard_format_version)
                .collect();
            let dense = reader
                .read_all_csc_shards_for(0)
                .unwrap()
                .to_dense()
                .unwrap();
            (versions, dense)
        };

        let (v_unframed, d_unframed) = produce("csc_unframed.scx", None);
        let (v_framed, d_framed) = produce(
            "csc_framed.scx",
            Some(FramingConfig {
                row_group_rows: 1, // one column-group per gene
                target_nnz: None,
                trial: false,
                decode_target: None,
            }),
        );

        assert!(
            !v_unframed.is_empty() && v_unframed.iter().all(|&v| v == 1),
            "control must be v1"
        );
        assert!(
            !v_framed.is_empty() && v_framed.iter().all(|&v| v == 2),
            "framed CSC must be v2"
        );
        assert_eq!(d_framed, d_unframed, "framed CSC densifies identically");
        assert_eq!(d_framed, csr.to_dense().unwrap());
    }
}
