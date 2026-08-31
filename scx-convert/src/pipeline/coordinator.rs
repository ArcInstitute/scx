//! The shard coordinators: sequential, parallel, and the range planners.
//!
//! [`run_streaming_writer_coordinator`] is the dispatcher -- it derates against
//! the memory budget and routes to the sequential or the parallel body. The
//! pool, bounded channel, rolling-window spawn, reorder buffer and
//! panic-to-`Err` conversion are NOT here: they live in
//! `crate::parallel_drain`, shared with the export coordinator (ORG-11.16-2).
//! What stays is what is genuinely ingest-specific -- encoding a shard, and the
//! envelope its failures wear.

use scx_codec::CodecId;
use scx_format_io::modality::ModalityType;
use scx_format_io::section::SectionType;
use scx_format_io::writer::{PreEncodedSection, ScxWriter};
use scx_format_io::{encode_one_shard, FramingConfig};
use scx_sparse::canonicalize_csr;

use super::bitmap::{
    build_and_write_bitmap_for_shard, maybe_build_bitmap_shard, BitmapBuildOutcome,
};
use super::error::ConvertError;
use super::threads::{derate_threads_and_depth, ensure_shard_fits_budget, resolve_reader_threads};
use crate::options::IngestOptions;
use crate::stream::CsrShardStream;
use crate::warnings::{ConvertWarning, WarningSink};
use scx_format_io::BitmapPolicy;

/// Drain a [`CsrShardStream`] into an [`ScxWriter`] one shard at a
/// time, encoding each shard through [`encode_one_shard`].
///
/// Phase 0.2 seam. The initial implementation is sequential — one
/// shard read → canonicalize → encode → write per iteration —
/// matching the behaviour of the original inline loop in
/// `h5ad_to_scx_streaming`. Phase 8c will replace the body with a
/// bounded shard queue plus an ordered writer stage, but the
/// signature is the same: every entry point that drives a
/// `CsrShardStream` (X, layers, h5mu modalities, Zarr) goes through
/// this helper.
///
/// `section_name_prefix` is appended with `_{shard_idx}` to produce
/// the per-shard section name. Returns the number of shards
/// written, useful for callers that need to track per-source shard
/// counts (e.g. the layer loop).
#[allow(clippy::too_many_arguments)]
pub fn streaming_writer_coordinator(
    reader: &mut dyn CsrShardStream,
    writer: &mut ScxWriter,
    opts: &IngestOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let target_rows = opts.shard_target_rows as usize;
    let mut shard_idx: u32 = 0;
    let mut row_ranges: Vec<(u64, u64)> = Vec::new();
    while let Some(mut shard) = reader.next_csr_shard(target_rows)? {
        // Surface upstream duplicate-coordinate canonicalisation
        // (Phase 2 CSC external transpose) as a typed warning. Other
        // readers always set `duplicates_merged = 0` so this is a
        // no-op for them.
        if shard.duplicates_merged > 0 {
            sink.emit(ConvertWarning::DuplicateCoordinatesMerged {
                count: shard.duplicates_merged,
                policy: "sum".to_string(),
            });
        }
        canonicalize_csr(&mut shard.indptr, &mut shard.indices, &mut shard.values);
        let row_start = shard.row_start;
        let n_rows = shard.n_rows as u64;
        let mut enc_opts = scx_format_io::EncodeShardOptions::new(
            format!("{section_name_prefix}_{shard_idx}"),
            section_type,
            n_vars_u32 as u64,
            row_start,
        );
        enc_opts.explicit_codec = opts.codec;
        enc_opts.index_dtype = index_dtype;
        enc_opts.modality_type = modality_type;
        enc_opts.framing = opts.framing();
        let pre = encode_one_shard(&shard.indptr, &shard.indices, &shard.values, &enc_opts)?;
        let encoded_csr_size = pre.section_length as usize;
        writer.write_preencoded_shard(pre)?;
        // Detection bitmap (only for primary X shards; layer
        // shards are skipped — bitmaps are per X-axis presence today).
        if section_type == SectionType::CsrShard {
            build_and_write_bitmap_for_shard(
                writer,
                &shard.indptr,
                &shard.indices,
                row_start,
                shard.n_rows,
                n_vars_u32,
                encoded_csr_size,
                opts.bitmap,
                modality_type,
                None,
                sink,
            )?;
        }
        row_ranges.push((row_start, row_start + n_rows));
        shard_idx += 1;
    }
    Ok((shard_idx, row_ranges))
}

/// Split the matrix into `[(row_start, n_rows)]` shard
/// ranges. The sequential coordinator's `while next_csr_shard(...)`
/// loop produces the same partition implicitly; the parallel
/// coordinator hoists it so workers can claim ranges without
/// coordination.
pub(crate) fn compute_shard_row_ranges(n_obs: u64, target_rows: u32) -> Vec<(u64, u32)> {
    if target_rows == 0 || n_obs == 0 {
        return Vec::new();
    }
    let target = target_rows as u64;
    let cap = (n_obs / target + 1) as usize;
    let mut out = Vec::with_capacity(cap);
    let mut row = 0u64;
    while row < n_obs {
        let n = (n_obs - row).min(target) as u32;
        out.push((row, n));
        row += n as u64;
    }
    out
}

/// Phase 7.4: expand a group planner's `shard_starts` (emit-row indices at
/// which a new shard begins, excluding 0 and EOF) into the explicit
/// `[(row_start, n_rows)]` shard ranges covering `[0, n_obs)` that the
/// coordinator drives. Groups are never split, so ranges are contiguous and
/// gap-free.
pub(super) fn group_shard_starts_to_ranges(shard_starts: &[u64], n_obs: u64) -> Vec<(u64, u32)> {
    let mut ranges: Vec<(u64, u32)> = Vec::with_capacity(shard_starts.len() + 1);
    let mut start = 0u64;
    for &brk in shard_starts {
        if brk > start {
            ranges.push((start, (brk - start) as u32));
            start = brk;
        }
    }
    if start < n_obs {
        ranges.push((start, (n_obs - start) as u32));
    }
    ranges
}

/// Payload from a worker thread to the ordered writer.
struct EncodedShardOutput {
    pre: PreEncodedSection,
    /// For sequential `BitmapShard` write after the CSR shard.
    /// Built by the worker; the writer just calls `write_bitmap_shard`.
    bitmap: Option<BitmapBuildOutcome>,
    duplicates_merged: u64,
}

/// Parallel sibling of [`streaming_writer_coordinator`].
///
/// Partitions the reader into `[(row_start, n_rows)]` ranges, fans
/// them out across a rayon worker pool, and writes encoded shards in
/// shard-index order via a bounded reorder buffer. Output is
/// byte-identical to the sequential coordinator.
///
/// Requires `reader.as_indexed()` to return `Some(_)`; callers must
/// verify that and the libhdf5 thread-safety check before routing
/// here. The top-level [`run_streaming_writer_coordinator`] handles
/// the fallback to sequential when those preconditions fail.
#[allow(clippy::too_many_arguments)]
fn streaming_writer_coordinator_parallel(
    reader: &dyn crate::stream::IndexedCsrShardStream,
    writer: &mut ScxWriter,
    opts: &IngestOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
    reader_threads: usize,
    queue_depth: usize,
    ranges: Vec<(u64, u32)>,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    if ranges.is_empty() {
        return Ok((0, Vec::new()));
    }

    let row_ranges: Vec<(u64, u64)> = ranges
        .iter()
        .map(|&(rs, nr)| (rs, rs + nr as u64))
        .collect();

    let n_ranges = ranges.len();
    let source_name: String = reader.source_matrix_name().to_string();
    let opts_codec = opts.codec;
    let opts_bitmap = opts.bitmap;
    let opts_framing = opts.framing();
    let want_bitmap = section_type == SectionType::CsrShard;
    let name_prefix = section_name_prefix.to_string();

    // Capture the fault-injection setting on the calling thread (the
    // caller's `FailIngestShardGuard` lives in this thread's
    // thread-local). The captured `Option<usize>` is `Copy` and
    // propagates into rayon workers via the spawn closure capture, so
    // each coordinator invocation carries its own fault config —
    // concurrent non-fault tests on other threads see `None`.
    #[cfg(test)]
    let captured_fault_shard = super::test_hooks::current_ingest_fault_shard();

    // Panic-injection sibling of the fault switch above: forces a real
    // `panic!` inside the worker body, exercising the `catch_unwind` that
    // `ordered_parallel_drain` wraps it in — which is what turns a panic into
    // a delivered `Err` rather than a lost send and a hung drain.
    #[cfg(test)]
    let captured_panic_shard = super::test_hooks::current_ingest_panic_shard();

    // Slow-worker injection, captured on the calling thread like the two
    // above. Makes one shard take long enough that every other shard
    // completes behind it — the skew the reorder-buffer bound exists for.
    #[cfg(test)]
    let captured_delay_shard = super::test_hooks::current_ingest_delay_shard();

    // The pool, the bounded channel, the rolling-window spawn, the reorder
    // buffer and the panic-to-`Err` conversion all live in
    // `crate::parallel_drain`, shared with the export coordinator in
    // `h5ad/stream_write.rs`. What stays here is what is genuinely
    // ingest-specific: encoding a shard, and the envelope its failures wear.
    crate::parallel_drain::ordered_parallel_drain(
        n_ranges,
        reader_threads,
        queue_depth,
        "scx-stream",
        |idx| -> Result<EncodedShardOutput, ConvertError> {
            let (row_start, n_rows) = ranges[idx];
            #[cfg(test)]
            if Some(idx) == captured_fault_shard {
                return Err(ConvertError::ShardRead {
                    row_start,
                    n_rows,
                    source: source_name.clone(),
                    inner: Box::new(ConvertError::Other(format!(
                        "test_hooks: injected failure at shard {idx}"
                    ))),
                });
            }
            #[cfg(test)]
            if Some(idx) == captured_panic_shard {
                panic!("test_hooks: injected panic at shard {idx}");
            }
            #[cfg(test)]
            if let Some((delay_idx, millis)) = captured_delay_shard {
                if delay_idx == idx {
                    std::thread::sleep(std::time::Duration::from_millis(millis));
                }
            }
            encode_one_shard_worker(
                reader,
                row_start,
                n_rows,
                opts_codec,
                index_dtype,
                n_vars_u32,
                section_type,
                modality_type,
                format!("{name_prefix}_{idx}"),
                opts_bitmap,
                want_bitmap,
                opts_framing,
            )
            .map_err(|inner| ConvertError::ShardRead {
                row_start,
                n_rows,
                source: source_name.clone(),
                inner: Box::new(inner),
            })
        },
        |idx, message| {
            let (row_start, n_rows) = ranges[idx];
            ConvertError::ShardRead {
                row_start,
                n_rows,
                source: source_name.clone(),
                inner: Box::new(ConvertError::Other(format!(
                    "worker panicked while encoding shard {idx}: {message}"
                ))),
            }
        },
        |_idx, out| {
            if out.duplicates_merged > 0 {
                sink.emit(ConvertWarning::DuplicateCoordinatesMerged {
                    count: out.duplicates_merged,
                    policy: "sum".to_string(),
                });
            }
            let bitmap = out.bitmap;
            writer.write_preencoded_shard(out.pre)?;
            if want_bitmap {
                match bitmap {
                    None => { /* policy was Off — nothing to do */ }
                    Some(BitmapBuildOutcome::Skip { reason }) => {
                        sink.emit(ConvertWarning::BitmapSkipped {
                            modality: None,
                            reason,
                        });
                    }
                    Some(BitmapBuildOutcome::Built(shard)) => {
                        writer
                            .write_bitmap_shard(&shard)
                            .map_err(ConvertError::from)?;
                    }
                }
            }
            Ok(())
        },
    )?;

    Ok((n_ranges as u32, row_ranges))
}

/// Worker body: read + canonicalise + encode one shard and (if
/// requested) build the bitmap for it. Pure with respect to the
/// writer — output is funnelled back through a channel to the
/// ordered writer thread.
#[allow(clippy::too_many_arguments)]
fn encode_one_shard_worker(
    reader: &dyn crate::stream::IndexedCsrShardStream,
    row_start: u64,
    n_rows: u32,
    codec: Option<CodecId>,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    name: String,
    bitmap_policy: BitmapPolicy,
    want_bitmap: bool,
    framing: Option<FramingConfig>,
) -> Result<EncodedShardOutput, ConvertError> {
    let mut shard = reader.read_range(row_start, n_rows)?;
    let duplicates_merged = shard.duplicates_merged;
    canonicalize_csr(&mut shard.indptr, &mut shard.indices, &mut shard.values);

    let mut enc_opts = scx_format_io::EncodeShardOptions::new(
        name,
        section_type,
        n_vars_u32 as u64,
        shard.row_start,
    );
    enc_opts.explicit_codec = codec;
    enc_opts.index_dtype = index_dtype;
    enc_opts.modality_type = modality_type;
    enc_opts.framing = framing;
    let pre = encode_one_shard(&shard.indptr, &shard.indices, &shard.values, &enc_opts)?;
    let encoded_csr_size = pre.section_length as usize;

    let bitmap = if want_bitmap {
        maybe_build_bitmap_shard(
            &shard.indptr,
            &shard.indices,
            shard.row_start,
            shard.n_rows,
            n_vars_u32,
            encoded_csr_size,
            bitmap_policy,
            modality_type,
        )
    } else {
        None
    };

    Ok(EncodedShardOutput {
        pre,
        bitmap,
        duplicates_merged,
    })
}

/// Top-level dispatcher: pick parallel vs sequential
/// coordinator based on `opts.reader_threads`, libhdf5 thread-safety,
/// reader trait support, and memory-budget derate.
///
/// Always returns `(shard_count, row_ranges)` matching the sequential
/// coordinator's contract. Output is byte-identical regardless of
/// path.
///
/// `explicit_ranges` (Phase 7.4 convert-time grouping): when `Some`, the
/// caller supplies group-aligned `[(row_start, n_rows)]` shard ranges instead
/// of the fixed-`shard_target_rows` partition. Groups are never split, so a
/// single range may exceed `shard_target_rows`; the per-worker budget is sized
/// by the largest range. The grouped X reader is always indexed, so the ranges
/// are honored in both the parallel pool and the sequential fallback. All
/// non-grouped callers pass `None` and are byte-identical to before.
#[allow(clippy::too_many_arguments)]
pub fn run_streaming_writer_coordinator(
    reader: &mut dyn CsrShardStream,
    writer: &mut ScxWriter,
    opts: &IngestOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
    explicit_ranges: Option<&[(u64, u32)]>,
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let requested = resolve_reader_threads(opts.reader_threads);

    // ----- Phase 7.4: group-aligned ranges -----
    if let Some(ranges) = explicit_ranges {
        if ranges.is_empty() {
            return Ok((0, Vec::new()));
        }
        // The grouped X reader is always an indexed (row-range) reader
        // (PermutedCsrReader). Without it we cannot honor variable breaks.
        let Some(indexed) = reader.as_indexed() else {
            return Err(ConvertError::Other(
                "grouped convert requires an indexed (row-range) X reader".to_string(),
            ));
        };
        // The largest group bounds the per-worker working set (groups are never
        // split), so size the derate by it rather than by `--shard-size`.
        let max_range_rows = ranges.iter().map(|&(_, n)| n).max().unwrap_or(0);
        let per_worker_bytes = indexed
            .per_worker_bytes(max_range_rows, modality_type)
            .max(1);
        // M2: the sequential fallback below buffers the largest group whole, so
        // it must honor `memory_budget` too — the parallel derate already
        // refuses an oversized shard, but the sequential branch previously did
        // not. Check before either dispatch so both routes reject identically.
        ensure_shard_fits_budget(
            opts.memory_budget,
            per_worker_bytes,
            "grouped shard",
            "raise --group-target-bytes/--memory-budget or accept a larger group shard",
        )?;
        let threadsafe = crate::hdf5_threadsafe::hdf5_is_threadsafe();
        if requested <= 1 || !threadsafe {
            if requested > 1 && !threadsafe {
                crate::hdf5_threadsafe::try_emit_not_threadsafe_warning(sink);
            }
            return streaming_writer_coordinator_ranges(
                indexed,
                writer,
                opts,
                index_dtype,
                n_vars_u32,
                section_type,
                modality_type,
                section_name_prefix,
                sink,
                ranges,
            );
        }
        let requested_depth = opts.writer_queue_depth.max(1);
        let (granted, granted_depth) = derate_threads_and_depth(
            opts.memory_budget,
            per_worker_bytes,
            requested,
            requested_depth,
            "grouped parallel-streaming shard",
            "raise --group-target-bytes/--memory-budget or accept a larger group shard",
            sink,
        )?;
        if granted <= 1 {
            return streaming_writer_coordinator_ranges(
                indexed,
                writer,
                opts,
                index_dtype,
                n_vars_u32,
                section_type,
                modality_type,
                section_name_prefix,
                sink,
                ranges,
            );
        }
        return streaming_writer_coordinator_parallel(
            indexed,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
            granted,
            granted_depth,
            ranges.to_vec(),
        );
    }

    if requested <= 1 {
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    }

    // Reader must expose row-range reads.
    let Some(indexed) = reader.as_indexed() else {
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    };

    // libhdf5 must be threadsafe (only when we hit the HDF5-backed
    // readers — for in-memory MaterializedCsrStream the check is
    // harmless but we still gate to keep behaviour uniform).
    if !crate::hdf5_threadsafe::hdf5_is_threadsafe() {
        crate::hdf5_threadsafe::try_emit_not_threadsafe_warning(sink);
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    }

    // Clamp the partition's shard size by the reader's hard slab cap.
    // Only `DenseXStreamReader` returns `Some(_)` today (when
    // `memory_budget` shrinks `max_slab_rows` below
    // `shard_target_rows`). Sequential `DenseXStreamReader::
    // next_csr_shard` already clamps the same way, so byte-identity
    // with the sequential path holds.
    let effective_target = indexed
        .max_slab_rows()
        .map_or(opts.shard_target_rows, |cap| {
            opts.shard_target_rows.min(cap)
        });

    // Memory-budget derate: cap workers so per-worker working set
    // fits under `memory_budget`. Estimate is delegated to the reader
    // via `IndexedCsrShardStream::per_worker_bytes`: the default impl
    // assumes the sparsified output is the binding bound and picks
    // density by modality (RNA/default 5 %, ATAC 10 %); the dense
    // reader override sizes the dense slab buffer instead. Pass
    // `effective_target` so the estimate matches the shard size we
    // are actually about to drive workers at.
    let per_worker_bytes = indexed
        .per_worker_bytes(effective_target, modality_type)
        .max(1);
    // Derate threads AND depth together: peak outstanding shards is
    // `granted + granted_depth` (see `streaming_writer_coordinator_parallel`'s
    // `in_flight_cap`), so the budget must cover both — sizing threads alone
    // overshot by up to `writer_queue_depth × per_worker_bytes`.
    let requested_depth = opts.writer_queue_depth.max(1);
    let (granted, granted_depth) = derate_threads_and_depth(
        opts.memory_budget,
        per_worker_bytes,
        requested,
        requested_depth,
        "parallel-streaming shard",
        "lower --shard-size or raise --memory-budget",
        sink,
    )?;

    if granted <= 1 {
        return streaming_writer_coordinator(
            reader,
            writer,
            opts,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            section_name_prefix,
            sink,
        );
    }

    let ranges = compute_shard_row_ranges(indexed.n_obs(), effective_target);
    streaming_writer_coordinator_parallel(
        indexed,
        writer,
        opts,
        index_dtype,
        n_vars_u32,
        section_type,
        modality_type,
        section_name_prefix,
        sink,
        granted,
        granted_depth,
        ranges,
    )
}

/// Sequential sibling of [`streaming_writer_coordinator`] that honors an
/// explicit list of (possibly variable-size) shard ranges — the group-aligned
/// breaks from the Phase 7.4 grouped-convert planner. Reuses
/// [`encode_one_shard_worker`] per range so the encoded bytes are identical to
/// the parallel path for the same ranges (which is in turn byte-identical to
/// the fixed-size sequential coordinator). Requires an indexed reader.
#[allow(clippy::too_many_arguments)]
fn streaming_writer_coordinator_ranges(
    reader: &dyn crate::stream::IndexedCsrShardStream,
    writer: &mut ScxWriter,
    opts: &IngestOptions,
    index_dtype: u8,
    n_vars_u32: u32,
    section_type: SectionType,
    modality_type: ModalityType,
    section_name_prefix: &str,
    sink: &mut WarningSink,
    ranges: &[(u64, u32)],
) -> Result<(u32, Vec<(u64, u64)>), ConvertError> {
    let want_bitmap = section_type == SectionType::CsrShard;
    let mut row_ranges: Vec<(u64, u64)> = Vec::with_capacity(ranges.len());
    for (shard_idx, &(row_start, n_rows)) in ranges.iter().enumerate() {
        let out = encode_one_shard_worker(
            reader,
            row_start,
            n_rows,
            opts.codec,
            index_dtype,
            n_vars_u32,
            section_type,
            modality_type,
            format!("{section_name_prefix}_{shard_idx}"),
            opts.bitmap,
            want_bitmap,
            opts.framing(),
        )?;
        if out.duplicates_merged > 0 {
            sink.emit(ConvertWarning::DuplicateCoordinatesMerged {
                count: out.duplicates_merged,
                policy: "sum".to_string(),
            });
        }
        writer.write_preencoded_shard(out.pre)?;
        if want_bitmap {
            match out.bitmap {
                None => { /* policy was Off — nothing to do */ }
                Some(BitmapBuildOutcome::Skip { reason }) => {
                    sink.emit(ConvertWarning::BitmapSkipped {
                        modality: None,
                        reason,
                    });
                }
                Some(BitmapBuildOutcome::Built(shard)) => {
                    writer
                        .write_bitmap_shard(&shard)
                        .map_err(ConvertError::from)?;
                }
            }
        }
        row_ranges.push((row_start, row_start + n_rows as u64));
    }
    Ok((ranges.len() as u32, row_ranges))
}
