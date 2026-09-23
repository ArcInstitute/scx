//! Streaming ingest: h5ad and 10x.
//!
//! The bounded-memory path: X is read shard by shard and never fully resident.
//! [`h5ad_to_scx_streaming`] is the long one because it is the sequencer -- it
//! decides source layout, obs sharding, CSC sidecar, bitmaps and predicate
//! indexes, and hands each to the module that owns it.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::GroupPass;
use scx_format_io::header::FileHeader;
use scx_format_io::modality::ModalityType;
use scx_format_io::provenance::ProvenanceEntry;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;

use super::coordinator::{group_shard_starts_to_ranges, run_streaming_writer_coordinator};
use super::error::ConvertError;
use super::index::build_and_write_predicate_indexes;
use super::mappings::{
    write_dense_mapping_section, write_sparse_mapping_section, DenseMappingKind, SparseMappingKind,
};
use super::shards::ingest_raw_streaming;
use super::threads::resolve_reader_threads;
use super::write_ingest_obs;
use crate::detect::{detect_input_format, detect_matrix_format, InputFormat, MatrixFormat};
use crate::dtype::index_dtype_for;
use crate::h5ad::csc_stream::{open_csc_layer_streaming, open_csc_streaming};
use crate::h5ad::dense_stream::{open_dense_layer_streaming, open_dense_streaming};
use crate::h5ad::read::{read_dataframe_group, read_uns};
use crate::h5ad::stream::{open_layer_streaming, open_x_streaming};
use crate::options::IngestOptions;
use crate::stream::CsrShardStream;
use crate::warnings::{ConvertWarning, WarningSink};
use scx_engine::ConversionPredicateIndexOptions;
use scx_format_io::BitmapPolicy;
use scx_format_io::CscPolicy;

/// Override hooks for [`h5ad_to_scx_streaming`]. Each `Some(...)`
/// field skips the corresponding on-disk read and uses the provided
/// value instead.
///
/// The pyscx backed-AnnData routing path in `from_anndata` uses this
/// to preserve in-Python mutations to `obs` / `var` / `uns` / `obsm` /
/// `varm` / `obsp` / `varp` that would otherwise be silently lost
/// when the streaming pipeline re-reads them from disk.
///
/// Layers are intentionally not overridable — they're streamed
/// directly from disk per shard, and the pyscx backed-mode path
/// emits a `UserWarning` if the in-memory AnnData has layers (where
/// any in-memory mutations would be dropped).
#[derive(Default)]
pub struct StreamingOverrides {
    pub obs: Option<arrow::record_batch::RecordBatch>,
    pub var: Option<arrow::record_batch::RecordBatch>,
    pub uns: Option<serde_json::Value>,
    pub obsm: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
    pub varm: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
    pub obsp: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
    pub varp: Option<Vec<(String, arrow::record_batch::RecordBatch)>>,
}

/// Streaming h5ad → SCX conversion. Reads the input one shard's worth
/// of rows at a time via [`crate::h5ad::stream::XStreamReader`] so peak
/// memory is bounded by `shard_target_rows × n_vars × density × ~16
/// bytes` plus the always-resident indptr (`(n_obs + 1) × 8 bytes`).
///
/// Shard processing dispatches through
/// [`run_streaming_writer_coordinator`], which fans the per-shard read →
/// sort → drop-zeros → encode work out across a rayon worker pool and
/// reassembles output in shard order via a bounded crossbeam reorder
/// buffer (output is byte-identical to the sequential path). It falls
/// back to the sequential coordinator when libhdf5 is not built
/// threadsafe (gated by an `H5is_library_threadsafe` probe), when the
/// reader can't do row-range reads, or when `reader_threads <= 1`.
///
/// When `opts.csc` asks for a sidecar it is built in the same pass as X
/// (`ScxWriter::enable_csc_sidecar`): each X shard is pushed into the builder
/// as it is written, and the CSC shards follow X in the file. No second read of
/// the output, and no second copy of it on disk.
///
/// `obsm` / `varm` / `obsp` / `varp` are hyperslab-read one row-range
/// at a time and emitted as row-sharded sections
/// (`<section>/<name>_shard_<idx>`). Peak memory per matrix is bounded
/// by `shard_target_rows × k × 4 B` (dense) or
/// `shard_target_rows × density × n_cols × 16 B` (sparse). Caller
/// overrides supplied via [`StreamingOverrides`] are partitioned
/// into the same row-shards on the way out. `uns` is read in full
/// from disk when not overridden (typically KB–MB; no shard format
/// makes sense for a JSON tree). CSR, dense and CSC-on-disk `X` all stream
/// (see `open_x_streaming`); a CSC-on-disk `X` cannot be reordered on the way
/// in, so `--sort-by` / `--group-by` refuse it.
/// True when the h5ad has a non-empty `obsp` group (ignoring `__`-prefixed
/// internal members). Used by the grouped-convert router (M3) to route
/// obsp-carrying inputs through the obsp-preserving two-pass path.
fn h5ad_has_obsp_members(file: &hdf5::File) -> bool {
    match file.group("obsp") {
        Ok(group) => group
            .member_names()
            .unwrap_or_default()
            .iter()
            .any(|n| !n.starts_with("__")),
        Err(_) => false,
    }
}

pub fn h5ad_to_scx_streaming(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    overrides: &StreamingOverrides,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let file = hdf5::File::open(input)?;

    // Format gating. `open_x_streaming` re-checks CSC/dense and the
    // encoding-type attribute; this branch only catches the 10x case
    // (which has no `X` group at all).
    let input_format = detect_input_format(&file)?;
    if matches!(input_format, InputFormat::TenX) {
        return Err(ConvertError::FormatMismatch {
            expected: "h5ad".to_string(),
            got: "10x".to_string(),
        });
    }
    let matrix_format = detect_matrix_format(&file, sink)?;

    // Open the X reader first — it surfaces shape via `n_obs` /
    // `n_vars`, both of which the file header needs before any section
    // write. CSR uses the indptr-eager reader; Dense slabs rows on
    // demand; CSC routes through the Phase 2 dispatcher which picks
    // in-memory vs. external-memory transpose based on the budget.
    // Reorder-on-convert: `--sort-by` (Phase 2) and `--group-by` (Phase 7.4)
    // both permute the obs axis during ingest. A CSC-on-disk X cannot be
    // reordered (the CSC→CSR transposer is sequential and cannot serve the
    // permuted gather). Fail fast rather than silently emit a misordered file.
    if opts.reference.is_some() && opts.group_by.is_none() {
        return Err(ConvertError::Other(
            "convert --reference requires --group-by".to_string(),
        ));
    }
    let want_sort = !opts.sort_by.is_empty();
    let want_group = opts.group_by.is_some();
    let want_reorder = want_sort || want_group;
    if want_reorder && matches!(matrix_format, MatrixFormat::Csc) {
        return Err(ConvertError::Other(
            "reorder-on-convert (--sort-by / --group-by) requires a CSR or dense h5ad X; the \
             on-disk CSC X cannot be reordered during conversion. Re-export X as CSR/dense, or \
             reorder after conversion with `scx sort`."
                .to_string(),
        ));
    }

    // Phase 7.4 group-pass routing. The one-pass streaming grouped gather is a
    // win for CSR (reads only nnz/row) but a ~4–5× loss for dense (full-width
    // random row reads), so `Auto` routes dense → two-pass (plain convert then
    // `scx sort --group-by`), which is faster and lighter and produces the same
    // grouped layout. `One`/`Two` force the choice.
    if want_group {
        // M3: the one-pass streaming grouped route drops obsp (obsp remap in the
        // streaming writer is unsupported), while two-pass preserves it via
        // `scx sort`. Route any obsp-carrying grouped input through two-pass so
        // obsp survives regardless of X density — matching the byte-equivalent
        // convert-then-sort guarantee. `overrides.obsp` (in-memory, forwarded to
        // the two-pass plain pass) counts as obsp too.
        let has_obsp =
            overrides.obsp.as_ref().is_some_and(|v| !v.is_empty()) || h5ad_has_obsp_members(&file);
        let two_pass = match opts.group_pass {
            GroupPass::One => false,
            GroupPass::Two => true,
            GroupPass::Auto => matches!(matrix_format, MatrixFormat::Dense) || has_obsp,
        };
        // Byte-mode grouping needs a cheap per-row nnz source, which the one-pass
        // path only has for CSR. A forced one-pass over a non-CSR source with a
        // byte budget would silently degrade to row-count sizing, producing a
        // *different* layout than the `Auto`/`Two` route for the same flags —
        // error instead of diverging. (`Auto` already routes dense → two-pass.)
        if !two_pass
            && opts.group_target_bytes.is_some()
            && !matches!(matrix_format, MatrixFormat::Csr)
        {
            return Err(ConvertError::Other(format!(
                "convert --group-by --group-target-bytes with --group-pass one is unsupported for \
                 a {matrix_format:?} source (byte-budget sizing needs per-row nnz, available only \
                 for CSR in one pass); use --group-pass two (or auto) for byte-mode grouping"
            )));
        }
        // M3: a forced one-pass with obsp present would silently drop obsp. Refuse
        // rather than lose the graph; two-pass (or auto) preserves it.
        if !two_pass && has_obsp {
            return Err(ConvertError::Other(
                "convert --group-by with --group-pass one drops obsp (obsp remap in the one-pass \
                 streaming route is unsupported); use --group-pass two (or auto) to preserve obsp, \
                 or drop obsp before converting"
                    .to_string(),
            ));
        }
        if two_pass {
            log::info!(
                "convert --group-by: routing to two-pass (plain convert + scx sort) for \
                 {matrix_format:?} source"
            );
            return convert_then_sort_grouped(input, output, opts, overrides, sink);
        }
    }

    let mut x_reader: Box<dyn CsrShardStream> = match matrix_format {
        MatrixFormat::Csr => Box::new(open_x_streaming(&file, "X", matrix_format, sink)?),
        MatrixFormat::Dense => Box::new(open_dense_streaming(&file, "X", opts, sink)?),
        MatrixFormat::Csc => open_csc_streaming(&file, "X", opts, sink)?,
    };
    let n_obs = x_reader.n_obs() as usize;
    let n_vars = x_reader.n_vars() as usize;
    let n_vars_u32: u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;
    let index_dtype: u8 = index_dtype_for(n_vars as u64);

    // Placeholder header. `nnz`, `n_csr_shards`, `n_csc_shards`, and
    // `codec_id` are overwritten by `ScxWriter::finish()` from
    // running accumulators (see scx-format-io/src/writer.rs).
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        0,
        opts.shard_target_rows,
        0,
        index_dtype,
    );
    // F5 Phase 1: row-group framing produces a v4 file (its shards are v2).
    if opts.framing().is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    // F5-b: frame CSC sidecars / layers / obsp shards written through this writer
    // (CSR X shards frame via encode_one_shard). No-op unless framing is on.
    writer.set_framing(opts.framing());

    // obs / var. Override-or-disk per section: any `Some(...)` field
    // wins over the on-disk read so the backed-AnnData routing path
    // can preserve in-memory mutations.
    let obs = match overrides.obs.as_ref() {
        Some(batch) => batch.clone(),
        None => read_dataframe_group(&file, "obs", sink)?,
    };
    let var = match overrides.var.as_ref() {
        Some(batch) => batch.clone(),
        None => read_dataframe_group(&file, "var", sink)?,
    };

    // Reorder-on-convert: compute the obs-axis permutation, reorder obs to
    // match, and wrap the X reader so the coordinator gathers source rows in the
    // new order. `--group-by` (Phase 7.4) takes precedence over `--sort-by`: it
    // computes a reference-first / group-by permutation via the shared
    // `scx_ops::compute_grouped_order` (the same routine `scx sort --group-by`
    // uses, so the output is byte-equivalent to convert-then-sort), plans
    // group-aligned shard breaks, and stages the `group_index` sidecar (written
    // after X). The non-reorder path keeps `obs` / `x_reader` untouched.
    let mut group_ranges: Option<Vec<(u64, u32)>> = None;
    let mut group_index_bytes: Option<Vec<u8>> = None;
    let sort_perm: Option<std::sync::Arc<Vec<u64>>> = if let Some(group_col) = &opts.group_by {
        let go =
            scx_ops::compute_grouped_order(&obs, group_col, &opts.sort_by, opts.reference.as_ref())
                .map_err(|e| ConvertError::Other(format!("convert --group-by: {e}")))?;

        // Per-row nnz in emission order (byte-budget mode, CSR only): the h5ad
        // CSR indptr is the cheap per-row nnz source. Dense/CSC have no cheap
        // per-row nnz, so byte mode falls back to row-count with a warning.
        let (per_row_nnz, target_units, bytes_per_nnz) = match opts.group_target_bytes {
            Some(tb) if matches!(matrix_format, MatrixFormat::Csr) => {
                let indptr_ds = file.group("X")?.dataset("indptr")?;
                let indptr = crate::h5ad::read::read_i64_dataset(&indptr_ds)?;
                let prn: Vec<u64> = go
                    .perm
                    .iter()
                    .map(|&src| {
                        let s = src as usize;
                        (indptr[s + 1] - indptr[s]) as u64
                    })
                    .collect();
                (prn, tb.max(1), scx_ops::GROUP_BYTES_PER_NNZ)
            }
            Some(_) => {
                sink.emit(ConvertWarning::GroupByteModeUnsupported {
                    source_format: match matrix_format {
                        MatrixFormat::Dense => "dense".to_string(),
                        MatrixFormat::Csc => "csc".to_string(),
                        MatrixFormat::Csr => "csr".to_string(),
                    },
                });
                (Vec::new(), opts.shard_target_rows.max(1) as u64, 0u64)
            }
            None => (Vec::new(), opts.shard_target_rows.max(1) as u64, 0u64),
        };
        let max_units = opts
            .group_max_bytes
            .unwrap_or_else(|| target_units.saturating_mul(scx_ops::GROUP_MAX_BYTES_MULTIPLE));
        let plan = scx_ops::plan_group_shards(
            &go.group_of_new,
            &go.ref_of_new,
            &go.labels,
            &per_row_nnz,
            target_units,
            bytes_per_nnz,
            max_units,
        );
        group_ranges = Some(group_shard_starts_to_ranges(
            &plan.shard_starts,
            n_obs as u64,
        ));
        let payload = plan.to_sidecar_json(group_col, &go.reference_labels);
        group_index_bytes = Some(serde_json::to_vec(&payload).map_err(|e| {
            ConvertError::Other(format!(
                "convert --group-by: failed to serialize group index: {e}"
            ))
        })?);
        Some(std::sync::Arc::new(go.perm))
    } else if want_sort {
        let perm =
            crate::permuted_reader::compute_sort_perm(&obs, &opts.sort_by, opts.sort_reverse)?;
        Some(std::sync::Arc::new(perm))
    } else {
        None
    };
    let obs = match &sort_perm {
        Some(perm) => crate::permuted_reader::take_record_batch(&obs, perm)?,
        None => obs,
    };
    if let Some(perm) = &sort_perm {
        // Re-open X as an indexed reader and wrap it in the permuted gather.
        drop(x_reader);
        let inner: Box<dyn crate::stream::IndexedCsrShardStream> = match matrix_format {
            MatrixFormat::Csr => Box::new(open_x_streaming(&file, "X", matrix_format, sink)?),
            MatrixFormat::Dense => Box::new(open_dense_streaming(&file, "X", opts, sink)?),
            MatrixFormat::Csc => unreachable!("CSC + sort guarded above"),
        };
        x_reader = Box::new(crate::permuted_reader::PermutedCsrReader::new(
            inner,
            perm.clone(),
        ));
    }

    // After the `--sort-by` / `--group-by` permutation above, so the shards
    // carry post-permutation rows.
    write_ingest_obs(&mut writer, &obs, opts)?;
    writer.write_var(&var)?;

    // X shards (streaming). `run_streaming_writer_coordinator`
    // dispatches to the parallel encoder pool when the reader is
    // indexable + libhdf5 is thread-safe + the resolved thread count
    // is > 1; otherwise it falls back to the sequential path. Output
    // is byte-identical regardless of route.
    let x_opts = begin_same_pass_csc(&mut writer, opts, n_obs, n_vars)?;
    let (_csr_shard_count, csr_row_ranges) = run_streaming_writer_coordinator(
        x_reader.as_mut(),
        &mut writer,
        x_opts.as_ref().unwrap_or(opts),
        index_dtype,
        n_vars_u32,
        SectionType::CsrShard,
        ModalityType::Rna,
        // Canonical single-modality CSR shard name is `X_shard_{idx}`
        // (writer.rs::write_csr_shard, catalog_view, and the explode/push
        // name->path mapper all expect uppercase). The coordinator appends
        // `_{idx}` to this prefix. A lowercase prefix produced `x_shard_0`,
        // which the reader tolerated (it resolves shards by SectionType, not
        // name) but `scx explode`/`scx push` rejected.
        "X_shard",
        sink,
        group_ranges.as_deref(),
    )?;
    drop(x_reader);
    // The same-pass sidecar, if one was started: emitted before layers and
    // obsm so its buckets are released first.
    writer.emit_csc_sidecar()?;

    // Phase 7.4: write the `group_index` sidecar (same bytes `scx sort` emits).
    if let Some(bytes) = &group_index_bytes {
        writer.write_group_index(bytes)?;
    }

    // obsm / varm / obsp / varp / uns. The dense + sparse mappings are
    // emitted as row-sharded sections — one Arrow IPC section per
    // shard — so peak memory is bounded by `shard_target_rows` worth
    // of rows per matrix. Override path: an in-memory `RecordBatch`
    // supplied by the caller (typically pyscx's backed-routing path
    // for sections the user mutated in Python) is sliced into shards
    // on the way out, also keeping peak memory to one shard at a time.
    // Disk-streaming path: the source h5ad is hyperslab-read one
    // row-range at a time per key.
    // obsm is obs-axis → reorder it by the same permutation when sorting.
    let obsm_perm: Option<&[u64]> = sort_perm.as_ref().map(|p| p.as_slice());
    write_dense_mapping_section(
        &file,
        &mut writer,
        overrides.obsm.as_ref(),
        "obsm",
        opts.shard_target_rows,
        DenseMappingKind::Obsm,
        obsm_perm,
        sink,
    )?;
    // varm is var-axis → never reordered by an obs sort.
    write_dense_mapping_section(
        &file,
        &mut writer,
        overrides.varm.as_ref(),
        "varm",
        opts.shard_target_rows,
        DenseMappingKind::Varm,
        None,
        sink,
    )?;
    // obsp (obs×obs) needs both axes remapped through the permutation. The
    // standalone `scx sort` engine does this remap; the convert-on-sort path
    // does not yet, so drop obsp with a warning here rather than emit a
    // misaligned graph. varp (var×var) is untouched by an obs sort.
    if sort_perm.is_some() {
        if let Ok(group) = file.group("obsp") {
            for name in group.member_names().unwrap_or_default() {
                if name.starts_with("__") {
                    continue;
                }
                sink.emit(ConvertWarning::DroppedObsp {
                    name: format!("obsp/{name}"),
                    reason: "obsp remap under sort-on-convert is not yet supported \
                             (Phase 5); dropped to avoid an axis-misaligned graph"
                        .to_string(),
                });
            }
        }
    } else {
        write_sparse_mapping_section(
            &file,
            &mut writer,
            overrides.obsp.as_ref(),
            "obsp",
            opts.shard_target_rows,
            SparseMappingKind::Obsp,
            sink,
        )?;
    }
    write_sparse_mapping_section(
        &file,
        &mut writer,
        overrides.varp.as_ref(),
        "varp",
        opts.shard_target_rows,
        SparseMappingKind::Varp,
        sink,
    )?;

    // Optional `adata.raw` count matrix (its own var axis) → raw section
    // family, streamed shard-by-shard so peak RSS stays bounded (mirrors
    // the streaming `/X` path).
    ingest_raw_streaming(&file, &mut writer, n_obs, opts, sink)?;

    match overrides.uns.as_ref() {
        Some(json) => writer.write_uns(json)?,
        None => {
            if file.group("uns").is_ok() {
                let uns = read_uns(&file, opts.strict_uns, sink)?;
                writer.write_uns(&uns)?;
            }
        }
    }

    // Layers (one streaming pass per layer). Best-effort per layer:
    // an open failure (dense/CSC layer, malformed encoding, shape
    // mismatch with X) is logged and skipped, mirroring the
    // non-streaming `read_layers` warn-and-continue behaviour
    // (scx-convert/src/h5ad/read.rs). Once a layer's shards start
    // writing, a mid-stream shard error aborts — leaving a
    // half-written layer in the SCX file would be worse than failing
    // loudly. Width-dependent encoding values are recomputed per
    // layer instead of inheriting `X`'s.
    if let Ok(layers_group) = file.group("layers") {
        let layer_names = layers_group.member_names()?;
        for layer_name in &layer_names {
            let layer_path = format!("layers/{layer_name}");
            let layer_format =
                match crate::detect::detect_matrix_format_at(&file, &layer_path, sink) {
                    Ok(f) => f,
                    Err(e) => {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: layer_name.clone(),
                            reason: format!("{e}"),
                        });
                        continue;
                    }
                };
            let mut layer_reader: Box<dyn CsrShardStream> = match layer_format {
                MatrixFormat::Csr => match open_layer_streaming(&file, layer_name, sink) {
                    Ok(r) => match &sort_perm {
                        Some(p) => Box::new(crate::permuted_reader::PermutedCsrReader::new(
                            Box::new(r),
                            p.clone(),
                        )),
                        None => Box::new(r),
                    },
                    Err(e) => {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: layer_name.clone(),
                            reason: format!("{e}"),
                        });
                        continue;
                    }
                },
                MatrixFormat::Dense => {
                    match open_dense_layer_streaming(&file, layer_name, opts, sink) {
                        Ok(r) => match &sort_perm {
                            Some(p) => Box::new(crate::permuted_reader::PermutedCsrReader::new(
                                Box::new(r),
                                p.clone(),
                            )),
                            None => Box::new(r),
                        },
                        Err(e) => {
                            sink.emit(ConvertWarning::LayerSkipped {
                                name: layer_name.clone(),
                                reason: format!("{e}"),
                            });
                            continue;
                        }
                    }
                }
                MatrixFormat::Csc => {
                    // A CSC-on-disk layer can't be reordered (sequential
                    // transposer); under sort-on-convert, drop it rather than
                    // emit an axis-misaligned layer.
                    if sort_perm.is_some() {
                        sink.emit(ConvertWarning::LayerSkipped {
                            name: layer_name.clone(),
                            reason: "cannot reorder a CSC-on-disk layer during \
                                     sort-on-convert; layer dropped"
                                .to_string(),
                        });
                        continue;
                    }
                    match open_csc_layer_streaming(&file, layer_name, opts, sink) {
                        Ok(r) => r,
                        Err(e) => {
                            sink.emit(ConvertWarning::LayerSkipped {
                                name: layer_name.clone(),
                                reason: format!("{e}"),
                            });
                            continue;
                        }
                    }
                }
            };
            let l_n_obs = layer_reader.n_obs() as usize;
            if l_n_obs != n_obs {
                sink.emit(ConvertWarning::LayerSkipped {
                    name: layer_name.clone(),
                    reason: format!("n_obs {l_n_obs} does not match X n_obs {n_obs}"),
                });
                continue;
            }
            let l_n_vars = layer_reader.n_vars() as usize;
            let l_n_vars_u32 = match u32::try_from(l_n_vars) {
                Ok(v) => v,
                Err(_) => {
                    sink.emit(ConvertWarning::LayerSkipped {
                        name: layer_name.clone(),
                        reason: format!("n_vars {l_n_vars} exceeds u32::MAX"),
                    });
                    continue;
                }
            };
            let l_index_dtype: u8 = index_dtype_for(l_n_vars as u64);
            let (_, _) = run_streaming_writer_coordinator(
                layer_reader.as_mut(),
                &mut writer,
                opts,
                l_index_dtype,
                l_n_vars_u32,
                SectionType::LayerCsrShard,
                ModalityType::Rna,
                &format!("{layer_name}_shard"),
                sink,
                // Layers keep fixed-size shard breaks even under a grouped
                // convert — matching `scx sort`, which group-breaks only X.
                None,
            )?;
        }
    }

    // Phase 5a: predicate indexes from the obs/var we just wrote and
    // the actual shard boundaries reported by `streaming_writer_coordinator`.
    // Reorder-on-convert auto-indexes its keys so the contiguous shard ranges
    // are emitted: the `--group-by` column leads (Phase 7.4), then `--sort-by`.
    let reorder_keys: Vec<String> = match &opts.group_by {
        Some(group_col) => {
            let mut keys = Vec::with_capacity(opts.sort_by.len() + 1);
            keys.push(group_col.clone());
            keys.extend(opts.sort_by.iter().filter(|c| *c != group_col).cloned());
            keys
        }
        None => opts.sort_by.clone(),
    };
    let (obs_indexed, var_indexed) = build_and_write_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        n_vars,
        opts,
        &reorder_keys,
        sink,
    )?;

    // Provenance carries the streaming flag so consumers can tell at
    // a glance how the file was produced.
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let source_format_str = match matrix_format {
        MatrixFormat::Csr => "csr_matrix",
        MatrixFormat::Csc => "csc_matrix",
        MatrixFormat::Dense => "array",
    };
    let resolved_reader_threads = resolve_reader_threads(opts.reader_threads);
    // Phase 7.4: record the grouping config when `--group-by` was used (mirrors
    // `scx sort`'s `grouping_provenance`), else `null`.
    let grouping_json = match &opts.group_by {
        Some(group_by) => {
            let reference = match &opts.reference {
                Some(scx_ops::ReferenceSpec::Labels(l)) => serde_json::json!({ "labels": l }),
                Some(scx_ops::ReferenceSpec::Column(c)) => serde_json::json!({ "column": c }),
                None => serde_json::Value::Null,
            };
            serde_json::json!({
                "group_by": group_by,
                "reference": reference,
                "group_target_bytes": opts.group_target_bytes,
                "group_max_bytes": opts.group_max_bytes,
            })
        }
        None => serde_json::Value::Null,
    };
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: opts.tool.clone(),
        params_json: serde_json::json!({
            "input": input.display().to_string(),
            "format": "h5ad",
            "stream": true,
            "codec_selection": opts.codec_selection_value(),
            "source_matrix_format": source_format_str,
            "warnings": sink.summary_json(),
            "predicate_index": {
                "obs_columns": obs_indexed,
                "var_columns": var_indexed,
                "preset": opts.index_preset,
            },
            "sort_by": opts.sort_by,
            "sort_reverse": opts.sort_reverse,
            "grouping": grouping_json,
            "reader_threads": resolved_reader_threads,
            "writer_queue_depth": opts.writer_queue_depth,
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Phase 7.4 two-pass grouped convert: plain convert to a temp SCX, then
/// `scx sort --group-by` into `output`. The `Auto` group-pass routes dense
/// sources here because the one-pass streaming gather reads full rows for a
/// dense matrix (~4–5× slower / ~2× memory than this). Produces the same
/// grouped layout as a manual convert-then-sort; the temp is removed on exit.
fn convert_then_sort_grouped(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    overrides: &StreamingOverrides,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    let group_by = opts
        .group_by
        .clone()
        .expect("convert_then_sort_grouped requires group_by");

    // Temp plain SCX alongside the output (same filesystem). Grouping / sort /
    // index / bitmap / csc are stripped from the plain pass — `scx sort` owns
    // the final grouped layout, predicate index, bitmaps, and (rebuilt) CSC.
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let stem = output.file_name().and_then(|s| s.to_str()).unwrap_or("out");
    let tmp = parent.join(format!(".{stem}.grouptmp.scx"));

    let plain_opts = IngestOptions {
        group_by: None,
        reference: None,
        group_target_bytes: None,
        group_max_bytes: None,
        group_pass: GroupPass::One,
        sort_by: Vec::new(),
        sort_reverse: false,
        csc: CscPolicy::Off,
        bitmap: BitmapPolicy::Off,
        index_obs: Vec::new(),
        index_var: Vec::new(),
        index_preset: None,
        ..opts.clone()
    };
    h5ad_to_scx_streaming(input, &tmp, &plain_opts, overrides, sink)?;

    // The CSC policy is decided on the plain pass's shape — the sort keeps
    // every row and column — and honoured by the sort itself, which builds the
    // sidecar in the same pass as its X shards.
    let build_csc = {
        let reader = scx_format_io::reader::ScxReader::open(&tmp)?;
        opts.csc.should_build_csc(reader.n_obs(), reader.n_vars())
    };

    let mut by = Vec::with_capacity(opts.sort_by.len() + 1);
    by.push(group_by.clone());
    by.extend(opts.sort_by.iter().filter(|c| **c != group_by).cloned());
    let sort_opts = scx_ops::SortOptions {
        by,
        reverse: false,
        // Convert-time grouping is a key sort by construction; 1D's shuffle is
        // exposed only on `scx sort` / `pyscx.shuffle`.
        shuffle: None,
        shard_target_rows: opts.shard_target_rows,
        // Carry this convert's own codec intent into the grouping sort, so
        // `scx convert --group-by --codec auto` gets the same adaptive
        // per-shard selection as the ungrouped path. Reconstructed from the
        // fields `resolve_codec` populated on `IngestOptions` rather than
        // collapsed to Auto-or-explicit, which is what dropped `decode_target`
        // and left the grouped path on the `fast` heuristic.
        codec: scx_format_io::ResolvedCodec {
            explicit_codec: opts.codec,
            codec_trial: opts.codec_trial,
            decode_target: opts.decode_target,
            // Framing was already validated for this convert; the sort inherits
            // the output's framing rather than deciding it.
            requires_framing: false,
            profile: "auto",
        },
        index_options: ConversionPredicateIndexOptions {
            index_obs: opts.index_obs.clone(),
            index_var: opts.index_var.clone(),
            index_preset: opts.index_preset.clone(),
            index_auto_threshold: opts.index_auto_threshold,
        },
        memory_budget: opts.memory_budget,
        temp_dir: opts.temp_dir.clone(),
        bitmap: opts.bitmap,
        group_by: Some(group_by),
        reference: opts.reference.clone(),
        group_target_bytes: opts.group_target_bytes,
        group_max_bytes: opts.group_max_bytes,
        // None => the sort engine's default block cap (256 MB), giving the
        // convert two-pass sort the F6 grouped-write OOM fix for free.
        group_write_block_bytes: None,
        csc: scx_ops::CscCarryOptions {
            mode: if build_csc {
                scx_ops::CscOutput::Always
            } else {
                scx_ops::CscOutput::Off
            },
            cols_per_shard: opts.csc_cols_per_shard,
            memory_limit: crate::budget::csc_sidecar_bytes(opts.memory_budget).to_string(),
            temp_dir: opts.temp_dir.clone(),
            framing: opts.framing_preserving_codec(),
        },
    };
    let sort_result = scx_ops::sort(&tmp, output, &sort_opts)
        .map_err(|e| ConvertError::Other(format!("convert --group-by (two-pass sort): {e}")));
    // Always clean up the temp, even on sort failure.
    let _ = std::fs::remove_file(&tmp);
    sort_result?;
    Ok(())
}

/// Streaming 10x HDF5 → SCX conversion. Reads `/matrix` one shard's worth of
/// rows at a time via [`crate::h5ad::stream::open_tenx_x_streaming`], so peak
/// memory is bounded by the always-resident `indptr` (`(n_cells + 1) × 8`
/// bytes), `obs` / `var`, and one shard's working set per outstanding worker —
/// instead of the whole `indptr` + `indices` + `data` triple the eager
/// [`tenx_to_scx`](super::entry::tenx_to_scx) holds.
///
/// Shard processing dispatches through [`run_streaming_writer_coordinator`],
/// exactly as the h5ad path does, so the rayon fan-out, the
/// `H5is_library_threadsafe` probe, the sequential fallback and the
/// `--memory-budget` derate all apply unchanged.
///
/// **This is not byte-identical to the eager path, and neither is h5ad's.**
/// `tenx_to_scx` detects one value encoding over the whole matrix and forces the
/// resulting codec into every shard; the coordinator passes `opts.codec`
/// through, which is `None` under the default `--codec auto`, so each shard
/// seeds its own codec from its own values. Since `select_codec` samples only
/// the first 10 000 values, shard 0 agrees by construction and later shards can
/// differ — and a shard that picked a different codec differs in its payload,
/// its shard header and its catalog length and checksum, so the claim is
/// **not** "every section but `Provenance` is identical". What holds under
/// `auto` is: the decoded values and the shard row ranges agree, and the
/// non-CSR sections agree. Naming a codec makes the two paths agree on every
/// section's content (`Strictness::Content`, which excludes `Provenance` and
/// does not pin physical offsets). Both are pinned by
/// `convert_tests_tenx_stream`, and nothing here claims a whole-file byte
/// compare.
///
/// 10x has no `uns` / `obsm` / `varm` / `obsp` / `varp` / `layers` / `raw` and
/// the CLI rejects `--sort-by` / `--group-by` on this direction, so this is the
/// whole sequence: header → obs/var → X → indexes → provenance → CSC.
pub fn tenx_to_scx_streaming(
    input: &Path,
    output: &Path,
    opts: &IngestOptions,
    sink: &mut WarningSink,
) -> Result<(), ConvertError> {
    // Same gate as the eager path: an h5ad misrouted here, and a CellBender
    // output redirected to `scx cellbender-import`.
    let file = super::open_tenx_input(input)?;

    // Open the reader first — the header needs the shape before any section
    // write, and `n_obs` is the *cell* axis (10x's `shape` is gene-major).
    let mut x_reader = crate::h5ad::stream::open_tenx_x_streaming(&file)?;
    let n_obs = x_reader.n_obs;
    let n_vars = x_reader.n_vars;
    let n_vars_u32: u32 = u32::try_from(n_vars)
        .map_err(|_| ConvertError::Other(format!("n_vars {n_vars} exceeds u32::MAX")))?;
    let index_dtype: u8 = index_dtype_for(n_vars as u64);

    // Placeholder header. `nnz`, `n_csr_shards` and `codec_id` are overwritten
    // by `ScxWriter::finish()` from running accumulators — the eager path can
    // stamp them up front only because it has the whole matrix in hand.
    let mut header = FileHeader::new_single_modality(
        n_obs as u64,
        n_vars as u64,
        0,
        opts.shard_target_rows,
        0,
        index_dtype,
    );
    if opts.framing().is_some() {
        header.format_version = scx_format_io::header::CURRENT_FORMAT_VERSION;
    }

    let mut writer = ScxWriter::new(output, header)?;
    writer.set_framing(opts.framing());

    let matrix = file.group("matrix")?;
    let obs = crate::tenx_read::read_tenx_obs(&matrix, n_obs)?;
    let var = crate::tenx_read::read_tenx_var(&matrix, n_vars)?;
    write_ingest_obs(&mut writer, &obs, opts)?;
    writer.write_var(&var)?;

    // `"X_shard"` uppercase is load-bearing (see the h5ad call site above);
    // `explicit_ranges = None` because this direction has no reorder.
    let x_opts = begin_same_pass_csc(&mut writer, opts, n_obs, n_vars)?;
    let (_csr_shard_count, csr_row_ranges) = run_streaming_writer_coordinator(
        &mut x_reader,
        &mut writer,
        x_opts.as_ref().unwrap_or(opts),
        index_dtype,
        n_vars_u32,
        SectionType::CsrShard,
        ModalityType::Rna,
        "X_shard",
        sink,
        None,
    )?;
    drop(x_reader);
    writer.emit_csc_sidecar()?;

    // Ranges come from the coordinator, never from an assumed partition.
    let (obs_indexed, var_indexed) = build_and_write_predicate_indexes(
        &mut writer,
        &obs,
        &var,
        &csr_row_ranges,
        n_vars,
        opts,
        &[],
        sink,
    )?;

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let resolved_reader_threads = resolve_reader_threads(opts.reader_threads);
    writer.write_provenance(vec![ProvenanceEntry {
        timestamp,
        action: "convert".to_string(),
        tool: opts.tool.clone(),
        params_json: serde_json::json!({
            "input": input.display().to_string(),
            "format": "10x",
            "stream": true,
            "codec_selection": opts.codec_selection_value(),
            "warnings": sink.summary_json(),
            "predicate_index": {
                "obs_columns": obs_indexed,
                "var_columns": var_indexed,
                "preset": opts.index_preset,
            },
            "reader_threads": resolved_reader_threads,
            "writer_queue_depth": opts.writer_queue_depth,
        })
        .to_string(),
        input_checksums: vec![],
    }])?;

    writer.finish()?;
    Ok(())
}

/// Start a CSC sidecar in the same pass as X when `opts.csc` asks for one on
/// an `n_obs` x `n_vars` matrix. Returns the options the X coordinator must run
/// with: under a `--memory-budget` the builder's buckets are live beside the
/// ingest workers, so ingest sizes itself against what the builder leaves
/// (`budget::csc_same_pass_split`). `None` means "run X with `opts`".
///
/// Pair with `writer.emit_csc_sidecar()` right after X. The sidecar is the one
/// a second pass (`scx_ops::rebuild_csc_inplace`) would have built — same
/// budget, width and framing — without re-reading the output.
fn begin_same_pass_csc(
    writer: &mut ScxWriter,
    opts: &IngestOptions,
    n_obs: usize,
    n_vars: usize,
) -> Result<Option<IngestOptions>, ConvertError> {
    if !opts.csc.should_build_csc(n_obs as u64, n_vars as u64) {
        return Ok(None);
    }
    let (spill_after_bytes, ingest_budget) = crate::budget::csc_same_pass_split(opts.memory_budget);
    writer.enable_csc_sidecar(scx_format_io::CscBuildOptions {
        cols_per_shard: opts.csc_cols_per_shard,
        memory_bytes: crate::budget::csc_sidecar_bytes(opts.memory_budget) as usize,
        spill_after_bytes,
        // `--temp-dir` already names the spill root for the external CSC
        // transpose; the builder's buckets are the same kind of spill.
        spill_root: opts.temp_dir.clone(),
        // So `--csc <policy> --row-group-rows N` frames the sidecar at X's G.
        framing: opts.framing_preserving_codec(),
    })?;
    Ok(
        (ingest_budget != opts.memory_budget).then(|| IngestOptions {
            memory_budget: ingest_budget,
            ..opts.clone()
        }),
    )
}
