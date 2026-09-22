// SCX -> SCX streaming rewrite (backed / lazy sources).
//
// Extracted from the former pyscx/src/anndata.rs (T5.7).

use arrow::array::RecordBatch;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scx_codec::CodecId;
use scx_format_io::section::SectionType;
use scx_format_io::{ProvenanceEntry, ScxReader, ScxWriter};

use crate::to_pyerr;

use super::*;

/// Warn that a source file's `adata.raw` will not reach the output.
///
/// The SCX → SCX routes write the sections the in-memory AnnData holds,
/// and `to_anndata(backed=True)` sets `.raw` to `None` — so raw is absent
/// from the object, not deliberately dropped by the caller. Silence here
/// would be the worst case: the read side already told the user "the
/// on-disk raw sections are preserved", which is true of the *source* and
/// says nothing about the file being written. Hence
/// [`ConvertWarning::DroppedRawOnWrite`] rather than the read-side
/// `DroppedRaw`, whose reassurance would be actively wrong here.
fn warn_source_raw_dropped(py: Python<'_>, src_reader: &ScxReader) -> PyResult<()> {
    if src_reader.header().has_raw() {
        warn_python_convert(
            py,
            &scx_convert::ConvertWarning::DroppedRawOnWrite {
                raw_n_vars: src_reader.raw_n_vars().unwrap_or(0),
                reason: "this path writes the sections the in-memory AnnData holds, and \
                         backed reconstruction leaves .raw unset. Convert from the h5ad \
                         (pyscx.from_h5ad), or from an in-memory AnnData whose .raw is set, \
                         to keep raw",
            },
        )?;
    }
    Ok(())
}

/// How a source file's CSR shards are *actually* encoded.
///
/// The file header's `codec_id` is only a **default**: `ShardHeader::codec_id`
/// "overrides file header" (`scx-format/src/shard.rs`), and `docs/codec.md` §1
/// requires a reader to use the shard header, not the file header. Two
/// consequences make the file header unsafe to inherit as an encode codec:
///
/// * a writer that selects per shard (`codec="auto"`, the default) leaves a
///   file header that need not describe any shard; and
/// * the streaming h5ad → SCX converter builds its header with a `0`
///   placeholder and nothing ever replaces it, so its files claim
///   `codec_id = 0` (`none`, raw little-endian) over genuinely compressed
///   shards.
///
/// Inheriting that value re-encoded every shard **raw**. Measured on the
/// 2026-08-29 Replogle artifact (966,728 × 6,143, nnz 2,671,238,445, file
/// header `none`): dropping 537 rows turned a 2.19 GB source into a 10.85 GB
/// output — 4 bytes/nnz, exactly its uncompressed size. A far larger subset of
/// a *different* source whose header said `scx1` stayed compact, which is why
/// the blowup looked like it depended on the subset.
///
/// So resolve the default against the shards themselves.
struct SourceCsrCodec {
    /// The first CSR shard's codec — a truthful *default* for the output
    /// header, which is a hint rather than a per-shard contract.
    first: Option<CodecId>,
    /// Whether every CSR shard shares [`Self::first`]. When they differ, no
    /// single codec describes the source and the adaptive per-shard heuristic
    /// should choose again rather than have one imposed on it.
    uniform: bool,
}

/// Read the source's CSR shard codecs.
///
/// Cheap: 76-byte shard-header reads off the mmap — the same reads the framing
/// precondition in [`route_scx_backed_to_scx`] already performs.
///
/// `modality_id` scopes the scan when the wrapper addresses a single modality
/// of a multimodal source. Without it, an RNA-`Scx1` / ADT-`Zstd` file reports
/// non-uniform for *either* slice — safe (it falls back to adaptive) but blind
/// to the fact that each modality is internally uniform.
fn source_csr_codec(src_reader: &ScxReader, modality_id: Option<u8>) -> SourceCsrCodec {
    let mut first: Option<CodecId> = None;
    let mut uniform = true;
    for entry in src_reader.catalog().entries.iter().filter(|e| {
        matches!(e.section_type, SectionType::CsrShard)
            && modality_id.is_none_or(|m| e.modality_id == m)
    }) {
        let Ok(shard) = src_reader.read_shard_header(entry) else {
            // An unreadable shard header means we cannot claim uniformity;
            // fall back to adaptive selection rather than guess.
            return SourceCsrCodec {
                first,
                uniform: false,
            };
        };
        let Some(codec) = CodecId::from_u8(shard.codec_id) else {
            return SourceCsrCodec {
                first,
                uniform: false,
            };
        };
        match first {
            None => first = Some(codec),
            // Nothing later can restore uniformity, and `first` is already
            // recorded, so stop reading shard headers here.
            Some(seen) if seen != codec => {
                uniform = false;
                break;
            }
            Some(_) => {}
        }
    }
    SourceCsrCodec { first, uniform }
}

/// Route an AnnData with `adata.X = ScxBackedSparseDataset` through
/// an SCX → SCX streaming writer.
///
/// Two modes:
///
/// * **Byte-passthrough**: when the source and target shard layouts
///   agree (same `shard_target_rows`, same codec, no row deletions,
///   no column projection, source is built from a single modality
///   with `modality_id == None`), pre-encoded CSR shards are copied
///   from the source file into the target writer via
///   [`ScxWriter::copy_section_verbatim`]. No decode + re-encode.
/// * **Decode + encode**: otherwise the writer iterates the
///   wrapper's user-visible shard boundaries, calls
///   `wrapper[start:end]` to materialise each shard as a scipy CSR
///   (deletions and column projection already applied by the
///   wrapper), then hands it to [`scx_format_io::encode_one_shard`]
///   plus [`ScxWriter::write_preencoded_shard`].
///
/// `shard_size_overridden` reflects whether the caller passed
/// `shard_size=…`. When `explicit_codec` is `None` and `shard_size`
/// was not overridden *and* every other precondition holds, the
/// rewrite is byte-faithful; otherwise it falls back to
/// decode + encode.
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_scx_backed_to_scx(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    backed: &crate::backed::ScxBackedSparseDataset,
    out_path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format_io::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
    shard_size_overridden: bool,
) -> PyResult<()> {
    let src_path = backed.source_path().ok_or_else(|| {
        pyo3::exceptions::PyNotImplementedError::new_err(
            "ScxBackedSparseDataset has no known source path; the SCX → SCX writer needs a \
             path-backed wrapper (constructed via pyscx.open(path).to_anndata(backed=True)).",
        )
    })?;
    let src_path_owned = src_path.to_path_buf();

    let src_reader = ScxReader::open(&src_path_owned).map_err(to_pyerr)?;
    // Before any write path is chosen — passthrough and decode-encode both
    // drop raw, so the notice belongs here rather than in either branch.
    warn_source_raw_dropped(py, &src_reader)?;
    let src_header = src_reader.header().clone();
    let src_n_obs = src_header.n_obs;
    let src_n_vars = src_header.n_vars;
    let src_codec_id = src_header.codec_id;
    let src_shard_rows = src_header.shard_target_rows;
    let src_has_csc = src_header.has_csc();
    let modality_id = backed.modality_id;

    // User-visible dimensions after any row deletions / column
    // projection. Used by the decode-encode path's output header so
    // the new file's shape matches what the AnnData wrapper exposes.
    let (out_n_obs_visible, out_n_vars_visible) =
        (backed.shape_val.0 as u64, backed.shape_val.1 as u64);

    // What the source's shards actually use, as opposed to what its file
    // header claims. Needed before the passthrough gate below, not just for
    // the encode codec. See `SourceCsrCodec`.
    let src_shard_codec = source_csr_codec(&src_reader, modality_id);

    // Passthrough preconditions. Any false → fall through to
    // decode-encode.
    let target_codec_for_passthrough = match explicit_codec {
        // An explicit codec has to hold for every shard we would copy
        // verbatim. Comparing it against the *file header* let
        // `codec="scx1"` pass through a source whose shards were half
        // `zstd`, silently ignoring the request and returning a file that
        // does not honour it. `codec="none"` — the GDS fast path — is where
        // that matters most. A source with no CSR shard has nothing to copy,
        // so `first == None` correctly refuses passthrough here.
        Some(c) => src_shard_codec.uniform && src_shard_codec.first == Some(c),
        None => true,
    };
    let target_shard_rows_matches = if shard_size_overridden {
        shard_target_rows == src_shard_rows
    } else {
        true
    };
    let no_deletions = backed.kept_to_global.is_none();
    let no_projection = backed.col_projection().is_none();
    let single_modality_source = modality_id.is_none();
    // Framing gate. A v4 source is passthrough-eligible when *every* CSR-class
    // shard is row-group-framed (shard-v2): the block index is embedded in the
    // shard body, so the verbatim copy path (`copy_section_verbatim`, CSR entries
    // only) carries it correctly and each v2 shard passes the writer's v4 guard.
    // A legacy v4 source that still holds any unframed (shard-v1) CSR shard would
    // trip that guard on verbatim copy, so it takes the decode-encode path (which
    // reframes). Checked by reading each CSR-class shard header (cheap 76-byte
    // reads from the mmap) rather than by any section-level marker.
    let source_unframed = src_header.format_version <= scx_format_io::DEFAULT_WRITE_FORMAT_VERSION;
    let source_pure_framed = src_header.format_version <= scx_format_io::CURRENT_FORMAT_VERSION
        && src_reader
            .catalog()
            .entries
            .iter()
            .filter(|e| {
                matches!(
                    e.section_type,
                    SectionType::CsrShard | SectionType::LayerCsrShard | SectionType::ObspCsrShard
                )
            })
            .all(|e| {
                src_reader
                    .read_shard_header(e)
                    .map(|h| {
                        h.shard_format_version
                            > scx_format_io::shard::DEFAULT_WRITE_SHARD_FORMAT_VERSION
                    })
                    .unwrap_or(false)
            });
    let passthrough_ok = target_codec_for_passthrough
        && target_shard_rows_matches
        && no_deletions
        && no_projection
        && single_modality_source
        && (source_unframed || source_pure_framed);

    // Output header / writer setup. For passthrough, mirror the
    // source's codec / shard_target_rows / index_dtype so the
    // catalog and per-shard headers stay self-consistent. On the
    // decode-encode fallback, default to the source's codec choice
    // (preserves Scx1/Pcodec/etc — only override when the caller
    // passes `codec=`).
    let out_codec_id: u8 = if passthrough_ok {
        // Seed from the shards, not `src_codec_id`. This is only a
        // pre-encode placeholder either way — `ScxWriter::finish` restamps it
        // from the first CSR shard the writer sees, and on this branch that is
        // the verbatim copy — but seeding it from the source's file header
        // would mean carrying forward the very value the rest of this function
        // argues is untrustworthy.
        src_shard_codec
            .first
            .map(|c| c as u8)
            .unwrap_or(src_codec_id)
    } else {
        match explicit_codec {
            Some(c) => c as u8,
            // Deliberately NOT `src_codec_id`: a file-header codec need not
            // describe any shard, and inheriting a `none` header wrote every
            // shard raw. See `SourceCsrCodec`.
            None => src_shard_codec
                .first
                .map(|c| c as u8)
                .unwrap_or(src_codec_id),
        }
    };
    let out_shard_rows = if passthrough_ok {
        src_shard_rows
    } else {
        shard_target_rows
    };
    // Key index_dtype off the user-visible n_vars (after any column
    // projection), not the source's. Allows u16 when projecting a
    // large source down to a small gene subset, and matches the
    // in-memory path which sees only the visible shape.
    let out_index_dtype = if passthrough_ok {
        src_header.index_dtype
    } else if out_n_vars_visible <= 65535 {
        0
    } else {
        1
    };

    let codec_for_header = CodecId::from_u8(out_codec_id).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "unknown codec id {out_codec_id} from source SCX header"
        ))
    })?;
    // Output dimensions: passthrough mirrors the source header
    // (preconditions guarantee no deletions / projection). The
    // decode-encode path uses the wrapper's user-visible shape so
    // the rewrite drops any deleted rows and respects column
    // projection.
    let (out_n_obs, out_n_vars) = if passthrough_ok {
        (src_n_obs, src_n_vars)
    } else {
        (out_n_obs_visible, out_n_vars_visible)
    };
    let mut header = build_output_header(
        out_n_obs,
        out_n_vars,
        out_shard_rows,
        codec_for_header,
        out_index_dtype,
        src_header.format_version,
    );
    // Passthrough byte-copies the source's shards verbatim, so the output must
    // carry the source's exact `format_version` — NOT the `rewrite_output_
    // format_version` clamp (which caps at v3). A pure-framed v4 source holds
    // shard-v2 shards with per-group local-rebased BlockIndexes; stamping a v3
    // header over them would let a v3-only reader accept the file and mis-decode
    // each group's local indptr as global. Mirror the source version exactly.
    if passthrough_ok {
        header.format_version = src_header.format_version;
    }
    // When the passthrough output is a framed (v4) file, every CSR-class section
    // it contains must be framed — the X shards are copied verbatim (already v2),
    // but the layers are decode-encoded below and would otherwise be written
    // unframed (v1), which `guard_no_legacy_shard_in_v4` rejects. Frame them.
    let out_framing = if header.format_version >= scx_format_io::CURRENT_FORMAT_VERSION {
        Some(scx_format_io::FramingConfig::default())
    } else {
        None
    };

    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // Extract metadata overrides up front so the writer can interleave
    // metadata writes with shard I/O in the canonical order.
    let ov = extract_scx_overrides(py, adata, uns_format_parsed)?;

    // CSC sidecar policy. Resolve `Auto` against the output shape so a
    // CSC sidecar is (re)built only when the rewritten dataset is large
    // enough to benefit.
    let csc_build = csc_policy.should_build_csc(out_n_obs, out_n_vars);
    // An empty output has no sidecar to rebuild under any policy, so telling
    // the caller to pass `csc="always"` would be wrong; the source's sidecar
    // is simply gone with its rows.
    let csc_dropped = src_has_csc && !csc_build && out_n_obs > 0 && out_n_vars > 0;
    if csc_dropped {
        warn_csc_dropped(py);
    }

    // Write obs/var first.
    py.detach(|| -> Result<(), scx_format_io::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    let n_vars_u32 = u32::try_from(out_n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {out_n_vars} exceeds u32::MAX")))?;
    // The codec every re-encoded shard is forced to. An explicit `codec=`
    // wins outright. Otherwise the source's *shards* decide:
    //
    // * all CSR shards agree -> reuse it, which preserves a deliberate
    //   `codec="none"` (the GDS fast path) and a uniform Scx1/Pcodec source;
    // * they disagree, or the source has no readable CSR shard -> `None`, so
    //   `encode_one_shard` re-runs the per-shard adaptive heuristic instead of
    //   having one shard's choice imposed on every other shard.
    //
    // Pinning this to the *file header* is what produced the 5x blowup
    // (`SourceCsrCodec`). The header stays a hint: per `docs/codec.md` §1 a
    // reader takes each shard's codec from that shard's own header, so an
    // adaptive mix needs no single header value to be true of every shard.
    let codec_for_encode = match explicit_codec {
        Some(_) => Some(codec_for_header),
        None if src_shard_codec.uniform => src_shard_codec.first,
        None => None,
    };

    // The sidecar, when the policy asks for one, is built in the same pass
    // as X — from the copied bytes on the passthrough branch, from the
    // encoded shards otherwise — and emitted before the layers.
    if csc_build {
        writer
            .enable_csc_sidecar(csc_build_options(csc_cols_per_shard))
            .map_err(to_pyerr)?;
    }
    if passthrough_ok {
        // Byte-passthrough. Iterate source CSR shards in row order;
        // copy each verbatim. `modality_id == None` is already
        // enforced above, so we can write at the global modality
        // (current_modality_id == 0).
        let csr_shards = src_reader.catalog().csr_shards_sorted();
        py.detach(|| -> Result<(), scx_format_io::ScxError> {
            for entry in csr_shards {
                let bytes = src_reader.read_raw_shard_bytes(entry)?;
                writer.copy_section_verbatim(entry, bytes)?;
            }
            Ok(())
        })
        .map_err(to_pyerr)?;
    } else {
        // Decode + encode. Drive iteration over user-visible shard
        // boundaries so deletions / column projection already apply
        // via the wrapper's `__getitem__`.
        let bounds = compute_wrapper_boundaries_backed(backed, out_shard_rows);
        let adata_x = adata.getattr("X")?;
        for (i, (start, end)) in bounds.iter().enumerate() {
            let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
            // Use the Python-visible wrapper to honour deletion /
            // projection semantics. Calling through PyAny gives us
            // the wrapper's __getitem__ (returns scipy CSR).
            let shard_obj = adata_x.call_method1("__getitem__", (py_slice,))?;
            let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
                let mut enc_opts = scx_format_io::EncodeShardOptions::new(
                    format!("X_shard_{i}"),
                    SectionType::CsrShard,
                    n_vars_u32 as u64,
                    *start as u64,
                    out_index_dtype,
                );
                enc_opts.explicit_codec = codec_for_encode;
                py.detach(|| scx_format_io::encode_one_shard(indptr, indices, data, &enc_opts))
                    .map_err(to_pyerr)
            })?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
        }
    }
    py.detach(|| writer.emit_csc_sidecar()).map_err(to_pyerr)?;

    // Layers (decode-encode, never passthrough — keeps the byte
    // path bounded to X). `adata.layers` from a backed AnnData
    // contains `ScxBackedLayerDataset` instances which slice-via-
    // `__getitem__` exactly like X.
    stream_write_layers(
        py,
        adata,
        &mut writer,
        out_n_obs,
        out_n_vars,
        out_shard_rows,
        codec_for_encode,
        out_index_dtype,
        out_framing,
    )?;

    // Write remaining metadata (obsm / varm / obsp / varp / uns) after
    // the X shards, matching the canonical layout. obsm / varm / obsp /
    // varp are emitted as row-sharded sections so the on-disk layout
    // matches what the streaming pipeline produces (readers handle
    // both sharded and legacy single-section layouts transparently).
    py.detach(|| -> Result<(), scx_format_io::ScxError> {
        write_mapping_overrides(&mut writer, &ov, out_shard_rows)?;
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance: x_source / passthrough / source_path / csc_dropped.
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let params_json = serde_json::json!({
        "x_source": "backed",
        "passthrough": passthrough_ok,
        "source_path": src_path_owned.display().to_string(),
        "csc_dropped": csc_dropped,
    });
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}

/// Route an AnnData with `adata.X = ScxLazyTransformedDataset`
/// through an SCX → SCX streaming writer.
///
/// Always decode + encode. The wrapper's `__getitem__` returns the
/// transformed scipy CSR for the requested row slice; the writer
/// hands it to `encode_one_shard` and `write_preencoded_shard`.
/// Any source CSC sidecar is invalidated by the transforms and is
/// dropped with a `UserWarning` unless `csc="always"` is passed (in
/// which case it is rebuilt post-finalise).
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_scx_lazy_to_scx(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    lazy: &crate::lazy_transform::ScxLazyTransformedDataset,
    out_path: &str,
    explicit_codec: Option<CodecId>,
    shard_target_rows: u32,
    csc_policy: scx_format_io::CscPolicy,
    csc_cols_per_shard: usize,
    uns_format_parsed: UnsFormat,
) -> PyResult<()> {
    let (n_obs_usize, n_vars_usize) = lazy.shape_val;
    let n_obs = n_obs_usize as u64;
    let n_vars = n_vars_usize as u64;
    // Resolve `Auto` against the (lazy) source shape.
    let csc_build = csc_policy.should_build_csc(n_obs, n_vars);

    // Open the source once for format_version. The lazy per-shard re-encode
    // applies value transforms only (no column reorder), so it preserves a
    // canonical source but cannot canonicalize a pre-v3 one — gate the v3
    // stamp on the source version.
    let src_reader = lazy.source_path().and_then(|p| ScxReader::open(p).ok());
    if let Some(ref r) = src_reader {
        warn_source_raw_dropped(py, r)?;
    }
    let src_format_version: u16 = src_reader
        .as_ref()
        .map(|r| r.header().format_version)
        // Unframed rewrite fallback: the default (v3), not the max-readable v4.
        .unwrap_or(scx_format_io::DEFAULT_WRITE_FORMAT_VERSION);
    // The source codec is NOT a default here, for two independent reasons.
    //
    // 1. It used to be read from the *file header*, which need not describe any
    //    shard — the same defect `SourceCsrCodec` documents for the backed
    //    route. A source whose header says `none` forced every transformed
    //    shard to be written raw.
    // 2. Even a truthful source codec is the wrong default on this route: the
    //    transforms have already rewritten the values (`normalize_total` /
    //    `log1p` produce f32 from integer counts), so the codec that suited the
    //    source's integers does not suit the output's floats.
    //
    // So an explicit `codec=` wins, and otherwise each output shard re-selects
    // adaptively from what it actually holds. The header value below is only the
    // pre-encode placeholder; `ScxWriter::finish` restamps it from the first
    // shard actually written.
    let codec_for_encode = explicit_codec;
    let out_codec = explicit_codec.unwrap_or(CodecId::Zstd);
    let index_dtype: u8 = if n_vars <= 65535 { 0 } else { 1 };
    let n_vars_u32 = u32::try_from(n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {n_vars} exceeds u32::MAX")))?;

    let header = build_output_header(
        n_obs,
        n_vars,
        shard_target_rows,
        out_codec,
        index_dtype,
        src_format_version,
    );
    let mut writer = ScxWriter::new(out_path, header).map_err(to_pyerr)?;

    // Source CSC sidecar (if any) is always invalidated by the
    // transform chain. Warn unless the user opted into a rebuild.
    let src_has_csc = lazy.backed_csc.is_some();
    // Same as the backed route: no rebuild advice on an empty output.
    let csc_dropped = src_has_csc && !csc_build && n_obs > 0 && n_vars > 0;
    if csc_dropped {
        warn_csc_dropped(py);
    }

    let ov = extract_scx_overrides(py, adata, uns_format_parsed)?;

    py.detach(|| -> Result<(), scx_format_io::ScxError> {
        writer.write_obs(&ov.obs)?;
        writer.write_var(&ov.var)?;
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Lazy transforms always force decode + encode. Iterate the
    // wrapper's user-visible shard boundaries so any deletion vector
    // already applies. `codec_for_encode` was resolved above: explicit
    // `codec=` or per-shard adaptive, never the source's file header.
    let bounds = compute_wrapper_boundaries_lazy(lazy, shard_target_rows);
    let adata_x = adata.getattr("X")?;
    if csc_build {
        writer
            .enable_csc_sidecar(csc_build_options(csc_cols_per_shard))
            .map_err(to_pyerr)?;
    }
    for (i, (start, end)) in bounds.iter().enumerate() {
        let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
        let shard_obj = adata_x.call_method1("__getitem__", (py_slice,))?;
        let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
            let mut enc_opts = scx_format_io::EncodeShardOptions::new(
                format!("X_shard_{i}"),
                SectionType::CsrShard,
                n_vars_u32 as u64,
                *start as u64,
                index_dtype,
            );
            enc_opts.explicit_codec = codec_for_encode;
            py.detach(|| scx_format_io::encode_one_shard(indptr, indices, data, &enc_opts))
                .map_err(to_pyerr)
        })?;
        py.detach(|| writer.write_preencoded_shard(pre))
            .map_err(to_pyerr)?;
    }
    py.detach(|| writer.emit_csc_sidecar()).map_err(to_pyerr)?;

    // Layers — never transformed by the lazy X chain, so we just
    // stream them through the same decode-encode pipeline as X. The lazy path
    // always decode-encodes to an unframed v3 output, so no framing.
    stream_write_layers(
        py,
        adata,
        &mut writer,
        n_obs,
        n_vars,
        shard_target_rows,
        codec_for_encode,
        index_dtype,
        None,
    )?;

    py.detach(|| -> Result<(), scx_format_io::ScxError> {
        write_mapping_overrides(&mut writer, &ov, shard_target_rows)?;
        if let Some(ref uns_json) = ov.uns {
            writer.write_uns(uns_json)?;
        }
        Ok(())
    })
    .map_err(to_pyerr)?;

    // Provenance: lazy_transforms summary.
    let transforms_repr: Vec<serde_json::Value> = lazy
        .transforms()
        .iter()
        .map(|t| match t {
            crate::lazy_transform::Transform::NormalizeTotal {
                row_sums,
                target_sum,
            } => serde_json::json!({
                "name": "normalize_total",
                "params": { "row_sums_len": row_sums.len(), "target_sum": target_sum },
            }),
            crate::lazy_transform::Transform::Log1p => serde_json::json!({
                "name": "log1p",
                "params": {},
            }),
            crate::lazy_transform::Transform::RowScale { factors } => serde_json::json!({
                "name": "row_scale",
                "params": { "factors_len": factors.len() },
            }),
            crate::lazy_transform::Transform::Scale { factor } => serde_json::json!({
                "name": "scale",
                "params": { "factor": factor },
            }),
        })
        .collect();
    let source_path_json: serde_json::Value = lazy
        .source_path()
        .map(|p| serde_json::Value::String(p.display().to_string()))
        .unwrap_or(serde_json::Value::Null);
    let params_json = serde_json::json!({
        "x_source": "lazy",
        "passthrough": false,
        "source_path": source_path_json,
        "lazy_transforms": transforms_repr,
        "csc_dropped": csc_dropped,
    });
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    writer
        .write_provenance(vec![ProvenanceEntry {
            timestamp,
            action: "from_anndata".to_string(),
            tool: format!("pyscx {}", env!("CARGO_PKG_VERSION")),
            params_json: params_json.to_string(),
            input_checksums: vec![],
        }])
        .map_err(to_pyerr)?;

    writer.finish().map_err(to_pyerr)?;
    Ok(())
}

/// Compute output shard boundaries for the backed decode-encode
/// path. Sized by the **target** `shard_target_rows` (not the
/// source's), since the user may have asked for a different chunk
/// size — and that's the only reason we're on the decode-encode path
/// instead of byte-passthrough.
pub(crate) fn compute_wrapper_boundaries_backed(
    backed: &crate::backed::ScxBackedSparseDataset,
    target_shard_rows: u32,
) -> Vec<(usize, usize)> {
    let n_obs = backed.shape_val.0;
    chunk_boundaries(n_obs, target_shard_rows as usize)
}

pub(crate) fn compute_wrapper_boundaries_lazy(
    lazy: &crate::lazy_transform::ScxLazyTransformedDataset,
    target_shard_rows: u32,
) -> Vec<(usize, usize)> {
    let n_obs = lazy.shape_val.0;
    chunk_boundaries(n_obs, target_shard_rows as usize)
}

pub(crate) fn chunk_boundaries(n_obs: usize, target_rows: usize) -> Vec<(usize, usize)> {
    if n_obs == 0 || target_rows == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n_obs.div_ceil(target_rows));
    let mut start = 0usize;
    while start < n_obs {
        let end = (start + target_rows).min(n_obs);
        out.push((start, end));
        start = end;
    }
    out
}

/// Stream `adata.layers` shard-by-shard into the writer using the
/// same decode-encode pattern as the X path. Shared by both the
/// backed and lazy SCX → SCX routes — neither transforms layers,
/// so the logic is identical.
///
/// Each layer wrapper (`ScxBackedLayerDataset` or a scipy CSR) must
/// support `__getitem__(slice)` and report `(n_obs, n_vars)` via
/// `.shape`. The shape must match the output X dims; otherwise we
/// raise a `ValueError` matching the in-memory path's contract.
/// The `UserWarning` every `from_anndata` door raises when a 0-row write has
/// layers to drop. A layer exists on disk only as its CSR shards, and a 0-row
/// file has none (the format forbids framed zero-row shards: "emit no shard
/// at all instead"), so the layer's very name cannot be recorded. Say so
/// rather than drop it silently; X's shape, obs / var, obsm / varm and uns all
/// survive. See docs/api.md § `pyscx.from_anndata`.
pub(crate) fn warn_layers_dropped_at_zero_rows(
    py: Python<'_>,
    layer_keys: &[String],
) -> PyResult<()> {
    let msg = format!(
        "from_anndata: the AnnData has no rows; layers {layer_keys:?} exist on disk only as CSR shards and a 0-row file has none, so they were not written. X's shape, obs / var, obsm / varm and uns are kept."
    );
    crate::pyimport::import_module(py, "warnings")?.call_method1("warn", (msg,))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_write_layers(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    writer: &mut ScxWriter,
    out_n_obs: u64,
    out_n_vars: u64,
    out_shard_rows: u32,
    codec_for_encode: Option<CodecId>,
    index_dtype: u8,
    framing: Option<scx_format_io::FramingConfig>,
) -> PyResult<()> {
    let layers = match adata.getattr("layers") {
        Ok(l) => l,
        Err(_) => return Ok(()),
    };
    let keys: Vec<String> = crate::pyimport::import_module(py, "builtins")?
        .call_method1("list", (layers.call_method0("keys")?,))?
        .extract()?;
    if keys.is_empty() {
        return Ok(());
    }
    if out_n_obs == 0 {
        // Same policy as the in-memory path: a layer exists on disk only as its
        // CSR shards, a 0-row file has none, so the layer is dropped and the
        // caller is told. `chunk_boundaries` would otherwise return no bounds
        // and the loop below would drop every layer silently.
        return warn_layers_dropped_at_zero_rows(py, &keys);
    }

    let n_vars_u32 = u32::try_from(out_n_vars)
        .map_err(|_| PyRuntimeError::new_err(format!("n_vars {out_n_vars} exceeds u32::MAX")))?;
    let bounds = chunk_boundaries(out_n_obs as usize, out_shard_rows as usize);

    for layer_name in &keys {
        let layer = layers.call_method1("__getitem__", (layer_name,))?;
        let l_shape: (u64, u64) = layer.getattr("shape")?.extract()?;
        if l_shape != (out_n_obs, out_n_vars) {
            return Err(PyValueError::new_err(format!(
                "Layer '{layer_name}' has shape ({}, {}), expected ({}, {})",
                l_shape.0, l_shape.1, out_n_obs, out_n_vars
            )));
        }

        for (i, (start, end)) in bounds.iter().enumerate() {
            let py_slice = pyo3::types::PySlice::new(py, *start as isize, *end as isize, 1);
            let shard_obj = layer.call_method1("__getitem__", (py_slice,))?;
            let pre = decompose_scipy_csr_with(py, &shard_obj, |indptr, indices, data| {
                let mut enc_opts = scx_format_io::EncodeShardOptions::new(
                    format!("{layer_name}_shard_{i}"),
                    SectionType::LayerCsrShard,
                    n_vars_u32 as u64,
                    *start as u64,
                    index_dtype,
                );
                enc_opts.explicit_codec = codec_for_encode;
                enc_opts.framing = framing;
                py.detach(|| scx_format_io::encode_one_shard(indptr, indices, data, &enc_opts))
                    .map_err(to_pyerr)
            })?;
            py.detach(|| writer.write_preencoded_shard(pre))
                .map_err(to_pyerr)?;
        }
    }
    Ok(())
}

/// Mutation detection for the backed-routing path. Compares the
/// top-level key set of `getattr(adata, attr)` against the on-disk
/// h5py group at `/<attr>`. Returns `true` when the two key sets match
/// exactly (sender hasn't mutated this section in Python, so the
/// pipeline can stream from the source h5ad instead of materialising
/// the Python copy).
///
/// Cheapness invariants:
/// 1. `adata.obsm.keys()` on a backed AnnData lists the underlying
///    h5py group members without loading any dataset.
/// 2. `h5_file[attr].keys()` is a pure metadata read on h5py.
///
/// We never touch `adata.obsm[key]` here — that would force the very
/// h5py-to-numpy read we're trying to avoid.
///
/// Limitation: same-key replacements ("user did `adata.obsm['X_pca'] =
/// new_array`" without renaming) aren't detected. Document this with a
/// `UserWarning` on the routing path so users have a breadcrumb.
///
/// Gated behind the `hdf5` feature because the sole caller
/// (`route_backed_anndata_to_streaming`) is. Without `hdf5` the
/// streaming backed-AnnData path falls through to the in-memory branch
/// in `from_anndata_impl` and this helper would be dead code.
#[cfg(feature = "hdf5")]
pub(crate) fn section_keys_match(
    py: Python<'_>,
    h5_file: &Bound<'_, PyAny>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<bool> {
    let py_keys: Vec<String> = match adata.getattr(attr) {
        Ok(section) => match section.call_method0("keys") {
            Ok(keys_obj) => crate::pyimport::import_module(py, "builtins")?
                .call_method1("list", (keys_obj,))?
                .extract()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let disk_keys: Vec<String> = match h5_file.get_item(attr) {
        Ok(group) => match group.call_method0("keys") {
            Ok(keys_obj) => crate::pyimport::import_module(py, "builtins")?
                .call_method1("list", (keys_obj,))?
                .extract()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        },
        Err(_) => Vec::new(),
    };
    let py_set: std::collections::BTreeSet<&String> = py_keys.iter().collect();
    let disk_set: std::collections::BTreeSet<&String> = disk_keys.iter().collect();
    Ok(py_set == disk_set)
}
/// Helper for the backed-routing path. Reads a dense mapping
/// (`obsm` / `varm`) from a Python AnnData and returns
/// `Vec<(name, RecordBatch)>`. Missing groups → empty Vec.
pub(crate) fn extract_dense_mapping(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group = match adata.getattr(attr) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };
    let keys: Vec<String> = crate::pyimport::import_module(py, "builtins")?
        .call_method1("list", (group.call_method0("keys")?,))?
        .extract()?;
    keys.iter()
        .map(|key| {
            let arr = group.call_method1("__getitem__", (key,))?;
            let pd = crate::pyimport::import_module(py, "pandas")?;
            let df = pd.call_method1("DataFrame", (&arr,))?;
            let batch = pandas_to_record_batch(py, &df)?;
            Ok((key.clone(), batch))
        })
        .collect()
}

/// Helper for the backed-routing path. Reads a sparse pairwise
/// mapping (`obsp` / `varp`) as COO RecordBatches. Missing groups →
/// empty Vec.
pub(crate) fn extract_coo_mapping(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Vec<(String, RecordBatch)>> {
    let group = match adata.getattr(attr) {
        Ok(g) => g,
        Err(_) => return Ok(Vec::new()),
    };
    let keys: Vec<String> = crate::pyimport::import_module(py, "builtins")?
        .call_method1("list", (group.call_method0("keys")?,))?
        .extract()?;
    keys.iter()
        .map(|key| {
            let mat = group.call_method1("__getitem__", (key,))?;
            let batch = sparse_to_coo_record_batch(py, &mat)?;
            Ok((key.clone(), batch))
        })
        .collect()
}

/// Helper for the backed-routing path. Extracts `uns` from a Python
/// AnnData into an optional `serde_json::Value`. Returns `None` if
/// `uns` is empty (no `__scx_uns__` section written), matching the
/// non-backed path.
pub(crate) fn extract_uns_value(
    py: Python<'_>,
    adata: &Bound<'_, PyAny>,
    uns_format_parsed: UnsFormat,
) -> PyResult<Option<serde_json::Value>> {
    let uns = adata.getattr("uns")?;
    let uns_len: usize = uns.call_method0("__len__")?.extract()?;
    if uns_len == 0 {
        return Ok(None);
    }
    let np = crate::pyimport::import_module(py, "numpy")?;
    let np_generic = np.getattr("generic")?;
    let np_ndarray = np.getattr("ndarray")?;
    let mut ctx = UnsWriteCtx::new(uns_format_parsed, &np_generic, &np_ndarray);
    Ok(Some(normalize_uns_value(&uns, "uns", &mut ctx)?))
}

/// Write an [`ScxOverrides`]' four mapping families as row-shards.
///
/// Three call sites want exactly this — the backed and lazy `from_anndata`
/// rewrites and PFlog's materialise — differing only in the shard target, so it
/// is one function rather than three copies of the same four loops.
pub(crate) fn write_mapping_overrides(
    writer: &mut ScxWriter,
    ov: &crate::convert::h5ad::ScxOverrides,
    shard_target_rows: u32,
) -> std::result::Result<(), scx_format_io::ScxError> {
    for (k, b) in &ov.obsm {
        scx_format_io::for_each_dense_mapping_shard(b, shard_target_rows, |m, shard| {
            writer.write_obsm_shard(
                k,
                m.shard_idx,
                m.row_start,
                m.n_shard_rows,
                m.n_rows_total,
                shard,
            )
        })?;
    }
    for (k, b) in &ov.varm {
        scx_format_io::for_each_dense_mapping_shard(b, shard_target_rows, |m, shard| {
            writer.write_varm_shard(
                k,
                m.shard_idx,
                m.row_start,
                m.n_shard_rows,
                m.n_rows_total,
                shard,
            )
        })?;
    }
    for (k, b) in &ov.obsp {
        scx_format_io::for_each_coo_mapping_shard(
            &format!("obsp/{k}"),
            b,
            shard_target_rows,
            |m, shard| {
                writer.write_obsp_shard_coo(
                    k,
                    m.shard_idx,
                    m.row_start,
                    m.n_shard_rows,
                    m.n_rows_total,
                    shard,
                )
            },
        )?;
    }
    for (k, b) in &ov.varp {
        scx_format_io::for_each_coo_mapping_shard(
            &format!("varp/{k}"),
            b,
            shard_target_rows,
            |m, shard| {
                writer.write_varp_shard_coo(
                    k,
                    m.shard_idx,
                    m.row_start,
                    m.n_shard_rows,
                    m.n_rows_total,
                    shard,
                )
            },
        )?;
    }
    Ok(())
}

/// The same-pass sidecar's build options on the `from_anndata` SCX -> SCX
/// paths: `build_csc`'s defaults (4 GiB, the output's directory for spill,
/// framed iff the output is v4) at the caller's shard width — what the second
/// pass these paths used to run over the finished file built.
fn csc_build_options(cols_per_shard: usize) -> scx_format_io::CscBuildOptions {
    scx_format_io::CscBuildOptions {
        cols_per_shard,
        ..Default::default()
    }
}
