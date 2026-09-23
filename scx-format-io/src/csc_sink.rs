// Same-pass CSC sidecar: a `CscBuilder` fed by the writer as it writes X.
//
// `scx build-csc` (and every `--rebuild-csc` before it) builds a sidecar in a
// second pass: finish the file, reopen it, decode every CSR shard again and
// push it into a `CscBuilder`. A rewrite op or a streaming ingest already holds
// each X shard at the moment it writes it, in row order, so the builder can be
// fed there instead and the second read disappears.
//
// The feed lives in the writer rather than in each op because X reaches disk
// through five methods — `write_csr_shard`, `write_preencoded_shard`,
// `write_csr_shard_raw_copy`, `copy_section_verbatim` and `write_raw_shard` —
// and the ops between them use all five. A hook in one op, or only in
// `write_shard_inner`, silently misses the shards the others write. The one
// predicate every path shares is `section_type == CsrShard`; layers
// (`LayerCsrShard`) and `adata.raw` (`RawCsrShard`) are excluded by type.
//
// Output is byte-identical to `scx_ops::rebuild_csc_inplace` over the same CSR
// shards for the same `(cols_per_shard, memory_bytes)`: the values are the
// same f32s a decode would produce (both go through `values_raw_to_f32`), the
// encoding comes from the same `pick_csc_encoding` over the same inputs (each
// shard's declared encoding and codec as stamped in its header, and the
// catalog's `value_max`), and the framing rule is `rebuild_csc_inplace`'s.
// `memory_bytes` sizes the emitted shard widths, so it must match too;
// `spill_after_bytes` only decides what spills and never changes a byte.

use std::io::Cursor;
use std::path::PathBuf;

use scx_codec::{CodecId, ValueEncoding};
use scx_sparse::{CscBuilder, CscBuilderConfig, ScxCsr};

use crate::encoder::FramingConfig;
use crate::error::{Result, ScxError};
use crate::shard::ShardHeader;
use crate::validated_section::ValidatedSection;

/// How [`crate::ScxWriter::enable_csc_sidecar`] builds its sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CscBuildOptions {
    /// Upper bound on each emitted shard's column count.
    pub cols_per_shard: usize,
    /// The sidecar budget. Sizes the emitted shard widths (with
    /// `cols_per_shard`) exactly as `scx build-csc --memory-limit` does, so a
    /// same-pass sidecar and a `build-csc` one agree byte for byte only when
    /// this matches.
    pub memory_bytes: usize,
    /// How many bytes of column buckets stay resident before spilling. `None`
    /// is `build-csc`'s share, [`crate::csc_budget::CSC_BUILD_BUCKET_SHARE`] of
    /// `memory_bytes`. A caller running the builder beside other work that
    /// shares the budget (streaming ingest) passes its own, smaller share.
    pub spill_after_bytes: Option<usize>,
    /// Where the builder spills. `None` is the output's own directory.
    pub spill_root: Option<PathBuf>,
    /// Row-group framing for the sidecar on a v4 output. `None` is the default
    /// config. Whatever is passed, `trial` and `decode_target` are cleared (a
    /// sidecar's codec follows its source; it is not re-selected). Passing
    /// `Some` for a ≤ v3 output is an error, as it is for `build-csc`.
    pub framing: Option<FramingConfig>,
    /// The caller named a memory budget, so the emit batches whole shards
    /// against [`crate::csc_budget::csc_emit_batch_nnz`] rather than handing
    /// out whole emit groups. See [`crate::csc_sidecar::CscEmitOptions::whole_groups`]
    /// for what that trades. A `spill_after_bytes` counts as a budget too:
    /// only a budgeted caller passes one.
    pub bounded_emit: bool,
}

impl Default for CscBuildOptions {
    fn default() -> Self {
        let sidecar = crate::csc_sidecar::CscSidecarOptions::default();
        Self {
            cols_per_shard: sidecar.cols_per_shard,
            memory_bytes: sidecar.memory_budget_bytes,
            spill_after_bytes: None,
            spill_root: None,
            framing: None,
            bounded_emit: false,
        }
    }
}

/// The builder and the encoding inputs gathered while X is written.
pub(crate) struct CscSink {
    builder: CscBuilder,
    rows_pushed: u64,
    n_cols: usize,
    declared: Vec<ValueEncoding>,
    max_int_val: u32,
    first_codec: Option<CodecId>,
    framing: Option<FramingConfig>,
    /// The builder's `memory_bytes`, which also sizes the emit's batches.
    memory_bytes: usize,
    /// See [`CscBuildOptions::bounded_emit`].
    bounded_emit: bool,
}

impl CscSink {
    pub(crate) fn new(
        n_rows: usize,
        n_cols: usize,
        opts: CscBuildOptions,
        default_spill_root: Option<PathBuf>,
    ) -> Result<Self> {
        let spill_root = opts.spill_root.or(default_spill_root);
        let store = crate::csc_spill::TempDirSpillStore::new(spill_root.as_deref())?;
        let spill_after_bytes = opts.spill_after_bytes.unwrap_or(
            crate::csc_budget::CSC_BUILD_BUCKET_SHARE.of(opts.memory_bytes as u64) as usize,
        );
        let builder = CscBuilder::new(
            n_rows,
            n_cols,
            CscBuilderConfig {
                cols_per_shard: opts.cols_per_shard,
                memory_bytes: opts.memory_bytes,
                spill_after_bytes,
                ..Default::default()
            },
            Box::new(store),
        )
        .map_err(csc_err)?;
        Ok(Self {
            builder,
            rows_pushed: 0,
            n_cols,
            declared: Vec::new(),
            max_int_val: 0,
            first_codec: None,
            framing: opts.framing,
            memory_bytes: opts.memory_bytes,
            bounded_emit: opts.bounded_emit || opts.spill_after_bytes.is_some(),
        })
    }

    /// Feed a shard the writer encoded itself, from the buffers it encoded.
    /// `header` is the one it wrote, so the codec is the one it chose rather
    /// than the caller's candidate.
    pub(crate) fn push_buffers(
        &mut self,
        header: &ShardHeader,
        value_max: Option<u32>,
        indptr: &[u64],
        indices: &[u32],
        values: &[u8],
    ) -> Result<()> {
        let enc = self.note_header(header, value_max)?;
        let (indptr, indices, data) = scx_codec::decoded_shard_to_scipy(
            (indptr.to_vec(), indices.to_vec(), values.to_vec()),
            enc,
            index_bound(self.n_cols)?,
        )?;
        self.push(header, indptr, indices, data)
    }

    /// Feed an already-encoded shard: decode its regions (framed or not).
    pub(crate) fn push_regions(
        &mut self,
        header: &ShardHeader,
        value_max: Option<u32>,
        indptr_bytes: &[u8],
        indices_bytes: &[u8],
        values_bytes: &[u8],
        block_index_bytes: &[u8],
    ) -> Result<()> {
        self.note_header(header, value_max)?;
        #[cfg(feature = "parallel")]
        let decode = crate::shard_decode::decode_shard_regions_scipy_parallel;
        #[cfg(not(feature = "parallel"))]
        let decode = crate::shard_decode::decode_shard_regions_scipy;
        let (indptr, indices, data) = decode(
            header,
            indptr_bytes,
            indices_bytes,
            values_bytes,
            block_index_bytes,
        )?;
        self.push(header, indptr, indices, data)
    }

    /// Feed a complete encoded section (header + payload).
    pub(crate) fn push_section(&mut self, section: &[u8], value_max: Option<u32>) -> Result<()> {
        let vs = ValidatedSection::new(section);
        let sh = ShardHeader::read_from(&mut Cursor::new(vs.header()?))?;
        self.push_regions(
            &sh,
            value_max,
            vs.subslice(sh.indptr_rel_offset, sh.indptr_length)?,
            vs.subslice(sh.indices_rel_offset, sh.indices_length)?,
            vs.subslice(sh.values_rel_offset, sh.values_length)?,
            vs.subslice(sh.block_index_rel_offset, sh.block_index_length)?,
        )
    }

    /// Record the `pick_csc_encoding` inputs `build-csc` reads from the same
    /// header and catalog entry, and return the shard's value encoding.
    fn note_header(
        &mut self,
        header: &ShardHeader,
        value_max: Option<u32>,
    ) -> Result<ValueEncoding> {
        let enc = ValueEncoding::from_u8(header.value_encoding)
            .ok_or(ScxError::UnknownValueEncoding(header.value_encoding))?;
        self.declared.push(enc);
        if self.first_codec.is_none() {
            self.first_codec = Some(
                CodecId::from_u8(header.codec_id).ok_or(ScxError::UnknownCodec(header.codec_id))?,
            );
        }
        if let Some(v) = value_max {
            self.max_int_val = self.max_int_val.max(v);
        }
        Ok(enc)
    }

    fn push(
        &mut self,
        header: &ShardHeader,
        indptr: Vec<i64>,
        indices: Vec<i32>,
        data: Vec<f32>,
    ) -> Result<()> {
        // The header's own first row, checked against the running count —
        // `build-csc`'s predicate. The builder checks the same thing against
        // the `row_start` it is handed, but that is `rows_pushed` by
        // construction; this is the check that can fail.
        if header.global_offset != self.rows_pushed {
            return Err(ScxError::InvalidCatalog(format!(
                "same-pass CSC: an X shard declares row_start {}, but {} rows precede it; \
                 X shards must be written in row order to carry a sidecar",
                header.global_offset, self.rows_pushed
            )));
        }
        let n_rows = indptr.len().saturating_sub(1);
        let shard = ScxCsr::new_unchecked((n_rows, self.n_cols), indptr, indices, data);
        self.builder
            .push_shard(self.rows_pushed, &shard)
            .map_err(csc_err)?;
        self.rows_pushed += n_rows as u64;
        Ok(())
    }

    /// Close the builder. Returns the emitter, the emit options and the
    /// framing to scope around the emit; `None` when no X shard was written
    /// (an empty matrix carries no sidecar, as on every other path).
    pub(crate) fn finish(
        self,
        output_v4: bool,
    ) -> Result<
        Option<(
            scx_sparse::CscEmitter,
            crate::csc_sidecar::CscEmitOptions,
            Option<FramingConfig>,
        )>,
    > {
        if !output_v4 && self.framing.is_some() {
            return Err(ScxError::InvalidCatalog(
                "same-pass CSC: framing a sidecar needs a v4 output; this one is ≤ v3".to_string(),
            ));
        }
        if self.declared.is_empty() || self.n_cols == 0 {
            return Ok(None);
        }
        let (value_encoding, codec_id) = crate::csc_sidecar::pick_csc_encoding(
            &self.declared,
            self.max_int_val,
            self.first_codec,
        )
        .ok_or_else(|| {
            ScxError::InvalidCatalog("same-pass CSC: no X shard to take a codec from".into())
        })?;
        // `rebuild_csc_inplace`'s rule: framed iff the file is v4, with the
        // codec-re-selecting knobs cleared.
        let framing = output_v4.then(|| FramingConfig {
            trial: false,
            decode_target: None,
            ..self.framing.unwrap_or_default()
        });
        let emitter = self.builder.finish().map_err(csc_err)?;
        Ok(Some((
            emitter,
            crate::csc_sidecar::CscEmitOptions {
                value_encoding,
                codec_id,
                modality_id: None,
                batch_nnz: crate::csc_budget::csc_emit_batch_nnz(self.memory_bytes as u64),
                whole_groups: !self.bounded_emit,
            },
            framing,
        )))
    }
}

fn index_bound(n_cols: usize) -> Result<u32> {
    u32::try_from(n_cols).map_err(|_| ScxError::NVarsOverflow(n_cols as u64))
}

fn csc_err(e: scx_sparse::CscBuilderError) -> ScxError {
    ScxError::CscTranspose(e.to_string())
}

#[cfg(test)]
#[path = "csc_sink_tests.rs"]
mod tests;
