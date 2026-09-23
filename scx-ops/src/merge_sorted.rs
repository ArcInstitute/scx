//! Sorted k-way merge: `scx merge --sort-by`.
//!
//! The plain merge concatenates inputs (all of input 0's rows, then input
//! 1's, …). With a sort key it instead emits the obs axis **globally
//! ordered** by that key. When every input is already a *sorted run*
//! (e.g. produced by `scx convert --sort-by`), a k-way merge consumes each
//! input strictly in its natural row order, so per-input **forward cursors**
//! plus a min-heap on the head keys suffice — no random access, no spill.
//!
//! `compute_merge_order` produces a `Vec<u32>` (which input contributes each
//! successive output row) by k-way merging the inputs' obs sort-key streams;
//! each section (obs, X, layers) is then emitted by replaying that order with
//! per-input cursors. obsm is rejected under sort for now (see `has_obsm`);
//! var-axis sections and obsp are handled by the caller exactly as the plain
//! merge does.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::row::{OwnedRow, Rows};
use scx_codec::ValueEncoding;
use scx_format_io::catalog::FullCatalogEntry;
use scx_format_io::codec_select::select_codec;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::ScxReader;

use crate::error::{OpsError, Result};
use crate::helpers::{encode_values, widest_value_encoding};

/// True if any input carries an obsm (obs-axis dense mapping) section.
/// Sorted merge does not yet reorder obsm — the caller errors when this is
/// true under `--sort-by`.
pub(crate) fn has_obsm(readers: &[ScxReader]) -> bool {
    readers.iter().any(|r| {
        r.catalog().entries.iter().any(|e| {
            matches!(
                e.section_type,
                SectionType::ObsmEmbeddingShard | SectionType::ObsmEmbedding
            )
        })
    })
}

// Local alias so the type name reads clearly below.
use crate::sort as scx_ops_sort;

/// Canonical sort-key dtype: any string-like type (incl. dictionary-of-string)
/// collapses to `Utf8` so keys compare identically across inputs that store
/// obs as `Utf8` vs `LargeUtf8` vs dictionary-encoded; numerics pass through.
fn canonical_key_dtype(dt: &DataType) -> DataType {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8,
        DataType::Dictionary(_, v) => canonical_key_dtype(v),
        other => other.clone(),
    }
}

/// Project the `by` columns from `batch` into a key-only batch with canonical
/// dtypes, so a single `RowConverter` works across heterogeneous inputs.
fn canonicalize_keys(batch: &RecordBatch, by: &[String]) -> Result<RecordBatch> {
    let mut fields = Vec::with_capacity(by.len());
    let mut cols = Vec::with_capacity(by.len());
    for name in by {
        let col = batch.column_by_name(name).ok_or_else(|| {
            OpsError::InvalidInput(format!("sort key column '{name}' not found in obs"))
        })?;
        let dt = canonical_key_dtype(col.data_type());
        cols.push(arrow::compute::cast(col, &dt)?);
        fields.push(Field::new(name, dt, true));
    }
    Ok(RecordBatch::try_new(Arc::new(Schema::new(fields)), cols)?)
}

/// Per-input forward cursor over obs sort-key rows, asserting the input is a
/// sorted run.
struct KeyCursor<'a> {
    chunks: Box<dyn Iterator<Item = Result<RecordBatch>> + 'a>,
    extractor: &'a scx_ops_sort::SortKeyExtractor,
    by: &'a [String],
    cur: Option<Rows>,
    local: usize,
    prev: Option<OwnedRow>,
    input_id: u32,
}

impl<'a> KeyCursor<'a> {
    fn next_key(&mut self) -> Result<Option<OwnedRow>> {
        loop {
            if self.cur.is_none() {
                match self.chunks.next() {
                    None => return Ok(None),
                    Some(chunk) => {
                        // No dictionary handling needed: `canonicalize_keys`
                        // casts every key column through `canonical_key_dtype`,
                        // which already recurses through `Dictionary(_, v)`, so
                        // the row encoding is the same whichever way the column
                        // was stored.
                        let keys = canonicalize_keys(&chunk?, self.by)?;
                        self.cur = Some(self.extractor.rows(&keys)?);
                        self.local = 0;
                    }
                }
            }
            let rows = self.cur.as_ref().unwrap();
            if self.local >= rows.num_rows() {
                self.cur = None;
                continue;
            }
            let owned = rows.row(self.local).owned();
            self.local += 1;
            // Sorted-run check. The extractor bakes `--reverse` into the row
            // encoding, so a correctly (reverse-)sorted input always has
            // non-decreasing encoded rows.
            if let Some(prev) = &self.prev {
                if owned < *prev {
                    return Err(OpsError::InvalidInput(format!(
                        "merge --sort-by: input {} is not sorted by the key; pre-sort each \
                         input with `scx convert --sort-by` (or `scx sort`) before a sorted merge",
                        self.input_id
                    )));
                }
            }
            self.prev = Some(owned.clone());
            return Ok(Some(owned));
        }
    }
}

#[derive(PartialEq, Eq)]
struct HeapItem {
    key: OwnedRow,
    input: u32,
}
impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key).then(self.input.cmp(&other.input))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Compute the global merge order: `order[i]` is the input id that contributes
/// output row `i`. K-way merge of the inputs' (sorted) obs key streams; ties
/// broken by input id (= concatenation order for equal keys) for determinism.
pub(crate) fn compute_merge_order(
    readers: &[ScxReader],
    unified_obs_schema: &arrow::datatypes::Schema,
    by: &[String],
    reverse: bool,
    shard_target_rows: u64,
    total_n_obs: u64,
) -> Result<Vec<u32>> {
    // Build the comparator over a canonical key schema (string-like → Utf8)
    // so a single RowConverter works for inputs that differ in Utf8 vs
    // LargeUtf8 vs dictionary encoding.
    let key_fields: Vec<Field> = by
        .iter()
        .map(|name| {
            let f = unified_obs_schema.field_with_name(name).map_err(|_| {
                OpsError::InvalidInput(format!("sort key column '{name}' not found in obs"))
            })?;
            Ok(Field::new(name, canonical_key_dtype(f.data_type()), true))
        })
        .collect::<Result<_>>()?;
    let key_schema = Schema::new(key_fields);
    let extractor = scx_ops_sort::SortKeyExtractor::new(&key_schema, by, reverse)?;
    let mut cursors: Vec<KeyCursor> = readers
        .iter()
        .enumerate()
        .map(|(i, r)| {
            Ok(KeyCursor {
                chunks: crate::merge::input_obs_chunks(r, shard_target_rows)?,
                extractor: &extractor,
                by,
                cur: None,
                local: 0,
                prev: None,
                input_id: i as u32,
            })
        })
        .collect::<Result<_>>()?;

    let mut heap: BinaryHeap<Reverse<HeapItem>> = BinaryHeap::with_capacity(cursors.len());
    for (i, c) in cursors.iter_mut().enumerate() {
        if let Some(key) = c.next_key()? {
            heap.push(Reverse(HeapItem {
                key,
                input: i as u32,
            }));
        }
    }

    let mut order: Vec<u32> = Vec::with_capacity(total_n_obs as usize);
    while let Some(Reverse(item)) = heap.pop() {
        let input = item.input;
        order.push(input);
        if let Some(key) = cursors[input as usize].next_key()? {
            heap.push(Reverse(HeapItem { key, input }));
        }
    }
    Ok(order)
}

/// Per-input forward cursor pushing exactly `n` obs (or dense) rows at a time,
/// across shard boundaries, into the caller's pending buffer.
///
/// It appends slices rather than returning one concatenated batch, so the only
/// concat in the emitter is the per-output-shard one in `emit`. Concatenating
/// here as well would mean doing it twice, and — now that categoricals stay
/// dictionary-encoded — each concat is a place where two batches can disagree
/// on key width or hold distinct values arrays, which arrow resolves by either
/// duplicating the vocabulary or pruning it. One concat, one pipeline.
///
/// An arrow slice of a `DictionaryArray` keeps the whole values array, so a
/// slice carries its chunk's full declared vocabulary even when no row in the
/// slice uses part of it. That is deliberate: it is what lets a
/// declared-but-unused level survive a shard split.
struct ObsCursor<'a> {
    chunks: Box<dyn Iterator<Item = Result<RecordBatch>> + 'a>,
    cur: Option<Arc<RecordBatch>>,
    local: usize,
}

/// A run of rows from one source chunk, named rather than materialised.
///
/// A fully interleaved sorted merge emits **one row per run**, so an output
/// shard's buffer holds `shard_target_rows` of these — 16 384 by default. The
/// concat pipeline they eventually go through does per-batch work (widening,
/// reconciling, re-keying onto a shared vocabulary), and paying that
/// 16 384 times instead of once per *source chunk* measured at ~470 ms per
/// output shard. Holding the parent and an offset lets `emit` prepare the two
/// or three distinct chunks and slice the prepared result, which is the same
/// bytes for ~1 % of the cost.
struct ObsSlice {
    chunk: Arc<RecordBatch>,
    offset: usize,
    len: usize,
}

impl<'a> ObsCursor<'a> {
    fn extend_rows(&mut self, n: usize, out: &mut Vec<ObsSlice>) -> Result<()> {
        let mut need = n;
        while need > 0 {
            if self.cur.is_none() {
                let chunk =
                    self.chunks.next().transpose()?.ok_or_else(|| {
                        OpsError::InvalidInput("merge order exceeds obs rows".into())
                    })?;
                // Widen here, **once per chunk**, not once per emit. These two
                // steps are per-row (`upcast_to_large_types` rewrites string
                // buffers; `widen_dictionary_keys` re-keys) and they depend
                // only on the chunk, so paying them inside `materialize_obs_shard`
                // would re-widen every row of every still-open parent on every
                // output shard — amplified by the input count on a high-fan-in
                // merge, where each emit takes a few rows from each of many
                // 16K-row chunks. The cross-batch half (reconcile + share)
                // genuinely depends on which chunks meet in a given shard and
                // stays there.
                let chunk = scx_format_io::widen_metadata_batch_for_concat(&chunk)?;
                self.cur = Some(Arc::new(chunk));
                self.local = 0;
            }
            let cur = self.cur.as_ref().unwrap();
            let avail = cur.num_rows() - self.local;
            let take = need.min(avail);
            out.push(ObsSlice {
                chunk: Arc::clone(cur),
                offset: self.local,
                len: take,
            });
            self.local += take;
            need -= take;
            if self.local >= cur.num_rows() {
                self.cur = None;
            }
        }
        Ok(())
    }
}

/// Materialise one output shard's worth of [`ObsSlice`]s as a single batch.
///
/// The slices can name chunks from **different inputs**, which agree only up
/// to `validate_obs_identity`'s logical comparison: they may disagree on string
/// width, on a categorical's key width, on Dictionary-vs-plain, and they hold
/// distinct dictionary values arrays. A bare `concat_batches` fails on the
/// first three and, on the fourth, either appends both vocabularies verbatim
/// (duplicate categories, which pandas rejects) or merges and prunes them
/// row-count-dependently. `scx_format_io`'s prepare/concat pair is the read
/// side's own pipeline for exactly this.
///
/// It runs over the **distinct source chunks**, not over the slices: after
/// preparation every chunk shares one values array per dictionary column, so
/// slicing preserves that and `concat` takes its cheap keys-only path. The
/// per-row half of the preparation (widening) already happened once, when
/// `ObsCursor` loaded the chunk.
fn materialize_obs_shard(pending: &[ObsSlice]) -> Result<RecordBatch> {
    // No single-slice shortcut. It is tempting — one slice of one chunk needs
    // no reconciling — but `ObsCursor` widened that chunk's dictionary keys to
    // Int32 at load, and nothing downstream narrows them again
    // (`write_obs_shard` does not). Returning the slice verbatim would write
    // Int32 keys for every categorical on the shape this k-way merge is *for*:
    // already-sorted, non-interleaved inputs, where each output shard is one
    // run from one input. That is 4x the code buffer against an Int8 input.
    // `concat_prepared_metadata_batches` costs a concat of one batch and gets
    // the minimal key back from `unify_dictionary_columns`.
    // Identity, not equality: two slices share a parent iff they name the same
    // `Arc`. Comparing the batches themselves would be both wrong (a clone is
    // a different pointer) and expensive.
    //
    // Pointer-keyed rather than a linear scan: `pending` holds one entry per
    // sorted run, so a fully interleaved merge fills it with
    // `shard_target_rows` entries (16 384 by default), and a high-fan-in merge
    // pushes the distinct-chunk count up with the input count — the product is
    // what a `Vec::position` walk would pay per emit.
    let mut seen: HashMap<*const RecordBatch, usize> = HashMap::with_capacity(pending.len());
    let mut distinct: Vec<Arc<RecordBatch>> = Vec::new();
    let mut owner: Vec<usize> = Vec::with_capacity(pending.len());
    for s in pending {
        let key = Arc::as_ptr(&s.chunk);
        match seen.get(&key) {
            Some(&i) => owner.push(i),
            None => {
                distinct.push(Arc::clone(&s.chunk));
                seen.insert(key, distinct.len() - 1);
                owner.push(distinct.len() - 1);
            }
        }
    }
    // Already widened at chunk load; only the cross-batch half is left.
    let prepared = scx_format_io::reconcile_and_share_metadata_batches(
        distinct.iter().map(|b| (**b).clone()).collect(),
    )?;
    let slices: Vec<RecordBatch> = pending
        .iter()
        .zip(&owner)
        .map(|(s, &i)| prepared[i].slice(s.offset, s.len))
        .collect();
    Ok(scx_format_io::concat_prepared_metadata_batches(&slices)?)
}

/// Per-input forward cursor over CSR shards (X or a layer), appending rows
/// into output buffers.
struct CsrCursor<'a> {
    reader: &'a ScxReader,
    entries: Vec<FullCatalogEntry>,
    shard_pos: usize,
    cur: Option<(Vec<i64>, Vec<i32>, Vec<f32>)>,
    local: usize,
}
impl<'a> CsrCursor<'a> {
    fn ensure(&mut self) -> Result<bool> {
        loop {
            if let Some((indptr, _, _)) = &self.cur {
                if self.local < indptr.len() - 1 {
                    return Ok(true);
                }
                self.cur = None;
            }
            if self.shard_pos >= self.entries.len() {
                return Ok(false);
            }
            let entry = &self.entries[self.shard_pos];
            self.shard_pos += 1;
            self.cur = Some(self.reader.read_shard_from_entry(entry)?);
            self.local = 0;
        }
    }

    fn append_rows(
        &mut self,
        n: usize,
        out_indptr: &mut Vec<u64>,
        out_indices: &mut Vec<u32>,
        out_data: &mut Vec<f32>,
    ) -> Result<()> {
        let mut need = n;
        while need > 0 {
            if !self.ensure()? {
                return Err(OpsError::InvalidInput(
                    "merge order exceeds input X rows".into(),
                ));
            }
            let (indptr, indices, data) = self.cur.as_ref().unwrap();
            let rows_in_shard = indptr.len() - 1;
            let take = need.min(rows_in_shard - self.local);
            for r in self.local..self.local + take {
                let s = indptr[r] as usize;
                let e = indptr[r + 1] as usize;
                for j in s..e {
                    out_indices.push(indices[j] as u32);
                    out_data.push(data[j]);
                }
                let prev = *out_indptr.last().unwrap();
                out_indptr.push(prev + (e - s) as u64);
            }
            self.local += take;
            need -= take;
        }
        Ok(())
    }
}

/// CSR shard entries for a layer in one input, sorted by `row_start`.
fn layer_entries(reader: &ScxReader, layer_name: &str) -> Vec<FullCatalogEntry> {
    let prefix = format!("{layer_name}_shard_");
    let mut entries: Vec<FullCatalogEntry> = reader
        .catalog()
        .entries
        .iter()
        .filter(|e| {
            e.section_type == SectionType::LayerCsrShard
                && e.modality_id == 0
                && e.name.starts_with(&prefix)
        })
        .cloned()
        .collect();
    entries.sort_by_key(|e| e.stats.as_ref().map_or(u64::MAX, |s| s.row_start));
    entries
}

/// Walk `order` in maximal same-input runs, calling `bite(input, take)` in
/// `shard_target`-bounded pieces and `flush()` whenever a full shard's worth
/// of rows has accumulated. `flush` returns the rows it emitted (always the
/// accumulated count); the trailing partial shard is flushed by the caller.
fn emit_x_like(
    readers: &[ScxReader],
    order: &[u32],
    shard_target_rows: u64,
    mut entries_for: impl FnMut(&ScxReader) -> Vec<FullCatalogEntry>,
    value_encoding: ValueEncoding,
    mut write_shard: impl FnMut(&[u64], &[u32], &[u8], ValueEncoding, u64) -> Result<u64>,
) -> Result<()> {
    let mut cursors: Vec<CsrCursor> = readers
        .iter()
        .map(|r| CsrCursor {
            reader: r,
            entries: entries_for(r),
            shard_pos: 0,
            cur: None,
            local: 0,
        })
        .collect();

    let mut indptr: Vec<u64> = vec![0];
    let mut indices: Vec<u32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    let mut rows_in_shard: u64 = 0;
    let mut cumulative: u64 = 0;
    let target = shard_target_rows.max(1);

    let mut i = 0usize;
    while i < order.len() {
        let inp = order[i] as usize;
        let mut run = 1usize;
        while i + run < order.len() && order[i + run] as usize == inp {
            run += 1;
        }
        let mut done = 0usize;
        while done < run {
            let space = (target - rows_in_shard) as usize;
            let take = (run - done).min(space);
            cursors[inp].append_rows(take, &mut indptr, &mut indices, &mut data)?;
            rows_in_shard += take as u64;
            done += take;
            if rows_in_shard == target {
                let mut bytes = Vec::new();
                encode_values(&mut bytes, &data, value_encoding)?;
                cumulative += write_shard(&indptr, &indices, &bytes, value_encoding, cumulative)?;
                indptr = vec![0];
                indices.clear();
                data.clear();
                rows_in_shard = 0;
            }
        }
        i += run;
    }
    if rows_in_shard > 0 {
        let mut bytes = Vec::new();
        encode_values(&mut bytes, &data, value_encoding)?;
        write_shard(&indptr, &indices, &bytes, value_encoding, cumulative)?;
    }
    Ok(())
}

/// Emit obs, X, and layers in the merged `order`. Returns the per-output-CSR-
/// shard `(row_start, row_end)` ranges for the predicate-index builder.
/// `obs_index_builder` is fed each emitted obs shard in row order.
pub(crate) fn emit_sorted(
    readers: &[ScxReader],
    writer: &mut ScxWriter,
    order: &[u32],
    shard_target_rows: u64,
    total_n_obs: u64,
    obs_index_builder: &mut Option<scx_engine::ObsPredicateIndexBuilder>,
) -> Result<Vec<(u64, u64)>> {
    let target = shard_target_rows.max(1);

    // ---- obs ----
    {
        let mut cursors: Vec<ObsCursor> = readers
            .iter()
            .map(|r| {
                Ok(ObsCursor {
                    chunks: crate::merge::input_obs_chunks(r, shard_target_rows)?,
                    cur: None,
                    local: 0,
                })
            })
            .collect::<Result<_>>()?;

        let mut pending: Vec<ObsSlice> = Vec::new();
        let mut pending_rows: u64 = 0;
        let mut out_shard_idx: u32 = 0;
        let mut cumulative: u64 = 0;

        let emit = |writer: &mut ScxWriter,
                    builder: &mut Option<scx_engine::ObsPredicateIndexBuilder>,
                    pending: &mut Vec<ObsSlice>,
                    pending_rows: &mut u64,
                    out_shard_idx: &mut u32,
                    cumulative: &mut u64|
         -> Result<()> {
            // Output shards may disagree on key width — a shard drawn from one
            // input keeps that input's, a shard spanning inputs carries the
            // union's. The assembler reconciles that on read.
            let batch = materialize_obs_shard(pending)?;
            let n = batch.num_rows() as u64;
            if let Some(b) = builder.as_mut() {
                b.push_shard(&batch, *cumulative)?;
            }
            writer.write_obs_shard(*out_shard_idx, *cumulative, n, total_n_obs, &batch)?;
            *out_shard_idx += 1;
            *cumulative += n;
            pending.clear();
            *pending_rows = 0;
            Ok(())
        };

        let mut i = 0usize;
        while i < order.len() {
            let inp = order[i] as usize;
            let mut run = 1usize;
            while i + run < order.len() && order[i + run] as usize == inp {
                run += 1;
            }
            let mut done = 0usize;
            while done < run {
                let space = (target - pending_rows) as usize;
                let take = (run - done).min(space);
                cursors[inp].extend_rows(take, &mut pending)?;
                pending_rows += take as u64;
                done += take;
                if pending_rows == target {
                    emit(
                        writer,
                        obs_index_builder,
                        &mut pending,
                        &mut pending_rows,
                        &mut out_shard_idx,
                        &mut cumulative,
                    )?;
                }
            }
            i += run;
        }
        if pending_rows > 0 {
            emit(
                writer,
                obs_index_builder,
                &mut pending,
                &mut pending_rows,
                &mut out_shard_idx,
                &mut cumulative,
            )?;
        }
        if out_shard_idx == 0 {
            // Every input was empty: `order` had nothing to drain, so no obs
            // shard was written — and a file without an obs section is
            // unreadable. Write input 0's own 0-row obs verbatim, exactly as
            // the chunk path now writes its batches (mirrors the plain merge
            // emitter), so an empty output has the schema a populated merge of
            // these inputs would have.
            debug_assert_eq!(total_n_obs, 0);
            let empty_obs = readers[0].read_obs()?;
            writer.write_obs(&empty_obs)?;
        }
    }

    // ---- X ----
    let x_value_encoding = {
        let mut encs = Vec::new();
        for r in readers {
            for e in r.catalog().shards_sorted() {
                let sh = r.read_shard_header(e)?;
                encs.push(
                    ValueEncoding::from_u8(sh.value_encoding)
                        .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?,
                );
            }
        }
        widest_value_encoding(&encs)
    };
    let mut ranges: Vec<(u64, u64)> = Vec::new();
    emit_x_like(
        readers,
        order,
        shard_target_rows,
        |r| r.catalog().shards_sorted().into_iter().cloned().collect(),
        x_value_encoding,
        |indptr, indices, bytes, enc, row_start| {
            let n = indptr.len() as u64 - 1;
            let codec = select_codec(bytes, enc);
            writer.write_csr_shard(indptr, indices, bytes, codec, enc, row_start)?;
            ranges.push((row_start, row_start + n));
            Ok(n)
        },
    )?;
    // The same-pass sidecar (if the caller enabled one) is complete once X
    // is; emit it before the layers, which go through the same `emit_x_like`.
    crate::csc_carry::emit(writer, "merge")?;

    // ---- layers ----
    let layer_names = {
        let mut names: Vec<String> = readers.iter().flat_map(|r| r.layer_names()).collect();
        names.sort();
        names.dedup();
        names
    };
    for layer_name in &layer_names {
        // Every input must carry the layer (matches plain-merge semantics) —
        // except a 0-row input, which has no layer shards for any layer and
        // contributes nothing.
        for (idx, r) in readers.iter().enumerate() {
            if layer_entries(r, layer_name).is_empty() {
                if r.n_obs() == 0 {
                    continue;
                }
                return Err(OpsError::LayerMissing {
                    name: layer_name.clone(),
                    file_index: idx,
                });
            }
        }
        let enc = {
            let mut encs = Vec::new();
            for r in readers {
                for e in layer_entries(r, layer_name) {
                    let sh = r.read_shard_header(&e)?;
                    encs.push(
                        ValueEncoding::from_u8(sh.value_encoding)
                            .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))?,
                    );
                }
            }
            widest_value_encoding(&encs)
        };
        let mut shard_idx: u32 = 0;
        emit_x_like(
            readers,
            order,
            shard_target_rows,
            |r| layer_entries(r, layer_name),
            enc,
            |indptr, indices, bytes, e, row_start| {
                let n = indptr.len() as u64 - 1;
                let codec = select_codec(bytes, e);
                let shard = scx_format_io::ShardBuffers::new(indptr, indices, bytes, codec, e);
                writer.write_layer_csr_shard(layer_name, shard_idx, row_start, shard)?;
                shard_idx += 1;
                Ok(n)
            },
        )?;
    }

    Ok(ranges)
}
