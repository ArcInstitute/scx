//! `scx sort` standalone engine (SCX-SORT-SPEC §13 Phase 4).
//!
//! Globally reorders the obs (cell) axis of an already-built, codec-compressed
//! `.scx` by a sort key and writes a new file. The pure primitives (key
//! extraction, partition sizing, predicate-index feed, provenance) live in
//! [`crate::sort`]; this module is the I/O engine + strategy selector on top.
//!
//! ## The `new_pos` reduction (design note)
//!
//! A global row reorder is a random scatter over compressed CSR shards
//! (decoding one arbitrary row means decoding its whole shard), so a
//! permute-then-seek is O(n_shards²). The spec's external partition sort
//! solves this by spilling decoded rows tagged with `(full_key,
//! source_row_id)`. We use a simpler, equivalent reduction: the obs **key**
//! set fits in RAM at the same memory class the predicate-index build already
//! requires (O(n_obs)), so we sort the keys once up front (`stable_argsort`)
//! to get the global order, then tag each spilled X row with a single integer
//! **`new_pos`** (its destination row). Everything downstream keys off
//! `new_pos`:
//!
//! - correctness for *any* key kind (categorical / numeric / composite /
//!   reverse) — the key comparison happened once, in pass 0;
//! - stability is inherent (the order comes from `stable_argsort`, ties broken
//!   by source row-id);
//! - parallel scatter is trivially safe (append order is irrelevant — a row's
//!   destination is fixed by `new_pos`);
//! - partitions are `new_pos` ranges, so each holds ≤ `P` rows **regardless of
//!   key skew** — the spec's pass-2 sub-split is unnecessary (a dominant
//!   category cannot blow a partition past budget).
//!
//! The binding constraint (the multi-TB X matrix) is still bounded to one
//! partition at a time; the O(n_obs) order array is the same memory class as
//! the predicate index it must build either way.
//!
//! ## Strategies
//!
//! All three compute the *same* `order` in pass 0 and share obs / obsm / layer
//! / index emission, differing only in how they gather X rows (so a/b/c are
//! byte-identical):
//! - **(a) in-memory** — decode all X into RAM, gather by `order`.
//! - **(b) K-pass-by-category** — single categorical key, low K: stream X once
//!   per category, zero spill.
//! - **(c) external partition sort** — spill X rows tagged by `new_pos`, emit
//!   per `new_pos`-range partition.
//!
//! ## Scope (Phase 4)
//!
//! Local input only (cloud input deferred). obsm and layers are gathered
//! in-memory (the bounded-memory guarantee is for X). obsp is dropped with a
//! warning (remap is Phase 5); the detection bitmap is dropped (T0.1);
//! multimodal input is rejected (Phase 5). CSC sidecar is dropped; the CLI
//! re-emits it post-write via `rebuild_csc_inplace` on `--rebuild-csc`.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{Array, ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::DataType;
use scx_codec::{CodecId, CodecSelection, ValueEncoding};
use scx_format_io::codec_select::select_codec;
use scx_format_io::header::FileHeader;
use scx_format_io::section::SectionType;
use scx_format_io::writer::ScxWriter;
use scx_format_io::{FullCatalogEntry, ScxReader, ShardHeader, SHARD_HEADER_SIZE};

use crate::error::{OpsError, Result};
use crate::flock::SharedFileLock;
use crate::helpers::encode_value;
use crate::sort::{
    partition_target_rows, rebuild_obs_predicate_index_streaming, sort_provenance_entry,
    stable_argsort, SortKeyExtractor, SortOptions, SortStrategy, SortSummary,
};

/// K-pass is only considered for a single categorical key with at most this
/// many distinct values (above which the external spill path wins).
const K_PASS_MAX_CARDINALITY: usize = 32;

/// Sort an SCX file: reorder the obs axis by `opts.by`, writing a new file.
///
/// Selects the execution strategy automatically from the input shape and
/// `opts.memory_budget`. See the module docs for the design.
pub fn sort(input: &Path, output: &Path, opts: &SortOptions) -> Result<SortSummary> {
    sort_with_strategy(input, output, opts, None)
}

/// Sort with an explicit strategy override. `None` auto-selects. The override
/// exists for the strategy-differential gate (§10) — all three strategies must
/// produce byte-identical output on a shared fixture.
pub fn sort_with_strategy(
    input: &Path,
    output: &Path,
    opts: &SortOptions,
    force_strategy: Option<SortStrategy>,
) -> Result<SortSummary> {
    let _lock = SharedFileLock::acquire(input)?;
    let reader = ScxReader::open(input)?;
    let in_header = reader.header().clone();

    if reader.is_multimodal() {
        return Err(OpsError::InvalidInput(
            "scx sort does not yet support multimodal inputs (Phase 5); \
             sort each modality's source before merge, or use a single-modality file"
                .to_string(),
        ));
    }
    if in_header.has_raw() {
        log::warn!(
            "scx sort: input {} carries an adata.raw matrix, which is not preserved \
             through sort — raw will be dropped from the output",
            input.display()
        );
    }

    let n_obs = in_header.n_obs as usize;
    let n_vars = in_header.n_vars;

    // ----- Pass 0: compute the global order (shared by all strategies) -----
    let obs_full = reader.read_obs()?;
    let keep_mask = reader.deletion_keep_mask()?;

    // Live rows only (deletions are materialized away, as compact does).
    let live_ids: Vec<u64> = match &keep_mask {
        Some(mask) => (0..n_obs as u64).filter(|&i| mask[i as usize]).collect(),
        None => (0..n_obs as u64).collect(),
    };
    let live_obs = match &keep_mask {
        Some(mask) => {
            let bool_arr = arrow::array::BooleanArray::from(mask.clone());
            arrow::compute::filter_record_batch(&obs_full, &bool_arr)?
        }
        None => obs_full.clone(),
    };
    let n_live = live_obs.num_rows();
    if n_live == 0 {
        return Err(OpsError::InvalidInput(
            "scx sort: input has no live rows to sort".to_string(),
        ));
    }

    let extractor = SortKeyExtractor::new(&live_obs.schema(), &opts.by, opts.reverse)?;
    let rows = extractor.rows(&live_obs)?;
    // Local indices into `live_obs`, in sorted order (stable, ties by source id).
    let order_local = stable_argsort(&rows, 0);
    // Output row -> original (global) old row id.
    let order_old: Vec<u64> = order_local.iter().map(|&l| live_ids[l as usize]).collect();

    let sorted_obs = take_rows(&live_obs, &order_local)?;

    // ----- Output writer + header -----
    // Drop has_deletion_vectors (bit 5) and has_csc (bit 0); the sort applies
    // the deletion vector and drops the now-stale column-major sidecar.
    let out_flags = in_header.flags & !(1 << 5) & !(1 << 0);
    if in_header.has_csc() {
        log::warn!(
            "scx sort dropped CSC shards from {}: pass --rebuild-csc to restore the \
             column-major sidecar",
            input.display()
        );
    }
    let out_header = FileHeader {
        format_version: scx_format_io::rewrite_output_format_version(
            &[in_header.format_version],
            1,
        ),
        flags: out_flags,
        n_obs: n_live as u64,
        n_vars,
        shard_target_rows: opts.shard_target_rows,
        index_dtype: in_header.index_dtype,
        ..Default::default()
    };
    let mut writer = ScxWriter::new(output, out_header)?
        .with_data_generation(reader.catalog().data_generation + 1);

    // ----- obs (sorted, re-sharded) + var -----
    write_obs_sharded(&mut writer, &sorted_obs, opts.shard_target_rows)?;
    let var = reader.read_var()?;
    writer.write_var(&var)?;

    // ----- X: strategy-specific gather -----
    let value_encoding = x_value_encoding(&reader)?;
    let total_nnz: u64 = reader
        .catalog()
        .shards_sorted()
        .iter()
        .filter_map(|e| e.stats.as_ref().map(|s| s.nnz))
        .sum();
    let density = if n_obs > 0 && n_vars > 0 {
        (total_nnz as f64 / (n_obs as f64 * n_vars as f64)).clamp(1e-9, 1.0)
    } else {
        1.0
    };
    let est_x_bytes = total_nnz.saturating_mul(8).saturating_add(n_obs as u64 * 8);

    // Leading-key cardinality + single-categorical-key flag for selector / K-pass.
    let leading_categorical = !is_numeric(
        live_obs
            .schema()
            .field_with_name(&opts.by[0])
            .map(|f| f.data_type().clone())
            .unwrap_or(DataType::Utf8),
    );
    let single_key = opts.by.len() == 1;

    let mut spill_bytes = 0u64;
    let mut partitions = 1usize;

    // Build the category map only if K-pass is a candidate (avoids the scan
    // when an explicit non-K-pass strategy is forced or selected).
    let categories = if single_key && leading_categorical {
        Some(distinct_categories(&live_obs, &opts.by[0], opts.reverse)?)
    } else {
        None
    };
    let k = categories.as_ref().map(|c| c.len()).unwrap_or(usize::MAX);

    let strategy = force_strategy.unwrap_or_else(|| {
        select_strategy(opts, est_x_bytes, single_key && leading_categorical, k)
    });

    let mut x_emitter = CsrEmitter::new(
        EmitTarget::X,
        opts.shard_target_rows,
        opts.codec,
        value_encoding,
    );
    match strategy {
        SortStrategy::InMemory => {
            emit_x_in_memory(&reader, &mut writer, &mut x_emitter, &order_old)?;
        }
        SortStrategy::KPassByCategory => {
            let cats = categories.as_ref().ok_or_else(|| {
                OpsError::InvalidInput(
                    "K-pass strategy requires a single categorical sort key".to_string(),
                )
            })?;
            let cat_of_old = category_of_old(&live_obs, &live_ids, &opts.by[0], cats, n_obs)?;
            emit_x_kpass(
                &reader,
                &mut writer,
                &mut x_emitter,
                &cat_of_old,
                cats.len(),
            )?;
        }
        SortStrategy::ExternalPartition => {
            let p = partition_target_rows(
                opts.memory_budget,
                n_vars as usize,
                density,
                opts.shard_target_rows,
            );
            // Refuse if a single output shard's rows cannot fit the budget
            // (T4.7 — no silent cap).
            if let Some(budget) = opts.memory_budget {
                let per_shard =
                    (opts.shard_target_rows as f64 * n_vars as f64 * density * 16.0) as u64;
                if per_shard > budget {
                    return Err(OpsError::InvalidInput(format!(
                        "scx sort: --memory-budget {budget} too small for one output shard \
                         (~{per_shard} bytes for {} rows); raise the budget or lower --shard-size",
                        opts.shard_target_rows
                    )));
                }
            }
            // old row -> new position (-1 = deleted / absent).
            let mut new_pos_of_old = vec![-1i64; n_obs];
            for (new, &old) in order_old.iter().enumerate() {
                new_pos_of_old[old as usize] = new as i64;
            }
            let (sb, np) = emit_x_external(
                &reader,
                &mut writer,
                &mut x_emitter,
                &new_pos_of_old,
                p,
                n_live,
                opts.temp_dir.as_deref(),
            )?;
            spill_bytes = sb;
            partitions = np;
            log::info!(
                "scx sort: external partition sort spilled {spill_bytes} bytes across \
                 {partitions} partitions (~{p} rows each)"
            );
        }
        other => {
            return Err(OpsError::InvalidInput(format!(
                "scx sort engine does not run strategy {other:?} (build-time forms use \
                 convert / merge)"
            )));
        }
    }
    x_emitter.finish(&mut writer)?;
    let output_shard_row_ranges = x_emitter.ranges.clone();

    // ----- layers (in-memory gather by the same order) -----
    emit_layers_in_memory(&reader, &mut writer, &order_old, opts)?;

    // ----- obsm (in-memory take) -----
    let mut obsm: Vec<_> = reader.read_all_obsm()?.into_iter().collect();
    obsm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &obsm {
        let reordered = take_rows(batch, &order_old)?;
        writer.write_obsm(name, &reordered)?;
    }

    // ----- uns / varm / varp passthrough; obsp dropped (Phase 5 remap) -----
    if let Ok(uns) = reader.read_uns() {
        writer.write_uns(&uns)?;
    }
    let mut varm: Vec<_> = reader.read_all_varm()?.into_iter().collect();
    varm.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varm {
        writer.write_varm(name, batch)?;
    }
    let mut varp: Vec<_> = reader.read_all_varp()?.into_iter().collect();
    varp.sort_by(|a, b| a.0.cmp(&b.0));
    for (name, batch) in &varp {
        writer.write_varp(name, batch)?;
    }
    let obsp = reader.read_all_obsp()?;
    if !obsp.is_empty() {
        log::warn!(
            "scx sort dropped {} obsp section(s) from {}: obs×obs remap under a row \
             reorder is a Phase-5 follow-up",
            obsp.len(),
            input.display()
        );
    }

    // ----- predicate index (always (re)built; sort key auto-added) -----
    let index_result = rebuild_obs_predicate_index_streaming(
        &mut writer,
        sorted_obs.schema(),
        std::iter::once((sorted_obs.clone(), 0u64)),
        &var,
        &output_shard_row_ranges,
        n_vars as usize,
        &opts.by,
        &opts.index_options,
    )?;
    let indexed_columns = index_result.obs_indexed_columns.clone();

    // ----- provenance -----
    let mut prov = reader
        .read_provenance()
        .map(|p| p.operations)
        .unwrap_or_default();
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    prov.push(sort_provenance_entry(
        &opts.by,
        opts.reverse,
        opts.shard_target_rows,
        &indexed_columns,
        ts,
    ));
    writer.write_provenance(prov)?;

    writer.finish()?;

    Ok(SortSummary {
        n_obs: n_live as u64,
        n_output_shards: output_shard_row_ranges.len() as u64,
        strategy,
        spill_bytes,
        partitions,
        indexed_columns,
    })
}

// ---------------------------------------------------------------------------
// Strategy selection
// ---------------------------------------------------------------------------

fn select_strategy(
    opts: &SortOptions,
    est_x_bytes: u64,
    single_categorical: bool,
    k: usize,
) -> SortStrategy {
    match opts.memory_budget {
        // No budget set: the caller did not ask us to bound memory, so use the
        // simplest path. Pass --memory-budget to force the bounded engine.
        None => SortStrategy::InMemory,
        Some(budget) => {
            if est_x_bytes <= budget {
                SortStrategy::InMemory
            } else if single_categorical && k > 0 && k <= K_PASS_MAX_CARDINALITY {
                // K passes over the (local) compressed input beat writing +
                // re-reading a multi-TB decoded spill at low cardinality.
                SortStrategy::KPassByCategory
            } else {
                SortStrategy::ExternalPartition
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CSR re-accumulation emitter (factored from compact.rs:247-326 / :347-431)
// ---------------------------------------------------------------------------

enum EmitTarget {
    X,
    Layer(String),
}

struct CsrEmitter {
    target: EmitTarget,
    shard_target: u64,
    codec: CodecSelection,
    value_encoding: ValueEncoding,
    acc_indptr: Vec<u64>,
    acc_indices: Vec<u32>,
    acc_values: Vec<u8>,
    acc_row_count: u64,
    emitted_rows: u64,
    shard_idx: u32,
    ranges: Vec<(u64, u64)>,
}

impl CsrEmitter {
    fn new(
        target: EmitTarget,
        shard_target: u32,
        codec: CodecSelection,
        value_encoding: ValueEncoding,
    ) -> Self {
        Self {
            target,
            shard_target: shard_target.max(1) as u64,
            codec,
            value_encoding,
            acc_indptr: vec![0],
            acc_indices: Vec::new(),
            acc_values: Vec::new(),
            acc_row_count: 0,
            emitted_rows: 0,
            shard_idx: 0,
            ranges: Vec::new(),
        }
    }

    fn push_row(&mut self, writer: &mut ScxWriter, indices: &[i32], data: &[f32]) -> Result<()> {
        for (k, &col) in indices.iter().enumerate() {
            self.acc_indices.push(col as u32);
            encode_value(&mut self.acc_values, data[k], self.value_encoding)?;
        }
        let prev = *self.acc_indptr.last().unwrap();
        self.acc_indptr.push(prev + indices.len() as u64);
        self.acc_row_count += 1;
        if self.acc_row_count >= self.shard_target {
            self.flush(writer)?;
        }
        Ok(())
    }

    fn flush(&mut self, writer: &mut ScxWriter) -> Result<()> {
        if self.acc_row_count == 0 {
            return Ok(());
        }
        let codec = resolve_codec(self.codec, &self.acc_values, self.value_encoding);
        let row_start = self.emitted_rows;
        match &self.target {
            EmitTarget::X => writer.write_csr_shard(
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
            )?,
            EmitTarget::Layer(name) => writer.write_layer_csr_shard(
                &self.acc_indptr,
                &self.acc_indices,
                &self.acc_values,
                codec,
                self.value_encoding,
                row_start,
                name,
                self.shard_idx,
            )?,
        }
        self.emitted_rows += self.acc_row_count;
        self.ranges.push((row_start, self.emitted_rows));
        self.acc_indptr = vec![0];
        self.acc_indices.clear();
        self.acc_values.clear();
        self.acc_row_count = 0;
        self.shard_idx += 1;
        Ok(())
    }

    fn finish(&mut self, writer: &mut ScxWriter) -> Result<()> {
        self.flush(writer)
    }
}

fn resolve_codec(sel: CodecSelection, values: &[u8], enc: ValueEncoding) -> CodecId {
    match sel {
        CodecSelection::Auto => select_codec(values, enc),
        CodecSelection::Explicit(c) => c,
    }
}

// ---------------------------------------------------------------------------
// Strategy (a) — in-memory
// ---------------------------------------------------------------------------

fn emit_x_in_memory(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    emitter: &mut CsrEmitter,
    order_old: &[u64],
) -> Result<()> {
    let csr = reader.read_all_csr_shards()?;
    for &old in order_old {
        let s = csr.indptr[old as usize] as usize;
        let e = csr.indptr[old as usize + 1] as usize;
        emitter.push_row(writer, &csr.indices[s..e], &csr.data[s..e])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy (b) — K-pass-by-category (single categorical key, zero spill)
// ---------------------------------------------------------------------------

fn emit_x_kpass(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    emitter: &mut CsrEmitter,
    cat_of_old: &[i32],
    n_categories: usize,
) -> Result<()> {
    let shards = reader.catalog().shards_sorted();
    for cat in 0..n_categories as i32 {
        for shard in &shards {
            let (indptr, indices, data) = reader.read_shard_from_entry(shard)?;
            let row_start = shard.stats.as_ref().map(|s| s.row_start).unwrap_or(0);
            for local in 0..indptr.len() - 1 {
                let global = row_start as usize + local;
                if cat_of_old[global] == cat {
                    let s = indptr[local] as usize;
                    let e = indptr[local + 1] as usize;
                    emitter.push_row(writer, &indices[s..e], &data[s..e])?;
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy (c) — external partition sort (spill, bounded memory)
// ---------------------------------------------------------------------------

/// A spill session directory removed on drop (success or error).
struct SpillDir {
    path: PathBuf,
}

impl SpillDir {
    fn create(base: Option<&Path>) -> Result<Self> {
        let base = base
            .map(|p| p.to_path_buf())
            .unwrap_or_else(std::env::temp_dir);
        // Session uniqueness via pid + monotonic nanos (Math.random is not used
        // here; a collision would only be across concurrent same-pid sorts).
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = base.join(format!("scx-sort-{}-{}", std::process::id(), nanos));
        std::fs::create_dir_all(&path)?;
        Ok(Self { path })
    }

    fn partition_file(&self, p: usize) -> PathBuf {
        self.path.join(format!("part_{p}.bin"))
    }
}

impl Drop for SpillDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Pass 1 (scatter) + pass 2 (load/sort/emit). Returns `(spill_bytes,
/// n_partitions)`. Partitions are `new_pos` ranges of width `p`, so each holds
/// at most `p` rows regardless of key skew (no sub-split needed).
fn emit_x_external(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    emitter: &mut CsrEmitter,
    new_pos_of_old: &[i64],
    p: usize,
    n_live: usize,
    temp_dir: Option<&Path>,
) -> Result<(u64, usize)> {
    let p = p.max(1);
    let n_parts = n_live.div_ceil(p);
    let spill = SpillDir::create(temp_dir)?;

    // ----- Pass 1: scatter decoded rows to per-partition spill files -----
    let mut part_writers: Vec<BufWriter<File>> = (0..n_parts)
        .map(|i| Ok(BufWriter::new(File::create(spill.partition_file(i))?)))
        .collect::<Result<_>>()?;
    let mut spill_bytes = 0u64;

    let shards = reader.catalog().shards_sorted();
    for shard in &shards {
        let (indptr, indices, data) = reader.read_shard_from_entry(shard)?;
        let row_start = shard.stats.as_ref().map(|s| s.row_start).unwrap_or(0) as usize;
        for local in 0..indptr.len() - 1 {
            let global = row_start + local;
            let np = new_pos_of_old[global];
            if np < 0 {
                continue; // deleted row
            }
            let np = np as u64;
            let part = (np as usize) / p;
            let s = indptr[local] as usize;
            let e = indptr[local + 1] as usize;
            spill_bytes +=
                write_spill_row(&mut part_writers[part], np, &indices[s..e], &data[s..e])?;
        }
    }
    for w in &mut part_writers {
        w.flush()?;
    }
    drop(part_writers);

    // ----- Pass 2: per partition, load + sort by new_pos + emit -----
    for part in 0..n_parts {
        let bytes = std::fs::read(spill.partition_file(part))?;
        let mut rows = parse_spill(&bytes)?;
        rows.sort_by_key(|r| r.0);
        for (_np, indices, data) in &rows {
            emitter.push_row(writer, indices, data)?;
        }
    }

    Ok((spill_bytes, n_parts))
}

/// Write one spill record: `new_pos:u64, nnz:u32, indices:i32×nnz,
/// values:f32×nnz`. Returns the bytes written.
fn write_spill_row(
    w: &mut BufWriter<File>,
    new_pos: u64,
    indices: &[i32],
    data: &[f32],
) -> Result<u64> {
    let nnz = indices.len() as u32;
    w.write_all(&new_pos.to_le_bytes())?;
    w.write_all(&nnz.to_le_bytes())?;
    for &idx in indices {
        w.write_all(&idx.to_le_bytes())?;
    }
    for &v in data {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(8 + 4 + nnz as u64 * 8)
}

/// Parse a partition spill buffer into `(new_pos, indices, values)` rows.
#[allow(clippy::type_complexity)]
fn parse_spill(buf: &[u8]) -> Result<Vec<(u64, Vec<i32>, Vec<f32>)>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    let err = || OpsError::InvalidInput("scx sort: truncated spill record".to_string());
    while off < buf.len() {
        if off + 12 > buf.len() {
            return Err(err());
        }
        let np = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
        let nnz = u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap()) as usize;
        off += 12;
        let idx_end = off + nnz * 4;
        let val_end = idx_end + nnz * 4;
        if val_end > buf.len() {
            return Err(err());
        }
        let mut indices = Vec::with_capacity(nnz);
        for k in 0..nnz {
            let b = off + k * 4;
            indices.push(i32::from_le_bytes(buf[b..b + 4].try_into().unwrap()));
        }
        let mut data = Vec::with_capacity(nnz);
        for k in 0..nnz {
            let b = idx_end + k * 4;
            data.push(f32::from_le_bytes(buf[b..b + 4].try_into().unwrap()));
        }
        off = val_end;
        out.push((np, indices, data));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Layers (in-memory gather, all strategies)
// ---------------------------------------------------------------------------

fn emit_layers_in_memory(
    reader: &ScxReader,
    writer: &mut ScxWriter,
    order_old: &[u64],
    opts: &SortOptions,
) -> Result<()> {
    for layer_name in reader.layer_names() {
        let ve = layer_value_encoding(reader, &layer_name)?;
        let layer = reader.read_layer(&layer_name)?;
        let mut emitter = CsrEmitter::new(
            EmitTarget::Layer(layer_name.clone()),
            opts.shard_target_rows,
            opts.codec,
            ve,
        );
        for &old in order_old {
            let s = layer.indptr[old as usize] as usize;
            let e = layer.indptr[old as usize + 1] as usize;
            emitter.push_row(writer, &layer.indices[s..e], &layer.data[s..e])?;
        }
        emitter.finish(writer)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Gather `batch` rows by global index list (arrow `take` per column).
fn take_rows(batch: &RecordBatch, idx: &[u64]) -> Result<RecordBatch> {
    let idx_arr = UInt64Array::from(idx.to_vec());
    let cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c, &idx_arr, None))
        .collect::<std::result::Result<_, _>>()?;
    Ok(RecordBatch::try_new(batch.schema(), cols)?)
}

/// Write a sorted obs batch as `shard_target_rows`-sized `ObsMetadataShard`
/// sections (atlas-friendly; `write_obs_shard` upcasts `Utf8 → LargeUtf8`
/// per shard so no shard hits the Arrow IPC 2 GB ceiling).
fn write_obs_sharded(writer: &mut ScxWriter, obs: &RecordBatch, shard_target: u32) -> Result<()> {
    let total = obs.num_rows() as u64;
    let st = shard_target.max(1) as usize;
    let mut shard_idx = 0u32;
    let mut start = 0usize;
    while start < obs.num_rows() {
        let len = (obs.num_rows() - start).min(st);
        let slice = obs.slice(start, len);
        writer.write_obs_shard(shard_idx, start as u64, len as u64, total, &slice)?;
        start += len;
        shard_idx += 1;
    }
    Ok(())
}

fn is_numeric(dt: DataType) -> bool {
    use DataType::*;
    matches!(
        dt,
        Int8 | Int16
            | Int32
            | Int64
            | UInt8
            | UInt16
            | UInt32
            | UInt64
            | Float16
            | Float32
            | Float64
    )
}

/// Distinct values of a categorical/string-like column in sorted (or reversed)
/// order. Nulls excluded (routed to the leading partition under `nulls_first`).
fn distinct_categories(obs: &RecordBatch, name: &str, reverse: bool) -> Result<Vec<String>> {
    use arrow::array::StringArray;
    let col = obs.column_by_name(name).ok_or_else(|| {
        OpsError::InvalidInput(format!("sort key column '{name}' missing from obs"))
    })?;
    let utf8 = arrow::compute::cast(col, &DataType::Utf8)?;
    let arr = utf8.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        OpsError::InvalidInput(format!("sort key '{name}' is not categorical/string-like"))
    })?;
    let mut set = std::collections::BTreeSet::new();
    for i in 0..arr.len() {
        if !arr.is_null(i) {
            set.insert(arr.value(i).to_string());
        }
    }
    let mut cats: Vec<String> = set.into_iter().collect();
    if reverse {
        cats.reverse();
    }
    Ok(cats)
}

/// Map global old row id -> category index (-1 = deleted / null). `cats` is the
/// sorted category order; the leading key column is read from `live_obs`.
fn category_of_old(
    live_obs: &RecordBatch,
    live_ids: &[u64],
    name: &str,
    cats: &[String],
    n_obs: usize,
) -> Result<Vec<i32>> {
    use arrow::array::StringArray;
    let col = live_obs.column_by_name(name).ok_or_else(|| {
        OpsError::InvalidInput(format!("sort key column '{name}' missing from obs"))
    })?;
    let utf8 = arrow::compute::cast(col, &DataType::Utf8)?;
    let arr = utf8.as_any().downcast_ref::<StringArray>().unwrap();
    let index: std::collections::HashMap<&str, i32> = cats
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as i32))
        .collect();
    let mut out = vec![-1i32; n_obs];
    for local in 0..arr.len() {
        if arr.is_null(local) {
            continue;
        }
        if let Some(&ci) = index.get(arr.value(local)) {
            out[live_ids[local] as usize] = ci;
        }
    }
    Ok(out)
}

/// Value encoding of the main X matrix (first CSR shard header).
fn x_value_encoding(reader: &ScxReader) -> Result<ValueEncoding> {
    let shards = reader.catalog().shards_sorted();
    entry_value_encoding(reader, shards.first().copied())
}

/// Value encoding of a layer (first layer-shard header).
fn layer_value_encoding(reader: &ScxReader, layer_name: &str) -> Result<ValueEncoding> {
    let prefix = format!("{layer_name}_shard_");
    let first = reader
        .catalog()
        .entries
        .iter()
        .find(|e| e.section_type == SectionType::LayerCsrShard && e.name.starts_with(&prefix));
    entry_value_encoding(reader, first)
}

fn entry_value_encoding(
    reader: &ScxReader,
    entry: Option<&FullCatalogEntry>,
) -> Result<ValueEncoding> {
    match entry {
        Some(e) => {
            let section = reader.section_bytes(e)?;
            let sh =
                ShardHeader::read_from(&mut std::io::Cursor::new(&section[..SHARD_HEADER_SIZE]))?;
            ValueEncoding::from_u8(sh.value_encoding)
                .ok_or(OpsError::UnknownValueEncoding(sh.value_encoding))
        }
        None => Ok(ValueEncoding::Uint8),
    }
}

#[cfg(test)]
#[path = "sort_engine_tests.rs"]
mod tests;
