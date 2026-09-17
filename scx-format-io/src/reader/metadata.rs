//! Arrow IPC sections: obs / var / obsm / varm / obsp / varp, sharded
//! metadata assembly, and dictionary unification.
//!
//! The largest coherent block in the reader, and the cleanest cut: it
//! reaches the mapping only through [`ScxReader::section_bytes`].

use super::*;

/// Per-shard metadata stamped by the writer
/// (see `crate::writer::stamp_dense_shard_meta`). Parsed by the
/// sharded reader path to verify a contiguous, ordered cover of the
/// logical matrix. Distinct from `crate::shard::ShardHeader`, which is
/// the on-disk 76-byte CSR/CSC shard header.
struct RowShardMetadata {
    shard_idx: u32,
    row_start: u64,
    n_shard_rows: u64,
    n_rows_total: u64,
}

/// Physical layout of a row-sharded Arrow IPC mapping, resolved by
/// [`ScxReader::row_sharded_mapping_layout`] and consumed by
/// [`crate::BackedDenseReader`] (`obsm`, dense) and
/// [`crate::BackedPairwiseReader`] (`obsp`, COO).
///
/// The shard cover is identical for both families — every sharded
/// mapping stamps `shard_idx` / `row_start` / `n_shard_rows` /
/// `n_rows_total` through `writer::stamp_dense_shard_meta`. What differs
/// is the *legacy* single-section case and what the column fields mean;
/// see [`LegacyRowCount`].
pub(crate) struct MappingLayout {
    /// Per-shard rows, ordered by `shard_idx` (== sorted by `row_start`).
    pub(crate) entries: Vec<MappingShardLayoutEntry>,
    /// Total logical row count (last shard's `n_rows_total`).
    pub(crate) n_rows: u64,
    /// Number of Arrow columns. Dense: the embedding dimensionality.
    /// **Pairwise: 3** (`row` / `col` / `data`) — it is *not* the
    /// matrix's column count, which lives in the `n_cols` schema
    /// metadata key instead.
    pub(crate) n_cols: usize,
    /// Column-0 dtype, as a representative for the whole mapping.
    pub(crate) dtype: arrow::datatypes::DataType,
    /// Canonical column fields (per-shard schema metadata stripped) —
    /// the row-gather output schema. Taken from the first shard.
    pub(crate) fields: arrow::datatypes::Fields,
    /// The **matrix's** column count, from the `n_cols` schema metadata
    /// key. Only pairwise (COO) sections stamp it, because only they
    /// have a column axis that the Arrow columns do not describe; `None`
    /// on a dense mapping, where `n_cols` above is the answer.
    pub(crate) matrix_n_cols: Option<u64>,
}

/// How to read the row count of a **legacy single-section** mapping,
/// which carries no `row_start` / `n_shard_rows` stamp.
///
/// This is the one place the dense and pairwise families genuinely
/// diverge, and getting it wrong is silent: for a dense `obsm` section
/// one Arrow row *is* one obs row, but for a COO `obsp` section one
/// Arrow row is one `(row, col, data)` triple, so `batch.num_rows()`
/// answers **nnz**. Since phase 9 `scx sort` and `scx compact` write obsp
/// back **sharded** (both go through
/// `scx_format_io::for_each_coo_mapping_shard`), so the legacy arm is no
/// longer something those two ops produce — but it is still not a hypothetical
/// branch: every file written before phase 9 takes it, and so does anything
/// `from_h5mu` or an in-place metadata op writes today.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LegacyRowCount {
    /// One Arrow row per logical row — dense `obsm` / `varm`.
    BatchRows,
    /// The `n_rows` schema metadata key — COO `obsp` / `varp`.
    SchemaNRows,
}

/// One shard's catalog offset + stamped row range. Mirrors the fields
/// the backed readers need (no `nnz`, since these are not CSR shards).
///
/// `Copy` so a backed reader can retain these directly rather than defining a
/// private twin with the same six fields and copying them across one by one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MappingShardLayoutEntry {
    pub(crate) offset: u64,
    pub(crate) length: u64,
    pub(crate) section_type: SectionType,
    pub(crate) modality_id: u8,
    pub(crate) row_start: u64,
    pub(crate) n_shard_rows: u64,
}

/// Pull `shard_idx` / `row_start` / `n_shard_rows` / `n_rows_total`
/// off a sharded batch's schema metadata. Returns
/// `ScxError::InvalidCatalog` if any field is missing or unparseable,
/// naming the logical section so the caller can produce a useful error.
fn parse_shard_metadata(logical: &str, batch: &RecordBatch) -> Result<RowShardMetadata> {
    parse_shard_metadata_md(logical, batch.schema_ref().metadata())
}

/// Like [`parse_shard_metadata`] but reads from a schema metadata map
/// directly, so the backed dense reader can pull row ranges from an
/// Arrow IPC **footer schema** (no batch deserialisation) at open time.
fn parse_shard_metadata_md(
    logical: &str,
    md: &std::collections::HashMap<String, String>,
) -> Result<RowShardMetadata> {
    let get = |key: &str| -> Result<u64> {
        md.get(key)
            .ok_or_else(|| {
                ScxError::InvalidCatalog(format!("{logical}: shard schema missing '{key}'"))
            })
            .and_then(|s| {
                s.parse::<u64>().map_err(|_| {
                    ScxError::InvalidCatalog(format!(
                        "{logical}: shard schema '{key}'='{s}' is not a u64"
                    ))
                })
            })
    };
    let shard_idx = u32::try_from(get("shard_idx")?)
        .map_err(|_| ScxError::InvalidCatalog(format!("{logical}: shard_idx exceeds u32::MAX")))?;
    Ok(RowShardMetadata {
        shard_idx,
        row_start: get("row_start")?,
        n_shard_rows: get("n_shard_rows")?,
        n_rows_total: get("n_rows_total")?,
    })
}

/// Assemble a set of raw (un-downcast) metadata-shard `RecordBatch`es into
/// one logical batch.
///
/// `raw_batches` are `(shard_idx, batch)` pairs decoded from disk or an
/// object store **without** any wide→narrow downcast — the caller is
/// responsible only for fetching and Arrow-IPC-decoding the shard bytes.
/// The shared assembly steps live here so the local mmap reader
/// ([`ScxReader::read_sharded_layout_by_prefix`]) and the cloud reader
/// produce byte-identical results:
///
/// 1. upcast every batch to `LargeUtf8` / `LargeBinary` (no-op when the
///    writer already serialised wide types);
/// 2. validate the shards form a contiguous, ordered cover via the stamped
///    `shard_idx` / `row_start` / `n_shard_rows` / `n_rows_total` metadata
///    (monotonic non-decreasing `n_rows_total`, the last shard's total
///    equals the cumulative cover);
/// 3. concat on the wide schema (safe to `i64::MAX` offsets);
/// 4. downcast back to narrow `Utf8` / `Binary` for columns whose combined
///    offsets fit, leaving over-`i32::MAX` columns wide;
/// 5. strip the per-shard metadata keys from the result schema.
///
/// Errors with `InvalidCatalog` on any cover violation and
/// `SectionNotFound` when `raw_batches` is empty.
/// Collapse every dictionary (categorical) column in `batch` to a unified
/// dictionary with distinct values.
///
/// `arrow::compute::concat` concatenates per-shard dictionary value arrays
/// without deduplicating, so concatenating N shards that each hold the same
/// category produces a dictionary with that category repeated N times. Such a
/// batch round-trips through Arrow IPC fine, but `pyarrow.Table.to_pandas()`
/// raises `ValueError: Categorical categories must be unique`. Interning the
/// *declared* values ([`dedup_dictionary_column`]) rebuilds a deduplicated
/// dictionary with keys remapped onto it and every declared entry kept, used or
/// not; a value type the interner cannot key falls back to decode→re-encode
/// (which keeps only referenced values). Non-dictionary columns pass through
/// untouched; the schema (and its `pandas` index metadata) is preserved.
fn unify_dictionary_columns(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::datatypes::{DataType, Field, Schema};
    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| matches!(f.data_type(), DataType::Dictionary(_, _)))
    {
        return Ok(batch.clone());
    }
    let mut new_fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<arrow::array::ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        match field.data_type() {
            DataType::Dictionary(_, value_type) => {
                // Fast path: deduplicate by remapping over the (small)
                // concatenated dictionary *values* array — O(dict_len) hashing
                // + an O(n_obs) integer key gather — instead of decoding the
                // whole column to a flat `n_obs`-length Utf8 array (the
                // multi-GB-per-column transient that OOMs atlas-scale
                // `read_obs`; see SCX-SORT-OOM-BUG Part 3). Falls back to the
                // decode/re-encode cast only for a value type the row encoder
                // cannot key (a nested type).
                let (encoded, final_dt) = match dedup_dictionary_column(col, value_type)? {
                    Some(pair) => pair,
                    None => {
                        // Decode to the plain value array (drops the per-shard,
                        // possibly-duplicated dictionary), then re-encode to a
                        // fresh unified dictionary. Encode once with a wide Int32
                        // key to learn the deduplicated cardinality, then
                        // re-encode with the minimal signed key type that fits.
                        let values = arrow::compute::cast(col, value_type.as_ref())?;
                        let wide_dt =
                            DataType::Dictionary(Box::new(DataType::Int32), value_type.clone());
                        let wide = arrow::compute::cast(&values, &wide_dt)?;
                        let n_distinct = wide
                            .as_any()
                            .downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>()
                            .map(|d| d.values().len())
                            .unwrap_or(usize::MAX);
                        let key_type = min_dictionary_key_type(n_distinct);
                        let final_dt = DataType::Dictionary(Box::new(key_type), value_type.clone());
                        let encoded = if final_dt == wide_dt {
                            wide
                        } else {
                            arrow::compute::cast(&values, &final_dt)?
                        };
                        (encoded, final_dt)
                    }
                };
                new_columns.push(encoded);
                new_fields.push(
                    Field::new(field.name(), final_dt, field.is_nullable())
                        .with_metadata(field.metadata().clone()),
                );
            }
            _ => {
                new_columns.push(col.clone());
                new_fields.push(field.as_ref().clone());
            }
        }
    }
    let new_schema = Schema::new(new_fields).with_metadata(schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(new_schema), new_columns)?)
}

/// Deduplicate a `Dictionary(Int32, value_type)` column without materializing
/// the full `n_obs`-length value array. Returns the unified `(array, dtype)`
/// (key narrowed via [`min_dictionary_key_type`]), or `None` for a shape that
/// should take the decode/re-encode fallback (non-`Int32` keys — not produced
/// by the assembler's `widen_dictionary_keys` — or a value type the row
/// encoder cannot key, see [`intern_declared_values`]).
///
/// Generic over the value type through [`intern_declared_values`]: string,
/// Boolean, every primitive (an integer or float categorical, a timestamp)
/// all go through one interner over the *declared* values, so an entry no row
/// references survives whatever its type. A dictionary that is already unique
/// — the usual case now that [`share_dictionary_values`] runs before concat —
/// skips the O(n_obs) key remap and only narrows the key type.
fn dedup_dictionary_column(
    col: &arrow::array::ArrayRef,
    value_type: &arrow::datatypes::DataType,
) -> Result<Option<(arrow::array::ArrayRef, arrow::datatypes::DataType)>> {
    use arrow::array::Array;
    use arrow::datatypes::Int32Type;

    let Some(dict) = col
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<Int32Type>>()
    else {
        return Ok(None);
    };
    let Some((unified, mut maps)) = intern_declared_values(&[dict.values()])? else {
        return Ok(None);
    };
    let map = maps.pop().expect("one input, one map");
    let n_distinct = unified.len();
    let keys = if n_distinct == dict.values().len() {
        // Already unique: the remap would be the identity.
        dict.keys().clone()
    } else {
        remap_dictionary_keys(dict.keys(), &map)?
    };
    Ok(Some(finish_deduped_dictionary(
        keys, unified, n_distinct, value_type,
    )?))
}

/// Intern the union of several dictionaries' **declared** values.
///
/// Returns the unified values array — first-occurrence order over `inputs` in
/// order and, within one, declared order; null values collapse to one slot
/// (null == null in SCX categorical semantics) — plus, per input, the map from
/// its old value index to the unified index. `None` when the value type is one
/// `arrow::row::RowConverter` cannot encode, which for a dictionary's value
/// type means a nested type; every primitive, string, binary and Boolean value
/// type is covered.
///
/// One interner for every value type is the point: the dedup and the
/// shared-values step used to special-case strings and Boolean and fall back to
/// decode→re-encode for the rest, which rebuilds the vocabulary from the *rows*
/// and so prunes and reorders a numeric categorical's declared levels.
/// Encoding through the row format keys on the value itself, whatever its type
/// (a float `-0.0` and `0.0` are distinct entries, as in pandas; NaNs collapse).
/// Cost is O(total declared values) hashing plus one `take`; no `n_obs`-length
/// transient.
fn intern_declared_values(
    inputs: &[&arrow::array::ArrayRef],
) -> Result<Option<(arrow::array::ArrayRef, Vec<Vec<u32>>)>> {
    use arrow::array::{Array, UInt32Array};
    use arrow::row::{RowConverter, SortField};

    let Some(first) = inputs.first() else {
        return Ok(None);
    };
    let field = SortField::new(first.data_type().clone());
    if !RowConverter::supports_fields(std::slice::from_ref(&field)) {
        return Ok(None);
    }
    let all = match inputs {
        [one] => (*one).clone(),
        many => arrow::compute::concat(&many.iter().map(|a| a.as_ref()).collect::<Vec<_>>())?,
    };
    let converter = RowConverter::new(vec![field])?;
    let rows = converter.convert_columns(std::slice::from_ref(&all))?;

    let mut interner: HashMap<arrow::row::Row<'_>, u32> = HashMap::new();
    let mut keep: Vec<u32> = Vec::new();
    let mut flat: Vec<u32> = Vec::with_capacity(all.len());
    for i in 0..rows.num_rows() {
        let code = *interner.entry(rows.row(i)).or_insert_with(|| {
            keep.push(i as u32);
            (keep.len() - 1) as u32
        });
        flat.push(code);
    }
    let unified = if keep.len() == all.len() {
        all
    } else {
        arrow::compute::take(all.as_ref(), &UInt32Array::from(keep), None)?
    };
    let mut maps = Vec::with_capacity(inputs.len());
    let mut off = 0usize;
    for a in inputs {
        maps.push(flat[off..off + a.len()].to_vec());
        off += a.len();
    }
    Ok(Some((unified, maps)))
}

/// Remap a dictionary's keys through `old_to_new` (an O(n_obs) integer
/// gather; null keys preserved). Each key must index into the source
/// dictionary; a malformed file with an out-of-range (or negative) key is
/// rejected rather than panicking on the slice index (readers return errors
/// on malformed input — see docs/conventions.md).
fn remap_dictionary_keys(
    keys: &arrow::array::Int32Array,
    old_to_new: &[u32],
) -> Result<arrow::array::Int32Array> {
    keys.iter()
        .map(|k| {
            k.map(|k| {
                let idx = usize::try_from(k).ok().filter(|&i| i < old_to_new.len());
                match idx {
                    Some(i) => Ok(old_to_new[i] as i32),
                    None => Err(ScxError::InvalidCatalog(format!(
                        "dictionary key {k} out of bounds (dictionary length {})",
                        old_to_new.len()
                    ))),
                }
            })
            .transpose()
        })
        .collect::<Result<arrow::array::Int32Array>>()
}

/// Assemble the deduplicated `(keys, values)` into a dictionary array whose
/// key type is the minimal fit for `n_distinct`.
fn finish_deduped_dictionary(
    new_keys: arrow::array::Int32Array,
    unified_values: arrow::array::ArrayRef,
    n_distinct: usize,
    value_type: &arrow::datatypes::DataType,
) -> Result<(arrow::array::ArrayRef, arrow::datatypes::DataType)> {
    use arrow::array::DictionaryArray;
    use arrow::datatypes::{DataType, Int32Type};

    let wide = DictionaryArray::<Int32Type>::try_new(new_keys, unified_values)?;
    let key_type = min_dictionary_key_type(n_distinct);
    let is_int32 = key_type == DataType::Int32;
    let final_dt = DataType::Dictionary(Box::new(key_type), Box::new(value_type.clone()));
    let arr: arrow::array::ArrayRef = if is_int32 {
        Arc::new(wide)
    } else {
        // Narrows keys only (values untouched) — no full-column materialization.
        // `cast` on a dictionary only re-keys, so it is safe for value types
        // that cannot be dictionary-*packed* from a plain array.
        arrow::compute::cast(&wide, &final_dt)?
    };
    Ok((arr, final_dt))
}

/// Smallest signed Arrow dictionary key (index) type that can address
/// `n_distinct` values: `Int8` for ≤ `i8::MAX`, `Int16` for ≤ `i16::MAX`,
/// else `Int32`. Keys are non-negative indices, so the signed maxima are the
/// addressable counts. Mirrors the compact code widths anndata/pandas use for
/// categoricals while guaranteeing no overflow.
fn min_dictionary_key_type(n_distinct: usize) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType;
    if n_distinct <= i8::MAX as usize {
        DataType::Int8
    } else if n_distinct <= i16::MAX as usize {
        DataType::Int16
    } else {
        DataType::Int32
    }
}

/// Decode an Arrow IPC schema from a raw section byte slice.
///
/// - **Fast path** (no `LargeUtf8` / `LargeBinary` / `Dictionary(_, Large*)`
///   in the footer): return the IPC footer schema directly. No data
///   deserialization — ~KB of work.
/// - **Slow path** (any wide type present): deserialize only the *first*
///   batch and run [`crate::arrow_compat::downcast_large_types`] so the
///   returned schema matches what the data path produces — narrow types
///   when offsets fit, wide types when they overflow.
///
/// Cost is bounded by the bytes passed in (one section / one shard), never
/// the full logical table. This is the shared core of
/// [`ScxReader::read_obs_schema`] / [`ScxReader::read_var_schema`]; the
/// cloud reader calls it on a single fetched shard so its schema reads stay
/// byte-identical to the local path without assembling the whole obs/var.
pub fn decode_arrow_ipc_schema(bytes: &[u8]) -> Result<arrow::datatypes::Schema> {
    let cursor = Cursor::new(bytes);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
    let on_disk = reader.schema();

    let has_wide = on_disk.fields().iter().any(|f| {
        use arrow::datatypes::DataType;
        matches!(f.data_type(), DataType::LargeUtf8 | DataType::LargeBinary)
            || matches!(
                f.data_type(),
                DataType::Dictionary(_, v)
                    if matches!(v.as_ref(), DataType::LargeUtf8 | DataType::LargeBinary)
            )
    });
    if !has_wide {
        return Ok(on_disk.as_ref().clone());
    }

    // Wide types present — drive the slow path through the same logic the
    // data path uses, so the schema reflects whether offsets actually
    // overflow per column.
    let cursor = Cursor::new(bytes);
    let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
    let mut batches = reader.into_iter();
    let batch = batches
        .next()
        .ok_or_else(|| {
            ScxError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Arrow IPC file contains no batches",
            ))
        })?
        .map_err(ScxError::Arrow)?;
    let normalized = crate::arrow_compat::downcast_large_types(&batch)?;
    Ok(normalized.schema().as_ref().clone())
}

/// Rebuild a projected obs key shard into fresh, compact buffers so it no
/// longer aliases the full Arrow IPC message body. Column projection returns
/// each column as a zero-copy slice into the shard's whole-batch body buffer,
/// so a *retained* projected batch keeps every un-projected column resident —
/// accumulating thousands of shards would hold the entire obs table. String
/// columns are dictionary-encoded (drops the alias **and** collapses
/// categoricals stored plain — a merge/append artifact — to compact codes);
/// other columns are deep-copied. The shard's schema metadata (cover stamps
/// `shard_idx`/`row_start`/…) is preserved for the assembler.
pub fn compact_key_shard(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::array::UInt32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    let schema = batch.schema();
    // Identity gather index — `take` always writes fresh output buffers, so it
    // forces a real copy that drops the IPC-body alias for non-string key
    // columns. (`concat(&[single])` does NOT: Arrow returns the lone input
    // as-is, zero-copy, leaving the full message body alive.)
    let identity = UInt32Array::from_iter_values(0..batch.num_rows() as u32);
    let mut fields: Vec<Field> = Vec::with_capacity(batch.num_columns());
    let mut cols: Vec<arrow::array::ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, f) in schema.fields().iter().enumerate() {
        let c = batch.column(i);
        match f.data_type() {
            DataType::Utf8 | DataType::LargeUtf8 => {
                let dt = DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(f.data_type().clone()),
                );
                cols.push(arrow::compute::cast(c, &dt)?);
                fields.push(
                    Field::new(f.name(), dt, f.is_nullable()).with_metadata(f.metadata().clone()),
                );
            }
            _ => {
                // Deep-copy via identity `take` to drop the body alias; dtype preserved.
                cols.push(arrow::compute::take(c.as_ref(), &identity, None)?);
                fields.push(f.as_ref().clone());
            }
        }
    }
    let new_schema = Schema::new(fields).with_metadata(schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(new_schema), cols)?)
}

/// Give every shard's dictionary column **one shared values array** before
/// `concat_batches`.
///
/// Arrow's `concat` of dictionary arrays has two behaviours. When every input
/// shares the same values buffer it appends the keys and keeps the values;
/// otherwise, once the summed per-shard dictionary lengths reach the row count
/// (`should_merge_dictionary_values`: `total_values >= len`), it *merges* the
/// dictionaries — keeping only the values some key references, in an order of
/// its own. Every shard of a `from_anndata` file carries the full declared
/// vocabulary (a slice of a dictionary array keeps the whole values array), so
/// on a small file that threshold is met and a declared-but-unused category (a
/// `pd.Categorical` level no cell has) silently vanished from `read_obs()` and
/// the surviving levels came back reordered — while the same column on a large
/// file kept both. A row-count-dependent prune is the worst kind of
/// intermittent, and the h5ad exporter and `obs_categorical`, which fold shards
/// themselves, never had it.
///
/// This interns the union of every shard's declared values through
/// [`intern_declared_values`] — first-occurrence order over shards in index
/// order and, within a shard, declared order — remaps each shard's keys onto
/// it, and rebuilds each column over the **same `Arc`**, so `concat` takes its
/// keys-only path and [`unify_dictionary_columns`] afterwards has nothing to
/// deduplicate beyond narrowing the key. Runs after
/// [`crate::arrow_compat::widen_dictionary_keys`] and
/// [`crate::arrow_compat::reconcile_dictionary_representations`], so every
/// batch's column for a field is already `Dictionary(Int32, V)` with one `V`.
/// Every value type the row encoder can key is covered (strings, Boolean, all
/// primitives); a nested value type passes through untouched.
fn share_dictionary_values(batches: Vec<RecordBatch>) -> Result<Vec<RecordBatch>> {
    use arrow::array::{Array, ArrayRef, DictionaryArray};
    use arrow::datatypes::{DataType, Int32Type};

    if batches.len() < 2 {
        return Ok(batches);
    }
    let schema = batches[0].schema();
    let targets: Vec<usize> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(
            |(_, f)| matches!(f.data_type(), DataType::Dictionary(k, _) if **k == DataType::Int32),
        )
        .map(|(i, _)| i)
        .collect();
    if targets.is_empty() {
        return Ok(batches);
    }

    // Per target column: one replacement array per batch, all over one values
    // `Arc`.
    let mut replacements: Vec<(usize, Vec<ArrayRef>)> = Vec::with_capacity(targets.len());
    for ci in targets {
        let dicts: Vec<&DictionaryArray<Int32Type>> = batches
            .iter()
            .map(|b| {
                b.column(ci)
                    .as_any()
                    .downcast_ref::<DictionaryArray<Int32Type>>()
                    .ok_or_else(|| {
                        ScxError::InvalidCatalog(format!(
                            "column '{}' declared Dictionary(Int32, _) but failed downcast",
                            schema.field(ci).name()
                        ))
                    })
            })
            .collect::<Result<_>>()?;
        let values: Vec<&ArrayRef> = dicts.iter().map(|d| d.values()).collect();
        let Some((shared, maps)) = intern_declared_values(&values)? else {
            continue; // a value type the row encoder cannot key: leave it to arrow
        };
        let rebuilt = dicts
            .iter()
            .zip(maps)
            .map(|(d, map)| {
                // Shard 0's map is the identity by construction, and so is every
                // shard's when they declare the same vocabulary (the
                // `from_anndata` case) — skip the O(n_obs) walk and only swap the
                // values `Arc`, which is the part `concat` looks at.
                let identity = map.iter().enumerate().all(|(i, &m)| m as usize == i);
                let keys = if identity {
                    d.keys().clone()
                } else {
                    remap_dictionary_keys(d.keys(), &map)?
                };
                Ok(
                    Arc::new(DictionaryArray::<Int32Type>::try_new(keys, shared.clone())?)
                        as ArrayRef,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        replacements.push((ci, rebuilt));
    }
    if replacements.is_empty() {
        return Ok(batches);
    }

    batches
        .into_iter()
        .enumerate()
        .map(|(bi, batch)| {
            let mut columns = batch.columns().to_vec();
            for (ci, arrays) in &replacements {
                columns[*ci] = arrays[bi].clone();
            }
            Ok(RecordBatch::try_new(batch.schema(), columns)?)
        })
        .collect()
}

/// Drop every dictionary value no row of `batch` references, keeping the
/// survivors in declared order — pandas' `remove_unused_categories`.
///
/// The filtered read's half of the categorical contract. A full `read_obs()`
/// keeps the declared vocabulary whole (unused levels included); a
/// `filter_obs(...).collect()` result carries only the categories present in
/// the surviving rows (`docs/api.md` § Filtered-obs categorical semantics), the
/// way an AnnData subset does. The engine applies this to a result whose rows
/// the caller narrowed (an obs predicate or a limit) on both obs layouts; an
/// unfiltered `collect()` keeps the declared list like `read_obs()`. Before
/// this step existed the prune was arrow's data-dependent dictionary merge in
/// `concat` — it fired on a small result and not on a large one.
///
/// Only a **valid** row marks its category as used: a null row's stored key is
/// arbitrary (arrow guarantees it in range and nothing more), so reading it
/// would keep whatever category it happens to point at. A result with no
/// surviving rows — or only null ones — ends with no categories, as
/// `remove_unused_categories` on an empty subset does. Linear in the row count:
/// one key cast, one usage pass, one remap. Field metadata (the `ordered`
/// stamp) and the schema envelope are kept; the key type is narrowed to the
/// surviving count.
pub fn prune_unused_dictionary_values(batch: &RecordBatch) -> Result<RecordBatch> {
    use arrow::array::{Array, ArrayRef, AsArray, UInt32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};

    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| matches!(f.data_type(), DataType::Dictionary(_, _)))
    {
        return Ok(batch.clone());
    }
    let mut new_fields: Vec<Field> = Vec::with_capacity(schema.fields().len());
    let mut new_columns: Vec<ArrayRef> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let col = batch.column(i);
        let DataType::Dictionary(_, value_type) = field.data_type() else {
            new_columns.push(col.clone());
            new_fields.push(field.as_ref().clone());
            continue;
        };
        let dict = col.as_any_dictionary();
        let n_values = dict.values().len();
        // Keys as `Int32` with their validity, whatever width the file used.
        let keys = arrow::compute::cast(dict.keys(), &DataType::Int32)?;
        let keys = keys.as_primitive::<Int32Type>();
        let mut used = vec![false; n_values];
        for k in keys.iter().flatten() {
            match usize::try_from(k).ok().and_then(|k| used.get_mut(k)) {
                Some(slot) => *slot = true,
                None => {
                    return Err(ScxError::InvalidCatalog(format!(
                    "column '{}': dictionary key {k} out of bounds (dictionary length {n_values})",
                    field.name()
                )))
                }
            }
        }
        if used.iter().all(|&u| u) {
            new_columns.push(col.clone());
            new_fields.push(field.as_ref().clone());
            continue;
        }
        // Declared order of the survivors; an unreferenced entry gets no slot,
        // and no valid key can reach it.
        let mut old_to_new: Vec<u32> = vec![u32::MAX; n_values];
        let mut keep: Vec<u32> = Vec::new();
        for (old, &u) in used.iter().enumerate() {
            if u {
                old_to_new[old] = keep.len() as u32;
                keep.push(old as u32);
            }
        }
        let n_kept = keep.len();
        let values = arrow::compute::take(dict.values().as_ref(), &UInt32Array::from(keep), None)?;
        let new_keys = remap_dictionary_keys(keys, &old_to_new)?;
        let (encoded, final_dt) = finish_deduped_dictionary(new_keys, values, n_kept, value_type)?;
        new_columns.push(encoded);
        new_fields.push(
            Field::new(field.name(), final_dt, field.is_nullable())
                .with_metadata(field.metadata().clone()),
        );
    }
    let new_schema = Schema::new(new_fields).with_metadata(schema.metadata().clone());
    Ok(RecordBatch::try_new(Arc::new(new_schema), new_columns)?)
}

pub fn assemble_sharded_metadata(
    logical: &str,
    mut raw_batches: Vec<(u32, RecordBatch)>,
) -> Result<RecordBatch> {
    if raw_batches.is_empty() {
        return Err(ScxError::SectionNotFound(format!("{logical} (no shards)")));
    }
    raw_batches.sort_by_key(|(idx, _)| *idx);

    // Force every batch to the wide encoding before concat. The writer's
    // `write_arrow_ipc` always upcasts to LargeUtf8 / LargeBinary before
    // serialising, so per-shard reads typically come back wide already;
    // upcasting is a no-op in that case but covers shards whose individual
    // payload was narrow on disk. Concatenating on narrow offsets would
    // otherwise reproduce the original `Offset overflow error` once the
    // combined string payload exceeds `i32::MAX`.
    //
    // Also widen every categorical (dictionary) column's KEY type to Int32
    // before concat. Per-shard categoricals are written with a key sized to
    // each shard's *local* vocabulary (e.g. Int8 for ≤127 local categories);
    // `concat_batches` appends the per-shard dictionaries and offsets their
    // keys, so once the *combined* vocabulary across shards exceeds the narrow
    // key's range the key overflows with `Dictionary key bigger than the key
    // type`. Int32 keys can't overflow at any realistic scale (combined
    // pre-dedup dictionary length ≤ total rows). `unify_dictionary_columns`
    // narrows the key back to the minimal fit after deduplication.
    let batches: Vec<RecordBatch> = raw_batches
        .iter()
        .map(|(_, b)| {
            crate::arrow_compat::upcast_to_large_types(b)
                .and_then(|b| crate::arrow_compat::widen_dictionary_keys(&b))
        })
        .collect::<Result<_>>()?;

    // Reconcile columns that disagree on Dictionary-vs-plain encoding across
    // shards (an append writes obs categoricals as plain Utf8 while
    // `from_anndata` writes them as Dictionary, so a sharded axis can carry both
    // representations). `concat_batches` requires one shared schema, so encode
    // the plain shards' columns to Dictionary before concat. No-op when every
    // shard already agrees.
    let batches = crate::arrow_compat::reconcile_dictionary_representations(batches)?;

    // Give every shard's dictionary column one shared values array, so
    // `concat` appends keys instead of merging (and pruning) dictionaries — a
    // declared-but-unused category must survive whatever the row count.
    let batches = share_dictionary_values(batches)?;

    // Verify the shards form a contiguous, ordered cover by walking their
    // stamped metadata. Each shard's `n_rows_total` is the file's logical
    // row count *at the time that shard was written* — for single-pass
    // writes every shard carries the same value, but for append-grown files
    // older shards carry their smaller original stamps while later-appended
    // shards carry the bumped total. So the invariant is: `n_rows_total` is
    // monotonically non-decreasing across shards, and the **last shard's**
    // `n_rows_total` equals the cumulative row cover.
    let first_hdr = parse_shard_metadata(logical, &batches[0])?;
    if first_hdr.shard_idx != 0 {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: first shard has shard_idx={} (expected 0)",
            first_hdr.shard_idx
        )));
    }
    if first_hdr.row_start != 0 {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: first shard has row_start={} (expected 0)",
            first_hdr.row_start
        )));
    }
    let mut prev_n_rows_total = first_hdr.n_rows_total;
    let mut next_expected_row_start = first_hdr.n_shard_rows;
    for (i, batch) in batches.iter().enumerate().skip(1) {
        let hdr = parse_shard_metadata(logical, batch)?;
        let expected_idx = i as u32;
        if hdr.shard_idx != expected_idx {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: shard at position {i} has shard_idx={} (expected {expected_idx})",
                hdr.shard_idx
            )));
        }
        if hdr.n_rows_total < prev_n_rows_total {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: shard {i} has n_rows_total={} which contracts the prior \
                 shard's stamp of {prev_n_rows_total} — append-grown obs must stamp \
                 monotonically non-decreasing totals",
                hdr.n_rows_total
            )));
        }
        if hdr.row_start != next_expected_row_start {
            return Err(ScxError::InvalidCatalog(format!(
                "{logical}: shard {i} has row_start={} (expected {next_expected_row_start})",
                hdr.row_start
            )));
        }
        next_expected_row_start = next_expected_row_start.saturating_add(hdr.n_shard_rows);
        prev_n_rows_total = hdr.n_rows_total;
    }
    if next_expected_row_start != prev_n_rows_total {
        return Err(ScxError::InvalidCatalog(format!(
            "{logical}: shards cover {next_expected_row_start} rows but the last shard's \
             n_rows_total is {prev_n_rows_total}"
        )));
    }

    // Concat on the wide schema (every batch was upcast above), then
    // opportunistically narrow back to `Utf8`/`Binary` for columns whose
    // combined offsets still fit in `i32::MAX`.
    let wide_schema = batches[0].schema();
    let concatenated = arrow::compute::concat_batches(&wide_schema, batches.iter())?;
    // Arrow's `concat` appends each shard's dictionary verbatim without
    // deduplicating, so a categorical column that is `["batch1"]` in every
    // one of N shards comes back with a dictionary of `["batch1"; N]`. pandas
    // (`pyarrow.Table.to_pandas()`) rejects non-unique categories, so collapse
    // every dictionary column to a unified dictionary before narrowing.
    let unified = unify_dictionary_columns(&concatenated)?;
    let narrowed = crate::arrow_compat::downcast_large_types(&unified)?;

    // Strip the per-shard metadata (shard_idx / row_start / n_shard_rows)
    // from the merged batch's schema. Keep n_rows_total and any
    // payload-level metadata.
    let narrowed_schema = narrowed.schema();
    let mut clean_metadata = narrowed_schema.metadata().clone();
    clean_metadata.remove("shard_idx");
    clean_metadata.remove("row_start");
    clean_metadata.remove("n_shard_rows");
    let clean_schema = Arc::new(arrow::datatypes::Schema::new_with_metadata(
        narrowed_schema.fields().clone(),
        clean_metadata,
    ));
    Ok(RecordBatch::try_new(
        clean_schema,
        narrowed.columns().to_vec(),
    )?)
}

/// Assemble an **arbitrary, already-row-filtered** subset of metadata
/// shard batches into one logical batch.
///
/// Runs the same `upcast → widen-dict → reconcile → share-dict-values →
/// concat → unify-dict → downcast → strip-per-shard-metadata` pipeline as
/// [`assemble_sharded_metadata`], and like it keeps every declared dictionary
/// value; the engine decides whether a result is a row subset that should
/// carry only its surviving categories ([`prune_unused_dictionary_values`]) —
/// but WITHOUT the contiguous-cover validation and WITHOUT requiring the
/// per-shard `shard_idx` / `row_start` stamps — the input batches are an
/// arbitrary subset (any order, possibly empty), already filtered to the
/// rows the caller wants. Callers are responsible for passing the batches
/// in the final row order they want concatenated.
///
/// `template_schema` is only consulted when `batches` is empty, to build a
/// correctly-typed 0-row result; pass the schema a normal read of this
/// axis would produce (e.g. a single shard run through this same pipeline,
/// or a prior assembled batch's schema).
///
/// Used by the query engine to rebuild the filtered obs metadata for a
/// `filter_obs(...).collect()` result while only ever holding the matching
/// rows in memory — see `scx-engine/src/collect/execute.rs`.
pub fn assemble_filtered_metadata(
    template_schema: &Arc<arrow::datatypes::Schema>,
    batches: Vec<RecordBatch>,
) -> Result<RecordBatch> {
    if batches.is_empty() {
        return Ok(RecordBatch::new_empty(template_schema.clone()));
    }

    // Widen exactly as `assemble_sharded_metadata` does so concat can't
    // overflow narrow string offsets or per-shard dictionary key widths.
    let wide: Vec<RecordBatch> = batches
        .iter()
        .map(|b| {
            crate::arrow_compat::upcast_to_large_types(b)
                .and_then(|b| crate::arrow_compat::widen_dictionary_keys(&b))
        })
        .collect::<Result<_>>()?;

    // Reconcile Dictionary-vs-plain disagreement across the filtered shards
    // (see `assemble_sharded_metadata`) so concat can't reject a mixed-encoding
    // column produced by an append. No-op when shards already agree.
    let wide = crate::arrow_compat::reconcile_dictionary_representations(wide)?;
    // One shared values array per dictionary column (see
    // `assemble_sharded_metadata`), so `concat` cannot prune unused levels.
    let wide = share_dictionary_values(wide)?;

    let wide_schema = wide[0].schema();
    let concatenated = arrow::compute::concat_batches(&wide_schema, wide.iter())?;
    let unified = unify_dictionary_columns(&concatenated)?;
    let narrowed = crate::arrow_compat::downcast_large_types(&unified)?;

    // Strip the per-shard metadata keys so the result schema matches a
    // normal (full) read's assembled batch.
    let narrowed_schema = narrowed.schema();
    let mut clean_metadata = narrowed_schema.metadata().clone();
    clean_metadata.remove("shard_idx");
    clean_metadata.remove("row_start");
    clean_metadata.remove("n_shard_rows");
    let clean_schema = Arc::new(arrow::datatypes::Schema::new_with_metadata(
        narrowed_schema.fields().clone(),
        clean_metadata,
    ));
    Ok(RecordBatch::try_new(
        clean_schema,
        narrowed.columns().to_vec(),
    )?)
}

/// Drop the rows of an obs-axis batch that a deletion keep mask marks deleted
/// (`true` = keep), preserving row order.
///
/// The single implementation of a filter that four call sites had each
/// open-coded. Errors rather than truncating when `keep` and the batch disagree
/// on length: a mask sized against `header.n_obs` silently applied to a shard
/// would drop the wrong cells.
pub fn filter_batch_by_keep_mask(batch: &RecordBatch, keep: &[bool]) -> Result<RecordBatch> {
    if keep.len() != batch.num_rows() {
        return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
            "deletion keep mask has {} entries but the batch has {} rows",
            keep.len(),
            batch.num_rows()
        ))
        .into());
    }
    let mask = arrow::array::BooleanArray::from(keep.to_vec());
    Ok(arrow::compute::filter_record_batch(batch, &mask)?)
}

/// Expand a **live-row** (logical) obs-axis batch back onto the physical row
/// axis a deletion keep mask describes (`true` = live) — the forward
/// counterpart of [`filter_batch_by_keep_mask`]: `filter(scatter(b, keep),
/// keep)` is `b` again, up to nullability. Rows the mask marks deleted come
/// back **null in every column**: the caller never saw them, so it supplied
/// nothing for them, and a fabricated value would be a claim nobody made.
///
/// Every output field is forced nullable — a `take` through a null index
/// produces nulls whatever the source declared, and `RecordBatch::try_new`
/// rejects a null in a non-nullable field — while field metadata and the
/// schema metadata (the pandas envelope) are carried verbatim. A dictionary
/// column keeps its dictionary values (`take` moves keys, not values), so
/// declared categories and the `scx.categorical.ordered` stamp survive.
///
/// Errors when `batch.num_rows()` is not the mask's live count: scattering a
/// physical-length batch, or a live batch against another file's mask, would
/// land every row on the wrong cell. Also refuses a mask longer than
/// `u32::MAX` rows (the take index is `u32`).
pub fn scatter_batch_to_physical(batch: &RecordBatch, keep: &[bool]) -> Result<RecordBatch> {
    use arrow::datatypes::{Field, Schema};

    let n_live = keep.iter().filter(|k| **k).count();
    if n_live != batch.num_rows() {
        return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
            "deletion keep mask has {n_live} live rows (of {} physical) but the batch has {} rows",
            keep.len(),
            batch.num_rows()
        ))
        .into());
    }
    if keep.len() > u32::MAX as usize {
        return Err(arrow::error::ArrowError::InvalidArgumentError(format!(
            "cannot scatter onto {} physical rows: the take index is u32",
            keep.len()
        ))
        .into());
    }
    let mut next_live = 0u32;
    let take_idx: arrow::array::UInt32Array = keep
        .iter()
        .map(|&k| {
            if k {
                let i = next_live;
                next_live += 1;
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let columns = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c.as_ref(), &take_idx, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let fields: Vec<Field> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| {
            Field::new(f.name(), f.data_type().clone(), true).with_metadata(f.metadata().clone())
        })
        .collect();
    let schema = Arc::new(Schema::new(fields).with_metadata(batch.schema().metadata().clone()));
    Ok(RecordBatch::try_new_with_options(
        schema,
        columns,
        &arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(keep.len())),
    )?)
}

impl ScxReader {
    /// Read the Arrow IPC schema from a catalog entry.
    ///
    /// Stays in lockstep with [`Self::read_arrow_ipc`] under the
    /// opportunistic downcast in [`crate::arrow_compat`]: the
    /// canonical schema depends on whether columns' actual offsets fit
    /// back in `i32`, which can only be determined by inspecting the
    /// data. So:
    ///
    /// - **Fast path** (no `LargeUtf8` / `LargeBinary` /
    ///   `Dictionary(_, Large*)` on disk): return the IPC footer
    ///   schema directly. No data deserialization. ~KB of work.
    /// - **Slow path** (any wide type on disk): re-read the first
    ///   batch and run `downcast_large_types` so the returned schema
    ///   matches what `read_arrow_ipc` would produce — narrow types
    ///   when offsets fit, wide types when they overflow.
    fn read_arrow_ipc_schema(&self, entry: &FullCatalogEntry) -> Result<arrow::datatypes::Schema> {
        decode_arrow_ipc_schema(self.section_bytes(entry)?)
    }

    /// Read an Arrow IPC section from a catalog entry.
    ///
    /// Downcasts `LargeUtf8 → Utf8` / `LargeBinary → Binary` so callers
    /// always see canonical narrow types regardless of the on-disk
    /// encoding (see [`crate::arrow_compat`]).
    fn read_arrow_ipc(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        let batch = self.read_arrow_ipc_raw(entry)?;
        crate::arrow_compat::downcast_large_types(&batch)
    }

    /// Decode a single Arrow IPC entry **without** the wide→narrow
    /// downcast. The writer always upcasts to `LargeUtf8`/`LargeBinary`
    /// before serialising (so the bytes-on-disk are typically wide), but
    /// columns whose payload fits in narrow offsets may still come back
    /// downcast-eligible. This helper preserves the on-disk encoding so
    /// callers that need to concatenate batches across shards can defer
    /// the narrow choice until after [`arrow::compute::concat_batches`]
    /// — concatenating on narrow offsets reproduces the original
    /// `Offset overflow error` once the combined per-column string
    /// payload exceeds `i32::MAX` (the same failure mode the streaming
    /// merge-write path eliminated).
    fn read_arrow_ipc_raw(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        let mut batches = reader.into_iter();
        batches
            .next()
            .ok_or_else(|| {
                ScxError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Arrow IPC file contains no batches",
                ))
            })?
            .map_err(ScxError::Arrow)
    }

    /// Phase 3b: decode a single Arrow IPC dense-mapping section by
    /// catalog entry, applying the same wide→narrow downcast as the
    /// whole-batch readers. Public counterpart to
    /// [`Self::read_shard_from_entry`] (which targets encoded CSR
    /// shards): used by the streaming merge path to walk
    /// `Obsm/VarmEmbeddingShard` (and their legacy single-section
    /// counterparts) one shard at a time without going through
    /// `read_obsm` / `read_varm` (which reassemble the full mapping).
    pub fn read_dense_mapping_entry(&self, entry: &FullCatalogEntry) -> Result<RecordBatch> {
        self.read_arrow_ipc(entry)
    }

    /// Resolve the per-shard physical layout of a row-sharded Arrow IPC
    /// mapping — `obsm/<name>` (dense) or `obsp/<name>` (COO) — for the
    /// backed row-range readers ([`crate::BackedDenseReader`],
    /// [`crate::BackedPairwiseReader`]).
    ///
    /// Unlike CSR shards, these catalog entries carry no `stats` block,
    /// so the per-shard row ranges live only in each shard's Arrow
    /// schema metadata (`row_start` / `n_shard_rows` / `n_rows_total`,
    /// stamped by `writer::stamp_dense_shard_meta`). We read each
    /// shard's IPC **footer schema only** (no batch deserialisation) and
    /// validate a contiguous, ordered cover with the same invariant as
    /// [`assemble_sharded_metadata`].
    ///
    /// Falls back to the legacy single-section layout (`single_type`)
    /// treated as one shard spanning `[0, n_rows)`. How that path learns its
    /// row count is [`LegacyRowCount`]'s decision, and the two differ in cost
    /// as well as in meaning: `BatchRows` must deserialise the one batch,
    /// because the count IS `batch.num_rows()`; `SchemaNRows` reads the IPC
    /// footer schema only, which matters because the batch it would otherwise
    /// decode-and-drop is every triple in the graph.
    pub(crate) fn row_sharded_mapping_layout(
        &self,
        prefix: &str,
        name: &str,
        shard_type: SectionType,
        single_type: SectionType,
        legacy_rows: LegacyRowCount,
    ) -> Result<MappingLayout> {
        let shard_name_prefix = format!("{prefix}/{name}_shard_");
        let logical = format!("{prefix}/{name}");

        let mut shards: Vec<(u32, &FullCatalogEntry)> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(&shard_name_prefix))
            .filter_map(|e| {
                let suffix = e.name.strip_prefix(&shard_name_prefix)?;
                let idx: u32 = suffix.parse().ok()?;
                Some((idx, e))
            })
            .collect();

        if !shards.is_empty() {
            shards.sort_by_key(|(idx, _)| *idx);
            let mut entries: Vec<MappingShardLayoutEntry> = Vec::with_capacity(shards.len());
            let mut n_cols = 0usize;
            let mut matrix_n_cols: Option<u64> = None;
            let mut dtype = arrow::datatypes::DataType::Float32;
            let mut fields: arrow::datatypes::Fields = Default::default();
            let mut prev_n_rows_total = 0u64;
            let mut next_expected_row_start = 0u64;

            for (i, (idx, entry)) in shards.iter().enumerate() {
                let schema = self.read_arrow_ipc_schema_physical(entry)?;
                let hdr = parse_shard_metadata_md(&logical, schema.metadata())?;
                let expected_idx = i as u32;
                if hdr.shard_idx != expected_idx {
                    return Err(ScxError::InvalidCatalog(format!(
                        "{logical}: shard at position {i} has shard_idx={} (expected {expected_idx})",
                        hdr.shard_idx
                    )));
                }
                if i == 0 {
                    if hdr.row_start != 0 {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: first shard has row_start={} (expected 0)",
                            hdr.row_start
                        )));
                    }
                    n_cols = schema.fields().len();
                    if n_cols > 0 {
                        dtype = schema.field(0).data_type().clone();
                    }
                    fields = schema.fields().clone();
                    matrix_n_cols = schema
                        .metadata()
                        .get("n_cols")
                        .and_then(|v| v.parse::<u64>().ok());
                    prev_n_rows_total = hdr.n_rows_total;
                    next_expected_row_start = hdr.n_shard_rows;
                } else {
                    if hdr.n_rows_total < prev_n_rows_total {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: shard {i} has n_rows_total={} which contracts the prior \
                             shard's stamp of {prev_n_rows_total}",
                            hdr.n_rows_total
                        )));
                    }
                    if hdr.row_start != next_expected_row_start {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: shard {i} has row_start={} (expected {next_expected_row_start})",
                            hdr.row_start
                        )));
                    }
                    next_expected_row_start =
                        next_expected_row_start.saturating_add(hdr.n_shard_rows);
                    prev_n_rows_total = hdr.n_rows_total;
                    // The column extent is taken from shard 0 and then used for
                    // the whole mapping, so a later shard that disagrees about
                    // it would be read against the wrong axis. Checked only
                    // where shard 0 had one — a dense `obsm` stamps no
                    // `n_cols`, and absence there is not a disagreement.
                    if let Some(expected) = matrix_n_cols {
                        let here = schema
                            .metadata()
                            .get("n_cols")
                            .and_then(|v| v.parse::<u64>().ok());
                        if here != Some(expected) {
                            return Err(ScxError::InvalidCatalog(format!(
                                "{logical}: shard {i} declares n_cols={here:?} but shard 0 \
                                 declares {expected}"
                            )));
                        }
                    }
                    if schema.fields() != &fields {
                        return Err(ScxError::InvalidCatalog(format!(
                            "{logical}: shard {i}'s column schema differs from shard 0's"
                        )));
                    }
                }
                let _ = idx;
                entries.push(MappingShardLayoutEntry {
                    offset: entry.offset,
                    length: entry.length,
                    section_type: entry.section_type,
                    modality_id: entry.modality_id,
                    row_start: hdr.row_start,
                    n_shard_rows: hdr.n_shard_rows,
                });
            }
            if next_expected_row_start != prev_n_rows_total {
                return Err(ScxError::InvalidCatalog(format!(
                    "{logical}: shards cover {next_expected_row_start} rows but the last shard's \
                     n_rows_total is {prev_n_rows_total}"
                )));
            }
            return Ok(MappingLayout {
                entries,
                n_rows: prev_n_rows_total,
                n_cols,
                dtype,
                fields,
                matrix_n_cols,
            });
        }

        // Legacy single section — one batch, no shard metadata.
        let entry = self
            .full_catalog
            .get(&logical)
            .filter(|e| e.section_type == single_type)
            .ok_or_else(|| ScxError::SectionNotFound(logical.clone()))?;

        // The pairwise arm reads the IPC **footer schema** and stops there.
        // Everything it needs — the column fields, the dtype, and the `n_rows`
        // / `n_cols` stamps — is in the schema, while the payload is every COO
        // triple in the graph: on an atlas-scale unsharded obsp that is a
        // multi-gigabyte decode, performed here only to be dropped and then
        // performed again by the first `read_rows_range`. The dense arm cannot
        // take this path: its row count IS `batch.num_rows()`, which the footer
        // does not carry.
        let (n_rows, n_cols, dtype, fields, matrix_n_cols) = match legacy_rows {
            LegacyRowCount::SchemaNRows => {
                let schema = self.read_arrow_ipc_schema_physical(entry)?;
                let md = schema.metadata();
                let n_rows = md
                    .get("n_rows")
                    .and_then(|v| v.parse::<u64>().ok())
                    .ok_or_else(|| {
                        ScxError::InvalidCatalog(format!(
                            "{logical}: legacy pairwise section has no parseable 'n_rows' schema \
                             metadata, so its logical row count is unknown"
                        ))
                    })?;
                let n_cols = schema.fields().len();
                let dtype = if n_cols > 0 {
                    schema.field(0).data_type().clone()
                } else {
                    arrow::datatypes::DataType::Float32
                };
                let matrix_n_cols = md.get("n_cols").and_then(|v| v.parse::<u64>().ok());
                (
                    n_rows,
                    n_cols,
                    dtype,
                    schema.fields().clone(),
                    matrix_n_cols,
                )
            }
            LegacyRowCount::BatchRows => {
                let batch = self.read_arrow_ipc(entry)?;
                let n_cols = batch.num_columns();
                let dtype = if n_cols > 0 {
                    batch.column(0).data_type().clone()
                } else {
                    arrow::datatypes::DataType::Float32
                };
                let matrix_n_cols = batch
                    .schema_ref()
                    .metadata()
                    .get("n_cols")
                    .and_then(|v| v.parse::<u64>().ok());
                (
                    batch.num_rows() as u64,
                    n_cols,
                    dtype,
                    batch.schema_ref().fields().clone(),
                    matrix_n_cols,
                )
            }
        };
        Ok(MappingLayout {
            entries: vec![MappingShardLayoutEntry {
                offset: entry.offset,
                length: entry.length,
                section_type: entry.section_type,
                modality_id: entry.modality_id,
                row_start: 0,
                n_shard_rows: n_rows,
            }],
            n_rows,
            n_cols,
            dtype,
            fields,
            matrix_n_cols,
        })
    }

    /// Read the obs schema. Uses the Arrow IPC footer fast path, falling
    /// back to a per-column wide-vs-narrow refinement (via
    /// [`crate::arrow_compat::downcast_large_types`]) when any column on
    /// disk is `LargeUtf8` / `LargeBinary` / `Dictionary(_, Large*)`.
    /// For atlas-scale obs that legitimately remain wide on disk, the
    /// refinement deserialises the first batch — for sharded files this
    /// is bounded to one shard rather than the whole obs table, but is
    /// still O(MB). Use [`Self::read_obs_schema_physical`] (no
    /// deserialisation) or [`Self::read_obs_schema_logical_lossy`]
    /// (unconditional schema-level narrowing) for the cheap paths.
    pub fn read_obs_schema(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_obs_section_entry()?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Read the var schema. Mirror of [`Self::read_obs_schema`].
    pub fn read_var_schema(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_var_section_entry()?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Return the obs schema **exactly as stored on disk** — including
    /// any `LargeUtf8` / `LargeBinary` columns left wide for >2 GB
    /// payloads. Pure Arrow IPC footer read; no batch deserialisation.
    /// Constant cost regardless of obs size.
    ///
    /// Use this when your code can handle wide types and you want a
    /// faithful picture of what the writer emitted. Pair with
    /// [`Self::read_obs_schema_logical_lossy`] if you'd rather always
    /// see narrow types and don't mind the lossy conversion.
    pub fn read_obs_schema_physical(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_obs_section_entry()?;
        self.read_arrow_ipc_schema_physical(entry)
    }

    /// Read var schema as on disk. Mirror of
    /// [`Self::read_obs_schema_physical`].
    pub fn read_var_schema_physical(&self) -> Result<arrow::datatypes::Schema> {
        let entry = self.first_var_section_entry()?;
        self.read_arrow_ipc_schema_physical(entry)
    }

    /// Return the obs schema with `LargeUtf8 → Utf8` / `LargeBinary →
    /// Binary` (and `Dictionary` variants) **unconditionally narrowed**
    /// at the schema level. Pure Arrow IPC footer read; no batch
    /// deserialisation. Lossy in the technical sense — a column the
    /// reader reports as `Utf8` may, on the data path, still come back
    /// as `LargeUtf8` if its actual offsets overflow `i32::MAX`. The
    /// trade-off is constant-cost schema reads for predicate parsing
    /// and validation paths that prefer the historical narrow types.
    pub fn read_obs_schema_logical_lossy(&self) -> Result<arrow::datatypes::Schema> {
        let physical = self.read_obs_schema_physical()?;
        Ok(crate::arrow_compat::downcast_large_types_schema(&physical))
    }

    /// Lossy logical schema for var. Mirror of
    /// [`Self::read_obs_schema_logical_lossy`].
    pub fn read_var_schema_logical_lossy(&self) -> Result<arrow::datatypes::Schema> {
        let physical = self.read_var_schema_physical()?;
        Ok(crate::arrow_compat::downcast_large_types_schema(&physical))
    }

    /// Resolve the first obs section catalog entry: shard 0 if obs is
    /// sharded, else the legacy single section. Used by both schema
    /// APIs and the assembled-batch fallback.
    fn first_obs_section_entry(&self) -> Result<&FullCatalogEntry> {
        if self.obs_metadata_shard_count() > 0 {
            let key = "obs_metadata/shard_0";
            self.full_catalog
                .get(key)
                .ok_or_else(|| ScxError::SectionNotFound(key.to_string()))
        } else {
            self.full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))
        }
    }

    /// Mirror of [`Self::first_obs_section_entry`] for var.
    fn first_var_section_entry(&self) -> Result<&FullCatalogEntry> {
        if self.var_metadata_shard_count() > 0 {
            let key = "var_metadata/shard_0";
            self.full_catalog
                .get(key)
                .ok_or_else(|| ScxError::SectionNotFound(key.to_string()))
        } else {
            self.full_catalog
                .get("var")
                .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))
        }
    }

    /// Pure Arrow IPC footer read: no batch deserialisation, no wide-vs-
    /// narrow refinement. Returns whatever the writer recorded in the
    /// footer schema. Counterpart to [`Self::read_arrow_ipc_schema`]
    /// which does the conditional first-batch re-read.
    fn read_arrow_ipc_schema_physical(
        &self,
        entry: &FullCatalogEntry,
    ) -> Result<arrow::datatypes::Schema> {
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReader::try_new(cursor, None)?;
        Ok(reader.schema().as_ref().clone())
    }

    /// Read the obs (observation) metadata as an Arrow RecordBatch.
    ///
    /// Transparently handles both file layouts: returns the single
    /// [`SectionType::ObsMetadata`] section on legacy files, or
    /// reassembles every [`SectionType::ObsMetadataShard`] section in
    /// `shard_idx` order on Phase 2 sharded files (with a contiguous-
    /// cover verification — any gap, duplicate, or shrinking
    /// `n_rows_total` is rejected as [`ScxError::InvalidCatalog`]).
    ///
    /// **Memory cost:** allocates a buffer sized to the entire logical
    /// obs table. For atlas-scale files (tens of GB) this can dominate
    /// peak RSS. Prefer the streaming [`Self::obs_shards`] iterator or
    /// per-shard [`Self::read_obs_shard`] when you can process the
    /// table in chunks.
    pub fn read_obs(&self) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts.read_obs.fetch_add(1, Ordering::Relaxed);
        if self.obs_metadata_shard_count() > 0 {
            self.read_sharded_layout_by_prefix(
                "obs_metadata/shard_",
                "obs_metadata",
                SectionType::ObsMetadataShard,
            )?
            .ok_or_else(|| ScxError::SectionNotFound("obs_metadata/shard_*".to_string()))
        } else {
            let entry = self
                .full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
            self.read_arrow_ipc(entry)
        }
    }

    /// Read only the named obs columns, assembled in global obs row order, for
    /// computing a sort/order over them. Per-row **values** match a full
    /// [`Self::read_obs`]; string columns are returned **dictionary-encoded**
    /// (see below), so the dtype may differ from `read_obs` for columns stored
    /// plain on disk — fine for ordering (the comparator keys off values).
    ///
    /// On sharded files this projects each [`SectionType::ObsMetadataShard`] to
    /// `col_names`, **compacts** each projected shard (see [`compact_key_shard`]),
    /// then runs the shared [`assemble_sharded_metadata`] pipeline. Compaction
    /// is load-bearing: Arrow IPC column projection returns the projected column
    /// as a zero-copy slice into the shard's full message body, so a retained
    /// projected batch keeps *every* un-projected column resident — accumulating
    /// all shards would hold the entire obs table (the projection saving
    /// nothing). Compaction rebuilds each key column into fresh compact buffers
    /// (dictionary-encoding strings, which also collapses categoricals stored
    /// plain), so peak RSS is the small key data, not the whole obs.
    pub fn read_obs_keys(&self, col_names: &[String]) -> Result<RecordBatch> {
        let physical = self.read_obs_schema_physical()?;
        let projection: Vec<usize> = col_names
            .iter()
            .map(|name| {
                physical
                    .index_of(name)
                    .map_err(|_| ScxError::SectionNotFound(format!("obs column '{name}'")))
            })
            .collect::<Result<_>>()?;

        if self.obs_metadata_shard_count() > 0 {
            let mut shards: Vec<(u32, &FullCatalogEntry)> = self
                .full_catalog
                .entries
                .iter()
                .filter(|e| {
                    e.section_type == SectionType::ObsMetadataShard
                        && e.name.starts_with("obs_metadata/shard_")
                })
                .filter_map(|e| {
                    let suffix = e.name.strip_prefix("obs_metadata/shard_")?;
                    let idx: u32 = suffix.parse().ok()?;
                    Some((idx, e))
                })
                .collect();
            shards.sort_by_key(|(idx, _)| *idx);
            // Project + **compact** each shard. Arrow IPC column projection
            // returns the projected column as a zero-copy slice into the shard's
            // full message body, so the body of every (un-projected) column
            // stays resident as long as the batch is held — accumulating all
            // shards would retain the entire obs table (the projection saves
            // nothing). `compact_key_shard` rebuilds each key column into fresh,
            // compact buffers (dictionary-encoding plain string columns, which
            // also collapses categoricals stored plain), dropping the body
            // alias so only the small key data is retained.
            let mut raw_batches: Vec<(u32, RecordBatch)> = Vec::with_capacity(shards.len());
            for (idx, _) in shards.iter() {
                let projected = self.read_obs_shard_projected(*idx, &projection)?;
                raw_batches.push((*idx, compact_key_shard(&projected)?));
            }
            assemble_sharded_metadata("obs_metadata", raw_batches)
        } else {
            // Legacy single-section obs (non-atlas): project the one
            // section and narrow, matching read_obs()'s downcast.
            let entry = self
                .full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
            let slice = self.section_bytes(entry)?;
            let cursor = Cursor::new(slice);
            let reader = arrow::ipc::reader::FileReaderBuilder::new()
                .with_projection(projection)
                .build(cursor)?;
            let batch = reader
                .into_iter()
                .next()
                .ok_or_else(|| {
                    ScxError::Io(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Arrow IPC file contains no batches",
                    ))
                })?
                .map_err(ScxError::Arrow)?;
            crate::arrow_compat::downcast_large_types(&batch)
        }
    }

    /// Distinct values of a single **string/categorical** obs column, computed
    /// shard-by-shard without assembling the full obs table.
    ///
    /// Routes each shard's projected column through [`DistinctAccumulator`]:
    /// `Dictionary` columns scan only the per-shard dictionary catalog (rows
    /// never decoded), plain `Utf8`/`LargeUtf8` columns scan the value buffer.
    /// X is never touched. Nulls are excluded. See [`DistinctAccumulator`] for
    /// the dictionary-superset, `limit`, and `sort` semantics.
    ///
    /// Returns `(values, has_more)`. Errors with
    /// [`ScxError::UnsupportedColumnType`] for non-string columns and
    /// [`ScxError::SectionNotFound`] for an unknown column name.
    pub fn distinct_obs_values(
        &self,
        col: &str,
        limit: Option<usize>,
        sort: bool,
    ) -> Result<(Vec<String>, bool)> {
        let physical = self.read_obs_schema_physical()?;
        let col_idx = physical
            .index_of(col)
            .map_err(|_| ScxError::SectionNotFound(format!("obs column '{col}'")))?;
        let projection = [col_idx];
        let mut acc = DistinctAccumulator::new(col, limit, sort);

        if self.obs_metadata_shard_count() > 0 {
            // Iterate shards in index order; stop as soon as the accumulator
            // has its answer (first-N overflow already observed).
            let mut shard_indices: Vec<u32> = self
                .full_catalog
                .entries
                .iter()
                .filter(|e| {
                    e.section_type == SectionType::ObsMetadataShard
                        && e.name.starts_with("obs_metadata/shard_")
                })
                .filter_map(|e| e.name.strip_prefix("obs_metadata/shard_")?.parse().ok())
                .collect();
            shard_indices.sort_unstable();
            for idx in shard_indices {
                let batch = self.read_obs_shard_projected(idx, &projection)?;
                acc.push(batch.column(0))?;
                if acc.done() {
                    break;
                }
            }
        } else {
            let entry = self
                .full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
            let slice = self.section_bytes(entry)?;
            let cursor = Cursor::new(slice);
            let reader = arrow::ipc::reader::FileReaderBuilder::new()
                .with_projection(vec![col_idx])
                .build(cursor)?;
            for batch in reader {
                let batch = batch.map_err(ScxError::Arrow)?;
                acc.push(batch.column(0))?;
                if acc.done() {
                    break;
                }
            }
        }
        Ok(acc.finish())
    }

    /// `(codes, categories)` for a single **string/categorical** obs column,
    /// computed shard-by-shard without assembling the full obs table.
    ///
    /// The numpy-level accessor every model's vocabulary/one-hot setup actually
    /// wants: `codes[i]` is the global category code of obs row `i`, `-1` for
    /// null (pandas convention), and `categories[code]` is its string value. X is
    /// never touched.
    ///
    /// Unlike [`Self::read_obs_keys`] this never concatenates the column, so peak
    /// memory is `n_obs × 4 B` plus the vocabulary rather than a full
    /// materialised Arrow column. Accepts both `Dictionary(_, Utf8|LargeUtf8)`
    /// (as `from_anndata` writes) and plain `Utf8`/`LargeUtf8` (as `append`
    /// writes), including a file that carries both across its shards. See
    /// [`GlobalCategoryAccum`] for the full ordering / null / unreferenced-level
    /// semantics.
    ///
    /// # Physical row space
    ///
    /// `codes` is indexed in the **physical** obs row space — `codes.len()`
    /// equals the header's `n_obs`, *not* the logical post-deletion count. On a
    /// file with deletion vectors the two differ, and indexing `codes` with a
    /// logical row id addresses the wrong cell: a correctly *shaped* array of
    /// wrong rows, which is the worst failure mode available. Callers that work
    /// in logical space must either filter with
    /// [`Self::deletion_keep_mask`] first, or refuse such files outright (as
    /// state3's `_ScxBackend` does). This matches [`Self::read_obs`] /
    /// [`Self::read_obs_keys`], which are physical for the same reason.
    ///
    /// Errors with [`ScxError::UnsupportedColumnType`] for non-string columns and
    /// [`ScxError::SectionNotFound`] for an unknown column name.
    pub fn obs_categorical(&self, col: &str) -> Result<(Vec<i32>, Vec<String>)> {
        Ok(self
            .obs_categorical_many(std::slice::from_ref(&col.to_string()))?
            .pop()
            .expect("one column in ⇒ one column out"))
    }

    /// [`Self::obs_categorical`] for several columns in **one** shard pass.
    ///
    /// N columns cost one projected read per shard rather than N, which is the
    /// difference that matters at manifest scale: a catalog build resolving four
    /// covariate columns over 26k files does 26k reads, not 104k.
    ///
    /// **Result order is guaranteed to match `cols` positionally** — `out[i]`
    /// is always `cols[i]`, independent of the columns' on-disk schema order.
    /// Callers index the result by position, so this is contractual, not
    /// incidental; pinned by
    /// `test_obs_categorical_many_preserves_request_order_on_legacy_obs`.
    pub fn obs_categorical_many(&self, cols: &[String]) -> Result<Vec<(Vec<i32>, Vec<String>)>> {
        if cols.is_empty() {
            return Ok(Vec::new());
        }
        let physical = self.read_obs_schema_physical()?;
        let projection: Vec<usize> = cols
            .iter()
            .map(|name| {
                physical
                    .index_of(name)
                    .map_err(|_| ScxError::SectionNotFound(format!("obs column '{name}'")))
            })
            .collect::<Result<_>>()?;

        let n_obs_hint = self.header.n_obs as usize;
        let mut accs: Vec<GlobalCategoryAccum> = cols
            .iter()
            .map(|c| GlobalCategoryAccum::new(c.clone(), n_obs_hint))
            .collect();

        if self.obs_metadata_shard_count() > 0 {
            // Shard order is load-bearing: the accumulators append codes, so an
            // out-of-order shard would misalign every subsequent code against its
            // obs row — a correctly *shaped* result with wrong rows.
            let mut shard_indices: Vec<u32> = self
                .full_catalog
                .entries
                .iter()
                .filter(|e| {
                    e.section_type == SectionType::ObsMetadataShard
                        && e.name.starts_with("obs_metadata/shard_")
                })
                .filter_map(|e| e.name.strip_prefix("obs_metadata/shard_")?.parse().ok())
                .collect();
            shard_indices.sort_unstable();
            for idx in shard_indices {
                let batch = self.read_obs_shard_projected(idx, &projection)?;
                for (i, acc) in accs.iter_mut().enumerate() {
                    acc.push(batch.column(i))?;
                }
            }
        } else {
            let entry = self
                .full_catalog
                .get("obs")
                .ok_or_else(|| ScxError::SectionNotFound("obs".to_string()))?;
            let slice = self.section_bytes(entry)?;
            let cursor = Cursor::new(slice);
            let reader = arrow::ipc::reader::FileReaderBuilder::new()
                .with_projection(projection)
                .build(cursor)?;
            for batch in reader {
                let batch = batch.map_err(ScxError::Arrow)?;
                for (i, acc) in accs.iter_mut().enumerate() {
                    acc.push(batch.column(i))?;
                }
            }
        }

        // A shard-cover gap would silently truncate `codes` and misalign it
        // against obs — cheap to catch here, expensive to debug downstream.
        for (acc, col) in accs.iter().zip(cols.iter()) {
            if acc.n_rows() != n_obs_hint {
                return Err(ScxError::InvalidCatalog(format!(
                    "obs_categorical('{col}') folded {} rows but the header \
                     declares n_obs = {n_obs_hint}; obs shards do not cover the axis",
                    acc.n_rows()
                )));
            }
        }
        Ok(accs.into_iter().map(|a| a.finish()).collect())
    }

    /// Read the var (variable/gene) metadata as an Arrow RecordBatch.
    /// Mirror of [`Self::read_obs`] for the var axis — same dual-layout
    /// handling, same memory cost caveat, and same streaming
    /// alternatives ([`Self::var_shards`], [`Self::read_var_shard`]).
    pub fn read_var(&self) -> Result<RecordBatch> {
        if self.var_metadata_shard_count() > 0 {
            self.read_sharded_layout_by_prefix(
                "var_metadata/shard_",
                "var_metadata",
                SectionType::VarMetadataShard,
            )?
            .ok_or_else(|| ScxError::SectionNotFound("var_metadata/shard_*".to_string()))
        } else {
            let entry = self
                .full_catalog
                .get("var")
                .ok_or_else(|| ScxError::SectionNotFound("var".to_string()))?;
            self.read_arrow_ipc(entry)
        }
    }

    /// Number of [`SectionType::ObsMetadataShard`] sections in the
    /// catalog. Pure catalog scan — no payload read. Zero on legacy
    /// single-section ([`SectionType::ObsMetadata`]) files.
    pub fn obs_metadata_shard_count(&self) -> usize {
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::ObsMetadataShard)
            .count()
    }

    /// Number of [`SectionType::VarMetadataShard`] sections in the
    /// catalog. Mirror of [`Self::obs_metadata_shard_count`].
    pub fn var_metadata_shard_count(&self) -> usize {
        self.full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == SectionType::VarMetadataShard)
            .count()
    }

    /// Read one row-shard of obs metadata by index. Returns the on-disk
    /// `RecordBatch` with its stamped shard schema metadata
    /// (`shard_idx`, `row_start`, `n_shard_rows`, `n_rows_total`)
    /// preserved — callers may consult those fields directly.
    pub fn read_obs_shard(&self, shard_idx: u32) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_obs_shard
            .fetch_add(1, Ordering::Relaxed);
        let key = format!("obs_metadata/shard_{shard_idx}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or(ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read one obs row-shard projected to `projection` (column indices
    /// into the on-disk obs schema), **without** the wide→narrow
    /// downcast — the raw, column-projected counterpart to
    /// [`Self::read_obs_shard`] used by [`Self::read_obs_keys`]. Arrow IPC
    /// column projection skips decoding the unselected columns. The
    /// shard's stamped schema metadata (`shard_idx` / `row_start` /
    /// `n_shard_rows` / `n_rows_total`) is preserved by `Schema::project`,
    /// so the result feeds [`assemble_sharded_metadata`] unchanged.
    pub fn read_obs_shard_projected(
        &self,
        shard_idx: u32,
        projection: &[usize],
    ) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_obs_shard_projected
            .fetch_add(1, Ordering::Relaxed);
        let key = format!("obs_metadata/shard_{shard_idx}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or(ScxError::SectionNotFound(key))?;
        let slice = self.section_bytes(entry)?;
        let cursor = Cursor::new(slice);
        let reader = arrow::ipc::reader::FileReaderBuilder::new()
            .with_projection(projection.to_vec())
            .build(cursor)?;
        reader
            .into_iter()
            .next()
            .ok_or_else(|| {
                ScxError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Arrow IPC file contains no batches",
                ))
            })?
            .map_err(ScxError::Arrow)
    }

    /// Read one row-shard of var metadata. Mirror of
    /// [`Self::read_obs_shard`].
    pub fn read_var_shard(&self, shard_idx: u32) -> Result<RecordBatch> {
        let key = format!("var_metadata/shard_{shard_idx}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or(ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Iterate obs metadata shards in `shard_idx` order, yielding one
    /// `RecordBatch` per shard. The iterator never materialises more
    /// than one shard at a time, so peak memory is bounded by the
    /// largest single shard regardless of the logical obs size.
    ///
    /// Returns an empty iterator on legacy single-section files; call
    /// [`Self::read_obs`] for those (and for any caller that genuinely
    /// needs the full obs table).
    pub fn obs_shards(&self) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        self.metadata_shards_iter(SectionType::ObsMetadataShard, "obs_metadata/shard_")
    }

    /// Iterate var metadata shards in `shard_idx` order. Mirror of
    /// [`Self::obs_shards`].
    pub fn var_shards(&self) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        self.metadata_shards_iter(SectionType::VarMetadataShard, "var_metadata/shard_")
    }

    /// Shared iterator builder for [`Self::obs_shards`] /
    /// [`Self::var_shards`]. Materialises only the sorted catalog
    /// entries up front (cheap pointer slice); each shard's payload is
    /// fetched lazily as the consumer advances the iterator.
    fn metadata_shards_iter(
        &self,
        shard_type: SectionType,
        name_prefix: &'static str,
    ) -> impl Iterator<Item = Result<RecordBatch>> + '_ {
        let mut entries: Vec<(u32, &FullCatalogEntry)> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(name_prefix))
            .filter_map(|e| {
                let suffix = e.name.strip_prefix(name_prefix)?;
                let idx: u32 = suffix.parse().ok()?;
                Some((idx, e))
            })
            .collect();
        entries.sort_by_key(|(idx, _)| *idx);
        entries
            .into_iter()
            .map(move |(_, e)| self.read_arrow_ipc(e))
    }

    /// Read a named obsm embedding as an Arrow RecordBatch.
    ///
    /// Prefers the sharded on-disk layout (`obsm/<name>_shard_<idx>`,
    /// section type [`SectionType::ObsmEmbeddingShard`]) and falls back
    /// to the legacy single-section [`SectionType::ObsmEmbedding`]
    /// layout for files written before sharding was introduced.
    pub fn read_obsm(&self, name: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts.read_obsm.fetch_add(1, Ordering::Relaxed);
        if let Some(batch) =
            self.read_sharded_layout("obsm", name, SectionType::ObsmEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("obsm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all obsm embeddings, keyed by name.
    pub fn read_all_obsm(&self) -> Result<HashMap<String, RecordBatch>> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_all_obsm
            .fetch_add(1, Ordering::Relaxed);
        self.read_all_sharded_or_single(
            "obsm",
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        )
    }

    /// Read a named varm embedding as an Arrow RecordBatch.
    ///
    /// See [`Self::read_obsm`] for the sharded / legacy layout handling.
    pub fn read_varm(&self, name: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts.read_varm.fetch_add(1, Ordering::Relaxed);
        if let Some(batch) =
            self.read_sharded_layout("varm", name, SectionType::VarmEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("varm/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all varm embeddings, keyed by name.
    pub fn read_all_varm(&self) -> Result<HashMap<String, RecordBatch>> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_all_varm
            .fetch_add(1, Ordering::Relaxed);
        self.read_all_sharded_or_single(
            "varm",
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        )
    }

    /// Read a named obsp pairwise sparse matrix (COO format).
    ///
    /// Lazy counterpart to [`Self::read_all_obsp`]: used by
    /// `pyscx::lazy_mapping::ScxLazyPairwiseMapping` so `to_anndata()`
    /// can defer obsp materialization until the consumer actually
    /// accesses `ad.obsp[name]`. Handles both sharded and legacy
    /// single-section layouts; see [`Self::read_obsm`].
    pub fn read_obsp(&self, name: &str) -> Result<RecordBatch> {
        if let Some(batch) =
            self.read_sharded_layout("obsp", name, SectionType::ObspEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("obsp/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all obsp pairwise sparse matrices (COO format), keyed by name.
    pub fn read_all_obsp(&self) -> Result<HashMap<String, RecordBatch>> {
        self.read_all_sharded_or_single(
            "obsp",
            SectionType::ObspEmbedding,
            SectionType::ObspEmbeddingShard,
        )
    }

    /// Read a named varp pairwise sparse matrix (COO format).
    ///
    /// Lazy counterpart to [`Self::read_all_varp`]; see [`Self::read_obsp`].
    pub fn read_varp(&self, name: &str) -> Result<RecordBatch> {
        if let Some(batch) =
            self.read_sharded_layout("varp", name, SectionType::VarpEmbeddingShard)?
        {
            return Ok(batch);
        }
        let key = format!("varp/{name}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read all varp pairwise sparse matrices (COO format), keyed by name.
    pub fn read_all_varp(&self) -> Result<HashMap<String, RecordBatch>> {
        self.read_all_sharded_or_single(
            "varp",
            SectionType::VarpEmbedding,
            SectionType::VarpEmbeddingShard,
        )
    }

    /// List the keys (entry names with their section prefix stripped) of
    /// every `obsp/*` catalog entry. Pure catalog scan — no section
    /// bytes are read. Used by `pyscx::lazy_mapping` to pre-populate the
    /// key set of the lazy obsp wrapper at `to_anndata()` time. Includes
    /// both sharded (`obsp/<name>_shard_<idx>`) and legacy single-section
    /// keys, deduplicating shard names back to their logical key.
    pub fn list_obsp(&self) -> Vec<String> {
        self.list_logical_names(
            "obsp",
            SectionType::ObspEmbedding,
            SectionType::ObspEmbeddingShard,
        )
    }

    /// List the keys of every `varp/*` catalog entry. See [`Self::list_obsp`].
    pub fn list_varp(&self) -> Vec<String> {
        self.list_logical_names(
            "varp",
            SectionType::VarpEmbedding,
            SectionType::VarpEmbeddingShard,
        )
    }

    /// List the keys of every `varm/*` catalog entry. See [`Self::list_obsp`].
    pub fn list_varm(&self) -> Vec<String> {
        self.list_logical_names(
            "varm",
            SectionType::VarmEmbedding,
            SectionType::VarmEmbeddingShard,
        )
    }

    /// List the keys of every `obsm/*` catalog entry. See [`Self::list_obsp`].
    pub fn list_obsm(&self) -> Vec<String> {
        self.list_logical_names(
            "obsm",
            SectionType::ObsmEmbedding,
            SectionType::ObsmEmbeddingShard,
        )
    }

    /// Read a sharded `<prefix>/<name>` section, concatenating shards in
    /// `shard_idx` order. Returns `Ok(None)` if no shards exist for
    /// `<name>` (caller can fall back to the legacy single-section path).
    ///
    /// Concatenation is row-axis. The per-shard `shard_idx` /
    /// `row_start` / `n_shard_rows` / `n_rows_total` metadata stamped
    /// by the writer (see `stamp_dense_shard_meta`) is used to
    /// verify that the catalog entries form a contiguous, ordered cover
    /// of the logical matrix; any gap, duplicate, mismatch, or missing
    /// metadata returns `ScxError::InvalidCatalog` rather than silently
    /// producing a truncated matrix.
    ///
    /// The returned `RecordBatch`'s schema metadata has the per-shard
    /// fields stripped (`shard_idx` / `row_start` / `n_shard_rows`) so
    /// downstream consumers don't see misleading first-shard values;
    /// `n_rows_total` and any payload-level metadata (e.g. sparse
    /// `n_rows` / `n_cols`) are preserved.
    fn read_sharded_layout(
        &self,
        prefix: &str,
        name: &str,
        shard_type: SectionType,
    ) -> Result<Option<RecordBatch>> {
        let shard_name_prefix = format!("{prefix}/{name}_shard_");
        let logical = format!("{prefix}/{name}");
        self.read_sharded_layout_by_prefix(&shard_name_prefix, &logical, shard_type)
    }

    /// Generic worker shared by [`Self::read_sharded_layout`] (which
    /// handles `<prefix>/<name>_shard_<idx>` naming) and the obs/var
    /// metadata shard readers (which use the flatter
    /// `<axis>/shard_<idx>` naming because there is no logical
    /// sub-name). All cover-verification and metadata-stripping logic
    /// lives here; callers just supply the shard-name prefix to scan
    /// for and the display string used in error messages.
    fn read_sharded_layout_by_prefix(
        &self,
        shard_name_prefix: &str,
        logical: &str,
        shard_type: SectionType,
    ) -> Result<Option<RecordBatch>> {
        let mut shards: Vec<(u32, &FullCatalogEntry)> = self
            .full_catalog
            .entries
            .iter()
            .filter(|e| e.section_type == shard_type && e.name.starts_with(shard_name_prefix))
            .filter_map(|e| {
                let suffix = e.name.strip_prefix(shard_name_prefix)?;
                let idx: u32 = suffix.parse().ok()?;
                Some((idx, e))
            })
            .collect();
        if shards.is_empty() {
            return Ok(None);
        }
        shards.sort_by_key(|(idx, _)| *idx);

        // Decode shards **without** the per-shard wide→narrow downcast;
        // [`assemble_sharded_metadata`] handles the upcast → cover
        // validation → concat → downcast pipeline (shared with the cloud
        // reader so both paths produce byte-identical results).
        //
        // Each shard decode is an independent zstd + Arrow-IPC deserialize of a
        // read-only mmap slice (`read_arrow_ipc_raw` never mutates `self`), so
        // the decode fans out across the rayon pool — mirroring the CSR read
        // path. `collect` on an indexed parallel iterator preserves order, so
        // `raw_batches` is byte-identical to the serial decode (the subsequent
        // assemble is order-sensitive: shards are pre-sorted by index above).
        // This is the dominant cost of assembling obs/var on atlas-scale sharded
        // files, which the single-threaded loop left serial.
        //
        // `SCX_METADATA_DECODE_SERIAL=1` forces the serial decode even in a
        // parallel build (per-call read; mirrors `SCX_SHUFDELTA_GPU_SEQUENTIAL`)
        // — the A/B safety valve for measuring the parallelization. The serial
        // and parallel arms produce byte-identical `raw_batches` (indexed collect
        // preserves order; shards are pre-sorted above).
        #[cfg(feature = "parallel")]
        let force_serial = std::env::var("SCX_METADATA_DECODE_SERIAL")
            .map(|v| v == "1")
            .unwrap_or(false);
        #[cfg(feature = "parallel")]
        let raw_batches: Vec<(u32, RecordBatch)> = if force_serial {
            shards
                .iter()
                .map(|(idx, entry)| Ok((*idx, self.read_arrow_ipc_raw(entry)?)))
                .collect::<Result<_>>()?
        } else {
            shards
                .par_iter()
                .map(|(idx, entry)| Ok((*idx, self.read_arrow_ipc_raw(entry)?)))
                .collect::<Result<_>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let raw_batches: Vec<(u32, RecordBatch)> = shards
            .iter()
            .map(|(idx, entry)| Ok((*idx, self.read_arrow_ipc_raw(entry)?)))
            .collect::<Result<_>>()?;
        Ok(Some(assemble_sharded_metadata(logical, raw_batches)?))
    }

    /// Walk the catalog for both single-section and sharded entries
    /// under `<prefix>/`, returning a `name -> RecordBatch` map. Sharded
    /// entries are concatenated via [`Self::read_sharded_layout`];
    /// single-section entries fall through unchanged.
    fn read_all_sharded_or_single(
        &self,
        prefix: &str,
        single_type: SectionType,
        shard_type: SectionType,
    ) -> Result<HashMap<String, RecordBatch>> {
        let mut result = HashMap::new();
        let path_prefix = format!("{prefix}/");
        let shard_marker = "_shard_";

        // Sharded keys: collect unique logical names, then read each.
        let mut sharded_names: std::collections::BTreeSet<String> = Default::default();
        for entry in &self.full_catalog.entries {
            if entry.section_type != shard_type {
                continue;
            }
            if let Some(rest) = entry.name.strip_prefix(&path_prefix) {
                if let Some(idx) = rest.rfind(shard_marker) {
                    sharded_names.insert(rest[..idx].to_string());
                }
            }
        }
        for name in &sharded_names {
            if let Some(batch) = self.read_sharded_layout(prefix, name, shard_type)? {
                result.insert(name.clone(), batch);
            }
        }

        // Legacy single-section entries (only included if the same name
        // wasn't already produced from shards — shards win).
        for entry in &self.full_catalog.entries {
            if entry.section_type != single_type {
                continue;
            }
            let name = entry
                .name
                .strip_prefix(&path_prefix)
                .unwrap_or(&entry.name)
                .to_string();
            if result.contains_key(&name) {
                continue;
            }
            result.insert(name, self.read_arrow_ipc(entry)?);
        }
        Ok(result)
    }

    /// Pure catalog scan: return the de-duplicated set of logical names
    /// under `<prefix>/`, considering both legacy single-section and
    /// sharded entries. Used by `list_obsm` / `list_obsp` / etc.
    ///
    /// Delegates to [`FullCatalog::list_logical_names`] so the cloud reader
    /// shares the same logic.
    fn list_logical_names(
        &self,
        prefix: &str,
        single_type: SectionType,
        shard_type: SectionType,
    ) -> Vec<String> {
        self.full_catalog
            .list_logical_names(prefix, single_type, shard_type)
    }

    /// Read the `var` metadata batch for a specific modality.
    /// `modality_id == 0` reads the global / single-modality `var`
    /// section (matches `read_var()`).
    pub fn read_var_for(&self, modality_id: u8) -> Result<RecordBatch> {
        // Global / single-modality var: delegate to `read_var`, which assembles
        // sharded var (`var_metadata/shard_*`) and falls back to the legacy
        // single `var` section — a naive `get("var")` would miss sharded var.
        if modality_id == 0 {
            return self.read_var();
        }
        let mname = self.modality_name_for_id(modality_id)?;
        let key = format!("var/{mname}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc(entry)
    }

    /// Read the `var` schema for a specific modality without
    /// materialising the full RecordBatch. `modality_id == 0` reads the
    /// global / single-modality `var` section (matches
    /// [`Self::read_var_schema`]). Mirror of [`Self::read_var_for`].
    pub fn read_var_schema_for(&self, modality_id: u8) -> Result<arrow::datatypes::Schema> {
        // Global / single-modality var schema: delegate to `read_var_schema`,
        // which resolves the first var section for both sharded and legacy
        // single-section layouts (a naive `get("var")` would miss sharded var).
        if modality_id == 0 {
            return self.read_var_schema();
        }
        let mname = self.modality_name_for_id(modality_id)?;
        let key = format!("var/{mname}");
        let entry = self
            .full_catalog
            .get(&key)
            .ok_or_else(|| ScxError::SectionNotFound(key))?;
        self.read_arrow_ipc_schema(entry)
    }

    /// Read an obsm batch keyed by `(modality_id, key)`. Section
    /// names are `obsm/{modality_name}/{key}` for `modality_id >= 1`
    /// and `obsm/{key}` for `modality_id == 0` (global).
    pub fn read_obsm_for(&self, modality_id: u8, key: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_obsm_for
            .fetch_add(1, Ordering::Relaxed);
        let (shard_prefix, logical) = if modality_id == 0 {
            (format!("obsm/{key}_shard_"), format!("obsm/{key}"))
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            (
                format!("obsm/{mname}/{key}_shard_"),
                format!("obsm/{mname}/{key}"),
            )
        };
        if let Some(batch) = self.read_sharded_layout_by_prefix(
            &shard_prefix,
            &logical,
            SectionType::ObsmEmbeddingShard,
        )? {
            return Ok(batch);
        }
        let entry = self
            .full_catalog
            .get(&logical)
            .ok_or_else(|| ScxError::SectionNotFound(logical))?;
        self.read_arrow_ipc(entry)
    }

    /// Read a per-modality varm embedding as an Arrow RecordBatch.
    ///
    /// Mirrors [`Self::read_obsm_for`]: tries the sharded layout
    /// (`varm/{modality_name}/{key}_shard_<idx>`, section type
    /// [`SectionType::VarmEmbeddingShard`]) first, falling back to the
    /// legacy single-section [`SectionType::VarmEmbedding`].
    pub fn read_varm_for(&self, modality_id: u8, key: &str) -> Result<RecordBatch> {
        #[cfg(debug_assertions)]
        self.debug_counts
            .read_varm_for
            .fetch_add(1, Ordering::Relaxed);
        let (shard_prefix, logical) = if modality_id == 0 {
            (format!("varm/{key}_shard_"), format!("varm/{key}"))
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            (
                format!("varm/{mname}/{key}_shard_"),
                format!("varm/{mname}/{key}"),
            )
        };
        if let Some(batch) = self.read_sharded_layout_by_prefix(
            &shard_prefix,
            &logical,
            SectionType::VarmEmbeddingShard,
        )? {
            return Ok(batch);
        }
        let entry = self
            .full_catalog
            .get(&logical)
            .ok_or_else(|| ScxError::SectionNotFound(logical))?;
        self.read_arrow_ipc(entry)
    }

    /// Read the per-modality `uns` JSON, decoded to `serde_json::Value`.
    pub fn read_uns_for(&self, modality_id: u8) -> Result<serde_json::Value> {
        let section_name = if modality_id == 0 {
            "uns".to_string()
        } else {
            let mname = self.modality_name_for_id(modality_id)?;
            format!("uns/{mname}")
        };
        let entry = self
            .full_catalog
            .get(&section_name)
            .ok_or_else(|| ScxError::SectionNotFound(section_name))?;
        let bytes = self.section_bytes(entry)?;
        scx_format::parse_uns_json(bytes)
    }

    /// Read the `adata.raw.var` DataFrame ([`SectionType::RawVarMetadata`]).
    pub fn read_raw_var(&self) -> Result<RecordBatch> {
        let entry = self
            .full_catalog
            .get("raw/var")
            .ok_or_else(|| ScxError::SectionNotFound("raw/var".to_string()))?;
        self.read_arrow_ipc(entry)
    }
}
