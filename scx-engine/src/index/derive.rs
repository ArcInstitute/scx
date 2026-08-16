//! Level-1 catalog statistics derived from a finished obs predicate index.
//!
//! Level 2 (the row-set fast path) reads the index itself; level 1 prunes whole
//! shards from the `column_stats` these functions attach to catalog entries.

use crate::error::Result;

use super::{IndexedColumn, PredicateIndex};

/// Derive per-shard catalog column statistics from a finished obs
/// [`PredicateIndex`].
///
/// `per_shard[k]` holds the stats for the k-th shard in the `shard_id` space
/// the index was built against (== catalog CSR-shard sorted order for
/// convert/merge/compact). For each categorical column we emit a
/// [`ColumnStat::CategoryBitset`] whose bit `i` is set iff dictionary value `i`
/// (the i-th entry, which is sorted ascending) is present in that shard — the
/// exact convention `pushdown::can_exclude_shard` checks
/// (`values.binary_search` → bit `i`). For each numeric column we fold the B+
/// tree leaf entries into a per-shard [`ColumnStat::MinMax`].
///
/// Precondition: every `shard_id` recorded in the index lies in `0..n_shards`
/// (the index is built against exactly `n_shards` `shard_row_ranges`). An
/// out-of-range id signals an index/catalog shard-space misalignment, which
/// would silently mis-place "value present" bits and risk incorrect shard
/// skipping; it trips a `debug_assert` and is otherwise dropped defensively.
///
/// **This positional `shard_id` → CSR-shard mapping is a write-time concern,
/// and multimodal safety does not rest on it alone.** Each modality's CSR shards
/// independently tile `[0, n_obs)`, so a flattened walk over "the i-th CSR
/// shard" stops meaning "shard_id i" the moment a second modality exists. Three
/// *independent* protections exist; none of them is "the file cannot contain an
/// index":
///
/// - **Write side (here).**
///   [`scx_format_io::writer::assign_csr_shard_column_stats`] counts only
///   `modality_id == 0` CSR entries and returns `ColumnStatsShardCountMismatch`
///   rather than mis-assigning, so these derived stats cannot land on a
///   multimodal file's shards
///   (`writer_tests::bulk_csr_shard_column_stats_refuses_a_multimodal_file`).
/// - **Level-2 read.** `collect::build_plan` forces `obs_predicate_index` to
///   `None` for every `modality_id != 0` pipeline, so `csr_shard_ranges_table`
///   never runs on a modality-scoped query even if an index section is present.
/// - **Level-1 read.** `pushdown::prune_shards_by_catalog_with_dict` reads the
///   `column_stats` already attached to each catalog entry and never consults an
///   index `shard_id`, so it is unaffected by this mapping either way.
///
/// **Index omission is a convention, not an invariant.** `scx convert` on an
/// h5mu emits `ConvertWarning::PredicateIndexSkippedMultimodal`,
/// `scx-ops::merge` records `multimodal_skip`, and `merge_multimodal` never
/// calls this — but `ScxWriter::write_obs_predicate_index` is public and accepts
/// a writer with registered modalities, so a file *can* carry one. That is why
/// the two read-side guards above matter and must not be removed on the grounds
/// that such a file "cannot exist".
///
/// Shipping multimodal indexing means giving `ShardRange` a modality scope, not
/// relaxing any of this.
pub fn derive_shard_column_stats(
    index: &PredicateIndex,
    n_shards: usize,
) -> Vec<Vec<scx_format_io::catalog::ColumnStat>> {
    use scx_format_io::catalog::ColumnStat;
    let mut per_shard: Vec<Vec<ColumnStat>> = vec![Vec::new(); n_shards];
    for column in &index.columns {
        match column {
            IndexedColumn::Categorical(cat) => {
                let hash = scx_format_io::column_name_hash(&cat.column_name);
                let n_values = cat.entries.len();
                let n_bytes = n_values.div_ceil(8);
                let mut bitsets: Vec<Vec<u8>> = vec![vec![0u8; n_bytes]; n_shards];
                for (bit, entry) in cat.entries.iter().enumerate() {
                    for range in &entry.shard_ranges {
                        let shard = range.shard_id as usize;
                        debug_assert!(
                            shard < n_shards,
                            "categorical shard_id {shard} >= n_shards {n_shards} — \
                             index/catalog shard-space misalignment"
                        );
                        if shard < n_shards {
                            bitsets[shard][bit / 8] |= 1 << (bit % 8);
                        }
                    }
                }
                for (shard, bitset) in bitsets.into_iter().enumerate() {
                    per_shard[shard].push(ColumnStat::CategoryBitset {
                        column_name_hash: hash,
                        bitset,
                    });
                }
            }
            IndexedColumn::Numeric(num) => {
                let hash = scx_format_io::column_name_hash(&num.column_name);
                let mut minmax: Vec<Option<(f64, f64)>> = vec![None; n_shards];
                for page in &num.leaf_pages {
                    for entry in &page.entries {
                        let shard = entry.shard_id as usize;
                        debug_assert!(
                            shard < n_shards,
                            "numeric shard_id {shard} >= n_shards {n_shards} — \
                             index/catalog shard-space misalignment"
                        );
                        if shard < n_shards {
                            let slot =
                                minmax[shard].get_or_insert((entry.min_value, entry.max_value));
                            slot.0 = slot.0.min(entry.min_value);
                            slot.1 = slot.1.max(entry.max_value);
                        }
                    }
                }
                for (shard, mm) in minmax.into_iter().enumerate() {
                    if let Some((min, max)) = mm {
                        per_shard[shard].push(ColumnStat::MinMax {
                            column_name_hash: hash,
                            min,
                            max,
                        });
                    }
                }
            }
        }
    }
    per_shard
}

/// Parse a serialized obs predicate index and attach the per-shard column stats
/// it implies to the writer's CSR shard catalog entries (one bulk pass).
///
/// `n_shards` must equal the number of CSR shards written and the length of the
/// `shard_row_ranges` the index was built with. Called after
/// `write_obs_predicate_index` in every fresh-writer build path.
pub fn apply_obs_shard_column_stats(
    writer: &mut scx_format_io::ScxWriter,
    obs_index_bytes: &[u8],
    n_shards: usize,
) -> Result<()> {
    let index = PredicateIndex::read_from(&mut std::io::Cursor::new(obs_index_bytes))?;
    let per_shard = derive_shard_column_stats(&index, n_shards);
    writer.set_csr_shard_column_stats_bulk(per_shard)?;
    Ok(())
}
