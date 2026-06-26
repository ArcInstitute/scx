//! F2 — reader for the `group_index` sidecar (condition/label-grouped sharding).
//!
//! Loads the JSON payload written by `scx sort --group-by` (section
//! [`scx_format_io::section::SectionType::GroupIndex`]) and exposes the
//! label → record lookup the grouped-read API (`read_group` / `read_reference`
//! / `iter_group_shards`) drives.
//!
//! Records carry **global** output-row ranges (`[row_start, row_stop)`), so the
//! default read route (`QueryPipeline::read_row_range`) slices them directly.

use std::collections::HashMap;

use crate::error::{EngineError, Result};
use crate::reader::SectionReader;

/// Role of a group record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRole {
    /// A normal `group_by` group.
    Group,
    /// Reference cells (packed first / isolated).
    Reference,
}

/// One group record: a contiguous run of one (label, role) in the global output
/// order, entirely within one shard (never-split-a-group invariant).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupRecord {
    pub label: String,
    pub shard: u32,
    /// First global output row (inclusive).
    pub row_start: u64,
    /// One past the last global output row (exclusive).
    pub row_stop: u64,
    pub role: GroupRole,
}

/// In-memory view of the `group_index` sidecar.
#[derive(Debug, Clone)]
pub struct GroupIndex {
    pub group_by: String,
    pub reference_shard: Option<u32>,
    pub reference_labels: Vec<String>,
    records: Vec<GroupRecord>,
    /// label → index into `records`. On a collision (a label that split into a
    /// reference and a group record under `ReferenceSpec::Column`)
    by_label: HashMap<String, usize>,
}

impl GroupIndex {
    /// Load the group index from a reader. `Err(EngineError::NotGrouped)` if the
    /// archive has no `group_index` section.
    pub fn open(reader: &dyn SectionReader) -> Result<Self> {
        let bytes = reader
            .read_group_index_bytes()?
            .ok_or(EngineError::NotGrouped)?;
        Self::from_bytes(&bytes)
    }

    /// Parse the JSON payload (§ format.md `group_index`).
    ///
    /// Deserializes via the shared `scx_format::GroupIndexPayload` derive, so
    /// every field is required by construction (a missing field is a hard
    /// deserialization error) and the wire schema has a single definition.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let payload: scx_format::GroupIndexPayload = serde_json::from_slice(bytes)
            .map_err(|e| EngineError::Generic(format!("malformed group_index payload: {e}")))?;
        let scx_format::GroupIndexPayload {
            group_by,
            reference_shard,
            reference_labels,
            records: records_wire,
        } = payload;
        let records: Vec<GroupRecord> = records_wire
            .into_iter()
            .map(|r| GroupRecord {
                label: r.label,
                shard: r.shard,
                row_start: r.row_start,
                row_stop: r.row_stop,
                // Lenient on unknown role strings (default Group), matching the
                // historical parser; the writer only ever emits group/reference.
                role: if r.role == "reference" {
                    GroupRole::Reference
                } else {
                    GroupRole::Group
                },
            })
            .collect();

        // Build the label map: non-reference wins on collision.
        let mut by_label: HashMap<String, usize> = HashMap::new();
        for (i, rec) in records.iter().enumerate() {
            match by_label.get(&rec.label) {
                Some(&existing) if records[existing].role == GroupRole::Group => {
                    // keep the existing non-reference record
                }
                _ => {
                    by_label.insert(rec.label.clone(), i);
                }
            }
        }

        Ok(Self {
            group_by,
            reference_shard,
            reference_labels,
            records,
            by_label,
        })
    }

    /// The group-role record for `label` (non-reference wins on a split label).
    pub fn record(&self, label: &str) -> Option<&GroupRecord> {
        self.by_label.get(label).map(|&i| &self.records[i])
    }

    /// All records (group + reference), in shard order.
    pub fn records(&self) -> &[GroupRecord] {
        &self.records
    }

    /// Distinct labels present (for discovery / error messages).
    pub fn labels(&self) -> Vec<String> {
        let mut v: Vec<String> = self.by_label.keys().cloned().collect();
        v.sort();
        v
    }

    /// The contiguous reference region `[start, stop)` in global output rows —
    /// the union of all reference records (always the leading range by
    /// construction). `None` if the archive has no reference rows.
    pub fn reference_range(&self) -> Option<(u64, u64)> {
        let mut start = u64::MAX;
        let mut stop = 0u64;
        let mut any = false;
        for r in self
            .records
            .iter()
            .filter(|r| r.role == GroupRole::Reference)
        {
            any = true;
            start = start.min(r.row_start);
            stop = stop.max(r.row_stop);
        }
        if any {
            Some((start, stop))
        } else {
            None
        }
    }

    /// One handle per non-reference shard, in shard order, each carrying its
    /// global row range and a label → shard-local `[start, stop)` map. The
    /// reference shard is excluded.
    pub fn shard_handles(&self) -> Vec<GroupShardHandle> {
        use std::collections::BTreeMap;
        let mut by_shard: BTreeMap<u32, Vec<&GroupRecord>> = BTreeMap::new();
        for r in self.records.iter().filter(|r| r.role == GroupRole::Group) {
            by_shard.entry(r.shard).or_default().push(r);
        }
        by_shard
            .into_iter()
            .map(|(shard, recs)| {
                let global_start = recs.iter().map(|r| r.row_start).min().unwrap_or(0);
                let global_stop = recs.iter().map(|r| r.row_stop).max().unwrap_or(0);
                let groups = recs
                    .iter()
                    .map(|r| {
                        (
                            r.label.clone(),
                            r.row_start - global_start,
                            r.row_stop - global_start,
                        )
                    })
                    .collect();
                GroupShardHandle {
                    shard_index: shard,
                    global_start,
                    global_stop,
                    groups,
                }
            })
            .collect()
    }

    /// `difflib.get_close_matches`-equivalent near matches (normalized
    /// Levenshtein via `strsim`), best-first, capped at `n`.
    pub fn close_matches(&self, label: &str, n: usize) -> Vec<String> {
        let mut scored: Vec<(f64, &String)> = self
            .by_label
            .keys()
            .map(|cand| (strsim::normalized_levenshtein(label, cand), cand))
            .filter(|(score, _)| *score >= 0.6) // difflib's default cutoff
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(n).map(|(_, s)| s.clone()).collect()
    }
}

/// A non-reference shard's grouped contents: its global row range plus a
/// label → shard-local `[start, stop)` map. The caller reads the rows via
/// [`crate::QueryPipeline::read_row_range`] over `[global_start, global_stop)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupShardHandle {
    pub shard_index: u32,
    pub global_start: u64,
    pub global_stop: u64,
    /// `(label, local_start, local_stop)`, local to this shard's `global_start`.
    pub groups: Vec<(String, u64, u64)>,
}

impl GroupShardHandle {
    /// The **global** `[start, stop)` row range of `label` within this shard, or
    /// `None` if the label is not resident here. Reads of a single label go
    /// through [`crate::QueryPipeline::read_row_range`] over this range.
    pub fn range(&self, label: &str) -> Option<(u64, u64)> {
        self.groups
            .iter()
            .find(|(l, _, _)| l == label)
            .map(|(_, ls, le)| (self.global_start + ls, self.global_start + le))
    }

    /// Label → shard-local `(start, stop)` map for introspection.
    pub fn local_ranges(&self) -> Vec<(String, u64, u64)> {
        self.groups.clone()
    }
}

#[cfg(test)]
#[path = "group_tests.rs"]
mod tests;
