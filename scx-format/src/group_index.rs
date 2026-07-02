//! Wire schema for the `group_index` sidecar (section
//! [`crate::section::SectionType::GroupIndex`], F1/F2 condition-grouped sharding).
//!
//! This is the single serde-derive definition of the JSON payload written by
//! `scx sort --group-by` and read back by the grouped-read API. The writer
//! (`scx-ops`) and reader (`scx-engine`) both go through these structs so the
//! wire schema has one home and every field is required by construction (a
//! missing field is a deserialization error).
//!
//! Field declaration order is significant and **must stay alphabetical**: serde
//! serializes struct fields in declaration order, and the historical sidecar was
//! emitted through `serde_json::json!`/`to_value`, which build a `BTreeMap`
//! (the workspace does not enable serde_json's `preserve_order` feature) and so
//! wrote keys in **alphabetical** order. Declaring these fields alphabetically
//! keeps the serde-derive output byte-identical to that historical JSON
//! (top: `{group_by, records, reference_labels, reference_shard}`; record:
//! `{label, role, row_start, row_stop, shard}`). Do not reorder without bumping
//! a format note — `scx-ops/src/group_plan_tests.rs::sidecar_bytes_are_byte_identical_to_legacy_json`
//! guards this.

use serde::{Deserialize, Serialize};

/// Top-level `group_index` payload. Fields are alphabetical (see module docs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupIndexPayload {
    /// The obs column the file was grouped by.
    pub group_by: String,
    /// One record per `(label, role)` contiguous run, in global output order.
    pub records: Vec<GroupRecordWire>,
    /// The labels treated as reference (empty under `ReferenceSpec::Column`).
    pub reference_labels: Vec<String>,
    /// Shard holding the reference role (`0` by construction) or `null`.
    pub reference_shard: Option<u32>,
}

/// One persisted group record. `row_start`/`row_stop` are **global** output-row
/// indices, half-open `[start, stop)`; each record lies entirely within one
/// shard (never-split-a-group invariant). `role` is the lowercase wire form
/// (`"group"` / `"reference"`). Fields are alphabetical (see module docs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupRecordWire {
    pub label: String,
    pub role: String,
    pub row_start: u64,
    pub row_stop: u64,
    pub shard: u32,
}
