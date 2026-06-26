//! F1 — condition/label-grouped shard planner.
//!
//! The global sort order is already computed by the sort engine (reference-first via a
//! synthetic key, then `group_by` label, then any secondary keys), so this
//! planner consumes the **emission-order** group ids, per-row reference flags,
//! and per-row nnz, and produces the shard-break offsets the `CsrEmitter`
//! obeys, plus the `GroupRecord`s persisted in the `GroupIndex` sidecar.
//!
//! It runs after the order is known and before emission. Offline (not
//! streaming-greedy) is exactly what lets it cut **only at group edges** and
//! isolate the reference role.

/// Role of a group block within the grouped layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// A normal `group_by` group.
    Group,
    /// Reference cells (e.g. "non-targeting"), packed first / isolated.
    Reference,
}

impl Role {
    /// Lowercase wire form (`"group"` / `"reference"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Group => "group",
            Role::Reference => "reference",
        }
    }
}

/// One persisted group record. `row_start`/`row_stop` are **global** output-row
/// indices (post-reorder, emission order), half-open `[start, stop)`. With the
/// never-split-a-group invariant each record lies entirely within one shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupRecord {
    /// The `group_by` value (string-coerced).
    pub label: String,
    /// Shard index containing this (label, role) run.
    pub shard: u32,
    /// First global output row (inclusive).
    pub row_start: u64,
    /// One past the last global output row (exclusive).
    pub row_stop: u64,
    /// `group` or `reference`.
    pub role: Role,
}

/// Output of the planner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupPlan {
    /// Emit-row indices at which the emitter must seal a shard *before* pushing
    /// that row (the interior shard starts; excludes 0 and the final EOF).
    /// Authoritative breaks — the legacy fixed-size cap is disabled in group
    /// mode.
    pub shard_starts: Vec<u64>,
    /// One record per (label, role) contiguous run.
    pub records: Vec<GroupRecord>,
    /// Shard holding the reference role (always 0 by construction) or `None`.
    pub reference_shard: Option<u32>,
    /// Total number of output shards.
    pub n_shards: u32,
}

impl GroupPlan {
    /// Serialize the sidecar payload (`group_index` section, §2.5 of the spec):
    /// `{group_by, reference_shard, reference_labels, records[]}`.
    pub fn to_sidecar_json(
        &self,
        group_by: &str,
        reference_labels: &[String],
    ) -> serde_json::Value {
        let records: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "label": r.label,
                    "shard": r.shard,
                    "row_start": r.row_start,
                    "row_stop": r.row_stop,
                    "role": r.role.as_str(),
                })
            })
            .collect();
        serde_json::json!({
            "group_by": group_by,
            "reference_shard": self.reference_shard,
            "reference_labels": reference_labels,
            "records": records,
        })
    }
}

/// Plan group-aligned shard boundaries over an already-sorted emission order.
///
/// Inputs are all in **emission order** (i.e. already permuted reference-first,
/// then by group label, then secondary):
/// - `group_of_new[i]`: group id of emission row `i` (a valid index into
///   `labels`; the caller maps null/deleted to a real synthetic group such as
///   `"__ungrouped__"`, never `-1`).
/// - `ref_of_new[i]`: `true` if emission row `i` is a reference cell.
/// - `labels`: group id → label string.
/// - `per_row_nnz`: emission-order per-row nnz. **Empty ⇒ row-count mode** (the
///   budget is interpreted as a row count and `bytes_per_nnz` is ignored).
/// - `target_units`: byte budget per shard (byte mode) or row budget (row-count
///   mode).
/// - `bytes_per_nnz`: width estimate sizing each block (byte mode only).
/// - `max_units`: oversize threshold; a single block exceeding it gets its own
///   shard + `log::warn!`. Never splits a group regardless.
///
/// Faithful to `grouping.py:237-334` (steps 2–3); step 1 (the sort) is done by
/// the engine upstream.
pub fn plan_group_shards(
    group_of_new: &[i32],
    ref_of_new: &[bool],
    labels: &[String],
    per_row_nnz: &[u64],
    target_units: u64,
    bytes_per_nnz: u64,
    max_units: u64,
) -> GroupPlan {
    let n = group_of_new.len();
    assert_eq!(ref_of_new.len(), n, "ref_of_new length mismatch");
    let row_count_mode = per_row_nnz.is_empty();
    assert!(
        row_count_mode || per_row_nnz.len() == n,
        "per_row_nnz length mismatch"
    );
    let target = target_units.max(1);

    if n == 0 {
        return GroupPlan {
            shard_starts: Vec::new(),
            records: Vec::new(),
            reference_shard: None,
            n_shards: 0,
        };
    }

    // --- Step 2: derive contiguous blocks at composite (ref, group) changes. ---
    // Each block: (group_id, role, [start_emit, stop_emit)).
    struct Block {
        gid: i32,
        role: Role,
        start: u64,
        stop: u64,
    }
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let gid = group_of_new[i];
        let is_ref = ref_of_new[i];
        let start = i;
        i += 1;
        while i < n && group_of_new[i] == gid && ref_of_new[i] == is_ref {
            i += 1;
        }
        blocks.push(Block {
            gid,
            role: if is_ref { Role::Reference } else { Role::Group },
            start: start as u64,
            stop: i as u64,
        });
    }

    // Block size in budget units.
    let block_units = |b: &Block| -> u64 {
        if row_count_mode {
            b.stop - b.start
        } else {
            let nnz: u64 = per_row_nnz[b.start as usize..b.stop as usize].iter().sum();
            nnz.saturating_mul(bytes_per_nnz)
        }
    };

    // --- Step 3: bin-pack, cutting only at group edges. ---
    let mut records: Vec<GroupRecord> = Vec::with_capacity(blocks.len());
    let mut shard_starts: Vec<u64> = Vec::new();
    let mut cur_units: u64 = 0;
    let mut cur_shard: u32 = 0;
    let mut last_role: Option<Role> = None;
    let mut emit_pos: u64 = 0;
    let mut total_ref_units: u64 = 0;

    for b in &blocks {
        let gb = block_units(b);

        let boundary = match b.role {
            // Reference must be isolated: seal if we already hold non-reference
            // content. After the upstream sort, reference is the leading run so
            // this only fires defensively.
            Role::Reference => emit_pos > 0 && last_role != Some(Role::Reference),
            // Non-reference: seal on role switch (ref → group) or when adding
            // this group would exceed the target.
            Role::Group => {
                last_role == Some(Role::Reference) || (cur_units + gb > target && emit_pos > 0)
            }
        };

        if boundary {
            // Seal the current shard *before* this block.
            shard_starts.push(emit_pos);
            cur_shard += 1;
            cur_units = 0;
        }

        debug_assert!(
            b.gid >= 0 && (b.gid as usize) < labels.len(),
            "group id {} out of range for {} labels (caller must remap null/deleted to a real \
             synthetic group, never -1)",
            b.gid,
            labels.len()
        );

        if b.role == Role::Reference {
            total_ref_units += gb;
        } else if gb > max_units {
            log::warn!(
                "group {:?} occupies {} units, exceeding max {}; it will be an oversized shard",
                labels
                    .get(b.gid as usize)
                    .map(String::as_str)
                    .unwrap_or("?"),
                gb,
                max_units
            );
        }

        records.push(GroupRecord {
            label: labels
                .get(b.gid as usize)
                .cloned()
                .unwrap_or_else(|| "?".to_string()),
            shard: cur_shard,
            row_start: b.start,
            row_stop: b.stop,
            role: b.role,
        });
        cur_units += gb;
        last_role = Some(b.role);
        emit_pos = b.stop;
    }

    if total_ref_units > max_units {
        log::warn!(
            "combined reference shard occupies {} units, exceeding max {}; it will be oversized",
            total_ref_units,
            max_units
        );
    }

    let reference_shard = records
        .iter()
        .find(|r| r.role == Role::Reference)
        .map(|r| r.shard);

    GroupPlan {
        shard_starts,
        records,
        reference_shard,
        n_shards: cur_shard + 1,
    }
}

#[cfg(test)]
#[path = "group_plan_tests.rs"]
mod tests;
