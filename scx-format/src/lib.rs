//! On-disk layout and spec for the SCX format: file header, catalogs, shard
//! structs, modality table, provenance, codec selection, and the error/checksum
//! primitives. This crate is **pure** — no `std::fs`, no `memmap2`, no decode
//! dispatch. The runtime reader/writer and backed/streaming access live in the
//! `scx-format-io` crate, which depends on this one. This is the surface an
//! independent reader implementation and the format conformance vectors verify
//! against.

pub mod catalog;
pub mod catalog_view;
pub mod checksum;
pub mod codec_select;
pub mod csc_policy;
pub mod error;
pub mod group_index;
pub mod header;
pub mod modality;
pub mod obs_shard_policy;
pub mod provenance;
pub mod section;
pub mod shard;
pub mod versioned;

/// Arrow `Field::metadata` key marking a dictionary column as an *ordered*
/// categorical (R `ordered` factor / pandas ordered Categorical). Canonical
/// home shared by every binding (scx-convert re-exports it; pyscx and rscx
/// read it from here) so the wire key has a single definition.
pub const CATEGORICAL_ORDERED_KEY: &str = "scx.categorical.ordered";

/// Deepest container nesting an `uns` tree may have, counted in containers
/// (dict / list / tuple / nested HDF5 group) on the path from the `uns` root
/// to a leaf. Canonical home shared by every producer and consumer of the
/// `uns_blob` section: `pyscx`'s Python↔JSON walkers and `scx-convert`'s
/// h5ad↔JSON walkers all bound themselves by this one number, so a tree that
/// any writer accepts is a tree every reader can rebuild.
///
/// Two independent reasons this has to exist at all, both of which bite well
/// before any "reasonable metadata" threshold:
///
/// 1. **Deep is not the same as cyclic.** The write path detects cycles by
///    object identity, which says nothing about a merely-deep tree; recursing
///    into one exhausts the Rust stack and aborts the *process*. That is not a
///    catchable Python exception, so no `try`/`except` can contain it.
/// 2. **`serde_json`'s reader is stricter than its writer.** The serializer
///    has no depth limit, but `serde_json::Deserializer` refuses more than
///    [`SERDE_JSON_MAX_NESTING`] levels. Without a cap the writer happily
///    emits `uns` sections that no reader — including this crate's own — can
///    ever parse back.
pub const MAX_UNS_DEPTH: usize = 60;

/// Deepest JSON nesting `serde_json`'s default `Deserializer` will parse
/// before returning `RecursionLimitExceeded`. Its `remaining_depth` counter
/// starts at 128 and errors on *entering* the level that drives it to zero,
/// so 127 levels parse and 128 do not. Measured against the pinned version,
/// not just read off the source: a `uns` of 125 nested lists (126 arrays plus
/// the `uns` object itself = 127) round-trips, and 126 fails to parse.
///
/// Raising this is not a local edit — it is a claim about every reader that
/// will ever open the file, so it stays pinned to what the default
/// deserializer guarantees rather than to what a particular call site enables.
pub const SERDE_JSON_MAX_NESTING: usize = 127;

// `MAX_UNS_DEPTH` is *derived* from `SERDE_JSON_MAX_NESTING`, not chosen, and
// the derivation is enforced here so that raising the cap past what the reader
// can parse is a compile error rather than a class of silently unreadable
// files.
//
// The conversion factor is the tagged-envelope encoding: a Python tuple is
// written as `{"__scx_type__": "tuple", "data": [...]}`, which is *two* JSON
// levels for one container, while a dict or list is one. The `uns` root is
// always a dict, so the worst case is one object plus `MAX_UNS_DEPTH - 1`
// tuples:
//
//     1 + 2 * (MAX_UNS_DEPTH - 1)  <=  SERDE_JSON_MAX_NESTING
//
// Confirmed by measurement at the boundary: 63 nested tuples under the `uns`
// dict is 64 containers and 127 JSON levels and reads back; 64 tuples is 129
// levels and does not. `MAX_UNS_DEPTH` sits below that ceiling deliberately —
// real `uns` is 3–6 deep (scanpy's deepest standard structure,
// `rank_genes_groups`, is 3), so the headroom costs nothing and absorbs any
// future envelope that expands by more than 2x.
const _: () = assert!(2 * MAX_UNS_DEPTH - 1 <= SERDE_JSON_MAX_NESTING);

/// Reject an `uns` tree that could not be read back, before it is serialized.
///
/// The last gate before bytes hit the disk, and the only one every writer
/// shares: `pyscx`, the h5ad ingest pipeline, `scx-cli subset`, `scx-ops`
/// merge and any other Rust caller all reach the on-disk section through
/// `ScxWriter::write_uns`. The per-walker caps upstream bound the trees this
/// workspace *builds*; this bounds the trees it *stores*, so "an SCX file
/// always has a readable `uns`" holds for a hand-built `serde_json::Value`
/// too.
///
/// Bounded by [`SERDE_JSON_MAX_NESTING`] rather than [`MAX_UNS_DEPTH`]: the
/// invariant being defended is readability, and holding stored trees to the
/// stricter producer cap would reject legitimate output — `scx-ops` merge
/// under `UnsPolicy::Namespace` wraps each input in one extra level, so
/// merging inputs that were themselves at the producer cap is expected to
/// land just above it.
///
/// **Iterative on purpose.** A recursive depth check on a pathological tree
/// overflows the stack while measuring it — failing in exactly the way it
/// exists to prevent.
pub fn validate_uns_depth(json: &serde_json::Value) -> Result<()> {
    let mut stack: Vec<(&serde_json::Value, usize)> = vec![(json, 1)];
    while let Some((value, depth)) = stack.pop() {
        let children: Box<dyn Iterator<Item = &serde_json::Value>> = match value {
            serde_json::Value::Array(items) => Box::new(items.iter()),
            serde_json::Value::Object(map) => Box::new(map.values()),
            _ => continue,
        };
        if depth > SERDE_JSON_MAX_NESTING {
            return Err(ScxError::UnsTooDeep {
                max_nesting: SERDE_JSON_MAX_NESTING,
            });
        }
        stack.extend(children.map(|child| (child, depth + 1)));
    }
    Ok(())
}

/// Parse the bytes of an `uns_blob` section into a JSON tree.
///
/// The single entry point every reader — local, cloud, CLI — goes through, so
/// that `serde_json`'s depth refusal is reported once, in terms of `uns`,
/// instead of as a bare `recursion limit exceeded at line 1 column 135` that
/// names neither the section nor anything the user can act on.
///
/// Detection is by message text because `serde_json` does not expose its error
/// codes. That is deliberately fail-soft: if upstream ever rewords it, the
/// original [`ScxError::Json`] surfaces, which is exactly today's behaviour.
pub fn parse_uns_json(bytes: &[u8]) -> Result<serde_json::Value> {
    serde_json::from_slice(bytes).map_err(|e| {
        if e.classify() == serde_json::error::Category::Syntax
            && e.to_string().contains("recursion limit exceeded")
        {
            ScxError::UnsTooDeep {
                max_nesting: SERDE_JSON_MAX_NESTING,
            }
        } else {
            ScxError::Json(e)
        }
    })
}

#[allow(deprecated)]
pub use catalog::SHARD_STATS_BASE_SIZE;
pub use catalog::{
    column_name_hash, ColumnStat, FullCatalog, FullCatalogEntry, LazyShardStats, RootCatalog,
    RootCatalogEntry, ShardStats, CURRENT_CATALOG_VERSION, ROOT_CATALOG_ENTRY_SIZE,
    ROOT_CATALOG_MAX_SIZE, SHARD_STATS_BASE_SIZE_V1, SHARD_STATS_BASE_SIZE_V2,
};
pub use catalog_view::{CatalogView, CatalogViewEntry, ShardStatsLite};
pub use checksum::{blake3_hash, blake3_truncated_64};
pub use codec_select::{
    pick_codec_v2, resolve_codec, select_codec, select_codec_for_modality, DecodeTarget,
    ResolvedCodec, ADOPT_MARGIN,
};
pub use csc_policy::{
    auto_obs_threshold, auto_vars_threshold, CscPolicy, AUTO_CSC_OBS_THRESHOLD,
    AUTO_CSC_VARS_THRESHOLD,
};
pub use error::{validate_allocation, Result, ScxError, ScxErrorClass};
pub use group_index::{GroupIndexPayload, GroupRecordWire};
pub use header::{
    rewrite_output_format_version, FileHeader, CURRENT_FORMAT_VERSION,
    DEFAULT_WRITE_FORMAT_VERSION, HEADER_SIZE, MAGIC,
};
pub use modality::{
    ModalityFlags, ModalityInfo, ModalityTable, ModalityType, MAX_MODALITIES,
    MODALITY_NAME_MAX_BYTES, MODALITY_TABLE_MAGIC, MODALITY_TABLE_VERSION,
};
pub use obs_shard_policy::ObsShardPolicy;
pub use provenance::{Provenance, ProvenanceEntry};
pub use section::{align_to_8, SectionType};
pub use shard::{
    clamped_reserve, derive_shard_type, resolve_block_index, BlockIndex, BlockIndexEntry,
    ShardHeader, BLOCK_INDEX_ENTRY_SIZE, CURRENT_SHARD_FORMAT_VERSION,
    DEFAULT_WRITE_SHARD_FORMAT_VERSION, MAX_ELEMENTS_PER_ENCODED_BYTE, MAX_RESERVE_BYTES,
    MIN_RESERVE_ELEMENTS, SHARD_HEADER_SIZE, SHARD_MAGIC,
};
pub use versioned::VersionedSection;

/// Default number of rows per CSR shard when callers don't override it.
///
/// 16 384 is a power of two (aligns with typical GPU batch sizes and
/// memory page boundaries) and matches the `ScxWriter` test-fixture
/// default. Used by the pyscx ops bindings and the `scx convert` /
/// `scx append` / `scx subset` CLI subcommands so identical inputs
/// through CLI and Python produce identical shard layouts.
pub const DEFAULT_SHARD_TARGET_ROWS: u32 = 16384;

#[cfg(test)]
mod uns_depth_tests {
    use super::*;

    /// Pins [`SERDE_JSON_MAX_NESTING`] to what `serde_json` actually does.
    ///
    /// The constant is load-bearing: [`MAX_UNS_DEPTH`] is derived from it at
    /// compile time, so if a dependency bump moved the parser's limit, every
    /// writer in the workspace would quietly start emitting `uns` sections
    /// that no reader could take back. A comment asserting "the limit is 127"
    /// would not have caught that; this does.
    #[test]
    fn serde_json_nesting_limit_is_where_the_constant_says() {
        let arrays = |n: usize| format!("{}1{}", "[".repeat(n), "]".repeat(n));

        assert!(
            parse_uns_json(arrays(SERDE_JSON_MAX_NESTING).as_bytes()).is_ok(),
            "{SERDE_JSON_MAX_NESTING} levels must parse — writers are allowed up to here"
        );
        assert!(
            matches!(
                parse_uns_json(arrays(SERDE_JSON_MAX_NESTING + 1).as_bytes()),
                Err(ScxError::UnsTooDeep { .. })
            ),
            "one level past the constant must be refused, not accepted"
        );
    }

    /// A file written before `uns` nesting was capped is unreadable, and the
    /// error has to say so. `serde_json`'s own wording — "recursion limit
    /// exceeded at line 1 column 135" — names neither `uns` nor anything the
    /// user can act on, and this is the one JSON error they can act on.
    #[test]
    fn over_deep_uns_reports_itself_rather_than_serde_jargon() {
        let deep = format!("{}1{}", "[".repeat(500), "]".repeat(500));
        let err = parse_uns_json(deep.as_bytes()).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("uns nests deeper"), "got: {msg}");
        // Must stay true on both sides: the same variant is now raised by
        // `validate_uns_depth` before any file exists, so wording that claims
        // a pre-existing file would be wrong there.
        assert!(!msg.contains("This file was written"), "got: {msg}");
        assert!(
            msg.contains(&SERDE_JSON_MAX_NESTING.to_string()),
            "got: {msg}"
        );

        // The remap must be scoped to the depth case: every other malformed
        // payload still surfaces as an ordinary JSON error.
        assert!(matches!(parse_uns_json(b"{oops"), Err(ScxError::Json(_))));
        assert!(matches!(parse_uns_json(b""), Err(ScxError::Json(_))));
    }

    /// `UnsTooDeep` must not be classed `CorruptFile`.
    ///
    /// The bindings append "the file appears corrupt or was written by an
    /// incompatible SCX version; re-run conversion to regenerate it" to every
    /// `CorruptFile`, which contradicts this variant on both counts: the file
    /// is intact, and it was written by an *older* SCX, not a newer one. The
    /// whole point of splitting this out of `ScxError::Json` was to say
    /// something accurate, so the class is part of the contract.
    #[test]
    fn uns_too_deep_is_not_classed_as_a_corrupt_file() {
        let err = ScxError::UnsTooDeep {
            max_nesting: SERDE_JSON_MAX_NESTING,
        };
        assert_eq!(err.class(), ScxErrorClass::Validation);
    }

    /// `validate_uns_depth` is the last gate before bytes exist, so it has to
    /// survive inputs that would overflow a recursive checker — measuring the
    /// tree must not fail the way storing it would.
    #[test]
    fn validate_uns_depth_is_iterative_and_bounds_the_stored_tree() {
        let nest = |n: usize| {
            let mut v = serde_json::json!(1);
            for _ in 0..n {
                v = serde_json::Value::Array(vec![v]);
            }
            v
        };

        // Cross-check against the parser rather than hardcoding a boundary:
        // the invariant is that the write gate and the read path agree
        // *exactly*. An off-by-one either way is a bug — one direction stores
        // unreadable files, the other rejects readable ones — and a literal
        // in this test would only pin whichever mistake was made first.
        for n in (SERDE_JSON_MAX_NESTING - 2)..=(SERDE_JSON_MAX_NESTING + 2) {
            let tree = nest(n);
            let stored_ok = validate_uns_depth(&tree).is_ok();
            let parsed_ok = parse_uns_json(serde_json::to_vec(&tree).unwrap().as_slice()).is_ok();
            assert_eq!(
                stored_ok, parsed_ok,
                "depth {n}: write gate says ok={stored_ok}, parser says ok={parsed_ok}"
            );
        }

        // A recursive depth check dies well before here: the equivalent walk
        // in `scx-convert` aborted the test binary at depth 500. This one
        // returns an error instead, which is the whole point of the explicit
        // stack.
        //
        // Do not raise 5_000 much further. `serde_json::Value` has no manual
        // `Drop`, so *destroying* a tree recurses through drop glue — measured
        // here, 50_000 aborts this test with a stack overflow on the drop, not
        // inside `validate_uns_depth`. The ceiling is the 2 MB test thread, not
        // the function under test.
        assert!(matches!(
            validate_uns_depth(&nest(5_000)),
            Err(ScxError::UnsTooDeep { .. })
        ));

        // Depth is the longest root-to-leaf path, not a node count: a wide but
        // shallow tree must not trip the guard.
        let wide = serde_json::Value::Array((0..10_000).map(|i| serde_json::json!(i)).collect());
        assert!(validate_uns_depth(&wide).is_ok());
    }

    /// The derivation `MAX_UNS_DEPTH` is built on: under `uns_format="tagged"`
    /// a tuple costs two JSON levels, so the deepest legal tree is checked
    /// against the parser as the encoder would actually emit it.
    #[test]
    fn worst_case_tagged_encoding_at_max_depth_still_parses() {
        // `uns` root object, then MAX_UNS_DEPTH - 1 tuple envelopes.
        let mut json = String::from("1");
        for _ in 0..MAX_UNS_DEPTH - 1 {
            json = format!(r#"{{"__scx_type__":"tuple","data":[{json}]}}"#);
        }
        json = format!(r#"{{"deep":{json}}}"#);
        assert!(
            parse_uns_json(json.as_bytes()).is_ok(),
            "a tree at the write cap must be readable, or the cap is wrong"
        );
    }
}
