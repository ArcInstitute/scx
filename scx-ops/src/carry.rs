//! One declared answer to "which sections survive this op".
//!
//! Before this module the question was encoded four times in four incompatible
//! shapes — an allowlist of copy calls
//! ([`crate::rewrite_helpers::copy_auxiliary_sections`]), an inline `matches!`
//! over eight variants (`optimize`), per-type read/filter/write blocks
//! (`compact`), and two `should_drop_old_entry` denylists (`external_obs`,
//! `external_layer`). Nothing connected them, so `SectionType::DeletionVectors`
//! was handled in `optimize` and forgotten in `build_csc` and `merge` — one
//! missing invariant, found independently in four crates by the 2026-08-05
//! review.
//!
//! What this module changes is **not** the behaviour. Every entry below records
//! what the tree does today, including the drops that are open bugs. What it
//! changes is that the answer is now
//!
//! * **in one place**, so a reviewer reads a table instead of five call graphs;
//! * **exhaustive**, so a new section type cannot default to "dropped"; and
//! * **checked**, via [`audit`] — an op that stops writing a section it
//!   declares it carries fails loudly instead of shipping.
//!
//! # The compile-time tripwire, precisely
//!
//! Measured, not assumed — a 30th `SectionType` and a 21st [`SectionFamily`]
//! were each added to a scratch tree and the errors counted:
//!
//! * A new **`SectionType`** produces **one** error, in [`family`]. That is the
//!   whole intent: most new types are another physical encoding of something
//!   that already exists (a third `obsp` layout, a row-sharded twin), and once
//!   it is assigned to an existing family every op already handles it right.
//!   The tripwire's job is to make that assignment *mandatory*, not to force
//!   six edits for one.
//! * A new **`SectionFamily`** — a genuinely new thing to decide about —
//!   produces **six** errors: [`SectionFamily::label`] plus the five explicit
//!   policies. [`upgrade`] is the deliberate exception; see its doc comment.
//!
//! So the accurate claim is not "adding a section type breaks six matches". It
//! is that nothing can reach an op without a declared family, and no new
//! *decision* can reach an op without a declared answer.
//!
//! # What this module does not cover
//!
//! **The two `should_drop_old_entry` denylists.** The in-place ops
//! (`obs_import`, `cellbender_import`, `append`, `modify_metadata`) start from
//! the old catalog and drop selectively, which is the mirror image of the
//! rewrite ops here and maps onto the same table — but it is a separate
//! adoption and is deliberately not attempted here. Until it lands, "the single
//! truth" is true of the six ops in [`RewriteOp`] and of nothing else.
//!
//! **Whether a rebuild actually happened.** See [`Carry::Rebuilt`].

use scx_format_io::catalog::FullCatalog;
use scx_format_io::section::SectionType;

use crate::error::{OpsError, Result as OpsResult};

/// The ~20 things an op actually decides about.
///
/// The 29 section types do not each get their own decision: a legacy single
/// section and its row-shard twin are one decision (`ObsMetadata` /
/// `ObsMetadataShard`), and three physical encodings of `obsp` are still just
/// "obsp". Keying the policy on the family rather than the type is not
/// cosmetic — `compact` can read a legacy `ObsMetadata` input and emit
/// `ObsMetadataShard`s, so a type-keyed audit would report a section that
/// "vanished" on a perfectly correct rewrite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SectionFamily {
    /// obs table: `ObsMetadata` + `ObsMetadataShard`.
    ObsMetadata,
    /// var table: `VarMetadata` + `VarMetadataShard`.
    VarMetadata,
    /// The main count matrix, `CsrShard`.
    X,
    /// The column-major sidecar for X, `CscShard`.
    XCsc,
    /// `LayerCsrShard`.
    Layer,
    /// `LayerCscShard`.
    LayerCsc,
    /// `ObsmEmbedding` + `ObsmEmbeddingShard`.
    Obsm,
    /// `VarmEmbedding` + `VarmEmbeddingShard`.
    Varm,
    /// `ObspCsrShard` + `ObspEmbedding` + `ObspEmbeddingShard`.
    Obsp,
    /// `VarpEmbedding` + `VarpEmbeddingShard`.
    Varp,
    /// `UnsBlob`.
    Uns,
    /// `Provenance`.
    Provenance,
    /// `DeletionVectors`.
    DeletionVectors,
    /// `BitmapShard` — the per-shard gene→local-row detection bitmaps.
    Bitmap,
    /// `ObsPredicateIndex`.
    ObsPredicateIndex,
    /// `VarPredicateIndex`.
    VarPredicateIndex,
    /// `ModalityTable`.
    ModalityTable,
    /// `RawCsrShard` + `RawVarMetadata` — `adata.raw`.
    Raw,
    /// `GroupIndex` — the `scx sort --group-by` sidecar.
    GroupIndex,
    /// `ObsIndex` (id 1) and `VarIndex` (id 3): declared in the format,
    /// written by nothing in this workspace.
    ///
    /// Deliberately **not** folded into `ObsMetadata` / `VarMetadata`, because
    /// we do not know what a foreign writer would have put in them. Silently
    /// dropping a section nobody here understands is precisely the bug class
    /// this module exists for, so [`audit`] warns rather than the table
    /// pretending the ids do not exist.
    Unwritten,
}

impl SectionFamily {
    /// Every family, in declaration order. Used by the table snapshot test and
    /// by [`audit`]; a new family added to the enum and not to this list is
    /// caught by `families_list_is_complete`.
    pub const ALL: &'static [SectionFamily] = &[
        SectionFamily::ObsMetadata,
        SectionFamily::VarMetadata,
        SectionFamily::X,
        SectionFamily::XCsc,
        SectionFamily::Layer,
        SectionFamily::LayerCsc,
        SectionFamily::Obsm,
        SectionFamily::Varm,
        SectionFamily::Obsp,
        SectionFamily::Varp,
        SectionFamily::Uns,
        SectionFamily::Provenance,
        SectionFamily::DeletionVectors,
        SectionFamily::Bitmap,
        SectionFamily::ObsPredicateIndex,
        SectionFamily::VarPredicateIndex,
        SectionFamily::ModalityTable,
        SectionFamily::Raw,
        SectionFamily::GroupIndex,
        SectionFamily::Unwritten,
    ];

    /// A stable label for log messages and for the table snapshot.
    pub fn label(self) -> &'static str {
        match self {
            Self::ObsMetadata => "obs",
            Self::VarMetadata => "var",
            Self::X => "X",
            Self::XCsc => "X CSC sidecar",
            Self::Layer => "layers",
            Self::LayerCsc => "layer CSC sidecars",
            Self::Obsm => "obsm",
            Self::Varm => "varm",
            Self::Obsp => "obsp",
            Self::Varp => "varp",
            Self::Uns => "uns",
            Self::Provenance => "provenance",
            Self::DeletionVectors => "deletion vectors",
            Self::Bitmap => "detection bitmaps",
            Self::ObsPredicateIndex => "obs predicate index",
            Self::VarPredicateIndex => "var predicate index",
            Self::ModalityTable => "modality table",
            Self::Raw => "adata.raw",
            Self::GroupIndex => "grouped-sort group index",
            Self::Unwritten => "unwritten legacy index section",
        }
    }
}

/// Map a section type onto the decision it belongs to.
///
/// **This match has no `_` arm on purpose.** It is the compile-time tripwire: a
/// new `SectionType` variant lands here first, and until its family is declared
/// nothing in `scx-ops` builds. A new *family* then propagates into five of the
/// six policies for the same reason — see the module docs for the measured
/// counts and for why `upgrade` is the exception.
pub fn family(ty: SectionType) -> SectionFamily {
    match ty {
        SectionType::ObsMetadata | SectionType::ObsMetadataShard => SectionFamily::ObsMetadata,
        SectionType::VarMetadata | SectionType::VarMetadataShard => SectionFamily::VarMetadata,
        SectionType::CsrShard => SectionFamily::X,
        SectionType::CscShard => SectionFamily::XCsc,
        SectionType::LayerCsrShard => SectionFamily::Layer,
        SectionType::LayerCscShard => SectionFamily::LayerCsc,
        SectionType::ObsmEmbedding | SectionType::ObsmEmbeddingShard => SectionFamily::Obsm,
        SectionType::VarmEmbedding | SectionType::VarmEmbeddingShard => SectionFamily::Varm,
        SectionType::ObspCsrShard
        | SectionType::ObspEmbedding
        | SectionType::ObspEmbeddingShard => SectionFamily::Obsp,
        SectionType::VarpEmbedding | SectionType::VarpEmbeddingShard => SectionFamily::Varp,
        SectionType::UnsBlob => SectionFamily::Uns,
        SectionType::Provenance => SectionFamily::Provenance,
        SectionType::DeletionVectors => SectionFamily::DeletionVectors,
        SectionType::BitmapShard => SectionFamily::Bitmap,
        SectionType::ObsPredicateIndex => SectionFamily::ObsPredicateIndex,
        SectionType::VarPredicateIndex => SectionFamily::VarPredicateIndex,
        SectionType::ModalityTable => SectionFamily::ModalityTable,
        SectionType::RawCsrShard | SectionType::RawVarMetadata => SectionFamily::Raw,
        SectionType::GroupIndex => SectionFamily::GroupIndex,
        SectionType::ObsIndex | SectionType::VarIndex => SectionFamily::Unwritten,
    }
}

/// What an op does with a family **that is present in its input**.
///
/// Every variant is conditional on presence, which is why there is no
/// `NotApplicable`: a policy is never consulted for a family the input does not
/// carry, so `ModalityTable` on a unimodal file and `Raw` on a file without
/// `.raw` need no special case.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carry {
    /// Copied through unchanged — byte-for-byte, or re-serialised with
    /// identical content. Must be present in the output.
    Verbatim,
    /// Rows filtered through the obs keep-mask. Must be present in the output
    /// unless the output has no rows left.
    RowFiltered,
    /// Content rewritten into the output's coordinate space: obsp COO remapped
    /// through a keep-mask, rows permuted by a sort, deletion-vector row ids
    /// rebased across a merge. Must be present in the output.
    Remapped,
    /// Recomputed from the output rather than copied.
    ///
    /// **Asserts nothing about the output.** A predicate index is legitimately
    /// absent when no index was requested; a merged `uns` is legitimately
    /// absent under [`crate::merge_options::UnsPolicy`]; and a catalog cannot
    /// show whether the bytes that *are* present were rebuilt or copied stale.
    ///
    /// So `Rebuilt` is the weakest entry in the table and it is the one to be
    /// careful about: it records intent, and the intent is not enforced here.
    /// The known live instance is `build-csc`, which copies the predicate-index
    /// sections and never re-derives the per-shard `column_stats` they need, so
    /// the sections survive and Level-1 pushdown does not — pinned, unfixed, at
    /// `scx-cli/tests/cli_ops_integration.rs`
    /// (`test_build_csc_preserves_predicate_index_sections_but_not_pushdown`).
    Rebuilt,
    /// Carried or dropped depending on something decided at run time, so
    /// neither presence nor absence can be asserted.
    ///
    /// Distinct from [`Carry::Rebuilt`] on purpose. `Rebuilt` means "the output's
    /// copy is derived from the output"; `Conditional` means "the op looked at
    /// this run and chose". Collapsing them would hide a real difference behind
    /// the same non-assertion — `on` records the actual condition, so a reader
    /// of the table learns which runs keep it.
    ///
    /// Introduced because the first table said `upgrade` drops the CSC sidecar
    /// and the audit immediately disagreed: `upgrade` re-emits CSC **unless**
    /// canonicalising actually changed the matrix, in which case the old sidecar
    /// would be a second view that disagrees with the first, and it is dropped
    /// with a warning.
    Conditional { on: &'static str },
    /// Deliberately not carried. Must be **absent** from the output.
    ///
    /// `warns` records whether the op tells the user *today*. It is part of the
    /// description, not an aspiration — several of these drops are silent, and
    /// recording that truthfully is what makes the anomaly visible in one table
    /// instead of invisible across five call graphs.
    Dropped { why: &'static str, warns: bool },
    /// The op refuses to run at all on an input carrying this family, and does
    /// so before writing anything. Reaching [`audit`] with one means the guard
    /// did not fire, which is a bug in the guard.
    Refuse,
}

impl Carry {
    /// Short tag for the table snapshot. Deliberately terse: the snapshot is
    /// read as a grid, and a grid whose cells wrap is not read.
    pub fn tag(self) -> &'static str {
        match self {
            Self::Verbatim => "verbatim",
            Self::RowFiltered => "row-filtered",
            Self::Remapped => "remapped",
            Self::Rebuilt => "rebuilt",
            Self::Conditional { .. } => "conditional",
            Self::Dropped { warns: true, .. } => "dropped(warns)",
            Self::Dropped { warns: false, .. } => "dropped(SILENT)",
            Self::Refuse => "refuse",
        }
    }

    /// Whether the output is required to carry the family.
    fn requires_presence(self) -> bool {
        matches!(self, Self::Verbatim | Self::RowFiltered | Self::Remapped)
    }
}

/// The ops that rewrite a whole file through an `ScxWriter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RewriteOp {
    Compact,
    Merge,
    Optimize,
    Sort,
    BuildCsc,
    Upgrade,
}

impl RewriteOp {
    /// Every op, in declaration order — for the table snapshot.
    pub const ALL: &'static [RewriteOp] = &[
        RewriteOp::Compact,
        RewriteOp::Merge,
        RewriteOp::Optimize,
        RewriteOp::Sort,
        RewriteOp::BuildCsc,
        RewriteOp::Upgrade,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Compact => "compact",
            Self::Merge => "merge",
            Self::Optimize => "optimize",
            Self::Sort => "sort",
            Self::BuildCsc => "build-csc",
            Self::Upgrade => "upgrade",
        }
    }
}

/// The table.
///
/// Read it as: *if the input carries this family, this is what the op does with
/// it.* Entries marked `Dropped { warns: false }` are silent data loss; the
/// four flagged `§6.3` / `§6.4` below are open Majors from the 2026-08-05
/// review, declared here rather than fixed so the fix arrives as a visible diff
/// against a stated baseline.
pub fn policy(op: RewriteOp, family: SectionFamily) -> Carry {
    match op {
        RewriteOp::Compact => compact(family),
        RewriteOp::Merge => merge(family),
        RewriteOp::Optimize => optimize(family),
        RewriteOp::Sort => sort(family),
        RewriteOp::BuildCsc => build_csc(family),
        RewriteOp::Upgrade => upgrade(family),
    }
}

/// `compact` applies the deletion vector and re-shards the surviving rows.
fn compact(family: SectionFamily) -> Carry {
    use SectionFamily as F;
    match family {
        F::ObsMetadata | F::X | F::Layer | F::Obsm => Carry::RowFiltered,
        // Both COO axes are the obs axis, so entries touching a deleted row go
        // too — that is a remap, not a filter.
        F::Obsp => Carry::Remapped,
        // var-axis data; compact never drops columns.
        F::VarMetadata | F::Varm | F::Varp | F::Uns => Carry::Verbatim,
        F::Provenance => Carry::Rebuilt,
        // Rebuilt against the post-deletion obs and the freshly emitted shard
        // row ranges.
        F::ObsPredicateIndex | F::VarPredicateIndex => Carry::Rebuilt,
        // Derived by the writer from the sections actually written, not copied.
        F::ModalityTable => Carry::Rebuilt,
        // Applied, not carried: the surviving rows are the output's rows.
        F::DeletionVectors => Carry::Dropped {
            why: "applied — the output contains only the surviving rows",
            warns: false,
        },
        F::XCsc => Carry::Dropped {
            why: "re-sharding invalidates column-major offsets (rebuild: scx build-csc)",
            warns: true,
        },
        F::LayerCsc => Carry::Dropped {
            why: "re-sharding invalidates column-major offsets (rebuild: scx build-csc)",
            warns: false,
        },
        // Correct to drop — bitmap row keys are shard-local and compact
        // re-shards — but silent, which §6.13 asks for a warning about.
        F::Bitmap => Carry::Dropped {
            why: "shard-local row keys are invalidated by re-sharding (rebuild: --bitmap)",
            warns: false,
        },
        F::GroupIndex => Carry::Dropped {
            why: "records global output-row ranges, which re-sharding invalidates",
            warns: false,
        },
        F::Raw => Carry::Dropped {
            why: "raw's obs axis is not filtered in lockstep with X (planned follow-up)",
            warns: true,
        },
        F::Unwritten => Carry::Dropped {
            why: "no writer in this workspace produces it",
            warns: false,
        },
    }
}

/// `merge` concatenates row spaces, so nothing obs-axis survives verbatim.
fn merge(family: SectionFamily) -> Carry {
    use SectionFamily as F;
    match family {
        // Concatenated across inputs under a unified schema.
        F::ObsMetadata | F::X | F::Layer | F::Obsm => Carry::Rebuilt,
        // Validated equal across inputs (or assumed so), then written once.
        F::VarMetadata | F::Varm => Carry::Verbatim,
        // Row ids rebased by each input's offset in the concatenated space.
        F::DeletionVectors => Carry::Remapped,
        // Combined under `UnsPolicy`, which can legitimately yield nothing.
        F::Uns => Carry::Rebuilt,
        F::Provenance => Carry::Rebuilt,
        F::ObsPredicateIndex | F::VarPredicateIndex => Carry::Rebuilt,
        F::ModalityTable => Carry::Rebuilt,
        // §6.4 — OPEN MAJOR. `compact` remaps obsp through its keep-mask and
        // `sort` through its permutation, so a merged kNN graph vanishes where
        // a compacted or sorted one survives, with no warning. Phase 5b.
        F::Obsp => Carry::Dropped {
            why: "§6.4 (open): merge has no obsp writer; a merged graph is lost silently",
            warns: false,
        },
        // §6.4 — OPEN MAJOR, same shape on the var axis.
        F::Varp => Carry::Dropped {
            why: "§6.4 (open): merge has no varp writer; lost silently",
            warns: false,
        },
        F::XCsc => Carry::Dropped {
            why: "output row space differs from every input's (rebuild: scx build-csc)",
            warns: true,
        },
        F::LayerCsc => Carry::Dropped {
            why: "output row space differs from every input's (rebuild: scx build-csc)",
            warns: false,
        },
        F::Bitmap => Carry::Dropped {
            why: "shard-local row keys do not survive concatenation (rebuild: --bitmap)",
            warns: false,
        },
        F::GroupIndex => Carry::Dropped {
            why: "records one input's global row ranges, meaningless after concatenation",
            warns: false,
        },
        F::Raw => Carry::Dropped {
            why: "raw carries its own var axis, which merge does not reconcile",
            warns: true,
        },
        F::Unwritten => Carry::Dropped {
            why: "no writer in this workspace produces it",
            warns: false,
        },
    }
}

/// `optimize` re-encodes shards in place: row order and shard boundaries are
/// preserved 1:1, which is why it can carry more than any other op.
fn optimize(family: SectionFamily) -> Carry {
    use SectionFamily as F;
    match family {
        F::ObsMetadata | F::VarMetadata => Carry::Verbatim,
        // Re-encoded (canonicalised), same rows in the same order.
        F::X | F::Layer => Carry::Verbatim,
        F::Obsm | F::Varm | F::Obsp | F::Varp | F::Uns => Carry::Verbatim,
        // Row order and CSR shard boundaries are 1:1, so shard-local bitmap row
        // keys and the group index's global ranges both stay valid.
        F::Bitmap | F::GroupIndex => Carry::Verbatim,
        F::DeletionVectors => Carry::Verbatim,
        F::ObsPredicateIndex | F::VarPredicateIndex => Carry::Verbatim,
        F::Provenance => Carry::Rebuilt,
        // Rejected up front with a message pointing at `scx compact`.
        F::ModalityTable => Carry::Refuse,
        F::XCsc => Carry::Dropped {
            why: "re-canonicalisation can change nnz, staleing column offsets (rebuild: scx build-csc)",
            warns: true,
        },
        F::LayerCsc => Carry::Dropped {
            why: "re-canonicalisation can change nnz, staleing column offsets (rebuild: scx build-csc)",
            warns: false,
        },
        F::Raw => Carry::Dropped {
            why: "not in optimize's section allowlist (planned follow-up)",
            warns: true,
        },
        F::Unwritten => Carry::Dropped {
            why: "no writer in this workspace produces it",
            warns: false,
        },
    }
}

/// `sort` applies the deletion vector and permutes the surviving rows.
fn sort(family: SectionFamily) -> Carry {
    use SectionFamily as F;
    match family {
        F::ObsMetadata | F::X | F::Layer | F::Obsm | F::Obsp => Carry::Remapped,
        F::VarMetadata | F::Varm | F::Varp | F::Uns => Carry::Verbatim,
        // Rebuilt against the permuted rows when `--bitmap auto/always` asks
        // for it, and dropped otherwise — which is the default. Shard-local
        // row keys cannot survive a permutation, so there is no third option:
        // sort either regenerates the sidecar or loses it.
        F::Bitmap => Carry::Conditional {
            on: "rebuilt under --bitmap auto/always; dropped otherwise (the default)",
        },
        F::Provenance => Carry::Rebuilt,
        F::ObsPredicateIndex | F::VarPredicateIndex => Carry::Rebuilt,
        F::ModalityTable => Carry::Rebuilt,
        // Written afresh by `--group-by`; a pre-existing one describes the
        // input's row ranges and cannot survive a permutation, so a sort
        // without `--group-by` drops it.
        F::GroupIndex => Carry::Conditional {
            on: "written by --group-by; dropped otherwise",
        },
        F::DeletionVectors => Carry::Dropped {
            why: "applied — the output contains only the surviving rows",
            warns: false,
        },
        F::XCsc => Carry::Dropped {
            why: "permuting rows invalidates column-major offsets (rebuild: scx build-csc)",
            warns: true,
        },
        F::LayerCsc => Carry::Dropped {
            why: "permuting rows invalidates column-major offsets (rebuild: scx build-csc)",
            warns: false,
        },
        F::Raw => Carry::Dropped {
            why: "raw's obs axis is not permuted in lockstep with X (planned follow-up)",
            warns: true,
        },
        F::Unwritten => Carry::Dropped {
            why: "no writer in this workspace produces it",
            warns: false,
        },
    }
}

/// `build-csc` rewrites the file 1:1 and appends a column-major sidecar. Its
/// carry set is [`crate::rewrite_helpers::copy_auxiliary_sections`]'s allowlist,
/// which is the narrowest of any op here — and its output renames over the
/// input with no prior catalog, so `scx rollback` cannot recover what it drops.
fn build_csc(family: SectionFamily) -> Carry {
    use SectionFamily as F;
    match family {
        F::ObsMetadata | F::VarMetadata => Carry::Verbatim,
        F::X | F::Layer | F::Obsm | F::Uns => Carry::Verbatim,
        F::DeletionVectors => Carry::Verbatim,
        // Sections copied through; see `Carry::Rebuilt` for why this entry is
        // the weak one — the per-shard `column_stats` those sections need are
        // not re-derived, so Level-1 pushdown is lost while the section stays.
        F::ObsPredicateIndex | F::VarPredicateIndex => Carry::Verbatim,
        // The whole point of the op.
        F::XCsc => Carry::Rebuilt,
        F::Provenance => Carry::Rebuilt,
        // Rejected up front, pointing at `scx subset --modality`.
        F::ModalityTable => Carry::Refuse,
        // §6.3 — OPEN MAJOR (all four below). `optimize` carries every one of
        // these and `compact` carries varm/varp. Phase 5b.
        F::Varm => Carry::Dropped {
            why: "§6.3 (open): not in the copy allowlist; user data with no rebuild path",
            warns: true,
        },
        F::Obsp => Carry::Dropped {
            why: "§6.3 (open): not in the copy allowlist; user data with no rebuild path",
            warns: true,
        },
        F::Varp => Carry::Dropped {
            why: "§6.3 (open): not in the copy allowlist; user data with no rebuild path",
            warns: true,
        },
        F::Raw => Carry::Dropped {
            why: "§6.3 (open): not in the copy allowlist",
            warns: true,
        },
        F::Bitmap => Carry::Dropped {
            why: "§6.3 (open): not in the copy allowlist (rebuild: --bitmap)",
            warns: true,
        },
        F::GroupIndex => Carry::Dropped {
            why: "§6.3 (open): not in the copy allowlist",
            warns: true,
        },
        F::LayerCsc => Carry::Dropped {
            why: "not in the copy allowlist (rebuild: scx build-csc)",
            warns: true,
        },
        F::Unwritten => Carry::Dropped {
            why: "no writer in this workspace produces it",
            warns: false,
        },
    }
}

/// `scx upgrade` shares `build_csc`'s copy allowlist, with layers canonicalised
/// on the way through. It emits no CSC sidecar, which is the only difference
/// the table can see.
///
/// **The `other =>` arm is deliberate, and it is the one place the compile-time
/// tripwire does not fire.** A new [`SectionFamily`] reaches `upgrade` carrying
/// `build_csc`'s answer rather than a compile error. That is sound *because the
/// two ops share one implementation* —
/// [`crate::rewrite_helpers::copy_auxiliary_sections_canonicalizing`] is the
/// carry for both, and it differs only in whether layers are canonicalised,
/// which is not a carry decision. Writing nineteen `F::X => build_csc(family)`
/// arms to buy the error back would state the delegation less clearly and would
/// let the two drift, which is the disease. `upgrade_matches_build_csc` pins the
/// relationship instead.
fn upgrade(family: SectionFamily) -> Carry {
    use SectionFamily as F;
    match family {
        // Re-emitted from the input's own CSC shards — but only when
        // canonicalising left the CSR matrix alone. If it summed a duplicate
        // coordinate or dropped an explicit zero, the sidecar built against the
        // old matrix is no longer a faithful second view of it, and carrying it
        // forward would leave the file holding two matrices that disagree,
        // under a `csc_build_generation` fresh enough to bless the stale one.
        // Dropped with a warning in exactly that case, and only that case.
        F::XCsc => Carry::Conditional {
            on: "carried unless canonicalisation changed the CSR matrix",
        },
        other => build_csc(other),
    }
}

/// What [`audit`] observed. Assertable directly, so a test can pin the whole
/// surviving set rather than spot-checking one family — a spot check passes
/// just as happily when everything else vanished.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CarryReport {
    /// Families present in some input and absent from the output, sorted.
    pub dropped: Vec<SectionFamily>,
    /// Families present in some input and present in the output, sorted.
    pub carried: Vec<SectionFamily>,
}

/// Check an op's output against [`policy`].
///
/// Only families **present in an input** are considered, so every table entry
/// reads "if present, then…" and no conditional family needs a special case.
///
/// `output_n_obs` exists for one rule: a [`Carry::RowFiltered`] family may
/// legitimately be absent when every row was deleted. Passing the output's row
/// count rather than inferring it keeps that exemption narrow — an op that
/// wrote no rows for some *other* reason still fails.
///
/// Fails closed: a violation is an error, not a warning, matching how
/// `assign_csr_shard_column_stats` treats a shard-count mismatch. A rewrite that
/// has silently lost a section has already produced the file; the only useful
/// moment to say so is before the caller believes it succeeded.
pub fn audit(
    op: RewriteOp,
    inputs: &[&FullCatalog],
    output: &FullCatalog,
    output_n_obs: u64,
) -> OpsResult<CarryReport> {
    let present = |catalog: &FullCatalog, f: SectionFamily| {
        catalog.entries.iter().any(|e| family(e.section_type) == f)
    };

    let mut dropped = Vec::new();
    let mut carried = Vec::new();

    for &f in SectionFamily::ALL {
        if !inputs.iter().any(|c| present(c, f)) {
            continue;
        }
        let in_output = present(output, f);
        let rule = policy(op, f);

        match rule {
            Carry::Refuse => {
                return Err(OpsError::SectionCarryViolation {
                    op: op.label(),
                    family: f.label(),
                    detail: "op declares it refuses this input, but the guard did not fire"
                        .to_string(),
                });
            }
            // `warns` is not consulted here: the warnings live at their existing
            // sites in each op, where the input path and the remedy are in
            // scope. The flag is part of the table's description of today, and
            // what reads it is the snapshot test — where `dropped(SILENT)` is
            // legible next to `dropped(warns)` as the anomaly it is.
            Carry::Dropped { why, warns: _ } => {
                if in_output {
                    return Err(OpsError::SectionCarryViolation {
                        op: op.label(),
                        family: f.label(),
                        detail: format!("declared dropped ({why}) but the output carries it"),
                    });
                }
                if f == SectionFamily::Unwritten {
                    log::warn!(
                        "scx {}: input carries a legacy `obs_index` / `var_index` section, which \
                         no writer in this workspace produces and which this op drops. If it \
                         holds data you need, extract it before rewriting.",
                        op.label()
                    );
                }
                dropped.push(f);
            }
            _ if rule.requires_presence() => {
                // A row-filtered family may vanish legitimately, but only when
                // there is nothing left to carry.
                let exempt = rule == Carry::RowFiltered && output_n_obs == 0;
                if !in_output && !exempt {
                    return Err(OpsError::SectionCarryViolation {
                        op: op.label(),
                        family: f.label(),
                        detail: format!("declared {} but the output does not carry it", rule.tag()),
                    });
                }
                if in_output {
                    carried.push(f);
                } else {
                    dropped.push(f);
                }
            }
            // `Rebuilt` asserts nothing — see its doc comment.
            _ => {
                if in_output {
                    carried.push(f);
                } else {
                    dropped.push(f);
                }
            }
        }
    }

    dropped.sort();
    carried.sort();
    Ok(CarryReport { dropped, carried })
}

/// [`audit`] against a finished file on disk.
///
/// The ops call this **after** `ScxWriter::finish()`, not before. `finish()`
/// consumes the writer and its catalog is private, so auditing earlier would
/// mean widening `scx-format-io`'s public API — and it would check the writer's
/// intent rather than the artifact. Reopening costs one catalog parse per op
/// invocation, which is not on any per-row or per-shard path.
///
/// Callers that rename the output over the input (`build-csc`, `optimize
/// --output == input`) must have captured their input catalogs before writing;
/// all of them do, since every one opens its reader first.
pub fn audit_output(
    op: RewriteOp,
    inputs: &[&FullCatalog],
    output_path: &std::path::Path,
) -> OpsResult<CarryReport> {
    let out = scx_format_io::ScxReader::open(output_path)?;
    audit(op, inputs, out.catalog(), out.header().n_obs)
}

/// Render the whole table as text, one line per (op, family) with a non-default
/// outcome, so a policy change is a **visible diff in a test file** rather than
/// a silent edit inside a `match`.
pub fn render_table() -> String {
    let mut out = String::new();
    for &op in RewriteOp::ALL {
        out.push_str(op.label());
        out.push('\n');
        for &f in SectionFamily::ALL {
            out.push_str(&format!("  {:<28} {}\n", f.label(), policy(op, f).tag()));
        }
    }
    out
}

#[cfg(test)]
#[path = "carry_tests.rs"]
mod tests;
