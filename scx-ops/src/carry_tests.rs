//! Tests for the section-carry table itself.
//!
//! These do not run any op — they check the table is total and that a change to
//! it is visible. Whether the table matches what the ops actually do is
//! `tests/scx-integration-tests/tests/section_carry.rs`'s job, and it is the
//! more important half: a table nothing checks is just a comment.

use super::*;

/// Every section id the format defines maps to a family.
///
/// Driven off `SectionType::from_u8` rather than off a hand-written list of
/// variants, so a new id is caught even by a future edit that adds a `_` arm to
/// `family()` — the tripwire this test exists to back up is a compile error,
/// and a compile error can be defeated by the one edit nobody reviews.
#[test]
fn every_section_id_has_a_family() {
    let mut seen = 0;
    for id in 0u8..=29 {
        match SectionType::from_u8(id) {
            Some(ty) => {
                // Must not panic, and must land in `ALL`.
                let f = family(ty);
                assert!(
                    SectionFamily::ALL.contains(&f),
                    "section id {id} maps to {f:?}, which is missing from SectionFamily::ALL"
                );
                seen += 1;
            }
            // 26 is the reserved hole (formerly DecodeMetadataShard).
            None => assert_eq!(id, 26, "unexpected gap in the SectionType id space at {id}"),
        }
    }
    assert_eq!(seen, 29, "expected 29 live section types, found {seen}");
}

/// `SectionFamily::ALL` is what `audit` iterates, so a family missing from it is
/// a family the audit silently never checks.
#[test]
fn families_list_is_complete() {
    // Every family reachable from a section type must be in ALL — covered
    // above. This side catches the reverse: a family in ALL twice, which would
    // double-count it in a report.
    let mut sorted = SectionFamily::ALL.to_vec();
    sorted.sort();
    let len_before = sorted.len();
    sorted.dedup();
    assert_eq!(
        len_before,
        sorted.len(),
        "duplicate entry in SectionFamily::ALL"
    );
}

/// Every (op, family) pair resolves. Exhaustiveness of the `match`es is a
/// compile-time property; this catches a policy that panics or is unreachable.
#[test]
fn table_is_total() {
    for &op in RewriteOp::ALL {
        for &f in SectionFamily::ALL {
            let _ = policy(op, f).tag();
        }
    }
}

/// The whole table, pinned.
///
/// The point is not that these values are *right*. The point is that changing
/// one shows up here, in a diff a reviewer reads, instead of inside a `match`
/// arm in a 600-line module.
///
/// It has already earned that once: Phase 5b flipped ten cells — `merge`'s
/// obsp/varp and `build-csc`'s varm/obsp/varp/raw/bitmaps/group-index — and the
/// diff to this string is the most legible artifact that PR produced.
#[test]
fn table_snapshot() {
    let expected = "\
compact
  obs                          row-filtered
  var                          verbatim
  X                            row-filtered
  X CSC sidecar                conditional
  layers                       row-filtered
  layer CSC sidecars           dropped(SILENT)
  obsm                         row-filtered
  varm                         verbatim
  obsp                         remapped
    (per-modality)             dropped(warns)
  obsp (CSR-backed)            dropped(SILENT)
    (per-modality)             dropped(warns)
  varp                         verbatim
    (per-modality)             dropped(warns)
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             dropped(SILENT)
  detection bitmaps            dropped(SILENT)
  obs predicate index          rebuilt
  var predicate index          rebuilt
  modality table               rebuilt
  adata.raw                    dropped(warns)
  grouped-sort group index     dropped(SILENT)
  unwritten legacy index section dropped(SILENT)
merge
  obs                          rebuilt
  var                          verbatim
  X                            rebuilt
  X CSC sidecar                conditional
  layers                       rebuilt
  layer CSC sidecars           dropped(SILENT)
  obsm                         rebuilt
  varm                         conditional
  obsp                         remapped
    (per-modality)             dropped(warns)
  obsp (CSR-backed)            dropped(warns)
    (per-modality)             dropped(warns)
  varp                         conditional
    (per-modality)             dropped(warns)
  uns                          rebuilt
  provenance                   rebuilt
  deletion vectors             remapped
  detection bitmaps            dropped(SILENT)
  obs predicate index          rebuilt
  var predicate index          rebuilt
  modality table               rebuilt
  adata.raw                    dropped(warns)
  grouped-sort group index     dropped(SILENT)
  unwritten legacy index section dropped(SILENT)
optimize
  obs                          verbatim
  var                          verbatim
  X                            verbatim
  X CSC sidecar                conditional
  layers                       verbatim
  layer CSC sidecars           dropped(SILENT)
  obsm                         verbatim
  varm                         verbatim
  obsp                         verbatim
  obsp (CSR-backed)            verbatim
  varp                         verbatim
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             verbatim
  detection bitmaps            conditional
  obs predicate index          verbatim
  var predicate index          verbatim
  modality table               refuse
  adata.raw                    dropped(warns)
  grouped-sort group index     verbatim
  unwritten legacy index section dropped(SILENT)
sort
  obs                          remapped
  var                          verbatim
  X                            remapped
  X CSC sidecar                conditional
  layers                       remapped
  layer CSC sidecars           dropped(SILENT)
  obsm                         remapped
  varm                         verbatim
  obsp                         remapped
    (per-modality)             dropped(SILENT)
  obsp (CSR-backed)            dropped(SILENT)
    (per-modality)             dropped(SILENT)
  varp                         verbatim
    (per-modality)             dropped(SILENT)
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             dropped(SILENT)
  detection bitmaps            conditional
  obs predicate index          rebuilt
  var predicate index          rebuilt
  modality table               rebuilt
  adata.raw                    dropped(warns)
  grouped-sort group index     conditional
  unwritten legacy index section dropped(SILENT)
build-csc
  obs                          verbatim
  var                          verbatim
  X                            verbatim
  X CSC sidecar                rebuilt
  layers                       verbatim
  layer CSC sidecars           conditional
  obsm                         verbatim
  varm                         verbatim
  obsp                         verbatim
  obsp (CSR-backed)            verbatim
  varp                         verbatim
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             verbatim
  detection bitmaps            verbatim
  obs predicate index          verbatim
  var predicate index          verbatim
  modality table               refuse
  adata.raw                    verbatim
  grouped-sort group index     verbatim
  unwritten legacy index section verbatim
upgrade
  obs                          verbatim
  var                          verbatim
  X                            verbatim
  X CSC sidecar                conditional
  layers                       verbatim
  layer CSC sidecars           dropped(warns)
  obsm                         verbatim
  varm                         verbatim
  obsp                         verbatim
  obsp (CSR-backed)            verbatim
  varp                         verbatim
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             verbatim
  detection bitmaps            conditional
  obs predicate index          verbatim
  var predicate index          verbatim
  modality table               refuse
  adata.raw                    verbatim
  grouped-sort group index     verbatim
  unwritten legacy index section dropped(SILENT)
";
    assert_eq!(render_table(), expected);
}

/// `upgrade` and `build-csc` agree on every family except four, and this pins
/// which four.
///
/// They no longer share a mechanism — `upgrade` rewrites through an allowlist,
/// `build-csc` appends in place — so agreement elsewhere is a fact about the
/// two ops today, not a delegation. Pinned anyway, because a divergence that
/// appears without a line here is exactly the kind of drift the table exists
/// to make visible.
#[test]
fn upgrade_matches_build_csc_except_four_families() {
    for &f in SectionFamily::ALL {
        match f {
            // build-csc *creates* the sidecar; upgrade re-emits the input's,
            // and only when canonicalising did not change the matrix under it.
            SectionFamily::XCsc => {
                assert_eq!(policy(RewriteOp::BuildCsc, f), Carry::Rebuilt);
                assert!(matches!(
                    policy(RewriteOp::Upgrade, f),
                    Carry::Conditional { .. }
                ));
            }
            // Only `upgrade` canonicalises, and canonicalisation drops explicit
            // zeros, which changes which genes a row *stores* and therefore
            // what the bitmap should say.
            SectionFamily::Bitmap => {
                assert_eq!(policy(RewriteOp::BuildCsc, f), Carry::Verbatim);
                assert!(matches!(
                    policy(RewriteOp::Upgrade, f),
                    Carry::Conditional { .. }
                ));
            }
            // An append keeps a fresh layer sidecar by never touching it (and
            // drops a stale one its new stamp would bless); a rewrite through
            // the allowlist has no way to bring it along at all.
            SectionFamily::LayerCsc => {
                assert!(matches!(
                    policy(RewriteOp::BuildCsc, f),
                    Carry::Conditional { .. }
                ));
                assert!(matches!(
                    policy(RewriteOp::Upgrade, f),
                    Carry::Dropped { .. }
                ));
            }
            // The legacy `obs_index` / `var_index`: an append carries whatever
            // is there, a rewrite drops it.
            SectionFamily::Unwritten => {
                assert_eq!(policy(RewriteOp::BuildCsc, f), Carry::Verbatim);
                assert!(matches!(
                    policy(RewriteOp::Upgrade, f),
                    Carry::Dropped { .. }
                ));
            }
            _ => assert_eq!(
                policy(RewriteOp::Upgrade, f),
                policy(RewriteOp::BuildCsc, f),
                "{}: a new divergence between upgrade and build-csc; if it is \
                 intended, give it an arm in this test",
                f.label()
            ),
        }
    }
}

/// No cell claims to be an open bug.
///
/// The inverse of what this test used to be. Phase 5a listed ten cells as open
/// Majors (§6.3's seven for `build-csc`, §6.4's three for `merge`) and asserted
/// that each one's `why:` string said `(open)`, so that a reader would not take
/// the table for a specification and so that a fix had to come *here* and delete
/// the entry rather than leaving stale prose behind.
///
/// Phase 5b closed all ten, which empties the list — and an empty list is a test
/// that cannot fail. So the assertion is turned around: nothing in the table may
/// still call itself open. That catches the thing the list was really guarding
/// against (a `why:` that outlives the bug it describes) without needing anyone
/// to maintain a roster, and it needs no edit at all if a future cell is
/// correctly marked and later correctly fixed.
#[test]
fn no_cell_still_claims_to_be_an_open_bug() {
    for &op in RewriteOp::ALL {
        for &f in SectionFamily::ALL {
            for scope in [SectionScope::Global, SectionScope::PerModality] {
                if let Carry::Dropped { why, .. } = policy_scoped(op, f, scope) {
                    assert!(
                        !why.contains("(open)"),
                        "{}/{} ({}) still describes itself as an open bug: {why:?}. \
                         If it is open, that is fine — but §6.3 and §6.4 are closed, \
                         so say which review section this one is.",
                        op.label(),
                        f.label(),
                        scope.label(),
                    );
                }
            }
        }
    }
}

/// `audit_staged` runs before `ScxWriter::finish()`, and `finish()` writes two
/// things of its own: the `ModalityTable`, and a CSC sidecar auto-emitted for
/// any modality registered with `build_csc = true`. Neither is in the staged
/// catalog, so neither can be asserted.
///
/// That is only safe while no op's policy for those two families requires
/// **presence**. This pins exactly that and no more. Without it, flipping
/// `ModalityTable` to `Verbatim` would not break a unit test — it would make
/// every multimodal compact and merge fail at run time.
///
/// ⚠️ **It does not pin the mirror direction**, and the doc on `audit_staged`
/// should not be read as claiming it does: a family declared `Dropped` that
/// `finish()` then emits would pass the pre-persist audit and leave a persisted
/// artifact violating the policy. That shape is reachable for `XCsc` in two
/// ways, and neither is closed by this table: a rewrite call site registering a
/// modality with `build_csc = true` (none does), and a same-pass sidecar
/// (`ScxWriter::enable_csc_sidecar`) left for `finish()` to emit. The rewrite
/// ops emit theirs explicitly right after X, before `audit_staged`, which is
/// why their `XCsc` cells can be `Conditional`: the audit sees the sidecar.
#[test]
fn finish_writes_only_non_asserting_families() {
    for &op in RewriteOp::ALL {
        for f in [SectionFamily::ModalityTable, SectionFamily::XCsc] {
            let rule = policy(op, f);
            assert!(
                !matches!(rule, Carry::Verbatim | Carry::RowFiltered | Carry::Remapped),
                "{}/{} is {} — but that family is written by ScxWriter::finish(), \
                 which runs after audit_staged, so the audit would demand a \
                 section that cannot be there yet. Either keep it non-asserting \
                 or move that op's audit after finish()",
                op.label(),
                f.label(),
                rule.tag()
            );
        }
    }
}

/// `audit_in_place` asserts what only an append can promise: a `Verbatim`
/// family is the same catalog entry, not merely present. Each mutation below
/// keeps the family present — so `audit` alone passes it — and must fail here.
#[test]
fn audit_in_place_rejects_a_moved_or_rewritten_verbatim_entry() {
    use scx_format_io::catalog::{FullCatalog, FullCatalogEntry};
    let entry = |name: &str, t: SectionType, offset: u64| FullCatalogEntry {
        name: name.to_string(),
        offset,
        length: 100,
        section_type: t,
        checksum: [offset as u8; 32],
        modality_id: 0,
        stats: None,
    };
    let old = FullCatalog {
        catalog_version: scx_format_io::CURRENT_CATALOG_VERSION,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        n_obs: 10,
        entries: vec![
            entry("obs", SectionType::ObsMetadata, 4352),
            entry("X_shard_0", SectionType::CsrShard, 4456),
        ],
        data_generation: 1,
        csc_build_generation: 0,
    };
    let mut good = old.clone();
    good.entries
        .push(entry("X_csc_shard_0", SectionType::CscShard, 9000));
    assert!(audit_in_place(RewriteOp::BuildCsc, &old, &good).is_ok());

    let mut moved = good.clone();
    moved.entries[1].offset = 20_000;
    assert!(audit(RewriteOp::BuildCsc, &[&old], &moved.entries, 10).is_ok());
    let err = audit_in_place(RewriteOp::BuildCsc, &old, &moved).unwrap_err();
    assert!(err.to_string().contains("X_shard_0"), "{err}");

    let mut rewritten = good.clone();
    rewritten.entries[1].checksum = [0xff; 32];
    assert!(audit_in_place(RewriteOp::BuildCsc, &old, &rewritten).is_err());

    let mut renamed = good;
    renamed.entries[0].name = "obs_metadata/shard_0".to_string();
    assert!(audit_in_place(RewriteOp::BuildCsc, &old, &renamed).is_err());

    // Same bytes, same place, stats gone: the Level-1 pruning loss the table
    // exists for. `audit` cannot see it; this must.
    let mut with_stats = old.clone();
    with_stats.entries[1].stats = Some(scx_format_io::catalog::ShardStats {
        row_start: 0,
        row_end: 10,
        col_start: 0,
        col_end: 5,
        nnz: 7,
        value_min: 1,
        value_max: 3,
        value_sum: 9,
        n_indexed_columns: 0,
        column_stats: Vec::new(),
    });
    let mut cleared = with_stats.clone();
    cleared.entries[1].stats = None;
    assert!(audit(RewriteOp::BuildCsc, &[&with_stats], &cleared.entries, 10).is_ok());
    assert!(audit_in_place(RewriteOp::BuildCsc, &with_stats, &with_stats).is_ok());
    assert!(audit_in_place(RewriteOp::BuildCsc, &with_stats, &cleared).is_err());

    // A legacy `obs_index` section is carried, not refused: an append cannot
    // drop it without rewriting something.
    let mut legacy = old.clone();
    legacy
        .entries
        .push(entry("obs_index", SectionType::ObsIndex, 12_000));
    let mut legacy_new = legacy.clone();
    legacy_new
        .entries
        .push(entry("X_csc_shard_0", SectionType::CscShard, 20_000));
    assert!(audit_in_place(RewriteOp::BuildCsc, &legacy, &legacy_new).is_ok());
}
