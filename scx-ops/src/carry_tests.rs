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
/// The point is not that these values are *right* — several of them are open
/// bugs and are labelled as such in `carry.rs`. The point is that changing one
/// shows up here, in a diff a reviewer reads, instead of inside a `match` arm
/// in a 600-line module.
///
/// When a Phase 5b fix flips `merge`'s obsp/varp or `build-csc`'s
/// varm/obsp/varp/raw/bitmaps, this string changes with it, and that change is
/// the fix's most legible artifact.
#[test]
fn table_snapshot() {
    let expected = "\
compact
  obs                          row-filtered
  var                          verbatim
  X                            row-filtered
  X CSC sidecar                dropped(warns)
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
  X CSC sidecar                dropped(warns)
  layers                       rebuilt
  layer CSC sidecars           dropped(SILENT)
  obsm                         rebuilt
  varm                         conditional
  obsp                         remapped
    (per-modality)             dropped(warns)
  obsp (CSR-backed)            dropped(SILENT)
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
  X CSC sidecar                dropped(warns)
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
  detection bitmaps            verbatim
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
  X CSC sidecar                dropped(warns)
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
  layer CSC sidecars           dropped(warns)
  obsm                         verbatim
  varm                         dropped(warns)
  obsp                         dropped(warns)
  obsp (CSR-backed)            dropped(warns)
  varp                         dropped(warns)
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             verbatim
  detection bitmaps            dropped(warns)
  obs predicate index          verbatim
  var predicate index          verbatim
  modality table               refuse
  adata.raw                    dropped(warns)
  grouped-sort group index     dropped(warns)
  unwritten legacy index section dropped(SILENT)
upgrade
  obs                          verbatim
  var                          verbatim
  X                            verbatim
  X CSC sidecar                conditional
  layers                       verbatim
  layer CSC sidecars           dropped(warns)
  obsm                         verbatim
  varm                         dropped(warns)
  obsp                         dropped(warns)
  obsp (CSR-backed)            dropped(warns)
  varp                         dropped(warns)
  uns                          verbatim
  provenance                   rebuilt
  deletion vectors             verbatim
  detection bitmaps            dropped(warns)
  obs predicate index          verbatim
  var predicate index          verbatim
  modality table               refuse
  adata.raw                    dropped(warns)
  grouped-sort group index     dropped(warns)
  unwritten legacy index section dropped(SILENT)
";
    assert_eq!(render_table(), expected);
}

/// `upgrade` delegates to `build_csc` through an `other =>` arm, so it is the
/// one policy a new `SectionFamily` reaches without a compile error.
///
/// Pin the relationship the delegation asserts — identical everywhere except
/// the CSC sidecar — so the two cannot drift silently, which is the only thing
/// the missing compile error would have caught.
#[test]
fn upgrade_matches_build_csc_except_the_csc_sidecar() {
    for &f in SectionFamily::ALL {
        if f == SectionFamily::XCsc {
            // build-csc *creates* the sidecar; upgrade re-emits the input's,
            // and only when canonicalising did not change the matrix under it.
            assert_eq!(policy(RewriteOp::BuildCsc, f), Carry::Rebuilt);
            assert!(matches!(
                policy(RewriteOp::Upgrade, f),
                Carry::Conditional { .. }
            ));
            continue;
        }
        assert_eq!(
            policy(RewriteOp::Upgrade, f),
            policy(RewriteOp::BuildCsc, f),
            "{}: upgrade and build-csc share one carry implementation, so their \
             policies must agree",
            f.label()
        );
    }
}

/// Four cells of the table are open Majors, and a reader who does not already
/// know that will read the table as a specification.
///
/// Pinned so that a Phase 5b fix has to come here and delete the entry, rather
/// than leaving a `why:` string that still says "(open)" about something that
/// has been closed.
#[test]
fn open_bugs_are_labelled_in_the_table() {
    let open: Vec<(RewriteOp, SectionFamily)> = vec![
        // §6.4's three cells were here until Phase 5b closed them: merge now
        // remaps obsp by each input's row offset and takes varp from input 0.
        // `ObspCsr` is still a drop but no longer an *open* one — nothing in
        // the crate reads a CSR-backed graph outside `optimize`'s shard loop,
        // which is the same reason `compact` and `sort` drop it.
        (RewriteOp::BuildCsc, SectionFamily::Varm),
        (RewriteOp::BuildCsc, SectionFamily::Obsp),
        (RewriteOp::BuildCsc, SectionFamily::ObspCsr),
        (RewriteOp::BuildCsc, SectionFamily::Varp),
        (RewriteOp::BuildCsc, SectionFamily::Raw),
        (RewriteOp::BuildCsc, SectionFamily::Bitmap),
        (RewriteOp::BuildCsc, SectionFamily::GroupIndex),
    ];
    for (op, f) in open {
        match policy(op, f) {
            Carry::Dropped { why, .. } => assert!(
                why.contains("(open)"),
                "{}/{}: a known-open drop must say so in its `why`, got {why:?}",
                op.label(),
                f.label()
            ),
            other => panic!(
                "{}/{} is no longer a drop ({other:?}) — if it was fixed, remove it from \
                 this list and from the §6.3/§6.4 notes in carry.rs",
                op.label(),
                f.label()
            ),
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
/// `finish()` then auto-emits would pass the pre-persist audit and leave a
/// persisted artifact violating the policy. `XCsc` is `Dropped` for most ops, so
/// that shape is only unreachable because every rewrite call site registers its
/// modalities with `build_csc = false`. That is a property of the call sites,
/// not of this table, and nothing here enforces it — if a rewrite op ever passes
/// `build_csc = true`, this test will still be green and the invariant will be
/// false.
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
