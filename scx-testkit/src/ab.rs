//! Cross-tree A/B: did this refactor change what an op writes?
//!
//! [`crate::digest`] answers that for **one** file. This module is the layer
//! above it: a labelled collection of per-file digests
//! ([`OpDigestManifest`]) that can be written to JSON in one worktree and
//! asserted against in another, plus the run-it-twice helpers an op-level
//! identity test needs.
//!
//! It exists because the harness this replaces was never committed. The
//! organization series' Phase 5a/5b behaviour-identity claims were measured
//! with a scratch `ab_dump` test hand-copied into two worktrees and deleted
//! afterwards, which means the measurement cannot be reproduced, re-run
//! against a later commit, or audited. Everything here is deliberately
//! op-agnostic — it takes closures and paths — so the crate keeps its
//! dependency floor of `{scx-format, scx-format-io, scx-codec}` and any op
//! crate above that floor can dev-depend on it.
//!
//! ## What this harness cannot see
//!
//! Three regions of the file are outside a [`crate::digest::FileDigest`], and
//! a claim that rests on any of them needs its own oracle:
//!
//! * **`FileHeader::file_checksum`** — deliberately absent from
//!   [`crate::digest::HeaderDigest`] (see its doc comment) because it covers
//!   the `Provenance` section and therefore differs between two runs of the
//!   same op. A change to what `file_checksum` *means* — its extent, its
//!   producers, its verifier — is invisible here.
//! * **The `Provenance` section itself** —
//!   [`crate::digest::DEFAULT_EXCLUDED`]. Every mutating write path stamps
//!   `SystemTime::now()` into it.
//! * **The 4096-byte root catalog at offset 256** — it has zero production
//!   readers workspace-wide; see [`crate::digest::CatalogDigest`].
//!
//! Everything else about the file is covered, per section, by name.
//!
//! ## The three modes
//!
//! A test built on this typically wants all three, selected by environment so
//! one committed test binary serves both the standing regression net and the
//! ad-hoc A/B:
//!
//! | Env | [`AbMode`] | What happens |
//! |---|---|---|
//! | `SCX_TESTKIT_AB_DUMP=<path>` | [`AbMode::Dump`] | write the manifest, assert nothing |
//! | `SCX_TESTKIT_AB_BASE=<path>` | [`AbMode::CompareTo`] | assert the manifest equals the one at `<path>` |
//! | neither | [`AbMode::Golden`] | assert against the checked-in golden |
//!
//! So a cross-tree A/B is two commands and no scratch code:
//!
//! ```text
//! (cd ../base-worktree && SCX_TESTKIT_AB_DUMP=/tmp/base.json cargo test --test op_output_identity)
//! SCX_TESTKIT_AB_BASE=/tmp/base.json cargo test --test op_output_identity
//! ```
//!
//! ⚠️ **The base worktree must already contain the test target.** Cargo has no
//! such target at a commit predating it, so this recipe works from the first
//! commit that carries the harness onward, not against an arbitrary merge base.
//! `tests/scx-integration-tests/tests/op_output_identity.rs` carries the full
//! statement of that constraint and of what its own matrix does and does not
//! cover; treat it, not this comment, as authoritative.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::digest::{
    assert_digests_eq, digest_file, digest_file_excluding, FileDigest, Strictness, BLESS_ENV,
};
use scx_format_io::Result;

/// Environment variable naming a path to write the manifest to.
pub const DUMP_ENV: &str = "SCX_TESTKIT_AB_DUMP";

/// Environment variable naming a base manifest to compare against.
pub const BASE_ENV: &str = "SCX_TESTKIT_AB_BASE";

/// Stamped into every manifest.
///
/// Not decoration: a manifest read back from a tree whose harness has since
/// changed shape must fail loudly rather than compare as "no ops in common",
/// which is what a bare `BTreeMap` would do.
pub const MANIFEST_SCHEMA: &str = "scx-testkit/op-digest-manifest/1";

/// Several files' digests under one label each.
///
/// The label is the op — `"compact"`, `"merge"`, `"append"` — not the path, so
/// two worktrees writing to different temp directories still produce
/// comparable manifests.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpDigestManifest {
    /// [`MANIFEST_SCHEMA`].
    pub schema: String,
    /// Label → digest, ordered so the serialised JSON is stable.
    pub ops: BTreeMap<String, FileDigest>,
}

impl Default for OpDigestManifest {
    fn default() -> Self {
        Self::new()
    }
}

impl OpDigestManifest {
    pub fn new() -> Self {
        Self {
            schema: MANIFEST_SCHEMA.to_string(),
            ops: BTreeMap::new(),
        }
    }

    /// Digest `path` and file it under `label`.
    ///
    /// # Panics
    ///
    /// If `label` is already present. A silently-overwritten label is a test
    /// that measures one op twice and reports coverage of two.
    pub fn record(&mut self, label: &str, path: &Path, strictness: Strictness) -> Result<()> {
        let digest = digest_file(path, strictness)?;
        assert!(
            self.ops.insert(label.to_string(), digest).is_none(),
            "duplicate manifest label {label:?} — each op gets one entry, or the \
             manifest reports coverage it does not have"
        );
        Ok(())
    }

    pub fn labels(&self) -> impl Iterator<Item = &str> {
        self.ops.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn write_json(&self, path: &Path) -> Result<()> {
        // `Path::new("base.json").parent()` is `Some("")`, not `None`. On Linux
        // `create_dir_all("")` happens to return `Ok(())` — measured, and
        // independently by a reviewer — so a bare relative dump path works
        // today. It is not documented to, and `SCX_TESTKIT_AB_DUMP` is a
        // user-facing env var that invites exactly that spelling, so do not
        // depend on it.
        match path.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => {
                std::fs::create_dir_all(parent)?;
            }
            _ => {}
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    /// Read a manifest written by [`Self::write_json`].
    ///
    /// # Panics
    ///
    /// If the file is not a manifest, or carries a different
    /// [`MANIFEST_SCHEMA`]. Both are "this comparison is not the one you think
    /// it is", which must not degrade into a quiet pass.
    pub fn read_json(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let m: Self = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("{} is not an OpDigestManifest: {e}", path.display()));
        assert_eq!(
            m.schema,
            MANIFEST_SCHEMA,
            "manifest {} was written by a different harness version; \
             regenerate it in the tree it came from rather than comparing across shapes",
            path.display()
        );
        Ok(m)
    }
}

/// Assert two manifests agree, naming the op and then the section.
///
/// # Panics
///
/// On any difference. Labels present on one side only are reported first and
/// separately from content differences: a missing op means the two runs did
/// not measure the same thing, which is a different (and worse) failure than
/// an op whose output moved.
pub fn assert_manifests_eq(actual: &OpDigestManifest, expected: &OpDigestManifest) {
    assert_eq!(
        actual.schema, expected.schema,
        "manifest schema {:?} vs {:?} — the two are not comparable",
        actual.schema, expected.schema
    );

    let mut missing: Vec<&str> = expected
        .labels()
        .filter(|l| !actual.ops.contains_key(*l))
        .collect();
    let mut extra: Vec<&str> = actual
        .labels()
        .filter(|l| !expected.ops.contains_key(*l))
        .collect();
    missing.sort_unstable();
    extra.sort_unstable();

    assert!(
        missing.is_empty() && extra.is_empty(),
        "the two manifests cover different ops — missing {missing:?}, unexpected {extra:?}. \
         Equal digests for the ops they share would say nothing about the ones they do not."
    );

    // Only now, on a matched label set, is a per-op comparison meaningful.
    for (label, a) in &actual.ops {
        let e = &expected.ops[label];
        if a == e {
            continue;
        }
        eprintln!("op {label:?} differs:");
        assert_digests_eq(a, e);
        // `assert_digests_eq` panics on any difference, so reaching here means
        // the two compared unequal by `PartialEq` yet equal field-by-field —
        // which would be a bug in the digest types, not in the op.
        unreachable!("digest for {label:?} is unequal but reports no difference");
    }
}

/// Assert a manifest matches a checked-in golden, or rewrite it under
/// [`BLESS_ENV`]`=1`.
///
/// A missing golden is written and then fails, so a first run cannot pass by
/// writing its own expectation. Same contract as
/// [`crate::digest::assert_matches_golden`], and the same env var — one bless
/// switch for the whole crate.
pub fn assert_manifest_matches_golden(actual: &OpDigestManifest, golden: &Path) -> Result<()> {
    let bless = std::env::var(BLESS_ENV).as_deref() == Ok("1");

    if bless || !golden.exists() {
        actual.write_json(golden)?;
        assert!(
            bless,
            "golden {} did not exist; it has been written. Review it and re-run \
             — a first run must not pass by writing its own expectation.",
            golden.display()
        );
        return Ok(());
    }

    let expected = OpDigestManifest::read_json(golden)?;
    assert_manifests_eq(actual, &expected);
    Ok(())
}

/// What a manifest-producing test should do with the manifest it built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbMode {
    /// Write it to this path and assert nothing.
    Dump(PathBuf),
    /// Assert it equals the manifest at this path.
    CompareTo(PathBuf),
    /// Assert it equals the checked-in golden.
    Golden,
}

/// Resolve [`AbMode`] from [`DUMP_ENV`] / [`BASE_ENV`].
///
/// # Panics
///
/// If both are set. They ask for incompatible things, and picking one would
/// silently discard the other half of what the caller asked for.
pub fn ab_mode_from_env() -> AbMode {
    let dump = std::env::var(DUMP_ENV).ok().filter(|s| !s.is_empty());
    let base = std::env::var(BASE_ENV).ok().filter(|s| !s.is_empty());
    match (dump, base) {
        (Some(_), Some(_)) => panic!(
            "{DUMP_ENV} and {BASE_ENV} are both set; the first writes a manifest and \
             the second asserts against one — set exactly one"
        ),
        (Some(d), None) => AbMode::Dump(PathBuf::from(d)),
        (None, Some(b)) => AbMode::CompareTo(PathBuf::from(b)),
        (None, None) => AbMode::Golden,
    }
}

/// Dispatch a built manifest through [`ab_mode_from_env`].
///
/// Returns `true` when the run asserted something, `false` when it only
/// dumped — so a caller can skip the assertions that only make sense in a
/// self-checking run.
pub fn resolve_against_env(actual: &OpDigestManifest, golden: &Path) -> Result<bool> {
    assert!(
        !actual.is_empty(),
        "refusing to resolve an empty manifest: a run that measured nothing must \
         not report agreement with anything"
    );
    match ab_mode_from_env() {
        AbMode::Dump(path) => {
            actual.write_json(&path)?;
            eprintln!(
                "{DUMP_ENV}: wrote {} op digest(s) to {}",
                actual.len(),
                path.display()
            );
            Ok(false)
        }
        AbMode::CompareTo(base) => {
            let expected = OpDigestManifest::read_json(&base)?;
            assert_manifests_eq(actual, &expected);
            Ok(true)
        }
        AbMode::Golden => {
            assert_manifest_matches_golden(actual, golden)?;
            Ok(true)
        }
    }
}

// ---------------------------------------------------------------------------
// Run-it-twice helpers.
//
// Moved here from `tests/scx-integration-tests/tests/testkit_against_real_ops.rs`,
// where they were private to one file. They prove a weaker property than the
// manifest comparison above -- that provenance's clock does not leak into the
// digest, i.e. that the harness is *usable* on an op -- but every op-level
// identity test needs that property established before its own claim means
// anything, so it belongs next to the harness rather than in one test binary.
// ---------------------------------------------------------------------------

/// Block until the wall-clock second changes, so a following run's provenance
/// timestamp is guaranteed to differ from the preceding one's.
///
/// Without this a two-run comparison passes trivially whenever both runs land
/// in the same second — which is most of the time, and exactly when it proves
/// nothing.
pub fn wait_for_next_second() {
    let now = || {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
    };
    let start = now();
    while now() == start {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// The premise every clock-independence case depends on: the two runs really
/// did stamp different provenance, so "the digests agree" is a statement about
/// the exclusion rather than about two identical files.
///
/// # Panics
///
/// If the two files are byte-identical, or differ in no section once
/// provenance is *included*.
pub fn assert_provenance_actually_differs(a: &Path, b: &Path) {
    let inc = |p: &Path| digest_file_excluding(p, Strictness::Content, &[]).unwrap();
    assert_ne!(
        inc(a).sections,
        inc(b).sections,
        "premise: the two runs must have stamped different provenance; \
         without that this test cannot fail"
    );
    assert_ne!(
        std::fs::read(a).unwrap(),
        std::fs::read(b).unwrap(),
        "premise: the two files must differ on disk"
    );
}

/// Run an **in-place** `op` on two fresh copies of `src`, a second apart, and
/// return the two mutated files.
pub fn run_twice_across_a_second(
    src: &Path,
    dir: &Path,
    label: &str,
    op: impl Fn(&Path),
) -> (PathBuf, PathBuf) {
    let (a, b) = (
        dir.join(format!("{label}_a.scx")),
        dir.join(format!("{label}_b.scx")),
    );
    std::fs::copy(src, &a).unwrap();
    op(&a);
    wait_for_next_second();
    std::fs::copy(src, &b).unwrap();
    op(&b);
    (a, b)
}

/// Run a **copy-out** `op` twice over one `src`, a second apart, and assert the
/// digests agree while the raw bytes do not.
///
/// One source, not two. Building a second fixture would make the two outputs
/// differ for a reason that has nothing to do with the clock — a fixture that
/// ends with `mark_deleted` stamps its own `SystemTime::now()`, so two inputs'
/// `file_checksum`s differ and `merge` copies those into its provenance. The
/// premise assertion then passes with [`wait_for_next_second`] deleted, which
/// is how this helper was first written and how it was caught: removing the
/// wait left every case green.
pub fn assert_op_is_clock_independent(
    src: &Path,
    dir: &Path,
    label: &str,
    op: impl Fn(&Path, &Path),
) {
    let a = dir.join(format!("{label}_a.scx"));
    op(src, &a);

    wait_for_next_second();

    let b = dir.join(format!("{label}_b.scx"));
    op(src, &b);

    assert_provenance_actually_differs(&a, &b);
    assert_digests_eq(
        &digest_file(&a, Strictness::Content).unwrap(),
        &digest_file(&b, Strictness::Content).unwrap(),
    );
}

#[cfg(test)]
#[path = "ab_tests.rs"]
mod tests;
