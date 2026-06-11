//! Doc-drift CI guards (code-review item 10 / Phase 0.3).
//!
//! These tests pin the magic numbers and identifiers that documentation
//! quotes verbatim, so a code change that silently diverges from the docs
//! fails CI instead of rotting into stale prose (the D1/D3/D5/D6 finding
//! class). Cheap to run, no fixtures, no network.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Workspace root, derived from this crate's manifest dir
/// (`<root>/tests/scx-integration-tests`).
fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root should resolve")
}

/// The default CSR shard row count is documented in docs/format.md,
/// docs/sharding.md, and AGENTS.md as 16,384. Pin it so the code and docs
/// move together (D1).
#[test]
fn default_shard_target_rows_pinned() {
    assert_eq!(
        scx_format::DEFAULT_SHARD_TARGET_ROWS,
        16_384,
        "docs (format.md / sharding.md / AGENTS.md) cite 16,384; update them together"
    );
}

/// The CPU covariance-PCA var cutoff (5000) is documented in AGENTS.md and
/// docs/scanpy.md. Pin it (D7). The GPU cutoff (8000) lives behind the `gpu`
/// feature and is pinned in scx-accel's own gpu tests.
#[test]
fn cpu_pca_threshold_pinned() {
    assert_eq!(
        scx_accel::pca::COVARIANCE_PCA_THRESHOLD,
        5_000,
        "AGENTS.md / docs/scanpy.md cite the 5000-var CPU PCA cutoff; update them together"
    );
}

/// The workspace member set must match the documented crate graph (14 code
/// crates + the integration-test crate). A crate added or renamed without
/// updating AGENTS.md / docs/architecture.md trips this (D3).
#[test]
fn workspace_members_match_documented_crates() {
    let cargo = std::fs::read_to_string(workspace_root().join("Cargo.toml")).unwrap();
    let members = parse_members(&cargo);
    let expected: BTreeSet<String> = [
        "scx-format",
        "scx-codec",
        "scx-sparse",
        "scx-cli",
        "scx-convert",
        "scx-ops",
        "scx-engine",
        "scx-loader",
        "scx-cloud",
        "scx-gpu",
        "scx-mtx",
        "scx-accel",
        "pyscx",
        "rscx",
        "tests/scx-integration-tests",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    assert_eq!(
        members, expected,
        "workspace members changed — update the crate graph + count in AGENTS.md \
         and docs/architecture.md, then update this guard"
    );
}

/// Tracked source and docs must not cite the gitignored root scratch docs
/// (`ACC-RUST-OPT-V2`/`-V3`) or reference the removed `SCX_GPU_DE_V2=` /
/// `SCX_GPU_DE_V3=` opt-in gates (or the removed `SCX_GPU_DE_V3_TRACE=` debug
/// trace) as if functional (D5/D6). The `=` form is deliberately required so
/// "the gates/trace were removed" historical prose (which names the bare
/// `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3`/`SCX_GPU_DE_V3_TRACE` symbols) is not flagged.
#[test]
fn tracked_files_free_of_scratch_doc_citations_and_removed_gates() {
    const BANNED: &[&str] = &[
        "ACC-RUST-OPT-V2",
        "ACC-RUST-OPT-V3",
        "SCX_GPU_DE_V2=",
        "SCX_GPU_DE_V3=",
        "SCX_GPU_DE_V3_TRACE=",
    ];
    let root = workspace_root();
    let mut files = Vec::new();
    for crate_dir in [
        "scx-format",
        "scx-codec",
        "scx-sparse",
        "scx-cli",
        "scx-convert",
        "scx-ops",
        "scx-engine",
        "scx-loader",
        "scx-cloud",
        "scx-gpu",
        "scx-mtx",
        "scx-accel",
        "pyscx",
        "rscx",
    ] {
        collect_files(&root.join(crate_dir).join("src"), "rs", &mut files);
    }
    collect_files(&root.join("docs"), "md", &mut files);
    for top in ["AGENTS.md", "ROADMAP.md"] {
        let p = root.join(top);
        if p.is_file() {
            files.push(p);
        }
    }

    let mut offenders = Vec::new();
    for path in files {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for needle in BANNED {
            if content.contains(needle) {
                offenders.push(format!("{} cites `{}`", path.display(), needle));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "tracked files reference gitignored scratch docs or removed env gates \
         (inline the substance / drop the gate):\n{}",
        offenders.join("\n")
    );
}

/// Correctness guards at trust boundaries must be **always-on validation**,
/// not `debug_assert!`/`#[cfg(debug_assertions)]` (which vanish in release and
/// let the code "silently produce wrong numbers" — review Abstraction 11). This
/// guard bans the `debug_assert!` macro family in the decode / scatter / rank
/// hot-path files. A genuinely-internal invariant (not an untrusted-input
/// boundary) may stay debug-only if it carries a `debug-assert-ok:`
/// justification on the line above or the same line (the policy's "documented
/// justification" escape hatch).
#[test]
fn no_debug_assert_in_hot_paths() {
    // (crate dir, source file relative to `<crate>/src`)
    const HOT_PATHS: &[(&str, &str)] = &[
        ("scx-format", "shard_decode.rs"),
        ("scx-codec", "bitstream.rs"),
        ("scx-codec", "rice.rs"),
        ("scx-codec", "delta_golomb.rs"),
        ("scx-gpu", "gpu_shard_source.rs"),
        ("scx-accel", "diffexp.rs"),
    ];
    let root = workspace_root();
    let mut offenders = Vec::new();
    for (crate_dir, file) in HOT_PATHS {
        let path = root.join(crate_dir).join("src").join(file);
        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("hot-path file {} unreadable: {e}", path.display()));
        let lines: Vec<&str> = content.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let is_macro_call = line.contains("debug_assert!(")
                || line.contains("debug_assert_eq!(")
                || line.contains("debug_assert_ne!(");
            if !is_macro_call {
                continue;
            }
            let justified = line.contains("debug-assert-ok:")
                || (i > 0 && lines[i - 1].contains("debug-assert-ok:"));
            if !justified {
                offenders.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "decode/scatter/rank hot paths must use always-on validation (return an error at the \
         trust boundary), not debug_assert! (which is compiled out in release). Convert these, \
         or add a `debug-assert-ok:` justification if the assert is a genuinely-internal \
         invariant:\n{}",
        offenders.join("\n")
    );
}

/// Extract the quoted members from a workspace `Cargo.toml`'s
/// `members = [ ... ]` array (single- or multi-line).
fn parse_members(cargo_toml: &str) -> BTreeSet<String> {
    let start = cargo_toml
        .find("members")
        .and_then(|i| cargo_toml[i..].find('[').map(|j| i + j + 1))
        .expect("members array");
    let end = start + cargo_toml[start..].find(']').expect("members array close");
    cargo_toml[start..end]
        .split(',')
        .filter_map(|tok| {
            let t = tok.trim().trim_matches('"').trim();
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect()
}

/// Recursively collect files with the given extension under `dir`.
fn collect_files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}
