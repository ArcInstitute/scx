//! A GPU test must declare itself as one — enforced by scanning the source.
//!
//! # Why this exists
//!
//! `require_gpu!()` used to expand to `eprintln! + return`. On a host with no
//! CUDA device — every CI runner and most dev machines — that made 170 tests in
//! this crate and 34 in `scx-accel` guaranteed no-ops that libtest counted as
//! **passed**, with the message swallowed by output capture. Inverting a plane
//! index in a decode kernel would have moved no counter anywhere it was run.
//!
//! The repair is a pairing: the gate macro decides at runtime, and
//! `#[ignore = "requires a CUDA GPU"]` tells libtest not to select the test by
//! default so a plain run reports it as ignored rather than passed. Either half
//! alone is worthless — an un-`#[ignore]`d gated test is the original bug, and
//! an `#[ignore]`d test with no gate never runs anywhere at all.
//!
//! # Why it lives in `tests/` and runs without a GPU
//!
//! **The drift happens on CPU hosts, so the guard has to run there.** A check
//! that only fires on the GPU node would not have caught a single one of the
//! ten hand-rolled gates that existed before this file: they were written, and
//! reviewed, and merged, on machines that never executed them. This is a
//! source scan, needs no device, and rides along with `cargo test --workspace`.
//!
//! # What it checks
//!
//! 1. A test invokes a device gate **iff** it carries the `#[ignore]` reason.
//! 2. In test-only code, a failure to open a device is never a quiet `return`:
//!    `GpuDevice::new(0)` must be `.unwrap()`/`.expect()`-terminated, and the
//!    capability probes may not be called at all. This catches a gate hidden in
//!    a helper function — 11 of `scx-accel`'s 34 GPU tests gated through
//!    `no_gpu()` / `run_parity()`, invisible to any check that only reads test
//!    bodies. Which files count as test-only is resolved by following
//!    `#[cfg(test)] #[path = "…"] mod …;`, **not** by a filename suffix: the
//!    suffix version of this rule missed `scx-accel/src/harmony/tests.rs`
//!    entirely, and a `no_gpu()` planted there passed.
//! 3. A gate macro may not appear outside a `#[test]` body — wrapped in a
//!    helper, its `return` leaves the helper and the test runs on regardless.
//! 4. Both discovered sets are non-empty, so a rename cannot make this pass
//!    vacuously, and no extracted body over-ran its function.
//!
//! An intentional exception carries `// gpu-gate-exempt: <reason>` on the same
//! or preceding line. Every one is printed on a passing run, so exemptions are
//! visible in CI output rather than accumulating quietly.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Prefix of the reason string that marks a test as GPU-requiring.
const IGNORE_PREFIX: &str = "#[ignore = \"requires a CUDA GPU";

/// Macros that acquire a device and return early when there is none.
const DEVICE_GATES: [&str; 2] = ["require_gpu!()", "require_gpu_or_skip!()"];

/// Opt-out marker; must be followed by a reason.
const EXEMPT_MARKER: &str = "gpu-gate-exempt:";

/// Probes that must not be used to decide whether a test runs.
const CAPABILITY_PROBES: [&str; 3] = ["gpu_available()", "cuvs_available()", "nvcomp_available()"];

/// Sanctioned endings for a `GpuDevice::new(0)` in test code: both are hard
/// errors, so neither can turn into a silent pass.
const HARD_FAIL_ENDINGS: [&str; 2] = [".unwrap()", ".expect("];

struct TestFn {
    name: String,
    line: usize,
    attrs: String,
    body: String,
}

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rust_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The `{…}` block starting at the first `{` at or after `from`.
///
/// Braces inside string literals, char literals and comments are skipped. A
/// counter that does not do this is wrong in both directions, and only one of
/// them announces itself: a stray `"{"` over-runs into the next function, which
/// the `#[test]`-in-body assertion catches — but a stray `"}"` closes the body
/// *early* and silently, hiding whatever follows it from every rule here.
///
/// Returns `None` when the braces never balance before EOF; callers treat that
/// as a hard error rather than as "no body".
fn brace_block(text: &str, from: usize) -> Option<(usize, usize)> {
    let b = text.as_bytes();
    let start = text[from..].find('{')? + from;
    let mut depth = 0usize;
    let mut i = start;

    while i < b.len() {
        match b[i] {
            // Line comment.
            b'/' if b.get(i + 1) == Some(&b'/') => {
                i = text[i..].find('\n').map_or(b.len(), |p| i + p);
            }
            // Block comment. Rust nests these; so does this.
            b'/' if b.get(i + 1) == Some(&b'*') => {
                let mut nest = 1usize;
                i += 2;
                while i < b.len() && nest > 0 {
                    if b[i] == b'/' && b.get(i + 1) == Some(&b'*') {
                        nest += 1;
                        i += 2;
                    } else if b[i] == b'*' && b.get(i + 1) == Some(&b'/') {
                        nest -= 1;
                        i += 2;
                    } else {
                        i += 1;
                    }
                }
            }
            // Raw string: r"…", r#"…"#, r##"…"##.
            b'r' if matches!(b.get(i + 1), Some(&b'"') | Some(&b'#')) => {
                let mut hashes = 0usize;
                let mut j = i + 1;
                while b.get(j) == Some(&b'#') {
                    hashes += 1;
                    j += 1;
                }
                if b.get(j) != Some(&b'"') {
                    i += 1; // an identifier starting with `r`, not a raw string
                    continue;
                }
                let terminator = format!("\"{}", "#".repeat(hashes));
                i = text[j + 1..]
                    .find(&terminator)
                    .map_or(b.len(), |p| j + 1 + p + terminator.len());
            }
            // Ordinary string, with backslash escapes.
            b'"' => {
                i += 1;
                while i < b.len() && b[i] != b'"' {
                    i += if b[i] == b'\\' { 2 } else { 1 };
                }
                i += 1;
            }
            // Char literal — only the two that can affect the count, plus the
            // escaped forms. A bare `'` is far more often a lifetime (`&'a str`),
            // and treating that as a literal would swallow the rest of the file.
            b'\'' => {
                let rest = &text[i..];
                let lit = ["'{'", "'}'", "'\\''", "'\\\\'"]
                    .iter()
                    .find(|l| rest.starts_with(*l));
                i += lit.map_or(1, |l| l.len());
            }
            b'{' => {
                depth += 1;
                i += 1;
            }
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((start, i + 1));
                }
                i += 1;
            }
            _ => i += 1,
        }
    }
    None
}

fn line_of(text: &str, offset: usize) -> usize {
    text[..offset].bytes().filter(|&b| b == b'\n').count() + 1
}

/// Every `#[test]` function in `text`, with the attribute block that precedes
/// its `fn` line and its brace-matched body.
fn test_fns(text: &str, path: &Path) -> Vec<TestFn> {
    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = text[search..].find("#[test]") {
        let at = search + rel;
        search = at + "#[test]".len();

        // Only a real attribute, not the token inside a string or doc comment.
        let line_start = text[..at].rfind('\n').map_or(0, |p| p + 1);
        let prefix = &text[line_start..at];
        if !prefix.trim().is_empty() {
            continue;
        }

        let Some(fn_rel) = text[at..].find("fn ") else {
            continue;
        };
        let fn_at = at + fn_rel;
        let name_start = fn_at + "fn ".len();
        let name: String = text[name_start..]
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();

        // Attribute block: the run of `#[…]` / comment lines ending at `fn`.
        let fn_line_start = text[..fn_at].rfind('\n').map_or(0, |p| p + 1);
        let mut attr_start = fn_line_start;
        while attr_start > 0 {
            let prev_end = attr_start - 1;
            let prev_start = text[..prev_end].rfind('\n').map_or(0, |p| p + 1);
            let line = text[prev_start..prev_end].trim();
            if line.starts_with("#[") || line.starts_with("//") {
                attr_start = prev_start;
            } else {
                break;
            }
        }

        let (body_start, body_end) = brace_block(text, fn_at).unwrap_or_else(|| {
            panic!(
                "{}: braces never balance for `{name}` — the scan cannot be trusted",
                path.display()
            )
        });

        out.push(TestFn {
            name,
            line: line_of(text, fn_at),
            attrs: text[attr_start..fn_line_start].to_string(),
            body: text[body_start..body_end].to_string(),
        });
        search = body_end;
    }
    out
}

/// Files pulled in whole by a `#[cfg(…test…)] #[path = "…"] mod …;`.
///
/// Resolved by following the declaration rather than by matching a filename
/// suffix. The suffix heuristic (`*_tests.rs`) was wrong by exactly one file:
/// `scx-accel/src/harmony/tests.rs` is included this way, holds four GPU tests,
/// and has no inline `#[cfg(test)] mod` of its own — so the probe rule below
/// never scanned it, and a `no_gpu()` helper reintroduced there passed CI.
fn extracted_test_files(sources: &[PathBuf]) -> HashSet<PathBuf> {
    let mut out = HashSet::new();
    for src in sources {
        // Union, not replacement: the suffix convention still holds for the 15
        // `*_tests.rs` files, and a future `#[cfg(test)] mod foo_tests;` with no
        // `#[path]` would otherwise recreate exactly the hole this fixes.
        if src
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with("_tests.rs"))
        {
            out.insert(src.clone());
        }
        let Ok(text) = std::fs::read_to_string(src) else {
            continue;
        };
        for m in text.match_indices("#[path = \"") {
            let start = m.0 + "#[path = \"".len();
            let Some(end) = text[start..].find('"') else {
                continue;
            };
            let name = &text[start..start + end];
            // The `#[cfg(…test…)]` sits within a couple of lines above.
            let ctx_start = text[..m.0].rfind("#[cfg(").unwrap_or(0);
            if !text[ctx_start..m.0].contains("test") {
                continue;
            }
            if let Some(dir) = src.parent() {
                out.insert(dir.join(name));
            }
        }
    }
    out
}

/// Regions of `text` that are compiled only under `cfg(test)`.
///
/// A file listed in `extracted` is test-only in its entirety; elsewhere it is
/// the body of each `#[cfg(…test…)] mod … { … }`.
fn test_regions(text: &str, path: &Path, extracted: &HashSet<PathBuf>) -> Vec<String> {
    if extracted.contains(path) {
        return vec![text.to_string()];
    }

    let mut out = Vec::new();
    let mut search = 0usize;
    while let Some(rel) = text[search..].find("#[cfg(") {
        let at = search + rel;
        search = at + "#[cfg(".len();
        let Some(close) = text[at..].find(")]") else {
            break;
        };
        let cfg = &text[at..at + close];
        if !cfg.contains("test") {
            continue;
        }
        let after = at + close + ")]".len();
        // The attribute must apply to an inline `mod … {`, not a `mod …;`.
        let rest = text[after..].trim_start();
        if !rest.starts_with("mod ") {
            continue;
        }
        let Some(semi_or_brace) = rest.find(['{', ';']) else {
            continue;
        };
        if rest.as_bytes()[semi_or_brace] == b';' {
            continue;
        }
        if let Some((s, e)) = brace_block(text, after) {
            out.push(text[s..e].to_string());
            search = e;
        }
    }
    out
}

/// Lines carrying an exemption marker, as `path:line — reason`.
fn exemptions(region: &str, path: &Path, base_line: usize) -> Vec<String> {
    region
        .lines()
        .enumerate()
        .filter_map(|(i, line)| {
            let reason = line.split_once(EXEMPT_MARKER)?.1.trim();
            assert!(
                !reason.is_empty(),
                "{}: `{EXEMPT_MARKER}` with no reason on line {}",
                path.display(),
                base_line + i
            );
            Some(format!("{}:{} — {reason}", path.display(), base_line + i))
        })
        .collect()
}

fn is_exempt_near(region: &str, occurrence: usize) -> bool {
    // The marker may sit on the occurrence's own line or the one above it.
    let line_start = region[..occurrence].rfind('\n').map_or(0, |p| p + 1);
    let prev_start = region[..line_start.saturating_sub(1)]
        .rfind('\n')
        .map_or(0, |p| p + 1);
    let line_end = region[occurrence..]
        .find('\n')
        .map_or(region.len(), |p| occurrence + p);
    region[prev_start..line_end].contains(EXEMPT_MARKER)
}

struct Scan {
    gate_outside_a_test: Vec<String>,
    gated_without_ignore: Vec<String>,
    ignored_without_gate: Vec<String>,
    quiet_probes: Vec<String>,
    gated: usize,
    tests: usize,
    regions: usize,
    exempt: Vec<String>,
}

fn scan(tree: &Path) -> Scan {
    let mut s = Scan {
        gate_outside_a_test: Vec::new(),
        gated_without_ignore: Vec::new(),
        ignored_without_gate: Vec::new(),
        quiet_probes: Vec::new(),
        gated: 0,
        tests: 0,
        regions: 0,
        exempt: Vec::new(),
    };

    let sources = rust_sources(tree);
    let extracted = extracted_test_files(&sources);

    for path in &sources {
        let path = path.as_path();
        let text = std::fs::read_to_string(path).expect("read source");
        if !text.contains("#[test]") && !text.contains("#[cfg(test") {
            continue;
        }

        for t in test_fns(&text, path) {
            s.tests += 1;
            assert!(
                !t.body.contains("#[test]"),
                "{}: the body extracted for `{}` ran past the end of the function — \
                 brace matching cannot be trusted, so every other verdict here is suspect",
                path.display(),
                t.name
            );

            let gated = DEVICE_GATES.iter().any(|g| t.body.contains(g));
            let ignored = t.attrs.contains(IGNORE_PREFIX);
            let where_ = format!("{}:{} {}", path.display(), t.line, t.name);
            match (gated, ignored) {
                (true, false) => s.gated_without_ignore.push(where_),
                (false, true) => s.ignored_without_gate.push(where_),
                (true, true) => s.gated += 1,
                (false, false) => {}
            }
        }

        for region in test_regions(&text, path, &extracted) {
            s.regions += 1;
            let base_line = text
                .find(&region)
                .map_or(1, |off| line_of(&text, off))
                .saturating_sub(1);
            s.exempt.extend(exemptions(&region, path, base_line));

            // A gate macro outside a `#[test]` body is the last way to hide
            // one: its `return` leaves the *helper*, so the test carries on
            // regardless — and neither rule above sees a gate or a raw probe.
            // Whole-line comments are stripped so a rustdoc that merely names
            // the macro is not counted as an invocation.
            let code: String = region
                .lines()
                .filter(|l| !l.trim_start().starts_with("//"))
                .collect::<Vec<_>>()
                .join("\n");
            let in_bodies: String = test_fns(&region, path)
                .iter()
                .map(|t| t.body.as_str())
                .collect();
            for gate in DEVICE_GATES {
                let outside = code
                    .matches(gate)
                    .count()
                    .saturating_sub(in_bodies.matches(gate).count());
                if outside > 0 {
                    s.gate_outside_a_test.push(format!(
                        "{}: {outside} × `{gate}` outside any #[test] fn",
                        path.display()
                    ));
                }
            }

            for (i, _) in region.match_indices("GpuDevice::new(0)") {
                let tail = &region[i..(i + 120).min(region.len())];
                let hard = HARD_FAIL_ENDINGS.iter().any(|e| tail.contains(e));
                if !hard && !is_exempt_near(&region, i) {
                    s.quiet_probes.push(format!(
                        "{}:{} GpuDevice::new(0) that neither unwraps nor goes through a gate",
                        path.display(),
                        base_line + line_of(&region, i)
                    ));
                }
            }
            for probe in CAPABILITY_PROBES {
                for (i, _) in region.match_indices(probe) {
                    if !is_exempt_near(&region, i) {
                        s.quiet_probes.push(format!(
                            "{}:{} `{probe}` called from test code — use require_gpu_cap!()",
                            path.display(),
                            base_line + line_of(&region, i)
                        ));
                    }
                }
            }
        }
    }
    s
}

/// `brace_block` is the primitive every rule here stands on, and it had no
/// direct test — the integration scan only exercises it on real files, which
/// happen to contain no brace inside a string. Each case is a shape that
/// silently mis-scanned before the lexer landed; the lifetime case is the one
/// that would break a naive char-literal handler.
#[test]
fn brace_block_skips_strings_comments_and_char_literals() {
    for (src, label) in [
        (
            r#"fn t() { let s = "}"; let x = 1; }"#,
            "close brace in a string",
        ),
        (r#"fn t() { let s = "{"; }"#, "open brace in a string"),
        ("fn t() { // }\n let x = 1; }", "line comment"),
        ("fn t() { /* } */ let x = 1; }", "block comment"),
        (
            "fn t() { /* /* } */ */ let x = 1; }",
            "nested block comment",
        ),
        ("fn t() { let c = '}'; }", "char literal"),
        (r##"fn t() { let r = r"}"; }"##, "raw string"),
        (
            r###"fn t() { let r = r#"a"}"#; }"###,
            "raw string with hash",
        ),
        (
            r#"fn t() { let s = "\""; let u = "}"; }"#,
            "escaped quote then brace",
        ),
        (
            "fn t<'a>(x: &'a str) { let _ = x; }",
            "lifetime is not a char literal",
        ),
    ] {
        let open = src.find('{').expect("fixture has a brace");
        assert_eq!(
            brace_block(src, 0),
            Some((open, src.len())),
            "{label}: body should span the whole fn"
        );
    }
}

/// An unbalanced block never returns a span; callers turn that into a panic
/// rather than silently treating it as "no body".
#[test]
fn brace_block_reports_unbalanced_input() {
    // Unbalanced: no closing brace before EOF.
    assert_eq!(brace_block("fn t() { let x = 1;", 0), None);
    // A `{` hidden in a string is not an opener, so there is none to find.
    assert_eq!(brace_block(r#"let s = "{";"#, 0), None);
    // Searching past the last brace finds nothing rather than panicking.
    let src = "fn t() -> u8 { 0 }";
    assert_eq!(brace_block(src, src.len()), None);
}

#[test]
fn gpu_tests_are_gated_and_ignored() {
    let gpu_src = crate_root().join("src");
    let accel_src = crate_root().join("../scx-accel/src");

    let mut trees = vec![("scx-gpu", scan(&gpu_src))];
    // Absent when this crate is consumed outside the workspace; the scx-gpu
    // half is never optional.
    if accel_src.is_dir() {
        trees.push(("scx-accel", scan(&accel_src)));
    } else {
        eprintln!("scx-accel sources not present — scanning scx-gpu only");
    }

    let mut failures = Vec::new();
    for (name, s) in &trees {
        assert!(
            s.tests > 0 && s.regions > 0,
            "{name}: discovered {} tests in {} test regions — the scan found nothing, \
             which means it is measuring nothing",
            s.tests,
            s.regions
        );
        assert!(
            s.gated > 0,
            "{name}: no gated GPU tests discovered; if the gate macros were renamed, \
             this guard is now vacuous"
        );
        eprintln!("{name}: {} GPU-gated of {} tests", s.gated, s.tests);
        for e in &s.exempt {
            eprintln!("  gpu-gate-exempt {e}");
        }

        for f in &s.gated_without_ignore {
            failures.push(format!(
                "gated but not ignored (counts as PASSED on a CPU host): {f}"
            ));
        }
        for f in &s.ignored_without_gate {
            failures.push(format!(
                "ignored as GPU-requiring but never gates (so it runs nowhere): {f}"
            ));
        }
        for f in &s.gate_outside_a_test {
            failures.push(format!(
                "gate macro in a helper — its `return` leaves the helper, not the test: {f}"
            ));
        }
        for f in &s.quiet_probes {
            failures.push(format!("test decides for itself whether to run: {f}"));
        }
    }

    assert!(
        failures.is_empty(),
        "GPU test gating violations:\n  {}\n\nEvery GPU test needs both \
         `require_gpu!()` (or `require_gpu_or_skip!()`) and \
         `#[ignore = \"requires a CUDA GPU\"]`. See scx-gpu/src/test_gate.rs.",
        failures.join("\n  ")
    );
}
