//! A rewrite op's same-pass CSC sidecar is byte-identical to the one a second
//! pass would build over the op's output.
//!
//! Before same-pass building, `--rebuild-csc` ran the op without a sidecar and
//! then `rebuild_csc_inplace` over the finished file. Each test here runs the
//! op both ways — `--csc always`, and `--csc off` followed by that second pass
//! — and requires:
//!
//! * the CSC entries to agree on `(name, length, checksum)`, i.e. the same
//!   bytes, so the same-pass sidecar is not merely *a* transpose but the one
//!   users already had;
//! * every other entry but provenance (a wall-clock timestamp) to agree
//!   between the two op runs, so building the sidecar changes nothing else in
//!   the output;
//! * the sidecar to be the output's own transpose, and fresh.
//!
//! Each X write path the writer feeds is reached by at least one op: compact,
//! merge_sorted and sort's row emitter (`write_csr_shard`); optimize, sort's
//! external and grouped paths (`write_preencoded_shard`); merge's raw-copy fast
//! path (`write_csr_shard_raw_copy`).

mod common;

use std::path::Path;

use common::fixture_all_families;
use scx_format_io::section::SectionType;
use scx_format_io::ScxReader;
use scx_ops::{CscCarryOptions, CscOutput};

/// Narrow enough that every fixture here gets more than one CSC shard.
const COLS: usize = 4;

fn csc(mode: CscOutput) -> CscCarryOptions {
    CscCarryOptions {
        cols_per_shard: COLS,
        ..CscCarryOptions::with_mode(mode)
    }
}

type Entry = (String, SectionType, u64, [u8; 32]);

fn entries(path: &Path, want_csc: bool) -> Vec<Entry> {
    let r = ScxReader::open(path).unwrap();
    let mut v: Vec<Entry> = r
        .catalog()
        .entries
        .iter()
        .filter(|e| (e.section_type == SectionType::CscShard) == want_csc)
        // Provenance carries a wall-clock timestamp, so two runs of one op
        // can differ in it across a second boundary.
        .filter(|e| e.section_type != SectionType::Provenance)
        .map(|e| (e.name.clone(), e.section_type, e.length, e.checksum))
        .collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    v
}

fn assert_csc_is_the_transpose(path: &Path) {
    let r = ScxReader::open(path).unwrap();
    let csr = r.read_all_csr_shards().unwrap();
    let csc = r.read_all_csc_shards().unwrap();
    let (n_obs, n_vars) = (r.n_obs() as usize, r.n_vars() as usize);
    let mut a = vec![0f32; n_obs * n_vars];
    for row in 0..n_obs {
        for k in csr.indptr[row] as usize..csr.indptr[row + 1] as usize {
            a[row * n_vars + csr.indices[k] as usize] = csr.data[k];
        }
    }
    let mut b = vec![0f32; n_obs * n_vars];
    for col in 0..n_vars {
        for k in csc.indptr[col] as usize..csc.indptr[col + 1] as usize {
            b[csc.indices[k] as usize * n_vars + col] = csc.data[k];
        }
    }
    assert_eq!(a, b, "{}: the sidecar is not X's transpose", path.display());
    assert_eq!(
        r.catalog().csc_build_generation,
        r.catalog().data_generation,
        "{}: a same-pass sidecar is fresh",
        path.display()
    );
}

/// Run `op` with `--csc always` and with `--csc off` + a second pass, and
/// compare. `op(output, csc)` writes one output.
fn same_pass_equals_second_pass(dir: &Path, tag: &str, op: impl Fn(&Path, CscCarryOptions)) {
    let always = dir.join(format!("{tag}_always.scx"));
    op(&always, csc(CscOutput::Always));

    let off = dir.join(format!("{tag}_off.scx"));
    op(&off, csc(CscOutput::Off));
    assert!(entries(&off, true).is_empty(), "{tag}: --csc off wrote CSC");
    let off_rest = entries(&off, false);
    scx_ops::rebuild_csc_inplace(
        &off,
        COLS,
        "4G",
        scx_ops::framing_for_csc_rebuild(&off),
        None,
    )
    .unwrap();

    let same_pass = entries(&always, true);
    assert!(
        same_pass.len() > 1,
        "{tag}: expected a multi-shard sidecar, got {}",
        same_pass.len()
    );
    assert_eq!(same_pass, entries(&off, true), "{tag}: CSC bytes differ");
    assert_eq!(
        entries(&always, false),
        off_rest,
        "{tag}: building the sidecar changed another section"
    );
    assert_csc_is_the_transpose(&always);
}

#[test]
fn compact() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    for (tag, src) in [
        ("all", fixture_all_families(d, "all.scx")),
        (
            "mixed",
            scx_testkit::fixtures::mixed_codec_file(&d.join("mixed.scx")).unwrap(),
        ),
    ] {
        same_pass_equals_second_pass(d, &format!("compact_{tag}"), |out, csc| {
            let opts = scx_ops::CompactOptions {
                csc,
                ..Default::default()
            };
            scx_ops::compact_with_options(&src, out, &opts).unwrap();
        });
    }
}

#[test]
fn sort_every_strategy() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let src = fixture_all_families(d, "src.scx");
    for (tag, strategy, group_by, memory_budget) in [
        ("inmem", scx_ops::SortStrategy::InMemory, None, None),
        (
            "external",
            scx_ops::SortStrategy::ExternalPartition,
            None,
            None,
        ),
        // Under a budget the builder and the sort split it
        // (`csc_budget::same_pass_split`); the split moves only the builder's
        // spill threshold and the sort's plan, never a byte.
        (
            "external_budget",
            scx_ops::SortStrategy::ExternalPartition,
            None,
            Some(1u64 << 20),
        ),
        // Grouped + in-memory is the parallel fast path, which writes
        // pre-encoded shards rather than going through the row emitter.
        (
            "grouped",
            scx_ops::SortStrategy::InMemory,
            Some("cell_type"),
            None,
        ),
    ] {
        same_pass_equals_second_pass(d, &format!("sort_{tag}"), |out, csc| {
            let opts = scx_ops::SortOptions {
                by: if group_by.is_some() {
                    Vec::new()
                } else {
                    vec!["cell_type".to_string()]
                },
                group_by: group_by.map(str::to_string),
                memory_budget,
                csc,
                ..Default::default()
            };
            scx_ops::sort_engine::sort_with_strategy(&src, out, &opts, Some(strategy)).unwrap();
        });
    }
}

#[test]
fn optimize_framed_and_not() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let src = scx_testkit::fixtures::mixed_codec_file(&d.join("mixed.scx")).unwrap();
    for (tag, framing) in [
        ("unframed", None),
        ("framed", Some(scx_format_io::FramingConfig::default())),
    ] {
        same_pass_equals_second_pass(d, &format!("optimize_{tag}"), |out, csc| {
            scx_ops::optimize_with_csc(
                &src,
                out,
                None,
                scx_format_io::ObsShardPolicy::Off,
                framing,
                None,
                &csc,
            )
            .unwrap();
        });
    }
}

#[test]
fn merge_concat_and_raw_copy() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // Two all-v4 inputs: their framed shards take the raw-copy fast path, so
    // the sidecar builder is fed bytes it has to decode.
    let m1 = scx_testkit::fixtures::mixed_codec_file(&d.join("m1.scx")).unwrap();
    let m2 = scx_testkit::fixtures::mixed_codec_file(&d.join("m2.scx")).unwrap();
    let a1 = fixture_all_families(d, "a1.scx");
    let a2 = fixture_all_families(d, "a2.scx");
    for (tag, a, b) in [("mixed", &m1, &m2), ("all", &a1, &a2)] {
        same_pass_equals_second_pass(d, &format!("merge_{tag}"), |out, csc| {
            let opts = scx_ops::MergeOptions {
                csc,
                ..Default::default()
            };
            scx_ops::merge_with_options(&[a.as_path(), b.as_path()], out, &opts).unwrap();
        });
    }
}

#[test]
fn merge_sorted() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    // Sorted merge needs sorted runs, and refuses obsm (so not the
    // all-families fixture).
    let mut runs = Vec::new();
    for i in 0..2 {
        let src = scx_testkit::fixtures::mixed_codec_file(&d.join(format!("src{i}.scx"))).unwrap();
        let run = d.join(format!("run{i}.scx"));
        let opts = scx_ops::SortOptions {
            by: vec!["cell_type".to_string()],
            ..Default::default()
        };
        scx_ops::sort_engine::sort(&src, &run, &opts).unwrap();
        runs.push(run);
    }
    same_pass_equals_second_pass(d, "merge_sorted", |out, csc| {
        let opts = scx_ops::MergeOptions {
            sort_by: vec!["cell_type".to_string()],
            csc,
            ..Default::default()
        };
        scx_ops::merge_with_options(&[runs[0].as_path(), runs[1].as_path()], out, &opts).unwrap();
    });
}

/// The default is `carry`: an input with a sidecar gives an output with one,
/// an input without gives none.
#[test]
fn carry_is_the_default_and_follows_the_input() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let plain = fixture_all_families(d, "plain.scx");
    let carried = common::fixture_all_families_with_csc(d, "carried.scx", COLS);

    for (src, want) in [(&plain, false), (&carried, true)] {
        let out = d.join("compact_default.scx");
        scx_ops::compact(src, &out).unwrap();
        assert_eq!(ScxReader::open(&out).unwrap().header().has_csc(), want);
        if want {
            assert_csc_is_the_transpose(&out);
        }

        let out = d.join("sort_default.scx");
        let opts = scx_ops::SortOptions {
            by: vec!["cell_type".to_string()],
            ..Default::default()
        };
        scx_ops::sort_engine::sort(src, &out, &opts).unwrap();
        assert_eq!(ScxReader::open(&out).unwrap().header().has_csc(), want);

        let out = d.join("optimize_default.scx");
        scx_ops::optimize(src, &out, None, scx_format_io::ObsShardPolicy::Off).unwrap();
        assert_eq!(ScxReader::open(&out).unwrap().header().has_csc(), want);

        let out = d.join("merge_default.scx");
        scx_ops::merge(&[src.as_path(), plain.as_path()], &out).unwrap();
        assert_eq!(ScxReader::open(&out).unwrap().header().has_csc(), want);
    }
}

/// `--csc always` on a multimodal input is refused before the output exists:
/// the same-pass builder is single-modality. (It used to be accepted, the op
/// would write its output, and `--rebuild-csc`'s second pass would then fail
/// on it.)
#[test]
fn always_on_a_multimodal_input_is_refused_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let d = dir.path();
    let src = common::fixture_multimodal_per_modality_obsp(d, "mm.scx");
    let always = csc(CscOutput::Always);

    let out = d.join("compact.scx");
    let opts = scx_ops::CompactOptions {
        csc: always.clone(),
        ..Default::default()
    };
    let err = scx_ops::compact_with_options(&src, &out, &opts).unwrap_err();
    assert!(err.to_string().contains("--csc always"), "{err}");
    assert!(!out.exists());

    let out = d.join("merge.scx");
    let opts = scx_ops::MergeOptions {
        csc: always.clone(),
        ..Default::default()
    };
    let err =
        scx_ops::merge_with_options(&[src.as_path(), src.as_path()], &out, &opts).unwrap_err();
    assert!(err.to_string().contains("--csc always"), "{err}");
    assert!(!out.exists());

    let out = d.join("sort.scx");
    let opts = scx_ops::SortOptions {
        by: vec!["cell_type".to_string()],
        csc: always,
        ..Default::default()
    };
    let err = scx_ops::sort_engine::sort(&src, &out, &opts).unwrap_err();
    assert!(err.to_string().contains("--csc always"), "{err}");
    assert!(!out.exists());

    // `carry` on the same input is not an error.
    let out = d.join("compact_carry.scx");
    scx_ops::compact(&src, &out).unwrap();
}
