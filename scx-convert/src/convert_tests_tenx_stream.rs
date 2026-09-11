//! Streaming `--from 10x` (OPT-CONVERT-9).
//!
//! `tenx_to_scx` reads `/matrix/{indptr,indices,data}` whole; `tenx_to_scx_streaming`
//! reads the same group one row range at a time through the shared
//! [`crate::h5ad::stream::XStreamReader`]. These tests pin what the two paths
//! agree on, what they deliberately do not, and the axis swap that makes the
//! reuse correct.
//!
//! **The axis swap is the whole risk.** 10x writes `shape = [n_genes, n_cells]`,
//! the transpose of the `(n_obs, n_vars)` that `read_shape_2d` returns for an
//! h5ad sparse group. A streaming path built by pointing `open_x_streaming` at
//! `"matrix"` — the obvious route — gets an obs axis of *genes*, and on a square
//! fixture that is invisible. Every fixture here therefore has
//! `n_genes != n_cells`.
//!
//! **What the two paths do not agree on: the codec seed.** `tenx_to_scx`
//! detects one value encoding over the whole matrix and forces the resulting
//! codec into every shard; the coordinator passes `opts.codec` through, which is
//! `None` under the default `--codec auto`, so each shard seeds its own codec
//! from its own values. `select_codec` samples only the first 10 000 values, so
//! shard 0 agrees by construction and later shards can differ. The eager/streaming
//! h5ad pair has had exactly this divergence since streaming shipped;
//! `streaming_tenx_seeds_the_codec_per_shard` pins it so a later "make these
//! consistent" change has to argue with a test rather than a comment.

use super::convert_tests_common::*;
use scx_format_io::section::SectionType;
use scx_testkit::digest::{assert_digests_eq, digest_file, Strictness};

/// Where `/matrix/shape` lives. Real CellRanger and CellBender write a dataset;
/// `create_test_tenx_h5` writes an attribute. `read_tenx_shape` accepts both, so
/// both belong in the streaming tests.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShapeForm {
    Dataset,
    Attr,
}

/// Write a 10x v3 `/matrix` group: CSC over genes×cells, which is the CSR over
/// cells×genes the readers reinterpret.
///
/// `value_for(cell, gene)` supplies the count, so a caller can shape the
/// per-shard value distribution — which is what decides the codec seed.
fn write_tenx(
    path: &Path,
    n_cells: usize,
    n_genes: usize,
    shape_form: ShapeForm,
    nnz_per_cell: usize,
    value_for: impl Fn(usize, usize) -> f32,
) {
    let file = hdf5::File::create(path).unwrap();
    let matrix = file.create_group("matrix").unwrap();

    let mut indptr = vec![0i64];
    let mut indices: Vec<i32> = Vec::new();
    let mut data: Vec<f32> = Vec::new();
    for cell in 0..n_cells {
        let mut genes: Vec<usize> = (0..nnz_per_cell)
            .map(|k| (cell * nnz_per_cell + k) % n_genes)
            .collect();
        genes.sort_unstable();
        genes.dedup();
        for g in genes {
            indices.push(g as i32);
            data.push(value_for(cell, g));
        }
        indptr.push(data.len() as i64);
    }

    match shape_form {
        ShapeForm::Dataset => matrix
            .new_dataset::<i32>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_genes as i32, n_cells as i32])
            .unwrap(),
        ShapeForm::Attr => matrix
            .new_attr::<i64>()
            .shape([2])
            .create("shape")
            .unwrap()
            .write(&[n_genes as i64, n_cells as i64])
            .unwrap(),
    }
    matrix
        .new_dataset::<i64>()
        .shape([indptr.len()])
        .create("indptr")
        .unwrap()
        .write(&indptr)
        .unwrap();
    matrix
        .new_dataset::<i32>()
        .shape([indices.len()])
        .create("indices")
        .unwrap()
        .write(&indices)
        .unwrap();
    matrix
        .new_dataset::<f32>()
        .shape([data.len()])
        .create("data")
        .unwrap()
        .write(&data)
        .unwrap();

    let barcodes: Vec<VarLenUnicode> = (0..n_cells)
        .map(|i| vlu(&format!("BARCODE{i:05}-1")))
        .collect();
    matrix
        .new_dataset::<VarLenUnicode>()
        .shape([n_cells])
        .create("barcodes")
        .unwrap()
        .write(&barcodes)
        .unwrap();

    let features = matrix.create_group("features").unwrap();
    for (name, mk) in [
        (
            "id",
            Box::new(|i: usize| format!("ENSG{i:08}")) as Box<dyn Fn(usize) -> String>,
        ),
        ("name", Box::new(|i: usize| format!("Gene{i}"))),
        (
            "feature_type",
            Box::new(|_: usize| "Gene Expression".to_string()),
        ),
    ] {
        let vals: Vec<VarLenUnicode> = (0..n_genes).map(|i| vlu(&mk(i))).collect();
        features
            .new_dataset::<VarLenUnicode>()
            .shape([n_genes])
            .create(name)
            .unwrap()
            .write(&vals)
            .unwrap();
    }
}

/// Per-CSR-shard `codec_id`, read off each shard header in catalog order.
fn shard_codecs(path: &Path) -> Vec<u8> {
    let reader = ScxReader::open(path).unwrap();
    let bytes = std::fs::read(path).unwrap();
    let mut out = Vec::new();
    for entry in &reader.catalog().entries {
        if entry.section_type != SectionType::CsrShard {
            continue;
        }
        let section = &bytes[entry.offset as usize..][..entry.length as usize];
        let sh = scx_format_io::shard::ShardHeader::read_from(&mut std::io::Cursor::new(
            &section[..scx_format_io::shard::SHARD_HEADER_SIZE],
        ))
        .unwrap();
        out.push(sh.codec_id);
    }
    out
}

fn convert_both(
    dir: &Path,
    tenx: &Path,
    opts: &IngestOptions,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let streamed = dir.join("stream.scx");
    let eager = dir.join("eager.scx");
    tenx_to_scx_streaming(tenx, &streamed, opts, &mut WarningSink::log()).unwrap();
    tenx_to_scx(tenx, &eager, opts, &mut WarningSink::log()).unwrap();
    (streamed, eager)
}

/// Single shard, default options: the two paths agree **byte for byte** on every
/// section but `Provenance` (which the digest excludes, and which legitimately
/// differs — the streaming stamp carries `"stream": true`).
///
/// This is the strongest identity claim in the tree for a stream/eager pair:
/// before this, the only such proof was value-level (`read_all_csr_shards`).
/// It holds here because one shard's values *are* the whole matrix's, so the
/// file-wide codec seed and the per-shard seed are the same number.
#[test]
fn streaming_tenx_matches_the_eager_output_on_a_single_shard() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    write_tenx(&tenx, 20, 6, ShapeForm::Dataset, 3, |c, g| {
        ((c + g) % 7 + 1) as f32
    });

    let (streamed, eager) = convert_both(dir.path(), &tenx, &IngestOptions::default());
    assert_digests_eq(
        &digest_file(&streamed, Strictness::Content).unwrap(),
        &digest_file(&eager, Strictness::Content).unwrap(),
    );
}

/// The axis swap. `shape[1]` is cells and becomes `n_obs`; `shape[0]` is genes
/// and becomes `n_vars`.
///
/// Watched red by dropping the swap in `open_tenx_x_streaming` (i.e. by taking
/// `read_shape_2d`'s `(n_obs, n_vars)` order, which is what the filed
/// prescription for this item would have produced): the open then fails with
/// `indptr length 21 != n_obs + 1 (7)` — the `indptr` is over cells while the
/// unswapped shape claims 6 rows.
#[test]
fn streaming_tenx_takes_n_cells_as_the_obs_axis() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    write_tenx(&tenx, 20, 6, ShapeForm::Dataset, 2, |c, _| {
        (c % 5 + 1) as f32
    });

    let out = dir.path().join("out.scx");
    tenx_to_scx_streaming(
        &tenx,
        &out,
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.header().n_obs, 20, "cells are the obs axis");
    assert_eq!(reader.header().n_vars, 6, "genes are the var axis");
    assert_eq!(reader.read_obs().unwrap().num_rows(), 20);
    assert_eq!(reader.read_var().unwrap().num_rows(), 6);
}

/// Both `shape` spellings stream, and to the same file. Real 10x output writes
/// the dataset; `create_test_tenx_h5`-style fixtures write the attribute.
///
/// Watched red by deleting either arm of `read_tenx_shape`.
#[test]
fn streaming_tenx_reads_both_shape_spellings() {
    let dir = tempfile::tempdir().unwrap();
    let ds = dir.path().join("ds.h5");
    let attr = dir.path().join("attr.h5");
    let value_for = |c: usize, g: usize| ((c + g) % 9 + 1) as f32;
    write_tenx(&ds, 14, 5, ShapeForm::Dataset, 2, value_for);
    write_tenx(&attr, 14, 5, ShapeForm::Attr, 2, value_for);

    let from_ds = dir.path().join("from_ds.scx");
    let from_attr = dir.path().join("from_attr.scx");
    let opts = IngestOptions::default();
    tenx_to_scx_streaming(&ds, &from_ds, &opts, &mut WarningSink::log()).unwrap();
    tenx_to_scx_streaming(&attr, &from_attr, &opts, &mut WarningSink::log()).unwrap();

    assert_digests_eq(
        &digest_file(&from_ds, Strictness::Content).unwrap(),
        &digest_file(&from_attr, Strictness::Content).unwrap(),
    );
}

/// A 10x `/matrix` has no `encoding-type` attribute, so routing it through the
/// h5ad opener would emit `ConvertWarning::InferredEncoding` on every convert —
/// a warning the eager 10x path never emits. Neither path should.
///
/// Watched red by making `open_tenx_x_streaming` emit the warning (done, and
/// this test went red with `left: 1, right: 0`) — so the counter is wired, and
/// the assertion is not passing because nothing could ever increment it.
#[test]
fn streaming_tenx_emits_no_inferred_encoding_warning() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    write_tenx(&tenx, 12, 4, ShapeForm::Dataset, 2, |c, _| {
        (c % 4 + 1) as f32
    });

    let out = dir.path().join("out.scx");
    let inferred = std::sync::Arc::new(std::sync::Mutex::new(0u64));
    let seen = inferred.clone();
    let mut sink = WarningSink::with_handler(move |w| {
        if matches!(w, super::warnings::ConvertWarning::InferredEncoding { .. }) {
            *seen.lock().unwrap() += 1;
        }
    });
    tenx_to_scx_streaming(&tenx, &out, &IngestOptions::default(), &mut sink).unwrap();

    assert_eq!(
        *inferred.lock().unwrap(),
        0,
        "10x has no encoding-type attr by design; warning about it is noise"
    );
}

/// **The deliberate divergence.** Multi-shard, and the last shard's own value
/// distribution lands on the other side of `select_codec`'s median-8 boundary
/// from the whole matrix's.
///
/// The values, the shard row ranges and the shard count agree; at least one
/// shard's `codec_id` does not. That is the contract chosen for this direction
/// (per-shard seeding, matching every other streaming ingest path and the
/// documented `codec="auto"` intent), not an accident.
///
/// Watched red by forcing the eager file-wide seed into the coordinator
/// (`enc_opts.explicit_codec = Some(detect_value_encoding(..).1)`), which makes
/// the codecs equal and this assertion fail.
#[test]
fn streaming_tenx_seeds_the_codec_per_shard() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    // 18 low-count cells then 2 high-count ones: the whole matrix's median is
    // small (Scx1), the second shard's is not (Zstd).
    write_tenx(&tenx, 20, 6, ShapeForm::Dataset, 2, |c, _| {
        if c < 18 {
            (c % 2 + 1) as f32
        } else {
            (100 * (c - 17)) as f32
        }
    });

    let opts = IngestOptions {
        shard_target_rows: 18,
        ..IngestOptions::default()
    };
    let (streamed, eager) = convert_both(dir.path(), &tenx, &opts);

    let a = ScxReader::open(&streamed).unwrap();
    let b = ScxReader::open(&eager).unwrap();
    assert_eq!(a.header().n_csr_shards, 2, "fixture must be multi-shard");
    assert_eq!(a.header().n_obs, b.header().n_obs);
    assert_eq!(a.header().nnz, b.header().nnz);
    assert_eq!(a.header().n_csr_shards, b.header().n_csr_shards);

    let csr_a = a.read_all_csr_shards().unwrap();
    let csr_b = b.read_all_csr_shards().unwrap();
    assert_eq!(csr_a.shape, csr_b.shape);
    assert_eq!(csr_a.indptr, csr_b.indptr, "shard row ranges must agree");
    assert_eq!(csr_a.indices, csr_b.indices);
    assert_eq!(csr_a.data, csr_b.data, "values must agree exactly");

    let codecs_stream = shard_codecs(&streamed);
    let codecs_eager = shard_codecs(&eager);
    // Measured, not assumed: the whole matrix's median puts Scx1 on both eager
    // shards, while shard 1's own median (100, 200) picks Zstd.
    let (scx1, zstd) = (CodecId::Scx1 as u8, CodecId::Zstd as u8);
    assert_eq!(
        codecs_eager,
        vec![scx1, scx1],
        "eager forces the one file-wide seed onto every shard"
    );
    assert_eq!(
        codecs_stream,
        vec![scx1, zstd],
        "streaming seeds per shard; if this became [Scx1, Scx1] the divergence \
         this test documents is gone — decide that deliberately, do not delete \
         the test"
    );

    // Positive control: the divergence is *only* about the unforced seed. Name a
    // codec and both paths honour it on every shard, so the two files agree
    // byte for byte on the same multi-shard fixture that diverges above.
    let forced = IngestOptions {
        codec: Some(CodecId::Zstd),
        ..opts
    };
    let dir2 = tempfile::tempdir().unwrap();
    let (streamed2, eager2) = convert_both(dir2.path(), &tenx, &forced);
    assert_eq!(shard_codecs(&streamed2), vec![zstd, zstd]);
    assert_eq!(shard_codecs(&eager2), vec![zstd, zstd]);
    assert_digests_eq(
        &digest_file(&streamed2, Strictness::Content).unwrap(),
        &digest_file(&eager2, Strictness::Content).unwrap(),
    );
}

/// `--csc always` on the streaming path goes through `rebuild_csc_inplace`
/// rather than a resident transpose, the same two-pass shape the streaming h5ad
/// route uses. The sidecar it produces must match the eager one's layout.
#[test]
fn streaming_tenx_csc_always_matches_the_eager_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    write_tenx(&tenx, 16, 12, ShapeForm::Dataset, 3, |c, g| {
        ((c * g) % 6 + 1) as f32
    });

    let opts = IngestOptions {
        csc: crate::pipeline::CscPolicy::Always,
        csc_cols_per_shard: 5,
        ..IngestOptions::default()
    };
    let (streamed, eager) = convert_both(dir.path(), &tenx, &opts);

    let a = ScxReader::open(&streamed).unwrap();
    let b = ScxReader::open(&eager).unwrap();
    assert!(
        a.header().has_csc(),
        "streaming --csc always must emit a sidecar"
    );
    assert_eq!(a.header().n_csc_shards, b.header().n_csc_shards);

    let csc_a: Vec<_> = a
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .map(|e| (e.name.clone(), e.length))
        .collect();
    let csc_b: Vec<_> = b
        .catalog()
        .entries
        .iter()
        .filter(|e| e.section_type == SectionType::CscShard)
        .map(|e| (e.name.clone(), e.length))
        .collect();
    assert_eq!(csc_a, csc_b, "CSC sidecar layout must agree across paths");
}

/// Predicate indexes come from the ranges the coordinator actually emitted, not
/// an assumed partition — the same contract the h5ad streaming path has.
#[test]
fn streaming_tenx_builds_predicate_indexes_over_the_emitted_shards() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    write_tenx(&tenx, 30, 7, ShapeForm::Dataset, 2, |c, _| {
        (c % 3 + 1) as f32
    });

    let opts = IngestOptions {
        shard_target_rows: 8,
        index_var: vec!["feature_type".to_string()],
        ..IngestOptions::default()
    };
    let (streamed, eager) = convert_both(dir.path(), &tenx, &opts);

    let a = ScxReader::open(&streamed).unwrap();
    let b = ScxReader::open(&eager).unwrap();
    assert_eq!(a.header().n_csr_shards, 4);
    assert_eq!(a.header().n_csr_shards, b.header().n_csr_shards);
    assert_digests_eq(
        &digest_file(&streamed, Strictness::Content).unwrap(),
        &digest_file(&eager, Strictness::Content).unwrap(),
    );
}

/// Obs sharding is decided by the shared `write_ingest_obs` on this path too.
#[test]
fn streaming_tenx_shards_obs_above_the_threshold() {
    let dir = tempfile::tempdir().unwrap();
    let tenx = dir.path().join("in.h5");
    write_tenx(&tenx, 40, 6, ShapeForm::Dataset, 2, |c, _| {
        (c % 4 + 1) as f32
    });

    let out = dir.path().join("out.scx");
    let opts = IngestOptions {
        shard_target_rows: 10,
        ..IngestOptions::default()
    };
    tenx_to_scx_streaming(&tenx, &out, &opts, &mut WarningSink::log()).unwrap();

    let reader = ScxReader::open(&out).unwrap();
    assert_eq!(reader.obs_metadata_shard_count(), 4);
    assert_eq!(reader.read_obs().unwrap().num_rows(), 40);
}

/// An h5ad misrouted to `--from 10x`, and a CellBender output, are rejected on
/// the streaming path too. Streaming is the **default** for this direction, so a
/// gate that lived only in `tenx_to_scx` would be missing from the route
/// everyone takes.
#[test]
fn streaming_tenx_keeps_the_eager_paths_input_gates() {
    let dir = tempfile::tempdir().unwrap();

    let h5ad = dir.path().join("in.h5ad");
    create_test_h5ad(&h5ad, 10, 5, "csr", false);
    let out = dir.path().join("out.scx");
    let err = tenx_to_scx_streaming(
        &h5ad,
        &out,
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap_err();
    assert!(
        matches!(&err, ConvertError::FormatMismatch { expected, got } if expected == "10x" && got == "h5ad"),
        "{err:?}"
    );

    let cb = dir.path().join("cb.h5");
    write_tenx(&cb, 8, 4, ShapeForm::Dataset, 2, |c, _| (c % 3 + 1) as f32);
    {
        let file = hdf5::File::open_rw(&cb).unwrap();
        file.create_group("droplet_latents").unwrap();
    }
    let err = tenx_to_scx_streaming(
        &cb,
        &out,
        &IngestOptions::default(),
        &mut WarningSink::log(),
    )
    .unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("CellBender"), "{msg}");
    assert!(msg.contains("cellbender-import"), "{msg}");
}
