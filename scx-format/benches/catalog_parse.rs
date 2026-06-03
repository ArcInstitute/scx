//! Criterion microbenchmark for `FullCatalog::read_from` at census scale.
//!
//! Motivation: the `index_plan/scx_auto/{census_500k, census_1m}` cells
//! TIMEOUT at 125/130-min budgets because the workers2 path opens a
//! `BackedCsrReader` per worker, each parsing the v2 catalog over
//! ~16K shards (`census_1m` × `shard_size=16K` cells/shard → 64 shards
//! per shard-stack × further per-modality entries → ~16K total catalog
//! entries depending on layout). Per-shard cost in
//! `ShardStats::read_from` is ~57 bytes for v2 + per-entry `modality_id`
//! u8 — small individually but at 16K entries the sum is non-trivial.
//!
//! This bench gives a hard number for the round-trip cost so any future
//! optimisation (memoise catalog parse / lazy-parse `ShardStats` / drop
//! `col_start`/`col_end` from CSR entries) has a baseline to beat.
//!
//! Run:
//! ```text
//! cargo bench -p scx-format --bench catalog_parse
//! cargo bench -p scx-format --bench catalog_parse -- 'parse/16384'
//! ```

use std::io::Cursor;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use scx_format::catalog::{FullCatalog, FullCatalogEntry, ShardStats};
use scx_format::section::SectionType;

fn make_shard_stats() -> ShardStats {
    // Representative CSR shard stats with 0 indexed columns (the
    // common case when no obs predicates are pushed down to the shard
    // level). With predicates the per-shard cost grows ~16 bytes per
    // ColumnStat, but that's a separate axis.
    ShardStats {
        row_start: 0,
        row_end: 16_384,
        col_start: 0,
        col_end: 60_000,
        nnz: 32_768_000,
        value_min: 0,
        value_max: 65_535,
        value_sum: 100_000_000,
        n_indexed_columns: 0,
        column_stats: Vec::new(),
    }
}

fn make_catalog(n_shards: usize) -> FullCatalog {
    let mut entries = Vec::with_capacity(n_shards + 6);
    // Non-shard entries (obs / var metadata, indexes, provenance, uns).
    for (name, stype) in [
        ("obs", SectionType::ObsMetadata),
        ("obs_index", SectionType::ObsIndex),
        ("var", SectionType::VarMetadata),
        ("var_index", SectionType::VarIndex),
        ("provenance", SectionType::Provenance),
        ("uns", SectionType::UnsBlob),
    ] {
        entries.push(FullCatalogEntry {
            name: name.to_string(),
            offset: 4352 + (name.len() as u64) * 1000,
            length: 50_000,
            section_type: stype,
            checksum: [0xCD; 32],
            modality_id: 0,
            stats: None,
        });
    }
    // CSR shards — the population we're profiling.
    for i in 0..n_shards {
        entries.push(FullCatalogEntry {
            name: format!("X_shard_{}", i),
            offset: 100_000 + (i as u64) * 200_000,
            length: 50_000,
            section_type: SectionType::CsrShard,
            checksum: [0xEF; 32],
            modality_id: 0,
            stats: Some(make_shard_stats()),
        });
    }
    FullCatalog {
        catalog_version: 2,
        manifest_sequence: 1,
        prev_catalog_offset: 0,
        n_obs: (n_shards as u64) * 16_384,
        entries,
        data_generation: 0,
        csc_build_generation: 0,
    }
}

fn bench_catalog_parse(c: &mut Criterion) {
    let mut group = c.benchmark_group("FullCatalog::read_from");
    // 64 = small fixture (pbmc10k-ish), 1024 = tabula_sapiens_100k-ish,
    // 16384 = census_1m-ish (the actual TIMEOUT scenario).
    for n_shards in [64usize, 1024, 16384] {
        let catalog = make_catalog(n_shards);
        let mut buf = Vec::new();
        catalog.write_to(&mut buf).expect("write");
        let total_len = buf.len();
        group.throughput(Throughput::Elements(catalog.entries.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("parse", n_shards),
            &(buf, total_len),
            |b, (buf, total_len)| {
                b.iter(|| {
                    let mut cur = Cursor::new(buf.as_slice());
                    let parsed =
                        FullCatalog::read_from(&mut cur, *total_len, false).expect("parse");
                    black_box(parsed);
                });
            },
        );
    }
    group.finish();
}

fn bench_stats_only(c: &mut Criterion) {
    // Isolate the per-stats cost — useful for sizing the impact of any
    // optimisation that touches ShardStats specifically (lazy-parse,
    // dropping col_start/col_end on CSR entries, ...).
    let mut group = c.benchmark_group("ShardStats::read_from");
    let stats = make_shard_stats();
    let mut buf = Vec::new();
    stats.write_to(&mut buf).expect("write");
    group.throughput(Throughput::Elements(1));
    group.bench_function("one_v2_stats", |b| {
        b.iter(|| {
            let mut cur = Cursor::new(buf.as_slice());
            let parsed = ShardStats::read_from(&mut cur, 2).expect("parse");
            black_box(parsed);
        });
    });
    group.finish();
}

criterion_group!(benches, bench_catalog_parse, bench_stats_only);
criterion_main!(benches);
