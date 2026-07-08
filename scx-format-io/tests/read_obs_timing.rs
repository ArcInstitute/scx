//! Diagnostic + micro-benchmark harness: times the bare Rust
//! `ScxReader::read_obs()` / `read_var()` on a high-shard-count file, bypassing
//! pyo3 / AnnData / the query engine. Two uses:
//!
//! 1. **Diagnostic** — isolate whether a slow full-obs read lives in obs
//!    assembly itself or downstream in the query/collect path (`read_obs()`
//!    here exercises only the reader).
//! 2. **Parallel-decode A/B** — on a *sharded* file, `read_obs`/`read_var` fan
//!    the per-shard zstd + Arrow-IPC decode across the rayon pool
//!    (`reader::read_sharded_layout_by_prefix`). Compare the parallel path
//!    against the serial path via `SCX_METADATA_DECODE_SERIAL=1`, and scale the
//!    pool via `RAYON_NUM_THREADS`.
//!
//! This is **not** a correctness test — it is `#[ignore]`d and run manually
//! against a real repro shard:
//!
//! ```bash
//! # parallel, default thread count:
//! REPRO_SHARD=/large_storage/.../census_1m_scx1.scx \
//!   cargo test --release -p scx-format-io --test read_obs_timing -- --ignored --nocapture
//!
//! # serial baseline (forces the single-threaded shard decode):
//! SCX_METADATA_DECODE_SERIAL=1 REPRO_SHARD=/.../census_1m_scx1.scx \
//!   cargo test --release -p scx-format-io --test read_obs_timing -- --ignored --nocapture
//!
//! # thread scaling (one process per thread count; rayon reads it at pool init):
//! RAYON_NUM_THREADS=8 REPRO_SHARD=/.../census_1m_scx1.scx \
//!   cargo test --release -p scx-format-io --test read_obs_timing -- --ignored --nocapture
//! ```
//!
//! Reports the median of `N_TIMED` reads (after one warm-up to prime the page
//! cache) for both `read_obs` and `read_var`, plus the config (serial toggle,
//! rayon threads, obs/var shard counts) so the driver can tabulate a
//! serial-vs-parallel speedup and a thread-scaling curve.

use std::time::{Duration, Instant};

use scx_format_io::ScxReader;

/// Timed reads per method (after one warm-up). Odd so the median is a real sample.
const N_TIMED: usize = 5;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

#[test]
#[ignore = "manual diagnostic; needs REPRO_SHARD pointing at a real high-shard-count .scx"]
fn read_obs_timing() {
    let path = std::env::var("REPRO_SHARD").expect(
        "set REPRO_SHARD to a .scx path, e.g. \
         /large_storage/.../census_1m_scx1.scx",
    );

    let open_start = Instant::now();
    let reader = ScxReader::open(&path).expect("failed to open REPRO_SHARD");
    let open_elapsed = open_start.elapsed();

    let n_obs = reader.n_obs();
    let n_vars = reader.n_vars();
    let obs_shards = reader.obs_metadata_shard_count();
    let var_shards = reader.var_metadata_shard_count();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());
    let serial = std::env::var("SCX_METADATA_DECODE_SERIAL")
        .map(|v| v == "1")
        .unwrap_or(false);

    // Warm-up (prime page cache) + N_TIMED timed reads → median. The obs/var
    // bytes are mmap-resident after the first read, so the median reflects the
    // decode (zstd + Arrow IPC + assemble), not disk.
    let obs_cols;
    let obs_rows;
    {
        let warm = reader.read_obs().expect("read_obs warm-up failed");
        obs_rows = warm.num_rows();
        obs_cols = warm.num_columns();
    }
    let mut obs_times = Vec::with_capacity(N_TIMED);
    for _ in 0..N_TIMED {
        let t = Instant::now();
        let _ = reader.read_obs().expect("read_obs failed");
        obs_times.push(t.elapsed());
    }
    let obs_median = median(obs_times.clone());

    let var_cols;
    let var_rows;
    {
        let warm = reader.read_var().expect("read_var warm-up failed");
        var_rows = warm.num_rows();
        var_cols = warm.num_columns();
    }
    let mut var_times = Vec::with_capacity(N_TIMED);
    for _ in 0..N_TIMED {
        let t = Instant::now();
        let _ = reader.read_var().expect("read_var failed");
        var_times.push(t.elapsed());
    }
    let var_median = median(var_times.clone());

    eprintln!(
        "read_obs_timing: path={path}\n  \
         mode={} rayon_threads={threads} n_obs={n_obs} n_vars={n_vars} \
         obs_shards={obs_shards} var_shards={var_shards}\n  \
         open={open_elapsed:?}\n  \
         read_obs: median={obs_median:?} all={obs_times:?} rows={obs_rows} cols={obs_cols}\n  \
         read_var: median={var_median:?} all={var_times:?} rows={var_rows} cols={var_cols}",
        if serial { "serial" } else { "parallel" },
    );
}
