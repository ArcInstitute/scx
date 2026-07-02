//! Diagnostic harness: times the bare Rust `ScxReader::read_obs()` on a
//! high-shard-count file, bypassing pyo3 / AnnData / the query engine. Used to
//! isolate whether a slow full-obs read lives in obs assembly itself or
//! downstream in the query/collect path — `read_obs()` here exercises only the
//! reader, so a fast time here points the finger elsewhere.
//!
//! This is **not** a correctness test — it is `#[ignore]`d and run manually
//! against a real repro shard:
//!
//! ```bash
//! # default-threaded:
//! REPRO_SHARD=/large_storage/.../CD14_Mono_described.scx \
//!   cargo test --release -p scx-format-io --test read_obs_timing -- --ignored --nocapture
//!
//! # single rayon thread (A/B for the collect-path contention hypothesis —
//! # read_obs itself is not rayon-parallel, so a large delta would implicate
//! # something else):
//! RAYON_NUM_THREADS=1 REPRO_SHARD=/large_storage/.../CD14_Mono_described.scx \
//!   cargo test --release -p scx-format-io --test read_obs_timing -- --ignored --nocapture
//! ```
//!
//! Record both wall-clock timings to feed the decision matrix in the spec
//! (< 30 s → read_obs() is a legitimate workaround; hang → the obs assembler
//! itself is the bug and the binding does not work around it).

use std::time::Instant;

use scx_format_io::ScxReader;

#[test]
#[ignore = "manual diagnostic; needs REPRO_SHARD pointing at a real high-shard-count .scx"]
fn read_obs_timing() {
    let path = std::env::var("REPRO_SHARD").expect(
        "set REPRO_SHARD to a .scx path, e.g. \
         /large_storage/.../CD14_Mono_described.scx",
    );

    let open_start = Instant::now();
    let reader = ScxReader::open(&path).expect("failed to open REPRO_SHARD");
    let open_elapsed = open_start.elapsed();

    let n_obs = reader.n_obs();
    let shard_count = reader.obs_metadata_shard_count();
    let threads = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "default".into());

    let read_start = Instant::now();
    let obs = reader.read_obs().expect("read_obs failed");
    let read_elapsed = read_start.elapsed();

    eprintln!(
        "read_obs_timing: path={path}\n  \
         n_obs={n_obs} obs_metadata_shard_count={shard_count} rayon_threads={threads}\n  \
         open={open_elapsed:?} read_obs={read_elapsed:?} rows={} cols={}",
        obs.num_rows(),
        obs.num_columns(),
    );
}
