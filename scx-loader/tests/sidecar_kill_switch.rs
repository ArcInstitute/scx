//! T4.1 / acceptance criterion 5 — the `SCX_SCATTER_SIDECAR` env kill-switch.
//!
//! Dedicated test binary because `scatter_sidecar_enabled()` caches the env via
//! a `OnceLock` on first read, so it cannot be toggled inside the shared
//! lib-test process. Here the env is set to `0` before any sidecar code runs,
//! so the (sidecar-carrying) Scx1 fixture is forced onto the full-shard path and
//! must produce output byte-identical to a `CodecId::None` twin.
//!
//! The matching env-ON case — the same fixture + plan *does* reach the sidecar —
//! is `index_plan::tests::gather_pairs_dense_reaches_sidecar_and_matches_full_decode`,
//! so this test is not vacuous: it proves the switch flips an otherwise-eligible
//! gather back to legacy behaviour.

mod common;

use std::sync::atomic::Ordering;

use common::{count_sidecars, write_dense_scx1_fixture};
use scx_codec::CodecId;
use scx_loader::{IndexPlanLoader, LoaderConfig};

#[test]
fn scx_scatter_sidecar_env_zero_forces_full_shard_and_matches() {
    // Must be set before the first `scatter_sidecar_enabled()` read (OnceLock).
    // This is the only test in this binary, so the read happens here first.
    std::env::set_var("SCX_SCATTER_SIDECAR", "0");

    let dir = tempfile::tempdir().unwrap();
    let scx1 = write_dense_scx1_fixture(&dir.path().join("scx1.scx"), 256, 8, 256, CodecId::Scx1);
    let none = write_dense_scx1_fixture(&dir.path().join("none.scx"), 256, 8, 256, CodecId::None);
    assert_eq!(
        count_sidecars(&scx1),
        8,
        "fixture must carry sidecars — else the kill-switch isn't the reason none are used"
    );

    // Sparse-per-shard plan that would be sidecar-eligible with the switch on.
    let plan: Vec<(u64, u64)> = vec![(3, 200), (40, 200), (70, 12), (130, 130), (250, 5)];
    let mk = |p: &std::path::Path| {
        let mut config = LoaderConfig::default();
        config.normalize = false;
        config.log1p = false;
        config.obs_columns = vec!["cell_id".to_string()];
        config.max_memory_mb = 4096;
        IndexPlanLoader::new(
            p, config, /*cache_shards*/ 8, /*sort_by_shard*/ false, /*lookahead*/ 0,
            /*max_plan_size*/ 256, /*scatter_sidecar*/ true,
        )
        .unwrap()
    };
    let scx1_loader = mk(&scx1);
    let none_loader = mk(&none);
    let sb = scx1_loader.process_plan(plan.clone()).unwrap();
    let nb = none_loader.process_plan(plan.clone()).unwrap();

    // Kill-switch engaged: even the sidecar-carrying file takes full-shard.
    assert_eq!(
        scx1_loader
            .cache_metrics()
            .sidecar_groups
            .load(Ordering::Relaxed),
        0,
        "SCX_SCATTER_SIDECAR=0 must force full-shard even on a sidecar fixture"
    );
    assert!(
        scx1_loader
            .cache_metrics()
            .full_shard_groups
            .load(Ordering::Relaxed)
            > 0,
        "groups must be served via full-shard decode under the kill-switch"
    );

    // Output still byte-identical — legacy behaviour fully restored.
    assert_eq!(sb.pairs, nb.pairs);
    let bits = |b: &[f32]| b.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&sb.x),
        bits(&nb.x),
        "perturbed-side must match the None twin"
    );
    assert_eq!(
        bits(&sb.x_paired),
        bits(&nb.x_paired),
        "control-side must match the None twin"
    );
}
