//! Tests for the route planners and the shared routing layer.
//!
//! `#[path]`-included from `route.rs`, so `super` is the `route` module.

use super::*;

/// `plan_de_route_from_source` reads CSC capability from the source's
/// `available_layouts()` (not a separate flag). Asserts only
/// `csc_available`, which `plan_de_route` sets unconditionally — so the
/// test is independent of whether the host happens to have CUDA.
#[cfg(feature = "gpu")]
#[test]
fn plan_from_source_reads_csc_capability() {
    use scx_gpu::{GpuMatrixSource, LayoutSet};

    struct FakeSource(LayoutSet);
    impl GpuMatrixSource for FakeSource {
        fn shape(&self) -> (usize, usize) {
            (100, 50)
        }
        fn available_layouts(&self) -> LayoutSet {
            self.0
        }
    }

    let csr_only = FakeSource(LayoutSet::CSR);
    let info = plan_de_route_from_source(DeviceRequest::Gpu, &csr_only, InputLayout::BackedCsr);
    assert_eq!(info.csc_available, Some(false));

    let with_csc = FakeSource(LayoutSet::CSR | LayoutSet::CSC);
    let info = plan_de_route_from_source(DeviceRequest::Gpu, &with_csc, InputLayout::BackedCsc);
    assert_eq!(info.csc_available, Some(true));
}

#[test]
fn dense_never_routes_to_csc() {
    // Dense host on GPU → CSR-direct v3 (the entry point densifies to CSR),
    // never CSC, even with the csc flag on.
    let info = plan_de_route(DeviceRequest::Gpu, InputLayout::DenseHost, true, true, true);
    assert_eq!(info.route, AccelRoute::GpuCsrV3);
    assert_ne!(info.route, AccelRoute::GpuCscV3);

    // Dense host on CPU → cpu_dense.
    let info = plan_de_route(
        DeviceRequest::Cpu,
        InputLayout::DenseHost,
        true,
        true,
        false,
    );
    assert_eq!(info.route, AccelRoute::CpuDense);
}

#[test]
fn in_memory_csr_is_csr_direct_not_csc() {
    // In-memory CSR has no CSC sidecar: v3 CSR-direct.
    let info = plan_de_route(DeviceRequest::Gpu, InputLayout::CsrHost, true, true, false);
    assert_eq!(info.route, AccelRoute::GpuCsrV3);
    assert_eq!(info.fallback_reason, FallbackReason::NoCscSidecar);
}

#[test]
fn backed_csc_with_sidecar_is_csc_direct() {
    let info = plan_de_route(DeviceRequest::Gpu, InputLayout::BackedCsc, true, true, true);
    assert_eq!(info.route, AccelRoute::GpuCscV3);
    assert_eq!(info.fallback_reason, FallbackReason::None);
    assert_eq!(info.csc_available, Some(true));
}

#[test]
fn lazy_csr_falls_back_to_csr_direct() {
    let info = plan_de_route(DeviceRequest::Gpu, InputLayout::LazyCsr, true, true, false);
    assert_eq!(info.route, AccelRoute::GpuCsrV3);
    assert_eq!(info.fallback_reason, FallbackReason::NoCscSidecar);
}

#[test]
fn backed_csr_without_csc_falls_back() {
    // A backed CSR reader with no CSC sidecar → CSR-direct.
    let info = plan_de_route(
        DeviceRequest::Gpu,
        InputLayout::BackedCsr,
        true,
        true,
        false,
    );
    assert_eq!(info.route, AccelRoute::GpuCsrV3);
    assert_eq!(info.fallback_reason, FallbackReason::NoCscSidecar);
}

#[test]
fn forced_cpu_records_user_forced() {
    let info = plan_de_route(DeviceRequest::Cpu, InputLayout::BackedCsr, true, true, true);
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
}

#[test]
fn no_cuda_auto_records_no_cuda_and_cpu_route() {
    let info = plan_de_route(
        DeviceRequest::Auto,
        InputLayout::BackedCsc,
        false,
        true,
        true,
    );
    assert!(!info.route.is_gpu());
    assert_eq!(info.route, AccelRoute::CpuCsc);
    assert_eq!(info.fallback_reason, FallbackReason::NoCuda);
}

#[test]
fn gpu_present_but_layout_ineligible_records_unsupported_layout() {
    // prefer_format="csc" on a GPU host: GPU is available but there is no
    // GPU CSC kernel, so dispatch runs CPU CSC. The reason must be the
    // layout, not NoCuda (CUDA is present) or UserForcedCpu (user asked
    // for gpu/auto). This is the case `finalize_exec_info` used to infer.
    for device in [DeviceRequest::Gpu, DeviceRequest::Auto] {
        let info = plan_de_route(
            device,
            InputLayout::BackedCsc,
            true,  // gpu_available
            false, // gpu_eligible — no GPU CSC kernel
            true,  // csc_available
        );
        assert!(!info.route.is_gpu());
        assert_eq!(info.route, AccelRoute::CpuCsc);
        assert_eq!(info.fallback_reason, FallbackReason::UnsupportedInputLayout);
        assert_eq!(info.csc_available, Some(true));
    }
}

#[test]
fn sparse_gpu_is_unconditionally_v3() {
    // After the V1b default-flip the planner has no v2/v1 GPU route for
    // sparse inputs: CSC sidecar → CSC-direct, otherwise CSR-direct.
    let info = plan_de_route(DeviceRequest::Gpu, InputLayout::BackedCsc, true, true, true);
    assert_eq!(info.route, AccelRoute::GpuCscV3);
    let info = plan_de_route(
        DeviceRequest::Gpu,
        InputLayout::BackedCsr,
        true,
        true,
        false,
    );
    assert_eq!(info.route, AccelRoute::GpuCsrV3);
}

#[test]
fn nb_glm_gpu_eligible_routes_to_gpu_csr() {
    for device in [DeviceRequest::Gpu, DeviceRequest::Auto] {
        let info = plan_nb_glm_route(device, true, true);
        assert_eq!(info.route, AccelRoute::GpuNbGlmCsr);
        assert!(info.route.is_gpu());
        assert_eq!(info.fallback_reason, FallbackReason::None);
    }
}

#[test]
fn nb_glm_forced_cpu_records_user_forced() {
    let info = plan_nb_glm_route(DeviceRequest::Cpu, true, true);
    assert_eq!(info.route, AccelRoute::CpuNbGlm);
    assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
}

#[test]
fn nb_glm_no_cuda_falls_back_to_cpu() {
    let info = plan_nb_glm_route(DeviceRequest::Auto, false, true);
    assert_eq!(info.route, AccelRoute::CpuNbGlm);
    assert_eq!(info.fallback_reason, FallbackReason::NoCuda);
}

#[test]
fn nb_glm_ineligible_dims_fall_back_to_cpu() {
    // p / n_sub beyond the kernel register bounds → CPU, not NoCuda.
    let info = plan_nb_glm_route(DeviceRequest::Gpu, true, false);
    assert_eq!(info.route, AccelRoute::CpuNbGlm);
    assert_eq!(info.fallback_reason, FallbackReason::UnsupportedDimensions);
    // NB-GLM is native — never a rapids fallback.
    assert_ne!(info.fallback_reason, FallbackReason::NoRapids);
}

#[test]
fn route_strings_are_stable() {
    assert_eq!(AccelRoute::GpuCscV3.as_str(), "gpu_csc_v3");
    assert_eq!(AccelRoute::CpuCsr.as_str(), "cpu_csr");
    // The fused device-resident pipeline route string is a wire contract:
    // the accel_pipeline residency benchmark's `pipeline_route_gpu_correct`
    // gate (V3 task 2.7) matches on exactly this value.
    assert_eq!(
        AccelRoute::GpuDeviceResident.as_str(),
        "gpu_device_resident"
    );
    assert_eq!(FallbackReason::NoCscSidecar.as_str(), "no_csc_sidecar");
    assert_eq!(FallbackReason::None.as_str(), "none");
    // Wire contracts: the rapids route + no_rapids reason are matched by
    // the `*_route_rapids_correct` gates.
    assert_eq!(
        AccelRoute::RapidsSinglecell.as_str(),
        "rapids_singlecell_gpu"
    );
    assert!(AccelRoute::RapidsSinglecell.is_gpu());
    assert_eq!(FallbackReason::NoRapids.as_str(), "no_rapids");
    // Read back out of `uns["scx_accel"]["to_gpu_anndata"]` by
    // `pyscx/tests/test_gpu_device_handoff.py`.
    assert_eq!(
        FallbackReason::GpuRuntimeError.as_str(),
        "gpu_runtime_error"
    );
}

/// No two reasons may serialise to the same string: the value is what a
/// gate and a user branch on, so a collision would make two different
/// diagnoses indistinguishable on the wire.
#[test]
fn fallback_reason_strings_are_distinct() {
    let all = [
        FallbackReason::None,
        FallbackReason::NoCuda,
        FallbackReason::NoCscSidecar,
        FallbackReason::UnsupportedDimensions,
        FallbackReason::UnsupportedInputLayout,
        FallbackReason::UserForcedCpu,
        FallbackReason::PerfPolicy,
        FallbackReason::NoRapids,
        FallbackReason::GpuRuntimeError,
    ];
    let mut seen = std::collections::HashSet::new();
    for r in all {
        assert!(seen.insert(r.as_str()), "duplicate string for {r:?}");
    }
}

/// A runtime fallback is never a *planned* route: every planner decides
/// from pre-flight facts only, and this reason is recorded after the fact
/// by the op that survived. If a planner ever starts returning it, the
/// contract in its doc comment (and this test) needs revisiting.
#[test]
fn no_planner_returns_the_runtime_reason() {
    for device in [DeviceRequest::Cpu, DeviceRequest::Gpu, DeviceRequest::Auto] {
        for gpu_available in [false, true] {
            for gpu_eligible in [false, true] {
                for csc_available in [false, true] {
                    for layout in [
                        InputLayout::DenseHost,
                        InputLayout::CsrHost,
                        InputLayout::BackedCsr,
                        InputLayout::BackedCsc,
                        InputLayout::LazyCsr,
                    ] {
                        let info = plan_de_route(
                            device,
                            layout,
                            gpu_available,
                            gpu_eligible,
                            csc_available,
                        );
                        assert_ne!(info.fallback_reason, FallbackReason::GpuRuntimeError);
                    }
                    let info = plan_hvg_route(device, gpu_available, gpu_eligible, csc_available);
                    assert_ne!(info.fallback_reason, FallbackReason::GpuRuntimeError);
                }
                let info = plan_nb_glm_route(device, gpu_available, gpu_eligible);
                assert_ne!(info.fallback_reason, FallbackReason::GpuRuntimeError);
            }
        }
    }
}

// --- rapids dispatch decision (plan_rapids_decision) ---

#[test]
fn rapids_cpu_device_is_native() {
    // device="cpu" (no gpu_id) → the existing CPU/native dispatch, and the
    // rapids probe must not even be consulted.
    let decision = plan_rapids_decision(None, false, false, || {
        panic!("the rapids probe must not run for a CPU device")
    });
    assert_eq!(decision, RapidsDecision::Native);
}

#[test]
fn rapids_available_routes_to_rapids_with_the_device_id() {
    let decision = plan_rapids_decision(Some(3), false, false, || true);
    assert_eq!(decision, RapidsDecision::Rapids(3));
}

#[test]
fn rapids_absent_falls_back_to_cpu() {
    let decision = plan_rapids_decision(Some(0), false, false, || false);
    assert_eq!(decision, RapidsDecision::NoRapidsCpu);
}

/// `SCX_DISABLE_RAPIDS` is honored ahead of `SCX_FORCE_NATIVE_GPU`, so the
/// Phase-2.2 fallback gate pins the `no_rapids` CPU path rather than the
/// native kernels — and the rapids import probe is never consulted (the
/// disable exists so a test host can exercise the rapids-absent contract
/// without importing rapids at all).
#[test]
fn rapids_disabled_beats_force_native_and_skips_the_probe() {
    let decision = plan_rapids_decision(Some(0), true, true, || {
        panic!("the rapids probe must not run when rapids is disabled")
    });
    assert_eq!(decision, RapidsDecision::NoRapidsCpu);
}

#[test]
fn force_native_skips_rapids_even_when_available() {
    let decision = plan_rapids_decision(Some(0), false, true, || true);
    assert_eq!(decision, RapidsDecision::Native);
}

// --- device intent parsing (DeviceRequest::from_device_str) ---

#[test]
fn device_request_parses_intent_without_collapsing_auto() {
    assert_eq!(DeviceRequest::from_device_str("cpu"), DeviceRequest::Cpu);
    assert_eq!(DeviceRequest::from_device_str("auto"), DeviceRequest::Auto);
    assert_eq!(DeviceRequest::from_device_str("gpu"), DeviceRequest::Gpu);
    assert_eq!(DeviceRequest::from_device_str("gpu:1"), DeviceRequest::Gpu);
}

// --- exec-info builders ---

#[test]
fn cpu_only_exec_info_reasons() {
    // No GPU kernel exists at all: a forced-CPU request records the user's
    // choice; a GPU request on any host records a non-none reason.
    let info = cpu_only_exec_info(DeviceRequest::Cpu);
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
    let info = cpu_only_exec_info(DeviceRequest::Gpu);
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_ne!(info.fallback_reason, FallbackReason::None);
}

#[test]
fn cpu_exec_info_carries_chunk_size() {
    let info = cpu_exec_info(
        DeviceRequest::Cpu,
        InputLayout::BackedCsr,
        false,
        false,
        Some(512),
    );
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.chunk_size, Some(512));
    assert_eq!(info.csc_available, Some(false));
}

// --- serialization (AccelExecutionInfo::fields) ---

/// The key list and order are the `uns["scx_accel"][<op>]` wire contract:
/// pyscx's dict serializer and rscx's list builder are both loops over this,
/// so a drifted or renamed key here changes what every gate and user reads.
#[test]
fn fields_pins_the_wire_keys_in_order() {
    let info = AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::UserForcedCpu);
    let keys: Vec<&'static str> = info.fields().iter().map(|(k, _)| *k).collect();
    assert_eq!(
        keys,
        [
            "route",
            "fallback_reason",
            "chunk_size",
            "graph_replay",
            "csc_available",
            "shards_decoded",
            "shards_uploaded",
            "math_mode",
            "spmm_policy",
            "rapids_version",
            "cuml_version",
            "cupy_version",
            "transfer_mode",
            "device_id",
            "bytes_uploaded",
            "n_shards_shufdelta_gpu",
            "resident_csr",
        ]
    );
}

#[test]
fn fields_carries_the_values_not_defaults() {
    let mut info = AccelExecutionInfo::new(AccelRoute::RapidsSinglecell, FallbackReason::None);
    info.device_id = Some(2);
    info.rapids_version = Some("0.12.1".to_string());
    info.bytes_uploaded = Some(42);
    info.resident_csr = Some(true);
    let fields = info.fields();
    let get = |name: &str| {
        fields
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.clone())
            .unwrap()
    };
    assert!(matches!(
        get("route"),
        RouteValue::Str("rapids_singlecell_gpu")
    ));
    assert!(matches!(get("fallback_reason"), RouteValue::Str("none")));
    assert!(matches!(get("device_id"), RouteValue::OptUsize(Some(2))));
    assert!(matches!(
        get("rapids_version"),
        RouteValue::OptStr(Some("0.12.1"))
    ));
    assert!(matches!(
        get("bytes_uploaded"),
        RouteValue::OptU64(Some(42))
    ));
    assert!(matches!(
        get("resident_csr"),
        RouteValue::OptBool(Some(true))
    ));
    // Unset optionals serialise as explicit None entries, not omissions.
    assert!(matches!(get("cuml_version"), RouteValue::OptStr(None)));
    assert!(matches!(get("chunk_size"), RouteValue::OptUsize(None)));
}

// --- GPU-fallback warning predicate (moved from pyscx) ---

#[test]
fn warns_explicit_gpu_request_landing_on_cpu() {
    assert!(should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::NoCuda)
    ));
    assert!(should_warn_gpu_fallback(
        "gpu:1",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::UnsupportedInputLayout)
    ));
    assert!(should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsc, FallbackReason::UnsupportedDimensions)
    ));
    assert!(should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::PerfPolicy)
    ));
}

#[test]
fn no_warn_for_auto_or_cpu_requests() {
    // `auto`→CPU on a CPU host is expected, not a misconfiguration.
    assert!(!should_warn_gpu_fallback(
        "auto",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::NoCuda)
    ));
    assert!(!should_warn_gpu_fallback(
        "cpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::UserForcedCpu)
    ));
}

#[test]
fn no_warn_for_gpu_route_or_benign_reasons() {
    // Took the GPU as asked.
    assert!(!should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::GpuCsr, FallbackReason::None)
    ));
    // CPU route but not a GPU-unusable reason.
    assert!(!should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::UserForcedCpu)
    ));
    // NoRapids has its own dedicated one-shot warning (pyscx accel::rapids).
    assert!(!should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::NoRapids)
    ));
    // NoCscSidecar still runs on the GPU (CSR-direct), so no warning.
    assert!(!should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::GpuCsrV3, FallbackReason::NoCscSidecar)
    ));
}

/// The negative match in `should_warn_gpu_fallback` is what makes a new
/// `FallbackReason` warn by default rather than pass silently. Pin it on
/// the variant that exercised the property, so a future edit that turns
/// the guard into a positive allow-list fails here.
#[test]
fn a_runtime_failure_landing_on_cpu_warns() {
    assert!(should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::CpuCsr, FallbackReason::GpuRuntimeError)
    ));
    assert!(should_warn_gpu_fallback(
        "gpu:1",
        &AccelExecutionInfo::new(AccelRoute::CpuDense, FallbackReason::GpuRuntimeError)
    ));
    // Still on a GPU route (to_gpu_anndata's host-assemble fallback ends up
    // on the device either way), so this warning — which is specifically
    // "you asked for GPU and got CPU" — must stay silent. That path emits
    // its own warning naming the device failure.
    assert!(!should_warn_gpu_fallback(
        "gpu",
        &AccelExecutionInfo::new(AccelRoute::GpuCsr, FallbackReason::GpuRuntimeError)
    ));
}

// --- Restored from the pre-extraction inline module: the HVG and
// simple-planner matrices are live production planners and keep their
// full route/reason coverage (review round 1, Cursor Agent + codex).

// --- HVG planner (plan_hvg_route) ---

#[test]
fn hvg_gpu_seurat_v3_is_gpu_csr() {
    // seurat_v3 on a GPU host → the atomic-CSR GPU kernel.
    let info = plan_hvg_route(DeviceRequest::Gpu, true, true, false);
    assert_eq!(info.route, AccelRoute::GpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::None);
    assert!(info.route.is_gpu());
}

#[test]
fn hvg_cpu_forced_records_user_forced() {
    let info = plan_hvg_route(DeviceRequest::Cpu, true, true, false);
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
}

#[test]
fn hvg_no_cuda_records_no_cuda() {
    let info = plan_hvg_route(DeviceRequest::Auto, false, true, false);
    assert!(!info.route.is_gpu());
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::NoCuda);
}

#[test]
fn hvg_seurat_flavor_ineligible_on_gpu() {
    // A non-seurat_v3 flavor on a GPU host: GPU present but no kernel.
    let info = plan_hvg_route(DeviceRequest::Gpu, true, false, false);
    assert!(!info.route.is_gpu());
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::UnsupportedInputLayout);
}

#[test]
fn hvg_csc_ineligible_flavor_is_cpu_csc() {
    // A CSC sidecar is present but the flavor has no GPU kernel
    // (gpu_eligible=false): run CPU CSC, record the layout reason.
    let info = plan_hvg_route(DeviceRequest::Auto, true, false, true);
    assert_eq!(info.route, AccelRoute::CpuCsc);
    assert_eq!(info.fallback_reason, FallbackReason::UnsupportedInputLayout);
    assert_eq!(info.csc_available, Some(true));
}

#[test]
fn hvg_gpu_csc_sidecar_is_gpu_csc_v3() {
    // seurat_v3 on a GPU host with a reachable CSC sidecar → the
    // column-major CSC reduce route (mirrors DE's gpu_csc_v3).
    let info = plan_hvg_route(DeviceRequest::Gpu, true, true, true);
    assert_eq!(info.route, AccelRoute::GpuCscV3);
    assert_eq!(info.fallback_reason, FallbackReason::None);
    assert!(info.route.is_gpu());
    assert_eq!(info.csc_available, Some(true));
}

#[test]
fn hvg_cpu_forced_with_sidecar_is_cpu_csc() {
    let info = plan_hvg_route(DeviceRequest::Cpu, true, true, true);
    assert_eq!(info.route, AccelRoute::CpuCsc);
    assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
}

// --- Generic single-route planner (plan_simple_gpu_route) ---

#[test]
fn simple_gpu_auto_runs_gpu() {
    let info = plan_simple_gpu_route(
        DeviceRequest::Auto,
        true,
        true,
        AccelRoute::GpuCsr,
        AccelRoute::CpuCsr,
    );
    assert_eq!(info.route, AccelRoute::GpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::None);
}

#[test]
fn simple_gpu_lib_missing_is_unsupported_layout() {
    // CUDA present but the op's GPU library (e.g. cuVS) is unavailable.
    let info = plan_simple_gpu_route(
        DeviceRequest::Gpu,
        true,
        false,
        AccelRoute::GpuDense,
        AccelRoute::CpuDense,
    );
    assert_eq!(info.route, AccelRoute::CpuDense);
    assert_eq!(info.fallback_reason, FallbackReason::UnsupportedInputLayout);
}

#[test]
fn simple_no_cuda_records_no_cuda() {
    let info = plan_simple_gpu_route(
        DeviceRequest::Auto,
        false,
        true,
        AccelRoute::GpuCsr,
        AccelRoute::CpuCsr,
    );
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::NoCuda);
}

#[test]
fn simple_forced_cpu() {
    let info = plan_simple_gpu_route(
        DeviceRequest::Cpu,
        true,
        true,
        AccelRoute::GpuCsr,
        AccelRoute::CpuCsr,
    );
    assert_eq!(info.route, AccelRoute::CpuCsr);
    assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
}
