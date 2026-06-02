//! Accelerator execution planner: route + fallback metadata.
//!
//! The accelerator dispatch for differential expression (and, in future, other
//! ops) selects between many concrete code paths driven by a mix of API
//! arguments (`device`, `prefer_format`), input layout (dense / in-memory CSR /
//! backed CSR / backed CSC / lazy), and runtime availability (CUDA present,
//! CSC sidecar present). GPU DE v3 is the unconditional default; the former
//! `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` opt-in gates were removed in
//! ACC-RUST-OPT-V2 §5 Phase V1b. Historically the *only* signal of which route
//! ran was an ad-hoc `SCX_GPU_DE_V3_TRACE` stderr line — which made it easy to
//! benchmark one route while believing another ran.
//!
//! This module makes the route an explicit, recorded artifact:
//!
//! * [`plan_de_route`] is the **single source of truth** for the route label.
//!   It is a pure function of the dispatch inputs, so the whole route-decision
//!   matrix is unit-testable on CPU without a GPU.
//! * [`AccelExecutionInfo`] is stamped onto every DE result
//!   ([`crate::DiffExpResult`] / [`crate::PdexRefResult`]) and surfaced to
//!   Python on `adata.uns["scx_accel"][<op>]`.
//!
//! The enums intentionally cover *all* accelerator operations (PCA / kNN /
//! UMAP / Harmony / HVG) even though only DE is wired today, so those ops can
//! adopt the planner later without a redesign.

/// The concrete execution route an accelerator op took.
///
/// String forms (via [`AccelRoute::as_str`]) are stable wire identifiers used
/// in Python (`adata.uns["scx_accel"]`) and benchmark JSON — do not rename
/// without updating those consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccelRoute {
    /// Not yet stamped. Default for freshly constructed results before the
    /// dispatch entry point records the real route (e.g. internal chunk
    /// drivers, merge intermediates, test fixtures).
    #[default]
    Unknown,
    /// CPU, dense row-major input.
    CpuDense,
    /// CPU, CSR (row-major) input.
    CpuCsr,
    /// CPU, CSC (column-major / gene-major) input.
    CpuCsc,
    /// GPU, dense input, legacy v1 host-upload path.
    GpuDenseV1,
    /// GPU, CSR input, legacy v1 dense-materialization path.
    GpuCsrV1,
    /// GPU, CSR input, v2 direct-scatter path (`SCX_GPU_DE_V2`).
    GpuCsrV2,
    /// GPU, CSR input, v3 sparse-direct path (`SCX_GPU_DE_V3`, no CSC sidecar).
    GpuCsrV3,
    /// GPU, CSC input, v3 column-direct path (`SCX_GPU_DE_V3` + CSC sidecar).
    /// The perf-winning route for column algorithms.
    GpuCscV3,
    /// GPU, fully device-resident graph pipeline. Reserved for the future
    /// preprocess → PCA → kNN → UMAP handoff; no DE route emits this today.
    GpuDeviceResident,
}

impl AccelRoute {
    /// Stable snake-case identifier for Python / benchmark JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            AccelRoute::Unknown => "unknown",
            AccelRoute::CpuDense => "cpu_dense",
            AccelRoute::CpuCsr => "cpu_csr",
            AccelRoute::CpuCsc => "cpu_csc",
            AccelRoute::GpuDenseV1 => "gpu_dense_v1",
            AccelRoute::GpuCsrV1 => "gpu_csr_v1",
            AccelRoute::GpuCsrV2 => "gpu_csr_v2",
            AccelRoute::GpuCsrV3 => "gpu_csr_v3",
            AccelRoute::GpuCscV3 => "gpu_csc_v3",
            AccelRoute::GpuDeviceResident => "gpu_device_resident",
        }
    }

    /// Whether this route runs on the GPU.
    pub fn is_gpu(self) -> bool {
        matches!(
            self,
            AccelRoute::GpuDenseV1
                | AccelRoute::GpuCsrV1
                | AccelRoute::GpuCsrV2
                | AccelRoute::GpuCsrV3
                | AccelRoute::GpuCscV3
                | AccelRoute::GpuDeviceResident
        )
    }
}

/// Why the planner did not take the "ideal" route for the request.
///
/// `None` means the request was satisfied as asked (including a deliberate
/// CPU request — that records `UserForcedCpu`, not `None`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackReason {
    /// No fallback — the route matches the request.
    #[default]
    None,
    /// GPU requested or auto-selected but CUDA is unavailable.
    NoCuda,
    /// GPU column route requested but no CSC sidecar was available, so a
    /// CSR-direct route was used instead.
    NoCscSidecar,
    /// Input dimensions exceed a supported limit (e.g. `n_obs > i32::MAX` for
    /// the 32-bit GPU index path).
    UnsupportedDimensions,
    /// The input layout has no GPU kernel and was routed to CPU.
    UnsupportedInputLayout,
    /// The user explicitly forced `device="cpu"`.
    UserForcedCpu,
    /// A performance policy chose CPU/another route despite GPU availability.
    PerfPolicy,
}

impl FallbackReason {
    /// Stable snake-case identifier for Python / benchmark JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            FallbackReason::None => "none",
            FallbackReason::NoCuda => "no_cuda",
            FallbackReason::NoCscSidecar => "no_csc_sidecar",
            FallbackReason::UnsupportedDimensions => "unsupported_dimensions",
            FallbackReason::UnsupportedInputLayout => "unsupported_input_layout",
            FallbackReason::UserForcedCpu => "user_forced_cpu",
            FallbackReason::PerfPolicy => "perf_policy",
        }
    }
}

/// The recorded execution plan for one accelerator call.
///
/// Stamped onto DE result structs by the dispatch entry points and surfaced to
/// Python. `chunk_size` / `csc_available` are filled by the entry point;
/// `shards_decoded` / `shards_uploaded` / `graph_replay` are left `None` for
/// now (future enrichment from the chunk drivers).
#[derive(Debug, Clone, Default)]
pub struct AccelExecutionInfo {
    /// The concrete route taken.
    pub route: AccelRoute,
    /// Why the ideal route was not taken (or `None`).
    pub fallback_reason: FallbackReason,
    /// Gene-chunk size used by streaming/chunked routes, if applicable.
    pub chunk_size: Option<usize>,
    /// Whether a captured CUDA graph was replayed, if known.
    pub graph_replay: Option<bool>,
    /// Whether a CSC sidecar was available at dispatch time, if known.
    pub csc_available: Option<bool>,
    /// Number of shards decoded (streaming routes), if tracked.
    pub shards_decoded: Option<usize>,
    /// Number of shards uploaded to device (streaming routes), if tracked.
    pub shards_uploaded: Option<usize>,
}

impl AccelExecutionInfo {
    /// Construct from a route + fallback reason, leaving the optional counters
    /// unset.
    pub fn new(route: AccelRoute, fallback_reason: FallbackReason) -> Self {
        AccelExecutionInfo {
            route,
            fallback_reason,
            ..Default::default()
        }
    }
}

/// User device intent, decoupled from runtime availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    /// Force CPU.
    Cpu,
    /// Force GPU (error elsewhere if unavailable).
    Gpu,
    /// Prefer GPU when available, else CPU.
    Auto,
}

/// The layout the input matrix is presented in to the accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputLayout {
    /// Dense row-major host buffer.
    DenseHost,
    /// In-memory CSR (e.g. scipy CSR) — never has a CSC sidecar.
    CsrHost,
    /// SCX-backed CSR reader.
    BackedCsr,
    /// SCX-backed CSC reader (gene-major sidecar).
    BackedCsc,
    /// Generic lazy `ShardSource` (CSR-shaped; no CSC capability surface).
    LazyCsr,
}

/// Decide the DE execution route from the dispatch inputs.
///
/// This is the **single source of truth** for the route. Both the GPU
/// `pdex_ref` and Wilcoxon (`rank_genes_groups`) dispatch sites `match` on the
/// returned [`AccelRoute`] to select the kernel (via
/// [`plan_de_route_from_source`] in `diffexp_gpu.rs`), and the pyscx CPU
/// dispatch sites stamp the returned info directly — so the recorded route
/// always matches the code that ran, and the whole decision matrix is testable
/// without a GPU.
///
/// `gpu_eligible` distinguishes "the GPU has no kernel for this op+layout"
/// (e.g. `prefer_format="csc"`, which has no GPU CSC kernel) from CUDA simply
/// being absent: a GPU/auto request with `gpu_available && !gpu_eligible`
/// records a CPU route with [`FallbackReason::UnsupportedInputLayout`].
///
/// GPU route selection: v3 is the unconditional default for sparse inputs.
/// CSC-direct (`GpuCscV3`) is taken when the layout carries a CSC sidecar
/// (`BackedCsc` + `csc_available`); every other sparse layout uses `GpuCsrV3`
/// with [`FallbackReason::NoCscSidecar`]. Dense-host stays on the legacy
/// single-upload `GpuDenseV1` path (the dense → CSR rewrite is a separate
/// follow-up). The former `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` opt-in gates were
/// removed once v3 was promoted to default (ACC-RUST-OPT-V2 §5 Phase V1b).
pub fn plan_de_route(
    device: DeviceRequest,
    layout: InputLayout,
    gpu_available: bool,
    gpu_eligible: bool,
    csc_available: bool,
) -> AccelExecutionInfo {
    // Resolve whether we actually run on GPU, and why not when we don't.
    let (use_gpu, cpu_reason) = match device {
        DeviceRequest::Cpu => (false, FallbackReason::UserForcedCpu),
        DeviceRequest::Gpu | DeviceRequest::Auto => {
            if !gpu_available {
                (false, FallbackReason::NoCuda)
            } else if !gpu_eligible {
                // GPU is present but this op+layout has no GPU kernel (e.g.
                // prefer_format="csc"): run CPU, record the layout as the
                // reason rather than implying CUDA was missing.
                (false, FallbackReason::UnsupportedInputLayout)
            } else {
                (true, FallbackReason::None)
            }
        }
    };

    let mut info = if !use_gpu {
        let route = match layout {
            InputLayout::DenseHost => AccelRoute::CpuDense,
            InputLayout::BackedCsc => AccelRoute::CpuCsc,
            InputLayout::CsrHost | InputLayout::BackedCsr | InputLayout::LazyCsr => {
                AccelRoute::CpuCsr
            }
        };
        AccelExecutionInfo::new(route, cpu_reason)
    } else {
        match layout {
            // Dense stays on the legacy single-upload v1 path — never CSC.
            InputLayout::DenseHost => {
                AccelExecutionInfo::new(AccelRoute::GpuDenseV1, FallbackReason::None)
            }
            // Sparse inputs: CSC-direct when a sidecar is reachable, else
            // CSR-direct. v3 is unconditional.
            InputLayout::BackedCsc if csc_available => {
                AccelExecutionInfo::new(AccelRoute::GpuCscV3, FallbackReason::None)
            }
            _ => AccelExecutionInfo::new(AccelRoute::GpuCsrV3, FallbackReason::NoCscSidecar),
        }
    };

    info.csc_available = Some(csc_available);
    info
}

/// Decide the highly-variable-genes (HVG) execution route.
///
/// HVG has a single GPU kernel: the `seurat_v3` atomic-CSR reduction, recorded
/// as [`AccelRoute::GpuCsrV1`] (the wire id is honest — HVG GPU is a CSR atomic
/// kernel; the gate only tests [`AccelRoute::is_gpu`]). The CPU path is
/// [`AccelRoute::CpuCsr`], or [`AccelRoute::CpuCsc`] when the caller prefers the
/// gene-major sidecar (`prefer_csc`), which has no GPU kernel.
///
/// `gpu_eligible` is `false` for non-`seurat_v3` flavors (`seurat` /
/// `cell_ranger`) and for the CSC-preferred path; a GPU/auto request that is
/// not eligible records [`FallbackReason::UnsupportedInputLayout`] (CUDA is
/// present, but there is no GPU kernel for this flavor/layout) rather than
/// [`FallbackReason::NoCuda`]. Mirrors the resolution shape of [`plan_de_route`]
/// so the decision matrix is unit-testable without a GPU.
///
/// A future CSC-reduce GPU HVG kernel (section D) should reuse
/// [`AccelRoute::GpuCscV3`], matching the DE CSC-direct convention.
pub fn plan_hvg_route(
    device: DeviceRequest,
    gpu_available: bool,
    gpu_eligible: bool,
    prefer_csc: bool,
) -> AccelExecutionInfo {
    let cpu_route = if prefer_csc {
        AccelRoute::CpuCsc
    } else {
        AccelRoute::CpuCsr
    };
    plan_simple_gpu_route(
        device,
        gpu_available,
        gpu_eligible,
        AccelRoute::GpuCsrV1,
        cpu_route,
    )
}

/// Decide the route for a single-route accelerator op (PCA / kNN / UMAP /
/// Leiden / preprocessing): GPU when available and eligible, otherwise CPU.
///
/// Unlike DE/HVG there is no version cascade — the op has exactly one GPU route
/// (`gpu_route`) and one CPU route (`cpu_route`), supplied by the caller because
/// they differ by op (CSR-shaped input → `GpuCsrV1`/`CpuCsr`; dense embedding →
/// `GpuDenseV1`/`CpuDense`). `gpu_eligible` distinguishes a missing GPU library
/// (cuVS / cuML / cuGraph / cuSPARSE absent → CPU fallback with
/// [`FallbackReason::UnsupportedInputLayout`]) from CUDA simply being absent
/// ([`FallbackReason::NoCuda`]). A forced-CPU request records
/// [`FallbackReason::UserForcedCpu`].
pub fn plan_simple_gpu_route(
    device: DeviceRequest,
    gpu_available: bool,
    gpu_eligible: bool,
    gpu_route: AccelRoute,
    cpu_route: AccelRoute,
) -> AccelExecutionInfo {
    let (route, reason) = match device {
        DeviceRequest::Cpu => (cpu_route, FallbackReason::UserForcedCpu),
        DeviceRequest::Gpu | DeviceRequest::Auto => {
            if !gpu_available {
                (cpu_route, FallbackReason::NoCuda)
            } else if !gpu_eligible {
                (cpu_route, FallbackReason::UnsupportedInputLayout)
            } else {
                (gpu_route, FallbackReason::None)
            }
        }
    };
    AccelExecutionInfo::new(route, reason)
}

/// Convenience wrapper over [`plan_de_route`] that reads CSC capability from a
/// [`GpuMatrixSource`](scx_gpu::GpuMatrixSource) instead of a separate
/// `csc_available` flag.
///
/// `layout` stays an explicit parameter: the source's
/// [`SourceRouteMetadata`](scx_gpu::SourceRouteMetadata) deliberately omits
/// [`InputLayout`] (it lives in this crate, which depends on `scx-gpu` — the
/// reverse would be circular), so only the construction site knows the layout.
///
/// `gpu_available` is resolved via [`crate::gpu_available`] (a `GpuDevice::count`
/// probe). In normal DE dispatch this is reached only after `open_device`
/// already succeeded, so it is expected `true`; the probe is what keeps the
/// `other =>` "unreachable route" arm in the dispatch `match` from being dead
/// code — if the GPU vanished between `open_device` and here, the planner
/// stamps a CPU route and that arm reports it rather than silently mis-running.
#[cfg(feature = "gpu")]
pub fn plan_de_route_from_source(
    device: DeviceRequest,
    source: &dyn scx_gpu::GpuMatrixSource,
    layout: InputLayout,
) -> AccelExecutionInfo {
    let csc_available = source.available_layouts().contains(scx_gpu::LayoutSet::CSC);
    plan_de_route(device, layout, crate::gpu_available(), true, csc_available)
}

#[cfg(test)]
mod tests {
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
        // Dense host on GPU → v1 dense, never CSC, even with the csc flag on.
        let info = plan_de_route(DeviceRequest::Gpu, InputLayout::DenseHost, true, true, true);
        assert_eq!(info.route, AccelRoute::GpuDenseV1);
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
    fn route_strings_are_stable() {
        assert_eq!(AccelRoute::GpuCscV3.as_str(), "gpu_csc_v3");
        assert_eq!(AccelRoute::CpuCsr.as_str(), "cpu_csr");
        assert_eq!(FallbackReason::NoCscSidecar.as_str(), "no_csc_sidecar");
        assert_eq!(FallbackReason::None.as_str(), "none");
    }

    // --- HVG planner (plan_hvg_route) ---

    #[test]
    fn hvg_gpu_seurat_v3_is_gpu_csr_v1() {
        // seurat_v3 on a GPU host → the atomic-CSR GPU kernel.
        let info = plan_hvg_route(DeviceRequest::Gpu, true, true, false);
        assert_eq!(info.route, AccelRoute::GpuCsrV1);
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
    fn hvg_csc_is_cpu_csc() {
        // prefer_format="csc": no GPU CSC HVG kernel → CPU CSC route.
        let info = plan_hvg_route(DeviceRequest::Auto, true, false, true);
        assert_eq!(info.route, AccelRoute::CpuCsc);
        assert_eq!(info.fallback_reason, FallbackReason::UnsupportedInputLayout);
    }

    // --- Generic single-route planner (plan_simple_gpu_route) ---

    #[test]
    fn simple_gpu_auto_runs_gpu() {
        let info = plan_simple_gpu_route(
            DeviceRequest::Auto,
            true,
            true,
            AccelRoute::GpuCsrV1,
            AccelRoute::CpuCsr,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV1);
        assert_eq!(info.fallback_reason, FallbackReason::None);
    }

    #[test]
    fn simple_gpu_lib_missing_is_unsupported_layout() {
        // CUDA present but the op's GPU library (e.g. cuVS) is unavailable.
        let info = plan_simple_gpu_route(
            DeviceRequest::Gpu,
            true,
            false,
            AccelRoute::GpuDenseV1,
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
            AccelRoute::GpuCsrV1,
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
            AccelRoute::GpuCsrV1,
            AccelRoute::CpuCsr,
        );
        assert_eq!(info.route, AccelRoute::CpuCsr);
        assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
    }
}
