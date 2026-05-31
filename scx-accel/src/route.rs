//! Accelerator execution planner: route + fallback metadata.
//!
//! The accelerator dispatch for differential expression (and, in future, other
//! ops) selects between many concrete code paths driven by a mix of API
//! arguments (`device`, `prefer_format`), environment variables
//! (`SCX_GPU_DE_V2`, `SCX_GPU_DE_V3`), input layout (dense / in-memory CSR /
//! backed CSR / backed CSC / lazy), and runtime availability (CUDA present,
//! CSC sidecar present). Historically the *only* signal of which route ran was
//! an ad-hoc `SCX_GPU_DE_V3_TRACE` stderr line — which made it easy to
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
/// This is the **single source of truth** for the route. The GPU `pdex_ref`
/// dispatch sites `match` on the returned [`AccelRoute`] to select the kernel,
/// and the pyscx CPU dispatch sites stamp the returned info directly — so the
/// recorded route always matches the code that ran, and the whole decision
/// matrix is testable without a GPU. (Wilcoxon GPU has a single fixed v1
/// kernel and stamps its route directly, *not* through this planner.)
///
/// `gpu_eligible` distinguishes "the GPU has no kernel for this op+layout"
/// (e.g. `prefer_format="csc"`, which has no GPU CSC kernel) from CUDA simply
/// being absent: a GPU/auto request with `gpu_available && !gpu_eligible`
/// records a CPU route with [`FallbackReason::UnsupportedInputLayout`].
///
/// Precedence on GPU: v3 > v2 > v1 (mirrors `diffexp_gpu.rs`). CSC-direct
/// (`GpuCscV3`) is only taken when the layout actually carries a CSC sidecar
/// (`BackedCsc`); every other layout under v3 falls back to `GpuCsrV3` with
/// [`FallbackReason::NoCscSidecar`]. Dense never reaches a CSC or v2/v3 route.
pub fn plan_de_route(
    device: DeviceRequest,
    layout: InputLayout,
    gpu_available: bool,
    gpu_eligible: bool,
    v2_enabled: bool,
    v3_enabled: bool,
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
            // Dense always uses the legacy single-upload v1 path — no v2/v3,
            // never CSC.
            InputLayout::DenseHost => {
                AccelExecutionInfo::new(AccelRoute::GpuDenseV1, FallbackReason::None)
            }
            _ => {
                if v3_enabled {
                    if layout == InputLayout::BackedCsc && csc_available {
                        AccelExecutionInfo::new(AccelRoute::GpuCscV3, FallbackReason::None)
                    } else {
                        // v3 enabled but no CSC sidecar reachable for this
                        // layout: CSR-direct fallback.
                        AccelExecutionInfo::new(AccelRoute::GpuCsrV3, FallbackReason::NoCscSidecar)
                    }
                } else if v2_enabled {
                    AccelExecutionInfo::new(AccelRoute::GpuCsrV2, FallbackReason::None)
                } else {
                    AccelExecutionInfo::new(AccelRoute::GpuCsrV1, FallbackReason::None)
                }
            }
        }
    };

    info.csc_available = Some(csc_available);
    info
}

/// Convenience wrapper over [`plan_de_route`] that reads CSC capability from a
/// [`GpuMatrixSource`](scx_gpu::GpuMatrixSource) instead of a separate
/// `csc_available` flag.
///
/// `layout` stays an explicit parameter: the source's
/// [`SourceRouteMetadata`](scx_gpu::SourceRouteMetadata) deliberately omits
/// [`InputLayout`] (it lives in this crate, which depends on `scx-gpu` — the
/// reverse would be circular), so only the construction site knows the layout.
/// `gpu_available` is resolved via [`crate::gpu_available`].
#[cfg(feature = "gpu")]
pub fn plan_de_route_from_source(
    device: DeviceRequest,
    source: &dyn scx_gpu::GpuMatrixSource,
    layout: InputLayout,
    v2_enabled: bool,
    v3_enabled: bool,
) -> AccelExecutionInfo {
    let csc_available = source.available_layouts().contains(scx_gpu::LayoutSet::CSC);
    plan_de_route(
        device,
        layout,
        crate::gpu_available(),
        true,
        v2_enabled,
        v3_enabled,
        csc_available,
    )
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
        let info = plan_de_route_from_source(
            DeviceRequest::Gpu,
            &csr_only,
            InputLayout::BackedCsr,
            false,
            true,
        );
        assert_eq!(info.csc_available, Some(false));

        let with_csc = FakeSource(LayoutSet::CSR | LayoutSet::CSC);
        let info = plan_de_route_from_source(
            DeviceRequest::Gpu,
            &with_csc,
            InputLayout::BackedCsc,
            false,
            true,
        );
        assert_eq!(info.csc_available, Some(true));
    }

    #[test]
    fn dense_never_routes_to_csc() {
        // Dense host on GPU → v1 dense, never CSC, even with v3 + csc flags on.
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::DenseHost,
            true,
            true,
            true,
            true,
            true,
        );
        assert_eq!(info.route, AccelRoute::GpuDenseV1);
        assert_ne!(info.route, AccelRoute::GpuCscV3);

        // Dense host on CPU → cpu_dense.
        let info = plan_de_route(
            DeviceRequest::Cpu,
            InputLayout::DenseHost,
            true,
            true,
            false,
            false,
            false,
        );
        assert_eq!(info.route, AccelRoute::CpuDense);
    }

    #[test]
    fn in_memory_csr_v3_is_csr_direct_not_csc() {
        // In-memory CSR has no CSC sidecar: v3 must fall back to CSR-direct.
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::CsrHost,
            true,
            true,
            false,
            true,
            false,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV3);
        assert_eq!(info.fallback_reason, FallbackReason::NoCscSidecar);
    }

    #[test]
    fn backed_csc_with_sidecar_v3_is_csc_direct() {
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::BackedCsc,
            true,
            true,
            false,
            true,
            true,
        );
        assert_eq!(info.route, AccelRoute::GpuCscV3);
        assert_eq!(info.fallback_reason, FallbackReason::None);
        assert_eq!(info.csc_available, Some(true));
    }

    #[test]
    fn lazy_csr_v3_falls_back_to_csr_direct() {
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::LazyCsr,
            true,
            true,
            false,
            true,
            false,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV3);
        assert_eq!(info.fallback_reason, FallbackReason::NoCscSidecar);
    }

    #[test]
    fn backed_csr_v3_without_csc_falls_back() {
        // Even a backed CSR reader under v3: no CSC sidecar → CSR-direct.
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::BackedCsr,
            true,
            true,
            false,
            true,
            false,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV3);
        assert_eq!(info.fallback_reason, FallbackReason::NoCscSidecar);
    }

    #[test]
    fn forced_cpu_records_user_forced() {
        let info = plan_de_route(
            DeviceRequest::Cpu,
            InputLayout::BackedCsr,
            true,
            true,
            false,
            true,
            true,
        );
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
                false,
                true,
                true,
            );
            assert!(!info.route.is_gpu());
            assert_eq!(info.route, AccelRoute::CpuCsc);
            assert_eq!(info.fallback_reason, FallbackReason::UnsupportedInputLayout);
            assert_eq!(info.csc_available, Some(true));
        }
    }

    #[test]
    fn version_precedence_v3_beats_v2_beats_v1() {
        // v3 + v2 both on → v3 wins.
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::BackedCsr,
            true,
            true,
            true,
            true,
            false,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV3);

        // v2 only → v2.
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::BackedCsr,
            true,
            true,
            true,
            false,
            false,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV2);

        // neither → v1.
        let info = plan_de_route(
            DeviceRequest::Gpu,
            InputLayout::BackedCsr,
            true,
            true,
            false,
            false,
            false,
        );
        assert_eq!(info.route, AccelRoute::GpuCsrV1);
    }

    #[test]
    fn route_strings_are_stable() {
        assert_eq!(AccelRoute::GpuCscV3.as_str(), "gpu_csc_v3");
        assert_eq!(AccelRoute::CpuCsr.as_str(), "cpu_csr");
        assert_eq!(FallbackReason::NoCscSidecar.as_str(), "no_csc_sidecar");
        assert_eq!(FallbackReason::None.as_str(), "none");
    }
}
