//! Accelerator execution planner: route + fallback metadata.
//!
//! The accelerator dispatch for differential expression (and, in future, other
//! ops) selects between many concrete code paths driven by a mix of API
//! arguments (`device`, `prefer_format`), input layout (dense / in-memory CSR /
//! backed CSR / backed CSC / lazy), and runtime availability (CUDA present,
//! CSC sidecar present). GPU DE v3 is the unconditional default; the former
//! `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3` opt-in gates were removed when v3 was
//! promoted to default. Historically the *only* signal of which route
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
    /// GPU, dense input. The route id for GPU UMAP (its native CUDA / cuML SGD
    /// runs on a dense embedding). Not a DE route — dense-host DE densifies to
    /// CSR and reports [`GpuCsrV3`](AccelRoute::GpuCsrV3).
    GpuDense,
    /// GPU, CSR input. The generic single-kernel GPU CSR route id shared by the
    /// non-DE ops: PCA, kNN, Leiden, HVG, and preprocessing. Not a DE route —
    /// sparse GPU DE is always v3 ([`GpuCsrV3`](AccelRoute::GpuCsrV3) /
    /// [`GpuCscV3`](AccelRoute::GpuCscV3)).
    GpuCsr,
    /// GPU, CSR input, v3 sparse-direct DE path (no CSC sidecar).
    GpuCsrV3,
    /// GPU, CSC input, v3 column-direct DE path (CSC sidecar present).
    /// The perf-winning route for column algorithms.
    GpuCscV3,
    /// GPU, fully device-resident graph pipeline. Reserved for the future
    /// preprocess → PCA → kNN → UMAP handoff; no DE route emits this today.
    GpuDeviceResident,
    /// GPU compute handed off to rapids-singlecell (cuML/cuVS/cuGraph) on a
    /// device-resident matrix (ACC-RUST-OPT-V4 Phase 1). The in-VRAM route for
    /// the ops rapids supersedes (UMAP / in-VRAM PCA / in-VRAM kNN / extra HVG
    /// flavors / preprocess); see `plan_rapids_route`.
    RapidsSinglecell,
}

impl AccelRoute {
    /// Stable snake-case identifier for Python / benchmark JSON.
    pub fn as_str(self) -> &'static str {
        match self {
            AccelRoute::Unknown => "unknown",
            AccelRoute::CpuDense => "cpu_dense",
            AccelRoute::CpuCsr => "cpu_csr",
            AccelRoute::CpuCsc => "cpu_csc",
            AccelRoute::GpuDense => "gpu_dense",
            AccelRoute::GpuCsr => "gpu_csr",
            AccelRoute::GpuCsrV3 => "gpu_csr_v3",
            AccelRoute::GpuCscV3 => "gpu_csc_v3",
            AccelRoute::GpuDeviceResident => "gpu_device_resident",
            AccelRoute::RapidsSinglecell => "rapids_singlecell_gpu",
        }
    }

    /// Whether this route runs on the GPU.
    pub fn is_gpu(self) -> bool {
        matches!(
            self,
            AccelRoute::GpuDense
                | AccelRoute::GpuCsr
                | AccelRoute::GpuCsrV3
                | AccelRoute::GpuCscV3
                | AccelRoute::GpuDeviceResident
                | AccelRoute::RapidsSinglecell
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
    /// A rapids-routed op (ACC-RUST-OPT-V4) was requested on GPU but
    /// rapids-singlecell is not importable, so the op fell back to CPU. See
    /// `plan_rapids_route` and `docs/gpu-setup.md` (rapids analysis backend).
    NoRapids,
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
            FallbackReason::NoRapids => "no_rapids",
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
    /// GPU floating-point math mode for the op, if applicable (Task 2.5):
    /// `"strict_fp32"` / `"allow_tf32"`. `None` for CPU / ops without the knob.
    pub math_mode: Option<&'static str>,
    /// cuSPARSE SpMM algorithm policy for the op, if applicable (Task 2.5):
    /// `"default"` / `"deterministic"` / `"benchmark_once"`. `None` otherwise.
    pub spmm_policy: Option<&'static str>,
    // --- ACC-RUST-OPT-V4 §4.4: rapids-route / device-handoff metadata ---
    /// rapids-singlecell version, when a rapids route ran (`None` otherwise).
    pub rapids_version: Option<String>,
    /// cuML version, when known.
    pub cuml_version: Option<String>,
    /// cuPy version, when known (also set by the `to_gpu_anndata` device handoff).
    pub cupy_version: Option<String>,
    /// How the GPU-resident matrix was produced: `"anndata_to_gpu"` (rapids host
    /// upload) vs `"scx_device_handoff"` (SCX decoded onto the device). Lets a
    /// gate distinguish a real device handoff from a host re-upload.
    pub transfer_mode: Option<&'static str>,
    /// CUDA device ordinal the op ran on, if applicable.
    pub device_id: Option<usize>,
    /// Bytes uploaded host→device (HtoD) for the op / handoff, if tracked.
    pub bytes_uploaded: Option<u64>,
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
/// GPU route selection: v3 is the unconditional default. CSC-direct
/// (`GpuCscV3`) is taken when the layout carries a CSC sidecar (`BackedCsc` +
/// `csc_available`); every other layout — including dense-host, which the entry
/// point densifies to CSR — uses `GpuCsrV3` with
/// [`FallbackReason::NoCscSidecar`]. The former `SCX_GPU_DE_V2`/`SCX_GPU_DE_V3`
/// opt-in gates were removed once v3 was promoted to default; the v1/v2
/// dense-materialization drivers were deleted alongside them.
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
            // CSC-direct when a sidecar is reachable; every other layout —
            // including dense-host, which the entry point densifies to CSR —
            // uses CSR-direct. v3 is unconditional.
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
/// HVG has two GPU kernels, mirroring the DE CSC-direct convention:
///
/// * [`AccelRoute::GpuCscV3`] — the column-major CSC reduce (one block per
///   gene, no `atomicAdd` contention), taken when a CSC sidecar is reachable
///   (`csc_available`). Single-batch `seurat_v3` only.
/// * [`AccelRoute::GpuCsr`] — the `seurat_v3` atomic-CSR reduction, taken when
///   no sidecar is present.
///
/// The CPU path is [`AccelRoute::CpuCsr`], or [`AccelRoute::CpuCsc`] when a
/// gene-major sidecar is used (`csc_available`).
///
/// `gpu_eligible` is `false` for non-`seurat_v3` flavors (`seurat` /
/// `cell_ranger`); a GPU/auto request that is not eligible records
/// [`FallbackReason::UnsupportedInputLayout`] (CUDA is present, but there is no
/// GPU kernel for this flavor) rather than [`FallbackReason::NoCuda`]. Mirrors
/// the resolution shape of [`plan_de_route`] so the decision matrix is
/// unit-testable without a GPU. The caller supplies
/// `csc_available = sidecar_reachable && single_batch` (the flavor gate already
/// lives in `gpu_eligible`), keeping this planner pure.
pub fn plan_hvg_route(
    device: DeviceRequest,
    gpu_available: bool,
    gpu_eligible: bool,
    csc_available: bool,
) -> AccelExecutionInfo {
    let cpu_route = if csc_available {
        AccelRoute::CpuCsc
    } else {
        AccelRoute::CpuCsr
    };
    let (route, reason) = match device {
        DeviceRequest::Cpu => (cpu_route, FallbackReason::UserForcedCpu),
        DeviceRequest::Gpu | DeviceRequest::Auto => {
            if !gpu_available {
                (cpu_route, FallbackReason::NoCuda)
            } else if !gpu_eligible {
                (cpu_route, FallbackReason::UnsupportedInputLayout)
            } else if csc_available {
                (AccelRoute::GpuCscV3, FallbackReason::None)
            } else {
                (AccelRoute::GpuCsr, FallbackReason::None)
            }
        }
    };
    let mut info = AccelExecutionInfo::new(route, reason);
    info.csc_available = Some(csc_available);
    info
}

/// Decide the route for a single-route accelerator op (PCA / kNN / UMAP /
/// Leiden / preprocessing): GPU when available and eligible, otherwise CPU.
///
/// Unlike DE/HVG there is no version cascade — the op has exactly one GPU route
/// (`gpu_route`) and one CPU route (`cpu_route`), supplied by the caller because
/// they differ by op (CSR-shaped input → `GpuCsr`/`CpuCsr`; dense embedding →
/// `GpuDense`/`CpuDense`). `gpu_eligible` distinguishes a missing GPU library
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

/// Decide the route for an op that hands in-VRAM GPU compute to
/// rapids-singlecell (ACC-RUST-OPT-V4 Phase 1): UMAP, in-VRAM PCA/kNN, extra HVG
/// flavors, preprocess. The decision is a pure function of the dispatch facts so
/// it is unit-testable without a GPU; the pyscx caller supplies the runtime
/// signals (`gpu_available`, `rapids_available` from an import probe, `fits_vram`
/// from the VRAM pre-flight).
///
/// * `Cpu` → `cpu_route`, [`FallbackReason::UserForcedCpu`].
/// * no CUDA → `cpu_route`, [`FallbackReason::NoCuda`].
/// * **>VRAM** (`!fits_vram`) → `native_gpu_route`, [`FallbackReason::None`]: the
///   streaming moat. rapids has no automatic out-of-core fallback (it OOMs), so
///   the >VRAM regime stays on SCX's native streaming path regardless of whether
///   rapids is installed.
/// * ≤VRAM **and** rapids importable → [`AccelRoute::RapidsSinglecell`],
///   [`FallbackReason::None`].
/// * ≤VRAM **and** rapids absent → `cpu_route`, [`FallbackReason::NoRapids`]
///   (the detected-dependency contract, §4.3). The native in-VRAM GPU path is
///   reachable only via the transitional `SCX_FORCE_NATIVE_GPU` override (wired
///   in 1.3/1.4), so the default rapids-absent behaviour is a CPU fallback.
pub fn plan_rapids_route(
    device: DeviceRequest,
    gpu_available: bool,
    rapids_available: bool,
    fits_vram: bool,
    native_gpu_route: AccelRoute,
    cpu_route: AccelRoute,
) -> AccelExecutionInfo {
    let (route, reason) = match device {
        DeviceRequest::Cpu => (cpu_route, FallbackReason::UserForcedCpu),
        DeviceRequest::Gpu | DeviceRequest::Auto => {
            if !gpu_available {
                (cpu_route, FallbackReason::NoCuda)
            } else if !fits_vram {
                // >VRAM: the streaming moat — rapids OOMs in-VRAM with no
                // automatic fallback, so stay native-streaming.
                (native_gpu_route, FallbackReason::None)
            } else if rapids_available {
                (AccelRoute::RapidsSinglecell, FallbackReason::None)
            } else {
                (cpu_route, FallbackReason::NoRapids)
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
        // ACC-RUST-OPT-V4 wire contracts: the rapids route + no_rapids reason
        // are matched by the Phase 2 `*_route_rapids_correct` gates.
        assert_eq!(
            AccelRoute::RapidsSinglecell.as_str(),
            "rapids_singlecell_gpu"
        );
        assert!(AccelRoute::RapidsSinglecell.is_gpu());
        assert_eq!(FallbackReason::NoRapids.as_str(), "no_rapids");
    }

    // --- rapids router (plan_rapids_route) ---

    #[test]
    fn rapids_in_vram_with_rapids_routes_to_rapids() {
        for device in [DeviceRequest::Gpu, DeviceRequest::Auto] {
            let info = plan_rapids_route(
                device,
                true, // gpu_available
                true, // rapids_available
                true, // fits_vram
                AccelRoute::GpuCsr,
                AccelRoute::CpuCsr,
            );
            assert_eq!(info.route, AccelRoute::RapidsSinglecell);
            assert_eq!(info.fallback_reason, FallbackReason::None);
        }
    }

    #[test]
    fn rapids_in_vram_without_rapids_falls_back_to_cpu() {
        // ≤VRAM but rapids absent → CPU fallback with no_rapids (the §4.3
        // detected-dependency contract). The native in-VRAM path is reachable
        // only via the transitional override, wired later.
        let info = plan_rapids_route(
            DeviceRequest::Gpu,
            true,
            false, // rapids_available
            true,  // fits_vram
            AccelRoute::GpuCsr,
            AccelRoute::CpuCsr,
        );
        assert_eq!(info.route, AccelRoute::CpuCsr);
        assert_eq!(info.fallback_reason, FallbackReason::NoRapids);
    }

    #[test]
    fn rapids_over_vram_stays_native_streaming() {
        // >VRAM: rapids OOMs with no auto-fallback, so the streaming moat keeps
        // the native GPU route regardless of whether rapids is installed.
        for rapids in [true, false] {
            let info = plan_rapids_route(
                DeviceRequest::Auto,
                true,
                rapids,
                false, // !fits_vram → >VRAM
                AccelRoute::GpuCsr,
                AccelRoute::CpuCsr,
            );
            assert_eq!(info.route, AccelRoute::GpuCsr);
            assert_eq!(info.fallback_reason, FallbackReason::None);
        }
    }

    #[test]
    fn rapids_no_cuda_records_no_cuda() {
        let info = plan_rapids_route(
            DeviceRequest::Auto,
            false, // gpu_available
            true,
            true,
            AccelRoute::GpuCsr,
            AccelRoute::CpuCsr,
        );
        assert_eq!(info.route, AccelRoute::CpuCsr);
        assert_eq!(info.fallback_reason, FallbackReason::NoCuda);
    }

    #[test]
    fn rapids_forced_cpu_records_user_forced() {
        let info = plan_rapids_route(
            DeviceRequest::Cpu,
            true,
            true,
            true,
            AccelRoute::GpuCsr,
            AccelRoute::CpuCsr,
        );
        assert_eq!(info.route, AccelRoute::CpuCsr);
        assert_eq!(info.fallback_reason, FallbackReason::UserForcedCpu);
    }

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
}
