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
    /// CPU, native pseudobulk negative-binomial GLM (IRLS + Cox–Reid dispersion).
    /// A first-class native CPU route — never a rapids fallback (no `NoRapids`).
    CpuNbGlm,
    /// GPU, pseudobulk NB-GLM, host-fed dense pseudobulk (Stage A). The host
    /// aggregates the small `[n_genes × n_sub]` pseudobulk; the per-gene
    /// IRLS ↔ Cox–Reid fit runs on the device (`scx-gpu/kernels/nb_glm.cu`).
    GpuNbGlmCsr,
    /// GPU, pseudobulk NB-GLM, device-resident column aggregation (Stage C,
    /// reserved). Selected when a CSC sidecar feeds the on-device sum kernels.
    GpuNbGlmCsc,
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
    /// device-resident matrix. The in-VRAM route for the ops rapids supersedes
    /// (UMAP / in-VRAM PCA / in-VRAM kNN / extra HVG flavors / preprocess);
    /// see [`plan_rapids_decision`].
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
            AccelRoute::CpuNbGlm => "cpu_nb_glm",
            AccelRoute::GpuNbGlmCsr => "gpu_nb_glm_csr",
            AccelRoute::GpuNbGlmCsc => "gpu_nb_glm_csc",
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
                | AccelRoute::GpuNbGlmCsr
                | AccelRoute::GpuNbGlmCsc
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
    /// A rapids-routed op was requested on GPU but rapids-singlecell is not
    /// importable, so the op fell back to CPU. See [`plan_rapids_decision`]
    /// and `docs/gpu-setup.md` (rapids analysis backend).
    NoRapids,
    /// The op reached a *slower* path because the faster one failed on the
    /// device at run time — not because the request or the input ruled it out.
    ///
    /// Every other variant here is a pre-flight condition, decided before any
    /// device work. This one is decided by a failure that already happened, so
    /// it is only ever recorded on an op that **succeeded** by another route.
    /// A GPU op that fails outright raises instead, and its route stamp is
    /// rolled back (`pyscx`'s `RouteStamp`) rather than rewritten to claim a
    /// CPU run that never happened — `device="auto"` resolves the device up
    /// front and does not re-run on CPU after a runtime GPU failure.
    ///
    /// Currently produced by `Experiment.to_gpu_anndata`, which falls through
    /// to host-assemble when the in-VRAM shard decode fails. Without this the
    /// slow path is indistinguishable from the one chosen up front because the
    /// request needed filtering.
    GpuRuntimeError,
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
            FallbackReason::GpuRuntimeError => "gpu_runtime_error",
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
    ///
    /// `"deterministic"` names an algorithm (`CUSPARSE_SPMM_CSR_ALG2`), **not**
    /// a bit-reproducibility guarantee — cuSPARSE gives none for transpose
    /// operations, which the PCA power loop issues every iteration. See
    /// `scx_gpu::SpmmAlgPolicy`. `"benchmark_once"` is reserved and currently
    /// resolves to the same algorithm as `"default"`.
    pub spmm_policy: Option<&'static str>,
    // --- rapids-route / device-handoff metadata ---
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
    /// Number of shards that took the ShufDeltaZstd GPU decode path
    /// (CPU zstd + GPU undelta/unshuffle/convert),
    /// set by the `to_gpu_anndata` device handoff. Lets a benchmark/gate observe
    /// per-codec GPU routing on `compact-trial` (mixed-codec) files, which the
    /// aggregate `transfer_mode` cannot (it stays `scx_device_handoff_streamed`
    /// while any shard uploads decompressed bytes). `None` outside that path.
    pub n_shards_shufdelta_gpu: Option<u32>,
    /// Whether the route held the whole matrix device-resident instead of
    /// re-decoding it, on the two ops that make that choice.
    ///
    /// `Some(true)` = resident, `Some(false)` = the streaming path ran,
    /// `None` = the route has no residency decision to make (CPU, dense, or a
    /// CSC-direct DE route, which prefilters by column range and never
    /// re-decodes in the first place).
    ///
    /// - **GPU CSR DE routes** (§9.11): residency is across *gene chunks*.
    ///   Streaming means the matrix was over the VRAM budget, there was only
    ///   one gene chunk, or `SCX_GPU_DE_RESIDENT=0`.
    /// - **Native GPU PCA** (§8.11): residency is across the randomized *power
    ///   loop*. Streaming means over the VRAM budget or
    ///   `SCX_GPU_PCA_RESIDENT=0`, in which case the operator re-decodes and
    ///   re-uploads the whole matrix on every `matmat`/`rmatmat`.
    ///
    /// Both decisions are made dynamically against *free* VRAM at call time, so
    /// the same input can go either way run to run — which is why this is
    /// recorded rather than inferred from shape.
    ///
    /// A gate reads this to catch a *silent* fall back to streaming, the same
    /// way `de_route_csc_direct` catches a silent CSR fallback. Note the
    /// companion counter `shards_decoded` keeps its old meaning — slab passes,
    /// which residency does not change — so it is **not** the signal for this.
    pub resident_csr: Option<bool>,
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

    /// The record as `(key, value)` pairs — the single definition of the
    /// `uns["scx_accel"][<op>]` wire serialization.
    ///
    /// Every binding's sink is a loop over this (pyscx → a Python dict, rscx →
    /// a named R list), so the key set and order cannot drift between
    /// bindings. Unset optionals are emitted as explicit
    /// `None`/`NULL` entries, not omitted — a gate distinguishing "not
    /// tracked" from "key missing entirely" depends on that.
    pub fn fields(&self) -> [(&'static str, RouteValue<'_>); 17] {
        [
            ("route", RouteValue::Str(self.route.as_str())),
            (
                "fallback_reason",
                RouteValue::Str(self.fallback_reason.as_str()),
            ),
            ("chunk_size", RouteValue::OptUsize(self.chunk_size)),
            ("graph_replay", RouteValue::OptBool(self.graph_replay)),
            ("csc_available", RouteValue::OptBool(self.csc_available)),
            ("shards_decoded", RouteValue::OptUsize(self.shards_decoded)),
            (
                "shards_uploaded",
                RouteValue::OptUsize(self.shards_uploaded),
            ),
            ("math_mode", RouteValue::OptStr(self.math_mode)),
            ("spmm_policy", RouteValue::OptStr(self.spmm_policy)),
            (
                "rapids_version",
                RouteValue::OptStr(self.rapids_version.as_deref()),
            ),
            (
                "cuml_version",
                RouteValue::OptStr(self.cuml_version.as_deref()),
            ),
            (
                "cupy_version",
                RouteValue::OptStr(self.cupy_version.as_deref()),
            ),
            ("transfer_mode", RouteValue::OptStr(self.transfer_mode)),
            ("device_id", RouteValue::OptUsize(self.device_id)),
            ("bytes_uploaded", RouteValue::OptU64(self.bytes_uploaded)),
            (
                "n_shards_shufdelta_gpu",
                RouteValue::OptU32(self.n_shards_shufdelta_gpu),
            ),
            ("resident_csr", RouteValue::OptBool(self.resident_csr)),
        ]
    }
}

/// One serialized [`AccelExecutionInfo`] field value, typed so each binding's
/// sink can map it to its native scalar without re-knowing the field list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteValue<'a> {
    /// Always-present string (route / fallback_reason).
    Str(&'a str),
    /// Optional string (versions, modes).
    OptStr(Option<&'a str>),
    /// Optional boolean flag.
    OptBool(Option<bool>),
    /// Optional count/index.
    OptUsize(Option<usize>),
    /// Optional 32-bit counter.
    OptU32(Option<u32>),
    /// Optional byte count.
    OptU64(Option<u64>),
}

/// User device intent, decoupled from runtime availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceRequest {
    /// Force CPU.
    Cpu,
    /// Force GPU (error elsewhere if unavailable).
    Gpu,
    /// Prefer GPU when it is **available at dispatch**, else CPU.
    ///
    /// Availability is resolved once, before the op starts, from the
    /// pre-flight conditions [`FallbackReason`] enumerates. It is **not**
    /// re-evaluated afterwards: a GPU that is present but then fails at run
    /// time — out of memory, a driver fault — raises, rather than silently
    /// re-running the op on CPU. That is deliberate. A CPU re-run of an
    /// atlas-scale op is not a graceful degradation the caller can ignore; it
    /// is hours of work they did not ask for, discoverable only after the
    /// fact. The error names the shortfall and the remedy instead
    /// (`device="cpu"`).
    Auto,
}

impl DeviceRequest {
    /// Map a `device=` string to the user's [`DeviceRequest`] intent.
    ///
    /// Deliberately infallible and grammar-blind: full validation (including
    /// the `gpu:N` suffix and runtime availability) is
    /// [`resolve_device`](crate::device::resolve_device)'s job, and resolution
    /// collapses `"auto"` to CPU when no GPU is present — losing the intent
    /// the planners need to distinguish `NoCuda` from `UserForcedCpu`. This
    /// re-parses the raw string for the planner instead.
    pub fn from_device_str(device: &str) -> DeviceRequest {
        if device == "cpu" {
            DeviceRequest::Cpu
        } else if device == "auto" {
            DeviceRequest::Auto
        } else {
            // "gpu" / "gpu:N"
            DeviceRequest::Gpu
        }
    }
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
/// [`plan_de_route_from_source`] in `diffexp/gpu.rs`), and the pyscx CPU
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

/// Decide the pseudobulk NB-GLM execution route (Stage A).
///
/// Unlike [`plan_de_route`], NB-GLM has no per-layout GPU dispatch in Stage A:
/// the host always aggregates the small dense pseudobulk and feeds the GPU
/// fitter, so an eligible GPU request takes [`AccelRoute::GpuNbGlmCsr`].
/// `gpu_eligible` is `false` when the design width (`n_features`) or sample
/// count (`n_sub`) exceeds the kernel's register bounds
/// (`scx_gpu::GPU_NB_GLM_PMAX` / `GPU_NB_GLM_NSUB_MAX`) — those record a CPU
/// route with [`FallbackReason::UnsupportedDimensions`]. Never `NoRapids`:
/// NB-GLM is a native path, not a rapids-routed op. ([`AccelRoute::GpuNbGlmCsc`]
/// / CSC-direct device aggregation is reserved for Stage C.)
pub fn plan_nb_glm_route(
    device: DeviceRequest,
    gpu_available: bool,
    gpu_eligible: bool,
) -> AccelExecutionInfo {
    match device {
        DeviceRequest::Cpu => {
            AccelExecutionInfo::new(AccelRoute::CpuNbGlm, FallbackReason::UserForcedCpu)
        }
        DeviceRequest::Gpu | DeviceRequest::Auto => {
            if !gpu_available {
                AccelExecutionInfo::new(AccelRoute::CpuNbGlm, FallbackReason::NoCuda)
            } else if !gpu_eligible {
                AccelExecutionInfo::new(AccelRoute::CpuNbGlm, FallbackReason::UnsupportedDimensions)
            } else {
                AccelExecutionInfo::new(AccelRoute::GpuNbGlmCsr, FallbackReason::None)
            }
        }
    }
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

/// How a GPU-eligible standalone op should dispatch in the in-VRAM regime.
///
/// Produced by [`plan_rapids_decision`]; the binding's dispatch site matches
/// on it to select between the rapids-singlecell handoff, the existing
/// native/CPU dispatch, and the rapids-absent CPU fallback.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RapidsDecision {
    /// Hand off to rapids-singlecell on this device.
    Rapids(usize),
    /// Proceed with the existing native / CPU dispatch unchanged — `device="cpu"`,
    /// or `SCX_FORCE_NATIVE_GPU=1` selecting the native GPU kernels.
    Native,
    /// GPU requested + rapids absent + not forcing native → route CPU and stamp
    /// `no_rapids` (the binding emits its one-shot diagnostic on this arm).
    NoRapidsCpu,
}

/// `SCX_FORCE_NATIVE_GPU` override — keep the native SCX GPU kernels instead of
/// routing in-VRAM GPU compute to rapids-singlecell. After Phase 3 it pins only
/// the surviving native paths (streaming preprocess kernels, randomized PCA);
/// the in-VRAM UMAP / covariance-PCA / CAGRA-kNN kernels it also used to pin were
/// removed, so for those ops it now falls through to the CPU path.
pub fn force_native_gpu() -> bool {
    matches!(std::env::var("SCX_FORCE_NATIVE_GPU"), Ok(v) if v != "0" && !v.is_empty())
}

/// `SCX_DISABLE_RAPIDS` override — treat rapids-singlecell as if it were not
/// importable, so a GPU op takes the `no_rapids` CPU-fallback path even on a host
/// where rapids *is* installed. Exists so the Phase 2.2 fallback gate can exercise
/// the rapids-absent contract without uninstalling rapids; honored ahead of the
/// `force_native` override so the fallback (not the native kernels) is what runs.
pub fn rapids_disabled() -> bool {
    matches!(std::env::var("SCX_DISABLE_RAPIDS"), Ok(v) if v != "0" && !v.is_empty())
}

/// Decide how a GPU-eligible in-VRAM op dispatches between rapids-singlecell,
/// the native path, and the rapids-absent CPU fallback.
///
/// Pure given its inputs, so the whole decision matrix — including the
/// override precedence — is unit-testable without a GPU or a rapids install.
/// The binding supplies the runtime facts: `gpu_id` from device resolution
/// (`None` = CPU), the two env overrides (usually [`rapids_disabled`] /
/// [`force_native_gpu`]), and `rapids_available` as a **lazy** probe — it is
/// consulted only when the decision actually depends on it, so
/// `SCX_DISABLE_RAPIDS=1` (or `device="cpu"`) never triggers the rapids
/// import and the CUDA init it would cause.
///
/// Precedence: `rapids_disabled` is honored **ahead of** `force_native`, so
/// the Phase-2.2 fallback gate pins the `no_rapids` CPU path rather than the
/// native kernels.
///
/// The out-of-VRAM streaming moat never reaches this function: a backed/lazy
/// `X` stays on the native streaming dispatch, enforced by the caller's
/// layout check before this decision is consulted.
pub fn plan_rapids_decision(
    gpu_id: Option<usize>,
    rapids_disabled: bool,
    force_native: bool,
    rapids_available: impl FnOnce() -> bool,
) -> RapidsDecision {
    let Some(gpu_id) = gpu_id else {
        return RapidsDecision::Native; // device="cpu" → existing CPU path
    };
    if rapids_disabled {
        return RapidsDecision::NoRapidsCpu;
    }
    if force_native {
        return RapidsDecision::Native;
    }
    if rapids_available() {
        RapidsDecision::Rapids(gpu_id)
    } else {
        RapidsDecision::NoRapidsCpu
    }
}

/// Whether a CUDA GPU is available at runtime (always `false` when this crate
/// is built without the `gpu` feature). The availability input the exec-info
/// builders below feed to the planners, shared so each binding does not carry
/// its own cfg shim.
pub fn gpu_runtime_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        crate::gpu_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}

/// Build the execution info for a CPU DE dispatch via the single planner.
///
/// `gpu_eligible` is `false` for layouts the GPU has no kernel for (e.g.
/// `prefer_format="csc"` → no GPU CSC kernel), so a `device="auto"`/`"gpu"`
/// request on a GPU host records [`FallbackReason::UnsupportedInputLayout`]
/// rather than implying CUDA was absent.
pub fn cpu_exec_info(
    device: DeviceRequest,
    layout: InputLayout,
    gpu_eligible: bool,
    csc_available: bool,
    chunk_size: Option<usize>,
) -> AccelExecutionInfo {
    let mut info = plan_de_route(
        device,
        layout,
        gpu_runtime_available(),
        gpu_eligible,
        csc_available,
    );
    info.chunk_size = chunk_size;
    info
}

/// Build the execution info for an HVG dispatch via the single planner.
///
/// `gpu_eligible` is `true` only for the `seurat_v3` flavor family (the one
/// flavor with a GPU kernel). `csc_available` is `true` when a CSC sidecar is
/// reachable for a single-batch run, routing GPU to the column-major reduce
/// (`gpu_csc_v3`) and CPU to `cpu_csc`. The caller passes the validated
/// device request and the actual flavor/layout so the recorded route matches
/// the code that ran.
pub fn hvg_exec_info(
    device: DeviceRequest,
    gpu_eligible: bool,
    csc_available: bool,
) -> AccelExecutionInfo {
    plan_hvg_route(device, gpu_runtime_available(), gpu_eligible, csc_available)
}

/// Build the execution info for a single-route op (PCA / kNN / UMAP / Leiden /
/// preprocessing) via the generic planner. `gpu_eligible` reflects whether the
/// op's GPU library was actually usable at dispatch (cuVS / cuML / cuGraph /
/// cuSPARSE present); `gpu_route` / `cpu_route` are the op's CSR- or
/// dense-shaped route pair.
pub fn simple_exec_info(
    device: DeviceRequest,
    gpu_eligible: bool,
    gpu_route: AccelRoute,
    cpu_route: AccelRoute,
) -> AccelExecutionInfo {
    plan_simple_gpu_route(
        device,
        gpu_runtime_available(),
        gpu_eligible,
        gpu_route,
        cpu_route,
    )
}

/// Build the execution info for a pseudobulk NB-GLM dispatch via
/// [`plan_nb_glm_route`]. `gpu_eligible` is `false` when the design width or
/// sample count exceeds the kernel's register bounds (`scx_gpu::GPU_NB_GLM_PMAX`
/// / `GPU_NB_GLM_NSUB_MAX`) — recorded as `UnsupportedDimensions`. Stage A is
/// always CSR (host-fed dense); never a rapids fallback.
pub fn nb_glm_exec_info(device: DeviceRequest, gpu_eligible: bool) -> AccelExecutionInfo {
    plan_nb_glm_route(device, gpu_runtime_available(), gpu_eligible)
}

/// Build the execution info for a CPU-only op that has no GPU kernel at all
/// (gene-set scoring). `gpu_eligible=false` records `UserForcedCpu` for
/// `device="cpu"`, `NoCuda` when no GPU is present, and `UnsupportedInputLayout`
/// for an explicit GPU request on a GPU host — there is no GPU kernel to take.
pub fn cpu_only_exec_info(device: DeviceRequest) -> AccelExecutionInfo {
    simple_exec_info(device, false, AccelRoute::CpuCsr, AccelRoute::CpuCsr)
}

/// Whether a planned route warrants a default-visible GPU→CPU fallback
/// warning at the binding's dispatch point. Pure (no binding types) so it is
/// unit-testable and shared.
///
/// True only when the user **explicitly** asked for a GPU (`"gpu"` / `"gpu:N"`,
/// not `"auto"` — `auto`→CPU on a CPU host is expected), the planned route is a
/// CPU route, and the reason is a GPU-was-unusable reason. `NoRapids` is excluded
/// (the rapids path already emits its own richer one-shot warning); `UserForcedCpu`
/// / `None` / `NoCscSidecar` (still a GPU route) are not fallbacks worth warning
/// about. Note device resolution already hard-errors an explicit `device="gpu"`
/// when no CUDA GPU is present, so in practice this fires for
/// GPU-present-but-unsupported-layout/-dimensions cases.
pub fn should_warn_gpu_fallback(device: &str, info: &AccelExecutionInfo) -> bool {
    // Negative match (forward-compatible): a *new* `FallbackReason` variant
    // warns by default rather than silently passing through. The excluded
    // reasons are the non-fallbacks: `None` (took the GPU), `UserForcedCpu`
    // (the user asked for CPU), `NoRapids` (the rapids path emits its own
    // richer one-shot warning), and `NoCscSidecar` (still a GPU route —
    // CSR-direct instead of CSC-direct).
    device.starts_with("gpu")
        && !info.route.is_gpu()
        && !matches!(
            info.fallback_reason,
            FallbackReason::None
                | FallbackReason::UserForcedCpu
                | FallbackReason::NoRapids
                | FallbackReason::NoCscSidecar
        )
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
#[path = "route_tests.rs"]
mod tests;
