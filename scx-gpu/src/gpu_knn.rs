//! GPU kNN via cuVS CAGRA with runtime library loading.
//!
//! Provides [`gpu_knn_cagra_device`] for GPU-accelerated k-nearest neighbor
//! search using NVIDIA's CAGRA algorithm (part of the cuVS library). The cuVS
//! C API (`libcuvs_c.so`) is loaded at runtime via `libloading`, so cuVS is
//! NOT required at build time — users without cuVS get a clear error and can
//! fall back to CPU HNSW.
//!
//! ## DLPack v0.8
//!
//! cuVS uses `DLManagedTensor` from DLPack v0.8 for data exchange. We define
//! the minimal DLPack structs needed as `#[repr(C)]` Rust types. When cuVS
//! migrates to DLPack v1.0 (`DLManagedTensorVersioned`), these FFI definitions
//! will need updating.
//!
//! ## Pipeline
//!
//! 1. Upload embeddings to GPU (or take existing `CudaSlice<f32>`)
//! 2. Load `libcuvs_c.so` at runtime, cache function pointers
//! 3. Build CAGRA index: `cuvsCagraBuild` (GPU graph construction)
//! 4. Search CAGRA index: `cuvsCagraSearch` (GPU batch query)
//! 5. Download indices (u32 → i64) and distances to host
//!
//! ## Graceful Fallback
//!
//! When `libcuvs_c.so` is not available, [`cuvs_available`] returns `false`
//! and [`gpu_knn_cagra_device`] returns `GpuError::LibraryNotFound`. The
//! caller (the fused `scx-accel` PCA→kNN pipeline) falls back to CPU HNSW.

use std::sync::OnceLock;

use cudarc::driver::safe::{DevicePtr, DevicePtrMut};

use crate::device::GpuDevice;
use crate::device_resident::{DeviceEmbedding, DeviceKnnGraph};
use crate::error::GpuError;

// ---------------------------------------------------------------------------
// DLPack v0.8 FFI types
// ---------------------------------------------------------------------------

/// DLPack device type.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
struct DLDeviceType(i32);

#[allow(dead_code)]
impl DLDeviceType {
    const CPU: Self = Self(1);
    const CUDA: Self = Self(2);
}

/// DLPack device (type + id).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DLDevice {
    device_type: DLDeviceType,
    device_id: i32,
}

/// DLPack data type.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct DLDataType {
    code: u8, // 0=int, 1=uint, 2=float
    bits: u8,
    lanes: u16,
}

impl DLDataType {
    fn float32() -> Self {
        Self {
            code: 2,
            bits: 32,
            lanes: 1,
        }
    }
    fn uint32() -> Self {
        Self {
            code: 1,
            bits: 32,
            lanes: 1,
        }
    }
}

/// DLPack tensor — the core tensor descriptor.
#[repr(C)]
struct DLTensor {
    data: *mut std::ffi::c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

/// DLPack managed tensor (v0.8) — owns a DLTensor + optional deleter.
#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut std::ffi::c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// ---------------------------------------------------------------------------
// cuVS C API opaque types
// ---------------------------------------------------------------------------

type CuvsError = i32;
type CuvsResources = usize; // cuvsResources_t is a pointer-sized handle
type CuvsCagraIndex = usize; // opaque handle

/// Check a cuVS return code and convert to GpuError.
/// cuVS C API uses: CUVS_ERROR = 0, CUVS_SUCCESS = 1.
const CUVS_SUCCESS: CuvsError = 1;

fn check_cuvs(code: CuvsError, context: &str) -> Result<(), GpuError> {
    if code == CUVS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::CuVsError(format!(
            "{context}: cuVS error code {code}"
        )))
    }
}

// ---------------------------------------------------------------------------
// cuVS C API function pointer types
// ---------------------------------------------------------------------------

type FnResourcesCreate = unsafe extern "C" fn(*mut CuvsResources) -> CuvsError;
type FnResourcesDestroy = unsafe extern "C" fn(CuvsResources) -> CuvsError;

/// `cuvsStreamSet(cuvsResources_t, cudaStream_t)` — bind the RAFT resources to
/// a caller-supplied stream. `cudaStream_t` is ABI-compatible with the driver
/// API's `CUstream`, which is what cudarc hands out (same convention as
/// `nvcomp.rs`).
type FnStreamSet = unsafe extern "C" fn(CuvsResources, *mut std::ffi::c_void) -> CuvsError;
/// `cuvsStreamGet(cuvsResources_t, cudaStream_t*)` — read back the stream the
/// resources are on. Used only to report it.
type FnStreamGet = unsafe extern "C" fn(CuvsResources, *mut *mut std::ffi::c_void) -> CuvsError;
/// `cuvsStreamSync(cuvsResources_t)` — block until the resources' stream drains.
type FnStreamSync = unsafe extern "C" fn(CuvsResources) -> CuvsError;

type FnCagraIndexParamsCreate = unsafe extern "C" fn(*mut *mut CagraIndexParams) -> CuvsError;
type FnCagraIndexParamsDestroy = unsafe extern "C" fn(*mut CagraIndexParams) -> CuvsError;

type FnCagraSearchParamsCreate = unsafe extern "C" fn(*mut *mut CagraSearchParams) -> CuvsError;
type FnCagraSearchParamsDestroy = unsafe extern "C" fn(*mut CagraSearchParams) -> CuvsError;

type FnCagraIndexCreate = unsafe extern "C" fn(*mut CuvsCagraIndex) -> CuvsError;
type FnCagraIndexDestroy = unsafe extern "C" fn(CuvsCagraIndex) -> CuvsError;

type FnCagraBuild = unsafe extern "C" fn(
    CuvsResources,
    *const CagraIndexParams,
    *mut DLManagedTensor,
    CuvsCagraIndex,
) -> CuvsError;

/// cuvsFilter struct for search filtering (cuVS 26.02+).
/// Pass `CuvsFilter { addr: 0, filter_type: 0 }` for no filtering.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvsFilter {
    addr: usize,      // uintptr_t — device pointer to filter data (0 = no filter)
    filter_type: i32, // NO_FILTER=0, BITSET=1, BITMAP=2
}

type FnCagraSearch = unsafe extern "C" fn(
    CuvsResources,
    *const CagraSearchParams,
    CuvsCagraIndex,
    *mut DLManagedTensor, // queries
    *mut DLManagedTensor, // neighbors (output)
    *mut DLManagedTensor, // distances (output)
    CuvsFilter,           // filter (pass NO_FILTER for unfiltered search)
) -> CuvsError;

// ---------------------------------------------------------------------------
// cuVS CAGRA parameter structs (repr(C) matching cuvs/c_api.h)
// ---------------------------------------------------------------------------

/// CAGRA build algorithm selection (matches enum cuvsCagraGraphBuildAlgo in cuvs 26.02).
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum CagraBuildAlgo {
    AutoSelect = 0,
    IvfPq = 1,
    NnDescent = 2,
    IterativeCagraSearch = 3,
}

/// CAGRA index build parameters.
///
/// Layout matches `struct cuvsCagraIndexParams` in cuvs/neighbors/cagra.h (26.02).
/// The struct is heap-allocated by cuVS via `cuvsCagraIndexParamsCreate` and
/// initialized with defaults. We only modify specific fields after creation.
#[repr(C)]
#[allow(dead_code)]
struct CagraIndexParams {
    metric: i32, // cuvsDistanceType (L2InnerProduct=0, ...)
    intermediate_graph_degree: usize,
    graph_degree: usize,
    build_algo: CagraBuildAlgo,
    nn_descent_niter: usize,
    compression: *mut std::ffi::c_void, // cuvsCagraCompressionParams_t (nullable)
    graph_build_params: *mut std::ffi::c_void, // optional build params (nullable)
}

/// CAGRA search parameters.
///
/// Allocated by cuVS via `cuvsCagraSearchParamsCreate`.
#[repr(C)]
#[allow(dead_code)]
struct CagraSearchParams {
    max_queries: usize,
    itopk_size: usize,
    max_iterations: usize,
    algo: i32, // SINGLE_CTA=0, MULTI_CTA=1, MULTI_KERNEL=2, AUTO=3
    team_size: usize,
    search_width: usize,
    min_iterations: usize,
    thread_block_size: usize,
    hashmap_mode: i32,
    hashmap_min_bitlen: usize,
    hashmap_max_fill_rate: f32,
    num_random_samplings: u32,
    rand_xor_mask: u64,
}

// ---------------------------------------------------------------------------
// Runtime library loader
// ---------------------------------------------------------------------------

/// Cached cuVS library handle with resolved function pointers.
struct CuvsLibrary {
    _lib: libloading::Library,
    resources_create: FnResourcesCreate,
    resources_destroy: FnResourcesDestroy,
    index_params_create: FnCagraIndexParamsCreate,
    index_params_destroy: FnCagraIndexParamsDestroy,
    search_params_create: FnCagraSearchParamsCreate,
    search_params_destroy: FnCagraSearchParamsDestroy,
    index_create: FnCagraIndexCreate,
    index_destroy: FnCagraIndexDestroy,
    build: FnCagraBuild,
    search: FnCagraSearch,
    /// Optional because resolving them with `?` would make an older libcuvs
    /// that lacks one read as "cuVS not installed" — `load_cuvs_library` fails
    /// the whole load on any unresolved symbol, and `cuvs_available()` is what
    /// every caller probes before choosing CAGRA over CPU HNSW. A missing
    /// symbol must degrade the synchronization strategy, not the library.
    stream_set: Option<FnStreamSet>,
    stream_get: Option<FnStreamGet>,
    stream_sync: Option<FnStreamSync>,
}

// SAFETY: cuVS library handles are thread-safe per NVIDIA documentation.
// Function pointers are resolved once and immutable thereafter.
unsafe impl Send for CuvsLibrary {}
unsafe impl Sync for CuvsLibrary {}

/// Find the first directory matching a simple `*` glob pattern.
/// Only supports a single `*` wildcard in one path component.
fn glob_first(pattern: &str) -> Result<std::path::PathBuf, String> {
    // Split on the component containing `*`
    let components: Vec<&str> = pattern.split('/').collect();
    let star_idx = components
        .iter()
        .position(|c| c.contains('*'))
        .ok_or("no wildcard in pattern")?;

    let parent = components[..star_idx].join("/");
    let glob_part = components[star_idx];
    let suffix_parts = &components[star_idx + 1..];

    // Read parent dir and find matching entries
    let entries = std::fs::read_dir(&parent).map_err(|e| format!("read_dir {parent}: {e}"))?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        // Simple prefix/suffix match for patterns like "python*"
        let prefix = glob_part.split('*').next().unwrap_or("");
        let suffix_glob = glob_part.split('*').nth(1).unwrap_or("");
        if name_str.starts_with(prefix) && name_str.ends_with(suffix_glob) {
            let mut candidate = entry.path();
            for part in suffix_parts {
                candidate = candidate.join(part);
            }
            if candidate.is_dir() {
                return Ok(candidate);
            }
        }
    }
    Err(format!("no match for {pattern}"))
}

/// Global cached cuVS library. Loaded once on first use.
static CUVS_LIB: OnceLock<Result<CuvsLibrary, String>> = OnceLock::new();

fn load_cuvs_library() -> Result<CuvsLibrary, String> {
    // Try common library names via standard dlopen paths (LD_LIBRARY_PATH, etc.)
    let lib_names = ["libcuvs_c.so", "libcuvs.so"];

    // Also probe pip/conda install locations where the .so may live
    let mut search_paths: Vec<std::path::PathBuf> = Vec::new();

    // pip: .venv/lib/pythonX.Y/site-packages/libcuvs/lib64/
    // conda: $CONDA_PREFIX/lib/
    if let Ok(virtual_env) = std::env::var("VIRTUAL_ENV") {
        // Glob for any python version in the venv
        let site_pattern = format!("{virtual_env}/lib/python*/site-packages/libcuvs/lib64");
        if let Ok(entries) = glob_first(&site_pattern) {
            search_paths.push(entries);
        }
        // libcuvs_c.so has transitive deps on librmm.so, librapids_logger.so,
        // and libraft.so which live in separate pip package directories.
        // Pre-load them so they are in the process link map when dlopen
        // resolves DT_NEEDED entries for libcuvs_c.so.
        //
        // NOTE: We intentionally leak the Library handles (via std::mem::forget)
        // so the shared objects stay loaded for the lifetime of the process.
        // Setting LD_LIBRARY_PATH at runtime does NOT work — the dynamic linker
        // caches it at process startup (see ld.so(8)).
        let dep_dirs = ["librmm/lib64", "rapids_logger/lib64", "libraft/lib64"];
        let dep_lib_names = ["librmm.so", "librapids_logger.so", "libraft.so"];
        for dep_dir in &dep_dirs {
            let pattern = format!("{virtual_env}/lib/python*/site-packages/{dep_dir}");
            if let Ok(dep_path) = glob_first(&pattern) {
                for dep_name in &dep_lib_names {
                    let full = dep_path.join(dep_name);
                    if full.exists() {
                        if let Ok(lib) = unsafe { libloading::Library::new(&full) } {
                            // Leak the handle so the library stays loaded.
                            std::mem::forget(lib);
                        }
                    }
                }
            }
        }
    }
    if let Ok(conda_prefix) = std::env::var("CONDA_PREFIX") {
        search_paths.push(std::path::PathBuf::from(format!("{conda_prefix}/lib")));
    }

    // Try standard dlopen first
    let lib = lib_names
        .iter()
        .find_map(|name| unsafe { libloading::Library::new(name).ok() })
        // Then try explicit paths
        .or_else(|| {
            for dir in &search_paths {
                for name in &lib_names {
                    let full = dir.join(name);
                    if full.exists() {
                        if let Ok(l) = unsafe { libloading::Library::new(&full) } {
                            return Some(l);
                        }
                    }
                }
            }
            None
        })
        .ok_or_else(|| {
            format!(
                "cuVS library not found. Tried: {} (+ paths: {:?}). \
                 Install via: conda install -c rapidsai -c conda-forge libcuvs \
                 or pip install cuvs-cu12",
                lib_names.join(", "),
                search_paths,
            )
        })?;

    unsafe {
        let resources_create: FnResourcesCreate = *lib
            .get(b"cuvsResourcesCreate\0")
            .map_err(|e| format!("cuvsResourcesCreate: {e}"))?;
        let resources_destroy: FnResourcesDestroy = *lib
            .get(b"cuvsResourcesDestroy\0")
            .map_err(|e| format!("cuvsResourcesDestroy: {e}"))?;
        let index_params_create: FnCagraIndexParamsCreate = *lib
            .get(b"cuvsCagraIndexParamsCreate\0")
            .map_err(|e| format!("cuvsCagraIndexParamsCreate: {e}"))?;
        let index_params_destroy: FnCagraIndexParamsDestroy = *lib
            .get(b"cuvsCagraIndexParamsDestroy\0")
            .map_err(|e| format!("cuvsCagraIndexParamsDestroy: {e}"))?;
        let search_params_create: FnCagraSearchParamsCreate = *lib
            .get(b"cuvsCagraSearchParamsCreate\0")
            .map_err(|e| format!("cuvsCagraSearchParamsCreate: {e}"))?;
        let search_params_destroy: FnCagraSearchParamsDestroy = *lib
            .get(b"cuvsCagraSearchParamsDestroy\0")
            .map_err(|e| format!("cuvsCagraSearchParamsDestroy: {e}"))?;
        let index_create: FnCagraIndexCreate = *lib
            .get(b"cuvsCagraIndexCreate\0")
            .map_err(|e| format!("cuvsCagraIndexCreate: {e}"))?;
        let index_destroy: FnCagraIndexDestroy = *lib
            .get(b"cuvsCagraIndexDestroy\0")
            .map_err(|e| format!("cuvsCagraIndexDestroy: {e}"))?;
        let build: FnCagraBuild = *lib
            .get(b"cuvsCagraBuild\0")
            .map_err(|e| format!("cuvsCagraBuild: {e}"))?;
        let search: FnCagraSearch = *lib
            .get(b"cuvsCagraSearch\0")
            .map_err(|e| format!("cuvsCagraSearch: {e}"))?;
        // `.ok()`, not `?` — see the fields' doc on `CuvsLibrary`.
        let stream_set: Option<FnStreamSet> = lib.get(b"cuvsStreamSet\0").ok().map(|f| *f);
        let stream_get: Option<FnStreamGet> = lib.get(b"cuvsStreamGet\0").ok().map(|f| *f);
        let stream_sync: Option<FnStreamSync> = lib.get(b"cuvsStreamSync\0").ok().map(|f| *f);

        // cuVS version check — hard-fail if FFI struct layouts may differ.
        // CagraIndexParams/CagraSearchParams are pinned to cuVS 26.02 C headers.
        // Set SCX_CUVS_TRUST_LAYOUT=1 to downgrade to a warning for newer versions.
        type FnCuvsVersion = unsafe extern "C" fn() -> *const std::ffi::c_char;
        if let Ok(version_sym) = lib.get::<FnCuvsVersion>(b"cuvs_version\0") {
            let version_ptr = (*version_sym)();
            if !version_ptr.is_null() {
                let version_str = std::ffi::CStr::from_ptr(version_ptr).to_string_lossy();
                check_cuvs_version(&version_str)?;
            }
        }

        Ok(CuvsLibrary {
            _lib: lib,
            resources_create,
            resources_destroy,
            index_params_create,
            index_params_destroy,
            search_params_create,
            search_params_destroy,
            index_create,
            index_destroy,
            build,
            search,
            stream_set,
            stream_get,
            stream_sync,
        })
    }
}

/// Check cuVS version compatibility with our pinned FFI struct layouts.
///
/// Returns `Ok(())` if the version is compatible (26.02) or if the user has
/// opted in via `SCX_CUVS_TRUST_LAYOUT=1`. Returns `Err` with a descriptive
/// message if the version is incompatible and the override is not set.
fn check_cuvs_version(version_str: &str) -> std::result::Result<(), String> {
    let parts: Vec<&str> = version_str.split('.').collect();
    if parts.len() >= 2 {
        let major_minor = format!("{}.{}", parts[0], parts[1]);
        if major_minor != "26.02" {
            if std::env::var("SCX_CUVS_TRUST_LAYOUT").as_deref() == Ok("1") {
                eprintln!(
                    "scx-gpu: WARNING — cuVS {version_str} detected, but FFI struct \
                     layouts are pinned to 26.02. Proceeding because \
                     SCX_CUVS_TRUST_LAYOUT=1. kNN results may be incorrect if \
                     CagraIndexParams/CagraSearchParams changed. \
                     See scx-gpu/src/gpu_knn.rs."
                );
            } else {
                return Err(format!(
                    "cuVS {version_str} detected, but FFI struct layouts are pinned \
                     to 26.02. CagraIndexParams/CagraSearchParams may have changed, \
                     risking incorrect kNN results. Set SCX_CUVS_TRUST_LAYOUT=1 to \
                     override this check. See scx-gpu/src/gpu_knn.rs."
                ));
            }
        }
    }
    Ok(())
}

fn get_cuvs() -> Result<&'static CuvsLibrary, GpuError> {
    let result = CUVS_LIB.get_or_init(load_cuvs_library);
    match result {
        Ok(lib) => Ok(lib),
        Err(msg) => Err(GpuError::LibraryNotFound(msg.clone())),
    }
}

/// Bind cuVS's resources to `dev`'s stream, returning whether it took.
///
/// Returns `Ok(true)` when `cuvsStreamSet` was available and succeeded, so
/// every CAGRA launch is ordered against this crate's allocations, kernels and
/// downloads with no explicit synchronization at all. Measured on an H100 with
/// cuVS 26.02, this moves the resources from `cudaStreamPerThread` (`0x2`) to
/// cudarc's legacy NULL stream (`0x0`). Returns `Ok(false)` when
/// the loaded libcuvs has no `cuvsStreamSet`: the caller then falls back to
/// draining both sides explicitly, which is correct but costs two extra device
/// syncs per call. A non-zero return from the symbol is also `Ok(false)` — it
/// means cuVS declined the stream, not that the call is unusable — and is
/// logged once so a silent downgrade is still visible.
fn bind_cuvs_stream(
    cuvs: &CuvsLibrary,
    res: CuvsResources,
    dev: &GpuDevice,
) -> Result<bool, GpuError> {
    if let Some(stream_set) = cuvs.stream_set {
        // Same convention as `nvcomp.rs`: cudarc hands out a driver-API
        // `CUstream`, which is ABI-compatible with the runtime's `cudaStream_t`.
        let stream = dev.stream().cu_stream() as *mut std::ffi::c_void;
        let rc = unsafe { stream_set(res, stream) };
        if rc == CUVS_SUCCESS {
            return Ok(true);
        }
        log::warn!(
            "scx-gpu: cuvsStreamSet returned {rc}; falling back to explicit \
             stream synchronization around CAGRA (correct, but two extra device \
             drains per call)"
        );
    }
    // The input side of the hazard, for the unbound path: order everything
    // already queued on this crate's stream — the embedding upload, the output
    // allocations — before cuVS reads or writes any of it.
    dev.synchronize()?;
    Ok(false)
}

/// Block until cuVS's stream has drained, if the loaded libcuvs exposes
/// `cuvsStreamSync`.
///
/// A no-op on a libcuvs without the symbol, which is the one configuration this
/// cannot make safe; the version floor in `docs/gpu-setup.md` (cuVS >= 25.10)
/// is well above where these three entered the C API, so it is a theoretical
/// gap rather than a supported one.
fn sync_cuvs_stream(cuvs: &CuvsLibrary, res: CuvsResources, context: &str) -> Result<(), GpuError> {
    let Some(stream_sync) = cuvs.stream_sync else {
        return Ok(());
    };
    check_cuvs(unsafe { stream_sync(res) }, context)
}

/// The stream cuVS's resources are on, as a raw pointer, for diagnostics.
///
/// `None` when the loaded libcuvs has no `cuvsStreamGet` or the call fails.
/// Exists so a GPU-node run can *show* that cuVS and cudarc are on the same
/// stream rather than asserting it — cudarc's is the legacy NULL stream
/// (`std::ptr::null_mut()`), so "bound" is visibly `0x0`.
#[cfg_attr(not(test), allow(dead_code))]
fn cuvs_stream_ptr(cuvs: &CuvsLibrary, res: CuvsResources) -> Option<*mut std::ffi::c_void> {
    let stream_get = cuvs.stream_get?;
    let mut out: *mut std::ffi::c_void = std::ptr::null_mut();
    (unsafe { stream_get(res, &mut out) } == CUVS_SUCCESS).then_some(out)
}

/// What a fresh `cuvsResources_t` looks like before and after this crate binds
/// its stream.
///
/// Test-only, and the premise check for review §8.5. `before` is the stream
/// `cuvsResourcesCreate` chose on its own — that is the one that had no
/// ordering edge with anything this crate queues, and reading it is the only
/// way to say whether the hazard was *live* on a given device or merely
/// structural. `after` is what the binding leaves.
#[cfg(test)]
struct CuvsStreamProbe {
    /// cudarc's stream — the legacy NULL stream, i.e. `0x0`.
    cudarc: *mut std::ffi::c_void,
    /// cuVS's own stream, as created. `None` if `cuvsStreamGet` is unavailable.
    before: Option<*mut std::ffi::c_void>,
    /// cuVS's stream after `bind_cuvs_stream`.
    after: Option<*mut std::ffi::c_void>,
    /// Whether `cuvsStreamSet` was present and took.
    bound: bool,
}

#[cfg(test)]
fn probe_cuvs_stream_binding(dev: &GpuDevice) -> Result<CuvsStreamProbe, GpuError> {
    let cuvs = get_cuvs()?;
    let mut res: CuvsResources = 0;
    unsafe { check_cuvs((cuvs.resources_create)(&mut res), "cuvsResourcesCreate")? };
    let _guard = CuvsResourcesGuard {
        handle: res,
        destroy_fn: cuvs.resources_destroy,
    };
    let before = cuvs_stream_ptr(cuvs, res);
    let bound = bind_cuvs_stream(cuvs, res, dev)?;
    Ok(CuvsStreamProbe {
        cudarc: dev.stream().cu_stream() as *mut std::ffi::c_void,
        before,
        after: cuvs_stream_ptr(cuvs, res),
        bound,
    })
}

// ---------------------------------------------------------------------------
// RAII guards for cuVS resources (replaces scopeguard)
// ---------------------------------------------------------------------------

/// RAII guard for cuVS resources handle.
struct CuvsResourcesGuard {
    handle: CuvsResources,
    destroy_fn: FnResourcesDestroy,
}

impl Drop for CuvsResourcesGuard {
    fn drop(&mut self) {
        unsafe {
            let ret = (self.destroy_fn)(self.handle);
            if ret != CUVS_SUCCESS {
                eprintln!("scx-gpu: cuvsResourcesDestroy failed (error code {ret})");
            }
        }
    }
}

/// RAII guard for cuVS CAGRA index.
struct CuvsIndexGuard {
    handle: CuvsCagraIndex,
    destroy_fn: FnCagraIndexDestroy,
}

impl Drop for CuvsIndexGuard {
    fn drop(&mut self) {
        unsafe {
            let ret = (self.destroy_fn)(self.handle);
            if ret != CUVS_SUCCESS {
                eprintln!("scx-gpu: cuvsCagraIndexDestroy failed (error code {ret})");
            }
        }
    }
}

/// RAII guard for cuVS CAGRA index params.
struct CuvsIndexParamsGuard {
    ptr: *mut CagraIndexParams,
    destroy_fn: FnCagraIndexParamsDestroy,
}

impl Drop for CuvsIndexParamsGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let ret = (self.destroy_fn)(self.ptr);
                if ret != CUVS_SUCCESS {
                    eprintln!("scx-gpu: cuvsCagraIndexParamsDestroy failed (error code {ret})");
                }
            }
        }
    }
}

/// RAII guard for cuVS CAGRA search params.
struct CuvsSearchParamsGuard {
    ptr: *mut CagraSearchParams,
    destroy_fn: FnCagraSearchParamsDestroy,
}

impl Drop for CuvsSearchParamsGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let ret = (self.destroy_fn)(self.ptr);
                if ret != CUVS_SUCCESS {
                    eprintln!("scx-gpu: cuvsCagraSearchParamsDestroy failed (error code {ret})");
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Check whether the cuVS CAGRA library is available at runtime.
///
/// Returns `true` if `libcuvs_c.so` (or `libcuvs.so`) can be loaded and
/// all required function symbols are resolved. The result is cached after
/// the first call.
pub fn cuvs_available() -> bool {
    CUVS_LIB.get_or_init(load_cuvs_library).is_ok()
}

/// Result of GPU kNN search.
#[derive(Debug, Clone)]
pub struct GpuKnnResult {
    /// Neighbor indices: flat row-major `(n_obs × n_neighbors)`, i64.
    pub indices: Vec<i64>,
    /// Neighbor distances: flat row-major `(n_obs × n_neighbors)`, f32.
    pub distances: Vec<f32>,
    /// Number of observations.
    pub n_obs: usize,
    /// Number of neighbors.
    pub n_neighbors: usize,
}

/// GPU kNN via cuVS CAGRA, **device-resident** input and output (V3 plan
/// Phase 2.3).
///
/// Reads `embedding` (a row-major `(n_obs × n_dims)` [`DeviceEmbedding`], e.g.
/// from `gpu_randomized_pca_device`) directly as the CAGRA dataset — no host
/// upload — and returns a [`DeviceKnnGraph`] holding the raw `u32` neighbor
/// indices + L2-squared distances **on the GPU** (no download, no self-filter).
/// The fused PCA → kNN path uses this to keep the embedding resident across the
/// handoff; call [`DeviceKnnGraph::to_host`] to materialize the host
/// [`GpuKnnResult`].
///
/// # Errors
///
/// * `GpuError::LibraryNotFound` — cuVS not installed
/// * `GpuError::CuVsError` — bad `n_neighbors`, or CAGRA build/search failed
/// * `GpuError::OutOfMemory` — GPU memory exhausted
pub fn gpu_knn_cagra_device(
    dev: &GpuDevice,
    embedding: &DeviceEmbedding,
    n_neighbors: usize,
) -> Result<DeviceKnnGraph, GpuError> {
    let (n_obs, n_dims) = embedding.shape();

    if n_neighbors == 0 || n_neighbors > n_obs {
        return Err(GpuError::CuVsError(format!(
            "n_neighbors ({n_neighbors}) must be in [1, n_obs ({n_obs})]"
        )));
    }

    let cuvs = get_cuvs()?;

    // cuVS internally uses the CUDA Runtime API (cudaSetDevice, cudaStreamCreate).
    // cudarc uses the Driver API (cuCtxCreate). We must ensure the runtime is
    // initialized on the correct device before calling cuvsResourcesCreate,
    // which creates a RAFT handle that expects a valid runtime context.
    // Calling cudaSetDevice bridges the driver ↔ runtime context gap.
    unsafe {
        let ordinal = dev.context().ordinal() as i32;
        let rc = cudarc::runtime::sys::cudaSetDevice(ordinal);
        if rc != cudarc::runtime::sys::cudaError_t::cudaSuccess {
            return Err(GpuError::CudaError(format!(
                "cudaSetDevice({ordinal}) failed: {rc:?}"
            )));
        }
    }

    // Extract device ordinal for DLPack tensor construction.
    // All DLDevice structs must reference the correct GPU so that cuVS
    // addresses the right device's memory on multi-GPU hosts.
    let device_ordinal = dev.context().ordinal() as i32;

    // The CAGRA dataset reads directly from the device-resident embedding —
    // no host round-trip.
    let d_data = embedding.data();

    // Allocate output buffers on GPU
    // CAGRA returns u32 indices — we search k+1 to filter self-hits
    let search_k = (n_neighbors + 1).min(n_obs);
    let mut d_neighbors = dev.alloc_zeros::<u32>(n_obs * search_k)?;
    let mut d_distances = dev.alloc_zeros::<f32>(n_obs * search_k)?;

    // Create cuVS resources (wraps CUDA stream + allocator)
    let mut res: CuvsResources = 0;
    unsafe { check_cuvs((cuvs.resources_create)(&mut res), "cuvsResourcesCreate")? };
    let _res_guard = CuvsResourcesGuard {
        handle: res,
        destroy_fn: cuvs.resources_destroy,
    };

    // Put cuVS on the stream this crate already uses, so build and search are
    // ordered against the allocations and kernels around them.
    //
    // They were not, and this is **measured**, not inferred: on an H100 with
    // cuVS 26.02, `cuvsResourcesCreate` leaves the resources on `0x2` —
    // `cudaStreamPerThread`, the per-thread default stream — while cudarc's
    // `default_stream()` is `0x0`, the legacy NULL stream
    // (`cu_stream: null_mut()`). Those two are precisely the pair that does
    // **not** implicitly synchronize: PTDS exists to opt out of the legacy
    // default stream's cross-stream serialization. So the hazard was live here,
    // not merely structural. Nothing called `cuvsStreamSync`,
    // so the hazard ran in **both** directions: `d_data` is written by cudarc
    // kernels and then read by `cuvsCagraBuild` with no edge between them, and
    // `d_neighbors` / `d_distances` are stream-ordered `alloc_zeros` on the
    // NULL stream that cuVS then writes. `DeviceKnnGraph::to_host` synchronizes
    // — the wrong stream — so a download could return the `alloc_zeros`
    // contents while the search was still in flight: every neighbour `0`, every
    // distance `0.0`, and the self-filter padding that into a structurally
    // valid, meaningless kNN graph with no error (review §8.5).
    //
    // Binding the stream closes both directions by construction, which
    // `cuvsStreamSync` after the fact cannot. `sync_after` is the fallback for
    // a libcuvs without `cuvsStreamSet`: sync the cudarc stream *before* the
    // build so the input is ordered, and drain cuVS's stream after each call.
    let bound_stream = bind_cuvs_stream(cuvs, res, dev)?;

    // Create and configure CAGRA index parameters
    let mut index_params_ptr: *mut CagraIndexParams = std::ptr::null_mut();
    unsafe {
        check_cuvs(
            (cuvs.index_params_create)(&mut index_params_ptr),
            "cuvsCagraIndexParamsCreate",
        )?;
    }
    let _ip_guard = CuvsIndexParamsGuard {
        ptr: index_params_ptr,
        destroy_fn: cuvs.index_params_destroy,
    };

    // Set CAGRA build parameters (tuned for single-cell PCA embeddings)
    unsafe {
        (*index_params_ptr).graph_degree = 64;
        (*index_params_ptr).intermediate_graph_degree = 128;
        // IVF_PQ is faster than NN_DESCENT for >100K points
        (*index_params_ptr).build_algo = if n_obs > 100_000 {
            CagraBuildAlgo::IvfPq
        } else {
            CagraBuildAlgo::NnDescent
        };
    }

    // Create CAGRA index
    let mut index: CuvsCagraIndex = 0;
    unsafe { check_cuvs((cuvs.index_create)(&mut index), "cuvsCagraIndexCreate")? };
    let _index_guard = CuvsIndexGuard {
        handle: index,
        destroy_fn: cuvs.index_destroy,
    };

    // Build DLManagedTensor for the dataset
    let (data_ptr, _data_guard) = d_data.device_ptr(dev.stream());
    let mut dataset_shape: Vec<i64> = vec![n_obs as i64, n_dims as i64];
    let mut dataset_strides: Vec<i64> = vec![n_dims as i64, 1];
    let mut dataset_tensor = DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: device_ordinal,
            },
            ndim: 2,
            dtype: DLDataType::float32(),
            shape: dataset_shape.as_mut_ptr(),
            strides: dataset_strides.as_mut_ptr(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    };

    // Build CAGRA index
    unsafe {
        check_cuvs(
            (cuvs.build)(res, index_params_ptr, &mut dataset_tensor, index),
            "cuvsCagraBuild",
        )?;
    }
    sync_cuvs_stream(cuvs, res, "after cuvsCagraBuild")?;

    // Create and configure search parameters
    let mut search_params_ptr: *mut CagraSearchParams = std::ptr::null_mut();
    unsafe {
        check_cuvs(
            (cuvs.search_params_create)(&mut search_params_ptr),
            "cuvsCagraSearchParamsCreate",
        )?;
    }
    let _sp_guard = CuvsSearchParamsGuard {
        ptr: search_params_ptr,
        destroy_fn: cuvs.search_params_destroy,
    };

    // CAGRA requires `itopk_size >= search_k` (and a multiple of 32); the cuVS
    // default is 64, so leaving it unset silently fails for `search_k > 64`
    // (i.e. n_neighbors >= 64). Raise it to cover the request while keeping the
    // default 64 floor — the common small-k path is unchanged, and only the
    // previously-broken large-k path is affected. cuVS single-CTA caps itopk at
    // 1024; beyond that we let cuVS surface its own error.
    let itopk_size = search_k.next_multiple_of(32).max(64);
    unsafe {
        (*search_params_ptr).itopk_size = itopk_size;
    }

    // Build DLManagedTensors for queries (= dataset, self-query) and outputs
    // Queries tensor — same data as dataset
    let mut queries_shape: Vec<i64> = vec![n_obs as i64, n_dims as i64];
    let mut queries_strides: Vec<i64> = vec![n_dims as i64, 1];
    let mut queries_tensor = DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: device_ordinal,
            },
            ndim: 2,
            dtype: DLDataType::float32(),
            shape: queries_shape.as_mut_ptr(),
            strides: queries_strides.as_mut_ptr(),
            byte_offset: 0,
        },
        manager_ctx: std::ptr::null_mut(),
        deleter: None,
    };

    // Build DLManagedTensors and execute search in a scope so that the
    // `SyncOnDrop` guards from `device_ptr_mut` are dropped before the caller's
    // download. Those guards record an event on **cudarc's** stream and say
    // nothing about cuVS's work, so they were never the edge that made this
    // safe — `bind_cuvs_stream` and `sync_cuvs_stream` are.
    {
        // Neighbors tensor (u32, CUDA)
        let (neighbors_ptr, _ng_guard) = d_neighbors.device_ptr_mut(dev.stream());
        let mut neighbors_shape: Vec<i64> = vec![n_obs as i64, search_k as i64];
        let mut neighbors_strides: Vec<i64> = vec![search_k as i64, 1];
        let mut neighbors_tensor = DLManagedTensor {
            dl_tensor: DLTensor {
                data: neighbors_ptr as *mut std::ffi::c_void,
                device: DLDevice {
                    device_type: DLDeviceType::CUDA,
                    device_id: device_ordinal,
                },
                ndim: 2,
                dtype: DLDataType::uint32(),
                shape: neighbors_shape.as_mut_ptr(),
                strides: neighbors_strides.as_mut_ptr(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        // Distances tensor (f32, CUDA)
        let (distances_ptr, _dg_guard) = d_distances.device_ptr_mut(dev.stream());
        let mut distances_shape: Vec<i64> = vec![n_obs as i64, search_k as i64];
        let mut distances_strides: Vec<i64> = vec![search_k as i64, 1];
        let mut distances_tensor = DLManagedTensor {
            dl_tensor: DLTensor {
                data: distances_ptr as *mut std::ffi::c_void,
                device: DLDevice {
                    device_type: DLDeviceType::CUDA,
                    device_id: device_ordinal,
                },
                ndim: 2,
                dtype: DLDataType::float32(),
                shape: distances_shape.as_mut_ptr(),
                strides: distances_strides.as_mut_ptr(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: None,
        };

        // Execute CAGRA search (no filtering)
        let no_filter = CuvsFilter {
            addr: 0,
            filter_type: 0, // NO_FILTER
        };
        unsafe {
            check_cuvs(
                (cuvs.search)(
                    res,
                    search_params_ptr,
                    index,
                    &mut queries_tensor,
                    &mut neighbors_tensor,
                    &mut distances_tensor,
                    no_filter,
                ),
                "cuvsCagraSearch",
            )?;
        }
    } // SyncOnDrop guards dropped here

    // Drain cuVS's stream before the guard destroys the resources. Redundant
    // when `bound_stream` is true (the search ran on the stream `to_host` will
    // synchronize) and load-bearing when it is false, where it is the only
    // edge between the search and the download. `cudaStreamDestroy` is
    // non-blocking, so the guard is not a substitute for it either way.
    sync_cuvs_stream(cuvs, res, "after cuvsCagraSearch")?;

    // Return the raw CAGRA output device-resident. The self-hit filter, u32→i64
    // conversion, and sqrt (L2² → Euclidean) live in `DeviceKnnGraph::to_host`,
    // which synchronizes `dev.stream()` before downloading — the same stream
    // CAGRA ran on, once `bind_cuvs_stream` has bound it.
    if !bound_stream {
        log::debug!(
            "scx-gpu: CAGRA ran on cuVS's own stream, synchronized explicitly \
             (cuvsStreamSet unavailable on the loaded libcuvs)"
        );
    }
    DeviceKnnGraph::new(d_neighbors, d_distances, n_obs, search_k, n_neighbors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Run CAGRA on host data via the device-resident path: upload →
    /// [`DeviceEmbedding`] → [`gpu_knn_cagra_device`] → [`DeviceKnnGraph::to_host`].
    /// This is the same three-step sequence the (removed) host-bounce wrapper
    /// performed, kept here so the host-data correctness tests below exercise
    /// the surviving device entry point.
    fn knn_cagra_via_device(
        dev: &GpuDevice,
        data: &[f32],
        n_obs: usize,
        n_dims: usize,
        n_neighbors: usize,
    ) -> GpuKnnResult {
        let d_data = dev.htod_copy(data).unwrap();
        let embedding = DeviceEmbedding::new(d_data, n_obs, n_dims).unwrap();
        let graph = gpu_knn_cagra_device(dev, &embedding, n_neighbors).unwrap();
        graph.to_host(dev).unwrap()
    }

    #[test]
    fn test_cuvs_available_check() {
        // This just checks that the availability check doesn't panic.
        // On machines without cuVS, it returns false.
        // gpu-gate-exempt: the probe is the subject under test, not a gate on it.
        let available = cuvs_available();
        println!("cuVS available: {available}");
    }

    /// Requires both a GPU and a cuVS installation. `#[ignore]`d for the
    /// former; the cuVS half reports a skip marker on a GPU node without it.
    #[test]
    #[ignore = "requires a CUDA GPU + cuVS"]
    fn test_gpu_knn_cagra_basic() {
        let dev = require_gpu!();

        require_gpu_cap!(cuvs);

        // Two clusters in 3D: cluster A at origin, cluster B at (10,10,10)
        let n_per_cluster = 25;
        let n_obs = n_per_cluster * 2;
        let n_dims = 3;
        let n_neighbors = 5;

        let mut data = Vec::with_capacity(n_obs * n_dims);
        for i in 0..n_per_cluster {
            data.push(0.1 * (i as f32));
            data.push(0.1 * ((i % 5) as f32));
            data.push(0.05 * (i as f32));
        }
        for i in 0..n_per_cluster {
            data.push(10.0 + 0.1 * (i as f32));
            data.push(10.0 + 0.1 * ((i % 5) as f32));
            data.push(10.0 + 0.05 * (i as f32));
        }

        let result = knn_cagra_via_device(&dev, &data, n_obs, n_dims, n_neighbors);

        assert_eq!(result.n_obs, n_obs);
        assert_eq!(result.n_neighbors, n_neighbors);
        assert_eq!(result.indices.len(), n_obs * n_neighbors);
        assert_eq!(result.distances.len(), n_obs * n_neighbors);

        // All indices should be valid
        for &idx in &result.indices {
            assert!(idx >= 0 && (idx as usize) < n_obs, "invalid index: {idx}");
        }

        // All distances should be non-negative
        for &d in &result.distances {
            assert!(d >= 0.0, "negative distance: {d}");
        }

        // §8.5: the assertions below used to check cluster A only. With every
        // index read as `0` — the `alloc_zeros` contents, which is exactly what
        // an unsynchronized download returns — every neighbour of every cluster-A
        // point is index 0, which *is* in cluster A, so the separation check
        // passed and the test could not see the defect at all. Three additions
        // close that: cluster B (whose points must find neighbours at index
        // >= n_per_cluster), a non-degenerate distance spectrum, and distinct
        // neighbour sets per row.
        for i in 0..n_per_cluster {
            let same_cluster = (0..n_neighbors)
                .filter(|&j| (result.indices[i * n_neighbors + j] as usize) < n_per_cluster)
                .count();
            assert!(
                same_cluster >= 4,
                "point {i} has only {same_cluster}/{n_neighbors} neighbors in same cluster"
            );
        }
        for i in n_per_cluster..n_obs {
            let same_cluster = (0..n_neighbors)
                .filter(|&j| (result.indices[i * n_neighbors + j] as usize) >= n_per_cluster)
                .count();
            assert!(
                same_cluster >= 4,
                "cluster-B point {i} has only {same_cluster}/{n_neighbors} neighbors in \
                 cluster B — all-zero indices would land here"
            );
        }

        // Two well-separated clusters cannot produce a single distance value.
        // A zero-filled buffer is the degenerate case this rules out.
        let distinct = result
            .distances
            .iter()
            .any(|&d| (d - result.distances[0]).abs() > 1e-6);
        assert!(
            distinct,
            "every distance is {}, which is what an unsynchronized read of the \
             zero-initialised output buffer returns",
            result.distances[0]
        );

        // Distinct points have distinct neighbourhoods. All-zero output gives
        // every row the identical list.
        let first: &[i64] = &result.indices[0..n_neighbors];
        let differs =
            (1..n_obs).any(|i| &result.indices[i * n_neighbors..(i + 1) * n_neighbors] != first);
        assert!(differs, "every row has the same neighbour list: {first:?}");
    }

    /// cuVS and this crate must run on one CUDA stream.
    ///
    /// `cuvsResourcesCreate` builds RAFT resources on their own (RMM,
    /// non-blocking) stream, while cudarc's `default_stream()` is the legacy
    /// NULL stream — which has no implicit dependency on a non-blocking one. So
    /// a CAGRA build/search and the allocations and downloads around it had no
    /// ordering edge in either direction, and nothing in the tree called
    /// `cuvsStreamSync` (review §8.5).
    ///
    /// This reports the streams rather than inferring the fix from a downstream
    /// result, and reads cuVS's **before** binding as well as after — the
    /// before is the premise. Measured on an H100 with cuVS 26.02:
    /// `before Some(0x2), after Some(0x0)`, against a cudarc stream of `0x0`.
    /// `0x2` is `cudaStreamPerThread` and `0x0` is the legacy NULL stream, and
    /// those two do not implicitly synchronize with each other — which is what
    /// makes §8.5 a live defect on this hardware rather than a latent one.
    #[test]
    #[ignore = "requires a CUDA GPU + cuVS"]
    fn cagra_runs_on_the_same_stream_as_the_rest_of_the_crate() {
        let dev = require_gpu!();
        require_gpu_cap!(cuvs);

        let probe = probe_cuvs_stream_binding(&dev).unwrap();
        // Printed, not only asserted: `before` is the premise — whether this
        // device's cuVS actually chose a stream other than cudarc's, i.e.
        // whether §8.5 was live here or only structural.
        println!(
            "cuvsStreamSet applied: {}; cudarc stream {:?}; cuVS stream before {:?}, \
             after {:?}",
            probe.bound, probe.cudarc, probe.before, probe.after
        );

        let Some(cuvs_stream) = probe.after else {
            // No `cuvsStreamGet` to read back with. The fallback path in
            // `bind_cuvs_stream` still synchronizes both sides, so this is a
            // coverage gap on this libcuvs, not a failure.
            eprintln!(
                "{}: cuvsStreamGet unavailable, cannot read the bound stream back",
                crate::test_gate::SKIP_MARKER
            );
            return;
        };
        assert!(
            probe.bound,
            "cuvsStreamSet is present in this libcuvs but did not take"
        );
        assert_eq!(
            cuvs_stream, probe.cudarc,
            "cuVS is on a different stream from the one this crate allocates, \
             launches and downloads on — the hazard §8.5 describes"
        );
    }

    /// Regression: `n_neighbors >= 64` requires raising CAGRA's `itopk_size`
    /// above the cuVS default of 64 (`itopk_size >= search_k`). Before that fix
    /// this hard-errored in `cuvsCagraSearch`.
    #[test]
    #[ignore = "requires a CUDA GPU + cuVS"]
    fn test_gpu_knn_cagra_large_k() {
        let dev = require_gpu!();
        require_gpu_cap!(cuvs);

        // n_neighbors (80) well above the default itopk_size (64).
        let n_obs = 400usize;
        let n_dims = 8usize;
        let n_neighbors = 80usize;
        let mut data = Vec::with_capacity(n_obs * n_dims);
        for i in 0..n_obs {
            for d in 0..n_dims {
                data.push(0.01 * ((i * n_dims + d) % 97) as f32);
            }
        }

        let result = knn_cagra_via_device(&dev, &data, n_obs, n_dims, n_neighbors);
        assert_eq!(result.n_neighbors, n_neighbors);
        assert_eq!(result.indices.len(), n_obs * n_neighbors);
        for &idx in &result.indices {
            assert!(idx >= 0 && (idx as usize) < n_obs, "invalid index: {idx}");
        }
        for &dist in &result.distances {
            assert!(dist >= 0.0 && dist.is_finite(), "bad distance: {dist}");
        }
    }

    // --- cuVS version check tests ---

    #[test]
    fn test_cuvs_version_check_compatible() {
        // 26.02.x should always pass
        assert!(check_cuvs_version("26.02.00").is_ok());
        assert!(check_cuvs_version("26.02.1").is_ok());
        assert!(check_cuvs_version("26.02").is_ok());
    }

    #[test]
    fn test_cuvs_version_check_incompatible_hard_fail() {
        // Ensure the env var is NOT set for this test.
        // (We can't unset globally because tests run in parallel, but we
        // can verify the error message content.)
        // If SCX_CUVS_TRUST_LAYOUT happens to be set in the environment,
        // skip this test to avoid flakiness.
        if std::env::var("SCX_CUVS_TRUST_LAYOUT").as_deref() == Ok("1") {
            eprintln!("SCX_CUVS_TRUST_LAYOUT=1 is set — skipping hard-fail test");
            return;
        }
        let result = check_cuvs_version("27.01.00");
        assert!(result.is_err(), "should fail for incompatible version");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("27.01.00"),
            "error should mention the detected version"
        );
        assert!(
            msg.contains("SCX_CUVS_TRUST_LAYOUT"),
            "error should mention the override env var"
        );
    }

    #[test]
    fn test_cuvs_version_check_single_component() {
        // Version strings with < 2 components should pass (no check possible)
        assert!(check_cuvs_version("26").is_ok());
        assert!(check_cuvs_version("").is_ok());
    }
}
