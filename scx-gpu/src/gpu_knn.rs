//! GPU kNN via cuVS CAGRA with runtime library loading.
//!
//! Provides [`gpu_knn_cagra`] for GPU-accelerated k-nearest neighbor search
//! using NVIDIA's CAGRA algorithm (part of the cuVS library). The cuVS C API
//! (`libcuvs_c.so`) is loaded at runtime via `libloading`, so cuVS is NOT
//! required at build time — users without cuVS get a clear error and can fall
//! back to CPU HNSW.
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
//! and [`gpu_knn_cagra`] returns `GpuError::LibraryNotFound`. The caller
//! (typically `scx-accel::neighbors::build_knn_graph_gpu`) falls back to
//! CPU HNSW.

use std::sync::OnceLock;

use cudarc::driver::safe::{DevicePtr, DevicePtrMut};

use crate::device::GpuDevice;
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
        })
    }
}

fn get_cuvs() -> Result<&'static CuvsLibrary, GpuError> {
    let result = CUVS_LIB.get_or_init(load_cuvs_library);
    match result {
        Ok(lib) => Ok(lib),
        Err(msg) => Err(GpuError::LibraryNotFound(msg.clone())),
    }
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
            let _ = (self.destroy_fn)(self.handle);
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
            let _ = (self.destroy_fn)(self.handle);
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
                let _ = (self.destroy_fn)(self.ptr);
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
                let _ = (self.destroy_fn)(self.ptr);
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

/// GPU kNN via cuVS CAGRA.
///
/// Builds a CAGRA graph index from the embedding matrix and performs
/// batch k-nearest neighbor search, all on GPU.
///
/// # Arguments
///
/// * `dev` — GPU device handle
/// * `data` — Row-major embedding matrix `(n_obs × n_dims)` on host
/// * `n_obs` — Number of data points (rows)
/// * `n_dims` — Dimensionality (columns, typically 50 PCs)
/// * `n_neighbors` — Number of nearest neighbors to find
///
/// # Returns
///
/// `GpuKnnResult` with neighbor indices and distances on host.
/// Indices are 0-based, self-hits are excluded (CAGRA may return self).
///
/// # Errors
///
/// * `GpuError::LibraryNotFound` — cuVS not installed
/// * `GpuError::CuVsError` — CAGRA build or search failed
/// * `GpuError::OutOfMemory` — GPU memory exhausted
pub fn gpu_knn_cagra(
    dev: &GpuDevice,
    data: &[f32],
    n_obs: usize,
    n_dims: usize,
    n_neighbors: usize,
) -> Result<GpuKnnResult, GpuError> {
    if data.len() != n_obs * n_dims {
        return Err(GpuError::ShapeMismatch {
            expected: format!(
                "data length = n_obs × n_dims = {} × {} = {}",
                n_obs,
                n_dims,
                n_obs * n_dims
            ),
            got: format!("{}", data.len()),
        });
    }
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

    // Upload data to GPU
    let d_data = dev.htod_copy(data)?;

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
                device_id: 0,
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

    // Build DLManagedTensors for queries (= dataset, self-query) and outputs
    // Queries tensor — same data as dataset
    let mut queries_shape: Vec<i64> = vec![n_obs as i64, n_dims as i64];
    let mut queries_strides: Vec<i64> = vec![n_dims as i64, 1];
    let mut queries_tensor = DLManagedTensor {
        dl_tensor: DLTensor {
            data: data_ptr as *mut std::ffi::c_void,
            device: DLDevice {
                device_type: DLDeviceType::CUDA,
                device_id: 0,
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
    // SyncOnDrop guards from device_ptr_mut are dropped before dtoh_copy.
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
                    device_id: 0,
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
                    device_id: 0,
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

    // Synchronize and download results
    dev.synchronize()?;
    let neighbors_u32 = dev.dtoh_copy(&d_neighbors)?;
    let distances_f32 = dev.dtoh_copy(&d_distances)?;

    // Post-process: convert u32→i64, filter self-hits, take Euclidean distance (sqrt)
    // CAGRA returns L2 squared distances — take sqrt for Euclidean distance
    let mut indices = Vec::with_capacity(n_obs * n_neighbors);
    let mut distances = Vec::with_capacity(n_obs * n_neighbors);

    for i in 0..n_obs {
        let row_start = i * search_k;
        let mut count = 0;
        for j in 0..search_k {
            if count >= n_neighbors {
                break;
            }
            let neighbor_idx = neighbors_u32[row_start + j] as usize;
            if neighbor_idx == i {
                // Skip self
                continue;
            }
            indices.push(neighbor_idx as i64);
            // sqrt for Euclidean distance
            distances.push(distances_f32[row_start + j].sqrt());
            count += 1;
        }
        // Pad if we didn't find enough neighbors (shouldn't happen normally)
        while count < n_neighbors {
            indices.push(i as i64);
            distances.push(0.0);
            count += 1;
        }
    }

    Ok(GpuKnnResult {
        indices,
        distances,
        n_obs,
        n_neighbors,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuvs_available_check() {
        // This just checks that the availability check doesn't panic.
        // On machines without cuVS, it returns false.
        let available = cuvs_available();
        println!("cuVS available: {available}");
    }

    /// This test requires both a GPU and cuVS installation.
    /// It will be skipped gracefully on CI machines without these.
    #[test]
    fn test_gpu_knn_cagra_basic() {
        let dev = match GpuDevice::new(0) {
            Ok(dev) => dev,
            Err(_) => {
                eprintln!("CUDA not available — skipping GPU kNN test");
                return;
            }
        };

        if !cuvs_available() {
            eprintln!("cuVS not available — skipping GPU kNN test");
            return;
        }

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

        let result = gpu_knn_cagra(&dev, &data, n_obs, n_dims, n_neighbors).unwrap();

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

        // Cluster separation: points in cluster A should mostly find
        // neighbors in cluster A
        for i in 0..n_per_cluster {
            let same_cluster = (0..n_neighbors)
                .filter(|&j| (result.indices[i * n_neighbors + j] as usize) < n_per_cluster)
                .count();
            assert!(
                same_cluster >= 4,
                "point {i} has only {same_cluster}/{n_neighbors} neighbors in same cluster"
            );
        }
    }
}
