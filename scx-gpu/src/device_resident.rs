//! Device-resident handoff types for the GPU pipeline (V3 plan Phase 2.2).
//!
//! These wrap raw [`CudaSlice`] buffers + shape metadata so GPU accelerators
//! can hand GPU-resident results to one another without a host round-trip.
//! The first consumer (Phase 2.3) is the fused PCA → kNN path: GPU PCA produces
//! a [`DeviceEmbedding`] that `gpu_knn_cagra_device` reads directly as the CAGRA
//! dataset, replacing the previous embedding GPU→host→GPU trip.
//!
//! Each type owns its device buffers (so the embedding/graph outlives the
//! producing call) and exposes a `to_host` method matching the existing
//! download convention ([`GpuDevice::dtoh_copy`]). Buffers are tied to the
//! [`GpuDevice`] / stream they were allocated on; download with the same
//! device.

use cudarc::driver::safe::CudaSlice;

use crate::device::GpuDevice;
use crate::error::GpuError;
use crate::gpu_knn::GpuKnnResult;

/// Host CSR connectivity triplet `(indptr, indices, data)` — the layout
/// produced by `scx_accel::neighbors::compute_connectivities`.
pub type HostCsrTriplet = (Vec<i64>, Vec<i32>, Vec<f32>);

/// A device-resident dense embedding, **row-major contiguous**
/// `(n_obs × n_components)`.
///
/// Produced by the device-returning randomized PCA entry point
/// (`gpu_randomized_pca_device`) and consumed by `gpu_knn_cagra_device`. (The
/// covariance device PCA entry point was removed.) Row-major layout matches
/// what CAGRA expects for its
/// DLPack dataset tensor (strides `[n_components, 1]`) and what
/// `obsm["X_pca"]` stores on the host, so [`Self::to_host`] needs no transpose.
pub struct DeviceEmbedding {
    data: CudaSlice<f32>,
    n_obs: usize,
    n_components: usize,
}

impl DeviceEmbedding {
    /// Wrap a row-major `(n_obs × n_components)` device buffer.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::ShapeMismatch`] if `data.len() != n_obs * n_components`.
    pub fn new(data: CudaSlice<f32>, n_obs: usize, n_components: usize) -> Result<Self, GpuError> {
        if data.len() != n_obs * n_components {
            return Err(GpuError::ShapeMismatch {
                expected: format!(
                    "n_obs × n_components = {n_obs} × {n_components} = {}",
                    n_obs * n_components
                ),
                got: format!("{}", data.len()),
            });
        }
        Ok(Self {
            data,
            n_obs,
            n_components,
        })
    }

    /// Borrow the underlying row-major device buffer.
    pub fn data(&self) -> &CudaSlice<f32> {
        &self.data
    }

    /// `(n_obs, n_components)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.n_obs, self.n_components)
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Number of components (columns).
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Download to a host `Vec<f32>`, row-major `(n_obs × n_components)`.
    pub fn to_host(&self, dev: &GpuDevice) -> Result<Vec<f32>, GpuError> {
        dev.dtoh_copy(&self.data)
    }
}

/// A device-resident kNN result: raw CAGRA neighbor indices + distances kept on
/// the GPU.
///
/// Holds the search output **before** the self-hit filter and `sqrt`, exactly
/// as CAGRA writes it: `indices` are `u32` and `distances` are L2-**squared**,
/// both laid out `(n_obs × search_k)` row-major where `search_k = min(n_obs,
/// n_neighbors + 1)` (CAGRA may return self, so one extra slot is searched).
/// [`Self::to_host`] applies the self-filter + `sqrt` post-process, yielding
/// the host [`GpuKnnResult`].
pub struct DeviceKnnGraph {
    indices: CudaSlice<u32>,
    distances: CudaSlice<f32>,
    n_obs: usize,
    search_k: usize,
    n_neighbors: usize,
}

impl DeviceKnnGraph {
    /// Wrap raw CAGRA output buffers.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::ShapeMismatch`] if either buffer length differs from
    /// `n_obs * search_k`.
    pub fn new(
        indices: CudaSlice<u32>,
        distances: CudaSlice<f32>,
        n_obs: usize,
        search_k: usize,
        n_neighbors: usize,
    ) -> Result<Self, GpuError> {
        let expect = n_obs * search_k;
        if indices.len() != expect || distances.len() != expect {
            return Err(GpuError::ShapeMismatch {
                expected: format!("n_obs × search_k = {n_obs} × {search_k} = {expect}"),
                got: format!("indices={}, distances={}", indices.len(), distances.len()),
            });
        }
        Ok(Self {
            indices,
            distances,
            n_obs,
            search_k,
            n_neighbors,
        })
    }

    /// Borrow the raw device neighbor-index buffer (`u32`, `n_obs × search_k`).
    pub fn indices(&self) -> &CudaSlice<u32> {
        &self.indices
    }

    /// Borrow the raw device distance buffer (L2-squared, `n_obs × search_k`).
    pub fn distances(&self) -> &CudaSlice<f32> {
        &self.distances
    }

    /// Number of observations (rows).
    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    /// Requested neighbor count (post self-filter).
    pub fn n_neighbors(&self) -> usize {
        self.n_neighbors
    }

    /// Searched neighbor count (`min(n_obs, n_neighbors + 1)`, pre self-filter).
    pub fn search_k(&self) -> usize {
        self.search_k
    }

    /// Download + post-process into a host [`GpuKnnResult`].
    ///
    /// Converts `u32` → `i64`, drops the self-hit, takes `sqrt` of the
    /// L2-squared distances (Euclidean), and pads with self / `0.0` if fewer
    /// than `n_neighbors` non-self neighbors were returned.
    pub fn to_host(&self, dev: &GpuDevice) -> Result<GpuKnnResult, GpuError> {
        dev.synchronize()?;
        let neighbors_u32 = dev.dtoh_copy(&self.indices)?;
        let distances_f32 = dev.dtoh_copy(&self.distances)?;

        let n_obs = self.n_obs;
        let search_k = self.search_k;
        let n_neighbors = self.n_neighbors;

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
}
