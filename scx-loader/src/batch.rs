use std::collections::HashMap;
use std::sync::Arc;

/// Observation metadata column types stored in each batch.
#[derive(Debug, Clone)]
pub enum ObsColumn {
    /// Integer column (i64 values).
    Int64(Vec<i64>),
    /// Float column (f64 values).
    Float64(Vec<f64>),
    /// Categorical column: (encoded integer codes, category label strings).
    ///
    /// The category list is the file-global, stable set shared across every
    /// batch and epoch, so it is held behind an `Arc` — each batch clones only
    /// a refcount bump rather than re-allocating the (potentially `O(n_obs)`,
    /// e.g. for `cell_id`) string vector.
    Categorical(Vec<u32>, Arc<[String]>),
}

/// A training mini-batch ready for GPU transfer.
///
/// Produced by the decode stage (Stage 2) and consumed by the Python/PyTorch
/// iterator (Stage 3). The dense expression matrix `x` is stored in
/// contiguous row-major order.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Dense expression matrix [batch_size × n_genes], contiguous row-major.
    /// Pinned if CUDA is available, heap-allocated otherwise.
    pub x: Vec<f32>,
    /// Shape: (n_rows, n_genes).
    pub x_shape: (usize, usize),
    /// Observation metadata columns, one entry per requested column.
    pub obs: HashMap<String, ObsColumn>,
    /// Global cell indices for reproducibility/debugging.
    pub cell_indices: Vec<u64>,
}

impl Batch {
    /// Allocate a zero-filled batch (used for pre-allocation in the ring buffer).
    pub fn zeros(n_rows: usize, n_genes: usize) -> Self {
        Batch {
            x: vec![0.0f32; n_rows * n_genes],
            x_shape: (n_rows, n_genes),
            obs: HashMap::new(),
            cell_indices: Vec::new(),
        }
    }

    /// Zero-fill `x` without reallocating. Used for ring buffer reuse.
    pub fn reset(&mut self) {
        self.x.fill(0.0);
        self.obs.clear();
        self.cell_indices.clear();
    }

    /// Number of rows in this batch.
    pub fn n_rows(&self) -> usize {
        self.x_shape.0
    }

    /// Number of genes (columns) in this batch.
    pub fn n_genes(&self) -> usize {
        self.x_shape.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_zeros_correct_shape() {
        let batch = Batch::zeros(128, 2000);
        assert_eq!(batch.n_rows(), 128);
        assert_eq!(batch.n_genes(), 2000);
        assert_eq!(batch.x.len(), 128 * 2000);
        assert!(batch.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_batch_zeros_empty() {
        let batch = Batch::zeros(0, 0);
        assert_eq!(batch.n_rows(), 0);
        assert_eq!(batch.n_genes(), 0);
        assert!(batch.x.is_empty());
    }

    #[test]
    fn test_batch_reset_zeros_data() {
        let mut batch = Batch::zeros(4, 3);
        // Set some non-zero values
        batch.x[0] = 1.0;
        batch.x[5] = 2.5;
        batch.x[11] = 3.0;
        batch.cell_indices = vec![10, 20, 30, 40];
        batch
            .obs
            .insert("test".to_string(), ObsColumn::Int64(vec![1, 2, 3, 4]));

        batch.reset();

        // x should be zeroed
        assert!(batch.x.iter().all(|&v| v == 0.0));
        // Shape should be unchanged
        assert_eq!(batch.n_rows(), 4);
        assert_eq!(batch.n_genes(), 3);
        assert_eq!(batch.x.len(), 12);
        // obs and cell_indices should be cleared
        assert!(batch.obs.is_empty());
        assert!(batch.cell_indices.is_empty());
    }

    #[test]
    fn test_obs_column_int64() {
        let col = ObsColumn::Int64(vec![1, 2, 3]);
        if let ObsColumn::Int64(values) = col {
            assert_eq!(values, vec![1, 2, 3]);
        } else {
            panic!("expected Int64 variant");
        }
    }

    #[test]
    fn test_obs_column_float64() {
        let col = ObsColumn::Float64(vec![1.0, 2.5, 3.7]);
        if let ObsColumn::Float64(values) = col {
            assert_eq!(values, vec![1.0, 2.5, 3.7]);
        } else {
            panic!("expected Float64 variant");
        }
    }

    #[test]
    fn test_obs_column_categorical() {
        let codes = vec![0, 1, 0, 2];
        let categories = vec![
            "cat_a".to_string(),
            "cat_b".to_string(),
            "cat_c".to_string(),
        ];
        let col = ObsColumn::Categorical(codes.clone(), categories.clone().into());
        if let ObsColumn::Categorical(c, cats) = col {
            assert_eq!(c, codes);
            assert_eq!(&cats[..], categories.as_slice());
        } else {
            panic!("expected Categorical variant");
        }
    }
}
