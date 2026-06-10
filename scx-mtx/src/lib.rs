//! scx-mtx: Matrix Market (MTX) I/O for SCX.
//!
//! Read and write Cell Ranger–style MTX directories:
//!
//! ```text
//! filtered_feature_bc_matrix/
//! ├── matrix.mtx[.gz]
//! ├── barcodes.tsv[.gz]
//! └── features.tsv[.gz]   (or genes.tsv[.gz])
//! ```

mod convert;
mod error;
mod read;
mod write;

pub use convert::mtx_to_scx;
pub use error::MtxError;
pub use read::{read_mtx_directory, MtxData, MtxOrientation};
pub use write::write_scx_to_mtx;
