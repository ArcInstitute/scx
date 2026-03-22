pub mod backend;
pub mod error;

// Stub modules for future phases
// pub mod cloud_optimize;
// pub mod coalesce;
// pub mod explode;
// pub mod pack;
// pub mod pull;
// pub mod push;

pub use backend::{create_backend, parse_location, CloudLocation};
pub use error::{CloudError, Result};
