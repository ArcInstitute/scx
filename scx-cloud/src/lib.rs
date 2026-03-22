pub mod backend;
pub mod cloud_optimize;
pub mod error;
pub mod explode;
pub mod pack;

// Stub modules for future phases
// pub mod coalesce;
// pub mod pull;
// pub mod push;

pub use backend::{create_backend, parse_location, CloudLocation};
pub use cloud_optimize::cloud_optimize;
pub use error::{CloudError, Result};
pub use explode::explode;
pub use pack::pack;
