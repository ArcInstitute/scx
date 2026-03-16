pub mod catalog;
pub mod checksum;
pub mod error;
pub mod header;
pub mod provenance;
pub mod reader;
pub mod section;
pub mod shard;
pub mod writer;

pub use error::{Result, ScxError};
pub use header::{FileHeader, HEADER_SIZE, MAGIC};
pub use section::{align_to_8, SectionType};
