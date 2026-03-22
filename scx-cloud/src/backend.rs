//! Object-store backend abstraction: URL parsing and `ObjectStore` instantiation.
//!
//! Supports `gs://`, `s3://`, `az://`, and local filesystem paths.
//! Authentication relies on `object_store`'s built-in credential chains
//! (env vars, instance metadata, service accounts).

use object_store::local::LocalFileSystem;
use object_store::ObjectStore;
use url::Url;

use crate::error::{CloudError, Result};

/// Parsed cloud or local location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudLocation {
    Gcs { bucket: String, prefix: String },
    S3 { bucket: String, prefix: String },
    Azure { container: String, prefix: String },
    Local(std::path::PathBuf),
}

/// Parse a URL string into a [`CloudLocation`].
///
/// Supports:
/// - `gs://bucket/prefix/` — Google Cloud Storage
/// - `s3://bucket/prefix/` — Amazon S3
/// - `az://container/prefix/` — Azure Blob Storage
/// - Absolute or relative filesystem paths
pub fn parse_location(url: &str) -> Result<CloudLocation> {
    let trimmed = url.trim();
    if trimmed.is_empty() {
        return Err(CloudError::InvalidUrl("empty URL string".to_string()));
    }

    // Try parsing as a URL first
    if let Ok(parsed) = Url::parse(trimmed) {
        match parsed.scheme() {
            "gs" => {
                let bucket = parsed
                    .host_str()
                    .ok_or_else(|| CloudError::InvalidUrl(format!("missing bucket in: {url}")))?
                    .to_string();
                let prefix = parsed.path().trim_start_matches('/').to_string();
                Ok(CloudLocation::Gcs { bucket, prefix })
            }
            "s3" => {
                let bucket = parsed
                    .host_str()
                    .ok_or_else(|| CloudError::InvalidUrl(format!("missing bucket in: {url}")))?
                    .to_string();
                let prefix = parsed.path().trim_start_matches('/').to_string();
                Ok(CloudLocation::S3 { bucket, prefix })
            }
            "az" => {
                let container = parsed
                    .host_str()
                    .ok_or_else(|| {
                        CloudError::InvalidUrl(format!("missing container in: {url}"))
                    })?
                    .to_string();
                let prefix = parsed.path().trim_start_matches('/').to_string();
                Ok(CloudLocation::Azure { container, prefix })
            }
            "file" => {
                let path = parsed
                    .to_file_path()
                    .map_err(|_| CloudError::InvalidUrl(format!("invalid file URL: {url}")))?;
                Ok(CloudLocation::Local(path))
            }
            other => Err(CloudError::InvalidUrl(format!(
                "unsupported scheme '{other}' in: {url}"
            ))),
        }
    } else {
        // Not a valid URL — treat as local filesystem path
        Ok(CloudLocation::Local(std::path::PathBuf::from(trimmed)))
    }
}

/// Create an [`ObjectStore`] backend from a [`CloudLocation`].
///
/// Uses default credential chains:
/// - **GCS**: `GOOGLE_APPLICATION_CREDENTIALS` env var, or instance metadata on GCE/GKE
/// - **S3**: `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`, or instance profile on EC2
/// - **Azure**: `AZURE_STORAGE_ACCOUNT`/`AZURE_STORAGE_KEY`, or managed identity
pub async fn create_backend(location: &CloudLocation) -> Result<Box<dyn ObjectStore>> {
    match location {
        CloudLocation::Gcs { bucket, .. } => {
            let store = object_store::gcp::GoogleCloudStorageBuilder::from_env()
                .with_bucket_name(bucket)
                .build()?;
            Ok(Box::new(store))
        }
        CloudLocation::S3 { bucket, .. } => {
            let store = object_store::aws::AmazonS3Builder::from_env()
                .with_bucket_name(bucket)
                .build()?;
            Ok(Box::new(store))
        }
        CloudLocation::Azure { container, .. } => {
            let mut builder = object_store::azure::MicrosoftAzureBuilder::new()
                .with_container_name(container);
            if let Ok(account) = std::env::var("AZURE_STORAGE_ACCOUNT") {
                builder = builder.with_account(account);
            }
            if let Ok(key) = std::env::var("AZURE_STORAGE_KEY") {
                builder = builder.with_access_key(key);
            }
            let store = builder.build()?;
            Ok(Box::new(store))
        }
        CloudLocation::Local(path) => {
            let root = if path.is_file() {
                path.parent().unwrap_or(path)
            } else {
                path.as_path()
            };
            let store = LocalFileSystem::new_with_prefix(root)?;
            Ok(Box::new(store))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_gcs_url() {
        let loc = parse_location("gs://my-bucket/some/prefix/").unwrap();
        assert_eq!(
            loc,
            CloudLocation::Gcs {
                bucket: "my-bucket".to_string(),
                prefix: "some/prefix/".to_string(),
            }
        );
    }

    #[test]
    fn parse_gcs_url_no_prefix() {
        let loc = parse_location("gs://my-bucket").unwrap();
        assert_eq!(
            loc,
            CloudLocation::Gcs {
                bucket: "my-bucket".to_string(),
                prefix: "".to_string(),
            }
        );
    }

    #[test]
    fn parse_s3_url() {
        let loc = parse_location("s3://data-bucket/experiments/exp1.scxd/").unwrap();
        assert_eq!(
            loc,
            CloudLocation::S3 {
                bucket: "data-bucket".to_string(),
                prefix: "experiments/exp1.scxd/".to_string(),
            }
        );
    }

    #[test]
    fn parse_azure_url() {
        let loc = parse_location("az://mycontainer/path/to/data").unwrap();
        assert_eq!(
            loc,
            CloudLocation::Azure {
                container: "mycontainer".to_string(),
                prefix: "path/to/data".to_string(),
            }
        );
    }

    #[test]
    fn parse_local_absolute_path() {
        let loc = parse_location("/tmp/experiment.scx").unwrap();
        assert_eq!(
            loc,
            CloudLocation::Local(std::path::PathBuf::from("/tmp/experiment.scx"))
        );
    }

    #[test]
    fn parse_local_relative_path() {
        let loc = parse_location("./data/test.scx").unwrap();
        assert_eq!(
            loc,
            CloudLocation::Local(std::path::PathBuf::from("./data/test.scx"))
        );
    }

    #[test]
    fn parse_empty_url_fails() {
        let err = parse_location("").unwrap_err();
        assert!(matches!(err, CloudError::InvalidUrl(_)));
    }

    #[test]
    fn parse_unsupported_scheme_fails() {
        let err = parse_location("ftp://host/path").unwrap_err();
        assert!(matches!(err, CloudError::InvalidUrl(_)));
    }

    #[test]
    fn parse_file_url() {
        let loc = parse_location("file:///tmp/data.scx").unwrap();
        assert_eq!(
            loc,
            CloudLocation::Local(std::path::PathBuf::from("/tmp/data.scx"))
        );
    }

    #[tokio::test]
    async fn create_local_backend() {
        let dir = tempfile::tempdir().unwrap();
        let loc = CloudLocation::Local(dir.path().to_path_buf());
        let backend = create_backend(&loc).await;
        assert!(backend.is_ok(), "local backend creation should succeed");
    }

    #[tokio::test]
    async fn create_gcs_backend() {
        // GCS backend creation succeeds (builder + build) even without credentials.
        // Actual I/O would fail at request time, but instantiation is fine.
        let loc = CloudLocation::Gcs {
            bucket: "test-bucket".to_string(),
            prefix: "prefix/".to_string(),
        };
        let backend = create_backend(&loc).await;
        assert!(
            backend.is_ok(),
            "GCS backend instantiation should succeed without credentials"
        );
    }

    #[tokio::test]
    async fn create_s3_backend() {
        let loc = CloudLocation::S3 {
            bucket: "test-bucket".to_string(),
            prefix: "prefix/".to_string(),
        };
        let backend = create_backend(&loc).await;
        assert!(
            backend.is_ok(),
            "S3 backend instantiation should succeed without credentials"
        );
    }

    #[tokio::test]
    async fn create_azure_backend() {
        let loc = CloudLocation::Azure {
            container: "test-container".to_string(),
            prefix: "prefix/".to_string(),
        };
        let backend = create_backend(&loc).await;
        // Azure builder requires an account name to build successfully.
        // Without AZURE_STORAGE_ACCOUNT env var, build may fail.
        if std::env::var("AZURE_STORAGE_ACCOUNT").is_ok() {
            assert!(
                backend.is_ok(),
                "Azure backend instantiation should succeed with credentials: {:?}",
                backend.err()
            );
        }
    }
}
