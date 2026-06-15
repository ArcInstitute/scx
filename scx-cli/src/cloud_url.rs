// Shared cloud-URL / exploded-directory detection used by the I/O
// subcommands (`query`, `info`, …). Centralizes the recognised URL scheme
// list so every verb routes the same way.

use std::path::Path;

/// Return true if `source` should be opened via `scx_cloud::open_cloud`.
///
/// Routes to the cloud path when:
///   - `source` carries a recognised URL scheme (`s3`, `gs`, `az`,
///     `azure`, `http`, `https`, `file`), or
///   - `source` is an existing local **directory** (typically an
///     exploded `.scxd/`), which `scx_cloud::open_cloud` handles via
///     the `LocalFileSystem` backend.
///
/// Plain regular files fall through to the local mmap path.
pub fn is_cloud_url(source: &str) -> bool {
    has_cloud_scheme(source) || Path::new(source).is_dir()
}

/// Return true if `source` carries a recognised remote URL scheme.
///
/// Narrower than [`is_cloud_url`]: this returns `false` for local
/// directories, which are routed through `LocalFileSystem` but are not
/// actually remote. Callers that need to gate behaviour on "round-trip
/// cost" (e.g. whether peeking a shard header is cheap) want this
/// helper, not `is_cloud_url`.
pub fn has_cloud_scheme(source: &str) -> bool {
    source.split_once("://").is_some_and(|(scheme, rest)| {
        !rest.is_empty()
            && matches!(
                scheme.to_ascii_lowercase().as_str(),
                "s3" | "gs" | "az" | "azure" | "http" | "https" | "file"
            )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_cloud_scheme_matches_known_schemes() {
        assert!(has_cloud_scheme("gs://bucket/path"));
        assert!(has_cloud_scheme("s3://bucket/path"));
        assert!(has_cloud_scheme("az://acct/container"));
        assert!(has_cloud_scheme("azure://acct/container"));
        assert!(has_cloud_scheme("http://example.com/data.scx"));
        assert!(has_cloud_scheme("https://example.com/data.scx"));
        assert!(has_cloud_scheme("file:///tmp/data.scx"));
        assert!(has_cloud_scheme("GS://Bucket/Path")); // case-insensitive
    }

    #[test]
    fn has_cloud_scheme_rejects_local_paths_and_dirs() {
        assert!(!has_cloud_scheme("/tmp/data.scx"));
        assert!(!has_cloud_scheme("./local.scxd"));
        assert!(!has_cloud_scheme("relative/path"));
        // Local directory paths must NOT trigger the cloud-scheme branch
        // (this is the Fix 1 regression: previously is_cloud_url returned
        // true for any local directory, forcing Float32 output even when
        // the source shards were integer-encoded).
        assert!(!has_cloud_scheme("/var/data/atlas.scxd"));
        // Empty / scheme-only inputs.
        assert!(!has_cloud_scheme("gs://"));
        assert!(!has_cloud_scheme("://"));
    }
}
