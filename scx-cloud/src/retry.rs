//! Cloud read resilience: the single timeout + retry + backoff implementation
//! for SCX, applied once at the object-store boundary.
//!
//! [`RetryingStore`] is an [`ObjectStore`] decorator installed by
//! [`crate::backend::create_backend`]. It wraps the underlying store so that
//! *every* consumer — `pull`, the cloud query path (`CloudReader`), metadata
//! reads, predicate-index and deletion-vector reads — inherits one identical,
//! configurable per-request deadline and retry budget (finding LC1). This
//! removes the prior pull-vs-query resilience divergence and the hazard where a
//! single stalled GET hung the whole parallel decode with no deadline.
//!
//! Only [`ObjectStore::get_opts`] (plus `head`) is wrapped: in `object_store`
//! 0.12 `get`, `get_range`, and `get_ranges` all route through `get_opts`, so
//! decorating it covers every read path. `get_opts` collects the full response
//! body *under the deadline* and hands back a buffered [`GetResult`], so the
//! timeout covers body streaming (not just connection establishment) and the
//! caller's subsequent `.bytes()` merely drains memory.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{BoxStream, StreamExt};
use object_store::path::Path as ObjPath;
use object_store::{
    GetOptions, GetResult, GetResultPayload, ListResult, MultipartUpload, ObjectMeta, ObjectStore,
    PutMultipartOptions, PutOptions, PutPayload, PutResult,
};

/// Retry policy applied to every individual cloud read.
///
/// Each request is wrapped in a [`tokio::time::timeout`] and
/// application-classified transient failures are retried with exponential
/// backoff + jitter. Since [`RetryingStore`] is the sole retry authority,
/// `object_store`'s own inner retry is disabled at backend construction (see
/// [`crate::backend::create_backend_with_retry`]) so there is no
/// `outer × inner` compounding — the effective attempt count is exactly
/// `max_retries + 1`.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Number of retry attempts on top of the first try. Total attempts
    /// = `max_retries + 1`. Default 3.
    pub max_retries: usize,
    /// Initial backoff delay before retry attempt 1. Default 500 ms.
    pub base_delay: Duration,
    /// Cap on backoff delay; the exponential schedule is clamped here.
    /// Default 30 s.
    pub max_delay: Duration,
    /// Jitter amplitude as a fraction of the computed delay (e.g. 0.1
    /// means ±10%). Default 0.1.
    pub jitter_factor: f64,
    /// Per-request hard wall-clock timeout. A timed-out request is
    /// retried subject to `max_retries`; final exhaustion surfaces as
    /// [`crate::CloudError::Timeout`]. Default 120 s.
    pub request_timeout: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(500),
            max_delay: Duration::from_secs(30),
            jitter_factor: 0.1,
            request_timeout: Duration::from_secs(120),
        }
    }
}

impl RetryConfig {
    /// Retry policy that disables both outer retries and timeout
    /// enforcement. Use in tests or for fail-fast call sites.
    pub fn disabled() -> Self {
        Self {
            max_retries: 0,
            base_delay: Duration::from_millis(0),
            max_delay: Duration::from_millis(0),
            jitter_factor: 0.0,
            request_timeout: Duration::from_secs(60 * 60 * 24),
        }
    }
}

/// Terminal failure of the retry loop, carried as the boxed `source` of an
/// [`object_store::Error::Generic`] so the rich, structured SCX error survives
/// the `ObjectStore` trait boundary. [`crate::CloudError::from_store_error`]
/// downcasts it back into [`crate::CloudError::Timeout`] /
/// [`crate::CloudError::DownloadFailed`].
#[derive(Debug)]
pub(crate) enum RetryError {
    /// Per-request deadline exhausted on every attempt.
    Timeout {
        path: String,
        duration: Duration,
        last_error: Option<String>,
    },
    /// Retryable error budget exhausted. `message` already embeds the path.
    DownloadFailed { retries: usize, message: String },
}

impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetryError::Timeout {
                path,
                duration,
                last_error,
            } => {
                write!(f, "request timed out after {duration:?}: {path}")?;
                if let Some(m) = last_error {
                    write!(f, " (last error: {m})")?;
                }
                Ok(())
            }
            RetryError::DownloadFailed { retries, message } => {
                write!(f, "download failed after {retries} retries: {message}")
            }
        }
    }
}

impl std::error::Error for RetryError {}

fn retry_error_to_store(e: RetryError) -> object_store::Error {
    object_store::Error::Generic {
        store: "RetryingStore",
        source: Box::new(e),
    }
}

/// Classify an `object_store::Error` as retryable (transient) or
/// permanent. Conservative: only the explicit transient signals from
/// `object_store` plus our own heuristic substring matches against
/// throttle / 5xx / connection-reset wording are considered retryable.
pub(crate) fn is_retryable(err: &object_store::Error) -> bool {
    use object_store::Error;
    match err {
        // Definite permanent failures.
        Error::NotFound { .. } | Error::AlreadyExists { .. } | Error::NotModified { .. } => false,
        // `object_store` has internal classification — when in doubt
        // (Generic, JoinError, etc.) we treat the error as retryable so
        // the outer layer gets a chance.
        _ => {
            let msg = format!("{err}");
            let lower = msg.to_ascii_lowercase();
            lower.contains("timed out")
                || lower.contains("timeout")
                || lower.contains("connection reset")
                || lower.contains("connection refused")
                || lower.contains("connection closed")
                || lower.contains("broken pipe")
                || lower.contains("rate limit")
                || lower.contains("throttle")
                || lower.contains("throttled")
                || lower.contains("503")
                || lower.contains("502")
                || lower.contains("500")
                || lower.contains("504")
                || lower.contains("server error")
                || lower.contains("temporarily unavailable")
                || matches!(
                    err,
                    Error::Generic { .. }
                        | Error::JoinError { .. }
                        | Error::UnknownConfigurationKey { .. }
                )
        }
    }
}

/// Compute the exponential-backoff delay for a given attempt number.
///
/// Delay grows as `base * 2^(attempt-1)`, clamped to `max_delay`, then
/// multiplied by `1 ± jitter_factor` sampled uniformly from
/// `rand::thread_rng()`. Concurrent clients sampling independently is
/// what actually breaks thundering-herd retry spikes.
pub(crate) fn backoff_delay(cfg: &RetryConfig, attempt: usize) -> Duration {
    let exp = (attempt as u32).saturating_sub(1).min(20);
    let raw = cfg.base_delay.saturating_mul(1u32 << exp);
    let bounded = std::cmp::min(raw, cfg.max_delay);
    let jitter = if cfg.jitter_factor > 0.0 {
        use rand::Rng;
        rand::thread_rng().gen_range(-cfg.jitter_factor..=cfg.jitter_factor)
    } else {
        0.0
    };
    let nanos = bounded.as_nanos() as f64 * (1.0 + jitter);
    let nanos = nanos.max(0.0) as u64;
    Duration::from_nanos(nanos)
}

/// An [`ObjectStore`] decorator that applies [`RetryConfig`] (per-request
/// timeout + classified retry/backoff) to every read.
///
/// Installed once at the backend boundary so all consumers share one
/// resilience implementation (LC1).
#[derive(Debug)]
pub(crate) struct RetryingStore {
    inner: Arc<dyn ObjectStore>,
    cfg: RetryConfig,
}

impl RetryingStore {
    pub(crate) fn new(inner: Arc<dyn ObjectStore>, cfg: RetryConfig) -> Self {
        Self { inner, cfg }
    }

    /// The single retry/timeout loop. `op` is re-invoked (fresh future) per
    /// attempt; on terminal failure a [`RetryError`] is boxed into an
    /// `object_store::Error::Generic`.
    async fn retry_read<T, F, Fut>(&self, path: &ObjPath, op: F) -> object_store::Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = object_store::Result<T>>,
    {
        let cfg = &self.cfg;
        let mut attempt: usize = 0;
        let mut last_msg: Option<String> = None;
        loop {
            let started = Instant::now();
            match tokio::time::timeout(cfg.request_timeout, op()).await {
                Ok(Ok(v)) => return Ok(v),
                Ok(Err(e)) => {
                    if attempt < cfg.max_retries && is_retryable(&e) {
                        last_msg = Some(format!("{e}"));
                        attempt += 1;
                        tokio::time::sleep(backoff_delay(cfg, attempt)).await;
                        continue;
                    }
                    if attempt >= cfg.max_retries && is_retryable(&e) {
                        return Err(retry_error_to_store(RetryError::DownloadFailed {
                            retries: attempt,
                            message: format!("{path}: {e}"),
                        }));
                    }
                    // Permanent error — surface unchanged.
                    return Err(e);
                }
                Err(_) => {
                    if attempt < cfg.max_retries {
                        last_msg = Some(format!("timeout after {:?}", started.elapsed()));
                        attempt += 1;
                        tokio::time::sleep(backoff_delay(cfg, attempt)).await;
                        continue;
                    }
                    return Err(retry_error_to_store(RetryError::Timeout {
                        path: path.to_string(),
                        duration: cfg.request_timeout,
                        last_error: last_msg,
                    }));
                }
            }
        }
    }
}

impl std::fmt::Display for RetryingStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RetryingStore({})", self.inner)
    }
}

#[async_trait]
impl ObjectStore for RetryingStore {
    async fn get_opts(
        &self,
        location: &ObjPath,
        options: GetOptions,
    ) -> object_store::Result<GetResult> {
        self.retry_read(location, || {
            let inner = self.inner.clone();
            let loc = location.clone();
            let options = options.clone();
            async move {
                let r = inner.get_opts(&loc, options).await?;
                // Capture metadata before consuming the body stream, then
                // collect the full body under the deadline and hand back a
                // buffered result so the timeout covers body transfer.
                let meta = r.meta.clone();
                let range = r.range.clone();
                let attributes = r.attributes.clone();
                let bytes = r.bytes().await?;
                Ok(GetResult {
                    payload: GetResultPayload::Stream(
                        futures::stream::once(async move { Ok(bytes) }).boxed(),
                    ),
                    meta,
                    range,
                    attributes,
                })
            }
        })
        .await
    }

    async fn head(&self, location: &ObjPath) -> object_store::Result<ObjectMeta> {
        self.retry_read(location, || {
            let inner = self.inner.clone();
            let loc = location.clone();
            async move { inner.head(&loc).await }
        })
        .await
    }

    // --- Non-read methods delegate straight to the inner store. ---

    async fn put_opts(
        &self,
        location: &ObjPath,
        payload: PutPayload,
        opts: PutOptions,
    ) -> object_store::Result<PutResult> {
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &ObjPath,
        opts: PutMultipartOptions,
    ) -> object_store::Result<Box<dyn MultipartUpload>> {
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn delete(&self, location: &ObjPath) -> object_store::Result<()> {
        self.inner.delete(location).await
    }

    fn list(
        &self,
        prefix: Option<&ObjPath>,
    ) -> BoxStream<'static, object_store::Result<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(
        &self,
        prefix: Option<&ObjPath>,
    ) -> object_store::Result<ListResult> {
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &ObjPath, to: &ObjPath) -> object_store::Result<()> {
        self.inner.copy_if_not_exists(from, to).await
    }
}
