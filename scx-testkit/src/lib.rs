//! Test-only harness for asserting that a refactor did not change what SCX
//! writes.
//!
//! ## Why this is not "assert the file's BLAKE3 against a checked-in constant"
//!
//! That is the obvious harness and it does not work here. Roughly twenty write
//! paths — every `scx-ops` mutation, both `scx-convert` directions, `scx-cli
//! subset`, `pyscx.from_anndata` — stamp `SystemTime::now()` into the
//! `Provenance` section, and `FileHeader::file_checksum` covers it. Two runs of
//! the same op a second apart produce two different files. Only `scx_ops::sort`
//! takes an injectable timestamp.
//!
//! The tree already knows this: `scx-convert`'s
//! `parallel_streaming_byte_identical_to_sequential` compares per-shard
//! checksums and says so in a comment, and `phase1_streaming_dense_determinism`
//! gave up and compares `bytes_a.len() == bytes_b.len()` — an assertion that
//! passes for almost any wrong answer.
//!
//! So [`digest`] hashes **per catalog section** and excludes `Provenance` by
//! construction. Everything else about the file is pinned, the result is stable
//! across runs, and a checked-in golden works after all. A mismatch names the
//! section rather than reporting that two files differ.
//!
//! ## Using it
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use scx_testkit::digest::{assert_digests_eq, digest_file, Strictness};
//!
//! # let (before, after) = (std::path::Path::new("a"), std::path::Path::new("b"));
//! let a = digest_file(before, Strictness::Content)?;
//! let b = digest_file(after, Strictness::Content)?;
//! assert_digests_eq(&a, &b);
//! # Ok(())
//! # }
//! ```
//!
//! For a contract that must survive across commits, pin it instead:
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let out = std::path::Path::new("out.scx");
//! scx_testkit::digest::assert_matches_golden(
//!     out,
//!     std::path::Path::new("tests/golden/compact.digest.json"),
//!     scx_testkit::digest::Strictness::Content,
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! `SCX_TESTKIT_BLESS=1` rewrites the golden instead of asserting against it.
//! Review the diff: a blessed golden is a claim that the change was intended.
//!
//! ## Choosing a fixture
//!
//! Use [`fixtures::mixed_codec_file`] unless you need something else. An
//! identity claim tested on the usual `CodecId::None` / `Uint8` fixture is
//! close to vacuous — it exercises no codec at all. That fixture carries one
//! unframed, one row-group-framed integer, and one float (`Pcodec`) shard, so a
//! single digest covers all three encoder paths.

pub mod digest;
pub mod fixtures;
