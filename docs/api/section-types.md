# Section Types

> Part of the [SCX API reference](README.md).

29 section types (IDs 0–29, with ID 26 reserved) are defined in `scx-format/src/section.rs`:

```
ObsMetadata (0)        — Arrow IPC metadata for observations
ObsIndex (1)           — Arrow IPC index for observations
VarMetadata (2)        — Arrow IPC metadata for variables
VarIndex (3)           — Arrow IPC index for variables
CsrShard (4)           — Main expression matrix (row-major)
CscShard (5)           — Column-major (gene-major) sparse shard. Used
                         for column-axis analytical workloads (DE, HVG,
                         per-gene QC). Optional sidecar; written by default on
                         qualifying datasets via `scx convert` (`--csc auto`, or
                         `--csc always`), `pyscx.from_anndata` (`csc="auto"` /
                         `csc="always"`), or after the fact via `scx build-csc`
                         / `pyscx.build_csc`.
BitmapShard (6)        — Per-shard detection bitmap sidecar (gene →
                         local-row roaring bitmaps; `SCXB` magic).
                         Written by `scx convert --bitmap auto|always`
                         and the pyscx `bitmap="..."` kwarg; consumed
                         by `Experiment.detection_counts` /
                         `cells_expressing`. See
                         [docs/format.md § Detection Bitmap](../format.md#12-detection-bitmap-optional).
LayerCsrShard (7)      — Alternative expression layers
ObsmEmbedding (8)      — Embeddings (obsm)
ObspCsrShard (9)       — CSR-backed obs x obs graph. Written by
                         `ScxWriter::write_obsp_shard`, checked by
                         `scx validate --deep` against the v3 canonical-CSR
                         invariant, re-encoded by `scx optimize`, and
                         preserved by `scx upgrade` / `scx build-csc` —
                         build-csc never touches it, while upgrade
                         canonicalizes a non-canonical pre-v3 graph and
                         re-encodes it, carrying the source shard's own
                         minor extent rather than re-deriving one. No read
                         API materialises it: `read_obsp` / `list_obsp` — and
                         so `to_anndata`'s `obsp` — see only the COO forms,
                         ObspEmbedding (18) / ObspEmbeddingShard (22), which
                         is what every conversion path writes.
UnsBlob (10)           — Unstructured metadata (JSON)
Provenance (11)        — Operation history
DeletionVectors (12)   — Logical deletion tracking (Roaring Bitmap)
ObsPredicateIndex (13) — Obs predicate index for query pushdown
VarPredicateIndex (14) — Var predicate index for query pushdown
ModalityTable (15)     — v2; ordered list of named modalities (CITE-seq,
                         10x Multiome, …). See docs/format.md § 13.
LayerCscShard (16)     — v2; per-modality CSC sidecar for a named layer
                         (parallel to LayerCsrShard).
VarmEmbedding (17)     — Dense var embeddings (varm); same wire format
                         as ObsmEmbedding but indexed by var.
ObspEmbedding (18)     — Sparse obs×obs pairwise matrices (e.g.
                         kNN connectivities/distances). COO Arrow IPC
                         with schema metadata n_rows / n_cols; data is
                         stored as float32.
VarpEmbedding (19)     — Sparse var×var pairwise matrices. Same wire
                         format as ObspEmbedding.
ObsmEmbeddingShard (20)— Row-sharded obsm (section per shard × key).
VarmEmbeddingShard (21)— Row-sharded varm (mirror of 20).
ObspEmbeddingShard (22)— Row-sharded obsp (section per shard × key).
VarpEmbeddingShard (23)— Row-sharded varp (mirror of 22).
ObsMetadataShard (24)  — Row-sharded obs Arrow IPC. Produced by merge,
                         append, from_anndata, optimize --shard-obs and
                         every scx convert / from_h5ad / from_h5mu /
                         from_mtx ingest,
                         all on the same n_obs > shard_size boundary;
                         --shard-obs off|always overrides it on the
                         convert paths. Mutually exclusive with
                         ObsMetadata (0) in the same file.
VarMetadataShard (25)  — Row-sharded var Arrow IPC (mirror of 24). NOT
                         emitted by convert ingest at any n_vars — only by
                         merge, append and from_anndata.
(26)                   — RESERVED (formerly DecodeMetadataShard, removed;
                         random access now via the codec-agnostic row-group
                         BlockIndex, framing). Legacy files carrying id 26 are
                         skipped by the catalog reader.
RawCsrShard (27)       — Row-sharded `raw.X` CSR (anndata `.raw` layer),
                         parallel to CsrShard (4).
RawVarMetadata (28)    — Arrow IPC var metadata for the `.raw` layer
                         (mirror of VarMetadata (2)).
GroupIndex (29)        — Condition/label-grouped sharding sidecar (one per
                         file; `group_index` JSON). Written by
                         `scx sort --group-by`; consumed by the grouped-read
                         API. See docs/format.md § section ids.
```
