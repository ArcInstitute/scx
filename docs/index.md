# SCX Documentation

**SCX** (Sparse Cell eXpression System) is a purpose-built binary file format,
compression codec, query engine, and ML data loader for single-cell RNA-seq
data. It replaces AnnData/h5ad with a unified Rust-native stack providing
3–7× smaller files, fastest reads at census scale, 4–44× less memory, a
GPU-saturating training loader, lazy query engine, and Rust-native analysis
accelerators — with native bindings for Python and R, fully compatible with
the scverse ecosystem and Seurat v5.

This site bundles the architecture and design references that ship in the
repo with an auto-generated reference for the `pyscx` Python package.

For a runnable end-to-end example (install → convert → analyze), see the
[Quickstart](quickstart.md).

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Architecture & Design
:link: architecture
:link-type: doc

High-level architecture, crate layout, multithreading model, and sharding.
:::

:::{grid-item-card} Format & Codec
:link: format
:link-type: doc

The on-disk binary format and compression codec specifications.
:::

:::{grid-item-card} Python API
:link: python_api
:link-type: doc

Reference for the `pyscx` Python package — both the hand-written API guide
and autodoc for the Python-side helpers and integrations.
:::

:::{grid-item-card} Performance & Operations
:link: performance
:link-type: doc

Benchmarks, operational semantics (append/delete/compact), and test infrastructure.
:::

::::

```{toctree}
:caption: User Guide
:maxdepth: 2
:hidden:

quickstart
migrating-from-h5ad
training
tokenize
scanpy
pseudobulk_nb_glm
gpu-setup
cloud
operations
```

```{toctree}
:caption: Architecture & Design
:maxdepth: 2
:hidden:

architecture
multithreading
sharding
conventions
multimodal
```

```{toctree}
:caption: Format Specification
:maxdepth: 2
:hidden:

format
codec
```

```{toctree}
:caption: API Reference
:maxdepth: 2
:hidden:

api
python_api
```

```{toctree}
:caption: Performance & Testing
:maxdepth: 2
:hidden:

performance
testing
```
