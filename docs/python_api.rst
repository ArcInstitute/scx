Python API (``pyscx``)
======================

Auto-generated reference for the ``pyscx`` Python package. Most symbols
are defined in the compiled Rust extension (PyO3); the docstrings shown
on each subpage are the Rust ``///`` doc comments that PyO3 surfaces as
Python docstrings at runtime.

Click any name in the tables below to jump to its dedicated page. For
prose-style narrative covering the same surface (with diagrams, worked
examples, and design rationale), see :doc:`api`.

.. currentmodule:: pyscx

I/O & file lifecycle
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   open
   validate
   from_anndata
   from_10x
   from_mtx
   from_mudata
   to_mtx

Mutating operations
-------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   append
   append_from_anndata
   compact
   merge
   mark_deleted
   rollback
   save_layer
   preprocess

Cloud operations (requires ``--features cloud``)
-------------------------------------------------

.. note::

   The RTD build compiles ``pyscx`` without ``--features cloud``, so these
   entries will have empty autodoc stubs on the rendered site.  For the
   prose-style reference, see :doc:`api` § Cloud operations.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pull
   push
   cloud_optimize
   explode
   pack
   open_cloud

Datasets & readers
------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   PyExperiment
   ScxBackedSparseDataset
   ScxBackedLayerDataset
   ScxLazyTransformedDataset

ML training datasets
--------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   TrainingDataset
   IndexPlanDataset

Multimodal
----------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   MultimodalTrainingDataset

Query pipeline
--------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   PyQueryPipeline
   PyQueryResult

Pure-Python helpers
-------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   iter_chunks

Analysis accelerators (``pyscx.accel``)
---------------------------------------

Rust-native accelerators that read SCX directly and write results back to
standard AnnData slots. All accept a ``device=`` parameter
(``"auto" | "cpu" | "gpu" | "gpu:N"``) where applicable.

Dimensionality reduction & embedding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: pyscx.accel

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pca
   neighbors
   umap
   leiden
   harmony_integrate
   compute_lisi

Preprocessing
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   normalize_total
   log1p
   highly_variable_genes
   filter_cells
   filter_genes
   subset_obs
   calculate_qc_metrics

Differential expression & pseudobulk
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   pseudobulk_means
   pseudobulk_dex
   rank_genes_groups
   rank_genes_groups_df

Perturbation evaluation metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   perturbation_metrics
   discrimination_score
   energy_distance
   energy_distance_details
   knockdown_efficiency
   clustering_agreement

Cluster-comparison scores
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   adjusted_mutual_info
   adjusted_rand_index
   normalized_mutual_info

GPU diagnostics
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   estimate_gpu_memory
   gpu_info

ML framework integrations
-------------------------

.. currentmodule:: pyscx.scx_integrations

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   scvi.ScxDataModule
