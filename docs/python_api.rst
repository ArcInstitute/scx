Python API (``pyscx``)
======================

Auto-generated reference for the ``pyscx`` Python package. Symbols with a
pure-Python wrapper in ``pyscx/__init__.py`` (``open``, ``from_h5ad``,
``obs_import``, …) show the **wrapper's** docstring — the canonical one, since
the wrapper's input coercions are part of the contract; their Rust ``///``
docs are two-line pointers back at it. Symbols with no wrapper
(``from_anndata``, ``merge``, ``compact``, …) show the Rust ``///`` doc
comments that PyO3 surfaces as Python docstrings at runtime.

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
   read
   write
   validate
   from_anndata
   from_h5ad
   from_10x
   from_mtx
   from_mudata
   from_h5mu
   to_mtx
   to_h5ad
   to_h5mu
   read_h5ad_metadata
   export_batches

Mutating operations
-------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   append
   append_from_anndata
   compact
   optimize
   merge
   mark_deleted
   rollback
   save_layer
   preprocess
   modify_metadata
   set_uns

External annotation import
--------------------------

In-place, key-joined landing of externally computed per-cell annotations
(doublet callers, CellBender, any CSV/DataFrame). See :doc:`operations`
§ External obs import / § External var import for the prose guides.

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   obs_import
   attach_obs_columns
   diagnose_obs_key
   var_import
   attach_var_columns
   diagnose_var_key
   doublet_import
   doublet_consensus
   doublet_tools
   doublet_profiles
   cellbender_import

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
   read_cloud
   CloudExperiment

Datasets & readers
------------------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   Experiment
   ScxBackedSparseDataset
   ScxBackedLayerDataset
   ScxBackedMuDataset
   ScxBackedMuModality
   ScxLazyTransformedDataset
   _ComparisonResult

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
   pflog_reconstruct

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

Column aggregates
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   col_sums
   col_nnz
   col_min
   col_max
   col_var

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
