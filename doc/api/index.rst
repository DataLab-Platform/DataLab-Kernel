.. _api:

API Reference
=============

The public API of DataLab-Kernel provides a simple interface for
scientific data processing in Jupyter notebooks.

.. list-table::
    :header-rows: 1
    :align: left

    * - Module
      - Purpose

    * - :mod:`datalab_kernel`
      - Main entry point with ``Workspace``, ``Plotter``, and ``WorkspaceMode``

    * - :mod:`datalab_kernel.objects`
      - Data objects (``SignalObj``, ``ImageObj``) and creation functions
        (``create_signal``, ``create_image``) - re-exported from Sigima

    * - :mod:`datalab_kernel.workspace`
      - Workspace implementation with backend abstraction


.. toctree::
   :maxdepth: 2
   :caption: API Modules:

   workspace
   objects
   plotter
