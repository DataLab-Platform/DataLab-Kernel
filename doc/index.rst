DataLab-Kernel
==============

**DataLab-Kernel** is a standalone `Xeus-Python <https://github.com/jupyter-xeus/xeus-python>`_-based
Jupyter kernel for scientific data processing with optional live synchronization
to `DataLab <https://datalab-platform.com>`_.

It runs seamlessly in both **native Jupyter** (JupyterLab, Jupyter Notebook, VS Code)
and **JupyterLite** (browser-based, no server required), enabling flexible
notebook-based workflows for signal and image analysis with
`Sigima <https://sigima.readthedocs.io>`_.

.. code-block:: python

    from datalab_kernel import Workspace, Plotter, create_signal
    import numpy as np

    # Create workspace and plotter
    workspace = Workspace()
    plotter = Plotter(workspace)

    # Create and add signal
    x = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * x) * np.exp(-x / 5)
    signal = create_signal("Damped Sine", x, y)
    workspace.add("my_signal", signal)

    # Visualize
    plotter.plot("my_signal")

.. figure:: _static/DataLab-Banner.svg
    :align: center
    :width: 300 px
    :class: dark-light no-scaled-link

    Part of the `DataLab Platform <https://www.datalab-platform.com>`_.

.. only:: html and not latex

    .. grid:: 2 2 4 4
        :gutter: 1 2 3 4

        .. grid-item-card:: :octicon:`rocket;1em;sd-text-info`  User Guide
            :link: user_guide/index
            :link-type: doc

            Installation, overview, and features

        .. grid-item-card:: :octicon:`book;1em;sd-text-info`  API Reference
            :link: api/index
            :link-type: doc

            Reference documentation

        .. grid-item-card:: :octicon:`code;1em;sd-text-info`  Examples
            :link: examples/index
            :link-type: doc

            Notebooks and tutorials

        .. grid-item-card:: :octicon:`gear;1em;sd-text-info`  Contributing
            :link: contributing/index
            :link-type: doc

            Getting involved in the project


Key Features
------------

**Dual Operating Modes**
    - **Standalone mode**: Work independently with HDF5 persistence
    - **Live mode**: Synchronize with running DataLab instance via XML-RPC

**Unified API**
    Same code works in both modes - switch seamlessly between local notebooks
    and DataLab-connected workflows.

**Scientific Data Objects**
    Full access to Sigima's `SignalObj` and `ImageObj` with metadata,
    units, labels, and ROI support.

**Interactive Visualization**
    Built-in plotting with matplotlib integration.

**Reproducibility**
    HDF5-based workspace persistence ensures complete reproducibility.


.. toctree::
   :maxdepth: 2
   :hidden:

   user_guide/index
   api/index
   examples/index
   contributing/index
   requirements
