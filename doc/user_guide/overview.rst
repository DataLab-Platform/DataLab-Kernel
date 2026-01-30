.. _overview:

Overview
========

What is DataLab-Kernel?
-----------------------

**DataLab-Kernel** is a Jupyter kernel designed for scientific signal and image
processing workflows. Built on `Xeus-Python <https://github.com/jupyter-xeus/xeus-python>`_,
a lightweight and efficient Python kernel for Jupyter, it provides a bridge between
interactive notebook computing and the DataLab application, enabling:

- **Reproducible analysis** in Jupyter notebooks
- **Seamless integration** with DataLab's GUI-based processing
- **Flexible workflows** that work with or without DataLab
- **Cross-platform compatibility** with native Jupyter and JupyterLite (browser-based)

DataLab-Kernel leverages `Sigima <https://sigima.readthedocs.io>`_,
the computation engine that powers DataLab.


Architecture
------------

The kernel follows a clean architectural pattern with three main components:

.. code-block:: text

    ┌─────────────────────────────────────────────────────────┐
    │                    Jupyter Notebook                     │
    │                                                         │
    │  from datalab_kernel import Workspace, Plotter          │
    │  workspace = Workspace()                                │
    │  plotter = Plotter(workspace)                           │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │                     Workspace API                       │
    │                                                         │
    │  • add(), get(), remove(), rename()                     │
    │  • list(), exists(), clear()                            │
    │  • save(), load() (HDF5 persistence)                    │
    │  • calc() (live mode only)                              │
    │  • resync() (switch to live mode)                       │
    └─────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌─────────────────────┐       ┌─────────────────────┐
    │  StandaloneBackend  │       │    LiveBackend      │
    │                     │       │                     │
    │  • Local storage    │       │  • RemoteProxy      │
    │  • HDF5 files       │       │  • XML-RPC to DL    │
    │  • No DataLab req.  │       │  • Full DL access   │
    └─────────────────────┘       └─────────────────────┘


Operating Modes
---------------

Standalone Mode
^^^^^^^^^^^^^^^

When DataLab is not running, the kernel operates independently:

- Objects stored in local memory
- Full signal/image creation and manipulation via Sigima
- Persistence to/from HDF5 files
- Visualization with matplotlib

This mode is ideal for:

- Offline data analysis
- Reproducible research notebooks
- Teaching and demonstrations
- CI/CD pipelines

Live Mode
^^^^^^^^^

When DataLab is running, the kernel connects via XML-RPC:

- Objects synchronized with DataLab panels
- Access to DataLab's full processing pipeline via ``calc()``
- Direct proxy access for advanced operations
- Changes visible in real-time in DataLab GUI

This mode enables:

- Interactive exploration with GUI feedback
- Access to DataLab's extensive processing library
- Hybrid workflows (notebook + GUI)


The Workspace
-------------

The ``Workspace`` class is the central API:

.. code-block:: python

    from datalab_kernel import Workspace

    # Auto-detect mode (connects to DataLab if available)
    workspace = Workspace()

    # Check mode
    print(workspace.mode)  # WorkspaceMode.STANDALONE or LIVE

    # Standard operations work in both modes
    workspace.add("name", object)
    obj = workspace.get("name")
    workspace.remove("name")
    workspace.rename("old", "new")
    names = workspace.list()
    exists = workspace.exists("name")
    workspace.clear()

    # Persistence
    workspace.save("file.h5")
    workspace.load("file.h5")

    # Live mode only
    workspace.calc("function_name", params)
    workspace.proxy  # Direct RemoteProxy access


The Plotter
-----------

The ``Plotter`` class provides visualization:

.. code-block:: python

    from datalab_kernel import Plotter

    plotter = Plotter(workspace)

    # Plot by name
    plotter.plot("my_signal")

    # Custom options
    plotter.plot("my_image", cmap="viridis", colorbar=True)


Data Objects
------------

DataLab-Kernel re-exports Sigima's data objects:

**SignalObj** - 1D data with X/Y arrays, error bars, metadata

.. code-block:: python

    from sigima import create_signal

    signal = create_signal(
        "My Signal",
        x_array, y_array,
        units=("s", "V"),
        labels=("Time", "Voltage"),
    )

**ImageObj** - 2D data with metadata, coordinate system

.. code-block:: python

    from sigima import create_image

    image = create_image(
        "My Image",
        data_2d,
        units=("mm", "mm", "counts"),
        labels=("X", "Y", "Intensity"),
    )


Mode Transparency
-----------------

A key design goal is **mode transparency**: the same code works in both modes.

.. code-block:: python

    # This code works identically in standalone or live mode
    workspace = Workspace()

    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    signal = create_signal("Sine", x, y)

    workspace.add("sine", signal)
    retrieved = workspace.get("sine")

    plotter = Plotter(workspace)
    plotter.plot("sine")

    workspace.save("analysis.h5")

The only difference is that in live mode:

- Objects appear in DataLab's GUI
- ``calc()`` method is available
- ``proxy`` attribute provides direct DataLab access
