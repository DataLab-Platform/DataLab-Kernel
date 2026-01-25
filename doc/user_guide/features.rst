.. _features:

Features
========

This page describes the key features of DataLab-Kernel.

Dual Operating Modes
--------------------

**Standalone Mode**

- Works without DataLab installed
- Local object storage in memory
- HDF5 persistence for reproducibility
- Full Sigima processing capabilities

**Live Mode**

- Automatic connection to running DataLab
- Real-time object synchronization
- Access to DataLab's processing pipeline
- GUI feedback for all operations


Connection Methods
------------------

**Web API** (recommended)

Connect via HTTP/JSON using environment variables:

.. code-block:: bash

    export DATALAB_WORKSPACE_URL=http://127.0.0.1:8080
    export DATALAB_WORKSPACE_TOKEN=<your-token>

Features:

- WASM/Pyodide compatible
- Efficient NPZ binary format for data transfer
- Bearer token authentication
- Modern REST API

**XML-RPC** (legacy)

Automatic connection when DataLab is running with remote control enabled.
No configuration required.

**Environment Variables**

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``DATALAB_WORKSPACE_URL``
     - Web API server URL
     - (none)
   * - ``DATALAB_WORKSPACE_TOKEN``
     - Web API authentication token
     - (none)
   * - ``DATALAB_KERNEL_MODE``
     - Force mode: ``auto``, ``live``, ``standalone``
     - ``auto``


Unified Workspace API
---------------------

A single, consistent API regardless of mode:

.. code-block:: python

    workspace.add(name, obj)      # Add object
    workspace.get(name)           # Retrieve object
    workspace.remove(name)        # Remove object
    workspace.rename(old, new)    # Rename object
    workspace.list()              # List all objects
    workspace.exists(name)        # Check existence
    workspace.clear()             # Remove all objects
    len(workspace)                # Object count
    name in workspace             # Containment check
    for name in workspace: ...    # Iteration


HDF5 Persistence
----------------

Save and restore complete workspace state:

.. code-block:: python

    # Save everything
    workspace.save("analysis.h5")

    # Later, restore
    workspace.load("analysis.h5")

HDF5 format ensures:

- Platform-independent storage
- Efficient compression
- Complete metadata preservation
- Compatibility with other HDF5 tools


Mode Switching with Resync
--------------------------

Start standalone, switch to live when DataLab becomes available:

.. code-block:: python

    # Start without DataLab
    workspace = Workspace()
    workspace.add("signal", my_signal)

    # ... work offline ...

    # Later, DataLab is started
    if workspace.resync():
        # Objects transferred to DataLab
        # Now in live mode
        workspace.calc("normalize")


DataLab Processing (Live Mode)
------------------------------

Access DataLab's extensive processing library:

.. code-block:: python

    import sigima.params

    # Select and process
    workspace.proxy.select_objects(["my_signal"], panel="signal")
    workspace.calc("normalize", sigima.params.NormalizeParam())

    # Results appear as new objects


Interactive Visualization
-------------------------

Built-in matplotlib-based plotting:

.. code-block:: python

    plotter = Plotter(workspace)

    # Signal plotting
    plotter.plot("signal_name")

    # Image plotting with options
    plotter.plot("image_name", cmap="viridis", colorbar=True)


Sigima Integration
------------------

Full access to Sigima's data objects:

- **SignalObj**: 1D signals with error bars, units, metadata
- **ImageObj**: 2D images with coordinate systems
- **ROI support**: Regions of interest for focused analysis
- **Processing functions**: Via Sigima's proc module

.. code-block:: python

    from sigima import SignalObj, ImageObj, create_signal, create_image

    # Create rich data objects
    signal = create_signal(
        "Temperature",
        time_array, temp_array,
        units=("s", "°C"),
        labels=("Time", "Temperature"),
    )


Jupyter Integration
-------------------

Works seamlessly with:

- **Jupyter Notebook**: Classic notebook interface
- **JupyterLab**: Modern lab environment
- **VS Code**: With Jupyter extension

Register the kernel once:

.. code-block:: console

    $ datalab-kernel-install

Then select "DataLab" kernel in any Jupyter frontend.


Error Handling
--------------

User-friendly error messages:

.. code-block:: python

    >>> workspace.get("nonexistent")
    KeyError: "Object 'nonexistent' not found. Available: ['signal1', 'image1']"

Errors avoid exposing internal implementation details.


Data Integrity
--------------

Objects are copied on add to prevent accidental modification:

.. code-block:: python

    workspace.add("test", signal)
    signal.y[0] = 999  # Modify original

    retrieved = workspace.get("test")
    assert retrieved.y[0] != 999  # Workspace copy unaffected
