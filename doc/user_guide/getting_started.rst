.. _getting_started:

Getting Started
===============

This guide will help you get up and running with DataLab-Kernel quickly.

Basic Usage
-----------

Here's a minimal example to get started:

.. code-block:: python

    from datalab_kernel import Workspace, Plotter
    from sigima import create_signal, create_image
    import numpy as np

    # Create workspace (auto-detects DataLab if running)
    workspace = Workspace()
    plotter = Plotter(workspace)

    # Create a signal
    x = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * x) * np.exp(-x / 3)
    signal = create_signal("Damped Oscillation", x, y,
                           units=("s", "V"),
                           labels=("Time", "Amplitude"))

    # Add to workspace
    workspace.add("oscillation", signal)

    # Visualize
    plotter.plot("oscillation")


Working with Images
-------------------

.. code-block:: python

    # Create a 2D Gaussian image
    size = 256
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    data = np.exp(-(X**2 + Y**2) / 2)

    image = create_image("Gaussian", data,
                         units=("mm", "mm", "a.u."),
                         labels=("X", "Y", "Intensity"))

    workspace.add("gaussian", image)
    plotter.plot("gaussian")


Standalone vs Live Mode
-----------------------

DataLab-Kernel operates in two modes:

**Standalone Mode** (default when DataLab is not running)

- Objects stored locally in memory
- Persistence via HDF5 files
- Full functionality without DataLab

**Live Mode** (when DataLab is running)

- Objects synchronized with DataLab
- Access to DataLab's processing functions via ``calc()``
- Direct proxy access for advanced operations

Check your current mode:

.. code-block:: python

    from datalab_kernel import Workspace, WorkspaceMode

    workspace = Workspace()

    if workspace.mode == WorkspaceMode.LIVE:
        print("Connected to DataLab!")
    else:
        print("Running standalone")


Connecting via Web API
----------------------

DataLab-Kernel automatically discovers and connects to a running DataLab instance.
No configuration is needed in most cases.

**Auto-Discovery (recommended)**

Simply start DataLab with the Web API enabled (View → Web API → Start Web API Server),
then load the kernel extension:

.. code-block:: python

    %load_ext datalab_kernel

    # Already connected! Start using workspace immediately:
    workspace.list()

The auto-discovery mechanism tries the following methods in order:

1. **Environment variables** (``DATALAB_WORKSPACE_URL``, ``DATALAB_WORKSPACE_TOKEN``)
2. **Connection file** written by DataLab to ``~/.config/datalab/webapi_connection.json``
   (Linux/Mac) or ``%APPDATA%/datalab/webapi_connection.json`` (Windows)
3. **URL query parameters** (JupyterLite: ``?datalab_url=...&datalab_token=...``)
4. **Well-known port probing** (``http://127.0.0.1:18080`` with localhost bypass enabled)

If discovery fails, the workspace starts in standalone mode and you can connect later.

**Manual Connection**

If auto-discovery doesn't work (e.g., running on a different machine), set
environment variables before starting your notebook:

.. code-block:: bash

    export DATALAB_WORKSPACE_URL=http://127.0.0.1:18080
    export DATALAB_WORKSPACE_TOKEN=<your-token>

Or call ``connect()`` explicitly:

.. code-block:: python

    workspace.connect(url="http://192.168.1.100:18080", token="your-token")

.. note::

    The Web API supports WASM/Pyodide environments and provides efficient
    binary data transfer via NPZ format.


Switching to Live Mode
----------------------

If you started in standalone mode and DataLab becomes available,
use ``resync()`` to switch:

.. code-block:: python

    # Started without DataLab...
    workspace = Workspace()  # Standalone mode

    # ... later, after starting DataLab:
    if workspace.resync():
        print("Now connected to DataLab!")
        print(f"Mode: {workspace.mode}")


Saving and Loading
------------------

Persist your workspace to HDF5:

.. code-block:: python

    # Save current state
    workspace.save("my_analysis.h5")

    # Later, load it back
    workspace2 = Workspace()
    workspace2.load("my_analysis.h5")

    # All objects are restored
    print(workspace2.list())


Using DataLab Processing (Live Mode)
------------------------------------

When connected to DataLab, access its processing functions:

.. code-block:: python

    import sigima.params

    # Ensure we're in live mode
    if workspace.mode == WorkspaceMode.LIVE:
        # Select object in DataLab
        workspace.proxy.select_objects(["oscillation"], panel="signal")

        # Apply normalization
        param = sigima.params.NormalizeParam()
        workspace.calc("normalize", param)

        # Result appears as new object in DataLab
