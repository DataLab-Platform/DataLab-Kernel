.. _example_live_mode:

Live Mode
=========

This example shows how to use DataLab-Kernel with a running DataLab instance.

Prerequisites
-------------

1. Install DataLab: ``pip install datalab-platform``
2. Start DataLab before running this example

Connecting to DataLab
---------------------

.. code-block:: python

    from datalab_kernel import Workspace, Plotter, WorkspaceMode
    from datalab_kernel.objects import create_signal
    import numpy as np

    # Create workspace - auto-connects if DataLab is running
    workspace = Workspace()

    if workspace.mode == WorkspaceMode.LIVE:
        print("✓ Connected to DataLab!")
        print(f"  Version: {workspace.proxy.get_version()}")
    else:
        print("DataLab not detected - running in standalone mode")
        print("Start DataLab and call workspace.resync() to connect")


Creating Objects in DataLab
---------------------------

.. code-block:: python

    # Create and add signal - appears in DataLab immediately
    x = np.linspace(0, 10, 500)
    y = np.sin(x) * np.exp(-x / 5)

    signal = create_signal(
        "Damped Oscillation",
        x, y,
        units=("s", "V"),
        labels=("Time", "Voltage"),
    )

    workspace.add("oscillation", signal)
    print("Signal added - check DataLab's signal panel!")


Using DataLab Processing
------------------------

.. code-block:: python

    import sigima.params

    if workspace.mode == WorkspaceMode.LIVE:
        # Select the signal in DataLab
        workspace.proxy.select_objects(["oscillation"], panel="signal")

        # Apply normalization with explicit parameters
        param = sigima.params.NormalizeParam()  # Uses default: method="Maximum"
        workspace.calc("normalize", param)

        print("Normalized signal created in DataLab")

        # List objects - includes the new normalized result
        print("Current objects:", workspace.list())


Switching from Standalone to Live
---------------------------------

.. code-block:: python

    from datalab_kernel.workspace import StandaloneBackend

    # Start explicitly in standalone mode
    workspace = Workspace(backend=StandaloneBackend())
    print(f"Initial mode: {workspace.mode}")

    # Create some objects
    workspace.add("test_signal", create_signal("Test", x, np.sin(x)))

    # Later, after starting DataLab...
    if workspace.resync():
        print("Switched to live mode!")
        print("Objects transferred to DataLab")
    else:
        print("Could not connect to DataLab")


Direct Proxy Access
-------------------

For advanced operations, access the RemoteProxy directly:

.. code-block:: python

    if workspace.mode == WorkspaceMode.LIVE:
        proxy = workspace.proxy

        # Get DataLab version
        print("DataLab version:", proxy.get_version())

        # Select specific objects
        proxy.select_objects(["oscillation"], panel="signal")

        # Get current selection
        # (DataLab API dependent)

        # Call any proxy method
        # See DataLab documentation for full API


Clearing DataLab
----------------

.. code-block:: python

    # Clear all objects from workspace/DataLab
    workspace.clear()
    print("All objects removed from DataLab")
