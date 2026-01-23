.. _example_basic_usage:

Basic Usage
===========

This example demonstrates the fundamental operations of DataLab-Kernel.

Creating a Workspace
--------------------

.. code-block:: python

    from datalab_kernel import Workspace, Plotter
    from datalab_kernel.objects import create_signal, create_image
    import numpy as np

    # Create workspace (auto-detects DataLab)
    workspace = Workspace()
    plotter = Plotter(workspace)

    print(f"Operating mode: {workspace.mode}")


Working with Signals
--------------------

.. code-block:: python

    # Create time-domain signal
    t = np.linspace(0, 1, 1000)
    freq = 5  # Hz
    signal = np.sin(2 * np.pi * freq * t) + 0.2 * np.random.randn(len(t))

    # Create signal object with metadata
    sig = create_signal(
        "Noisy Sine Wave",
        t, signal,
        units=("s", "V"),
        labels=("Time", "Amplitude"),
    )

    # Add to workspace
    workspace.add("sine_wave", sig)

    # Visualize
    plotter.plot("sine_wave")


Working with Images
-------------------

.. code-block:: python

    # Create 2D Gaussian
    size = 128
    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)
    X, Y = np.meshgrid(x, y)
    gaussian = np.exp(-(X**2 + Y**2) / 2)

    # Add some noise
    noisy_gaussian = gaussian + 0.1 * np.random.randn(size, size)

    # Create image object
    img = create_image(
        "Noisy Gaussian",
        noisy_gaussian.astype(np.float32),
        units=("mm", "mm", "a.u."),
        labels=("X", "Y", "Intensity"),
    )

    # Add and visualize
    workspace.add("gaussian", img)
    plotter.plot("gaussian")


Workspace Operations
--------------------

.. code-block:: python

    # List all objects
    print("Objects:", workspace.list())

    # Check existence
    print("Has 'sine_wave':", workspace.exists("sine_wave"))
    print("Has 'unknown':", workspace.exists("unknown"))

    # Object count
    print("Count:", len(workspace))

    # Iteration
    for name in workspace:
        print(f"  - {name}")

    # Retrieve object
    retrieved = workspace.get("sine_wave")
    print("Retrieved signal title:", retrieved.title)

    # Rename
    workspace.rename("gaussian", "gaussian_2d")
    print("After rename:", workspace.list())

    # Remove
    workspace.remove("gaussian_2d")
    print("After remove:", workspace.list())


Containment Check
-----------------

.. code-block:: python

    # Using 'in' operator
    if "sine_wave" in workspace:
        print("Signal exists!")

    if "nonexistent" not in workspace:
        print("Object not found")
