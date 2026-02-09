.. _api_plotter:

Plotter
=======

The ``Plotter`` class provides visualization for workspace objects.
It auto-selects the best available backend — **Plotly** (interactive) when
installed, falling back to **Matplotlib** (static PNG) otherwise.

.. automodule:: datalab_kernel.plotter
   :members:
   :undoc-members:
   :show-inheritance:

Usage
-----

.. code-block:: python

    from datalab_kernel import Workspace, Plotter

    workspace = Workspace()
    plotter = Plotter(workspace)

    # Add some objects
    workspace.add("signal", signal_obj)
    workspace.add("image", image_obj)

    # Plot by name (uses the best available backend)
    plotter.plot("signal")
    plotter.plot("image")

Backend Selection
-----------------

DataLab-Kernel ships two plotting backends:

.. list-table::
   :header-rows: 1

   * - Backend
     - Output
     - Install
   * - **Plotly**
     - Interactive HTML figures
     - ``pip install datalab-kernel[plotly]``
   * - **Matplotlib**
     - Static PNG images
     - Included by default

When both are installed **Plotly is preferred automatically**.
You can override this at any time:

.. code-block:: python

    # Check the active backend
    print(plotter.backend)  # "plotly" or "matplotlib"

    # Switch at runtime
    plotter.set_backend("matplotlib")
    plotter.plot("my_signal")  # static PNG output

    plotter.set_backend("plotly")
    plotter.plot("my_signal")  # interactive figure

    # Or pass the backend at construction time
    plotter = Plotter(workspace, backend="matplotlib")

You can also set the default via an **environment variable** before starting
the kernel:

.. code-block:: console

    $ export DATALAB_PLOTTER_BACKEND=matplotlib

If the requested backend is not installed, the plotter falls back to the other
one with a warning.  If *neither* is installed, an ``ImportError`` is raised.

Signal Plotting
---------------

Signals are plotted as line plots:

.. code-block:: python

    plotter.plot("my_signal")

Image Plotting
--------------

Images are displayed as 2D colormaps:

.. code-block:: python

    plotter.plot("my_image")

    # With a specific colormap
    plotter.plot("my_image", colormap="hot")

    # Override the auto-computed figure height (pixels)
    plotter.plot("my_image", height=600)

Multi-Object Plotting
---------------------

Pass a **list** to ``plot()`` to display multiple objects at once.
The plotter auto-detects whether items are signals or images:

**Multiple signals** — overlaid on shared axes with automatic legend and
color cycling:

.. code-block:: python

    plotter.plot([sig1, sig2, sig3])

    # With explicit axis labels and units
    plotter.plot(
        [sig1, sig2],
        title="Comparison",
        xlabel="Frequency", ylabel="Magnitude",
        xunit="Hz", yunit="dB",
    )

    # Raw numpy arrays and (x, y) tuples are also supported
    import numpy as np
    plotter.plot([np.sin(np.linspace(0, 10, 200)), (x_array, y_array)])

**Multiple images** — displayed in a subplot grid (up to 4 columns):

.. code-block:: python

    plotter.plot([img1, img2, img3])

    # With per-image titles and a shared colormap
    plotter.plot(
        [img1, img2],
        titles=["Before", "After"],
        colormap="hot",
    )

**Mixed lists** of signals and images raise a ``TypeError``.
A single-item list behaves identically to passing the object directly.
