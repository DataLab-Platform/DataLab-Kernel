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
