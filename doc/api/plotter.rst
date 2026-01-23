.. _api_plotter:

Plotter
=======

The ``Plotter`` class provides visualization for workspace objects.

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

    # Plot by name
    plotter.plot("signal")
    plotter.plot("image")

Signal Plotting
---------------

Signals are plotted as line plots:

.. code-block:: python

    plotter.plot("my_signal")

    # With custom options
    plotter.plot("my_signal", color="red", linewidth=2)

Image Plotting
--------------

Images are displayed as 2D colormaps:

.. code-block:: python

    plotter.plot("my_image")

    # With colormap and colorbar
    plotter.plot("my_image", cmap="viridis", colorbar=True)

    # With specific value range
    plotter.plot("my_image", vmin=0, vmax=100)
