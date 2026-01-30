# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Jupyter Kernel
======================

A standalone Jupyter kernel providing seamless, reproducible access to DataLab
workspaces, with optional live synchronization to the DataLab GUI.

This kernel uses **xeus-python** as its backend for:
- Improved performance compared to ipykernel
- Native debugger support
- JupyterLite compatibility
- Better Qt event loop integration

Main components:
- `workspace`: Data access and persistence API
- `plotter`: Visualization frontend
- `sigima`: Scientific processing library (re-exported)

Example usage::

    # In a Jupyter notebook with datalab-kernel
    img = workspace.get("i042")
    filtered = sigima.proc.image.butterworth(img, cut_off=0.2)
    workspace.add("filtered_i042", filtered)
    plotter.plot("filtered_i042")
"""

from __future__ import annotations

__version__ = "0.2.6"
__author__ = "DataLab Platform Developers"

from datalab_kernel.plotter import Plotter

# Re-export IPython extension functions for %load_ext datalab_kernel
from datalab_kernel.startup import load_ipython_extension, unload_ipython_extension
from datalab_kernel.workspace import Workspace

# Global instances (initialized when kernel starts)
# pylint: disable=invalid-name
workspace: Workspace | None = None
plotter: Plotter | None = None
# pylint: enable=invalid-name

__all__ = [
    "__version__",
    "Workspace",
    "Plotter",
    "workspace",
    "plotter",
    "load_ipython_extension",
    "unload_ipython_extension",
]
