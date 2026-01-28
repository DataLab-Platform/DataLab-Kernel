# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Kernel Startup Script
=============================

This module provides the startup logic for the DataLab xeus-python kernel.
It is executed when the kernel starts to inject the DataLab namespace into
the user's interactive environment.

The startup script:
- Creates Workspace and Plotter instances
- Imports numpy as np and sigima
- Adds create_signal and create_image convenience functions
- Logs the kernel mode (standalone or live)
"""

from __future__ import annotations

import logging

import numpy as np
from sigima import create_image, create_signal

from datalab_kernel import __version__
from datalab_kernel.plotter import Plotter
from datalab_kernel.workspace import Workspace

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("datalab-kernel")


def setup_namespace() -> dict:
    """Set up the DataLab namespace for the kernel.

    Returns:
        Dictionary of objects to inject into user namespace.
    """
    # Create workspace (auto-detects mode)
    workspace = Workspace()
    plotter = Plotter(workspace)

    # Build namespace dictionary
    namespace = {
        "workspace": workspace,
        "plotter": plotter,
        "np": np,
        "create_signal": create_signal,
        "create_image": create_image,
    }

    # Try to import and add sigima
    try:
        import sigima

        namespace["sigima"] = sigima
    except ImportError:
        pass

    # Log mode info
    mode_str = workspace.mode.value
    logger.info(f"DataLab Kernel {__version__} started in {mode_str} mode")

    return namespace


def run_startup():
    """Execute the startup script and inject namespace into IPython.

    This function is called by xeus-python via the startup script mechanism.
    It injects the DataLab namespace into the current IPython shell.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None:
            namespace = setup_namespace()
            ip.user_ns.update(namespace)
            logger.info("DataLab namespace injected successfully")
        else:
            logger.warning("No IPython instance found, namespace not injected")
    except Exception as e:
        logger.error(f"Failed to setup DataLab namespace: {e}")
        raise


# =============================================================================
# IPython Extension API
# =============================================================================
# These functions allow datalab_kernel to be loaded as an IPython extension,
# which is the recommended way to use it in JupyterLite environments.
#
# Usage in IPython/Jupyter:
#   %load_ext datalab_kernel
#
# Or configure auto-loading in ipython_config.py:
#   c.InteractiveShellApp.extensions = ['datalab_kernel']
# =============================================================================


def load_ipython_extension(ipython):
    """Load the DataLab kernel extension.

    This function is called by IPython when the extension is loaded via:
        %load_ext datalab_kernel

    It injects the DataLab namespace (workspace, plotter, sigima, etc.)
    into the user's interactive environment.

    This is the recommended way to use DataLab-Kernel in JupyterLite,
    where the traditional kernel installation mechanism doesn't apply.

    Args:
        ipython: The active IPython shell instance.
    """
    namespace = setup_namespace()
    ipython.user_ns.update(namespace)
    logger.info("DataLab extension loaded successfully")


def unload_ipython_extension(ipython):
    """Unload the DataLab kernel extension.

    This function is called by IPython when the extension is unloaded.
    It removes the DataLab namespace from the user's environment.

    Args:
        ipython: The active IPython shell instance.
    """
    # List of names we injected
    names_to_remove = [
        "workspace",
        "plotter",
        "np",
        "sigima",
        "create_signal",
        "create_image",
    ]
    for name in names_to_remove:
        ipython.user_ns.pop(name, None)
    logger.info("DataLab extension unloaded")


# When this module is imported as a startup script, run the startup
if __name__ == "__main__":
    run_startup()
