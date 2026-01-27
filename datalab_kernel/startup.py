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


# When this module is imported as a startup script, run the startup
if __name__ == "__main__":
    run_startup()
