# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Jupyter Kernel Implementation (xeus-python backend)
============================================================

This module provides kernel metadata and launch functionality for the
DataLab kernel, which is built on xeus-python for improved performance,
JupyterLite compatibility, and better debugging support.

Unlike ipykernel-based kernels, xeus-python is a C++ implementation that
cannot be directly subclassed. Customization is done via:
- IPython startup scripts for namespace injection
- Kernel specification configuration
"""

from __future__ import annotations

from datalab_kernel import __version__

# Kernel metadata (used by kernel spec and for identification)
KERNEL_NAME = "datalab-kernel"
KERNEL_DISPLAY_NAME = "DataLab"
KERNEL_IMPLEMENTATION = "datalab-kernel"
KERNEL_IMPLEMENTATION_VERSION = __version__
KERNEL_LANGUAGE = "python"
KERNEL_LANGUAGE_VERSION = "3.9"

KERNEL_LANGUAGE_INFO = {
    "name": "python",
    "version": "3.9",
    "mimetype": "text/x-python",
    "codemirror_mode": {"name": "ipython", "version": 3},
    "pygments_lexer": "ipython3",
    "nbconvert_exporter": "python",
    "file_extension": ".py",
}

KERNEL_BANNER = f"DataLab Kernel {__version__} (xeus-python backend)"

KERNEL_HELP_LINKS = [
    {
        "text": "DataLab Documentation",
        "url": "https://datalab-platform.com/",
    },
    {
        "text": "DataLab Kernel Documentation",
        "url": "https://datalab-kernel.readthedocs.io/",
    },
    {
        "text": "xeus-python Documentation",
        "url": "https://xeus-python.readthedocs.io/",
    },
]


def get_kernel_info() -> dict:
    """Return kernel info dictionary.

    This provides the same information that would be returned by
    a kernel's kernel_info_reply message.
    """
    return {
        "protocol_version": "5.3",
        "implementation": KERNEL_IMPLEMENTATION,
        "implementation_version": KERNEL_IMPLEMENTATION_VERSION,
        "language_info": KERNEL_LANGUAGE_INFO,
        "banner": KERNEL_BANNER,
        "help_links": KERNEL_HELP_LINKS,
    }


def launch_kernel():
    """Launch the DataLab kernel using xeus-python.

    This is the main entry point when running:
        python -m datalab_kernel

    It launches xeus-python with the DataLab startup configuration.
    """
    import sys

    # Import and launch xeus-python
    try:
        # xeus-python provides an extension module for launching
        from xeus_python_shell.shell import XPythonShellApp

        # Create and initialize the shell app
        app = XPythonShellApp.instance()
        app.initialize()

        # Inject our namespace into the shell
        from datalab_kernel.startup import setup_namespace

        namespace = setup_namespace()
        app.shell.user_ns.update(namespace)

        # The kernel is now ready - xeus-python handles the rest
        # via the native C++ kernel when invoked properly

    except ImportError as e:
        # Fall back to informative error if xeus-python not available
        print(
            f"Error: xeus-python is required but not installed: {e}\n"
            "Install with: pip install xeus-python xeus-python-shell",
            file=sys.stderr,
        )
        sys.exit(1)


# Entry point for kernel launch (when run as module with xpython)
if __name__ == "__main__":
    launch_kernel()
