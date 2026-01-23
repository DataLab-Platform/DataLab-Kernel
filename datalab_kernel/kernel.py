# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Jupyter Kernel Implementation
=====================================

This module implements the Jupyter kernel for DataLab, providing:
- Automatic mode detection (standalone vs live)
- Pre-configured namespace with workspace, plotter, and sigima
- Standard IPython kernel functionality
"""

from __future__ import annotations

from ipykernel.ipkernel import IPythonKernel

from datalab_kernel import __version__
from datalab_kernel.plotter import Plotter
from datalab_kernel.workspace import Workspace


class DataLabKernel(IPythonKernel):
    """
    DataLab Jupyter Kernel.

    Extends IPythonKernel to provide DataLab-specific functionality:
    - Pre-initialized workspace and plotter objects
    - Automatic mode detection (standalone vs live DataLab)
    - sigima module pre-imported if available
    """

    implementation = "datalab-kernel"
    implementation_version = __version__
    language = "python"
    language_version = "3.9"
    language_info = {
        "name": "python",
        "version": "3.9",
        "mimetype": "text/x-python",
        "codemirror_mode": {"name": "ipython", "version": 3},
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
        "file_extension": ".py",
    }
    banner = f"DataLab Kernel {__version__}"
    help_links = [
        {
            "text": "DataLab Documentation",
            "url": "https://datalab-platform.com/",
        },
        {
            "text": "DataLab Kernel Documentation",
            "url": "https://datalab-kernel.readthedocs.io/",
        },
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._setup_namespace()

    def _setup_namespace(self) -> None:
        """Set up the default kernel namespace with workspace and plotter."""
        # Create workspace (auto-detects mode)
        workspace = Workspace()
        plotter = Plotter(workspace)

        # Get user namespace
        ns = self.shell.user_ns

        # Add core objects
        ns["workspace"] = workspace
        ns["plotter"] = plotter

        # Add numpy as np (common convention)
        import numpy as np

        ns["np"] = np

        # Try to import and add sigima
        try:
            import sigima

            ns["sigima"] = sigima
        except ImportError:
            pass

        # Add convenience functions
        from datalab_kernel.objects import create_image, create_signal

        ns["create_signal"] = create_signal
        ns["create_image"] = create_image

        # Store mode info
        mode_str = workspace.mode.value
        self.log.info(f"DataLab Kernel started in {mode_str} mode")

    @property
    def kernel_info(self) -> dict:
        """Return kernel info dictionary."""
        return {
            "protocol_version": "5.3",
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "language_info": self.language_info,
            "banner": self.banner,
            "help_links": self.help_links,
        }


# Entry point for kernel launch
if __name__ == "__main__":
    from ipykernel.kernelapp import IPKernelApp

    IPKernelApp.launch_instance(kernel_class=DataLabKernel)
