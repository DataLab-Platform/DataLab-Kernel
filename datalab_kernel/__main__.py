# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Kernel main module.

Supports running as:
    python -m datalab_kernel install
    python -m datalab_kernel uninstall
"""

from __future__ import annotations

import sys

from datalab_kernel.install import main as install_main


def main() -> None:
    """Main entry point for the datalab_kernel module."""
    if len(sys.argv) > 1 and sys.argv[1] in ("install", "uninstall"):
        # Handle install/uninstall commands
        install_main()
    else:
        # Launch kernel (default behavior when run by Jupyter)
        from ipykernel.kernelapp import IPKernelApp

        from datalab_kernel.kernel import DataLabKernel

        IPKernelApp.launch_instance(kernel_class=DataLabKernel)


if __name__ == "__main__":
    main()
