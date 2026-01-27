# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Kernel main module.

Supports running as:
    python -m datalab_kernel install
    python -m datalab_kernel uninstall
    python -m datalab_kernel  (launches kernel)
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
        # With xeus-python, we need to inject our namespace and then
        # let xeus-python handle the kernel protocol

        # First, set up the environment to indicate we want DataLab startup
        import os

        os.environ["DATALAB_KERNEL_STARTUP"] = "1"

        try:
            # Try to use xeus-python's native launch
            # xeus-python installs an 'xpython' executable
            import subprocess

            xpython = subprocess.run(
                ["xpython", "--version"],
                capture_output=True,
                text=True,
            )

            if xpython.returncode == 0:
                # xpython is available, use it
                # Pass through all arguments
                args = ["xpython"] + sys.argv[1:]
                subprocess.run(args)
            else:
                raise FileNotFoundError("xpython not found")

        except (FileNotFoundError, subprocess.SubprocessError):
            # xpython not available, try Python module approach
            try:
                # Import xeus-python's main entry point
                import xeus_python

                # Set up our namespace first
                from datalab_kernel.startup import run_startup

                run_startup()

                # Now launch xeus-python
                xeus_python.main()

            except ImportError as e:
                print(
                    f"Error: xeus-python is required but not installed: {e}\n"
                    "Install with: pip install xeus-python xeus-python-shell",
                    file=sys.stderr,
                )
                sys.exit(1)


if __name__ == "__main__":
    main()
