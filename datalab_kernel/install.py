# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Kernel installation and management
==================================

This module provides commands to install and uninstall the DataLab kernel.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

KERNEL_NAME = "datalab-kernel"
KERNEL_DISPLAY_NAME = "DataLab"


def get_kernel_spec() -> dict:
    """Generate the kernel.json specification."""
    return {
        "argv": [
            sys.executable,
            "-m",
            "datalab_kernel.kernel",
            "-f",
            "{connection_file}",
        ],
        "display_name": KERNEL_DISPLAY_NAME,
        "language": "python",
        "metadata": {
            "debugger": True,
        },
    }


def get_kernel_dir(user: bool = True) -> Path:
    """Get the directory where the kernel spec should be installed.

    Args:
        user: If True, install in user directory; otherwise system-wide

    Returns:
        Path to kernel directory
    """
    # Delayed import to avoid dependency at module load time
    # pylint: disable=import-outside-toplevel
    from jupyter_client.kernelspec import KernelSpecManager

    ksm = KernelSpecManager()

    if user:
        # User kernels directory
        data_dir = Path(ksm.user_kernel_dir)
    else:
        # System kernels directory (first one in path)
        data_dir = Path(ksm.kernel_dirs[0]) if ksm.kernel_dirs else None
        if data_dir is None:
            raise RuntimeError("Could not find system kernel directory")

    return data_dir / KERNEL_NAME


def install_kernel(user: bool = True, prefix: str | None = None) -> Path:
    """Install the DataLab kernel.

    Args:
        user: If True, install for current user only
        prefix: Install prefix (overrides user setting)

    Returns:
        Path to installed kernel directory
    """
    if prefix:
        kernel_dir = Path(prefix) / "share" / "jupyter" / "kernels" / KERNEL_NAME
    else:
        kernel_dir = get_kernel_dir(user=user)

    # Create directory
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Write kernel.json
    kernel_json_path = kernel_dir / "kernel.json"
    with open(kernel_json_path, "w", encoding="utf-8") as f:
        json.dump(get_kernel_spec(), f, indent=2)

    # Copy logo files if available
    logo_dir = Path(__file__).parent / "resources"
    if logo_dir.exists():
        for logo_file in logo_dir.glob("logo-*.png"):
            shutil.copy(logo_file, kernel_dir)

    print(f"Installed DataLab kernel to: {kernel_dir}")
    return kernel_dir


def uninstall_kernel(user: bool = True, prefix: str | None = None) -> bool:
    """Uninstall the DataLab kernel.

    Args:
        user: If True, uninstall from user directory
        prefix: Install prefix (overrides user setting)

    Returns:
        True if kernel was found and removed
    """
    if prefix:
        kernel_dir = Path(prefix) / "share" / "jupyter" / "kernels" / KERNEL_NAME
    else:
        kernel_dir = get_kernel_dir(user=user)

    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
        print(f"Uninstalled DataLab kernel from: {kernel_dir}")
        return True
    print(f"DataLab kernel not found at: {kernel_dir}")
    return False


def main() -> None:
    """Main entry point for kernel installation CLI."""
    parser = argparse.ArgumentParser(
        description="Install or uninstall the DataLab Jupyter kernel"
    )
    parser.add_argument(
        "action",
        choices=["install", "uninstall"],
        help="Action to perform",
    )
    parser.add_argument(
        "--user",
        action="store_true",
        default=True,
        help="Install/uninstall for current user only (default)",
    )
    parser.add_argument(
        "--system",
        action="store_true",
        help="Install/uninstall system-wide",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Install prefix path",
    )

    args = parser.parse_args()

    user = not args.system

    if args.action == "install":
        install_kernel(user=user, prefix=args.prefix)
    else:
        uninstall_kernel(user=user, prefix=args.prefix)


if __name__ == "__main__":
    main()
