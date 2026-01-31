# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Development Wheel Server for JupyterLite Testing
=================================================

This script builds development wheels for guidata, Sigima and DataLab-Kernel,
then serves them via HTTP for installation in JupyterLite.

Usage:
    python scripts/serve_dev_wheels.py

    # Or with custom port
    python scripts/serve_dev_wheels.py --port 9999

    # Build only (no server)
    python scripts/serve_dev_wheels.py --build-only

Then in JupyterLite:
    import micropip
    await micropip.install([
        "http://localhost:8888/sigima-<version>-py3-none-any.whl",
        "http://localhost:8888/datalab_kernel-<version>-py3-none-any.whl",
    ])
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Default paths - adjust if your workspace layout differs
DEFAULT_GUIDATA_PATH = Path("C:/Dev/guidata")
DEFAULT_SIGIMA_PATH = Path("C:/Dev/Sigima")
DEFAULT_KERNEL_PATH = Path("C:/Dev/DataLab-Kernel")
DEFAULT_WHEELS_DIR = Path("C:/Dev/wheels")
DEFAULT_PORT = 8888


def ensure_build_module() -> bool:
    """Ensure the 'build' module is installed.

    Returns:
        True if build module is available, False otherwise
    """
    try:
        import build  # noqa: F401

        return True
    except ImportError:
        print("The 'build' module is not installed. Installing it now...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "build"],
            capture_output=False,
        )
        if result.returncode != 0:
            print("ERROR: Failed to install 'build' module.")
            print("Please install it manually: pip install build")
            return False
        print("Successfully installed 'build' module.\n")
        return True


def build_wheel(project_path: Path, wheels_dir: Path) -> Path | None:
    """Build a wheel for a project and copy it to wheels_dir.

    Args:
        project_path: Path to the project root (containing pyproject.toml)
        wheels_dir: Directory to copy the built wheel to

    Returns:
        Path to the copied wheel, or None if build failed
    """
    project_name = project_path.name
    print(f"\n{'=' * 60}")
    print(f"Building wheel for {project_name}...")
    print(f"{'=' * 60}")

    if not project_path.exists():
        print(f"ERROR: Project path does not exist: {project_path}")
        return None

    # Build the wheel
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel"],
        cwd=project_path,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"ERROR: Failed to build wheel for {project_name}")
        return None

    # Find the built wheel
    dist_dir = project_path / "dist"
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        print(f"ERROR: No wheel found in {dist_dir}")
        return None

    # Get the most recent wheel
    latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
    print(f"Built: {latest_wheel.name}")

    # Copy to wheels directory
    dest_path = wheels_dir / latest_wheel.name
    shutil.copy2(latest_wheel, dest_path)
    print(f"Copied to: {dest_path}")

    return dest_path


def generate_install_snippet(wheels_dir: Path, port: int) -> str:
    """Generate Python code snippet for installing wheels in JupyterLite.

    Args:
        wheels_dir: Directory containing the wheels
        port: HTTP server port

    Returns:
        Python code snippet as a string
    """
    wheels = list(wheels_dir.glob("*.whl"))
    if not wheels:
        return "# No wheels found"

    lines = [
        "# === RUN THIS CELL FIRST AFTER EVERY KERNEL RESTART ===",
        "# JupyterLite resets to the original environment on restart,",
        "# so you must re-install dev wheels each time.",
        "",
        "import micropip",
        "",
        "# Install dev wheels "
        "(deps=False skips xeus-python which can't run in Pyodide)",
        "await micropip.install([",
    ]

    for wheel in sorted(wheels):
        lines.append(f'    "http://localhost:{port}/{wheel.name}",')

    lines.append("], deps=False)")
    lines.append("")
    lines.append("print('Dev wheels installed! You can now import:')")
    lines.append("print('  from sigima import SignalObj, ImageObj')")
    lines.append("print('  from datalab_kernel import Workspace, Plotter')")

    return "\n".join(lines)


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler with CORS support for JupyterLite."""

    def end_headers(self):
        # Add CORS headers for JupyterLite access
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def serve_wheels(wheels_dir: Path, port: int) -> None:
    """Start HTTP server to serve wheels.

    Args:
        wheels_dir: Directory containing wheels to serve
        port: Port to serve on
    """
    os.chdir(wheels_dir)

    handler = functools.partial(CORSRequestHandler, directory=str(wheels_dir))
    server = http.server.HTTPServer(("", port), handler)

    print(f"\n{'=' * 60}")
    print(f"Serving wheels at http://localhost:{port}/")
    print(f"{'=' * 60}")
    print("\nAvailable wheels:")
    for wheel in sorted(wheels_dir.glob("*.whl")):
        print(f"  - {wheel.name}")

    print(f"\n{generate_install_snippet(wheels_dir, port)}")

    print(f"\n{'=' * 60}")
    print("Press Ctrl+C to stop the server")
    print(f"{'=' * 60}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Build and serve development wheels for JupyterLite testing"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"HTTP server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--guidata-path",
        type=Path,
        default=DEFAULT_GUIDATA_PATH,
        help=f"Path to guidata project (default: {DEFAULT_GUIDATA_PATH})",
    )
    parser.add_argument(
        "--sigima-path",
        type=Path,
        default=DEFAULT_SIGIMA_PATH,
        help=f"Path to Sigima project (default: {DEFAULT_SIGIMA_PATH})",
    )
    parser.add_argument(
        "--kernel-path",
        type=Path,
        default=DEFAULT_KERNEL_PATH,
        help=f"Path to DataLab-Kernel project (default: {DEFAULT_KERNEL_PATH})",
    )
    parser.add_argument(
        "--wheels-dir",
        type=Path,
        default=DEFAULT_WHEELS_DIR,
        help=f"Directory to store wheels (default: {DEFAULT_WHEELS_DIR})",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build wheels only, don't start server",
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Serve existing wheels only, don't rebuild",
    )
    parser.add_argument(
        "--skip-guidata",
        action="store_true",
        help="Skip building guidata wheel",
    )
    parser.add_argument(
        "--skip-sigima",
        action="store_true",
        help="Skip building Sigima wheel",
    )
    parser.add_argument(
        "--skip-kernel",
        action="store_true",
        help="Skip building DataLab-Kernel wheel",
    )

    args = parser.parse_args()

    # Create wheels directory
    args.wheels_dir.mkdir(parents=True, exist_ok=True)

    # Build wheels unless --serve-only
    if not args.serve_only:
        # Ensure build module is available
        if not ensure_build_module():
            return 1

        built_wheels = []

        if not args.skip_guidata:
            wheel = build_wheel(args.guidata_path, args.wheels_dir)
            if wheel:
                built_wheels.append(wheel)

        if not args.skip_sigima:
            wheel = build_wheel(args.sigima_path, args.wheels_dir)
            if wheel:
                built_wheels.append(wheel)

        if not args.skip_kernel:
            wheel = build_wheel(args.kernel_path, args.wheels_dir)
            if wheel:
                built_wheels.append(wheel)

        if not built_wheels:
            print("\nNo wheels were built successfully.")
            if not args.build_only:
                print("Use --serve-only to serve existing wheels.")
            return 1

        print(f"\n✓ Built {len(built_wheels)} wheel(s)")

    # Serve wheels unless --build-only
    if not args.build_only:
        serve_wheels(args.wheels_dir, args.port)

    return 0


if __name__ == "__main__":
    sys.exit(main())
