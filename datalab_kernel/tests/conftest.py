# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Pytest configuration for DataLab Kernel tests.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "standalone: tests for standalone mode only")
    config.addinivalue_line("markers", "live: tests requiring live DataLab instance")
    config.addinivalue_line("markers", "contract: contract tests for both modes")
    config.addinivalue_line(
        "markers", "integration: integration tests with external services"
    )
    config.addinivalue_line("markers", "webapi: tests requiring WebAPI backend")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests requiring a live DataLab instance",
    )
    parser.addoption(
        "--start-datalab",
        action="store_true",
        default=False,
        help="Automatically start DataLab for live tests (requires --live)",
    )
    parser.addoption(
        "--webapi",
        action="store_true",
        default=False,
        help="Run WebAPI integration tests (starts WebAPI server in DataLab)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on --live and --webapi flags."""
    if config.getoption("--live"):
        # Skip standalone-only tests when running in live mode
        skip_standalone = pytest.mark.skip(
            reason="Test only runs without --live (tests standalone fallback)"
        )
        for item in items:
            if "standalone" in item.keywords:
                item.add_marker(skip_standalone)
    else:
        # Skip live tests when not in live mode
        skip_live = pytest.mark.skip(reason="Need --live option to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)

    # Handle WebAPI tests
    if not config.getoption("--webapi"):
        skip_webapi = pytest.mark.skip(reason="Need --webapi option to run")
        for item in items:
            if "webapi" in item.keywords:
                item.add_marker(skip_webapi)


# Global process handle for DataLab
# pylint: disable=invalid-name
_datalab_process = None
_webapi_info = None  # {"url": ..., "token": ...}


def _is_datalab_running():
    """Check if DataLab is running and accessible."""
    # pylint: disable=import-outside-toplevel
    try:
        from datalab.control.proxy import RemoteProxy

        proxy = RemoteProxy(autoconnect=False)
        proxy.connect(timeout=2.0)
        proxy.disconnect()
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _start_datalab():
    """Start DataLab in background."""
    global _datalab_process  # pylint: disable=global-statement

    if _is_datalab_running():
        return  # Already running

    # Set environment to prevent auto-quit
    env = os.environ.copy()
    env["DATALAB_DO_NOT_QUIT"] = env["GUIDATA_UNATTENDED"] = "1"

    # Start DataLab in background
    # Note: This may not work if DataLab requires display or has other issues
    # pylint: disable=consider-using-with
    _datalab_process = subprocess.Popen(
        [sys.executable, "-m", "datalab.app"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for DataLab to be ready
    max_wait = 30  # seconds
    start_time = time.time()
    while time.time() - start_time < max_wait:
        # Check if process has exited
        if _datalab_process.poll() is not None:
            stdout, stderr = _datalab_process.communicate()
            raise RuntimeError(
                f"DataLab process exited immediately with code "
                f"{_datalab_process.returncode}.\n"
                f"stdout: {stdout.decode('utf-8', errors='replace')[:500]}\n"
                f"stderr: {stderr.decode('utf-8', errors='replace')[:500]}\n"
                "Try starting DataLab manually before running tests."
            )
        if _is_datalab_running():
            return
        time.sleep(0.5)

    raise RuntimeError(
        "DataLab failed to start within 30 seconds. "
        "Please start DataLab manually and run: pytest --live"
    )


def _stop_datalab():
    """Stop DataLab if we started it."""
    global _datalab_process  # pylint: disable=global-statement

    if _datalab_process is not None:
        # pylint: disable=import-outside-toplevel
        try:
            from datalab.control.proxy import RemoteProxy

            proxy = RemoteProxy(autoconnect=False)
            proxy.connect(timeout=2.0)
            proxy.close_application()
            proxy.disconnect()
        except Exception:  # pylint: disable=broad-exception-caught
            # Force kill if close_application fails
            _datalab_process.kill()
        finally:
            _datalab_process = None


def _close_datalab_session():
    """Close DataLab session via proxy (regardless of who started it)."""
    # pylint: disable=import-outside-toplevel
    try:
        from datalab.control.proxy import RemoteProxy

        proxy = RemoteProxy(autoconnect=False)
        proxy.connect(timeout=2.0)
        proxy.close_application()
        proxy.disconnect()
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # DataLab may already be closed or not running


def _start_webapi_server():
    """Start the WebAPI server in DataLab and configure environment."""
    global _webapi_info  # pylint: disable=global-statement
    # pylint: disable=import-outside-toplevel

    try:
        from datalab.control.proxy import RemoteProxy

        proxy = RemoteProxy(autoconnect=False)
        proxy.connect(timeout=5.0)

        # Check if WebAPI is available
        status = proxy.get_webapi_status()
        if not status.get("available"):
            raise RuntimeError(
                "WebAPI not available. "
                "Install with: pip install datalab-platform[webapi]"
            )

        # Start WebAPI server if not already running
        if not status.get("running"):
            info = proxy.start_webapi_server()
            _webapi_info = info
        else:
            _webapi_info = {"url": status["url"], "token": status["token"]}

        # Set environment variables for WebApiBackend
        os.environ["DATALAB_WORKSPACE_URL"] = _webapi_info["url"]
        os.environ["DATALAB_WORKSPACE_TOKEN"] = _webapi_info["token"]

        proxy.disconnect()

    except Exception as e:  # pylint: disable=broad-exception-caught
        raise RuntimeError(f"Failed to start WebAPI server: {e}") from e


def _stop_webapi_server():
    """Stop the WebAPI server in DataLab."""
    global _webapi_info  # pylint: disable=global-statement
    # pylint: disable=import-outside-toplevel

    # Clear environment variables
    os.environ.pop("DATALAB_WORKSPACE_URL", None)
    os.environ.pop("DATALAB_WORKSPACE_TOKEN", None)

    try:
        from datalab.control.proxy import RemoteProxy

        proxy = RemoteProxy(autoconnect=False)
        proxy.connect(timeout=2.0)
        proxy.stop_webapi_server()
        proxy.disconnect()
    except Exception:  # pylint: disable=broad-exception-caught
        pass  # DataLab may already be closed

    _webapi_info = None


@pytest.fixture(scope="session")
def datalab_instance(request):
    """Session-scoped fixture that manages DataLab lifecycle for live tests.

    This fixture:
    - Starts DataLab if --start-datalab is provided
    - Always closes DataLab at the end of the live test session

    Usage:
        pytest --live                   # Uses existing DataLab, closes it after
        pytest --live --start-datalab   # Starts DataLab, closes it after

    For tests that need DataLab running, use this fixture.
    """
    if request.config.getoption("--start-datalab"):
        _start_datalab()

    # Start WebAPI server if --webapi flag is provided
    if request.config.getoption("--webapi"):
        _start_webapi_server()

    yield

    # Stop WebAPI server if it was started
    if request.config.getoption("--webapi"):
        _stop_webapi_server()

    # Always close DataLab when live tests are done
    if request.config.getoption("--live"):
        _close_datalab_session()


@pytest.fixture(scope="session")
def webapi_backend(
    request,
    datalab_instance,  # pylint: disable=unused-argument,redefined-outer-name
):
    """Session-scoped fixture providing a WebApiBackend connected to DataLab.

    This fixture:
    - Ensures DataLab is running with WebAPI server started
    - Returns a WebApiBackend instance connected to DataLab

    Requires --webapi flag to run.

    Note: datalab_instance is required as a dependency to ensure DataLab
    is running before this fixture runs, even though it's not directly used.
    """
    if not request.config.getoption("--webapi"):
        pytest.skip("Need --webapi option to run")

    # pylint: disable=import-outside-toplevel
    from datalab_kernel.backends.webapi import WebApiBackend

    return WebApiBackend()
