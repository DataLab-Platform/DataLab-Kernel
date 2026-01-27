# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Pytest configuration for DataLab Kernel tests.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time

import httpx
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
        help="Force live mode tests only (skip standalone-only tests). "
        "If DataLab is not running, tests will fail.",
    )
    parser.addoption(
        "--standalone-only",
        action="store_true",
        default=False,
        help="Run only standalone tests (skip live tests). "
        "Useful for quick testing without DataLab.",
    )
    parser.addoption(
        "--start-datalab",
        action="store_true",
        default=False,
        help="Automatically start DataLab for live tests. Implies --webapi.",
    )
    parser.addoption(
        "--webapi",
        action="store_true",
        default=False,
        help="Run WebAPI integration tests (starts WebAPI server in DataLab). "
        "Requires DataLab to be running.",
    )


def pytest_collection_modifyitems(config, items):
    """Reorder and configure tests for comprehensive execution.

    Default behavior (no flags):
    1. Run all standalone tests first (no DataLab needed)
    2. Start DataLab automatically when first live test is encountered
    3. Run all live/webapi tests

    Result: Complete test coverage in one pytest run.

    Flags:
    - --standalone-only: Skip all live/webapi tests (quick testing)
    - --live: Force live mode only, skip standalone tests
    - --start-datalab: Explicitly start DataLab at session start
    - --webapi: Enable WebAPI tests (auto-enabled in default mode)
    """
    standalone_only_mode = config.getoption("--standalone-only")
    force_live_mode = config.getoption("--live")

    if force_live_mode:
        # --live flag: Skip standalone-only tests
        skip_standalone = pytest.mark.skip(
            reason="Standalone-only test skipped in --live mode"
        )
        for item in items:
            if "standalone" in item.keywords:
                item.add_marker(skip_standalone)

    elif standalone_only_mode:
        # --standalone-only flag: Skip all live and webapi tests
        skip_live = pytest.mark.skip(reason="Live test skipped with --standalone-only")
        skip_webapi = pytest.mark.skip(
            reason="WebAPI test skipped with --standalone-only"
        )
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
            if "webapi" in item.keywords:
                item.add_marker(skip_webapi)

    else:
        # Default mode: Run everything with smart ordering
        # Reorder tests: standalone tests first, then live tests
        standalone_tests = []
        live_tests = []
        other_tests = []

        for item in items:
            if "live" in item.keywords or "webapi" in item.keywords:
                live_tests.append(item)
            elif "standalone" in item.keywords:
                standalone_tests.append(item)
            else:
                # Tests without specific markers (unit tests, contract tests)
                # These can run in standalone mode
                other_tests.append(item)

        # Reorder: standalone + other tests first, then live tests
        items[:] = standalone_tests + other_tests + live_tests


# Global process handle for DataLab
# pylint: disable=invalid-name
_datalab_process = None

# Fixed WebAPI configuration for tests
WEBAPI_HOST = "127.0.0.1"
WEBAPI_PORT = 18080  # Use non-standard port to avoid conflicts
WEBAPI_TOKEN = "datalab-kernel-test-token"
WEBAPI_URL = f"http://{WEBAPI_HOST}:{WEBAPI_PORT}"


def _is_datalab_running():
    """Check if DataLab WebAPI is running via HTTP.

    Returns True if the WebAPI server responds to a status request.
    """
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{WEBAPI_URL}/api/v1/status")
            return response.status_code == 200
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def _start_datalab():
    """Start DataLab in background with WebAPI auto-enabled.

    Uses environment variables to configure DataLab to:
    - Auto-start WebAPI server at a known port
    - Use a known authentication token
    - Run in unattended mode

    This eliminates the need for XML-RPC to configure WebAPI after startup.
    """
    global _datalab_process  # pylint: disable=global-statement

    if _is_datalab_running():
        return  # Already running

    # Set environment variables for DataLab
    env = os.environ.copy()
    # Prevent auto-quit and enable unattended mode
    env["DATALAB_DO_NOT_QUIT"] = "1"
    env["GUIDATA_UNATTENDED"] = "1"
    # Auto-start WebAPI with known configuration
    env["DATALAB_WEBAPI_ENABLED"] = "1"
    env["DATALAB_WEBAPI_HOST"] = WEBAPI_HOST
    env["DATALAB_WEBAPI_PORT"] = str(WEBAPI_PORT)
    env["DATALAB_WEBAPI_TOKEN"] = WEBAPI_TOKEN

    # Also set the workspace environment variables so WebApiBackend can find them
    os.environ["DATALAB_WORKSPACE_URL"] = WEBAPI_URL
    os.environ["DATALAB_WORKSPACE_TOKEN"] = WEBAPI_TOKEN

    # Try to find the correct Python executable for DataLab
    # In development, DataLab may be in a different virtualenv
    python_exe = sys.executable

    # Check if there's a DataLab virtualenv nearby
    # Look for ../DataLab/.venv/Scripts/python.exe (Windows)
    # or ../DataLab/.venv/bin/python (Unix)
    datalab_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DataLab")
    )
    if os.path.isdir(datalab_root):
        if sys.platform == "win32":
            datalab_python = os.path.join(
                datalab_root, ".venv", "Scripts", "python.exe"
            )
        else:
            datalab_python = os.path.join(datalab_root, ".venv", "bin", "python")
        if os.path.isfile(datalab_python):
            python_exe = datalab_python

    # Start DataLab in background
    # pylint: disable=consider-using-with
    _datalab_process = subprocess.Popen(
        [python_exe, "-c", "from datalab.app import run; run()"],
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
    """Stop DataLab if we started it.

    Uses process termination since WebAPI doesn't have a shutdown endpoint.
    """
    global _datalab_process  # pylint: disable=global-statement

    if _datalab_process is not None:
        try:
            # Terminate gracefully
            _datalab_process.terminate()
            _datalab_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Force kill if graceful termination fails
            _datalab_process.kill()
            _datalab_process.wait(timeout=3)
        except Exception:  # pylint: disable=broad-exception-caught
            pass
        finally:
            _datalab_process = None
            # Clean up environment variables
            os.environ.pop("DATALAB_WORKSPACE_URL", None)
            os.environ.pop("DATALAB_WORKSPACE_TOKEN", None)


@pytest.fixture(scope="session")
def datalab_instance(request):
    """Session-scoped fixture that manages DataLab lifecycle.

    Default mode (no flags):
    - DataLab is started lazily by auto_datalab fixture when first live test runs
    - Nothing happens here

    With --start-datalab flag:
    - Starts DataLab immediately at session start
    - Sets up environment variables for WebAPI connection

    With --standalone-only flag:
    - Does nothing (DataLab not needed)
    """
    standalone_only = request.config.getoption("--standalone-only")
    force_live = request.config.getoption("--live")
    explicit_start = request.config.getoption("--start-datalab")

    # Determine if we should manage DataLab
    should_manage_datalab = not standalone_only

    # Determine when to start DataLab
    start_immediately = explicit_start or force_live

    datalab_started_by_us = False

    if should_manage_datalab and start_immediately:
        # Set environment variables for WebAPI connection
        os.environ["DATALAB_WORKSPACE_URL"] = WEBAPI_URL
        os.environ["DATALAB_WORKSPACE_TOKEN"] = WEBAPI_TOKEN

        # Start DataLab at session start (explicit --start-datalab or --live)
        print("\n" + "=" * 70)
        print("🚀 Starting DataLab (explicit mode)...")
        print("=" * 70)
        _start_datalab()
        print("✅ DataLab ready")
        print("=" * 70 + "\n")
        datalab_started_by_us = True

    yield

    # Cleanup: stop DataLab if we started it (explicitly or lazily)
    if datalab_started_by_us or _lazy_datalab_started:
        _stop_datalab()


# Global flag to track if we've started DataLab lazily
_lazy_datalab_started = False


# Track if we've detected that DataLab cannot be started (CI environment)
_datalab_start_failed = False


@pytest.fixture(scope="session")
def auto_datalab(request, datalab_instance):  # pylint: disable=W0621,W0613
    """Lazy DataLab starter - starts DataLab when first live test runs.

    This fixture is automatically used by live tests in default mode.
    It ensures DataLab is started only after standalone tests complete.

    Also sets environment variables for WebAPI connection.

    In CI environments (no display), skips tests gracefully instead of failing.
    """
    global _lazy_datalab_started, _datalab_start_failed  # pylint: disable=global-statement

    standalone_only = request.config.getoption("--standalone-only")
    explicit_start = request.config.getoption("--start-datalab")
    force_live = request.config.getoption("--live")

    # Don't start if standalone-only mode
    if standalone_only:
        pytest.skip("Live tests skipped in standalone-only mode")

    # If we previously failed to start DataLab (e.g., CI environment), skip
    if _datalab_start_failed:
        pytest.skip("DataLab not available (previous start attempt failed)")

    # Set environment variables for WebAPI connection
    # These are used by WebApiBackend to find the server
    os.environ["DATALAB_WORKSPACE_URL"] = WEBAPI_URL
    os.environ["DATALAB_WORKSPACE_TOKEN"] = WEBAPI_TOKEN

    # Don't start if already started explicitly
    if explicit_start or force_live:
        return

    # Start DataLab lazily on first live test
    if not _lazy_datalab_started:
        _lazy_datalab_started = True
        print("\n" + "=" * 70)
        print("🚀 Starting DataLab for live tests...")
        print("=" * 70)
        if not _is_datalab_running():
            try:
                _start_datalab()
            except RuntimeError as exc:
                # Failed to start DataLab - likely CI environment without display
                _datalab_start_failed = True
                pytest.skip(
                    f"DataLab cannot be started (CI/headless environment?): {exc}"
                )
            # _start_datalab() already waits for WebAPI to be ready
        print("✅ DataLab ready for live tests")
        print("=" * 70 + "\n")


@pytest.fixture
def live_workspace(auto_datalab):  # pylint: disable=W0621,W0613
    """Create a live workspace connected to DataLab.

    In default mode: Automatically starts DataLab when first used.
    With --start-datalab: Uses already-running DataLab from session start.

    Uses Workspace() with auto-detection, which will prefer WebAPI
    if DATALAB_WORKSPACE_URL is set (configured by auto_datalab fixture).

    Skips test if connection fails.
    """
    global _datalab_start_failed  # pylint: disable=global-statement

    # Skip if DataLab start failed previously
    if _datalab_start_failed:
        pytest.skip("DataLab not available (previous start attempt failed)")

    # pylint: disable=import-outside-toplevel
    from datalab_kernel.workspace import Workspace, WorkspaceMode

    try:
        workspace = Workspace()
    except ConnectionError as exc:
        _datalab_start_failed = True
        pytest.skip(f"Cannot connect to DataLab: {exc}")

    if workspace.mode != WorkspaceMode.LIVE:
        pytest.skip("Expected live mode but got standalone (DataLab not available)")

    # Clear DataLab before test (with error handling)
    with contextlib.suppress(Exception):
        workspace.clear()
    yield workspace
    # Cleanup after test (with error handling to prevent test failures)
    with contextlib.suppress(Exception):
        workspace.clear()


@pytest.fixture
def webapi_backend(request, auto_datalab):  # pylint: disable=W0621,W0613
    """Function-scoped fixture providing a WebApiBackend connected to DataLab.

    In default mode: Uses auto-started DataLab.
    With flags: Uses DataLab started by datalab_instance fixture.

    Returns a WebApiBackend instance connected to DataLab.
    Cleans up objects before and after each test to prevent contamination.

    Skips test if running in standalone-only mode or if connection fails.
    """
    global _datalab_start_failed  # pylint: disable=global-statement

    # Skip if standalone-only mode (no DataLab available)
    if request.config.getoption("--standalone-only"):
        pytest.skip(
            "WebApiBackend tests require DataLab (skipped in standalone-only mode)"
        )

    # Skip if DataLab start failed previously
    if _datalab_start_failed:
        pytest.skip("DataLab not available (previous start attempt failed)")

    # pylint: disable=import-outside-toplevel
    from datalab_kernel.backends.webapi import WebApiBackend

    try:
        backend = WebApiBackend()
    except ConnectionError as exc:
        # Connection failed - mark as failed and skip
        _datalab_start_failed = True
        pytest.skip(f"Cannot connect to DataLab WebAPI: {exc}")

    # Clear before test
    with contextlib.suppress(Exception):
        backend.clear()
    yield backend
    # Cleanup after test
    with contextlib.suppress(Exception):
        backend.clear()
