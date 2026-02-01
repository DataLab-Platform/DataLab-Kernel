# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Connection Discovery
====================

This module provides auto-discovery of DataLab Web API connections.

Discovery methods (in priority order):
1. Environment variables (DATALAB_WORKSPACE_URL, DATALAB_WORKSPACE_TOKEN)
2. Connection file written by DataLab (~/.config/datalab/webapi_connection.json)
3. URL query parameters (for JupyterLite: ?datalab_url=...&datalab_token=...)
4. Well-known port probing (http://127.0.0.1:18080 with /api/v1/status check)

Usage::

    from datalab_kernel.discovery import discover_connection

    # Returns (url, token) or (None, None) if not found
    url, token = discover_connection()
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
from pathlib import Path

# Default port used by DataLab Web API
DEFAULT_WEBAPI_PORT = 18080
DEFAULT_WEBAPI_URL = f"http://127.0.0.1:{DEFAULT_WEBAPI_PORT}"

logger = logging.getLogger("datalab-kernel")


def is_pyodide() -> bool:
    """Check if running in Pyodide/browser environment."""
    return "pyodide" in sys.modules or sys.platform == "emscripten"


def get_connection_file_path() -> Path:
    """Get the path to the connection info file.

    The file is stored in a platform-specific location:
    - Windows: %APPDATA%/DataLab/webapi_connection.json
    - Linux/Mac: ~/.config/datalab/webapi_connection.json

    Returns:
        Path to the connection file.
    """
    if os.name == "nt":
        # Windows: use APPDATA
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        # Linux/Mac: use XDG_CONFIG_HOME or ~/.config
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))

    return base / "datalab" / "webapi_connection.json"


def _discover_from_env() -> tuple[str | None, str | None]:
    """Discover connection from environment variables.

    Returns:
        Tuple of (url, token) or (None, None) if not set.
    """
    url = os.environ.get("DATALAB_WORKSPACE_URL")
    token = os.environ.get("DATALAB_WORKSPACE_TOKEN")
    if url and token:
        return url, token
    return None, None


def _discover_from_file() -> tuple[str | None, str | None]:
    """Discover connection from connection file.

    Returns:
        Tuple of (url, token) or (None, None) if file not found or invalid.
    """
    if is_pyodide():
        # No filesystem access in browser
        return None, None

    try:
        connection_file = get_connection_file_path()
        if not connection_file.exists():
            return None, None

        data = json.loads(connection_file.read_text())
        url = data.get("url")
        token = data.get("token")

        if url and token:
            # Verify the process is still running (optional check)
            pid = data.get("pid")
            if pid and not _is_process_running(pid):
                # Stale connection file, remove it
                with contextlib.suppress(Exception):
                    connection_file.unlink()
                return None, None
            return url, token
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return None, None


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running.

    Args:
        pid: Process ID to check.

    Returns:
        True if process is running, False otherwise.
    """
    try:
        if os.name == "nt":
            # Windows: use tasklist
            import subprocess

            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return str(pid) in result.stdout
        else:
            # Unix: check /proc or use os.kill with signal 0
            os.kill(pid, 0)
            return True
    except (OSError, subprocess.SubprocessError):
        return False


def _discover_from_url_params() -> tuple[str | None, str | None]:
    """Discover connection from URL query parameters (JupyterLite).

    Looks for ?datalab_url=...&datalab_token=... in the browser URL.

    Returns:
        Tuple of (url, token) or (None, None) if not in browser or not set.
    """
    if not is_pyodide():
        return None, None

    try:
        # Access browser URL via JavaScript
        # pylint: disable=import-outside-toplevel
        from urllib.parse import parse_qs, urlparse

        from js import window  # type: ignore[import-not-found]

        current_url = str(window.location.href)
        parsed = urlparse(current_url)
        params = parse_qs(parsed.query)

        url = params.get("datalab_url", [None])[0]
        token = params.get("datalab_token", [None])[0]

        if url and token:
            return url, token
    except Exception:  # pylint: disable=broad-exception-caught
        pass

    return None, None


def _probe_wellknown_port(timeout: float = 2.0) -> tuple[str | None, str | None]:
    """Probe well-known port for DataLab Web API.

    Checks if DataLab is running on the default port (18080) by calling
    the /api/v1/status endpoint (which doesn't require authentication).

    Note: This only finds the URL, not the token. Useful when localhost
    token bypass is enabled on the server.

    Args:
        timeout: Connection timeout in seconds.

    Returns:
        Tuple of (url, None) if server found, or (None, None) if not.
    """
    url = DEFAULT_WEBAPI_URL
    logger.debug(f"Probing well-known port: {url}")

    if is_pyodide():
        # Use XMLHttpRequest synchronously in browser
        # pyfetch is async and can't be used in synchronous context
        try:
            # pylint: disable=import-outside-toplevel
            from js import XMLHttpRequest  # type: ignore[import-not-found]

            xhr = XMLHttpRequest.new()
            # Open synchronous request (False = synchronous)
            xhr.open("GET", f"{url}/api/v1/status", False)
            xhr.timeout = int(timeout * 1000)  # Convert to milliseconds
            try:
                xhr.send()
                logger.debug(f"Port probe response: status={xhr.status}")
                if xhr.status == 200:
                    logger.info(f"Found DataLab at {url}")
                    return url, None
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Network error or timeout - likely CORS or Private Network Access block
                logger.debug(f"Port probe network error: {e}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug(f"Port probe XMLHttpRequest error: {e}")
    else:
        # Use httpx in native Python
        try:
            # pylint: disable=import-outside-toplevel
            import httpx

            with httpx.Client(timeout=timeout) as client:
                response = client.get(f"{url}/api/v1/status")
                logger.debug(f"Port probe response: status={response.status_code}")
                if response.status_code == 200:
                    logger.info(f"Found DataLab at {url}")
                    return url, None
        except Exception:  # pylint: disable=broad-exception-caught
            pass

    return None, None


def discover_connection(
    probe_port: bool = True, timeout: float = 2.0
) -> tuple[str | None, str | None]:
    """Discover DataLab Web API connection.

    Tries multiple discovery methods in priority order:
    1. Environment variables (DATALAB_WORKSPACE_URL, DATALAB_WORKSPACE_TOKEN)
    2. Connection file (~/.config/datalab/webapi_connection.json)
    3. URL query parameters (JupyterLite: ?datalab_url=...&datalab_token=...)
    4. Well-known port probing (http://127.0.0.1:18080)

    Args:
        probe_port: Whether to probe the well-known port if other methods fail.
        timeout: Timeout for port probing in seconds.

    Returns:
        Tuple of (url, token). Token may be None if only URL was discovered
        (e.g., via port probing when localhost bypass is enabled).
        Returns (None, None) if discovery fails completely.

    Example::

        url, token = discover_connection()
        if url:
            workspace.connect(url, token)
    """
    # 1. Environment variables (highest priority)
    url, token = _discover_from_env()
    if url:
        return url, token

    # 2. Connection file (native Python only)
    url, token = _discover_from_file()
    if url:
        return url, token

    # 3. URL query parameters (JupyterLite only)
    url, token = _discover_from_url_params()
    if url:
        return url, token

    # 4. Well-known port probing (optional)
    if probe_port:
        url, token = _probe_wellknown_port(timeout)
        if url:
            return url, token

    return None, None
