# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Pyodide-compatible HTTP Client
==============================

This module provides an HTTP client that works in both native Python (using httpx)
and pyodide/WASM environments (using pyodide.http.pyfetch or js.fetch).

This allows the WebApiBackend to work seamlessly in JupyterLite.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def is_pyodide() -> bool:
    """Check if we're running in a pyodide environment."""
    return "pyodide" in sys.modules or sys.platform == "emscripten"


class HttpResponse:
    """Unified HTTP response wrapper."""

    def __init__(self, status_code: int, content: bytes, headers: dict[str, str]):
        self.status_code = status_code
        self.content = content
        self.headers = headers

    def json(self) -> Any:
        """Parse response as JSON."""
        return json.loads(self.content.decode("utf-8"))

    def raise_for_status(self) -> None:
        """Raise an exception for non-2xx status codes."""
        if self.status_code >= 400:
            raise HttpError(
                self.status_code, self.content.decode("utf-8", errors="replace")
            )


class HttpError(Exception):
    """HTTP error with status code."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}: {message}")


class PyodideHttpClient:
    """HTTP client for pyodide using pyfetch."""

    def __init__(self, base_url: str, headers: dict[str, str], timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = timeout

    def get(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous GET request (runs async internally)."""
        return self._sync_request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous POST request."""
        return self._sync_request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous PUT request."""
        return self._sync_request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous DELETE request."""
        return self._sync_request("DELETE", path, **kwargs)

    def _sync_request(self, method: str, path: str, **kwargs) -> HttpResponse:
        """Make a synchronous request using pyodide's sync pyfetch.

        In pyodide, we use the synchronous version of pyfetch which is available
        when running in a WebWorker (like xeus-python kernel).
        """
        try:
            # Try using pyodide.http.open_url for GET requests (simpler, sync)
            if method == "GET" and not kwargs:
                return self._simple_get(path)

            # For other requests, use the async pyfetch with run_sync
            return self._run_async_request(method, path, **kwargs)

        except Exception as e:
            raise HttpError(0, f"Request failed: {e}") from e

    def _simple_get(self, path: str) -> HttpResponse:
        """Simple synchronous GET using open_url (limited but sync)."""
        # open_url doesn't support custom headers, so we need pyfetch
        return self._run_async_request("GET", path)

    def _run_async_request(self, method: str, path: str, **kwargs) -> HttpResponse:
        """Run async pyfetch in a way that works in pyodide."""
        import asyncio

        try:
            # Check if we have pyodide's run_sync helper
            from pyodide.ffi import run_sync

            return run_sync(self._async_request(method, path, **kwargs))
        except ImportError:
            pass

        # Fallback: try to get or create event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in a running loop - use nest_asyncio or create_task
            import nest_asyncio

            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                self._async_request(method, path, **kwargs)
            )
        except (RuntimeError, ImportError):
            # No running loop or no nest_asyncio - just run it
            try:
                return asyncio.run(self._async_request(method, path, **kwargs))
            except RuntimeError:
                # Last resort: create new loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(
                        self._async_request(method, path, **kwargs)
                    )
                finally:
                    loop.close()

    async def _async_request(self, method: str, path: str, **kwargs) -> HttpResponse:
        """Make an async HTTP request using pyodide's pyfetch."""
        from pyodide.http import pyfetch

        url = f"{self._base_url}{path}"

        # Build fetch options
        fetch_options: dict[str, Any] = {
            "method": method,
            "headers": dict(self._headers),
        }

        # Handle content/data
        if "content" in kwargs:
            fetch_options["body"] = kwargs["content"]
            if "headers" not in fetch_options:
                fetch_options["headers"] = {}
            # Don't set content-type for binary data (NPZ)
        elif "json" in kwargs:
            fetch_options["body"] = json.dumps(kwargs["json"])
            fetch_options["headers"]["Content-Type"] = "application/json"

        # Add any extra headers
        if "headers" in kwargs:
            fetch_options["headers"].update(kwargs["headers"])

        # Make the request
        response = await pyfetch(url, **fetch_options)

        # Get response content
        content = await response.bytes()

        # Build header dict
        response_headers = {}
        # pyfetch response has headers as a dict-like object
        if hasattr(response, "headers"):
            for key in response.headers:
                response_headers[key.lower()] = response.headers.get(key)

        return HttpResponse(
            status_code=response.status,
            content=content,
            headers=response_headers,
        )


class HttpxClientWrapper:
    """Wrapper around httpx.Client to match our interface."""

    def __init__(self, base_url: str, headers: dict[str, str], timeout: float = 30.0):
        import httpx

        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        )

    def get(self, path: str, **kwargs) -> HttpResponse:
        """GET request."""
        response = self._client.get(path, **kwargs)
        return self._wrap_response(response)

    def post(self, path: str, **kwargs) -> HttpResponse:
        """POST request."""
        response = self._client.post(path, **kwargs)
        return self._wrap_response(response)

    def put(self, path: str, **kwargs) -> HttpResponse:
        """PUT request."""
        response = self._client.put(path, **kwargs)
        return self._wrap_response(response)

    def delete(self, path: str, **kwargs) -> HttpResponse:
        """DELETE request."""
        response = self._client.delete(path, **kwargs)
        return self._wrap_response(response)

    def _wrap_response(self, response) -> HttpResponse:
        """Wrap httpx response in our unified type."""
        return HttpResponse(
            status_code=response.status_code,
            content=response.content,
            headers=dict(response.headers),
        )


def create_http_client(
    base_url: str, token: str, timeout: float = 30.0
) -> PyodideHttpClient | HttpxClientWrapper:
    """Create an HTTP client appropriate for the current environment.

    In pyodide/WASM (JupyterLite), uses pyfetch.
    In native Python, uses httpx.

    Args:
        base_url: Base URL for the API
        token: Bearer token for authentication
        timeout: Request timeout in seconds

    Returns:
        HTTP client instance
    """
    headers = {"Authorization": f"Bearer {token}"}

    if is_pyodide():
        return PyodideHttpClient(base_url, headers, timeout)
    else:
        try:
            return HttpxClientWrapper(base_url, headers, timeout)
        except ImportError:
            raise ImportError(
                "httpx is required for WebApiBackend in native Python. "
                "Install with: pip install httpx"
            ) from None
