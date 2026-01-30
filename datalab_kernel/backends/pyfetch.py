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
    """HTTP client for pyodide using synchronous XMLHttpRequest.

    In pyodide/JupyterLite, we use JavaScript's synchronous XMLHttpRequest
    because async pyfetch doesn't work well with synchronous Python code.

    PERFORMANCE NOTE (v0.2.5):
    This implementation uses optimized binary data handling:
    - responseType="arraybuffer" for efficient binary responses
    - Direct Uint8Array creation from bytes for efficient uploads
    - Avoids O(n) Python loops that caused slow image transfers
    """

    def __init__(self, base_url: str, headers: dict[str, str], timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._headers = headers
        self._timeout = int(timeout * 1000)  # Convert to milliseconds

    def get(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous GET request."""
        return self._request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous POST request."""
        return self._request("POST", path, **kwargs)

    def put(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous PUT request."""
        return self._request("PUT", path, **kwargs)

    def delete(self, path: str, **kwargs) -> HttpResponse:
        """Synchronous DELETE request."""
        return self._request("DELETE", path, **kwargs)

    @staticmethod
    def _arraybuffer_to_bytes(arraybuffer) -> bytes:
        """Convert a JavaScript ArrayBuffer to Python bytes efficiently.

        Uses Pyodide's optimized conversion when available, falling back
        to manual conversion for compatibility.

        Args:
            arraybuffer: JavaScript ArrayBuffer object

        Returns:
            Python bytes object
        """
        import js  # pyodide's JavaScript bridge

        if arraybuffer is None:
            return b""

        # Create a Uint8Array view of the ArrayBuffer
        uint8_view = js.Uint8Array.new(arraybuffer)

        # Use Pyodide's to_py() for efficient conversion if available
        # This is the fastest path in modern Pyodide versions
        if hasattr(uint8_view, "to_py"):
            return bytes(uint8_view.to_py())

        # Fallback: use tobytes() if available (older Pyodide)
        if hasattr(uint8_view, "tobytes"):
            return uint8_view.tobytes()

        # Last resort fallback: manual conversion (slow but works)
        return bytes([uint8_view[i] for i in range(uint8_view.length)])

    def _request(self, method: str, path: str, **kwargs) -> HttpResponse:
        """Make a synchronous HTTP request using JavaScript XMLHttpRequest.

        This works in pyodide because we can call synchronous JS APIs.

        PERFORMANCE OPTIMIZATION (v1.1):
        - Uses responseType="arraybuffer" for efficient binary response handling
        - Uses Uint8Array with memoryview for efficient binary request upload
        - Avoids O(n) Python loops that were causing slow image transfers
        """
        import js  # pyodide's JavaScript bridge

        url = f"{self._base_url}{path}"

        # Create XMLHttpRequest
        xhr = js.XMLHttpRequest.new()
        xhr.open(method, url, False)  # False = synchronous
        xhr.timeout = self._timeout

        # Set headers
        for key, value in self._headers.items():
            xhr.setRequestHeader(key, value)

        # Handle request body
        body = None
        if "content" in kwargs:
            body = kwargs["content"]
            # OPTIMIZED: Use Uint8Array.new() with the bytes object directly
            # Pyodide automatically handles the conversion efficiently
            if isinstance(body, bytes):
                body = js.Uint8Array.new(body)
        elif "json" in kwargs:
            body = json.dumps(kwargs["json"])
            xhr.setRequestHeader("Content-Type", "application/json")

        # Add any extra headers
        if "headers" in kwargs:
            for key, value in kwargs["headers"].items():
                xhr.setRequestHeader(key, value)

        # OPTIMIZED: Use arraybuffer responseType for efficient binary data handling
        # This avoids the slow text-to-bytes conversion
        xhr.responseType = "arraybuffer"

        try:
            # Send request
            if body is not None:
                xhr.send(body)
            else:
                xhr.send()
        except Exception as e:
            raise HttpError(0, f"Request failed: {e}") from e

        # Check for network errors
        if xhr.status == 0:
            raise HttpError(0, "Network error - request blocked or server unreachable")

        # OPTIMIZED: Get response as bytes directly from ArrayBuffer
        # This is much faster than the old text-based approach
        content = self._arraybuffer_to_bytes(xhr.response)

        # Parse response headers
        response_headers = {}
        header_str = xhr.getAllResponseHeaders() or ""
        for line in header_str.split("\r\n"):
            if ": " in line:
                key, value = line.split(": ", 1)
                response_headers[key.lower()] = value

        return HttpResponse(
            status_code=xhr.status,
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
