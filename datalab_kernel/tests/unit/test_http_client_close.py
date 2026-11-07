# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""Tests for Web API HTTP client cleanup."""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

import datalab_kernel.backends.pyfetch as pyfetch
import datalab_kernel.backends.webapi as webapi
from datalab_kernel.backends.pyfetch import (
    HttpResponse,
    HttpxClientWrapper,
    PyodideHttpClient,
)
from datalab_kernel.backends.webapi import WebApiBackend


@pytest.fixture(params=[False, True], ids=["httpx", "pyodide"])
def http_backend(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[WebApiBackend]:
    """Create a backend without connecting to a DataLab server."""
    in_pyodide = request.param
    monkeypatch.setattr(pyfetch, "is_pyodide", lambda: in_pyodide)
    monkeypatch.setattr(webapi, "is_pyodide", lambda: in_pyodide)

    native_client = None
    if in_pyodide:

        def request_status(_self, method: str, path: str, **_kwargs) -> HttpResponse:
            assert (method, path) == ("GET", "/api/v1/status")
            return HttpResponse(200, b'{"running": true}', {})

        monkeypatch.setattr(PyodideHttpClient, "_request", request_status)
    else:
        transport = httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"running": True})
        )
        native_client = httpx.Client(
            base_url="http://datalab.invalid", transport=transport
        )
        monkeypatch.setattr(httpx, "Client", lambda **_kwargs: native_client)

    try:
        backend = WebApiBackend(base_url="http://datalab.invalid")
        expected_type = PyodideHttpClient if in_pyodide else HttpxClientWrapper
        assert isinstance(backend._client, expected_type)
        yield backend
    finally:
        if native_client is not None:
            native_client.close()


def assert_native_client_closed(backend: WebApiBackend) -> None:
    """Check that the native transport has actually released its client."""
    if isinstance(backend._client, HttpxClientWrapper):
        assert backend._client._client.is_closed


@pytest.mark.standalone
class TestHttpClientClose:
    """Check backend cleanup through both HTTP client implementations."""

    def test_explicit_close(self, http_backend: WebApiBackend) -> None:
        """Closing a backend repeatedly succeeds and releases native resources."""
        assert http_backend.close() is None
        assert http_backend.close() is None
        assert_native_client_closed(http_backend)

    def test_context_manager_close(self, http_backend: WebApiBackend) -> None:
        """Leaving a context closes its HTTP client."""
        with http_backend as backend:
            assert backend is http_backend
        assert_native_client_closed(http_backend)

    def test_context_manager_preserves_error(self, http_backend: WebApiBackend) -> None:
        """Cleanup also occurs when the context body raises."""
        with pytest.raises(ValueError, match="context failed"), http_backend:
            raise ValueError("context failed")
        assert_native_client_closed(http_backend)

    def test_native_close_propagates_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Do not hide failures from the underlying HTTP client."""
        wrapper = HttpxClientWrapper("http://datalab.invalid", {})
        try:
            with monkeypatch.context() as patch:

                def fail_close() -> None:
                    raise RuntimeError("close failed")

                patch.setattr(wrapper._client, "close", fail_close)
                with pytest.raises(RuntimeError, match="close failed"):
                    wrapper.close()
        finally:
            wrapper._client.close()


@pytest.mark.webapi
class TestWebApiFixtureCleanup:
    """Verify that the live-test fixture closes its real HTTP client."""

    @pytest.fixture
    def observed_client(self) -> Iterator[tuple[httpx.Client, list[str]]]:
        """Provide an HTTP client with an in-memory server."""
        requests = []

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.path)
            if request.url.path == "/api/v1/status":
                return httpx.Response(200, json={"running": True})
            assert request.url.path == "/api/v1/objects"
            return httpx.Response(200, json={"objects": []})

        client = httpx.Client(
            base_url="http://datalab.invalid", transport=httpx.MockTransport(respond)
        )
        try:
            yield client, requests
            if requests:
                assert requests == [
                    "/api/v1/status",
                    "/api/v1/objects",
                    "/api/v1/objects",
                ]
                assert client.is_closed
        finally:
            client.close()

    @pytest.fixture
    def auto_datalab(
        self,
        monkeypatch: pytest.MonkeyPatch,
        observed_client: tuple[httpx.Client, list[str]],
    ) -> None:
        """Override DataLab startup with the in-memory HTTP transport."""
        client, _requests = observed_client
        monkeypatch.setenv("DATALAB_WORKSPACE_URL", "http://datalab.invalid")
        monkeypatch.delenv("DATALAB_WORKSPACE_TOKEN", raising=False)
        monkeypatch.setattr(pyfetch, "is_pyodide", lambda: False)
        monkeypatch.setattr(webapi, "is_pyodide", lambda: False)
        monkeypatch.setattr(httpx, "Client", lambda **_kwargs: client)

    def test_fixture_closes(
        self,
        webapi_backend: WebApiBackend,
        observed_client: tuple[httpx.Client, list[str]],
    ) -> None:
        """The fixture closes the same client it used during the test."""
        client, requests = observed_client
        assert isinstance(webapi_backend._client, HttpxClientWrapper)
        assert webapi_backend._client._client is client
        assert not client.is_closed
        assert requests == ["/api/v1/status", "/api/v1/objects"]
