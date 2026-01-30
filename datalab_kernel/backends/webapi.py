# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Backend
===============

HTTP/JSON backend for synchronizing with a DataLab instance via the Web API.

This backend uses:

- HTTP/JSON for metadata operations (list, get metadata, delete)
- NPZ binary format for data transfer (efficient for large arrays)
- Bearer token authentication

Usage
-----

Set environment variables::

    DATALAB_WORKSPACE_URL=http://127.0.0.1:8080
    DATALAB_WORKSPACE_TOKEN=your-token-here

Then in your notebook::

    from datalab_kernel import workspace

    # Auto-detection will use WebApiBackend if URL is set
    workspace.list()
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import TYPE_CHECKING
from urllib.parse import quote

import h5py
import numpy as np
from sigima import ImageObj, SignalObj

from datalab_kernel.backends.pyfetch import (
    HttpError,
    HttpResponse,
    create_http_client,
    is_pyodide,
)
from datalab_kernel.serialization_npz import (
    deserialize_object_from_npz,
    serialize_object_to_npz,
)

if TYPE_CHECKING:
    DataObject = SignalObj | ImageObj


class WebApiBackend:
    """Web API backend for DataLab synchronization.

    This backend connects to a running DataLab instance via the HTTP/JSON
    Web API, providing efficient workspace synchronization using NPZ format
    for binary data transfer.

    Attributes:
        base_url: Base URL of the DataLab Web API server.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize the Web API backend.

        Args:
            base_url: Base URL of the DataLab Web API (e.g., "http://127.0.0.1:8080").
                If None, reads from DATALAB_WORKSPACE_URL environment variable.
            token: Authentication token. If None, reads from DATALAB_WORKSPACE_TOKEN.
            timeout: Request timeout in seconds.

        Raises:
            ImportError: If httpx is not installed (native Python only).
            ValueError: If URL or token is not provided.
        """
        self._base_url = base_url or os.environ.get("DATALAB_WORKSPACE_URL")
        self._token = token or os.environ.get("DATALAB_WORKSPACE_TOKEN")
        self._timeout = timeout

        if not self._base_url:
            raise ValueError(
                "DataLab Web API URL not provided. "
                "Set DATALAB_WORKSPACE_URL environment variable or pass base_url."
            )

        if not self._token:
            raise ValueError(
                "DataLab Web API token not provided. "
                "Set DATALAB_WORKSPACE_TOKEN environment variable or pass token."
            )

        # Ensure base URL doesn't have trailing slash
        self._base_url = self._base_url.rstrip("/")

        # Create HTTP client (uses pyfetch in pyodide, httpx in native Python)
        self._client = create_http_client(self._base_url, self._token, self._timeout)

        # Verify connection
        self._verify_connection()

    @property
    def base_url(self) -> str:
        """Return the base URL of the DataLab Web API."""
        return self._base_url

    def _verify_connection(self) -> None:
        """Verify connection to the DataLab server.

        Retries up to 10 times with exponential backoff to allow time for
        the Uvicorn server to start accepting connections.
        """
        # In pyodide (browser), we can't use time.sleep and should retry fewer times
        in_pyodide = is_pyodide()
        max_retries = 1 if in_pyodide else 10  # Single attempt in browser
        base_delay = 0.1  # Start with 100ms

        for attempt in range(max_retries):
            try:
                response = self._client.get("/api/v1/status")
                response.raise_for_status()
                data = response.json()
                if not data.get("running"):
                    raise ConnectionError("DataLab Web API is not running")
                return  # Success!
            except HttpError as e:
                if attempt == max_retries - 1:
                    # Last attempt failed
                    raise ConnectionError(
                        f"Failed to connect to DataLab Web API after {max_retries} attempts: {e}"
                    ) from e

                # Wait before retrying (only in native Python)
                if not in_pyodide:
                    import time

                    delay = base_delay * (2**attempt)
                    time.sleep(delay)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ConnectionError(
                        f"Failed to connect to DataLab Web API: {e}"
                    ) from e
                if not in_pyodide:
                    import time

                    delay = base_delay * (2**attempt)
                    time.sleep(delay)

    def _raise_for_status(self, response: HttpResponse) -> None:
        """Check response status and raise appropriate exceptions.

        Provides user-friendly error messages for common HTTP errors.
        """
        if response.status_code == 401:
            raise PermissionError("Invalid authentication token")

        if response.status_code == 404:
            # Extract object name from error message if possible
            try:
                data = response.json()
                raise KeyError(data.get("detail", "Object not found")) from None
            except KeyError:
                raise
            except Exception:
                raise KeyError("Object not found") from None

        if response.status_code == 409:
            try:
                data = response.json()
                raise ValueError(data.get("detail", "Conflict")) from None
            except ValueError:
                raise
            except Exception:
                raise ValueError("Object already exists") from None

        if response.status_code == 500:
            # Internal server error - try to extract detail
            try:
                data = response.json()
                detail = data.get("detail", "Internal server error")
                raise RuntimeError(f"DataLab server error: {detail}") from None
            except RuntimeError:
                raise
            except Exception:
                raise RuntimeError("DataLab server error (HTTP 500)") from None

        response.raise_for_status()

    def list(self) -> list[str]:
        """List all object names in the workspace.

        Returns:
            List of object names.
        """
        return [obj["name"] for obj in self._list_objects_raw()]

    def _list_objects_raw(self) -> list[dict]:
        """List all objects with full metadata.

        Returns:
            List of object dictionaries with 'name', 'type', etc.
        """
        response = self._client.get("/api/v1/objects")
        self._raise_for_status(response)
        data = response.json()
        return data.get("objects", [])

    def _resolve_short_id(self, short_id: str) -> str | None:
        """Resolve a short ID (e.g., 's001', 'i002') to an object name.

        Short IDs follow the pattern:
        - 's001', 's002', ... for signals (1-indexed)
        - 'i001', 'i002', ... for images (1-indexed)

        Args:
            short_id: Short ID string.

        Returns:
            Object name if found, None otherwise.
        """
        match = re.match(r"^([si])(\d{3})$", short_id.lower())
        if not match:
            return None

        obj_type = "signal" if match.group(1) == "s" else "image"
        index = int(match.group(2)) - 1  # Convert to 0-indexed

        if index < 0:
            return None

        # Get objects of the specified type
        objects = self._list_objects_raw()
        typed_objects = [obj for obj in objects if obj.get("type") == obj_type]

        if index >= len(typed_objects):
            return None

        return typed_objects[index]["name"]

    def _encode_name(self, name: str) -> str:
        """URL-encode an object name for use in API paths.

        Args:
            name: Object name/title.

        Returns:
            URL-encoded name safe for use in path segments.
        """
        # Encode special characters but preserve forward slashes aren't expected
        return quote(name, safe="")

    def get(self, name: str, *, compress: bool = False) -> DataObject:
        """Retrieve an object by name or short ID.

        Args:
            name: Object name/title, or short ID (e.g., 's001' for first signal,
                'i002' for second image).
            compress: If True, request compressed NPZ data from server.
                Default is False for faster transfer (10-30x faster serialization
                on server side with ~10% size increase).

        Returns:
            SignalObj or ImageObj.

        Raises:
            KeyError: If object not found.
        """
        # Try to resolve as short ID first
        resolved_name = self._resolve_short_id(name)
        if resolved_name is not None:
            name = resolved_name

        # URL-encode the name to handle special characters
        encoded_name = self._encode_name(name)
        # Request uncompressed by default for faster transfer
        url = f"/api/v1/objects/{encoded_name}/data?compress={str(compress).lower()}"
        response = self._client.get(url)
        self._raise_for_status(response)
        return deserialize_object_from_npz(response.content)

    def get_metadata(self, name: str) -> dict:
        """Get object metadata without data.

        Args:
            name: Object name/title, or short ID.

        Returns:
            Metadata dictionary.

        Raises:
            KeyError: If object not found.
        """
        # Try to resolve as short ID first
        resolved_name = self._resolve_short_id(name)
        if resolved_name is not None:
            name = resolved_name

        encoded_name = self._encode_name(name)
        response = self._client.get(f"/api/v1/objects/{encoded_name}")
        self._raise_for_status(response)
        return response.json()

    def add(self, name: str, obj: DataObject, overwrite: bool = False) -> None:
        """Add an object to the workspace.

        Args:
            name: Object name/title.
            obj: SignalObj or ImageObj to add.
            overwrite: If True, replace existing object.

        Raises:
            ValueError: If object exists and overwrite is False.
        """
        # Set title on object
        obj.title = name

        # Serialize to NPZ
        npz_data = serialize_object_to_npz(obj)

        # URL-encode the name for the path
        encoded_name = self._encode_name(name)

        # Upload
        response = self._client.put(
            f"/api/v1/objects/{encoded_name}/data",
            content=npz_data,
            params={"overwrite": str(overwrite).lower()},
            headers={
                "Content-Type": "application/x-npz",
                "Authorization": f"Bearer {self._token}",
            },
        )
        self._raise_for_status(response)

    def remove(self, name: str) -> None:
        """Remove an object from the workspace.

        Args:
            name: Object name/title, or short ID.

        Raises:
            KeyError: If object not found.
        """
        # Try to resolve as short ID first
        resolved_name = self._resolve_short_id(name)
        if resolved_name is not None:
            name = resolved_name

        encoded_name = self._encode_name(name)
        response = self._client.delete(f"/api/v1/objects/{encoded_name}")
        self._raise_for_status(response)

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an object.

        Args:
            old_name: Current object name.
            new_name: New object name.

        Raises:
            KeyError: If old_name not found.
            ValueError: If new_name already exists.
        """
        # WebAPI doesn't have a direct rename endpoint, so we:
        # 1. Get the object
        # 2. Add with new name
        # 3. Delete old name
        obj = self.get(old_name)
        self.add(new_name, obj, overwrite=False)
        self.remove(old_name)

    def exists(self, name: str) -> bool:
        """Check if an object exists.

        Args:
            name: Object name/title, or short ID.

        Returns:
            True if object exists.
        """
        # Check for short ID resolution
        resolved_name = self._resolve_short_id(name)
        if resolved_name is not None:
            return True  # If we could resolve it, it exists
        return name in self.list()

    def clear(self) -> None:
        """Remove all objects from the workspace."""
        for name in self.list():
            with contextlib.suppress(KeyError):
                self.remove(name)

    def set_metadata(self, name: str, metadata: dict) -> None:
        """Update object metadata.

        Args:
            name: Object name/title, or short ID.
            metadata: Metadata fields to update.

        Raises:
            KeyError: If object not found.
        """
        # Try to resolve as short ID first
        resolved_name = self._resolve_short_id(name)
        if resolved_name is not None:
            name = resolved_name

        encoded_name = self._encode_name(name)
        response = self._client.patch(
            f"/api/v1/objects/{encoded_name}/metadata",
            json=metadata,
        )
        self._raise_for_status(response)

    def save(self, filepath: str) -> None:
        """Save workspace to HDF5 file.

        Note: This is a local operation - downloads all objects and saves locally.

        Args:
            filepath: Path to save HDF5 file.
        """
        # Download all objects and save locally
        if not filepath.endswith(".h5"):
            filepath = filepath + ".h5"

        with h5py.File(filepath, "w") as f:
            f.attrs["datalab_kernel_version"] = "0.1.0"
            f.attrs["format_version"] = "1.0"

            for name in self.list():
                obj = self.get(name)
                grp = f.create_group(name)
                self._save_object_to_group(grp, obj)

    def _save_object_to_group(self, grp, obj: DataObject) -> None:
        """Save object to HDF5 group."""
        obj_type = type(obj).__name__
        grp.attrs["type"] = obj_type

        if obj_type == "SignalObj":
            grp.create_dataset("x", data=obj.x)
            grp.create_dataset("y", data=obj.y)
            if obj.dx is not None:
                grp.create_dataset("dx", data=obj.dx)
            if obj.dy is not None:
                grp.create_dataset("dy", data=obj.dy)
            if hasattr(obj, "title") and obj.title:
                grp.attrs["title"] = obj.title
        elif obj_type == "ImageObj":
            grp.create_dataset("data", data=obj.data)
            if hasattr(obj, "title") and obj.title:
                grp.attrs["title"] = obj.title
            for attr in ("x0", "y0", "dx", "dy"):
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if val is not None:
                        grp.attrs[attr] = val

    def load(self, filepath: str) -> None:
        """Load workspace from HDF5 file.

        Note: This uploads objects from a local HDF5 file to DataLab.

        Args:
            filepath: Path to HDF5 file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            for name in f:
                grp = f[name]
                obj = self._load_object_from_group(grp, name)
                if obj is not None:
                    self.add(name, obj, overwrite=True)

    def _load_object_from_group(self, grp, name: str) -> DataObject | None:
        """Load object from HDF5 group."""
        obj_type = grp.attrs.get("type", "unknown")

        if obj_type == "SignalObj" or ("x" in grp and "y" in grp):
            x = np.array(grp["x"])
            y = np.array(grp["y"])
            dx = np.array(grp["dx"]) if "dx" in grp else None
            dy = np.array(grp["dy"]) if "dy" in grp else None

            obj = SignalObj()
            obj.set_xydata(x, y, dx=dx, dy=dy)
            obj.title = grp.attrs.get("title", name)
            return obj

        if obj_type == "ImageObj" or "data" in grp:
            data = np.array(grp["data"])
            obj = ImageObj()
            obj.data = data
            obj.title = grp.attrs.get("title", name)
            for attr in ("x0", "y0", "dx", "dy"):
                if attr in grp.attrs:
                    setattr(obj, attr, float(grp.attrs[attr]))
            return obj

        return None

    # =========================================================================
    # Computation operations
    # =========================================================================

    def select_objects(
        self, names: list[str], panel: str | None = None
    ) -> tuple[list[str], str]:
        """Select objects by name in DataLab.

        Args:
            names: List of object names/titles to select.
            panel: Panel name ("signal" or "image"). None = auto-detect.

        Returns:
            Tuple of (list of selected names, panel name).

        Raises:
            KeyError: If any object not found.
            ValueError: If objects span multiple panels.
        """
        response = self._client.post(
            "/api/v1/select",
            json={"selection": names, "panel": panel},
        )
        self._raise_for_status(response)
        data = response.json()
        return data["selected"], data["panel"]

    def get_selected_objects(self, panel: str | None = None) -> list[str]:
        """Get names of currently selected objects.

        Note: This is not currently exposed via the Web API.
        Returns an empty list as a fallback.

        Args:
            panel: Panel name. None = current panel.

        Returns:
            List of selected object names.
        """
        # TODO: Add GET /api/v1/selection endpoint to DataLab Web API
        return []

    def calc(self, name: str, param: dict | None = None) -> tuple[bool, list[str]]:
        """Call a computation function on selected objects.

        Args:
            name: Computation function name (e.g., "normalize", "fft").
            param: Optional parameters as a dictionary.

        Returns:
            Tuple of (success, list of new object names created).

        Raises:
            ValueError: If computation function not found.
            RuntimeError: If computation fails.
        """
        request_body = {"name": name}
        if param is not None:
            # Convert DataSet to dict if needed
            if hasattr(param, "__dict__") and not isinstance(param, dict):
                # It's a DataSet-like object, convert to dict
                param_dict = {}
                for key in dir(param):
                    if not key.startswith("_"):
                        try:
                            value = getattr(param, key)
                            if not callable(value):
                                # Try to serialize - skip non-serializable
                                import json

                                json.dumps(value)
                                param_dict[key] = value
                        except (TypeError, ValueError):
                            pass
                request_body["param"] = param_dict
            else:
                request_body["param"] = param

        response = self._client.post("/api/v1/calc", json=request_body)
        self._raise_for_status(response)
        data = response.json()
        return data["success"], data.get("result_names", [])

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
