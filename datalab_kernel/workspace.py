# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Workspace API
=============

The Workspace class provides data access and persistence for the DataLab kernel.
It supports two backends:

- Standalone backend: local memory storage with HDF5 persistence
- Live backend: synchronized with a running DataLab instance

The backend is selected automatically at kernel startup.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING

from sigima import ImageObj, SignalObj

if TYPE_CHECKING:
    DataObject = SignalObj | ImageObj


class WorkspaceMode(Enum):
    """Workspace execution mode."""

    STANDALONE = "standalone"
    LIVE = "live"


class WorkspaceBackend(ABC):
    """Abstract base class for workspace backends."""

    @abstractmethod
    def list(self) -> list[str]:
        """List all object names in the workspace."""

    @abstractmethod
    def get(self, name: str) -> DataObject:
        """Retrieve an object by name."""

    @abstractmethod
    def add(self, name: str, obj: DataObject, overwrite: bool = False) -> None:
        """Add an object to the workspace."""

    @abstractmethod
    def remove(self, name: str) -> None:
        """Remove an object from the workspace."""

    @abstractmethod
    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an object."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if an object exists."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all objects from the workspace."""

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Save workspace to HDF5 file."""

    @abstractmethod
    def load(self, filepath: str) -> None:
        """Load workspace from HDF5 file."""


class Workspace:
    """
    Workspace API for data access and persistence.

    The Workspace provides a unified interface to access, modify, and persist
    scientific data objects (signals and images). It automatically selects
    the appropriate backend:

    - **Standalone mode**: Local memory storage with HDF5 persistence
    - **Live mode**: Synchronized with a running DataLab instance

    Example::

        # List objects
        workspace.list()

        # Get an object
        img = workspace.get("i042")

        # Add a new object
        workspace.add("filtered", processed_img)

        # Save to file
        workspace.save("analysis.h5")
    """

    def __init__(self, backend: WorkspaceBackend | None = None) -> None:
        """Initialize workspace with the given backend.

        Args:
            backend: Backend to use. If None, auto-detect.
        """
        self._backend: WorkspaceBackend
        self._mode: WorkspaceMode

        if backend is not None:
            self._backend = backend
            self._mode = self._detect_mode_from_backend(backend)
        else:
            # Auto-detect mode
            self._backend, self._mode = self._auto_detect_backend()

    def _detect_mode_from_backend(self, backend: WorkspaceBackend) -> WorkspaceMode:
        """Detect mode from backend type."""
        # Check for WebApiBackend (imported conditionally)
        backend_class_name = type(backend).__name__
        if backend_class_name == "WebApiBackend":
            return WorkspaceMode.LIVE
        return WorkspaceMode.STANDALONE

    def _auto_detect_backend(self) -> tuple[WorkspaceBackend, WorkspaceMode]:
        """Auto-detect and create appropriate backend.

        Priority order:
        1. WebAPI backend if DATALAB_WORKSPACE_URL is set
        2. StandaloneBackend (fallback)
        """
        logger = logging.getLogger("datalab-kernel")

        # Check kernel mode environment variable
        kernel_mode = os.environ.get("DATALAB_KERNEL_MODE", "auto").lower()

        if kernel_mode == "standalone":
            from datalab_kernel.backends.standalone import StandaloneBackend

            return StandaloneBackend(), WorkspaceMode.STANDALONE

        # Try WebAPI (if URL is set)
        webapi_url = os.environ.get("DATALAB_WORKSPACE_URL")
        if webapi_url:
            try:
                from datalab_kernel.backends.webapi import WebApiBackend

                backend = WebApiBackend()
                return backend, WorkspaceMode.LIVE
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"Failed to connect to DataLab Web API: {e}")
                if kernel_mode == "live":
                    # User explicitly requested live mode, raise error
                    raise ConnectionError(
                        f"Failed to connect to DataLab Web API at {webapi_url}: {e}"
                    ) from None
                # Fall through to standalone

        # Fallback to standalone
        from datalab_kernel.backends.standalone import StandaloneBackend

        return StandaloneBackend(), WorkspaceMode.STANDALONE

    def resync(self) -> bool:
        """Attempt to resync with DataLab via Web API.

        If currently in standalone mode and DataLab Web API becomes available,
        switch to live mode. Objects in the standalone workspace are
        transferred to DataLab.

        Returns:
            True if switched to live mode, False if already live or
             DataLab Web API is not available.
        """
        if self._mode == WorkspaceMode.LIVE:
            return False

        # Try to connect to DataLab Web API
        try:
            from datalab_kernel.backends.webapi import WebApiBackend

            new_backend = WebApiBackend()
        except Exception:  # pylint: disable=broad-exception-caught
            return False

        # Transfer objects from standalone to live backend
        old_backend = self._backend
        for name in old_backend.list():
            obj = old_backend.get(name)
            new_backend.add(name, obj)

        # Switch backends
        self._backend = new_backend
        self._mode = WorkspaceMode.LIVE
        return True

    def connect(self, url: str | None = None, token: str | None = None) -> bool:
        """Connect to DataLab Web API.

        Attempts to establish a connection to DataLab using the Web API.
        If currently in standalone mode with objects, they will be
        transferred to the DataLab workspace.

        Args:
            url: DataLab Web API URL (e.g., "http://127.0.0.1:8080").
                If None, reads from DATALAB_WORKSPACE_URL.
            token: Authentication token. If None, reads from DATALAB_WORKSPACE_TOKEN.

        Returns:
            True if connected successfully, False otherwise.

        Example::

            # Connect using environment variables
            workspace.connect()

            # Connect with explicit credentials
            workspace.connect("http://127.0.0.1:8080", "my-token")
        """
        if self._mode == WorkspaceMode.LIVE:
            return True  # Already connected

        try:
            from datalab_kernel.backends.webapi import WebApiBackend

            new_backend = WebApiBackend(base_url=url, token=token)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"Failed to connect: {e}")
            return False

        # Transfer objects from standalone to live backend
        old_backend = self._backend
        for name in old_backend.list():
            obj = old_backend.get(name)
            new_backend.add(name, obj)

        # Switch backends
        self._backend = new_backend
        self._mode = WorkspaceMode.LIVE
        return True

    def status(self) -> dict:
        """Get current workspace status.

        Returns:
            Dictionary with mode, backend type, and connection info.

        Example::

            >>> workspace.status()
            {'mode': 'live', 'backend': 'WebApiBackend', 'url': 'http://127.0.0.1:8080'}
        """
        backend_name = type(self._backend).__name__
        result = {
            "mode": self._mode.value,
            "backend": backend_name,
            "object_count": len(self.list()),
        }

        # Add connection info for WebAPI backend
        if hasattr(self._backend, "base_url"):
            result["url"] = self._backend.base_url

        return result

    @property
    def mode(self) -> WorkspaceMode:
        """Get current execution mode."""
        return self._mode

    def list(self) -> list[str]:
        """List all object names in the workspace.

        Returns:
            List of object names
        """
        return self._backend.list()

    def get(self, name: str) -> DataObject:
        """Retrieve an object by name.

        Args:
            name: Object name

        Returns:
            The requested object (SignalObj or ImageObj)

        Raises:
            KeyError: If object not found
        """
        return self._backend.get(name)

    def add(self, name: str, obj: DataObject, overwrite: bool = False) -> DataObject:
        """Add an object to the workspace.

        Args:
            name: Object name
            obj: Object to add (SignalObj or ImageObj)
            overwrite: If True, replace existing object with same name

        Returns:
            The added object

        Raises:
            ValueError: If object exists and overwrite=False
        """
        self._backend.add(name, obj, overwrite=overwrite)
        # Backend waits for the object to appear, so get() should work
        return self._backend.get(name)

    def remove(self, name: str) -> None:
        """Remove an object from the workspace.

        Args:
            name: Object name

        Raises:
            KeyError: If object not found
        """
        self._backend.remove(name)

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an object.

        Args:
            old_name: Current object name
            new_name: New object name

        Raises:
            KeyError: If old_name not found
            ValueError: If new_name already exists
        """
        self._backend.rename(old_name, new_name)

    def exists(self, name: str) -> bool:
        """Check if an object exists.

        Args:
            name: Object name

        Returns:
            True if object exists
        """
        return self._backend.exists(name)

    def clear(self) -> None:
        """Remove all objects from the workspace."""
        self._backend.clear()

    def save(self, filepath: str) -> None:
        """Save workspace to HDF5 file.

        Args:
            filepath: Path to save file (should end with .h5)
        """
        self._backend.save(filepath)

    def load(self, filepath: str) -> None:
        """Load workspace from HDF5 file.

        Args:
            filepath: Path to HDF5 file
        """
        self._backend.load(filepath)

    def select_objects(
        self, names: list[str], panel: str | None = None
    ) -> tuple[list[str], str]:
        """Select objects by name in DataLab.

        This method is only available in live mode. It selects the specified
        objects, making them the active selection for subsequent operations.

        Args:
            names: List of object names/titles to select.
            panel: Panel name ("signal" or "image"). None = auto-detect.

        Returns:
            Tuple of (list of selected names, panel name).

        Raises:
            RuntimeError: If not in live mode.
            KeyError: If any object not found.
            ValueError: If objects span multiple panels.

        Example::

            # Select objects before calling calc
            workspace.select_objects(["signal1", "signal2"])
            workspace.calc("average")
        """
        if self._mode != WorkspaceMode.LIVE:
            raise RuntimeError("select_objects() is only available in live mode")

        backend = self._backend
        if hasattr(backend, "select_objects"):
            return backend.select_objects(names, panel)
        raise RuntimeError("Backend does not support select_objects")

    def calc(self, name: str, param: object | None = None) -> object | None:
        """Call a DataLab computation function.

        This method is only available in live mode. It calls DataLab's
        computation feature by name on the currently selected objects.

        Args:
            name: Computation function name (e.g., "normalize", "fft", "denoise")
            param: Optional parameter DataSet or dict for the computation

        Returns:
            Tuple of (success, list of new object names), or None if backend
            doesn't support returning results.

        Raises:
            RuntimeError: If not in live mode
            ValueError: If computation function not found

        Example::

            # Simple computation (select objects first)
            workspace.select_objects(["my_signal"])
            workspace.calc("normalize")

            # Computation with parameters
            workspace.calc("moving_average", {"n": 5})

            # Or with DataSet
            from sigima.params import MovingAverageParam
            workspace.calc("moving_average", MovingAverageParam.create(n=5))
        """
        if self._mode != WorkspaceMode.LIVE:
            raise RuntimeError("calc() is only available in live mode")

        backend = self._backend
        # Call backend's calc method
        if hasattr(backend, "calc"):
            return backend.calc(name, param)
        raise RuntimeError("Backend does not support calc()")

    def __len__(self) -> int:
        """Return number of objects in workspace."""
        return len(self.list())

    def __iter__(self) -> Iterator[str]:
        """Iterate over object names."""
        return iter(self.list())

    def __contains__(self, name: str) -> bool:
        """Check if object exists (supports 'in' operator)."""
        return self.exists(name)

    def __repr__(self) -> str:
        """Return string representation."""
        names = self.list()
        count = len(names)
        mode_str = self._mode.value
        if count == 0:
            return f"Workspace({mode_str}, empty)"
        if count <= 5:
            return f"Workspace({mode_str}, objects=[{', '.join(names)}])"
        shown = ", ".join(names[:5])
        return f"Workspace({mode_str}, objects=[{shown}, ...] ({count} total))"
