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

import os
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sigima.objects import ImageObj, SignalObj

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


class StandaloneBackend(WorkspaceBackend):
    """Standalone backend using local memory storage with HDF5 persistence."""

    def __init__(self) -> None:
        self._objects: dict[str, DataObject] = {}

    def list(self) -> list[str]:
        """List all object names in the workspace."""
        return list(self._objects.keys())

    def get(self, name: str) -> DataObject:
        """Retrieve an object by name.

        Args:
            name: Object name

        Returns:
            The requested object

        Raises:
            KeyError: If object not found
        """
        if name not in self._objects:
            available = ", ".join(self._objects.keys()) if self._objects else "(empty)"
            raise KeyError(
                f"Object '{name}' not found. Available objects: [{available}]"
            )
        return self._objects[name]

    def add(self, name: str, obj: DataObject, overwrite: bool = False) -> None:
        """Add an object to the workspace.

        Args:
            name: Object name
            obj: Object to add (SignalObj or ImageObj)
            overwrite: If True, replace existing object with same name

        Raises:
            ValueError: If object with name exists and overwrite=False
        """
        if name in self._objects and not overwrite:
            raise ValueError(
                f"Object '{name}' already exists. Use overwrite=True to replace."
            )
        # Make a copy to ensure isolation
        self._objects[name] = obj.copy() if hasattr(obj, "copy") else obj

    def remove(self, name: str) -> None:
        """Remove an object from the workspace.

        Args:
            name: Object name

        Raises:
            KeyError: If object not found
        """
        if name not in self._objects:
            available = ", ".join(self._objects.keys()) if self._objects else "(empty)"
            raise KeyError(
                f"Object '{name}' not found. Available objects: [{available}]"
            )
        del self._objects[name]

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an object.

        Args:
            old_name: Current object name
            new_name: New object name

        Raises:
            KeyError: If old_name not found
            ValueError: If new_name already exists
        """
        if old_name not in self._objects:
            raise KeyError(f"Object '{old_name}' not found.")
        if new_name in self._objects:
            raise ValueError(f"Object '{new_name}' already exists.")
        self._objects[new_name] = self._objects.pop(old_name)
        # Update object's internal title if it has one
        obj = self._objects[new_name]
        if hasattr(obj, "title"):
            obj.title = new_name

    def exists(self, name: str) -> bool:
        """Check if an object exists."""
        return name in self._objects

    def clear(self) -> None:
        """Remove all objects from the workspace."""
        self._objects.clear()

    def save(self, filepath: str) -> None:
        """Save workspace to HDF5 file.

        Args:
            filepath: Path to save file (should end with .h5)
        """
        # Delayed import: h5py is optional for HDF5 persistence
        import h5py  # pylint: disable=import-outside-toplevel

        # Ensure .h5 extension
        if not filepath.endswith(".h5"):
            filepath = filepath + ".h5"

        with h5py.File(filepath, "w") as f:
            # Store metadata
            f.attrs["datalab_kernel_version"] = "0.1.0"
            f.attrs["format_version"] = "1.0"

            for name, obj in self._objects.items():
                grp = f.create_group(name)
                self._save_object_to_group(grp, obj)

    def _save_object_to_group(self, grp, obj: DataObject) -> None:
        """Save a single object to an HDF5 group."""
        # Detect object type
        obj_type = type(obj).__name__
        grp.attrs["type"] = obj_type

        if obj_type == "SignalObj":
            # Save signal data
            grp.create_dataset("x", data=obj.x)
            grp.create_dataset("y", data=obj.y)
            if obj.dx is not None:
                grp.create_dataset("dx", data=obj.dx)
            if obj.dy is not None:
                grp.create_dataset("dy", data=obj.dy)
            # Save metadata
            if hasattr(obj, "title") and obj.title:
                grp.attrs["title"] = obj.title
            if hasattr(obj, "xlabel") and obj.xlabel:
                grp.attrs["xlabel"] = obj.xlabel
            if hasattr(obj, "ylabel") and obj.ylabel:
                grp.attrs["ylabel"] = obj.ylabel
            if hasattr(obj, "xunit") and obj.xunit:
                grp.attrs["xunit"] = obj.xunit
            if hasattr(obj, "yunit") and obj.yunit:
                grp.attrs["yunit"] = obj.yunit

        elif obj_type == "ImageObj":
            # Save image data
            grp.create_dataset("data", data=obj.data)
            # Save coordinate info
            for attr in ("x0", "y0", "dx", "dy"):
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    if val is not None:
                        grp.attrs[attr] = val
            # Save metadata
            if hasattr(obj, "title") and obj.title:
                grp.attrs["title"] = obj.title
            if hasattr(obj, "xlabel") and obj.xlabel:
                grp.attrs["xlabel"] = obj.xlabel
            if hasattr(obj, "ylabel") and obj.ylabel:
                grp.attrs["ylabel"] = obj.ylabel
            if hasattr(obj, "zlabel") and obj.zlabel:
                grp.attrs["zlabel"] = obj.zlabel
            if hasattr(obj, "xunit") and obj.xunit:
                grp.attrs["xunit"] = obj.xunit
            if hasattr(obj, "yunit") and obj.yunit:
                grp.attrs["yunit"] = obj.yunit
            if hasattr(obj, "zunit") and obj.zunit:
                grp.attrs["zunit"] = obj.zunit

    def load(self, filepath: str) -> None:
        """Load workspace from HDF5 file.

        Args:
            filepath: Path to HDF5 file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        # Delayed import: h5py is optional for HDF5 persistence
        import h5py  # pylint: disable=import-outside-toplevel

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with h5py.File(filepath, "r") as f:
            for name in f:
                grp = f[name]
                obj = self._load_object_from_group(grp, name)
                if obj is not None:
                    self._objects[name] = obj

    def _load_object_from_group(self, grp, name: str) -> DataObject | None:
        """Load a single object from an HDF5 group."""
        obj_type = grp.attrs.get("type", "unknown")

        if obj_type == "SignalObj":
            return self._load_signal(grp, name)
        if obj_type == "ImageObj":
            return self._load_image(grp, name)
        # Try to infer type from data
        if "x" in grp and "y" in grp:
            return self._load_signal(grp, name)
        if "data" in grp:
            return self._load_image(grp, name)
        return None

    def _load_signal(self, grp, name: str) -> DataObject:
        """Load a SignalObj from an HDF5 group."""
        # Conditional import: prefer sigima, fallback to local objects
        # pylint: disable=import-outside-toplevel
        try:
            from sigima import SignalObj
        except ImportError:
            from datalab_kernel.objects import SignalObj

        x = np.array(grp["x"])
        y = np.array(grp["y"])
        dx = np.array(grp["dx"]) if "dx" in grp else None
        dy = np.array(grp["dy"]) if "dy" in grp else None

        obj = SignalObj()
        obj.set_xydata(x, y, dx=dx, dy=dy)
        obj.title = grp.attrs.get("title", name)
        if "xlabel" in grp.attrs:
            obj.xlabel = grp.attrs["xlabel"]
        if "ylabel" in grp.attrs:
            obj.ylabel = grp.attrs["ylabel"]
        if "xunit" in grp.attrs:
            obj.xunit = grp.attrs["xunit"]
        if "yunit" in grp.attrs:
            obj.yunit = grp.attrs["yunit"]

        return obj

    def _load_image(self, grp, name: str) -> DataObject:
        """Load an ImageObj from an HDF5 group."""
        # Conditional import: prefer sigima, fallback to local objects
        # pylint: disable=import-outside-toplevel
        try:
            from sigima import ImageObj
        except ImportError:
            from datalab_kernel.objects import ImageObj

        data = np.array(grp["data"])

        obj = ImageObj()
        obj.data = data
        obj.title = grp.attrs.get("title", name)

        for attr in ("x0", "y0", "dx", "dy"):
            if attr in grp.attrs:
                setattr(obj, attr, float(grp.attrs[attr]))

        if "xlabel" in grp.attrs:
            obj.xlabel = grp.attrs["xlabel"]
        if "ylabel" in grp.attrs:
            obj.ylabel = grp.attrs["ylabel"]
        if "zlabel" in grp.attrs:
            obj.zlabel = grp.attrs["zlabel"]
        if "xunit" in grp.attrs:
            obj.xunit = grp.attrs["xunit"]
        if "yunit" in grp.attrs:
            obj.yunit = grp.attrs["yunit"]
        if "zunit" in grp.attrs:
            obj.zunit = grp.attrs["zunit"]

        return obj


class LiveBackend(WorkspaceBackend):
    """Live backend synchronized with a running DataLab instance.

    This backend communicates with a running DataLab application via
    the RemoteProxy XML-RPC interface, enabling real-time synchronization
    between the Jupyter kernel workspace and DataLab's data panels.
    """

    def __init__(self) -> None:
        self._proxy = None
        self._connect()

    def _connect(self) -> None:
        """Connect to DataLab instance."""
        # Conditional import: datalab is optional dependency
        # pylint: disable=import-outside-toplevel
        try:
            from datalab.control.proxy import RemoteProxy

            self._proxy = RemoteProxy()
            # Test connection by getting version
            self._proxy.get_version()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._proxy = None
            raise ConnectionError("Failed to connect to DataLab instance") from e

    @property
    def proxy(self):
        """Get the DataLab proxy.

        Returns:
            RemoteProxy connected to DataLab instance.

        Raises:
            ConnectionError: If not connected to DataLab.
        """
        if self._proxy is None:
            raise ConnectionError("Not connected to DataLab")
        return self._proxy

    def _get_panel_for_title(self, title: str) -> str | None:
        """Find which panel contains an object with the given title.

        Args:
            title: Object title to search for.

        Returns:
            Panel name ("signal" or "image") or None if not found.
        """
        # Check signals
        try:
            sig_titles = self.proxy.get_object_titles(panel="signal")
            if title in sig_titles:
                return "signal"
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Check images
        try:
            img_titles = self.proxy.get_object_titles(panel="image")
            if title in img_titles:
                return "image"
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return None

    def list(self) -> list[str]:
        """List all object names in the workspace.

        Returns:
            List of object titles from both signal and image panels.
        """
        titles = []

        # Get signal titles
        try:
            sig_titles = self.proxy.get_object_titles(panel="signal")
            titles.extend(sig_titles)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Get image titles
        try:
            img_titles = self.proxy.get_object_titles(panel="image")
            titles.extend(img_titles)
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        return titles

    def get(self, name: str) -> DataObject:
        """Retrieve an object by name.

        Args:
            name: Object title to retrieve.

        Returns:
            SignalObj or ImageObj from DataLab.

        Raises:
            KeyError: If object not found.
        """
        # Try to find in signals first
        try:
            sig_titles = self.proxy.get_object_titles(panel="signal")
            if name in sig_titles:
                return self.proxy.get_object(name, panel="signal")
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        # Try to find in images
        try:
            img_titles = self.proxy.get_object_titles(panel="image")
            if name in img_titles:
                return self.proxy.get_object(name, panel="image")
        except Exception:  # pylint: disable=broad-exception-caught
            pass

        raise KeyError(f"Object '{name}' not found in DataLab workspace")

    def add(self, name: str, obj: DataObject, overwrite: bool = False) -> None:
        """Add an object to the workspace.

        Args:
            name: Object title.
            obj: SignalObj or ImageObj to add.
            overwrite: If True, replace existing object with same name.

        Raises:
            ValueError: If object exists and overwrite is False.
            TypeError: If object type is not supported.
        """
        if not overwrite and self.exists(name):
            raise ValueError(f"Object '{name}' already exists. Use overwrite=True.")

        if overwrite and self.exists(name):
            self.remove(name)

        # Set the title on the object
        obj.title = name

        # add_object auto-detects panel based on object type
        self.proxy.add_object(obj)

    def remove(self, name: str) -> None:
        """Remove an object from the workspace.

        Args:
            name: Object title to remove.

        Raises:
            KeyError: If object not found.
            NotImplementedError: If DataLab version doesn't support remove.
        """
        panel = self._get_panel_for_title(name)
        if panel is None:
            raise KeyError(f"Object '{name}' not found")

        # Select the object by title and remove it
        self.proxy.select_objects([name], panel=panel)

        # Try different methods depending on DataLab version
        # DataLab 1.1+ has call_method, older versions don't support individual remove
        if hasattr(self.proxy, "call_method"):
            self.proxy.call_method("remove_object", force=True)
        elif "remove_object" in self.proxy.get_method_list():
            # Direct XML-RPC call if server exposes it
            # pylint: disable=protected-access
            self.proxy._datalab.remove_object(True)  # noqa: SLF001
            # pylint: enable=protected-access
        else:
            raise NotImplementedError(
                "Individual object removal requires DataLab 1.1+. "
                "Use clear() to remove all objects, or upgrade DataLab."
            )

    def rename(self, old_name: str, new_name: str) -> None:
        """Rename an object.

        Args:
            old_name: Current object title.
            new_name: New object title.

        Raises:
            KeyError: If old_name not found.
            ValueError: If new_name already exists.
        """
        if not self.exists(old_name):
            raise KeyError(f"Object '{old_name}' not found")
        if self.exists(new_name):
            raise ValueError(f"Object '{new_name}' already exists")

        # Get the object, update title, remove old, add new
        obj = self.get(old_name)
        obj.title = new_name
        self.remove(old_name)
        self.proxy.add_object(obj)

    def exists(self, name: str) -> bool:
        """Check if an object exists.

        Args:
            name: Object title to check.

        Returns:
            True if object exists, False otherwise.
        """
        return name in self.list()

    def clear(self) -> None:
        """Remove all objects from the workspace."""
        # Use DataLab's reset_all to clear everything
        self.proxy.reset_all()

    def save(self, filepath: str) -> None:
        """Save workspace to HDF5 file.

        Args:
            filepath: Path to save HDF5 file.
        """
        self.proxy.save_h5_workspace(filepath)

    def load(self, filepath: str) -> None:
        """Load workspace from HDF5 file.

        Args:
            filepath: Path to HDF5 file to load.
        """
        self.proxy.load_h5_workspace([filepath], reset_all=False)


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
            self._mode = (
                WorkspaceMode.LIVE
                if isinstance(backend, LiveBackend)
                else WorkspaceMode.STANDALONE
            )
        else:
            # Auto-detect mode
            self._backend, self._mode = self._auto_detect_backend()

    def _auto_detect_backend(self) -> tuple[WorkspaceBackend, WorkspaceMode]:
        """Auto-detect and create appropriate backend.

        Priority order:
        1. WebAPI backend if DATALAB_WORKSPACE_URL is set
        2. XML-RPC LiveBackend if DataLab is running
        3. StandaloneBackend (fallback)
        """
        # Check kernel mode environment variable
        kernel_mode = os.environ.get("DATALAB_KERNEL_MODE", "auto").lower()

        if kernel_mode == "standalone":
            return StandaloneBackend(), WorkspaceMode.STANDALONE

        # Try WebAPI first (if URL is set)
        webapi_url = os.environ.get("DATALAB_WORKSPACE_URL")
        if webapi_url:
            try:
                from datalab_kernel.backends.webapi import WebApiBackend

                backend = WebApiBackend()
                return backend, WorkspaceMode.LIVE
            except Exception:  # pylint: disable=broad-exception-caught
                if kernel_mode == "live":
                    # User explicitly requested live mode, raise error
                    raise ConnectionError(
                        "Failed to connect to DataLab Web API. "
                        "Check DATALAB_WORKSPACE_URL and DATALAB_WORKSPACE_TOKEN."
                    ) from None
                # Fall through to try XML-RPC

        # Try XML-RPC LiveBackend
        try:
            backend = LiveBackend()
            return backend, WorkspaceMode.LIVE
        except Exception:  # pylint: disable=broad-exception-caught
            if kernel_mode == "live":
                raise ConnectionError(
                    "Failed to connect to DataLab. "
                    "Ensure DataLab is running with remote control enabled."
                ) from None

        # Fallback to standalone
        return StandaloneBackend(), WorkspaceMode.STANDALONE

    def resync(self) -> bool:
        """Attempt to resync with DataLab.

        If currently in standalone mode and DataLab becomes available,
        switch to live mode. Objects in the standalone workspace are
        transferred to DataLab.

        Returns:
            True if switched to live mode, False if already live or
             DataLab is not available.
        """
        if self._mode == WorkspaceMode.LIVE:
            return False

        # Try to connect to DataLab
        try:
            new_backend = LiveBackend()
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
        if hasattr(self._backend, "_base_url"):
            result["url"] = self._backend._base_url

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
        # For live backend, allow retry window for object to appear
        # This handles race conditions with XML-RPC (especially on Python 3.9)
        for _ in range(20):
            try:
                return self._backend.get(name)
            except KeyError:
                time.sleep(0.1)
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

    def calc(self, name: str, param: object | None = None) -> object | None:
        """Call a DataLab computation function.

        This method is only available in live mode. It calls DataLab's
        computation feature by name on the currently selected objects.

        Args:
            name: Computation function name (e.g., "normalize", "fft", "denoise")
            param: Optional parameter DataSet for the computation

        Returns:
            Result from the computation (typically a DataSet with results),
            or None in standalone mode.

        Raises:
            RuntimeError: If not in live mode

        Example::

            # Simple computation
            workspace.calc("normalize")

            # Computation with parameters
            from sigima.params import MovingAverageParam
            workspace.calc("moving_average", MovingAverageParam.create(n=5))
        """
        if self._mode != WorkspaceMode.LIVE:
            raise RuntimeError("calc() is only available in live mode")

        backend = self._backend
        if isinstance(backend, LiveBackend):
            return backend.proxy.calc(name, param)
        return None

    @property
    def proxy(self):
        """Access the DataLab proxy for advanced operations.

        This property is only available in live mode. It provides direct
        access to the RemoteProxy for operations not covered by the
        Workspace API.

        Returns:
            RemoteProxy connected to DataLab

        Raises:
            RuntimeError: If not in live mode

        Example::

            # Access proxy for advanced operations
            proxy = workspace.proxy
            proxy.set_current_panel("signal")
            proxy.select_objects([1, 2, 3])
        """
        if self._mode != WorkspaceMode.LIVE:
            raise RuntimeError("proxy is only available in live mode")

        backend = self._backend
        if isinstance(backend, LiveBackend):
            return backend.proxy
        raise RuntimeError("Backend is not LiveBackend")

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
