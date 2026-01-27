# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Standalone Backend
==================

Local memory storage backend with HDF5 persistence for standalone mode.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from sigima import ImageObj, SignalObj

from datalab_kernel.workspace import WorkspaceBackend

if TYPE_CHECKING:
    DataObject = SignalObj | ImageObj


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
