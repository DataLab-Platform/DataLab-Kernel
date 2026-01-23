# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Plotter API
===========

The Plotter class provides visualization capabilities for the DataLab kernel.
It supports inline notebook display and optional DataLab GUI synchronization.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datalab_kernel.workspace import DataObject, Workspace


class Plotter:
    """
    Visualization frontend for the DataLab kernel.

    The Plotter provides methods to display signals and images inline in
    Jupyter notebooks, and optionally synchronize views with a running
    DataLab instance.

    Example::

        # Plot by name
        plotter.plot("i042")

        # Plot object directly
        plotter.plot(workspace.get("i042"))
    """

    def __init__(self, workspace: Workspace) -> None:
        """Initialize plotter with workspace reference.

        Args:
            workspace: The workspace containing objects to plot
        """
        self._workspace = workspace

    def plot(
        self,
        obj_or_name: DataObject | str,
        title: str | None = None,
        **kwargs,
    ) -> PlotResult:
        """Plot an object or retrieve and plot by name.

        Args:
            obj_or_name: Object to plot, or name of object in workspace
            title: Optional plot title override
            **kwargs: Additional plotting options

        Returns:
            PlotResult with display capabilities

        Raises:
            KeyError: If name not found in workspace
        """
        if isinstance(obj_or_name, str):
            obj = self._workspace.get(obj_or_name)
            if title is None:
                title = obj_or_name
        else:
            obj = obj_or_name
            if title is None and hasattr(obj, "title"):
                title = obj.title

        return PlotResult(obj, title=title, **kwargs)


class PlotResult:
    """
    Result of a plot operation with rich display capabilities.

    Supports Jupyter's rich display protocol for inline rendering.
    """

    def __init__(self, obj: DataObject, title: str | None = None, **kwargs) -> None:
        """Initialize plot result.

        Args:
            obj: Object to display
            title: Plot title
            **kwargs: Additional options
        """
        self._obj = obj
        self._title = title
        self._kwargs = kwargs

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "Untitled")

        if obj_type == "SignalObj":
            return self._signal_to_html()
        elif obj_type == "ImageObj":
            return self._image_to_html()
        else:
            return f"<div><strong>{title}</strong>: {obj_type}</div>"

    def _repr_png_(self) -> bytes:
        """Return PNG representation for Jupyter display."""
        return self._render_to_png()

    def _signal_to_html(self) -> str:
        """Render signal to HTML with embedded plot."""
        try:
            png_data = self._render_to_png()
            b64_data = base64.b64encode(png_data).decode("utf-8")
            title = self._title or getattr(self._obj, "title", "Signal")
            return f"""
            <div style="text-align: center;">
                <h4>{title}</h4>
                <img src="data:image/png;base64,{b64_data}" />
            </div>
            """
        except Exception as e:
            return f"<div>Error rendering signal: {e}</div>"

    def _image_to_html(self) -> str:
        """Render image to HTML with embedded plot."""
        try:
            png_data = self._render_to_png()
            b64_data = base64.b64encode(png_data).decode("utf-8")
            title = self._title or getattr(self._obj, "title", "Image")
            return f"""
            <div style="text-align: center;">
                <h4>{title}</h4>
                <img src="data:image/png;base64,{b64_data}" />
            </div>
            """
        except Exception as e:
            return f"<div>Error rendering image: {e}</div>"

    def _render_to_png(self) -> bytes:
        """Render object to PNG bytes using matplotlib."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "")

        fig, ax = plt.subplots(figsize=(8, 5))

        if obj_type == "SignalObj":
            x = self._obj.x
            y = self._obj.y
            ax.plot(x, y, "-", linewidth=1)

            xlabel = getattr(self._obj, "xlabel", None) or "X"
            ylabel = getattr(self._obj, "ylabel", None) or "Y"
            xunit = getattr(self._obj, "xunit", None)
            yunit = getattr(self._obj, "yunit", None)

            if xunit:
                xlabel = f"{xlabel} [{xunit}]"
            if yunit:
                ylabel = f"{ylabel} [{yunit}]"

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        elif obj_type == "ImageObj":
            data = self._obj.data
            if np.iscomplexobj(data):
                data = np.abs(data)

            im = ax.imshow(data, aspect="auto", origin="lower")
            plt.colorbar(im, ax=ax)

            xlabel = getattr(self._obj, "xlabel", None) or "X"
            ylabel = getattr(self._obj, "ylabel", None) or "Y"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def __repr__(self) -> str:
        """Return string representation."""
        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "Untitled")
        return f"PlotResult({obj_type}: {title})"
