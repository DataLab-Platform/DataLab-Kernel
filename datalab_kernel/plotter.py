# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Plotter API
===========

The Plotter class provides visualization capabilities for the DataLab kernel.
It supports inline notebook display and optional DataLab GUI synchronization.

This module provides matplotlib-based visualization for signals and images,
including support for:
- Multiple curves on a single plot
- Multiple images in grid layout
- ROI (Region of Interest) visualization
- Geometry result overlays (rectangles, circles, ellipses, segments, polygons)
- Mask visualization with semi-transparent overlay
- Axis labels with units
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from datalab_kernel.workspace import DataObject, Workspace

# Style configuration for multi-curve/multi-image plots
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive"]
LINESTYLES = ["-", "--", "-.", ":"]
MASK_OPACITY = 0.35  # Opacity for mask overlay


def _get_next_style(index: int) -> tuple[str, str]:
    """Get color and linestyle for the next plot item.

    Args:
        index: Sequential index of the item to style

    Returns:
        A tuple (color, linestyle) for styling the plot item
    """
    color = COLORS[index % len(COLORS)]
    linestyle = LINESTYLES[(index // len(COLORS)) % len(LINESTYLES)]
    return color, linestyle


def _add_single_roi_to_axes(ax: Axes, roi, obj=None) -> None:
    """Add single ROI overlay to matplotlib axes.

    Args:
        ax: Matplotlib axes object
        roi: Single ROI object (SegmentROI, RectangularROI, CircularROI, or
         PolygonalROI)
        obj: Parent object (used for SegmentROI to get physical coordinates)
    """
    # Delayed import
    # pylint: disable=import-outside-toplevel
    from matplotlib import patches

    roi_class = type(roi).__name__

    if roi_class == "RectangularROI":
        # coords = [x0, y0, dx, dy]
        x0, y0, dx, dy = roi.coords
        rect = patches.Rectangle(
            (x0, y0),
            dx,
            dy,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="ROI",
        )
        ax.add_patch(rect)
    elif roi_class == "CircularROI":
        # coords = [xc, yc, r]
        xc, yc, r = roi.coords
        circle = patches.Circle(
            (xc, yc), r, linewidth=2, edgecolor="red", facecolor="none", label="ROI"
        )
        ax.add_patch(circle)
    elif roi_class == "PolygonalROI":
        # coords = [x0, y0, x1, y1, x2, y2, ...]
        points = roi.coords.reshape(-1, 2)
        polygon = patches.Polygon(
            points,
            closed=True,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            label="ROI",
        )
        ax.add_patch(polygon)
    elif roi_class == "SegmentROI" and obj is not None:
        # Signal ROI: X interval
        x0, x1 = roi.get_physical_coords(obj)
        ax.axvspan(x0, x1, alpha=0.2, color="red", label="ROI")


def _add_geometry_to_axes(ax: Axes, result) -> None:
    """Add geometry result overlay to matplotlib axes.

    Args:
        ax: Matplotlib axes object
        result: GeometryResult object with shape information
    """
    # Delayed import
    # pylint: disable=import-outside-toplevel
    from matplotlib import patches
    from sigima.objects import KindShape

    if result.kind == KindShape.RECTANGLE:
        x0, y0, dx, dy = result.coords
        rect = patches.Rectangle(
            (x0, y0),
            dx,
            dy,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
    elif result.kind == KindShape.CIRCLE:
        xc, yc, r = result.coords
        circle = patches.Circle(
            (xc, yc),
            r,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(circle)
    elif result.kind == KindShape.SEGMENT:
        x0, y0, x1, y1 = result.coords
        ax.plot([x0, x1], [y0, y1], "y--", linewidth=2)
    elif result.kind == KindShape.ELLIPSE:
        # For ellipse, coords are (xc, yc, a, b, theta)
        xc, yc, a, b, theta = result.coords
        ellipse = patches.Ellipse(
            (xc, yc),
            2 * a,
            2 * b,
            angle=np.degrees(theta),
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(ellipse)
    elif result.kind == KindShape.POLYGON:
        x = result.coords[::2]
        y = result.coords[1::2]
        ax.plot(x, y, "y--", linewidth=2, marker="o", markersize=4)


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

        # Plot multiple signals
        plotter.plot_curves([sig1, sig2, sig3])

        # Plot multiple images
        plotter.plot_images([img1, img2])
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
        show_roi: bool = True,
        **kwargs,
    ) -> PlotResult:
        """Plot an object or retrieve and plot by name.

        Args:
            obj_or_name: Object to plot, or name of object in workspace
            title: Optional plot title override
            show_roi: Whether to show ROIs defined in the object
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

        return PlotResult(obj, title=title, show_roi=show_roi, **kwargs)

    def plot_curves(
        self,
        objs_or_names: list[DataObject | str | np.ndarray | tuple[np.ndarray, ...]],
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xunit: str | None = None,
        yunit: str | None = None,
        show_roi: bool = True,
        **kwargs,
    ) -> MultiCurvePlotResult:
        """Plot multiple curves on a single plot.

        Args:
            objs_or_names: List of signals to plot. Can be SignalObj, workspace names,
             numpy arrays (y data), or tuples of (x, y) arrays
            title: Optional plot title
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            xunit: Unit for the x-axis
            yunit: Unit for the y-axis
            show_roi: Whether to show ROIs defined in SignalObj instances
            **kwargs: Additional plotting options

        Returns:
            MultiCurvePlotResult with display capabilities
        """
        objs = []
        for obj_or_name in objs_or_names:
            if isinstance(obj_or_name, str):
                objs.append(self._workspace.get(obj_or_name))
            else:
                objs.append(obj_or_name)

        return MultiCurvePlotResult(
            objs,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            show_roi=show_roi,
            **kwargs,
        )

    def plot_images(
        self,
        objs_or_names: list[DataObject | str | np.ndarray],
        title: str | None = None,
        titles: list[str] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        xunit: str | None = None,
        yunit: str | None = None,
        zunit: str | None = None,
        show_roi: bool = True,
        results: list | None = None,
        **kwargs,
    ) -> MultiImagePlotResult:
        """Plot multiple images in a grid layout.

        Args:
            objs_or_names: List of images to plot. Can be ImageObj, workspace names,
             or numpy arrays
            title: Optional overall figure title
            titles: Optional list of titles for each image
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            zlabel: Label for the colorbar (z-axis)
            xunit: Unit for the x-axis
            yunit: Unit for the y-axis
            zunit: Unit for the colorbar
            show_roi: Whether to show ROIs defined in ImageObj instances
            results: Optional list of GeometryResult objects to overlay
            **kwargs: Additional plotting options (e.g., colormap)

        Returns:
            MultiImagePlotResult with display capabilities
        """
        objs = []
        for obj_or_name in objs_or_names:
            if isinstance(obj_or_name, str):
                objs.append(self._workspace.get(obj_or_name))
            else:
                objs.append(obj_or_name)

        return MultiImagePlotResult(
            objs,
            title=title,
            titles=titles,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel,
            xunit=xunit,
            yunit=yunit,
            zunit=zunit,
            show_roi=show_roi,
            results=results,
            **kwargs,
        )


class PlotResult:
    """
    Result of a plot operation with rich display capabilities.

    Supports Jupyter's rich display protocol for inline rendering.
    """

    def __init__(
        self,
        obj: DataObject,
        title: str | None = None,
        show_roi: bool = True,
        results: list | None = None,
        **kwargs,
    ) -> None:
        """Initialize plot result.

        Args:
            obj: Object to display
            title: Plot title
            show_roi: Whether to show ROIs
            results: Optional list of GeometryResult objects to overlay (for images)
            **kwargs: Additional options
        """
        self._obj = obj
        self._title = title
        self._show_roi = show_roi
        self._results = results
        self._kwargs = kwargs

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "Untitled")

        if obj_type == "SignalObj":
            return self._signal_to_html()
        if obj_type == "ImageObj":
            return self._image_to_html()
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
        except Exception as e:  # pylint: disable=broad-exception-caught
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
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering image: {e}</div>"

    def _render_to_png(self) -> bytes:
        """Render object to PNG bytes using matplotlib."""
        # Delayed import: matplotlib is optional and heavy
        # pylint: disable=import-outside-toplevel
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "")

        fig, ax = plt.subplots(figsize=(8, 5))

        if obj_type == "SignalObj":
            self._render_signal(ax)
        elif obj_type == "ImageObj":
            self._render_image(ax, fig)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _render_signal(self, ax: Axes) -> None:
        """Render signal data to axes.

        Args:
            ax: Matplotlib axes object
        """
        obj = self._obj
        x = obj.x
        y = obj.y
        ax.plot(x, y, "-", linewidth=1, color="blue")

        # Axis labels with units
        xlabel = getattr(obj, "xlabel", None) or "X"
        ylabel = getattr(obj, "ylabel", None) or "Y"
        xunit = getattr(obj, "xunit", None)
        yunit = getattr(obj, "yunit", None)

        if xunit:
            xlabel = f"{xlabel} ({xunit})"
        if yunit:
            ylabel = f"{ylabel} ({yunit})"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Show ROIs
        if self._show_roi and hasattr(obj, "roi") and obj.roi:
            for roi in obj.roi:
                _add_single_roi_to_axes(ax, roi, obj)

    def _render_image(self, ax: Axes, fig) -> None:
        """Render image data to axes.

        Args:
            ax: Matplotlib axes object
            fig: Matplotlib figure object
        """
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt

        obj = self._obj
        data = obj.data
        if np.iscomplexobj(data):
            data = np.abs(data)

        colormap = self._kwargs.get("colormap", "viridis")
        im = ax.imshow(data, aspect="auto", origin="upper", cmap=colormap)

        # Overlay mask if present
        if hasattr(obj, "maskdata") and obj.maskdata is not None:
            mask = obj.maskdata
            mask_rgba = np.zeros((*mask.shape, 4))
            mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
            ax.imshow(mask_rgba, origin="upper")

        # Colorbar with label
        zlabel = getattr(obj, "zlabel", None) or ""
        zunit = getattr(obj, "zunit", None)
        cbar = plt.colorbar(im, ax=ax)
        if zlabel:
            cbar.set_label(f"{zlabel} ({zunit})" if zunit else zlabel)

        # Axis labels
        xlabel = getattr(obj, "xlabel", None) or "X"
        ylabel = getattr(obj, "ylabel", None) or "Y"
        xunit = getattr(obj, "xunit", None)
        yunit = getattr(obj, "yunit", None)

        if xunit:
            xlabel = f"{xlabel} ({xunit})"
        if yunit:
            ylabel = f"{ylabel} ({yunit})"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Show ROIs
        if self._show_roi and hasattr(obj, "roi") and obj.roi:
            for roi in obj.roi:
                _add_single_roi_to_axes(ax, roi, obj)

        # Overlay geometry results
        if self._results is not None:
            result_list = (
                self._results
                if isinstance(self._results, (list, tuple))
                else [self._results]
            )
            for result in result_list:
                _add_geometry_to_axes(ax, result)

    def __repr__(self) -> str:
        """Return string representation."""
        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "Untitled")
        return f"PlotResult({obj_type}: {title})"


class MultiCurvePlotResult:
    """
    Result of a multi-curve plot operation with rich display capabilities.

    Supports plotting multiple SignalObj, numpy arrays, or (x, y) tuples
    on a single plot with automatic styling.
    """

    def __init__(
        self,
        objs: list,
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xunit: str | None = None,
        yunit: str | None = None,
        show_roi: bool = True,
        **kwargs,
    ) -> None:
        """Initialize multi-curve plot result.

        Args:
            objs: List of objects to display (SignalObj, ndarray, or (x, y) tuples)
            title: Plot title
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            xunit: Unit for the x-axis
            yunit: Unit for the y-axis
            show_roi: Whether to show ROIs
            **kwargs: Additional options
        """
        self._objs = objs
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xunit = xunit
        self._yunit = yunit
        self._show_roi = show_roi
        self._kwargs = kwargs

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        try:
            png_data = self._render_to_png()
            b64_data = base64.b64encode(png_data).decode("utf-8")
            title = self._title or "Curves"
            return f"""
            <div style="text-align: center;">
                <h4>{title}</h4>
                <img src="data:image/png;base64,{b64_data}" />
            </div>
            """
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering curves: {e}</div>"

    def _repr_png_(self) -> bytes:
        """Return PNG representation for Jupyter display."""
        return self._render_to_png()

    def _render_to_png(self) -> bytes:
        """Render curves to PNG bytes using matplotlib."""
        # pylint: disable=import-outside-toplevel
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        if self._title:
            fig.suptitle(self._title)

        # Track labels/units from first SignalObj
        x_label = self._xlabel
        y_label = self._ylabel
        x_unit = self._xunit
        y_unit = self._yunit

        for idx, data_or_obj in enumerate(self._objs):
            color, linestyle = _get_next_style(idx)
            obj_type = type(data_or_obj).__name__

            if obj_type == "SignalObj":
                obj = data_or_obj
                xdata, ydata = obj.xydata
                label = obj.title or f"Signal {idx + 1}"

                # Update labels/units from first SignalObj
                if idx == 0:
                    x_label = x_label or getattr(obj, "xlabel", None) or ""
                    y_label = y_label or getattr(obj, "ylabel", None) or ""
                    x_unit = x_unit or getattr(obj, "xunit", None) or ""
                    y_unit = y_unit or getattr(obj, "yunit", None) or ""

                # Plot signal
                ax.plot(xdata, ydata, color=color, linestyle=linestyle, label=label)

                # Plot ROIs if requested
                if self._show_roi and hasattr(obj, "roi") and obj.roi:
                    for roi_idx, single_roi in enumerate(obj.roi):
                        x0, x1 = single_roi.get_physical_coords(obj)
                        ax.axvspan(
                            x0,
                            x1,
                            alpha=0.2,
                            color=color,
                            label=f"{label} ROI {roi_idx + 1}"
                            if roi_idx == 0
                            else None,
                        )

            elif isinstance(data_or_obj, tuple) and len(data_or_obj) == 2:
                # Tuple of (x, y) arrays
                xdata, ydata = data_or_obj
                ax.plot(
                    xdata,
                    ydata,
                    color=color,
                    linestyle=linestyle,
                    label=f"Curve {idx + 1}",
                )

            elif isinstance(data_or_obj, np.ndarray):
                # Just y data, use indices for x
                ydata = data_or_obj
                xdata = np.arange(len(ydata))
                ax.plot(
                    xdata,
                    ydata,
                    color=color,
                    linestyle=linestyle,
                    label=f"Curve {idx + 1}",
                )

            else:
                raise TypeError(f"Unsupported data type: {type(data_or_obj)}")

        # Set axis labels with units
        if x_label:
            ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
        if y_label:
            ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)

        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MultiCurvePlotResult({len(self._objs)} curves)"


class MultiImagePlotResult:
    """
    Result of a multi-image plot operation with rich display capabilities.

    Supports plotting multiple ImageObj or numpy arrays in a grid layout
    with automatic styling, ROI overlays, and geometry result overlays.
    """

    def __init__(
        self,
        objs: list,
        title: str | None = None,
        titles: list[str] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        zlabel: str | None = None,
        xunit: str | None = None,
        yunit: str | None = None,
        zunit: str | None = None,
        show_roi: bool = True,
        results: list | None = None,
        rows: int | None = None,
        share_axes: bool = True,
        **kwargs,
    ) -> None:
        """Initialize multi-image plot result.

        Args:
            objs: List of objects to display (ImageObj or ndarray)
            title: Overall figure title
            titles: Optional list of titles for each image
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            zlabel: Label for the colorbar
            xunit: Unit for the x-axis
            yunit: Unit for the y-axis
            zunit: Unit for the colorbar
            show_roi: Whether to show ROIs
            results: Optional list of GeometryResult objects to overlay
            rows: Fixed number of rows in the grid, or None to compute automatically
            share_axes: Whether to share axes across plots
            **kwargs: Additional options (e.g., colormap)
        """
        self._objs = objs
        self._title = title
        self._titles = titles
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._zlabel = zlabel
        self._xunit = xunit
        self._yunit = yunit
        self._zunit = zunit
        self._show_roi = show_roi
        self._results = results
        self._rows = rows
        self._share_axes = share_axes
        self._kwargs = kwargs

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        try:
            png_data = self._render_to_png()
            b64_data = base64.b64encode(png_data).decode("utf-8")
            title = self._title or "Images"
            return f"""
            <div style="text-align: center;">
                <h4>{title}</h4>
                <img src="data:image/png;base64,{b64_data}" />
            </div>
            """
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering images: {e}</div>"

    def _repr_png_(self) -> bytes:
        """Return PNG representation for Jupyter display."""
        return self._render_to_png()

    def _render_to_png(self) -> bytes:
        """Render images to PNG bytes using matplotlib."""
        # pylint: disable=import-outside-toplevel
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_images = len(self._objs)

        # Compute grid layout
        if self._rows is not None:
            nrows = self._rows
            ncols = (n_images + nrows - 1) // nrows
        else:
            ncols = min(4, n_images)
            nrows = (n_images + ncols - 1) // ncols

        # Create figure
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, 6 * nrows),
            sharex=self._share_axes,
            sharey=self._share_axes,
            squeeze=False,
        )

        if self._title:
            fig.suptitle(self._title)

        # Flatten axes for easier iteration
        axes_flat = axes.flatten()

        # Prepare titles list
        titles = self._titles or [None] * n_images

        # Prepare results list
        if self._results is None:
            results_list = [None] * n_images
        elif isinstance(self._results, (list, tuple)):
            if len(self._results) != n_images:
                # If single result, apply to all images
                results_list = self._results * n_images
            else:
                results_list = self._results
        else:
            results_list = [self._results] * n_images

        # Track labels/units from first ImageObj
        x_label = self._xlabel
        y_label = self._ylabel
        z_label = self._zlabel
        x_unit = self._xunit
        y_unit = self._yunit
        z_unit = self._zunit

        colormap = self._kwargs.get("colormap", "viridis")

        for idx, (ax, img, img_title, result) in enumerate(
            zip(axes_flat, self._objs, titles, results_list)
        ):
            obj_type = type(img).__name__

            # Extract data
            if obj_type == "ImageObj":
                data = img.data
                img_title = (
                    img_title or getattr(img, "title", None) or f"Image {idx + 1}"
                )
                is_image_obj = True

                # Update labels/units from first ImageObj
                if idx == 0:
                    x_label = x_label or getattr(img, "xlabel", None) or ""
                    y_label = y_label or getattr(img, "ylabel", None) or ""
                    z_label = z_label or getattr(img, "zlabel", None) or ""
                    x_unit = x_unit or getattr(img, "xunit", None) or ""
                    y_unit = y_unit or getattr(img, "yunit", None) or ""
                    z_unit = z_unit or getattr(img, "zunit", None) or ""
            elif isinstance(img, np.ndarray):
                data = img
                img_title = img_title or f"Image {idx + 1}"
                is_image_obj = False
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # Handle complex data
            if np.iscomplexobj(data):
                data = np.abs(data)
                img_title = f"|{img_title}|"

            # Display image
            im = ax.imshow(data, cmap=colormap, origin="upper")
            ax.set_title(img_title)

            # Overlay mask if ImageObj has maskdata
            if is_image_obj and hasattr(img, "maskdata") and img.maskdata is not None:
                mask = img.maskdata
                mask_rgba = np.zeros((*mask.shape, 4))
                mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
                ax.imshow(mask_rgba, origin="upper")

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if z_label:
                cbar.set_label(f"{z_label} ({z_unit})" if z_unit else z_label)

            # Set axis labels
            if x_label:
                ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
            if y_label:
                ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)

            # Overlay ROIs
            if self._show_roi and is_image_obj and hasattr(img, "roi") and img.roi:
                for roi in img.roi:
                    _add_single_roi_to_axes(ax, roi, img)

            # Overlay geometry results
            if result is not None:
                result_list_item = (
                    result if isinstance(result, (list, tuple)) else [result]
                )
                for res in result_list_item:
                    _add_geometry_to_axes(ax, res)

        # Hide unused subplots
        for idx in range(n_images, len(axes_flat)):
            axes_flat[idx].axis("off")

        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MultiImagePlotResult({len(self._objs)} images)"
