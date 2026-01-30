# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Plotter API
===========

The Plotter class provides visualization capabilities for the DataLab kernel.
It supports inline notebook display and optional DataLab GUI synchronization.

This module provides matplotlib-based visualization for signals and images,
including support for:
- Multiple signals on a single plot
- Multiple images in grid layout
- ROI (Region of Interest) visualization
- Geometry result overlays (points, markers, rectangles, circles, ellipses,
  segments, polygons)
- Table result display with rich HTML rendering
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

# Style configuration for multi-signal/multi-image plots
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive"]
LINESTYLES = ["-", "--", "-.", ":"]
MASK_OPACITY = 0.35  # Opacity for mask overlay

# Metadata prefix for geometry results (consistent with DataLab's GeometryAdapter)
GEOMETRY_META_PREFIX = "Geometry_"
# Metadata prefix for table results (consistent with DataLab's TableAdapter)
TABLE_META_PREFIX = "Table_"


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


def _extract_geometry_results_from_metadata(obj) -> list:
    """Extract GeometryResult objects from object metadata.

    DataLab stores geometry results in object metadata with keys starting with
    'Geometry_'. This function extracts and reconstructs those GeometryResult
    objects for visualization.

    Args:
        obj: SignalObj or ImageObj with potential geometry results in metadata

    Returns:
        List of GeometryResult objects extracted from metadata
    """
    results = []
    if not hasattr(obj, "metadata") or obj.metadata is None:
        return results

    # Delayed import
    # pylint: disable=import-outside-toplevel
    from sigima.objects import GeometryResult

    for key, value in obj.metadata.items():
        if key.startswith(GEOMETRY_META_PREFIX) and isinstance(value, dict):
            try:
                geometry = GeometryResult.from_dict(value)
                results.append(geometry)
            except (ValueError, TypeError, KeyError):
                # Skip invalid entries
                pass

    return results


def _extract_table_results_from_metadata(obj) -> list:
    """Extract TableResult objects from object metadata.

    DataLab stores table results in object metadata with keys starting with
    'Table_'. This function extracts and reconstructs those TableResult
    objects for visualization.

    Args:
        obj: SignalObj or ImageObj with potential table results in metadata

    Returns:
        List of TableResult objects extracted from metadata
    """
    results = []
    if not hasattr(obj, "metadata") or obj.metadata is None:
        return results

    # Delayed import
    # pylint: disable=import-outside-toplevel
    from sigima.objects import TableResult

    for key, value in obj.metadata.items():
        if key.startswith(TABLE_META_PREFIX) and isinstance(value, dict):
            try:
                table = TableResult.from_dict(value)
                results.append(table)
            except (ValueError, TypeError, KeyError):
                # Skip invalid entries
                pass

    return results


def _add_table_results_to_axes(
    ax: Axes, table_results: list, geometry_results: list | None = None
) -> None:
    """Add table and geometry results as text annotation to matplotlib axes.

    Formats TableResult and GeometryResult objects as a text box displayed in
    the upper-left corner of the axes, similar to DataLab's result label display.

    Args:
        ax: Matplotlib axes object
        table_results: List of TableResult objects to display
        geometry_results: Optional list of GeometryResult objects to display
    """
    if not table_results and not geometry_results:
        return

    # Build text content from all results
    text_lines = []

    # Add table results first (statistics)
    for table in table_results:
        # Add table title as header
        text_lines.append(f"{table.title}:")

        # Get headers and data
        headers = list(table.headers)
        data = table.data

        # Format each row (typically just one row for statistics)
        for row in data:
            for header, value in zip(headers, row):
                # Format numeric values
                if isinstance(value, float):
                    if abs(value) < 0.001 or abs(value) >= 10000:
                        formatted = f"{value:.3g}"
                    else:
                        formatted = f"{value:.3f}"
                else:
                    formatted = str(value)
                text_lines.append(f"  {header}: {formatted}")

        text_lines.append("")  # Empty line between results

    # Add geometry results after table results
    if geometry_results:
        for geometry in geometry_results:
            text_lines.append(f"{geometry.title}:")
            text_lines.append("  Value")

            # Get coordinate labels based on geometry kind
            coord_labels = _get_geometry_coord_labels(geometry)

            # Display first row of coords (most geometry results have one row)
            if len(geometry.coords) > 0:
                coords = (
                    geometry.coords[0] if geometry.coords.ndim > 1 else geometry.coords
                )
                for label, value in zip(coord_labels, coords):
                    if isinstance(value, float):
                        if abs(value) < 0.001 or abs(value) >= 10000:
                            formatted = f"{value:.3g}"
                        else:
                            formatted = f"{value:.3f}"
                    else:
                        formatted = str(value)
                    text_lines.append(f"  {label}: {formatted}")

            text_lines.append("")  # Empty line between results

    # Remove trailing empty line
    if text_lines and text_lines[-1] == "":
        text_lines.pop()

    text = "\n".join(text_lines)

    # Add text box annotation in upper-left corner
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.85,
        },
    )


def _get_geometry_coord_labels(geometry) -> list[str]:
    """Get coordinate labels for a geometry result based on its kind.

    Args:
        geometry: GeometryResult object

    Returns:
        List of coordinate labels (e.g., ["x", "y", "r"] for circle)
    """
    # Delayed import
    # pylint: disable=import-outside-toplevel
    from sigima.objects import KindShape

    if geometry.kind == KindShape.POINT:
        return ["x", "y"]
    if geometry.kind == KindShape.MARKER:
        return ["x", "y"]
    if geometry.kind == KindShape.RECTANGLE:
        return ["x0", "y0", "dx", "dy"]
    if geometry.kind == KindShape.CIRCLE:
        return ["x", "y", "r"]
    if geometry.kind == KindShape.SEGMENT:
        return ["x0", "y0", "x1", "y1"]
    if geometry.kind == KindShape.ELLIPSE:
        return ["x", "y", "a", "b", "θ"]
    # Default for POLYGON and others
    return [
        f"c{i}"
        for i in range(
            len(geometry.coords[0])
            if geometry.coords.ndim > 1
            else len(geometry.coords)
        )
    ]


def _get_image_extent_and_aspect(obj) -> tuple[list[float], float]:
    """Compute matplotlib extent and aspect ratio from image physical coordinates.

    DataLab images use physical coordinates defined by:
    - x0, y0: Origin (center of top-left pixel)
    - dx, dy: Pixel spacing

    For matplotlib's imshow:
    - extent defines pixel edges: [left, right, bottom, top]
    - aspect ratio is dx/dy to preserve physical proportions

    With origin="upper", matplotlib expects:
    - extent = [xmin - dx/2, xmax + dx/2, ymax + dy/2, ymin - dy/2]

    Args:
        obj: ImageObj with physical coordinate attributes

    Returns:
        Tuple of (extent, aspect_ratio) where:
        - extent: [left, right, bottom, top] for imshow
        - aspect_ratio: dx/dy for proper physical display
    """
    # Get image shape
    nrows, ncols = obj.data.shape[:2]

    # Check if image has physical coordinates
    has_coords = hasattr(obj, "x0") and hasattr(obj, "dx")

    if has_coords:
        x0 = getattr(obj, "x0", 0.0)
        y0 = getattr(obj, "y0", 0.0)
        dx = getattr(obj, "dx", 1.0)
        dy = getattr(obj, "dy", 1.0)

        # Compute pixel centers range (as in Sigima)
        xmin = x0  # Center of leftmost column
        xmax = x0 + (ncols - 1) * dx  # Center of rightmost column
        ymin = y0  # Center of topmost row
        ymax = y0 + (nrows - 1) * dy  # Center of bottommost row

        # Convert to pixel edges for matplotlib extent
        # extent = [left, right, bottom, top]
        # For origin="upper", bottom is ymax and top is ymin
        left = xmin - dx / 2
        right = xmax + dx / 2
        bottom = ymax + dy / 2  # Lower edge of bottom-most pixel
        top = ymin - dy / 2  # Upper edge of top-most pixel

        extent = [left, right, bottom, top]

        # Aspect ratio preserves physical pixel proportions
        aspect_ratio = dx / dy
    else:
        # No physical coordinates, use pixel indices
        extent = [-0.5, ncols - 0.5, nrows - 0.5, -0.5]
        aspect_ratio = 1.0

    return extent, aspect_ratio


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

    Iterates over all rows in result.coords to draw each geometric shape.
    Supports POINT, MARKER, RECTANGLE, CIRCLE, SEGMENT, ELLIPSE, and POLYGON.

    Args:
        ax: Matplotlib axes object
        result: GeometryResult object with shape information (coords is 2D array)
    """
    # Delayed import
    # pylint: disable=import-outside-toplevel
    from matplotlib import patches
    from sigima.objects import KindShape

    # Iterate over all rows in coords (each row is one shape)
    for coords in result.coords:
        if result.kind == KindShape.POINT:
            x0, y0 = coords
            ax.plot(
                x0,
                y0,
                marker="o",
                markersize=6,
                color="yellow",
                markeredgecolor="black",
                markeredgewidth=1,
            )
        elif result.kind == KindShape.MARKER:
            x0, y0 = coords
            # Marker with crosshair style
            ax.axhline(y0, color="yellow", linestyle="--", linewidth=1, alpha=0.7)
            ax.axvline(x0, color="yellow", linestyle="--", linewidth=1, alpha=0.7)
            ax.plot(
                x0,
                y0,
                marker="+",
                markersize=10,
                color="yellow",
                markeredgewidth=2,
            )
        elif result.kind == KindShape.RECTANGLE:
            x0, y0, dx, dy = coords
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
            xc, yc, r = coords
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
            x0, y0, x1, y1 = coords
            ax.plot([x0, x1], [y0, y1], "y--", linewidth=2)
        elif result.kind == KindShape.ELLIPSE:
            # For ellipse, coords are (xc, yc, a, b, theta)
            xc, yc, a, b, theta = coords
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
            x = coords[::2]
            y = coords[1::2]
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
        plotter.plot_signals([sig1, sig2, sig3])

        # Plot multiple images
        plotter.plot_images([img1, img2])

        # Display table results (e.g., from statistics computation)
        result = proxy.compute_statistics()
        plotter.display_table(result)

        # Display geometry results (e.g., from peak detection)
        result = proxy.compute_peak_detection()
        plotter.display_geometry(result)
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
        show_results: bool = True,
        **kwargs,
    ) -> PlotResult:
        """Plot an object or retrieve and plot by name.

        Args:
            obj_or_name: Object to plot, or name of object in workspace
            title: Optional plot title override
            show_roi: Whether to show ROIs defined in the object
            show_results: Whether to show geometry/table results from metadata
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

        return PlotResult(
            obj, title=title, show_roi=show_roi, show_results=show_results, **kwargs
        )

    def plot_signals(
        self,
        objs_or_names: list[DataObject | str | np.ndarray | tuple[np.ndarray, ...]],
        title: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xunit: str | None = None,
        yunit: str | None = None,
        show_roi: bool = True,
        show_results: bool = True,
        **kwargs,
    ) -> MultiSignalPlotResult:
        """Plot multiple signals on a single plot.

        Args:
            objs_or_names: List of signals to plot. Can be SignalObj, workspace names,
             numpy arrays (y data), or tuples of (x, y) arrays
            title: Optional plot title
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            xunit: Unit for the x-axis
            yunit: Unit for the y-axis
            show_roi: Whether to show ROIs defined in SignalObj instances
            show_results: Whether to show geometry/table results from metadata
            **kwargs: Additional plotting options

        Returns:
            MultiSignalPlotResult with display capabilities
        """
        objs = []
        for obj_or_name in objs_or_names:
            if isinstance(obj_or_name, str):
                objs.append(self._workspace.get(obj_or_name))
            else:
                objs.append(obj_or_name)

        return MultiSignalPlotResult(
            objs,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            show_roi=show_roi,
            show_results=show_results,
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
        show_results: bool = True,
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
            show_results: Whether to show geometry/table results from metadata
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
            show_results=show_results,
            results=results,
            **kwargs,
        )

    def display_table(
        self,
        result,
        title: str | None = None,
        visible_only: bool = True,
        transpose_single_row: bool = True,
    ) -> TableResultDisplay:
        """Display a TableResult with rich HTML rendering.

        Args:
            result: TableResult object to display
            title: Optional title override (uses result.title if None)
            visible_only: If True, show only visible columns based on display prefs
            transpose_single_row: If True, transpose single-row tables for readability

        Returns:
            TableResultDisplay with Jupyter display capabilities
        """
        return TableResultDisplay(
            result,
            title=title,
            visible_only=visible_only,
            transpose_single_row=transpose_single_row,
        )

    def display_geometry(
        self,
        result,
        title: str | None = None,
    ) -> GeometryResultDisplay:
        """Display a GeometryResult with rich HTML rendering.

        Args:
            result: GeometryResult object to display
            title: Optional title override (uses result.title if None)

        Returns:
            GeometryResultDisplay with Jupyter display capabilities
        """
        return GeometryResultDisplay(result, title=title)


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
        show_results: bool = True,
        results: list | None = None,
        **kwargs,
    ) -> None:
        """Initialize plot result.

        Args:
            obj: Object to display
            title: Plot title
            show_roi: Whether to show ROIs
            show_results: Whether to show geometry/table results from metadata
            results: Optional list of GeometryResult objects to overlay (for images)
            **kwargs: Additional options
        """
        self._obj = obj
        self._title = title
        self._show_roi = show_roi
        self._show_results = show_results
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

        # Auto-extract and display geometry/table results from object metadata
        if self._show_results:
            metadata_results = _extract_geometry_results_from_metadata(obj)
            for result in metadata_results:
                _add_geometry_to_axes(ax, result)

            table_results = _extract_table_results_from_metadata(obj)
            _add_table_results_to_axes(ax, table_results, metadata_results)

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

        # Compute extent and aspect ratio from physical coordinates
        extent, aspect_ratio = _get_image_extent_and_aspect(obj)

        im = ax.imshow(
            data, aspect=aspect_ratio, origin="upper", cmap=colormap, extent=extent
        )

        # Overlay mask if present
        if hasattr(obj, "maskdata") and obj.maskdata is not None:
            mask = obj.maskdata
            mask_rgba = np.zeros((*mask.shape, 4))
            mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
            ax.imshow(mask_rgba, origin="upper", extent=extent)

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

        # Overlay geometry/table results (explicit or from metadata)
        if self._show_results:
            results_to_display = []
            if self._results is not None:
                result_list = (
                    self._results
                    if isinstance(self._results, (list, tuple))
                    else [self._results]
                )
                results_to_display.extend(result_list)

            # Auto-extract geometry results from object metadata
            metadata_results = _extract_geometry_results_from_metadata(obj)
            results_to_display.extend(metadata_results)

            for result in results_to_display:
                _add_geometry_to_axes(ax, result)

            # Auto-extract and display table results (statistics) from metadata
            table_results = _extract_table_results_from_metadata(obj)
            _add_table_results_to_axes(ax, table_results, results_to_display)

    def __repr__(self) -> str:
        """Return string representation."""
        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "Untitled")
        return f"PlotResult({obj_type}: {title})"


class MultiSignalPlotResult:
    """
    Result of a multi-signal plot operation with rich display capabilities.

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
        show_results: bool = True,
        **kwargs,
    ) -> None:
        """Initialize multi-signal plot result.

        Args:
            objs: List of objects to display (SignalObj, ndarray, or (x, y) tuples)
            title: Plot title
            xlabel: Label for the x-axis
            ylabel: Label for the y-axis
            xunit: Unit for the x-axis
            yunit: Unit for the y-axis
            show_roi: Whether to show ROIs
            show_results: Whether to show geometry/table results from metadata
            **kwargs: Additional options
        """
        self._objs = objs
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xunit = xunit
        self._yunit = yunit
        self._show_roi = show_roi
        self._show_results = show_results
        self._kwargs = kwargs

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        try:
            png_data = self._render_to_png()
            b64_data = base64.b64encode(png_data).decode("utf-8")
            title = self._title or "Signals"
            return f"""
            <div style="text-align: center;">
                <h4>{title}</h4>
                <img src="data:image/png;base64,{b64_data}" />
            </div>
            """
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering signals: {e}</div>"

    def _repr_png_(self) -> bytes:
        """Return PNG representation for Jupyter display."""
        return self._render_to_png()

    def _render_to_png(self) -> bytes:
        """Render signals to PNG bytes using matplotlib."""
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

                # Auto-extract and display geometry/table results from metadata
                if self._show_results:
                    metadata_results = _extract_geometry_results_from_metadata(obj)
                    for result in metadata_results:
                        _add_geometry_to_axes(ax, result)

                    table_results = _extract_table_results_from_metadata(obj)
                    _add_table_results_to_axes(ax, table_results, metadata_results)

            elif isinstance(data_or_obj, tuple) and len(data_or_obj) == 2:
                # Tuple of (x, y) arrays
                xdata, ydata = data_or_obj
                ax.plot(
                    xdata,
                    ydata,
                    color=color,
                    linestyle=linestyle,
                    label=f"Signal {idx + 1}",
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
                    label=f"Signal {idx + 1}",
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
        return f"MultiSignalPlotResult({len(self._objs)} signals)"


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
        show_results: bool = True,
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
            show_results: Whether to show geometry/table results from metadata
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
        self._show_results = show_results
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

            # Compute extent and aspect ratio for ImageObj, use defaults for arrays
            if is_image_obj:
                extent, aspect_ratio = _get_image_extent_and_aspect(img)
            else:
                nrows_img, ncols_img = data.shape[:2]
                extent = [-0.5, ncols_img - 0.5, nrows_img - 0.5, -0.5]
                aspect_ratio = 1.0

            # Display image
            im = ax.imshow(
                data,
                cmap=colormap,
                origin="upper",
                aspect=aspect_ratio,
                extent=extent,
            )
            ax.set_title(img_title)

            # Overlay mask if ImageObj has maskdata
            if is_image_obj and hasattr(img, "maskdata") and img.maskdata is not None:
                mask = img.maskdata
                mask_rgba = np.zeros((*mask.shape, 4))
                mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
                ax.imshow(mask_rgba, origin="upper", extent=extent)

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

            # Collect and display geometry/table results if enabled
            if self._show_results:
                results_to_display = []
                if result is not None:
                    result_list_item = (
                        result if isinstance(result, (list, tuple)) else [result]
                    )
                    results_to_display.extend(result_list_item)

                # Auto-extract geometry results from object metadata
                if is_image_obj:
                    metadata_results = _extract_geometry_results_from_metadata(img)
                    results_to_display.extend(metadata_results)

                for res in results_to_display:
                    _add_geometry_to_axes(ax, res)

                # Auto-extract and display table results (statistics) from metadata
                if is_image_obj:
                    table_results = _extract_table_results_from_metadata(img)
                    _add_table_results_to_axes(ax, table_results, results_to_display)

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


class TableResultDisplay:
    """
    Display wrapper for TableResult with rich Jupyter notebook rendering.

    Provides HTML table display with automatic formatting, optional DataFrame
    conversion, and support for ROI-indexed results.

    Example::

        # Display a TableResult from computation
        result = proxy.compute_statistics()
        display = TableResultDisplay(result)
        display  # Shows styled HTML table in Jupyter

        # Get as DataFrame for further analysis
        df = display.to_dataframe()
    """

    # CSS styling for HTML tables
    _TABLE_STYLE = """
    <style>
        .sigima-table {
            border-collapse: collapse;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                         'Helvetica Neue', Arial, sans-serif;
            font-size: 13px;
            margin: 10px 0;
        }
        .sigima-table th {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
        }
        .sigima-table td {
            border: 1px solid #dee2e6;
            padding: 8px 12px;
            text-align: right;
        }
        .sigima-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .sigima-table tr:hover {
            background-color: #e9ecef;
        }
        .sigima-table-title {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #495057;
        }
    </style>
    """

    def __init__(
        self,
        result,
        title: str | None = None,
        visible_only: bool = True,
        transpose_single_row: bool = True,
    ) -> None:
        """Initialize TableResult display.

        Args:
            result: TableResult object to display
            title: Optional title override (uses result.title if None)
            visible_only: If True, show only visible columns based on display prefs
            transpose_single_row: If True, transpose single-row tables for readability
        """
        self._result = result
        self._title = title
        self._visible_only = visible_only
        self._transpose_single_row = transpose_single_row

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        try:
            title = self._title or self._result.title

            # Use TableResult's built-in to_html if available
            if hasattr(self._result, "to_html"):
                table_html = self._result.to_html(
                    visible_only=self._visible_only,
                    transpose_single_row=self._transpose_single_row,
                )
                return f"""
                {self._TABLE_STYLE}
                <div>
                    <div class="sigima-table-title">{title}</div>
                    {table_html}
                </div>
                """

            # Fallback: manual HTML generation from DataFrame
            df = self.to_dataframe()

            # Transpose single-row tables for better readability
            if self._transpose_single_row and len(df) == 1:
                df = df.T
                df.columns = ["Value"]

            # Format numbers for display
            html_table = df.to_html(
                classes="sigima-table",
                float_format=lambda x: f"{x:.6g}" if isinstance(x, float) else str(x),
            )

            return f"""
            {self._TABLE_STYLE}
            <div>
                <div class="sigima-table-title">{title}</div>
                {html_table}
            </div>
            """
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering table: {e}</div>"

    def to_dataframe(self):
        """Convert the TableResult to a pandas DataFrame.

        Returns:
            pandas DataFrame with result data
        """
        if hasattr(self._result, "to_dataframe"):
            return self._result.to_dataframe(visible_only=self._visible_only)

        # Fallback: manual DataFrame creation
        # pylint: disable=import-outside-toplevel
        import pandas as pd

        df = pd.DataFrame(self._result.data, columns=list(self._result.headers))
        if self._result.roi_indices is not None:
            df.insert(0, "roi_index", self._result.roi_indices)
        return df

    def __repr__(self) -> str:
        """Return string representation."""
        result_type = type(self._result).__name__
        title = self._title or getattr(self._result, "title", "Untitled")
        n_rows = len(self._result.data) if hasattr(self._result, "data") else "?"
        n_cols = len(self._result.headers) if hasattr(self._result, "headers") else "?"
        return f"TableResultDisplay({result_type}: {title}, {n_rows}×{n_cols})"


class GeometryResultDisplay:
    """
    Display wrapper for GeometryResult with rich Jupyter notebook rendering.

    Provides HTML table display showing coordinates and metadata for geometric
    results like points, segments, circles, ellipses, rectangles, and polygons.

    Example::

        # Display a GeometryResult from computation
        result = proxy.compute_peak_detection()
        display = GeometryResultDisplay(result)
        display  # Shows styled HTML table in Jupyter

        # Get as DataFrame for further analysis
        df = display.to_dataframe()
    """

    def __init__(
        self,
        result,
        title: str | None = None,
    ) -> None:
        """Initialize GeometryResult display.

        Args:
            result: GeometryResult object to display
            title: Optional title override (uses result.title if None)
        """
        self._result = result
        self._title = title

    def _repr_html_(self) -> str:
        """Return HTML representation for Jupyter display."""
        try:
            title = self._title or self._result.title

            # Use GeometryResult's built-in to_html if available
            if hasattr(self._result, "to_html"):
                table_html = self._result.to_html()
                return f"""
                {TableResultDisplay._TABLE_STYLE}
                <div>
                    <div class="sigima-table-title">{title}</div>
                    {table_html}
                </div>
                """

            # Fallback: manual HTML generation from DataFrame
            df = self.to_dataframe()

            # Format numbers for display
            html_table = df.to_html(
                classes="sigima-table",
                float_format=lambda x: f"{x:.6g}" if isinstance(x, float) else str(x),
            )

            return f"""
            {TableResultDisplay._TABLE_STYLE}
            <div>
                <div class="sigima-table-title">{title} ({self._result.kind.value})</div>
                {html_table}
            </div>
            """
        except Exception as e:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering geometry: {e}</div>"

    def to_dataframe(self):
        """Convert the GeometryResult to a pandas DataFrame.

        Returns:
            pandas DataFrame with coordinate data
        """
        if hasattr(self._result, "to_dataframe"):
            return self._result.to_dataframe()

        # Fallback: manual DataFrame creation based on kind
        # pylint: disable=import-outside-toplevel
        import pandas as pd
        from sigima.objects import KindShape

        coords = self._result.coords
        kind = self._result.kind

        # Create column names based on geometry kind
        if kind == KindShape.POINT or kind == KindShape.MARKER:
            columns = ["x", "y"]
        elif kind == KindShape.SEGMENT:
            columns = ["x0", "y0", "x1", "y1"]
        elif kind == KindShape.RECTANGLE:
            columns = ["x0", "y0", "width", "height"]
        elif kind == KindShape.CIRCLE:
            columns = ["xc", "yc", "radius"]
        elif kind == KindShape.ELLIPSE:
            columns = ["xc", "yc", "a", "b", "theta"]
        elif kind == KindShape.POLYGON:
            # Variable number of columns for polygons
            n_coords = coords.shape[1]
            columns = [
                f"x{i // 2}" if i % 2 == 0 else f"y{i // 2}" for i in range(n_coords)
            ]
        else:
            columns = [f"c{i}" for i in range(coords.shape[1])]

        df = pd.DataFrame(coords, columns=columns)
        if self._result.roi_indices is not None:
            df.insert(0, "roi_index", self._result.roi_indices)
        return df

    def __repr__(self) -> str:
        """Return string representation."""
        title = self._title or getattr(self._result, "title", "Untitled")
        kind = getattr(self._result, "kind", "?")
        n_rows = len(self._result.coords) if hasattr(self._result, "coords") else "?"
        return f"GeometryResultDisplay({kind}: {title}, {n_rows} items)"
