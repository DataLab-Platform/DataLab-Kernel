# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Matplotlib Backend
==================

Matplotlib-based visualization backend for the DataLab kernel.

This module provides static PNG rendering for signals and images using
matplotlib with the Agg backend.  It is the default backend and works
in all environments (JupyterLite, standard Jupyter, scripts).

Features include:

- Single and multi-signal plotting with error bars, curve styles, and legend
- Single and multi-image plotting in grid layout with colorbars
- ROI (Region of Interest) overlays
- Geometry result overlays (points, markers, rectangles, circles, ellipses,
  segments, polygons)
- Table/geometry result annotation text boxes
- Mask visualization with semi-transparent overlay
- Axis labels with units, log scale, and axis bounds
- Colormap support with LUT range
- Non-uniform image coordinates (pcolormesh)

Usage::

    from datalab_kernel.matplotlib_backend import MatplotlibPlotter

    plotter = MatplotlibPlotter(workspace)
    plotter.plot("s001")                    # Single signal
    plotter.plot([sig1, sig2])               # Multiple signals
    plotter.plot([img1, img2])               # Multiple images
    plotter.display_table(table_result)      # HTML table
    plotter.display_geometry(geom_result)    # HTML table
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

import numpy as np

from datalab_kernel.plotter import (
    DEFAULT_PLOT_WIDTH,
    MASK_OPACITY,
    GeometryResultDisplay,
    TableResultDisplay,
    _apply_axis_bounds,
    _apply_log_scale,
    _extract_geometry_results_from_metadata,
    _extract_table_results_from_metadata,
    _get_curve_style,
    _get_geometry_coord_labels,
    _get_image_colormap,
    _get_image_extent_and_aspect,
    _get_image_lut_range,
    _get_signal_style_from_metadata,
    _is_non_uniform_image,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from datalab_kernel.workspace import DataObject, Workspace


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Color palette for multi-signal/multi-image cycling.
COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive"]
#: Linestyle palette for multi-signal cycling.
LINESTYLES = ["-", "--", "-.", ":"]


# ---------------------------------------------------------------------------
# Internal helpers (matplotlib-specific)
# ---------------------------------------------------------------------------


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


# ============================================================================
# Result classes
# ============================================================================


class MplPlotResult:
    """Result of a single-object plot rendered via matplotlib.

    Supports Jupyter's rich display protocol: ``_repr_png_()`` returns
    static PNG bytes and ``_repr_html_()`` returns an embedded PNG in HTML.
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
            **kwargs: Additional options (e.g., ``colormap``,
             ``height`` to override the default figure height in pixels)
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

        if obj_type in ("SignalObj", "ImageObj"):
            try:
                png_data = self._render_to_png()
                b64_data = base64.b64encode(png_data).decode("utf-8")
                return (
                    '<div style="text-align: center;">'
                    f"<h4>{title}</h4>"
                    f'<img src="data:image/png;base64,{b64_data}" />'
                    "</div>"
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"<div>Error rendering {obj_type}: {e}</div>"
        return f"<div><strong>{title}</strong>: {obj_type}</div>"

    def _repr_png_(self) -> bytes:
        """Return PNG representation for Jupyter display."""
        return self._render_to_png()

    def _render_to_png(self) -> bytes:
        """Render object to PNG bytes using matplotlib."""
        # Delayed import: matplotlib is optional and heavy
        # pylint: disable=import-outside-toplevel
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "")

        # Figure size: honour user height override (pixels → inches at 100 dpi)
        _dpi = 100
        _h_in = self._kwargs.get("height", 500) / _dpi
        _w_in = DEFAULT_PLOT_WIDTH / _dpi
        fig, ax = plt.subplots(figsize=(_w_in, _h_in))

        if obj_type == "SignalObj":
            self._render_signal(ax)
        elif obj_type == "ImageObj":
            self._render_image(ax, fig)

        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, pil_kwargs={"compress_level": 1})
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _render_signal(self, ax: Axes) -> None:
        """Render signal data to axes.

        Supports error bars (dx/dy), curve styles (Lines/Sticks/Steps),
        per-object line color/style from metadata, log scale, and axis bounds.

        Args:
            ax: Matplotlib axes object
        """
        obj = self._obj
        x = obj.x
        y = obj.y

        # Determine base style: metadata overrides > defaults
        base_style = {"linewidth": 1, "color": "blue"}
        meta_style = _get_signal_style_from_metadata(obj)
        base_style.update(meta_style)

        # Determine curve style
        curvestyle = _get_curve_style(obj)

        has_errorbars = obj.dy is not None or obj.dx is not None

        if curvestyle == "Sticks":
            stem_color = base_style.get("color", "blue")
            ax.stem(
                x,
                y,
                linefmt=f"{stem_color}",
                markerfmt=" ",
                basefmt="k-",
            )
        elif curvestyle == "Steps":
            ax.step(x, y, where="mid", **base_style)
        elif has_errorbars:
            # Use errorbar instead of plot when error data is available
            ax.errorbar(
                x,
                y,
                yerr=obj.dy,
                xerr=obj.dx,
                fmt="-",
                linewidth=base_style.get("linewidth", 1),
                color=base_style.get("color", "blue"),
                capsize=3,
                elinewidth=0.8,
            )
        else:
            ax.plot(x, y, **base_style)

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

        # Log scale
        _apply_log_scale(ax, obj)

        # Axis bounds (when autoscale is disabled)
        _apply_axis_bounds(ax, obj)

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

        Supports per-object colormap from metadata, LUT range (vmin/vmax),
        non-uniform coordinates (pcolormesh), log scale, and axis bounds.

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

        # Colormap: explicit kwarg > object metadata > default
        colormap = _get_image_colormap(obj, self._kwargs)

        # LUT range from object attributes
        vmin, vmax = _get_image_lut_range(obj)

        # Check for non-uniform coordinates
        if _is_non_uniform_image(obj):
            # Use pcolormesh for non-uniform grids
            im = ax.pcolormesh(
                obj.xcoords,
                obj.ycoords,
                data,
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
                shading="auto",
            )
            ax.set_aspect("auto")
            ax.invert_yaxis()
        else:
            # Compute extent and aspect ratio from physical coordinates
            extent, aspect_ratio = _get_image_extent_and_aspect(obj)

            im = ax.imshow(
                data,
                aspect=aspect_ratio,
                origin="upper",
                cmap=colormap,
                extent=extent,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )

            # Overlay mask if present
            if hasattr(obj, "maskdata") and obj.maskdata is not None:
                mask = obj.maskdata
                mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
                ax.imshow(
                    mask_rgba,
                    origin="upper",
                    extent=extent,
                    interpolation="nearest",
                )

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

        # Log scale
        _apply_log_scale(ax, obj)

        # Axis bounds (when autoscale is disabled)
        _apply_axis_bounds(ax, obj)

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
        return f"MplPlotResult({obj_type}: {title})"


class MplMultiSignalResult:
    """Result of a multi-signal plot rendered via matplotlib.

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
            return (
                '<div style="text-align: center;">'
                f"<h4>{title}</h4>"
                f'<img src="data:image/png;base64,{b64_data}" />'
                "</div>"
            )
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

        fig, ax = plt.subplots(figsize=(DEFAULT_PLOT_WIDTH / 100, 6))

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
                xdata = obj.x
                ydata = obj.y
                label = obj.title or f"Signal {idx + 1}"

                # Update labels/units from first SignalObj
                if idx == 0:
                    x_label = x_label or getattr(obj, "xlabel", None) or ""
                    y_label = y_label or getattr(obj, "ylabel", None) or ""
                    x_unit = x_unit or getattr(obj, "xunit", None) or ""
                    y_unit = y_unit or getattr(obj, "yunit", None) or ""

                # Determine style: metadata overrides > cycling defaults
                plot_style = {"color": color, "linestyle": linestyle}
                meta_style = _get_signal_style_from_metadata(obj)
                plot_style.update(meta_style)

                # Determine curve style
                curvestyle = _get_curve_style(obj)

                has_errorbars = obj.dy is not None or obj.dx is not None

                if curvestyle == "Sticks":
                    stem_color = plot_style.get("color", color)
                    ax.stem(
                        xdata,
                        ydata,
                        linefmt=f"{stem_color}",
                        markerfmt=" ",
                        basefmt="k-",
                        label=label,
                    )
                elif curvestyle == "Steps":
                    ax.step(xdata, ydata, where="mid", label=label, **plot_style)
                elif has_errorbars:
                    ax.errorbar(
                        xdata,
                        ydata,
                        yerr=obj.dy,
                        xerr=obj.dx,
                        fmt="-",
                        linewidth=plot_style.get("linewidth", 1),
                        color=plot_style.get("color", color),
                        capsize=3,
                        elinewidth=0.8,
                        label=label,
                    )
                else:
                    ax.plot(xdata, ydata, label=label, **plot_style)

                # Log scale (apply from first SignalObj)
                if idx == 0:
                    _apply_log_scale(ax, obj)
                    _apply_axis_bounds(ax, obj)

                # Plot ROIs if requested
                if self._show_roi and hasattr(obj, "roi") and obj.roi:
                    for roi_idx, single_roi in enumerate(obj.roi):
                        x0, x1 = single_roi.get_physical_coords(obj)
                        ax.axvspan(
                            x0,
                            x1,
                            alpha=0.2,
                            color=plot_style.get("color", color),
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
        fig.savefig(buf, format="png", dpi=100, pil_kwargs={"compress_level": 1})
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MplMultiSignalResult({len(self._objs)} signals)"


class MplMultiImageResult:
    """Result of a multi-image plot rendered via matplotlib.

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
            **kwargs: Additional options (e.g., ``colormap``,
             ``height`` to override default per-subplot height in pixels)
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
            return (
                '<div style="text-align: center;">'
                f"<h4>{title}</h4>"
                f'<img src="data:image/png;base64,{b64_data}" />'
                "</div>"
            )
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

        # Create figure — honour user height override (pixels → inches at 100 dpi)
        _dpi = 100
        _h_in = self._kwargs.get("height", 600) / _dpi
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6 * ncols, _h_in * nrows),
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

        default_colormap = self._kwargs.get("colormap", None)

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

            # Determine colormap: explicit kwarg > per-object metadata > viridis
            if is_image_obj:
                kwargs_for_cmap = self._kwargs if default_colormap else {}
                colormap = _get_image_colormap(img, kwargs_for_cmap)
            else:
                colormap = default_colormap or "viridis"

            # LUT range for ImageObj
            vmin, vmax = _get_image_lut_range(img) if is_image_obj else (None, None)

            # Non-uniform image coordinates
            if is_image_obj and _is_non_uniform_image(img):
                im = ax.pcolormesh(
                    img.xcoords,
                    img.ycoords,
                    data,
                    cmap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    shading="auto",
                )
                ax.set_aspect("auto")
                ax.invert_yaxis()
                extent = None  # No extent for pcolormesh
            else:
                # Compute extent and aspect ratio
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
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )

            ax.set_title(img_title)

            # Overlay mask if ImageObj has maskdata
            if (
                is_image_obj
                and extent is not None
                and hasattr(img, "maskdata")
                and img.maskdata is not None
            ):
                mask = img.maskdata
                mask_rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
                mask_rgba[mask, :] = [1, 0, 0, MASK_OPACITY]
                ax.imshow(
                    mask_rgba,
                    origin="upper",
                    extent=extent,
                    interpolation="nearest",
                )

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            if z_label:
                cbar.set_label(f"{z_label} ({z_unit})" if z_unit else z_label)

            # Set axis labels
            if x_label:
                ax.set_xlabel(f"{x_label} ({x_unit})" if x_unit else x_label)
            if y_label:
                ax.set_ylabel(f"{y_label} ({y_unit})" if y_unit else y_label)

            # Log scale and axis bounds (from first ImageObj)
            if is_image_obj and idx == 0:
                _apply_log_scale(ax, img)
                _apply_axis_bounds(ax, img)

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
        fig.savefig(buf, format="png", dpi=100, pil_kwargs={"compress_level": 1})
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MplMultiImageResult({len(self._objs)} images)"


# ============================================================================
# Main Plotter class (drop-in API)
# ============================================================================


class MatplotlibPlotter:
    """Matplotlib-based visualization frontend for the DataLab kernel.

    This class provides the same public API as
    :class:`datalab_kernel.plotly_backend.PlotlyPlotter` but produces static
    matplotlib PNGs instead of interactive Plotly figures.

    Example::

        from datalab_kernel.matplotlib_backend import MatplotlibPlotter

        plotter = MatplotlibPlotter(workspace)
        plotter.plot("s001")                    # Single signal
        plotter.plot([sig1, sig2])               # Multiple signals
        plotter.plot([img1, img2])               # Multiple images
        plotter.display_table(table_result)      # HTML table
        plotter.display_geometry(geom_result)    # HTML table
    """

    def __init__(self, workspace: Workspace) -> None:
        """Initialize plotter with workspace reference.

        Args:
            workspace: The workspace containing objects to plot
        """
        self._workspace = workspace

    def plot(
        self,
        obj_or_name: DataObject | str | list,
        title: str | None = None,
        show_roi: bool = True,
        show_results: bool = True,
        *,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xunit: str | None = None,
        yunit: str | None = None,
        zlabel: str | None = None,
        zunit: str | None = None,
        titles: list[str] | None = None,
        results: list | None = None,
        **kwargs,
    ) -> MplPlotResult | MplMultiSignalResult | MplMultiImageResult:
        """Plot one or more objects.

        Accepts a single object (or workspace name) **or** a list.

        * **Single object** — renders one signal or image.
        * **List of signals** — overlays all curves on shared axes.
        * **List of images** — displays in a subplot grid.
        * **Mixed list** — raises :class:`TypeError`.

        A single-item list is unwrapped and treated as a single object.

        Args:
            obj_or_name: Object to plot, workspace name, or a *list* of
             objects / names.
            title: Plot title (overall figure title for multi-plots).
            show_roi: Whether to show ROIs defined in the objects.
            show_results: Whether to show geometry/table results from metadata.
            xlabel: X-axis label override (multi-plots).
            ylabel: Y-axis label override (multi-plots).
            xunit: X-axis unit override (multi-plots).
            yunit: Y-axis unit override (multi-plots).
            zlabel: Colorbar label override (images only).
            zunit: Colorbar unit override (images only).
            titles: Per-image title list (images only).
            results: List of ``GeometryResult`` objects to overlay (images only).
            **kwargs: Additional plotting options (``height``, ``colormap``).

        Returns:
            A result object with Jupyter display capabilities.

        Raises:
            TypeError: If a list mixes signals and images.
            KeyError: If a workspace name is not found.
        """
        from datalab_kernel.plotter import (  # pylint: disable=import-outside-toplevel
            _IMAGE,
            _resolve_and_classify,
        )

        # --- list input: multi-object dispatch ---
        if isinstance(obj_or_name, list):
            # Single-item list → unwrap to single-object path
            if len(obj_or_name) == 1:
                item = obj_or_name[0]
                if isinstance(item, str):
                    item = self._workspace.get(item)
                return MplPlotResult(
                    item,
                    title=title,
                    show_roi=show_roi,
                    show_results=show_results,
                    results=results,
                    **kwargs,
                )

            objs, category = _resolve_and_classify(obj_or_name, self._workspace)
            if category == _IMAGE:
                return MplMultiImageResult(
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
            return MplMultiSignalResult(
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

        # --- scalar input: single-object path ---
        if isinstance(obj_or_name, str):
            obj = self._workspace.get(obj_or_name)
            if title is None:
                title = obj_or_name
        else:
            obj = obj_or_name
            if title is None and hasattr(obj, "title"):
                title = obj.title

        return MplPlotResult(
            obj,
            title=title,
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
