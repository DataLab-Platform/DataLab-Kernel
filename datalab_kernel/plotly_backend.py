# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Plotly Backend
==============

Plotly-based visualization backend for the DataLab kernel.

This module provides a drop-in alternative to the matplotlib-based
:mod:`datalab_kernel.plotter` module, producing interactive Plotly figures
instead of static PNGs. The public API mirrors that of
:class:`datalab_kernel.plotter.Plotter`.

Features include:

- Interactive signal and image visualization
- Multiple signals on a single plot with legend and color cycling
- Multiple images in a subplot grid layout
- ROI (Region of Interest) visualization
- Geometry result overlays (points, markers, rectangles, circles, ellipses,
  segments, polygons)
- Table/geometry result display (reused from plotter module)
- Mask visualization with semi-transparent overlay
- Axis labels with units, log scale, and axis bounds
- Curve styles (Lines, Sticks, Steps) and shade/fill
- Colormap support with inversion and LUT range
- Non-uniform image coordinates

Usage::

    from datalab_kernel.plotly_backend import PlotlyPlotter

    plotter = PlotlyPlotter(workspace)
    plotter.plot("s001")                    # Single signal
    plotter.plot([sig1, sig2])               # Multiple signals
    plotter.plot([img1, img2])               # Multiple images
    plotter.display_table(table_result)      # HTML table
    plotter.display_geometry(geom_result)    # HTML table
"""

from __future__ import annotations

import contextlib
import json
import math
import uuid
from typing import TYPE_CHECKING

import numpy as np

# Reuse display classes and helper functions from the matplotlib plotter
from datalab_kernel.plotter import (
    DEFAULT_PLOT_WIDTH,
    MASK_OPACITY,
    GeometryResultDisplay,
    TableResultDisplay,
    _build_results_html,
    _extract_geometry_results_from_metadata,
    _extract_table_results_from_metadata,
    _get_curve_style,
    _get_geometry_coord_labels,
    _get_image_colormap,
    _get_image_lut_range,
    _is_non_uniform_image,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

    from datalab_kernel.workspace import DataObject, Workspace


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Plotly default color sequence matching the matplotlib plotter's palette.
PLOTLY_COLORS = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
]

#: Dash styles cycling (matching matplotlib's LINESTYLES)
PLOTLY_DASHES = ["solid", "dash", "dashdot", "dot"]

#: Qt linestyle name → Plotly dash value mapping
_QT_TO_PLOTLY_DASH = {
    "SolidLine": "solid",
    "DashLine": "dash",
    "DashDotLine": "dashdot",
    "DashDotDotLine": "dot",
}

#: Matplotlib linestyle → Plotly dash value mapping
_MPL_TO_PLOTLY_DASH = {
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
}

#: ROI overlay colour
ROI_COLOR = "rgba(255, 0, 0, 0.55)"
ROI_FILL_COLOR = "rgba(255, 0, 0, 0.15)"

#: Geometry overlay colour
GEOMETRY_COLOR = "rgba(255, 255, 0, 0.85)"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_next_plotly_style(index: int) -> tuple[str, str]:
    """Return ``(color, dash)`` for the given sequential index.

    Args:
        index: Sequential index of the plot item

    Returns:
        Tuple of (hex color string, Plotly dash string)
    """
    color = PLOTLY_COLORS[index % len(PLOTLY_COLORS)]
    dash = PLOTLY_DASHES[(index // len(PLOTLY_COLORS)) % len(PLOTLY_DASHES)]
    return color, dash


def _format_axis_title(label: str | None, unit: str | None) -> str:
    """Build an axis title string, appending the unit in parentheses if present.

    Args:
        label: Axis label text (may be ``None`` or empty)
        unit: Axis unit text (may be ``None`` or empty)

    Returns:
        Formatted axis title string
    """
    label = label or ""
    if unit:
        return f"{label} ({unit})" if label else f"({unit})"
    return label


def _plotly_dash_from_metadata(meta_linestyle: str) -> str:
    """Convert a metadata linestyle value to a Plotly dash string.

    Accepts both Qt style names (``"SolidLine"``) and matplotlib shorthand
    (``"-"``).

    Args:
        meta_linestyle: Linestyle string from object metadata

    Returns:
        Plotly dash string (``"solid"``, ``"dash"``, ``"dashdot"``, ``"dot"``)
    """
    return _QT_TO_PLOTLY_DASH.get(
        meta_linestyle, _MPL_TO_PLOTLY_DASH.get(meta_linestyle, meta_linestyle)
    )


def _signal_line_params(obj) -> dict:
    """Extract Plotly ``line`` dict from signal metadata.

    Reads ``color``, ``linestyle``, ``linewidth`` from the object's metadata
    dict and converts them to Plotly-compatible values.

    Args:
        obj: SignalObj with potential style metadata

    Returns:
        Dict suitable for ``go.Scatter(line=...)``
    """
    meta = getattr(obj, "metadata", None) or {}
    line: dict = {}
    if "color" in meta:
        line["color"] = meta["color"]
    if "linestyle" in meta:
        line["dash"] = _plotly_dash_from_metadata(meta["linestyle"])
    if "linewidth" in meta:
        line["width"] = meta["linewidth"]
    return line


def _build_annotation_text(
    table_results: list, geometry_results: list | None = None
) -> str:
    """Build a multi-line text string from table/geometry results.

    Used for Plotly ``fig.add_annotation()`` overlays matching the matplotlib
    text-box behaviour.

    Args:
        table_results: List of TableResult objects
        geometry_results: Optional list of GeometryResult objects

    Returns:
        Multi-line plain-text string (may be empty)
    """
    lines: list[str] = []

    for table in table_results:
        lines.append(f"<b>{table.title}:</b>")
        headers = list(table.headers)
        for row in table.data:
            for header, value in zip(headers, row):
                if isinstance(value, float):
                    formatted = (
                        f"{value:.3g}"
                        if abs(value) < 0.001 or abs(value) >= 10000
                        else f"{value:.3f}"
                    )
                else:
                    formatted = str(value)
                lines.append(f"  {header}: {formatted}")
        lines.append("")

    if geometry_results:
        for geometry in geometry_results:
            lines.append(f"<b>{geometry.title}:</b>")
            coord_labels = _get_geometry_coord_labels(geometry)
            if len(geometry.coords) > 0:
                coords = (
                    geometry.coords[0] if geometry.coords.ndim > 1 else geometry.coords
                )
                for lbl, val in zip(coord_labels, coords):
                    if isinstance(val, float):
                        formatted = (
                            f"{val:.3g}"
                            if abs(val) < 0.001 or abs(val) >= 10000
                            else f"{val:.3f}"
                        )
                    else:
                        formatted = str(val)
                    lines.append(f"  {lbl}: {formatted}")
            lines.append("")

    # Strip trailing blank line
    while lines and lines[-1] == "":
        lines.pop()

    return "<br>".join(lines)


def _add_result_annotation(fig: go.Figure, text: str, row=None, col=None) -> None:
    """Add a text-box annotation to the upper-left corner of a Plotly figure.

    Args:
        fig: Plotly Figure object
        text: HTML-formatted annotation text
        row: Subplot row (1-indexed) for ``make_subplots`` figures, or None
        col: Subplot column (1-indexed) for ``make_subplots`` figures, or None
    """
    if not text:
        return

    # Determine the correct xref/yref for subplots
    if row is not None and col is not None:
        # For subplots, reference the specific subplot axis
        # plotly.subplots uses x, x2, x3, ... and y, y2, y3, ...
        subplot_idx = (
            (row - 1) * fig._grid_ref[0].__len__() + col
            if hasattr(fig, "_grid_ref")
            else 1
        )  # noqa: E501
        xref = f"x{subplot_idx} domain" if subplot_idx > 1 else "x domain"
        yref = f"y{subplot_idx} domain" if subplot_idx > 1 else "y domain"
    else:
        xref = "x domain"
        yref = "y domain"

    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref=xref,
        yref=yref,
        text=text,
        showarrow=False,
        font={"family": "monospace", "size": 11},
        align="left",
        bgcolor="rgba(255, 255, 255, 0.85)",
        bordercolor="gray",
        borderwidth=1,
        borderpad=6,
        xanchor="left",
        yanchor="top",
    )


def _add_signal_roi_shapes(fig: go.Figure, obj) -> None:
    """Add ROI vertical rectangles for signal objects.

    Args:
        fig: Plotly Figure object
        obj: SignalObj with ROI list
    """
    if not hasattr(obj, "roi") or not obj.roi:
        return
    for roi in obj.roi:
        roi_class = type(roi).__name__
        if roi_class == "SegmentROI" and obj is not None:
            x0, x1 = roi.get_physical_coords(obj)
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=ROI_FILL_COLOR,
                line={"color": ROI_COLOR, "width": 2},
                annotation_text="ROI",
                annotation_position="top left",
            )


def _add_image_roi_shapes(
    fig: go.Figure, obj, row: int | None = None, col: int | None = None
) -> None:
    """Add ROI shapes for image objects.

    Args:
        fig: Plotly Figure object
        obj: ImageObj with ROI list
        row: Subplot row (1-indexed) or None
        col: Subplot column (1-indexed) or None
    """
    if not hasattr(obj, "roi") or not obj.roi:
        return
    for roi in obj.roi:
        roi_class = type(roi).__name__
        shape_kwargs: dict = {}
        if row is not None and col is not None:
            shape_kwargs["row"] = row
            shape_kwargs["col"] = col

        if roi_class == "RectangularROI":
            x0, y0, dx, dy = roi.coords
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x0 + dx,
                y1=y0 + dy,
                line={"color": "red", "width": 2},
                **shape_kwargs,
            )
        elif roi_class == "CircularROI":
            xc, yc, r = roi.coords
            fig.add_shape(
                type="circle",
                x0=xc - r,
                y0=yc - r,
                x1=xc + r,
                y1=yc + r,
                line={"color": "red", "width": 2},
                **shape_kwargs,
            )
        elif roi_class == "PolygonalROI":
            points = roi.coords.reshape(-1, 2)
            # Close the polygon
            xs = list(points[:, 0]) + [points[0, 0]]
            ys = list(points[:, 1]) + [points[0, 1]]
            path = "M " + " L ".join(f"{x},{y}" for x, y in zip(xs, ys)) + " Z"
            fig.add_shape(
                type="path",
                path=path,
                line={"color": "red", "width": 2},
                **shape_kwargs,
            )


def _add_geometry_traces(
    fig: go.Figure,
    result,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add geometry result overlays to a Plotly figure.

    Iterates over all rows in ``result.coords`` to draw each geometric shape.
    Uses ``fig.add_shape()`` for rectangles, circles, ellipses, lines/segments
    and ``fig.add_trace()`` for point/marker overlays.

    A compact text label is placed near each shape showing the result title
    and key value (e.g., ``"FWHM: 0.52"`` near a segment midpoint).

    Args:
        fig: Plotly Figure object
        result: GeometryResult object with shape information
        row: Subplot row (1-indexed) or None
        col: Subplot column (1-indexed) or None
    """
    # Delayed import
    # pylint: disable=import-outside-toplevel
    from sigima.objects import KindShape

    trace_kwargs: dict = {}
    shape_kwargs: dict = {}
    if row is not None and col is not None:
        trace_kwargs["row"] = row
        trace_kwargs["col"] = col
        shape_kwargs["row"] = row
        shape_kwargs["col"] = col

    # Determine annotation axis references for subplots
    ann_kwargs: dict = {}
    if row is not None and col is not None:
        subplot_idx = (
            (row - 1) * fig._grid_ref[0].__len__() + col
            if hasattr(fig, "_grid_ref")
            else 1
        )
        ann_kwargs["xref"] = f"x{subplot_idx}" if subplot_idx > 1 else "x"
        ann_kwargs["yref"] = f"y{subplot_idx}" if subplot_idx > 1 else "y"

    label_title = getattr(result, "title", "")

    def _fmt(v: float) -> str:
        """Format a float value compactly."""
        if abs(v) < 0.001 or abs(v) >= 10000:
            return f"{v:.3g}"
        return f"{v:.3f}"

    def _add_label(x: float, y: float, text: str) -> None:
        """Add a small text label near a geometry shape."""
        fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font={"family": "sans-serif", "size": 10, "color": "#333"},
            bgcolor="rgba(255, 255, 200, 0.8)",
            bordercolor="rgba(200, 200, 0, 0.6)",
            borderwidth=1,
            borderpad=3,
            xanchor="left",
            yanchor="bottom",
            **ann_kwargs,
        )

    for coords in result.coords:
        if result.kind == KindShape.POINT:
            x0, y0 = coords
            fig.add_trace(
                _make_go().Scatter(
                    x=[x0],
                    y=[y0],
                    mode="markers",
                    marker={
                        "color": "yellow",
                        "size": 8,
                        "line": {"color": "black", "width": 1},
                    },
                    showlegend=False,
                ),
                **trace_kwargs,
            )
            _add_label(x0, y0, f"{label_title}: ({_fmt(x0)}, {_fmt(y0)})")
        elif result.kind == KindShape.MARKER:
            x0, y0 = coords
            # Crosshair lines
            fig.add_hline(
                y=y0,
                line={"color": "yellow", "width": 1, "dash": "dash"},
                opacity=0.7,
                **shape_kwargs,
            )
            fig.add_vline(
                x=x0,
                line={"color": "yellow", "width": 1, "dash": "dash"},
                opacity=0.7,
                **shape_kwargs,
            )
            fig.add_trace(
                _make_go().Scatter(
                    x=[x0],
                    y=[y0],
                    mode="markers",
                    marker={
                        "symbol": "cross",
                        "color": "yellow",
                        "size": 12,
                        "line": {"color": "yellow", "width": 2},
                    },
                    showlegend=False,
                ),
                **trace_kwargs,
            )
            _add_label(x0, y0, f"{label_title}: ({_fmt(x0)}, {_fmt(y0)})")
        elif result.kind == KindShape.RECTANGLE:
            x0, y0, dx, dy = coords
            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x0 + dx,
                y1=y0 + dy,
                line={"color": "yellow", "width": 2, "dash": "dash"},
                **shape_kwargs,
            )
        elif result.kind == KindShape.CIRCLE:
            xc, yc, r = coords
            fig.add_shape(
                type="circle",
                x0=xc - r,
                y0=yc - r,
                x1=xc + r,
                y1=yc + r,
                line={"color": "yellow", "width": 2, "dash": "dash"},
                **shape_kwargs,
            )
            _add_label(xc + r, yc, f"{label_title}: r={_fmt(r)}")
        elif result.kind == KindShape.SEGMENT:
            x0, y0, x1, y1 = coords
            fig.add_shape(
                type="line",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                line={"color": "yellow", "width": 2, "dash": "dash"},
                **shape_kwargs,
            )
            length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            _add_label(mx, my, f"{label_title}: {_fmt(length)}")
        elif result.kind == KindShape.ELLIPSE:
            xc, yc, a, b, theta = coords
            # Generate parametric ellipse points
            t = np.linspace(0, 2 * np.pi, 80)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            ex = xc + a * np.cos(t) * cos_t - b * np.sin(t) * sin_t
            ey = yc + a * np.cos(t) * sin_t + b * np.sin(t) * cos_t
            fig.add_trace(
                _make_go().Scatter(
                    x=np.append(ex, ex[0]),
                    y=np.append(ey, ey[0]),
                    mode="lines",
                    line={"color": "yellow", "width": 2, "dash": "dash"},
                    showlegend=False,
                ),
                **trace_kwargs,
            )
        elif result.kind == KindShape.POLYGON:
            xs = coords[::2]
            ys = coords[1::2]
            fig.add_trace(
                _make_go().Scatter(
                    x=np.append(xs, xs[0]),
                    y=np.append(ys, ys[0]),
                    mode="lines+markers",
                    line={"color": "yellow", "width": 2, "dash": "dash"},
                    marker={"color": "yellow", "size": 5},
                    showlegend=False,
                ),
                **trace_kwargs,
            )


def _make_go():
    """Lazy import of ``plotly.graph_objects``.

    Returns:
        The ``plotly.graph_objects`` module

    Raises:
        ImportError: If plotly is not installed
    """
    import plotly.graph_objects as go  # pylint: disable=import-outside-toplevel

    return go


def _make_subplots():
    """Lazy import of ``plotly.subplots.make_subplots``.

    Returns:
        The ``make_subplots`` function

    Raises:
        ImportError: If plotly is not installed
    """
    from plotly.subplots import make_subplots  # pylint: disable=import-outside-toplevel

    return make_subplots


def _get_image_coords(obj) -> tuple[np.ndarray, np.ndarray]:
    """Compute X and Y coordinate arrays for a uniform image.

    Uses ``obj.x0``, ``obj.y0``, ``obj.dx``, ``obj.dy`` and the image shape
    to produce arrays of pixel-centre coordinates.

    Args:
        obj: ImageObj with physical coordinate attributes

    Returns:
        Tuple ``(x_coords, y_coords)`` as 1-D NumPy arrays
    """
    nrows, ncols = obj.data.shape[:2]
    x0 = getattr(obj, "x0", 0.0)
    y0 = getattr(obj, "y0", 0.0)
    dx = getattr(obj, "dx", 1.0)
    dy = getattr(obj, "dy", 1.0)
    x_coords = x0 + np.arange(ncols) * dx
    y_coords = y0 + np.arange(nrows) * dy
    return x_coords, y_coords


#: Default signal figure width/height in pixels.
#: Matches matplotlib's default 6.4×4.8 in at 100 DPI (4:3 aspect ratio).
_SIGNAL_DEFAULT_WIDTH = DEFAULT_PLOT_WIDTH
_SIGNAL_DEFAULT_HEIGHT = int(DEFAULT_PLOT_WIDTH * 3 / 4)  # 480

#: Default plot width in pixels for single-image figures.
_IMAGE_BASE_WIDTH = DEFAULT_PLOT_WIDTH

#: Extra height in pixels for title, axis labels, and margins.
_IMAGE_HEIGHT_PADDING = 80

#: Minimum / maximum figure height (px) to keep the output readable.
_IMAGE_MIN_HEIGHT = 350
_IMAGE_MAX_HEIGHT = 750


def _compute_image_figure_dims(
    data: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    base_width: int = _IMAGE_BASE_WIDTH,
) -> tuple[int, int]:
    """Compute figure width and height for an image plot.

    Derives the height from the image's physical aspect ratio so that the
    heatmap fills the plotting area without excessive whitespace.

    Args:
        data: 2-D image array (used as fallback for shape)
        x_coords: 1-D array of X pixel-centre coordinates
        y_coords: 1-D array of Y pixel-centre coordinates
        base_width: Desired figure width in pixels

    Returns:
        ``(width, height)`` in pixels
    """
    x_range = float(np.ptp(x_coords)) if len(x_coords) > 1 else 1.0
    y_range = float(np.ptp(y_coords)) if len(y_coords) > 1 else 1.0
    if x_range <= 0:
        x_range = 1.0
    if y_range <= 0:
        y_range = 1.0

    aspect = y_range / x_range  # height / width in data space
    plot_height = int(base_width * aspect) + _IMAGE_HEIGHT_PADDING
    plot_height = max(_IMAGE_MIN_HEIGHT, min(_IMAGE_MAX_HEIGHT, plot_height))
    return base_width, plot_height


def _figure_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an embeddable HTML fragment.

    Uses the CDN for Plotly.js to keep the output lightweight.

    Args:
        fig: Plotly Figure object

    Returns:
        HTML string
    """
    return fig.to_html(include_plotlyjs="cdn", full_html=False)


def _figure_to_mimebundle(fig: go.Figure) -> dict:
    """Convert a Plotly figure to a Jupyter MIME bundle.

    Returns a dictionary with the ``application/vnd.plotly.v1+json`` MIME type
    so that JupyterLab, VS Code, and other modern notebook frontends render
    the interactive figure natively, without relying on CDN ``<script>`` tags
    (which are blocked by Content Security Policy in sandboxed outputs).

    A ``text/html`` fallback is included for classic Jupyter Notebook.

    The JSON dict is generated once via ``to_plotly_json()`` and reused to
    build the HTML fallback, avoiding a costly second serialisation pass
    over large numpy arrays.

    Args:
        fig: Plotly Figure object

    Returns:
        MIME bundle dictionary
    """
    # Single-pass: convert numpy arrays to Python types once
    json_dict = fig.to_plotly_json()

    # Build lightweight HTML from the already-converted dict instead of
    # calling fig.to_html() which would re-traverse all numpy arrays.
    div_id = uuid.uuid4().hex
    data_str = json.dumps(json_dict.get("data", []), allow_nan=True)
    layout_str = json.dumps(json_dict.get("layout", {}), allow_nan=True)
    config = json_dict.get("config", {})
    config.setdefault("responsive", True)
    config_str = json.dumps(config)

    html = (
        "<div>"
        '<script src="https://cdn.plot.ly/plotly-latest.min.js"'
        ' charset="utf-8"></script>'
        f'<div id="{div_id}" class="plotly-graph-div"'
        ' style="height:100%; width:100%;"></div>'
        '<script type="text/javascript">'
        f"Plotly.newPlot('{div_id}', {data_str}, {layout_str}, {config_str});"
        "</script></div>"
    )

    bundle: dict = {
        "application/vnd.plotly.v1+json": json_dict,
        "text/html": html,
    }
    return bundle


def _figure_to_png_bytes(fig: go.Figure) -> bytes | None:
    """Attempt to convert a Plotly figure to PNG bytes.

    Requires the ``kaleido`` package.  Returns ``None`` if kaleido is not
    available.

    Args:
        fig: Plotly Figure object

    Returns:
        PNG bytes or None
    """
    try:
        return fig.to_image(format="png", width=900, height=550, scale=2)
    except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
        return None


# ============================================================================
# Result classes
# ============================================================================


class PlotlyPlotResult:
    """Result of a single-object plot operation rendered via Plotly.

    Supports Jupyter's rich display protocol: ``_repr_html_()`` returns an
    interactive Plotly figure and ``_repr_png_()`` returns a static fallback.
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
        """Initialize Plotly plot result.

        Args:
            obj: Object to display (SignalObj or ImageObj)
            title: Plot title
            show_roi: Whether to show ROIs
            show_results: Whether to show geometry/table results from metadata
            results: Optional list of GeometryResult objects to overlay
            **kwargs: Additional plotting options (e.g., ``colormap``,
             ``height`` to override the auto-computed figure height in pixels)
        """
        self._obj = obj
        self._title = title
        self._show_roi = show_roi
        self._show_results = show_results
        self._results = results
        self._kwargs = kwargs
        self._results_html = ""

    # ---- Jupyter display protocol ----

    def _ipython_display_(self, **kwargs) -> None:
        """Display figure and results as separate outputs in Jupyter.

        Uses IPython's display API to emit the Plotly figure via its native
        JSON MIME type, then appends any analysis results as styled HTML
        tables below the figure.
        """
        from IPython.display import (  # pylint: disable=import-outside-toplevel
            HTML,
            display,
        )

        try:
            fig = self._build_figure()
            bundle = _figure_to_mimebundle(fig)
            display(bundle, raw=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            title = self._title or getattr(self._obj, "title", "Untitled")
            display(HTML(f"<div>Error rendering {title}: {exc}</div>"))
            return

        if self._results_html:
            display(HTML(self._results_html))

    def _repr_mimebundle_(self, **kwargs) -> dict:
        """Return MIME bundle for Jupyter display.

        Provides the ``application/vnd.plotly.v1+json`` MIME type so that
        JupyterLab, VS Code, and other modern notebook frontends can render
        the interactive figure natively without relying on CDN scripts.
        Falls back to HTML for classic Jupyter Notebook.
        """
        try:
            fig = self._build_figure()
            return _figure_to_mimebundle(fig)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            title = self._title or getattr(self._obj, "title", "Untitled")
            return {"text/html": f"<div>Error rendering {title}: {exc}</div>"}

    def _repr_html_(self) -> str:
        """Return interactive HTML representation for Jupyter display."""
        try:
            fig = self._build_figure()
            return _figure_to_html(fig)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            title = self._title or getattr(self._obj, "title", "Untitled")
            return f"<div>Error rendering {title}: {exc}</div>"

    def _repr_png_(self) -> bytes | None:
        """Return static PNG representation for Jupyter display."""
        try:
            fig = self._build_figure()
            return _figure_to_png_bytes(fig)
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            return None

    # ---- Figure builders ----

    def _build_figure(self) -> go.Figure:
        """Dispatch to the appropriate figure builder."""
        go = _make_go()  # noqa: F841 – side-effect import
        obj_type = type(self._obj).__name__
        if obj_type == "SignalObj":
            return self._build_signal_figure()
        if obj_type == "ImageObj":
            return self._build_image_figure()
        # Fallback: empty figure with message
        fig = _make_go().Figure()
        fig.add_annotation(text=f"Unsupported type: {obj_type}", showarrow=False)
        return fig

    def _build_signal_figure(self) -> go.Figure:
        """Build a Plotly figure for a single signal.

        Returns:
            Plotly Figure with signal trace and overlays
        """
        go = _make_go()
        obj = self._obj
        x = np.asarray(obj.x)
        y = np.asarray(obj.y)
        title = self._title or getattr(obj, "title", "Signal")

        # Line styling from metadata
        line_kw = {"color": PLOTLY_COLORS[0], "width": 1}
        meta_line = _signal_line_params(obj)
        line_kw.update(meta_line)

        curvestyle = _get_curve_style(obj)

        fig = go.Figure()

        # ---- Main trace ----
        if curvestyle == "Sticks":
            # Stem-like: vertical lines from baseline to each point
            baseline = 0.0
            if hasattr(obj, "get_metadata_option"):
                with contextlib.suppress(KeyError, AttributeError, ValueError):
                    baseline = float(obj.get_metadata_option("baseline"))
            # Build interleaved arrays: (x, baseline) -> (x, y) -> None gap
            xs: list = []
            ys: list = []
            for xi, yi in zip(x, y):
                xs.extend([xi, xi, None])
                ys.extend([baseline, yi, None])
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": line_kw.get("color", PLOTLY_COLORS[0]), "width": 1},
                    name=title,
                    showlegend=True,
                )
            )
        elif curvestyle == "Steps":
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=dict(**line_kw, shape="hv"),
                    name=title,
                )
            )
        else:
            # Default "Lines" with optional error bars
            error_x_kw = None
            error_y_kw = None
            if obj.dx is not None:
                error_x_kw = {
                    "type": "data",
                    "array": np.asarray(obj.dx),
                    "visible": True,
                }
            if obj.dy is not None:
                error_y_kw = {
                    "type": "data",
                    "array": np.asarray(obj.dy),
                    "visible": True,
                }
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    line=line_kw,
                    name=title,
                    error_x=error_x_kw,
                    error_y=error_y_kw,
                )
            )

        # ---- Shade / fill ----
        if hasattr(obj, "get_metadata_option"):
            with contextlib.suppress(KeyError, AttributeError, ValueError):
                shade = float(obj.get_metadata_option("shade"))
                if shade > 0:
                    fill_color = line_kw.get("color", PLOTLY_COLORS[0])
                    # Build rgba with alpha = shade
                    fig.data[-1].update(
                        fill="tozeroy",
                        fillcolor=_color_with_alpha(fill_color, shade),
                    )

        # ---- Layout ----
        xlabel_str = _format_axis_title(
            getattr(obj, "xlabel", None), getattr(obj, "xunit", None)
        )
        ylabel_str = _format_axis_title(
            getattr(obj, "ylabel", None), getattr(obj, "yunit", None)
        )

        # Allow user override via kwargs
        fig_w = self._kwargs.get("width", _SIGNAL_DEFAULT_WIDTH)
        fig_h = self._kwargs.get("height", _SIGNAL_DEFAULT_HEIGHT)

        layout_kw: dict = {
            "title": title,
            "xaxis_title": xlabel_str,
            "yaxis_title": ylabel_str,
            "template": "plotly_white",
            "showlegend": False,
            "width": fig_w,
            "height": fig_h,
        }

        # Log scale
        if getattr(obj, "xscalelog", False):
            layout_kw["xaxis_type"] = "log"
        if getattr(obj, "yscalelog", False):
            layout_kw["yaxis_type"] = "log"

        # Axis bounds
        if not getattr(obj, "autoscale", True):
            xmin = getattr(obj, "xscalemin", None)
            xmax = getattr(obj, "xscalemax", None)
            ymin = getattr(obj, "yscalemin", None)
            ymax = getattr(obj, "yscalemax", None)
            if xmin is not None and xmax is not None:
                layout_kw["xaxis_range"] = [xmin, xmax]
            if ymin is not None and ymax is not None:
                layout_kw["yaxis_range"] = [ymin, ymax]

        fig.update_layout(**layout_kw)

        # Grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

        # ---- ROI ----
        if self._show_roi:
            _add_signal_roi_shapes(fig, obj)

        # ---- Geometry & table results ----
        if self._show_results:
            geo_results = _extract_geometry_results_from_metadata(obj)
            for res in geo_results:
                _add_geometry_traces(fig, res)

            tbl_results = _extract_table_results_from_metadata(obj)
            self._results_html = _build_results_html(tbl_results, geo_results)

        return fig

    def _build_image_figure(self) -> go.Figure:
        """Build a Plotly figure for a single image.

        Returns:
            Plotly Figure with heatmap trace and overlays
        """
        go = _make_go()
        obj = self._obj
        data = obj.data
        if np.iscomplexobj(data):
            data = np.abs(data)

        title = self._title or getattr(obj, "title", "Image")

        # Colormap
        colormap = _get_image_colormap(obj, self._kwargs)
        # Check for inversion
        invert = False
        if hasattr(obj, "get_metadata_option"):
            with contextlib.suppress(KeyError, AttributeError, ValueError):
                invert = bool(obj.get_metadata_option("invert_colormap"))
        if invert:
            colormap = colormap + "_r"

        # LUT range
        vmin, vmax = _get_image_lut_range(obj)

        # Coordinates
        if _is_non_uniform_image(obj):
            x_coords = np.asarray(obj.xcoords)
            y_coords = np.asarray(obj.ycoords)
        else:
            x_coords, y_coords = _get_image_coords(obj)

        # Colorbar label
        zlabel = getattr(obj, "zlabel", None) or ""
        zunit = getattr(obj, "zunit", None)
        cbar_title = _format_axis_title(zlabel, zunit)

        fig = go.Figure()

        fig.add_trace(
            go.Heatmap(
                z=data,
                x=x_coords,
                y=y_coords,
                zmin=vmin,
                zmax=vmax,
                colorscale=colormap,
                colorbar={"title": cbar_title} if cbar_title else None,
            )
        )

        # Mask overlay
        if hasattr(obj, "maskdata") and obj.maskdata is not None:
            mask = obj.maskdata
            # Create a trace with NaN where mask is False, 1 where mask is True
            mask_z = np.where(mask, 1.0, np.nan)
            fig.add_trace(
                go.Heatmap(
                    z=mask_z,
                    x=x_coords,
                    y=y_coords,
                    colorscale=[
                        [0, f"rgba(255,0,0,{MASK_OPACITY})"],
                        [1, f"rgba(255,0,0,{MASK_OPACITY})"],
                    ],
                    showscale=False,
                    hoverinfo="skip",
                )
            )

        # Layout
        xlabel_str = _format_axis_title(
            getattr(obj, "xlabel", None), getattr(obj, "xunit", None)
        )
        ylabel_str = _format_axis_title(
            getattr(obj, "ylabel", None), getattr(obj, "yunit", None)
        )

        # Figure dimensions based on image aspect ratio
        fig_w, fig_h = _compute_image_figure_dims(data, x_coords, y_coords)
        # Allow user override via kwargs
        fig_h = self._kwargs.get("height", fig_h)

        layout_kw: dict = {
            "title": title,
            "xaxis_title": xlabel_str,
            "yaxis_title": ylabel_str,
            "template": "plotly_white",
            "showlegend": False,
            "yaxis_autorange": "reversed",  # Top-left origin like DataLab
            "yaxis_scaleanchor": "x",  # Equal aspect ratio
            "yaxis_constrain": "domain",
            "width": fig_w,
            "height": fig_h,
        }

        # Log scale
        if getattr(obj, "xscalelog", False):
            layout_kw["xaxis_type"] = "log"
        if getattr(obj, "yscalelog", False):
            layout_kw["yaxis_type"] = "log"

        # Axis bounds
        if not getattr(obj, "autoscale", True):
            xmin = getattr(obj, "xscalemin", None)
            xmax = getattr(obj, "xscalemax", None)
            ymin = getattr(obj, "yscalemin", None)
            ymax = getattr(obj, "yscalemax", None)
            if xmin is not None and xmax is not None:
                layout_kw["xaxis_range"] = [xmin, xmax]
            if ymin is not None and ymax is not None:
                # Reversed axis: swap min/max
                layout_kw["yaxis_range"] = [ymax, ymin]

        fig.update_layout(**layout_kw)

        # ROI
        if self._show_roi:
            _add_image_roi_shapes(fig, obj)

        # Geometry & table results
        if self._show_results:
            results_to_display: list = []
            if self._results is not None:
                rlist = (
                    self._results
                    if isinstance(self._results, (list, tuple))
                    else [self._results]
                )
                results_to_display.extend(rlist)

            geo_results = _extract_geometry_results_from_metadata(obj)
            results_to_display.extend(geo_results)

            for res in results_to_display:
                _add_geometry_traces(fig, res)

            tbl_results = _extract_table_results_from_metadata(obj)
            self._results_html = _build_results_html(tbl_results, results_to_display)

        return fig

    def __repr__(self) -> str:
        """Return string representation."""
        obj_type = type(self._obj).__name__
        title = self._title or getattr(self._obj, "title", "Untitled")
        return f"PlotlyPlotResult({obj_type}: {title})"


class PlotlyMultiSignalResult:
    """Result of a multi-signal plot operation rendered via Plotly.

    Supports plotting multiple ``SignalObj``, NumPy arrays, or ``(x, y)``
    tuples on a single interactive Plotly figure.
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
        """Initialize multi-signal Plotly result.

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
        self._results_html = ""

    # ---- Jupyter display protocol ----

    def _ipython_display_(self, **kwargs) -> None:
        """Display figure and results as separate outputs in Jupyter."""
        from IPython.display import (  # pylint: disable=import-outside-toplevel
            HTML,
            display,
        )

        try:
            fig = self._build_figure()
            bundle = _figure_to_mimebundle(fig)
            display(bundle, raw=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            display(HTML(f"<div>Error rendering signals: {exc}</div>"))
            return

        if self._results_html:
            display(HTML(self._results_html))

    def _repr_mimebundle_(self, **kwargs) -> dict:
        """Return MIME bundle for Jupyter display.

        Provides the ``application/vnd.plotly.v1+json`` MIME type so that
        JupyterLab, VS Code, and other modern notebook frontends can render
        the interactive figure natively without relying on CDN scripts.
        Falls back to HTML for classic Jupyter Notebook.
        """
        try:
            fig = self._build_figure()
            return _figure_to_mimebundle(fig)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {"text/html": f"<div>Error rendering signals: {exc}</div>"}

    def _repr_html_(self) -> str:
        """Return interactive HTML representation for Jupyter display."""
        try:
            fig = self._build_figure()
            return _figure_to_html(fig)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering signals: {exc}</div>"

    def _repr_png_(self) -> bytes | None:
        """Return static PNG representation for Jupyter display."""
        try:
            fig = self._build_figure()
            return _figure_to_png_bytes(fig)
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            return None

    # ---- Figure builder ----

    def _build_figure(self) -> go.Figure:
        """Build a Plotly figure with all signal traces.

        Returns:
            Plotly Figure with one trace per signal and overlays
        """
        go = _make_go()
        fig = go.Figure()

        x_label = self._xlabel
        y_label = self._ylabel
        x_unit = self._xunit
        y_unit = self._yunit

        all_geo_results: list = []
        all_tbl_results: list = []

        for idx, data_or_obj in enumerate(self._objs):
            color, dash = _get_next_plotly_style(idx)
            obj_type = type(data_or_obj).__name__

            if obj_type == "SignalObj":
                obj = data_or_obj
                x = np.asarray(obj.x)
                y = np.asarray(obj.y)
                label = obj.title or f"Signal {idx + 1}"

                # Labels/units from first SignalObj
                if idx == 0:
                    x_label = x_label or getattr(obj, "xlabel", None) or ""
                    y_label = y_label or getattr(obj, "ylabel", None) or ""
                    x_unit = x_unit or getattr(obj, "xunit", None) or ""
                    y_unit = y_unit or getattr(obj, "yunit", None) or ""

                # Line style
                line_kw: dict = {"color": color, "dash": dash, "width": 1}
                meta_line = _signal_line_params(obj)
                line_kw.update(meta_line)

                curvestyle = _get_curve_style(obj)

                if curvestyle == "Sticks":
                    baseline = 0.0
                    if hasattr(obj, "get_metadata_option"):
                        with contextlib.suppress(KeyError, AttributeError, ValueError):
                            baseline = float(obj.get_metadata_option("baseline"))
                    xs: list = []
                    ys: list = []
                    for xi, yi in zip(x, y):
                        xs.extend([xi, xi, None])
                        ys.extend([baseline, yi, None])
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line={"color": line_kw.get("color", color), "width": 1},
                            name=label,
                        )
                    )
                elif curvestyle == "Steps":
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line=dict(**line_kw, shape="hv"),
                            name=label,
                        )
                    )
                else:
                    error_x_kw = None
                    error_y_kw = None
                    if obj.dx is not None:
                        error_x_kw = {
                            "type": "data",
                            "array": np.asarray(obj.dx),
                            "visible": True,
                        }
                    if obj.dy is not None:
                        error_y_kw = {
                            "type": "data",
                            "array": np.asarray(obj.dy),
                            "visible": True,
                        }
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines",
                            line=line_kw,
                            name=label,
                            error_x=error_x_kw,
                            error_y=error_y_kw,
                        )
                    )

                # Shade / fill
                if hasattr(obj, "get_metadata_option"):
                    with contextlib.suppress(KeyError, AttributeError, ValueError):
                        shade = float(obj.get_metadata_option("shade"))
                        if shade > 0:
                            fill_c = line_kw.get("color", color)
                            fig.data[-1].update(
                                fill="tozeroy",
                                fillcolor=_color_with_alpha(fill_c, shade),
                            )

                # Log scale / axis bounds from first SignalObj
                if idx == 0:
                    layout_kw: dict = {}
                    if getattr(obj, "xscalelog", False):
                        layout_kw["xaxis_type"] = "log"
                    if getattr(obj, "yscalelog", False):
                        layout_kw["yaxis_type"] = "log"
                    if not getattr(obj, "autoscale", True):
                        xmin = getattr(obj, "xscalemin", None)
                        xmax = getattr(obj, "xscalemax", None)
                        ymin = getattr(obj, "yscalemin", None)
                        ymax = getattr(obj, "yscalemax", None)
                        if xmin is not None and xmax is not None:
                            layout_kw["xaxis_range"] = [xmin, xmax]
                        if ymin is not None and ymax is not None:
                            layout_kw["yaxis_range"] = [ymin, ymax]
                    if layout_kw:
                        fig.update_layout(**layout_kw)

                # ROIs
                if self._show_roi and hasattr(obj, "roi") and obj.roi:
                    for roi_idx, single_roi in enumerate(obj.roi):
                        x0, x1 = single_roi.get_physical_coords(obj)
                        fig.add_vrect(
                            x0=x0,
                            x1=x1,
                            fillcolor=_color_with_alpha(
                                line_kw.get("color", color), 0.15
                            ),
                            line={"color": line_kw.get("color", color), "width": 1},
                            annotation_text=(
                                f"{label} ROI {roi_idx + 1}" if roi_idx == 0 else None
                            ),
                            annotation_position="top left",
                        )

                # Geometry / table results
                if self._show_results:
                    geo_results = _extract_geometry_results_from_metadata(obj)
                    for res in geo_results:
                        _add_geometry_traces(fig, res)
                    all_geo_results.extend(geo_results)

                    tbl_results = _extract_table_results_from_metadata(obj)
                    all_tbl_results.extend(tbl_results)

            elif isinstance(data_or_obj, tuple) and len(data_or_obj) == 2:
                xdata, ydata = data_or_obj
                fig.add_trace(
                    go.Scatter(
                        x=np.asarray(xdata),
                        y=np.asarray(ydata),
                        mode="lines",
                        line={"color": color, "dash": dash},
                        name=f"Signal {idx + 1}",
                    )
                )

            elif isinstance(data_or_obj, np.ndarray):
                ydata = data_or_obj
                xdata = np.arange(len(ydata))
                fig.add_trace(
                    go.Scatter(
                        x=xdata,
                        y=ydata,
                        mode="lines",
                        line={"color": color, "dash": dash},
                        name=f"Signal {idx + 1}",
                    )
                )

            else:
                raise TypeError(f"Unsupported data type: {type(data_or_obj)}")

        # Build results HTML for display below the plot
        if all_tbl_results or all_geo_results:
            self._results_html = _build_results_html(all_tbl_results, all_geo_results)

        # Final layout
        xlabel_str = _format_axis_title(x_label, x_unit)
        ylabel_str = _format_axis_title(y_label, y_unit)

        # Allow user override via kwargs
        fig_w = self._kwargs.get("width", _SIGNAL_DEFAULT_WIDTH)
        fig_h = self._kwargs.get("height", _SIGNAL_DEFAULT_HEIGHT)

        fig.update_layout(
            title=self._title or "Signals",
            xaxis_title=xlabel_str,
            yaxis_title=ylabel_str,
            template="plotly_white",
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "left",
                "x": 0,
            },
            width=fig_w,
            height=fig_h,
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(0,0,0,0.1)")

        return fig

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PlotlyMultiSignalResult({len(self._objs)} signals)"


class PlotlyMultiImageResult:
    """Result of a multi-image plot operation rendered via Plotly.

    Uses ``plotly.subplots.make_subplots()`` to arrange multiple images in a
    grid layout with individual colorbars, ROI overlays, and result annotations.
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
        **kwargs,
    ) -> None:
        """Initialize multi-image Plotly result.

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
            rows: Fixed number of rows, or None to compute automatically
            **kwargs: Additional options (e.g., ``colormap``,
             ``height`` to override auto-computed per-subplot height in pixels)
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
        self._kwargs = kwargs
        self._results_html = ""

    # ---- Jupyter display protocol ----

    def _ipython_display_(self, **kwargs) -> None:
        """Display figure and results as separate outputs in Jupyter."""
        from IPython.display import (  # pylint: disable=import-outside-toplevel
            HTML,
            display,
        )

        try:
            fig = self._build_figure()
            bundle = _figure_to_mimebundle(fig)
            display(bundle, raw=True)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            display(HTML(f"<div>Error rendering images: {exc}</div>"))
            return

        if self._results_html:
            display(HTML(self._results_html))

    def _repr_mimebundle_(self, **kwargs) -> dict:
        """Return MIME bundle for Jupyter display.

        Provides the ``application/vnd.plotly.v1+json`` MIME type so that
        JupyterLab, VS Code, and other modern notebook frontends can render
        the interactive figure natively without relying on CDN scripts.
        Falls back to HTML for classic Jupyter Notebook.
        """
        try:
            fig = self._build_figure()
            return _figure_to_mimebundle(fig)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return {"text/html": f"<div>Error rendering images: {exc}</div>"}

    def _repr_html_(self) -> str:
        """Return interactive HTML representation for Jupyter display."""
        try:
            fig = self._build_figure()
            return _figure_to_html(fig)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return f"<div>Error rendering images: {exc}</div>"

    def _repr_png_(self) -> bytes | None:
        """Return static PNG representation for Jupyter display."""
        try:
            fig = self._build_figure()
            return _figure_to_png_bytes(fig)
        except Exception:  # pylint: disable=broad-exception-caught  # noqa: BLE001
            return None

    # ---- Figure builder ----

    def _build_figure(self) -> go.Figure:
        """Build a Plotly subplot figure with all image heatmaps.

        Returns:
            Plotly Figure with one heatmap per subplot
        """
        go = _make_go()
        make_subplots = _make_subplots()

        n_images = len(self._objs)

        # Grid layout
        if self._rows is not None:
            nrows = self._rows
            ncols = math.ceil(n_images / nrows)
        else:
            ncols = min(4, n_images)
            nrows = math.ceil(n_images / ncols)

        subplot_titles = self._titles or [None] * n_images
        # Fill missing titles
        subplot_titles = list(subplot_titles) + [None] * (
            n_images - len(subplot_titles)
        )

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[
                t or f"Image {i + 1}" for i, t in enumerate(subplot_titles)
            ],
        )

        # Results list alignment
        if self._results is None:
            results_list = [None] * n_images
        elif isinstance(self._results, (list, tuple)):
            if len(self._results) != n_images:
                results_list = list(self._results) * n_images
            else:
                results_list = list(self._results)
        else:
            results_list = [self._results] * n_images

        x_label = self._xlabel
        y_label = self._ylabel
        z_label = self._zlabel
        x_unit = self._xunit
        y_unit = self._yunit
        z_unit = self._zunit

        default_colormap = self._kwargs.get("colormap", None)

        all_tbl_results: list = []
        all_geo_results: list = []

        for idx, (img, result) in enumerate(zip(self._objs, results_list)):
            row = idx // ncols + 1
            col = idx % ncols + 1

            obj_type = type(img).__name__
            is_image_obj = obj_type == "ImageObj"

            # Extract data
            if is_image_obj:
                data = img.data
                # Labels from first ImageObj
                if idx == 0:
                    x_label = x_label or getattr(img, "xlabel", None) or ""
                    y_label = y_label or getattr(img, "ylabel", None) or ""
                    z_label = z_label or getattr(img, "zlabel", None) or ""
                    x_unit = x_unit or getattr(img, "xunit", None) or ""
                    y_unit = y_unit or getattr(img, "yunit", None) or ""
                    z_unit = z_unit or getattr(img, "zunit", None) or ""
            elif isinstance(img, np.ndarray):
                data = img
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")

            # Complex data
            if np.iscomplexobj(data):
                data = np.abs(data)

            # Colormap
            if is_image_obj:
                cm_kwargs = self._kwargs if default_colormap else {}
                colormap = _get_image_colormap(img, cm_kwargs)
                # Inversion
                invert = False
                if hasattr(img, "get_metadata_option"):
                    with contextlib.suppress(KeyError, AttributeError, ValueError):
                        invert = bool(img.get_metadata_option("invert_colormap"))
                if invert:
                    colormap = colormap + "_r"
            else:
                colormap = default_colormap or "viridis"

            # LUT range
            vmin, vmax = _get_image_lut_range(img) if is_image_obj else (None, None)

            # Coordinates
            if is_image_obj and _is_non_uniform_image(img):
                x_coords = np.asarray(img.xcoords)
                y_coords = np.asarray(img.ycoords)
            elif is_image_obj:
                x_coords, y_coords = _get_image_coords(img)
            else:
                nr, nc = data.shape[:2]
                x_coords = np.arange(nc, dtype=float)
                y_coords = np.arange(nr, dtype=float)

            # Colorbar title
            cbar_title = _format_axis_title(z_label, z_unit)

            fig.add_trace(
                go.Heatmap(
                    z=data,
                    x=x_coords,
                    y=y_coords,
                    zmin=vmin,
                    zmax=vmax,
                    colorscale=colormap,
                    colorbar={"title": cbar_title} if cbar_title else None,
                ),
                row=row,
                col=col,
            )

            # Mask overlay
            if is_image_obj and hasattr(img, "maskdata") and img.maskdata is not None:
                mask = img.maskdata
                mask_z = np.where(mask, 1.0, np.nan)
                fig.add_trace(
                    go.Heatmap(
                        z=mask_z,
                        x=x_coords,
                        y=y_coords,
                        colorscale=[
                            [0, f"rgba(255,0,0,{MASK_OPACITY})"],
                            [1, f"rgba(255,0,0,{MASK_OPACITY})"],
                        ],
                        showscale=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )

            # Reverse Y axis for this subplot + equal aspect ratio
            axis_suffix = "" if idx == 0 else str(idx + 1)
            fig.update_layout(
                **{
                    f"yaxis{axis_suffix}": {
                        "autorange": "reversed",
                        "scaleanchor": f"x{axis_suffix or ''}",
                        "constrain": "domain",
                    }
                }
            )

            # Axis labels
            xlabel_str = _format_axis_title(x_label, x_unit)
            ylabel_str = _format_axis_title(y_label, y_unit)
            fig.update_xaxes(title_text=xlabel_str, row=row, col=col)
            fig.update_yaxes(title_text=ylabel_str, row=row, col=col)

            # Log scale / bounds from first ImageObj
            if is_image_obj and idx == 0:
                if getattr(img, "xscalelog", False):
                    fig.update_xaxes(type="log", row=row, col=col)
                if getattr(img, "yscalelog", False):
                    fig.update_yaxes(type="log", row=row, col=col)
                if not getattr(img, "autoscale", True):
                    xmin = getattr(img, "xscalemin", None)
                    xmax = getattr(img, "xscalemax", None)
                    ymin = getattr(img, "yscalemin", None)
                    ymax = getattr(img, "yscalemax", None)
                    if xmin is not None and xmax is not None:
                        fig.update_xaxes(range=[xmin, xmax], row=row, col=col)
                    if ymin is not None and ymax is not None:
                        fig.update_yaxes(range=[ymax, ymin], row=row, col=col)

            # ROIs
            if self._show_roi and is_image_obj:
                _add_image_roi_shapes(fig, img, row=row, col=col)

            # Geometry / table results
            if self._show_results:
                results_to_display: list = []
                if result is not None:
                    rlist = result if isinstance(result, (list, tuple)) else [result]
                    results_to_display.extend(rlist)

                if is_image_obj:
                    geo_results = _extract_geometry_results_from_metadata(img)
                    results_to_display.extend(geo_results)

                for res in results_to_display:
                    _add_geometry_traces(fig, res, row=row, col=col)

                if is_image_obj:
                    tbl_results = _extract_table_results_from_metadata(img)
                    all_tbl_results.extend(tbl_results)
                    all_geo_results.extend(results_to_display)

        # Build HTML for results below the plot
        self._results_html = _build_results_html(all_tbl_results, all_geo_results)

        # Overall layout — derive height from first image's aspect ratio
        first_data = self._objs[0]
        if hasattr(first_data, "data"):
            _fd = first_data.data
        elif isinstance(first_data, np.ndarray):
            _fd = first_data
        else:
            _fd = np.empty((1, 1))
        if np.iscomplexobj(_fd):
            _fd = np.abs(_fd)
        if hasattr(first_data, "xcoords") and _is_non_uniform_image(first_data):
            _fx = np.asarray(first_data.xcoords)
            _fy = np.asarray(first_data.ycoords)
        elif hasattr(first_data, "data"):
            _fx, _fy = _get_image_coords(first_data)
        else:
            nr, nc = _fd.shape[:2]
            _fx = np.arange(nc, dtype=float)
            _fy = np.arange(nr, dtype=float)
        _cw, _ch = _compute_image_figure_dims(_fd, _fx, _fy, base_width=500)
        # Allow user override via kwargs
        _ch = self._kwargs.get("height", _ch)
        fig.update_layout(
            title=self._title or "Images",
            template="plotly_white",
            height=_ch * nrows,
            width=(_cw + 50) * ncols,
        )

        return fig

    def __repr__(self) -> str:
        """Return string representation."""
        return f"PlotlyMultiImageResult({len(self._objs)} images)"


# ============================================================================
# Color utilities
# ============================================================================


def _color_with_alpha(color: str, alpha: float) -> str:
    """Convert a colour string to an ``rgba(...)`` string with given alpha.

    Handles hex strings (``"#rrggbb"``), named CSS colours via a small
    built-in palette, and pass-through for strings already in ``rgba(...)``
    format.

    Args:
        color: CSS colour string
        alpha: Opacity value in ``[0, 1]``

    Returns:
        ``rgba(r, g, b, alpha)`` string
    """
    if color.startswith("rgba"):
        return color  # already rgba
    if color.startswith("rgb("):
        # rgb(r, g, b) → rgba(r, g, b, alpha)
        return color.replace("rgb(", "rgba(").replace(")", f", {alpha})")
    if color.startswith("#") and len(color) in (7, 9):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"

    # Named colour fallback – small lookup for the plotter palette
    _named = {
        "blue": (31, 119, 180),
        "red": (214, 39, 40),
        "green": (44, 160, 44),
        "orange": (255, 127, 14),
        "purple": (148, 103, 189),
        "brown": (140, 86, 75),
        "pink": (227, 119, 194),
        "gray": (127, 127, 127),
        "olive": (188, 189, 34),
        "yellow": (255, 255, 0),
    }
    rgb = _named.get(color.lower())
    if rgb:
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"

    # Unrecognised – return with opacity hint; Plotly may still handle it
    return color


# ============================================================================
# Main Plotter class (drop-in replacement API)
# ============================================================================


class PlotlyPlotter:
    """Plotly-based visualization frontend for the DataLab kernel.

    This class mirrors the public API of
    :class:`datalab_kernel.plotter.Plotter` but produces interactive Plotly
    figures instead of static matplotlib PNGs.

    Example::

        from datalab_kernel.plotly_backend import PlotlyPlotter

        plotter = PlotlyPlotter(workspace)

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
    ) -> PlotlyPlotResult | PlotlyMultiSignalResult | PlotlyMultiImageResult:
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
            **kwargs: Additional plotting options (``height``, ``colormap``,
             ``width``).

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
                return PlotlyPlotResult(
                    item,
                    title=title,
                    show_roi=show_roi,
                    show_results=show_results,
                    results=results,
                    **kwargs,
                )

            objs, category = _resolve_and_classify(obj_or_name, self._workspace)
            if category == _IMAGE:
                return PlotlyMultiImageResult(
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
            return PlotlyMultiSignalResult(
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

        return PlotlyPlotResult(
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

        This reuses the matplotlib plotter's :class:`TableResultDisplay` class,
        which produces framework-agnostic HTML tables.

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

        This reuses the matplotlib plotter's :class:`GeometryResultDisplay`
        class, which produces framework-agnostic HTML tables.

        Args:
            result: GeometryResult object to display
            title: Optional title override (uses result.title if None)

        Returns:
            GeometryResultDisplay with Jupyter display capabilities
        """
        return GeometryResultDisplay(result, title=title)
