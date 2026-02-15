# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Plotter API
===========

The Plotter class provides visualization capabilities for the DataLab kernel.
It supports inline notebook display and optional DataLab GUI synchronization.

This module serves as the public API facade and contains shared helpers used
by both the matplotlib and Plotly backends:

- :mod:`datalab_kernel.matplotlib_backend` — static PNG rendering
- :mod:`datalab_kernel.plotly_backend` — interactive HTML rendering

The :class:`Plotter` class auto-selects the best available backend:
Plotly (interactive) if installed, otherwise matplotlib (static).
Users can override the choice via :meth:`Plotter.set_backend` or the
``DATALAB_PLOTTER_BACKEND`` environment variable.

Shared helper functions for metadata extraction, coordinate computation,
and style resolution live in this module so both backends can reuse them.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from datalab_kernel.workspace import Workspace

logger = logging.getLogger("datalab-kernel")

# Valid backend names (lowercase)
BACKEND_MATPLOTLIB = "matplotlib"
BACKEND_PLOTLY = "plotly"
_VALID_BACKENDS = {BACKEND_MATPLOTLIB, BACKEND_PLOTLY}

# Environment variable for backend override
BACKEND_ENV_VAR = "DATALAB_PLOTTER_BACKEND"


def matplotlib_available() -> bool:
    """Check whether matplotlib is importable."""
    try:
        # pylint: disable=import-outside-toplevel,unused-import
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def plotly_available() -> bool:
    """Check whether plotly is importable."""
    try:
        # pylint: disable=import-outside-toplevel,unused-import
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


_NO_BACKEND_MSG = (
    "Neither plotly nor matplotlib is installed. "
    "Install at least one: pip install matplotlib  or  "
    "pip install datalab-kernel[plotly]"
)


def resolve_backend(requested: str | None = None) -> str:
    """Determine which plotting backend to use.

    Resolution order:

    1. *requested* argument (from :meth:`Plotter.set_backend` or constructor)
    2. ``DATALAB_PLOTTER_BACKEND`` environment variable
    3. Auto-detect: Plotly if available, else matplotlib

    When the chosen backend is not installed the function falls back to the
    other one and emits a :class:`UserWarning`.  If **neither** is installed
    an :class:`ImportError` is raised.

    Args:
        requested: ``"plotly"`` or ``"matplotlib"``, case-insensitive.
         *None* means auto-detect.

    Returns:
        Resolved backend name (``"plotly"`` or ``"matplotlib"``).

    Raises:
        ValueError: If *requested* is not a recognised backend name.
        ImportError: If neither matplotlib nor plotly is installed.
    """
    # --- Step 1: determine the candidate ----------------------------------
    if requested is not None:
        candidate = requested.strip().lower()
        if candidate not in _VALID_BACKENDS:
            raise ValueError(
                f"Unknown backend {requested!r}. Choose from {sorted(_VALID_BACKENDS)}."
            )
    else:
        env = os.environ.get(BACKEND_ENV_VAR)
        if env is not None:
            candidate = env.strip().lower()
            if candidate not in _VALID_BACKENDS:
                warnings.warn(
                    f"Ignoring invalid {BACKEND_ENV_VAR}={env!r}. "
                    f"Choose from {sorted(_VALID_BACKENDS)}.",
                    UserWarning,
                    stacklevel=2,
                )
                candidate = None
            else:
                logger.debug("Backend from %s: %s", BACKEND_ENV_VAR, candidate)
        else:
            candidate = None

    # --- Step 2: auto-detect if no explicit choice ------------------------
    if candidate is None:
        return _auto_detect_backend()

    # --- Step 3: validate availability, fallback with warning -------------
    avail = {
        BACKEND_PLOTLY: plotly_available,
        BACKEND_MATPLOTLIB: matplotlib_available,
    }
    if avail[candidate]():
        return candidate

    # Candidate unavailable — try the other
    other = BACKEND_PLOTLY if candidate == BACKEND_MATPLOTLIB else BACKEND_MATPLOTLIB
    if avail[other]():
        warnings.warn(
            f"Requested backend {candidate!r} is not installed. "
            f"Falling back to {other!r}.",
            UserWarning,
            stacklevel=2,
        )
        return other

    raise ImportError(_NO_BACKEND_MSG)


def _auto_detect_backend() -> str:
    """Return the best available backend (Plotly preferred)."""
    if plotly_available():
        return BACKEND_PLOTLY
    if matplotlib_available():
        return BACKEND_MATPLOTLIB
    raise ImportError(_NO_BACKEND_MSG)


def _create_delegate(backend: str, workspace: Workspace):
    """Instantiate the concrete plotter for *backend*.

    Args:
        backend: ``"plotly"`` or ``"matplotlib"``.
        workspace: The workspace containing objects to plot.

    Returns:
        A ``PlotlyPlotter`` or ``MatplotlibPlotter`` instance.
    """
    # pylint: disable=import-outside-toplevel,redefined-outer-name
    if backend == BACKEND_PLOTLY:
        from datalab_kernel.plotly_backend import PlotlyPlotter

        return PlotlyPlotter(workspace)

    from datalab_kernel.matplotlib_backend import MatplotlibPlotter

    return MatplotlibPlotter(workspace)


# Style configuration constants
MASK_OPACITY = 0.35  # Opacity for mask overlay

#: Default plot width in pixels for figure output.
#: Matches matplotlib's default figure width (6.4 in × 100 DPI) and fits
#: comfortably in standard Jupyter layouts (~960 px classic, ~700–900 px Lab).
DEFAULT_PLOT_WIDTH = 640

# Metadata prefix for geometry results (consistent with DataLab's GeometryAdapter)
GEOMETRY_META_PREFIX = "Geometry_"
# Metadata prefix for table results (consistent with DataLab's TableAdapter)
TABLE_META_PREFIX = "Table_"


def _build_results_html(
    table_results: list,
    geometry_results: list | None = None,
) -> str:
    """Build HTML for analysis results to display below a plot.

    Renders table and geometry results as styled HTML tables using the
    existing :class:`TableResultDisplay` and :class:`GeometryResultDisplay`
    classes.  The output is intended to be appended below the figure in
    notebook cell output (via ``_ipython_display_``).

    Args:
        table_results: List of TableResult objects
        geometry_results: Optional list of GeometryResult objects

    Returns:
        Concatenated HTML string (empty if no results)
    """
    if not table_results and not geometry_results:
        return ""

    parts: list[str] = []
    for tbl in table_results:
        parts.append(TableResultDisplay(tbl)._repr_html_())
    if geometry_results:
        for geo in geometry_results:
            parts.append(GeometryResultDisplay(geo)._repr_html_())
    return "\n".join(parts)


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


def _get_signal_style_from_metadata(obj) -> dict:
    """Extract line style parameters from signal object metadata.

    DataLab stores line style parameters as direct keys in the metadata dict
    (not prefixed with ``__``). Supported keys: ``color``, ``linestyle``,
    ``linewidth``.

    Args:
        obj: SignalObj with potential style metadata

    Returns:
        Dict of matplotlib-compatible style parameters
    """
    meta = getattr(obj, "metadata", None) or {}
    style: dict = {}
    if "color" in meta:
        style["color"] = meta["color"]
    if "linestyle" in meta:
        qt_to_mpl = {
            "SolidLine": "-",
            "DashLine": "--",
            "DashDotLine": "-.",
            "DashDotDotLine": ":",
        }
        style["linestyle"] = qt_to_mpl.get(meta["linestyle"], meta["linestyle"])
    if "linewidth" in meta:
        style["linewidth"] = meta["linewidth"]
    return style


def _get_curve_style(obj) -> str:
    """Get curve style from signal object metadata.

    Reads the ``__curvestyle`` metadata option. Possible values are
    ``"Lines"`` (default), ``"Sticks"``, or ``"Steps"``.

    Args:
        obj: SignalObj with potential curve style metadata

    Returns:
        Curve style string ("Lines", "Sticks", or "Steps")
    """
    if hasattr(obj, "get_metadata_option"):
        try:
            return obj.get_metadata_option("curvestyle")
        except (KeyError, AttributeError, ValueError):
            pass
    return "Lines"


def _apply_log_scale(ax: Axes, obj) -> None:
    """Apply logarithmic scale to axes if enabled on the object.

    Checks ``xscalelog`` and ``yscalelog`` attributes and sets the
    corresponding matplotlib axis scale to ``"log"``.

    Args:
        ax: Matplotlib axes object
        obj: SignalObj or ImageObj with potential log scale attributes
    """
    if getattr(obj, "xscalelog", False):
        ax.set_xscale("log")
    if getattr(obj, "yscalelog", False):
        ax.set_yscale("log")


def _apply_axis_bounds(ax: Axes, obj) -> None:
    """Apply explicit axis bounds when autoscale is disabled.

    If ``autoscale`` is ``False``, reads ``xscalemin``, ``xscalemax``,
    ``yscalemin``, ``yscalemax`` and applies them to the axes.

    Args:
        ax: Matplotlib axes object
        obj: SignalObj or ImageObj with axis bound attributes
    """
    if not getattr(obj, "autoscale", True):
        xmin = getattr(obj, "xscalemin", None)
        xmax = getattr(obj, "xscalemax", None)
        ymin = getattr(obj, "yscalemin", None)
        ymax = getattr(obj, "yscalemax", None)
        if xmin is not None and xmax is not None:
            ax.set_xlim(xmin, xmax)
        if ymin is not None and ymax is not None:
            ax.set_ylim(ymin, ymax)


def _is_non_uniform_image(obj) -> bool:
    """Check if an image object uses non-uniform coordinates.

    Args:
        obj: ImageObj to check

    Returns:
        True if the image has non-uniform coordinate arrays
    """
    if not hasattr(obj, "is_uniform_coords"):
        return False
    if obj.is_uniform_coords:
        return False
    xcoords = getattr(obj, "xcoords", None)
    ycoords = getattr(obj, "ycoords", None)
    return xcoords is not None and ycoords is not None


def _get_image_colormap(obj, kwargs: dict) -> str:
    """Determine colormap for an image from kwargs or object metadata.

    Priority: explicit kwarg > object metadata option > default ``"viridis"``.

    Args:
        obj: ImageObj with potential colormap metadata
        kwargs: Keyword arguments dict (may contain ``"colormap"`` key)

    Returns:
        Matplotlib colormap name string
    """
    colormap = kwargs.get("colormap")
    if colormap is None and hasattr(obj, "get_metadata_option"):
        with contextlib.suppress(KeyError, AttributeError, ValueError):
            colormap = obj.get_metadata_option("colormap")
    return colormap or "viridis"


def _get_image_lut_range(obj) -> tuple[float | None, float | None]:
    """Get LUT range (vmin, vmax) from image object attributes.

    Args:
        obj: ImageObj with potential ``zscalemin``/``zscalemax`` attributes

    Returns:
        Tuple of (vmin, vmax), either or both may be None
    """
    vmin = getattr(obj, "zscalemin", None)
    vmax = getattr(obj, "zscalemax", None)
    return vmin, vmax


# -- Object classification --------------------------------------------------

#: Category name for signal-like objects.
_SIGNAL = "signal"
#: Category name for image-like objects.
_IMAGE = "image"


def _classify_object(obj) -> str:
    """Classify an object as ``"signal"`` or ``"image"``.

    Classification rules:

    * ``SignalObj`` → ``"signal"``
    * ``ImageObj`` → ``"image"``
    * ``tuple`` of length 2 → ``"signal"`` (interpreted as *(x, y)*)
    * 1-D ``numpy.ndarray`` → ``"signal"`` (y-only data)
    * 2-D ``numpy.ndarray`` → ``"image"``

    Args:
        obj: The object to classify.

    Returns:
        ``"signal"`` or ``"image"``.

    Raises:
        TypeError: If *obj* cannot be classified.
    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    type_name = type(obj).__name__
    if type_name == "SignalObj":
        return _SIGNAL
    if type_name == "ImageObj":
        return _IMAGE
    if isinstance(obj, tuple) and len(obj) == 2:
        return _SIGNAL
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            return _SIGNAL
        if obj.ndim == 2:
            return _IMAGE
    raise TypeError(f"Cannot classify object of type {type(obj)!r} as signal or image.")


def _resolve_and_classify(objs_or_names, workspace) -> tuple[list, str]:
    """Resolve workspace names and classify a list of objects.

    All items must be of the same category (all signals or all images).
    Single-string inputs are **not** treated as lists — callers should handle
    the scalar-vs-list distinction before calling this helper.

    Args:
        objs_or_names: Iterable of objects and/or workspace name strings.
        workspace: The :class:`Workspace` used to resolve string names.

    Returns:
        A ``(resolved_objects, category)`` tuple where *category* is
        ``"signal"`` or ``"image"``.

    Raises:
        TypeError: If the list mixes signals and images.
    """
    resolved: list = []
    categories: set[str] = set()
    for item in objs_or_names:
        if isinstance(item, str):
            item = workspace.get(item)
        resolved.append(item)
        categories.add(_classify_object(item))

    if len(categories) > 1:
        raise TypeError(
            "Cannot mix signals and images in a single plot call. "
            "Pass only signals or only images."
        )
    # Default to signal for empty lists (will produce an empty figure)
    category = categories.pop() if categories else _SIGNAL
    return resolved, category


# ============================================================================
# Plotter facade (delegates to active backend)
# ============================================================================


class Plotter:
    """Visualization frontend for the DataLab kernel.

    The Plotter provides methods to display signals and images inline in
    Jupyter notebooks, and optionally synchronize views with a running
    DataLab instance.

    The backend is chosen automatically (Plotly if installed, otherwise
    matplotlib) but can be overridden via the *backend* parameter, the
    ``DATALAB_PLOTTER_BACKEND`` environment variable, or at runtime with
    :meth:`set_backend`.

    Example::

        # Single object
        plotter.plot("i042")
        plotter.plot(workspace.get("i042"))

        # Multiple signals (overlay on shared axes)
        plotter.plot([sig1, sig2, sig3])

        # Multiple images (subplot grid)
        plotter.plot([img1, img2])

        # Display analysis results
        plotter.display_table(result)
        plotter.display_geometry(result)

        # Switch backend at runtime
        plotter.set_backend("matplotlib")
    """

    def __init__(self, workspace: Workspace, backend: str | None = None) -> None:
        """Initialize plotter with workspace reference.

        Args:
            workspace: The workspace containing objects to plot
            backend: ``"plotly"`` or ``"matplotlib"``.  *None* (default)
             auto-detects the best available backend (Plotly preferred).
        """
        self._workspace = workspace
        resolved = resolve_backend(backend)
        self._delegate = _create_delegate(resolved, workspace)
        self._backend = resolved
        logger.debug("Plotter initialised with %s backend", resolved)

    # -- Backend selection API ------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the name of the active backend (``"plotly"`` or
        ``"matplotlib"``)."""
        return self._backend

    def set_backend(self, backend: str) -> Plotter:
        """Switch the plotting backend at runtime.

        Args:
            backend: ``"plotly"`` or ``"matplotlib"`` (case-insensitive).

        Returns:
            *self*, for call-chaining.

        Raises:
            ValueError: If *backend* is not recognised.
            ImportError: If the requested backend is not installed and no
             fallback is available.
        """
        resolved = resolve_backend(backend)
        if resolved != self._backend:
            self._delegate = _create_delegate(resolved, self._workspace)
            self._backend = resolved
            logger.debug("Plotter switched to %s backend", resolved)
        return self

    # -- Delegated plotting methods ------------------------------------------

    def plot(
        self,
        obj_or_name,
        title=None,
        show_roi=True,
        show_results=True,
        *,
        xlabel=None,
        ylabel=None,
        xunit=None,
        yunit=None,
        zlabel=None,
        zunit=None,
        titles=None,
        results=None,
        **kwargs,
    ):
        """Plot one or more objects.

        Accepts a single object (or workspace name) **or** a list of objects.

        * **Single object** — renders one signal or image.
        * **List of signals** — overlays all curves on shared axes.
        * **List of images** — displays in a subplot grid.
        * **Mixed list** — raises :class:`TypeError`.

        A single-item list is unwrapped and treated as a single object.

        Args:
            obj_or_name: Object to plot, workspace name, or a *list* of
             objects / names. Lists may contain ``SignalObj``, ``ImageObj``,
             ``numpy.ndarray`` (1-D → signal, 2-D → image), ``(x, y)``
             tuples (signal), or workspace name strings.
            title: Plot title (overall figure title for multi-plots).
            show_roi: Whether to show ROIs defined in the objects.
            show_results: Whether to show geometry/table results from
             metadata.
            xlabel: X-axis label override (multi-plots).
            ylabel: Y-axis label override (multi-plots).
            xunit: X-axis unit override (multi-plots).
            yunit: Y-axis unit override (multi-plots).
            zlabel: Colorbar label override (images only).
            zunit: Colorbar unit override (images only).
            titles: Per-image title list (images only).
            results: List of ``GeometryResult`` objects to overlay
             (images only).

        Keyword Args:
            height (int): Figure height in pixels. For images defaults to
             an auto-computed value based on the aspect ratio.
            colormap (str): Colormap name override (images only).

        Returns:
            A backend-specific result object with Jupyter display
            capabilities.

        Raises:
            TypeError: If a list mixes signals and images.
            KeyError: If a workspace name is not found.
        """
        return self._delegate.plot(
            obj_or_name,
            title=title,
            show_roi=show_roi,
            show_results=show_results,
            xlabel=xlabel,
            ylabel=ylabel,
            xunit=xunit,
            yunit=yunit,
            zlabel=zlabel,
            zunit=zunit,
            titles=titles,
            results=results,
            **kwargs,
        )

    def display_table(
        self, result, title=None, visible_only=True, transpose_single_row=True
    ):
        """Display a TableResult with rich HTML rendering.

        See the active backend's ``display_table`` method for full
        documentation.
        """
        return self._delegate.display_table(
            result,
            title=title,
            visible_only=visible_only,
            transpose_single_row=transpose_single_row,
        )

    def display_geometry(self, result, title=None):
        """Display a GeometryResult with rich HTML rendering.

        See the active backend's ``display_geometry`` method for full
        documentation.
        """
        return self._delegate.display_geometry(result, title=title)


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
            # (to_html() already includes a styled title header)
            if hasattr(self._result, "to_html"):
                table_html = self._result.to_html(
                    visible_only=self._visible_only,
                    transpose_single_row=self._transpose_single_row,
                )
                return f"""
                {self._TABLE_STYLE}
                <div>
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
            # (to_html() already includes a styled title header)
            if hasattr(self._result, "to_html"):
                table_html = self._result.to_html()
                return f"""
                {TableResultDisplay._TABLE_STYLE}
                <div>
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


# ============================================================================
# Backward-compatibility re-exports
# ============================================================================
# These aliases keep existing imports working after the matplotlib rendering
# code was extracted to :mod:`datalab_kernel.matplotlib_backend`.

# pylint: disable=wrong-import-position
try:
    from datalab_kernel.matplotlib_backend import (  # noqa: E402
        MatplotlibPlotter,
    )
    from datalab_kernel.matplotlib_backend import (  # noqa: E402
        MplPlotResult as PlotResult,
    )
except ImportError:  # pragma: no cover — matplotlib is normally required
    pass

__all__ = [
    # Public API
    "Plotter",
    "TableResultDisplay",
    "GeometryResultDisplay",
    # Backend selection
    "BACKEND_MATPLOTLIB",
    "BACKEND_PLOTLY",
    "BACKEND_ENV_VAR",
    "matplotlib_available",
    "plotly_available",
    "resolve_backend",
    # Backward-compat re-exports
    "PlotResult",
    "MatplotlibPlotter",
    # Classification helpers (for backend use)
    "_classify_object",
    "_resolve_and_classify",
    "_SIGNAL",
    "_IMAGE",
    # Shared helpers (for backend use)
    "DEFAULT_PLOT_WIDTH",
    "MASK_OPACITY",
    "GEOMETRY_META_PREFIX",
    "TABLE_META_PREFIX",
    "_build_results_html",
]
