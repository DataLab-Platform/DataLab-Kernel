# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Comprehensive tests for the Plotly backend
==========================================

Tests all rendering features of the Plotly backend including signals,
images, error bars, ROIs, geometry/table results, curve styles, colormaps,
LUT ranges, log scale, axis bounds, and non-uniform coordinates.

These tests require ``plotly`` to be installed. They are automatically
skipped if plotly is unavailable.

Pass ``--gui`` to open interactive HTML views in the browser.
"""

# pylint: disable=protected-access

from __future__ import annotations

import atexit

import numpy as np
import pytest
from sigima import create_image, create_signal

plotly = pytest.importorskip("plotly", reason="plotly is not installed")

pytestmark = [pytest.mark.standalone]

# pylint: disable=wrong-import-position
from datalab_kernel.backends import StandaloneBackend  # noqa: E402
from datalab_kernel.plotly_backend import (  # noqa: E402
    PlotlyMultiImageResult,
    PlotlyMultiSignalResult,
    PlotlyPlotResult,
    PlotlyPlotter,
)
from datalab_kernel.plotter import (  # noqa: E402
    GeometryResultDisplay,
    TableResultDisplay,
)
from datalab_kernel.tests.data import (  # noqa: E402
    make_geometry_result_circles,
    make_geometry_result_points,
    make_table_result_stats,
    make_test_image,
    make_test_image_complex,
    make_test_image_nonuniform,
    make_test_image_rich,
    make_test_image_with_colormap,
    make_test_image_with_lut,
    make_test_image_with_mask,
    make_test_image_with_roi,
    make_test_signal,
    make_test_signal_log,
    make_test_signal_rich,
    make_test_signal_steps,
    make_test_signal_sticks,
    make_test_signal_styled,
    make_test_signal_with_errorbars,
    make_test_signal_with_roi,
)
from datalab_kernel.workspace import Workspace  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Accumulated (test_name, html_fragment) pairs shown in the gallery at exit.
_GUI_RESULTS: list[tuple[str, str]] = []


def _show_if_gui(gui_mode: bool, result) -> None:
    """If ``--gui`` is active, accumulate the plot for the end-of-session gallery.

    All accumulated plots are displayed together in a single tabbed HTML page
    (see :func:`_show_gallery`) when the test process exits.
    """
    if not gui_mode:
        return
    # pylint: disable=import-outside-toplevel
    import inspect

    frame = inspect.currentframe().f_back
    test_name = frame.f_code.co_qualname

    html_fragment = result._repr_html_()
    _GUI_RESULTS.append((test_name, html_fragment))


def _show_gallery() -> None:
    """Open all accumulated Plotly figures in a single tabbed HTML page.

    Generates a dark-themed HTML page with a sidebar listing test names
    and a content area showing the corresponding Plotly figures.
    Clicking a test name in the sidebar reveals the associated plot.
    """
    if not _GUI_RESULTS:
        return
    # pylint: disable=import-outside-toplevel
    import tempfile
    import webbrowser
    from html import escape

    nav_items = []
    content_items = []
    for idx, (name, html_frag) in enumerate(_GUI_RESULTS):
        active_cls = " active" if idx == 0 else ""
        display = "block" if idx == 0 else "none"
        nav_items.append(
            f'<li class="nav-item{active_cls}" onclick="showTab({idx})">'
            f"{escape(name)}</li>"
        )
        content_items.append(
            f'<div class="tab-content" id="tab-{idx}" '
            f'style="display:{display}">'
            f"<h3>{escape(name)}</h3>{html_frag}</div>"
        )

    html_page = (
        "<!DOCTYPE html>\n"
        "<html><head><meta charset='utf-8'>\n"
        f"<title>Plotly Backend Tests \u2014 {len(_GUI_RESULTS)} plots</title>\n"
        "<style>\n"
        "  * { box-sizing:border-box; }\n"
        "  body { margin:0; font-family:'Segoe UI',sans-serif; display:flex;\n"
        "         height:100vh; background:#1e1e1e; color:#d4d4d4; }\n"
        "  .sidebar { width:340px; min-width:340px; overflow-y:auto;\n"
        "             background:#252526; border-right:1px solid #3c3c3c; }\n"
        "  .sidebar h2 { padding:14px 16px; margin:0; font-size:14px;\n"
        "                color:#ccc; border-bottom:1px solid #3c3c3c; }\n"
        "  ul { list-style:none; padding:0; margin:0; }\n"
        "  .nav-item { padding:8px 16px; cursor:pointer; font-size:12px;\n"
        "              font-family:Consolas,monospace;\n"
        "              border-bottom:1px solid #2d2d2d; }\n"
        "  .nav-item:hover { background:#2a2d2e; }\n"
        "  .nav-item.active { background:#264f78; color:white; }\n"
        "  .content { flex:1; overflow-y:auto; padding:16px; }\n"
        "  .tab-content h3 { margin:0 0 8px; font-size:13px;\n"
        "                    color:#9cdcfe; font-family:Consolas,monospace; }\n"
        "</style>\n"
        "<script>\n"
        "function showTab(idx) {\n"
        "  document.querySelectorAll('.tab-content')\n"
        "    .forEach(el => el.style.display='none');\n"
        "  document.querySelectorAll('.nav-item')\n"
        "    .forEach(el => el.classList.remove('active'));\n"
        "  document.getElementById('tab-'+idx).style.display='block';\n"
        "  document.querySelectorAll('.nav-item')[idx].classList.add('active');\n"
        "  window.dispatchEvent(new Event('resize'));\n"  # Plotly relayout
        "}\n"
        "</script>\n"
        "</head><body>\n"
        "<div class='sidebar'>\n"
        f"  <h2>Tests ({len(_GUI_RESULTS)})</h2>\n"
        f"  <ul>{''.join(nav_items)}</ul>\n"
        "</div>\n"
        f"<div class='content'>{''.join(content_items)}</div>\n"
        "</body></html>"
    )

    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html_page)
        webbrowser.open(f"file://{f.name}")


atexit.register(_show_gallery)


def _make_workspace() -> Workspace:
    return Workspace(backend=StandaloneBackend())


def _assert_valid_html(html: str) -> None:
    assert isinstance(html, str)
    assert len(html) > 50, "HTML suspiciously short"
    assert "<" in html


# ============================================================================
# PlotlyPlotter facade
# ============================================================================


class TestPlotlyPlotter:
    """Verify PlotlyPlotter resolves objects and returns result types."""

    def test_plotter_creation(self):
        """PlotlyPlotter instantiates."""
        ws = _make_workspace()
        plotter = PlotlyPlotter(ws)
        assert plotter is not None

    def test_plot_signal(self):
        """plot(signal) returns PlotlyPlotResult."""
        ws = _make_workspace()
        sig = make_test_signal("s")
        ws.add("s", sig)
        plotter = PlotlyPlotter(ws)
        result = plotter.plot(sig)
        assert isinstance(result, PlotlyPlotResult)

    def test_plot_by_name(self):
        """plot('name') resolves from workspace."""
        ws = _make_workspace()
        ws.add("s", make_test_signal("s"))
        plotter = PlotlyPlotter(ws)
        result = plotter.plot("s")
        assert isinstance(result, PlotlyPlotResult)

    def test_plot_missing_raises(self):
        """plot('unknown') raises KeyError."""
        plotter = PlotlyPlotter(_make_workspace())
        with pytest.raises(KeyError, match="not found"):
            plotter.plot("unknown")

    def test_plot_signals(self):
        """plot_signals returns PlotlyMultiSignalResult."""
        ws = _make_workspace()
        sigs = [make_test_signal(f"s{i}") for i in range(3)]
        plotter = PlotlyPlotter(ws)
        result = plotter.plot_signals(sigs)
        assert isinstance(result, PlotlyMultiSignalResult)

    def test_plot_images(self):
        """plot_images returns PlotlyMultiImageResult."""
        ws = _make_workspace()
        imgs = [make_test_image(f"i{i}", (64, 64)) for i in range(2)]
        plotter = PlotlyPlotter(ws)
        result = plotter.plot_images(imgs)
        assert isinstance(result, PlotlyMultiImageResult)

    def test_display_table(self):
        """display_table returns TableResultDisplay."""
        plotter = PlotlyPlotter(_make_workspace())
        table = make_table_result_stats()
        result = plotter.display_table(table)
        assert isinstance(result, TableResultDisplay)

    def test_display_geometry(self):
        """display_geometry returns GeometryResultDisplay."""
        plotter = PlotlyPlotter(_make_workspace())
        geom = make_geometry_result_points()
        result = plotter.display_geometry(geom)
        assert isinstance(result, GeometryResultDisplay)


# ============================================================================
# Signal rendering
# ============================================================================


class TestPlotlySignalRendering:
    """Test Plotly signal rendering features."""

    def test_basic_signal(self, gui_mode):
        """Basic sine wave renders to valid HTML."""
        sig = make_test_signal("basic_sine")
        result = PlotlyPlotResult(sig, title="Basic Signal")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_with_errorbars(self, gui_mode):
        """Signal with dx/dy error bars renders without error."""
        sig = make_test_signal_with_errorbars()
        result = PlotlyPlotResult(sig, title="Error Bars")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_with_roi(self, gui_mode):
        """Signal with ROI segments renders with shaded regions."""
        sig = make_test_signal_with_roi()
        result = PlotlyPlotResult(sig, title="Signal ROI", show_roi=True)
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_sticks(self, gui_mode):
        """Signal with curvestyle='Sticks'."""
        sig = make_test_signal_sticks()
        result = PlotlyPlotResult(sig, title="Stick Style")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_steps(self, gui_mode):
        """Signal with curvestyle='Steps'."""
        sig = make_test_signal_steps()
        result = PlotlyPlotResult(sig, title="Step Style")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_log_scale(self, gui_mode):
        """Signal with log scale on both axes."""
        sig = make_test_signal_log()
        result = PlotlyPlotResult(sig, title="Log Scale")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_styled(self, gui_mode):
        """Signal with custom color/linestyle/linewidth metadata."""
        sig = make_test_signal_styled()
        result = PlotlyPlotResult(sig, title="Styled Signal")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_rich(self, gui_mode):
        """Rich signal with all features renders without error."""
        sig = make_test_signal_rich()
        result = PlotlyPlotResult(sig, title="Rich Signal")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_signal_repr(self):
        """String representation includes type info."""
        sig = make_test_signal("repr")
        result = PlotlyPlotResult(sig, title="repr")
        assert "PlotlyPlotResult" in repr(result)
        assert "SignalObj" in repr(result)


# ============================================================================
# Image rendering
# ============================================================================


class TestPlotlyImageRendering:
    """Test Plotly image rendering features."""

    def test_basic_image(self, gui_mode):
        """Basic random image renders to valid HTML."""
        img = make_test_image("basic", (64, 64))
        result = PlotlyPlotResult(img, title="Basic Image")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_with_roi(self, gui_mode):
        """Image with rectangular + circular ROIs."""
        img = make_test_image_with_roi()
        result = PlotlyPlotResult(img, title="Image ROI", show_roi=True)
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_with_mask(self, gui_mode):
        """Image with mask overlay."""
        img = make_test_image_with_mask()
        result = PlotlyPlotResult(img, title="Masked Image")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_with_colormap(self, gui_mode):
        """Image with custom colormap metadata."""
        img = make_test_image_with_colormap(colormap="hot")
        result = PlotlyPlotResult(img, title="Hot Colormap")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_with_lut(self, gui_mode):
        """Image with custom LUT range."""
        img = make_test_image_with_lut(vmin=0.3, vmax=0.7)
        result = PlotlyPlotResult(img, title="LUT Range")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_nonuniform(self, gui_mode):
        """Image with non-uniform coordinates."""
        img = make_test_image_nonuniform()
        result = PlotlyPlotResult(img, title="Non-Uniform")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_complex(self, gui_mode):
        """Complex image renders absolute values."""
        img = make_test_image_complex()
        result = PlotlyPlotResult(img, title="Complex Image")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_image_rich(self, gui_mode):
        """Rich image with all features renders without error."""
        img = make_test_image_rich()
        result = PlotlyPlotResult(img, title="Rich Image")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)


# ============================================================================
# Multi-signal rendering
# ============================================================================


class TestPlotlyMultiSignal:
    """Test Plotly multi-signal rendering."""

    def test_basic_multi_signal(self, gui_mode):
        """Multiple signals on one plot."""
        sigs = [make_test_signal(f"sig_{i}", freq=0.5 * (i + 1)) for i in range(3)]
        result = PlotlyMultiSignalResult(sigs, title="Multi Signal")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_with_roi(self, gui_mode):
        """Multi-signal with ROIs."""
        sigs = [make_test_signal_with_roi(f"roi_{i}") for i in range(2)]
        result = PlotlyMultiSignalResult(sigs, title="Multi ROI", show_roi=True)
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_numpy_arrays(self, gui_mode):
        """Multi-signal with plain numpy arrays."""
        y1 = np.sin(np.linspace(0, 10, 100))
        y2 = np.cos(np.linspace(0, 10, 100))
        result = PlotlyMultiSignalResult([y1, y2], title="NumPy Arrays")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_tuples(self, gui_mode):
        """Multi-signal with (x, y) tuples."""
        x = np.linspace(0, 10, 100)
        result = PlotlyMultiSignalResult(
            [(x, np.sin(x)), (x, np.cos(x))], title="Tuples"
        )
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_repr(self):
        """String representation shows count."""
        sigs = [make_test_signal(f"s{i}") for i in range(3)]
        result = PlotlyMultiSignalResult(sigs)
        assert "3" in repr(result)


# ============================================================================
# Multi-image rendering
# ============================================================================


class TestPlotlyMultiImage:
    """Test Plotly multi-image grid rendering."""

    def test_basic_multi_image(self, gui_mode):
        """Multiple images in a grid."""
        imgs = [make_test_image(f"img_{i}", (64, 64)) for i in range(3)]
        result = PlotlyMultiImageResult(imgs, title="Multi Image")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_with_roi(self, gui_mode):
        """Multi-image with ROIs."""
        imgs = [make_test_image_with_roi(f"roi_{i}") for i in range(2)]
        result = PlotlyMultiImageResult(imgs, title="Multi ROI", show_roi=True)
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_numpy_arrays(self, gui_mode):
        """Multi-image with plain numpy arrays."""
        arrays = [np.random.rand(64, 64).astype(np.float32) for _ in range(4)]
        result = PlotlyMultiImageResult(arrays, title="NumPy Arrays")
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_with_geometry(self, gui_mode):
        """Multi-image with explicit geometry results."""
        imgs = [make_test_image("img", (128, 128), "gaussian")]
        circles = make_geometry_result_circles()
        result = PlotlyMultiImageResult(imgs, title="With Geometry", results=[circles])
        _assert_valid_html(result._repr_html_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_repr(self):
        """String representation shows count."""
        imgs = [make_test_image(f"i{i}", (64, 64)) for i in range(4)]
        result = PlotlyMultiImageResult(imgs)
        assert "4" in repr(result)


# ============================================================================
# Edge cases
# ============================================================================


class TestPlotlyEdgeCases:
    """Test edge cases and error handling."""

    def test_show_results_false(self):
        """show_results=False skips geometry/table extraction."""
        sig = make_test_signal_rich()
        result = PlotlyPlotResult(sig, show_results=False)
        _assert_valid_html(result._repr_html_())

    def test_show_roi_false(self):
        """show_roi=False skips ROI overlay."""
        sig = make_test_signal_with_roi()
        result = PlotlyPlotResult(sig, show_roi=False)
        _assert_valid_html(result._repr_html_())

    def test_signal_no_metadata(self):
        """Signal with no metadata options."""
        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        sig = create_signal(title="bare", x=x, y=y)
        result = PlotlyPlotResult(sig)
        _assert_valid_html(result._repr_html_())

    def test_image_no_metadata(self):
        """Image with no metadata options."""
        data = np.random.rand(64, 64).astype(np.float32)
        img = create_image(title="bare", data=data)
        result = PlotlyPlotResult(img)
        _assert_valid_html(result._repr_html_())

    def test_small_image(self):
        """Very small image (4x4) renders."""
        img = make_test_image("tiny", (4, 4))
        result = PlotlyPlotResult(img)
        _assert_valid_html(result._repr_html_())
