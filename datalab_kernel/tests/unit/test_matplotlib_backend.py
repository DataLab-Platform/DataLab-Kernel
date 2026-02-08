# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Comprehensive tests for the matplotlib backend
===============================================

Tests all rendering features of the matplotlib backend including signals,
images, error bars, ROIs, geometry/table results, curve styles, colormaps,
LUT ranges, log scale, axis bounds, and non-uniform coordinates.

These tests run headlessly by default (Agg backend). Pass ``--gui`` to
open interactive matplotlib windows for visual inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from datalab_kernel.backends import StandaloneBackend

pytestmark = [pytest.mark.standalone]
from datalab_kernel.matplotlib_backend import (
    MatplotlibPlotter,
    MplMultiImageResult,
    MplMultiSignalResult,
    MplPlotResult,
)
from datalab_kernel.plotter import (
    GeometryResultDisplay,
    MultiImagePlotResult,
    MultiSignalPlotResult,
    PlotResult,
    Plotter,
    TableResultDisplay,
)
from datalab_kernel.tests.data import (
    make_geometry_result_circles,
    make_geometry_result_points,
    make_geometry_result_segments,
    make_table_result_multi_row,
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
from datalab_kernel.workspace import Workspace

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#: Accumulated (test_name, png_bytes) pairs shown in the gallery at exit.
_GUI_RESULTS: list[tuple[str, bytes]] = []


def _show_if_gui(gui_mode: bool, result) -> None:
    """If ``--gui`` is active, accumulate the plot for the end-of-session gallery.

    All accumulated plots are displayed together in a tkinter gallery window
    (see :func:`_show_gallery`) when the test process exits.
    """
    if not gui_mode:
        return
    # pylint: disable=import-outside-toplevel
    import inspect

    frame = inspect.currentframe().f_back
    test_name = frame.f_code.co_qualname

    png = result._repr_png_()
    _GUI_RESULTS.append((test_name, png))


def _show_gallery() -> None:
    """Show all accumulated plots in a tkinter gallery with a test selector.

    Left panel: scrollable listbox of test names.
    Right panel: the selected plot image, auto-scaled to fit.
    Navigation: click list items, Up/Down arrows, Escape to close.
    """
    if not _GUI_RESULTS:
        return
    # pylint: disable=import-outside-toplevel
    import tkinter as tk
    from io import BytesIO

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title(f"Matplotlib Backend Tests \u2014 {len(_GUI_RESULTS)} plots")
    root.geometry("1280x780")

    # ---- Left panel: test list ----
    left = tk.Frame(root, width=340, bg="#252526")
    left.pack(side=tk.LEFT, fill=tk.Y)
    left.pack_propagate(False)

    header = tk.Label(
        left,
        text=f"  Tests ({len(_GUI_RESULTS)})",
        font=("Segoe UI", 11, "bold"),
        bg="#252526",
        fg="#cccccc",
        anchor="w",
        pady=8,
    )
    header.pack(fill=tk.X)

    list_frame = tk.Frame(left)
    list_frame.pack(fill=tk.BOTH, expand=True)

    scrollbar = tk.Scrollbar(list_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    listbox = tk.Listbox(
        list_frame,
        font=("Consolas", 9),
        selectmode=tk.SINGLE,
        bg="#1e1e1e",
        fg="#d4d4d4",
        selectbackground="#264f78",
        selectforeground="white",
        borderwidth=0,
        highlightthickness=0,
        yscrollcommand=scrollbar.set,
    )
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    # ---- Right panel: image display ----
    right = tk.Frame(root, bg="#333333")
    right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    title_var = tk.StringVar(value="")
    title_label = tk.Label(
        right,
        textvariable=title_var,
        font=("Segoe UI", 11),
        bg="#333333",
        fg="white",
        anchor="w",
        padx=12,
        pady=8,
    )
    title_label.pack(fill=tk.X)

    img_label = tk.Label(right, bg="#333333")
    img_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    # Pre-load PIL images
    pil_images: list[Image.Image] = []
    for name, png_bytes in _GUI_RESULTS:
        listbox.insert(tk.END, name)
        pil_images.append(Image.open(BytesIO(png_bytes)))

    _tk_ref: list[ImageTk.PhotoImage | None] = [None]  # prevent GC

    def _display(idx: int) -> None:
        pil_img = pil_images[idx].copy()
        avail_w = max(right.winfo_width() - 30, 400)
        avail_h = max(right.winfo_height() - 60, 400)
        pil_img.thumbnail((avail_w, avail_h), Image.LANCZOS)
        _tk_ref[0] = ImageTk.PhotoImage(pil_img)
        img_label.config(image=_tk_ref[0])
        title_var.set(_GUI_RESULTS[idx][0])

    def _on_select(_event=None) -> None:
        sel = listbox.curselection()
        if sel:
            _display(sel[0])

    listbox.bind("<<ListboxSelect>>", _on_select)
    root.bind("<Escape>", lambda _: root.destroy())

    # Select first entry after layout is computed
    listbox.select_set(0)
    root.after(100, lambda: _display(0))

    root.mainloop()


import atexit  # noqa: E402

atexit.register(_show_gallery)


def _make_workspace() -> Workspace:
    return Workspace(backend=StandaloneBackend())


def _assert_valid_png(data: bytes) -> None:
    assert isinstance(data, bytes)
    assert len(data) > 100, "PNG data suspiciously small"
    assert data[:8] == PNG_MAGIC


def _assert_valid_html(html: str) -> None:
    assert isinstance(html, str)
    assert "<" in html


# ============================================================================
# Backward compatibility
# ============================================================================


class TestBackwardCompat:
    """Verify backward-compat re-exports from :mod:`datalab_kernel.plotter`."""

    def test_plot_result_alias(self):
        """``PlotResult`` re-export is identical to ``MplPlotResult``."""
        assert PlotResult is MplPlotResult

    def test_multi_signal_alias(self):
        """``MultiSignalPlotResult`` is ``MplMultiSignalResult``."""
        assert MultiSignalPlotResult is MplMultiSignalResult

    def test_multi_image_alias(self):
        """``MultiImagePlotResult`` is ``MplMultiImageResult``."""
        assert MultiImagePlotResult is MplMultiImageResult


# ============================================================================
# Plotter facade
# ============================================================================


class TestPlotterFacade:
    """Verify the Plotter facade delegates to MatplotlibPlotter."""

    def test_delegate_type(self):
        """Plotter._delegate is a MatplotlibPlotter."""
        ws = _make_workspace()
        plotter = Plotter(ws)
        assert isinstance(plotter._delegate, MatplotlibPlotter)

    def test_plot_signal(self):
        """Plotter.plot(signal) returns MplPlotResult."""
        ws = _make_workspace()
        sig = make_test_signal("sig")
        ws.add("sig", sig)
        plotter = Plotter(ws)
        result = plotter.plot(sig)
        assert isinstance(result, MplPlotResult)

    def test_plot_by_name(self):
        """Plotter.plot('name') resolves from workspace."""
        ws = _make_workspace()
        ws.add("s", make_test_signal("s"))
        plotter = Plotter(ws)
        result = plotter.plot("s")
        assert isinstance(result, MplPlotResult)

    def test_plot_missing_raises(self):
        """Plotter.plot('unknown') raises KeyError."""
        plotter = Plotter(_make_workspace())
        with pytest.raises(KeyError, match="not found"):
            plotter.plot("unknown")

    def test_plot_signals(self):
        """Plotter.plot_signals returns MplMultiSignalResult."""
        ws = _make_workspace()
        sigs = [make_test_signal(f"s{i}") for i in range(3)]
        plotter = Plotter(ws)
        result = plotter.plot_signals(sigs)
        assert isinstance(result, MplMultiSignalResult)

    def test_plot_images(self):
        """Plotter.plot_images returns MplMultiImageResult."""
        ws = _make_workspace()
        imgs = [make_test_image(f"i{i}", (64, 64)) for i in range(2)]
        plotter = Plotter(ws)
        result = plotter.plot_images(imgs)
        assert isinstance(result, MplMultiImageResult)

    def test_display_table(self):
        """Plotter.display_table returns TableResultDisplay."""
        plotter = Plotter(_make_workspace())
        table = make_table_result_stats()
        result = plotter.display_table(table)
        assert isinstance(result, TableResultDisplay)

    def test_display_geometry(self):
        """Plotter.display_geometry returns GeometryResultDisplay."""
        plotter = Plotter(_make_workspace())
        geom = make_geometry_result_points()
        result = plotter.display_geometry(geom)
        assert isinstance(result, GeometryResultDisplay)


# ============================================================================
# Signal rendering
# ============================================================================


class TestMplSignalRendering:
    """Test matplotlib signal rendering features."""

    def test_basic_signal(self, gui_mode):
        """Basic sine wave renders to valid PNG."""
        sig = make_test_signal("basic_sine")
        result = MplPlotResult(sig, title="Basic Signal")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_with_errorbars(self, gui_mode):
        """Signal with dx/dy error bars renders without error."""
        sig = make_test_signal_with_errorbars()
        result = MplPlotResult(sig, title="Error Bars")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_with_roi(self, gui_mode):
        """Signal with ROI segments renders with shaded regions."""
        sig = make_test_signal_with_roi()
        result = MplPlotResult(sig, title="Signal ROI", show_roi=True)
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_sticks(self, gui_mode):
        """Signal with curvestyle='Sticks' renders stem plot."""
        sig = make_test_signal_sticks()
        result = MplPlotResult(sig, title="Stick Style")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_steps(self, gui_mode):
        """Signal with curvestyle='Steps' renders step plot."""
        sig = make_test_signal_steps()
        result = MplPlotResult(sig, title="Step Style")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_log_scale(self, gui_mode):
        """Signal with log scale on both axes."""
        sig = make_test_signal_log()
        result = MplPlotResult(sig, title="Log Scale")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_styled(self, gui_mode):
        """Signal with custom color/linestyle/linewidth metadata."""
        sig = make_test_signal_styled()
        result = MplPlotResult(sig, title="Styled Signal")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_rich(self, gui_mode):
        """Rich signal with all features renders without error."""
        sig = make_test_signal_rich()
        result = MplPlotResult(sig, title="Rich Signal")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_signal_html(self):
        """Signal _repr_html_ returns valid HTML with embedded image."""
        sig = make_test_signal("html_test")
        result = MplPlotResult(sig, title="HTML Test")
        html = result._repr_html_()
        _assert_valid_html(html)
        assert "data:image/png;base64," in html

    def test_signal_repr(self):
        """String representation includes type info."""
        sig = make_test_signal("repr_test")
        result = MplPlotResult(sig, title="repr_test")
        assert "PlotResult" in repr(result) or "MplPlotResult" in repr(result)
        assert "SignalObj" in repr(result)


# ============================================================================
# Image rendering
# ============================================================================


class TestMplImageRendering:
    """Test matplotlib image rendering features."""

    def test_basic_image(self, gui_mode):
        """Basic random image renders to valid PNG."""
        img = make_test_image("basic", (128, 128))
        result = MplPlotResult(img, title="Basic Image")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_gradient_image(self, gui_mode):
        """Gradient pattern image."""
        img = make_test_image("gradient", (128, 128), pattern="gradient")
        result = MplPlotResult(img, title="Gradient")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_gaussian_image(self, gui_mode):
        """Gaussian pattern image."""
        img = make_test_image("gaussian", (128, 128), pattern="gaussian")
        result = MplPlotResult(img, title="Gaussian")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_with_roi(self, gui_mode):
        """Image with rectangular + circular ROIs."""
        img = make_test_image_with_roi()
        result = MplPlotResult(img, title="Image ROI", show_roi=True)
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_with_mask(self, gui_mode):
        """Image with mask overlay (from ROI)."""
        img = make_test_image_with_mask()
        result = MplPlotResult(img, title="Masked Image")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_with_colormap(self, gui_mode):
        """Image with custom colormap metadata."""
        img = make_test_image_with_colormap(colormap="hot")
        result = MplPlotResult(img, title="Hot Colormap")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_colormap_kwarg(self, gui_mode):
        """Image with colormap passed as kwarg."""
        img = make_test_image("cmap_kwarg", (128, 128), pattern="gaussian")
        result = MplPlotResult(img, title="Plasma kwarg", colormap="plasma")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_with_lut(self, gui_mode):
        """Image with custom LUT range (zscalemin/zscalemax)."""
        img = make_test_image_with_lut(vmin=0.3, vmax=0.7)
        result = MplPlotResult(img, title="LUT Range")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_nonuniform(self, gui_mode):
        """Image with non-uniform coordinates (pcolormesh)."""
        img = make_test_image_nonuniform()
        result = MplPlotResult(img, title="Non-Uniform")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_complex(self, gui_mode):
        """Complex image renders absolute values."""
        img = make_test_image_complex()
        result = MplPlotResult(img, title="Complex Image")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_rich(self, gui_mode):
        """Rich image with all features renders without error."""
        img = make_test_image_rich()
        result = MplPlotResult(img, title="Rich Image")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_image_html(self):
        """Image _repr_html_ returns valid HTML with embedded image."""
        img = make_test_image("html_test", (64, 64))
        result = MplPlotResult(img, title="HTML Test")
        html = result._repr_html_()
        _assert_valid_html(html)
        assert "data:image/png;base64," in html


# ============================================================================
# Multi-signal rendering
# ============================================================================


class TestMplMultiSignal:
    """Test multi-signal plot rendering."""

    def test_basic_multi_signal(self, gui_mode):
        """Multiple signals on one plot."""
        sigs = [make_test_signal(f"sig_{i}", freq=0.5 * (i + 1)) for i in range(3)]
        result = MplMultiSignalResult(sigs, title="Multi Signal")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_with_roi(self, gui_mode):
        """Multi-signal with ROIs."""
        sigs = [make_test_signal_with_roi(f"roi_{i}") for i in range(2)]
        result = MplMultiSignalResult(sigs, title="Multi ROI", show_roi=True)
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_mixed_styles(self, gui_mode):
        """Mix of regular, sticks, and error bar signals."""
        sigs = [
            make_test_signal("normal", freq=1.0),
            make_test_signal_sticks("sticks"),
            make_test_signal_with_errorbars("errorbars"),
        ]
        result = MplMultiSignalResult(sigs, title="Mixed Styles")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_numpy_arrays(self, gui_mode):
        """Multi-signal with plain numpy arrays."""
        y1 = np.sin(np.linspace(0, 10, 100))
        y2 = np.cos(np.linspace(0, 10, 100))
        result = MplMultiSignalResult([y1, y2], title="NumPy Arrays")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_tuples(self, gui_mode):
        """Multi-signal with (x, y) tuples."""
        x = np.linspace(0, 10, 100)
        result = MplMultiSignalResult([(x, np.sin(x)), (x, np.cos(x))], title="Tuples")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_signal_html(self):
        """Multi-signal _repr_html_ returns valid HTML."""
        sigs = [make_test_signal(f"s{i}") for i in range(2)]
        result = MplMultiSignalResult(sigs, title="HTML Test")
        _assert_valid_html(result._repr_html_())

    def test_multi_signal_repr(self):
        """String representation shows count."""
        sigs = [make_test_signal(f"s{i}") for i in range(3)]
        result = MplMultiSignalResult(sigs)
        assert "3" in repr(result)

    def test_multi_signal_with_labels(self, gui_mode):
        """Multi-signal with explicit axis labels/units."""
        sigs = [make_test_signal(f"s{i}") for i in range(2)]
        result = MplMultiSignalResult(
            sigs,
            title="Labeled",
            xlabel="Frequency",
            ylabel="Amplitude",
            xunit="Hz",
            yunit="dB",
        )
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)


# ============================================================================
# Multi-image rendering
# ============================================================================


class TestMplMultiImage:
    """Test multi-image grid rendering."""

    def test_basic_multi_image(self, gui_mode):
        """Multiple images in a grid."""
        imgs = [
            make_test_image(f"img_{i}", (64, 64), pattern)
            for i, pattern in enumerate(["random", "gradient", "gaussian"])
        ]
        result = MplMultiImageResult(imgs, title="Multi Image")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_with_roi(self, gui_mode):
        """Multi-image with ROIs."""
        imgs = [make_test_image_with_roi(f"roi_{i}") for i in range(2)]
        result = MplMultiImageResult(imgs, title="Multi ROI", show_roi=True)
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_with_colormaps(self, gui_mode):
        """Each image with a different colormap."""
        imgs = [
            make_test_image_with_colormap("hot", "hot"),
            make_test_image_with_colormap("cool", "cool"),
        ]
        result = MplMultiImageResult(imgs, title="Different Colormaps")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_numpy_arrays(self, gui_mode):
        """Multi-image with plain numpy arrays."""
        arrays = [np.random.rand(64, 64).astype(np.float32) for _ in range(4)]
        result = MplMultiImageResult(arrays, title="NumPy Arrays")
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_with_titles(self, gui_mode):
        """Multi-image with per-image titles."""
        imgs = [make_test_image(f"i{i}", (64, 64)) for i in range(3)]
        result = MplMultiImageResult(
            imgs, title="Titled", titles=["Alpha", "Beta", "Gamma"]
        )
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_with_geometry(self, gui_mode):
        """Multi-image with explicit geometry results."""
        imgs = [make_test_image("img", (128, 128), "gaussian")]
        circles = make_geometry_result_circles()
        result = MplMultiImageResult(imgs, title="With Geometry", results=[circles])
        _assert_valid_png(result._repr_png_())
        _show_if_gui(gui_mode, result)

    def test_multi_image_html(self):
        """Multi-image _repr_html_ returns valid HTML."""
        imgs = [make_test_image(f"i{i}", (64, 64)) for i in range(2)]
        result = MplMultiImageResult(imgs, title="HTML Test")
        _assert_valid_html(result._repr_html_())

    def test_multi_image_repr(self):
        """String representation shows count."""
        imgs = [make_test_image(f"i{i}", (64, 64)) for i in range(4)]
        result = MplMultiImageResult(imgs)
        assert "4" in repr(result)


# ============================================================================
# Geometry/table result display
# ============================================================================


class TestTableResultDisplay:
    """Test TableResultDisplay rendering."""

    def test_table_html(self):
        """TableResultDisplay renders valid HTML table."""
        table = make_table_result_stats()
        display = TableResultDisplay(table)
        html = display._repr_html_()
        _assert_valid_html(html)
        assert "Mean" in html

    def test_table_multi_row_html(self):
        """Multi-row table renders all rows."""
        table = make_table_result_multi_row()
        display = TableResultDisplay(table)
        html = display._repr_html_()
        _assert_valid_html(html)
        assert "FWHM" in html

    def test_table_repr(self):
        """String representation includes title."""
        table = make_table_result_stats()
        display = TableResultDisplay(table, title="My Stats")
        assert "My Stats" in repr(display)


class TestGeometryResultDisplay:
    """Test GeometryResultDisplay rendering."""

    def test_geometry_points_html(self):
        """GeometryResultDisplay with points renders valid HTML."""
        geom = make_geometry_result_points()
        display = GeometryResultDisplay(geom)
        html = display._repr_html_()
        _assert_valid_html(html)

    def test_geometry_circles_html(self):
        """GeometryResultDisplay with circles renders valid HTML."""
        geom = make_geometry_result_circles()
        display = GeometryResultDisplay(geom)
        html = display._repr_html_()
        _assert_valid_html(html)

    def test_geometry_segments_html(self):
        """GeometryResultDisplay with segments renders valid HTML."""
        geom = make_geometry_result_segments()
        display = GeometryResultDisplay(geom)
        html = display._repr_html_()
        _assert_valid_html(html)

    def test_geometry_repr(self):
        """String representation includes kind and title."""
        geom = make_geometry_result_points()
        display = GeometryResultDisplay(geom)
        r = repr(display)
        assert "GeometryResultDisplay" in r
        assert "Peak detection" in r


# ============================================================================
# Edge cases
# ============================================================================


class TestMplEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_roi_list(self):
        """Signal with empty ROI list renders fine."""
        sig = make_test_signal("no_roi")
        sig.roi = None
        result = MplPlotResult(sig, show_roi=True)
        _assert_valid_png(result._repr_png_())

    def test_show_results_false(self):
        """show_results=False skips geometry/table extraction."""
        sig = make_test_signal_rich()
        result = MplPlotResult(sig, show_results=False)
        _assert_valid_png(result._repr_png_())

    def test_show_roi_false(self):
        """show_roi=False skips ROI overlay."""
        sig = make_test_signal_with_roi()
        result = MplPlotResult(sig, show_roi=False)
        _assert_valid_png(result._repr_png_())

    def test_signal_no_metadata(self):
        """Signal with no metadata options (no __curvestyle, etc.)."""
        from sigima import create_signal

        x = np.linspace(0, 5, 50)
        y = np.sin(x)
        sig = create_signal(title="bare", x=x, y=y)
        result = MplPlotResult(sig)
        _assert_valid_png(result._repr_png_())

    def test_image_no_metadata(self):
        """Image with no metadata options (no __colormap, etc.)."""
        from sigima import create_image

        data = np.random.rand(64, 64).astype(np.float32)
        img = create_image(title="bare", data=data)
        result = MplPlotResult(img)
        _assert_valid_png(result._repr_png_())

    def test_small_image(self):
        """Very small image (4x4) renders."""
        img = make_test_image("tiny", (4, 4))
        result = MplPlotResult(img)
        _assert_valid_png(result._repr_png_())

    def test_single_point_signal(self):
        """Signal with a single data point."""
        from sigima import create_signal

        sig = create_signal(title="1pt", x=np.array([1.0]), y=np.array([2.0]))
        result = MplPlotResult(sig)
        _assert_valid_png(result._repr_png_())

    def test_unsupported_type_raises(self):
        """Unsupported data type in multi-signal raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            MplMultiSignalResult([42], title="bad")._render_to_png()

    def test_unsupported_image_type_raises(self):
        """Unsupported data type in multi-image raises TypeError."""
        with pytest.raises(TypeError, match="Unsupported"):
            MplMultiImageResult(["not_an_image"], title="bad")._render_to_png()
