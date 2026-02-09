# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for Plotter API
==========================

Tests visualization capabilities in standalone mode.
"""
# pylint: disable=protected-access

from __future__ import annotations

import numpy as np
import pytest

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.matplotlib_backend import MplMultiImageResult, MplMultiSignalResult
from datalab_kernel.plotter import PlotResult, Plotter, _classify_object
from datalab_kernel.tests.data import make_test_image, make_test_signal
from datalab_kernel.workspace import Workspace

pytestmark = [pytest.mark.standalone]


class TestPlotterBasic:
    """Basic plotter operations."""

    def test_plotter_creation(self):
        """Verify plotter can be created."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace)
        assert plotter is not None

    def test_plotter_plot_signal(self):
        """plotter.plot(signal) returns PlotResult."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot(signal)

        assert isinstance(result, PlotResult)

    def test_plotter_plot_image(self):
        """plotter.plot(image) returns PlotResult."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")
        workspace.add("my_image", image)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot(image)

        assert isinstance(result, PlotResult)

    def test_plotter_plot_by_name(self):
        """plotter.plot('object_name') works."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal("my_signal"))
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_signal")

        assert isinstance(result, PlotResult)

    def test_plotter_plot_missing_raises(self):
        """plotter.plot('unknown') raises KeyError."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")

        with pytest.raises(KeyError, match="not found"):
            plotter.plot("unknown")


class TestPlotResultSignal:
    """Tests for PlotResult with signals."""

    def test_plot_result_signal_repr_html(self):
        """PlotResult._repr_html_() returns valid HTML for signal."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_signal")
        html = result._repr_html_()

        assert html is not None
        assert isinstance(html, str)
        assert "<" in html  # Contains HTML tags

    def test_plot_result_signal_repr_png(self):
        """PlotResult._repr_png_() returns valid PNG bytes for signal."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_signal")
        png = result._repr_png_()

        assert png is not None
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic bytes


class TestPlotResultImage:
    """Tests for PlotResult with images."""

    def test_plot_result_image_repr_html(self):
        """PlotResult._repr_html_() returns valid HTML for image."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")
        workspace.add("my_image", image)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_image")
        html = result._repr_html_()

        assert html is not None
        assert isinstance(html, str)
        assert "<" in html

    def test_plot_result_image_repr_png(self):
        """PlotResult._repr_png_() returns valid PNG bytes for image."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")
        workspace.add("my_image", image)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_image")
        png = result._repr_png_()

        assert png is not None
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


class TestPlotListInput:
    """Tests for plot() with list input and type auto-detection."""

    def test_plot_signal_list(self):
        """plot([signal, signal]) returns MplMultiSignalResult."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        sigs = [make_test_signal(f"s{i}") for i in range(3)]
        result = plotter.plot(sigs)
        assert isinstance(result, MplMultiSignalResult)

    def test_plot_image_list(self):
        """plot([image, image]) returns MplMultiImageResult."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        imgs = [make_test_image(f"i{i}") for i in range(2)]
        result = plotter.plot(imgs)
        assert isinstance(result, MplMultiImageResult)

    def test_plot_single_item_list_unwraps(self):
        """plot([single_signal]) unwraps to PlotResult."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        sig = make_test_signal("s0")
        result = plotter.plot([sig])
        assert isinstance(result, PlotResult)

    def test_plot_mixed_list_raises(self):
        """plot([signal, image]) raises TypeError."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        sig = make_test_signal("s")
        img = make_test_image("i")
        with pytest.raises(TypeError, match="Cannot mix"):
            plotter.plot([sig, img])

    def test_plot_raw_1d_array(self):
        """plot(1d_ndarray) is treated as a signal."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        result = plotter.plot(np.arange(100, dtype=float))
        assert isinstance(result, PlotResult)

    def test_plot_raw_2d_array(self):
        """plot(2d_ndarray) is treated as an image."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        result = plotter.plot(np.random.rand(64, 64))
        assert isinstance(result, PlotResult)

    def test_plot_raw_tuple(self):
        """plot((x, y) tuple) is treated as a signal."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        result = plotter.plot((x, y))
        assert isinstance(result, PlotResult)

    def test_plot_list_of_1d_arrays(self):
        """plot([1d, 1d]) returns MplMultiSignalResult."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        arrays = [np.arange(50, dtype=float), np.arange(50, dtype=float)]
        result = plotter.plot(arrays)
        assert isinstance(result, MplMultiSignalResult)

    def test_plot_list_of_2d_arrays(self):
        """plot([2d, 2d]) returns MplMultiImageResult."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace, backend="matplotlib")
        arrays = [np.random.rand(32, 32), np.random.rand(32, 32)]
        result = plotter.plot(arrays)
        assert isinstance(result, MplMultiImageResult)

    def test_plot_list_by_name(self):
        """plot(['name1', 'name2']) resolves names from workspace."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("s0", make_test_signal("s0"))
        workspace.add("s1", make_test_signal("s1"))
        plotter = Plotter(workspace, backend="matplotlib")
        result = plotter.plot(["s0", "s1"])
        assert isinstance(result, MplMultiSignalResult)


class TestClassifyObject:
    """Tests for _classify_object helper."""

    def test_signal_obj(self):
        """Test that a SignalObj is classified as "signal"."""
        assert _classify_object(make_test_signal("s")) == "signal"

    def test_image_obj(self):
        """Test that an ImageObj is classified as "image"."""
        assert _classify_object(make_test_image("i")) == "image"

    def test_1d_array(self):
        """Test that a 1D ndarray is classified as "signal"."""
        assert _classify_object(np.arange(10, dtype=float)) == "signal"

    def test_2d_array(self):
        """Test that a 2D ndarray is classified as "image"."""
        assert _classify_object(np.ones((4, 4))) == "image"

    def test_tuple_pair(self):
        """Test that a tuple of (x, y) arrays is classified as "signal"."""
        assert _classify_object((np.arange(5), np.arange(5))) == "signal"

    def test_unsupported_raises(self):
        """Test that an unsupported object raises TypeError."""
        with pytest.raises(TypeError):
            _classify_object("not_an_object")

    def test_3d_array_raises(self):
        """Test that a 3D ndarray raises TypeError."""
        with pytest.raises(TypeError):
            _classify_object(np.ones((2, 3, 4)))


class TestPlotResultRepr:
    """Tests for PlotResult string representation."""

    def test_plot_result_repr_signal(self):
        """Verify PlotResult repr for signal."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_signal")
        repr_str = repr(result)

        assert "PlotResult" in repr_str
        assert "SignalObj" in repr_str

    def test_plot_result_repr_image(self):
        """Verify PlotResult repr for image."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")
        workspace.add("my_image", image)
        plotter = Plotter(workspace, backend="matplotlib")

        result = plotter.plot("my_image")
        repr_str = repr(result)

        assert "PlotResult" in repr_str
        assert "ImageObj" in repr_str
