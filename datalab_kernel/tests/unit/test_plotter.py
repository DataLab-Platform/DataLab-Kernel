# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for Plotter API
==========================

Tests visualization capabilities in standalone mode.
"""
# pylint: disable=protected-access

from __future__ import annotations

import pytest

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.plotter import PlotResult, Plotter
from datalab_kernel.tests.data import make_test_image, make_test_signal
from datalab_kernel.workspace import Workspace


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
        plotter = Plotter(workspace)

        result = plotter.plot(signal)

        assert isinstance(result, PlotResult)

    def test_plotter_plot_image(self):
        """plotter.plot(image) returns PlotResult."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")
        workspace.add("my_image", image)
        plotter = Plotter(workspace)

        result = plotter.plot(image)

        assert isinstance(result, PlotResult)

    def test_plotter_plot_by_name(self):
        """plotter.plot('object_name') works."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal("my_signal"))
        plotter = Plotter(workspace)

        result = plotter.plot("my_signal")

        assert isinstance(result, PlotResult)

    def test_plotter_plot_missing_raises(self):
        """plotter.plot('unknown') raises KeyError."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace)

        with pytest.raises(KeyError, match="not found"):
            plotter.plot("unknown")


class TestPlotResultSignal:
    """Tests for PlotResult with signals."""

    def test_plot_result_signal_repr_html(self):
        """PlotResult._repr_html_() returns valid HTML for signal."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)
        plotter = Plotter(workspace)

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
        plotter = Plotter(workspace)

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
        plotter = Plotter(workspace)

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
        plotter = Plotter(workspace)

        result = plotter.plot("my_image")
        png = result._repr_png_()

        assert png is not None
        assert isinstance(png, bytes)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


class TestPlotResultRepr:
    """Tests for PlotResult string representation."""

    def test_plot_result_repr_signal(self):
        """Verify PlotResult repr for signal."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)
        plotter = Plotter(workspace)

        result = plotter.plot("my_signal")
        repr_str = repr(result)

        assert "PlotResult" in repr_str
        assert "SignalObj" in repr_str

    def test_plot_result_repr_image(self):
        """Verify PlotResult repr for image."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")
        workspace.add("my_image", image)
        plotter = Plotter(workspace)

        result = plotter.plot("my_image")
        repr_str = repr(result)

        assert "PlotResult" in repr_str
        assert "ImageObj" in repr_str
