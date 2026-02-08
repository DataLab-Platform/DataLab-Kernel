# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Tests for plotter backend selection
====================================

Tests the backend auto-detection, ``set_backend()``, environment variable
override, and fallback-with-warning behaviour.
"""

from __future__ import annotations

import contextlib
import os
import sys
from unittest import mock

import pytest

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.plotter import (
    BACKEND_ENV_VAR,
    BACKEND_MATPLOTLIB,
    BACKEND_PLOTLY,
    Plotter,
    matplotlib_available,
    plotly_available,
    resolve_backend,
)
from datalab_kernel.tests.data import make_test_signal
from datalab_kernel.workspace import Workspace

# Grab the *module* from sys.modules because the ``plotter`` attribute on the
# ``datalab_kernel`` package is shadowed by a global variable (``None`` until
# the kernel starts).  ``mock.patch.object`` then patches the correct namespace.
_plotter_mod = sys.modules["datalab_kernel.plotter"]

pytestmark = [pytest.mark.standalone]


def _make_workspace() -> Workspace:
    return Workspace(backend=StandaloneBackend())


# ============================================================================
# resolve_backend()
# ============================================================================


class TestResolveBackend:
    """Unit tests for :func:`resolve_backend`."""

    def test_explicit_matplotlib(self):
        """Explicit 'matplotlib' returns matplotlib when available."""
        assert resolve_backend("matplotlib") == BACKEND_MATPLOTLIB

    @pytest.mark.skipif(not plotly_available(), reason="plotly not installed")
    def test_explicit_plotly(self):
        """Explicit 'plotly' returns plotly when available."""
        assert resolve_backend("plotly") == BACKEND_PLOTLY

    def test_explicit_case_insensitive(self):
        """Backend name matching is case-insensitive."""
        assert resolve_backend("Matplotlib") == BACKEND_MATPLOTLIB
        assert resolve_backend("MATPLOTLIB") == BACKEND_MATPLOTLIB

    def test_invalid_backend_raises(self):
        """Unknown backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            resolve_backend("bokeh")

    def test_auto_prefers_plotly(self):
        """Auto-detect prefers plotly when both are available."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=True
                )
            )
            assert resolve_backend(None) == BACKEND_PLOTLY

    def test_auto_falls_back_to_matplotlib(self):
        """Auto-detect falls back to matplotlib when plotly is missing."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=True
                )
            )
            assert resolve_backend(None) == BACKEND_MATPLOTLIB

    def test_auto_plotly_only(self):
        """Auto-detect returns plotly when only plotly is available."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=True)
            )
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=False
                )
            )
            assert resolve_backend(None) == BACKEND_PLOTLY

    def test_neither_raises(self):
        """ImportError when neither backend is available."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=False
                )
            )
            with pytest.raises(ImportError, match="Neither plotly nor matplotlib"):
                resolve_backend(None)

    def test_fallback_with_warning(self):
        """Falls back to the other backend with a warning."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=True
                )
            )
            with pytest.warns(UserWarning, match="not installed.*Falling back"):
                result = resolve_backend("plotly")
            assert result == BACKEND_MATPLOTLIB

    def test_fallback_other_direction(self):
        """Falls back to plotly when matplotlib is requested but missing."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=False
                )
            )
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=True)
            )
            with pytest.warns(UserWarning, match="not installed.*Falling back"):
                result = resolve_backend("matplotlib")
            assert result == BACKEND_PLOTLY

    def test_both_missing_with_explicit_raises(self):
        """ImportError when explicit backend and fallback are both missing."""
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(_plotter_mod, "plotly_available", return_value=False)
            )
            stack.enter_context(
                mock.patch.object(
                    _plotter_mod, "matplotlib_available", return_value=False
                )
            )
            with pytest.raises(ImportError, match="Neither plotly nor matplotlib"):
                resolve_backend("plotly")


# ============================================================================
# Environment variable
# ============================================================================


class TestEnvVar:
    """Tests for DATALAB_PLOTTER_BACKEND environment variable."""

    def test_env_var_matplotlib(self):
        """Env var selects matplotlib."""
        with mock.patch.dict(os.environ, {BACKEND_ENV_VAR: "matplotlib"}):
            assert resolve_backend(None) == BACKEND_MATPLOTLIB

    @pytest.mark.skipif(not plotly_available(), reason="plotly not installed")
    def test_env_var_plotly(self):
        """Env var selects plotly."""
        with mock.patch.dict(os.environ, {BACKEND_ENV_VAR: "plotly"}):
            assert resolve_backend(None) == BACKEND_PLOTLY

    def test_env_var_invalid_ignored(self):
        """Invalid env var value is ignored with a warning."""
        with mock.patch.dict(os.environ, {BACKEND_ENV_VAR: "bokeh"}):
            with pytest.warns(UserWarning, match="Ignoring invalid"):
                # Falls through to auto-detect
                result = resolve_backend(None)
            assert result in {BACKEND_MATPLOTLIB, BACKEND_PLOTLY}

    def test_explicit_overrides_env(self):
        """Explicit argument takes precedence over env var."""
        with mock.patch.dict(os.environ, {BACKEND_ENV_VAR: "plotly"}):
            assert resolve_backend("matplotlib") == BACKEND_MATPLOTLIB


# ============================================================================
# Plotter class
# ============================================================================


class TestPlotterBackendProperty:
    """Tests for Plotter.backend and Plotter.set_backend()."""

    def test_default_backend_property(self):
        """Plotter.backend returns the auto-detected backend name."""
        ws = _make_workspace()
        plotter = Plotter(ws)
        assert plotter.backend in {BACKEND_MATPLOTLIB, BACKEND_PLOTLY}

    def test_explicit_matplotlib_backend(self):
        """Plotter(backend='matplotlib') uses matplotlib."""
        ws = _make_workspace()
        plotter = Plotter(ws, backend="matplotlib")
        assert plotter.backend == BACKEND_MATPLOTLIB

    @pytest.mark.skipif(not plotly_available(), reason="plotly not installed")
    def test_explicit_plotly_backend(self):
        """Plotter(backend='plotly') uses plotly."""
        ws = _make_workspace()
        plotter = Plotter(ws, backend="plotly")
        assert plotter.backend == BACKEND_PLOTLY

    def test_set_backend_switches(self):
        """set_backend() switches the delegate."""
        ws = _make_workspace()
        plotter = Plotter(ws, backend="matplotlib")
        assert plotter.backend == BACKEND_MATPLOTLIB

        # Switching to the same backend is a no-op (no error)
        result = plotter.set_backend("matplotlib")
        assert result is plotter  # returns self for chaining

    @pytest.mark.skipif(not plotly_available(), reason="plotly not installed")
    def test_set_backend_plotly_then_matplotlib(self):
        """set_backend() can switch between backends."""
        ws = _make_workspace()
        plotter = Plotter(ws, backend="plotly")
        assert plotter.backend == BACKEND_PLOTLY

        plotter.set_backend("matplotlib")
        assert plotter.backend == BACKEND_MATPLOTLIB

        plotter.set_backend("plotly")
        assert plotter.backend == BACKEND_PLOTLY

    def test_set_backend_invalid_raises(self):
        """set_backend() with unknown name raises ValueError."""
        ws = _make_workspace()
        plotter = Plotter(ws, backend="matplotlib")
        with pytest.raises(ValueError, match="Unknown backend"):
            plotter.set_backend("bokeh")

    def test_plotting_works_after_set_backend(self):
        """Plotting still works after switching backend."""
        ws = _make_workspace()
        sig = make_test_signal("sig")
        ws.add("sig", sig)
        plotter = Plotter(ws, backend="matplotlib")
        result = plotter.plot(sig)
        assert result is not None

        # Switch to same backend and plot again
        plotter.set_backend("matplotlib")
        result2 = plotter.plot(sig)
        assert result2 is not None


# ============================================================================
# Availability helpers
# ============================================================================


class TestAvailabilityHelpers:
    """Tests for matplotlib_available() and plotly_available()."""

    def test_matplotlib_available_returns_bool(self):
        """matplotlib_available() returns a boolean."""
        result = matplotlib_available()
        assert isinstance(result, bool)

    def test_plotly_available_returns_bool(self):
        """plotly_available() returns a boolean."""
        result = plotly_available()
        assert isinstance(result, bool)

    def test_matplotlib_is_available(self):
        """matplotlib should be available (it's a required dep)."""
        assert matplotlib_available() is True
