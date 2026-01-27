# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for kernel components
================================

Tests kernel registration, startup, and namespace initialization.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import json

import pytest


class TestKernelSpec:
    """Tests for kernel specification and discovery."""

    def test_kernel_spec_valid(self):
        """Verify kernel.json specification is valid and contains required fields."""
        from datalab_kernel.install import get_kernel_spec

        spec = get_kernel_spec()

        assert "argv" in spec
        assert "display_name" in spec
        assert "language" in spec

        assert spec["display_name"] == "DataLab"
        assert spec["language"] == "python"
        assert len(spec["argv"]) >= 3
        # Should reference xpython or have connection file placeholder
        assert "{connection_file}" in " ".join(spec["argv"])

    def test_kernel_spec_json_serializable(self):
        """Verify kernel spec can be serialized to JSON."""
        from datalab_kernel.install import get_kernel_spec

        spec = get_kernel_spec()
        # Should not raise
        json_str = json.dumps(spec)
        assert json_str
        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed == spec

    def test_kernel_spec_has_debugger_metadata(self):
        """Verify kernel spec enables debugger support."""
        from datalab_kernel.install import get_kernel_spec

        spec = get_kernel_spec()
        assert "metadata" in spec
        assert spec["metadata"].get("debugger") is True

    def test_kernel_spec_has_env(self):
        """Verify kernel spec sets environment variables."""
        from datalab_kernel.install import get_kernel_spec

        spec = get_kernel_spec()
        assert "env" in spec
        assert spec["env"].get("DATALAB_KERNEL_STARTUP") == "1"


class TestKernelModule:
    """Tests for kernel module imports and structure."""

    def test_kernel_metadata_exists(self):
        """Verify kernel metadata constants exist and are valid."""
        from datalab_kernel.kernel import (
            KERNEL_BANNER,
            KERNEL_DISPLAY_NAME,
            KERNEL_HELP_LINKS,
            KERNEL_IMPLEMENTATION,
            KERNEL_LANGUAGE,
            KERNEL_LANGUAGE_INFO,
        )

        assert KERNEL_IMPLEMENTATION == "datalab-kernel"
        assert KERNEL_LANGUAGE == "python"
        assert KERNEL_DISPLAY_NAME == "DataLab"
        assert "xeus-python" in KERNEL_BANNER
        assert isinstance(KERNEL_LANGUAGE_INFO, dict)
        assert isinstance(KERNEL_HELP_LINKS, list)

    def test_get_kernel_info_function(self):
        """Verify get_kernel_info returns proper structure."""
        from datalab_kernel.kernel import get_kernel_info

        info = get_kernel_info()

        assert "protocol_version" in info
        assert "implementation" in info
        assert "implementation_version" in info
        assert "language_info" in info
        assert "banner" in info
        assert "help_links" in info

        assert info["implementation"] == "datalab-kernel"


class TestStartupModule:
    """Tests for the startup module."""

    def test_setup_namespace_returns_dict(self):
        """Verify setup_namespace returns a proper namespace dictionary."""
        from datalab_kernel.startup import setup_namespace

        ns = setup_namespace()

        assert isinstance(ns, dict)
        assert "workspace" in ns
        assert "plotter" in ns
        assert "np" in ns
        assert "create_signal" in ns
        assert "create_image" in ns

    def test_setup_namespace_has_sigima(self):
        """Verify sigima is included in namespace when available."""
        from datalab_kernel.startup import setup_namespace

        ns = setup_namespace()
        # sigima should be available in test environment
        assert "sigima" in ns


class TestNamespaceAvailability:
    """Tests for default namespace objects."""

    def test_workspace_class_importable(self):
        """Verify Workspace class is importable."""
        from datalab_kernel.workspace import Workspace

        assert Workspace is not None

    def test_plotter_class_importable(self):
        """Verify Plotter class is importable."""
        from datalab_kernel.plotter import Plotter

        assert Plotter is not None

    def test_objects_importable(self):
        """Verify object classes are importable from Sigima."""
        from sigima import ImageObj, SignalObj

        assert SignalObj is not None
        assert ImageObj is not None

    def test_create_functions_importable(self):
        """Verify create functions are importable from Sigima."""
        from sigima import create_image, create_signal

        assert callable(create_signal)
        assert callable(create_image)

    def test_numpy_available(self):
        """Verify NumPy is available."""
        import numpy as np

        assert np is not None
        assert hasattr(np, "array")


class TestModeDetection:
    """Tests for mode detection."""

    @pytest.mark.standalone
    def test_mode_detection_standalone(self):
        """Verify kernel can be forced to standalone mode."""
        from datalab_kernel.backends import StandaloneBackend
        from datalab_kernel.workspace import Workspace, WorkspaceMode

        # Create workspace with explicit standalone backend
        # (auto-detection would use live mode if DataLab is running)
        workspace = Workspace(backend=StandaloneBackend())
        assert workspace.mode == WorkspaceMode.STANDALONE

    def test_workspace_mode_enum(self):
        """Verify WorkspaceMode enum has required values."""
        from datalab_kernel.workspace import WorkspaceMode

        assert WorkspaceMode.STANDALONE.value == "standalone"
        assert WorkspaceMode.LIVE.value == "live"
