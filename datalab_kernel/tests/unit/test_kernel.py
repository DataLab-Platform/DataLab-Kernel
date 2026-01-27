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
        assert len(spec["argv"]) >= 4
        assert "-m" in spec["argv"]
        assert "datalab_kernel.kernel" in spec["argv"]

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


class TestKernelModule:
    """Tests for kernel module imports and structure."""

    def test_kernel_class_exists(self):
        """Verify DataLabKernel class exists and is importable."""
        from datalab_kernel.kernel import DataLabKernel

        assert DataLabKernel is not None

    def test_kernel_attributes(self):
        """Verify kernel class has required attributes."""
        from datalab_kernel.kernel import DataLabKernel

        assert hasattr(DataLabKernel, "implementation")
        assert hasattr(DataLabKernel, "implementation_version")
        assert hasattr(DataLabKernel, "language")
        assert hasattr(DataLabKernel, "language_info")
        assert hasattr(DataLabKernel, "banner")

        assert DataLabKernel.implementation == "datalab-kernel"
        assert DataLabKernel.language == "python"


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
