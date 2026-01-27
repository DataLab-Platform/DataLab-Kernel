# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for Workspace persistence (HDF5)
===========================================

Tests save/load functionality.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.tests.data import make_test_image, make_test_signal
from datalab_kernel.workspace import Workspace


class TestWorkspaceSave:
    """Tests for saving workspace to HDF5."""

    def test_workspace_save_h5(self):
        """workspace.save('file.h5') creates valid HDF5 file."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            assert os.path.exists(filepath)

    def test_workspace_save_adds_extension(self):
        """save() adds .h5 extension if missing."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test")
            workspace.save(filepath)

            assert os.path.exists(filepath + ".h5")

    def test_workspace_save_empty(self):
        """Saving empty workspace creates valid file."""
        workspace = Workspace(backend=StandaloneBackend())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty.h5")
            workspace.save(filepath)

            assert os.path.exists(filepath)


class TestWorkspaceLoad:
    """Tests for loading workspace from HDF5."""

    def test_workspace_load_h5(self):
        """workspace.load('file.h5') restores objects."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            # Create new workspace and load
            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            assert "my_signal" in workspace2.list()

    def test_workspace_load_missing_file_raises(self):
        """Loading non-existent file raises FileNotFoundError."""
        workspace = Workspace(backend=StandaloneBackend())

        with pytest.raises(FileNotFoundError):
            workspace.load("nonexistent.h5")


class TestWorkspaceSaveLoadRoundtrip:
    """Tests for save/load roundtrip."""

    def test_workspace_save_load_roundtrip_signal(self):
        """Save then load preserves signal objects and metadata."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            loaded = workspace2.get("my_signal")
            assert loaded is not None
            assert hasattr(loaded, "x")
            assert hasattr(loaded, "y")
            assert len(loaded.x) == len(signal.x)
            np.testing.assert_array_almost_equal(loaded.x, signal.x)
            np.testing.assert_array_almost_equal(loaded.y, signal.y)

    def test_workspace_save_load_roundtrip_image(self):
        """Save then load preserves image objects and metadata."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image", shape=(64, 64))
        workspace.add("my_image", image)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            loaded = workspace2.get("my_image")
            assert loaded is not None
            assert hasattr(loaded, "data")
            assert loaded.data.shape == image.data.shape
            np.testing.assert_array_almost_equal(loaded.data, image.data)

    def test_workspace_save_load_roundtrip_multiple(self):
        """Save then load preserves all objects."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("sig1", make_test_signal("sig1"))
        workspace.add("sig2", make_test_signal("sig2"))
        workspace.add("img1", make_test_image("img1"))

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            assert len(workspace2) == 3
            assert "sig1" in workspace2
            assert "sig2" in workspace2
            assert "img1" in workspace2

    def test_workspace_save_load_metadata(self):
        """Save then load preserves object metadata."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        signal.xlabel = "Custom X"
        signal.ylabel = "Custom Y"
        signal.xunit = "MHz"
        signal.yunit = "dB"
        workspace.add("my_signal", signal)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            loaded = workspace2.get("my_signal")
            assert loaded.xlabel == "Custom X"
            assert loaded.ylabel == "Custom Y"
            assert loaded.xunit == "MHz"
            assert loaded.yunit == "dB"


class TestH5Format:
    """Tests for HDF5 file format."""

    def test_h5_readable_with_h5py(self):
        """Saved .h5 file is readable with h5py."""
        import h5py

        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal())

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            with h5py.File(filepath, "r") as f:
                assert "my_signal" in f
                assert "x" in f["my_signal"]
                assert "y" in f["my_signal"]
