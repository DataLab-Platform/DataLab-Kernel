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
from datalab_kernel.tests.data import (
    make_test_image,
    make_test_image_nonuniform,
    make_test_signal,
)
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

    def test_workspace_save_load_roundtrip_image_coordinates(self):
        """Save then load preserves uniform and non-uniform image coordinates."""
        workspace = Workspace(backend=StandaloneBackend())
        uniform = make_test_image("uniform", shape=(16, 32))
        uniform.set_uniform_coords(dx=0.5, dy=0.25, x0=10.0, y0=-2.0)
        nonuniform = make_test_image_nonuniform("nonuniform")
        workspace.add("uniform", uniform)
        workspace.add("nonuniform", nonuniform)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            workspace.save(filepath)

            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            loaded = workspace2.get("uniform")
            assert loaded.is_uniform_coords
            assert (loaded.x0, loaded.y0, loaded.dx, loaded.dy) == (
                10.0,
                -2.0,
                0.5,
                0.25,
            )
            loaded = workspace2.get("nonuniform")
            assert not loaded.is_uniform_coords
            np.testing.assert_array_equal(loaded.xcoords, nonuniform.xcoords)
            np.testing.assert_array_equal(loaded.ycoords, nonuniform.ycoords)

    def test_workspace_load_image_without_coordinate_flag(self):
        """Images of files written by earlier versions have uniform coordinates."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "legacy.h5")
            with h5py.File(filepath, "w") as f:
                grp = f.create_group("img")
                grp.attrs["type"] = "ImageObj"
                grp.create_dataset("data", data=np.zeros((3, 4)))
                for attr, value in {
                    "x0": 1.0,
                    "y0": 2.0,
                    "dx": 0.5,
                    "dy": 0.25,
                }.items():
                    grp.attrs[attr] = value

            workspace = Workspace(backend=StandaloneBackend())
            workspace.load(filepath)

            loaded = workspace.get("img")
            assert loaded.is_uniform_coords
            assert (loaded.x0, loaded.y0, loaded.dx, loaded.dy) == (1.0, 2.0, 0.5, 0.25)

    def test_workspace_load_image_with_invalid_coordinates(self):
        """Non-uniform coordinates that do not match the image are rejected."""
        import h5py

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "invalid.h5")
            with h5py.File(filepath, "w") as f:
                grp = f.create_group("img")
                grp.attrs["type"] = "ImageObj"
                grp.attrs["is_uniform_coords"] = False
                grp.create_dataset("data", data=np.zeros((3, 4)))
                grp.create_dataset("xcoords", data=np.arange(3.0))
                grp.create_dataset("ycoords", data=np.arange(3.0))

            workspace = Workspace(backend=StandaloneBackend())
            with pytest.raises(ValueError, match="do not match image shape"):
                workspace.load(filepath)

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
