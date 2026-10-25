# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Web API Backend Tests
=====================

Unit tests for the WebApiBackend and NPZ serialization.
"""

from __future__ import annotations

import io
import json
import re
import zipfile

import h5py
import numpy as np
import pytest
from sigima import ImageObj, SignalObj

from datalab_kernel.backends.webapi import WebApiBackend
from datalab_kernel.serialization_npz import (
    deserialize_object_from_npz,
    serialize_object_to_npz,
)
from datalab_kernel.tests.data import make_test_image, make_test_image_nonuniform


class TestNPZSerialization:
    """Tests for NPZ serialization module."""

    @pytest.mark.parametrize("compress", [False, True])
    def test_signal_round_trip(self, compress):
        """Test serializing and deserializing a SignalObj."""
        # Create a signal
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        obj = SignalObj()
        obj.set_xydata(x, y)
        obj.title = "Test Signal"
        obj.xlabel = "Time"
        obj.ylabel = "Amplitude"
        obj.xunit = "s"
        obj.yunit = "V"
        obj.metadata["acquisition"] = {"sample": "signal", "gain": 2.5}

        # Serialize
        data = serialize_object_to_npz(obj, compress=compress)
        assert isinstance(data, bytes)
        assert len(data) > 0

        # Verify it's a valid zip
        buffer = io.BytesIO(data)
        with zipfile.ZipFile(buffer, "r") as zf:
            assert "x.npy" in zf.namelist()
            assert "y.npy" in zf.namelist()
            assert "metadata.json" in zf.namelist()

        # Deserialize
        result = deserialize_object_from_npz(data)

        # Verify
        assert type(result).__name__ == "SignalObj"
        np.testing.assert_array_equal(result.x, x)
        np.testing.assert_array_equal(result.y, y)
        assert result.title == "Test Signal"
        assert result.xlabel == "Time"
        assert result.ylabel == "Amplitude"
        assert result.xunit == "s"
        assert result.yunit == "V"
        assert result.metadata["acquisition"] == obj.metadata["acquisition"]
        assert result.dx is None
        assert result.dy is None

    def test_signal_with_uncertainties(self):
        """Test signal with dx/dy uncertainties."""
        x = np.linspace(0, 10, 50)
        y = np.cos(x)
        dx = np.ones_like(x) * 0.01
        dy = np.abs(y) * 0.05

        obj = SignalObj()
        obj.set_xydata(x, y, dx=dx, dy=dy)
        obj.title = "Signal with Errors"

        data = serialize_object_to_npz(obj)
        result = deserialize_object_from_npz(data)

        np.testing.assert_array_equal(result.dx, dx)
        np.testing.assert_array_equal(result.dy, dy)

    @pytest.mark.parametrize("compress", [False, True])
    def test_image_round_trip(self, compress):
        """Test serializing and deserializing an ImageObj."""
        # Create an image
        data = np.random.rand(128, 128).astype(np.float32)
        obj = ImageObj()
        obj.data = data
        obj.title = "Test Image"
        obj.xlabel = "X"
        obj.ylabel = "Y"
        obj.zlabel = "Intensity"
        obj.xunit = "mm"
        obj.yunit = "mm"
        obj.zunit = "counts"
        obj.metadata["acquisition"] = {"sample": "image", "gain": 2.5}
        obj.x0 = 10.0
        obj.y0 = 20.0
        obj.dx = 0.5
        obj.dy = 0.5

        # Serialize
        npz_data = serialize_object_to_npz(obj, compress=compress)
        assert isinstance(npz_data, bytes)

        # Verify structure
        buffer = io.BytesIO(npz_data)
        with zipfile.ZipFile(buffer, "r") as zf:
            assert "data.npy" in zf.namelist()
            assert "metadata.json" in zf.namelist()

        # Deserialize
        result = deserialize_object_from_npz(npz_data)

        # Verify
        assert type(result).__name__ == "ImageObj"
        np.testing.assert_array_equal(result.data, data)
        assert result.title == "Test Image"
        assert result.xlabel == "X"
        assert result.ylabel == "Y"
        assert result.zlabel == "Intensity"
        assert result.xunit == "mm"
        assert result.yunit == "mm"
        assert result.zunit == "counts"
        assert result.metadata["acquisition"] == obj.metadata["acquisition"]
        assert result.x0 == 10.0
        assert result.y0 == 20.0
        assert result.dx == 0.5
        assert result.dy == 0.5

    def test_image_preserves_dtype(self):
        """Test that image dtype is preserved through serialization."""
        for dtype in [np.uint8, np.uint16, np.float32, np.float64]:
            obj = ImageObj()
            obj.data = np.random.randint(0, 255, (64, 64)).astype(dtype)
            obj.title = f"Image {dtype.__name__}"

            data = serialize_object_to_npz(obj)
            result = deserialize_object_from_npz(data)

            assert result.data.dtype == dtype, (
                f"Expected {dtype}, got {result.data.dtype}"
            )

    @pytest.mark.parametrize("compress", [False, True])
    def test_image_nonuniform_round_trip(self, compress):
        """Non-uniform coordinates are kept, in the same layout as DataLab."""
        obj = make_test_image_nonuniform()

        npz_data = serialize_object_to_npz(obj, compress=compress)
        with zipfile.ZipFile(io.BytesIO(npz_data), "r") as zf:
            assert {"xcoords.npy", "ycoords.npy"} <= set(zf.namelist())
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["is_uniform_coords"] is False

        result = deserialize_object_from_npz(npz_data)
        assert not result.is_uniform_coords
        np.testing.assert_array_equal(result.xcoords, obj.xcoords)
        np.testing.assert_array_equal(result.ycoords, obj.ycoords)
        np.testing.assert_array_equal(result.data, obj.data)

    def test_image_uniform_has_no_coordinate_arrays(self):
        """Uniform coordinates are given by x0, y0, dx and dy only."""
        obj = make_test_image("uniform")
        obj.set_uniform_coords(dx=0.5, dy=0.25, x0=10.0, y0=-2.0)
        npz_data = serialize_object_to_npz(obj)
        with zipfile.ZipFile(io.BytesIO(npz_data), "r") as zf:
            assert "xcoords.npy" not in zf.namelist()
            assert "ycoords.npy" not in zf.namelist()
            metadata = json.loads(zf.read("metadata.json"))
        assert metadata["is_uniform_coords"] is True
        result = deserialize_object_from_npz(npz_data)
        assert result.is_uniform_coords
        assert (result.x0, result.y0, result.dx, result.dy) == (
            obj.x0,
            obj.y0,
            obj.dx,
            obj.dy,
        )

    def test_image_archive_without_coordinate_flag(self):
        """Archives written by earlier versions have uniform coordinates."""
        metadata = {"type": "image", "x0": 1.0, "y0": 2.0, "dx": 0.5, "dy": 0.25}
        npz_data = _make_image_archive(np.zeros((3, 4)), metadata)
        result = deserialize_object_from_npz(npz_data)
        assert result.is_uniform_coords
        assert (result.x0, result.y0, result.dx, result.dy) == (1.0, 2.0, 0.5, 0.25)

    @pytest.mark.parametrize(
        "arrays",
        [
            {},
            {"xcoords.npy": np.arange(4.0)},
            {"xcoords.npy": np.arange(3.0), "ycoords.npy": np.arange(3.0)},
        ],
        ids=["missing", "missing_y", "wrong_shape"],
    )
    def test_image_invalid_nonuniform_archive(self, arrays):
        """Missing or mismatched non-uniform coordinates are rejected."""
        metadata = {"type": "image", "is_uniform_coords": False}
        npz_data = _make_image_archive(np.zeros((3, 4)), metadata, arrays)
        with pytest.raises(ValueError, match="Invalid image NPZ"):
            deserialize_object_from_npz(npz_data)


def _make_image_archive(
    data: np.ndarray, metadata: dict, arrays: dict[str, np.ndarray] | None = None
) -> bytes:
    """Build an image NPZ archive by hand, as written by DataLab."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, array in {"data.npy": data, **(arrays or {})}.items():
            array_buffer = io.BytesIO()
            np.save(array_buffer, array, allow_pickle=False)
            zf.writestr(name, array_buffer.getvalue())
        zf.writestr("metadata.json", json.dumps(metadata))
    return buffer.getvalue()


class TestWebApiBackendHelpers:
    """Unit tests for WebApiBackend helper methods (no live server required)."""

    def test_encode_name_special_characters(self):
        """Test URL encoding of names with special characters."""
        # Create a backend instance (we won't actually connect)
        backend = WebApiBackend.__new__(WebApiBackend)

        # Test various special characters
        test_cases = [
            ("simple", "simple"),
            ("with spaces", "with%20spaces"),
            ("Gauss(a=10)", "Gauss%28a%3D10%29"),
            ("μ=0,σ=1", "%CE%BC%3D0%2C%CF%83%3D1"),
            ("test/slash", "test%2Fslash"),
            ("test#hash", "test%23hash"),
            ("test?query", "test%3Fquery"),
        ]

        for original, expected in test_cases:
            # pylint: disable=protected-access
            result = backend._encode_name(original)
            assert result == expected, f"Expected {expected!r}, got {result!r}"

    def test_short_id_pattern_matching(self):
        """Test short ID pattern detection (no server)."""
        # The pattern from the implementation
        pattern = r"^([si])(\d{3})$"

        # Should match
        valid_cases = ["s001", "s123", "i001", "i999", "S001", "I042"]
        for case in valid_cases:
            match = re.match(pattern, case.lower())
            assert match is not None, f"Should match: {case}"

        # Should not match
        invalid_cases = [
            "s01",  # Not enough digits
            "s0001",  # Too many digits
            "x001",  # Wrong prefix
            "signal1",  # Not a short ID
            "",  # Empty
            "s",  # Just letter
            "001",  # Just numbers
        ]
        for case in invalid_cases:
            match = re.match(pattern, case.lower())
            assert match is None, f"Should not match: {case}"

    @pytest.mark.parametrize("nonuniform", [False, True])
    def test_hdf5_group_keeps_image_coordinates(self, tmp_path, nonuniform):
        """Saving a workspace locally keeps uniform and non-uniform coordinates."""
        backend = WebApiBackend.__new__(WebApiBackend)
        if nonuniform:
            obj = make_test_image_nonuniform()
        else:
            obj = make_test_image("uniform")
            obj.set_uniform_coords(dx=0.5, dy=0.25, x0=10.0, y0=-2.0)
        filepath = tmp_path / "workspace.h5"
        # pylint: disable=protected-access
        with h5py.File(filepath, "w") as f:
            backend._save_object_to_group(f.create_group("img"), obj)
        with h5py.File(filepath, "r") as f:
            result = backend._load_object_from_group(f["img"], "img")
        assert result.is_uniform_coords is not nonuniform
        np.testing.assert_array_equal(result.xcoords, obj.xcoords)
        np.testing.assert_array_equal(result.ycoords, obj.ycoords)
        if not nonuniform:
            assert (result.x0, result.y0, result.dx, result.dy) == (
                10.0,
                -2.0,
                0.5,
                0.25,
            )


class TestWorkspaceBackendSelection:
    """Tests for workspace backend auto-detection."""

    @pytest.mark.standalone
    def test_standalone_mode_default(self, monkeypatch):
        """Test that standalone mode is selected when no DataLab is available.

        Note: This test only runs without --live flag, since it tests the fallback
        behavior when DataLab is not running.
        """
        # Clear any environment variables that would trigger live mode
        monkeypatch.delenv("DATALAB_WORKSPACE_URL", raising=False)
        monkeypatch.delenv("DATALAB_WORKSPACE_TOKEN", raising=False)
        monkeypatch.delenv("DATALAB_KERNEL_MODE", raising=False)

        # pylint: disable=import-outside-toplevel
        from datalab_kernel.backends import StandaloneBackend
        from datalab_kernel.workspace import Workspace, WorkspaceMode

        ws = Workspace()
        assert ws.mode == WorkspaceMode.STANDALONE
        # pylint: disable=protected-access
        assert isinstance(ws._backend, StandaloneBackend)

    def test_forced_standalone_mode(self, monkeypatch):
        """Test that DATALAB_KERNEL_MODE=standalone forces standalone mode."""
        monkeypatch.setenv("DATALAB_KERNEL_MODE", "standalone")
        monkeypatch.setenv("DATALAB_WORKSPACE_URL", "http://127.0.0.1:9999")
        monkeypatch.setenv("DATALAB_WORKSPACE_TOKEN", "test-token")

        # pylint: disable=import-outside-toplevel
        from datalab_kernel.backends import StandaloneBackend
        from datalab_kernel.workspace import Workspace, WorkspaceMode

        ws = Workspace()
        assert ws.mode == WorkspaceMode.STANDALONE
        # pylint: disable=protected-access
        assert isinstance(ws._backend, StandaloneBackend)

    def test_status_method(self, monkeypatch):
        """Test the status() method."""
        monkeypatch.delenv("DATALAB_WORKSPACE_URL", raising=False)
        monkeypatch.setenv("DATALAB_KERNEL_MODE", "standalone")

        # pylint: disable=import-outside-toplevel
        from datalab_kernel.workspace import Workspace

        ws = Workspace()
        status = ws.status()

        assert status["mode"] == "standalone"
        assert status["backend"] == "StandaloneBackend"
        assert "object_count" in status


# Integration tests require a running DataLab instance with WebAPI enabled
@pytest.mark.webapi
@pytest.mark.integration
class TestWebApiBackendIntegration:
    """Integration tests for WebApiBackend (require DataLab with WebAPI running).

    These tests require:
    - DataLab running with WebAPI server started
    - Run with: pytest --live --start-datalab --webapi
    """

    def test_list_objects(self, webapi_backend):
        """Test listing objects from DataLab."""
        # Should return a list (may be empty)
        result = webapi_backend.list()
        assert isinstance(result, list)

    def test_add_and_get_signal(self, webapi_backend):
        """Test adding and retrieving a signal."""
        # Create a signal
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        obj = SignalObj()
        obj.set_xydata(x, y)
        obj.title = "WebAPI Test Signal"

        # Add to DataLab
        webapi_backend.add("webapi_test_signal", obj, overwrite=True)

        # Verify it exists
        assert webapi_backend.exists("webapi_test_signal")

        # Retrieve and verify
        retrieved = webapi_backend.get("webapi_test_signal")
        assert retrieved.title == "webapi_test_signal"
        np.testing.assert_array_almost_equal(retrieved.x, x)
        np.testing.assert_array_almost_equal(retrieved.y, y)

        # Cleanup
        webapi_backend.remove("webapi_test_signal")

    def test_add_and_get_image(self, webapi_backend):
        """Test adding and retrieving an image."""
        # Create an image
        data = np.random.rand(64, 64).astype(np.float32)
        obj = ImageObj()
        obj.data = data
        obj.title = "WebAPI Test Image"

        # Add to DataLab
        webapi_backend.add("webapi_test_image", obj, overwrite=True)

        # Verify it exists
        assert webapi_backend.exists("webapi_test_image")

        # Retrieve and verify
        retrieved = webapi_backend.get("webapi_test_image")
        assert retrieved.title == "webapi_test_image"
        np.testing.assert_array_almost_equal(retrieved.data, data)

        # Cleanup
        webapi_backend.remove("webapi_test_image")

    def test_add_and_get_nonuniform_image(self, webapi_backend):
        """Non-uniform coordinates survive a round trip through DataLab."""
        # pylint: disable=protected-access
        version = webapi_backend._client.get("/api/v1/status").json()["version"]
        if _version_tuple(version) < (1, 4, 0):
            pytest.skip(f"DataLab {version} does not transfer non-uniform coordinates")
        obj = make_test_image_nonuniform("webapi_test_nonuniform")

        webapi_backend.add("webapi_test_nonuniform", obj, overwrite=True)
        retrieved = webapi_backend.get("webapi_test_nonuniform")

        assert not retrieved.is_uniform_coords
        np.testing.assert_array_equal(retrieved.xcoords, obj.xcoords)
        np.testing.assert_array_equal(retrieved.ycoords, obj.ycoords)
        np.testing.assert_array_equal(retrieved.data, obj.data)

        webapi_backend.remove("webapi_test_nonuniform")


def _version_tuple(version: str) -> tuple[int, ...]:
    """Return the numeric release part of a version string (e.g. (1, 4, 0))."""
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


@pytest.mark.webapi
@pytest.mark.integration
class TestWebApiBackendSpecialCharacters:
    """Test handling of special characters in object names.

    These tests require DataLab with WebAPI running.
    """

    def test_url_encoding_special_chars(self, webapi_backend):
        """Test that special characters in names are properly URL-encoded."""
        # Create a signal with special characters in name
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        obj = SignalObj()
        obj.set_xydata(x, y)

        # Test various special characters
        special_name = "Gauss(a=10,μ=0,σ=1)"

        # Add to DataLab
        webapi_backend.add(special_name, obj, overwrite=True)

        # Verify it exists
        assert webapi_backend.exists(special_name)

        # Retrieve and verify
        retrieved = webapi_backend.get(special_name)
        np.testing.assert_array_almost_equal(retrieved.x, x)
        np.testing.assert_array_almost_equal(retrieved.y, y)

        # Cleanup
        webapi_backend.remove(special_name)

    def test_short_id_resolution(self, webapi_backend):
        """Test that short IDs (s001, i001) resolve to object names."""
        # Create test objects
        signal = SignalObj()
        signal.set_xydata(np.arange(10), np.arange(10))
        signal.title = "Short ID Test Signal"

        image = ImageObj()
        image.data = np.zeros((10, 10), dtype=np.uint8)
        image.title = "Short ID Test Image"

        # Clear existing objects first
        webapi_backend.clear()

        # Add objects
        webapi_backend.add("Short ID Test Signal", signal, overwrite=True)
        webapi_backend.add("Short ID Test Image", image, overwrite=True)

        # Verify short ID resolution works
        # s001 should refer to first signal
        retrieved_signal = webapi_backend.get("s001")
        np.testing.assert_array_equal(retrieved_signal.x, signal.x)

        # i001 should refer to first image
        retrieved_image = webapi_backend.get("i001")
        np.testing.assert_array_equal(retrieved_image.data, image.data)

        # Cleanup
        webapi_backend.clear()
