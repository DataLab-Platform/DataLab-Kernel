# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for SignalObj and ImageObj
=====================================

Tests for the Sigima object re-exports and helper functions.
"""

from __future__ import annotations

import numpy as np
from sigima import (
    ImageObj,
    SignalObj,
    create_image,
    create_signal,
)


class TestSignalObj:
    """Tests for SignalObj class (re-exported from Sigima)."""

    def test_signal_creation(self):
        """Verify SignalObj can be created."""
        signal = SignalObj()
        assert signal is not None

    def test_signal_set_xydata(self):
        """Verify set_xydata works."""
        signal = SignalObj()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        signal.set_xydata(x, y)

        assert len(signal.x) == 100
        assert len(signal.y) == 100
        np.testing.assert_array_almost_equal(signal.x, x)

    def test_signal_xydata_property(self):
        """Verify xydata property returns data."""
        signal = SignalObj()
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        signal.set_xydata(x, y)

        xydata = signal.xydata

        # Sigima's xydata is a 2D array with shape (2, N)
        assert xydata is not None
        assert len(xydata) == 2

    def test_signal_copy(self):
        """Verify copy creates independent object."""
        signal = SignalObj()
        signal.set_xydata(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))
        signal.title = "original"

        copied = signal.copy()
        copied.title = "copy"
        copied.y[0] = 999

        assert signal.title == "original"
        assert signal.y[0] == 4.0

    def test_signal_dtype_conversion(self):
        """Verify integer arrays are converted to float64."""
        signal = SignalObj()
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])

        signal.set_xydata(x, y)

        assert signal.x.dtype == np.float64
        assert signal.y.dtype == np.float64

    def test_signal_repr(self):
        """Verify signal repr."""
        signal = create_signal("test", np.array([1, 2, 3]), np.array([4, 5, 6]))

        repr_str = repr(signal)

        assert "SignalObj" in repr_str


class TestImageObj:
    """Tests for ImageObj class (re-exported from Sigima)."""

    def test_image_creation(self):
        """Verify ImageObj can be created."""
        image = ImageObj()
        assert image is not None

    def test_image_set_data(self):
        """Verify data can be set."""
        image = ImageObj()
        data = np.random.rand(100, 100).astype(np.float32)

        image.data = data

        assert image.data.shape == (100, 100)

    def test_image_shape_property(self):
        """Verify shape property."""
        image = ImageObj()
        image.data = np.zeros((50, 100))

        # Sigima's ImageObj uses size property (height, width)
        assert image.data.shape == (50, 100)

    def test_image_copy(self):
        """Verify copy creates independent object."""
        image = ImageObj()
        image.data = np.ones((10, 10))
        image.title = "original"

        copied = image.copy()
        copied.title = "copy"
        copied.data[0, 0] = 999

        assert image.title == "original"
        assert image.data[0, 0] == 1

    def test_image_coordinates(self):
        """Verify coordinate attributes."""
        image = ImageObj()
        image.data = np.zeros((100, 100))
        image.x0 = 10.0
        image.y0 = 20.0
        image.dx = 0.5
        image.dy = 0.5

        assert image.x0 == 10.0
        assert image.y0 == 20.0
        assert image.dx == 0.5
        assert image.dy == 0.5

    def test_image_repr(self):
        """Verify image repr."""
        image = create_image("test", np.zeros((64, 64)))

        repr_str = repr(image)

        assert "ImageObj" in repr_str


class TestCreateFunctions:
    """Tests for create_signal and create_image helper functions."""

    def test_create_signal(self):
        """Verify create_signal function."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        signal = create_signal("test", x, y)

        assert signal is not None
        assert hasattr(signal, "x")
        assert hasattr(signal, "y")
        assert signal.title == "test"

    def test_create_signal_with_labels(self):
        """Verify create_signal sets labels and units."""
        signal = create_signal(
            "test",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            labels=("Time", "Amplitude"),
            units=("s", "V"),
        )

        assert signal.xlabel == "Time"
        assert signal.ylabel == "Amplitude"
        assert signal.xunit == "s"
        assert signal.yunit == "V"

    def test_create_image(self):
        """Verify create_image function."""
        data = np.random.rand(64, 64).astype(np.float32)

        image = create_image("test", data)

        assert image is not None
        assert hasattr(image, "data")
        assert image.title == "test"

    def test_create_image_with_labels(self):
        """Verify create_image sets labels and units."""
        image = create_image(
            "test",
            np.zeros((32, 32)),
            labels=("X", "Y", "Intensity"),
            units=("mm", "mm", "counts"),
        )

        assert image.xlabel == "X"
        assert image.ylabel == "Y"
        assert image.zlabel == "Intensity"
        assert image.xunit == "mm"
        assert image.yunit == "mm"
        assert image.zunit == "counts"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_signal(self):
        """Zero-length signal handled correctly."""
        signal = SignalObj()
        signal.set_xydata(np.array([]), np.array([]))

        assert len(signal.x) == 0
        assert len(signal.y) == 0

    def test_unicode_title(self):
        """Object names with Unicode characters work."""
        signal = create_signal(
            "Température (°C)",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        )

        assert signal.title == "Température (°C)"

    def test_special_characters_in_title(self):
        """Names with spaces, dots, hyphens work."""
        signal = create_signal(
            "my-signal.v2 (final)",
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        )

        assert signal.title == "my-signal.v2 (final)"
