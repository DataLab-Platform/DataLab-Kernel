# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Test data generation utilities
==============================

Provides functions to create test signals and images with varying levels
of richness (error bars, ROIs, metadata options, geometry/table results).

Helper functions
----------------
.. autofunction:: make_test_signal
.. autofunction:: make_test_signal_with_errorbars
.. autofunction:: make_test_signal_with_roi
.. autofunction:: make_test_signal_sticks
.. autofunction:: make_test_signal_steps
.. autofunction:: make_test_signal_log
.. autofunction:: make_test_signal_rich
.. autofunction:: make_test_image
.. autofunction:: make_test_image_with_roi
.. autofunction:: make_test_image_with_mask
.. autofunction:: make_test_image_with_colormap
.. autofunction:: make_test_image_nonuniform
.. autofunction:: make_test_image_rich
"""

from __future__ import annotations

import numpy as np
from sigima import create_image, create_image_roi, create_signal, create_signal_roi
from sigima.objects import (
    CircularROI,
    GeometryResult,
    ImageObj,
    ImageROI,
    KindShape,
    RectangularROI,
    SignalObj,
    TableKind,
    TableResult,
)
from sigima.objects.scalar.common import NO_ROI

# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------


def make_test_signal(
    name: str = "test_signal",
    n_points: int = 1000,
    freq: float = 1.0,
) -> SignalObj:
    """Create a simple test signal (sine wave with noise).

    Args:
        name: Signal title
        n_points: Number of data points
        freq: Sine wave frequency

    Returns:
        SignalObj instance
    """
    x = np.linspace(0, 10, n_points)
    y = np.sin(2 * np.pi * freq * x) + 0.1 * np.random.randn(len(x))
    return create_signal(
        title=name,
        x=x,
        y=y,
        labels=("Time", "Amplitude"),
        units=("s", "V"),
    )


def make_test_signal_with_errorbars(
    name: str = "signal_errorbars",
    n_points: int = 200,
) -> SignalObj:
    """Create a signal with dx and dy error bars.

    Args:
        name: Signal title
        n_points: Number of data points

    Returns:
        SignalObj with dx and dy arrays set
    """
    x = np.linspace(0, 10, n_points)
    y = np.exp(-0.3 * x) * np.sin(2 * np.pi * 0.5 * x)
    dx = np.full_like(x, 0.05)
    dy = np.abs(y) * 0.15 + 0.02
    return create_signal(
        title=name,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        labels=("Frequency", "Power"),
        units=("Hz", "dB"),
    )


def make_test_signal_with_roi(
    name: str = "signal_roi",
    n_points: int = 500,
) -> SignalObj:
    """Create a signal with two ROI segments.

    Args:
        name: Signal title
        n_points: Number of data points

    Returns:
        SignalObj with two SegmentROI attached
    """
    x = np.linspace(0, 10, n_points)
    y = np.sin(2 * np.pi * x)
    sig = create_signal(
        title=name, x=x, y=y, labels=("Time", "Voltage"), units=("s", "mV")
    )
    sig.roi = create_signal_roi([[1.0, 3.5], [6.0, 8.5]])
    return sig


def make_test_signal_sticks(
    name: str = "signal_sticks",
    n_points: int = 50,
) -> SignalObj:
    """Create a signal with ``curvestyle='Sticks'`` metadata.

    Args:
        name: Signal title
        n_points: Number of data points

    Returns:
        SignalObj configured for stick-style rendering
    """
    x = np.linspace(0, 10, n_points)
    y = np.sin(2 * np.pi * 0.3 * x)
    sig = create_signal(title=name, x=x, y=y)
    sig.set_metadata_option("curvestyle", "Sticks")
    return sig


def make_test_signal_steps(
    name: str = "signal_steps",
    n_points: int = 30,
) -> SignalObj:
    """Create a signal with ``curvestyle='Steps'`` metadata.

    Args:
        name: Signal title
        n_points: Number of data points

    Returns:
        SignalObj configured for step-style rendering
    """
    x = np.linspace(0, 5, n_points)
    y = np.floor(np.sin(2 * np.pi * 0.5 * x) * 3)
    sig = create_signal(title=name, x=x, y=y)
    sig.set_metadata_option("curvestyle", "Steps")
    return sig


def make_test_signal_log(
    name: str = "signal_log",
) -> SignalObj:
    """Create a signal with logarithmic axis scale.

    Args:
        name: Signal title

    Returns:
        SignalObj with ``xscalelog`` and ``yscalelog`` set to True
    """
    x = np.logspace(0, 3, 200)
    y = 1.0 / x
    sig = create_signal(
        title=name, x=x, y=y, labels=("Frequency", "Magnitude"), units=("Hz", "dB")
    )
    sig.xscalelog = True
    sig.yscalelog = True
    return sig


def make_test_signal_styled(
    name: str = "signal_styled",
    n_points: int = 200,
) -> SignalObj:
    """Create a signal with custom line style via metadata.

    Sets ``color``, ``linestyle``, and ``linewidth`` in the metadata dict.

    Args:
        name: Signal title
        n_points: Number of data points

    Returns:
        SignalObj with line styling metadata
    """
    x = np.linspace(0, 10, n_points)
    y = np.cos(2 * np.pi * 0.5 * x)
    sig = create_signal(title=name, x=x, y=y)
    sig.metadata["color"] = "red"
    sig.metadata["linestyle"] = "DashLine"
    sig.metadata["linewidth"] = 2
    return sig


def make_test_signal_rich(
    name: str = "signal_rich",
    n_points: int = 500,
) -> SignalObj:
    """Create a rich signal with all features (errorbars, ROI, geometry,
    table results, axis bounds, style).

    Args:
        name: Signal title
        n_points: Number of data points

    Returns:
        Fully featured SignalObj
    """
    x = np.linspace(0, 10, n_points)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(n_points)
    dx = np.full(n_points, 0.01)
    dy = np.abs(y) * 0.05

    sig = create_signal(
        title=name,
        x=x,
        y=y,
        dx=dx,
        dy=dy,
        labels=("Time", "Voltage"),
        units=("s", "mV"),
    )

    # ROIs
    sig.roi = create_signal_roi([[1.0, 3.0], [5.0, 8.0]])

    # Axis bounds
    sig.autoscale = False
    sig.xscalemin = 0.0
    sig.xscalemax = 10.0
    sig.yscalemin = -2.0
    sig.yscalemax = 2.0

    # Style metadata
    sig.set_metadata_option("shade", 0.2)
    sig.set_metadata_option("curvestyle", "Lines")

    # Geometry result (FWHM segment)
    geom = GeometryResult(
        title="FWHM",
        kind=KindShape.SEGMENT,
        coords=np.array([[2.0, 0.5, 4.0, 0.5]]),
        func_name="fwhm",
    )
    sig.metadata["Geometry_fwhm"] = geom.to_dict()

    # Table result (statistics)
    table = TableResult(
        title="Statistics",
        kind=TableKind.STATISTICS,
        headers=["Mean", "Std", "Min", "Max"],
        data=[[0.01, 0.72, -1.3, 1.35]],
        roi_indices=[NO_ROI],
        func_name="statistics",
    )
    sig.metadata["Table_statistics"] = table.to_dict()

    return sig


# ---------------------------------------------------------------------------
# Image generators
# ---------------------------------------------------------------------------


def make_test_image(
    name: str = "test_image",
    shape: tuple[int, int] = (256, 256),
    pattern: str = "random",
) -> ImageObj:
    """Create a simple test image.

    Args:
        name: Image title
        shape: Image dimensions (height, width)
        pattern: Pattern type ("random", "gradient", "gaussian")

    Returns:
        ImageObj instance
    """
    if pattern == "random":
        data = np.random.rand(*shape).astype(np.float32)
    elif pattern == "gradient":
        y_grad = np.linspace(0, 1, shape[0])[:, np.newaxis]
        x_grad = np.linspace(0, 1, shape[1])[np.newaxis, :]
        data = (y_grad + x_grad).astype(np.float32) / 2
    elif pattern == "gaussian":
        y = np.linspace(-1, 1, shape[0])[:, np.newaxis]
        x = np.linspace(-1, 1, shape[1])[np.newaxis, :]
        data = np.exp(-(x**2 + y**2)).astype(np.float32)
    else:
        data = np.zeros(shape, dtype=np.float32)

    return create_image(
        title=name,
        data=data,
        labels=("X", "Y", "Intensity"),
        units=("px", "px", "a.u."),
    )


def make_test_image_with_roi(
    name: str = "image_roi",
    shape: tuple[int, int] = (200, 300),
) -> ImageObj:
    """Create an image with rectangular + circular ROIs.

    Args:
        name: Image title
        shape: Image dimensions (height, width)

    Returns:
        ImageObj with mixed ROI types
    """
    data = np.random.rand(*shape).astype(np.float32) * 100
    img = create_image(
        title=name, data=data, labels=("X", "Y", "I"), units=("µm", "µm", "a.u.")
    )
    roi = ImageROI()
    roi.add_roi(
        RectangularROI([30.0, 40.0, 80.0, 60.0], indices=False, title="Region A")
    )
    roi.add_roi(CircularROI([150.0, 100.0, 40.0], indices=False, title="Spot"))
    img.roi = roi
    return img


def make_test_image_with_mask(
    name: str = "image_masked",
    shape: tuple[int, int] = (200, 200),
) -> ImageObj:
    """Create an image with a rectangular ROI that generates a mask.

    Args:
        name: Image title
        shape: Image dimensions (height, width)

    Returns:
        ImageObj with ROI (maskdata is computed automatically from ROI)
    """
    data = np.random.rand(*shape).astype(np.float32)
    img = create_image(title=name, data=data)
    img.roi = create_image_roi("rectangle", [40.0, 40.0, 120.0, 120.0])
    return img


def make_test_image_with_colormap(
    name: str = "image_colormap",
    colormap: str = "hot",
    shape: tuple[int, int] = (200, 200),
) -> ImageObj:
    """Create an image with a custom colormap via metadata.

    Args:
        name: Image title
        colormap: Matplotlib colormap name
        shape: Image dimensions (height, width)

    Returns:
        ImageObj with colormap metadata option set
    """
    y = np.linspace(-2, 2, shape[0])[:, np.newaxis]
    x = np.linspace(-2, 2, shape[1])[np.newaxis, :]
    data = np.sin(x) * np.cos(y)
    img = create_image(title=name, data=data.astype(np.float32), labels=("X", "Y", "Z"))
    img.set_metadata_option("colormap", colormap)
    return img


def make_test_image_with_lut(
    name: str = "image_lut",
    shape: tuple[int, int] = (200, 200),
    vmin: float = 0.2,
    vmax: float = 0.8,
) -> ImageObj:
    """Create an image with custom LUT (z-scale) range.

    Args:
        name: Image title
        shape: Image dimensions
        vmin: Lower bound
        vmax: Upper bound

    Returns:
        ImageObj with zscalemin/zscalemax set
    """
    data = np.random.rand(*shape).astype(np.float32)
    img = create_image(title=name, data=data)
    img.autoscale = False
    img.zscalemin = vmin
    img.zscalemax = vmax
    return img


def make_test_image_nonuniform(
    name: str = "image_nonuniform",
    shape: tuple[int, int] = (50, 80),
) -> ImageObj:
    """Create an image with non-uniform coordinates.

    X coordinates follow a logarithmic spacing; Y coordinates are linear.

    Args:
        name: Image title
        shape: Image dimensions (height, width)

    Returns:
        ImageObj with is_uniform_coords == False
    """
    data = np.random.rand(*shape).astype(np.float32)
    img = create_image(title=name, data=data)
    xcoords = np.logspace(0, 2, shape[1])
    ycoords = np.linspace(0, 10, shape[0])
    img.set_coords(xcoords, ycoords)
    return img


def make_test_image_complex(
    name: str = "image_complex",
    shape: tuple[int, int] = (128, 128),
) -> ImageObj:
    """Create a complex-valued image (FFT-like).

    Args:
        name: Image title
        shape: Image dimensions

    Returns:
        ImageObj with complex data
    """
    y = np.linspace(-2, 2, shape[0])[:, np.newaxis]
    x = np.linspace(-2, 2, shape[1])[np.newaxis, :]
    real = np.sin(x) * np.cos(y)
    imag = np.cos(x) * np.sin(y)
    data = (real + 1j * imag).astype(np.complex64)
    return create_image(title=name, data=data)


def make_test_image_rich(
    name: str = "image_rich",
    shape: tuple[int, int] = (200, 300),
) -> ImageObj:
    """Create a rich image with all features (ROI, colormap, LUT range,
    custom coordinates, geometry results).

    Args:
        name: Image title
        shape: Image dimensions

    Returns:
        Fully featured ImageObj
    """
    y = np.linspace(-2, 2, shape[0])[:, np.newaxis]
    x = np.linspace(-2, 2, shape[1])[np.newaxis, :]
    data = (np.sin(x * 3) * np.cos(y * 2) + 1).astype(np.float32) / 2

    img = create_image(
        title=name,
        data=data,
        labels=("X", "Y", "Intensity"),
        units=("µm", "µm", "counts"),
    )
    img.set_uniform_coords(dx=0.5, dy=0.5, x0=10.0, y0=20.0)

    # ROIs
    roi = ImageROI()
    roi.add_roi(
        RectangularROI([30.0, 40.0, 80.0, 60.0], indices=False, title="Region A")
    )
    roi.add_roi(CircularROI([100.0, 80.0, 25.0], indices=False, title="Spot"))
    img.roi = roi

    # LUT range
    img.autoscale = False
    img.zscalemin = 0.1
    img.zscalemax = 0.9

    # Colormap
    img.set_metadata_option("colormap", "viridis")

    # Geometry results
    geom = GeometryResult(
        title="Blob detection",
        kind=KindShape.CIRCLE,
        coords=np.array([[50.0, 60.0, 10.0], [120.0, 90.0, 15.0]]),
        roi_indices=np.array([0, 1]),
        func_name="blob_detection",
    )
    img.metadata["Geometry_blob_detection"] = geom.to_dict()

    # Table result
    table = TableResult(
        title="Image Stats",
        kind=TableKind.STATISTICS,
        headers=["Mean", "Std", "Min", "Max"],
        data=[[0.49, 0.28, 0.0, 1.0]],
        roi_indices=[NO_ROI],
        func_name="statistics",
    )
    img.metadata["Table_statistics"] = table.to_dict()

    return img


def make_geometry_result_points() -> GeometryResult:
    """Create a GeometryResult with POINT shapes."""
    return GeometryResult(
        title="Peak detection",
        kind=KindShape.POINT,
        coords=np.array([[5.0, 3.2], [7.1, 1.8], [2.3, 4.0]]),
        roi_indices=np.array([NO_ROI, NO_ROI, NO_ROI]),
        func_name="peak_detection",
    )


def make_geometry_result_circles() -> GeometryResult:
    """Create a GeometryResult with CIRCLE shapes."""
    return GeometryResult(
        title="Circle detection",
        kind=KindShape.CIRCLE,
        coords=np.array([[100.0, 100.0, 30.0], [200.0, 150.0, 20.0]]),
        roi_indices=np.array([0, 1]),
        func_name="circle_detection",
    )


def make_geometry_result_segments() -> GeometryResult:
    """Create a GeometryResult with SEGMENT shapes."""
    return GeometryResult(
        title="FWHM",
        kind=KindShape.SEGMENT,
        coords=np.array([[2.0, 0.5, 6.0, 0.5]]),
        func_name="fwhm",
    )


def make_table_result_stats() -> TableResult:
    """Create a TableResult with statistics."""
    return TableResult(
        title="Statistics",
        kind=TableKind.STATISTICS,
        headers=["Mean", "Std", "Min", "Max"],
        data=[[1.23, 0.45, -0.5, 3.1]],
        roi_indices=[NO_ROI],
        func_name="statistics",
    )


def make_table_result_multi_row() -> TableResult:
    """Create a TableResult with multiple rows (one per ROI)."""
    return TableResult(
        title="FWHM Results",
        kind=TableKind.CUSTOM,
        headers=["FWHM", "Center", "Height"],
        data=[
            [2.5, 5.0, 1.0],
            [3.1, 7.2, 0.8],
        ],
        roi_indices=[0, 1],
        func_name="fwhm",
    )
