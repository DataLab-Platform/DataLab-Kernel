# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Test data generation utilities
==============================

Provides functions to create test signals and images.
"""

from __future__ import annotations

import numpy as np
from sigima import ImageObj, SignalObj, create_image, create_signal


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
