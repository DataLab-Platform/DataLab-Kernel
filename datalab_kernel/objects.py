# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Data objects for DataLab Kernel
===============================

This module re-exports signal and image objects from Sigima.
Sigima is a required dependency for the DataLab Kernel since all
data processing relies on it.

Usage:
    from datalab_kernel.objects import SignalObj, ImageObj, create_signal, create_image
"""

from sigima import ImageObj, SignalObj, create_image, create_signal

__all__ = ["SignalObj", "ImageObj", "create_signal", "create_image"]
