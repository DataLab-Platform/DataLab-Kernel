# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
HDF5 Image Coordinates
======================

Storage of the pixel coordinates of images in the HDF5 workspace files written by
the backends.

An image group holds:

- for uniform coordinates, the ``x0``, ``y0``, ``dx`` and ``dy`` attributes;
- for non-uniform coordinates, the ``xcoords`` and ``ycoords`` datasets (pixel
  center coordinates along X and Y).

The ``is_uniform_coords`` attribute tells which one applies. Files written by
earlier versions (without this attribute) have uniform coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sigima import ImageObj

UNIFORM_ATTRS = ("x0", "y0", "dx", "dy")


def save_image_coords(grp: Any, obj: ImageObj) -> None:
    """Save the pixel coordinates of an image to an HDF5 group.

    Args:
        grp: HDF5 group of the image.
        obj: Image.
    """
    grp.attrs["is_uniform_coords"] = bool(obj.is_uniform_coords)
    if obj.is_uniform_coords:
        for attr in UNIFORM_ATTRS:
            value = getattr(obj, attr)
            if value is not None:
                grp.attrs[attr] = value
    else:
        grp.create_dataset("xcoords", data=np.asarray(obj.xcoords, dtype=float))
        grp.create_dataset("ycoords", data=np.asarray(obj.ycoords, dtype=float))


def load_image_coords(grp: Any, obj: ImageObj) -> None:
    """Restore the pixel coordinates of an image from an HDF5 group.

    Args:
        grp: HDF5 group of the image.
        obj: Image, whose data is already set.

    Raises:
        ValueError: If non-uniform coordinates are missing or do not match the
         shape of the image.
    """
    if bool(grp.attrs.get("is_uniform_coords", True)):
        for attr in UNIFORM_ATTRS:
            if attr in grp.attrs:
                setattr(obj, attr, float(grp.attrs[attr]))
        return
    if "xcoords" not in grp or "ycoords" not in grp:
        raise ValueError(
            "Invalid image group: missing xcoords or ycoords "
            "for non-uniform coordinates"
        )
    xcoords = np.array(grp["xcoords"], dtype=float)
    ycoords = np.array(grp["ycoords"], dtype=float)
    nrows, ncols = obj.data.shape
    if xcoords.shape != (ncols,) or ycoords.shape != (nrows,):
        raise ValueError(
            f"Invalid image group: coordinates of shape {xcoords.shape} (X) and "
            f"{ycoords.shape} (Y) do not match image shape {obj.data.shape}"
        )
    obj.set_coords(xcoords, ycoords)
