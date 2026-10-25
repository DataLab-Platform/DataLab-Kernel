# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
NPZ Serialization
=================

Serialization of SignalObj and ImageObj to/from NumPy NPZ format, used by the
DataLab Web API for efficient binary data transfer.

The format is implemented by Sigima (:mod:`sigima.io.npz`, where it is
described), so that DataLab and DataLab-Kernel share one implementation:
:func:`serialize_object_to_npz` and :func:`deserialize_object_from_npz` are those
of :mod:`sigima.io`.
"""

from __future__ import annotations

from sigima.io import deserialize_object_from_npz, serialize_object_to_npz

__all__ = ["deserialize_object_from_npz", "serialize_object_to_npz"]
