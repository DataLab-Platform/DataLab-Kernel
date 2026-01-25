# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Workspace Backends
==================

Backend implementations for the DataLab Kernel workspace.

Available backends:

- :class:`StandaloneBackend`: Local memory storage with HDF5 persistence
- :class:`LiveBackend`: XML-RPC synchronization with DataLab (legacy)
- :class:`WebApiBackend`: HTTP/JSON + NPZ synchronization with DataLab (recommended)
"""

from __future__ import annotations

__all__ = [
    "StandaloneBackend",
    "WebApiBackend",
    "WorkspaceBackend",
]

# Re-export base classes from workspace module for convenience
from datalab_kernel.workspace import StandaloneBackend, WorkspaceBackend

# Import WebAPI backend (handles missing dependencies gracefully)
try:
    from datalab_kernel.backends.webapi import WebApiBackend
except ImportError:
    # httpx not installed, WebApiBackend not available
    WebApiBackend = None  # type: ignore
