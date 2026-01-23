# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
DataLab Kernel Tests
====================

Test suite for the DataLab Jupyter kernel.
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "standalone: tests for standalone mode only")
    config.addinivalue_line("markers", "live: tests requiring live DataLab instance")
    config.addinivalue_line("markers", "contract: contract tests for both modes")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run tests requiring a live DataLab instance",
    )


def pytest_collection_modifyitems(config, items):
    """Skip live tests unless --live flag is provided."""
    if not config.getoption("--live"):
        skip_live = pytest.mark.skip(reason="Need --live option to run")
        for item in items:
            if "live" in item.keywords:
                item.add_marker(skip_live)
