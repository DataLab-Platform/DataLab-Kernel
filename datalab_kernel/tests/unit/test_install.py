# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for kernel installation
==================================

Tests kernel install/uninstall functionality.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import tempfile


class TestKernelInstall:
    """Tests for kernel installation."""

    def test_kernel_install_creates_spec(self):
        """Verify install creates kernel.json."""
        from datalab_kernel.install import install_kernel

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_dir = install_kernel(prefix=tmpdir)

            kernel_json = kernel_dir / "kernel.json"
            assert kernel_json.exists()

    def test_kernel_uninstall_removes_spec(self):
        """Verify uninstall removes kernel directory."""
        from datalab_kernel.install import install_kernel, uninstall_kernel

        with tempfile.TemporaryDirectory() as tmpdir:
            kernel_dir = install_kernel(prefix=tmpdir)
            assert kernel_dir.exists()

            result = uninstall_kernel(prefix=tmpdir)
            assert result is True
            assert not kernel_dir.exists()

    def test_kernel_uninstall_missing_returns_false(self):
        """Verify uninstall of non-existent kernel returns False."""
        from datalab_kernel.install import uninstall_kernel

        with tempfile.TemporaryDirectory() as tmpdir:
            result = uninstall_kernel(prefix=tmpdir)
            assert result is False
