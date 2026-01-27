# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Contract tests for API guarantees
=================================

Tests validating the user-facing guarantees from specification.md.
These tests verify that the same code runs in both modes.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from sigima import create_image, create_signal

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.plotter import Plotter
from datalab_kernel.workspace import Workspace

# Canonical notebook code that must work in both modes
CANONICAL_CODE = """
import numpy as np

# Create and add signal
x = np.linspace(0, 10, 100)
y = np.sin(x)
sig = create_signal("test_signal", x, y)
workspace.add("my_signal", sig)

# Create and add image
data = np.random.rand(64, 64).astype(np.float32)
img = create_image("test_image", data)
workspace.add("my_image", img)

# Verify objects exist
assert workspace.exists("my_signal")
assert workspace.exists("my_image")
assert len(workspace.list()) == 2

# Retrieve and verify
retrieved_sig = workspace.get("my_signal")
assert hasattr(retrieved_sig, 'x')
assert hasattr(retrieved_sig, 'y')

retrieved_img = workspace.get("my_image")
assert hasattr(retrieved_img, 'data')
"""


@pytest.mark.contract
class TestAPIComplete:
    """Tests that all documented API methods exist."""

    def test_workspace_api_complete(self):
        """All documented workspace methods exist."""
        workspace = Workspace(backend=StandaloneBackend())

        # Required methods from spec
        assert callable(workspace.list)
        assert callable(workspace.get)
        assert callable(workspace.add)
        assert callable(workspace.remove)
        assert callable(workspace.rename)
        assert callable(workspace.exists)
        assert callable(workspace.save)
        assert callable(workspace.load)
        assert callable(workspace.clear)

    def test_plotter_api_complete(self):
        """All documented plotter methods exist."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace)

        # Required methods from spec
        assert callable(plotter.plot)

    def test_workspace_mode_attribute(self):
        """Workspace has mode attribute."""
        workspace = Workspace(backend=StandaloneBackend())
        assert hasattr(workspace, "mode")


@pytest.mark.contract
class TestModeTransparency:
    """Tests that code runs transparently in standalone mode."""

    def test_canonical_code_standalone(self):
        """Run canonical code in standalone mode."""
        workspace = Workspace(backend=StandaloneBackend())
        plotter = Plotter(workspace)

        # Execute canonical code
        # pylint: disable=exec-used
        exec(
            CANONICAL_CODE,
            {
                "workspace": workspace,
                "plotter": plotter,
                "create_signal": create_signal,
                "create_image": create_image,
                "np": np,
            },
        )

        # Verify results
        assert len(workspace) == 2
        assert "my_signal" in workspace
        assert "my_image" in workspace

    def test_no_mode_specific_exceptions(self):
        """Standalone mode doesn't raise mode-specific exceptions."""
        workspace = Workspace(backend=StandaloneBackend())

        # These should all work without mode-related errors
        workspace.add("test", create_signal("test", np.array([1, 2]), np.array([3, 4])))
        workspace.get("test")
        workspace.exists("test")
        workspace.rename("test", "test2")
        workspace.remove("test2")


@pytest.mark.contract
class TestErrorMessages:
    """Tests for user-friendly error messages."""

    def test_missing_object_error_message(self):
        """KeyError message lists available objects."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("sig1", create_signal("sig1", np.array([1]), np.array([2])))
        workspace.add("sig2", create_signal("sig2", np.array([1]), np.array([2])))

        with pytest.raises(KeyError) as exc_info:
            workspace.get("unknown")

        error_msg = str(exc_info.value)
        # Error should mention available objects
        assert "sig1" in error_msg or "sig2" in error_msg or "Available" in error_msg

    def test_error_messages_user_friendly(self):
        """Errors don't expose internal implementation details."""
        workspace = Workspace(backend=StandaloneBackend())

        with pytest.raises(KeyError) as exc_info:
            workspace.get("unknown")

        error_msg = str(exc_info.value)
        # Should not contain internal class names
        assert "StandaloneBackend" not in error_msg
        assert "_objects" not in error_msg


@pytest.mark.contract
class TestH5Reproducibility:
    """Tests for HDF5-based reproducibility."""

    def test_h5_reproducibility(self):
        """Saved .h5 files ensure full reproducibility."""
        # Create and populate workspace
        workspace1 = Workspace(backend=StandaloneBackend())
        sig = create_signal(
            "signal", np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100))
        )
        img = create_image("image", np.random.rand(64, 64).astype(np.float32))
        workspace1.add("signal", sig)
        workspace1.add("image", img)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "analysis.h5")

            # Save
            workspace1.save(filepath)

            # Load into new workspace
            workspace2 = Workspace(backend=StandaloneBackend())
            workspace2.load(filepath)

            # Verify reproducibility
            assert len(workspace2) == len(workspace1)
            assert set(workspace2.list()) == set(workspace1.list())

            # Verify data integrity
            sig1 = workspace1.get("signal")
            sig2 = workspace2.get("signal")
            np.testing.assert_array_almost_equal(sig1.x, sig2.x)
            np.testing.assert_array_almost_equal(sig1.y, sig2.y)


@pytest.mark.contract
class TestDataIntegrity:
    """Tests for data integrity guarantees."""

    def test_add_creates_copy(self):
        """Adding an object creates an independent copy."""
        workspace = Workspace(backend=StandaloneBackend())
        original = create_signal(
            "test", np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0])
        )

        workspace.add("test", original)

        # Modify original
        original.y[0] = 999.0

        # Workspace copy should be unaffected
        retrieved = workspace.get("test")
        assert retrieved.y[0] != 999.0
