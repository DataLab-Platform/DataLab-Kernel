# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Integration tests for live DataLab connection.

These tests verify that the workspace correctly communicates with DataLab
via the Web API.
"""

from __future__ import annotations

import os

import httpx
import numpy as np
import pytest
import sigima.params
from sigima import create_image, create_signal

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.backends.webapi import WebApiBackend
from datalab_kernel.workspace import Workspace, WorkspaceMode


def require_datalab():
    """Skip test if DataLab WebAPI is not running.

    Checks Web API connectivity as the indicator that DataLab is running.
    """
    url = os.environ.get("DATALAB_WORKSPACE_URL", "http://127.0.0.1:18080")
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{url}/api/v1/status")
            if response.status_code != 200:
                pytest.skip("DataLab WebAPI not responding correctly")
    except Exception:  # pylint: disable=broad-exception-caught
        pytest.skip("DataLab WebAPI not running or not available")


def is_webapi_backend(workspace: Workspace) -> bool:
    """Check if workspace is using WebApiBackend."""
    # pylint: disable=protected-access
    return isinstance(workspace._backend, WebApiBackend)


@pytest.mark.live
class TestLiveConnection:
    """Test connection to DataLab."""

    def test_connect_to_datalab(self):
        """Test that we can connect to a running DataLab instance."""
        require_datalab()
        workspace = Workspace()
        assert workspace.mode == WorkspaceMode.LIVE
        # Verify we can call a method
        status = workspace.status()
        assert status["mode"] == "live"
        # Should be using WebApiBackend when started with --start-datalab
        assert "WebApiBackend" in status.get("backend", "")

    def test_workspace_mode_detection(self):
        """Test that workspace correctly detects live mode."""
        require_datalab()
        workspace = Workspace()
        assert workspace.mode == WorkspaceMode.LIVE


@pytest.mark.live
class TestLiveOperations:
    """Test basic operations on live workspace."""

    def test_list_empty(self, live_workspace):
        """Test listing objects in empty workspace."""
        names = live_workspace.list()
        assert names == []

    def test_add_and_get_signal(self, live_workspace):
        """Test adding and retrieving a signal."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        signal = create_signal("test_signal", x, y)

        live_workspace.add("test_signal", signal)

        # Verify it's in the list
        names = live_workspace.list()
        assert "test_signal" in names

        # Retrieve it
        retrieved = live_workspace.get("test_signal")
        assert retrieved is not None
        assert retrieved.title == "test_signal"
        np.testing.assert_array_almost_equal(retrieved.x, x, decimal=5)
        np.testing.assert_array_almost_equal(retrieved.y, y, decimal=5)

    def test_add_and_get_image(self, live_workspace):
        """Test adding and retrieving an image."""
        data = np.random.rand(50, 50).astype(np.float64)
        image = create_image("test_image", data)

        live_workspace.add("test_image", image)

        # Verify it's in the list
        names = live_workspace.list()
        assert "test_image" in names

        # Retrieve it
        retrieved = live_workspace.get("test_image")
        assert retrieved is not None
        assert retrieved.title == "test_image"
        np.testing.assert_array_almost_equal(retrieved.data, data, decimal=5)

    def test_exists(self, live_workspace):
        """Test existence check."""
        assert not live_workspace.exists("nonexistent")

        x = np.linspace(0, 10, 50)
        y = np.cos(x)
        signal = create_signal("exists_test", x, y)
        live_workspace.add("exists_test", signal)

        assert live_workspace.exists("exists_test")
        assert not live_workspace.exists("nonexistent")

    def test_remove(self, live_workspace):
        """Test removing an object."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        signal = create_signal("to_remove", x, y)
        live_workspace.add("to_remove", signal)

        assert live_workspace.exists("to_remove")

        try:
            live_workspace.remove("to_remove")
        except NotImplementedError:
            pytest.skip("Individual object removal requires DataLab 1.1+")

        assert not live_workspace.exists("to_remove")

    def test_remove_nonexistent_raises(self, live_workspace):
        """Test that removing nonexistent object raises KeyError."""
        with pytest.raises(KeyError):
            live_workspace.remove("nonexistent")

    def test_overwrite(self, live_workspace):
        """Test overwriting an existing object."""
        x = np.linspace(0, 10, 50)
        y1 = np.sin(x)
        y2 = np.cos(x)

        signal1 = create_signal("overwrite_test", x, y1)
        signal2 = create_signal("overwrite_test", x, y2)

        live_workspace.add("overwrite_test", signal1)

        # Should raise without overwrite flag
        with pytest.raises(ValueError):
            live_workspace.add("overwrite_test", signal2)

        # Should succeed with overwrite flag (requires remove support)
        try:
            live_workspace.add("overwrite_test", signal2, overwrite=True)
        except NotImplementedError:
            pytest.skip("Overwrite requires DataLab 1.1+ (uses remove internally)")

        retrieved = live_workspace.get("overwrite_test")
        np.testing.assert_array_almost_equal(retrieved.y, y2, decimal=5)

    def test_rename(self, live_workspace):
        """Test renaming an object."""
        x = np.linspace(0, 10, 50)
        y = np.sin(x)
        signal = create_signal("old_name", x, y)
        live_workspace.add("old_name", signal)

        try:
            live_workspace.rename("old_name", "new_name")
        except NotImplementedError:
            pytest.skip("Rename requires DataLab 1.1+ (uses remove internally)")

        assert not live_workspace.exists("old_name")
        assert live_workspace.exists("new_name")

    def test_clear(self, live_workspace):
        """Test clearing all objects."""
        # Add some objects
        x = np.linspace(0, 10, 50)
        signal = create_signal("signal1", x, np.sin(x))
        image = create_image("image1", np.random.rand(30, 30))

        live_workspace.add("signal1", signal)
        live_workspace.add("image1", image)

        assert len(live_workspace) >= 2

        live_workspace.clear()

        assert len(live_workspace) == 0


@pytest.mark.live
class TestLiveWorkspacePersistence:
    """Test workspace save/load with DataLab."""

    def test_save_and_load(self, live_workspace, tmp_path):
        """Test saving and loading workspace."""
        # Create and add signal
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        signal = create_signal("persist_signal", x, y)
        live_workspace.add("persist_signal", signal)

        # Save
        filepath = str(tmp_path / "test_workspace.h5")
        live_workspace.save(filepath)

        # Clear and reload
        live_workspace.clear()
        assert not live_workspace.exists("persist_signal")

        live_workspace.load(filepath)

        # Verify the signal is back
        assert live_workspace.exists("persist_signal")
        retrieved = live_workspace.get("persist_signal")
        np.testing.assert_array_almost_equal(retrieved.x, x, decimal=5)
        np.testing.assert_array_almost_equal(retrieved.y, y, decimal=5)


@pytest.mark.live
class TestLiveWorkspaceCalc:
    """Test computation features via DataLab."""

    def test_calc_normalize(self, live_workspace):
        """Test calling normalize computation.

        This test uses the workspace.select_objects() and workspace.calc()
        methods to interact with DataLab via the Web API.
        """
        # Create signal with values > 1
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 5 + 10  # Range roughly [5, 15]
        signal = create_signal("to_normalize", x, y)
        live_workspace.add("to_normalize", signal)

        # Select the signal and call normalize with explicit parameters
        # (passing params avoids blocking UI waiting for user input)
        live_workspace.select_objects(["to_normalize"], panel="signal")
        param = sigima.params.NormalizeParam()  # Use default: method="Maximum"
        live_workspace.calc("normalize", param)

        # Get the result (should be a new normalized signal)
        names = live_workspace.list()
        # Find the normalized result
        normalized_names = [n for n in names if "normalize" in n.lower()]
        assert len(normalized_names) > 0, f"Expected normalized signal, got: {names}"

    def test_calc_with_dict_params(self, live_workspace):
        """Test calling computation with dict parameters."""
        # Create signal
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 5 + 10
        signal = create_signal("to_normalize_dict", x, y)
        live_workspace.add("to_normalize_dict", signal)

        # Select and call with dict params
        live_workspace.select_objects(["to_normalize_dict"])
        live_workspace.calc("normalize", {"method": "Maximum"})

        # Verify result exists
        names = live_workspace.list()
        assert len(names) >= 2, f"Expected at least 2 objects, got: {names}"


class TestStandaloneModeRestrictions:
    """Test that live-only features raise in standalone mode.

    These tests do NOT require a running DataLab instance.
    """

    def test_calc_not_available_in_standalone(self):
        """Test that calc() raises in standalone mode."""
        workspace = Workspace(backend=StandaloneBackend())
        assert workspace.mode == WorkspaceMode.STANDALONE

        with pytest.raises(RuntimeError, match="only available in live mode"):
            workspace.calc("normalize")

    def test_select_objects_not_available_in_standalone(self):
        """Test that select_objects() raises in standalone mode."""
        workspace = Workspace(backend=StandaloneBackend())
        assert workspace.mode == WorkspaceMode.STANDALONE

        with pytest.raises(RuntimeError, match="only available in live mode"):
            workspace.select_objects(["test"])


@pytest.mark.live
class TestLiveWorkspaceDunderMethods:
    """Test dunder method support in live mode."""

    def test_len(self, live_workspace):
        """Test __len__ support."""
        assert len(live_workspace) == 0

        x = np.linspace(0, 10, 50)
        signal = create_signal("signal1", x, np.sin(x))
        live_workspace.add("signal1", signal)

        assert len(live_workspace) == 1

    def test_contains(self, live_workspace):
        """Test __contains__ support."""
        assert "test" not in live_workspace

        x = np.linspace(0, 10, 50)
        signal = create_signal("test", x, np.sin(x))
        live_workspace.add("test", signal)

        assert "test" in live_workspace

    def test_iter(self, live_workspace):
        """Test __iter__ support."""
        x = np.linspace(0, 10, 50)
        live_workspace.add("s1", create_signal("s1", x, np.sin(x)))
        live_workspace.add("s2", create_signal("s2", x, np.cos(x)))

        names = list(live_workspace)
        assert "s1" in names
        assert "s2" in names


@pytest.mark.live
class TestWorkspaceResync:
    """Test workspace resync from standalone to live mode."""

    def test_resync_transfers_objects(self):
        """Verify resync transfers objects from standalone to DataLab."""
        require_datalab()

        # Start with explicit standalone backend
        workspace = Workspace(backend=StandaloneBackend())
        assert workspace.mode == WorkspaceMode.STANDALONE

        # Add objects in standalone mode
        x = np.linspace(0, 10, 50)
        signal = create_signal("standalone_signal", x, np.sin(x))
        workspace.add("standalone_signal", signal)

        assert workspace.exists("standalone_signal")
        assert len(workspace) == 1

        # Resync to DataLab
        result = workspace.resync()

        assert result is True
        assert workspace.mode == WorkspaceMode.LIVE

        # Object should now be in DataLab
        assert workspace.exists("standalone_signal")
        retrieved = workspace.get("standalone_signal")
        np.testing.assert_array_almost_equal(retrieved.y, np.sin(x), decimal=5)

        # Cleanup
        workspace.clear()

    def test_resync_already_live_returns_false(self, live_workspace):
        """Verify resync returns False when already in live mode."""
        assert live_workspace.mode == WorkspaceMode.LIVE

        result = live_workspace.resync()

        assert result is False
        assert live_workspace.mode == WorkspaceMode.LIVE


@pytest.mark.standalone
class TestWorkspaceResyncStandalone:
    """Test workspace resync in standalone mode."""

    def test_resync_no_datalab_returns_false(self):
        """Verify resync returns False when DataLab is not available.

        This test should only run in standalone mode (without --live flag).
        """
        # Create standalone workspace (no DataLab running for this test)
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("test", create_signal("test", np.array([1, 2]), np.array([3, 4])))

        # Attempt resync when DataLab is not available
        result = workspace.resync()

        # Should return False when DataLab is not available
        assert result is False
        assert workspace.mode == WorkspaceMode.STANDALONE
        assert workspace.exists("test")
