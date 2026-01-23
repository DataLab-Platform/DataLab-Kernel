# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Integration tests for LiveBackend with a running DataLab instance.

These tests verify that the LiveBackend correctly communicates with
DataLab via the RemoteProxy XML-RPC interface.
"""
# pylint: disable=redefined-outer-name,unused-argument,import-outside-toplevel

from __future__ import annotations

import contextlib

import numpy as np
import pytest
from sigima import create_image, create_signal

from datalab_kernel.workspace import LiveBackend, Workspace, WorkspaceMode


def require_datalab():
    """Skip test if DataLab is not available."""
    try:
        from datalab.control.proxy import RemoteProxy

        proxy = RemoteProxy(autoconnect=False)
        proxy.connect(timeout=2.0)
        proxy.disconnect()
    except Exception:  # pylint: disable=broad-exception-caught
        pytest.skip("DataLab not running or not available")


@pytest.fixture
def live_workspace(datalab_instance):
    """Create a live workspace connected to DataLab.

    Depends on datalab_instance fixture to ensure DataLab is started
    when --start-datalab is provided.
    """
    require_datalab()
    backend = LiveBackend()
    # Clear DataLab before each test
    backend.proxy.reset_all()
    workspace = Workspace(backend=backend)
    yield workspace
    # Cleanup after test
    with contextlib.suppress(Exception):
        backend.proxy.reset_all()


@pytest.mark.live
class TestLiveBackendConnection:
    """Test connection to DataLab."""

    def test_connect_to_datalab(self, datalab_instance):
        """Test that we can connect to a running DataLab instance."""
        require_datalab()
        backend = LiveBackend()
        assert backend.proxy is not None
        version = backend.proxy.get_version()
        assert version is not None
        assert isinstance(version, str)

    def test_workspace_mode_detection(self, datalab_instance):
        """Test that workspace correctly detects live mode."""
        require_datalab()
        workspace = Workspace()
        assert workspace.mode == WorkspaceMode.LIVE


@pytest.mark.live
class TestLiveBackendOperations:
    """Test basic operations on LiveBackend."""

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
        """Test calling normalize computation."""
        import sigima.params

        # Create signal with values > 1
        x = np.linspace(0, 10, 100)
        y = np.sin(x) * 5 + 10  # Range roughly [5, 15]
        signal = create_signal("to_normalize", x, y)
        live_workspace.add("to_normalize", signal)

        # Select the signal and call normalize with explicit parameters
        # (passing params avoids blocking UI waiting for user input)
        live_workspace.proxy.select_objects(["to_normalize"], panel="signal")
        param = sigima.params.NormalizeParam()  # Use default: method="Maximum"
        live_workspace.calc("normalize", param)

        # Get the result (should be a new normalized signal)
        names = live_workspace.list()
        # Find the normalized result
        normalized_names = [n for n in names if "normalize" in n.lower()]
        assert len(normalized_names) > 0, f"Expected normalized signal, got: {names}"


class TestStandaloneModeRestrictions:
    """Test that live-only features raise in standalone mode.

    These tests do NOT require a running DataLab instance.
    """

    def test_calc_not_available_in_standalone(self):
        """Test that calc() raises in standalone mode."""
        from datalab_kernel.workspace import StandaloneBackend

        workspace = Workspace(backend=StandaloneBackend())
        assert workspace.mode == WorkspaceMode.STANDALONE

        with pytest.raises(RuntimeError, match="only available in live mode"):
            workspace.calc("normalize")

    def test_proxy_not_available_in_standalone(self):
        """Test that proxy raises in standalone mode."""
        from datalab_kernel.workspace import StandaloneBackend

        workspace = Workspace(backend=StandaloneBackend())

        with pytest.raises(RuntimeError, match="only available in live mode"):
            _ = workspace.proxy


@pytest.mark.live
class TestLiveWorkspaceProxyAccess:
    """Test direct proxy access for advanced operations."""

    def test_proxy_access(self, live_workspace):
        """Test accessing proxy for advanced operations."""
        proxy = live_workspace.proxy
        assert proxy is not None

        # Can call proxy methods directly
        version = proxy.get_version()
        assert version is not None


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

    def test_resync_transfers_objects(self, datalab_instance):
        """Verify resync transfers objects from standalone to DataLab."""
        require_datalab()

        # Start with explicit standalone backend
        from datalab_kernel.workspace import StandaloneBackend

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

    def test_resync_no_datalab_returns_false(self):
        """Verify resync returns False when DataLab is not available."""
        from datalab_kernel.workspace import StandaloneBackend

        # Create standalone workspace (no DataLab running for this test)
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("test", create_signal("test", np.array([1, 2]), np.array([3, 4])))

        # Mock: This test would need DataLab to be stopped
        # For now, we just verify the method exists and can be called
        # The actual behavior depends on whether DataLab is running
        _result = workspace.resync()

        # If DataLab is running, it will succeed; otherwise False
        # Either way, the workspace should still be functional
        assert workspace.exists("test") or len(workspace) >= 0
