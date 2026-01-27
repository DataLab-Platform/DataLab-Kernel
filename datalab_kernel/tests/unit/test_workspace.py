# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Unit tests for Workspace API (standalone backend)
==================================================

Tests all workspace operations in standalone mode.
"""

from __future__ import annotations

import pytest

from datalab_kernel.backends import StandaloneBackend
from datalab_kernel.tests.data import make_test_image, make_test_signal
from datalab_kernel.workspace import Workspace, WorkspaceMode


class TestWorkspaceBasic:
    """Basic workspace operations."""

    def test_workspace_creation(self):
        """Verify workspace can be created."""
        workspace = Workspace(backend=StandaloneBackend())
        assert workspace is not None
        assert workspace.mode == WorkspaceMode.STANDALONE

    def test_workspace_list_empty(self):
        """workspace.list() returns empty list on fresh workspace."""
        workspace = Workspace(backend=StandaloneBackend())
        assert workspace.list() == []

    def test_workspace_len_empty(self):
        """len(workspace) returns 0 for empty workspace."""
        workspace = Workspace(backend=StandaloneBackend())
        assert len(workspace) == 0

    def test_workspace_repr_empty(self):
        """Verify workspace repr for empty workspace."""
        workspace = Workspace(backend=StandaloneBackend())
        repr_str = repr(workspace)
        assert "standalone" in repr_str
        assert "empty" in repr_str


class TestWorkspaceAdd:
    """Tests for adding objects to workspace."""

    def test_workspace_add_signal(self):
        """Add a SignalObj, verify it appears in list()."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")

        workspace.add("my_signal", signal)

        assert "my_signal" in workspace.list()
        assert len(workspace) == 1

    def test_workspace_add_image(self):
        """Add an ImageObj, verify it appears in list()."""
        workspace = Workspace(backend=StandaloneBackend())
        image = make_test_image("my_image")

        workspace.add("my_image", image)

        assert "my_image" in workspace.list()
        assert len(workspace) == 1

    def test_workspace_add_returns_object(self):
        """Verify add() returns the added object."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")

        result = workspace.add("my_signal", signal)

        assert result is not None
        assert hasattr(result, "x")
        assert hasattr(result, "y")

    def test_workspace_add_overwrite_false(self):
        """Adding duplicate name without overwrite=True raises."""
        workspace = Workspace(backend=StandaloneBackend())
        signal1 = make_test_signal("sig1")
        signal2 = make_test_signal("sig2")

        workspace.add("my_signal", signal1)

        with pytest.raises(ValueError, match="already exists"):
            workspace.add("my_signal", signal2)

    def test_workspace_add_overwrite_true(self):
        """Adding duplicate name with overwrite=True replaces."""
        workspace = Workspace(backend=StandaloneBackend())
        signal1 = make_test_signal("sig1")
        signal2 = make_test_signal("sig2")

        workspace.add("my_signal", signal1)
        workspace.add("my_signal", signal2, overwrite=True)

        assert len(workspace) == 1
        assert "my_signal" in workspace.list()


class TestWorkspaceGet:
    """Tests for retrieving objects from workspace."""

    def test_workspace_get_by_name(self):
        """Retrieve object by name with workspace.get()."""
        workspace = Workspace(backend=StandaloneBackend())
        signal = make_test_signal("my_signal")
        workspace.add("my_signal", signal)

        result = workspace.get("my_signal")

        assert result is not None
        assert hasattr(result, "x")
        assert hasattr(result, "y")

    def test_workspace_get_missing_raises(self):
        """workspace.get('unknown') raises KeyError."""
        workspace = Workspace(backend=StandaloneBackend())

        with pytest.raises(KeyError, match="not found"):
            workspace.get("unknown")

    def test_workspace_get_error_lists_available(self):
        """KeyError message lists available objects."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("sig1", make_test_signal("sig1"))
        workspace.add("sig2", make_test_signal("sig2"))

        with pytest.raises(KeyError) as exc_info:
            workspace.get("unknown")

        error_msg = str(exc_info.value)
        assert "sig1" in error_msg or "sig2" in error_msg


class TestWorkspaceExists:
    """Tests for checking object existence."""

    def test_workspace_exists_true(self):
        """workspace.exists() returns True for existing object."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal())

        assert workspace.exists("my_signal") is True

    def test_workspace_exists_false(self):
        """workspace.exists() returns False for missing object."""
        workspace = Workspace(backend=StandaloneBackend())

        assert workspace.exists("unknown") is False

    def test_workspace_contains(self):
        """Verify 'in' operator works."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal())

        assert "my_signal" in workspace
        assert "unknown" not in workspace


class TestWorkspaceRemove:
    """Tests for removing objects from workspace."""

    def test_workspace_remove(self):
        """Remove object, verify it disappears from list()."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("my_signal", make_test_signal())

        workspace.remove("my_signal")

        assert "my_signal" not in workspace.list()
        assert len(workspace) == 0

    def test_workspace_remove_missing_raises(self):
        """Removing non-existent object raises error."""
        workspace = Workspace(backend=StandaloneBackend())

        with pytest.raises(KeyError, match="not found"):
            workspace.remove("unknown")


class TestWorkspaceRename:
    """Tests for renaming objects."""

    def test_workspace_rename(self):
        """Rename object, verify old name gone, new name exists."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("old_name", make_test_signal())

        workspace.rename("old_name", "new_name")

        assert "old_name" not in workspace
        assert "new_name" in workspace
        assert len(workspace) == 1

    def test_workspace_rename_missing_raises(self):
        """Renaming non-existent object raises KeyError."""
        workspace = Workspace(backend=StandaloneBackend())

        with pytest.raises(KeyError):
            workspace.rename("unknown", "new_name")

    def test_workspace_rename_to_existing_raises(self):
        """Renaming to existing name raises ValueError."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("obj1", make_test_signal("obj1"))
        workspace.add("obj2", make_test_signal("obj2"))

        with pytest.raises(ValueError, match="already exists"):
            workspace.rename("obj1", "obj2")


class TestWorkspaceIteration:
    """Tests for workspace iteration."""

    def test_workspace_iteration(self):
        """Verify workspace is iterable (for name in workspace)."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("sig1", make_test_signal("sig1"))
        workspace.add("sig2", make_test_signal("sig2"))

        names = list(workspace)

        assert len(names) == 2
        assert "sig1" in names
        assert "sig2" in names

    def test_workspace_len(self):
        """Verify len(workspace) returns object count."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("sig1", make_test_signal("sig1"))
        workspace.add("sig2", make_test_signal("sig2"))
        workspace.add("img1", make_test_image("img1"))

        assert len(workspace) == 3


class TestWorkspaceClear:
    """Tests for clearing workspace."""

    def test_workspace_clear(self):
        """Verify clear() removes all objects."""
        workspace = Workspace(backend=StandaloneBackend())
        workspace.add("sig1", make_test_signal("sig1"))
        workspace.add("sig2", make_test_signal("sig2"))

        workspace.clear()

        assert len(workspace) == 0
        assert workspace.list() == []
