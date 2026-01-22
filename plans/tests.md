# DataLab Jupyter Kernel — Test Plans

This document specifies the test strategy and test cases for validating the **DataLab Jupyter kernel** (`kernel-datalab`). Tests are organized by category and execution context.

---

## 1. Test Strategy Overview

### 1.1 Testing Principles

The kernel must be tested in **two execution modes**:

| Mode | Description | DataLab Required |
|------|-------------|------------------|
| **Standalone** | Kernel runs independently, no DataLab instance | ❌ |
| **Live** | Kernel attached to a running DataLab instance | ✅ |

**Key invariant**: The same user code must produce equivalent results in both modes.

### 1.2 Test Categories

| Category | Purpose | Requires DataLab |
|----------|---------|------------------|
| **Unit Tests** | Test individual components in isolation | ❌ |
| **Integration Tests** | Test kernel ↔ DataLab communication | ✅ |
| **Contract Tests** | Validate user-facing API guarantees | Both modes |
| **Regression Tests** | Prevent regressions on known issues | Both modes |

### 1.3 Test Infrastructure

Tests requiring a live DataLab instance follow the pattern established in
`datalab/tests/features/control/simpleclient_unit_test.py`:

```python
from datalab.tests import run_datalab_in_background, close_datalab_background

def test_with_live_datalab():
    """Test requiring a live DataLab instance."""
    run_datalab_in_background(wait_until_ready=True)
    try:
        # ... test code using kernel in live mode ...
        pass
    finally:
        close_datalab_background()
```

---

## 2. Unit Tests (Standalone Mode)

These tests validate kernel components **without requiring DataLab**.

### 2.1 Kernel Registration and Discovery

| Test | Description |
|------|-------------|
| `test_kernel_spec_valid` | Verify `kernel.json` is valid and contains required fields |
| `test_kernel_discoverable` | Verify kernel appears in `jupyter kernelspec list` |
| `test_kernel_install_command` | Verify `python -m datalab_kernel install` works |
| `test_kernel_uninstall_command` | Verify `python -m datalab_kernel uninstall` works |

### 2.2 Kernel Startup and Namespace

| Test | Description |
|------|-------------|
| `test_kernel_starts` | Verify kernel starts without errors |
| `test_namespace_workspace_available` | Verify `workspace` is available at startup |
| `test_namespace_plotter_available` | Verify `plotter` is available at startup |
| `test_namespace_sigima_available` | Verify `sigima` module is importable |
| `test_namespace_numpy_available` | Verify `np` (NumPy) is available |
| `test_mode_detection_standalone` | Verify kernel detects standalone mode when no DataLab |

### 2.3 Workspace API (Standalone Backend)

| Test | Description |
|------|-------------|
| `test_workspace_list_empty` | `workspace.list()` returns empty list on fresh kernel |
| `test_workspace_add_signal` | Add a `SignalObj`, verify it appears in `list()` |
| `test_workspace_add_image` | Add an `ImageObj`, verify it appears in `list()` |
| `test_workspace_get_by_name` | Retrieve object by name with `workspace.get()` |
| `test_workspace_get_missing_raises` | `workspace.get("unknown")` raises `KeyError` |
| `test_workspace_exists_true` | `workspace.exists()` returns `True` for existing object |
| `test_workspace_exists_false` | `workspace.exists()` returns `False` for missing object |
| `test_workspace_remove` | Remove object, verify it disappears from `list()` |
| `test_workspace_remove_missing_raises` | Removing non-existent object raises error |
| `test_workspace_rename` | Rename object, verify old name gone, new name exists |
| `test_workspace_add_overwrite_false` | Adding duplicate name without `overwrite=True` raises |
| `test_workspace_add_overwrite_true` | Adding duplicate name with `overwrite=True` replaces |
| `test_workspace_iteration` | Verify workspace is iterable (for obj in workspace) |
| `test_workspace_len` | Verify `len(workspace)` returns object count |

### 2.4 Workspace Persistence

| Test | Description |
|------|-------------|
| `test_workspace_save_h5` | `workspace.save("file.h5")` creates valid HDF5 file |
| `test_workspace_load_h5` | `workspace.load("file.h5")` restores objects |
| `test_workspace_save_load_roundtrip` | Save then load preserves all objects and metadata |
| `test_workspace_h5_format_compatible` | Saved `.h5` file is readable by DataLab |
| `test_workspace_load_datalab_h5` | Kernel can load `.h5` files saved by DataLab |

### 2.5 Plotter API (Standalone Backend)

| Test | Description |
|------|-------------|
| `test_plotter_plot_signal_returns_html` | `plotter.plot(signal)` returns displayable output |
| `test_plotter_plot_image_returns_html` | `plotter.plot(image)` returns displayable output |
| `test_plotter_plot_by_name` | `plotter.plot("object_name")` works |
| `test_plotter_plot_missing_raises` | `plotter.plot("unknown")` raises `KeyError` |
| `test_signal_repr_html` | `SignalObj._repr_html_()` returns valid HTML |
| `test_image_repr_html` | `ImageObj._repr_html_()` returns valid HTML |
| `test_signal_repr_png` | `SignalObj._repr_png_()` returns valid PNG bytes |
| `test_image_repr_png` | `ImageObj._repr_png_()` returns valid PNG bytes |

### 2.6 Sigima Integration

| Test | Description |
|------|-------------|
| `test_sigima_process_signal` | Apply `sigima.proc.signal` function, verify result |
| `test_sigima_process_image` | Apply `sigima.proc.image` function, verify result |
| `test_sigima_result_addable` | Sigima processing result can be added to workspace |
| `test_sigima_param_creation` | `sigima.params` classes are accessible |

---

## 3. Integration Tests (Live Mode)

These tests validate kernel ↔ DataLab communication and require a running DataLab instance.

### 3.1 Test Setup Pattern

All live mode tests use the DataLab background execution pattern:

```python
import pytest
from datalab.tests import run_datalab_in_background, close_datalab_background

@pytest.fixture(scope="module")
def live_datalab():
    """Fixture providing a live DataLab instance."""
    run_datalab_in_background(wait_until_ready=True)
    yield
    close_datalab_background()

def test_example_with_live_datalab(live_datalab):
    """Test requiring live DataLab."""
    # Kernel should detect live mode automatically
    pass
```

### 3.2 Mode Detection

| Test | Description |
|------|-------------|
| `test_mode_detection_live` | Kernel detects live mode when DataLab is running |
| `test_mode_fallback_to_standalone` | Kernel falls back to standalone if DataLab unreachable |
| `test_mode_attribute_readable` | `workspace.mode` or similar attribute indicates mode |

### 3.3 Workspace Synchronization (Notebook → DataLab)

| Test | Description |
|------|-------------|
| `test_add_signal_appears_in_datalab` | Signal added via kernel appears in DataLab panel |
| `test_add_image_appears_in_datalab` | Image added via kernel appears in DataLab panel |
| `test_remove_object_removed_in_datalab` | Object removed via kernel disappears from DataLab |
| `test_rename_object_renamed_in_datalab` | Object renamed via kernel updated in DataLab |
| `test_metadata_synced_to_datalab` | Object metadata (units, labels) synced to DataLab |

### 3.4 Workspace Synchronization (DataLab → Notebook)

| Test | Description |
|------|-------------|
| `test_datalab_object_visible_in_kernel` | Object added in DataLab GUI visible via `workspace.list()` |
| `test_datalab_object_retrievable` | Object created in DataLab retrievable via `workspace.get()` |
| `test_datalab_removal_reflected` | Object removed in DataLab disappears from `workspace.list()` |

### 3.5 Computation via Kernel

| Test | Description |
|------|-------------|
| `test_calc_via_kernel` | `workspace.calc("normalize")` triggers DataLab processing |
| `test_calc_with_params` | `workspace.calc("gaussian_filter", param)` works |
| `test_calc_result_visible` | Result of `calc()` appears in both kernel and DataLab |

### 3.6 Visualization Synchronization

| Test | Description |
|------|-------------|
| `test_plotter_updates_datalab_view` | `plotter.plot()` updates DataLab visualization |
| `test_plotter_inline_and_gui` | Plot appears both inline and in DataLab |

### 3.7 Project Persistence

| Test | Description |
|------|-------------|
| `test_save_syncs_datalab_project` | `workspace.save()` saves DataLab project state |
| `test_load_updates_datalab` | `workspace.load()` updates DataLab with loaded data |

---

## 4. Contract Tests

These tests validate the **user-facing guarantees** from `specification.md`.

### 4.1 API Stability

| Test | Description |
|------|-------------|
| `test_workspace_api_complete` | All documented `workspace` methods exist |
| `test_plotter_api_complete` | All documented `plotter` methods exist |
| `test_api_signatures_stable` | Method signatures match specification |

### 4.2 Mode Transparency

| Test | Description |
|------|-------------|
| `test_same_code_both_modes` | Identical code runs in standalone and live modes |
| `test_no_mode_specific_exceptions` | No errors specific to one mode |

**Implementation approach**:

```python
CANONICAL_NOTEBOOK_CODE = """
import numpy as np
from sigima import create_signal

# Create and add signal
x = np.linspace(0, 10, 100)
y = np.sin(x)
sig = create_signal("test_signal", x, y)
workspace.add("my_signal", sig)

# Process
result = sigima.proc.signal.normalize(workspace.get("my_signal"))
workspace.add("normalized", result)

# Verify
assert workspace.exists("my_signal")
assert workspace.exists("normalized")
assert len(workspace.list()) == 2
"""

def test_canonical_code_standalone():
    """Run canonical code in standalone mode."""
    # Execute CANONICAL_NOTEBOOK_CODE in standalone kernel
    pass

def test_canonical_code_live(live_datalab):
    """Run canonical code in live mode."""
    # Execute CANONICAL_NOTEBOOK_CODE in live kernel
    pass
```

### 4.3 Error Messages

| Test | Description |
|------|-------------|
| `test_missing_object_error_message` | `KeyError` message lists available objects |
| `test_error_messages_user_friendly` | Errors don't expose internal implementation details |

### 4.4 Performance

| Test | Description |
|------|-------------|
| `test_large_signal_no_copy` | Large arrays not duplicated unnecessarily |
| `test_workspace_operations_responsive` | Basic operations complete in < 100ms |

---

## 5. Regression Tests

Tests for specific bugs or edge cases discovered during development.

| Test | Description |
|------|-------------|
| `test_unicode_object_names` | Object names with Unicode characters work |
| `test_special_characters_in_names` | Names with spaces, dots, hyphens work |
| `test_empty_signal` | Zero-length signal handled correctly |
| `test_empty_image` | Zero-size image handled correctly |
| `test_complex_image` | Complex-valued images supported |
| `test_masked_image` | Images with masks preserved |
| `test_roi_preservation` | ROIs on objects preserved through operations |
| `test_metadata_preservation` | All metadata preserved through save/load |

---

## 6. Jupyter Protocol Tests

Tests validating Jupyter kernel protocol compliance.

| Test | Description |
|------|-------------|
| `test_execute_request` | Kernel handles `execute_request` correctly |
| `test_complete_request` | Tab completion works |
| `test_inspect_request` | Object inspection (Shift+Tab) works |
| `test_interrupt_request` | Kernel can be interrupted |
| `test_shutdown_request` | Kernel shuts down cleanly |
| `test_kernel_info_reply` | Kernel info contains correct metadata |

---

## 7. Test Execution

### 7.1 Running Tests

```bash
# All tests (standalone mode only)
pytest datalab_kernel/tests/

# Include live mode tests (requires DataLab)
pytest datalab_kernel/tests/ --live

# Specific category
pytest datalab_kernel/tests/unit/
pytest datalab_kernel/tests/integration/
pytest datalab_kernel/tests/contract/
```

### 7.2 CI Configuration

```yaml
# Standalone tests (fast, no GUI)
test-standalone:
  script:
    - pytest datalab_kernel/tests/unit/
    - pytest datalab_kernel/tests/contract/ --standalone-only

# Live tests (requires display, slower)
test-live:
  script:
    - pytest datalab_kernel/tests/integration/ --live
    - pytest datalab_kernel/tests/contract/ --live
```

### 7.3 Test Markers

```python
import pytest

@pytest.mark.standalone
def test_standalone_only():
    """Test that only runs in standalone mode."""
    pass

@pytest.mark.live
def test_live_only():
    """Test that requires live DataLab instance."""
    pass

@pytest.mark.contract
def test_contract():
    """Contract test (runs in both modes)."""
    pass
```

---

## 8. Test Data

### 8.1 Sample Data Generation

```python
# datalab_kernel/tests/data.py

import numpy as np
from sigima import create_signal, create_image

def make_test_signal(name: str = "test_signal") -> SignalObj:
    """Create a simple test signal."""
    x = np.linspace(0, 10, 1000)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))
    return create_signal(name, x, y, xlabel="Time", xunit="s",
                         ylabel="Amplitude", yunit="V")

def make_test_image(name: str = "test_image") -> ImageObj:
    """Create a simple test image."""
    data = np.random.rand(256, 256).astype(np.float32)
    return create_image(name, data, xlabel="X", xunit="px",
                        ylabel="Y", yunit="px", zlabel="Intensity", zunit="a.u.")
```

### 8.2 Reference HDF5 Files

Store reference `.h5` files in `datalab_kernel/tests/data/` for:

- Format compatibility testing
- Regression testing
- Cross-version compatibility

---

## 9. Coverage Goals

| Category | Target Coverage |
|----------|-----------------|
| Workspace API | 100% |
| Plotter API | 100% |
| Mode detection | 100% |
| Live synchronization | 90% |
| Error handling | 90% |
| Edge cases | 80% |

---

## 10. References

- **Specification**: `plans/specification.md`
- **Architecture**: `plans/architecture.md`
- **DataLab test patterns**: `datalab/tests/features/control/simpleclient_unit_test.py`
- **DataLab background execution**: `datalab/tests/__init__.py::run_datalab_in_background`
- **Sigima visualization tools**: `sigima/tests/vistools/vistools_mpl.py`

