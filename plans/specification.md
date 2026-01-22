# DataLab Jupyter Kernel — User-Facing Specification

## 1. Purpose and Scope

This document specifies the **user-facing behavior and guarantees** of the **DataLab Jupyter kernel** (`kernel-datalab`).

The goal is to allow scientists and engineers to:

- interactively process scientific data using Jupyter notebooks,
- transparently synchronize computations with the DataLab graphical interface when available,
- reproduce and share analyses **without requiring DataLab**.

This specification is **implementation-agnostic**:
it does not prescribe how the kernel is implemented internally (ipykernel, xeus, in-process, IPC, etc.),
only what the **user experience and observable behavior must be**.

---

## 2. Architectural Principle

### 2.1 Separation of Concerns

- **DataLab** is a Qt-based scientific application (GUI, project management, visualization).
- **`kernel-datalab`** is a **standalone Jupyter kernel**, distributed as an independent Python package.

They may cooperate tightly, but:

- neither must require the other to exist,
- neither must impose its lifecycle on the other.

This separation is mandatory to ensure:

- notebook reproducibility,
- headless execution (HPC, servers),
- long-term maintainability.

---

## 3. Execution Modes (Transparent to the User)

The kernel operates in two modes, selected automatically at runtime.

### 3.1 Live Mode (DataLab-attached)

- DataLab launches a Jupyter server and starts `kernel-datalab`.
- The kernel detects a live DataLab instance.
- All operations are synchronized with the DataLab GUI.

Typical use case:
> Interactive exploration of experimental data loaded in DataLab.

### 3.2 Standalone Mode (Notebook-only)

- The user launches Jupyter manually (`jupyter lab`, `jupyter notebook`).
- `kernel-datalab` is selected like any standard kernel.
- No DataLab instance is running.

Typical use case:
> Reproducing, reviewing, or sharing an analysis without DataLab.

**Important:**  
The **same notebook code must run unchanged in both modes**.

---

## 4. Kernel Installation and Discovery

### 4.1 Standalone Installation

Users install the kernel explicitly:

```bash
pip install datalab-kernel sigima
```

Then launch Jupyter normally.

### 4.2 DataLab Integration

When DataLab is installed:

- it declares `kernel-datalab` as a dependency,
- it exposes a UI action such as *“Open Jupyter Notebook”*,
- it launches Jupyter with `kernel-datalab` preselected.

No additional user configuration is required.

---

## 5. Default Notebook Namespace

When a notebook starts with `kernel-datalab`, the following objects are immediately available:

- `workspace` — data access and persistence
- `plotter` — visualization frontend
- `sigima` — scientific processing library (standard import)

Example:

```python
workspace
plotter
import sigima
```

No manual imports or connections are required.

---

## 6. Workspace API (User Contract)

### 6.1 Object Listing and Queries

```python
workspace.list()              # -> list of object names
workspace.exists("i042")      # -> bool
```

### 6.2 Retrieving Data

```python
img = workspace.get("i042")
```

Behavior:

- returns a manipulable object (NumPy-backed, with metadata),
- printable representation is informative (type, shape, dtype, units if available).

### 6.3 Adding or Replacing Data

```python
workspace.add("filtered_i042", img)
workspace.add("i042", img, overwrite=True)
```

Live mode:

- object appears immediately in DataLab,
- metadata and views are updated.

Standalone mode:

- object is stored locally (in memory or disk backend).

### 6.4 Removal and Renaming

```python
workspace.remove("filtered_i042")
workspace.rename("i042", "raw_i042")
```

State is kept consistent across kernel and DataLab (if attached).

---

## 7. Visualization API

### 7.1 Implicit Display

The last expression of a cell:

```python
workspace.get("i042")
```

Behavior:

- displayed inline in the notebook,
- no mandatory DataLab view creation.

### 7.2 Explicit Display

```python
plotter.plot("i042")
plotter.plot(workspace.get("i042"))
```

Behavior:

- always displayed inline in the notebook,
- additionally displayed or updated in DataLab if attached.

---

## 8. Canonical Usage Example

```python
img = workspace.get("i042")
filtered = sigima.proc.image.butterworth(img, cut_off=0.2)
workspace.add("filtered_i042", filtered)
plotter.plot("filtered_i042")
```

Expected result:

- filtered data displayed inline,
- synchronized appearance in DataLab (workspace + views),
- consistent metadata propagation.

---

## 9. Persistence and Reproducibility

### 9.1 Save Workspace State

```python
workspace.save("analysis.h5")
```

- Uses the `.h5` format.
- Stores data arrays and required metadata.
- Format is compatible with DataLab projects.

### 9.2 Reload Workspace State

```python
workspace.load("analysis.h5")
```

Behavior:

- restores objects in the current workspace,
- if DataLab is attached, imports them into the active project.

---

## 10. Synchronization Rules

### 10.1 Notebook → DataLab

Always synchronized:

- add / remove / rename operations,
- metadata updates.

### 10.2 DataLab → Notebook (Limited)

Required:

- workspace state remains consistent (`workspace.list()` reflects reality).

Optional:

- change notifications via callbacks or polling.

Real-time mirroring of GUI actions is **not required**.

---

## 11. Performance Expectations

From the user’s perspective:

- workspace operations must feel fluid,
- no visible latency for typical interactive workloads,
- large datasets must avoid unnecessary data copies.

The kernel must avoid:

- text-based serialization of large arrays,
- redundant memory duplication when DataLab is live.

---

## 12. Error Handling and UX

- Missing objects raise clear, actionable errors.
- Running standalone never fails due to missing DataLab.
- Messages must remain user-oriented, not implementation-oriented.

Example:

```python
workspace.get("unknown")
```

→ `KeyError: Object 'unknown' not found. Available objects: [...]`

---

## 13. Acceptance Criteria

The kernel is considered compliant if:

- the same notebook runs unchanged in live and standalone modes,
- `.h5` files ensure full reproducibility,
- DataLab integration is transparent and optional,
- no user-visible configuration is required.

---

## 14. Non-Goals

This specification does **not** require:

- bidirectional real-time GUI mirroring,
- exposure of DataLab internal APIs,
- kernel awareness of Qt or UI internals,
- a specific Jupyter kernel implementation technology.

---

## 15. Summary

`kernel-datalab` provides a **single, stable scientific notebook API**
capable of operating:

- as a high-performance companion to DataLab,
- or as a fully autonomous, reproducible execution environment.

For the user, **there is only one way to write code** —
everything else is an implementation detail.
