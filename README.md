# DataLab-Kernel

**A standalone Jupyter kernel providing seamless, reproducible access to DataLab workspaces, with optional live synchronization to the DataLab GUI.**

---

## Overview

**DataLab-Kernel** is a custom Jupyter kernel designed to bridge **DataLab** and the **Jupyter** ecosystem.

It enables scientists and engineers to:

- run reproducible analyses in Jupyter notebooks,
- interact transparently with DataLab’s internal workspace when DataLab is running,
- share notebooks that can be replayed **with or without DataLab**,
- combine narrative, code, and results without sacrificing interactive visualization.

DataLab-Kernel is **not** a replacement for DataLab’s GUI.  
It is a **complementary execution layer** that turns DataLab into a hybrid scientific platform:
**GUI-driven when needed, notebook-driven when appropriate.**

---

## Key Features

- **Single, stable user API**
  - `workspace` for data access and persistence
  - `plotter` for visualization
  - `sigima` for scientific processing

- **Two execution modes, one notebook**
  - **Live mode**: automatic synchronization with a running DataLab instance
  - **Standalone mode**: notebook-only execution, fully reproducible

- **Reproducibility by design**
  - Analyses can be saved and reloaded using `.h5` files
  - Notebooks run unchanged across environments

- **Performance-aware**
  - Optimized data handling when DataLab is attached
  - No unnecessary serialization for large datasets

- **Decoupled architecture**
  - Installable independently of DataLab
  - DataLab is a privileged host, not a requirement

---

## Typical Usage

```python
img = workspace.get("i042")
filtered = sigima.proc.image.butterworth(img, cut_off=0.2)
workspace.add("filtered_i042", filtered)
plotter.plot("filtered_i042")
```

Depending on the execution context:

- the result appears inline in the notebook,
- and, if DataLab is running, it also appears automatically in the DataLab GUI,
  with views and metadata kept in sync.

---

## Execution Modes

### Live Mode (DataLab-attached)

- DataLab launches a Jupyter server and starts `kernel-datalab`.
- The kernel detects DataLab at runtime.
- Workspace operations and visualizations are synchronized with the GUI.

### Standalone Mode (Notebook-only)

- The kernel is used like any standard Jupyter kernel.
- No DataLab installation or GUI is required.
- Data are managed locally and persisted to `.h5` files.

**The same notebook runs unchanged in both modes.**

---

## Installation

### Standalone usage

```bash
pip install datalab-kernel sigima
jupyter lab
```

Then select **DataLab Kernel** from the kernel list.

### With DataLab

When installed alongside DataLab, the kernel is automatically available and can be launched directly from the DataLab interface.

---

## Persistence and Sharing

Workspace state can be saved and reloaded:

```python
workspace.save("analysis.h5")
workspace.load("analysis.h5")
```

This enables:

- sharing notebooks and data with collaborators,
- replaying analyses without DataLab,
- resuming workflows inside DataLab by reopening the associated project.

---

## Documentation

- **User contract and behavior**: see `plans/specification.md`
- **Vision and architectural principles**: see `plans/architecture.md`

---

## Project Status

DataLab-Kernel is under active design and development.

---

## License

This project is released under an open-source license (see `LICENSE` file).
