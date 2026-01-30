# DataLab-Kernel

[![license](https://img.shields.io/pypi/l/datalab-kernel.svg)](./LICENSE)
[![pypi version](https://img.shields.io/pypi/v/datalab-kernel.svg)](https://pypi.org/project/datalab-kernel/)
[![PyPI status](https://img.shields.io/pypi/status/datalab-kernel.svg)](https://github.com/DataLab-Platform/DataLab-Kernel)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/datalab-kernel.svg)](https://pypi.org/project/datalab-kernel/)

**A standalone [Xeus-Python](https://github.com/jupyter-xeus/xeus-python)-based Jupyter kernel providing seamless, reproducible access to DataLab workspaces, with optional live synchronization to the DataLab GUI.**

---

## Overview

**DataLab-Kernel** is a custom Jupyter kernel designed to bridge **DataLab** and the **Jupyter** ecosystem. It is built on top of [**Xeus-Python**](https://github.com/jupyter-xeus/xeus-python), a lightweight and efficient Python kernel for Jupyter that offers improved performance, native debugger support, and excellent Qt event loop integration.

Thanks to Xeus-Python's architecture, DataLab-Kernel runs seamlessly in both:

- **Native Jupyter environments** (JupyterLab, Jupyter Notebook, VS Code)
- **[JupyterLite](https://jupyterlite.readthedocs.io/)** (browser-based Jupyter, no server required)

This enables scientists and engineers to:

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

Two connection methods are supported:

- **Web API** (recommended): HTTP/JSON connection using `DATALAB_WORKSPACE_URL` and `DATALAB_WORKSPACE_TOKEN` environment variables
- **XML-RPC** (legacy): Automatic connection when DataLab is running with remote control enabled

### Standalone Mode (Notebook-only)

- The kernel is used like any standard Jupyter kernel.
- No DataLab installation or GUI is required.
- Data are managed locally and persisted to `.h5` files.

**The same notebook runs unchanged in both modes.**

---

## Installation

### Standalone usage (desktop Jupyter)

```bash
pip install datalab-kernel[cli] sigima
python -m datalab_kernel install
jupyter lab
```

Then select **DataLab Kernel** from the kernel list.

### JupyterLite

DataLab-Kernel is compatible with **JupyterLite** (browser-based Jupyter).
In this environment, kernels are bundled at build time, so you load DataLab-Kernel
as an IPython extension instead.

**1. Add to your `environment.yml`:**

```yaml
name: xeus-python-kernel
channels:
  - https://repo.mamba.pm/emscripten-forge
  - conda-forge
dependencies:
  - numpy
  - matplotlib
  - h5py
  - datalab-kernel
  - sigima
```

**2. Load the extension in your notebook:**

```python
%load_ext datalab_kernel
```

This injects the DataLab namespace (`workspace`, `plotter`, `sigima`, etc.)
into your environment.

The `[cli]` extra is not needed in JupyterLite since `jupyter-client` depends
on `pyzmq`, which requires native sockets unavailable in WebAssembly.

### Dependencies

DataLab-Kernel uses **xeus-python** as its backend, which provides:

- Improved performance compared to ipykernel
- Native debugger support
- JupyterLite compatibility
- Better Qt event loop integration

The kernel requires:

- `xeus-python>=0.17.0` - The xeus-based Python kernel
- `xeus-python-shell>=0.6.0` - Python shell utilities for xeus-python
- `sigima>=1.0` - Scientific signal and image processing
- `numpy>=1.22`, `h5py>=3.0`, `matplotlib>=3.5`

Optional dependency (via `[cli]` extra):

- `jupyter-client>=7.0` - For `install`/`uninstall` CLI commands

### With DataLab

When installed alongside DataLab, the kernel is automatically available and can be launched directly from the DataLab interface.

### Installing from conda-forge (recommended)

For best compatibility, especially on Windows:

```bash
mamba create -n datalab-kernel
mamba activate datalab-kernel
mamba install xeus-python datalab-kernel -c conda-forge
python -m datalab_kernel install
```

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
