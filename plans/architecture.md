# DataLab Jupyter Kernel — Vision and Architecture

## 1. Vision

The goal of the **DataLab Jupyter kernel** (`kernel-datalab`) is to provide **seamless, bidirectional interoperability** between **DataLab** and the **Jupyter** ecosystem, turning DataLab into a **hybrid scientific analysis platform**:

- a **rich graphical application** for interactive exploration and workflows,
- a **reproducible computational environment** for notebooks, scripting, and sharing.

This kernel is **not intended to replace DataLab** with Jupyter, nor to reduce DataLab to a web widget.
Instead, it establishes a **robust technical bridge** enabling:

- exposure of DataLab’s internal state (signals, images, processing results, projects) to notebooks,
- execution of Python code in a Jupyter kernel that may interact directly with a running DataLab instance,
- documentation, narration, and sharing of analyses while preserving the power of the GUI.

This document complements the user-facing specification (`plans/specification.md`) by describing **intent and architectural principles**, not user contracts.

---

## 2. User-Centric Target Scenario

> A researcher uses DataLab to load experimental data from the lab.
> From the GUI, they start an embedded Jupyter server that launches a **DataLab kernel**.
> They open a notebook (locally or remotely) connected to this kernel and write:
>
> ```python
> img = workspace.get("i042")
> filtered = sigima.proc.image.butterworth(img, cut_off=0.2)
> flt = workspace.add("filtered_i042", filtered)
> plotter.plot(flt)
> ```
>
> When executing the cell:
>
> - the processed image appears **inline in the notebook**,
> - **and simultaneously in the DataLab GUI**,
> - with views and metadata updated consistently.
>
> The notebook is then shared with a colleague, who can:
>
> - reproduce the analysis **without launching DataLab** (standalone mode),
> - or resume the workflow in the GUI by opening the associated DataLab project.

This scenario combines:

- **Reproducibility, narration, sharing** → Jupyter,
- **Analysis power, interactive visualization, structured workflows** → DataLab.

---

## 3. Core Design Principles

### 3.1 Single User API

From the user’s point of view, there is **one way to write code**:

- `workspace` for data access and persistence,
- `sigima` for scientific processing,
- `plotter` for visualization.

The same notebook must run **unchanged**:

- when DataLab is attached (*live mode*),
- or when the kernel runs alone (*standalone mode*).

### 3.2 Kernel Independence

`kernel-datalab` is:

- a **standalone Jupyter kernel**, installable independently of DataLab,
- capable of detecting at runtime whether a DataLab instance is available.

DataLab may embed and drive the kernel, but **must not be a prerequisite** for its use.

### 3.3 Backend Transparency

Differences between:

- in-process execution (high-performance, DataLab live),
- and standalone execution (reproducibility, headless environments),

are treated as **internal backend details**, invisible to the user-facing API.

---

## 4. Architectural Overview

### 4.1 Logical Components

- **Jupyter Kernel (`kernel-datalab`)**
  - Exposes the stable user API (`workspace`, `plotter`, `sigima`).
  - Selects the appropriate backend at startup.

- **Live Backend (optional)**
  - Activated when DataLab is running.
  - Provides direct or optimized access to DataLab’s data model and visualization.
  - Enables immediate GUI updates triggered from notebook execution.

- **Standalone Backend**
  - Used when no DataLab instance is detected.
  - Manages data locally (memory and `.h5` persistence).
  - Provides inline notebook visualization.

- **DataLab Application**
  - Hosts the GUI, project management, and visualization tools.
  - Optionally launches a Jupyter server preconfigured with `kernel-datalab`.

### 4.2 Kernel Implementation Strategy

The kernel may be implemented using:

- a pure-Python Jupyter kernel stack,
- or a native kernel implementation embedded in an application runtime.

The choice of technology (e.g. Python-based or native, in-process or IPC) is an **implementation decision**, guided by performance and deployment constraints, and **explicitly out of scope** for this document.

---

## 5. Alignment with the User Specification

This document intentionally defers all **user-visible behavior** to `spec.md`, including:

- workspace API guarantees,
- visualization behavior,
- persistence format (`.h5`),
- performance and error-handling expectations.

Any implementation of `kernel-datalab` must comply with `spec.md`.

---

## 6. Strategic Impact

Introducing a DataLab Jupyter kernel enables:

- **Scientific reproducibility by design**,
- **Seamless transition between GUI exploration and scripted analysis**,
- **Adoption in notebook-centric environments** (HPC, remote servers, teaching),
- **Long-term architectural resilience**, decoupling computation from presentation.

This positions DataLab as:
> not just a desktop application, but a **first-class scientific platform** in the modern Python ecosystem.

---

## 7. Conclusion

The DataLab Jupyter kernel is not an auxiliary feature.
It is a **structural extension** of DataLab toward open science workflows.

By separating:

- *vision* (this document),
- *user contract* (`plans/specification.md`),
- *implementation choices* (left open),

DataLab can evolve incrementally while preserving clarity, stability, and scientific credibility.
