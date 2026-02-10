# Version 0.2 #

## DataLab-Kernel Version 0.2.10 (2026-02-10) ##

### ✨ New features since version 0.2.9 ###

* **Multi-object plotting**: `plotter.plot()` now accepts lists of signals or images,
  enabling side-by-side visualization of multiple objects in a single call

* **Analysis result display**: Geometry and table results now display as separate
  HTML tables below the plot instead of cluttering the figure with on-plot text
  annotations — especially important when working with multiple ROIs

* **Consistent figure sizing**: Introduced `DEFAULT_PLOT_WIDTH` constant for
  uniform figure dimensions across Plotly and Matplotlib backends

* **User-defined figure dimensions**: Plotly plots now accept custom `width` and
  `height` keyword arguments for fine-grained control over figure sizing

* **Advanced plotting showcase notebook**: Added a comprehensive notebook
  demonstrating advanced features: multiple curve styles, colormaps, ROI-based
  analysis, geometry/table results, and interactive Plotly visualization

### 🛠️ Bug Fixes since version 0.2.9 ###

* Fixed legend display in Plotly plots — legends are now hidden by default for
  cleaner visualization

* Fixed default Plotly signal figure dimensions to match Matplotlib's aspect ratio

* Fixed figure serialization performance in Plotly backend

* Fixed image rendering edge cases in Matplotlib backend

### 🔧 Other changes since version 0.2.9 ###

* Updated documentation with structured "Try it Online" section linking to both
  Quick Start and Advanced Showcase notebooks

* Improved portability of development scripts by using relative paths

## DataLab-Kernel Version 0.2.9 (2026-02-08) ##

### ✨ New features ###

#### Plotly visualization backend ####

* Added a new **Plotly** interactive visualization backend as an alternative to
  the existing Matplotlib static backend. Plotly figures provide rich
  interactivity (zoom, pan, hover tooltips) directly in Jupyter notebooks.
* Install with `pip install datalab-kernel[plotly]`.

#### Smart backend selection ####

* When both Plotly and Matplotlib are available, **Plotly is automatically
  preferred** for its richer interactive experience.
* Switch backends at runtime with `plotter.set_backend("matplotlib")` or
  `plotter.set_backend("plotly")`.
* Override the default backend via the `DATALAB_PLOTTER_BACKEND` environment
  variable.

#### Enhanced Matplotlib backend ####

* Extracted the Matplotlib rendering logic into a dedicated `matplotlib_backend`
  module with improved support for:
  * Error bars
  * Curve styles (Sticks, Steps)
  * Per-object line styling from metadata
  * Log scale and axis bounds
  * Non-uniform image coordinates (`pcolormesh`)
  * Per-object colormaps

#### Figure height customization ####

* Enhanced figure height customization for both backends, with updated
  documentation.

#### MIME bundle support for Plotly ####

* Plotly figures now produce proper MIME bundles for native Jupyter display
  integration.

### 🔧 Improvements ###

#### Documentation ####

* Removed references to the legacy XML-RPC connection method; documentation now
  emphasizes the **Web API** as the primary (and recommended) connection method.
* Updated README with visualization backend documentation, usage examples, and
  runtime switching instructions.
* Updated online demo badge link.

#### Python version support ####

* Added Python **3.13** and **3.14** to the supported version classifiers.

#### Optional dependency ####

* Added `plotly` optional dependency group in `pyproject.toml`
  (`pip install datalab-kernel[plotly]`).

#### Dev tooling ####

* Enhanced wheel building script (`serve_dev_wheels.py`) to support package
  selection via `--packages` argument.
* New VS Code tasks for interactive plot tests, dev wheel serving, and
  backend-specific testing.

### 🛠️ Bug Fixes ###

* Fixed test compatibility with **Python 3.9** by using `ExitStack` instead of
  multiple `mock.patch` context managers.
* Fixed test module shadowing issue by using `mock.patch.object` instead of
  `mock.patch` with string targets.
* Fixed various Ruff and Pylint warnings across the codebase.
* Fixed test workflow branch references for DataLab installation.

### 🧪 Test Suite ###

* Added **65 tests** for the Matplotlib backend covering all rendering features,
  backward-compatibility aliases, and edge cases.
* Added **40 tests** for the Plotly backend covering all rendering features and
  edge cases.
* Added backend selection and resolution logic tests.
* Added `--gui` interactive gallery viewers for visual test validation (tkinter
  for Matplotlib, tabbed HTML for Plotly).

## DataLab-Kernel Versions 0.2.0–0.2.8 ##

These early development releases established the core architecture of
DataLab-Kernel, including:

* Workspace API (`add`, `get`, `remove`, `list`, `save`, `load`)
* Standalone and Live operating modes with auto-discovery
* Matplotlib-based plotting (`Plotter` class)
* Geometry and table result display
* HDF5 persistence for reproducible analysis
* `%load_ext datalab_kernel` extension for JupyterLite compatibility
* Web API-based connection to DataLab (replacing legacy XML-RPC)
* Comprehensive test suite foundation
