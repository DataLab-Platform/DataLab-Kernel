.. _installation:

Installation
============

This section provides instructions on how to install DataLab-Kernel.

How to install
--------------

DataLab-Kernel is available in several forms:

- As a Python package via :ref:`install_pip`
- From :ref:`install_source` for development

.. _install_pip:

Package manager ``pip``
^^^^^^^^^^^^^^^^^^^^^^^

:octicon:`info;1em;sd-text-info` :bdg-info-line:`GNU/Linux` :bdg-info-line:`Windows` :bdg-info-line:`macOS`

Installing DataLab-Kernel from PyPI:

.. code-block:: console

    $ pip install datalab-kernel[cli]

This will install the kernel and all required dependencies including Sigima.

.. note::

    The ``[cli]`` extra installs ``jupyter-client``, which is required for
    the ``install``/``uninstall`` CLI commands. See :ref:`install_jupyterlite`
    for environments where this is not needed.

Installing the Jupyter Kernel
-----------------------------

After installing the package, register the kernel with Jupyter:

.. code-block:: console

    $ datalab-kernel-install

This makes the "DataLab" kernel available in Jupyter Notebook, JupyterLab,
and VS Code.

.. _install_jupyterlite:

JupyterLite (browser-based)
---------------------------

DataLab-Kernel is compatible with **JupyterLite**, a browser-based Jupyter
environment running on WebAssembly.

In JupyterLite, kernels are bundled at build time and the ``install``/``uninstall``
commands are not used. Instead, you load DataLab-Kernel as an IPython extension.

**Step 1: Add to your JupyterLite environment**

Create or edit your ``environment.yml``:

.. code-block:: yaml

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

**Step 2: Load the extension in your notebook**

In the first cell of your notebook, load the extension:

.. code-block:: python

    %load_ext datalab_kernel

This injects the DataLab namespace (``workspace``, ``plotter``, ``sigima``, etc.)
into your environment, just like the native kernel does.

**Why no ``[cli]`` extra?**

The ``[cli]`` extra includes ``jupyter-client``, which depends on ``pyzmq`` (ZeroMQ).
Since ZeroMQ requires native sockets unavailable in WebAssembly environments,
this dependency is not needed in JupyterLite.

Optional: DataLab Integration
-----------------------------

To enable live mode (synchronization with DataLab), install DataLab:

.. code-block:: console

    $ pip install datalab-platform

Or install both together:

.. code-block:: console

    $ pip install datalab-kernel[datalab]

.. _install_source:

Installing from source
^^^^^^^^^^^^^^^^^^^^^^

Clone the repository and install in development mode:

.. code-block:: console

    $ git clone https://github.com/DataLab-Platform/DataLab-Kernel.git
    $ cd DataLab-Kernel
    $ pip install -e .

For development with all tools:

.. code-block:: console

    $ pip install -e ".[dev]"


Verifying the Installation
--------------------------

Check that the kernel is properly installed:

.. code-block:: python

    from datalab_kernel import Workspace, Plotter
    from sigima import create_signal
    import numpy as np

    # Create workspace
    workspace = Workspace()
    print(f"Mode: {workspace.mode}")

    # Create a test signal
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    signal = create_signal("Test", x, y)
    workspace.add("test", signal)

    print(f"Objects: {workspace.list()}")
