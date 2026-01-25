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

    $ pip install datalab-kernel

This will install the kernel and all required dependencies including Sigima.

Installing the Jupyter Kernel
-----------------------------

After installing the package, register the kernel with Jupyter:

.. code-block:: console

    $ datalab-kernel-install

This makes the "DataLab" kernel available in Jupyter Notebook, JupyterLab,
and VS Code.

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
