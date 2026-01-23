.. _development:

Development Setup
=================

This guide helps you set up a development environment for DataLab-Kernel.


Clone the Repository
--------------------

.. code-block:: console

    $ git clone https://github.com/DataLab-Platform/DataLab-Kernel.git
    $ cd DataLab-Kernel


Create Virtual Environment
--------------------------

.. code-block:: console

    $ python -m venv .venv
    $ source .venv/bin/activate  # Linux/macOS
    $ .venv\Scripts\activate     # Windows


Install in Development Mode
---------------------------

.. code-block:: console

    $ pip install -e ".[dev]"


Environment Configuration
-------------------------

Create a ``.env`` file for local development:

.. code-block:: text

    PYTHONPATH=.

If developing with local Sigima/DataLab clones:

.. code-block:: text

    PYTHONPATH=.;../sigima;../DataLab


Running Tests
-------------

.. code-block:: console

    # Unit and contract tests (no DataLab needed)
    $ pytest datalab_kernel/tests/unit datalab_kernel/tests/contract -v

    # Integration tests (requires DataLab)
    $ pytest --live --start-datalab datalab_kernel/tests/integration -v

    # All tests with coverage
    $ pytest --cov=datalab_kernel --cov-report=html


Linting
-------

.. code-block:: console

    # Format code
    $ ruff format datalab_kernel

    # Lint and fix
    $ ruff check datalab_kernel --fix


Building Documentation
----------------------

.. code-block:: console

    $ pip install sphinx pydata-sphinx-theme myst-nb sphinx-design sphinx-copybutton
    $ sphinx-build doc build/doc -b html


VS Code Configuration
---------------------

The repository includes VS Code configuration:

- ``.vscode/settings.json``: Editor settings
- ``.vscode/tasks.json``: Build tasks (Ruff, Pytest)
- ``.vscode/launch.json``: Debug configurations


Project Structure
-----------------

.. code-block:: text

    DataLab-Kernel/
    ├── datalab_kernel/
    │   ├── __init__.py       # Package entry point
    │   ├── workspace.py      # Workspace, backends
    │   ├── plotter.py        # Visualization
    │   ├── objects.py        # Re-exports from Sigima
    │   ├── persistence.py    # HDF5 save/load
    │   ├── install.py        # Kernel installation
    │   └── tests/
    │       ├── unit/         # Unit tests
    │       ├── contract/     # API contract tests
    │       └── integration/  # Live mode tests
    ├── doc/                  # Sphinx documentation
    ├── .github/workflows/    # CI configuration
    └── pyproject.toml        # Project configuration
