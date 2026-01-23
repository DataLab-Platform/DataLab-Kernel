.. _guidelines:

Guidelines
==========

Code Style
----------

- Follow PEP 8
- Use Ruff for formatting and linting
- Maximum line length: 88 characters (Black default)
- Use type annotations

Run linting:

.. code-block:: console

    $ ruff format datalab_kernel
    $ ruff check datalab_kernel --fix


Docstrings
----------

Use Google-style docstrings:

.. code-block:: python

    def my_function(param1: str, param2: int) -> bool:
        """Short description.

        Longer description if needed.

        Args:
            param1: Description of param1
            param2: Description of param2

        Returns:
            Description of return value

        Raises:
            ValueError: When input is invalid
        """


Testing
-------

Write tests for all new functionality:

.. code-block:: console

    # Run all tests
    $ pytest

    # Run with coverage
    $ pytest --cov=datalab_kernel

    # Run specific test file
    $ pytest datalab_kernel/tests/unit/test_workspace.py


Test Categories
^^^^^^^^^^^^^^^

- **Unit tests** (``tests/unit/``): Fast, isolated tests
- **Contract tests** (``tests/contract/``): API contract verification
- **Integration tests** (``tests/integration/``): Live DataLab tests

Integration tests require ``--live`` flag and optionally ``--start-datalab``:

.. code-block:: console

    $ pytest --live --start-datalab datalab_kernel/tests/integration


Commit Messages
---------------

Use conventional commit format:

.. code-block:: text

    feat: Add workspace resync functionality
    fix: Handle empty signal arrays correctly
    docs: Update installation instructions
    test: Add integration tests for live mode
    refactor: Simplify backend abstraction


Pull Request Process
--------------------

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/my-feature``
3. Make changes and add tests
4. Run linting and tests
5. Commit with descriptive messages
6. Push and create a pull request
7. Address review feedback
