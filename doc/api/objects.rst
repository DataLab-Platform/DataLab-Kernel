.. _api_objects:

Objects
=======

Data objects for signals and images, re-exported from Sigima.

.. automodule:: datalab_kernel.objects
   :members:
   :undoc-members:
   :show-inheritance:

SignalObj
---------

The ``SignalObj`` class represents 1D signal data with X/Y arrays,
optional error bars, metadata, units, and labels.

See `Sigima SignalObj documentation <https://sigima.readthedocs.io/en/latest/api/objects.html>`_
for complete details.

Basic usage:

.. code-block:: python

    from datalab_kernel.objects import SignalObj, create_signal
    import numpy as np

    # Using create_signal (recommended)
    signal = create_signal(
        "My Signal",
        np.linspace(0, 10, 100),
        np.sin(np.linspace(0, 10, 100)),
        units=("s", "V"),
        labels=("Time", "Voltage"),
    )

    # Direct instantiation
    signal = SignalObj()
    signal.set_xydata(x_array, y_array)
    signal.title = "My Signal"


ImageObj
--------

The ``ImageObj`` class represents 2D image data with metadata,
coordinate system, units, and labels.

See `Sigima ImageObj documentation <https://sigima.readthedocs.io/en/latest/api/objects.html>`_
for complete details.

Basic usage:

.. code-block:: python

    from datalab_kernel.objects import ImageObj, create_image
    import numpy as np

    # Using create_image (recommended)
    image = create_image(
        "My Image",
        np.random.rand(256, 256),
        units=("mm", "mm", "counts"),
        labels=("X", "Y", "Intensity"),
    )

    # Direct instantiation
    image = ImageObj()
    image.data = data_2d
    image.title = "My Image"


create_signal
-------------

Factory function for creating signal objects.

.. code-block:: python

    create_signal(
        title: str,
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
        dx: np.ndarray | None = None,
        dy: np.ndarray | None = None,
        metadata: dict | None = None,
        units: tuple[str, str] | None = None,
        labels: tuple[str, str] | None = None,
    ) -> SignalObj

**Parameters:**

- ``title``: Signal title
- ``x``: X data array
- ``y``: Y data array
- ``dx``: X error bars (optional)
- ``dy``: Y error bars (optional)
- ``metadata``: Additional metadata dictionary
- ``units``: Tuple of (x_unit, y_unit)
- ``labels``: Tuple of (x_label, y_label)


create_image
------------

Factory function for creating image objects.

.. code-block:: python

    create_image(
        title: str,
        data: np.ndarray | None = None,
        metadata: dict | None = None,
        units: tuple | None = None,
        labels: tuple | None = None,
    ) -> ImageObj

**Parameters:**

- ``title``: Image title
- ``data``: 2D data array
- ``metadata``: Additional metadata dictionary
- ``units``: Tuple of (x_unit, y_unit, z_unit)
- ``labels``: Tuple of (x_label, y_label, z_label)
