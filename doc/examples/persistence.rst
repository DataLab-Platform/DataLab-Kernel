.. _example_persistence:

Persistence
===========

This example demonstrates saving and loading workspaces with HDF5.

Saving a Workspace
------------------

.. code-block:: python

    from datalab_kernel import Workspace, Plotter
    from datalab_kernel.objects import create_signal, create_image
    import numpy as np

    # Create workspace with objects
    workspace = Workspace()

    # Add signals
    t = np.linspace(0, 10, 1000)
    workspace.add("sine", create_signal("Sine Wave", t, np.sin(t)))
    workspace.add("cosine", create_signal("Cosine Wave", t, np.cos(t)))

    # Add image
    data = np.random.rand(128, 128).astype(np.float32)
    workspace.add("noise", create_image("Random Noise", data))

    print("Objects:", workspace.list())

    # Save to HDF5
    workspace.save("my_analysis.h5")
    print("Workspace saved to my_analysis.h5")


Loading a Workspace
-------------------

.. code-block:: python

    # Create new workspace
    workspace2 = Workspace()

    # Load from HDF5
    workspace2.load("my_analysis.h5")

    print("Loaded objects:", workspace2.list())

    # Verify data integrity
    original_sine = workspace.get("sine")
    loaded_sine = workspace2.get("sine")

    np.testing.assert_array_equal(original_sine.x, loaded_sine.x)
    np.testing.assert_array_equal(original_sine.y, loaded_sine.y)
    print("Data integrity verified!")


Incremental Saves
-----------------

.. code-block:: python

    workspace = Workspace()

    # Initial work
    workspace.add("signal1", create_signal("First", t, np.sin(t)))
    workspace.save("work_v1.h5")

    # More work
    workspace.add("signal2", create_signal("Second", t, np.cos(t)))
    workspace.save("work_v2.h5")

    # Now you have checkpoints at each version


Live Mode Persistence
---------------------

Persistence also works in live mode:

.. code-block:: python

    workspace = Workspace()  # May be live or standalone

    # Save current DataLab state
    workspace.save("datalab_snapshot.h5")

    # Load into same or different workspace
    # In live mode, objects are added to DataLab
    workspace.load("datalab_snapshot.h5")


HDF5 File Structure
-------------------

The saved HDF5 file contains:

.. code-block:: text

    my_analysis.h5
    ├── signals/
    │   ├── sine/
    │   │   ├── x [dataset]
    │   │   ├── y [dataset]
    │   │   └── metadata [attributes]
    │   └── cosine/
    │       ├── x [dataset]
    │       ├── y [dataset]
    │       └── metadata [attributes]
    └── images/
        └── noise/
            ├── data [dataset]
            └── metadata [attributes]

You can inspect the file with any HDF5 viewer (HDFView, h5py, etc.):

.. code-block:: python

    import h5py

    with h5py.File("my_analysis.h5", "r") as f:
        print("Contents:")
        f.visititems(lambda name, obj: print(f"  {name}"))
