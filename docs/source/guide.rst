User Guide
============

.. toctree::
    :maxdepth: 2
    :hidden:
    
    guide/what_mesh
    guide/mesh_class
    guide/elem_types
    guide/connectivity
    guide/implicit_meshing
    guide/image_meshing
    guide/csg
    guide/io
    guide/ref

Getting Started
---------------

Python Package Index (PyPI)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
MyMesh can be installed from the python package index (PyPI): 

...

Installing from source:
^^^^^^^^^^^^^^^^^^^^^^^
Download/clone the repository, then run:

.. code-block::

    pip install -e <path>/mymesh

with :code:`<path>` replaced with the file path to the mymesh root directory.

MyMesh depends primarily on built-in or well established python packages like numpy and scipy. The following are optional dependencies required only for specific functions:

=======  ==================== =============================
Package  Purpose              Install
=======  ==================== =============================
meshio   Mesh file I/O        ``pip install meshio``
pydicom  DICOM image file I/O ``pip install pydicom``  
cv2      Image file I/O       ``pip install opencv-python``
vispy    Mesh visualization   ``pip install vispy``
=======  ==================== =============================

MyMesh can be used without these optional dependencies and if a function requires them, an error will be raised instructing the user to install the needed dependency.

