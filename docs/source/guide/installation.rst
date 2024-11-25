Installation
============

Python Package Index (PyPI)
---------------------------
MyMesh can be installed from the `python package index (PyPI) <https://pypi.org/project/mymesh/>_`: 

.. code-block::

    pip install mymesh[all]

Installing from source:
-----------------------
Download/clone the repository from `github <https://github.com/BU-SMBL/mymesh>`_, 
then run:

.. code-block::

    pip install -e <path>/mymesh[all]

with :code:`<path>` replaced with the file path to the mymesh root directory. 
To install with only the required dependencies, the :code:`[all]` can be omitted.

Dependencies
------------

MyMesh depends on a small number of built-in or well established python packages. 
Additionally, there are several optional dependencies that are required only for 
specific functions or use-cases.

Core dependencies
^^^^^^^^^^^^^^^^^

================ ======================
Package          Install
================ ======================
`numpy`_         ``pip install numpy``
`scipy`_         ``pip install scipy``
`sympy`_         ``pip install sympy``
================ ======================

Optional dependencies
^^^^^^^^^^^^^^^^^^^^^

================ ==================== ===================================================================== =============================
Package          Purpose              Used in                                                               Install
================ ==================== ===================================================================== =============================
`meshio`_        Mesh file I/O        :class:`~mymesh.mesh`                                                 ``pip install meshio``
`numba`_         Enhanced efficiency  :mod:`~mymesh.delaunay`, :mod:`~mymesh.utils`, :mod:`~mymesh.quality` ``pip install numba``
`pydicom`_       DICOM image file I/O :mod:`~mymesh.image`                                                  ``pip install pydicom``  
`opencv (cv2)`_  Image file I/O       :mod:`~mymesh.image`                                                  ``pip install opencv-python``
`triangle`_      Constrained Delaunay :mod:`~mymesh.delaunay`                                               ``pip install triangle``
`vispy`_         Mesh visualization   :mod:`~mymesh.visualize`                                              ``pip install vispy``
`matplotlib`_    Mesh visualization   :mod:`~mymesh.visualize`                                              ``pip install matplotlib``
`pillow`_        Mesh visualization   :mod:`~mymesh.visualize`                                              ``pip install pillow``
`jupyter_rfb`_   Mesh visualization   :mod:`~mymesh.visualize`                                              ``pip install jupyter_rfb``
`colorspacious`_ Mesh visualization   :mod:`~mymesh.visualize`                                              ``pip install colorspacious``
================ ==================== ===================================================================== =============================

MyMesh can be used without these optional dependencies and if a function requires them, an error will be raised instructing the user to install the needed dependency.

.. _numpy: https://numpy.org/
.. _scipy: https://scipy.org/
.. _sympy: https://sympy.org/
.. _meshio: https://github.com/nschloe/meshio
.. _numba: http://numba.pydata.org/
.. _pydicom: https://github.com/pydicom/pydicom
.. _opencv (cv2): https://github.com/opencv/opencv-python
.. _triangle: https://github.com/drufat/triangle
.. _vispy: https://vispy.org/
.. _matplotlib: https://matplotlib.org/
.. _pillow: https://github.com/python-pillow/Pillow
.. _jupyter_rfb: https://github.com/vispy/jupyter_rfb
.. _colorspacious: https://github.com/njsmith/colorspacious