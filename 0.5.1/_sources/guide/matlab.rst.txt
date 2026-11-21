Using MyMesh in MATLAB
----------------------
MyMesh can be used in MATLAB using `MATLAB's python interface <https://www.mathworks.com/help/matlab/python-language.html?s_tid=CRUX_lftnav>`_. 
The python environment must be set to an environment where MyMesh has been 
installed (see `pyenv <https://www.mathworks.com/help/matlab/ref/pyenv.html>`_).
See the `list of compatible versions <https://www.mathworks.com/support/requirements/python-compatibility.html>`_ 
to be sure an appropriate python version is set for the MATLAB version being used.

`GIBBON <https://www.gibboncode.org/>`_ is recommended as a native-MATLAB solution for meshing.

Once the environment is set, MyMesh commands (or other python functions) can be 
run by adding a :code:`py.` prefix, for example:

.. code-block:: matlab

    M = py.mymesh.primitives.Torus([0,0,0], 1, 0.5);

There is no need to first `import` MyMesh like you would in Python; as long as it's installed in the environment it will be available.

Only limited testing of MyMesh in Matlab has been performed, so proceed with caution and please report any issues that you encounter. 

Tips for using MyMesh in MATLAB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- If a :class:`mesh` object is created without being suppressed by a semicolon, all properties (e.g. :attr:`~mymesh.mesh.mesh.Centroids`, :attr:`~mymesh.mesh.mesh.NodeNeighbors`, etc.) will be evaluated. For large meshes, this may cause a significant amount of overhead. 
- MyMesh seems to run better when the execution mode is set to "OutOfProcess" (:code:`pyenv("ExecutionMode", "OutOfProcess")`).
