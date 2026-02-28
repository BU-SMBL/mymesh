.. currentmodule:: mymesh.mesh

.. _Working_with_the_mesh_class:

Working with the :class:`mesh` class
====================================


The :class:`mesh` class, at its core, stores the definition of a mesh. A
mesh is defined by the coordinates of its nodes (``NodeCoords``) and set of node
connections (``NodeConn``) that define the elements (:attr:`mesh.points` and 
:attr:`mesh.cells` are offered as aliases to :attr:`mesh.NodeCoords` and 
:attr:`mesh.NodeConn` since these are commonly used terms for the same attributes in 
other softwares). 

On-demand properties
--------------------

There are a number of properties that can calculate various mesh features 
on-demand and cache them for quick access later. For example, 
:attr:`mesh.Centroids` will calculate the element centroids the first time it is 
referenced, and then they will be stored to avoid the need for subsequent 
recalculation. This offers a balance of convenience and efficiency, without the 
need to calculate unnecessary features or recalculate features that are needed 
multiple times (see :class:`mesh` for the full list of properties). 

Unpacking
---------

One of the goals in creating a MyMesh-specific mesh class was to make it easy
to come and go from the MyMesh "ecosystem" and avoid lock-in, so that users can
easily take advantage of the functionality offered by other well established
mesh packages or custom code. To make it easy to move to and from instances of 
this class, objects are unpackable into ``NodeCoords`` and ``NodeConn``. This
means that the following three code blocks all achieve the same thing:

.. code-block::

    sphere = primitives.Sphere([0,0,0], .5)
    NodeCoords = sphere.NodeCoords
    NodeConn = sphere.NodeConn

.. code-block::

    sphere = primitives.Sphere([0,0,0], .5)
    NodeCoords, NodeConn = sphere

.. code-block::

    NodeCoords, NodeConn = primitives.Sphere([0,0,0], .5)



Additionally, since many of the lower-level functions in MyMesh (such as 
those in :mod:`mymesh.utils`) take ``NodeCoords`` and ``NodeConn`` as inputs,
users can use the `*` operator to unpack the mesh directly in the function call:

.. code-block::

    sphere = primitives.Sphere([0,0,0], .5)
    NodeNeighbors = utils.getNodeNeighbors(*sphere)

Similarly, many functions in the :mod:`mymesh.converter` module take ``NodeCoords``
and ``NodeConn`` as inputs and return new or modified node coordinates and node 
connectivities, so you can avoid having to deal with separate intermediate 
values:

.. code-block::

    sphere = primitives.Sphere([0,0,0], .5)
    VoxelNodeCoords, VoxelNodeConn  = converter.surf2voxel(sphere.NodeCoords, sphere.NodeConn, .1)
    voxelized = mesh(VoxelNodeCoords, VoxelNodeConn)

is equivalent to

.. code-block::

    sphere = primitives.Sphere([0,0,0], .5)
    voxelized = mesh(*converter.surf2voxel(*sphere, .1))


Storing data with meshes
------------------------

Data associated with meshes can be stored within the :class:`mesh` object. 
Node data (data for which there are values associated with each node of the mesh) can be stored in the :attr:`mesh.NodeData` attribute. 
Likewise, data associated with the elements of the mesh (element data) can be stored in the :attr:`mesh.ElemData` attribute.
Both :attr:`mesh.NodeData` and :attr:`mesh.ElemData` data are dictionaries (:class:`dict`), so named data entries can be stored, for example, :code:`M.NodeData['strain'] = strain`. 
Scalar and vector node/element data can be stored, and these values will be saved when writing the mesh to a .vtu file or other file formats that support such properties (see :ref:`File Input/Output` for more details on reading/writing files). 

From meshes to meshes
---------------------

There are several :class:`mesh` class methods that produce new meshes based on
the original. 

:meth:`~mesh.copy`
^^^^^^^^^^^^^^^^^^
:meth:`~mesh.copy` produces an identical copy of the original mesh. The copies 
are separate and do not reference each other, meaning any modification to one 
mesh won't modify the other.

.. code::

    M = primitives.Grid([0,1,0,1,0,1], 0.1)
    M2 = M.copy()

:meth:`~mesh.Clip`
^^^^^^^^^^^^^^^^^^
:meth:`~mesh.Clip` cuts the mesh along a plane 

.. plot::

    M = primitives.Grid([-1,1,-1,1,-1,1], 0.1)
    M2 = M.Clip(normal=[1,1,-1])
    M2.plot(show_edges=False, view='trimetric', bgcolor='w')

:meth:`~mesh.Threshold`
^^^^^^^^^^^^^^^^^^^^^^^
:meth:`~mesh.Threshold` generates a new mesh that keeps elements from the 
original mesh based on scalar values and the chosen thresholding rule.

.. plot::

    M = primitives.Grid([-1,1,-1,1,-1,1], 0.1)
    M2 = M.Threshold(scalars=implicit.sphere([0,0,0], 1)(*M.points.T), threshold=0, mode='<=')
    M2.plot(show_edges=False, view='trimetric', bgcolor='w')

:meth:`~mesh.Contour`
^^^^^^^^^^^^^^^^^^^^^

:meth:`~mesh.Contour` generates a new mesh that contours the 
original mesh based on scalar values. 

.. plot::

    M = primitives.Grid([-1,1,-1,1,-1,1], 0.1)
    M2 = M.Contour(scalars=implicit.sphere([0,0,0], 1)(*M.points.T), threshold=0, threshold_direction=-1, Type='surf')
    M2.plot(show_edges=False, view='trimetric', bgcolor='w')

