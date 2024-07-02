.. currentmodule:: mymesh

File Input/Output
=================

Mesh files
----------
Reading and writing meshes from various file types is supported through the `meshio <https://github.com/nschloe/meshio>`_ which can read and write to a variety of common filetypes, including .vtu, .stl, .obj, and .inp. The :class:`~mymesh.mesh` class provides :func:`~mesh.mesh.read` and :meth:`~mesh.mesh.write` methods which utilize meshio to read and write from any meshio-supported format. Additionally, :class:`mesh` objects can be converted to/from meshio's mesh class using :meth:`~mesh.mesh.mymesh2meshio` and :meth:`~mesh.mesh.meshio2mymesh`.

.. code-block::

    from Mesh import mesh
    m = mesh.read('example.stl')
    m.write('example.vtu')

Image files
-----------
Images can be converted to/from voxel meshes using :func:`~mymesh.converter.im2voxel` and :func:`~mymesh.converter.voxel2im`. Additionally, an image can be read into a :class:`~mymesh.mesh` object directly using :meth:`mesh.imread`. Currently :meth:`mesh.imread` only creates voxel meshes, but it can be paired with contouring options from :mod:`contour` to create surface or volume meshes. There are a number of ways to convert a non-voxel mesh to an image (including using :func:`~mymesh.implicit.mesh2sdf`). A more comprehensive ``imwrite`` function is planned, but not yet implemented.