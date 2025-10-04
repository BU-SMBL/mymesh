.. currentmodule:: mymesh

File Input/Output
=================

Mesh files
----------
Reading and writing meshes from various file types is supported through the use 
of the `meshio <https://github.com/nschloe/meshio>`_ package which can read and 
write to a variety of common filetypes, including .vtu, .stl, .obj, and .inp. 
The :class:`~mymesh.mesh` class provides :func:`~mymesh.mesh.read` and 
:meth:`~mymesh.mesh.write` methods which utilize meshio to read and write from 
any meshio-supported format. 
Additionally, :class:`mesh` objects can be converted 
to/from meshio's mesh class using :meth:`~mymesh.mesh.to_meshio` and 
:meth:`~mymesh.mesh.meshio2mymesh`.

The VTK ".vtu" file format is recommend for saving meshes as it supports the storage of node and element data, can be reliable saved and re-loaded without alteration, and can be used for visualization with paraview.  

.. code-block::

    from mymesh import mesh
    m = mesh.read('example.stl')
    m.write('example.vtu')

Image files
-----------
The :mod:`~mymesh.image` module has functions for reading/writing image files and 
generating image-based meshes. See :ref:`Image-based Meshing` for more details.