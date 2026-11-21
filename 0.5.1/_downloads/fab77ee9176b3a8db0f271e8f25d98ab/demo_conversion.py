"""
.. currentmodule:: mymesh

Conversion
==========

The :mod:`mymesh.converter` module contains many functions to convert between different types of meshes.
This example will highlight a few of them.

"""
#%%
# We'll start with a volumetric sphere mesh, that begins as a mix of hexahedra, tetrahedra, pyramids, and wedges.
# Most of the sphere is made of hexahedra (gray), the core is made up of pyramids (orange), and the central axis is made up of wedges (blue), except in the core where there are a few small tetrahedra (green).
from mymesh import converter, primitives, mesh

sphere = primitives.Sphere([0,0,0], 1, Type='vol')
sphere.ElemData['Element Type'] = [len(elem) for elem in sphere.NodeConn]
sphere.verbose = False
sphere.Clip().plot(scalars='Element Type', show_edges=True, view='trimetric', color='Accent', show_colorbar=False)
print(sphere.ElemType)

#%%
# .. note::
#     Most functions in :mod:`~mymesh.converter` take as inputs :code:`NodeCoords`,
#     :code:`NodeConn` and return updated versions of both.
#
#     .. code::
#
#         NewCoords, NewConn = converter.solid2tets(NodeCoords, NodeConn)
#         NewMesh = mesh(NewCoords, NewConn)
#
#     can be simplified as 
#
#     .. code::
#
#         NewMesh = mesh(*converter.solid2tets(NodeCoords, NodeConn))

# %%
# Element Type
# ------------
# This mixed-element mesh can be converted to a purely tetrahedral mesh with
# :func:`~mymesh.converter.solid2tets`. 
# Mixed-element surface meshes can be similarly converted to triangular meshes 
# with :func:`~mymesh.converter.surf2tris`. 

tet_sphere = mesh(*converter.solid2tets(sphere.NodeCoords, sphere.NodeConn), verbose=False)
print(tet_sphere.ElemType)
# %%
# Mesh Type
# ---------
# This mixed-element mesh can be converted to a purely tetrahedral mesh with
# :func:`~mymesh.converter.solid2surface`.

surf_sphere = mesh(sphere.NodeCoords, converter.solid2surface(sphere.NodeCoords, sphere.NodeConn), verbose=False)
print(surf_sphere.ElemType)

# %%
# Or equivalently, the :attr:`~mymesh.mesh.mesh.Surface` property can be used

surf_sphere = sphere.Surface
print(surf_sphere.ElemType)

# %%
# Element Order
# -------------
# Most meshes and functions in :mod:`mymesh` use first-order (or "linear") 
# elements (e.g. 4 node tetrahedra, 8 node hexahedra) by default, but for some 
# types of simulations, such as finite element methods, second-order (or 
# "quadratic") elements (e.g. 10 node tetrahedra, 20 node hexahedra) are 
# sometimes preferable for improved accuracy in simulations.
# This conversion can be achieved with :func:`~mymesh.converter.linear2quadratic`.
# See also :ref:`Element Types`.
quadratic_sphere = mesh(*converter.linear2quadratic(sphere.NodeCoords, sphere.NodeConn), verbose=False)
print(quadratic_sphere.ElemType)

# %%
# The reverse operation can be performed with :func:`~mymesh.converter.quadratic2linear`. 
# This function removes the excess nodes from the node connectivity, but keeps
# them in :code:`NodeCoords`, so any data associate with the nodes will still 
# match with the mesh.
# To remove the excess nodes, you can use :meth:`~mymesh.mesh.mesh.cleanup` or 
# :func:`mymesh.utils.RemoveNodes`.

linear_sphere = mesh(*converter.quadratic2linear(quadratic_sphere.NodeCoords, quadratic_sphere.NodeConn), verbose=False)
print(linear_sphere.ElemType)
# %%
