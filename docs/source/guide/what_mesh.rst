What is a mesh?
===============
A mesh is a collection of points (*nodes*) and shapes (*elements*) that 
represent a larger geometry or computational domain. Meshes can be used for
a variety of purposes, including computational simulations (finite element, 
volume, and difference methods), computer graphics, image analysis, and additive 
manufacturing. 

In MyMesh, a mesh is defined primarily by the set of node coordinates 
(``NodeCoords``) and the set of node connectivities (``NodeConn``) which 
indicate the nodes that are connected to form each element. The elements are 
convex polygons or polyhedra, each defined by ordering nodes according to 
standard conventions. 

Mesh Types
----------
MyMesh considers three main :func:`Type <mymesh.utils.identify_type>`s of mesh and
several sub-types.

Surface Meshes (`Type='line'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Line meshes are made up of edge elements. These meshes could represent a 1D 
mesh (e.g. a series of springs), the outer boundary of an open surface mesh,
or the wireframe of a volumetric mesh. 

Surface Meshes (`Type='surf'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Surface meshes are comprised of surface elements (namely `tri`s and `quad`s), 
including both 2D planar meshes and 3D surfaces. 

2D Planar Meshes
""""""""""""""""

3D Surfaces
"""""""""""


Volume Meshes (`Type='vol'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Voxel Meshes
""""""""""""

Volu


