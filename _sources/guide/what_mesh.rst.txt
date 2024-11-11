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
MyMesh considers three main :func:`Type <mymesh.utils.identify_type>`\ s of mesh and
several sub-types.

Line Meshes (``Type='line'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Line meshes are made up of edge elements. These meshes could represent a 1D 
mesh (e.g. a series of springs), the outer boundary of an open surface mesh,
or the wireframe of a volumetric mesh. 

Surface Meshes (``Type='surf'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Surface meshes are compos of surface elements (namely ``tri``\ s and ``quad``\ s), 
including both 2D planar meshes and 3D surfaces. 

2D Planar Meshes
""""""""""""""""
2D planar meshes exist in a plane (most commonly the x-y plane) such as a mesh
based on a 2D image. They contain both interior elements and elements with 
boundary edges.

3D Surfaces
"""""""""""
3D surfaces consist of 2D elements but exist within a three dimensional space. 
These surfaces can either be open (with exposed edges) or closed. 


Volume Meshes (``Type='vol'``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Volumetric meshes are made of three dimensional elements such as tetrahedra 
or hexahedra.

Voxel Meshes
""""""""""""
Voxel meshes are a special case of hexahedral meshes consisting of uniform 
cubic or rectangular elements that arise from three dimensional images where 
each voxel (the three dimensional, *vo*\ lumetric analog to a pixel) is converted
to an element. While a voxel mesh could be full grid of voxels, more commonly
the mesh will be thresholded to obtain a voxelized geometry. 


