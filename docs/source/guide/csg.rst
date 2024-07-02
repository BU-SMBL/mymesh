Constructive Solid Geometry
===========================
Constructive solid geometry (CSG) is the process of combining a set of initial, 
often simple geometries into more complex shapes through the use of boolean
operations such as unions, intersections, and differences :cite:p:`Laidlaw1986`. 
In computer graphics applications, this may involve ray tracing or other 
approaches for efficient visualization, but MyMesh is primarily concerned with 
generating and/or modifying meshes. MyMesh offers two approaches to CSG: 
implicit and explicit. 

For implicit CSG, boolean operations are performed on implicit functions or a 
set of values evaluated at points in a grid or mesh (e.g. image data). Performing
boolean operations in this way is quite efficient, but a contouring step is 
required to generate a mesh, which may result in a loss of detail at sharp 
features or interfaces. In MyMesh, implicit CSG operations rely on 
:mod:`mymesh.implicit` and :mod:`mymesh.contour`.

Explicit CSG operates directly on existing meshes, rather than functions or 
values. This involves calculating intersections between meshes (utilizing 
:mod:`mymesh.rays` and :mod:`mymesh.octree`) and then splitting and joining 
elements to create the new mesh. These operations are more computationally 
demanding and generally slower than implicit CSG, especially for large meshes, 
but and can be used when no functional representation of a mesh exists. 
Floating point errors in the identification of intersections and
splitting of elements can result in mesh defects and unclosed surfaces, which
may be problematic for some applications. If performing explicit CSG on surface
meshes with the aim of producing models that require volumetric meshes, 
`fTetWild <https://github.com/wildmeshing/fTetWild>`_ may be useful for 
generating high quality tetrahedral meshes from imperfect surfaces 
:cite:p:`Hu2020`. Explicit CSG mesh boolean functions can be found in 
:mod:`mymesh.booleans`. 

Examples
--------