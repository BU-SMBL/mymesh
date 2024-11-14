User Guide
==========

.. toctree::
    :maxdepth: 2
    :hidden:
    
    guide/installation
    guide/what_mesh
    guide/mesh_class
    guide/elem_types
    guide/connectivity
    guide/implicit_meshing
    guide/image_meshing
    guide/csg
    guide/io
    guide/ref

Package Overview
----------------

MyMesh has a collection of algorithms for generation, manipulation, and analysis
of 2D and 3D meshes. These capabilities are organized into the following 
submodules:

.. autosummary:: 
    :nosignatures:
    
    ~mymesh.booleans
    ~mymesh.contour
    ~mymesh.converter
    ~mymesh.curvature
    ~mymesh.delaunay
    ~mymesh.image
    ~mymesh.implicit
    ~mymesh.improvement
    ~mymesh.octree
    ~mymesh.primitives
    ~mymesh.quality
    ~mymesh.rays
    ~mymesh.utils
    ~mymesh.visualize