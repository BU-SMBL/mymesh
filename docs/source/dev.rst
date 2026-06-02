Development
===========

.. toctree::
    :maxdepth: 2
    :hidden:
    
    dev/CHANGELOG
    dev/dev_guide

MyMesh is developed and maintained by Tim Josephson. 
This project was originally developed to support academic research of orthopaedic biomechanics and mechanobiology, which heavily involves mesh-dependent computational simulations, but is a general-purpose library that can be used in many applications. 
If you find MyMesh useful (or not useful), or have any questions or feature requests, I'd love to hear from you.


Roadmap & Planned features
--------------------------

Current areas of development
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Improved Delaunay triangulation/tetrahedralization (and constrained Delaunay)
- Speed enhancements for contouring algorithms
- Improved implicit surface reconstruction capabilities

Ongoing
^^^^^^^
- Expansion of documentation, especially the :doc:`theory` and :doc:`perf` evaluations
- Improving test coverage

Plans for the future
^^^^^^^^^^^^^^^^^^^^
- Create `interpolate` module for interpolation within meshes and mapping between meshes
- Explore options for parallelization and/or GPU acceleration