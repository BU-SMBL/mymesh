Development
===========
MyMesh is individually developed and maintained by me, Tim Josephson. 
This project was originally, and continues to be, developed to support my 
academic research of orthopaedic biomechanics and mechanobiology, which heavily
involves mesh-dependent computational simulations. If you find MyMesh useful 
(or not useful), or have any questions or feature requests, I'd love to hear 
from you. 


Roadmap & Planned features
--------------------------

- Expansion of the :doc:`theory` 
- Improved Delaunay triangulation/tetraheralization (and constrained Delaunay)
- More sophisticated mesh2sdf/implicit surface reconstruction capabilities
- Create `interpolate` module for interpolation within meshes and mapping between meshes
- Create `register` module for mesh and image registration
- 
    Explore and possibly provide options for substituting CuPy for NumPy to 
    enable Nvidia CUDA-based GPU computation. Since MyMesh is heavily dependent
    on NumPy for vectorized computation, CuPy could allow for dramatic performance
    enhancements across the board.