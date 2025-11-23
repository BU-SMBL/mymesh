Getting Started
===============

MyMesh is a Python package that can be used to write python scripts are be used interactively, for example in Jupyter Notebooks or other IPython environments.
It can also be used in Matlab through Matlab's Python interface (see :ref:`Using MyMesh in MATLAB`).

What do you want to do?
-----------------------
There are a lot of different things you can do with MyMesh, depending on what 
you're trying to do, there are different places to start.

.. tabs::

    .. tab:: Create

        What do you want to create a mesh from?

        See also :ref:`Mesh Generation Examples`.

        .. tabs::

            .. tab:: Function

                Functions, specifically :ref:`implicit functions<What is an implicit function?>`, can be turned into meshes using the 
                :mod:`~mymesh.implicit` module.

                A few pre-defined implicit functions are available in 
                :mod:`~mymesh.implicit`, such as :func:`~mymesh.implicit.sphere`, :func:`~mymesh.implicit.torus`, and triply periodic
                minimal surfaces like :func:`~mymesh.implicit.gyroid`.

                See the user guide on :ref:`Implicit Meshing` for further 
                explanation of what implicit functions are and how to pre-defined them, and the implicit mesh generation tools
                available in the :mod:`~mymesh.implicit` module: 
                :func:`~mymesh.implicit.VoxelMesh`, :func:`~mymesh.implicit.SurfaceMesh`, :func:`~mymesh.implicit.TetMesh`.

            .. tab:: Image
                
                Both :ref:`2D and 3D images<Images in MyMesh>` can be converted 
                into meshes using the :mod:`~mymesh.image` module.
                
            .. tab:: Points
                
                Point clouds can be triangulated/tetrahedralized with the 
                :mod:`~mymesh.delaunay` module. The convex hull and alpha shapes
                (concave hulls) can be by identified with 
                :func:`mymesh.delaunay.ConvexHull`/:func:`mymesh.delaunay.AlphaShape`.

                Oriented points (those with normal vectors associated with them)
                can be reconstructed into an implicit function using 
                :func:`mymesh.implicit.SurfaceReconstruction`.

            .. tab:: Nothing!

                If you're starting from scratch, a number of options are 
                available. You can start with predefined shapes in the 
                :mod:`~mymesh.primitives` module, including spheres, boxes, cylinders, or use demo models from :func:`mymesh.demo_mesh`.
                From there, you can use 
                :ref:`explicit mesh boolean<Explicit CSG>` operations to make
                more complex shapes from simple shapes.

                You can also use sweep construction methods like 
                :func:`mymesh.primitives.Revolve` and 
                :func:`mymesh.primitives.Extrude` to build up meshes from
                1D to 2D and 2D to 3D.
            
            .. tab:: CAD

                If you have a model designed in a computer aided design (CAD) software, most softwares have features to export models as STL files (which store the geometry as a triangular surface mesh). 
                You can then :meth:`~mymesh.mesh.mesh.read` the STL file to 
                work with it in MyMesh. 
                
                MyMesh doesn't currently have any capabilities to work with 
                other CAD files like .step or .iges files.


    .. tab:: Analyze

        See also :ref:`Mesh Analysis Examples`.

        .. tabs::

            .. tab:: Geometry

                There are many geometric properties that can be calculated from
                meshes.
                The :class:`~mymesh.mesh` object can calculate many of these 
                on-demand, for example surface normal vectors (:attr:`~mymesh.mesh.mesh.ElemNormals`, :attr:`~mymesh.mesh.mesh.NodeNormals`), bounding boxes (:attr:`~mymesh.mesh.mesh.aabb`, :attr:`~mymesh.mesh.mesh.mvbb`). Additionally, the :mod:`~mymesh.curvature` module can be used to calculate surface curvatures in several different ways (see also the :ref:`Curvature Analysis` examples and the :ref:`Curvature` theory guide). 
            
            .. tab:: Element quality

                Maintaining high element quality is essential in many mesh-based simulations in order to obtain accurate results.
                The :mod:`~mymesh.quality` module contains functions to calculate a variety of different mesh quality metrics. 

            .. tab:: Connectivity

                In addition to the fundamental connectivity of nodes into elements, there is other :ref:`connectivity information <Connectivity Representations>` that can be useful in various situations, such as the nodes neighboring each node (:attr:`~mymesh.mesh.mesh.NodeNeighbors`), the elements connected to each node (:attr:`~mymesh.mesh.mesh.ElemConn`), etc. 

                In meshes that may contain multiple, disconnected regions, it can also be useful to identify the connected nodes (:func:`mymesh.utils.getConnectedNodes`) or connected elements (:func:`mymesh.utils.getConnectedElements`).

    .. tab:: Modify

        See also :ref:`Mesh Modification Examples`.

        .. tabs::

            .. tab:: Quality improvement

                The :mod:`~mymesh.improvement` module has various tools that can be to improve mesh quality, such as smoothing (e.g. :func:`~mymesh.improvement.LocalLaplacianSmoothing`, :func:`~mymesh.improvement.TaubinSmoothing`) and edge contraction/coarsening (:func:`~mymesh.improvement.Contract`) 

            .. tab:: Conversion

                Sometimes it can be useful to convert meshes from one form to another, such as extracting the surface from a volumetric mesh, converting a hexahedral mesh to a tetrahedral mesh, or converting first-order (linear) elements to second-order (quadratic) elements (or vice versa).
                The :mod:`~mymesh.converter` module has a variety of functions for performing such conversions. 
                the :class:`~mymesh.mesh.mesh` class also has attributes to access the :attr:`~mymesh.mesh.mesh.Surface` or :attr:`~mymesh.mesh.mesh.Boundary` representations of meshes.


            .. tab:: Thresholding/Contouring/Cropping

                If you have an existing :class:`~mymesh.mesh.mesh` with values associated with the nodes and/or elements of the mesh, you can :meth:`~mymesh.mesh.mesh.Threshold` or :meth:`~mymesh.mesh.mesh.Contour` the mesh based on those values. 
                You can also :meth:`~mymesh.mesh.mesh.Crop` the a mesh by specifying bounds or :meth:`~mymesh.mesh.mesh.Clip` the mesh by a plane.

            .. tab:: Registration/Alignment

                If you have two representations of an object, or two similar objects, you can perform registration to align them with the :mod:`~mymesh.register` module. 
                
