Getting Started
===============


What do you want to do?
-----------------------
There are a lot of different things you can do with MyMesh, depending on what 
you're trying to do, there are different places to start.

.. tabs::

    .. tab:: Create

        What do you want to create a mesh from?

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
                :mod:`~mymesh.primitives` module, including spheres, boxes, cylinders.
                From there, you can use 
                :ref:`explicit mesh boolean<Explicit CSG>` operations to make
                more complex shapes from simple shapes.

                You can also use sweep construction methods like 
                :func:`mymesh.primitives.Revolve` and 
                :func:`mymesh.primitives.Extrude` to build up meshes from
                1D to 2D and 2D to 3D.



    .. tab:: Modify/Manipulate

        .. tabs::

            .. tab:: Quality improvement

            .. tab:: Conversion

            .. tab:: Thresholding/Contouring/Cropping


    .. tab:: Evaluate

