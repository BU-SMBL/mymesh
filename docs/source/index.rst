.. Mesh documentation master file, created by
   sphinx-quickstart on Tue Nov 28 22:50:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MyMesh: General purpose, implicit, and image-based meshing in python
====================================================================
**Documentation Build Date**: |today| **Version**: |release|

.. toctree::
   :maxdepth: 2
   :hidden:

   guide
   api
   theory
   examples/index
   dev
   verify


.. grid:: 1 1 3 3

    .. grid-item-card::

        :octicon:`question` User Guide
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The user guide has information on getting started with MyMesh, 
        an overview of how to use MyMesh, and demos & examples that highlight 
        some of the key features.

        +++

        .. button-ref:: guide
            :expand:
            :color: primary
            :click-parent:

            To the user guide

    .. grid-item-card::

        :octicon:`terminal` API Reference 
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The API reference has detailed descriptions of all objects and functions 
        contained in the MyMesh library. 

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide 
    
    .. grid-item-card::

        :octicon:`repo` Theory Guide
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        The theory guide explains the details of the algorithms and code for a
        variety of functions in the MyMesh library.

        +++

        .. button-ref:: theory
            :expand:
            :color: info
            :click-parent:

            To the theory guide 

Summary
-------
MyMesh is a general purpose toolbox for generating, manipulating, and analyzing 
meshes. It's particularly focused on :mod:`~mymesh.implicit` function and
:mod:`~mymesh.image`-based meshing, with other functionality including:

- Mesh type :ref:`conversion<mymesh.converter>` (e.g. volume to surface, hexahedral or mixed-element to tetrahedral),
- Mesh :mod:`~mymesh.quality` evaluation and :mod:`~mymesh.improvement`,
- Mesh :mod:`~mymesh.curvature` analysis,
- Mesh :ref:`boolean<mymesh.booleans>` operations (intersection, union, difference).

MyMesh was originally developed in support of research within the Skeletal Mechanobiology and Biomechanics Lab at Boston University. 
It was used extensively in the scaffold design optimization research by :cite:t:`Josephson2024b` and is currently being used in various ongoing projects, including vertebral modeling, hip fracture modeling, growth modeling of skeletal tissue, and analysis of objects imaged using micro-computed tomography (μCT). 
MyMesh has proven useful in a variety of research applications, well beyond those that inspired its original development, and we expect it to remain a valuable tool in future research efforts. 

Statement of need
-----------------

Mesh-based representations of geometries are essential in a wide variety of research applications, and as such, there is a need for robust, efficient, and easy-to-use software for creating, analyzing, and manipulating meshes.
There are a variety of software packages for working with and generating meshes. 
Some are general purpose, like `CGAL <https://www.cgal.org/>`_, `VTK <https://vtk.org/>`_ :cite:p:`Schroeder2006`, and Gmsh :cite:p:`Geuzaine2009`, while others are more focused on specific tasks, such as triangular or tetrahedral mesh generation (e.g. Triangle :cite:p:`Shewchuk1996` and TetGen :cite:p:`Si2015`, respectively). 
In Python, most meshing packages depend on (or are direct wrappers to) one or more of these libraries, such as `PyVista <https://pyvista.org/>`_ :cite:p:`Sullivan2019` (a pythonic interface to VTK), `MeshPy <https://github.com/inducer/meshpy>`_ (which interfaces to Triangle and TetGen), and `PyMesh <https://pymesh.readthedocs.io/en/latest/>`_ (which depends on CGAL, Triangle, TetGen, and others). 
While these interfaces are useful and provide access to powerful mesh generation tools, their reliance on external dependencies can make them less easy to use and limit code readability, making it more difficult to understand how the code works. 
`TriMesh <https://trimesh.org/>`_ stands out as a capable, pure-Python library focused on triangular surface meshes, but it isn't intended for use with quadrilateral, mixed-element, or volumetric meshes. 
There is thus a need for a full-featured, accessible, and easy to use Python package for creating and working with meshes.

MyMesh strives to meet this need as a library of meshing tools, written in Python, with clear documentation that makes it both easy to use and easy to understand.
MyMesh has a particular focus on implicit function and image-based meshes, but also supplies a wide variety of general purpose tools. 
Rather than wrapping other libraries, algorithms are implemented from scratch, often based on or inspired by published algorithms and research. 
By providing an easily usable interface to both high-level and low-level functionality, we hope to provide both complete solutions and a set of building blocks for the development of other mesh-related tools.

.. Note::
    MyMesh is intended for research purposes. Any uses of MyMesh should
    be validated and verified appropriately. 

Examples
--------

See the gallery of :ref:`examples <Mesh Analysis Examples>` to see some of the ways MyMesh can be used.

Acknowledgements
----------------
This work was developed with funding support from the National Institutes of Health (Grant #AG073671). 
We are additionally grateful to all of the users who have tested the code, reported bugs, requested features, and provided feedback which has been vital to the development of MyMesh.

Colors used throughout this documentation are based on the 
`Nord Theme <https://www.nordtheme.com/>`_