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

MyMesh was originally developed in support of the Ph.D. research of 
`Tim Josephson <https://scholar.google.com/citations?user=ZsqbtjQAAAAJ&hl=en>`_ 
in `Elise Morgan <https://scholar.google.com/citations?user=hLf0lzEAAAAJ&hl=en&oi=ao>`_'s 
`Skeletal Mechanobiology and Biomechanics Lab <https://morganresearchlab.org/>`_ 
at Boston University. MyMesh was used extensively in the scaffold design 
optimization research by :cite:t:`Josephson2024b` and is currently 
being used in  various ongoing projects at Boston University, including vertebral 
modeling (Andre Gutierrez Marty, Neilesh Frings), hip fracture modeling (Joshua 
Auger, Ariella Blake), mechanobiologically-driven growth modeling of skeletal 
tissue (Tim Josephson, Vivian Shi), and analysis of micro CT-scanned teeth 
(Sydney Holder, Shadi Mohebi). 

Statement of need
-----------------
.. --
.. There are two main goals behind the MyMesh package - one technical, and one 
.. philosophical. 

.. The technical goal is to have a mesh toolbox for generating,
.. manipulating, and analyzing implicit function and image-based meshes with
.. a robust set of general purpose tools. The initial driving force for this was
.. to support computational simulations and generative design for additive 
.. manufacturing of bone tissue engineering scaffolds, though it has applications
.. well beyond that specific one. MyMesh features tools for creating voxel, surface,
.. and tetrahedral meshes of both images and implicit functions. Additionally,
.. functions are available for calculating surface curvatures using both mesh-based
.. :cite:p:`Goldfeather2004` and analytical approaches, the latter of which is 
.. generally superior when images/functional representations are available. To 
.. our knowledge, such an approach is not available in any existing software packages.

.. The philosophical goal is...

There are a variety of software packages for working with and generating 
meshes. Some are general purpose, like CGAL, VTK, and gmsh, and others are more
focused and do specific tasks very well, such as triangular (Triangle
:cite:p:`Shewchuk1996`) or tetrahedral (TetGen :cite:p:`Si2015`) mesh generation. 
In Python, most meshing packages depend on (or are direct wrappers to) one or more 
of these libraries, such as PyVista (a pythonic interface to VTK), PyMesh (which
depends on CGAL, Triangle, TetGen, and others), and MeshPy (which interfaces to 
Triangle and TetGen). While these interfaces are useful and provide access to 
powerful mesh generation tools, their reliance on external, compiled dependencies 
limits code readability and makes it difficult to build upon and extend the 
algorithms. A notable exception is TriMesh, a pure-python library focused on 
triangular surfaces meshes. 

MyMesh strives to be a comprehensive library of meshing tools, written in Python
with clear documentation that makes it both easy to use and easy to understand.
MyMesh has a particular focus on implicit function and image-based meshes, but 
also supplies a wide variety of general purpose meshing tools. Rather than wrapping 
other libraries, algorithms are implemented from scratch, either directly based 
on or loosely inspired by published algorithms and research. 

As we aim to contribute to the existing community of meshing software, MyMesh 
was designed to allow for easy extraction of key mesh information that can be 
passed to other libraries' data structures in python or written to various file 
types using meshio.

.. Note::
    MyMesh is intended for research purposes. Any applications of MyMesh should
    be should be validated and verified appropriately. 

Acknowledgements
----------------
MyMesh was initially developed to further the aims of research funded by the 
National Institutes of Health (Grant #AG073671). 

Colors used throughout this documentation are based on the 
`Nord Theme <https://www.nordtheme.com/>`_