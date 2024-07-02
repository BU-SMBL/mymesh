.. Mesh documentation master file, created by
   sphinx-quickstart on Tue Nov 28 22:50:25 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MyMesh documentation
====================
**Documentation Build Date**: |today| **Version**: |release|

MyMesh is a general purpose toolbox for generating, manipulating, and analyzing 
meshes for finite element, finite difference, or finite volume simulations. It 
has particular focuses on implicit function and image-based mesh generation.

MyMesh was originally developed in support of the Ph.D. research of 
`Tim Josephson <https://scholar.google.com/citations?user=ZsqbtjQAAAAJ&hl=en>`_ 
in `Elise Morgan <https://scholar.google.com/citations?user=hLf0lzEAAAAJ&hl=en&oi=ao>`_'s 
`Skeletal Mechanobiology and Biomechanics Lab <https://morganresearchlab.org/>`_ 
at Boston University.

Statement of Need
-----------------
There are two main goals behind the MyMesh package - one technical, and one 
philosophical. 

The technical goal is to have a mesh toolbox for generating,
manipulating, and analyzing implicit function and image-based meshes with
a robust set of general purpose tools. The initial driving force for this was
to support computational simulations and generative design for additive 
manufacturing of bone tissue engineering scaffolds, though it has applications
well beyond that specific one. MyMesh features tools for creating voxel, surface,
and tetrahedral meshes of both images and implicit functions. Additionally,
functions are available for calculating surface curvatures using both mesh-based
:cite:p:`Goldfeather2004` and analytical approaches, the latter of which is 
generally superior when images/functional representations are available. To 
our knowledge, such an approach is not available in any existing software packages.

The philosophical goal is...

There are many available software packages for working with and generating 
meshes. Some are general purpose, like CGAL, VTK, and gmsh, and others are more
focused and do specific tasks very well, such as triangular (Triangle
:cite:p:`Shewchuk1996`) or tetrahedral (TetGen :cite:p:`Si2015`) mesh generation. 
In Python, most meshing packages depend on (or are direct wrappers to) one or more 
of these libraries. Of these, PyVista is an excellent example of a well 
documented pythonic interface to VTK. 

.. While it's best not to reinvent the wheel, sometimes re-implementing the wheel
.. can help to understand it, improve it, or make something new. 


.. toctree::
   :maxdepth: 2
   :hidden:

   guide
   api
   theory
   dev


.. grid:: 2

    .. grid-item-card::

        User guide
        ^^^^^^^^^^^^^

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

        API reference
        ^^^^^^^^^^^^^

        The API reference has detailed descriptions of all objects and functions 
        contained in the MyMesh package.

        +++

        .. button-ref:: api
            :expand:
            :color: secondary
            :click-parent:

            To the reference guide