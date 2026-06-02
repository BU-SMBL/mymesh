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
   perf
   dev


.. grid:: 1 1 2 4

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

        :octicon:`rocket` Examples
        ^^^^^^^^^^^^^^^^^^^^^^^^

        Examples that highlight features and demonstrate how to use MyMesh.
        
        +++

        .. button-ref:: examples/index
            :expand:
            :color: warning
            :click-parent:

            To the example gallery
    
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

News
----

.. grid:: 1 

    .. grid-item-card::

        :octicon:`file` Paper Published in JOSS
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        Check out the new paper about MyMesh in the Journal of Open Source Software!

        +++

        .. button-link:: https://joss.theoj.org/papers/10.21105/joss.10003
            :expand:
            :color: primary
            :click-parent:

            MyMesh: General purpose, implicit, and image-based meshing in Python

Summary
-------
A mesh is a discrete representation that subdivides a geometry or computational domain into a collection of points (nodes) connected by simple shapes (elements).
Meshes are used for a variety of purposes, including simulations (e.g. finite element, finite volume, and finite difference methods), visualization and computer graphics, image analysis, and additive manufacturing.
`mymesh` is a general purpose set of tools for generating, manipulating, and analyzing meshes. 
`mymesh` is particularly focused on implicit function and image-based meshing, with other functionality including:

- geometric and curvature analysis,
- intersection and inclusion tests (e.g. ray-surface intersection and point-in-surface tests),
- mesh boolean operations (intersection, union, difference),
- sweep construction methods (extrusions, revolutions),
- point set, mesh, and image registration,
- mesh quality evaluation and improvement,
- mesh type conversion (e.g. volume to surface, hexahedral or mixed-element to tetrahedral, first-order elements to second-order elements).

.. Note::
    MyMesh is intended for research purposes. Any uses of MyMesh should
    be validated and verified appropriately. 

Examples
--------

See the gallery of :ref:`examples <Mesh Analysis Examples>` to see some of the ways MyMesh can be used.

Attribution
-----------
If you've found MyMesh useful in your work, please cite it by referring to the paper in the `Journal of Open Source Software <https://doi.org/10.21105/joss.10003>`_. 
You can also cite specific versions by referring to the `Zenodo archive <https://zenodo.org/records/17511909>`_.

.. admonition:: Citation

    Josephson & Morgan, (2026). MyMesh: General purpose, implicit, and image-based meshing in Python. Journal of Open Source Software, 11(122), 10003, https://doi.org/10.21105/joss.10003

Acknowledgements
----------------
This work was developed with funding support from the National Institutes of Health (Grant #AG073671). 
We are additionally grateful to all of the users who have tested the code, reported bugs, requested features, and provided feedback which has been vital to the development of MyMesh.

Colors used throughout this documentation are based on the 
`Nord Theme <https://www.nordtheme.com/>`_