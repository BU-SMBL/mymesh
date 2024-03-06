# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 13:51:50 2022

@author: toj

.. currentmodule:: Mesh

Objects
=======
.. autosummary::
    :toctree: generated/

    mesh
 
Submodules
===============
.. autosummary::
    :toctree: generated/

    booleans        --- Mesh booleans
    contour
    converter
    curvature
    delaunay
    implicit
    improvement
    octree
    primitives
    quality
    rays
    utils
    visualize

"""
from .mesh import mesh
from . import booleans, contour, curvature, delaunay, implicit, improvement, octree, primitives, quality, TetGen, utils, visualize