# -*- coding: utf-8 -*-
# Created on Thu Mar 24 13:51:50 2022
# @author: toj
"""



Objects
=======
.. autosummary::
    :toctree: generated/

    .. currentmodule:: mymesh.mesh

    mesh


.. currentmodule:: mymesh

Submodules
===============
.. autosummary::
    :toctree: generated/

    booleans  
    contour
    converter
    curvature
    delaunay
    image
    implicit
    improvement
    octree
    primitives
    quality
    rays
    utils
    visualize

"""
from functools import wraps

try: 
    from numba import njit
    _MYMESH_USE_NUMBA = True
except ImportError:
    njit = None
    _MYMESH_USE_NUMBA = False

def use_numba(enabled=True):
    global _MYMESH_USE_NUMBA
    if njit is None:
        warnings.warn('numba is not available for import. Install with `conda install numba` or `pip install numba`.')
    _MYMESH_USE_NUMBA = enabled and (njit is not None)

def check_numba():
    global _MYMESH_USE_NUMBA
    if _MYMESH_USE_NUMBA and (njit is not None):
        check = True
    else:
        check = False
    return check

def try_njit(func=None, *njit_args, **njit_kwargs):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _MYMESH_USE_NUMBA:
                jit_func = njit(*njit_args, **njit_kwargs)(func)
                return jit_func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    
    return decorator(func) if func else decorator

from .mesh import mesh
from . import booleans, contour, curvature, delaunay, image, implicit, improvement, octree, primitives, quality, utils, visualize
__all__ = ["use_numba", "try_njit", "mesh", "booleans", "contour", "curvature",
"delaunay", "image", "implicit", "improvement", "octree", "primitives", 
"quality", "utils", "visualize", "_MYMESH_USE_NUMBA"]