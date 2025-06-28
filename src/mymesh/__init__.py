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
    tree
    primitives
    quality
    rays
    utils
    visualize

"""
from functools import wraps
import warnings, urllib, tarfile, io, re
import numpy as np

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
    @wraps(func)
    def decorator(func):
        if check_numba():
            jit_func = njit(*njit_args, **njit_kwargs)(func)
        else:
            jit_func = func
        
        return jit_func
    
    return decorator(func) if func else decorator

def demo_image(name='bunny', normalize=True):
    """
    Access example image data. Data is obtained from online sources, requires
    internet connectivity.

    Parameters
    ----------
    name : str, optional
        Name of the image to access, by default 'bunny'.
        Available options are:

        - "bunny" - CT scan of the Stanford Bunny from the Stanford volume data archive

    normalize : bool, optional
        Normalize image data to the range 0-255 in uint8 format, by default True

    Returns
    -------
    img : np.ndarray
        Image array

    """    

    if name == 'bunny':
        # CT scan of "Stanford Bunny" from the Stanford volume data archive
        # https://graphics.stanford.edu/data/voldata/voldata.html
        url = 'https://graphics.stanford.edu/data/voldata/bunny-ctscan.tar.gz'

        # Get data and extract archive
        response = urllib.request.urlopen(url)
        tar_bytes = response.read()
        tar_file = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")

        # Parse file names
        file_names = np.array([name for name in tar_file.getnames() if re.match('bunny/[0-9]', name)], dtype=object)
        file_numbers = [int(name.split('/')[1]) for name in file_names]
        file_names = file_names[np.argsort(file_numbers)]

        # Load image data - "The data is raw 512x512 slices, unsigned, 12 bit data stored as 16bit (2-byte) pixels."
        # Binary data stored in big-endian ">u2" format
        img = np.array([np.frombuffer(tar_file.extractfile(file).read(),dtype='>u2').reshape((512,512)) for file in file_names])
        img[img == np.max(img)] = 0 # Set the outer boundary to 0 ("black")
    else:
        raise ValueError(f'Unknown image option: {name:s}')

    
    if normalize:
        # normalize the image to 0-255, unit8
        img = (img/np.max(img)*255).astype(dtype=np.uint8)

    return img

from .mesh import mesh
from . import booleans, contour, curvature, delaunay, image, implicit, improvement, tree, primitives, quality, utils, visualize
__all__ = ["check_numba", "use_numba", "try_njit", "mesh", "booleans", "contour", "converter",
"curvature", "delaunay", "image", "implicit", "improvement", "tree", 
"primitives", "quality", "rays", "utils", "visualize"]