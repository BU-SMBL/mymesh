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
==========
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

Functions
=========
.. autosummary::
    :toctree: generated/

    demo_mesh  
    demo_image

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
    Example image data from online databases

    Since data is obtained on-demand from online sources, internet connectivity 
    is required.

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

def demo_mesh(name='bunny'):
    """
    Example mesh models from online databases
    
    Since data is obtained on-demand from online sources, internet connectivity 
    is required.

    The Stanford models (bunny, dragon, Lucy) are obtained from the 
    `Stanford Computer Graphics Library <https://graphics.stanford.edu/data/3Dscanrep/>`__.
    The following notices from the library's website apply to these models:

        Please be sure to acknowledge the source of the data and models you take 
        from this repository. In each of the listings below, we have cited the 
        source of the range data and reconstructed models. You are welcome to 
        use the data and models for research purposes. You are also welcome to 
        mirror or redistribute them for free. Finally, you may publish images 
        made using these models, or the images on this web site, in a scholarly
        article or book - as long as credit is given to the Stanford Computer 
        Graphics Laboratory. However, such models or images are not to be used 
        for commercial purposes, nor should they appear in a product for sale 
        (with the exception of scholarly journals or books), without our 
        permission. 

        As you browse this repository and think about how you might use our 3D 
        models and range datasets, please remember that several of these 
        artifacts have religious or cultural significance. Aside from the buddha, 
        which is a religious symbol revered by hundreds of millions of people, 
        the dragon is a symbol of Chinese culture, the Thai statue contains 
        elements of religious significance to Hindus, and Lucy is a Christian 
        angel; statues like her are commonly seen in Italian churches. Keep your 
        renderings and other uses of these particular models in good taste. 
        Don't animate or morph them, don't apply Boolean operators to them, and 
        don't simulate nasty things happening to them (like breaking, exploding, 
        melting, etc.). Choose another model for these sorts of experiments. 
        (You can do anything you want to the Stanford bunny or the armadillo.) 
    

    Parameters
    ----------
    name : str, optional
        Name of the image to access, by default 'bunny'.
        Available options are:

        - "bunny" - Stanford Bunny from the Stanford 3D Scanning Repository.
            - Coarser versions of the model are available as "bunny_res2", "bunny_res3", "bunny_res4"
            - This is a non-manifold surface, the mesh has several holes on the bottom

        - "dragon" - Stanford Dragon from the Stanford 3D Scanning Repository 
            - Coarser versions of the model are available as "dragon_res2", "dragon_res3", "dragon_res4"
            - This is a non-manifold surface, the mesh has numerous small holes

        - "Lucy" - "Angel of Light" statue from the Stanford 3D Scanning Repository
            - This surface is hole free but has some topological defects/bridging
    Returns
    -------
    M : mymesh.mesh
        Mesh object

    Examples
    --------

    .. plot::

        import mymesh
        bunny = mymesh.demo_mesh("bunny")
        bunny.plot(view='xy')

    .. plot::

        import mymesh
        dragon = mymesh.demo_mesh("dragon")
        dragon.plot(view='xy')

    .. plot::

        import mymesh
        lucy = mymesh.demo_mesh("Lucy")
        lucy.plot(view='-xz')

    """    
    try:
        import meshio
    except:
        raise ImportError('meshio library is required to load demo meshes. Install with: pip install meshio')
        
    if 'bunny' in name:
        # "Stanford Bunny" from the Stanford 3D Scanning Repository
        # https://graphics.stanford.edu/data/3Dscanrep/
        url = 'https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'

        # Get data and extract archive
        response = urllib.request.urlopen(url)
        tar_bytes = response.read()
        tar_file = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")

        if 'res2' in name:
            f = tar_file.extractfile('bunny/reconstruction/bun_zipper_res2.ply')
        elif 'res3' in name:
            f = tar_file.extractfile('bunny/reconstruction/bun_zipper_res3.ply')
        elif 'res4' in name:
            f = tar_file.extractfile('bunny/reconstruction/bun_zipper_res4.ply')
        else:
            f = tar_file.extractfile('bunny/reconstruction/bun_zipper.ply')

        m = meshio.ply._ply.read_buffer(f)
        M = mesh.meshio2mymesh(m)
    elif 'dragon' in name:
        # "Stanford Dragon" from the Stanford 3D Scanning Repository
        # https://graphics.stanford.edu/data/3Dscanrep/
        url = 'https://graphics.stanford.edu/pub/3Dscanrep/dragon/dragon_recon.tar.gz'

        # Get data and extract archive
        response = urllib.request.urlopen(url)
        tar_bytes = response.read()
        tar_file = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")

        if 'res2' in name:
            f = tar_file.extractfile('dragon_recon/dragon_vrip_res2.ply')
        elif 'res3' in name:
            f = tar_file.extractfile('dragon_recon/dragon_vrip_res3.ply')
        elif 'res4' in name:
            f = tar_file.extractfile('dragon_recon/dragon_vrip_res4.ply')
        else:
            f = tar_file.extractfile('dragon_recon/dragon_vrip.ply')

        m = meshio.ply._ply.read_buffer(f)
        M = mesh.meshio2mymesh(m)

    elif 'Lucy' in name:
        # Lucy statue from the Stanford 3D Scanning Repository
        # https://graphics.stanford.edu/data/3Dscanrep/
        url = 'https://graphics.stanford.edu/data/3Dscanrep/lucy.tar.gz'

        # Get data and extract archive
        response = urllib.request.urlopen(url)
        tar_bytes = response.read()
        tar_file = tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz")

        f = tar_file.extractfile('lucy.ply')

        m = meshio.ply._ply.read_buffer(f)
        M = mesh.meshio2mymesh(m)
    else:
        raise ValueError(f'Unknown image option: {name:s}')

    
    return M

from .mesh import mesh
from . import booleans, contour, curvature, delaunay, image, implicit, improvement, tree, primitives, quality, utils, visualize
__all__ = ["check_numba", "use_numba", "try_njit", "mesh", "booleans", "contour", "converter",
"curvature", "delaunay", "image", "implicit", "improvement", "tree", 
"primitives", "quality", "rays", "utils", "visualize"]