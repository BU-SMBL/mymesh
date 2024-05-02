# -*- coding: utf-8 -*-
# Created on Tue Mar 12 13:26:02 2024
# @author: toj
"""
Image-based meshing tools


Image-based Mesh Generation
===========================
.. autosummary::
    :toctree: submodules/

    VoxelMesh
    SurfaceMesh
    TetMesh
    SurfaceNodeOptimization

File IO
=======
.. autosummary::
    :toctree: submodules/

    read
"""


import numpy as np
from scipy import interpolate, ndimage
import sys, os, copy, warnings, glob

from . import utils, converter, contour, quality, improvement, rays, octree, mesh, primitives

# Mesh generators
def VoxelMesh(img, h, threshold=None, threshold_direction=1, scalefactor=1, scaleorder=1, return_nodedata=False):
    """
    Generate voxel mesh of an image function

    Parameters
    ----------
    img : str or np.ndarray
        Image array or file path to an image
    h : scalar, tuple
        Element side length. Can be specified as a single scalar value, or a three element tuple (or array_like).
    threshold : scalar
        Isovalue threshold to use for keeping/removing elements, by default 0.
    threshold_direction : signed integer
        If threshold_direction is negative, values less than or equal to the threshold will be considered "inside" the mesh and the opposite if threshold_direction is positive, by default 1.
    scalefactor : float, optional
        Scale factor for resampling the image. If greater than 1, there will be more than
        1 elements per voxel. If less than 1, will coarsen the image, by default 1.
    scaleorder : int, optional
        Interpolation order for scaling the image (see scipy.ndimage.zoom), by default 1.
        Must be 0-5.
    return_nodedata : bool, optional
        Option to interpolate image data to the nodes rather than just the voxels, by default False.
        This can add significant computational costs for large images.

    Returns
    -------
    voxel : mymesh.mesh
        Mesh object containing the voxel mesh. The image data are stored in voxel.ElemData['Image Data'] and optional voxel.NodeData['Image Data']

        .. note:: Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = implicit.VoxelMesh(...)``

    Examples
    --------


    """        
    
    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)

    if return_nodedata:
        VoxelCoords, VoxelConn, VoxelData, NodeData = converter.im2voxel(img, h,scalefactor=scalefactor, scaleorder=scaleorder, threshold=threshold, threshold_direction=threshold_direction, return_nodedata=True)

        if 'mesh' in dir(mesh):
            voxel = mesh.mesh(VoxelCoords, VoxelConn)
        else:
            voxel = mesh(VoxelCoords, VoxelConn)

        voxel.NodeData['Image Data'] = NodeData
        voxel.ElemData['Image Data'] = VoxelData

    else:
        VoxelCoords, VoxelConn, VoxelData = converter.im2voxel(img, h,scalefactor=scalefactor, scaleorder=scaleorder, threshold=threshold, threshold_direction=threshold_direction)

        if 'mesh' in dir(mesh):
            voxel = mesh.mesh(VoxelCoords, VoxelConn)
        else:
            voxel = mesh(VoxelCoords, VoxelConn)

        voxel.ElemData['Image Data'] = VoxelData

    
    

    return voxel

def SurfaceMesh(img, h, threshold=None, threshold_direction=1, scalefactor=1, scaleorder=1, method='mc', interpolation='linear'):
    """
    Generate a surface mesh of an image function 

    Parameters
    ----------
    img : str or np.ndarray
        Image array or file path to an image
    h : scalar, tuple
        Element side length. Can be specified as a single scalar value, or a three element tuple (or array_like).
    threshold : scalar
        Isovalue threshold to use for keeping/removing elements, by default 0.
    threshold_dir : signed integer
        If threshold_dir is negative (default), values less than or equal to the threshold will be considered "inside" the mesh and the opposite if threshold_dir is positive, by default 1.
    scalefactor : float, optional
        Scale factor for resampling the image. If greater than 1, there will be more than
        1 elements per voxel. If less than 1, will coarsen the image, by default 1.
    scaleorder : int, optional
        Interpolation order for scaling the image (see scipy.ndimage.zoom), by default 1.
        Must be 0-5.
    method : str, optional
        Surface triangulation method, by default 'mc'.
        'mc' : Marching cubes (see contour.MarchingCubesImage) (default)

        'mc33' : Marching cubes 33 (see contour.MarchingCubes)

        'mt' : Marching tetrahedra (see contour.MarchingTetrahedra)
    interpolation : str, optional
        Method of interpolation used for placing the vertices on the approximated isosurface. This can be 'midpoint', 'linear', or 'cubic', by default 'linear'. If 'cubic' is selected, method is overridden to be 'mc'. 
    

    Returns
    -------
    surface : mymesh.mesh
        Mesh object containing the surface mesh.

        .. note:: Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = implicit.SurfaceMesh(...)``

    Examples
    --------
    .. plot::

        surface = implicit.SurfaceMesh(implicit.gyroid, [0,1,0,1,0,1], 0.05)
        surface.plot(bgcolor='w')
    """

    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)

    if np.sign(threshold_direction) == 1:
        flip = True
    else:
        flip = False

    img = read(img, scalefactor=scalefactor, scaleorder=scaleorder)
    h = tuple([hi/scalefactor for hi in h])

    if method == 'mc' or interpolation=='cubic':
        if method != 'mc':
            warnings.warn('Using cubic interpolation overrides contour method to be marching cubes ("mc").')
        
        
        SurfCoords, SurfConn = contour.MarchingCubesImage(img, h=h, threshold=threshold, flip=flip, method='original', interpolation=interpolation,VertexValues=False)
    else:
        voxel = VoxelMesh(img, h, threshold=None, scalefactor=1, scaleorder=1, return_nodedata=True)

        if method == 'mc33':
            SurfCoords, SurfConn = contour.MarchingCubes(voxel.NodeCoords, voxel.NodeConn, voxel.NodeData['Image Data'], method='33', threshold=threshold, flip=flip)

        elif method == 'mt':
            NodeCoords, NodeConn = converter.hex2tet(voxel.NodeCoords, voxel.NodeConn, method='1to6')
            SurfCoords, SurfConn = contour.MarchingTetrahedra(NodeCoords, NodeConn, voxel.NodeData['Image Data'], method='surface', threshold=threshold, flip=flip)

    
    if 'mesh' in dir(mesh):
        surface = mesh.mesh(SurfCoords, SurfConn)
    else:
        surface = mesh(SurfCoords, SurfConn)
        
    return surface

def TetMesh(img, h, threshold=None, threshold_direction=1, scalefactor=1, scaleorder=1, interpolation='linear'):
    """
    Generate a tetrahedral mesh of an image  

    Parameters
    ----------
    img : str or np.ndarray
        Image array or file path to an image
    h : scalar, tuple
        Element side length. Can be specified as a single scalar value, or a three element tuple (or array_like).
    threshold : scalar
        Isovalue threshold to use for keeping/removing elements, by default 0.
    threshold_dir : signed integer
        If threshold_dir is negative (default), values less than or equal to the threshold will be considered "inside" the mesh and the opposite if threshold_dir is positive, by default 1.
    interpolation : str, optional
        Method of interpolation used for placing the vertices on the approximated isosurface. This can be 'midpoint', 'linear', by default 'linear'. 

    Returns
    -------
    tet : mymesh.mesh
        Mesh object containing the tetrahedral mesh.

        .. note:: Due to the ability to unpack the mesh object to NodeCoords and NodeConn, the NodeCoords and NodeConn array can be returned directly (instead of the mesh object) by running: ``NodeCoords, NodeConn = implicit.TetMesh(...)``

    """
    
    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)

    if np.sign(threshold_direction) == 1:
        flip = True
    else:
        flip = False

    img = read(img, scalefactor=scalefactor, scaleorder=scaleorder)
    h = tuple([hi/scalefactor for hi in h])

    voxel = VoxelMesh(img, h, threshold=None, scalefactor=1, scaleorder=1, return_nodedata=True)
    NodeCoords, NodeConn = converter.hex2tet(voxel.NodeCoords, voxel.NodeConn, method='1to6')
    TetCoords, TetConn = contour.MarchingTetrahedra(NodeCoords, NodeConn, voxel.NodeData['Image Data'], method='volume', threshold=threshold, flip=flip)


    if 'mesh' in dir(mesh):
        tet = mesh.mesh(TetCoords, TetConn)
    else:
        tet = mesh(TetCoords, TetConn)
        
    return tet

def read(img, scalefactor=1, scaleorder=1):
    """
    Read image data into a numpy array format. Data can be input as an existing
    image array (in which case the only operation will be scaling), a file path
    to either a single image file or a directory of images, or a list of file
    paths to image files.

    Parameters
    ----------
    img : str, list, or np.ndarray
        If given as a string, file path to image directory or file. If given
        as a list, treated as a list of file paths. If given a numpy array,
        assumed to be an image data matrix and only scaling will be performed
        (if scalefactor != 1). 
    scalefactor : float, optional
        Scale factor for resampling the image. If greater than 1, there will be more than
        1 elements per voxel. If less than 1, will coarsen the image, by default 1.
    scaleorder : int, optional
        Interpolation order for scaling the image (see scipy.ndimage.zoom), by default 1.
        Must be 0-5.

    Returns
    -------
    I : np.ndarray or tuple of np.ndarray
        Image data array. For images with multichannel (e.g. RGB) data, this
        will be a tuple of arrays. For 3D image data separate files are assumed
        to be "z-slices", and are stored along the first axis of the image matrix
        (i.e. slice0 = I[0,:,:]), so the 0, 1, 2 axes correspond with z, y, x.

    """    

    multichannel = False

    if type(img) == np.ndarray:
        # If img is an array, only perform scaling
        assert len(img.shape) == 3, 'Image data must be a 3D array.'
        if scalefactor != 1:
            I = ndimage.zoom(img,scalefactor,order=scaleorder)
        else:
            I = img
        return I
    
    elif type(img) == str:
        # If img is a str, treat it as a file path
        # Collect files
        path = img
        assert os.path.exists(path), f'File path {path:s} does not exist.'

        if os.path.isdir(path):
            # Image directory
            tiffs = glob.glob(os.path.join(path,'*.tiff')) + glob.glob(os.path.join(path,'*.tif')) + glob.glob(os.path.join(path,'*.jpg')) + glob.glob(os.path.join(path,'*.jpeg')) + glob.glob(os.path.join(path,'*.png'))
            tiffs.sort()

            dicoms = glob.glob(os.path.join(path,'*.dcm'))
            if len(dicoms) == 0:
                dicoms = glob.glob(os.path.join(path,'*.DCM'))
            dicoms.sort()

            if len(tiffs) > 0 & len(dicoms) > 0:
                warnings.warn('Image directory: "{:s}" contains .dcm files as well as other image file types - only loading dcm files.')
                files = dicoms
                ftype = 'dcm'
            elif len(tiffs) > 0:
                files = tiffs
                ftype = 'tiff'
            elif len(dicoms) > 0:
                files = dicoms
                ftype = 'dcm'
            else:
                raise Exception('Image directory empty.')

        else:
            # Single file
            ext = os.path.splitext(path)[1]
            if ext.lower() in ('.tiff', '.tif', '.jpg', '.jpeg', '.png'):
                ftype = 'tiff'
                files = [path]
            elif ext.lower() in ('.dcm'):
                ftype = 'dcm'
                files = [path]
            else:
                raise ValueError('Image file must have one of the following extensions: .tiff, .tif, .jpg, .jpeg, .png, .dcm')

    elif type(img) == list:
        # If img is a list, treat it as a list of file paths
        if type(img[0]) == str:
            ext = os.path.splitext(img[0])
            if ext.lower() in ('.tiff', '.tif', '.jpg', '.jpeg', '.png'):
                ftype = 'tiff'
                files = img
            elif ext.lower() in ('.dcm'):
                ftype = 'dcm'
                files = img
            else:
                raise ValueError('Image file must have one of the following extensions: .tiff, .tif, .jpg, .jpeg, .png, .dcm')
        else:
            raise ValueError('If provided as a list, img must be a list of file paths. For a list of numeric image data, first convert to a numpy array.')
    
    else:
        raise ValueError(f'img mut be an array, str, or list, not {str(type(img)):s}')

    # Import appropriate image reader
    if ftype == 'dcm':
        try:
            import pydicom
        except:
            raise ImportError('pydicom must be installed to load DICOM files. Install with: pip install pydicom')
    else:
        try:
            import cv2
        except:
            raise ImportError('opencv-python (cv2) must be installed to load tiff, jpg, or png files. Install with: pip install opencv-python')

    # Load data
    print('Loading image data from {:s} ...'.format(img), end='')
    if ftype == 'tiff':
        
        temp = cv2.imread(files[0])
        if len(temp.shape) > 2:
            multichannel = True
            multiimgs = tuple([np.array([cv2.imread(file)[:,:,i] for file in files]) for i in range(temp.shape[2])])
        else:
            multichannel = False
            imgs = np.array([cv2.imread(file) for file in files])
    else:
        # temp = pydicom.dcmread(files[0]).pixel_array
        imgs = np.array([pydicom.dcmread(file).pixel_array for file in files])
    
    if scalefactor != 1:
        if multichannel:
            imgs = tuple([ndimage.zoom(I,scalefactor,order=scaleorder) for I in multiimgs])
        else:
            imgs = ndimage.zoom(imgs,scalefactor,order=scaleorder)
    else:
        if multichannel:
            imgs = multiimgs
    
    print(' done.')
    return imgs

def SurfaceNodeOptimization(M, img, voxelsize, iterate=1, threshold=0, FixedNodes=set(), FixEdges=False, gaussian_sigma=1, smooth=True, copy=True, interpolation='linear'):
    """
    Optimize the placement of surface node to lie on the "true" surface. This
    This simultaneously moves nodes towards the isosurface and redistributes
    nodes more evenly, thus smoothing the mesh without shrinkage or destruction
    of features. This method is consistes of using the Z-flow (and R-flow if 
    smooth=True) from :cite:p:`Ohtake2001a`.
    

    Parameters
    ----------
    M : mymesh.mesh
        Mesh object
    img : str or np.ndarray
        Image array or file path to an image
    voxelsize : float
        Voxel size of the image
    iterate : int, optional
        Number of iterations to perform, by default 1
    FixedNodes : set, optional
        Nodes to hold in place during repositioning, by default set()
    FixEdges : bool, optional
        Option to detect and hold in place exposed surface edges, by default False
    gaussian_sigma : float, optional
        Standard deviation used for getting the gradient via convolution with
        a Gaussian kernel in units of voxels (see scipy.ndimage.gaussian_filter), 
        by default 1 voxel.
    smooth : bool, optional
        Option to perform tangential Laplacian smoothing, by default True
    copy : bool, optional
        If true, will create a copy of the mesh, rather than altering node 
        positions of the original mesh object "in-place", by default True
    interpolation : str, optional
        Interpolation method used for evaluating surface points inside the image
        using scipy.interpolate.RegularGridInterpolator (input must be one of the
        allowable "method" inputs), by default 'linear'.

    Returns
    -------
    M : mesh.mesh
        Mesh with repositioned surface vertices

    """    

    if copy:
        M = M.copy()
    # Process nodes
    SurfNodes = set([n for elem in M.SurfConn for n in elem])
    if FixEdges:
        EdgeNodes = set(converter.surf2edges(M.NodeCoords, M.SurfConn).flatten())
    else:
        EdgeNodes = set()
    FreeNodes = np.array(list(SurfNodes.difference(EdgeNodes).difference(FixedNodes)))
    
    # Get gradient
    Fx = ndimage.gaussian_filter(img,gaussian_sigma,order=(1,0,0))
    Fy = ndimage.gaussian_filter(img,gaussian_sigma,order=(0,1,0))
    Fz = ndimage.gaussian_filter(img,gaussian_sigma,order=(0,0,1))
    
    X = np.arange(img.shape[2])*voxelsize 
    Y = np.arange(img.shape[1])*voxelsize
    Z = np.arange(img.shape[0])*voxelsize

    F = lambda x,y,z : interpolate.RegularGridInterpolator((X,Y,Z),img.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    gradFx = lambda x,y,z : interpolate.RegularGridInterpolator((X,Y,Z),Fx.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    gradFy = lambda x,y,z : interpolate.RegularGridInterpolator((X,Y,Z),Fy.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    gradFz = lambda x,y,z : interpolate.RegularGridInterpolator((X,Y,Z),Fx.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)

    gradF = lambda x,y,z : np.column_stack([gradFx(x,y,z), gradFy(x,y,z), gradFz(x,y,z)])

    NodeCoords = np.vstack([np.asarray(M.NodeCoords), [np.nan,np.nan,np.nan]])
    points = np.asarray(NodeCoords)[FreeNodes]

    if smooth:
        x = points[:,0]; y = points[:,1]; z = points[:,2]
        g = gradF(x,y,z)
        r = utils.PadRagged(np.array(M.SurfNodeNeighbors,dtype=object)[FreeNodes].tolist())
        lengths = np.array([len(M.SurfNodeNeighbors[i]) for i in FreeNodes])

    for i in range(iterate):
        x = points[:,0]; y = points[:,1]; z = points[:,2]
        f = F(x,y,z) - threshold
        g = gradF(x,y,z)
        fg = f[:,None]*g
        tau = voxelsize/(100*np.max(np.linalg.norm(fg,axis=1)))

        Zflow = -2*tau*fg

        if smooth:
            Q = NodeCoords[r]
            U = (1/lengths)[:,None] * np.nansum(Q - points[:,None,:],axis=1)
            NodeNormals = (g/np.linalg.norm(g,axis=0))
            Rflow = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        else:
            Rflow = 0

        points = points + Zflow + Rflow
    
    M.NodeCoords[FreeNodes] = points

    return M