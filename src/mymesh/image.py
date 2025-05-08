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
    write
"""


import numpy as np
from scipy import interpolate, ndimage
import sys, os, copy, warnings, glob

from . import utils, converter, contour, quality, improvement, rays, mesh, primitives

# Mesh generators
def VoxelMesh(img, h, threshold=None, threshold_direction=1, scalefactor=1, scaleorder=1, return_nodedata=False):
    """
    Generate voxel mesh of an image function

    Parameters
    ----------
    img : str or np.ndarray
        Image array or file path to an image
    h : scalar, tuple
        Voxel size of the image. Can be specified as a single scalar value, or a three element tuple (or array_like).
        If a tuple, entries should correspond to (hx, hy, hz).
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
        Voxel size of the image. Can be specified as a single scalar value, or a three element tuple (or array_like).
        If a tuple, entries should correspond to (hx, hy, hz).
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
            SurfCoords, SurfConn = contour.MarchingTetrahedra(NodeCoords, NodeConn, voxel.NodeData['Image Data'], Type='surf', threshold=threshold, flip=flip)

    
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
        Voxel size of the image. Can be specified as a single scalar value, or a three element tuple (or array_like).
        If a tuple, entries should correspond to (hx, hy, hz).
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
    TetCoords, TetConn, Values = contour.MarchingTetrahedra(NodeCoords, NodeConn, voxel.NodeData['Image Data'], Type='vol', threshold=threshold, flip=flip, return_NodeValues=True)


    if 'mesh' in dir(mesh):
        tet = mesh.mesh(TetCoords, TetConn)
    else:
        tet = mesh(TetCoords, TetConn)
    tet.NodeData['Image Data'] = Values
    return tet

def read(img, scalefactor=1, scaleorder=1, verbose=True):
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
    verbose : bool, optional
        If Tru, will print status message for loading the image.

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
            tiffs = glob.glob(os.path.join(path,'*.TIF*')) + glob.glob(os.path.join(path,'*.tif*'))
            tiffs = np.unique(tiffs).tolist() # This catches double reads on Windows due to case insensitive glob
            tiffs.sort()

            dicoms = glob.glob(os.path.join(path,'*.dcm*'))
            if len(dicoms) == 0:
                dicoms = glob.glob(os.path.join(path,'*.DCM*'))
            dicoms.sort()

            cv2s = glob.glob(os.path.join(path,'*.jpg*')) + glob.glob(os.path.join(path,'*.JPG*')) + glob.glob(os.path.join(path,'*.jpeg*')) + glob.glob(os.path.join(path,'*.JPEG*')) + glob.glob(os.path.join(path,'*.png*')) + glob.glob(os.path.join(path,'*.PNG*'))
            cv2s = np.unique(cv2s).tolist()
            cv2s.sort()

            if len(tiffs) > 0 & len(dicoms) > 0 & len(cv2s) > 0:
                raise Exception('Image directory: "{:s}" contains multiple image file types.')
            elif len(tiffs) > 0:
                files = tiffs
                ftype = 'tiff'
            elif len(dicoms) > 0:
                files = dicoms
                ftype = 'dcm'
            elif len(cv2s) > 0:
                files = cv2s
                ftype = 'cv2_type'
            else:
                raise Exception('Image directory empty.')

        else:
            # Single file
            ext = os.path.splitext(path)[1]
            if ext.lower() in ('.tiff', '.tif'):
                ftype = 'tiff'
                files = [path]
            elif ext.lower() in ('.dcm'):
                ftype = 'dcm'
                files = [path]
            elif ext.lower() in ('.jpg', '.jpeg', '.png'):
                ftype = 'cv2_type'
                files = [path]
            else:
                raise ValueError('Image file must have one of the following extensions: .tiff, .tif, .jpg, .jpeg, .png, .dcm')

    elif type(img) == list:
        # If img is a list, treat it as a list of file paths
        if type(img[0]) == str:
            ext = os.path.splitext(img[0])
            if ext.lower() in ('.tiff', '.tif'):
                ftype = 'tiff'
                files = img
            elif ext.lower() in ('.dcm'):
                ftype = 'dcm'
                files = img
            elif ext.lower() in ('.jpg', '.jpeg', '.png'):
                ftype = 'cv2_type'
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
    if verbose: print('Loading image data from {:s} ...'.format(img), end='')
    if ftype == 'tiff':
        try:
            import tifffile
        except:
            raise ImportError('tifffile needed for reading tiff images. Install with: pip install tifffile')
            
        if len(files) == 1:
            imgs = tifffile.imread(files[0])
            if len(imgs.shape) > 2:
                multichannel = True
                multiimgs = np.array([imgs[...,i] for i in range(np.shape(imgs)[-1])])
            
        else:
            temp = tifffile.imread(files[0])
            
            if len(temp.shape) > 2:
                multichannel = True
                multiimgs = tuple([np.array([tifffile.imread(file)[...,i] for file in files]) for i in range(temp.shape[-1])])
            else:
                multichannel = False
                imgs = np.array([tifffile.imread(file) for file in files])
    elif ftype == 'cv2_type':
        if len(files) == 1:
            imgs = cv2.imread(files[0])
            if len(imgs.shape) > 2:
                multichannel = True
                multiimgs = np.array([imgs[...,i] for i in range(np.shape(imgs)[-1])])
        else:
            temp = cv2.imread(files[0])
            
            if len(temp.shape) > 2:
                multichannel = True
                multiimgs = tuple([np.array([cv2.imread(file)[...,i] for file in files]) for i in range(temp.shape[-1])])
            else:
                multichannel = False
                imgs = np.array([cv2.imread(file) for file in files])
    else:
        # temp = pydicom.dcmread(files[0]).pixel_array
        imgs = np.array([pydicom.dcmread(file).pixel_array for file in files])
    # Check if true multichannel
    if multichannel:
        if np.all([np.all(channel == multiimgs[0]) for channel in multiimgs[1:]]):
            # data is the same in every channel, converting to single channel
            imgs = multiimgs[0]
            multichannel = False
    if scalefactor != 1:
        if multichannel:
            imgs = tuple([ndimage.zoom(I,scalefactor,order=scaleorder) for I in multiimgs])
        else:
            imgs = ndimage.zoom(imgs,scalefactor,order=scaleorder)
    else:
        if multichannel:
            imgs = multiimgs
    
    if verbose: print(' done.')
    return imgs

def write(impath, I, filetype=None, verbose=True, dtype=np.int16):
    """
    Write an image array to an image file or stack of image files
    
    .. note:: 
        When writing a jpg, png, or tiff, if the input image data has a maximum
        value greater than 255 or less than or equal to 1, the data will be 
        automatically scaled to be in the range from, 0-255

    Parameters
    ----------
    impath : str
        File path to image directory or file name to save the images. If given
        as a filename with an extension, writing of a single image file will be 
        attempted. For certain data type and file types (e.g. 3D multichannel data), 
        this may not be supported and a stack of images prefixed with the filename
        will be written instead.
    I : array_like or tuple
        2D or 3D array_like containing image data, or 3 or 4 element tuple of 
        2D or 3D array_like of image data for rgb or rgba multichannel image
        data.
    filetype : str, optional
        file specification, by default None. If impath is a filename with an
        extension, the extension will override the filetype input if given. 
        If impath is a directory and no filetype is specified, tiff will
        be used by default.
        
        Options are:
            
        - 'dicom'
        
        - 'tiff'
        
        - 'png'
        
        - 'jpeg'
        
    """    
    
    import datetime
    import cv2
    
    
    # initialize file specification variables
    multichannel = False
    singlefile = False
    stack = False
    filename_prefix = 'I'
    
    # Process impath and filetype to determine type of file being written
    path, ext = os.path.splitext(impath)
    if ext != '':
        singlefile = True
        imdir, filename_prefix = os.path.split(path)
        if ext.lower() == '.tif' or ext.lower() == '.tiff':
            filetype = 'tiff'
        elif ext.lower() == '.jpg' or ext.lower() == '.jpeg':
            filetype = 'jpeg'
        elif ext.lower() == '.png':
            filetype = 'png'
        elif ext.lower() == '.dcm':
            filetype = 'dicom'
        else:
            raise ValueError(f'Unknown or unsupported file extension: "{ext:s}"')
    else:
        singlefile = False
        imdir = path
        if filetype is None or filetype.lower() == 'tif' or filetype.lower() == 'tiff' or filetype.lower() == '.tif' or filetype.lower() == '.tiff':
            filetype = 'tiff'
            ext = '.tif'
        elif filetype.lower() == 'jpeg' or filetype.lower() ==  'jpg' or filetype.lower() ==  '.jpeg' or filetype.lower() ==  '.jpg':
            filetype = 'jpeg'
            ext = '.jpg'
        elif filetype.lower() == 'png' or filetype.lower() == '.png':
            filetype = 'png'
            ext = '.png'
        elif filetype.lower() == 'dicom' or filetype.lower() == 'dcm' or filetype.lower() == '.dcm':
            filetype = 'dicom'
            ext = '.dcm'
        else:
            raise ValueError(f'Unknown or unsupported file type: "{filetype:s}"')
    if not os.path.exists(imdir) and imdir != '':
        os.mkdir(imdir)
    # Assess type/shape of image data
    if type(I) is tuple and (len(I) == 3 or len(I) == 4):
        # multichannel rgb or rgba image
        multichannel = True
        if len(np.shape(I[0])) == 3:
            singlefile = False 
            stack = True
        else:
            singlefile = True
            stack = False
    elif len(np.shape(I)) == 3:
        stack = True
        if singlefile and filetype != 'tiff':
            singlefile = False
    elif len(np.shape(I)) == 2:
        stack = False
        singlefile = True
            
    # evaluate case
    if singlefile:
        if filetype == 'dicom':
            if multichannel:
                raise ValueError("Writing multichannel images to dicoms is not supported. Convert to grayscale or use a different filetype.")
            _writedcm(imdir, filename_prefix+'.dcm', I, dtype)
        elif filetype == 'tiff':
            try:
                import tifffile
                tifffile.imwrite(os.path.join(imdir, filename_prefix+ext), I)
            except:
                raise ImportError('tifffile needed to write tiffs. Install with `pip install tifffile`.')
                    
            # if multichannel:
            #     tifffile.imwrite(os.path.join(imdir, filename_prefix+ext), np.stack(I,axis=-1))
            # else:
            tifffile.imwrite(os.path.join(imdir, filename_prefix+ext), I)
        elif filetype == 'jpeg' or filetype == 'png':
            img = I*255/np.max(I).astype(dtype)
            cv2.imwrite(os.path.join(imdir, filename_prefix+ext), img)
    
    if not singlefile:
        if multichannel:
            nfiles = len(I[0])
        else:
            nfiles = len(I)
        
        ndigits = max(int(np.floor(np.log10(nfiles))+1),4)
        if filetype == 'dicom':
            if multichannel:
                raise ValueError("Writing multichannel images to dicoms is not supported. Convert to grayscale or use a different filetype.")
            for i in range(len(I)):
                _writedcm(imdir, filename_prefix+f'_{str(i).zfill(ndigits):s}.dcm', I, dtype)
        else:
            for i in range(len(I)):
                if multichannel:
                    img = np.stack([channel[i] for channel in I], axis=-1)
                else:
                    img = I[i]
                if np.max(img) > 255 or np.max(img) <= 1:
                    img = img*255/np.max(I).astype(dtype)
                cv2.imwrite(os.path.join(imdir, filename_prefix+f'_{str(i).zfill(ndigits):s}{ext:s}'), img)
             
def _writedcm(imdir, filename, I, dtype):
    # I should be a 2D image at this point
    import datetime
    try:
        import pydicom
    except:
        raise ImportError('pydicom must be installed to write DICOM files. Install with: pip install pydicom')
    file = filename
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.UID('1.2.840.10008.5.1.4.1.1.2')
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.UID("1.2.3")
    file_meta.ImplementationClassUID = pydicom.uid.UID("1.2.3.4")
    ds = pydicom.dataset.FileDataset(imdir, {},
            file_meta=file_meta, preamble=b"\0" * 128)
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime('%Y%m%d')
    timeStr = dt.strftime('%H%M%S.%f')  # long format with micro seconds
    ds.ContentTime = timeStr
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    ds.is_little_endian = True
    ds.is_implicit_VR = True
    
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.Rows = I.shape[0]
    ds.Columns = I.shape[1]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    if np.min(I) >= 0:
        ds.PixelRepresentation = 0
    else:
        ds.PixelRepresentation = 1
    ds.PixelData = I.astype(dtype).tobytes()
    ds.save_as(os.path.join(imdir,file))
       
def SurfaceNodeOptimization(M, img, h, iterate=1, threshold=0, FixedNodes=set(), FixEdges=False, gaussian_sigma=1, smooth=True, copy=True, interpolation='linear', 
springs=True):
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
    smooth : str, optional
        Option to perform smoothing. This can be either 'local' for local 
        Laplacian smoothing or 'tangential' for tangential Laplacian smoothing, 
        by default 'tangential'. For any other option, smoothing will not be 
        performed. Tangential smoothing differs from local in that nodes 
        are only repositioned in the tangent plane (based on the normal vector
        obtain from the gradient). 
    InPlace : bool, optional
        If False, will create a copy of the mesh, rather than altering node 
        positions of the original mesh object "in-place", by default False
    interpolation : str, optional
        Interpolation method used for evaluating surface points inside the image
        using scipy.interpolate.RegularGridInterpolator (input must be one of the
        allowable "method" inputs), by default 'linear'.
    springs : bool, optional
        If True and the mesh is a volume mesh, internal nodes will be treated as 
        if they are connected by springs (see :func:`~mymesh.improvements.NodeSpringSmoothing`)
        to reduce risk of element distortion or inversion, by default True.
    

    Returns
    -------
    M : mesh.mesh
        Mesh with repositioned surface vertices

    """    
    if not isinstance(h, (list, tuple, np.ndarray)):
        h = (h,h,h)
    if copy:
        M = M.copy()
    M.NodeCoords = np.asarray(M.NodeCoords)
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
    
    Ximg = np.arange(img.shape[2])*h[2] 
    Yimg = np.arange(img.shape[1])*h[1]
    Zimg = np.arange(img.shape[0])*h[0]

    F = lambda x,y,z : interpolate.RegularGridInterpolator((Ximg,Yimg,Zimg),img.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    gradFx = lambda x,y,z : interpolate.RegularGridInterpolator((Ximg,Yimg,Zimg),Fx.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    gradFy = lambda x,y,z : interpolate.RegularGridInterpolator((Ximg,Yimg,Zimg),Fy.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    gradFz = lambda x,y,z : interpolate.RegularGridInterpolator((Ximg,Yimg,Zimg),Fx.T,method='linear',bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)

    gradF = lambda x,y,z : np.column_stack([gradFx(x,y,z), gradFy(x,y,z), gradFz(x,y,z)])

    NodeCoords = np.vstack([M.NodeCoords, [np.nan,np.nan,np.nan]])
    points = NodeCoords[FreeNodes]
    if smooth == True:
        smooth = 'tangential'
    if smooth is not None and smooth is not False:
        x = points[:,0]; y = points[:,1]; z = points[:,2]
        g = gradF(x,y,z)
        r = utils.PadRagged(np.array(M.SurfNodeNeighbors,dtype=object)[FreeNodes].tolist())
        lengths = np.array([len(M.SurfNodeNeighbors[i]) for i in FreeNodes])

    for i in range(iterate):
        X = points[:,0]; Y = points[:,1]; Z = points[:,2]
        f = F(X,Y,Z) - threshold
        g = np.squeeze(gradF(X,Y,Z))
        fg = (f[:,None]*g)
        tau = h/(100*np.max(np.linalg.norm(fg,axis=0)))

        Zflow = -2*tau*fg

        # Rflow = np.zeros((len(NodeCoords),3))
        if smooth == 'tangential':
            Q = NodeCoords[r]
            U = (1/lengths)[:,None] * np.nansum(Q - points[:,None,:],axis=1)
            gnorm = np.linalg.norm(g,axis=1)[:,None]
            NodeNormals = np.divide(g, gnorm, out=np.zeros_like(g), where=gnorm!=0)
            Rflow = 1*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        elif smooth == 'local':
            Q = NodeCoords[r]
            U = (1/lengths)[:,None] * np.nansum(Q - points[:,None,:],axis=1)
            Rflow = U
        else:
            Rflow = 0

        if springs and M.Type == 'vol':
            Forces = np.zeros((M.NNode, 3))
            Forces[FreeNodes] = Zflow + Rflow
            M = improvement.NodeSpringSmoothing(M, Displacements=Forces, options=dict(FixSurf=False, FixedNodes=FixedNodes, InPlace=True))
            NodeCoords = M.NodeCoords
            points = np.asarray(M.NodeCoords)[FreeNodes]
        else:
            points = points + Zflow + Rflow
            M.NodeCoords[FreeNodes] = points
    
    M.NodeCoords[FreeNodes] = points

    return M