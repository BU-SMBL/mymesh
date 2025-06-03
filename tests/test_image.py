import pytest
import numpy as np
from mymesh import image, implicit
import tempfile, os

@pytest.mark.parametrize("img, filetype", [
    (
        255*np.eye(3),
        None
    ),
    (
        255*np.eye(3),
        'tiff'
    ),
    (
        255*np.eye(3),
        'png'
    ),
    (
        np.eye(3),
        'dcm'
    ),
    (
        255*np.stack([np.eye(3),np.eye(3)[::-1],np.eye(3)]),
        None
    ),
    (
        255*np.stack([np.eye(3),np.eye(3)[::-1],np.eye(3)]),
        'tiff'
    ),
    (
        255*np.stack([np.eye(3),np.eye(3)[::-1],np.eye(3)]),
        'png'
    ),
    (
        255*np.stack([np.eye(3),np.eye(3)[::-1],np.eye(3)]),
        'dcm'
    ),
])
def test_read_write(img, filetype):

    with tempfile.TemporaryDirectory() as path:
        # directory read/write
        image.write(path, img, filetype=filetype)
        I = image.read(path)
    if filetype == 'tiff' or (len(np.shape(img)) == 2 and filetype is not None):
        # single-file read/write
        
        with tempfile.TemporaryDirectory() as path:
            fname = os.path.join(path, '.'+filetype)
            image.write(fname, img)
            I = image.read(fname)
    
    assert np.all(I == img), 'Image read/write mismatch'

@pytest.mark.parametrize("img, threshold", [
    (
        255*np.stack([np.eye(3),np.eye(3)[::-1],np.eye(3)]),
        None
    ),
    (
        255*np.stack([np.eye(3),np.eye(3)[::-1],np.eye(3)]),
        100
    ),
])
def test_VoxelMesh(img, threshold):

    M = image.VoxelMesh(img, 1, threshold=threshold, return_nodedata=True)
    if threshold is None:
        assert M.NElem == np.size(img), 'Incorrect number of elements.'
    else:
        assert M.NElem == np.sum(img > threshold), 'Incorrect number of elements.'

@pytest.mark.parametrize("img, h, threshold, method", [
    (
        implicit.sphere([0,0,0],1)(
            *np.meshgrid(
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            indexing='ij')).T,
        1,
        0,
        'mc'        
    ),
    (
        implicit.sphere([0,0,0],1)(
            *np.meshgrid(
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            indexing='ij')).T,
        1,
        0,
        'mc33'        
    ),
    (
        implicit.sphere([0,0,0],1)(
            *np.meshgrid(
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            indexing='ij')).T,
        1,
        0,
        'mt'        
    ),
])
def test_SurfaceMesh(img, h, threshold, method):

    M = image.SurfaceMesh(img, h, threshold=threshold, method=method)
    

@pytest.mark.parametrize("img, h, threshold", [
    (
        implicit.sphere([0,0,0],1)(
            *np.meshgrid(
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            indexing='ij')).T,
        1,
        0       
    ),
])
def test_TetMesh(img, h, threshold):

    M = image.TetMesh(img, h, threshold=threshold)

@pytest.mark.parametrize("img, h, threshold", [
    # unit sphere
    (
        implicit.sphere([0,0,0],1)(
            *np.meshgrid(
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            np.arange(-1, 1, .05),
            indexing='ij')).T,
        1,
        0       
    ),
])
def test_SurfaceNodeOptimization(img, h, threshold, ):

    M = image.TetMesh(img, h, threshold=threshold)

    M = image.SurfaceNodeOptimization(M, img, h, iterate=5, threshold=threshold,
        springs=True)

    M = image.SurfaceNodeOptimization(M, img, h, iterate=5, threshold=threshold,
        springs=False)

    assert ~np.any(np.all(np.isnan(M.NodeCoords)))


    
