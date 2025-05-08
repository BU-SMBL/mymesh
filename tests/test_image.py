import pytest
import numpy as np
from mymesh import image
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

