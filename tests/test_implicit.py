import pytest
import numpy as np
from mymesh import implicit

@pytest.mark.parametrize("func, bounds, h, threshold, threshold_direction", [
    # unit sphere
    (implicit.sphere([0,0,0],1), 
    [-1,1,-1,1],
    0.1,
    0,
    -1,
    ),
    # gyroid
    (implicit.gyroid, 
    [0,1,0,.5],
    0.1,
    0,
    1,
    ),
])
def test_PlanarMesh(func, bounds, h, threshold, threshold_direction):
    M = implicit.PlanarMesh(func, bounds, h, threshold=threshold, threshold_direction=threshold_direction)

    X, Y, Z = M.NodeCoords.T

    assert np.all((X >= bounds[0]) & (X <= bounds[1]) & (Y >= bounds[2]) & (Y <= bounds[3])), 'Incorrect Bounds'

@pytest.mark.parametrize("func, bounds, h, threshold, threshold_direction, mode", [
    # unit sphere
    (implicit.sphere([0,0,0],1), 
    [-1,1,-1,1,-1,1],
    0.1,
    0,
    -1,
    'any',
    ),
    # gyroid
    (implicit.gyroid, 
    [0,1,0,.5,0,.75],
    0.1,
    0,
    1,
    'all',
    ),
])
def test_VoxelMesh(func, bounds, h, threshold, threshold_direction, mode):
    M = implicit.VoxelMesh(func, bounds, h, threshold=threshold, threshold_direction=threshold_direction, mode=mode)

    X, Y, Z = M.NodeCoords.T

    assert np.all((X >= bounds[0]) & (X <= bounds[1]) & (Y >= bounds[2]) & (Y <= bounds[3]) & (Z >= bounds[4]) & (Z <= bounds[5])), 'Incorrect Bounds'

@pytest.mark.parametrize("func, bounds, h, threshold, threshold_direction, method, interpolation", [
    # unit sphere
    (implicit.sphere([0,0,0],1), 
    [-1,1,-1,1,-1,1],
    0.1,
    0,
    -1,
    'mc',
    'cubic'
    ),
    # gyroid
    (implicit.gyroid, 
    [0,1,0,.5,0,.75],
    0.1,
    0,
    1,
    'mt',
    'quadratic'
    ),
    # primitive
    (implicit.primitive, 
    [0,1,0,.5,0,75],
    0.1,
    0,
    1,
    'mc33',
    'linear'
    ),
])
def test_SurfaceMesh(func, bounds, h, threshold, threshold_direction, method, interpolation):
    M = implicit.SurfaceMesh(func, bounds, h, threshold=threshold, threshold_direction=threshold_direction, method=method, interpolation=interpolation)

    X, Y, Z = M.NodeCoords.T

    assert np.all((X >= bounds[0]) & (X <= bounds[1]) & (Y >= bounds[2]) & (Y <= bounds[3]) & (Z >= bounds[4]) & (Z <= bounds[5])), 'Incorrect Bounds'

@pytest.mark.parametrize("func, bounds, h, threshold, threshold_direction, interpolation", [
    # unit sphere
    (implicit.sphere([0,0,0],1), 
    [-1,1,-1,1,-1,1],
    0.1,
    0,
    -1,
    'linear',
    ),
    # gyroid
    (implicit.gyroid, 
    [0,1,0,.5,0,.75],
    0.1,
    0,
    1,
    'quadratic',
    ),
])
def test_TetMesh(func, bounds, h, threshold, threshold_direction, interpolation):
    M = implicit.TetMesh(func, bounds, h, threshold=threshold, threshold_direction=threshold_direction, interpolation=interpolation)

    X, Y, Z = M.NodeCoords.T

    assert np.all((X >= bounds[0]) & (X <= bounds[1]) & (Y >= bounds[2]) & (Y <= bounds[3]) & (Z >= bounds[4]) & (Z <= bounds[5])), 'Incorrect Bounds'

@pytest.mark.parametrize("func, bounds, h, threshold, threshold_direction, interpolation", [
    # unit sphere
    (implicit.sphere([0,0,0],1), 
    [-1,1,-1,1,-1,1],
    0.1,
    0,
    -1,
    'linear',
    ),
    # gyroid
    (implicit.gyroid, 
    [0,1,0,.5,0,75],
    0.1,
    0,
    1,
    'quadratic',
    ),
])
def test_SurfaceNodeOptimization(func, bounds, h, threshold, threshold_direction, interpolation):

    M = implicit.TetMesh(func, bounds, h, threshold=threshold, threshold_direction=threshold_direction, interpolation=interpolation)

    M = implicit.SurfaceNodeOptimization(M, func, h, iterate=5, threshold=0,
        springs=True)

    M = implicit.SurfaceNodeOptimization(M, func, h, iterate=5, threshold=0,
        springs=False)

    assert ~np.any(np.all(np.isnan(M.NodeCoords)))
