import pytest
import numpy as np
from mymesh import primitives, quality

@pytest.mark.parametrize("bounds, h, ElemType", [
    # 1x1x1 cube, quad elements
    ([0,1,0,1,0,1], 
     0.1, 
     'quad',
     ),
     # 4x3x2 cube, tri elements
    ([-2,2,-1.5,1.5,-1,1], 
     0.13, 
     'tri',
     ),
])
def test_Box(bounds, h, ElemType):
    box = primitives.Box(bounds, h, ElemType=ElemType)

    assert np.all(np.max(box.NodeCoords,axis=0) == [bounds[1],bounds[3],bounds[5]]) & np.all(np.min(box.NodeCoords,axis=0) == [bounds[0],bounds[2],bounds[4]]), "Non-matching bounds"
    if ElemType == 'quad':
        assert np.shape(box.NodeConn)[1] == 4, "Incorrect element type"
    else:
        assert np.shape(box.NodeConn)[1] == 3, "Incorrect element type"
    assert len(box.BoundaryNodes) == 0, "Unclosed edges"

@pytest.mark.parametrize("bounds, h, exact_h, ElemType", [
    # 1x1x1 cube, hex elements
    ([0,1,0,1,0,1], 
     0.1, 
     True,
     'hex',
     ),
     # 4x3x2 cube, exact h, tet elements
    ([-2,2,-1.5,1.5,-1,1], 
     0.13, 
     True,
     'tet',
     ),
     # 4x3x2 cube, exact bounds, tet elements 
    ([-2,2,-1.5,1.5,-1,1], 
     (0.13,0.09, 0.14), 
     False,
     'tet',
     ),
     # 4x3x2 cube, exact h, tet elements, anisotropic
    ([-2,2,-1.5,1.5,-1,1], 
     0.13,
     True,
     'tet',
     ),
])
def test_Grid(bounds, h, exact_h, ElemType):
    grid = primitives.Grid(bounds, h, exact_h=exact_h, ElemType=ElemType)
    
    if exact_h:
        if type(h) is tuple:
            h = np.array(h)
        assert np.all(np.isclose((grid.NodeCoords-np.min(grid.NodeCoords,axis=0))/h, np.round((grid.NodeCoords-np.min(grid.NodeCoords,axis=0))/h).astype(int))), "Incorrect element spacing"
    else:
        assert np.all(np.max(grid.NodeCoords,axis=0) == [bounds[1],bounds[3],bounds[5]]) & np.all(np.min(grid.NodeCoords,axis=0) == [bounds[0],bounds[2],bounds[4]]), "Non-matching bounds"
    if ElemType == 'hex':
        assert np.shape(grid.NodeConn)[1] == 8, "Incorrect element type"
    else:
        assert np.shape(grid.NodeConn)[1] == 4, "Incorrect element type"
    assert len(grid.Surface.BoundaryNodes) == 0, "Unclosed edges"

@pytest.mark.parametrize("bounds, h, exact_h, ElemType", [
    # 1x1x1 grid, quad elements
    ([0,1,0,1], 
     0.1, 
     True,
     'quad',
     ),
     # 4x3x2 grid, exact h, tri elements
    ([-2,2,-1.5,1.5], 
     0.13, 
     True,
     'tri',
     ),
     # 4x3x2 grid, exact bounds, tri elements 
     ([-2,2,-1.5,1.5], 
     (0.13,0.09), 
     False,
     'tri',
     ),
     # 4x3x2 grid, exact h, tri elements, anisotropic
     ([-2,2,-1.5,1.5], 
     0.13, 
     True,
     'tri',
     ),
])
def test_Grid2D(bounds, h, exact_h, ElemType):
    grid = primitives.Grid2D(bounds, h, exact_h=exact_h, ElemType=ElemType)
    
    if exact_h:
        if type(h) is tuple:
            h = np.array(h)
        assert np.all(np.isclose((grid.NodeCoords-np.min(grid.NodeCoords,axis=0))/h, np.round((grid.NodeCoords-np.min(grid.NodeCoords,axis=0))/h).astype(int))), "Incorrect element spacing"
    else:
        assert np.all(np.max(grid.NodeCoords,axis=0)[:2] == [bounds[1],bounds[3]]) & np.all(np.min(grid.NodeCoords,axis=0)[:2] == [bounds[0],bounds[2]]), "Non-matching bounds"
    if ElemType == 'quad':
        assert np.shape(grid.NodeConn)[1] == 4, "Incorrect element type"
    else:
        assert np.shape(grid.NodeConn)[1] == 3, "Incorrect element type"

@pytest.mark.parametrize("center, radius, theta_resolution, ElemType, Type", [
    # unit circle
    ([0,0,0], 
    1, 
    20,
    None,
    'surf'
    ),
    # unit circle offset
    ([1,-1,2], 
    1, 
    20, 
    'tri',
    'surf'
    ),
    # unit circle
    ([0,0,0], 
    1, 
    20,
    None,
    'line'
    ),
    
])
def test_Circle(center, radius, theta_resolution, ElemType, Type):
    circle = primitives.Circle(center, radius, theta_resolution=theta_resolution, ElemType=ElemType)
        
    X, Y, Z = circle.NodeCoords[circle.BoundaryNodes].T
    
    assert np.all(np.isclose((X - center[0])**2/radius**2 + (Y - center[1])**2/radius**2, 1)), 'Off-circle coordinates'

@pytest.mark.parametrize("center, point, theta_resolution, ElemType, Type", [
    # unit circle
    ([0,0,0], 
    [1,0,0], 
    20,
    None,
    'surf'
    ),
    # unit circle offset
    ([1,-1,2], 
    [2,-1,2], 
    20, 
    'tri',
    'surf'
    ),
    # unit circle
    ([0,0,0], 
    [0,1,0], 
    20,
    None,
    'line'
    ),
])
def test_CirclePt(center, point, theta_resolution, ElemType, Type):
    circle = primitives.CirclePt(center, point, theta_resolution=theta_resolution, ElemType=ElemType)
        
    X, Y, Z = circle.NodeCoords[circle.BoundaryNodes].T
    radius = np.linalg.norm(np.subtract(point, center))
    assert np.all(np.isclose((X - center[0])**2/radius**2 + (Y - center[1])**2/radius**2, 1)), 'Off-circle coordinates'

@pytest.mark.parametrize("center, radius, height, theta_resolution, axial_resolution, radial_resolution, axis, cap, ElemType, Type", [
    # unit cylinder (surface)
    ([0,0,0],
    1,
    1,
    20,
    10,
    10,
    2,
    True,
    None,
    'surf'
    ),
    # unit cylinder (volume)
    ([0,0,0],
    1,
    1,
    20,
    10,
    10,
    2,
    True,
    None,
    'vol'
    ),
    # elliptical cylinder (surface)
    ([0,0,0],
    (1,2),
    1,
    20,
    10,
    10,
    2,
    True,
    None,
    'surf'
    ),  
])
def test_Cylinder(center, radius, height, theta_resolution, axial_resolution, radial_resolution, axis, cap, ElemType, Type):
    cylinder = primitives.Cylinder(center, radius, height, theta_resolution, axial_resolution, radial_resolution, axis, cap, ElemType, Type)
    
    if cap:
        assert len(cylinder.BoundaryNodes) == 0, 'Unclosed edges'

@pytest.mark.parametrize("center, radius, theta_resolution, phi_resolution, ElemType, Type", [
    # unit sphere
    ([0,0,0], 
    1, 
    20, 20,
    None,
    'surf'
    ),
    # unit sphere offset
    ([1,-1,2], 
    1, 
    20, 20,
    'tri',
    'surf'
    ),
    # ellipsoid
    ([0,0,0], 
    (1,2,3), 
    30, 20,
    None,
    'surf'
    ),
    # unit sphere offset
    ([1,-1,2], 
    (1,2,3), 
    20, 30,
    'tri',
    'surf'
    ),
    # unit sphere
    ([0,0,0], 
    1, 
    20, 20,
    None,
    'vol'
    ),
    # unit sphere
    ([0,0,0], 
    1, 
    20, 20,
    'tet',
    'vol'
    ),
])
def test_Sphere(center, radius, theta_resolution, phi_resolution, ElemType, Type):
    sphere = primitives.Sphere(center, radius, theta_resolution=theta_resolution, phi_resolution=phi_resolution, ElemType=ElemType)
    
    if not isinstance(radius, (tuple, list, np.ndarray)):
        radius = (radius, radius, radius)
    
    X, Y, Z = sphere.NodeCoords[sphere.SurfNodes].T
    
    assert np.all(np.isclose((X - center[0])**2/radius[0]**2 + (Y - center[1])**2/radius[1]**2 + (Z - center[2])**2/radius[2]**2, 1)), 'Off-sphere coordinates'

    assert len(sphere.BoundaryNodes) == 0, 'Unclosed edges'

@pytest.mark.parametrize("center, R, r, axis, theta_resolution, phi_resolution, ElemType, Type", [
    ([0,0,0], 
    1, 
    1,
    0,
    20,
    20,
    None,
    'surf'
    ),
    ([1,-1,2], 
    2, 
    1,
    1,
    10,
    20,
    'tri',
    'surf'
    ),
    ([-.5,.3,2], 
    10, 
    1,
    2,
    20,
    10,
    'tri',
    'surf'
    ),
    ([0,0,0], 
    1, 
    1,
    0,
    20,
    20,
    None,
    'vol'
    ),
    ([0,0,0], 
    1, 
    1,
    0,
    20,
    20,
    'tet',
    'vol'
    ),
])
def test_Torus(center, R, r, axis, theta_resolution, phi_resolution, ElemType, Type):
    torus = primitives.Torus(center, R, r, axis=axis, theta_resolution=theta_resolution, phi_resolution=phi_resolution, ElemType=ElemType)
    
    assert len(torus.BoundaryNodes) == 0, 'Unclosed edges'

@pytest.mark.parametrize("m, distance, step, ElemType", [
    (primitives.Circle([0,0,0],1, Type='line'),
    10,
    1,
    None
    ),
    (primitives.Circle([0,0,0],1, Type='line'),
    10,
    1,
    'tri'
    ),
    (primitives.Circle([0,0,0],1, Type='surf'),
    10,
    1,
    None
    ),
    (primitives.Circle([0,0,0],1, Type='surf'),
    10,
    1,
    'tet'
    ),
])
def test_Extrude(m, distance, step, ElemType):

    r = primitives.Extrude(m, distance, step, ElemType=ElemType)

    if r.Type == 'surf':
        A = quality.Area(*r)
        assert np.all(A > 0), 'Inverted elements'
    elif r.Type == 'vol':
        V = quality.Volume(*r)
        assert np.all(V > 0), 'Inverted elements'

@pytest.mark.parametrize("m, angle, anglestep, axis, ElemType", [
    (primitives.Circle([0,0,0],1, Type='line'),
    np.pi/2,
    np.pi/6,
    0,
    None
    ),
    (primitives.Circle([0,0,0],1, Type='line'),
    np.pi/2,
    np.pi/6,
    0,
    'tri'
    ),
    (primitives.Circle([0,0,0],1, Type='surf'),
    -np.pi/2,
    -np.pi/6,
    0,
    None
    ),
    (primitives.Circle([0,0,0],1, Type='surf'),
    -np.pi/2,
    -np.pi/6,
    0,
    'tet'
    ),
    (primitives.Circle([0,0,0],1, Type='surf'),
    np.pi/2,
    np.pi/6,
    0,
    None
    ),
    (primitives.Circle([0,0,0],1, Type='surf'),
    np.pi/2,
    np.pi/6,
    0,
    'tet'
    ),
])
def test_Revolve(m, angle, anglestep, axis, ElemType):

    r = primitives.Revolve(m, angle, anglestep, axis=axis, ElemType=ElemType)

    # if r.Type == 'surf':
    #     A = quality.Area(*r)
    #     assert np.all(A > 0), 'Inverted elements'
    # elif r.Type == 'vol':
    #     V = quality.Volume(*r)
    #     assert np.all(V > 0), 'Inverted elements'