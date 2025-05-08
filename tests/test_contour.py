import pytest
import numpy as np
from mymesh import contour, quality, implicit, primitives

@pytest.mark.parametrize("I, interpolation, Type, flip",[
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'midpoint',
        'surf',
        False
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'linear',
        'surf',
        False
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'cubic',
        'surf',
        True
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'linear',
        'surf',
        True
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'midpoint',
        'line',
        False
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'linear',
        'line',
        False
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2)),0),
        'cubic',
        'line',
        False
    ),

])
def test_MarchingSquaresImage(I, interpolation, Type, flip):

    NodeCoords, NodeConn = contour.MarchingSquaresImage(I, interpolation=interpolation, Type=Type, flip=flip)

    assert not np.any(np.isnan(NodeCoords)), 'NaN in NodeCoords'

    A = quality.Area(NodeCoords, NodeConn)
    if Type == 'surf':
        assert np.all(A > 0), 'Inverted elements.'
    
@pytest.mark.parametrize("NodeCoords, NodeConn, NodeVals, interpolation, Type, flip",[
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        'surf',
        False
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'surf',
        False
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'surf',
        True
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        'line',
        False
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'line',
        False
    ),
])
def test_MarchingSquares(NodeCoords, NodeConn, NodeVals, interpolation, Type, flip):

    NewCoords, NewConn = contour.MarchingSquares(NodeCoords, NodeConn, NodeVals, interpolation=interpolation, Type=Type, flip=flip)

    assert not np.any(np.isnan(NewCoords)), 'NaN in NodeCoords'

    A = quality.Area(NewCoords, NewConn)
    if Type == 'surf':
        assert np.all(A > 0), 'Inverted elements.'
    
@pytest.mark.parametrize("NodeCoords, NodeConn, NodeVals, interpolation, Type, flip",[
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2, ElemType='tri'),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        'surf',
        False
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2, ElemType='tri'),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'surf',
        False
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2, ElemType='tri'),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'surf',
        True
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2, ElemType='tri'),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        'line',
        False
    ),
    (
        *primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2, ElemType='tri'),
        implicit.sphere([0,0,0],1)(*primitives.Grid2D([-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'line',
        False
    ),
])
def test_MarchingTriangles(NodeCoords, NodeConn, NodeVals, interpolation, Type, flip):

    NewCoords, NewConn = contour.MarchingTriangles(NodeCoords, NodeConn, NodeVals, interpolation=interpolation, Type=Type, flip=flip)

    assert not np.any(np.isnan(NewCoords)), 'NaN in NodeCoords'

    A = quality.Area(NewCoords, NewConn)
    if Type == 'surf':
        assert np.all(A > 0), 'Inverted elements.'
    
@pytest.mark.parametrize("I, interpolation, flip",[
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2))),
        'midpoint',
        False
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2))),
        'linear',
        False
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2))),
        'cubic',
        True
    ),
    (
        implicit.sphere([0,0,0],1)(*np.meshgrid(np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2),np.arange(-1.5,1.5,.2))),
        'linear',
        True
    ),

])
def test_MarchingCubesImage(I, interpolation, flip):

    NodeCoords, NodeConn = contour.MarchingCubesImage(I, interpolation=interpolation, flip=flip)

    assert not np.any(np.isnan(NodeCoords)), 'NaN in NodeCoords'

    A = quality.Area(NodeCoords, NodeConn)
    assert np.all(A > 0), 'Inverted elements.'

@pytest.mark.parametrize("NodeCoords, NodeConn, NodeVals, interpolation, method, flip",[
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        'original',
        False
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'original',
        False
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        'original',
        True
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        '33',
        False
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        '33',
        False
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sp2np(implicit.lidinoid)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        '33',
        False
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2),
        implicit.sp2np(implicit.thickenf(implicit.gyroid,1))(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        '33',
        False
    ),

])
def test_MarchingCubes(NodeCoords, NodeConn, NodeVals, interpolation, method, flip):

    NewCoords, NewConn = contour.MarchingCubes(NodeCoords, NodeConn, NodeVals, interpolation=interpolation, method=method, flip=flip)

    assert not np.any(np.isnan(NewCoords)), 'NaN in NodeCoords'

    A = quality.Area(NewCoords, NewConn)
    assert np.all(A > 0), 'Inverted elements.'

@pytest.mark.parametrize("NodeCoords, NodeConn, NodeVals, interpolation, flip, Type",[
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        False,
        'surf'
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        False,
        'surf'
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        True,
        'surf'
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        False,
        'vol'
    ),
])
def test_MarchingTetrahedra(NodeCoords, NodeConn, NodeVals, interpolation, flip, Type):

    NewCoords, NewConn = contour.MarchingTetrahedra(NodeCoords, NodeConn, NodeVals, interpolation=interpolation, flip=flip, Type=Type)

    assert not np.any(np.isnan(NewCoords)), 'NaN in NodeCoords'

    if Type == 'surf':
        A = quality.Area(NewCoords, NewConn)
        assert np.all(A > 0), 'Inverted elements.'
    elif Type == 'vol':
        V = quality.Volume(NewCoords, NewConn)
        assert np.all(V > 0), 'Inverted elements.'

@pytest.mark.parametrize("NodeCoords, NodeConn, NodeVals, interpolation, flip, Type",[
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'midpoint',
        False,
        'surf'
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        False,
        'surf'
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        True,
        'surf'
    ),
    (
        *primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2, ElemType='tet'),
        implicit.sphere([0,0,0],1)(*primitives.Grid([-1.5,1.5,-1.5,1.5,-1.5,1.5],0.2).NodeCoords.T),
        'linear',
        False,
        'vol'
    ),
    (
        *primitives.Sphere([0,0,0],1, Type='vol'),
        implicit.sphere([0.5,0,0],1)(*primitives.Sphere([0,0,0],1, Type='vol').NodeCoords.T),
        'linear',
        False,
        'vol'
    ),
    (
        *primitives.Sphere([0,0,0],1, Type='vol'),
        implicit.sphere([0.5,0,0],1)(*primitives.Sphere([0,0,0],1, Type='vol').NodeCoords.T),
        'linear',
        False,
        'surf'
    ),
    (
        *primitives.Sphere([0,0,0],1, Type='surf'),
        implicit.sphere([0.5,0,0],1)(*primitives.Sphere([0,0,0],1, Type='surf').NodeCoords.T),
        'linear',
        False,
        'surf'
    ),
    (
        *primitives.Sphere([0,0,0],1, Type='surf'),
        implicit.sphere([0.5,0,0],1)(*primitives.Sphere([0,0,0],1, Type='surf').NodeCoords.T),
        'linear',
        False,
        'line'
    ),
])
def test_MarchingElements(NodeCoords, NodeConn, NodeVals, interpolation, flip, Type):

    NewCoords, NewConn = contour.MarchingElements(NodeCoords, NodeConn, NodeVals, interpolation=interpolation, flip=flip, Type=Type)

    assert not np.any(np.isnan(NewCoords)), 'NaN in NodeCoords'

    if Type == 'surf':
        A = quality.Area(NewCoords, NewConn)
        assert np.all(A > 0), 'Inverted elements.'
    elif Type == 'vol':
        V = quality.Volume(NewCoords, NewConn)
        assert np.all(V > 0), 'Inverted elements.'

@pytest.mark.parametrize("func, bounds, method, Type",[
    (
        implicit.box(-1,1,-1,1,-1,1),
        [-2,2,-2,2,-2,2],
        'mc',
        'surf'
    ),
    (
        implicit.box(-1,1,-1,1,-1,1),
        [-2,2,-2,2,-2,2],
        'mc33',
        'surf'
    ),
    (
        implicit.box(-1,1,-1,1,-1,1),
        [-2,2,-2,2,-2,2],
        'mt',
        'surf'
    ),
    (
        implicit.box(-1,1,-1,1,-1,1),
        [-2,2,-2,2,-2,2],
        'mt',
        'vol'
    ),
])
def test_Adaptive(func, bounds, method, Type):

    NewCoords, NewConn = contour.Adaptive(func, bounds, method=method, Type=Type)

    assert not np.any(np.isnan(NewCoords)), 'NaN in NodeCoords'

    if Type == 'surf':
        A = quality.Area(NewCoords, NewConn)
        assert np.all(A > 0), 'Inverted elements.'
    elif Type == 'vol':
        V = quality.Volume(NewCoords, NewConn)
        assert np.all(V > 0), 'Inverted elements.'

@pytest.mark.parametrize("method",[
    (
        '33'
    ),
    (
        'original'
    ),
])
def test_generateLookup(method):

    if method == '33':
        LookupTable, Cases, FaceTests, Signs = contour._generateLookup33()
        assert len(FaceTests) == 256
        assert len(Signs) == 256
    else:
        LookupTable, Cases = contour._generateLookup()

    assert len(LookupTable) == 256
    assert len(Cases) == 256
