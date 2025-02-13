import pytest
import numpy as np
from mymesh import rays, primitives

@pytest.mark.parametrize("pt, ray, TriCoords, bidirectional, Intersection", [
    (
    [0,0,0],
    np.array([1,1,0]),
    np.array([
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
    ]),
    False,
    np.array([0.75, 0.75, 0.  ])
    ),
    (
    [0,0,0],
    -np.array([1,1,0]),
    np.array([
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
    ]),
    False,
    []
    ),
    (
    [0,0,0],
    np.array([1,1,0]),
    np.array([
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
    ]),
    True,
    np.array([0.75, 0.75, 0.  ])
    )
])
def test_RayTriangleIntersection(pt, ray, TriCoords, bidirectional, Intersection):

    ix = rays.RayTriangleIntersection(pt, ray, TriCoords, bidirectional=bidirectional)

    assert np.all(np.isclose(ix, Intersection)), 'Incorrect intersection.'

@pytest.mark.parametrize("pt, ray, TriCoords, bidirectional, Intersection", [
    # Case 1: intersection with two of the same triangle
    (
    [0,0,0],
    np.array([1,1,0]),
    np.array([[
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
        ],
        [
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
        ]
    ]),
    False,
    np.array([[0.75, 0.75, 0.  ], [0.75, 0.75, 0.  ]])
    ),
    # Case 2: non-intersection with two of the same triangle
    (
    [2,0,0],
    np.array([1,1,0]),
    np.array([[
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
        ],
        [
        [1,0,-1],
        [0,1,-1],
        [1,1,1]
        ]
    ]),
    False,
    np.empty((0,3))
    ),
])
def test_RayTrianglesIntersection(pt, ray, TriCoords, bidirectional, Intersection):

    ixidx, ix = rays.RayTrianglesIntersection(pt, ray, TriCoords, bidirectional=bidirectional)

    assert np.all(np.isclose(ix, Intersection)), 'Incorrect intersection.'

@pytest.mark.parametrize("pt, ray, xlim, ylim, zlim, Intersection", [
    # Case 1: unit cube intersection
    (
        [-1,0.5,0.5], 
        [1,0,0], 
        [0,1],
        [0,1],
        [0,1],
        True
    ),
    # Case 2: unit cube non-intersection
    (
        [-1,2,0.5], 
        [1,1,1], 
        [0,1],
        [0,1],
        [0,1],
        False
    )
])
def test_RayBoxIntersection(pt, ray, xlim, ylim, zlim, Intersection):

    ix = rays.RayBoxIntersection(pt, ray, xlim, ylim, zlim)

    assert ix == Intersection, 'Incorrect intersection.'

@pytest.mark.parametrize("pt, ray, xlim, ylim, zlim, Intersection", [
    # Case 1: unit cube intersection
    (
        [-1,0.5,0.5], 
        [1,0,0], 
        [[0,1],[0,1]],
        [[0,1],[0,1]],
        [[0,1],[1,2]],
        [True,False]
    ),
])
def test_RayBoxesIntersection(pt, ray, xlim, ylim, zlim, Intersection):

    ix = rays.RayBoxesIntersection(pt, ray, xlim, ylim, zlim)

    assert np.array_equal(ix, Intersection), 'Incorrect intersection.'

@pytest.mark.parametrize("pt, Normal, xlim, ylim, zlim, Intersection", [
    # Case 1: unit cube intersection
    (
        [0.5,0.5,0.5], 
        [1,0,0], 
        [0,1],
        [0,1],
        [0,1],
        True
    ),
    # Case 2: unit cube non-intersection
    (
        [1.5,0.5,0.5], 
        [1,0,0], 
        [0,1],
        [0,1],
        [0,1],
        False
    ),
])
def test_PlaneBoxIntersection(pt, Normal, xlim, ylim, zlim, Intersection):

    ix = rays.PlaneBoxIntersection(pt, Normal, xlim, ylim, zlim)

    assert ix == Intersection, 'Incorrect intersection.'

@pytest.mark.parametrize("pt, Normal, TriCoords, Intersection", [
    # Case 1: triangle intersection
    (
        [0.5,0.5,0.5], 
        [1,0,0], 
        [[0,0,0],[1,0,0],[1,1,0]],
        True
    ),
    # Case 2: triangle non-intersection
    (
        [1.5,0.5,0.5], 
        [1,0,0], 
        [[0,0,0],[1,0,0],[0.5,1,0]],
        False
    ),
])
def test_PlaneTriangleIntersection(pt, Normal, TriCoords, Intersection):

    ix = rays.PlaneTriangleIntersection(pt, Normal, TriCoords)

    assert ix == Intersection, 'Incorrect intersection.'

@pytest.mark.parametrize("pt, Normal, Tris, Intersection", [
    # Case 1: triangle intersection
    (
        [0.5,0.5,0.5], 
        [1,0,0], 
        [[[0,0,0],[1,0,0],[1,1,0]],[[0,0,0],[1,0,0],[1,1,0]]],
        [True, True]
    ),
    # Case 2: triangle non-intersection
    (
        [1.5,0.5,0.5], 
        [1,0,0], 
        [[[0,0,0],[1,0,0],[0.5,1,0]],[[0,0,0],[1,0,0],[1,1,0]]],
        [False, False]
    ),
])
def test_PlaneTrianglesIntersection(pt, Normal, Tris, Intersection):

    ix = rays.PlaneTrianglesIntersection(pt, Normal, Tris)

    assert np.array_equal(ix, Intersection), 'Incorrect intersection.'

# @pytest.mark.parametrize("Tri1, Tri2, edgeedge, Intersection", [
#     (
#         [[0,0,0],[1,0,0],[0.5,1,0]],
#         [[0.5,0,-1],[0.5,0,1],[0.5,1,0]],
#         False,
#         True
#     ),
# ])
# def test_TriangleTriangleIntersection(Tri1, Tri2, edgeedge, Intersection):

#     ix = rays.TriangleTriangleIntersection(Tri1,Tri2,edgeedge=False)
#     assert ix == Intersection, 'Incorrect intersection.'

@pytest.mark.parametrize("TriCoords, xlim, ylim, zlim, Intersection", [
    # Case 1:  intersection
    (
        [[0,0,0.5],[1,0,0.5],[1,1,0.5]],
        [0,1],[0,1],[0,1],
        True
    ),
    # Case 2: non-intersection
    (
        [[0,0,1.5],[1,0,1.5],[1,1,1.5]],
        [0,1],[0,1],[0,1],
        False
    ),
    # Case 3: fully inclosed triangle
    (
        [[0,0,0.5],[1,0,0.5],[1,1,0.5]],
        [-1,2],[-1,2],[-1,2],
        True
    ),
])
def test_TriangleBoxIntersection(TriCoords, xlim, ylim, zlim, Intersection):

    ix = rays.TriangleBoxIntersection(TriCoords, xlim, ylim, zlim)

    assert ix == Intersection, 'Incorrect intersection.'

@pytest.mark.parametrize("s1, s2, endpt_inclusive, expected_ix, expected_pt", [
    # case 1 - 2D intersection
    ([[-1,0,0],[1,0,0]],
     [[0,-1,0],[0,1,0]],
     False,
     True,
     [0,0,0]),
    # case 2 - 2D non-intersection
    ([[-3,0,0],[-2,0,0]],
     [[0,-1,0],[0,1,0]],
     False,
     False,
     np.empty((0,3))),
    # case 3 - 2D endpoint non-intersection
    ([[-1,0,0],[0,0,0]],
     [[0,-1,0],[0,1,0]],
     False,
     False,
     np.empty((0,3))),
    # case 4 - 2D endpoint intersection
    ([[-1,0,0],[0,0,0]],
     [[0,-1,0],[0,1,0]],
     True,
     True,
     [0,0,0]),
    # case 5 - 3D intersection
    ([[-1,0,1],[1,0,-1]],
     [[0,-1,-1],[0,1,1]],
     False,
     True,
     [0,0,0]),
    # case 6 - 3D non-intersection
    ([[-1,0,1],[1,0,2]],
     [[0,-1,0],[0,1,-1]],
     False,
     False,
     np.empty((0,3))),
])
def test_SegmentSegmentIntersection(s1, s2, endpt_inclusive, expected_ix, expected_pt):

    intersection, pt = rays.SegmentSegmentIntersection(s1, s2, return_intersection=True, endpt_inclusive=endpt_inclusive)
    
    assert expected_ix == intersection, 'Incorrect intersection classification'
    assert np.all(np.isclose(expected_pt, pt)), 'Incorrect intersection point'

@pytest.mark.parametrize("s1, s2, endpt_inclusive, expected_ix, expected_pt", [
    # case 1 - 2D intersection, end point exclusive
    ([[[-1,0,0],[1,0,0]],
      [[-3,0,0],[-2,0,0]],
      [[-1,0,0],[0,0,0]],
      [[-1,0,1],[1,0,-1]],
      [[-1,0,1],[1,0,2]]
    ],
    [[[0,-1,0],[0,1,0]],
     [[0,-1,0],[0,1,0]],
     [[0,-1,0],[0,1,0]],
     [[0,-1,-1],[0,1,1]],
     [[0,-1,0],[0,1,-1]]
    ],
     True,
     [True, False, True, True, False],
     np.array([[ 0.,  0.,  0.],
        [np.nan, np.nan, np.nan],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [np.nan, np.nan, np.nan]])
    ),
    # case 2 - 2D intersection, end point inclusive
    ([[[-1,0,0],[1,0,0]],
      [[-3,0,0],[-2,0,0]],
      [[-1,0,0],[0,0,0]],
      [[-1,0,1],[1,0,-1]],
      [[-1,0,1],[1,0,2]]
    ],
    [[[0,-1,0],[0,1,0]],
     [[0,-1,0],[0,1,0]],
     [[0,-1,0],[0,1,0]],
     [[0,-1,-1],[0,1,1]],
     [[0,-1,0],[0,1,-1]]
    ],
     False,
     [True, False, False, True, False],
     np.array([[ 0.,  0.,  0.],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [ 0.,  0.,  0.],
        [np.nan, np.nan, np.nan]])
    ),
])
def test_SegmentsSegmentsIntersection(s1, s2, endpt_inclusive, expected_ix, expected_pt):

    intersection, pt = rays.SegmentsSegmentsIntersection(s1, s2, return_intersection=True, endpt_inclusive=endpt_inclusive)
    
    for i in range(len(s1)):
        print(expected_pt[i], pt[i])
        assert expected_ix[i] == intersection[i], 'Incorrect intersection classification'
        assert np.all(np.isclose(expected_pt[i], pt[i], equal_nan=True)), 'Incorrect intersection point'

@pytest.mark.parametrize("pt, ray, segment, endpt_inclusive, expected_ix, expected_pt", [
    # case 1: 2d orthogonal intersection
    ([-2,0,0],
     [1,0,0],
     [[0,-1,0],[0,1,0]],
     True,
     True,
     [0,0,0]),
    # case 2: 2d orthogonal non-intersection
    ([-2,0,0],
     [-1,0,0],
     [[0,-1,0],[0,1,0]],
     True,
     False,
     np.empty((0,3))),
    # case 3: 2d angled intersection
    ([-1,-1,0],
     [1,1,0],
     [[1,-1,0],[-1,1,0]],
     True,
     True,
     [0,0,0]),
    # case 4: 2d angled endpoint non-intersection
    ([-1,0,0],
     [1,1,0],
     [[0,-1,0],[0,1,0]],
     False,
     False,
     np.empty((0,3))),
    # case 5: 2d angled endpoint intersection
    ([-1,0,0],
     [1,1,0],
     [[0,-1,0],[0,1,0]],
     True,
     True,
     [0,1,0]),
    # case 6: 3d intersection
    ([-1,-1,-1],
     [1,1,1],
     [[0,-1,0],[0,1,0]],
     True,
     True,
     [0,0,0]),
    # case 7: 3d planar non-intersection
    ([-1,-1,-1],
     [1,1,1],
     [[0,-1,0],[0,-0.5,0]],
     True,
     False,
     np.empty((0,3))),
     # case 8: 3d non-planar non-intersection
    ([-1,-1,-1],
     [0,0,1],
     [[0,-1,0],[0,1,0]],
     True,
     False,
     np.empty((0,3))),
])
def test_RaySegmentIntersection(pt, ray, segment, endpt_inclusive, expected_ix, expected_pt):

    intersection, pt = rays.RaySegmentIntersection(pt, ray, segment, return_intersection=True, endpt_inclusive=endpt_inclusive)
    
    assert expected_ix == intersection, 'Incorrect intersection classification'
    assert np.all(np.isclose(expected_pt, pt, equal_nan=True)), 'Incorrect intersection point'

@pytest.mark.parametrize("pt, ray, segments, endpt_inclusive, expected_ix, expected_pt", [
    # case 1: 2d intersections
    ([-2,0,0],
     [1,0,0],
     [[[0,-1,0],[0,1,0]],
      [[-1,-1,0],[1,1,0]],
      [[0,1,0],[0,2,0]],
      [[0,-1,0],[0,1,1]],
      [[0,-1,0],[0,0,0]]
     ],
     True,
     [True, True,False,False,True],
     [[0,0,0],
      [0,0,0],
      [np.nan, np.nan, np.nan],
      [np.nan, np.nan, np.nan],
      [0,0,0]
     ]),
     # case 2: 3d intersections
    ([0,0,-2],
     [0,0,1],
     [[[0,-1,0],[0,1,0]],
      [[-1,-1,-1],[1,1,1]],
      [[0,-1,0],[0,0,0]]
     ],
     False,
     [True, True,False],
     [[0,0,0],
      [0,0,0],
      [np.nan, np.nan, np.nan]
     ]),
])
def test_RaySegmentsIntersection(pt, ray, segments, endpt_inclusive, expected_ix, expected_pt):

    intersection, ixpt = rays.RaySegmentsIntersection(pt, ray, segments, return_intersection=True, endpt_inclusive=endpt_inclusive)
    
    assert np.all(expected_ix == intersection), 'Incorrect intersection classification'
    assert np.all(np.isclose(expected_pt, ixpt, equal_nan=True)), 'Incorrect intersection point'

@pytest.mark.parametrize("pt, NodeCoords, BoundaryConn, Inside", [
    # case 1 : point inside circle 
    (
        [0,0,0],
        *primitives.Circle([0,0,0],1,Type='line'),
        True
    ),
    # case 2 : point outside circle (planar)
    (
        [1,1,0],
        *primitives.Circle([0,0,0],1,Type='line'),
        False
    ),
    # case 3 : point outside circle (non-planar)
    (
        [0,0,1],
        *primitives.Circle([0,0,0],1,Type='line'),
        False
    ),
    # case 4 : point inside square
    (
        [-1,1,0],
        [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
        [[0,1],[1,2],[2,3],[3,1]],
        True
    ),
    # case 5 : point outside square
    (
        [2,1,0],
        [[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],
        [[0,1],[1,2],[2,3],[3,1]],
        False
    ),
])
def test_PointInBoundary(pt, NodeCoords, BoundaryConn, Inside):

    inside = rays.PointInBoundary(pt, NodeCoords, BoundaryConn, Inside)

    assert inside == Inside, 'Incorrect inclusion.'

@pytest.mark.parametrize("pt, NodeCoords, SurfConn, Inside", [
    # Case 1: point inside sphere
    (
        [0,0,0],
        *primitives.Sphere([0,0,0],1, ElemType='tri'),
        True
    ),
    # Case 2: point outside sphere
    (
        [2,2,2],
        *primitives.Sphere([0,0,0],1, ElemType='tri'),
        False
    ),
    # Case 3: point outside torus
    (
        [0,0,0],
        *primitives.Torus([0,0,0], 1, 0.5, ElemType='tri'),
        False
    ),
    # Case 4: point inside torus
    (
        [1,0,0],
        *primitives.Torus([0,0,0], 1, 0.5, ElemType='tri'),
        True
    ),
])
def test_PointInSurf(pt, NodeCoords, SurfConn, Inside):

    inside = rays.PointInSurf(pt, NodeCoords, SurfConn, Inside)

    assert inside == Inside, 'Incorrect inclusion.'

@pytest.mark.parametrize("pts, NodeCoords, SurfConn, Inside", [
    # Case 1: sphere
    (
        [[0,0,0],[2,2,2]],
        *primitives.Sphere([0,0,0],1, ElemType='tri'),
        [True,False]
    ),
    # Case 3: point outside torus
    (
        [[0,0,0], [1,0,0]],
        *primitives.Torus([0,0,0], 1, 0.5, ElemType='tri'),
        [False, True]
    ),
])
def test_PointsInSurf(pts, NodeCoords, SurfConn, Inside):

    inside = rays.PointsInSurf(pts, NodeCoords, SurfConn, Inside)

    assert np.array_equal(inside, Inside), 'Incorrect inclusion.'

