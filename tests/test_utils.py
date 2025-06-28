import pytest
import numpy as np
import mymesh
from mymesh import utils, primitives, implicit, mesh, quality

@pytest.mark.parametrize("NodeCoords, NodeConn, ElemType, expected", [
    # Case 1: Single triangle on the XY plane
    (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), 
     [[0, 1, 2]], 
     'auto',
     [[1, 2],[0, 2], [0, 1]]),
    # Case 2: Two quads on the XY plane
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]]), 
     [[0, 1, 2, 3],[1, 4, 5, 2]], 
     'quad',
     [[3, 1], [0, 2, 4], [1, 5, 3], [2, 0], [1, 5], [4, 2]]),
    # Case 2: Two tets
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, -1]]), 
     [[0, 1, 2, 3],[4, 0, 1, 2]], 
     'auto',
     [[1, 4, 3, 2], [0, 4, 3, 2], [0, 4, 1, 3], [2, 1, 0], [1, 0, 2]]),
])
def test_getNodeNeighbors(NodeCoords,NodeConn,ElemType,expected):
    neighbors = utils.getNodeNeighbors(NodeCoords,NodeConn,ElemType=ElemType)
    # Sort because it doesn't matter if the ordering of changes
    sorted_neighbors = [sorted(n) for n in neighbors]
    sorted_expected = [sorted(n) for n in expected]
    assert sorted_neighbors == sorted_expected, "Incorrect node neighbors"

@pytest.mark.parametrize("NodeCoords, NodeConn, expected", [
    # Case 1: Single triangle on the XY plane
    (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), 
     [[0, 1, 2]], 
     [[0],[0],[0]]),
    # Case 2: Two quads on the XY plane
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]]), 
     [[0, 1, 2, 3],[1, 4, 5, 2]], 
     [[0],[0,1],[0,1],[0],[1],[1]]),
    # Case 2: Two tets
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, -1]]), 
     [[0, 1, 2, 3],[4, 0, 1, 2]], 
     [[0,1],[0,1],[0,1],[0],[1]]),
])
def test_getElemConnectivity(NodeCoords,NodeConn,expected):
    ElemConn = utils.getElemConnectivity(NodeCoords,NodeConn)
    # Sort because it doesn't matter if the ordering of changes for some reason
    sorted_conn = [sorted(n) for n in ElemConn]
    sorted_expected = [sorted(n) for n in expected]
    assert sorted_conn == sorted_expected, "Incorrect node-element connectivity"

@pytest.mark.parametrize("NodeCoords, NodeConn, nRings, expected", [  
    # Case 1: Two triangles - 1-ring
    (
    np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]),
    [[0,1,2], [1,3,2]],
    1,
    [[1,2],[0,2,3],[0,1,3],[1,2]]
    ),

    # Case 2: Two triangles - 2-ring
    (
    np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]),
    [[0,1,2], [1,3,2]],
    2,
    [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    ),
    
    # Case 3: Single quad - 1-ring
    (
    np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
    [[0,1,2,3]],
    1,
    [[1,3],[0,2],[1,3],[0,2]
    ]
    ),

    # Case 4: Single quad - 2-ring
    (
    np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
    [[0,1,2,3]],
    2,
    [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    ),
])
def test_getNodeNeighborhood(NodeCoords, NodeConn, nRings, expected):

    neighborhoods = utils.getNodeNeighborhood(NodeCoords, NodeConn, nRings)
    sorted_neighborhoods = [sorted(n) for n in neighborhoods]
    sorted_expected = [sorted(n) for n in expected]
    assert sorted_neighborhoods == sorted_expected, "Incorrect node neighborhoods"

@pytest.mark.parametrize("NodeCoords, NodeConn, Radius, expected", [  
    # Case 1: Two triangles - Radius = 1
    (
    np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]),
    [[0,1,2], [1,3,2]],
    1,
    [[1,2],[0,3],[0,3],[1,2]]
    ),

    # Case 2: Two triangles - Radius = 2
    (
    np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0]]),
    [[0,1,2], [1,3,2]],
    2,
    [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    ),
    
    # Case 3: Single quad - Radius = 1
    (
    np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
    [[0,1,2,3]],
    1,
    [[1,3],[0,2],[1,3],[0,2]
    ]
    ),

    # Case 4: Single quad - Radius = 2
    (
    np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
    [[0,1,2,3]],
    2,
    [[1,2,3],[0,2,3],[0,1,3],[0,1,2]]
    ),
])
def test_getNodeNeighborhoodByRadius(NodeCoords, NodeConn, Radius, expected):

    neighborhoods = utils.getNodeNeighborhoodByRadius(NodeCoords, NodeConn, Radius)
    sorted_neighborhoods = [sorted(n) for n in neighborhoods]
    sorted_expected = [sorted(n) for n in expected]
    assert sorted_neighborhoods == sorted_expected, "Incorrect node neighborhoods"

@pytest.mark.parametrize("NodeCoords, NodeConn, mode, expected", [
    # Case 1: Single triangle on the XY plane
    (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), 
     [[0, 1, 2]], 
     'edge',
     [[]],),
    # Case 2: Two quads on the XY plane
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0]]), 
     [[0, 1, 2, 3],[1, 4, 5, 2]], 
     'edge',
     [[1],[0]],
     ),
    # Case 2: Two tets
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, -1]]), 
     [[0, 1, 2, 3],[4, 0, 1, 2]], 
     'face',
     [[1],[0]],
     ),
])
def test_getElemNeighbors(NodeCoords,NodeConn,mode,expected):
    ElemNeighbors = utils.getElemNeighbors(NodeCoords,NodeConn,mode=mode)
    # Sort because it doesn't matter if the ordering of changes for some reason
    sorted_conn = [sorted(n) for n in ElemNeighbors]
    sorted_expected = [sorted(n) for n in expected]
    assert sorted_conn == sorted_expected, "Incorrect element neighbors"

@pytest.mark.parametrize("M, expected", [
    # Case 1: Single sphere
    (primitives.Sphere([0,0,0],1), 1),
    # Case 2: Two spheres
    (mesh(*utils.MergeMesh(*primitives.Sphere([0,0,0],1),*primitives.Sphere([3,0,0],1))), 2),
    # Case 3: Three spheres
    (mesh(*utils.MergeMesh(*utils.MergeMesh(*primitives.Sphere([0,0,0],1),*primitives.Sphere([3,0,0],1)),*primitives.Sphere([3,0,3],1))), 3),
])
def test_getConnectedNodes(M, expected):
    
    R = utils.getConnectedNodes(*M)
    assert len(R) == expected, 'Incorrect number of regions identified.'

@pytest.mark.parametrize("M, mode, expected", [
    # Case 1: Single sphere (surface)
    (primitives.Sphere([0,0,0],1), 'edge', 1),
    # Case 2: Two spheres (surface)
    (mesh(*utils.MergeMesh(*primitives.Sphere([0,0,0],1),*primitives.Sphere([3,0,0],1))), 'edge', 2),
    # Case 3: Three spheres (surface)
    (mesh(*utils.MergeMesh(*utils.MergeMesh(*primitives.Sphere([0,0,0],1),*primitives.Sphere([3,0,0],1)),*primitives.Sphere([3,0,3],1))), 'node', 3),
    # Case 4: Two spheres (volume)
    (mesh(*utils.MergeMesh(*implicit.TetMesh(implicit.sphere([0,0,0],1),[-1,1,-1,1,-1,1],.1),*implicit.TetMesh(implicit.sphere([3,0,0],1),[2,4,-1,1,-1,1],.1))), 'face', 2),
])
def test_getConnectedElements(M, mode, expected):
    
    R = utils.getConnectedElements(*M, mode=mode)
    assert len(R) == expected, 'Incorrect number of regions identified.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected", [
    # Case 1: Unit hexahedron centered at (0.5, 0.5, 0.5)
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.],  
            [1., 1., 0.],  
            [0., 1., 0.],  
            [0., 0., 1.],  
            [1., 0., 1.],  
            [1., 1., 1.],  
            [0., 1., 1.],  
        ]),
        np.array([[0, 1, 2, 3, 4, 5, 6, 7]]),
        np.array([[0.5, 0.5, 0.5]])
    ),
    
    # Case 2: Unit tetrahedron with known centroid at (0.25, 0.25, 0.25)
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.],  
            [0., 1., 0.],  
            [0., 0., 1.],  
        ]),
        np.array([[0, 1, 2, 3]]),
        np.array([[0.25, 0.25, 0.25]])
    ),
    
    # Case 3: Two unit triangles (testing multiple elements)
    (
        np.array([
            [0., 0., 0.], 
            [1., 0., 0.],  
            [0., 1., 0.],  
            [1., 1., 0.],  
        ]),
        np.array([
            [0, 1, 2],    
            [1, 3, 2]     
        ]),
        np.array([
            [1/3, 1/3, 0.],  
            [2/3, 2/3, 0.]   
        ])
    ),
    
    # Case 4: Unit quad in XY plane 
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.],  
            [1., 1., 0.],  
            [0., 1., 0.],  
        ]),
        np.array([[0, 1, 2, 3]]),
        np.array([[0.5, 0.5, 0.]])
    ),
    
    # Case 5: Single line element (testing 1D element)
    (
        np.array([
            [0., 0., 0.],  
            [1., 1., 1.],  
        ]),
        np.array([[0, 1]]),
        np.array([[0.5, 0.5, 0.5]])
    ),
    # Case 6: Mixed element mesh (quad and triangle)
    (
        np.array([
            [0., 0., 0.],  # vertex 0
            [1., 0., 0.],  # vertex 1
            [1., 1., 0.],  # vertex 2
            [0., 1., 0.],  # vertex 3
            [2., 0., 0.],  # vertex 4
            [2., 1., 0.],  # vertex 5
        ]),
        [[0, 1, 2, 3],    # quad element
         [1, 4, 5]],      # triangle element
        np.array([
            [0.5, 0.5, 0.],    # centroid of quad
            [5/3, 1/3, 0.]     # centroid of triangle
        ])
    )
])
def test_Centroids(NodeCoords, NodeConn, expected):
    centroids = utils.Centroids(NodeCoords, NodeConn)
    np.testing.assert_array_almost_equal_nulp(centroids, expected)

@pytest.mark.parametrize("NodeCoords, SurfConn, expected", [
    # Case 1: Single triangle in XY plane (normal in +Z)
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.], 
            [0., 1., 0.]  
        ]),
        [[0, 1, 2]],
        np.array([[0., 0., 1.]])  
    ),
    # Case 2: Unit quad in XY plane (normal in +Z)
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.],  
            [1., 1., 0.],  
            [0., 1., 0.]   
        ]),
        [[0, 1, 2, 3]],
        np.array([[0., 0., 1.]])  
    ),
    # Case 3: Multiple triangles with different orientations
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.],  
            [0., 1., 0.],  
            [0., 0., 1.]   
        ]),
        [[0, 1, 2],       # triangle in XY plane
         [0, 2, 3]],      # triangle in YZ plane
        np.array([
            [0., 0., 1.],  # normal in +Z direction
            [1., 0., 0.]   # normal in +X direction
        ])
    ),
    # Case 4: Mixed elements (triangle and quad)
    (
        np.array([
            [0., 0., 0.],  
            [1., 0., 0.],  
            [1., 1., 0.],  
            [0., 1., 0.],  
            [2., 0., 0.],  
            [2., 1., 0.]   
        ]),
        [[0, 1, 2, 3],    # quad in XY plane
         [1, 4, 5]],      # triangle in XY plane
        np.array([
            [0., 0., 1.],  # quad normal in +Z
            [0., 0., 1.]   # triangle normal in +Z
        ])
    ),
])
def test_CalFaceNormal(NodeCoords, SurfConn, expected):

    ElemNormals = utils.CalcFaceNormal(NodeCoords, SurfConn)
    np.testing.assert_array_almost_equal_nulp(ElemNormals, expected)

@pytest.mark.parametrize("NodeCoords, NodeConn, method, expected",
    [
    # 2 Flat triangles in XY plane (method='Angle')
    (
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.]
        ]),
        [[0, 1, 2], [0, 2, 3]],
        'Angle',
        np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    ),
    # 2 Flat triangles in XY plane (method='Average')
    (
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.]
        ]),
        [[0, 1, 2], [0, 2, 3]],
        'Average',
        np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    ),
    # 4 Flat triangles in XY plane (method='MostVisible')
    (
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [.5, .5, 0.]
        ]),
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        'MostVisible',
        np.array([[0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [0., 0., 1.]])
    ),
    ]
)
def test_Face2NodeNormal(NodeCoords, NodeConn, method, expected):
    
    ElemConn = utils.getElemConnectivity(NodeCoords,NodeConn)
    ElemNormals = utils.CalcFaceNormal(NodeCoords,NodeConn)
    NodeNormals = utils.Face2NodeNormal(NodeCoords, NodeConn, ElemConn, ElemNormals, method=method)

    np.testing.assert_array_almost_equal_nulp(NodeNormals, expected)

@pytest.mark.parametrize("Nodes, Pt, expected", [
    # Case 1: Pt at first vertex
    (np.array([[0,0,0],[0,1,0],[1,1,0]]),
    np.array([0,0,0]),
    (1,0,0)
    ),
    # Case 2: Pt at second vertex
    (np.array([[0,0,0],[0,1,0],[1,1,0]]),
    np.array([0,1,0]),
    (0,1,0)
    ),
    # Case 3: Pt at third vertex
    (np.array([[0,0,0],[0,1,0],[1,1,0]]),
    np.array([1,1,0]),
    (0,0,1)
    ),
    # Case 4: Pt at centroid vertex
    (np.array([[0,0,0],[0,1,0],[1,1,0]]),
    np.array([1/3,2/3,0]),
    (1/3, 1/3, 1/3)
    ),
])
def test_BaryTri(Nodes, Pt, expected):

    BaryCoords = utils.BaryTri(Nodes, Pt)
    np.testing.assert_array_almost_equal_nulp(np.array(BaryCoords), np.array(expected))
    assert np.isclose(np.sum(BaryCoords), 1), "Barycentric coordinates don't add to 1."
    if mymesh.check_numba():
        BaryCoords = utils.BaryTri.py_func(Nodes, Pt)

@pytest.mark.parametrize("Nodes, Pt, expected", [
    # Case 1: Single point
    (np.array([[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[0,1,0],[1,1,0]]]),
    np.array([0,0,0]),
    (np.array([1,1,1,1]),np.array([0,0,0,0]),np.array([0,0,0,0]))
    ),
    # Case 2: Pairwise points
    (np.array([[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[0,1,0],[1,1,0]],[[0,0,0],[0,1,0],[1,1,0]]]),
    np.array([[0,0,0], [0,1,0], [1,1,0], [1/3,2/3,0]]),
    (np.array([1,0,0,1/3]),np.array([0,1,0,1/3]),np.array([0,0,1,1/3]))
    ),
])
def test_BaryTris(Nodes, Pt, expected):

    BaryCoords = utils.BaryTris(Nodes, Pt)
    np.testing.assert_array_almost_equal_nulp(np.array(BaryCoords), np.array(expected))
    assert np.all(np.isclose(np.sum(BaryCoords,axis=0), 1)), "Barycentric coordinates don't add to 1."

@pytest.mark.parametrize("Nodes, Pt, expected", [
    # Case 1: Pt at first vertex
    (np.array([[0.,0,0],[0,1,0],[1,1,0],[0,0,1]]),
    np.array([0.,0,0]),
    (1,0,0,0)
    ),
    # Case 2: Pt at second vertex
    (np.array([[0.,0,0],[0,1,0],[1,1,0],[0,0,1]]),
    np.array([0.,1,0]),
    (0,1,0,0)
    ),
    # Case 3: Pt at third vertex
    (np.array([[0.,0,0],[0,1,0],[1,1,0],[0,0,1]]),
    np.array([1.,1,0]),
    (0,0,1,0)
    ),
    # Case 4: Pt at third vertex
    (np.array([[0.,0,0],[0,1,0],[1,1,0],[0,0,1]]),
    np.array([0.,0,1]),
    (0,0,0,1)
    ),
    # Case 5: Pt at centroid vertex
    (np.array([[0.,0,0],[0,1,0],[1,1,0],[0,0,1]]),
    np.array([0.25, 0.5, 0.25]),
    (0.25, 0.25, 0.25, 0.25)
    ),
])
def test_BaryTet(Nodes, Pt, expected):

    BaryCoords = utils.BaryTet(Nodes, Pt)
    np.testing.assert_array_almost_equal_nulp(np.array(BaryCoords), np.array(expected))
    assert np.isclose(np.sum(BaryCoords), 1), "Barycentric coordinates don't add to 1."
    if mymesh.check_numba():
        BaryCoords = utils.BaryTet.py_func(Nodes, Pt)

@pytest.mark.parametrize("NodeCoords, NodeConn", [
    # Case 1: Two tets
    (np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0], [1, 0, 0], [1, 1, 0]]), 
     [[0, 1, 2, 3],[4, 5, 6, 7]], 
    )
])
def test_DeleteDuplicateNodes(NodeCoords, NodeConn):

    NewCoords, NewConn = utils.DeleteDuplicateNodes(NodeCoords, NodeConn)
    assert len(np.unique(NewCoords, axis=0)) == len(NewCoords), 'Duplicate nodes remain.'
    assert np.min(quality.Volume(NewCoords, NewConn)) > 0, 'Elements inverted by deleting duplicate nodes.'

@pytest.mark.parametrize("NodeCoords, NodeConn", [
    ([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1]], np.array([[0,1,2],[0,1,4]])),

    ([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1]], [[0,1,2,3],[0,1,5]])
])
def test_RemoveNodes(NodeCoords, NodeConn):

    CleanCoords, CleanConn, ids = utils.RemoveNodes(NodeCoords, NodeConn)
    assert len(np.unique([n for elem in CleanConn for n in elem]) == len(CleanCoords))

@pytest.mark.parametrize("NodeCoords, NodeConn, Type, expected", [
    # Case 1: Hex with collapsed face (-> wedge)
    (
        np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]),
        [[0,1,2,3,4,5,5,4]],  # Top face collapsed to line
        'auto',
        [[0,3,4,1,2,5]]     # Should reduce to wedge
    ),
    
    # Case 2: Hex -> Pyramid (four nodes merged to one)
    (
        np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]),
        [[0,1,2,3,4,4,4,4]],  # All top nodes are same
        'auto',
        [[0,1,2,3,4]]       # Should reduce to pyramid
    ),
    
    # Case 3: Hex -> Tet (multiple merged nodes)
    (
        np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]),
        [[0,1,1,0,2,2,3,2]],  # Degenerate pattern that should form tet
        'auto',
        [[0,1,2,3]]         # Should reduce to tet
    ),
    
    # Case 4: Wedge -> Tet (merged nodes)
    (
        np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [0,1,1]]),
        [[0,1,2,3,3,3]],      # Three top nodes merged
        'auto',
        [[0,1,2,3]]         # Should reduce to tet
    ),
    
    # Case 5: Pyramid -> Tet (merged nodes)
    (
        np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1]]),
        [[0,1,2,2,3]],        # Two nodes on quad face merged
        'auto',
        [[0,1,2,3]]         # Should reduce to tet
    ),
    
    # Case 6: Degenerate Tet (should be removed)
    (
        np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]),
        [[0,0,1,2]],          # Two nodes same
        'vol',
        []                  # Should be removed
    ),
    # 2D Elements
    # Case 7: Quad -> Tri (merged nodes)
    (
        np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]]),
        [[0,1,2,2]],          # Two nodes merged
        'surf',
        [[0,1,2]]           # Should reduce to triangle
    ),
    
    # Case 8: Degenerate Triangle (should be removed)
    (
        np.array([[0,0,0], [1,0,0], [0,1,0]]),
        [[0,0,1]],            # Two nodes same
        'auto',
        []                  # Should be removed
    ),
    
    # Case 9: Fully degenerate Hex (all same node)
    (
        np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]]),
        [[0,0,0,0,0,0,0,0]],  # All nodes same
        'auto',
        []                  # Should be removed
    ),
])
def test_CleanupDegenerateElements(NodeCoords, NodeConn, Type, expected):

    NewCoords, NewConn = utils.CleanupDegenerateElements(NodeCoords, NodeConn, Type=Type, return_idx=False)

    assert np.array_equal(NewConn, expected)
    if len(NewConn) > 0:
        assert np.min(quality.Volume(NewCoords, NewConn)) >= 0, 'Elements inverted by cleanup.'

@pytest.mark.parametrize("Surf, Vol", [
    # Case 1: unit sphere (primitive)
    (primitives.Sphere([0,0,0], 1, 100, 100, ElemType='tri'),
    4/3*np.pi*1**3
    ),
    # Case 2: unit sphere (implicit)
    (implicit.SurfaceMesh(implicit.sphere([0,0,0],1),[-1,1,-1,1,-1,1],.05),
    4/3*np.pi*1**3
    ),
    # Case 3: unit cube (primitive)
    (primitives.Grid([0,1,0,1,0,1], .1, ElemType='tri', Type='surf'),
    1
    ),
    # Case 4: unit cube (implicit)
    (implicit.SurfaceMesh(implicit.box(0,1,0,1,0,1),[-.1,1.1,-.1,1.1,-.1,1.1],.1),
    1
    ),
])
def test_TriSurfVol(Surf, Vol):

    SurfVol = utils.TriSurfVol(*Surf)

    assert np.isclose(SurfVol, Vol, atol=1e-2), 'Incorrect volume'

@pytest.mark.parametrize("M, Vol", [
    # Case 1: unit sphere (implicit)
    (implicit.TetMesh(implicit.sphere([0,0,0],1),[-1,1,-1,1,-1,1],.05),
    4/3*np.pi*1**3
    ),
    # Case 2: unit cube (primitive)
    (primitives.Grid([0,1,0,1,0,1], .1, ElemType='tet'),
    1
    ),
    # Case 3: unit cube (implicit)
    (implicit.TetMesh(implicit.box(0,1,0,1,0,1),[-.1,1.1,-.1,1.1,-.1,1.1],.05),
    1
    ),
])
def test_TetMeshVol(M, Vol):

    MeshVol = utils.TetMeshVol(*M)

    assert np.isclose(MeshVol, Vol, atol=1e-2), 'Incorrect volume'

@pytest.mark.parametrize("Points, BB", [
    # simple cube
    (primitives.Grid([-1,1,-1,1,-1,1],.1).NodeCoords, 
    np.array([[-1.,-1.,-1.],
       [ 1., -1., -1.],
       [ 1.,  1., -1.],
       [-1.,  1., -1.],
       [-1., -1.,  1.],
       [ 1., -1.,  1.],
       [ 1.,  1.,  1.],
       [-1.,  1.,  1.]])
       ),
    # rotated cube
    ((np.array([[ 1.        ,  0.        ,  0.        ],
       [ 0.        ,  np.cos(np.pi/4), -np.cos(np.pi/4)],
       [ 0.        ,  np.sin(np.pi/4),  np.cos(np.pi/4)]])@primitives.Grid([-1,1,-1,1,-1,1],.1).NodeCoords.T).T,
    np.array([[-1.       , -np.sqrt(8)/2,  0.        ],
             [ 1.        , -np.sqrt(8)/2,  0.        ],
             [ 1.        ,  0.          , -np.sqrt(8)/2],
             [-1.        ,  0.          , -np.sqrt(8)/2],
             [-1.        ,  0.          ,  np.sqrt(8)/2],
             [ 1.        ,  0.          ,  np.sqrt(8)/2],
             [ 1.        ,  np.sqrt(8)/2,  0.        ],
             [-1.        ,  np.sqrt(8)/2,  0.        ]])
             )
])
def test_MVBB(Points, BB):

    MVBB = utils.MVBB(Points)
    
    assert np.all(np.isclose(MVBB, BB)), 'Incorrect Bounding Box'
    
@pytest.mark.parametrize("Points, BB", [
    # simple cube
    (primitives.Grid([-1,1,-1,1,-1,1],.1).NodeCoords, 
    np.array([[-1.,-1.,-1.],
       [ 1., -1., -1.],
       [ 1.,  1., -1.],
       [-1.,  1., -1.],
       [-1., -1.,  1.],
       [ 1., -1.,  1.],
       [ 1.,  1.,  1.],
       [-1.,  1.,  1.]])
       ),
    # rotated cube
    ((np.array([[ 1.        ,  0.        ,  0.        ],
       [ 0.        ,  np.cos(np.pi/4), -np.cos(np.pi/4)],
       [ 0.        ,  np.sin(np.pi/4),  np.cos(np.pi/4)]])@primitives.Grid([-1,1,-1,1,-1,1],.1).NodeCoords.T).T,
    np.array([[-1.       , -np.sqrt(8)/2, -np.sqrt(8)/2],
             [ 1.        , -np.sqrt(8)/2, -np.sqrt(8)/2],
             [ 1.        ,  np.sqrt(8)/2, -np.sqrt(8)/2],
             [-1.        ,  np.sqrt(8)/2, -np.sqrt(8)/2],
             [-1.        , -np.sqrt(8)/2,  np.sqrt(8)/2],
             [ 1.        , -np.sqrt(8)/2,  np.sqrt(8)/2],
             [ 1.        ,  np.sqrt(8)/2,  np.sqrt(8)/2],
             [-1.        ,  np.sqrt(8)/2,  np.sqrt(8)/2]])
             )
])
def test_AABB(Points, BB):

    AABB = utils.AABB(Points)
    
    assert np.all(np.isclose(AABB, BB)), 'Incorrect Bounding Box'

@pytest.mark.parametrize("Ragged, Expected", [
    # Case 1
    ([[],[2,3],[4],[5,6,7]],
    [[], [4], [2, 3], [5, 6, 7]]
    )
])
def test_SortRaggedByLength(Ragged, Expected):

    Padded = utils.SortRaggedByLength(Ragged)

    assert Padded == Expected, "Incorrect sorting"

@pytest.mark.parametrize("Ragged, Expected", [
    # Case 1
    ([[],[1],[2,3],[4],[5,6,7],[8,9]],
    [[[]], [[1], [4]], [[2, 3], [8, 9]], [[5, 6, 7]]]
    )
])
def test_SplitRaggedByLength(Ragged, Expected):

    Padded = utils.SplitRaggedByLength(Ragged)

    assert Padded == Expected, "Incorrect splitting"

@pytest.mark.parametrize("Ragged, fillval, Expected", [
    # Case 1
    ([[],[1],[2,3],[4],[5,6,7],[8,9]],
    -1,
    np.array([[-1,-1,-1],[1,-1,-1],[2,3,-1],[4,-1,-1],[5,6,7],[8,9,-1]])),
    # Case 2
    ([[],[1],[2,3],[4],[5,6,7],[8,9]],
    np.nan,
    np.array([[np.nan,np.nan,np.nan],[1,np.nan,np.nan],[2,3,np.nan],[4,np.nan, np.nan],[5,6,7],[8,9,np.nan]])),
])
def test_PadRagged(Ragged, fillval, Expected):

    Padded = utils.PadRagged(Ragged, fillval=fillval)

    assert np.array_equal(Padded,Expected, equal_nan=True), "Incorrect padding"

@pytest.mark.parametrize("Padded, delval, dtype, Expected", [
    # Case 1
    (np.array([[-1,-1,-1],[1,-1,-1],[2,3,-1],[4,-1,-1],[5,6,7],[8,9,-1]]),
    -1,
    int,
    [[],[1],[2,3],[4],[5,6,7],[8,9]]),
    # Case 2
    (np.array([[np.nan,np.nan,np.nan],[1,np.nan,np.nan],[2,3,np.nan],[4,np.nan, np.nan],[5,6,7],[8,9,np.nan]]),
    np.nan, 
    int,
    [[],[1],[2,3],[4],[5,6,7],[8,9]]),
])
def test_ExtractRagged(Padded, delval, dtype, Expected):

    Ragged = utils.ExtractRagged(Padded, delval=delval, dtype=dtype)

    assert np.all([np.array_equal(ragged, expected) for ragged, expected in zip(Ragged,Expected)]), "Incorrect extraction"

@pytest.mark.parametrize("M, Type", [
    # Case 1: volume mesh
    (implicit.TetMesh(implicit.sphere([0,0,0],1),[-1,1,-1,1,-1,1],.05),
    'vol'
    ),
    # Case 2: surface mesh
    (primitives.Sphere([0,0,0], 1, Type='surf'),
    'surf'
    ),
    # Case 3: line mesh
    (primitives.Line([0,0,0], [1,1,1], n=10),
    'line'
    ),
    ])
def test_identify_type(M, Type):

    identified = utils.identify_type(M.NodeCoords, M.NodeConn)
    assert identified == Type, 'Incorrect Type.'
