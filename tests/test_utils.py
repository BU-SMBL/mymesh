import pytest
import numpy as np
from mymesh import utils, primitives, implicit

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

@pytest.mark.parametrize("Surf, Vol", [
    # Case 1: unit sphere (primitive)
    (primitives.Sphere([0,0,0], 1, 100, 100),
    4/3*np.pi*1**3
    ),
    # Case 2: unit sphere (implicit)
    (implicit.SurfaceMesh(implicit.sphere([0,0,0],1),[-1,1,-1,1,-1,1],.05),
    4/3*np.pi*1**3
    ),
    # Case 3: unit cube (primitive)
    (primitives.Box([0,1,0,1,0,1], .1, ElemType='tri'),
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


@pytest.mark.parametrize("Ragged, Expected", [
    # Case 1
    ([[],[1],[2,3],[4],[5,6,7],[8,9]],
    [[[]], [[1], [4]], [[2, 3], [8, 9]], [[5, 6, 7]]]
    )
])
def test_PadRagged(Ragged, fillval, Expected):

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
