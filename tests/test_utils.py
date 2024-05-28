import pytest
import numpy as np
from mymesh import utils

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