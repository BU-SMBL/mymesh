import pytest
import numpy as np
from mymesh import converter, quality

@pytest.mark.parametrize("NodeCoords, NodeConn, return_SurfElem, expected",[
    # Single hex, False
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    False,
    [[0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7]]
    ),
    # Single hex, True
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    True,
    ([[0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7]],
    np.array([0, 0, 0, 0, 0, 0]))
    )
])
def test_solid2surface(NodeCoords, NodeConn, return_SurfElem, expected):

    out = converter.solid2surface(NodeCoords,NodeConn, return_SurfElem=return_SurfElem)

    if type(out) is tuple:
        for i in range(len(out)):
            assert np.all(out[i] == expected[i])
    else:
        assert np.all(out == expected)

@pytest.mark.parametrize("NodeCoords, NodeConn, return_FaceConn, return_FaceElem, expected",[
    # Single hex, False, False
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    False,
    False,
    [[0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7]]
    ),
    # Single hex, True, True
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    True,
    True,
    ([[0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7]],
    [[0, 1, 2, 3, 4, 5]],
    np.array([0, 0, 0, 0, 0, 0]))
    ),
    # Single hex, True, False
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    True,
    False,
    ([[0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7]],
    [[0, 1, 2, 3, 4, 5]])
    ),
    # Single hex, False, True
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    False,
    True,
    ([[0, 3, 2, 1],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [4, 5, 6, 7]],
    np.array([0, 0, 0, 0, 0, 0]))
    )   
])
def test_solid2faces(NodeCoords, NodeConn, return_FaceConn, return_FaceElem, expected):

    out = converter.solid2faces(NodeCoords,NodeConn, return_FaceConn=return_FaceConn, return_FaceElem=return_FaceElem)
    if type(out) is tuple:
        for i in range(len(out)):
            assert np.all(out[i] == expected[i])
    else:
        assert np.all(out == expected)

@pytest.mark.parametrize("NodeCoords, NodeConn, method, n_expected", [
    # Case 1: single hex, 1to5
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),[[0,1,2,3,4,5,6,7]], '1to5', 5),
    # Case 2: single hex, 1to6
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),[[0,1,2,3,4,5,6,7]], '1to6', 6),
    # Case 3: single hex, 1to25
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),[[0,1,2,3,4,5,6,7]], '1to24', 24),
])
def test_hex2tet(NodeCoords, NodeConn, method, n_expected):

    NewCoords, NewConn = converter.hex2tet(NodeCoords, NodeConn, method=method)

    assert len(NewConn) == n_expected, 'Incorrect number of tets created.'
    assert not np.any(np.isnan(NewCoords)), 'NaN introduced to NodeCoords.'
    assert np.min(quality.Volume(NewCoords,NewConn)) > 0, 'Inverted Elements.'
    assert np.shape(NewConn)[1] == 4, 'Tets not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, method, n_expected", [
    # Case 1: single wedge, 1to3
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1],[1,0,1],[1,1,1]]),[[0,1,2,3,4,5]], '1to3', 3),
    # Case 2: single wedge, 1to3c
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1],[1,0,1],[1,1,1]]),[[0,1,2,3,4,5]], '1to3c', 3),
    # Case 3: single wedge, 1to14
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1],[1,0,1],[1,1,1]]),[[0,1,2,3,4,5]], '1to14', 14),
    # Case 3: single wedge, 1to36
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1],[1,0,1],[1,1,1]]),[[0,1,2,3,4,5]], '1to36', 36),
])
def test_wedge2tet(NodeCoords, NodeConn, method, n_expected):

    NewCoords, NewConn = converter.wedge2tet(NodeCoords, NodeConn, method=method)

    assert len(NewConn) == n_expected, 'Incorrect number of tets created.'
    assert not np.any(np.isnan(NewCoords)), 'NaN introduced to NodeCoords.'
    assert np.min(quality.Volume(NewCoords,NewConn)) > 0, 'Inverted Elements.'
    assert np.shape(NewConn)[1] == 4, 'Tets not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, method, n_expected", [
    # Case 1: single pyramid, 1to2
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1]]),[[0,1,2,3,4]], '1to2', 2),
    # Case 2: single pyramid, 1to2c
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1]]),[[0,1,2,3,4]], '1to2c', 2),
    # Case 3: single pyramid, 1to4
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1]]),[[0,1,2,3,4]], '1to4', 4),
])
def test_pyramid2tet(NodeCoords, NodeConn, method, n_expected):

    NewCoords, NewConn = converter.pyramid2tet(NodeCoords, NodeConn, method=method)

    assert len(NewConn) == n_expected, 'Incorrect number of tets created.'
    assert not np.any(np.isnan(NewCoords)), 'NaN introduced to NodeCoords.'
    assert np.min(quality.Volume(NewCoords,NewConn)) > 0, 'Inverted Elements.'
    assert np.shape(NewConn)[1] == 4, 'Tets not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single tri
    (np.array([[0,0,0],[1,0,0],[1,1,0]]),[[0,1,2]], 3),
])
def test_tri2edges(NodeCoords, NodeConn, n_expected):

    Edges = converter.tri2edges(NodeCoords, NodeConn)

    assert len(Edges) == n_expected, 'Incorrect number of edges created.'
    assert np.shape(Edges)[1] == 2, 'Edges not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single quad
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),[[0,1,2,3]], 4),
])
def test_quad2edges(NodeCoords, NodeConn, n_expected):

    Edges = converter.quad2edges(NodeCoords, NodeConn)

    assert len(Edges) == n_expected, 'Incorrect number of edges created.'
    assert np.shape(Edges)[1] == 2, 'Edges not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single tet
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,1]]),[[0,1,2,3]], 6),
])
def test_tet2edges(NodeCoords, NodeConn, n_expected):

    Edges = converter.tet2edges(NodeCoords, NodeConn)

    assert len(Edges) == n_expected, 'Incorrect number of edges created.'
    assert np.shape(Edges)[1] == 2, 'Edges not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single pyramid
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1]]),[[0,1,2,3,4]], 8),
])
def test_pyramid2edges(NodeCoords, NodeConn, n_expected):

    Edges = converter.pyramid2edges(NodeCoords, NodeConn)

    assert len(Edges) == n_expected, 'Incorrect number of edges created.'
    assert np.shape(Edges)[1] == 2, 'Edges not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single wedge
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1],[1,0,1],[1,1,1]]),[[0,1,2,3,4,5]], 9),
])
def test_wedge2edges(NodeCoords, NodeConn, n_expected):

    Edges = converter.wedge2edges(NodeCoords, NodeConn)

    assert len(Edges) == n_expected, 'Incorrect number of edges created.'
    assert np.shape(Edges)[1] == 2, 'Edges not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),[[0,1,2,3,4,5,6,7]], 12),
])
def test_hex2edges(NodeCoords, NodeConn, n_expected):

    Edges = converter.hex2edges(NodeCoords, NodeConn)

    assert len(Edges) == n_expected, 'Incorrect number of edges created.'
    assert np.shape(Edges)[1] == 2, 'Edges not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn, n_expected", [
    # Case 1: single quad
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),[[0,1,2,3]], 2),
])
def test_quad2tri(NodeCoords, NodeConn, n_expected):

    NewCoords, NewConn = converter.quad2tri(NodeCoords, NodeConn)

    assert len(NewConn) == n_expected, 'Incorrect number of tets created.'
    assert not np.any(np.isnan(NewCoords)), 'NaN introduced to NodeCoords.'
    assert np.min(quality.Area(NewCoords,NewConn)) > 0, 'Inverted Elements.'
    assert np.shape(NewConn)[1] == 3, 'Tris not created properly.'

@pytest.mark.parametrize("NodeCoords, NodeConn", [
    # Case 1: single tet
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,0,1]]),[[0,1,2,3]]),
])
def test_tet42tet10(NodeCoords, NodeConn):

    NewCoords, NewConn = converter.tet42tet10(NodeCoords, NodeConn)
    
    assert np.shape(NewConn)[1] == 10, 'Incorrect number of nodes per element' 

@pytest.mark.parametrize("NodeCoords, NodeConn", [
    # Case 1: single tet
    (np.array([[0. , 0. , 0. ],
       [0. , 0. , 0.5],
       [0. , 0. , 1. ],
       [0.5, 0. , 0. ],
       [0.5, 0. , 0.5],
       [0.5, 0.5, 0. ],
       [0.5, 0.5, 0.5],
       [1. , 0. , 0. ],
       [1. , 0.5, 0. ],
       [1. , 1. , 0. ]]),
     [[0, 7, 9, 2, 3, 8, 5, 1, 4, 6]]),
])
def test_tet102tet4(NodeCoords, NodeConn):

    NewCoords, NewConn = converter.tet102tet4(NodeCoords, NodeConn)
    
    assert np.shape(NewConn)[1] == 4, 'Incorrect number of nodes per element' 
    assert np.all(quality.Volume(NewCoords, NewConn) > 0), 'Inverted elements'

@pytest.mark.parametrize("NodeCoords, NodeConn", [
    # Case 1: single hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),[[0,1,2,3,4,5,6,7]]),
])
def test_hex82hex20(NodeCoords, NodeConn):

    NewCoords, NewConn = converter.hex82hex20(NodeCoords, NodeConn)
    
    assert np.shape(NewConn)[1] == 20, 'Incorrect number of nodes per element' 

@pytest.mark.parametrize("NodeCoords, NodeConn", [
    # Case 1: single hex
    (np.array([[0. , 0. , 0. ],
       [0. , 0. , 0.5],
       [0. , 0. , 1. ],
       [0. , 0.5, 0. ],
       [0. , 0.5, 1. ],
       [0. , 1. , 0. ],
       [0. , 1. , 0.5],
       [0. , 1. , 1. ],
       [0.5, 0. , 0. ],
       [0.5, 0. , 1. ],
       [0.5, 1. , 0. ],
       [0.5, 1. , 1. ],
       [1. , 0. , 0. ],
       [1. , 0. , 0.5],
       [1. , 0. , 1. ],
       [1. , 0.5, 0. ],
       [1. , 0.5, 1. ],
       [1. , 1. , 0. ],
       [1. , 1. , 0.5],
       [1. , 1. , 1. ]]),
     np.array([[ 0, 12, 17,  5,  2, 14, 19,  7,  8, 15, 10,  3,  9, 16, 11,  4, 1, 13, 18,  6]])),
])
def test_hex202hex8(NodeCoords, NodeConn):

    NewCoords, NewConn = converter.hex202hex8(NodeCoords, NodeConn)
    
    assert np.shape(NewConn)[1] == 8, 'Incorrect number of nodes per element' 
    assert np.all(quality.Volume(NewCoords, NewConn) > 0), 'Inverted elements'