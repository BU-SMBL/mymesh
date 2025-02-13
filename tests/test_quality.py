import pytest
import numpy as np
from mymesh import quality

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [1]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [1]
    ),
])
def test_AspectRatio(NodeCoords, NodeConn, expected):

    q = quality.AspectRatio(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [1]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [1]
    ),
])
def test_Orthogonality(NodeCoords, NodeConn, expected):

    q = quality.Orthogonality(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [0]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [0]
    ),
])
def test_InverseOrthogonality(NodeCoords, NodeConn, expected):

    q = quality.InverseOrthogonality(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [1]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [1]
    ),
])
def test_OrthogonalQuality(NodeCoords, NodeConn, expected):

    q = quality.OrthogonalQuality(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [0]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [0]
    ),
])
def test_InverseOrthogonalQuality(NodeCoords, NodeConn, expected):

    q = quality.InverseOrthogonalQuality(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, simplexmethod, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    'size',
    [0]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    'size',
    [0]
    ),
    # case 3: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    'angle',
    [0]
    ),
    # case 4: Unit tri
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0]]),
    [[0,1,2]],
    'size',
    [0]
    ),
    # case 5: Unit tri
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0]]),
    [[0,1,2]],
    'angle',
    [0]
    ),
])
def test_Skewness(NodeCoords, NodeConn, simplexmethod, expected):

    q = quality.Skewness(NodeCoords, NodeConn, simplexmethod=simplexmethod, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [np.pi/2]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [np.arctan(2*np.sqrt(2))]
    ),
])
def test_MinDihedral(NodeCoords, NodeConn, expected):

    q = quality.MinDihedral(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [np.pi/2]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [np.arctan(2*np.sqrt(2))]
    ),
])
def test_MaxDihedral(NodeCoords, NodeConn, expected):

    q = quality.MaxDihedral(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [6]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [1**2 * np.sqrt(3)]
    ),
    # case 3: Unit square
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),
    [[0,1,2,3]],
    [1]
    ),
    # case 4: Unit tri
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0]]),
    [[0,1,2]],
    [np.sqrt(3)/4*1**2]
    ),
])
def test_Area(NodeCoords, NodeConn, expected):

    q = quality.Area(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'

@pytest.mark.parametrize("NodeCoords, NodeConn, expected",[
    # case 1: Unit hex
    (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]]),
    [[0,1,2,3,4,5,6,7]],
    [1]
    ),
    # case 2: Unit tet
    (np.array([[0,0,0],[1,0,0],[1/2,np.sqrt(3)/2,0],[1/2,np.sqrt(3)/6,np.sqrt(2)/np.sqrt(3)]]),
    [[0,1,2,3]],
    [1**3/(6*np.sqrt(2))]
    ),
])
def test_Volume(NodeCoords, NodeConn, expected):

    q = quality.Volume(NodeCoords, NodeConn, verbose=True)

    assert np.all(np.isclose(q, expected)), 'Incorrect quality calculated.'