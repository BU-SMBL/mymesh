import pytest
import numpy as np
from mymesh import improvement, mesh, implicit, quality, primitives

@pytest.mark.parametrize("M, options", [
    # Case 1: surface mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0.4,0.4,0]]),np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])),
    dict()
    ),
    # Case 2: surface mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0.4,0.4,0]]),np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])),
    dict(iterate=1)
    ),
    # Case 3: volume mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1],[0.4,0.4,0.4]]),np.array([[0,1,2,3,8],[0,4,5,1,8],[1,5,6,2,8],[2,6,7,3,8],[0,3,7,4,8],[4,7,6,5,8]])),
    dict(iterate=1)
    ),
    ])
def test_LocalLaplacianSmoothing(M, options):

    Mnew = improvement.LocalLaplacianSmoothing(M, options)
    
    assert M.NNode == Mnew.NNode, 'Number of nodes incorrectly altered.'
    assert M.NElem == Mnew.NElem, 'Number of elements incorrectly altered.'

@pytest.mark.parametrize("M, options", [
    # Case 1: surface mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0.4,0.4,0]]),np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])),
    dict()
    ),
    # Case 2: surface mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0.4,0.4,0]]),np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])),
    dict(iterate=1)
    ),
    # Case 3: volume mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1],[0.4,0.4,0.4]]),np.array([[0,1,2,3,8],[0,4,5,1,8],[1,5,6,2,8],[2,6,7,3,8],[0,3,7,4,8],[4,7,6,5,8]])),
    dict(iterate=1)
    ),
    ])
def test_TangentialLaplacianSmoothing(M, options):

    Mnew = improvement.TangentialLaplacianSmoothing(M, options)
    
    assert M.NNode == Mnew.NNode, 'Number of nodes incorrectly altered.'
    assert M.NElem == Mnew.NElem, 'Number of elements incorrectly altered.'

@pytest.mark.parametrize("M, options", [
    # Case 1: surface mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0.4,0.4,0]]),np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])),
    dict()
    ),
    # Case 2: surface mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0.4,0.4,0]]),np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])),
    dict(iterate=1)
    ),
    # Case 3: volume mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1],[0.4,0.4,0.4]]),np.array([[0,1,2,3,8],[0,4,5,1,8],[1,5,6,2,8],[2,6,7,3,8],[0,3,7,4,8],[4,7,6,5,8]])),
    dict(iterate=1)
    ),
    ])
def test_SmartLaplacianSmoothing(M, options):

    Mnew = improvement.SmartLaplacianSmoothing(M, options)
    
    assert M.NNode == Mnew.NNode, 'Number of nodes incorrectly altered.'
    assert M.NElem == Mnew.NElem, 'Number of elements incorrectly altered.'

@pytest.mark.parametrize("M, options", [
    # Case 3: volume mesh
    (mesh(np.array([[0,0,0],[0,1,0],[1,1,0],[1,0,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1],[0.4,0.4,0.4]]),np.array([[0,1,2,3,8],[0,4,5,1,8],[1,5,6,2,8],[2,6,7,3,8],[0,3,7,4,8],[4,7,6,5,8]])),
    dict(iterate=1)
    ),
    ])
def test_SmartLaplacianSmoothing(M, options):

    Mnew = improvement.SmartLaplacianSmoothing(M, options)
    
    assert M.NNode == Mnew.NNode, 'Number of nodes incorrectly altered.'
    assert M.NElem == Mnew.NElem, 'Number of elements incorrectly altered.'

@pytest.mark.parametrize("M, h, labels", [
    # Case 1: solid sphere
    (
        implicit.TetMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .2,
        None
    ),
    # Case 2: surface sphere
    (
        implicit.SurfaceMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .2,
        None
    ),
    # Case 3: labeled solid spgere
    (
        implicit.TetMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .2,
        True
    ),
    # Case 4: labeled surface sphere
    (
        implicit.SurfaceMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .2,
        True
    ),
    ])
def test_Contract(M, h, labels):
    if labels is True:
        labels = np.zeros(M.NElem)
        labels[M.Centroids[:,0]<0] = 1
    Mnew = improvement.Contract(M, h, labels=labels)
    
    if labels is not None:
        assert len(Mnew.ElemData['labels']) == Mnew.NElem
    if M.Type == 'vol':
        # check for inversions
        assert np.all(quality.Volume(*Mnew) > 0), 'Inverted elements'
    elif M.Type == 'surf':
        assert len(Mnew.BoundaryNodes) == 0, 'Exposed Edges'

@pytest.mark.parametrize("M, h, labels", [
    # Case 1: solid sphere
    (
        implicit.TetMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .05,
        None
    ),
    # Case 2: surface sphere
    (
        implicit.SurfaceMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .05,
        None
    ),
    # Case 3: labeled solid sphere
    (
        implicit.TetMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .05,
        True
    ),
    # Case 4: labeled surface sphere
    (
        implicit.SurfaceMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .05,
        True
    ),
    ])
def test_Split(M, h, labels):
    if labels is True:
        labels = np.zeros(M.NElem)
        labels[M.Centroids[:,0]<0] = 1
    Mnew = improvement.Split(M, h, labels=labels)
    
    if labels is not None:
        assert len(Mnew.ElemData['labels']) == Mnew.NElem
    if M.Type == 'vol':
        # check for inversions
        assert np.all(quality.Volume(*Mnew) > 0), 'Inverted elements'
    elif M.Type == 'surf':
        assert len(Mnew.BoundaryNodes) == 0, 'Exposed Edges'

@pytest.mark.parametrize("M, h", [
    # Case 1: solid sphere
    (
        implicit.TetMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .2,
    ),
    # Case 2: surface sphere
    (
        implicit.SurfaceMesh(implicit.sphere([0,0,0], 1), [-1,1,-1,1,-1,1], .1),
        .05,
    ),
    # Case 3: 2d surface circle
    (
        primitives.Circle([0,0,0], 1, ElemType='tri', 
                        theta_resolution=40, radial_resolution=20),
        .01
    )
    ])
def test_Improve(M, h):

    Mnew = improvement.Improve(M, h, repeat=2, verbose=False)
    