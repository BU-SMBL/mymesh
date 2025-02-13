import pytest
import numpy as np
from mymesh import improvement, mesh

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

