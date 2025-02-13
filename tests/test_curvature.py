import pytest
import numpy as np
from mymesh import curvature, implicit, primitives


@pytest.mark.parametrize("S, k1, k2", [
    # Case 1: unit sphere 
    (primitives.Sphere([0,0,0], 1, 60, 60, ElemType='tri'),
    1, 1
    ),
    # Case 2: 2D flat grid
    (primitives.Grid2D([0,1,0,1], .05, ElemType='tri'),
    0,0
    ),
])
def test_NormCurve(S, k1, k2):

    k1_c, k2_c = curvature.NormCurve(S.NodeCoords, S.NodeConn, S.NodeNeighbors, S.NodeNormals)
    mean_k1 = np.nanmean(k1_c)
    mean_k2 = np.nanmean(k2_c)

    assert np.isclose(mean_k1, k1, atol=1e-1) and np.isclose(mean_k2, k2, atol=1e-1), 'Incorrect curvature'

@pytest.mark.parametrize("S, k1, k2", [
    # Case 1: unit sphere 
    (primitives.Sphere([0,0,0], 1, 60, 60, ElemType='tri'),
    1, 1
    ),
    # Case 2: 2D flat grid
    (primitives.Grid2D([0,1,0,1], .05, ElemType='tri'),
    0,0
    ),
])
def test_QuadFit(S, k1, k2):

    k1_c, k2_c = curvature.QuadFit(S.NodeCoords, S.NodeConn, S.NodeNeighbors, S.NodeNormals)
    mean_k1 = np.nanmean(k1_c)
    mean_k2 = np.nanmean(k2_c)

    assert np.isclose(mean_k1, k1, atol=1e-1) and np.isclose(mean_k2, k2, atol=1e-1), 'Incorrect curvature'

@pytest.mark.parametrize("S, k1, k2", [
    # Case 1: unit sphere 
    (primitives.Sphere([0,0,0], 1, 60, 60, ElemType='tri'),
    1, 1
    ),
    # Case 2: 2D flat grid
    (primitives.Grid2D([0,1,0,1], .05, ElemType='tri'),
    0,0
    ),
])
def test_CubicFit(S, k1, k2):

    k1_c, k2_c = curvature.CubicFit(S.NodeCoords, S.NodeConn, S.NodeNeighbors, S.NodeNormals)
    mean_k1 = np.nanmean(k1_c)
    mean_k2 = np.nanmean(k2_c)

    assert np.isclose(mean_k1, k1, atol=1e-1) and np.isclose(mean_k2, k2, atol=1e-1), 'Incorrect curvature'

@pytest.mark.parametrize("func, NodeCoords, k1, k2", [
    # Case 1: unit sphere 
    (implicit.sphere([0,0,0],1), implicit.SurfaceMesh(implicit.sphere([0,0,0],1),[-1,1,-1,1,-1,1],.1).NodeCoords,
    1, 1
    ),
    (implicit.box(-.5,.5,-.5,.5,-.5,.5), implicit.SurfaceMesh(implicit.box(-.5,.5,-.5,.5,-.5,.5),[-1,1,-1,1,-1,1],.1).NodeCoords,
    0, 0
    ),
])
def test_AnalyticalCurvature(func, NodeCoords, k1, k2):

    k1_a, k2_a, _, _ = curvature.AnalyticalCurvature(func, NodeCoords)
    mean_k1 = np.nanmean(k1_a)
    mean_k2 = np.nanmean(k2_a)

    assert np.isclose(mean_k1, k1, atol=1e-2) and np.isclose(mean_k2, k2, atol=1e-2), 'Incorrect curvature'

@pytest.mark.parametrize("MaxPrincipal, MinPrincipal, MeanCurvature", [
    # Case 1 
    (1, -1, 0),
    (1, 1, 1),
    (np.array([-1]),np.array([-1]),np.array([-1]))
])
def test_MeanCurvature(MaxPrincipal, MinPrincipal, MeanCurvature):
    mean = curvature.MeanCurvature(MaxPrincipal, MinPrincipal)
    if isinstance(MaxPrincipal, (tuple, list, np.ndarray)):
        assert np.all(mean == MeanCurvature), 'Incorrect mean curvature'
    else:
        assert mean == MeanCurvature, 'Incorrect mean curvature'

@pytest.mark.parametrize("MaxPrincipal, MinPrincipal, GaussianCurvature", [
    # Case 1 
    (1, -1, -1),
    (1, 1, 1),
    (np.array([-1]),np.array([-1]),np.array([1]))
])
def test_GaussianCurvature(MaxPrincipal, MinPrincipal, GaussianCurvature):
    gauss = curvature.GaussianCurvature(MaxPrincipal, MinPrincipal)
    if isinstance(MaxPrincipal, (tuple, list, np.ndarray)):
        assert np.all(gauss == GaussianCurvature), 'Incorrect Gaussian curvature'
    else:
        assert gauss == GaussianCurvature, 'Incorrect Gaussian curvature'

@pytest.mark.parametrize("MaxPrincipal, MinPrincipal, Curvedness", [
    # Case 1 
    (1, -1, 1),
    (1, 1, 1),
    (np.array([-1]),np.array([-1]),np.array([1]))
])
def test_Curvedness(MaxPrincipal, MinPrincipal, Curvedness):
    c = curvature.Curvedness(MaxPrincipal, MinPrincipal)
    if isinstance(MaxPrincipal, (tuple, list, np.ndarray)):
        assert np.all(c == Curvedness), 'Incorrect Curvedness'
    else:
        assert c == Curvedness, 'Incorrect Curvedness'

@pytest.mark.parametrize("MaxPrincipal, MinPrincipal, ShapeIndex", [
    # Case 1 
    (1, -1, 0),
    (1, 1, 1),
    (np.array([-1]),np.array([-1]),np.array([-1]))
])
def test_ShapeIndex(MaxPrincipal, MinPrincipal, ShapeIndex):
    s = curvature.ShapeIndex(MaxPrincipal, MinPrincipal)
    if isinstance(MaxPrincipal, (tuple, list, np.ndarray)):
        assert np.all(s == ShapeIndex), 'Incorrect Shape Index'
    else:
        assert s == ShapeIndex, 'Incorrect Shape Index'

@pytest.mark.parametrize("MaxPrincipal, MinPrincipal, ShapeCategory", [
    # Case 1 
    (1, -1, 4),
    (1, 1, 8),
    (np.array([-1]),np.array([-1]),np.array([0]))
])
def test_ShapeCategory(MaxPrincipal, MinPrincipal, ShapeCategory):
    s = curvature.ShapeIndex(MaxPrincipal, MinPrincipal)
    sc = curvature.ShapeCategory(s)
    if isinstance(MaxPrincipal, (tuple, list, np.ndarray)):
        assert np.all(sc == ShapeCategory), 'Incorrect Shape Category'
    else:
        assert sc == ShapeCategory, 'Incorrect Shape Category'