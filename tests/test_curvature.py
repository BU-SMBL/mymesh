import pytest
import numpy as np
from mymesh import curvature, implicit, primitives


@pytest.mark.parametrize("S, k1, k2", [
    # Case 1: unit sphere 
    (primitives.Sphere([0,0,0], 1, 60, 60),
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
    (primitives.Sphere([0,0,0], 1, 60, 60),
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
    (primitives.Sphere([0,0,0], 1, 60, 60),
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
