import pytest
import numpy as np
import mymesh
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

    k1_c, k2_c = curvature.CubicFit(S.NodeCoords, S.NodeConn, S.NodeNeighbors, S.NodeNormals, jit=False)
    mean_k1 = np.nanmean(k1_c)
    mean_k2 = np.nanmean(k2_c)

    assert np.isclose(mean_k1, k1, atol=1e-1) and np.isclose(mean_k2, k2, atol=1e-1), 'Incorrect curvature'

@pytest.mark.parametrize("NodeCoords, neighborhood, normals", [
    (
        np.array([[ 0.00000000e+00,  1.22464680e-16, -1.00000000e+00],
                    [ 1.33159821e-01, -9.67462728e-02, -9.86361303e-01],
                    [ 9.67462728e-02,  1.33159821e-01, -9.86361303e-01],
                    [-2.01570238e-17, -1.64594590e-01, -9.86361303e-01],
                    [ 5.08625256e-02, -1.56538758e-01, -9.86361303e-01],
                    [ 1.33159821e-01,  9.67462728e-02, -9.86361303e-01],
                    [-9.67462728e-02, -1.33159821e-01, -9.86361303e-01],
                    [ 1.56538758e-01,  5.08625256e-02, -9.86361303e-01],
                    [-9.67462728e-02,  1.33159821e-01, -9.86361303e-01],
                    [-5.08625256e-02, -1.56538758e-01, -9.86361303e-01],
                    [-5.08625256e-02,  1.56538758e-01, -9.86361303e-01],
                    [-1.33159821e-01,  9.67462728e-02, -9.86361303e-01],
                    [-1.33159821e-01, -9.67462728e-02, -9.86361303e-01],
                    [-1.56538758e-01,  5.08625256e-02, -9.86361303e-01],
                    [-1.56538758e-01, -5.08625256e-02, -9.86361303e-01],
                    [-1.64594590e-01,  1.00785119e-17, -9.86361303e-01],
                    [ 0.00000000e+00,  1.64594590e-01, -9.86361303e-01],
                    [ 5.08625256e-02,  1.56538758e-01, -9.86361303e-01],
                    [ 1.64594590e-01, -3.02355357e-17, -9.86361303e-01],
                    [ 1.56538758e-01, -5.08625256e-02, -9.86361303e-01],
                    [ 9.67462728e-02, -1.33159821e-01, -9.86361303e-01]]),
        np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]),
        np.array([[-3.75323147e-18, -1.44568916e-17, -1.00000000e+00],
                [ 1.35276559e-01, -1.05651729e-01, -9.85158852e-01],
                [ 9.56045995e-02,  1.29770119e-01, -9.86924251e-01],
                [-2.18933157e-17, -1.65564594e-01, -9.86198948e-01],
                [ 4.68800966e-02, -1.56150312e-01, -9.86620158e-01],
                [ 1.29770119e-01,  9.56045995e-02, -9.86924251e-01],
                [-1.02285435e-01, -1.42301502e-01, -9.84524237e-01],
                [ 1.57752521e-01,  5.30909703e-02, -9.86050451e-01],
                [-1.05651729e-01,  1.35276559e-01, -9.85158852e-01],
                [-5.13183622e-02, -1.63254961e-01, -9.85248316e-01],
                [-5.67779109e-02,  1.64435583e-01, -9.84752359e-01],
                [-1.38024192e-01,  9.23617129e-02, -9.86112892e-01],
                [-1.42301502e-01, -1.02285435e-01, -9.84524237e-01],
                [-1.56150312e-01,  4.68800966e-02, -9.86620158e-01],
                [-1.63254961e-01, -5.13183622e-02, -9.85248316e-01],
                [-1.65564594e-01,  2.31096111e-17, -9.86198948e-01],
                [-1.12108333e-17,  1.71880069e-01, -9.85117882e-01],
                [ 5.30909703e-02,  1.57752521e-01, -9.86050451e-01],
                [ 1.71880069e-01, -9.54602452e-16, -9.85117882e-01],
                [ 1.64435583e-01, -5.67779109e-02, -9.84752359e-01],
                [ 9.23617129e-02, -1.38024192e-01, -9.86112892e-01]])
    )
])
def test__CubicFit(NodeCoords, neighborhood, normals):

    maxp, minp = curvature._CubicFit(NodeCoords, neighborhood, normals)
    if mymesh.check_numba():
        maxp, minp = curvature._CubicFit.py_func(NodeCoords, neighborhood, normals)


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