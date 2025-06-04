import pytest
import numpy as np
import mymesh
from mymesh import delaunay

@pytest.mark.parametrize("points", [
    (
        np.random.rand(10,2)
    ),
])
def test_Triangulate(points):

    methods = ['BowyerWatson', 'scipy', 'triangle']
    for method in methods:
        T = delaunay.Triangulate(points, method=method)

@pytest.mark.parametrize("points", [
    (
        np.random.rand(10,3)
    ),
])
def test_Tetrahedralize(points):

    methods = ['BowyerWatson', 'scipy']
    for method in methods:
        T = delaunay.Tetrahedralize(points, method=method)


@pytest.mark.parametrize("points", [
    (
        np.random.rand(10,2)
    ),
    (
        np.random.rand(10,3)
    ),
])
def test_ConvexHull(points):

    methods = ['BowyerWatson', 'scipy', 'GiftWrapping']
    for method in methods:
        if method == 'GiftWrapping' and np.shape(points)[1] == 3:
            continue
        H = delaunay.ConvexHull(points, method=method)