import pytest
import numpy as np
from mymesh import booleans, primitives

@pytest.mark.parametrize("Surf1, Surf2", [
    (
        primitives.Sphere([0,0,0], 1),
        primitives.Sphere([0.5,0,0], 1)
    ),
    (
        primitives.Sphere([0,0,0], 1),
        primitives.Torus([0,0,0], 1, 0.5)
    ),
])
def test_MeshBooleans(Surf1, Surf2):

    U, I, D = booleans.MeshBooleans(Surf1, Surf2)

    assert len(U.BoundaryNodes) == 0, 'Unclosed edges in the generated surface.'
    assert len(I.BoundaryNodes) == 0, 'Unclosed edges in the generated surface.'
    assert len(D.BoundaryNodes) == 0, 'Unclosed edges in the generated surface.'