import pytest
import numpy as np
from mymesh import primitives, mesh

@pytest.mark.parametrize("M", [
    # Case 1: hexahedral grid mesh
    (primitives.Grid([0,1,0,1,0,1],.1)
    ),
    ])
def test_Properties(M):

    M.points = M.points
    M.cells = M.cells
    M.ND
    M.NNode
    M.NElem
    M.NEdge
    M.NFace
    M.Faces
    M.FaceConn
    M.FaceElemConn
    M.Edges
    M.EdgeConn
    M.EdgeElemConn
    M.SurfConn
    M.SurfNodes
    M.BoundaryConn
    M.BoundaryNodes
    M.NodeNeighbors
    M.ElemNeighbors
    M.ElemConn
    M.SurfNodeNeighbors
    M.SurfElemConn
    M.ElemNormals
    M.NodeNormalsMethod = M.NodeNormalsMethod
    M.NodeNormals
    M.Centroids
    M.EulerCharacteristic
    M.Genus
    M.Surface

