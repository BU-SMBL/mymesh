import pytest
import numpy as np
from mymesh import implicit, primitives, quality, tree

@pytest.mark.parametrize("points", [
    (
        primitives.Sphere([0,0,0], 1).NodeCoords
    ),
])
def test_Points2Octree(points):
    try:
        root = tree.Points2Octree(points)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')
    
@pytest.mark.parametrize("Voxel", [
    (
        primitives.Grid([0,1,0,1,0,1], .2)
    ),
])
def test_Voxel2Octree(Voxel):
    try:
        root = tree.Voxel2Octree(Voxel.NodeCoords, Voxel.NodeConn)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')
    
@pytest.mark.parametrize("Surface", [
    (
        primitives.Sphere([0,0,0], 1, ElemType='tri')
    ),
])
def test_Surface2Octree(Surface):
    try:
        root = tree.Surface2Octree(Surface.NodeCoords, Surface.NodeConn)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')
    
@pytest.mark.parametrize("Mesh", [
    (
        primitives.Sphere([0,0,0], 1, Type='vol')
    ),
])
def test_Mesh2Octree(Mesh):
    try:
        root = tree.Mesh2Octree(Mesh.NodeCoords, Mesh.NodeConn)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')

@pytest.mark.parametrize("func, bounds", [
    (
        implicit.sphere([0,0,0], 1),
        [-1,1,-1,1,-1,1]
    ),
    (
        implicit.gyroid,
        [0,1,0,1,0,1]
    ),
])
def test_Function2Octree(func, bounds):
    try:
        root = tree.Function2Octree(func, bounds)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')

@pytest.mark.parametrize("points, sparse", [
    (
        primitives.Sphere([0,0,0], 1).NodeCoords,
        True
    ),
    (
        primitives.Sphere([0,0,0], 1).NodeCoords,
        False
    ),
])
def test_Octree2Voxel(points, sparse):
    try:
        root = tree.Points2Octree(points)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')
    
    try:
        NodeCoords, NodeConn = tree.Octree2Voxel(root, sparse)
    except Exception as e:
        print(e)
        raise Exception('Failed to create voxel mesh')
    
    assert np.all(quality.Volume(NodeCoords, NodeConn) > 0), 'Inverted Elements.'
    
@pytest.mark.parametrize("points", [
    (
        primitives.Sphere([0,0,0], 1).NodeCoords
    ),
])
def test_Octree2Dual(points):
    try:
        root = tree.Points2Octree(points)
    except Exception as e:
        print(e)
        raise Exception('Failed to create octree')
    
    try:
        NodeCoords, NodeConn = tree.Octree2Dual(root)
    except Exception as e:
        print(e)
        raise Exception('Failed to create dual mesh')
    
    assert np.all(quality.Volume(NodeCoords, NodeConn) > 0), 'Inverted Elements.'
    
@pytest.mark.parametrize("points", [
    (
        primitives.Circle([0,0,0], 1, Type='line').NodeCoords
    ),
])
def test_Points2Quadtree(points):
    try:
        root = tree.Points2Quadtree(points)
    except Exception as e:
        print(e)
        raise Exception('Failed to create quadtree')

@pytest.mark.parametrize("boundary", [
    (
        primitives.Circle([0,0,0], 1, Type='line')
    ),
])
def test_Edges2Quadtree(boundary):
    try:
        root = tree.Edges2Quadtree(boundary.NodeCoords, boundary.NodeConn)
    except Exception as e:
        print(e)
        raise Exception('Failed to create quadtree')
    
@pytest.mark.parametrize("points", [
    (
        primitives.Circle([0,0,0], 1, Type='line').NodeCoords
    ),
])
def test_Quadtree2Pixel(points):
    try:
        root = tree.Points2Quadtree(points)
    except Exception as e:
        print(e)
        raise Exception('Failed to create quadtree')
    
    try:
        root = tree.Quadtree2Pixel(root)
    except Exception as e:
        print(e)
        raise Exception('Failed to create pixel mesh')
    
@pytest.mark.parametrize("points", [
    (
        primitives.Circle([0,0,0], 1, Type='line').NodeCoords
    ),
])
def test_Quadtree2Dual(points):
    try:
        root = tree.Points2Quadtree(points)
    except Exception as e:
        print(e)
        raise Exception('Failed to create quadtree')
    
    try:
        root = tree.Quadtree2Dual(root)
    except Exception as e:
        print(e)
        raise Exception('Failed to create dual mesh')

@pytest.mark.parametrize("root", [
    (
        tree.Surface2Octree(*primitives.Sphere([0,0,0],1, ElemType='tri'))
    ),
    (
        tree.Edges2Quadtree(*primitives.Circle([0,0,0],1, Type='line'))
    ),
])
def test_getAllLeaf(root):
    leaves = tree.getAllLeaf(root)
    for leaf in leaves:
        if leaf.state != 'leaf':
            raise Exception('Non-leaf included.')
        elif len(leaf.children) > 0:
            raise Exception('Node marked as leaf but has children.')
        
@pytest.mark.parametrize("root", [
    (
        tree.Surface2Octree(*primitives.Sphere([0,0,0],1, ElemType='tri'))
    ),
    (
        tree.Edges2Quadtree(*primitives.Circle([0,0,0],1, Type='line'))
    ),
])
def test_Print(root):
    tree.Print(root)