# -*- coding: utf-8 -*-
# Created on Mon Jan 31 22:52:03 2022
# @author: toj
"""
Octree data structure and related methods.


Octree Creation
===============
.. autosummary::
    :toctree: submodules/

    Points2Octree
    Function2Octree
    Surface2Octree
    Voxel2Octree

Conversion From Octree
======================
.. autosummary::
    :toctree: submodules/

    Octree2Voxel
    Octree2Dual

Octree Querying
===============
.. autosummary::
    :toctree: submodules/

    getAllLeaf
    SearchOctree
    SearchOctreeTri

Octree Utilities
================
.. autosummary::
    :toctree: submodules/

    Print


"""
import numpy as np
import scipy
import sys, copy
from . import rays, utils
import sympy as sp

class OctreeNode:
          
    def __init__(self,centroid,size,parent=None,data=None,level=0,state='unknown'):
        """
        The OctreeNode is the basic unit of the octree data structure. The structure
        consists of a series of nodes that reference their parent and child nodes, 
        allowing for traversal of the tree structure.

        Parameters
        ----------
        centroid : array_like
            Location of the center of the octree node
        size : float
            Side length of the cube associated with the octree node
        parent : octree.OctreeNode, optional
            The octree node that contains this node, by default None
        data : list or dict, optional
            Data associated with the octree node. The type of data depends on 
            the how the octree was created, by default None.
        level : int, optional
            Depth within the tree structure, by default 0.
            The root node is at level 0, the root's children are at level 1, etc.
        state : str, optional
            Specifies whether the node's place in the tree structure, by default
            'unknown'.

            Possible states are:
            - 'root': This node is the root of the octree
            - 'branch': This is node is an intermediate node between the root and leaves
            - 'leaf': This is node is a terminal end and has no children.
            - 'empty': No data is contained within this node, and it has no children
            - 'unknown': State hasn't been specified.
        """  
        self.centroid = centroid
        self.size = size
        self.children = []
        self.parent = parent
        self.state = state
        self.data = data
        self.limits = None
        self.vertices = None
        self.level = level

    def __repr__(self):
        out = f'Octree Node ({self.state:s})\nCentroid: {str(self.centroid):s}\nSize: {self.size:f} \n'
        return out
    
    def getMaxDepth(self):
        """
        Get the maximum depth of the octree. The depth is the highest level
        reachable from the current node. The depth is given as the absolute level, 
        rather than relative to the current node, i.e., the max depth of an octree will
        be the same regardless of whether use search using the root node 
        or some other node 

        Returns
        -------
        depth : int
            Depth of the octree
        """        
        depth = self.level
        def recur(node, depth):
            if node.level > depth:
                depth = node.level
            for child in node.children:
                depth = recur(child, depth)
            return depth
        depth = recur(self, depth)
        return depth

    def getLevel(self, level):
        """
        Get all child nodes at a particular octree level

        Parameters
        ----------
        level : _type_
            _description_
        """        
        def recur(node,nodes):
            if node.level == level:
                nodes.append(node)
                return nodes
            elif node.state == 'empty':
                return nodes
            elif node.state == 'root' or node.state == 'branch':
                for child in node.children:
                    nodes = recur(child,nodes)
            return nodes

        nodes = []
        return recur(self,nodes)

    def getLimits(self):
        """
        Get the spatial bounds of the current octree node. Limits are formatted
        as [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. These are equivalent 
        to node.centroid +/- (node.size/2).

        Returns
        -------
        limits : list
            list of x, y and z bounds of the octree node
        """        
        if self.limits is None:
            self.limits = np.array([[self.centroid[d]-self.size/2,self.centroid[d]+self.size/2] for d in range(3)])
        return self.limits
    
    def getVertices(self):
        """
        Get the coordinates of the 8 vertices of the cube that correspond to the
        octree node. These are ordered following the hexahedral element node 
        numbering scheme, with the 4 minimum z vertices ordered counter clockwise
        followed by the 4 maximum z vertices.

        Returns
        -------
        vertices : np.ndarray
            Array of vertex coordinates
        """        
        if self.vertices is None:
            [x0,x1],[y0,y1],[z0,z1] = self.getLimits()
            self.vertices = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        return self.vertices

    def PointInNode(self,point,inclusive=True):
        """
        Check if a point is within the bounds of the current node.

        Parameters
        ----------
        point : np.ndarray
            Three element coordinate array
        inclusive : bool, optional
            Specify whether a point exactly on the boundary is include as in
            the node, by default True.

        Returns
        -------
        inside : bool
            True if the point is inside the node, otherwise False.
        """        
        inside = rays.PointInBox(point, *self.getLimits(), inclusive=inclusive)
        return inside
    
    def PointsInNode(self,points,inclusive=True):
        """
        Check if a set of points is within the bounds of the current node.

        Parameters
        ----------
        points : array_like
            nx3 coordinate array
        inclusive : bool, optional
            Specify whether a point exactly on the boundary is include as in
            the node, by default True.

        Returns
        -------
        inside : np.ndarray
            Array of bools for each point in points. True if the point is inside 
            the node, otherwise False.
        """
        if inclusive:
            return np.all([((self.centroid[d]-self.size/2) <= points[:,d]) & ((self.centroid[d]+self.size/2) >= points[:,d]) for d in range(3)], axis=0)
        else:
            return np.all([(self.centroid[d]-self.size/2) < points[:,d] and (self.centroid[d]+self.size/2) > points[:,d] for d in range(3)], axis=0)

    def TriInNode(self,tri,TriNormal,inclusive=True):
        
        lims = self.getLimits()
        return rays.TriangleBoxIntersection(tri, lims[0], lims[1], lims[2], BoxCenter=self.centroid,TriNormal=TriNormal)
        
        # return (any([self.PointInNode(pt,inclusive=inclusive) for pt in tri]) or rays.TriangleBoxIntersection(tri, lims[0], lims[1], lims[2]))
    
    def Contains(self,points):
        out = [idx for idx,point in enumerate(points) if self.PointInNode(point)]
        # out = np.where(self.PointsInNode(points))[0]
        return out
    
    def ContainsTris(self,tris,TriNormals):
        
        lims = self.getLimits()
        Intersections = np.where(rays.BoxTrianglesIntersection(tris, lims[0], lims[1], lims[2], TriNormals=TriNormals, BoxCenter=self.centroid))[0]
        return Intersections
    
    def ContainsBoxes(self, boxes):
        Intersections = np.where([rays.BoxBoxIntersection(self.getLimits(), box) for box in boxes])[0]
        return Intersections
    
    def isEmpty(self,points):
        return any([self.PointInNode(point) for point in points])
    
    def hasChildren(self):
        # includes both "leaf" and "empty" nodes
        return len(self.children) == 0
    
    def makeChildrenPts(self,points,minsize=0,maxsize=np.inf,maxdepth=np.inf):
        if self.size > minsize and self.level<maxdepth:
            self.makeChildren()
            
            for child in self.children:
                ptIds = child.Contains(points)
                ptsInChild = points[ptIds]#[points[idx] for idx in ptIds]
                if self.data:
                    child.data = [self.data[idx] for idx in ptIds]
                if len(ptsInChild) > 1: 
                    if child.size/2 <= minsize:
                        child.state = 'leaf'
                    else:
                        child.makeChildrenPts(ptsInChild,minsize=minsize,maxsize=maxsize)
                        child.state = 'branch'
                elif len(ptsInChild) == 1:
                    if child.size <= maxsize:
                        child.state = 'leaf'
                    else:
                        child.makeChildrenPts(ptsInChild,minsize=minsize,maxsize=maxsize)
                        child.state = 'branch'
                else:
                    child.state = 'empty'
                # self.children.append(child)  
        else:
            self.state = 'leaf'
            
    def makeChildrenTris(self,tris,TriNormals,minsize=0,maxsize=np.inf,maxdepth=np.inf):
        # tris is a list of Triangular vertices [tri1,tri2,...] where tri1 = [pt1,pt2,pt3]
        self.makeChildren()
                    
        for child in self.children:
            triIds = child.ContainsTris(tris,TriNormals)
            trisInChild = tris[triIds]# [tris[idx] for idx in triIds]
            normalsInChild = TriNormals[triIds]#[TriNormals[idx] for idx in triIds]
            if self.data is not None:
                child.data = [self.data[idx] for idx in triIds]
            if len(trisInChild) > 1: 
                if child.size/2 <= minsize or child.level >= maxdepth:
                    child.state = 'leaf'
                else:
                    child.makeChildrenTris(trisInChild,normalsInChild,minsize=minsize,maxsize=maxsize,maxdepth=maxdepth)
                    child.state = 'branch'
            elif len(trisInChild) == 1:
                if child.size > maxsize or child.level < maxdepth:
                    child.makeChildrenTris(trisInChild,normalsInChild,minsize=minsize,maxsize=maxsize,maxdepth=maxdepth)
                    child.state = 'branch'
                else:
                    child.state = 'leaf'
            elif len(trisInChild) == 0:
                child.state = 'empty'

    def makeChildrenBoxes(self,boxes,minsize=0,maxsize=np.inf,maxdepth=np.inf):
        
        self.makeChildren()
                    
        for child in self.children:
            boxIds = child.ContainsBoxes(boxes)
            boxesInChild = boxes[boxIds]
            if self.data is not None:
                child.data = [self.data[idx] for idx in boxIds]
            if len(boxesInChild) > 1: 
                if child.size/2 <= minsize or child.level >= maxdepth:
                    child.state = 'leaf'
                else:
                    child.makeChildrenBoxes(boxesInChild,minsize=minsize,maxsize=maxsize,maxdepth=maxdepth)
                    child.state = 'branch'
            elif len(boxesInChild) == 1:
                if child.size > maxsize or child.level < maxdepth:
                    child.makeChildrenBoxes(boxesInChild,minsize=minsize,maxsize=maxsize,maxdepth=maxdepth)
                    child.state = 'branch'
                else:
                    child.state = 'leaf'
            elif len(boxesInChild) == 0:
                child.state = 'empty'

    def makeChildren(self, childstate='unknown'):
        childSize = self.size/2
        self.children = []
        # Note other things (e.g. Function2Octree) depend on this ordering not changing 
        for xSign,ySign,zSign in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]:
            centroid = np.array([self.centroid[0]+xSign*self.size/4, self.centroid[1]+ySign*self.size/4, self.centroid[2]+zSign*self.size/4])
            self.children.append(OctreeNode(centroid,childSize,parent=self,data=[],level=self.level+1,state=childstate))

    def addTri(self,tri,triId=None,minsize=None):
            # triId can be an identifier for the element corresponding to the given triangle
            # If given, triId will be stored in the octree node data instead of the tri itself
            if not minsize:
                # By default creates octree with a minimum node size equal to the max edge length of a triangle
                minsize = max([max([pt[0] for pt in tri])-min([pt[0] for pt in tri]),
                            max([pt[1] for pt in tri])-min([pt[1] for pt in tri]),
                            max([pt[2] for pt in tri])-min([pt[2] for pt in tri]),
                            ])
            def recur(node,tri,triId,minsize):
                if node.TriInNode(tri):
                    if node.state == 'unknown' or node.state == 'empty':
                        node.state = 'branch'
                    if node.size/2 <= minsize:
                        node.state = 'leaf'
                        if triId:
                            node.data.append(triId)
                        else:
                            node.data.append(tri)
                    else:
                        if node.state == 'leaf':
                            node.state = 'branch'
                        if len(node.children) == 0:
                            node.makeChildren()
                        for child in node.children:
                            recur(child,tri,triId,minsize)
                elif node.state == 'unknown':
                    node.state = 'empty'
            recur(self,tri,triId,minsize)
            
    def clearData(self,clearChildren=True):
        self.data = []
        if clearChildren:
            for child in self.children:
                child.clearData()                 

def isInsideOctree(pt,node,inclusive=True):   
    if node.PointInNode(pt,inclusive=inclusive):
        if node.state == 'leaf':
            return True
        else:
            for child in node.children:
                if isInsideOctree(pt,child):
                    return True
            return False
    else:
        return False
            
def SearchOctree(pt,root):
    """
    Retrieve the octree leaf node that contains the given point.

    Parameters
    ----------
    pt : array_like
        3D coordinate ([x,y,z])
    root : octree.OctreeNode
        Root of the octree to be searched

    Returns
    -------
    node : octree.OctreeNode or NoneType
        Octree node containing the point. If the no node can be found to contain the point, None will be returned.
    """    
    if rays.PointInBox(pt, *root.getLimits(), inclusive=True): #root.PointInNode(pt,inclusive=True):
        if root.state == 'leaf' or len(root.children) == 0:
            return root
        else:
            for child in root.children:
                check = SearchOctree(pt,child)
                if check:
                    return check
            return None
                    
    else:
        return None
    
def SearchOctreeTri(tri,root,inclusive=True):
    """
    Retrieve the octree leaf node(s) that contain the triangle

    Parameters
    ----------
    tri : array_like
        3x3 list or np.ndarray containing the coordinates of the three vertices
        of a triangle.
    root : octree.OctreeNode
        Root node of the octree to be searched
    inclusive : bool, optional
        Specifies whether to include leaf nodes that the triangle is exactly
        on the boundary of, by default True.

    Returns
    -------
    nodes : list
        List of octree nodes.
    """    
    def recur(tri, node, nodes, inclusive):
        if node.TriInNode(tri,inclusive=inclusive):
            if root.state == 'leaf':
                nodes.append(node)
            else:
                for i,child in enumerate(node.children):
                    if child.state == 'empty':
                        continue
                    nodes = SearchOctreeTri(tri,child,nodes=nodes,inclusive=inclusive)
    nodes = recur(tri, root, [], inclusive)
    return nodes
    
def getAllLeaf(root):
    """
    Retrieve a list of all leaf nodes of the octree

    Parameters
    ----------
    root : octree.OctreeNode
        Root node of the octree of which the leaf nodes will be retrieved.

    Returns
    -------
    leaves : list
        List of octree leaf nodes.
    """    
    # Return a list of all terminal(leaf) nodes in the octree
    def recur(node,leaves):
        if node.state == 'leaf':
            leaves.append(node)
            return leaves
        elif node.state == 'empty':
            return leaves
        elif node.state == 'root' or node.state == 'branch':
            for child in node.children:
                leaves = recur(child,leaves)
        return leaves
    leaves = []
    return recur(root,leaves)

def Points2Octree(Points, maxdepth=10):
    """
    Generate an octree structure from a set of points. The octree will be 
    subdivided until each node contains only one point or the maximum depth
    is met. 

    Parameters
    ----------
    Points : array_like
        Point coordinates (nx3)
    maxdepth : int, optional
        Maximum depth of the octree, by default 10

    Returns
    -------
    root : octree.OctreeNode
        Root node of the generated octree structure.
    """    
    if type(Points) is list:
        Points = np.array(Points)
    minx = np.min(Points[:,0])
    maxx = np.max(Points[:,0])
    miny = np.min(Points[:,1])
    maxy = np.max(Points[:,1])
    minz = np.min(Points[:,2])
    maxz = np.max(Points[:,2])
    size = np.max([maxx-minx,maxy-miny,maxz-minz])
    
    centroid = np.array([minx + size/2, miny+size/2, minz+size/2])
    
    root = OctreeNode(centroid,size,data=[])
    root.state = 'root'
    root.makeChildrenPts(Points, maxdepth=maxdepth)    
    
    return root

def Voxel2Octree(VoxelCoords, VoxelConn):
    """
    Generate an octree representation of an isotropic voxel mesh. 

    Parameters
    ----------
    VoxelCoords : array_like
        Node coordinates of the voxel mesh
    VoxelConn : array_like
        Node connectivity of the voxel mesh

    Returns
    -------
    root : octree.OctreeNode
        Root node of the generated octree structure
    """    
    if type(VoxelCoords) is list:
        VoxelCoords = np.array(VoxelCoords)
    # Assumes (and requires) that all voxels are cubic and the same size
    VoxelSize = abs(sum(VoxelCoords[VoxelConn[0][0]] - VoxelCoords[VoxelConn[0][1]]))
    centroids = [np.mean(VoxelCoords[elem],axis=0) for elem in VoxelConn]
    minx = min(VoxelCoords[:,0])
    maxx = max(VoxelCoords[:,0])
    miny = min(VoxelCoords[:,1])
    maxy = max(VoxelCoords[:,1])
    minz = min(VoxelCoords[:,2])
    maxz = max(VoxelCoords[:,2])
    minsize = max([maxx-minx,maxy-miny,maxz-minz])
    size = VoxelSize
    while size < minsize:
        size *= 2
    
    centroid = np.array([minx + size/2, miny+size/2, minz+size/2])
    
    Root = OctreeNode(centroid,size,data=[])
    Root.state = 'root'
    Root.makeChildrenPts(centroids, maxsize=VoxelSize)    
    
    return Root

def Surface2Octree(NodeCoords, SurfConn, minsize=None, maxdepth=5):
    """
    Generate an octree representation of a triangular surface mesh. The octree
    will be refined until each node contains only one triangle or the maximum
    depth or minimum size criteria are met. Each node contains a list of 
    element ids corresponding to the elements that are contained within that 
    node in the OctreeNode.data field.

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates of the surface mesh
    SurfConn : array_like
        Node connectivity of the triangular surface mesh. This must be an nx3
        array or list.
    minsize : float, optional
        Minimum size for an octree node, by default None.
        If supplied, octree nodes will not be divided to be smaller than this
        size.
    maxdepth : int, optional
        Maximum depth of the octree, by default 5

    Returns
    -------
    root : octree.OctreeNode
        Root node of the generate octree
    """    
    if type(NodeCoords) is list:
        NodeCoords = np.array(NodeCoords)          
    
    ArrayConn = np.asarray(SurfConn).astype(int)
    if minsize is None and maxdepth is None:
        # By default creates octree with a minimum node size equal to the mean size of a triangle
        minsize = np.nanmean(np.nanmax([np.linalg.norm(NodeCoords[ArrayConn][:,0] - NodeCoords[ArrayConn][:,1],axis=1),
            np.linalg.norm(NodeCoords[ArrayConn][:,1] - NodeCoords[ArrayConn][:,2],axis=1),
            np.linalg.norm(NodeCoords[ArrayConn][:,2] - NodeCoords[ArrayConn][:,0],axis=1)],axis=0
            ))
    elif minsize is None and maxdepth is not None:
        minsize = 0
        
    minx = min(NodeCoords[:,0])
    maxx = max(NodeCoords[:,0])
    miny = min(NodeCoords[:,1])
    maxy = max(NodeCoords[:,1])
    minz = min(NodeCoords[:,2])
    maxz = max(NodeCoords[:,2])
    
    size = max([maxx-minx,maxy-miny,maxz-minz])
    centroid = np.array([minx + size/2, miny+size/2, minz+size/2])
    ElemIds = list(range(len(SurfConn)))
    root = OctreeNode(centroid,size,data=ElemIds)
    root.state = 'root'

    TriNormals = np.array(utils.CalcFaceNormal(NodeCoords,SurfConn))
    root.makeChildrenTris(NodeCoords[ArrayConn], TriNormals, maxsize=size, minsize=minsize,  maxdepth=maxdepth)

    return root

def Mesh2Octree(NodeCoords, NodeConn, minsize=None, mindepth=2, maxdepth=5):
    
    NodeCoords = np.asarray(NodeCoords)
    # Bounds of each element (minx, maxx, miny, maxy, minz, maxz)
    elembounds = np.array([[[NodeCoords[:,0][elem].min(), NodeCoords[:,0][elem].max()], [NodeCoords[:,1][elem].min(), NodeCoords[:,1][elem].max()], [NodeCoords[:,2][elem].min(), NodeCoords[:,2][elem].max()]] for elem in NodeConn])
    # Bounds for the full mesh
    bounds = np.array([np.min(elembounds[:,0,0]), np.max(elembounds[:,0,1]), np.min(elembounds[:,1,0]), np.max(elembounds[:,1,1]), np.min(elembounds[:,2,0]), np.max(elembounds[:,2,1])])

    size = max([bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4]])
    centroid = np.array([bounds[0] + size/2, bounds[2]+size/2, bounds[4]+size/2])

    if minsize is None:
        minsize = 0

    ElemIds = np.arange(len(NodeConn))
    root = OctreeNode(centroid, size, data=ElemIds)
    root.state = 'root'
    root.makeChildrenBoxes(elembounds, maxsize=size, minsize=minsize,  maxdepth=maxdepth)

    return root

def Function2Octree(func, bounds, threshold=0, grad=None, mindepth=2, maxdepth=5, strategy='EDerror', eps=0.1):
    """
    Generate an octree structure adapted to an implicit function.
    Based on octree generation approaches used by :cite:`Schaefer2005`, 
    :cite:`Zhang2003`. 

    Parameters
    ----------
    func : function
        Implicit function that describes the geometry of the object to be meshed. 
        The function should be of the form v = f(x,y,z,*args,**kwargs) where 
        x, y, and z are numpy arrays of x, y and z coordinates and v is a numpy 
        array of function values. 
    bounds : array_like
        6 element array, list, or tuple with the minimum and maximum bounds in 
        each direction that the function will be evaluated. This should be 
        formatted as: [xmin, xmax, ymin, ymax, zmin, zmax]
    threshold : int, optional
        Isosurface level, by default 0
    grad : _type_, optional
        _description_, by default None
    mindepth : int, optional
        Minimum octree depth, by default 2. This correspond to a maximum octree
        node size of L/(2^(mindepth)), where L is the maximum span between the
        x, y, or z bounds.
    maxdepth : int, optional
        Maximum octree depth, by default 5. This correspond to a minimum octree
        node size of L/(2^(maxdepth)), where L is the maximum span between the
        x, y, or z bounds.
    strategy : str, optional
        Strategy to guide subdivision, by default 'EDerror'.
        
        - 'EDerror': Uses the Euclidian distance error function proposed by 
            :cite:`Zhang2003` to assess the error between linear interpolation within
            an octree node and with the evaluation of the function at vertices at 
            the next level of refinement. If the error is less than the threshold
            specified by `eps` or if there are no sign changes detected, subdivision
            is halted.
        - 'QEF': Uses the quadratic error function proposed by 
            :cite:`Schaefer2005`. This approach is not fully implemented yet.
    eps : float, optional
        Error threshold value used to determine whether further subdivision is
        necessary, by default 0.01

    Returns
    -------
    root : octree.OctreeNode
        The root node of the generated octree.

    """    
    
    # Function value and gradient evaluated at the vertices is stored as `data` in each node
    # func and grad should both accept 3 arguments (x,y,z), and handle both vectorized and scalar inputs

    size = max([bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4]])
    centroid = np.array([bounds[0] + size/2, bounds[2]+size/2, bounds[4]+size/2])

    if grad is None:
        x, y, z = sp.symbols('x y z', real=True)
        if callable(func):
            if isinstance(func(centroid[0], centroid[1], centroid[2]), sp.Basic):
                def DiracDelta(x):
                    if type(x) is np.ndarray:
                        return (x == 0).astype(float)
                    else:
                        return float(x==0)
                F = sp.lambdify((x, y, z), func(x,y,z), 'numpy')
                
                Fx = sp.diff(func(x, y, z),x)
                Fy = sp.diff(func(x, y, z),y)
                Fz = sp.diff(func(x, y, z),z)
                Grad = sp.Matrix([Fx, Fy, Fz]).T
                grad = sp.lambdify((x,y,z),Grad,['numpy',{'DiracDelta':DiracDelta}])

            else:
                F = func
                finite_diff_step = 1e-5
                def grad(X,Y,Z):
                    gradx = (F(X+finite_diff_step/2,Y,Z) - F(X-finite_diff_step/2,Y,Z))/finite_diff_step
                    grady = (F(X,Y+finite_diff_step/2,Z) - F(X,Y-finite_diff_step/2,Z))/finite_diff_step
                    gradz = (F(X,Y,Z+finite_diff_step/2) - F(X,Y,Z-finite_diff_step/2))/finite_diff_step
                    gradf = np.vstack((gradx,grady,gradz))
                    return gradf
        elif isinstance(func, sp.Basic):
            F = sp.lambdify((x, y, z), func, 'numpy')

            Fx = sp.diff(func,x)
            Fy = sp.diff(func,y)
            Fz = sp.diff(func,z)
            Grad = sp.Matrix([Fx, Fy, Fz]).T
            grad = sp.lambdify((x,y,z),Grad,['numpy',{'DiracDelta':DiracDelta}])
        else:
            raise TypeError('func must be a sympy function or callable function of three arguments (x,y,z).')

    root = OctreeNode(centroid, size)
    root.state = 'root'

    if strategy == 'QEF':
        for level in range(maxdepth):

            nodes = root.getLevel(level)
            nodes = [node for node in nodes if node.state != 'leaf']
            if level < mindepth:
                for node in nodes:
                    node.makeChildren(childstate='branch')
                continue

            if level == mindepth:
                for node in nodes:
                    node.makeChildren(childstate='branch')
            
            # 9 Evaluation points - corners and center
            vertices = np.array([np.append(node.getVertices(), np.atleast_2d(node.centroid), axis=0) for node in nodes])
            xi = vertices[:,:,0]
            yi = vertices[:,:,1]
            zi = vertices[:,:,2]
            f = func(xi.flatten(), yi.flatten(), zi.flatten()).reshape((vertices.shape[0],vertices.shape[1]))
            g = grad(xi.flatten(), yi.flatten(), zi.flatten())
            gx = g[0].reshape((vertices.shape[0],vertices.shape[1]))
            gy = g[1].reshape((vertices.shape[0],vertices.shape[1]))
            gz = g[2].reshape((vertices.shape[0],vertices.shape[1]))

            # QEF denominator
            vi = 1/(1 + np.linalg.norm(np.stack([gx, gy, gz]), axis=0))

            # Construct A matrix of quadratic terms, summing over sample points for each octree node
            A = np.empty((vertices.shape[0],4,4))
            
            A[:,0,0] = np.sum(vi,axis=1)
            A[:,0,1] = np.sum(-vi*gx, axis=1)
            A[:,0,2] = np.sum(-vi*gy, axis=1)
            A[:,0,3] = np.sum(-vi*gz, axis=1)
            A[:,1,0] = np.sum(-vi*gx, axis=1)
            A[:,1,1] = np.sum(vi*gx**2, axis=1)
            A[:,1,2] = np.sum(vi*gx*gy, axis=1)
            A[:,1,3] = np.sum(vi*gx*gz, axis=1)
            A[:,2,0] = np.sum(-vi*gy, axis=1)
            A[:,2,1] = np.sum(vi*gy*gx, axis=1)
            A[:,2,2] = np.sum(vi*gy**2, axis=1)
            A[:,2,3] = np.sum(vi*gy*gz, axis=1)
            A[:,3,0] = np.sum(-vi*gz, axis=1)
            A[:,3,1] = np.sum(vi*gz*gx, axis=1)
            A[:,3,2] = np.sum(vi*gz*gy, axis=1)
            A[:,3,3] = np.sum(vi*gz**2, axis=1)

            # Construct b vector of linear terms, summing over sample points for each octree node
            b = np.empty((vertices.shape[0],4,1))
            b[:,0,0] = np.sum(vi*(gx*xi + gy*yi + gz*zi), axis=1)
            b[:,1,0] = np.sum(vi*(-gx*gx*xi - gx*gy*yi - gx*gy*zi), axis=1)
            b[:,2,0] = np.sum(vi*(-gy*gx*xi - gy*gy*yi - gy*gz*zi), axis=1)
            b[:,3,0] = np.sum(vi*(-gz*gx*xi - gz*gy*yi - gz*gz*zi), axis=1)

            # Construct c coefficient of constant terms, summing over sample points for each octree node
            c = np.sum(vi*(gx**2*xi**2 + gy**2*yi**2 + gz**2*zi**2 + 2*gx*gy*xi*yi + 2*gx*gz*xi*zi + 2*gy*gz*yi*zi), axis=1)

            # Find the minimizer of the quadratic error function E(x) = x^T A x + 2b^T x + c
            # Minimum where gradient equals zero: grad(E) = 2Ax + 2b = 0 -> Ax = -b
            # Solving with SVD for robustness in case of singular matrices
            U, S, Vt = np.linalg.svd(A)
            Sinv = np.zeros(S.shape)
            tol = 1e-6*S[:,0,None]
            Sinv[S > tol] = 1/S[S > tol]
            X = (Vt.swapaxes(1,2) @ (Sinv[:, :, None] * (U.swapaxes(1,2) @ (-b))))

            # X =(wi, xi, yi, zi) vector for cell centers 
            Xhat = np.column_stack([f[:,8], vertices[:,8,:]])[:,:,None]
            # Robust SVD approach from Lindstrom (2000)
            X2 = Xhat + (Vt.swapaxes(1,2) @ Sinv[:, :, None]) * (U.swapaxes(1,2) @ (b - A @ Xhat))
            
            E =  (X.swapaxes(1,2) @ A @ X)[:,0,0]  + (2*b.swapaxes(1,2) @ X)[:,0,0] + c

    elif strategy == 'EDerror':
        # Each of the eight children nodes have eight vertices (64 total), but 
        # many of these vertices are shared, so there are only 27 unique vertices.
        # 8 of these values (the corners of the parent cube) are identical between
        # the interpolated and evaluated values and can be excluded, leaving 19 values
        # These indices extract the unique values:
        NodeIdx = [0,1,2,3,4,5,6,7,0,1,2,3,0,0,1,2,3,4,0]
        VertIdx = [1,2,3,0,5,6,7,4,4,5,6,7,2,5,6,7,4,6,6]
        
        for level in range(maxdepth):

            nodes = root.getLevel(level)
            nodes = [node for node in nodes if node.state != 'leaf']
            if len(nodes) == 0:
                break
            if level < mindepth:
                for node in nodes:
                    node.makeChildren(childstate='branch')
                    
                continue

            if level == mindepth:
                for node in nodes:
                    node.makeChildren(childstate='branch')
            
            vertices = np.array([node.getVertices() for node in nodes])
            f = F(vertices[:,:,0].flatten(), vertices[:,:,1].flatten(), vertices[:,:,2].flatten()).reshape((vertices.shape[0],vertices.shape[1])) - threshold

            # else:
            #     # Copy calculations from previous iteration
            #     vertices = vertices_plus
            #     f = f_plus

            # Vertices for the next level 
            nodes_plus = [node.children for node in nodes] #root.getLevel(level+1)
            # vertices_plus is a 4th order tensor
            # first index corresponds to parent nodes, second index is the child, third index is each vertex, fourth index is coordinate (x,y,z)
            vertices_plus = np.array([[node.getVertices() for node in n] for n in nodes_plus])
            
            x = vertices_plus[:,NodeIdx,VertIdx,0]
            y = vertices_plus[:,NodeIdx,VertIdx,1]
            z = vertices_plus[:,NodeIdx,VertIdx,2]

            # Function values for the next level 
            f_plus = F(x, y, z) - threshold

            # Vertex coordinates normalized to unit cube.
            # Every octree node has the same set of normalized coordinates, these 
            # are order to be consistent with vertex numbering used for function
            # evaluations
            # X = (x - np.min(x,axis=(1,2))[:,None,None])/nodes[0].size
            # Y = (y - np.min(y,axis=(1,2))[:,None,None])/nodes[0].size
            # Z = (z - np.min(z,axis=(1,2))[:,None,None])/nodes[0].size
            X = np.array([0.5, 1. , 0.5, 0. , 0.5, 1. , 0.5, 0. , 0. , 1. , 1. , 0. , 0.5, 0.5, 1. , 0.5, 0. , 0.5, 0.5])[None,:]
            Y = np.array([0. , 0.5, 1. , 0.5, 0. , 0.5, 1. , 0.5, 0. , 0. , 1. , 1. , 0.5, 0. , 0.5, 1. , 0.5, 0.5, 0.5])[None,:]
            Z = np.array([0. , 0. , 0. , 0. , 1. , 1. , 1. , 1. , 0.5, 0.5, 0.5, 0.5, 0. , 0.5, 0.5, 0.5, 0.5, 1. , 0.5])[None,:]

            # Interpolate function values:
            f000 = f[:,0][:,None]
            f100 = f[:,1][:,None]
            f110 = f[:,2][:,None]
            f010 = f[:,3][:,None]
            f001 = f[:,4][:,None]
            f101 = f[:,5][:,None]
            f111 = f[:,6][:,None]
            f011 = f[:,7][:,None]

            f_interp = f000*(1-X)*(1-Y)*(1-Z) + f011*(1-X)*Y*Z + \
                    f001*(1-X)*(1-Y)*Z     + f101*X*(1-Y)*Z + \
                    f010*(1-X)*Y*(1-Z)     + f110*X*Y*(1-Z) + \
                    f100*X*(1-Y)*(1-Z)     + f111*X*Y*Z

            # TODO: grad currently set up to return 3xn. There will be problems if the shape is different
            gradnorm = np.linalg.norm(grad(x.flatten(), y.flatten(), z.flatten()).reshape(3, np.size(x)),axis=0).reshape(x.shape)

            # dfdx_interp = -f000*(1-Y)*(1-Z) - f011*Y*Z + \
            #             -f001*(1-Y)*Z     + f101*(1-Y)*Z + \
            #             -f010*Y*(1-Z)     + f110*Y*(1-Z) + \
            #             f100*(1-Y)*(1-Z) + f111*Y*Z

            # dfdy_interp = -f000*(1-X)*(1-Z) + f011*(1-X)*Z + \
            #             -f001*(1-X)*Z     - f101*X*Z + \
            #             f010*(1-X)*(1-Z) + f110*X*(1-Z) + \
            #             -f100*X*(1-Z)     + f111*X*Z

            # dfdz_interp = -f000*(1-X)*(1-Y) + f011*(1-X)*Y + \
            #             f001*(1-X)*(1-Y) + f101*X*(1-Y) + \
            #             -f010*(1-X)*Y     - f110*X*Y + \
            #             -f100*X*(1-Y)     + f111*X*Y

            # grad_interp = np.sqrt(dfdx_interp**2 + dfdy_interp**2 + dfdz_interp**2)

            EDerror = np.nansum(np.abs(f_plus - f_interp)/gradnorm, axis=1)


            if level == maxdepth - 1:
                childstate = 'leaf'
            else:
                childstate = 'branch'
            for i,row in enumerate(nodes_plus):
                if EDerror[i] > eps and not (np.all(f_plus[i] > 0) or np.all(f_plus[i] < 0)):
                    for n in row:
                        n.makeChildren(childstate=childstate)
                else:
                    nodes[i].state = 'leaf'


    return root

def Octree2Voxel(root, mode='sparse'):
    """
    Convert an octree to a voxel mesh

    Parameters
    ----------
    root : octree.OctreeNode
        Octree node from which the mesh will be generated. 
    mode : str, optional
        Determines voxelization mode. If "sparse", only leaf nodes that contain
        data will be included, otherwise both leaf and empty nodes
        will be include, by default 'sparse'. 

    Returns
    -------
    VoxelCoords : np.ndarray
        Node coordinates of the voxel mesh.
    VoxelConn : np.ndarray
        Node connectivity of the hexahedral voxel mesh.

    """    
    VoxelConn = []
    VoxelCoords = []
    if mode == 'sparse':
        condition = lambda node : node.state == 'leaf'
    elif mode == 'full':
        condition = lambda node : node.state == 'leaf' or node.state == 'empty' or len(node.children) == 0
    else:
        raise ValueError(f'mode must be "sparse" or "full", not {str(mode):s}')

    def recurSearch(node):
        if condition(node):
            VoxelConn.append([len(VoxelCoords)+0, len(VoxelCoords)+1, len(VoxelCoords)+2, len(VoxelCoords)+3,
                            len(VoxelCoords)+4, len(VoxelCoords)+5, len(VoxelCoords)+6, len(VoxelCoords)+7])
            VoxelCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] - node.size/2, node.centroid[2] - node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] - node.size/2, node.centroid[2] - node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] + node.size/2, node.centroid[2] - node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] + node.size/2, node.centroid[2] - node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] - node.size/2, node.centroid[2] + node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] - node.size/2, node.centroid[2] + node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] + node.size/2, node.centroid[2] + node.size/2]
                )
            VoxelCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] + node.size/2, node.centroid[2] + node.size/2]
                )
        elif node.state == 'branch' or node.state == 'root' or node.state == 'unknown':
            for child in node.children:
                recurSearch(child)
    
    recurSearch(root)
    VoxelCoords = np.asarray(VoxelCoords)
    VoxelConn = np.asarray(VoxelConn)
    return VoxelCoords, VoxelConn

def Octree2Dual(root, method='centroid'):
    """
    Converts an octree to a mesh that is dual to the octree structure. This mesh
    contains hexahedral elements with nodes contained inside octree nodes, rather
    than at the octree vertices. At transitions between octree node levels,
    some hexahedra may be partially degenerate (i.e. form pyramids rather than
    hexahedra). Based on the algorithm proposed by :cite:`Schaefer2005` and
    explained by :cite:`Holmlid2010`. This `website <https://www.volume-gfx.com/volume-rendering/dual-marching-cubes/deriving-the-dualgrid/>`_ is another useful reference.

    Parameters
    ----------
    root : octree.OctreeNode
        Root node of the octree
    method : str, optional
        Method used for placing the dual vertices within the octree nodes, by 
        default 'centroid'.
        
        Currently the only implemented option is to place the vertices at 
        the centroids of the octree nodes.

    Returns
    -------
    DualCoords : np.ndarray
        Array of nodal coordinates.
    DualConn : np.ndarray
        List of node connectivities for the hexahedral mesh.
    """    
    def nodeProc(node, DualCoords, DualConn):
        if not node.hasChildren():
            for child in node.children:
                nodeProc(child, DualCoords, DualConn)

            for idx in [(0,4), (1,5), (2,6), (3,7)]:
                faceProcXY(node.children[idx[0]],node.children[idx[1]], DualCoords, DualConn)
            for idx in [(0,1), (3,2), (4,5), (7,6)]:
                faceProcYZ(node.children[idx[0]],node.children[idx[1]], DualCoords, DualConn)
            for idx in [(0,3), (1,2), (4,7), (5,6)]:
                faceProcXZ(node.children[idx[0]],node.children[idx[1]], DualCoords, DualConn)
            
            for idx in [(0,3,7,4), (1,2,6,5)]:
                edgeProcX(node.children[idx[0]],node.children[idx[1]],node.children[idx[2]],node.children[idx[3]], DualCoords, DualConn)
            for idx in [(0,1,5,4), (3,2,6,7)]:
                edgeProcY(node.children[idx[0]],node.children[idx[1]],node.children[idx[2]],node.children[idx[3]], DualCoords, DualConn)
            for idx in [(0,1,2,3), (4,5,6,7)]:
                edgeProcZ(node.children[idx[0]],node.children[idx[1]],node.children[idx[2]],node.children[idx[3]], DualCoords, DualConn)

            vertProc(*node.children, DualCoords, DualConn)
 
    def faceProcXY(n0, n1, DualCoords, DualConn):
        # Nodes should be ordered bottom-top (n0 is below n1)
        if not (n0.hasChildren() and n1.hasChildren()):    
            # c0, c1, c2, c3 are the *top* nodes of n0 and c4, c5, c6, c7 are the *bottom* nodes of n1
            c0 = n0 if n0.hasChildren() else n0.children[4]
            c1 = n0 if n0.hasChildren() else n0.children[5]
            c2 = n0 if n0.hasChildren() else n0.children[6]
            c3 = n0 if n0.hasChildren() else n0.children[7]
        
            c4 = n1 if n1.hasChildren() else n1.children[0]
            c5 = n1 if n1.hasChildren() else n1.children[1]
            c6 = n1 if n1.hasChildren() else n1.children[2]
            c7 = n1 if n1.hasChildren() else n1.children[3]

            faceProcXY(c0,c4, DualCoords, DualConn)
            faceProcXY(c1,c5, DualCoords, DualConn)
            faceProcXY(c2,c6, DualCoords, DualConn)
            faceProcXY(c3,c7, DualCoords, DualConn)

            edgeProcX(c0,c3,c7,c4, DualCoords, DualConn)
            edgeProcX(c1,c2,c6,c5, DualCoords, DualConn)

            edgeProcY(c0,c1,c5,c4, DualCoords, DualConn)
            edgeProcY(c3,c2,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def faceProcYZ(n0, n1, DualCoords, DualConn):
        # Nodes should be ordered left-right (n0 is left of n1)
        if not (n0.hasChildren() and n1.hasChildren()):    
            # c0, c3, c7, c4 are the *right* nodes of n0 and c1, c2, c6, c5 are the *left* nodes of n1
            # The 2x2 of adjacent children is thus [c0,c1,c2,c3,c4,c5,c6,c7,c8]
            c0 = n0 if n0.hasChildren() else n0.children[1]
            c3 = n0 if n0.hasChildren() else n0.children[2]
            c7 = n0 if n0.hasChildren() else n0.children[6]
            c4 = n0 if n0.hasChildren() else n0.children[5]
        
            c1 = n1 if n1.hasChildren() else n1.children[0]
            c2 = n1 if n1.hasChildren() else n1.children[3]
            c6 = n1 if n1.hasChildren() else n1.children[7]
            c5 = n1 if n1.hasChildren() else n1.children[4]

            faceProcYZ(c0,c1, DualCoords, DualConn)
            faceProcYZ(c3,c2, DualCoords, DualConn)
            faceProcYZ(c7,c6, DualCoords, DualConn)
            faceProcYZ(c4,c5, DualCoords, DualConn)

            edgeProcY(c0,c1,c5,c4, DualCoords, DualConn)
            edgeProcY(c3,c2,c6,c7, DualCoords, DualConn)

            edgeProcZ(c0,c1,c2,c3, DualCoords, DualConn)
            edgeProcZ(c4,c5,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def faceProcXZ(n0, n1, DualCoords, DualConn):
        # Nodes should be ordered front-back (n0 is in front of n1)
        if not (n0.hasChildren() and n1.hasChildren()):    
            # c0, c1, c5, c4 are the *back* nodes of n0 and c3, c2, c6, c7 are the *front* nodes of n1
            # The 2x2 of adjacent children is thus [c0,c1,c2,c3,c4,c5,c6,c7,c8]
            c0 = n0 if n0.hasChildren() else n0.children[3]
            c1 = n0 if n0.hasChildren() else n0.children[2]
            c5 = n0 if n0.hasChildren() else n0.children[6]
            c4 = n0 if n0.hasChildren() else n0.children[7]
            c3 = n1 if n1.hasChildren() else n1.children[0]
            c2 = n1 if n1.hasChildren() else n1.children[1]
            c6 = n1 if n1.hasChildren() else n1.children[5]
            c7 = n1 if n1.hasChildren() else n1.children[4]

            faceProcXZ(c0,c3, DualCoords, DualConn)
            faceProcXZ(c1,c2, DualCoords, DualConn)
            faceProcXZ(c5,c6, DualCoords, DualConn)
            faceProcXZ(c4,c7, DualCoords, DualConn)

            edgeProcX(c0,c3,c7,c4, DualCoords, DualConn)
            edgeProcX(c1,c2,c6,c5, DualCoords, DualConn)

            edgeProcZ(c0,c1,c2,c3, DualCoords, DualConn)
            edgeProcZ(c4,c5,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def edgeProcX(n0,n1,n2,n3, DualCoords, DualConn):
        if not all([n0.hasChildren(), n1.hasChildren(), n2.hasChildren(), n3.hasChildren()]):
            c1 = n0 if n0.hasChildren() else n0.children[6]
            c0 = n0 if n0.hasChildren() else n0.children[7]
            c3 = n1 if n1.hasChildren() else n1.children[4]
            c2 = n1 if n1.hasChildren() else n1.children[5]
            c7 = n2 if n2.hasChildren() else n2.children[0]
            c6 = n2 if n2.hasChildren() else n2.children[1]
            c5 = n3 if n3.hasChildren() else n3.children[2]
            c4 = n3 if n3.hasChildren() else n3.children[3]

            edgeProcX(c1,c2,c6,c5, DualCoords, DualConn)
            edgeProcX(c0,c3,c7,c4, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def edgeProcY(n0,n1,n2,n3, DualCoords, DualConn):
        # Nodes should be ordered counter clockwise about the axis
        if not all([n0.hasChildren(), n1.hasChildren(), n2.hasChildren(), n3.hasChildren()]):
            c0 = n0 if n0.hasChildren() else n0.children[5]
            c3 = n0 if n0.hasChildren() else n0.children[6]
            c1 = n1 if n1.hasChildren() else n1.children[4]
            c2 = n1 if n1.hasChildren() else n1.children[7]
            c5 = n2 if n2.hasChildren() else n2.children[0]
            c6 = n2 if n2.hasChildren() else n2.children[3]
            c4 = n3 if n3.hasChildren() else n3.children[1]
            c7 = n3 if n3.hasChildren() else n3.children[2]

            edgeProcY(c0,c1,c5,c4, DualCoords, DualConn)
            edgeProcY(c3,c2,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def edgeProcZ(n0,n1,n2,n3, DualCoords, DualConn):
        # Nodes should be ordered counter clockwise about the axis
        if not all([n0.hasChildren(), n1.hasChildren(), n2.hasChildren(), n3.hasChildren()]):
            c0 = n0 if n0.hasChildren() else n0.children[2]
            c4 = n0 if n0.hasChildren() else n0.children[6]
            c1 = n1 if n1.hasChildren() else n1.children[3]
            c5 = n1 if n1.hasChildren() else n1.children[7]
            c2 = n2 if n2.hasChildren() else n2.children[0]
            c6 = n2 if n2.hasChildren() else n2.children[4]
            c3 = n3 if n3.hasChildren() else n3.children[1]
            c7 = n3 if n3.hasChildren() else n3.children[5]

            edgeProcZ(c0,c1,c2,c3, DualCoords, DualConn)
            edgeProcZ(c4,c5,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def vertProc(n0, n1, n2, n3, n4, n5, n6, n7, DualCoords, DualConn):
        ns = [n0, n1, n2, n3, n4, n5, n6, n7]
        
        if not all([n.hasChildren() for n in ns]):
            c0 = n0 if n0.hasChildren() else n0.children[6]
            c1 = n1 if n1.hasChildren() else n1.children[7]
            c2 = n2 if n2.hasChildren() else n2.children[4]
            c3 = n3 if n3.hasChildren() else n3.children[5]
            c4 = n4 if n4.hasChildren() else n4.children[2]
            c5 = n5 if n5.hasChildren() else n5.children[3]
            c6 = n6 if n6.hasChildren() else n6.children[0]
            c7 = n7 if n7.hasChildren() else n7.children[1]

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7,DualCoords,DualConn)
        else:
            # create a dual grid element
            if method=='centroid':
                coord = [n.centroid for n in ns]
            elif method=='qef_min':
                coord = [n.data['xopt'] for n in ns]
            DualConn.append(list(range(len(DualCoords),len(DualCoords)+8)))
            DualCoords += coord
            if len(DualConn) == 24:
                a = 2
    
    DualConn = []
    DualCoords = []     
    nodeProc(root, DualCoords, DualConn)
    DualCoords = np.asarray(DualCoords)
    DualConn = np.asarray(DualConn)
    return DualCoords, DualConn

def Print(root, show_empty=False):
    """
    Prints a formatted list of all nodes in the octree.

    Parameters
    ----------
    root : octree.OctreeNode
        Root node of the octree
    show_empty : bool, optional
        Option to include 'empty' nodes in the printed octree, by default False.
    """    
    def recur(node):
        if show_empty or node.state != 'empty':
            print('    '*node.level + str(node.level) +'. '+ node.state)
        for child in node.children:
            recur(child)
    
    recur(root)
    