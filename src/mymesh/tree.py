# -*- coding: utf-8 -*-
# Created on Mon Jan 31 22:52:03 2022
# @author: toj
"""
Tree data structures and related methods. Tree structures are used to organize
space into a hierarchy that makes it more efficient to search through them. This
module currently includes an Octree data structure and its two dimensional 
analog the Quadtree. 

These trees are based on uniform subdivision of an initial cube/square into 
eight/four smaller cubes/squares. The main principal is that if you know an 
entity (point, triangle, etc.) isn't in a particular node of the tree, you know 
it's not in any of the branches of that node and they don't need to be searched. 
This can lead to significant efficiency improvements in a variety of operations, 
for example if searching for intersections between a ray and a surface mesh 
(:func:`mymesh.rays.RaySurfIntersection`), by testing for intersections between 
a ray and the cube represented by an octree node, you can eliminate the need to 
test for intersections between the ray and any of the triangles within that node. 

Tree
====
.. autosummary::
    :toctree: submodules/

    TreeNode

Tree Utilities
----------------
.. autosummary::
    :toctree: submodules/

    Print
    getAllLeaf

Quadtree
========
.. autosummary::
    :toctree: submodules/

    QuadtreeNode

Quadtree Creation
---------------
.. autosummary::
    :toctree: submodules/

    Points2Quadtree
    Edges2Quadtree  

Conversion From Quadtree
----------------------
.. autosummary::
    :toctree: submodules/

    Quadtree2Pixel
    Quadtree2Dual

Octree
======
.. autosummary::
    :toctree: submodules/

    OctreeNode

Octree Creation
---------------
.. autosummary::
    :toctree: submodules/

    Points2Octree
    Function2Octree
    Surface2Octree
    Voxel2Octree

Conversion From Octree
----------------------
.. autosummary::
    :toctree: submodules/

    Octree2Voxel
    Octree2Dual

Octree Querying
---------------
.. autosummary::
    :toctree: submodules/

    SearchOctree
    SearchOctreeTri

"""
import numpy as np
from . import rays, utils, mesh
import sympy as sp

class TreeNode:
    def __init__(self,parent=None,data=None,level=0,state='unknown'):
        """
        .. autoclass:: TreeNode
            :members:
            :inherited-members:

        The OctreeNode is the basic unit of the octree data structure. The structure
        consists of a series of nodes that reference their parent and child nodes, 
        allowing for traversal of the tree structure.

        Parameters
        ----------
        parent : tree.TreeNode, optional
            The tree node that contains this node, by default None
        data : list or dict, optional
            Data associated with the tree node. The type of data depends on 
            the how the tree was created, by default None.
        level : int, optional
            Depth within the tree structure, by default 0.
            The root node is at level 0, the root's children are at level 1, etc.
        state : str, optional
            Specifies whether the node's place in the tree structure, by default
            'unknown'.

            Possible states are:

            - 'root': This node is the root of the tree

            - 'branch': This is node is an intermediate node between the root and leaves

            - 'leaf': This is node is a terminal end and has no children.

            - 'empty': No data is contained within this node, and it has no children

            - 'unknown': State hasn't been specified.

        """  
        self.children = []
        self.parent = parent
        self.state = state
        self.data = data
        self.level = level

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
        level : int
            Octree level, 0 refers to the root node
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

    def clearData(self,clearChildren=True):
        """
        Reset the data attribute for this node, and optionally all children

        Parameters
        ----------
        clearChildren : bool, optional
            If True, data from child nodes will be recursively cleared, by default True
        """        
        self.data = None
        if clearChildren:
            for child in self.children:
                child.clearData()  
    
    def hasChildren(self):
        """
        Check if the node has any child nodes

        Returns
        -------
        bool
        """        
        # includes both "leaf" and "empty" nodes
        return len(self.children) == 0

class QuadtreeNode(TreeNode):
          
    def __init__(self,centroid,size,parent=None,data=None,level=0,state='unknown'):
        """

        The QuadtreeNode is the basic unit of the quadtree data structure. The structure
        consists of a series of nodes that reference their parent and child nodes, 
        allowing for traversal of the tree structure.

        Parameters
        ----------
        parent : tree.QuadtreeNode, optional
            The quadtree node that contains this node, by default None
        data : list or dict, optional
            Data associated with the quadtree node. The type of data depends on 
            the how the quadtree was created, by default None.
        level : int, optional
            Depth within the tree structure, by default 0.
            The root node is at level 0, the root's children are at level 1, etc.
        state : str, optional
            Specifies whether the node's place in the tree structure, by default
            'unknown'.

            Possible states are:

            - 'root': This node is the root of the quadtree

            - 'branch': This is node is an intermediate node between the root and leaves

            - 'leaf': This is node is a terminal end and has no children.

            - 'empty': No data is contained within this node, and it has no children

            - 'unknown': State hasn't been specified.
        
        centroid : array_like
            Location of the center of the quadtree node
        size : float
            Side length of the cube associated with the quadtree node
        limits : list
            bounds of the quadtree node
        vertices : np.ndarray
            Coordinates of the vertices of the quadtree node

        """  
        self.children = []
        self.parent = parent
        self.state = state
        self.data = data
        self.level = level
        self.centroid = centroid
        self.size = size
        self.limits = None
        self.vertices = None

    def __repr__(self):
        out = f'Quadtree Node ({self.state:s})\nCentroid: {str(self.centroid):s}\nSize: {self.size:f} \n'
        return out
    
    def getLimits(self):
        """
        Get the spatial bounds of the current quadtree node. Limits are formatted
        as [[xmin, xmax], [ymin, ymax]]. These are equivalent 
        to node.centroid +/- (node.size/2).

        Returns
        -------
        limits : list
            list of x, y bounds of the quadtree node
        """        
        if self.limits is None:
            self.limits = np.array([[self.centroid[d]-self.size/2,self.centroid[d]+self.size/2] for d in range(2)])
        return self.limits
    
    def getVertices(self):
        """
        Get the coordinates of the 4 vertices of the square that correspond to the
        quadtree node. 

        Returns
        -------
        vertices : np.ndarray
            Array of vertex coordinates
        """        
        if self.vertices is None:
            [x0,x1],[y0,y1] = self.getLimits()
            self.vertices = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
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
        inside = rays.PointInBox2D(point, *self.getLimits(), inclusive=inclusive)
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
        limits = self.getLimits()
        inside =  np.array([rays.PointInBox2D(point, *limits, inclusive=inclusive) for point in points])
        
        return inside
    
    def ContainsPts(self,points):
        """
        Identify which of a set of a points is contained in the node

        Parameters
        ----------
        points : array_like
            Coordinates of the points (shape=(n,3))

        Returns
        -------
        out : list
            List of indices of the points that are contained within the node
        """
        out = [idx for idx,point in enumerate(points) if self.PointInNode(point)]

        return out
     
    def ContainsEdges(self,edges):
        """
        Identify which of a set of a edges is contained in the node

        Parameters
        ----------
        edges : array_like
            Coordinates of the points of the edges (shape=(n,2,2))

        Returns
        -------
        Intersections : np.ndarray
            List of indices of the edges that are contained within the node
        """
        lims = self.getLimits()
        Intersections = np.array([i for i in range(len(edges)) if rays.SegmentBox2DIntersection(edges[i], lims[0], lims[1])],dtype=int)
        return Intersections
    
    def makeChildren(self, childstate='unknown'):
        """
        Initialize child nodes for the current node

        Parameters
        ----------
        childstate : str, optional
            state to be given to the children, by default 'unknown'.

            Other options are

            - 'branch': This is node is an intermediate node between the root and leaves

            - 'leaf': This is node is a terminal end and has no children.

            - 'empty': No data is contained within this node, and it has no children

        """        
        childSize = self.size/2
        self.children = []
        for xSign,ySign in [(-1,-1),(1,-1),(1,1),(-1,1)]:
            centroid = np.array([self.centroid[0]+xSign*self.size/4, self.centroid[1]+ySign*self.size/4])
            self.children.append(QuadtreeNode(centroid,childSize,parent=self,data=[],level=self.level+1,state=childstate))
            
    def makeChildrenPts(self,points,minsize=0,maxsize=np.inf,maxdepth=np.inf):
        """
        Make child nodes based on points.

        Parameters
        ----------
        points : array_like
            Coordinates of the points (shape=(n,3))
        minsize : float, optional
            Minimum size for octree subdivision, by default 0
        maxsize : float, optional
            Maximum size of a leaf node, nodes large than this must be further subdivided, by default np.inf
        maxdepth : int, optional
            Maximum depth of the occur tree, by default np.inf
        """        
        if self.size > minsize and self.level<maxdepth:
            self.makeChildren()
            
            for child in self.children:
                ptIds = child.ContainsPts(points)
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
        else:
            self.state = 'leaf'
            
    def makeChildrenEdges(self, edges, minsize=0, maxsize=np.inf, maxdepth=np.inf):
        """
        Make child nodes based on edges.

        Parameters
        ----------
        edges : array_like
            Coordinates of the points of the triangles (shape=(n,2,2))
        minsize : float, optional
            Minimum size for quadtree subdivision, by default 0
        maxsize : float, optional
            Maximum size of a leaf node, nodes large than this must be further subdivided, by default np.inf
        maxdepth : int, optional
            Maximum depth of the occur tree, by default np.inf
        """  
        self.makeChildren()
                    
        for child in self.children:
            edgeIds = child.ContainsEdges(edges)
            try:
                edgesInChild = edges[edgeIds]
            except:
                a = 2
            if self.data is not None:
                child.data = [self.data[idx] for idx in edgeIds]
            if len(edgesInChild) > 1: 
                if child.size/2 <= minsize or child.level >= maxdepth:
                    child.state = 'leaf'
                else:
                    child.makeChildrenEdges(edgesInChild,minsize=minsize,maxsize=maxsize,maxdepth=maxdepth)
                    child.state = 'branch'
            elif len(edgesInChild) == 1:
                if child.size > maxsize or child.level < maxdepth:
                    child.makeChildrenEdges(edgesInChild,minsize=minsize,maxsize=maxsize,maxdepth=maxdepth)
                    child.state = 'branch'
                else:
                    child.state = 'leaf'
            elif len(edgesInChild) == 0:
                child.state = 'empty'

class OctreeNode(TreeNode):
          
    def __init__(self,centroid,size,parent=None,data=None,level=0,state='unknown'):
        """
        The OctreeNode is the basic unit of the octree data structure. The structure
        consists of a series of nodes that reference their parent and child nodes, 
        allowing for traversal of the tree structure.

        Parameters
        ----------
        parent : tree.OctreeNode, optional
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
        
        centroid : array_like
            Location of the center of the octree node
        size : float
            Side length of the cube associated with the octree node
        limits : list
            bounds of the octree node
        vertices : np.ndarray
            Coordinates of the vertices of the octree node

        """  
        self.children = []
        self.parent = parent
        self.state = state
        self.data = data
        self.level = level
        self.centroid = centroid
        self.size = size
        self.limits = None
        self.vertices = None

    def __repr__(self):
        out = f'Octree Node ({self.state:s})\nCentroid: {str(self.centroid):s}\nSize: {self.size:f} \n'
        return out
    
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
        limits = self.getLimits()
        inside =  np.array([rays.PointInBox(point, *limits, inclusive=inclusive) for point in points])
        
        return inside
    
    def ContainsPts(self,points):
        """
        Identify which of a set of a points is contained in the node

        Parameters
        ----------
        points : array_like
            Coordinates of the points (shape=(n,3))

        Returns
        -------
        out : list
            List of indices of the points that are contained within the node
        """
        out = [idx for idx,point in enumerate(points) if self.PointInNode(point)]

        return out
    
    def ContainsTris(self,tris,TriNormals=None):
        """
        Identify which of a set of a triangles is contained in the node

        Parameters
        ----------
        tris : array_like
            Coordinates of the points of the triangles (shape=(n,3,3))
        TriNormals : array_like, optional
            Normal vectors of the triangles (shape=(n,3))

        Returns
        -------
        Intersections : np.ndarray
            List of indices of the triangles that are contained within the node
        """
        lims = self.getLimits()
        Intersections = np.where(rays.BoxTrianglesIntersection(tris, lims[0], lims[1], lims[2], TriNormals=TriNormals, BoxCenter=self.centroid))[0]
        return Intersections
    
    def ContainsBoxes(self, boxes):
        """
        Identify which of a set of a boxes is contained in the node

        Parameters
        ----------
        boxes : list
            List of box bounds, formatted as [((xmin, xmax), (ymin, ymax), (zmin, zmax)), ...]

        Returns
        -------
        Intersections : np.ndarray
            List of indices of the boxes that are contained within the node
        """
        Intersections = np.where([rays.BoxBoxIntersection(self.getLimits(), box) for box in boxes])[0]
        return Intersections
    
    def makeChildren(self, childstate='unknown'):
        """
        Initialize child nodes for the current node

        Parameters
        ----------
        childstate : str, optional
            state to be given to the children, by default 'unknown'.

            Other options are

            - 'branch': This is node is an intermediate node between the root and leaves

            - 'leaf': This is node is a terminal end and has no children.

            - 'empty': No data is contained within this node, and it has no children

        """        
        childSize = self.size/2
        self.children = []
        # Note other things (e.g. Function2Octree) depend on this ordering not changing 
        for xSign,ySign,zSign in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]:
            centroid = np.array([self.centroid[0]+xSign*self.size/4, self.centroid[1]+ySign*self.size/4, self.centroid[2]+zSign*self.size/4])
            self.children.append(OctreeNode(centroid,childSize,parent=self,data=[],level=self.level+1,state=childstate))
            
    def makeChildrenPts(self, points, minsize=0, maxsize=np.inf, maxdepth=np.inf):
        """
        Make child nodes based on points.

        Parameters
        ----------
        points : array_like
            Coordinates of the points (shape=(n,3))
        minsize : float, optional
            Minimum size for octree subdivision, by default 0
        maxsize : float, optional
            Maximum size of a leaf node, nodes large than this must be further subdivided, by default np.inf
        maxdepth : int, optional
            Maximum depth of the occur tree, by default np.inf
        """        
        if self.size > minsize and self.level<maxdepth:
            self.makeChildren()
            
            for child in self.children:
                ptIds = child.ContainsPts(points)
                ptsInChild = points[ptIds]#[points[idx] for idx in ptIds]
                if self.data:
                    child.data = [self.data[idx] for idx in ptIds]
                if len(ptsInChild) > 1: 
                    if child.size/2 < minsize:
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
        else:
            self.state = 'leaf'
            
    def makeChildrenTris(self, tris, TriNormals, minsize=0, maxsize=np.inf, maxdepth=np.inf):
        """
        Make child nodes based on triangles.

        Parameters
        ----------
        tris : array_like
            Coordinates of the points of the triangles (shape=(n,3,3))
        TriNormals : array_like, optional
            Normal vectors of the triangles (shape=(n,3))
        minsize : float, optional
            Minimum size for octree subdivision, by default 0
        maxsize : float, optional
            Maximum size of a leaf node, nodes large than this must be further subdivided, by default np.inf
        maxdepth : int, optional
            Maximum depth of the occur tree, by default np.inf
        """  
        self.makeChildren()
                    
        for child in self.children:
            triIds = child.ContainsTris(tris,TriNormals)
            trisInChild = tris[triIds]# [tris[idx] for idx in triIds]
            normalsInChild = TriNormals[triIds]#[TriNormals[idx] for idx in triIds]
            if self.data is not None:
                child.data = [self.data[idx] for idx in triIds]
            if len(trisInChild) >= 1: 
                if child.size/2 < minsize or child.level >= maxdepth:
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
        """
        Make child nodes based on boxes.

        Parameters
        ----------
        boxes : list
            List of box bounds, formatted as [((xmin, xmax), (ymin, ymax), (zmin, zmax)), ...]
        minsize : float, optional
            Minimum size for octree subdivision, by default 0
        maxsize : float, optional
            Maximum size of a leaf node, nodes large than this must be further subdivided, by default np.inf
        maxdepth : int, optional
            Maximum depth of the occur tree, by default np.inf
        """ 
        self.makeChildren()
                    
        for child in self.children:
            boxIds = child.ContainsBoxes(boxes)
            boxesInChild = boxes[boxIds]
            if self.data is not None:
                child.data = [self.data[idx] for idx in boxIds]
            if len(boxesInChild) > 1: 
                if child.size/2 < minsize or child.level >= maxdepth:
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

# Octree Functions               
def isInsideOctree(pt,node,inclusive=True):  
    # This might not be necessary - possibly redundant with OctreeNode.PointInNode - need to verify, then adjust usage in rays
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
    root : tree.OctreeNode
        Root of the octree to be searched

    Returns
    -------
    node : tree.OctreeNode or NoneType
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
    root : tree.OctreeNode
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
    root : tree.OctreeNode
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
    root : tree.OctreeNode
        Root node of the generated octree structure
    """    
    if type(VoxelCoords) is list:
        VoxelCoords = np.array(VoxelCoords)
    # Assumes (and requires) that all voxels are cubic and the same size
    VoxelSize = abs(sum(VoxelCoords[VoxelConn[0][0]] - VoxelCoords[VoxelConn[0][1]]))
    centroids = utils.Centroids(VoxelCoords, VoxelConn)
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

def Surface2Octree(NodeCoords, SurfConn, minsize=None, maxdepth=None, exact_minsize=True):
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
        Maximum depth of the octree, by default 5 unless minsize is set

    Returns
    -------
    root : tree.OctreeNode
        Root node of the generate octree
    """    
    # if type(NodeCoords) is list:
    NodeCoords = np.asarray(NodeCoords, dtype=np.float64)          
    
    ArrayConn = np.asarray(SurfConn, dtype=np.int64)

    minx = min(NodeCoords[:,0])
    maxx = max(NodeCoords[:,0])
    miny = min(NodeCoords[:,1])
    maxy = max(NodeCoords[:,1])
    minz = min(NodeCoords[:,2])
    maxz = max(NodeCoords[:,2])
    
    size = max([maxx-minx,maxy-miny,maxz-minz])

    if minsize is not None and exact_minsize:
        n = np.ceil(np.log2(size/minsize))
        size = minsize * 2**n

    if minsize is None and maxdepth is None:
        maxdepth = 5
        # By default creates octree with a minimum node size equal to the mean size of a triangle
        # minsize = np.nanmean(np.nanmax([np.linalg.norm(NodeCoords[ArrayConn][:,0] - NodeCoords[ArrayConn][:,1],axis=1),
        #     np.linalg.norm(NodeCoords[ArrayConn][:,1] - NodeCoords[ArrayConn][:,2],axis=1),
        #     np.linalg.norm(NodeCoords[ArrayConn][:,2] - NodeCoords[ArrayConn][:,0],axis=1)],axis=0
        #     ))
        minsize = size/2**maxdepth
    elif minsize is None and maxdepth is not None:
        minsize = size/2**maxdepth
    elif minsize is not None and maxdepth is None:
        maxdepth = np.ceil(np.log2(size/minsize))
    
    centroid = np.array([(minx + maxx)/2, (miny+maxy)/2, (minz+maxz)/2])
    ElemIds = list(range(len(SurfConn)))
    root = OctreeNode(centroid,size,data=ElemIds)
    root.state = 'root'

    TriNormals = np.array(utils.CalcFaceNormal(NodeCoords,SurfConn))
    root.makeChildrenTris(NodeCoords[ArrayConn], TriNormals, maxsize=size, minsize=minsize, maxdepth=maxdepth)

    return root

def Mesh2Octree(NodeCoords, NodeConn, minsize=None, mindepth=2, maxdepth=5):
    """
    Generate an octree representation of a volumetric mesh. The octree
    will be generated based on bounding boxes for each element in the mesh

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates of the surface mesh
    NodeConn : array_like
        Node connectivity of the triangular surface mesh. This must be an nx3
        array or list.
    mindepth : float, optional
        Minimum depth of the octree, by default 2.
    maxdepth : int, optional
        Maximum depth of the octree, by default 5

    Returns
    -------
    root : tree.OctreeNode
        Root node of the generate octree
    """   
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
    root : tree.OctreeNode
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

def Octree2Voxel(root, sparse=True):
    """
    Convert an octree to a voxel mesh

    Parameters
    ----------
    root : tree.OctreeNode
        Octree node from which the mesh will be generated. 
    sparse : bool, optional
        Determines voxelization mode. If sparse is True, only leaf nodes that contain
        data will be included, otherwise both leaf and empty nodes
        will be include, by default True. 

    Returns
    -------
    VoxelCoords : np.ndarray
        Node coordinates of the voxel mesh.
    VoxelConn : np.ndarray
        Node connectivity of the hexahedral voxel mesh.

    """    

    nodes = getAllLeaf(root, (not sparse))
    N = len(nodes)
    if N > np.iinfo(np.uint32).max:
        itype = np.uint64
    else:
        itype = np.uint32

    VoxelConn = np.empty((N, 8), dtype=itype)
    VoxelCoords = np.empty((N*8, 3), dtype=np.float64)

    for i,node in enumerate(nodes):
        indices = np.arange(i*8, i*8 + 8)
        VoxelConn[i] = indices
        VoxelCoords[indices,:] = node.getVertices()

    ###
   
    if 'mesh' in dir(mesh):
        Voxel = mesh.mesh(VoxelCoords,VoxelConn,'vol')
    else:
        Voxel = mesh(VoxelCoords,VoxelConn,'vol')
    
    Voxel.cleanup()

    return Voxel

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
    root : tree.OctreeNode
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

# Quadtree Functions
def Points2Quadtree(Points, maxdepth=10):
    """
    Generate an quadtree structure from a set of points. The quadtree will be 
    subdivided until each node contains only one point or the maximum depth
    is met. 

    Parameters
    ----------
    Points : array_like
        Point coordinates (shape=(n,3) or (n,2). If (n,3), the third dimension is ignored).
    maxdepth : int, optional
        Maximum depth of the quadtree, by default 10

    Returns
    -------
    root : tree.Quadtree
        Root node of the generated octree structure.
    """    
    if type(Points) is list:
        Points = np.array(Points)
    minx = np.min(Points[:,0])
    maxx = np.max(Points[:,0])
    miny = np.min(Points[:,1])
    maxy = np.max(Points[:,1])
    size = np.max([maxx-minx,maxy-miny])
    
    centroid = np.array([minx + size/2, miny+size/2])
    
    root = QuadtreeNode(centroid,size,data=[])
    root.state = 'root'
    root.makeChildrenPts(Points, maxdepth=maxdepth)    
    
    return root

def Edges2Quadtree(NodeCoords, LineConn, minsize=None, maxdepth=5):
    """
    Generate an octree representation of a line mesh. The quad
    will be refined until each node contains only one line or the maximum
    depth or minimum size criteria are met. Each node contains a list of 
    element ids corresponding to the elements that are contained within that 
    node in the OctreeNode.data field.

    Parameters
    ----------
    NodeCoords : array_like
        Node coordinates of the line mesh
    LineConn : array_like
        Node connectivity of the line mesh. This must have shape (n,2)
        array or list.
    minsize : float, optional
        Minimum size for an octree node, by default None.
        If supplied, octree nodes will not be divided to be smaller than this
        size.
    maxdepth : int, optional
        Maximum depth of the octree, by default 5

    Returns
    -------
    root : tree.QuadtreeNode
        Root node of the generate quadtree
    """    
    if type(NodeCoords) is list:
        NodeCoords = np.array(NodeCoords)          
    
    ArrayConn = np.asarray(LineConn).astype(int)
    defaultmin = np.mean(np.linalg.norm(NodeCoords[ArrayConn][:,0] - NodeCoords[ArrayConn][:,1],axis=1),axis=0)
    if minsize is None and maxdepth is None:
        # By default creates octree with a minimum node size equal to the mean edge length
        minsize =defaultmin
    elif minsize is None and maxdepth is not None:
        minsize = 0
        
    minx = min(NodeCoords[:,0]) - defaultmin
    maxx = max(NodeCoords[:,0]) + defaultmin
    miny = min(NodeCoords[:,1]) - defaultmin
    maxy = max(NodeCoords[:,1]) + defaultmin
    
    size = max([maxx-minx,maxy-miny])
    centroid = np.array([minx + size/2, miny+size/2])
    ElemIds = list(range(len(LineConn)))
    root = QuadtreeNode(centroid,size,data=ElemIds)
    root.state = 'root'

    root.makeChildrenEdges(NodeCoords[ArrayConn], maxsize=size, minsize=minsize,  maxdepth=maxdepth)

    return root

def Quadtree2Pixel(root, sparse=True):
    """
    Convert an quadtree to a pixel mesh

    Parameters
    ----------
    root : tree.QuadtreeNode
        Quadtree node from which the mesh will be generated. 
    sparse : bool, optional
        Determines pixelization mode. If sparse is True, only leaf nodes that contain
        data will be included, otherwise both leaf and empty nodes
        will be include, by default True. 

    Returns
    -------
    PixelCoords : np.ndarray
        Node coordinates of the pixel mesh.
    PixelConn : np.ndarray
        Node connectivity of the hexahedral pixel mesh.

    """    
    PixelConn = []
    PixelCoords = []
    if sparse:
        condition = lambda node : node.state == 'leaf'
    else:
        condition = lambda node : node.state == 'leaf' or node.state == 'empty' or len(node.children) == 0

    def recurSearch(node):
        if condition(node):
            PixelConn.append([len(PixelCoords)+0, len(PixelCoords)+1, len(PixelCoords)+2, len(PixelCoords)+3])
            PixelCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] - node.size/2, 0]
                )
            PixelCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] - node.size/2, 0]
                )
            PixelCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] + node.size/2, 0]
                )
            PixelCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] + node.size/2, 0]
                )
        elif node.state == 'branch' or node.state == 'root' or node.state == 'unknown':
            for child in node.children:
                recurSearch(child)
    
    recurSearch(root)
    PixelCoords = np.asarray(PixelCoords)
    PixelConn = np.asarray(PixelConn)
    if 'mesh' in dir(mesh):
        Pixel = mesh.mesh(PixelCoords, PixelConn)
    else:
        Pixel = mesh(PixelCoords, PixelConn)
    
    Pixel.cleanup()
    return Pixel

def Quadtree2Dual(root, method='centroid'):
    """
    Converts an quadtree to a mesh that is dual to the quadtree structure. This mesh
    contains quadrilateral elements with nodes contained inside quadtree nodes, rather
    than at the quadtree vertices. At transitions between quadtree node levels,
    some quads may be partially degenerate (i.e. form tris rather than
    quads). Based on the algorithm proposed by :cite:`Schaefer2005` and
    explained by :cite:`Holmlid2010`. 

    Parameters
    ----------
    root : tree.QuadNode
        Root node of the quadtree
    method : str, optional
        Method used for placing the dual vertices within the quadtree nodes, by 
        default 'centroid'.
        
        Currently the only implemented option is to place the vertices at 
        the centroids of the octree nodes.

    Returns
    -------
    DualCoords : np.ndarray
        Array of nodal coordinates.
    DualConn : np.ndarray
        List of node connectivities for the dual mesh.
    """    
    def nodeProc(node, DualCoords, DualConn):
        if not node.hasChildren():
            for child in node.children:
                nodeProc(child, DualCoords, DualConn)

            for idx in [(0,3), (1,2)]:
                faceProcX(node.children[idx[0]],node.children[idx[1]], DualCoords, DualConn)
            for idx in [(0,1), (3,2)]:
                faceProcY(node.children[idx[0]],node.children[idx[1]], DualCoords, DualConn)

            vertProc(*node.children, DualCoords, DualConn)
 
    def faceProcX(n0, n1, DualCoords, DualConn):
        # Nodes should be ordered bottom-top (n0 is below n1)
        if not (n0.hasChildren() and n1.hasChildren()):    
            # c0, c1 are the *top* nodes of n0 and c2, c3, are the *bottom* nodes of n1
            c0 = n0 if n0.hasChildren() else n0.children[3]
            c1 = n0 if n0.hasChildren() else n0.children[2]
        
            c2 = n1 if n1.hasChildren() else n1.children[1]
            c3 = n1 if n1.hasChildren() else n1.children[0]

            faceProcX(c0,c3, DualCoords, DualConn)
            faceProcX(c1,c2, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3, DualCoords, DualConn)

    def faceProcY(n0, n1, DualCoords, DualConn):
        # Nodes should be ordered left-right (n0 is left of n1)
        if not (n0.hasChildren() and n1.hasChildren()):    
            # c0, c3,  are the *right* nodes of n0 and c1, c2, are the *left* nodes of n1
            # The 2x2 of adjacent children is thus [c0,c1,c2,c3]
            c0 = n0 if n0.hasChildren() else n0.children[1]
            c3 = n0 if n0.hasChildren() else n0.children[2]
        
            c1 = n1 if n1.hasChildren() else n1.children[0]
            c2 = n1 if n1.hasChildren() else n1.children[3]

            faceProcY(c0,c1, DualCoords, DualConn)
            faceProcY(c3,c2, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3, DualCoords, DualConn)


    def vertProc(n0, n1, n2, n3, DualCoords, DualConn):
        ns = [n0, n1, n2, n3]
        
        if not all([n.hasChildren() for n in ns]):
            # 4 child nodes that share the same central vertex
            c0 = n0 if n0.hasChildren() else n0.children[2]
            c1 = n1 if n1.hasChildren() else n1.children[3]
            c2 = n2 if n2.hasChildren() else n2.children[0]
            c3 = n3 if n3.hasChildren() else n3.children[1]

            vertProc(c0,c1,c2,c3,DualCoords,DualConn)

        else:
            # create a dual grid element
            if method=='centroid':
                coord = [n.centroid for n in ns]
            DualConn.append(list(range(len(DualCoords),len(DualCoords)+4)))
            DualCoords += coord
    
    DualConn = []
    DualCoords = []     
    nodeProc(root, DualCoords, DualConn)
    DualCoords = np.column_stack([DualCoords, np.zeros(len(DualCoords))])
    DualConn = np.asarray(DualConn)

    if 'mesh' in dir(mesh):
       Dual = mesh.mesh(DualCoords, DualConn)
    else:
        Dual = mesh(DualCoords, DualConn)
    Dual.cleanup()
    return Dual

# Generic Tree Functions
def getAllLeaf(root, include_empty=False):
    """
    Retrieve a list of all leaf nodes of the tree

    Parameters
    ----------
    root : tree.TreeNode
        Root node of the tree of which the leaf nodes will be retrieved.
    include_empty : bool, optional
        Option to include "empty" nodes in the set of leaves, by default False. 
        "empty" nodes are terminal nodes that contain no data.

    Returns
    -------
    leaves : list
        List of tree leaf nodes.
    """    
    # Return a list of all terminal(leaf) nodes in the tree
    def recur(node,leaves):
        if node.state == 'leaf':
            leaves.append(node)
            return leaves
        elif node.state == 'empty':
            if include_empty:
                leaves.append(node)
            return leaves
        elif node.state == 'root' or node.state == 'branch':
            for child in node.children:
                leaves = recur(child,leaves)
        return leaves
    leaves = []
    return recur(root,leaves)

def Print(root, show_empty=False):
    """
    Prints a formatted list of all nodes in the tree.

    Parameters
    ----------
    root : tree.TreeNode
        Root node of the tree
    show_empty : bool, optional
        Option to include 'empty' nodes in the printed tree, by default False.
    """    
    def recur(node):
        if show_empty or node.state != 'empty':
            print('    '*node.level + str(node.level) +'. '+ node.state)
        for child in node.children:
            recur(child)
    
    recur(root)
    