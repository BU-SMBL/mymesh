# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:52:03 2022

@author: toj
"""
import numpy as np
import sys, copy
from . import Rays, MeshUtils

class OctreeNode():
    def __init__(self,centroid,size,parent=[],data=[]):
        self.centroid = centroid
        self.size = size
        self.children = []
        self.parent = parent
        self.state = 'unknown'
        self.data = data
        self.limits = []
        
    def PointInNode(self,point,inclusive=True):
        if inclusive:
            return all([(self.centroid[d]-self.size/2) <= point[d] and (self.centroid[d]+self.size/2) >= point[d] for d in range(3)])
        else:
            return all([(self.centroid[d]-self.size/2) < point[d] and (self.centroid[d]+self.size/2) > point[d] for d in range(3)])
        
    def TriInNode(self,tri,TriNormal,inclusive=True):
        lims = self.getLimits()
        return Rays.TriangleBoxIntersection(tri, lims[0], lims[1], lims[2], BoxCenter=self.centroid,TriNormal=TriNormal)
        
        # return (any([self.PointInNode(pt,inclusive=inclusive) for pt in tri]) or Rays.TriangleBoxIntersection(tri, lims[0], lims[1], lims[2]))
    
    def Contains(self,points):
        return [idx for idx,point in enumerate(points) if self.PointInNode(point)]
    
    def ContainsTris(self,tris,TriNormals):
        
        # return [idx for idx,tri in enumerate(tris) if self.TriInNode(tri,TriNormals[idx])]
        lims = self.getLimits()
        Intersections = np.where(Rays.BoxTrianglesIntersection(tris, lims[0], lims[1], lims[2], TriNormals=TriNormals, BoxCenter=self.centroid))[0]
        return Intersections
    
    def isEmpty(self,points):
        return any([self.PointInNode(point) for point in points])
    
    def getLimits(self):
        if self.limits == []:
            self.limits = [[self.centroid[d]-self.size/2,self.centroid[d]+self.size/2] for d in range(3)]
        return self.limits
    
    def makeChildrenPts(self,points,minSize=0,maxSize=np.inf):
        if self.size > minSize:
            self.makeChildren()
            
            for child in self.children:
                ptIds = child.Contains(points)
                ptsInChild = [points[idx] for idx in ptIds]
                if self.data:
                    child.data = [self.data[idx] for idx in ptIds]
                if len(ptsInChild) > 1: 
                    if child.size/2 <= minSize:
                        child.state = 'leaf'
                    else:
                        child.makeChildrenPts(ptsInChild,minSize=minSize,maxSize=maxSize)
                        child.state = 'branch'
                elif len(ptsInChild) == 1:
                    if child.size <= maxSize:
                        child.state = 'leaf'
                    else:
                        child.makeChildrenPts(ptsInChild,minSize=minSize,maxSize=maxSize)
                        child.state = 'branch'
                else:
                    child.state = 'empty'
                # self.children.append(child)  
        else:
            self.state = 'leaf'
            
    def makeChildrenTris(self,tris,TriNormals,minSize=0,maxSize=np.inf):
        # tris is a list of Triangular vertices [tri1,tri2,...] where tri1 = [pt1,pt2,pt3]
        self.makeChildren()
                    
        for child in self.children:
            triIds = child.ContainsTris(tris,TriNormals)
            trisInChild = tris[triIds]# [tris[idx] for idx in triIds]
            normalsInChild = TriNormals[triIds]#[TriNormals[idx] for idx in triIds]
            if self.data:
                child.data = self.data[triIds] #[self.data[idx] for idx in triIds]
            if len(trisInChild) > 1: 
                if child.size/2 <= minSize:
                    child.state = 'leaf'
                else:
                    child.makeChildrenTris(trisInChild,normalsInChild,minSize=minSize,maxSize=maxSize)
                    child.state = 'branch'
            elif len(trisInChild) == 1:
                if child.size <= maxSize:
                    child.state = 'leaf'
                else:
                    child.makeChildrenTris(trisInChild,normalsInChild,minSize=minSize,maxSize=maxSize)
                    child.state = 'branch'
            elif len(trisInChild) == 0:
                child.state = 'empty'
            # self.children.append(child)  
            
    def makeChildren(self):
        childSize = self.size/2
        self.children = []
        for zSign in [-1,1]:
            for ySign in [-1,1]:
                for xSign in [-1,1]:
                    centroid = [self.centroid[0]+xSign*self.size/4, self.centroid[1]+ySign*self.size/4, self.centroid[2]+zSign*self.size/4]
                    self.children.append(OctreeNode(centroid,childSize,parent=self,data=[]))

    def addTri(self,tri,triId=None,minSize=None):
            # triId can be an identifier for the element corresponding to the given triangle
            # If given, triId will be stored in the octree node data instead of the tri itself
            if not minSize:
                # By default creates octree with a minimum node size equal to the max edge length of a triangle
                minSize = max([max([pt[0] for pt in tri])-min([pt[0] for pt in tri]),
                            max([pt[1] for pt in tri])-min([pt[1] for pt in tri]),
                            max([pt[2] for pt in tri])-min([pt[2] for pt in tri]),
                            ])
            def recur(node,tri,triId,minSize):
                if node.TriInNode(tri):
                    if node.state == 'unknown' or node.state == 'empty':
                        node.state = 'branch'
                    if node.size/2 <= minSize:
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
                            recur(child,tri,triId,minSize)
                elif node.state == 'unknown':
                    node.state = 'empty'
            recur(self,tri,triId,minSize)
            
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
            
def SearchOctree(pt,node,inclusive=True):
    if node.PointInNode(pt,inclusive=inclusive):
        if node.state == 'leaf':
            return node
        else:
            for child in node.children:
                if child.state == 'empty':
                    continue
                check = SearchOctree(pt,child)
                if check:
                    return check
            return False
                    
    else:
        return False
    
def SearchOctreeTri(tri,node,nodes=[],inclusive=True):
    # print(nodes)
    if node.TriInNode(tri,inclusive=inclusive):
        if node.state == 'leaf':
            nodes.append(node)
        else:
            for i,child in enumerate(node.children):
                if child.state == 'empty':
                    continue
                nodes = SearchOctreeTri(tri,child,nodes=nodes,inclusive=inclusive)
    return nodes
    
def getAllLeaf(root):
    # Return a list of all terminal(leaf) nodes in the octree
    def recur(node,leaf):
        if node.state == 'leaf':
            leaf.append(node)
            return leaf
        elif node.state == 'empty':
            return leaf
        elif node.state == 'root' or node.state == 'branch':
            for child in node.children:
                leaf = recur(child,leaf)
        return leaf
    leaf = []
    return recur(root,leaf)

def Voxel2Octree(VoxelCoords, VoxelConn):
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
    
    centroid = [minx + size/2, miny+size/2, minz+size/2]
    
    Root = OctreeNode(centroid,size,data=[])
    Root.state = 'root'
    Root.makeChildrenPts(centroids, maxSize=VoxelSize)    
    
    return Root

def Octree2Sparse(root):
    # Needs to be tested
    root = copy.deepcopy(root)
    def recur(node):
        if all(child.state == 'leaf' for child in node.children):
            node.children = []
            node.state = 'leaf'
            recur.changes += 1
        else:
            for child in node.children:
                if child.state == 'branch':
                    recur(child)
    thinking = True
    while thinking:
        recur.changes = 0
        recur(root)
        if recur.changes == 0:
            thinking = False
            
    return root
    
def Surf2Octree(NodeCoords, SurfConn, minsize=None):
    
    if type(NodeCoords) is list:
        NodeCoords = np.array(NodeCoords)          
    # centroids = [np.mean(NodeCoords[elem],axis=0) for elem in SurfConn]
    if not minsize:
        # By default creates octree with a minimum node size equal to the mean size of a triangle
        # minsize = np.mean([max([max(NodeCoords[elem][:,0])-min(NodeCoords[elem][:,0]),
        #             max(NodeCoords[elem][:,1])-min(NodeCoords[elem][:,1]),
        #             max(NodeCoords[elem][:,2])-min(NodeCoords[elem][:,2]),
        #             ]) for elem in SurfConn])
        ArrayConn = np.asarray(SurfConn)
        minsize = np.nanmean(np.nanmax([np.linalg.norm(NodeCoords[ArrayConn][:,0] - NodeCoords[ArrayConn][:,1],axis=1),
            np.linalg.norm(NodeCoords[ArrayConn][:,1] - NodeCoords[ArrayConn][:,2],axis=1),
            np.linalg.norm(NodeCoords[ArrayConn][:,2] - NodeCoords[ArrayConn][:,0],axis=1)],axis=0
            ))
        # print(minsize)
    minx = min(NodeCoords[:,0])
    maxx = max(NodeCoords[:,0])
    miny = min(NodeCoords[:,1])
    maxy = max(NodeCoords[:,1])
    minz = min(NodeCoords[:,2])
    maxz = max(NodeCoords[:,2])
    
    maxx = np.arange(minx,maxx+minsize,minsize)[-1]
    maxy = np.arange(miny,maxy+minsize,minsize)[-1]
    maxz = np.arange(minz,maxz+minsize,minsize)[-1]
    
    size = max([maxx-minx,maxy-miny,maxz-minz])
    centroid = [minx + size/2, miny+size/2, minz+size/2]
    ElemIds = list(range(len(SurfConn)))
    Root = OctreeNode(centroid,size,data=ElemIds)
    Root.state = 'root'
    # Root.makeChildrenTris([(NodeCoords[elem]) for elem in SurfConn],maxSize=minsize,minSize=minsize)
    TriNormals = np.array(MeshUtils.CalcFaceNormal(NodeCoords,SurfConn))
    Root.makeChildrenTris(NodeCoords[ArrayConn],TriNormals,maxSize=minsize,minSize=minsize)
    return Root
    
def Octree2Voxel(root, mode='full'):
    # TODO: Implement option for sparse
    VoxConn = []
    VoxCoords = []
    def recurSearch(node):
        if node.state == 'leaf':
            VoxConn.append([len(VoxCoords)+0, len(VoxCoords)+1, len(VoxCoords)+2, len(VoxCoords)+3,
                            len(VoxCoords)+4, len(VoxCoords)+5, len(VoxCoords)+6, len(VoxCoords)+7])
            VoxCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] - node.size/2, node.centroid[2] - node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] - node.size/2, node.centroid[2] - node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] + node.size/2, node.centroid[2] - node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] + node.size/2, node.centroid[2] - node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] - node.size/2, node.centroid[2] + node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] - node.size/2, node.centroid[2] + node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] + node.size/2, node.centroid[1] + node.size/2, node.centroid[2] + node.size/2]
                )
            VoxCoords.append(
                [node.centroid[0] - node.size/2, node.centroid[1] + node.size/2, node.centroid[2] + node.size/2]
                )
        elif node.state == 'branch' or node.state == 'root' or node.state == 'unknown':
            for child in node.children:
                recurSearch(child)
    
    recurSearch(root)
    return VoxCoords, VoxConn

def Octree2Image(root,OctreeDepth,mode='corners'):
    
    def recur(node):
        for child in node.children:
            pass
            
    
    recur.depth = 0