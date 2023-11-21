# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:52:03 2022

@author: toj
"""
import numpy as np
import sys, copy
from . import rays, utils

class OctreeNode():
    def __init__(self,centroid,size,parent=[],data=[],level=0):
        self.centroid = centroid
        self.size = size
        self.children = []
        self.parent = parent
        self.state = 'unknown'
        self.data = data
        self.limits = []
        self.vertices = []
        self.level = level
        
    def PointInNode(self,point,inclusive=True):
        if inclusive:
            return all([(self.centroid[d]-self.size/2) <= point[d] and (self.centroid[d]+self.size/2) >= point[d] for d in range(3)])
        else:
            return all([(self.centroid[d]-self.size/2) < point[d] and (self.centroid[d]+self.size/2) > point[d] for d in range(3)])
        
    def TriInNode(self,tri,TriNormal,inclusive=True):
        lims = self.getLimits()
        return rays.TriangleBoxIntersection(tri, lims[0], lims[1], lims[2], BoxCenter=self.centroid,TriNormal=TriNormal)
        
        # return (any([self.PointInNode(pt,inclusive=inclusive) for pt in tri]) or rays.TriangleBoxIntersection(tri, lims[0], lims[1], lims[2]))
    
    def Contains(self,points):
        return [idx for idx,point in enumerate(points) if self.PointInNode(point)]
    
    def ContainsTris(self,tris,TriNormals):
        
        # return [idx for idx,tri in enumerate(tris) if self.TriInNode(tri,TriNormals[idx])]
        lims = self.getLimits()
        Intersections = np.where(rays.BoxTrianglesIntersection(tris, lims[0], lims[1], lims[2], TriNormals=TriNormals, BoxCenter=self.centroid))[0]
        return Intersections
    
    def isEmpty(self,points):
        return any([self.PointInNode(point) for point in points])
    
    def isLeaf(self):
        if self.state == 'leaf':
            return True
        return False

    def getLimits(self):
        if self.limits == []:
            self.limits = [[self.centroid[d]-self.size/2,self.centroid[d]+self.size/2] for d in range(3)]
        return self.limits
    
    def getVertices(self):
        if self.vertices == []:
            [x0,x1],[y0,y1],[z0,z1] = self.getLimits()
            self.vertices = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                                      [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        return self.vertices
    
    def makeChildrenPts(self,points,minsize=0,maxsize=np.inf):
        if self.size > minsize:
            self.makeChildren()
            
            for child in self.children:
                ptIds = child.Contains(points)
                ptsInChild = [points[idx] for idx in ptIds]
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
            
    def makeChildrenTris(self,tris,TriNormals,minsize=0,maxsize=np.inf):
        # tris is a list of Triangular vertices [tri1,tri2,...] where tri1 = [pt1,pt2,pt3]
        self.makeChildren()
                    
        for child in self.children:
            triIds = child.ContainsTris(tris,TriNormals)
            trisInChild = tris[triIds]# [tris[idx] for idx in triIds]
            normalsInChild = TriNormals[triIds]#[TriNormals[idx] for idx in triIds]
            if self.data:
                child.data = [self.data[idx] for idx in triIds]
            if len(trisInChild) > 1: 
                if child.size/2 <= minsize:
                    child.state = 'leaf'
                else:
                    child.makeChildrenTris(trisInChild,normalsInChild,minsize=minsize,maxsize=maxsize)
                    child.state = 'branch'
            elif len(trisInChild) == 1:
                if child.size <= maxsize:
                    child.state = 'leaf'
                else:
                    child.makeChildrenTris(trisInChild,normalsInChild,minsize=minsize,maxsize=maxsize)
                    child.state = 'branch'
            elif len(trisInChild) == 0:
                child.state = 'empty'
            # self.children.append(child) 

    def makeChildrenFunc(self, func, grad, minsize=0, maxsize=np.inf, strategy='QEF', npts=5, eps=0.01):
        # Currently only strategy is QEF, but making an option for future expansion
        # QEF (Quadratic Error Function) strategy based on Dual Marching Cubes: Primal Contouring of Dual Grids, Schaeffer & Warren (2005)
        # EDError (Euclidean Distance Error) based on Zhang, Bajaj, & Sohn (2003, 2005)
        subdivide = False
        MinSize = minsize
        if self.size > maxsize:
            subdivide = True
        elif strategy == 'QEF':
            # https://www.mattkeeter.com/projects/qef/
            #
            # create grid of sample points
            
            e, crosses_zero = runQEF(self, func, grad, npts=npts, eps=eps)
            if e > eps and crosses_zero:
                subdivide = True
            
        elif strategy == 'EDError':
            npts = 3
            [x0,x1],[y0,y1],[z0,z1] = self.getLimits()
            
            nodef = self.data['f']
            f_interp = lambda x,y,z : nodef[0]*(1-x)*(1-y)*(1-z) +  \
                                        nodef[1]*x*(1-y)*(1-z) + \
                                        nodef[2]*x*y*(1-z) + \
                                        nodef[3]*(1-x)*y*(1-z) + \
                                        nodef[4]*(1-x)*(1-y)*z +  \
                                        nodef[5]*x*(1-y)*z + \
                                        nodef[6]*x*y*z + \
                                        nodef[7]*(1-x)*y*z
            # NOTE: edge numbering consistent with that of contour.py
            half = self.size/2
            sample_pts = np.array([[x0+half, y0, z0], # 0-edge 0
                                   [x1, y0+half, z0], # 1-edge 1
                                   [x0+half, y1, z0], # 2-edge 2
                                   [x0, y0+half, z0], # 3-edge 3
                                   [x0, y0, z0+half], # 4-edge 4
                                   [x1, y0, z0+half], # 5-edge 5
                                   [x1, y1, z0+half], # 6-edge 6
                                   [x0, y1, z0+half], # 7-edge 7
                                   [x0+half, y0, z1], # 8-edge 8
                                   [x1, y0+half, z1], # 9-edge 9
                                   [x0+half, y1, z1], # 10-edge 10
                                   [x0, y0+half, z1], # 11-edge 11
                                   [x0+half, y0+half, z0], # 12-face 0
                                   [x0+half, y0, z0+half], # 13-face 1
                                   [x1, y0+half, z0+half], # 14-face 2
                                   [x0+half, y1, z0+half], # 15-face 3
                                   [x0, y0+half, z0+half], # 16-face 4
                                   [x0+half, y0+half, z1], # 17-face 5
                                   [x0+half, y0+half, z0+half] # 18-center
                                   ])
            
            sample_f = func(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2])
            sample_g = grad(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2])
            interp_f = f_interp(sample_pts[:,0], sample_pts[:,1], sample_pts[:,2])
            
            e = np.nansum(np.abs(sample_f - interp_f)/np.linalg.norm(sample_g,axis=1))
            if e > eps:
                subdivide = True
            else:
                a = 2
        
        elif strategy == 'spatial':
            assert callable(minsize), 'minsize must be a function of (x,y,z) if strategy=="spatial".'
            v = np.append(self.getVertices(), [self.centroid], axis=0)
            if not np.all(minsize(v[:,0],v[:,1],v[:,2])) <= self.size:
                subdivide = True
            
            
        if subdivide:
            self.makeChildren()        
            for child in self.children: child.data = dict()
            if strategy == 'EDError':
                if self.size > maxsize:
                    for child in self.children:
                        v = child.getVertices()
                        child.data['f'] = func(v[:,0],v[:,1],v[:,2])
                        child.data['g'] = grad(v[:,0],v[:,1],v[:,2])
                else:
                    # Pass the vertex values that have already been calculated to children
                    self.children[0].data['f'] = np.array([self.data['f'][0], sample_f[0], sample_f[12], sample_f[3],
                                                            sample_f[4], sample_f[13], sample_f[18], sample_f[16]])
                    self.children[1].data['f'] = np.array([sample_f[0], self.data['f'][1], sample_f[1], sample_f[12],
                                                            sample_f[13], sample_f[5], sample_f[14], sample_f[18]])
                    self.children[2].data['f'] = np.array([sample_f[12], sample_f[1], self.data['f'][2], sample_f[2],
                                                            sample_f[18], sample_f[14], sample_f[6], sample_f[15]])
                    self.children[3].data['f'] = np.array([sample_f[3], sample_f[12], sample_f[2], self.data['f'][3],
                                                            sample_f[16], sample_f[18], sample_f[15], sample_f[7]])
                    self.children[4].data['f'] = np.array([sample_f[4], sample_f[13], sample_f[18], sample_f[16],
                                                            self.data['f'][4], sample_f[8], sample_f[17], sample_f[11]])
                    self.children[5].data['f'] = np.array([sample_f[13], sample_f[5], sample_f[14], sample_f[18],
                                                            sample_f[8], self.data['f'][5], sample_f[9], sample_f[17]])
                    self.children[6].data['f'] = np.array([sample_f[18], sample_f[14], sample_f[6], sample_f[15],
                                                            sample_f[17], sample_f[9], self.data['f'][6], sample_f[10]])
                    self.children[7].data['f'] = np.array([sample_f[16], sample_f[18], sample_f[15], sample_f[7],
                                                            sample_f[11], sample_f[17], sample_f[10], self.data['f'][7]])
                    
                    self.children[0].data['g'] = np.array([self.data['g'][0], sample_g[0], sample_g[12], sample_g[3],
                                                            sample_g[4], sample_g[13], sample_g[18], sample_g[16]])
                    self.children[1].data['g'] = np.array([sample_g[0], self.data['g'][1], sample_g[1], sample_g[12],
                                                            sample_g[13], sample_g[5], sample_g[14], sample_g[18]])
                    self.children[2].data['g'] = np.array([sample_g[12], sample_g[1], self.data['g'][2], sample_g[2],
                                                            sample_g[18], sample_g[14], sample_g[6], sample_g[15]])
                    self.children[3].data['g'] = np.array([sample_g[3], sample_g[12], sample_g[2], self.data['g'][3],
                                                            sample_g[16], sample_g[18], sample_g[15], sample_g[7]])
                    self.children[4].data['g'] = np.array([sample_g[4], sample_g[13], sample_g[18], sample_g[16],
                                                            self.data['g'][4], sample_g[8], sample_g[17], sample_g[11]])
                    self.children[5].data['g'] = np.array([sample_g[13], sample_g[5], sample_g[14], sample_g[18],
                                                            sample_g[8], self.data['g'][5], sample_g[9], sample_g[17]])
                    self.children[6].data['g'] = np.array([sample_g[18], sample_g[14], sample_g[6], sample_g[15],
                                                            sample_g[17], sample_g[9], self.data['g'][6], sample_g[10]])
                    self.children[7].data['g'] = np.array([sample_g[16], sample_g[18], sample_g[15], sample_g[7],
                                                            sample_g[11], sample_g[17], sample_g[10], self.data['g'][7]])
                
                                
                
            for child in self.children:
                if strategy == 'spatial':
                    MinSize = minsize(child.centroid[0], child.centroid[1], child.centroid[2])
                if child.size/2 <= MinSize:
                    child.state = 'leaf'
                    if strategy == 'QEF':
                        ec = runQEF(child, func, grad, npts=npts, eps=eps)
                        
                else:
                    child.makeChildrenFunc(func,grad,minsize=minsize,maxsize=maxsize, strategy=strategy, npts=npts, eps=eps)
                    if len(child.children) == 0:
                        child.state = 'leaf'
                    else:
                        child.state = 'branch'
        else:
            self.state = 'leaf'

    def makeChildren(self):
        childSize = self.size/2
        self.children = []
        for xSign,ySign,zSign in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]:
            centroid = [self.centroid[0]+xSign*self.size/4, self.centroid[1]+ySign*self.size/4, self.centroid[2]+zSign*self.size/4]
            self.children.append(OctreeNode(centroid,childSize,parent=self,data=[],level=self.level+1))

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
    Root.makeChildrenPts(centroids, maxsize=VoxelSize)    
    
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
    ArrayConn = np.asarray(SurfConn).astype(int)
    if not minsize:
        # By default creates octree with a minimum node size equal to the mean size of a triangle
        # minsize = np.mean([max([max(NodeCoords[elem][:,0])-min(NodeCoords[elem][:,0]),
        #             max(NodeCoords[elem][:,1])-min(NodeCoords[elem][:,1]),
        #             max(NodeCoords[elem][:,2])-min(NodeCoords[elem][:,2]),
        #             ]) for elem in SurfConn])
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
    # Root.makeChildrenTris([(NodeCoords[elem]) for elem in SurfConn],maxsize=minsize,minsize=minsize)
    TriNormals = np.array(utils.CalcFaceNormal(NodeCoords,SurfConn))
    Root.makeChildrenTris(NodeCoords[ArrayConn],TriNormals,maxsize=minsize,minsize=minsize)
    return Root

def Function2Octree(func, grad, bounds, minsize=None, maxsize=None, strategy='QEF', npts=5, eps=0.01):
    # Function value and gradient evaluated at the vertices is stored as `data` in each node
    # func and grad should both accept 3 arguments (x,y,z), and handle both vectorized and scalar inputs

    size = max([bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4]])
    centroid = [bounds[0] + size/2, bounds[2]+size/2, bounds[4]+size/2]

    Root = OctreeNode(centroid, size)
    vertices = Root.getVertices()
    Root.data = dict(f = func(vertices[:,0],vertices[:,1],vertices[:,2]), 
                     g = grad(vertices[:,0],vertices[:,1],vertices[:,2]))
    Root.state = 'root'
    Root.makeChildrenFunc(func, grad, minsize=minsize, maxsize=maxsize, npts=npts, eps=eps, strategy=strategy)


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
    VoxCoords = np.asarray(VoxCoords)
    return VoxCoords, VoxConn

def Octree2Image(root,OctreeDepth,mode='corners'):
    
    def recur(node):
        for child in node.children:
            pass
            
    
    recur.depth = 0

def Octree2Dual(root, method='centroid'):
    # https://www.volume-gfx.com/volume-rendering/dual-marching-cubes/deriving-the-dualgrid/
    # Holmlid, 2010
    def nodeProc(node, DualCoords, DualConn):
        if not node.isLeaf():
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
        if not (n0.isLeaf() and n1.isLeaf()):    
            # c0, c1, c2, c3 are the *top* nodes of n0 and c4, c5, c6, c7 are the *bottom* nodes of n1
            c0 = n0 if n0.isLeaf() else n0.children[4]
            c1 = n0 if n0.isLeaf() else n0.children[5]
            c2 = n0 if n0.isLeaf() else n0.children[6]
            c3 = n0 if n0.isLeaf() else n0.children[7]
        
            c4 = n1 if n1.isLeaf() else n1.children[0]
            c5 = n1 if n1.isLeaf() else n1.children[1]
            c6 = n1 if n1.isLeaf() else n1.children[2]
            c7 = n1 if n1.isLeaf() else n1.children[3]

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
        if not (n0.isLeaf() and n1.isLeaf()):    
            # c0, c3, c7, c4 are the *right* nodes of n0 and c1, c2, c6, c5 are the *left* nodes of n1
            # The 2x2 of adjacent children is thus [c0,c1,c2,c3,c4,c5,c6,c7,c8]
            c0 = n0 if n0.isLeaf() else n0.children[1]
            c3 = n0 if n0.isLeaf() else n0.children[2]
            c7 = n0 if n0.isLeaf() else n0.children[6]
            c4 = n0 if n0.isLeaf() else n0.children[5]
        
            c1 = n1 if n1.isLeaf() else n1.children[0]
            c2 = n1 if n1.isLeaf() else n1.children[3]
            c6 = n1 if n1.isLeaf() else n1.children[7]
            c5 = n1 if n1.isLeaf() else n1.children[4]

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
        if not (n0.isLeaf() and n1.isLeaf()):    
            # c0, c1, c5, c4 are the *back* nodes of n0 and c3, c2, c6, c7 are the *front* nodes of n1
            # The 2x2 of adjacent children is thus [c0,c1,c2,c3,c4,c5,c6,c7,c8]
            c0 = n0 if n0.isLeaf() else n0.children[3]
            c1 = n0 if n0.isLeaf() else n0.children[2]
            c5 = n0 if n0.isLeaf() else n0.children[6]
            c4 = n0 if n0.isLeaf() else n0.children[7]
            c3 = n1 if n1.isLeaf() else n1.children[0]
            c2 = n1 if n1.isLeaf() else n1.children[1]
            c6 = n1 if n1.isLeaf() else n1.children[5]
            c7 = n1 if n1.isLeaf() else n1.children[4]

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
        if not all([n0.isLeaf(), n1.isLeaf(), n2.isLeaf(), n3.isLeaf()]):
            c1 = n0 if n0.isLeaf() else n0.children[6]
            c0 = n0 if n0.isLeaf() else n0.children[7]
            c3 = n1 if n1.isLeaf() else n1.children[4]
            c2 = n1 if n1.isLeaf() else n1.children[5]
            c7 = n2 if n2.isLeaf() else n2.children[0]
            c6 = n2 if n2.isLeaf() else n2.children[1]
            c5 = n3 if n3.isLeaf() else n3.children[2]
            c4 = n3 if n3.isLeaf() else n3.children[3]

            edgeProcX(c1,c2,c6,c5, DualCoords, DualConn)
            edgeProcX(c0,c3,c7,c4, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def edgeProcY(n0,n1,n2,n3, DualCoords, DualConn):
        # Nodes should be ordered counter clockwise about the axis
        if not all([n0.isLeaf(), n1.isLeaf(), n2.isLeaf(), n3.isLeaf()]):
            c0 = n0 if n0.isLeaf() else n0.children[5]
            c3 = n0 if n0.isLeaf() else n0.children[6]
            c1 = n1 if n1.isLeaf() else n1.children[4]
            c2 = n1 if n1.isLeaf() else n1.children[7]
            c5 = n2 if n2.isLeaf() else n2.children[0]
            c6 = n2 if n2.isLeaf() else n2.children[3]
            c4 = n3 if n3.isLeaf() else n3.children[1]
            c7 = n3 if n3.isLeaf() else n3.children[2]

            edgeProcY(c0,c1,c5,c4, DualCoords, DualConn)
            edgeProcY(c3,c2,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def edgeProcZ(n0,n1,n2,n3, DualCoords, DualConn):
        # Nodes should be ordered counter clockwise about the axis
        if not all([n0.isLeaf(), n1.isLeaf(), n2.isLeaf(), n3.isLeaf()]):
            c0 = n0 if n0.isLeaf() else n0.children[2]
            c4 = n0 if n0.isLeaf() else n0.children[6]
            c1 = n1 if n1.isLeaf() else n1.children[3]
            c5 = n1 if n1.isLeaf() else n1.children[7]
            c2 = n2 if n2.isLeaf() else n2.children[0]
            c6 = n2 if n2.isLeaf() else n2.children[4]
            c3 = n3 if n3.isLeaf() else n3.children[1]
            c7 = n3 if n3.isLeaf() else n3.children[5]

            edgeProcZ(c0,c1,c2,c3, DualCoords, DualConn)
            edgeProcZ(c4,c5,c6,c7, DualCoords, DualConn)

            vertProc(c0,c1,c2,c3,c4,c5,c6,c7, DualCoords, DualConn)

    def vertProc(n0, n1, n2, n3, n4, n5, n6, n7, DualCoords, DualConn):
        ns = [n0, n1, n2, n3, n4, n5, n6, n7]
        
        if not all([n.isLeaf() for n in ns]):
            c0 = n0 if n0.isLeaf() else n0.children[6]
            c1 = n1 if n1.isLeaf() else n1.children[7]
            c2 = n2 if n2.isLeaf() else n2.children[4]
            c3 = n3 if n3.isLeaf() else n3.children[5]
            c4 = n4 if n4.isLeaf() else n4.children[2]
            c5 = n5 if n5.isLeaf() else n5.children[3]
            c6 = n6 if n6.isLeaf() else n6.children[0]
            c7 = n7 if n7.isLeaf() else n7.children[1]

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
    return DualCoords, DualConn

def Print(root):
    
    def recur(node):
        print('    '*node.level + str(node.level) +'.'+ node.state)
        for child in node.children:
            recur(child)
    
    recur(root)
    
def runQEF(node, func, grad, npts=5, eps=0.01):
    # ref: https://www.mattkeeter.com/projects/qef/
    # n = 5 # arbitrary
    # eps = .01 # arbitrary

    [x0,x1],[y0,y1],[z0,z1] = node.getLimits()
    X,Y,Z = np.meshgrid(np.linspace(x0,x1,npts),np.linspace(y0,y1,npts),np.linspace(z0,z1,npts))
    xi = X.flatten()
    yi = Y.flatten()
    zi = Z.flatten()
    
    f = func(xi,yi,zi)
    g = grad(xi,yi,zi)
    
    p = np.vstack([xi,yi,zi]).T
    
    # A = np.nan_to_num(g/np.linalg.norm(g,axis=1)[:,None])
    # B = np.sum(A*p,axis=1)
    
    # xopt = np.linalg.solve(A.T@A,A.T@B)
    # node.data['xopt'] = xopt
    
    
    # e = (A@xopt - B).T @ (A@xopt - B)
    
    # More readable, less efficient:
    # Ti = lambda i, x, y, z : np.dot(g[i], np.array([x,y,z]) - np.array([xi[i],yi[i],zi[i]]))
    # E = lambda w, x, y, z : np.sum([(w - Ti(i,x,y,z))**2 / (1 + np.linalg.norm(g[i])**2) for i in range(len(xi))])
    # e = [E(f[i],xi[i],yi[i],zi[i]) for i in range(len(xi))]

    # Evaluate QEF at each subgrid point
    def E(f,x,y,z):
        t = np.sum(g[:,None,:] * (np.vstack([x,y,z]).T - np.vstack([xi, yi, zi]).T[:,None,:]),axis=2)
        e = np.sum((f[None,:] - t)**2 / (1 + np.linalg.norm(g,axis=1)**2)[:,None], axis=0)
        return e
    
    e = np.nan_to_num(E(f,xi,yi,zi))
    idx = np.argmin(e)
    node.data['xopt'] = p[idx]
    # print(e[idx])
    crosses_zero = True if len(np.unique(np.sign(f))) > 1 else False
    if crosses_zero:
        a = 2
    return e[idx], crosses_zero