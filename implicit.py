# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 17:43:57 2022

@author: toj
"""

# %%

import numpy as np
from scipy.spatial import distance
from scipy import optimize, interpolate
import sys, os, time, copy, warnings, bisect
import meshio

from . import utils, converter, contour, quality, improvement, rays, octree, mesh, primitives

try:
    from sklearn.neighbors import KDTree
except:
    warnings.warn('Optional dependencies not found - some functions may not work properly')

# Implicit functions
def gyroid(x,y,z):
    return np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + np.sin(2*np.pi*y)*np.cos(2*np.pi*z) + np.sin(2*np.pi*z)*np.cos(2*np.pi*x)

def lidinoid(x,y,z):
    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z
    return 0.5*(np.sin(2*X)*np.cos(Y)*np.sin(Z) + np.sin(2*Y)*np.cos(Z)*np.sin(X) + np.sin(2*Z)*np.cos(X)*np.sin(Y)) - 0.5*(np.cos(2*X)*np.cos(2*Y) + np.cos(2*Y)*np.cos(2*Z) + np.cos(2*Z)*np.cos(2*X)) + 0.15

def primitive(x,y,z):
    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z
    return np.cos(X) + np.cos(Y) + np.cos(Z)

def neovius(x,y,z):
    X = 2*np.pi*x
    Y = 2*np.pi*y
    Z = 2*np.pi*z
    return 3*(np.cos(X) + np.cos(Y) + np.cos(Z)) + 4*np.cos(X)*np.cos(Y)*np.cos(Z)

def diamond(x,y,z):
    return np.sin(2*np.pi*x)*np.sin(2*np.pi*y)*np.sin(2*np.pi*z) + np.sin(2*np.pi*x)*np.cos(2*np.pi*y)*np.cos(2*np.pi*z) + np.cos(2*np.pi*x)*np.sin(2*np.pi*y)*np.cos(2*np.pi*z) + np.cos(2*np.pi*x)*np.cos(2*np.pi*y)*np.sin(2*np.pi*z)

def cylinder(x,y,r):
    return (x**2 + y**2) - r**2

def cube(x,y,z,x1,x2,y1,y2,z1,z2):    
    return intersection(intersection(intersection(x1-x,x-x2),intersection(y1-y,y-y2)),intersection(z1-z,z-z2))

def xplane(x,y,z,x0):
    return np.linalg.norm(x0-x)

def yplane(x,y,z,y0):
    return np.linalg.norm(y0-y)

def zplane(x,y,z,z0):
    return np.linalg.norm(z0-z)

def sphere(x,y,z,r,c):
    return (x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2 - r**2

# Implicit function operators
def offset(fval,offset):
    return fval-offset

def union(fval1,fval2):
#    return np.minimum(fval1,fval2)
    return rMin(fval1,fval2)

def diff(fval1,fval2):
#    return np.minimum(fval1,-fval2)
    # This has maybe been wrong for a long time changing min to max
    # return rMin(fval1,-fval2)
    return rMax(fval1,-fval2)

def intersection(fval1,fval2):
#    return np.maximum(fval1,fval2)
    return rMax(fval1,fval2)

def rMax(a,b,alpha=0,m=0,p=2):
    # R-Function version of max(a,b) to yield a smoothly differentiable max - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b+(np.maximum(a**p+b**p - 2*alpha*a*b,0))**(1/p)) * (a**2 + b**2)**(m/2)

def rMin(a,b,alpha=0,m=0,p=2):
    # R-Function version of min(a,b) to yield a smoothly differentiable min - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b-(np.maximum(a**p+b**p - 2*alpha*a*b,0))**(1/p)) * (a**2 + b**2)**(m/2)

def VoxelMesh(sdf,xlims,ylims,zlims,h,mode='liberal',values='nodes'):
    """
    VoxelMesh Generate voxel mesh of a signed distance function

    Parameters
    ----------
    sdf : function
        DESCRIPTION.
    xlims : list
        [xmin, xmax]
    ylims : list
        [ymin, ymax].
    zlims : list
        [zmin, zmax].
    h : numeric
        Element side length.
    mode : str
        Voxel trimming model
        notrim - will keep all voxels in the grid
        liberal - will keep all voxels with at least 1 node inside the negative region of the field (val <= 0)
        moderate - will keep all voxels with the centroid inside the negative region of the field (val <= 0)
        conservative - will keep only voxels that are entirely inside the negative region of the field (val <= 0)
        The default is liberal.
    Returns
    -------
    NodeCoords : list
        List of nodal coordinates for the voxel mesh
    NodeConn : list
        Nodal connectivity list for the voxel mesh 

    """        
    
    NodeCoords, NodeConn1 = primitives.Grid([xlims[0],xlims[1],ylims[0],ylims[1],zlims[0],zlims[1]],h,exact_h=True, meshobj=False)
    NodeVals = sdf(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])
    if mode != 'notrim':
        NodeConn = []
        for elem in NodeConn1:
            if mode == 'conservative':
                if all([NodeVals[node] <= 0 for node in elem]):
                    NodeConn.append(elem)
            elif mode == 'moderate':
                if np.mean([NodeVals[node] for node in elem]) <= 0:                   
                    NodeConn.append(elem)
            elif mode =='liberal':
                if any([NodeVals[node] <= 0 for node in elem]):
                    NodeConn.append(elem)
        NodeCoords,NodeConn,OriginalIds = converter.removeNodes(NodeCoords.tolist(),NodeConn)
        NodeVals = NodeVals[OriginalIds]
    else:   
        NodeConn = NodeConn1
    
    return NodeCoords, NodeConn, NodeVals

def SurfaceMesh(sdf,xlims,ylims,zlims,h,threshold=0, flip=False, mcmethod='33'):
    # NodeCoords, NodeConn, NodeVals = VoxelMesh(sdf,xlims,ylims,zlims,h,mode='notrim',reinitialize=False)
    # SurfCoords, SurfConn = contour.MarchingCubes(NodeCoords, NodeConn, NodeVals, threshold=threshold, flip=flip, method=mcmethod)

    if isinstance(h, (list, tuple, np.ndarray)):
        hx = h[0];hy = h[1]; hz = h[2]
    else:
        hx = h; hy = h; hz = h
    xs = np.arange(xlims[0],xlims[1]+hx,hx)
    ys = np.arange(ylims[0],ylims[1]+hy,hy)
    zs = np.arange(zlims[0],zlims[1]+hz,hz)

    X,Y,Z = np.meshgrid(xs,ys,zs)
    F = sdf(X,Y,Z)
    SurfCoords, SurfConn = contour.MarchingCubesImage(F, h=(hx, hy, hz), threshold=threshold, flip=False, method='original', interpolation='cubic',VertexValues=True)
    SurfCoords[:,0] += xlims[0]
    SurfCoords[:,1] += ylims[0]
    SurfCoords[:,2] += zlims[0]
    return SurfCoords, SurfConn

def grid2fun(VoxelCoords,VoxelConn,Vals,method='linear',fill_value=None):
    """
    grid2fun converts a voxel grid mesh (as made by VoxelMesh(mode='notrim') 
    or converter.makeGrid) into a function that can be evaluated at any point
    within the bounds of the grid

    Parameters
    ----------
    VoxelCoords : List of Lists
        List of nodal coordinates for the voxel mesh
    VoxelConn : List of Lists
       Nodal connectivity list for the voxel mesh.
    Vals : list
        List of values at each node or at each element.

    Returns
    -------
    fun : function
        Interpolation function, takes arguments (x,y,z), to return an
        evaluation of the function at the specified point.

    """
    
    if len(Vals) == len(VoxelCoords):
        Coords = np.asarray(VoxelCoords)
    elif len(Vals) == len(VoxelConn):
        Coords = utils.Centroids(VoxelCoords,VoxelConn)
    else:
        raise Exception('Vals must be the same length as either VoxelCoords or VoxelConn')
    # VoxelCoords = np.array(VoxelCoords)
    X = np.unique(Coords[:,0])
    Y = np.unique(Coords[:,1])
    Z = np.unique(Coords[:,2])
    
    points = (X,Y,Z)
    V = np.reshape(Vals,[len(X),len(Y),len(Z)])
    
    V[np.isnan(V)] = np.array([np.nan]).astype(int)[0]
    fun = lambda x,y,z : interpolate.RegularGridInterpolator(points,V,method=method,bounds_error=False,fill_value=None)(np.vstack([x,y,z]).T)
    
    return fun

def grid2grad(VoxelCoords,VoxelConn,NodeVals,method='linear'):
    """
    grid2grad converts a voxel grid mesh (as made by VoxelMesh(mode='notrim') 
    or converter.makeGrid) into a function that can be evaluated at any point
    within the bounds of the grid to return the gradient of the function

    Parameters
    ----------
    VoxelCoords : List of Lists
        List of nodal coordinates for the voxel mesh
    VoxelConn : List of Lists
       Nodal connectivity list for the voxel mesh.
    NodeVals : list
        List of values at each node.

    Returns
    -------
    frad : function
        Interpolation function, takes arguments (x,y,z), to return an
        evaluation of the function gradient at the specified point.

    """
    
    VoxelCoords = np.array(VoxelCoords)
    X = np.unique(VoxelCoords[:,0])
    Y = np.unique(VoxelCoords[:,1])
    Z = np.unique(VoxelCoords[:,2])
    
    points = (X,Y,Z)
    V = np.reshape(NodeVals,[len(X),len(Y),len(Z)])
    # Assumes (and requires) that all voxels are cubic and the same size
    VoxelSize = abs(sum(VoxelCoords[VoxelConn[0][0]] - VoxelCoords[VoxelConn[0][1]]))
    G = np.gradient(V,VoxelSize)
    grad = lambda x,y,z : np.vstack([interpolate.interpn(points,G[i],np.vstack([x,y,z]).T,bounds_error=False,method=method) for i in range(len(G))]).T
    
    return grad

def mesh2sdf(M,VoxelCoords,VoxelConn,method='nodes+centroids'):
    """
    mesh2sdf Generates a signed distance field for a mesh

    Parameters
    ----------
    M : mesh.mesh
        Mesh object that will be used to define the distance field.
    VoxelCoords : list
        List of nodal coordinates for the voxel mesh on which the distance
        field will be evaluated.
    VoxelConn : list
        Nodal connectivity list for the voxel mesh on which the distance field
        will be evaluated.
    method : str
        Method to be used 
        nodes 
        nodes+centroids
        centroids

    Returns
    -------
    NodeVals : list
        List of signed distance values evaluated at each node in the voxel grid.

    """
    
    if method == 'nodes':
        Normals = np.asarray(M.NodeNormals)
        SurfNodes = set(np.unique(M.SurfConn))
        Coords = np.array([n if i in SurfNodes else [10**32,10**32,10**32] for i,n in enumerate(M.NodeCoords)])
    elif method == 'centroids':
        Normals = np.asarray(M.ElemNormals)
        NodeCoords = np.array(M.NodeCoords)
        Coords = utils.Centroids(M.NodeCoords,M.NodeConn) #np.array([np.mean(NodeCoords[elem],axis=0) for elem in M.SurfConn])
    elif method == 'nodes+centroids':
        Normals = np.array(list(M.NodeNormals) + list(M.ElemNormals))
        NodeCoords = np.array(M.NodeCoords)
        SurfNodes = set(np.unique(M.SurfConn))
        Coords = np.append([n if i in SurfNodes else [10**32,10**32,10**32] for i,n in enumerate(M.NodeCoords)], utils.Centroids(M.NodeCoords,M.NodeConn),axis=0)
    else:
        raise Exception('Invalid method - use "nodes", "centroids", or "nodes+centroids"')
    
    tree = KDTree(Coords, leaf_size=2)  
    Out = tree.query(VoxelCoords,1)
    ds = Out[0].flatten()
    cs = Out[1].flatten()
    rs = VoxelCoords - Coords[cs]
    signs = np.sign(np.sum(rs*Normals[cs,:],axis=1))# [np.sign(np.dot(rs[i],Normals[cs[i]])) for i in range(len(ds))]
    NodeVals = signs*ds
    
    return NodeVals

def mesh2udf(M,VoxelCoords,VoxelConn):
    """
    mesh2udf Generates an unsigned distance field for a mesh

    Parameters
    ----------
    M : mesh.mesh
        Mesh object that will be used to define the distance field.
    VoxelCoords : list
        List of nodal coordinates for the voxel mesh on which the distance
        field will be evaluated.
    VoxelConn : list
        Nodal connectivity list for the voxel mesh on which the distance field
        will be evaluated.

    Returns
    -------
    NodeVals : list
        List of signed distance values evaluated at each node in the voxel grid.

    """
    Coords = np.asarray(M.NodeCoords)

    tree = KDTree(Coords, leaf_size=2)  
    Out = tree.query(VoxelCoords,1)
    NodeVals = Out[0].flatten()
    
    return NodeVals
