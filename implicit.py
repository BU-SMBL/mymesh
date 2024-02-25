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

from . import utils, converter, contour, quality, improvement, TetGen, rays, octree, mesh, primitives

# implicit function primitives
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

# SDF Operators
def offset(fval,offset):
    return fval-offset

def union(fval1,fval2):
    return rMin(fval1,fval2)

def diff(fval1,fval2):
    # This has maybe been wrong for a long time changing min to max
    # return rMin(fval1,-fval2)
    return rMax(fval1,-fval2)

def diff_old(fval1, fval2):
    return rMin(fval1,-fval2)
    
def intersection(fval1,fval2):
    return rMax(fval1,fval2)

def unionf(f1,f2):
    return lambda x,y,z : rMin(f1(x,y,z),f2(x,y,z))

def difff(f1,f2):
    return lambda x,y,z : rMax(f1(x,y,z),-f2(x,y,z))

def intersectionf(f1,f2):
    return lambda x,y,z : rMax(f1(x,y,z),f2(x,y,z))

def rMax(a,b,alpha=0,m=0,p=2):
    # R-Function version of max(a,b) to yield a smoothly differentiable max - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b+(np.maximum(a**p+b**p - 2*alpha*a*b,0))**(1/p)) * (a**2 + b**2)**(m/2)

def rMin(a,b,alpha=0,m=0,p=2):
    # R-Function version of min(a,b) to yield a smoothly differentiable min - R0
    # Implicit Functions With Guaranteed Differential Properties - Shapiro & Tsukanov
    return 1/(1+alpha)*(a+b-(np.maximum(a**p+b**p - 2*alpha*a*b,0))**(1/p)) * (a**2 + b**2)**(m/2)

def sMax(a,b,p):
    # p-norm smooth maximum
    return np.linalg.norm([a,b],ord=p,axis=0)

def sMin(a,b,k):
    # polynomial smooth minimum
    h = np.max([k-abs(a-b), np.zeros(np.shape(a))],axis=0)/k
    return np.min([a,b],axis=0)-h**2*k*(1/4)

def sUnion(fval1,fval2,k):
    return sMin(fval1,fval2,k)

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
    
    NodeCoords, NodeConn1 = primitives.Grid([xlims[0],xlims[1],ylims[0],ylims[1],zlims[0],zlims[1]],h,exact_h=True)
    NodeVals = sdf(NodeCoords[:,0], NodeCoords[:,1], NodeCoords[:,2])
    if np.min(NodeVals) > 0:
        return [], []
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

def FastMarchingMethod(VoxelCoords, VoxelConn, NodeVals):
    """
    FastMarchingMethod based on J.A. Sethian. A Fast Marching Level Set Method
    for Monotonically Advancing Fronts, Proc. Natl. Acad. Sci., 93, 4, 
    pp.1591--1595, 1996

    Parameters
    ----------
    VoxelCoords : list
        List of nodal coordinates for the voxel mesh.
    VoxelConn : list
        Nodal connectivity list for the voxel mesh.
    NodeVals : list
        List of value at each node.

    Returns
    -------
    T : list
        Lists of reinitialized node values.

    """
    warnings.warn('FastMarchingMethod is not fully functional.')

    # 3D
    N = 3
    # For now this is only for obtaining a signed distance function, so F = 1 everywhere
    F = 1
    NodeVals = np.array(NodeVals)
    # Get Neighbors
    if len(VoxelConn[0]) == 4:
        ElemType = 'quad'
    else:
        ElemType = 'hex'
    NodeNeighbors = utils.getNodeNeighbors(VoxelCoords, VoxelConn, ElemType=ElemType)
    xNeighbors = [[n for n in NodeNeighbors[i] if (VoxelCoords[n][1] == VoxelCoords[i][1]) and (VoxelCoords[n][2] == VoxelCoords[i][2])] for i in range(len(NodeNeighbors))]
    yNeighbors = [[n for n in NodeNeighbors[i] if (VoxelCoords[n][0] == VoxelCoords[i][0]) and (VoxelCoords[n][2] == VoxelCoords[i][2])] for i in range(len(NodeNeighbors))]
    zNeighbors = [[n for n in NodeNeighbors[i] if (VoxelCoords[n][0] == VoxelCoords[i][0]) and (VoxelCoords[n][1] == VoxelCoords[i][1])] for i in range(len(NodeNeighbors))]
    # Assumes (and requires) that all voxels are the same size
    h = abs(sum(np.array(VoxelCoords[VoxelConn[0][0]]) - np.array(VoxelCoords[VoxelConn[0][1]])))
    # Initialize Labels - Accepted if on the surface, Narrow Band if an adjacent node has a different sign (i.e. cross the surface), otherwise Far
    Accepted = set([i for i,v in enumerate(NodeVals) if v == 0])
    Narrow = [i for i,v in enumerate(NodeVals) if any(np.sign(NodeVals[NodeNeighbors[i]]) != np.sign(v)) and i not in Accepted]
    NarrowVals = []
    for i in Narrow:
        crosses = []
        for n in xNeighbors[i]:
            if np.sign(NodeVals[i]) != np.sign(NodeVals[n]):
                crosses.append(np.sign(NodeVals[i])*np.abs((0-NodeVals[i])*(VoxelCoords[n][0]-VoxelCoords[i][0])/(NodeVals[n]-NodeVals[i])))
        for n in yNeighbors[i]:
            if np.sign(NodeVals[i]) != np.sign(NodeVals[n]):
                crosses.append(np.sign(NodeVals[i])*np.abs((0-NodeVals[i])*(VoxelCoords[n][1]-VoxelCoords[i][1])/(NodeVals[n]-NodeVals[i])))
        for n in zNeighbors[i]:
            if np.sign(NodeVals[i]) != np.sign(NodeVals[n]):
                crosses.append(np.sign(NodeVals[i])*np.abs((0-NodeVals[i])*(VoxelCoords[n][2]-VoxelCoords[i][2])/(NodeVals[n]-NodeVals[i])))
        # NarrowVals.append(np.mean(crosses))
        NarrowVals.append(min(crosses))
    Far = set(range(len(NodeVals))).difference(Accepted.union(set(Narrow)))
    # Initialize Values (inf for Far, 0 for accepted)
    infty = 1e16 * max(NodeVals)
    T = infty*np.ones(len(NodeVals))
    for i in range(len(NarrowVals)):
        T[Narrow[i]] = NarrowVals[i]
    for i in Accepted:
        T[i] = 0
    
    Nar = sorted([t for i,t in enumerate(zip(NarrowVals,Narrow))], key=lambda x: x[0])
    while len(Far) + len(Nar) > 0:
        if len(Nar) > 0:
            pt = Nar[0][1]
        else:
            n = Far.pop()
            Nar.append((T[n],n))
        Accepted.add(pt)
        Nar.pop(0)
        for n in NodeNeighbors[pt]:
            if n in Far:
                Far.remove(n)
                Nar.insert(bisect.bisect_left(Nar, (T[n],n)), (T[n],n))
            if n not in Accepted:
                # Eikonal Update:
                Tx = min([T[x] for x in xNeighbors[n]]+[0])
                Ty = min([T[y] for y in yNeighbors[n]]+[0])
                Tz = min([T[z] for z in zNeighbors[n]]+[0])
                
                discriminant = sum([Tx,Ty,Tz])**2 - N*(sum([Tx**2,Ty**2,Tz**2]) - h**2/F**2)
                if discriminant > 0:
                    t = 1/N * sum([Tx,Ty,Tz]) + 1/N * np.sqrt(discriminant)
                else:
                    t = h/F + min([Tx,Ty,Tz])
                                
                Nar.pop(bisect.bisect_left(Nar, (T[n],n)))
                if t < T[n]: T[n] = t
                Nar.insert(bisect.bisect_left(Nar, (T[n],n)), (T[n],n))        
    T = [-1*t if np.sign(t) != np.sign(NodeVals[i]) else t for i,t in enumerate(T)]
    return T

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
    try:
        from sklearn.neighbors import KDTree
    except:
        raise ImportError('mesh2sdf requires scikit-learn (sklearn). Install with: pip install scikit-learn')
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
        Coords = np.append([n if i in SurfNodes else [10**32,10**32,10**32] for i,n in enumerate(M.NodeCoords)], utils.Centroids(M.NodeCoords,M.SurfConn),axis=0).astype(float)
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
    try:
        from sklearn.neighbors import KDTree
    except:
        raise ImportError('mesh2udf requires scikit-learn (sklearn). Install with: pip install scikit-learn')
    Coords = np.asarray(M.NodeCoords)

    tree = KDTree(Coords, leaf_size=2)  
    Out = tree.query(VoxelCoords,1)
    NodeVals = Out[0].flatten()
    
    return NodeVals

def DoubleDualResampling(sdf,NodeCoords,NodeConn,DualCoords,DualConn,eps=1e-3,c=2):
    warnings.warn('DoubleDualResampling is not fully functional and may be unstable.')

    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    DualCoords,DualConn,gradP = DualMeshOptimization(sdf,NodeCoords,NodeConn,DualCoords,DualConn,eps=eps,return_grad=True)
    DualNeighbors,ElemConn = utils.getNodeNeighbors(DualCoords,DualConn,ElemType='polygon')
    NewNodeCoords = [[] for i in range(len(NodeCoords))]
    gradPnorms = [np.linalg.norm(gradP[i]) for i in range(len(gradP))]
    Normals = [gradP[j]/gradPnorms[j] if gradPnorms[j] > 0 else utils.CalcFaceNormal(DualCoords,[DualConn[ElemConn[j][0]]])[0] for j in range(len(DualCoords))]
    for i in range(len(NodeCoords)):
        Ps = DualConn[i]
        # Ns = [gradP[j]/gradPnorms[j] if gradPnorms[j] > 0 else utils.CalcFaceNormal(DualCoords,[Ps])[0] for j in Ps]
        ks = []
        for j,Pj in enumerate(Ps):
            NeighborPs = DualNeighbors[Pj][:3]
            # NNPs =[gradP[k]/gradPnorms[k] for k in NeighborPs]
            # ks.append(sum([np.arccos(np.dot(Ns[j],NNPs[k]))/(np.linalg.norm(DualCoords[Pj])*np.linalg.norm(DualCoords[NeighborPs[k]]))  for k in range(len(NNPs))]))
            
            ks.append(sum([np.arccos(min(np.dot(Normals[Pj],Normals[NeighborPs[k]]),1))/np.linalg.norm(np.subtract(DualCoords[Pj],DualCoords[NeighborPs[k]])) for k in range(len(NeighborPs))]))
            if np.isnan(ks[-1]):
                print('merp')
        
        weights = [1+c*ki for ki in ks]
        
        NewNodeCoords[i] = sum([np.multiply(weights[j],DualCoords[Ps[j]]) for j in range(len(Ps))])/sum(weights).tolist()
    return NewNodeCoords, NodeConn
    
def DualMeshOptimization(sdf,NodeCoords,NodeConn,DualCoords,DualConn,eps=1e-3,return_grad=False):
    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    warnings.warn('DualMeshOptimization is not fully functional and may be unstable.')
    
    def GradF(q,h):
        g = [-1,0,1]
        X = np.array([q[0]+h*x for x in g for y in g for z in g])
        Y = np.array([q[1]+h*y for x in g for y in g for z in g])
        Z = np.array([q[2]+h*z for x in g for y in g for z in g])
        F = sdf(X,Y,Z).reshape([3,3,3])
        dF = np.gradient(F,h)
        dFq = [dF[0][1,1,1],dF[1][1,1,1],dF[2][1,1,1]]
        return dFq
    def bisection(sdf, a, b, fa, fb, tol=eps):
        assert (fa < 0 and fb > 0) or (fa > 0 and fb < 0), 'Invalid bounds for bisection'
        
        thinking = True
        while thinking:
            c = np.mean([a,b],axis=0)
            fc = sdf(*c)
            # if fc == 0 or (np.linalg.norm(b-a)/2) < tol:
                # merp = 'meep'
            if abs(fc) < tol:
                thinking = False
            else:
                if np.sign(fc) == np.sign(fa):
                    a = c
                    fa = fc
                else:
                    b = c
                    fb = fc                
        return c
    def secant(sdf, a, b, fa, fb, tol=eps):
        assert (fa < 0 and fb > 0) or (fa > 0 and fb < 0), 'Invalid bounds for secant method'
        origA, origB, origFa, origFb = a, b, fa, fb
        thinking = True
        k = 0
        while thinking:
            k += 1
            c = np.subtract(b,fb*(np.subtract(b,a))/(fb-fa))
            fc = sdf(*c)
            if fc == 0 or abs(fc) < tol:
                thinking = False
            else:
                a,b = b,c
                fa,fb = fb,fc
            if k > 50 or fa == fb:
                thinking = False
                c = bisection(sdf, origA, origB, origFa, origFb, tol=tol)
        
        if not ((a[0] <= c[0] <= b[0] or a[0] >= c[0] >= b[0]) and (a[1] <= c[1] <= b[1] or a[1] >= c[1] >= b[1] ) and (a[2] <= c[2] <= b[2] or a[2] >= c[2] >= b[2])):
            c = bisection(sdf, origA, origB, origFa, origFb, tol=tol)
            
        return c
    ArrayCoords = np.array(NodeCoords)
    
    # _,ElemConn = utils.getNodeNeighbors(NodeCoords,NodeConn)
    # DualCoords,DualConn = converter.surf2dual(ArrayCoords,NodeConn,ElemConn=ElemConn)
    
    # Optimize dual mesh coordinates     
    gradP = [[] for i in range(len(DualCoords))]
    for c,P in enumerate(DualCoords):
        pts = ArrayCoords[NodeConn[c]]
        edgelengths = [np.linalg.norm(pts[1]-pts[0]),np.linalg.norm(pts[2]-pts[1]),np.linalg.norm(pts[0]-pts[2])]
        e = np.mean(edgelengths)
        lamb = e/2
        fP = sdf(*P)
        if abs(fP) < eps:
            if return_grad:
                gradP[c] = GradF(P,lamb/1000)
                # gradP[c] = gradF(*P)[0]
            continue
        
        Q = P
        fQ = fP

        it = 0
        thinking = True
        while thinking:
            dfQ = GradF(Q,lamb/1000)
            # dfQ = gradF(*Q)[0]
            d = -np.multiply(dfQ,fQ)
            d = d/np.linalg.norm(d)
            R = Q + lamb*d
            fR = sdf(*R)
            if fQ*fR < 0:
                P2 = bisection(sdf, Q, R, fQ, fR)
                thinking = False
            else: 
                Q = R
                fQ = sdf(*Q)
                it += 1
                if it == 3:
                    lamb = lamb/2
                if it > 500:
                    thinking = False
                    P2 = P
                    # raise Exception("Too many iterations - This probably shouldn't happen")
        #if np.linalg.norm(np.subtract(P2,P)) < e:

        S = P-2*P2
        fS = sdf(*S)
        if fP*fS < 0:
            P3 = bisection(sdf, P, S, fP, fS)
            if np.linalg.norm(np.subtract(P,P2)) < np.linalg.norm(np.subtract(P,P3)):
                P = P2
            else:
                P = P3
        else:
            P = P2
        # P = P2
        DualCoords[c] = P
        if return_grad: gradP[c] = GradF(P,lamb/10)
        # if return_grad: gradP[c] = gradF(*P)[0]
        
    if return_grad:
        return DualCoords, DualConn, gradP
    else:
        return DualCoords, DualConn
    
def AdaptiveSubdivision(sdf,NodeCoords,NodeConn,threshold=1e-3):
    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    def gradF(q,h=1e-6):
        if type(q) is list: q = np.array(q)
        if len(q.shape)==1: q = np.array([q])
        gradx = (sdf(q[:,0]+h/2,q[:,1],q[:,2]) - sdf(q[:,0]-h/2,q[:,1],q[:,2]))/h
        grady = (sdf(q[:,0],q[:,1]+h/2,q[:,2]) - sdf(q[:,0],q[:,1]-h/2,q[:,2]))/h
        gradz = (sdf(q[:,0],q[:,1],q[:,2]+h/2) - sdf(q[:,0],q[:,1],q[:,2]-h/2))/h
        gradf = np.vstack((gradx,grady,gradz)).T
        if len(gradf) == 1: gradf = gradf[0]
        return gradf
    
    NewNodeCoords = copy.copy(NodeCoords)
    NewNodeConn = copy.copy(NodeConn)
    ElemNeighbors = utils.getElemNeighbors(NodeCoords, NodeConn, mode='edge')

    ###
    Points = np.array(NodeCoords)[np.array(NodeConn)]
    cross = np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0])
    norm = np.linalg.norm(cross,axis=1)
    ElemNormals = cross/norm[:,None]
    Area = norm/2
    splitCentroids = np.swapaxes(np.array([np.mean(Points,axis=1),
                          np.mean([Points[:,0], np.mean([Points[:,0], Points[:,1]],axis=0), np.mean([Points[:,0], Points[:,2]],axis=0)],axis=0),
                          np.mean([Points[:,1], np.mean([Points[:,0], Points[:,1]],axis=0), np.mean([Points[:,1], Points[:,2]],axis=0)],axis=0),
                          np.mean([Points[:,2], np.mean([Points[:,2], Points[:,1]],axis=0), np.mean([Points[:,0], Points[:,2]],axis=0)],axis=0),
                          ]),0,1)
    splitCentroids2 = splitCentroids.reshape((len(splitCentroids)*4,3))
    gradFCi = gradF(splitCentroids2)
    mCi = gradFCi/np.linalg.norm(gradFCi,axis=1)[:,None]
    mCi2 = mCi.reshape(splitCentroids.shape)
    en = np.array([Area[i]*sum(1-np.abs(np.dot(ElemNormals[i],mCi2[i].T))) for i in range(len(NodeConn))])
    for i,elem in enumerate(NodeConn):
        if en[i] > threshold:
            id01 = len(NewNodeCoords)
            NewNodeCoords.append(np.mean([NewNodeCoords[elem[0]],NewNodeCoords[elem[1]]],axis=0).tolist())
            id12 = len(NewNodeCoords)
            NewNodeCoords.append(np.mean([NewNodeCoords[elem[1]],NewNodeCoords[elem[2]]],axis=0).tolist())
            id20 = len(NewNodeCoords)
            NewNodeCoords.append(np.mean([NewNodeCoords[elem[2]],NewNodeCoords[elem[0]]],axis=0).tolist())
            NewNodeConn[i] = [
                [elem[0],id01,id20],
                [id01,elem[1],id12],
                [id20,id12,elem[2]],
                [id01,id12,id20]
                ]

    # Check for neighbors of split elements
    thinking = True
    mode = '1-4'
    while thinking:
        changes = 0
        for i,elem in enumerate(NewNodeConn):
            if type(elem[0]) is list:
                # Already subdivided
                continue
            nSplitNeighbors = 0
            SplitNeighbors = []
            for n in ElemNeighbors[i]:
                if type(NewNodeConn[n][0]) is list and len(NewNodeConn[n]) > 2: 
                    nSplitNeighbors += 1
                    SplitNeighbors.append(n)
            if mode == '1-4' and nSplitNeighbors > 1:
                changes += 1
                id01 = len(NewNodeCoords)
                NewNodeCoords.append(np.mean([NewNodeCoords[elem[0]],NewNodeCoords[elem[1]]],axis=0).tolist())
                id12 = len(NewNodeCoords)
                NewNodeCoords.append(np.mean([NewNodeCoords[elem[1]],NewNodeCoords[elem[2]]],axis=0).tolist())
                id20 = len(NewNodeCoords)
                NewNodeCoords.append(np.mean([NewNodeCoords[elem[2]],NewNodeCoords[elem[0]]],axis=0).tolist())
                NewNodeConn[i] = [
                    [elem[0],id01,id20],
                    [id01,elem[1],id12],
                    [id20,id12,elem[2]],
                    [id01,id12,id20]
                    ]
                
            elif mode == '1-2' and nSplitNeighbors == 1:
                changes += 1
                if elem[0] in NodeConn[SplitNeighbors[0]] and elem[1] in NodeConn[SplitNeighbors[0]]:
                    idx = len(NewNodeCoords)
                    NewNodeCoords.append(np.mean([NewNodeCoords[elem[0]],NewNodeCoords[elem[1]]],axis=0).tolist())
                    NewNodeConn[i] = [
                        [elem[0],idx,elem[2]],
                        [idx,elem[1],elem[2]]
                        ]
                elif elem[1] in NodeConn[SplitNeighbors[0]] and elem[2] in NodeConn[SplitNeighbors[0]]:
                    idx = len(NewNodeCoords)
                    NewNodeCoords.append(np.mean([NewNodeCoords[elem[1]],NewNodeCoords[elem[2]]],axis=0).tolist())
                    NewNodeConn[i] = [
                        [elem[1],idx,elem[0]],
                        [idx,elem[2],elem[0]]
                        ]
                else:
                    idx = len(NewNodeCoords)
                    NewNodeCoords.append(np.mean([NewNodeCoords[elem[2]],NewNodeCoords[elem[0]]],axis=0).tolist())
                    NewNodeConn[i] = [
                        [elem[0],elem[1],idx],
                        [elem[1],elem[2],idx]
                        ]
        if mode == '1-4' and changes == 0:
            # After all necessary 1-4 splits are completed, perform 1-2 splits
            mode = '1-2'
        elif mode == '1-2' and changes == 0:
            thinking = False
            
    NewNodeConn = [elem for elem in NewNodeConn if (type(elem[0]) != list)] + [e for elem in NewNodeConn if (type(elem[0]) == list) for e in elem]
    NewNodeCoords,NewNodeConn,_ = utils.DeleteDuplicateNodes(NewNodeCoords,NewNodeConn)
            
    return NewNodeCoords,NewNodeConn
     
def DualPrimalOptimization(sdf,NodeCoords,NodeConn,eps=1e-3,nIter=2):
    # Ohtake, Y., and Belyaev, A. G. (March 26, 2003). "Dual-Primal Mesh Optimization for Polygonized Implicit Surfaces With Sharp Features ." ASME. J. Comput. Inf. Sci. Eng. December 2002; 2(4): 277–284. https://doi.org/10.1115/1.1559153
    warnings.warn('DualPrimalOptimization is not fully functional and may be unstable.')
       
    def PrimalMeshOptimization(DualCoords,DualConn,gradP,tau=10**3):
        ArrayCoords = np.zeros([len(DualConn),3])
        DualCoords = np.array(DualCoords)
        DualNeighbors,ElemConn = utils.getNodeNeighbors(DualCoords,DualConn,ElemType='polygon')
        centroids = utils.Centroids(DualCoords,DualConn)
        gradPnorms = [np.linalg.norm(gradP[i]) for i in range(len(gradP))]
        TransCoords = copy.copy(DualCoords)
        for j,Pis in enumerate(DualConn):
            
            # Transfrom Coordinates to local system centered on the centroid
            TransCoords[:,0] -= centroids[j][0]
            TransCoords[:,1] -= centroids[j][1]
            TransCoords[:,2] -= centroids[j][2]
            
            # Normal vector TODO: gradP[i] could = 0 at sharp features, in this case, need to use something else (maybe the element normal of the primal element corresponding to the dual node)
            # Ns = [np.divide(gradP[i],gradPnorms[i]) for i in Pis]
            Ns = [np.divide(gradP[i],gradPnorms[i]) if gradPnorms[i] > 0 else utils.CalcFaceNormal(DualCoords,[DualConn[ElemConn[i][0]]])[0] for i in Pis]
            
            r = np.linalg.norm(centroids[j]-DualCoords[Pis[0]])*2
            A = np.diag([sum([N[0]**2 for i,N in enumerate(Ns)]), 
                         sum([N[1]**2 for i,N in enumerate(Ns)]),
                         sum([N[2]**2 for i,N in enumerate(Ns)])])
            
            b = [sum([N[0]**2*TransCoords[Pis[i]][0] for i,N in enumerate(Ns)]), 
                 sum([N[1]**2*TransCoords[Pis[i]][1] for i,N in enumerate(Ns)]),
                 sum([N[2]**2*TransCoords[Pis[i]][2] for i,N in enumerate(Ns)])]
            x = np.linalg.lstsq(A,b,rcond=1/tau)[0]
            ArrayCoords[j] = x + centroids[j]
            # Reset TransCoords
            TransCoords[:,0] += centroids[j][0]
            TransCoords[:,1] += centroids[j][1]
            TransCoords[:,2] += centroids[j][2]
        return ArrayCoords.tolist()
    
    OptCoords = copy.copy(NodeCoords)
    OptConn = copy.copy(NodeConn)
    k = 0
    tau = 10**3
    for it in range(nIter):
        DualCoords, DualConn = converter.surf2dual(OptCoords,OptConn,sort='ccw')
        writeVTK('{:d}_Dual.vtk'.format(k),DualCoords,DualConn)
        OptCoords,_ = DoubleDualResampling(sdf,OptCoords,OptConn,DualCoords,DualConn)
        writeVTK('{:d}_PrimalResampled.vtk'.format(k),OptCoords,OptConn)
        DualCoords = utils.Centroids(OptCoords,OptConn)
        writeVTK('{:d}_Dual2.vtk'.format(k),DualCoords,DualConn)
        DualCoords, DualConn, gradP = DualMeshOptimization(sdf,OptCoords,OptConn,DualCoords,DualConn,eps=eps,return_grad=True) 
        writeVTK('{:d}_DualOpt.vtk'.format(k),DualCoords,DualConn)
        OptCoords = PrimalMeshOptimization(DualCoords,DualConn,gradP,tau=tau)
        writeVTK('{:d}_PrimalOpt.vtk'.format(k),OptCoords,OptConn)
        OptCoords,OptConn = AdaptiveSubdivision(sdf,OptCoords,OptConn)
        writeVTK('{:d}_PrimalOptSub.vtk'.format(k),OptCoords,OptConn)
        k += 1
        if k > 1 and tau > 10:
            tau = tau/10
    DualCoords, DualConn = converter.surf2dual(OptCoords,OptConn,sort='ccw')
    OptCoords,_ = DoubleDualResampling(sdf,OptCoords,OptConn,DualCoords,DualConn)
    writeVTK('{:d}_PrimalResampled.vtk'.format(k),OptCoords,OptConn)
    DualCoords, DualConn, gradP = DualMeshOptimization(sdf,OptCoords,OptConn,DualCoords,DualConn,eps=eps,return_grad=True) 
    writeVTK('{:d}_DualOpt.vtk'.format(k),DualCoords,DualConn)
    OptCoords = PrimalMeshOptimization(DualCoords,DualConn,gradP,tau=tau)
    writeVTK('{:d}_PrimalOpt.vtk'.format(k),OptCoords,OptConn)
    return OptCoords,OptConn

def SurfFlowOptimization(sdf,NodeCoords,NodeConn,h,ZRIter=50,NZRIter=50,NZIter=50,Subdivision=True,FixedNodes=set(), gradF=None):
    
    C = 0.1     # Positive Constant
    FreeNodes = list(set(range(len(NodeCoords))).difference(FixedNodes))
    if gradF is None:
        def gradF(q):
            hdiff = 1e-6    # Finite Diff Step Size
            if type(q) is list: q = np.array(q)
            if len(q.shape)==1: q = np.array([q])
            gradx = (sdf(q[:,0]+hdiff/2,q[:,1],q[:,2]) - sdf(q[:,0]-hdiff/2,q[:,1],q[:,2]))/hdiff
            grady = (sdf(q[:,0],q[:,1]+hdiff/2,q[:,2]) - sdf(q[:,0],q[:,1]-hdiff/2,q[:,2]))/hdiff
            gradz = (sdf(q[:,0],q[:,1],q[:,2]+hdiff/2) - sdf(q[:,0],q[:,1],q[:,2]-h/2))/hdiff
            gradf = np.vstack((gradx,grady,gradz)).T
            if len(gradf) == 1: gradf = gradf[0]
            return gradf
    def NFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        gradC = -gradF(Centroids)
        gradCnorm = np.linalg.norm(gradC,axis=1)
        m = np.divide(gradC,np.reshape(gradCnorm,(len(gradC),1)))
        
        # This is a slower but more straightforward version of what is done below
        # A = np.array([sum([Area[e] for e in ElemConn[i]]) for i in range(len(NodeCoords))])
        # tau = 1/(1000*max(A))
        # N1 = tau*np.array([1/sum(Area[T] for T in ElemConn[i]) * sum([Area[T]*np.dot((Centroids[T]-P),m[T])*m[T] for T in ElemConn[i]]) for i,P in enumerate(NodeCoords)])

        # Converting the ragged ElemConn array to a padded rectangular array (R) for significant speed improvements
        Area2 = np.append(Area,0)
        m2 = np.vstack([m,[0,0,0]])
        Centroids2 = np.vstack([Centroids,[0,0,0]])
        R = utils.PadRagged(ElemConn,fillval=-1)
        a = Area2[R]
        A = np.sum(a,axis=1)
        tau = tf*.75 # 1/(100*max(A))
        v = np.sum(m2[R]*(Centroids2[R] - NodeCoords[:,None,:]),axis=2)[:,:,None]*m2[R]
        C = np.sum(a[:,:,None]*v,axis=1)
        N = (tau/np.sum(Area2[R],axis=1))[:,None]*C
        return N
    def N2Flow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids):
        
        # Orthocenter coordinates: https://en.wikipedia.org/wiki/Triangle_center#Position_vectors
        tic = time.time()
        Points = NodeCoords[np.array(NodeConn)]
        a = np.linalg.norm(Points[:,1]-Points[:,2],axis=1)
        b = np.linalg.norm(Points[:,2]-Points[:,0],axis=1)
        c = np.linalg.norm(Points[:,1]-Points[:,0],axis=1)
        wA = a**4 - (b**2 - c**2)**2
        wB = b**4 - (c**2 - a**2)**2
        wC = c**4 - (a**2 - b**2)**2
        # Orthocenters
        H = (wA[:,None]*Points[:,0] + wB[:,None]*Points[:,1] + wC[:,None]*Points[:,2])/(wA + wB + wC)[:,None]
        H2 = np.vstack([H,[0,0,0]])
        # 
        lens = [len(e) for e in ElemConn]
        maxlens = max(lens)
        R = utils.PadRagged(ElemConn,fillval=-1)
        Mask0 = (R>=0).astype(int)
        Masknan = Mask0.astype(float)
        Masknan[Mask0 == 0] = np.nan
        
        PH = (H2[R] - NodeCoords[:,None,:])*Mask0[:,:,None]
        PHnorm = np.linalg.norm(PH,axis=2)
        e = PH/PHnorm[:,:,None]

        # For each point, gives the node connectivity of each incident element
        IncidentNodes = np.array(NodeConn)[R]*Masknan[:,:,None]
        ## TODO: This needs a speedup
        OppositeEdges = (((np.array([[np.delete(x,x==i) if i in x else [np.nan,np.nan] for x in IncidentNodes[i]] for i in range(len(IncidentNodes))])).astype(int)+1)*Mask0[:,:,None]-1)
        ##
        OppositeLength = np.linalg.norm(NodeCoords[OppositeEdges[:,:,0]] - NodeCoords[OppositeEdges[:,:,1]],axis=2)

        TriAntiGradient = e*OppositeLength[:,:,None]/2
        PointAntiGradient = np.nansum(TriAntiGradient,axis=1)
        degree = np.array([len(E) for E in ElemConn])
        N = 1/(5*degree[:,None]) * PointAntiGradient
        print(time.time()-tic)
        return N
    def ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        fP = sdf(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
        gradP = gradF(NodeCoords)
        # A = np.array([sum([Area[T] for T in ElemConn[i]]) for i in range(len(NodeCoords))])
        Area2 = np.append(Area,0)
        R = utils.PadRagged(ElemConn,fillval=-1)
        A = np.sum(Area2[R],axis=1)

        # tau = 1/(500*max(A))
        # Z = np.divide(-2*(tau*A)[:,None]*(fP[:,None]*gradP),np.linalg.norm(fP[:,None]*gradP,axis=1)[:,None],where=(fP!=0)[:,None])
        # tau = tf*1/(100*max(A*np.linalg.norm(fP[:,None]*gradP,axis=1)))
        tau = tf*h/(100*max(np.linalg.norm(fP[:,None]*gradP,axis=1)))
        Z = -2*tau*A[:,None]*fP[:,None]*gradP
        return Z
    def Z2Flow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids):
        fC = sdf(Centroids[:,0],Centroids[:,1],Centroids[:,2])
        gradC = -gradF(Centroids)
        Area2 = np.append(Area,0)
        fC = np.append(fC,0)
        gradC = np.vstack([gradC,[0,0,0]])
        R = utils.PadRagged(ElemConn,fillval=-1)
        A = np.sum(Area2[R],axis=1)
        tau = 1/(100*max(A))
        Z = 2*tau*np.sum(Area2[R][:,:,None]*gradC[R]*fC[R][:,:,None],axis=1)/3
        return Z
    def RFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids):
        ### Old slow version ###
        #     U = [1/len(N)*sum([np.subtract(NodeCoords[n],NodeCoords[i]) for n in N]) for i,N in enumerate(NodeNeighbors)]
        #     R = C*np.array([U[i] - np.dot(U[i],NodeNormals[i])*NodeNormals[i] for i in range(len(NodeCoords))])
        ###
        lens = np.array([len(n) for n in NodeNeighbors])
        r = utils.PadRagged(NodeNeighbors,fillval=-1)
        ArrayCoords = np.vstack([NodeCoords,[np.nan,np.nan,np.nan]])
        Q = ArrayCoords[r]
        U = (1/lens)[:,None] * np.nansum(Q - ArrayCoords[:-1,None,:],axis=1)
        R = C*(U - np.sum(U*NodeNormals,axis=1)[:,None]*NodeNormals)
        return R
    def NZRFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        N = NFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        Z = ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        R = RFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids)
        NZR = N + Z + R
        return NZR
    def ZRFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=1):
        Z = ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        R = RFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids)
        ZR = Z + R
        return ZR
    def NZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids, tf=1):
        N = NFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        Z = ZFlow(NodeCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids,tf=tf)
        NZ = N + Z
        return NZ
    def Flip(NodeCoords, NodeConn, ElemNormals, ElemNeighbors, Area, Centroids,threshold=1e-4):
        NodeCoords = np.array(NodeCoords)
        NewConn = copy.copy(NodeConn)
        gradC = gradF(Centroids)
        gradCnorm = np.linalg.norm(gradC,axis=1)
        m = np.divide(gradC,np.reshape(gradCnorm,(len(gradC),1)))
        NormalError = Area*np.array([(1-np.dot(ElemNormals[T],m[T])) for T in range(len(ElemNormals))])
        todo = np.where(NormalError > threshold)[0]
        for i in todo:
            restart = True
            while restart:
                for j in ElemNeighbors[i]:
                    tic = time.time()
                    if len(set(ElemNeighbors[i]).intersection(ElemNeighbors[j])) > 0:
                        # This condition checks if the flip will be legal
                        continue

                    Newi,Newj = improvement.FlipEdge(NodeCoords,NewConn,i,j)
                    [Ci,Cj] = utils.Centroids(NodeCoords,np.array([Newi,Newj]))
                    gradC = gradF(np.vstack([Ci,Cj]))
                    gradCnorm = np.linalg.norm(gradC,axis=1)
                    mi = gradC[0]/gradCnorm[0]
                    mj = gradC[1]/gradCnorm[1]
                    [Ni,Nj] = utils.CalcFaceNormal(NodeCoords,np.array([Newi,Newj]))
                    
                    Ai = np.linalg.norm(np.cross(NodeCoords[Newi[1]]-NodeCoords[Newi[0]],NodeCoords[Newi[2]]-NodeCoords[Newi[0]]))/2
                    Aj = np.linalg.norm(np.cross(NodeCoords[Newj[1]]-NodeCoords[Newj[0]],NodeCoords[Newj[2]]-NodeCoords[Newj[0]]))/2
                    Ei = Ai*(1-np.dot(Ni,mi))
                    Ej = Aj*(1-np.dot(Nj,mj))
                    # Ei = np.arccos(np.dot(Ni,mi))
                    # Ej = np.arccos(np.dot(Nj,mj))
                    OldError = NormalError[i] + NormalError[j]
                    NewError = Ei + Ej
                    if NewError < OldError:
                        NormalError[i] = Ei; NormalError[j] = Ej
                        NewConn[i] = Newi; NewConn[j] = Newj
                        
                        ENi = []; ENj = []
                        Si = set(Newi); Sj = set(Newj)
                        for k in np.unique(ElemNeighbors[i] + ElemNeighbors[j]):
                            if i in ElemNeighbors[k]: ElemNeighbors[k].remove(i)
                            if j in ElemNeighbors[k]: ElemNeighbors[k].remove(j)
                            if len(Si.intersection(NewConn[k])) == 2:
                                ENi.append(k)
                                ElemNeighbors[k].append(i)
                            if len(Sj.intersection(NewConn[k])) == 2:
                                ENj.append(k)
                                ElemNeighbors[k].append(j)

                        ElemNeighbors[i] = ENi; ElemNeighbors[j] = ENj
                        restart = True
                        break
                    else:
                        restart = False
        return NewConn, ElemNeighbors
    def Error(NodeCoords, ElemConn, ElemNormals, Area, Centroids):
        fP = sdf(NodeCoords[:,0],NodeCoords[:,1],NodeCoords[:,2])
        gradP = gradF(NodeCoords)
        gradPnorm = np.linalg.norm(gradP,axis=1)
        gradC = gradF(Centroids)
        gradCnorm = np.linalg.norm(gradC,axis=1)
        m = np.divide(gradC,np.reshape(gradCnorm,(len(gradC),1)))

        area = np.append(Area,0)
        R = utils.PadRagged(ElemConn,fillval=-1)
        A = np.sum(area[R],axis=1)

        VertexError = 1/(3*sum(Area)) * sum((fP**2/gradPnorm**2)*A)
        NormalError = 1/(sum(Area)) * sum(Area*(1-np.sum(ElemNormals*m,axis=1)))

        return VertexError, NormalError

    # edges = converter.surf2edges(NodeCoords,NodeConn)
    # if len(edges) > 0: warnings.warn('Input mesh should be closed and contain no exposed edges.')
    k = 0
    # mesh.mesh(NodeCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1

    if Subdivision: NodeCoords, NodeConn = AdaptiveSubdivision(sdf, NodeCoords, NodeConn,threshold=1e-3)
    NodeCoords,NodeConn,_ = utils.DeleteDuplicateNodes(NodeCoords,NodeConn)
    NewCoords = np.array(NodeCoords)
    # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1

    NodeNeighbors = utils.getNodeNeighbors(NewCoords, NodeConn) 
    ElemConn = utils.getElemConnectivity(NewCoords, NodeConn)
    ElemNeighbors = utils.getElemNeighbors(NodeCoords,NodeConn,mode='edge')
    # NodeConn, ElemNeighbors = improvement.ValenceImprovementFlips(NodeCoords,NodeConn,NodeNeighbors,ElemNeighbors)
    # vE = [];    nE = []   
    ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
    NodeNormals = np.array(utils.Face2NodeNormal(NewCoords, NodeConn, ElemConn, ElemNormals))
    
    tfs = np.linspace(1,0,ZRIter+1)
    for i in range(ZRIter):
        tf = tfs[i]
        Points = NewCoords[np.array(NodeConn)]
        Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2   
        Centroids = utils.Centroids(NewCoords, NodeConn)
        # v,n = Error(NewCoords, ElemConn, ElemNormals, Area, Centroids); vE.append(v); nE.append(n)
        NewCoords[FreeNodes] += ZRFlow(NewCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids, tf=tf)[FreeNodes]
        ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
        NodeNormals = np.array(utils.Face2NodeNormal(NewCoords, NodeConn, ElemConn, ElemNormals))
        # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1
    for i in range(NZRIter):
        Points = NewCoords[np.array(NodeConn)]
        Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2   
        Centroids = utils.Centroids(NewCoords, NodeConn)
        # v,n = Error(NewCoords, ElemConn, ElemNormals, Area, Centroids); vE.append(v); nE.append(n)
        NewCoords[FreeNodes] += NZRFlow(NewCoords, NodeConn, NodeNormals, NodeNeighbors, ElemConn, Area, Centroids)[FreeNodes]
        ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
        NodeNormals = np.array(utils.Face2NodeNormal(NewCoords, NodeConn, ElemConn, ElemNormals))
        # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1
    if NZIter > 0:
        if Subdivision: NewCoords, NodeConn = AdaptiveSubdivision(sdf, NewCoords.tolist(), NodeConn, threshold=1e-4)
        NewCoords = np.array(NewCoords)
        NodeNeighbors = utils.getNodeNeighbors(NewCoords, NodeConn)    
        ElemConn = utils.getElemConnectivity(NewCoords, NodeConn)    
        ElemNeighbors = utils.getElemNeighbors(NewCoords,NodeConn,mode='edge')
        ElemNormals = utils.CalcFaceNormal(NewCoords, NodeConn)
        tfs = np.linspace(1,0,NZIter+1)
    for i in range(NZIter):
        tf = tfs[i]
        Points = NewCoords[np.array(NodeConn)]
        Area = np.linalg.norm(np.cross(Points[:,1]-Points[:,0],Points[:,2]-Points[:,0]),axis=1)/2   
        Centroids = utils.Centroids(NewCoords, NodeConn)

        # v,n = Error(NewCoords, ElemConn, ElemNormals, Area, Centroids)
        # vE.append(v); nE.append(n)

        NewCoords[FreeNodes] += NZFlow(NewCoords, NodeConn, [], NodeNeighbors, ElemConn, Area, Centroids,tf=tf)[FreeNodes]
        
        NewElemNormals = np.array(utils.CalcFaceNormal(NewCoords, NodeConn))

        ### Check for near-intersections ###
        # Angles:
        Points = NewCoords[np.array(NodeConn)]
        v01 = Points[:,1]-Points[:,0]; l01 = np.linalg.norm(v01,axis=1)
        v12 = Points[:,2]-Points[:,1]; l12 = np.linalg.norm(v12,axis=1)
        v20 = Points[:,0]-Points[:,2]; l20 = np.linalg.norm(v20,axis=1)
        alpha = np.arccos(np.sum(v01*-v20,axis=1)/(l01*l20))
        beta = np.arccos(np.sum(v12*-v01,axis=1)/(l12*l01))
        gamma = np.arccos(np.sum(v20*-v12,axis=1)/(l20*l12))
        angles = np.vstack([alpha,beta,gamma]).T
        # Dihedrals:
        dihedrals = quality.SurfDihedralAngles(NewElemNormals,ElemNeighbors)
        # Normal Flipping:
        NormDot = np.sum(NewElemNormals * ElemNormals,axis=1)

        Risk = np.any(angles<5*np.pi/180,axis=1) | np.any(dihedrals > 175*np.pi/180,axis=1) | (NormDot < 0)
        Intersected = []
        if i >= NZIter-5:
            # NodeConn, ElemNeighbors = Flip(NewCoords,NodeConn,ElemNormals,ElemNeighbors,Area,Centroids)
            IntersectionPairs = rays.SurfSelfIntersection(NewCoords,NodeConn)
            Intersected = np.unique(IntersectionPairs).tolist()
            
        if np.any(Risk) or len(Intersected):
            # print('possible intersection')
            ArrayConn = np.array(NodeConn)
            AtRiskElems = np.where(Risk)[0].tolist() + Intersected
            NeighborhoodElems = np.unique([e for i in (ArrayConn[AtRiskElems]).flatten() for e in ElemConn[i]])
            PatchConn = ArrayConn[NeighborhoodElems] 
            BoundaryEdges = converter.surf2edges(NewCoords,PatchConn)
            FixedNodes = set([n for edge in BoundaryEdges for n in edge])
            NewCoords = np.array(improvement.LocalLaplacianSmoothing(NewCoords,PatchConn,2,FixedNodes=FixedNodes))

            # NodeConn = improvement.AngleReductionFlips(NewCoords,NodeConn,NodeNeighbors)
            NodeNeighbors,ElemConn = utils.getNodeNeighbors(NewCoords, NodeConn)    
            ElemNeighbors = utils.getElemNeighbors(NewCoords,NodeConn,mode='edge')
            ElemNormals = np.array(utils.CalcFaceNormal(NewCoords, NodeConn))
        else:
            ElemNormals = NewElemNormals
        # mesh.mesh(NewCoords,NodeConn).Mesh2Meshio().write(str(k)+'.vtu');k+=1
    # px.line(y=vE).show()
    # px.line(y=nE).show()
    return NewCoords.tolist(), NodeConn

def HexCore(sdf,bounds,h,nBL=3,nPeel=1,verbose=True,TetgenSwitches=['-pq1.1/25','-Y','-o/150'],
           MCInterp='midpoint',SurfOpt=True,TetOpt=True,GLS=False,SkewThreshold=None,
           BLStiffnessFactor=1,BLMaxThickness=None,TetSliverThreshold=None):
    """
    HexCore - Generate a mesh with a core of regular hexahedral elements, 
    wrapped in a layer of pyramids and tetrahedrons, and (optionally) boundary 
    layer wedge elements for a given signed distance function
    (or similar isosurface implicit function)

    Parameters
    ----------
    sdf : function
        Signed distance (or similar) function that takes x, y, z coordinates
        and returns a value
    bounds : list
        Bounds of the domain of interest in the form 
        [xmin,xmax,ymin,ymax,zmin,zmax].
    h : Element Size
        Element size of the initial voxel field
    nBL : int, optional
        Number of boundary layers. Specify 0 for no boundary layers. 
        The default is 3.
    nPeel : int, optional
        Number of layers of voxel elements to peel off when generating
        the hexahedral core. The default is 1.
    verbose : bool, optional
        Specifies whether or not to print status updates during runtime. The default is True.
    TetgenSwitches : list, optional
        List of TetGen switches to specify tetrahedral meshing options. 
        See https://www.wias-berlin.de/software/tetgen/switches.html 
        The default is ['-pq', '-Y'].
    SkewThreshold : float, optional
        Skewness threshold above which triangles will be considered slivers
        and collapsed. The default is 0.9.

    Returns
    -------
    hexcore : mesh.mesh
        mesh object containing the generated mesh.
    """
    
    VoxelTime = MarchingTime = SurfOptTime = TetTime = BLTime = AssemblyTime = 0

    if (BLMaxThickness) and BLMaxThickness >= (nPeel-1/2)*h:
        nPeel = int(np.ceil(BLMaxThickness/h+2/3))
        warnings.warn('Increasing number of peel layers to {:d} accomdate BLThickness'.format(nPeel))

    if verbose: print('Generating Initial Voxel Mesh...')
    tic = time.time()
    NodeCoords, NodeConn, NodeVals = VoxelMesh(sdf,[bounds[0]-h,bounds[1]+3*h],[bounds[2]-h,bounds[3]+3*h],[bounds[4]-h,bounds[5]+3*h],h,mode='liberal')
    VoxelTime += time.time()-tic
    
    if verbose: print('Performing Marching Cubes Surface Construction...')
    tic = time.time()
    TriNodeCoords,TriNodeConn = contour.MarchingCubes(NodeCoords,NodeConn,NodeVals,interpolation=MCInterp,method='33',threshold=0)
    MarchingTime += time.time()-tic
    # if verbose: print('Surface Optimization...')
    # tic = time.time()
    # ImprovedCoords,ImprovedConn = SurfFlowOptimization(sdf,TriNodeCoords,TriNodeConn,h)
    # SurfOptTime = time.time()-tic

    if verbose: print('Surface Optimization...')
    tic = time.time()  
    if SurfOpt:
        TriNodeCoords,TriNodeConn = SurfFlowOptimization(sdf,TriNodeCoords,TriNodeConn)

    if GLS:
        tic = time.time()
        TriNodeCoords = improvement.GlobalLaplacianSmoothing(TriNodeCoords,TriNodeConn,[],FeatureWeight=1,BaryWeight=1/3)
    if not (SurfOpt or GLS):
        TriNodeCoords = TriNodeCoords
        TriNodeConn = TriNodeConn
    
    if SkewThreshold:
        TriNodeCoords,TriNodeConn = improvement.CollapseSlivers(TriNodeCoords,TriNodeConn,skewThreshold=SkewThreshold,verbose=verbose)
    SurfOptTime += time.time()-tic
    # mesh.mesh(TriNodeCoords,TriNodeConn).Mesh2Meshio().write('opt.vtu')

    if verbose: print('Peeling Voxels...')
    tic = time.time()
    cutvoxels = set([i for i,elem in enumerate(NodeConn) if 0 < sum([NodeVals[n] > 0 for n in elem]) < 8])  # Voxels cut by marching cubes
    PeeledCoords, PeeledConn, PeelCoords, PeelConn = utils.PeelHex(NodeCoords,[elem for i,elem in enumerate(NodeConn) if i not in cutvoxels],nLayers=nPeel)
    PeelTime = time.time() - tic

    if verbose: print('Generating Pyramids...')
    tic = time.time()
    if len(PeeledConn) > 0:
        PyramidCoords, PyramidConn = utils.makePyramidLayer(PeeledCoords,PeeledConn)
        PyrSurf = converter.solid2surface(PyramidCoords,PyramidConn)
        TriSurf = [elem for elem in PyrSurf if len(elem) == 3]
        TriCoords,TriSurf,_ = utils.DeleteDuplicateNodes(PyramidCoords,TriSurf)
    else:
        TriCoords = []
        TriSurf = []
    PyramidTime = time.time()-tic
    
    if verbose: print('Tet Meshing with Tetgen...')    
    tic = time.time()
    BoundaryCoords = TriCoords + TriNodeCoords
    BoundaryConn = TriSurf + (np.array(TriNodeConn)+len(TriCoords)).tolist()

    FixedNodes = set(range(len(TriCoords)))
    BoundaryCoords = improvement.ResolveSurfSelfIntersections(BoundaryCoords,BoundaryConn,FixedNodes=FixedNodes)

    if len(PeeledConn) > 0:
        holes = utils.Centroids(PeeledCoords,PeeledConn)
    else:
        holes = []
    
    # if '-Y' not in TetgenSwitches:
    #     TetgenSwitches.append('-Y') 
    #     warnings.warn('-Y switch is required to maintain the surface mesh input to tetgen. Proceeding with -Y')
    TetCoords, TetConn = TetGen.tetgen(BoundaryCoords,BoundaryConn,verbose=verbose==2,BoundingBox=True,holes=holes,switches=TetgenSwitches)
    TetTime = time.time()-tic
    ### Tet Optimization
    if TetOpt:
        if verbose: print('Optimizing Tetrahedral Mesh...')
        tic = time.time()
        # Tets = [elem for elem in hexcore.NodeConn if len(elem)==4]
        # skew = quality.Skewness(hexcore.NodeCoords,Tets)
        # BadElems = set(np.where(skew>0.9)[0])
        # ElemNeighbors = utils.getElemNeighbors(hexcore.NodeCoords,Tets)
        # BadElems.update([e for i in BadElems for e in ElemNeighbors[i]])
        # BadNodes = set([n for i in BadElems for n in Tets[i]])

        # SurfConn = converter.solid2surface(hexcore.NodeCoords,Tets)
        # SurfNodes = set([n for elem in SurfConn for n in elem])

        # FreeNodes = BadNodes.difference(SurfNodes)
        SurfConn = converter.solid2surface(TetCoords,TetConn)
        SurfNodes = set([n for elem in SurfConn for n in elem])
        FreeNodes = set(range(len(TetCoords))).difference(SurfNodes)
        TetCoords = improvement.TetOpt(TetCoords,TetConn,FreeNodes=FreeNodes,objective='eta',method='BFGS',iterate=3)
        TetTime += time.time()-tic

    if verbose: print('Assembling Mesh...')
    tic = time.time()
    hexcore = mesh.mesh()
    hexcore.verbose = False
    tets = mesh.mesh(TetCoords,TetConn)
    if len(PeeledConn) > 0:
        core = mesh.mesh(PeeledCoords,PeeledConn)
        pyramids = mesh.mesh(PyramidCoords,PyramidConn)
        hexcore.merge([core,pyramids,tets])
    else:
        hexcore = tets
    hexcore.NodeCoords,hexcore.NodeConn,_ = converter.removeNodes(hexcore.NodeCoords,hexcore.NodeConn)
    hexcore.initialize(cleanup=False)
    AssemblyTime = time.time() - tic
    
    NonTetNodes = set(np.unique([n for i,elem in enumerate(hexcore.NodeConn) if len(elem)!=4 for n in elem]))

    # if verbose: print('Creating Boundary Layers...')
    # tic = time.time()
    # if nBL > 0:
    #     # Create boundary layer elements, only allowing tet elements to be moved to make room
    #     hexcore.reset('SurfConn')
    #     NonTetNodes = set(np.unique([n for i,elem in enumerate(hexcore.NodeConn) if len(elem)!=4 for n in elem]))
    #     hexcore.CreateBoundaryLayer(nBL,FixedNodes=NonTetNodes,StiffnessFactor=BLStiffnessFactor,Thickness=BLMaxThickness)
    #     hexcore.verbose = verbose
        
    #     BLTime += time.time()-tic
    
    
    if TetSliverThreshold:
        if verbose: print('Peeling Surface Tet Slivers...')
        tic = time.time()
        hexcore.NodeConn = improvement.SliverPeel(*hexcore,skewThreshold=TetSliverThreshold)
        TetTime += time.time()-tic
    
    ### 

    tic = time.time()
    hexcore.initialize()
    hexcore.validate()
    AssemblyTime += time.time()-tic
    
    if verbose:
        print('------------------------------------------')
        print('Voxel Mesh Generated in %.4fs' % VoxelTime)
        print('Marching Cubes Performed in %.4fs' % MarchingTime)
        print('Surface Optimization Completed in %.4fs' % SurfOptTime)
        print('Voxels Peeled in %.4fs' % PeelTime)
        print('Pyramids Generated in %.4fs' % PyramidTime)
        print('Tets Generated in %.4fs' % TetTime)
        if nBL > 0: print('Boundary Layers Created in %.4fs' % BLTime)
        print('Meshes Assembled in %.4fs' % AssemblyTime)
        print('Total Meshing Time: %.4f min' % ((VoxelTime + MarchingTime + SurfOptTime  + BLTime + PeelTime + PyramidTime + TetTime + AssemblyTime)/60))
        print('------------------------------------------')
    return hexcore  

def TetMesh(sdf,bounds,h,verbose=True,TetgenSwitches=['-pq1.1/25','-Y','-o/150'],
            MCInterp='midpoint',SurfOpt=True,GLS=False,SkewThreshold=None):
    """
    TetMesh - Generate a Tetrahedral Mesh for a given signed distance function
    (or similar isosurface implicit function)

    Parameters
    ----------
    sdf : function
        Signed distance (or similar) function that takes x, y, z coordinates
        and returns a value
    bounds : list
        Bounds of the domain of interest in the form 
        [xmin,xmax,ymin,ymax,zmin,zmax].
    h : Element Size
        Element size of the initial voxel field
    verbose : bool, optional
        Specifies whether or not to print status updates during runtime. The default is True.
    TetgenSwitches : list, optional
        List of TetGen switches to specify tetrahedral meshing options. 
        See https://www.wias-berlin.de/software/tetgen/switches.html 
        The default is ['-pq', '-Y'].
    MCInterp : Interpolaton method for marching cubes. 
        Options are: 'midpoint', 'linear'.
        The default is 'midpoint'.
    SurfOpt : Perform Surface Flow Optimization. The default is True.
    GLS : Perform Global Laplacian Smoothing on the surface mesh. The default is False.
    skewThreshold : float, optional
        Skewness threshold above which triangles will be considered slivers
        and collapsed. The default is None.

    Returns
    -------
    tetmesh : mesh.mesh
        mesh object containing the generated mesh.
    """
    
    VoxelTime = MarchingTime = SurfOptTime = SmoothingTime = SliverTime = TetTime = AssemblyTime = 0
    
    if verbose: print('Generating Initial Voxel Mesh...')
    tic = time.time()
    NodeCoords, NodeConn, NodeVals = VoxelMesh(sdf,[bounds[0]-h,bounds[1]+3*h],[bounds[2]-h,bounds[3]+3*h],[bounds[4]-h,bounds[5]+3*h],h,mode='liberal')
    VoxelTime += time.time()-tic
    
    if verbose: print('Performing Marching Cubes Surface Construction...')
    tic = time.time()
    TriNodeCoords,TriNodeConn = contour.MarchingCubes(NodeCoords,NodeConn,NodeVals,interpolation=MCInterp,method='33',threshold=0)
    MarchingTime += time.time()-tic
    if SurfOpt:
        if verbose: print('Surface Optimization...')
        tic = time.time()        
        TriNodeCoords,TriNodeConn = SurfFlowOptimization(sdf,TriNodeCoords,TriNodeConn)
        SurfOptTime += time.time()-tic
   
    if GLS:
        if verbose: print('Smoothing Mesh...')
        tic = time.time()
        TriNodeCoords = improvement.GlobalLaplacianSmoothing(TriNodeCoords,TriNodeConn,[],FeatureWeight=1,BaryWeight=1/3)
        SurfOptTime += time.time()-tic

    if SkewThreshold:
        if verbose: print('Collapsing Surface Mesh Slivers...')
        tic = time.time()
        TriNodeCoords,TriNodeConn = improvement.CollapseSlivers(TriNodeCoords,TriNodeConn,skewThreshold=SkewThreshold,verbose=verbose)
        SurfOptTime += time.time()-tic
    # mesh.mesh(TriNodeCoords,TriNodeConn).Mesh2Meshio().write('opt.vtu')
    if verbose: print('Tet Meshing with TetGen...')    
    tic = time.time()
    v = np.array(NodeVals)
    hconn = [elem for elem in NodeConn if all(v[elem] > 0)]
    holes = utils.Centroids(NodeCoords,hconn)
    
    if '-Y' not in TetgenSwitches:
        TetgenSwitches.append('-Y') 
        warnings.warn('-Y switch is required to maintain the surface mesh input to tetgen. Proceeding with -Y')
    TetCoords, TetConn = TetGen.tetgen(TriNodeCoords,TriNodeConn,holes=holes,verbose=verbose==2,BoundingBox=True,switches=TetgenSwitches)
    TetTime = time.time()-tic
    
    if verbose: print('Assembling Mesh...')
    tic = time.time()
    tetmesh = mesh.mesh(TetCoords,TetConn)
    tetmesh.initialize()
    tetmesh.validate()
    AssemblyTime = time.time() - tic
    
    if verbose:
        print('------------------------------------------')
        print('Voxel Mesh Generated in %.4fs' % VoxelTime)
        print('Marching Cubes Performed in %.4fs' % MarchingTime)
        print('Surface Optimization Completed in %.4fs' % SurfOptTime)
        print('Tets Generated in %.4fs' % TetTime)
        print('Meshes Assembled in %.4fs' % AssemblyTime)
        print('Total Meshing Time: %.4f min' % ((VoxelTime + MarchingTime + SurfOptTime + SmoothingTime + SliverTime + TetTime + AssemblyTime)/60))
        print('------------------------------------------')
    return tetmesh 
